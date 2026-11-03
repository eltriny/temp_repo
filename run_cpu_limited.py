#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
run_cpu_limited.py

train.py / predict.py 같은 명령을 CPU 사용률 상한(hard cap) 안에서 실행합니다.
상한은 자식 프로세스(DataLoader worker 포함)까지 합산하여 적용됩니다.

- Linux  : systemd-run --user --scope -p CPUQuota=...  (cgroup v2 CPU 쿼터)
- Windows: Job Object CPU rate control (JOB_OBJECT_CPU_RATE_CONTROL_HARD_CAP)

상한만 걸고 스레드 수를 그대로 두면 스레드들이 쿼터에 걸려 번갈아 멈추면서 처리량이 크게 떨어집니다.
그래서 OMP_NUM_THREADS / MKL_NUM_THREADS를 상한에 맞는 값으로 함께 지정합니다.

사용 예:
python run_cpu_limited.py --cpu_percent 50 -- python train.py --data_dir dataset --device cpu
"""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List


def build_env(num_threads: int) -> Dict[str, str]:
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = str(num_threads)
    env["MKL_NUM_THREADS"] = str(num_threads)
    return env


def check_user_cpu_delegation() -> None:
    """
    systemd 사용자 세션에 cpu 컨트롤러가 위임되지 않은 배포판에서는
    CPUQuota가 오류 없이 무시되므로 미리 확인합니다.
    """
    uid = os.getuid()
    controllers = Path(
        f"/sys/fs/cgroup/user.slice/user-{uid}.slice/user@{uid}.service/cgroup.controllers"
    )

    # cgroup v1 등 경로 구조가 다른 환경에서는 확인을 생략
    if controllers.is_file() and "cpu" not in controllers.read_text().split():
        raise RuntimeError(
            "systemd 사용자 세션에 cpu 컨트롤러가 위임되어 있지 않아 CPUQuota가 적용되지 않습니다. "
            "/etc/systemd/system/user@.service.d/delegate.conf 에 "
            "'[Service]' / 'Delegate=cpu cpuset io memory pids' 를 추가한 뒤 재로그인하세요."
        )


def run_linux(command: List[str], cpu_percent: int, env: Dict[str, str]) -> None:
    if shutil.which("systemd-run") is None:
        raise RuntimeError(
            "systemd-run을 찾을 수 없습니다. systemd가 없는 환경(컨테이너 등)에서는 "
            "docker run --cpus=<CPU 개수> 처럼 컨테이너 단위로 상한을 지정하세요."
        )

    check_user_cpu_delegation()

    # CPUQuota는 CPU 1개 = 100% 기준이므로 전체 CPU 대비 비율로 환산 (16 CPU의 50% -> 800%)
    quota = cpu_percent * (os.cpu_count() or 1)
    systemd_cmd = [
        "systemd-run", "--user", "--scope", "--quiet",
        "-p", f"CPUQuota={quota}%",
        *command,
    ]

    # --scope 모드에서는 systemd-run이 현재 터미널/환경 그대로 명령을 직접 실행
    os.execvpe(systemd_cmd[0], systemd_cmd, env)


def run_windows(command: List[str], cpu_percent: int, env: Dict[str, str]) -> int:
    import ctypes
    from ctypes import wintypes

    class JOBOBJECT_CPU_RATE_CONTROL_INFORMATION(ctypes.Structure):
        # 원래는 ControlFlags 뒤에 CpuRate/Weight/MinRate+MaxRate union이 오지만 모두 4바이트
        _fields_ = [
            ("ControlFlags", wintypes.DWORD),
            ("CpuRate", wintypes.DWORD),
        ]

    JOB_OBJECT_CPU_RATE_CONTROL_ENABLE = 0x1
    JOB_OBJECT_CPU_RATE_CONTROL_HARD_CAP = 0x4
    JobObjectCpuRateControlInformation = 15  # JOBOBJECTINFOCLASS
    ERROR_NOT_SUPPORTED = 50

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CreateJobObjectW.argtypes = [wintypes.LPVOID, wintypes.LPCWSTR]
    kernel32.CreateJobObjectW.restype = wintypes.HANDLE
    kernel32.SetInformationJobObject.argtypes = [
        wintypes.HANDLE, ctypes.c_int, wintypes.LPVOID, wintypes.DWORD,
    ]
    kernel32.SetInformationJobObject.restype = wintypes.BOOL
    kernel32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
    kernel32.AssignProcessToJobObject.restype = wintypes.BOOL
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE

    job = kernel32.CreateJobObjectW(None, None)
    if not job:
        raise ctypes.WinError(ctypes.get_last_error())

    # CpuRate = 10,000 사이클 중 사용 가능한 사이클 수, 시스템 전체 기준 (50% -> 5000)
    info = JOBOBJECT_CPU_RATE_CONTROL_INFORMATION(
        JOB_OBJECT_CPU_RATE_CONTROL_ENABLE | JOB_OBJECT_CPU_RATE_CONTROL_HARD_CAP,
        cpu_percent * 100,
    )

    if not kernel32.SetInformationJobObject(
        job, JobObjectCpuRateControlInformation, ctypes.byref(info), ctypes.sizeof(info)
    ):
        err = ctypes.get_last_error()
        if err == ERROR_NOT_SUPPORTED:
            raise RuntimeError(
                "CPU rate control을 지원하지 않는 환경입니다. "
                "원격 데스크톱 서비스(RDS)의 Dynamic Fair Share Scheduling이 켜져 있으면 사용할 수 없습니다."
            )
        raise ctypes.WinError(err)

    # 자기 자신을 job에 넣으면 이후 생성되는 자식 프로세스(train.py, DataLoader worker)도 자동으로 포함
    if not kernel32.AssignProcessToJobObject(job, kernel32.GetCurrentProcess()):
        raise ctypes.WinError(ctypes.get_last_error())

    return subprocess.call(command, env=env)


def main():
    parser = argparse.ArgumentParser(description="Run a command under a hard CPU usage cap")
    parser.add_argument("--cpu_percent", type=int, default=50,
                        help="전체 CPU 대비 사용률 상한 (1~100)")
    parser.add_argument("--num_threads", type=int, default=0,
                        help="PyTorch 연산 스레드 수 (0=상한에 맞춰 자동). "
                             "--num_workers를 쓰면 worker도 같은 상한을 나눠 쓰므로 필요 시 줄여서 비교")
    parser.add_argument("command", nargs=argparse.REMAINDER,
                        help="-- 뒤에 실행할 명령")

    args = parser.parse_args()

    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("실행할 명령이 없습니다. 예: -- python train.py --data_dir dataset")
    if not 1 <= args.cpu_percent <= 100:
        parser.error("--cpu_percent는 1~100 사이여야 합니다.")

    cpu_count = os.cpu_count() or 1
    num_threads = args.num_threads or max(1, cpu_count * args.cpu_percent // 100)
    env = build_env(num_threads)

    print(f"[INFO] cpu limit   = {args.cpu_percent}% of {cpu_count} logical CPUs")
    print(f"[INFO] cpu threads = {num_threads} (OMP_NUM_THREADS / MKL_NUM_THREADS)")
    sys.stdout.flush()

    if sys.platform.startswith("linux"):
        run_linux(command, args.cpu_percent, env)
    elif sys.platform == "win32":
        sys.exit(run_windows(command, args.cpu_percent, env))
    else:
        raise RuntimeError(f"지원하지 않는 OS입니다: {sys.platform}")


if __name__ == "__main__":
    main()
