"""
배치 스케줄러 - 디렉토리 스캔 및 순차적 비디오 분석 스케줄링

5분 주기로 지정 디렉토리를 스캔하여 새 비디오 파일을 발견하고,
한 번에 한 파일씩 순차적으로 VideoAnalyzerApp을 실행합니다.

사용 예시:
    scheduler = BatchScheduler(
        watch_dir=Path("./videos"),
        output_dir=Path("./data"),
        template_name="기본 템플릿",
    )
    scheduler.run()   # Ctrl+C로 종료

Phase 1-E 안정화 항목:
    - Circuit breaker: 연속 사이클/파일 실패 시 명시적 종료 (systemd 재시작 유도).
    - Watchdog: 파일 처리에 비디오 길이 기반 타임아웃 적용.
    - HeartbeatWriter: output_dir/heartbeat.json 에 주기적 상태 기록.
    - 디스크 공간 체크: 사이클 시작 시 `min_free_disk_mb` 미만이면 warn/skip/abort.
    - H-14 수정: 세션 역조회 시 `get_latest_session_by_source_path` O(1) 경로 사용.

TODO(Phase 2): CaptureManager 핫패스 디스크 체크
    현재 디스크 공간 체크는 BatchScheduler 사이클 수준에서만 수행됩니다.
    장시간 단일 파일 처리 중 디스크가 채워지는 경우를 감지하려면
    CaptureManager.save_capture() 등 핫패스에서 주기적 (예: 매 100회 save)
    psutil.disk_usage() 체크를 추가해야 합니다. Phase 2에서 다룹니다.
"""

from __future__ import annotations

import logging
import os
import signal
import sys
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

from .app import VideoAnalyzerApp
from .config import StorageConfig
from .observability.heartbeat import HeartbeatWriter
from .observability.lifecycle import ShutdownCoordinator
from .storage.database import (
    DatabaseManager,
    VideoFile,
    VideoFileCreate,
    VideoFileStatus,
)

logger = logging.getLogger(__name__)

# 지원하는 비디오 파일 확장자
_VIDEO_EXTENSIONS: frozenset[str] = frozenset(
    {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".m4v", ".ts"}
)


# ========================================
# 배치 안정화 설정 dataclass
# ========================================


@dataclass(frozen=True)
class BatchBreakerConfig:
    """배치 스케줄러 실패 회복 정책.

    Attributes:
        max_consecutive_cycle_failures: 연속 사이클 실패 임계값. 초과 시 sys.exit(2).
        max_consecutive_file_failures: 연속 파일 처리 실패 임계값. 초과 시 sys.exit(3).
        cooldown_seconds: Breaker trip 이후 재시도까지 권장 대기 시간 (초).
            현재 설계는 trip 시 즉시 프로세스 종료이므로 실질적으로는
            외부 supervisor(systemd 등)의 재시작 주기와 함께 의미를 가집니다.
            추후 인프로세스 복구 기능 추가 시 사용 예정.
    """

    max_consecutive_cycle_failures: int = 3
    max_consecutive_file_failures: int = 5
    cooldown_seconds: int = 600  # 10분


@dataclass(frozen=True)
class BatchWatchdogConfig:
    """파일 처리 워치독 타임아웃 정책.

    Attributes:
        min_file_timeout_sec: 파일당 최소 처리 허용 시간(초).
        file_timeout_multiplier: 비디오 길이 대비 허용 처리 시간 배수.
            실제 타임아웃 = max(min_file_timeout_sec, duration * multiplier).
    """

    min_file_timeout_sec: int = 300
    file_timeout_multiplier: float = 5.0


def _default_timeout_exit(code: int) -> None:
    """B5: 타임아웃 발생 시 프로세스 즉시 종료.

    ``ThreadPoolExecutor.__exit__`` 는 ``shutdown(wait=True)`` 를 호출하므로
    ``with`` 블록 내부에서 ``sys.exit`` 를 던지면 멈춘 워커 대기로 hang 된다.
    ``os._exit`` 는 cleanup 을 우회하여 프로세스를 즉시 종료하고 외부 supervisor
    (systemd 등) 의 재시작을 유도한다.
    """
    os._exit(code)


class BatchScheduler:
    """배치 비디오 분석 스케줄러

    지정 디렉토리를 주기적으로 스캔하여 신규 비디오 파일을 발견하고,
    video_files 테이블에 등록한 뒤 순차적으로 분석합니다.

    Attributes:
        watch_dir: 스캔할 비디오 디렉토리
        output_dir: 분석 결과 저장 디렉토리
        storage_config: 데이터베이스 연결 설정
        interval_seconds: 배치 사이클 간격 (기본 300초 = 5분)
    """

    def __init__(
        self,
        watch_dir: Path,
        output_dir: Path,
        storage_config: Optional[StorageConfig] = None,
        template_id: Optional[int] = None,
        template_name: Optional[str] = None,
        interval_seconds: int = 300,
        use_gpu: bool = False,
        frame_interval: float = 1.0,
        ssim_threshold: float = 0.95,
        confidence_threshold: float = 0.7,
        max_workers: Optional[int] = None,
        *,
        breaker_config: Optional[BatchBreakerConfig] = None,
        watchdog_config: Optional[BatchWatchdogConfig] = None,
        shutdown_coordinator: Optional[ShutdownCoordinator] = None,
    ) -> None:
        """BatchScheduler 초기화.

        Args:
            watch_dir: 스캔할 비디오 디렉토리.
            output_dir: 분석 결과 저장 디렉토리.
            storage_config: DB 연결 설정.
            template_id: 사용할 템플릿 ID.
            template_name: 사용할 템플릿 이름.
            interval_seconds: 사이클 간격(초).
            use_gpu: OCR GPU 사용 여부.
            frame_interval: 프레임 샘플링 간격(초).
            ssim_threshold: SSIM 임계값.
            confidence_threshold: 탐지 신뢰도 임계값.
            max_workers: 병렬 처리 워커 수.
            breaker_config: 실패 회복 circuit breaker 정책
                (기본 ``BatchBreakerConfig()``).
            watchdog_config: 파일 처리 워치독 타임아웃 정책
                (기본 ``BatchWatchdogConfig()``).
        """
        if not template_id and not template_name:
            raise ValueError(
                "배치 모드에는 --template 또는 --template-id가 필요합니다."
            )

        self.watch_dir = watch_dir.resolve()
        self.output_dir = output_dir
        self.storage_config = storage_config or StorageConfig(output_dir=output_dir)
        self.template_id = template_id
        self.template_name = template_name
        self.interval_seconds = interval_seconds
        self.use_gpu = use_gpu
        self.frame_interval = frame_interval
        self.ssim_threshold = ssim_threshold
        self.confidence_threshold = confidence_threshold
        self.max_workers = max_workers
        self._shutdown_flag: bool = False

        # Phase 1-E: 안정화 설정
        self._breaker_config = breaker_config or BatchBreakerConfig()
        self._watchdog_config = watchdog_config or BatchWatchdogConfig()

        # B1: 외부 ShutdownCoordinator 통합 (main.py 가 주입한 경우 중복 시그널 등록 방지)
        self._shutdown_coordinator: Optional[ShutdownCoordinator] = shutdown_coordinator

        # Circuit breaker 내부 상태
        self._cycle_failure_count: int = 0
        self._file_failure_count: int = 0
        self._last_failure_time: Optional[datetime] = None

        # 현재 사이클 번호 (heartbeat 공유)
        self._cycle_number: int = 0

        # 출력 디렉토리 보장
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Heartbeat writer (start/stop은 run() 에서)
        self._heartbeat = HeartbeatWriter(
            path=self.output_dir / "heartbeat.json",
            interval=30.0,
        )

        logger.info(
            "BatchScheduler 초기화: watch_dir=%s, interval=%ds, "
            "breaker(cycle=%d, file=%d), watchdog(min=%ds, x%.1f)",
            self.watch_dir,
            self.interval_seconds,
            self._breaker_config.max_consecutive_cycle_failures,
            self._breaker_config.max_consecutive_file_failures,
            self._watchdog_config.min_file_timeout_sec,
            self._watchdog_config.file_timeout_multiplier,
        )

    # ========================================
    # 시그널 핸들링
    # ========================================

    def _timeout_exit(self, code: int) -> None:
        """B5: 타임아웃 종료 훅. 테스트에서 override 가능하도록 인스턴스 메서드로 분리.

        기본 구현은 ``os._exit`` 로 프로세스를 즉시 종료한다. 테스트는 이
        메서드를 mock 하여 종료 경로의 side effect (cancel_futures 전달 등) 만
        검증한다.
        """
        _default_timeout_exit(code)

    def _graceful_shutdown_hook(self) -> None:
        """ShutdownCoordinator 에 등록되는 shutdown hook.

        coordinator.trigger() -> cleanups LIFO 실행 경로에서 호출되어
        shutdown_flag 를 True 로 전환합니다.
        """
        if not self._shutdown_flag:
            logger.info("ShutdownCoordinator 경로를 통한 종료 요청 수신")
        self._shutdown_flag = True

    def _register_signal_handlers(self) -> None:
        """SIGINT(Ctrl+C) 및 SIGTERM 핸들러 등록

        B1: ShutdownCoordinator 가 주입된 경우 main.py 의 통합 핸들러를 신뢰하고
        여기서는 cleanup 만 등록합니다. 독립 실행(coordinator=None)에서는
        기존 단순 핸들러를 유지하여 후방 호환성을 보장합니다.
        """
        if self._shutdown_coordinator is not None:
            # main.py 가 이미 signal.signal 을 설치했으므로 덮어쓰지 않는다.
            # coordinator.trigger() 경로로 shutdown_flag 가 전환되도록 hook 만 등록.
            self._shutdown_coordinator.register(self._graceful_shutdown_hook)
            logger.debug(
                "ShutdownCoordinator 에 graceful shutdown hook 등록 완료"
            )
            return

        def _handler(signum: int, frame: object) -> None:
            logger.info(
                "종료 신호 수신 (signal=%d). 현재 파일 처리 완료 후 종료합니다.",
                signum,
            )
            self._shutdown_flag = True

        signal.signal(signal.SIGINT, _handler)
        signal.signal(signal.SIGTERM, _handler)

    # ========================================
    # 메인 루프
    # ========================================

    def run(self) -> None:
        """스케줄러 메인 루프

        Ctrl+C 또는 SIGTERM 수신까지 루프를 반복합니다.
        각 사이클: 디스크 체크 → 디렉토리 스캔 → 신규 등록 → PENDING 파일 순차 처리 → 대기

        Circuit breaker가 trip되면 sys.exit(2|3|4)로 즉시 종료합니다.
        """
        self._register_signal_handlers()
        self._recover_stale_processing()

        logger.info("배치 스케줄러 시작. 종료: Ctrl+C")

        # Heartbeat 시작
        self._heartbeat.update(status="starting", watch_dir=str(self.watch_dir))
        self._heartbeat.start()

        try:
            while not self._shutdown_flag:
                self._cycle_number += 1
                logger.info("=== 배치 사이클 #%d 시작 ===", self._cycle_number)
                self._heartbeat.update(
                    cycle=self._cycle_number,
                    status="running",
                    cycle_started_at=datetime.now().isoformat(timespec="seconds"),
                )

                try:
                    self._run_cycle()
                except SystemExit:
                    # breaker trip 등 명시적 종료는 그대로 전파
                    raise
                except Exception:
                    logger.exception(
                        "배치 사이클 #%d 예외 발생. 다음 사이클을 기다립니다.",
                        self._cycle_number,
                    )
                    self._cycle_failure_count += 1
                    self._last_failure_time = datetime.now()
                    self._heartbeat.update(
                        status="cycle_failed",
                        cycle_failure_count=self._cycle_failure_count,
                        last_failure_at=self._last_failure_time.isoformat(
                            timespec="seconds"
                        ),
                    )
                    if (
                        self._cycle_failure_count
                        >= self._breaker_config.max_consecutive_cycle_failures
                    ):
                        logger.critical(
                            "배치 사이클 %d회 연속 실패, breaker trip. 프로세스 종료.",
                            self._cycle_failure_count,
                        )
                        self._heartbeat.update(
                            status="failed", reason="cycle_breaker"
                        )
                        # stop() 은 finally 에서 호출되지만 flush 확실성 확보
                        self._heartbeat.stop()
                        self._shutdown_flag = True
                        sys.exit(2)
                else:
                    # 사이클 성공 → breaker 카운터 리셋
                    self._cycle_failure_count = 0

                if self._shutdown_flag:
                    break

                logger.info(
                    "=== 배치 사이클 #%d 완료. %d초 후 다음 사이클 ===",
                    self._cycle_number,
                    self.interval_seconds,
                )
                self._heartbeat.update(status="idle")
                self._interruptible_sleep(self.interval_seconds)

            logger.info("배치 스케줄러 종료.")
        finally:
            self._heartbeat.update(status="stopped")
            self._heartbeat.stop()

    def _interruptible_sleep(self, total_seconds: int) -> None:
        """1초 단위로 슬립하며 종료 플래그 확인

        time.sleep(300)을 통째로 걸면 Ctrl+C 이후 응답이 최대 300초 지연됩니다.
        1초 단위로 나누어 플래그를 확인합니다.
        """
        for _ in range(total_seconds):
            if self._shutdown_flag:
                return
            time.sleep(1)

    # ========================================
    # 비정상 종료 복구
    # ========================================

    def _recover_stale_processing(self) -> None:
        """프로세스 비정상 종료 시 PROCESSING 상태로 남은 파일을 PENDING으로 복구

        정상 종료에서는 PROCESSING이 즉시 COMPLETED/FAILED로 전환되므로,
        시작 시 PROCESSING 레코드가 남아 있다면 이전 실행의 비정상 종료를 의미합니다.
        """
        with DatabaseManager.from_config(self.storage_config) as db:
            stale = db.get_video_files_by_status(VideoFileStatus.PROCESSING)
            for video_file in stale:
                logger.warning(
                    "비정상 종료로 인해 PROCESSING 상태로 남은 파일을 PENDING으로 초기화: %s",
                    video_file.file_path,
                )
                db.update_video_file_status(
                    video_file.id,
                    VideoFileStatus.PENDING,
                )

    # ========================================
    # 디스크 공간 체크 (C-11)
    # ========================================

    def _check_disk_space(self) -> bool:
        """디스크 공간 부족 여부 체크.

        Returns:
            True: 사이클 진행 가능. False: 사이클 건너뛰기.
            ``abort`` action 시 sys.exit(4) 로 즉시 종료합니다.
        """
        try:
            import psutil  # 지연 import: 선택적 의존성
        except ImportError:
            logger.debug("psutil 미설치: 디스크 공간 체크 건너뜀.")
            return True

        try:
            usage = psutil.disk_usage(str(self.output_dir))
            free_mb = usage.free / (1024 * 1024)
        except Exception:
            logger.exception("디스크 공간 확인 실패")
            return True  # 안전 기본: 진행

        min_free = self.storage_config.min_free_disk_mb
        if free_mb >= min_free:
            return True

        action = self.storage_config.disk_check_action
        if action == "warn":
            logger.warning(
                "디스크 공간 부족 (free=%d MB < min=%d MB). 계속 진행.",
                int(free_mb),
                min_free,
            )
            return True
        if action == "skip":
            logger.warning(
                "디스크 공간 부족 (free=%d MB < min=%d MB). 사이클 건너뜀.",
                int(free_mb),
                min_free,
            )
            self._heartbeat.update(
                status="disk_low",
                free_mb=int(free_mb),
                min_free_mb=min_free,
            )
            return False
        if action == "abort":
            logger.critical(
                "디스크 공간 부족 (free=%d MB < min=%d MB). 배치 중단.",
                int(free_mb),
                min_free,
            )
            self._heartbeat.update(
                status="failed",
                reason="disk_full",
                free_mb=int(free_mb),
                min_free_mb=min_free,
            )
            self._heartbeat.stop()
            self._shutdown_flag = True
            sys.exit(4)
        # 알 수 없는 action은 안전 기본으로 진행
        logger.warning("알 수 없는 disk_check_action=%r, 진행합니다.", action)
        return True

    # ========================================
    # 배치 사이클
    # ========================================

    def _run_cycle(self) -> None:
        """단일 배치 사이클 실행

        0. 디스크 공간 체크 (부족 시 skip/abort)
        1. 디렉토리 스캔 → DB 등록
        2. PENDING 파일 순차 처리
        """
        if not self._check_disk_space():
            return

        with DatabaseManager.from_config(self.storage_config) as db:
            new_count = self._scan_and_register(db)
            logger.info("신규 파일 %d개 등록됨.", new_count)

        self._process_pending_files()

    # ========================================
    # 디렉토리 스캐너
    # ========================================

    def _scan_and_register(self, db: DatabaseManager) -> int:
        """watch_dir을 재귀 스캔하여 신규 비디오 파일을 video_files에 등록

        Args:
            db: 열린 DatabaseManager 인스턴스

        Returns:
            새로 등록된 파일 수
        """
        if not self.watch_dir.exists():
            logger.warning("watch_dir이 존재하지 않습니다: %s", self.watch_dir)
            return 0

        registered = 0
        for path in sorted(self.watch_dir.rglob("*")):
            if self._shutdown_flag:
                break
            if not path.is_file():
                continue
            if path.suffix.lower() not in _VIDEO_EXTENSIONS:
                continue

            abs_path = str(path.resolve())

            existing = db.get_video_file_by_path(abs_path)
            if existing is not None:
                continue

            try:
                file_size = path.stat().st_size
                db.create_video_file(
                    VideoFileCreate(
                        file_path=abs_path,
                        file_name=path.name,
                        file_size=file_size,
                    )
                )
                logger.info("신규 파일 등록: %s (%d bytes)", abs_path, file_size)
                registered += 1
            except Exception:
                logger.exception("파일 등록 실패: %s", abs_path)

        return registered

    # ========================================
    # 파일 처리
    # ========================================

    def _process_pending_files(self) -> None:
        """PENDING 상태의 파일을 순차적으로 처리

        처리 실패해도 다음 파일로 계속 진행합니다
        (단, 파일 단위 breaker가 trip하면 sys.exit(3)).
        """
        with DatabaseManager.from_config(self.storage_config) as db:
            pending = db.get_video_files_by_status(VideoFileStatus.PENDING)

        logger.info("처리 대기 파일: %d개", len(pending))

        for video_file in pending:
            if self._shutdown_flag:
                logger.info("종료 플래그 확인: 처리 루프 중단.")
                break
            self._process_one_file(video_file)

    def _build_app(self, video_file: VideoFile, file_output_dir: Path) -> VideoAnalyzerApp:
        """VideoAnalyzerApp 인스턴스 생성 (타임아웃 워커에서 재사용)."""
        return VideoAnalyzerApp(
            video_path=Path(video_file.file_path),
            output_dir=file_output_dir,
            template_id=self.template_id,
            template_name=self.template_name,
            use_gpu=self.use_gpu,
            frame_interval=self.frame_interval,
            ssim_threshold=self.ssim_threshold,
            confidence_threshold=self.confidence_threshold,
            max_workers=self.max_workers,
        )

    def _get_video_duration(self, path: Path) -> Optional[float]:
        """비디오 길이(초)를 계산. 실패 시 None.

        OpenCV VideoCapture의 CAP_PROP_FRAME_COUNT / CAP_PROP_FPS 사용.
        """
        try:
            import cv2  # 지연 import
        except ImportError:
            logger.debug("cv2 미설치: 비디오 길이 계산 건너뜀.")
            return None

        cap = None
        try:
            cap = cv2.VideoCapture(str(path))
            if not cap.isOpened():
                return None
            fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
            frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0
            if fps <= 0 or frames <= 0:
                return None
            return float(frames) / float(fps)
        except Exception:
            logger.exception("비디오 길이 계산 실패: %s", path)
            return None
        finally:
            if cap is not None:
                try:
                    cap.release()
                except Exception:
                    pass

    def _compute_file_timeout(self, duration_sec: Optional[float]) -> int:
        """파일 타임아웃(초) 계산.

        duration_sec 이 None 이면 min_file_timeout_sec 을 사용합니다.
        """
        min_timeout = self._watchdog_config.min_file_timeout_sec
        if duration_sec is None or duration_sec <= 0:
            return min_timeout
        computed = int(duration_sec * self._watchdog_config.file_timeout_multiplier)
        return max(min_timeout, computed)

    def _mark_file_failed(self, video_file: VideoFile, reason: str) -> None:
        """video_files 테이블 상태를 FAILED 로 업데이트."""
        try:
            with DatabaseManager.from_config(self.storage_config) as db:
                db.update_video_file_status(
                    video_file.id,
                    VideoFileStatus.FAILED,
                    completed_at=datetime.now(),
                    error_message=reason[:2000],
                )
        except Exception:
            logger.exception("실패 상태 업데이트 실패: id=%s", video_file.id)

    def _process_one_file(self, video_file: VideoFile) -> None:
        """단일 비디오 파일 분석 (워치독 타임아웃 + 파일 breaker)

        상태 전환: PENDING → PROCESSING → COMPLETED 또는 FAILED

        타임아웃 메커니즘:
            별도 ThreadPoolExecutor 에서 app.run() 을 실행하고
            ``future.result(timeout=...)`` 로 워치독을 건다.
            Python 한계상 타임아웃된 스레드를 강제 종료할 수 없으므로,
            타임아웃 발생 시 프로세스 전체를 ``sys.exit(3)`` 로 종료하여
            외부 supervisor(systemd 등)가 재시작하도록 위임한다.

        Args:
            video_file: 처리할 VideoFile 엔티티
        """
        file_path = Path(video_file.file_path)
        logger.info("처리 시작: %s", file_path.name)

        self._heartbeat.update(
            last_file=str(file_path),
            status="processing",
            file_started_at=datetime.now().isoformat(timespec="seconds"),
        )

        # 파일 존재 확인 (스캔 이후 삭제되었을 수 있음)
        if not file_path.exists():
            logger.warning("파일이 존재하지 않습니다. FAILED로 표시: %s", file_path)
            self._mark_file_failed(
                video_file, "파일을 찾을 수 없음 (처리 시작 전 삭제됨)"
            )
            return

        # PROCESSING으로 전환
        with DatabaseManager.from_config(self.storage_config) as db:
            db.update_video_file_status(
                video_file.id,
                VideoFileStatus.PROCESSING,
                started_at=datetime.now(),
            )

        # 파일별 출력 디렉토리: <output_dir>/<파일명>_<id>/
        file_output_dir = self.output_dir / f"{file_path.stem}_{video_file.id}"

        # 워치독 타임아웃 계산
        duration = self._get_video_duration(file_path)
        timeout_sec = self._compute_file_timeout(duration)
        logger.info(
            "파일 타임아웃 설정: %ds (duration=%s)",
            timeout_sec,
            f"{duration:.1f}s" if duration else "unknown",
        )

        def _run() -> None:
            app = self._build_app(video_file, file_output_dir)
            try:
                app.run()
            finally:
                try:
                    app.close()
                except Exception:
                    logger.exception("VideoAnalyzerApp.close() 실패")

        session_id: Optional[int] = None
        error_message: Optional[str] = None

        # B5: ThreadPoolExecutor 를 context manager 로 사용하지 않는다.
        # ``with`` 블록 내부에서 sys.exit 하면 __exit__ 의 shutdown(wait=True) 가
        # 멈춘 워커를 기다려 hang 된다. 수동 관리하여 타임아웃 시 cancel_futures 로
        # 즉시 반납하고 os._exit 로 프로세스를 강제 종료한다.
        ex = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="batch-file"
        )
        timed_out = False
        try:
            future = ex.submit(_run)
            try:
                future.result(timeout=timeout_sec)
                session_id = self._lookup_session_id_for_file(str(file_path))
            except FuturesTimeoutError:
                timed_out = True
                logger.error(
                    "파일 처리 타임아웃 (%ds): %s",
                    timeout_sec,
                    file_path,
                )
                self._mark_file_failed(
                    video_file, reason=f"timeout_{timeout_sec}s"
                )
                self._file_failure_count += 1
                self._last_failure_time = datetime.now()
                self._heartbeat.update(
                    status="failed",
                    reason=f"file_timeout_{timeout_sec}s",
                    last_file=str(file_path),
                )
                self._heartbeat.stop()
                logger.critical(
                    "타임아웃으로 프로세스 강제 종료 (os._exit 3)."
                )
                # 멈춘 워커를 기다리지 않도록 cancel_futures=True
                ex.shutdown(wait=False, cancel_futures=True)
                # B5: sys.exit 대신 os._exit 사용 — ThreadPoolExecutor.__exit__ 의
                # wait=True 대기로 인한 hang 을 우회하고 systemd 재시작 유도.
                self._timeout_exit(3)
                return  # _timeout_exit 가 mock 된 테스트 경로
            except Exception as e:
                error_message = f"{type(e).__name__}: {str(e)[:2000]}"
                logger.exception(
                    "분석 실패: %s - %s", file_path.name, error_message
                )
        finally:
            if not timed_out:
                ex.shutdown(wait=True)

        # 완료/실패 상태 업데이트
        if error_message is None:
            with DatabaseManager.from_config(self.storage_config) as db:
                db.update_video_file_status(
                    video_file.id,
                    VideoFileStatus.COMPLETED,
                    session_id=session_id,
                    completed_at=datetime.now(),
                )
            logger.info(
                "처리 완료: %s (session_id=%s)", file_path.name, session_id
            )
            # 성공 → 파일 breaker 카운터 리셋
            self._file_failure_count = 0
            self._heartbeat.update(
                status="file_completed",
                last_file=str(file_path),
                last_session_id=session_id,
            )
        else:
            self._mark_file_failed(video_file, error_message)
            self._file_failure_count += 1
            self._last_failure_time = datetime.now()
            logger.warning(
                "처리 실패: %s (연속 실패 %d회)",
                file_path.name,
                self._file_failure_count,
            )
            self._heartbeat.update(
                status="file_failed",
                last_file=str(file_path),
                file_failure_count=self._file_failure_count,
                last_failure_at=self._last_failure_time.isoformat(
                    timespec="seconds"
                ),
            )

            if (
                self._file_failure_count
                >= self._breaker_config.max_consecutive_file_failures
            ):
                logger.critical(
                    "파일 연속 실패 %d회, breaker trip. 프로세스 종료.",
                    self._file_failure_count,
                )
                self._heartbeat.update(
                    status="failed", reason="file_breaker"
                )
                self._heartbeat.stop()
                self._shutdown_flag = True
                sys.exit(3)

    def _lookup_session_id_for_file(self, source_path: str) -> Optional[int]:
        """VideoAnalyzerApp이 생성한 세션의 ID를 source_path로 역조회 (H-14).

        Phase 1-E 변경: 기존 ``get_all_sessions()`` O(N) 전수 스캔 대신,
        ``DatabaseManager.get_latest_session_by_source_path`` 저장소 API를 사용.

        Args:
            source_path: 비디오 파일 절대 경로 문자열

        Returns:
            session_id 또는 None
        """
        try:
            with DatabaseManager.from_config(self.storage_config) as db:
                session = db.get_latest_session_by_source_path(source_path)
            return session.id if session else None
        except Exception:
            logger.exception("세션 ID 역조회 실패: %s", source_path)
            return None
