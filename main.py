"""
산업용 비디오 모니터링 분석 프로그램 (v2.0 - 사전 정의 ROI 기반)

주요 변경사항 (v2.0):
  - 자동 ROI 감지 제거
  - 사전 정의된 ROI 템플릿 사용
  - ROI별 병렬 처리 지원

실행 방법:
  프로젝트 루트에서: python -m src.main --video video.mp4 --output ./results --template "기본 템플릿"
  또는: cd src && python main.py --video ../video.mp4 --output ../results --template-id 1
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

# 상대 임포트를 위한 경로 설정 (스크립트 직접 실행 시)
if __name__ == "__main__" and __package__ is None:
    _src_dir = Path(__file__).resolve().parent
    _project_root = _src_dir.parent
    if str(_project_root) not in sys.path:
        sys.path.insert(0, str(_project_root))
    __package__ = "src"

from .config import StorageConfig
from .observability.lifecycle import ShutdownCoordinator
from .observability.logging import LoggingConfig, configure_logging
from .storage.database import DatabaseManager, ROIType
from .storage.roi_template_manager import ROITemplateManager

logger = logging.getLogger(__name__)


# ========================================
# 템플릿 관리 CLI
# ========================================


def list_templates(storage_config: StorageConfig) -> None:
    """템플릿 목록 출력"""
    with DatabaseManager.from_config(storage_config) as db:
        manager = ROITemplateManager(db)
        templates = manager.list_templates()

        if not templates:
            print("등록된 템플릿이 없습니다.")
            return

        print("\n" + "=" * 60)
        print("ROI 템플릿 목록")
        print("=" * 60)
        for info in templates:
            t = info.template
            print(f"  ID: {t.id}")
            print(f"  이름: {t.name}")
            print(f"  설명: {t.description or '(없음)'}")
            print(f"  ROI 개수: {info.roi_count}")
            print(f"  생성일: {t.created_at}")
            print("-" * 60)
        print()


def create_template_interactive(storage_config: StorageConfig) -> None:
    """템플릿 대화형 생성"""
    print("\n" + "=" * 60)
    print("새 ROI 템플릿 생성")
    print("=" * 60)

    name = input("템플릿 이름: ").strip()
    if not name:
        print("취소: 이름이 필요합니다.")
        return

    description = input("설명 (선택): ").strip() or None

    with DatabaseManager.from_config(storage_config) as db:
        manager = ROITemplateManager(db)
        template = manager.create_template(name=name, description=description)
        print(f"\n템플릿 생성 완료: ID={template.id}")

        # ROI 추가
        print("\nROI를 추가합니다. (종료: 빈 입력)")
        roi_count = 0

        while True:
            print(f"\n--- ROI #{roi_count + 1} ---")
            roi_name = input("ROI 이름 (빈 입력=종료): ").strip()
            if not roi_name:
                break

            roi_type_str = (
                input("유형 (numeric/text/chart) [numeric]: ").strip() or "numeric"
            )
            try:
                roi_type = ROIType(roi_type_str)
            except ValueError:
                print(f"잘못된 유형: {roi_type_str}")
                continue

            try:
                x = int(input("X 좌표: "))
                y = int(input("Y 좌표: "))
                width = int(input("너비: "))
                height = int(input("높이: "))
            except ValueError:
                print("잘못된 좌표 입력")
                continue

            manager.add_roi_to_template(
                template_id=template.id,
                name=roi_name,
                roi_type=roi_type,
                x=x,
                y=y,
                width=width,
                height=height,
            )
            roi_count += 1
            print(f"ROI '{roi_name}' 추가됨")

        print(f"\n템플릿 '{name}'에 {roi_count}개 ROI 추가 완료")


def show_template_rois(storage_config: StorageConfig, template_id: int) -> None:
    """템플릿의 ROI 목록 출력"""
    with DatabaseManager.from_config(storage_config) as db:
        manager = ROITemplateManager(db)
        template = manager.get_template(template_id)

        if not template:
            print(f"템플릿 ID={template_id}를 찾을 수 없습니다.")
            return

        rois = manager.get_template_rois(template_id)

        print("\n" + "=" * 60)
        print(f"템플릿: {template.name} (ID={template.id})")
        print("=" * 60)

        if not rois:
            print("등록된 ROI가 없습니다.")
            return

        for roi in rois:
            print(f"  [{roi.id}] {roi.name}")
            print(f"      유형: {roi.roi_type.value}")
            print(f"      좌표: ({roi.x}, {roi.y}) - {roi.width}x{roi.height}")
            print(f"      임계값: {roi.threshold}")
        print()


# ========================================
# 명령행 인터페이스
# ========================================


def parse_args() -> argparse.Namespace:
    """명령행 인자 파싱"""
    parser = argparse.ArgumentParser(
        description="산업용 비디오 모니터링 분석 프로그램 (v2.0 - 사전 정의 ROI 기반)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용 예시:
  # 비디오 분석 (템플릿 ID 지정)
  python main.py --video input.mp4 --output ./results --template-id 1

  # 비디오 분석 (템플릿 이름 지정)
  python main.py --video input.mp4 --output ./results --template "모니터링 레이아웃 A"

  # 템플릿 관리
  python main.py --list-templates
  python main.py --create-template
  python main.py --show-template 1

  # 배치 스케줄링 모드 (디렉토리 감시, 5분 주기)
  python main.py --watch-dir ./videos --template "모니터링 레이아웃 A"

  # 커스텀 주기 (10분)
  python main.py --watch-dir ./videos --template-id 1 --batch-interval 600
        """,
    )

    # 비디오 분석 옵션
    parser.add_argument("--video", "-v", type=Path, help="분석할 비디오 파일 경로")

    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=Path("./data"),
        help="결과 저장 디렉토리 (기본값: ./data)",
    )

    parser.add_argument(
        "--db-dsn",
        type=str,
        default=None,
        help="PostgreSQL DSN (예: postgresql://user:pass@localhost:5432/video_detection)",
    )

    # 템플릿 선택
    parser.add_argument("--template-id", type=int, help="사용할 ROI 템플릿 ID")

    parser.add_argument("--template", "-t", type=str, help="사용할 ROI 템플릿 이름")

    # 처리 옵션
    parser.add_argument("--gpu", action="store_true", help="GPU 가속 사용 (PaddleOCR)")

    parser.add_argument(
        "--workers",
        "-w",
        type=int,
        default=None,
        help="병렬 처리 워커 수 (기본값: CPU 코어 수의 75%%)",
    )

    parser.add_argument(
        "--interval",
        "-i",
        type=float,
        default=1.0,
        help="프레임 분석 간격 (초, 기본값: 1.0)",
    )

    parser.add_argument(
        "--ssim-threshold",
        type=float,
        default=0.95,
        help="SSIM 변화 감지 임계값 (기본값: 0.95)",
    )

    parser.add_argument(
        "--confidence", type=float, default=0.7, help="OCR 신뢰도 임계값 (기본값: 0.7)"
    )

    parser.add_argument(
        "--auto-detect",
        action="store_true",
        help="첫 프레임에서 ROI 자동 탐지 (템플릿 불필요)",
    )

    parser.add_argument(
        "--roi-template",
        nargs="+",
        type=Path,
        metavar="IMAGE",
        help="색상으로 ROI를 표시한 템플릿 이미지 경로 (복수 가능). "
        "빨강=NUMERIC, 초록=TEXT, 파랑=CHART",
    )

    parser.add_argument(
        "--anchor-config",
        type=Path,
        metavar="YAML",
        help="앵커 기반 ROI 탐지 설정 YAML 파일 경로. "
        "고정 참조점(스니펫/텍스트)을 기준으로 상대 좌표로 ROI를 자동 탐지합니다.",
    )

    parser.add_argument("--debug", action="store_true", help="디버그 모드 활성화")

    # 템플릿 관리 옵션
    parser.add_argument(
        "--list-templates", action="store_true", help="등록된 템플릿 목록 출력"
    )

    parser.add_argument(
        "--create-template", action="store_true", help="새 템플릿 대화형 생성"
    )

    parser.add_argument(
        "--show-template", type=int, metavar="ID", help="템플릿 ROI 상세 정보 출력"
    )

    # 배치 스케줄링 옵션
    batch_group = parser.add_argument_group("배치 스케줄링 옵션")

    batch_group.add_argument(
        "--batch",
        action="store_true",
        help="배치 스케줄링 모드 활성화 (--watch-dir 지정 시 자동 활성화)",
    )

    batch_group.add_argument(
        "--watch-dir",
        type=Path,
        metavar="DIR",
        help="스캔할 비디오 디렉토리 경로 (지정 시 배치 모드 자동 실행)",
    )

    batch_group.add_argument(
        "--batch-interval",
        type=int,
        default=300,
        metavar="SECONDS",
        help="배치 사이클 간격 (초, 기본값: 300 = 5분)",
    )

    return parser.parse_args()


def _configure_runtime_logging(args: argparse.Namespace) -> None:
    """argparse 결과를 바탕으로 로깅 인프라를 1회 설정."""
    output_dir = getattr(args, "output", None) or Path("./data")
    log_dir = Path(output_dir) / "logs"
    level = "DEBUG" if getattr(args, "debug", False) else "INFO"
    log_cfg = LoggingConfig(
        level=level,
        log_dir=log_dir,
        json_format=os.getenv("VIDEO_DETECTION_ENV") == "prod",
    )
    configure_logging(log_cfg)


def main() -> int:
    """메인 진입점"""
    args = parse_args()

    # ★ Windows DLL 충돌 방지: 프로세스 최초 시점에서 torch DLL 선점 로드
    # paddleocr/albumentations가 torch를 간접 import하기 전에
    # torch의 libiomp5md.dll을 먼저 로드하여 DLL 버전 충돌 방지
    try:
        import torch  # noqa: F401 - Windows DLL preload at process start
    except (ImportError, OSError):
        pass  # torch 미설치 또는 로드 실패 시 무시

    # 로깅 설정 — main 진입 후 1회만 (import 시점 side effect 없음)
    _configure_runtime_logging(args)

    # Graceful shutdown 코디네이터
    shutdown = ShutdownCoordinator()
    shutdown.install_signal_handlers()
    shutdown.install_atexit()

    # 데이터베이스 설정 결정
    storage_config = StorageConfig(output_dir=args.output)
    if args.db_dsn:
        storage_config = StorageConfig(db_dsn=args.db_dsn, output_dir=args.output)

    # 템플릿 관리 명령
    if args.list_templates:
        list_templates(storage_config)
        return 0

    if args.create_template:
        create_template_interactive(storage_config)
        return 0

    if args.show_template:
        show_template_rois(storage_config, args.show_template)
        return 0

    # --video와 --watch-dir 동시 지정 방지
    if args.video and args.watch_dir:
        print("오류: --video와 --watch-dir은 동시에 사용할 수 없습니다.")
        print("  단일 분석: python main.py --video input.mp4 --template-id 1")
        print("  배치 모드: python main.py --watch-dir ./videos --template-id 1")
        return 1

    # 배치 스케줄링 모드: --batch 또는 --watch-dir 지정 시 진입
    if args.batch or args.watch_dir:
        if not args.watch_dir:
            print("오류: --batch 모드에는 --watch-dir이 필요합니다.")
            return 1

        if not args.watch_dir.exists():
            print(f"오류: watch-dir이 존재하지 않습니다: {args.watch_dir}")
            return 1

        if not args.template_id and not args.template:
            print("오류: 배치 모드에는 --template 또는 --template-id가 필요합니다.")
            return 1

        from .batch_scheduler import BatchScheduler

        scheduler = BatchScheduler(
            watch_dir=args.watch_dir,
            output_dir=args.output,
            storage_config=storage_config,
            template_id=args.template_id,
            template_name=args.template,
            interval_seconds=args.batch_interval,
            use_gpu=args.gpu,
            frame_interval=args.interval,
            ssim_threshold=args.ssim_threshold,
            confidence_threshold=args.confidence,
            max_workers=args.workers,
            shutdown_coordinator=shutdown,
        )
        scheduler.run()
        return 0

    # 단일 비디오 분석
    if not args.video:
        print("오류: --video 또는 --watch-dir 옵션이 필요합니다.")
        print("  단일 분석: python main.py --video input.mp4 --template-id 1")
        print("  배치 모드: python main.py --watch-dir ./videos --template-id 1")
        print("  도움말: python main.py --help")
        return 1

    if not args.video.exists():
        logger.error(f"비디오 파일을 찾을 수 없습니다: {args.video}")
        return 1

    roi_template_paths = getattr(args, "roi_template", None) or []

    # --roi-template 파일 존재 확인
    for path in roi_template_paths:
        if not path.exists():
            print(f"오류: 템플릿 이미지를 찾을 수 없습니다: {path}")
            return 1

    anchor_config = getattr(args, "anchor_config", None)

    # --anchor-config 파일 존재 확인
    if anchor_config and not anchor_config.exists():
        print(f"오류: 앵커 설정 파일을 찾을 수 없습니다: {anchor_config}")
        return 1

    if (
        not roi_template_paths
        and not args.auto_detect
        and not args.template_id
        and not args.template
        and not anchor_config
    ):
        print(
            "오류: --template-id, --template, --auto-detect, --roi-template, "
            "또는 --anchor-config 옵션이 필요합니다."
        )
        print("템플릿 목록 확인: python main.py --list-templates")
        print("템플릿 생성: python main.py --create-template")
        print("자동 ROI 탐지: python main.py --video input.mp4 --auto-detect")
        print(
            "색상 마커 ROI: python main.py --video input.mp4 --roi-template annotated.png"
        )
        print(
            "앵커 기반 ROI: python main.py --video input.mp4 --anchor-config config.yaml"
        )
        return 1

    # VideoAnalyzerApp은 여기서 import (로깅 설정 이후, paddle import 지연)
    from .app import VideoAnalyzerApp

    app = VideoAnalyzerApp(
        video_path=args.video,
        output_dir=args.output,
        template_id=args.template_id,
        template_name=args.template,
        use_gpu=args.gpu,
        frame_interval=args.interval,
        ssim_threshold=args.ssim_threshold,
        confidence_threshold=args.confidence,
        max_workers=args.workers,
        auto_detect=args.auto_detect,
        roi_template_paths=roi_template_paths,
        anchor_config_path=anchor_config,
    )
    shutdown.register(app.close)

    try:
        app.run()
        return 0
    except KeyboardInterrupt:
        logger.info("사용자에 의해 분석이 중단되었습니다.")
        shutdown.trigger()
        return 0
    except Exception as e:
        logger.exception("분석 중 오류 발생: %s", e)
        return 1
    finally:
        try:
            app.close()
        except Exception:
            logger.exception("앱 close 중 예외 발생")


if __name__ == "__main__":
    sys.exit(main())
