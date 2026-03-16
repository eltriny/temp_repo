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
"""

from __future__ import annotations

import logging
import signal
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

from .config import StorageConfig
from .main import VideoAnalyzerApp
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


class BatchScheduler:
    """배치 비디오 분석 스케줄러

    지정 디렉토리를 주기적으로 스캔하여 신규 비디오 파일을 발견하고,
    video_files 테이블에 등록한 뒤 순차적으로 분석합니다.

    Attributes:
        watch_dir: 스캔할 비디오 디렉토리
        output_dir: 분석 결과 저장 디렉토리
        storage_config: Oracle 데이터베이스 연결 설정
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
    ) -> None:
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

        # 출력 디렉토리 보장
        self.output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            "BatchScheduler 초기화: watch_dir=%s, interval=%ds",
            self.watch_dir,
            self.interval_seconds,
        )

    # ========================================
    # 시그널 핸들링
    # ========================================

    def _register_signal_handlers(self) -> None:
        """SIGINT(Ctrl+C) 및 SIGTERM 핸들러 등록"""

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
        각 사이클: 디렉토리 스캔 → 신규 등록 → PENDING 파일 순차 처리 → 대기
        """
        self._register_signal_handlers()
        self._recover_stale_processing()

        logger.info("배치 스케줄러 시작. 종료: Ctrl+C")

        cycle_number = 0
        while not self._shutdown_flag:
            cycle_number += 1
            logger.info("=== 배치 사이클 #%d 시작 ===", cycle_number)

            try:
                self._run_cycle()
            except Exception:
                logger.exception(
                    "배치 사이클 #%d 예외 발생. 다음 사이클을 기다립니다.",
                    cycle_number,
                )

            if self._shutdown_flag:
                break

            logger.info(
                "=== 배치 사이클 #%d 완료. %d초 후 다음 사이클 ===",
                cycle_number,
                self.interval_seconds,
            )
            self._interruptible_sleep(self.interval_seconds)

        logger.info("배치 스케줄러 종료.")

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
    # 배치 사이클
    # ========================================

    def _run_cycle(self) -> None:
        """단일 배치 사이클 실행

        1. 디렉토리 스캔 → DB 등록
        2. PENDING 파일 순차 처리
        """
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

        처리 실패해도 다음 파일로 계속 진행합니다.
        """
        with DatabaseManager.from_config(self.storage_config) as db:
            pending = db.get_video_files_by_status(VideoFileStatus.PENDING)

        logger.info("처리 대기 파일: %d개", len(pending))

        for video_file in pending:
            if self._shutdown_flag:
                logger.info("종료 플래그 확인: 처리 루프 중단.")
                break
            self._process_one_file(video_file)

    def _process_one_file(self, video_file: VideoFile) -> None:
        """단일 비디오 파일 분석

        상태 전환: PENDING → PROCESSING → COMPLETED 또는 FAILED

        Args:
            video_file: 처리할 VideoFile 엔티티
        """
        file_path = Path(video_file.file_path)
        logger.info("처리 시작: %s", file_path.name)

        # 파일 존재 확인 (스캔 이후 삭제되었을 수 있음)
        if not file_path.exists():
            logger.warning("파일이 존재하지 않습니다. FAILED로 표시: %s", file_path)
            with DatabaseManager.from_config(self.storage_config) as db:
                db.update_video_file_status(
                    video_file.id,
                    VideoFileStatus.FAILED,
                    completed_at=datetime.now(),
                    error_message="파일을 찾을 수 없음 (처리 시작 전 삭제됨)",
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

        session_id: Optional[int] = None
        error_message: Optional[str] = None

        try:
            app = VideoAnalyzerApp(
                video_path=file_path,
                output_dir=file_output_dir,
                template_id=self.template_id,
                template_name=self.template_name,
                use_gpu=self.use_gpu,
                frame_interval=self.frame_interval,
                ssim_threshold=self.ssim_threshold,
                confidence_threshold=self.confidence_threshold,
                max_workers=self.max_workers,
            )
            app.run()
            session_id = self._lookup_session_id_for_file(str(file_path))

        except Exception as e:
            error_message = f"{type(e).__name__}: {str(e)[:2000]}"
            logger.exception("분석 실패: %s - %s", file_path.name, error_message)

        # 완료/실패 상태 업데이트
        if error_message is None:
            with DatabaseManager.from_config(self.storage_config) as db:
                db.update_video_file_status(
                    video_file.id,
                    VideoFileStatus.COMPLETED,
                    session_id=session_id,
                    completed_at=datetime.now(),
                )
            logger.info("처리 완료: %s (session_id=%s)", file_path.name, session_id)
        else:
            with DatabaseManager.from_config(self.storage_config) as db:
                db.update_video_file_status(
                    video_file.id,
                    VideoFileStatus.FAILED,
                    completed_at=datetime.now(),
                    error_message=error_message,
                )
            logger.warning("처리 실패: %s", file_path.name)

    def _lookup_session_id_for_file(self, source_path: str) -> Optional[int]:
        """VideoAnalyzerApp이 생성한 세션의 ID를 source_path로 역조회

        VideoAnalyzerApp.run()은 session_id를 외부에 반환하지 않으므로,
        완료 직후 source_path 기준으로 가장 최신 비활성 세션을 찾습니다.

        Args:
            source_path: 비디오 파일 절대 경로 문자열

        Returns:
            session_id 또는 None
        """
        try:
            with DatabaseManager.from_config(self.storage_config) as db:
                sessions = db.get_all_sessions()
            matches = [
                session
                for session in sessions
                if session.source_path == source_path and not session.is_active
            ]
            if matches:
                return max(matches, key=lambda session: session.created_at).id
        except Exception:
            logger.exception("세션 ID 역조회 실패: %s", source_path)
        return None
