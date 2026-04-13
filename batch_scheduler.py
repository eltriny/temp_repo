"""
배치 스케줄러 - 단일 사이클 실행 모드

Windows Task Scheduler 등 외부 스케줄러에서 주기적으로 호출됩니다.
한 번의 스캔/처리 사이클을 실행하고 종료합니다.

Exit codes:
    0: 성공 (파일 처리 완료 또는 처리 대상 없음)
    1: 에러 발생

사용 예시:
    # Windows Task Scheduler에서 5분마다 실행:
    python -m src.main --watch-dir ./videos --template-id 1

    # 수동 1회 실행:
    python -m src.main --watch-dir ./videos --template "모니터링 레이아웃 A"
"""

from __future__ import annotations

import logging
import os
import signal
import sys
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

_VIDEO_EXTENSIONS: frozenset[str] = frozenset(
    {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".m4v", ".ts"}
)

# Exit codes (Windows Task Scheduler는 non-zero를 실패로 표시)
EXIT_SUCCESS = 0
EXIT_ERROR = 1

# 연속 실패 제한 (circuit breaker)
MAX_CONSECUTIVE_FAILURES = 5


class PIDLock:
    """PID 파일 기반 프로세스 독점 락 (Windows/Linux 호환)

    동일 watch_dir에 대해 배치 프로세스가 중복 실행되는 것을 방지합니다.
    """

    def __init__(self, lock_dir: Path) -> None:
        self._lock_path = lock_dir / ".batch_scheduler.lock"
        self._lock_file = None

    def acquire(self) -> bool:
        """락 획득 시도. 이미 다른 프로세스가 보유 중이면 False 반환."""
        try:
            # 락 파일을 독점 모드로 오픈
            self._lock_file = open(self._lock_path, "w")
            if sys.platform == "win32":
                import msvcrt

                msvcrt.locking(self._lock_file.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(self._lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)

            self._lock_file.write(str(os.getpid()))
            self._lock_file.flush()
            logger.info("PID 락 획득: %s (PID=%d)", self._lock_path, os.getpid())
            return True
        except (IOError, OSError):
            if self._lock_file:
                self._lock_file.close()
                self._lock_file = None
            return False

    def release(self) -> None:
        """락 해제 및 파일 삭제."""
        if self._lock_file is not None:
            try:
                if sys.platform == "win32":
                    import msvcrt

                    msvcrt.locking(self._lock_file.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    import fcntl

                    fcntl.flock(self._lock_file.fileno(), fcntl.LOCK_UN)
                self._lock_file.close()
            except (IOError, OSError):
                pass
            self._lock_file = None
            try:
                self._lock_path.unlink(missing_ok=True)
            except OSError:
                pass

    def __enter__(self):
        if not self.acquire():
            raise RuntimeError(
                f"다른 배치 프로세스가 이미 실행 중입니다. 락 파일: {self._lock_path}"
            )
        return self

    def __exit__(self, *args):
        self.release()


class BatchScheduler:
    """단일 사이클 배치 스케줄러

    외부 스케줄러(Windows Task Scheduler)에서 주기적으로 호출되어
    한 번의 스캔/처리 사이클을 실행합니다.
    """

    def __init__(
        self,
        watch_dir: Path,
        output_dir: Path,
        storage_config: Optional[StorageConfig] = None,
        template_id: Optional[int] = None,
        template_name: Optional[str] = None,
        use_gpu: bool = False,
        frame_interval: float = 1.0,
        ssim_threshold: float = 0.95,
        confidence_threshold: float = 0.7,
        max_workers: Optional[int] = None,
    ) -> None:
        if template_id is None and template_name is None:
            raise ValueError(
                "배치 모드에는 --template 또는 --template-id가 필요합니다."
            )

        self.watch_dir = watch_dir.resolve()
        self.output_dir = output_dir
        self.storage_config = storage_config or StorageConfig(output_dir=output_dir)
        self.template_id = template_id
        self.template_name = template_name
        self.use_gpu = use_gpu
        self.frame_interval = frame_interval
        self.ssim_threshold = ssim_threshold
        self.confidence_threshold = confidence_threshold
        self.max_workers = max_workers
        self._shutdown_flag: bool = False

        self.output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            "BatchScheduler 초기화: watch_dir=%s (단일 사이클 모드)",
            self.watch_dir,
        )

    # ========================================
    # 시그널 핸들링
    # ========================================

    def _register_signal_handlers(self) -> None:
        """SIGINT/SIGTERM 핸들러 등록 (처리 중 graceful shutdown)"""

        def _handler(signum: int, frame: object) -> None:
            logger.info(
                "종료 신호 수신 (signal=%d). 현재 파일 처리 완료 후 종료합니다.",
                signum,
            )
            self._shutdown_flag = True

        signal.signal(signal.SIGINT, _handler)
        signal.signal(signal.SIGTERM, _handler)

    # ========================================
    # 단일 사이클 실행
    # ========================================

    def run_single_cycle(self) -> int:
        """단일 배치 사이클 실행

        1. PID 락 획득 (중복 실행 방지)
        2. 비정상 종료 복구 (PROCESSING → PENDING)
        3. 디렉토리 스캔 → 신규 등록
        4. PENDING 파일 순차 처리 (circuit breaker 적용)

        Returns:
            exit code: 0=성공(처리완료 또는 대상없음), 1=에러
        """
        self._register_signal_handlers()

        try:
            with PIDLock(self.watch_dir):
                return self._execute_cycle()
        except RuntimeError as e:
            logger.warning("배치 실행 건너뜀: %s", e)
            return EXIT_SUCCESS
        except Exception:
            logger.exception("배치 사이클 치명적 예외 발생")
            return EXIT_ERROR

    def _execute_cycle(self) -> int:
        """PID 락 내부에서 실제 사이클 실행"""
        self._recover_stale_processing()

        # 스캔 및 등록
        with DatabaseManager.from_config(self.storage_config) as db:
            new_count = self._scan_and_register(db)
            logger.info("신규 파일 %d개 등록됨.", new_count)

        # 처리
        processed, failed = self._process_pending_files()

        if processed == 0 and failed == 0:
            logger.info("처리 대상 파일 없음.")
            return EXIT_SUCCESS

        if failed > 0:
            logger.warning(
                "배치 사이클 완료: 성공=%d, 실패=%d", processed, failed
            )
            return EXIT_ERROR if processed == 0 else EXIT_SUCCESS

        logger.info("배치 사이클 완료: %d개 파일 처리 성공.", processed)
        return EXIT_SUCCESS

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

    def _process_pending_files(self) -> tuple[int, int]:
        """PENDING 파일 순차 처리 (circuit breaker 적용)

        Returns:
            (처리 성공 수, 처리 실패 수)
        """
        with DatabaseManager.from_config(self.storage_config) as db:
            pending = db.get_video_files_by_status(VideoFileStatus.PENDING)

        logger.info("처리 대기 파일: %d개", len(pending))

        processed = 0
        failed = 0
        consecutive_failures = 0

        for video_file in pending:
            if self._shutdown_flag:
                logger.info("종료 플래그 확인: 처리 루프 중단.")
                break

            # Circuit breaker: 연속 실패 제한
            if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                logger.error(
                    "연속 %d회 실패 (circuit breaker). 남은 파일 처리 중단.",
                    consecutive_failures,
                )
                break

            success = self._process_one_file(video_file)
            if success:
                processed += 1
                consecutive_failures = 0
            else:
                failed += 1
                consecutive_failures += 1

        return processed, failed

    def _process_one_file(self, video_file: VideoFile) -> bool:
        """단일 비디오 파일 분석. Returns True on success."""
        file_path = Path(video_file.file_path)
        logger.info("처리 시작: %s", file_path.name)

        if not file_path.exists():
            logger.warning("파일 미존재. FAILED 처리: %s", file_path)
            with DatabaseManager.from_config(self.storage_config) as db:
                db.update_video_file_status(
                    video_file.id,
                    VideoFileStatus.FAILED,
                    completed_at=datetime.now(),
                    error_message="파일을 찾을 수 없음",
                )
            return False

        # PROCESSING 전환
        with DatabaseManager.from_config(self.storage_config) as db:
            db.update_video_file_status(
                video_file.id,
                VideoFileStatus.PROCESSING,
                started_at=datetime.now(),
            )

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

        # 상태 업데이트
        status = (
            VideoFileStatus.COMPLETED
            if error_message is None
            else VideoFileStatus.FAILED
        )
        with DatabaseManager.from_config(self.storage_config) as db:
            db.update_video_file_status(
                video_file.id,
                status,
                session_id=session_id,
                completed_at=datetime.now(),
                error_message=error_message,
            )

        if error_message is None:
            logger.info(
                "처리 완료: %s (session_id=%s)", file_path.name, session_id
            )
            return True
        else:
            logger.warning("처리 실패: %s", file_path.name)
            return False

    def _lookup_session_id_for_file(self, source_path: str) -> Optional[int]:
        """최적화된 세션 ID 역조회 (DB 쿼리 기반)"""
        try:
            with DatabaseManager.from_config(self.storage_config) as db:
                # 직접 DB 쿼리로 최적화 (기존: get_all_sessions() 전체 로드 후 Python 필터)
                row = db._execute(
                    """
                    SELECT id FROM sessions
                    WHERE source_path = ?
                      AND is_active = 0
                    ORDER BY created_at DESC
                    FETCH FIRST 1 ROWS ONLY
                    """,
                    (source_path,),
                    fetch="one",
                )
                if row:
                    return int(row.get("id"))
        except Exception:
            logger.exception("세션 ID 역조회 실패: %s", source_path)
        return None
