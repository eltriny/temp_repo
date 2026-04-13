"""Graceful shutdown 코디네이터.

SIGINT/SIGTERM(+ Windows SIGBREAK)을 통합 처리하고, 등록된 cleanup을 LIFO로 실행.
"""
from __future__ import annotations

import atexit
import logging
import signal
import threading
from typing import Callable, Optional

logger = logging.getLogger(__name__)


class ShutdownCoordinator:
    def __init__(self) -> None:
        self._flag = threading.Event()
        self._cleanups: list[Callable[[], None]] = []
        self._lock = threading.Lock()
        self._atexit_installed = False

    def register(self, cleanup: Callable[[], None]) -> None:
        """LIFO 순서로 실행될 cleanup 등록."""
        with self._lock:
            self._cleanups.append(cleanup)

    def trigger(self, signum: Optional[int] = None) -> None:
        """종료 플래그 설정 및 cleanup 실행."""
        if self._flag.is_set():
            return
        self._flag.set()
        if signum is not None:
            logger.info("Shutdown triggered by signal %d", signum)
        self._run_cleanups()

    def is_set(self) -> bool:
        return self._flag.is_set()

    def wait(self, timeout: Optional[float] = None) -> bool:
        return self._flag.wait(timeout)

    def install_signal_handlers(self) -> None:
        def _handler(signum: int, frame: object) -> None:
            self.trigger(signum)

        signal.signal(signal.SIGINT, _handler)
        signal.signal(signal.SIGTERM, _handler)
        if hasattr(signal, "SIGBREAK"):  # Windows Ctrl+Break
            signal.signal(signal.SIGBREAK, _handler)  # type: ignore[attr-defined]

    def install_atexit(self) -> None:
        if self._atexit_installed:
            return
        atexit.register(self._atexit_flush)
        self._atexit_installed = True

    def _atexit_flush(self) -> None:
        """IDE 강제 종료 / 정상 exit 등 signal 없는 경로 대비."""
        if not self._flag.is_set():
            self._run_cleanups()

    def _run_cleanups(self) -> None:
        with self._lock:
            cleanups = list(reversed(self._cleanups))
            self._cleanups.clear()
        for cleanup in cleanups:
            try:
                cleanup()
            except Exception:
                logger.exception("Shutdown cleanup 실패: %s", cleanup)
