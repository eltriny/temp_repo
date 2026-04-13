"""헬스체크 heartbeat 파일 writer."""
from __future__ import annotations

import json
import logging
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class HeartbeatWriter:
    def __init__(self, path: Path, interval: float = 30.0) -> None:
        self._path = Path(path)
        self._interval = interval
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._state: dict[str, object] = {"status": "init"}
        self._state_lock = threading.Lock()

    def update(self, **fields: object) -> None:
        with self._state_lock:
            self._state.update(fields)

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._loop, name="heartbeat", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                self._write_once()
            except Exception:
                logger.exception("Heartbeat 쓰기 실패")
            self._stop.wait(self._interval)

    def _write_once(self) -> None:
        with self._state_lock:
            payload = {
                "ts": datetime.now(timezone.utc).isoformat(),
                **self._state,
            }
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._path.with_suffix(self._path.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        tmp.replace(self._path)  # 원자적 교체
