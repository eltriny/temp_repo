"""로깅 인프라 — RotatingFileHandler + 구조화 로깅 지원.

import 시점에는 아무것도 설정하지 않음 (side-effect-free).
main/entrypoint에서 configure_logging()을 1회 호출.
"""
from __future__ import annotations

import json
import logging
import logging.config
import threading
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class LoggingConfig:
    level: str = "INFO"
    log_dir: Path = field(default_factory=lambda: Path("./logs"))
    file_name: str = "video_analyzer.log"
    max_bytes: int = 50_000_000
    backup_count: int = 5
    json_format: bool = False
    namespace_logger: str = "src"
    suppress_paddle_root: bool = True


_CONFIGURED = False
_LOCK = threading.Lock()


class JsonFormatter(logging.Formatter):
    """최소 구조화 JSON 포맷터. python-json-logger 의존성 없음."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "lvl": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


def configure_logging(cfg: LoggingConfig) -> None:
    """로깅 설정을 멱등 적용.

    - root logger에 NullHandler만 부착 (paddle/glog 격리)
    - src namespace logger에 Stream + RotatingFile handler 부착
    - 중복 호출 시 기존 handler 제거 후 재설정
    """
    global _CONFIGURED
    with _LOCK:
        log_dir = Path(cfg.log_dir).resolve()
        log_dir.mkdir(parents=True, exist_ok=True)
        file_path = log_dir / cfg.file_name

        if cfg.json_format:
            fmt: logging.Formatter = JsonFormatter()
        else:
            fmt = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )

        # Root logger: 기존 handler 제거 후 NullHandler만
        root = logging.getLogger()
        for h in list(root.handlers):
            root.removeHandler(h)
        root.addHandler(logging.NullHandler())
        # 외부 라이브러리(paddle/glog 등)는 WARNING 이상만
        root.setLevel(logging.WARNING)

        # Namespace logger 설정
        ns_logger = logging.getLogger(cfg.namespace_logger)
        for h in list(ns_logger.handlers):
            ns_logger.removeHandler(h)

        stream = logging.StreamHandler()
        stream.setFormatter(fmt)
        ns_logger.addHandler(stream)

        from logging.handlers import RotatingFileHandler

        file_h = RotatingFileHandler(
            file_path,
            maxBytes=cfg.max_bytes,
            backupCount=cfg.backup_count,
            encoding="utf-8",
        )
        file_h.setFormatter(fmt)
        ns_logger.addHandler(file_h)

        ns_logger.setLevel(cfg.level)
        ns_logger.propagate = False

        _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """얇은 getLogger 래퍼. 미래에 StructuredLogger 주입 여지."""
    return logging.getLogger(name)


def is_configured() -> bool:
    return _CONFIGURED
