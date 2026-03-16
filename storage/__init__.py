"""
Storage Module for Industrial Video Monitoring System.

This module provides database and file storage management for video
analysis sessions, ROI definitions, change detection events, and
captured frame images.

Components:
    - DatabaseManager: Oracle database operations with CRUD support
    - CaptureManager: Image file storage with compression options

Usage:
    from storage import DatabaseManager, CaptureManager, CaptureConfig
    from storage import Session, ROIDefinition, ChangeEvent
    from storage import ROIType, ImageFormat

    # Database operations
    with DatabaseManager(user="app", password="pw", dsn="localhost:1521/ORCL") as db:
        session = db.create_session(SessionCreate(...))
        db.create_roi(ROICreate(...))

    # Capture operations
    config = CaptureConfig(base_directory=Path("./captures"))
    with CaptureManager(config) as manager:
        result = manager.save_capture(image, session_id=1, roi_name="display")
"""

from .capture_manager import (
    CaptureConfig,
    CaptureError,
    CaptureFormatError,
    CaptureIOError,
    CaptureManager,
    CaptureResult,
    CompressionLevel,
    ImageFormat,
    ImageSaver,
    PILImageSaver,
)
from .database import (
    ChangeEvent,
    ChangeEventCreate,
    DatabaseConnectionError,
    DatabaseError,
    DatabaseIntegrityError,
    DatabaseManager,
    ROICreate,
    ROIDefinition,
    ROIType,
    Session,
    SessionCreate,
    VideoFile,
    VideoFileCreate,
    VideoFileStatus,
)

__all__ = [
    # Database Manager
    "DatabaseManager",
    "DatabaseError",
    "DatabaseConnectionError",
    "DatabaseIntegrityError",
    # Database Models
    "Session",
    "SessionCreate",
    "ROIDefinition",
    "ROICreate",
    "ChangeEvent",
    "ChangeEventCreate",
    "ROIType",
    # Video File Models
    "VideoFile",
    "VideoFileCreate",
    "VideoFileStatus",
    # Capture Manager
    "CaptureManager",
    "CaptureConfig",
    "CaptureResult",
    "CaptureError",
    "CaptureIOError",
    "CaptureFormatError",
    "ImageFormat",
    "CompressionLevel",
    "ImageSaver",
    "PILImageSaver",
]

__version__ = "1.0.0"
