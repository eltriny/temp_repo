"""
산업용 비디오 모니터링 시스템

장시간 비디오(1시간+)를 효율적으로 처리하여 객체 감지 및 OCR 분석을 수행하는 시스템.

주요 기능:
- Generator 패턴 기반 메모리 효율적 비디오 처리
- 적응형 프레임 스킵 (기본 1초, 변화 감지 시 0.17초)
- ROI 기반 관심 영역 분석
- SQLite 기반 결과 저장
"""

from .config import Config, ROIConfig, ProcessingConfig, StorageConfig

__version__ = "0.1.0"
__author__ = "Video Detection Team"

__all__ = [
    "Config",
    "ROIConfig",
    "ProcessingConfig",
    "StorageConfig",
]
