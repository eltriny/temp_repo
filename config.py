"""
설정 관리 모듈

dataclass 기반의 타입 안전한 설정 관리를 제공합니다.
환경변수, YAML 파일, 런타임 오버라이드를 지원합니다.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import yaml


class FrameSkipMode(Enum):
    """프레임 스킵 모드 정의"""

    FIXED = "fixed"  # 고정 간격
    ADAPTIVE = "adaptive"  # 적응형 (변화 감지 기반)


@dataclass(frozen=True)
class ROIConfig:
    """관심 영역(ROI) 설정

    좌표는 정규화된 값(0.0 ~ 1.0) 또는 픽셀 값으로 지정 가능.
    정규화 좌표 사용 시 다양한 해상도에서 일관된 ROI 적용.

    Attributes:
        name: ROI 식별 이름
        x: 좌상단 X 좌표
        y: 좌상단 Y 좌표
        width: ROI 너비
        height: ROI 높이
        normalized: True면 0.0~1.0 정규화 좌표, False면 픽셀 좌표
    """

    name: str
    x: float
    y: float
    width: float
    height: float
    normalized: bool = True

    def __post_init__(self) -> None:
        """좌표 값 검증"""
        if self.normalized:
            for attr_name, value in [
                ("x", self.x),
                ("y", self.y),
                ("width", self.width),
                ("height", self.height),
            ]:
                if not 0.0 <= value <= 1.0:
                    raise ValueError(
                        f"정규화 좌표 {attr_name}은 0.0~1.0 범위여야 합니다: {value}"
                    )
        else:
            for attr_name, value in [
                ("x", self.x),
                ("y", self.y),
                ("width", self.width),
                ("height", self.height),
            ]:
                if value < 0:
                    raise ValueError(
                        f"픽셀 좌표 {attr_name}은 0 이상이어야 합니다: {value}"
                    )

    def to_pixel_coords(
        self,
        frame_width: int,
        frame_height: int,
    ) -> tuple[int, int, int, int]:
        """정규화 좌표를 픽셀 좌표로 변환

        Args:
            frame_width: 프레임 너비 (픽셀)
            frame_height: 프레임 높이 (픽셀)

        Returns:
            (x, y, width, height) 픽셀 좌표 튜플
        """
        if self.normalized:
            return (
                int(self.x * frame_width),
                int(self.y * frame_height),
                int(self.width * frame_width),
                int(self.height * frame_height),
            )
        return (
            int(self.x),
            int(self.y),
            int(self.width),
            int(self.height),
        )


@dataclass
class ProcessingConfig:
    """비디오 처리 설정

    프레임 추출 및 분석 관련 설정을 관리합니다.

    Attributes:
        frame_skip_mode: 프레임 스킵 모드 (FIXED/ADAPTIVE)
        default_interval_sec: 기본 프레임 추출 간격 (초)
        change_detection_interval_sec: 변화 감지 시 프레임 추출 간격 (초)
        change_threshold: 프레임 변화 감지 임계값 (0.0~1.0)
        batch_size: 배치 처리 시 프레임 수
        max_workers: 병렬 처리 워커 수 (0=자동, CPU 코어의 75%)
        resize_width: 처리용 리사이즈 너비 (0=원본 유지)
        resize_height: 처리용 리사이즈 높이 (0=원본 유지)
        
        # 병렬 ROI 처리 설정 (v2.0)
        parallel_timeout_per_roi: ROI당 타임아웃 (초)
        use_gpu: GPU 가속 사용 여부
    """

    frame_skip_mode: FrameSkipMode = FrameSkipMode.ADAPTIVE
    default_interval_sec: float = 1.0
    change_detection_interval_sec: float = 0.17  # ~6fps
    change_threshold: float = 0.05
    batch_size: int = 32
    max_workers: int = 0
    resize_width: int = 0
    resize_height: int = 0
    
    # 병렬 ROI 처리 설정
    parallel_timeout_per_roi: float = 10.0
    use_gpu: bool = False

    def __post_init__(self) -> None:
        """설정 값 검증"""
        if self.default_interval_sec <= 0:
            raise ValueError(
                f"default_interval_sec는 양수여야 합니다: {self.default_interval_sec}"
            )
        if self.change_detection_interval_sec <= 0:
            raise ValueError(
                f"change_detection_interval_sec는 양수여야 합니다: "
                f"{self.change_detection_interval_sec}"
            )
        if not 0.0 <= self.change_threshold <= 1.0:
            raise ValueError(
                f"change_threshold는 0.0~1.0 범위여야 합니다: {self.change_threshold}"
            )
        if self.batch_size < 1:
            raise ValueError(f"batch_size는 1 이상이어야 합니다: {self.batch_size}")
        if self.parallel_timeout_per_roi <= 0:
            raise ValueError(
                f"parallel_timeout_per_roi는 양수여야 합니다: {self.parallel_timeout_per_roi}"
            )

    @property
    def effective_max_workers(self) -> int:
        """실제 사용할 워커 수 반환 (0이면 CPU 코어의 75%)"""
        if self.max_workers > 0:
            return self.max_workers
        cpu_count = os.cpu_count() or 4
        return max(1, int(cpu_count * 0.75))


@dataclass
class StorageConfig:
    """저장소 설정

    분석 결과 저장 관련 설정을 관리합니다.

    Attributes:
        db_host: Oracle 데이터베이스 호스트
        db_port: Oracle 데이터베이스 포트
        db_service_name: Oracle 서비스 이름
        db_user: Oracle 사용자명
        db_password: Oracle 비밀번호
        db_dsn: Oracle DSN 문자열 (직접 지정 시 host/port/service_name 대신 사용)
        output_dir: 결과 출력 디렉토리
        save_frames: 분석된 프레임 이미지 저장 여부
        frame_format: 저장 프레임 이미지 형식
        frame_quality: JPEG 품질 (1-100)
    """

    db_host: str = "localhost"
    db_port: int = 1521
    db_service_name: str = "ORCL"
    db_user: str = ""
    db_password: str = ""
    db_dsn: str = ""
    output_dir: Path = field(default_factory=lambda: Path("data/output"))
    save_frames: bool = False
    frame_format: str = "jpg"
    frame_quality: int = 85

    def __post_init__(self) -> None:
        """경로를 Path 객체로 변환 및 검증"""
        if isinstance(self.output_dir, str):
            object.__setattr__(self, "output_dir", Path(self.output_dir))

        if not 1 <= self.frame_quality <= 100:
            raise ValueError(
                f"frame_quality는 1~100 범위여야 합니다: {self.frame_quality}"
            )

        valid_formats = {"jpg", "jpeg", "png", "webp"}
        if self.frame_format.lower() not in valid_formats:
            raise ValueError(
                f"frame_format은 {valid_formats} 중 하나여야 합니다: {self.frame_format}"
            )

    def ensure_directories(self) -> None:
        """필요한 디렉토리 생성"""
        self.output_dir.mkdir(parents=True, exist_ok=True)


@dataclass
class Config:
    """전체 시스템 설정

    모든 설정을 통합 관리하며 환경변수/YAML 로딩을 지원합니다.

    Attributes:
        video_path: 처리할 비디오 파일 경로
        processing: 비디오 처리 설정
        storage: 저장소 설정
        rois: 관심 영역 목록
        debug: 디버그 모드 활성화
        log_level: 로깅 레벨
    """

    video_path: Path | None = None
    processing: ProcessingConfig = field(default_factory=ProcessingConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    rois: list[ROIConfig] = field(default_factory=list)
    debug: bool = False
    log_level: str = "INFO"

    def __post_init__(self) -> None:
        """설정 후처리"""
        if isinstance(self.video_path, str):
            object.__setattr__(self, "video_path", Path(self.video_path))

        valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if self.log_level.upper() not in valid_levels:
            raise ValueError(
                f"log_level은 {valid_levels} 중 하나여야 합니다: {self.log_level}"
            )

    @classmethod
    def from_yaml(cls, config_path: Path | str) -> Config:
        """YAML 파일에서 설정 로드

        Args:
            config_path: YAML 설정 파일 경로

        Returns:
            로드된 Config 인스턴스

        Raises:
            FileNotFoundError: 설정 파일이 없는 경우
            yaml.YAMLError: YAML 파싱 실패
        """
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"설정 파일을 찾을 수 없습니다: {config_path}")

        with open(config_path, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        return cls._from_dict(data)

    @classmethod
    def from_env(cls) -> Config:
        """환경변수에서 설정 로드

        환경변수 형식:
        - VIDEO_PATH: 비디오 파일 경로
        - DB_HOST: Oracle 호스트
        - DB_PORT: Oracle 포트
        - DB_SERVICE_NAME: Oracle 서비스 이름
        - DB_USER: Oracle 사용자명
        - DB_PASSWORD: Oracle 비밀번호
        - DB_DSN: Oracle DSN 문자열
        - OUTPUT_DIR: 출력 디렉토리
        - FRAME_INTERVAL: 기본 프레임 간격 (초)
        - DEBUG: 디버그 모드 (true/false)
        - LOG_LEVEL: 로깅 레벨

        Returns:
            환경변수 기반 Config 인스턴스
        """
        video_path = os.getenv("VIDEO_PATH")

        processing_kwargs: dict[str, Any] = {}
        if interval := os.getenv("FRAME_INTERVAL"):
            processing_kwargs["default_interval_sec"] = float(interval)

        storage_kwargs: dict[str, Any] = {}
        if db_host := os.getenv("DB_HOST"):
            storage_kwargs["db_host"] = db_host
        if db_port := os.getenv("DB_PORT"):
            storage_kwargs["db_port"] = int(db_port)
        if db_service_name := os.getenv("DB_SERVICE_NAME"):
            storage_kwargs["db_service_name"] = db_service_name
        if db_user := os.getenv("DB_USER"):
            storage_kwargs["db_user"] = db_user
        if db_password := os.getenv("DB_PASSWORD"):
            storage_kwargs["db_password"] = db_password
        if db_dsn := os.getenv("DB_DSN"):
            storage_kwargs["db_dsn"] = db_dsn
        if output_dir := os.getenv("OUTPUT_DIR"):
            storage_kwargs["output_dir"] = Path(output_dir)

        return cls(
            video_path=Path(video_path) if video_path else None,
            processing=ProcessingConfig(**processing_kwargs),
            storage=StorageConfig(**storage_kwargs),
            debug=os.getenv("DEBUG", "").lower() == "true",
            log_level=os.getenv("LOG_LEVEL", "INFO"),
        )

    @classmethod
    def _from_dict(cls, data: dict[str, Any]) -> Config:
        """딕셔너리에서 Config 생성"""
        processing_data = data.get("processing", {})
        if "frame_skip_mode" in processing_data:
            processing_data["frame_skip_mode"] = FrameSkipMode(
                processing_data["frame_skip_mode"]
            )

        storage_data = data.get("storage", {})
        if "output_dir" in storage_data:
            storage_data["output_dir"] = Path(storage_data["output_dir"])
        if "db_port" in storage_data:
            storage_data["db_port"] = int(storage_data["db_port"])

        rois_data = data.get("rois", [])
        rois = [ROIConfig(**roi) for roi in rois_data]

        return cls(
            video_path=Path(data["video_path"]) if data.get("video_path") else None,
            processing=ProcessingConfig(**processing_data),
            storage=StorageConfig(**storage_data),
            rois=rois,
            debug=data.get("debug", False),
            log_level=data.get("log_level", "INFO"),
        )

    def to_dict(self) -> dict[str, Any]:
        """설정을 딕셔너리로 변환"""
        return {
            "video_path": str(self.video_path) if self.video_path else None,
            "processing": {
                "frame_skip_mode": self.processing.frame_skip_mode.value,
                "default_interval_sec": self.processing.default_interval_sec,
                "change_detection_interval_sec": (
                    self.processing.change_detection_interval_sec
                ),
                "change_threshold": self.processing.change_threshold,
                "batch_size": self.processing.batch_size,
                "max_workers": self.processing.max_workers,
                "resize_width": self.processing.resize_width,
                "resize_height": self.processing.resize_height,
            },
            "storage": {
                "db_host": self.storage.db_host,
                "db_port": self.storage.db_port,
                "db_service_name": self.storage.db_service_name,
                "db_user": self.storage.db_user,
                "db_password": self.storage.db_password,
                "db_dsn": self.storage.db_dsn,
                "output_dir": str(self.storage.output_dir),
                "save_frames": self.storage.save_frames,
                "frame_format": self.storage.frame_format,
                "frame_quality": self.storage.frame_quality,
            },
            "rois": [
                {
                    "name": roi.name,
                    "x": roi.x,
                    "y": roi.y,
                    "width": roi.width,
                    "height": roi.height,
                    "normalized": roi.normalized,
                }
                for roi in self.rois
            ],
            "debug": self.debug,
            "log_level": self.log_level,
        }

    def save_yaml(self, config_path: Path | str) -> None:
        """설정을 YAML 파일로 저장

        Args:
            config_path: 저장할 YAML 파일 경로
        """
        config_path = Path(config_path)
        config_path.parent.mkdir(parents=True, exist_ok=True)

        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(
                self.to_dict(),
                f,
                default_flow_style=False,
                allow_unicode=True,
                sort_keys=False,
            )

    def with_overrides(self, **kwargs: Any) -> Config:
        """오버라이드된 새 Config 인스턴스 생성

        기존 설정을 복사하고 지정된 값만 덮어씁니다.

        Args:
            **kwargs: 덮어쓸 설정 값들

        Returns:
            새 Config 인스턴스
        """
        current = self.to_dict()

        for key, value in kwargs.items():
            if "." in key:
                # 중첩 키 처리: "processing.batch_size" -> processing["batch_size"]
                parts = key.split(".")
                target = current
                for part in parts[:-1]:
                    target = target[part]
                target[parts[-1]] = value
            else:
                current[key] = value

        return self._from_dict(current)


@dataclass
class OCREnhancementConfig:
    """OCR 인식률 개선 설정

    OCR 파이프라인의 다양한 개선 기능을 제어하는 설정입니다.

    Attributes:
        # 앙상블 설정
        enable_ensemble: 다중 OCR 앙상블 활성화
        ensemble_engines: 사용할 OCR 엔진 목록 ("paddle", "easyocr", "tesseract")
        ensemble_strategy: 투표 전략 ("weighted", "majority", "confidence_max")
        
        # 후처리 설정
        enable_text_correction: 유사 문자 교정 활성화
        enable_context_validation: 컨텍스트 기반 검증 활성화
        
        # 전처리 설정
        enable_blur_detection: 블러 감지 및 적응형 선명화 활성화
        enable_adaptive_preprocessing: 품질 기반 적응형 전처리 활성화
        
        # 재시도 설정
        min_confidence_threshold: 재시도를 위한 최소 신뢰도 임계값
        max_retries: 최대 재시도 횟수
        retry_with_enhanced_preprocessing: 재시도 시 강화된 전처리 사용

    Example:
        >>> config = OCREnhancementConfig(
        ...     enable_ensemble=True,
        ...     ensemble_engines=("paddle", "easyocr"),
        ...     min_confidence_threshold=0.8,
        ... )
    """

    # ========================================
    # 앙상블 설정
    # ========================================
    enable_ensemble: bool = True
    ensemble_engines: tuple[str, ...] = ("paddle", "easyocr")
    ensemble_strategy: str = "weighted"
    ensemble_parallel: bool = True
    ensemble_timeout: float = 30.0

    # ========================================
    # 후처리 설정 (텍스트 교정)
    # ========================================
    enable_text_correction: bool = True
    enable_context_validation: bool = True
    correction_confidence_penalty: float = 0.01  # 교정당 신뢰도 페널티

    # 공백 정규화
    enable_space_normalize: bool = True

    # 유사 문자 교정 (0↔O, 1↔l, !→1 등)
    enable_similar_char_correction: bool = True

    # 라벨 내 공백 보존 (Temperature: 25.5의 콜론 뒤 공백)
    preserve_spaces_in_labels: bool = True

    # 단위 정규화 (0C→°C, HZ→Hz 등)
    enable_unit_normalize: bool = True

    # 최대 신뢰도 페널티
    max_correction_penalty: float = 0.15

    # 패턴 일치 시 신뢰도 보너스
    pattern_match_bonus: float = 0.03

    # ========================================
    # 전처리 설정
    # ========================================
    enable_blur_detection: bool = True
    enable_adaptive_preprocessing: bool = True
    enable_display_type_detection: bool = False  # 자동 디스플레이 유형 감지

    # ========================================
    # 재시도 설정
    # ========================================
    min_confidence_threshold: float = 0.7
    max_retries: int = 2
    retry_with_enhanced_preprocessing: bool = True
    retry_preprocessing_levels: tuple[str, ...] = ("standard", "enhanced", "aggressive")

    # ========================================
    # 신뢰도 조정
    # ========================================
    ensemble_agreement_bonus: float = 0.1  # 여러 엔진 동의 시 보너스
    high_quality_threshold: float = 0.8  # 고품질 이미지 판정 임계값

    def __post_init__(self) -> None:
        """설정 값 검증"""
        if not 0.0 <= self.min_confidence_threshold <= 1.0:
            raise ValueError(
                f"min_confidence_threshold는 0.0~1.0 범위여야 합니다: "
                f"{self.min_confidence_threshold}"
            )
        if self.max_retries < 0:
            raise ValueError(f"max_retries는 0 이상이어야 합니다: {self.max_retries}")
        
        valid_strategies = {"weighted", "majority", "confidence_max", "consensus"}
        if self.ensemble_strategy not in valid_strategies:
            raise ValueError(
                f"ensemble_strategy는 {valid_strategies} 중 하나여야 합니다: "
                f"{self.ensemble_strategy}"
            )
        
        valid_engines = {"paddle", "easyocr", "tesseract"}
        for engine in self.ensemble_engines:
            if engine.lower() not in valid_engines:
                raise ValueError(
                    f"ensemble_engines에 유효하지 않은 엔진: {engine}. "
                    f"허용: {valid_engines}"
                )

    def to_dict(self) -> dict[str, Any]:
        """설정을 딕셔너리로 변환"""
        return {
            "enable_ensemble": self.enable_ensemble,
            "ensemble_engines": list(self.ensemble_engines),
            "ensemble_strategy": self.ensemble_strategy,
            "ensemble_parallel": self.ensemble_parallel,
            "ensemble_timeout": self.ensemble_timeout,
            "enable_text_correction": self.enable_text_correction,
            "enable_context_validation": self.enable_context_validation,
            "correction_confidence_penalty": self.correction_confidence_penalty,
            "enable_space_normalize": self.enable_space_normalize,
            "enable_similar_char_correction": self.enable_similar_char_correction,
            "preserve_spaces_in_labels": self.preserve_spaces_in_labels,
            "enable_unit_normalize": self.enable_unit_normalize,
            "max_correction_penalty": self.max_correction_penalty,
            "pattern_match_bonus": self.pattern_match_bonus,
            "enable_blur_detection": self.enable_blur_detection,
            "enable_adaptive_preprocessing": self.enable_adaptive_preprocessing,
            "enable_display_type_detection": self.enable_display_type_detection,
            "min_confidence_threshold": self.min_confidence_threshold,
            "max_retries": self.max_retries,
            "retry_with_enhanced_preprocessing": self.retry_with_enhanced_preprocessing,
            "retry_preprocessing_levels": list(self.retry_preprocessing_levels),
            "ensemble_agreement_bonus": self.ensemble_agreement_bonus,
            "high_quality_threshold": self.high_quality_threshold,
        }

    def to_text_correction_config(self) -> "TextCorrectionConfig":
        """TextCorrectionConfig 객체로 변환합니다.

        Returns:
            TextCorrectionConfig 인스턴스
        """
        from src.ocr.text_corrector import TextCorrectionConfig

        return TextCorrectionConfig(
            enable_space_normalize=self.enable_space_normalize,
            enable_similar_char_correction=self.enable_similar_char_correction,
            enable_context_detection=self.enable_context_validation,
            enable_pattern_validation=True,
            enable_unit_normalize=self.enable_unit_normalize,
            confidence_penalty_per_correction=self.correction_confidence_penalty,
            max_confidence_penalty=self.max_correction_penalty,
            pattern_match_bonus=self.pattern_match_bonus,
            preserve_spaces_in_labels=self.preserve_spaces_in_labels,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> OCREnhancementConfig:
        """딕셔너리에서 설정 생성"""
        # 튜플로 변환이 필요한 필드들
        if "ensemble_engines" in data and isinstance(data["ensemble_engines"], list):
            data["ensemble_engines"] = tuple(data["ensemble_engines"])
        if "retry_preprocessing_levels" in data and isinstance(
            data["retry_preprocessing_levels"], list
        ):
            data["retry_preprocessing_levels"] = tuple(data["retry_preprocessing_levels"])
        return cls(**data)


def create_default_ocr_enhancement_config() -> OCREnhancementConfig:
    """기본 OCR 개선 설정 생성

    균형 잡힌 기본 설정을 반환합니다.
    """
    return OCREnhancementConfig()


def create_accuracy_focused_config() -> OCREnhancementConfig:
    """정확도 우선 OCR 개선 설정

    속도보다 정확도를 우선시하는 설정입니다.
    오프라인 분석에 적합합니다.
    """
    return OCREnhancementConfig(
        enable_ensemble=True,
        ensemble_engines=("paddle", "easyocr", "tesseract"),
        ensemble_strategy="weighted",
        enable_text_correction=True,
        enable_context_validation=True,
        enable_blur_detection=True,
        enable_adaptive_preprocessing=True,
        min_confidence_threshold=0.8,
        max_retries=3,
        retry_with_enhanced_preprocessing=True,
    )


def create_speed_focused_config() -> OCREnhancementConfig:
    """속도 우선 OCR 개선 설정

    정확도보다 속도를 우선시하는 설정입니다.
    실시간 분석에 적합합니다.
    """
    return OCREnhancementConfig(
        enable_ensemble=False,
        ensemble_engines=("paddle",),
        enable_text_correction=True,
        enable_context_validation=False,
        enable_blur_detection=False,
        enable_adaptive_preprocessing=False,
        min_confidence_threshold=0.6,
        max_retries=0,
        retry_with_enhanced_preprocessing=False,
    )
