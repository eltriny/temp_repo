"""
설정 관리 모듈

dataclass 기반의 타입 안전한 설정 관리를 제공합니다.
환경변수, YAML 파일, 런타임 오버라이드를 지원합니다.
"""

from __future__ import annotations

import os
import re
import warnings
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

if TYPE_CHECKING:
    # B11: TextCorrectionConfig 는 to_text_correction_config() 의 forward
    # reference 로만 사용된다. 런타임 의존을 만들지 않기 위해 TYPE_CHECKING 가드.
    from src.ocr.text_corrector import TextCorrectionConfig  # noqa: F401


class MissingEnvVarError(RuntimeError):
    """YAML 설정에서 참조하는 환경 변수를 찾을 수 없을 때 발생하는 예외."""


class Secret:
    """민감 정보(비밀번호 등) 래퍼.

    ``repr``/``str``/로깅 출력에서 실제 값이 노출되지 않도록 마스킹합니다.
    실제 값을 꺼내려면 :meth:`get_secret_value`를 명시적으로 호출해야
    합니다. 이는 ``config.db_password``를 실수로 로그에 찍거나 YAML에
    평문으로 덤프하는 사고를 방지합니다.
    """

    __slots__ = ("_value",)

    def __init__(self, value: str = "") -> None:
        self._value = value or ""

    def get_secret_value(self) -> str:
        return self._value

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return "Secret('***')"

    def __str__(self) -> str:  # pragma: no cover - trivial
        return "***"

    def __bool__(self) -> bool:
        return bool(self._value)

    def __eq__(self, other: object) -> bool:
        """B8: Secret 끼리만 비교 허용, 평문 str 비교는 금지.

        타이밍 공격 방지를 위해 ``hmac.compare_digest`` 사용.
        str 비교는 ``NotImplemented`` 를 반환하여 실수로 평문 상수와 비교하는
        코드 경로를 차단한다 (e.g. ``config.db_password == "admin"``).
        """
        if isinstance(other, Secret):
            import hmac

            return hmac.compare_digest(self._value, other._value)
        return NotImplemented

    def __hash__(self) -> int:
        """B8: 평문 기반 hash 를 피하기 위해 객체 id 를 사용.

        Secret 인스턴스끼리의 ``__eq__`` 가 같은 평문에 대해 True 를 반환하므로
        hash/eq 계약상 완전하지는 않지만, Secret 을 dict/set key 로 쓰는 경우는
        없으므로 실무상 무해하다. 반대로 id 기반 hash 는 평문이 레지스트리/해시
        테이블에 인덱스 키로 저장되는 것을 원천 차단한다.
        """
        return object.__hash__(self)


_ENV_VAR_PATTERN = re.compile(r"\$\{([A-Z_][A-Z0-9_]*)\}")


def _expand_env_vars(value: Any) -> Any:
    """재귀적으로 문자열 내 ``${ENV_VAR}`` 참조를 치환합니다.

    - 문자열 내 패턴 발견 시 ``os.environ``에서 값을 읽어 치환합니다.
    - 대상 환경 변수가 정의되지 않으면 :class:`MissingEnvVarError`를
      발생시킵니다 (silent empty 금지).
    - dict/list는 재귀적으로 처리됩니다.
    """
    if isinstance(value, str):
        def replacer(match: re.Match[str]) -> str:
            name = match.group(1)
            env = os.environ.get(name)
            if env is None:
                raise MissingEnvVarError(
                    f"YAML 설정이 참조하는 환경 변수 '{name}'가 정의되지 않았습니다"
                )
            return env

        return _ENV_VAR_PATTERN.sub(replacer, value)
    if isinstance(value, dict):
        return {k: _expand_env_vars(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand_env_vars(v) for v in value]
    return value


def _mask_dsn_password(dsn: str) -> str:
    """DSN 내 비밀번호를 ``***``로 마스킹합니다.

    지원 형식:
        1. URL 형식: ``postgresql://user:pass@host:5432/db``
        2. keyword/value 형식: ``host=x password=y dbname=z``

    B9: 기존 구현은 URL 형식만 처리하여 keyword/value DSN 의 password 가
    그대로 노출되었다. URL 파싱으로 fallback 후 남은 문자열에 대해
    ``password=...`` 패턴도 추가로 치환한다.
    """
    if not dsn:
        return dsn

    masked = dsn
    if "://" in masked:
        masked = re.sub(r"(://[^:/@]+:)([^@]+)(@)", r"\1***\3", masked)

    # keyword/value 형식 (URL 형식에도 쿼리스트링으로 섞여 올 수 있음)
    if "password=" in masked:
        # password=... 는 공백 또는 문자열 끝까지. 따옴표로 감싸진 경우도 처리.
        masked = re.sub(
            r"password\s*=\s*(?:'[^']*'|\"[^\"]*\"|\S+)",
            "password=***",
            masked,
        )

    return masked


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
    """저장소 설정 (PostgreSQL).

    Phase 1-C에서 Oracle에서 PostgreSQL로 전환되었습니다.
    ``db_service_name`` 필드는 ``db_name``의 하위 호환 별칭으로만 유지되며,
    지정된 경우 ``__post_init__``에서 ``db_name``으로 마이그레이션됩니다.

    Attributes:
        db_host: PostgreSQL 호스트
        db_port: PostgreSQL 포트 (기본 5432)
        db_name: 데이터베이스 이름
        db_user: 사용자명
        db_password: 비밀번호 (Secret 래퍼)
        db_dsn: PostgreSQL DSN URI (지정 시 host/port/name보다 우선)
        db_schema: 스키마 (기본 "public")
        db_pool_min: 풀 최소 커넥션 수
        db_pool_max: 풀 최대 커넥션 수
        db_service_name: (deprecated) Oracle 서비스 이름 호환 별칭
        output_dir: 결과 출력 디렉토리
        save_frames: 분석된 프레임 이미지 저장 여부
        frame_format: 저장 프레임 이미지 형식
        frame_quality: JPEG 품질 (1-100)
        min_free_disk_mb: 최소 여유 디스크 (MB)
        disk_check_action: 디스크 부족 시 동작 ("warn"|"skip"|"abort")
    """

    db_host: str = "localhost"
    db_port: int = 5432
    db_name: str = "video_detection"
    db_user: str = ""
    db_password: Secret = field(default_factory=Secret)
    db_dsn: str = ""
    db_schema: str = "public"
    db_pool_min: int = 2
    db_pool_max: int = 10

    # Deprecated: Oracle 시절의 service_name 별칭. 한 minor 버전 유지.
    db_service_name: str = ""

    output_dir: Path = field(default_factory=lambda: Path("data/output"))
    save_frames: bool = False
    frame_format: str = "jpg"
    frame_quality: int = 85

    # Phase 1-E에서 사용될 디스크 체크 설정
    min_free_disk_mb: int = 2048
    disk_check_action: str = "warn"

    def __post_init__(self) -> None:
        """경로 변환 + 값 검증 + 하위 호환 처리."""
        # 경로 정규화
        if isinstance(self.output_dir, str):
            object.__setattr__(self, "output_dir", Path(self.output_dir))

        # 비밀번호가 평문 문자열로 전달된 경우 Secret으로 래핑
        if isinstance(self.db_password, str):
            object.__setattr__(self, "db_password", Secret(self.db_password))
        elif self.db_password is None:
            object.__setattr__(self, "db_password", Secret(""))

        # db_service_name deprecation: db_name이 기본값이면 자동 할당
        if self.db_service_name and self.db_name == "video_detection":
            warnings.warn(
                "StorageConfig.db_service_name은 deprecated입니다. "
                "대신 db_name을 사용하세요. db_name에 db_service_name 값을 "
                "자동 할당합니다.",
                DeprecationWarning,
                stacklevel=2,
            )
            object.__setattr__(self, "db_name", self.db_service_name)

        if not 1 <= self.frame_quality <= 100:
            raise ValueError(
                f"frame_quality는 1~100 범위여야 합니다: {self.frame_quality}"
            )

        valid_formats = {"jpg", "jpeg", "png", "webp"}
        if self.frame_format.lower() not in valid_formats:
            raise ValueError(
                f"frame_format은 {valid_formats} 중 하나여야 합니다: {self.frame_format}"
            )

        valid_actions = {"warn", "skip", "abort"}
        if self.disk_check_action not in valid_actions:
            raise ValueError(
                f"disk_check_action은 {valid_actions} 중 하나여야 합니다: "
                f"{self.disk_check_action}"
            )

        if self.db_pool_min < 1:
            raise ValueError(f"db_pool_min은 1 이상이어야 합니다: {self.db_pool_min}")
        if self.db_pool_max < self.db_pool_min:
            raise ValueError(
                f"db_pool_max({self.db_pool_max})는 "
                f"db_pool_min({self.db_pool_min}) 이상이어야 합니다"
            )

    def ensure_directories(self) -> None:
        """필요한 디렉토리 생성"""
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def __repr__(self) -> str:
        """B6: 비밀번호/DSN 마스킹 repr.

        dataclass 기본 repr 은 ``db_dsn='postgresql://u:pass@h/d'`` 를 그대로
        노출하여 ``logger.info(config.storage)`` 한 줄로 비밀번호가 누수된다.
        커스텀 repr 로 db_dsn 을 마스킹하고 db_password 는 Secret 레이어를 거쳐
        ``***`` 로 표시한다.
        """
        masked_dsn = _mask_dsn_password(self.db_dsn) if self.db_dsn else ""
        return (
            f"StorageConfig(db_host={self.db_host!r}, db_port={self.db_port}, "
            f"db_name={self.db_name!r}, db_user={self.db_user!r}, "
            f"db_password={self.db_password!r}, db_dsn={masked_dsn!r}, "
            f"db_schema={self.db_schema!r}, db_pool_min={self.db_pool_min}, "
            f"db_pool_max={self.db_pool_max}, output_dir={self.output_dir!r}, "
            f"save_frames={self.save_frames}, frame_format={self.frame_format!r}, "
            f"frame_quality={self.frame_quality}, "
            f"min_free_disk_mb={self.min_free_disk_mb}, "
            f"disk_check_action={self.disk_check_action!r})"
        )


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
        """환경변수에서 설정 로드.

        ``VIDEO_DETECTION_*`` prefix를 우선 사용하고, 없으면 과거 ``DB_*``/
        기타 레거시 이름을 fallback으로 사용합니다.

        지원 환경 변수 (PostgreSQL):
            - VIDEO_DETECTION_DB_HOST / DB_HOST
            - VIDEO_DETECTION_DB_PORT / DB_PORT
            - VIDEO_DETECTION_DB_NAME / DB_NAME (fallback: DB_SERVICE_NAME)
            - VIDEO_DETECTION_DB_USER / DB_USER
            - VIDEO_DETECTION_DB_PASSWORD / DB_PASSWORD
            - VIDEO_DETECTION_DB_DSN / DB_DSN
            - VIDEO_DETECTION_DB_SCHEMA / DB_SCHEMA
            - VIDEO_DETECTION_DB_POOL_MIN / DB_POOL_MIN
            - VIDEO_DETECTION_DB_POOL_MAX / DB_POOL_MAX

        Returns:
            환경변수 기반 Config 인스턴스
        """
        # .env 파일 자동 로드 (python-dotenv 설치 시)
        try:
            from dotenv import load_dotenv

            load_dotenv()
        except ImportError:
            pass

        def _env(primary: str, *fallbacks: str) -> str | None:
            """VIDEO_DETECTION_* 우선, 없으면 fallback 탐색."""
            value = os.getenv(primary)
            if value is not None:
                return value
            for name in fallbacks:
                value = os.getenv(name)
                if value is not None:
                    return value
            return None

        video_path = _env("VIDEO_DETECTION_VIDEO_PATH", "VIDEO_PATH")

        processing_kwargs: dict[str, Any] = {}
        if interval := _env("VIDEO_DETECTION_FRAME_INTERVAL", "FRAME_INTERVAL"):
            processing_kwargs["default_interval_sec"] = float(interval)

        storage_kwargs: dict[str, Any] = {}
        if db_host := _env("VIDEO_DETECTION_DB_HOST", "DB_HOST"):
            storage_kwargs["db_host"] = db_host
        if db_port := _env("VIDEO_DETECTION_DB_PORT", "DB_PORT"):
            storage_kwargs["db_port"] = int(db_port)
        if db_name := _env(
            "VIDEO_DETECTION_DB_NAME", "DB_NAME", "DB_SERVICE_NAME"
        ):
            storage_kwargs["db_name"] = db_name
        if db_user := _env("VIDEO_DETECTION_DB_USER", "DB_USER"):
            storage_kwargs["db_user"] = db_user
        db_password_value = _env(
            "VIDEO_DETECTION_DB_PASSWORD", "DB_PASSWORD"
        )
        if db_password_value is not None:
            storage_kwargs["db_password"] = Secret(db_password_value)
        if db_dsn := _env("VIDEO_DETECTION_DB_DSN", "DB_DSN"):
            storage_kwargs["db_dsn"] = db_dsn
        if db_schema := _env("VIDEO_DETECTION_DB_SCHEMA", "DB_SCHEMA"):
            storage_kwargs["db_schema"] = db_schema
        if db_pool_min := _env("VIDEO_DETECTION_DB_POOL_MIN", "DB_POOL_MIN"):
            storage_kwargs["db_pool_min"] = int(db_pool_min)
        if db_pool_max := _env("VIDEO_DETECTION_DB_POOL_MAX", "DB_POOL_MAX"):
            storage_kwargs["db_pool_max"] = int(db_pool_max)
        if output_dir := _env("VIDEO_DETECTION_OUTPUT_DIR", "OUTPUT_DIR"):
            storage_kwargs["output_dir"] = Path(output_dir)

        return cls(
            video_path=Path(video_path) if video_path else None,
            processing=ProcessingConfig(**processing_kwargs),
            storage=StorageConfig(**storage_kwargs),
            debug=_env("VIDEO_DETECTION_DEBUG", "DEBUG", "").lower() == "true"  # type: ignore[union-attr]
            if _env("VIDEO_DETECTION_DEBUG", "DEBUG") is not None
            else False,
            log_level=_env("VIDEO_DETECTION_LOG_LEVEL", "LOG_LEVEL") or "INFO",
        )

    @classmethod
    def _from_dict(cls, data: dict[str, Any]) -> Config:
        """딕셔너리(YAML 로드 결과)에서 Config 생성.

        YAML 문자열 내 ``${ENV_VAR}`` 패턴은 :func:`_expand_env_vars`로
        치환됩니다. 환경 변수가 정의되지 않으면 :class:`MissingEnvVarError`가
        발생합니다 (silent empty 금지).
        """
        data = _expand_env_vars(data)

        processing_data = data.get("processing", {})
        if "frame_skip_mode" in processing_data:
            processing_data["frame_skip_mode"] = FrameSkipMode(
                processing_data["frame_skip_mode"]
            )

        storage_data = dict(data.get("storage", {}))
        if "output_dir" in storage_data:
            storage_data["output_dir"] = Path(storage_data["output_dir"])
        if "db_port" in storage_data:
            storage_data["db_port"] = int(storage_data["db_port"])
        if "db_pool_min" in storage_data:
            storage_data["db_pool_min"] = int(storage_data["db_pool_min"])
        if "db_pool_max" in storage_data:
            storage_data["db_pool_max"] = int(storage_data["db_pool_max"])
        # 비밀번호는 항상 Secret으로 래핑
        if "db_password" in storage_data:
            pw = storage_data["db_password"]
            if not isinstance(pw, Secret):
                storage_data["db_password"] = Secret(str(pw) if pw is not None else "")

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
                "db_name": self.storage.db_name,
                "db_user": self.storage.db_user,
                # 비밀번호는 항상 마스킹 (평문 YAML 덤프 방지)
                "db_password": "***",
                "db_dsn": _mask_dsn_password(self.storage.db_dsn),
                "db_schema": self.storage.db_schema,
                "db_pool_min": self.storage.db_pool_min,
                "db_pool_max": self.storage.db_pool_max,
                "output_dir": str(self.storage.output_dir),
                "save_frames": self.storage.save_frames,
                "frame_format": self.storage.frame_format,
                "frame_quality": self.storage.frame_quality,
                "min_free_disk_mb": self.storage.min_free_disk_mb,
                "disk_check_action": self.storage.disk_check_action,
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
