"""
PaddleOCR 엔진 - 산업용 디스플레이 문자 인식기

산업용 디스플레이 및 모니터링 시스템에서 숫자/텍스트를 인식하기 위한
프로덕션 수준의 PaddleOCR PP-OCRv4 래퍼 모듈입니다.

주요 기능:
    - PP-OCRv4 모델 통합 (GPU/CPU 지원)
    - 숫자 전용 인식 모드 (디지털 필터링)
    - 신뢰도 임계값 기반 필터링
    - 메모리 효율적인 청킹을 통한 배치 처리
    - 모델 재사용을 위한 스레드 안전 싱글톤 패턴
    - 포괄적인 오류 처리 및 로깅

========================================
PaddleOCR PP-OCRv4 파이프라인 개요
========================================

    ┌─────────────────────────────────────────────────────────────────────┐
    │                      입력 이미지                                      │
    └────────────────────────────┬────────────────────────────────────────┘
                                 │
                                 ▼
    ┌─────────────────────────────────────────────────────────────────────┐
    │               텍스트 검출기 (Detection)                               │
    │                                                                     │
    │  알고리즘: DB (Differentiable Binarization)                         │
    │                                                                     │
    │  처리 과정:                                                          │
    │    1. 백본 네트워크 (ResNet/MobileNet)로 특징 추출                    │
    │    2. FPN으로 다중 스케일 특징 융합                                   │
    │    3. Probability Map 생성 (텍스트 영역 확률)                         │
    │    4. Threshold Map 생성 (이진화 임계값)                              │
    │    5. DB 공식으로 미분 가능한 이진화 수행:                             │
    │                                                                     │
    │       B = sigmoid((P - T) * k)                                      │
    │                                                                     │
    │       P: Probability Map, T: Threshold Map, k: 증폭 계수            │
    │                                                                     │
    │    6. 연결된 컴포넌트로 텍스트 영역 추출                               │
    │    7. Unclip으로 바운딩 박스 확장 (det_db_unclip_ratio)               │
    │                                                                     │
    └────────────────────────────┬────────────────────────────────────────┘
                                 │
                                 ▼
    ┌─────────────────────────────────────────────────────────────────────┐
    │               방향 분류기 (Angle Classification)                      │
    │                       [선택적 단계]                                   │
    │                                                                     │
    │  목적: 180도 회전된 텍스트 감지 및 보정                                │
    │                                                                     │
    │  처리 과정:                                                          │
    │    1. 검출된 각 텍스트 영역을 분류기에 입력                            │
    │    2. 0° / 180° 이진 분류 수행                                       │
    │    3. 180°로 분류된 영역은 회전 보정                                  │
    │                                                                     │
    │  ※ use_angle_cls=True일 때만 활성화                                 │
    │                                                                     │
    └────────────────────────────┬────────────────────────────────────────┘
                                 │
                                 ▼
    ┌─────────────────────────────────────────────────────────────────────┐
    │               텍스트 인식기 (Recognition)                             │
    │                                                                     │
    │  알고리즘: SVTR (Single Visual Model for Scene Text Recognition)    │
    │                                                                     │
    │  아키텍처:                                                           │
    │    1. Patch Embedding: 이미지를 패치 시퀀스로 변환                    │
    │    2. Transformer Encoder: 패치 간 글로벌 관계 학습                   │
    │    3. CTC Decoder: 가변 길이 시퀀스 출력                              │
    │                                                                     │
    │  출력:                                                               │
    │    - 인식된 텍스트 문자열                                            │
    │    - 신뢰도 점수 (0.0 ~ 1.0)                                        │
    │                                                                     │
    └────────────────────────────┬────────────────────────────────────────┘
                                 │
                                 ▼
    ┌─────────────────────────────────────────────────────────────────────┐
    │                      후처리 (Post-processing)                        │
    │                                                                     │
    │    1. 신뢰도 임계값 필터링 (confidence_threshold)                    │
    │    2. 숫자 전용 필터링 (numeric_only=True인 경우)                     │
    │    3. 결과 정렬 및 중복 제거                                         │
    │                                                                     │
    └────────────────────────────┬────────────────────────────────────────┘
                                 │
                                 ▼
    ┌─────────────────────────────────────────────────────────────────────┐
    │                      최종 출력 결과                                    │
    │                                                                     │
    │    - OCRResult 튜플                                                  │
    │    - 각 결과: (텍스트, 신뢰도, 바운딩박스, 원본텍스트)                  │
    │                                                                     │
    └─────────────────────────────────────────────────────────────────────┘

========================================
주요 파라미터 설명
========================================

검출 파라미터:
    - det_db_thresh: DB 이진화 임계값 (기본값: 0.3)
      → 낮을수록 더 많은 영역을 텍스트로 인식
    - det_db_box_thresh: 박스 필터링 임계값 (기본값: 0.6)
      → 높을수록 더 확실한 박스만 유지
    - det_db_unclip_ratio: 박스 확장 비율 (기본값: 1.6)
      → 높을수록 텍스트 주변 여백 포함

인식 파라미터:
    - rec_batch_num: 인식 배치 크기 (기본값: 6)
      → GPU 메모리와 속도의 균형
    - max_text_length: 최대 텍스트 길이 (기본값: 25)
      → 산업용 디스플레이에 적합한 길이

설계 패턴:
    - Singleton 패턴: 무거운 모델 인스턴스의 재사용
    - Lazy Loading: 첫 사용 시점에 모델 초기화
    - Factory Method: 다양한 설정으로 인스턴스 생성
    - Strategy 패턴: numeric_only 여부에 따른 필터링 전략

사용 예시:
    >>> from src.ocr.ocr_engine import OCREngine, OCRConfig
    >>> config = OCRConfig(numeric_only=True, confidence_threshold=0.8)
    >>> engine = OCREngine(config)
    >>> result = engine.recognize(image)
    >>> for r in result:
    ...     print(f"{r.text}: {r.confidence:.2f}")
"""

from __future__ import annotations

import logging
import re
import threading
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Sequence

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from .text_corrector import TextCorrector

# 모듈 전용 로거 인스턴스
logger = logging.getLogger(__name__)


class OCRLanguage(Enum):
    """OCR 지원 언어 열거형.

    PaddleOCR이 지원하는 언어 모델을 정의합니다.
    각 언어는 해당 언어에 최적화된 인식 모델을 사용합니다.

    Attributes:
        ENGLISH: 영어 모델 (en)
        KOREAN: 한국어 모델 (korean)
        CHINESE: 중국어 모델 (ch) - 영어/숫자 인식에도 가장 안정적
        JAPANESE: 일본어 모델 (japan)

    Note:
        중국어 모델(ch)은 다국어 문자와 숫자 인식에서
        가장 안정적인 성능을 보여줍니다.
    """

    ENGLISH = "en"
    KOREAN = "korean"
    CHINESE = "ch"
    JAPANESE = "japan"


@dataclass(frozen=True, slots=True)
class OCRConfig:
    """OCR 엔진 설정 클래스.

    PaddleOCR 엔진의 모든 설정 파라미터를 관리하는 불변 데이터클래스입니다.
    각 파라미터는 검출, 인식, 분류 단계에서 사용됩니다.

    ========================================
    파라미터 카테고리별 설명
    ========================================

    1. 하드웨어 설정:
        - use_gpu: GPU 가속 사용 여부
        - gpu_mem: GPU 메모리 제한 (MB)
        - enable_mkldnn: MKL-DNN 가속 (CPU 전용)
        - cpu_threads: CPU 스레드 수

    2. 검출(Detection) 설정:
        - det_db_thresh: DB 확률 맵 이진화 임계값
        - det_db_box_thresh: 박스 신뢰도 임계값
        - det_db_unclip_ratio: 박스 확장 비율

    3. 인식(Recognition) 설정:
        - language: OCR 언어 모델
        - rec_batch_num: 인식 배치 크기
        - max_text_length: 최대 인식 텍스트 길이
        - use_space_char: 공백 문자 인식

    4. 분류(Classification) 설정:
        - use_angle_cls: 방향 분류기 사용
        - cls_batch_num: 분류 배치 크기

    5. 후처리 설정:
        - numeric_only: 숫자만 추출
        - confidence_threshold: 결과 신뢰도 임계값
        - drop_score: 결과 제거 임계값

    Attributes:
        use_gpu: GPU 가속 사용 여부 (기본값: False).
        gpu_mem: GPU 메모리 제한 (MB 단위, 기본값: 500).
        language: OCR 언어 모델 (기본값: CHINESE - 가장 안정적).
        numeric_only: 숫자 전용 필터링 (0-9, ., -, +만 추출).
        confidence_threshold: 최소 신뢰도 점수 (0.0-1.0).
        det_db_thresh: DB 검출 임계값.
        det_db_box_thresh: DB 박스 임계값.
        det_db_unclip_ratio: DB 언클립 비율.
        rec_batch_num: 인식 배치 크기.
        max_text_length: 최대 텍스트 길이.
        use_angle_cls: 텍스트 방향 분류 사용.
        cls_batch_num: 분류 배치 크기.
        enable_mkldnn: MKL-DNN 가속 사용 (CPU 전용).
        cpu_threads: CPU 스레드 수.
        use_space_char: 공백 문자 인식 활성화.
        drop_score: 이 점수 미만 결과 제거.
        model_dir: 커스텀 모델 디렉토리 경로.
        show_log: PaddleOCR 로그 표시.

    Example:
        >>> # 산업용 디스플레이 숫자 인식 설정
        >>> config = OCRConfig(
        ...     numeric_only=True,
        ...     confidence_threshold=0.8,
        ...     use_gpu=True,
        ...     gpu_mem=1000
        ... )
    """

    # ========================================
    # 하드웨어 설정
    # ========================================
    use_gpu: bool = False  # GPU 가속 활성화 여부
    gpu_mem: int = 500  # GPU 메모리 제한 (MB)

    # ========================================
    # 언어 및 필터링 설정
    # ========================================
    # 중국어 모델(ch)이 영어/숫자 인식에도 가장 안정적
    language: OCRLanguage = OCRLanguage.CHINESE
    # 텍스트+숫자 모두 추출 (산업용 디스플레이용)
    numeric_only: bool = False
    # 신뢰도 임계값 (이 값 미만은 필터링)
    confidence_threshold: float = 0.7

    # ========================================
    # 텍스트 검출(Detection) 파라미터
    # ========================================
    # DB 이진화 임계값 - 낮을수록 더 많은 영역 감지 (0.25: 저대비 환경 대응)
    det_db_thresh: float = 0.25
    # 박스 필터링 임계값 - PaddleOCR 기본값 (높을수록 확실한 박스만)
    det_db_box_thresh: float = 0.6
    # 박스 확장 비율 - 텍스트 주변 여백 포함 정도 (1.8: 잘린 문자 방지)
    det_db_unclip_ratio: float = 1.8

    # ========================================
    # 텍스트 인식(Recognition) 파라미터
    # ========================================
    # 인식 배치 크기 (GPU 메모리와 속도 균형)
    rec_batch_num: int = 6
    # 최대 텍스트 길이 (산업용 디스플레이에 적합)
    max_text_length: int = 25

    # ========================================
    # 방향 분류(Classification) 파라미터
    # ========================================
    # 180도 회전된 텍스트 자동 보정
    use_angle_cls: bool = True
    cls_batch_num: int = 6

    # ========================================
    # CPU 최적화 파라미터
    # ========================================
    # Intel MKL-DNN 가속 (CPU 모드에서만 효과)
    enable_mkldnn: bool = True
    cpu_threads: int = 4

    # ========================================
    # 후처리 파라미터
    # ========================================
    # 공백 문자 인식 비활성화 (산업용 디스플레이: 불필요한 공백 방지)
    use_space_char: bool = False
    # 이 점수 미만 결과는 내부적으로 제거
    drop_score: float = 0.5

    # ========================================
    # 텍스트 교정 설정
    # ========================================
    # 텍스트 교정 활성화 (공백 정규화, 유사 문자 교정)
    enable_text_correction: bool = False  # 기본 비활성화 (점진적 활성화)
    # 교정 설정 객체 (None이면 기본 설정 사용)
    # TextCorrectionConfig 타입이지만 순환 참조 방지를 위해 Any 사용
    correction_config: Any = None

    # ========================================
    # 모델 및 로깅 설정
    # ========================================
    # 커스텀 모델 경로 (None이면 기본 모델 사용)
    model_dir: Path | None = None
    # PaddleOCR 내부 로그 표시
    show_log: bool = False


@dataclass(frozen=True, slots=True)
class BoundingBox:
    """텍스트 영역 바운딩 박스 클래스.

    감지된 텍스트 영역의 사각형 경계를 표현하는 데이터클래스입니다.
    PaddleOCR의 다각형 검출 결과를 축 정렬 사각형으로 변환합니다.

    좌표 체계:
        ┌─────────────────────────────────────┐
        │ (0,0)                               │
        │    ┌───────────────────┐            │
        │    │ (x_min, y_min)    │            │
        │    │                   │  height    │
        │    │      center       │            │
        │    │                   │            │
        │    │        (x_max, y_max)          │
        │    └───────────────────┘            │
        │           width                     │
        └─────────────────────────────────────┘

    Attributes:
        x_min: 최소 x 좌표 (왼쪽 경계).
        y_min: 최소 y 좌표 (위쪽 경계).
        x_max: 최대 x 좌표 (오른쪽 경계).
        y_max: 최대 y 좌표 (아래쪽 경계).
        points: 검출에서 얻은 원본 다각형 꼭짓점 좌표.

    Example:
        >>> box = BoundingBox.from_polygon([[10, 20], [100, 20], [100, 50], [10, 50]])
        >>> print(f"크기: {box.width}x{box.height}, 중심: {box.center}")
    """

    x_min: int  # 왼쪽 경계 x 좌표
    y_min: int  # 위쪽 경계 y 좌표
    x_max: int  # 오른쪽 경계 x 좌표
    y_max: int  # 아래쪽 경계 y 좌표
    # 원본 다각형 꼭짓점 (PaddleOCR 검출 결과)
    points: tuple[tuple[float, float], ...] = field(default_factory=tuple)

    @property
    def width(self) -> int:
        """바운딩 박스의 너비를 계산합니다.

        Returns:
            x_max - x_min 값 (픽셀 단위).
        """
        return self.x_max - self.x_min

    @property
    def height(self) -> int:
        """바운딩 박스의 높이를 계산합니다.

        Returns:
            y_max - y_min 값 (픽셀 단위).
        """
        return self.y_max - self.y_min

    @property
    def area(self) -> int:
        """바운딩 박스의 면적을 계산합니다.

        Returns:
            width * height 값 (제곱 픽셀 단위).
        """
        return self.width * self.height

    @property
    def center(self) -> tuple[int, int]:
        """바운딩 박스의 중심점을 계산합니다.

        Returns:
            (중심_x, 중심_y) 좌표 튜플.
        """
        return (
            self.x_min + self.width // 2,
            self.y_min + self.height // 2,
        )

    @classmethod
    def from_polygon(cls, points: Sequence[Sequence[float]]) -> BoundingBox:
        """다각형 꼭짓점에서 바운딩 박스를 생성합니다.

        PaddleOCR의 4점 다각형 검출 결과를
        축 정렬 바운딩 박스(AABB)로 변환합니다.

        변환 과정:
            원본 다각형:           바운딩 박스:
                 p1────p2             ┌──────┐
                /        \\           │      │
               p4────────p3   →      │      │
                                      └──────┘
            (기울어진 사각형)      (축 정렬 사각형)

        Args:
            points: [x, y] 좌표 쌍의 리스트.
                    일반적으로 4개의 꼭짓점 (시계방향/반시계방향).

        Returns:
            다각형을 포함하는 최소 바운딩 박스 인스턴스.

        Example:
            >>> points = [[10.5, 20.3], [100.2, 22.1], [99.8, 50.5], [11.2, 49.8]]
            >>> box = BoundingBox.from_polygon(points)
        """
        # 모든 x, y 좌표 추출
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]

        return cls(
            x_min=int(min(xs)),  # 가장 왼쪽 x
            y_min=int(min(ys)),  # 가장 위쪽 y
            x_max=int(max(xs)),  # 가장 오른쪽 x
            y_max=int(max(ys)),  # 가장 아래쪽 y
            # 원본 다각형 꼭짓점 보존
            points=tuple((float(p[0]), float(p[1])) for p in points),
        )


@dataclass(frozen=True, slots=True)
class OCRResult:
    """단일 OCR 인식 결과 클래스.

    하나의 텍스트 영역에 대한 인식 결과를 담는 데이터클래스입니다.
    텍스트, 신뢰도, 위치 정보를 포함합니다.

    Attributes:
        text: 인식된 텍스트 문자열.
        confidence: 인식 신뢰도 점수 (0.0 ~ 1.0).
        bounding_box: 텍스트 영역의 바운딩 박스.
        raw_text: 필터링 전 원본 텍스트 (numeric_only 모드용).

    Example:
        >>> result = OCRResult(
        ...     text="123.45",
        ...     confidence=0.95,
        ...     bounding_box=box,
        ...     raw_text="123.45 kg"
        ... )
        >>> if result.is_valid:
        ...     value = result.numeric_value  # 123.45
    """

    text: str  # 인식된 (필터링된) 텍스트
    confidence: float  # 인식 신뢰도 (0.0 ~ 1.0)
    bounding_box: BoundingBox  # 텍스트 위치 정보
    raw_text: str = ""  # 필터링 전 원본 텍스트

    @property
    def is_valid(self) -> bool:
        """결과에 유효한 텍스트가 있는지 확인합니다.

        공백만 있거나 빈 문자열인 경우 False를 반환합니다.

        Returns:
            텍스트가 비어있지 않으면 True.
        """
        return bool(self.text.strip())

    @property
    def numeric_value(self) -> float | None:
        """텍스트를 숫자 값으로 파싱을 시도합니다.

        산업용 디스플레이에서 읽은 숫자 값을 추출할 때 사용합니다.
        공백을 제거한 후 float로 변환을 시도합니다.

        Returns:
            파싱된 float 값, 파싱 불가능하면 None.

        Example:
            >>> result.text = "  123.45  "
            >>> result.numeric_value  # 123.45
            >>> result.text = "N/A"
            >>> result.numeric_value  # None
        """
        try:
            # 공백 제거 후 숫자 변환 시도
            cleaned = self.text.strip().replace(" ", "")
            return float(cleaned)
        except ValueError:
            return None


@dataclass(frozen=True, slots=True)
class OCRBatchResult:
    """배치 OCR 처리 결과 클래스.

    여러 이미지를 배치로 처리한 결과를 담는 데이터클래스입니다.
    처리 통계와 실패 정보를 포함합니다.

    배치 처리 흐름:
        이미지들 ──┬── 청크 1 ──> 처리 ──┐
                  ├── 청크 2 ──> 처리 ──┼──> 결과 취합
                  └── 청크 N ──> 처리 ──┘

    Attributes:
        results: 이미지별 OCR 결과 튜플의 튜플.
        total_images: 처리된 총 이미지 수.
        successful_count: 유효한 결과가 있는 이미지 수.
        failed_indices: 처리 실패한 이미지의 인덱스들.

    Example:
        >>> batch_result = engine.recognize_batch(images, chunk_size=10)
        >>> print(f"성공률: {batch_result.success_rate:.1%}")
        >>> texts = batch_result.get_all_texts()
    """

    # 이미지별 OCR 결과 (튜플의 튜플)
    results: tuple[tuple[OCRResult, ...], ...]
    total_images: int  # 총 이미지 수
    successful_count: int  # 성공한 이미지 수
    failed_indices: tuple[int, ...]  # 실패한 인덱스들

    @property
    def success_rate(self) -> float:
        """배치 처리 성공률을 계산합니다.

        Returns:
            성공한 이미지 비율 (0.0 ~ 1.0).
            총 이미지가 0이면 0.0 반환.
        """
        if self.total_images == 0:
            return 0.0
        return self.successful_count / self.total_images

    def get_all_texts(self) -> list[list[str]]:
        """배치 결과에서 모든 인식된 텍스트를 추출합니다.

        이미지별로 인식된 텍스트 문자열을 중첩 리스트로 반환합니다.

        Returns:
            이미지별 텍스트 리스트의 리스트.

        Example:
            >>> texts = batch_result.get_all_texts()
            >>> # [['123', '456'], ['789'], ['ABC', 'DEF']]
        """
        return [[r.text for r in image_results] for image_results in self.results]


class OCREngine:
    """PaddleOCR 엔진 - 산업용 디스플레이 인식 전용.

    산업용 디스플레이에서 숫자/텍스트를 인식하기 위한
    스레드 안전한 프로덕션 수준의 OCR 엔진입니다.

    ========================================
    엔진 아키텍처
    ========================================

        ┌───────────────────────────────────────────────────┐
        │                  OCREngine                        │
        │                                                   │
        │  ┌─────────────────────────────────────────────┐ │
        │  │              Thread-Safe Layer               │ │
        │  │         (Singleton + Lock 패턴)              │ │
        │  └──────────────────┬──────────────────────────┘ │
        │                     │                            │
        │  ┌──────────────────▼──────────────────────────┐ │
        │  │             Lazy Loading Layer               │ │
        │  │        (첫 호출 시 모델 초기화)               │ │
        │  └──────────────────┬──────────────────────────┘ │
        │                     │                            │
        │  ┌──────────────────▼──────────────────────────┐ │
        │  │           PaddleOCR PP-OCRv4                 │ │
        │  │                                             │ │
        │  │  ┌─────────┐  ┌─────────┐  ┌─────────────┐ │ │
        │  │  │ 검출기   │ →│ 분류기  │ →│   인식기     │ │ │
        │  │  │ (DB)    │  │(Angle) │  │  (SVTR)    │ │ │
        │  │  └─────────┘  └─────────┘  └─────────────┘ │ │
        │  └─────────────────────────────────────────────┘ │
        │                                                   │
        │  ┌─────────────────────────────────────────────┐ │
        │  │             Post-Processing                  │ │
        │  │     (신뢰도 필터링, 숫자 필터링)              │ │
        │  └─────────────────────────────────────────────┘ │
        └───────────────────────────────────────────────────┘

    주요 기능:
        - PP-OCRv4 모델 통합 (최신 SOTA 성능)
        - GPU 가속 및 메모리 제한 설정
        - 숫자 전용 필터링 (산업용 디스플레이용)
        - 신뢰도 기반 결과 필터링
        - 메모리 효율적인 청킹 배치 처리
        - 스레드 안전한 싱글톤 패턴

    설계 패턴:
        - Singleton: 무거운 모델 인스턴스 재사용
        - Lazy Loading: 첫 사용 시점에 초기화
        - Template Method: recognize() 메서드의 처리 흐름

    Example:
        >>> config = OCRConfig(numeric_only=True, confidence_threshold=0.8)
        >>> engine = OCREngine(config)
        >>> result = engine.recognize(image)
        >>> for r in result:
        ...     print(f"{r.text}: {r.confidence:.2f}")

    Note:
        - 모델 초기화는 첫 번째 recognize() 호출 시 수행됩니다.
        - warmup() 메서드로 미리 초기화할 수 있습니다.
        - GPU 사용 시 gpu_mem 파라미터로 메모리 제한을 설정하세요.
    """

    # 클래스 레벨 싱글톤 저장소 (설정 해시 → 인스턴스)
    _instances: dict[int, OCREngine] = {}
    # 스레드 안전을 위한 클래스 레벨 락
    _lock: threading.Lock = threading.Lock()

    # ========================================
    # 숫자 필터링용 정규식 패턴
    # ========================================
    # 숫자가 아닌 문자 제거 (0-9, ., -, +, 콤마, 공백만 유지)
    _NUMERIC_PATTERN: re.Pattern[str] = re.compile(r"[^\d.\-+,\s]")
    # 연속 공백을 제거하는 패턴
    _DIGIT_CLEANUP_PATTERN: re.Pattern[str] = re.compile(r"\s+")

    def __init__(self, config: OCRConfig | None = None) -> None:
        """OCR 엔진을 초기화합니다.

        Args:
            config: OCR 설정 객체. None이면 기본값 사용.

        Note:
            실제 PaddleOCR 모델 로딩은 첫 recognize() 호출 시 수행됩니다.
            이는 불필요한 초기화 비용을 방지하기 위함입니다.
        """
        self._config = config or OCRConfig()
        self._ocr = None  # PaddleOCR 인스턴스 (지연 로딩)
        self._initialized = False  # 모델 초기화 상태
        self._init_lock = threading.Lock()  # 초기화 동기화 락

        # 텍스트 교정기 초기화
        self._text_corrector: TextCorrector | None = None
        if self._config.enable_text_correction:
            from .text_corrector import TextCorrector, TextCorrectionConfig

            correction_config = self._config.correction_config
            if correction_config is None:
                correction_config = TextCorrectionConfig()
            self._text_corrector = TextCorrector(correction_config)
            logger.info("TextCorrector 활성화됨")

        logger.info("OCREngine 생성됨, 설정: %s", self._config)

    @property
    def config(self) -> OCRConfig:
        """현재 설정을 반환합니다.

        Returns:
            OCRConfig 인스턴스.
        """
        return self._config

    @property
    def is_initialized(self) -> bool:
        """OCR 모델 초기화 상태를 확인합니다.

        Returns:
            모델이 초기화되었으면 True.
        """
        return self._initialized

    def _ensure_initialized(self) -> None:
        """OCR 모델이 초기화되었는지 확인합니다 (지연 로딩).

        Double-checked locking 패턴을 사용하여
        스레드 안전하게 모델을 초기화합니다.

        초기화 흐름:
            1. 이미 초기화됨? → 바로 반환
            2. 락 획득
            3. 다시 확인 (다른 스레드가 초기화했을 수 있음)
            4. PaddleOCR 인스턴스 생성
            5. 초기화 플래그 설정

        Raises:
            RuntimeError: PaddleOCR 초기화 실패 시.
        """
        if self._initialized:
            return

        with self._init_lock:
            # Double-checked locking: 다른 스레드가 초기화했을 수 있음
            if self._initialized:
                return

            try:
                # ★ Windows DLL 충돌 방지: torch를 paddleocr보다 먼저 import
                # paddleocr → albumentations → torch 경로에서
                # paddle의 libiomp5md.dll이 먼저 로드되면 torch shm.dll이 실패함.
                # torch를 먼저 import하면 torch의 호환 DLL이 사용됨.
                try:
                    import torch  # noqa: F401 - Windows DLL preload
                    logger.debug("torch DLL 사전 로드 완료")
                except (ImportError, OSError) as torch_err:
                    # torch 미설치: ImportError → albumentations가 pytorch를 건너뜀
                    # torch DLL 손상: OSError → 경고 후 paddleocr import 시도
                    logger.debug("torch 사전 로드 건너뜀: %s", torch_err)

                from paddleocr import PaddleOCR

                # ========================================
                # PaddleOCR 2.7.x API 파라미터 구성
                # ========================================
                # 중국어 모델(ch)이 영어/숫자 인식에도 가장 안정적
                ocr_params = {
                    # 방향 분류기 사용 여부
                    "use_angle_cls": self._config.use_angle_cls,
                    # 언어 모델 선택
                    "lang": self._config.language.value,
                    # GPU 설정
                    "use_gpu": self._config.use_gpu,
                    "gpu_mem": self._config.gpu_mem,
                    # 검출 파라미터
                    "det_db_thresh": self._config.det_db_thresh,
                    "det_db_box_thresh": self._config.det_db_box_thresh,
                    "det_db_unclip_ratio": self._config.det_db_unclip_ratio,
                    # 인식 파라미터
                    "rec_batch_num": self._config.rec_batch_num,
                    "max_text_length": self._config.max_text_length,
                    # 분류 파라미터
                    "cls_batch_num": self._config.cls_batch_num,
                    # 후처리 파라미터
                    "use_space_char": self._config.use_space_char,
                    "drop_score": self._config.drop_score,
                    # 로깅
                    "show_log": self._config.show_log,
                }

                # 커스텀 모델 디렉토리가 지정된 경우
                if self._config.model_dir is not None:
                    ocr_params["det_model_dir"] = str(
                        self._config.model_dir / "det"
                    )
                    ocr_params["rec_model_dir"] = str(
                        self._config.model_dir / "rec"
                    )
                    ocr_params["cls_model_dir"] = str(
                        self._config.model_dir / "cls"
                    )

                # PaddleOCR 인스턴스 생성
                self._ocr = PaddleOCR(**ocr_params)
                self._initialized = True
                logger.info("PaddleOCR PP-OCRv4 초기화 완료")

            except ImportError as e:
                msg = (
                    "PaddleOCR이 설치되지 않았습니다. "
                    "설치 명령: pip install paddlepaddle paddleocr"
                )
                logger.error(msg)
                raise RuntimeError(msg) from e
            except Exception as e:
                msg = f"PaddleOCR 초기화 실패: {e}"
                logger.error(msg)
                raise RuntimeError(msg) from e

    def _filter_numeric(self, text: str) -> str:
        """텍스트에서 숫자 문자만 추출합니다.

        산업용 디스플레이에서 숫자 값만 필요한 경우 사용합니다.
        숫자(0-9), 소수점(.), 부호(+/-), 콤마만 유지합니다.

        필터링 과정:
            "Temperature: 123.45°C" → "123.45"
            "Value: -67.89 kg"     → "-67.89"
            "Error N/A"            → ""

        Args:
            text: 입력 텍스트 문자열.

        Returns:
            숫자, 소수점, 부호만 포함된 필터링된 텍스트.
        """
        # 숫자가 아닌 문자 제거
        filtered = self._NUMERIC_PATTERN.sub("", text)
        # 연속 공백 제거
        filtered = self._DIGIT_CLEANUP_PATTERN.sub("", filtered)
        return filtered.strip()

    def _process_result(
        self,
        detection: list,
    ) -> OCRResult | None:
        """단일 PaddleOCR 검출 결과를 처리합니다.

        PaddleOCR의 원시 출력을 OCRResult 객체로 변환합니다.
        신뢰도 필터링과 숫자 필터링을 적용합니다.

        처리 흐름:
            1. 박스 좌표와 텍스트/신뢰도 추출
            2. 신뢰도 임계값 확인
            3. numeric_only 모드시 숫자 필터링
            4. 바운딩 박스 변환
            5. OCRResult 객체 생성

        Args:
            detection: PaddleOCR 원시 검출 결과.
                       형식: [box_points, (text, confidence)]

        Returns:
            변환된 OCRResult 객체.
            필터링으로 제거되면 None.
        """
        try:
            # PaddleOCR 결과 언패킹
            box_points = detection[0]  # 4점 다각형 좌표
            text, confidence = detection[1]  # 텍스트와 신뢰도

            # 신뢰도 임계값 필터링
            if confidence < self._config.confidence_threshold:
                logger.debug(
                    "신뢰도 미달로 필터링됨: %s (%.2f < %.2f)",
                    text,
                    confidence,
                    self._config.confidence_threshold,
                )
                return None

            # 원본 텍스트 보존
            raw_text = text

            # 텍스트 교정 적용 (숫자 필터링 전에 적용)
            if self._text_corrector is not None:
                correction_result = self._text_corrector.correct(text)
                text = correction_result.corrected_text
                confidence += correction_result.confidence_delta
                confidence = max(0.0, min(1.0, confidence))  # 0~1 범위 유지

                if correction_result.applied_corrections:
                    logger.debug(
                        "텍스트 교정 적용: '%s' → '%s' (%s)",
                        raw_text,
                        text,
                        ", ".join(correction_result.applied_corrections),
                    )

            # 숫자 전용 모드 필터링
            if self._config.numeric_only:
                text = self._filter_numeric(text)
                if not text:
                    logger.debug(
                        "숫자가 없어 필터링됨: %s",
                        raw_text,
                    )
                    return None

            # 다각형 → 바운딩 박스 변환
            bounding_box = BoundingBox.from_polygon(box_points)

            return OCRResult(
                text=text,
                confidence=confidence,
                bounding_box=bounding_box,
                raw_text=raw_text,
            )

        except (IndexError, TypeError, ValueError) as e:
            logger.warning("검출 결과 처리 실패: %s", e)
            return None

    def recognize(
        self,
        image: NDArray[np.uint8] | str | Path,
    ) -> tuple[OCRResult, ...]:
        """단일 이미지에서 텍스트를 인식합니다.

        PaddleOCR PP-OCRv4를 사용하여 이미지에서
        텍스트를 검출하고 인식합니다.

        처리 파이프라인:
            ┌────────────────┐
            │   입력 이미지   │ (ndarray / 파일경로)
            └───────┬────────┘
                    │
                    ▼
            ┌────────────────┐
            │   이미지 전처리  │
            │  - 그레이→BGR   │
            │  - 크기 조정    │
            └───────┬────────┘
                    │
                    ▼
            ┌────────────────┐
            │ PaddleOCR 추론  │
            │ (검출→분류→인식) │
            └───────┬────────┘
                    │
                    ▼
            ┌────────────────┐
            │    후처리       │
            │ - 신뢰도 필터   │
            │ - 숫자 필터    │
            └───────┬────────┘
                    │
                    ▼
            ┌────────────────┐
            │  OCRResult 튜플 │
            └────────────────┘

        Args:
            image: 입력 이미지. 다음 형식 지원:
                   - numpy 배열 (BGR 형식, OpenCV 컨벤션)
                   - 파일 경로 문자열
                   - Path 객체

        Returns:
            검출된 텍스트 영역의 OCRResult 튜플.
            텍스트가 없으면 빈 튜플.

        Raises:
            RuntimeError: OCR 초기화 실패 시.
            ValueError: 이미지 형식이 잘못된 경우.

        Example:
            >>> results = engine.recognize(image)
            >>> for r in results:
            ...     print(f"텍스트: {r.text}, 신뢰도: {r.confidence:.2f}")
        """
        # 지연 초기화 확인
        self._ensure_initialized()

        # ========================================
        # 입력 이미지 전처리
        # ========================================
        if isinstance(image, (str, Path)):
            # 파일 경로는 그대로 전달
            image_input = str(image)
        elif isinstance(image, np.ndarray):
            # numpy 배열 유효성 검사
            if image.ndim not in (2, 3):
                raise ValueError(
                    f"잘못된 이미지 차원: {image.ndim}. "
                    "2D 그레이스케일 또는 3D 컬러 이미지가 필요합니다."
                )
            import cv2

            # 그레이스케일을 BGR로 변환 (PaddleOCR 입력 형식)
            if image.ndim == 2:
                image_input = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            else:
                image_input = image.copy()

            # ========================================
            # 작은 이미지 확대 (OCR 성능 향상)
            # ========================================
            # PaddleOCR은 최소 32px 높이를 권장
            h, w = image_input.shape[:2]
            min_height, min_width = 32, 50
            if h < min_height or w < min_width:
                # 최소 크기를 충족하도록 스케일 계산
                scale = max(min_height / h, min_width / w, 1.0)
                if scale > 1.0:
                    new_w, new_h = int(w * scale), int(h * scale)
                    # 큐빅 보간으로 품질 유지하며 확대
                    image_input = cv2.resize(
                        image_input, (new_w, new_h),
                        interpolation=cv2.INTER_CUBIC
                    )
                    logger.debug(
                        "작은 이미지 확대: %dx%d → %dx%d (OCR 성능 향상)",
                        w, h, new_w, new_h
                    )
        else:
            raise ValueError(
                f"지원하지 않는 이미지 타입: {type(image)}. "
                "numpy 배열, str, 또는 Path가 필요합니다."
            )

        # ========================================
        # PaddleOCR 추론 실행
        # ========================================
        # ★ PaddleOCR 호출과 결과 후처리를 별도 try 블록으로 격리
        # PaddleOCR 내부 IndexError가 전체 결과를 폐기하지 않도록 분리
        raw_results = None
        try:
            raw_results = self._ocr.ocr(image_input, cls=self._config.use_angle_cls)
        except IndexError as e:
            logger.warning("PaddleOCR 내부 IndexError: %s", e)
        except Exception as e:
            logger.warning("PaddleOCR 추론 실패: %s", e)

        # 결과 없음 처리
        if raw_results is None or len(raw_results) == 0:
            logger.debug("이미지에서 텍스트가 감지되지 않음")
            return ()

        # PaddleOCR은 배치 호환성을 위해 중첩 리스트 반환
        try:
            detections = raw_results[0] if raw_results[0] else []
        except (IndexError, TypeError):
            logger.warning("PaddleOCR 결과 구조 이상: %s", type(raw_results))
            return ()

        logger.debug(
            "PaddleOCR raw detections: %d개",
            len(detections),
        )

        # 각 검출 결과 개별 처리 (_process_result 내부에서 개별 에러 처리)
        results = []
        for detection in detections:
            result = self._process_result(detection)
            if result is not None:
                results.append(result)

        logger.debug("%d개의 텍스트 영역 인식됨", len(results))
        return tuple(results)

    def recognize_batch(
        self,
        images: Sequence[NDArray[np.uint8] | str | Path],
        chunk_size: int = 10,
    ) -> OCRBatchResult:
        """여러 이미지를 배치로 처리합니다.

        대량의 이미지를 메모리 효율적으로 처리하기 위해
        청킹 방식으로 배치 처리합니다.

        배치 처리 전략:
            총 이미지: N개, 청크 크기: C

            ┌─────────────────────────────────────────────────┐
            │ 청크 1: images[0:C]     → 순차 처리 → results    │
            │ 청크 2: images[C:2C]    → 순차 처리 → results    │
            │ ...                                             │
            │ 청크 M: images[(M-1)*C:] → 순차 처리 → results   │
            └─────────────────────────────────────────────────┘
                                    │
                                    ▼
                           OCRBatchResult
            (전체 결과, 성공/실패 통계, 실패 인덱스)

        Args:
            images: 입력 이미지 시퀀스.
            chunk_size: 배치당 처리할 이미지 수 (메모리 관리용).
                        기본값: 10.

        Returns:
            모든 인식 결과가 포함된 OCRBatchResult.

        Raises:
            RuntimeError: OCR 초기화 실패 시.

        Example:
            >>> result = engine.recognize_batch(images, chunk_size=5)
            >>> print(f"성공률: {result.success_rate:.1%}")
            >>> print(f"실패 인덱스: {result.failed_indices}")
        """
        # 지연 초기화 확인
        self._ensure_initialized()

        # 결과 저장용 리스트
        all_results: list[tuple[OCRResult, ...]] = []
        failed_indices: list[int] = []
        successful_count = 0

        # 청크 단위로 처리
        for i in range(0, len(images), chunk_size):
            chunk = images[i : i + chunk_size]

            # 청크 내 각 이미지 처리
            for j, image in enumerate(chunk):
                global_idx = i + j  # 전체 인덱스
                try:
                    results = self.recognize(image)
                    all_results.append(results)
                    # 유효한 결과가 있으면 성공 카운트
                    if results:
                        successful_count += 1
                except Exception as e:
                    logger.warning(
                        "이미지 처리 실패 (인덱스 %d): %s",
                        global_idx,
                        e,
                    )
                    all_results.append(())  # 빈 결과
                    failed_indices.append(global_idx)

        return OCRBatchResult(
            results=tuple(all_results),
            total_images=len(images),
            successful_count=successful_count,
            failed_indices=tuple(failed_indices),
        )

    def recognize_region(
        self,
        image: NDArray[np.uint8],
        region: tuple[int, int, int, int],
    ) -> tuple[OCRResult, ...]:
        """이미지의 특정 영역에서 텍스트를 인식합니다.

        전체 이미지가 아닌 관심 영역(ROI)만 처리할 때 사용합니다.
        결과의 좌표는 원본 이미지 기준으로 변환됩니다.

        좌표 변환 과정:
            원본 이미지:
            ┌─────────────────────────────────┐
            │                                 │
            │    ROI 영역 (x, y, w, h)        │
            │    ┌───────────────┐            │
            │    │   텍스트      │            │
            │    │  (로컬 좌표)   │            │
            │    └───────────────┘            │
            │                                 │
            └─────────────────────────────────┘

            결과: 텍스트 위치 = 로컬 좌표 + (x, y)

        Args:
            image: 입력 이미지 (BGR 형식 numpy 배열).
            region: 관심 영역 (x, y, width, height).

        Returns:
            해당 영역에서 감지된 텍스트의 OCRResult 튜플.
            좌표는 원본 이미지 기준.

        Raises:
            ValueError: 영역이 유효하지 않거나 이미지 범위를 벗어남.

        Example:
            >>> region = (100, 50, 200, 100)  # x, y, w, h
            >>> results = engine.recognize_region(image, region)
            >>> # results의 좌표는 원본 이미지 기준
        """
        x, y, w, h = region

        # 영역 유효성 검사
        if x < 0 or y < 0 or w <= 0 or h <= 0:
            raise ValueError(f"잘못된 영역 크기: {region}")

        img_h, img_w = image.shape[:2]
        if x + w > img_w or y + h > img_h:
            raise ValueError(
                f"영역 {region}이 이미지 경계 ({img_w}, {img_h})를 벗어남"
            )

        # ROI 추출
        roi = image[y : y + h, x : x + w]

        # ROI에서 텍스트 인식
        results = self.recognize(roi)

        # ========================================
        # 바운딩 박스를 전역 좌표로 변환
        # ========================================
        adjusted_results = []
        for r in results:
            # 로컬 좌표에 ROI 오프셋 추가
            adjusted_box = BoundingBox(
                x_min=r.bounding_box.x_min + x,
                y_min=r.bounding_box.y_min + y,
                x_max=r.bounding_box.x_max + x,
                y_max=r.bounding_box.y_max + y,
                # 다각형 꼭짓점도 변환
                points=tuple(
                    (px + x, py + y) for px, py in r.bounding_box.points
                ),
            )
            adjusted_results.append(
                OCRResult(
                    text=r.text,
                    confidence=r.confidence,
                    bounding_box=adjusted_box,
                    raw_text=r.raw_text,
                )
            )

        return tuple(adjusted_results)

    def warmup(self, dummy_size: tuple[int, int] = (100, 100)) -> None:
        """더미 이미지로 OCR 모델을 워밍업합니다.

        첫 번째 실제 인식 호출의 지연 시간을 줄이기 위해
        모델을 미리 로드하고 초기 추론을 수행합니다.

        워밍업 효과:
            - 모델 가중치를 메모리/GPU에 로드
            - CUDA 커널 컴파일 (GPU 모드)
            - JIT 최적화 트리거

        Args:
            dummy_size: 더미 이미지 크기 (width, height).
                        기본값: (100, 100).

        Example:
            >>> engine = OCREngine(config)
            >>> engine.warmup()  # 앱 시작 시 호출
            >>> # 이후 첫 recognize()가 더 빠름
        """
        # 모델 초기화 확인
        self._ensure_initialized()

        # 흰색 배경의 더미 이미지 생성
        dummy_image = np.zeros(
            (dummy_size[1], dummy_size[0], 3),
            dtype=np.uint8,
        )
        dummy_image.fill(255)  # 흰색 배경

        try:
            # 더미 추론 실행
            self.recognize(dummy_image)
            logger.info("OCR 모델 워밍업 완료")
        except Exception as e:
            # 워밍업 실패는 치명적이지 않음
            logger.warning("OCR 워밍업 실패 (비치명적): %s", e)

    def __repr__(self) -> str:
        """문자열 표현을 반환합니다.

        Returns:
            OCREngine 상태를 보여주는 문자열.
        """
        return (
            f"OCREngine(initialized={self._initialized}, "
            f"language={self._config.language.value}, "
            f"numeric_only={self._config.numeric_only}, "
            f"confidence_threshold={self._config.confidence_threshold})"
        )


# ========================================
# OCRConfig 팩토리 함수
# ========================================


def create_industrial_ocr_config() -> OCRConfig:
    """산업용 디스플레이 OCR 최적화 설정을 생성합니다.

    일반적인 산업용 디스플레이(HMI, 계기판, LCD 패널 등)에
    최적화된 OCR 설정입니다.

    특징:
        - 공백 인식 비활성화 (불필요한 공백 방지)
        - 박스 확장 비율 증가 (잘린 문자 방지)
        - 낮은 검출 임계값 (저대비 환경 대응)
        - 텍스트 교정 활성화 (유사 문자 교정)

    Returns:
        산업용 디스플레이에 최적화된 OCRConfig 인스턴스

    Example:
        >>> config = create_industrial_ocr_config()
        >>> engine = OCREngine(config)
        >>> result = engine.recognize(industrial_display_image)
    """
    return OCRConfig(
        use_space_char=False,
        det_db_thresh=0.25,
        det_db_unclip_ratio=1.8,
        confidence_threshold=0.7,
        enable_text_correction=True,
    )


def create_7segment_ocr_config() -> OCRConfig:
    """7-세그먼트 디스플레이 전용 OCR 설정을 생성합니다.

    7-세그먼트 LED 디스플레이에 특화된 설정입니다.
    숫자만 인식하며, 끊어진 세그먼트 대응을 위해
    박스 확장 비율을 높게 설정합니다.

    특징:
        - 숫자 전용 모드 활성화
        - 공백 인식 비활성화
        - 더 낮은 검출 임계값 (세그먼트 감지)
        - 더 큰 박스 확장 비율 (끊어진 세그먼트 연결)

    Returns:
        7-세그먼트 디스플레이에 최적화된 OCRConfig 인스턴스

    Example:
        >>> config = create_7segment_ocr_config()
        >>> engine = OCREngine(config)
        >>> result = engine.recognize(seven_segment_image)
    """
    return OCRConfig(
        numeric_only=True,
        use_space_char=False,
        det_db_thresh=0.2,
        det_db_unclip_ratio=2.0,
        confidence_threshold=0.6,
        enable_text_correction=True,
    )


def create_label_value_ocr_config() -> OCRConfig:
    """라벨+값 형식 디스플레이 OCR 설정을 생성합니다.

    "Temperature: 25.5°C" 같은 라벨과 값이 함께 있는
    디스플레이에 최적화된 설정입니다.

    특징:
        - 공백 인식 활성화 (라벨과 값 구분 유지)
        - 텍스트 교정 활성화 (유사 문자 교정)
        - 균형잡힌 검출 파라미터

    Returns:
        라벨+값 형식 디스플레이에 최적화된 OCRConfig 인스턴스

    Example:
        >>> config = create_label_value_ocr_config()
        >>> engine = OCREngine(config)
        >>> result = engine.recognize(label_value_image)
        >>> # "Temperature: 25.5°C" 형식 유지
    """
    return OCRConfig(
        numeric_only=False,
        use_space_char=True,  # 라벨 형식 유지를 위해 공백 활성화
        det_db_thresh=0.25,
        det_db_unclip_ratio=1.7,
        confidence_threshold=0.72,
        enable_text_correction=True,
    )
