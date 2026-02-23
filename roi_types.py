"""관심 영역 (ROI) 기본 타입 정의

이 모듈은 ROI 관련 기본 데이터 타입을 제공합니다.
사전 정의된 ROI 기반 비디오 분석 시스템에서 사용됩니다.

주요 타입:
    - ROIType: ROI 유형 열거형 (NUMERIC, TEXT, CHART 등)
    - BoundingBox: 불변 바운딩 박스 좌표
    - ROI: 관심 영역 메타데이터

사용 예시:
    >>> from detection.roi_types import ROI, BoundingBox, ROIType
    >>> bbox = BoundingBox(x=100, y=200, width=50, height=30)
    >>> roi = ROI(id="roi_1", bbox=bbox, roi_type=ROIType.NUMERIC)
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


# ========================================
# ROI 타입 열거형
# ========================================
class ROIType(Enum):
    """감지된 영역의 타입 분류

    산업용 디스플레이에서 일반적으로 발견되는 영역 유형입니다.

    Attributes:
        NUMERIC: 숫자 디스플레이 (온도, 압력, 유량 등의 수치)
        TEXT: 텍스트 라벨 (장비명, 상태 메시지 등)
        CHART: 차트/그래프 영역 (트렌드, 히스토리 그래프)
        UNKNOWN: 분류되지 않은 영역
    """

    NUMERIC = "numeric"
    TEXT = "text"
    CHART = "chart"
    UNKNOWN = "unknown"


# ========================================
# 바운딩 박스 데이터클래스
# ========================================
@dataclass(frozen=True)
class BoundingBox:
    """불변 바운딩 박스 좌표

    영역의 위치와 크기를 정의하는 불변 객체입니다.
    다양한 좌표 변환 및 연산 메서드를 제공합니다.

    Attributes:
        x: 좌상단 x 좌표
        y: 좌상단 y 좌표
        width: 너비 (픽셀)
        height: 높이 (픽셀)

    Note:
        frozen=True로 설정되어 생성 후 수정 불가합니다.
        새로운 바운딩 박스가 필요하면 새 객체를 생성하세요.
    """

    x: int
    y: int
    width: int
    height: int

    def __post_init__(self) -> None:
        """numpy 정수 등 비표준 int 타입을 Python int로 강제 변환

        OpenCV 함수(connectedComponentsWithStats 등)가 np.int32를 반환하면
        sqlite3가 이를 BLOB으로 저장하여 bytes로 읽히는 문제를 방지합니다.
        frozen=True이므로 object.__setattr__를 사용합니다.
        """
        object.__setattr__(self, "x", int(self.x))
        object.__setattr__(self, "y", int(self.y))
        object.__setattr__(self, "width", int(self.width))
        object.__setattr__(self, "height", int(self.height))

    @property
    def x2(self) -> int:
        """오른쪽 경계 x 좌표 (x + width)"""
        return self.x + self.width

    @property
    def y2(self) -> int:
        """아래쪽 경계 y 좌표 (y + height)"""
        return self.y + self.height

    @property
    def center(self) -> tuple[int, int]:
        """바운딩 박스의 중심점 좌표

        Returns:
            (center_x, center_y) 튜플
        """
        return (self.x + self.width // 2, self.y + self.height // 2)

    @property
    def area(self) -> int:
        """바운딩 박스의 면적 (픽셀 단위)"""
        return self.width * self.height

    def to_tuple(self) -> tuple[int, int, int, int]:
        """(x, y, width, height) 튜플로 변환"""
        return (self.x, self.y, self.width, self.height)

    def to_slice(self) -> tuple[slice, slice]:
        """numpy 배열 슬라이스로 변환

        Returns:
            (y_slice, x_slice) - numpy 배열 인덱싱용

        Note:
            numpy 배열은 [행, 열] 순서이므로 y가 먼저옵니다.
        """
        return (slice(self.y, self.y2), slice(self.x, self.x2))

    def expand(self, margin: int) -> BoundingBox:
        """마진을 추가하여 확장된 바운딩 박스 생성

        Args:
            margin: 각 방향으로 확장할 픽셀 수

        Returns:
            확장된 새 바운딩 박스 (x는 0 미만으로 내려가지 않음)
        """
        return BoundingBox(
            x=max(0, self.x - margin),
            y=max(0, self.y - margin),
            width=self.width + 2 * margin,
            height=self.height + 2 * margin,
        )

    def intersects(self, other: BoundingBox) -> bool:
        """다른 바운딩 박스와 겹치는지 확인

        Args:
            other: 비교할 바운딩 박스

        Returns:
            겹치면 True, 아니면 False
        """
        return not (
            self.x2 <= other.x
            or other.x2 <= self.x
            or self.y2 <= other.y
            or other.y2 <= self.y
        )

    def iou(self, other: BoundingBox) -> float:
        """IoU (Intersection over Union) 계산

        두 바운딩 박스의 겹침 정도를 0~1 사이 값으로 반환합니다.

        Args:
            other: 비교할 바운딩 박스

        Returns:
            IoU 값 (0.0 ~ 1.0, 겹치지 않으면 0.0)
        """
        if not self.intersects(other):
            return 0.0

        inter_x1 = max(self.x, other.x)
        inter_y1 = max(self.y, other.y)
        inter_x2 = min(self.x2, other.x2)
        inter_y2 = min(self.y2, other.y2)

        inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
        union_area = self.area + other.area - inter_area

        return inter_area / union_area if union_area > 0 else 0.0


# ========================================
# ROI 데이터클래스
# ========================================
@dataclass
class ROI:
    """관심 영역 (Region of Interest) 메타데이터 포함

    영역의 위치, 타입, 신뢰도 등의 정보를 담는 객체입니다.

    Attributes:
        id: 고유 식별자
        bbox: 바운딩 박스 좌표
        roi_type: 영역 타입 (NUMERIC, TEXT, CHART, UNKNOWN)
        confidence: 감지 신뢰도 (0.0 ~ 1.0)
        label: 영역 라벨
        metadata: 추가 메타데이터 딕셔너리
    """

    id: str
    bbox: BoundingBox
    roi_type: ROIType
    confidence: float = 0.0
    label: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def extract_region(self, frame: NDArray[np.uint8]) -> NDArray[np.uint8]:
        """프레임에서 ROI 영역 추출

        Args:
            frame: 소스 프레임 (numpy 배열)

        Returns:
            추출된 영역의 복사본
        """
        y_slice, x_slice = self.bbox.to_slice()
        return frame[y_slice, x_slice].copy()

    def to_dict(self) -> dict[str, Any]:
        """ROI를 딕셔너리로 직렬화

        Returns:
            JSON 저장용 딕셔너리
        """
        return {
            "id": self.id,
            "bbox": asdict(self.bbox),
            "roi_type": self.roi_type.value,
            "confidence": self.confidence,
            "label": self.label,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ROI:
        """딕셔너리에서 ROI 역직렬화

        Args:
            data: JSON에서 로드된 딕셔너리

        Returns:
            복원된 ROI 객체
        """
        return cls(
            id=data["id"],
            bbox=BoundingBox(**data["bbox"]),
            roi_type=ROIType(data["roi_type"]),
            confidence=data.get("confidence", 0.0),
            label=data.get("label", ""),
            metadata=data.get("metadata", {}),
        )


def save_rois(rois: list[ROI], path: str | Path) -> None:
    """ROI 리스트를 JSON 파일로 저장

    Args:
        rois: 저장할 ROI 리스트
        path: 저장 경로
    """
    path = Path(path)
    data = [roi.to_dict() for roi in rois]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    logger.info(f"{len(rois)}개 ROI를 {path}에 저장")


def load_rois(path: str | Path) -> list[ROI]:
    """JSON 파일에서 ROI 리스트 로드

    Args:
        path: 로드할 파일 경로

    Returns:
        복원된 ROI 리스트
    """
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    rois = [ROI.from_dict(item) for item in data]
    logger.info(f"{path}에서 {len(rois)}개 ROI 로드")
    return rois
