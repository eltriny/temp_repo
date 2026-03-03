"""앵커 기반 동적 ROI 탐지기

프레임 내 '앵커' 요소(스니펫 이미지 또는 텍스트 라벨)를 찾고,
앵커 위치로부터 정규화 오프셋을 적용하여 ROI 좌표를 계산합니다.

탐지 파이프라인:
    [프레임] → _detect_windows()  (윈도우 경계 + 타이틀 OCR)
            → _find_anchors()    (스니펫/텍스트 앵커 탐색)
            → _compute_rois()    (정규화 오프셋 → 절대 좌표 변환)
            → _remap_to_global() (윈도우 로컬 → 전체 프레임 좌표)
            → _merge_overlapping() (IoU 기반 중복 제거)

앵커 유형:
    - snippet: 작은 이미지 조각을 TemplateMatcher로 캐스케이드 매칭
    - text:    OCR 결과에서 정규식 패턴으로 텍스트 라벨 검색

사용 예시:
    >>> from detection.anchor_detector import AnchorDetector, AnchorDetectorConfig
    >>> config = AnchorDetectorConfig(anchors=[...], roi_mappings=[...])
    >>> detector = AnchorDetector(config)
    >>> rois = detector.detect(frame)

    또는 YAML 설정에서 생성:
    >>> detector = AnchorDetector.from_yaml(Path("config/anchors.yaml"))
    >>> rois = detector.detect(frame)
"""

from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray

from .roi_types import ROI, BoundingBox, ROIType
from .template_matcher import MatchCandidate, MatcherConfig, TemplateMatcher
from .window_boundary_detector import WindowBoundaryConfig, WindowBoundaryDetector

try:
    import yaml
except ImportError:
    yaml = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)


# ========================================
# 데이터 타입 정의
# ========================================
@dataclass(frozen=True)
class AnchorDefinition:
    """앵커 정의

    프레임 내에서 찾을 기준점(앵커)의 속성을 정의합니다.
    snippet 타입은 이미지 매칭, text 타입은 OCR 매칭을 사용합니다.

    Attributes:
        name: 앵커 식별자 (예: "temp_icon")
        anchor_type: 앵커 유형 ("snippet" 또는 "text")
        snippet_path: snippet 앵커: 이미지 파일 경로
        search_text: text 앵커: OCR 검색 텍스트 (정규식)
        match_threshold: 매칭 신뢰도 임계값
        window_pattern: 이 앵커를 찾을 윈도우 타이틀 패턴 (정규식)
    """

    name: str
    anchor_type: str  # "snippet" | "text"
    snippet_path: Path | None = None
    search_text: str | None = None
    match_threshold: float = 0.7
    window_pattern: str | None = None
    source_resolution: tuple[int, int] | None = (
        None  # (width, height) 스니펫 캡처 시 프레임 크기
    )


@dataclass(frozen=True)
class NormalizedOffset:
    """정규화 좌표 오프셋 (0.0~1.0, 기준 영역 대비 비율)

    앵커 위치를 기준으로, 기준 영역(윈도우 또는 전체 프레임)의
    너비/높이에 대한 비율로 ROI 위치와 크기를 정의합니다.

    Attributes:
        nx: X 오프셋 (앵커 위치 기준, 기준 영역 너비 대비)
        ny: Y 오프셋 (앵커 위치 기준, 기준 영역 높이 대비)
        nw: ROI 너비 (기준 영역 너비 대비)
        nh: ROI 높이 (기준 영역 높이 대비)
    """

    nx: float
    ny: float
    nw: float
    nh: float


@dataclass(frozen=True)
class AnchorROIMapping:
    """앵커와 ROI 간의 매핑

    하나의 앵커에서 하나의 ROI를 생성하기 위한 매핑 정보입니다.
    동일한 앵커에서 여러 ROI를 생성하려면 매핑을 여러 개 정의합니다.

    Attributes:
        anchor_name: 참조할 앵커 이름 (AnchorDefinition.name)
        roi_name: 생성될 ROI 이름 (라벨에 사용)
        offset: 정규화 오프셋
        roi_type: ROI 유형
    """

    anchor_name: str
    roi_name: str
    offset: NormalizedOffset
    roi_type: ROIType = ROIType.NUMERIC


@dataclass
class AnchorDetectorConfig:
    """앵커 기반 ROI 탐지 설정

    Attributes:
        anchors: 앵커 정의 리스트
        roi_mappings: 앵커-ROI 매핑 리스트
        enable_window_detection: 윈도우 경계 탐지 활성화 여부
        window_title_patterns: 탐지할 윈도우 타이틀 패턴 (정규식)
        fallback_to_homography: 앵커 탐지 실패 시 호모그래피 폴백 여부
        reference_frame_path: 호모그래피 폴백용 참조 프레임 경로
        matcher_config: TemplateMatcher 설정
        iou_merge_threshold: 중복 ROI 병합 IoU 임계값
    """

    anchors: list[AnchorDefinition] = field(default_factory=list)
    roi_mappings: list[AnchorROIMapping] = field(default_factory=list)
    enable_window_detection: bool = True
    window_title_patterns: list[str] = field(default_factory=list)
    fallback_to_homography: bool = False
    reference_frame_path: Path | None = None
    matcher_config: MatcherConfig = field(default_factory=MatcherConfig)
    iou_merge_threshold: float = 0.5

    # 다중 후보 공간 검증
    enable_spatial_validation: bool = False
    spatial_validation_top_n: int = 5
    expected_region: dict[str, tuple[float, float, float, float]] = field(
        default_factory=dict
    )
    spatial_penalty_weight: float = 0.3


# ========================================
# 메인 탐지기 클래스
# ========================================
class AnchorDetector:
    """앵커 기반 동적 ROI 탐지기

    프레임 내 앵커 요소(스니펫 이미지 또는 텍스트 라벨)를 찾고,
    앵커 위치로부터 정규화 오프셋을 적용하여 ROI 좌표를 계산합니다.

    윈도우 경계 탐지가 활성화되면, 각 윈도우 내에서 독립적으로
    앵커를 탐색하고 ROI를 계산한 뒤, 전체 프레임 좌표로 변환합니다.

    Example:
        >>> config = AnchorDetectorConfig(
        ...     anchors=[AnchorDefinition(name="icon", anchor_type="snippet",
        ...                               snippet_path=Path("icon.png"))],
        ...     roi_mappings=[AnchorROIMapping(
        ...         anchor_name="icon", roi_name="value",
        ...         offset=NormalizedOffset(nx=0.05, ny=0.0, nw=0.1, nh=0.03))],
        ... )
        >>> detector = AnchorDetector(config)
        >>> rois = detector.detect(frame)
    """

    def __init__(self, config: AnchorDetectorConfig) -> None:
        self.config = config

        # 캐스케이드 템플릿 매칭 엔진
        self._matcher = TemplateMatcher(config.matcher_config)

        # 스니펫 이미지 사전 로딩
        self._snippet_images: dict[str, NDArray[np.uint8]] = {}
        self._load_snippet_images()

        # 윈도우 경계 탐지기 (선택적)
        self._window_detector: WindowBoundaryDetector | None = None
        if config.enable_window_detection:
            self._window_detector = WindowBoundaryDetector(
                config=WindowBoundaryConfig(),
            )

        # 호모그래피 폴백용 참조 프레임
        self._reference_frame: NDArray[np.uint8] | None = None
        if config.fallback_to_homography and config.reference_frame_path is not None:
            ref = cv2.imread(str(config.reference_frame_path))
            if ref is not None:
                self._reference_frame = ref
                logger.info(
                    "호모그래피 폴백 참조 프레임 로드: %s (%dx%d)",
                    config.reference_frame_path,
                    ref.shape[1],
                    ref.shape[0],
                )
            else:
                logger.warning(
                    "참조 프레임 로드 실패: %s",
                    config.reference_frame_path,
                )

        # OCR 엔진 (지연 초기화, text 앵커 사용 시에만)
        self._ocr_engine: Any = None
        self._ocr_initialized = False

        # 다중 후보 공간 검증: 이전 앵커 위치 캐시
        self._last_anchor_positions: dict[str, BoundingBox] = {}

    def _load_snippet_images(self) -> None:
        """설정에 정의된 스니펫 앵커 이미지를 사전 로딩합니다."""
        for anchor in self.config.anchors:
            if anchor.anchor_type != "snippet":
                continue
            if anchor.snippet_path is None:
                logger.warning(
                    "스니펫 앵커 '%s'에 snippet_path가 없습니다",
                    anchor.name,
                )
                continue
            if not anchor.snippet_path.exists():
                logger.warning(
                    "스니펫 이미지 파일이 존재하지 않습니다: %s",
                    anchor.snippet_path,
                )
                continue

            img = cv2.imread(str(anchor.snippet_path))
            if img is None:
                logger.error(
                    "스니펫 이미지 로드 실패: %s",
                    anchor.snippet_path,
                )
                continue

            self._snippet_images[anchor.name] = img
            logger.info(
                "스니펫 앵커 '%s' 로드: %s (%dx%d)",
                anchor.name,
                anchor.snippet_path.name,
                img.shape[1],
                img.shape[0],
            )

    def _ensure_ocr(self) -> bool:
        """OCR 엔진을 지연 초기화합니다.

        PaddleOCR 의존성이 없으면 False를 반환하여
        text 앵커 탐색을 건너뛰도록 합니다.

        Returns:
            OCR 사용 가능 여부
        """
        if self._ocr_initialized:
            return self._ocr_engine is not None

        self._ocr_initialized = True
        try:
            from paddleocr import PaddleOCR

            self._ocr_engine = PaddleOCR(
                use_angle_cls=False,
                lang="korean",
                show_log=False,
            )
            logger.info("PaddleOCR 초기화 완료 (text 앵커 지원)")
            return True
        except ImportError:
            logger.warning(
                "PaddleOCR를 사용할 수 없습니다. text 앵커는 비활성화됩니다. "
                "(pip install paddleocr paddlepaddle)"
            )
            self._ocr_engine = None
            return False
        except Exception as e:
            logger.warning("PaddleOCR 초기화 실패: %s", e)
            self._ocr_engine = None
            return False

    # ========================================
    # 공개 API
    # ========================================
    def detect(self, frame: NDArray[np.uint8]) -> list[ROI]:
        """프레임에서 앵커 기반 ROI를 탐지합니다.

        파이프라인:
            1. _detect_windows() - 윈도우 경계 탐지 (선택적)
            2. _find_anchors() - 각 윈도우(또는 전체 프레임) 내 앵커 탐색
            3. _compute_rois() - 앵커 위치 + 정규화 오프셋 → ROI 좌표 계산
            4. _remap_to_global() - 윈도우 로컬 좌표를 전체 프레임 좌표로 변환
            5. _merge_overlapping() - IoU 기반 중복 ROI 병합

        Args:
            frame: 입력 프레임 (BGR 형식)

        Returns:
            탐지된 ROI 리스트 (고유 ID 부여, 중복 제거, 전체 프레임 좌표)
        """
        if frame is None or frame.size == 0:
            logger.warning("빈 프레임이 입력됨")
            return []

        # 1. 윈도우 경계 탐지
        windows = self._detect_windows(frame)
        logger.info("윈도우 %d개 탐지", len(windows))

        all_rois: list[ROI] = []

        for window_bbox, window_title in windows:
            # 윈도우 영역 크롭 (프레임 범위 내로 클리핑)
            frame_h, frame_w = frame.shape[:2]
            y_start = max(0, window_bbox.y)
            y_end = min(frame_h, window_bbox.y2)
            x_start = max(0, window_bbox.x)
            x_end = min(frame_w, window_bbox.x2)

            if y_end <= y_start or x_end <= x_start:
                logger.debug(
                    "윈도우 영역이 프레임 범위를 벗어남: %s",
                    window_bbox.to_tuple(),
                )
                continue

            region = frame[y_start:y_end, x_start:x_end].copy()

            # 2. 앵커 탐색
            anchors = self._find_anchors(region, window_bbox, window_title)
            if not anchors:
                logger.debug(
                    "윈도우 '%s'에서 앵커를 찾지 못했습니다",
                    window_title or "(타이틀 없음)",
                )
                # 호모그래피 폴백 시도
                if (
                    self.config.fallback_to_homography
                    and self._reference_frame is not None
                ):
                    homography_rois = self._fallback_homography(
                        region,
                        window_bbox,
                        window_title,
                    )
                    if homography_rois:
                        # 호모그래피 결과도 윈도우 로컬 좌표 → 전체 프레임 좌표 변환
                        homography_rois = self._remap_to_global(
                            homography_rois, window_bbox
                        )
                        all_rois.extend(homography_rois)
                        logger.info(
                            "호모그래피 폴백으로 %d개 ROI 복원",
                            len(homography_rois),
                        )
                continue

            logger.info(
                "윈도우 '%s'에서 앵커 %d개 탐지: %s",
                window_title or "(타이틀 없음)",
                len(anchors),
                list(anchors.keys()),
            )

            # 3. ROI 좌표 계산 (윈도우 로컬 좌표)
            rois = self._compute_rois(anchors, window_bbox)

            # 4. 전체 프레임 좌표로 변환
            rois = self._remap_to_global(rois, window_bbox)
            all_rois.extend(rois)

        if not all_rois:
            logger.warning("앵커 기반 ROI 탐지 결과 없음")
            return []

        # 5. 중복 ROI 병합
        merged = self._merge_overlapping(all_rois)

        # 6. 고유 ID 할당
        result = self._assign_ids(merged)

        logger.info(
            "앵커 기반 ROI 탐지 완료: 총 %d개 (병합 전 %d개)",
            len(result),
            len(all_rois),
        )

        return result

    # ========================================
    # 윈도우 경계 탐지
    # ========================================
    def _detect_windows(
        self,
        frame: NDArray[np.uint8],
    ) -> list[tuple[BoundingBox, str | None]]:
        """프레임 내 윈도우 경계를 탐지하고 타이틀을 OCR합니다.

        윈도우 경계 탐지가 비활성화되어 있으면,
        전체 프레임을 하나의 윈도우로 취급합니다.
        활성화 시 detect_all_with_titles()를 사용하여 다중 윈도우를
        동시에 탐지합니다.

        Args:
            frame: 입력 프레임 (BGR 형식)

        Returns:
            (BoundingBox, 타이틀 텍스트 또는 None) 튜플 리스트
        """
        frame_h, frame_w = frame.shape[:2]
        full_frame_bbox = BoundingBox(x=0, y=0, width=frame_w, height=frame_h)

        if not self.config.enable_window_detection or self._window_detector is None:
            return [(full_frame_bbox, None)]

        try:
            detected = self._window_detector.detect_all_with_titles(frame)
        except Exception as e:
            logger.warning("윈도우 경계 탐지 예외: %s — 전체 프레임 사용", e)
            return [(full_frame_bbox, None)]

        if not detected:
            logger.info("윈도우 경계 미탐지 — 전체 프레임 사용")
            return [(full_frame_bbox, None)]

        for bbox, title in detected:
            logger.info(
                "윈도우 경계 탐지: (%d,%d) %dx%d, 타이틀='%s'",
                bbox.x,
                bbox.y,
                bbox.width,
                bbox.height,
                title or "(없음)",
            )

        # 윈도우 타이틀 패턴 필터링
        if self.config.window_title_patterns:
            filtered: list[tuple[BoundingBox, str | None]] = []
            for bbox, title_text in detected:
                if title_text is None:
                    logger.debug(
                        "윈도우 (%d,%d) 타이틀 OCR 불가 — 건너뜀",
                        bbox.x,
                        bbox.y,
                    )
                    continue
                matched = any(
                    re.search(pattern, title_text)
                    for pattern in self.config.window_title_patterns
                )
                if matched:
                    filtered.append((bbox, title_text))
                else:
                    logger.debug(
                        "윈도우 타이틀 '%s'가 패턴과 일치하지 않음 — 건너뜀",
                        title_text,
                    )
            if not filtered:
                logger.info("패턴과 일치하는 윈도우 없음 — 전체 프레임 사용")
                return [(full_frame_bbox, None)]
            return filtered

        return detected

    def _ocr_title_bar(
        self,
        frame: NDArray[np.uint8],
        window_bbox: BoundingBox,
    ) -> str | None:
        """윈도우 타이틀 바 영역에서 텍스트를 추출합니다.

        Args:
            frame: 전체 프레임 (BGR)
            window_bbox: 윈도우 경계 BoundingBox

        Returns:
            추출된 타이틀 텍스트 또는 None (OCR 불가/실패 시)
        """
        if not self._ensure_ocr():
            return None

        # 타이틀 바 영역: 윈도우 상단 30px (최소 높이 보장)
        title_height = min(30, window_bbox.height // 4)
        if title_height < 10:
            return None

        frame_h, frame_w = frame.shape[:2]
        y_start = max(0, window_bbox.y)
        y_end = min(frame_h, window_bbox.y + title_height)
        x_start = max(0, window_bbox.x)
        x_end = min(frame_w, window_bbox.x2)

        if y_end <= y_start or x_end <= x_start:
            return None

        title_region = frame[y_start:y_end, x_start:x_end]

        try:
            results = self._ocr_engine.ocr(title_region, cls=False)
            if not results or not results[0]:
                return None

            # 모든 인식 텍스트를 연결
            texts: list[str] = []
            for line in results[0]:
                if line and len(line) >= 2:
                    text_info = line[1]
                    text = (
                        text_info[0]
                        if isinstance(text_info, (list, tuple))
                        else str(text_info)
                    )
                    texts.append(text)

            title = " ".join(texts).strip()
            if title:
                logger.debug("윈도우 타이틀 OCR: '%s'", title)
                return title
        except Exception as e:
            logger.debug("타이틀 바 OCR 실패: %s", e)

        return None

    # ========================================
    # 앵커 탐색
    # ========================================
    def _find_anchors(
        self,
        region: NDArray[np.uint8],
        window_bbox: BoundingBox,
        window_title: str | None,
    ) -> dict[str, tuple[BoundingBox, float]]:
        """주어진 영역에서 앵커를 탐지합니다.

        snippet 앵커는 TemplateMatcher를 사용하고,
        text 앵커는 OCR + 정규식 매칭을 사용합니다.

        Args:
            region: 탐색 대상 영역 (BGR, 윈도우 크롭)
            window_bbox: 윈도우 경계 (기준 좌표 참고용)
            window_title: 윈도우 타이틀 텍스트 (앵커 필터링용)

        Returns:
            {앵커 이름: (탐지된 BoundingBox, 매칭 신뢰도)} 딕셔너리
        """
        found_anchors: dict[str, tuple[BoundingBox, float]] = {}

        for anchor in self.config.anchors:
            # 윈도우 패턴 필터링
            if anchor.window_pattern is not None:
                if window_title is None:
                    logger.debug(
                        "앵커 '%s': 윈도우 타이틀 없음 — 패턴 매칭 건너뜀",
                        anchor.name,
                    )
                    continue
                if not re.search(anchor.window_pattern, window_title):
                    logger.debug(
                        "앵커 '%s': 윈도우 타이틀 '%s'가 패턴 '%s'와 불일치",
                        anchor.name,
                        window_title,
                        anchor.window_pattern,
                    )
                    continue

            if anchor.anchor_type == "snippet":
                if self.config.enable_spatial_validation:
                    bbox, score = self._find_snippet_anchor_with_validation(
                        anchor, region, window_bbox
                    )
                else:
                    bbox, score = self._find_snippet_anchor(anchor, region)
            elif anchor.anchor_type == "text":
                bbox, score = self._find_text_anchor(anchor, region)
            else:
                logger.warning(
                    "알 수 없는 앵커 유형: '%s' (앵커: '%s')",
                    anchor.anchor_type,
                    anchor.name,
                )
                continue

            if bbox is not None:
                found_anchors[anchor.name] = (bbox, score)
                logger.debug(
                    "앵커 '%s' 탐지 성공: (%d,%d) %dx%d, score=%.4f",
                    anchor.name,
                    bbox.x,
                    bbox.y,
                    bbox.width,
                    bbox.height,
                    score,
                )

        return found_anchors

    def _find_snippet_anchor(
        self,
        anchor: AnchorDefinition,
        region: NDArray[np.uint8],
    ) -> tuple[BoundingBox | None, float]:
        """스니펫 이미지 매칭으로 앵커를 탐지합니다.

        TemplateMatcher.cascade_match()를 사용하여 에지 → 특징점 → 마스크
        3단계 캐스케이드 매칭을 수행합니다.

        Args:
            anchor: 스니펫 앵커 정의
            region: 탐색 대상 영역 (BGR)

        Returns:
            (탐지된 BoundingBox, 매칭 신뢰도) 또는 (None, 0.0)
        """
        snippet = self._snippet_images.get(anchor.name)
        if snippet is None:
            logger.debug("스니펫 이미지 없음: '%s'", anchor.name)
            return None, 0.0

        region_h, region_w = region.shape[:2]

        # 해상도 기반 스니펫 사전 리사이즈
        if anchor.source_resolution is not None:
            src_w, src_h = anchor.source_resolution
            if src_w > 0 and src_h > 0:
                scale_x = region_w / src_w
                scale_y = region_h / src_h
                avg_scale = (scale_x + scale_y) / 2.0

                if abs(avg_scale - 1.0) > 0.05:  # 5% 이상 차이시 리사이즈
                    new_w = max(1, round(snippet.shape[1] * avg_scale))
                    new_h = max(1, round(snippet.shape[0] * avg_scale))
                    interp = cv2.INTER_AREA if avg_scale < 1.0 else cv2.INTER_LINEAR
                    snippet = cv2.resize(snippet, (new_w, new_h), interpolation=interp)
                    logger.debug(
                        "앵커 '%s' 스니펫 리사이즈: scale=%.3f (%dx%d → %dx%d)",
                        anchor.name,
                        avg_scale,
                        self._snippet_images[anchor.name].shape[1],
                        self._snippet_images[anchor.name].shape[0],
                        new_w,
                        new_h,
                    )

        snippet_h, snippet_w = snippet.shape[:2]

        # 스니펫이 탐색 영역보다 큰 경우
        if snippet_w > region_w or snippet_h > region_h:
            logger.debug(
                "스니펫이 탐색 영역보다 큼: snippet(%dx%d) > region(%dx%d)",
                snippet_w,
                snippet_h,
                region_w,
                region_h,
            )
            return None, 0.0

        # 캐스케이드 매칭
        result = self._matcher.cascade_match(snippet, region)
        if result is None:
            logger.debug("앵커 '%s' 캐스케이드 매칭 실패", anchor.name)
            return None, 0.0

        # 신뢰도 임계값 검증
        if result.score < anchor.match_threshold:
            logger.debug(
                "앵커 '%s' 신뢰도 미달: %.4f < %.4f",
                anchor.name,
                result.score,
                anchor.match_threshold,
            )
            return None, 0.0

        return (
            BoundingBox(
                x=result.x,
                y=result.y,
                width=snippet_w,
                height=snippet_h,
            ),
            result.score,
        )

    def _find_text_anchor(
        self,
        anchor: AnchorDefinition,
        region: NDArray[np.uint8],
    ) -> tuple[BoundingBox | None, float]:
        """OCR + 정규식 매칭으로 텍스트 앵커를 탐지합니다.

        PaddleOCR을 직접 사용하여 (TextROIDetector와의 순환 의존 방지)
        영역에서 텍스트를 인식하고 정규식 패턴과 매칭합니다.

        Args:
            anchor: 텍스트 앵커 정의
            region: 탐색 대상 영역 (BGR)

        Returns:
            (탐지된 BoundingBox, OCR 신뢰도) 또는 (None, 0.0)
        """
        if anchor.search_text is None:
            logger.debug("텍스트 앵커 '%s'에 search_text가 없습니다", anchor.name)
            return None, 0.0

        if not self._ensure_ocr():
            return None, 0.0

        try:
            results = self._ocr_engine.ocr(region, cls=False)
            if not results or not results[0]:
                logger.debug("텍스트 앵커 '%s': OCR 결과 없음", anchor.name)
                return None, 0.0

            pattern = re.compile(anchor.search_text)

            for line in results[0]:
                if line is None or len(line) < 2:
                    continue

                # PaddleOCR 결과 형식: [[좌표], (텍스트, 신뢰도)]
                text_info = line[1]
                if isinstance(text_info, (list, tuple)):
                    text = str(text_info[0])
                    confidence = float(text_info[1]) if len(text_info) > 1 else 0.0
                else:
                    text = str(text_info)
                    confidence = 0.0

                # 정규식 매칭
                if not pattern.search(text):
                    continue

                # 신뢰도 검증
                if confidence < anchor.match_threshold:
                    logger.debug(
                        "텍스트 앵커 '%s' 텍스트 '%s' 신뢰도 미달: %.4f < %.4f",
                        anchor.name,
                        text,
                        confidence,
                        anchor.match_threshold,
                    )
                    continue

                # 좌표 추출 (PaddleOCR: [[x1,y1],[x2,y2],[x3,y3],[x4,y4]])
                box_points = line[0]
                if box_points is None or len(box_points) < 4:
                    continue

                xs = [pt[0] for pt in box_points]
                ys = [pt[1] for pt in box_points]
                x_min = int(min(xs))
                y_min = int(min(ys))
                x_max = int(max(xs))
                y_max = int(max(ys))

                bbox = BoundingBox(
                    x=x_min,
                    y=y_min,
                    width=max(1, x_max - x_min),
                    height=max(1, y_max - y_min),
                )

                logger.debug(
                    "텍스트 앵커 '%s' 매칭 성공: '%s' (confidence=%.4f)",
                    anchor.name,
                    text,
                    confidence,
                )
                return bbox, confidence

        except Exception as e:
            logger.warning("텍스트 앵커 '%s' OCR 실패: %s", anchor.name, e)

        return None, 0.0

    def _find_snippet_anchor_with_validation(
        self,
        anchor: AnchorDefinition,
        region: NDArray[np.uint8],
        ref_bbox: BoundingBox,
    ) -> tuple[BoundingBox | None, float]:
        """다중 후보 공간 검증을 통해 스니펫 앵커를 탐지합니다.

        Top-N 후보를 추출한 뒤, 각 후보에 대해:
        1. 원본 매칭 score
        2. 이전 프레임 위치와의 거리 (temporal consistency)
        3. 예상 영역 포함 여부

        를 종합하여 최적 후보를 선택합니다.

        Args:
            anchor: 스니펫 앵커 정의
            region: 탐색 대상 영역 (BGR)
            ref_bbox: 기준 영역 BoundingBox (공간 검증 참조용)

        Returns:
            (탐지된 BoundingBox, 종합 신뢰도) 또는 (None, 0.0)
        """
        snippet = self._snippet_images.get(anchor.name)
        if snippet is None:
            logger.debug("스니펫 이미지 없음: '%s'", anchor.name)
            return None, 0.0

        region_h, region_w = region.shape[:2]

        # 해상도 기반 스니펫 사전 리사이즈
        if anchor.source_resolution is not None:
            src_w, src_h = anchor.source_resolution
            if src_w > 0 and src_h > 0:
                avg_scale = ((region_w / src_w) + (region_h / src_h)) / 2.0
                if abs(avg_scale - 1.0) > 0.05:
                    new_w = max(1, round(snippet.shape[1] * avg_scale))
                    new_h = max(1, round(snippet.shape[0] * avg_scale))
                    interp = cv2.INTER_AREA if avg_scale < 1.0 else cv2.INTER_LINEAR
                    snippet = cv2.resize(snippet, (new_w, new_h), interpolation=interp)

        snippet_h, snippet_w = snippet.shape[:2]

        if snippet_w > region_w or snippet_h > region_h:
            return None, 0.0

        # Top-N 후보 추출
        candidates = self._matcher.match_by_edges_topn(
            snippet,
            region,
            top_n=self.config.spatial_validation_top_n,
        )

        if not candidates:
            # 폴백: 일반 캐스케이드 매칭 (AKAZE/masked 포함)
            logger.debug(
                "앵커 '%s' Top-N 후보 없음, 캐스케이드 폴백",
                anchor.name,
            )
            return self._find_snippet_anchor(anchor, region)

        # 각 후보에 대해 종합 점수 계산
        best_candidate: MatchCandidate | None = None
        best_combined_score = -1.0
        weight = self.config.spatial_penalty_weight

        for candidate in candidates:
            if candidate.score < anchor.match_threshold:
                continue

            combined_score = candidate.score

            # (a) Temporal consistency: 이전 위치와의 거리
            prev_pos = self._last_anchor_positions.get(anchor.name)
            if prev_pos is not None:
                distance = math.sqrt(
                    (candidate.x - prev_pos.x) ** 2 + (candidate.y - prev_pos.y) ** 2
                )
                max_drift = max(ref_bbox.width, ref_bbox.height) * 0.1
                if distance < max_drift:
                    combined_score += weight * 0.5
                else:
                    drift_penalty = min(1.0, (distance / max_drift) * 0.3)
                    combined_score -= weight * drift_penalty

            # (b) Expected region: 예상 영역 포함 여부
            expected = self.config.expected_region.get(anchor.name)
            if expected is not None:
                enx, eny, enw, enh = expected
                ex = int(enx * ref_bbox.width)
                ey = int(eny * ref_bbox.height)
                ew = int(enw * ref_bbox.width)
                eh = int(enh * ref_bbox.height)

                if ex <= candidate.x <= ex + ew and ey <= candidate.y <= ey + eh:
                    combined_score += weight * 0.3
                else:
                    combined_score -= weight * 0.3

            if combined_score > best_combined_score:
                best_combined_score = combined_score
                best_candidate = candidate

        if best_candidate is None:
            logger.debug(
                "앵커 '%s' 공간 검증 후 유효 후보 없음",
                anchor.name,
            )
            return None, 0.0

        result_bbox = BoundingBox(
            x=best_candidate.x,
            y=best_candidate.y,
            width=snippet_w,
            height=snippet_h,
        )

        # 위치 캐시 업데이트
        self._last_anchor_positions[anchor.name] = result_bbox

        logger.debug(
            "앵커 '%s' 공간 검증 완료: (%d,%d) score=%.4f (raw=%.4f, combined=%.4f)",
            anchor.name,
            best_candidate.x,
            best_candidate.y,
            best_candidate.score,
            best_candidate.score,
            best_combined_score,
        )

        return result_bbox, best_candidate.score

    # ========================================
    # ROI 좌표 계산
    # ========================================
    def _compute_rois(
        self,
        anchors: dict[str, tuple[BoundingBox, float]],
        ref_bbox: BoundingBox,
    ) -> list[ROI]:
        """앵커 위치 + 정규화 오프셋으로 ROI 좌표를 계산합니다.

        각 AnchorROIMapping에 대해, 참조된 앵커가 탐지되었으면
        정규화 오프셋을 적용하여 ROI 절대 좌표를 산출합니다.

        오프셋 계산:
            roi_x = anchor_bbox.x + offset.nx * ref_bbox.width
            roi_y = anchor_bbox.y + offset.ny * ref_bbox.height
            roi_w = offset.nw * ref_bbox.width
            roi_h = offset.nh * ref_bbox.height

        Args:
            anchors: 탐지된 앵커 딕셔너리 {앵커이름: (BoundingBox, 신뢰도)}
            ref_bbox: 기준 영역 BoundingBox (윈도우 또는 전체 프레임)

        Returns:
            계산된 ROI 리스트 (윈도우 로컬 좌표)
        """
        rois: list[ROI] = []
        ref_w = ref_bbox.width
        ref_h = ref_bbox.height

        for mapping in self.config.roi_mappings:
            anchor_info = anchors.get(mapping.anchor_name)
            if anchor_info is None:
                logger.debug(
                    "매핑 '%s': 앵커 '%s'가 탐지되지 않음 — 건너뜀",
                    mapping.roi_name,
                    mapping.anchor_name,
                )
                continue

            anchor_bbox, anchor_score = anchor_info

            offset = mapping.offset

            # 절대 좌표 계산
            roi_x = int(anchor_bbox.x + offset.nx * ref_w)
            roi_y = int(anchor_bbox.y + offset.ny * ref_h)
            roi_w = int(offset.nw * ref_w)
            roi_h = int(offset.nh * ref_h)

            # 좌표 클리핑: 음수 방지 + 기준 영역 상한 제한
            roi_x = max(0, roi_x)
            roi_y = max(0, roi_y)
            roi_w = min(roi_w, ref_w - roi_x)
            roi_h = min(roi_h, ref_h - roi_y)
            roi_w = max(1, roi_w)
            roi_h = max(1, roi_h)

            bbox = BoundingBox(x=roi_x, y=roi_y, width=roi_w, height=roi_h)

            rois.append(
                ROI(
                    id="",  # _assign_ids()에서 재할당
                    bbox=bbox,
                    roi_type=mapping.roi_type,
                    confidence=anchor_score,
                    label=mapping.roi_name,
                    metadata={
                        "source": "anchor",
                        "detection_method": "anchor",
                        "anchor_name": mapping.anchor_name,
                        "anchor_score": round(anchor_score, 4),
                        "anchor_position": anchor_bbox.to_tuple(),
                        "normalized_offset": (
                            offset.nx,
                            offset.ny,
                            offset.nw,
                            offset.nh,
                        ),
                    },
                )
            )

            logger.debug(
                "ROI '%s' 계산: anchor=(%d,%d) + offset=(%.3f,%.3f) "
                "-> bbox=(%d,%d,%d,%d)",
                mapping.roi_name,
                anchor_bbox.x,
                anchor_bbox.y,
                offset.nx,
                offset.ny,
                roi_x,
                roi_y,
                roi_w,
                roi_h,
            )

        return rois

    # ========================================
    # 좌표 변환
    # ========================================
    def _remap_to_global(
        self,
        rois: list[ROI],
        window_bbox: BoundingBox,
    ) -> list[ROI]:
        """윈도우 로컬 좌표를 전체 프레임 좌표로 변환합니다.

        윈도우 경계 탐지가 비활성화되어 전체 프레임을 사용하는 경우,
        오프셋이 (0,0)이므로 좌표가 변경되지 않습니다.

        Args:
            rois: 윈도우 로컬 좌표 기준 ROI 리스트
            window_bbox: 윈도우 경계 BoundingBox (전체 프레임 좌표)

        Returns:
            전체 프레임 좌표로 변환된 ROI 리스트
        """
        if not rois:
            return rois

        result: list[ROI] = []
        for roi in rois:
            global_bbox = BoundingBox(
                x=roi.bbox.x + window_bbox.x,
                y=roi.bbox.y + window_bbox.y,
                width=roi.bbox.width,
                height=roi.bbox.height,
            )
            result.append(
                ROI(
                    id=roi.id,
                    bbox=global_bbox,
                    roi_type=roi.roi_type,
                    confidence=roi.confidence,
                    label=roi.label,
                    metadata={
                        **roi.metadata,
                        "window_offset": (window_bbox.x, window_bbox.y),
                    },
                )
            )

        logger.debug(
            "좌표 복원: offset (%d,%d), %d개 ROI",
            window_bbox.x,
            window_bbox.y,
            len(result),
        )

        return result

    # ========================================
    # 중복 ROI 병합
    # ========================================
    def _merge_overlapping(self, rois: list[ROI]) -> list[ROI]:
        """IoU 기반으로 중복 ROI를 병합합니다.

        높은 신뢰도의 ROI를 우선하여, 겹치는 낮은 신뢰도
        ROI를 흡수합니다.

        Args:
            rois: 병합 전 ROI 리스트

        Returns:
            병합된 ROI 리스트
        """
        if len(rois) <= 1:
            return rois

        threshold = self.config.iou_merge_threshold
        keep = [True] * len(rois)

        for i in range(len(rois)):
            if not keep[i]:
                continue
            for j in range(i + 1, len(rois)):
                if not keep[j]:
                    continue
                iou = rois[i].bbox.iou(rois[j].bbox)
                if iou > threshold:
                    # 신뢰도가 높은 쪽을 유지
                    if rois[i].confidence >= rois[j].confidence:
                        keep[j] = False
                        logger.debug(
                            "ROI 중복 병합: '%s' (IoU=%.2f) -> '%s' 흡수",
                            rois[j].label,
                            iou,
                            rois[i].label,
                        )
                    else:
                        keep[i] = False
                        logger.debug(
                            "ROI 중복 병합: '%s' (IoU=%.2f) -> '%s' 흡수",
                            rois[i].label,
                            iou,
                            rois[j].label,
                        )
                        break

        return [roi for roi, k in zip(rois, keep) if k]

    # ========================================
    # 호모그래피 폴백
    # ========================================
    def _fallback_homography(
        self,
        region: NDArray[np.uint8],
        window_bbox: BoundingBox,
        window_title: str | None = None,
    ) -> list[ROI]:
        """앵커 탐지 실패 시 호모그래피 기반으로 ROI를 복원합니다.

        참조 프레임과 현재 프레임 간의 호모그래피 행렬을 계산하고,
        참조 프레임에서 정의된 ROI 좌표를 현재 프레임으로 변환합니다.

        Args:
            region: 현재 프레임 윈도우 영역 (BGR)
            window_bbox: 윈도우 경계 BoundingBox
            window_title: 현재 윈도우 타이틀 (앵커 필터링에 사용)

        Returns:
            호모그래피로 변환된 ROI 리스트 (빈 리스트이면 실패)
        """
        if self._reference_frame is None:
            return []

        homography = self._matcher.compute_homography(
            self._reference_frame,
            region,
        )
        if homography is None:
            logger.debug("호모그래피 폴백: 행렬 추정 실패")
            return []

        logger.info("호모그래피 폴백: 행렬 추정 성공")

        # 참조 프레임에서의 앵커 탐지 시도
        # window_title을 전달하여 window_pattern 필터가 있는 앵커도 탐색
        ref_bbox = BoundingBox(
            x=0,
            y=0,
            width=self._reference_frame.shape[1],
            height=self._reference_frame.shape[0],
        )
        ref_anchors = self._find_anchors(
            self._reference_frame,
            ref_bbox,
            window_title,
        )
        if not ref_anchors:
            logger.debug("호모그래피 폴백: 참조 프레임에서도 앵커 없음")
            return []

        # 참조 프레임 기준 ROI 계산
        ref_rois = self._compute_rois(ref_anchors, ref_bbox)

        # 호모그래피로 ROI 좌표 변환
        transformed_rois: list[ROI] = []
        for roi in ref_rois:
            # ROI 네 꼭짓점을 호모그래피 변환
            corners = np.float32(
                [
                    [roi.bbox.x, roi.bbox.y],
                    [roi.bbox.x2, roi.bbox.y],
                    [roi.bbox.x2, roi.bbox.y2],
                    [roi.bbox.x, roi.bbox.y2],
                ]
            ).reshape(-1, 1, 2)

            transformed = cv2.perspectiveTransform(corners, homography)
            bx, by, bw, bh = cv2.boundingRect(transformed)

            new_bbox = BoundingBox(
                x=max(0, int(bx)),
                y=max(0, int(by)),
                width=max(1, int(bw)),
                height=max(1, int(bh)),
            )

            transformed_rois.append(
                ROI(
                    id=roi.id,
                    bbox=new_bbox,
                    roi_type=roi.roi_type,
                    confidence=roi.confidence * 0.8,  # 폴백 신뢰도 감소
                    label=roi.label,
                    metadata={
                        **roi.metadata,
                        "detection_method": "homography_fallback",
                    },
                )
            )

        return transformed_rois

    # ========================================
    # ID 할당
    # ========================================
    @staticmethod
    def _assign_ids(rois: list[ROI]) -> list[ROI]:
        """ROI에 고유 ID를 순차적으로 할당합니다.

        ID 형식: anchor_{roi_type}_{index}
        예: anchor_numeric_0, anchor_text_1

        Args:
            rois: ID 재할당 대상 ROI 리스트

        Returns:
            새 ID가 부여된 ROI 리스트
        """
        counters: dict[str, int] = {}
        result: list[ROI] = []

        for roi in rois:
            type_key = roi.roi_type.value
            idx = counters.get(type_key, 0)
            counters[type_key] = idx + 1

            new_id = f"anchor_{type_key}_{idx}"

            result.append(
                ROI(
                    id=new_id,
                    bbox=roi.bbox,
                    roi_type=roi.roi_type,
                    confidence=roi.confidence,
                    label=roi.label,
                    metadata=roi.metadata,
                )
            )

        return result

    # ========================================
    # YAML 설정 로딩
    # ========================================
    @classmethod
    def from_yaml(cls, yaml_path: Path) -> AnchorDetector:
        """YAML 설정 파일에서 AnchorDetector를 생성합니다.

        YAML 구조 예시::

            anchors:
              - name: temp_icon
                anchor_type: snippet
                snippet_path: anchors/temp_icon.png
                match_threshold: 0.7
                window_pattern: "모니터링"

              - name: pressure_label
                anchor_type: text
                search_text: "압력.*kPa"
                match_threshold: 0.6

            roi_mappings:
              - anchor_name: temp_icon
                roi_name: temperature_value
                offset:
                  nx: 0.05
                  ny: 0.0
                  nw: 0.08
                  nh: 0.025
                roi_type: numeric

            enable_window_detection: true
            window_title_patterns:
              - "모니터링.*시스템"

            fallback_to_homography: false
            iou_merge_threshold: 0.5

            matcher_config:
              edge_match_threshold: 0.5
              feature_match_min_count: 8

        Args:
            yaml_path: YAML 설정 파일 경로

        Returns:
            설정이 적용된 AnchorDetector 인스턴스

        Raises:
            ImportError: PyYAML이 설치되지 않은 경우
            FileNotFoundError: YAML 파일이 존재하지 않는 경우
            ValueError: YAML 구조가 올바르지 않은 경우
        """
        if yaml is None:
            raise ImportError(
                "YAML 설정 로딩에 PyYAML이 필요합니다. " "pip install pyyaml"
            )

        yaml_path = Path(yaml_path)
        if not yaml_path.exists():
            raise FileNotFoundError(f"YAML 설정 파일이 존재하지 않습니다: {yaml_path}")

        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        if not isinstance(data, dict):
            raise ValueError(f"YAML 최상위 구조는 딕셔너리여야 합니다: {yaml_path}")

        # YAML 기준 경로 (상대 경로 해석용)
        base_dir = yaml_path.parent

        # 앵커 정의 파싱
        anchors: list[AnchorDefinition] = []
        for anchor_data in data.get("anchors", []):
            snippet_path = None
            if anchor_data.get("snippet_path"):
                snippet_path = Path(anchor_data["snippet_path"])
                if not snippet_path.is_absolute():
                    snippet_path = base_dir / snippet_path

            # source_resolution 파싱 (해상도 적응용)
            source_res = anchor_data.get("source_resolution")
            source_resolution = (
                tuple(source_res) if source_res and len(source_res) == 2 else None
            )

            anchors.append(
                AnchorDefinition(
                    name=anchor_data["name"],
                    anchor_type=anchor_data["anchor_type"],
                    snippet_path=snippet_path,
                    search_text=anchor_data.get("search_text"),
                    match_threshold=anchor_data.get("match_threshold", 0.7),
                    window_pattern=anchor_data.get("window_pattern"),
                    source_resolution=source_resolution,
                )
            )

        # ROI 매핑 파싱
        roi_mappings: list[AnchorROIMapping] = []
        for mapping_data in data.get("roi_mappings", []):
            offset_data = mapping_data.get("offset", {})
            offset = NormalizedOffset(
                nx=float(offset_data.get("nx", 0.0)),
                ny=float(offset_data.get("ny", 0.0)),
                nw=float(offset_data.get("nw", 0.0)),
                nh=float(offset_data.get("nh", 0.0)),
            )

            roi_type_str = mapping_data.get("roi_type", "numeric")
            try:
                roi_type = ROIType(roi_type_str)
            except ValueError:
                logger.warning(
                    "알 수 없는 ROI 유형 '%s' — UNKNOWN으로 설정",
                    roi_type_str,
                )
                roi_type = ROIType.UNKNOWN

            roi_mappings.append(
                AnchorROIMapping(
                    anchor_name=mapping_data["anchor_name"],
                    roi_name=mapping_data["roi_name"],
                    offset=offset,
                    roi_type=roi_type,
                )
            )

        # anchor_name 교차 검증: roi_mapping이 참조하는 앵커가 실제 정의되어 있는지 확인
        anchor_names = {a.name for a in anchors}
        for mapping in roi_mappings:
            if mapping.anchor_name not in anchor_names:
                raise ValueError(
                    f"roi_mapping '{mapping.roi_name}'의 anchor_name "
                    f"'{mapping.anchor_name}'이 정의된 앵커에 없습니다. "
                    f"사용 가능: {sorted(anchor_names)}"
                )

        # MatcherConfig 파싱 (인스턴스 기본값 사용으로 field(default_factory=...) 안전)
        matcher_data = data.get("matcher_config", {})
        _mc_defaults = MatcherConfig()
        matcher_config = MatcherConfig(
            edge_match_threshold=matcher_data.get(
                "edge_match_threshold", _mc_defaults.edge_match_threshold
            ),
            canny_threshold1=matcher_data.get(
                "canny_threshold1", _mc_defaults.canny_threshold1
            ),
            canny_threshold2=matcher_data.get(
                "canny_threshold2", _mc_defaults.canny_threshold2
            ),
            clahe_clip_limit=matcher_data.get(
                "clahe_clip_limit", _mc_defaults.clahe_clip_limit
            ),
            clahe_grid_size=tuple(
                matcher_data.get("clahe_grid_size", list(_mc_defaults.clahe_grid_size))
            ),
            multi_scale_factors=tuple(
                matcher_data.get(
                    "multi_scale_factors",
                    list(_mc_defaults.multi_scale_factors),
                )
            ),
            feature_match_min_count=matcher_data.get(
                "feature_match_min_count",
                _mc_defaults.feature_match_min_count,
            ),
            feature_inlier_ratio_min=matcher_data.get(
                "feature_inlier_ratio_min",
                _mc_defaults.feature_inlier_ratio_min,
            ),
            fallback_match_threshold=matcher_data.get(
                "fallback_match_threshold",
                _mc_defaults.fallback_match_threshold,
            ),
            # Phase 1: 소형 스니펫 AKAZE 완화
            feature_match_min_count_small=matcher_data.get(
                "feature_match_min_count_small",
                _mc_defaults.feature_match_min_count_small,
            ),
            small_template_threshold=matcher_data.get(
                "small_template_threshold",
                _mc_defaults.small_template_threshold,
            ),
            # Phase 2: 전처리 견고성
            auto_canny=matcher_data.get("auto_canny", _mc_defaults.auto_canny),
            multi_clahe=matcher_data.get("multi_clahe", _mc_defaults.multi_clahe),
            clahe_clip_limits=tuple(
                matcher_data.get(
                    "clahe_clip_limits",
                    list(_mc_defaults.clahe_clip_limits),
                )
            ),
            min_edge_density=matcher_data.get(
                "min_edge_density", _mc_defaults.min_edge_density
            ),
            # Phase 3: 적응적 스케일
            adaptive_scale=matcher_data.get(
                "adaptive_scale", _mc_defaults.adaptive_scale
            ),
            adaptive_scale_steps=matcher_data.get(
                "adaptive_scale_steps",
                _mc_defaults.adaptive_scale_steps,
            ),
            adaptive_scale_step_size=matcher_data.get(
                "adaptive_scale_step_size",
                _mc_defaults.adaptive_scale_step_size,
            ),
            # Phase 4: 다중 후보 매칭
            top_n_candidates=matcher_data.get(
                "top_n_candidates",
                _mc_defaults.top_n_candidates,
            ),
            candidate_nms_radius=matcher_data.get(
                "candidate_nms_radius",
                _mc_defaults.candidate_nms_radius,
            ),
            # Tier 0: 원본 픽셀 직접 매칭
            raw_pixel_match_enabled=matcher_data.get(
                "raw_pixel_match_enabled",
                _mc_defaults.raw_pixel_match_enabled,
            ),
            raw_pixel_match_threshold=matcher_data.get(
                "raw_pixel_match_threshold",
                _mc_defaults.raw_pixel_match_threshold,
            ),
            # 디버그 시각화
            debug_visualize=matcher_data.get(
                "debug_visualize",
                _mc_defaults.debug_visualize,
            ),
            debug_output_dir=matcher_data.get(
                "debug_output_dir",
                _mc_defaults.debug_output_dir,
            ),
        )

        # 참조 프레임 경로 파싱
        reference_frame_path = None
        if data.get("reference_frame_path"):
            reference_frame_path = Path(data["reference_frame_path"])
            if not reference_frame_path.is_absolute():
                reference_frame_path = base_dir / reference_frame_path

        # 공간 검증: expected_region 파싱
        raw_expected_region = data.get("expected_region", {})
        expected_region: dict[str, tuple[float, float, float, float]] = {}
        for anchor_name, region_data in raw_expected_region.items():
            if isinstance(region_data, dict):
                expected_region[anchor_name] = (
                    float(region_data.get("nx", 0.0)),
                    float(region_data.get("ny", 0.0)),
                    float(region_data.get("nw", 1.0)),
                    float(region_data.get("nh", 1.0)),
                )
            elif isinstance(region_data, (list, tuple)) and len(region_data) == 4:
                expected_region[anchor_name] = tuple(float(v) for v in region_data)

        # AnchorDetectorConfig 생성
        config = AnchorDetectorConfig(
            anchors=anchors,
            roi_mappings=roi_mappings,
            enable_window_detection=data.get("enable_window_detection", True),
            window_title_patterns=data.get("window_title_patterns", []),
            fallback_to_homography=data.get("fallback_to_homography", False),
            reference_frame_path=reference_frame_path,
            matcher_config=matcher_config,
            iou_merge_threshold=data.get("iou_merge_threshold", 0.5),
            enable_spatial_validation=data.get("enable_spatial_validation", False),
            spatial_validation_top_n=data.get("spatial_validation_top_n", 5),
            expected_region=expected_region,
            spatial_penalty_weight=data.get("spatial_penalty_weight", 0.3),
        )

        logger.info(
            "YAML 설정 로드: %s (앵커 %d개, 매핑 %d개)",
            yaml_path.name,
            len(anchors),
            len(roi_mappings),
        )

        return cls(config)
