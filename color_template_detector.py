"""색상 마커 기반 ROI 템플릿 탐지기

사용자가 비디오 프레임(전체 또는 일부)을 캡쳐한 뒤 특정 색상으로
ROI 영역을 칠한 이미지를 입력으로 받아, 색상 영역을 자동 추출하고
원본 비디오 프레임에서 해당 위치를 매칭하여 ROI 좌표를 산출합니다.

매칭 전략 (캐스케이드):
    1차: 에지 기반 매칭 (Canny + TM_CCOEFF_NORMED, 멀티스케일)
    2차: 특징점 기반 매칭 (AKAZE + RANSAC homography)
    3차: 마스크 기반 상관관계 (TM_CCORR_NORMED + NaN/Inf 가드)

색상-ROI 타입 매핑 (기본값):
    빨강(#FF0000) → NUMERIC  (숫자 디스플레이)
    초록(#00FF00) → TEXT     (텍스트 라벨)
    파랑(#0000FF) → CHART    (차트/파형)
    노랑(#FFFF00) → NUMERIC  (추가 숫자)
    시안(#00FFFF) → TEXT     (추가 텍스트)
    마젠타(#FF00FF) → CHART  (추가 차트)

사용 예시:
    >>> from detection.color_template_detector import ColorTemplateDetector
    >>> detector = ColorTemplateDetector()
    >>> rois = detector.detect_from_templates(
    ...     template_paths=[Path("area1.png"), Path("area2.png")],
    ...     first_frame=frame,
    ... )
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar, NamedTuple

import cv2
import numpy as np
from numpy.typing import NDArray

from .roi_types import ROI, BoundingBox, ROIType

logger = logging.getLogger(__name__)


# ========================================
# 매칭 결과 타입
# ========================================
class MatchResult(NamedTuple):
    """템플릿 매칭 결과"""

    x: int
    y: int
    score: float
    method: str  # "edge", "feature", "masked"


# ========================================
# 설정 데이터클래스
# ========================================
@dataclass
class ColorROIMapping:
    """하나의 마커 색상과 ROI 타입의 매핑 정의

    Attributes:
        color_name: 색상 이름 (예: "red", "green")
        hsv_lower: HSV 색상 범위 하한 (H: 0-179, S: 0-255, V: 0-255)
        hsv_upper: HSV 색상 범위 상한
        hsv_lower2: 두 번째 HSV 범위 하한 (Red처럼 H가 0/179 경계를 감싸는 경우)
        hsv_upper2: 두 번째 HSV 범위 상한
        roi_type: 이 색상에 할당할 ROI 타입
        label_prefix: ROI 라벨 접두사
    """

    color_name: str
    hsv_lower: tuple[int, int, int]
    hsv_upper: tuple[int, int, int]
    hsv_lower2: tuple[int, int, int] | None = None
    hsv_upper2: tuple[int, int, int] | None = None
    roi_type: ROIType = ROIType.NUMERIC
    label_prefix: str = ""


@dataclass
class ColorTemplateConfig:
    """색상 마커 기반 ROI 탐지 설정

    Attributes:
        color_mappings: 색상-ROI 타입 매핑 리스트 (빈 리스트면 기본 6색 사용)
        min_region_area: 최소 영역 면적 (px², 이하 필터링)
        morph_kernel_size: 모폴로지 연산 커널 크기
        morph_close_iterations: Close 연산 반복 횟수 (안티앨리어싱 간극 채움)
        morph_open_iterations: Open 연산 반복 횟수 (스펙클 노이즈 제거)
        padding: ROI 바운딩 박스 마진 (px)
        iou_merge_threshold: 다중 템플릿의 중복 ROI 병합 IoU 임계값
        edge_match_threshold: 에지 매칭 최소 신뢰도 (TM_CCOEFF_NORMED: -1~1)
        feature_match_min_count: AKAZE 특징점 최소 매치 수
        fallback_match_threshold: mask 폴백 매칭 임계값 (상향된 값)
        canny_threshold1: Canny 에지 검출 하한
        canny_threshold2: Canny 에지 검출 상한
        clahe_clip_limit: CLAHE 대비 제한
        clahe_grid_size: CLAHE 그리드 크기
        multi_scale_factors: 멀티스케일 매칭 스케일 팩터
        max_color_stddev: 마커 영역 내 HSV H채널 표준편차 상한 (균일도 검증)
        min_rectangularity: contour 면적 / bbox 면적 최소 비율 (직사각형 검증)
        min_fill_ratio: bbox 내 색상 마스크 픽셀 최소 비율 (채움율 검증)
    """

    color_mappings: list[ColorROIMapping] = field(default_factory=list)
    min_region_area: int = 5000
    morph_kernel_size: tuple[int, int] = (5, 5)
    morph_close_iterations: int = 2
    morph_open_iterations: int = 1
    padding: int = 3
    iou_merge_threshold: float = 0.5
    # 1차: 에지 기반 매칭
    edge_match_threshold: float = 0.5
    canny_threshold1: int = 50
    canny_threshold2: int = 200
    clahe_clip_limit: float = 2.0
    clahe_grid_size: tuple[int, int] = (8, 8)
    multi_scale_factors: tuple[float, ...] = (1.0, 0.75, 0.5, 1.25, 1.5)
    # 2차: AKAZE 특징점 폴백
    feature_match_min_count: int = 8
    # 3차: mask 기반 폴백
    fallback_match_threshold: float = 0.85
    # 마커 검증 필터 (UI 요소 오탐지 방지)
    max_color_stddev: float = 15.0
    min_rectangularity: float = 0.65
    min_fill_ratio: float = 0.7


# ========================================
# 메인 탐지기 클래스
# ========================================
class ColorTemplateDetector:
    """색상 마커 기반 ROI 템플릿 탐지기

    사용자가 색상으로 표시한 템플릿 이미지에서 ROI 영역을 추출하고,
    비디오 프레임에서 해당 위치를 매칭하여 글로벌 좌표로 변환합니다.

    매칭 전략 (캐스케이드):
        1차: 에지 기반 매칭 -색칠 영역 제거 후 Canny 에지로 TM_CCOEFF_NORMED
        2차: AKAZE 특징점 -비-마스크 영역의 특징점으로 homography 추정
        3차: mask 기반 -기존 TM_CCORR_NORMED + mask (NaN/Inf 가드 포함)
    """

    DEFAULT_MAPPINGS: ClassVar[list[ColorROIMapping]] = [
        # 빨강 → NUMERIC (H가 0/179 경계를 감쌈)
        # S/V ≥ 150: JPEG 압축 후에도 순색 마커를 검출하되, 원본 UI 색상은 제외
        ColorROIMapping(
            color_name="red",
            hsv_lower=(0, 150, 150),
            hsv_upper=(10, 255, 255),
            hsv_lower2=(170, 150, 150),
            hsv_upper2=(179, 255, 255),
            roi_type=ROIType.NUMERIC,
            label_prefix="numeric",
        ),
        # 초록 → TEXT
        ColorROIMapping(
            color_name="green",
            hsv_lower=(35, 150, 150),
            hsv_upper=(85, 255, 255),
            roi_type=ROIType.TEXT,
            label_prefix="text",
        ),
        # 파랑 → CHART
        ColorROIMapping(
            color_name="blue",
            hsv_lower=(100, 150, 150),
            hsv_upper=(130, 255, 255),
            roi_type=ROIType.CHART,
            label_prefix="chart",
        ),
        # 노랑 → NUMERIC
        ColorROIMapping(
            color_name="yellow",
            hsv_lower=(20, 150, 150),
            hsv_upper=(35, 255, 255),
            roi_type=ROIType.NUMERIC,
            label_prefix="numeric",
        ),
        # 시안 → TEXT
        ColorROIMapping(
            color_name="cyan",
            hsv_lower=(85, 150, 150),
            hsv_upper=(100, 255, 255),
            roi_type=ROIType.TEXT,
            label_prefix="text",
        ),
        # 마젠타 → CHART
        ColorROIMapping(
            color_name="magenta",
            hsv_lower=(140, 150, 150),
            hsv_upper=(170, 255, 255),
            roi_type=ROIType.CHART,
            label_prefix="chart",
        ),
    ]

    def __init__(self, config: ColorTemplateConfig | None = None) -> None:
        self.config = config or ColorTemplateConfig()
        if not self.config.color_mappings:
            self.config.color_mappings = list(self.DEFAULT_MAPPINGS)

    # ----------------------------------------
    # 공개 API
    # ----------------------------------------
    def detect_from_templates(
        self,
        template_paths: list[Path],
        first_frame: NDArray[np.uint8],
    ) -> list[ROI]:
        """여러 템플릿 이미지에서 ROI를 탐지하고 병합합니다.

        Args:
            template_paths: 색상으로 ROI가 표시된 이미지 경로 리스트
            first_frame: 비디오 첫 프레임 (BGR numpy 배열)

        Returns:
            병합 및 중복 제거된 ROI 리스트
        """
        all_rois: list[ROI] = []

        for path in template_paths:
            template = cv2.imread(str(path))
            if template is None:
                logger.error("템플릿 이미지 로드 실패: %s", path)
                continue

            logger.info(
                "템플릿 처리 중: %s (%dx%d)",
                path.name,
                template.shape[1],
                template.shape[0],
            )
            rois = self.detect_from_template(template, first_frame)
            logger.info("  → %d개 ROI 탐지", len(rois))
            all_rois.extend(rois)

        if not all_rois:
            logger.warning("모든 템플릿에서 ROI를 찾지 못했습니다")
            return []

        merged = self._merge_overlapping_rois(all_rois)
        result = self._assign_ids(merged)
        logger.info("최종 ROI: %d개 (병합 전 %d개)", len(result), len(all_rois))
        return result

    def detect_from_template(
        self,
        template_image: NDArray[np.uint8],
        first_frame: NDArray[np.uint8],
    ) -> list[ROI]:
        """단일 템플릿 이미지에서 ROI를 탐지합니다.

        Args:
            template_image: 색상으로 ROI가 표시된 BGR 이미지
            first_frame: 비디오 첫 프레임 (BGR numpy 배열)

        Returns:
            탐지된 ROI 리스트 (글로벌 좌표)
        """
        # 1. 색상 영역 추출
        colored_regions = self._extract_colored_regions(template_image)
        if not colored_regions:
            logger.warning("템플릿에서 색상 마커를 찾지 못했습니다")
            return []

        # 2. 통합 색상 마스크 생성
        combined_mask = self._create_combined_mask(template_image)

        # 3. 스케일 비율 계산
        scale_x, scale_y = self._compute_scale(
            template_image.shape,
            first_frame.shape,
        )

        # 4. 캐스케이드 템플릿 위치 매칭
        match_result = self._find_template_location(
            template_image,
            first_frame,
            combined_mask,
            scale_x,
            scale_y,
        )
        if match_result is None:
            return []

        dx, dy = match_result.x, match_result.y
        match_score = match_result.score

        logger.info(
            "매칭 성공: method=%s, score=%.4f, offset=(%d, %d)",
            match_result.method,
            match_score,
            dx,
            dy,
        )

        # 5. 로컬 좌표 → 글로벌 좌표 변환
        frame_h, frame_w = first_frame.shape[:2]
        rois: list[ROI] = []

        for i, (bbox, roi_type, color_name) in enumerate(colored_regions):
            global_bbox = BoundingBox(
                x=int(bbox.x * scale_x) + dx,
                y=int(bbox.y * scale_y) + dy,
                width=int(bbox.width * scale_x),
                height=int(bbox.height * scale_y),
            )
            global_bbox = self._clip_to_frame(global_bbox, frame_h, frame_w)
            if global_bbox.area < self.config.min_region_area:
                continue

            rois.append(
                ROI(
                    id=f"color_{color_name}_{i}",
                    bbox=global_bbox,
                    roi_type=roi_type,
                    confidence=match_score,
                    label=f"{color_name}_{roi_type.value}_{i}",
                    metadata={
                        "source": "color_template",
                        "detection_method": match_result.method,
                        "marker_color": color_name,
                        "match_score": round(match_score, 4),
                    },
                )
            )

        return rois

    # ========================================
    # 색상 추출
    # ========================================
    def _extract_colored_regions(
        self,
        image: NDArray[np.uint8],
    ) -> list[tuple[BoundingBox, ROIType, str]]:
        """이미지에서 각 마커 색상의 영역을 추출합니다.

        Returns:
            (BoundingBox, ROIType, color_name) 튜플 리스트
        """
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            self.config.morph_kernel_size,
        )
        regions: list[tuple[BoundingBox, ROIType, str]] = []

        for mapping in self.config.color_mappings:
            mask = self._create_color_mask(hsv, mapping)

            # 모폴로지: Close로 안티앨리어싱 간극 채움, Open으로 스펙클 제거
            mask = cv2.morphologyEx(
                mask,
                cv2.MORPH_CLOSE,
                kernel,
                iterations=self.config.morph_close_iterations,
            )
            mask = cv2.morphologyEx(
                mask,
                cv2.MORPH_OPEN,
                kernel,
                iterations=self.config.morph_open_iterations,
            )

            contours, _ = cv2.findContours(
                mask,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE,
            )

            filtered_count = 0
            for contour in contours:
                area = cv2.contourArea(contour)
                if area < self.config.min_region_area:
                    continue

                x, y, w, h = cv2.boundingRect(contour)
                bbox_area = w * h

                # degenerate contour (선형 등) 즉시 제외
                if bbox_area == 0:
                    filtered_count += 1
                    continue

                # 필터 1: 직사각형 비율 검증
                rectangularity = area / bbox_area
                if rectangularity < self.config.min_rectangularity:
                    logger.debug(
                        "직사각형 비율 미달로 제외: %.2f < %.2f (color=%s, area=%d)",
                        rectangularity,
                        self.config.min_rectangularity,
                        mapping.color_name,
                        area,
                    )
                    filtered_count += 1
                    continue

                # 필터 2: 색상 채움율 검증 (bbox 내 마스크 픽셀 비율)
                roi_mask = mask[y : y + h, x : x + w]
                fill_ratio = np.count_nonzero(roi_mask) / bbox_area
                if fill_ratio < self.config.min_fill_ratio:
                    logger.debug(
                        "채움율 미달로 제외: %.2f < %.2f (color=%s, area=%d)",
                        fill_ratio,
                        self.config.min_fill_ratio,
                        mapping.color_name,
                        area,
                    )
                    filtered_count += 1
                    continue

                # 필터 3: 색상 균일도 검증 (순색 마커만 통과)
                if not self._is_uniform_color_block(
                    hsv, contour, max_std=self.config.max_color_stddev
                ):
                    logger.debug(
                        "색상 균일도 미달로 제외: %s (area=%d)",
                        mapping.color_name,
                        area,
                    )
                    filtered_count += 1
                    continue

                # 패딩 적용
                pad = self.config.padding
                img_h, img_w = image.shape[:2]
                x = max(0, x - pad)
                y = max(0, y - pad)
                w = min(img_w - x, w + 2 * pad)
                h = min(img_h - y, h + 2 * pad)

                bbox = BoundingBox(x=x, y=y, width=w, height=h)
                regions.append((bbox, mapping.roi_type, mapping.color_name))

            if filtered_count > 0:
                logger.info(
                    "  %s: %d개 후보 중 %d개 필터링됨",
                    mapping.color_name,
                    len(contours),
                    filtered_count,
                )

        logger.info("색상 추출 결과: %d개 영역 발견", len(regions))
        return regions

    @staticmethod
    def _circular_std_hue(hue_values: NDArray[np.uint8]) -> float:
        """OpenCV Hue(0-179) 원형 표준편차를 계산합니다.

        빨강(Red)처럼 H가 0/179 경계에 걸치는 색상은 일반 np.std()로
        계산하면 σ가 비정상적으로 커집니다. 원형 통계를 사용하면
        [1, 2, 178, 179] → σ≈2 로 정확히 계산됩니다.

        Args:
            hue_values: 0-179 범위의 Hue 값 배열

        Returns:
            원형 표준편차 (degree 단위, 0-90 범위)
        """
        # OpenCV H(0-179)를 라디안 각도로 변환
        angles = hue_values.astype(np.float64) * (np.pi / 90.0)
        sin_mean = np.mean(np.sin(angles))
        cos_mean = np.mean(np.cos(angles))
        # mean resultant length: 1이면 완전 균일, 0이면 완전 분산
        r = np.sqrt(sin_mean**2 + cos_mean**2)
        r = np.clip(r, 1e-10, 1.0)
        # 원형 표준편차를 degree 단위로 변환
        circular_std_rad = np.sqrt(-2.0 * np.log(r))
        return float(circular_std_rad * (90.0 / np.pi))

    @staticmethod
    def _is_uniform_color_block(
        hsv_image: NDArray[np.uint8],
        contour: NDArray,
        max_std: float = 15.0,
    ) -> bool:
        """contour 내부의 색상 균일도를 검증합니다.

        사용자가 칠한 순색 마커는 H/S/V 채널 모두 표준편차가 매우 낮고,
        UI 아이콘이나 텍스트는 그라데이션/텍스처로 인해 표준편차가 높습니다.

        H채널은 0/179 경계에 걸치는 빨강 등을 위해 원형 통계를 사용합니다.

        Args:
            hsv_image: HSV 색공간 이미지
            contour: 검증할 영역의 컨투어
            max_std: H/S/V 표준편차 허용 최댓값

        Returns:
            True이면 균일한 색상 블록 (=사용자 마커)
        """
        x, y, w, h = cv2.boundingRect(contour)
        roi_hsv = hsv_image[y : y + h, x : x + w]

        # contour 내부만 마스킹
        roi_mask = np.zeros((h, w), dtype=np.uint8)
        shifted_contour = contour - np.array([x, y])
        cv2.drawContours(roi_mask, [shifted_contour], -1, 255, -1)

        masked_h = roi_hsv[:, :, 0][roi_mask > 0]
        masked_s = roi_hsv[:, :, 1][roi_mask > 0]
        masked_v = roi_hsv[:, :, 2][roi_mask > 0]

        if len(masked_s) == 0:
            return False

        # H채널: 원형 통계 (0/179 경계 안전)
        std_h = ColorTemplateDetector._circular_std_hue(masked_h)
        std_s = float(np.std(masked_s))
        std_v = float(np.std(masked_v))

        return std_h < max_std and std_s < max_std and std_v < max_std

    def _create_color_mask(
        self,
        hsv_image: NDArray[np.uint8],
        mapping: ColorROIMapping,
    ) -> NDArray[np.uint8]:
        """단일 색상에 대한 바이너리 마스크를 생성합니다.

        Red처럼 HSV Hue가 0/179 경계를 감싸는 색상은
        두 범위의 OR 결합으로 처리합니다.
        """
        lower = np.array(mapping.hsv_lower, dtype=np.uint8)
        upper = np.array(mapping.hsv_upper, dtype=np.uint8)
        mask = cv2.inRange(hsv_image, lower, upper)

        if mapping.hsv_lower2 is not None and mapping.hsv_upper2 is not None:
            lower2 = np.array(mapping.hsv_lower2, dtype=np.uint8)
            upper2 = np.array(mapping.hsv_upper2, dtype=np.uint8)
            mask2 = cv2.inRange(hsv_image, lower2, upper2)
            mask = cv2.bitwise_or(mask, mask2)

        return mask

    def _create_combined_mask(
        self,
        image: NDArray[np.uint8],
    ) -> NDArray[np.uint8]:
        """모든 마커 색상의 통합 마스크를 생성합니다.

        _extract_colored_regions와 동일한 모폴로지 파이프라인을 적용하여
        노이즈 픽셀이 매칭 마스크를 오염시키지 않도록 합니다.

        Returns:
            색칠된 영역=255, 원본 영역=0 인 단일 채널 마스크
        """
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, w = image.shape[:2]
        combined = np.zeros((h, w), dtype=np.uint8)
        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            self.config.morph_kernel_size,
        )

        for mapping in self.config.color_mappings:
            mask = self._create_color_mask(hsv, mapping)
            # Close로 간극 채움 + Open으로 스펙클 제거 (extract와 동일 파이프라인)
            mask = cv2.morphologyEx(
                mask,
                cv2.MORPH_CLOSE,
                kernel,
                iterations=self.config.morph_close_iterations,
            )
            mask = cv2.morphologyEx(
                mask,
                cv2.MORPH_OPEN,
                kernel,
                iterations=self.config.morph_open_iterations,
            )
            # min_region_area 미만 및 마커 검증 필터 적용
            contours, _ = cv2.findContours(
                mask,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE,
            )
            filtered = np.zeros_like(mask)
            for contour in contours:
                area = cv2.contourArea(contour)
                if area < self.config.min_region_area:
                    continue
                x, y, cw, ch = cv2.boundingRect(contour)
                bbox_area = cw * ch
                if bbox_area == 0:
                    continue
                if area / bbox_area < self.config.min_rectangularity:
                    continue
                roi_mask = mask[y : y + ch, x : x + cw]
                if np.count_nonzero(roi_mask) / bbox_area < self.config.min_fill_ratio:
                    continue
                if not self._is_uniform_color_block(
                    hsv, contour, max_std=self.config.max_color_stddev
                ):
                    continue
                cv2.drawContours(filtered, [contour], -1, 255, -1)
            combined = cv2.bitwise_or(combined, filtered)

        return combined

    # ========================================
    # 캐스케이드 템플릿 매칭
    # ========================================
    def _find_template_location(
        self,
        template: NDArray[np.uint8],
        frame: NDArray[np.uint8],
        color_mask: NDArray[np.uint8],
        scale_x: float,
        scale_y: float,
    ) -> MatchResult | None:
        """캐스케이드 전략으로 프레임 내 템플릿 위치를 찾습니다.

        1차: 에지 기반 매칭 (멀티스케일)
        2차: AKAZE 특징점 매칭
        3차: mask 기반 상관관계 폴백

        Args:
            template: 색칠된 템플릿 이미지 (BGR)
            frame: 비디오 프레임 (BGR)
            color_mask: 색칠 영역 통합 마스크 (색칠=255)
            scale_x: 수평 스케일 비율
            scale_y: 수직 스케일 비율

        Returns:
            MatchResult 또는 매칭 실패 시 None
        """
        # 스케일 적용된 템플릿/마스크 준비
        tmpl_for_match = template
        mask_for_match = color_mask
        if abs(scale_x - 1.0) > 0.01 or abs(scale_y - 1.0) > 0.01:
            frame_h, frame_w = frame.shape[:2]
            tmpl_h, tmpl_w = template.shape[:2]
            new_w = min(int(tmpl_w * scale_x), frame_w)
            new_h = min(int(tmpl_h * scale_y), frame_h)
            tmpl_for_match = cv2.resize(
                template,
                (new_w, new_h),
                interpolation=cv2.INTER_AREA,
            )
            mask_for_match = cv2.resize(
                color_mask,
                (new_w, new_h),
                interpolation=cv2.INTER_NEAREST,
            )

        # --- 1차: 에지 기반 멀티스케일 매칭 ---
        best_edge = self._match_by_edges(tmpl_for_match, frame, mask_for_match)
        if best_edge is not None:
            logger.info(
                "[1차 에지] 성공: score=%.4f at (%d,%d)",
                best_edge.score,
                best_edge.x,
                best_edge.y,
            )
            return best_edge

        logger.info("[1차 에지] 임계값 미달, 2차 AKAZE 시도")

        # --- 2차: AKAZE 특징점 폴백 ---
        feat_result = self._match_by_features(
            tmpl_for_match,
            frame,
            mask_for_match,
        )
        if feat_result is not None:
            logger.info(
                "[2차 AKAZE] 성공: score=%.4f at (%d,%d)",
                feat_result.score,
                feat_result.x,
                feat_result.y,
            )
            return feat_result

        logger.info("[2차 AKAZE] 실패, 3차 mask 폴백 시도")

        # --- 3차: mask 기반 폴백 ---
        mask_result = self._match_by_masked_correlation(
            tmpl_for_match,
            frame,
            mask_for_match,
        )
        if mask_result is not None:
            logger.info(
                "[3차 mask] 성공: score=%.4f at (%d,%d)",
                mask_result.score,
                mask_result.x,
                mask_result.y,
            )
            return mask_result

        logger.warning("모든 매칭 전략 실패")
        return None

    # ----------------------------------------
    # 1차: 에지 기반 멀티스케일 매칭
    # ----------------------------------------
    def _match_by_edges(
        self,
        template: NDArray[np.uint8],
        frame: NDArray[np.uint8],
        color_mask: NDArray[np.uint8],
    ) -> MatchResult | None:
        """에지 기반 멀티스케일 매칭.

        색칠 영역을 검정으로 대체한 뒤 Canny 에지를 추출하여
        TM_CCOEFF_NORMED로 매칭합니다. mask 파라미터를 사용하지 않아
        OpenCV mask 관련 버그를 완전히 우회합니다.
        """
        tmpl_edges = self._preprocess_for_edge_matching(template, color_mask)
        tmpl_h, tmpl_w = tmpl_edges.shape[:2]

        # 에지가 거의 없으면 스킵 (의미 있는 구조가 없음)
        tmpl_edge_ratio = np.count_nonzero(tmpl_edges) / tmpl_edges.size
        if tmpl_edge_ratio < 0.01:
            logger.debug(
                "템플릿 에지 비율 %.2f%% -에지 매칭 스킵", tmpl_edge_ratio * 100
            )
            return None

        best: MatchResult | None = None

        for scale in self.config.multi_scale_factors:
            if scale == 1.0:
                frame_scaled = frame
            else:
                frame_scaled = cv2.resize(
                    frame,
                    None,
                    fx=scale,
                    fy=scale,
                    interpolation=cv2.INTER_AREA,
                )

            scaled_h, scaled_w = frame_scaled.shape[:2]
            if tmpl_w > scaled_w or tmpl_h > scaled_h:
                continue

            frame_edges = self._preprocess_for_edge_matching(frame_scaled)

            result = cv2.matchTemplate(
                frame_edges,
                tmpl_edges,
                cv2.TM_CCOEFF_NORMED,
            )
            _, max_val, _, max_loc = cv2.minMaxLoc(result)

            # 원본 스케일로 좌표 변환
            orig_x = int(max_loc[0] / scale) if scale != 1.0 else max_loc[0]
            orig_y = int(max_loc[1] / scale) if scale != 1.0 else max_loc[1]

            logger.debug(
                "  에지 scale=%.2f → score=%.4f at (%d,%d)",
                scale,
                max_val,
                orig_x,
                orig_y,
            )

            if max_val >= self.config.edge_match_threshold:
                if best is None or max_val > best.score:
                    best = MatchResult(
                        x=orig_x,
                        y=orig_y,
                        score=max_val,
                        method="edge",
                    )

        return best

    # ----------------------------------------
    # 2차: AKAZE 특징점 폴백
    # ----------------------------------------
    def _match_by_features(
        self,
        template: NDArray[np.uint8],
        frame: NDArray[np.uint8],
        color_mask: NDArray[np.uint8],
    ) -> MatchResult | None:
        """AKAZE 특징점 매칭으로 템플릿 위치를 추정합니다.

        색칠 영역을 마스킹하여 원본 콘텐츠의 특징점만 추출하고,
        RANSAC homography로 대응 관계를 추정합니다.
        """
        # 그레이스케일 변환
        gray_tmpl = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 특징점 추출 (템플릿: 비-마스크 영역만)
        feature_mask = cv2.bitwise_not(color_mask)
        akaze = cv2.AKAZE_create()

        kp1, desc1 = akaze.detectAndCompute(gray_tmpl, mask=feature_mask)
        kp2, desc2 = akaze.detectAndCompute(gray_frame, None)

        if desc1 is None or desc2 is None:
            logger.debug("AKAZE: 디스크립터 없음")
            return None

        if len(kp1) < 4 or len(kp2) < 4:
            logger.debug("AKAZE: 키포인트 부족 (tmpl=%d, frame=%d)", len(kp1), len(kp2))
            return None

        # 매칭
        bf = cv2.BFMatcher(cv2.NORM_HAMMING)
        try:
            raw_matches = bf.knnMatch(desc1, desc2, k=2)
        except cv2.error:
            logger.debug("AKAZE: knnMatch 실패")
            return None

        # Lowe's ratio test
        good = []
        for pair in raw_matches:
            if len(pair) == 2:
                m, n = pair
                if m.distance < 0.75 * n.distance:
                    good.append(m)

        logger.debug("AKAZE: %d/%d good matches", len(good), len(raw_matches))

        if len(good) < self.config.feature_match_min_count:
            return None

        # RANSAC homography
        src_pts = np.float32(
            [kp1[m.queryIdx].pt for m in good],
        ).reshape(-1, 1, 2)
        dst_pts = np.float32(
            [kp2[m.trainIdx].pt for m in good],
        ).reshape(-1, 1, 2)

        M, inlier_mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        if M is None:
            logger.debug("AKAZE: homography 추정 실패")
            return None

        # 인라이어 비율 계산
        inliers = int(inlier_mask.sum()) if inlier_mask is not None else 0
        inlier_ratio = inliers / len(good) if good else 0.0

        if inlier_ratio < 0.3:
            logger.debug("AKAZE: 인라이어 비율 %.2f 너무 낮음", inlier_ratio)
            return None

        # 템플릿 꼭짓점을 프레임 좌표로 변환
        tmpl_h, tmpl_w = template.shape[:2]
        corners = np.float32(
            [[0, 0], [tmpl_w, 0], [tmpl_w, tmpl_h], [0, tmpl_h]],
        ).reshape(-1, 1, 2)
        transformed = cv2.perspectiveTransform(corners, M)

        x, y, _, _ = cv2.boundingRect(transformed)
        return MatchResult(
            x=int(x),
            y=int(y),
            score=inlier_ratio,
            method="feature",
        )

    # ----------------------------------------
    # 3차: mask 기반 폴백
    # ----------------------------------------
    def _match_by_masked_correlation(
        self,
        template: NDArray[np.uint8],
        frame: NDArray[np.uint8],
        color_mask: NDArray[np.uint8],
    ) -> MatchResult | None:
        """mask 기반 TM_CCORR_NORMED 매칭 (NaN/Inf 가드 포함).

        1차, 2차 전략 모두 실패했을 때의 최후 폴백입니다.
        OpenCV mask 관련 버그에 대한 방어 코드를 포함합니다.
        """
        # 마스크 준비 (원본 영역=255, 색칠 영역=0)
        match_mask = cv2.bitwise_not(color_mask)

        tmpl_h, tmpl_w = template.shape[:2]
        frame_h, frame_w = frame.shape[:2]
        if tmpl_w > frame_w or tmpl_h > frame_h:
            return None

        # 유효 픽셀 비율 확인
        valid_ratio = np.count_nonzero(match_mask) / match_mask.size
        if valid_ratio < 0.1:
            logger.warning(
                "mask 폴백: 유효 영역 %.1f%% - 매칭 불안정",
                valid_ratio * 100,
            )
            return None

        # 3채널 마스크 확장
        match_mask_3ch = cv2.merge([match_mask, match_mask, match_mask])

        result = cv2.matchTemplate(
            frame,
            template,
            cv2.TM_CCORR_NORMED,
            mask=match_mask_3ch,
        )

        # NaN/Inf 가드 (OpenCV #15768, #23257 버그 대응)
        result = np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)
        result = np.clip(result, 0.0, 1.0)

        _, max_val, _, max_loc = cv2.minMaxLoc(result)

        logger.debug(
            "mask 폴백: score=%.4f (threshold=%.4f)",
            max_val,
            self.config.fallback_match_threshold,
        )

        if max_val < self.config.fallback_match_threshold:
            return None

        return MatchResult(
            x=max_loc[0],
            y=max_loc[1],
            score=max_val,
            method="masked",
        )

    # ----------------------------------------
    # 전처리 유틸리티
    # ----------------------------------------
    def _preprocess_for_edge_matching(
        self,
        image: NDArray[np.uint8],
        color_mask: NDArray[np.uint8] | None = None,
    ) -> NDArray[np.uint8]:
        """에지 매칭을 위한 전처리 파이프라인.

        색칠 영역을 검정으로 대체 → Grayscale → CLAHE → GaussianBlur → Canny

        Args:
            image: BGR 이미지
            color_mask: 색칠 영역 마스크 (있으면 해당 영역을 검정으로)

        Returns:
            Canny 에지 이미지 (단일 채널)
        """
        clean = image.copy()
        if color_mask is not None:
            # 색칠 영역을 검정으로 대체 → Canny가 해당 내부에 에지 미생성
            clean[color_mask > 0] = 0

        gray = cv2.cvtColor(clean, cv2.COLOR_BGR2GRAY)

        clahe = cv2.createCLAHE(
            clipLimit=self.config.clahe_clip_limit,
            tileGridSize=self.config.clahe_grid_size,
        )
        equalized = clahe.apply(gray)

        blurred = cv2.GaussianBlur(equalized, (5, 5), 0)

        edges = cv2.Canny(
            blurred,
            self.config.canny_threshold1,
            self.config.canny_threshold2,
        )
        return edges

    # ----------------------------------------
    # 좌표 변환 유틸리티 (변경 없음)
    # ----------------------------------------
    @staticmethod
    def _compute_scale(
        template_shape: tuple[int, ...],
        frame_shape: tuple[int, ...],
    ) -> tuple[float, float]:
        """템플릿과 프레임 간 스케일 비율을 계산합니다.

        동일 해상도면 (1.0, 1.0)을 반환합니다.
        """
        tmpl_h, tmpl_w = template_shape[:2]
        frame_h, frame_w = frame_shape[:2]

        # 동일 해상도 또는 부분 크롭인 경우 스케일 불필요
        # (크롭은 원본 해상도의 일부이므로 픽셀 비율이 동일)
        if tmpl_w <= frame_w and tmpl_h <= frame_h:
            return 1.0, 1.0

        # 템플릿이 프레임보다 큰 경우 (HiDPI 캡쳐 등)
        scale_x = frame_w / tmpl_w
        scale_y = frame_h / tmpl_h
        logger.info(
            "스케일 조정: template(%dx%d) → frame(%dx%d), scale=(%.3f, %.3f)",
            tmpl_w,
            tmpl_h,
            frame_w,
            frame_h,
            scale_x,
            scale_y,
        )
        return scale_x, scale_y

    @staticmethod
    def _clip_to_frame(
        bbox: BoundingBox,
        frame_h: int,
        frame_w: int,
    ) -> BoundingBox:
        """바운딩 박스를 프레임 경계 안으로 클리핑합니다."""
        x = max(0, min(bbox.x, frame_w - 1))
        y = max(0, min(bbox.y, frame_h - 1))
        w = min(bbox.width, frame_w - x)
        h = min(bbox.height, frame_h - y)
        return BoundingBox(x=x, y=y, width=max(1, w), height=max(1, h))

    # ----------------------------------------
    # ROI 병합 및 ID 할당 (변경 없음)
    # ----------------------------------------
    def _merge_overlapping_rois(self, rois: list[ROI]) -> list[ROI]:
        """IoU 기반으로 중복 ROI를 병합합니다."""
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
                    if rois[i].confidence >= rois[j].confidence:
                        keep[j] = False
                    else:
                        keep[i] = False
                        break

        return [roi for roi, k in zip(rois, keep) if k]

    @staticmethod
    def _assign_ids(rois: list[ROI]) -> list[ROI]:
        """ROI에 순차적 ID를 재할당합니다."""
        counters: dict[str, int] = {}
        result: list[ROI] = []

        for roi in rois:
            type_key = roi.roi_type.value
            idx = counters.get(type_key, 0)
            counters[type_key] = idx + 1

            result.append(
                ROI(
                    id=f"color_{type_key}_{idx}",
                    bbox=roi.bbox,
                    roi_type=roi.roi_type,
                    confidence=roi.confidence,
                    label=f"{type_key}_{idx}",
                    metadata=roi.metadata,
                )
            )

        return result
