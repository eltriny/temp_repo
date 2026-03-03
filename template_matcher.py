"""공유 캐스케이드 템플릿 매칭 유틸리티

ColorTemplateDetector와 AnchorDetector 등에서 공통으로 사용하는
템플릿 매칭 전략을 독립 모듈로 제공합니다.

매칭 전략 (캐스케이드):
    0차: 원본 픽셀 직접 매칭 (TM_CCOEFF_NORMED, 1:1 스케일, 임계값 0.90)
    1차: 에지 기반 매칭 (Canny + TM_CCOEFF_NORMED, 멀티스케일)
    2차: 특징점 기반 매칭 (AKAZE + RANSAC homography)
    3차: 마스크 기반 상관관계 (TM_CCORR_NORMED + NaN/Inf 가드)

사용 예시:
    >>> from detection.template_matcher import TemplateMatcher, MatcherConfig
    >>> matcher = TemplateMatcher()
    >>> result = matcher.cascade_match(template, frame)
    >>> result = matcher.cascade_match(template, frame, mask=color_mask)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import NamedTuple

import cv2
import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


# ========================================
# 매칭 결과 타입
# ========================================
class MatchResult(NamedTuple):
    """템플릿 매칭 결과

    Attributes:
        x: 매칭 위치 X 좌표 (프레임 기준)
        y: 매칭 위치 Y 좌표 (프레임 기준)
        score: 매칭 신뢰도 (0.0~1.0)
        method: 사용된 매칭 방법 ("raw", "edge", "feature", "masked")
    """

    x: int
    y: int
    score: float
    method: str  # "raw", "edge", "feature", "masked"


class MatchCandidate(NamedTuple):
    """다중 후보 매칭 결과 (Top-N 매칭용)

    Attributes:
        x: 매칭 위치 X 좌표
        y: 매칭 위치 Y 좌표
        score: 매칭 신뢰도 (0.0~1.0)
        method: 사용된 매칭 방법
        scale: 매칭된 스케일 팩터
    """

    x: int
    y: int
    score: float
    method: str
    scale: float


# ========================================
# 매칭 설정
# ========================================
@dataclass
class MatcherConfig:
    """캐스케이드 템플릿 매칭 설정

    Attributes:
        raw_pixel_match_enabled: Tier 0 원본 픽셀 직접 매칭 활성화 여부
        raw_pixel_match_threshold: 원본 픽셀 매칭 최소 신뢰도 (TM_CCOEFF_NORMED)
        edge_match_threshold: 에지 매칭 최소 신뢰도 (TM_CCOEFF_NORMED)
        canny_threshold1: Canny 에지 검출 하한
        canny_threshold2: Canny 에지 검출 상한
        clahe_clip_limit: CLAHE 대비 제한
        clahe_grid_size: CLAHE 그리드 크기
        multi_scale_factors: 멀티스케일 매칭 스케일 팩터
        feature_match_min_count: AKAZE 특징점 최소 매치 수
        feature_inlier_ratio_min: RANSAC 인라이어 최소 비율
        fallback_match_threshold: mask 폴백 매칭 임계값
    """

    # --- Tier 0: 원본 픽셀 직접 매칭 (빠른 경로) ---
    raw_pixel_match_enabled: bool = True
    raw_pixel_match_threshold: float = 0.90

    # --- Tier 1: 에지 기반 매칭 ---
    edge_match_threshold: float = 0.55
    canny_threshold1: int = 50
    canny_threshold2: int = 200
    clahe_clip_limit: float = 2.0
    clahe_grid_size: tuple[int, int] = (8, 8)
    multi_scale_factors: tuple[float, ...] = (1.0, 0.75, 0.5, 1.25, 1.5)
    feature_match_min_count: int = 8
    feature_inlier_ratio_min: float = 0.3
    fallback_match_threshold: float = 0.85

    # 소형 스니펫용 AKAZE 완화
    feature_match_min_count_small: int = 4
    small_template_threshold: int = 100

    # 전처리 견고성
    auto_canny: bool = True
    multi_clahe: bool = False
    clahe_clip_limits: tuple[float, ...] = (2.0, 4.0, 8.0)
    min_edge_density: float = 0.01

    # 적응적 스케일 범위
    adaptive_scale: bool = False
    adaptive_scale_steps: int = 5
    adaptive_scale_step_size: float = 0.05

    # 다중 후보 매칭
    top_n_candidates: int = 5
    candidate_nms_radius: int = 20


# ========================================
# 캐스케이드 템플릿 매처
# ========================================
class TemplateMatcher:
    """캐스케이드 전략 기반 템플릿 매칭 엔진

    원본 픽셀 → 에지 → 특징점 → 마스크 상관의 4단계 폴백으로
    다양한 조건에서 견고한 매칭을 수행합니다.

    Args:
        config: 매칭 파라미터 설정. None이면 기본값 사용.
    """

    def __init__(self, config: MatcherConfig | None = None) -> None:
        self.config = config or MatcherConfig()
        # CLAHE 객체 캐시 (매 호출마다 재생성 방지)
        self._clahe = cv2.createCLAHE(
            clipLimit=self.config.clahe_clip_limit,
            tileGridSize=self.config.clahe_grid_size,
        )

    # ========================================
    # 공개 API
    # ========================================
    def cascade_match(
        self,
        template: NDArray[np.uint8],
        frame: NDArray[np.uint8],
        mask: NDArray[np.uint8] | None = None,
    ) -> MatchResult | None:
        """캐스케이드 전략으로 프레임 내 템플릿 위치를 찾습니다.

        0차: 원본 픽셀 직접 매칭 (마스크 없는 경우만, 빠른 경로)
        1차: 에지 기반 매칭 (멀티스케일)
        2차: AKAZE 특징점 매칭
        3차: mask 기반 상관관계 폴백

        Args:
            template: 템플릿 이미지 (BGR)
            frame: 대상 프레임 (BGR)
            mask: 템플릿 내 무시할 영역 마스크 (마스크=255 → 무시).
                  None이면 마스크 없이 매칭합니다.

        Returns:
            MatchResult 또는 매칭 실패 시 None
        """
        # mask=None은 각 하위 메서드에서 자체 처리 (불필요한 제로 배열 생성 방지)

        # --- 0차: 원본 픽셀 직접 매칭 (빠른 경로) ---
        # 마스크가 없는 깨끗한 템플릿에서만 시도
        if self.config.raw_pixel_match_enabled:
            has_mask = mask is not None and np.any(mask > 0)
            if not has_mask:
                raw_result = self.match_by_raw_pixels(template, frame)
                if raw_result is not None:
                    logger.info(
                        "[0차 raw] 성공: score=%.4f at (%d,%d)",
                        raw_result.score,
                        raw_result.x,
                        raw_result.y,
                    )
                    return raw_result
                logger.info("[0차 raw] 임계값 미달, 1차 에지 시도")

        # --- 1차: 에지 기반 멀티스케일 매칭 ---
        best_edge = self.match_by_edges(template, frame, mask)
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
        feat_result = self.match_by_features(template, frame, mask)
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
        mask_result = self.match_by_masked_correlation(template, frame, mask)
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

    def match_by_raw_pixels(
        self,
        template: NDArray[np.uint8],
        frame: NDArray[np.uint8],
    ) -> MatchResult | None:
        """원본 픽셀 직접 매칭 (Tier 0 빠른 경로).

        전처리 없이 원본 BGR 이미지에서 TM_CCOEFF_NORMED로 매칭합니다.
        동일 해상도 화면 캡처처럼 이상적인 조건에서 빠르게 매칭하기 위한
        1:1 스케일 전용 메서드입니다.

        Args:
            template: 템플릿 이미지 (BGR)
            frame: 대상 프레임 (BGR)

        Returns:
            MatchResult 또는 매칭 실패 시 None
        """
        tmpl_h, tmpl_w = template.shape[:2]
        frame_h, frame_w = frame.shape[:2]

        if tmpl_w > frame_w or tmpl_h > frame_h:
            return None

        result = cv2.matchTemplate(
            frame,
            template,
            cv2.TM_CCOEFF_NORMED,
        )
        _, max_val, _, max_loc = cv2.minMaxLoc(result)

        logger.debug(
            "Tier 0 raw pixel: score=%.4f (threshold=%.4f)",
            max_val,
            self.config.raw_pixel_match_threshold,
        )

        if max_val < self.config.raw_pixel_match_threshold:
            return None

        return MatchResult(
            x=max_loc[0],
            y=max_loc[1],
            score=max_val,
            method="raw",
        )

    def match_by_edges(
        self,
        template: NDArray[np.uint8],
        frame: NDArray[np.uint8],
        mask: NDArray[np.uint8] | None = None,
    ) -> MatchResult | None:
        """에지 기반 멀티스케일 매칭.

        마스크 영역을 검정으로 대체한 뒤 Canny 에지를 추출하여
        TM_CCOEFF_NORMED로 매칭합니다.

        Args:
            template: 템플릿 이미지 (BGR)
            frame: 대상 프레임 (BGR)
            mask: 무시할 영역 마스크 (마스크=255 → 검정 대체). None 허용.

        Returns:
            MatchResult 또는 매칭 실패 시 None
        """
        tmpl_edges = self._preprocess_for_edge_matching(template, mask)
        tmpl_h, tmpl_w = tmpl_edges.shape[:2]

        # 에지가 거의 없으면 스킵
        tmpl_edge_ratio = np.count_nonzero(tmpl_edges) / tmpl_edges.size
        if tmpl_edge_ratio < 0.01:
            logger.debug(
                "템플릿 에지 비율 %.2f%% - 에지 매칭 스킵", tmpl_edge_ratio * 100
            )
            return None

        best: MatchResult | None = None

        scales = self._get_scales(template, frame)
        for scale in scales:
            if scale == 1.0:
                frame_scaled = frame
            else:
                # 다운스케일은 INTER_AREA (품질 우수), 업스케일은 INTER_LINEAR 사용
                interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
                frame_scaled = cv2.resize(
                    frame,
                    None,
                    fx=scale,
                    fy=scale,
                    interpolation=interp,
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

            # 원본 스케일로 좌표 변환 (반올림으로 정밀도 향상)
            orig_x = round(max_loc[0] / scale) if scale != 1.0 else max_loc[0]
            orig_y = round(max_loc[1] / scale) if scale != 1.0 else max_loc[1]

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

    def match_by_features(
        self,
        template: NDArray[np.uint8],
        frame: NDArray[np.uint8],
        mask: NDArray[np.uint8] | None = None,
    ) -> MatchResult | None:
        """AKAZE 특징점 매칭으로 템플릿 위치를 추정합니다.

        마스크 영역을 제외한 원본 콘텐츠의 특징점만 추출하고,
        RANSAC homography로 대응 관계를 추정합니다.

        Args:
            template: 템플릿 이미지 (BGR)
            frame: 대상 프레임 (BGR)
            mask: 무시할 영역 마스크 (마스크=255 → 제외). None이면 전체 사용.

        Returns:
            MatchResult 또는 매칭 실패 시 None
        """
        gray_tmpl = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        feature_mask = None
        if mask is not None and np.any(mask > 0):
            feature_mask = cv2.bitwise_not(mask)

        result = self._akaze_match(gray_tmpl, gray_frame, feature_mask)
        if result is None:
            return None

        M, inlier_ratio = result

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

    def match_by_masked_correlation(
        self,
        template: NDArray[np.uint8],
        frame: NDArray[np.uint8],
        mask: NDArray[np.uint8] | None = None,
    ) -> MatchResult | None:
        """mask 기반 TM_CCORR_NORMED 매칭 (NaN/Inf 가드 포함).

        1차, 2차 전략 모두 실패했을 때의 최후 폴백입니다.

        Args:
            template: 템플릿 이미지 (BGR)
            frame: 대상 프레임 (BGR)
            mask: 무시할 영역 마스크 (마스크=255 → 제외). None이면 마스크 없이 매칭.

        Returns:
            MatchResult 또는 매칭 실패 시 None
        """
        tmpl_h, tmpl_w = template.shape[:2]
        frame_h, frame_w = frame.shape[:2]
        if tmpl_w > frame_w or tmpl_h > frame_h:
            return None

        # 마스크가 있으면 반전하여 유효 영역 마스크 생성
        if mask is not None and np.any(mask > 0):
            match_mask = cv2.bitwise_not(mask)

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
        else:
            # 마스크 없이 직접 매칭
            result = cv2.matchTemplate(
                frame,
                template,
                cv2.TM_CCORR_NORMED,
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

    # ========================================
    # 스케일 및 다중 후보 유틸리티
    # ========================================
    def _get_scales(
        self,
        template: NDArray[np.uint8],
        frame: NDArray[np.uint8],
    ) -> tuple[float, ...]:
        """현재 설정에 따라 스케일 팩터 목록을 반환합니다."""
        if self.config.adaptive_scale:
            return self._compute_adaptive_scales(template, frame)
        return self.config.multi_scale_factors

    def _compute_adaptive_scales(
        self,
        template: NDArray[np.uint8],
        frame: NDArray[np.uint8],
    ) -> tuple[float, ...]:
        """적응적 스케일 범위를 생성합니다.

        고정 스케일 목록과 중심 스케일(1.0) 기준 ±steps 범위를
        병합하여 더 촘촘한 스케일 탐색을 수행합니다.
        """
        center_scale = 1.0
        steps = self.config.adaptive_scale_steps
        step_size = self.config.adaptive_scale_step_size

        scales: set[float] = set()
        for i in range(-steps, steps + 1):
            s = center_scale + i * step_size
            if s > 0.1:
                scales.add(round(s, 4))

        # 고정 스케일도 포함
        for s in self.config.multi_scale_factors:
            scales.add(s)

        return tuple(sorted(scales))

    def match_by_edges_topn(
        self,
        template: NDArray[np.uint8],
        frame: NDArray[np.uint8],
        mask: NDArray[np.uint8] | None = None,
        top_n: int | None = None,
    ) -> list[MatchCandidate]:
        """에지 기반 Top-N 후보 매칭 (NMS 포함).

        단일 최고점 대신 모든 로컬 맥시마를 탐지하고,
        Non-Maximum Suppression을 적용하여 상위 N개 후보를 반환합니다.

        Args:
            template: 템플릿 이미지 (BGR)
            frame: 대상 프레임 (BGR)
            mask: 무시할 영역 마스크. None 허용.
            top_n: 반환할 후보 수. None이면 config 사용.

        Returns:
            NMS 적용된 상위 N개 MatchCandidate 리스트 (score 내림차순)
        """
        top_n = top_n or self.config.top_n_candidates

        tmpl_edges = self._preprocess_for_edge_matching(template, mask)
        tmpl_h, tmpl_w = tmpl_edges.shape[:2]

        tmpl_edge_ratio = np.count_nonzero(tmpl_edges) / tmpl_edges.size
        if tmpl_edge_ratio < 0.01:
            return []

        all_candidates: list[MatchCandidate] = []
        scales = self._get_scales(template, frame)

        for scale in scales:
            if scale == 1.0:
                frame_scaled = frame
            else:
                interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
                frame_scaled = cv2.resize(
                    frame, None, fx=scale, fy=scale, interpolation=interp
                )

            scaled_h, scaled_w = frame_scaled.shape[:2]
            if tmpl_w > scaled_w or tmpl_h > scaled_h:
                continue

            frame_edges = self._preprocess_for_edge_matching(frame_scaled)
            result_map = cv2.matchTemplate(
                frame_edges, tmpl_edges, cv2.TM_CCOEFF_NORMED
            )

            # dilation 기반 로컬 맥시마 탐지
            kernel_size = max(3, self.config.candidate_nms_radius)
            kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
            local_max = cv2.dilate(result_map, kernel)
            peaks = (result_map == local_max) & (
                result_map >= self.config.edge_match_threshold
            )

            peak_locs = np.where(peaks)
            for py, px in zip(peak_locs[0], peak_locs[1]):
                score = float(result_map[py, px])
                orig_x = round(int(px) / scale) if scale != 1.0 else int(px)
                orig_y = round(int(py) / scale) if scale != 1.0 else int(py)
                all_candidates.append(
                    MatchCandidate(
                        x=orig_x,
                        y=orig_y,
                        score=score,
                        method="edge",
                        scale=scale,
                    )
                )

        # score 내림차순 정렬 + NMS
        all_candidates.sort(key=lambda c: c.score, reverse=True)
        nms_candidates = self._apply_nms(
            all_candidates, self.config.candidate_nms_radius
        )

        return nms_candidates[:top_n]

    @staticmethod
    def _apply_nms(
        candidates: list[MatchCandidate],
        radius: int,
    ) -> list[MatchCandidate]:
        """Non-Maximum Suppression: 가까운 위치의 중복 후보를 제거합니다."""
        kept: list[MatchCandidate] = []
        for c in candidates:
            suppressed = False
            for k in kept:
                if abs(c.x - k.x) < radius and abs(c.y - k.y) < radius:
                    suppressed = True
                    break
            if not suppressed:
                kept.append(c)
        return kept

    # ========================================
    # 전처리 유틸리티
    # ========================================
    def _preprocess_for_edge_matching(
        self,
        image: NDArray[np.uint8],
        mask: NDArray[np.uint8] | None = None,
    ) -> NDArray[np.uint8]:
        """에지 매칭을 위한 전처리 파이프라인 (폴백 포함).

        기본 전처리 → 에지 밀도 부족 시 Auto-Canny → Multi-CLAHE 순서로 시도합니다.

        Args:
            image: BGR 이미지
            mask: 무시할 영역 마스크 (있으면 해당 영역을 검정으로)

        Returns:
            Canny 에지 이미지 (단일 채널)
        """
        # 1차: 기본 전처리
        edges = self._preprocess_primary(image, mask)

        edge_density = np.count_nonzero(edges) / edges.size
        if edge_density >= self.config.min_edge_density:
            return edges

        # 2차: Auto-Canny (Otsu 기반 자동 임계값)
        if self.config.auto_canny:
            edges = self._preprocess_auto_canny(image, mask)
            edge_density = np.count_nonzero(edges) / edges.size
            if edge_density >= self.config.min_edge_density:
                logger.debug("Auto-Canny 폴백 적용: edge density=%.4f", edge_density)
                return edges

        # 3차: Multi-CLAHE (다른 clip limit으로 재시도)
        if self.config.multi_clahe:
            for clip_limit in self.config.clahe_clip_limits:
                if clip_limit == self.config.clahe_clip_limit:
                    continue  # 이미 시도함
                edges = self._preprocess_with_clahe(image, mask, clip_limit)
                edge_density = np.count_nonzero(edges) / edges.size
                if edge_density >= self.config.min_edge_density:
                    logger.debug(
                        "Multi-CLAHE 폴백 (clip=%.1f): density=%.4f",
                        clip_limit,
                        edge_density,
                    )
                    return edges

        # 모든 폴백 실패 시 1차 결과 반환
        return self._preprocess_primary(image, mask)

    def _preprocess_primary(
        self,
        image: NDArray[np.uint8],
        mask: NDArray[np.uint8] | None = None,
    ) -> NDArray[np.uint8]:
        """기본 에지 전처리: 마스크 → Grayscale → CLAHE → GaussianBlur → Canny"""
        clean = image.copy()
        if mask is not None and np.any(mask > 0):
            clean[mask > 0] = 0

        gray = cv2.cvtColor(clean, cv2.COLOR_BGR2GRAY)
        equalized = self._clahe.apply(gray)
        blurred = cv2.GaussianBlur(equalized, (5, 5), 0)
        edges = cv2.Canny(
            blurred,
            self.config.canny_threshold1,
            self.config.canny_threshold2,
        )
        return edges

    def _preprocess_auto_canny(
        self,
        image: NDArray[np.uint8],
        mask: NDArray[np.uint8] | None = None,
    ) -> NDArray[np.uint8]:
        """Otsu 기반 자동 Canny 임계값 전처리.

        이미지 히스토그램에서 최적 분리점을 찾아 Canny 임계값을
        자동 결정합니다. 조명/대비 변화에 적응적입니다.
        """
        clean = image.copy()
        if mask is not None and np.any(mask > 0):
            clean[mask > 0] = 0

        gray = cv2.cvtColor(clean, cv2.COLOR_BGR2GRAY)
        equalized = self._clahe.apply(gray)
        blurred = cv2.GaussianBlur(equalized, (5, 5), 0)

        # Otsu 임계값으로 자동 Canny 범위 결정
        otsu_thresh, _ = cv2.threshold(
            blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        low_thresh = max(10, int(otsu_thresh * 0.5))
        high_thresh = min(250, int(otsu_thresh))

        edges = cv2.Canny(blurred, low_thresh, high_thresh)
        return edges

    def _preprocess_with_clahe(
        self,
        image: NDArray[np.uint8],
        mask: NDArray[np.uint8] | None = None,
        clip_limit: float = 4.0,
    ) -> NDArray[np.uint8]:
        """지정된 CLAHE clip limit으로 전처리."""
        alt_clahe = cv2.createCLAHE(
            clipLimit=clip_limit,
            tileGridSize=self.config.clahe_grid_size,
        )
        clean = image.copy()
        if mask is not None and np.any(mask > 0):
            clean[mask > 0] = 0

        gray = cv2.cvtColor(clean, cv2.COLOR_BGR2GRAY)
        equalized = alt_clahe.apply(gray)
        blurred = cv2.GaussianBlur(equalized, (5, 5), 0)
        edges = cv2.Canny(
            blurred,
            self.config.canny_threshold1,
            self.config.canny_threshold2,
        )
        return edges

    # ========================================
    # 호모그래피 기반 ROI 변환 (앵커 탐지 지원)
    # ========================================
    def compute_homography(
        self,
        reference: NDArray[np.uint8],
        target: NDArray[np.uint8],
        mask: NDArray[np.uint8] | None = None,
    ) -> NDArray[np.float64] | None:
        """참조 이미지와 대상 이미지 간의 호모그래피 행렬을 계산합니다.

        AnchorDetector의 호모그래피 폴백에서 사용됩니다.

        Args:
            reference: 참조 이미지 (BGR)
            target: 대상 이미지 (BGR)
            mask: 참조 이미지 내 무시할 영역 (마스크=255 → 제외)

        Returns:
            3x3 호모그래피 행렬 또는 실패 시 None
        """
        gray_ref = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
        gray_target = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)

        feature_mask = None
        if mask is not None and np.any(mask > 0):
            feature_mask = cv2.bitwise_not(mask)

        result = self._akaze_match(gray_ref, gray_target, feature_mask)
        if result is None:
            return None

        M, _ = result
        return M

    # ========================================
    # 내부 헬퍼: AKAZE 특징점 매칭 파이프라인
    # ========================================
    def _akaze_match(
        self,
        gray1: NDArray[np.uint8],
        gray2: NDArray[np.uint8],
        mask1: NDArray[np.uint8] | None = None,
    ) -> tuple[NDArray[np.float64], float] | None:
        """AKAZE + BFMatcher + Lowe's ratio test + RANSAC homography 파이프라인.

        match_by_features()와 compute_homography()의 공유 구현입니다.

        Args:
            gray1: 소스 그레이스케일 이미지
            gray2: 대상 그레이스케일 이미지
            mask1: 소스 이미지 특징점 추출 마스크 (255=추출 대상)

        Returns:
            (호모그래피 행렬, 인라이어 비율) 튜플 또는 실패 시 None
        """
        akaze = cv2.AKAZE_create()

        kp1, desc1 = akaze.detectAndCompute(gray1, mask=mask1)
        kp2, desc2 = akaze.detectAndCompute(gray2, None)

        if desc1 is None or desc2 is None:
            logger.debug("AKAZE: 디스크립터 없음")
            return None

        if len(kp1) < 4 or len(kp2) < 4:
            logger.debug("AKAZE: 키포인트 부족 (src=%d, dst=%d)", len(kp1), len(kp2))
            return None

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

        # 소형 템플릿은 최소 매치 수 완화
        min_count = self.config.feature_match_min_count
        if self.config.small_template_threshold > 0:
            h1, w1 = gray1.shape[:2]
            if max(h1, w1) < self.config.small_template_threshold:
                min_count = self.config.feature_match_min_count_small
                logger.debug(
                    "소형 템플릿 (%dx%d): 최소 매치 수 완화 %d → %d",
                    w1,
                    h1,
                    self.config.feature_match_min_count,
                    min_count,
                )

        if len(good) < min_count:
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

        inliers = int(inlier_mask.sum()) if inlier_mask is not None else 0
        inlier_ratio = inliers / len(good) if good else 0.0

        if inlier_ratio < self.config.feature_inlier_ratio_min:
            logger.debug("AKAZE: 인라이어 비율 %.2f 너무 낮음", inlier_ratio)
            return None

        return M, inlier_ratio
