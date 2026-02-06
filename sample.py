#!/usr/bin/env python
"""
파형(Waveform) 존재 여부 감지 독립 스크립트

이미지에서 오실로스코프 스타일의 파형이 존재하는지 판단합니다.
밝은 배경과 어두운 배경 모두 지원합니다.

사용법:
    python tests/test_waveform_detector.py <이미지_경로>
    python tests/test_waveform_detector.py tests/test_graph.png
    python tests/test_waveform_detector.py tests/test_graph.png --debug
    python tests/test_waveform_detector.py tests/test_graph.png --threshold 0.5

출력:
    - Boolean 결과 (True/False)
    - 신뢰도 점수 (0.0 ~ 1.0)
    - 디버그 모드: 중간 처리 이미지 저장
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List

import cv2
import numpy as np
from numpy.typing import NDArray


@dataclass
class WaveformDetectionResult:
    """파형 감지 결과"""

    detected: bool  # 파형 존재 여부
    confidence: float  # 신뢰도 (0.0 ~ 1.0)
    background_type: str  # 배경 타입 (dark/bright)
    edge_density: float  # 엣지 밀도
    component_count: int  # 유효 연결 성분 수
    horizontal_coverage: float  # 수평 커버리지

    def __str__(self) -> str:
        return (
            f"파형 감지 결과: {self.detected}\n"
            f"신뢰도: {self.confidence:.2f}\n"
            f"배경 타입: {self.background_type}\n"
            f"엣지 밀도: {self.edge_density:.4f}\n"
            f"유효 성분 수: {self.component_count}\n"
            f"수평 커버리지: {self.horizontal_coverage:.2f}"
        )


class WaveformDetector:
    """
    파형 존재 여부 감지기

    엣지 기반 연결 성분 분석을 사용하여 이미지에서 파형을 감지합니다.
    밝은/어두운 배경 모두 자동으로 처리합니다.

    알고리즘 파이프라인:
        입력 이미지 → 배경 타입 감지 → 전처리(CLAHE) → Canny 엣지 → 연결 성분 분석 → 파형 판정
    """

    def __init__(
        self,
        confidence_threshold: float = 0.6,
        min_edge_density: float = 0.01,
        max_edge_density: float = 0.30,
        min_component_area: int = 100,
        min_aspect_ratio: float = 3.0,
        min_horizontal_coverage: float = 0.3,
    ):
        """
        Args:
            confidence_threshold: 파형 판정 신뢰도 임계값 (기본 0.6)
            min_edge_density: 최소 엣지 밀도 (기본 1%)
            max_edge_density: 최대 엣지 밀도 (기본 30%)
            min_component_area: 유효 연결 성분의 최소 면적 (기본 100픽셀)
            min_aspect_ratio: 유효 연결 성분의 최소 가로/세로 비율 (기본 3.0)
            min_horizontal_coverage: 최소 수평 커버리지 (기본 30%)
        """
        self.confidence_threshold = confidence_threshold
        self.min_edge_density = min_edge_density
        self.max_edge_density = max_edge_density
        self.min_component_area = min_component_area
        self.min_aspect_ratio = min_aspect_ratio
        self.min_horizontal_coverage = min_horizontal_coverage

        # 디버그 이미지 저장용
        self._debug_images: dict[str, NDArray] = {}

    def detect(
        self,
        image: NDArray[np.uint8] | str | Path,
        debug: bool = False,
    ) -> WaveformDetectionResult:
        """
        파형 존재 여부 감지

        Args:
            image: 입력 이미지 (BGR numpy 배열) 또는 이미지 파일 경로
            debug: True면 중간 처리 이미지 저장

        Returns:
            WaveformDetectionResult: 감지 결과
        """
        # 1. 이미지 로드
        if isinstance(image, (str, Path)):
            img = cv2.imread(str(image))
            if img is None:
                raise FileNotFoundError(f"이미지를 로드할 수 없습니다: {image}")
        else:
            img = image

        # 그레이스케일 변환
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img.copy()

        # 2. 배경 타입 감지
        bg_type = self._detect_background_type(gray)

        # 3. 전처리
        preprocessed = self._preprocess(gray, bg_type)
        if debug:
            self._debug_images["preprocessed"] = preprocessed

        # 4. 엣지 감지
        edges = self._detect_edges(preprocessed)
        if debug:
            self._debug_images["edges"] = edges

        # 5. 연결 성분 분석
        components, closed_edges = self._analyze_components(edges)
        if debug:
            self._debug_images["closed_edges"] = closed_edges
            # 컴포넌트 시각화
            vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            for comp in components:
                color = (0, 255, 0) if self._is_valid_component(comp) else (0, 0, 255)
                cv2.rectangle(
                    vis,
                    (comp["x"], comp["y"]),
                    (comp["x"] + comp["w"], comp["y"] + comp["h"]),
                    color,
                    2,
                )
            self._debug_images["components"] = vis

        # 6. 신뢰도 계산 및 판정
        result = self._evaluate(edges, components, bg_type)

        return result

    def _detect_background_type(self, gray: NDArray[np.uint8]) -> str:
        """
        배경 타입 자동 감지

        평균 밝기를 기준으로 어두운/밝은 배경을 구분합니다.

        Args:
            gray: 그레이스케일 이미지

        Returns:
            "dark" 또는 "bright"
        """
        mean_intensity = np.mean(gray)
        return "dark" if mean_intensity < 127 else "bright"

    def _preprocess(
        self,
        gray: NDArray[np.uint8],
        bg_type: str,
    ) -> NDArray[np.uint8]:
        """
        전처리 (노이즈 제거 + 밝기 기반 이진화)

        노이즈가 많은 오실로스코프 이미지에 최적화되어 있습니다.
        밝은 파형만 추출하고 배경 노이즈를 제거합니다.

        Args:
            gray: 그레이스케일 이미지
            bg_type: 배경 타입 ("dark" 또는 "bright")

        Returns:
            전처리된 이미지 (이진화됨)
        """
        # 1. 가우시안 블러로 노이즈 감소
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # 2. 밝기 기반 이진화 (배경 타입에 따라 다르게 처리)
        if bg_type == "dark":
            # 어두운 배경: 밝은 픽셀(파형)만 추출
            # Otsu로 자동 임계값 계산
            _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        else:
            # 밝은 배경: 어두운 픽셀(파형)만 추출 후 반전
            _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # 3. 형태학적 열림 연산으로 작은 노이즈 제거
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

        # 4. 형태학적 닫힘으로 끊어진 파형 연결
        kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 3))
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel_close)

        return cleaned

    def _detect_edges(self, image: NDArray[np.uint8]) -> NDArray[np.uint8]:
        """
        엣지 감지 (이진화된 이미지에서 윤곽선 추출)

        전처리에서 이미 이진화가 완료되었으므로,
        직접 Canny를 적용하거나 이진 이미지를 그대로 사용합니다.

        Args:
            image: 전처리된 이미지 (이진화됨)

        Returns:
            엣지 이미지
        """
        # 이미 이진화된 이미지인 경우 그대로 반환
        # (전처리에서 Otsu 이진화 적용됨)
        unique_values = np.unique(image)
        if len(unique_values) <= 2:
            # 이진 이미지 - 그대로 사용
            return image

        # 그렇지 않으면 Canny 적용
        v = np.median(image)
        sigma = 0.33
        threshold1 = int(max(0, (1.0 - sigma) * v))
        threshold2 = int(min(255, (1.0 + sigma) * v))
        threshold1 = max(threshold1, 30)
        threshold2 = max(threshold2, threshold1 + 50)

        edges = cv2.Canny(image, threshold1, threshold2)
        return edges

    def _analyze_components(
        self,
        edges: NDArray[np.uint8],
    ) -> tuple[List[dict], NDArray[np.uint8]]:
        """
        연결 성분 분석

        강력한 수평 연결로 파형을 하나의 성분으로 인식합니다.

        Args:
            edges: 이진화 이미지

        Returns:
            (연결 성분 목록, 처리된 이미지)
        """
        height, width = edges.shape

        # 1단계: 강력한 수평 팽창으로 파형 전체를 연결
        # 이미지 너비의 5%를 커널 크기로 사용
        h_kernel_size = max(width // 20, 15)
        kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (h_kernel_size, 1))
        connected = cv2.dilate(edges, kernel_h, iterations=2)

        # 2단계: 수직 방향도 약간 팽창 (파형 두께 보정)
        kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 5))
        connected = cv2.dilate(connected, kernel_v, iterations=1)

        # 3단계: 닫힘 연산으로 내부 구멍 채우기
        kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        connected = cv2.morphologyEx(connected, cv2.MORPH_CLOSE, kernel_close)

        # 연결 성분 찾기
        contours, _ = cv2.findContours(
            connected, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        components = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = cv2.contourArea(contour)
            aspect_ratio = w / max(h, 1)

            # 원본 이진화 이미지에서 실제 파형 픽셀 수 계산
            mask = np.zeros(edges.shape, dtype=np.uint8)
            cv2.drawContours(mask, [contour], -1, 255, -1)
            pixel_area = np.sum((edges > 0) & (mask > 0))

            components.append(
                {
                    "x": x,
                    "y": y,
                    "w": w,
                    "h": h,
                    "area": area,
                    "pixel_area": pixel_area,
                    "aspect_ratio": aspect_ratio,
                }
            )

        # 면적 순으로 정렬 (가장 큰 성분이 파형일 가능성 높음)
        components.sort(key=lambda c: c["area"], reverse=True)

        return components, connected

    def _is_valid_component(self, component: dict, image_width: int = 0) -> bool:
        """
        유효한 파형 성분인지 확인

        파형의 특성:
        1. 수평으로 긴 형태 (aspect_ratio > 2)
        2. 일정 크기 이상 (노이즈 필터링)
        3. 이미지 너비의 일정 비율 이상 차지

        Args:
            component: 연결 성분 정보
            image_width: 이미지 너비 (커버리지 계산용)

        Returns:
            유효한 파형 성분이면 True
        """
        # 기본 조건: 면적과 가로/세로 비율
        area_ok = component["area"] >= self.min_component_area
        aspect_ok = component["aspect_ratio"] >= 2.0  # 파형은 가로로 길다

        # 추가 조건: 너비가 이미지의 10% 이상
        if image_width > 0:
            width_ratio = component["w"] / image_width
            width_ok = width_ratio >= 0.1
        else:
            width_ok = True

        return area_ok and aspect_ok and width_ok

    def _evaluate(
        self,
        edges: NDArray[np.uint8],
        components: List[dict],
        bg_type: str,
    ) -> WaveformDetectionResult:
        """
        신뢰도 계산 및 최종 판정

        세 가지 기준으로 신뢰도를 계산합니다:
        1. 신호 밀도 (20%) - 적절한 범위의 밝은 픽셀
        2. 유효 연결 성분 존재 (30%) - 파형 형태의 성분
        3. 수평 커버리지 (50%) - 가로로 충분히 퍼진 신호

        Args:
            edges: 이진화된 이미지
            components: 연결 성분 목록
            bg_type: 배경 타입

        Returns:
            WaveformDetectionResult
        """
        height, width = edges.shape

        # 1. 신호 밀도 계산 (밝은 픽셀 비율)
        edge_density = np.sum(edges > 0) / edges.size

        # 2. 유효 연결 성분 필터링 (이미지 너비 전달)
        valid_components = [
            c for c in components if self._is_valid_component(c, width)
        ]

        # 3. 수평 커버리지 계산
        if valid_components:
            min_x = min(c["x"] for c in valid_components)
            max_x = max(c["x"] + c["w"] for c in valid_components)
            horizontal_coverage = (max_x - min_x) / width
        else:
            horizontal_coverage = 0.0

        # 4. 신뢰도 계산
        scores = []

        # 신호 밀도 점수 (20%)
        # 파형 이미지: 보통 1% ~ 15% 범위
        if 0.005 <= edge_density <= 0.20:
            density_score = 0.2
        elif edge_density > 0.20:
            # 밀도가 높아도 부분 점수 부여
            density_score = 0.1
        else:
            density_score = 0.0
        scores.append(density_score)

        # 유효 성분 존재 점수 (30%)
        # 하나라도 유효한 파형 성분이 있으면 점수 부여
        if len(valid_components) >= 1:
            component_score = 0.3
        else:
            component_score = 0.0
        scores.append(component_score)

        # 수평 커버리지 점수 (50%)
        # 파형이 이미지 너비의 30% 이상을 차지하면 만점
        if horizontal_coverage >= 0.5:
            coverage_score = 0.5
        elif horizontal_coverage >= 0.3:
            coverage_score = 0.4
        elif horizontal_coverage >= 0.1:
            coverage_score = 0.2
        else:
            coverage_score = 0.0
        scores.append(coverage_score)

        confidence = sum(scores)
        detected = confidence >= self.confidence_threshold

        return WaveformDetectionResult(
            detected=detected,
            confidence=confidence,
            background_type=bg_type,
            edge_density=edge_density,
            component_count=len(valid_components),
            horizontal_coverage=horizontal_coverage,
        )

    def save_debug_images(self, output_dir: str | Path = ".") -> None:
        """
        디버그 이미지 저장

        Args:
            output_dir: 출력 디렉토리 (기본: 현재 디렉토리)
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        for name, img in self._debug_images.items():
            filepath = output_path / f"debug_{name}.png"
            cv2.imwrite(str(filepath), img)
            print(f"  - {filepath}")


def main():
    """CLI 진입점"""
    parser = argparse.ArgumentParser(
        description="파형(Waveform) 존재 여부 감지",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예시:
    python tests/test_waveform_detector.py tests/test_graph.png
    python tests/test_waveform_detector.py tests/test_graph.png --debug
    python tests/test_waveform_detector.py tests/test_graph.png --threshold 0.5
        """,
    )
    parser.add_argument("image_path", help="분석할 이미지 경로")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="디버그 모드 (중간 처리 이미지 저장)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.6,
        help="신뢰도 임계값 (기본: 0.6)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=".",
        help="디버그 이미지 출력 디렉토리 (기본: 현재 디렉토리)",
    )

    args = parser.parse_args()

    # 이미지 경로 확인
    image_path = Path(args.image_path)
    if not image_path.exists():
        print(f"오류: 이미지 파일을 찾을 수 없습니다: {image_path}", file=sys.stderr)
        sys.exit(2)

    # 감지기 생성 및 실행
    detector = WaveformDetector(confidence_threshold=args.threshold)

    try:
        result = detector.detect(image_path, debug=args.debug)
    except Exception as e:
        print(f"오류: 파형 감지 중 에러 발생: {e}", file=sys.stderr)
        sys.exit(2)

    # 결과 출력
    print(result)

    # 디버그 이미지 저장
    if args.debug:
        print("\n[DEBUG] 중간 이미지 저장됨:")
        detector.save_debug_images(args.output_dir)

    # 종료 코드 반환 (스크립트 연동용)
    # 0: 파형 감지됨, 1: 파형 미감지
    sys.exit(0 if result.detected else 1)


if __name__ == "__main__":
    main()
