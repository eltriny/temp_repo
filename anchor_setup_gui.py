"""앵커 설정 GUI 도구

참조 프레임(이미지/비디오 첫 프레임)을 열고,
마우스로 앵커 영역과 ROI 영역을 지정하여
앵커 설정 YAML 파일을 자동 생성합니다.

사용법:
    python -m src.tools.anchor_setup_gui --image reference.png --output config/anchors.yaml
    python -m src.tools.anchor_setup_gui --video input.mp4 --output config/anchors.yaml

조작법:
    - 좌클릭 + 드래그: 영역 선택
    - 'a': 선택한 영역을 앵커로 등록 (스니펫 이미지 저장)
    - 'r': 선택한 영역을 ROI로 등록 (마지막 앵커에 매핑)
    - 's': 설정을 YAML 파일로 저장
    - 'u': 마지막 등록 취소 (undo)
    - 'h': 도움말 오버레이 토글
    - 'q' / ESC: 종료
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray

try:
    import yaml

    _HAS_YAML = True
except ImportError:
    yaml = None  # type: ignore[assignment]
    _HAS_YAML = False

logger = logging.getLogger(__name__)

# ========================================
# 색상 상수 (BGR)
# ========================================
COLOR_ANCHOR = (0, 200, 0)  # 초록 - 앵커
COLOR_ROI = (0, 0, 220)  # 빨강 - ROI
COLOR_SELECTION = (220, 150, 0)  # 파랑 - 선택 중
COLOR_CONNECTION = (180, 180, 180)  # 회색 - 연결선
COLOR_STATUS_BG = (40, 40, 40)  # 상태바 배경
COLOR_STATUS_TEXT = (220, 220, 220)  # 상태바 텍스트
COLOR_HELP_BG = (30, 30, 30)  # 도움말 배경
COLOR_HELP_TEXT = (200, 200, 200)  # 도움말 텍스트

WINDOW_NAME = "Anchor Setup GUI"
STATUS_BAR_HEIGHT = 40
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE_LABEL = 0.5
FONT_SCALE_STATUS = 0.55
FONT_SCALE_HELP = 0.5
FONT_THICKNESS = 1


# ========================================
# 데이터 타입 정의
# ========================================
@dataclass
class AnchorEntry:
    """등록된 앵커 정보

    Attributes:
        name: 앵커 식별 이름 (예: "anchor_0")
        bbox: 앵커 영역 좌표 (x, y, w, h)
        snippet_path: 저장된 스니펫 이미지 파일 경로
    """

    name: str
    bbox: tuple[int, int, int, int]  # x, y, w, h
    snippet_path: Path | None = None


@dataclass
class ROIMappingEntry:
    """등록된 ROI 매핑 정보

    Attributes:
        anchor_name: 연결된 앵커 이름
        roi_name: ROI 식별 이름
        bbox: ROI 영역 좌표 (x, y, w, h)
        roi_type: ROI 유형 ("numeric" | "text" | "chart")
    """

    anchor_name: str
    roi_name: str
    bbox: tuple[int, int, int, int]  # x, y, w, h
    roi_type: str = "numeric"


# ========================================
# Undo 히스토리 항목
# ========================================
@dataclass
class HistoryEntry:
    """되돌리기 히스토리 항목

    Attributes:
        action: 수행된 동작 ("anchor" 또는 "roi")
        anchor_index: 관련 앵커의 인덱스 (앵커 리스트 내)
        roi_index: 관련 ROI의 인덱스 (ROI 리스트 내, roi 동작일 때만)
    """

    action: str  # "anchor" | "roi"
    anchor_index: int
    roi_index: int | None = None


# ========================================
# 메인 GUI 클래스
# ========================================
class AnchorSetupGUI:
    """앵커 기반 ROI 설정을 위한 시각적 GUI 도구

    OpenCV HighGUI를 사용하여 참조 프레임 위에서 마우스로
    앵커 영역과 ROI 영역을 선택하고, YAML 설정 파일을 생성합니다.

    Attributes:
        image: 원본 참조 프레임 (BGR)
        output_dir: 설정 파일 및 스니펫 저장 디렉토리
        anchors: 등록된 앵커 목록
        roi_mappings: 등록된 ROI 매핑 목록
    """

    def __init__(self, image: NDArray, output_dir: Path) -> None:
        self.image = image.copy()
        self.output_dir = output_dir
        self.anchors: list[AnchorEntry] = []
        self.roi_mappings: list[ROIMappingEntry] = []
        self._undo_stack: list[HistoryEntry] = []

        # 마우스 드래그 상태
        self._dragging = False
        self._drag_start: tuple[int, int] = (0, 0)
        self._drag_end: tuple[int, int] = (0, 0)
        self._selection: tuple[int, int, int, int] | None = None  # x, y, w, h

        # 표시 상태
        self._show_help = False
        self._status_message = (
            "Ready. Drag to select a region, then press 'a' (anchor) or 'r' (ROI)."
        )

        # 스니펫 저장 디렉토리
        self._snippet_dir = output_dir / "anchors"

        # ROI 타입 순환 목록
        self._roi_types = ["numeric", "text", "chart"]
        self._current_roi_type_index = 0

    # ========================================
    # 공개 API
    # ========================================
    def run(self) -> None:
        """메인 GUI 루프 실행

        OpenCV 윈도우를 생성하고 키보드/마우스 이벤트를 처리합니다.
        'q' 또는 ESC 키로 종료합니다.
        """
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(WINDOW_NAME, self._mouse_callback)

        # 초기 윈도우 크기를 이미지 비율에 맞게 조정
        img_h, img_w = self.image.shape[:2]
        display_w = min(img_w, 1600)
        scale = display_w / img_w
        display_h = int(img_h * scale) + STATUS_BAR_HEIGHT
        cv2.resizeWindow(WINDOW_NAME, display_w, display_h)

        while True:
            display = self._draw_overlay()
            cv2.imshow(WINDOW_NAME, display)

            key = cv2.waitKey(30) & 0xFF

            if key == ord("q") or key == 27:  # q 또는 ESC
                break
            elif key == ord("a"):
                self._register_anchor()
            elif key == ord("r"):
                self._register_roi()
            elif key == ord("s"):
                output_path = self.output_dir / "anchors.yaml"
                self._save_yaml(output_path)
            elif key == ord("u"):
                self._undo_last()
            elif key == ord("h"):
                self._show_help = not self._show_help
            elif key == ord("t"):
                self._cycle_roi_type()

            # 윈도우가 닫혔는지 확인
            if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                break

        cv2.destroyAllWindows()

    # ========================================
    # 마우스 이벤트 핸들러
    # ========================================
    def _mouse_callback(
        self, event: int, x: int, y: int, flags: int, param: object
    ) -> None:
        """마우스 이벤트 핸들러 (드래그로 영역 선택)

        좌클릭 + 드래그로 사각형 영역을 선택합니다.
        선택된 영역은 파란색 사각형으로 표시됩니다.

        Args:
            event: OpenCV 마우스 이벤트 타입
            x: 마우스 x 좌표 (이미지 기준)
            y: 마우스 y 좌표 (이미지 기준)
            flags: 이벤트 플래그
            param: 콜백 파라미터 (사용하지 않음)
        """
        img_h, img_w = self.image.shape[:2]

        # 좌표를 이미지 범위 내로 클리핑
        x = max(0, min(x, img_w - 1))
        y = max(0, min(y, img_h - 1))

        if event == cv2.EVENT_LBUTTONDOWN:
            self._dragging = True
            self._drag_start = (x, y)
            self._drag_end = (x, y)
            self._selection = None

        elif event == cv2.EVENT_MOUSEMOVE:
            if self._dragging:
                self._drag_end = (x, y)

        elif event == cv2.EVENT_LBUTTONUP:
            if self._dragging:
                self._dragging = False
                self._drag_end = (x, y)

                # 선택 영역 계산 (최소 크기 5x5 보장)
                x1 = min(self._drag_start[0], self._drag_end[0])
                y1 = min(self._drag_start[1], self._drag_end[1])
                x2 = max(self._drag_start[0], self._drag_end[0])
                y2 = max(self._drag_start[1], self._drag_end[1])

                w = x2 - x1
                h = y2 - y1

                if w >= 5 and h >= 5:
                    self._selection = (x1, y1, w, h)
                    self._status_message = (
                        f"Selected: ({x1}, {y1}, {w}x{h}). "
                        f"Press 'a' to register as anchor, "
                        f"'r' to register as ROI (type: {self._current_roi_type})."
                    )
                else:
                    self._selection = None
                    self._status_message = (
                        "Selection too small. Drag a larger area (min 5x5)."
                    )

    # ========================================
    # 앵커 등록
    # ========================================
    def _register_anchor(self) -> None:
        """선택된 영역을 앵커로 등록

        선택 영역을 스니펫 이미지로 저장하고 앵커 목록에 추가합니다.
        """
        if self._selection is None:
            self._status_message = "No region selected. Drag to select first."
            return

        sx, sy, sw, sh = self._selection

        # 앵커 이름 자동 생성
        anchor_index = len(self.anchors)
        name = f"anchor_{anchor_index}"

        # 스니펫 이미지 저장
        self._snippet_dir.mkdir(parents=True, exist_ok=True)
        snippet_filename = f"{name}.png"
        snippet_path = self._snippet_dir / snippet_filename

        snippet = self.image[sy : sy + sh, sx : sx + sw].copy()
        cv2.imwrite(str(snippet_path), snippet)

        entry = AnchorEntry(
            name=name,
            bbox=(sx, sy, sw, sh),
            snippet_path=snippet_path,
        )
        self.anchors.append(entry)

        # Undo 히스토리 추가
        self._undo_stack.append(
            HistoryEntry(action="anchor", anchor_index=anchor_index)
        )

        self._selection = None
        self._status_message = (
            f"Anchor '{name}' registered at ({sx}, {sy}, {sw}x{sh}). "
            f"Snippet saved to {snippet_path.name}."
        )
        logger.info("Anchor registered: %s at (%d,%d,%d,%d)", name, sx, sy, sw, sh)

    # ========================================
    # ROI 등록
    # ========================================
    def _register_roi(self) -> None:
        """선택된 영역을 ROI로 등록 (마지막 앵커에 매핑)

        선택 영역과 마지막 등록된 앵커 간의 정규화 오프셋을 계산하고
        ROI 매핑 목록에 추가합니다.
        """
        if self._selection is None:
            self._status_message = "No region selected. Drag to select first."
            return

        if not self.anchors:
            self._status_message = (
                "No anchors registered yet. Register an anchor first ('a')."
            )
            return

        last_anchor = self.anchors[-1]
        sx, sy, sw, sh = self._selection

        # ROI 이름 자동 생성 (앵커별 순차 번호)
        anchor_roi_count = sum(
            1 for m in self.roi_mappings if m.anchor_name == last_anchor.name
        )
        roi_name = f"{last_anchor.name}_roi_{anchor_roi_count}"

        roi_type = self._roi_types[self._current_roi_type_index]

        entry = ROIMappingEntry(
            anchor_name=last_anchor.name,
            roi_name=roi_name,
            bbox=(sx, sy, sw, sh),
            roi_type=roi_type,
        )
        self.roi_mappings.append(entry)

        roi_index = len(self.roi_mappings) - 1
        anchor_index = len(self.anchors) - 1

        # Undo 히스토리 추가
        self._undo_stack.append(
            HistoryEntry(action="roi", anchor_index=anchor_index, roi_index=roi_index)
        )

        self._selection = None
        self._status_message = (
            f"ROI '{roi_name}' registered (type: {roi_type}), "
            f"mapped to anchor '{last_anchor.name}'."
        )
        logger.info(
            "ROI registered: %s -> %s at (%d,%d,%d,%d), type=%s",
            roi_name,
            last_anchor.name,
            sx,
            sy,
            sw,
            sh,
            roi_type,
        )

    # ========================================
    # 정규화 오프셋 계산
    # ========================================
    def _compute_normalized_offset(
        self,
        anchor_bbox: tuple[int, int, int, int],
        roi_bbox: tuple[int, int, int, int],
        frame_shape: tuple[int, ...],
    ) -> dict[str, float]:
        """앵커 bbox 기준 정규화 오프셋 계산

        앵커의 좌상단을 원점으로, 프레임 전체 크기에 대한
        비율로 ROI의 위치와 크기를 정규화합니다.

        Args:
            anchor_bbox: 앵커 영역 (x, y, w, h)
            roi_bbox: ROI 영역 (x, y, w, h)
            frame_shape: 프레임 shape (height, width, ...)

        Returns:
            {"nx": float, "ny": float, "nw": float, "nh": float}
        """
        frame_h, frame_w = frame_shape[:2]
        anchor_x, anchor_y, _, _ = anchor_bbox
        roi_x, roi_y, roi_w, roi_h = roi_bbox

        nx = (roi_x - anchor_x) / frame_w if frame_w > 0 else 0.0
        ny = (roi_y - anchor_y) / frame_h if frame_h > 0 else 0.0
        nw = roi_w / frame_w if frame_w > 0 else 0.0
        nh = roi_h / frame_h if frame_h > 0 else 0.0

        return {
            "nx": round(nx, 6),
            "ny": round(ny, 6),
            "nw": round(nw, 6),
            "nh": round(nh, 6),
        }

    # ========================================
    # 시각화
    # ========================================
    def _draw_overlay(self) -> NDArray:
        """현재 상태를 시각적으로 오버레이하여 표시용 이미지를 생성

        - 앵커: 초록색 점선 사각형 + 이름 라벨
        - ROI: 빨간색 점선 사각형 + 이름 라벨
        - 선택 중인 영역: 파란색 사각형
        - 앵커-ROI 연결선: 회색 선
        - 하단 상태바: 현재 모드 및 안내 텍스트
        - 도움말 오버레이 ('h' 토글)

        Returns:
            표시용 이미지 (BGR)
        """
        display = self.image.copy()
        img_h, img_w = display.shape[:2]

        # --- 앵커-ROI 연결선 ---
        for mapping in self.roi_mappings:
            anchor = self._find_anchor_by_name(mapping.anchor_name)
            if anchor is None:
                continue
            ax, ay, aw, ah = anchor.bbox
            anchor_center = (ax + aw // 2, ay + ah // 2)

            rx, ry, rw, rh = mapping.bbox
            roi_center = (rx + rw // 2, ry + rh // 2)

            cv2.line(
                display, anchor_center, roi_center, COLOR_CONNECTION, 1, cv2.LINE_AA
            )

        # --- 등록된 앵커 그리기 ---
        for anchor in self.anchors:
            ax, ay, aw, ah = anchor.bbox
            self._draw_dashed_rect(
                display, (ax, ay), (ax + aw, ay + ah), COLOR_ANCHOR, 2
            )

            # 이름 라벨
            label = anchor.name
            (tw, th), _ = cv2.getTextSize(label, FONT, FONT_SCALE_LABEL, FONT_THICKNESS)
            label_y = max(ay - 5, th + 2)
            cv2.rectangle(
                display,
                (ax, label_y - th - 2),
                (ax + tw + 4, label_y + 2),
                COLOR_ANCHOR,
                cv2.FILLED,
            )
            cv2.putText(
                display,
                label,
                (ax + 2, label_y),
                FONT,
                FONT_SCALE_LABEL,
                (255, 255, 255),
                FONT_THICKNESS,
                cv2.LINE_AA,
            )

        # --- 등록된 ROI 그리기 ---
        for mapping in self.roi_mappings:
            rx, ry, rw, rh = mapping.bbox
            self._draw_dashed_rect(display, (rx, ry), (rx + rw, ry + rh), COLOR_ROI, 2)

            # 이름 라벨
            label = f"{mapping.roi_name} [{mapping.roi_type}]"
            (tw, th), _ = cv2.getTextSize(label, FONT, FONT_SCALE_LABEL, FONT_THICKNESS)
            label_y = max(ry - 5, th + 2)
            cv2.rectangle(
                display,
                (rx, label_y - th - 2),
                (rx + tw + 4, label_y + 2),
                COLOR_ROI,
                cv2.FILLED,
            )
            cv2.putText(
                display,
                label,
                (rx + 2, label_y),
                FONT,
                FONT_SCALE_LABEL,
                (255, 255, 255),
                FONT_THICKNESS,
                cv2.LINE_AA,
            )

        # --- 현재 선택 영역 (드래그 중 또는 확정) ---
        if self._dragging:
            x1 = min(self._drag_start[0], self._drag_end[0])
            y1 = min(self._drag_start[1], self._drag_end[1])
            x2 = max(self._drag_start[0], self._drag_end[0])
            y2 = max(self._drag_start[1], self._drag_end[1])
            cv2.rectangle(display, (x1, y1), (x2, y2), COLOR_SELECTION, 2)
            # 선택 크기 표시
            sw, sh = x2 - x1, y2 - y1
            size_text = f"{sw}x{sh}"
            cv2.putText(
                display,
                size_text,
                (x1, max(y1 - 8, 12)),
                FONT,
                FONT_SCALE_LABEL,
                COLOR_SELECTION,
                FONT_THICKNESS,
                cv2.LINE_AA,
            )
        elif self._selection is not None:
            sx, sy, sw, sh = self._selection
            cv2.rectangle(display, (sx, sy), (sx + sw, sy + sh), COLOR_SELECTION, 2)

        # --- 상태바 ---
        status_bar = np.full(
            (STATUS_BAR_HEIGHT, img_w, 3), COLOR_STATUS_BG, dtype=np.uint8
        )

        # 상태 메시지 텍스트
        status_text = self._status_message
        cv2.putText(
            status_bar,
            status_text,
            (8, STATUS_BAR_HEIGHT - 12),
            FONT,
            FONT_SCALE_STATUS,
            COLOR_STATUS_TEXT,
            FONT_THICKNESS,
            cv2.LINE_AA,
        )

        # 앵커/ROI 카운트 표시 (우측)
        count_text = f"Anchors: {len(self.anchors)} | ROIs: {len(self.roi_mappings)} | Type: {self._current_roi_type}"
        (ctw, _), _ = cv2.getTextSize(
            count_text, FONT, FONT_SCALE_STATUS, FONT_THICKNESS
        )
        cv2.putText(
            status_bar,
            count_text,
            (max(8, img_w - ctw - 8), STATUS_BAR_HEIGHT - 12),
            FONT,
            FONT_SCALE_STATUS,
            (100, 200, 255),
            FONT_THICKNESS,
            cv2.LINE_AA,
        )

        display = np.vstack([display, status_bar])

        # --- 도움말 오버레이 ---
        if self._show_help:
            display = self._draw_help_overlay(display)

        return display

    @staticmethod
    def _draw_dashed_rect(
        img: NDArray,
        pt1: tuple[int, int],
        pt2: tuple[int, int],
        color: tuple[int, int, int],
        thickness: int = 2,
        dash_length: int = 10,
    ) -> None:
        """점선 사각형 그리기

        Args:
            img: 대상 이미지
            pt1: 좌상단 (x1, y1)
            pt2: 우하단 (x2, y2)
            color: BGR 색상
            thickness: 선 두께
            dash_length: 점선 간격 (픽셀)
        """
        x1, y1 = pt1
        x2, y2 = pt2

        # 네 변의 점선 그리기
        edges = [
            ((x1, y1), (x2, y1)),  # 상단
            ((x2, y1), (x2, y2)),  # 우측
            ((x2, y2), (x1, y2)),  # 하단
            ((x1, y2), (x1, y1)),  # 좌측
        ]

        for start, end in edges:
            sx, sy = start
            ex, ey = end
            length = max(abs(ex - sx), abs(ey - sy))
            if length == 0:
                continue

            dx = (ex - sx) / length
            dy = (ey - sy) / length

            i = 0
            drawing = True
            while i < length:
                seg_len = min(dash_length, length - i)
                if drawing:
                    p1 = (int(sx + dx * i), int(sy + dy * i))
                    p2 = (int(sx + dx * (i + seg_len)), int(sy + dy * (i + seg_len)))
                    cv2.line(img, p1, p2, color, thickness, cv2.LINE_AA)
                drawing = not drawing
                i += dash_length

    def _draw_help_overlay(self, display: NDArray) -> NDArray:
        """도움말 오버레이 그리기

        반투명 배경 위에 키보드 단축키 안내를 표시합니다.

        Args:
            display: 대상 이미지

        Returns:
            도움말 오버레이가 추가된 이미지
        """
        overlay = display.copy()
        h, w = display.shape[:2]

        help_lines = [
            "=== Keyboard Shortcuts ===",
            "",
            "  Drag       : Select region",
            "  a          : Register selection as Anchor (saves snippet)",
            "  r          : Register selection as ROI (mapped to last anchor)",
            "  t          : Cycle ROI type (numeric -> text -> chart)",
            "  s          : Save configuration to YAML",
            "  u          : Undo last registration",
            "  h          : Toggle this help",
            "  q / ESC    : Quit",
            "",
            "=== Workflow ===",
            "",
            "  1. Drag to select an anchor region, press 'a'",
            "  2. Drag to select an ROI region, press 'r'",
            "     (ROI is mapped to the most recent anchor)",
            "  3. Repeat for additional anchors/ROIs",
            "  4. Press 't' to change ROI type before pressing 'r'",
            "  5. Press 's' to save the YAML config",
        ]

        line_height = 22
        padding = 20
        box_h = len(help_lines) * line_height + 2 * padding
        box_w = 480

        # 도움말 박스 위치 (중앙)
        bx = (w - box_w) // 2
        by = (h - box_h) // 2

        # 반투명 배경
        cv2.rectangle(
            overlay, (bx, by), (bx + box_w, by + box_h), COLOR_HELP_BG, cv2.FILLED
        )
        alpha = 0.85
        cv2.addWeighted(overlay, alpha, display, 1 - alpha, 0, display)

        # 텍스트 렌더링
        for i, line in enumerate(help_lines):
            ty = by + padding + (i + 1) * line_height
            color = (100, 200, 255) if line.startswith("===") else COLOR_HELP_TEXT
            cv2.putText(
                display,
                line,
                (bx + padding, ty),
                FONT,
                FONT_SCALE_HELP,
                color,
                FONT_THICKNESS,
                cv2.LINE_AA,
            )

        return display

    # ========================================
    # ROI 타입 순환
    # ========================================
    @property
    def _current_roi_type(self) -> str:
        """현재 선택된 ROI 타입 문자열"""
        return self._roi_types[self._current_roi_type_index]

    def _cycle_roi_type(self) -> None:
        """ROI 타입을 다음 유형으로 순환 전환"""
        self._current_roi_type_index = (self._current_roi_type_index + 1) % len(
            self._roi_types
        )
        self._status_message = f"ROI type changed to: {self._current_roi_type}"

    # ========================================
    # YAML 저장
    # ========================================
    def _save_yaml(self, output_path: Path) -> None:
        """설정을 YAML 파일로 저장

        AnchorDetector.from_yaml()과 호환되는 형식으로
        앵커 정의와 ROI 매핑 정보를 YAML 파일에 기록합니다.

        Args:
            output_path: 저장할 YAML 파일 경로
        """
        if not _HAS_YAML:
            self._status_message = "PyYAML is not installed. Run: pip install pyyaml"
            logger.error("PyYAML is not installed")
            return

        if not self.anchors:
            self._status_message = "No anchors to save. Register at least one anchor."
            return

        frame_shape = self.image.shape

        # YAML 기준 경로 (상대 경로 산출용)
        output_path = output_path.resolve()
        base_dir = output_path.parent

        # 앵커 정의 구성
        yaml_anchors = []
        for anchor in self.anchors:
            anchor_dict: dict = {
                "name": anchor.name,
                "anchor_type": "snippet",
                "match_threshold": 0.7,
                "source_resolution": [frame_shape[1], frame_shape[0]],
            }
            if anchor.snippet_path is not None:
                try:
                    rel_path = anchor.snippet_path.resolve().relative_to(base_dir)
                    anchor_dict["snippet_path"] = str(rel_path).replace("\\", "/")
                except ValueError:
                    # 상대 경로 산출 불가 시 절대 경로 사용
                    anchor_dict["snippet_path"] = str(
                        anchor.snippet_path.resolve()
                    ).replace("\\", "/")
            yaml_anchors.append(anchor_dict)

        # ROI 매핑 구성
        yaml_mappings = []
        for mapping in self.roi_mappings:
            anchor = self._find_anchor_by_name(mapping.anchor_name)
            if anchor is None:
                logger.warning(
                    "ROI '%s' references unknown anchor '%s' -- skipping",
                    mapping.roi_name,
                    mapping.anchor_name,
                )
                continue

            offset = self._compute_normalized_offset(
                anchor.bbox, mapping.bbox, frame_shape
            )

            mapping_dict = {
                "anchor_name": mapping.anchor_name,
                "roi_name": mapping.roi_name,
                "offset": offset,
                "roi_type": mapping.roi_type,
            }
            yaml_mappings.append(mapping_dict)

        # 전체 YAML 구조
        yaml_data: dict = {
            "anchors": yaml_anchors,
            "roi_mappings": yaml_mappings,
            "enable_window_detection": True,
            "window_title_patterns": [],
            "fallback_to_homography": False,
            "iou_merge_threshold": 0.5,
        }

        # 디렉토리 생성 및 저장
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            yaml.dump(
                yaml_data,
                f,
                default_flow_style=False,
                allow_unicode=True,
                sort_keys=False,
            )

        self._status_message = f"Configuration saved to {output_path}"
        logger.info(
            "YAML saved: %s (%d anchors, %d ROIs)",
            output_path,
            len(yaml_anchors),
            len(yaml_mappings),
        )

    # ========================================
    # Undo
    # ========================================
    def _undo_last(self) -> None:
        """마지막 등록 동작을 취소

        Undo 스택에서 마지막 항목을 꺼내고 해당하는
        앵커 또는 ROI를 삭제합니다. 앵커 삭제 시
        연결된 스니펫 파일도 함께 삭제합니다.
        """
        if not self._undo_stack:
            self._status_message = "Nothing to undo."
            return

        entry = self._undo_stack.pop()

        if entry.action == "roi" and entry.roi_index is not None:
            if entry.roi_index < len(self.roi_mappings):
                removed = self.roi_mappings.pop(entry.roi_index)
                self._status_message = f"Undone: ROI '{removed.roi_name}' removed."
                logger.info("Undo ROI: %s", removed.roi_name)
            else:
                self._status_message = "Undo failed: ROI index out of range."

        elif entry.action == "anchor":
            if entry.anchor_index < len(self.anchors):
                removed_anchor = self.anchors.pop(entry.anchor_index)

                # 연결된 ROI 매핑도 함께 삭제
                removed_rois = [
                    m for m in self.roi_mappings if m.anchor_name == removed_anchor.name
                ]
                self.roi_mappings = [
                    m for m in self.roi_mappings if m.anchor_name != removed_anchor.name
                ]

                # 관련 undo 항목 정리
                self._undo_stack = [
                    e
                    for e in self._undo_stack
                    if not (
                        e.action == "roi"
                        and e.roi_index is not None
                        and any(
                            m.roi_name
                            == (
                                self.roi_mappings[e.roi_index].roi_name
                                if e.roi_index < len(self.roi_mappings)
                                else None
                            )
                            for m in removed_rois
                        )
                    )
                ]

                # 스니펫 파일 삭제
                if (
                    removed_anchor.snippet_path is not None
                    and removed_anchor.snippet_path.exists()
                ):
                    removed_anchor.snippet_path.unlink()

                roi_count = len(removed_rois)
                self._status_message = (
                    f"Undone: Anchor '{removed_anchor.name}' removed"
                    f" (+ {roi_count} linked ROI{'s' if roi_count != 1 else ''})."
                )
                logger.info(
                    "Undo anchor: %s (+%d ROIs)", removed_anchor.name, roi_count
                )
            else:
                self._status_message = "Undo failed: Anchor index out of range."

    # ========================================
    # 유틸리티
    # ========================================
    def _find_anchor_by_name(self, name: str) -> AnchorEntry | None:
        """이름으로 앵커를 검색

        Args:
            name: 검색할 앵커 이름

        Returns:
            일치하는 AnchorEntry 또는 None
        """
        for anchor in self.anchors:
            if anchor.name == name:
                return anchor
        return None


# ========================================
# 이미지/비디오 로딩 유틸리티
# ========================================
def load_reference_frame(
    image_path: Path | None = None, video_path: Path | None = None
) -> NDArray:
    """참조 프레임 로드

    이미지 파일 또는 비디오 파일의 첫 프레임을 로드합니다.

    Args:
        image_path: 이미지 파일 경로 (우선)
        video_path: 비디오 파일 경로 (이미지 없을 시)

    Returns:
        로드된 프레임 (BGR)

    Raises:
        FileNotFoundError: 파일이 존재하지 않을 때
        RuntimeError: 이미지/비디오 로드 실패 시
    """
    if image_path is not None:
        if not image_path.exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")
        frame = cv2.imread(str(image_path))
        if frame is None:
            raise RuntimeError(f"Failed to load image: {image_path}")
        logger.info(
            "Reference frame loaded from image: %s (%dx%d)",
            image_path,
            frame.shape[1],
            frame.shape[0],
        )
        return frame

    if video_path is not None:
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {video_path}")
        ret, frame = cap.read()
        cap.release()
        if not ret or frame is None:
            raise RuntimeError(f"Failed to read first frame from video: {video_path}")
        logger.info(
            "Reference frame loaded from video: %s (%dx%d)",
            video_path,
            frame.shape[1],
            frame.shape[0],
        )
        return frame

    raise ValueError("Either --image or --video must be specified.")


# ========================================
# CLI 엔트리포인트
# ========================================
def main() -> None:
    """CLI 엔트리포인트

    커맨드라인 인자를 파싱하고 앵커 설정 GUI를 실행합니다.
    """
    parser = argparse.ArgumentParser(
        description="앵커 기반 ROI 설정 도구 (Anchor Setup GUI)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "사용 예시:\n"
            "  python -m src.tools.anchor_setup_gui --image reference.png\n"
            "  python -m src.tools.anchor_setup_gui --video input.mp4 --output config/anchors.yaml\n"
            "\n"
            "GUI 조작법:\n"
            "  Drag       : 영역 선택\n"
            "  a          : 앵커 등록\n"
            "  r          : ROI 등록 (마지막 앵커에 매핑)\n"
            "  t          : ROI 타입 전환 (numeric -> text -> chart)\n"
            "  s          : YAML 저장\n"
            "  u          : 마지막 등록 취소\n"
            "  h          : 도움말 토글\n"
            "  q / ESC    : 종료\n"
        ),
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--image",
        type=Path,
        help="참조 이미지 파일 경로",
    )
    group.add_argument(
        "--video",
        type=Path,
        help="비디오 파일 경로 (첫 프레임 사용)",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("config/anchors.yaml"),
        help="출력 YAML 파일 경로 (기본: config/anchors.yaml)",
    )

    args = parser.parse_args()

    # 로깅 설정
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # 참조 프레임 로드
    try:
        frame = load_reference_frame(image_path=args.image, video_path=args.video)
    except (FileNotFoundError, RuntimeError, ValueError) as e:
        logger.error("Failed to load reference frame: %s", e)
        sys.exit(1)

    # 출력 디렉토리 결정
    output_path: Path = args.output.resolve()
    output_dir = output_path.parent

    # GUI 실행
    gui = AnchorSetupGUI(image=frame, output_dir=output_dir)

    logger.info("Starting Anchor Setup GUI...")
    logger.info("Output will be saved to: %s", output_path)

    gui.run()

    logger.info("Anchor Setup GUI closed.")


if __name__ == "__main__":
    main()
