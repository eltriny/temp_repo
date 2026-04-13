"""
비디오 프레임 추출 모듈

Generator 패턴을 활용하여 장시간 비디오(1시간+)를 메모리 효율적으로 처리합니다.
적응형 프레임 스킵 로직으로 중요한 프레임만 선별 추출합니다.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Generator, Iterator

import cv2
import numpy as np
from numpy.typing import NDArray

from ..config import Config, FrameSkipMode, ProcessingConfig

if TYPE_CHECKING:
    from cv2 import VideoCapture


logger = logging.getLogger(__name__)


@dataclass
class FrameData:
    """프레임 데이터 컨테이너

    추출된 프레임의 이미지 데이터와 메타데이터를 담습니다.

    Attributes:
        frame: BGR 형식의 프레임 이미지 (NumPy 배열)
        frame_number: 원본 비디오에서의 프레임 번호
        timestamp_ms: 프레임 타임스탬프 (밀리초)
        is_change_detected: 이전 프레임 대비 변화 감지 여부
        change_score: 변화량 점수 (0.0 ~ 1.0)
    """

    frame: NDArray[np.uint8]
    frame_number: int
    timestamp_ms: float
    is_change_detected: bool = False
    change_score: float = 0.0

    @property
    def timestamp(self) -> timedelta:
        """타임스탬프를 timedelta로 반환"""
        return timedelta(milliseconds=self.timestamp_ms)

    @property
    def timestamp_str(self) -> str:
        """타임스탬프를 HH:MM:SS.mmm 형식 문자열로 반환"""
        total_seconds = self.timestamp_ms / 1000
        hours = int(total_seconds // 3600)
        minutes = int((total_seconds % 3600) // 60)
        seconds = total_seconds % 60
        return f"{hours:02d}:{minutes:02d}:{seconds:06.3f}"

    @property
    def shape(self) -> tuple[int, int, int]:
        """프레임 shape (height, width, channels)"""
        return self.frame.shape

    @property
    def height(self) -> int:
        """프레임 높이"""
        return self.frame.shape[0]

    @property
    def width(self) -> int:
        """프레임 너비"""
        return self.frame.shape[1]


@dataclass
class VideoMetadata:
    """비디오 메타데이터

    Attributes:
        path: 비디오 파일 경로
        width: 프레임 너비
        height: 프레임 높이
        fps: 초당 프레임 수
        total_frames: 총 프레임 수
        duration_ms: 총 재생 시간 (밀리초)
        codec: 비디오 코덱 FourCC
    """

    path: Path
    width: int
    height: int
    fps: float
    total_frames: int
    duration_ms: float
    codec: str

    @property
    def duration(self) -> timedelta:
        """재생 시간을 timedelta로 반환"""
        return timedelta(milliseconds=self.duration_ms)

    @property
    def duration_str(self) -> str:
        """재생 시간을 HH:MM:SS 형식으로 반환"""
        total_seconds = int(self.duration_ms / 1000)
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        seconds = total_seconds % 60
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


@dataclass
class _ProcessingState:
    """내부 처리 상태 추적용 데이터클래스"""

    last_processed_frame: NDArray[np.uint8] | None = None
    frames_since_last_change: int = 0
    total_frames_processed: int = 0
    change_detected_count: int = 0
    current_interval_frames: int = field(default=0, init=False)


class VideoProcessor:
    """비디오 프레임 추출기

    Generator 패턴을 사용하여 메모리 효율적으로 프레임을 추출합니다.
    적응형 프레임 스킵으로 변화가 감지되면 추출 간격을 좁힙니다.

    Example:
        >>> config = Config(video_path=Path("video.mp4"))
        >>> processor = VideoProcessor(config)
        >>> for frame_data in processor.extract_frames():
        ...     # 프레임 처리
        ...     print(f"Frame {frame_data.frame_number}: {frame_data.timestamp_str}")
    """

    def __init__(self, config: Config) -> None:
        """VideoProcessor 초기화

        Args:
            config: 전체 설정 객체

        Raises:
            ValueError: video_path가 설정되지 않은 경우
            FileNotFoundError: 비디오 파일이 존재하지 않는 경우
        """
        if config.video_path is None:
            raise ValueError("video_path가 설정되어야 합니다")
        if not config.video_path.exists():
            raise FileNotFoundError(
                f"비디오 파일을 찾을 수 없습니다: {config.video_path}"
            )

        self._config = config
        self._processing_config = config.processing
        self._video_path = config.video_path
        self._metadata: VideoMetadata | None = None

        # FPS 기반 프레임 간격 계산 (초기값, 메타데이터 로드 후 재계산)
        self._default_interval_frames = 30  # 기본값
        self._change_interval_frames = 5  # 기본값

        logger.info(f"VideoProcessor 초기화: {self._video_path}")

    @property
    def metadata(self) -> VideoMetadata:
        """비디오 메타데이터 (지연 로딩)"""
        if self._metadata is None:
            self._metadata = self._load_metadata()
            self._calculate_frame_intervals()
        return self._metadata

    def _load_metadata(self) -> VideoMetadata:
        """비디오 메타데이터 로드"""
        cap = cv2.VideoCapture(str(self._video_path))
        try:
            if not cap.isOpened():
                raise RuntimeError(f"비디오를 열 수 없습니다: {self._video_path}")

            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            codec_int = int(cap.get(cv2.CAP_PROP_FOURCC))
            codec = "".join(chr((codec_int >> 8 * i) & 0xFF) for i in range(4))

            duration_ms = (total_frames / fps * 1000) if fps > 0 else 0

            metadata = VideoMetadata(
                path=self._video_path,
                width=width,
                height=height,
                fps=fps,
                total_frames=total_frames,
                duration_ms=duration_ms,
                codec=codec,
            )

            logger.info(
                f"비디오 메타데이터 로드: {width}x{height}, "
                f"{fps:.2f}fps, {metadata.duration_str}"
            )

            return metadata
        finally:
            cap.release()

    def _calculate_frame_intervals(self) -> None:
        """FPS 기반 프레임 추출 간격 계산"""
        fps = self.metadata.fps
        if fps <= 0:
            fps = 30.0  # 기본값

        self._default_interval_frames = max(
            1,
            int(fps * self._processing_config.default_interval_sec),
        )
        self._change_interval_frames = max(
            1,
            int(fps * self._processing_config.change_detection_interval_sec),
        )

        logger.debug(
            f"프레임 간격 설정: 기본={self._default_interval_frames}, "
            f"변화감지={self._change_interval_frames}"
        )

    @contextmanager
    def _open_video(self) -> Generator[VideoCapture, None, None]:
        """비디오 파일을 안전하게 열고 닫는 컨텍스트 매니저"""
        cap = cv2.VideoCapture(str(self._video_path))
        try:
            if not cap.isOpened():
                raise RuntimeError(f"비디오를 열 수 없습니다: {self._video_path}")
            yield cap
        finally:
            cap.release()

    def _compute_frame_difference(
        self,
        frame1: NDArray[np.uint8],
        frame2: NDArray[np.uint8],
    ) -> float:
        """두 프레임 간 차이 점수 계산 (0.0 ~ 1.0)

        그레이스케일 변환 후 절대 차이의 평균을 정규화하여 반환합니다.

        Args:
            frame1: 첫 번째 프레임
            frame2: 두 번째 프레임

        Returns:
            정규화된 차이 점수 (0.0 = 동일, 1.0 = 완전히 다름)
        """
        # 그레이스케일 변환 후 리사이즈 (1채널 리사이즈가 3채널보다 ~3배 빠름)
        # 1920x1080 원본(~2M px) → 160x90(~14K px) 로 ~143배 연산량 감소.
        small_size = (160, 90)
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        small1 = cv2.resize(gray1, small_size)
        small2 = cv2.resize(gray2, small_size)

        # 절대 차이 계산 — 반드시 리사이즈된 small1/small2 사용 (C-09 회귀 방지)
        diff = cv2.absdiff(small1, small2)
        score = float(np.mean(diff) / 255.0)

        return score

    def _resize_frame_if_needed(
        self,
        frame: NDArray[np.uint8],
    ) -> NDArray[np.uint8]:
        """설정에 따라 프레임 리사이즈"""
        target_width = self._processing_config.resize_width
        target_height = self._processing_config.resize_height

        if target_width <= 0 and target_height <= 0:
            return frame

        h, w = frame.shape[:2]

        if target_width > 0 and target_height > 0:
            new_size = (target_width, target_height)
        elif target_width > 0:
            ratio = target_width / w
            new_size = (target_width, int(h * ratio))
        else:
            ratio = target_height / h
            new_size = (int(w * ratio), target_height)

        return cv2.resize(frame, new_size, interpolation=cv2.INTER_AREA)

    def extract_frames(
        self,
        start_ms: float = 0,
        end_ms: float | None = None,
    ) -> Generator[FrameData, None, None]:
        """프레임 추출 제너레이터

        메모리 효율적으로 비디오에서 프레임을 추출합니다.
        적응형 모드에서는 변화 감지 시 추출 간격을 좁힙니다.

        Args:
            start_ms: 시작 위치 (밀리초)
            end_ms: 종료 위치 (밀리초, None이면 끝까지)

        Yields:
            FrameData: 추출된 프레임 데이터

        Raises:
            RuntimeError: 비디오 열기 실패 시
        """
        # 메타데이터 확인 (필요 시 로드)
        _ = self.metadata

        state = _ProcessingState()
        is_adaptive = self._processing_config.frame_skip_mode == FrameSkipMode.ADAPTIVE
        threshold = self._processing_config.change_threshold

        try:
            with self._open_video() as cap:
                # ★ 디버깅: 비디오 열기 확인
                logger.info(f"비디오 열기 성공: isOpened={cap.isOpened()}")

                # 시작 위치 설정
                if start_ms > 0:
                    cap.set(cv2.CAP_PROP_POS_MSEC, start_ms)

                # 현재 프레임 간격 (적응형에서 동적 조정)
                current_interval = self._default_interval_frames
                frames_until_next = 0

                # ★ 디버깅: 초기 설정 로그
                logger.debug(
                    f"프레임 추출 시작: interval={current_interval}, skip_mode={self._processing_config.frame_skip_mode}"
                )

                while True:
                    # 종료 조건 확인
                    current_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
                    if end_ms is not None and current_ms > end_ms:
                        logger.debug(
                            f"end_ms 도달로 종료: current={current_ms}, end={end_ms}"
                        )
                        break

                    ret, frame = cap.read()

                    # ★ 디버깅: 첫 프레임 읽기 결과 로그
                    if state.total_frames_processed == 0 and frames_until_next == 0:
                        frame_shape = (
                            frame.shape if ret and frame is not None else "None"
                        )
                        logger.info(
                            f"첫 프레임 읽기 결과: ret={ret}, frame_shape={frame_shape}"
                        )

                    if not ret:
                        logger.warning(
                            f"프레임 읽기 실패 (ret=False), 총 읽은 프레임: {state.total_frames_processed}"
                        )
                        break

                    frame_number = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1

                    # 프레임 스킵 로직
                    if frames_until_next > 0:
                        frames_until_next -= 1
                        continue

                    # 리사이즈 적용
                    processed_frame = self._resize_frame_if_needed(frame)

                    # 변화 감지 (적응형 모드)
                    change_score = 0.0
                    is_change_detected = False

                    if is_adaptive and state.last_processed_frame is not None:
                        change_score = self._compute_frame_difference(
                            state.last_processed_frame,
                            processed_frame,
                        )
                        is_change_detected = change_score > threshold

                        if is_change_detected:
                            # 변화 감지 시 간격 축소
                            current_interval = self._change_interval_frames
                            state.change_detected_count += 1
                            state.frames_since_last_change = 0
                            logger.debug(
                                f"변화 감지 (score={change_score:.3f}) at frame {frame_number}, "
                                f"간격 → {current_interval}"
                            )
                        else:
                            state.frames_since_last_change += 1
                            # 일정 프레임 동안 변화 없으면 간격 복원
                            if (
                                state.frames_since_last_change
                                > self._default_interval_frames * 2
                            ):
                                current_interval = self._default_interval_frames

                    # 프레임 데이터 생성 및 yield
                    frame_data = FrameData(
                        frame=processed_frame,
                        frame_number=frame_number,
                        timestamp_ms=current_ms,
                        is_change_detected=is_change_detected,
                        change_score=change_score,
                    )

                    state.last_processed_frame = processed_frame.copy()
                    state.total_frames_processed += 1
                    frames_until_next = current_interval - 1

                    yield frame_data
        finally:
            # ★ 디버깅: 제너레이터 종료 시 반드시 실행 (break, return, 예외 모두 포함)
            logger.info(
                f"프레임 추출 종료: 총 {state.total_frames_processed}개, "
                f"변화감지 {state.change_detected_count}회"
            )

    def extract_frame_at(self, timestamp_ms: float) -> FrameData | None:
        """특정 타임스탬프의 프레임 추출

        Args:
            timestamp_ms: 추출할 위치 (밀리초)

        Returns:
            FrameData 또는 None (실패 시)
        """
        with self._open_video() as cap:
            cap.set(cv2.CAP_PROP_POS_MSEC, timestamp_ms)
            ret, frame = cap.read()

            if not ret:
                logger.warning(f"프레임 추출 실패: {timestamp_ms}ms")
                return None

            frame_number = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
            actual_ms = cap.get(cv2.CAP_PROP_POS_MSEC)

            processed_frame = self._resize_frame_if_needed(frame)

            return FrameData(
                frame=processed_frame,
                frame_number=frame_number,
                timestamp_ms=actual_ms,
            )

    def iterate_all_frames(self) -> Iterator[FrameData]:
        """모든 프레임을 순차적으로 반환 (스킵 없음)

        메모리 사용에 주의가 필요합니다.
        특수한 분석 목적으로만 사용하세요.

        Yields:
            FrameData: 각 프레임 데이터
        """
        with self._open_video() as cap:
            frame_number = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                timestamp_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
                processed_frame = self._resize_frame_if_needed(frame)

                yield FrameData(
                    frame=processed_frame,
                    frame_number=frame_number,
                    timestamp_ms=timestamp_ms,
                )
                frame_number += 1

    def get_progress(
        self,
        current_frame: int,
    ) -> tuple[float, str]:
        """처리 진행률 계산

        Args:
            current_frame: 현재 프레임 번호

        Returns:
            (진행률 0.0~1.0, "HH:MM:SS / HH:MM:SS" 형식 문자열) 튜플
        """
        total = self.metadata.total_frames
        progress = current_frame / total if total > 0 else 0.0

        current_ms = (
            (current_frame / self.metadata.fps * 1000) if self.metadata.fps > 0 else 0
        )
        current_time = timedelta(milliseconds=current_ms)
        total_time = self.metadata.duration

        # 시간 포맷팅
        def format_time(td: timedelta) -> str:
            total_seconds = int(td.total_seconds())
            hours = total_seconds // 3600
            minutes = (total_seconds % 3600) // 60
            seconds = total_seconds % 60
            return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

        time_str = f"{format_time(current_time)} / {format_time(total_time)}"

        return progress, time_str
