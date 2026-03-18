"""
산업용 비디오 모니터링 분석 프로그램 (v2.0 - 사전 정의 ROI 기반)

주요 변경사항 (v2.0):
  - 자동 ROI 감지 제거
  - 사전 정의된 ROI 템플릿 사용
  - ROI별 병렬 처리 지원

실행 방법:
  프로젝트 루트에서: python -m src.main --video video.mp4 --output ./results --template "기본 템플릿"
  또는: cd src && python main.py --video ../video.mp4 --output ../results --template-id 1
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

# 상대 임포트를 위한 경로 설정 (스크립트 직접 실행 시)
if __name__ == "__main__" and __package__ is None:
    _src_dir = Path(__file__).resolve().parent
    _project_root = _src_dir.parent
    if str(_project_root) not in sys.path:
        sys.path.insert(0, str(_project_root))
    __package__ = "src"

import cv2

from .config import Config, ProcessingConfig, StorageConfig
from .core.parallel_processor import (
    ParallelProcessor,
    ProcessorConfig,
    ROIProcessType,
    ROIResult,
)
from .core.video_processor import VideoProcessor
from .detection.dynamic_roi_detector import (
    DynamicROIConfig,
    DynamicROIDetector,
    WindowDetectionStrategy,
)
from .detection.roi_types import ROIType as DetROIType
from .ocr.ocr_engine import create_industrial_ocr_config
from .storage.capture_manager import CaptureConfig, CaptureManager, ImageFormat
from .storage.database import (
    CaptureCreate,
    ChangeEventCreate,
    DatabaseManager,
    ROICreate,
    ROIType,
    SessionCreate,
)
from .storage.roi_template_manager import ROITemplateManager
from .visualization import ROIVisualizer, VisualizationConfig, VisualizationStyle

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("video_analyzer.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)

# detection ROIType → database ROIType 매핑 (자동 탐지, 색상 템플릿 양쪽에서 사용)
_DET_TO_DB_ROI_TYPE: dict[DetROIType, ROIType] = {
    DetROIType.NUMERIC: ROIType.NUMERIC,
    DetROIType.TEXT: ROIType.TEXT,
    DetROIType.CHART: ROIType.CHART,
    DetROIType.UNKNOWN: ROIType.NUMERIC,
}


class VideoAnalyzerApp:
    """비디오 분석 애플리케이션 메인 클래스 (v2.0)

    사전 정의된 ROI 템플릿을 사용하여 비디오를 분석합니다.
    ROI별 병렬 처리를 지원합니다.
    """

    @classmethod
    def from_config(cls, config: Config) -> VideoAnalyzerApp:
        """Config 객체에서 VideoAnalyzerApp 생성

        Args:
            config: 병합된 설정 객체

        Returns:
            VideoAnalyzerApp 인스턴스
        """
        return cls(
            video_path=config.video_path,
            output_dir=config.storage.output_dir,
            template_id=config.template.id,
            template_name=config.template.name,
            use_gpu=config.processing.use_gpu,
            frame_interval=config.processing.default_interval_sec,
            ssim_threshold=config.detection.ssim_threshold,
            confidence_threshold=config.detection.confidence_threshold,
            max_workers=config.processing.max_workers or None,
            auto_detect=config.detection.auto_detect,
            roi_template_paths=[
                Path(p) for p in config.detection.roi_template_paths
            ],
            anchor_config_path=(
                Path(config.detection.anchor_config_path)
                if config.detection.anchor_config_path
                else None
            ),
        )

    def __init__(
        self,
        video_path: Path,
        output_dir: Path,
        template_id: Optional[int] = None,
        template_name: Optional[str] = None,
        use_gpu: bool = False,
        frame_interval: float = 1.0,
        ssim_threshold: float = 0.95,
        confidence_threshold: float = 0.7,
        max_workers: Optional[int] = None,
        auto_detect: bool = False,
        roi_template_paths: Optional[list[Path]] = None,
        anchor_config_path: Optional[Path] = None,
    ):
        """
        Args:
            video_path: 분석할 비디오 파일 경로
            output_dir: 결과 저장 디렉토리
            template_id: 사용할 ROI 템플릿 ID
            template_name: 사용할 ROI 템플릿 이름 (template_id 우선)
            use_gpu: GPU 가속 사용 여부
            frame_interval: 프레임 분석 간격 (초)
            ssim_threshold: SSIM 변화 감지 임계값
            confidence_threshold: OCR 신뢰도 임계값
            max_workers: 병렬 처리 워커 수
            auto_detect: 첫 프레임 자동 ROI 탐지 모드
            roi_template_paths: 색상 마커 ROI 템플릿 이미지 경로 리스트
            anchor_config_path: 앵커 기반 ROI 탐지 YAML 설정 파일 경로
        """
        self.video_path = video_path
        self.output_dir = output_dir
        self.template_id = template_id
        self.template_name = template_name
        self.use_gpu = use_gpu
        self.max_workers = max_workers
        self.ssim_threshold = ssim_threshold
        self.confidence_threshold = confidence_threshold
        self.auto_detect = auto_detect
        self.roi_template_paths = roi_template_paths or []
        self.anchor_config_path = anchor_config_path

        # 출력 디렉토리 생성
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "captures").mkdir(exist_ok=True)

        # 전체 설정 초기화
        self.config = Config(
            video_path=video_path,
            processing=ProcessingConfig(
                default_interval_sec=frame_interval,
            ),
            storage=StorageConfig(
                output_dir=self.output_dir,
            ),
        )

        # 컴포넌트 초기화
        logger.info("컴포넌트 초기화 중...")

        # 비디오 프로세서
        self.video_processor = VideoProcessor(self.config)

        # OCR 설정 - 산업용 디스플레이 최적화 설정 사용
        base_ocr_config = create_industrial_ocr_config()
        self.ocr_config = replace(
            base_ocr_config,
            use_gpu=use_gpu,
            confidence_threshold=confidence_threshold,
        )

        # 데이터베이스 매니저
        self.db_manager = DatabaseManager.from_config(self.config.storage)

        # ROI 템플릿 매니저
        self.template_manager: Optional[ROITemplateManager] = None

        # 병렬 처리기 설정 - OCRConfig 설정 연동
        self.processor_config = ProcessorConfig(
            max_workers=max_workers,
            use_gpu=use_gpu,
            ocr_confidence_threshold=confidence_threshold,
            # OCRConfig에서 가져온 설정
            det_db_thresh=self.ocr_config.det_db_thresh,
            det_db_box_thresh=self.ocr_config.det_db_box_thresh,
            det_db_unclip_ratio=self.ocr_config.det_db_unclip_ratio,
            use_space_char=self.ocr_config.use_space_char,
            language=self.ocr_config.language.value,
            use_angle_cls=self.ocr_config.use_angle_cls,
            enable_text_correction=self.ocr_config.enable_text_correction,
        )
        # 캡쳐 매니저
        self.capture_manager = CaptureManager(
            CaptureConfig(
                base_directory=self.output_dir / "captures",
                default_format=ImageFormat.JPEG,
                compression_quality=90,
            )
        )

        # ROI 시각화기 (캡처 이미지에 ROI 영역 표시)
        self.roi_visualizer = ROIVisualizer(
            VisualizationConfig(
                style=VisualizationStyle.DASHED,
                show_label=True,
                show_confidence=True,
                line_thickness=2,
            )
        )

        logger.info("모든 컴포넌트 초기화 완료")

    def run(self) -> None:
        """분석 실행"""
        logger.info(f"비디오 분석 시작: {self.video_path}")

        # 메타데이터 로드
        metadata = self.video_processor.metadata
        logger.info(
            f"비디오 정보: {metadata.width}x{metadata.height}, "
            f"{metadata.fps:.2f}fps, {metadata.duration_str}"
        )

        # 데이터베이스 연결 및 템플릿 로드
        with self.db_manager as db:
            self.template_manager = ROITemplateManager(db)

            # 세션 생성
            session = db.create_session(
                SessionCreate(
                    name=f"Analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    source_path=str(self.video_path),
                )
            )
            session_id = session.id
            logger.info(f"분석 세션 생성: ID={session_id}")

            # === ROI 획득: 앵커 / 색상 템플릿 / auto-detect / DB 템플릿 ===
            if self.anchor_config_path:
                rois = self._detect_rois_from_anchor(db, session_id)
                if not rois:
                    logger.error(
                        "앵커 기반 ROI 탐지 결과가 없습니다. "
                        "앵커 설정(스니펫 이미지/텍스트 패턴)을 확인하세요."
                    )
                    return
            elif self.roi_template_paths:
                rois = self._detect_rois_from_color_template(db, session_id)
                if not rois:
                    logger.error(
                        "색상 템플릿에서 ROI를 찾지 못했습니다. "
                        "이미지에 빨강/초록/파랑 등 마커 색상으로 영역을 칠했는지 확인하세요."
                    )
                    return
            elif self.auto_detect:
                rois = self._detect_rois_from_first_frame(db, session_id)
                if not rois:
                    logger.error(
                        "자동 ROI 탐지 결과가 없습니다. "
                        "영상에서 텍스트/파형 패턴을 찾지 못했습니다."
                    )
                    return
            else:
                # 기존 템플릿 경로 (변경 없음)
                template_id = self._resolve_template_id()
                if template_id is None:
                    logger.error(
                        "사용할 ROI 템플릿이 없습니다. "
                        "--template 또는 --template-id 옵션으로 템플릿을 지정하세요."
                    )
                    return

                copied_rois = self.template_manager.copy_template_to_session(
                    template_id=template_id,
                    session_id=session_id,
                )

                if not copied_rois:
                    logger.error(f"템플릿 복사 실패: template_id={template_id}")
                    return

                rois = self.template_manager.load_session_rois_as_detection_rois(
                    session_id
                )

            logger.info(f"{len(rois)}개의 ROI 로드됨")

            # 프레임 분석 루프 (병렬 처리)
            self._analyze_frames_parallel(session_id, rois)

            # 세션 종료
            db.update_session(session_id, is_active=False)

        logger.info("비디오 분석 완료")
        self._print_summary(session_id)

    def _detect_rois_from_anchor(self, db: "DatabaseManager", session_id: int) -> list:
        """앵커 기반 ROI 탐지 후 DB에 저장

        AnchorDetector를 사용하여 첫 프레임에서 앵커(스니펫/텍스트)를
        탐지하고, 정규화 오프셋으로 ROI 좌표를 계산합니다.

        Args:
            db: DatabaseManager 인스턴스
            session_id: 현재 분석 세션 ID

        Returns:
            탐지된 detection ROI 리스트 (DB 저장 후 재로드)
        """
        from .detection.anchor_detector import AnchorDetector

        logger.info("=== 앵커 기반 ROI 탐지 시작 ===")
        logger.info("앵커 설정 파일: %s", self.anchor_config_path)

        # YAML 설정에서 AnchorDetector 생성
        anchor_detector = AnchorDetector.from_yaml(self.anchor_config_path)
        logger.info(
            "앵커 %d개, ROI 매핑 %d개 로드됨",
            len(anchor_detector.config.anchors),
            len(anchor_detector.config.roi_mappings),
        )

        # 첫 프레임 추출
        first_frame = None
        for frame_data in self.video_processor.extract_frames():
            first_frame = frame_data.frame
            break

        if first_frame is None:
            logger.error("첫 프레임을 추출할 수 없습니다")
            return []

        # 앵커 탐지 실행
        detected_rois = anchor_detector.detect(first_frame)

        if not detected_rois:
            logger.warning("앵커 기반 ROI 탐지 결과 없음")
            return []

        logger.info(f"앵커 기반 ROI {len(detected_rois)}개 탐지됨:")

        # DB에 저장
        for roi in detected_rois:
            db_roi_type = _DET_TO_DB_ROI_TYPE.get(roi.roi_type, ROIType.NUMERIC)

            roi_create = ROICreate(
                session_id=session_id,
                name=roi.id,
                roi_type=db_roi_type,
                x=roi.bbox.x,
                y=roi.bbox.y,
                width=roi.bbox.width,
                height=roi.bbox.height,
                threshold=0.1,
                metadata=json.dumps(roi.metadata, ensure_ascii=False),
            )
            db.create_roi(roi_create)
            logger.info(
                f"  {roi.id}: {roi.roi_type.value} "
                f"({roi.bbox.x},{roi.bbox.y},{roi.bbox.width}x{roi.bbox.height}) "
                f"confidence={roi.confidence:.2f}"
            )

        # DB에서 detection ROI로 재로드
        rois = self.template_manager.load_session_rois_as_detection_rois(session_id)
        logger.info(f"=== 앵커 기반 ROI 탐지 완료: {len(rois)}개 ROI 등록 ===")
        return rois

    def _detect_rois_from_first_frame(
        self, db: "DatabaseManager", session_id: int
    ) -> list:
        """첫 프레임에서 ROI 자동 탐지 후 DB에 저장

        DynamicROIDetector를 사용하여 첫 프레임에서 텍스트/파형 ROI를
        자동 탐지하고, 결과를 roi_definitions 테이블에 저장합니다.

        Args:
            db: DatabaseManager 인스턴스
            session_id: 현재 분석 세션 ID

        Returns:
            탐지된 detection ROI 리스트 (DB 저장 후 재로드)
        """
        logger.info("=== 첫 프레임 자동 ROI 탐지 시작 ===")

        # 첫 프레임 추출
        first_frame = None
        for frame_data in self.video_processor.extract_frames():
            first_frame = frame_data.frame
            break

        if first_frame is None:
            logger.error("첫 프레임을 추출할 수 없습니다")
            return []

        # DynamicROIDetector로 ROI 탐지
        # ★ OCR 엔진을 전달하지 않아 TextROIDetector가 자체 엔진 생성
        # (confidence_threshold=0.3, use_space_char=True — 전체 프레임 스캔에 최적)
        # App의 OCR 엔진(0.7 threshold)은 ROI 영역 정밀 인식용이므로 별도 유지
        dynamic_detector = DynamicROIDetector(
            config=DynamicROIConfig(
                enable_text_detection=True,
                window_detection_strategy=WindowDetectionStrategy.CONTOUR,
            ),
            ocr_engine=None,
        )

        try:
            detected_rois = dynamic_detector.detect(first_frame)
        except Exception as e:
            logger.warning("텍스트 ROI 탐지 중 예외, 파형 전용으로 재시도: %s", e)
            dynamic_detector = DynamicROIDetector(
                config=DynamicROIConfig(
                    enable_text_detection=False,
                    window_detection_strategy=WindowDetectionStrategy.CONTOUR,
                ),
            )
            detected_rois = dynamic_detector.detect(first_frame)

        if not detected_rois:
            logger.warning("자동 ROI 탐지 결과 없음")
            return []

        # 윈도우 경계 탐지 결과 로깅
        window_bbox = dynamic_detector.last_window_bbox
        if window_bbox is not None:
            logger.info(
                "윈도우 경계: (%d,%d) %dx%d",
                window_bbox.x,
                window_bbox.y,
                window_bbox.width,
                window_bbox.height,
            )
        else:
            logger.info("윈도우 경계 미탐지 — 전체 프레임 사용")

        logger.info(f"자동 탐지된 ROI {len(detected_rois)}개:")

        # DB에 저장
        for roi in detected_rois:
            db_roi_type = _DET_TO_DB_ROI_TYPE.get(roi.roi_type, ROIType.NUMERIC)

            roi_create = ROICreate(
                session_id=session_id,
                name=roi.id,
                roi_type=db_roi_type,
                x=roi.bbox.x,
                y=roi.bbox.y,
                width=roi.bbox.width,
                height=roi.bbox.height,
                threshold=0.1,
                metadata=json.dumps(roi.metadata, ensure_ascii=False),
            )
            db.create_roi(roi_create)
            logger.info(
                f"  {roi.id}: {roi.roi_type.value} "
                f"({roi.bbox.x},{roi.bbox.y},{roi.bbox.width}x{roi.bbox.height}) "
                f"confidence={roi.confidence:.2f}"
            )

        # DB에서 detection ROI로 재로드 (db_roi_id 포함)
        rois = self.template_manager.load_session_rois_as_detection_rois(session_id)
        logger.info(f"=== 자동 ROI 탐지 완료: {len(rois)}개 ROI 등록 ===")
        return rois

    def _detect_rois_from_color_template(
        self, db: "DatabaseManager", session_id: int
    ) -> list:
        """색상 마커 템플릿 이미지에서 ROI를 탐지하여 DB에 저장

        사용자가 색상으로 칠한 템플릿 이미지에서 ROI 영역을 추출하고,
        비디오 첫 프레임과 매칭하여 글로벌 좌표로 변환합니다.

        Args:
            db: DatabaseManager 인스턴스
            session_id: 현재 분석 세션 ID

        Returns:
            탐지된 detection ROI 리스트 (DB 저장 후 재로드)
        """
        logger.info("=== 색상 마커 템플릿 ROI 탐지 시작 ===")
        logger.info("템플릿 이미지: %s", [str(p) for p in self.roi_template_paths])

        # 첫 프레임 추출
        first_frame = None
        for frame_data in self.video_processor.extract_frames():
            first_frame = frame_data.frame
            break

        if first_frame is None:
            logger.error("첫 프레임을 추출할 수 없습니다")
            return []

        # ColorTemplateDetector로 ROI 탐지
        from .detection.color_template_detector import ColorTemplateDetector

        detector = ColorTemplateDetector()
        detected_rois = detector.detect_from_templates(
            template_paths=self.roi_template_paths,
            first_frame=first_frame,
        )

        if not detected_rois:
            logger.warning("색상 템플릿에서 ROI를 찾지 못했습니다")
            return []

        logger.info(f"색상 템플릿에서 {len(detected_rois)}개 ROI 탐지:")

        # DB에 저장
        for roi in detected_rois:
            db_roi_type = _DET_TO_DB_ROI_TYPE.get(roi.roi_type, ROIType.NUMERIC)

            roi_create = ROICreate(
                session_id=session_id,
                name=roi.id,
                roi_type=db_roi_type,
                x=roi.bbox.x,
                y=roi.bbox.y,
                width=roi.bbox.width,
                height=roi.bbox.height,
                threshold=0.1,
                metadata=json.dumps(roi.metadata, ensure_ascii=False),
            )
            db.create_roi(roi_create)
            logger.info(
                f"  {roi.id}: {roi.roi_type.value} "
                f"({roi.bbox.x},{roi.bbox.y},{roi.bbox.width}x{roi.bbox.height})"
            )

        # DB에서 detection ROI로 재로드 (db_roi_id 포함)
        rois = self.template_manager.load_session_rois_as_detection_rois(session_id)
        logger.info(f"=== 색상 템플릿 ROI 탐지 완료: {len(rois)}개 ROI 등록 ===")
        return rois

    def _resolve_template_id(self) -> Optional[int]:
        """템플릿 ID 해석

        template_id 또는 template_name에서 실제 템플릿 ID를 반환합니다.

        Returns:
            템플릿 ID (찾지 못하면 None)
        """
        if self.template_manager is None:
            return None

        # template_id가 직접 지정된 경우
        if self.template_id is not None:
            template = self.template_manager.get_template(self.template_id)
            if template:
                logger.info(f"템플릿 ID={self.template_id}에서 ROI 로드")
                return template.id
            logger.warning(f"템플릿 ID={self.template_id}를 찾을 수 없습니다")

        # template_name이 지정된 경우
        if self.template_name is not None:
            template = self.template_manager.get_template_by_name(self.template_name)
            if template:
                logger.info(f"템플릿 '{self.template_name}'에서 ROI 로드")
                return template.id
            logger.warning(f"템플릿 '{self.template_name}'를 찾을 수 없습니다")

        return None

    def _analyze_frames_parallel(self, session_id: int, rois: list) -> None:
        """프레임 분석 루프 (트리거 ROI 기반 캡쳐)

        트리거 ROI(is_trigger=True)가 변경될 때만 캡쳐를 수행합니다.
        캡쳐 시점에 모든 ROI의 현재 값을 하나의 capture_id로 그룹화하여 저장합니다.

        트리거 ROI가 없으면 모든 ROI 변화를 감지합니다.
        """
        prev_values: dict[str, str] = {}  # ROI별 이전 값 저장
        frame_count = 0
        capture_count = 0
        is_first_valid_frame = True  # 첫 번째 유효한 프레임 플래그

        logger.info(
            f"프레임 분석 시작 (트리거 기반 캡쳐, workers={self.processor_config.max_workers})"
        )

        with ParallelProcessor(self.processor_config) as processor:
            # ROI를 작업 리스트로 변환
            tasks = processor.create_tasks_from_detection_rois(rois)

            logger.info(f"총 {len(tasks)}개 ROI 작업 생성됨")
            for task in tasks:
                logger.info(
                    f"  - ROI: {task.name} (id={task.roi_id}, type={task.process_type.name}, is_trigger={task.is_trigger})"
                )

            # 트리거 ROI 식별
            trigger_roi_ids = [task.roi_id for task in tasks if task.is_trigger]
            if not trigger_roi_ids:
                logger.warning("트리거 ROI가 없습니다. 모든 ROI 변화 시 캡쳐합니다.")
                # 트리거가 없으면 모든 OCR ROI를 트리거로 사용
                trigger_roi_ids = [
                    task.roi_id
                    for task in tasks
                    if task.process_type == ROIProcessType.OCR
                ]
                logger.info(
                    f"OCR ROI {len(trigger_roi_ids)}개를 트리거로 사용: {trigger_roi_ids}"
                )
            else:
                logger.info(f"트리거 ROI: {trigger_roi_ids}")

            # ★ 디버깅: for 루프 진입 전 로그
            logger.info("프레임 추출 제너레이터 시작...")

            for frame_data in self.video_processor.extract_frames():
                frame_count += 1

                # ★ 디버깅: 첫 프레임 진입 확인
                if frame_count == 1:
                    logger.info(
                        f"첫 프레임 진입 확인: frame_number={frame_data.frame_number}, timestamp_ms={frame_data.timestamp_ms:.0f}"
                    )

                current_frame = frame_data.frame

                # 진행률 출력 (매 100프레임마다)
                if frame_count % 100 == 0:
                    progress, time_str = self.video_processor.get_progress(
                        frame_data.frame_number
                    )
                    logger.info(
                        f"진행률: {progress * 100:.1f}% [{time_str}] (프레임: {frame_count}, 캡쳐: {capture_count})"
                    )

                # ROI별 병렬 처리
                # ★ 디버깅: process_frame 호출 전
                if frame_count <= 3:
                    logger.info(
                        f"프레임 {frame_count}: ROI 처리 시작 (tasks={len(tasks)}개)"
                    )

                results = processor.process_frame(current_frame, tasks)

                # ★ 디버깅: process_frame 호출 후
                if frame_count <= 3:
                    logger.info(
                        f"프레임 {frame_count}: ROI 처리 완료 (results={len(results)}개)"
                    )

                # 현재 프레임의 모든 ROI 값 수집
                current_values: dict[str, tuple[ROIResult, str]] = {}

                for roi_id, result in results.items():
                    if not result.success:
                        if frame_count <= 10:
                            logger.warning(
                                f"프레임 {frame_count}, ROI 처리 실패: {roi_id} - {result.error}"
                            )
                        continue

                    # 공백 제거하여 값 정규화
                    current_value = self._get_result_value(result).strip()
                    current_values[roi_id] = (result, current_value)

                    # 상세 디버그 로깅 (처음 10프레임)
                    if frame_count <= 10:
                        logger.info(
                            f"프레임 {frame_count}, ROI '{result.name}': text='{current_value}', confidence={result.confidence:.2f}"
                        )

                # ROI 처리 결과가 없으면 스킵
                if not current_values:
                    if frame_count <= 10:
                        logger.warning(
                            f"프레임 {frame_count}: 성공한 ROI 처리 결과가 없습니다 (총 {len(results)}개 중)"
                        )
                    continue

                # 첫 번째 유효한 프레임은 기준값 설정 및 캡쳐
                if is_first_valid_frame:
                    is_first_valid_frame = False
                    capture_count += 1
                    logger.info(
                        f"=== 첫 프레임 캡쳐 (기준값 설정): 프레임 {frame_count} ==="
                    )

                    for roi_id, (result, val) in current_values.items():
                        logger.info(f"  기준값 설정: ROI '{result.name}' = '{val}'")

                    # 기준값 캡쳐
                    self._create_capture_group(
                        session_id=session_id,
                        trigger_roi_id=list(current_values.keys())[0],
                        frame=current_frame,
                        frame_data=frame_data,
                        current_values=current_values,
                        prev_values=prev_values,
                        rois=rois,
                    )

                    # prev_values 초기화
                    for roi_id, (_, current_val) in current_values.items():
                        prev_values[roi_id] = current_val
                    continue

                # 트리거 ROI 변화 감지
                trigger_changed = False
                trigger_roi_id_for_capture = None
                changed_roi_info = []  # 변화된 ROI 정보 수집

                for trigger_id in trigger_roi_ids:
                    if trigger_id not in current_values:
                        if frame_count <= 10:
                            logger.debug(
                                f"프레임 {frame_count}: 트리거 ROI {trigger_id}가 current_values에 없음"
                            )
                        continue

                    result, current_val = current_values[trigger_id]
                    prev_val = prev_values.get(trigger_id)

                    # 디버그: 값 비교 상세 정보
                    if frame_count <= 10:
                        logger.debug(
                            f"프레임 {frame_count}, ROI '{result.name}' 비교: prev='{prev_val}' vs current='{current_val}'"
                        )

                    # 이전 값이 없거나 (새로운 ROI) 값이 변경된 경우
                    if prev_val is None:
                        # 새로운 ROI - 값이 있으면 변화로 감지
                        if current_val:
                            trigger_changed = True
                            trigger_roi_id_for_capture = trigger_id
                            changed_roi_info.append(
                                f"{result.name}: (새로운) → '{current_val}'"
                            )
                    elif current_val != prev_val:
                        # 값이 변경됨
                        # 둘 다 빈 문자열인 경우만 스킵 (이미 != 조건에서 걸러짐)
                        trigger_changed = True
                        trigger_roi_id_for_capture = trigger_id
                        changed_roi_info.append(
                            f"{result.name}: '{prev_val}' → '{current_val}'"
                        )

                # 트리거 발생 시 캡쳐 그룹 생성
                if trigger_changed and trigger_roi_id_for_capture:
                    capture_count += 1
                    logger.info(
                        f"=== 변화 감지 #{capture_count} (프레임 {frame_count}): {', '.join(changed_roi_info)} ==="
                    )

                    self._create_capture_group(
                        session_id=session_id,
                        trigger_roi_id=trigger_roi_id_for_capture,
                        frame=current_frame,
                        frame_data=frame_data,
                        current_values=current_values,
                        prev_values=prev_values,
                        rois=rois,
                    )

                # prev_values 업데이트 (모든 ROI)
                for roi_id, (_, current_val) in current_values.items():
                    prev_values[roi_id] = current_val

            # ★ 디버깅: for 루프 종료 확인
            logger.info(f"프레임 처리 루프 종료: 총 {frame_count}개 프레임 처리됨")

        logger.info(
            f"=== 분석 완료: 총 {frame_count}개 프레임, {capture_count}개 캡쳐 그룹 생성 ==="
        )

    def _get_result_value(self, result: ROIResult) -> str:
        """결과에서 비교용 값 추출 (공백 정규화 포함)"""
        if result.process_type == ROIProcessType.OCR:
            return result.text.strip() if result.text else ""
        elif result.process_type == ROIProcessType.CHART:
            return f"chart:{result.has_chart}"
        return ""

    def _create_capture_group(
        self,
        session_id: int,
        trigger_roi_id: str,
        frame,
        frame_data,
        current_values: dict[str, tuple[ROIResult, str]],
        prev_values: dict[str, str],
        rois: list,
    ) -> int:
        """트리거 시점에 캡쳐 그룹 생성

        1. 전체 프레임 이미지 1회 저장
        2. Capture 레코드 생성
        3. 모든 ROI 현재 값을 change_events로 저장 (capture_id로 연결)

        Args:
            session_id: 세션 ID
            trigger_roi_id: 트리거가 된 ROI ID (문자열)
            frame: 현재 프레임 이미지
            frame_data: 프레임 메타데이터
            current_values: 현재 프레임의 모든 ROI 값 {roi_id: (ROIResult, value)}
            prev_values: 이전 프레임의 ROI 값 {roi_id: value}
            rois: ROI 목록 (시각화용)

        Returns:
            생성된 capture_id
        """
        # 1. ROI 시각화 적용 후 프레임 이미지 저장
        visualized_frame = self.roi_visualizer.draw_rois(frame.copy(), rois)
        capture_result = self.capture_manager.save_capture(
            visualized_frame,  # 시각화 적용된 프레임
            session_id=session_id,
            roi_name=f"frame_{frame_data.frame_number}",
        )

        frame_path = ""
        if capture_result and capture_result.success:
            frame_path = str(capture_result.file_path)

        # 2. 트리거 ROI의 DB ID 확인
        trigger_result, _ = current_values.get(trigger_roi_id, (None, ""))
        if trigger_result is None:
            logger.warning(f"트리거 ROI 결과를 찾을 수 없음: {trigger_roi_id}")
            return -1

        trigger_db_roi_id = trigger_result.metadata.get("db_roi_id")
        if trigger_db_roi_id is None:
            logger.warning(f"트리거 ROI의 DB ID를 찾을 수 없음: {trigger_roi_id}")
            return -1

        # 3. Capture 레코드 생성
        capture_id = -1
        try:
            with self.db_manager as db:
                capture_data = CaptureCreate(
                    session_id=session_id,
                    trigger_roi_id=trigger_db_roi_id,
                    frame_path=frame_path,
                    frame_number=frame_data.frame_number,
                    timestamp_ms=frame_data.timestamp_ms,
                    metadata=json.dumps(
                        {
                            "trigger_roi_name": trigger_result.name,
                        }
                    ),
                )
                capture = db.create_capture(capture_data)
                capture_id = capture.id

                # 4. 모든 ROI의 change_events 생성 (capture_id로 그룹화)
                events_to_create = []
                for roi_id, (result, current_value) in current_values.items():
                    db_roi_id = result.metadata.get("db_roi_id")
                    if db_roi_id is None:
                        continue

                    prev_value = prev_values.get(roi_id, "")

                    # OCR vs Chart 분기
                    if result.process_type == ROIProcessType.OCR:
                        is_chart = False
                        confidence = result.confidence
                        extracted_text = result.text
                    else:
                        is_chart = result.has_chart
                        confidence = result.chart_confidence
                        extracted_text = None

                    event_data = ChangeEventCreate(
                        roi_id=db_roi_id,
                        session_id=session_id,
                        capture_id=capture_id,  # 그룹화 핵심!
                        previous_value=prev_value if prev_value else None,
                        current_value=current_value,
                        frame_path=frame_path,  # 동일한 프레임 경로 공유
                        extracted_text=extracted_text,
                        is_chart=is_chart,
                        confidence=confidence,
                        metadata=json.dumps(
                            {
                                "processing_time_ms": result.processing_time_ms,
                                "frame_number": frame_data.frame_number,
                                "timestamp_ms": frame_data.timestamp_ms,
                                "is_trigger": result.is_trigger,
                            }
                        ),
                    )
                    events_to_create.append(event_data)

                # 배치 생성
                if events_to_create:
                    db.create_change_events_batch(events_to_create)

                logger.debug(
                    f"캡쳐 그룹 생성: capture_id={capture_id}, "
                    f"frame={frame_data.frame_number}, events={len(events_to_create)}"
                )

        except Exception as e:
            logger.warning(f"캡쳐 그룹 생성 실패: {e}")
            return -1

        return capture_id

    def _print_summary(self, session_id: int) -> None:
        """분석 결과 요약 출력"""
        print("\n" + "=" * 60)
        print("분석 결과 요약")
        print("=" * 60)
        print(f"비디오: {self.video_path}")
        print(f"캡쳐 이미지 저장 위치: {self.output_dir / 'captures'}")
        print(f"데이터베이스: Oracle ({self.config.storage.db_host}:{self.config.storage.db_port}/{self.config.storage.db_service_name})")
        print("=" * 60 + "\n")


# ========================================
# 템플릿 관리 CLI
# ========================================


def list_templates(storage_config: StorageConfig) -> None:
    """템플릿 목록 출력"""
    with DatabaseManager.from_config(storage_config) as db:
        manager = ROITemplateManager(db)
        templates = manager.list_templates()

        if not templates:
            print("등록된 템플릿이 없습니다.")
            return

        print("\n" + "=" * 60)
        print("ROI 템플릿 목록")
        print("=" * 60)
        for info in templates:
            t = info.template
            print(f"  ID: {t.id}")
            print(f"  이름: {t.name}")
            print(f"  설명: {t.description or '(없음)'}")
            print(f"  ROI 개수: {info.roi_count}")
            print(f"  생성일: {t.created_at}")
            print("-" * 60)
        print()


def create_template_interactive(storage_config: StorageConfig) -> None:
    """템플릿 대화형 생성"""
    print("\n" + "=" * 60)
    print("새 ROI 템플릿 생성")
    print("=" * 60)

    name = input("템플릿 이름: ").strip()
    if not name:
        print("취소: 이름이 필요합니다.")
        return

    description = input("설명 (선택): ").strip() or None

    with DatabaseManager.from_config(storage_config) as db:
        manager = ROITemplateManager(db)
        template = manager.create_template(name=name, description=description)
        print(f"\n템플릿 생성 완료: ID={template.id}")

        # ROI 추가
        print("\nROI를 추가합니다. (종료: 빈 입력)")
        roi_count = 0

        while True:
            print(f"\n--- ROI #{roi_count + 1} ---")
            roi_name = input("ROI 이름 (빈 입력=종료): ").strip()
            if not roi_name:
                break

            roi_type_str = (
                input("유형 (numeric/text/chart) [numeric]: ").strip() or "numeric"
            )
            try:
                roi_type = ROIType(roi_type_str)
            except ValueError:
                print(f"잘못된 유형: {roi_type_str}")
                continue

            try:
                x = int(input("X 좌표: "))
                y = int(input("Y 좌표: "))
                width = int(input("너비: "))
                height = int(input("높이: "))
            except ValueError:
                print("잘못된 좌표 입력")
                continue

            manager.add_roi_to_template(
                template_id=template.id,
                name=roi_name,
                roi_type=roi_type,
                x=x,
                y=y,
                width=width,
                height=height,
            )
            roi_count += 1
            print(f"ROI '{roi_name}' 추가됨")

        print(f"\n템플릿 '{name}'에 {roi_count}개 ROI 추가 완료")


def show_template_rois(storage_config: StorageConfig, template_id: int) -> None:
    """템플릿의 ROI 목록 출력"""
    with DatabaseManager.from_config(storage_config) as db:
        manager = ROITemplateManager(db)
        template = manager.get_template(template_id)

        if not template:
            print(f"템플릿 ID={template_id}를 찾을 수 없습니다.")
            return

        rois = manager.get_template_rois(template_id)

        print("\n" + "=" * 60)
        print(f"템플릿: {template.name} (ID={template.id})")
        print("=" * 60)

        if not rois:
            print("등록된 ROI가 없습니다.")
            return

        for roi in rois:
            print(f"  [{roi.id}] {roi.name}")
            print(f"      유형: {roi.roi_type.value}")
            print(f"      좌표: ({roi.x}, {roi.y}) - {roi.width}x{roi.height}")
            print(f"      임계값: {roi.threshold}")
        print()


# ========================================
# 명령행 인터페이스
# ========================================


def parse_args() -> argparse.Namespace:
    """명령행 인자 파싱

    모든 선택적 인자의 default는 None으로 설정하여
    "사용자가 명시적으로 지정한 값"과 "미지정"을 구분합니다.
    실제 기본값은 Config dataclass에서 관리됩니다.
    """
    parser = argparse.ArgumentParser(
        description="산업용 비디오 모니터링 분석 프로그램 (v2.0 - YAML 설정 파일 지원)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용 예시:
  # 설정 파일 기반 실행
  python main.py --config config/config.yml

  # 환경별 실행
  python main.py --config config/config.yml --env prod

  # 설정 파일 + CLI 오버라이드
  python main.py --config config/config.yml --video input.mp4 --template-id 1

  # 순수 CLI 모드 (설정 파일 없이, 하위 호환)
  python main.py --video input.mp4 --output ./results --template-id 1

  # 템플릿 관리
  python main.py --list-templates
  python main.py --create-template
  python main.py --show-template 1

  # 배치 스케줄링 모드 (디렉토리 감시, 5분 주기)
  python main.py --watch-dir ./videos --template "모니터링 레이아웃 A"
        """,
    )

    # 설정 파일 옵션
    parser.add_argument(
        "--config",
        "-c",
        type=Path,
        default=None,
        help="설정 파일 경로 (기본값: config/config.yml 자동 탐색)",
    )

    parser.add_argument(
        "--env",
        "-e",
        type=str,
        default=None,
        help="환경 이름 (dev, prod 등). APP_ENV 환경변수도 지원",
    )

    # 비디오 분석 옵션 (default=None으로 센티넬 패턴 적용)
    parser.add_argument(
        "--video", "-v", type=Path, default=None, help="분석할 비디오 파일 경로"
    )

    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="결과 저장 디렉토리 (기본값: ./data)",
    )

    parser.add_argument(
        "--db-dsn",
        type=str,
        default=None,
        help="Oracle 데이터베이스 DSN (예: localhost:1521/ORCL)",
    )

    # 템플릿 선택
    parser.add_argument(
        "--template-id", type=int, default=None, help="사용할 ROI 템플릿 ID"
    )

    parser.add_argument(
        "--template", "-t", type=str, default=None, help="사용할 ROI 템플릿 이름"
    )

    # 처리 옵션
    parser.add_argument(
        "--gpu",
        action="store_true",
        default=None,
        help="GPU 가속 사용 (PaddleOCR)",
    )

    parser.add_argument(
        "--workers",
        "-w",
        type=int,
        default=None,
        help="병렬 처리 워커 수 (기본값: CPU 코어 수의 75%%)",
    )

    parser.add_argument(
        "--interval",
        "-i",
        type=float,
        default=None,
        help="프레임 분석 간격 (초, 기본값: 1.0)",
    )

    parser.add_argument(
        "--ssim-threshold",
        type=float,
        default=None,
        help="SSIM 변화 감지 임계값 (기본값: 0.95)",
    )

    parser.add_argument(
        "--confidence",
        type=float,
        default=None,
        help="OCR 신뢰도 임계값 (기본값: 0.7)",
    )

    parser.add_argument(
        "--auto-detect",
        action="store_true",
        default=None,
        help="첫 프레임에서 ROI 자동 탐지 (템플릿 불필요)",
    )

    parser.add_argument(
        "--roi-template",
        nargs="+",
        type=Path,
        default=None,
        metavar="IMAGE",
        help="색상으로 ROI를 표시한 템플릿 이미지 경로 (복수 가능). "
        "빨강=NUMERIC, 초록=TEXT, 파랑=CHART",
    )

    parser.add_argument(
        "--anchor-config",
        type=Path,
        default=None,
        metavar="YAML",
        help="앵커 기반 ROI 탐지 설정 YAML 파일 경로. "
        "고정 참조점(스니펫/텍스트)을 기준으로 상대 좌표로 ROI를 자동 탐지합니다.",
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        default=None,
        help="디버그 모드 활성화",
    )

    # 템플릿 관리 옵션 (명령 플래그 — 설정이 아닌 동작 지정)
    parser.add_argument(
        "--list-templates", action="store_true", help="등록된 템플릿 목록 출력"
    )

    parser.add_argument(
        "--create-template", action="store_true", help="새 템플릿 대화형 생성"
    )

    parser.add_argument(
        "--show-template", type=int, default=None, metavar="ID",
        help="템플릿 ROI 상세 정보 출력",
    )

    # 배치 스케줄링 옵션
    batch_group = parser.add_argument_group("배치 스케줄링 옵션")

    batch_group.add_argument(
        "--batch",
        action="store_true",
        default=None,
        help="배치 스케줄링 모드 활성화 (--watch-dir 지정 시 자동 활성화)",
    )

    batch_group.add_argument(
        "--watch-dir",
        type=Path,
        default=None,
        metavar="DIR",
        help="스캔할 비디오 디렉토리 경로 (지정 시 배치 모드 자동 실행)",
    )

    batch_group.add_argument(
        "--batch-interval",
        type=int,
        default=None,
        metavar="SECONDS",
        help="배치 사이클 간격 (초, 기본값: 300 = 5분)",
    )

    return parser.parse_args()


def build_cli_overrides(args: argparse.Namespace) -> dict[str, Any]:
    """None이 아닌 CLI 인자만 Config.with_overrides() 형식의 dict로 변환

    Args:
        args: parse_args()의 반환값

    Returns:
        dot 표기법 키를 사용하는 오버라이드 딕셔너리
    """
    overrides: dict[str, Any] = {}

    # 최상위 필드
    if args.video is not None:
        overrides["video_path"] = str(args.video)
    if args.debug is not None:
        overrides["debug"] = args.debug

    # template 섹션
    if args.template_id is not None:
        overrides["template.id"] = args.template_id
    if args.template is not None:
        overrides["template.name"] = args.template

    # detection 섹션
    if args.auto_detect is not None:
        overrides["detection.auto_detect"] = args.auto_detect
    if args.ssim_threshold is not None:
        overrides["detection.ssim_threshold"] = args.ssim_threshold
    if args.confidence is not None:
        overrides["detection.confidence_threshold"] = args.confidence
    if args.roi_template is not None:
        overrides["detection.roi_template_paths"] = [
            str(p) for p in args.roi_template
        ]
    if args.anchor_config is not None:
        overrides["detection.anchor_config_path"] = str(args.anchor_config)

    # processing 섹션
    if args.gpu is not None:
        overrides["processing.use_gpu"] = args.gpu
    if args.workers is not None:
        overrides["processing.max_workers"] = args.workers
    if args.interval is not None:
        overrides["processing.default_interval_sec"] = args.interval

    # storage 섹션
    if args.output is not None:
        overrides["storage.output_dir"] = str(args.output)
    if args.db_dsn is not None:
        overrides["storage.db_dsn"] = args.db_dsn

    # batch 섹션
    if args.batch is not None:
        overrides["batch.enabled"] = args.batch
    if args.watch_dir is not None:
        overrides["batch.watch_dir"] = str(args.watch_dir)
        overrides["batch.enabled"] = True  # --watch-dir은 batch 모드 암시
    if args.batch_interval is not None:
        overrides["batch.interval_seconds"] = args.batch_interval

    return overrides


def main():
    """메인 진입점

    설정 우선순위: CLI args > config.{env}.yml > config.yml > 환경변수 > 기본값
    """
    args = parse_args()

    # ★ Windows DLL 충돌 방지: 프로세스 최초 시점에서 torch DLL 선점 로드
    # paddleocr/albumentations가 torch를 간접 import하기 전에
    # torch의 libiomp5md.dll을 먼저 로드하여 DLL 버전 충돌 방지
    try:
        import torch  # noqa: F401 - Windows DLL preload at process start
    except (ImportError, OSError):
        pass  # torch 미설치 또는 로드 실패 시 무시

    # ===== 설정 로드 (YAML + CLI 오버라이드 병합) =====
    cli_overrides = build_cli_overrides(args)
    try:
        config = Config.load(
            config_path=args.config,
            env=args.env,
            cli_overrides=cli_overrides,
        )
    except FileNotFoundError as e:
        print(f"오류: {e}")
        sys.exit(1)
    except ValueError as e:
        print(f"설정 오류: {e}")
        sys.exit(1)

    # 디버그 모드
    if config.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    # 템플릿 관리 명령
    if args.list_templates:
        list_templates(config.storage)
        return

    if args.create_template:
        create_template_interactive(config.storage)
        return

    if args.show_template:
        show_template_rois(config.storage, args.show_template)
        return

    # 배치 모드
    if config.batch.enabled or config.batch.watch_dir:
        _run_batch_mode(config)
        return

    # 단일 비디오 분석
    if not config.video_path:
        print("오류: --video 또는 config 파일에 video_path가 필요합니다.")
        print("  설정 파일: python main.py --config config/config.yml")
        print("  단일 분석: python main.py --video input.mp4 --template-id 1")
        print("  배치 모드: python main.py --watch-dir ./videos --template-id 1")
        print("  도움말: python main.py --help")
        sys.exit(1)

    _run_single_analysis(config)


def _run_batch_mode(config: Config) -> None:
    """배치 스케줄링 모드 실행

    Args:
        config: 병합된 설정 객체
    """
    if not config.batch.watch_dir:
        print("오류: 배치 모드에는 --watch-dir 또는 config의 batch.watch_dir이 필요합니다.")
        sys.exit(1)

    if not config.batch.watch_dir.exists():
        print(f"오류: watch-dir이 존재하지 않습니다: {config.batch.watch_dir}")
        sys.exit(1)

    if not config.template.id and not config.template.name:
        print("오류: 배치 모드에는 --template 또는 --template-id가 필요합니다.")
        sys.exit(1)

    # --video와 --watch-dir 동시 지정 방지
    if config.video_path and config.batch.watch_dir:
        print("오류: video_path와 batch.watch_dir은 동시에 사용할 수 없습니다.")
        print("  단일 분석: python main.py --video input.mp4 --template-id 1")
        print("  배치 모드: python main.py --watch-dir ./videos --template-id 1")
        sys.exit(1)

    from .batch_scheduler import BatchScheduler

    scheduler = BatchScheduler.from_config(config)
    scheduler.run()


def _run_single_analysis(config: Config) -> None:
    """단일 비디오 분석 실행

    Args:
        config: 병합된 설정 객체
    """
    if not config.video_path.exists():
        logger.error(f"비디오 파일을 찾을 수 없습니다: {config.video_path}")
        sys.exit(1)

    # ROI 템플릿 이미지 파일 존재 확인
    for path_str in config.detection.roi_template_paths:
        path = Path(path_str)
        if not path.exists():
            print(f"오류: 템플릿 이미지를 찾을 수 없습니다: {path}")
            sys.exit(1)

    # 앵커 설정 파일 존재 확인
    if config.detection.anchor_config_path:
        anchor_path = Path(config.detection.anchor_config_path)
        if not anchor_path.exists():
            print(f"오류: 앵커 설정 파일을 찾을 수 없습니다: {anchor_path}")
            sys.exit(1)

    # ROI 탐지 방법 확인
    if (
        not config.detection.roi_template_paths
        and not config.detection.auto_detect
        and not config.template.id
        and not config.template.name
        and not config.detection.anchor_config_path
    ):
        print(
            "오류: --template-id, --template, --auto-detect, --roi-template, "
            "또는 --anchor-config 옵션이 필요합니다."
        )
        print("템플릿 목록 확인: python main.py --list-templates")
        print("템플릿 생성: python main.py --create-template")
        print("자동 ROI 탐지: python main.py --video input.mp4 --auto-detect")
        print(
            "색상 마커 ROI: python main.py --video input.mp4 --roi-template annotated.png"
        )
        print(
            "앵커 기반 ROI: python main.py --video input.mp4 --anchor-config config.yaml"
        )
        sys.exit(1)

    try:
        app = VideoAnalyzerApp.from_config(config)
        app.run()

    except KeyboardInterrupt:
        logger.info("사용자에 의해 분석이 중단되었습니다.")
        sys.exit(0)
    except Exception as e:
        logger.exception(f"분석 중 오류 발생: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
