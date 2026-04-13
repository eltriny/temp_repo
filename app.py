"""VideoAnalyzerApp — 비디오 분석 애플리케이션 코어.

src/main.py에서 분리된 VideoAnalyzerApp 클래스. import 시점 side effect 없음.
모듈 로거만 획득하고, 로깅 설정은 entrypoint(src/main.py)가 수행.
"""

from __future__ import annotations

import json
import logging
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Optional

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

logger = logging.getLogger(__name__)


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
            self._analyze_frames_parallel(session_id, rois, db=db)

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

        # B10: template_manager 는 run() 내부에서만 초기화되므로 None 가드 필요.
        if self.template_manager is None:
            raise RuntimeError(
                "_detect_rois_from_anchor must be called from within run() "
                "context where template_manager is initialized"
            )

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

        # B10: template_manager None 가드
        if self.template_manager is None:
            raise RuntimeError(
                "_detect_rois_from_first_frame must be called from within run() "
                "context where template_manager is initialized"
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

        # B10: template_manager None 가드
        if self.template_manager is None:
            raise RuntimeError(
                "_detect_rois_from_color_template must be called from within run() "
                "context where template_manager is initialized"
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

    def _analyze_frames_parallel(
        self,
        session_id: int,
        rois: list,
        *,
        db: "DatabaseManager",
    ) -> None:
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
            self._active_processor = processor
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
                        db=db,
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
                        db=db,
                    )

                # prev_values 업데이트 (모든 ROI)
                for roi_id, (_, current_val) in current_values.items():
                    prev_values[roi_id] = current_val

            # ★ 디버깅: for 루프 종료 확인
            logger.info(f"프레임 처리 루프 종료: 총 {frame_count}개 프레임 처리됨")

        self._active_processor = None
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
        *,
        db: "DatabaseManager",
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

        # 3. Capture 레코드 생성 + change_events 배치 INSERT 를 단일 트랜잭션에 묶는다.
        # B4: 기존 중첩 ``with self.db_manager`` 는 새 트랜잭션 경계를 열지 않아
        # capture INSERT 성공 후 events INSERT 가 실패하면 고아 capture 가 남았다.
        # 이제 호출자의 ``db`` 를 재사용하고 ``db.transaction()`` 으로 원자성을 보장한다.
        capture_id = -1
        try:
            with db.transaction():
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
        print(f"데이터베이스: PostgreSQL ({self.config.storage.db_host}:{self.config.storage.db_port}/{self.config.storage.db_name})")
        print("=" * 60 + "\n")

    def close(self) -> None:
        """앱 리소스 LIFO 정리.

        호출 순서: ParallelProcessor → CaptureManager → (DB는 Facade가 풀 유지하므로 건드리지 않음)
        각 단계는 예외를 삼키고 로깅만 (cleanup 도중 에러가 다른 cleanup을 막지 않도록).
        """
        # ParallelProcessor (run() 중 with 블록으로 관리되므로 보통 이미 닫혀있음)
        processor = getattr(self, "_active_processor", None)
        if processor is not None:
            try:
                processor.shutdown(wait=True)
            except Exception:
                logger.exception("ParallelProcessor close 실패")
            self._active_processor = None

        # CaptureManager
        try:
            if hasattr(self, "capture_manager") and self.capture_manager is not None:
                self.capture_manager.close()
        except Exception:
            logger.exception("CaptureManager close 실패")

