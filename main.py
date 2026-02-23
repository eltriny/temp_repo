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
import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional

# 상대 임포트를 위한 경로 설정 (스크립트 직접 실행 시)
if __name__ == "__main__" and __package__ is None:
    _src_dir = Path(__file__).resolve().parent
    _project_root = _src_dir.parent
    if str(_project_root) not in sys.path:
        sys.path.insert(0, str(_project_root))
    __package__ = "src"

import cv2

from .config import Config, ProcessingConfig, StorageConfig
from .core.video_processor import VideoProcessor
from .core.parallel_processor import (
    ParallelProcessor,
    ProcessorConfig,
    ROIResult,
    ROIProcessType,
)
from .detection.change_detector import ChangeDetector, ChangeDetectorConfig
from .detection.chart_detector import ChartDetector, ChartDetectorConfig
from .detection.dynamic_roi_detector import DynamicROIDetector, DynamicROIConfig
from .ocr.ocr_engine import OCREngine, OCRConfig
from .storage.database import (
    DatabaseManager,
    SessionCreate,
    ChangeEventCreate,
    CaptureCreate,
    ROICreate,
    ROIType,
)
from .storage.roi_template_manager import ROITemplateManager
from .storage.capture_manager import CaptureManager, CaptureConfig, ImageFormat
from .visualization import ROIVisualizer, VisualizationConfig, VisualizationStyle

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('video_analyzer.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)


class VideoAnalyzerApp:
    """비디오 분석 애플리케이션 메인 클래스 (v2.0)
    
    사전 정의된 ROI 템플릿을 사용하여 비디오를 분석합니다.
    ROI별 병렬 처리를 지원합니다.
    """

    def __init__(
        self,
        video_path: Path,
        output_dir: Path,
        db_path: Optional[Path] = None,
        template_id: Optional[int] = None,
        template_name: Optional[str] = None,
        use_gpu: bool = False,
        frame_interval: float = 1.0,
        ssim_threshold: float = 0.95,
        confidence_threshold: float = 0.7,
        max_workers: Optional[int] = None,
        auto_detect: bool = False,
    ):
        """
        Args:
            video_path: 분석할 비디오 파일 경로
            output_dir: 결과 저장 디렉토리
            db_path: SQLite 데이터베이스 경로
            template_id: 사용할 ROI 템플릿 ID
            template_name: 사용할 ROI 템플릿 이름 (template_id 우선)
            use_gpu: GPU 가속 사용 여부
            frame_interval: 프레임 분석 간격 (초)
            ssim_threshold: SSIM 변화 감지 임계값
            confidence_threshold: OCR 신뢰도 임계값
            max_workers: 병렬 처리 워커 수
            auto_detect: 첫 프레임 자동 ROI 탐지 모드
        """
        self.video_path = video_path
        self.output_dir = output_dir
        self.db_path = db_path or output_dir / "analysis.db"
        self.template_id = template_id
        self.template_name = template_name
        self.use_gpu = use_gpu
        self.max_workers = max_workers
        self.ssim_threshold = ssim_threshold
        self.confidence_threshold = confidence_threshold
        self.auto_detect = auto_detect

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
                db_path=self.db_path,
                output_dir=self.output_dir,
            ),
        )

        # 컴포넌트 초기화
        logger.info("컴포넌트 초기화 중...")

        # 비디오 프로세서
        self.video_processor = VideoProcessor(self.config)

        # 변화 감지기
        self.change_detector = ChangeDetector(
            ChangeDetectorConfig(ssim_threshold=ssim_threshold)
        )

        # 차트 감지기
        self.chart_detector = ChartDetector(ChartDetectorConfig())

        # OCR 엔진 - 산업용 디스플레이 최적화 설정 사용
        from dataclasses import replace
        from .ocr.ocr_engine import create_industrial_ocr_config

        base_ocr_config = create_industrial_ocr_config()
        self.ocr_config = replace(
            base_ocr_config,
            use_gpu=use_gpu,
            confidence_threshold=confidence_threshold,
        )
        self.ocr_engine = OCREngine(self.ocr_config)

        # 데이터베이스 매니저
        self.db_manager = DatabaseManager(str(self.db_path))

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
        logger.info(f"비디오 정보: {metadata.width}x{metadata.height}, "
                   f"{metadata.fps:.2f}fps, {metadata.duration_str}")

        # 데이터베이스 연결 및 템플릿 로드
        with self.db_manager as db:
            self.template_manager = ROITemplateManager(db)

            # 세션 생성
            session = db.create_session(SessionCreate(
                name=f"Analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                source_path=str(self.video_path),
            ))
            session_id = session.id
            logger.info(f"분석 세션 생성: ID={session_id}")

            # === ROI 획득: auto-detect 또는 템플릿 ===
            if self.auto_detect:
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

                rois = self.template_manager.load_session_rois_as_detection_rois(session_id)

            logger.info(f"{len(rois)}개의 ROI 로드됨")

            # 프레임 분석 루프 (병렬 처리)
            self._analyze_frames_parallel(session_id, rois)

            # 세션 종료
            db.update_session(session_id, is_active=False)

        logger.info("비디오 분석 완료")
        self._print_summary(session_id)

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
        import json

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
            config=DynamicROIConfig(enable_text_detection=True),
            ocr_engine=None,
        )

        try:
            detected_rois = dynamic_detector.detect(first_frame)
        except Exception as e:
            logger.warning("텍스트 ROI 탐지 중 예외, 파형 전용으로 재시도: %s", e)
            dynamic_detector = DynamicROIDetector(
                config=DynamicROIConfig(enable_text_detection=False),
            )
            detected_rois = dynamic_detector.detect(first_frame)

        if not detected_rois:
            logger.warning("자동 ROI 탐지 결과 없음")
            return []

        logger.info(f"자동 탐지된 ROI {len(detected_rois)}개:")

        # detection ROIType → database ROIType 매핑
        from .detection.roi_types import ROIType as DetROIType

        roi_type_map = {
            DetROIType.NUMERIC: ROIType.NUMERIC,
            DetROIType.TEXT: ROIType.TEXT,
            DetROIType.CHART: ROIType.CHART,
            DetROIType.UNKNOWN: ROIType.NUMERIC,
        }

        # DB에 저장
        for roi in detected_rois:
            db_roi_type = roi_type_map.get(roi.roi_type, ROIType.NUMERIC)

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

        logger.info(f"프레임 분석 시작 (트리거 기반 캡쳐, workers={self.processor_config.max_workers})")

        with ParallelProcessor(self.processor_config) as processor:
            # ROI를 작업 리스트로 변환
            tasks = processor.create_tasks_from_detection_rois(rois)
            
            logger.info(f"총 {len(tasks)}개 ROI 작업 생성됨")
            for task in tasks:
                logger.info(f"  - ROI: {task.name} (id={task.roi_id}, type={task.process_type.name}, is_trigger={task.is_trigger})")
            
            # 트리거 ROI 식별
            trigger_roi_ids = [task.roi_id for task in tasks if task.is_trigger]
            if not trigger_roi_ids:
                logger.warning("트리거 ROI가 없습니다. 모든 ROI 변화 시 캡쳐합니다.")
                # 트리거가 없으면 모든 OCR ROI를 트리거로 사용
                trigger_roi_ids = [task.roi_id for task in tasks if task.process_type == ROIProcessType.OCR]
                logger.info(f"OCR ROI {len(trigger_roi_ids)}개를 트리거로 사용: {trigger_roi_ids}")
            else:
                logger.info(f"트리거 ROI: {trigger_roi_ids}")

            # ★ 디버깅: for 루프 진입 전 로그
            logger.info("프레임 추출 제너레이터 시작...")

            for frame_data in self.video_processor.extract_frames():
                frame_count += 1

                # ★ 디버깅: 첫 프레임 진입 확인
                if frame_count == 1:
                    logger.info(f"첫 프레임 진입 확인: frame_number={frame_data.frame_number}, timestamp_ms={frame_data.timestamp_ms:.0f}")

                current_frame = frame_data.frame

                # 진행률 출력 (매 100프레임마다)
                if frame_count % 100 == 0:
                    progress, _ = self.video_processor.get_progress(frame_count)
                    logger.info(f"진행률: {progress:.1f}% (프레임: {frame_count}, 캡쳐: {capture_count})")

                # ROI별 병렬 처리
                # ★ 디버깅: process_frame 호출 전
                if frame_count <= 3:
                    logger.info(f"프레임 {frame_count}: ROI 처리 시작 (tasks={len(tasks)}개)")

                results = processor.process_frame(current_frame, tasks)

                # ★ 디버깅: process_frame 호출 후
                if frame_count <= 3:
                    logger.info(f"프레임 {frame_count}: ROI 처리 완료 (results={len(results)}개)")

                # 현재 프레임의 모든 ROI 값 수집
                current_values: dict[str, tuple[ROIResult, str]] = {}

                for roi_id, result in results.items():
                    if not result.success:
                        if frame_count <= 10:
                            logger.warning(f"프레임 {frame_count}, ROI 처리 실패: {roi_id} - {result.error}")
                        continue
                    
                    # 공백 제거하여 값 정규화
                    current_value = self._get_result_value(result).strip()
                    current_values[roi_id] = (result, current_value)
                    
                    # 상세 디버그 로깅 (처음 10프레임)
                    if frame_count <= 10:
                        logger.info(f"프레임 {frame_count}, ROI '{result.name}': text='{current_value}', confidence={result.confidence:.2f}")

                # ROI 처리 결과가 없으면 스킵
                if not current_values:
                    if frame_count <= 10:
                        logger.warning(f"프레임 {frame_count}: 성공한 ROI 처리 결과가 없습니다 (총 {len(results)}개 중)")
                    continue

                # 첫 번째 유효한 프레임은 기준값 설정 및 캡쳐
                if is_first_valid_frame:
                    is_first_valid_frame = False
                    capture_count += 1
                    logger.info(f"=== 첫 프레임 캡쳐 (기준값 설정): 프레임 {frame_count} ===")
                    
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
                            logger.debug(f"프레임 {frame_count}: 트리거 ROI {trigger_id}가 current_values에 없음")
                        continue
                        
                    result, current_val = current_values[trigger_id]
                    prev_val = prev_values.get(trigger_id)
                    
                    # 디버그: 값 비교 상세 정보
                    if frame_count <= 10:
                        logger.debug(f"프레임 {frame_count}, ROI '{result.name}' 비교: prev='{prev_val}' vs current='{current_val}'")
                    
                    # 이전 값이 없거나 (새로운 ROI) 값이 변경된 경우
                    if prev_val is None:
                        # 새로운 ROI - 값이 있으면 변화로 감지
                        if current_val:
                            trigger_changed = True
                            trigger_roi_id_for_capture = trigger_id
                            changed_roi_info.append(f"{result.name}: (새로운) → '{current_val}'")
                    elif current_val != prev_val:
                        # 값이 변경됨
                        # 둘 다 빈 문자열인 경우만 스킵 (이미 != 조건에서 걸러짐)
                        trigger_changed = True
                        trigger_roi_id_for_capture = trigger_id
                        changed_roi_info.append(f"{result.name}: '{prev_val}' → '{current_val}'")

                # 트리거 발생 시 캡쳐 그룹 생성
                if trigger_changed and trigger_roi_id_for_capture:
                    capture_count += 1
                    logger.info(f"=== 변화 감지 #{capture_count} (프레임 {frame_count}): {', '.join(changed_roi_info)} ===")
                    
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

        logger.info(f"=== 분석 완료: 총 {frame_count}개 프레임, {capture_count}개 캡쳐 그룹 생성 ===")

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
        import json

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
                    metadata=json.dumps({
                        "trigger_roi_name": trigger_result.name,
                    }),
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
                        metadata=json.dumps({
                            "processing_time_ms": result.processing_time_ms,
                            "frame_number": frame_data.frame_number,
                            "timestamp_ms": frame_data.timestamp_ms,
                            "is_trigger": result.is_trigger,
                        }),
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
        print("\n" + "="*60)
        print("분석 결과 요약")
        print("="*60)
        print(f"비디오: {self.video_path}")
        print(f"캡쳐 이미지 저장 위치: {self.output_dir / 'captures'}")
        print(f"데이터베이스: {self.db_path}")
        print("="*60 + "\n")


# ========================================
# 템플릿 관리 CLI
# ========================================

def list_templates(db_path: Path) -> None:
    """템플릿 목록 출력"""
    with DatabaseManager(str(db_path)) as db:
        manager = ROITemplateManager(db)
        templates = manager.list_templates()

        if not templates:
            print("등록된 템플릿이 없습니다.")
            return

        print("\n" + "="*60)
        print("ROI 템플릿 목록")
        print("="*60)
        for info in templates:
            t = info.template
            print(f"  ID: {t.id}")
            print(f"  이름: {t.name}")
            print(f"  설명: {t.description or '(없음)'}")
            print(f"  ROI 개수: {info.roi_count}")
            print(f"  생성일: {t.created_at}")
            print("-"*60)
        print()


def create_template_interactive(db_path: Path) -> None:
    """템플릿 대화형 생성"""
    print("\n" + "="*60)
    print("새 ROI 템플릿 생성")
    print("="*60)

    name = input("템플릿 이름: ").strip()
    if not name:
        print("취소: 이름이 필요합니다.")
        return

    description = input("설명 (선택): ").strip() or None

    with DatabaseManager(str(db_path)) as db:
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

            roi_type_str = input("유형 (numeric/text/chart) [numeric]: ").strip() or "numeric"
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


def show_template_rois(db_path: Path, template_id: int) -> None:
    """템플릿의 ROI 목록 출력"""
    with DatabaseManager(str(db_path)) as db:
        manager = ROITemplateManager(db)
        template = manager.get_template(template_id)

        if not template:
            print(f"템플릿 ID={template_id}를 찾을 수 없습니다.")
            return

        rois = manager.get_template_rois(template_id)

        print("\n" + "="*60)
        print(f"템플릿: {template.name} (ID={template.id})")
        print("="*60)

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
    """명령행 인자 파싱"""
    parser = argparse.ArgumentParser(
        description="산업용 비디오 모니터링 분석 프로그램 (v2.0 - 사전 정의 ROI 기반)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용 예시:
  # 비디오 분석 (템플릿 ID 지정)
  python main.py --video input.mp4 --output ./results --template-id 1

  # 비디오 분석 (템플릿 이름 지정)
  python main.py --video input.mp4 --output ./results --template "모니터링 레이아웃 A"

  # 템플릿 관리
  python main.py --list-templates
  python main.py --create-template
  python main.py --show-template 1

  # 배치 스케줄링 모드 (디렉토리 감시, 5분 주기)
  python main.py --watch-dir ./videos --template "모니터링 레이아웃 A"

  # 커스텀 주기 (10분)
  python main.py --watch-dir ./videos --template-id 1 --batch-interval 600
        """
    )

    # 비디오 분석 옵션
    parser.add_argument(
        "--video", "-v",
        type=Path,
        help="분석할 비디오 파일 경로"
    )

    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=Path("./data"),
        help="결과 저장 디렉토리 (기본값: ./data)"
    )

    parser.add_argument(
        "--db",
        type=Path,
        default=None,
        help="SQLite 데이터베이스 경로 (기본값: <output>/analysis.db)"
    )

    # 템플릿 선택
    parser.add_argument(
        "--template-id",
        type=int,
        help="사용할 ROI 템플릿 ID"
    )

    parser.add_argument(
        "--template", "-t",
        type=str,
        help="사용할 ROI 템플릿 이름"
    )

    # 처리 옵션
    parser.add_argument(
        "--gpu",
        action="store_true",
        help="GPU 가속 사용 (PaddleOCR)"
    )

    parser.add_argument(
        "--workers", "-w",
        type=int,
        default=None,
        help="병렬 처리 워커 수 (기본값: CPU 코어 수의 75%%)"
    )

    parser.add_argument(
        "--interval", "-i",
        type=float,
        default=1.0,
        help="프레임 분석 간격 (초, 기본값: 1.0)"
    )

    parser.add_argument(
        "--ssim-threshold",
        type=float,
        default=0.95,
        help="SSIM 변화 감지 임계값 (기본값: 0.95)"
    )

    parser.add_argument(
        "--confidence",
        type=float,
        default=0.7,
        help="OCR 신뢰도 임계값 (기본값: 0.7)"
    )

    parser.add_argument(
        "--auto-detect",
        action="store_true",
        help="첫 프레임에서 ROI 자동 탐지 (템플릿 불필요)"
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="디버그 모드 활성화"
    )

    # 템플릿 관리 옵션
    parser.add_argument(
        "--list-templates",
        action="store_true",
        help="등록된 템플릿 목록 출력"
    )

    parser.add_argument(
        "--create-template",
        action="store_true",
        help="새 템플릿 대화형 생성"
    )

    parser.add_argument(
        "--show-template",
        type=int,
        metavar="ID",
        help="템플릿 ROI 상세 정보 출력"
    )

    # 배치 스케줄링 옵션
    batch_group = parser.add_argument_group("배치 스케줄링 옵션")

    batch_group.add_argument(
        "--batch",
        action="store_true",
        help="배치 스케줄링 모드 활성화 (--watch-dir 지정 시 자동 활성화)",
    )

    batch_group.add_argument(
        "--watch-dir",
        type=Path,
        metavar="DIR",
        help="스캔할 비디오 디렉토리 경로 (지정 시 배치 모드 자동 실행)",
    )

    batch_group.add_argument(
        "--batch-interval",
        type=int,
        default=300,
        metavar="SECONDS",
        help="배치 사이클 간격 (초, 기본값: 300 = 5분)",
    )

    return parser.parse_args()


def main():
    """메인 진입점"""
    args = parse_args()

    # ★ Windows DLL 충돌 방지: 프로세스 최초 시점에서 torch DLL 선점 로드
    # paddleocr/albumentations가 torch를 간접 import하기 전에
    # torch의 libiomp5md.dll을 먼저 로드하여 DLL 버전 충돌 방지
    try:
        import torch  # noqa: F401 - Windows DLL preload at process start
    except (ImportError, OSError):
        pass  # torch 미설치 또는 로드 실패 시 무시

    # 디버그 모드
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    # 데이터베이스 경로 결정
    db_path = args.db or args.output / "analysis.db"

    # 템플릿 관리 명령
    if args.list_templates:
        list_templates(db_path)
        return

    if args.create_template:
        create_template_interactive(db_path)
        return

    if args.show_template:
        show_template_rois(db_path, args.show_template)
        return

    # --video와 --watch-dir 동시 지정 방지
    if args.video and args.watch_dir:
        print("오류: --video와 --watch-dir은 동시에 사용할 수 없습니다.")
        print("  단일 분석: python main.py --video input.mp4 --template-id 1")
        print("  배치 모드: python main.py --watch-dir ./videos --template-id 1")
        sys.exit(1)

    # 배치 스케줄링 모드: --batch 또는 --watch-dir 지정 시 진입
    if args.batch or args.watch_dir:
        if not args.watch_dir:
            print("오류: --batch 모드에는 --watch-dir이 필요합니다.")
            sys.exit(1)

        if not args.watch_dir.exists():
            print(f"오류: watch-dir이 존재하지 않습니다: {args.watch_dir}")
            sys.exit(1)

        if not args.template_id and not args.template:
            print("오류: 배치 모드에는 --template 또는 --template-id가 필요합니다.")
            sys.exit(1)

        from .batch_scheduler import BatchScheduler

        scheduler = BatchScheduler(
            watch_dir=args.watch_dir,
            output_dir=args.output,
            db_path=args.db,
            template_id=args.template_id,
            template_name=args.template,
            interval_seconds=args.batch_interval,
            use_gpu=args.gpu,
            frame_interval=args.interval,
            ssim_threshold=args.ssim_threshold,
            confidence_threshold=args.confidence,
            max_workers=args.workers,
        )
        scheduler.run()
        return

    # 단일 비디오 분석
    if not args.video:
        print("오류: --video 또는 --watch-dir 옵션이 필요합니다.")
        print("  단일 분석: python main.py --video input.mp4 --template-id 1")
        print("  배치 모드: python main.py --watch-dir ./videos --template-id 1")
        print("  도움말: python main.py --help")
        sys.exit(1)

    if not args.video.exists():
        logger.error(f"비디오 파일을 찾을 수 없습니다: {args.video}")
        sys.exit(1)

    if not args.auto_detect and not args.template_id and not args.template:
        print("오류: --template-id, --template, 또는 --auto-detect 옵션이 필요합니다.")
        print("템플릿 목록 확인: python main.py --list-templates")
        print("템플릿 생성: python main.py --create-template")
        print("자동 ROI 탐지: python main.py --video input.mp4 --auto-detect")
        sys.exit(1)

    try:
        app = VideoAnalyzerApp(
            video_path=args.video,
            output_dir=args.output,
            db_path=args.db,
            template_id=args.template_id,
            template_name=args.template,
            use_gpu=args.gpu,
            frame_interval=args.interval,
            ssim_threshold=args.ssim_threshold,
            confidence_threshold=args.confidence,
            max_workers=args.workers,
            auto_detect=args.auto_detect,
        )
        app.run()

    except KeyboardInterrupt:
        logger.info("사용자에 의해 분석이 중단되었습니다.")
        sys.exit(0)
    except Exception as e:
        logger.exception(f"분석 중 오류 발생: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
