"""병렬 ROI 처리기 - ROI별 병렬 처리 오케스트레이터

이 모듈은 사전 정의된 ROI를 병렬로 처리합니다.
각 ROI는 별도 프로세스에서 OCR 또는 차트 감지를 수행합니다.

주요 기능:
    - ROI별 병렬 처리 (ThreadPoolExecutor)
    - OCR 영역과 차트 영역 분리 처리
    - 결과 수집 및 병합
    - 에러 격리 (한 ROI 실패가 다른 ROI에 영향 없음)

아키텍처:
    ┌─────────────────────────────────────────┐
    │         ParallelProcessor               │
    ├──────────┬──────────┬──────────┬────────┤
    │ Worker 1 │ Worker 2 │ Worker 3 │  ...   │
    │ (OCR)    │ (OCR)    │ (Chart)  │        │
    └──────────┴──────────┴──────────┴────────┘

사용 예시:
    >>> from core.parallel_processor import ParallelProcessor, ProcessorConfig
    >>> 
    >>> config = ProcessorConfig(max_workers=8)
    >>> processor = ParallelProcessor(config)
    >>> 
    >>> # 프레임에서 ROI 병렬 처리
    >>> results = processor.process_frame(frame, rois)
    >>> 
    >>> for roi_id, result in results.items():
    ...     if result.roi_type == 'ocr':
    ...         print(f"{roi_id}: {result.text}")
    ...     elif result.roi_type == 'chart':
    ...         print(f"{roi_id}: 차트 존재={result.has_chart}")
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor, Future, as_completed
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional, Callable
import multiprocessing as mp
import threading

import cv2
import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


# ========================================
# 단일 OCR 인스턴스 관리 (PaddleOCR 스레드 안전성 문제 대응)
# ========================================

_ocr_instance: Optional["PaddleOCR"] = None
_ocr_init_lock = threading.Lock()
_ocr_config: Optional["ProcessorConfig"] = None

# TextCorrector 인스턴스 (지연 초기화)
_text_corrector: Optional["TextCorrector"] = None
_text_corrector_enabled: bool = False


def _init_ocr_config(config: "ProcessorConfig") -> None:
    """OCR 설정 초기화 (메인 스레드에서 호출)

    Args:
        config: ProcessorConfig 설정 객체
    """
    global _ocr_config
    _ocr_config = config
    logger.debug("OCR 설정 초기화: language=%s, use_space_char=%s",
                 config.language, config.use_space_char)


def _init_text_corrector(enabled: bool) -> None:
    """텍스트 교정기 초기화 (메인 스레드에서 호출)

    Args:
        enabled: TextCorrector 활성화 여부
    """
    global _text_corrector, _text_corrector_enabled
    _text_corrector_enabled = enabled
    if enabled and _text_corrector is None:
        from src.ocr.text_corrector import TextCorrector, TextCorrectionConfig
        _text_corrector = TextCorrector(TextCorrectionConfig())
        logger.info("TextCorrector 초기화 완료")


def _get_ocr(use_gpu: bool) -> "PaddleOCR":
    """단일 PaddleOCR 인스턴스 획득 (지연 초기화)

    PaddleOCR은 스레드 안전하지 않으므로 단일 인스턴스를 사용합니다.
    max_workers=1과 함께 사용하여 동시 접근을 방지합니다.

    Args:
        use_gpu: GPU 사용 여부

    Returns:
        PaddleOCR 인스턴스
    """
    global _ocr_instance, _ocr_config
    if _ocr_instance is None:
        with _ocr_init_lock:
            if _ocr_instance is None:
                # ★ Windows에서 PyTorch DLL(shm.dll) 로딩 문제 방지
                # paddleocr → albumentations → torch 경로로 간접 import 시
                # DLL 로딩 컨텍스트 문제가 발생하므로 torch를 먼저 직접 import
                try:
                    import torch  # noqa: F401 - Windows DLL initialization
                except (OSError, ImportError) as e:
                    logger.warning("torch 로드 실패 (shm.dll): %s", e)
                    raise RuntimeError(
                        "PaddleOCR 초기화에 필요한 PyTorch를 로드할 수 없습니다. "
                        "torch 재설치가 필요합니다: pip install torch --force-reinstall"
                    ) from e

                from paddleocr import PaddleOCR

                # ProcessorConfig에서 설정 읽기 (없으면 기본값)
                config = _ocr_config

                _ocr_instance = PaddleOCR(
                    use_angle_cls=config.use_angle_cls if config else True,
                    lang=config.language if config else 'ch',
                    use_gpu=use_gpu,
                    show_log=False,
                    # OCR 검출 파라미터
                    det_db_thresh=config.det_db_thresh if config else 0.25,
                    det_db_box_thresh=config.det_db_box_thresh if config else 0.6,
                    det_db_unclip_ratio=config.det_db_unclip_ratio if config else 1.8,
                    # 공백 문자 처리
                    use_space_char=config.use_space_char if config else False,
                )
                logger.info(
                    "PaddleOCR 인스턴스 생성 완료 (lang=%s, use_space_char=%s, det_db_thresh=%.2f)",
                    config.language if config else 'ch',
                    config.use_space_char if config else False,
                    config.det_db_thresh if config else 0.25,
                )
    return _ocr_instance


# ========================================
# 설정 및 데이터 클래스
# ========================================

class ROIProcessType(Enum):
    """ROI 처리 유형"""
    OCR = "ocr"           # 텍스트/숫자 추출
    CHART = "chart"       # 차트 존재 여부 감지


@dataclass
class ProcessorConfig:
    """병렬 처리기 설정

    Attributes:
        max_workers: 최대 워커 수 (None이면 CPU 코어 수)
        timeout_per_roi: ROI당 타임아웃 (초)
        use_gpu: GPU 사용 여부 (OCR에 영향)
        ocr_confidence_threshold: OCR 신뢰도 임계값

        # OCR 설정 (PaddleOCR에 전달)
        det_db_thresh: 텍스트 검출 임계값 (낮을수록 많이 검출)
        det_db_box_thresh: 박스 신뢰도 임계값
        det_db_unclip_ratio: 박스 확장 비율
        use_space_char: 공백 문자 인식 여부
        language: OCR 언어 (ch, en, korean 등)
        use_angle_cls: 각도 분류 사용 여부
        enable_text_correction: TextCorrector 활성화 여부
    """
    max_workers: Optional[int] = None
    timeout_per_roi: float = 10.0
    use_gpu: bool = False
    ocr_confidence_threshold: float = 0.7

    # OCR 검출 파라미터
    det_db_thresh: float = 0.25
    det_db_box_thresh: float = 0.6
    det_db_unclip_ratio: float = 1.8

    # OCR 인식 파라미터
    use_space_char: bool = False
    language: str = "ch"  # 기본: 중국어 모델 (영문/숫자 인식에 더 적합)
    use_angle_cls: bool = True

    # 텍스트 교정
    enable_text_correction: bool = True

    def __post_init__(self):
        if self.max_workers is None:
            # PaddleOCR이 스레드 안전하지 않으므로 기본 1개 워커 사용
            # 안정성 확보 후 점진적으로 증가 가능
            self.max_workers = 1


@dataclass
class ROITask:
    """ROI 처리 작업 정의
    
    Attributes:
        roi_id: ROI 고유 식별자
        name: ROI 이름
        process_type: 처리 유형 (OCR/CHART)
        x, y, width, height: 좌표 및 크기
        threshold: 변화 감지 임계값
        metadata: 추가 메타데이터
        is_trigger: 트리거 ROI 여부 (True면 이 ROI 변경 시 캡쳐 발생)
    """
    roi_id: str
    name: str
    process_type: ROIProcessType
    x: int
    y: int
    width: int
    height: int
    threshold: float = 0.1
    metadata: dict[str, Any] = field(default_factory=dict)
    is_trigger: bool = False


@dataclass
class ROIResult:
    """ROI 처리 결과
    
    Attributes:
        roi_id: ROI 고유 식별자
        name: ROI 이름
        process_type: 처리 유형
        success: 처리 성공 여부
        
        # OCR 결과 (process_type == OCR인 경우)
        text: 추출된 텍스트
        confidence: OCR 신뢰도
        
        # 차트 결과 (process_type == CHART인 경우)
        has_chart: 차트 존재 여부
        chart_confidence: 차트 감지 신뢰도
        
        # 공통
        processing_time_ms: 처리 시간 (밀리초)
        error: 에러 메시지 (실패 시)
        metadata: 추가 메타데이터 (db_roi_id 등)
        is_trigger: 트리거 ROI 여부
    """
    roi_id: str
    name: str
    process_type: ROIProcessType
    success: bool
    
    # OCR 결과
    text: str = ""
    confidence: float = 0.0
    
    # 차트 결과
    has_chart: bool = False
    chart_confidence: float = 0.0
    
    # 공통
    processing_time_ms: float = 0.0
    error: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)
    is_trigger: bool = False


# ========================================
# 워커 함수 (프로세스에서 실행)
# ========================================

def _process_ocr_roi(
    roi_image: NDArray[np.uint8],
    task: ROITask,
    use_gpu: bool,
    confidence_threshold: float,
) -> ROIResult:
    """OCR ROI 처리 워커 함수

    별도 스레드에서 실행되며, OCR 엔진을 로컬에서 초기화합니다.
    
    Args:
        roi_image: ROI 영역 이미지
        task: ROI 작업 정의
        use_gpu: GPU 사용 여부
        confidence_threshold: OCR 신뢰도 임계값
        
    Returns:
        ROIResult: 처리 결과
    """
    start_time = time.perf_counter()
    
    try:
        # 단일 OCR 인스턴스 획득 (max_workers=1이므로 동시 접근 없음)
        ocr = _get_ocr(use_gpu)

        # OCR 수행
        result = ocr.ocr(roi_image, cls=True)
        
        # 결과 파싱
        texts = []
        total_confidence = 0.0
        count = 0
        
        if result and result[0]:
            for line in result[0]:
                if line and len(line) >= 2:
                    text = line[1][0]
                    conf = line[1][1]
                    if conf >= confidence_threshold:
                        texts.append(text)
                        total_confidence += conf
                        count += 1

        combined_text = " ".join(texts)
        avg_confidence = total_confidence / count if count > 0 else 0.0

        # TextCorrector 적용 (활성화된 경우)
        if _text_corrector_enabled and _text_corrector is not None and combined_text:
            correction_result = _text_corrector.correct(combined_text)
            original_text = combined_text
            combined_text = correction_result.corrected_text

            # 교정으로 인한 신뢰도 조정 (0.0 ~ 1.0 범위 보장)
            if count > 0:
                avg_confidence = max(0.0, min(1.0, avg_confidence + correction_result.confidence_delta))

            if correction_result.was_corrected:
                logger.debug(
                    "텍스트 교정 적용 [%s]: '%s' → '%s' (%d개 교정)",
                    task.roi_id,
                    original_text,
                    combined_text,
                    correction_result.correction_count,
                )
        
        processing_time = (time.perf_counter() - start_time) * 1000
        
        return ROIResult(
            roi_id=task.roi_id,
            name=task.name,
            process_type=ROIProcessType.OCR,
            success=True,
            text=combined_text,
            confidence=avg_confidence,
            processing_time_ms=processing_time,
            metadata=task.metadata,
            is_trigger=task.is_trigger,
        )
        
    except Exception as e:
        processing_time = (time.perf_counter() - start_time) * 1000
        return ROIResult(
            roi_id=task.roi_id,
            name=task.name,
            process_type=ROIProcessType.OCR,
            success=False,
            error=str(e),
            processing_time_ms=processing_time,
            metadata=task.metadata,
            is_trigger=task.is_trigger,
        )


def _process_chart_roi(
    roi_image: NDArray[np.uint8],
    task: ROITask,
) -> ROIResult:
    """차트 ROI 처리 워커 함수

    별도 스레드에서 실행되어 차트 존재 여부를 감지합니다.
    
    Args:
        roi_image: ROI 영역 이미지
        task: ROI 작업 정의
        
    Returns:
        ROIResult: 처리 결과
    """
    start_time = time.perf_counter()
    
    try:
        # 간단한 엣지 기반 차트 감지
        # (전체 ChartDetector를 로드하지 않고 경량 분석)
        
        gray = cv2.cvtColor(roi_image, cv2.COLOR_BGR2GRAY) if len(roi_image.shape) == 3 else roi_image
        
        # 엣지 감지
        edges = cv2.Canny(gray, 50, 150)
        
        # Hough 라인 변환
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=50,
            minLineLength=30,
            maxLineGap=10,
        )
        
        # 차트 특성 분석
        has_chart = False
        chart_confidence = 0.0
        
        if lines is not None and len(lines) > 5:
            # 수평/수직 라인 비율 분석
            horizontal_count = 0
            vertical_count = 0
            diagonal_count = 0
            
            for line in lines:
                x1, y1, x2, y2 = line[0]
                angle = abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
                
                if angle < 15 or angle > 165:
                    horizontal_count += 1
                elif 75 < angle < 105:
                    vertical_count += 1
                else:
                    diagonal_count += 1
            
            # 차트는 보통 축(수평/수직)과 데이터 라인(대각)을 포함
            total_lines = len(lines)
            axis_ratio = (horizontal_count + vertical_count) / total_lines
            data_ratio = diagonal_count / total_lines
            
            # 축이 존재하고 데이터 라인도 있으면 차트로 판단
            if axis_ratio > 0.2 and data_ratio > 0.1:
                has_chart = True
                chart_confidence = min(0.5 + axis_ratio * 0.3 + data_ratio * 0.2, 1.0)
        
        processing_time = (time.perf_counter() - start_time) * 1000
        
        return ROIResult(
            roi_id=task.roi_id,
            name=task.name,
            process_type=ROIProcessType.CHART,
            success=True,
            has_chart=has_chart,
            chart_confidence=chart_confidence,
            processing_time_ms=processing_time,
            metadata=task.metadata,
            is_trigger=task.is_trigger,
        )
        
    except Exception as e:
        processing_time = (time.perf_counter() - start_time) * 1000
        return ROIResult(
            roi_id=task.roi_id,
            name=task.name,
            process_type=ROIProcessType.CHART,
            success=False,
            error=str(e),
            processing_time_ms=processing_time,
            metadata=task.metadata,
            is_trigger=task.is_trigger,
        )


# ========================================
# 메인 병렬 처리기 클래스
# ========================================

class ParallelProcessor:
    """ROI별 병렬 처리기
    
    사전 정의된 ROI를 병렬로 처리합니다.
    각 ROI는 유형에 따라 OCR 또는 차트 감지를 수행합니다.
    
    Attributes:
        config: 처리기 설정
        _executor: ThreadPoolExecutor 인스턴스
    """
    
    def __init__(self, config: Optional[ProcessorConfig] = None) -> None:
        """ParallelProcessor 초기화
        
        Args:
            config: 처리기 설정 (None이면 기본값 사용)
        """
        self.config = config or ProcessorConfig()
        self._executor: Optional[ThreadPoolExecutor] = None
        self._ocr_available: bool = True
        
        logger.info(
            f"ParallelProcessor 초기화: "
            f"max_workers={self.config.max_workers}, "
            f"use_gpu={self.config.use_gpu}"
        )
    
    def __enter__(self) -> ParallelProcessor:
        """컨텍스트 매니저 진입"""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """컨텍스트 매니저 종료"""
        self.shutdown()
    
    def start(self) -> None:
        """Executor 시작

        ThreadPoolExecutor를 사용하여 ROI 처리를 병렬화합니다.
        PaddleOCR의 C++ 백엔드는 GIL을 해제하므로
        ThreadPoolExecutor에서도 효과적인 병렬 처리가 가능합니다.

        Note:
            ProcessPoolExecutor + spawn 방식은 Windows에서 PyTorch DLL
            로딩 문제(shm.dll)를 일으킬 수 있어 ThreadPoolExecutor로 전환.

        Important:
            Windows에서 PyTorch DLL(shm.dll)은 메인 스레드에서 먼저 로드되어야 합니다.
            워커 스레드에서 처음 로드하면 WinError 127 오류가 발생할 수 있습니다.
            따라서 ThreadPoolExecutor 시작 전에 메인 스레드에서 OCR을 미리 초기화합니다.
        """
        if self._executor is None:
            # ★ 핵심: 메인 스레드에서 먼저 설정 및 PaddleOCR 초기화
            logger.info("메인 스레드에서 OCR 설정 및 사전 초기화 중...")

            # 1. OCR 설정 전달 (PaddleOCR 초기화 전에 호출)
            _init_ocr_config(self.config)

            # 2. TextCorrector 초기화
            _init_text_corrector(self.config.enable_text_correction)

            # 3. PaddleOCR 초기화 (Windows DLL 로딩 문제 방지)
            try:
                _get_ocr(self.config.use_gpu)
            except RuntimeError as e:
                logger.warning(
                    "PaddleOCR 사전 초기화 실패: %s. OCR ROI는 처리할 수 없습니다.", e
                )
                self._ocr_available = False

            self._executor = ThreadPoolExecutor(
                max_workers=self.config.max_workers,
            )
            logger.info(
                "ThreadPoolExecutor 시작 (workers=%d, text_correction=%s)",
                self.config.max_workers,
                self.config.enable_text_correction,
            )
    
    def shutdown(self, wait: bool = True) -> None:
        """Executor 종료
        
        Args:
            wait: 진행 중인 작업 완료 대기 여부
        """
        if self._executor is not None:
            self._executor.shutdown(wait=wait)
            self._executor = None
            logger.info("ThreadPoolExecutor 종료")
    
    def _extract_roi_region(
        self,
        frame: NDArray[np.uint8],
        task: ROITask,
    ) -> NDArray[np.uint8]:
        """프레임에서 ROI 영역 추출
        
        Args:
            frame: 원본 프레임
            task: ROI 작업 정의
            
        Returns:
            ROI 영역 이미지
        """
        h, w = frame.shape[:2]
        
        # 경계 클리핑
        x = max(0, min(task.x, w - 1))
        y = max(0, min(task.y, h - 1))
        x2 = max(x + 1, min(task.x + task.width, w))
        y2 = max(y + 1, min(task.y + task.height, h))
        
        return frame[y:y2, x:x2].copy()
    
    def process_frame(
        self,
        frame: NDArray[np.uint8],
        tasks: list[ROITask],
        callback: Optional[Callable[[ROIResult], None]] = None,
    ) -> dict[str, ROIResult]:
        """프레임에서 모든 ROI 병렬 처리
        
        Args:
            frame: 입력 프레임 (BGR)
            tasks: ROI 작업 리스트
            callback: 각 결과에 대한 콜백 함수 (선택)
            
        Returns:
            {roi_id: ROIResult} 딕셔너리
        """
        if not tasks:
            return {}
        
        if self._executor is None:
            self.start()
        
        start_time = time.perf_counter()
        results: dict[str, ROIResult] = {}
        futures: dict[Future, ROITask] = {}
        
        # 작업 제출
        for task in tasks:
            roi_image = self._extract_roi_region(frame, task)
            
            if task.process_type == ROIProcessType.OCR:
                future = self._executor.submit(
                    _process_ocr_roi,
                    roi_image,
                    task,
                    self.config.use_gpu,
                    self.config.ocr_confidence_threshold,
                )
            elif task.process_type == ROIProcessType.CHART:
                future = self._executor.submit(
                    _process_chart_roi,
                    roi_image,
                    task,
                )
            else:
                continue
            
            futures[future] = task
        
        # 결과 수집
        for future in as_completed(futures, timeout=self.config.timeout_per_roi * len(tasks)):
            task = futures[future]
            try:
                result = future.result(timeout=self.config.timeout_per_roi)
                results[result.roi_id] = result
                
                if callback:
                    callback(result)
                    
            except Exception as e:
                # 타임아웃 또는 예외 처리
                error_result = ROIResult(
                    roi_id=task.roi_id,
                    name=task.name,
                    process_type=task.process_type,
                    success=False,
                    error=str(e),
                )
                results[task.roi_id] = error_result
                
                if callback:
                    callback(error_result)
                
                logger.warning(f"ROI 처리 실패: {task.roi_id} - {e}")
        
        total_time = (time.perf_counter() - start_time) * 1000
        logger.debug(
            f"프레임 처리 완료: {len(results)}/{len(tasks)} ROI, "
            f"총 {total_time:.1f}ms"
        )
        
        return results
    
    def create_tasks_from_detection_rois(
        self,
        rois: list,
    ) -> list[ROITask]:
        """detection.roi_types.ROI 리스트에서 ROITask 생성
        
        ROITemplateManager.load_template_as_detection_rois()의
        결과를 ParallelProcessor용 작업으로 변환합니다.
        
        Args:
            rois: detection.roi_types.ROI 리스트
            
        Returns:
            ROITask 리스트
        """
        tasks = []
        
        for roi in rois:
            # ROI 타입에 따라 처리 유형 결정
            if roi.roi_type.value in ("numeric", "text"):
                process_type = ROIProcessType.OCR
            elif roi.roi_type.value == "chart":
                process_type = ROIProcessType.CHART
            else:
                # unknown 등 기타 타입은 OCR로 처리
                process_type = ROIProcessType.OCR
            
            # metadata에서 is_trigger 추출 (기본값 False)
            is_trigger = roi.metadata.get("is_trigger", False)
            
            task = ROITask(
                roi_id=roi.id,
                name=roi.label,
                process_type=process_type,
                x=roi.bbox.x,
                y=roi.bbox.y,
                width=roi.bbox.width,
                height=roi.bbox.height,
                threshold=roi.metadata.get("threshold", 0.1),
                metadata=roi.metadata,
                is_trigger=is_trigger,
            )
            tasks.append(task)
        
        return tasks


# ========================================
# 유틸리티 함수
# ========================================

def process_frame_simple(
    frame: NDArray[np.uint8],
    rois: list,
    max_workers: Optional[int] = None,
    use_gpu: bool = False,
) -> dict[str, ROIResult]:
    """단순 인터페이스: 프레임과 ROI 리스트로 바로 처리
    
    일회성 처리에 적합합니다. 반복 처리에는 ParallelProcessor를
    직접 사용하는 것이 효율적입니다.
    
    Args:
        frame: 입력 프레임
        rois: detection.roi_types.ROI 리스트
        max_workers: 최대 워커 수
        use_gpu: GPU 사용 여부
        
    Returns:
        {roi_id: ROIResult} 딕셔너리
    """
    config = ProcessorConfig(max_workers=max_workers, use_gpu=use_gpu)
    
    with ParallelProcessor(config) as processor:
        tasks = processor.create_tasks_from_detection_rois(rois)
        return processor.process_frame(frame, tasks)
