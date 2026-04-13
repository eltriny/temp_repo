"""
데이터베이스 매니저 - 산업용 비디오 모니터링 시스템

분석 세션, ROI 정의, 변화 감지 이벤트를 관리하는 Oracle 데이터베이스
작업을 트랜잭션 안전성과 함께 제공합니다.

주요 기능:
    - 세션(Session) CRUD: 분석 세션 생성, 조회, 수정, 삭제
    - ROI 정의 CRUD: 관심 영역 정의 관리
    - 변화 이벤트 CRUD: 감지된 변화 기록 관리
    - 트랜잭션 관리: 자동 커밋/롤백 지원
    - 스레드 안전: 멀티스레드 환경 지원

설계 패턴:
    - 컨텍스트 매니저: with 문을 통한 자원 관리
    - DTO (Data Transfer Object): 생성 요청용 데이터 클래스
    - Repository 패턴: 데이터 접근 로직 캡슐화

데이터베이스 스키마:
    - sessions: 분석 세션 정보 (비디오 소스, 생성일시 등)
    - roi_definitions: ROI 정의 (좌표, 타입, 임계값 등)
    - change_events: 변화 감지 이벤트 (이전값, 현재값, 신뢰도 등)

사용 예시:
    >>> with DatabaseManager(user="app", password="pw", dsn="localhost:1521/ORCL") as db:
    ...     session = db.create_session(SessionCreate(name="test", source_path="/video.mp4"))
    ...     events = db.get_events_by_session(session.id)
"""

from __future__ import annotations

import logging
import re
import threading

import oracledb

logger = logging.getLogger(__name__)

from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Generator, Optional, Sequence, Union


def _parse_datetime(value: Union[str, datetime, None]) -> datetime:
    """데이터베이스 저장 형식에서 datetime 객체로 변환

    Oracle은 TIMESTAMP를 datetime 객체로 직접 반환하지만,
    문자열로 들어올 수 있는 경우를 방어적으로 처리합니다.

    Args:
        value: datetime 문자열 또는 datetime 객체

    Returns:
        파싱된 datetime 객체

    Raises:
        ValueError: 파싱할 수 없는 값인 경우
    """
    if value is None:
        return datetime.now()
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value)
        except ValueError:
            return datetime.strptime(value, "%Y-%m-%d %H:%M:%S")
    raise ValueError(f"{type(value)}에서 datetime 파싱 불가: {value}")


def _convert_placeholders(sql: str) -> str:
    """SQLite 스타일 ? 플레이스홀더를 Oracle :1, :2, ... 형식으로 변환

    Args:
        sql: ? 플레이스홀더를 포함한 SQL 문자열

    Returns:
        Oracle 바인드 변수 형식으로 변환된 SQL
    """
    counter = [0]

    def replacer(match: re.Match) -> str:
        counter[0] += 1
        return f":{counter[0]}"

    return re.sub(r"\?", replacer, sql)


def _make_dict_factory(cursor: oracledb.Cursor):
    """oracledb 커서에 대한 dict row factory 생성

    Oracle 결과를 row["column_name"] 딕셔너리 스타일로 접근할 수 있게 합니다.
    """
    column_names = [d[0].lower() for d in cursor.description]

    def create_row(*args):
        return dict(zip(column_names, args))

    return create_row


class VideoFileStatus(Enum):
    """비디오 파일 처리 상태 열거형

    배치 프로그램에서 비디오 파일의 처리 상태를 추적합니다.

    Attributes:
        PENDING: 대기 중 (발견되었으나 아직 처리 시작 안 됨)
        PROCESSING: 처리 중 (현재 분석 진행 중)
        COMPLETED: 완료 (분석 성공)
        FAILED: 실패 (오류로 인한 분석 실패)
    """

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class ROIType(Enum):
    """관심 영역(ROI) 타입 열거형

    산업용 디스플레이에서 감지할 수 있는 영역 유형을 정의합니다.

    Attributes:
        NUMERIC: 숫자 표시 영역 (7세그먼트, LCD 숫자 등)
        TEXT: 텍스트 표시 영역 (상태 메시지, 라벨 등)
        CHART: 차트/그래프 영역 (트렌드, 히스토그램 등)
        INDICATOR: 지시등 영역 (ON/OFF 상태 표시)
        GAUGE: 게이지 영역 (아날로그 미터, 바 그래프 등)
    """

    NUMERIC = "numeric"
    TEXT = "text"
    CHART = "chart"
    INDICATOR = "indicator"
    GAUGE = "gauge"


@dataclass(frozen=True)
class VideoFile:
    """비디오 파일 엔티티

    배치 프로그램에서 관리하는 비디오 파일 정보입니다.

    Attributes:
        id: 파일 고유 식별자 (자동 생성)
        file_path: 비디오 파일 전체 경로 (고유)
        file_name: 파일명
        file_size: 파일 크기 (bytes)
        discovered_at: 파일 발견 시각
        status: 처리 상태 (pending/processing/completed/failed)
        session_id: 분석 세션 ID (완료 시 연결)
        started_at: 분석 시작 시각
        completed_at: 분석 완료 시각
        error_message: 오류 메시지 (실패 시)
        metadata: 추가 메타데이터 (JSON 문자열)
    """

    id: int
    file_path: str
    file_name: str
    file_size: Optional[int]
    discovered_at: datetime
    status: VideoFileStatus
    session_id: Optional[int] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    metadata: Optional[str] = None


@dataclass(frozen=True)
class ROITemplate:
    """ROI 템플릿 엔티티

    여러 세션에서 재사용 가능한 ROI 정의 묶음입니다.
    세션과 독립적으로 존재하여 동일한 비디오 레이아웃에 반복 적용 가능합니다.

    Attributes:
        id: 템플릿 고유 식별자 (자동 생성)
        name: 템플릿 이름 (고유)
        description: 템플릿 설명
        created_at: 생성 시각
        updated_at: 수정 시각
    """

    id: int
    name: str
    description: Optional[str]
    created_at: datetime
    updated_at: datetime


@dataclass(frozen=True)
class Session:
    """분석 세션 엔티티

    하나의 비디오 분석 작업을 나타내는 불변 데이터 클래스입니다.
    세션은 여러 ROI와 변화 이벤트를 포함합니다.

    Attributes:
        id: 세션 고유 식별자 (자동 생성)
        name: 세션 이름 (사용자 지정)
        source_path: 분석 대상 비디오/이미지 경로
        created_at: 세션 생성 시각
        updated_at: 마지막 수정 시각
        is_active: 활성 상태 여부 (False면 아카이브됨)
        metadata: 추가 메타데이터 (JSON 문자열)
    """

    id: int
    name: str
    source_path: str
    created_at: datetime
    updated_at: datetime
    is_active: bool = True
    metadata: Optional[str] = None


@dataclass(frozen=True)
class ROIDefinition:
    """관심 영역(ROI) 정의 엔티티

    이미지 내에서 모니터링할 특정 영역을 정의하는 불변 데이터 클래스입니다.

    Attributes:
        id: ROI 고유 식별자 (자동 생성)
        session_id: 소속 세션 ID (외래 키)
        name: ROI 이름 (세션 내 고유)
        roi_type: ROI 유형 (숫자, 텍스트, 차트 등)
        x: 좌상단 X 좌표 (픽셀)
        y: 좌상단 Y 좌표 (픽셀)
        width: 영역 너비 (픽셀, 양수)
        height: 영역 높이 (픽셀, 양수)
        created_at: 생성 시각
        threshold: 변화 감지 임계값 (0.0-1.0, 기본 0.1)
        metadata: 추가 메타데이터 (JSON 문자열)
    """

    id: int
    session_id: int
    name: str
    roi_type: ROIType
    x: int
    y: int
    width: int
    height: int
    created_at: datetime
    threshold: float = 0.1
    metadata: Optional[str] = None


@dataclass(frozen=True)
class ChangeEvent:
    """변화 감지 이벤트 엔티티

    ROI에서 감지된 값 변화를 기록하는 불변 데이터 클래스입니다.

    Attributes:
        id: 이벤트 고유 식별자 (자동 생성)
        roi_id: 관련 ROI ID (외래 키)
        session_id: 소속 세션 ID (외래 키)
        timestamp: 변화 감지 시각
        previous_value: 이전 값 (첫 감지시 None)
        current_value: 현재 감지된 값
        frame_path: 변화 감지 시점의 프레임 이미지 경로
        extracted_text: OCR로 추출된 원본 텍스트
        is_chart: 차트 영역 여부
        confidence: 감지 신뢰도 (0.0-1.0)
        metadata: 추가 메타데이터 (JSON 문자열)
        capture_id: 소속 캡쳐 그룹 ID (그룹화용, None이면 미그룹화)
    """

    id: int
    roi_id: int
    session_id: int
    timestamp: datetime
    previous_value: Optional[str]
    current_value: str
    frame_path: Optional[str]
    extracted_text: Optional[str]
    is_chart: bool
    confidence: float
    metadata: Optional[str] = None
    capture_id: Optional[int] = None


@dataclass(frozen=True)
class Capture:
    """캡쳐 그룹 엔티티

    트리거 ROI 변경 시점에 생성되는 캡쳐 그룹을 나타내는 불변 데이터 클래스입니다.
    하나의 캡쳐 그룹은 해당 시점의 전체 프레임 이미지와
    모든 ROI의 현재 값(change_events)을 포함합니다.

    Attributes:
        id: 캡쳐 고유 식별자 (자동 생성)
        session_id: 소속 세션 ID (외래 키)
        trigger_roi_id: 트리거가 된 ROI ID (외래 키)
        frame_path: 전체 프레임 이미지 저장 경로
        frame_number: 비디오 내 프레임 번호
        timestamp_ms: 프레임의 타임스탬프 (밀리초)
        captured_at: 캡쳐 레코드 생성 시각
        metadata: 추가 메타데이터 (JSON 문자열)
    """

    id: int
    session_id: int
    trigger_roi_id: int
    frame_path: str
    frame_number: int
    timestamp_ms: float
    captured_at: datetime
    metadata: Optional[str] = None


@dataclass
class VideoFileCreate:
    """비디오 파일 생성용 DTO

    새 비디오 파일 등록 시 필요한 데이터를 포함합니다.

    Attributes:
        file_path: 비디오 파일 전체 경로
        file_name: 파일명
        file_size: 파일 크기 (bytes)
        metadata: 선택적 메타데이터
    """

    file_path: str
    file_name: str
    file_size: Optional[int] = None
    metadata: Optional[str] = None


@dataclass
class SessionCreate:
    """세션 생성용 DTO (Data Transfer Object)

    새 세션 생성 시 필요한 데이터만 포함합니다.
    ID와 타임스탬프는 데이터베이스에서 자동 생성됩니다.

    Attributes:
        name: 세션 이름
        source_path: 비디오/이미지 소스 경로
        metadata: 선택적 메타데이터
    """

    name: str
    source_path: str
    metadata: Optional[str] = None


@dataclass
class ROICreate:
    """ROI 정의 생성용 DTO

    새 ROI 정의 생성 시 필요한 데이터를 포함합니다.

    Attributes:
        session_id: 소속 세션 ID
        name: ROI 이름 (세션 내 고유해야 함)
        roi_type: ROI 유형
        x: 좌상단 X 좌표
        y: 좌상단 Y 좌표
        width: 영역 너비 (양수)
        height: 영역 높이 (양수)
        threshold: 변화 감지 임계값 (기본 0.1)
        metadata: 선택적 메타데이터
    """

    session_id: int
    name: str
    roi_type: ROIType
    x: int
    y: int
    width: int
    height: int
    threshold: float = 0.1
    metadata: Optional[str] = None


@dataclass
class ROITemplateCreate:
    """ROI 템플릿 생성용 DTO

    새 템플릿 생성 시 필요한 데이터를 포함합니다.

    Attributes:
        name: 템플릿 이름 (고유해야 함)
        description: 템플릿 설명 (선택)
    """

    name: str
    description: Optional[str] = None


@dataclass
class TemplateROICreate:
    """템플릿 내 ROI 정의 생성용 DTO

    템플릿에 ROI를 추가할 때 사용합니다.
    세션 독립적이므로 session_id가 없습니다.

    Attributes:
        template_id: 소속 템플릿 ID
        name: ROI 이름 (템플릿 내 고유)
        roi_type: ROI 유형 (ocr 또는 chart)
        x: 좌상단 X 좌표 (픽셀)
        y: 좌상단 Y 좌표 (픽셀)
        width: 영역 너비 (양수)
        height: 영역 높이 (양수)
        threshold: 변화 감지 임계값 (기본 0.1)
        metadata: 선택적 메타데이터 (JSON)
    """

    template_id: int
    name: str
    roi_type: ROIType
    x: int
    y: int
    width: int
    height: int
    threshold: float = 0.1
    metadata: Optional[str] = None


@dataclass(frozen=True)
class TemplateROI:
    """템플릿 내 ROI 정의 엔티티

    템플릿에 속한 개별 ROI 정의입니다.
    세션과 독립적으로 존재합니다.

    Attributes:
        id: ROI 고유 식별자
        template_id: 소속 템플릿 ID
        name: ROI 이름
        roi_type: ROI 유형
        x, y, width, height: 좌표 및 크기
        threshold: 변화 감지 임계값
        created_at: 생성 시각
        metadata: 추가 메타데이터
    """

    id: int
    template_id: int
    name: str
    roi_type: ROIType
    x: int
    y: int
    width: int
    height: int
    threshold: float
    created_at: datetime
    metadata: Optional[str] = None


@dataclass
class CaptureCreate:
    """캡쳐 그룹 생성용 DTO

    트리거 ROI 변경 시 새 캡쳐 그룹 생성에 필요한 데이터를 포함합니다.

    Attributes:
        session_id: 소속 세션 ID
        trigger_roi_id: 트리거가 된 ROI ID
        frame_path: 전체 프레임 이미지 저장 경로
        frame_number: 비디오 내 프레임 번호
        timestamp_ms: 프레임의 타임스탬프 (밀리초)
        metadata: 선택적 메타데이터 (JSON)
    """

    session_id: int
    trigger_roi_id: int
    frame_path: str
    frame_number: int
    timestamp_ms: float
    metadata: Optional[str] = None


@dataclass
class ChangeEventCreate:
    """변화 이벤트 생성용 DTO

    새 변화 이벤트 기록 시 필요한 데이터를 포함합니다.

    Attributes:
        roi_id: 관련 ROI ID
        session_id: 소속 세션 ID
        previous_value: 이전 값 (첫 감지시 None)
        current_value: 현재 감지된 값
        frame_path: 프레임 이미지 저장 경로
        extracted_text: OCR 추출 원본 텍스트
        is_chart: 차트 영역 여부
        confidence: 감지 신뢰도
        metadata: 선택적 메타데이터
        capture_id: 소속 캡쳐 그룹 ID (선택, 그룹화용)
    """

    roi_id: int
    session_id: int
    previous_value: Optional[str]
    current_value: str
    frame_path: Optional[str] = None
    extracted_text: Optional[str] = None
    is_chart: bool = False
    confidence: float = 1.0
    metadata: Optional[str] = None
    capture_id: Optional[int] = None


class DatabaseError(Exception):
    """데이터베이스 작업 기본 예외

    모든 데이터베이스 관련 예외의 기본 클래스입니다.
    """

    pass


class DatabaseConnectionError(DatabaseError):
    """데이터베이스 연결 실패 예외

    데이터베이스 접근 불가, 권한 오류 등의 경우 발생합니다.
    """

    pass


class DatabaseIntegrityError(DatabaseError):
    """데이터베이스 무결성 제약 위반 예외

    외래 키 제약, 고유 제약 등이 위반된 경우 발생합니다.
    """

    pass


class DatabaseManager:
    """Oracle 데이터베이스 매니저

    스레드 안전한 데이터베이스 작업을 제공합니다.
    세션, ROI 정의, 변화 이벤트에 대한 CRUD 작업과
    트랜잭션 관리, 컨텍스트 매니저를 지원합니다.

    설계 특징:
        - 컨텍스트 매니저: with 문으로 자동 연결/해제
        - 스레드 안전: RLock으로 동시 접근 제어
        - 트랜잭션: 자동 커밋/롤백 지원

    Attributes:
        _user: Oracle 사용자명
        _password: Oracle 비밀번호
        _dsn: Oracle DSN 문자열
        _connection: Oracle 연결 객체
        _lock: 스레드 동기화용 재진입 락

    Example:
        >>> # 컨텍스트 매니저 사용 (권장)
        >>> with DatabaseManager(user="app", password="pw", dsn="localhost:1521/ORCL") as db:
        ...     session = db.create_session(SessionCreate(...))
        ...     events = db.get_events_by_session(session.id)

        >>> # 수동 연결 관리
        >>> db = DatabaseManager(user="app", password="pw", dsn="localhost:1521/ORCL")
        >>> db.connect()
        >>> try:
        ...     session = db.create_session(SessionCreate(...))
        ... finally:
        ...     db.close()
    """

    # 클래스 레벨 커넥션 풀 (모든 인스턴스 공유)
    _pool: Optional[oracledb.ConnectionPool] = None
    _pool_lock: threading.Lock = threading.Lock()
    _pool_config: Optional[tuple] = None  # (user, dsn) for pool identity

    # ========================================
    # 데이터베이스 스키마 정의
    # ========================================
    # Oracle DDL (Data Definition Language)
    # 테이블, 인덱스, 제약 조건 정의
    # 각 DDL은 PL/SQL 블록으로 감싸서 이미 존재하는 경우 무시

    _TABLE_DDLS = [
        # -- sessions 테이블: 분석 세션 정보
        """
        BEGIN
            EXECUTE IMMEDIATE '
                CREATE TABLE sessions (
                    id NUMBER GENERATED BY DEFAULT AS IDENTITY PRIMARY KEY,
                    name VARCHAR2(500) NOT NULL,
                    source_path VARCHAR2(4000) NOT NULL,
                    created_at TIMESTAMP DEFAULT SYSTIMESTAMP,
                    updated_at TIMESTAMP DEFAULT SYSTIMESTAMP,
                    is_active NUMBER(1) DEFAULT 1,
                    metadata CLOB
                )
            ';
        EXCEPTION
            WHEN OTHERS THEN
                IF SQLCODE = -955 THEN NULL;
                ELSE RAISE;
                END IF;
        END;
        """,
        # -- roi_definitions 테이블: ROI 정의
        """
        BEGIN
            EXECUTE IMMEDIATE '
                CREATE TABLE roi_definitions (
                    id NUMBER GENERATED BY DEFAULT AS IDENTITY PRIMARY KEY,
                    session_id NUMBER(19) NOT NULL,
                    name VARCHAR2(500) NOT NULL,
                    roi_type VARCHAR2(100) NOT NULL,
                    x NUMBER(19) NOT NULL,
                    y NUMBER(19) NOT NULL,
                    width NUMBER(19) NOT NULL CHECK(width > 0),
                    height NUMBER(19) NOT NULL CHECK(height > 0),
                    threshold NUMBER DEFAULT 0.1 CHECK(threshold >= 0 AND threshold <= 1),
                    created_at TIMESTAMP DEFAULT SYSTIMESTAMP,
                    metadata CLOB,
                    CONSTRAINT fk_roi_session FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE,
                    CONSTRAINT uq_roi_session_name UNIQUE(session_id, name)
                )
            ';
        EXCEPTION
            WHEN OTHERS THEN
                IF SQLCODE = -955 THEN NULL;
                ELSE RAISE;
                END IF;
        END;
        """,
        # -- change_events 테이블: 변화 감지 이벤트
        """
        BEGIN
            EXECUTE IMMEDIATE '
                CREATE TABLE change_events (
                    id NUMBER GENERATED BY DEFAULT AS IDENTITY PRIMARY KEY,
                    roi_id NUMBER(19) NOT NULL,
                    session_id NUMBER(19) NOT NULL,
                    capture_id NUMBER(19),
                    timestamp TIMESTAMP DEFAULT SYSTIMESTAMP,
                    previous_value VARCHAR2(4000),
                    current_value VARCHAR2(4000) NOT NULL,
                    frame_path VARCHAR2(4000),
                    extracted_text VARCHAR2(4000),
                    is_chart NUMBER(1) DEFAULT 0,
                    confidence NUMBER DEFAULT 1.0 CHECK(confidence >= 0 AND confidence <= 1),
                    metadata CLOB,
                    CONSTRAINT fk_event_roi FOREIGN KEY (roi_id) REFERENCES roi_definitions(id) ON DELETE CASCADE,
                    CONSTRAINT fk_event_session FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE,
                    CONSTRAINT fk_event_capture FOREIGN KEY (capture_id) REFERENCES captures(id) ON DELETE SET NULL
                )
            ';
        EXCEPTION
            WHEN OTHERS THEN
                IF SQLCODE = -955 THEN NULL;
                ELSE RAISE;
                END IF;
        END;
        """,
        # -- roi_templates 테이블: ROI 템플릿 정의
        """
        BEGIN
            EXECUTE IMMEDIATE '
                CREATE TABLE roi_templates (
                    id NUMBER GENERATED BY DEFAULT AS IDENTITY PRIMARY KEY,
                    name VARCHAR2(500) NOT NULL UNIQUE,
                    description VARCHAR2(4000),
                    created_at TIMESTAMP DEFAULT SYSTIMESTAMP,
                    updated_at TIMESTAMP DEFAULT SYSTIMESTAMP
                )
            ';
        EXCEPTION
            WHEN OTHERS THEN
                IF SQLCODE = -955 THEN NULL;
                ELSE RAISE;
                END IF;
        END;
        """,
        # -- template_rois 테이블: 템플릿 내 ROI 정의
        """
        BEGIN
            EXECUTE IMMEDIATE '
                CREATE TABLE template_rois (
                    id NUMBER GENERATED BY DEFAULT AS IDENTITY PRIMARY KEY,
                    template_id NUMBER(19) NOT NULL,
                    name VARCHAR2(500) NOT NULL,
                    roi_type VARCHAR2(100) NOT NULL,
                    x NUMBER(19) NOT NULL,
                    y NUMBER(19) NOT NULL,
                    width NUMBER(19) NOT NULL CHECK(width > 0),
                    height NUMBER(19) NOT NULL CHECK(height > 0),
                    threshold NUMBER DEFAULT 0.1 CHECK(threshold >= 0 AND threshold <= 1),
                    created_at TIMESTAMP DEFAULT SYSTIMESTAMP,
                    metadata CLOB,
                    CONSTRAINT fk_tmplroi_template FOREIGN KEY (template_id) REFERENCES roi_templates(id) ON DELETE CASCADE,
                    CONSTRAINT uq_tmplroi_template_name UNIQUE(template_id, name)
                )
            ';
        EXCEPTION
            WHEN OTHERS THEN
                IF SQLCODE = -955 THEN NULL;
                ELSE RAISE;
                END IF;
        END;
        """,
        # -- video_files 테이블: 비디오 파일 목록 관리
        """
        BEGIN
            EXECUTE IMMEDIATE '
                CREATE TABLE video_files (
                    id NUMBER GENERATED BY DEFAULT AS IDENTITY PRIMARY KEY,
                    file_path VARCHAR2(4000) NOT NULL UNIQUE,
                    file_name VARCHAR2(500) NOT NULL,
                    file_size NUMBER(19),
                    discovered_at TIMESTAMP DEFAULT SYSTIMESTAMP,
                    status VARCHAR2(100) DEFAULT ''pending'' NOT NULL,
                    session_id NUMBER(19),
                    started_at TIMESTAMP,
                    completed_at TIMESTAMP,
                    error_message CLOB,
                    metadata CLOB,
                    CONSTRAINT fk_vf_session FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE SET NULL
                )
            ';
        EXCEPTION
            WHEN OTHERS THEN
                IF SQLCODE = -955 THEN NULL;
                ELSE RAISE;
                END IF;
        END;
        """,
        # -- captures 테이블: 캡쳐 그룹 정보
        """
        BEGIN
            EXECUTE IMMEDIATE '
                CREATE TABLE captures (
                    id NUMBER GENERATED BY DEFAULT AS IDENTITY PRIMARY KEY,
                    session_id NUMBER(19) NOT NULL,
                    trigger_roi_id NUMBER(19) NOT NULL,
                    frame_path VARCHAR2(4000) NOT NULL,
                    frame_number NUMBER(19) NOT NULL,
                    timestamp_ms NUMBER NOT NULL,
                    captured_at TIMESTAMP DEFAULT SYSTIMESTAMP,
                    metadata CLOB,
                    CONSTRAINT fk_cap_session FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE,
                    CONSTRAINT fk_cap_trigger_roi FOREIGN KEY (trigger_roi_id) REFERENCES roi_definitions(id) ON DELETE CASCADE
                )
            ';
        EXCEPTION
            WHEN OTHERS THEN
                IF SQLCODE = -955 THEN NULL;
                ELSE RAISE;
                END IF;
        END;
        """,
    ]

    _INDEX_DDLS = [
        "CREATE INDEX idx_roi_session ON roi_definitions(session_id)",
        "CREATE INDEX idx_events_session ON change_events(session_id)",
        "CREATE INDEX idx_events_roi ON change_events(roi_id)",
        "CREATE INDEX idx_events_timestamp ON change_events(timestamp)",
        "CREATE INDEX idx_events_capture ON change_events(capture_id)",
        "CREATE INDEX idx_template_rois_template ON template_rois(template_id)",
        "CREATE INDEX idx_video_files_status ON video_files(status)",
        "CREATE INDEX idx_video_files_discovered ON video_files(discovered_at)",
        "CREATE INDEX idx_captures_session ON captures(session_id)",
        "CREATE INDEX idx_captures_trigger_roi ON captures(trigger_roi_id)",
    ]

    def __init__(
        self,
        *,
        user: str = "",
        password: str = "",
        dsn: str = "",
        host: str = "localhost",
        port: int = 1521,
        service_name: str = "ORCL",
    ) -> None:
        """데이터베이스 매니저 초기화

        Args:
            user: Oracle 사용자명
            password: Oracle 비밀번호
            dsn: Oracle DSN 문자열 (직접 지정 시 host/port/service_name 대신 사용)
            host: Oracle 호스트 (dsn 미지정 시 사용)
            port: Oracle 포트 (dsn 미지정 시 사용)
            service_name: Oracle 서비스 이름 (dsn 미지정 시 사용)

        Note:
            초기화 시 연결이 생성되지 않습니다.
            connect() 또는 with 문으로 연결을 시작하세요.
            중첩 컨텍스트 매니저 사용을 지원합니다.
        """
        self._user = user
        self._password = password
        self._dsn = dsn
        self._host = host
        self._port = port
        self._service_name = service_name
        self._connection: Optional[oracledb.Connection] = None
        # 재진입 락: 같은 스레드에서 중첩 획득 허용
        self._lock = threading.RLock()
        # 참조 카운팅: 중첩 컨텍스트 매니저 지원
        self._ref_count = 0
        # ping 주기 제한 (30초마다)
        self._last_ping_time: float = 0.0

    @classmethod
    def from_config(cls, config) -> "DatabaseManager":
        """StorageConfig에서 DatabaseManager 생성

        Args:
            config: StorageConfig 인스턴스

        Returns:
            DatabaseManager 인스턴스
        """
        return cls(
            user=config.db_user,
            password=config.db_password,
            dsn=config.db_dsn,
            host=config.db_host,
            port=config.db_port,
            service_name=config.db_service_name,
        )

    def __enter__(self) -> "DatabaseManager":
        """컨텍스트 매니저 진입 - 연결 시작

        중첩 사용을 지원합니다. 첫 번째 진입 시에만 연결을 생성하고,
        참조 카운트를 증가시킵니다.
        """
        with self._lock:
            if self._ref_count == 0:
                self.connect()
            self._ref_count += 1
        return self

    def __exit__(
        self,
        exc_type: Optional[type],
        exc_val: Optional[Exception],
        exc_tb: Optional[Any],
    ) -> None:
        """컨텍스트 매니저 종료 - 자동 정리

        중첩 사용을 지원합니다. 마지막 컨텍스트가 종료될 때만
        실제로 연결을 닫습니다. 예외 발생 여부와 관계없이 안전하게 처리합니다.
        """
        with self._lock:
            self._ref_count -= 1
            if self._ref_count == 0:
                self.close()

    def connect(self) -> None:
        """데이터베이스 연결 및 스키마 초기화

        Oracle DB 커넥션 풀에서 연결을 획득하고 스키마가 없으면
        테이블과 인덱스를 생성합니다. 풀이 없거나 설정이 다르면 새 풀을 생성합니다.

        Raises:
            DatabaseConnectionError: 연결 실패 시
        """
        try:
            dsn = self._dsn or oracledb.makedsn(
                self._host, self._port, service_name=self._service_name
            )
            # 풀이 없거나 설정이 다르면 새 풀 생성
            pool_key = (self._user, dsn)
            if type(self)._pool is None or type(self)._pool_config != pool_key:
                with type(self)._pool_lock:
                    if type(self)._pool is None or type(self)._pool_config != pool_key:
                        if type(self)._pool is not None:
                            try:
                                type(self)._pool.close(force=True)
                            except oracledb.Error:
                                pass
                        type(self)._pool = oracledb.create_pool(
                            user=self._user,
                            password=self._password,
                            dsn=dsn,
                            min=2,
                            max=10,
                            increment=1,
                        )
                        type(self)._pool_config = pool_key
                        logger.info("DB 커넥션 풀 생성: min=2, max=10")

            self._connection = type(self)._pool.acquire()
            self._connection.autocommit = False
            self._initialize_schema()
        except oracledb.Error as e:
            raise DatabaseConnectionError(f"데이터베이스 연결 실패: {e}") from e

    def close(self) -> None:
        """데이터베이스 연결 안전 종료

        풀이 있으면 연결을 풀에 반환하고, 없으면 직접 닫습니다.
        예외가 발생해도 연결 참조는 해제됩니다.
        """
        with self._lock:
            if self._connection is not None:
                try:
                    pool = type(self)._pool
                    if pool is not None:
                        pool.release(self._connection)
                    else:
                        self._connection.close()
                finally:
                    self._connection = None

    @property
    def is_connected(self) -> bool:
        """데이터베이스 연결 상태 확인"""
        return self._connection is not None

    def _get_connection(self) -> oracledb.Connection:
        """활성 연결 객체 반환 (주기적 건전성 검사 포함)

        30초마다 ping()으로 연결 상태를 확인하고, 실패 시 자동 재연결합니다.
        매 호출마다 ping하면 네트워크 왕복 오버헤드가 과도하므로 시간 기반으로 제한합니다.

        Raises:
            DatabaseConnectionError: 연결되지 않았거나 재연결 실패 시
        """
        if self._connection is None:
            raise DatabaseConnectionError("데이터베이스에 연결되지 않음")
        # 30초마다 건전성 검사 (매 호출 ping은 과도한 오버헤드)
        import time
        now = time.monotonic()
        if now - self._last_ping_time > 30.0:
            try:
                self._connection.ping()
                self._last_ping_time = now
            except oracledb.Error:
                logger.warning("DB 연결 건전성 검사 실패. 재연결 시도...")
                self._reconnect()
                self._last_ping_time = time.monotonic()
        return self._connection

    def _reconnect(self) -> None:
        """스테일 연결 감지 시 풀에서 새 연결 획득"""
        old_conn = self._connection
        self._connection = None
        # 기존 연결 반환 시도
        if old_conn is not None:
            try:
                pool = type(self)._pool
                if pool is not None:
                    pool.release(old_conn)
                else:
                    old_conn.close()
            except oracledb.Error:
                pass
        # 새 연결 획득
        try:
            pool = type(self)._pool
            if pool is not None:
                self._connection = pool.acquire()
                self._connection.autocommit = False
                logger.info("DB 재연결 성공 (풀에서 새 연결 획득)")
            else:
                self.connect()
        except oracledb.Error as e:
            raise DatabaseConnectionError(f"DB 재연결 실패: {e}") from e

    def _initialize_schema(self) -> None:
        """데이터베이스 스키마 초기화

        테이블과 인덱스가 없으면 생성합니다.
        이미 존재하면 무시됩니다 (PL/SQL 예외 처리).
        """
        with self._lock:
            conn = self._get_connection()
            cursor = conn.cursor()
            try:
                # captures 테이블을 change_events보다 먼저 생성 (FK 참조)
                # _TABLE_DDLS 순서: sessions, roi_definitions, change_events,
                #   roi_templates, template_rois, video_files, captures
                # change_events가 captures를 참조하므로, captures를 먼저 생성해야 함
                # 순서 재배치: sessions(0), roi_definitions(1), captures(6),
                #   change_events(2), roi_templates(3), template_rois(4), video_files(5)
                ordered_indices = [0, 1, 6, 2, 3, 4, 5]
                for idx in ordered_indices:
                    cursor.execute(self._TABLE_DDLS[idx])

                # 인덱스 생성 (이미 존재하면 무시)
                for index_ddl in self._INDEX_DDLS:
                    try:
                        cursor.execute(f"""
                            BEGIN
                                EXECUTE IMMEDIATE '{index_ddl}';
                            EXCEPTION
                                WHEN OTHERS THEN
                                    IF SQLCODE = -955 OR SQLCODE = -1408 THEN NULL;
                                    ELSE RAISE;
                                    END IF;
                            END;
                        """)
                    except oracledb.Error:
                        pass  # 인덱스 생성 실패 시 무시

                conn.commit()
            finally:
                cursor.close()

    def _execute(
        self,
        sql: str,
        params: tuple | list = (),
        *,
        fetch: str = "none",
    ) -> Any:
        """SQL 실행 래퍼 - cursor 생성/rowfactory 설정을 캡슐화

        ? 플레이스홀더를 Oracle :n 형식으로 자동 변환합니다.

        Args:
            sql: SQL 문자열 (? 플레이스홀더 사용 가능)
            params: 바인드 파라미터
            fetch: "none" (INSERT/UPDATE/DELETE), "one" (단일 행), "all" (전체 행)

        Returns:
            fetch="none": 영향받은 행 수 (int)
            fetch="one": dict 또는 None
            fetch="all": list[dict]
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        try:
            oracle_sql = _convert_placeholders(sql)
            cursor.execute(oracle_sql, list(params))

            if fetch == "one":
                if cursor.description:
                    cursor.rowfactory = _make_dict_factory(cursor)
                return cursor.fetchone()
            elif fetch == "all":
                if cursor.description:
                    cursor.rowfactory = _make_dict_factory(cursor)
                return cursor.fetchall()
            return cursor.rowcount
        finally:
            cursor.close()

    def _insert_returning_id(self, sql: str, params: tuple | list) -> int:
        """INSERT 실행 후 생성된 ID 반환 (RETURNING 절 사용)

        Args:
            sql: INSERT SQL 문자열 (? 플레이스홀더 사용 가능)
            params: 바인드 파라미터

        Returns:
            생성된 레코드의 ID
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        try:
            oracle_sql = _convert_placeholders(sql)
            # RETURNING id INTO :out_id 추가
            oracle_sql = oracle_sql.rstrip().rstrip(";")
            id_var = cursor.var(oracledb.NUMBER)
            oracle_sql += " RETURNING id INTO :out_id"
            cursor.execute(oracle_sql, list(params) + [id_var])
            return int(id_var.getvalue()[0])
        finally:
            cursor.close()

    @contextmanager
    def transaction(self) -> Generator[oracledb.Connection, None, None]:
        """트랜잭션 컨텍스트 매니저

        자동 커밋/롤백을 지원하는 트랜잭션 컨텍스트를 제공합니다.

        작동 방식:
        1. with 블록 진입 시 연결 객체 반환
        2. 정상 종료 시 자동 커밋
        3. 예외 발생 시 자동 롤백 후 예외 재발생

        Yields:
            활성 데이터베이스 연결

        Raises:
            DatabaseIntegrityError: 무결성 제약 위반 시
            DatabaseError: 기타 트랜잭션 실패 시

        Example:
            >>> with db.transaction() as conn:
            ...     cursor = conn.cursor()
            ...     cursor.execute("INSERT INTO ...")
            # 자동 커밋 (또는 예외 시 롤백)
        """
        conn = self._get_connection()
        with self._lock:
            try:
                yield conn
                conn.commit()
            except oracledb.IntegrityError as e:
                conn.rollback()
                raise DatabaseIntegrityError(f"무결성 제약 위반: {e}") from e
            except oracledb.Error as e:
                conn.rollback()
                raise DatabaseError(f"트랜잭션 실패: {e}") from e

    # ========================================
    # 세션 CRUD 작업
    # ========================================

    def create_session(self, data: SessionCreate) -> Session:
        """새 분석 세션 생성

        Args:
            data: 세션 생성 데이터 (이름, 소스 경로, 메타데이터)

        Returns:
            생성된 세션 객체 (ID, 타임스탬프 포함)

        Raises:
            DatabaseIntegrityError: 동일 이름의 세션이 이미 존재하는 경우
        """
        with self.transaction():
            new_id = self._insert_returning_id(
                """
                INSERT INTO sessions (name, source_path, metadata)
                VALUES (?, ?, ?)
                """,
                (data.name, data.source_path, data.metadata),
            )
            return self.get_session_by_id(new_id)  # type: ignore

    def get_session_by_id(self, session_id: int) -> Optional[Session]:
        """ID로 세션 조회

        Args:
            session_id: 세션 식별자

        Returns:
            세션 객체 또는 None (존재하지 않는 경우)
        """
        with self._lock:
            row = self._execute(
                "SELECT * FROM sessions WHERE id = ?", (session_id,), fetch="one"
            )

        if row is None:
            return None

        return Session(
            id=row["id"],
            name=row["name"],
            source_path=row["source_path"],
            created_at=_parse_datetime(row["created_at"]),
            updated_at=_parse_datetime(row["updated_at"]),
            is_active=bool(row["is_active"]),
            metadata=row["metadata"],
        )

    def get_active_sessions(self) -> list[Session]:
        """모든 활성 세션 조회

        is_active=1인 세션만 반환합니다.
        생성일시 기준 내림차순으로 정렬됩니다.

        Returns:
            활성 세션 목록
        """
        with self._lock:
            rows = self._execute(
                "SELECT * FROM sessions WHERE is_active = 1 ORDER BY created_at DESC",
                fetch="all",
            )

        return [
            Session(
                id=row["id"],
                name=row["name"],
                source_path=row["source_path"],
                created_at=_parse_datetime(row["created_at"]),
                updated_at=_parse_datetime(row["updated_at"]),
                is_active=bool(row["is_active"]),
                metadata=row["metadata"],
            )
            for row in rows
        ]

    def get_all_sessions(self) -> list[Session]:
        """모든 세션 조회 (활성/비활성 포함)

        생성일시 기준 내림차순으로 정렬됩니다.

        Returns:
            전체 세션 목록
        """
        with self._lock:
            rows = self._execute(
                "SELECT * FROM sessions ORDER BY created_at DESC",
                fetch="all",
            )

        return [
            Session(
                id=row["id"],
                name=row["name"],
                source_path=row["source_path"],
                created_at=_parse_datetime(row["created_at"]),
                updated_at=_parse_datetime(row["updated_at"]),
                is_active=bool(row["is_active"]),
                metadata=row["metadata"],
            )
            for row in rows
        ]

    def get_latest_session_by_source_path(
        self, source_path: str
    ) -> Optional[Session]:
        """source_path로 가장 최신 비활성 세션 조회

        Args:
            source_path: 비디오 파일 절대 경로

        Returns:
            Session 또는 None
        """
        with self._lock:
            row = self._execute(
                """
                SELECT * FROM sessions
                WHERE source_path = ?
                  AND is_active = 0
                ORDER BY created_at DESC
                FETCH FIRST 1 ROWS ONLY
                """,
                (source_path,),
                fetch="one",
            )
        if row is None:
            return None
        return Session(
            id=row["id"],
            name=row["name"],
            source_path=row["source_path"],
            created_at=_parse_datetime(row["created_at"]),
            updated_at=_parse_datetime(row["updated_at"]),
            is_active=bool(row["is_active"]),
            metadata=row["metadata"],
        )

    def update_session(
        self,
        session_id: int,
        *,
        name: Optional[str] = None,
        source_path: Optional[str] = None,
        is_active: Optional[bool] = None,
        metadata: Optional[str] = None,
    ) -> Optional[Session]:
        """세션 필드 업데이트

        지정된 필드만 업데이트합니다. None인 필드는 변경되지 않습니다.
        업데이트 시 updated_at이 자동으로 현재 시각으로 설정됩니다.

        Args:
            session_id: 세션 식별자
            name: 새 세션 이름 (선택)
            source_path: 새 소스 경로 (선택)
            is_active: 새 활성 상태 (선택)
            metadata: 새 메타데이터 (선택)

        Returns:
            업데이트된 세션 또는 None (존재하지 않는 경우)
        """
        # 업데이트할 필드와 값 수집
        updates: list[str] = []
        values: list[Any] = []

        if name is not None:
            updates.append("name = ?")
            values.append(name)
        if source_path is not None:
            updates.append("source_path = ?")
            values.append(source_path)
        if is_active is not None:
            updates.append("is_active = ?")
            values.append(int(is_active))
        if metadata is not None:
            updates.append("metadata = ?")
            values.append(metadata)

        # 업데이트할 필드가 없으면 현재 상태 반환
        if not updates:
            return self.get_session_by_id(session_id)

        # updated_at 자동 갱신
        updates.append("updated_at = SYSTIMESTAMP")
        values.append(session_id)

        with self.transaction():
            self._execute(
                f"UPDATE sessions SET {', '.join(updates)} WHERE id = ?",
                values,
            )

        return self.get_session_by_id(session_id)

    def delete_session(self, session_id: int) -> bool:
        """세션 삭제 (연관 데이터 연쇄 삭제)

        세션 삭제 시 관련된 모든 ROI 정의와 변화 이벤트도
        CASCADE로 함께 삭제됩니다.

        Args:
            session_id: 세션 식별자

        Returns:
            True: 삭제됨, False: 존재하지 않음
        """
        with self.transaction():
            rowcount = self._execute(
                "DELETE FROM sessions WHERE id = ?", (session_id,)
            )
            return rowcount > 0

    # ========================================
    # ROI 정의 CRUD 작업
    # ========================================

    def create_roi(self, data: ROICreate) -> ROIDefinition:
        """새 ROI 정의 생성

        Args:
            data: ROI 생성 데이터

        Returns:
            생성된 ROI 정의 객체

        Raises:
            DatabaseIntegrityError: 세션 내 동일 이름의 ROI가 존재하거나
                                   session_id가 유효하지 않은 경우
        """
        with self.transaction():
            new_id = self._insert_returning_id(
                """
                INSERT INTO roi_definitions
                (session_id, name, roi_type, x, y, width, height, threshold, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    data.session_id,
                    data.name,
                    data.roi_type.value,
                    data.x,
                    data.y,
                    data.width,
                    data.height,
                    data.threshold,
                    data.metadata,
                ),
            )
            return self.get_roi_by_id(new_id)  # type: ignore

    def get_roi_by_id(self, roi_id: int) -> Optional[ROIDefinition]:
        """ID로 ROI 정의 조회

        Args:
            roi_id: ROI 식별자

        Returns:
            ROI 정의 또는 None (존재하지 않는 경우)
        """
        with self._lock:
            row = self._execute(
                "SELECT * FROM roi_definitions WHERE id = ?", (roi_id,), fetch="one"
            )

        if row is None:
            return None

        return ROIDefinition(
            id=row["id"],
            session_id=row["session_id"],
            name=row["name"],
            roi_type=ROIType(row["roi_type"]),
            x=int(row["x"]) if not isinstance(row["x"], int) else row["x"],
            y=int(row["y"]) if not isinstance(row["y"], int) else row["y"],
            width=int(row["width"]) if not isinstance(row["width"], int) else row["width"],
            height=int(row["height"]) if not isinstance(row["height"], int) else row["height"],
            threshold=row["threshold"],
            created_at=_parse_datetime(row["created_at"]),
            metadata=row["metadata"],
        )

    def get_rois_by_session(self, session_id: int) -> list[ROIDefinition]:
        """세션의 모든 ROI 정의 조회

        이름 기준 오름차순으로 정렬됩니다.

        Args:
            session_id: 세션 식별자

        Returns:
            ROI 정의 목록
        """
        with self._lock:
            rows = self._execute(
                "SELECT * FROM roi_definitions WHERE session_id = ? ORDER BY name",
                (session_id,),
                fetch="all",
            )

        return [
            ROIDefinition(
                id=row["id"],
                session_id=row["session_id"],
                name=row["name"],
                roi_type=ROIType(row["roi_type"]),
                x=int(row["x"]) if not isinstance(row["x"], int) else row["x"],
                y=int(row["y"]) if not isinstance(row["y"], int) else row["y"],
                width=int(row["width"]) if not isinstance(row["width"], int) else row["width"],
                height=int(row["height"]) if not isinstance(row["height"], int) else row["height"],
                threshold=row["threshold"],
                created_at=_parse_datetime(row["created_at"]),
                metadata=row["metadata"],
            )
            for row in rows
        ]

    def update_roi(
        self,
        roi_id: int,
        *,
        name: Optional[str] = None,
        roi_type: Optional[ROIType] = None,
        x: Optional[int] = None,
        y: Optional[int] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
        threshold: Optional[float] = None,
        metadata: Optional[str] = None,
    ) -> Optional[ROIDefinition]:
        """ROI 정의 필드 업데이트

        지정된 필드만 업데이트합니다. None인 필드는 변경되지 않습니다.

        Args:
            roi_id: ROI 식별자
            name: 새 ROI 이름 (선택)
            roi_type: 새 ROI 타입 (선택)
            x: 새 X 좌표 (선택)
            y: 새 Y 좌표 (선택)
            width: 새 너비 (선택)
            height: 새 높이 (선택)
            threshold: 새 임계값 (선택)
            metadata: 새 메타데이터 (선택)

        Returns:
            업데이트된 ROI 정의 또는 None (존재하지 않는 경우)
        """
        # 업데이트할 필드와 값 수집
        updates: list[str] = []
        values: list[Any] = []

        if name is not None:
            updates.append("name = ?")
            values.append(name)
        if roi_type is not None:
            updates.append("roi_type = ?")
            values.append(roi_type.value)
        if x is not None:
            updates.append("x = ?")
            values.append(x)
        if y is not None:
            updates.append("y = ?")
            values.append(y)
        if width is not None:
            updates.append("width = ?")
            values.append(width)
        if height is not None:
            updates.append("height = ?")
            values.append(height)
        if threshold is not None:
            updates.append("threshold = ?")
            values.append(threshold)
        if metadata is not None:
            updates.append("metadata = ?")
            values.append(metadata)

        # 업데이트할 필드가 없으면 현재 상태 반환
        if not updates:
            return self.get_roi_by_id(roi_id)

        values.append(roi_id)

        with self.transaction():
            self._execute(
                f"UPDATE roi_definitions SET {', '.join(updates)} WHERE id = ?",
                values,
            )

        return self.get_roi_by_id(roi_id)

    def delete_roi(self, roi_id: int) -> bool:
        """ROI 정의 삭제 (연관 이벤트 연쇄 삭제)

        ROI 삭제 시 관련된 모든 변화 이벤트도
        CASCADE로 함께 삭제됩니다.

        Args:
            roi_id: ROI 식별자

        Returns:
            True: 삭제됨, False: 존재하지 않음
        """
        with self.transaction():
            rowcount = self._execute(
                "DELETE FROM roi_definitions WHERE id = ?", (roi_id,)
            )
            return rowcount > 0

    # ========================================
    # 변화 이벤트 CRUD 작업
    # ========================================

    def create_change_event(self, data: ChangeEventCreate) -> ChangeEvent:
        """새 변화 이벤트 생성

        Args:
            data: 변화 이벤트 생성 데이터

        Returns:
            생성된 변화 이벤트 객체

        Raises:
            DatabaseIntegrityError: roi_id 또는 session_id가 유효하지 않은 경우
        """
        with self.transaction():
            new_id = self._insert_returning_id(
                """
                INSERT INTO change_events
                (roi_id, session_id, capture_id, previous_value, current_value,
                 frame_path, extracted_text, is_chart, confidence, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    data.roi_id,
                    data.session_id,
                    data.capture_id,
                    data.previous_value,
                    data.current_value,
                    data.frame_path,
                    data.extracted_text,
                    int(data.is_chart),
                    data.confidence,
                    data.metadata,
                ),
            )
            return self.get_event_by_id(new_id)  # type: ignore

    def get_event_by_id(self, event_id: int) -> Optional[ChangeEvent]:
        """ID로 변화 이벤트 조회

        Args:
            event_id: 이벤트 식별자

        Returns:
            변화 이벤트 또는 None (존재하지 않는 경우)
        """
        with self._lock:
            row = self._execute(
                "SELECT * FROM change_events WHERE id = ?", (event_id,), fetch="one"
            )

        if row is None:
            return None

        return ChangeEvent(
            id=row["id"],
            roi_id=row["roi_id"],
            session_id=row["session_id"],
            timestamp=_parse_datetime(row["timestamp"]),
            previous_value=row["previous_value"],
            current_value=row["current_value"],
            frame_path=row["frame_path"],
            extracted_text=row["extracted_text"],
            is_chart=bool(row["is_chart"]),
            confidence=row["confidence"],
            metadata=row["metadata"],
            capture_id=row["capture_id"],
        )

    def get_events_by_session(
        self,
        session_id: int,
        *,
        limit: Optional[int] = None,
        offset: int = 0,
    ) -> list[ChangeEvent]:
        """세션의 변화 이벤트 조회

        타임스탬프 기준 내림차순(최신순)으로 정렬됩니다.
        페이지네이션을 위한 limit/offset을 지원합니다.

        Args:
            session_id: 세션 식별자
            limit: 최대 반환 개수 (None이면 전체)
            offset: 건너뛸 개수 (페이지네이션용)

        Returns:
            변화 이벤트 목록
        """
        query = """
            SELECT * FROM change_events
            WHERE session_id = ?
            ORDER BY timestamp DESC
        """
        params: list[Any] = [session_id]

        # 페이지네이션 적용 (Oracle 12c+ OFFSET/FETCH 문법)
        if limit is not None:
            query += " OFFSET ? ROWS FETCH NEXT ? ROWS ONLY"
            params.extend([offset, limit])

        with self._lock:
            rows = self._execute(query, params, fetch="all")

        return [
            ChangeEvent(
                id=row["id"],
                roi_id=row["roi_id"],
                session_id=row["session_id"],
                timestamp=_parse_datetime(row["timestamp"]),
                previous_value=row["previous_value"],
                current_value=row["current_value"],
                frame_path=row["frame_path"],
                extracted_text=row["extracted_text"],
                is_chart=bool(row["is_chart"]),
                confidence=row["confidence"],
                metadata=row["metadata"],
                capture_id=row["capture_id"],
            )
            for row in rows
        ]

    def get_events_by_roi(
        self,
        roi_id: int,
        *,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: Optional[int] = None,
    ) -> list[ChangeEvent]:
        """특정 ROI의 변화 이벤트 조회

        타임스탬프 기준 내림차순(최신순)으로 정렬됩니다.
        시간 범위 필터링을 지원합니다.

        Args:
            roi_id: ROI 식별자
            start_time: 시작 시각 필터 (이 시각 이후 이벤트)
            end_time: 종료 시각 필터 (이 시각 이전 이벤트)
            limit: 최대 반환 개수

        Returns:
            변화 이벤트 목록
        """
        conditions = ["roi_id = ?"]
        params: list[Any] = [roi_id]

        # 시간 범위 필터 적용
        if start_time is not None:
            conditions.append("timestamp >= ?")
            params.append(start_time.isoformat())
        if end_time is not None:
            conditions.append("timestamp <= ?")
            params.append(end_time.isoformat())

        query = f"""
            SELECT * FROM change_events
            WHERE {' AND '.join(conditions)}
            ORDER BY timestamp DESC
        """

        if limit is not None:
            query += " FETCH FIRST ? ROWS ONLY"
            params.append(limit)

        with self._lock:
            rows = self._execute(query, params, fetch="all")

        return [
            ChangeEvent(
                id=row["id"],
                roi_id=row["roi_id"],
                session_id=row["session_id"],
                timestamp=_parse_datetime(row["timestamp"]),
                previous_value=row["previous_value"],
                current_value=row["current_value"],
                frame_path=row["frame_path"],
                extracted_text=row["extracted_text"],
                is_chart=bool(row["is_chart"]),
                confidence=row["confidence"],
                metadata=row["metadata"],
                capture_id=row["capture_id"],
            )
            for row in rows
        ]

    def get_latest_event_for_roi(self, roi_id: int) -> Optional[ChangeEvent]:
        """ROI의 가장 최근 변화 이벤트 조회

        Args:
            roi_id: ROI 식별자

        Returns:
            최신 변화 이벤트 또는 None (이벤트가 없는 경우)
        """
        events = self.get_events_by_roi(roi_id, limit=1)
        return events[0] if events else None

    def delete_event(self, event_id: int) -> bool:
        """변화 이벤트 삭제

        Args:
            event_id: 이벤트 식별자

        Returns:
            True: 삭제됨, False: 존재하지 않음
        """
        with self.transaction():
            rowcount = self._execute(
                "DELETE FROM change_events WHERE id = ?", (event_id,)
            )
            return rowcount > 0

    def delete_events_before(self, before: datetime, session_id: Optional[int] = None) -> int:
        """특정 시각 이전의 변화 이벤트 일괄 삭제

        오래된 이벤트를 정리하여 데이터베이스 크기를 관리합니다.

        Args:
            before: 이 시각 이전의 이벤트 삭제
            session_id: 특정 세션으로 제한 (None이면 전체)

        Returns:
            삭제된 이벤트 수
        """
        with self.transaction():
            if session_id is not None:
                rowcount = self._execute(
                    "DELETE FROM change_events WHERE timestamp < ? AND session_id = ?",
                    (before.isoformat(), session_id),
                )
            else:
                rowcount = self._execute(
                    "DELETE FROM change_events WHERE timestamp < ?",
                    (before.isoformat(),),
                )
            return rowcount

    # ========================================
    # 배치 작업
    # ========================================

    def create_change_events_batch(
        self, events: Sequence[ChangeEventCreate]
    ) -> list[ChangeEvent]:
        """여러 변화 이벤트 일괄 생성

        단일 트랜잭션에서 여러 이벤트를 생성합니다.
        모두 성공하거나 모두 롤백됩니다.

        Args:
            events: 생성할 이벤트 데이터 시퀀스

        Returns:
            생성된 변화 이벤트 목록

        Raises:
            DatabaseIntegrityError: 무결성 제약 위반 시 전체 롤백
        """
        if not events:
            return []

        created_ids: list[int] = []
        with self.transaction():
            for e in events:
                new_id = self._insert_returning_id(
                    """
                    INSERT INTO change_events
                    (roi_id, session_id, capture_id, previous_value, current_value,
                     frame_path, extracted_text, is_chart, confidence, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        e.roi_id,
                        e.session_id,
                        e.capture_id,
                        e.previous_value,
                        e.current_value,
                        e.frame_path,
                        e.extracted_text,
                        int(e.is_chart),
                        e.confidence,
                        e.metadata,
                    ),
                )
                created_ids.append(new_id)

        # 생성된 이벤트 객체 조회하여 반환
        return [
            event
            for event_id in created_ids
            if (event := self.get_event_by_id(event_id)) is not None
        ]

    # ========================================
    # 통계 및 집계
    # ========================================

    def get_event_count_by_session(self, session_id: int) -> int:
        """세션의 총 이벤트 수 조회

        Args:
            session_id: 세션 식별자

        Returns:
            이벤트 수
        """
        with self._lock:
            row = self._execute(
                "SELECT COUNT(*) as count FROM change_events WHERE session_id = ?",
                (session_id,),
                fetch="one",
            )
        return row["count"] if row else 0

    def get_event_count_by_roi(self, roi_id: int) -> int:
        """ROI의 총 이벤트 수 조회

        Args:
            roi_id: ROI 식별자

        Returns:
            이벤트 수
        """
        with self._lock:
            row = self._execute(
                "SELECT COUNT(*) as count FROM change_events WHERE roi_id = ?",
                (roi_id,),
                fetch="one",
            )
        return row["count"] if row else 0

    def vacuum(self) -> None:
        """데이터베이스 최적화

        Oracle은 공간 관리가 자동으로 이루어지므로 no-op입니다.
        """
        pass

    # ========================================
    # Capture CRUD 메서드
    # ========================================

    def create_capture(self, data: CaptureCreate) -> Capture:
        """새 캡쳐 그룹 생성

        트리거 ROI 변경 시점에 전체 프레임 캡쳐 정보를 기록합니다.

        Args:
            data: 캡쳐 생성 데이터

        Returns:
            생성된 Capture 객체

        Raises:
            DatabaseIntegrityError: session_id 또는 trigger_roi_id가 유효하지 않은 경우
        """
        with self.transaction():
            new_id = self._insert_returning_id(
                """
                INSERT INTO captures
                (session_id, trigger_roi_id, frame_path, frame_number, timestamp_ms, metadata)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    data.session_id,
                    data.trigger_roi_id,
                    data.frame_path,
                    data.frame_number,
                    data.timestamp_ms,
                    data.metadata,
                ),
            )
            return self.get_capture_by_id(new_id)  # type: ignore

    def get_capture_by_id(self, capture_id: int) -> Optional[Capture]:
        """ID로 캡쳐 조회

        Args:
            capture_id: 캡쳐 식별자

        Returns:
            Capture 또는 None (존재하지 않는 경우)
        """
        with self._lock:
            row = self._execute(
                "SELECT * FROM captures WHERE id = ?", (capture_id,), fetch="one"
            )

        if row is None:
            return None

        return Capture(
            id=row["id"],
            session_id=row["session_id"],
            trigger_roi_id=row["trigger_roi_id"],
            frame_path=row["frame_path"],
            frame_number=row["frame_number"],
            timestamp_ms=row["timestamp_ms"],
            captured_at=_parse_datetime(row["captured_at"]),
            metadata=row["metadata"],
        )

    def get_captures_by_session(
        self,
        session_id: int,
        *,
        limit: Optional[int] = None,
        offset: int = 0,
    ) -> list[Capture]:
        """세션의 모든 캡쳐 조회

        캡쳐 시각 기준 내림차순(최신순)으로 정렬됩니다.

        Args:
            session_id: 세션 식별자
            limit: 최대 반환 개수 (None이면 전체)
            offset: 건너뛸 개수 (페이지네이션용)

        Returns:
            Capture 목록
        """
        query = """
            SELECT * FROM captures
            WHERE session_id = ?
            ORDER BY captured_at DESC
        """
        params: list[Any] = [session_id]

        if limit is not None:
            query += " OFFSET ? ROWS FETCH NEXT ? ROWS ONLY"
            params.extend([offset, limit])

        with self._lock:
            rows = self._execute(query, params, fetch="all")

        return [
            Capture(
                id=row["id"],
                session_id=row["session_id"],
                trigger_roi_id=row["trigger_roi_id"],
                frame_path=row["frame_path"],
                frame_number=row["frame_number"],
                timestamp_ms=row["timestamp_ms"],
                captured_at=_parse_datetime(row["captured_at"]),
                metadata=row["metadata"],
            )
            for row in rows
        ]

    def get_events_by_capture(self, capture_id: int) -> list[ChangeEvent]:
        """캡쳐 그룹에 속한 모든 변화 이벤트 조회

        같은 capture_id를 가진 모든 change_events를 반환합니다.
        이를 통해 특정 캡쳐 시점의 모든 ROI 값을 조회할 수 있습니다.

        Args:
            capture_id: 캡쳐 식별자

        Returns:
            해당 캡쳐에 연결된 ChangeEvent 목록
        """
        with self._lock:
            rows = self._execute(
                """
                SELECT * FROM change_events
                WHERE capture_id = ?
                ORDER BY roi_id
                """,
                (capture_id,),
                fetch="all",
            )

        return [
            ChangeEvent(
                id=row["id"],
                roi_id=row["roi_id"],
                session_id=row["session_id"],
                timestamp=_parse_datetime(row["timestamp"]),
                previous_value=row["previous_value"],
                current_value=row["current_value"],
                frame_path=row["frame_path"],
                extracted_text=row["extracted_text"],
                is_chart=bool(row["is_chart"]),
                confidence=row["confidence"],
                metadata=row["metadata"],
                capture_id=row["capture_id"],
            )
            for row in rows
        ]

    def delete_capture(self, capture_id: int) -> bool:
        """캡쳐 삭제

        캡쳐 삭제 시 연결된 change_events의 capture_id는
        SET NULL로 처리됩니다 (이벤트 자체는 유지).

        Args:
            capture_id: 캡쳐 식별자

        Returns:
            True: 삭제됨, False: 존재하지 않음
        """
        with self.transaction():
            rowcount = self._execute(
                "DELETE FROM captures WHERE id = ?", (capture_id,)
            )
            return rowcount > 0

    def get_capture_count_by_session(self, session_id: int) -> int:
        """세션의 총 캡쳐 수 조회

        Args:
            session_id: 세션 식별자

        Returns:
            캡쳐 수
        """
        with self._lock:
            row = self._execute(
                "SELECT COUNT(*) as count FROM captures WHERE session_id = ?",
                (session_id,),
                fetch="one",
            )
        return row["count"] if row else 0

    # ========================================
    # video_files CRUD 메서드
    # ========================================

    def _row_to_video_file(self, row: dict) -> VideoFile:
        """dict row → VideoFile 엔티티 변환 (내부 헬퍼)

        started_at, completed_at, session_id 등 nullable 필드를 안전하게 처리합니다.
        """
        return VideoFile(
            id=row["id"],
            file_path=row["file_path"],
            file_name=row["file_name"],
            file_size=row["file_size"],
            discovered_at=_parse_datetime(row["discovered_at"]),
            status=VideoFileStatus(row["status"]),
            session_id=row["session_id"],
            started_at=_parse_datetime(row["started_at"]) if row["started_at"] else None,
            completed_at=_parse_datetime(row["completed_at"]) if row["completed_at"] else None,
            error_message=row["error_message"],
            metadata=row["metadata"],
        )

    def create_video_file(self, data: VideoFileCreate) -> VideoFile:
        """새 비디오 파일 레코드 등록

        file_path에 UNIQUE 제약이 있으므로 이미 등록된 경로이면
        DatabaseIntegrityError가 발생합니다.

        Args:
            data: VideoFileCreate DTO (file_path, file_name, file_size, metadata)

        Returns:
            생성된 VideoFile 엔티티 (status=PENDING, discovered_at=now())

        Raises:
            DatabaseIntegrityError: 동일 file_path가 이미 존재하는 경우
        """
        with self.transaction():
            new_id = self._insert_returning_id(
                """
                INSERT INTO video_files (file_path, file_name, file_size, metadata)
                VALUES (?, ?, ?, ?)
                """,
                (data.file_path, data.file_name, data.file_size, data.metadata),
            )
        return self.get_video_file_by_id(new_id)  # type: ignore

    def get_video_file_by_id(self, video_file_id: int) -> Optional[VideoFile]:
        """ID로 비디오 파일 레코드 조회

        Args:
            video_file_id: video_files.id

        Returns:
            VideoFile 엔티티 또는 None
        """
        with self._lock:
            row = self._execute(
                "SELECT * FROM video_files WHERE id = ?", (video_file_id,), fetch="one"
            )
        return self._row_to_video_file(row) if row else None

    def get_video_file_by_path(self, file_path: str) -> Optional[VideoFile]:
        """파일 경로로 비디오 파일 레코드 조회

        배치 스캔 시 중복 확인에 사용됩니다.

        Args:
            file_path: 절대 경로 문자열

        Returns:
            VideoFile 엔티티 또는 None
        """
        with self._lock:
            row = self._execute(
                "SELECT * FROM video_files WHERE file_path = ?", (file_path,), fetch="one"
            )
        return self._row_to_video_file(row) if row else None

    def get_video_files_by_status(
        self,
        status: VideoFileStatus,
        *,
        limit: Optional[int] = None,
    ) -> list[VideoFile]:
        """특정 상태의 비디오 파일 목록 조회

        discovered_at 오름차순(먼저 발견된 파일 우선)으로 반환합니다.

        Args:
            status: 조회할 상태 (PENDING, PROCESSING, COMPLETED, FAILED)
            limit: 최대 반환 개수 (None이면 전체)

        Returns:
            VideoFile 목록 (discovered_at ASC)
        """
        query = "SELECT * FROM video_files WHERE status = ? ORDER BY discovered_at ASC"
        params: list[Any] = [status.value]
        if limit is not None:
            query += " FETCH FIRST ? ROWS ONLY"
            params.append(limit)
        with self._lock:
            rows = self._execute(query, params, fetch="all")
        return [self._row_to_video_file(r) for r in rows]

    def update_video_file_status(
        self,
        video_file_id: int,
        status: VideoFileStatus,
        *,
        session_id: Optional[int] = None,
        started_at: Optional[datetime] = None,
        completed_at: Optional[datetime] = None,
        error_message: Optional[str] = None,
    ) -> Optional[VideoFile]:
        """비디오 파일 상태 업데이트

        상태 전환 규칙:
            PENDING -> PROCESSING  : started_at 설정
            PROCESSING -> COMPLETED: session_id, completed_at 설정
            PROCESSING -> FAILED   : error_message, completed_at 설정

        Args:
            video_file_id: 업데이트할 레코드 ID
            status: 새 상태
            session_id: 연결된 분석 세션 ID (COMPLETED 시)
            started_at: 분석 시작 시각 (PROCESSING 시)
            completed_at: 분석 완료 시각 (COMPLETED/FAILED 시)
            error_message: 오류 메시지 (FAILED 시)

        Returns:
            업데이트된 VideoFile 또는 None (ID 없음)
        """
        updates: list[str] = ["status = ?"]
        values: list[Any] = [status.value]

        if session_id is not None:
            updates.append("session_id = ?")
            values.append(session_id)
        if started_at is not None:
            updates.append("started_at = ?")
            values.append(started_at.strftime("%Y-%m-%d %H:%M:%S"))
        if completed_at is not None:
            updates.append("completed_at = ?")
            values.append(completed_at.strftime("%Y-%m-%d %H:%M:%S"))
        if error_message is not None:
            updates.append("error_message = ?")
            values.append(error_message)

        values.append(video_file_id)
        with self.transaction():
            self._execute(
                f"UPDATE video_files SET {', '.join(updates)} WHERE id = ?",
                values,
            )
        return self.get_video_file_by_id(video_file_id)

    def get_all_video_files(
        self,
        *,
        limit: Optional[int] = None,
        offset: int = 0,
    ) -> list[VideoFile]:
        """전체 비디오 파일 목록 조회 (관리/모니터링용)

        discovered_at 내림차순으로 반환합니다.

        Args:
            limit: 최대 반환 개수
            offset: 페이지네이션 오프셋

        Returns:
            VideoFile 목록
        """
        query = "SELECT * FROM video_files ORDER BY discovered_at DESC"
        params: list[Any] = []
        if limit is not None:
            query += " OFFSET ? ROWS FETCH NEXT ? ROWS ONLY"
            params.extend([offset, limit])
        with self._lock:
            rows = self._execute(query, params, fetch="all")
        return [self._row_to_video_file(r) for r in rows]

    # ========================================
    # ROI 템플릿 CRUD 메서드
    # ========================================

    def create_template(self, data: ROITemplateCreate) -> ROITemplate:
        """새 ROI 템플릿 생성

        Args:
            data: 템플릿 생성 데이터

        Returns:
            생성된 ROITemplate 객체

        Raises:
            DatabaseIntegrityError: 동일 이름의 템플릿이 이미 존재하는 경우
        """
        try:
            with self.transaction():
                new_id = self._insert_returning_id(
                    """
                    INSERT INTO roi_templates (name, description)
                    VALUES (?, ?)
                    """,
                    (data.name, data.description),
                )
        except oracledb.IntegrityError as e:
            raise DatabaseIntegrityError(f"템플릿 생성 실패: {e}") from e

        return self.get_template_by_id(new_id)  # type: ignore

    def get_template_by_id(self, template_id: int) -> Optional[ROITemplate]:
        """ID로 템플릿 조회

        Args:
            template_id: 템플릿 ID

        Returns:
            ROITemplate 또는 None
        """
        row = self._execute(
            "SELECT * FROM roi_templates WHERE id = ?",
            (template_id,),
            fetch="one",
        )

        if row is None:
            return None

        return ROITemplate(
            id=row["id"],
            name=row["name"],
            description=row["description"],
            created_at=_parse_datetime(row["created_at"]),
            updated_at=_parse_datetime(row["updated_at"]),
        )

    def get_template_by_name(self, name: str) -> Optional[ROITemplate]:
        """이름으로 템플릿 조회

        Args:
            name: 템플릿 이름

        Returns:
            ROITemplate 또는 None
        """
        row = self._execute(
            "SELECT * FROM roi_templates WHERE name = ?",
            (name,),
            fetch="one",
        )

        if row is None:
            return None

        return ROITemplate(
            id=row["id"],
            name=row["name"],
            description=row["description"],
            created_at=_parse_datetime(row["created_at"]),
            updated_at=_parse_datetime(row["updated_at"]),
        )

    def get_all_templates(self) -> list[ROITemplate]:
        """모든 템플릿 조회

        Returns:
            ROITemplate 리스트 (생성일 내림차순)
        """
        rows = self._execute(
            "SELECT * FROM roi_templates ORDER BY created_at DESC",
            fetch="all",
        )

        return [
            ROITemplate(
                id=row["id"],
                name=row["name"],
                description=row["description"],
                created_at=_parse_datetime(row["created_at"]),
                updated_at=_parse_datetime(row["updated_at"]),
            )
            for row in rows
        ]

    def update_template(
        self,
        template_id: int,
        name: Optional[str] = None,
        description: Optional[str] = None,
    ) -> Optional[ROITemplate]:
        """템플릿 정보 수정

        Args:
            template_id: 수정할 템플릿 ID
            name: 새 이름 (None이면 변경 안 함)
            description: 새 설명 (None이면 변경 안 함)

        Returns:
            수정된 ROITemplate 또는 None (존재하지 않는 경우)
        """
        updates = []
        values = []

        if name is not None:
            updates.append("name = ?")
            values.append(name)
        if description is not None:
            updates.append("description = ?")
            values.append(description)

        if not updates:
            return self.get_template_by_id(template_id)

        updates.append("updated_at = SYSTIMESTAMP")
        values.append(template_id)

        with self.transaction():
            rowcount = self._execute(
                f"UPDATE roi_templates SET {', '.join(updates)} WHERE id = ?",
                values,
            )

            if rowcount == 0:
                return None

        return self.get_template_by_id(template_id)

    def delete_template(self, template_id: int) -> bool:
        """템플릿 삭제 (CASCADE로 소속 ROI도 삭제됨)

        Args:
            template_id: 삭제할 템플릿 ID

        Returns:
            삭제 성공 여부
        """
        with self.transaction():
            rowcount = self._execute(
                "DELETE FROM roi_templates WHERE id = ?",
                (template_id,),
            )
            return rowcount > 0

    # ========================================
    # 템플릿 ROI CRUD 메서드
    # ========================================

    def create_template_roi(self, data: TemplateROICreate) -> TemplateROI:
        """템플릿에 ROI 추가

        Args:
            data: 템플릿 ROI 생성 데이터

        Returns:
            생성된 TemplateROI 객체

        Raises:
            DatabaseIntegrityError: 동일 이름의 ROI가 이미 존재하거나
                                   템플릿이 존재하지 않는 경우
        """
        try:
            with self.transaction():
                new_id = self._insert_returning_id(
                    """
                    INSERT INTO template_rois
                    (template_id, name, roi_type, x, y, width, height, threshold, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        data.template_id,
                        data.name,
                        data.roi_type.value,
                        data.x,
                        data.y,
                        data.width,
                        data.height,
                        data.threshold,
                        data.metadata,
                    ),
                )
        except oracledb.IntegrityError as e:
            raise DatabaseIntegrityError(f"템플릿 ROI 생성 실패: {e}") from e

        return self.get_template_roi_by_id(new_id)  # type: ignore

    def get_template_roi_by_id(self, roi_id: int) -> Optional[TemplateROI]:
        """ID로 템플릿 ROI 조회

        Args:
            roi_id: ROI ID

        Returns:
            TemplateROI 또는 None
        """
        row = self._execute(
            "SELECT * FROM template_rois WHERE id = ?",
            (roi_id,),
            fetch="one",
        )

        if row is None:
            return None

        return TemplateROI(
            id=row["id"],
            template_id=row["template_id"],
            name=row["name"],
            roi_type=ROIType(row["roi_type"]),
            x=row["x"],
            y=row["y"],
            width=row["width"],
            height=row["height"],
            threshold=row["threshold"],
            created_at=_parse_datetime(row["created_at"]),
            metadata=row["metadata"],
        )

    def get_rois_by_template(self, template_id: int) -> list[TemplateROI]:
        """템플릿에 속한 모든 ROI 조회

        Args:
            template_id: 템플릿 ID

        Returns:
            TemplateROI 리스트
        """
        rows = self._execute(
            "SELECT * FROM template_rois WHERE template_id = ? ORDER BY id",
            (template_id,),
            fetch="all",
        )

        return [
            TemplateROI(
                id=row["id"],
                template_id=row["template_id"],
                name=row["name"],
                roi_type=ROIType(row["roi_type"]),
                x=row["x"],
                y=row["y"],
                width=row["width"],
                height=row["height"],
                threshold=row["threshold"],
                created_at=_parse_datetime(row["created_at"]),
                metadata=row["metadata"],
            )
            for row in rows
        ]

    def update_template_roi(
        self,
        roi_id: int,
        *,
        name: Optional[str] = None,
        roi_type: Optional[ROIType] = None,
        x: Optional[int] = None,
        y: Optional[int] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
        threshold: Optional[float] = None,
        metadata: Optional[str] = None,
    ) -> Optional[TemplateROI]:
        """템플릿 ROI 수정

        Args:
            roi_id: 수정할 ROI ID
            나머지: 수정할 필드들 (None이면 변경 안 함)

        Returns:
            수정된 TemplateROI 또는 None
        """
        updates = []
        values = []

        if name is not None:
            updates.append("name = ?")
            values.append(name)
        if roi_type is not None:
            updates.append("roi_type = ?")
            values.append(roi_type.value)
        if x is not None:
            updates.append("x = ?")
            values.append(x)
        if y is not None:
            updates.append("y = ?")
            values.append(y)
        if width is not None:
            updates.append("width = ?")
            values.append(width)
        if height is not None:
            updates.append("height = ?")
            values.append(height)
        if threshold is not None:
            updates.append("threshold = ?")
            values.append(threshold)
        if metadata is not None:
            updates.append("metadata = ?")
            values.append(metadata)

        if not updates:
            return self.get_template_roi_by_id(roi_id)

        values.append(roi_id)

        with self.transaction():
            rowcount = self._execute(
                f"UPDATE template_rois SET {', '.join(updates)} WHERE id = ?",
                values,
            )

            if rowcount == 0:
                return None

        return self.get_template_roi_by_id(roi_id)

    def delete_template_roi(self, roi_id: int) -> bool:
        """템플릿 ROI 삭제

        Args:
            roi_id: 삭제할 ROI ID

        Returns:
            삭제 성공 여부
        """
        with self.transaction():
            rowcount = self._execute(
                "DELETE FROM template_rois WHERE id = ?",
                (roi_id,),
            )
            return rowcount > 0

    def create_template_rois_batch(
        self,
        rois: Sequence[TemplateROICreate],
    ) -> list[TemplateROI]:
        """여러 템플릿 ROI 일괄 생성

        Args:
            rois: 생성할 ROI 데이터 시퀀스

        Returns:
            생성된 TemplateROI 리스트
        """
        if not rois:
            return []

        created_ids = []

        with self.transaction():
            for roi in rois:
                new_id = self._insert_returning_id(
                    """
                    INSERT INTO template_rois
                    (template_id, name, roi_type, x, y, width, height, threshold, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        roi.template_id,
                        roi.name,
                        roi.roi_type.value,
                        roi.x,
                        roi.y,
                        roi.width,
                        roi.height,
                        roi.threshold,
                        roi.metadata,
                    ),
                )
                created_ids.append(new_id)

        return [
            self.get_template_roi_by_id(roi_id)  # type: ignore
            for roi_id in created_ids
        ]
