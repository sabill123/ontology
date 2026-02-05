"""
Temporal FK Detector (v16.0)

시간 기반 FK 탐지:
- SCD Type 2 (Slowly Changing Dimension)
- 유효 기간 FK (valid_from, valid_to)
- 이력 테이블 연결
- Bitemporal 관계

EnhancedFKDetector와 통합되어 사용됨

사용법:
    from .temporal_fk_detector import detect_temporal_fks_and_update_context
    temporal_fks = detect_temporal_fks_and_update_context(shared_context)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple, TYPE_CHECKING
from enum import Enum
from datetime import datetime
import re
import logging

if TYPE_CHECKING:
    from ...shared_context import SharedContext

# 순환 import 방지를 위한 지연 import
def _get_fk_candidate_class():
    from .enhanced_fk_detector import FKCandidate
    return FKCandidate

logger = logging.getLogger(__name__)


class TemporalType(Enum):
    """Temporal FK 유형"""
    SCD_TYPE_1 = "scd_type_1"      # 덮어쓰기 (이력 없음)
    SCD_TYPE_2 = "scd_type_2"      # 이력 유지 (valid_from/to)
    SCD_TYPE_3 = "scd_type_3"      # 이전 값 컬럼 (previous_*)
    BITEMPORAL = "bitemporal"      # 트랜잭션 시간 + 유효 시간
    SNAPSHOT = "snapshot"          # 시점 스냅샷
    EVENT_TIME = "event_time"      # 이벤트 시간 기반


@dataclass
class TemporalFKCandidate:
    """Temporal FK 후보"""
    from_table: str
    from_column: str
    to_table: str
    to_column: str

    temporal_type: TemporalType

    # Temporal columns
    valid_from_col: Optional[str] = None
    valid_to_col: Optional[str] = None
    transaction_time_col: Optional[str] = None

    # SCD Type 3 컬럼
    previous_value_col: Optional[str] = None

    # Scores
    pattern_score: float = 0.0
    temporal_score: float = 0.0
    value_score: float = 0.0

    confidence: float = 0.0

    def to_fk_candidate(self):
        """FKCandidate로 변환 (하위 호환성)"""
        FKCandidate = _get_fk_candidate_class()
        return FKCandidate(
            from_table=self.from_table,
            from_column=self.from_column,
            to_table=self.to_table,
            to_column=self.to_column,
            fk_score=self.confidence,
            confidence="high" if self.confidence >= 0.8 else "medium",
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "from_table": self.from_table,
            "from_column": self.from_column,
            "to_table": self.to_table,
            "to_column": self.to_column,
            "temporal_type": self.temporal_type.value,
            "valid_from_col": self.valid_from_col,
            "valid_to_col": self.valid_to_col,
            "transaction_time_col": self.transaction_time_col,
            "previous_value_col": self.previous_value_col,
            "pattern_score": round(self.pattern_score, 4),
            "temporal_score": round(self.temporal_score, 4),
            "value_score": round(self.value_score, 4),
            "confidence": round(self.confidence, 4),
        }


class TemporalFKDetector:
    """
    Temporal FK 탐지기

    알고리즘:
    1. Temporal 컬럼 식별 (valid_from, valid_to, effective_date 등)
    2. SCD 패턴 분류 (Type 1/2/3, Bitemporal, Snapshot)
    3. 이력 테이블 ↔ 메인 테이블 연결
    4. 시간 범위 기반 FK 검증
    5. 기존 FK에 Temporal 메타데이터 추가

    SCD Type 분류:
    - Type 1: 덮어쓰기 (이력 없음, updated_at만 존재)
    - Type 2: 이력 유지 (valid_from, valid_to, is_current)
    - Type 3: 이전 값 컬럼 (previous_*, old_*)
    - Bitemporal: 유효 시간 + 트랜잭션 시간 모두 존재
    """

    # Temporal 컬럼 패턴
    VALID_FROM_PATTERNS = [
        r"valid_?from",
        r"effective_?(?:from|date|start)",
        r"start_?date",
        r"begin_?date",
        r"from_?date",
        r"active_?from",
        r"valid_?start",
    ]

    VALID_TO_PATTERNS = [
        r"valid_?to",
        r"effective_?(?:to|end)",
        r"end_?date",
        r"expire[sd]?_?(?:date|at)?",
        r"to_?date",
        r"active_?to",
        r"valid_?end",
        r"terminated_?(?:date|at)?",
    ]

    TRANSACTION_TIME_PATTERNS = [
        r"created_?at",
        r"updated_?at",
        r"modified_?at",
        r"transaction_?time",
        r"sys_?start",
        r"sys_?end",
        r"db_?timestamp",
        r"recorded_?at",
    ]

    # SCD Type 2 플래그 컬럼
    IS_CURRENT_PATTERNS = [
        r"is_?current",
        r"is_?active",
        r"is_?latest",
        r"current_?flag",
        r"active_?flag",
        r"status",  # 'active', 'inactive' 값
    ]

    # SCD Type 3 컬럼 (이전 값)
    PREVIOUS_VALUE_PATTERNS = [
        r"previous_?(.+)",
        r"old_?(.+)",
        r"prior_?(.+)",
        r"last_?(.+)",
        r"former_?(.+)",
    ]

    # 이력 테이블 패턴
    HISTORY_TABLE_PATTERNS = [
        (r"(.+)_hist(?:ory)?$", 1),          # employees_history → employees
        (r"(.+)_archive[sd]?$", 1),          # orders_archive → orders
        (r"(.+)_audit$", 1),                 # users_audit → users
        (r"(.+)_log$", 1),                   # changes_log → changes
        (r"(.+)_versions?$", 1),             # documents_versions → documents
        (r"(.+)_snapshots?$", 1),            # data_snapshots → data
        (r"(.+)_backups?$", 1),              # config_backup → config
        (r"hist(?:ory)?_(.+)$", 1),          # history_employees → employees
        (r"archive_(.+)$", 1),               # archive_orders → orders
    ]

    # 시간 관련 데이터 타입
    TEMPORAL_DATA_TYPES = [
        "date", "datetime", "timestamp", "timestamptz",
        "datetime2", "smalldatetime", "datetimeoffset",
    ]

    def __init__(self):
        self.candidates: List[TemporalFKCandidate] = []
        self._compiled_patterns = self._compile_patterns()

    def _compile_patterns(self) -> Dict[str, List[re.Pattern]]:
        """패턴 사전 컴파일"""
        return {
            "valid_from": [re.compile(p, re.IGNORECASE) for p in self.VALID_FROM_PATTERNS],
            "valid_to": [re.compile(p, re.IGNORECASE) for p in self.VALID_TO_PATTERNS],
            "transaction_time": [re.compile(p, re.IGNORECASE) for p in self.TRANSACTION_TIME_PATTERNS],
            "is_current": [re.compile(p, re.IGNORECASE) for p in self.IS_CURRENT_PATTERNS],
            "previous_value": [re.compile(p, re.IGNORECASE) for p in self.PREVIOUS_VALUE_PATTERNS],
        }

    def detect_temporal_fks(
        self,
        tables: Dict[str, Dict[str, Any]],
        sample_data: Dict[str, List[Dict[str, Any]]],
        existing_fks: Optional[List[Any]] = None,
    ) -> List[TemporalFKCandidate]:
        """
        Temporal FK 탐지

        Args:
            tables: {table_name: {"columns": [...], "row_count": int}}
            sample_data: {table_name: [row_dicts]}
            existing_fks: 기존 FK 후보 (시간 조건 추가용)

        Returns:
            Temporal FK 후보 리스트
        """
        self.candidates = []
        existing_fks = existing_fks or []

        # 1. 테이블별 Temporal 컬럼 식별
        table_temporal_info = {}
        for table_name, table_info in tables.items():
            columns = [c.get("name", "") if isinstance(c, dict) else c
                      for c in table_info.get("columns", [])]
            column_types = {
                (c.get("name", "") if isinstance(c, dict) else ""):
                (c.get("data_type", "") if isinstance(c, dict) else "")
                for c in table_info.get("columns", [])
            }

            temporal_info = self._identify_temporal_columns(columns, column_types)
            temporal_info["table_name"] = table_name

            if temporal_info["has_temporal"]:
                table_temporal_info[table_name] = temporal_info
                logger.debug(f"[TemporalFKDetector] {table_name}: {temporal_info['temporal_type'].value}")

        # 2. 이력 테이블 ↔ 메인 테이블 연결
        table_names = list(tables.keys())
        for table_name in table_names:
            base_table = self._find_base_table(table_name, table_names)
            if base_table and base_table != table_name:
                candidate = self._create_history_fk(
                    history_table=table_name,
                    base_table=base_table,
                    tables=tables,
                    sample_data=sample_data,
                    temporal_cols=table_temporal_info.get(table_name, {}),
                )
                if candidate:
                    self.candidates.append(candidate)
                    logger.info(f"[TemporalFKDetector] History FK: {table_name} → {base_table}")

        # 3. 기존 FK에 Temporal 메타데이터 추가
        for fk in existing_fks:
            from_table = getattr(fk, 'from_table', '') or (fk.get('from_table', '') if isinstance(fk, dict) else '')
            to_table = getattr(fk, 'to_table', '') or (fk.get('to_table', '') if isinstance(fk, dict) else '')

            from_temporal = table_temporal_info.get(from_table, {})
            to_temporal = table_temporal_info.get(to_table, {})

            if from_temporal.get("has_temporal") or to_temporal.get("has_temporal"):
                temporal_fk = self._enhance_fk_with_temporal(
                    fk,
                    from_temporal,
                    to_temporal,
                    sample_data.get(from_table, []),
                    sample_data.get(to_table, []),
                )
                if temporal_fk:
                    self.candidates.append(temporal_fk)

        # 4. 중복 제거 및 정렬
        self.candidates = self._deduplicate_candidates(self.candidates)
        self.candidates.sort(key=lambda x: x.confidence, reverse=True)

        logger.info(f"[TemporalFKDetector] Found {len(self.candidates)} temporal FK candidates")
        return self.candidates

    def _identify_temporal_columns(
        self,
        columns: List[str],
        column_types: Dict[str, str],
    ) -> Dict[str, Any]:
        """Temporal 컬럼 식별 및 SCD 유형 분류"""
        result = {
            "has_temporal": False,
            "valid_from": None,
            "valid_to": None,
            "transaction_time": None,
            "is_current_col": None,
            "previous_value_cols": [],
            "temporal_type": None,
        }

        for col in columns:
            col_lower = col.lower()
            col_type = column_types.get(col, "").lower()

            # Valid From
            for pattern in self._compiled_patterns["valid_from"]:
                if pattern.match(col_lower):
                    result["valid_from"] = col
                    result["has_temporal"] = True
                    break

            # Valid To
            for pattern in self._compiled_patterns["valid_to"]:
                if pattern.match(col_lower):
                    result["valid_to"] = col
                    result["has_temporal"] = True
                    break

            # Transaction Time
            for pattern in self._compiled_patterns["transaction_time"]:
                if pattern.match(col_lower):
                    if not result["transaction_time"]:  # 첫 번째 매칭만
                        result["transaction_time"] = col
                    break

            # Is Current Flag (SCD Type 2 indicator)
            for pattern in self._compiled_patterns["is_current"]:
                if pattern.match(col_lower):
                    result["is_current_col"] = col
                    result["has_temporal"] = True
                    break

            # Previous Value Columns (SCD Type 3 indicator)
            for pattern in self._compiled_patterns["previous_value"]:
                match = pattern.match(col_lower)
                if match:
                    result["previous_value_cols"].append(col)
                    result["has_temporal"] = True
                    break

        # Temporal Type 분류
        result["temporal_type"] = self._classify_temporal_type(result)

        return result

    def _classify_temporal_type(self, temporal_info: Dict[str, Any]) -> TemporalType:
        """SCD 유형 분류"""
        has_valid_from = temporal_info.get("valid_from") is not None
        has_valid_to = temporal_info.get("valid_to") is not None
        has_transaction = temporal_info.get("transaction_time") is not None
        has_is_current = temporal_info.get("is_current_col") is not None
        has_previous = len(temporal_info.get("previous_value_cols", [])) > 0

        # Bitemporal: 유효 시간 + 트랜잭션 시간 모두
        if has_valid_from and has_valid_to and has_transaction:
            return TemporalType.BITEMPORAL

        # SCD Type 2: valid_from/to 또는 is_current 플래그
        if (has_valid_from and has_valid_to) or has_is_current:
            return TemporalType.SCD_TYPE_2

        # SCD Type 3: previous_* 컬럼
        if has_previous:
            return TemporalType.SCD_TYPE_3

        # Snapshot: 유효 시작만 있거나 트랜잭션 시간만
        if has_valid_from or has_transaction:
            return TemporalType.SNAPSHOT

        # Default: SCD Type 1 (이력 없음)
        return TemporalType.SCD_TYPE_1

    def _find_base_table(self, table_name: str, all_tables: List[str]) -> Optional[str]:
        """이력 테이블의 베이스 테이블 찾기"""
        table_lower = table_name.lower()

        for pattern_str, group_idx in self.HISTORY_TABLE_PATTERNS:
            pattern = re.compile(pattern_str, re.IGNORECASE)
            match = pattern.match(table_lower)
            if match:
                base_name = match.group(group_idx)

                # 베이스 테이블 존재 확인 (정확 매칭 또는 복수형)
                for t in all_tables:
                    t_lower = t.lower()
                    if t_lower == base_name:
                        return t
                    # 복수형 처리
                    if t_lower == base_name + "s":
                        return t
                    if t_lower == base_name + "es":
                        return t
                    # 단수형 역변환
                    if base_name.endswith("s") and t_lower == base_name[:-1]:
                        return t

        return None

    def _create_history_fk(
        self,
        history_table: str,
        base_table: str,
        tables: Dict[str, Dict[str, Any]],
        sample_data: Dict[str, List[Dict[str, Any]]],
        temporal_cols: Dict[str, Any],
    ) -> Optional[TemporalFKCandidate]:
        """이력 테이블 → 메인 테이블 FK 생성"""

        base_info = tables.get(base_table, {})
        history_info = tables.get(history_table, {})

        base_cols = [c.get("name", "") if isinstance(c, dict) else c
                    for c in base_info.get("columns", [])]
        history_cols = [c.get("name", "") if isinstance(c, dict) else c
                       for c in history_info.get("columns", [])]

        # 공통 ID 컬럼 찾기 (우선순위 기반)
        common_id = None

        # 1. PK 컬럼 확인
        base_pks = base_info.get("primary_keys", [])
        for pk in base_pks:
            if pk in history_cols:
                common_id = pk
                break

        # 2. 테이블명_id 패턴
        if not common_id:
            base_id_pattern = f"{base_table.lower()}_id"
            for col in history_cols:
                if col.lower() == base_id_pattern:
                    common_id = col
                    break

        # 3. 공통 *_id 컬럼
        if not common_id:
            for col in history_cols:
                col_lower = col.lower()
                if col_lower.endswith("_id") or col_lower == "id":
                    if col in base_cols:
                        common_id = col
                        break

        if not common_id:
            return None

        # 값 검증 (이력 테이블 값이 베이스에 존재하는지)
        value_score = self._calculate_value_inclusion(
            sample_data.get(history_table, []),
            sample_data.get(base_table, []),
            common_id,
            common_id,
        )

        if value_score < 0.3:
            return None  # 값 포함도 너무 낮음

        # Temporal 정보
        temporal_type = temporal_cols.get("temporal_type", TemporalType.SCD_TYPE_2)
        if temporal_type == TemporalType.SCD_TYPE_1:
            temporal_type = TemporalType.SCD_TYPE_2  # 이력 테이블이면 최소 Type 2

        # 신뢰도 계산
        pattern_score = 0.9  # 이력 테이블 패턴 매칭
        temporal_score = 0.85 if temporal_cols.get("has_temporal") else 0.6
        confidence = (pattern_score * 0.3 + temporal_score * 0.3 + value_score * 0.4)

        return TemporalFKCandidate(
            from_table=history_table,
            from_column=common_id,
            to_table=base_table,
            to_column=common_id,
            temporal_type=temporal_type,
            valid_from_col=temporal_cols.get("valid_from"),
            valid_to_col=temporal_cols.get("valid_to"),
            transaction_time_col=temporal_cols.get("transaction_time"),
            pattern_score=pattern_score,
            temporal_score=temporal_score,
            value_score=value_score,
            confidence=confidence,
        )

    def _enhance_fk_with_temporal(
        self,
        fk: Any,  # FKCandidate or dict
        from_temporal: Dict[str, Any],
        to_temporal: Dict[str, Any],
        from_data: List[Dict],
        to_data: List[Dict],
    ) -> Optional[TemporalFKCandidate]:
        """기존 FK에 Temporal 메타데이터 추가"""

        # FK 정보 추출
        if isinstance(fk, dict):
            from_table = fk.get("from_table", "")
            from_column = fk.get("from_column", "")
            to_table = fk.get("to_table", "")
            to_column = fk.get("to_column", "")
            fk_score = fk.get("fk_score", 0)
            name_sim = fk.get("name_similarity", 0)
            val_inc = fk.get("value_inclusion", 0)
        else:
            from_table = getattr(fk, 'from_table', '')
            from_column = getattr(fk, 'from_column', '')
            to_table = getattr(fk, 'to_table', '')
            to_column = getattr(fk, 'to_column', '')
            fk_score = getattr(fk, 'fk_score', 0)
            name_sim = getattr(fk, 'name_similarity', 0)
            val_inc = getattr(fk, 'value_inclusion', 0)

        # 이미 낮은 신뢰도 FK는 스킵
        if fk_score < 0.4:
            return None

        # Temporal 정보 결정 (from 테이블 우선)
        if from_temporal.get("has_temporal"):
            temporal_cols = from_temporal
        elif to_temporal.get("has_temporal"):
            temporal_cols = to_temporal
        else:
            return None

        temporal_type = temporal_cols.get("temporal_type", TemporalType.SCD_TYPE_2)

        # 신뢰도 보너스 (Temporal 정보로 강화)
        temporal_bonus = 0.05 if temporal_cols.get("has_temporal") else 0
        confidence = min(1.0, fk_score + temporal_bonus)

        return TemporalFKCandidate(
            from_table=from_table,
            from_column=from_column,
            to_table=to_table,
            to_column=to_column,
            temporal_type=temporal_type,
            valid_from_col=temporal_cols.get("valid_from"),
            valid_to_col=temporal_cols.get("valid_to"),
            transaction_time_col=temporal_cols.get("transaction_time"),
            previous_value_col=temporal_cols.get("previous_value_cols", [None])[0] if temporal_cols.get("previous_value_cols") else None,
            pattern_score=name_sim,
            temporal_score=0.8 if temporal_cols.get("has_temporal") else 0.5,
            value_score=val_inc,
            confidence=confidence,
        )

    def _calculate_value_inclusion(
        self,
        from_data: List[Dict],
        to_data: List[Dict],
        from_col: str,
        to_col: str,
    ) -> float:
        """값 포함도 계산"""
        if not from_data or not to_data:
            return 0.5  # 데이터 없으면 기본값

        # To 테이블의 값 집합
        to_values = set()
        for row in to_data:
            val = row.get(to_col)
            if val is not None:
                to_values.add(str(val).lower())

        if not to_values:
            return 0.0

        # From 테이블 값 중 To에 포함된 비율
        matched = 0
        total = 0
        for row in from_data:
            val = row.get(from_col)
            if val is not None and str(val).lower() not in ("", "null", "none"):
                total += 1
                if str(val).lower() in to_values:
                    matched += 1

        return matched / total if total > 0 else 0.0

    def _deduplicate_candidates(
        self,
        candidates: List[TemporalFKCandidate]
    ) -> List[TemporalFKCandidate]:
        """중복 제거 (더 높은 confidence 유지)"""
        seen = {}
        for c in candidates:
            key = (c.from_table, c.from_column, c.to_table, c.to_column)
            if key not in seen or c.confidence > seen[key].confidence:
                seen[key] = c
        return list(seen.values())


# SharedContext 연동 함수
def detect_temporal_fks_and_update_context(
    shared_context: "SharedContext",
) -> List[TemporalFKCandidate]:
    """
    Temporal FK 탐지 후 SharedContext 업데이트

    사용법 (RelationshipDetector 에이전트에서):
        from .temporal_fk_detector import detect_temporal_fks_and_update_context
        temporal_fks = detect_temporal_fks_and_update_context(self.shared_context)
    """
    FKCandidate = _get_fk_candidate_class()

    # 테이블 데이터 추출
    tables = {}
    sample_data = {}

    for table_name, table_info in shared_context.tables.items():
        tables[table_name] = {
            "columns": table_info.columns if hasattr(table_info, 'columns') else table_info.get('columns', []),
            "row_count": table_info.row_count if hasattr(table_info, 'row_count') else table_info.get('row_count', 0),
            "primary_keys": table_info.primary_keys if hasattr(table_info, 'primary_keys') else table_info.get('primary_keys', []),
        }
        sample_data[table_name] = (
            table_info.sample_data if hasattr(table_info, 'sample_data')
            else table_info.get('sample_data', [])
        )

    # 기존 FK 후보
    existing_fks = []
    for fk_dict in shared_context.enhanced_fk_candidates:
        if isinstance(fk_dict, dict):
            existing_fks.append(FKCandidate(
                from_table=fk_dict.get("from_table", ""),
                from_column=fk_dict.get("from_column", ""),
                to_table=fk_dict.get("to_table", ""),
                to_column=fk_dict.get("to_column", ""),
                fk_score=fk_dict.get("fk_score", 0),
                name_similarity=fk_dict.get("name_similarity", 0),
                value_inclusion=fk_dict.get("value_inclusion", 0),
            ))

    # Temporal FK 탐지
    detector = TemporalFKDetector()
    temporal_fks = detector.detect_temporal_fks(tables, sample_data, existing_fks)

    # SharedContext에 추가 (FKCandidate 형태로 변환)
    for tfk in temporal_fks:
        fk_dict = tfk.to_fk_candidate().to_dict()
        fk_dict["is_temporal"] = True
        fk_dict["temporal_type"] = tfk.temporal_type.value
        fk_dict["valid_from_col"] = tfk.valid_from_col
        fk_dict["valid_to_col"] = tfk.valid_to_col
        fk_dict["transaction_time_col"] = tfk.transaction_time_col

        if tfk.previous_value_col:
            fk_dict["previous_value_col"] = tfk.previous_value_col

        shared_context.enhanced_fk_candidates.append(fk_dict)

    # dynamic_data에도 원본 저장
    shared_context.set_dynamic("temporal_fk_candidates", [
        tfk.to_dict() for tfk in temporal_fks
    ])

    logger.info(f"[TemporalFKDetector] Added {len(temporal_fks)} temporal FKs to context")
    return temporal_fks


# 유틸리티 함수
def identify_scd_type(table_info: Dict[str, Any]) -> TemporalType:
    """테이블의 SCD 유형 식별 (단독 사용 가능)"""
    columns = [c.get("name", "") if isinstance(c, dict) else c
              for c in table_info.get("columns", [])]
    column_types = {
        (c.get("name", "") if isinstance(c, dict) else ""):
        (c.get("data_type", "") if isinstance(c, dict) else "")
        for c in table_info.get("columns", [])
    }

    detector = TemporalFKDetector()
    temporal_info = detector._identify_temporal_columns(columns, column_types)
    return temporal_info.get("temporal_type", TemporalType.SCD_TYPE_1)
