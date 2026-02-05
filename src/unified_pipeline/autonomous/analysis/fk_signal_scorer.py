# analysis/fk_signal_scorer.py
"""
Solution 1: Multi-Signal FK Scoring System

단일 지표(Jaccard/Containment) 대신 5가지 신호를 종합하여 FK 판별
- 각 신호는 독립적으로 FK 가능성을 평가
- 가중치는 신호의 신뢰도에 따라 조정
"""

from dataclasses import dataclass, field
from typing import List, Set, Any, Tuple, Dict
import math
import logging

logger = logging.getLogger(__name__)


@dataclass
class FKSignals:
    """FK 판별을 위한 다중 신호"""

    # Signal 1: Value Overlap
    jaccard: float = 0.0
    containment_fk_in_pk: float = 0.0
    containment_pk_in_fk: float = 0.0

    # Signal 2: Cardinality Pattern
    fk_unique_count: int = 0
    pk_unique_count: int = 0
    fk_row_count: int = 0
    pk_row_count: int = 0

    # Signal 3: Referential Integrity
    orphan_count: int = 0        # FK에만 있는 값 수
    orphan_ratio: float = 0.0    # orphan / FK unique count

    # Signal 4: Naming Semantics
    name_similarity: float = 0.0
    has_fk_suffix: bool = False
    has_pk_suffix: bool = False
    entity_match_score: float = 0.0

    # Signal 5: Structural Pattern
    pk_is_unique: bool = False
    pk_unique_ratio: float = 0.0
    table_size_ratio: float = 0.0  # FK rows / PK rows


class FKSignalScorer:
    """
    Multi-Signal FK Scoring System

    단일 지표 대신 5가지 신호를 종합하여 FK 판별
    - 각 신호는 독립적으로 FK 가능성을 평가
    - 가중치는 신호의 신뢰도에 따라 조정
    """

    # FK suffix 패턴
    FK_SUFFIXES = {'_id', '_code', '_no', '_key', '_ref', '_fk', '_num'}
    PK_PATTERNS = {'id', 'pk', 'key', 'code'}

    # Entity synonym groups (domain-agnostic)
    ENTITY_SYNONYMS = {
        "customer": {"buyer", "user", "client", "cust", "customer", "consumer", "member"},
        "product": {"prod", "item", "goods", "product", "sku", "article"},
        "order": {"order", "purchase", "transaction", "txn", "sale"},
        "campaign": {"campaign", "cmp", "promotion", "marketing"},
        "employee": {"emp", "employee", "worker", "staff"},
        "account": {"acct", "account"},
        # Healthcare domain (v22.0 추가)
        "doctor": {"doctor", "physician", "provider", "clinician", "doc", "dr",
                   "attending", "ordering", "prescribing", "referring"},
        "patient": {"patient", "pt", "pat", "member", "enrollee", "beneficiary"},
        "appointment": {"appointment", "appt", "visit", "encounter", "consultation"},
        "medication": {"medication", "med", "meds", "drug", "rx", "prescription"},
    }

    def compute_signals(
        self,
        fk_values: List[Any],
        pk_values: List[Any],
        fk_column: str,
        pk_column: str,
        fk_table: str,
        pk_table: str,
    ) -> FKSignals:
        """모든 신호 계산"""

        # None 값 필터링 및 문자열 변환
        fk_set = set(str(v) for v in fk_values if v is not None and str(v).strip())
        pk_set = set(str(v) for v in pk_values if v is not None and str(v).strip())

        if not fk_set or not pk_set:
            return FKSignals()

        # Signal 1: Value Overlap
        intersection = fk_set & pk_set
        union = fk_set | pk_set
        jaccard = len(intersection) / len(union) if union else 0.0
        containment_fk_in_pk = len(intersection) / len(fk_set) if fk_set else 0.0
        containment_pk_in_fk = len(intersection) / len(pk_set) if pk_set else 0.0

        # Signal 2: Cardinality
        fk_unique = len(fk_set)
        pk_unique = len(pk_set)

        # Signal 3: Referential Integrity
        orphans = fk_set - pk_set
        orphan_ratio = len(orphans) / len(fk_set) if fk_set else 1.0

        # Signal 4: Naming
        name_sim = self._compute_name_similarity(fk_column, pk_column, pk_table)
        has_fk_suf = any(fk_column.lower().endswith(s) for s in self.FK_SUFFIXES)
        has_pk_suf = any(pk_column.lower().endswith(s) for s in self.FK_SUFFIXES)
        entity_score = self._compute_entity_match(fk_column, pk_table)

        # Signal 5: Structure
        pk_unique_ratio = pk_unique / len(pk_values) if pk_values else 0
        pk_is_unique = pk_unique_ratio > 0.95
        size_ratio = len(fk_values) / len(pk_values) if pk_values else 0

        return FKSignals(
            jaccard=jaccard,
            containment_fk_in_pk=containment_fk_in_pk,
            containment_pk_in_fk=containment_pk_in_fk,
            fk_unique_count=fk_unique,
            pk_unique_count=pk_unique,
            fk_row_count=len(fk_values),
            pk_row_count=len(pk_values),
            orphan_count=len(orphans),
            orphan_ratio=orphan_ratio,
            name_similarity=name_sim,
            has_fk_suffix=has_fk_suf,
            has_pk_suffix=has_pk_suf,
            entity_match_score=entity_score,
            pk_is_unique=pk_is_unique,
            pk_unique_ratio=pk_unique_ratio,
            table_size_ratio=size_ratio,
        )

    def compute_confidence(self, signals: FKSignals) -> Tuple[float, Dict]:
        """
        다중 신호 기반 종합 confidence 계산

        Returns:
            (confidence, breakdown)
        """
        breakdown = {}

        # === Signal 1: Value Overlap (25%) ===
        # FK 값이 PK에 포함되어야 함
        overlap_score = signals.containment_fk_in_pk * 0.25
        breakdown["value_overlap"] = {
            "score": overlap_score,
            "containment_fk_in_pk": signals.containment_fk_in_pk,
            "jaccard": signals.jaccard,
            "weight": 0.25
        }

        # === Signal 2: Cardinality Pattern (20%) ===
        # FK cardinality <= PK cardinality
        card_score = 0.0
        if signals.pk_unique_count > 0:
            ratio = signals.fk_unique_count / signals.pk_unique_count
            if ratio <= 1.0:
                card_score = 0.20 * (1.0 - ratio * 0.5)  # ratio 낮을수록 좋음
            else:
                card_score = 0.20 * max(0, 1.0 - (ratio - 1.0))  # 초과시 감점
        breakdown["cardinality"] = {
            "score": card_score,
            "fk_unique": signals.fk_unique_count,
            "pk_unique": signals.pk_unique_count,
            "ratio": signals.fk_unique_count / signals.pk_unique_count if signals.pk_unique_count > 0 else 0,
            "weight": 0.20
        }

        # === Signal 3: Referential Integrity (25%) ===
        # Orphan (FK에만 있는 값) 적어야 함
        integrity_score = (1.0 - signals.orphan_ratio) * 0.25
        breakdown["referential_integrity"] = {
            "score": integrity_score,
            "orphan_ratio": signals.orphan_ratio,
            "orphan_count": signals.orphan_count,
            "weight": 0.25
        }

        # === Signal 4: Naming Semantics (15%) ===
        naming_score = 0.0
        if signals.has_fk_suffix:
            naming_score += 0.05
        naming_score += signals.entity_match_score * 0.05
        naming_score += signals.name_similarity * 0.05
        breakdown["naming"] = {
            "score": naming_score,
            "has_fk_suffix": signals.has_fk_suffix,
            "entity_match": signals.entity_match_score,
            "name_similarity": signals.name_similarity,
            "weight": 0.15
        }

        # === Signal 5: Structural Pattern (15%) ===
        struct_score = 0.0
        if signals.pk_is_unique:
            struct_score += 0.10  # PK가 실제로 unique면 가점
        if signals.table_size_ratio >= 1.0:
            struct_score += 0.05  # FK 테이블이 더 크면 N:1 패턴
        breakdown["structure"] = {
            "score": struct_score,
            "pk_is_unique": signals.pk_is_unique,
            "pk_unique_ratio": signals.pk_unique_ratio,
            "size_ratio": signals.table_size_ratio,
            "weight": 0.15
        }

        total = (overlap_score + card_score + integrity_score +
                 naming_score + struct_score)

        return (min(total, 1.0), breakdown)

    def _compute_name_similarity(
        self, fk_col: str, pk_col: str, pk_table: str
    ) -> float:
        """컬럼명 유사도 계산"""
        fk_lower = fk_col.lower()
        pk_lower = pk_col.lower()

        # 동일하면 1.0
        if fk_lower == pk_lower:
            return 1.0

        # FK가 테이블명_id 패턴이면 0.8
        pk_table_singular = pk_table.rstrip('s').lower()
        if pk_table_singular in fk_lower:
            return 0.8

        # 공통 토큰 기반
        fk_tokens = set(fk_lower.replace('_', ' ').split())
        pk_tokens = set(pk_lower.replace('_', ' ').split())
        if fk_tokens & pk_tokens:
            return 0.5

        return 0.0

    def _compute_entity_match(self, fk_col: str, pk_table: str) -> float:
        """FK 컬럼명과 PK 테이블명의 entity 일치도"""
        fk_entity = self._extract_entity(fk_col)
        pk_entity = self._extract_entity(pk_table)

        for canonical, synonyms in self.ENTITY_SYNONYMS.items():
            fk_match = fk_entity in synonyms
            pk_match = pk_entity in synonyms or canonical in pk_table.lower()
            if fk_match and pk_match:
                return 1.0

        # 직접 비교
        if fk_entity == pk_entity:
            return 0.8

        # 부분 일치
        if fk_entity in pk_entity or pk_entity in fk_entity:
            return 0.5

        return 0.0

    def _extract_entity(self, name: str) -> str:
        """이름에서 entity 추출"""
        name = name.lower()
        for suffix in self.FK_SUFFIXES:
            if name.endswith(suffix):
                return name[:-len(suffix)]
        return name.rstrip('s')  # 복수형 제거
