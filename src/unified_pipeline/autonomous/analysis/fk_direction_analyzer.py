# analysis/fk_direction_analyzer.py
"""
Solution 2: Probabilistic FK Direction Analyzer

단일 룰 대신 여러 신호를 종합하여 방향 결정
- 각 신호가 독립적으로 방향 제안
- 신호들의 합의로 최종 방향 결정
- 합의 부족시 "uncertain" 반환 → LLM에 위임
"""

from dataclasses import dataclass
from typing import Tuple, Optional, List, Any
import logging

logger = logging.getLogger(__name__)


@dataclass
class DirectionEvidence:
    """방향 판단 근거"""
    signal_name: str
    direction: str  # "A→B" or "B→A"
    strength: float  # 0.0 ~ 1.0
    reason: str


class FKDirectionAnalyzer:
    """
    확률적 FK 방향 분석기

    단일 룰 대신 여러 신호를 종합하여 방향 결정
    - 각 신호가 독립적으로 방향 제안
    - 신호들의 합의로 최종 방향 결정
    - 합의 부족시 "uncertain" 반환 → LLM에 위임
    """

    # FK suffix 패턴
    FK_SUFFIXES = ['_id', '_code', '_no', '_key', '_ref', '_fk', '_num']

    def analyze_direction(
        self,
        col_a_name: str,
        col_b_name: str,
        col_a_values: List[Any],
        col_b_values: List[Any],
        table_a: str,
        table_b: str,
    ) -> Tuple[Optional[str], Optional[str], float, List[DirectionEvidence]]:
        """
        FK 방향 분석

        Returns:
            (fk_col, pk_col, confidence, evidences)
            - 방향 불확실시 (None, None, confidence, evidences)
        """
        evidences = []

        # None 값 필터링 및 문자열 변환
        a_set = set(str(v) for v in col_a_values if v is not None and str(v).strip())
        b_set = set(str(v) for v in col_b_values if v is not None and str(v).strip())

        if not a_set or not b_set:
            return (None, None, 0.0, evidences)

        # === Evidence 1: Uniqueness ===
        # PK는 unique, FK는 중복 허용
        a_unique_ratio = len(a_set) / len(col_a_values) if col_a_values else 0
        b_unique_ratio = len(b_set) / len(col_b_values) if col_b_values else 0

        if a_unique_ratio > 0.95 and b_unique_ratio < 0.90:
            evidences.append(DirectionEvidence(
                signal_name="uniqueness",
                direction="B→A",
                strength=0.8,
                reason=f"A is unique ({a_unique_ratio:.1%}), B has duplicates ({b_unique_ratio:.1%})"
            ))
        elif b_unique_ratio > 0.95 and a_unique_ratio < 0.90:
            evidences.append(DirectionEvidence(
                signal_name="uniqueness",
                direction="A→B",
                strength=0.8,
                reason=f"B is unique ({b_unique_ratio:.1%}), A has duplicates ({a_unique_ratio:.1%})"
            ))
        elif a_unique_ratio > 0.95 and b_unique_ratio > 0.95:
            # 둘 다 unique - 추가 신호 필요
            pass
        elif abs(a_unique_ratio - b_unique_ratio) > 0.2:
            # 유의미한 차이가 있으면 약한 증거
            if a_unique_ratio > b_unique_ratio:
                evidences.append(DirectionEvidence(
                    signal_name="uniqueness_weak",
                    direction="B→A",
                    strength=0.4,
                    reason=f"A more unique ({a_unique_ratio:.1%} vs {b_unique_ratio:.1%})"
                ))
            else:
                evidences.append(DirectionEvidence(
                    signal_name="uniqueness_weak",
                    direction="A→B",
                    strength=0.4,
                    reason=f"B more unique ({b_unique_ratio:.1%} vs {a_unique_ratio:.1%})"
                ))

        # === Evidence 2: Containment ===
        # FK ⊆ PK (FK 값이 PK에 포함)
        a_in_b = len(a_set & b_set) / len(a_set) if a_set else 0  # A가 B에 포함된 비율
        b_in_a = len(a_set & b_set) / len(b_set) if b_set else 0  # B가 A에 포함된 비율

        if a_in_b > 0.9 and b_in_a < 0.7:
            evidences.append(DirectionEvidence(
                signal_name="containment",
                direction="A→B",
                strength=0.7,
                reason=f"A values contained in B ({a_in_b:.1%}), B not in A ({b_in_a:.1%})"
            ))
        elif b_in_a > 0.9 and a_in_b < 0.7:
            evidences.append(DirectionEvidence(
                signal_name="containment",
                direction="B→A",
                strength=0.7,
                reason=f"B values contained in A ({b_in_a:.1%}), A not in B ({a_in_b:.1%})"
            ))
        elif a_in_b > 0.8 and b_in_a > 0.8:
            # 양방향 높은 포함률 - 약한 증거 (cardinality로 판단)
            pass

        # === Evidence 3: Row Count ===
        # FK 테이블이 보통 더 큼 (N:1 관계)
        a_rows = len(col_a_values)
        b_rows = len(col_b_values)

        if a_rows > b_rows * 2:
            evidences.append(DirectionEvidence(
                signal_name="row_count",
                direction="A→B",
                strength=0.5,
                reason=f"A has more rows ({a_rows} vs {b_rows})"
            ))
        elif b_rows > a_rows * 2:
            evidences.append(DirectionEvidence(
                signal_name="row_count",
                direction="B→A",
                strength=0.5,
                reason=f"B has more rows ({b_rows} vs {a_rows})"
            ))

        # === Evidence 4: Column Naming ===
        # FK는 보통 _id, _code 등 suffix, PK는 테이블의 primary column
        a_has_fk_pattern = self._has_fk_pattern(col_a_name)
        b_has_fk_pattern = self._has_fk_pattern(col_b_name)
        a_is_pk_pattern = self._is_pk_pattern(col_a_name, table_a)
        b_is_pk_pattern = self._is_pk_pattern(col_b_name, table_b)

        if a_has_fk_pattern and b_is_pk_pattern:
            evidences.append(DirectionEvidence(
                signal_name="naming",
                direction="A→B",
                strength=0.6,
                reason=f"A ({col_a_name}) has FK pattern, B ({col_b_name}) is PK"
            ))
        elif b_has_fk_pattern and a_is_pk_pattern:
            evidences.append(DirectionEvidence(
                signal_name="naming",
                direction="B→A",
                strength=0.6,
                reason=f"B ({col_b_name}) has FK pattern, A ({col_a_name}) is PK"
            ))

        # === Evidence 5: Entity Reference Pattern ===
        # buyer_id → customers (buyer는 customer의 synonym)
        a_refs_b = self._references_table(col_a_name, table_b)
        b_refs_a = self._references_table(col_b_name, table_a)

        if a_refs_b and not b_refs_a:
            evidences.append(DirectionEvidence(
                signal_name="entity_reference",
                direction="A→B",
                strength=0.7,
                reason=f"Column '{col_a_name}' references table '{table_b}'"
            ))
        elif b_refs_a and not a_refs_b:
            evidences.append(DirectionEvidence(
                signal_name="entity_reference",
                direction="B→A",
                strength=0.7,
                reason=f"Column '{col_b_name}' references table '{table_a}'"
            ))

        # === Aggregate Evidences ===
        return self._aggregate_evidences(
            evidences, col_a_name, col_b_name, table_a, table_b
        )

    def _aggregate_evidences(
        self,
        evidences: List[DirectionEvidence],
        col_a: str, col_b: str,
        table_a: str, table_b: str,
    ) -> Tuple[Optional[str], Optional[str], float, List[DirectionEvidence]]:
        """증거들을 종합하여 방향 결정"""

        if not evidences:
            return (None, None, 0.0, evidences)

        # 방향별 점수 합산
        score_a_to_b = sum(e.strength for e in evidences if e.direction == "A→B")
        score_b_to_a = sum(e.strength for e in evidences if e.direction == "B→A")

        total = score_a_to_b + score_b_to_a
        if total == 0:
            return (None, None, 0.0, evidences)

        # 확신도 계산: 한쪽이 압도적이어야 높음
        diff = abs(score_a_to_b - score_b_to_a)
        confidence = diff / total if total > 0 else 0

        # 로깅
        logger.debug(f"Direction analysis: {table_a}.{col_a} <-> {table_b}.{col_b}")
        logger.debug(f"  A→B score: {score_a_to_b:.2f}, B→A score: {score_b_to_a:.2f}")
        logger.debug(f"  Confidence: {confidence:.2f}")

        # 최소 0.3 이상 차이나야 방향 결정
        if confidence < 0.3:
            return (None, None, confidence, evidences)  # uncertain

        if score_a_to_b > score_b_to_a:
            fk = f"{table_a}.{col_a}"
            pk = f"{table_b}.{col_b}"
        else:
            fk = f"{table_b}.{col_b}"
            pk = f"{table_a}.{col_a}"

        return (fk, pk, confidence, evidences)

    def _has_fk_pattern(self, col_name: str) -> bool:
        """FK 패턴 컬럼인지"""
        col = col_name.lower()
        return any(col.endswith(s) for s in self.FK_SUFFIXES)

    def _is_pk_pattern(self, col_name: str, table_name: str) -> bool:
        """PK 패턴 컬럼인지"""
        col = col_name.lower()
        table = table_name.lower().rstrip('s')  # customers → customer

        # customer_id in customers, product_id in products
        if col == f"{table}_id" or col == "id":
            return True

        # 테이블명 + _id
        if col.endswith('_id') and col[:-3] == table:
            return True

        return False

    def _references_table(self, col_name: str, table_name: str) -> bool:
        """컬럼명이 테이블을 참조하는지 (synonym 포함)"""
        # buyer_id → customers 테이블 참조
        SYNONYMS = {
            "customer": {"buyer", "user", "client", "cust", "customer"},
            "product": {"prod", "item", "product"},
            "order": {"order", "purchase"},
            "campaign": {"campaign", "cmp"},
        }

        col_entity = self._extract_entity(col_name)
        table_entity = table_name.lower().rstrip('s')

        # 직접 일치
        if col_entity == table_entity:
            return True

        # synonym 체크
        for canonical, syns in SYNONYMS.items():
            col_match = col_entity in syns
            table_match = table_entity in syns or canonical == table_entity
            if col_match and table_match:
                return True

        return False

    def _extract_entity(self, name: str) -> str:
        """이름에서 entity 추출"""
        name = name.lower()
        for suffix in self.FK_SUFFIXES:
            if name.endswith(suffix):
                return name[:-len(suffix)]
        return name
