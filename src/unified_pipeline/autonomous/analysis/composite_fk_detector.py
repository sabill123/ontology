"""
Composite FK Detector (v16.0)

복합키 FK 탐지:
- 주문-상세 (order_id + line_no)
- 다중 컬럼 유니크 제약
- 브릿지 테이블 복합 참조

EnhancedFKDetector와 통합되어 사용됨

References:
- Papenbrock et al. "A Hybrid Approach to Functional Dependency Discovery" (2015)
- Abedjan et al. "Profiling Relational Data: A Survey" (2015)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Tuple, Optional, Set, TYPE_CHECKING
from itertools import combinations
import logging

from .enhanced_fk_detector import FKCandidate, EnhancedFKDetector

if TYPE_CHECKING:
    from ...shared_context import SharedContext

logger = logging.getLogger(__name__)


@dataclass
class CompositeFKCandidate:
    """복합 FK 후보"""
    from_table: str
    from_columns: Tuple[str, ...]  # (order_id, line_no)
    to_table: str
    to_columns: Tuple[str, ...]    # (order_id, line_no)

    # Scores
    column_match_score: float = 0.0
    value_inclusion_score: float = 0.0
    uniqueness_score: float = 0.0

    # Cardinality
    cardinality: str = "N:1"  # "1:1", "N:1", "M:N"

    # Final
    composite_score: float = 0.0
    confidence: str = "low"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "from_table": self.from_table,
            "from_columns": list(self.from_columns),
            "to_table": self.to_table,
            "to_columns": list(self.to_columns),
            "column_match_score": round(self.column_match_score, 4),
            "value_inclusion_score": round(self.value_inclusion_score, 4),
            "uniqueness_score": round(self.uniqueness_score, 4),
            "cardinality": self.cardinality,
            "composite_score": round(self.composite_score, 4),
            "confidence": self.confidence,
            "is_composite": True,
        }

    def to_fk_candidates(self) -> List[FKCandidate]:
        """단일 FK 후보 리스트로 변환 (하위 호환성)"""
        candidates = []
        for fc, tc in zip(self.from_columns, self.to_columns):
            candidates.append(FKCandidate(
                from_table=self.from_table,
                from_column=fc,
                to_table=self.to_table,
                to_column=tc,
                fk_score=self.composite_score,
                confidence=self.confidence,
            ))
        return candidates


class CompositeFKDetector:
    """
    복합 FK 탐지기

    알고리즘:
    1. 테이블별 컬럼 조합 생성 (2-4개)
    2. 조합의 유니크성 검사
    3. Cross-table 값 포함도 계산
    4. Cardinality 분석
    5. 복합 FK 점수 계산

    EnhancedFKDetector와 연동:
    - enhanced_fk_detector.candidates에 단일 FK가 있으면 복합 FK 후보에서 제외
    - 결과는 SharedContext.enhanced_fk_candidates에 추가
    """

    MAX_COMPOSITE_SIZE = 4  # 최대 복합키 컬럼 수
    MIN_UNIQUENESS_RATIO = 0.8  # 최소 유니크 비율
    MIN_COMPOSITE_SCORE = 0.5  # 최소 복합 FK 점수

    def __init__(self, single_fk_detector: Optional[EnhancedFKDetector] = None):
        """
        Args:
            single_fk_detector: 단일 FK 탐지기 (이미 탐지된 FK 제외용)
        """
        self.single_fk_detector = single_fk_detector or EnhancedFKDetector()
        self.candidates: List[CompositeFKCandidate] = []

    def detect_composite_fks(
        self,
        tables: Dict[str, Dict[str, Any]],
        sample_data: Dict[str, List[Dict[str, Any]]],
        existing_fks: Optional[List[FKCandidate]] = None,
    ) -> List[CompositeFKCandidate]:
        """
        복합 FK 탐지

        Args:
            tables: {table_name: {"columns": [...], "row_count": int}}
            sample_data: {table_name: [row_dicts]}
            existing_fks: 이미 탐지된 단일 FK 목록 (제외용)

        Returns:
            복합 FK 후보 리스트
        """
        self.candidates = []
        existing_fks = existing_fks or []

        # 이미 탐지된 단일 FK 쌍 기록
        existing_pairs = {
            (fk.from_table, fk.from_column, fk.to_table, fk.to_column)
            for fk in existing_fks
        }

        table_names = list(tables.keys())

        # 1. 각 테이블의 잠재적 복합키 컬럼 조합 생성
        table_composite_keys: Dict[str, List[Tuple[Tuple[str, ...], float]]] = {}

        for table_name, table_info in tables.items():
            columns = [c.get("name", "") for c in table_info.get("columns", [])]
            id_columns = [c for c in columns if self._is_key_column(c)]

            if len(id_columns) >= 2:
                # v28.5: cap 적용 — C(89,4)=2.6M → C(15,4)=1,365 (1,900x 감소)
                id_columns = id_columns[:15]
                # 2-4개 조합 생성
                composites = []
                for size in range(2, min(len(id_columns), self.MAX_COMPOSITE_SIZE) + 1):
                    for combo in combinations(id_columns, size):
                        # 유니크성 검사
                        uniqueness = self._check_uniqueness(
                            sample_data.get(table_name, []),
                            combo
                        )
                        if uniqueness >= self.MIN_UNIQUENESS_RATIO:
                            composites.append((combo, uniqueness))

                if composites:
                    table_composite_keys[table_name] = composites

        logger.debug(f"[CompositeFKDetector] Found composite key candidates in {len(table_composite_keys)} tables")

        # 2. Cross-table 복합 FK 탐지
        for from_table in table_names:
            from_composites = table_composite_keys.get(from_table, [])
            from_data = sample_data.get(from_table, [])

            for to_table in table_names:
                if from_table == to_table:
                    continue

                to_composites = table_composite_keys.get(to_table, [])
                to_data = sample_data.get(to_table, [])

                # v28.5: cross-table composites cap — O(C₁²×C₂²) 방지
                from_composites = from_composites[:30]
                to_composites_capped = to_composites[:30]
                # 복합키 쌍 비교
                for from_combo, from_uniq in from_composites:
                    for to_combo, to_uniq in to_composites_capped:
                        # 컬럼 수가 같아야 함
                        if len(from_combo) != len(to_combo):
                            continue

                        # 이미 단일 FK로 탐지된 컬럼 조합 제외
                        if self._overlaps_single_fk(from_table, from_combo,
                                                   to_table, to_combo,
                                                   existing_pairs):
                            continue

                        # 복합 FK 점수 계산
                        candidate = self._score_composite_candidate(
                            from_table, from_combo, from_data, from_uniq,
                            to_table, to_combo, to_data, to_uniq,
                        )

                        if candidate and candidate.composite_score >= self.MIN_COMPOSITE_SCORE:
                            self.candidates.append(candidate)

        # 3. 정렬 및 중복 제거
        self.candidates = self._deduplicate(self.candidates)
        self.candidates.sort(key=lambda x: x.composite_score, reverse=True)

        logger.info(f"[CompositeFKDetector] Found {len(self.candidates)} composite FK candidates")
        return self.candidates

    def _is_key_column(self, col_name: str) -> bool:
        """키 컬럼 여부 (복합키 후보)"""
        col_lower = col_name.lower()
        key_indicators = [
            "_id", "id_", "_key", "_no", "_num", "_code", "_ref",
            "_seq", "_idx", "_number", "line_", "seq_", "item_"
        ]
        return any(ind in col_lower for ind in key_indicators) or col_lower == "id"

    def _check_uniqueness(
        self,
        data: List[Dict[str, Any]],
        columns: Tuple[str, ...]
    ) -> float:
        """복합키 유니크 비율 계산"""
        if not data:
            return 0.0

        # 복합 값 생성
        composite_values = []
        for row in data:
            values = tuple(row.get(c) for c in columns)
            if None not in values:
                composite_values.append(values)

        if not composite_values:
            return 0.0

        unique_count = len(set(composite_values))
        return unique_count / len(composite_values)

    def _overlaps_single_fk(
        self,
        from_table: str,
        from_combo: Tuple[str, ...],
        to_table: str,
        to_combo: Tuple[str, ...],
        existing_pairs: Set[Tuple],
    ) -> bool:
        """단일 FK와 모든 컬럼이 겹치는지 확인"""
        overlap_count = 0
        for fc, tc in zip(from_combo, to_combo):
            if (from_table, fc, to_table, tc) in existing_pairs:
                overlap_count += 1

        # 모든 컬럼이 이미 단일 FK로 탐지된 경우만 제외
        return overlap_count == len(from_combo)

    def _score_composite_candidate(
        self,
        from_table: str,
        from_combo: Tuple[str, ...],
        from_data: List[Dict],
        from_uniq: float,
        to_table: str,
        to_combo: Tuple[str, ...],
        to_data: List[Dict],
        to_uniq: float,
    ) -> Optional[CompositeFKCandidate]:
        """복합 FK 후보 점수 계산"""

        # 1. 컬럼명 매칭 점수
        column_match = self._calc_column_match(from_combo, to_combo)

        if column_match < 0.3:
            return None  # 컬럼명 매칭 미달

        # 2. 값 포함도 (복합 값 기준)
        inclusion = self._calc_composite_inclusion(
            from_data, from_combo,
            to_data, to_combo
        )

        if inclusion < 0.3:
            return None  # 최소 포함도 미달

        # 3. 유니크성 점수
        uniqueness = (from_uniq + to_uniq) / 2

        # 4. Cardinality 분석
        cardinality = self._determine_cardinality(
            from_data, from_combo,
            to_data, to_combo
        )

        # 5. 최종 점수
        composite_score = (
            column_match * 0.3 +
            inclusion * 0.5 +
            uniqueness * 0.2
        )

        # Confidence 결정
        if composite_score >= 0.85:
            confidence = "high"
        elif composite_score >= 0.65:
            confidence = "medium"
        else:
            confidence = "low"

        return CompositeFKCandidate(
            from_table=from_table,
            from_columns=from_combo,
            to_table=to_table,
            to_columns=to_combo,
            column_match_score=column_match,
            value_inclusion_score=inclusion,
            uniqueness_score=uniqueness,
            cardinality=cardinality,
            composite_score=composite_score,
            confidence=confidence,
        )

    def _calc_column_match(
        self,
        from_combo: Tuple[str, ...],
        to_combo: Tuple[str, ...]
    ) -> float:
        """컬럼명 매칭 점수"""
        matches = 0
        for fc, tc in zip(from_combo, to_combo):
            fc_base = self.single_fk_detector._extract_base_name(fc)
            tc_base = self.single_fk_detector._extract_base_name(tc)

            if fc_base == tc_base:
                matches += 1
            elif self.single_fk_detector._levenshtein_similarity(fc_base, tc_base) > 0.7:
                matches += 0.7
            else:
                # 축약어 확장 후 비교
                fc_expanded = self.single_fk_detector._expand_abbreviation(fc_base)
                tc_expanded = self.single_fk_detector._expand_abbreviation(tc_base)
                if fc_expanded == tc_expanded:
                    matches += 0.9

        return matches / len(from_combo)

    def _calc_composite_inclusion(
        self,
        from_data: List[Dict],
        from_combo: Tuple[str, ...],
        to_data: List[Dict],
        to_combo: Tuple[str, ...]
    ) -> float:
        """복합 값 포함도 계산"""
        if not from_data or not to_data:
            return 0.0

        # From 복합 값
        from_values = set()
        for row in from_data:
            values = tuple(str(row.get(c, "")).lower().strip() for c in from_combo)
            if "" not in values:
                from_values.add(values)

        # To 복합 값
        to_values = set()
        for row in to_data:
            values = tuple(str(row.get(c, "")).lower().strip() for c in to_combo)
            if "" not in values:
                to_values.add(values)

        if not from_values:
            return 0.0

        matched = from_values & to_values
        return len(matched) / len(from_values)

    def _determine_cardinality(
        self,
        from_data: List[Dict],
        from_combo: Tuple[str, ...],
        to_data: List[Dict],
        to_combo: Tuple[str, ...]
    ) -> str:
        """Cardinality 결정"""
        # From 유니크 수
        from_values = set()
        for row in from_data:
            values = tuple(row.get(c) for c in from_combo)
            if None not in values:
                from_values.add(values)

        # To 유니크 수
        to_values = set()
        for row in to_data:
            values = tuple(row.get(c) for c in to_combo)
            if None not in values:
                to_values.add(values)

        from_unique = len(from_values)
        to_unique = len(to_values)
        from_total = len(from_data)
        to_total = len(to_data)

        # 1:1, N:1, M:N 판단
        if from_total > 0 and to_total > 0:
            from_ratio = from_unique / from_total
            to_ratio = to_unique / to_total

            if from_ratio > 0.95 and to_ratio > 0.95:
                return "1:1"
            elif from_unique >= to_unique:
                return "N:1"
            else:
                return "M:N"

        return "N:1"

    def _deduplicate(
        self,
        candidates: List[CompositeFKCandidate]
    ) -> List[CompositeFKCandidate]:
        """중복 제거 (더 높은 점수 유지)"""
        seen: Dict[Tuple, CompositeFKCandidate] = {}
        for c in candidates:
            key = (c.from_table, c.from_columns, c.to_table, c.to_columns)
            if key not in seen or c.composite_score > seen[key].composite_score:
                seen[key] = c
        return list(seen.values())


def detect_composite_fks_and_update_context(
    shared_context: "SharedContext",
) -> List[CompositeFKCandidate]:
    """
    복합 FK 탐지 후 SharedContext 업데이트

    사용법 (RelationshipDetector 에이전트에서):
        from ..analysis.composite_fk_detector import detect_composite_fks_and_update_context
        composite_fks = detect_composite_fks_and_update_context(self.shared_context)

    Args:
        shared_context: 공유 컨텍스트

    Returns:
        탐지된 복합 FK 후보 리스트
    """
    # 테이블 데이터 추출
    tables: Dict[str, Dict[str, Any]] = {}
    sample_data: Dict[str, List[Dict[str, Any]]] = {}

    for table_name, table_info in shared_context.tables.items():
        tables[table_name] = {
            "columns": table_info.columns,
            "row_count": table_info.row_count,
        }
        sample_data[table_name] = table_info.sample_data

    # 기존 단일 FK 변환
    existing_fks: List[FKCandidate] = []
    for fk_dict in shared_context.enhanced_fk_candidates:
        if isinstance(fk_dict, dict) and not fk_dict.get("is_composite", False):
            existing_fks.append(FKCandidate(
                from_table=fk_dict.get("from_table", ""),
                from_column=fk_dict.get("from_column", ""),
                to_table=fk_dict.get("to_table", ""),
                to_column=fk_dict.get("to_column", ""),
                fk_score=fk_dict.get("fk_score", 0),
                confidence=fk_dict.get("confidence", "low"),
            ))

    # 복합 FK 탐지
    detector = CompositeFKDetector()
    composite_fks = detector.detect_composite_fks(tables, sample_data, existing_fks)

    # SharedContext에 추가 (단일 FK 형태로 변환하여 호환성 유지)
    for cfk in composite_fks:
        # 복합 FK 정보 포함하여 저장
        composite_dict = cfk.to_dict()
        composite_dict["composite_columns"] = {
            "from": list(cfk.from_columns),
            "to": list(cfk.to_columns),
        }
        shared_context.enhanced_fk_candidates.append(composite_dict)

    # dynamic_data에 원본 저장
    shared_context.set_dynamic("composite_fk_candidates", [
        cfk.to_dict() for cfk in composite_fks
    ])

    logger.info(f"[CompositeFKDetector] Added {len(composite_fks)} composite FKs to context")
    return composite_fks
