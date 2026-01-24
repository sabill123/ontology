"""
Similarity Analysis Module

값 유사도 및 겹침 분석
- Jaccard similarity
- Value overlap
- Fuzzy matching
"""

import logging
import re
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Set, Tuple
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class JaccardResult:
    """Jaccard 유사도 결과"""
    table_a: str
    column_a: str
    table_b: str
    column_b: str
    jaccard_similarity: float
    intersection_size: int
    union_size: int
    sample_overlapping_values: List[Any]

    @property
    def is_strong_match(self) -> bool:
        return self.jaccard_similarity > 0.5

    @property
    def is_moderate_match(self) -> bool:
        return 0.3 <= self.jaccard_similarity <= 0.5

    def to_dict(self) -> Dict[str, Any]:
        return {
            "table_a": self.table_a,
            "column_a": self.column_a,
            "table_b": self.table_b,
            "column_b": self.column_b,
            "jaccard_similarity": self.jaccard_similarity,
            "intersection_size": self.intersection_size,
            "union_size": self.union_size,
            "sample_overlapping_values": self.sample_overlapping_values[:10],
            "is_strong_match": self.is_strong_match,
            "is_moderate_match": self.is_moderate_match,
        }


@dataclass
class ValueOverlap:
    """값 겹침 분석 결과"""
    table_a: str
    column_a: str
    table_b: str
    column_b: str

    # 메트릭
    jaccard_similarity: float
    overlap_ratio_a: float  # A의 값 중 B에도 있는 비율
    overlap_ratio_b: float  # B의 값 중 A에도 있는 비율
    containment_a_in_b: float  # A가 B에 포함되는 정도
    containment_b_in_a: float  # B가 A에 포함되는 정도

    # 통계
    unique_count_a: int
    unique_count_b: int
    intersection_count: int

    # 샘플
    sample_overlapping: List[Any] = field(default_factory=list)
    sample_only_a: List[Any] = field(default_factory=list)
    sample_only_b: List[Any] = field(default_factory=list)

    # 추론
    relationship_type: str = "none"  # one_to_one, one_to_many, many_to_many, subset
    confidence: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "table_a": self.table_a,
            "column_a": self.column_a,
            "table_b": self.table_b,
            "column_b": self.column_b,
            "jaccard_similarity": round(self.jaccard_similarity, 4),
            "overlap_ratio_a": round(self.overlap_ratio_a, 4),
            "overlap_ratio_b": round(self.overlap_ratio_b, 4),
            "containment_a_in_b": round(self.containment_a_in_b, 4),
            "containment_b_in_a": round(self.containment_b_in_a, 4),
            "unique_count_a": self.unique_count_a,
            "unique_count_b": self.unique_count_b,
            "intersection_count": self.intersection_count,
            "sample_overlapping_values": self.sample_overlapping[:10],
            "relationship_type": self.relationship_type,
            "confidence": round(self.confidence, 4),
        }


class SimilarityAnalyzer:
    """
    유사도 분석기

    테이블 간 컬럼 값 유사도 및 겹침 분석
    """

    def __init__(self):
        self.results: Dict[str, ValueOverlap] = {}

    def compute_jaccard(
        self,
        values_a: Set[Any],
        values_b: Set[Any],
    ) -> Tuple[float, int, int]:
        """
        Jaccard 유사도 계산

        J(A, B) = |A ∩ B| / |A ∪ B|

        Returns:
            (jaccard_similarity, intersection_size, union_size)
        """
        if not values_a and not values_b:
            return 0.0, 0, 0

        intersection = values_a & values_b
        union = values_a | values_b

        intersection_size = len(intersection)
        union_size = len(union)

        if union_size == 0:
            return 0.0, 0, 0

        return intersection_size / union_size, intersection_size, union_size

    def analyze_value_overlap(
        self,
        table_a: str,
        column_a: str,
        values_a: List[Any],
        table_b: str,
        column_b: str,
        values_b: List[Any],
        normalize_values: bool = True,
    ) -> ValueOverlap:
        """
        두 컬럼 간 값 겹침 분석

        Args:
            table_a, column_a: 테이블 A의 컬럼
            values_a: A의 값 리스트
            table_b, column_b: 테이블 B의 컬럼
            values_b: B의 값 리스트
            normalize_values: 값 정규화 여부

        Returns:
            ValueOverlap 결과
        """
        # 값 정규화 및 Set 변환
        if normalize_values:
            set_a = self._normalize_value_set(values_a)
            set_b = self._normalize_value_set(values_b)
        else:
            set_a = set(v for v in values_a if v is not None)
            set_b = set(v for v in values_b if v is not None)

        # Jaccard 계산
        jaccard, intersection_size, union_size = self.compute_jaccard(set_a, set_b)

        # 겹침 비율
        overlap_ratio_a = intersection_size / len(set_a) if set_a else 0.0
        overlap_ratio_b = intersection_size / len(set_b) if set_b else 0.0

        # Containment (포함도)
        containment_a_in_b = intersection_size / len(set_a) if set_a else 0.0
        containment_b_in_a = intersection_size / len(set_b) if set_b else 0.0

        # 겹치는 값 샘플
        intersection = set_a & set_b
        only_a = set_a - set_b
        only_b = set_b - set_a

        sample_overlapping = list(intersection)[:10]
        sample_only_a = list(only_a)[:5]
        sample_only_b = list(only_b)[:5]

        # 관계 타입 추론
        relationship_type = self._infer_relationship_type(
            len(set_a), len(set_b), intersection_size,
            overlap_ratio_a, overlap_ratio_b
        )

        # 신뢰도 계산
        confidence = self._compute_confidence(
            jaccard, overlap_ratio_a, overlap_ratio_b,
            len(set_a), len(set_b)
        )

        result = ValueOverlap(
            table_a=table_a,
            column_a=column_a,
            table_b=table_b,
            column_b=column_b,
            jaccard_similarity=jaccard,
            overlap_ratio_a=overlap_ratio_a,
            overlap_ratio_b=overlap_ratio_b,
            containment_a_in_b=containment_a_in_b,
            containment_b_in_a=containment_b_in_a,
            unique_count_a=len(set_a),
            unique_count_b=len(set_b),
            intersection_count=intersection_size,
            sample_overlapping=sample_overlapping,
            sample_only_a=sample_only_a,
            sample_only_b=sample_only_b,
            relationship_type=relationship_type,
            confidence=confidence,
        )

        key = f"{table_a}.{column_a}:{table_b}.{column_b}"
        self.results[key] = result

        return result

    def _normalize_value_set(self, values: List[Any]) -> Set[str]:
        """값 정규화하여 Set으로 변환"""
        normalized = set()

        for v in values:
            if v is None:
                continue

            # 문자열 변환 및 정규화
            if isinstance(v, str):
                # 소문자, 공백 정규화
                norm = v.strip().lower()
                # 특수문자 정규화
                norm = re.sub(r'\s+', ' ', norm)
                normalized.add(norm)
            elif isinstance(v, (int, float)):
                # 숫자는 문자열로
                normalized.add(str(v))
            else:
                normalized.add(str(v).strip().lower())

        return normalized

    def _infer_relationship_type(
        self,
        size_a: int,
        size_b: int,
        intersection_size: int,
        overlap_a: float,
        overlap_b: float,
    ) -> str:
        """관계 타입 추론"""
        if size_a == 0 or size_b == 0:
            return "none"

        if intersection_size == 0:
            return "none"

        # A가 B의 부분집합
        if overlap_a > 0.9 and overlap_b < 0.5:
            return "subset_a_in_b"

        # B가 A의 부분집합
        if overlap_b > 0.9 and overlap_a < 0.5:
            return "subset_b_in_a"

        # 높은 양방향 겹침 = 동일 엔티티
        if overlap_a > 0.7 and overlap_b > 0.7:
            return "one_to_one"

        # 중간 수준 겹침
        if overlap_a > 0.3 or overlap_b > 0.3:
            return "many_to_many"

        return "weak_association"

    def _compute_confidence(
        self,
        jaccard: float,
        overlap_a: float,
        overlap_b: float,
        size_a: int,
        size_b: int,
        column_a: str = "",
        column_b: str = "",
    ) -> float:
        """
        신뢰도 계산 (v6.1 개선)

        개선사항:
        - 컬럼명 유사도 반영: 같은 이름이면 부스트
        - 비대칭 페널티 강화: 0.3 → 0.5
        - FK 패턴 인식: _id suffix 매칭 시 부스트
        """
        # 기본 신뢰도: Jaccard 기반
        base_confidence = jaccard

        # 샘플 크기 보정 (너무 작은 샘플은 신뢰도 감소)
        min_size = min(size_a, size_b)
        if min_size < 10:
            size_penalty = 0.5
        elif min_size < 100:
            size_penalty = 0.8
        else:
            size_penalty = 1.0

        # v6.1: 비대칭 겹침 보정 강화 (0.3 → 0.5)
        asymmetry = abs(overlap_a - overlap_b)
        overlap_balance = 1 - asymmetry * 0.5

        # v6.1: 컬럼명 유사도 부스트
        column_boost = self._compute_column_name_boost(column_a, column_b)

        confidence = base_confidence * size_penalty * overlap_balance + column_boost

        return min(1.0, max(0.0, confidence))

    def _compute_column_name_boost(self, col_a: str, col_b: str) -> float:
        """
        컬럼명 기반 신뢰도 부스트 (v6.1)

        - 완전 일치: +0.3
        - FK 패턴 매칭 (table_id 형태): +0.2
        - 부분 일치 (공통 접미사): +0.1
        """
        col_a_lower = col_a.lower().strip()
        col_b_lower = col_b.lower().strip()

        # 완전 일치
        if col_a_lower == col_b_lower:
            return 0.3

        # FK 패턴: _id로 끝나는 컬럼끼리 비교
        if col_a_lower.endswith('_id') and col_b_lower.endswith('_id'):
            # 같은 entity를 참조하는지 확인 (customer_id, cust_id 등)
            prefix_a = col_a_lower.replace('_id', '').replace('id', '')
            prefix_b = col_b_lower.replace('_id', '').replace('id', '')

            # 접두사가 유사하면 부스트
            if prefix_a and prefix_b:
                if prefix_a == prefix_b:
                    return 0.25
                elif prefix_a in prefix_b or prefix_b in prefix_a:
                    return 0.15

        # 공통 접미사 (_id, _code, _key 등)
        common_suffixes = ['_id', '_code', '_key', '_no', '_num']
        for suffix in common_suffixes:
            if col_a_lower.endswith(suffix) and col_b_lower.endswith(suffix):
                return 0.05

        return 0.0

    def find_all_overlaps(
        self,
        tables_data: Dict[str, Dict[str, List[Any]]],
        min_jaccard: float = 0.1,
        exclude_same_table: bool = True,
    ) -> List[ValueOverlap]:
        """
        모든 테이블 쌍에서 컬럼 값 겹침 탐색

        Args:
            tables_data: {table_name: {column_name: [values]}}
            min_jaccard: 최소 Jaccard 임계값
            exclude_same_table: 같은 테이블 내 비교 제외

        Returns:
            ValueOverlap 리스트
        """
        results = []
        table_names = list(tables_data.keys())

        for i, table_a in enumerate(table_names):
            for j, table_b in enumerate(table_names):
                if exclude_same_table and i >= j:
                    continue
                if not exclude_same_table and i > j:
                    continue

                columns_a = tables_data[table_a]
                columns_b = tables_data[table_b]

                for col_a, values_a in columns_a.items():
                    for col_b, values_b in columns_b.items():
                        # 같은 테이블의 같은 컬럼 스킵
                        if table_a == table_b and col_a == col_b:
                            continue

                        overlap = self.analyze_value_overlap(
                            table_a, col_a, values_a,
                            table_b, col_b, values_b,
                        )

                        if overlap.jaccard_similarity >= min_jaccard:
                            results.append(overlap)

        # Jaccard 내림차순 정렬
        results.sort(key=lambda x: x.jaccard_similarity, reverse=True)

        return results

    def find_join_keys(
        self,
        tables_data: Dict[str, Dict[str, List[Any]]],
        min_jaccard: float = 0.3,
    ) -> List[Dict[str, Any]]:
        """
        테이블 간 Join Key 후보 탐색

        Returns:
            [
                {
                    "table_a": str,
                    "column_a": str,
                    "table_b": str,
                    "column_b": str,
                    "jaccard": float,
                    "recommendation": "strong" | "moderate" | "weak",
                    "evidence": {...}
                }
            ]
        """
        overlaps = self.find_all_overlaps(tables_data, min_jaccard=min_jaccard)
        recommendations = []

        for overlap in overlaps:
            if overlap.relationship_type in ["one_to_one", "subset_a_in_b", "subset_b_in_a"]:
                strength = "strong"
            elif overlap.jaccard_similarity > 0.5:
                strength = "strong"
            elif overlap.jaccard_similarity > 0.3:
                strength = "moderate"
            else:
                strength = "weak"

            recommendations.append({
                "table_a": overlap.table_a,
                "column_a": overlap.column_a,
                "table_b": overlap.table_b,
                "column_b": overlap.column_b,
                "jaccard": round(overlap.jaccard_similarity, 4),
                "overlap_ratio_a": round(overlap.overlap_ratio_a, 4),
                "overlap_ratio_b": round(overlap.overlap_ratio_b, 4),
                "recommendation": strength,
                "relationship_type": overlap.relationship_type,
                "sample_values": overlap.sample_overlapping[:5],
                "confidence": round(overlap.confidence, 4),
            })

        return recommendations


class FuzzyMatcher:
    """
    Fuzzy String Matching

    유사하지만 정확히 같지 않은 값 매칭
    """

    def __init__(self, threshold: float = 0.8):
        self.threshold = threshold

    def levenshtein_distance(self, s1: str, s2: str) -> int:
        """Levenshtein 편집 거리"""
        if len(s1) < len(s2):
            s1, s2 = s2, s1

        if len(s2) == 0:
            return len(s1)

        previous_row = range(len(s2) + 1)

        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row

        return previous_row[-1]

    def similarity(self, s1: str, s2: str) -> float:
        """문자열 유사도 (0-1)"""
        if not s1 or not s2:
            return 0.0

        s1, s2 = s1.lower().strip(), s2.lower().strip()

        if s1 == s2:
            return 1.0

        max_len = max(len(s1), len(s2))
        distance = self.levenshtein_distance(s1, s2)

        return 1 - (distance / max_len)

    def find_matches(
        self,
        values_a: List[str],
        values_b: List[str],
    ) -> List[Tuple[str, str, float]]:
        """
        두 값 리스트 간 fuzzy 매칭

        Returns:
            [(value_a, value_b, similarity), ...]
        """
        matches = []

        for va in values_a:
            for vb in values_b:
                sim = self.similarity(str(va), str(vb))
                if sim >= self.threshold:
                    matches.append((va, vb, sim))

        # 유사도 내림차순 정렬
        matches.sort(key=lambda x: x[2], reverse=True)

        return matches
