"""
Advanced Similarity Analysis Module

대용량 데이터를 위한 고성능 유사도 분석:
1. MinHash - 대용량 집합 유사도 추정
2. LSH (Locality Sensitive Hashing) - 유사 아이템 빠른 탐색
3. HyperLogLog - 카디널리티 추정
4. SimHash - 텍스트 유사도
5. Containment Analysis - 포함 관계 분석

References:
- Broder "On the Resemblance and Containment of Documents" (1997)
- Indyk & Motwani "Approximate Nearest Neighbors" (1998)
- Flajolet et al. "HyperLogLog: Analysis of a Near-Optimal Cardinality Estimation Algorithm" (2007)
- Charikar "Similarity Estimation Techniques from Rounding Algorithms" (2002)
"""

import hashlib
import logging
import struct
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Set, Tuple, Iterator
from collections import defaultdict
import math
import random

logger = logging.getLogger(__name__)

# Constants
LARGE_PRIME = 2**61 - 1
MAX_HASH = 2**32 - 1


@dataclass
class SimilarityResult:
    """유사도 분석 결과"""
    table_a: str
    column_a: str
    table_b: str
    column_b: str

    # 유사도 메트릭
    jaccard_estimate: float = 0.0
    containment_a_in_b: float = 0.0  # A ⊆ B
    containment_b_in_a: float = 0.0  # B ⊆ A

    # 카디널리티
    cardinality_a: int = 0
    cardinality_b: int = 0
    intersection_estimate: int = 0

    # 분석 메타
    sample_matches: List[Any] = field(default_factory=list)
    relationship_type: str = "unknown"  # one_to_one, one_to_many, many_to_one, many_to_many, subset
    confidence: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "table_a": self.table_a,
            "column_a": self.column_a,
            "table_b": self.table_b,
            "column_b": self.column_b,
            "jaccard_estimate": round(self.jaccard_estimate, 4),
            "containment_a_in_b": round(self.containment_a_in_b, 4),
            "containment_b_in_a": round(self.containment_b_in_a, 4),
            "cardinality_a": self.cardinality_a,
            "cardinality_b": self.cardinality_b,
            "intersection_estimate": self.intersection_estimate,
            "sample_matches": self.sample_matches[:10],
            "relationship_type": self.relationship_type,
            "confidence": round(self.confidence, 4),
        }


class MinHash:
    """
    MinHash 알고리즘

    대용량 집합의 Jaccard 유사도를 O(1) 공간에서 추정
    """

    def __init__(self, num_perm: int = 128, seed: int = 42):
        """
        Args:
            num_perm: 해시 함수 수 (정확도 vs 메모리 트레이드오프)
            seed: 랜덤 시드
        """
        self.num_perm = num_perm
        self.seed = seed

        # 해시 함수 파라미터 생성
        random.seed(seed)
        self._hash_params = [
            (random.randint(1, LARGE_PRIME - 1), random.randint(0, LARGE_PRIME - 1))
            for _ in range(num_perm)
        ]

    def create_signature(self, values: Set[str]) -> List[int]:
        """
        집합의 MinHash 시그니처 생성

        Args:
            values: 문자열 집합

        Returns:
            MinHash 시그니처 (길이 = num_perm)
        """
        signature = [MAX_HASH] * self.num_perm

        for val in values:
            # 값을 해시
            h = self._hash_value(val)

            # 각 해시 함수에 대해 최솟값 업데이트
            for i, (a, b) in enumerate(self._hash_params):
                minhash = (a * h + b) % LARGE_PRIME % MAX_HASH
                signature[i] = min(signature[i], minhash)

        return signature

    def estimate_jaccard(self, sig_a: List[int], sig_b: List[int]) -> float:
        """
        두 시그니처의 Jaccard 유사도 추정

        Args:
            sig_a: 집합 A의 MinHash 시그니처
            sig_b: 집합 B의 MinHash 시그니처

        Returns:
            추정 Jaccard 유사도
        """
        if len(sig_a) != len(sig_b):
            raise ValueError("Signature lengths must match")

        matches = sum(1 for a, b in zip(sig_a, sig_b) if a == b)
        return matches / len(sig_a)

    def _hash_value(self, val: str) -> int:
        """값을 정수로 해시"""
        return int(hashlib.sha256(val.encode()).hexdigest()[:8], 16)


class LSH:
    """
    Locality Sensitive Hashing

    유사한 아이템을 빠르게 찾기 위한 인덱싱 구조
    """

    def __init__(self, num_perm: int = 128, num_bands: int = 16, seed: int = 42):
        """
        Args:
            num_perm: MinHash 시그니처 길이
            num_bands: 밴드 수 (더 많으면 false positive 증가, false negative 감소)
            seed: 랜덤 시드
        """
        self.num_perm = num_perm
        self.num_bands = num_bands
        self.rows_per_band = num_perm // num_bands
        self.seed = seed

        # MinHash 인스턴스
        self.minhash = MinHash(num_perm, seed)

        # LSH 버킷
        self.buckets: Dict[int, Dict[int, Set[str]]] = {
            i: defaultdict(set) for i in range(num_bands)
        }

        # 시그니처 저장
        self.signatures: Dict[str, List[int]] = {}

    def add(self, item_id: str, values: Set[str]):
        """
        아이템을 LSH에 추가

        Args:
            item_id: 아이템 식별자
            values: 아이템의 값 집합
        """
        signature = self.minhash.create_signature(values)
        self.signatures[item_id] = signature

        # 각 밴드에 대해 버킷에 추가
        for band_idx in range(self.num_bands):
            start = band_idx * self.rows_per_band
            end = start + self.rows_per_band
            band = tuple(signature[start:end])
            bucket_hash = hash(band)
            self.buckets[band_idx][bucket_hash].add(item_id)

    def query(self, values: Set[str], threshold: float = 0.5) -> List[Tuple[str, float]]:
        """
        유사한 아이템 찾기

        Args:
            values: 쿼리 값 집합
            threshold: 최소 유사도 임계값

        Returns:
            [(item_id, similarity), ...]
        """
        query_sig = self.minhash.create_signature(values)
        candidates = set()

        # 후보 수집
        for band_idx in range(self.num_bands):
            start = band_idx * self.rows_per_band
            end = start + self.rows_per_band
            band = tuple(query_sig[start:end])
            bucket_hash = hash(band)

            if bucket_hash in self.buckets[band_idx]:
                candidates.update(self.buckets[band_idx][bucket_hash])

        # 후보 필터링
        results = []
        for cand_id in candidates:
            if cand_id in self.signatures:
                similarity = self.minhash.estimate_jaccard(query_sig, self.signatures[cand_id])
                if similarity >= threshold:
                    results.append((cand_id, similarity))

        results.sort(key=lambda x: x[1], reverse=True)
        return results

    def find_similar_pairs(self, threshold: float = 0.5) -> List[Tuple[str, str, float]]:
        """
        모든 유사 쌍 찾기

        Args:
            threshold: 최소 유사도 임계값

        Returns:
            [(item_a, item_b, similarity), ...]
        """
        # 후보 쌍 수집
        candidate_pairs = set()

        for band_idx in range(self.num_bands):
            for bucket_items in self.buckets[band_idx].values():
                if len(bucket_items) > 1:
                    items = list(bucket_items)
                    for i in range(len(items)):
                        for j in range(i + 1, len(items)):
                            candidate_pairs.add((min(items[i], items[j]), max(items[i], items[j])))

        # 후보 검증
        results = []
        for item_a, item_b in candidate_pairs:
            if item_a in self.signatures and item_b in self.signatures:
                similarity = self.minhash.estimate_jaccard(
                    self.signatures[item_a],
                    self.signatures[item_b]
                )
                if similarity >= threshold:
                    results.append((item_a, item_b, similarity))

        results.sort(key=lambda x: x[2], reverse=True)
        return results


class HyperLogLog:
    """
    HyperLogLog 알고리즘

    대용량 집합의 카디널리티(unique count)를 O(1) 공간에서 추정
    """

    def __init__(self, precision: int = 14):
        """
        Args:
            precision: 정밀도 (4-18, 높을수록 정확하지만 메모리 증가)
        """
        self.precision = precision
        self.m = 1 << precision  # 레지스터 수
        self.alpha = self._get_alpha(self.m)
        self.registers = [0] * self.m

    def add(self, value: str):
        """값 추가"""
        h = int(hashlib.sha256(value.encode()).hexdigest()[:16], 16)

        # 레지스터 인덱스
        idx = h >> (64 - self.precision)

        # 나머지 비트에서 첫 번째 1의 위치
        remaining = h & ((1 << (64 - self.precision)) - 1)
        rank = self._count_leading_zeros(remaining) + 1

        self.registers[idx] = max(self.registers[idx], rank)

    def estimate(self) -> int:
        """카디널리티 추정"""
        # 조화 평균
        indicator = sum(2**(-r) for r in self.registers)
        estimate = self.alpha * self.m * self.m / indicator

        # 작은 범위 보정
        if estimate <= 2.5 * self.m:
            zeros = self.registers.count(0)
            if zeros > 0:
                estimate = self.m * math.log(self.m / zeros)

        # 큰 범위 보정
        elif estimate > (1 << 32) / 30:
            estimate = -(1 << 32) * math.log(1 - estimate / (1 << 32))

        return int(estimate)

    def merge(self, other: "HyperLogLog") -> "HyperLogLog":
        """두 HLL 병합"""
        if self.precision != other.precision:
            raise ValueError("Precisions must match")

        result = HyperLogLog(self.precision)
        result.registers = [max(a, b) for a, b in zip(self.registers, other.registers)]
        return result

    def _get_alpha(self, m: int) -> float:
        """알파 상수 계산"""
        if m == 16:
            return 0.673
        elif m == 32:
            return 0.697
        elif m == 64:
            return 0.709
        else:
            return 0.7213 / (1 + 1.079 / m)

    def _count_leading_zeros(self, x: int) -> int:
        """Leading zeros 카운트"""
        if x == 0:
            return 64 - self.precision
        n = 0
        while x & (1 << 63) == 0:
            n += 1
            x <<= 1
        return n


class AdvancedSimilarityAnalyzer:
    """
    고급 유사도 분석기

    MinHash, LSH, HyperLogLog를 활용한 대용량 데이터 분석
    """

    def __init__(
        self,
        num_perm: int = 128,
        num_bands: int = 16,
        hll_precision: int = 14,
    ):
        self.num_perm = num_perm
        self.num_bands = num_bands
        self.hll_precision = hll_precision

        self.minhash = MinHash(num_perm)
        self.lsh = LSH(num_perm, num_bands)

    def analyze_column_similarity(
        self,
        tables_data: Dict[str, Dict[str, List[Any]]],
        min_jaccard: float = 0.1,
        sample_size: int = 10000,
    ) -> List[SimilarityResult]:
        """
        모든 테이블 쌍의 컬럼 유사도 분석

        Args:
            tables_data: {table_name: {column_name: [values]}}
            min_jaccard: 최소 Jaccard 유사도
            sample_size: 샘플 크기 (대용량 데이터 처리)

        Returns:
            유사도 결과 리스트
        """
        results = []

        # 컬럼별 데이터 수집 및 LSH 인덱싱
        column_index = {}  # (table, column) -> values_set

        for table_name, columns in tables_data.items():
            for col_name, values in columns.items():
                # 값을 문자열로 정규화
                normalized = set()
                for v in values[:sample_size]:
                    if v is not None:
                        normalized.add(str(v).strip().lower())

                if len(normalized) < 2:
                    continue

                key = f"{table_name}.{col_name}"
                column_index[key] = normalized

                # LSH에 추가
                self.lsh.add(key, normalized)

        # 유사 쌍 찾기
        similar_pairs = self.lsh.find_similar_pairs(threshold=min_jaccard)

        for key_a, key_b, jaccard in similar_pairs:
            table_a, col_a = key_a.rsplit(".", 1)
            table_b, col_b = key_b.rsplit(".", 1)

            # 같은 테이블 스킵
            if table_a == table_b:
                continue

            values_a = column_index[key_a]
            values_b = column_index[key_b]

            # 정밀 분석
            result = self._compute_detailed_similarity(
                table_a, col_a, values_a,
                table_b, col_b, values_b,
            )
            result.jaccard_estimate = jaccard

            results.append(result)

        # Jaccard 내림차순 정렬
        results.sort(key=lambda x: x.jaccard_estimate, reverse=True)

        return results

    def _compute_detailed_similarity(
        self,
        table_a: str,
        col_a: str,
        values_a: Set[str],
        table_b: str,
        col_b: str,
        values_b: Set[str],
    ) -> SimilarityResult:
        """상세 유사도 계산"""

        # 정확한 집합 연산
        intersection = values_a & values_b
        union = values_a | values_b

        # 메트릭 계산
        jaccard = len(intersection) / len(union) if union else 0.0
        containment_a_in_b = len(intersection) / len(values_a) if values_a else 0.0
        containment_b_in_a = len(intersection) / len(values_b) if values_b else 0.0

        # 관계 타입 추론
        relationship_type = self._infer_relationship_type(
            len(values_a), len(values_b),
            containment_a_in_b, containment_b_in_a
        )

        # 신뢰도 계산
        confidence = self._compute_confidence(
            jaccard, containment_a_in_b, containment_b_in_a,
            len(values_a), len(values_b)
        )

        return SimilarityResult(
            table_a=table_a,
            column_a=col_a,
            table_b=table_b,
            column_b=col_b,
            jaccard_estimate=jaccard,
            containment_a_in_b=containment_a_in_b,
            containment_b_in_a=containment_b_in_a,
            cardinality_a=len(values_a),
            cardinality_b=len(values_b),
            intersection_estimate=len(intersection),
            sample_matches=list(intersection)[:10],
            relationship_type=relationship_type,
            confidence=confidence,
        )

    def _infer_relationship_type(
        self,
        size_a: int,
        size_b: int,
        containment_a_in_b: float,
        containment_b_in_a: float,
    ) -> str:
        """관계 타입 추론"""

        if containment_a_in_b >= 0.95 and containment_b_in_a >= 0.95:
            return "one_to_one"

        if containment_a_in_b >= 0.9 and containment_b_in_a < 0.5:
            return "many_to_one"  # A의 값들이 B의 subset

        if containment_b_in_a >= 0.9 and containment_a_in_b < 0.5:
            return "one_to_many"  # B의 값들이 A의 subset

        if containment_a_in_b >= 0.7 or containment_b_in_a >= 0.7:
            return "subset"

        return "many_to_many"

    def _compute_confidence(
        self,
        jaccard: float,
        containment_a_in_b: float,
        containment_b_in_a: float,
        size_a: int,
        size_b: int,
    ) -> float:
        """신뢰도 계산"""

        # 기본 점수
        base = max(jaccard, containment_a_in_b, containment_b_in_a)

        # 샘플 크기 보정
        min_size = min(size_a, size_b)
        if min_size < 10:
            size_factor = 0.5
        elif min_size < 100:
            size_factor = 0.8
        else:
            size_factor = 1.0

        return min(1.0, base * size_factor)

    def estimate_cardinality(self, values: Iterator[str]) -> int:
        """HyperLogLog를 사용한 카디널리티 추정"""
        hll = HyperLogLog(self.hll_precision)
        for v in values:
            hll.add(str(v))
        return hll.estimate()

    # =========================================================================
    # 편의 메서드: discovery.py 에이전트에서 사용하는 API
    # =========================================================================

    def add_minhash(self, col_key: str, values: Set[str]) -> None:
        """
        MinHash 시그니처 생성 및 LSH에 추가

        Args:
            col_key: 컬럼 키 (예: "table_name.column_name")
            values: 값 집합 (문자열 set)
        """
        if len(values) < 2:
            return
        # LSH에 추가 (내부적으로 MinHash 시그니처 생성)
        self.lsh.add(col_key, values)

    def query_similar_all(self, threshold: float = None) -> List[Tuple[str, str, float]]:
        """
        LSH를 사용하여 모든 유사 쌍 찾기

        Args:
            threshold: Jaccard 유사도 임계값 (None이면 초기화 시 설정된 threshold 사용)

        Returns:
            [(col_key_a, col_key_b, jaccard_similarity), ...]
        """
        if threshold is None:
            threshold = getattr(self.lsh, 'threshold', 0.5)  # v4.2: LSH에 threshold가 없으면 기본값 사용
        return self.lsh.find_similar_pairs(threshold)

    def add_to_hyperloglog(self, col_key: str, values: List[str]) -> None:
        """
        HyperLogLog에 값 추가 (카디널리티 추정용)

        Args:
            col_key: 컬럼 키
            values: 값 리스트
        """
        # HLL 인스턴스 관리를 위한 딕셔너리 (없으면 생성)
        if not hasattr(self, '_hll_instances'):
            self._hll_instances: Dict[str, HyperLogLog] = {}

        if col_key not in self._hll_instances:
            self._hll_instances[col_key] = HyperLogLog(self.hll_precision)

        hll = self._hll_instances[col_key]
        for v in values:
            hll.add(str(v))

    def estimate_cardinality_for_column(self, col_key: str) -> int:
        """
        특정 컬럼의 카디널리티 추정

        Args:
            col_key: 컬럼 키

        Returns:
            추정 카디널리티 (컬럼이 없으면 0)
        """
        if not hasattr(self, '_hll_instances'):
            return 0
        if col_key not in self._hll_instances:
            return 0
        return self._hll_instances[col_key].estimate()
