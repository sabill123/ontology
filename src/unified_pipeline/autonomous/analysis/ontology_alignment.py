"""
Ontology Alignment Module - 온톨로지 정렬 및 매칭

서로 다른 스키마/온톨로지 간의 의미적 정렬을 수행합니다.

주요 기능:
- 스키마 매칭 (Concept, Property, Instance)
- 유사도 기반 정렬 (문자열, 구조, 의미)
- LLM 기반 의미 해석
- 정렬 품질 평가
- 충돌 해결

References:
- Euzenat & Shvaiko, "Ontology Matching", 2nd ed., Springer 2013
- OAEI (Ontology Alignment Evaluation Initiative)
- He et al., "LLMs for Ontology Matching: A Preliminary Study", 2024
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Any
from enum import Enum
import numpy as np
from collections import defaultdict
import re


class AlignmentType(Enum):
    """정렬 타입"""
    EQUIVALENT = "equivalent"           # 동등 (=)
    SUBSUMES = "subsumes"              # 상위 개념 (⊇)
    SUBSUMED_BY = "subsumed_by"        # 하위 개념 (⊆)
    DISJOINT = "disjoint"              # 상호 배제 (⊥)
    RELATED = "related"                # 관련 (~)
    OVERLAP = "overlap"                # 겹침 (∩)


class MatcherType(Enum):
    """매처 타입"""
    STRING = "string"                  # 문자열 기반
    STRUCTURAL = "structural"          # 구조 기반
    SEMANTIC = "semantic"              # 의미 기반
    INSTANCE = "instance"              # 인스턴스 기반
    HYBRID = "hybrid"                  # 하이브리드


@dataclass
class OntologyElement:
    """온톨로지 요소 (개념, 속성, 인스턴스)"""
    uri: str
    name: str
    element_type: str                  # concept, property, instance
    domain: Optional[str] = None
    range: Optional[str] = None
    parents: List[str] = field(default_factory=list)
    children: List[str] = field(default_factory=list)
    properties: Dict[str, Any] = field(default_factory=dict)
    instances: List[str] = field(default_factory=list)
    labels: List[str] = field(default_factory=list)
    comments: List[str] = field(default_factory=list)


@dataclass
class Alignment:
    """정렬 결과"""
    source: str                        # 소스 URI
    target: str                        # 타겟 URI
    alignment_type: AlignmentType
    confidence: float
    method: MatcherType
    evidence: List[str] = field(default_factory=list)


@dataclass
class AlignmentSet:
    """정렬 집합"""
    source_ontology: str
    target_ontology: str
    alignments: List[Alignment] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MatchingMetrics:
    """매칭 품질 지표"""
    precision: float
    recall: float
    f1_score: float
    accuracy: float
    num_alignments: int
    avg_confidence: float


class StringMatcher:
    """문자열 기반 매처"""

    def __init__(self):
        self.stop_words = {
            'the', 'a', 'an', 'of', 'to', 'in', 'for', 'on', 'with', 'is', 'are',
            'has', 'have', 'id', 'type', 'name', 'value', 'data', 'info'
        }

    def compute_similarity(self, s1: str, s2: str) -> float:
        """문자열 유사도 계산 (여러 방법 조합)"""
        # 정규화
        s1_norm = self._normalize(s1)
        s2_norm = self._normalize(s2)

        if s1_norm == s2_norm:
            return 1.0

        # 여러 유사도 계산
        jaro = self._jaro_winkler(s1_norm, s2_norm)
        jaccard = self._jaccard_tokens(s1_norm, s2_norm)
        edit = self._normalized_edit_distance(s1_norm, s2_norm)
        prefix = self._common_prefix_ratio(s1_norm, s2_norm)

        # 가중 평균
        return 0.35 * jaro + 0.25 * jaccard + 0.25 * (1 - edit) + 0.15 * prefix

    def _normalize(self, s: str) -> str:
        """문자열 정규화"""
        # 카멜케이스/스네이크케이스 분리
        s = re.sub(r'([a-z])([A-Z])', r'\1 \2', s)
        s = s.replace('_', ' ').replace('-', ' ')
        # 소문자화 및 불용어 제거
        tokens = [t.lower() for t in s.split() if t.lower() not in self.stop_words]
        return ' '.join(tokens)

    def _jaro_winkler(self, s1: str, s2: str) -> float:
        """Jaro-Winkler 유사도"""
        if not s1 or not s2:
            return 0.0
        if s1 == s2:
            return 1.0

        len1, len2 = len(s1), len(s2)
        match_distance = max(len1, len2) // 2 - 1

        s1_matches = [False] * len1
        s2_matches = [False] * len2

        matches = 0
        transpositions = 0

        for i in range(len1):
            start = max(0, i - match_distance)
            end = min(i + match_distance + 1, len2)

            for j in range(start, end):
                if s2_matches[j] or s1[i] != s2[j]:
                    continue
                s1_matches[i] = True
                s2_matches[j] = True
                matches += 1
                break

        if matches == 0:
            return 0.0

        k = 0
        for i in range(len1):
            if not s1_matches[i]:
                continue
            while not s2_matches[k]:
                k += 1
            if s1[i] != s2[k]:
                transpositions += 1
            k += 1

        jaro = (matches / len1 + matches / len2 +
                (matches - transpositions / 2) / matches) / 3

        # Winkler 보정
        prefix = 0
        for i in range(min(4, len1, len2)):
            if s1[i] == s2[i]:
                prefix += 1
            else:
                break

        return jaro + prefix * 0.1 * (1 - jaro)

    def _jaccard_tokens(self, s1: str, s2: str) -> float:
        """토큰 Jaccard 유사도"""
        tokens1 = set(s1.split())
        tokens2 = set(s2.split())

        if not tokens1 and not tokens2:
            return 1.0
        if not tokens1 or not tokens2:
            return 0.0

        intersection = len(tokens1 & tokens2)
        union = len(tokens1 | tokens2)

        return intersection / union if union > 0 else 0.0

    def _normalized_edit_distance(self, s1: str, s2: str) -> float:
        """정규화된 편집 거리"""
        len1, len2 = len(s1), len(s2)
        if len1 == 0 and len2 == 0:
            return 0.0
        if len1 == 0 or len2 == 0:
            return 1.0

        # DP로 편집 거리 계산
        dp = [[0] * (len2 + 1) for _ in range(len1 + 1)]

        for i in range(len1 + 1):
            dp[i][0] = i
        for j in range(len2 + 1):
            dp[0][j] = j

        for i in range(1, len1 + 1):
            for j in range(1, len2 + 1):
                cost = 0 if s1[i-1] == s2[j-1] else 1
                dp[i][j] = min(
                    dp[i-1][j] + 1,      # 삭제
                    dp[i][j-1] + 1,      # 삽입
                    dp[i-1][j-1] + cost  # 대체
                )

        return dp[len1][len2] / max(len1, len2)

    def _common_prefix_ratio(self, s1: str, s2: str) -> float:
        """공통 접두사 비율"""
        min_len = min(len(s1), len(s2))
        if min_len == 0:
            return 0.0

        prefix_len = 0
        for i in range(min_len):
            if s1[i] == s2[i]:
                prefix_len += 1
            else:
                break

        return prefix_len / min_len


class StructuralMatcher:
    """구조 기반 매처"""

    def compute_similarity(
        self,
        elem1: OntologyElement,
        elem2: OntologyElement,
        parent_sim: float = 0.0,
        sibling_sim: float = 0.0
    ) -> float:
        """구조적 유사도 계산"""
        scores = []

        # 요소 타입 일치
        if elem1.element_type == elem2.element_type:
            scores.append(1.0)
        else:
            scores.append(0.3)

        # 부모-자식 구조 유사도
        if elem1.parents and elem2.parents:
            parent_overlap = len(set(elem1.parents) & set(elem2.parents))
            parent_union = len(set(elem1.parents) | set(elem2.parents))
            scores.append(parent_overlap / parent_union if parent_union > 0 else 0)

        if elem1.children and elem2.children:
            child_overlap = len(set(elem1.children) & set(elem2.children))
            child_union = len(set(elem1.children) | set(elem2.children))
            scores.append(child_overlap / child_union if child_union > 0 else 0)

        # 속성 구조 유사도
        if elem1.properties and elem2.properties:
            prop_overlap = len(set(elem1.properties.keys()) & set(elem2.properties.keys()))
            prop_union = len(set(elem1.properties.keys()) | set(elem2.properties.keys()))
            scores.append(prop_overlap / prop_union if prop_union > 0 else 0)

        # 상위 정렬 전파
        if parent_sim > 0:
            scores.append(parent_sim * 0.3)

        # 형제 정렬 전파
        if sibling_sim > 0:
            scores.append(sibling_sim * 0.2)

        return np.mean(scores) if scores else 0.0


class SemanticMatcher:
    """의미 기반 매처"""

    def __init__(self):
        # 동의어/이의어 사전
        self.synonyms = {
            'customer': {'client', 'buyer', 'consumer', 'patron'},
            'product': {'item', 'goods', 'merchandise', 'article'},
            'order': {'purchase', 'transaction', 'buy'},
            'employee': {'staff', 'worker', 'personnel'},
            'price': {'cost', 'amount', 'value', 'rate'},
            'date': {'time', 'datetime', 'timestamp'},
            'id': {'identifier', 'key', 'code', 'number'},
            'name': {'title', 'label', 'designation'},
            'address': {'location', 'place'},
            'phone': {'telephone', 'mobile', 'contact'},
            'email': {'mail', 'e-mail'},
            'quantity': {'amount', 'count', 'number'},
            'patient': {'client', 'customer'},  # 의료 도메인
            'doctor': {'physician', 'provider'},
            'diagnosis': {'condition', 'disease'},
            'prescription': {'medication', 'drug', 'medicine'},
            'hospital': {'clinic', 'facility', 'center'},
        }

        # 도메인 용어 확장
        self.domain_mappings = {
            'healthcare': {
                'pt': 'patient',
                'dx': 'diagnosis',
                'rx': 'prescription',
                'dr': 'doctor',
                'hosp': 'hospital',
                'med': 'medication',
            },
            'finance': {
                'acct': 'account',
                'txn': 'transaction',
                'amt': 'amount',
                'cust': 'customer',
            },
            'retail': {
                'sku': 'product',
                'inv': 'inventory',
                'po': 'purchase_order',
            }
        }

    def compute_similarity(
        self,
        elem1: OntologyElement,
        elem2: OntologyElement,
        domain: str = None
    ) -> Tuple[float, List[str]]:
        """의미적 유사도 계산"""
        evidence = []

        # 이름 유사도
        name_sim = self._semantic_name_similarity(elem1.name, elem2.name, domain)
        if name_sim > 0.8:
            evidence.append(f"이름 유사도 높음: {elem1.name} ~ {elem2.name}")

        # 레이블 유사도
        label_sim = 0.0
        if elem1.labels and elem2.labels:
            for l1 in elem1.labels:
                for l2 in elem2.labels:
                    sim = self._semantic_name_similarity(l1, l2, domain)
                    if sim > label_sim:
                        label_sim = sim
            if label_sim > 0.7:
                evidence.append(f"레이블 유사도: {max(elem1.labels)} ~ {max(elem2.labels)}")

        # 주석 유사도 (간단한 키워드 매칭)
        comment_sim = 0.0
        if elem1.comments and elem2.comments:
            comment_sim = self._comment_similarity(elem1.comments, elem2.comments)
            if comment_sim > 0.5:
                evidence.append("주석에서 공통 키워드 발견")

        # 도메인/범위 유사도
        domain_range_sim = 0.0
        if elem1.domain and elem2.domain:
            domain_range_sim += self._semantic_name_similarity(
                elem1.domain, elem2.domain, domain
            ) * 0.5
        if elem1.range and elem2.range:
            domain_range_sim += self._semantic_name_similarity(
                elem1.range, elem2.range, domain
            ) * 0.5
        if domain_range_sim > 0.5:
            evidence.append("도메인/범위 유사")

        # 가중 평균
        weights = [0.5, 0.2, 0.15, 0.15]
        scores = [name_sim, label_sim, comment_sim, domain_range_sim]
        total_sim = sum(w * s for w, s in zip(weights, scores))

        return total_sim, evidence

    def _semantic_name_similarity(
        self,
        name1: str,
        name2: str,
        domain: str = None
    ) -> float:
        """이름의 의미적 유사도"""
        # 정규화
        n1 = self._normalize_name(name1, domain)
        n2 = self._normalize_name(name2, domain)

        if n1 == n2:
            return 1.0

        # 동의어 체크
        n1_syns = self._get_synonyms(n1)
        n2_syns = self._get_synonyms(n2)

        if n1 in n2_syns or n2 in n1_syns:
            return 0.95

        if n1_syns & n2_syns:
            return 0.85

        # 부분 매칭
        tokens1 = set(n1.split())
        tokens2 = set(n2.split())

        if tokens1 & tokens2:
            return 0.7 * len(tokens1 & tokens2) / len(tokens1 | tokens2)

        return 0.0

    def _normalize_name(self, name: str, domain: str = None) -> str:
        """이름 정규화 (도메인 약어 확장)"""
        name = name.lower()
        name = re.sub(r'([a-z])([A-Z])', r'\1_\2', name)
        name = name.replace('-', '_')

        # 도메인별 약어 확장
        if domain and domain in self.domain_mappings:
            for abbr, full in self.domain_mappings[domain].items():
                name = re.sub(rf'\b{abbr}\b', full, name)

        return name

    def _get_synonyms(self, word: str) -> Set[str]:
        """동의어 집합 반환"""
        synonyms = {word}
        for base, syns in self.synonyms.items():
            if word == base:
                synonyms.update(syns)
            elif word in syns:
                synonyms.add(base)
                synonyms.update(syns)
        return synonyms

    def _comment_similarity(
        self,
        comments1: List[str],
        comments2: List[str]
    ) -> float:
        """주석 유사도 (키워드 기반)"""
        words1 = set()
        words2 = set()

        for c in comments1:
            words1.update(c.lower().split())
        for c in comments2:
            words2.update(c.lower().split())

        # 불용어 제거
        stop_words = {'the', 'a', 'an', 'is', 'are', 'of', 'to', 'and', 'or'}
        words1 = words1 - stop_words
        words2 = words2 - stop_words

        if not words1 or not words2:
            return 0.0

        intersection = len(words1 & words2)
        union = len(words1 | words2)

        return intersection / union if union > 0 else 0.0


class InstanceMatcher:
    """인스턴스 기반 매처"""

    def compute_similarity(
        self,
        instances1: List[Any],
        instances2: List[Any],
        sample_size: int = 100
    ) -> Tuple[float, Dict[str, Any]]:
        """인스턴스 데이터 유사도 계산"""
        if not instances1 or not instances2:
            return 0.0, {"reason": "인스턴스 없음"}

        # 샘플링
        sample1 = instances1[:sample_size] if len(instances1) > sample_size else instances1
        sample2 = instances2[:sample_size] if len(instances2) > sample_size else instances2

        # 값 집합 유사도
        set1 = set(str(v) for v in sample1 if v is not None)
        set2 = set(str(v) for v in sample2 if v is not None)

        if not set1 or not set2:
            return 0.0, {"reason": "유효 값 없음"}

        # Jaccard 유사도
        jaccard = len(set1 & set2) / len(set1 | set2)

        # 포함 관계
        containment1 = len(set1 & set2) / len(set1) if set1 else 0
        containment2 = len(set1 & set2) / len(set2) if set2 else 0

        # 값 패턴 유사도
        pattern_sim = self._pattern_similarity(sample1, sample2)

        # 통계적 유사도 (수치 데이터인 경우)
        stat_sim = self._statistical_similarity(sample1, sample2)

        # 종합
        overall = 0.4 * jaccard + 0.2 * max(containment1, containment2) + \
                  0.2 * pattern_sim + 0.2 * stat_sim

        return overall, {
            "jaccard": jaccard,
            "containment": (containment1, containment2),
            "pattern_similarity": pattern_sim,
            "statistical_similarity": stat_sim
        }

    def _pattern_similarity(
        self,
        values1: List[Any],
        values2: List[Any]
    ) -> float:
        """값 패턴 유사도"""
        patterns1 = self._extract_patterns(values1)
        patterns2 = self._extract_patterns(values2)

        if not patterns1 or not patterns2:
            return 0.0

        common = set(patterns1.keys()) & set(patterns2.keys())
        if not common:
            return 0.0

        # 공통 패턴의 비율 유사도
        sim = 0.0
        for pattern in common:
            ratio1 = patterns1[pattern]
            ratio2 = patterns2[pattern]
            sim += 1 - abs(ratio1 - ratio2)

        return sim / len(common)

    def _extract_patterns(self, values: List[Any]) -> Dict[str, float]:
        """값 패턴 추출"""
        patterns = defaultdict(int)
        total = 0

        for v in values:
            if v is None:
                patterns['null'] += 1
            elif isinstance(v, bool):
                patterns['boolean'] += 1
            elif isinstance(v, (int, float)):
                patterns['numeric'] += 1
            else:
                s = str(v)
                if re.match(r'^\d{4}-\d{2}-\d{2}', s):
                    patterns['date'] += 1
                elif re.match(r'^[\w.-]+@[\w.-]+\.\w+$', s):
                    patterns['email'] += 1
                elif re.match(r'^\+?[\d\s-]+$', s):
                    patterns['phone'] += 1
                elif re.match(r'^[A-Z]{2,3}\d+$', s):
                    patterns['code'] += 1
                else:
                    patterns['string'] += 1
            total += 1

        return {k: v / total for k, v in patterns.items()} if total > 0 else {}

    def _statistical_similarity(
        self,
        values1: List[Any],
        values2: List[Any]
    ) -> float:
        """통계적 유사도 (수치 데이터)"""
        nums1 = [v for v in values1 if isinstance(v, (int, float))]
        nums2 = [v for v in values2 if isinstance(v, (int, float))]

        if len(nums1) < 5 or len(nums2) < 5:
            return 0.5  # 데이터 부족시 중립

        # 기본 통계량 비교
        mean1, std1 = np.mean(nums1), np.std(nums1)
        mean2, std2 = np.mean(nums2), np.std(nums2)

        if std1 == 0 and std2 == 0:
            return 1.0 if mean1 == mean2 else 0.0

        # 분포 유사도 (정규화된 차이)
        mean_diff = abs(mean1 - mean2) / (max(abs(mean1), abs(mean2)) + 1e-10)
        std_diff = abs(std1 - std2) / (max(std1, std2) + 1e-10)

        return max(0, 1 - 0.5 * mean_diff - 0.5 * std_diff)


class OntologyAligner:
    """통합 온톨로지 정렬기"""

    def __init__(
        self,
        string_weight: float = 0.25,
        structural_weight: float = 0.20,
        semantic_weight: float = 0.35,
        instance_weight: float = 0.20,
        threshold: float = 0.5
    ):
        self.string_matcher = StringMatcher()
        self.structural_matcher = StructuralMatcher()
        self.semantic_matcher = SemanticMatcher()
        self.instance_matcher = InstanceMatcher()

        self.string_weight = string_weight
        self.structural_weight = structural_weight
        self.semantic_weight = semantic_weight
        self.instance_weight = instance_weight
        self.threshold = threshold

    def align(
        self,
        source_elements: List[OntologyElement],
        target_elements: List[OntologyElement],
        domain: str = None
    ) -> AlignmentSet:
        """온톨로지 정렬 수행"""
        alignments = []

        # 모든 쌍에 대해 유사도 계산
        similarity_matrix = np.zeros((len(source_elements), len(target_elements)))

        for i, src in enumerate(source_elements):
            for j, tgt in enumerate(target_elements):
                sim, evidence = self._compute_combined_similarity(src, tgt, domain)
                similarity_matrix[i, j] = sim

                if sim >= self.threshold:
                    # 정렬 타입 결정
                    alignment_type = self._determine_alignment_type(src, tgt, sim)

                    alignments.append(Alignment(
                        source=src.uri,
                        target=tgt.uri,
                        alignment_type=alignment_type,
                        confidence=sim,
                        method=MatcherType.HYBRID,
                        evidence=evidence
                    ))

        # 최적 매칭 선택 (헝가리안 알고리즘 대신 그리디)
        alignments = self._select_optimal_alignments(alignments, similarity_matrix)

        return AlignmentSet(
            source_ontology="source",
            target_ontology="target",
            alignments=alignments,
            metadata={
                "num_source_elements": len(source_elements),
                "num_target_elements": len(target_elements),
                "threshold": self.threshold,
                "domain": domain
            }
        )

    def _compute_combined_similarity(
        self,
        src: OntologyElement,
        tgt: OntologyElement,
        domain: str = None
    ) -> Tuple[float, List[str]]:
        """종합 유사도 계산"""
        evidence = []

        # 문자열 유사도
        string_sim = self.string_matcher.compute_similarity(src.name, tgt.name)

        # 구조적 유사도
        structural_sim = self.structural_matcher.compute_similarity(src, tgt)

        # 의미적 유사도
        semantic_sim, sem_evidence = self.semantic_matcher.compute_similarity(src, tgt, domain)
        evidence.extend(sem_evidence)

        # 인스턴스 유사도
        instance_sim = 0.0
        if src.instances and tgt.instances:
            instance_sim, inst_details = self.instance_matcher.compute_similarity(
                src.instances, tgt.instances
            )
            if instance_sim > 0.6:
                evidence.append(f"인스턴스 유사도: {instance_sim:.2f}")

        # 가중 평균
        total_weight = self.string_weight + self.structural_weight + \
                       self.semantic_weight + self.instance_weight

        combined = (
            self.string_weight * string_sim +
            self.structural_weight * structural_sim +
            self.semantic_weight * semantic_sim +
            self.instance_weight * instance_sim
        ) / total_weight

        return combined, evidence

    def _determine_alignment_type(
        self,
        src: OntologyElement,
        tgt: OntologyElement,
        similarity: float
    ) -> AlignmentType:
        """정렬 타입 결정"""
        if similarity >= 0.9:
            return AlignmentType.EQUIVALENT

        # 계층 관계 체크
        if src.uri in tgt.parents or src.name.lower() in [p.lower() for p in tgt.parents]:
            return AlignmentType.SUBSUMES
        if tgt.uri in src.parents or tgt.name.lower() in [p.lower() for p in src.parents]:
            return AlignmentType.SUBSUMED_BY

        # 속성 겹침 체크
        if src.properties and tgt.properties:
            src_props = set(src.properties.keys())
            tgt_props = set(tgt.properties.keys())
            if src_props & tgt_props:
                return AlignmentType.OVERLAP

        return AlignmentType.RELATED

    def _select_optimal_alignments(
        self,
        alignments: List[Alignment],
        similarity_matrix: np.ndarray
    ) -> List[Alignment]:
        """최적 정렬 선택 (그리디)"""
        # 신뢰도 순 정렬
        sorted_alignments = sorted(alignments, key=lambda a: -a.confidence)

        selected = []
        used_sources = set()
        used_targets = set()

        for alignment in sorted_alignments:
            if alignment.source not in used_sources and alignment.target not in used_targets:
                selected.append(alignment)
                used_sources.add(alignment.source)
                used_targets.add(alignment.target)

        return selected


class ConflictResolver:
    """정렬 충돌 해결"""

    def resolve(
        self,
        alignments: List[Alignment]
    ) -> Tuple[List[Alignment], List[Dict]]:
        """충돌 탐지 및 해결"""
        conflicts = []
        resolved = []

        # 1:N 매핑 탐지
        source_mappings = defaultdict(list)
        target_mappings = defaultdict(list)

        for a in alignments:
            source_mappings[a.source].append(a)
            target_mappings[a.target].append(a)

        # 소스 충돌 해결
        for source, mappings in source_mappings.items():
            if len(mappings) > 1:
                # 최고 신뢰도 선택
                best = max(mappings, key=lambda x: x.confidence)
                resolved.append(best)
                conflicts.append({
                    "type": "source_conflict",
                    "source": source,
                    "options": [m.target for m in mappings],
                    "selected": best.target,
                    "confidence_diff": best.confidence - sorted(
                        [m.confidence for m in mappings], reverse=True
                    )[1] if len(mappings) > 1 else 0
                })
            else:
                resolved.append(mappings[0])

        # 타겟 충돌 체크 (이미 해결된 것 제외)
        final_resolved = []
        final_targets = set()

        for a in sorted(resolved, key=lambda x: -x.confidence):
            if a.target not in final_targets:
                final_resolved.append(a)
                final_targets.add(a.target)
            else:
                conflicts.append({
                    "type": "target_conflict",
                    "target": a.target,
                    "dropped_source": a.source
                })

        return final_resolved, conflicts


class QualityEvaluator:
    """정렬 품질 평가"""

    def evaluate(
        self,
        predicted: AlignmentSet,
        reference: AlignmentSet = None
    ) -> MatchingMetrics:
        """정렬 품질 평가"""
        if reference is None:
            # 참조 없으면 내부 지표만 계산
            return MatchingMetrics(
                precision=0.0,
                recall=0.0,
                f1_score=0.0,
                accuracy=0.0,
                num_alignments=len(predicted.alignments),
                avg_confidence=np.mean([a.confidence for a in predicted.alignments])
                    if predicted.alignments else 0.0
            )

        # 정렬 집합
        pred_set = {(a.source, a.target) for a in predicted.alignments}
        ref_set = {(a.source, a.target) for a in reference.alignments}

        # 지표 계산
        tp = len(pred_set & ref_set)
        fp = len(pred_set - ref_set)
        fn = len(ref_set - pred_set)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        # 정확도 (모든 가능한 쌍 대비)
        n_possible = predicted.metadata.get("num_source_elements", 1) * \
                     predicted.metadata.get("num_target_elements", 1)
        tn = n_possible - tp - fp - fn
        accuracy = (tp + tn) / n_possible if n_possible > 0 else 0.0

        return MatchingMetrics(
            precision=precision,
            recall=recall,
            f1_score=f1,
            accuracy=accuracy,
            num_alignments=len(predicted.alignments),
            avg_confidence=np.mean([a.confidence for a in predicted.alignments])
                if predicted.alignments else 0.0
        )


class SchemaToOntologyConverter:
    """스키마를 온톨로지 요소로 변환"""

    def convert_table(
        self,
        table_name: str,
        columns: List[Dict],
        data: List[Dict] = None
    ) -> List[OntologyElement]:
        """테이블을 온톨로지 요소로 변환"""
        elements = []

        # 테이블 -> Concept
        table_element = OntologyElement(
            uri=f"table:{table_name}",
            name=table_name,
            element_type="concept",
            labels=[table_name],
            children=[f"column:{table_name}.{self._get_col_name(c)}" for c in columns]
        )
        elements.append(table_element)

        # 컬럼 -> Property
        for col in columns:
            col_name = self._get_col_name(col)
            col_type = self._get_col_type(col)

            # 인스턴스 데이터 추출
            instances = []
            if data:
                instances = [row.get(col_name) for row in data if col_name in row]

            col_element = OntologyElement(
                uri=f"column:{table_name}.{col_name}",
                name=col_name,
                element_type="property",
                domain=table_name,
                range=col_type,
                parents=[f"table:{table_name}"],
                instances=instances,
                labels=[col_name],
                properties={
                    "data_type": col_type,
                    "nullable": col.get("nullable", True) if isinstance(col, dict) else True
                }
            )
            elements.append(col_element)

        return elements

    def _get_col_name(self, col) -> str:
        """컬럼명 추출"""
        if isinstance(col, dict):
            return col.get("name", col.get("column_name", "unknown"))
        return str(col)

    def _get_col_type(self, col) -> str:
        """컬럼 타입 추출"""
        if isinstance(col, dict):
            return col.get("type", col.get("data_type", "string"))
        return "string"


class OntologyAlignmentAnalyzer:
    """통합 온톨로지 정렬 분석기"""

    def __init__(self):
        self.converter = SchemaToOntologyConverter()
        self.aligner = OntologyAligner()
        self.conflict_resolver = ConflictResolver()
        self.evaluator = QualityEvaluator()

    def align_tables(
        self,
        tables: Dict[str, Dict],
        domain: str = None
    ) -> Dict[str, Any]:
        """테이블 간 정렬 수행"""
        # 테이블을 온톨로지 요소로 변환
        all_elements = {}
        for table_name, table_info in tables.items():
            columns = table_info.get("columns", [])
            data = table_info.get("data", [])
            elements = self.converter.convert_table(table_name, columns, data)
            all_elements[table_name] = elements

        # 테이블 쌍별 정렬
        table_names = list(tables.keys())
        all_alignments = []
        pairwise_results = []

        for i in range(len(table_names)):
            for j in range(i + 1, len(table_names)):
                src_table = table_names[i]
                tgt_table = table_names[j]

                src_elements = all_elements[src_table]
                tgt_elements = all_elements[tgt_table]

                alignment_set = self.aligner.align(
                    src_elements, tgt_elements, domain
                )

                # 충돌 해결
                resolved, conflicts = self.conflict_resolver.resolve(
                    alignment_set.alignments
                )

                all_alignments.extend(resolved)

                pairwise_results.append({
                    "source_table": src_table,
                    "target_table": tgt_table,
                    "alignments": [
                        {
                            "source": a.source,
                            "target": a.target,
                            "type": a.alignment_type.value,
                            "confidence": a.confidence,
                            "evidence": a.evidence
                        }
                        for a in resolved
                    ],
                    "conflicts": conflicts
                })

        # 전체 품질 평가
        overall_set = AlignmentSet(
            source_ontology="all_tables",
            target_ontology="all_tables",
            alignments=all_alignments,
            metadata={
                "num_source_elements": sum(len(e) for e in all_elements.values()),
                "num_target_elements": sum(len(e) for e in all_elements.values())
            }
        )
        metrics = self.evaluator.evaluate(overall_set)

        return {
            "pairwise_alignments": pairwise_results,
            "total_alignments": len(all_alignments),
            "metrics": {
                "num_alignments": metrics.num_alignments,
                "avg_confidence": metrics.avg_confidence
            },
            "summary": self._generate_summary(pairwise_results, all_alignments)
        }

    def _generate_summary(
        self,
        pairwise: List[Dict],
        alignments: List[Alignment]
    ) -> Dict[str, Any]:
        """정렬 요약 생성"""
        # 정렬 타입별 카운트
        type_counts = defaultdict(int)
        for a in alignments:
            type_counts[a.alignment_type.value] += 1

        # 고신뢰 정렬
        high_confidence = [a for a in alignments if a.confidence >= 0.8]

        # 테이블별 연결도
        table_connections = defaultdict(int)
        for result in pairwise:
            if result["alignments"]:
                table_connections[result["source_table"]] += 1
                table_connections[result["target_table"]] += 1

        return {
            "alignment_types": dict(type_counts),
            "high_confidence_count": len(high_confidence),
            "table_connectivity": dict(table_connections),
            "most_connected_table": max(table_connections.items(), key=lambda x: x[1])[0]
                if table_connections else None
        }

    # =========================================================================
    # 편의 메서드: refinement.py 에이전트에서 사용하는 API
    # =========================================================================

    def find_alignments(
        self,
        source_elements: List[Dict[str, Any]],
        domain: str = None
    ) -> List[Dict[str, Any]]:
        """
        소스 요소들 간의 정렬 찾기 (에이전트 호출용)

        Args:
            source_elements: [{"id": ..., "type": ..., "name": ..., "properties": {...}}, ...]
            domain: 도메인 이름

        Returns:
            정렬 결과 리스트
        """
        # OntologyElement 리스트로 변환
        elements = []
        for elem in source_elements:
            elements.append(OntologyElement(
                uri=elem.get("id", elem.get("uri", "")),
                name=elem.get("name", ""),
                element_type=elem.get("type", "class"),
                properties=elem.get("properties", {}),
                comments=[elem.get("description", "")] if elem.get("description") else [],
                labels=[elem.get("source", "unknown")] if elem.get("source") else [],
            ))

        # 자기 자신과 정렬 (동일 요소 그룹 찾기)
        alignments = []

        for i in range(len(elements)):
            for j in range(i + 1, len(elements)):
                elem_a = elements[i]
                elem_b = elements[j]

                # 이름 유사도 확인
                name_a = elem_a.name.lower().replace("_", " ")
                name_b = elem_b.name.lower().replace("_", " ")

                # 간단한 유사도 체크 (Jaccard on character n-grams)
                def char_ngrams(s, n=3):
                    return set(s[i:i+n] for i in range(max(len(s)-n+1, 1)))

                ngrams_a = char_ngrams(name_a)
                ngrams_b = char_ngrams(name_b)

                if ngrams_a and ngrams_b:
                    jaccard = len(ngrams_a & ngrams_b) / len(ngrams_a | ngrams_b)
                else:
                    jaccard = 0.0

                if jaccard >= 0.3:  # 임계값
                    alignments.append({
                        "source_id": elem_a.uri,
                        "source_name": elem_a.name,
                        "target_id": elem_b.uri,
                        "target_name": elem_b.name,
                        "alignment_type": "equivalent" if jaccard >= 0.8 else "related",
                        "confidence": jaccard,
                        "evidence": {"method": "name_similarity", "jaccard": jaccard},
                    })

        # 신뢰도 내림차순 정렬
        alignments.sort(key=lambda x: x["confidence"], reverse=True)

        return alignments
