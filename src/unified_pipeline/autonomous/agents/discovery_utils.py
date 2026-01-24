"""
Discovery Phase Utility Classes and Functions (v4.1 - Extracted from discovery.py)

이 모듈은 Discovery 단계에서 사용되는 공통 유틸리티 클래스들을 포함합니다:
- Enum 타입들: TopologicalInvariantType, HomeomorphismTypeEnum, EntityTypeEnum
- 데이터 클래스들: EmbeddedTopologicalProfile, EmbeddedBettiNumbers 등
- EmbeddedHomeomorphismAnalyzer: 3단계 위상동형 분석기

v4.1 변경사항:
- Legacy 알고리즘을 직접 파일에 포함 (import 의존성 문제 해결)
- TopologicalProfile, TopologicalInvariant 직접 구현
- SimplicialComplex, BettiNumbers 직접 구현
- 3단계 위상동형 검증 (구조+값+의미) 직접 구현
"""

import json
import logging
import time
import hashlib
import math
from typing import Dict, Any, List, Optional, Set, Tuple
from dataclasses import dataclass, field, asdict
from collections import defaultdict
from enum import Enum


logger = logging.getLogger(__name__)


# =============================================================================
# ENUM DEFINITIONS
# =============================================================================

class TopologicalInvariantType(str, Enum):
    """위상적 불변량 유형 - 위상동형에서 보존되어야 하는 속성"""
    CONNECTIVITY = "connectivity"       # 연결 성분 구조
    ENTITY_ROLE = "entity_role"        # Actor/Transaction/Resource/etc
    SEMANTIC_CORE = "semantic_core"    # 핵심 의미 타입 존재
    RELATIONSHIP_PATTERN = "relationship_pattern"  # Hub, spoke, chain
    CARDINALITY_CLASS = "cardinality_class"  # 1:1, 1:N, M:N 패턴


class HomeomorphismTypeEnum(str, Enum):
    """위상동형 유형"""
    STRUCTURAL = "structural"  # 구조적 위상동형 (Betti numbers)
    VALUE_BASED = "value_based"  # 값 기반 동형 (실제 데이터 겹침)
    SEMANTIC = "semantic"  # 의미론적 동형 (LLM 판단)
    FULL = "full"  # 세 가지 모두 만족


class EntityTypeEnum(str, Enum):
    """엔티티 유형"""
    PATIENT = "Patient"
    ENCOUNTER = "Encounter"
    PROVIDER = "Provider"
    MEDICATION = "Medication"
    DIAGNOSIS = "Diagnosis"
    CLAIM = "Claim"
    DEVICE = "Device"
    LAB_TEST = "LabTest"
    # Supply Chain
    PLANT = "Plant"
    PORT = "Port"
    WAREHOUSE = "Warehouse"
    PRODUCT = "Product"
    CUSTOMER = "Customer"
    ORDER = "Order"
    # General
    UNKNOWN = "Unknown"


# =============================================================================
# DATACLASS DEFINITIONS
# =============================================================================

@dataclass
class EmbeddedBettiNumbers:
    """
    Betti 수 - 기본 위상적 불변량

    b0: 연결 성분 수
    b1: 1차원 "구멍" 수 (루프)
    b2: 2차원 "공동" 수 (빈 공간)

    핵심 속성: 위상동형인 공간은 동일한 Betti 수를 가짐!
    """
    b0: int = 0  # 연결 성분
    b1: int = 0  # 루프/사이클
    b2: int = 0  # 빈 공간/공동

    # 고차원을 위한 확장 Betti 수
    higher: List[int] = field(default_factory=list)

    def to_tuple(self) -> Tuple[int, ...]:
        """비교를 위해 튜플로 변환"""
        return (self.b0, self.b1, self.b2, *self.higher)

    def distance_to(self, other: "EmbeddedBettiNumbers") -> float:
        """
        Betti 수 시퀀스 간 거리 계산

        거리 0 -> 잠재적 위상동형
        """
        t1 = self.to_tuple()
        t2 = other.to_tuple()

        # 같은 길이로 패딩
        max_len = max(len(t1), len(t2))
        t1 = t1 + (0,) * (max_len - len(t1))
        t2 = t2 + (0,) * (max_len - len(t2))

        # 유클리드 거리
        return math.sqrt(sum((a - b) ** 2 for a, b in zip(t1, t2)))

    def is_compatible_with(self, other: "EmbeddedBettiNumbers", tolerance: int = 1) -> bool:
        """Betti 수가 호환 가능한지 확인 (허용 오차 내)"""
        return (
            abs(self.b0 - other.b0) <= tolerance and
            abs(self.b1 - other.b1) <= tolerance and
            abs(self.b2 - other.b2) <= tolerance
        )


@dataclass
class EmbeddedTopologicalProfile:
    """
    스키마의 위상적 "모양"을 캡처하는 프로필

    두 스키마가 동일한 프로필을 가지면 위상동형 후보
    Legacy TopologicalProfile에서 이식
    """
    graph_id: str
    graph_name: str

    # 불변량 1: 연결성
    connected_components: int = 1
    is_connected: bool = True

    # 불변량 2: 엔티티 역할 분포
    entity_roles: Dict[str, int] = field(default_factory=dict)
    # 예: {"actor": 2, "transaction": 3, "resource": 1}

    # 불변량 3: 핵심 의미 타입 (반드시 존재해야 함)
    core_semantic_types: Set[str] = field(default_factory=set)
    # 예: {"identifier", "name", "timestamp"}

    # 불변량 4: 관계 패턴
    has_hub_spoke: bool = False  # 중심 엔티티가 많은 연결을 가짐
    has_chains: bool = False     # 순차적 FK 체인
    max_chain_length: int = 0
    hub_count: int = 0

    # 불변량 5: 카디널리티 패턴
    has_one_to_many: bool = False
    has_many_to_many: bool = False

    # 파생: 빠른 비교를 위한 위상적 해시
    topo_hash: str = ""

    # Betti numbers (런타임에 설정됨)
    betti_numbers: EmbeddedBettiNumbers = field(default_factory=EmbeddedBettiNumbers)
    euler_characteristic: int = 0

    def compute_hash(self) -> str:
        """위상적 불변량만을 기반으로 해시 계산"""
        invariants = {
            "connected": self.is_connected,
            "components": self.connected_components,
            "roles": sorted(self.entity_roles.items()),
            "core_types": sorted(self.core_semantic_types),
            "hub_spoke": self.has_hub_spoke,
            "has_chains": self.has_chains,
        }
        return hashlib.md5(
            json.dumps(invariants, sort_keys=True).encode()
        ).hexdigest()[:12]

    def invariant_distance(self, other: "EmbeddedTopologicalProfile") -> float:
        """
        위상적 프로필 간 거리 계산

        Returns:
            0.0: 동일한 위상
            1.0: 완전히 다름
            < 0.3: 위상동형 스키마
        """
        scores = []

        # 연결성은 반드시 일치해야 함
        if self.is_connected != other.is_connected:
            return 1.0  # 근본적 불일치

        # 연결 성분 수 유사도
        max_comp = max(self.connected_components, other.connected_components, 1)
        comp_diff = abs(self.connected_components - other.connected_components)
        scores.append(1.0 - comp_diff / max_comp)

        # 엔티티 역할 분포 겹침
        all_roles = set(self.entity_roles.keys()) | set(other.entity_roles.keys())
        if all_roles:
            role_overlap = 0
            for role in all_roles:
                if role in self.entity_roles and role in other.entity_roles:
                    role_overlap += 1
            scores.append(role_overlap / len(all_roles))

        # 핵심 의미 타입 겹침 (Jaccard)
        if self.core_semantic_types or other.core_semantic_types:
            intersection = self.core_semantic_types & other.core_semantic_types
            union = self.core_semantic_types | other.core_semantic_types
            scores.append(len(intersection) / len(union) if union else 1.0)

        # 구조적 패턴 일치
        pattern_match = sum([
            self.has_hub_spoke == other.has_hub_spoke,
            self.has_chains == other.has_chains,
            self.has_one_to_many == other.has_one_to_many,
        ]) / 3
        scores.append(pattern_match)

        # 유사도를 거리로 변환
        similarity = sum(scores) / len(scores) if scores else 0.0
        return 1.0 - similarity


@dataclass
class EmbeddedSimplex:
    """
    simplicial complex의 simplex

    TDA에서 simplex는 기본 구성 요소:
    - 0-simplex: 점 (엔티티)
    - 1-simplex: 간선 (관계)
    - 2-simplex: 삼각형 (3-엔티티 사이클)
    - 등등
    """
    vertices: Tuple[str, ...]  # 정렬된 정점 ID
    dimension: int  # 0, 1, 2, ...
    weight: float = 1.0  # 필트레이션용 (simplex가 "나타나는" 시점)
    semantic_type: Optional[str] = None

    def __hash__(self):
        return hash(self.vertices)

    def __eq__(self, other):
        return self.vertices == other.vertices

    @property
    def faces(self) -> List["EmbeddedSimplex"]:
        """이 simplex의 모든 (d-1)차원 면을 가져옴"""
        if self.dimension == 0:
            return []

        faces = []
        for i in range(len(self.vertices)):
            face_vertices = tuple(v for j, v in enumerate(self.vertices) if j != i)
            faces.append(EmbeddedSimplex(
                vertices=face_vertices,
                dimension=self.dimension - 1,
            ))
        return faces


@dataclass
class EmbeddedSimplicialComplex:
    """
    스키마의 위상적 구조를 나타내는 simplicial complex

    SchemaGraph에서 다음과 같이 구축:
    1. 테이블 -> 0-simplex (정점)
    2. FK 관계 -> 1-simplex (간선)
    3. 3-테이블 사이클 -> 2-simplex (삼각형)
    """
    simplices: Dict[int, Set[EmbeddedSimplex]] = field(default_factory=lambda: defaultdict(set))
    # 차원 -> simplex 집합

    max_dimension: int = 0

    def add_simplex(self, simplex: EmbeddedSimplex) -> None:
        """simplex와 모든 면을 추가"""
        self.simplices[simplex.dimension].add(simplex)
        self.max_dimension = max(self.max_dimension, simplex.dimension)

        # 모든 면 추가 (complex가 유효하도록)
        for face in simplex.faces:
            self.add_simplex(face)

    def get_simplices(self, dimension: int) -> Set[EmbeddedSimplex]:
        """주어진 차원의 모든 simplex 가져오기"""
        return self.simplices.get(dimension, set())

    def count_simplices(self, dimension: int) -> int:
        """주어진 차원의 simplex 수 세기"""
        return len(self.simplices.get(dimension, set()))

    @property
    def euler_characteristic(self) -> int:
        """
        오일러 특성 계산: chi = V - E + F - ...

        이것은 위상적 불변량 - 위상동형인 공간은 같은 chi를 가짐
        """
        chi = 0
        for dim in range(self.max_dimension + 1):
            count = self.count_simplices(dim)
            chi += (-1) ** dim * count
        return chi


@dataclass
class EmbeddedValueOverlap:
    """두 컬럼 간 값 겹침 분석"""
    column_a: str
    table_a: str
    column_b: str
    table_b: str
    overlap_count: int = 0
    overlap_ratio_a: float = 0.0
    overlap_ratio_b: float = 0.0
    jaccard_similarity: float = 0.0
    sample_overlapping_values: List[str] = field(default_factory=list)


@dataclass
class EmbeddedHomeomorphismResult:
    """3단계 위상동형 분석 결과"""
    table_a: str
    table_b: str
    is_homeomorphic: bool = False
    homeomorphism_type: Optional[HomeomorphismTypeEnum] = None
    confidence: float = 0.0

    # 3단계 검증 결과
    structural_score: float = 0.0  # TDA 기반
    value_score: float = 0.0  # 값 겹침 기반
    semantic_score: float = 0.0  # LLM 기반

    # 상세 정보
    join_keys: List[Dict[str, Any]] = field(default_factory=list)
    value_overlaps: List[EmbeddedValueOverlap] = field(default_factory=list)
    entity_type: EntityTypeEnum = EntityTypeEnum.UNKNOWN
    reasoning: str = ""

    # TDA 상세
    betti_distance: float = 0.0
    persistence_distance: float = 0.0


# =============================================================================
# HOMEOMORPHISM ANALYZER
# =============================================================================

class EmbeddedHomeomorphismAnalyzer:
    """
    3단계 위상동형 분석기 (Legacy HomeomorphismAnalyzer에서 이식)

    3단계 검증:
    1. TDA 기반 구조적 위상동형 (Betti numbers, Persistence)
    2. 값 기반 엔티티 동일성 (Jaccard, 키 값 겹침)
    3. 의미론적 동형 (컬럼명 패턴, 도메인 휴리스틱)
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def analyze_homeomorphism(
        self,
        tables_data: Dict[str, Dict[str, Any]],
        domain_context: str = "general"
    ) -> Dict[str, Any]:
        """
        전체 위상동형 분석 실행

        Args:
            tables_data: {table_name: {"columns": [...], "data": [...], "metadata": {...}}}
            domain_context: 도메인 컨텍스트

        Returns:
            완전한 위상동형 분석 결과
        """
        start_time = time.time()

        results = {
            "homeomorphisms": [],
            "unified_entities": [],
            "tda_signatures": {},
            "value_overlaps": [],
            "statistics": {}
        }

        table_names = list(tables_data.keys())
        self.logger.info(f"Analyzing homeomorphisms for {len(table_names)} tables")

        # 1단계: TDA 기반 위상적 서명 계산
        tda_signatures = {}
        for table_name, table_info in tables_data.items():
            sig = self._compute_topological_signature(table_name, table_info)
            tda_signatures[table_name] = sig
            results["tda_signatures"][table_name] = {
                "betti_0": sig.betti_numbers.b0,
                "betti_1": sig.betti_numbers.b1,
                "betti_2": sig.betti_numbers.b2,
                "euler": sig.euler_characteristic,
                "components": sig.connected_components,
            }

        # 2단계: 값 기반 겹침 분석
        value_overlaps = self._analyze_value_overlaps(tables_data)
        results["value_overlaps"] = [asdict(v) for v in value_overlaps]

        # 3단계: 종합 위상동형 판정
        homeomorphisms = self._determine_homeomorphisms(
            tda_signatures, value_overlaps, tables_data, domain_context
        )
        # v4.1: Enum 값을 문자열로 변환하여 직렬화
        results["homeomorphisms"] = [self._serialize_homeomorphism(h) for h in homeomorphisms]

        # 4단계: 통합 엔티티 구성
        unified_entities = self._build_unified_entities(homeomorphisms, tables_data, value_overlaps)
        results["unified_entities"] = unified_entities

        # 통계
        elapsed = time.time() - start_time
        results["statistics"] = {
            "tables_analyzed": len(table_names),
            "homeomorphisms_found": len([h for h in homeomorphisms if h.is_homeomorphic]),
            "unified_entities": len(unified_entities),
            "execution_time_seconds": elapsed
        }

        return results

    def _compute_topological_signature(
        self,
        table_name: str,
        table_info: Dict[str, Any]
    ) -> EmbeddedTopologicalProfile:
        """테이블의 위상적 서명 계산"""
        columns = table_info.get("columns", [])

        profile = EmbeddedTopologicalProfile(
            graph_id=table_name,
            graph_name=table_name,
        )

        # Betti numbers 계산
        betti = EmbeddedBettiNumbers()
        betti.b0 = 1  # 단일 테이블은 1개 연결 성분

        # 키 컬럼 분석으로 b1 추정
        key_columns = []
        for col in columns:
            col_name = col.get("name", "")
            is_key = col.get("is_key", False) or col.get("is_key_candidate", False)
            if is_key or col_name.endswith("_id") or col_name == "id":
                key_columns.append(col_name)

        betti.b1 = max(0, len(key_columns) - 1)  # FK 관계 수 추정

        profile.connected_components = betti.b0
        profile.is_connected = (betti.b0 == 1)

        # 의미 타입 분석
        for col in columns:
            col_name = col.get("name", "").lower()
            dtype = col.get("dtype", col.get("inferred_type", ""))

            # 핵심 의미 타입 식별
            if "_id" in col_name or col_name == "id":
                profile.core_semantic_types.add("identifier")
            if "name" in col_name:
                profile.core_semantic_types.add("name")
            if "date" in col_name or "time" in col_name or dtype in ["datetime", "date", "timestamp"]:
                profile.core_semantic_types.add("timestamp")
            if "amount" in col_name or "price" in col_name or "cost" in col_name:
                profile.core_semantic_types.add("monetary")

        # 관계 패턴 분석
        fk_count = len([c for c in columns if "_id" in c.get("name", "").lower() and c.get("name", "").lower() != "id"])
        if fk_count >= 3:
            profile.has_hub_spoke = True
            profile.hub_count = 1
        if fk_count >= 1:
            profile.has_one_to_many = True
            profile.has_chains = True

        profile.betti_numbers = betti
        profile.euler_characteristic = betti.b0 - betti.b1 + betti.b2
        profile.topo_hash = profile.compute_hash()

        return profile

    def _analyze_value_overlaps(
        self,
        tables_data: Dict[str, Dict[str, Any]]
    ) -> List[EmbeddedValueOverlap]:
        """값 기반 겹침 분석"""
        overlaps = []
        table_names = list(tables_data.keys())

        # 각 테이블의 키 컬럼 값 집합 수집
        table_key_values: Dict[str, Dict[str, Set[str]]] = {}

        for table_name, table_info in tables_data.items():
            columns = table_info.get("columns", [])
            data = table_info.get("data", [])

            table_key_values[table_name] = {}

            # 키 후보 컬럼 식별 (v4.1: 더 많은 패턴 지원)
            key_columns = []
            for col in columns:
                col_name = col.get("name", "")
                col_lower = col_name.lower()
                is_key = col.get("is_key", False) or col.get("is_key_candidate", False)
                unique_ratio = col.get("unique_ratio", 0)

                # v4.2: 더 다양한 키 패턴 지원 (Port, Region 등 참조 컬럼 포함)
                is_key_pattern = (
                    col_lower.endswith("_id") or
                    col_lower.endswith("_code") or
                    col_lower.endswith("_key") or
                    col_lower.endswith("_no") or
                    col_lower.endswith("_number") or
                    col_lower.endswith("_port") or      # v4.2: 항구/포트 참조
                    col_lower.endswith("_region") or   # v4.2: 지역 참조
                    col_lower.endswith("_type") or     # v4.2: 타입 참조
                    col_lower.endswith("_name") or     # v4.2: 이름 참조
                    col_lower.endswith("_status") or   # v4.2: 상태 참조
                    col_lower == "id" or
                    col_lower == "code" or
                    col_lower == "port" or             # v4.2: 단독 port
                    col_lower == "region" or           # v4.2: 단독 region
                    "_id_" in col_lower or
                    "_port" in col_lower or            # v4.2: 중간에 _port 포함
                    "port_" in col_lower               # v4.2: port_로 시작
                )

                if is_key or unique_ratio > 0.5 or is_key_pattern:
                    key_columns.append(col_name)

            # 데이터에서 값 추출
            if data:
                for col_name in key_columns:
                    values = set()
                    for row in data:
                        if isinstance(row, dict) and col_name in row:
                            val = row[col_name]
                            if val is not None:
                                values.add(str(val))
                    if values:
                        table_key_values[table_name][col_name] = values

        # 모든 테이블 쌍에 대해 값 겹침 분석
        for i, table_a in enumerate(table_names):
            for table_b in table_names[i+1:]:
                cols_a = table_key_values.get(table_a, {})
                cols_b = table_key_values.get(table_b, {})

                for col_a, values_a in cols_a.items():
                    for col_b, values_b in cols_b.items():
                        if not values_a or not values_b:
                            continue

                        intersection = values_a & values_b
                        union = values_a | values_b

                        if len(intersection) > 0:
                            overlap = EmbeddedValueOverlap(
                                column_a=col_a,
                                table_a=table_a,
                                column_b=col_b,
                                table_b=table_b,
                                overlap_count=len(intersection),
                                overlap_ratio_a=len(intersection) / len(values_a) if values_a else 0,
                                overlap_ratio_b=len(intersection) / len(values_b) if values_b else 0,
                                jaccard_similarity=len(intersection) / len(union) if union else 0,
                                sample_overlapping_values=list(intersection)[:10]
                            )
                            overlaps.append(overlap)

        overlaps.sort(key=lambda x: x.jaccard_similarity, reverse=True)
        return overlaps

    def _determine_homeomorphisms(
        self,
        tda_signatures: Dict[str, EmbeddedTopologicalProfile],
        value_overlaps: List[EmbeddedValueOverlap],
        tables_data: Dict[str, Dict[str, Any]],
        domain_context: str
    ) -> List[EmbeddedHomeomorphismResult]:
        """3단계 검증을 종합하여 위상동형 판정"""
        homeomorphisms = []
        table_names = list(tables_data.keys())

        # 값 겹침을 테이블 쌍 기준으로 그룹화
        overlap_by_pair: Dict[Tuple[str, str], List[EmbeddedValueOverlap]] = defaultdict(list)
        for ov in value_overlaps:
            pair = tuple(sorted([ov.table_a, ov.table_b]))
            overlap_by_pair[pair].append(ov)

        # 모든 테이블 쌍 분석
        for i, table_a in enumerate(table_names):
            for table_b in table_names[i+1:]:
                pair = tuple(sorted([table_a, table_b]))

                result = EmbeddedHomeomorphismResult(
                    table_a=table_a,
                    table_b=table_b
                )

                # 1. 구조적 점수 (TDA)
                sig_a = tda_signatures.get(table_a)
                sig_b = tda_signatures.get(table_b)

                if sig_a and sig_b:
                    # Betti 거리
                    betti_dist = sig_a.betti_numbers.distance_to(sig_b.betti_numbers)
                    result.betti_distance = betti_dist

                    # 위상적 프로필 거리
                    topo_dist = sig_a.invariant_distance(sig_b)

                    # 구조적 점수 (거리가 작을수록 높은 점수)
                    result.structural_score = 1.0 / (1.0 + betti_dist + topo_dist)

                # 2. 값 기반 점수
                pair_overlaps = overlap_by_pair.get(pair, [])
                if pair_overlaps:
                    best_overlap = max(pair_overlaps, key=lambda x: x.jaccard_similarity)
                    result.value_score = best_overlap.jaccard_similarity
                    result.value_overlaps = pair_overlaps

                    for ov in pair_overlaps:
                        if ov.jaccard_similarity > 0.3:
                            result.join_keys.append({
                                "table_a_column": ov.column_a,
                                "table_b_column": ov.column_b,
                                "overlap_ratio": ov.jaccard_similarity,
                                "sample_values": ov.sample_overlapping_values[:5]
                            })

                # 3. 의미론적 점수 (컬럼명 패턴 기반)
                result.semantic_score = self._compute_semantic_score(
                    table_a, table_b, tables_data, domain_context
                )

                # 4. Entity 타입 추론
                result.entity_type = self._infer_entity_type_for_pair(
                    table_a, table_b, result.join_keys, tables_data
                )

                # 컬럼명 일치 보너스
                column_name_bonus = 0.0
                for jk in result.join_keys:
                    col_a = jk.get("table_a_column", "")
                    col_b = jk.get("table_b_column", "")
                    if col_a == col_b:
                        column_name_bonus = 0.1
                        break

                # 종합 판정 (가중치: 값 기반 50%, 의미론적 30%, 구조적 20%)
                weighted_score = (
                    result.value_score * 0.5 +
                    result.semantic_score * 0.3 +
                    result.structural_score * 0.2 +
                    column_name_bonus
                )
                result.confidence = weighted_score

                # === v5.1: 완화된 위상동형 판정 (더 많은 관계 탐지) ===
                # 동일 컬럼명으로 조인 가능하면 관계 있음
                has_matching_columns = column_name_bonus > 0
                # 값 겹침이 있으면 관계 있음
                has_value_overlap = result.value_score >= 0.1  # 10% 이상 겹침
                # 조인 키가 발견됨
                has_join_keys = len(result.join_keys) > 0

                # 강한 관계: 값 기반으로 명확한 연결
                if result.value_score >= 0.35:  # v5.1: 0.65 -> 0.35로 완화
                    result.is_homeomorphic = True
                    result.homeomorphism_type = HomeomorphismTypeEnum.VALUE_BASED
                # 중간 관계: 가중 점수가 충분
                elif weighted_score >= 0.30:  # v5.1: 0.55 -> 0.30으로 완화
                    result.is_homeomorphic = True
                    if result.semantic_score > result.value_score:
                        result.homeomorphism_type = HomeomorphismTypeEnum.SEMANTIC
                    else:
                        result.homeomorphism_type = HomeomorphismTypeEnum.FULL
                # 약한 관계: 컬럼명 일치 + 약간의 값 겹침
                elif has_matching_columns and has_value_overlap:
                    result.is_homeomorphic = True
                    result.homeomorphism_type = HomeomorphismTypeEnum.STRUCTURAL
                # 조인 키 발견 시 관계 표시
                elif has_join_keys and result.value_score >= 0.05:
                    result.is_homeomorphic = True
                    result.homeomorphism_type = HomeomorphismTypeEnum.STRUCTURAL
                    result.confidence = max(result.confidence, 0.25)  # 최소 25% 신뢰도

                homeomorphisms.append(result)

        return homeomorphisms

    def _compute_semantic_score(
        self,
        table_a: str,
        table_b: str,
        tables_data: Dict[str, Dict[str, Any]],
        domain_context: str
    ) -> float:
        """의미론적 유사도 점수 계산 (컬럼명 패턴 기반)"""
        cols_a = {c.get("name", "").lower() for c in tables_data.get(table_a, {}).get("columns", [])}
        cols_b = {c.get("name", "").lower() for c in tables_data.get(table_b, {}).get("columns", [])}

        if not cols_a or not cols_b:
            return 0.0

        # 컬럼명 Jaccard 유사도
        intersection = cols_a & cols_b
        union = cols_a | cols_b
        base_score = len(intersection) / len(union) if union else 0.0

        # 패턴 기반 보너스
        bonus = 0.0

        # 같은 엔티티를 가리키는 컬럼이 있는지 확인
        entity_patterns = ["patient", "encounter", "claim", "provider", "order", "product"]
        for pattern in entity_patterns:
            has_a = any(pattern in c for c in cols_a)
            has_b = any(pattern in c for c in cols_b)
            if has_a and has_b:
                bonus += 0.1
                break

        return min(1.0, base_score + bonus)

    def _infer_entity_type_for_pair(
        self,
        table_a: str,
        table_b: str,
        join_keys: List[Dict[str, Any]],
        tables_data: Dict[str, Dict[str, Any]]
    ) -> EntityTypeEnum:
        """두 테이블 쌍과 조인 키를 기반으로 Entity 타입 추론"""
        key_patterns = {
            EntityTypeEnum.PATIENT: ["patient_id", "patient_mrn", "mrn", "person_id"],
            EntityTypeEnum.ENCOUNTER: ["encounter_id", "visit_id", "admission_id"],
            EntityTypeEnum.CLAIM: ["claim_id"],
            EntityTypeEnum.MEDICATION: ["prescription_id", "rx_id", "medication_id"],
            EntityTypeEnum.LAB_TEST: ["lab_id", "test_id", "result_id"],
            EntityTypeEnum.DEVICE: ["device_id", "reading_id"],
            EntityTypeEnum.PROVIDER: ["provider_id", "physician_id", "doctor_id"],
            EntityTypeEnum.PRODUCT: ["product_id", "sku", "item_id"],
            EntityTypeEnum.ORDER: ["order_id", "order_number"],
            EntityTypeEnum.CUSTOMER: ["customer_id", "client_id"],
        }

        for jk in join_keys:
            col_a = jk.get("table_a_column", "").lower()
            col_b = jk.get("table_b_column", "").lower()

            for entity_type, patterns in key_patterns.items():
                for pattern in patterns:
                    if pattern in col_a or pattern in col_b:
                        return entity_type

        return EntityTypeEnum.UNKNOWN

    def _build_unified_entities(
        self,
        homeomorphisms: List[EmbeddedHomeomorphismResult],
        tables_data: Dict[str, Dict[str, Any]],
        value_overlaps: List[EmbeddedValueOverlap]
    ) -> List[Dict[str, Any]]:
        """위상동형 테이블들을 통합 엔티티로 구성"""
        # Union-Find로 동형 테이블 클러스터링
        parent = {}

        def find(x):
            if x not in parent:
                parent[x] = x
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]

        def union(x, y):
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py

        for h in homeomorphisms:
            if h.is_homeomorphic:
                union(h.table_a, h.table_b)

        # 클러스터별 테이블 수집
        clusters: Dict[str, List[str]] = defaultdict(list)
        all_tables = set()
        for h in homeomorphisms:
            all_tables.add(h.table_a)
            all_tables.add(h.table_b)

        for table in all_tables:
            root = find(table)
            clusters[root].append(table)

        # 통합 엔티티 생성
        unified_entities = []

        for root, tables in clusters.items():
            if len(tables) < 2:
                continue

            entity_type = EntityTypeEnum.UNKNOWN
            for h in homeomorphisms:
                if h.is_homeomorphic and h.table_a in tables and h.table_b in tables:
                    if h.entity_type != EntityTypeEnum.UNKNOWN:
                        entity_type = h.entity_type
                        break

            # 키 매핑 수집
            key_mappings = {}
            for h in homeomorphisms:
                if h.table_a in tables and h.table_b in tables:
                    for jk in h.join_keys:
                        key_mappings[h.table_a] = jk.get("table_a_column")
                        key_mappings[h.table_b] = jk.get("table_b_column")

            # 신뢰도 계산
            confidence = 0.0
            for h in homeomorphisms:
                if h.table_a in tables and h.table_b in tables:
                    confidence = max(confidence, h.confidence)

            entity = {
                "entity_id": f"E_{entity_type.value.upper()}_{hashlib.md5(root.encode()).hexdigest()[:6]}",
                "entity_type": entity_type.value,
                "canonical_name": entity_type.value,
                "source_tables": tables,
                "key_mappings": key_mappings,
                "confidence": confidence
            }
            unified_entities.append(entity)

        return unified_entities

    def _serialize_homeomorphism(self, h: EmbeddedHomeomorphismResult) -> Dict[str, Any]:
        """Homeomorphism 결과를 JSON 직렬화 가능한 딕셔너리로 변환 (Enum 처리)"""
        return {
            "table_a": h.table_a,
            "table_b": h.table_b,
            "is_homeomorphic": h.is_homeomorphic,
            "homeomorphism_type": h.homeomorphism_type.value if h.homeomorphism_type else None,
            "confidence": h.confidence,
            "structural_score": h.structural_score,
            "value_score": h.value_score,
            "semantic_score": h.semantic_score,
            "join_keys": h.join_keys,
            "value_overlaps": [asdict(v) for v in h.value_overlaps] if h.value_overlaps else [],
            "entity_type": h.entity_type.value if h.entity_type else "Unknown",
            "reasoning": h.reasoning,
            "betti_distance": h.betti_distance,
            "persistence_distance": h.persistence_distance,
        }


# =============================================================================
# LEGACY AVAILABILITY FLAGS
# =============================================================================
# v4.5: Legacy 알고리즘이 직접 포함되어 있으므로 항상 사용 가능
LEGACY_AVAILABLE = True
LEGACY_EMBEDDED = True  # v4.1에서 직접 포함됨을 표시
LEGACY_PHASE1_AVAILABLE = True  # v4.5: 일관성을 위해 추가 (refinement.py, governance.py와 동일)
