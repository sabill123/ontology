"""
Causal Impact Models - 인과 추론 데이터 모델

Enums, dataclasses for causal inference and impact analysis.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Any
from enum import Enum
import numpy as np


class TreatmentType(Enum):
    """처리(Treatment) 타입"""
    DATA_INTEGRATION = "data_integration"       # 데이터 통합
    SCHEMA_UNIFICATION = "schema_unification"   # 스키마 통합
    FK_ESTABLISHMENT = "fk_establishment"       # FK 관계 수립
    ENTITY_RESOLUTION = "entity_resolution"     # 엔티티 통합
    DATA_QUALITY_FIX = "data_quality_fix"       # 데이터 품질 개선


class EffectType(Enum):
    """효과 타입"""
    ATE = "average_treatment_effect"            # 평균 처리 효과
    ATT = "average_treatment_on_treated"        # 처리군 평균 효과
    CATE = "conditional_average_treatment"      # 조건부 평균 효과


@dataclass
class CausalNode:
    """인과 그래프 노드"""
    name: str
    node_type: str  # treatment, outcome, confounder, mediator, instrument
    observed: bool = True
    values: Optional[np.ndarray] = None


@dataclass
class CausalEdge:
    """인과 그래프 엣지"""
    source: str
    target: str
    weight: float = 1.0
    mechanism: str = "linear"  # linear, nonlinear, threshold


@dataclass
class CausalGraph:
    """인과 그래프 (DAG)"""
    nodes: Dict[str, CausalNode] = field(default_factory=dict)
    edges: List[CausalEdge] = field(default_factory=list)

    def add_node(self, node: CausalNode):
        self.nodes[node.name] = node

    def add_edge(self, edge: CausalEdge):
        self.edges.append(edge)

    def get_parents(self, node_name: str) -> List[str]:
        """부모 노드 조회"""
        return [e.source for e in self.edges if e.target == node_name]

    def get_children(self, node_name: str) -> List[str]:
        """자식 노드 조회"""
        return [e.target for e in self.edges if e.source == node_name]

    def get_confounders(self, treatment: str, outcome: str) -> List[str]:
        """교란 변수 식별 (treatment와 outcome의 공통 원인)"""
        treatment_ancestors = self._get_ancestors(treatment)
        outcome_ancestors = self._get_ancestors(outcome)
        return list(treatment_ancestors & outcome_ancestors)

    def _get_ancestors(self, node_name: str) -> Set[str]:
        """모든 조상 노드"""
        ancestors = set()
        queue = [node_name]
        while queue:
            current = queue.pop(0)
            for parent in self.get_parents(current):
                if parent not in ancestors:
                    ancestors.add(parent)
                    queue.append(parent)
        return ancestors


@dataclass
class TreatmentEffect:
    """처리 효과 추정 결과"""
    effect_type: EffectType
    estimate: float
    std_error: float
    confidence_interval: Tuple[float, float]
    p_value: float
    treatment: str
    outcome: str
    method: str
    sample_size: int


@dataclass
class CounterfactualResult:
    """반사실적 분석 결과"""
    factual_outcome: float
    counterfactual_outcome: float
    individual_effect: float
    confidence: float
    explanation: str


@dataclass
class SensitivityResult:
    """민감도 분석 결과"""
    robustness_value: float  # 결과를 뒤집는데 필요한 교란 정도
    critical_bias: float     # 임계 편향
    interpretation: str


@dataclass
class CausalImpactResult:
    """인과 영향 분석 종합 결과"""
    treatment_effects: List[TreatmentEffect]
    counterfactuals: List[CounterfactualResult]
    sensitivity: SensitivityResult
    causal_graph: CausalGraph
    recommendations: List[str]
    summary: Dict[str, Any]


@dataclass
class GrangerCausalityResult:
    """Granger 인과성 테스트 결과"""
    cause_variable: str
    effect_variable: str
    f_statistic: float
    p_value: float
    optimal_lag: int
    is_causal: bool  # p < 0.05
    r2_restricted: float  # 제한 모델 R^2
    r2_unrestricted: float  # 비제한 모델 R^2
    interpretation: str


@dataclass
class PCAlgorithmResult:
    """PC Algorithm 결과"""
    adjacency_matrix: np.ndarray
    causal_graph: 'CausalGraph'
    skeleton: Set[Tuple[str, str]]
    directed_edges: Set[Tuple[str, str]]
    variable_names: List[str]
    separation_sets: Dict[Tuple[str, str], Set[str]]
    n_conditional_tests: int
    alpha: float


@dataclass
class CATEResult:
    """CATE 추정 결과 (개선된 버전)"""
    cate_estimates: np.ndarray  # 개별 처리 효과
    std_errors: np.ndarray  # 표준 오차
    confidence_intervals: np.ndarray  # (n, 2) 신뢰구간
    ate: float  # 평균 처리 효과
    method: str
    feature_importances: Optional[Dict[str, float]]
    subgroup_effects: Optional[Dict[str, float]]
