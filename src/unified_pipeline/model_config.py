"""
Unified Pipeline Model Configuration

에이전트별 LLM 모델 설정 - 각 모델의 강점에 맞게 할당

모델 특성:
- gpt-5.1 (fast): 빠른 처리, 데이터 분석, 구조적 작업에 최적
- claude-opus-4-5-20251101 (balanced): 복잡한 추론, 의미 분석, 판단에 최적
- gemini-3-pro-preview (creative): 창의적 탐색, 관계 발견, 패턴 인식에 최적
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, List, Optional


class ModelType(str, Enum):
    """모델 유형"""
    FAST = "fast"           # 빠른 처리, 명확한 작업
    BALANCED = "balanced"   # 균형잡힌 성능, 복잡한 추론
    CREATIVE = "creative"   # 창의적 탐색, 패턴 발견


# 모델 이름 매핑 (Letsur AI Gateway)
MODELS = {
    ModelType.FAST: "gpt-5.1",
    ModelType.BALANCED: "claude-opus-4-5-20251101",
    ModelType.CREATIVE: "gemini-3-pro-preview",
}


def get_model(model_type: ModelType) -> str:
    """모델 타입으로 모델 이름 가져오기"""
    return MODELS.get(model_type, MODELS[ModelType.BALANCED])


# =============================================================================
# Stage 1: Discovery Agents Model Assignments
# =============================================================================
# Discovery 스테이지: 데이터 탐색, 구조 분석
# - 구조적 분석이 많아 fast 모델 위주
# - 관계 탐지는 creative 모델

DISCOVERY_AGENT_MODELS = {
    "tda_expert": ModelType.FAST,           # 위상 분석: 수학적, 구조적 → GPT (fast)
    "schema_analyst": ModelType.FAST,       # 스키마 분석: 구조적 → GPT (fast)
    "value_matcher": ModelType.FAST,        # 값 매칭: 데이터 비교 → GPT (fast)
    "entity_classifier": ModelType.BALANCED, # 엔티티 분류: 의미 이해 필요 → Claude (balanced)
    "relationship_detector": ModelType.CREATIVE,  # 관계 탐지: 패턴 발견 → Gemini (creative)
}


# =============================================================================
# Stage 2: Refinement Agents Model Assignments
# =============================================================================
# Refinement 스테이지: 온톨로지 구축, 품질 평가
# - 의미론적 분석이 많아 balanced 모델 위주
# - 충돌 해결은 creative 모델

REFINEMENT_AGENT_MODELS = {
    "ontology_architect": ModelType.BALANCED,  # 온톨로지 설계: 복잡한 추론 → Claude (balanced)
    "conflict_resolver": ModelType.CREATIVE,   # 충돌 해결: 창의적 해결책 → Gemini (creative)
    "quality_judge": ModelType.BALANCED,       # 품질 평가: 판단력 → Claude (balanced)
    "semantic_validator": ModelType.BALANCED,  # 의미 검증: 의미 분석 → Claude (balanced)
}


# =============================================================================
# Stage 3: Governance Agents Model Assignments
# =============================================================================
# Governance 스테이지: 전략, 정책, 리스크 평가
# - 전략적 판단이 많아 balanced 모델 위주
# - 정책 생성은 creative 모델

GOVERNANCE_AGENT_MODELS = {
    "governance_strategist": ModelType.BALANCED,  # 거버넌스 전략: 전략적 판단 → Claude (balanced)
    "action_prioritizer": ModelType.FAST,         # 액션 우선순위: 구조적 결정 → GPT (fast)
    "risk_assessor": ModelType.BALANCED,          # 리스크 평가: 신중한 분석 → Claude (balanced)
    "policy_generator": ModelType.CREATIVE,       # 정책 생성: 창의적 규칙 → Gemini (creative)
}


# 전체 에이전트 모델 매핑
ALL_AGENT_MODELS = {
    **DISCOVERY_AGENT_MODELS,
    **REFINEMENT_AGENT_MODELS,
    **GOVERNANCE_AGENT_MODELS,
}


def get_agent_model(agent_role: str) -> str:
    """에이전트 역할로 모델 이름 가져오기"""
    model_type = ALL_AGENT_MODELS.get(agent_role, ModelType.BALANCED)
    return get_model(model_type)


def get_agent_model_type(agent_role: str) -> ModelType:
    """에이전트 역할로 모델 타입 가져오기"""
    return ALL_AGENT_MODELS.get(agent_role, ModelType.BALANCED)


# =============================================================================
# Agent Configuration
# =============================================================================

@dataclass
class UnifiedAgentConfig:
    """통합 파이프라인 에이전트 설정"""
    name: str
    role: str
    stage: str
    model_type: ModelType
    temperature: float = 0.3
    max_tokens: int = 2000
    expertise_keywords: List[str] = field(default_factory=list)
    weight: float = 1.0  # 컨센서스에서의 가중치

    @property
    def model(self) -> str:
        """실제 모델 이름"""
        return get_model(self.model_type)


# =============================================================================
# Pre-defined Agent Configurations
# =============================================================================

UNIFIED_AGENT_CONFIGS = {
    # Stage 1: Discovery
    "tda_expert": UnifiedAgentConfig(
        name="TDA Expert",
        role="tda_expert",
        stage="discovery",
        model_type=ModelType.FAST,
        temperature=0.2,
        expertise_keywords=[
            "betti", "topology", "persistent", "homology",
            "structure", "manifold", "simplex", "filtration"
        ],
        weight=1.2,
    ),
    "schema_analyst": UnifiedAgentConfig(
        name="Schema Analyst",
        role="schema_analyst",
        stage="discovery",
        model_type=ModelType.FAST,
        temperature=0.2,
        expertise_keywords=[
            "schema", "column", "type", "key", "primary", "foreign",
            "constraint", "nullable", "unique", "index"
        ],
        weight=1.0,
    ),
    "value_matcher": UnifiedAgentConfig(
        name="Value Matcher",
        role="value_matcher",
        stage="discovery",
        model_type=ModelType.FAST,
        temperature=0.2,
        expertise_keywords=[
            "value", "overlap", "jaccard", "match", "entity",
            "resolution", "duplicate", "similarity"
        ],
        weight=1.5,
    ),
    "entity_classifier": UnifiedAgentConfig(
        name="Entity Classifier",
        role="entity_classifier",
        stage="discovery",
        model_type=ModelType.BALANCED,
        temperature=0.3,
        expertise_keywords=[
            "entity", "patient", "encounter", "claim", "provider",
            "domain", "healthcare", "classification", "type"
        ],
        weight=1.0,
    ),
    "relationship_detector": UnifiedAgentConfig(
        name="Relationship Detector",
        role="relationship_detector",
        stage="discovery",
        model_type=ModelType.CREATIVE,
        temperature=0.3,
        expertise_keywords=[
            "relationship", "foreign", "key", "join", "link",
            "cardinality", "one-to-many", "many-to-many", "bridge"
        ],
        weight=1.1,
    ),

    # Stage 2: Refinement
    "ontology_architect": UnifiedAgentConfig(
        name="Ontology Architect",
        role="ontology_architect",
        stage="refinement",
        model_type=ModelType.BALANCED,
        temperature=0.3,
        expertise_keywords=[
            "ontology", "concept", "definition", "semantic",
            "model", "object", "property", "link"
        ],
        weight=1.3,
    ),
    "conflict_resolver": UnifiedAgentConfig(
        name="Conflict Resolver",
        role="conflict_resolver",
        stage="refinement",
        model_type=ModelType.CREATIVE,
        temperature=0.4,
        expertise_keywords=[
            "conflict", "merge", "resolution", "consensus",
            "contradiction", "reconcile", "align"
        ],
        weight=1.1,
    ),
    "quality_judge": UnifiedAgentConfig(
        name="Quality Judge",
        role="quality_judge",
        stage="refinement",
        model_type=ModelType.BALANCED,
        temperature=0.2,
        expertise_keywords=[
            "quality", "evaluate", "score", "approve",
            "reject", "assess", "judge", "criteria"
        ],
        weight=1.4,
    ),
    "semantic_validator": UnifiedAgentConfig(
        name="Semantic Validator",
        role="semantic_validator",
        stage="refinement",
        model_type=ModelType.BALANCED,
        temperature=0.2,
        expertise_keywords=[
            "semantic", "validate", "meaning", "consistency",
            "domain", "terminology", "standard"
        ],
        weight=1.0,
    ),

    # Stage 3: Governance
    "governance_strategist": UnifiedAgentConfig(
        name="Governance Strategist",
        role="governance_strategist",
        stage="governance",
        model_type=ModelType.BALANCED,
        temperature=0.3,
        expertise_keywords=[
            "governance", "strategy", "roadmap", "framework",
            "compliance", "oversight", "control"
        ],
        weight=1.3,
    ),
    "action_prioritizer": UnifiedAgentConfig(
        name="Action Prioritizer",
        role="action_prioritizer",
        stage="governance",
        model_type=ModelType.FAST,
        temperature=0.2,
        expertise_keywords=[
            "priority", "action", "backlog", "urgency",
            "impact", "dependency", "sequence"
        ],
        weight=1.1,
    ),
    "risk_assessor": UnifiedAgentConfig(
        name="Risk Assessor",
        role="risk_assessor",
        stage="governance",
        model_type=ModelType.BALANCED,
        temperature=0.2,
        expertise_keywords=[
            "risk", "assess", "mitigation", "vulnerability",
            "threat", "exposure", "impact"
        ],
        weight=1.2,
    ),
    "policy_generator": UnifiedAgentConfig(
        name="Policy Generator",
        role="policy_generator",
        stage="governance",
        model_type=ModelType.CREATIVE,
        temperature=0.4,
        expertise_keywords=[
            "policy", "rule", "constraint", "enforcement",
            "automation", "template", "generate"
        ],
        weight=1.0,
    ),
}


def get_agent_config(agent_role: str) -> UnifiedAgentConfig:
    """에이전트 역할로 설정 가져오기"""
    return UNIFIED_AGENT_CONFIGS.get(agent_role)


# 모델별 에이전트 그룹
def get_agents_by_model_type(model_type: ModelType) -> List[str]:
    """특정 모델 타입을 사용하는 에이전트 목록"""
    return [
        role for role, mtype in ALL_AGENT_MODELS.items()
        if mtype == model_type
    ]


def print_model_assignments():
    """모델 할당 현황 출력"""
    print("\n" + "=" * 60)
    print("UNIFIED PIPELINE MODEL ASSIGNMENTS")
    print("=" * 60)

    for model_type in ModelType:
        model_name = get_model(model_type)
        agents = get_agents_by_model_type(model_type)
        print(f"\n{model_type.value.upper()} ({model_name}):")
        for agent in agents:
            config = UNIFIED_AGENT_CONFIGS.get(agent)
            if config:
                print(f"  - {config.name} ({config.stage})")

    print("\n" + "=" * 60)
