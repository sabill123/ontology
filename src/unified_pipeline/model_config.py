"""
Unified Pipeline Model Configuration

에이전트별 LLM 모델 설정 - 각 모델의 강점에 맞게 할당

모델 특성:
- gpt-5.1 (fast): 빠른 처리, 데이터 분석, 구조적 작업에 최적
- claude-opus-4-5-20251101 (balanced): 복잡한 추론, 의미 분석, 판단에 최적
- gemini-3-pro-preview (creative): 창의적 탐색, 관계 발견, 패턴 인식에 최적
- gemini-3-pro-preview (high_context): 대용량 입력 (전체 CSV 등) 처리에 최적

v22.1: HIGH_CONTEXT 모델 타입 추가 — input token이 매우 많은 데이터 분석 태스크용
"""

import os
from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, List, Optional


class ModelType(str, Enum):
    """모델 유형"""
    FAST = "fast"                 # 빠른 처리, 명확한 작업
    BALANCED = "balanced"         # 균형잡힌 성능, 복잡한 추론
    CREATIVE = "creative"         # 창의적 탐색, 패턴 발견
    HIGH_CONTEXT = "high_context" # v22.1: 대용량 입력 (전체 CSV 데이터 분석 등)


# =============================================================================
# v22.1: Model Specifications — 모델별 토큰 한도
# =============================================================================
MODEL_SPECS: Dict[str, Dict[str, int]] = {
    "gemini-3-pro-preview":       {"input_max": 1_048_576, "output_max": 65_535},
    "gpt-5.2":                    {"input_max": 400_000,   "output_max": 128_000},
    "claude-sonnet-4-20250514":   {"input_max": 1_000_000, "output_max": 64_000},
    # 기존 모델 (참고용)
    "gpt-5.1":                    {"input_max": 400_000,   "output_max": 128_000},
    "claude-opus-4-5-20251101":   {"input_max": 200_000,   "output_max": 32_000},
}


# 모델 이름 매핑 (Letsur AI Gateway) — 환경변수로 오버라이드 가능
MODELS = {
    ModelType.FAST: os.environ.get("MODEL_FAST", "gpt-5.1"),
    ModelType.BALANCED: os.environ.get("MODEL_BALANCED", "claude-opus-4-5-20251101"),
    ModelType.CREATIVE: os.environ.get("MODEL_CREATIVE", "gemini-3-pro-preview"),
    ModelType.HIGH_CONTEXT: os.environ.get("MODEL_HIGH_CONTEXT", "gemini-3-pro-preview"),
}


def get_model(model_type: ModelType) -> str:
    """모델 타입으로 모델 이름 가져오기"""
    return MODELS.get(model_type, MODELS[ModelType.BALANCED])


def get_model_spec(model_name: str) -> Dict[str, int]:
    """모델의 토큰 스펙 가져오기 (input_max, output_max)"""
    return MODEL_SPECS.get(model_name, {"input_max": 200_000, "output_max": 32_000})


def get_best_model_for_tokens(estimated_input_tokens: int) -> str:
    """
    v22.1: 예상 입력 토큰 수에 따라 최적 모델 선택.
    입력이 클수록 context window가 큰 모델을 선택한다.

    Args:
        estimated_input_tokens: 예상 입력 토큰 수

    Returns:
        모델 이름
    """
    # HIGH_CONTEXT 모델의 한도를 초과하지 않는 한 HIGH_CONTEXT 모델 사용
    high_ctx_model = MODELS[ModelType.HIGH_CONTEXT]
    high_ctx_spec = get_model_spec(high_ctx_model)

    if estimated_input_tokens <= high_ctx_spec["input_max"]:
        return high_ctx_model

    # 모든 모델 중 input_max가 충분한 모델 찾기
    for model_name, spec in sorted(MODEL_SPECS.items(), key=lambda x: x[1]["input_max"], reverse=True):
        if estimated_input_tokens <= spec["input_max"]:
            return model_name

    # 가장 큰 모델 반환 (truncation은 호출 측에서 처리)
    return max(MODEL_SPECS.items(), key=lambda x: x[1]["input_max"])[0]


# =============================================================================
# Stage 1: Discovery Agents Model Assignments
# =============================================================================
# Discovery 스테이지: 데이터 탐색, 구조 분석
# - 구조적 분석이 많아 fast 모델 위주
# - 관계 탐지는 creative 모델

DISCOVERY_AGENT_MODELS = {
    "data_analyst": ModelType.HIGH_CONTEXT, # v22.1: 전체 CSV 데이터 입력 → 대용량 컨텍스트 모델 (gemini 1M)
    "tda_expert": ModelType.FAST,           # 위상 분석: 수학적, 구조적 → GPT (fast)
    "schema_analyst": ModelType.FAST,       # 스키마 분석: 구조적 → GPT (fast)
    "value_matcher": ModelType.FAST,        # 값 매칭: 데이터 비교 → GPT (fast)
    "entity_classifier": ModelType.CREATIVE,  # v26.2: 엔티티 분류 — 넓은 테이블 분해, 1M 컨텍스트 활용 → Gemini (creative)
    "relationship_detector": ModelType.CREATIVE,  # 관계 탐지: 패턴 발견 → Gemini (creative)
}


# =============================================================================
# Stage 2: Refinement Agents Model Assignments
# =============================================================================
# Refinement 스테이지: 온톨로지 구축, 품질 평가
# - 의미론적 분석이 많아 balanced 모델 위주
# - 충돌 해결은 creative 모델

REFINEMENT_AGENT_MODELS = {
    "ontology_architect": ModelType.CREATIVE,   # v26.2: 온톨로지 설계 — 창의적 구조 설계 + 대용량 컨텍스트 → Gemini (creative)
    "conflict_resolver": ModelType.CREATIVE,   # 충돌 해결: 창의적 해결책 → Gemini (creative)
    "quality_judge": ModelType.BALANCED,       # 품질 평가: 판단력 → Claude (balanced)
    "semantic_validator": ModelType.FAST,       # v26.2: 의미 검증 — 구조적 체크/규칙 검증 → GPT (fast)
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
        model_type=ModelType.CREATIVE,  # v26.2
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
        model_type=ModelType.CREATIVE,  # v26.2
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
        model_type=ModelType.FAST,  # v26.2
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


# =============================================================================
# v17.0 Service Model Assignments
# =============================================================================
# v17.0 서비스별 모델 할당 - 각 서비스의 특성에 맞게 선택

V17_SERVICE_MODELS = {
    # v26.2: 분석 서비스 — 각 모델 강점에 맞게 재배분
    "what_if_analyzer": ModelType.CREATIVE,       # v26.2: 시나리오 해석 — 창의적 탐색 → Gemini
    "decision_explainer": ModelType.FAST,         # v26.2: 결정 설명 — 구조적 설명 생성 → GPT
    "report_generator": ModelType.CREATIVE,       # v26.2: 보고서 — 합성/자연어 생성 → Gemini

    # 데이터 처리 서비스 - fast (구조적 작업)
    "remediation_engine": ModelType.FAST,        # 데이터 수정: 규칙 기반 + LLM 보조
    "semantic_searcher": ModelType.FAST,         # 검색: 쿼리 이해
    "nl2sql_engine": ModelType.FAST,             # NL2SQL: SQL 변환

    # 도구 선택 및 실시간 - creative (패턴 인식)
    "tool_selector": ModelType.CREATIVE,         # 도구 선택: 창의적 매칭
    "streaming_reasoner": ModelType.CREATIVE,    # 실시간: 패턴 발견
}


def get_v17_service_model(service_name: str) -> str:
    """v17.0 서비스 이름으로 모델 가져오기"""
    model_type = V17_SERVICE_MODELS.get(service_name, ModelType.BALANCED)
    return get_model(model_type)


# =============================================================================
# v21.0 Insight Pipeline Multi-Agent Model Assignments
# =============================================================================
# UnifiedInsightPipeline 내부 멀티에이전트 — 각 에이전트 전문성에 맞는 모델

INSIGHT_PIPELINE_MODELS = {
    "data_validator": ModelType.FAST,           # 수치 검증, 데이터 그라운딩 → GPT (fast)
    "business_interpreter": ModelType.CREATIVE,   # v26.2: 비즈니스 해석 — 패턴 합성 → Gemini
    "pattern_discoverer": ModelType.CREATIVE,    # 교차 패턴, 신규 연결 → Gemini (creative)
    "insight_synthesizer": ModelType.CREATIVE,   # v26.2: 최종 합성 — 창의적 합성/처방 → Gemini
}

# 유틸리티 서비스 모델 (에이전트 시스템 우회 코드용)
UTILITY_SERVICE_MODELS = {
    "schema_analysis": ModelType.FAST,            # llm_schema_analyzer.py: 구조 분석
    "fk_verification": ModelType.FAST,            # llm_fk_verifier.py: FK 방향 검증
    "governance_judge": ModelType.BALANCED,        # governance_utils.py: 규칙 판단 → Claude (유지)
    "domain_detection": ModelType.FAST,            # v26.2: 도메인 감지 — 분류 작업 → GPT
    "cross_entity_scenario": ModelType.CREATIVE,   # cross_entity_correlation.py: 시나리오 설계 → Gemini
    "cross_entity_interpret": ModelType.CREATIVE,  # v26.2: 해석 — 패턴 합성 → Gemini
    "default_pipeline": ModelType.FAST,            # v26.2: 기본값 — 범용 → GPT
}


def get_service_model(service_name: str) -> str:
    """v21.0: 서비스/파이프라인 에이전트 이름으로 모델 가져오기"""
    combined = {**INSIGHT_PIPELINE_MODELS, **UTILITY_SERVICE_MODELS}
    model_type = combined.get(service_name, ModelType.BALANCED)
    return get_model(model_type)


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

    # v17.0 서비스
    print("\n" + "-" * 40)
    print("v17.0 SERVICES:")
    for service, model_type in V17_SERVICE_MODELS.items():
        model_name = get_model(model_type)
        print(f"  - {service}: {model_name}")

    print("\n" + "=" * 60)
