"""
LLM 응답용 Pydantic 스키마

Instructor 기반 구조화된 LLM 응답 검증에 사용됩니다.
기존 models.py의 모델들과 호환되도록 설계되었습니다.

v1.0: 초기 구현
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from enum import Enum


# ============================================================
# FK Detection Schemas
# ============================================================

class FKRelationshipType(str, Enum):
    """FK 관계 유형"""
    FOREIGN_KEY = "foreign_key"
    CROSS_SILO_REFERENCE = "cross_silo_reference"
    SEMANTIC_LINK = "semantic_link"
    NONE = "none"


class Cardinality(str, Enum):
    """카디널리티"""
    ONE_TO_ONE = "ONE_TO_ONE"
    ONE_TO_MANY = "ONE_TO_MANY"
    MANY_TO_ONE = "MANY_TO_ONE"
    MANY_TO_MANY = "MANY_TO_MANY"
    UNKNOWN = "UNKNOWN"


class FKAnalysisResult(BaseModel):
    """FK 관계 분석 결과"""
    is_relationship: bool = Field(
        description="두 컬럼 간 FK 관계 존재 여부"
    )
    relationship_type: FKRelationshipType = Field(
        default=FKRelationshipType.NONE,
        description="관계 유형: foreign_key, cross_silo_reference, semantic_link, none"
    )
    confidence: float = Field(
        ge=0.0, le=1.0,
        description="분석 신뢰도 (0.0-1.0)"
    )
    cardinality: Cardinality = Field(
        default=Cardinality.UNKNOWN,
        description="카디널리티: ONE_TO_ONE, ONE_TO_MANY, MANY_TO_ONE, MANY_TO_MANY"
    )
    semantic_relationship: str = Field(
        default="",
        description="의미론적 관계 설명 (예: 'customer places orders')"
    )
    reasoning: str = Field(
        description="판단 근거"
    )


class FKDirectionAnalysis(BaseModel):
    """FK 방향 분석 결과"""
    source_table: str = Field(description="FK가 있는 테이블 (참조하는 쪽)")
    source_column: str = Field(description="FK 컬럼")
    target_table: str = Field(description="PK가 있는 테이블 (참조되는 쪽)")
    target_column: str = Field(description="참조되는 PK 컬럼")
    direction_confidence: float = Field(
        ge=0.0, le=1.0,
        description="방향 판단 신뢰도"
    )
    reasoning: str = Field(description="방향 판단 근거")


# ============================================================
# Column Analysis Schemas
# ============================================================

class SemanticRole(str, Enum):
    """컬럼 의미역"""
    IDENTIFIER = "IDENTIFIER"
    DIMENSION = "DIMENSION"
    MEASURE = "MEASURE"
    TEMPORAL = "TEMPORAL"
    DESCRIPTOR = "DESCRIPTOR"
    FLAG = "FLAG"
    REFERENCE = "REFERENCE"
    UNKNOWN = "UNKNOWN"


class ColumnTypeAnalysis(BaseModel):
    """컬럼 타입 분석 결과"""
    inferred_type: str = Field(
        description="추론된 데이터 타입: string, integer, float, boolean, date, datetime, email, phone, uuid, code, id"
    )
    semantic_type: str = Field(
        description="의미론적 타입 (예: customer_identifier, transaction_date)"
    )
    should_preserve_as_string: bool = Field(
        description="문자열로 유지해야 하는지 (선행 0, 코드 등)"
    )
    is_identifier: bool = Field(description="식별자 여부")
    is_primary_key_candidate: bool = Field(description="PK 후보 여부")
    is_foreign_key_candidate: bool = Field(description="FK 후보 여부")
    referenced_entity: Optional[str] = Field(
        default=None,
        description="FK 후보인 경우 참조 엔티티 이름"
    )
    pattern_description: str = Field(description="감지된 패턴 설명")
    confidence: float = Field(ge=0.0, le=1.0, description="분석 신뢰도")


class SemanticRoleClassification(BaseModel):
    """컬럼 의미역 분류 결과"""
    semantic_role: SemanticRole = Field(description="의미역 분류")
    confidence: float = Field(ge=0.0, le=1.0, description="분류 신뢰도")
    business_name: str = Field(description="비즈니스 표시 이름")
    business_description: str = Field(description="비즈니스 관점 설명")
    domain_category: Optional[str] = Field(
        default=None,
        description="도메인 카테고리: finance, customer, time, location, product, risk, marketing, healthcare, operations"
    )
    reasoning: str = Field(description="분류 근거")


class BatchColumnClassification(BaseModel):
    """배치 컬럼 분류 결과"""
    classifications: Dict[str, SemanticRoleClassification] = Field(
        default_factory=dict,
        description="컬럼별 분류 결과"
    )


# ============================================================
# Entity Analysis Schemas
# ============================================================

class EntityType(str, Enum):
    """엔티티 유형"""
    MASTER_DATA = "master_data"
    TRANSACTION = "transaction"
    REFERENCE = "reference"
    JUNCTION = "junction"
    UNKNOWN = "unknown"


class EntitySemantics(BaseModel):
    """엔티티 의미론 분석 결과"""
    entity_name: str = Field(description="엔티티 이름 (PascalCase)")
    display_name: str = Field(description="표시 이름 (한글 가능)")
    description: str = Field(description="엔티티 설명")
    entity_type: EntityType = Field(description="엔티티 유형")
    business_domain: str = Field(description="비즈니스 도메인")
    key_attributes: List[str] = Field(
        default_factory=list,
        description="핵심 속성 컬럼명"
    )
    relationships_hint: List[str] = Field(
        default_factory=list,
        description="관계 힌트 (예: 'Customer -> Orders')"
    )


# ============================================================
# Consensus Schemas
# ============================================================

class VoteDecision(str, Enum):
    """투표 결정"""
    APPROVE = "approve"
    REJECT = "reject"
    ABSTAIN = "abstain"


class AgentVote(BaseModel):
    """에이전트 투표 결과"""
    decision: VoteDecision = Field(description="투표 결정")
    confidence: float = Field(ge=0.0, le=1.0, description="투표 신뢰도")
    key_observations: List[str] = Field(
        default_factory=list,
        description="핵심 관찰 사항"
    )
    concerns: List[str] = Field(
        default_factory=list,
        description="우려 사항"
    )
    suggestions: List[str] = Field(
        default_factory=list,
        description="제안 사항"
    )
    reasoning: str = Field(description="투표 근거")


class ConsensusResult(BaseModel):
    """컨센서스 결과"""
    final_decision: VoteDecision = Field(description="최종 결정")
    combined_confidence: float = Field(ge=0.0, le=1.0, description="통합 신뢰도")
    vote_distribution: Dict[str, int] = Field(
        default_factory=dict,
        description="투표 분포 {'approve': N, 'reject': M, 'abstain': K}"
    )
    key_reasons: List[str] = Field(
        default_factory=list,
        description="주요 결정 이유"
    )
    dissenting_opinions: List[str] = Field(
        default_factory=list,
        description="반대 의견"
    )


# ============================================================
# Dynamic Threshold Schemas
# ============================================================

class ThresholdRecommendation(BaseModel):
    """동적 임계값 추천 결과"""
    recommended_value: float = Field(
        ge=0.0, le=1.0,
        description="추천 임계값"
    )
    reasoning: str = Field(description="추천 근거")
    context_factors: List[str] = Field(
        default_factory=list,
        description="고려된 컨텍스트 요소"
    )
    confidence: float = Field(ge=0.0, le=1.0, description="추천 신뢰도")
    alternative_values: Dict[str, float] = Field(
        default_factory=dict,
        description="대안 값 {'conservative': 0.X, 'moderate': 0.Y, 'aggressive': 0.Z}"
    )


class BatchThresholdRecommendation(BaseModel):
    """배치 임계값 추천 결과"""
    thresholds: Dict[str, ThresholdRecommendation] = Field(
        default_factory=dict,
        description="임계값 유형별 추천"
    )
    overall_confidence: float = Field(
        ge=0.0, le=1.0,
        description="전체 추천 신뢰도"
    )


# ============================================================
# Quality Assessment Schemas
# ============================================================

class QualityDimension(BaseModel):
    """품질 차원별 점수"""
    score: float = Field(ge=0.0, le=100.0, description="점수 (0-100)")
    issues: List[str] = Field(default_factory=list, description="발견된 이슈")
    recommendations: List[str] = Field(default_factory=list, description="개선 권장사항")


class QualityAssessmentResult(BaseModel):
    """품질 평가 결과"""
    overall_score: float = Field(ge=0.0, le=100.0, description="전체 품질 점수")
    definition_quality: QualityDimension = Field(description="정의 품질")
    evidence_strength: QualityDimension = Field(description="근거 강도")
    consistency: QualityDimension = Field(description="일관성")
    completeness: QualityDimension = Field(description="완전성")
    reasoning: str = Field(description="평가 근거")


# ============================================================
# Cross-Silo Analysis Schemas
# ============================================================

class CrossSiloRelationship(BaseModel):
    """크로스 사일로 관계"""
    from_source: str = Field(description="소스 데이터 출처")
    from_table: str = Field(description="소스 테이블")
    from_column: str = Field(description="소스 컬럼")
    to_source: str = Field(description="대상 데이터 출처")
    to_table: str = Field(description="대상 테이블")
    to_column: str = Field(description="대상 컬럼")
    relationship_type: str = Field(description="관계 유형")
    confidence: float = Field(ge=0.0, le=1.0, description="신뢰도")
    reasoning: str = Field(description="판단 근거")


class CrossSiloAnalysisResult(BaseModel):
    """크로스 사일로 분석 결과"""
    relationships: List[CrossSiloRelationship] = Field(
        default_factory=list,
        description="발견된 관계들"
    )
    entity_mappings: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="엔티티 매핑 (시스템 간 동일 엔티티)"
    )


# ============================================================
# Reasoning Chain Schemas
# ============================================================

class ReasoningStep(BaseModel):
    """추론 단계"""
    step_number: int = Field(description="단계 번호")
    question: str = Field(description="이 단계의 질문")
    analysis: str = Field(description="분석 내용")
    findings: List[str] = Field(default_factory=list, description="발견 사항")
    confidence: float = Field(ge=0.0, le=1.0, description="단계 신뢰도")
    next_question: Optional[str] = Field(
        default=None,
        description="후속 질문 (없으면 추론 종료)"
    )


class ReasoningChainResult(BaseModel):
    """추론 체인 결과"""
    initial_question: str = Field(description="초기 질문")
    steps: List[ReasoningStep] = Field(default_factory=list, description="추론 단계들")
    final_conclusion: str = Field(description="최종 결론")
    insights_discovered: List[str] = Field(
        default_factory=list,
        description="발견된 인사이트"
    )
    confidence: float = Field(ge=0.0, le=1.0, description="전체 신뢰도")
    reasoning_depth: int = Field(description="추론 깊이 (단계 수)")


# ============================================================
# LLM Judge Schemas (for llm_judge.py)
# ============================================================

class JudgeEvaluationResult(BaseModel):
    """Judge 평가 결과 - 개별 Judge의 LLM 응답"""
    score: float = Field(
        ge=0.0, le=1.0,
        description="평가 점수 (0.0-1.0)"
    )
    reasoning: str = Field(
        description="평가 근거 설명"
    )
    strengths: List[str] = Field(
        default_factory=list,
        description="강점 목록"
    )
    weaknesses: List[str] = Field(
        default_factory=list,
        description="약점 목록"
    )
    improvement_suggestions: List[str] = Field(
        default_factory=list,
        description="개선 제안 목록"
    )
    confidence: float = Field(
        ge=0.0, le=1.0,
        default=0.5,
        description="Judge 자신의 평가 확신도"
    )
