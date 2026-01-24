"""
Unified Pipeline Pydantic Models

v6.0: 강타입 데이터 모델
- Dict[str, Any] 대신 명시적 타입
- 자동 검증
- IDE 자동완성 지원

사용 예:
    # 기존 (Dict[str, Any])
    overlap = {"table_a": "t1", "overlap_ratio": 0.8}  # 타입 불명확

    # 개선 (Pydantic)
    overlap = ValueOverlap(table_a="t1", table_b="t2", overlap_ratio=0.8)  # 검증됨
"""

from datetime import datetime
from typing import Dict, List, Optional, Any, Union
from enum import Enum
from pydantic import BaseModel, Field, field_validator, model_validator


# === Enums ===

class ConceptType(str, Enum):
    """개념 유형"""
    OBJECT_TYPE = "object_type"
    LINK_TYPE = "link_type"
    PROPERTY = "property"


class ConceptStatus(str, Enum):
    """개념 상태"""
    PENDING = "pending"
    APPROVED = "approved"
    PROVISIONAL = "provisional"
    REJECTED = "rejected"


class RelationshipType(str, Enum):
    """관계 유형"""
    ONE_TO_ONE = "1:1"
    ONE_TO_MANY = "1:N"
    MANY_TO_MANY = "N:M"


class DecisionType(str, Enum):
    """결정 유형"""
    APPROVE = "approve"
    REJECT = "reject"
    ESCALATE = "escalate"
    SCHEDULE_REVIEW = "schedule_review"


class RiskLevel(str, Enum):
    """리스크 수준"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


# === Discovery Models ===

class ColumnInfo(BaseModel):
    """컬럼 정보"""
    name: str
    data_type: str = "string"
    nullable: bool = True
    unique_count: int = 0
    null_count: int = 0
    sample_values: List[str] = Field(default_factory=list)

    # 의미론적 정보
    semantic_type: Optional[str] = None
    is_identifier: bool = False
    is_primary_key: bool = False
    is_foreign_key: bool = False
    referenced_table: Optional[str] = None


class TableInfoModel(BaseModel):
    """테이블 정보 모델"""
    name: str
    columns: List[str]
    row_count: int = 0
    column_info: Dict[str, ColumnInfo] = Field(default_factory=dict)
    sample_data: Optional[List[Dict[str, Any]]] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ValueOverlap(BaseModel):
    """값 겹침 정보"""
    table_a: str
    table_b: str
    column_a: str
    column_b: str
    overlap_ratio: float = Field(..., ge=0.0, le=1.0)
    shared_value_count: int = 0
    total_unique_a: int = 0
    total_unique_b: int = 0
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    evidence_type: str = "exact"  # exact, semantic, pattern

    class Config:
        validate_assignment = True


class JoinKey(BaseModel):
    """조인 키 정보"""
    table_a_column: str
    table_b_column: str
    match_type: str = "exact"  # exact, fuzzy, semantic
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class HomeomorphismPair(BaseModel):
    """위상동형 쌍"""
    table_a: str
    table_b: str
    is_homeomorphic: bool = False
    homeomorphism_type: str = "value_based"  # value_based, structural, semantic
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    join_keys: List[JoinKey] = Field(default_factory=list)
    relationship_type: RelationshipType = RelationshipType.ONE_TO_MANY
    evidence: Dict[str, Any] = Field(default_factory=dict)


class ExtractedEntity(BaseModel):
    """추출된 엔티티"""
    entity_id: str
    name: str
    source_table: str
    source_columns: List[str] = Field(default_factory=list)
    entity_type: str = "unknown"  # person, organization, product, etc.
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    attributes: Dict[str, Any] = Field(default_factory=dict)


# === Refinement Models ===

class OntologyConceptModel(BaseModel):
    """온톨로지 개념 모델"""
    concept_id: str
    concept_type: ConceptType = ConceptType.OBJECT_TYPE
    name: str
    description: str = ""
    definition: Dict[str, Any] = Field(default_factory=dict)

    # 출처 정보
    source_tables: List[str] = Field(default_factory=list)
    source_evidence: Dict[str, Any] = Field(default_factory=dict)

    # 평가 정보
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    status: ConceptStatus = ConceptStatus.PENDING

    # 에이전트 평가
    agent_assessments: Dict[str, Any] = Field(default_factory=dict)

    # 메타데이터
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)

    @field_validator('confidence')
    @classmethod
    def validate_confidence(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError('confidence must be between 0 and 1')
        return round(v, 4)


class ConflictInfo(BaseModel):
    """충돌 정보"""
    conflict_id: str
    conflict_type: str  # name_collision, semantic_overlap, structural_conflict
    concepts_involved: List[str] = Field(default_factory=list)
    severity: str = "medium"  # low, medium, high
    resolution_status: str = "unresolved"  # unresolved, resolved, escalated
    resolution_strategy: Optional[str] = None
    resolved_at: Optional[datetime] = None


class QualityScore(BaseModel):
    """품질 점수"""
    concept_id: str
    overall_score: float = Field(..., ge=0.0, le=100.0)

    # 세부 점수
    definition_quality: float = Field(default=0.0, ge=0.0, le=100.0)
    evidence_strength: float = Field(default=0.0, ge=0.0, le=100.0)
    consistency_score: float = Field(default=0.0, ge=0.0, le=100.0)
    completeness_score: float = Field(default=0.0, ge=0.0, le=100.0)

    # 메타
    evaluated_by: str = "algorithm"
    evaluated_at: datetime = Field(default_factory=datetime.now)


# === Governance Models ===

class GovernanceDecisionModel(BaseModel):
    """거버넌스 결정 모델"""
    decision_id: str
    concept_id: str
    decision_type: DecisionType

    # 결정 근거
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    reasoning: str = ""
    quality_score: float = Field(default=0.0, ge=0.0, le=100.0)
    evidence_strength: float = Field(default=0.0, ge=0.0, le=100.0)

    # 비즈니스 컨텍스트
    business_value: str = "medium"  # low, medium, high
    risk_level: RiskLevel = RiskLevel.MEDIUM

    # 권장 액션
    recommended_actions: List[Dict[str, Any]] = Field(default_factory=list)

    # 메타데이터
    made_by: str = "algorithm"
    made_at: datetime = Field(default_factory=datetime.now)

    # 에이전트 의견
    agent_opinions: Dict[str, Any] = Field(default_factory=dict)


class RiskAssessmentModel(BaseModel):
    """리스크 평가 모델"""
    item_id: str
    item_type: str = "concept"  # concept, action, policy
    category: str  # data_quality, security, compliance, operational

    # 리스크 점수
    likelihood: int = Field(..., ge=1, le=5)
    impact: int = Field(..., ge=1, le=5)
    risk_score: int = Field(..., ge=1, le=25)
    risk_level: RiskLevel

    # 완화 전략
    mitigation: str = ""
    mitigation_status: str = "pending"  # pending, in_progress, completed

    assessed_at: datetime = Field(default_factory=datetime.now)

    @model_validator(mode='after')
    def validate_risk_score(self):
        expected_score = self.likelihood * self.impact
        if self.risk_score != expected_score:
            self.risk_score = expected_score
        return self


class ActionItem(BaseModel):
    """액션 아이템"""
    action_id: str
    title: str
    description: str = ""
    action_type: str  # implementation, documentation, monitoring, review

    # 우선순위
    priority: str = "medium"  # critical, high, medium, low
    priority_score: float = Field(default=0.0, ge=0.0, le=5.0)

    # 영향
    impact: str = "medium"
    effort: str = "medium"

    # 상태
    status: str = "pending"  # pending, in_progress, completed, cancelled
    assignee_role: Optional[str] = None

    # 의존성
    dependencies: List[str] = Field(default_factory=list)

    # 출처
    source_decision: Optional[str] = None

    created_at: datetime = Field(default_factory=datetime.now)


class PolicyRule(BaseModel):
    """정책 규칙"""
    rule_id: str
    rule_name: str
    rule_type: str  # data_quality, access_control, monitoring

    target_concept: str
    condition: str
    severity: str = "medium"
    action_on_violation: str

    is_active: bool = True
    created_at: datetime = Field(default_factory=datetime.now)


# === Consensus Models ===

class AgentOpinionModel(BaseModel):
    """에이전트 의견 모델"""
    agent_id: str
    agent_type: str
    agent_name: str

    # 신뢰도 점수
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    uncertainty: float = Field(default=0.2, ge=0.0, le=1.0)
    doubt: float = Field(default=0.1, ge=0.0, le=1.0)
    evidence_weight: float = Field(default=0.5, ge=0.0, le=1.0)

    # 의견 내용
    reasoning: str = ""
    key_observations: List[str] = Field(default_factory=list)
    concerns: List[str] = Field(default_factory=list)
    suggestions: List[str] = Field(default_factory=list)

    # 타임스탬프
    timestamp: datetime = Field(default_factory=datetime.now)
    round_number: int = 0

    @property
    def adjusted_confidence(self) -> float:
        """조정된 신뢰도"""
        return max(0.0, self.confidence * (1 - self.uncertainty / 2) * (1 - self.doubt / 2))

    @property
    def vote(self) -> str:
        """투표 결과"""
        adj = self.adjusted_confidence
        if adj >= 0.45:
            return "approve"
        elif adj <= 0.2:
            return "reject"
        return "abstain"


# === Pipeline Results ===

class PhaseResult(BaseModel):
    """Phase 결과"""
    phase_name: str
    success: bool
    started_at: datetime
    completed_at: Optional[datetime] = None
    duration_seconds: Optional[float] = None

    # 결과 통계
    items_processed: int = 0
    items_succeeded: int = 0
    items_failed: int = 0

    # 에러 정보
    errors: List[Dict[str, Any]] = Field(default_factory=list)

    # 출력
    output_summary: Dict[str, Any] = Field(default_factory=dict)


class PipelineResult(BaseModel):
    """파이프라인 전체 결과"""
    pipeline_id: str
    started_at: datetime
    completed_at: Optional[datetime] = None
    success: bool = False

    # Phase 결과
    phase_results: Dict[str, PhaseResult] = Field(default_factory=dict)

    # 최종 통계
    tables_analyzed: int = 0
    entities_discovered: int = 0
    relationships_found: int = 0
    concepts_created: int = 0
    decisions_made: int = 0

    # Knowledge Graph
    triples_generated: int = 0

    # 메타데이터
    domain: str = "general"
    version: str = "6.0"


# === Helper Functions ===

def convert_dict_to_model(data: Dict[str, Any], model_class: type) -> BaseModel:
    """
    Dict를 Pydantic 모델로 변환

    Args:
        data: 원본 딕셔너리
        model_class: 대상 Pydantic 모델 클래스

    Returns:
        변환된 모델 인스턴스
    """
    try:
        return model_class(**data)
    except Exception as e:
        # 필수 필드만 추출하여 재시도
        model_fields = model_class.model_fields
        filtered = {k: v for k, v in data.items() if k in model_fields}
        return model_class(**filtered)
