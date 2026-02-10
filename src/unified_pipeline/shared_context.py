"""
Shared Context Memory

스테이지 간 공유되는 메모리 시스템
- 파일 I/O 없이 메모리에서 직접 전달
- 각 스테이지의 결과가 다음 스테이지의 입력으로 자동 전달
- 전체 파이프라인 히스토리 추적
- 자동 탐지된 DomainContext 포함
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Any, Optional, Set, Union, TYPE_CHECKING
from enum import Enum
import hashlib
import json

# Import DomainContext
from .domain import DomainContext, DEFAULT_DOMAIN_CONTEXT

# v25.0: Composition registries (extracted from SharedContext)
from .table_registry import TableRegistry
from .entity_registry import EntityRegistry
from .ontology_registry import OntologyRegistry
from .governance_registry import GovernanceRegistry

# v17.0: Type hints for new features
if TYPE_CHECKING:
    from .calibration import ConfidenceCalibrator, CalibrationConfig


class StageType(str, Enum):
    """파이프라인 스테이지 유형"""
    DISCOVERY = "discovery"          # Stage 1: 데이터 발견, TDA, Homeomorphism
    REFINEMENT = "refinement"        # Stage 2: 온톨로지 정제, 충돌 해결
    GOVERNANCE = "governance"        # Stage 3: 거버넌스, 액션 우선순위


@dataclass
class TableInfo:
    """테이블 정보"""
    name: str
    source: str
    columns: List[Dict[str, Any]]
    row_count: int
    sample_data: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    # TDA 결과 (Stage 1에서 추가)
    tda_signature: Optional[Dict[str, Any]] = None
    betti_numbers: Optional[List[int]] = None

    # 키 정보
    primary_keys: List[str] = field(default_factory=list)
    foreign_keys: List[Dict[str, str]] = field(default_factory=list)


@dataclass
class HomeomorphismPair:
    """위상동형 쌍"""
    table_a: str
    table_b: str
    is_homeomorphic: bool
    homeomorphism_type: str  # structural, value_based, semantic, full
    confidence: float

    # 검증 점수
    structural_score: float = 0.0
    value_score: float = 0.0
    semantic_score: float = 0.0

    # 조인 키
    join_keys: List[Dict[str, Any]] = field(default_factory=list)

    # 엔티티 타입
    entity_type: str = "Unknown"

    # 에이전트 분석 결과
    agent_analysis: Dict[str, Any] = field(default_factory=dict)


@dataclass
class UnifiedEntity:
    """통합 엔티티"""
    entity_id: str
    entity_type: str
    canonical_name: str
    source_tables: List[str]
    key_mappings: Dict[str, str]  # table -> key_column
    confidence: float

    # 속성 통합
    unified_attributes: List[Dict[str, Any]] = field(default_factory=list)

    # 검증 정보
    verified_by_values: bool = False
    sample_shared_keys: List[str] = field(default_factory=list)


@dataclass
class OntologyConcept:
    """온톨로지 개념"""
    concept_id: str
    concept_type: str  # object_type, property, link_type, constraint
    name: str
    description: str
    definition: Dict[str, Any]

    # 소스 정보
    source_tables: List[str] = field(default_factory=list)
    source_evidence: Dict[str, Any] = field(default_factory=dict)

    # 검증 상태
    status: str = "pending"  # pending, approved, rejected, provisional
    confidence: float = 0.0

    # 에이전트 평가
    agent_assessments: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # v25.2: 생성 에이전트 식별
    source_agent: str = ""

    # v27.0: 계층 및 속성
    unified_attributes: List[Dict[str, Any]] = field(default_factory=list)
    parent_concept: str = ""  # 상위 개념명 (e.g., "Metrics")


@dataclass
class GovernanceDecision:
    """거버넌스 결정"""
    decision_id: str
    concept_id: str
    decision_type: str  # approve, reject, escalate, schedule_review
    confidence: float
    reasoning: str

    # 에이전트 의견
    agent_opinions: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # 액션
    recommended_actions: List[Dict[str, Any]] = field(default_factory=list)

    # 메타데이터
    made_at: str = field(default_factory=lambda: datetime.now().isoformat())
    made_by: str = "UnifiedPipeline"


@dataclass
class StageResult:
    """스테이지 실행 결과"""
    stage: StageType
    success: bool
    execution_time_seconds: float

    # 결과 요약
    summary: Dict[str, Any] = field(default_factory=dict)

    # 에이전트별 기여
    agent_contributions: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # 에러/경고
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    # 다음 스테이지를 위한 권고
    recommendations_for_next_stage: List[str] = field(default_factory=list)


@dataclass
class SharedContext:
    """
    공유 컨텍스트 메모리

    모든 스테이지가 이 객체를 공유하며, 각 스테이지의 결과가
    자동으로 다음 스테이지의 입력으로 전달됨
    """

    # 메타정보
    scenario_name: str
    domain_context: Union[str, DomainContext] = field(default_factory=lambda: DEFAULT_DOMAIN_CONTEXT)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())

    # Auto-detected domain (new)
    _detected_domain: Optional[DomainContext] = field(default=None, repr=False)

    # === Stage 0: 입력 데이터 ===
    tables: Dict[str, TableInfo] = field(default_factory=dict)
    raw_data: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # === Stage 1: Discovery 결과 ===
    tda_signatures: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    homeomorphisms: List[HomeomorphismPair] = field(default_factory=list)
    unified_entities: List[UnifiedEntity] = field(default_factory=list)
    value_overlaps: List[Dict[str, Any]] = field(default_factory=list)
    indirect_fks: List[Dict[str, Any]] = field(default_factory=list)  # v7.3: Bridge Table 경유 간접 FK
    enhanced_fk_candidates: List[Dict[str, Any]] = field(default_factory=list)  # v2.1
    extracted_entities: List[Dict[str, Any]] = field(default_factory=list)  # v2.1
    schema_analysis: Dict[str, Any] = field(default_factory=dict)  # v2.1
    cross_table_mappings: List[Dict[str, Any]] = field(default_factory=list)  # v2.1
    discovery_stage_result: Optional[StageResult] = None

    # === v13.2: Dynamic Data Storage (LLM 분석 결과 동적 저장) ===
    # LLM이 분석한 결과를 스키마 변경 없이 동적으로 저장
    # 예: data_understanding, column_semantics, data_patterns 등
    dynamic_data: Dict[str, Any] = field(default_factory=dict)

    # === Stage 2: Refinement 결과 ===
    ontology_concepts: List[OntologyConcept] = field(default_factory=list)
    concept_relationships: List[Dict[str, Any]] = field(default_factory=list)
    conflicts_resolved: List[Dict[str, Any]] = field(default_factory=list)
    refinement_stage_result: Optional[StageResult] = None

    # === Stage 3: Governance 결과 ===
    governance_decisions: List[GovernanceDecision] = field(default_factory=list)
    action_backlog: List[Dict[str, Any]] = field(default_factory=list)
    policy_rules: List[Dict[str, Any]] = field(default_factory=list)
    business_insights: List[Dict[str, Any]] = field(default_factory=list)  # v2.1
    governance_stage_result: Optional[StageResult] = None

    # === 파이프라인 히스토리 ===
    stage_history: List[StageResult] = field(default_factory=list)
    agent_messages: List[Dict[str, Any]] = field(default_factory=list)

    # === Knowledge Graph (v4.3) ===
    knowledge_graph_triples: List[Dict[str, Any]] = field(default_factory=list)  # OntologyArchitect 트리플
    semantic_base_triples: List[Dict[str, Any]] = field(default_factory=list)  # SemanticValidator 기본 트리플
    inferred_triples: List[Dict[str, Any]] = field(default_factory=list)  # 추론된 트리플
    governance_triples: List[Dict[str, Any]] = field(default_factory=list)  # GovernanceStrategist 트리플

    # === v6.1: Phase 2 Advanced Features ===
    causal_relationships: List[Dict[str, Any]] = field(default_factory=list)  # Causal Inference 관계
    causal_insights: Dict[str, Any] = field(default_factory=dict)  # 인과 분석 인사이트
    virtual_entities: List[Dict[str, Any]] = field(default_factory=list)  # 가상 시나리오 엔티티
    palantir_insights: List[Dict[str, Any]] = field(default_factory=list)  # v10.0: 팔란티어 스타일 예측 인사이트
    data_directory: Optional[str] = None  # v10.0: 원본 데이터 디렉토리 경로

    # === v6.1: Phase 3 Advanced Features ===
    debate_results: Dict[str, Any] = field(default_factory=dict)  # Structured Debate 결과
    counterfactual_analysis: List[Dict[str, Any]] = field(default_factory=list)  # 반사실적 분석

    # === v11.0: Evidence Chain (블록체인 스타일 근거 체인) ===
    evidence_chain: Any = None  # EvidenceChain 인스턴스
    evidence_registry: Any = None  # EvidenceRegistry 인스턴스
    evidence_based_debate_results: Dict[str, Any] = field(default_factory=dict)  # v11.0 토론 결과

    # === v17.0: Palantir-style Features ===
    # Dataset Versioning (Git-style)
    dataset_versions: Dict[str, List[str]] = field(default_factory=dict)  # dataset_id -> [version_ids]
    current_dataset_version: Optional[str] = None  # 현재 체크아웃된 버전
    dataset_snapshots: Dict[str, Dict[str, Any]] = field(default_factory=dict)  # version_id -> snapshot_data

    # Branching & Merging
    branches: Dict[str, List[str]] = field(default_factory=dict)  # branch_name -> [commit_ids]
    current_branch: str = "main"
    merge_history: List[Dict[str, Any]] = field(default_factory=list)

    # OSDK (Ontology SDK)
    osdk_object_types: Dict[str, Any] = field(default_factory=dict)  # type_id -> ObjectType definition
    osdk_link_types: Dict[str, Any] = field(default_factory=dict)  # link_id -> LinkType definition
    osdk_instances: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)  # type_id -> [instances]

    # Writeback
    writeback_history: List[Dict[str, Any]] = field(default_factory=list)  # 변경 이력
    pending_writebacks: List[Dict[str, Any]] = field(default_factory=list)  # 대기 중인 변경

    # NL2SQL
    nl2sql_schema_context: Optional[Dict[str, Any]] = None  # 스키마 컨텍스트 캐시
    nl2sql_conversation_history: List[Dict[str, Any]] = field(default_factory=list)  # 대화 이력

    # Real-time Streaming
    streaming_insights: List[Dict[str, Any]] = field(default_factory=list)  # 실시간 인사이트
    active_event_triggers: List[Dict[str, Any]] = field(default_factory=list)  # 활성 트리거

    # Confidence Calibration
    confidence_calibrator: Any = None  # ConfidenceCalibrator 인스턴스
    calibration_history: List[Dict[str, Any]] = field(default_factory=list)  # 보정 이력

    # Semantic Search
    vector_index_status: Dict[str, Any] = field(default_factory=dict)  # 벡터 인덱스 상태
    entity_embeddings: Dict[str, List[float]] = field(default_factory=dict)  # 엔티티 임베딩 캐시

    # What-If Analysis
    simulation_scenarios: List[Dict[str, Any]] = field(default_factory=list)  # 시뮬레이션 시나리오
    scenario_results: Dict[str, Dict[str, Any]] = field(default_factory=dict)  # 시나리오 결과

    # === v17.1: Unified Integration (Palantir-style 연결) ===
    # 모든 기능의 결과가 서로 연결되어 참조할 수 있음

    # Unified Insight Pipeline (Algorithm → Simulation → LLM)
    unified_insights_pipeline: List[Dict[str, Any]] = field(default_factory=list)  # 통합 인사이트

    # Cross-component references (컴포넌트 간 참조)
    component_references: Dict[str, Dict[str, List[str]]] = field(default_factory=dict)
    # 예: {"ontology_concept_123": {"related_insights": ["UI-abc"], "related_actions": ["ACT-xyz"]}}

    # LLM Analysis Cache (LLM 분석 결과 캐시)
    llm_analysis_cache: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    # 예: {"table_name:column_name": {"semantic_type": "...", "business_role": "..."}}

    # Integration Metrics (통합 메트릭)
    integration_metrics: Dict[str, Any] = field(default_factory=lambda: {
        "algorithm_confidence": 0.0,
        "simulation_coverage": 0.0,
        "llm_enrichment_rate": 0.0,
        "component_connectivity": 0.0,
    })

    # === 현재 상태 ===
    current_stage: Optional[StageType] = None
    is_complete: bool = False

    # ------------------------------------------------------------------
    # v25.0: Composition registries (created in __post_init__)
    # ------------------------------------------------------------------

    def __post_init__(self) -> None:
        """Initialise composition registries and bind them to our data."""
        # --- Table Registry ---
        self._table_registry = TableRegistry()
        self._table_registry.bind(
            tables=self.tables,
            raw_data=self.raw_data,
            data_directory=self.data_directory,
        )

        # --- Entity Registry ---
        self._entity_registry = EntityRegistry()
        self._entity_registry.bind(
            unified_entities=self.unified_entities,
            homeomorphisms=self.homeomorphisms,
            extracted_entities=self.extracted_entities,
            enhanced_fk_candidates=self.enhanced_fk_candidates,
            value_overlaps=self.value_overlaps,
            indirect_fks=self.indirect_fks,
            cross_table_mappings=self.cross_table_mappings,
            schema_analysis=self.schema_analysis,
            tda_signatures=self.tda_signatures,
        )

        # --- Ontology Registry ---
        self._ontology_registry = OntologyRegistry()
        self._ontology_registry.bind(
            ontology_concepts=self.ontology_concepts,
            concept_relationships=self.concept_relationships,
            knowledge_graph_triples=self.knowledge_graph_triples,
            semantic_base_triples=self.semantic_base_triples,
            inferred_triples=self.inferred_triples,
            governance_triples=self.governance_triples,
        )

        # --- Governance Registry ---
        self._governance_registry = GovernanceRegistry()
        self._governance_registry.bind(
            governance_decisions=self.governance_decisions,
            action_backlog=self.action_backlog,
            policy_rules=self.policy_rules,
            business_insights=self.business_insights,
        )

    # === v13.2: Dynamic Data Access Methods ===

    def get_dynamic(self, key: str, default: Any = None) -> Any:
        """
        동적 데이터 조회

        LLM이 저장한 분석 결과를 키로 조회합니다.
        예: context.get_dynamic('data_patterns', [])
        """
        return self.dynamic_data.get(key, default)

    def set_dynamic(self, key: str, value: Any) -> None:
        """
        동적 데이터 저장

        LLM 분석 결과를 스키마 변경 없이 저장합니다.
        예: context.set_dynamic('custom_analysis', {...})
        """
        self.dynamic_data[key] = value

    def has_dynamic(self, key: str) -> bool:
        """동적 데이터 존재 여부 확인"""
        return key in self.dynamic_data

    def add_table(self, table_info: TableInfo) -> None:
        """테이블 추가"""
        self._table_registry.add_table(table_info)

    def add_homeomorphism(self, pair: HomeomorphismPair) -> None:
        """위상동형 쌍 추가"""
        self._entity_registry.add_homeomorphism(pair)

    def add_unified_entity(self, entity: UnifiedEntity) -> None:
        """통합 엔티티 추가"""
        self._entity_registry.add_unified_entity(entity)

    def add_concept(self, concept: OntologyConcept) -> None:
        """온톨로지 개념 추가"""
        self._ontology_registry.add_concept(concept)

    def add_governance_decision(self, decision: GovernanceDecision) -> None:
        """거버넌스 결정 추가"""
        self._governance_registry.add_governance_decision(decision)

    def add_agent_message(
        self,
        agent_name: str,
        agent_role: str,
        stage: StageType,
        message: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """에이전트 메시지 추가"""
        self.agent_messages.append({
            "agent_name": agent_name,
            "agent_role": agent_role,
            "stage": stage.value,
            "message": message,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {}
        })

    def record_stage_result(self, result: StageResult) -> None:
        """스테이지 결과 기록"""
        self.stage_history.append(result)

        if result.stage == StageType.DISCOVERY:
            self.discovery_stage_result = result
        elif result.stage == StageType.REFINEMENT:
            self.refinement_stage_result = result
        elif result.stage == StageType.GOVERNANCE:
            self.governance_stage_result = result
            self.is_complete = True

    def get_tables_for_entity(self, entity_type: str) -> List[str]:
        """특정 엔티티 타입에 해당하는 테이블 목록"""
        return self._entity_registry.get_tables_for_entity(entity_type)

    def get_homeomorphisms_for_table(self, table_name: str) -> List[HomeomorphismPair]:
        """특정 테이블과 관련된 위상동형 쌍"""
        return self._entity_registry.get_homeomorphisms_for_table(table_name)

    def get_concepts_by_type(self, concept_type: str) -> List[OntologyConcept]:
        """특정 타입의 온톨로지 개념"""
        return self._ontology_registry.get_concepts_by_type(concept_type)

    def get_approved_concepts(self) -> List[OntologyConcept]:
        """승인된 온톨로지 개념"""
        return self._ontology_registry.get_approved_concepts()

    def get_pending_actions(self) -> List[Dict[str, Any]]:
        """대기 중인 액션"""
        return self._governance_registry.get_pending_actions()

    # === Knowledge Graph Methods (v4.3) ===

    def add_triple(
        self,
        subject: str,
        predicate: str,
        obj: str,
        confidence: float = 1.0,
        source: str = "asserted"
    ) -> None:
        """트리플 추가"""
        self._ontology_registry.add_triple(subject, predicate, obj, confidence, source)

    def add_inferred_triple(
        self,
        subject: str,
        predicate: str,
        obj: str,
        confidence: float = 1.0,
        rule_applied: str = None
    ) -> None:
        """추론된 트리플 추가"""
        self._ontology_registry.add_inferred_triple(subject, predicate, obj, confidence, rule_applied)

    def get_all_triples(self) -> List[Dict[str, Any]]:
        """모든 트리플 (ontology + semantic + inferred + governance) 반환"""
        return self._ontology_registry.get_all_triples()

    def get_triple_count(self) -> Dict[str, int]:
        """트리플 통계"""
        return self._ontology_registry.get_triple_count()

    def build_knowledge_graph(self) -> Any:
        """
        KnowledgeGraphBuilder를 사용하여 전체 지식 그래프 빌드

        Returns:
            KnowledgeGraphBuilder 인스턴스
        """
        return self._ontology_registry.build_knowledge_graph(self)

    # === Domain Context Methods (NEW) ===

    def get_domain(self) -> DomainContext:
        """
        Get the domain context (auto-detected or default).

        Returns DomainContext object regardless of whether domain_context
        is a string or DomainContext.
        """
        if self._detected_domain is not None:
            return self._detected_domain
        if isinstance(self.domain_context, DomainContext):
            return self.domain_context
        # Legacy string format - return default with industry set
        return DomainContext(
            industry=self.domain_context if isinstance(self.domain_context, str) else "general",
            is_auto_detected=False
        )

    def set_domain(self, domain: DomainContext) -> None:
        """Set the detected domain context"""
        self._detected_domain = domain
        # Also update the string field for backwards compatibility
        self.domain_context = domain

    def get_industry(self) -> str:
        """Get the industry domain (string)"""
        domain = self.get_domain()
        return domain.industry

    def get_key_entities(self) -> List[str]:
        """Get key entities for this domain"""
        return self.get_domain().key_entities

    def get_key_metrics(self) -> List[str]:
        """Get key metrics for this domain"""
        return self.get_domain().key_metrics

    def get_recommended_workflows(self) -> List[str]:
        """Get recommended workflows for this domain"""
        return self.get_domain().recommended_workflows

    def get_recommended_dashboards(self) -> List[str]:
        """Get recommended dashboards for this domain"""
        return self.get_domain().recommended_dashboards

    def get_insight_patterns(self) -> List[Any]:
        """Get required insight patterns for Phase 2"""
        return self.get_domain().required_insight_patterns

    def get_escalation_path(self, severity: str) -> List[str]:
        """Get escalation path for given severity"""
        return self.get_domain().get_escalation_path(severity)

    def get_workflow_config(self, workflow_type: str) -> Dict[str, Any]:
        """Get configuration for a specific workflow"""
        return self.get_domain().get_workflow_config(workflow_type)

    def get_dashboard_config(self, dashboard_type: str) -> Dict[str, Any]:
        """Get configuration for a specific dashboard"""
        return self.get_domain().get_dashboard_config(dashboard_type)

    def get_statistics(self) -> Dict[str, Any]:
        """전체 통계"""
        domain = self.get_domain()

        # v11.0: Evidence Chain 통계
        evidence_stats = {}
        if self.evidence_chain:
            try:
                evidence_stats = {
                    "evidence_blocks": len(self.evidence_chain.blocks),
                    "evidence_chain_valid": self.evidence_chain.verify_chain_integrity().get("valid", False),
                }
            except Exception:
                evidence_stats = {"evidence_blocks": 0, "evidence_chain_valid": False}

        return {
            "scenario": self.scenario_name,
            "domain": domain.industry if isinstance(domain, DomainContext) else str(domain),
            "domain_confidence": domain.industry_confidence if isinstance(domain, DomainContext) else 1.0,
            "is_auto_detected": domain.is_auto_detected if isinstance(domain, DomainContext) else False,
            "tables_count": len(self.tables),
            "homeomorphisms_found": len([h for h in self.homeomorphisms if h.is_homeomorphic]),
            "unified_entities": len(self.unified_entities),
            "ontology_concepts": len(self.ontology_concepts),
            "approved_concepts": len(self.get_approved_concepts()),
            "governance_decisions": len(self.governance_decisions),
            "pending_actions": len(self.get_pending_actions()),
            "stages_completed": len(self.stage_history),
            "is_complete": self.is_complete,
            # v11.0: Evidence Chain
            **evidence_stats,
        }

    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환"""
        def safe_dict(obj, seen=None):
            if seen is None:
                seen = set()

            obj_id = id(obj)
            if obj_id in seen:
                return "<circular_ref>"
            seen.add(obj_id)

            if hasattr(obj, '__dict__') and not isinstance(obj, type):
                return {k: safe_dict(v, seen.copy()) for k, v in obj.__dict__.items()
                        if not k.startswith('_')}
            elif isinstance(obj, list):
                return [safe_dict(item, seen.copy()) for item in obj]
            elif isinstance(obj, dict):
                return {k: safe_dict(v, seen.copy()) for k, v in obj.items()}
            elif isinstance(obj, Enum):
                return obj.value
            elif isinstance(obj, (str, int, float, bool, type(None))):
                return obj
            else:
                return str(obj)

        # Get domain context as dict
        domain = self.get_domain()
        domain_dict = domain.to_dict() if isinstance(domain, DomainContext) else {"industry": str(domain)}

        return {
            "scenario_name": self.scenario_name,
            "domain_context": domain_dict,
            "created_at": self.created_at,
            "statistics": self.get_statistics(),
            "tables": {k: safe_dict(v) for k, v in self.tables.items()},
            "homeomorphisms": [safe_dict(h) for h in self.homeomorphisms],
            "unified_entities": [safe_dict(e) for e in self.unified_entities],
            "ontology_concepts": [safe_dict(c) for c in self.ontology_concepts],
            "governance_decisions": [safe_dict(d) for d in self.governance_decisions],
            "action_backlog": self.action_backlog,
            "stage_history": [safe_dict(s) for s in self.stage_history],
            # v2.1: Enhanced fields
            "enhanced_fk_candidates": self.enhanced_fk_candidates,
            "extracted_entities": self.extracted_entities,
            "value_overlaps": self.value_overlaps,
            "indirect_fks": self.indirect_fks,  # v7.3: Bridge Table 경유 간접 FK
            "business_insights": self.business_insights,
            "schema_analysis": self.schema_analysis,
            "cross_table_mappings": self.cross_table_mappings,
            "concept_relationships": self.concept_relationships,
            "policy_rules": self.policy_rules,
            # v2.2: Domain-specific outputs
            "key_entities": self.get_key_entities(),
            "key_metrics": self.get_key_metrics(),
            "recommended_workflows": self.get_recommended_workflows(),
            "recommended_dashboards": self.get_recommended_dashboards(),
            "insight_patterns": [
                p.to_dict() if hasattr(p, 'to_dict') else p
                for p in self.get_insight_patterns()
            ],
            # v4.3: Knowledge Graph triples
            "knowledge_graph_triples": self.knowledge_graph_triples,
            "semantic_base_triples": self.semantic_base_triples,
            "inferred_triples": self.inferred_triples,
            "governance_triples": self.governance_triples,
            "triple_statistics": self.get_triple_count(),
            # v6.1: Phase 2 Advanced Features
            "causal_relationships": self.causal_relationships,
            "causal_insights": self.causal_insights,
            "virtual_entities": self.virtual_entities,
            # v10.0: Palantir-style Predictive Insights
            "palantir_insights": self.palantir_insights,
            "data_directory": self.data_directory,
            # v17.1: Unified Insights Pipeline 결과
            "unified_insights_pipeline": [safe_dict(i) for i in self.unified_insights_pipeline],
            "component_references": safe_dict(self.component_references),
            "integration_metrics": safe_dict(self.integration_metrics),
            # v19.2: dynamic_data 직렬화 (column_semantics 등)
            "dynamic_data": self.dynamic_data if hasattr(self, 'dynamic_data') else {},
            # v6.1: Phase 3 Advanced Features
            "debate_results": self.debate_results,
            "counterfactual_analysis": self.counterfactual_analysis,
            # v11.0: Evidence Chain
            "evidence_chain_summary": self.get_evidence_summary(),
            "evidence_based_debate_results": self.evidence_based_debate_results,
            # v22.0: TDA Signatures (이전에 누락됨)
            "tda_signatures": self.tda_signatures,
        }

    def calculate_hash(self) -> str:
        """컨텍스트 해시 계산"""
        content = json.dumps(self.get_statistics(), sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    # === v11.0: Evidence Chain Methods (블록체인 스타일 근거 관리) ===

    def initialize_evidence_chain(self) -> None:
        """
        v11.0: Evidence Chain 초기화

        블록체인 스타일의 근거 체인을 초기화합니다.
        모든 에이전트가 이 체인에 근거 블록을 추가하고 참조할 수 있습니다.
        """
        try:
            from .autonomous.evidence_chain import EvidenceChain, EvidenceRegistry
            self.evidence_chain = EvidenceChain()
            # EvidenceRegistry는 싱글톤이므로 인수 없이 초기화
            self.evidence_registry = EvidenceRegistry()
            # Registry의 chain을 우리 chain으로 설정
            self.evidence_registry.chain = self.evidence_chain
            import logging
            logging.getLogger(__name__).info("v11.0: Evidence Chain initialized successfully")
        except ImportError as e:
            import logging
            logging.getLogger(__name__).warning(f"v11.0: Failed to initialize Evidence Chain: {e}")
            self.evidence_chain = None
            self.evidence_registry = None

    def get_evidence_chain(self) -> Any:
        """Evidence Chain 인스턴스 반환"""
        if self.evidence_chain is None:
            self.initialize_evidence_chain()
        return self.evidence_chain

    def get_evidence_registry(self) -> Any:
        """Evidence Registry 인스턴스 반환"""
        if self.evidence_registry is None:
            self.initialize_evidence_chain()
        return self.evidence_registry

    def get_evidence_by_phase(self, phase: str) -> List[Dict[str, Any]]:
        """
        특정 Phase의 근거 블록들 조회

        Args:
            phase: "phase1_discovery", "phase2_refinement", "phase3_governance"

        Returns:
            해당 Phase의 근거 블록들 (dict 형태)
        """
        if self.evidence_chain is None:
            return []

        try:
            blocks = self.evidence_chain.get_blocks_by_phase(phase)
            return [
                {
                    "block_id": b.block_id,
                    "agent": b.agent,
                    "finding": b.finding,
                    "reasoning": b.reasoning,
                    "conclusion": b.conclusion,
                    "confidence": b.confidence,
                    "evidence_type": b.evidence_type.value if hasattr(b.evidence_type, 'value') else str(b.evidence_type),
                    "data_references": b.data_references,
                    "metrics": b.metrics,
                    "supporting_block_ids": b.supporting_block_ids,
                    "block_hash": b.block_hash,
                }
                for b in blocks
            ]
        except Exception:
            return []

    def get_evidence_summary(self) -> Dict[str, Any]:
        """
        v11.0: Evidence Chain 전체 요약

        Returns:
            phase별 블록 수, 무결성 검증 결과 등
        """
        if self.evidence_chain is None:
            return {"initialized": False, "total_blocks": 0}

        try:
            integrity = self.evidence_chain.verify_chain_integrity()
            phase1_blocks = len(self.evidence_chain.get_blocks_by_phase("phase1_discovery"))
            phase2_blocks = len(self.evidence_chain.get_blocks_by_phase("phase2_refinement"))
            phase3_blocks = len(self.evidence_chain.get_blocks_by_phase("phase3_governance"))

            return {
                "initialized": True,
                "total_blocks": len(self.evidence_chain.blocks),
                "phase1_discovery_blocks": phase1_blocks,
                "phase2_refinement_blocks": phase2_blocks,
                "phase3_governance_blocks": phase3_blocks,
                "chain_integrity": integrity,
            }
        except Exception as e:
            return {"initialized": True, "error": str(e)}

    def get_domain_context(self) -> Optional["DomainContext"]:
        """
        v9.5: 도메인 컨텍스트 반환 (Embedded LLM Judge용)

        Returns:
            DomainContext 인스턴스 또는 None
        """
        return self.get_domain()

    def get_data_directory(self) -> Optional[str]:
        """
        v10.0: 원본 데이터 디렉토리 경로 반환

        Returns:
            데이터 디렉토리 경로 또는 None
        """
        return self._table_registry.get_data_directory()

    def set_data_directory(self, path: str) -> None:
        """
        v25.0: 데이터 디렉토리 설정 (dataclass 필드 + registry 동기화)

        Args:
            path: 데이터 디렉토리 경로
        """
        self.data_directory = path
        self._table_registry.set_data_directory(path)

    def get_full_data(self, table_name: str) -> List[Dict[str, Any]]:
        """
        v22.1: 테이블의 전체 데이터를 반환 (샘플링 금지).

        우선순위:
        1) source CSV 파일에서 전체 로드
        2) sample_data 반환 (전체 데이터가 이미 들어있을 수 있음)

        Returns:
            전체 행 목록 (List[Dict])
        """
        return self._table_registry.get_full_data(table_name)

    def get_all_full_data(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        v22.1: 모든 테이블의 전체 데이터를 반환.

        Returns:
            {table_name: [rows]} 딕셔너리
        """
        return self._table_registry.get_all_full_data()

    # === v14.0: Phase 간 검증 게이트 ===

    def validate_phase1_output(self) -> Dict[str, Any]:
        """
        v14.0: Phase 1 (Discovery) 출력 검증 게이트

        Phase 2로 넘어가기 전에 필수 출력이 생성되었는지 확인합니다.
        누락된 경우 폴백 데이터를 생성합니다.

        Returns:
            {"valid": bool, "warnings": List[str], "fallbacks_applied": List[str]}
        """
        import logging
        logger = logging.getLogger(__name__)

        warnings = []
        fallbacks_applied = []

        # 1. unified_entities 검증
        if not self.unified_entities:
            warnings.append("unified_entities is empty")
            # 폴백: 테이블 이름 기반 엔티티 생성
            if self.tables:
                for table_name, table_info in self.tables.items():
                    fallback_entity = UnifiedEntity(
                        entity_id=f"fallback_{table_name}",
                        entity_type=table_name.replace("_", " ").title().replace(" ", ""),
                        canonical_name=table_name,
                        source_tables=[table_name],
                        key_mappings={table_name: table_info.primary_keys[0] if table_info.primary_keys else "id"},
                        confidence=0.3,
                    )
                    self.unified_entities.append(fallback_entity)
                fallbacks_applied.append("created_fallback_entities_from_tables")
                logger.warning(f"[v14.0 Gate] Created {len(self.tables)} fallback entities from table names")

        # 2. enhanced_fk_candidates 검증 (빈 리스트도 OK, 검증만)
        if not self.enhanced_fk_candidates:
            warnings.append("enhanced_fk_candidates is empty (no foreign key relationships detected)")

        # 3. column_semantics 검증 (Phase 1 DataAnalyst 결과)
        column_semantics = self.get_dynamic("column_semantics", {})
        if not column_semantics:
            warnings.append("column_semantics not found in dynamic_data")
            # v19.1: 규칙 기반 컬럼 시맨틱 추론 (패턴 매칭)
            # v19.2: 우선순위 기반 시맨틱 패턴 매칭
            # 순서: 구체적 패턴(longer) → 일반 패턴(shorter)
            # 각 패턴은 (pattern, role, specificity) — specificity가 높을수록 우선
            _ROLE_PATTERNS_ORDERED = [
                # === 고우선순위: 구체적 복합 패턴 ===
                ("years_exp", "metric"),
                ("experience", "metric"),
                ("_exp", "metric"),
                ("overtime", "metric"),
                ("satisfaction", "metric"),
                ("performance", "dimension"),
                ("remote_ratio", "metric"),
                ("turnover_risk", "flag"),
                ("parent_income", "dimension"),
                ("study_hours", "metric"),
                ("private_tutoring", "flag"),
                ("club_activity", "flag"),
                ("attendance", "metric"),
                # === identifier ===
                ("_id", "identifier"), ("_key", "identifier"),
                ("_code", "identifier"), ("_no", "identifier"),
                ("_num", "identifier"), ("_pk", "identifier"),
                # === metric ===
                ("salary", "metric"), ("score", "metric"),
                ("amount", "metric"), ("count", "metric"),
                ("rate", "metric"), ("ratio", "metric"),
                ("price", "metric"), ("cost", "metric"),
                ("revenue", "metric"), ("profit", "metric"),
                ("total", "metric"), ("sum", "metric"),
                ("avg", "metric"), ("mean", "metric"),
                ("hours", "metric"), ("quantity", "metric"),
                ("percent", "metric"), ("budget", "metric"),
                ("age", "metric"),
                # === flag ===
                ("is_", "flag"), ("has_", "flag"),
                ("flag", "flag"), ("active", "flag"),
                ("enabled", "flag"), ("risk", "flag"),
                ("turnover", "flag"), ("churn", "flag"),
                ("attrition", "flag"),
                # === dimension ===
                ("department", "dimension"), ("role", "dimension"),
                ("type", "dimension"), ("category", "dimension"),
                ("status", "dimension"), ("level", "dimension"),
                ("class", "dimension"), ("group", "dimension"),
                ("region", "dimension"), ("country", "dimension"),
                ("city", "dimension"), ("state", "dimension"),
                ("gender", "dimension"), ("grade", "dimension"),
                ("tier", "dimension"), ("segment", "dimension"),
                # === timestamp ===
                ("date", "timestamp"), ("created", "timestamp"),
                ("updated", "timestamp"), ("timestamp", "timestamp"),
                ("_at", "timestamp"), ("_on", "timestamp"),
                ("month", "timestamp"), ("day", "timestamp"),
                ("quarter", "timestamp"),
                ("year", "timestamp"),  # 일반 "year" — 낮은 우선순위
                # === descriptive ===
                ("name", "descriptive"), ("title", "descriptive"),
                ("label", "descriptive"), ("description", "descriptive"),
                ("comment", "descriptive"), ("note", "descriptive"),
                ("text", "descriptive"), ("address", "descriptive"),
                ("email", "descriptive"), ("phone", "descriptive"),
            ]
            fallback_semantics = {}
            for table_name, table_info in self.tables.items():
                for col_info in table_info.columns:
                    col_name = col_info.get("name", "")
                    col_lower = col_name.lower()
                    dtype = col_info.get("dtype", "")

                    # v19.2: 우선순위 순서대로 첫 번째 매칭 사용
                    inferred_role = "unknown"
                    for pattern, role in _ROLE_PATTERNS_ORDERED:
                        if pattern in col_lower:
                            inferred_role = role
                            break

                    # dtype 기반 보조 추론
                    if inferred_role == "unknown":
                        if dtype in ("int64", "float64", "int32", "float32"):
                            inferred_role = "metric"
                        elif dtype == "object":
                            inferred_role = "dimension"
                        elif "date" in dtype or "time" in dtype:
                            inferred_role = "timestamp"

                    # nullable 추론: identifier/key는 not null, 나머지는 nullable
                    nullable = inferred_role not in ("identifier",)

                    fallback_semantics[col_name] = {
                        "business_role": inferred_role,
                        "nullable_by_design": nullable,
                        "inference_method": "rule_based_v19.1",
                    }
            if fallback_semantics:
                self.set_dynamic("column_semantics", fallback_semantics)
                fallbacks_applied.append("created_fallback_column_semantics")
                inferred_count = sum(1 for v in fallback_semantics.values() if v["business_role"] != "unknown")
                logger.warning(f"[v19.1 Gate] Rule-based column_semantics: {inferred_count}/{len(fallback_semantics)} columns inferred")

        is_valid = len(fallbacks_applied) == 0  # 폴백 없이 통과하면 valid

        if warnings:
            logger.info(f"[v14.0 Gate] Phase 1 validation warnings: {warnings}")

        return {
            "valid": is_valid,
            "warnings": warnings,
            "fallbacks_applied": fallbacks_applied,
        }

    # === v15.0: Agent Drift Mitigation - EMC (Episodic Memory Consolidation) ===

    def consolidate_context(
        self,
        max_agent_messages: int = 50,
        max_evidence_per_phase: int = 20,
        compress_triples: bool = True,
    ) -> Dict[str, Any]:
        """
        v15.0: 에피소딕 메모리 통합 (EMC)

        Agent Drift 논문의 Context Window Pollution 완화를 위해
        컨텍스트를 주기적으로 압축/정리합니다.

        압축 전략:
        1. 오래된 agent_messages 제거 (최근 N개만 유지)
        2. 낮은 confidence 증거 제거
        3. 중복 트리플 병합
        4. 메타 요약 생성

        Args:
            max_agent_messages: 유지할 최대 에이전트 메시지 수
            max_evidence_per_phase: Phase당 유지할 최대 증거 수
            compress_triples: 트리플 압축 여부

        Returns:
            압축 통계 {
                "messages_removed": int,
                "triples_merged": int,
                "evidence_pruned": int,
                "memory_saved_ratio": float
            }
        """
        import logging
        logger = logging.getLogger(__name__)

        stats = {
            "messages_removed": 0,
            "triples_merged": 0,
            "evidence_pruned": 0,
            "memory_saved_ratio": 0.0,
        }

        # 1. Agent Messages 압축 (오래된 메시지 제거)
        original_msg_count = len(self.agent_messages)
        if original_msg_count > max_agent_messages:
            # 중요 메시지 (결정, 에러) 보존
            important_messages = [
                m for m in self.agent_messages
                if m.get("metadata", {}).get("importance") == "high"
                or "decision" in m.get("message", "").lower()
                or "error" in m.get("message", "").lower()
            ]
            # 최근 메시지와 중요 메시지 결합
            recent_messages = self.agent_messages[-max_agent_messages:]
            self.agent_messages = list({
                m.get("timestamp", ""): m
                for m in important_messages + recent_messages
            }.values())[-max_agent_messages:]
            stats["messages_removed"] = original_msg_count - len(self.agent_messages)

        # 2. Knowledge Graph Triples 압축 (중복 제거)
        if compress_triples:
            original_triple_count = len(self.knowledge_graph_triples)

            # 중복 트리플 제거 (subject-predicate-object 기준)
            seen_triples = set()
            unique_triples = []
            for triple in self.knowledge_graph_triples:
                key = (triple.get("subject"), triple.get("predicate"), triple.get("object"))
                if key not in seen_triples:
                    seen_triples.add(key)
                    unique_triples.append(triple)
            self.knowledge_graph_triples = unique_triples

            # Semantic base triples도 동일 처리
            seen_semantic = set()
            unique_semantic = []
            for triple in self.semantic_base_triples:
                key = (triple.get("subject"), triple.get("predicate"), triple.get("object"))
                if key not in seen_semantic:
                    seen_semantic.add(key)
                    unique_semantic.append(triple)
            self.semantic_base_triples = unique_semantic

            stats["triples_merged"] = (
                original_triple_count - len(self.knowledge_graph_triples)
            )

        # 3. Evidence Chain 정리 (낮은 신뢰도 증거 제거)
        if self.evidence_chain and hasattr(self.evidence_chain, 'blocks'):
            original_evidence_count = len(self.evidence_chain.blocks)
            if original_evidence_count > max_evidence_per_phase * 3:  # 3 phases
                # 높은 신뢰도 증거만 유지
                high_conf_blocks = sorted(
                    self.evidence_chain.blocks,
                    key=lambda b: getattr(b, 'confidence', 0),
                    reverse=True
                )[:max_evidence_per_phase * 3]
                # 실제 제거는 하지 않고 통계만 기록 (블록체인 무결성 유지)
                stats["evidence_pruned"] = original_evidence_count - len(high_conf_blocks)

        # 4. 메모리 절감률 추정
        total_before = original_msg_count + (original_triple_count if compress_triples else 0)
        total_after = len(self.agent_messages) + len(self.knowledge_graph_triples)
        if total_before > 0:
            stats["memory_saved_ratio"] = 1.0 - (total_after / total_before)

        logger.info(
            f"[EMC] Context consolidated: "
            f"messages={stats['messages_removed']} removed, "
            f"triples={stats['triples_merged']} merged, "
            f"memory saved={stats['memory_saved_ratio']:.1%}"
        )

        return stats

    def get_context_size_estimate(self) -> Dict[str, int]:
        """
        v15.0: 컨텍스트 크기 추정 (EMC 트리거용)

        Returns:
            각 컴포넌트별 대략적인 토큰 수 추정
        """
        import json

        estimates = {}

        # Agent messages (평균 메시지당 ~50 토큰 추정)
        estimates["agent_messages"] = len(self.agent_messages) * 50

        # Triples (트리플당 ~20 토큰)
        estimates["triples"] = (
            len(self.knowledge_graph_triples) +
            len(self.semantic_base_triples) +
            len(self.inferred_triples) +
            len(self.governance_triples)
        ) * 20

        # Concepts (개념당 ~100 토큰)
        estimates["concepts"] = len(self.ontology_concepts) * 100

        # Relationships (관계당 ~30 토큰)
        estimates["relationships"] = len(self.homeomorphisms) * 30

        # Dynamic data (JSON 직렬화 크기 / 4로 토큰 추정)
        try:
            dynamic_json = json.dumps(self.dynamic_data, default=str)
            estimates["dynamic_data"] = len(dynamic_json) // 4
        except Exception:
            estimates["dynamic_data"] = 0

        estimates["total"] = sum(estimates.values())

        return estimates

    def should_consolidate(self, threshold_tokens: int = 50000) -> bool:
        """
        v15.0: EMC 필요 여부 판단

        Args:
            threshold_tokens: 압축 트리거 토큰 임계값

        Returns:
            True이면 consolidate_context() 호출 권장
        """
        size_estimate = self.get_context_size_estimate()
        return size_estimate["total"] > threshold_tokens

    def validate_phase2_output(self) -> Dict[str, Any]:
        """
        v14.0: Phase 2 (Refinement) 출력 검증 게이트

        Phase 3으로 넘어가기 전에 필수 출력이 생성되었는지 확인합니다.

        Returns:
            {"valid": bool, "warnings": List[str], "fallbacks_applied": List[str]}
        """
        import logging
        logger = logging.getLogger(__name__)

        warnings = []
        fallbacks_applied = []

        # 1. ontology_concepts 검증
        if not self.ontology_concepts:
            warnings.append("ontology_concepts is empty")
            # 폴백: unified_entities에서 개념 생성
            if self.unified_entities:
                for entity in self.unified_entities:
                    fallback_concept = OntologyConcept(
                        concept_id=f"concept_{entity.entity_id}",
                        concept_type="object_type",
                        name=entity.canonical_name,
                        description=f"Entity type derived from {', '.join(entity.source_tables)}",
                        definition={"source_entity": entity.entity_id},
                        source_tables=entity.source_tables,
                        status="provisional",
                        confidence=entity.confidence * 0.8,
                        source_agent="phase2_fallback_gate",
                    )
                    self.ontology_concepts.append(fallback_concept)
                fallbacks_applied.append("created_fallback_concepts_from_entities")
                logger.warning(f"[v14.0 Gate] Created {len(self.unified_entities)} fallback concepts from entities")

        # 2. concept_relationships 검증
        if not self.concept_relationships:
            warnings.append("concept_relationships is empty")
            # 폴백: enhanced_fk_candidates에서 관계 생성
            if self.enhanced_fk_candidates:
                for fk in self.enhanced_fk_candidates:
                    if hasattr(fk, 'confidence') and fk.confidence >= 0.3:
                        self.concept_relationships.append({
                            "from_concept": fk.from_table if hasattr(fk, 'from_table') else fk.get("from_table", ""),
                            "to_concept": fk.to_table if hasattr(fk, 'to_table') else fk.get("to_table", ""),
                            "relationship_type": "references",
                            "cardinality": "N:1",
                            "confidence": fk.confidence if hasattr(fk, 'confidence') else fk.get("confidence", 0.3),
                            "source": "fallback_from_fk_candidates",
                        })
                if self.concept_relationships:
                    fallbacks_applied.append("created_fallback_relationships_from_fk")
                    logger.warning(f"[v14.0 Gate] Created {len(self.concept_relationships)} fallback relationships from FK candidates")

        is_valid = len(fallbacks_applied) == 0

        if warnings:
            logger.info(f"[v14.0 Gate] Phase 2 validation warnings: {warnings}")

        return {
            "valid": is_valid,
            "warnings": warnings,
            "fallbacks_applied": fallbacks_applied,
        }

    def validate_phase3_output(self) -> Dict[str, Any]:
        """
        v17.1: Phase 3 (Governance) 출력 검증 게이트

        파이프라인 완료 전에 필수 출력이 생성되었는지 확인합니다.

        Returns:
            {"valid": bool, "warnings": List[str], "fallbacks_applied": List[str]}
        """
        import logging
        logger = logging.getLogger(__name__)

        warnings = []
        fallbacks_applied = []

        # 1. governance_decisions 검증
        if not self.governance_decisions:
            warnings.append("governance_decisions is empty")
            # 폴백: ontology_concepts에서 기본 결정 생성
            if self.ontology_concepts:
                from .shared_context import GovernanceDecision
                for concept in self.ontology_concepts:
                    # approved 상태면 approve, 아니면 schedule_review
                    decision_type = "approve" if concept.status == "approved" else "schedule_review"
                    fallback_decision = GovernanceDecision(
                        decision_id=f"fallback_decision_{concept.concept_id}",
                        concept_id=concept.concept_id,
                        decision_type=decision_type,
                        confidence=concept.confidence * 0.7,
                        reasoning=f"[v17.1 Fallback] Auto-generated from concept status: {concept.status}",
                    )
                    self.governance_decisions.append(fallback_decision)
                fallbacks_applied.append("created_fallback_decisions_from_concepts")
                logger.warning(f"[v17.1 Gate] Created {len(self.ontology_concepts)} fallback governance decisions")

        # 2. action_backlog 검증 (선택, 빈 리스트도 OK)
        if not self.action_backlog:
            warnings.append("action_backlog is empty (no recommended actions)")
            # 폴백: 기본 액션 생성 (승인된 개념에 대해)
            approved_decisions = [d for d in self.governance_decisions if d.decision_type == "approve"]
            for decision in approved_decisions[:5]:
                self.action_backlog.append({
                    "action_id": f"fallback_action_{decision.decision_id}",
                    "action_type": "implement_concept",
                    "target_concept": decision.concept_id,
                    "priority": "medium",
                    "status": "pending",
                    "description": f"Implement approved concept: {decision.concept_id}",
                    "source": "v17.1_fallback",
                })
            if self.action_backlog:
                fallbacks_applied.append("created_fallback_actions")
                logger.warning(f"[v17.1 Gate] Created {len(self.action_backlog)} fallback actions")

        # 3. policy_rules 검증 (선택)
        if not self.policy_rules:
            warnings.append("policy_rules is empty (no governance policies)")

        # 4. unified_insights_pipeline 검증 (v17.1)
        if not self.unified_insights_pipeline:
            warnings.append("unified_insights_pipeline is empty (Algorithm→Simulation→LLM pipeline not run)")

        # 5. component_references 연결성 검증
        if not self.component_references:
            warnings.append("component_references is empty (no cross-component linking)")

        is_valid = len(fallbacks_applied) == 0

        if warnings:
            logger.info(f"[v17.1 Gate] Phase 3 validation warnings: {warnings}")

        return {
            "valid": is_valid,
            "warnings": warnings,
            "fallbacks_applied": fallbacks_applied,
            "governance_decisions_count": len(self.governance_decisions),
            "action_backlog_count": len(self.action_backlog),
            "unified_insights_count": len(self.unified_insights_pipeline),
        }

    # === v17.0: Confidence Calibration Methods ===

    def initialize_confidence_calibrator(
        self,
        config: Optional["CalibrationConfig"] = None
    ) -> None:
        """
        v17.0: Confidence Calibrator 초기화

        LLM 신뢰도 보정 시스템을 초기화합니다.
        """
        try:
            from .calibration import ConfidenceCalibrator, CalibrationConfig
            self.confidence_calibrator = ConfidenceCalibrator(
                context=self,
                config=config or CalibrationConfig(),
            )
            import logging
            logging.getLogger(__name__).info("v17.0: Confidence Calibrator initialized")
        except ImportError as e:
            import logging
            logging.getLogger(__name__).warning(f"v17.0: Failed to initialize Confidence Calibrator: {e}")
            self.confidence_calibrator = None

    def calibrate_confidence(
        self,
        agent: str,
        raw_confidence: float,
        samples: Optional[List[Dict[str, Any]]] = None,
    ) -> float:
        """
        v17.0: 신뢰도 보정

        Args:
            agent: 에이전트 이름
            raw_confidence: 원본 신뢰도 (0-1)
            samples: 다중 샘플링 결과 (옵션)

        Returns:
            보정된 신뢰도
        """
        if self.confidence_calibrator is None:
            self.initialize_confidence_calibrator()

        if self.confidence_calibrator is None:
            return raw_confidence  # 초기화 실패 시 원본 반환

        result = self.confidence_calibrator.calibrate(
            agent=agent,
            raw_confidence=raw_confidence,
            samples=samples,
        )

        # 히스토리 기록
        self.calibration_history.append(result.to_dict())

        return result.final_confidence

    def record_prediction_outcome(
        self,
        agent: str,
        confidence: float,
        was_correct: bool
    ) -> None:
        """
        v17.0: 예측 결과 기록 (온라인 학습용)

        실제 결과가 밝혀지면 호출하여 보정 매개변수를 업데이트합니다.
        """
        if self.confidence_calibrator is not None:
            self.confidence_calibrator.record_outcome(agent, confidence, was_correct)

    def get_calibration_summary(self) -> Dict[str, Any]:
        """v17.0: 보정 요약 정보 반환"""
        if self.confidence_calibrator is None:
            return {"initialized": False, "calibration_count": 0}

        return self.confidence_calibrator.get_calibration_summary()

    # === v17.0: Dataset Versioning Methods ===

    def create_dataset_snapshot(
        self,
        dataset_id: str,
        message: str = "",
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        v17.0: 데이터셋 스냅샷 생성 (Git commit과 유사)

        Args:
            dataset_id: 데이터셋 식별자
            message: 커밋 메시지
            metadata: 추가 메타데이터

        Returns:
            생성된 버전 ID
        """
        import hashlib
        from datetime import datetime

        # 버전 ID 생성
        timestamp = datetime.now().isoformat()
        content_hash = hashlib.sha256(
            f"{dataset_id}:{timestamp}:{message}".encode()
        ).hexdigest()[:12]
        version_id = f"v_{content_hash}"

        # 스냅샷 데이터
        snapshot = {
            "version_id": version_id,
            "dataset_id": dataset_id,
            "message": message,
            "timestamp": timestamp,
            "metadata": metadata or {},
            "parent_version": self.current_dataset_version,
            # 현재 상태 저장
            "tables": {k: v.__dict__ if hasattr(v, '__dict__') else v for k, v in self.tables.items()},
            "ontology_concepts_count": len(self.ontology_concepts),
            "unified_entities_count": len(self.unified_entities),
        }

        self.dataset_snapshots[version_id] = snapshot

        # 버전 목록 업데이트
        if dataset_id not in self.dataset_versions:
            self.dataset_versions[dataset_id] = []
        self.dataset_versions[dataset_id].append(version_id)

        # 현재 버전 업데이트
        self.current_dataset_version = version_id

        return version_id

    def get_version_history(self, dataset_id: str) -> List[Dict[str, Any]]:
        """v17.0: 데이터셋 버전 히스토리 조회"""
        version_ids = self.dataset_versions.get(dataset_id, [])
        return [
            self.dataset_snapshots.get(vid, {"version_id": vid, "error": "not_found"})
            for vid in version_ids
        ]

    # === v17.0: NL2SQL Methods ===

    def add_nl2sql_query(
        self,
        natural_query: str,
        sql_query: str,
        result_preview: Optional[Dict[str, Any]] = None,
        confidence: float = 0.0
    ) -> None:
        """
        v17.0: NL2SQL 대화 기록 추가

        Args:
            natural_query: 자연어 쿼리
            sql_query: 생성된 SQL
            result_preview: 결과 미리보기
            confidence: 신뢰도
        """
        self.nl2sql_conversation_history.append({
            "timestamp": datetime.now().isoformat(),
            "natural_query": natural_query,
            "sql_query": sql_query,
            "result_preview": result_preview,
            "confidence": confidence,
        })

    def get_recent_nl2sql_context(self, limit: int = 5) -> List[Dict[str, Any]]:
        """v17.0: 최근 NL2SQL 대화 컨텍스트 반환"""
        return self.nl2sql_conversation_history[-limit:]

    # === v17.0: Streaming Insights Methods ===

    def add_streaming_insight(
        self,
        insight_type: str,
        title: str,
        description: str,
        severity: str = "info",
        source_event: Optional[Dict[str, Any]] = None,
        confidence: float = 0.0
    ) -> str:
        """
        v17.0: 실시간 스트리밍 인사이트 추가

        Returns:
            생성된 인사이트 ID
        """
        insight_id = f"stream_insight_{len(self.streaming_insights) + 1}"

        self.streaming_insights.append({
            "id": insight_id,
            "type": insight_type,
            "title": title,
            "description": description,
            "severity": severity,
            "source_event": source_event,
            "confidence": confidence,
            "timestamp": datetime.now().isoformat(),
            "acknowledged": False,
        })

        return insight_id

    def get_unacknowledged_insights(self) -> List[Dict[str, Any]]:
        """v17.0: 확인되지 않은 스트리밍 인사이트 반환"""
        return [i for i in self.streaming_insights if not i.get("acknowledged", False)]

    # === v17.0: What-If Analysis Methods ===

    def add_simulation_scenario(
        self,
        name: str,
        description: str,
        parameters: Dict[str, Any],
        baseline: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        v17.0: What-If 시뮬레이션 시나리오 추가

        Returns:
            생성된 시나리오 ID
        """
        scenario_id = f"scenario_{len(self.simulation_scenarios) + 1}"

        self.simulation_scenarios.append({
            "id": scenario_id,
            "name": name,
            "description": description,
            "parameters": parameters,
            "baseline": baseline,
            "created_at": datetime.now().isoformat(),
            "status": "pending",
        })

        return scenario_id

    def record_scenario_result(
        self,
        scenario_id: str,
        result: Dict[str, Any],
        impact_summary: Optional[str] = None
    ) -> None:
        """v17.0: 시나리오 실행 결과 기록"""
        self.scenario_results[scenario_id] = {
            "result": result,
            "impact_summary": impact_summary,
            "completed_at": datetime.now().isoformat(),
        }

        # 시나리오 상태 업데이트
        for scenario in self.simulation_scenarios:
            if scenario["id"] == scenario_id:
                scenario["status"] = "completed"
                break

    # === v17.0: OSDK Methods ===

    def register_object_type(
        self,
        type_id: str,
        name: str,
        properties: Dict[str, Any],
        source_tables: Optional[List[str]] = None
    ) -> None:
        """
        v17.0: OSDK Object Type 등록

        Args:
            type_id: 타입 식별자
            name: 타입 이름
            properties: 속성 정의
            source_tables: 소스 테이블 목록
        """
        self.osdk_object_types[type_id] = {
            "type_id": type_id,
            "name": name,
            "properties": properties,
            "source_tables": source_tables or [],
            "created_at": datetime.now().isoformat(),
        }

    def register_link_type(
        self,
        link_id: str,
        name: str,
        from_type: str,
        to_type: str,
        cardinality: str = "many-to-one"
    ) -> None:
        """v17.0: OSDK Link Type 등록"""
        self.osdk_link_types[link_id] = {
            "link_id": link_id,
            "name": name,
            "from_type": from_type,
            "to_type": to_type,
            "cardinality": cardinality,
            "created_at": datetime.now().isoformat(),
        }

    def get_osdk_schema(self) -> Dict[str, Any]:
        """v17.0: OSDK 스키마 전체 반환"""
        return {
            "object_types": self.osdk_object_types,
            "link_types": self.osdk_link_types,
            "instance_counts": {
                type_id: len(instances)
                for type_id, instances in self.osdk_instances.items()
            },
        }

    # === v17.0: Writeback Methods ===

    def queue_writeback(
        self,
        action_type: str,
        target_type: str,
        target_id: str,
        changes: Dict[str, Any],
        reason: str = ""
    ) -> str:
        """
        v17.0: Writeback 액션 큐에 추가

        Returns:
            생성된 writeback ID
        """
        writeback_id = f"wb_{len(self.pending_writebacks) + 1}"

        self.pending_writebacks.append({
            "id": writeback_id,
            "action_type": action_type,  # create, update, delete
            "target_type": target_type,
            "target_id": target_id,
            "changes": changes,
            "reason": reason,
            "queued_at": datetime.now().isoformat(),
            "status": "pending",
        })

        return writeback_id

    def execute_writeback(self, writeback_id: str) -> Dict[str, Any]:
        """
        v17.0: Writeback 액션 실행

        실제 구현은 writeback 모듈에서 처리
        """
        for wb in self.pending_writebacks:
            if wb["id"] == writeback_id:
                wb["status"] = "executed"
                wb["executed_at"] = datetime.now().isoformat()

                # 히스토리에 기록
                self.writeback_history.append(wb.copy())

                return {"success": True, "writeback": wb}

        return {"success": False, "error": "writeback_not_found"}

    def get_pending_writebacks(self) -> List[Dict[str, Any]]:
        """v17.0: 대기 중인 writeback 목록 반환"""
        return [wb for wb in self.pending_writebacks if wb["status"] == "pending"]

    # === v17.0: 서비스 접근 헬퍼 ===

    def get_v17_service(self, service_name: str) -> Any:
        """
        v17.0 서비스 접근 (에이전트용)

        Args:
            service_name: 서비스 이름
                - calibrator: ConfidenceCalibrator
                - version_manager: VersionManager
                - branch_manager: BranchManager
                - osdk_client: OSDKClient
                - writeback_engine: WritebackEngine
                - semantic_searcher: SemanticSearcher
                - remediation_engine: RemediationEngine
                - decision_explainer: DecisionExplainer
                - whatif_analyzer: WhatIfAnalyzer
                - report_generator: ReportGenerator
                - nl2sql_engine: NL2SQLEngine
                - tool_registry: ToolRegistry
                - streaming_reasoner: StreamingReasoner

        Returns:
            서비스 인스턴스 또는 None
        """
        services = self.get_dynamic("_v17_services", {})
        return services.get(service_name)

    def get_all_v17_services(self) -> Dict[str, Any]:
        """
        모든 v17.0 서비스 반환

        Returns:
            {service_name: service_instance} 딕셔너리
        """
        return self.get_dynamic("_v17_services", {})

    def has_v17_services(self) -> bool:
        """v17.0 서비스 활성화 여부"""
        services = self.get_dynamic("_v17_services", {})
        return bool(services)

    def calibrate_confidence_simple(self, raw_confidence: float, agent_id: str = "unknown") -> float:
        """
        v17.0: 신뢰도 보정 (에이전트용 간편 메서드)

        Note: 이것은 간편 버전입니다. 전체 기능은 calibrate_confidence()를 사용하세요.

        Args:
            raw_confidence: 원본 신뢰도
            agent_id: 에이전트 ID

        Returns:
            보정된 신뢰도
        """
        calibrator = self.get_v17_service("calibrator")
        if calibrator:
            try:
                result = calibrator.calibrate(raw_confidence, agent_id=agent_id)
                return result.calibrated_confidence
            except Exception:
                pass
        return raw_confidence

    # === v17.1: Unified Integration Methods (Palantir-style 연결) ===

    def add_unified_insight(self, insight: Dict[str, Any]) -> str:
        """
        v17.1: 통합 인사이트 추가

        Algorithm → Simulation → LLM 파이프라인에서 생성된 인사이트를 저장합니다.

        Args:
            insight: UnifiedInsight.to_dict() 결과

        Returns:
            인사이트 ID
        """
        if not insight.get("insight_id"):
            import uuid
            insight["insight_id"] = f"UI-{uuid.uuid4().hex[:8]}"

        self.unified_insights_pipeline.append(insight)
        return insight["insight_id"]

    def link_components(
        self,
        source_id: str,
        target_id: str,
        relation_type: str = "related_to"
    ) -> None:
        """
        v17.1: 컴포넌트 간 참조 연결

        온톨로지 개념, 인사이트, 액션 등을 서로 연결합니다.

        Args:
            source_id: 소스 컴포넌트 ID (예: concept_123)
            target_id: 타겟 컴포넌트 ID (예: UI-abc123)
            relation_type: 관계 유형 (related_insights, related_actions, etc.)
        """
        if source_id not in self.component_references:
            self.component_references[source_id] = {}

        if relation_type not in self.component_references[source_id]:
            self.component_references[source_id][relation_type] = []

        if target_id not in self.component_references[source_id][relation_type]:
            self.component_references[source_id][relation_type].append(target_id)

    def get_linked_components(
        self,
        component_id: str,
        relation_type: Optional[str] = None
    ) -> Dict[str, List[str]]:
        """
        v17.1: 특정 컴포넌트와 연결된 컴포넌트들 조회

        Args:
            component_id: 컴포넌트 ID
            relation_type: 특정 관계 유형만 조회 (옵션)

        Returns:
            관계 유형별 연결된 컴포넌트 ID 목록
        """
        refs = self.component_references.get(component_id, {})

        if relation_type:
            return {relation_type: refs.get(relation_type, [])}

        return refs

    def cache_llm_analysis(
        self,
        key: str,
        analysis_result: Dict[str, Any]
    ) -> None:
        """
        v17.1: LLM 분석 결과 캐시

        동일한 데이터에 대한 중복 LLM 호출을 방지합니다.

        Args:
            key: 캐시 키 (예: "table_name:column_name")
            analysis_result: LLM 분석 결과
        """
        analysis_result["cached_at"] = datetime.now().isoformat()
        self.llm_analysis_cache[key] = analysis_result

    def get_cached_llm_analysis(self, key: str) -> Optional[Dict[str, Any]]:
        """
        v17.1: 캐시된 LLM 분석 결과 조회

        Args:
            key: 캐시 키

        Returns:
            캐시된 분석 결과 또는 None
        """
        return self.llm_analysis_cache.get(key)

    def update_integration_metrics(
        self,
        metric_name: str,
        value: float
    ) -> None:
        """
        v17.1: 통합 메트릭 업데이트

        Args:
            metric_name: 메트릭 이름
            value: 메트릭 값 (0.0 - 1.0)
        """
        self.integration_metrics[metric_name] = value
        self.integration_metrics["last_updated"] = datetime.now().isoformat()

    def calculate_component_connectivity(self) -> float:
        """
        v17.1: 컴포넌트 연결성 계산

        전체 컴포넌트 중 다른 컴포넌트와 연결된 비율을 계산합니다.

        Returns:
            연결성 비율 (0.0 - 1.0)
        """
        total_components = (
            len(self.ontology_concepts) +
            len(self.unified_insights_pipeline) +
            len(self.action_backlog) +
            len(self.governance_decisions)
        )

        if total_components == 0:
            return 0.0

        connected_components = len(self.component_references)

        connectivity = connected_components / total_components
        self.update_integration_metrics("component_connectivity", connectivity)

        return connectivity

    def get_integration_summary(self) -> Dict[str, Any]:
        """
        v17.1: 통합 상태 요약

        Palantir 스타일의 전체 연결 상태를 요약합니다.

        Returns:
            통합 요약 정보
        """
        return {
            "unified_insights_count": len(self.unified_insights_pipeline),
            "component_references_count": len(self.component_references),
            "llm_cache_size": len(self.llm_analysis_cache),
            "integration_metrics": self.integration_metrics,
            "connectivity": self.calculate_component_connectivity(),
            "phases_connected": {
                "discovery_to_refinement": len(self.unified_entities) > 0 and len(self.ontology_concepts) > 0,
                "refinement_to_governance": len(self.ontology_concepts) > 0 and len(self.governance_decisions) > 0,
                "insights_to_actions": any(
                    "related_actions" in refs for refs in self.component_references.values()
                ),
            },
        }