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

    # === 현재 상태 ===
    current_stage: Optional[StageType] = None
    is_complete: bool = False

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
        self.tables[table_info.name] = table_info

    def add_homeomorphism(self, pair: HomeomorphismPair) -> None:
        """위상동형 쌍 추가"""
        self.homeomorphisms.append(pair)

    def add_unified_entity(self, entity: UnifiedEntity) -> None:
        """통합 엔티티 추가"""
        self.unified_entities.append(entity)

    def add_concept(self, concept: OntologyConcept) -> None:
        """온톨로지 개념 추가"""
        self.ontology_concepts.append(concept)

    def add_governance_decision(self, decision: GovernanceDecision) -> None:
        """거버넌스 결정 추가"""
        self.governance_decisions.append(decision)

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
        tables = []
        for entity in self.unified_entities:
            if entity.entity_type == entity_type:
                tables.extend(entity.source_tables)
        return list(set(tables))

    def get_homeomorphisms_for_table(self, table_name: str) -> List[HomeomorphismPair]:
        """특정 테이블과 관련된 위상동형 쌍"""
        return [
            h for h in self.homeomorphisms
            if h.table_a == table_name or h.table_b == table_name
        ]

    def get_concepts_by_type(self, concept_type: str) -> List[OntologyConcept]:
        """특정 타입의 온톨로지 개념"""
        return [c for c in self.ontology_concepts if c.concept_type == concept_type]

    def get_approved_concepts(self) -> List[OntologyConcept]:
        """승인된 온톨로지 개념"""
        return [c for c in self.ontology_concepts if c.status == "approved"]

    def get_pending_actions(self) -> List[Dict[str, Any]]:
        """대기 중인 액션"""
        return [a for a in self.action_backlog if a.get("status") == "pending"]

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
        self.knowledge_graph_triples.append({
            "subject": subject,
            "predicate": predicate,
            "object": obj,
            "confidence": confidence,
            "source": source,
        })

    def add_inferred_triple(
        self,
        subject: str,
        predicate: str,
        obj: str,
        confidence: float = 1.0,
        rule_applied: str = None
    ) -> None:
        """추론된 트리플 추가"""
        self.inferred_triples.append({
            "subject": subject,
            "predicate": predicate,
            "object": obj,
            "confidence": confidence,
            "source": "inferred",
            "rule_applied": rule_applied,
        })

    def get_all_triples(self) -> List[Dict[str, Any]]:
        """모든 트리플 (ontology + semantic + inferred + governance) 반환"""
        return (
            self.knowledge_graph_triples +
            self.semantic_base_triples +
            self.inferred_triples +
            self.governance_triples
        )

    def get_triple_count(self) -> Dict[str, int]:
        """트리플 통계"""
        return {
            "ontology_asserted": len(self.knowledge_graph_triples),
            "semantic_base": len(self.semantic_base_triples),
            "inferred": len(self.inferred_triples),
            "governance": len(self.governance_triples),
            "total": (
                len(self.knowledge_graph_triples) +
                len(self.semantic_base_triples) +
                len(self.inferred_triples) +
                len(self.governance_triples)
            ),
        }

    def build_knowledge_graph(self) -> Any:
        """
        KnowledgeGraphBuilder를 사용하여 전체 지식 그래프 빌드

        Returns:
            KnowledgeGraphBuilder 인스턴스
        """
        from .autonomous.analysis.knowledge_graph import KnowledgeGraphBuilder
        builder = KnowledgeGraphBuilder()
        return builder.build_from_context(self)

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
                    "evidence_chain_valid": self.evidence_chain.verify_chain_integrity().get("is_valid", False),
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
            # v6.1: Phase 3 Advanced Features
            "debate_results": self.debate_results,
            "counterfactual_analysis": self.counterfactual_analysis,
            # v11.0: Evidence Chain
            "evidence_chain_summary": self.get_evidence_summary(),
            "evidence_based_debate_results": self.evidence_based_debate_results,
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
        return self.data_directory

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
            # 폴백: 기본 컬럼 의미론 생성
            fallback_semantics = {}
            for table_name, table_info in self.tables.items():
                for col_info in table_info.columns:
                    col_name = col_info.get("name", "")
                    fallback_semantics[col_name] = {
                        "business_role": "unknown",
                        "nullable_by_design": True,  # 안전한 기본값
                    }
            if fallback_semantics:
                self.set_dynamic("column_semantics", fallback_semantics)
                fallbacks_applied.append("created_fallback_column_semantics")
                logger.warning(f"[v14.0 Gate] Created fallback column_semantics for {len(fallback_semantics)} columns")

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
