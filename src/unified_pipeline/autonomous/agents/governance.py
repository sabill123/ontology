"""
Governance Phase Autonomous Agents (v11.0 - Evidence Chain & BFT Debate)

Governance 단계 자율 에이전트들 - 블록체인 스타일 Evidence Chain + BFT 합의
- Governance Strategist: 거버넌스 전략 + Evidence-Based Council Debate
- Action Prioritizer: 액션 우선순위 (알고리즘 기반 우선순위 + LLM 보강)
- Risk Assessor: 리스크 평가 + CausalImpact + AnomalyDetector
- Policy Generator: 정책 생성 + KPIPredictor

v11.0 변경사항:
- Evidence Chain: Phase 1/2의 모든 분석 결과를 블록체인 스타일로 연결
- DebateProtocol: BFT 기반 Council 토론 (Challenge/Defense 라운드)
- 모든 거버넌스 결정에 근거 블록 ID 참조
- Council Agent들이 Phase 1/2 근거를 기반으로 투표

v4.5 변경사항 (유지):
- Legacy Phase 3 Agent Council 코드 직접 임베딩 (더 이상 _legacy 폴더 의존 없음)
- EmbeddedPhase3LLMJudge: 가중치 기반 평가 직접 구현
- 4개 에이전트 (RiskAssessor, PolicyValidator, QualityAuditor, ActionPlanner) 임베딩
- 모든 의존성 제거로 안정적인 실행 보장
"""

import json
import logging
import concurrent.futures
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from ..base import AutonomousAgent
from ..analysis import (
    DecisionMatrix,
    GovernanceRiskAssessor,
    PriorityCalculator,
    PolicyGenerator,
    parse_llm_json,
    # v2.1: Business Insights
    BusinessInsightsAnalyzer,
    # === 고급 분석 모듈 (v3.0 신규 통합) ===
    CausalImpactAnalyzer,  # 인과 추론
    AnomalyDetector,  # 앙상블 이상탐지
    KPIPredictor,  # 시계열 예측 + ROI
    # v3.1: 시계열 예측 (TimeSeriesForecaster)
    TimeSeriesForecaster,
    ForecastResult,
    # v3.2: 프로세스 분석 (ProcessArchitect 기능 통합)
    ProcessAnalyzer,
    ProcessContext,
    IntegrationComplexity,
    # === v6.1: Structured Debate ===
    StructuredDebateEngine,
    create_debate_summary,
    # === v6.1: Counterfactual Reasoning ===
    CounterfactualAnalyzer,
    CounterfactualResult,
)

# === v19.0: Unified Insight Pipeline ===
try:
    from ..analysis.unified_insight_pipeline import UnifiedInsightPipeline, PipelineResult
    UNIFIED_PIPELINE_AVAILABLE = True
except ImportError:
    UNIFIED_PIPELINE_AVAILABLE = False
    UnifiedInsightPipeline = None
    PipelineResult = None

# === v16.0: Column-level Lineage ===
try:
    from ...lineage import LineageTracker, build_lineage_from_context, ImpactAnalysis
    LINEAGE_AVAILABLE = True
except ImportError:
    LINEAGE_AVAILABLE = False
    LineageTracker = None
    build_lineage_from_context = None
    ImpactAnalysis = None

# === v16.0: Actions Engine ===
try:
    from ...actions import ActionsEngine, Action, Trigger, TriggerType, ActionType, ActionStatus
    ACTIONS_ENGINE_AVAILABLE = True
except ImportError:
    ACTIONS_ENGINE_AVAILABLE = False
    ActionsEngine = None
    Action = None
    Trigger = None
    TriggerType = None
    ActionType = None
    ActionStatus = None

# === v16.0: CDC Sync Engine ===
try:
    from ...sync import CDCEngine, create_cdc_engine_with_context, ChangeType, SyncStatus
    CDC_ENGINE_AVAILABLE = True
except ImportError:
    CDC_ENGINE_AVAILABLE = False
    CDCEngine = None
    create_cdc_engine_with_context = None
    ChangeType = None
    SyncStatus = None

# === v16.0: Agent Learning System ===
try:
    from ..learning import (
        AgentMemory,
        Experience,
        ExperienceType,
        Outcome,
        create_agent_memory_with_context,
    )
    AGENT_LEARNING_AVAILABLE = True
except ImportError:
    AGENT_LEARNING_AVAILABLE = False
    AgentMemory = None
    Experience = None
    ExperienceType = None
    Outcome = None
    create_agent_memory_with_context = None

# === v4.5: Embedded Phase 3 Agent Council from utils ===
from .governance_utils import (
    EmbeddedPhase3AgentRole,
    EMBEDDED_PHASE3_AGENT_CONFIG,
    EmbeddedEvaluationCriterion,
    EmbeddedPhase3EvaluationCriteria,
    EmbeddedPhase3LLMJudge,
    EmbeddedPhase3GovernanceOrchestrator,
    LEGACY_PHASE3_AVAILABLE,
)

# === v5.0: Enhanced Validation System ===
try:
    from ..analysis.enhanced_validator import (
        EnhancedValidator,
        StatisticalValidator,
        DempsterShaferFusion,
        ConfidenceCalibrator,
        KnowledgeGraphQualityValidator,
        BFTConsensus,
        AgentVote,
        VoteType,
    )
    ENHANCED_VALIDATOR_AVAILABLE = True
except ImportError:
    ENHANCED_VALIDATOR_AVAILABLE = False

# === v11.0: Evidence Chain & BFT Debate Protocol ===
try:
    from ..evidence_chain import EvidenceChain, EvidenceRegistry, EvidenceType
    from ..debate_protocol import DebateProtocol, CouncilAgent, VoteDecision
    EVIDENCE_CHAIN_AVAILABLE = True
except ImportError:
    EVIDENCE_CHAIN_AVAILABLE = False
    logger.warning("Evidence Chain module not available - falling back to legacy debate")


if TYPE_CHECKING:
    from ...todo.models import PipelineTodo, TodoResult
    from ...shared_context import SharedContext

logger = logging.getLogger(__name__)


class GovernanceStrategistAutonomousAgent(AutonomousAgent):
    """
    거버넌스 전략가 자율 에이전트 (v11.0 - Evidence Chain & BFT Debate)

    Algorithm-First 접근:
    1. 결정 매트릭스를 통한 알고리즘 기반 판단
    2. Evidence Chain: Phase 1/2 근거를 블록체인 스타일로 참조
    3. DebateProtocol: BFT 기반 Council 토론 (Challenge/Defense)
    4. 모든 거버넌스 결정에 근거 블록 ID 연결

    v11.0 변경사항:
    - Evidence Chain 통합: Phase 1/2 분석 결과를 근거로 활용
    - DebateProtocol: 4개 Council Agent (Ontologist, Risk, Business, Steward)
    - BFT 합의: Challenge/Defense 라운드를 통한 robust consensus
    - 거버넌스 결정에 cited_evidence 필드 추가

    v4.5~v4.6 변경사항 (유지):
    - EmbeddedPhase3LLMJudge 직접 사용
    - ProcessAnalyzer 통합
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # v6.0: LLM 클라이언트를 전달하여 동적 임계값 활성화
        self.decision_matrix = DecisionMatrix(llm_client=self.llm_client)
        self._thresholds_initialized = False

        # v9.5: Embedded LLM Judge - LLM 클라이언트 + 도메인 컨텍스트 전달
        self.embedded_llm_judge = EmbeddedPhase3LLMJudge(llm_client=self.llm_client)
        # v4.6: ProcessAnalyzer - 비즈니스 프로세스 컨텍스트 분석
        self.process_analyzer = ProcessAnalyzer()
        # v6.1: Structured Debate Engine - 라운드 기반 토론
        self.debate_engine = StructuredDebateEngine(
            llm_client=self.llm_client,
            max_rounds=3,
            consensus_threshold=0.7,
        )

        # v5.0: Enhanced Validation System
        # v17.0: Calibrator는 이제 베이스 클래스의 calibrate_confidence() 사용
        if ENHANCED_VALIDATOR_AVAILABLE:
            self.ds_fusion = DempsterShaferFusion()
            self.kg_validator = KnowledgeGraphQualityValidator()
            self.bft_consensus = BFTConsensus(n_agents=4)
            logger.info("GovernanceStrategist: Enhanced Validation (DS Fusion, KG Quality, BFT) enabled, v17 Calibrator integrated")

        # v11.0: Evidence-Based Debate Protocol
        self.debate_protocol = None
        if EVIDENCE_CHAIN_AVAILABLE:
            self.debate_protocol = DebateProtocol(llm_client=self.llm_client)
            logger.info("GovernanceStrategist: Evidence Chain & BFT Debate Protocol enabled")

    @property
    def agent_type(self) -> str:
        return "governance_strategist"

    @property
    def agent_name(self) -> str:
        return "Governance Strategist"

    @property
    def phase(self) -> str:
        return "governance"

    def get_system_prompt(self) -> str:
        return """You are a Data Governance Strategist with executive-level decision authority.

Your role is to REVIEW and VALIDATE algorithmic governance decisions.

v5.2 Decision Framework (More Permissive):
- APPROVE: Confidence ≥55%, evidence ≥40%, risk not high
  → Default to APPROVE when data is present and reasonable
- REJECT: Only for clearly problematic cases (quality <25%)
- ESCALATE: Medium confidence (35-55%) with high business value
- SCHEDULE_REVIEW: Provisional, needs periodic monitoring

Governance Criteria:
- Data Quality: Is underlying data present? (presence > perfection)
- Business Value: Does concept represent real business entity?
- Compliance: Basic regulatory awareness
- Sustainability: Can be improved iteratively?

IMPORTANT:
- PREFER APPROVE over REJECT when evidence exists
- Data discovery is exploratory - initial findings should be accepted provisionally
- Perfect is the enemy of good - approve reasonable concepts
- Base decisions on algorithmic scores. Override only with strong justification.

Respond ONLY with valid JSON."""

    async def execute_task(
        self,
        todo: "PipelineTodo",
        context: "SharedContext",
    ) -> "TodoResult":
        """거버넌스 전략 작업 실행"""
        from ...todo.models import TodoResult
        from ...shared_context import GovernanceDecision

        # v9.5: 도메인 컨텍스트를 Judge에 전달
        domain_context = context.get_domain_context() if hasattr(context, 'get_domain_context') else None
        if domain_context:
            self.embedded_llm_judge.set_domain_context(domain_context)
            logger.info(f"v9.5: Judge domain context set to: {getattr(domain_context, 'industry', 'general')}")

        # v6.0: 동적 임계값 초기화 (첫 실행 시)
        if not self._thresholds_initialized:
            self._report_progress(0.05, "Initializing dynamic thresholds via LLM")
            try:
                dynamic_thresholds = await self.decision_matrix.initialize_thresholds(context)
                logger.info(
                    f"v6.0 Dynamic thresholds: "
                    f"domain={dynamic_thresholds.domain_context}, "
                    f"risk={dynamic_thresholds.risk_tolerance}, "
                    f"reasoning={dynamic_thresholds.reasoning[:100]}..."
                )
                self._thresholds_initialized = True
            except Exception as e:
                logger.warning(f"Failed to initialize dynamic thresholds: {e}, using defaults")

        self._report_progress(0.1, "Computing algorithmic decisions")

        # 1단계: 알고리즘 기반 결정
        concepts_by_status = {"approved": [], "provisional": [], "rejected": [], "pending": []}
        algorithmic_decisions = {}

        for concept in context.ontology_concepts:
            status = concept.status or "pending"
            concept_dict = {
                "concept_id": concept.concept_id,
                "concept_type": concept.concept_type,
                "name": concept.name,
                "confidence": concept.confidence,
                "status": status,
                "source_evidence": concept.source_evidence,
                "source_tables": concept.source_tables,
            }
            concepts_by_status.get(status, concepts_by_status["pending"]).append(concept_dict)

            # 결정 매트릭스 적용
            decision = self.decision_matrix.make_decision(
                concept_dict,
                concept.agent_assessments,
            )
            algorithmic_decisions[concept.concept_id] = {
                "decision_type": decision.decision_type,
                "confidence": decision.confidence,
                "reasoning": decision.reasoning,
                "business_value": decision.business_value,
                "risk_level": decision.risk_level,
                "quality_score": decision.quality_score,
                "evidence_strength": decision.evidence_strength,
            }

        self._report_progress(0.30, f"Computed {len(algorithmic_decisions)} algorithmic decisions")

        # === v4.6: ProcessAnalyzer 분석 ===
        process_analyses = {}
        self._report_progress(0.35, "Running ProcessAnalyzer (business process context)")
        try:
            for concept in context.ontology_concepts:
                concept_id = concept.concept_id

                # OntologyConcept의 definition에서 properties와 relationships 추출
                definition = concept.definition or {}
                properties = definition.get("properties", [])
                relationships = definition.get("relationships", [])

                # 프로세스 컨텍스트 분석
                process_context = self.process_analyzer.identify_process_context(
                    entity_name=concept.name,
                    description=concept.description or "",
                    properties=[p.get("name", "") if isinstance(p, dict) else str(p) for p in properties],
                    source_tables=concept.source_tables,
                )

                # 통합 복잡도 평가
                integration_assessment = self.process_analyzer.assess_integration_complexity(
                    source_tables=concept.source_tables,
                    related_entities=[r.get("to", "") if isinstance(r, dict) else str(r) for r in relationships],
                    has_external_sources=False,
                )

                process_analyses[concept_id] = {
                    "process_context": process_context.to_dict(),
                    "integration_assessment": integration_assessment.to_dict(),
                    "automation_score": process_context.confidence * 100,
                }

            logger.info(f"ProcessAnalyzer analyzed {len(process_analyses)} concepts")
        except Exception as e:
            logger.warning(f"ProcessAnalyzer failed: {e}")

        # === v17.1: Phase 2 데이터 통합 (causal_relationships, palantir_insights) ===
        phase2_enrichment = {}
        self._report_progress(0.40, "Integrating Phase 2 insights (v17.1)")

        try:
            # 1. Causal Relationships 통합 - 개념 간 인과 관계 정보
            causal_map = {}  # concept_name → causal_info
            for causal in (context.causal_relationships or []):
                if isinstance(causal, dict):
                    source = causal.get("source", causal.get("from_concept", ""))
                    target = causal.get("target", causal.get("to_concept", ""))
                    effect = causal.get("effect", causal.get("causal_effect", 0.0))
                    if source:
                        if source not in causal_map:
                            causal_map[source] = {"causes": [], "caused_by": []}
                        causal_map[source]["causes"].append({
                            "target": target,
                            "effect": effect,
                            "confidence": causal.get("confidence", 0.5)
                        })
                    if target:
                        if target not in causal_map:
                            causal_map[target] = {"causes": [], "caused_by": []}
                        causal_map[target]["caused_by"].append({
                            "source": source,
                            "effect": effect,
                            "confidence": causal.get("confidence", 0.5)
                        })

            # 2. Palantir Insights 통합 - 예측 인사이트
            insights_by_table = {}  # table_name → [insights]
            for insight in (context.palantir_insights or []):
                if isinstance(insight, dict):
                    tables = insight.get("source_tables", insight.get("related_tables", []))
                    for table in tables:
                        if table not in insights_by_table:
                            insights_by_table[table] = []
                        insights_by_table[table].append({
                            "title": insight.get("title", "Insight"),
                            "severity": insight.get("severity", "medium"),
                            "confidence": insight.get("confidence", 0.5),
                            "business_impact": insight.get("business_impact", ""),
                        })

            # 3. Business Insights 통합 - Phase 2에서 이미 생성된 비즈니스 인사이트
            existing_insights_by_table = {}
            for bi in (context.business_insights or []):
                if isinstance(bi, dict):
                    table = bi.get("source_table", bi.get("table", ""))
                    if table:
                        if table not in existing_insights_by_table:
                            existing_insights_by_table[table] = []
                        existing_insights_by_table[table].append({
                            "title": bi.get("title", ""),
                            "severity": bi.get("severity", "medium"),
                            "description": bi.get("description", ""),
                        })

            # 4. 각 개념에 Phase 2 정보 연결
            for concept in context.ontology_concepts:
                concept_name = concept.name
                concept_tables = concept.source_tables or []

                enrichment = {
                    "causal_info": causal_map.get(concept_name, {"causes": [], "caused_by": []}),
                    "related_insights": [],
                    "risk_factors_from_insights": [],
                }

                # 테이블 기반 인사이트 수집
                for table in concept_tables:
                    enrichment["related_insights"].extend(insights_by_table.get(table, []))
                    enrichment["related_insights"].extend(existing_insights_by_table.get(table, []))

                # 높은 심각도 인사이트를 리스크 요소로 추출
                for insight in enrichment["related_insights"]:
                    if insight.get("severity") in ("critical", "high"):
                        enrichment["risk_factors_from_insights"].append(insight.get("title", "Risk"))

                phase2_enrichment[concept.concept_id] = enrichment

            # 5. v22.0: Knowledge Graph Triples → 개념별 온톨로지 관계 매핑
            kg_triples = context.knowledge_graph_triples or []
            kg_triple_map = {}
            for triple in kg_triples:
                if isinstance(triple, dict):
                    subject = triple.get("subject", "")
                    concept_id = subject.split(":", 1)[1] if ":" in subject else subject
                    if concept_id not in kg_triple_map:
                        kg_triple_map[concept_id] = []
                    kg_triple_map[concept_id].append({
                        "predicate": triple.get("predicate", ""),
                        "object": triple.get("object", ""),
                    })

            # KG triples을 enrichment에 추가
            for concept in context.ontology_concepts:
                cid = concept.concept_id
                if cid in phase2_enrichment:
                    phase2_enrichment[cid]["kg_triples"] = kg_triple_map.get(cid, [])
                    phase2_enrichment[cid]["kg_triple_count"] = len(kg_triple_map.get(cid, []))

            logger.info(f"[v17.1] Phase 2 enrichment: {len(causal_map)} causal nodes, "
                       f"{sum(len(v) for v in insights_by_table.values())} insights, "
                       f"{len(kg_triples)} KG triples ({len(kg_triple_map)} concepts mapped)")

        except Exception as e:
            logger.warning(f"[v17.1] Phase 2 integration failed: {e}")

        # === v22.0: OWL2/SHACL + Calibration + v17 서비스 통합 ===
        owl2_shacl_context = {}
        v17_calibration_context = {}
        try:
            # OWL2 추론 결과 (Phase 2에서 context_updates로 전파됨)
            owl2_results = context.get_dynamic("owl2_reasoning_results", {})
            shacl_results = context.get_dynamic("shacl_validation_results", {})
            if owl2_results or shacl_results:
                owl2_shacl_context = {
                    "owl2_axioms_discovered": owl2_results.get("axioms_discovered", 0),
                    "owl2_inferred_classes": owl2_results.get("inferred_classes", []),
                    "owl2_property_chains": owl2_results.get("property_chains", []),
                    "shacl_is_conformant": shacl_results.get("is_conformant", True),
                    "shacl_violations_count": shacl_results.get("violations_count", 0),
                    "shacl_violations": shacl_results.get("violations", [])[:10],
                }
                logger.info(
                    f"[v22.0] OWL2/SHACL context: {owl2_shacl_context.get('owl2_axioms_discovered', 0)} axioms, "
                    f"conformant={owl2_shacl_context.get('shacl_is_conformant')}"
                )

            # v17 Calibrator 서비스에서 보정된 신뢰도 데이터
            v17_services = context.get_dynamic("_v17_services", {})
            calibrator = v17_services.get("calibrator")
            if calibrator:
                try:
                    calibration_summary = calibrator.get_calibration_summary()
                    v17_calibration_context = {
                        "calibration_available": True,
                        "calibration_summary": calibration_summary,
                    }
                    logger.info(f"[v22.0] Calibration data integrated into governance")
                except Exception as cal_err:
                    logger.debug(f"[v22.0] Calibration summary failed: {cal_err}")

        except Exception as e:
            logger.debug(f"[v22.0] OWL2/SHACL/Calibration integration skipped: {e}")

        # === v22.0: QualityJudge 평가 결과 통합 ===
        quality_context = {}
        try:
            quality_summary = context.get_dynamic("quality_assessment_summary", {})
            quality_stats = context.get_dynamic("quality_enhanced_stats", {})
            if quality_summary:
                quality_context = {
                    "total_evaluated": quality_summary.get("total_evaluated", 0),
                    "approved": quality_summary.get("approved", 0),
                    "provisional": quality_summary.get("provisional", 0),
                    "rejected": quality_summary.get("rejected", 0),
                    "avg_score": quality_summary.get("avg_score", 0),
                }
                if quality_stats:
                    quality_context["calibrated_confidence"] = quality_stats.get("avg_calibrated_confidence", 0)
                    quality_context["bft_agreement"] = quality_stats.get("avg_bft_agreement", 0)
                logger.info(
                    f"[v22.0] Quality context: {quality_context.get('total_evaluated', 0)} evaluated, "
                    f"approved={quality_context.get('approved', 0)}, rejected={quality_context.get('rejected', 0)}"
                )
        except Exception as e:
            logger.debug(f"[v22.0] Quality assessment integration skipped: {e}")

        # === v5.0: Enhanced Validation System ===
        enhanced_validation_results = {}
        if ENHANCED_VALIDATOR_AVAILABLE:
            self._report_progress(0.38, "Running Enhanced Validation (DS Fusion, BFT, Calibration)")

            for concept in context.ontology_concepts:
                concept_id = concept.concept_id
                algo_decision = algorithmic_decisions.get(concept_id, {})

                # 다중 에이전트 의견 생성 (Process, Algorithm, Quality, Risk 가상 에이전트)
                opinions = [
                    {
                        'decision_type': algo_decision.get('decision_type', 'review'),
                        'confidence': algo_decision.get('confidence', 0.5),
                        'agent': 'algorithmic'
                    },
                    {
                        'decision_type': 'approve' if algo_decision.get('quality_score', 50) > 50 else 'review',
                        'confidence': algo_decision.get('quality_score', 50) / 100,
                        'agent': 'quality'
                    },
                    {
                        'decision_type': 'approve' if algo_decision.get('evidence_strength', 50) > 50 else 'review',
                        'confidence': algo_decision.get('evidence_strength', 50) / 100,
                        'agent': 'evidence'
                    },
                    {
                        'decision_type': 'reject' if algo_decision.get('risk_level', 'medium') == 'high' else 'approve',
                        'confidence': 0.3 if algo_decision.get('risk_level') == 'high' else 0.7,
                        'agent': 'risk'
                    },
                ]

                # Dempster-Shafer Fusion
                ds_result = self.ds_fusion.fuse_opinions(opinions)

                # BFT Consensus
                votes = [
                    AgentVote(
                        agent_id=op['agent'],
                        vote=VoteType.APPROVE if 'approve' in op['decision_type'] else
                             VoteType.REJECT if 'reject' in op['decision_type'] else VoteType.REVIEW,
                        confidence=op['confidence'],
                    )
                    for op in opinions
                ]
                bft_result = self.bft_consensus.reach_consensus(votes)

                # Confidence Calibration (v17.0: 베이스 클래스 메서드 사용)
                raw_confidence = (ds_result['confidence'] + bft_result['confidence']) / 2
                calibrated = self.calibrate_confidence(raw_confidence)

                enhanced_validation_results[concept_id] = {
                    'ds_decision': ds_result['decision'],
                    'ds_confidence': ds_result['confidence'],
                    'ds_conflict': ds_result['conflict'],
                    'bft_consensus': bft_result['consensus'].value,
                    'bft_confidence': bft_result['confidence'],
                    'bft_agreement': bft_result['agreement_ratio'],
                    'byzantine_detected': bft_result.get('byzantine_detected', []),
                    'calibrated_confidence': calibrated['calibrated'],
                    'confidence_interval': calibrated['confidence_interval'],
                }

            logger.info(f"Enhanced Validation: {len(enhanced_validation_results)} concepts validated")

        # === v4.5: Embedded LLM Judge 평가 ===
        embedded_evaluations = {}
        self._report_progress(0.45, "Running Embedded LLM Judge (multi-criteria evaluation)")
        try:
            for concept in context.ontology_concepts:
                concept_id = concept.concept_id
                algo_decision = algorithmic_decisions.get(concept_id, {})

                # 인사이트 형식 생성
                insight_data = {
                    "insight_id": concept_id,
                    "confidence": concept.confidence,
                    "severity": "high" if concept.confidence > 0.7 else "medium",
                    "explanation": concept.description or concept.name,
                    "source_signatures": concept.source_tables,
                }

                # 결정 형식 생성 (v8.3: enhanced_validation 포함)
                enhanced_val = enhanced_validation_results.get(concept_id, {}) if ENHANCED_VALIDATOR_AVAILABLE else {}
                decision_data = {
                    "decision_id": f"decision_{concept_id}",
                    "decision_type": algo_decision.get("decision_type", "pending"),
                    "confidence": algo_decision.get("confidence", 0.5),
                    "reasoning": algo_decision.get("reasoning", ""),
                    "enhanced_validation": enhanced_val,  # v8.3: 경량 검증 지원
                }

                # Embedded LLM Judge 평가 (규칙 기반)
                # v18.0: asyncio.to_thread로 블로킹 호출 방지
                import asyncio
                evaluation = await asyncio.to_thread(
                    self.embedded_llm_judge.evaluate_governance_decision,
                    insight_data,
                    decision_data,
                    [],  # agent_opinions
                )
                embedded_evaluations[concept_id] = evaluation

            logger.info(f"Embedded LLM Judge evaluated {len(embedded_evaluations)} decisions")
        except Exception as e:
            logger.warning(f"Embedded LLM Judge failed: {e}")

        # === v11.0: Evidence-Based Council Debate ===
        debate_results = {}
        evidence_based_debate_results = {}
        self._report_progress(0.50, "Running Evidence-Based Council Debate (v11.0)")

        # v11.0: Evidence Chain에서 Phase 1/2 근거 조회
        phase1_evidence = []
        phase2_evidence = []
        evidence_chain = None

        if EVIDENCE_CHAIN_AVAILABLE and self.debate_protocol:
            try:
                # EvidenceRegistry 싱글톤에서 직접 가져오기 (SharedContext 대신)
                from ..evidence_chain import EvidenceRegistry
                registry = EvidenceRegistry()
                evidence_chain = registry.chain

                if evidence_chain and len(evidence_chain.blocks) > 0:
                    phase1_evidence = self.get_evidence_for_reasoning(
                        phases=["phase1_discovery"],
                        min_confidence=0.5
                    )
                    phase2_evidence = self.get_evidence_for_reasoning(
                        phases=["phase2_refinement"],
                        min_confidence=0.5
                    )
                    logger.info(f"v11.0: Retrieved {len(phase1_evidence)} Phase 1, {len(phase2_evidence)} Phase 2 evidence blocks from {len(evidence_chain.blocks)} total blocks")
                else:
                    logger.warning("v11.0: Evidence Chain is empty or not initialized")
            except Exception as e:
                logger.warning(f"v11.0: Failed to retrieve evidence chain: {e}")

        try:
            # 토론 대상 선정: confidence < 0.85 또는 risk가 있는 것들
            debate_candidates = []
            for concept in context.ontology_concepts[:10]:
                algo_decision = algorithmic_decisions.get(concept.concept_id, {})
                if (algo_decision.get("confidence", 1) < 0.85 or
                    algo_decision.get("risk_level") in ["high", "medium", "low"]):
                    debate_candidates.append(concept)

            # 최소 2개 후보 보장
            if len(debate_candidates) < 2 and len(context.ontology_concepts) >= 2:
                for concept in context.ontology_concepts[:3]:
                    if concept not in debate_candidates:
                        debate_candidates.append(concept)
                        if len(debate_candidates) >= 2:
                            break

            for concept in debate_candidates[:5]:
                algo_decision = algorithmic_decisions.get(concept.concept_id, {})

                # === v11.0: Evidence-Based Debate Protocol ===
                if EVIDENCE_CHAIN_AVAILABLE and self.debate_protocol and evidence_chain:
                    try:
                        # 해당 개념과 관련된 근거 찾기
                        concept_evidence = []
                        for ev in (phase1_evidence + phase2_evidence):
                            # 테이블명이나 개념명이 관련된 근거 수집
                            if any(t in ev.get("data_references", []) for t in concept.source_tables):
                                concept_evidence.append(ev)
                            elif concept.name.lower() in ev.get("finding", "").lower():
                                concept_evidence.append(ev)

                        # 관련 근거 블록 ID 수집
                        supporting_block_ids = [ev["block_id"] for ev in concept_evidence[:10]]

                        # Proposal 구성
                        proposal = {
                            "concept_id": concept.concept_id,
                            "concept_name": concept.name,
                            "concept_type": concept.concept_type,
                            "proposed_decision": algo_decision.get("decision_type", "approve"),
                            "confidence": algo_decision.get("confidence", 0.5),
                            "evidence_strength": algo_decision.get("evidence_strength", 50),
                            "quality_score": algo_decision.get("quality_score", 50),
                            "risk_level": algo_decision.get("risk_level", "unknown"),
                            "source_tables": concept.source_tables,
                            "description": concept.description or "",
                            # v22.0: OWL2/SHACL + Calibration + Quality 컨텍스트 추가
                            "owl2_shacl_context": owl2_shacl_context if owl2_shacl_context else None,
                            "calibration_context": v17_calibration_context if v17_calibration_context else None,
                            "quality_context": quality_context if quality_context else None,
                        }

                        # BFT 기반 Council Debate 수행 (동기 함수)
                        debate_result = self.debate_protocol.conduct_debate(
                            proposal_id=f"proposal_{concept.concept_id}",
                            proposal=proposal,
                            evidence_chain=evidence_chain,
                            context={
                                "domain": context.get_industry() if hasattr(context, 'get_industry') else "general",
                                "supporting_evidence": concept_evidence[:5],
                                "algorithmic_decision": algo_decision,
                            },
                        )

                        # DebateResult의 실제 필드명 사용
                        final_votes = list(debate_result.votes.values())
                        evidence_based_debate_results[concept.concept_id] = {
                            "verdict": debate_result.final_decision.value,
                            "confidence": debate_result.confidence,
                            "consensus_reached": debate_result.consensus_reached,
                            "consensus_type": debate_result.consensus_type,
                            "rounds_completed": debate_result.rounds_conducted,
                            "cited_evidence": supporting_block_ids,
                            "vote_summary": {
                                "approve": sum(1 for v in final_votes if v.decision == VoteDecision.APPROVE),
                                "reject": sum(1 for v in final_votes if v.decision == VoteDecision.REJECT),
                                "review": sum(1 for v in final_votes if v.decision == VoteDecision.REVIEW),
                                "abstain": sum(1 for v in final_votes if v.decision == VoteDecision.ABSTAIN),
                            },
                            "agent_votes": [
                                {
                                    "agent": v.agent_name,
                                    "decision": v.decision.value,
                                    "reasoning": v.reasoning[:200] if v.reasoning else "",
                                    "cited_evidence": v.cited_evidence[:3],
                                }
                                for v in final_votes
                            ],
                        }
                        logger.info(
                            f"v11.0 Evidence-Based Debate for {concept.concept_id}: "
                            f"verdict={debate_result.final_decision.value}, "
                            f"consensus={debate_result.consensus_reached}, "
                            f"cited_evidence={len(supporting_block_ids)}"
                        )
                        continue  # Evidence-based debate 성공 시 legacy skip
                    except Exception as e:
                        logger.warning(f"v11.0 Evidence-Based Debate failed for {concept.concept_id}: {e}")
                        # Fallback to legacy debate

                # === Fallback: v6.1 Legacy Structured Debate ===
                topic = f"Governance decision for '{concept.name}' ({concept.concept_type})"
                debate_context = {
                    "concept_id": concept.concept_id,
                    "name": concept.name,
                    "description": concept.description,
                    "confidence": concept.confidence,
                    "source_tables": concept.source_tables,
                }

                perspectives = {
                    "proposer": f"This concept has {algo_decision.get('evidence_strength', 50)}% evidence strength",
                    "opponent": f"Risk level is {algo_decision.get('risk_level', 'unknown')}",
                    "analyst": f"Quality score: {algo_decision.get('quality_score', 50)}%",
                }

                evidence = {
                    "quality_score": algo_decision.get("quality_score", 50) / 100,
                    "confidence": algo_decision.get("confidence", 0.5),
                    "risk_level": 0.3 if algo_decision.get("risk_level") == "low" else 0.6,
                    "data_sources": concept.source_tables,
                }

                outcome = await self.debate_engine.conduct_debate(
                    topic=topic,
                    context=debate_context,
                    agents_perspectives=perspectives,
                    evidence=evidence,
                )

                debate_results[concept.concept_id] = create_debate_summary(outcome)
                logger.info(f"Legacy Debate for {concept.concept_id}: verdict={outcome.verdict.value}, confidence={outcome.confidence:.2f}")

            logger.info(
                f"Council Debate completed: {len(evidence_based_debate_results)} evidence-based, "
                f"{len(debate_results)} legacy"
            )
        except Exception as e:
            logger.warning(f"Council Debate failed: {e}")

        # 2단계: LLM을 통한 검토 및 보강
        self._report_progress(0.60, "Requesting LLM review of decisions")

        # 검토가 필요한 항목만 LLM에게 전달
        needs_review = [
            {**concepts_by_status[status][i], "algorithmic_decision": algorithmic_decisions.get(concepts_by_status[status][i]["concept_id"])}
            for status in ["approved", "provisional"]
            for i in range(len(concepts_by_status[status]))
        ]

        # v9.0: Fallback - needs_review가 비어있으면 pending 상태의 concepts도 검토
        # Quality Judge가 실행되지 않은 경우에도 LLM Review가 동작하도록 보장
        if not needs_review and concepts_by_status.get("pending"):
            logger.warning("v9.0 Fallback: No approved/provisional concepts, reviewing pending concepts")
            needs_review = [
                {**concepts_by_status["pending"][i], "algorithmic_decision": algorithmic_decisions.get(concepts_by_status["pending"][i]["concept_id"])}
                for i in range(len(concepts_by_status["pending"]))
            ]
            # pending을 provisional로 임시 승격
            for item in needs_review:
                if item.get("algorithmic_decision"):
                    item["algorithmic_decision"]["decision_type"] = "provisional"
                    item["algorithmic_decision"]["reasoning"] = f"[v9.0 Fallback] {item['algorithmic_decision'].get('reasoning', 'Auto-promoted for review')}"

        enhanced_decisions = await self._llm_review_decisions(needs_review, context)

        self._report_progress(0.8, "Creating governance decision objects")

        # 3단계: GovernanceDecision 객체 생성
        governance_decisions = []
        for concept_id, algo_decision in algorithmic_decisions.items():
            # LLM 보강 결과 병합
            llm_enhancement = enhanced_decisions.get(concept_id, {})

            # v4.5: Embedded LLM Judge 평가 결과 병합
            embedded_eval = embedded_evaluations.get(concept_id, {})

            # v5.0: Enhanced Validation 결과 병합
            enhanced_val = enhanced_validation_results.get(concept_id, {}) if ENHANCED_VALIDATOR_AVAILABLE else {}

            # v4.5: Embedded LLM Judge 검증 결과 반영
            # Judge가 pass=False면 최종 결정을 schedule_review로 변경
            judge_assessment = embedded_eval.get("overall_assessment", {})
            judge_passed = judge_assessment.get("pass", True)  # 기본값은 통과
            judge_score = judge_assessment.get("percentage", 100)

            # 최종 decision_type 결정
            proposed_decision = llm_enhancement.get("decision_type", algo_decision["decision_type"])
            proposed_confidence = llm_enhancement.get("confidence", algo_decision["confidence"])

            # v5.0: Enhanced Validation 기반 신뢰도 조정
            if enhanced_val:
                calibrated_conf = enhanced_val.get('calibrated_confidence', proposed_confidence)
                bft_agreement = enhanced_val.get('bft_agreement', 1.0)
                ds_conflict = enhanced_val.get('ds_conflict', 0.0)

                # BFT 합의율이 낮거나 DS 충돌이 높으면 신뢰도 하향
                if bft_agreement < 0.5 or ds_conflict > 0.5:
                    proposed_confidence = min(proposed_confidence, calibrated_conf * 0.8)
                    logger.info(f"Enhanced Validation adjusted confidence for {concept_id}: "
                               f"BFT={bft_agreement:.2f}, DS_conflict={ds_conflict:.2f}")
                else:
                    # 검증 통과시 calibrated confidence 사용
                    proposed_confidence = calibrated_conf

            # v9.1: Enhanced Validation이 강하면 Judge 결과 재고
            strong_enhanced_validation = (
                enhanced_val and
                enhanced_val.get('ds_confidence', 0) >= 0.95 and
                enhanced_val.get('bft_agreement', 0) >= 0.95 and
                enhanced_val.get('ds_conflict', 1) == 0
            )

            if not judge_passed and proposed_decision == "approve":
                if strong_enhanced_validation:
                    # v9.1: Enhanced Validation이 매우 강하면 Judge override 무시
                    final_decision_type = proposed_decision
                    final_confidence = enhanced_val.get('calibrated_confidence', proposed_confidence)
                    final_reasoning = (
                        f"[v9.1 Enhanced Override] Judge 검증 실패 ({judge_score}%)했으나 "
                        f"Enhanced Validation이 강함 (DS={enhanced_val.get('ds_confidence', 0):.0%}, "
                        f"BFT={enhanced_val.get('bft_agreement', 0):.0%}). "
                        f"사유: {llm_enhancement.get('reasoning', algo_decision['reasoning'])}"
                    )
                    logger.info(f"v9.1 Enhanced override for {concept_id}: keeping {proposed_decision} despite judge failure")
                else:
                    # Judge 검증 실패 시: approve → schedule_review로 변경
                    final_decision_type = "schedule_review"
                    final_confidence = min(proposed_confidence, 0.6)  # confidence 제한
                    final_reasoning = (
                        f"[Judge Override] Embedded LLM Judge 검증 실패 ({judge_score}%). "
                        f"원래 결정: {proposed_decision}. "
                        f"사유: {llm_enhancement.get('reasoning', algo_decision['reasoning'])}"
                    )
                    logger.info(f"Judge override for {concept_id}: {proposed_decision} → {final_decision_type}")
            else:
                final_decision_type = proposed_decision
                final_confidence = proposed_confidence
                final_reasoning = llm_enhancement.get("reasoning", algo_decision["reasoning"])

            # v11.0: Evidence-Based Debate 결과 반영
            evidence_debate = evidence_based_debate_results.get(concept_id, {})
            if evidence_debate:
                # Evidence-based debate 결과로 decision_type 재평가
                debate_verdict = evidence_debate.get("verdict", "").lower()
                debate_confidence = evidence_debate.get("confidence", 0.5)
                consensus_reached = evidence_debate.get("consensus_reached", False)

                # 합의 도달 시 debate 결과 반영
                if consensus_reached:
                    if debate_verdict == "approve" and final_decision_type != "approve":
                        # Council이 승인하면 승인으로 변경
                        final_decision_type = "approve"
                        final_confidence = max(final_confidence, debate_confidence)
                        final_reasoning = (
                            f"[v11.0 Council Consensus] 4개 에이전트 Council 토론에서 승인 합의. "
                            f"근거 블록: {len(evidence_debate.get('cited_evidence', []))}개 참조. "
                            f"원래 사유: {final_reasoning[:200]}"
                        )
                    elif debate_verdict == "reject" and final_decision_type == "approve":
                        # Council이 거부하면 schedule_review로 변경
                        final_decision_type = "schedule_review"
                        final_confidence = min(final_confidence, 0.6)
                        final_reasoning = (
                            f"[v11.0 Council Override] 4개 에이전트 Council 토론에서 거부 합의. "
                            f"근거 블록: {len(evidence_debate.get('cited_evidence', []))}개 참조. "
                            f"원래 사유: {final_reasoning[:200]}"
                        )

                logger.info(
                    f"v11.0 Decision updated for {concept_id}: "
                    f"verdict={debate_verdict}, consensus={consensus_reached}, "
                    f"final_type={final_decision_type}"
                )

            # v17.1: Phase 2 enrichment data 가져오기
            p2_enrichment = phase2_enrichment.get(concept_id, {})

            # v17.1: 인과 관계 및 인사이트 기반 reasoning 보강
            if p2_enrichment.get("causal_info", {}).get("causes") or p2_enrichment.get("causal_info", {}).get("caused_by"):
                causal_causes = len(p2_enrichment.get("causal_info", {}).get("causes", []))
                causal_effects = len(p2_enrichment.get("causal_info", {}).get("caused_by", []))
                final_reasoning += f" [v17.1: Causal - causes {causal_causes}, caused_by {causal_effects}]"

            if p2_enrichment.get("risk_factors_from_insights"):
                risk_factors = p2_enrichment.get("risk_factors_from_insights", [])[:3]
                final_reasoning += f" [v17.1: Risk factors from insights: {', '.join(risk_factors)}]"

            # v22.0: OWL2/SHACL 위반 시 confidence 보정
            if owl2_shacl_context and not owl2_shacl_context.get("shacl_is_conformant", True):
                violations = owl2_shacl_context.get("shacl_violations_count", 0)
                if violations > 0:
                    final_confidence = min(final_confidence, 0.7)
                    final_reasoning += f" [v22.0: SHACL non-conformant ({violations} violations) - confidence capped]"

            decision = GovernanceDecision(
                decision_id=f"decision_{concept_id}",
                concept_id=concept_id,
                decision_type=final_decision_type,
                confidence=final_confidence,
                reasoning=final_reasoning,
                agent_opinions={
                    "algorithmic": algo_decision,
                    "llm_review": llm_enhancement,
                    "embedded_llm_judge": embedded_eval,  # v4.5
                    "enhanced_validation": enhanced_val,  # v5.0
                    "evidence_based_debate": evidence_debate,  # v11.0
                    "phase2_enrichment": p2_enrichment,  # v17.1: Phase 2 데이터 통합
                    "owl2_shacl": owl2_shacl_context,  # v22.0: OWL2/SHACL 온톨로지 추론
                    "calibration": v17_calibration_context,  # v22.0: 신뢰도 보정
                    "quality_assessment": quality_context,  # v22.0: QualityJudge 평가 결과
                },
                recommended_actions=llm_enhancement.get("recommended_actions", []),
            )
            governance_decisions.append(decision)

            # v11.0: 거버넌스 결정을 Evidence로 기록
            if EVIDENCE_CHAIN_AVAILABLE and evidence_debate:
                try:
                    self.record_evidence(
                        finding=f"거버넌스 결정: {concept_id} → {final_decision_type}",
                        reasoning=f"Council 토론 결과 (합의: {consensus_reached}). "
                                  f"투표: approve={evidence_debate.get('vote_summary', {}).get('approve', 0)}, "
                                  f"reject={evidence_debate.get('vote_summary', {}).get('reject', 0)}",
                        conclusion=final_reasoning[:300],
                        evidence_type="governance_decision",
                        data_references=[concept_id] + evidence_debate.get("cited_evidence", [])[:5],
                        metrics={
                            "decision_type": final_decision_type,
                            "confidence": final_confidence,
                            "consensus_reached": consensus_reached,
                            "vote_summary": evidence_debate.get("vote_summary", {}),
                        },
                        supporting_block_ids=evidence_debate.get("cited_evidence", [])[:10],
                        confidence=final_confidence,
                        topics=["governance", concept_id, final_decision_type],
                    )
                except Exception as e:
                    logger.warning(f"Failed to record governance evidence for {concept_id}: {e}")

        self._report_progress(0.95, "Finalizing governance strategy")

        # 결과 집계
        summary = {
            "approved": len([d for d in governance_decisions if d.decision_type == "approve"]),
            "rejected": len([d for d in governance_decisions if d.decision_type == "reject"]),
            "escalated": len([d for d in governance_decisions if d.decision_type == "escalate"]),
            "scheduled_review": len([d for d in governance_decisions if d.decision_type == "schedule_review"]),
        }

        # v4.3: Knowledge Graph 트리플 생성 (Governance 결정)
        governance_triples = self._generate_governance_triples(governance_decisions, context)
        logger.info(f"Generated {len(governance_triples)} knowledge graph triples from governance decisions")

        # v5.0: Enhanced Validation 통계
        enhanced_stats = {}
        if ENHANCED_VALIDATOR_AVAILABLE and enhanced_validation_results:
            avg_calibrated = sum(
                v.get('calibrated_confidence', 0.5) for v in enhanced_validation_results.values()
            ) / max(len(enhanced_validation_results), 1)
            avg_ds_conflict = sum(
                v.get('ds_conflict', 0) for v in enhanced_validation_results.values()
            ) / max(len(enhanced_validation_results), 1)
            avg_bft_agreement = sum(
                v.get('bft_agreement', 0) for v in enhanced_validation_results.values()
            ) / max(len(enhanced_validation_results), 1)
            byzantine_count = sum(
                len(v.get('byzantine_detected', [])) for v in enhanced_validation_results.values()
            )

            enhanced_stats = {
                'avg_calibrated_confidence': round(avg_calibrated, 4),
                'avg_ds_conflict': round(avg_ds_conflict, 4),
                'avg_bft_agreement': round(avg_bft_agreement, 4),
                'byzantine_agents_detected': byzantine_count,
            }
            logger.info(f"Enhanced Validation Stats: calibrated={avg_calibrated:.4f}, "
                       f"conflict={avg_ds_conflict:.4f}, agreement={avg_bft_agreement:.4f}")

        # v11.0: Evidence-Based Debate 통계
        evidence_debate_stats = {}
        if evidence_based_debate_results:
            total_debates = len(evidence_based_debate_results)
            consensus_count = sum(1 for d in evidence_based_debate_results.values() if d.get("consensus_reached"))
            avg_confidence = sum(d.get("confidence", 0) for d in evidence_based_debate_results.values()) / max(total_debates, 1)
            total_evidence_cited = sum(len(d.get("cited_evidence", [])) for d in evidence_based_debate_results.values())

            evidence_debate_stats = {
                "total_debates": total_debates,
                "consensus_reached_count": consensus_count,
                "consensus_rate": round(consensus_count / max(total_debates, 1), 4),
                "avg_confidence": round(avg_confidence, 4),
                "total_evidence_cited": total_evidence_cited,
                "phase1_evidence_used": len(phase1_evidence),
                "phase2_evidence_used": len(phase2_evidence),
            }
            logger.info(
                f"v11.0 Evidence-Based Debate Stats: "
                f"debates={total_debates}, consensus={consensus_count}, "
                f"evidence_cited={total_evidence_cited}"
            )

        # === v17.1: What-If Analyzer 통합 - 승인된 결정의 영향도 시뮬레이션 ===
        whatif_results = {"scenarios_analyzed": 0, "total_impact_score": 0.0}
        try:
            whatif_analyzer = self.get_v17_service("whatif_analyzer")
            if whatif_analyzer:
                approved_decisions = [d for d in governance_decisions if d.decision_type == "approve"]

                for decision in approved_decisions[:5]:  # 상위 5개만 분석
                    # 시나리오 정의: 해당 개념이 온톨로지에 추가될 경우의 영향
                    scenario = {
                        "concept_id": decision.concept_id,
                        "change_type": "add_concept",
                        "confidence": decision.confidence,
                        "source_tables": getattr(decision, 'source_tables', []),
                    }

                    # What-If 분석 실행
                    impact_result = whatif_analyzer.analyze_impact(
                        scenario=scenario,
                        context=context,
                    )

                    if impact_result:
                        whatif_results["scenarios_analyzed"] += 1
                        whatif_results["total_impact_score"] += impact_result.get("impact_score", 0)

                        # 결정에 영향 분석 결과 추가
                        decision.agent_opinions["whatif_analysis"] = {
                            "impact_score": impact_result.get("impact_score", 0),
                            "affected_entities": impact_result.get("affected_entities", []),
                            "risk_level": impact_result.get("risk_level", "unknown"),
                        }

                if whatif_results["scenarios_analyzed"] > 0:
                    avg_impact = whatif_results["total_impact_score"] / whatif_results["scenarios_analyzed"]
                    logger.info(
                        f"[v17.1] What-If Analysis: {whatif_results['scenarios_analyzed']} scenarios, "
                        f"avg impact={avg_impact:.2f}"
                    )
        except Exception as e:
            logger.debug(f"[v17.1] What-If Analysis skipped: {e}")

        # === v16.0: CDC Sync Engine 통합 ===
        cdc_sync_results = {}
        self._report_progress(0.92, "Configuring CDC Sync Engine (v16.0)")

        try:
            if CDC_ENGINE_AVAILABLE and create_cdc_engine_with_context:
                cdc_engine = create_cdc_engine_with_context(context)

                # 거버넌스 결정 기반 동기화 구독 설정
                subscriptions_created = 0
                for decision in governance_decisions[:10]:  # 상위 10개
                    if decision.decision == "approve":
                        # 승인된 개념에 대해 CDC 구독 생성
                        for source_table in (decision.source_tables or [])[:3]:
                            subscription = cdc_engine.subscribe(
                                table_name=source_table,
                                callback=None,  # 실제 콜백은 나중에 등록
                            )
                            if subscription:
                                subscriptions_created += 1

                cdc_sync_results = {
                    "subscriptions_created": subscriptions_created,
                    "sync_enabled": True,
                    "tables_monitored": subscriptions_created,
                }

                context.set_dynamic("cdc_sync_config", {
                    "subscriptions": subscriptions_created,
                    "enabled": True,
                })

                logger.info(
                    f"[v16.0] CDC Sync Engine: {subscriptions_created} subscriptions created"
                )
            else:
                logger.info("[v16.0] CDC Sync Engine not available, skipping")

        except Exception as e:
            logger.warning(f"[v16.0] CDC Sync Engine integration failed: {e}")
            import traceback
            logger.debug(traceback.format_exc())

        # === v22.0: decision_explainer로 거버넌스 결정 설명 생성 ===
        decision_explanations = []
        try:
            decision_explainer = self.get_v17_service("decision_explainer")
            if decision_explainer and governance_decisions:
                for decision in governance_decisions[:5]:  # 상위 5개 결정만
                    explanation = await decision_explainer.explain(
                        decision_type="governance",
                        decision_data={
                            "concept_id": decision.concept_id,
                            "decision": decision.decision,
                            "confidence": decision.confidence,
                            "reasoning": decision.reasoning[:200] if decision.reasoning else "",
                        },
                        context={"phase": "governance", "agent": self.agent_name},
                    )
                    if explanation:
                        decision_explanations.append({
                            "concept_id": decision.concept_id,
                            "explanation": explanation.get("summary", ""),
                            "factors": explanation.get("key_factors", []),
                        })
                if decision_explanations:
                    logger.info(f"[v22.0] DecisionExplainer generated {len(decision_explanations)} explanations")
        except Exception as e:
            logger.debug(f"[v22.0] DecisionExplainer skipped: {e}")

        # v22.0: Agent Learning - 경험 기록
        self.record_experience(
            experience_type="governance_decision",
            input_data={"task": "governance_strategy", "concepts": len(algorithmic_decisions)},
            output_data={"decisions_made": len(governance_decisions), "approved": summary.get("approved", 0)},
            outcome="success",
            confidence=0.85,
        )

        # v22.0: Agent Bus - 거버넌스 결정 결과 브로드캐스트
        await self.broadcast_discovery(
            discovery_type="governance_decisions_made",
            data={
                "decisions_made": len(governance_decisions),
                "approved": summary.get("approved", 0),
                "rejected": summary.get("rejected", 0),
                "needs_review": summary.get("needs_review", 0),
            },
        )

        return TodoResult(
            success=True,
            output={
                "decisions_made": len(governance_decisions),
                "summary": summary,
                # v4.6: ProcessAnalyzer 결과 요약
                "process_analyses_count": len(process_analyses),
                "decision_explanations": len(decision_explanations),  # v22.0
                # v4.3: Knowledge Graph 통계
                "kg_stats": {
                    "governance_triples": len(governance_triples),
                },
                # v5.0: Enhanced Validation 통계
                "enhanced_validation_stats": enhanced_stats,
                # v6.1: Structured Debate 통계 (legacy)
                "debate_conducted": len(debate_results),
                # v11.0: Evidence-Based Debate 통계
                "evidence_based_debate_stats": evidence_debate_stats,
                # v16.0: CDC Sync 통계
                "cdc_subscriptions_created": cdc_sync_results.get("subscriptions_created", 0),
            },
            context_updates={
                "governance_decisions": governance_decisions,
                "governance_triples": governance_triples,  # v4.3: Knowledge Graph 트리플
                "debate_results": debate_results,  # v6.1: 토론 결과
                "evidence_based_debate_results": evidence_based_debate_results,  # v11.0
            },
            metadata={
                "algorithmic_decisions": algorithmic_decisions,
                # v4.6: ProcessAnalyzer 상세 결과
                "process_analyses": process_analyses,
                # v5.0: Enhanced Validation 상세 결과
                "enhanced_validation": enhanced_validation_results,
                # v6.1: Structured Debate 상세 결과 (legacy)
                "debate_details": debate_results,
                # v11.0: Evidence-Based Council Debate 상세 결과
                "evidence_based_debate_details": evidence_based_debate_results,
                # v16.0: CDC Sync 상세 결과
                "cdc_sync_results": cdc_sync_results,
            },
        )

    def _generate_governance_triples(
        self,
        decisions: List[Any],
        context: "SharedContext",
    ) -> List[Dict[str, Any]]:
        """
        v4.3: 거버넌스 결정에서 Knowledge Graph 트리플 생성

        거버넌스 트리플 유형:
        - gov:Decision (rdf:type)
        - gov:hasDecisionType (approve/reject/escalate/schedule_review)
        - gov:hasConfidence
        - gov:hasReasoning
        - gov:targetConcept (연결)
        - gov:hasAction (추천 액션)
        - prov:wasGeneratedBy (생성 에이전트)
        - prov:generatedAtTime (생성 시간)
        """
        triples = []

        # Namespace prefixes
        GOV = "gov:"  # Governance namespace
        PROV = "prov:"  # Provenance namespace
        RDF = "rdf:"
        RDFS = "rdfs:"
        XSD = "xsd:"

        # Decision type -> OWL class mapping
        decision_type_map = {
            "approve": "gov:ApprovedDecision",
            "reject": "gov:RejectedDecision",
            "escalate": "gov:EscalatedDecision",
            "schedule_review": "gov:ScheduledReviewDecision",
        }

        for decision in decisions:
            decision_id = decision.decision_id
            decision_uri = f"gov:{decision_id}"

            # 1. 기본 타입
            triples.append({
                "subject": decision_uri,
                "predicate": f"{RDF}type",
                "object": "gov:GovernanceDecision",
                "confidence": 1.0,
                "source": "governance_strategist",
            })

            # 2. 구체적 결정 타입 (서브클래스)
            specific_type = decision_type_map.get(decision.decision_type, "gov:GovernanceDecision")
            triples.append({
                "subject": decision_uri,
                "predicate": f"{RDF}type",
                "object": specific_type,
                "confidence": 1.0,
                "source": "governance_strategist",
            })

            # 3. 결정 타입 (리터럴)
            triples.append({
                "subject": decision_uri,
                "predicate": f"{GOV}hasDecisionType",
                "object": f'"{decision.decision_type}"^^{XSD}string',
                "confidence": 1.0,
                "source": "governance_strategist",
            })

            # 4. Confidence
            triples.append({
                "subject": decision_uri,
                "predicate": f"{GOV}hasConfidence",
                "object": f'"{decision.confidence}"^^{XSD}decimal',
                "confidence": 1.0,
                "source": "governance_strategist",
            })

            # 5. Reasoning
            if decision.reasoning:
                reasoning_escaped = decision.reasoning.replace('"', '\\"')[:500]  # 길이 제한
                triples.append({
                    "subject": decision_uri,
                    "predicate": f"{GOV}hasReasoning",
                    "object": f'"{reasoning_escaped}"^^{XSD}string',
                    "confidence": 1.0,
                    "source": "governance_strategist",
                })

            # 6. 대상 개념과 연결
            triples.append({
                "subject": decision_uri,
                "predicate": f"{GOV}targetConcept",
                "object": f"ont:{decision.concept_id}",
                "confidence": 1.0,
                "source": "governance_strategist",
            })

            # 역방향 링크: 개념에서 결정으로
            triples.append({
                "subject": f"ont:{decision.concept_id}",
                "predicate": f"{GOV}hasGovernanceDecision",
                "object": decision_uri,
                "confidence": 1.0,
                "source": "governance_strategist",
            })

            # 7. 추천 액션들
            for idx, action in enumerate(decision.recommended_actions or []):
                action_uri = f"gov:action_{decision_id}_{idx}"

                # 액션 타입
                triples.append({
                    "subject": action_uri,
                    "predicate": f"{RDF}type",
                    "object": "gov:RecommendedAction",
                    "confidence": 1.0,
                    "source": "governance_strategist",
                })

                # 액션 연결
                triples.append({
                    "subject": decision_uri,
                    "predicate": f"{GOV}hasRecommendedAction",
                    "object": action_uri,
                    "confidence": 1.0,
                    "source": "governance_strategist",
                })

                # 액션 상세
                if isinstance(action, dict):
                    action_type = action.get("action_type", "unknown")
                    action_desc = action.get("description", "")[:200]
                    action_priority = action.get("priority", "medium")

                    triples.append({
                        "subject": action_uri,
                        "predicate": f"{GOV}actionType",
                        "object": f'"{action_type}"^^{XSD}string',
                        "confidence": 1.0,
                        "source": "governance_strategist",
                    })
                    triples.append({
                        "subject": action_uri,
                        "predicate": f"{RDFS}label",
                        "object": f'"{action_desc}"^^{XSD}string',
                        "confidence": 1.0,
                        "source": "governance_strategist",
                    })
                    triples.append({
                        "subject": action_uri,
                        "predicate": f"{GOV}hasPriority",
                        "object": f'"{action_priority}"^^{XSD}string',
                        "confidence": 1.0,
                        "source": "governance_strategist",
                    })

            # 8. Provenance 정보
            triples.append({
                "subject": decision_uri,
                "predicate": f"{PROV}wasGeneratedBy",
                "object": f"agent:governance_strategist",
                "confidence": 1.0,
                "source": "governance_strategist",
            })

            triples.append({
                "subject": decision_uri,
                "predicate": f"{PROV}generatedAtTime",
                "object": f'"{decision.made_at}"^^{XSD}dateTime',
                "confidence": 1.0,
                "source": "governance_strategist",
            })

        return triples

    async def _llm_review_decisions(
        self,
        concepts_with_decisions: List[Dict[str, Any]],
        context: "SharedContext",
    ) -> Dict[str, Dict[str, Any]]:
        """LLM을 통한 결정 검토"""
        if not concepts_with_decisions:
            return {}

        # 배치 처리 - v6.0: LLM 호출 최적화 (10→30)
        batch_size = 30
        all_reviews = {}

        for i in range(0, len(concepts_with_decisions), batch_size):
            batch = concepts_with_decisions[i:i + batch_size]

            instruction = f"""Review these algorithmic governance decisions.

Domain: {context.get_industry()}

Concepts with Algorithmic Decisions:
{json.dumps(batch, indent=2, ensure_ascii=False, default=str)}

For each concept:
1. Validate or adjust the algorithmic decision
2. Add business context and reasoning
3. Suggest specific recommended actions

Return JSON:
```json
{{
  "reviewed_decisions": [
    {{
      "concept_id": "<id>",
      "decision_type": "<approve|reject|escalate|schedule_review>",
      "confidence": <float 0-1>,
      "reasoning": "<enhanced reasoning with business context>",
      "business_value": "<high|medium|low>",
      "risk_level": "<high|medium|low>",
      "recommended_actions": [
        {{
          "action_type": "<implementation|review|monitoring|documentation>",
          "description": "<specific action>",
          "priority": "<critical|high|medium|low>"
        }}
      ],
      "stakeholders": ["role1", "role2"],
      "override_algorithmic": <bool>,
      "override_reason": "<if overriding, explain why>"
    }}
  ]
}}
```"""

            response = await self.call_llm(instruction, max_tokens=4000)
            batch_reviews = parse_llm_json(response, key="reviewed_decisions", default=[])

            for review in batch_reviews:
                concept_id = review.get("concept_id")
                if concept_id:
                    all_reviews[concept_id] = review

        return all_reviews


class ActionPrioritizerAutonomousAgent(AutonomousAgent):
    """
    액션 우선순위 자율 에이전트 (v2.0)

    Algorithm-First 접근:
    1. Impact × Urgency / Effort 기반 우선순위 계산
    2. 의존성 분석 및 실행 순서 결정
    3. LLM을 통한 설명 보강
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.priority_calculator = PriorityCalculator()

    @property
    def agent_type(self) -> str:
        return "action_prioritizer"

    @property
    def agent_name(self) -> str:
        return "Action Prioritizer"

    @property
    def phase(self) -> str:
        return "governance"

    def get_system_prompt(self) -> str:
        return """You are an Action Prioritization Expert for data governance execution.

Your role is to ENHANCE and SEQUENCE algorithmically prioritized actions.

Priority Framework (already calculated):
- CRITICAL: High impact, urgent, blocking issues
- HIGH: Significant business impact, near-term deadlines
- MEDIUM: Important improvements, moderate impact
- LOW: Nice-to-have, long-term improvements

Your Job:
1. Validate algorithmic priorities
2. Identify dependencies between actions
3. Create optimal execution sequence
4. Assign appropriate roles

## REQUIRED OUTPUT FORMAT (JSON Schema):
```json
{
  "prioritized_actions": [
    {
      "action_id": "ACT_001",
      "title": "action title",
      "description": "detailed description of the action",
      "priority": "critical|high|medium|low",
      "priority_score": 0.0,
      "category": "data_quality|governance|integration|automation|reporting",
      "estimated_effort": "hours|days|weeks",
      "dependencies": ["ACT_XXX"],
      "assigned_role": "data_engineer|data_steward|analyst|admin",
      "expected_roi": "high|medium|low",
      "execution_sequence": 1
    }
  ],
  "execution_plan": {
    "phase_1_quick_wins": ["ACT_001", "ACT_002"],
    "phase_2_core_improvements": ["ACT_003"],
    "phase_3_strategic": ["ACT_004"]
  },
  "priority_summary": {
    "critical": 0,
    "high": 0,
    "medium": 0,
    "low": 0
  },
  "total_estimated_effort": "X days/weeks",
  "recommendations": [
    "recommendation text"
  ]
}
```

Respond with ONLY valid JSON matching this schema."""

    async def execute_task(
        self,
        todo: "PipelineTodo",
        context: "SharedContext",
    ) -> "TodoResult":
        """액션 우선순위 작업 실행"""
        from ...todo.models import TodoResult

        self._report_progress(0.1, "Extracting actions from governance decisions")

        # v14.0: 빈 governance_decisions 처리 - 조기 반환
        if not context.governance_decisions:
            logger.warning("[v14.0] No governance decisions available - returning empty action backlog")
            return TodoResult(
                success=True,
                output={
                    "message": "No governance decisions available to prioritize",
                    "actions_created": 0,
                    "priority_summary": {"critical": 0, "high": 0, "medium": 0, "low": 0},
                    "quick_wins_count": 0,
                },
                context_updates={"action_backlog": []},
                metadata={"skipped_reason": "no_governance_decisions"},
            )

        # 1단계: 거버넌스 결정에서 액션 추출
        raw_actions = []
        action_count = 0

        for decision in context.governance_decisions:
            for action in decision.recommended_actions:
                action_count += 1
                action_item = {
                    "action_id": f"action_{action_count}",
                    "title": action.get("description", "")[:50] if isinstance(action, dict) else str(action)[:50],
                    "description": action.get("description", "") if isinstance(action, dict) else str(action),
                    "action_type": action.get("action_type", "implementation") if isinstance(action, dict) else "implementation",
                    "source_decision": decision.decision_id,
                    "source_concept": decision.concept_id,
                    "impact": self._infer_impact(decision, action),
                    "effort": self._infer_effort(action),
                    "risk_if_not_done": decision.agent_opinions.get("algorithmic", {}).get("risk_level", "medium"),
                }
                raw_actions.append(action_item)

        self._report_progress(0.3, f"Extracted {len(raw_actions)} actions")

        # 2단계: 알고리즘 기반 우선순위 계산
        self._report_progress(0.4, "Calculating algorithmic priorities")

        prioritized_actions = self.priority_calculator.sort_by_priority(raw_actions)
        quick_wins = self.priority_calculator.identify_quick_wins(prioritized_actions)

        self._report_progress(0.5, "Enhancing with LLM")

        # 3단계: LLM을 통한 보강
        enhanced_actions = await self._enhance_actions_with_llm(
            prioritized_actions, quick_wins, context
        )

        self._report_progress(0.8, "Determining execution sequence")

        # 4단계: 실행 순서 결정
        execution_sequence = [a.get("action_id") for a in enhanced_actions[:20]]

        # === v22.0: Semantic Search로 관련 엔티티 조회 및 액션 보강 ===
        try:
            semantic_searcher = self.get_v17_service("semantic_searcher")
            if semantic_searcher and enhanced_actions:
                for action in enhanced_actions[:10]:  # 상위 10개 액션만
                    action_desc = action.get("description", "") or action.get("title", "")
                    if action_desc:
                        related = await semantic_searcher.search(query=action_desc[:200], limit=3)
                        if related:
                            action["semantic_context"] = [r.get("name", r.get("id", "")) for r in related[:3]]
                logger.info(f"[v22.0] SemanticSearcher enriched {min(len(enhanced_actions), 10)} actions with context")
        except Exception as e:
            logger.debug(f"[v22.0] SemanticSearcher skipped in ActionPrioritizer: {e}")

        # === v16.0: Actions Engine 통합 ===
        actions_engine_results = {}
        self._report_progress(0.85, "Configuring Actions Engine (v16.0)")

        try:
            if ACTIONS_ENGINE_AVAILABLE and ActionsEngine:
                actions_engine = ActionsEngine()

                # 각 우선순위 액션을 Actions Engine에 등록
                registered_actions = []
                for action_data in enhanced_actions[:20]:  # 상위 20개
                    priority = action_data.get("calculated_priority", "medium")

                    # 트리거 유형 결정
                    trigger_type = TriggerType.MANUAL
                    if priority == "critical":
                        trigger_type = TriggerType.EVENT  # 자동 실행
                    elif action_data.get("action_type") == "data_quality":
                        trigger_type = TriggerType.DATA_QUALITY

                    # 액션 유형 매핑
                    action_type_map = {
                        "implementation": ActionType.CREATE,
                        "schema_change": ActionType.UPDATE,
                        "data_quality": ActionType.VALIDATE,
                        "monitoring": ActionType.NOTIFY,
                        "documentation": ActionType.CREATE,
                        "cleanup": ActionType.DELETE,
                    }
                    action_type = action_type_map.get(
                        action_data.get("action_type", "implementation"),
                        ActionType.CREATE
                    )

                    # 트리거 생성
                    trigger = Trigger(
                        trigger_id=f"trigger_{action_data.get('action_id', '')}",
                        trigger_type=trigger_type,
                        name=f"Trigger for {action_data.get('title', 'Action')[:30]}",
                        description=f"Auto-generated trigger for priority action",
                    )

                    # 액션 생성 및 등록
                    action = Action(
                        action_id=action_data.get("action_id", ""),
                        name=action_data.get("title", "Untitled Action")[:50],
                        description=action_data.get("description", "")[:200],
                        action_type=action_type,
                        triggers=[trigger],
                        parameters=[],
                        handler=None,  # 실제 핸들러는 나중에 등록
                        created_by="action_prioritizer_agent",
                    )

                    actions_engine.register_action(action)
                    registered_actions.append(action.action_id)

                    # Critical 액션은 즉시 실행 가능 상태로 표시
                    if priority == "critical":
                        action_data["auto_execute"] = True
                        action_data["trigger_configured"] = True

                actions_engine_results = {
                    "actions_registered": len(registered_actions),
                    "registered_action_ids": registered_actions[:10],
                    "engine_status": "configured",
                }

                # context에 저장
                context.set_dynamic("actions_engine_config", {
                    "registered_actions": registered_actions,
                    "total_registered": len(registered_actions),
                })

                logger.info(
                    f"[v16.0] Actions Engine: {len(registered_actions)} actions registered"
                )
            else:
                logger.info("[v16.0] Actions Engine not available, skipping")

        except Exception as e:
            logger.warning(f"[v16.0] Actions Engine integration failed: {e}")
            import traceback
            logger.debug(traceback.format_exc())

        self._report_progress(0.95, "Finalizing action backlog")

        # === v17.1: Report Generator 통합 - 액션 우선순위 보고서 생성 ===
        report_results = {"report_generated": False, "report_type": None}
        try:
            report_generator = self.get_v17_service("report_generator")
            if report_generator and len(enhanced_actions) > 0:
                # 액션 우선순위 보고서 생성
                report_data = {
                    "title": "Action Priority Report",
                    "total_actions": len(enhanced_actions),
                    "critical_count": len([a for a in enhanced_actions if a.get("calculated_priority") == "critical"]),
                    "high_count": len([a for a in enhanced_actions if a.get("calculated_priority") == "high"]),
                    "quick_wins": quick_wins[:5],
                    "top_actions": enhanced_actions[:10],
                    "generated_at": datetime.now().isoformat() if 'datetime' in dir() else "N/A",
                }

                report = report_generator.generate_summary(
                    report_type="action_priority",
                    data=report_data,
                    context=context,
                )

                if report:
                    report_results["report_generated"] = True
                    report_results["report_type"] = "action_priority"

                    # 보고서를 context에 저장
                    context.set_dynamic("action_priority_report", report)

                    logger.info(f"[v17.1] Report Generator: Action priority report generated")
        except Exception as e:
            logger.debug(f"[v17.1] Report Generator skipped: {e}")

        # 집계
        priority_summary = {
            "critical": len([a for a in enhanced_actions if a.get("calculated_priority") == "critical"]),
            "high": len([a for a in enhanced_actions if a.get("calculated_priority") == "high"]),
            "medium": len([a for a in enhanced_actions if a.get("calculated_priority") == "medium"]),
            "low": len([a for a in enhanced_actions if a.get("calculated_priority") == "low"]),
        }

        # v22.0: Agent Learning - 경험 기록
        self.record_experience(
            experience_type="governance_decision",
            input_data={"task": "action_prioritization", "raw_actions": len(raw_actions)},
            output_data={"actions_created": len(enhanced_actions), "quick_wins": len(quick_wins)},
            outcome="success",
            confidence=0.8,
        )

        return TodoResult(
            success=True,
            output={
                "actions_created": len(enhanced_actions),
                "priority_summary": priority_summary,
                "quick_wins_count": len(quick_wins),
                # v16.0
                "actions_engine_registered": actions_engine_results.get("actions_registered", 0),
            },
            context_updates={
                "action_backlog": enhanced_actions,
            },
            metadata={
                "execution_sequence": execution_sequence,
                "quick_wins": quick_wins,
                # v16.0
                "actions_engine_results": actions_engine_results,
            },
        )

    def _infer_impact(self, decision, action) -> str:
        """액션 영향도 추론"""
        business_value = decision.agent_opinions.get("algorithmic", {}).get("business_value", "medium")
        action_type = action.get("action_type", "") if isinstance(action, dict) else ""

        if business_value == "high" or action_type in ["implementation", "schema_change"]:
            return "high"
        elif action_type in ["monitoring", "documentation"]:
            return "low"
        return "medium"

    def _infer_effort(self, action) -> str:
        """액션 노력 추론"""
        action_type = action.get("action_type", "") if isinstance(action, dict) else ""
        description = action.get("description", "") if isinstance(action, dict) else str(action)

        # 키워드 기반 추론
        high_effort_keywords = ["redesign", "refactor", "migrate", "rebuild", "implement"]
        low_effort_keywords = ["document", "log", "alert", "monitor", "review"]

        desc_lower = description.lower()

        if any(kw in desc_lower for kw in high_effort_keywords):
            return "high"
        elif any(kw in desc_lower for kw in low_effort_keywords):
            return "low"
        return "medium"

    async def _enhance_actions_with_llm(
        self,
        actions: List[Dict[str, Any]],
        quick_wins: List[str],
        context: "SharedContext",
    ) -> List[Dict[str, Any]]:
        """LLM을 통한 액션 보강"""
        if not actions:
            return []

        instruction = f"""Enhance these prioritized actions with more detail.

Domain: {context.get_industry()}

Prioritized Actions (top 20):
{json.dumps(actions[:20], indent=2, ensure_ascii=False, default=str)}

Quick Wins (high impact, low effort):
{json.dumps(quick_wins, ensure_ascii=False)}

For each action:
1. Validate or adjust priority
2. Add specific implementation details
3. Identify dependencies
4. Suggest assignee role

Return JSON:
```json
{{
  "enhanced_actions": [
    {{
      "action_id": "<existing>",
      "title": "<improved title>",
      "description": "<detailed description>",
      "action_type": "<existing or improved>",
      "priority": "<calculated_priority>",
      "priority_score": <existing>,
      "impact": "<existing>",
      "effort": "<existing or adjusted>",
      "estimated_hours": <int>,
      "source_decision": "<existing>",
      "dependencies": ["action_id1"],
      "assignee_role": "<data_owner|developer|analyst|admin>",
      "status": "pending"
    }}
  ]
}}
```"""

        response = await self.call_llm(instruction, max_tokens=4000)
        enhanced = parse_llm_json(response, key="enhanced_actions", default=[])

        if enhanced:
            # 병합: LLM 결과 + 원본 (LLM이 놓친 것)
            enhanced_ids = {a.get("action_id") for a in enhanced}
            for action in actions:
                if action.get("action_id") not in enhanced_ids:
                    enhanced.append(action)
            return enhanced

        return actions


class RiskAssessorAutonomousAgent(AutonomousAgent):
    """
    리스크 평가 자율 에이전트 (v3.0)

    Algorithm-First 접근:
    1. NIST 기반 리스크 프레임워크 적용
    2. Likelihood × Impact = Risk Score
    3. Business Insights 기반 데이터 리스크 분석
    4. CausalImpactAnalyzer로 인과 분석 (신규)
    5. AnomalyDetector로 앙상블 이상탐지 (신규)
    6. LLM을 통한 완화 전략 상세화

    v3.0: CausalImpactAnalyzer, AnomalyDetector 통합
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # v14.0: 도메인 하드코딩 제거 - Lazy initialization (execute_task에서 context 기반 설정)
        self.risk_assessor = None
        self._current_domain = None
        # v14.0: LLM 클라이언트 전달하여 도메인 특화 인사이트 생성 활성화
        self.business_insights_analyzer = BusinessInsightsAnalyzer(llm_client=self.llm_client)
        # v3.0: 인과 추론 및 이상탐지 통합
        self.causal_analyzer = CausalImpactAnalyzer()
        self.anomaly_detector = AnomalyDetector()
        # v6.1: Counterfactual Reasoning - "만약 ~했다면?" 분석
        self.counterfactual_analyzer = CounterfactualAnalyzer()

    def _get_domain_risk_examples(self, domain: str) -> str:
        """v14.0: 도메인별 리스크 예시 반환 - LLM이 참고할 수 있는 컨텍스트"""
        domain_lower = domain.lower()

        if "healthcare" in domain_lower or "medical" in domain_lower:
            return """- PHI Data: High baseline (HIPAA) - Likelihood: 3, Impact: 5
- Clinical Data: Critical accuracy requirements - Likelihood: 4, Impact: 4
- Billing Data: Fraud detection sensitivity - Likelihood: 3, Impact: 4"""

        elif "supply" in domain_lower or "logistics" in domain_lower:
            return """- Inventory Data: Stock accuracy impacts operations - Likelihood: 3, Impact: 4
- Shipping Data: Delivery delays affect customer satisfaction - Likelihood: 4, Impact: 3
- Supplier Data: Vendor reliability tracking - Likelihood: 3, Impact: 4"""

        elif "finance" in domain_lower or "banking" in domain_lower:
            return """- Transaction Data: High fraud risk (SOX/PCI) - Likelihood: 4, Impact: 5
- Customer Financial Data: Privacy regulations - Likelihood: 3, Impact: 5
- Trading Data: Real-time accuracy critical - Likelihood: 4, Impact: 4"""

        elif "ecommerce" in domain_lower or "retail" in domain_lower:
            return """- Customer Data: PII protection (GDPR/CCPA) - Likelihood: 3, Impact: 4
- Order Data: Fulfillment accuracy - Likelihood: 3, Impact: 3
- Pricing Data: Revenue impact from errors - Likelihood: 2, Impact: 4"""

        elif "manufacturing" in domain_lower:
            return """- Production Data: Quality control critical - Likelihood: 3, Impact: 4
- Equipment Data: Maintenance scheduling impact - Likelihood: 4, Impact: 3
- Safety Data: Regulatory compliance (OSHA) - Likelihood: 2, Impact: 5"""

        else:
            return """- Sensitive Data: Data privacy requirements - Likelihood: 3, Impact: 4
- Operational Data: Business continuity impact - Likelihood: 3, Impact: 3
- External Data: Third-party dependency risk - Likelihood: 3, Impact: 3"""

    @property
    def agent_type(self) -> str:
        return "risk_assessor"

    @property
    def agent_name(self) -> str:
        return "Risk Assessor"

    @property
    def phase(self) -> str:
        return "governance"

    def get_system_prompt(self) -> str:
        # v14.0: 도메인 적응형 프롬프트 생성
        domain = getattr(self, '_current_domain', None) or "general"
        domain_examples = self._get_domain_risk_examples(domain)

        return f"""You are a Risk Assessment Expert for data governance in the **{domain}** domain.

Your role is to VALIDATE and ENHANCE algorithmic risk assessments.

## Risk Categories:
- Data Quality Risk: Inaccurate, incomplete data
- Security Risk: Unauthorized access, breaches
- Compliance Risk: Regulatory violations (domain-specific)
- Operational Risk: System failures, data loss
- Reputational Risk: Customer/stakeholder trust damage

## Risk Assessment Framework

For each risk, calculate: **Risk Score = Likelihood (1-5) x Impact (1-5)**

### Likelihood Scale
| Score | Level | Description |
|-------|-------|-------------|
| 1 | Rare | Less than 1% probability |
| 2 | Unlikely | 1-10% probability |
| 3 | Possible | 10-50% probability |
| 4 | Likely | 50-90% probability |
| 5 | Almost Certain | >90% probability |

### Impact Scale
| Score | Level | Description |
|-------|-------|-------------|
| 1 | Negligible | No regulatory impact, <$1K cost |
| 2 | Minor | Internal process disruption, $1K-$10K cost |
| 3 | Moderate | Department-level impact, $10K-$100K cost |
| 4 | Major | Organization-wide impact, $100K-$1M cost |
| 5 | Critical | Regulatory violation, >$1M cost |

### Risk Level Mapping
| Score Range | Level | Color | Action |
|-------------|-------|-------|--------|
| 1-5 | Low | Green | Accept |
| 6-12 | Medium | Yellow | Monitor |
| 13-20 | High | Orange | Mitigate |
| 21-25 | Critical | Red | Immediate Action Required |

## {domain.replace('_', ' ').title()} Domain Baseline Risks:
{domain_examples}

## Chain-of-Thought Requirement
For each identified risk, show your calculation:

1. **Risk Identification**: What specific risk did you identify?
2. **Likelihood Assessment**: Why did you assign this likelihood score? (1-5)
3. **Impact Assessment**: Why did you assign this impact score? (1-5)
4. **Risk Calculation**: Likelihood x Impact = Risk Score
5. **Mitigation Recommendation**: What action should be taken based on risk level?

## Output Format
```json
{{
  "risk_assessments": [
    {{
      "concept_id": "<id>",
      "risks": [
        {{
          "category": "<data_quality|security|compliance|operational|reputational>",
          "description": "<specific risk description>",
          "reasoning": {{
            "likelihood_rationale": "<why this likelihood score>",
            "impact_rationale": "<why this impact score>"
          }},
          "likelihood": <int 1-5>,
          "impact": <int 1-5>,
          "risk_score": <int 1-25>,
          "risk_level": "<low|medium|high|critical>",
          "mitigation": "<recommended action>"
        }}
      ],
      "overall_risk_level": "<low|medium|high|critical>",
      "priority_mitigations": ["action 1", "action 2"]
    }}
  ]
}}
```

Respond ONLY with valid JSON."""

    async def execute_task(
        self,
        todo: "PipelineTodo",
        context: "SharedContext",
    ) -> "TodoResult":
        """리스크 평가 작업 실행"""
        from ...todo.models import TodoResult
        from ..analysis.governance import RiskAssessor as GovernanceRiskAssessor

        # v14.0: 도메인 하드코딩 제거 - context에서 동적으로 도메인 가져오기
        domain = context.get_industry() or "general"
        if self.risk_assessor is None or self._current_domain != domain:
            self.risk_assessor = GovernanceRiskAssessor(domain=domain)
            self._current_domain = domain
            logger.info(f"[v14.0] RiskAssessor initialized with domain: {domain}")

        self._report_progress(0.1, "Computing algorithmic risk assessments")

        # === v17.1: Phase 1 operational_insights 로드 ===
        # DataAnalyst가 생성한 운영 인사이트 (취소율, 오류율, 이상치 등)
        operational_insights = context.get_dynamic("operational_insights", {})
        status_analysis = context.get_dynamic("status_analysis", {})

        # 운영 리스크 요소 추출
        operational_risk_factors = []
        if operational_insights and isinstance(operational_insights, dict):
            # 높은 취소율, 오류율 등은 리스크 증가 요인
            for key, value in operational_insights.items():
                if isinstance(value, dict):
                    rate = value.get("rate", value.get("percentage", 0))
                    if rate and rate > 0.1:  # 10% 이상이면 리스크 요소로 추가
                        operational_risk_factors.append({
                            "factor": key,
                            "rate": rate,
                            "tables": value.get("tables", value.get("affected_tables", [])),
                            "severity": "high" if rate > 0.3 else "medium",
                        })
            logger.info(f"[v17.1] Loaded {len(operational_risk_factors)} operational risk factors from Phase 1")

        # 상태 분석에서 문제 있는 필드 추출
        problematic_status_fields = []
        if status_analysis:
            if isinstance(status_analysis, list):
                # Discovery stores status_analysis as list of dicts
                for item in status_analysis:
                    if isinstance(item, dict) and item.get("problem_ratio", 0) > 0.05:
                        problematic_status_fields.append({
                            "table": item.get("table", "unknown"),
                            "field": item.get("column", "unknown"),
                            "issue_type": "high_problem_ratio",
                        })
            elif isinstance(status_analysis, dict):
                for table, analysis in status_analysis.items():
                    if isinstance(analysis, dict):
                        for field, info in analysis.items():
                            if isinstance(info, dict) and info.get("has_issues", False):
                                problematic_status_fields.append({
                                    "table": table,
                                    "field": field,
                                    "issue_type": info.get("issue_type", "unknown"),
                                })
            if problematic_status_fields:
                logger.info(f"[v17.1] Found {len(problematic_status_fields)} problematic status fields")

        # === v22.0: OWL2/SHACL + v17 서비스 리스크 보강 ===
        owl2_risk_factors = []
        try:
            owl2_results = context.get_dynamic("owl2_reasoning_results", {})
            shacl_results = context.get_dynamic("shacl_validation_results", {})

            # SHACL 위반은 온톨로지 리스크
            if shacl_results and not shacl_results.get("is_conformant", True):
                violations = shacl_results.get("violations", [])
                for v in violations[:5]:
                    owl2_risk_factors.append({
                        "factor": f"SHACL violation: {v.get('path', 'unknown')}",
                        "severity": "high" if v.get("severity") == "Violation" else "medium",
                        "source": "shacl_validation",
                    })
                logger.info(f"[v22.0] {len(owl2_risk_factors)} SHACL violation risk factors added")
        except Exception as e:
            logger.debug(f"[v22.0] OWL2/SHACL risk integration skipped: {e}")

        # v2.1: Business Insights 기반 데이터 리스크 분석
        business_insights = []
        try:
            tables_for_insights = {
                name: {"columns": info.columns, "row_count": info.row_count}
                for name, info in context.tables.items()
            }
            # v22.1: 전체 데이터 로드 (샘플링 금지)
            sample_data = context.get_all_full_data()

            logger.info(f"[RiskAssessor] Tables for insights: {len(tables_for_insights)}, Full data tables: {len(sample_data)}")

            # v14.0: DataAnalyst가 분석한 column_semantics 가져오기
            column_semantics = context.get_dynamic("column_semantics", {})
            if column_semantics:
                logger.info(f"[RiskAssessor v14.0] Using LLM-analyzed column_semantics: {len(column_semantics)} columns")

            if sample_data:
                # v14.0: domain_context
                domain_ctx = context.get_domain() if hasattr(context, 'get_domain') else None
                domain_str = domain_ctx.industry if domain_ctx and hasattr(domain_ctx, 'industry') else "general"

                # v19.0: Use UnifiedInsightPipeline (Algorithm→Simulation→LLM 3-phase)
                if UNIFIED_PIPELINE_AVAILABLE:
                    logger.info("[v22.0] UnifiedInsightPipeline AVAILABLE, using it")
                    logger.info(f"[RiskAssessor v19.0] Using UnifiedInsightPipeline (Algorithm→Simulation→LLM)")

                    # Build correlations from causal analysis treatment_effects
                    correlations = self._build_correlations_from_causal(context)
                    logger.info(f"[RiskAssessor v19.0] Built {len(correlations)} correlations from causal analysis")

                    # v18.3: pipeline_context for BusinessInsightsAnalyzer (Phase 1 내부)
                    pipeline_ctx = {
                        "causal_insights": getattr(context, 'causal_insights', {}) or {},
                        "causal_relationships": getattr(context, 'causal_relationships', []) or [],
                        "counterfactual_analysis": getattr(context, 'counterfactual_analysis', []) or [],
                        "virtual_entities": getattr(context, 'virtual_entities', []) or [],
                        "palantir_insights": getattr(context, 'palantir_insights', []) or [],
                    }
                    pipeline_ctx = {k: v for k, v in pipeline_ctx.items() if v}

                    # Set pipeline_context on the business analyzer inside UnifiedInsightPipeline
                    self.business_insights_analyzer.domain_context = domain_ctx
                    if pipeline_ctx:
                        self.business_insights_analyzer._pipeline_context = pipeline_ctx

                    unified_pipeline = UnifiedInsightPipeline(
                        context=context,
                        llm_client=self.llm_client,
                        domain=domain_str,
                        enable_simulation=bool(correlations),
                        enable_llm=True,
                        pipeline_context=pipeline_ctx,  # v19.4: ATE/stratified analysis → LLM
                    )

                    import asyncio
                    try:
                        loop = asyncio.get_event_loop()
                        if loop.is_running():
                            # Already in async context
                            pipeline_result = await unified_pipeline.run(
                                tables=tables_for_insights,
                                data=sample_data,
                                correlations=correlations,
                                column_semantics=column_semantics,
                            )
                        else:
                            pipeline_result = loop.run_until_complete(unified_pipeline.run(
                                tables=tables_for_insights,
                                data=sample_data,
                                correlations=correlations,
                                column_semantics=column_semantics,
                            ))
                    except RuntimeError:
                        # Fallback: create new event loop
                        pipeline_result = asyncio.run(unified_pipeline.run(
                            tables=tables_for_insights,
                            data=sample_data,
                            correlations=correlations,
                            column_semantics=column_semantics,
                        ))

                    # Convert UnifiedInsight → BusinessInsight-compatible format
                    for ui in pipeline_result.unified_insights:
                        ui_dict = ui.to_dict()
                        # Create a mock BusinessInsight-like object for backward compatibility
                        from ..analysis.business_insights import BusinessInsight, InsightType, InsightSeverity
                        try:
                            bi = BusinessInsight(
                                insight_id=ui_dict.get("insight_id", ""),
                                title=ui_dict.get("title", ""),
                                description=ui_dict.get("summary", ""),
                                insight_type=InsightType.OPERATIONAL,
                                severity=InsightSeverity(ui_dict.get("severity", "medium")),
                                source_table=ui_dict.get("source_table", ""),
                                source_column=ui_dict.get("source_column", ""),
                                source_system="UnifiedInsightPipeline",
                                recommendations=ui_dict.get("recommendations", []),
                                business_impact=ui_dict.get("business_impact", ""),
                            )
                            # Attach Palantir-style fields as extra attributes
                            bi._hypothesis = ui_dict.get("hypothesis", "")
                            bi._evidence = ui_dict.get("evidence", [])
                            bi._validation_summary = ui_dict.get("validation_summary", "")
                            bi._prescriptive_actions = ui_dict.get("prescriptive_actions", [])
                            bi._confidence_breakdown = ui_dict.get("confidence_breakdown", {})
                            bi._algorithm_findings = ui_dict.get("algorithm_findings", {})
                            bi._simulation_evidence = ui_dict.get("simulation_evidence", {})
                            bi._phases_completed = ui_dict.get("phases_completed", [])
                            business_insights.append(bi)
                        except Exception as e:
                            logger.warning(f"Failed to convert UnifiedInsight to BusinessInsight: {e}")

                    logger.info(f"[RiskAssessor v19.0] UnifiedInsightPipeline: {len(business_insights)} insights "
                               f"({pipeline_result.simulation_experiments_count} simulation experiments, "
                               f"phases={pipeline_result.phases_executed})")
                else:
                    # Fallback: legacy BusinessInsightsAnalyzer
                    logger.info(f"[RiskAssessor] Fallback: BusinessInsightsAnalyzer (UnifiedInsightPipeline not available)")
                    self.business_insights_analyzer.domain_context = domain_ctx
                    pipeline_ctx = {
                        "causal_insights": getattr(context, 'causal_insights', {}) or {},
                        "causal_relationships": getattr(context, 'causal_relationships', []) or [],
                    }
                    pipeline_ctx = {k: v for k, v in pipeline_ctx.items() if v}
                    business_insights = self.business_insights_analyzer.analyze(
                        tables=tables_for_insights,
                        data=sample_data,
                        column_semantics=column_semantics,
                        pipeline_context=pipeline_ctx,
                    )
                    logger.info(f"[RiskAssessor] BusinessInsightsAnalyzer found {len(business_insights)} insights")
            else:
                logger.warning("[RiskAssessor] No sample_data available, skipping BusinessInsightsAnalyzer")
        except Exception as e:
            logger.warning(f"Business Insights Analysis failed: {e}", exc_info=True)

        self._report_progress(0.2, "Assessing concept risks")

        # 1단계: 알고리즘 기반 리스크 평가
        algorithmic_risks = {}
        all_risks = []

        for concept in context.ontology_concepts:
            concept_dict = {
                "concept_id": concept.concept_id,
                "concept_type": concept.concept_type,
                "name": concept.name,
                "confidence": concept.confidence,
                "source_evidence": concept.source_evidence,
                "source_tables": concept.source_tables,
            }

            risks = self.risk_assessor.assess_concept_risk(concept_dict)
            risk_data = [
                {
                    "category": r.category,
                    "likelihood": r.likelihood,
                    "impact": r.impact,
                    "risk_score": r.risk_score,
                    "risk_level": r.risk_level,
                    "mitigation": r.mitigation,
                }
                for r in risks
            ]

            if risk_data:
                algorithmic_risks[concept.concept_id] = risk_data
                all_risks.extend(risks)

        self._report_progress(0.35, f"Assessed {len(algorithmic_risks)} concepts")

        # v3.0: CausalImpactAnalyzer로 인과 관계 분석
        self._report_progress(0.4, "Running Causal Impact Analysis (v3.0)")

        causal_results = {}
        try:
            # 테이블 간 인과 관계 분석
            tables_for_causal = list(context.tables.keys())
            if len(tables_for_causal) >= 2:
                causal_graph = self.causal_analyzer.build_causal_graph(
                    tables=tables_for_causal,
                    relationships=context.concept_relationships or [],
                )
                causal_effects = self.causal_analyzer.estimate_effects(causal_graph)
                causal_results = {
                    "causal_edges": len(causal_graph.edges) if causal_graph else 0,
                    "effects_estimated": len(causal_effects) if causal_effects else 0,
                    "key_drivers": causal_effects[:5] if causal_effects else [],
                }
                logger.info(f"CausalImpactAnalyzer found {causal_results.get('causal_edges', 0)} causal edges")
        except Exception as e:
            logger.warning(f"CausalImpactAnalyzer failed: {e}")

        # v3.0: AnomalyDetector로 이상치 탐지
        self._report_progress(0.45, "Running Anomaly Detection (v3.0)")

        anomaly_results = {}
        try:
            # v22.1: 전체 데이터 로드 (샘플링 금지)
            sample_data = context.get_all_full_data()
            if sample_data:
                anomalies = self.anomaly_detector.detect_anomalies(sample_data)
                anomaly_results = {
                    "anomalies_found": len(anomalies) if anomalies else 0,
                    "anomaly_details": [
                        {
                            "table": a.table_name,
                            "column": a.column_name,
                            "type": a.anomaly_type.value if hasattr(a.anomaly_type, 'value') else str(a.anomaly_type),
                            "severity": a.severity.value if hasattr(a.severity, 'value') else str(a.severity),
                            "description": a.description,
                        }
                        for a in (anomalies or [])[:10]
                    ],
                }
                logger.info(f"AnomalyDetector found {anomaly_results.get('anomalies_found', 0)} anomalies")
        except Exception as e:
            logger.warning(f"AnomalyDetector failed: {e}")

        # === v16.0: Column-level Lineage Tracking ===
        lineage_results = {}
        self._report_progress(0.47, "Building Column-level Lineage (v16.0)")
        try:
            if LINEAGE_AVAILABLE and build_lineage_from_context:
                lineage_tracker = build_lineage_from_context(context)
                lineage_stats = lineage_tracker.get_statistics()

                # 모든 고위험 개념에 대해 영향 분석 실행
                impact_analyses = []
                for concept in context.ontology_concepts[:5]:  # 상위 5개
                    # 개념의 소스 테이블에서 첫 번째 컬럼을 기준으로 영향 분석
                    if concept.source_tables:
                        for table in concept.source_tables[:1]:
                            # 테이블의 첫 번째 컬럼으로 영향 분석
                            if table in context.tables:
                                table_info = context.tables[table]
                                if table_info.columns:
                                    first_col = table_info.columns[0].get("name", table_info.columns[0]) if isinstance(table_info.columns[0], dict) else table_info.columns[0]
                                    col_id = f"{table}.{first_col}"
                                    impact = lineage_tracker.get_impact_analysis(col_id)
                                    if impact:
                                        impact_analyses.append({
                                            "concept_id": concept.concept_id,
                                            "source_column": col_id,
                                            "direct_downstream": impact.direct_downstream,
                                            "total_affected": impact.total_affected,
                                            "max_depth": impact.max_depth,
                                            "affected_tables": list(impact.affected_tables)[:5],
                                        })

                lineage_results = {
                    "total_nodes": lineage_stats.get("total_nodes", 0),
                    "total_edges": lineage_stats.get("total_edges", 0),
                    "tables_tracked": lineage_stats.get("tables_count", 0),
                    "impact_analyses": impact_analyses,
                    "by_edge_type": lineage_stats.get("by_edge_type", {}),
                }
                logger.info(
                    f"[v16.0] Lineage Tracker: {lineage_results['total_nodes']} nodes, "
                    f"{lineage_results['total_edges']} edges, {len(impact_analyses)} impact analyses"
                )

                # context에 lineage 저장
                context.set_dynamic("lineage_graph", lineage_tracker.to_dict())
            else:
                logger.info("[v16.0] Lineage Tracker not available, skipping")
        except Exception as e:
            logger.warning(f"[v16.0] Lineage Tracker failed: {e}")
            import traceback
            logger.debug(traceback.format_exc())

        # === v22.0: What-If Analyzer로 리스크 시나리오 시뮬레이션 ===
        whatif_risk_results = {"scenarios_analyzed": 0, "high_impact_scenarios": 0}
        try:
            whatif_analyzer = self.get_v17_service("whatif_analyzer")
            if whatif_analyzer and all_risks:
                # 고위험 개념들에 대한 What-If 시나리오 분석
                high_risk_concepts = [c for c in context.ontology_concepts if
                    any(r.risk_level in ["high", "critical"] for r in self.risk_assessor.assess_concept_risk({
                        "concept_id": c.concept_id, "concept_type": c.concept_type,
                        "name": c.name, "confidence": c.confidence,
                        "source_evidence": c.source_evidence, "source_tables": c.source_tables,
                    }))]

                for concept in high_risk_concepts[:5]:  # 상위 5개 고위험 개념
                    scenario = {
                        "scenario_type": "risk_mitigation",
                        "concept_id": concept.concept_id,
                        "hypothesis": f"What if {concept.name} data quality improves by 20%?",
                    }
                    impact_result = whatif_analyzer.analyze_impact(scenario=scenario, context=context)
                    if impact_result:
                        whatif_risk_results["scenarios_analyzed"] += 1
                        if impact_result.get("impact_score", 0) > 0.5:
                            whatif_risk_results["high_impact_scenarios"] += 1

                if whatif_risk_results["scenarios_analyzed"] > 0:
                    logger.info(f"[v22.0] What-If Risk Analysis: {whatif_risk_results['scenarios_analyzed']} scenarios, "
                               f"{whatif_risk_results['high_impact_scenarios']} high-impact")
        except Exception as e:
            logger.debug(f"[v22.0] What-If Analyzer skipped in RiskAssessor: {e}")

        # === v6.1: Counterfactual Reasoning - "만약 ~했다면?" 분석 ===
        counterfactual_results = []
        self._report_progress(0.50, "Running Counterfactual Reasoning (v6.1)")
        try:
            # v6.1 개선: 반사실적 분석 대상 확장
            # - 모든 리스크 레벨 포함 (high, critical, medium, low)
            # - 최소 2개 개념 보장 (기능 검증용)
            analysis_concepts = []
            for c in context.ontology_concepts[:10]:
                risks = self.risk_assessor.assess_concept_risk({
                    "concept_id": c.concept_id,
                    "concept_type": c.concept_type,
                    "name": c.name,
                    "confidence": c.confidence,
                    "source_evidence": c.source_evidence,
                    "source_tables": c.source_tables,
                })
                if any(r.risk_level in ["high", "critical", "medium", "low"] for r in risks):
                    analysis_concepts.append(c)

            # 최소 2개 후보 보장 (반사실 분석 기능 활성화)
            if len(analysis_concepts) < 2 and len(context.ontology_concepts) >= 2:
                for c in context.ontology_concepts[:3]:
                    if c not in analysis_concepts:
                        analysis_concepts.append(c)
                        if len(analysis_concepts) >= 2:
                            break

            high_risk_concepts = analysis_concepts[:5]  # 최대 5개

            for concept in high_risk_concepts:
                # 반사실적 시나리오: "만약 이 데이터 품질이 더 좋았다면?"
                cf_analysis = self._run_counterfactual_scenario(concept, context)
                if cf_analysis:
                    counterfactual_results.append({
                        "concept_id": concept.concept_id,
                        "concept_name": concept.name,
                        "scenario": cf_analysis.get("scenario", ""),
                        "factual_outcome": cf_analysis.get("factual", 0),
                        "counterfactual_outcome": cf_analysis.get("counterfactual", 0),
                        "potential_improvement": cf_analysis.get("improvement", 0),
                        "recommendation": cf_analysis.get("recommendation", ""),
                    })

            logger.info(f"Counterfactual Reasoning: Analyzed {len(counterfactual_results)} scenarios")
        except Exception as e:
            logger.warning(f"Counterfactual Reasoning failed: {e}")

        # 2단계: 액션 리스크 평가
        action_risks = self._assess_action_risks(context.action_backlog)

        self._report_progress(0.55, "Enhancing risk analysis with LLM")

        # 3단계: LLM을 통한 보강
        enhanced_analysis = await self._enhance_risk_analysis(
            algorithmic_risks, action_risks, context
        )

        self._report_progress(0.8, "Computing risk summary")

        # v2.1: Business Insights를 직렬화 가능한 형태로 변환
        insights_serialized = []
        for i in business_insights:
            d = {
                "insight_id": i.insight_id,
                "title": i.title,
                "description": i.description,
                "insight_type": i.insight_type.value,
                "severity": i.severity,
                "source_table": i.source_table,
                "source_column": i.source_column,
                "metric_value": i.metric_value,
                "recommendations": i.recommendations,
                "business_impact": i.business_impact,
                # v19.0: Palantir-style prescriptive fields
                "hypothesis": getattr(i, '_hypothesis', ''),
                "evidence": getattr(i, '_evidence', []),
                "validation_summary": getattr(i, '_validation_summary', ''),
                "prescriptive_actions": getattr(i, '_prescriptive_actions', []),
                "confidence_breakdown": getattr(i, '_confidence_breakdown', {}),
                # v19.4c: 누락 필드 추가
                "algorithm_findings": getattr(i, '_algorithm_findings', {}),
                "simulation_evidence": getattr(i, '_simulation_evidence', {}),
                "phases_completed": getattr(i, '_phases_completed', []),
                "source_system": getattr(i, 'source_system', 'UnifiedInsightPipeline'),
            }
            insights_serialized.append(d)

        # v19.2: Palantir-style insights를 palantir_insights로 라우팅
        logger.info(f"[v22.0] insights_serialized: {len(insights_serialized)}, has_hypothesis: {sum(1 for d in insights_serialized if d.get('hypothesis'))}")
        palantir_routed = []
        for d in insights_serialized:
            if d.get("hypothesis") and d.get("evidence"):
                palantir_routed.append({
                    "title": d.get("title", ""),
                    "hypothesis": d["hypothesis"],
                    "evidence": d["evidence"],
                    "validation_summary": d.get("validation_summary", ""),
                    "prescriptive_actions": d.get("prescriptive_actions", []),
                    "confidence_breakdown": d.get("confidence_breakdown", {}),
                    "severity": d.get("severity", "medium"),
                    "source_table": d.get("source_table", ""),
                    "source_system": "UnifiedInsightPipeline",
                    # v19.4c: 누락 필드 추가
                    "algorithm_findings": d.get("algorithm_findings", {}),
                    "simulation_evidence": d.get("simulation_evidence", {}),
                    "phases_completed": d.get("phases_completed", []),
                })

        # 4단계: 리스크 요약
        high_severity_insights = [i for i in business_insights if i.severity in ("critical", "high")]
        risk_summary = {
            "critical_risks": len([r for r in all_risks if r.risk_level == "critical"]),
            "high_risks": len([r for r in all_risks if r.risk_level == "high"]),
            "medium_risks": len([r for r in all_risks if r.risk_level == "medium"]),
            "low_risks": len([r for r in all_risks if r.risk_level == "low"]),
            "top_concerns": enhanced_analysis.get("top_concerns", []),
            "business_insights_count": len(business_insights),  # v2.1
            "critical_high_insights": len(high_severity_insights),  # v2.1
            # v17.1: Phase 1 operational_insights 통합
            "operational_risk_factors": len(operational_risk_factors),
            "problematic_status_fields": len(problematic_status_fields),
            # v22.0: OWL2/SHACL 리스크 요소
            "owl2_shacl_risk_factors": len(owl2_risk_factors),
        }

        self._report_progress(0.95, "Finalizing risk assessment")

        # v22.0: Agent Learning - 경험 기록
        total_risks = risk_summary["critical_risks"] + risk_summary["high_risks"]
        self.record_experience(
            experience_type="governance_decision",
            input_data={"task": "risk_assessment", "concepts": len(algorithmic_risks)},
            output_data={"critical_high_risks": total_risks, "insights_generated": len(business_insights)},
            outcome="success" if total_risks < len(algorithmic_risks) / 2 else "partial",
            confidence=0.75,
        )

        return TodoResult(
            success=True,
            output={
                "concepts_assessed": len(algorithmic_risks),
                "risk_summary": risk_summary,
                "business_insights_generated": len(business_insights),  # v2.1
                "counterfactual_scenarios_analyzed": len(counterfactual_results),  # v6.1
                # v16.0: Lineage 통계
                "lineage_nodes": lineage_results.get("total_nodes", 0),
                "lineage_edges": lineage_results.get("total_edges", 0),
                "impact_analyses_run": len(lineage_results.get("impact_analyses", [])),
            },
            context_updates={
                "business_insights": insights_serialized,  # v2.1
                "counterfactual_analysis": counterfactual_results,  # v6.1
                "palantir_insights": palantir_routed,  # v19.2: Palantir-style 라우팅
            },
            metadata={
                "concept_risks": enhanced_analysis.get("concept_risks", algorithmic_risks),
                "action_risks": enhanced_analysis.get("action_risks", action_risks),
                "business_insights_summary": insights_serialized[:10],  # v2.1
                "counterfactual_details": counterfactual_results,  # v6.1
                # v16.0: Lineage & Impact Analysis
                "lineage_results": lineage_results,
                "causal_results": causal_results,
                "anomaly_results": anomaly_results,
                # v17.1: Phase 1 operational_insights 통합
                "operational_risk_factors": operational_risk_factors,
                "problematic_status_fields": problematic_status_fields,
            },
        )

    def _build_correlations_from_causal(self, context) -> list:
        """
        v19.0: 인과분석 treatment_effects → CorrelationResult 변환 (SimulationEngine 입력용)

        CausalImpactAnalyzer의 significant treatment effects를 CorrelationResult로 변환하여
        SimulationEngine의 segmentation/threshold/bootstrap 실험에 사용.
        """
        from ..analysis.cross_entity_correlation import CorrelationResult
        import numpy as np

        correlations = []
        try:
            causal_insights = getattr(context, 'causal_insights', {}) or {}
            impact_analysis = causal_insights.get("impact_analysis", {})
            treatment_effects = impact_analysis.get("treatment_effects", [])

            # 첫 번째 테이블명 가져오기 (단일 테이블 시나리오)
            table_names = list(context.tables.keys()) if hasattr(context, 'tables') else []
            default_table = table_names[0] if table_names else "unknown"

            for te in treatment_effects:
                if not te.get("significant", False):
                    continue

                treatment_col = te.get("treatment", "")
                outcome_col = te.get("outcome", "")
                p_value = te.get("p_value", 1.0)
                ate = te.get("estimate", 0)
                sample_size = te.get("sample_size", 0)

                if not treatment_col or not outcome_col:
                    continue

                # ATE를 상관계수로 근사 (방향 유지)
                corr_approx = min(max(ate / 100.0, -1.0), 1.0) if abs(ate) < 100 else (1.0 if ate > 0 else -1.0)

                correlations.append(CorrelationResult(
                    source_table=default_table,
                    source_column=treatment_col,
                    target_table=default_table,
                    target_column=outcome_col,
                    correlation=corr_approx,
                    p_value=p_value,
                    sample_size=sample_size,
                    relationship_direction="positive" if ate > 0 else "negative",
                ))

            logger.info(f"[v19.0] Built {len(correlations)} correlations from {len(treatment_effects)} treatment effects")
        except Exception as e:
            logger.warning(f"[v19.0] Failed to build correlations from causal: {e}")

        return correlations

    def _assess_action_risks(
        self,
        actions: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """액션 리스크 평가"""
        action_risks = []

        for action in (actions or []):
            priority = action.get("calculated_priority", action.get("priority", "medium"))
            effort = action.get("effort", "medium")

            # 구현 리스크
            impl_risk = "high" if effort == "high" else "medium" if effort == "medium" else "low"

            # 미이행 리스크
            risk_if_not_done = "critical" if priority == "critical" else \
                              "high" if priority == "high" else "medium"

            action_risks.append({
                "action_id": action.get("action_id", ""),
                "implementation_risk": impl_risk,
                "risk_if_not_done": risk_if_not_done,
            })

        return action_risks

    def _run_counterfactual_scenario(
        self,
        concept: Any,
        context: "SharedContext",
    ) -> Optional[Dict[str, Any]]:
        """
        v6.1: Counterfactual Reasoning - 반사실적 시나리오 분석

        "만약 데이터 품질이 더 좋았다면 리스크가 어떻게 달라졌을까?"
        "만약 이 데이터를 통합했다면 KPI가 어떻게 변했을까?"

        Args:
            concept: 분석 대상 개념
            context: SharedContext

        Returns:
            반사실적 분석 결과
        """
        import numpy as np

        try:
            # 현재 상태 (Factual)
            current_confidence = concept.confidence if hasattr(concept, 'confidence') else 0.5
            current_quality = current_confidence * 100  # 품질 점수로 변환

            # 반사실적 상태 (Counterfactual) - "만약 품질이 90%였다면?"
            target_quality = 0.9

            # 리스크 점수 계산 (간단한 모델)
            # risk = base_risk * (1 - quality) + external_factors
            base_risk = 0.5  # 기본 리스크
            external_risk = 0.1  # 외부 요인

            factual_risk = base_risk * (1 - current_confidence) + external_risk
            counterfactual_risk = base_risk * (1 - target_quality) + external_risk

            # 개선 효과
            improvement = (factual_risk - counterfactual_risk) / max(factual_risk, 0.01) * 100

            # 시나리오 설명
            scenario = (
                f"If data quality for '{concept.name}' improved from "
                f"{current_confidence*100:.0f}% to {target_quality*100:.0f}%"
            )

            # 권장사항
            if improvement > 20:
                recommendation = "High impact: Prioritize data quality improvement"
            elif improvement > 10:
                recommendation = "Moderate impact: Consider quality enhancement"
            else:
                recommendation = "Low impact: Current quality acceptable"

            return {
                "scenario": scenario,
                "factual": round(factual_risk, 3),
                "counterfactual": round(counterfactual_risk, 3),
                "improvement": round(improvement, 1),
                "recommendation": recommendation,
                "current_quality": round(current_confidence * 100, 1),
                "target_quality": round(target_quality * 100, 1),
            }

        except Exception as e:
            logger.warning(f"Counterfactual scenario failed for {concept.concept_id}: {e}")
            return None

    async def _enhance_risk_analysis(
        self,
        concept_risks: Dict[str, List[Dict]],
        action_risks: List[Dict],
        context: "SharedContext",
    ) -> Dict[str, Any]:
        """LLM을 통한 리스크 분석 보강"""
        # 주요 리스크만 추출 (상위 10개 개념)
        top_risks = dict(list(concept_risks.items())[:10])

        instruction = f"""Enhance this risk analysis with mitigation strategies.

Domain: {context.get_industry()}

Concept Risks (algorithmic):
{json.dumps(top_risks, indent=2, ensure_ascii=False)}

Action Risks:
{json.dumps(action_risks[:10], indent=2, ensure_ascii=False)}

Provide:
1. Validate risk levels
2. Prioritize top concerns
3. Suggest specific mitigation strategies
4. Identify dependencies between risks

Return JSON:
```json
{{
  "concept_risks": {{
    "<concept_id>": [
      {{
        "category": "<category>",
        "likelihood": <int 1-5>,
        "impact": <int 1-5>,
        "risk_score": <int>,
        "risk_level": "<level>",
        "mitigation": "<enhanced mitigation strategy>",
        "mitigation_effort": "<high|medium|low>",
        "mitigation_priority": "<critical|high|medium|low>"
      }}
    ]
  }},
  "action_risks": [
    {{
      "action_id": "<id>",
      "implementation_risk": "<level>",
      "risk_if_not_done": "<level>",
      "notes": "<specific notes>"
    }}
  ],
  "top_concerns": ["concern1", "concern2"],
  "risk_dependencies": ["risk A affects risk B"]
}}
```"""

        response = await self.call_llm(instruction, max_tokens=4000)
        result = parse_llm_json(response, default={})

        if not result or not isinstance(result, dict):
            return {
                "concept_risks": concept_risks,
                "action_risks": action_risks,
                "top_concerns": [],
            }

        return result


class PolicyGeneratorAutonomousAgent(AutonomousAgent):
    """
    정책 생성 자율 에이전트 (v3.1)

    Algorithm-First 접근:
    1. 템플릿 기반 표준 정책 생성
    2. 개념 유형별 맞춤 규칙
    3. KPIPredictor로 시계열 예측 및 ROI 분석
    4. TimeSeriesForecaster로 메트릭 예측 (v3.1 신규)
    5. LLM을 통한 도메인 특화 커스터마이징

    v3.0: KPIPredictor 통합
    v3.1: TimeSeriesForecaster 통합 - 정책 효과 예측
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # v14.0: 도메인 하드코딩 제거 - Lazy initialization (execute_task에서 context 기반 설정)
        self.policy_generator = None
        self._current_domain = None
        # v3.0: KPI 예측 및 ROI 분석 통합
        self.kpi_predictor = KPIPredictor()
        # v3.1: 시계열 예측 (정책 효과 예측용)
        self.time_series_forecaster = TimeSeriesForecaster()

    @property
    def agent_type(self) -> str:
        return "policy_generator"

    @property
    def agent_name(self) -> str:
        return "Policy Generator"

    @property
    def phase(self) -> str:
        return "governance"

    def _get_domain_policy_examples(self, domain: str) -> str:
        """v14.0: 도메인별 정책 예시 반환"""
        domain_lower = domain.lower()

        if "healthcare" in domain_lower or "medical" in domain_lower:
            return """**Completeness Check:**
```yaml
rule_id: DQ_COMPLETENESS_001
rule_name: Patient ID Completeness Check
rule_type: data_quality
target: object_type.Patient
target_field: patient_id
condition: |
  SELECT COUNT(*) / TOTAL_COUNT * 100 AS completeness_rate
  FROM patient_table
  WHERE patient_id IS NOT NULL
threshold: ">= 99.5%"
severity: critical
```

## Healthcare-Specific:
- HIPAA compliance for PHI data
- Audit logging for sensitive access
- Role-based restrictions for clinical data"""

        elif "supply" in domain_lower or "logistics" in domain_lower:
            return """**Completeness Check:**
```yaml
rule_id: DQ_COMPLETENESS_001
rule_name: Order ID Completeness Check
rule_type: data_quality
target: object_type.Order
target_field: order_id
condition: |
  SELECT COUNT(*) / TOTAL_COUNT * 100 AS completeness_rate
  FROM order_list
  WHERE order_id IS NOT NULL
threshold: ">= 99.9%"
severity: critical
```

## Supply Chain-Specific:
- Inventory accuracy monitoring
- Delivery SLA compliance tracking
- Supplier data quality validation"""

        elif "finance" in domain_lower or "banking" in domain_lower:
            return """**Completeness Check:**
```yaml
rule_id: DQ_COMPLETENESS_001
rule_name: Transaction ID Completeness Check
rule_type: data_quality
target: object_type.Transaction
target_field: transaction_id
condition: |
  SELECT COUNT(*) / TOTAL_COUNT * 100 AS completeness_rate
  FROM transactions
  WHERE transaction_id IS NOT NULL
threshold: ">= 99.99%"
severity: critical
```

## Finance-Specific:
- SOX/PCI compliance for financial data
- Fraud detection alerting
- Audit trail requirements"""

        elif "ecommerce" in domain_lower or "retail" in domain_lower:
            return """**Completeness Check:**
```yaml
rule_id: DQ_COMPLETENESS_001
rule_name: Product SKU Completeness Check
rule_type: data_quality
target: object_type.Product
target_field: sku
condition: |
  SELECT COUNT(*) / TOTAL_COUNT * 100 AS completeness_rate
  FROM products
  WHERE sku IS NOT NULL
threshold: ">= 99.5%"
severity: high
```

## E-commerce-Specific:
- GDPR/CCPA compliance for customer PII
- Pricing accuracy monitoring
- Inventory sync validation"""

        else:
            return """**Completeness Check:**
```yaml
rule_id: DQ_COMPLETENESS_001
rule_name: Primary Key Completeness Check
rule_type: data_quality
target: object_type.Entity
target_field: id
condition: |
  SELECT COUNT(*) / TOTAL_COUNT * 100 AS completeness_rate
  FROM target_table
  WHERE id IS NOT NULL
threshold: ">= 99.0%"
severity: high
```

## General Best Practices:
- Data privacy compliance
- Audit logging for sensitive data
- Role-based access controls"""

    def get_system_prompt(self) -> str:
        # v14.0: 도메인 적응형 프롬프트 생성
        domain = getattr(self, '_current_domain', None) or "general"
        domain_examples = self._get_domain_policy_examples(domain)

        return f"""You are a Policy Generation Expert for data governance automation in the **{domain}** domain.

Your role is to CUSTOMIZE and ENHANCE algorithmically generated policies.

## Policy Types:
1. **Data Quality Rules**: Completeness, accuracy, consistency, timeliness
2. **Access Control Policies**: RBAC, ABAC, sensitivity classification
3. **Monitoring Rules**: Threshold alerts, pattern detection, SLA monitoring

## Policy Requirements:
- Be SPECIFIC with thresholds (99.9%, not "high")
- Include SQL-like conditions where applicable
- Define clear violation actions
- Assign appropriate alert recipients

## SQL-like Condition Format

Generate policies with executable SQL-like conditions:

### Data Quality Rule Template
```yaml
rule_id: DQ_<TYPE>_<SEQUENCE>
rule_name: <Descriptive Name>
rule_type: data_quality
target: object_type.<ConceptType>
target_field: <field_name>
condition: |
  SELECT COUNT(*) / TOTAL_COUNT * 100 AS metric_rate
  FROM {{target_table}}
  WHERE {{field_condition}}
threshold: ">= <percentage>%"
severity: <critical|high|medium|low>
action_on_violation:
  - alert: [<role_1>, <role_2>]
  - log: audit_table
  - block: downstream_pipeline  # optional
schedule: "<cron_expression>"
```

### Common SQL Conditions:
- Completeness: `WHERE {{field}} IS NOT NULL`
- Uniqueness: `SELECT COUNT(DISTINCT {{field}}) = COUNT(*)`
- Range Check: `WHERE {{field}} BETWEEN {{min}} AND {{max}}`
- Format Validation: `WHERE {{field}} REGEXP '{{pattern}}'`
- Referential Integrity: `WHERE {{fk_field}} IN (SELECT {{pk_field}} FROM {{ref_table}})`

### {domain.replace('_', ' ').title()} Domain Example Policies:

{domain_examples}

## Output Format
```json
{{
  "generated_policies": [
    {{
      "rule_id": "<TYPE>_<CATEGORY>_<SEQ>",
      "rule_name": "<descriptive name>",
      "rule_type": "<data_quality|access_control|monitoring>",
      "target_concept": "<concept_id>",
      "target_field": "<field_name or null>",
      "condition_sql": "<SQL-like condition>",
      "threshold": "<comparison operator + value>",
      "severity": "<critical|high|medium|low>",
      "action_on_violation": {{
        "alert": ["<role_1>", "<role_2>"],
        "log": "<table_name>",
        "block": "<pipeline_name or null>"
      }},
      "schedule": "<cron expression>",
      "rationale": "<why this policy is needed>"
    }}
  ]
}}
```

Respond ONLY with valid JSON."""

    async def execute_task(
        self,
        todo: "PipelineTodo",
        context: "SharedContext",
    ) -> "TodoResult":
        """정책 생성 작업 실행"""
        from ...todo.models import TodoResult
        from ..analysis.governance import PolicyGenerator

        # v14.0: 도메인 하드코딩 제거 - context에서 동적으로 도메인 가져오기
        domain = context.get_industry() or "general"
        if self.policy_generator is None or self._current_domain != domain:
            self.policy_generator = PolicyGenerator(domain=domain)
            self._current_domain = domain
            logger.info(f"[v14.0] PolicyGenerator initialized with domain: {domain}")

        self._report_progress(0.1, "Generating algorithmic policies")

        # 1단계: 승인된 개념에 대한 템플릿 기반 정책 생성
        approved_concepts = [
            c for c in context.ontology_concepts
            if c.status == "approved"
        ]

        algorithmic_policies = []
        for concept in approved_concepts:
            concept_dict = {
                "concept_id": concept.concept_id,
                "concept_type": concept.concept_type,
                "name": concept.name,
                "definition": concept.definition,
                "source_tables": concept.source_tables,
            }

            policies = self.policy_generator.generate_policies(concept_dict)
            for policy in policies:
                algorithmic_policies.append({
                    "rule_id": policy.rule_id,
                    "rule_name": policy.rule_name,
                    "rule_type": policy.rule_type,
                    "target_concept": policy.target_concept,
                    "condition": policy.condition,
                    "severity": policy.severity,
                    "action_on_violation": policy.action_on_violation,
                    "enabled": True,
                })

        self._report_progress(0.35, f"Generated {len(algorithmic_policies)} base policies")

        # v3.0: KPIPredictor로 시계열 예측 및 ROI 분석
        self._report_progress(0.4, "Running KPI Prediction & ROI Analysis (v3.0)")

        kpi_results = {}
        try:
            # v22.1: 전체 데이터 로드 (샘플링 금지)
            sample_data = context.get_all_full_data()

            if sample_data:
                # 시계열 KPI 예측
                predictions = self.kpi_predictor.predict_kpis(sample_data)

                # ROI 분석
                roi_analysis = self.kpi_predictor.analyze_roi(
                    context.governance_decisions or [],
                    context.action_backlog or [],
                )

                # 도메인별 KPI 계산
                domain_kpis = self.kpi_predictor.calculate_domain_kpis(
                    sample_data,
                    domain=context.get_industry(),
                )

                kpi_results = {
                    "predictions_made": len(predictions) if predictions else 0,
                    "key_predictions": predictions[:5] if predictions else [],
                    "roi_estimate": roi_analysis.npv if roi_analysis else 0.0,  # npv 사용
                    "payback_period": roi_analysis.break_even_month if roi_analysis else 0,  # break_even_month 사용
                    "irr": roi_analysis.irr if roi_analysis else 0.0,
                    "domain_kpis": domain_kpis[:10] if domain_kpis else [],
                }
                logger.info(f"KPIPredictor: {kpi_results.get('predictions_made', 0)} predictions, ROI={kpi_results.get('roi_estimate', 0):.2f}")
        except Exception as e:
            logger.warning(f"KPIPredictor failed: {e}")

        # v3.1: TimeSeriesForecaster로 정책 효과 예측
        self._report_progress(0.45, "Running Time Series Forecasting (v3.1)")

        forecast_results = {}
        try:
            # v22.1: 전체 데이터 로드 (샘플링 금지)
            sample_data = context.get_all_full_data()

            if sample_data and isinstance(sample_data, dict):
                # 시계열 메트릭 예측 (예: 데이터 품질 트렌드)
                for table_name, rows in sample_data.items():
                    if rows and len(rows) >= 10:
                        # 숫자형 컬럼 찾기
                        numeric_cols = []
                        for key in rows[0].keys():
                            if isinstance(rows[0].get(key), (int, float)):
                                numeric_cols.append(key)

                        for col_name in numeric_cols[:3]:  # 최대 3개 컬럼
                            values = [row.get(col_name, 0) for row in rows if row.get(col_name) is not None]
                            if len(values) >= 10:
                                forecast = self.time_series_forecaster.forecast(
                                    series_name=f"{table_name}.{col_name}",
                                    series=values,  # 'data' -> 'series' (올바른 파라미터명)
                                    horizon=5,  # 5 스텝 예측
                                )
                                if forecast:
                                    forecast_results[f"{table_name}.{col_name}"] = {
                                        "predictions": forecast.predictions[:5],
                                        "trend": forecast.trend.value if hasattr(forecast.trend, 'value') else forecast.trend,
                                        "model": forecast.model_used.value if hasattr(forecast.model_used, 'value') else forecast.model_used,
                                        "confidence": forecast.confidence,
                                    }

                logger.info(f"TimeSeriesForecaster: {len(forecast_results)} forecasts generated")
        except Exception as e:
            logger.warning(f"TimeSeriesForecaster failed: {e}")

        # 2단계: LLM을 통한 커스터마이징
        self._report_progress(0.55, "Customizing policies with LLM")

        enhanced_policies = await self._enhance_policies_with_llm(
            algorithmic_policies, approved_concepts, context
        )

        self._report_progress(0.8, "Generating access policies and monitoring")

        # 3단계: 접근 정책 및 모니터링 대시보드 생성
        access_policies = self._generate_access_policies(approved_concepts)
        monitoring_dashboards = self._generate_monitoring_dashboards(approved_concepts)

        self._report_progress(0.95, "Finalizing policies")

        # === v22.0: version_manager로 정책 버전 커밋 ===
        policy_version = None
        try:
            version_manager = self.get_v17_service("version_manager")
            if version_manager and enhanced_policies:
                policy_version = await version_manager.commit(
                    content_type="policy",
                    content={
                        "policies": enhanced_policies,
                        "access_policies": access_policies,
                        "monitoring_dashboards": monitoring_dashboards,
                    },
                    message=f"Policy generation: {len(enhanced_policies)} rules created",
                    tags=["governance", "policy", f"rules_{len(enhanced_policies)}"],
                )
                if policy_version:
                    logger.info(f"[v22.0] VersionManager committed policy version: {policy_version}")
        except Exception as e:
            logger.debug(f"[v22.0] VersionManager skipped: {e}")

        # 집계
        summary = {
            "total_rules": len(enhanced_policies),
            "by_type": {
                "data_quality": len([p for p in enhanced_policies if p.get("rule_type") == "data_quality"]),
                "access_control": len([p for p in enhanced_policies if p.get("rule_type") == "access_control"]),
                "monitoring": len([p for p in enhanced_policies if p.get("rule_type") == "monitoring"]),
            },
            "by_severity": {
                "critical": len([p for p in enhanced_policies if p.get("severity") == "critical"]),
                "high": len([p for p in enhanced_policies if p.get("severity") == "high"]),
                "medium": len([p for p in enhanced_policies if p.get("severity") == "medium"]),
                "low": len([p for p in enhanced_policies if p.get("severity") == "low"]),
            },
            "policy_version": policy_version,  # v22.0
        }

        # v22.0: Agent Learning - 경험 기록
        self.record_experience(
            experience_type="governance_decision",
            input_data={"task": "policy_generation", "concepts": len(algorithmic_rules)},
            output_data={"policies_generated": len(enhanced_policies), "by_type": summary["by_type"]},
            outcome="success",
            confidence=0.8,
        )

        return TodoResult(
            success=True,
            output={
                "policies_generated": len(enhanced_policies),
                "summary": summary,
                # v3.1: 시계열 예측 결과 요약
                "forecasts_generated": len(forecast_results),
            },
            context_updates={
                "policy_rules": enhanced_policies,
            },
            metadata={
                "access_policies": access_policies,
                "monitoring_dashboards": monitoring_dashboards,
                # v3.1: 시계열 예측 상세 결과
                "time_series_forecasts": forecast_results,
            },
        )

    def _generate_access_policies(
        self,
        concepts: List,
    ) -> List[Dict[str, Any]]:
        """접근 정책 생성"""
        access_policies = []

        for concept in concepts:
            name = concept.name.lower()

            # PHI 데이터 확인
            is_phi = any(kw in name for kw in ["patient", "diagnosis", "medication"])

            if is_phi:
                access_policies.append({
                    "policy_id": f"access_{concept.concept_id}",
                    "concept_id": concept.concept_id,
                    "access_level": "restricted",
                    "allowed_roles": ["physician", "nurse", "admin"],
                    "conditions": ["has_hipaa_training = true", "is_active_employee = true"],
                    "audit_required": True,
                })
            else:
                access_policies.append({
                    "policy_id": f"access_{concept.concept_id}",
                    "concept_id": concept.concept_id,
                    "access_level": "standard",
                    "allowed_roles": ["analyst", "developer", "admin"],
                    "conditions": [],
                    "audit_required": False,
                })

        return access_policies

    def _generate_monitoring_dashboards(
        self,
        concepts: List,
    ) -> List[Dict[str, Any]]:
        """모니터링 대시보드 생성"""
        # 핵심 개념들에 대한 대시보드
        object_types = [c for c in concepts if c.concept_type == "object_type"]

        dashboards = []

        if object_types:
            dashboards.append({
                "dashboard_id": "data_quality_overview",
                "name": "Data Quality Overview",
                "metrics": [
                    f"{c.name}_completeness" for c in object_types[:5]
                ] + [
                    "overall_quality_score",
                    "rule_violations_24h",
                ],
                "refresh_interval": "5m",
            })

            dashboards.append({
                "dashboard_id": "governance_health",
                "name": "Governance Health",
                "metrics": [
                    "approved_concepts",
                    "pending_reviews",
                    "policy_violations",
                    "risk_alerts",
                ],
                "refresh_interval": "1h",
            })

        return dashboards

    async def _enhance_policies_with_llm(
        self,
        policies: List[Dict[str, Any]],
        concepts: List,
        context: "SharedContext",
    ) -> List[Dict[str, Any]]:
        """LLM을 통한 정책 커스터마이징"""
        if not policies:
            return []

        # 개념 정보 요약
        concept_summaries = [
            {"name": c.name, "type": c.concept_type, "tables": c.source_tables[:2]}
            for c in concepts[:10]
        ]

        instruction = f"""Customize these data governance policies for the domain.

Domain: {context.get_industry()}

Base Policies (algorithmically generated):
{json.dumps(policies[:20], indent=2, ensure_ascii=False)}

Target Concepts:
{json.dumps(concept_summaries, indent=2, ensure_ascii=False)}

For each policy:
1. Adjust thresholds for the domain
2. Add specific conditions
3. Define appropriate alert recipients
4. Add description for documentation

Return JSON:
```json
{{
  "enhanced_policies": [
    {{
      "rule_id": "<existing>",
      "rule_name": "<improved name>",
      "rule_type": "<existing>",
      "target_concept": "<existing>",
      "target_field": "<specific field or *>",
      "condition": "<domain-specific SQL condition>",
      "threshold": "<specific like 99.9%>",
      "severity": "<adjusted severity>",
      "action_on_violation": "<specific action>",
      "alert_recipients": ["role1", "role2"],
      "description": "<human readable description>",
      "enabled": true
    }}
  ]
}}
```"""

        response = await self.call_llm(instruction, max_tokens=4000)
        enhanced = parse_llm_json(response, key="enhanced_policies", default=[])

        if enhanced:
            # 병합: LLM 결과 + 원본 (LLM이 놓친 것)
            enhanced_ids = {p.get("rule_id") for p in enhanced}
            for policy in policies:
                if policy.get("rule_id") not in enhanced_ids:
                    enhanced.append(policy)
            return enhanced

        return policies
