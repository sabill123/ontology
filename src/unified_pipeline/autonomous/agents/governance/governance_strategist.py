"""
Governance Strategist Autonomous Agent (v11.0 - Evidence Chain & BFT Debate)

거버넌스 전략가 자율 에이전트 - 블록체인 스타일 Evidence Chain + BFT 합의
- Evidence Chain: Phase 1/2의 모든 분석 결과를 블록체인 스타일로 연결
- DebateProtocol: BFT 기반 Council 토론 (Challenge/Defense 라운드)
- 모든 거버넌스 결정에 근거 블록 ID 참조
"""

import json
import logging
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from ...base import AutonomousAgent
from ...analysis import (
    DecisionMatrix,
    parse_llm_json,
    # v3.2: 프로세스 분석 (ProcessArchitect 기능 통합)
    ProcessAnalyzer,
    ProcessContext,
    IntegrationComplexity,
    # === v6.1: Structured Debate ===
    StructuredDebateEngine,
    create_debate_summary,
)

# === v16.0: CDC Sync Engine ===
try:
    from ....sync import CDCEngine, create_cdc_engine_with_context, ChangeType, SyncStatus
    CDC_ENGINE_AVAILABLE = True
except ImportError:
    CDC_ENGINE_AVAILABLE = False
    CDCEngine = None
    create_cdc_engine_with_context = None
    ChangeType = None
    SyncStatus = None

# === v4.5: Embedded Phase 3 Agent Council from utils ===
from ..governance_utils import (
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
    from ...analysis.enhanced_validator import (
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
    from ...evidence_chain import EvidenceChain, EvidenceRegistry, EvidenceType
    from ...debate_protocol import DebateProtocol, CouncilAgent, VoteDecision
    EVIDENCE_CHAIN_AVAILABLE = True
except ImportError:
    EVIDENCE_CHAIN_AVAILABLE = False

if TYPE_CHECKING:
    from ....todo.models import PipelineTodo, TodoResult
    from ....shared_context import SharedContext

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
        from ....todo.models import TodoResult
        from ....shared_context import GovernanceDecision

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
                from ...evidence_chain import EvidenceRegistry
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
            for concept in context.ontology_concepts:  # v27.1: 하드코딩 한도 제거 — 전체 개념 스캔
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

            for concept in debate_candidates:  # v27.1: 하드코딩 한도 제거 — 전체 후보 debate
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
                        supporting_block_ids = [ev["block_id"] for ev in concept_evidence[:30]]  # v25.2: 10→30

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
                                "supporting_evidence": concept_evidence[:15],  # v25.2: 5→15
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

            # v25.2: 인과 관계 상세 정보를 reasoning에 포함 (v17.1에서 카운트만 전달하던 것 개선)
            causal_info = p2_enrichment.get("causal_info", {})
            if causal_info.get("causes") or causal_info.get("caused_by"):
                causal_causes_list = causal_info.get("causes", [])[:3]
                causal_effects_list = causal_info.get("caused_by", [])[:3]
                cause_details = ", ".join(
                    f"{c.get('target', '?')}(eff={c.get('effect', 0):.2f})" for c in causal_causes_list
                )
                effect_details = ", ".join(
                    f"{c.get('source', '?')}(eff={c.get('effect', 0):.2f})" for c in causal_effects_list
                )
                final_reasoning += f" [v25.2: Causal - causes: {cause_details}; caused_by: {effect_details}]"

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

            # v27.0: governance decision → concept.status 동기화
            for concept in context.ontology_concepts:
                if concept.concept_id == concept_id:
                    if final_decision_type == "approve":
                        concept.status = "approved"
                    elif final_decision_type == "reject":
                        concept.status = "rejected"
                    elif final_decision_type == "schedule_review":
                        concept.status = "provisional"
                    break

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
                    if decision.decision_type == "approve":
                        # 승인된 개념에 대해 CDC 구독 생성
                        concept = next((c for c in context.ontology_concepts if c.concept_id == decision.concept_id), None)
                        source_tables = concept.source_tables if concept and hasattr(concept, 'source_tables') else []
                        for source_table in (source_tables or [])[:3]:
                            subscription = cdc_engine.subscribe(
                                table_name=source_table,
                                handler=lambda e: None,  # 실제 핸들러는 나중에 등록
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
                            "decision": decision.decision_type,
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
