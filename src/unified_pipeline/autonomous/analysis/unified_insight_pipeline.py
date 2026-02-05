"""
Unified Insight Pipeline (v17.1)

통합 인사이트 파이프라인 - Palantir AIP 스타일 연결:
Algorithm (BusinessInsightsAnalyzer) → Simulation (SimulationEngine) → LLM (Reasoning)

모든 분석 컴포넌트가 연결되어 정보를 공유:
1. Algorithm Phase: 통계 기반 인사이트 도출
2. Simulation Phase: What-if 시나리오 실험
3. LLM Reasoning Phase: 최종 비즈니스 인사이트 생성

사용 예시:
    pipeline = UnifiedInsightPipeline(context, llm_client)

    # 전체 파이프라인 실행
    unified_insights = await pipeline.run(
        tables=tables_data,
        data=data,
        correlations=correlations,
    )
"""

import asyncio
import logging
import json
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, TYPE_CHECKING
from datetime import datetime
from enum import Enum

from ...model_config import get_service_model

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from ...shared_context import SharedContext


class UnifiedInsightPhase(str, Enum):
    """인사이트 생성 단계"""
    ALGORITHM = "algorithm"
    SIMULATION = "simulation"
    LLM_REASONING = "llm_reasoning"


@dataclass
class UnifiedInsight:
    """
    통합 인사이트 - Algorithm + Simulation + LLM 결과 통합

    Palantir AIP 스타일의 풍부한 인사이트:
    - 알고리즘 분석 결과
    - 시뮬레이션 증거
    - LLM 비즈니스 해석
    """
    insight_id: str
    title: str
    summary: str  # LLM이 생성한 비즈니스 요약

    # 단계별 결과
    algorithm_findings: Dict[str, Any] = field(default_factory=dict)  # BusinessInsight 데이터
    simulation_evidence: Dict[str, Any] = field(default_factory=dict)  # SimulationExperiment 데이터
    llm_interpretation: str = ""  # LLM의 비즈니스 해석

    # 메트릭
    source_table: str = ""
    source_column: str = ""
    metric_value: Any = None
    benchmark: Optional[float] = None
    deviation: Optional[float] = None

    # v19.0: Palantir-Style Structured Fields
    hypothesis: str = ""  # 가설: 무엇이 일어나고 있는가
    evidence: List[Dict[str, Any]] = field(default_factory=list)  # 다중 모델 검증 결과
    validation_summary: str = ""  # 검증 요약 (증거 종합)
    prescriptive_actions: List[Dict[str, Any]] = field(default_factory=list)  # 처방적 액션 (정량화 포함)

    # 비즈니스 컨텍스트
    business_impact: str = ""
    recommendations: List[str] = field(default_factory=list)
    action_items: List[str] = field(default_factory=list)

    # 신뢰도 및 근거
    confidence: float = 0.0
    confidence_breakdown: Dict[str, float] = field(default_factory=dict)
    # e.g., {"algorithm": 0.85, "simulation": 0.75, "llm": 0.80}

    # 분류
    insight_type: str = "operational"  # kpi, risk, anomaly, trend, operational
    severity: str = "medium"  # critical, high, medium, low, info
    domain_tags: List[str] = field(default_factory=list)

    # 메타데이터
    phases_completed: List[str] = field(default_factory=list)
    generated_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "insight_id": self.insight_id,
            "title": self.title,
            "summary": self.summary,
            # v19.0: Palantir-style structured fields
            "hypothesis": self.hypothesis,
            "evidence": self.evidence,
            "validation_summary": self.validation_summary,
            "prescriptive_actions": self.prescriptive_actions,
            # Original phase results
            "algorithm_findings": self.algorithm_findings,
            "simulation_evidence": self.simulation_evidence,
            "llm_interpretation": self.llm_interpretation,
            "source_table": self.source_table,
            "source_column": self.source_column,
            "metric_value": self.metric_value,
            "benchmark": self.benchmark,
            "deviation": self.deviation,
            "business_impact": self.business_impact,
            "recommendations": self.recommendations,
            "action_items": self.action_items,
            "confidence": round(self.confidence, 3),
            "confidence_breakdown": self.confidence_breakdown,
            "insight_type": self.insight_type,
            "severity": self.severity,
            "domain_tags": self.domain_tags,
            "phases_completed": self.phases_completed,
            "generated_at": self.generated_at,
        }


@dataclass
class PipelineResult:
    """파이프라인 실행 결과"""
    unified_insights: List[UnifiedInsight]

    # 단계별 통계
    algorithm_insights_count: int = 0
    simulation_experiments_count: int = 0
    llm_enriched_count: int = 0

    # 실행 정보
    execution_time_ms: float = 0.0
    phases_executed: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "unified_insights": [i.to_dict() for i in self.unified_insights],
            "algorithm_insights_count": self.algorithm_insights_count,
            "simulation_experiments_count": self.simulation_experiments_count,
            "llm_enriched_count": self.llm_enriched_count,
            "execution_time_ms": self.execution_time_ms,
            "phases_executed": self.phases_executed,
            "errors": self.errors,
        }


class UnifiedInsightPipeline:
    """
    통합 인사이트 파이프라인 (v17.1)

    Palantir AIP 스타일 연결:
    - BusinessInsightsAnalyzer → SimulationEngine → LLM Reasoning
    - 모든 컴포넌트가 정보를 공유하고 상호 참조
    - 최종 인사이트는 세 단계의 결과를 종합
    """

    def __init__(
        self,
        context: Optional["SharedContext"] = None,
        llm_client: Any = None,
        domain: str = "general",
        enable_simulation: bool = True,
        enable_llm: bool = True,
        pipeline_context: Optional[Dict[str, Any]] = None,
    ):
        """
        Args:
            context: SharedContext
            llm_client: LLM 클라이언트 (OpenAI 호환)
            domain: 도메인 (healthcare, supply_chain, etc.)
            enable_simulation: 시뮬레이션 활성화
            enable_llm: LLM 추론 활성화
            pipeline_context: v19.4 — Phase 2 분석 결과 (causal_insights, counterfactual 등)
        """
        self.context = context
        self.llm_client = llm_client
        self.domain = domain
        self.enable_simulation = enable_simulation
        self.enable_llm = enable_llm
        self._pipeline_context = pipeline_context or {}

        # 컴포넌트 초기화
        self._business_analyzer = None
        self._simulation_engine = None

        logger.info(f"[v17.1] UnifiedInsightPipeline initialized (domain={domain}, simulation={enable_simulation}, llm={enable_llm})")

    async def run(
        self,
        tables: Dict[str, Dict[str, Any]],
        data: Dict[str, List[Dict[str, Any]]],
        correlations: Optional[List[Any]] = None,
        column_semantics: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> PipelineResult:
        """
        통합 인사이트 파이프라인 실행

        Args:
            tables: 테이블 정보 {table_name: {"columns": [...], "row_count": int}}
            data: 테이블 데이터 {table_name: [row_dicts]}
            correlations: 상관관계 목록 (SimulationEngine용)
            column_semantics: 컬럼 의미론 정보

        Returns:
            PipelineResult: 통합 인사이트 결과
        """
        import time
        start_time = time.time()

        unified_insights: List[UnifiedInsight] = []
        phases_executed = []
        errors = []

        # === Phase 1: Algorithm (BusinessInsightsAnalyzer) ===
        algorithm_insights = []
        logger.info(f"[Pipeline] Starting run(). enable_llm={self.enable_llm}, llm_client={self.llm_client is not None}")
        try:
            algorithm_insights = self._run_algorithm_phase(tables, data, column_semantics)
            phases_executed.append(UnifiedInsightPhase.ALGORITHM.value)
            logger.info(f"[Pipeline] Phase 1 Algorithm: {len(algorithm_insights)} insights")
            logger.info(f"[Phase 1] Algorithm: {len(algorithm_insights)} insights generated")
        except Exception as e:
            logger.error(f"[Phase 1] Algorithm failed: {e}")
            errors.append(f"Algorithm phase failed: {str(e)}")

        # === Phase 2: Simulation (SimulationEngine) ===
        simulation_report = None
        if self.enable_simulation and correlations:
            try:
                simulation_report = self._run_simulation_phase(data, correlations)
                phases_executed.append(UnifiedInsightPhase.SIMULATION.value)
                logger.info(f"[Phase 2] Simulation: {simulation_report.total_experiments if simulation_report else 0} experiments")
            except Exception as e:
                logger.error(f"[Phase 2] Simulation failed: {e}")
                errors.append(f"Simulation phase failed: {str(e)}")

        # === Phase 3: LLM Reasoning ===
        logger.debug(f"[Pipeline] Phase 3 check: enable_llm={self.enable_llm}, llm_client={type(self.llm_client)}")
        if self.enable_llm and self.llm_client:
            try:
                # Algorithm insights와 Simulation 결과를 통합하여 LLM에 전달
                unified_insights = await self._run_llm_reasoning_phase(
                    algorithm_insights=algorithm_insights,
                    simulation_report=simulation_report,
                    tables=tables,
                    data=data,
                )
                phases_executed.append(UnifiedInsightPhase.LLM_REASONING.value)
                logger.info(f"[Phase 3] LLM Reasoning: {len(unified_insights)} unified insights")
            except Exception as e:
                logger.error(f"[Phase 3] LLM Reasoning failed: {e}", exc_info=True)
                errors.append(f"LLM Reasoning phase failed: {str(e)}")
                # LLM 실패 시 알고리즘 결과만으로 인사이트 생성
                unified_insights = self._create_fallback_insights(algorithm_insights, simulation_report)
        else:
            # LLM 비활성화 시 알고리즘/시뮬레이션 결과로 인사이트 생성
            unified_insights = self._create_fallback_insights(algorithm_insights, simulation_report)

        # SharedContext 업데이트
        if self.context:
            self.context.unified_insights_pipeline = [i.to_dict() for i in unified_insights]

        execution_time_ms = (time.time() - start_time) * 1000

        return PipelineResult(
            unified_insights=unified_insights,
            algorithm_insights_count=len(algorithm_insights),
            simulation_experiments_count=simulation_report.total_experiments if simulation_report else 0,
            llm_enriched_count=len([i for i in unified_insights if i.llm_interpretation]),
            execution_time_ms=execution_time_ms,
            phases_executed=phases_executed,
            errors=errors,
        )

    def _run_algorithm_phase(
        self,
        tables: Dict[str, Dict[str, Any]],
        data: Dict[str, List[Dict[str, Any]]],
        column_semantics: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> List[Any]:
        """
        Phase 1: Algorithm-based 인사이트 생성

        BusinessInsightsAnalyzer 사용
        """
        from .business_insights import BusinessInsightsAnalyzer

        # DomainContext 가져오기
        domain_context = None
        if self.context:
            domain_context = getattr(self.context, 'domain_context', None)

        self._business_analyzer = BusinessInsightsAnalyzer(
            domain_context=domain_context,
            enable_ml_anomaly=True,
            enable_forecasting=True,
            llm_client=self.llm_client,
        )

        # v19.4: pipeline_context 전달 (ATE, stratified analysis 등)
        if self._pipeline_context:
            self._business_analyzer._pipeline_context = self._pipeline_context

        return self._business_analyzer.analyze(
            tables=tables,
            data=data,
            column_semantics=column_semantics,
            pipeline_context=self._pipeline_context if self._pipeline_context else None,
        )

    def _run_simulation_phase(
        self,
        data: Dict[str, List[Dict[str, Any]]],
        correlations: List[Any],
    ) -> Any:
        """
        Phase 2: Simulation-based 실험

        SimulationEngine 사용
        """
        from .simulation_engine import SimulationEngine
        import pandas as pd

        # 데이터를 DataFrame으로 변환 (SimulationEngine용)
        tables_data = {}
        for table_name, rows in data.items():
            if rows:
                try:
                    tables_data[table_name] = pd.DataFrame(rows)
                except Exception:
                    pass

        if not tables_data:
            return None

        self._simulation_engine = SimulationEngine(tables_data=tables_data)

        return self._simulation_engine.run_all_experiments(
            correlations=correlations,
            max_experiments_per_correlation=3,
        )

    async def _run_llm_reasoning_phase(
        self,
        algorithm_insights: List[Any],
        simulation_report: Any,
        tables: Dict[str, Dict[str, Any]],
        data: Dict[str, List[Dict[str, Any]]],
    ) -> List[UnifiedInsight]:
        """
        Phase 3: Multi-Agent LLM Reasoning (v21.0)

        3개 전문 에이전트 + 합성 단계:
        1. DATA_VALIDATOR (FAST/gpt-5.1): 수치 검증, 데이터 그라운딩
        2. BUSINESS_INTERPRETER (BALANCED/claude): 인과 추론, 비즈니스 해석
        3. PATTERN_DISCOVERER (CREATIVE/gemini): 교차 패턴, 신규 연결
        4. SYNTHESIZER (BALANCED/claude): 3개 결과 합성 → UnifiedInsight[]
        """
        unified_insights = []
        logger.info(f"[Pipeline] _run_llm_reasoning_phase entered. algorithm_insights={len(algorithm_insights)}")

        # === 절삭 없는 컨텍스트 구성 ===
        context_summary = self._build_context_summary(tables, data)
        algorithm_summary = self._summarize_algorithm_insights_full(algorithm_insights)
        simulation_summary = self._summarize_simulation_report(simulation_report) if simulation_report else "No simulation data available."
        causal_summary = self._summarize_causal_analysis()

        # === v21.0: 3개 에이전트 병렬 실행 ===
        logger.info("[Pipeline] Running 3 agents in parallel")
        logger.info(f"[v21.0] Running 3 insight agents in parallel (FAST/BALANCED/CREATIVE)")

        try:
            results = await asyncio.gather(
                self._run_data_validator(context_summary, algorithm_summary, simulation_summary),
                self._run_business_interpreter(context_summary, algorithm_summary, simulation_summary, causal_summary),
                self._run_pattern_discoverer(context_summary, algorithm_summary, simulation_summary, causal_summary),
                return_exceptions=True,
            )

            validation_result = results[0] if not isinstance(results[0], Exception) else None
            interpretation_result = results[1] if not isinstance(results[1], Exception) else None
            pattern_result = results[2] if not isinstance(results[2], Exception) else None

            # 에이전트 실패 로깅
            agent_names = ["DATA_VALIDATOR", "BUSINESS_INTERPRETER", "PATTERN_DISCOVERER"]
            for i, r in enumerate(results):
                if isinstance(r, Exception):
                    logger.warning(f"[Pipeline] {agent_names[i]} failed: {r}", exc_info=r)
                else:
                    logger.debug(f"[Pipeline] {agent_names[i]} OK: type={type(r).__name__}, keys={list(r.keys()) if isinstance(r, dict) else 'N/A'}")

            # 전체 실패 시 폴백
            if all(r is None for r in [validation_result, interpretation_result, pattern_result]):
                logger.warning("[Pipeline] All agents failed, using fallback")
                return self._create_fallback_insights(algorithm_insights, simulation_report)

        except Exception as e:
            logger.warning(f"[Pipeline] Agent orchestration failed: {e}", exc_info=True)
            return self._create_fallback_insights(algorithm_insights, simulation_report)

        # === v21.0: 합성 단계 (BALANCED/claude) ===
        logger.info(f"[Pipeline] Starting synthesis. val={validation_result is not None}, interp={interpretation_result is not None}, pat={pattern_result is not None}")
        try:
            llm_insights = await self._synthesize_insights(
                validation_result, interpretation_result, pattern_result,
                context_summary, causal_summary,
            )
            logger.debug(f"[Pipeline] Synthesis result: type={type(llm_insights).__name__}, len={len(llm_insights) if isinstance(llm_insights, list) else 'N/A'}")
        except Exception as e:
            logger.warning(f"[Pipeline] Synthesis failed: {e}", exc_info=True)
            # 합성 실패 시 interpreter 결과 직접 사용
            if interpretation_result and isinstance(interpretation_result, list):
                llm_insights = interpretation_result
            else:
                return self._create_fallback_insights(algorithm_insights, simulation_report)

        if not isinstance(llm_insights, list):
            llm_insights = [llm_insights] if llm_insights else []

        # === LLM 결과를 UnifiedInsight로 변환 (기존 후처리 유지) ===
        logger.info(f"[Pipeline] Converting {len(llm_insights)} LLM insights to UnifiedInsight")
        for i, llm_insight in enumerate(llm_insights):
            if not isinstance(llm_insight, dict):
                continue

            # 관련 알고리즘 인사이트 찾기
            related_algorithm = self._find_related_algorithm_insight(
                llm_insight, algorithm_insights
            )

            # 관련 시뮬레이션 실험 찾기
            related_simulation = self._find_related_simulation(
                llm_insight, simulation_report
            )

            # v19.4c: algorithm_findings가 비어있으면 pipeline_context causal data로 대체
            if not related_algorithm and self._pipeline_context:
                ci = self._pipeline_context.get("causal_insights", {})
                if isinstance(ci, dict) and ci.get("impact_analysis", {}).get("treatment_effects"):
                    title_lower = llm_insight.get("title", "").lower()
                    hyp_lower = llm_insight.get("hypothesis", "").lower()
                    search_text = f"{title_lower} {hyp_lower}"
                    relevant_effects = []
                    for te in ci["impact_analysis"]["treatment_effects"]:
                        treatment = str(te.get("treatment", "")).lower().replace("_", " ")
                        outcome = str(te.get("outcome", "")).lower()
                        if any(w in search_text for w in treatment.split() if len(w) > 3) or \
                           any(w in search_text for w in outcome.split() if len(w) > 3):
                            relevant_effects.append(te)
                    if not relevant_effects:
                        relevant_effects = ci["impact_analysis"]["treatment_effects"][:3]
                    related_algorithm = {
                        "source": "CausalImpactAnalysis",
                        "treatment_effects": relevant_effects[:5],
                        "causal_relationships": ci.get("causal_relationships", [])[:3],
                        "confidence": 0.8,
                    }

            # v19.4c: simulation_evidence가 비어있으면 시뮬레이션 리포트 요약으로 대체
            if not related_simulation and simulation_report:
                sim_summary = {}
                if hasattr(simulation_report, 'total_experiments'):
                    sim_summary["total_experiments"] = simulation_report.total_experiments
                if hasattr(simulation_report, 'summary'):
                    sim_summary["summary"] = str(simulation_report.summary)[:200]
                if sim_summary:
                    related_simulation = sim_summary

            unified = UnifiedInsight(
                insight_id=f"UI-{uuid.uuid4().hex[:8]}",
                title=llm_insight.get("title", f"Insight {i+1}"),
                summary=llm_insight.get("summary", ""),
                hypothesis=llm_insight.get("hypothesis", ""),
                evidence=llm_insight.get("evidence", []),
                validation_summary=llm_insight.get("validation_summary", ""),
                prescriptive_actions=llm_insight.get("prescriptive_actions", []),
                algorithm_findings=related_algorithm,
                simulation_evidence=related_simulation,
                llm_interpretation=llm_insight.get("validation_summary", llm_insight.get("llm_interpretation", "")),
                source_table=llm_insight.get("source_table", ""),
                source_column=llm_insight.get("source_column", ""),
                metric_value=llm_insight.get("metric_value"),
                benchmark=llm_insight.get("benchmark"),
                deviation=llm_insight.get("deviation"),
                business_impact=llm_insight.get("business_impact", ""),
                recommendations=llm_insight.get("recommendations", []),
                action_items=[a.get("action", "") for a in llm_insight.get("prescriptive_actions", []) if isinstance(a, dict)],
                confidence=self._calculate_combined_confidence(
                    related_algorithm, related_simulation, llm_insight
                ),
                confidence_breakdown=self._fix_confidence_breakdown(
                    llm_insight.get("confidence_breakdown", {}),
                    related_algorithm, related_simulation, llm_insight,
                ),
                insight_type=llm_insight.get("insight_type", "operational"),
                severity=llm_insight.get("severity", "medium"),
                domain_tags=llm_insight.get("domain_tags", []),
                phases_completed=[
                    UnifiedInsightPhase.ALGORITHM.value,
                    UnifiedInsightPhase.SIMULATION.value if simulation_report else None,
                    UnifiedInsightPhase.LLM_REASONING.value,
                ],
            )
            unified.phases_completed = [p for p in unified.phases_completed if p]

            # v19.1: Evidence Grounding
            unified.evidence = self._ground_evidence(unified.evidence, data, tables)

            # v19.5: 소표본 경고
            unified.evidence = self._add_sample_size_warnings(
                unified.evidence,
                insight_context={"title": llm_insight.get("title", ""), "hypothesis": llm_insight.get("hypothesis", "")},
            )

            # v19.4: prescriptive_actions 후처리
            unified.prescriptive_actions = self._sanitize_prescriptive_actions(
                unified.prescriptive_actions, data, tables,
                insight_context={"title": llm_insight.get("title", ""), "hypothesis": llm_insight.get("hypothesis", "")},
            )

            unified_insights.append(unified)

        # v21.0: 커버리지 검증
        logger.info(f"[Pipeline] Created {len(unified_insights)} UnifiedInsight objects")
        self._enforce_coverage(unified_insights)

        return unified_insights

    # =========================================================================
    # v21.0: Multi-Agent Methods
    # =========================================================================

    async def _run_data_validator(
        self, context: str, algo: str, sim: str,
    ) -> Dict[str, Any]:
        """
        Agent 1: DATA_VALIDATOR (FAST/gpt-5.1)
        수치 검증, 데이터 교차 참조, 불일치 플래그
        """
        model = get_service_model("data_validator")
        prompt = f"""You are a data validation specialist. Your job is to verify numerical consistency.

## Data Context
{context}

## Algorithm Findings
{algo}

## Simulation Evidence
{sim}

## Task
1. Cross-reference ALL numbers in Algorithm Findings against Data Context statistics.
2. Flag any inconsistencies (e.g., claimed average doesn't match actual data).
3. Identify which findings have STRONG data support vs WEAK/UNVERIFIABLE claims.
4. List the TOP 5 most data-grounded findings with exact numbers.

## Output (JSON)
{{
  "validated_findings": [
    {{
      "finding_title": "...",
      "data_support": "strong|moderate|weak",
      "verified_numbers": {{"metric": "value from data"}},
      "issues": ["list of inconsistencies if any"]
    }}
  ],
  "top_grounded_findings": ["finding 1", "finding 2", ...],
  "data_quality_notes": ["note about data coverage or gaps"]
}}

Respond ONLY with valid JSON."""

        logger.info(f"[Pipeline] DATA_VALIDATOR calling {model}...")
        response = self.llm_client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )
        result_text = response.choices[0].message.content or ""
        logger.debug(f"[Pipeline] DATA_VALIDATOR response: {len(result_text)} chars, first200: {result_text[:200] if result_text else 'EMPTY'}")
        if not result_text.strip():
            logger.warning(f"[Pipeline] DATA_VALIDATOR empty response. finish_reason={response.choices[0].finish_reason}, model={response.model}")
        return self._parse_llm_json(result_text, default={})

    async def _run_business_interpreter(
        self, context: str, algo: str, sim: str, causal: str,
    ) -> List[Dict[str, Any]]:
        """
        Agent 2: BUSINESS_INTERPRETER (BALANCED/claude)
        인과 추론, 비즈니스 해석, 처방적 액션
        """
        model = get_service_model("business_interpreter")
        prompt = f"""You are a senior business intelligence analyst for the **{self.domain}** domain.

## Data Context
{context}

## Algorithm Findings
{algo}

## Causal Inference Results
{causal if causal else "No causal analysis available."}

## Simulation Evidence
{sim}

## Task
Synthesize findings into 3-5 Palantir-style prescriptive insights.

## Output Format (JSON Array)
Each insight MUST follow this structure:
```json
[
  {{
    "title": "Short, actionable executive title",
    "summary": "2-3 sentence executive summary grounded in model results",
    "hypothesis": "Testable business claim",
    "evidence": [
      {{
        "model": "CausalImpact|SimulationEngine|AnomalyDetection|CorrelationAnalysis|TimeSeries",
        "finding": "Specific result with numbers",
        "detail": "Detailed explanation with values from analysis data",
        "metrics": {{"key_metric": "value", "p_value": 0.XX, "sample_size": "N", "effect_size": 0.XX}}
      }}
    ],
    "validation_summary": "How multiple evidence sources confirm the hypothesis",
    "prescriptive_actions": [
      {{
        "action": "Specific actionable step",
        "expected_impact": "Quantified outcome from data (ATE/baseline×100)",
        "estimated_savings": "Computed from data or '비용 데이터 부재 — 정량화 불가'",
        "priority": "immediate|short_term|long_term"
      }}
    ],
    "insight_type": "kpi|risk|anomaly|trend|operational|prediction",
    "severity": "critical|high|medium|low",
    "source_table": "primary table",
    "source_column": "key columns",
    "business_impact": "Quantified overall impact",
    "confidence_breakdown": {{"algorithm": 0.XX, "simulation": 0.XX, "llm": 0.XX}},
    "domain_tags": ["tag1", "tag2"]
  }}
]
```

ANTI-HALLUCINATION RULES:
1. evidence[].metrics MUST ONLY use numbers from the provided Data Context, Causal Results, or Algorithm Findings.
2. DO NOT invent percentages or statistics. If unavailable, write "inferred" and set metrics to {{}}.
3. ONLY reference columns listed under [AVAILABLE COLUMNS].
4. For estimated_savings: compute from data or write "비용 데이터 부재 — 정량화 불가".
5. For expected_impact: derive from data (ATE/baseline×100) or state direction only ("정량화 불가 — 추가 데이터 필요").
6. For causal claims: ALWAYS use Adjusted ATE (NOT Naive ATE). Mention Simpson's paradox if detected.
7. evidence[] needs min 2 models per insight. prescriptive_actions[] needs min 2 actions per insight.

Generate 3-5 insights. Prioritize by business impact.
Respond ONLY with valid JSON array."""

        logger.info(f"[Pipeline] BUSINESS_INTERPRETER calling {model}...")
        response = self.llm_client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )
        result_text = response.choices[0].message.content
        logger.debug(f"[Pipeline] BUSINESS_INTERPRETER response: {len(result_text)} chars, first200: {result_text[:200] if result_text else 'EMPTY'}")
        parsed = self._parse_llm_json(result_text, default=[])
        logger.debug(f"[Pipeline] BUSINESS_INTERPRETER parsed: type={type(parsed).__name__}, len={len(parsed) if isinstance(parsed, list) else 'N/A'}")
        return parsed if isinstance(parsed, list) else [parsed] if parsed else []

    async def _run_pattern_discoverer(
        self, context: str, algo: str, sim: str, causal: str,
    ) -> Dict[str, Any]:
        """
        Agent 3: PATTERN_DISCOVERER (CREATIVE/gemini)
        교차 패턴, 신규 연결, 숨겨진 관계
        """
        model = get_service_model("pattern_discoverer")
        prompt = f"""You are a creative data scientist specializing in discovering hidden patterns in the **{self.domain}** domain.

## Data Context
{context}

## Algorithm Findings
{algo}

## Causal Analysis
{causal if causal else "No causal analysis available."}

## Simulation Evidence
{sim}

## Task
Look BEYOND the obvious findings. Discover:
1. **Cross-cutting patterns**: How do different findings connect to each other? (e.g., A causes B, B correlates with C → what does A→C imply?)
2. **Hidden subgroup dynamics**: Are there segments where the overall trend reverses? (Simpson's paradox, interaction effects)
3. **Emerging risks/opportunities**: What trends could become critical if they continue?
4. **Novel hypotheses**: What unexpected business questions does the data raise?

IMPORTANT: Every pattern MUST reference specific numbers or columns from the provided data. Do NOT speculate without data support.

## Output (JSON)
{{
  "cross_patterns": [
    {{
      "pattern": "Description of the cross-cutting pattern",
      "data_support": "Specific numbers/columns that support this",
      "business_implication": "What this means for the business",
      "severity": "critical|high|medium|low"
    }}
  ],
  "novel_hypotheses": [
    {{
      "hypothesis": "Testable claim that warrants further investigation",
      "supporting_evidence": "What data hints at this",
      "recommended_analysis": "What additional analysis could confirm/deny this"
    }}
  ],
  "emerging_trends": [
    {{
      "trend": "Emerging pattern description",
      "direction": "increasing|decreasing|shifting",
      "urgency": "immediate|monitor|long_term"
    }}
  ]
}}

Respond ONLY with valid JSON."""

        logger.info(f"[Pipeline] PATTERN_DISCOVERER calling {model}...")
        response = self.llm_client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5,
        )
        result_text = response.choices[0].message.content
        logger.debug(f"[Pipeline] PATTERN_DISCOVERER response: {len(result_text)} chars, first200: {result_text[:200] if result_text else 'EMPTY'}")
        logger.info(f"[v21.0 PATTERN_DISCOVERER] Model={model}, response_len={len(result_text)}")
        return self._parse_llm_json(result_text, default={})

    async def _synthesize_insights(
        self,
        validation: Optional[Dict[str, Any]],
        interpretation: Optional[List[Dict[str, Any]]],
        patterns: Optional[Dict[str, Any]],
        context_summary: str,
        causal_summary: str,
    ) -> List[Dict[str, Any]]:
        """
        Synthesis: 3개 에이전트 출력을 합성하여 최종 UnifiedInsight[] 생성 (BALANCED/claude)
        """
        model = get_service_model("insight_synthesizer")

        # 에이전트 결과 요약
        validation_text = json.dumps(validation, ensure_ascii=False, indent=1)[:2000] if validation else "Validation agent did not produce results."
        interpretation_text = json.dumps(interpretation, ensure_ascii=False, indent=1)[:4000] if interpretation else "Interpretation agent did not produce results."
        patterns_text = json.dumps(patterns, ensure_ascii=False, indent=1)[:2000] if patterns else "Pattern discovery agent did not produce results."

        prompt = f"""You are the final synthesis analyst. Three specialized agents have analyzed the same data independently.
Your job: combine their outputs into a unified, high-quality set of Palantir-style prescriptive insights.

## Domain: {self.domain}

## Agent 1 — DATA VALIDATOR (verified numbers):
{validation_text}

## Agent 2 — BUSINESS INTERPRETER (causal insights):
{interpretation_text}

## Agent 3 — PATTERN DISCOVERER (hidden patterns):
{patterns_text}

## Causal Analysis Summary:
{causal_summary if causal_summary else "No causal analysis."}

## Synthesis Rules:
1. START from Agent 2's insights as the base structure.
2. ENRICH with Agent 1's validated numbers (replace any unverified numbers).
3. ADD Agent 3's cross-patterns as additional insights if they are data-supported and non-overlapping.
4. REMOVE or flag any claims that Agent 1 marked as "weak" data support.
5. Each final insight MUST have: title, summary, hypothesis, evidence[] (min 2), validation_summary, prescriptive_actions[] (min 2).
6. Generate 3-5 final insights total. Prioritize by business impact.
7. For expected_impact: ONLY use numbers derivable from data. If not possible, state direction only.
8. For estimated_savings: compute from data or write "비용 데이터 부재 — 정량화 불가".

## Output Format (JSON Array)
Same schema as Agent 2's output:
[{{"title", "summary", "hypothesis", "evidence": [...], "validation_summary", "prescriptive_actions": [...], "insight_type", "severity", "source_table", "source_column", "business_impact", "confidence_breakdown": {{"algorithm", "simulation", "llm"}}, "domain_tags": [...]}}]

Respond ONLY with valid JSON array."""

        logger.info(f"[Pipeline] SYNTHESIZER calling {model}...")
        response = self.llm_client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )
        result_text = response.choices[0].message.content or ""
        logger.debug(f"[Pipeline] SYNTHESIZER response: {len(result_text)} chars, first300: {result_text[:300] if result_text else 'EMPTY'}")
        parsed = self._parse_llm_json(result_text, default=[])
        logger.debug(f"[Pipeline] SYNTHESIZER parsed: type={type(parsed).__name__}, len={len(parsed) if isinstance(parsed, list) else 'N/A'}")
        return parsed if isinstance(parsed, list) else [parsed] if parsed else []

    def _enforce_coverage(self, insights: List[UnifiedInsight]) -> None:
        """
        v21.0: 커버리지 검증 — top treatment effects가 인사이트에 포함되었는지 확인
        """
        top_treatments = self._get_top_treatments()
        if not top_treatments:
            return

        covered = set()
        for insight in insights:
            search_text = f"{insight.title} {insight.hypothesis}".lower()
            for t in top_treatments:
                t_words = t.lower().replace("_", " ").split()
                if any(w in search_text for w in t_words if len(w) > 3):
                    covered.add(t)

        uncovered = set(top_treatments) - covered
        if uncovered:
            logger.warning(f"[v21.0 Coverage] Uncovered top treatments: {uncovered}")
            # 커버리지 부족 시 경고만 (향후: 추가 LLM 호출로 보충 가능)

    def _get_top_treatments(self) -> List[str]:
        """pipeline_context에서 p-value 기준 top 3 treatment variables 추출"""
        if not self._pipeline_context:
            return []
        ci = self._pipeline_context.get("causal_insights", {})
        if not isinstance(ci, dict):
            return []
        effects = ci.get("impact_analysis", {}).get("treatment_effects", [])
        if not effects:
            return []
        # p-value 기준 정렬 (낮을수록 유의)
        sorted_effects = sorted(effects, key=lambda e: e.get("p_value", 1.0))
        return [str(e.get("treatment", "")) for e in sorted_effects[:3]]

    def _ground_evidence(
        self,
        evidence_list: List[Dict[str, Any]],
        data: Dict[str, List[Dict[str, Any]]],
        tables: Dict[str, Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        v19.4: Evidence Grounding Layer
        LLM 생성 evidence의 수치를 실제 데이터와 교차 검증.
        - 존재하지 않는 컬럼 참조 → UNVERIFIED 마킹
        - 수치가 실제와 20% 이상 차이 → 실제값 첨부
        - v19.4: 모델 인용 검증 — 실행되지 않은 모델 참조 시 UNVERIFIED 마킹
        """
        # 실제 사용 가능한 컬럼 수집
        available_columns = set()
        for tinfo in tables.values():
            for col in tinfo.get("columns", []):
                cname = col.get("name", str(col)) if isinstance(col, dict) else str(col)
                available_columns.add(cname.lower())

        # 실제 데이터에서 컬럼별 기초 통계 사전 계산
        col_stats: Dict[str, Dict[str, float]] = {}
        for tname, rows in data.items():
            if not rows:
                continue
            for col_key in available_columns:
                vals = []
                for r in rows:
                    v = r.get(col_key)
                    if v is None:
                        # 대소문자 무시 매칭
                        for rk in r:
                            if rk.lower() == col_key:
                                v = r[rk]
                                break
                    if v is not None:
                        try:
                            vals.append(float(v))
                        except (ValueError, TypeError):
                            pass
                if vals:
                    import numpy as _np
                    col_stats[col_key] = {
                        "mean": float(_np.mean(vals)),
                        "min": float(_np.min(vals)),
                        "max": float(_np.max(vals)),
                        "std": float(_np.std(vals)),
                        "count": len(vals),
                    }

        # v19.4: 실행된 모델 목록 구성 (실제 결과가 있는 모델만)
        executed_models = set()
        if self._pipeline_context:
            ci = self._pipeline_context.get("causal_insights", {})
            if isinstance(ci, dict) and ci.get("impact_analysis", {}).get("treatment_effects"):
                executed_models.add("causalimpact")
            if ci.get("granger_causality"):
                executed_models.add("grangercausality")
        if self._simulation_engine:
            executed_models.add("simulationengine")
        # Algorithm phase always runs
        executed_models.add("anomalydetection")
        executed_models.add("correlationanalysis")
        executed_models.add("timeseries")

        grounded = []
        for ev in evidence_list:
            if not isinstance(ev, dict):
                grounded.append(ev)
                continue

            ev = dict(ev)  # shallow copy
            metrics = ev.get("metrics", {})
            finding = ev.get("finding", "")

            # v19.4: 모델 인용 검증
            model_name = ev.get("model", "").lower().replace(" ", "").replace("_", "")
            model_verified = any(m in model_name for m in executed_models) if model_name else True
            if not model_verified:
                ev["grounding_status"] = "UNVERIFIED_MODEL"
                ev["grounding_note"] = f"Model '{ev.get('model')}' did not produce results in this run"
                grounded.append(ev)
                continue

            # 메트릭의 키 이름에서 컬럼 참조 추출 후 검증
            has_unverifiable = False
            for mk, mv in list(metrics.items()):
                # 메트릭 키에서 컬럼명 추출 시도 (e.g., "salary_anomalies" → "salary")
                metric_col = mk.lower().replace("_anomalies", "").replace("_percentage", "").replace("_decrease", "").replace("_correlation", "")
                # 실제 컬럼 존재 확인
                if metric_col not in available_columns and metric_col not in col_stats:
                    # 부분 매칭 시도
                    partial_match = any(metric_col in ac for ac in available_columns)
                    if not partial_match:
                        has_unverifiable = True

            if has_unverifiable:
                ev["grounding_status"] = "PARTIAL"
            else:
                ev["grounding_status"] = "GROUNDED"

            # v19.5: evidence metrics 보강 — pipeline_context에서 p_value, sample_size, CI 주입
            if self._pipeline_context and "causal" in model_name:
                ev = self._enrich_evidence_metrics(ev)

            grounded.append(ev)

        return grounded

    def _enrich_evidence_metrics(self, ev: Dict[str, Any]) -> Dict[str, Any]:
        """
        v19.5: CausalImpact evidence의 metrics에 p_value, sample_size, CI 보강.
        pipeline_context.causal_insights.treatment_effects에서 매칭.
        """
        ci = self._pipeline_context.get("causal_insights", {})
        if not isinstance(ci, dict):
            return ev

        effects = ci.get("impact_analysis", {}).get("treatment_effects", [])
        if not effects:
            return ev

        finding = ev.get("finding", "").lower()
        metrics = ev.get("metrics", {})
        if not isinstance(metrics, dict):
            metrics = {}

        # finding 텍스트에서 매칭되는 treatment_effect 찾기
        # v19.5a: ATE 값 매칭도 활용하여 정확한 treatment_effect 선택
        import re as _re
        finding_numbers = set()
        for num_str in _re.findall(r'(?<![a-zA-Z])(\d+\.\d+)(?![a-zA-Z])', finding):
            try:
                finding_numbers.add(round(float(num_str), 1))
            except ValueError:
                pass

        best_te = None
        best_score = 0
        for te in effects:
            treatment = str(te.get("treatment", "")).lower().replace("_", " ")
            outcome = str(te.get("outcome", "")).lower()
            score = 0
            for w in treatment.split():
                if len(w) > 3 and w in finding:
                    score += 1
            for w in outcome.split():
                if len(w) > 3 and w in finding:
                    score += 1
            # ATE 값 매칭 보너스 (finding에 언급된 숫자와 일치하면 +5)
            ate_val = round(abs(te.get("estimate", 0)), 1)
            if ate_val and ate_val in finding_numbers:
                score += 5
            if score > best_score:
                best_score = score
                best_te = te

        if not best_te and effects:
            best_te = effects[0]

        if best_te:
            # v19.5a: 기존 ES와 best_te.estimate가 일치하는지 확인
            # 불일치하면 다른 treatment_effect의 CI/p를 주입하면 안 됨
            existing_es_val = metrics.get("effect_size", "")
            best_te_ate = round(best_te.get("estimate", 0), 1)
            es_matches_te = True
            if isinstance(existing_es_val, (int, float)) and existing_es_val != "" and best_te_ate:
                es_matches_te = abs(round(existing_es_val, 1) - abs(best_te_ate)) < 1.0

            # v19.5a: "inferred" 값도 미산출 취급
            existing_p = metrics.get("p_value", "")
            p_is_missing = existing_p == "" or str(existing_p).lower() in ("inferred", "n/a")
            if p_is_missing and best_te.get("p_value") is not None and es_matches_te:
                metrics["p_value"] = round(best_te["p_value"], 4)
            existing_n = metrics.get("sample_size", "")
            n_is_missing = existing_n == "" or str(existing_n).lower() in ("inferred", "n/a")
            if n_is_missing:
                strat = best_te.get("stratified_analysis", {})
                strata = strat.get("strata", [])
                total_n = sum(s.get("treated_n", 0) + s.get("control_n", 0) for s in strata)
                if total_n > 0:
                    metrics["sample_size"] = total_n
                elif best_te.get("group_comparison"):
                    gc = best_te["group_comparison"]
                    n1 = gc.get("group_1", {}).get("count", 0)
                    n0 = gc.get("group_0", {}).get("count", 0)
                    if n1 + n0 > 0:
                        metrics["sample_size"] = n1 + n0
            # v19.5a: "inferred"도 미산출 취급하여 실제 CI로 대체 (ES 일치 시에만)
            existing_ci = metrics.get("confidence_interval", "")
            ci_is_missing = not existing_ci or str(existing_ci).lower() in ("inferred", "n/a", "unknown")
            if ci_is_missing and best_te.get("std_error") and es_matches_te:
                ate = best_te.get("estimate", 0)
                se = best_te["std_error"]
                ci_lower = round(ate - 1.96 * se, 2)
                ci_upper = round(ate + 1.96 * se, 2)
                metrics["confidence_interval"] = f"{ci_lower} to {ci_upper}"
            existing_es = metrics.get("effect_size", "")
            es_is_missing = existing_es == "" or str(existing_es).lower() in ("inferred", "n/a")
            if es_is_missing and best_te.get("estimate") is not None:
                metrics["effect_size"] = round(best_te["estimate"], 4)

        ev["metrics"] = metrics

        # detail 필드가 비어있으면 보강
        if not ev.get("detail") and best_te:
            treatment = best_te.get("treatment", "?")
            outcome = best_te.get("outcome", "?")
            ate = best_te.get("estimate", 0)
            pval = best_te.get("p_value", "?")
            ev["detail"] = (
                f"{treatment}→{outcome}: Adjusted ATE={ate:+.1f}, p={pval}, "
                f"confounder={best_te.get('stratified_analysis', {}).get('confounder', 'N/A')}"
            )

        return ev

    def _add_sample_size_warnings(
        self,
        evidence_list: List[Dict[str, Any]],
        insight_context: Optional[Dict[str, str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        v19.5: pipeline_context의 stratified_analysis에서 control_n < 5인 계층을 감지하여
        evidence에 sample_size_warning 항목 추가.
        insight_context로 인사이트 관련 treatment/outcome만 필터링.
        """
        if not self._pipeline_context:
            return evidence_list

        ci = self._pipeline_context.get("causal_insights", {})
        if not isinstance(ci, dict):
            return evidence_list

        # 인사이트 컨텍스트에서 키워드 추출
        search_text = ""
        if insight_context:
            search_text = f"{insight_context.get('title', '')} {insight_context.get('hypothesis', '')}".lower()

        effects = ci.get("impact_analysis", {}).get("treatment_effects", [])

        # 관련도별 경고 분류: both_match > single_match
        both_match_warnings = []
        single_match_warnings = []

        for te in effects:
            treatment = str(te.get("treatment", "")).lower().replace("_", " ")
            outcome = str(te.get("outcome", "")).lower()

            # 인사이트와 관련된 treatment/outcome인지 확인
            if search_text:
                treatment_match = any(w in search_text for w in treatment.split() if len(w) > 3)
                outcome_match = any(w in search_text for w in outcome.split() if len(w) > 3)
                if not treatment_match and not outcome_match:
                    continue
                relevance = "both" if (treatment_match and outcome_match) else "single"
            else:
                relevance = "both"

            strat = te.get("stratified_analysis")
            if not strat:
                continue
            strata = strat.get("strata", [])
            for s in strata:
                control_n = s.get("control_n", 999)
                treated_n = s.get("treated_n", 999)
                if control_n < 5 or treated_n < 5:
                    stratum_name = s.get("stratum", "?")
                    treatment_raw = te.get("treatment", "?")
                    outcome_raw = te.get("outcome", "?")
                    small_group = "대조군" if control_n < 5 else "처리군"
                    small_n = control_n if control_n < 5 else treated_n
                    warning = {
                        "model": "SampleSizeValidator",
                        "finding": f"{treatment_raw}→{outcome_raw} 분석에서 {stratum_name} 계층 {small_group} n={small_n} — 통계적 불안정",
                        "detail": f"소표본 경고: {stratum_name} 계층의 {small_group} 표본 크기가 {small_n}으로 5 미만입니다. "
                                  f"이 계층의 ATE(={s.get('stratum_ate', '?')})는 통계적으로 불안정할 수 있습니다.",
                        "metrics": {
                            "stratum": stratum_name,
                            "control_n": control_n,
                            "treated_n": treated_n,
                            "stratum_ate": s.get("stratum_ate"),
                        },
                        "grounding_status": "WARNING",
                        "sample_size_warning": True,
                    }
                    if relevance == "both":
                        both_match_warnings.append(warning)
                    else:
                        single_match_warnings.append(warning)

            # v20: 제외된 strata 경고 (weighted ATE에 미포함된 계층)
            excluded = strat.get("excluded_strata", [])
            for ex in excluded:
                stratum_name = ex.get("stratum", "?")
                treatment_raw = te.get("treatment", "?")
                outcome_raw = te.get("outcome", "?")
                warning = {
                    "model": "SampleSizeValidator",
                    "finding": f"{treatment_raw}→{outcome_raw}: {stratum_name} 계층이 Adjusted ATE에서 제외됨 ({ex.get('reason', 'insufficient samples')})",
                    "detail": f"계층 {stratum_name}은 treated_n={ex.get('treated_n', 0)}, control_n={ex.get('control_n', 0)}으로 "
                              f"최소 요구(2)를 충족하지 못해 가중 ATE 계산에서 제외되었습니다.",
                    "metrics": {
                        "stratum": stratum_name,
                        "treated_n": ex.get("treated_n", 0),
                        "control_n": ex.get("control_n", 0),
                        "excluded": True,
                    },
                    "grounding_status": "WARNING",
                    "sample_size_warning": True,
                }
                if relevance == "both":
                    both_match_warnings.append(warning)
                else:
                    single_match_warnings.append(warning)

        # 최대 3개 경고: both_match 우선, 부족하면 single_match 추가
        MAX_WARNINGS = 3
        warnings = both_match_warnings[:MAX_WARNINGS]
        if len(warnings) < MAX_WARNINGS:
            warnings.extend(single_match_warnings[:MAX_WARNINGS - len(warnings)])

        if warnings:
            evidence_list = list(evidence_list) + warnings

        return evidence_list

    def _sanitize_prescriptive_actions(
        self,
        actions: List[Dict[str, Any]],
        data: Dict[str, List[Dict[str, Any]]],
        tables: Dict[str, Dict[str, Any]],
        insight_context: Optional[Dict[str, str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        v19.4: prescriptive_actions의 날조 퍼센트/금액을 후처리로 차단.
        - expected_impact에 데이터에서 도출 불가능한 퍼센트 → 방향만 표시
        - estimated_savings에 비용 데이터 없으면 → 정량화 불가
        """
        import re

        # v19.5: insight_context에서 핵심 outcome 키워드 추출
        primary_outcomes = set()
        if insight_context:
            ctx_text = f"{insight_context.get('title', '')} {insight_context.get('hypothesis', '')}".lower()
            # 일반적인 outcome 키워드 매칭
            outcome_keywords = ["math", "english", "science", "attendance", "study", "sleep",
                                "performance", "score", "grade", "revenue", "cost", "salary"]
            for kw in outcome_keywords:
                if kw in ctx_text:
                    primary_outcomes.add(kw)

        # 실제 ATE 값 수집 — treatment/outcome별로 인덱싱
        ate_by_treatment = {}  # key: treatment_lower, value: list of (outcome, ate)
        known_ate_values = set()
        if self._pipeline_context:
            ci = self._pipeline_context.get("causal_insights", {})
            if isinstance(ci, dict):
                for te in ci.get("impact_analysis", {}).get("treatment_effects", []):
                    ate = te.get("estimate", 0)
                    treatment = str(te.get("treatment", "")).lower()
                    outcome = str(te.get("outcome", ""))
                    if ate:
                        known_ate_values.add(round(abs(ate), 1))
                        if treatment:
                            ate_by_treatment.setdefault(treatment, []).append((outcome, round(ate, 1)))

        # 비용 관련 컬럼 존재 여부
        has_cost_data = False
        for tinfo in tables.values():
            for col in tinfo.get("columns", []):
                cname = (col.get("name", str(col)) if isinstance(col, dict) else str(col)).lower()
                if any(kw in cname for kw in ("cost", "salary", "price", "revenue", "budget", "expense")):
                    has_cost_data = True
                    break

        sanitized = []
        for action in actions:
            if not isinstance(action, dict):
                sanitized.append(action)
                continue

            action = dict(action)

            # expected_impact 검증
            impact = action.get("expected_impact", "")
            if impact:
                impact_str = str(impact)
                direction = "positive" if any(w in impact_str.lower() for w in ("increase", "improve", "enhance", "boost", "raise")) else "negative"

                # 규칙 1: ATE는 점수(point) 단위 — 퍼센트(%)는 항상 날조
                pct_match = re.findall(r'(\d+(?:\.\d+)?)\s*%', impact_str)
                if pct_match:
                    # 인사이트 action 텍스트에서 관련 treatment 찾기
                    action_text = str(action.get("action", "")).lower()
                    relevant_ates = []
                    for treatment_key, ate_pairs in ate_by_treatment.items():
                        # treatment 키워드가 action 텍스트에 포함되는지 확인
                        treatment_words = treatment_key.replace("_", " ").split()
                        if any(tw in action_text for tw in treatment_words if len(tw) > 3):
                            for outcome, ate_val in ate_pairs:
                                relevant_ates.append((outcome, ate_val))

                    if relevant_ates:
                        # v19.5: primary_outcomes가 있으면 해당 outcome만 하이라이트
                        if primary_outcomes:
                            primary_ates = [(o, v) for o, v in relevant_ates if any(pk in o.lower() for pk in primary_outcomes)]
                            secondary_ates = [(o, v) for o, v in relevant_ates if not any(pk in o.lower() for pk in primary_outcomes)]
                        else:
                            primary_ates = relevant_ates
                            secondary_ates = []

                        if primary_ates:
                            ate_desc = ", ".join(f"{out}: {'+' if v > 0 else ''}{v}점" for out, v in primary_ates[:3])
                            if secondary_ates:
                                sec_desc = ", ".join(f"{out}: {'+' if v > 0 else ''}{v}점" for out, v in secondary_ates[:2])
                                action["expected_impact"] = f"Adjusted ATE 기준: {ate_desc} (기타: {sec_desc})"
                            else:
                                action["expected_impact"] = f"Adjusted ATE 기준: {ate_desc}"
                        else:
                            ate_desc = ", ".join(f"{out}: {'+' if v > 0 else ''}{v}점" for out, v in relevant_ates[:5])
                            action["expected_impact"] = f"Adjusted ATE 기준: {ate_desc}"
                    elif known_ate_values:
                        action["expected_impact"] = f"Expected {direction} direction — ATE 기반 점수 변화 확인됨 (세부 매칭 필요)"
                    else:
                        action["expected_impact"] = f"Expected {direction} direction — 정확한 비율 미산출 (추가 데이터 필요)"
                else:
                    # 규칙 2: 숫자+points 형태도 ATE와 대조
                    pts_match = re.findall(r'(\d+(?:\.\d+)?)\s*(?:points?|pts?)', impact_str, re.IGNORECASE)
                    if pts_match:
                        for pv in pts_match:
                            pv_float = float(pv)
                            if not any(abs(pv_float - kv) < 0.5 for kv in known_ate_values):
                                # ATE와 불일치 → 관련 ATE로 대체
                                action_text = str(action.get("action", "")).lower()
                                relevant_ates = []
                                for treatment_key, ate_pairs in ate_by_treatment.items():
                                    treatment_words = treatment_key.replace("_", " ").split()
                                    if any(tw in action_text for tw in treatment_words if len(tw) > 3):
                                        relevant_ates.extend(ate_pairs[:5])
                                if relevant_ates:
                                    ate_desc = ", ".join(f"{out}: {'+' if v > 0 else ''}{v}점" for out, v in relevant_ates[:5])
                                    action["expected_impact"] = f"Adjusted ATE 기준: {ate_desc}"
                                else:
                                    action["expected_impact"] = f"Expected {direction} direction — 정확한 수치 미산출"
                                break

            # v20 규칙 3: 범용 ATE 화이트리스트 — 모든 수치를 known ATE와 대조
            # 규칙 1/2가 이미 치환한 경우 (Adjusted ATE / Expected 시작) 건너뜀
            current_impact = str(action.get("expected_impact", ""))
            if current_impact and not current_impact.startswith("Adjusted ATE") and not current_impact.startswith("Expected"):
                all_numbers = re.findall(r'[\d,]+(?:\.\d+)?', current_impact)
                unverified = []
                for ns in all_numbers:
                    try:
                        nv = float(ns.replace(",", ""))
                        if nv < 1.0:
                            continue  # p-value, 소수점 계수 무시
                        if not known_ate_values or not any(abs(nv - kv) < max(0.5, abs(kv) * 0.1) for kv in known_ate_values):
                            unverified.append(nv)
                    except ValueError:
                        continue

                if unverified:
                    # direction fallback (Rule 1/2 블록 밖에서 호출될 수 있으므로)
                    _dir = "positive" if any(w in current_impact.lower() for w in ("increase", "improve", "enhance", "boost", "raise")) else "negative"
                    action_text = str(action.get("action", "")).lower()
                    relevant_ates = []
                    for treatment_key, ate_pairs in ate_by_treatment.items():
                        treatment_words = treatment_key.replace("_", " ").split()
                        if any(tw in action_text for tw in treatment_words if len(tw) > 3):
                            relevant_ates.extend(ate_pairs[:5])
                    if relevant_ates:
                        ate_desc = ", ".join(f"{out}: {'+' if v > 0 else ''}{v}점" for out, v in relevant_ates[:5])
                        action["expected_impact"] = f"Adjusted ATE 기준: {ate_desc}"
                    elif known_ate_values:
                        action["expected_impact"] = f"Expected {_dir} direction — ATE 기반 점수 변화 확인됨 (세부 매칭 필요)"
                    else:
                        action["expected_impact"] = f"Expected {_dir} direction — 정확한 수치 미산출"

            # estimated_savings 검증
            savings = action.get("estimated_savings", "")
            if savings and "정량화 불가" not in str(savings) and not has_cost_data:
                action["estimated_savings"] = "비용 데이터 부재 — 정량화 불가"

            sanitized.append(action)

        return sanitized

    def _fix_confidence_breakdown(
        self,
        breakdown: Dict[str, float],
        related_algorithm: Dict[str, Any],
        related_simulation: Dict[str, Any],
        llm_insight: Dict[str, Any],
    ) -> Dict[str, float]:
        """
        v19.5: confidence_breakdown 보정.
        - LLM이 llm=0.0으로 설정하는 문제 수정 (floor=0.5)
        - 빈 breakdown이면 기본값 생성
        """
        if not breakdown or not isinstance(breakdown, dict):
            breakdown = {
                "algorithm": related_algorithm.get("confidence", 0.5) if related_algorithm else 0.0,
                "simulation": related_simulation.get("statistical_significance", 0.5) if related_simulation else 0.0,
                "llm": llm_insight.get("confidence", 0.7),
            }

        # llm이 0.0이면 floor 적용
        llm_val = breakdown.get("llm", 0.0)
        if not isinstance(llm_val, (int, float)) or llm_val < 0.1:
            breakdown["llm"] = max(0.5, llm_insight.get("confidence", 0.7) if isinstance(llm_insight.get("confidence"), (int, float)) else 0.7)

        return breakdown

    def _create_fallback_insights(
        self,
        algorithm_insights: List[Any],
        simulation_report: Any,
    ) -> List[UnifiedInsight]:
        """
        LLM 없이 알고리즘/시뮬레이션 결과만으로 인사이트 생성
        """
        unified_insights = []

        for i, algo_insight in enumerate(algorithm_insights[:10]):  # 상위 10개
            algo_dict = algo_insight.to_dict() if hasattr(algo_insight, 'to_dict') else {}

            unified = UnifiedInsight(
                insight_id=f"UI-{uuid.uuid4().hex[:8]}",
                title=algo_dict.get("title", f"Finding {i+1}"),
                summary=algo_dict.get("description", ""),
                algorithm_findings=algo_dict,
                simulation_evidence={},
                llm_interpretation="",  # LLM 없음
                source_table=algo_dict.get("source_table", ""),
                source_column=algo_dict.get("source_column", ""),
                metric_value=algo_dict.get("metric_value"),
                benchmark=algo_dict.get("benchmark"),
                deviation=algo_dict.get("deviation"),
                business_impact=algo_dict.get("business_impact", ""),
                recommendations=algo_dict.get("recommendations", []),
                action_items=[],
                confidence=algo_dict.get("confidence", 0.5) if algo_dict.get("confidence") else 0.5,
                insight_type=algo_dict.get("insight_type", "operational"),
                severity=algo_dict.get("severity", "medium"),
                domain_tags=[],
                phases_completed=[UnifiedInsightPhase.ALGORITHM.value],
            )

            unified_insights.append(unified)

        return unified_insights

    def _build_context_summary(
        self,
        tables: Dict[str, Dict[str, Any]],
        data: Dict[str, List[Dict[str, Any]]],
    ) -> str:
        """LLM용 데이터 컨텍스트 요약 — v19.1: 실제 통계 수치 포함"""
        lines = [f"- Domain: {self.domain}"]
        lines.append(f"- Tables: {len(tables)}")

        for table_name, table_info in list(tables.items())[:5]:
            rows = data.get(table_name, [])
            row_count = len(rows)
            cols = table_info.get("columns", [])
            col_names = [c.get("name", str(c)) if isinstance(c, dict) else str(c) for c in cols]
            lines.append(f"  - {table_name}: {row_count} rows, columns: {', '.join(col_names[:12])}")

            if not rows:
                continue

            # v19.1: 실제 통계 수치 주입 (LLM hallucination 방지)
            numeric_cols = []
            categorical_cols = []
            for col_info in cols:
                cname = col_info.get("name", str(col_info)) if isinstance(col_info, dict) else str(col_info)
                dtype = col_info.get("dtype", "") if isinstance(col_info, dict) else ""
                if dtype in ("int64", "float64", "int32", "float32"):
                    numeric_cols.append(cname)
                elif dtype == "object":
                    categorical_cols.append(cname)

            # 수치형 컬럼 통계
            if numeric_cols:
                lines.append(f"    [Numeric Column Statistics]")
                for nc in numeric_cols[:10]:
                    vals = [r.get(nc) for r in rows if r.get(nc) is not None]
                    try:
                        nums = [float(v) for v in vals if isinstance(v, (int, float))]
                        if nums:
                            import numpy as _np
                            lines.append(
                                f"    - {nc}: avg={_np.mean(nums):.2f}, "
                                f"min={_np.min(nums):.1f}, max={_np.max(nums):.1f}, "
                                f"std={_np.std(nums):.2f} (n={len(nums)})"
                            )
                    except (ValueError, TypeError):
                        pass

            # 카테고리형 컬럼 분포
            if categorical_cols:
                lines.append(f"    [Categorical Column Distributions]")
                for cc in categorical_cols[:8]:
                    from collections import Counter
                    vals = [str(r.get(cc)) for r in rows if r.get(cc) is not None]
                    dist = Counter(vals).most_common(6)
                    dist_str = ", ".join(f"'{k}': {v}" for k, v in dist)
                    lines.append(f"    - {cc}: {dist_str}")

            # 그룹별 교차 통계 (핵심: 카테고리×수치 교차)
            if categorical_cols and numeric_cols:
                lines.append(f"    [Group-level Statistics (category × numeric)]")
                for cc in categorical_cols[:3]:
                    for nc in numeric_cols[:4]:
                        from collections import defaultdict
                        groups = defaultdict(list)
                        for r in rows:
                            cv = r.get(cc)
                            nv = r.get(nc)
                            if cv is not None and nv is not None:
                                try:
                                    groups[str(cv)].append(float(nv))
                                except (ValueError, TypeError):
                                    pass
                        if len(groups) >= 2:
                            import numpy as _np
                            parts = []
                            for gk in sorted(groups.keys()):
                                gv = groups[gk]
                                if gv:
                                    parts.append(f"{cc}={gk}: avg={_np.mean(gv):.1f} (n={len(gv)})")
                            if parts:
                                lines.append(f"    - {nc} by {cc}: {'; '.join(parts[:6])}")

            # AVAILABLE COLUMNS 명시 (존재하지 않는 컬럼 참조 방지)
            lines.append(f"    [AVAILABLE COLUMNS — ONLY these exist in the data]")
            lines.append(f"    {', '.join(col_names)}")

        # v22.0: OWL2/SHACL 온톨로지 추론 컨텍스트
        owl2_results = self._pipeline_context.get("owl2_reasoning_results", {})
        shacl_results = self._pipeline_context.get("shacl_validation_results", {})
        if owl2_results:
            lines.append("")
            lines.append("[Ontology Reasoning (OWL2)]")
            lines.append(f"  - Axioms discovered: {owl2_results.get('axioms_discovered', 0)}")
            inferred_classes = owl2_results.get("inferred_classes", [])
            if inferred_classes:
                lines.append(f"  - Inferred classes: {', '.join(str(c) for c in inferred_classes[:10])}")
            property_chains = owl2_results.get("property_chains", [])
            if property_chains:
                lines.append(f"  - Property chains: {len(property_chains)}")

        if shacl_results:
            lines.append("")
            lines.append("[Ontology Validation (SHACL)]")
            lines.append(f"  - Conformant: {shacl_results.get('is_conformant', True)}")
            violations_count = shacl_results.get("violations_count", 0)
            if violations_count > 0:
                lines.append(f"  - Violations: {violations_count}")
                for v in shacl_results.get("violations", [])[:3]:
                    lines.append(f"    - {v.get('path', 'unknown')}: {v.get('message', '')[:100]}")

        # v22.0: Calibration 히스토리
        calibration_history = self._pipeline_context.get("calibration_history", [])
        if calibration_history:
            lines.append("")
            lines.append(f"[Calibration History: {len(calibration_history)} phases calibrated]")

        # v22.0: Inferred triples (Knowledge Graph 추론 결과)
        inferred_triples = self._pipeline_context.get("inferred_triples", [])
        if inferred_triples:
            lines.append("")
            lines.append(f"[Knowledge Graph: {len(inferred_triples)} inferred triples]")
            for triple in inferred_triples[:5]:
                if isinstance(triple, dict):
                    lines.append(f"  - {triple.get('subject', '?')} → {triple.get('predicate', '?')} → {triple.get('object', '?')}")

        return "\n".join(lines)

    def _summarize_algorithm_insights(self, insights: List[Any]) -> str:
        """알고리즘 인사이트 요약 (레거시 — 절삭 포함)"""
        if not insights:
            return "No algorithm insights available."

        lines = [f"Total findings: {len(insights)}"]

        for insight in insights[:10]:
            if hasattr(insight, 'to_dict'):
                d = insight.to_dict()
            else:
                d = {}

            title = d.get("title", "Finding")
            severity = d.get("severity", "medium")
            desc = d.get("description", "")[:100]

            lines.append(f"- [{severity.upper()}] {title}: {desc}")

        return "\n".join(lines)

    def _summarize_algorithm_insights_full(self, insights: List[Any]) -> str:
        """
        v21.0: 알고리즘 인사이트 전체 요약 — 절삭 없음
        모든 인사이트의 전체 description, metric_value, source_column 포함.
        멀티에이전트 파이프라인에서 사용.
        """
        if not insights:
            return "No algorithm insights available."

        lines = [f"Total findings: {len(insights)}"]

        for insight in insights:
            if hasattr(insight, 'to_dict'):
                d = insight.to_dict()
            elif isinstance(insight, dict):
                d = insight
            else:
                d = {}

            title = d.get("title", "Finding")
            severity = d.get("severity", "medium")
            desc = d.get("description", "")  # NO truncation
            metric = d.get("metric_value", "")
            source_col = d.get("source_column", "")
            source_table = d.get("source_table", "")
            insight_type = d.get("insight_type", "")

            lines.append(f"- [{severity.upper()}] {title}")
            if desc:
                lines.append(f"  Description: {desc}")
            if metric:
                lines.append(f"  Metric Value: {metric}")
            if source_col:
                lines.append(f"  Source Column: {source_col}")
            if source_table:
                lines.append(f"  Source Table: {source_table}")
            if insight_type:
                lines.append(f"  Type: {insight_type}")

        return "\n".join(lines)

    def _summarize_causal_analysis(self) -> str:
        """
        v19.4: pipeline_context에서 인과 분석 결과를 LLM 프롬프트용으로 요약.
        ATE(Adjusted), Simpson's paradox 경고, stratified details 포함.
        """
        ctx = self._pipeline_context
        if not ctx:
            return ""

        parts = []
        causal = ctx.get("causal_insights", {})
        if not isinstance(causal, dict) or not causal:
            return ""

        impact = causal.get("impact_analysis", {})
        if not isinstance(impact, dict):
            return ""

        summary = impact.get("summary", {})
        effects = impact.get("treatment_effects", [])
        recs = impact.get("recommendations", [])

        if summary.get("data_driven") and effects:
            parts.append("### CausalImpact Analysis (Phase 2 — data-driven)")
            parts.append(f"- Sample size: {summary.get('sample_size', '?')}")
            parts.append(f"- Significant effects: {summary.get('significant_effects', 0)}/{summary.get('total_effects_analyzed', 0)}")
            parts.append("")

            for te in effects[:8]:
                strat = te.get("stratified_analysis")
                gc = te.get("group_comparison", {})
                g1 = gc.get("group_1", {})
                g0 = gc.get("group_0", {})
                naive = te.get("naive_ate", 0)
                adjusted = te.get("estimate", 0)
                pval = te.get("p_value", 1)
                sig = "YES" if te.get("significant") else "no"

                parts.append(f"**Effect: {te.get('type', '?')}**")
                parts.append(f"  Naive ATE = {naive:+.1f} ({g1.get('label','?')} mean={g1.get('mean',0):.1f} vs {g0.get('label','?')} mean={g0.get('mean',0):.1f})")
                parts.append(f"  Adjusted ATE = {adjusted:+.1f}, p={pval:.4f}, significant={sig}")

                if strat:
                    conf = strat.get("confounder", "?")
                    reduction = strat.get("confounding_reduction_pct", 0)
                    # v19.5a: 방향 반전 시 100% 초과 가능 → clamp + 주석
                    if reduction > 100:
                        parts.append(f"  Confounder: {conf} (effect direction reversed after adjustment — confounding dominates)")
                    else:
                        parts.append(f"  Confounder: {conf} (removed {reduction:.0f}% bias)")

                    # Simpson's paradox 감지
                    strata = strat.get("strata", [])
                    if strata:
                        signs = set()
                        for s in strata:
                            ate_val = s.get("stratum_ate", 0)
                            if ate_val > 0:
                                signs.add("positive")
                            elif ate_val < 0:
                                signs.add("negative")
                        if len(signs) > 1:
                            parts.append(f"  ⚠ SIMPSON'S PARADOX DETECTED: Overall effect is {'+' if naive > 0 else '-'} but subgroups show REVERSED direction")

                    for s in strata[:5]:
                        parts.append(
                            f"    └ {s.get('stratum', '?')}: "
                            f"treated={s.get('treated_mean', 0):.1f}(n={s.get('treated_n', 0)}), "
                            f"control={s.get('control_mean', 0):.1f}(n={s.get('control_n', 0)}), "
                            f"stratum ATE={s.get('stratum_ate', 0):+.1f}"
                        )

                    # v20: 제외된 strata 표시
                    excluded = strat.get("excluded_strata", [])
                    for ex in excluded:
                        parts.append(
                            f"    └ {ex.get('stratum', '?')}: EXCLUDED "
                            f"(treated_n={ex.get('treated_n', 0)}, control_n={ex.get('control_n', 0)} — below min_n=2)"
                        )

                parts.append("")

            parts.append("CRITICAL: Use Adjusted ATE (NOT Naive ATE) for all causal claims.")
            parts.append("Naive ATE includes confounding bias. If Naive and Adjusted differ significantly,")
            parts.append("the confounder creates Simpson's paradox — mention this explicitly in your insight.")
            parts.append("")

            # v20: Treatment Ranking by statistical significance
            ranked_effects = sorted(
                [te for te in effects if te.get("p_value") is not None],
                key=lambda te: te.get("p_value", 1.0)
            )
            if ranked_effects:
                parts.append("### TREATMENT RANKING (by statistical significance)")
                for rank_i, te in enumerate(ranked_effects[:5], 1):
                    se = te.get("std_error", 0)
                    ate = te.get("estimate", 0)
                    t_stat = f"{abs(ate / se):.2f}" if se and se > 0 else "N/A"
                    parts.append(
                        f"  {rank_i}. {te.get('type', '?')} | "
                        f"ATE={ate:+.2f}, p={te.get('p_value', 1):.4f}, |t|={t_stat}, "
                        f"significant={te.get('significant', False)}"
                    )
                parts.append("")

            # v20: 다양성 + 커버리지 요구사항
            unique_treatments = list(set(str(te.get("treatment", "")) for te in effects if te.get("treatment")))
            if len(unique_treatments) >= 2:
                parts.append("DIVERSITY + COVERAGE REQUIREMENT:")
                parts.append(f"  Available distinct treatments: {', '.join(unique_treatments)}")
                parts.append("  Generate insights covering AT LEAST 2 different treatment/confounder variables.")
                parts.append("  DO NOT create 3+ insights about the same treatment variable.")
                parts.append("  Include at least 1 insight about a confounder's direct effect.")
                if ranked_effects:
                    top2 = [te.get("type", "?") for te in ranked_effects[:2]]
                    parts.append(f"  COVERAGE: Your insights MUST include the top 2 most significant effects: {', '.join(top2)}")
                parts.append("")

            if recs:
                parts.append("Causal Recommendations:")
                for r in recs[:5]:
                    parts.append(f"  - {r}")

        # Counterfactual
        counterfactual = ctx.get("counterfactual_analysis", [])
        if isinstance(counterfactual, list) and counterfactual:
            parts.append("")
            parts.append("### Counterfactual Scenarios")
            for cf in counterfactual[:3]:
                if isinstance(cf, dict):
                    parts.append(f"  - {cf.get('scenario', '?')} → {cf.get('recommendation', '')}")

        # Causal relationships
        causal_rels = ctx.get("causal_relationships", [])
        if isinstance(causal_rels, list) and causal_rels:
            parts.append("")
            parts.append("### Causal Relationships (DAG)")
            for r in causal_rels[:5]:
                if isinstance(r, dict):
                    parts.append(f"  - {r.get('source', '?')} → {r.get('target', '?')} (conf={r.get('confidence', '?')})")

        return "\n".join(parts)

    def _summarize_simulation_report(self, report: Any) -> str:
        """
        시뮬레이션 리포트 요약 — v19.4: SimulationEngine.format_for_llm() 위임
        그룹 통계, effect size, p-value, threshold 등 상세 정보 포함
        """
        if not report:
            return "No simulation data."

        # v19.4: SimulationEngine의 format_for_llm()으로 위임 (rich output)
        if self._simulation_engine and hasattr(self._simulation_engine, 'format_for_llm'):
            try:
                return self._simulation_engine.format_for_llm(report)
            except Exception as e:
                logger.warning(f"format_for_llm failed, falling back: {e}")

        # Fallback: 기본 요약
        lines = [f"Total experiments: {report.total_experiments}"]

        if hasattr(report, 'significant_findings'):
            lines.append(f"Significant findings: {len(report.significant_findings)}")
            for finding in report.significant_findings[:5]:
                if isinstance(finding, dict):
                    lines.append(f"  - {finding.get('description', 'Finding')[:80]}")

        if hasattr(report, 'unexpected_findings'):
            lines.append(f"Unexpected findings: {len(report.unexpected_findings)}")

        return "\n".join(lines)

    def _find_related_algorithm_insight(
        self,
        llm_insight: Dict[str, Any],
        algorithm_insights: List[Any],
    ) -> Dict[str, Any]:
        """
        v20: 다중 팩터 가중 스코어링으로 알고리즘 인사이트 매칭.
        - source_column exact: +10, substring: +6
        - insight_type 일치: +5
        - severity 일치: +2
        - 인과 변수 키워드 매칭: +8
        - 의미 있는 단어 겹침 (stopwords 제외): +1/단어
        - 최소 임계값 3 미만이면 {} 반환 (잘못된 fallback 방지)
        """
        import re as _re

        source_column = llm_insight.get("source_column", "").lower()
        title = llm_insight.get("title", "").lower()
        hypothesis = llm_insight.get("hypothesis", "").lower()
        insight_type = llm_insight.get("insight_type", llm_insight.get("category", "")).lower()
        severity = llm_insight.get("severity", "").lower()
        search_text = f"{title} {hypothesis}"

        # 도메인 무관 stopwords — 거짓 매칭을 유발하는 흔한 단어
        STOPWORDS = {
            "performance", "analysis", "impact", "data", "high", "low",
            "rate", "score", "level", "found", "shows", "below", "above",
            "patient", "student", "item", "value", "total", "average",
            "significant", "result", "effect", "factor", "based", "type",
            "through", "targeted", "improve", "outcome", "treatment",
        }

        # LLM 인사이트에서 인과 변수 키워드 추출 (화살표 패턴)
        causal_keywords = set()
        arrow_matches = _re.findall(r'(\w+)\s*[→\->]+\s*(\w+)', search_text)
        for src, tgt in arrow_matches:
            if len(src) > 3:
                causal_keywords.add(src.lower())
            if len(tgt) > 3:
                causal_keywords.add(tgt.lower())

        best_match = None
        best_score = 0

        for algo in algorithm_insights:
            algo_dict = algo.to_dict() if hasattr(algo, 'to_dict') else {}
            if not algo_dict:
                continue

            score = 0
            algo_column = algo_dict.get("source_column", "").lower()
            algo_title = algo_dict.get("title", "").lower()
            algo_desc = algo_dict.get("description", "").lower()
            algo_type = algo_dict.get("insight_type", "").lower()
            algo_severity = algo_dict.get("severity", "").lower()
            algo_text = f"{algo_title} {algo_desc}"

            # 팩터 1: source_column 매칭
            if source_column and algo_column:
                if source_column == algo_column:
                    score += 10
                elif source_column in algo_column or algo_column in source_column:
                    score += 6

            # 팩터 2: insight_type 매칭
            if insight_type and algo_type and insight_type == algo_type:
                score += 5

            # 팩터 3: severity 매칭
            if severity and algo_severity and severity == algo_severity:
                score += 2

            # 팩터 4: 인과 변수 키워드 매칭
            if causal_keywords:
                for ck in causal_keywords:
                    if ck in algo_text:
                        score += 8
                        break

            # 팩터 5: 의미 있는 단어 겹침 (stopwords 제외)
            search_words = set(w for w in search_text.split() if len(w) > 3 and w not in STOPWORDS)
            algo_words = set(w for w in algo_text.split() if len(w) > 3 and w not in STOPWORDS)
            overlap = search_words & algo_words
            score += len(overlap)

            if score > best_score:
                best_score = score
                best_match = algo_dict

        # 최소 임계값 3 — 미달 시 빈 dict (잘못된 fallback 방지)
        if best_match and best_score >= 3:
            return best_match

        return {}

    def _find_related_simulation(
        self,
        llm_insight: Dict[str, Any],
        simulation_report: Any,
    ) -> Dict[str, Any]:
        """LLM 인사이트와 관련된 시뮬레이션 실험 찾기"""
        if not simulation_report or not hasattr(simulation_report, 'experiments'):
            return {}

        if not simulation_report.experiments:
            return {}

        source_table = llm_insight.get("source_table", "").lower()
        source_column = llm_insight.get("source_column", "").lower()
        title = llm_insight.get("title", "").lower()
        hypothesis = llm_insight.get("hypothesis", "").lower()

        def _try_asdict(exp):
            from dataclasses import asdict
            try:
                return asdict(exp)
            except Exception:
                return {"description": str(exp)}

        # 1차: source_table/source_column 매칭
        for exp in simulation_report.experiments:
            exp_table = getattr(exp, 'source_table', "").lower()
            exp_column = getattr(exp, 'source_column', "").lower()

            if source_table and source_table in exp_table:
                return _try_asdict(exp)
            if source_column and source_column in exp_column:
                return _try_asdict(exp)

        # 2차: 제목/가설 키워드 매칭
        search_text = f"{title} {hypothesis}"
        search_words = set(w for w in search_text.split() if len(w) > 3)
        for exp in simulation_report.experiments:
            exp_name = getattr(exp, 'name', getattr(exp, 'description', '')).lower()
            exp_words = set(w for w in exp_name.split() if len(w) > 3)
            if len(search_words & exp_words) >= 1:
                return _try_asdict(exp)

        # 3차: 첫 번째 실험 반환 (fallback)
        return _try_asdict(simulation_report.experiments[0])

        return {}

    def _calculate_combined_confidence(
        self,
        algorithm: Dict[str, Any],
        simulation: Dict[str, Any],
        llm_insight: Dict[str, Any],
    ) -> float:
        """세 단계의 신뢰도 종합"""
        confidences = []
        weights = []

        # Algorithm confidence (weight: 0.35)
        if algorithm:
            algo_conf = algorithm.get("confidence", 0.5)
            if isinstance(algo_conf, (int, float)):
                confidences.append(algo_conf)
                weights.append(0.35)

        # Simulation confidence (weight: 0.35)
        if simulation:
            sim_conf = simulation.get("statistical_significance", 0.5)
            if isinstance(sim_conf, (int, float)):
                # p-value를 confidence로 변환 (1 - p-value)
                sim_confidence = 1.0 - min(sim_conf, 1.0) if sim_conf < 0.5 else sim_conf
                confidences.append(sim_confidence)
                weights.append(0.35)

        # LLM confidence (weight: 0.30)
        llm_conf = llm_insight.get("confidence", 0.7)
        if isinstance(llm_conf, (int, float)):
            confidences.append(llm_conf)
            weights.append(0.30)

        if not confidences:
            return 0.5

        # 가중 평균
        total_weight = sum(weights)
        weighted_sum = sum(c * w for c, w in zip(confidences, weights))

        return round(weighted_sum / total_weight, 3) if total_weight > 0 else 0.5

    def _parse_llm_json(self, text: str, default: Any = None) -> Any:
        """LLM 응답에서 JSON 파싱 (v21.0: 강화된 파서)"""
        import re

        if not text or not text.strip():
            return default

        original_text = text.strip()

        # Step 1: 코드 블록 제거 (string ops — regex보다 확실)
        stripped = original_text
        if stripped.startswith("```"):
            # 첫 줄 제거 (```json 또는 ```)
            first_newline = stripped.find('\n')
            if first_newline > 0:
                stripped = stripped[first_newline + 1:]
            # 마지막 ``` 제거
            if stripped.rstrip().endswith("```"):
                stripped = stripped.rstrip()
                stripped = stripped[:-3].rstrip()

        # Step 2: 직접 파싱
        for candidate in [stripped, original_text]:
            candidate = candidate.strip()
            try:
                return json.loads(candidate)
            except (json.JSONDecodeError, ValueError):
                pass

        # Step 3: 첫 번째 [ ... 마지막 ] 추출 (array)
        arr_start = original_text.find('[')
        arr_end = original_text.rfind(']')
        if arr_start >= 0 and arr_end > arr_start:
            try:
                return json.loads(original_text[arr_start:arr_end + 1])
            except (json.JSONDecodeError, ValueError):
                pass

        # Step 4: 첫 번째 { ... 마지막 } 추출 (object)
        obj_start = original_text.find('{')
        obj_end = original_text.rfind('}')
        if obj_start >= 0 and obj_end > obj_start:
            try:
                return json.loads(original_text[obj_start:obj_end + 1])
            except (json.JSONDecodeError, ValueError):
                pass

        logger.warning(f"[Pipeline] JSON parse failed. text_len={len(original_text)}, first300: {repr(original_text[:300])}")
        return default
