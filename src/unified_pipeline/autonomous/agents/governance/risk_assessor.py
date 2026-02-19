"""
Risk Assessor Autonomous Agent (v3.0)

리스크 평가 자율 에이전트 - NIST 기반 리스크 프레임워크
- Likelihood x Impact = Risk Score
- Business Insights 기반 데이터 리스크 분석
- CausalImpactAnalyzer로 인과 분석
- AnomalyDetector로 앙상블 이상탐지
- Counterfactual Reasoning (v6.1)
- Column-level Lineage (v16.0)
- UnifiedInsightPipeline (v19.0)
"""

import json
import logging
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from ...base import AutonomousAgent
from ...analysis import (
    GovernanceRiskAssessor,
    parse_llm_json,
    # v2.1: Business Insights
    BusinessInsightsAnalyzer,
    # === 고급 분석 모듈 (v3.0 신규 통합) ===
    CausalImpactAnalyzer,  # 인과 추론
    AnomalyDetector,  # 앙상블 이상탐지
    # === v6.1: Counterfactual Reasoning ===
    CounterfactualAnalyzer,
    CounterfactualResult,
)

# === v19.0: Unified Insight Pipeline ===
try:
    from ...analysis.unified_insight_pipeline import UnifiedInsightPipeline, PipelineResult
    UNIFIED_PIPELINE_AVAILABLE = True
except ImportError:
    UNIFIED_PIPELINE_AVAILABLE = False
    UnifiedInsightPipeline = None
    PipelineResult = None

# === v16.0: Column-level Lineage ===
try:
    from ....lineage import LineageTracker, build_lineage_from_context, ImpactAnalysis
    LINEAGE_AVAILABLE = True
except ImportError:
    LINEAGE_AVAILABLE = False
    LineageTracker = None
    build_lineage_from_context = None
    ImpactAnalysis = None

if TYPE_CHECKING:
    from ....todo.models import PipelineTodo, TodoResult
    from ....shared_context import SharedContext

logger = logging.getLogger(__name__)


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
        from ....todo.models import TodoResult
        from ...analysis.governance import RiskAssessor as GovernanceRiskAssessor

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
            # v22.1: 전체 데이터 로드 (샘플링 금지) | v28.6: 캐시 재사용
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
                        from ...analysis.business_insights import BusinessInsight, InsightType, InsightSeverity
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
        # v27.8: 리스크 평가 결과 캐시 (What-If, Counterfactual에서 재사용)
        _risk_cache: Dict[str, list] = {}

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
            _risk_cache[concept.concept_id] = risks
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
            # v22.1: 전체 데이터 로드 (샘플링 금지) | v28.6: 캐시 재사용
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
                lineage_stats = {
                    "total_nodes": len(getattr(lineage_tracker, 'nodes', {})),
                    "total_edges": len(getattr(lineage_tracker, 'edges', {})),
                }

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
                                    impact = lineage_tracker.analyze_impact(col_id)
                                    if impact:
                                        impact_analyses.append({
                                            "concept_id": concept.concept_id,
                                            "source_column": col_id,
                                            "direct_dependencies": impact.direct_dependencies,
                                            "total_impact_count": impact.total_impact_count,
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
                context.set_dynamic("lineage_graph", lineage_tracker.export_graph() if hasattr(lineage_tracker, 'export_graph') else lineage_tracker)
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
                # v27.8: 캐시된 리스크 평가 결과 사용 (중복 assess_concept_risk 호출 제거)
                high_risk_concepts = [c for c in context.ontology_concepts if
                    any(r.risk_level in ["high", "critical"] for r in _risk_cache.get(c.concept_id, []))]

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
            # v27.8: 캐시된 리스크 평가 결과 사용 (중복 assess_concept_risk 호출 제거)
            analysis_concepts = []
            for c in context.ontology_concepts[:10]:
                cached_risks = _risk_cache.get(c.concept_id, [])
                if any(r.risk_level in ["high", "critical", "medium", "low"] for r in cached_risks):
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
        from ...analysis.cross_entity_correlation import CorrelationResult
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
