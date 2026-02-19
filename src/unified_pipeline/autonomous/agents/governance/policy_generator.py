"""
Policy Generator Autonomous Agent (v3.1)

정책 생성 자율 에이전트 - 템플릿 기반 표준 정책 생성
- 개념 유형별 맞춤 규칙
- KPIPredictor로 시계열 예측 및 ROI 분석
- TimeSeriesForecaster로 메트릭 예측 (v3.1)
- LLM을 통한 도메인 특화 커스터마이징
"""

import json
import logging
from typing import Any, Dict, List, TYPE_CHECKING

from ...base import AutonomousAgent
from ...analysis import (
    PolicyGenerator,
    parse_llm_json,
    # v3.0: KPI 예측 및 ROI 분석
    KPIPredictor,
    # v3.1: 시계열 예측
    TimeSeriesForecaster,
    ForecastResult,
)

if TYPE_CHECKING:
    from ....todo.models import PipelineTodo, TodoResult
    from ....shared_context import SharedContext

logger = logging.getLogger(__name__)


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
        from ....todo.models import TodoResult
        from ...analysis.governance import PolicyGenerator

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
            # v22.1: 전체 데이터 로드 (샘플링 금지) | v28.6: 캐시 재사용
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
            # v22.1: 전체 데이터 로드 (샘플링 금지) | v28.6: 캐시 재사용
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
                                        "confidence": 1.0 - forecast.metrics.get("mape", 50) / 100.0 if forecast.metrics else 0.5,
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
            input_data={"task": "policy_generation", "concepts": len(enhanced_policies)},
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
