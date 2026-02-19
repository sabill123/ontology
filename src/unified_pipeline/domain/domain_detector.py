"""
Domain Detector - LLM-based Automatic Domain Detection

자동 도메인 탐지 시스템
- LLM을 사용한 산업 도메인 자동 분류
- 테이블/컬럼 이름에서 도메인 시그널 추출
- 도메인별 권장 워크플로우 및 대시보드 자동 추천
- Phase 2/3에서 사용할 인사이트 패턴 자동 생성

Based on: _legacy/phase1/domain_detector.py
"""

from typing import List, Dict, Any, Optional, TYPE_CHECKING
from dataclasses import dataclass
from datetime import datetime
import json
import logging

from ..model_config import get_service_model, get_model_spec

from .domain_context import (
    DomainContext,
    InsightPattern,
    IndustryDomain,
    BusinessStage,
    DataType,
    DEFAULT_DOMAIN_CONTEXT,
)

if TYPE_CHECKING:
    from ..shared_context import SharedContext, TableInfo

logger = logging.getLogger(__name__)


# Domain signal patterns for rule-based detection
DOMAIN_SIGNALS = {
    "beauty_cosmetics": {
        "column_signals": [
            "skin_type", "beauty", "cosmetic", "shade", "lipstick", "foundation",
            "serum", "skincare", "makeup", "preferred_finish", "sensitivities",
            "derm", "complexion", "tone", "facial", "beauty_consult", "shade_code",
            "routine_goal", "skin_concern", "product_category"
        ],
        "table_signals": [
            "beauty", "cosmetic", "skin", "loyalty_lab", "shade", "makeup",
            "skin_profile", "beauty_consultant", "product_catalog"
        ],
        "value_signals": [
            "oily", "dry", "combination", "sensitive", "matte", "dewy", "natural"
        ]
    },
    "retail": {
        "column_signals": [
            "order", "transaction", "receipt", "customer", "loyalty", "store",
            "ticket", "sku", "product", "basket", "checkout", "cart", "invoice"
        ],
        "table_signals": [
            "order_facts", "store_transactions", "ecommerce", "pos", "inventory",
            "products", "customers", "orders", "sales"
        ],
        "value_signals": []
    },
    "manufacturing": {
        "column_signals": [
            "machine", "equipment", "sensor", "production", "defect", "quality",
            "batch", "lot", "assembly", "work_order", "oee", "downtime", "yield"
        ],
        "table_signals": [
            "production", "sensor", "equipment", "quality", "work_orders",
            "maintenance", "defects", "machines"
        ],
        "value_signals": []
    },
    "healthcare": {
        "column_signals": [
            "patient", "diagnosis", "medication", "treatment", "provider",
            "appointment", "insurance", "claim", "lab_result", "prescription"
        ],
        "table_signals": [
            "patients", "encounters", "claims", "prescriptions", "lab_results",
            "providers", "appointments", "diagnoses"
        ],
        "value_signals": []
    },
    "finance": {
        "column_signals": [
            "account", "transaction", "balance", "credit", "debit", "portfolio",
            "investment", "loan", "interest", "payment", "currency", "exchange"
        ],
        "table_signals": [
            "accounts", "transactions", "portfolios", "loans", "payments",
            "investments", "ledger"
        ],
        "value_signals": []
    },
    "logistics": {
        "column_signals": [
            "shipment", "carrier", "tracking", "warehouse", "delivery", "route",
            "freight", "container", "logistics", "distribution"
        ],
        "table_signals": [
            "shipments", "warehouses", "carriers", "routes", "deliveries",
            "inventory", "tracking"
        ],
        "value_signals": []
    },
    "advertising": {
        "column_signals": [
            # Campaign & Ad Structure
            "campaign", "campaign_id", "ad_group", "ad_group_id", "ad_id", "adset",
            "creative", "headline", "description", "call_to_action", "cta",
            # Performance Metrics
            "impressions", "clicks", "ctr", "click_through_rate",
            "conversions", "conversion_rate", "cvr", "conversion_value",
            "spend", "cost", "cpc", "cost_per_click", "cpm", "cost_per_mille",
            "cpa", "cost_per_acquisition", "roas", "return_on_ad_spend",
            # Targeting
            "targeting", "audience", "interest", "demographic", "age_min", "age_max",
            "gender", "placement", "device", "platform",
            # Bidding & Budget
            "bid", "bid_amount", "budget", "daily_budget", "lifetime_budget",
            "bid_strategy", "optimization_goal",
            # Engagement
            "reach", "frequency", "engagement", "video_views", "view_rate",
            "actions", "link_click", "landing_page_view", "lead", "purchase",
            # Quality & Ranking
            "quality_score", "relevance_score", "ad_rank", "position"
        ],
        "table_signals": [
            "campaigns", "ad_groups", "ads", "adsets", "creatives",
            "ad_metrics", "ad_insights", "ad_performance",
            "google_ads", "meta_ads", "naver_ads", "facebook_ads",
            "google_campaigns", "meta_campaigns", "naver_campaigns",
            "keywords", "trend_keywords", "competitor_ads", "viral_alerts"
        ],
        "value_signals": [
            "enabled", "paused", "active", "removed", "archived",
            "search", "display", "video", "shopping", "discovery",
            "conversions", "brand_awareness", "traffic", "engagement",
            "cpc", "cpm", "cpa", "roas", "maximize_clicks", "target_cpa"
        ]
    },
    "hr_human_resources": {
        "column_signals": [
            "employee", "salary", "department", "role", "position", "title",
            "performance", "turnover", "overtime", "satisfaction", "tenure",
            "years_exp", "experience", "hire_date", "termination", "leave",
            "benefit", "bonus", "compensation", "payroll", "headcount",
            "remote", "onboarding", "promotion", "appraisal", "attendance",
            "manager", "team", "shift", "contract", "probation"
        ],
        "table_signals": [
            "employees", "staff", "payroll", "departments", "positions",
            "performance_reviews", "attendance", "leave", "benefits",
            "compensation", "headcount", "workforce"
        ],
        "value_signals": [
            "junior", "senior", "manager", "director", "lead",
            "full_time", "part_time", "contract", "intern",
            "high", "medium", "low", "active", "terminated", "resigned"
        ]
    },
    "education": {
        "column_signals": [
            "student", "grade", "score", "math", "english", "science", "reading",
            "writing", "attendance", "study_hours", "tutoring", "gpa", "credits",
            "semester", "course", "exam", "quiz", "homework", "scholarship",
            "enrollment", "teacher", "class", "subject", "curriculum", "academic",
            "student_id", "math_score", "english_score", "science_score",
            "private_tutoring", "parent_education", "school", "major",
            "extracurricular", "library_hours", "absent_days", "dropout"
        ],
        "table_signals": [
            "students", "courses", "enrollments", "grades", "exams", "classes",
            "teachers", "subjects", "academic", "transcripts", "attendance",
            "schools", "departments", "curricula", "semesters"
        ],
        "value_signals": [
            "pass", "fail", "freshman", "sophomore", "junior", "senior",
            "undergraduate", "graduate", "bachelor", "master", "doctorate",
            "high_school", "middle_school", "elementary"
        ]
    }
}

# Default recommendations by industry
INDUSTRY_RECOMMENDATIONS = {
    "beauty_cosmetics": {
        "workflows": [
            "customer_segmentation",
            "personalized_recommendation",
            "loyalty_program_analysis",
            "demand_forecasting",
            "inventory_optimization"
        ],
        "dashboards": [
            "customer_360",
            "store_performance",
            "kpi_dashboard",
            "chat_assistant"
        ]
    },
    "retail": {
        "workflows": [
            "customer_segmentation",
            "demand_forecasting",
            "inventory_optimization",
            "performance_tracking",
            "loyalty_program_analysis"
        ],
        "dashboards": [
            "store_performance",
            "kpi_dashboard",
            "timeline_view",
            "workflow_queue"
        ]
    },
    "manufacturing": {
        "workflows": [
            "preventive_maintenance",
            "quality_control",
            "demand_forecasting",
            "inventory_optimization",
            "performance_tracking"
        ],
        "dashboards": [
            "operations_room",
            "kpi_dashboard",
            "timeline_view",
            "workflow_queue"
        ]
    },
    "healthcare": {
        "workflows": [
            "patient_care_coordination",
            "performance_tracking",
            "demand_forecasting",
            "inventory_optimization",
            "general_monitoring"
        ],
        "dashboards": [
            "kpi_dashboard",
            "timeline_view",
            "workflow_queue",
            "chat_assistant"
        ]
    },
    "hr_human_resources": {
        "workflows": [
            "employee_performance_analysis",
            "turnover_prediction",
            "compensation_benchmarking",
            "workforce_planning",
            "engagement_monitoring"
        ],
        "dashboards": [
            "workforce_overview_dashboard",
            "kpi_dashboard",
            "turnover_risk_monitor",
            "compensation_analysis_view"
        ]
    },
    "education": {
        "workflows": [
            "student_performance_analysis",
            "attendance_monitoring",
            "learning_outcome_tracking",
            "intervention_effectiveness",
            "cohort_comparison"
        ],
        "dashboards": [
            "student_performance_dashboard",
            "kpi_dashboard",
            "cohort_analysis_view",
            "intervention_impact_view"
        ]
    },
    "general": {
        "workflows": [
            "general_monitoring",
            "performance_tracking",
            "demand_forecasting"
        ],
        "dashboards": [
            "kpi_dashboard",
            "timeline_view"
        ]
    },
    "advertising": {
        "workflows": [
            "campaign_performance_analysis",
            "cross_platform_optimization",
            "audience_segmentation",
            "budget_allocation_optimization",
            "creative_performance_analysis",
            "competitor_benchmarking",
            "trend_keyword_monitoring",
            "conversion_funnel_analysis"
        ],
        "dashboards": [
            "campaign_performance_dashboard",
            "cross_platform_comparison",
            "kpi_dashboard",
            "trend_analysis_view",
            "competitor_intelligence",
            "budget_pacing_monitor"
        ]
    }
}


class DomainDetector:
    """
    LLM-based domain detector that analyzes data sources to determine:
    - Industry domain
    - Business context (stage, department)
    - Key entities and metrics
    - Recommended workflows and dashboards
    - Required insight patterns (Dynamic)
    """

    def __init__(
        self,
        use_llm: bool = True,
        llm_client=None,
        model: str = None,
    ):
        self.use_llm = use_llm
        self.llm_client = llm_client
        self.model = model or get_service_model("domain_detection")

    def detect_domain(
        self,
        context: "SharedContext",
    ) -> DomainContext:
        """
        Detect domain context from SharedContext tables.

        Args:
            context: SharedContext with loaded tables

        Returns:
            DomainContext with detected information
        """
        if not context.tables:
            logger.warning("No tables in context, returning default domain")
            return DEFAULT_DOMAIN_CONTEXT

        # Step 1: Gather evidence from tables
        evidence = self._gather_evidence(context)

        # Step 2: Rule-based detection first
        rule_based_result = self._rule_based_detection(evidence)

        # Step 3: LLM-based refinement if enabled
        if self.use_llm and self.llm_client:
            try:
                domain_info = self._llm_classify_domain(evidence, rule_based_result)
            except Exception as e:
                logger.error(f"LLM domain detection failed: {e}")
                domain_info = rule_based_result
        else:
            domain_info = rule_based_result

        # Step 4: Build domain context
        industry = domain_info.get("industry", "general")
        recommendations = INDUSTRY_RECOMMENDATIONS.get(
            industry, INDUSTRY_RECOMMENDATIONS["general"]
        )

        # Generate insight patterns
        insight_patterns = self._generate_insight_patterns(
            evidence, domain_info, context
        )

        domain_context = DomainContext(
            industry=industry,
            industry_confidence=domain_info.get("industry_confidence", 0.5),
            industry_evidence=domain_info.get("evidence_signals", []),
            business_stage=domain_info.get("business_stage", "operations"),
            department=domain_info.get("department", "general"),
            data_type=domain_info.get("data_type", "operational"),
            temporal_granularity=domain_info.get("temporal_granularity", "daily"),
            key_entities=domain_info.get("key_entities", []),
            key_metrics=domain_info.get("key_metrics", []),
            recommended_workflows=domain_info.get(
                "recommended_workflows", recommendations["workflows"]
            ),
            recommended_dashboards=domain_info.get(
                "recommended_dashboards", recommendations["dashboards"]
            ),
            required_insight_patterns=insight_patterns,
            detection_timestamp=datetime.now().isoformat(),
            detection_evidence=evidence,
            is_auto_detected=True,
        )

        logger.info(
            f"Domain detected: {domain_context.industry} "
            f"(confidence: {domain_context.industry_confidence:.2f})"
        )

        return domain_context

    def _gather_evidence(self, context: "SharedContext") -> Dict[str, Any]:
        """Gather evidence from SharedContext tables"""
        evidence = {
            "table_names": [],
            "column_names": [],
            "column_patterns": {},
            "data_characteristics": {},
            "sample_values": {},
            "relationships": []
        }

        for table_name, table_info in context.tables.items():
            # Table names
            evidence["table_names"].append(table_name)

            # Column analysis
            for col in table_info.columns:
                col_name = col.get("name", "") if isinstance(col, dict) else col.name if hasattr(col, 'name') else str(col)
                evidence["column_names"].append(col_name.lower())

                # Get column type
                col_type = col.get("type", "unknown") if isinstance(col, dict) else getattr(col, 'inferred_type', 'unknown')
                if col_type not in evidence["column_patterns"]:
                    evidence["column_patterns"][col_type] = []
                evidence["column_patterns"][col_type].append({
                    "table": table_name,
                    "column": col_name,
                })

            # Data characteristics
            evidence["data_characteristics"][table_name] = {
                "row_count": table_info.row_count,
                "column_count": len(table_info.columns),
                "has_timestamps": any(
                    "time" in str(c.get("name", "") if isinstance(c, dict) else c).lower() or
                    "date" in str(c.get("name", "") if isinstance(c, dict) else c).lower()
                    for c in table_info.columns
                ),
                "has_ids": any(
                    "_id" in str(c.get("name", "") if isinstance(c, dict) else c).lower() or
                    "_code" in str(c.get("name", "") if isinstance(c, dict) else c).lower()
                    for c in table_info.columns
                ),
                "primary_keys": table_info.primary_keys,
                "foreign_keys": table_info.foreign_keys,
            }

            # Sample values (first few samples for pattern detection)
            if table_info.sample_data:
                evidence["sample_values"][table_name] = table_info.sample_data[:5]

        return evidence

    def _rule_based_detection(self, evidence: Dict[str, Any]) -> Dict[str, Any]:
        """Rule-based domain detection using signal patterns"""
        scores = {}
        signals_found = {}

        table_text = " ".join(evidence["table_names"]).lower()
        column_text = " ".join(evidence["column_names"])

        for domain, signals in DOMAIN_SIGNALS.items():
            score = 0.0
            found_signals = []

            # Check table signals
            for signal in signals["table_signals"]:
                if signal in table_text:
                    score += 0.3
                    found_signals.append(f"table:{signal}")

            # Check column signals
            for signal in signals["column_signals"]:
                if signal in column_text:
                    score += 0.2
                    found_signals.append(f"column:{signal}")

            # Check value signals in sample data
            for table_samples in evidence.get("sample_values", {}).values():
                for sample in table_samples:
                    for key, value in sample.items() if isinstance(sample, dict) else []:
                        value_str = str(value).lower()
                        for signal in signals.get("value_signals", []):
                            if signal in value_str:
                                score += 0.1
                                found_signals.append(f"value:{signal}")

            scores[domain] = min(score, 1.0)
            signals_found[domain] = found_signals

        # Find best match
        best_domain = max(scores.keys(), key=lambda k: scores[k])
        best_score = scores[best_domain]

        # If no strong signal, default to general
        if best_score < 0.2:
            best_domain = "general"
            best_score = 0.5

        # v14.0: 제거됨 - beauty + retail 하드코딩 규칙 제거 (LLM이 판단)
        # LLM-First 원칙: 특정 도메인 조합 규칙은 LLM이 데이터 특성으로 판단

        return {
            "industry": best_domain,
            "industry_confidence": best_score,
            "evidence_signals": signals_found.get(best_domain, []),
            "all_scores": scores,
            "business_stage": "operations",
            "department": "general",
            "data_type": "operational",
            "temporal_granularity": "daily",
            "key_entities": [],
            "key_metrics": [],
        }

    def _llm_classify_domain(
        self,
        evidence: Dict[str, Any],
        rule_based_result: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Use LLM to classify domain based on evidence"""
        from ..autonomous.analysis.utils import parse_llm_json

        prompt = f"""You are a domain expert analyzing a dataset to determine its business context.

IMPORTANT:
- Respond in JSON format only.
- For open text fields like 'key_entities', 'key_metrics', write in Korean for Korean users.
- Keep enum values (industry, business_stage, etc.) in English for system compatibility.

## CONTEXT:
Rule-based detection suggests: {rule_based_result['industry']} (confidence: {rule_based_result['industry_confidence']:.2f})
Signals found: {rule_based_result.get('evidence_signals', [])}

## DOMAIN DETECTION APPROACH (LLM-First):

You are an expert data analyst. Analyze the table/column names and data characteristics to determine:
1. What industry domain this data belongs to
2. What business processes it supports

DO NOT rely on predefined rules. Instead:
- Examine the semantic meaning of column names
- Consider the relationships between tables
- Look at data patterns and characteristics
- Use your knowledge of various industries to make the BEST determination

Be SPECIFIC - if data shows supply chain logistics, say "supply_chain" or "logistics", not "general".
If data shows beauty products with customer profiles, say "beauty_cosmetics", not "retail".

## DATA TO ANALYZE:

Table Names: {evidence['table_names']}
Column Names (sample): {evidence['column_names'][:50]}
Data Characteristics: {json.dumps(evidence['data_characteristics'], indent=2)}

## REQUIRED OUTPUT:

1. **industry**: Determine the most appropriate industry based on data characteristics. Examples include but are NOT limited to: supply_chain, logistics, manufacturing, retail, e_commerce, beauty_cosmetics, healthcare, finance, advertising, fashion, food_beverage, energy, telecom, hospitality, education, media_entertainment, technology, automotive, pharmaceuticals, agriculture, construction, real_estate, transportation, aerospace, chemicals, insurance, banking, consulting, legal, government, nonprofit, etc. Use snake_case format and choose the MOST SPECIFIC industry that fits the data.
2. **industry_confidence**: 0.0-1.0
3. **business_stage**: Determine the primary business function/stage this data represents. Examples include but are NOT limited to: production, sales, marketing, operations, quality, maintenance, finance, hr, customer_service, supply_chain, procurement, inventory_management, logistics, distribution, warehousing, shipping, demand_planning, order_fulfillment, vendor_management, etc. Use snake_case format and choose the MOST SPECIFIC stage.
4. **department**: Which department owns this data
5. **data_type**: One of: operational, transactional, analytical, master_data, reference_data
6. **temporal_granularity**: One of: real_time, hourly, daily, weekly, monthly, yearly
7. **key_entities**: List of main business entities (in Korean)
8. **key_metrics**: List of main KPIs (in Korean)

Respond ONLY with valid JSON:
{{
  "industry": "...",
  "industry_confidence": 0.0-1.0,
  "business_stage": "...",
  "department": "...",
  "data_type": "...",
  "temporal_granularity": "...",
  "key_entities": ["...", "..."],
  "key_metrics": ["...", "..."]
}}
"""

        try:
            response = self.llm_client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=get_model_spec(self.model).get("output_max", 32_000),
                response_format={"type": "json_object"}
            )

            result_text = response.choices[0].message.content
            result = parse_llm_json(result_text)

            # Merge with rule-based result
            result["evidence_signals"] = rule_based_result.get("evidence_signals", [])

            # Boost confidence if LLM agrees with rule-based
            if result.get("industry") == rule_based_result.get("industry"):
                result["industry_confidence"] = min(
                    result.get("industry_confidence", 0.5) + 0.1, 1.0
                )

            return result

        except Exception as e:
            logger.error(f"LLM classification failed: {e}")
            return rule_based_result

    def _generate_insight_patterns(
        self,
        evidence: Dict[str, Any],
        domain_info: Dict[str, Any],
        context: "SharedContext",
    ) -> List[InsightPattern]:
        """Generate insight patterns based on detected domain"""
        patterns = []
        industry = domain_info.get("industry", "general")
        table_names = evidence.get("table_names", [])

        # Pattern 1: Cross-table customer identity resolution
        customer_tables = [t for t in table_names if "customer" in t.lower() or "profile" in t.lower()]
        transaction_tables = [t for t in table_names if "order" in t.lower() or "transaction" in t.lower()]

        if customer_tables and transaction_tables:
            patterns.append(InsightPattern(
                pattern_id="INS-001",
                name="고객 ID 통합 분석",
                description=f"{customer_tables[0]}와 {transaction_tables[0]}의 고객 ID를 매칭하여 통합 고객 프로필 생성",
                source_tables=customer_tables[:1] + transaction_tables[:1],
                join_columns={"customer_id": "customer_id"},
                expected_kpis=["고객당 구매금액", "구매 빈도", "고객 생애가치"]
            ))

        # Pattern 2: Industry-specific patterns
        if industry == "beauty_cosmetics":
            skin_tables = [t for t in table_names if "skin" in t.lower() or "profile" in t.lower()]
            if skin_tables:
                patterns.append(InsightPattern(
                    pattern_id="INS-002",
                    name="피부 프로필-구매 매칭",
                    description="피부 타입별 제품 구매 패턴 분석",
                    source_tables=skin_tables + transaction_tables[:1],
                    expected_kpis=["피부타입별 전환율", "추천 정확도", "재구매율"]
                ))

        elif industry == "manufacturing":
            equipment_tables = [t for t in table_names if "equipment" in t.lower() or "machine" in t.lower()]
            quality_tables = [t for t in table_names if "quality" in t.lower() or "defect" in t.lower()]
            if equipment_tables and quality_tables:
                patterns.append(InsightPattern(
                    pattern_id="INS-002",
                    name="설비-품질 상관 분석",
                    description="설비 상태와 불량률 간 상관관계 분석",
                    source_tables=equipment_tables[:1] + quality_tables[:1],
                    expected_kpis=["OEE", "불량률", "예방정비 효율"]
                ))

        elif industry == "healthcare":
            patient_tables = [t for t in table_names if "patient" in t.lower()]
            encounter_tables = [t for t in table_names if "encounter" in t.lower() or "visit" in t.lower()]
            if patient_tables and encounter_tables:
                patterns.append(InsightPattern(
                    pattern_id="INS-002",
                    name="환자 여정 분석",
                    description="환자별 진료 경로 및 결과 분석",
                    source_tables=patient_tables[:1] + encounter_tables[:1],
                    expected_kpis=["재입원율", "평균 진료 시간", "환자 만족도"]
                ))

        # Pattern 3: Time-series analysis pattern
        timestamp_tables = [
            t for t in table_names
            if evidence["data_characteristics"].get(t, {}).get("has_timestamps", False)
        ]
        if timestamp_tables:
            patterns.append(InsightPattern(
                pattern_id="INS-003",
                name="시계열 트렌드 분석",
                description=f"{timestamp_tables[0]}의 시간별 패턴 및 트렌드 분석",
                source_tables=timestamp_tables[:1],
                expected_kpis=["성장률", "시즌성 지수", "예측 정확도"]
            ))

        return patterns


def create_domain_detector(
    use_llm: bool = True,
    llm_client=None,
) -> DomainDetector:
    """Factory function to create DomainDetector"""
    return DomainDetector(use_llm=use_llm, llm_client=llm_client)
