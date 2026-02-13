"""
Data Analyst Autonomous Agent (v14.0 - LLM-First)

LLM이 전체 데이터셋을 직접 읽고 분석하는 접근법.
도메인 파악, 컬럼 의미 분석, 데이터 패턴 발견, 의미론적 품질 판단.
"""

import json
import logging
from typing import Dict, Any, List, Optional, TYPE_CHECKING

from ...base import AutonomousAgent
from ...analysis import (
    parse_llm_json,
    DataQualityAnalyzer,
)

if TYPE_CHECKING:
    from ....todo.models import PipelineTodo, TodoResult
    from ....shared_context import SharedContext

logger = logging.getLogger(__name__)


class DataAnalystAutonomousAgent(AutonomousAgent):
    """
    v14.0: 데이터 분석 자율 에이전트 (LLM-First 접근법)

    핵심 원칙: LLM이 전체 데이터셋을 직접 읽고 분석합니다.
    - 알고리즘이 먼저 계산하고 LLM이 "해석"하는 방식 X
    - LLM이 데이터를 직접 보고 "분석 + 판단"하는 방식 O

    분석 목표:
    1. 도메인 파악 (웹 분석, 물류, 금융 등)
    2. 컬럼 의미 파악 + 비즈니스 특성 분류
    3. 데이터 패턴 발견 (A+B=1 수학적 관계)
    4. 의미론적 데이터 품질 이슈 (비즈니스 맥락 기반)
    5. 분석 방향 제안 (세그먼트별 분석이 핵심)

    v14.0 변경사항:
    - 컬럼 비즈니스 특성 분류 추가 (required/optional/derived/identifier)
    - 의미론적 품질 판단 (null이 많다고 무조건 문제 아님)
    - 테이블/컬럼 관계 재정의를 위한 구조 분석
    """

    @property
    def agent_type(self) -> str:
        return "data_analyst"

    @property
    def agent_name(self) -> str:
        return "Data Analyst"

    @property
    def phase(self) -> str:
        return "discovery"

    def get_system_prompt(self) -> str:
        return """You are a Senior Data Analyst with deep expertise in understanding datasets and their business context.

Your task is to DIRECTLY ANALYZE the raw data and understand its BUSINESS MEANING.

## CORE ANALYSIS TASKS:

1. **DOMAIN IDENTIFICATION**: What business domain is this data from?

2. **COLUMN BUSINESS CHARACTERISTICS**: For each column, determine:
   - **business_role**: How this column is used in the business process
     - "identifier": Primary key, foreign key, unique ID
     - "required_input": Must have value (null = data entry error)
     - "optional_input": Can be empty by design (notes, comments, remarks, memo fields)
     - "metric": Measured/calculated numeric value
     - "dimension": Category for grouping/filtering
     - "derived": Calculated from other columns
     - "timestamp": Date/time tracking

3. **DATA PATTERNS**: Mathematical relationships between columns

4. **SEMANTIC DATA QUALITY**: Only report issues that ACTUALLY matter for business:
   - ❌ DON'T report: "비고 column has 99% nulls" (remarks are OPTIONAL by nature)
   - ❌ DON'T report: "memo field is mostly empty" (memos are OPTIONAL)
   - ✅ DO report: "customer_id has nulls" (identifiers should NEVER be null)
   - ✅ DO report: "order_date is missing in 5% of rows" (required field is empty)
   - ✅ DO report: "price column has negative values" (business logic violation)

5. **TABLE STRUCTURE ANALYSIS**: Understand the data model
   - What entity does this table represent?
   - What are the relationships to other entities?
   - Is this a master table, transaction table, or mapping table?

## CRITICAL JUDGMENT RULES:

**Empty/Null Values:**
- If a column name suggests it's OPTIONAL (비고, 메모, remarks, notes, comments, description, etc.): Empty values are NORMAL, not a quality issue
- If a column is a REQUIRED business field (ID, date, amount, name): Empty values ARE a quality issue

**Quality Issues:**
- Only report issues that would affect BUSINESS DECISIONS or DATA INTEGRITY
- Think: "Would a business user care about this?" If not, don't report it.

## REQUIRED OUTPUT FORMAT (JSON Schema):
```json
{
  "domain": "identified_business_domain",
  "tables": [
    {
      "table_name": "string",
      "entity_type": "master|transaction|mapping|reference",
      "business_description": "what this table represents",
      "columns": [
        {
          "name": "column_name",
          "business_role": "identifier|required_input|optional_input|metric|dimension|derived|timestamp",
          "data_type": "string|integer|float|date|boolean",
          "semantic_meaning": "business description of this column",
          "nullable": true|false,
          "is_key": true|false
        }
      ],
      "relationships": [
        {
          "related_table": "other_table_name",
          "relationship_type": "1:N|N:1|1:1|N:M",
          "join_columns": ["column_name"]
        }
      ]
    }
  ],
  "quality_issues": [
    {
      "table": "table_name",
      "column": "column_name",
      "issue_type": "missing_required|invalid_value|referential_integrity|business_logic",
      "severity": "high|medium|low",
      "description": "description of the issue",
      "affected_rows_pct": 0.0
    }
  ],
  "key_metrics": [
    {
      "metric_name": "string",
      "value": 0.0,
      "table": "source_table",
      "column": "source_column"
    }
  ]
}
```

Respond with ONLY valid JSON matching this schema."""

    def _compute_pandas_stats(self, df: "pd.DataFrame", table_name: str) -> Dict[str, Any]:
        """
        v22.1: pandas로 전체 데이터셋의 정확한 집계 통계를 사전 계산.
        LLM은 이 값을 그대로 사용해야 하며, 샘플에서 재계산하지 않아야 함.
        """
        import pandas as pd
        import numpy as np

        stats: Dict[str, Any] = {
            "table_name": table_name,
            "total_rows": len(df),
            "total_columns": len(df.columns),
            "numeric_aggregates": {},
            "categorical_distributions": {},
            "date_ranges": {},
            "boolean_ratios": {},
            "null_counts": {},
        }

        for col in df.columns:
            series = df[col]
            null_count = int(series.isnull().sum())
            if null_count > 0:
                stats["null_counts"][col] = {
                    "count": null_count,
                    "ratio": round(null_count / len(df), 4),
                }

            # 수치형 컬럼: SUM, MEAN, MIN, MAX, MEDIAN, STD
            if pd.api.types.is_numeric_dtype(series):
                non_null = series.dropna()
                if len(non_null) > 0:
                    stats["numeric_aggregates"][col] = {
                        "sum": round(float(non_null.sum()), 4),
                        "mean": round(float(non_null.mean()), 4),
                        "median": round(float(non_null.median()), 4),
                        "min": round(float(non_null.min()), 4),
                        "max": round(float(non_null.max()), 4),
                        "std": round(float(non_null.std()), 4) if len(non_null) > 1 else 0.0,
                        "unique_count": int(non_null.nunique()),
                    }

            # 카테고리형 컬럼 (unique 비율 < 50% 이면 카테고리로 간주)
            elif pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series):
                non_null = series.dropna()
                nunique = non_null.nunique()
                if nunique <= 50 and len(non_null) > 0:
                    vc = non_null.value_counts()
                    stats["categorical_distributions"][col] = {
                        "unique_count": int(nunique),
                        "value_counts": {str(k): int(v) for k, v in vc.items()},
                        "value_ratios": {str(k): round(v / len(df), 4) for k, v in vc.items()},
                    }

                    # Yes/No, True/False 패턴 → boolean ratio로 변환
                    lower_vals = set(non_null.str.lower().unique())
                    if lower_vals <= {"yes", "no"}:
                        yes_count = int(non_null.str.lower().eq("yes").sum())
                        stats["boolean_ratios"][col] = {
                            "true_count": yes_count,
                            "false_count": len(non_null) - yes_count,
                            "true_ratio": round(yes_count / len(df), 4),
                        }

            # 날짜형 컬럼 시도
            if pd.api.types.is_object_dtype(series) and any(kw in col.lower() for kw in ["date", "time", "dt", "_at"]):
                try:
                    date_non_null = series.dropna()
                    parsed = pd.to_datetime(date_non_null, errors="coerce").dropna()
                    if len(parsed) > 0:
                        stats["date_ranges"][col] = {
                            "min": str(parsed.min()),
                            "max": str(parsed.max()),
                            "span_days": int((parsed.max() - parsed.min()).days),
                        }
                except (ValueError, TypeError, OverflowError):
                    pass  # 비날짜 컬럼 — 정상 스킵

        return stats

    def _format_stats_for_prompt(self, stats: Dict[str, Any]) -> str:
        """사전 계산된 통계를 LLM 프롬프트용 텍스트로 변환"""
        lines = [
            f"## PRE-COMPUTED AGGREGATE STATISTICS (pandas, AUTHORITATIVE - v22.1)",
            f"These are computed from the FULL dataset ({stats['total_rows']} rows). "
            f"Use these EXACT values for any SUM, AVG, COUNT, or RATIO metrics.",
            f"Do NOT recompute aggregates from the sample data above.",
            "",
        ]

        # Numeric aggregates
        num_agg = stats.get("numeric_aggregates", {})
        if num_agg:
            lines.append("### Numeric Column Aggregates (full dataset)")
            for col, agg in num_agg.items():
                lines.append(
                    f"- **{col}**: sum={agg['sum']}, mean={agg['mean']}, "
                    f"median={agg['median']}, min={agg['min']}, max={agg['max']}, "
                    f"std={agg['std']}, unique={agg['unique_count']}"
                )
            lines.append("")

        # Boolean ratios
        bool_ratios = stats.get("boolean_ratios", {})
        if bool_ratios:
            lines.append("### Boolean/Flag Ratios (full dataset)")
            for col, br in bool_ratios.items():
                lines.append(
                    f"- **{col}**: true_count={br['true_count']}/{stats['total_rows']} "
                    f"({br['true_ratio']:.1%})"
                )
            lines.append("")

        # Categorical distributions
        cat_dist = stats.get("categorical_distributions", {})
        if cat_dist:
            lines.append("### Categorical Value Distributions (full dataset)")
            for col, dist in cat_dist.items():
                if col in bool_ratios:
                    continue  # 이미 boolean으로 보고됨
                top_values = list(dist["value_counts"].items())[:10]
                vals_str = ", ".join(f"{k}={v}" for k, v in top_values)
                lines.append(f"- **{col}** ({dist['unique_count']} unique): {vals_str}")
            lines.append("")

        # Date ranges
        date_ranges = stats.get("date_ranges", {})
        if date_ranges:
            lines.append("### Date Ranges (full dataset)")
            for col, dr in date_ranges.items():
                lines.append(f"- **{col}**: {dr['min']} ~ {dr['max']} ({dr['span_days']} days)")
            lines.append("")

        # Null counts
        null_counts = stats.get("null_counts", {})
        if null_counts:
            lines.append("### Null Counts (full dataset)")
            for col, nc in null_counts.items():
                lines.append(f"- **{col}**: {nc['count']}/{stats['total_rows']} ({nc['ratio']:.1%})")
            lines.append("")

        return "\n".join(lines)

    async def execute_task(
        self,
        todo: "PipelineTodo",
        context: "SharedContext",
    ) -> "TodoResult":
        """전체 데이터셋을 LLM에게 직접 전달하여 분석"""
        from ....todo.models import TodoResult
        import pandas as pd
        import os

        self._report_progress(0.1, "Loading full dataset for LLM analysis")

        # v22.1: 전체 CSV 데이터 로드 (샘플링 절대 금지)
        # 우선순위: 1) data_directory + 테이블명.csv → 2) table_info.source 경로 → 3) sample_data에서 DataFrame 복원
        all_data = {}
        data_dir = context.get_data_directory()

        for table_name, table_info in context.tables.items():
            # 1차: data_directory에서 CSV 찾기
            if data_dir:
                possible_paths = [
                    os.path.join(data_dir, f"{table_name}.csv"),
                    os.path.join(data_dir, f"{table_name}_utf8.csv"),
                ]
                for path in possible_paths:
                    if os.path.exists(path):
                        try:
                            df = pd.read_csv(path)
                            all_data[table_name] = {
                                "path": path,
                                "row_count": len(df),
                                "columns": list(df.columns),
                                "data": df.to_csv(index=False),
                                "_df": df,  # v27.6: DataFrame 캐시 (이중 읽기 제거)
                            }
                            logger.info(f"[v22.1] Loaded full data from data_dir: {table_name} ({len(df)} rows)")
                            break
                        except Exception as e:
                            logger.warning(f"Failed to load {path}: {e}")

            # 2차: table_info.source 경로에서 직접 로드
            if table_name not in all_data:
                source_path = getattr(table_info, "source", None) or ""
                if source_path and os.path.exists(source_path) and source_path.endswith(".csv"):
                    try:
                        df = pd.read_csv(source_path)
                        all_data[table_name] = {
                            "path": source_path,
                            "row_count": len(df),
                            "columns": list(df.columns),
                            "data": df.to_csv(index=False),
                            "_df": df,  # v27.6: DataFrame 캐시
                        }
                        logger.info(f"[v22.1] Loaded full data from source: {source_path} ({len(df)} rows)")
                    except Exception as e:
                        logger.warning(f"[v22.1] Failed to load source {source_path}: {e}")

            # 3차: sample_data에서 전체 DataFrame 복원 (마지막 수단)
            if table_name not in all_data and table_info.sample_data:
                df = pd.DataFrame(table_info.sample_data)
                all_data[table_name] = {
                    "path": None,
                    "row_count": len(df),
                    "columns": list(df.columns),
                    "data": df.to_csv(index=False),
                    "_df": df,  # v27.6: DataFrame 캐시
                }
                actual_total = getattr(table_info, "row_count", len(df))
                if actual_total > len(df):
                    logger.warning(
                        f"[v22.1] Using sample_data for {table_name}: {len(df)} rows "
                        f"(total={actual_total}). Full CSV not accessible."
                    )
                else:
                    logger.info(f"[v22.1] Loaded from sample_data: {table_name} ({len(df)} rows)")

        if not all_data:
            logger.error("No data available for analysis from any source")
            return TodoResult(
                success=False,
                output={"error": "No data available for analysis"},
                context_updates={},
            )

        # v22.1: pandas로 전체 데이터셋의 정확한 집계 통계 사전 계산
        precomputed_stats = {}
        precomputed_stats_section = ""
        try:
            for table_name in all_data:
                # v27.6: 캐시된 DataFrame 사용 (이중 CSV 읽기 제거)
                df_full = all_data[table_name].get("_df")
                if df_full is None and all_data[table_name].get("path"):
                    df_full = pd.read_csv(all_data[table_name]["path"])
                if df_full is None:
                    continue
                precomputed_stats[table_name] = self._compute_pandas_stats(df_full, table_name)
                logger.info(f"[v22.1] Pre-computed stats for {table_name}: "
                           f"{len(precomputed_stats[table_name].get('numeric_aggregates', {}))} numeric cols, "
                           f"{len(precomputed_stats[table_name].get('categorical_distributions', {}))} categorical cols")
            # 주 테이블의 통계를 프롬프트 섹션으로 변환
            primary_table_name = max(all_data.keys(), key=lambda k: all_data[k]["row_count"])
            if primary_table_name in precomputed_stats:
                precomputed_stats_section = "\n" + self._format_stats_for_prompt(precomputed_stats[primary_table_name])
        except Exception as e:
            logger.warning(f"[v22.1] Pre-computed stats failed: {e}")

        self._report_progress(0.25, "Running Data Quality Analysis (v7.0)")

        # 2.5 v7.0: DataQualityAnalyzer로 Status/Enum, Boolean Flag, Outlier 분석
        data_quality_context = ""
        quality_report = None
        try:
            # v27.6: 캐시된 DataFrame 사용 (세 번째 CSV 읽기 제거)
            tables_for_analysis = {}
            for table_name, table_info in all_data.items():
                df = table_info.get("_df")
                if df is None and table_info.get("path"):
                    df = pd.read_csv(table_info["path"])
                if df is None:
                    continue
                tables_for_analysis[table_name] = {
                    col: df[col].dropna().tolist() for col in df.columns
                }

            quality_analyzer = DataQualityAnalyzer()
            quality_report = quality_analyzer.analyze(tables_for_analysis)
            data_quality_context = quality_analyzer.generate_llm_context()

            logger.info(f"[v7.0] Data Quality Analysis: "
                       f"{len(quality_report.status_analyses)} status issues, "
                       f"{len(quality_report.boolean_analyses)} flag issues, "
                       f"{len(quality_report.outlier_analyses)} outlier issues")
        except Exception as e:
            logger.warning(f"[v7.0] Data Quality Analysis failed: {e}")
            data_quality_context = ""

        self._report_progress(0.3, "Sending full data to LLM for analysis")

        # 3. LLM에게 전체 데이터 전달
        # 가장 큰 데이터셋 선택 (또는 모든 데이터셋)
        primary_table = max(all_data.keys(), key=lambda k: all_data[k]["row_count"])
        primary_data = all_data[primary_table]

        # 데이터가 너무 크면 잘라서 전달 (LLM 컨텍스트 한계)
        data_csv = primary_data["data"]
        max_chars = 100000  # 약 100KB
        if len(data_csv) > max_chars:
            lines = data_csv.split('\n')
            header = lines[0]
            # 앞부분과 뒷부분 샘플링
            sample_lines = [header] + lines[1:200] + ['...'] + lines[-100:]
            data_csv = '\n'.join(sample_lines)
            logger.info(f"[v13.0] Data truncated: {len(primary_data['data'])} -> {len(data_csv)} chars")

        # v7.0: Data Quality Analysis 컨텍스트 추가
        quality_section = ""
        if data_quality_context:
            quality_section = f"""

## PRE-COMPUTED DATA QUALITY ANALYSIS (v7.0)
The following issues were detected algorithmically. Please verify and add business context.

{data_quality_context}
"""

        # v22.0: Agent Learning - 유사 경험 조회 (few-shot learning)
        few_shot_section = ""
        similar_experiences = self.get_similar_experiences(
            {"task": "data_understanding", "tables": list(all_data.keys())}, limit=3
        )
        if similar_experiences:
            few_shot_examples = []
            for exp in similar_experiences[:2]:  # 최대 2개만 사용
                if hasattr(exp, 'output_data') and exp.output_data:
                    few_shot_examples.append(f"- Domain: {exp.output_data.get('domain_detected', 'unknown')}")
            if few_shot_examples:
                few_shot_section = f"""

## PREVIOUS SIMILAR ANALYSES (v22.0 Learning)
{chr(10).join(few_shot_examples)}
"""

        instruction = f"""Analyze this dataset directly and understand its BUSINESS CONTEXT.

## DATASET: {primary_table}
- Total rows: {primary_data['row_count']}
- Columns: {primary_data['columns']}

## FULL DATA (CSV format):
```csv
{data_csv}
```
{quality_section}{precomputed_stats_section}{few_shot_section}
## YOUR ANALYSIS TASKS:

1. **DOMAIN IDENTIFICATION** (v7.9 확장): What business domain is this data from?
   - Look at column names AND actual values
   - DOMAIN OPTIONS:
     * marketing / digital_marketing: campaigns, leads, conversions, ads, email, UTM tracking
     * e_commerce / retail: orders, products, customers, inventory
     * advertising: ad campaigns, impressions, clicks, CTR, ad performance
     * finance: transactions, accounts, payments, invoices
     * healthcare: patients, diagnoses, treatments
     * aerospace / aviation: flights, aircraft, passengers
     * logistics / supply_chain: shipments, warehouses, delivery
   - IMPORTANT: If data contains campaigns, leads, conversions → likely "marketing" domain

2. **COLUMN ANALYSIS** (CRITICAL - analyze each column's BUSINESS ROLE):
   - What it represents (based on actual values, not just name)
   - **business_role**: identifier | required_input | optional_input | metric | dimension | derived | timestamp
   - Data type (categorical, numeric, date, text)
   - Is it required for business operations? Or optional (like notes/remarks)?

3. **DATA PATTERNS**: Mathematical relationships between columns
   - Do any columns sum to 1 or 100?
   - Are any columns identical or derived from others?

4. **STATUS/ENUM VALUE ANALYSIS** (CRITICAL - v7.0 NEW!):
   For each status/state/enum column, identify:
   - **Normal values**: Values indicating normal operation (ACTIVE, ON_TIME, COMPLETED, etc.)
   - **Problem values**: Values indicating issues (CANCELLED, FAILED, DELAYED, GROUNDED, SUSPENDED, ERROR, etc.)
   - **Problem ratio**: What percentage of records have problem values?
   - **Business impact**: What does each problem status mean operationally?

5. **BOOLEAN FLAG ANALYSIS** (v7.0 NEW!):
   For each boolean/flag column (is_*, has_*, *_flag, *_reported):
   - What does TRUE mean in business terms?
   - What is the TRUE ratio?
   - Is a high TRUE ratio concerning? (e.g., damage_reported=TRUE being high is bad)

6. **OPERATIONAL INSIGHTS** (v7.0 CRITICAL!):
   Go beyond null ratios - provide BUSINESS METRICS:
   - **Delay rates**: How often are things delayed?
   - **Cancellation rates**: How often are things cancelled?
   - **Error/Failure rates**: How often do errors occur?
   - **Damage/Issue rates**: How often are problems reported?
   - **Maintenance status**: What's the maintenance backlog?
   ⚠️ CRITICAL (v22.1): For ALL aggregate metrics (SUM, MEAN, COUNT, RATIO), you MUST use the
   values from the PRE-COMPUTED AGGREGATE STATISTICS section above. Do NOT recompute from the data.
   The pre-computed stats are calculated from the FULL dataset using pandas and are authoritative.

7. **SEMANTIC DATA QUALITY** (IMPORTANT - only report MEANINGFUL issues):
   - ❌ DO NOT report nulls in optional fields (비고, memo, notes, remarks, comments, description)
   - ✅ DO report nulls in required fields (IDs, dates, amounts, names)
   - ✅ DO report business logic violations (negative prices, future dates for past events)
   - ✅ DO report data integrity issues (duplicate keys, referential integrity)
   - ✅ DO report high rates of problem statuses as OPERATIONAL concerns

8. **TABLE STRUCTURE**: What entity does this table represent?
   - Master data, transaction data, or mapping table?
   - Potential relationships to other entities

9. **RECOMMENDED ANALYSIS**: Key dimensions and metrics for analysis

Respond with JSON:
```json
{{
    "domain": {{
        "detected": "marketing|digital_marketing|e_commerce|retail|advertising|finance|healthcare|aerospace|aviation|logistics|supply_chain|other",
        "secondary_domain": "<optional secondary domain>",
        "confidence": 0.0-1.0,
        "reasoning": "<why this domain>",
        "keyword_evidence": ["campaign", "lead", "conversion", "..."]
    }},
    "table_structure": {{
        "entity_type": "<what entity this table represents>",
        "table_type": "master|transaction|mapping|aggregation",
        "description": "<business purpose of this table>"
    }},
    "columns": [
        {{
            "name": "<column_name>",
            "semantic_meaning": "<what it represents>",
            "business_role": "identifier|required_input|optional_input|metric|dimension|derived|timestamp|status",
            "data_type": "categorical|numeric|date|text|boolean|status_enum",
            "is_key_dimension": true/false,
            "is_key_metric": true/false,
            "nullable_by_design": true/false,
            "sample_values": ["<val1>", "<val2>", ...],
            "notes": "<any observations>"
        }}
    ],
    "status_analysis": [
        {{
            "table": "<table_name>",
            "column": "<status_column_name>",
            "normal_values": ["ACTIVE", "ON_TIME", "COMPLETED"],
            "problem_values": ["CANCELLED", "DELAYED", "GROUNDED"],
            "problem_ratio": 0.15,
            "business_impact": "<what problem values mean for operations>",
            "recommended_action": "<what to do about it>"
        }}
    ],
    "flag_analysis": [
        {{
            "table": "<table_name>",
            "column": "<flag_column_name>",
            "meaning_when_true": "<what TRUE means>",
            "true_ratio": 0.05,
            "is_concerning": true/false,
            "business_impact": "<why this ratio matters>"
        }}
    ],
    "operational_insights": {{
        "delay_rate": {{"value": 0.12, "description": "12% of flights are delayed"}},
        "cancellation_rate": {{"value": 0.03, "description": "3% of flights are cancelled"}},
        "error_rate": {{"value": 0.01, "description": "1% error rate in processing"}},
        "maintenance_backlog": {{"value": 5, "description": "5 aircraft in maintenance"}},
        "key_findings": [
            "<finding 1 about operational health>",
            "<finding 2 about performance metrics>"
        ]
    }},
    "data_patterns": [
        {{
            "pattern_type": "mathematical_constraint|duplicate_column|derived_column|correlation",
            "columns_involved": ["<col1>", "<col2>"],
            "description": "<what the pattern is>",
            "implication": "<what this means for analysis>"
        }}
    ],
    "data_quality_issues": [
        {{
            "issue_type": "missing_required|referential_integrity|business_logic|status_anomaly|high_error_rate",
            "columns": ["<affected_columns>"],
            "description": "<description>",
            "severity": "critical|high|medium|low",
            "business_impact": "<why this matters for business>",
            "affected_records": 150
        }}
    ],
    "recommended_analysis": {{
        "key_dimensions": ["<columns to group by>"],
        "key_metrics": ["<columns to measure>"],
        "suggested_analyses": [
            "<analysis 1>",
            "<analysis 2>"
        ],
        "data_story": "<what story can this data tell>",
        "kpis_to_track": ["<KPI 1>", "<KPI 2>"]
    }}
}}
```"""

        response = await self.call_llm(instruction, max_tokens=8000)
        analysis = parse_llm_json(response, default={})

        # parse_llm_json이 문자열을 반환할 수 있으므로 방어
        if not isinstance(analysis, dict):
            logger.warning(f"[v13.0] LLM returned non-dict analysis: {type(analysis)}, using empty dict")
            analysis = {}

        # LLM이 expected dict/list 필드를 string으로 반환할 수 있으므로 타입 강제
        def _safe_dict(val, fallback=None):
            """dict가 아니면 빈 dict 또는 fallback 반환"""
            if isinstance(val, dict):
                return val
            return fallback if fallback is not None else {}

        def _safe_list(val):
            """list가 아니면 빈 list 반환"""
            return val if isinstance(val, list) else []

        self._report_progress(0.8, "Processing LLM analysis results")

        # 4. 분석 결과를 context에 저장
        domain_raw = analysis.get("domain", {})
        if isinstance(domain_raw, str):
            domain_info = {"detected": domain_raw, "confidence": 0.5, "reasoning": "LLM returned string"}
        else:
            domain_info = _safe_dict(domain_raw)
        detected_domain = domain_info.get("detected", "general")

        # Evidence 기록
        self.record_evidence(
            finding=f"데이터 도메인: {detected_domain} (신뢰도: {domain_info.get('confidence', 0):.0%})",
            reasoning=domain_info.get("reasoning", "LLM 분석 기반"),
            conclusion=f"이 데이터는 {detected_domain} 도메인의 데이터입니다.",
            data_references=[primary_table],
            metrics={"confidence": domain_info.get("confidence", 0)},
            confidence=domain_info.get("confidence", 0.5),
            topics=[detected_domain, "domain_detection", "data_understanding"],
        )

        # 데이터 패턴 기록
        patterns = _safe_list(analysis.get("data_patterns", []))
        for pattern in patterns:
            if not isinstance(pattern, dict):
                continue
            self.record_evidence(
                finding=f"데이터 패턴 발견: {pattern.get('pattern_type')}",
                reasoning=pattern.get("description", ""),
                conclusion=pattern.get("implication", ""),
                data_references=pattern.get("columns_involved", []),
                metrics={},
                confidence=0.9,
                topics=["data_pattern", pattern.get("pattern_type", "")],
            )

        # 컬럼 분석 저장
        raw_columns = _safe_list(analysis.get("columns", []))
        column_analysis = {}
        for col in raw_columns:
            if isinstance(col, dict) and "name" in col:
                column_analysis[col["name"]] = col

        # v14.0: table_structure 추출
        table_structure = _safe_dict(analysis.get("table_structure", {}))

        # v7.0: Status/Flag 분석 추출
        status_analysis = _safe_list(analysis.get("status_analysis", []))
        flag_analysis = _safe_list(analysis.get("flag_analysis", []))
        operational_insights = _safe_dict(analysis.get("operational_insights", {}))

        # v7.0: Status 분석 결과를 Evidence로 기록
        for status in status_analysis:
            if not isinstance(status, dict):
                continue
            if status.get("problem_ratio", 0) > 0.05:  # 5% 이상 문제 비율
                self.record_evidence(
                    finding=f"운영 이슈 발견: {status.get('table')}.{status.get('column')} - 문제 비율 {status.get('problem_ratio', 0):.1%}",
                    reasoning=f"문제 값: {status.get('problem_values', [])}. 비즈니스 영향: {status.get('business_impact', 'N/A')}",
                    conclusion=status.get("recommended_action", "검토 필요"),
                    data_references=[f"{status.get('table')}.{status.get('column')}"],
                    metrics={"problem_ratio": status.get("problem_ratio", 0)},
                    confidence=0.9,
                    topics=["status_analysis", "operational_issue", status.get("table", "")],
                )

        # v7.0: Flag 분석 결과를 Evidence로 기록
        for flag in flag_analysis:
            if not isinstance(flag, dict):
                continue
            if flag.get("is_concerning", False):
                self.record_evidence(
                    finding=f"우려되는 플래그: {flag.get('table')}.{flag.get('column')} - TRUE 비율 {flag.get('true_ratio', 0):.1%}",
                    reasoning=f"TRUE 의미: {flag.get('meaning_when_true', 'N/A')}. 비즈니스 영향: {flag.get('business_impact', 'N/A')}",
                    conclusion="모니터링 및 조사 필요",
                    data_references=[f"{flag.get('table')}.{flag.get('column')}"],
                    metrics={"true_ratio": flag.get("true_ratio", 0)},
                    confidence=0.85,
                    topics=["flag_analysis", "data_quality", flag.get("table", "")],
                )

        quality_issues = _safe_list(analysis.get("data_quality_issues", []))
        recommended_analysis = _safe_dict(analysis.get("recommended_analysis", {}))

        logger.info(f"[v7.0] Data analysis complete: domain={detected_domain}, "
                   f"patterns={len(patterns)}, columns={len(column_analysis)}, "
                   f"quality_issues={len(quality_issues)}, "
                   f"status_analyses={len(status_analysis)}, flag_analyses={len(flag_analysis)}")

        # v22.0: Agent Learning - 경험 기록
        self.record_experience(
            experience_type="entity_mapping",
            input_data={"tables": list(all_data.keys()), "task": todo.description},
            output_data={"domain_detected": detected_domain, "patterns_found": len(patterns)},
            outcome="success",
            confidence=domain_info.get("confidence", 0.7),
        )

        # v22.0: Agent Bus - 도메인 감지 결과 브로드캐스트
        await self.broadcast_discovery(
            discovery_type="domain_detected",
            data={
                "domain": detected_domain,
                "confidence": domain_info.get("confidence", 0),
                "patterns": patterns[:10],  # 상위 10개 패턴만 전달
                "tables": list(all_data.keys()),
            },
        )

        return TodoResult(
            success=True,
            output={
                "domain_detected": detected_domain,
                "domain_confidence": domain_info.get("confidence", 0),
                "patterns_found": len(patterns),
                "columns_analyzed": len(column_analysis),
                "quality_issues_found": len(quality_issues),
                "status_issues_found": len(status_analysis),
                "flag_issues_found": len(flag_analysis),
                "recommended_dimensions": recommended_analysis.get("key_dimensions", []),
            },
            context_updates={
                "data_understanding": analysis,
                "detected_domain": detected_domain,
                "column_semantics": column_analysis,
                "data_patterns": patterns,
                "data_quality_issues": quality_issues,
                "recommended_analysis": recommended_analysis,
                "table_structure": table_structure,
                # v7.0: 새 분석 결과 추가
                "status_analysis": status_analysis,
                "flag_analysis": flag_analysis,
                "operational_insights": operational_insights,
                # v22.1: pandas 사전 계산 통계 (정확한 집계값)
                "precomputed_statistics": precomputed_stats if precomputed_stats else {},
            },
            metadata={
                "full_analysis": analysis,
                "primary_table": primary_table,
                "data_size": primary_data["row_count"],
                # v7.0: 알고리즘 기반 분석 결과 추가
                "v7_quality_report": {
                    "status_count": len(quality_report.status_analyses) if quality_report else 0,
                    "flag_count": len(quality_report.boolean_analyses) if quality_report else 0,
                    "outlier_count": len(quality_report.outlier_analyses) if quality_report else 0,
                } if quality_report else None,
            },
        )
