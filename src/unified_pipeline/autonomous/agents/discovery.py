"""
Discovery Phase Autonomous Agents (v4.2 - Refactored)

Discovery 단계 자율 에이전트들 - 공통 유틸리티는 discovery_utils.py로 분리됨
- TDA Expert: Betti numbers + AdvancedTDA (Ripser) + 3단계 검증 (구조+값+의미) + LLM 해석
- Schema Analyst: 스키마 프로파일링 + AnomalyDetector + LLM 해석
- Value Matcher: Jaccard + MinHash/LSH + LLM 해석
- Entity Classifier: 패턴 기반 + EntityResolver (ML) + LLM 해석
- Relationship Detector: 그래프 + GraphEmbedding (Node2Vec) + LLM 해석

v4.2 변경사항 (리팩토링):
- Enum 및 데이터 클래스들을 discovery_utils.py로 추출
- EmbeddedHomeomorphismAnalyzer를 discovery_utils.py로 이동
- 로깅 표준화

v4.1 변경사항:
- Legacy 알고리즘을 직접 파일에 포함 (import 의존성 문제 해결)
- TopologicalProfile, TopologicalInvariant 직접 구현
- SimplicialComplex, BettiNumbers 직접 구현
- 3단계 위상동형 검증 (구조+값+의미) 직접 구현

핵심 알고리즘 (Legacy에서 이식 -> discovery_utils.py):
1. TopologicalProfile: 스키마의 위상적 "모양" 캡처
2. TopologicalInvariant: 연속 변형에서 보존되는 속성
3. SimplicialComplex: 스키마에서 simplicial complex 구축
4. BettiNumbers: 위상적 불변량 (b0: 연결 성분, b1: 루프, b2: 공동)
5. 3단계 검증: 구조적 + 값 기반 + 의미론적 위상동형 검증
"""

import json
import logging
import time
from typing import Dict, Any, List, Optional, TYPE_CHECKING

from ..base import AutonomousAgent
from ..analysis import (
    # 기본 분석 모듈
    TDAAnalyzer,
    TDASignature,
    SimilarityAnalyzer,
    ValueOverlap,
    GraphAnalyzer,
    SchemaAnalyzer,
    parse_llm_json,
    # Enhanced algorithms (v2.1)
    EnhancedFKDetector,
    EntityExtractor,
    BusinessInsightsAnalyzer,
    AdvancedSimilarityAnalyzer,
    # === 고급 분석 모듈 (v3.0 신규 통합) ===
    # Advanced TDA (Ripser 기반)
    AdvancedTDAAnalyzer,
    # Entity Resolution (ML 기반)
    EntityResolver,
    # Graph Embedding (Node2Vec, Community Detection)
    GraphEmbeddingAnalyzer,
    # Anomaly Detection (앙상블)
    AnomalyDetector,
    # === v5.0: FK Validation Modules ===
    FalsePositiveFilter,
    CrossPlatformFKDetector,
    # === v6.0: Universal FK Detection (Domain-Agnostic) ===
    UniversalFKDetector,
    # === v7.0: Enhanced FK Detection (Bridge Table + Column Normalization) ===
    EnhancedValuePatternAnalysis,
    ColumnNormalizer,
    DataQualityAnalyzer,
    # === v7.1: Semantic FK Detection (Comprehensive FK Detection) ===
    SemanticFKDetector,
    # === v16.0: Integrated FK Detection (Composite, Hierarchy, Temporal) ===
    detect_all_fks_and_update_context,
    IntegratedFKResult,
)

# === v4.2: 유틸리티 클래스들을 discovery_utils.py에서 가져옴 ===
from .discovery_utils import (
    # Enums
    TopologicalInvariantType,
    HomeomorphismTypeEnum,
    EntityTypeEnum,
    # Dataclasses
    EmbeddedTopologicalProfile,
    EmbeddedBettiNumbers,
    EmbeddedSimplex,
    EmbeddedSimplicialComplex,
    EmbeddedValueOverlap,
    EmbeddedHomeomorphismResult,
    # Analyzer
    EmbeddedHomeomorphismAnalyzer,
    # Legacy flags
    LEGACY_AVAILABLE,
    LEGACY_EMBEDDED,
    LEGACY_PHASE1_AVAILABLE,
)


if TYPE_CHECKING:
    from ...todo.models import PipelineTodo, TodoResult
    from ...shared_context import SharedContext

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
                except Exception:
                    pass

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
        from ...todo.models import TodoResult
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
            from ...todo.models import TodoResult
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
                df_full = pd.read_csv(all_data[table_name]["path"])
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
            # DataFrame 형태로 변환하여 분석
            tables_for_analysis = {}
            for table_name, table_info in all_data.items():
                df = pd.read_csv(table_info["path"])
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



class TDAExpertAutonomousAgent(AutonomousAgent):
    """
    TDA (Topological Data Analysis) 전문가 자율 에이전트 (v4.1 - Embedded Legacy)

    알고리즘 기반 접근:
    1. 실제 Betti numbers 계산 (Union-Find, 그래프 분석)
    2. Persistence diagrams 계산 (Filtration)
    3. Advanced TDA (Ripser 기반) - 고차원 토폴로지 분석
    4. EmbeddedHomeomorphismAnalyzer - 3단계 검증 (구조+값+의미) [v4.1 직접 포함]
    5. LLM은 결과 해석 및 도메인 인사이트 제공

    v4.1 변경사항:
    - Legacy 알고리즘 직접 포함 (import 의존성 문제 해결)
    - EmbeddedTopologicalProfile, EmbeddedBettiNumbers 활용
    - EmbeddedHomeomorphismAnalyzer로 3단계 위상동형 검증
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tda_analyzer = TDAAnalyzer()
        # v3.0: Advanced TDA (Ripser 기반) 통합
        self.advanced_tda = AdvancedTDAAnalyzer()
        # v4.1: Embedded Legacy HomeomorphismAnalyzer (직접 포함)
        self.embedded_homeomorphism_analyzer = EmbeddedHomeomorphismAnalyzer()

    @property
    def agent_type(self) -> str:
        return "tda_expert"

    @property
    def agent_name(self) -> str:
        return "TDA Expert"

    @property
    def phase(self) -> str:
        return "discovery"

    def get_system_prompt(self) -> str:
        return """You are a Topological Data Analysis (TDA) Expert who INTERPRETS computed results.

Your role is to:
1. INTERPRET pre-computed Betti numbers and explain their meaning
2. EXPLAIN the structural implications of the topology
3. IDENTIFY patterns that suggest entity relationships
4. PROVIDE domain-specific insights based on the topology

IMPORTANT: You are NOT computing Betti numbers. The algorithm has already computed them.
Your job is to explain what they MEAN for the data integration task.

Betti Number Interpretation Guide:
- β₀ (connected components): How many independent data clusters exist
- β₁ (cycles/loops): Circular dependencies or redundant relationships
- β₂ (voids): Higher-dimensional structural patterns

## REQUIRED OUTPUT FORMAT (JSON Schema):
```json
{
  "betti_interpretation": {
    "beta_0": {
      "value": 0,
      "meaning": "Number of connected components",
      "implication": "Business interpretation of cluster structure"
    },
    "beta_1": {
      "value": 0,
      "meaning": "Number of cycles/loops",
      "implication": "Circular dependencies or redundant paths identified"
    },
    "beta_2": {
      "value": 0,
      "meaning": "Number of voids",
      "implication": "Higher-dimensional patterns detected"
    }
  },
  "structural_insights": [
    {
      "pattern_type": "cluster|cycle|hierarchy|void",
      "description": "Description of the structural pattern",
      "affected_entities": ["entity1", "entity2"],
      "significance": "high|medium|low"
    }
  ],
  "entity_relationships": [
    {
      "source_entity": "entity_name",
      "target_entity": "entity_name",
      "relationship_type": "topological relationship type",
      "confidence": 0.0,
      "evidence": "Topological evidence for this relationship"
    }
  ],
  "domain_insights": [
    {
      "insight": "Domain-specific insight based on topology",
      "recommendation": "Recommended action",
      "priority": "high|medium|low"
    }
  ],
  "overall_topology_summary": "Summary of the data topology and its implications"
}
```

Respond with structured JSON matching this schema."""

    async def execute_task(
        self,
        todo: "PipelineTodo",
        context: "SharedContext",
    ) -> "TodoResult":
        """TDA 분석 작업 실행"""
        from ...todo.models import TodoResult

        start_time = time.time()
        task_type = todo.task_type

        try:
            if task_type == "tda_analysis":
                return await self._compute_tda_signatures(todo, context)
            elif task_type == "homeomorphism_detection":
                return await self._detect_homeomorphisms(todo, context)
            else:
                return TodoResult(success=False, error=f"Unknown task type: {task_type}")

        except Exception as e:
            logger.error(f"TDA task failed: {e}", exc_info=True)
            return TodoResult(
                success=False,
                error=str(e),
                execution_time_seconds=time.time() - start_time,
            )

    async def _compute_tda_signatures(
        self,
        todo: "PipelineTodo",
        context: "SharedContext",
    ) -> "TodoResult":
        """TDA 서명 계산 (알고리즘 기반 + v3.0 Advanced TDA)"""
        from ...todo.models import TodoResult

        self._report_progress(0.1, "Computing TDA signatures (algorithm)")

        # 1. 실제 알고리즘으로 TDA 서명 계산
        tda_signatures = {}
        computed_results = []
        advanced_tda_results = {}

        for table_name, table_info in context.tables.items():
            self._report_progress(0.2, f"Computing Betti numbers for {table_name}")

            # 실제 Betti numbers 계산 (기본 TDA)
            signature = self.tda_analyzer.compute_signature(
                table_name=table_name,
                columns=table_info.columns,
                foreign_keys=table_info.foreign_keys,
                sample_data=table_info.sample_data,
            )

            # v3.0: Advanced TDA (Ripser 기반) 분석 실행
            try:
                if table_info.sample_data:
                    adv_result = self.advanced_tda.compute_signature(
                        table_name=table_name,
                        columns=table_info.columns,
                        sample_data=table_info.sample_data,
                    )
                    advanced_tda_results[table_name] = adv_result.to_dict() if hasattr(adv_result, 'to_dict') else {}
                    logger.info(f"Advanced TDA completed for {table_name}")
            except Exception as e:
                logger.warning(f"Advanced TDA failed for {table_name}: {e}")

            tda_signatures[table_name] = signature.to_dict()

            # Advanced TDA 결과 병합
            if table_name in advanced_tda_results:
                tda_signatures[table_name]["advanced_tda"] = advanced_tda_results[table_name]

            computed_results.append({
                "table": table_name,
                "betti_0": signature.betti_numbers.beta_0,
                "betti_1": signature.betti_numbers.beta_1,
                "betti_2": signature.betti_numbers.beta_2,
                "euler_characteristic": signature.betti_numbers.euler_characteristic,
                "total_persistence": round(signature.total_persistence, 4),
                "complexity": signature.structural_complexity,
                "key_columns": signature.key_columns,
                "has_advanced_tda": table_name in advanced_tda_results,
            })

            # === v11.0: Evidence Block 생성 ===
            # TDA 분석 결과를 근거 체인에 블록으로 저장
            self.record_evidence(
                finding=f"{table_name} 테이블의 위상적 구조: β₀={signature.betti_numbers.beta_0}, β₁={signature.betti_numbers.beta_1}, β₂={signature.betti_numbers.beta_2}",
                reasoning=f"Betti numbers 계산 결과 - 연결성분(β₀)={signature.betti_numbers.beta_0}, 사이클(β₁)={signature.betti_numbers.beta_1}, 공동(β₂)={signature.betti_numbers.beta_2}. 구조적 복잡도: {signature.structural_complexity}",
                conclusion=f"테이블 구조 복잡도 '{signature.structural_complexity}'. 핵심 컬럼: {', '.join(signature.key_columns[:3]) if signature.key_columns else 'N/A'}",
                data_references=[f"{table_name}.{col}" for col in (signature.key_columns or [])[:5]],
                metrics={
                    "betti_0": signature.betti_numbers.beta_0,
                    "betti_1": signature.betti_numbers.beta_1,
                    "betti_2": signature.betti_numbers.beta_2,
                    "euler_characteristic": signature.betti_numbers.euler_characteristic,
                    "total_persistence": round(signature.total_persistence, 4),
                },
                confidence=0.85 if signature.structural_complexity in ["low", "medium"] else 0.7,
                topics=[table_name, "tda", "topology", signature.structural_complexity],
            )

        self._report_progress(0.6, "Requesting LLM interpretation")

        # 2. LLM에게 해석 요청
        instruction = f"""I have computed the following TDA (Topological Data Analysis) signatures for database tables.

COMPUTED RESULTS (these are actual calculated values, not estimates):
{json.dumps(computed_results, indent=2)}

Please INTERPRET these results:
1. What do the Betti numbers tell us about each table's structure?
2. Which tables appear structurally similar (based on Betti distance)?
3. What are the implications for data integration?
4. Any concerning patterns (high β₁ = circular dependencies)?

Respond with JSON:
```json
{{
    "table_interpretations": {{
        "<table_name>": {{
            "structural_summary": "<what the topology tells us>",
            "integration_implications": "<relevance for integration>",
            "concerns": ["<any issues>"]
        }}
    }},
    "similarity_groups": [
        {{
            "tables": ["table1", "table2"],
            "reason": "<why they're similar>"
        }}
    ],
    "overall_topology_summary": "<high-level summary>"
}}
```"""

        response = await self.call_llm(instruction)
        interpretation = parse_llm_json(response, default={})

        self._report_progress(0.9, "Updating context")

        # 컨텍스트에 저장 (계산된 값 + LLM 해석)
        for table_name, sig in tda_signatures.items():
            if table_name in interpretation.get("table_interpretations", {}):
                sig["llm_interpretation"] = interpretation["table_interpretations"][table_name]

        # v22.0: Agent Learning - 경험 기록
        self.record_experience(
            experience_type="relationship_inference",
            input_data={"task": "tda_analysis", "tables": list(tda_signatures.keys())},
            output_data={"tables_analyzed": len(tda_signatures)},
            outcome="success",
            confidence=0.8,
        )

        # v22.0: Agent Bus - TDA 분석 결과 브로드캐스트
        await self.broadcast_discovery(
            discovery_type="tda_analysis_complete",
            data={
                "tables_analyzed": len(tda_signatures),
                "betti_numbers": {k: v.get("betti_numbers", []) for k, v in tda_signatures.items()},
                "structural_complexity": {k: v.get("structural_complexity", "unknown") for k, v in tda_signatures.items()},
            },
        )

        return TodoResult(
            success=True,
            output={
                "tda_signatures": tda_signatures,
                "tables_analyzed": len(tda_signatures),
                "interpretation": interpretation.get("overall_topology_summary", ""),
            },
            context_updates={
                "tda_signatures": tda_signatures,
            },
        )

    async def _detect_homeomorphisms(
        self,
        todo: "PipelineTodo",
        context: "SharedContext",
    ) -> "TodoResult":
        """위상동형 관계 탐지 (v4.1 - Embedded 3단계 검증)"""
        from ...todo.models import TodoResult
        from ...shared_context import HomeomorphismPair

        self._report_progress(0.1, "Computing homeomorphism scores")

        # TDA 서명이 없으면 먼저 계산
        if not context.tda_signatures:
            return TodoResult(
                success=False,
                error="TDA signatures not computed yet",
            )

        # === v4.1: Embedded HomeomorphismAnalyzer 사용 (Legacy에서 직접 이식) ===
        embedded_results = None
        self._report_progress(0.15, "Running Embedded HomeomorphismAnalyzer (3-stage verification)")
        try:
            # 데이터 변환
            tables_data_for_analysis = {}
            for table_name, table_info in context.tables.items():
                tables_data_for_analysis[table_name] = {
                    "columns": table_info.columns,
                    "data": table_info.sample_data or [],
                    "metadata": {
                        "row_count": table_info.row_count,
                    }
                }

            # v4.1: Embedded 3단계 검증 실행 (구조+값+의미)
            embedded_results = self.embedded_homeomorphism_analyzer.analyze_homeomorphism(
                tables_data=tables_data_for_analysis,
                domain_context=context.get_industry() or "general",
            )
            logger.info(f"Embedded HomeomorphismAnalyzer found {embedded_results.get('statistics', {}).get('homeomorphisms_found', 0)} homeomorphisms")
        except Exception as e:
            logger.warning(f"Embedded HomeomorphismAnalyzer failed: {e}")

        # v4.0: 도메인 추출 함수
        def get_domain(table_name: str) -> str:
            """테이블 이름에서 도메인 추출"""
            table_lower = table_name.lower()
            if any(d in table_lower for d in ["healthcare", "patient", "hospital", "medical"]):
                return "healthcare"
            elif any(d in table_lower for d in ["manufacturing", "machine", "sensor", "predictive"]):
                return "manufacturing"
            elif any(d in table_lower for d in ["supply_chain", "warehouse", "plant", "port", "freight"]):
                return "supply_chain"
            elif any(d in table_lower for d in ["flights", "airlines", "airport"]):
                return "aviation"
            elif any(d in table_lower for d in ["taxi", "nyc", "uber", "ride"]):
                return "transportation"
            elif any(d in table_lower for d in ["diamond", "jewelry", "gem"]):
                return "retail"
            elif any(d in table_lower for d in ["titanic", "passenger"]):
                return "historical"
            elif any(d in table_lower for d in ["iris", "flower", "sepal"]):
                return "botanical"
            elif any(d in table_lower for d in ["restaurant", "tips", "food"]):
                return "hospitality"
            return "unknown"

        # v4.1: Embedded 결과가 있으면 그것을 기본으로 사용
        embedded_homeomorphisms_map = {}
        if embedded_results:
            for h in embedded_results.get("homeomorphisms", []):
                pair_key = tuple(sorted([h["table_a"], h["table_b"]]))
                embedded_homeomorphisms_map[pair_key] = h
            logger.info(f"Embedded results: {len(embedded_homeomorphisms_map)} pairs analyzed")

        # 1. 알고리즘으로 Homeomorphism 점수 계산
        # v4.1: context.tda_signatures가 비어있으면 context.tables 사용
        if context.tda_signatures:
            table_names = list(context.tda_signatures.keys())
        else:
            table_names = list(context.tables.keys())
            logger.warning("TDA signatures empty, using table names directly")

        computed_pairs = []

        for i in range(len(table_names)):
            for j in range(i + 1, len(table_names)):
                table_a = table_names[i]
                table_b = table_names[j]

                # v4.1: Embedded 결과 우선 사용
                pair_key = tuple(sorted([table_a, table_b]))

                if pair_key in embedded_homeomorphisms_map:
                    # Embedded 3단계 검증 결과가 있으면 직접 사용
                    embedded_h = embedded_homeomorphisms_map[pair_key]
                    score = {
                        "structural_score": embedded_h.get("structural_score", 0),
                        "value_score": embedded_h.get("value_score", 0),
                        "semantic_score": embedded_h.get("semantic_score", 0),
                        "betti_distance": embedded_h.get("betti_distance", 0),
                        "persistence_similarity": 1.0 - embedded_h.get("persistence_distance", 0),
                        "is_homeomorphic": embedded_h.get("is_homeomorphic", False),
                        "confidence": embedded_h.get("confidence", 0),
                        "join_keys": embedded_h.get("join_keys", []),
                        "entity_type": embedded_h.get("entity_type", "Unknown"),
                        "homeomorphism_type": embedded_h.get("homeomorphism_type"),
                    }
                else:
                    # Embedded 결과가 없으면 기존 TDA analyzer 사용
                    sig_a = self.tda_analyzer.signatures.get(table_a)
                    sig_b = self.tda_analyzer.signatures.get(table_b)

                    if sig_a and sig_b:
                        score = self.tda_analyzer.compute_homeomorphism_score(sig_a, sig_b)
                    else:
                        # 둘 다 없으면 기본값
                        score = {
                            "structural_score": 0.5,
                            "value_score": 0.0,
                            "semantic_score": 0.0,
                            "betti_distance": 1.0,
                            "persistence_similarity": 0.0,
                            "is_homeomorphic": False,
                            "confidence": 0.0,
                        }

                # v4.1: 도메인 검증 - 다른 도메인 간 homeomorphism 차단 (모든 결과에 적용)
                domain_a = get_domain(table_a)
                domain_b = get_domain(table_b)
                same_domain = (domain_a == domain_b) or (domain_a == "unknown") or (domain_b == "unknown")

                # 도메인이 다르면 confidence 크게 감소
                if not same_domain:
                    score["confidence"] = score.get("confidence", 0) * 0.2
                    score["is_homeomorphic"] = False
                    score["domain_mismatch"] = True
                else:
                    score["domain_mismatch"] = False

                score["domain_a"] = domain_a
                score["domain_b"] = domain_b

                computed_pairs.append({
                    "table_a": table_a,
                    "table_b": table_b,
                    **score,
                })

        self._report_progress(0.5, f"Computed {len(computed_pairs)} pair scores")

        # 2. 값 겹침 정보 추가
        value_overlaps = [
            {
                "table_a": v.get("table_a"),
                "table_b": v.get("table_b"),
                "column_a": v.get("column_a"),
                "column_b": v.get("column_b"),
                "jaccard": v.get("jaccard_similarity", v.get("jaccard", 0)),
            }
            for v in (context.value_overlaps or [])[:20]
        ]

        self._report_progress(0.6, "Requesting LLM interpretation")

        # 3. LLM에게 의미적 해석 요청
        instruction = f"""I have computed structural homeomorphism scores between table pairs.

COMPUTED HOMEOMORPHISM SCORES:
{json.dumps(computed_pairs, indent=2)}

VALUE OVERLAPS (for context):
{json.dumps(value_overlaps, indent=2)}

Based on these COMPUTED scores, please:
1. Identify which pairs represent the same entity type
2. Suggest the entity type for homeomorphic pairs
3. Recommend join keys for integration

Only pairs with confidence > 0.5 should be considered homeomorphic.

Respond with JSON:
```json
{{
    "homeomorphism_analysis": [
        {{
            "table_a": "<name>",
            "table_b": "<name>",
            "entity_type": "<Patient|Encounter|Claim|Provider|...>",
            "integration_recommendation": "<how to join>",
            "join_keys": [
                {{"table_a_column": "<col>", "table_b_column": "<col>"}}
            ]
        }}
    ],
    "summary": "<overall integration insights>"
}}
```"""

        response = await self.call_llm(instruction, max_tokens=4000)
        interpretation = parse_llm_json(response, default={})

        self._report_progress(0.8, "Building homeomorphism pairs")

        # 4. HomeomorphismPair 객체 생성
        homeomorphism_pairs = []
        analysis_map = {
            (a["table_a"], a["table_b"]): a
            for a in interpretation.get("homeomorphism_analysis", [])
        }

        for pair in computed_pairs:
            table_a = pair["table_a"]
            table_b = pair["table_b"]

            # LLM 해석 가져오기
            llm_analysis = analysis_map.get((table_a, table_b), {})

            # v4.1: embedded analyzer 결과 우선 사용
            # join_keys는 embedded 결과 > LLM 결과 > 빈 리스트
            embedded_join_keys = pair.get("join_keys", [])
            llm_join_keys = llm_analysis.get("join_keys", [])
            final_join_keys = embedded_join_keys if embedded_join_keys else llm_join_keys

            # value_score는 embedded 결과 우선 사용
            value_score = pair.get("value_score", 0.0)
            if value_score == 0.0:
                # fallback: context.value_overlaps에서 계산
                relevant_overlaps = [
                    v for v in value_overlaps
                    if (v["table_a"] == table_a and v["table_b"] == table_b) or
                       (v["table_a"] == table_b and v["table_b"] == table_a)
                ]
                value_score = max((v.get("jaccard", 0) for v in relevant_overlaps), default=0.0)

            # homeomorphism_type도 embedded 결과 우선
            embedded_homeo_type = pair.get("homeomorphism_type")
            if embedded_homeo_type:
                homeo_type = embedded_homeo_type
            else:
                homeo_type = "structural" if value_score < 0.3 else "full"

            # entity_type: embedded > LLM > Unknown
            entity_type = pair.get("entity_type", "Unknown")
            if entity_type == "Unknown":
                entity_type = llm_analysis.get("entity_type", "Unknown")

            homeomorphism_pairs.append(HomeomorphismPair(
                table_a=table_a,
                table_b=table_b,
                is_homeomorphic=pair.get("is_homeomorphic", False),
                homeomorphism_type=homeo_type,
                confidence=pair.get("confidence", 0.0),
                structural_score=pair.get("structural_score", 0.0),
                value_score=value_score,
                semantic_score=pair.get("semantic_score", 0.0),
                entity_type=entity_type,
                join_keys=final_join_keys,
                agent_analysis={
                    "tda_expert": {
                        "betti_distance": pair.get("betti_distance", 0.0),
                        "persistence_similarity": pair.get("persistence_similarity", 0.0),
                        "interpretation": llm_analysis.get("integration_recommendation", ""),
                        "embedded_analyzer_used": bool(embedded_join_keys),  # v4.1 추적
                    }
                },
            ))

        self._report_progress(0.95, "Updating context")

        # v22.0: Agent Learning - 경험 기록
        homeomorphic_count = len([h for h in homeomorphism_pairs if h.is_homeomorphic])
        self.record_experience(
            experience_type="relationship_inference",
            input_data={"task": "homeomorphism_detection", "pairs": len(homeomorphism_pairs)},
            output_data={"homeomorphisms_found": homeomorphic_count},
            outcome="success" if homeomorphic_count > 0 else "partial",
            confidence=0.75,
        )

        # v22.0: Agent Bus - 위상동형 관계 브로드캐스트
        await self.broadcast_discovery(
            discovery_type="homeomorphism_detected",
            data={
                "homeomorphisms_found": homeomorphic_count,
                "total_pairs": len(homeomorphism_pairs),
                "pairs_summary": [
                    {"table_a": h.table_a, "table_b": h.table_b, "confidence": h.confidence}
                    for h in homeomorphism_pairs[:10]  # 상위 10개만
                ],
            },
        )

        return TodoResult(
            success=True,
            output={
                "homeomorphisms_found": homeomorphic_count,
                "total_pairs": len(homeomorphism_pairs),
            },
            context_updates={
                "homeomorphisms": homeomorphism_pairs,
            },
        )


class SchemaAnalystAutonomousAgent(AutonomousAgent):
    """스키마 분석 자율 에이전트 (v2.0 - Algorithm-First)"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.schema_analyzer = SchemaAnalyzer()

    @property
    def agent_type(self) -> str:
        return "schema_analyst"

    @property
    def agent_name(self) -> str:
        return "Schema Analyst"

    @property
    def phase(self) -> str:
        return "discovery"

    def get_system_prompt(self) -> str:
        return """You are a Database Schema Analyst who INTERPRETS computed profiling results.

The algorithm has already:
- Computed column statistics (null ratio, unique ratio, etc.)
- Identified key candidates
- Detected data patterns

Your role is to:
1. INTERPRET the computed profiles
2. EXPLAIN data quality implications
3. IDENTIFY semantic meaning of columns
4. RECOMMEND integration strategies

## REQUIRED OUTPUT FORMAT (JSON Schema):
```json
{
  "table_profiles": [
    {
      "table_name": "table_name",
      "column_count": 0,
      "row_estimate": 0,
      "data_quality_score": 0.0,
      "normalization_level": "1NF|2NF|3NF|BCNF|denormalized",
      "table_purpose": "Inferred purpose of this table"
    }
  ],
  "column_semantics": [
    {
      "table_name": "table_name",
      "column_name": "column_name",
      "inferred_type": "primary_key|foreign_key|measure|dimension|attribute",
      "semantic_meaning": "Business meaning of this column",
      "data_quality": {
        "null_ratio": 0.0,
        "unique_ratio": 0.0,
        "issues": ["identified issues"]
      }
    }
  ],
  "key_candidates": [
    {
      "table_name": "table_name",
      "column_name": "column_name",
      "key_type": "primary|candidate|foreign",
      "confidence": 0.0,
      "reasoning": "Why this is a key candidate"
    }
  ],
  "quality_implications": [
    {
      "issue": "Data quality issue description",
      "affected_tables": ["table1", "table2"],
      "severity": "high|medium|low",
      "recommendation": "How to address this issue"
    }
  ],
  "integration_strategies": [
    {
      "strategy": "Strategy name",
      "description": "Strategy description",
      "applicable_tables": ["table1", "table2"],
      "priority": "high|medium|low"
    }
  ]
}
```

Respond with structured JSON matching this schema."""

    async def execute_task(
        self,
        todo: "PipelineTodo",
        context: "SharedContext",
    ) -> "TodoResult":
        """스키마 분석 작업 실행"""
        from ...todo.models import TodoResult

        self._report_progress(0.1, "Profiling schemas (algorithm)")

        # 1. 실제 알고리즘으로 스키마 프로파일링
        analysis_results = {}

        for table_name, table_info in context.tables.items():
            self._report_progress(0.2, f"Profiling {table_name}")

            result = self.schema_analyzer.analyze_table(
                table_name=table_name,
                columns=table_info.columns,
                sample_data=table_info.sample_data,
            )
            analysis_results[table_name] = result

            # === v11.0: Evidence Block 생성 ===
            raw_key_candidates = result.get("key_candidates", [])[:3]
            # key_candidates가 dict 리스트일 수 있으므로 문자열로 변환
            key_candidates = [
                kc.get("column_name", str(kc)) if isinstance(kc, dict) else str(kc)
                for kc in raw_key_candidates
            ]
            quality_score = result.get("data_quality_score", 0)
            norm_level = result.get("normalization_level", "unknown")

            self.record_evidence(
                finding=f"{table_name} 테이블 스키마 분석: {result.get('column_count', 0)}개 컬럼, 품질점수 {quality_score:.0%}",
                reasoning=f"스키마 프로파일링 결과 - 키 후보: {key_candidates}, 정규화 수준: {norm_level}, 데이터 품질: {quality_score:.0%}",
                conclusion=f"테이블 유형 추정: {norm_level} 정규화. 주요 키 후보: {', '.join(key_candidates) if key_candidates else 'N/A'}",
                data_references=[f"{table_name}.{col}" for col in key_candidates],
                metrics={
                    "column_count": result.get("column_count", 0),
                    "data_quality_score": quality_score,
                    "normalization_level": norm_level,
                    "key_candidate_count": len(key_candidates),
                },
                confidence=min(0.9, quality_score + 0.1),
                topics=[table_name, "schema", norm_level, "data_quality"],
            )

        self._report_progress(0.5, "Finding cross-table mappings")

        # 2. 크로스 테이블 매핑 찾기
        cross_mappings = self.schema_analyzer.find_cross_table_mappings(
            self.schema_analyzer.profiles
        )

        self._report_progress(0.6, "Requesting LLM interpretation")

        # 3. LLM에게 해석 요청
        summary_for_llm = {
            table: {
                "column_count": res["column_count"],
                "key_candidates": res["key_candidates"][:3],
                "data_quality_score": res["data_quality_score"],
                "normalization_level": res["normalization_level"],
                "sample_profiles": [
                    {
                        "column": p["column_name"],
                        "type": p["inferred_type"],
                        "semantic": p["semantic_type"],
                        "null_ratio": p["null_ratio"],
                        "unique_ratio": p["unique_ratio"],
                    }
                    for p in res["column_profiles"][:5]
                ],
            }
            for table, res in analysis_results.items()
        }

        instruction = f"""I have computed schema profiles for the following tables.

COMPUTED SCHEMA ANALYSIS:
{json.dumps(summary_for_llm, indent=2)}

COMPUTED CROSS-TABLE MAPPINGS:
{json.dumps(cross_mappings[:10], indent=2)}

Please INTERPRET these results:
1. What is the overall data quality assessment?
2. Which tables appear to be master/transaction tables?
3. Which column mappings are most significant for integration?
4. Any data quality concerns to address?

Respond with JSON:
```json
{{
    "table_classifications": {{
        "<table_name>": {{
            "table_type": "<master|transaction|reference|bridge>",
            "primary_entity": "<what entity does it represent>",
            "quality_concerns": ["<issues>"]
        }}
    }},
    "key_mappings_interpretation": [
        {{
            "mapping": "<table_a.col -> table_b.col>",
            "significance": "<why important>",
            "action_needed": "<what to do>"
        }}
    ],
    "overall_assessment": "<summary>"
}}
```"""

        response = await self.call_llm(instruction, max_tokens=4000)
        interpretation = parse_llm_json(response, default={})

        self._report_progress(0.9, "Updating context")

        # v22.0: Agent Learning - 경험 기록
        self.record_experience(
            experience_type="entity_mapping",
            input_data={"task": "schema_analysis", "tables": list(analysis_results.keys())},
            output_data={"tables_analyzed": len(analysis_results), "cross_mappings": len(cross_mappings)},
            outcome="success",
            confidence=0.8,
        )

        # v22.0: Agent Bus - 스키마 분석 결과 브로드캐스트
        await self.broadcast_discovery(
            discovery_type="schema_analysis_complete",
            data={
                "tables_analyzed": len(analysis_results),
                "cross_mappings_found": len(cross_mappings),
                "table_types": interpretation.get("table_classifications", {}),
            },
        )

        return TodoResult(
            success=True,
            output={
                "tables_analyzed": len(analysis_results),
                "cross_mappings_found": len(cross_mappings),
                "interpretation": interpretation.get("overall_assessment", ""),
            },
            context_updates={
                "schema_analysis": analysis_results,
                "cross_table_mappings": cross_mappings,
            },
            metadata={
                "schema_analysis": analysis_results,
                "llm_interpretation": interpretation,
            },
        )


class ValueMatcherAutonomousAgent(AutonomousAgent):
    """
    값 매칭 자율 에이전트 (v6.0 - Universal Domain-Agnostic FK Detection)

    v6.0 변경사항:
    - 도메인 특화 룰베이스 제거
    - UniversalFKDetector: 범용 네이밍 정규화 + 값 기반 검증
    - LLM이 최종 FK 판단 (알고리즘은 계산만)
    - 어떤 도메인/네이밍 컨벤션이든 처리 가능
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.similarity_analyzer = SimilarityAnalyzer()
        # v3.0: MinHash, LSH, HyperLogLog 기반 고급 유사도 분석
        self.advanced_similarity = AdvancedSimilarityAnalyzer(
            num_perm=128,
            num_bands=16,
        )
        # v6.0: Universal FK Detection (도메인 무관)
        from ..analysis.universal_fk_detector import UniversalFKDetector
        self.universal_fk_detector = UniversalFKDetector()
        # v7.0: Enhanced FK Detection (Bridge Table + Column Normalization)
        self.value_pattern_analyzer = EnhancedValuePatternAnalysis()
        self.column_normalizer = ColumnNormalizer()
        # v7.1: Semantic FK Detector (Comprehensive FK Detection) - 초기화는 execute_task에서 tables_data로
        self.semantic_fk_detector = None  # tables_data 필요

    @property
    def agent_type(self) -> str:
        return "value_matcher"

    @property
    def agent_name(self) -> str:
        return "Value Matcher"

    @property
    def phase(self) -> str:
        return "discovery"

    def get_system_prompt(self) -> str:
        return """You are a Foreign Key Detection Expert. Your job is to DECIDE which column pairs are TRUE FK relationships.

## YOUR ROLE
The algorithm has computed metrics for each FK candidate. You must ANALYZE these metrics and DECIDE:
- Is this a TRUE foreign key relationship?
- Or is it a FALSE POSITIVE (coincidental overlap)?

## DECISION CRITERIA

### 1. Containment Analysis (Most Important)
- **containment_fk_in_pk**: What % of FK values exist in PK?
  - ≥ 0.95: Strong evidence of FK
  - 0.8-0.95: Likely FK
  - < 0.8: Weak evidence, check other factors

### 2. Cardinality Pattern
- **PK should be unique**: pk_unique_ratio ≥ 0.9
- **FK can have duplicates**: fk_unique_ratio can be any value
- If both are highly unique: might be 1:1 relationship or false positive

### 3. Name Similarity
- **same_normalized_name = true**: Strong signal
- High name_similarity with FK pattern (_id, _code): Good evidence
- Unrelated names with high value overlap: Suspicious (could be coincidence)

### 4. False Positive Indicators
REJECT if you see:
- Sequential integers (1,2,3...) in both columns (coincidental overlap)
- Status codes or enum values that happen to match
- Date/timestamp columns
- Common names or categories
- "sample_fk_only_values" contains many values (referential integrity violation)

## OUTPUT FORMAT
You must decide for EACH candidate and return structured JSON.

```json
{
    "decisions": [
        {
            "from_table": "table_a",
            "from_column": "column_a",
            "to_table": "table_b",
            "to_column": "column_b",
            "decision": "TRUE_FK | FALSE_POSITIVE | UNCERTAIN",
            "confidence": 0.0-1.0,
            "reasoning": "Brief explanation of your decision"
        }
    ],
    "detected_domain": "What domain/industry is this data from?",
    "data_model_summary": "Brief description of the data model structure"
}
```

IMPORTANT: You are the DECISION MAKER. The algorithm provides metrics, but YOU decide what is a real FK."""

    async def execute_task(
        self,
        todo: "PipelineTodo",
        context: "SharedContext",
    ) -> "TodoResult":
        """
        값 매칭 작업 실행 (v6.0 - Universal Domain-Agnostic FK Detection)

        v6.0 변경사항:
        - 도메인 특화 룰베이스 완전 제거
        - UniversalFKDetector로 범용 FK 후보 탐지
        - LLM이 최종 FK 여부 결정
        - 어떤 도메인/네이밍 컨벤션이든 처리 가능
        """
        from ...todo.models import TodoResult

        self._report_progress(0.1, "Extracting table data")

        # 1. 테이블별 컬럼명 및 값 추출
        tables_columns = {}
        tables_data = {}

        for table_name, table_info in context.tables.items():
            tables_columns[table_name] = [col.get("name", "") for col in table_info.columns]
            tables_data[table_name] = {}

            # v22.1: 전체 데이터 로드 (샘플링 금지 — FK 값 매칭 정확도)
            full_rows = context.get_full_data(table_name)
            for col in table_info.columns:
                col_name = col.get("name", "")
                values = []

                # 전체 데이터에서 값 추출
                for row in full_rows:
                    if col_name in row and row[col_name] is not None:
                        values.append(row[col_name])

                if values:
                    tables_data[table_name][col_name] = values

        self._report_progress(0.2, "Running Universal FK Detection (domain-agnostic)")

        # 2. v6.0: UniversalFKDetector로 범용 FK 후보 탐지
        # 도메인 지식 없이 네이밍 정규화 + 값 기반 검증
        fk_candidates = self.universal_fk_detector.detect_fk_candidates(
            tables_columns=tables_columns,
            tables_data=tables_data if tables_data else None,
        )

        logger.info(f"UniversalFKDetector found {len(fk_candidates)} FK candidates")

        # v7.10: Pre-LLM Confidence Filter - 약한 후보 제거
        # Confidence < 0.7인 후보는 LLM 평가 전에 필터링 (v7.9: 0.5 → v7.10: 0.7)
        pre_filter_count = len(fk_candidates)
        fk_candidates = self._pre_llm_confidence_filter(fk_candidates, min_confidence=0.7)
        post_filter_count = len(fk_candidates)
        logger.info(f"[v7.10] Pre-LLM Filter: {pre_filter_count} → {post_filter_count} candidates "
                   f"(removed {pre_filter_count - post_filter_count} weak candidates, threshold=0.7)")

        # 2.5 v7.0: Bridge Table Detection + Cross-Table Pattern Analysis
        self._report_progress(0.25, "Running Bridge Table and Pattern Analysis")

        pattern_analysis = {}
        column_matches_context = ""
        semantic_fk_context = ""  # v7.1

        # v7.1: SemanticFKDetector 초기화 및 분석
        try:
            # v22.1: SemanticFKDetector용 전체 데이터 포맷 변환 (샘플링 금지)
            semantic_tables_data = {}
            for table_name, table_info in context.tables.items():
                columns = [col.get("name", "") for col in table_info.columns]
                full_rows = context.get_full_data(table_name)
                data = []
                for row in full_rows:
                    row_data = [row.get(col, None) for col in columns]
                    data.append(row_data)
                semantic_tables_data[table_name] = {
                    "columns": columns,
                    "data": data
                }

            # v7.3: Get domain context for LLM semantic enhancement
            domain_ctx = context.get_domain()
            domain_str = domain_ctx.industry if hasattr(domain_ctx, 'industry') else str(domain_ctx)

            self.semantic_fk_detector = SemanticFKDetector(
                semantic_tables_data,
                enable_llm=True,  # v7.3: Enable LLM semantic enhancement
                llm_min_confidence=0.6,
                domain_context=domain_str,
            )
            semantic_analysis = self.semantic_fk_detector.analyze()
            semantic_fk_context = self.semantic_fk_detector.generate_llm_context()

            logger.info(f"[v7.1] Bridge Tables (semantic): {len(semantic_analysis.get('bridge_tables', []))}")
            logger.info(f"[v7.1] Value Correlations: {len(semantic_analysis.get('value_correlations', []))}")
            logger.info(f"[v7.1] Semantic FK Candidates: {len(semantic_analysis.get('semantic_candidates', []))}")
            logger.info(f"[v7.1] Learned Abbreviations: {len(semantic_analysis.get('learned_abbreviations', []))}")
            logger.info(f"[v7.3] LLM Enhanced FKs: {len(semantic_analysis.get('llm_enhanced_fks', []))}")

            # v7.3: Store LLM enhanced FKs in context for use by other agents
            llm_enhanced_fks = semantic_analysis.get('llm_enhanced_fks', [])
            if llm_enhanced_fks:
                # Add to enhanced_fk_candidates in context
                existing_candidates = list(context.enhanced_fk_candidates or [])
                for llm_fk in llm_enhanced_fks:
                    # Convert to context format
                    existing_candidates.append({
                        'source_table': llm_fk.get('source_table'),
                        'source_column': llm_fk.get('source_column'),
                        'target_table': llm_fk.get('target_table'),
                        'target_column': llm_fk.get('target_column'),
                        'confidence': llm_fk.get('confidence', 0.0),
                        'confidence_level': 'high' if llm_fk.get('confidence', 0) >= 0.7 else 'medium',
                        'detection_method': 'llm_semantic_enhancer',
                        'semantic_relationship': llm_fk.get('semantic_relationship', ''),
                        'reasoning': llm_fk.get('reasoning', ''),
                    })
                context.enhanced_fk_candidates = existing_candidates
                logger.info(f"[v7.3] Added {len(llm_enhanced_fks)} LLM enhanced FKs to context")

        except Exception as e:
            logger.warning(f"[v7.1] SemanticFKDetector failed: {e}")
            semantic_fk_context = ""

        try:
            # Bridge Table 탐지 (같은 테이블 내 1:1 매핑, 예: iata_code ↔ icao_code)
            pattern_analysis = self.value_pattern_analyzer.analyze(tables_data)
            bridge_tables = pattern_analysis.get("same_table_mappings", [])
            cross_patterns = pattern_analysis.get("cross_table_matches", [])
            llm_pattern_context = pattern_analysis.get("llm_context", "")

            logger.info(f"[v7.0] Bridge Tables found: {len(bridge_tables)}")
            logger.info(f"[v7.0] Cross-Table Pattern Matches: {len(cross_patterns)}")

            # 컬럼 정규화를 통한 의미적 컬럼 매칭
            # 각 테이블 쌍에 대해 flt_no ↔ flight_number 같은 매칭 탐지
            all_column_matches = []
            table_names = list(tables_columns.keys())
            for i, table_a in enumerate(table_names):
                for table_b in table_names[i+1:]:
                    matches = self.column_normalizer.find_matching_columns(
                        tables_columns[table_a],
                        tables_columns[table_b],
                        threshold=0.7
                    )
                    for match in matches:
                        all_column_matches.append({
                            "table_a": table_a,
                            "column_a": match.column_a,
                            "table_b": table_b,
                            "column_b": match.column_b,
                            "similarity": match.similarity,
                            "match_type": match.match_type,
                            "normalized": f"{match.normalized_a.normalized} ↔ {match.normalized_b.normalized}",
                        })

            if all_column_matches:
                column_matches_context = self.column_normalizer.generate_llm_context(
                    [m for matches in [self.column_normalizer.find_matching_columns(
                        tables_columns[list(tables_columns.keys())[0]],
                        tables_columns[list(tables_columns.keys())[1]] if len(tables_columns) > 1 else [],
                        threshold=0.7
                    )] for m in matches],
                    list(tables_columns.keys())[0],
                    list(tables_columns.keys())[1] if len(tables_columns) > 1 else ""
                )
                logger.info(f"[v7.0] Semantic Column Matches found: {len(all_column_matches)}")

        except Exception as e:
            logger.warning(f"[v7.0] Pattern analysis failed: {e}")
            pattern_analysis = {"error": str(e)}
            llm_pattern_context = ""
            all_column_matches = []

        self._report_progress(0.4, "Computing additional similarity metrics")

        # 3. 추가 유사도 분석 (MinHash/LSH)
        advanced_results = {}
        try:
            for table_name, columns in tables_data.items():
                for col_name, values in columns.items():
                    if len(values) >= 10:
                        col_key = f"{table_name}.{col_name}"
                        self.advanced_similarity.add_minhash(col_key, set(str(v) for v in values))

            lsh_candidates = self.advanced_similarity.query_similar_all()
            advanced_results = {
                "lsh_candidates": len(lsh_candidates),
                "lsh_pairs": lsh_candidates[:20],
            }
        except Exception as e:
            logger.warning(f"AdvancedSimilarityAnalyzer failed: {e}")
            advanced_results = {"error": str(e)}

        self._report_progress(0.5, "Preparing data for LLM decision")

        # 4. LLM 판단을 위한 데이터 준비
        # 상위 30개 후보를 LLM에게 전달
        candidates_for_llm = []
        for c in fk_candidates[:30]:
            confidence = sum(c.confidence_factors.values())
            candidates_for_llm.append({
                "from_table": c.from_table,
                "from_column": c.from_column,
                "to_table": c.to_table,
                "to_column": c.to_column,
                "metrics": {
                    "jaccard_similarity": round(c.jaccard_similarity, 4),
                    "containment_fk_in_pk": round(c.containment_fk_in_pk, 4),
                    "fk_unique_ratio": round(c.fk_unique_ratio, 4),
                    "pk_unique_ratio": round(c.pk_unique_ratio, 4),
                    "name_similarity": round(c.name_similarity, 4),
                },
                "evidence": {
                    "same_normalized_name": c.same_normalized_name,
                    "same_prefix_group": c.same_prefix_group,
                    "sample_overlapping_values": c.sample_overlapping_values[:5],
                    "sample_fk_only_values": c.sample_fk_only_values[:3],
                },
                "detection_method": c.detection_method,
                "algorithm_confidence": round(confidence, 3),
            })

        self._report_progress(0.6, "Requesting LLM FK decision")

        # 5. LLM에게 FK 판단 요청 (LLM이 결정권을 가짐) - v7.0 Enhanced
        # Bridge Table 및 컬럼 정규화 컨텍스트 준비
        bridge_table_section = ""
        if pattern_analysis.get("same_table_mappings"):
            bridge_tables_data = [
                {
                    "table": m.table,
                    "column_a": m.column_a,
                    "column_b": m.column_b,
                    "pattern_a": m.pattern_a,
                    "pattern_b": m.pattern_b,
                    "sample_mappings": m.sample_mappings[:5],
                    "confidence": m.confidence,
                }
                for m in pattern_analysis["same_table_mappings"]
            ]
            bridge_table_section = f"""
## BRIDGE TABLE ANALYSIS (v7.0 - Code Mapping Tables)
The following tables contain 1:1 mappings between different code systems.
These can connect tables that don't share direct value overlap.

{json.dumps(bridge_tables_data, indent=2)}

**IMPORTANT**: Use these bridge tables to find INDIRECT FK relationships!
Example: If table_a.iata_carrier doesn't match table_b.icao_carrier directly,
but airlines table has iata_code ↔ icao_code mapping, then there's an INDIRECT relationship.
"""

        semantic_matches_section = ""
        if all_column_matches:
            semantic_matches_section = f"""
## SEMANTIC COLUMN MATCHES (v7.0 - Abbreviation Expansion)
These column pairs have similar semantic meaning despite different names.
(flt_no → flight_number, carr_cd → carrier_code, etc.)

{json.dumps(all_column_matches[:20], indent=2)}

**IMPORTANT**: Even if column names look different, they may refer to the same entity!
"""

        # v7.1: Semantic FK context 추가
        semantic_fk_section = ""
        if semantic_fk_context:
            semantic_fk_section = f"""
## v7.1 COMPREHENSIVE SEMANTIC FK ANALYSIS
The following analysis was performed using advanced semantic detection:
- Column Name Semantic Tokenizer: Extracts table references from column names
- Bridge Table Detector: Identifies mapping tables connecting multiple code systems
- Value Pattern Correlator: Finds columns with high value overlap
- Dynamic Abbreviation Learner: Learns abbreviation patterns from data

{semantic_fk_context}
"""

        instruction = f"""Analyze these FK candidates and DECIDE which ones are TRUE foreign key relationships.

## FK CANDIDATES (computed by domain-agnostic algorithm)
The algorithm has normalized names and computed metrics. YOU must decide if each is a real FK.

{json.dumps(candidates_for_llm, indent=2)}
{bridge_table_section}
{semantic_matches_section}
{semantic_fk_section}
## TABLES IN THIS DATASET
{list(tables_columns.keys())}

## YOUR TASK
For EACH candidate above:
1. Look at the metrics (containment, cardinality, name similarity)
2. Consider the evidence (sample values, naming patterns)
3. **Check Bridge Tables**: Can indirect relationships be established?
4. **Check Semantic Matches**: Are there columns with same meaning but different names?
5. DECIDE: Is this a TRUE FK relationship or FALSE POSITIVE?

### INDIRECT FK RELATIONSHIPS (CRITICAL!)
Some columns may NOT have direct value overlap but can be connected through bridge tables.
Example: flights.iata_carrier → airlines.iata_code ↔ airlines.icao_code ← other_table.icao_carrier

For each potential indirect relationship:
- Identify the bridge table
- Specify the join path
- Explain why this connection makes business sense

IMPORTANT:
- The algorithm does NOT know the domain. YOU must understand the data context.
- Look at table names and column names to understand what domain this is.
- Use the metrics as evidence, but make your own judgment.
- **Actively look for INDIRECT FK relationships via bridge tables!**

Respond with JSON:
```json
{{
    "decisions": [
        {{
            "from_table": "<table>",
            "from_column": "<column>",
            "to_table": "<table>",
            "to_column": "<column>",
            "decision": "TRUE_FK | FALSE_POSITIVE | UNCERTAIN",
            "confidence": 0.0-1.0,
            "reasoning": "Why you made this decision",
            "is_indirect": false
        }}
    ],
    "indirect_fks": [
        {{
            "from_table": "<table>",
            "from_column": "<column>",
            "to_table": "<table>",
            "to_column": "<column>",
            "via_bridge_table": "<bridge_table_name>",
            "join_path": "table_a.col → bridge.col_a ↔ bridge.col_b ← table_b.col",
            "confidence": 0.0-1.0,
            "reasoning": "Why this indirect relationship exists"
        }}
    ],
    "detected_domain": "What domain/industry is this data from?",
    "data_model_summary": "Brief description of the data model structure",
    "hierarchy_detected": ["parent_table -> child_table relationships"]
}}
```"""

        response = await self.call_llm(instruction, max_tokens=4000)
        llm_decisions = parse_llm_json(response, default={})

        self._report_progress(0.8, "Processing LLM decisions")

        # 6. LLM 결정을 기반으로 최종 FK 리스트 생성
        validated_fks = []
        decisions_map = {}

        for decision in llm_decisions.get("decisions", []):
            key = (
                decision.get("from_table"),
                decision.get("from_column"),
                decision.get("to_table"),
                decision.get("to_column"),
            )
            decisions_map[key] = decision

            if decision.get("decision") == "TRUE_FK":
                validated_fks.append({
                    "table_a": decision.get("from_table"),
                    "column_a": decision.get("from_column"),
                    "table_b": decision.get("to_table"),
                    "column_b": decision.get("to_column"),
                    "confidence": decision.get("confidence", 0.8),
                    "detection_method": "universal_fk_detector_v6",
                    "llm_reasoning": decision.get("reasoning", ""),
                    "relationship_type": "fk_relationship",
                    # 호환성을 위한 필드
                    "jaccard_similarity": 0.0,
                    "overlap_ratio_a": 0.0,
                    "overlap_ratio_b": 0.0,
                    "containment_a_in_b": 0.0,
                    "containment_b_in_a": 0.0,
                })

        # 7. 알고리즘 후보 중 LLM이 결정하지 않은 것도 추가 (높은 신뢰도)
        for c in fk_candidates:
            key = (c.from_table, c.from_column, c.to_table, c.to_column)
            if key not in decisions_map:
                # LLM이 결정하지 않았지만 알고리즘 신뢰도가 높은 경우
                algo_confidence = sum(c.confidence_factors.values())
                if algo_confidence >= 0.6:
                    validated_fks.append({
                        "table_a": c.from_table,
                        "column_a": c.from_column,
                        "table_b": c.to_table,
                        "column_b": c.to_column,
                        "confidence": algo_confidence,
                        "detection_method": c.detection_method,
                        "relationship_type": "algorithm_detected",
                        "jaccard_similarity": c.jaccard_similarity,
                        "overlap_ratio_a": c.containment_fk_in_pk,
                        "overlap_ratio_b": c.containment_pk_in_fk,
                        "containment_a_in_b": c.containment_fk_in_pk,
                        "containment_b_in_a": c.containment_pk_in_fk,
                    })

        # 7.5 v7.0: Indirect FK 처리 (Bridge Table 경유)
        indirect_fks = []
        for indirect in llm_decisions.get("indirect_fks", []):
            indirect_fks.append({
                "table_a": indirect.get("from_table"),
                "column_a": indirect.get("from_column"),
                "table_b": indirect.get("to_table"),
                "column_b": indirect.get("to_column"),
                "confidence": indirect.get("confidence", 0.7),
                "detection_method": "bridge_table_v7",
                "llm_reasoning": indirect.get("reasoning", ""),
                "relationship_type": "indirect_fk",
                "via_bridge_table": indirect.get("via_bridge_table"),
                "join_path": indirect.get("join_path"),
            })

        if indirect_fks:
            logger.info(f"[v7.0] Indirect FKs via Bridge Tables: {len(indirect_fks)}")
            # Indirect FK도 Evidence로 기록
            for ifk in indirect_fks:
                self.record_evidence(
                    finding=f"간접 FK 관계 발견: {ifk['table_a']}.{ifk['column_a']} → {ifk['table_b']}.{ifk['column_b']} (via {ifk['via_bridge_table']})",
                    reasoning=f"Bridge Table 경유: {ifk.get('join_path', 'N/A')}. {ifk.get('llm_reasoning', '')[:100]}",
                    conclusion=f"Bridge Table을 통한 간접 FK 관계 확인됨",
                    data_references=[
                        f"{ifk['table_a']}.{ifk['column_a']}",
                        f"{ifk['table_b']}.{ifk['column_b']}",
                        f"{ifk['via_bridge_table']} (bridge)"
                    ],
                    metrics={"confidence": ifk.get("confidence", 0.7)},
                    confidence=ifk.get("confidence", 0.7),
                    topics=[ifk['table_a'], ifk['table_b'], ifk['via_bridge_table'], "indirect_fk", "bridge_table"],
                )

        self._report_progress(0.95, "Updating context")

        # === v11.0: Evidence Block 생성 ===
        # FK 탐지 결과를 근거 체인에 저장
        detected_domain = llm_decisions.get("detected_domain", "unknown")
        for fk in validated_fks[:10]:  # 상위 10개 FK만 근거로 저장
            self.record_evidence(
                finding=f"FK 관계 발견: {fk['table_a']}.{fk['column_a']} → {fk['table_b']}.{fk['column_b']}",
                reasoning=f"FK 탐지 방법: {fk.get('detection_method', 'unknown')}. LLM 판단: {fk.get('llm_reasoning', 'N/A')[:100]}",
                conclusion=f"유효한 FK 관계로 확인됨 (confidence: {fk.get('confidence', 0):.0%})",
                data_references=[
                    f"{fk['table_a']}.{fk['column_a']}",
                    f"{fk['table_b']}.{fk['column_b']}"
                ],
                metrics={
                    "confidence": fk.get("confidence", 0),
                    "jaccard_similarity": fk.get("jaccard_similarity", 0),
                    "containment_a_in_b": fk.get("containment_a_in_b", 0),
                },
                confidence=fk.get("confidence", 0.7),
                topics=[fk['table_a'], fk['table_b'], "fk", "relationship", detected_domain],
            )

        logger.info(f"v7.0 Universal FK Detection: {len(validated_fks)} direct FKs, {len(indirect_fks)} indirect FKs validated")

        # Bridge Table 정보 직렬화
        bridge_tables_serialized = []
        for m in pattern_analysis.get("same_table_mappings", []):
            bridge_tables_serialized.append({
                "table": m.table,
                "column_a": m.column_a,
                "column_b": m.column_b,
                "pattern_a": m.pattern_a,
                "pattern_b": m.pattern_b,
                "mapping_count": m.mapping_count,
                "is_bijective": m.is_bijective,
                "confidence": m.confidence,
            })

        # v22.0: Agent Learning - 경험 기록
        self.record_experience(
            experience_type="fk_detection",
            input_data={"task": "value_matching", "candidates": len(fk_candidates)},
            output_data={"validated_fks": len(validated_fks), "indirect_fks": len(indirect_fks)},
            outcome="success" if validated_fks else "partial",
            confidence=0.85 if validated_fks else 0.5,
        )

        # v22.0: Agent Bus - FK 탐지 결과 브로드캐스트
        await self.broadcast_discovery(
            discovery_type="fk_detected",
            data={
                "validated_fks": len(validated_fks),
                "indirect_fks": len(indirect_fks),
                "fk_candidates": [
                    {"from": f"{fk['table_a']}.{fk['column_a']}", "to": f"{fk['table_b']}.{fk['column_b']}", "confidence": fk.get('confidence', 0)}
                    for fk in validated_fks[:10]  # 상위 10개만
                ],
            },
        )

        return TodoResult(
            success=True,
            output={
                "candidates_found": len(fk_candidates),
                "fks_validated": len(validated_fks),
                "indirect_fks_found": len(indirect_fks),
                "bridge_tables_found": len(bridge_tables_serialized),
                "semantic_matches_found": len(all_column_matches),
                "detected_domain": llm_decisions.get("detected_domain", "unknown"),
                "lsh_candidates_found": advanced_results.get("lsh_candidates", 0),
            },
            context_updates={
                "value_overlaps": validated_fks,
                "indirect_fks": indirect_fks,
                "bridge_tables": bridge_tables_serialized,
                "semantic_column_matches": all_column_matches,
            },
            metadata={
                "fk_candidates": [
                    {
                        "from_table": c.from_table,
                        "from_column": c.from_column,
                        "to_table": c.to_table,
                        "to_column": c.to_column,
                        "jaccard": c.jaccard_similarity,
                        "containment": c.containment_fk_in_pk,
                        "confidence": sum(c.confidence_factors.values()),
                    }
                    for c in fk_candidates
                ],
                "llm_decisions": llm_decisions,
                "advanced_similarity_results": advanced_results,
                "data_model_summary": llm_decisions.get("data_model_summary", ""),
                "hierarchy_detected": llm_decisions.get("hierarchy_detected", []),
                "v7_bridge_tables": bridge_tables_serialized,
                "v7_indirect_fks": indirect_fks,
                "v7_semantic_matches": all_column_matches,
            },
        )

    # ==========================================================================
    # v6.0: 룰베이스 메서드 제거됨
    # 이전에 있던 메서드들:
    # - _detect_fk_by_column_names: 도메인 특화 컬럼명 패턴 (제거)
    # - _detect_ad_platform_fks: 광고 플랫폼 특화 FK 탐지 (제거)
    # - _detect_id_pattern_fks: 플랫폼별 ID 패턴 (제거)
    # - _detect_prefixed_id_fks: 플랫폼별 prefix FK (제거)
    # - _find_pk_candidate: 도메인 특화 PK 탐지 (제거)
    #
    # 대체: UniversalFKDetector (analysis/universal_fk_detector.py)
    # - 도메인 무관 네이밍 정규화
    # - 값 기반 FK 검증 (Jaccard, Containment, Cardinality)
    # - LLM이 최종 결정권을 가짐
    # ==========================================================================

    def _pre_llm_confidence_filter(
        self,
        candidates: List[Any],
        min_confidence: float = 0.5
    ) -> List[Any]:
        """
        v7.9: Pre-LLM Confidence Filter

        LLM 평가 전에 약한 FK 후보를 필터링하여:
        - LLM의 판단 부담 감소
        - False Positive 사전 제거
        - 처리 시간 및 토큰 비용 절감

        Args:
            candidates: FK 후보 리스트 (FKCandidate 객체)
            min_confidence: 최소 신뢰도 임계값 (기본 0.5)

        Returns:
            필터링된 FK 후보 리스트
        """
        if not candidates:
            return candidates

        filtered = []
        for candidate in candidates:
            # FKCandidate 객체의 confidence_factors 합계 계산
            if hasattr(candidate, 'confidence_factors'):
                confidence = sum(candidate.confidence_factors.values())
            elif hasattr(candidate, 'confidence'):
                confidence = candidate.confidence
            elif isinstance(candidate, dict):
                confidence = candidate.get('confidence', 0)
            else:
                confidence = 0

            # v7.9: 다중 요소 기반 스코어링
            # 0.4*Jaccard + 0.4*Containment + 0.2*NameSimilarity
            if hasattr(candidate, 'jaccard_similarity') and hasattr(candidate, 'containment_fk_in_pk'):
                composite_score = (
                    0.4 * candidate.jaccard_similarity +
                    0.4 * candidate.containment_fk_in_pk +
                    0.2 * (candidate.confidence_factors.get('name_similarity', 0) if hasattr(candidate, 'confidence_factors') else 0)
                )
                # composite_score와 기존 confidence 중 높은 값 사용
                confidence = max(confidence, composite_score)

            if confidence >= min_confidence:
                filtered.append(candidate)
            else:
                logger.debug(f"[v7.9] Filtered out weak candidate: "
                           f"{getattr(candidate, 'from_table', '?')}.{getattr(candidate, 'from_column', '?')} → "
                           f"{getattr(candidate, 'to_table', '?')}.{getattr(candidate, 'to_column', '?')} "
                           f"(confidence={confidence:.3f} < {min_confidence})")

        return filtered


class EntityClassifierAutonomousAgent(AutonomousAgent):
    """
    엔티티 분류 자율 에이전트 (v3.0 - Full Entity Resolution)

    v3.0 변경사항:
    - EntityExtractor: 패턴 기반 엔티티 추출
    - EntityResolver: ML 기반 엔티티 매칭/통합 (신규 통합)
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.entity_extractor = EntityExtractor()  # v2.1: Enhanced Entity Extraction
        # v3.0: ML 기반 엔티티 해결 통합
        # NOTE: use_embeddings=False to avoid SentenceTransformer loading issues
        self.entity_resolver = EntityResolver(use_embeddings=False)

    # ==========================================================================
    # v6.0: 도메인 특화 패턴 제거됨
    # 이전에 있던 패턴들:
    # - HEALTHCARE_PATTERNS: 의료 도메인 (Patient, Encounter, Provider, ...)
    # - SUPPLY_CHAIN_PATTERNS: 공급망 도메인 (Plant, Port, Order, ...)
    # - MANUFACTURING_PATTERNS: 제조 도메인 (Machine, Sensor, ...)
    #
    # 대체: LLM이 테이블명/컬럼명을 분석해서 엔티티 타입을 직접 결정
    # - 범용 알고리즘: 테이블 구조 분석 (ID 컬럼, FK 패턴 등)
    # - LLM 결정: 도메인 컨텍스트 이해 및 엔티티 타입 분류
    # ==========================================================================

    @property
    def agent_type(self) -> str:
        return "entity_classifier"

    @property
    def agent_name(self) -> str:
        return "Entity Classifier"

    @property
    def phase(self) -> str:
        return "discovery"

    def get_system_prompt(self) -> str:
        return """You are a Domain Entity Classification Expert who ANALYZES and CLASSIFIES table entities.

You are the DECISION MAKER. The algorithm provides structural analysis, but YOU decide:
1. What DOMAIN this data belongs to
2. What ENTITY TYPE each table represents
3. How tables should be GROUPED into unified entities (ONLY if truly duplicated)
4. What CANONICAL NAMES to use for entities

## DOMAIN OPTIONS (v7.9 확장):
- marketing / digital_marketing: campaigns, leads, conversions, ads, email marketing, UTM tracking
- e_commerce / retail: orders, products, customers, inventory, shopping cart
- advertising: ad campaigns, impressions, clicks, CTR, ad performance
- finance: transactions, accounts, payments, invoices
- healthcare: patients, diagnoses, treatments, providers
- manufacturing: work orders, inventory, production, machines
- supply_chain: shipments, logistics, warehouses, suppliers

## ENTITY CLASSIFICATION RULES (v7.9):

1. **INDIVIDUAL ENTITIES FIRST**: Each table = One entity (MINIMUM)
   - Do NOT merge tables unless there is CLEAR evidence of duplication
   - Even related tables (email_sends, email_events) are SEPARATE entities
   - Table count = Minimum entity count

2. **ENTITY ROLES** (v7.9 확장):
   - master: Core business entities (Customer, Product, Campaign)
   - transaction: Business transactions (Order, Lead, Conversion)
   - event: Time-based activity events (EmailSend, EmailEvent, WebSession, AdClick)
   - metric: Aggregated performance metrics (AdPerformance, CampaignMetrics)
   - reference: Lookup/reference tables
   - bridge: M:N relationship tables

3. **EVENT TYPE DETECTION** (v7.9 신규):
   - Tables with timestamp + action columns → EVENT type
   - Examples: web_sessions, email_events, conversions, ad_clicks
   - These should NEVER be merged with master entities

## REQUIRED OUTPUT FORMAT (JSON Schema):
```json
{
  "domain_classification": {
    "primary_domain": "marketing|digital_marketing|e_commerce|retail|advertising|finance|healthcare|manufacturing|supply_chain|other",
    "secondary_domain": "optional secondary domain if applicable",
    "confidence": 0.0,
    "evidence": ["List of evidence supporting domain classification"]
  },
  "individual_entities": [
    {
      "table_name": "original_table_name",
      "entity_name": "EntityName",
      "entity_type": "Campaign|Customer|Product|Order|Lead|EmailSend|EmailEvent|WebSession|Conversion|AdPerformance|etc",
      "entity_role": "master|transaction|event|metric|reference|bridge",
      "confidence": 0.0,
      "reasoning": "Why this classification was chosen"
    }
  ],
  "unified_entities": [
    {
      "canonical_name": "UnifiedEntityName",
      "source_tables": ["table1", "table2"],
      "entity_type": "The unified entity type",
      "primary_key_column": "id_column_name",
      "merge_strategy": "union|intersection|primary_source",
      "merge_justification": "REQUIRED: Clear evidence why these tables should be merged",
      "confidence": 0.0
    }
  ],
  "entity_relationships": [
    {
      "source_entity": "EntityName",
      "target_entity": "EntityName",
      "relationship_type": "one_to_many|many_to_many|one_to_one",
      "relationship_name": "has|belongs_to|contains|triggers|etc",
      "confidence": 0.0
    }
  ],
  "naming_recommendations": [
    {
      "original_name": "original_table_or_column",
      "recommended_name": "CanonicalName",
      "reason": "Why this name is recommended"
    }
  ],
  "classification_summary": "Overall summary of entity classification decisions",
  "entity_count_validation": {
    "table_count": 0,
    "individual_entity_count": 0,
    "unified_entity_count": 0,
    "validation_passed": true
  }
}
```

IMPORTANT: individual_entity_count MUST be >= table_count. Each table represents at least one entity.

Respond with structured JSON matching this schema."""

    async def execute_task(
        self,
        todo: "PipelineTodo",
        context: "SharedContext",
    ) -> "TodoResult":
        """엔티티 분류 작업 실행"""
        from ...todo.models import TodoResult
        from ...shared_context import UnifiedEntity

        self._report_progress(0.1, "Classifying entities (pattern matching)")

        # 1. 패턴 기반 분류
        classifications = {}
        for table_name, table_info in context.tables.items():
            columns = [c.get("name", "") for c in table_info.columns]
            classification = self._classify_by_pattern(table_name, columns)
            classifications[table_name] = classification

        self._report_progress(0.2, "Running Enhanced Entity Extraction (v2.1)")

        # 2. Enhanced Entity Extraction (v2.1) - 컬럼 기반 숨은 엔티티 추출
        tables_for_extraction = {
            name: {"columns": info.columns, "row_count": info.row_count}
            for name, info in context.tables.items()
        }
        # v22.1: 전체 데이터 로드 (샘플링 금지)
        sample_data = context.get_all_full_data()

        extracted_entities = []
        try:
            extracted_entities = self.entity_extractor.extract_entities(
                tables=tables_for_extraction,
                sample_data=sample_data,
            )
            logger.info(f"Enhanced Entity Extraction found {len(extracted_entities)} entities")
        except Exception as e:
            logger.warning(f"Enhanced Entity Extraction failed: {e}")

        self._report_progress(0.30, "Running Entity Resolution (v3.0 ML-based)")

        # v3.0: EntityResolver를 사용하여 엔티티 매칭/통합
        resolved_entities = []
        try:
            # EntityResolver로 중복 엔티티 매칭
            entity_candidates = [
                {
                    "entity_id": e.entity_id,
                    "name": e.name,
                    "source_tables": e.source_tables,
                    # inferred_attributes는 List[str]이므로 Dict로 변환
                    "attributes": {attr: attr for attr in e.inferred_attributes} if isinstance(e.inferred_attributes, list) else e.inferred_attributes,
                }
                for e in extracted_entities
            ]
            if entity_candidates:
                resolved_entities = self.entity_resolver.resolve_entities(entity_candidates)
                logger.info(f"EntityResolver merged into {len(resolved_entities)} resolved entities")
        except Exception as e:
            logger.warning(f"EntityResolver failed: {e}")

        self._report_progress(0.40, "Grouping by homeomorphisms")

        # 3. Homeomorphism 기반 그룹화
        entity_groups = self._group_by_homeomorphisms(
            classifications, context.homeomorphisms
        )

        # Enhanced 엔티티 요약
        enhanced_entity_summary = [
            {
                "name": e.name,
                "type": e.entity_type.value,
                "source_tables": e.source_tables[:3],
                "confidence": round(e.confidence, 2),
            }
            for e in extracted_entities[:20]
        ]

        # v17.0: SemanticSearcher를 사용한 Cross-table 엔티티 발견
        semantic_entity_matches = []
        try:
            for entity in extracted_entities[:10]:  # 상위 10개 엔티티만
                search_results = await self.semantic_search(
                    query=f"{entity.name} {' '.join(entity.source_tables)}",
                    limit=5,
                    expand_query=False,
                )
                if search_results:
                    semantic_entity_matches.append({
                        "entity": entity.name,
                        "matches": [r.get("entity_id", r.get("matched_text", "")) for r in search_results[:3]],
                        "top_score": search_results[0].get("score", 0) if search_results else 0,
                    })
            if semantic_entity_matches:
                logger.info(f"[v17.0] SemanticSearcher found {len(semantic_entity_matches)} potential entity matches")
        except Exception as e:
            logger.debug(f"[v17.0] Semantic entity search skipped: {e}")

        self._report_progress(0.5, "Requesting LLM validation")

        # 4. LLM에게 검증 요청
        instruction = f"""I have classified tables by pattern matching and grouped them by topological similarity.
Additionally, I have extracted hidden entities from column names (e.g., customer_id -> Customer entity).

PATTERN-BASED CLASSIFICATIONS:
{json.dumps(classifications, indent=2)}

ENTITY GROUPS (tables representing same entity):
{json.dumps(entity_groups, indent=2)}

ENHANCED ENTITY EXTRACTION (v2.1 - column-based hidden entities):
{json.dumps(enhanced_entity_summary, indent=2)}

Domain: {context.get_industry()}

Please:
1. VALIDATE or CORRECT the entity type assignments
2. CONFIRM or ADJUST the entity groups
3. INCORPORATE the hidden entities from column-based extraction
4. SUGGEST canonical names for unified entities
5. IDENTIFY key columns for each entity

Respond with JSON:
```json
{{
    "validated_classifications": {{
        "<table_name>": {{
            "entity_type": "<validated type>",
            "confidence": <float>,
            "corrections": "<any corrections made>"
        }}
    }},
    "unified_entities": [
        {{
            "entity_id": "<unique_id>",
            "entity_type": "<Patient|Encounter|...>",
            "canonical_name": "<unified name>",
            "source_tables": ["table1", "table2"],
            "key_mappings": {{"table1": "col1", "table2": "col2"}},
            "confidence": <float>,
            "reasoning": "<why these tables are same entity>"
        }}
    ],
    "summary": "<classification summary>"
}}
```"""

        response = await self.call_llm(instruction, max_tokens=4000)
        interpretation = parse_llm_json(response, default={})

        self._report_progress(0.8, "Building unified entities")

        # v14.0: LLM 파싱 실패 감지 및 폴백 체인
        llm_unified_entities = interpretation.get("unified_entities", [])
        if not llm_unified_entities and not interpretation.get("validated_classifications"):
            logger.warning("[v14.0 Fallback] LLM parsing failed or returned empty - using structural analysis as fallback")
            # 폴백: extracted_entities에서 직접 UnifiedEntity 생성
            for table_name, classification in classifications.items():
                if classification.get("entity_type") and classification.get("entity_type") != "Unknown":
                    llm_unified_entities.append({
                        "entity_id": f"fallback_entity_{table_name}",
                        "entity_type": classification["entity_type"],
                        "canonical_name": classification["entity_type"],
                        "source_tables": [table_name],
                        "key_mappings": {table_name: classification.get("key_column", "id")},
                        "confidence": classification.get("confidence", 0.3) * 0.8,  # 폴백이므로 신뢰도 감소
                        "reasoning": "Fallback from structural analysis (LLM parsing failed)"
                    })
            logger.info(f"[v14.0 Fallback] Generated {len(llm_unified_entities)} entities from structural analysis")

        # 4. UnifiedEntity 객체 생성
        unified_entities = []
        for ue_data in llm_unified_entities:
            entity = UnifiedEntity(
                entity_id=ue_data.get("entity_id", f"entity_{len(unified_entities)}"),
                entity_type=ue_data.get("entity_type", "Unknown"),
                canonical_name=ue_data.get("canonical_name", ""),
                source_tables=ue_data.get("source_tables", []),
                key_mappings=ue_data.get("key_mappings", {}),
                confidence=ue_data.get("confidence", 0.0),
                unified_attributes=[],
            )
            unified_entities.append(entity)

        # v6.0: LLM 결정 우선, 구조 분석 결과는 보조적으로 사용
        # (도메인 특화 CRITICAL_ENTITY_TYPES 하드코딩 제거됨)

        # 이미 분류된 테이블 추적
        classified_tables = set()
        for ue in unified_entities:
            classified_tables.update(ue.source_tables)

        # v6.0: LLM이 분류하지 않은 테이블만 구조 분석 결과로 보완
        for table_name, classification in classifications.items():
            entity_type = classification["entity_type"]
            confidence = classification["confidence"]

            # Unknown 타입은 건너뜀
            if entity_type == "Unknown":
                continue

            # LLM이 이미 분류한 테이블은 건너뜀
            if table_name in classified_tables:
                continue

            # 구조 분석 신뢰도가 높은 경우에만 보완
            if confidence >= 0.5:
                entity = UnifiedEntity(
                    entity_id=f"entity_{table_name}",
                    entity_type=entity_type,
                    canonical_name=entity_type,
                    source_tables=[table_name],
                    key_mappings={table_name: classification.get("key_column", "id")},
                    confidence=confidence,
                    unified_attributes=[],
                )
                unified_entities.append(entity)
                classified_tables.add(table_name)
                logger.info(f"v6.0: Added {entity_type} entity from structural analysis: {table_name} (conf: {confidence:.2f})")

        self._report_progress(0.95, "Updating context")

        # === v11.0: Evidence Block 생성 ===
        # 엔티티 분류 결과를 근거 체인에 저장
        for entity in unified_entities[:10]:  # 상위 10개 엔티티만
            self.record_evidence(
                finding=f"통합 엔티티 발견: {entity.canonical_name} (유형: {entity.entity_type})",
                reasoning=f"소스 테이블 {entity.source_tables}에서 추출. 키 매핑: {entity.key_mappings}",
                conclusion=f"'{entity.canonical_name}' 엔티티로 {len(entity.source_tables)}개 테이블 통합 가능",
                data_references=[f"{t}.{k}" for t, k in entity.key_mappings.items()],
                metrics={
                    "confidence": entity.confidence,
                    "source_table_count": len(entity.source_tables),
                },
                confidence=entity.confidence,
                topics=[entity.canonical_name, entity.entity_type, "entity", "classification"],
            )

        # v2.1: Enhanced 엔티티를 직렬화 가능한 형태로 변환
        extracted_entities_serialized = [
            {
                "name": e.name,
                "entity_type": e.entity_type.value,
                "source_tables": e.source_tables,
                "source_columns": e.source_columns,
                "confidence": e.confidence,
                "inferred_attributes": e.inferred_attributes,
            }
            for e in extracted_entities
        ]

        # === v22.0: semantic_searcher로 유사 엔티티 검색 ===
        semantic_suggestions = []
        try:
            semantic_searcher = self.get_v17_service("semantic_searcher")
            if semantic_searcher and unified_entities:
                for entity in unified_entities[:5]:  # 상위 5개 엔티티만
                    query = f"{entity.canonical_name} {entity.entity_type}"
                    similar = await semantic_searcher.search(query=query, limit=3)
                    if similar:
                        semantic_suggestions.append({
                            "entity": entity.canonical_name,
                            "similar_concepts": [s.get("name", s.get("id", "")) for s in similar[:3]],
                        })
                if semantic_suggestions:
                    logger.info(f"[v22.0] SemanticSearcher found {len(semantic_suggestions)} entity suggestions")
        except Exception as e:
            logger.debug(f"[v22.0] SemanticSearcher skipped: {e}")

        # v22.0: Agent Learning - 경험 기록
        self.record_experience(
            experience_type="entity_mapping",
            input_data={"task": "entity_classification", "tables": list(context.tables.keys())},
            output_data={"entities_classified": len(unified_entities), "entity_types": list(set(e.entity_type for e in unified_entities))},
            outcome="success",
            confidence=0.8,
        )

        return TodoResult(
            success=True,
            output={
                "entities_classified": len(unified_entities),
                "entity_types": list(set(e.entity_type for e in unified_entities)),
                "enhanced_entities_extracted": len(extracted_entities),  # v2.1
            },
            context_updates={
                "unified_entities": unified_entities,
                "extracted_entities": extracted_entities_serialized,  # v2.1
            },
            metadata={
                "classifications": classifications,
                "llm_validation": interpretation,
                "enhanced_entity_summary": enhanced_entity_summary,  # v2.1
            },
        )

    def _classify_by_pattern(
        self,
        table_name: str,
        columns: List[str],
    ) -> Dict[str, Any]:
        """
        v6.0: 범용 구조 분석 기반 분류 (도메인 무관)

        도메인 특화 패턴 대신 범용 알고리즘 사용:
        - 테이블명에서 엔티티명 추출 (네이밍 정규화)
        - ID/코드 컬럼 패턴 분석
        - 구조적 특징 분석 (마스터 vs 트랜잭션)

        LLM이 최종 엔티티 타입을 결정하므로, 여기서는 구조 분석만 수행
        """
        import re

        table_lower = table_name.lower()
        cols_lower = [c.lower() for c in columns]

        # === 1. 테이블명에서 엔티티명 추출 ===
        # 공통 prefix 제거: tbl_, t_, tb_, table_, google_ads_, meta_ads_ 등
        entity_name = table_lower
        prefixes_to_remove = [
            r'^tbl_', r'^t_', r'^tb_', r'^table_',
            r'^dim_', r'^fact_', r'^stg_', r'^raw_',
            r'^[a-z]+_ads_',  # google_ads_, meta_ads_, naver_ads_
            r'^[a-z]+_',      # 일반 prefix
        ]
        for prefix in prefixes_to_remove:
            entity_name = re.sub(prefix, '', entity_name)

        # v19.1: 복수형 → 단수형 변환 (패턴 우선순위 개선)
        if entity_name.endswith('ies'):
            entity_name = entity_name[:-3] + 'y'       # companies → company
        elif entity_name.endswith('ves'):
            entity_name = entity_name[:-3] + 'f'       # wolves → wolf
        elif entity_name.endswith('ices'):
            entity_name = entity_name[:-4] + 'ex'      # indices → index
        elif entity_name.endswith('ees'):
            entity_name = entity_name[:-1]              # employees → employee
        elif entity_name.endswith('ses') or entity_name.endswith('xes') or entity_name.endswith('zes') or entity_name.endswith('ches') or entity_name.endswith('shes'):
            entity_name = entity_name[:-2]              # databases → database, boxes → box
        elif entity_name.endswith('es'):
            entity_name = entity_name[:-1]              # types → type (단, 위 패턴에 안 걸리는 -es)
        elif entity_name.endswith('s') and not entity_name.endswith('ss') and not entity_name.endswith('us') and not entity_name.endswith('is'):
            entity_name = entity_name[:-1]              # users → user (단, class/status/analysis 보호)

        # snake_case → PascalCase
        entity_type = ''.join(word.capitalize() for word in entity_name.split('_'))

        # === 2. ID/Key 컬럼 탐지 ===
        key_column = None
        id_patterns = [
            rf'^{entity_name}_id$',
            rf'^{entity_name}id$',
            r'^id$',
            rf'^{entity_name}_code$',
            r'^code$',
        ]
        for col in cols_lower:
            for pattern in id_patterns:
                if re.match(pattern, col):
                    key_column = columns[cols_lower.index(col)]
                    break
            if key_column:
                break

        # === 3. 구조적 특징 분석 ===
        structural_features = {
            "has_id_column": any('_id' in c or c == 'id' for c in cols_lower),
            "has_date_column": any('date' in c or 'time' in c for c in cols_lower),
            "has_amount_column": any(x in c for c in cols_lower for x in ['amount', 'price', 'cost', 'qty', 'quantity']),
            "has_name_column": any('name' in c or 'title' in c or 'description' in c for c in cols_lower),
            "has_fk_columns": sum(1 for c in cols_lower if c.endswith('_id') and c != 'id') > 0,
            "fk_count": sum(1 for c in cols_lower if c.endswith('_id') and c != 'id'),
        }

        # 신뢰도 계산: 기본 0.3 + 구조적 특징 보너스
        confidence = 0.3
        if key_column:
            confidence += 0.2  # ID 컬럼 있으면 보너스
        if structural_features["has_name_column"]:
            confidence += 0.1  # 이름 컬럼 있으면 마스터 테이블 가능성
        if structural_features["has_date_column"] and structural_features["has_amount_column"]:
            confidence += 0.15  # 트랜잭션 테이블 가능성
        if structural_features["fk_count"] >= 2:
            confidence += 0.1  # FK가 여러 개면 연관 테이블

        confidence = min(1.0, confidence)

        # 엔티티 타입이 너무 짧거나 의미 없으면 Unknown
        if len(entity_type) < 2 or entity_type.lower() in ['a', 'b', 'c', 'x', 'y', 'z']:
            entity_type = "Unknown"
            confidence = 0.0

        return {
            "entity_type": entity_type,
            "confidence": confidence,
            "key_column": key_column,
            "domain": "auto_detected",  # LLM이 도메인을 결정
            "structural_features": structural_features,
        }

    def _group_by_homeomorphisms(
        self,
        classifications: Dict[str, Dict],
        homeomorphisms: List,
    ) -> List[Dict[str, Any]]:
        """Homeomorphism 기반 그룹화"""
        groups = []

        # 동일 엔티티 타입의 homeomorphic 테이블 그룹화
        processed = set()

        for h in homeomorphisms:
            if not h.is_homeomorphic or h.confidence < 0.5:
                continue

            table_a, table_b = h.table_a, h.table_b

            if table_a in processed and table_b in processed:
                continue

            type_a = classifications.get(table_a, {}).get("entity_type", "Unknown")
            type_b = classifications.get(table_b, {}).get("entity_type", "Unknown")

            # 같은 타입이거나 하나가 Unknown인 경우 그룹화
            if type_a == type_b or type_a == "Unknown" or type_b == "Unknown":
                entity_type = type_a if type_a != "Unknown" else type_b

                # 기존 그룹에 추가하거나 새 그룹 생성
                added = False
                for group in groups:
                    if table_a in group["tables"] or table_b in group["tables"]:
                        group["tables"].add(table_a)
                        group["tables"].add(table_b)
                        added = True
                        break

                if not added:
                    groups.append({
                        "entity_type": entity_type,
                        "tables": {table_a, table_b},
                        "confidence": h.confidence,
                    })

                processed.add(table_a)
                processed.add(table_b)

        # Set을 List로 변환
        return [
            {**g, "tables": list(g["tables"])}
            for g in groups
        ]


class RelationshipDetectorAutonomousAgent(AutonomousAgent):
    """
    관계 탐지 자율 에이전트 (v3.0 - Full Graph Analysis)

    v3.0 변경사항:
    - GraphAnalyzer: 기본 그래프 분석
    - EnhancedFKDetector: 강화된 FK 탐지
    - GraphEmbeddingAnalyzer: Node2Vec + Community Detection (신규 통합)
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.graph_analyzer = GraphAnalyzer()
        self.enhanced_fk_detector = EnhancedFKDetector()  # v2.1: Enhanced FK Detection
        # v3.0: 그래프 임베딩 분석 (Node2Vec, Community Detection) 통합
        self.graph_embedding = GraphEmbeddingAnalyzer()
        # v5.0: FK False Positive Filter
        self.fp_filter = FalsePositiveFilter()

    @property
    def agent_type(self) -> str:
        return "relationship_detector"

    @property
    def agent_name(self) -> str:
        return "Relationship Detector"

    @property
    def phase(self) -> str:
        return "discovery"

    def get_system_prompt(self) -> str:
        return """You are a Database Relationship Detection Expert who INTERPRETS graph analysis results.

The algorithm has already:
- Built a relationship graph from FK and value overlaps
- Found connected components (data silos)
- Detected bridge edges (cross-silo connections)
- Computed relationship cardinalities

Your role is to:
1. INTERPRET the graph structure
2. EXPLAIN relationship semantics
3. IDENTIFY data integration pathways
4. RECOMMEND join strategies

## REQUIRED OUTPUT FORMAT (JSON Schema):
```json
{
  "graph_structure": {
    "node_count": 0,
    "edge_count": 0,
    "connected_components": 0,
    "density": 0.0,
    "structure_type": "star|chain|mesh|tree|disconnected"
  },
  "data_silos": [
    {
      "silo_id": "silo_1",
      "tables": ["table1", "table2"],
      "central_table": "hub_table_name",
      "isolation_reason": "Why these tables form a silo"
    }
  ],
  "relationships": [
    {
      "source_table": "table_name",
      "source_column": "column_name",
      "target_table": "table_name",
      "target_column": "column_name",
      "relationship_type": "foreign_key|value_overlap|semantic",
      "cardinality": "one_to_one|one_to_many|many_to_many",
      "confidence": 0.0,
      "semantic_meaning": "Business meaning of this relationship"
    }
  ],
  "bridge_connections": [
    {
      "from_silo": "silo_1",
      "to_silo": "silo_2",
      "bridge_table": "bridge_table_name",
      "connection_strength": "strong|medium|weak",
      "integration_potential": "high|medium|low"
    }
  ],
  "integration_pathways": [
    {
      "pathway_name": "Pathway description",
      "tables_involved": ["table1", "table2", "table3"],
      "join_sequence": [
        {
          "left_table": "table1",
          "right_table": "table2",
          "join_columns": {"left": "col1", "right": "col2"},
          "join_type": "inner|left|right|full"
        }
      ],
      "complexity": "simple|moderate|complex",
      "recommendation": "When to use this pathway"
    }
  ],
  "join_strategies": [
    {
      "strategy_name": "Strategy name",
      "description": "Strategy description",
      "applicable_scenarios": ["scenario1", "scenario2"],
      "performance_consideration": "Performance notes",
      "priority": "high|medium|low"
    }
  ],
  "relationship_summary": "Overall summary of detected relationships and integration recommendations"
}
```

Respond with structured JSON matching this schema."""

    async def execute_task(
        self,
        todo: "PipelineTodo",
        context: "SharedContext",
    ) -> "TodoResult":
        """관계 탐지 작업 실행"""
        from ...todo.models import TodoResult

        self._report_progress(0.1, "Building relationship graph")

        # 1. 테이블 정보 수집
        tables = {
            name: {
                "columns": info.columns,
                "row_count": info.row_count,
            }
            for name, info in context.tables.items()
        }

        # v22.1: 전체 데이터 로드 (Enhanced FK Detection용, 샘플링 금지)
        sample_data = context.get_all_full_data()

        # FK 정보 수집
        foreign_keys = []
        for table_name, table_info in context.tables.items():
            for fk in table_info.foreign_keys:
                foreign_keys.append({
                    "from_table": table_name,
                    **fk,
                })

        self._report_progress(0.15, "Running Enhanced FK Detection (v2.1)")

        # 2. Enhanced FK Detection (v2.1) - 컬럼명 매칭 + Inclusion Dependency
        enhanced_fk_candidates = []
        if sample_data:
            try:
                enhanced_fk_candidates = self.enhanced_fk_detector.detect_foreign_keys(
                    tables=tables,
                    sample_data=sample_data,
                    min_score=0.3,
                )
                logger.info(f"Enhanced FK Detection found {len(enhanced_fk_candidates)} candidates")
            except Exception as e:
                logger.warning(f"Enhanced FK Detection failed: {e}")

        # v5.0: False Positive 필터링
        self._report_progress(0.2, "Filtering false positive FK candidates")
        filtered_fk_candidates = []
        fp_filtered_count = 0
        for fk in enhanced_fk_candidates:
            # sample values를 빈 리스트로 전달 (Enhanced FK에는 sample이 없을 수 있음)
            is_fp, fp_reason = self.fp_filter.is_likely_false_positive(
                fk.from_column, fk.to_column, []
            )
            if is_fp:
                fp_filtered_count += 1
                logger.debug(f"FP filtered: {fk.from_table}.{fk.from_column} -> {fk.to_table}.{fk.to_column}: {fp_reason}")
            else:
                filtered_fk_candidates.append(fk)

        logger.info(f"FK FP Filter: {fp_filtered_count} false positives removed, {len(filtered_fk_candidates)} remain")
        enhanced_fk_candidates = filtered_fk_candidates

        # === v7.10: Post-Processing Filters for Enhanced FK Candidates ===
        # Problem A: 역방향 FK 필터링 (Cardinality 기반 방향 판단)
        enhanced_fk_candidates = self._filter_reverse_enhanced_fks(enhanced_fk_candidates, sample_data)

        # Problem B: 간접 연결 필터링 (Transitive Closure 제거)
        enhanced_fk_candidates = self._filter_transitive_enhanced_fks(enhanced_fk_candidates)

        logger.info(f"[v7.10] Enhanced FK post-filter: {len(enhanced_fk_candidates)} candidates remain")

        # === v16.0: Integrated FK Detection (Composite, Hierarchy, Temporal) ===
        # 1. 기존 enhanced_fk_candidates를 context에 저장
        self._report_progress(0.25, "Running Integrated FK Detection (v16.0)")
        for fk in enhanced_fk_candidates:
            fk_dict = {
                "from_table": fk.from_table,
                "from_column": fk.from_column,
                "to_table": fk.to_table,
                "to_column": fk.to_column,
                "fk_score": fk.fk_score,
                "confidence": fk.confidence,
                "value_inclusion": getattr(fk, 'value_inclusion', 0.0),
                "name_similarity": getattr(fk, 'name_similarity', 0.0),
                "detection_method": "single_fk",
            }
            if fk_dict not in context.enhanced_fk_candidates:
                context.enhanced_fk_candidates.append(fk_dict)

        # 2. 통합 FK 탐지기 실행 (Composite, Hierarchy, Temporal FK 추가)
        try:
            integrated_result = detect_all_fks_and_update_context(
                shared_context=context,
                enable_single_fk=False,  # 이미 위에서 실행됨
                enable_composite_fk=True,
                enable_hierarchy=True,
                enable_temporal=True,
                min_score=0.3,
            )
            logger.info(
                f"[v16.0] Integrated FK Detection: {integrated_result.total_unique_fks} total FKs "
                f"(composite={integrated_result.composite_fk_count}, "
                f"hierarchy={integrated_result.hierarchy_fk_count}, "
                f"temporal={integrated_result.temporal_fk_count})"
            )

            # 3. 통합 결과를 metadata에 저장 (나중에 TodoResult에 포함)
            context.set_dynamic("integrated_fk_summary", {
                "total_unique_fks": integrated_result.total_unique_fks,
                "single_fk_count": integrated_result.single_fk_count,
                "composite_fk_count": integrated_result.composite_fk_count,
                "hierarchy_fk_count": integrated_result.hierarchy_fk_count,
                "temporal_fk_count": integrated_result.temporal_fk_count,
                "by_confidence": integrated_result.by_confidence,
                "by_type": integrated_result.by_type,
            })

        except Exception as e:
            logger.warning(f"[v16.0] Integrated FK Detection failed: {e}")
            import traceback
            logger.debug(traceback.format_exc())

        self._report_progress(0.3, "Analyzing graph structure")

        # 3. 그래프 구축 및 분석 (Enhanced FK 결과 포함)
        # Enhanced FK를 value_overlaps로 변환
        enhanced_overlaps = list(context.value_overlaps or [])
        for fk in enhanced_fk_candidates:
            enhanced_overlaps.append({
                "source_table": fk.from_table,
                "source_column": fk.from_column,
                "target_table": fk.to_table,
                "target_column": fk.to_column,
                "jaccard": fk.value_inclusion,
                "overlap_count": int(fk.value_inclusion * 1000),
                "fk_score": fk.fk_score,
                "confidence": fk.confidence,
            })

        graph = self.graph_analyzer.build_relationship_graph(
            tables=tables,
            value_overlaps=enhanced_overlaps,
            foreign_keys=foreign_keys,
            indirect_fks=list(context.indirect_fks or []),  # v7.3: Bridge Table 경유 간접 FK
        )

        # 그래프 분석
        components = self.graph_analyzer.find_connected_components()
        bridges = self.graph_analyzer.find_bridges()
        table_relationships = self.graph_analyzer.detect_table_relationships()
        entity_clusters = self.graph_analyzer.find_entity_clusters()
        stats = self.graph_analyzer.get_statistics()

        self._report_progress(0.4, "Running Graph Embedding Analysis (v3.0)")

        # v3.0: GraphEmbeddingAnalyzer로 Node2Vec + Community Detection 실행
        graph_embedding_results = {}
        try:
            # 스키마 그래프 생성
            schema_graph = self.graph_embedding.build_schema_graph(
                tables=tables,
                foreign_keys=foreign_keys,
                value_overlaps=enhanced_overlaps,
            )

            # Node2Vec 임베딩 계산
            embeddings = self.graph_embedding.compute_embeddings(schema_graph)

            # 커뮤니티 탐지
            communities = self.graph_embedding.detect_communities(schema_graph)

            # 중심성 분석
            centrality = self.graph_embedding.analyze_centrality(schema_graph)

            # CentralityMetrics에서 상위 5개 노드 추출 (pagerank 기준)
            top_central_nodes = []
            if centrality and centrality.pagerank:
                sorted_nodes = sorted(centrality.pagerank.items(), key=lambda x: -x[1])[:5]
                top_central_nodes = [{"node": n, "pagerank": pr} for n, pr in sorted_nodes]

            graph_embedding_results = {
                "embeddings_computed": len(embeddings.node_embeddings) if embeddings and hasattr(embeddings, 'node_embeddings') else 0,
                "communities_found": len(communities) if communities else 0,
                "community_details": [
                    {"id": c.id, "tables": c.nodes[:5], "size": len(c.nodes)}
                    for c in (communities or [])[:5]
                ],
                "centrality_top5": top_central_nodes,
            }
            logger.info(f"GraphEmbeddingAnalyzer found {len(communities or [])} communities")
        except Exception as e:
            logger.warning(f"GraphEmbeddingAnalyzer failed: {e}")

        self._report_progress(0.5, "Requesting LLM interpretation")

        # v17.0: FK 신뢰도 보정
        calibrated_fk_summary = []
        for fk in enhanced_fk_candidates[:20]:
            raw_confidence = fk.confidence
            calibrated = self.calibrate_confidence(raw_confidence)
            calibrated_fk_summary.append({
                "from": f"{fk.from_table}.{fk.from_column}",
                "to": f"{fk.to_table}.{fk.to_column}",
                "score": round(fk.fk_score, 2),
                "raw_confidence": raw_confidence,
                "calibrated_confidence": calibrated["calibrated"],
            })

        # 4. LLM에게 해석 요청
        # Enhanced FK 결과 요약 (v17.0: calibrated confidence 포함)
        enhanced_fk_summary = calibrated_fk_summary if calibrated_fk_summary else [
            {
                "from": f"{fk.from_table}.{fk.from_column}",
                "to": f"{fk.to_table}.{fk.to_column}",
                "score": round(fk.fk_score, 2),
                "confidence": fk.confidence,
            }
            for fk in enhanced_fk_candidates[:20]
        ]

        graph_summary = {
            "statistics": stats,
            "connected_components": len(components),
            "component_sizes": [len(c) for c in components],
            "bridges": [
                {"from": b.from_node, "to": b.to_node, "type": b.edge_type.value}
                for b in bridges
            ],
            "table_relationships": table_relationships[:15],
            "entity_clusters": entity_clusters,
            "enhanced_fk_candidates": enhanced_fk_summary,  # v2.1
        }

        instruction = f"""I have analyzed the relationship graph between database tables.

GRAPH ANALYSIS RESULTS:
{json.dumps(graph_summary, indent=2)}

Please INTERPRET these results:
1. What do the connected components tell us about data silos?
2. Which bridges are critical for cross-silo integration?
3. What is the recommended data integration pathway?
4. Any concerning patterns (orphaned tables, circular refs)?

Respond with JSON:
```json
{{
    "data_silo_analysis": {{
        "silo_count": <int>,
        "silos": [
            {{
                "tables": ["table1", "table2"],
                "primary_entity": "<main entity>",
                "integration_status": "<isolated|connected|bridge>"
            }}
        ]
    }},
    "critical_bridges": [
        {{
            "from": "<table.column>",
            "to": "<table.column>",
            "importance": "<why critical>",
            "integration_action": "<what to do>"
        }}
    ],
    "entity_relationships": [
        {{
            "from_entity": "<entity>",
            "to_entity": "<entity>",
            "relationship_name": "<has_many|belongs_to|...>",
            "cardinality": "<1:1|1:N|N:N>",
            "join_path": "<table_a.col -> table_b.col>"
        }}
    ],
    "integration_pathway": {{
        "recommended_order": ["step1", "step2"],
        "challenges": ["challenge1"],
        "quick_wins": ["win1"]
    }},
    "summary": "<overall assessment>"
}}
```"""

        response = await self.call_llm(instruction, max_tokens=4000)
        interpretation = parse_llm_json(response, default={})

        self._report_progress(0.9, "Updating context")

        # Entity relationships를 컨텍스트에 저장
        concept_relationships = interpretation.get("entity_relationships", [])

        # v7.6: LLM 결과와 FK 후보를 병합 (폴백이 아닌 보강)
        # LLM이 일부 관계만 반환해도 FK 후보에서 추가 관계를 보강
        if enhanced_fk_candidates:
            # 이미 존재하는 관계 쌍 추적 (중복 방지)
            existing_pairs = set()
            for rel in concept_relationships:
                from_e = rel.get("from_entity", "").lower()
                to_e = rel.get("to_entity", "").lower()
                existing_pairs.add((from_e, to_e))
                existing_pairs.add((to_e, from_e))  # 양방향

            added_count = 0
            for fk in enhanced_fk_candidates:
                # v7.6: 신뢰도 높은 FK (medium 이상) 추가
                conf_str = fk.confidence if isinstance(fk.confidence, str) else ""
                conf_val = fk.fk_score if hasattr(fk, 'fk_score') else 0

                # 신뢰도 필터: certain/high이거나 fk_score >= 0.6
                is_high_confidence = (
                    conf_str in ("certain", "high") or
                    conf_val >= 0.6
                )

                if not is_high_confidence:
                    continue

                from_t = fk.from_table.lower()
                to_t = fk.to_table.lower()

                # 이미 존재하는 관계면 스킵
                if (from_t, to_t) in existing_pairs:
                    continue

                concept_relationships.append({
                    "from_entity": fk.from_table,
                    "to_entity": fk.to_table,
                    "relationship_name": "references",
                    "cardinality": "N:1",  # FK는 기본적으로 N:1
                    "join_path": f"{fk.from_table}.{fk.from_column} -> {fk.to_table}.{fk.to_column}",
                    "confidence": conf_val,
                    "source": "merged_from_fk_detection"
                })
                existing_pairs.add((from_t, to_t))
                added_count += 1

            if added_count > 0:
                logger.info(f"[v7.6 Merge] Added {added_count} relationships from FK candidates (total: {len(concept_relationships)})")

        # Enhanced FK를 직렬화 가능한 형태로 변환
        enhanced_fk_results = [
            {
                "source_table": fk.from_table,
                "source_column": fk.from_column,
                "target_table": fk.to_table,
                "target_column": fk.to_column,
                "fk_score": fk.fk_score,
                "value_inclusion": fk.value_inclusion,
                "confidence": fk.confidence,
            }
            for fk in enhanced_fk_candidates
        ]

        # === v11.0: Evidence Block 생성 ===
        # 관계 탐지 결과를 근거 체인에 저장
        for rel in concept_relationships[:5]:
            self.record_evidence(
                finding=f"엔티티 관계 발견: {rel.get('from_entity', 'Unknown')} → {rel.get('to_entity', 'Unknown')} ({rel.get('relationship_name', 'related')})",
                reasoning=f"그래프 분석 결과 - 카디널리티: {rel.get('cardinality', 'N/A')}, Join Path: {rel.get('join_path', 'N/A')}",
                conclusion=f"엔티티 간 {rel.get('cardinality', 'N/A')} 관계 확인됨",
                data_references=[rel.get("join_path", "")] if rel.get("join_path") else [],
                metrics={
                    "cardinality": rel.get("cardinality", "N/A"),
                },
                confidence=0.8,
                topics=[rel.get("from_entity", ""), rel.get("to_entity", ""), "relationship", "graph"],
            )

        # Bridge (크로스 사일로 연결) 근거 기록
        for bridge_info in interpretation.get("critical_bridges", [])[:3]:
            self.record_evidence(
                finding=f"크로스 사일로 브릿지 발견: {bridge_info.get('from', '')} ↔ {bridge_info.get('to', '')}",
                reasoning=f"중요도: {bridge_info.get('importance', 'N/A')}",
                conclusion=f"데이터 통합 액션: {bridge_info.get('integration_action', 'N/A')}",
                data_references=[bridge_info.get("from", ""), bridge_info.get("to", "")],
                confidence=0.85,
                topics=["bridge", "integration", "cross_silo"],
            )

        # v3.0: Business Insights 분석 (Discovery 단계에서 실행)
        # v14.0: DataAnalyst의 column_semantics를 활용하여 의미론적 품질 판단
        self._report_progress(0.9, "Generating business insights")
        business_insights_list = []
        try:
            from ..analysis.business_insights import BusinessInsightsAnalyzer
            # v14.0: LLM 클라이언트와 도메인 컨텍스트 전달
            domain_ctx = context.get_domain() if hasattr(context, 'get_domain') else None
            insights_analyzer = BusinessInsightsAnalyzer(
                domain_context=domain_ctx,
                llm_client=self.llm_client,
            )

            tables_for_insights = {
                name: {"columns": info.columns, "row_count": info.row_count}
                for name, info in context.tables.items()
            }
            # v22.1: 전체 데이터 로드 (샘플링 금지)
            sample_data = context.get_all_full_data()

            # v14.0: DataAnalyst가 분석한 column_semantics 가져오기
            column_semantics = context.get_dynamic("column_semantics", {})
            if column_semantics:
                logger.info(f"[v14.0] Using LLM-analyzed column_semantics in Discovery: {len(column_semantics)} columns")

            if sample_data:
                raw_insights = insights_analyzer.analyze(
                    tables=tables_for_insights,
                    data=sample_data,
                    column_semantics=column_semantics,  # v14.0: LLM 분석 결과 전달
                )
                # 직렬화 가능한 형태로 변환
                business_insights_list = [
                    {
                        "insight_id": i.insight_id,
                        "title": i.title,
                        "description": i.description,
                        "insight_type": i.insight_type.value if hasattr(i.insight_type, 'value') else str(i.insight_type),
                        "severity": i.severity.value if hasattr(i.severity, 'value') else str(i.severity),
                        "source_table": i.source_table,
                        "source_column": i.source_column,
                        "metric_value": i.metric_value,
                        "recommendations": i.recommendations,
                        "business_impact": i.business_impact,
                    }
                    for i in raw_insights
                ]
                logger.info(f"Business Insights Analysis generated {len(business_insights_list)} insights in Discovery phase")
        except Exception as e:
            logger.warning(f"Business Insights Analysis failed in Discovery: {e}")

        # v16.0: 통합 FK 결과를 context_updates에 병합
        integrated_fk_summary = context.get_dynamic("integrated_fk_summary", {})
        all_enhanced_fk_results = list(context.enhanced_fk_candidates)  # v16.0: context에서 직접 가져옴

        # === v22.0: semantic_searcher로 의미적 관계 보강 ===
        semantic_relationships = []
        try:
            semantic_searcher = self.get_v17_service("semantic_searcher")
            if semantic_searcher and concept_relationships:
                for rel in concept_relationships[:5]:  # 상위 5개 관계만
                    query = f"{rel.get('from_entity', '')} {rel.get('to_entity', '')} relationship"
                    similar = await semantic_searcher.search(query=query, limit=2)
                    if similar:
                        semantic_relationships.append({
                            "relationship": f"{rel.get('from_entity')} -> {rel.get('to_entity')}",
                            "semantic_context": [s.get("name", "") for s in similar[:2]],
                        })
                if semantic_relationships:
                    logger.info(f"[v22.0] SemanticSearcher enriched {len(semantic_relationships)} relationships")
        except Exception as e:
            logger.debug(f"[v22.0] SemanticSearcher skipped in RelationshipDetector: {e}")

        # v22.0: Agent Learning - 경험 기록
        self.record_experience(
            experience_type="relationship_inference",
            input_data={"task": "relationship_detection", "tables": list(context.tables.keys())},
            output_data={"relationships_detected": len(table_relationships), "components": len(components)},
            outcome="success",
            confidence=0.8,
        )

        return TodoResult(
            success=True,
            output={
                "components_found": len(components),
                "bridges_found": len(bridges),
                "relationships_detected": len(table_relationships),
                "enhanced_fk_candidates": len(enhanced_fk_candidates),  # v2.1
                "integrated_fk_total": integrated_fk_summary.get("total_unique_fks", 0),  # v16.0
                "business_insights_generated": len(business_insights_list),  # v3.0
                "semantic_enrichments": len(semantic_relationships),  # v22.0
            },
            context_updates={
                "concept_relationships": concept_relationships,
                "enhanced_fk_candidates": all_enhanced_fk_results,  # v16.0: 통합 결과
                "value_overlaps": enhanced_overlaps,  # v2.1: Enhanced 결과로 업데이트
                "business_insights": business_insights_list,  # v3.0
            },
            metadata={
                "graph_stats": stats,
                "table_relationships": table_relationships,
                "entity_clusters": entity_clusters,
                "llm_interpretation": interpretation,
                "enhanced_fk_summary": enhanced_fk_summary,  # v2.1
                "integrated_fk_summary": integrated_fk_summary,  # v16.0
            },
        )

    # ==========================================================================
    # v7.10: Post-Processing Filters for Enhanced FK Candidates
    # Problem A: 역방향 FK 필터링 (Cardinality 기반 방향 판단)
    # Problem B: 간접 연결 필터링 (Transitive Closure 제거)
    # ==========================================================================

    def _filter_reverse_enhanced_fks(
        self,
        candidates: List[Any],
        sample_data: Dict[str, Any],
    ) -> List[Any]:
        """
        v7.10: 역방향 FK 필터링 (Cardinality 기반 방향 판단)

        FK-PK 관계에서:
        - PK: unique_ratio ≈ 1.0 (모든 값 고유)
        - FK: unique_ratio < 1.0 (중복 허용)

        A→B와 B→A 둘 다 있으면 올바른 방향만 유지
        """
        if not candidates or not sample_data:
            return candidates

        # Step 1: 양방향 쌍 찾기
        pair_map = {}  # frozenset -> [candidates]
        for c in candidates:
            pair = frozenset([
                (c.from_table, c.from_column),
                (c.to_table, c.to_column)
            ])
            if pair not in pair_map:
                pair_map[pair] = []
            pair_map[pair].append(c)

        filtered = []
        processed_pairs = set()

        for c in candidates:
            pair = frozenset([
                (c.from_table, c.from_column),
                (c.to_table, c.to_column)
            ])

            if pair in processed_pairs:
                continue

            pair_candidates = pair_map[pair]

            if len(pair_candidates) == 1:
                # 단방향만 있음 - 그대로 유지
                filtered.append(c)
            else:
                # 양방향 존재 - Cardinality로 방향 판단
                best = self._determine_enhanced_fk_direction(pair_candidates, sample_data)
                if best:
                    filtered.append(best)

            processed_pairs.add(pair)

        removed = len(candidates) - len(filtered)
        logger.info(f"[v7.10] Reverse enhanced FK filter: {len(candidates)} → {len(filtered)} "
                   f"(removed {removed} reverse FKs)")

        return filtered

    def _determine_enhanced_fk_direction(
        self,
        pair_candidates: List[Any],
        sample_data: Dict[str, Any],
    ) -> Optional[Any]:
        """Cardinality 기반 Enhanced FK 방향 결정"""
        best_candidate = None
        best_score = -1

        for c in pair_candidates:
            # sample_data 구조: {table: [rows]} 형태
            # 각 row는 dict: {column: value}
            from_table_data = sample_data.get(c.from_table, [])
            to_table_data = sample_data.get(c.to_table, [])

            # 컬럼 값 추출
            if isinstance(from_table_data, list):
                from_values = [row.get(c.from_column) for row in from_table_data
                              if isinstance(row, dict) and c.from_column in row]
            else:
                from_values = []

            if isinstance(to_table_data, list):
                to_values = [row.get(c.to_column) for row in to_table_data
                            if isinstance(row, dict) and c.to_column in row]
            else:
                to_values = []

            if not from_values or not to_values:
                # 데이터 없으면 기존 점수 사용
                score = getattr(c, 'fk_score', 0.5)
                if score > best_score:
                    best_score = score
                    best_candidate = c
                continue

            # unique ratio 계산
            from_unique = len(set(from_values))
            to_unique = len(set(to_values))
            from_ratio = from_unique / len(from_values) if from_values else 0
            to_ratio = to_unique / len(to_values) if to_values else 0

            # FK → PK 방향이 올바름:
            # - to (PK)가 더 unique해야 함 (to_ratio > from_ratio)
            direction_score = to_ratio - from_ratio

            # PK는 거의 항상 unique (>0.9)
            if to_ratio > 0.9 and from_ratio < 0.9:
                direction_score += 0.5

            # 기존 fk_score와 결합
            total_score = direction_score + getattr(c, 'fk_score', 0.5)

            if total_score > best_score:
                best_score = total_score
                best_candidate = c

        return best_candidate

    def _filter_transitive_enhanced_fks(
        self,
        candidates: List[Any],
        confidence_threshold: float = 0.6,
    ) -> List[Any]:
        """
        v7.10: 간접 연결 필터링 (Transitive Closure 제거)

        A → C, B → C 가 있을 때 A → B 제거
        (같은 PK를 참조하는 FK들끼리의 연결)
        """
        if not candidates:
            return candidates

        # Step 1: 높은 confidence FK로 target → sources 매핑
        target_to_sources = {}

        for c in candidates:
            conf = getattr(c, 'fk_score', 0)
            if conf >= confidence_threshold:
                target_key = f"{c.to_table}.{c.to_column}"
                source_key = f"{c.from_table}.{c.from_column}"
                if target_key not in target_to_sources:
                    target_to_sources[target_key] = set()
                target_to_sources[target_key].add(source_key)

        # Step 2: 같은 target을 공유하는 source들 간의 연결 제거
        filtered = []
        removed_count = 0

        for c in candidates:
            source_key = f"{c.from_table}.{c.from_column}"
            target_key = f"{c.to_table}.{c.to_column}"

            is_transitive = False
            for common_target, sources in target_to_sources.items():
                # source와 target 모두 같은 common_target을 참조하면 간접 연결
                if source_key in sources and target_key in sources:
                    is_transitive = True
                    break

            if not is_transitive:
                filtered.append(c)
            else:
                removed_count += 1

        logger.info(f"[v7.10] Transitive enhanced FK filter: removed {removed_count} indirect connections")

        return filtered
