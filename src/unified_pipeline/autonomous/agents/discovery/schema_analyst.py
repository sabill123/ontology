"""
Schema Analyst Autonomous Agent (v2.0 - Algorithm-First)

스키마 프로파일링 + AnomalyDetector + LLM 해석.
"""

import json
import logging
from typing import Dict, Any, List, Optional, TYPE_CHECKING

from ...base import AutonomousAgent
from ...analysis import (
    SchemaAnalyzer,
    parse_llm_json,
)

if TYPE_CHECKING:
    from ....todo.models import PipelineTodo, TodoResult
    from ....shared_context import SharedContext

logger = logging.getLogger(__name__)


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
        from ....todo.models import TodoResult

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
