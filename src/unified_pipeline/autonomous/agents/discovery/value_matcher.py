"""
Value Matcher Autonomous Agent module.

Handles FK (Foreign Key) detection using universal domain-agnostic algorithms,
bridge table analysis, semantic column matching, and LLM-based decision making.
"""

import json
import logging
from typing import Dict, Any, List, Optional, TYPE_CHECKING

import pandas as pd
import numpy as np

from ...base import AutonomousAgent
from ...analysis import (
    SimilarityAnalyzer,
    AdvancedSimilarityAnalyzer,
    parse_llm_json,
    EnhancedValuePatternAnalysis,
    ColumnNormalizer,
    SemanticFKDetector,
)

if TYPE_CHECKING:
    from ....todo.models import PipelineTodo, TodoResult
    from ....shared_context import SharedContext

logger = logging.getLogger(__name__)


class ValueMatcherAutonomousAgent(AutonomousAgent):
    """
    값 매칭 자율 에이전트 (v6.0 - Universal Domain-Agnostic FK Detection)

    v6.0 변경사항:
    - 도메인 특화 룰베이스 제거
    - UniversalFKDetector: 범용 네이밍 정규화 + 값 기반 검증
    - LLM이 최종 FK 판단 (알고리즘은 계산만)
    - 어떤 도메인/네이밍 컨벤션이든 처리 가능
    """

    # v28.3: FK 후보 컬럼 프리필터 — O(N²×C²) → O(N²×C_fk²) 감소
    _FK_NAME_PATTERN = None  # lazy init
    _FK_MAX_COLS = 60  # 테이블당 최대 FK 후보 컬럼 수 (216→60: 13x 감소)

    @classmethod
    def _get_fk_pattern(cls):
        if cls._FK_NAME_PATTERN is None:
            import re
            cls._FK_NAME_PATTERN = re.compile(
                r'(?:_id|id_|\bid\b|_key|key_|\bkey\b|_code|code_|\bcode\b'
                r'|_ref|ref_|\bref\b|_fk|fk_|_num|_no\b|_nr\b|_uuid|_guid'
                r'|_type\b|type_|\btype\b)',
                re.IGNORECASE,
            )
        return cls._FK_NAME_PATTERN

    @classmethod
    def _filter_fk_columns(cls, tables_data: Dict[str, Dict]) -> Dict[str, Dict]:
        """
        v28.3: FK 후보 컬럼만 선별 (UniversalFKDetector 입력 축소)
        - FK 네이밍 패턴 컬럼 우선
        - Low-cardinality 컬럼 포함 (unique_ratio < 0.5)
        - 최대 _FK_MAX_COLS 컬럼/테이블
        정확도: FK는 거의 항상 위 패턴에 해당 → 누락 최소화
        """
        import re
        pattern = cls._get_fk_pattern()
        max_cols = cls._FK_MAX_COLS
        filtered = {}
        for table_name, col_values in tables_data.items():
            fk_cols = {}
            other_low_card = {}
            for col_name, values in col_values.items():
                if pattern.search(str(col_name)):
                    fk_cols[col_name] = values
                elif values:
                    unique_ratio = len(set(str(v) for v in values[:500])) / min(500, len(values))
                    if unique_ratio < 0.5:
                        other_low_card[col_name] = values
            # FK 패턴 컬럼 + low-cardinality 컬럼 (총 max_cols 이내)
            selected = {}
            for col_name, values in list(fk_cols.items())[:max_cols]:
                selected[col_name] = values
            remaining = max_cols - len(selected)
            for col_name, values in list(other_low_card.items())[:remaining]:
                selected[col_name] = values
            filtered[table_name] = selected
        return filtered

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.similarity_analyzer = SimilarityAnalyzer()
        # v3.0: MinHash, LSH, HyperLogLog 기반 고급 유사도 분석
        self.advanced_similarity = AdvancedSimilarityAnalyzer(
            num_perm=128,
            num_bands=16,
        )
        # v6.0: Universal FK Detection (도메인 무관)
        from ...analysis.universal_fk_detector import UniversalFKDetector
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
        from ....todo.models import TodoResult

        self._report_progress(0.1, "Extracting table data")

        # 1. 테이블별 컬럼명 및 값 추출
        # v28.3: pandas 벡터화 — 단일 패스로 tables_data + semantic_tables_data 동시 구축
        tables_columns = {}
        tables_data = {}
        _table_dfs: Dict[str, pd.DataFrame] = {}  # SemanticFKDetector 재사용용

        for table_name, table_info in context.tables.items():
            col_names = [col.get("name", "") for col in table_info.columns]
            tables_columns[table_name] = col_names

            # v22.1: 전체 데이터 로드 (샘플링 금지 — FK 값 매칭 정확도)
            # v28.3: Python 이중 루프 → pd.DataFrame 단일 변환 (100x 속도 향상)
            full_rows = context.get_full_data(table_name)
            if full_rows:
                df = pd.DataFrame(full_rows)
                _table_dfs[table_name] = df
                tables_data[table_name] = {
                    col: df[col].dropna().tolist()
                    for col in col_names
                    if col and col in df.columns
                }
            else:
                _table_dfs[table_name] = pd.DataFrame()
                tables_data[table_name] = {}

        self._report_progress(0.2, "Running Universal FK Detection (domain-agnostic)")

        # 2. v6.0: UniversalFKDetector로 범용 FK 후보 탐지
        # v28.3: FK 후보 컬럼 프리필터 — O(N²×C²) 4.9M → O(N²×C_fk²) 262K 감소 (19x)
        # 정확도: FK 컬럼은 거의 항상 _id/_key/_code 패턴 or low-cardinality
        tables_data_for_fk = self._filter_fk_columns(tables_data) if tables_data else None
        total_fk_cols = sum(len(v) for v in (tables_data_for_fk or {}).values())
        total_orig_cols = sum(len(v) for v in tables_data.values())
        logger.info(f"[v28.3] FK column pre-filter: {total_orig_cols} → {total_fk_cols} cols "
                    f"({total_fk_cols/max(total_orig_cols,1)*100:.0f}% retained)")
        fk_candidates = self.universal_fk_detector.detect_fk_candidates(
            tables_columns=tables_columns,
            tables_data=tables_data_for_fk,
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
            # v28.3: _table_dfs 재사용 — 중복 로딩 제거 (이미 pandas DataFrame 보유)
            semantic_tables_data = {}
            for table_name, table_info in context.tables.items():
                columns = [col.get("name", "") for col in table_info.columns]
                df = _table_dfs.get(table_name, pd.DataFrame())
                if not df.empty:
                    valid_cols = [c for c in columns if c and c in df.columns]
                    # v28.3: df.values.tolist() — Python 루프 없이 행 리스트 변환
                    semantic_tables_data[table_name] = {
                        "columns": valid_cols,
                        "data": df[valid_cols].where(df[valid_cols].notna(), None).values.tolist(),
                    }
                else:
                    semantic_tables_data[table_name] = {
                        "columns": columns,
                        "data": [],
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
        _MINHASH_MAX_VALS = 10_000  # v28.3: 10K cap — FK 검출 정확도 유지 (유니크 값 기반)
        try:
            for table_name, columns in tables_data.items():
                for col_name, values in columns.items():
                    if len(values) >= 10:
                        col_key = f"{table_name}.{col_name}"
                        # v28.3: 유니크 값만 MinHash에 필요 → cap at 10K (FK는 cardinality 기반)
                        unique_vals = set(str(v) for v in values)
                        if len(unique_vals) > _MINHASH_MAX_VALS:
                            unique_vals = set(list(unique_vals)[:_MINHASH_MAX_VALS])
                        self.advanced_similarity.add_minhash(col_key, unique_vals)

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
