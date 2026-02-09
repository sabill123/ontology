"""
TDA Expert Autonomous Agent (v4.1 - Embedded Legacy)

Betti numbers 계산, Persistence diagrams, Advanced TDA (Ripser),
EmbeddedHomeomorphismAnalyzer를 통한 3단계 위상동형 검증 (구조+값+의미).
"""

import json
import logging
import time
from typing import Dict, Any, List, Optional, TYPE_CHECKING

from ...base import AutonomousAgent
from ...analysis import (
    TDAAnalyzer,
    AdvancedTDAAnalyzer,
    parse_llm_json,
)
from ..discovery_utils import EmbeddedHomeomorphismAnalyzer

if TYPE_CHECKING:
    from ....todo.models import PipelineTodo, TodoResult
    from ....shared_context import SharedContext

logger = logging.getLogger(__name__)


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
        from ....todo.models import TodoResult

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
        from ....todo.models import TodoResult

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
        from ....todo.models import TodoResult
        from ....shared_context import HomeomorphismPair

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
            if embedded_homeomorphisms_map:
                logger.info(f"Embedded results: {len(embedded_homeomorphisms_map)} pairs analyzed")
            else:
                # v24.0: 단일 테이블이면 debug로 격하
                logger.debug(f"Embedded results: 0 pairs (single table or no cross-table matches)")

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
