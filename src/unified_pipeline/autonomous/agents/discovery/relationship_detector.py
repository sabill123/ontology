"""
RelationshipDetectorAutonomousAgent - Relationship Detection Autonomous Agent

Extracted from discovery.py for modular architecture.
Handles graph-based relationship detection between database tables.

v3.0: Full Graph Analysis (GraphAnalyzer, EnhancedFKDetector, GraphEmbeddingAnalyzer)
v5.0: FK False Positive Filter
v7.10: Post-Processing Filters (Reverse FK, Transitive Closure)
v16.0: Integrated FK Detection (Composite, Hierarchy, Temporal)
v17.0: FK Confidence Calibration
v22.0: Semantic Search Enrichment + Agent Learning
"""

import json
import logging
from typing import Dict, Any, List, Optional, TYPE_CHECKING

from ...base import AutonomousAgent
from ...analysis import (
    GraphAnalyzer,
    EnhancedFKDetector,
    GraphEmbeddingAnalyzer,
    FalsePositiveFilter,
    parse_llm_json,
    detect_all_fks_and_update_context,
    IntegratedFKResult,
)

if TYPE_CHECKING:
    from ....todo.models import PipelineTodo, TodoResult
    from ....shared_context import SharedContext

logger = logging.getLogger(__name__)


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
        from ....todo.models import TodoResult

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
        # v28.6: table_registry 캐시로 중복 CSV 로딩 제거
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
            from ...analysis.business_insights import BusinessInsightsAnalyzer
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

            # v28.6: 전체 데이터 사용 (캐시 재사용으로 CSV 중복 읽기 없음)
            if sample_data:
                # v28.15: pandas nullable dtype 안전 처리 (JSON round-trip)
                import json
                safe_data = json.loads(json.dumps(sample_data, default=str))

                raw_insights = insights_analyzer.analyze(
                    tables=tables_for_insights,
                    data=safe_data,
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
