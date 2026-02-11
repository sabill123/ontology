"""
Ontology Architect Autonomous Agent (v7.5.1 - Enhanced FK Integration)

온톨로지 설계 자율 에이전트 - OntologyAligner + Embedded Agent Council
- Discovery 결과에서 구조화된 온톨로지 개념 추출
- 계층 관계 자동 추론
- OntologyAlignmentAnalyzer로 크로스 도메인 매칭
- Embedded Agent Council (4개 에이전트 병렬 실행)
- LLM을 통한 설명 보강 및 검증
"""

import json
import logging
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from ...base import AutonomousAgent
from ...analysis import (
    parse_llm_json,
    OntologyAlignmentAnalyzer,
    CausalImpactAnalyzer,
    GrangerCausalityAnalyzer,
    VirtualEntityGenerator,
    generate_virtual_entities_for_context,
)

from ..refinement_utils import (
    EmbeddedOntologyAgentCouncil,
)

if TYPE_CHECKING:
    from ....todo.models import PipelineTodo, TodoResult
    from ....shared_context import SharedContext

logger = logging.getLogger(__name__)


class OntologyArchitectAutonomousAgent(AutonomousAgent):
    """
    온톨로지 설계 자율 에이전트 (v4.5 - Embedded Agent Council)

    Algorithm-First 접근:
    1. Discovery 결과에서 구조화된 온톨로지 개념 추출
    2. 계층 관계 자동 추론
    3. OntologyAlignmentAnalyzer로 크로스 도메인 매칭
    4. Embedded Agent Council (4개 에이전트 병렬 실행)
    5. LLM을 통한 설명 보강 및 검증

    v4.5 변경사항:
    - EmbeddedOntologyAgentCouncil 직접 사용 (더 이상 _legacy 폴더 의존 없음)
    - EntityOntologist, RelationshipArchitect, PropertyEngineer, DomainExpert 임베딩됨
    - Agent Council Pattern: 4개 에이전트의 인사이트 병합
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # v3.0: 온톨로지 정렬 분석기 통합
        self.ontology_aligner = OntologyAlignmentAnalyzer()
        # v4.5: Embedded Agent Council (임베딩됨 - 항상 사용 가능)
        self.embedded_agent_council_class = EmbeddedOntologyAgentCouncil
        # v6.1: Dynamic Causal Inference Analyzer
        self.causal_analyzer = CausalImpactAnalyzer()
        self.granger_analyzer = GrangerCausalityAnalyzer(max_lag=5)
        # v6.1: Virtual Entity Generator
        self.virtual_entity_generator = None  # 도메인에 따라 lazily initialized
        # 온톨로지 구조 매핑 규칙
        self.type_mappings = {
            "entity": "object_type",
            "table": "object_type",
            "attribute": "property",
            "column": "property",
            "relationship": "link_type",
            "rule": "constraint",
        }

    @property
    def agent_type(self) -> str:
        return "ontology_architect"

    @property
    def agent_name(self) -> str:
        return "Ontology Architect"

    @property
    def phase(self) -> str:
        return "refinement"

    def get_system_prompt(self) -> str:
        return """You are an Ontology Architect Expert specializing in knowledge graph design.

Your role is to ENHANCE and VALIDATE ontology concepts already extracted from data.

v5.2 Design Philosophy:
- TRUST THE DATA: Algorithmic extraction from real data is reliable
- ENHANCE, DON'T REJECT: Your job is to improve, not gatekeep
- ITERATIVE REFINEMENT: Initial concepts are starting points

Expertise:
1. **Concept Refinement**: Improve names, descriptions, and definitions
2. **Hierarchy Design**: Identify is-a, part-of relationships
3. **Semantic Enrichment**: Add business context and domain knowledge
4. **Quality Assurance**: Ensure naming conventions and completeness

Naming Conventions:
- Object Types: PascalCase singular (Patient, Encounter, Claim)
- Properties: camelCase (firstName, serviceDate, totalAmount)
- Link Types: verb_noun format (has_diagnosis, performed_by)

IMPORTANT:
- DEFAULT TO ACCEPTING concepts with data evidence
- Focus on IMPROVEMENT, not rejection
- Concepts from data are likely valid

Respond ONLY with valid JSON."""

    async def execute_task(
        self,
        todo: "PipelineTodo",
        context: "SharedContext",
    ) -> "TodoResult":
        """온톨로지 설계 작업 실행"""
        from ....todo.models import TodoResult
        from ....shared_context import OntologyConcept

        self._report_progress(0.1, "Extracting concepts from discovery results")

        # 1단계: Discovery 결과에서 구조화된 온톨로지 개념 추출 (알고리즘)
        extracted_concepts = self._extract_concepts_from_discovery(context)

        self._report_progress(0.20, f"Extracted {len(extracted_concepts)} raw concepts")

        # === v17.1: LLM 심층 개념 분석 (새로 추가) ===
        self._report_progress(0.22, "Running LLM Deep Concept Analysis (v17.1)")
        try:
            extracted_concepts = await self._llm_deep_concept_analysis(extracted_concepts, context)
            logger.info(f"[v17.1] LLM deep analysis enriched {len(extracted_concepts)} concepts")
        except Exception as e:
            logger.warning(f"LLM deep concept analysis failed: {e}")

        # 2단계: 계층 관계 추론 (알고리즘)
        hierarchy = self._infer_hierarchy(extracted_concepts, context)

        # === v17.1: LLM 의미론적 관계 추론 (새로 추가) ===
        self._report_progress(0.26, "Running LLM Semantic Relationship Inference (v17.1)")
        llm_relationships = []
        try:
            llm_relationships = await self._llm_infer_semantic_relationships(extracted_concepts, context)
            # LLM이 발견한 관계를 concepts에 추가
            extracted_concepts.extend(llm_relationships)
            logger.info(f"[v17.1] LLM inferred {len(llm_relationships)} additional relationships")
        except Exception as e:
            import traceback
            logger.warning(f"LLM relationship inference failed: {e}")
            logger.warning(f"[TRACEBACK] {traceback.format_exc()}")

        self._report_progress(0.30, "Running Ontology Alignment (v3.0)")

        # v3.0: OntologyAlignmentAnalyzer로 크로스 도메인 온톨로지 매칭
        alignment_results = {}
        try:
            if extracted_concepts:
                # 온톨로지 요소 생성
                source_elements = [
                    {
                        "id": c.get("concept_id"),
                        "name": c.get("name"),
                        "type": c.get("concept_type"),
                        "attributes": c.get("definition", {}).get("attributes", []),
                    }
                    for c in extracted_concepts
                ]

                # 정렬 분석 실행
                alignments = self.ontology_aligner.find_alignments(source_elements)
                alignment_results = {
                    "alignments_found": len(alignments) if alignments else 0,
                    "alignment_details": alignments[:10] if alignments else [],
                }
                logger.info(f"OntologyAligner found {len(alignments or [])} alignments")
        except Exception as e:
            logger.warning(f"OntologyAlignmentAnalyzer failed: {e}")

        # === v4.5: Embedded Agent Council 실행 (4개 에이전트 병렬) ===
        embedded_insights = []
        self._report_progress(0.40, "Running Embedded Agent Council (4 agents parallel)")
        try:
            # Domain context 구성
            domain_context = {
                "industry": context.get_industry() or "general",
                "department": "operations",
                "key_entities": [e.canonical_name for e in (context.unified_entities or [])[:10]],
                "key_metrics": [],
            }

            # Embedded Agent Council 생성 (v4.5)
            embedded_council = self.embedded_agent_council_class(
                domain_context=domain_context,
                parallel_execution=True,
            )

            # 데이터 요약 생성
            data_summary = [
                {
                    "table_name": name,
                    "columns": [c.get("name", "") for c in info.columns[:10]],
                    "row_count": info.row_count,
                }
                for name, info in context.tables.items()
            ]

            # 매핑 정보 수집
            mappings = []
            for h in (context.homeomorphisms or []):
                if h.is_homeomorphic:
                    mappings.append({
                        "id": f"MAP-{len(mappings)+1:03d}",
                        "table_a": h.table_a,
                        "table_b": h.table_b,
                        "join_keys": h.join_keys if hasattr(h, 'join_keys') else [],
                        "confidence": h.confidence,
                    })

            # Canonical objects (unified entities)
            canonical_objects = [
                {
                    "entity_id": e.entity_id,
                    "canonical_name": e.canonical_name,
                    "entity_type": e.entity_type,
                    "source_tables": e.source_tables,
                }
                for e in (context.unified_entities or [])
            ]

            # Embedded Agent Council 실행 (4개 에이전트 병렬)
            embedded_insights = embedded_council.generate_ontology_insights(
                data_summary=data_summary,
                mappings=mappings,
                canonical_objects=canonical_objects,
                phase1_mappings_used=[m["id"] for m in mappings[:5]],
            )

            logger.info(f"Embedded Agent Council generated {len(embedded_insights)} insights")
        except Exception as e:
            logger.warning(f"Embedded Agent Council failed: {e}")

        # === v6.1: Dynamic Causal Inference Analysis ===
        causal_relationships = []
        causal_insights = {}
        palantir_insights = []  # v10.0: Palantir-style insights
        self._report_progress(0.48, "Running Dynamic Causal Inference (v6.1) + Cross-Entity Correlation (v10.0)")
        try:
            causal_results = self._run_causal_inference(context, extracted_concepts)
            causal_relationships = causal_results.get("causal_relationships", [])
            causal_insights = causal_results.get("insights", {})
            palantir_insights = causal_results.get("palantir_insights", [])  # v10.0
            logger.info(f"Causal Inference discovered {len(causal_relationships)} relationships, {len(palantir_insights)} Palantir insights")
        except Exception as e:
            logger.warning(f"Causal Inference analysis failed: {e}")

        # === v6.1: Virtual Entity Generation ===
        virtual_entities = []
        self._report_progress(0.52, "Generating Virtual Scenario Entities (v6.1)")
        try:
            virtual_entities = self._generate_virtual_entities(context, extracted_concepts)
            logger.info(f"Generated {len(virtual_entities)} virtual entities")
        except Exception as e:
            logger.warning(f"Virtual Entity generation failed: {e}")

        self._report_progress(0.55, "Enhancing concepts with LLM")

        # v27.1: LLM에 의존하지 않는 속성 인덱스 — concept_id → extracted_attributes
        # LLM enhancement가 속성을 유실해도 원본을 반드시 복원
        _attr_index: Dict[str, list] = {}
        for c in extracted_concepts:
            cid = c.get("concept_id", "")
            attrs = c.get("definition", {}).get("extracted_attributes", [])
            if cid and attrs:
                _attr_index[cid] = attrs

        # 3단계: LLM을 통한 개념 보강
        try:
            enhanced_concepts = await self._enhance_concepts_with_llm(
                extracted_concepts, hierarchy, context
            )
        except Exception as e:
            logger.warning(f"LLM concept enhancement failed, using raw concepts: {e}")
            enhanced_concepts = extracted_concepts

        self._report_progress(0.8, "Creating ontology concept objects")

        # 4단계: OntologyConcept 객체 생성
        # v25.2: TDA 시그니처 로드 (confidence 보정에 사용)
        tda_sigs = context.tda_signatures or {}
        ontology_concepts = []
        for c in enhanced_concepts:
            # v25.2: TDA 시그니처 기반 confidence 보정
            tda_boost = 0.0
            for src_table in c.get("source_tables", []):
                sig = tda_sigs.get(src_table, {})
                if isinstance(sig, dict) and sig.get("betti_numbers"):
                    betti = sig["betti_numbers"]
                    # betti_numbers는 list 또는 dict일 수 있음
                    b0 = betti[0] if isinstance(betti, list) and len(betti) > 0 else betti.get(0, betti.get("0", 0)) if isinstance(betti, dict) else 0
                    if b0 and b0 > 0:
                        tda_boost = max(tda_boost, min(0.1, b0 * 0.02))

            raw_confidence = c.get("confidence", 0.0)
            adjusted_confidence = min(1.0, raw_confidence + tda_boost)

            cid = c.get("concept_id", f"concept_{len(ontology_concepts)}")
            # v27.1: LLM 결과가 아닌 원본 인덱스에서 속성 복원
            original_attrs = _attr_index.get(cid, [])
            llm_attrs = c.get("definition", {}).get("extracted_attributes", [])
            final_attrs = original_attrs if original_attrs else llm_attrs

            concept = OntologyConcept(
                concept_id=cid,
                concept_type=c.get("concept_type", "object_type"),
                name=c.get("name", ""),
                description=c.get("description", ""),
                definition=c.get("definition", {}),
                source_tables=c.get("source_tables", []),
                source_evidence=c.get("source_evidence", {}),
                status="pending",
                confidence=adjusted_confidence,
                source_agent="ontology_architect",
                unified_attributes=final_attrs,
            )
            ontology_concepts.append(concept)

        # v27.0: 계층 정보를 concept.parent_concept에 반영
        for parent_name, children in hierarchy.items():
            for concept in ontology_concepts:
                if concept.name in children and concept.concept_type == "object_type":
                    concept.parent_concept = parent_name

        self._report_progress(0.95, "Finalizing ontology design")

        # === v11.0: Evidence Block 생성 - Phase 1 근거 참조 ===
        # Phase 1의 근거를 조회하여 참조
        phase1_evidence = self.get_evidence_for_reasoning(
            phases=["phase1_discovery"],
            min_confidence=0.5
        )
        supporting_block_ids = [e["block_id"] for e in phase1_evidence[:5]]

        # 온톨로지 설계 결과를 근거로 기록
        for concept in ontology_concepts[:10]:
            self.record_evidence(
                finding=f"온톨로지 개념 설계: {concept.name} (유형: {concept.concept_type})",
                reasoning=f"Phase 1 발견 결과에서 도출. 소스 테이블: {concept.source_tables}",
                conclusion=f"'{concept.name}' 개념으로 {len(concept.source_tables)}개 테이블 데이터 구조화 가능",
                evidence_type="ontology_design",
                data_references=[f"ontology.{concept.concept_id}"],
                metrics={
                    "confidence": concept.confidence,
                    "source_table_count": len(concept.source_tables),
                },
                supporting_block_ids=supporting_block_ids,  # Phase 1 근거 참조
                confidence=concept.confidence,
                topics=[concept.name, concept.concept_type, "ontology"],
            )

        # Causal insight를 근거로 기록
        for insight in palantir_insights[:5]:
            self.record_evidence(
                finding=f"예측 인사이트: {insight.get('predicted_impact', 'N/A')[:80]}",
                reasoning=f"상관계수: {insight.get('correlation_coefficient', 0):.2f}, p-value: {insight.get('p_value', 1):.4f}",
                conclusion=insight.get("business_interpretation", "N/A")[:150],
                evidence_type="causal_inference",
                data_references=[f"{insight.get('source_table', '')}.{insight.get('source_column', '')}"],
                metrics={
                    "correlation": insight.get("correlation_coefficient", 0),
                    "p_value": insight.get("p_value", 1),
                    "confidence": insight.get("confidence", 0),
                },
                supporting_block_ids=supporting_block_ids,
                confidence=insight.get("confidence", 0.5),
                topics=["causal", "prediction", "simulation"],
            )

        # 결과 집계
        by_type = {}
        for c in ontology_concepts:
            ct = c.concept_type
            by_type[ct] = by_type.get(ct, 0) + 1

        # v4.3: Knowledge Graph 트리플 생성
        kg_triples = self._generate_ontology_triples(ontology_concepts, hierarchy, context)
        logger.info(f"Generated {len(kg_triples)} knowledge graph triples from ontology concepts")

        # === v17.1: OSDK 연동 - 온톨로지 개념을 OSDK Object Types로 동기화 ===
        osdk_sync_results = {"synced_types": 0, "synced_links": 0}
        try:
            osdk_client = self.get_v17_service("osdk_client")
            if osdk_client:
                # Object Types 동기화
                for concept in ontology_concepts:
                    if concept.concept_type == "object_type":
                        # 속성 추출
                        properties = []
                        if concept.definition and isinstance(concept.definition, dict):
                            for attr in concept.definition.get("attributes", [])[:20]:
                                prop_name = attr.get("name", "") if isinstance(attr, dict) else str(attr)
                                prop_type = attr.get("type", "string") if isinstance(attr, dict) else "string"
                                properties.append({"name": prop_name, "type": prop_type})

                        osdk_client.create_object_type(
                            type_id=concept.concept_id,
                            name=concept.name,
                            description=concept.description,
                            properties=properties,
                        )
                        osdk_sync_results["synced_types"] += 1

                    elif concept.concept_type == "link_type":
                        osdk_client.create_link_type(
                            link_id=concept.concept_id,
                            name=concept.name,
                            source_type=concept.definition.get("from_concept", ""),
                            target_type=concept.definition.get("to_concept", ""),
                            cardinality=concept.definition.get("cardinality", "many_to_one"),
                        )
                        osdk_sync_results["synced_links"] += 1

                logger.info(
                    f"[v17.1] OSDK sync complete: {osdk_sync_results['synced_types']} object types, "
                    f"{osdk_sync_results['synced_links']} link types"
                )
        except Exception as e:
            logger.debug(f"[v17.1] OSDK sync skipped: {e}")

        # === v22.0: decision_explainer로 온톨로지 설계 결정 설명 생성 ===
        design_explanations = []
        try:
            decision_explainer = self.get_v17_service("decision_explainer")
            if decision_explainer and ontology_concepts:
                for concept in ontology_concepts[:5]:  # 상위 5개 개념만
                    explanation = await decision_explainer.explain(
                        decision_type="ontology_design",
                        decision_data={
                            "concept_id": concept.concept_id,
                            "concept_type": concept.concept_type,
                            "name": concept.name,
                            "source_tables": concept.source_tables,
                            "confidence": concept.confidence,
                        },
                        context={"phase": "refinement", "agent": self.agent_name},
                    )
                    if explanation:
                        design_explanations.append({
                            "concept_id": concept.concept_id,
                            "explanation": explanation.get("summary", ""),
                            "design_rationale": explanation.get("rationale", ""),
                        })
                if design_explanations:
                    logger.info(f"[v22.0] DecisionExplainer generated {len(design_explanations)} design explanations")
        except Exception as e:
            logger.debug(f"[v22.0] DecisionExplainer skipped: {e}")

        # v22.0: Agent Learning - 경험 기록
        self.record_experience(
            experience_type="entity_mapping",
            input_data={"task": "ontology_design", "concepts_input": len(extracted_concepts)},
            output_data={"concepts_proposed": len(ontology_concepts), "hierarchy_levels": len(hierarchy)},
            outcome="success",
            confidence=0.85,
        )

        return TodoResult(
            success=True,
            output={
                "concepts_proposed": len(ontology_concepts),
                "design_explanations": len(design_explanations),  # v22.0
                "by_type": by_type,
                "hierarchy_levels": len(hierarchy),
                "embedded_insights_generated": len(embedded_insights),  # v4.5
                "kg_triples_generated": len(kg_triples),  # v4.3
                "causal_relationships_discovered": len(causal_relationships),  # v6.1
                "virtual_entities_generated": len(virtual_entities),  # v6.1
                "palantir_insights_generated": len(palantir_insights),  # v10.0
            },
            context_updates={
                "ontology_concepts": ontology_concepts,
                "ontology_insights": embedded_insights,  # v4.5: Embedded Agent Council 인사이트
                "knowledge_graph_triples": kg_triples,  # v4.3: Knowledge Graph 트리플
                "causal_relationships": causal_relationships,  # v6.1: 인과 관계
                "causal_insights": causal_insights,  # v6.1: 인과 인사이트
                "virtual_entities": virtual_entities,  # v6.1: 가상 엔티티
                "palantir_insights": palantir_insights,  # v10.0: 팔란티어 스타일 예측 인사이트
                "confounding_variables": causal_insights.get("confounding_variables", []),  # v26.1
                "derived_columns": causal_insights.get("derived_columns", []),  # v26.1
                "concept_relationships": [
                    {
                        "source": c.definition.get("source", "") if isinstance(c.definition, dict) else "",
                        "target": c.definition.get("target", "") if isinstance(c.definition, dict) else "",
                        "relationship_type": c.definition.get("relationship_type", "related_to") if isinstance(c.definition, dict) else "related_to",
                        "name": c.name,
                        "concept_id": c.concept_id,
                        "confidence": c.confidence,
                    }
                    for c in ontology_concepts if c.concept_type == "link_type"
                ],
            },
            metadata={
                "raw_concepts": len(extracted_concepts),
                "enhanced_concepts": len(enhanced_concepts),
                "embedded_agent_council_insights": embedded_insights,  # v4.5
                "causal_analysis": causal_insights,  # v6.1
                "virtual_entity_details": virtual_entities,  # v6.1
                "palantir_insights_details": palantir_insights,  # v10.0
            },
        )

    def _extract_concepts_from_discovery(
        self,
        context: "SharedContext",
    ) -> List[Dict[str, Any]]:
        """Discovery 결과에서 개념 추출 (알고리즘) - v5.1 Enhanced, v17.1 Semantic Integration"""
        concepts = []
        seen_names = set()

        # === v17.1: Phase 1 의미 분석 데이터 통합 ===
        # DataAnalyst가 생성한 풍부한 의미 정보 로드
        column_semantics = context.get_dynamic("column_semantics", {})
        data_patterns = context.get_dynamic("data_patterns", [])
        operational_insights = context.get_dynamic("operational_insights", {})
        extracted_entities_p1 = context.extracted_entities or []

        # v22.0: Phase 1 누수 데이터 통합 — TDA/Schema/CrossTable
        tda_signatures = context.tda_signatures or {}
        schema_analysis_data = context.schema_analysis or {}
        cross_table_mappings = context.cross_table_mappings or []

        logger.info(
            f"[v17.1] Phase 1 semantic data: {len(column_semantics)} column semantics, "
            f"{len(data_patterns)} patterns, {len(extracted_entities_p1)} extracted entities, "
            f"{len(tda_signatures)} TDA signatures, {len(schema_analysis_data)} schema analyses, "
            f"{len(cross_table_mappings)} cross-table mappings"
        )

        # v27.0: 테이블명→엔티티명 매핑 구축 (관계 해결용)
        table_to_entity: Dict[str, str] = {}
        for entity in (context.unified_entities or []):
            for table_name in entity.source_tables:
                table_to_entity[table_name] = self._to_pascal_case(entity.canonical_name)

        # 1. Unified Entities에서 Object Types 추출
        for entity in context.unified_entities:
            name = self._to_pascal_case(entity.canonical_name)
            if name.lower() in seen_names:
                continue
            seen_names.add(name.lower())

            # === v5.1: 자동 속성 추출 강화 ===
            # source_tables에서 컬럼 정보를 가져와서 속성으로 매핑
            extracted_attributes = []
            for table_name in entity.source_tables:
                table_info = context.tables.get(table_name)
                if table_info and hasattr(table_info, 'columns'):
                    for col in table_info.columns[:15]:  # 테이블당 최대 15개 속성
                        col_name = col.get("name", "") if isinstance(col, dict) else str(col)
                        col_type = col.get("type", "string") if isinstance(col, dict) else "string"

                        # 키 컬럼은 속성에서 제외 (이미 key_mappings에 있음)
                        is_key = any(key_col.lower() == col_name.lower()
                                    for key_col in entity.key_mappings.values() if key_col)
                        if is_key:
                            continue

                        # 속성 정보 수집
                        attr_info = {
                            "name": col_name,
                            "type": self._infer_semantic_type(col_name, col_type),
                            "source_table": table_name,
                            "data_type": col_type,
                            "nullable": col.get("nullable", True) if isinstance(col, dict) else True,
                        }

                        # === v17.1: column_semantics에서 비즈니스 역할 추가 ===
                        col_key = f"{table_name}.{col_name}"
                        if col_key in column_semantics:
                            sem = column_semantics[col_key]
                            attr_info["business_role"] = sem.get("business_role", "unknown")
                            attr_info["nullable_by_design"] = sem.get("nullable_by_design", True)
                            attr_info["semantic_category"] = sem.get("category", "data")
                            attr_info["importance"] = sem.get("importance", "medium")
                            # 필수 입력 필드인 경우 nullable=False로 보정
                            if sem.get("business_role") == "required_input":
                                attr_info["nullable"] = False

                        # 중복 방지
                        if not any(a["name"].lower() == col_name.lower() for a in extracted_attributes):
                            extracted_attributes.append(attr_info)

            # unified_attributes가 비어있으면 추출된 속성 사용
            if not entity.unified_attributes or len(entity.unified_attributes) == 0:
                attr_names = [a["name"] for a in extracted_attributes]
            else:
                attr_names = []
                for attr in entity.unified_attributes[:20]:
                    if isinstance(attr, dict):
                        attr_names.append(attr.get("name", attr.get("column", "unknown")))
                    else:
                        attr_names.append(str(attr))

            concepts.append({
                "concept_id": f"obj_{entity.entity_id}",
                "concept_type": "object_type",
                "name": name,
                "description": f"Entity representing {entity.entity_type}: {entity.canonical_name}",
                "source_tables": entity.source_tables,
                "source_evidence": {
                    "match_rate": entity.confidence,
                    "confidence": entity.confidence,
                    "source_count": len(entity.source_tables),
                },
                "confidence": entity.confidence,
                "definition": {
                    "attributes": attr_names if attr_names else [a["name"] for a in extracted_attributes],
                    "primary_keys": list(entity.key_mappings.keys())[:5],
                    "extracted_attributes": extracted_attributes[:20],  # v5.1: 상세 속성 정보
                },
            })

            # 속성들도 Property로 추출
            for attr in entity.unified_attributes[:20]:
                if isinstance(attr, dict):
                    attr_name = attr.get("name", attr.get("column", "unknown"))
                    attr_info = attr
                else:
                    attr_name = str(attr)
                    attr_info = {}
                prop_name = self._to_camel_case(attr_name)
                if prop_name.lower() in seen_names:
                    continue
                seen_names.add(prop_name.lower())

                concepts.append({
                    "concept_id": f"prop_{entity.entity_id}_{attr_name}",
                    "concept_type": "property",
                    "name": prop_name,
                    "description": f"Property of {name}: {attr_name}",
                    "source_tables": entity.source_tables[:1],
                    "source_evidence": {"confidence": 0.7},
                    "confidence": 0.7,
                    "definition": {
                        "belongs_to": name,
                        "data_type": attr_info.get("type", "string") if isinstance(attr_info, dict) else "string",
                    },
                })

        # 2. Homeomorphisms에서 Link Types 추출
        for homeo in context.homeomorphisms:
            if not homeo.is_homeomorphic:
                continue

            # === v5.1: 비즈니스 의미 기반 Link 이름 생성 ===
            link_name = self._generate_business_link_name(
                homeo.table_a,
                homeo.table_b,
                homeo.join_keys
            )
            if link_name.lower() in seen_names:
                # 중복 시 테이블 이름 포함하여 고유하게
                link_name = f"{link_name}_{homeo.table_a.split('_')[-1]}"
            seen_names.add(link_name.lower())

            # v5.1: 관계 설명도 더 의미있게
            relation_desc = self._generate_link_description(homeo)

            concepts.append({
                "concept_id": f"link_{homeo.table_a}_{homeo.table_b}",
                "concept_type": "link_type",
                "name": link_name,
                "description": relation_desc,
                "source_tables": [homeo.table_a, homeo.table_b],
                "source_evidence": {
                    "homeomorphism_type": homeo.homeomorphism_type,
                    "confidence": homeo.confidence,
                },
                "confidence": homeo.confidence,
                "definition": {
                    "source": table_to_entity.get(homeo.table_a, homeo.table_a),
                    "target": table_to_entity.get(homeo.table_b, homeo.table_b),
                    "join_keys": homeo.join_keys,
                    "source_table": homeo.table_a,
                    "target_table": homeo.table_b,
                },
            })

        # === v7.5: Enhanced FK Candidates에서 Link Types 추출 ===
        # UniversalFKDetector가 탐지한 FK 관계를 Link Type으로 변환
        # 기존 homeomorphisms에서 누락된 관계를 보완
        for fk in (context.enhanced_fk_candidates or []):
            # v7.5: 타입 안전성 보장 (str/None → float 변환)
            try:
                fk_confidence = float(fk.get("confidence", 0) or 0)
            except (ValueError, TypeError):
                fk_confidence = 0.0
            if fk_confidence < 0.5:
                continue

            from_table = fk.get("source_table", "")
            to_table = fk.get("target_table", "")
            from_column = fk.get("source_column", "")
            to_column = fk.get("target_column", "")

            if not all([from_table, to_table, from_column, to_column]):
                continue

            # 중복 체크: 이미 homeomorphisms에서 처리된 관계인지 확인
            link_key = f"{from_table}_{to_table}".lower()
            reverse_key = f"{to_table}_{from_table}".lower()
            if link_key in seen_names or reverse_key in seen_names:
                continue

            # v7.5: 비즈니스 의미 기반 Link 이름 생성
            link_name = self._generate_business_link_name(
                from_table,
                to_table,
                [(from_column, to_column)]
            )
            if link_name.lower() in seen_names:
                link_name = f"{link_name}_{from_table.split('_')[-1]}"
            seen_names.add(link_name.lower())
            seen_names.add(link_key)

            # FK 관계 설명 생성 (타입 안전성 보장)
            try:
                fk_score = float(fk.get("fk_score", 0) or 0)
            except (ValueError, TypeError):
                fk_score = 0.0
            try:
                value_inclusion = float(fk.get("value_inclusion", 0) or 0)
            except (ValueError, TypeError):
                value_inclusion = 0.0
            relation_desc = (
                f"Foreign key relationship from {from_table}.{from_column} to {to_table}.{to_column}. "
                f"FK Score: {fk_score:.2f}, Value Inclusion: {value_inclusion:.1%}"
            )

            concepts.append({
                "concept_id": f"link_fk_{from_table}_{to_table}",
                "concept_type": "link_type",
                "name": link_name,
                "description": relation_desc,
                "source_tables": [from_table, to_table],
                "source_evidence": {
                    "fk_type": "enhanced_fk_candidate",
                    "fk_score": fk_score,
                    "value_inclusion": value_inclusion,
                    "confidence": fk_confidence,
                    "detection_method": "universal_fk_detector_v7.5",
                },
                "confidence": fk_confidence,
                "definition": {
                    "source": table_to_entity.get(from_table, from_table),
                    "target": table_to_entity.get(to_table, to_table),
                    "join_keys": [(from_column, to_column)],
                    "source_table": from_table,
                    "target_table": to_table,
                },
            })

        # === v7.5.1: value_overlaps (UniversalFKDetector 결과)에서 Link Types 추출 ===
        # UniversalFKDetector가 탐지한 FK 관계 (value_overlaps에 저장됨)
        for vo in (context.value_overlaps or []):
            # value_overlaps 구조: {table_a, column_a, table_b, column_b, confidence, ...}
            try:
                vo_confidence = float(vo.get("confidence", 0) or 0)
            except (ValueError, TypeError):
                vo_confidence = 0.0
            if vo_confidence < 0.5:
                continue

            from_table = vo.get("table_a", "")
            to_table = vo.get("table_b", "")
            from_column = vo.get("column_a", "")
            to_column = vo.get("column_b", "")

            if not all([from_table, to_table, from_column, to_column]):
                continue

            # 중복 체크
            link_key = f"{from_table}_{to_table}".lower()
            reverse_key = f"{to_table}_{from_table}".lower()
            if link_key in seen_names or reverse_key in seen_names:
                continue

            # v7.5.1: 비즈니스 의미 기반 Link 이름 생성
            link_name = self._generate_business_link_name(
                from_table,
                to_table,
                [(from_column, to_column)]
            )
            if link_name.lower() in seen_names:
                link_name = f"{link_name}_{from_table.split('_')[-1]}"
            seen_names.add(link_name.lower())
            seen_names.add(link_key)

            # FK 관계 설명 생성
            detection_method = vo.get("detection_method", "universal_fk_detector")
            relation_desc = (
                f"FK relationship from {from_table}.{from_column} to {to_table}.{to_column}. "
                f"Detection: {detection_method}, Confidence: {vo_confidence:.2f}"
            )

            concepts.append({
                "concept_id": f"link_vo_{from_table}_{to_table}",
                "concept_type": "link_type",
                "name": link_name,
                "description": relation_desc,
                "source_tables": [from_table, to_table],
                "source_evidence": {
                    "fk_type": "value_overlap",
                    "confidence": vo_confidence,
                    "detection_method": detection_method,
                },
                "confidence": vo_confidence,
                "definition": {
                    "source": table_to_entity.get(from_table, from_table),
                    "target": table_to_entity.get(to_table, to_table),
                    "join_keys": [(from_column, to_column)],
                    "source_table": from_table,
                    "target_table": to_table,
                },
            })

        # === v17.1: extracted_entities에서 추가 Object Types 추출 ===
        # Phase 1에서 LLM이 추출한 엔티티 (unified_entities와 다를 수 있음)
        for ext_entity in (context.extracted_entities or []):
            if hasattr(ext_entity, 'name'):
                name = self._to_pascal_case(ext_entity.name)
                entity_type = getattr(ext_entity, 'entity_type', 'Entity')
                source_tables = getattr(ext_entity, 'source_tables', [])
                confidence = getattr(ext_entity, 'confidence', 0.6)
            elif isinstance(ext_entity, dict):
                name = self._to_pascal_case(ext_entity.get("name", ""))
                entity_type = ext_entity.get("entity_type", "Entity")
                source_tables = ext_entity.get("source_tables", [])
                confidence = ext_entity.get("confidence", 0.6)
            else:
                continue

            if not name or name.lower() in seen_names:
                continue
            seen_names.add(name.lower())

            concepts.append({
                "concept_id": f"obj_ext_{name.lower()}",
                "concept_type": "object_type",
                "name": name,
                "description": f"LLM-extracted entity: {entity_type}",
                "source_tables": source_tables if source_tables else [],
                "source_evidence": {
                    "extraction_method": "llm_entity_extraction",
                    "confidence": confidence,
                },
                "confidence": confidence,
                "definition": {
                    "entity_type": entity_type,
                    "attributes": [],
                },
            })

        # === v17.1: indirect_fks에서 Bridge Table 경유 관계 추출 ===
        # Phase 1에서 탐지된 간접 FK 관계 (A→Bridge→B)
        for indirect in (context.indirect_fks or []):
            if isinstance(indirect, dict):
                from_table = indirect.get("from_table", indirect.get("source_table", ""))
                to_table = indirect.get("to_table", indirect.get("target_table", ""))
                bridge_table = indirect.get("bridge_table", indirect.get("via_table", ""))
                confidence = indirect.get("confidence", 0.6)
            else:
                continue

            if not all([from_table, to_table]):
                continue

            # 중복 체크
            link_key = f"{from_table}_{to_table}".lower()
            if link_key in seen_names:
                continue

            # 간접 관계 이름 생성
            if bridge_table:
                link_name = f"relates_via_{bridge_table.split('_')[-1]}"
            else:
                link_name = self._generate_business_link_name(from_table, to_table, [])

            if link_name.lower() in seen_names:
                link_name = f"{link_name}_{from_table.split('_')[-1]}"
            seen_names.add(link_name.lower())
            seen_names.add(link_key)

            concepts.append({
                "concept_id": f"link_indirect_{from_table}_{to_table}",
                "concept_type": "link_type",
                "name": link_name,
                "description": f"Indirect relationship from {from_table} to {to_table}" +
                              (f" via bridge table {bridge_table}" if bridge_table else ""),
                "source_tables": [from_table, to_table] + ([bridge_table] if bridge_table else []),
                "source_evidence": {
                    "fk_type": "indirect_via_bridge",
                    "bridge_table": bridge_table,
                    "confidence": confidence,
                },
                "confidence": confidence,
                "definition": {
                    "source": from_table,
                    "target": to_table,
                    "bridge": bridge_table,
                    "relationship_type": "many-to-many" if bridge_table else "indirect",
                },
            })

        # === v17.1: cross_table_mappings에서 추가 관계 추출 ===
        for mapping in (context.cross_table_mappings or []):
            if isinstance(mapping, dict):
                table_a = mapping.get("table_a", mapping.get("source_table", ""))
                table_b = mapping.get("table_b", mapping.get("target_table", ""))
                mapping_type = mapping.get("mapping_type", mapping.get("type", "cross_reference"))
                confidence = mapping.get("confidence", 0.5)
            else:
                continue

            if not all([table_a, table_b]) or confidence < 0.4:
                continue

            link_key = f"{table_a}_{table_b}".lower()
            if link_key in seen_names:
                continue

            link_name = f"cross_maps_{table_b.split('_')[-1]}"
            if link_name.lower() in seen_names:
                link_name = f"{link_name}_{table_a.split('_')[-1]}"
            seen_names.add(link_name.lower())
            seen_names.add(link_key)

            concepts.append({
                "concept_id": f"link_cross_{table_a}_{table_b}",
                "concept_type": "link_type",
                "name": link_name,
                "description": f"Cross-table mapping: {table_a} → {table_b} ({mapping_type})",
                "source_tables": [table_a, table_b],
                "source_evidence": {
                    "mapping_type": mapping_type,
                    "confidence": confidence,
                },
                "confidence": confidence,
                "definition": {
                    "source": table_a,
                    "target": table_b,
                    "mapping_type": mapping_type,
                },
            })

        # === v22.0: 단일 테이블 폴백 — 테이블 스키마에서 직접 개념 생성 ===
        # unified_entities, homeomorphisms, FKs, value_overlaps 모두 비어있을 때
        if not concepts and context.tables:
            logger.info(f"[v22.0] No concepts from multi-table sources. "
                        f"Generating from {len(context.tables)} table schema(s) directly.")
            for table_name, table_info in context.tables.items():
                # Object Type: 테이블 자체를 엔티티로
                obj_name = self._to_pascal_case(table_name)
                if obj_name.lower() in seen_names:
                    continue
                seen_names.add(obj_name.lower())

                columns = []
                if hasattr(table_info, 'columns'):
                    columns = table_info.columns or []

                col_names = []
                pk_cols = []
                for col in columns[:20]:
                    col_name = col.get("name", str(col)) if isinstance(col, dict) else str(col)
                    col_type = col.get("type", "string") if isinstance(col, dict) else "string"
                    col_names.append(col_name)
                    # PK 추정 (id, *_id 패턴)
                    if col_name.lower() in ("id",) or col_name.lower().endswith("_id"):
                        pk_cols.append(col_name)

                concepts.append({
                    "concept_id": f"obj_table_{table_name}",
                    "concept_type": "object_type",
                    "name": obj_name,
                    "description": f"Entity derived from table: {table_name}",
                    "source_tables": [table_name],
                    "source_evidence": {
                        "extraction_method": "single_table_schema",
                        "confidence": 0.75,
                        "column_count": len(col_names),
                    },
                    "confidence": 0.75,
                    "definition": {
                        "attributes": col_names,
                        "primary_keys": pk_cols[:3],
                    },
                })

                # Property: 각 컬럼을 프로퍼티로
                for col in columns[:20]:
                    col_name = col.get("name", str(col)) if isinstance(col, dict) else str(col)
                    col_type = col.get("type", "string") if isinstance(col, dict) else "string"
                    prop_name = self._to_camel_case(col_name)
                    if prop_name.lower() in seen_names:
                        continue
                    seen_names.add(prop_name.lower())

                    # 의미론적 분류 from column_semantics
                    col_key = f"{table_name}.{col_name}"
                    sem_info = column_semantics.get(col_key, {})

                    concepts.append({
                        "concept_id": f"prop_table_{table_name}_{col_name}",
                        "concept_type": "property",
                        "name": prop_name,
                        "description": f"Property of {obj_name}: {col_name}"
                                       + (f" ({sem_info.get('business_role', '')})" if sem_info.get('business_role') else ""),
                        "source_tables": [table_name],
                        "source_evidence": {
                            "confidence": 0.7,
                            "extraction_method": "single_table_schema",
                        },
                        "confidence": 0.7,
                        "definition": {
                            "belongs_to": obj_name,
                            "data_type": col_type,
                            "semantic_category": sem_info.get("category", ""),
                        },
                    })

            logger.info(f"[v22.0] Single-table fallback generated {len(concepts)} concepts")

        logger.info(f"[v17.1] Extracted {len(concepts)} concepts from all Phase 1 sources")

        return concepts

    def _infer_hierarchy(
        self,
        concepts: List[Dict[str, Any]],
        context: "SharedContext",
    ) -> Dict[str, List[str]]:
        """v27.0: 도메인 무관 계층 관계 추론 (토큰 겹침 기반)"""
        import re
        hierarchy = {}

        object_types = [c for c in concepts if c.get("concept_type") == "object_type"]
        if len(object_types) < 2:
            return hierarchy

        # PascalCase/snake_case를 토큰으로 분리
        concept_tokens = {}
        for obj in object_types:
            name = obj.get("name", "")
            tokens = set(
                t.lower() for t in re.findall(r'[A-Z][a-z]+|[a-z]+', name)
                if len(t) > 2
            )
            concept_tokens[name] = tokens

        # 공통 토큰으로 그룹핑 — 2개 이상 개념에 나타나는 토큰이 parent 후보
        token_groups: Dict[str, List[str]] = {}
        for name, tokens in concept_tokens.items():
            for token in tokens:
                token_groups.setdefault(token, []).append(name)

        for token, members in token_groups.items():
            if 2 <= len(members) <= len(object_types) // 2 + 1:
                parent_name = token.capitalize()
                hierarchy[parent_name] = members

        return hierarchy

    async def _enhance_concepts_with_llm(
        self,
        concepts: List[Dict[str, Any]],
        hierarchy: Dict[str, List[str]],
        context: "SharedContext",
    ) -> List[Dict[str, Any]]:
        """v5.1: LLM을 통한 개념 보강 - 강화된 프롬프트"""
        if not concepts:
            return []

        # 개념 수가 많으면 배치 처리
        batch_size = 30  # v6.0: LLM 호출 최적화 (10→30, 품질 동일)
        enhanced = []
        domain = context.get_industry()

        # v5.1: 테이블 컬럼 정보도 컨텍스트로 포함
        table_context = {}
        for table_name, table_info in list(context.tables.items())[:10]:
            if hasattr(table_info, 'columns'):
                cols = [c.get("name", str(c)) if isinstance(c, dict) else str(c)
                        for c in table_info.columns[:8]]
                table_context[table_name] = cols

        # v22.0: TDA 시그니처 요약 (Betti numbers, topological features)
        tda_summary = {}
        for tname, sig in list((context.tda_signatures or {}).items())[:10]:
            if isinstance(sig, dict):
                tda_summary[tname] = {
                    "betti_numbers": sig.get("betti_numbers", sig.get("betti", [])),
                    "homology_features": sig.get("num_features", sig.get("features_count", 0)),
                }

        # v22.0: Schema analysis 요약 (FK 패턴, 정규화 수준)
        schema_summary = {}
        for tname, analysis in list((context.schema_analysis or {}).items())[:10]:
            if isinstance(analysis, dict):
                schema_summary[tname] = {
                    "pk_columns": analysis.get("pk_columns", []),
                    "fk_hints": analysis.get("fk_hints", []),
                    "normalization": analysis.get("normalization_level", "unknown"),
                }

        # v22.0: Cross-table mappings 요약
        cross_mappings_summary = []
        for m in (context.cross_table_mappings or [])[:20]:
            if isinstance(m, dict):
                cross_mappings_summary.append({
                    "source": m.get("table_a", m.get("source_table", "")),
                    "target": m.get("table_b", m.get("target_table", "")),
                    "type": m.get("mapping_type", m.get("type", "unknown")),
                })

        for i in range(0, len(concepts), batch_size):
            batch = concepts[i:i + batch_size]

            instruction = f"""You are an expert ontologist specializing in {domain} domain.
Enhance these ontology concepts with professional, domain-specific descriptions.

## Domain Context
- Industry: {domain}
- Available tables and sample columns: {json.dumps(table_context, indent=2, default=str)}
- Concept hierarchy: {json.dumps(hierarchy, indent=2, default=str)}
- TDA topological signatures: {json.dumps(tda_summary, indent=2, default=str)}
- Schema analysis: {json.dumps(schema_summary, indent=2, default=str)}
- Cross-table mappings: {json.dumps(cross_mappings_summary, indent=2, default=str)}

## Concepts to Enhance
{json.dumps(batch, indent=2, ensure_ascii=False, default=str)}

## Enhancement Requirements

### For Object Types (Entities):
- Write a 2-3 sentence professional description explaining:
  1. What this entity represents in the {domain} domain
  2. Its business purpose and typical use cases
  3. Key characteristics that distinguish it
- Example: "A Plant represents a physical manufacturing facility within the supply chain network. Plants serve as production centers that transform raw materials into finished goods. Each plant has defined production capacities and is associated with specific geographic locations."

### For Link Types (Relationships):
- Describe the semantic meaning of the relationship
- Explain the cardinality (one-to-many, many-to-many)
- Mention business rules that govern this relationship
- Example: "Represents the operational assignment of warehouse capacity to plants. This many-to-one relationship indicates which warehouses serve as storage facilities for each plant's inventory. Critical for inventory management and logistics planning."

### For Properties (Attributes):
- Explain what this property measures or identifies
- Mention data type and typical value ranges
- Describe business significance

## Output Format
Return ONLY a JSON array. Each object must preserve original concept_id, concept_type, source_tables, source_evidence, and confidence. Enhance name (to PascalCase for entities, camelCase for properties) and description.

```json
[
  {{
    "concept_id": "<original>",
    "concept_type": "<original>",
    "name": "<enhanced name following conventions>",
    "description": "<professional 2-3 sentence description>",
    "definition": {{<original or enhanced>}},
    "source_tables": [<original>],
    "source_evidence": {{<original>}},
    "confidence": <original or slightly adjusted based on clarity>
  }}
]
```"""

            try:
                response = await self.call_llm(instruction, max_tokens=4000)
                batch_enhanced = parse_llm_json(response, default=[])

                if batch_enhanced and isinstance(batch_enhanced, list):
                    # v5.1: 각 개념에 원본 정보가 누락되면 보완
                    for j, enhanced_concept in enumerate(batch_enhanced):
                        if j < len(batch):
                            original = batch[j]
                            # 필수 필드 보존
                            for key in ["concept_id", "concept_type", "source_tables", "source_evidence", "confidence"]:
                                if key not in enhanced_concept and key in original:
                                    enhanced_concept[key] = original[key]
                            # v27.0: definition.extracted_attributes 보존
                            orig_attrs = original.get("definition", {}).get("extracted_attributes", [])
                            if orig_attrs:
                                enh_def = enhanced_concept.setdefault("definition", {})
                                if not enh_def.get("extracted_attributes"):
                                    enh_def["extracted_attributes"] = orig_attrs
                            # 설명이 여전히 템플릿 형태면 원본 개선
                            desc = enhanced_concept.get("description", "")
                            if desc.startswith("Entity representing") or len(desc) < 30:
                                enhanced_concept["description"] = self._generate_fallback_description(
                                    enhanced_concept, domain
                                )
                    enhanced.extend(batch_enhanced)
                else:
                    # LLM 실패 시 알고리즘 기반 개선
                    for concept in batch:
                        concept["description"] = self._generate_fallback_description(concept, domain)
                    enhanced.extend(batch)
            except Exception as e:
                logger.warning(f"LLM enhancement batch {i} failed: {e}")
                for concept in batch:
                    concept["description"] = self._generate_fallback_description(concept, domain)
                enhanced.extend(batch)

        return enhanced if enhanced else concepts

    def _generate_fallback_description(self, concept: Dict[str, Any], domain: str) -> str:
        """v5.1: LLM 실패 시 알고리즘 기반 설명 생성"""
        concept_type = concept.get("concept_type", "")
        name = concept.get("name", "")
        source_tables = concept.get("source_tables", [])

        # 이름에서 단어 추출
        words = name.replace("_", " ").replace("-", " ").split()
        readable_name = " ".join(words).title()

        if concept_type == "object_type":
            table_info = f" sourced from {', '.join(source_tables[:2])}" if source_tables else ""
            return (
                f"{readable_name} is a core entity in the {domain} domain{table_info}. "
                f"It captures essential business data for operational and analytical purposes. "
                f"This entity type is fundamental for tracking and managing {readable_name.lower()}-related activities."
            )
        elif concept_type == "link_type":
            if len(source_tables) >= 2:
                return (
                    f"Defines the relationship between {source_tables[0]} and {source_tables[1]}. "
                    f"This connection enables data integration and supports cross-entity analysis "
                    f"within the {domain} domain."
                )
            return f"Represents a data relationship within the {domain} domain for {readable_name}."
        elif concept_type == "property":
            return (
                f"The {readable_name} attribute captures specific data values "
                f"essential for {domain} domain operations and reporting."
            )
        else:
            return f"{readable_name} concept in the {domain} ontology."

    def _to_pascal_case(self, name: str) -> str:
        """PascalCase 변환"""
        words = name.replace("_", " ").replace("-", " ").split()
        return "".join(word.capitalize() for word in words)

    def _to_camel_case(self, name: str) -> str:
        """camelCase 변환"""
        words = name.replace("_", " ").replace("-", " ").split()
        if not words:
            return name
        return words[0].lower() + "".join(word.capitalize() for word in words[1:])

    def _infer_semantic_type(self, col_name: str, data_type: str) -> str:
        """
        v5.1: 컬럼명과 데이터 타입에서 의미론적 타입 추론

        Returns:
            identifier, measure, dimension, temporal, textual, categorical 등
        """
        col_lower = col_name.lower()

        # 식별자 패턴
        if any(p in col_lower for p in ["_id", "_code", "_key", "_no", "_number", "id_", "code_"]):
            return "identifier"
        if col_lower in ["id", "code", "key", "sku", "upc", "ean"]:
            return "identifier"

        # 시간 패턴
        if any(p in col_lower for p in ["date", "time", "created", "updated", "modified", "timestamp", "at"]):
            return "temporal"
        if any(p in col_lower for p in ["year", "month", "day", "hour", "minute"]):
            return "temporal"

        # 측정값 패턴
        if any(p in col_lower for p in ["amount", "price", "cost", "total", "sum", "count", "qty", "quantity"]):
            return "measure"
        if any(p in col_lower for p in ["rate", "ratio", "percent", "score", "weight", "height", "size"]):
            return "measure"
        if data_type in ["float", "double", "decimal", "numeric", "real"]:
            return "measure"

        # 차원 패턴
        if any(p in col_lower for p in ["type", "category", "status", "level", "class", "group"]):
            return "dimension"
        if any(p in col_lower for p in ["region", "country", "city", "state", "zone", "area"]):
            return "dimension"

        # 텍스트 패턴
        if any(p in col_lower for p in ["name", "title", "description", "comment", "note", "text"]):
            return "textual"
        if any(p in col_lower for p in ["address", "email", "phone", "url"]):
            return "textual"

        # 불리언 패턴
        if any(p in col_lower for p in ["is_", "has_", "can_", "flag", "active", "enabled"]):
            return "boolean"

        # 기본값
        if data_type in ["integer", "int", "bigint", "smallint"]:
            return "measure"
        if data_type in ["varchar", "text", "string", "char"]:
            return "textual"

        return "unknown"

    def _generate_business_link_name(self, table_a: str, table_b: str, join_keys: List[Dict]) -> str:
        """
        v5.1: 비즈니스 의미를 담은 Link 이름 생성 (알고리즘 기반)

        Examples:
            - plant_ports -> warehouse_capacities => "plant_has_warehouse_capacity"
            - orders -> products => "order_contains_product"
        """
        # 테이블 이름에서 핵심 엔티티 추출
        def extract_entity(table_name: str) -> str:
            # prefix 제거 (supply_chain_, healthcare_ 등)
            parts = table_name.lower().split("_")
            # 도메인 prefix 제거
            domain_prefixes = ["supply", "chain", "healthcare", "manufacturing"]
            clean_parts = [p for p in parts if p not in domain_prefixes]
            return "_".join(clean_parts[-2:]) if len(clean_parts) >= 2 else clean_parts[-1] if clean_parts else table_name

        entity_a = extract_entity(table_a)
        entity_b = extract_entity(table_b)

        # 관계 동사 추론
        # join_key 패턴에서 관계 유형 추론
        relation_verb = "relates_to"

        if join_keys:
            # v7.5.1: join_keys가 dict 또는 tuple 형식일 수 있음
            first_key = join_keys[0]
            if isinstance(first_key, dict):
                key_name = first_key.get("table_a_column", "").lower()
            elif isinstance(first_key, (tuple, list)) and len(first_key) >= 1:
                key_name = str(first_key[0]).lower()  # 첫 번째 요소 (from_column)
            else:
                key_name = str(first_key).lower()

            # 포함 관계
            if any(p in key_name for p in ["product", "item", "line"]):
                relation_verb = "contains"
            # 소유 관계
            elif any(p in key_name for p in ["_id", "_code", "code"]):
                relation_verb = "has"
            # 위치 관계
            elif any(p in key_name for p in ["location", "warehouse", "plant", "port"]):
                relation_verb = "located_at"
            # 시간 관계
            elif any(p in key_name for p in ["date", "time", "period"]):
                relation_verb = "occurred_at"
            else:
                relation_verb = "has"

        # 싱글라이즈 (간단한 복수형 -> 단수형)
        def singularize(word: str) -> str:
            if word.endswith("ies"):
                return word[:-3] + "y"
            elif word.endswith("es"):
                return word[:-2]
            elif word.endswith("s") and not word.endswith("ss"):
                return word[:-1]
            return word

        entity_b_singular = singularize(entity_b.replace("_", " ").split()[-1])

        return f"{entity_a}_{relation_verb}_{entity_b_singular}"

    def _generate_link_description(self, homeo: Any) -> str:
        """v5.1: Homeomorphism에서 의미 있는 관계 설명 생성"""
        table_a = homeo.table_a
        table_b = homeo.table_b
        join_keys = homeo.join_keys if hasattr(homeo, 'join_keys') else []
        confidence = homeo.confidence if hasattr(homeo, 'confidence') else 0.5
        homeo_type = homeo.homeomorphism_type if hasattr(homeo, 'homeomorphism_type') else "unknown"

        # 테이블 이름에서 엔티티 추출
        def extract_readable(table_name: str) -> str:
            parts = table_name.lower().replace("_", " ").split()
            # 도메인 접두사 제거
            skip_words = ["supply", "chain", "healthcare", "manufacturing", "data"]
            clean = [p for p in parts if p not in skip_words]
            return " ".join(clean[-2:]) if len(clean) >= 2 else " ".join(clean)

        entity_a_readable = extract_readable(table_a)
        entity_b_readable = extract_readable(table_b)

        # 조인 키 정보
        key_info = ""
        if join_keys:
            key_names = []
            for jk in join_keys:
                if isinstance(jk, dict):
                    key_names.append(jk.get("column_a", jk.get("key", "")))
                else:
                    key_names.append(str(jk))
            if key_names:
                key_info = f" via {', '.join(filter(None, key_names[:2]))}"

        # 신뢰도에 따른 관계 강도
        if confidence >= 0.7:
            strength = "strong"
        elif confidence >= 0.4:
            strength = "moderate"
        else:
            strength = "weak"

        # 관계 타입별 설명
        relation_type_desc = {
            "VALUE_BASED": "value-based join relationship",
            "STRUCTURAL": "structural relationship",
            "SEMANTIC": "semantic relationship",
            "ALGEBRAIC": "algebraic relationship",
        }
        type_desc = relation_type_desc.get(str(homeo_type), "data relationship")

        return (
            f"Represents a {strength} {type_desc} between {entity_a_readable} and "
            f"{entity_b_readable}{key_info}. "
            f"Confidence: {confidence:.0%}"
        )

    def _generate_ontology_triples(
        self,
        concepts: List[Any],
        hierarchy: Dict[str, Any],
        context: "SharedContext",
    ) -> List[Dict[str, Any]]:
        """
        v4.3: 온톨로지 개념에서 Knowledge Graph 트리플 생성

        Args:
            concepts: OntologyConcept 리스트
            hierarchy: 계층 관계 정보
            context: SharedContext

        Returns:
            트리플 딕셔너리 리스트
        """
        triples = []

        # 개념 타입 → RDF 타입 매핑
        type_map = {
            "object_type": "owl:Class",
            "property": "owl:DatatypeProperty",
            "link_type": "owl:ObjectProperty",
            "constraint": "owl:Restriction",
        }

        for concept in concepts:
            # OntologyConcept 또는 dict 처리
            if hasattr(concept, 'concept_id'):
                concept_id = concept.concept_id
                concept_type = concept.concept_type
                name = concept.name
                description = concept.description
                source_tables = concept.source_tables
                confidence = concept.confidence
            else:
                concept_id = concept.get('concept_id', '')
                concept_type = concept.get('concept_type', '')
                name = concept.get('name', '')
                description = concept.get('description', '')
                source_tables = concept.get('source_tables', [])
                confidence = concept.get('confidence', 0.0)

            concept_uri = f"ont:{concept_id}"
            rdf_type = type_map.get(concept_type, "owl:Class")

            # 1. rdf:type 트리플
            triples.append({
                "subject": concept_uri,
                "predicate": "rdf:type",
                "object": rdf_type,
                "confidence": 1.0,
                "source": "ontology_architect",
            })

            # 2. rdfs:label 트리플
            triples.append({
                "subject": concept_uri,
                "predicate": "rdfs:label",
                "object": f'"{name}"',
                "confidence": 1.0,
                "source": "ontology_architect",
            })

            # 3. rdfs:comment (설명)
            if description:
                triples.append({
                    "subject": concept_uri,
                    "predicate": "rdfs:comment",
                    "object": f'"{description[:200]}"',
                    "confidence": 1.0,
                    "source": "ontology_architect",
                })

            # 4. qual:hasConfidence
            triples.append({
                "subject": concept_uri,
                "predicate": "qual:hasConfidence",
                "object": f'"{confidence}"^^xsd:float',
                "confidence": 1.0,
                "source": "ontology_architect",
            })

            # 5. prov:hadPrimarySource (소스 테이블)
            for table in source_tables:
                triples.append({
                    "subject": concept_uri,
                    "predicate": "prov:hadPrimarySource",
                    "object": f"schema:Table_{table}",
                    "confidence": 1.0,
                    "source": "ontology_architect",
                })

        # 계층 관계 트리플 (rdfs:subClassOf)
        for parent, children_info in hierarchy.items():
            parent_uri = f"ont:{parent}"
            children = children_info if isinstance(children_info, list) else children_info.get("children", [])
            for child in children:
                child_id = child.get("concept_id") if isinstance(child, dict) else child
                if child_id:
                    triples.append({
                        "subject": f"ont:{child_id}",
                        "predicate": "rdfs:subClassOf",
                        "object": parent_uri,
                        "confidence": 0.8,
                        "source": "ontology_architect",
                    })

        # Homeomorphism 관계 (owl:equivalentClass)
        for homeo in context.homeomorphisms:
            if not homeo.is_homeomorphic:
                continue

            table_a_uri = f"schema:Table_{homeo.table_a}"
            table_b_uri = f"schema:Table_{homeo.table_b}"

            triples.append({
                "subject": table_a_uri,
                "predicate": "owl:equivalentClass",
                "object": table_b_uri,
                "confidence": homeo.confidence,
                "source": "ontology_architect",
            })

            # Join Key 관계
            for jk in homeo.join_keys:
                col_a = jk.get('table_a_column', '')
                col_b = jk.get('table_b_column', '')
                if col_a and col_b:
                    col_a_uri = f"schema:Column_{homeo.table_a}_{col_a}"
                    col_b_uri = f"schema:Column_{homeo.table_b}_{col_b}"
                    triples.append({
                        "subject": col_a_uri,
                        "predicate": "owl:sameAs",
                        "object": col_b_uri,
                        "confidence": jk.get('overlap_ratio', homeo.confidence),
                        "source": "ontology_architect",
                    })

        return triples

    def _run_causal_inference(
        self,
        context: "SharedContext",
        concepts: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        v6.1: Dynamic Causal Inference Analysis

        엔티티 간의 인과 관계를 분석하여 온톨로지에 반영합니다.
        - Granger Causality로 시계열 인과관계 탐지
        - 데이터 통합 시나리오별 영향도 분석
        - 인과 그래프 구축

        Args:
            context: SharedContext
            concepts: 추출된 온톨로지 개념들

        Returns:
            인과 관계 및 인사이트 딕셔너리
        """
        import numpy as np

        causal_relationships = []
        insights = {
            "granger_causality": [],
            "impact_analysis": {},
            "causal_graph_nodes": [],
            "causal_graph_edges": [],
        }

        # 1. 테이블 정보로 인과 영향 분석
        tables_dict = {}
        for table_name, table_info in context.tables.items():
            if hasattr(table_info, 'columns'):
                cols = []
                for c in table_info.columns[:20]:
                    if isinstance(c, dict):
                        cols.append({
                            "name": c.get("name", ""),
                            "type": c.get("type", "string"),
                        })
                    else:
                        cols.append({"name": str(c), "type": "string"})
                tables_dict[table_name] = {
                    "columns": cols,
                    "row_count": table_info.row_count if hasattr(table_info, 'row_count') else 0,
                }

        # FK 관계 정보 수집
        fk_relations = []
        for h in (context.homeomorphisms or []):
            if h.is_homeomorphic:
                fk_relations.append({
                    "source_table": h.table_a,
                    "target_table": h.table_b,
                    "join_keys": h.join_keys if hasattr(h, 'join_keys') else [],
                    "confidence": h.confidence,
                })

        # 엔티티 매칭 정보 수집
        entity_matches = []
        for e in (context.unified_entities or []):
            if len(e.source_tables) > 1:
                entity_matches.append({
                    "entity_id": e.entity_id,
                    "canonical_name": e.canonical_name,
                    "source_tables": e.source_tables,
                    "confidence": e.confidence,
                })

        # v22.1: 전체 데이터 로드 (샘플링 금지)
        real_data = context.get_all_full_data()

        # 2. CausalImpactAnalyzer로 실제 데이터 기반 인과 분석
        if tables_dict:
            try:
                impact_result = self.causal_analyzer.analyze(
                    tables=tables_dict,
                    fk_relations=fk_relations,
                    entity_matches=entity_matches,
                    target_kpis=["data_quality", "operational_efficiency"],
                    real_data=real_data,  # v18.4: 전체 원본 데이터 전달
                )

                insights["impact_analysis"] = {
                    "summary": impact_result.get("summary", {}),
                    "treatment_effects": impact_result.get("treatment_effects", []),
                    "recommendations": impact_result.get("recommendations", []),
                    "group_comparisons": impact_result.get("group_comparisons", []),  # v18.5
                }

                # 인과 그래프 정보 저장
                causal_graph = impact_result.get("causal_graph", {})
                insights["causal_graph_nodes"] = causal_graph.get("nodes", [])
                insights["causal_graph_edges"] = causal_graph.get("edges", [])

                # 통계적으로 유의미한 treatment effects를 인과 관계로 변환
                for te in impact_result.get("treatment_effects", []):
                    if te.get("significant", False):
                        causal_relationships.append({
                            "relationship_type": "causal_impact",
                            "source": "data_integration",
                            "target": te.get("type", "kpi"),
                            "effect_estimate": te.get("estimate", 0),
                            "confidence": 1 - te.get("p_value", 0.5),
                            "evidence": "treatment_effect_analysis",
                        })

                logger.info(f"CausalImpactAnalyzer: {len(impact_result.get('treatment_effects', []))} effects analyzed")
            except Exception as e:
                logger.warning(f"CausalImpactAnalyzer failed: {e}")

        # 3. 엔티티 간 인과 관계 탐지 (개념 기반)
        # Object Type 개념들 간의 관계 분석
        object_types = [c for c in concepts if c.get("concept_type") == "object_type"]
        link_types = [c for c in concepts if c.get("concept_type") == "link_type"]

        # Link Types에서 인과 관계 추론
        for link in link_types:
            definition = link.get("definition", {})
            source = definition.get("source", "")
            target = definition.get("target", "")
            if isinstance(source, dict):
                source = source.get("name", str(source))
            if isinstance(target, dict):
                target = target.get("name", str(target))
            source = str(source)
            target = str(target)

            if source and target:
                # 테이블 간 관계에서 인과 방향 추론
                causal_relationships.append({
                    "relationship_type": "data_dependency",
                    "source": source,
                    "target": target,
                    "confidence": link.get("confidence", 0.5),
                    "evidence": "schema_analysis",
                    "link_name": link.get("name", ""),
                })

        # 4. Granger Causality 분석 (시뮬레이션 데이터)
        # 실제 시계열 데이터가 있으면 분석, 없으면 구조적 분석만 수행
        # 향후: context에서 실제 시계열 데이터 활용
        if len(object_types) >= 2:
            # 데모용: 개념 간의 구조적 인과관계를 시뮬레이션
            for i, obj1 in enumerate(object_types[:5]):
                for obj2 in object_types[i+1:6]:
                    # 같은 소스 테이블을 공유하면 관계가 있을 가능성
                    shared_tables = set(obj1.get("source_tables", [])) & set(obj2.get("source_tables", []))
                    if shared_tables:
                        insights["granger_causality"].append({
                            "cause": obj1.get("name"),
                            "effect": obj2.get("name"),
                            "shared_tables": list(shared_tables),
                            "potential_relationship": True,
                            "note": "Structural co-occurrence suggests potential causal link",
                        })

        # === v10.0: Cross-Entity Correlation Analysis (Palantir-Style Insights) ===
        palantir_insights = []
        # v24.0: 단일 테이블이면 cross-entity 분석 스킵
        table_count = len(context.tables) if context.tables else 0
        if table_count < 2:
            logger.debug(f"Single table dataset ({table_count} tables): skipping cross-entity correlation")
        else:
            try:
                from ...analysis.cross_entity_correlation import CrossEntityCorrelationAnalyzer

                # 데이터 디렉토리 가져오기
                data_dir = getattr(context, 'data_directory', None)
                logger.info(f"v10.0: Checking data_directory: {data_dir}")
                if data_dir:
                    logger.info(f"Running Cross-Entity Correlation Analysis on {data_dir}")

                    correlation_analyzer = CrossEntityCorrelationAnalyzer(
                        data_dir=data_dir,
                        correlation_threshold=0.3,
                        p_value_threshold=0.05,
                        llm_client=self.llm_client,
                    )

                    # 도메인 컨텍스트 가져오기 (v13.2: dynamic_data에서 LLM 탐지 결과 조회)
                    domain_context = None
                    detected_domain = context.get_dynamic('detected_domain', '')
                    if detected_domain:
                        domain_context = detected_domain
                        logger.info(f"v13.2: Using LLM-detected domain from dynamic_data: {domain_context}")
                    elif hasattr(context, 'domain_context') and context.domain_context:
                        domain_context = getattr(context.domain_context, 'industry', None)

                    # v13.2: Phase 1 DataAnalystAgent 분석 결과를 dynamic_data에서 조회
                    phase1_analysis = {
                        "data_understanding": context.get_dynamic('data_understanding', {}),
                        "column_semantics": context.get_dynamic('column_semantics', {}),
                        "data_patterns": context.get_dynamic('data_patterns', []),
                        "data_quality_issues": context.get_dynamic('data_quality_issues', []),
                        "recommended_analysis": context.get_dynamic('recommended_analysis', {}),
                    }
                    logger.info(f"v13.2: Passing Phase 1 analysis from dynamic_data: "
                               f"patterns={len(phase1_analysis['data_patterns'])}, "
                               f"quality_issues={len(phase1_analysis['data_quality_issues'])}")

                    # 실제 데이터 기반 상관관계 분석
                    palantir_results = correlation_analyzer.analyze_sync(
                        table_names=list(context.tables.keys()),
                        fk_relations=fk_relations,
                        entity_matches=entity_matches,
                        domain_context=domain_context,
                        phase1_analysis=phase1_analysis,  # v13.0: Phase 1 분석 결과 전달
                    )

                    # 결과를 dict로 변환
                    palantir_insights = [pi.to_dict() for pi in palantir_results]
                    insights["palantir_insights"] = palantir_insights

                    # 높은 신뢰도의 인사이트를 causal_relationships에도 추가
                    for pi in palantir_results:
                        if pi.confidence > 0.5:
                            causal_relationships.append({
                                "relationship_type": "predictive_correlation",
                                "source": f"{pi.source_table}.{pi.source_column}",
                                "target": f"{pi.target_table}.{pi.target_column}",
                                "effect_estimate": pi.correlation_coefficient,
                                "confidence": pi.confidence,
                                "insight": pi.business_interpretation,
                                "predicted_impact": pi.predicted_impact,
                            })

                    logger.info(f"Cross-Entity Correlation: {len(palantir_insights)} Palantir-style insights generated")

                    # v26.1: 교란변수/파생컬럼 결과를 insights에 전파
                    if hasattr(correlation_analyzer, '_confounding_results') and correlation_analyzer._confounding_results:
                        insights["confounding_variables"] = correlation_analyzer._confounding_results
                        logger.info(f"[v26.1] {len(correlation_analyzer._confounding_results)} confounding variables detected")
                    if hasattr(correlation_analyzer, '_derived_columns') and correlation_analyzer._derived_columns:
                        insights["derived_columns"] = correlation_analyzer._derived_columns
                        logger.info(f"[v26.1] {len(correlation_analyzer._derived_columns)} derived columns detected")
                else:
                    logger.warning("Data directory not set in context, skipping Cross-Entity Correlation Analysis")

            except Exception as e:
                logger.warning(f"Cross-Entity Correlation Analysis failed: {e}")

        logger.info(f"Causal Inference: {len(causal_relationships)} relationships, {len(insights['granger_causality'])} Granger hints")

        return {
            "causal_relationships": causal_relationships,
            "insights": insights,
            "palantir_insights": palantir_insights,
        }

    def _generate_virtual_entities(
        self,
        context: "SharedContext",
        concepts: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        v6.1: Virtual Entity Generation

        비즈니스 분석에 필요한 가상/파생 엔티티를 생성합니다.
        - ChurnRisk: 이탈 위험도
        - CustomerLifetimeValue: 고객 생애 가치
        - InventoryRisk: 재고 위험도
        - PredictedDemand: 예측 수요

        Args:
            context: SharedContext
            concepts: 추출된 온톨로지 개념들

        Returns:
            생성된 가상 엔티티 목록
        """
        # 도메인 결정
        domain = context.get_industry() or "general"
        domain_lower = domain.lower()

        # 도메인 매핑
        if any(term in domain_lower for term in ["health", "medical", "hospital", "patient"]):
            mapped_domain = "healthcare"
        elif any(term in domain_lower for term in ["supply", "chain", "warehouse", "inventory", "logistics"]):
            mapped_domain = "supply_chain"
        elif any(term in domain_lower for term in ["ecommerce", "retail", "shop", "order", "customer"]):
            mapped_domain = "ecommerce"
        else:
            mapped_domain = "general"

        # VirtualEntityGenerator 초기화 (lazy)
        if self.virtual_entity_generator is None or getattr(self, '_ve_domain', None) != mapped_domain:
            self.virtual_entity_generator = VirtualEntityGenerator(domain=mapped_domain)
            self._ve_domain = mapped_domain

        # 기존 엔티티 정보 수집
        existing_entities = []
        for entity in (context.unified_entities or []):
            existing_entities.append({
                "entity_id": entity.entity_id,
                "name": entity.canonical_name,
                "entity_type": entity.entity_type,
                "source_tables": entity.source_tables,
                "confidence": entity.confidence,
            })

        # 개념에서도 엔티티 정보 추출
        for concept in concepts:
            if concept.get("concept_type") == "object_type":
                existing_entities.append({
                    "name": concept.get("name", ""),
                    "entity_type": concept.get("concept_type", ""),
                    "source_tables": concept.get("source_tables", []),
                    "confidence": concept.get("confidence", 0.5),
                })

        # 테이블 정보 수집
        tables_dict = {}
        for table_name, table_info in context.tables.items():
            if hasattr(table_info, 'columns'):
                cols = []
                for c in table_info.columns[:20]:
                    if isinstance(c, dict):
                        cols.append(c)
                    else:
                        cols.append({"name": str(c), "type": "string"})
                tables_dict[table_name] = {
                    "columns": cols,
                    "row_count": table_info.row_count if hasattr(table_info, 'row_count') else 0,
                }

        # 가상 엔티티 생성
        virtual_entities = generate_virtual_entities_for_context(
            context_tables=tables_dict,
            context_entities=existing_entities,
            domain=mapped_domain,
        )

        logger.info(f"VirtualEntityGenerator ({mapped_domain}): Generated {len(virtual_entities)} virtual entities")

        return virtual_entities

    # === v17.1: LLM-Powered Concept Analysis ===

    async def _llm_deep_concept_analysis(
        self,
        concepts: List[Dict[str, Any]],
        context: "SharedContext",
    ) -> List[Dict[str, Any]]:
        """
        v17.1: LLM 심층 개념 분석 - 개념 정의 및 비즈니스 의미 강화

        알고리즘 추출 후 LLM으로 각 개념의:
        1. 비즈니스 정의 (business_definition) 추가
        2. 관련 도메인 개념 (related_concepts) 식별
        3. 데이터 품질 지표 (quality_indicators) 평가
        4. 잠재적 사용 사례 (use_cases) 도출

        Args:
            concepts: 알고리즘으로 추출된 개념들
            context: SharedContext

        Returns:
            LLM으로 강화된 개념 리스트
        """
        if not concepts:
            return concepts

        domain = context.get_industry() or "general"

        # 테이블 컨텍스트 구성
        table_summary = {}
        for table_name, table_info in list(context.tables.items())[:15]:
            if hasattr(table_info, 'columns'):
                cols = [c.get("name", str(c)) if isinstance(c, dict) else str(c)
                        for c in table_info.columns[:10]]
                table_summary[table_name] = {
                    "columns": cols,
                    "row_count": getattr(table_info, 'row_count', 0),
                }

        # 배치 처리 (20개씩)
        batch_size = 20
        enriched_concepts = []

        for i in range(0, len(concepts), batch_size):
            batch = concepts[i:i + batch_size]

            # Object Types만 심층 분석 (Properties는 간략히)
            object_types = [c for c in batch if c.get("concept_type") == "object_type"]
            other_concepts = [c for c in batch if c.get("concept_type") != "object_type"]

            if object_types:
                prompt = f"""You are an expert ontologist specializing in the {domain} domain.

## Task
Perform DEEP ANALYSIS of these ontology concepts extracted from enterprise data.
Add business context, identify related concepts, and assess data quality indicators.

## Domain Context
- Industry: {domain}
- Available Tables: {json.dumps(list(table_summary.keys()), indent=2)}

## Concepts to Analyze
{json.dumps(object_types, indent=2, ensure_ascii=False, default=str)}

## Analysis Requirements
For EACH concept, add:

1. **business_definition**: A precise 2-3 sentence business definition explaining:
   - What this entity represents in the {domain} domain
   - Its role in business processes
   - Why it matters for analytics/operations

2. **related_concepts**: List of 2-4 concepts this entity likely relates to
   Format: ["ConceptA", "ConceptB", ...]

3. **quality_indicators**: Key data quality metrics to monitor
   Format: {{"completeness": "description", "uniqueness": "description", "timeliness": "description"}}

4. **use_cases**: 2-3 specific business use cases
   Format: ["Use case 1", "Use case 2", ...]

5. **semantic_tags**: Domain-specific tags for classification
   Format: ["tag1", "tag2", ...]

## Output Format
Return ONLY a JSON array. Preserve ALL original fields and ADD the new analysis fields.

```json
[
  {{
    "concept_id": "<original>",
    "concept_type": "<original>",
    "name": "<original>",
    "description": "<original or enhanced>",
    "source_tables": [<original>],
    "source_evidence": {{<original>}},
    "confidence": <original>,
    "definition": {{<original>}},
    "business_definition": "<new: detailed business definition>",
    "related_concepts": ["<new: related concept names>"],
    "quality_indicators": {{"<new: quality metrics>"}},
    "use_cases": ["<new: business use cases>"],
    "semantic_tags": ["<new: domain tags>"]
  }}
]
```"""

                try:
                    response = await self.call_llm(prompt, max_tokens=4000)
                    enriched_batch = parse_llm_json(response, default=[])

                    if enriched_batch and isinstance(enriched_batch, list):
                        # 원본 필드 보존
                        for j, enriched in enumerate(enriched_batch):
                            if j < len(object_types):
                                original = object_types[j]
                                for key in ["concept_id", "concept_type", "name", "source_tables",
                                          "source_evidence", "confidence", "definition"]:
                                    if key not in enriched and key in original:
                                        enriched[key] = original[key]
                        enriched_concepts.extend(enriched_batch)
                    else:
                        enriched_concepts.extend(object_types)
                except Exception as e:
                    logger.warning(f"LLM deep analysis batch failed: {e}")
                    enriched_concepts.extend(object_types)

            # Properties 등은 원본 유지
            enriched_concepts.extend(other_concepts)

        return enriched_concepts

    async def _llm_infer_semantic_relationships(
        self,
        concepts: List[Dict[str, Any]],
        context: "SharedContext",
    ) -> List[Dict[str, Any]]:
        """
        v17.1: LLM 의미론적 관계 추론 - 알고리즘이 놓친 관계 발견

        알고리즘 기반 관계 외에 LLM이 추론하는:
        1. 비즈니스 규칙 기반 관계 (is-a, part-of, has-a)
        2. 시간적/인과적 관계 (precedes, causes, affects)
        3. 프로세스 관계 (leads-to, enables, triggers)

        Args:
            concepts: 추출된 개념들
            context: SharedContext

        Returns:
            새로 발견된 관계 개념들
        """
        if not concepts:
            return []

        domain = context.get_industry() or "general"

        # Object Types만 추출
        object_types = [c for c in concepts if c.get("concept_type") == "object_type"]
        existing_links = [c for c in concepts if c.get("concept_type") == "link_type"]

        if len(object_types) < 2:
            return []

        # FK 관계 컨텍스트
        existing_fk_pairs = set()
        for link in existing_links:
            source = link.get("definition", {}).get("source", "")
            target = link.get("definition", {}).get("target", "")
            if isinstance(source, dict):
                source = source.get("name", str(source))
            if isinstance(target, dict):
                target = target.get("name", str(target))
            if source and target:
                existing_fk_pairs.add((str(source).lower(), str(target).lower()))

        # Homeomorphism 정보
        homeo_summary = []
        for h in (context.homeomorphisms or [])[:20]:
            homeo_summary.append({
                "table_a": h.table_a,
                "table_b": h.table_b,
                "confidence": h.confidence,
            })

        # Pre-build lists outside f-string to avoid {{}} parsing issues
        object_types_summary = [
            {"name": c.get("name"), "source_tables": c.get("source_tables", [])}
            for c in object_types
        ]
        existing_links_summary = [
            {
                "name": c.get("name"),
                "source": str(c.get("definition", {}).get("source", "")),
                "target": str(c.get("definition", {}).get("target", "")),
            }
            for c in existing_links[:15]
        ]

        prompt = f"""You are an expert ontologist specializing in semantic relationship inference.

## Task
Analyze these ontology concepts and DISCOVER IMPLICIT RELATIONSHIPS that algorithms may have missed.
Focus on business-level semantic relationships, not just FK relationships.

## Domain Context
- Industry: {domain}

## Existing Object Types (Entities)
{json.dumps(object_types_summary, indent=2, default=str)}

## Already Discovered Relationships (FK-based)
{json.dumps(existing_links_summary, indent=2, default=str)}

## Homeomorphism Evidence
{json.dumps(homeo_summary, indent=2, default=str)}

## Relationship Types to Discover
1. **Hierarchical**: is-a (Patient is-a Person), part-of (LineItem part-of Order)
2. **Association**: has-a (Patient has-a Diagnosis), belongs-to (Claim belongs-to Patient)
3. **Temporal**: precedes (Admission precedes Discharge), follows (Payment follows Claim)
4. **Causal**: causes (Treatment causes Outcome), affects (Procedure affects Cost)
5. **Process**: enables (Authorization enables Treatment), triggers (Event triggers Alert)

## Rules
- ONLY propose relationships between entities in the list above
- DO NOT duplicate existing FK relationships
- Each relationship should have clear business justification
- Confidence: 0.7+ for strong evidence, 0.5-0.7 for inferred

## Output Format
Return ONLY a JSON array of NEW relationship concepts:

```json
[
  {{
    "concept_id": "link_llm_<source>_<target>",
    "concept_type": "link_type",
    "name": "<verb_phrase_name>",
    "description": "<business justification for this relationship>",
    "source_tables": ["<source_entity_table>", "<target_entity_table>"],
    "source_evidence": {{
      "inference_type": "semantic|hierarchical|temporal|causal|process",
      "reasoning": "<why this relationship exists>"
    }},
    "confidence": 0.XX,
    "definition": {{
      "source": "<source_entity_name>",
      "target": "<target_entity_name>",
      "relationship_type": "<is-a|part-of|has-a|belongs-to|precedes|causes|enables|etc>",
      "cardinality": "one-to-one|one-to-many|many-to-many"
    }}
  }}
]
```

If no additional relationships can be confidently inferred, return an empty array: []"""

        try:
            response = await self.call_llm(prompt, max_tokens=3000)
            new_relationships = parse_llm_json(response, default=[])

            if new_relationships and isinstance(new_relationships, list):
                # 중복 및 유효성 검사
                valid_relationships = []
                for rel in new_relationships:
                    source = rel.get("definition", {}).get("source", "")
                    target = rel.get("definition", {}).get("target", "")
                    if isinstance(source, dict):
                        source = source.get("name", str(source))
                    if isinstance(target, dict):
                        target = target.get("name", str(target))
                    source = str(source)
                    target = str(target)

                    # 중복 체크
                    if (source.lower(), target.lower()) in existing_fk_pairs:
                        continue
                    if (target.lower(), source.lower()) in existing_fk_pairs:
                        continue

                    # 최소 confidence 체크
                    if rel.get("confidence", 0) < 0.4:
                        continue

                    # concept_id가 없으면 생성
                    if not rel.get("concept_id"):
                        rel["concept_id"] = f"link_llm_{source}_{target}".replace(" ", "_").lower()

                    valid_relationships.append(rel)
                    existing_fk_pairs.add((source.lower(), target.lower()))

                logger.info(f"[v17.1] LLM inferred {len(valid_relationships)} semantic relationships")
                return valid_relationships

        except Exception as e:
            import traceback
            logger.warning(f"LLM relationship inference failed: {e}\n{traceback.format_exc()}")

        return []
