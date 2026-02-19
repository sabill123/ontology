"""
Entity Classifier Autonomous Agent (v3.0 - Full Entity Resolution)

엔티티 분류 자율 에이전트 - discovery.py에서 분리됨
- EntityExtractor: 패턴 기반 엔티티 추출
- EntityResolver: ML 기반 엔티티 매칭/통합
- LLM 기반 검증 및 분류
"""

import json
import logging
import re
from typing import Dict, Any, List, Optional, TYPE_CHECKING

from ...base import AutonomousAgent
from ...analysis import (
    EntityExtractor,
    EntityResolver,
    parse_llm_json,
)

if TYPE_CHECKING:
    from ....todo.models import PipelineTodo, TodoResult
    from ....shared_context import SharedContext

logger = logging.getLogger(__name__)


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

## ENTITY CLASSIFICATION RULES (v26.0):

1. **INDIVIDUAL ENTITIES FIRST**: Each table = One entity (MINIMUM)
   - Do NOT merge tables unless there is CLEAR evidence of duplication
   - Even related tables (email_sends, email_events) are SEPARATE entities
   - Table count = Minimum entity count

2. **WIDE TABLE DECOMPOSITION** (v26.0 신규):
   - If a single table has 20+ columns, it is likely a DENORMALIZED (wide) table
   - Decompose into MULTIPLE LOGICAL ENTITIES based on column semantics
   - Use COLUMN CARDINALITY as primary evidence:
     * Column with unique values close to row_count → likely a PRIMARY KEY → separate entity
     * Column with much fewer unique values → likely a FOREIGN KEY or DIMENSION
     * Group of related columns (same prefix/suffix, semantic similarity) → attribute group
   - Examples:
     * Columns {video_id, title, duration, url} → "Video" entity (video_id as PK)
     * Columns {channel_id, username, follower_count} → "Creator" entity (channel_id as PK)
     * Columns {platform} with 3 unique values → "Platform" dimension entity
     * Columns {view_count, like_count, comment_count, share_count, like_rate} → "PerformanceMetrics" attribute group
   - The GRAIN of the table (which column has 1:1 with row count) defines the primary entity

3. **ENTITY ROLES** (v7.9 확장):
   - master: Core business entities (Customer, Product, Campaign)
   - transaction: Business transactions (Order, Lead, Conversion)
   - event: Time-based activity events (EmailSend, EmailEvent, WebSession, AdClick)
   - metric: Aggregated performance metrics (AdPerformance, CampaignMetrics)
   - reference: Lookup/reference tables
   - bridge: M:N relationship tables
   - attribute_group: Group of columns that form a logical sub-entity within a denormalized table (v26.0)

4. **EVENT TYPE DETECTION** (v7.9 신규):
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
For wide tables (20+ columns), you should produce MULTIPLE entities from a single table (sub-entity decomposition).

Respond with structured JSON matching this schema."""

    async def execute_task(
        self,
        todo: "PipelineTodo",
        context: "SharedContext",
    ) -> "TodoResult":
        """엔티티 분류 작업 실행"""
        from ....todo.models import TodoResult
        from ....shared_context import UnifiedEntity

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
            entity_candidates = []
            for e in extracted_entities:
                # v24.0: inferred_attributes 방어적 변환 강화
                attrs = e.inferred_attributes
                if isinstance(attrs, list):
                    attrs = {a: a for a in attrs if isinstance(a, str)}
                elif not isinstance(attrs, dict):
                    attrs = {}
                entity_candidates.append({
                    "entity_id": e.entity_id,
                    "name": e.name,
                    "source_tables": e.source_tables,
                    "attributes": attrs,
                })
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

        self._report_progress(0.45, "Analyzing column cardinality for sub-entity decomposition")

        # v26.0: 컬럼 카디널리티 분석 — 비정규화 테이블 분해용
        column_cardinality_analysis = {}
        for table_name, table_info in context.tables.items():
            columns = [c.get("name", "") for c in table_info.columns]
            row_count = table_info.row_count or 0
            if row_count < 10 or len(columns) < 15:
                continue  # 작은 테이블은 분해 불필요

            table_data = sample_data.get(table_name, [])
            if not table_data:
                continue

            col_analysis = []
            # v28.5: 루프 전 50 cap — 297 cols × 15K rows = 4.5M ops/table 방지
            # column_profiles 결과도 [:50] cap이므로 이후 컬럼 계산은 낭비
            for col_name in columns[:50]:
                values = [r.get(col_name) for r in table_data if r.get(col_name) is not None]
                unique_count = len(set(str(v) for v in values))
                total_count = len(values)

                # 유니크 비율과 절대 수로 분류
                if total_count == 0:
                    role = "empty"
                elif unique_count == total_count:
                    role = "UNIQUE_KEY (100% unique → primary key candidate)"
                elif unique_count / total_count > 0.8:
                    role = f"HIGH_CARDINALITY ({unique_count} unique / {total_count} rows → possible FK)"
                elif unique_count <= 10:
                    role = f"DIMENSION ({unique_count} unique values → categorical/dimension)"
                elif unique_count / total_count < 0.1:
                    role = f"LOW_CARDINALITY ({unique_count} unique → dimension or flag)"
                else:
                    role = f"MEASURE ({unique_count} unique / {total_count} rows)"

                col_analysis.append({
                    "column": col_name,
                    "unique_count": unique_count,
                    "total_count": total_count,
                    "role": role,
                })

            column_cardinality_analysis[table_name] = {
                "total_columns": len(columns),
                "total_rows": row_count,
                "is_wide_table": len(columns) >= 20,
                "column_profiles": col_analysis[:50],  # 최대 50 컬럼
            }

        cardinality_section = ""
        if column_cardinality_analysis:
            cardinality_section = f"""

COLUMN CARDINALITY ANALYSIS (v26.0 — for wide table decomposition):
{json.dumps(column_cardinality_analysis, indent=2, ensure_ascii=False)}

IMPORTANT: For tables with 20+ columns (is_wide_table=true), you MUST decompose them into
multiple logical entities. Use the cardinality information:
- UNIQUE_KEY columns → primary key of a sub-entity
- HIGH_CARDINALITY columns ending in _id → foreign key pointing to another sub-entity
- DIMENSION columns → dimension entities (e.g., platform with 3 values)
- Groups of MEASURE columns with similar names → metric/measurement attribute groups
"""

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
{cardinality_section}
Domain: {context.get_industry()}

Please:
1. VALIDATE or CORRECT the entity type assignments
2. CONFIRM or ADJUST the entity groups
3. INCORPORATE the hidden entities from column-based extraction
4. SUGGEST canonical names for unified entities
5. IDENTIFY key columns for each entity
6. **v26.0 CRITICAL**: For wide tables (20+ columns), DECOMPOSE into multiple logical entities:
   - Identify GRAIN (primary key) of the table → primary entity
   - Group related columns by semantic meaning → sub-entities
   - Detect FK relationships between sub-entities (e.g., channel_id links Video to Creator)
   - Each sub-entity must have source_tables pointing to the SAME original table
   - Include entity_role "attribute_group" for metric/measurement groups

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
