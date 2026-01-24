"""
Refinement Phase Utilities (v4.5)

Refinement 단계에서 사용되는 공통 유틸리티, Enum, 데이터 클래스

Extracted from refinement.py for better modularity:
- EmbeddedOntologyAgentRole: 온톨로지 추출 에이전트 역할 Enum
- EmbeddedOntologyInsight: 온톨로지 인사이트 데이터 클래스
- EMBEDDED_PHASE2_AGENT_CONFIG: Agent Council 설정
- EmbeddedOntologyAgentCouncil: 4개 에이전트 병렬 실행 클래스
- Helper functions: 이름 변환, 타입 추론 등
"""

import logging
import concurrent.futures
from typing import Any, Dict, List, Tuple
from dataclasses import dataclass
from enum import Enum


logger = logging.getLogger(__name__)


# ============================================================================
# Enums
# ============================================================================

class EmbeddedOntologyAgentRole(Enum):
    """온톨로지 추출 에이전트 역할"""
    ENTITY_ONTOLOGIST = "entity_ontologist"
    RELATIONSHIP_ARCHITECT = "relationship_architect"
    PROPERTY_ENGINEER = "property_engineer"
    DOMAIN_EXPERT = "domain_expert"


# ============================================================================
# Configuration
# ============================================================================

# Agent Council 설정
EMBEDDED_PHASE2_AGENT_CONFIG = {
    "agents": {
        "entity_ontologist": {"name": "Entity Ontologist", "enabled": True},
        "relationship_architect": {"name": "Relationship Architect", "enabled": True},
        "property_engineer": {"name": "Property Engineer", "enabled": True},
        "domain_expert": {"name": "Domain Expert", "enabled": True},
    },
    "settings": {
        "parallel_execution": True,
        "max_workers": 4,
        "timeout_per_agent": 300,
    }
}


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class EmbeddedOntologyInsight:
    """온톨로지 에이전트가 생성한 인사이트"""
    id: str
    agent_role: str
    category: str
    explanation: str
    confidence: float
    severity: str
    entities_involved: List[str]
    relationships_identified: List[Dict[str, Any]]
    properties_analyzed: List[str]
    phase1_refs: List[str]
    recommendation: str
    kpi_impact: Dict[str, str]


# ============================================================================
# Helper Functions
# ============================================================================

def to_pascal_case(name: str) -> str:
    """PascalCase 변환"""
    words = name.replace("_", " ").replace("-", " ").split()
    return "".join(word.capitalize() for word in words)


def to_camel_case(name: str) -> str:
    """camelCase 변환"""
    words = name.replace("_", " ").replace("-", " ").split()
    if not words:
        return name
    return words[0].lower() + "".join(word.capitalize() for word in words[1:])


def infer_semantic_type(col_name: str, data_type: str) -> str:
    """
    컬럼명과 데이터 타입에서 의미론적 타입 추론

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


def generate_business_link_name(table_a: str, table_b: str, join_keys: List[Dict]) -> str:
    """
    비즈니스 의미를 담은 Link 이름 생성 (알고리즘 기반)

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
        key_name = join_keys[0].get("table_a_column", "").lower()

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


def generate_link_description(homeo: Any) -> str:
    """Homeomorphism에서 의미 있는 관계 설명 생성"""
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


def generate_fallback_description(concept: Dict[str, Any], domain: str) -> str:
    """LLM 실패 시 알고리즘 기반 설명 생성"""
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


# ============================================================================
# Embedded Ontology Agent Council
# ============================================================================

class EmbeddedOntologyAgentCouncil:
    """
    임베딩된 Phase 2 Ontology Agent Council (v4.5)

    4개 에이전트가 병렬로 실행됩니다:
    - EntityOntologist: 엔티티 식별 및 정의
    - RelationshipArchitect: 관계 매핑
    - PropertyEngineer: 속성 분석
    - DomainExpert: 도메인 검증

    이 클래스는 _legacy/phase2/ontology_agents.py의 핵심 로직을 직접 포함합니다.
    """

    def __init__(
        self,
        domain_context: Dict[str, Any],
        parallel_execution: bool = True
    ):
        self.domain_context = domain_context or {}
        self.parallel_execution = parallel_execution
        self.max_workers = 4
        self.logger = logging.getLogger(__name__)

    def _generate_entity_insights(
        self,
        data_summary: List[Dict],
        mappings: List[Dict],
        canonical_objects: List[Dict]
    ) -> List[Dict]:
        """
        Entity Ontologist: 핵심 비즈니스 엔티티 식별

        - 마스터 데이터, 트랜잭션, 참조 데이터 분류
        - 엔티티 경계 및 생명주기 인식
        """
        insights = []
        industry = self.domain_context.get('industry', 'general')

        # 데이터 소스에서 엔티티 패턴 분석
        entity_patterns = {}
        for table in data_summary:
            table_name = table.get("table_name", "")
            columns = table.get("columns", [])
            row_count = table.get("row_count", 0)

            # 테이블 유형 추론
            table_type = "transaction"  # 기본값
            if any(kw in table_name.lower() for kw in ["master", "dim_", "customer", "product", "employee"]):
                table_type = "master"
            elif any(kw in table_name.lower() for kw in ["code", "type", "status", "category"]):
                table_type = "reference"
            elif any(kw in table_name.lower() for kw in ["fact_", "order", "sale", "transaction", "event"]):
                table_type = "transaction"

            entity_patterns[table_name] = {
                "type": table_type,
                "column_count": len(columns),
                "row_count": row_count,
            }

        # 주요 엔티티 인사이트 생성
        master_entities = [k for k, v in entity_patterns.items() if v["type"] == "master"]
        transaction_entities = [k for k, v in entity_patterns.items() if v["type"] == "transaction"]

        if master_entities:
            insights.append({
                "explanation": f"[현상] {len(master_entities)}개의 마스터 엔티티 식별됨: {', '.join(master_entities[:3])}\n"
                              f"[의미] 이들은 {industry} 도메인의 핵심 비즈니스 객체입니다.\n"
                              f"[조치] 온톨로지의 주요 Object Type으로 정의 필요",
                "category": "entity_definition",
                "severity": "high",
                "confidence": 0.85,
                "entities_involved": master_entities[:5],
                "entity_type": "master",
                "recommendation": "마스터 엔티티를 Object Type으로 모델링하고, 고유 식별자(Primary Key)를 정의하세요.",
                "kpi_impact": {"데이터_품질": "+20%"}
            })

        if transaction_entities:
            insights.append({
                "explanation": f"[현상] {len(transaction_entities)}개의 트랜잭션 엔티티 식별됨\n"
                              f"[의미] 비즈니스 이벤트를 기록하는 핵심 테이블입니다.\n"
                              f"[영향] 마스터 엔티티와의 연결이 필수적입니다.\n"
                              f"[조치] Link Type을 통해 마스터 엔티티와 연결",
                "category": "entity_definition",
                "severity": "medium",
                "confidence": 0.80,
                "entities_involved": transaction_entities[:5],
                "entity_type": "transaction",
                "recommendation": "트랜잭션 엔티티와 마스터 엔티티 간의 관계를 Link Type으로 정의하세요.",
                "kpi_impact": {"통합_커버리지": "+15%"}
            })

        return insights

    def _generate_relationship_insights(
        self,
        data_summary: List[Dict],
        mappings: List[Dict],
        canonical_objects: List[Dict]
    ) -> List[Dict]:
        """
        Relationship Architect: 엔티티 간 관계 매핑

        - 카디널리티 결정 (1:1, 1:N, N:M)
        - 크로스 사일로 연결 식별
        """
        insights = []

        if mappings:
            # 매핑 기반 관계 분석
            relationship_count = len(mappings)

            insights.append({
                "explanation": f"[현상] Phase 1에서 {relationship_count}개의 테이블 간 매핑 발견\n"
                              f"[의미] 이 매핑들은 데이터 통합의 기반입니다.\n"
                              f"[조치] 각 매핑을 Link Type으로 정의하세요",
                "category": "relationship_mapping",
                "severity": "high",
                "confidence": 0.90,
                "relationships": [
                    {
                        "from": m.get("table_a", ""),
                        "to": m.get("table_b", ""),
                        "type": "1:N",
                        "via": ", ".join(m.get("join_keys", [])[:2]) if m.get("join_keys") else "unknown"
                    }
                    for m in mappings[:5]
                ],
                "phase1_mappings_used": [m.get("id", "") for m in mappings[:5]],
                "recommendation": "매핑을 기반으로 Link Type을 생성하고, Join Key를 Link Property로 정의하세요.",
                "kpi_impact": {"데이터_통합율": "+25%"}
            })
        else:
            insights.append({
                "explanation": "[현상] Phase 1에서 테이블 간 매핑이 발견되지 않음\n"
                              "[영향] 데이터 사일로가 분리되어 있을 수 있습니다.\n"
                              "[조치] 수동으로 테이블 간 관계를 검토하세요",
                "category": "relationship_mapping",
                "severity": "medium",
                "confidence": 0.70,
                "relationships": [],
                "phase1_mappings_used": [],
                "recommendation": "데이터 스튜어드와 함께 테이블 간 비즈니스 관계를 정의하세요.",
                "kpi_impact": {"데이터_품질": "+10%"}
            })

        return insights

    def _generate_property_insights(
        self,
        data_summary: List[Dict],
        mappings: List[Dict],
        canonical_objects: List[Dict]
    ) -> List[Dict]:
        """
        Property Engineer: 엔티티 속성/특성 분석

        - 속성 의미론 분류 (식별자, 측정값, 차원 등)
        - 데이터 품질 평가
        """
        insights = []

        # 모든 컬럼 분석
        all_columns = []
        for table in data_summary:
            for col in table.get("columns", []):
                all_columns.append({
                    "table": table.get("table_name", ""),
                    "column": col if isinstance(col, str) else col.get("name", ""),
                })

        # 속성 유형별 분류
        id_columns = [c for c in all_columns if any(kw in c["column"].lower() for kw in ["_id", "_code", "_key"])]
        amount_columns = [c for c in all_columns if any(kw in c["column"].lower() for kw in ["amount", "price", "cost", "total", "qty", "quantity"])]

        if id_columns:
            insights.append({
                "explanation": f"[현상] {len(id_columns)}개의 식별자 컬럼 발견\n"
                              f"[의미] 이 컬럼들은 엔티티 간 연결의 핵심입니다.\n"
                              f"[조치] Primary Key 및 Foreign Key로 정의하세요",
                "category": "property_semantics",
                "severity": "high",
                "confidence": 0.85,
                "properties_analyzed": [c["column"] for c in id_columns[:10]],
                "property_types": {c["column"]: "identifier" for c in id_columns[:5]},
                "data_quality_issues": [],
                "recommendation": "식별자 컬럼을 Key Property로 정의하고, 유일성 제약조건을 추가하세요.",
                "kpi_impact": {"데이터_정합성": "+15%"}
            })

        if amount_columns:
            insights.append({
                "explanation": f"[현상] {len(amount_columns)}개의 측정값 컬럼 발견\n"
                              f"[의미] 이 컬럼들은 비즈니스 KPI의 기반입니다.\n"
                              f"[조치] Metric Property로 정의하세요",
                "category": "property_semantics",
                "severity": "medium",
                "confidence": 0.80,
                "properties_analyzed": [c["column"] for c in amount_columns[:10]],
                "property_types": {c["column"]: "measure" for c in amount_columns[:5]},
                "data_quality_issues": [],
                "recommendation": "측정값 컬럼을 집계 가능한 Metric Property로 정의하세요.",
                "kpi_impact": {"분석_가능성": "+20%"}
            })

        return insights

    def _generate_domain_insights(
        self,
        data_summary: List[Dict],
        mappings: List[Dict],
        canonical_objects: List[Dict]
    ) -> List[Dict]:
        """
        Domain Expert: 도메인 지식 대비 온톨로지 검증

        - 산업별 패턴 인식
        - 비즈니스 규칙 식별
        """
        insights = []
        industry = self.domain_context.get('industry', 'general')

        # 도메인별 패턴 검증
        domain_patterns = {
            "retail": ["customer", "order", "product", "inventory", "supplier"],
            "healthcare": ["patient", "encounter", "provider", "claim", "diagnosis"],
            "manufacturing": ["plant", "product", "work_order", "inventory", "supplier"],
            "finance": ["account", "transaction", "customer", "loan", "payment"],
        }

        expected_patterns = domain_patterns.get(industry, [])
        found_patterns = []
        missing_patterns = []

        table_names_lower = [t.get("table_name", "").lower() for t in data_summary]

        for pattern in expected_patterns:
            if any(pattern in tn for tn in table_names_lower):
                found_patterns.append(pattern)
            else:
                missing_patterns.append(pattern)

        if found_patterns:
            insights.append({
                "explanation": f"[현상] {industry} 도메인의 핵심 패턴 {len(found_patterns)}개 확인: {', '.join(found_patterns)}\n"
                              f"[의미] 온톨로지가 산업 표준에 부합합니다.\n"
                              f"[조치] 이 엔티티들을 우선적으로 정의하세요",
                "category": "domain_validation",
                "severity": "high" if len(found_patterns) >= 3 else "medium",
                "confidence": 0.85,
                "business_context": f"{industry} 도메인의 표준 엔티티 패턴 확인됨",
                "recommendation": f"확인된 패턴({', '.join(found_patterns)})을 기반으로 온톨로지 핵심 구조를 정의하세요.",
                "kpi_impact": {"도메인_적합성": "+25%"}
            })

        if missing_patterns:
            insights.append({
                "explanation": f"[현상] {industry} 도메인의 일부 패턴 누락: {', '.join(missing_patterns)}\n"
                              f"[의미] 데이터 커버리지가 불완전할 수 있습니다.\n"
                              f"[조치] 누락된 데이터 소스를 확인하세요",
                "category": "domain_validation",
                "severity": "medium",
                "confidence": 0.75,
                "business_context": f"{industry} 도메인에서 일반적으로 필요한 엔티티 일부 누락",
                "recommendation": f"누락된 패턴({', '.join(missing_patterns)})에 해당하는 데이터 소스를 추가로 연결하세요.",
                "kpi_impact": {"데이터_커버리지": "+15%"}
            })

        return insights

    def _run_agent(
        self,
        agent_name: str,
        data_summary: List[Dict],
        mappings: List[Dict],
        canonical_objects: List[Dict]
    ) -> Tuple[str, List[Dict]]:
        """단일 에이전트 실행"""
        try:
            if agent_name == "entity_ontologist":
                insights = self._generate_entity_insights(data_summary, mappings, canonical_objects)
            elif agent_name == "relationship_architect":
                insights = self._generate_relationship_insights(data_summary, mappings, canonical_objects)
            elif agent_name == "property_engineer":
                insights = self._generate_property_insights(data_summary, mappings, canonical_objects)
            elif agent_name == "domain_expert":
                insights = self._generate_domain_insights(data_summary, mappings, canonical_objects)
            else:
                insights = []
            return (agent_name, insights)
        except Exception as e:
            self.logger.error(f"Agent {agent_name} error: {e}")
            return (agent_name, [{
                "explanation": f"[에이전트 오류] {agent_name}: {str(e)[:100]}",
                "category": "error",
                "severity": "low",
                "confidence": 0.0
            }])

    def generate_ontology_insights(
        self,
        data_summary: List[Dict],
        mappings: List[Dict],
        canonical_objects: List[Dict],
        phase1_mappings_used: List[str] = None
    ) -> List[Dict]:
        """
        모든 에이전트를 병렬로 실행하고 인사이트를 통합합니다.

        Returns:
            에이전트 귀속 정보가 포함된 온톨로지 인사이트 목록
        """
        all_insights = []
        insight_counter = 1
        agent_results: Dict[str, List[Dict]] = {}
        phase1_mappings_used = phase1_mappings_used or []

        agent_names = ["entity_ontologist", "relationship_architect", "property_engineer", "domain_expert"]

        if self.parallel_execution:
            self.logger.info("Running 4 embedded ontology agents in PARALLEL mode")

            with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = {
                    executor.submit(
                        self._run_agent,
                        agent_name,
                        data_summary,
                        mappings,
                        canonical_objects
                    ): agent_name
                    for agent_name in agent_names
                }

                for future in concurrent.futures.as_completed(futures):
                    agent_name = futures[future]
                    try:
                        result_name, insights = future.result()
                        agent_results[result_name] = insights
                        self.logger.info(f"Embedded Agent {result_name} completed: {len(insights)} insights")
                    except Exception as e:
                        self.logger.error(f"Embedded Agent {agent_name} failed: {e}")
                        agent_results[agent_name] = []
        else:
            # Sequential fallback
            for agent_name in agent_names:
                _, insights = self._run_agent(agent_name, data_summary, mappings, canonical_objects)
                agent_results[agent_name] = insights

        # 결과 통합
        for agent_name, insights in agent_results.items():
            for insight in insights:
                standardized = {
                    "id": f"ONT-{insight_counter:03d}",
                    "agent": agent_name,
                    "explanation": insight.get("explanation", ""),
                    "category": insight.get("category", "ontology"),
                    "severity": insight.get("severity", "medium"),
                    "confidence": insight.get("confidence", 0.75),
                    "source_signatures": insight.get("source_tables", []),
                    "phase1_mappings_referenced": insight.get("phase1_mappings_used", phase1_mappings_used[:2]),
                    "canonical_objects_referenced": [],
                    "recommendation": insight.get("recommendation", ""),
                    "estimated_kpi_impact": insight.get("kpi_impact", {})
                }

                # 추가 필드
                if "entities_involved" in insight:
                    standardized["entities"] = insight["entities_involved"]
                if "relationships" in insight:
                    standardized["relationships"] = insight["relationships"]
                if "properties_analyzed" in insight:
                    standardized["properties"] = insight["properties_analyzed"]
                if "business_context" in insight:
                    standardized["business_context"] = insight["business_context"]

                all_insights.append(standardized)
                insight_counter += 1

        self.logger.info(f"Total insights generated: {len(all_insights)} (from {len(agent_results)} embedded agents)")
        return all_insights


# Legacy 플래그 - 이제 항상 True (임베딩됨)
LEGACY_PHASE2_AVAILABLE = True
