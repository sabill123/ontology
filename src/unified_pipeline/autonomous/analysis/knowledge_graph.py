"""
Knowledge Graph Builder Module - 지식 그래프 생성 및 관리

Phase 2 (Refinement)와 Phase 3 (Governance)에서 생성된 온톨로지 개념,
관계, 거버넌스 결정을 RDF 트리플로 변환하여 지식 그래프를 구축합니다.

주요 기능:
- OntologyConcept → RDF Class/Property 변환
- HomeomorphismPair → owl:equivalentClass 관계 생성
- GovernanceDecision → 거버넌스 트리플 생성
- RDF/Turtle 포맷 내보내기
- SPARQL 유사 쿼리 지원

Namespaces:
- ont: 온톨로지 개념 (classes, properties)
- data: 데이터 인스턴스
- gov: 거버넌스 메타데이터
- prov: 출처/증거 정보
- qual: 품질/신뢰도 정보
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Any, Iterator, Callable
from datetime import datetime
from enum import Enum
import json
import hashlib
import logging

from .semantic_reasoner import (
    Triple, KnowledgeBase, Rule, RuleType,
    ForwardChainer, RDFSReasoner, OWLReasoner,
    ConsistencyChecker, CompletenessChecker
)

logger = logging.getLogger(__name__)


# =============================================================================
# Namespace 정의
# =============================================================================

class Namespace(Enum):
    """RDF 네임스페이스"""
    RDF = "http://www.w3.org/1999/02/22-rdf-syntax-ns#"
    RDFS = "http://www.w3.org/2000/01/rdf-schema#"
    OWL = "http://www.w3.org/2002/07/owl#"
    XSD = "http://www.w3.org/2001/XMLSchema#"
    SKOS = "http://www.w3.org/2004/02/skos/core#"
    PROV = "http://www.w3.org/ns/prov#"
    DCAT = "http://www.w3.org/ns/dcat#"

    # Custom namespaces for this ontology
    ONT = "http://ontoloty.io/ontology#"           # 온톨로지 개념
    DATA = "http://ontoloty.io/data#"              # 데이터 인스턴스
    GOV = "http://ontoloty.io/governance#"         # 거버넌스
    QUAL = "http://ontoloty.io/quality#"           # 품질/신뢰도
    SCHEMA = "http://ontoloty.io/schema#"          # 스키마 관계


# =============================================================================
# 표준 Predicates
# =============================================================================

class Predicate:
    """표준 술어 정의"""
    # RDF/RDFS/OWL
    TYPE = "rdf:type"
    SUBCLASS_OF = "rdfs:subClassOf"
    SUBPROPERTY_OF = "rdfs:subPropertyOf"
    LABEL = "rdfs:label"
    COMMENT = "rdfs:comment"
    DOMAIN = "rdfs:domain"
    RANGE = "rdfs:range"
    EQUIVALENT_CLASS = "owl:equivalentClass"
    SAME_AS = "owl:sameAs"
    INVERSE_OF = "owl:inverseOf"

    # SKOS (개념 관계)
    BROADER = "skos:broader"
    NARROWER = "skos:narrower"
    RELATED = "skos:related"
    ALT_LABEL = "skos:altLabel"
    PREF_LABEL = "skos:prefLabel"

    # PROV (출처)
    WAS_DERIVED_FROM = "prov:wasDerivedFrom"
    HAD_PRIMARY_SOURCE = "prov:hadPrimarySource"
    WAS_GENERATED_BY = "prov:wasGeneratedBy"

    # Custom Ontoloty predicates
    HAS_SOURCE_TABLE = "ont:hasSourceTable"
    HAS_JOIN_KEY = "ont:hasJoinKey"
    HAS_CONFIDENCE = "qual:hasConfidence"
    HAS_STATUS = "gov:hasStatus"
    HAS_DECISION = "gov:hasDecision"
    HAS_RISK_LEVEL = "gov:hasRiskLevel"
    HAS_POLICY = "gov:hasPolicy"
    MAPS_TO_COLUMN = "schema:mapsToColumn"
    RELATED_TO_TABLE = "schema:relatedToTable"
    HAS_CARDINALITY = "schema:hasCardinality"


# =============================================================================
# Knowledge Graph 노드 타입
# =============================================================================

class NodeType(Enum):
    """노드 타입"""
    CLASS = "owl:Class"                    # 온톨로지 클래스
    OBJECT_PROPERTY = "owl:ObjectProperty" # 객체 속성
    DATA_PROPERTY = "owl:DatatypeProperty" # 데이터 속성
    INDIVIDUAL = "owl:NamedIndividual"     # 인스턴스
    TABLE = "schema:Table"                 # 테이블
    COLUMN = "schema:Column"               # 컬럼
    GOVERNANCE_DECISION = "gov:Decision"   # 거버넌스 결정
    POLICY = "gov:Policy"                  # 정책


# =============================================================================
# Knowledge Graph 구현
# =============================================================================

@dataclass
class KnowledgeGraphStats:
    """지식 그래프 통계"""
    total_triples: int = 0
    asserted_triples: int = 0
    inferred_triples: int = 0
    governance_triples: int = 0

    classes: int = 0
    properties: int = 0
    individuals: int = 0
    relationships: int = 0

    namespaces_used: Set[str] = field(default_factory=set)


class KnowledgeGraphBuilder:
    """
    지식 그래프 빌더

    Phase 2, 3의 결과물을 RDF 트리플로 변환하여 지식 그래프 구축
    """

    def __init__(self, base_uri: str = "http://ontoloty.io/"):
        self.base_uri = base_uri
        self.kb = KnowledgeBase()

        # 생성된 URI 추적
        self._generated_uris: Dict[str, str] = {}

        # 네임스페이스 프리픽스
        self.prefixes = {
            "rdf": Namespace.RDF.value,
            "rdfs": Namespace.RDFS.value,
            "owl": Namespace.OWL.value,
            "xsd": Namespace.XSD.value,
            "skos": Namespace.SKOS.value,
            "prov": Namespace.PROV.value,
            "dcat": Namespace.DCAT.value,
            "ont": Namespace.ONT.value,
            "data": Namespace.DATA.value,
            "gov": Namespace.GOV.value,
            "qual": Namespace.QUAL.value,
            "schema": Namespace.SCHEMA.value,
        }

        # 통계
        self.stats = KnowledgeGraphStats()

    # =========================================================================
    # URI 생성
    # =========================================================================

    def _generate_uri(self, namespace: str, local_name: str) -> str:
        """URI 생성"""
        # 특수문자 제거 및 정규화
        safe_name = local_name.replace(" ", "_").replace("-", "_")
        safe_name = "".join(c for c in safe_name if c.isalnum() or c == "_")

        cache_key = f"{namespace}:{safe_name}"
        if cache_key in self._generated_uris:
            return self._generated_uris[cache_key]

        uri = f"{namespace}:{safe_name}"
        self._generated_uris[cache_key] = uri
        return uri

    def _concept_to_uri(self, concept_id: str) -> str:
        """OntologyConcept ID를 URI로 변환"""
        return self._generate_uri("ont", concept_id)

    def _table_to_uri(self, table_name: str) -> str:
        """테이블명을 URI로 변환"""
        return self._generate_uri("schema", f"Table_{table_name}")

    def _column_to_uri(self, table_name: str, column_name: str) -> str:
        """컬럼을 URI로 변환"""
        return self._generate_uri("schema", f"Column_{table_name}_{column_name}")

    def _decision_to_uri(self, decision_id: str) -> str:
        """거버넌스 결정 ID를 URI로 변환"""
        return self._generate_uri("gov", f"Decision_{decision_id}")

    # =========================================================================
    # Triple 추가 헬퍼
    # =========================================================================

    def _add_triple(
        self,
        subject: str,
        predicate: str,
        obj: str,
        confidence: float = 1.0,
        source: str = "asserted"
    ) -> Triple:
        """트리플 추가"""
        triple = Triple(
            subject=subject,
            predicate=predicate,
            object=obj,
            confidence=confidence,
            source=source
        )
        self.kb.add_triple(triple)

        # 통계 업데이트
        self.stats.total_triples += 1
        if source == "asserted":
            self.stats.asserted_triples += 1
        elif source == "inferred":
            self.stats.inferred_triples += 1
        elif source.startswith("gov"):
            self.stats.governance_triples += 1

        return triple

    # =========================================================================
    # Phase 1: Discovery 결과 변환
    # =========================================================================

    def add_tables_and_columns(
        self,
        tables: Dict[str, Any]
    ) -> List[Triple]:
        """테이블과 컬럼 정보를 트리플로 변환"""
        triples = []

        for table_name, table_info in tables.items():
            table_uri = self._table_to_uri(table_name)

            # 테이블 타입
            triples.append(self._add_triple(
                table_uri,
                Predicate.TYPE,
                NodeType.TABLE.value
            ))

            # 테이블 레이블
            triples.append(self._add_triple(
                table_uri,
                Predicate.LABEL,
                f'"{table_name}"'
            ))

            # 컬럼 정보
            columns = table_info.columns if hasattr(table_info, 'columns') else table_info.get('columns', [])
            for col in columns:
                col_name = col.get("name", "") if isinstance(col, dict) else str(col)
                col_uri = self._column_to_uri(table_name, col_name)

                # 컬럼 타입
                triples.append(self._add_triple(
                    col_uri,
                    Predicate.TYPE,
                    NodeType.COLUMN.value
                ))

                # 컬럼 레이블
                triples.append(self._add_triple(
                    col_uri,
                    Predicate.LABEL,
                    f'"{col_name}"'
                ))

                # 컬럼-테이블 관계
                triples.append(self._add_triple(
                    col_uri,
                    Predicate.RELATED_TO_TABLE,
                    table_uri
                ))

        return triples

    # =========================================================================
    # Phase 2: RDF Triple 생성 (스펙 기반)
    # =========================================================================

    def generate_rdf_triples_from_schema(
        self,
        tables: Dict[str, Any],
        foreign_keys: List[Dict[str, Any]] = None
    ) -> List[Triple]:
        """
        스키마를 RDF 트리플로 변환 (Phase 2 스펙 기반)

        Mapping Rules:
        - Table -> owl:Class
        - Column -> owl:DatatypeProperty
        - FK -> owl:ObjectProperty

        Args:
            tables: 테이블 정보 딕셔너리
            foreign_keys: FK 관계 리스트

        Returns:
            생성된 트리플 목록
        """
        triples = []

        # 1. Table -> owl:Class
        for table_name, table_info in tables.items():
            class_triples = self._table_to_owl_class(table_name, table_info)
            triples.extend(class_triples)

            # 2. Column -> owl:DatatypeProperty
            columns = table_info.columns if hasattr(table_info, 'columns') else table_info.get('columns', [])
            for col in columns:
                col_triples = self._column_to_datatype_property(table_name, col)
                triples.extend(col_triples)

        # 3. FK -> owl:ObjectProperty
        if foreign_keys:
            for fk in foreign_keys:
                fk_triples = self._fk_to_object_property(fk)
                triples.extend(fk_triples)

        logger.info(f"[RDF Generation] Generated {len(triples)} RDF triples from schema")
        return triples

    def _table_to_owl_class(
        self,
        table_name: str,
        table_info: Any
    ) -> List[Triple]:
        """
        Table -> owl:Class 변환

        Args:
            table_name: 테이블명
            table_info: 테이블 정보

        Returns:
            생성된 트리플 목록
        """
        triples = []
        class_uri = self._generate_uri("ont", table_name)

        # rdf:type owl:Class
        triples.append(self._add_triple(
            class_uri,
            Predicate.TYPE,
            NodeType.CLASS.value
        ))
        self.stats.classes += 1

        # rdfs:label
        triples.append(self._add_triple(
            class_uri,
            Predicate.LABEL,
            f'"{table_name}"'
        ))

        # rdfs:comment (설명이 있으면)
        description = None
        if hasattr(table_info, 'description'):
            description = table_info.description
        elif isinstance(table_info, dict):
            description = table_info.get('description')

        if description:
            triples.append(self._add_triple(
                class_uri,
                Predicate.COMMENT,
                f'"{description}"'
            ))

        # 소스 테이블 메타데이터
        table_uri = self._table_to_uri(table_name)
        triples.append(self._add_triple(
            class_uri,
            Predicate.HAD_PRIMARY_SOURCE,
            table_uri
        ))

        return triples

    def _column_to_datatype_property(
        self,
        table_name: str,
        column: Any
    ) -> List[Triple]:
        """
        Column -> owl:DatatypeProperty 변환

        Args:
            table_name: 테이블명
            column: 컬럼 정보

        Returns:
            생성된 트리플 목록
        """
        triples = []

        # 컬럼 정보 추출
        if isinstance(column, dict):
            col_name = column.get('name', column.get('column_name', ''))
            col_type = column.get('type', column.get('data_type', 'string'))
            is_nullable = column.get('nullable', True)
            is_pk = column.get('is_primary_key', False)
        else:
            col_name = str(column)
            col_type = 'string'
            is_nullable = True
            is_pk = False

        if not col_name:
            return triples

        # Property URI
        prop_uri = self._generate_uri("ont", f"{table_name}_{col_name}")
        class_uri = self._generate_uri("ont", table_name)

        # rdf:type owl:DatatypeProperty
        triples.append(self._add_triple(
            prop_uri,
            Predicate.TYPE,
            NodeType.DATA_PROPERTY.value
        ))
        self.stats.properties += 1

        # rdfs:label
        triples.append(self._add_triple(
            prop_uri,
            Predicate.LABEL,
            f'"{col_name}"'
        ))

        # rdfs:domain (어떤 클래스에 속하는지)
        triples.append(self._add_triple(
            prop_uri,
            Predicate.DOMAIN,
            class_uri
        ))

        # rdfs:range (데이터 타입)
        xsd_type = self._sql_to_xsd_type(col_type)
        triples.append(self._add_triple(
            prop_uri,
            Predicate.RANGE,
            xsd_type
        ))

        # 컬럼 매핑 정보
        col_uri = self._column_to_uri(table_name, col_name)
        triples.append(self._add_triple(
            prop_uri,
            Predicate.MAPS_TO_COLUMN,
            col_uri
        ))

        # PK 여부
        if is_pk:
            triples.append(self._add_triple(
                prop_uri,
                "ont:isPrimaryKey",
                '"true"^^xsd:boolean'
            ))

        return triples

    def _fk_to_object_property(
        self,
        fk: Dict[str, Any]
    ) -> List[Triple]:
        """
        FK -> owl:ObjectProperty 변환

        Args:
            fk: FK 정보 딕셔너리

        Returns:
            생성된 트리플 목록
        """
        triples = []

        source_table = fk.get('source_table', fk.get('from_table', ''))
        source_column = fk.get('source_column', fk.get('from_column', ''))
        target_table = fk.get('target_table', fk.get('to_table', ''))
        target_column = fk.get('target_column', fk.get('to_column', ''))
        relationship_name = fk.get('relationship_name', fk.get('name', ''))

        if not (source_table and target_table):
            return triples

        # 관계명 생성 (없으면 자동 생성)
        if not relationship_name:
            relationship_name = f"has{target_table}"

        # Property URI
        prop_uri = self._generate_uri("ont", f"rel_{source_table}_{relationship_name}")
        source_class_uri = self._generate_uri("ont", source_table)
        target_class_uri = self._generate_uri("ont", target_table)

        # rdf:type owl:ObjectProperty
        triples.append(self._add_triple(
            prop_uri,
            Predicate.TYPE,
            NodeType.OBJECT_PROPERTY.value
        ))
        self.stats.properties += 1

        # rdfs:label
        triples.append(self._add_triple(
            prop_uri,
            Predicate.LABEL,
            f'"{relationship_name}"'
        ))

        # rdfs:domain (소스 테이블)
        triples.append(self._add_triple(
            prop_uri,
            Predicate.DOMAIN,
            source_class_uri
        ))

        # rdfs:range (타겟 테이블)
        triples.append(self._add_triple(
            prop_uri,
            Predicate.RANGE,
            target_class_uri
        ))

        # FK 메타데이터
        if source_column:
            source_col_uri = self._column_to_uri(source_table, source_column)
            triples.append(self._add_triple(
                prop_uri,
                "ont:sourceColumn",
                source_col_uri
            ))

        if target_column:
            target_col_uri = self._column_to_uri(target_table, target_column)
            triples.append(self._add_triple(
                prop_uri,
                "ont:targetColumn",
                target_col_uri
            ))

        # Inverse relation 자동 생성
        inverse_triples = self._generate_inverse_relation(
            prop_uri, relationship_name, source_class_uri, target_class_uri
        )
        triples.extend(inverse_triples)

        self.stats.relationships += 1

        return triples

    def _generate_inverse_relation(
        self,
        prop_uri: str,
        relationship_name: str,
        source_class_uri: str,
        target_class_uri: str
    ) -> List[Triple]:
        """
        역관계 자동 생성

        Args:
            prop_uri: 원본 속성 URI
            relationship_name: 관계명
            source_class_uri: 소스 클래스 URI
            target_class_uri: 타겟 클래스 URI

        Returns:
            역관계 트리플 목록
        """
        triples = []

        # 역관계명 생성
        inverse_name = f"isReferencedBy{relationship_name.replace('has', '')}"
        inverse_uri = self._generate_uri("ont", f"inv_{inverse_name}")

        # rdf:type owl:ObjectProperty
        triples.append(self._add_triple(
            inverse_uri,
            Predicate.TYPE,
            NodeType.OBJECT_PROPERTY.value
        ))

        # rdfs:label
        triples.append(self._add_triple(
            inverse_uri,
            Predicate.LABEL,
            f'"{inverse_name}"'
        ))

        # Domain/Range (역방향)
        triples.append(self._add_triple(
            inverse_uri,
            Predicate.DOMAIN,
            target_class_uri
        ))
        triples.append(self._add_triple(
            inverse_uri,
            Predicate.RANGE,
            source_class_uri
        ))

        # owl:inverseOf 관계
        triples.append(self._add_triple(
            prop_uri,
            Predicate.INVERSE_OF,
            inverse_uri
        ))
        triples.append(self._add_triple(
            inverse_uri,
            Predicate.INVERSE_OF,
            prop_uri
        ))

        return triples

    def _sql_to_xsd_type(self, sql_type: str) -> str:
        """
        SQL 타입을 XSD 타입으로 변환

        Args:
            sql_type: SQL 데이터 타입

        Returns:
            XSD 타입 URI
        """
        sql_type_lower = sql_type.lower()

        type_mapping = {
            # 정수
            'int': 'xsd:integer',
            'integer': 'xsd:integer',
            'bigint': 'xsd:long',
            'smallint': 'xsd:short',
            'tinyint': 'xsd:byte',

            # 실수
            'float': 'xsd:float',
            'double': 'xsd:double',
            'decimal': 'xsd:decimal',
            'numeric': 'xsd:decimal',
            'real': 'xsd:float',

            # 문자열
            'varchar': 'xsd:string',
            'char': 'xsd:string',
            'text': 'xsd:string',
            'nvarchar': 'xsd:string',
            'nchar': 'xsd:string',
            'string': 'xsd:string',

            # 날짜/시간
            'date': 'xsd:date',
            'datetime': 'xsd:dateTime',
            'timestamp': 'xsd:dateTime',
            'time': 'xsd:time',

            # 불린
            'boolean': 'xsd:boolean',
            'bool': 'xsd:boolean',
            'bit': 'xsd:boolean',

            # 기타
            'binary': 'xsd:base64Binary',
            'blob': 'xsd:base64Binary',
            'uuid': 'xsd:string',
        }

        for key, xsd in type_mapping.items():
            if key in sql_type_lower:
                return xsd

        return 'xsd:string'

    def add_homeomorphisms(
        self,
        homeomorphisms: List[Any]
    ) -> List[Triple]:
        """Homeomorphism 관계를 트리플로 변환"""
        triples = []

        for h in homeomorphisms:
            # HomeomorphismPair 또는 dict 처리
            if hasattr(h, 'table_a'):
                table_a, table_b = h.table_a, h.table_b
                is_homeo = h.is_homeomorphic
                confidence = h.confidence
                join_keys = h.join_keys
                entity_type = h.entity_type
            else:
                table_a = h.get('table_a', '')
                table_b = h.get('table_b', '')
                is_homeo = h.get('is_homeomorphic', False)
                confidence = h.get('confidence', 0.0)
                join_keys = h.get('join_keys', [])
                entity_type = h.get('entity_type', 'Unknown')

            if not is_homeo:
                continue

            table_a_uri = self._table_to_uri(table_a)
            table_b_uri = self._table_to_uri(table_b)

            # owl:equivalentClass 관계
            triples.append(self._add_triple(
                table_a_uri,
                Predicate.EQUIVALENT_CLASS,
                table_b_uri,
                confidence=confidence
            ))

            # 신뢰도 추가
            # Homeomorphism 노드 생성
            homeo_uri = self._generate_uri("ont", f"Homeomorphism_{table_a}_{table_b}")
            triples.append(self._add_triple(
                homeo_uri,
                Predicate.TYPE,
                "ont:Homeomorphism"
            ))
            triples.append(self._add_triple(
                homeo_uri,
                Predicate.HAS_CONFIDENCE,
                f'"{confidence}"^^xsd:float'
            ))

            # 조인 키 관계
            for jk in join_keys:
                col_a = jk.get('table_a_column', '')
                col_b = jk.get('table_b_column', '')

                if col_a and col_b:
                    col_a_uri = self._column_to_uri(table_a, col_a)
                    col_b_uri = self._column_to_uri(table_b, col_b)

                    # sameAs 관계 (같은 값을 참조하는 컬럼)
                    triples.append(self._add_triple(
                        col_a_uri,
                        Predicate.SAME_AS,
                        col_b_uri,
                        confidence=jk.get('overlap_ratio', confidence)
                    ))

                    # join key 관계
                    triples.append(self._add_triple(
                        homeo_uri,
                        Predicate.HAS_JOIN_KEY,
                        col_a_uri
                    ))

            # 엔티티 타입이 있으면 추가
            if entity_type and entity_type != "Unknown":
                entity_uri = self._generate_uri("ont", entity_type)
                triples.append(self._add_triple(
                    table_a_uri,
                    Predicate.TYPE,
                    entity_uri
                ))
                triples.append(self._add_triple(
                    table_b_uri,
                    Predicate.TYPE,
                    entity_uri
                ))

            self.stats.relationships += 1

        return triples

    # =========================================================================
    # Phase 2: Refinement 결과 변환
    # =========================================================================

    def add_ontology_concepts(
        self,
        concepts: List[Any]
    ) -> List[Triple]:
        """OntologyConcept을 RDF 클래스/속성으로 변환"""
        triples = []

        for concept in concepts:
            # OntologyConcept 또는 dict 처리
            if hasattr(concept, 'concept_id'):
                concept_id = concept.concept_id
                concept_type = concept.concept_type
                name = concept.name
                description = concept.description
                source_tables = concept.source_tables
                confidence = concept.confidence
                status = concept.status
            else:
                concept_id = concept.get('concept_id', '')
                concept_type = concept.get('concept_type', '')
                name = concept.get('name', '')
                description = concept.get('description', '')
                source_tables = concept.get('source_tables', [])
                confidence = concept.get('confidence', 0.0)
                status = concept.get('status', 'pending')

            concept_uri = self._concept_to_uri(concept_id)

            # 개념 타입에 따른 RDF 타입 결정
            if concept_type == "object_type":
                rdf_type = NodeType.CLASS.value
                self.stats.classes += 1
            elif concept_type == "property":
                rdf_type = NodeType.DATA_PROPERTY.value
                self.stats.properties += 1
            elif concept_type == "link_type":
                rdf_type = NodeType.OBJECT_PROPERTY.value
                self.stats.properties += 1
            else:
                rdf_type = NodeType.CLASS.value
                self.stats.classes += 1

            # 타입 트리플
            triples.append(self._add_triple(
                concept_uri,
                Predicate.TYPE,
                rdf_type
            ))

            # 레이블
            triples.append(self._add_triple(
                concept_uri,
                Predicate.LABEL,
                f'"{name}"'
            ))

            # 설명
            if description:
                triples.append(self._add_triple(
                    concept_uri,
                    Predicate.COMMENT,
                    f'"{description}"'
                ))

            # 신뢰도
            triples.append(self._add_triple(
                concept_uri,
                Predicate.HAS_CONFIDENCE,
                f'"{confidence}"^^xsd:float'
            ))

            # 상태 (거버넌스)
            triples.append(self._add_triple(
                concept_uri,
                Predicate.HAS_STATUS,
                f'gov:{status}'
            ))

            # 소스 테이블 (출처)
            for table in source_tables:
                table_uri = self._table_to_uri(table)
                triples.append(self._add_triple(
                    concept_uri,
                    Predicate.HAD_PRIMARY_SOURCE,
                    table_uri
                ))

        return triples

    def add_concept_relationships(
        self,
        relationships: List[Dict[str, Any]]
    ) -> List[Triple]:
        """개념 간 관계를 트리플로 변환"""
        triples = []

        for rel in relationships:
            from_entity = rel.get('from_entity', '')
            to_entity = rel.get('to_entity', '')
            rel_name = rel.get('relationship_name', 'relatedTo')
            cardinality = rel.get('cardinality', '')

            if not from_entity or not to_entity:
                continue

            from_uri = self._generate_uri("ont", from_entity)
            to_uri = self._generate_uri("ont", to_entity)

            # 관계 속성 생성
            rel_uri = self._generate_uri("ont", f"rel_{from_entity}_{rel_name}_{to_entity}")

            # 관계 타입
            triples.append(self._add_triple(
                rel_uri,
                Predicate.TYPE,
                NodeType.OBJECT_PROPERTY.value
            ))

            # 관계 레이블
            triples.append(self._add_triple(
                rel_uri,
                Predicate.LABEL,
                f'"{rel_name}"'
            ))

            # Domain과 Range
            triples.append(self._add_triple(
                rel_uri,
                Predicate.DOMAIN,
                from_uri
            ))
            triples.append(self._add_triple(
                rel_uri,
                Predicate.RANGE,
                to_uri
            ))

            # Cardinality 메타데이터
            if cardinality:
                triples.append(self._add_triple(
                    rel_uri,
                    Predicate.HAS_CARDINALITY,
                    f'"{cardinality}"'
                ))

            self.stats.relationships += 1

        return triples

    def add_inferred_triples(
        self,
        inferences: List[Triple]
    ) -> List[Triple]:
        """추론된 트리플 추가"""
        triples = []

        for inf in inferences:
            triple = self._add_triple(
                inf.subject,
                inf.predicate,
                inf.object,
                confidence=inf.confidence,
                source="inferred"
            )
            triple.rule_applied = inf.rule_applied
            triples.append(triple)

        return triples

    # =========================================================================
    # Forward Chaining 추론 엔진 (v4.5)
    # =========================================================================

    def run_inference(self, max_iterations: int = 50) -> List[Triple]:
        """
        Forward Chaining 추론 실행

        RDFS/OWL 스타일 추론 규칙을 적용하여 새로운 트리플 생성

        v4.5 추론 규칙:
        1. owl:equivalentClass 전이성
        2. rdfs:subClassOf 전이성
        3. owl:sameAs 대칭/전이성
        4. 도메인/범위 추론
        5. 역관계 추론

        Args:
            max_iterations: 최대 반복 횟수

        Returns:
            추론된 새 트리플 목록
        """
        from .semantic_reasoner import ForwardChainer, Rule, RuleType
        import logging
        logger = logging.getLogger(__name__)

        # v4.6: KB 상태 로깅
        logger.info(f"[Inference] KB has {len(self.kb.triples)} triples before inference")

        # owl:equivalentClass 트리플 확인
        equiv_triples = self.kb.query(predicate="owl:equivalentClass")
        logger.info(f"[Inference] Found {len(equiv_triples)} owl:equivalentClass triples")
        for t in equiv_triples[:5]:
            logger.debug(f"  - {t.subject} -> {t.object}")

        # owl:sameAs 트리플 확인
        sameas_triples = self.kb.query(predicate="owl:sameAs")
        logger.info(f"[Inference] Found {len(sameas_triples)} owl:sameAs triples")

        # 추론 규칙 정의
        inference_rules = self._get_inference_rules()
        logger.info(f"[Inference] Loaded {len(inference_rules)} inference rules")

        # Forward Chainer 생성 및 실행
        chainer = ForwardChainer(self.kb, inference_rules)
        inferred_triples = chainer.infer(max_iterations=max_iterations)

        logger.info(f"[Inference] Generated {len(inferred_triples)} inferred triples")

        # 통계 업데이트
        for triple in inferred_triples:
            self.stats.inferred_triples += 1
            self.stats.total_triples += 1
            logger.debug(f"  - Inferred: {triple.subject} --{triple.predicate}--> {triple.object} (rule: {triple.rule_applied})")

        return inferred_triples

    def _get_inference_rules(self) -> List:
        """
        RDFS/OWL 추론 규칙 정의 (Phase 2 스펙 기반)

        Phase 2 스펙 추론 규칙:
        - Rule 1: Transitivity (A sameAs B, B sameAs C -> A sameAs C)
        - Rule 2: SubClass inheritance (A subClassOf B, x instance of A -> x instance of B)
        - Rule 3: Inverse relation auto-generation

        Returns:
            Rule 객체 목록
        """
        rules = []

        # =================================================================
        # Phase 2 스펙 필수 규칙
        # =================================================================

        # [스펙 Rule 1] owl:sameAs 전이성 (Transitivity)
        # A sameAs B, B sameAs C -> A sameAs C
        rules.append(Rule(
            id="spec_rule_1_sameas_transitivity",
            name="[Spec] sameAs Transitivity",
            rule_type=RuleType.OWL_TRANSITIVE,
            antecedent=[
                ("?a", "owl:sameAs", "?b"),
                ("?b", "owl:sameAs", "?c")
            ],
            consequent=[("?a", "owl:sameAs", "?c")],
            confidence=1.0,
            priority=100
        ))

        # [스펙 Rule 2] SubClass inheritance
        # A subClassOf B, x instance of A -> x instance of B
        rules.append(Rule(
            id="spec_rule_2_subclass_inheritance",
            name="[Spec] SubClass Instance Inheritance",
            rule_type=RuleType.RDFS_SUBCLASS,
            antecedent=[
                ("?a", "rdfs:subClassOf", "?b"),
                ("?x", "rdf:type", "?a")
            ],
            consequent=[("?x", "rdf:type", "?b")],
            confidence=1.0,
            priority=100
        ))

        # [스펙 Rule 3] Inverse relation auto-generation
        # P inverseOf Q, X P Y -> Y Q X
        rules.append(Rule(
            id="spec_rule_3_inverse_relation",
            name="[Spec] Inverse Relation Auto-Generation",
            rule_type=RuleType.OWL_INVERSE,
            antecedent=[
                ("?p", "owl:inverseOf", "?q"),
                ("?x", "?p", "?y")
            ],
            consequent=[("?y", "?q", "?x")],
            confidence=1.0,
            priority=100
        ))

        # =================================================================
        # 추가 OWL/RDFS 추론 규칙
        # =================================================================

        # owl:sameAs 대칭성
        # A sameAs B -> B sameAs A
        rules.append(Rule(
            id="owl_sameas_symmetric",
            name="owl:sameAs Symmetric",
            rule_type=RuleType.OWL_SYMMETRIC,
            antecedent=[("?x", "owl:sameAs", "?y")],
            consequent=[("?y", "owl:sameAs", "?x")],
            confidence=1.0,
            priority=95
        ))

        # owl:equivalentClass 대칭성
        # A equivalentClass B -> B equivalentClass A
        rules.append(Rule(
            id="owl_equiv_symmetric",
            name="owl:equivalentClass Symmetric",
            rule_type=RuleType.OWL_SYMMETRIC,
            antecedent=[("?x", "owl:equivalentClass", "?y")],
            consequent=[("?y", "owl:equivalentClass", "?x")],
            confidence=1.0,
            priority=90
        ))

        # owl:equivalentClass 전이성
        # A equivalentClass B, B equivalentClass C -> A equivalentClass C
        rules.append(Rule(
            id="owl_equiv_transitive",
            name="owl:equivalentClass Transitive",
            rule_type=RuleType.OWL_TRANSITIVE,
            antecedent=[
                ("?x", "owl:equivalentClass", "?y"),
                ("?y", "owl:equivalentClass", "?z")
            ],
            consequent=[("?x", "owl:equivalentClass", "?z")],
            confidence=0.95,
            priority=85
        ))

        # rdfs:subClassOf 전이성
        # A subClassOf B, B subClassOf C -> A subClassOf C
        rules.append(Rule(
            id="rdfs_subclass_transitive",
            name="rdfs:subClassOf Transitive",
            rule_type=RuleType.OWL_TRANSITIVE,
            antecedent=[
                ("?x", "rdfs:subClassOf", "?y"),
                ("?y", "rdfs:subClassOf", "?z")
            ],
            consequent=[("?x", "rdfs:subClassOf", "?z")],
            confidence=1.0,
            priority=80
        ))

        # rdfs:subPropertyOf 전이성
        # P subPropertyOf Q, Q subPropertyOf R -> P subPropertyOf R
        rules.append(Rule(
            id="rdfs_subproperty_transitive",
            name="rdfs:subPropertyOf Transitive",
            rule_type=RuleType.RDFS_SUBPROPERTY,
            antecedent=[
                ("?p", "rdfs:subPropertyOf", "?q"),
                ("?q", "rdfs:subPropertyOf", "?r")
            ],
            consequent=[("?p", "rdfs:subPropertyOf", "?r")],
            confidence=1.0,
            priority=75
        ))

        # rdfs:subPropertyOf 상속
        # P subPropertyOf Q, X P Y -> X Q Y
        rules.append(Rule(
            id="rdfs_subproperty_inheritance",
            name="rdfs:subPropertyOf Inheritance",
            rule_type=RuleType.RDFS_SUBPROPERTY,
            antecedent=[
                ("?p", "rdfs:subPropertyOf", "?q"),
                ("?x", "?p", "?y")
            ],
            consequent=[("?x", "?q", "?y")],
            confidence=1.0,
            priority=70
        ))

        # equivalentClass + type -> type 전파
        # X type A, A equivalentClass B -> X type B
        rules.append(Rule(
            id="equiv_type_propagation",
            name="Type Propagation via equivalentClass",
            rule_type=RuleType.OWL_EQUIVALENT,
            antecedent=[
                ("?x", "rdf:type", "?a"),
                ("?a", "owl:equivalentClass", "?b")
            ],
            consequent=[("?x", "rdf:type", "?b")],
            confidence=0.9,
            priority=65
        ))

        # rdfs:domain 추론
        # P domain C, X P Y -> X type C
        rules.append(Rule(
            id="rdfs_domain_inference",
            name="Domain Type Inference",
            rule_type=RuleType.RDFS_DOMAIN,
            antecedent=[
                ("?p", "rdfs:domain", "?c"),
                ("?x", "?p", "?y")
            ],
            consequent=[("?x", "rdf:type", "?c")],
            confidence=0.9,
            priority=60
        ))

        # rdfs:range 추론
        # P range C, X P Y -> Y type C
        rules.append(Rule(
            id="rdfs_range_inference",
            name="Range Type Inference",
            rule_type=RuleType.RDFS_RANGE,
            antecedent=[
                ("?p", "rdfs:range", "?c"),
                ("?x", "?p", "?y")
            ],
            consequent=[("?y", "rdf:type", "?c")],
            confidence=0.9,
            priority=55
        ))

        # =================================================================
        # 도메인 특화 규칙 (스키마/온톨로지)
        # =================================================================

        # 관련 테이블 추론 (schema:relatedToTable 전이)
        # Column A relatedToTable T, T equivalentClass T2 -> Column A relatedToTable T2
        rules.append(Rule(
            id="related_table_propagation",
            name="Related Table Propagation",
            rule_type=RuleType.CUSTOM,
            antecedent=[
                ("?col", "schema:relatedToTable", "?table1"),
                ("?table1", "owl:equivalentClass", "?table2")
            ],
            consequent=[("?col", "schema:relatedToTable", "?table2")],
            confidence=0.85,
            priority=50
        ))

        # sameAs를 통한 속성 전파
        # X sameAs Y, X hasLabel L -> Y hasAltLabel L
        rules.append(Rule(
            id="sameas_property_propagation",
            name="Property Propagation via sameAs",
            rule_type=RuleType.OWL_SYMMETRIC,
            antecedent=[
                ("?x", "owl:sameAs", "?y"),
                ("?x", "rdfs:label", "?label")
            ],
            consequent=[("?y", "skos:altLabel", "?label")],
            confidence=0.8,
            priority=45
        ))

        # 조인 키 대칭성
        # Homeo hasJoinKey ColA, ColA sameAs ColB -> Homeo hasJoinKey ColB
        rules.append(Rule(
            id="joinkey_sameas_propagation",
            name="Join Key via sameAs",
            rule_type=RuleType.CUSTOM,
            antecedent=[
                ("?homeo", "ont:hasJoinKey", "?colA"),
                ("?colA", "owl:sameAs", "?colB")
            ],
            consequent=[("?homeo", "ont:hasJoinKey", "?colB")],
            confidence=0.85,
            priority=40
        ))

        # 동일 엔티티 타입 추론
        # TableA type EntityX, TableA equivalentClass TableB -> TableB type EntityX
        rules.append(Rule(
            id="entity_type_via_equiv",
            name="Entity Type via equivalentClass",
            rule_type=RuleType.OWL_EQUIVALENT,
            antecedent=[
                ("?tableA", "rdf:type", "?entityType"),
                ("?tableA", "owl:equivalentClass", "?tableB")
            ],
            consequent=[("?tableB", "rdf:type", "?entityType")],
            confidence=0.9,
            priority=35
        ))

        # owl:inverseOf 대칭성
        # P inverseOf Q -> Q inverseOf P
        rules.append(Rule(
            id="owl_inverse_symmetric",
            name="owl:inverseOf Symmetric",
            rule_type=RuleType.OWL_SYMMETRIC,
            antecedent=[("?p", "owl:inverseOf", "?q")],
            consequent=[("?q", "owl:inverseOf", "?p")],
            confidence=1.0,
            priority=30
        ))

        return rules

    # =========================================================================
    # Phase 3: Governance 결과 변환
    # =========================================================================

    def add_governance_decisions(
        self,
        decisions: List[Any]
    ) -> List[Triple]:
        """거버넌스 결정을 트리플로 변환"""
        triples = []

        for decision in decisions:
            # GovernanceDecision 또는 dict 처리
            if hasattr(decision, 'decision_id'):
                decision_id = decision.decision_id
                concept_id = decision.concept_id
                decision_type = decision.decision_type
                confidence = decision.confidence
                reasoning = decision.reasoning
            else:
                decision_id = decision.get('decision_id', '')
                concept_id = decision.get('concept_id', '')
                decision_type = decision.get('decision_type', '')
                confidence = decision.get('confidence', 0.0)
                reasoning = decision.get('reasoning', '')

            decision_uri = self._decision_to_uri(decision_id)
            concept_uri = self._concept_to_uri(concept_id)

            # 결정 타입
            triples.append(self._add_triple(
                decision_uri,
                Predicate.TYPE,
                NodeType.GOVERNANCE_DECISION.value,
                source="governance"
            ))

            # 대상 개념
            triples.append(self._add_triple(
                decision_uri,
                "gov:forConcept",
                concept_uri,
                source="governance"
            ))

            # 결정 유형
            triples.append(self._add_triple(
                decision_uri,
                Predicate.HAS_DECISION,
                f'gov:{decision_type}',
                source="governance"
            ))

            # 신뢰도
            triples.append(self._add_triple(
                decision_uri,
                Predicate.HAS_CONFIDENCE,
                f'"{confidence}"^^xsd:float',
                source="governance"
            ))

            # 개념에 상태 업데이트
            if decision_type == "approve":
                triples.append(self._add_triple(
                    concept_uri,
                    Predicate.HAS_STATUS,
                    "gov:approved",
                    source="governance"
                ))
            elif decision_type == "reject":
                triples.append(self._add_triple(
                    concept_uri,
                    Predicate.HAS_STATUS,
                    "gov:rejected",
                    source="governance"
                ))

        return triples

    def add_policy_rules(
        self,
        policies: List[Dict[str, Any]]
    ) -> List[Triple]:
        """정책 규칙을 트리플로 변환"""
        triples = []

        for policy in policies:
            rule_id = policy.get('rule_id', '')
            rule_type = policy.get('rule_type', '')
            target_concept = policy.get('target_concept', '')
            severity = policy.get('severity', 'medium')

            if not rule_id:
                continue

            policy_uri = self._generate_uri("gov", f"Policy_{rule_id}")

            # 정책 타입
            triples.append(self._add_triple(
                policy_uri,
                Predicate.TYPE,
                NodeType.POLICY.value,
                source="governance"
            ))

            # 정책 유형
            triples.append(self._add_triple(
                policy_uri,
                "gov:policyType",
                f'gov:{rule_type}',
                source="governance"
            ))

            # 대상 개념
            if target_concept:
                concept_uri = self._concept_to_uri(target_concept)
                triples.append(self._add_triple(
                    policy_uri,
                    Predicate.HAS_POLICY,
                    concept_uri,
                    source="governance"
                ))

            # 심각도
            triples.append(self._add_triple(
                policy_uri,
                Predicate.HAS_RISK_LEVEL,
                f'gov:{severity}',
                source="governance"
            ))

        return triples

    # =========================================================================
    # 전체 빌드
    # =========================================================================

    def build_from_context(self, context: Any) -> "KnowledgeGraphBuilder":
        """
        SharedContext에서 전체 지식 그래프 빌드

        Args:
            context: SharedContext 객체

        Returns:
            self (체이닝 지원)
        """
        # Phase 1: Discovery 결과
        if context.tables:
            self.add_tables_and_columns(context.tables)

        if context.homeomorphisms:
            self.add_homeomorphisms(context.homeomorphisms)

        # =================================================================
        # Phase 2: RDF Triple 생성 (스펙 기반)
        # Table -> owl:Class, Column -> owl:DatatypeProperty, FK -> owl:ObjectProperty
        # =================================================================
        if context.tables:
            # FK 정보 추출 (context에서 가져오기)
            foreign_keys = []
            if hasattr(context, 'foreign_keys') and context.foreign_keys:
                foreign_keys = context.foreign_keys
            elif hasattr(context, 'discovered_relationships') and context.discovered_relationships:
                # discovered_relationships에서 FK 정보 추출
                for rel in context.discovered_relationships:
                    if isinstance(rel, dict) and rel.get('type') == 'foreign_key':
                        foreign_keys.append(rel)

            rdf_triples = self.generate_rdf_triples_from_schema(context.tables, foreign_keys)
            logger.info(f"[KG Build] Generated {len(rdf_triples)} RDF triples from schema (Phase 2 Spec)")

        # Phase 2: Refinement 결과
        if context.ontology_concepts:
            self.add_ontology_concepts(context.ontology_concepts)

        if context.concept_relationships:
            self.add_concept_relationships(context.concept_relationships)

        # Phase 3: Governance 결과
        if context.governance_decisions:
            self.add_governance_decisions(context.governance_decisions)

        if context.policy_rules:
            self.add_policy_rules(context.policy_rules)

        # OntologyArchitect 트리플
        kg_count = 0
        if hasattr(context, 'knowledge_graph_triples') and context.knowledge_graph_triples:
            for triple_dict in context.knowledge_graph_triples:
                self._add_dict_triple(triple_dict)
                kg_count += 1
        logger.info(f"[KG Build] Added {kg_count} knowledge_graph_triples")

        # SemanticValidator 기본 트리플
        sem_count = 0
        if hasattr(context, 'semantic_base_triples') and context.semantic_base_triples:
            for triple_dict in context.semantic_base_triples:
                self._add_dict_triple(triple_dict)
                sem_count += 1
        logger.info(f"[KG Build] Added {sem_count} semantic_base_triples")

        # 추론된 트리플 (이전 실행에서 저장된 것)
        prev_inf_count = 0
        if hasattr(context, 'inferred_triples') and context.inferred_triples:
            for triple_dict in context.inferred_triples:
                self._add_dict_triple(triple_dict, is_inferred=True)
                prev_inf_count += 1
        logger.info(f"[KG Build] Added {prev_inf_count} previously inferred triples")

        # Governance 트리플
        gov_count = 0
        if hasattr(context, 'governance_triples') and context.governance_triples:
            for triple_dict in context.governance_triples:
                self._add_dict_triple(triple_dict)
                gov_count += 1
        logger.info(f"[KG Build] Added {gov_count} governance_triples")

        # v4.7: KB 상태 확인
        logger.info(f"[KG Build] Total KB triples before inference: {len(self.kb.triples)}")
        equiv_check = self.kb.query(predicate="owl:equivalentClass")
        sameas_check = self.kb.query(predicate="owl:sameAs")
        logger.info(f"[KG Build] owl:equivalentClass: {len(equiv_check)}, owl:sameAs: {len(sameas_check)}")

        # v4.5: Forward Chaining 추론 실행
        inferred = self.run_inference()
        if inferred:
            # 추론 결과를 context에도 저장
            if hasattr(context, 'inferred_triples'):
                for triple in inferred:
                    context.inferred_triples.append({
                        "subject": triple.subject,
                        "predicate": triple.predicate,
                        "object": triple.object,
                        "confidence": triple.confidence,
                        "source": "inferred",
                        "rule_applied": triple.rule_applied,
                    })

        return self

    def _add_dict_triple(self, triple_dict: Dict[str, Any], is_inferred: bool = False) -> None:
        """딕셔너리 형태의 트리플을 KnowledgeBase에 추가"""
        subject = triple_dict.get("subject", "")
        predicate = triple_dict.get("predicate", "")
        obj = triple_dict.get("object", "")
        confidence = triple_dict.get("confidence", 1.0)
        source = triple_dict.get("source", "unknown")

        if subject and predicate and obj:
            # is_inferred일 경우 source를 "inferred"로 변경
            if is_inferred:
                source = "inferred"

            triple = Triple(
                subject=subject,
                predicate=predicate,
                object=obj,
                confidence=confidence,
                source=source,
            )
            self.kb.add_triple(triple)
            self.stats.total_triples += 1
            if is_inferred:
                self.stats.inferred_triples += 1

    # =========================================================================
    # 쿼리
    # =========================================================================

    def query(
        self,
        subject: str = None,
        predicate: str = None,
        obj: str = None
    ) -> List[Triple]:
        """패턴 매칭 쿼리"""
        return self.kb.query(subject, predicate, obj)

    def get_all_triples(self) -> List[Triple]:
        """모든 트리플 반환"""
        return list(self.kb.triple_details.values())

    def get_classes(self) -> List[str]:
        """모든 클래스 URI 반환"""
        class_triples = self.query(predicate=Predicate.TYPE, obj=NodeType.CLASS.value)
        return [t.subject for t in class_triples]

    def get_properties(self) -> List[str]:
        """모든 속성 URI 반환"""
        obj_props = self.query(predicate=Predicate.TYPE, obj=NodeType.OBJECT_PROPERTY.value)
        data_props = self.query(predicate=Predicate.TYPE, obj=NodeType.DATA_PROPERTY.value)
        return [t.subject for t in obj_props + data_props]

    def get_stats(self) -> KnowledgeGraphStats:
        """통계 반환"""
        self.stats.namespaces_used = set(self.prefixes.keys())
        return self.stats

    # =========================================================================
    # 내보내기
    # =========================================================================

    def to_turtle(self) -> str:
        """Turtle 포맷으로 내보내기"""
        lines = []

        # Prefixes
        lines.append("# Knowledge Graph generated by Ontoloty")
        lines.append(f"# Generated: {datetime.now().isoformat()}")
        lines.append(f"# Total triples: {self.stats.total_triples}")
        lines.append("")

        for prefix, uri in self.prefixes.items():
            lines.append(f"@prefix {prefix}: <{uri}> .")
        lines.append("")

        # 주어별로 그룹화
        subjects: Dict[str, List[Tuple[str, str]]] = {}
        for triple in self.get_all_triples():
            if triple.subject not in subjects:
                subjects[triple.subject] = []
            subjects[triple.subject].append((triple.predicate, triple.object))

        # Turtle 출력
        for subject, predicates in subjects.items():
            lines.append(f"{subject}")
            for i, (pred, obj) in enumerate(predicates):
                separator = " ;" if i < len(predicates) - 1 else " ."
                lines.append(f"    {pred} {obj}{separator}")
            lines.append("")

        return "\n".join(lines)

    def to_ntriples(self) -> str:
        """N-Triples 포맷으로 내보내기"""
        lines = []
        for triple in self.get_all_triples():
            # URI 확장
            s = self._expand_uri(triple.subject)
            p = self._expand_uri(triple.predicate)
            o = self._expand_uri(triple.object)
            lines.append(f"{s} {p} {o} .")
        return "\n".join(lines)

    def _expand_uri(self, compact: str) -> str:
        """축약 URI를 전체 URI로 확장"""
        if compact.startswith('"'):
            return compact  # 리터럴은 그대로

        if ":" in compact and not compact.startswith("http"):
            prefix, local = compact.split(":", 1)
            if prefix in self.prefixes:
                return f"<{self.prefixes[prefix]}{local}>"

        return f"<{compact}>" if not compact.startswith("<") else compact

    def to_json_ld(self) -> Dict[str, Any]:
        """JSON-LD 포맷으로 내보내기"""
        context = {f"@{k}": v for k, v in self.prefixes.items()}
        context["@vocab"] = Namespace.ONT.value

        graph = []

        # 주어별로 그룹화
        subjects: Dict[str, Dict[str, Any]] = {}
        for triple in self.get_all_triples():
            if triple.subject not in subjects:
                subjects[triple.subject] = {"@id": triple.subject}

            pred = triple.predicate
            obj = triple.object

            # 기존 값이 있으면 배열로
            if pred in subjects[triple.subject]:
                existing = subjects[triple.subject][pred]
                if isinstance(existing, list):
                    existing.append(obj)
                else:
                    subjects[triple.subject][pred] = [existing, obj]
            else:
                subjects[triple.subject][pred] = obj

        graph = list(subjects.values())

        return {
            "@context": context,
            "@graph": graph
        }

    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 내보내기 (JSON 직렬화용)"""
        return {
            "metadata": {
                "generated_at": datetime.now().isoformat(),
                "base_uri": self.base_uri,
                "stats": {
                    "total_triples": self.stats.total_triples,
                    "asserted": self.stats.asserted_triples,
                    "inferred": self.stats.inferred_triples,
                    "governance": self.stats.governance_triples,
                    "classes": self.stats.classes,
                    "properties": self.stats.properties,
                    "relationships": self.stats.relationships,
                }
            },
            "prefixes": self.prefixes,
            "triples": [
                {
                    "subject": t.subject,
                    "predicate": t.predicate,
                    "object": t.object,
                    "confidence": t.confidence,
                    "source": t.source,
                    "rule_applied": t.rule_applied,
                }
                for t in self.get_all_triples()
            ]
        }


# =============================================================================
# 편의 함수
# =============================================================================

def build_knowledge_graph(context: Any) -> KnowledgeGraphBuilder:
    """
    SharedContext에서 지식 그래프 빌드

    Args:
        context: SharedContext 객체

    Returns:
        KnowledgeGraphBuilder 인스턴스
    """
    builder = KnowledgeGraphBuilder()
    return builder.build_from_context(context)


def export_to_turtle(context: Any, filepath: str) -> str:
    """
    SharedContext를 Turtle 파일로 내보내기

    Args:
        context: SharedContext 객체
        filepath: 출력 파일 경로

    Returns:
        생성된 Turtle 문자열
    """
    kg = build_knowledge_graph(context)
    turtle = kg.to_turtle()

    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(turtle)

    return turtle


# =============================================================================
# Phase 2: OWL Reasoner 통합
# =============================================================================

@dataclass
class InferenceResult:
    """추론 결과"""
    inferred_triples: List[Triple]
    applied_rules: List[str]
    inference_time_ms: float
    total_iterations: int
    stats: Dict[str, int]


class SimpleOWLReasoner:
    """
    간단한 OWL Reasoner (Phase 2 스펙 기반)

    RDFS/OWL 의미론적 추론을 수행하는 경량 추론기입니다.

    지원 기능:
    - Forward Chaining 추론
    - owl:sameAs, owl:equivalentClass, rdfs:subClassOf 전이성
    - owl:inverseOf 역관계 추론
    - rdfs:domain, rdfs:range 타입 추론
    - 일관성 검사
    - 추론 설명

    Example:
        >>> reasoner = SimpleOWLReasoner()
        >>> reasoner.add_triple("ont:A", "owl:sameAs", "ont:B")
        >>> reasoner.add_triple("ont:B", "owl:sameAs", "ont:C")
        >>> result = reasoner.run_inference()
        >>> # result.inferred_triples에 (ont:A, owl:sameAs, ont:C) 포함
    """

    def __init__(self):
        self.kb = KnowledgeBase()
        self._rules: List[Rule] = []
        self._custom_rules: List[Rule] = []
        self._inference_history: List[InferenceResult] = []

        # 기본 추론 규칙 로드
        self._load_default_rules()

    def _load_default_rules(self):
        """기본 RDFS/OWL 추론 규칙 로드"""
        # Phase 2 스펙 기반 규칙 사용
        kg_builder = KnowledgeGraphBuilder()
        self._rules = kg_builder._get_inference_rules()

    # =========================================================================
    # Triple 관리
    # =========================================================================

    def add_triple(
        self,
        subject: str,
        predicate: str,
        obj: str,
        confidence: float = 1.0
    ) -> Triple:
        """
        트리플 추가

        Args:
            subject: 주어
            predicate: 술어
            obj: 목적어
            confidence: 신뢰도

        Returns:
            생성된 Triple 객체
        """
        triple = Triple(
            subject=subject,
            predicate=predicate,
            object=obj,
            confidence=confidence,
            source="asserted"
        )
        self.kb.add_triple(triple)
        return triple

    def add_triples(self, triples: List[Dict[str, Any]]) -> List[Triple]:
        """
        여러 트리플 추가

        Args:
            triples: 트리플 딕셔너리 목록

        Returns:
            생성된 Triple 객체 목록
        """
        result = []
        for t in triples:
            triple = self.add_triple(
                t.get("subject", ""),
                t.get("predicate", ""),
                t.get("object", ""),
                t.get("confidence", 1.0)
            )
            result.append(triple)
        return result

    def query(
        self,
        subject: str = None,
        predicate: str = None,
        obj: str = None
    ) -> List[Triple]:
        """
        패턴 매칭 쿼리

        Args:
            subject: 주어 패턴 (None이면 와일드카드)
            predicate: 술어 패턴
            obj: 목적어 패턴

        Returns:
            매칭되는 트리플 목록
        """
        return self.kb.query(subject, predicate, obj)

    # =========================================================================
    # 추론
    # =========================================================================

    def add_custom_rule(self, rule: Rule):
        """
        사용자 정의 규칙 추가

        Args:
            rule: Rule 객체
        """
        self._custom_rules.append(rule)

    def run_inference(self, max_iterations: int = 50) -> InferenceResult:
        """
        Forward Chaining 추론 실행

        Args:
            max_iterations: 최대 반복 횟수

        Returns:
            InferenceResult 객체
        """
        import time
        start_time = time.time()

        # 모든 규칙 병합
        all_rules = self._rules + self._custom_rules

        # Forward Chainer 생성 및 실행
        chainer = ForwardChainer(self.kb, all_rules)
        inferred_triples = chainer.infer(max_iterations=max_iterations)

        # 적용된 규칙 수집
        applied_rules = list(set(t.rule_applied for t in inferred_triples if t.rule_applied))

        # 규칙별 통계
        rule_stats = {}
        for t in inferred_triples:
            if t.rule_applied:
                rule_stats[t.rule_applied] = rule_stats.get(t.rule_applied, 0) + 1

        elapsed_ms = (time.time() - start_time) * 1000

        result = InferenceResult(
            inferred_triples=inferred_triples,
            applied_rules=applied_rules,
            inference_time_ms=elapsed_ms,
            total_iterations=len(chainer.inference_chains),
            stats=rule_stats
        )

        self._inference_history.append(result)

        logger.info(f"[OWL Reasoner] Inference completed: {len(inferred_triples)} triples, "
                   f"{len(applied_rules)} rules, {elapsed_ms:.2f}ms")

        return result

    def run_full_reasoning(self) -> Dict[str, Any]:
        """
        전체 추론 (RDFS + OWL + 사용자 정의)

        Returns:
            추론 결과 딕셔너리
        """
        # RDFS 추론
        rdfs_reasoner = RDFSReasoner(self.kb)
        rdfs_inferred = rdfs_reasoner.infer()

        # OWL 추론
        owl_reasoner = OWLReasoner(self.kb)
        owl_inferred = owl_reasoner.infer()

        # 사용자 정의 규칙
        custom_inferred = []
        if self._custom_rules:
            custom_chainer = ForwardChainer(self.kb, self._custom_rules)
            custom_inferred = custom_chainer.infer()

        return {
            "rdfs_inferred": [(t.subject, t.predicate, t.object) for t in rdfs_inferred],
            "owl_inferred": [(t.subject, t.predicate, t.object) for t in owl_inferred],
            "custom_inferred": [(t.subject, t.predicate, t.object) for t in custom_inferred],
            "total_inferred": len(rdfs_inferred) + len(owl_inferred) + len(custom_inferred),
            "total_triples": len(self.kb.triples)
        }

    # =========================================================================
    # 일관성/완전성 검사
    # =========================================================================

    def check_consistency(self) -> Dict[str, Any]:
        """
        일관성 검사

        Returns:
            일관성 검사 결과
        """
        checker = ConsistencyChecker(self.kb)
        report = checker.check()

        return {
            "is_consistent": report.is_consistent,
            "violations": report.violations,
            "warnings": report.warnings,
            "violation_count": len(report.violations),
            "warning_count": len(report.warnings)
        }

    def check_completeness(
        self,
        required_properties: Dict[str, List[str]] = None
    ) -> Dict[str, Any]:
        """
        완전성 검사

        Args:
            required_properties: 클래스별 필수 속성 딕셔너리

        Returns:
            완전성 검사 결과
        """
        checker = CompletenessChecker(self.kb)
        report = checker.check(required_properties)

        return {
            "completeness_score": report.completeness_score,
            "missing_required": report.missing_required,
            "recommendations": report.recommendations,
            "missing_count": len(report.missing_required)
        }

    # =========================================================================
    # 추론 설명
    # =========================================================================

    def explain_inference(
        self,
        subject: str,
        predicate: str,
        obj: str
    ) -> List[Dict[str, Any]]:
        """
        특정 트리플의 추론 설명

        Args:
            subject: 주어
            predicate: 술어
            obj: 목적어

        Returns:
            추론 설명 목록
        """
        explanations = []

        key = (subject, predicate, obj)
        if key not in self.kb.triple_details:
            return [{"type": "not_found", "message": "해당 트리플이 존재하지 않습니다"}]

        triple = self.kb.triple_details[key]

        if triple.source == "asserted":
            explanations.append({
                "type": "fact",
                "message": "명시적으로 선언된 사실입니다",
                "confidence": triple.confidence
            })
        else:
            explanations.append({
                "type": "inferred",
                "rule": triple.rule_applied,
                "confidence": triple.confidence,
                "message": f"규칙 '{triple.rule_applied}'에 의해 추론되었습니다"
            })

        return explanations

    def get_inference_chain(
        self,
        subject: str,
        predicate: str,
        obj: str
    ) -> List[Dict[str, Any]]:
        """
        추론 체인 추적

        Args:
            subject: 주어
            predicate: 술어
            obj: 목적어

        Returns:
            추론 체인 (전제들의 목록)
        """
        chain = []
        target = (subject, predicate, obj)

        # 추론 이력에서 검색
        for result in self._inference_history:
            for triple in result.inferred_triples:
                if (triple.subject, triple.predicate, triple.object) == target:
                    chain.append({
                        "triple": target,
                        "rule": triple.rule_applied,
                        "confidence": triple.confidence,
                        "source": triple.source
                    })

        return chain

    # =========================================================================
    # 유틸리티
    # =========================================================================

    def get_all_triples(self) -> List[Triple]:
        """모든 트리플 반환"""
        return list(self.kb.triple_details.values())

    def get_inferred_triples(self) -> List[Triple]:
        """추론된 트리플만 반환"""
        return [t for t in self.kb.triple_details.values() if t.source == "inferred"]

    def get_asserted_triples(self) -> List[Triple]:
        """명시된 트리플만 반환"""
        return [t for t in self.kb.triple_details.values() if t.source == "asserted"]

    def get_statistics(self) -> Dict[str, Any]:
        """추론 통계 반환"""
        all_triples = self.get_all_triples()
        inferred = self.get_inferred_triples()
        asserted = self.get_asserted_triples()

        # 규칙별 통계
        rule_counts = {}
        for t in inferred:
            if t.rule_applied:
                rule_counts[t.rule_applied] = rule_counts.get(t.rule_applied, 0) + 1

        return {
            "total_triples": len(all_triples),
            "asserted_triples": len(asserted),
            "inferred_triples": len(inferred),
            "total_rules": len(self._rules) + len(self._custom_rules),
            "custom_rules": len(self._custom_rules),
            "rule_application_counts": rule_counts,
            "inference_runs": len(self._inference_history)
        }

    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 내보내기"""
        return {
            "triples": [
                {
                    "subject": t.subject,
                    "predicate": t.predicate,
                    "object": t.object,
                    "confidence": t.confidence,
                    "source": t.source,
                    "rule_applied": t.rule_applied
                }
                for t in self.get_all_triples()
            ],
            "statistics": self.get_statistics()
        }

    def clear(self):
        """모든 트리플 및 이력 초기화"""
        self.kb = KnowledgeBase()
        self._custom_rules = []
        self._inference_history = []


# =============================================================================
# Phase 2: 통합 추론 API
# =============================================================================

def create_owl_reasoner() -> SimpleOWLReasoner:
    """
    OWL Reasoner 인스턴스 생성

    Returns:
        SimpleOWLReasoner 인스턴스
    """
    return SimpleOWLReasoner()


def run_schema_inference(
    tables: Dict[str, Any],
    foreign_keys: List[Dict[str, Any]] = None,
    max_iterations: int = 50
) -> Dict[str, Any]:
    """
    스키마 기반 추론 실행

    Args:
        tables: 테이블 정보
        foreign_keys: FK 관계
        max_iterations: 최대 추론 반복

    Returns:
        추론 결과
    """
    # Knowledge Graph 빌드
    kg_builder = KnowledgeGraphBuilder()
    kg_builder.generate_rdf_triples_from_schema(tables, foreign_keys)

    # 추론 실행
    inferred = kg_builder.run_inference(max_iterations)

    return {
        "triples": kg_builder.to_dict()["triples"],
        "inferred_count": len(inferred),
        "statistics": kg_builder.get_stats().__dict__
    }
