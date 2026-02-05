"""
Palantir AIP Style Visualization Data Generator (v1.0)

프론트엔드 시각화를 위한 데이터 구조 생성:
1. Data Map - 테이블/컬럼 관계 시각화
2. Ontology Map - 클래스/프로퍼티 계층 시각화
3. Entity Relationship Map - 엔티티 간 관계 시각화
4. Lineage Map - 데이터 흐름 시각화
5. Insight Dashboard - 인사이트 대시보드 데이터

출력 포맷: JSON (D3.js, ECharts, Cytoscape.js 호환)
"""

import json
import hashlib
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Any, Optional, Tuple, Set
from enum import Enum
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# Enums for Visualization
# =============================================================================

class NodeType(str, Enum):
    """시각화 노드 타입"""
    # Data Map
    DATABASE = "database"
    TABLE = "table"
    COLUMN = "column"

    # Ontology Map
    OWL_CLASS = "owl_class"
    OWL_OBJECT_PROPERTY = "owl_object_property"
    OWL_DATA_PROPERTY = "owl_data_property"
    OWL_INDIVIDUAL = "owl_individual"

    # Entity Map
    ENTITY = "entity"
    ATTRIBUTE = "attribute"

    # Insight
    INSIGHT = "insight"
    KPI = "kpi"
    ALERT = "alert"


class EdgeType(str, Enum):
    """시각화 엣지 타입"""
    # Data Map
    CONTAINS = "contains"           # Database -> Table, Table -> Column
    FOREIGN_KEY = "foreign_key"     # Column -> Column
    REFERENCES = "references"       # Table -> Table

    # Ontology Map
    SUBCLASS_OF = "subclass_of"     # Class -> Class
    DOMAIN = "domain"               # Property -> Class
    RANGE = "range"                 # Property -> Class
    EQUIVALENT = "equivalent"       # Class <-> Class

    # Entity Map
    HAS_ATTRIBUTE = "has_attribute"
    RELATES_TO = "relates_to"
    SAME_AS = "same_as"

    # Lineage
    DERIVED_FROM = "derived_from"
    TRANSFORMS_TO = "transforms_to"

    # Insight
    IMPACTS = "impacts"
    CORRELATES = "correlates"
    CAUSES = "causes"


class SeverityLevel(str, Enum):
    """심각도 레벨"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


# =============================================================================
# Data Classes for Visualization
# =============================================================================

@dataclass
class VisNode:
    """시각화 노드"""
    id: str
    label: str
    type: str
    group: str = ""

    # 위치 (선택적, 레이아웃 엔진이 계산할 수도 있음)
    x: Optional[float] = None
    y: Optional[float] = None

    # 스타일
    size: float = 30.0
    color: Optional[str] = None
    icon: Optional[str] = None

    # 메타데이터
    metadata: Dict[str, Any] = field(default_factory=dict)

    # 상태
    highlighted: bool = False
    selected: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "label": self.label,
            "type": self.type,
            "group": self.group,
            "x": self.x,
            "y": self.y,
            "size": self.size,
            "color": self.color,
            "icon": self.icon,
            "metadata": self.metadata,
            "highlighted": self.highlighted,
            "selected": self.selected,
        }


@dataclass
class VisEdge:
    """시각화 엣지"""
    id: str
    source: str
    target: str
    type: str
    label: str = ""

    # 스타일
    weight: float = 1.0
    color: Optional[str] = None
    style: str = "solid"  # solid, dashed, dotted
    animated: bool = False

    # 메타데이터
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "source": self.source,
            "target": self.target,
            "type": self.type,
            "label": self.label,
            "weight": self.weight,
            "color": self.color,
            "style": self.style,
            "animated": self.animated,
            "metadata": self.metadata,
        }


@dataclass
class VisGraph:
    """시각화 그래프"""
    id: str
    name: str
    description: str
    graph_type: str  # data_map, ontology_map, entity_map, lineage_map

    nodes: List[VisNode] = field(default_factory=list)
    edges: List[VisEdge] = field(default_factory=list)

    # 그래프 메타데이터
    metadata: Dict[str, Any] = field(default_factory=dict)

    # 레이아웃 설정
    layout: str = "force"  # force, hierarchical, circular, grid

    # 통계
    node_count: int = 0
    edge_count: int = 0

    # 타임스탬프
    created_at: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "graph_type": self.graph_type,
            "nodes": [n.to_dict() for n in self.nodes],
            "edges": [e.to_dict() for e in self.edges],
            "metadata": self.metadata,
            "layout": self.layout,
            "node_count": len(self.nodes),
            "edge_count": len(self.edges),
            "created_at": self.created_at or datetime.now().isoformat(),
        }


@dataclass
class InsightCard:
    """인사이트 카드 (대시보드용)"""
    id: str
    title: str
    description: str
    insight_type: str  # prediction, anomaly, correlation, trend, alert

    # 수치
    value: Optional[float] = None
    unit: str = ""
    change: Optional[float] = None  # 변화율 (%)
    change_direction: str = ""  # up, down, stable

    # 심각도
    severity: str = "info"

    # 관련 엔티티
    related_entities: List[str] = field(default_factory=list)

    # 액션
    recommended_actions: List[str] = field(default_factory=list)

    # 신뢰도
    confidence: float = 0.0

    # 메타데이터
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Dashboard:
    """대시보드 데이터"""
    id: str
    name: str

    # KPI 요약
    kpi_summary: Dict[str, Any] = field(default_factory=dict)

    # 인사이트 카드들
    insight_cards: List[InsightCard] = field(default_factory=list)

    # 그래프들
    graphs: List[VisGraph] = field(default_factory=list)

    # 통계 요약
    statistics: Dict[str, Any] = field(default_factory=dict)

    # 타임스탬프
    generated_at: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "kpi_summary": self.kpi_summary,
            "insight_cards": [c.to_dict() for c in self.insight_cards],
            "graphs": [g.to_dict() for g in self.graphs],
            "statistics": self.statistics,
            "generated_at": self.generated_at or datetime.now().isoformat(),
        }


# =============================================================================
# Color Schemes (Palantir-inspired)
# =============================================================================

class ColorScheme:
    """팔란티어 스타일 색상 스킴"""

    # Node Colors by Type
    NODE_COLORS = {
        NodeType.DATABASE: "#1a1a2e",
        NodeType.TABLE: "#16213e",
        NodeType.COLUMN: "#0f3460",
        NodeType.OWL_CLASS: "#533483",
        NodeType.OWL_OBJECT_PROPERTY: "#e94560",
        NodeType.OWL_DATA_PROPERTY: "#ff6b6b",
        NodeType.OWL_INDIVIDUAL: "#feca57",
        NodeType.ENTITY: "#48dbfb",
        NodeType.ATTRIBUTE: "#1dd1a1",
        NodeType.INSIGHT: "#ff9f43",
        NodeType.KPI: "#5f27cd",
        NodeType.ALERT: "#ee5253",
    }

    # Edge Colors by Type
    EDGE_COLORS = {
        EdgeType.CONTAINS: "#576574",
        EdgeType.FOREIGN_KEY: "#e94560",
        EdgeType.REFERENCES: "#ff9f43",
        EdgeType.SUBCLASS_OF: "#5f27cd",
        EdgeType.DOMAIN: "#48dbfb",
        EdgeType.RANGE: "#1dd1a1",
        EdgeType.EQUIVALENT: "#feca57",
        EdgeType.RELATES_TO: "#00d2d3",
        EdgeType.IMPACTS: "#ff6b6b",
        EdgeType.CORRELATES: "#54a0ff",
        EdgeType.CAUSES: "#ee5253",
    }

    # Severity Colors
    SEVERITY_COLORS = {
        SeverityLevel.CRITICAL: "#ee5253",
        SeverityLevel.HIGH: "#ff6b6b",
        SeverityLevel.MEDIUM: "#feca57",
        SeverityLevel.LOW: "#1dd1a1",
        SeverityLevel.INFO: "#54a0ff",
    }

    @classmethod
    def get_node_color(cls, node_type: str) -> str:
        try:
            return cls.NODE_COLORS.get(NodeType(node_type), "#576574")
        except ValueError:
            return "#576574"

    @classmethod
    def get_edge_color(cls, edge_type: str) -> str:
        try:
            return cls.EDGE_COLORS.get(EdgeType(edge_type), "#576574")
        except ValueError:
            return "#576574"


# =============================================================================
# Data Map Generator
# =============================================================================

class DataMapGenerator:
    """
    데이터 맵 생성기

    테이블, 컬럼, FK 관계를 시각화 가능한 그래프로 변환
    """

    def __init__(self):
        self.color_scheme = ColorScheme()

    def generate(
        self,
        tables_data: Dict[str, Any],
        fk_results: Dict[str, Any],
        domain: str = "unknown",
    ) -> VisGraph:
        """
        데이터 맵 생성

        Args:
            tables_data: 테이블 정보 딕셔너리
            fk_results: FK 탐지 결과
            domain: 도메인명

        Returns:
            VisGraph: 시각화 가능한 그래프
        """
        nodes = []
        edges = []

        # 1. 데이터베이스 루트 노드
        db_node = VisNode(
            id="db_root",
            label=f"{domain.upper()} Database",
            type=NodeType.DATABASE.value,
            group="database",
            size=50.0,
            color=ColorScheme.get_node_color(NodeType.DATABASE.value),
            icon="database",
            metadata={"table_count": len(tables_data)}
        )
        nodes.append(db_node)

        # 2. 테이블 노드
        table_positions = {}
        for i, (table_name, table_info) in enumerate(tables_data.items()):
            table_id = f"table_{table_name}"

            # 테이블 노드
            table_node = VisNode(
                id=table_id,
                label=table_name,
                type=NodeType.TABLE.value,
                group="table",
                size=40.0,
                color=ColorScheme.get_node_color(NodeType.TABLE.value),
                icon="table",
                metadata={
                    "row_count": table_info.get("row_count", 0),
                    "column_count": table_info.get("column_count", 0),
                    "columns": table_info.get("columns", []),
                }
            )
            nodes.append(table_node)
            table_positions[table_name] = table_id

            # DB -> Table 엣지
            edges.append(VisEdge(
                id=f"edge_db_{table_name}",
                source="db_root",
                target=table_id,
                type=EdgeType.CONTAINS.value,
                color=ColorScheme.get_edge_color(EdgeType.CONTAINS.value),
            ))

            # 3. 컬럼 노드 (주요 컬럼만)
            columns = table_info.get("columns", [])
            for j, col_name in enumerate(columns[:10]):  # 최대 10개
                col_id = f"col_{table_name}_{col_name}"

                # PK/FK 여부 확인
                is_pk = col_name.lower().endswith("_id") and col_name.lower().startswith(table_name.lower()[:3])
                is_fk = col_name.lower().endswith("_id") and not is_pk

                col_node = VisNode(
                    id=col_id,
                    label=col_name,
                    type=NodeType.COLUMN.value,
                    group=table_name,
                    size=20.0,
                    color="#e94560" if is_fk else ("#feca57" if is_pk else ColorScheme.get_node_color(NodeType.COLUMN.value)),
                    icon="key" if is_pk or is_fk else "column",
                    metadata={
                        "is_pk": is_pk,
                        "is_fk": is_fk,
                        "table": table_name,
                    }
                )
                nodes.append(col_node)

                # Table -> Column 엣지
                edges.append(VisEdge(
                    id=f"edge_{table_name}_{col_name}",
                    source=table_id,
                    target=col_id,
                    type=EdgeType.CONTAINS.value,
                    color=ColorScheme.get_edge_color(EdgeType.CONTAINS.value),
                    style="dashed",
                ))

        # 4. FK 관계 엣지
        mandatory_fks = fk_results.get("mandatory_fks", [])
        for fk in mandatory_fks:
            source = fk.get("source", "")
            target = fk.get("target", "")

            if "." in source and "." in target:
                src_table, src_col = source.split(".")
                tgt_table, tgt_col = target.split(".")

                src_id = f"col_{src_table}_{src_col}"
                tgt_id = f"col_{tgt_table}_{tgt_col}"

                # FK 엣지
                fk_edge = VisEdge(
                    id=f"fk_{source}_{target}",
                    source=src_id,
                    target=tgt_id,
                    type=EdgeType.FOREIGN_KEY.value,
                    label=fk.get("fk_type", "FK"),
                    weight=fk.get("confidence", 1.0),
                    color=ColorScheme.get_edge_color(EdgeType.FOREIGN_KEY.value),
                    animated=True,
                    metadata={
                        "priority": fk.get("priority", "MEDIUM"),
                        "fk_type": fk.get("fk_type", "standard"),
                        "confidence": fk.get("confidence", 1.0),
                    }
                )
                edges.append(fk_edge)

                # 테이블 간 참조 엣지
                src_table_id = f"table_{src_table}"
                tgt_table_id = f"table_{tgt_table}"

                ref_edge = VisEdge(
                    id=f"ref_{src_table}_{tgt_table}",
                    source=src_table_id,
                    target=tgt_table_id,
                    type=EdgeType.REFERENCES.value,
                    label="references",
                    color=ColorScheme.get_edge_color(EdgeType.REFERENCES.value),
                    style="dashed",
                )
                edges.append(ref_edge)

        # 그래프 생성
        graph = VisGraph(
            id=f"data_map_{domain}_{datetime.now().strftime('%Y%m%d%H%M%S')}",
            name=f"{domain.title()} Data Map",
            description=f"Data schema visualization for {domain} domain",
            graph_type="data_map",
            nodes=nodes,
            edges=edges,
            layout="hierarchical",
            metadata={
                "domain": domain,
                "total_tables": len(tables_data),
                "total_fks": len(mandatory_fks),
            }
        )

        return graph


# =============================================================================
# Ontology Map Generator
# =============================================================================

class OntologyMapGenerator:
    """
    온톨로지 맵 생성기

    OWL 클래스, 프로퍼티, 제약조건을 시각화
    """

    def __init__(self):
        self.color_scheme = ColorScheme()

    def generate(
        self,
        ontology_data: Dict[str, Any],
        domain: str = "unknown",
    ) -> VisGraph:
        """
        온톨로지 맵 생성

        Args:
            ontology_data: 온톨로지 정보
            domain: 도메인명

        Returns:
            VisGraph: 시각화 가능한 그래프
        """
        nodes = []
        edges = []

        # 1. owl:Thing 루트 노드
        root_node = VisNode(
            id="owl_thing",
            label="owl:Thing",
            type=NodeType.OWL_CLASS.value,
            group="root",
            size=50.0,
            color=ColorScheme.get_node_color(NodeType.OWL_CLASS.value),
            icon="class",
        )
        nodes.append(root_node)

        # 2. 클래스 노드들
        classes = ontology_data.get("classes", [])
        class_ids = {}

        for i, cls in enumerate(classes):
            cls_name = cls.get("name", f"Class_{i}")
            cls_id = f"class_{cls_name.replace(' ', '_')}"
            class_ids[cls_name] = cls_id

            cls_node = VisNode(
                id=cls_id,
                label=cls_name,
                type=NodeType.OWL_CLASS.value,
                group="class",
                size=35.0,
                color=ColorScheme.get_node_color(NodeType.OWL_CLASS.value),
                icon="class",
                metadata={
                    "description": cls.get("description", ""),
                    "source_table": cls.get("source_table", ""),
                    "instance_count": cls.get("instance_count", 0),
                }
            )
            nodes.append(cls_node)

            # SubClassOf 관계
            parent = cls.get("parent", "owl:Thing")
            parent_id = class_ids.get(parent, "owl_thing")

            edges.append(VisEdge(
                id=f"subclass_{cls_id}",
                source=cls_id,
                target=parent_id,
                type=EdgeType.SUBCLASS_OF.value,
                label="subClassOf",
                color=ColorScheme.get_edge_color(EdgeType.SUBCLASS_OF.value),
            ))

        # 3. Object Property 노드들
        object_properties = ontology_data.get("object_properties", [])

        for i, prop in enumerate(object_properties):
            prop_name = prop.get("name", f"ObjectProperty_{i}")
            prop_id = f"op_{prop_name.replace(' ', '_')}"

            prop_node = VisNode(
                id=prop_id,
                label=prop_name,
                type=NodeType.OWL_OBJECT_PROPERTY.value,
                group="object_property",
                size=25.0,
                color=ColorScheme.get_node_color(NodeType.OWL_OBJECT_PROPERTY.value),
                icon="link",
                metadata={
                    "domain": prop.get("domain", ""),
                    "range": prop.get("range", ""),
                    "is_functional": prop.get("is_functional", False),
                    "is_inverse_functional": prop.get("is_inverse_functional", False),
                }
            )
            nodes.append(prop_node)

            # Domain 관계
            domain_cls = prop.get("domain", "")
            if domain_cls and domain_cls in class_ids:
                edges.append(VisEdge(
                    id=f"domain_{prop_id}",
                    source=prop_id,
                    target=class_ids[domain_cls],
                    type=EdgeType.DOMAIN.value,
                    label="domain",
                    color=ColorScheme.get_edge_color(EdgeType.DOMAIN.value),
                    style="dashed",
                ))

            # Range 관계
            range_cls = prop.get("range", "")
            if range_cls and range_cls in class_ids:
                edges.append(VisEdge(
                    id=f"range_{prop_id}",
                    source=prop_id,
                    target=class_ids[range_cls],
                    type=EdgeType.RANGE.value,
                    label="range",
                    color=ColorScheme.get_edge_color(EdgeType.RANGE.value),
                    style="dashed",
                ))

        # 4. Data Property 노드들
        data_properties = ontology_data.get("data_properties", [])

        for i, prop in enumerate(data_properties):
            prop_name = prop.get("name", f"DataProperty_{i}")
            prop_id = f"dp_{prop_name.replace(' ', '_')}"

            prop_node = VisNode(
                id=prop_id,
                label=prop_name,
                type=NodeType.OWL_DATA_PROPERTY.value,
                group="data_property",
                size=20.0,
                color=ColorScheme.get_node_color(NodeType.OWL_DATA_PROPERTY.value),
                icon="attribute",
                metadata={
                    "domain": prop.get("domain", ""),
                    "range": prop.get("range", "xsd:string"),
                }
            )
            nodes.append(prop_node)

            # Domain 관계
            domain_cls = prop.get("domain", "")
            if domain_cls and domain_cls in class_ids:
                edges.append(VisEdge(
                    id=f"domain_{prop_id}",
                    source=prop_id,
                    target=class_ids[domain_cls],
                    type=EdgeType.DOMAIN.value,
                    label="domain",
                    color=ColorScheme.get_edge_color(EdgeType.DOMAIN.value),
                    style="dotted",
                ))

        # 그래프 생성
        graph = VisGraph(
            id=f"ontology_map_{domain}_{datetime.now().strftime('%Y%m%d%H%M%S')}",
            name=f"{domain.title()} Ontology Map",
            description=f"OWL 2 ontology visualization for {domain} domain",
            graph_type="ontology_map",
            nodes=nodes,
            edges=edges,
            layout="hierarchical",
            metadata={
                "domain": domain,
                "total_classes": len(classes),
                "total_object_properties": len(object_properties),
                "total_data_properties": len(data_properties),
            }
        )

        return graph


# =============================================================================
# Entity Relationship Map Generator
# =============================================================================

class EntityRelationshipMapGenerator:
    """
    엔티티 관계 맵 생성기

    비즈니스 엔티티 간의 관계를 시각화
    """

    def __init__(self):
        self.color_scheme = ColorScheme()

    def generate(
        self,
        entities: List[Dict[str, Any]],
        relationships: List[Dict[str, Any]],
        domain: str = "unknown",
    ) -> VisGraph:
        """
        엔티티 관계 맵 생성
        """
        nodes = []
        edges = []
        entity_ids = {}

        # 1. 엔티티 노드들
        for i, entity in enumerate(entities):
            entity_name = entity.get("name", f"Entity_{i}")
            entity_id = f"entity_{entity_name.replace(' ', '_')}"
            entity_ids[entity_name] = entity_id

            # 엔티티 중요도에 따른 크기
            importance = entity.get("importance", 0.5)
            size = 25.0 + (importance * 30.0)

            entity_node = VisNode(
                id=entity_id,
                label=entity_name,
                type=NodeType.ENTITY.value,
                group=entity.get("category", "default"),
                size=size,
                color=ColorScheme.get_node_color(NodeType.ENTITY.value),
                icon="entity",
                metadata={
                    "description": entity.get("description", ""),
                    "source_table": entity.get("source_table", ""),
                    "attributes": entity.get("attributes", []),
                    "record_count": entity.get("record_count", 0),
                    "importance": importance,
                }
            )
            nodes.append(entity_node)

            # 2. 속성 노드들 (주요 속성만)
            attributes = entity.get("attributes", [])[:5]
            for attr in attributes:
                attr_name = attr.get("name", "") if isinstance(attr, dict) else attr
                attr_id = f"attr_{entity_name}_{attr_name}".replace(" ", "_")

                attr_node = VisNode(
                    id=attr_id,
                    label=attr_name,
                    type=NodeType.ATTRIBUTE.value,
                    group=entity_name,
                    size=15.0,
                    color=ColorScheme.get_node_color(NodeType.ATTRIBUTE.value),
                    icon="attribute",
                )
                nodes.append(attr_node)

                edges.append(VisEdge(
                    id=f"has_attr_{entity_id}_{attr_id}",
                    source=entity_id,
                    target=attr_id,
                    type=EdgeType.HAS_ATTRIBUTE.value,
                    color=ColorScheme.get_edge_color(EdgeType.HAS_ATTRIBUTE.value),
                    style="dotted",
                ))

        # 3. 관계 엣지들
        for i, rel in enumerate(relationships):
            source_entity = rel.get("source", "")
            target_entity = rel.get("target", "")
            rel_type = rel.get("type", "relates_to")
            rel_label = rel.get("label", rel_type)

            if source_entity in entity_ids and target_entity in entity_ids:
                edge = VisEdge(
                    id=f"rel_{i}_{source_entity}_{target_entity}",
                    source=entity_ids[source_entity],
                    target=entity_ids[target_entity],
                    type=EdgeType.RELATES_TO.value,
                    label=rel_label,
                    weight=rel.get("strength", 1.0),
                    color=ColorScheme.get_edge_color(EdgeType.RELATES_TO.value),
                    animated=rel.get("is_key_relationship", False),
                    metadata={
                        "cardinality": rel.get("cardinality", ""),
                        "is_mandatory": rel.get("is_mandatory", False),
                    }
                )
                edges.append(edge)

        # 그래프 생성
        graph = VisGraph(
            id=f"entity_map_{domain}_{datetime.now().strftime('%Y%m%d%H%M%S')}",
            name=f"{domain.title()} Entity Relationship Map",
            description=f"Business entity relationships for {domain} domain",
            graph_type="entity_map",
            nodes=nodes,
            edges=edges,
            layout="force",
            metadata={
                "domain": domain,
                "total_entities": len(entities),
                "total_relationships": len(relationships),
            }
        )

        return graph


# =============================================================================
# Insight Dashboard Generator
# =============================================================================

class InsightDashboardGenerator:
    """
    인사이트 대시보드 생성기

    팔란티어 스타일의 인사이트 카드 및 KPI 대시보드
    """

    def generate(
        self,
        insights: List[Dict[str, Any]],
        kpis: Dict[str, Any],
        correlations: List[Dict[str, Any]],
        domain: str = "unknown",
    ) -> Dashboard:
        """
        대시보드 생성
        """
        insight_cards = []

        # 1. KPI 요약
        kpi_summary = {
            "total_tables": kpis.get("total_tables", 0),
            "total_columns": kpis.get("total_columns", 0),
            "total_rows": kpis.get("total_rows", 0),
            "total_fks": kpis.get("total_fks", 0),
            "data_quality_score": kpis.get("data_quality_score", 0.0),
            "ontology_coverage": kpis.get("ontology_coverage", 0.0),
        }

        # 2. 예측 인사이트 카드
        for i, insight in enumerate(insights):
            card = InsightCard(
                id=f"insight_{i}",
                title=insight.get("title", f"Insight {i+1}"),
                description=insight.get("description", ""),
                insight_type=insight.get("type", "prediction"),
                value=insight.get("value"),
                unit=insight.get("unit", ""),
                change=insight.get("change"),
                change_direction=insight.get("change_direction", "stable"),
                severity=insight.get("severity", "info"),
                related_entities=insight.get("related_entities", []),
                recommended_actions=insight.get("recommended_actions", []),
                confidence=insight.get("confidence", 0.0),
                metadata=insight.get("metadata", {}),
            )
            insight_cards.append(card)

        # 3. 상관관계 인사이트 카드
        for i, corr in enumerate(correlations[:5]):  # 상위 5개
            source = corr.get("source", "")
            target = corr.get("target", "")
            correlation = corr.get("correlation", 0.0)

            card = InsightCard(
                id=f"corr_{i}",
                title=f"{source} → {target}",
                description=f"상관계수: r={correlation:.3f}",
                insight_type="correlation",
                value=correlation,
                unit="r",
                severity="high" if abs(correlation) > 0.7 else "medium",
                confidence=1 - corr.get("p_value", 0.05),
                metadata={
                    "source": source,
                    "target": target,
                    "p_value": corr.get("p_value", 0.0),
                }
            )
            insight_cards.append(card)

        # 4. 통계 요약
        statistics = {
            "insight_count": len(insights),
            "correlation_count": len(correlations),
            "high_severity_count": sum(1 for c in insight_cards if c.severity in ["critical", "high"]),
            "average_confidence": sum(c.confidence for c in insight_cards) / len(insight_cards) if insight_cards else 0,
        }

        # 대시보드 생성
        dashboard = Dashboard(
            id=f"dashboard_{domain}_{datetime.now().strftime('%Y%m%d%H%M%S')}",
            name=f"{domain.title()} Insight Dashboard",
            kpi_summary=kpi_summary,
            insight_cards=insight_cards,
            graphs=[],  # 별도로 추가
            statistics=statistics,
        )

        return dashboard


# =============================================================================
# Unified Visualization Generator
# =============================================================================

class PalantirStyleVisualizationGenerator:
    """
    팔란티어 AIP 스타일 통합 시각화 생성기

    모든 시각화 데이터를 통합 생성
    """

    def __init__(self):
        self.data_map_gen = DataMapGenerator()
        self.ontology_map_gen = OntologyMapGenerator()
        self.entity_map_gen = EntityRelationshipMapGenerator()
        self.dashboard_gen = InsightDashboardGenerator()

    def generate_all(
        self,
        tables_data: Dict[str, Any],
        fk_results: Dict[str, Any],
        ontology_data: Dict[str, Any],
        entities: List[Dict[str, Any]],
        relationships: List[Dict[str, Any]],
        insights: List[Dict[str, Any]],
        kpis: Dict[str, Any],
        correlations: List[Dict[str, Any]],
        domain: str = "unknown",
    ) -> Dict[str, Any]:
        """
        모든 시각화 데이터 통합 생성

        Returns:
            Dict containing all visualization data
        """
        result = {
            "generated_at": datetime.now().isoformat(),
            "domain": domain,
            "version": "1.0.0",
            "visualizations": {}
        }

        # 1. Data Map
        try:
            data_map = self.data_map_gen.generate(tables_data, fk_results, domain)
            result["visualizations"]["data_map"] = data_map.to_dict()
        except Exception as e:
            logger.error(f"Failed to generate data map: {e}")
            result["visualizations"]["data_map"] = {"error": str(e)}

        # 2. Ontology Map
        try:
            ontology_map = self.ontology_map_gen.generate(ontology_data, domain)
            result["visualizations"]["ontology_map"] = ontology_map.to_dict()
        except Exception as e:
            logger.error(f"Failed to generate ontology map: {e}")
            result["visualizations"]["ontology_map"] = {"error": str(e)}

        # 3. Entity Relationship Map
        try:
            entity_map = self.entity_map_gen.generate(entities, relationships, domain)
            result["visualizations"]["entity_map"] = entity_map.to_dict()
        except Exception as e:
            logger.error(f"Failed to generate entity map: {e}")
            result["visualizations"]["entity_map"] = {"error": str(e)}

        # 4. Dashboard
        try:
            dashboard = self.dashboard_gen.generate(insights, kpis, correlations, domain)
            result["visualizations"]["dashboard"] = dashboard.to_dict()
        except Exception as e:
            logger.error(f"Failed to generate dashboard: {e}")
            result["visualizations"]["dashboard"] = {"error": str(e)}

        # 5. 메타데이터
        result["metadata"] = {
            "total_nodes": sum(
                len(v.get("nodes", []))
                for k, v in result["visualizations"].items()
                if isinstance(v, dict) and "nodes" in v
            ),
            "total_edges": sum(
                len(v.get("edges", []))
                for k, v in result["visualizations"].items()
                if isinstance(v, dict) and "edges" in v
            ),
            "visualization_types": list(result["visualizations"].keys()),
        }

        return result

    def export_to_json(
        self,
        visualization_data: Dict[str, Any],
        output_path: str,
    ) -> str:
        """JSON 파일로 내보내기"""
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(visualization_data, f, ensure_ascii=False, indent=2)
        return output_path


# =============================================================================
# Factory Function
# =============================================================================

def create_visualization_generator() -> PalantirStyleVisualizationGenerator:
    """시각화 생성기 팩토리 함수"""
    return PalantirStyleVisualizationGenerator()
