"""
Knowledge Graph Engine

Enterprise-grade Knowledge Graph implementation with:
- Entity and Relation management
- Community Detection (GraphRAG style)
- Multi-hop path finding
- Subgraph extraction
- LLM-powered entity resolution

Based on:
- Microsoft GraphRAG architecture
- Palantir Foundry Ontology concepts
- arXiv:2410.05130 (Scalable Graph Reasoning)
"""

import json
import hashlib
from typing import Dict, List, Any, Optional, Set, Tuple, Union
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from collections import defaultdict
import logging

import networkx as nx

try:
    import community as community_louvain
    COMMUNITY_AVAILABLE = True
except ImportError:
    COMMUNITY_AVAILABLE = False

from ..common.utils.llm import (
    chat_completion_with_json,
    get_openai_client,
    extract_json_from_response,
    DEFAULT_MODEL,
)

logger = logging.getLogger(__name__)


class EntityType(Enum):
    """Entity types in the Knowledge Graph"""
    # Schema entities (for data silo integration)
    TABLE = "table"
    COLUMN = "column"
    DATABASE = "database"
    SCHEMA = "schema"

    # Core business entities
    SUPPLIER = "supplier"
    CUSTOMER = "customer"
    PRODUCT = "product"
    ORDER = "order"
    PURCHASE_ORDER = "purchase_order"
    SHIPMENT = "shipment"
    WAREHOUSE = "warehouse"
    INVENTORY = "inventory"
    INVOICE = "invoice"
    PAYMENT = "payment"

    # Dimensional entities
    LOCATION = "location"
    TIME_PERIOD = "time_period"
    CATEGORY = "category"
    STATUS = "status"

    # Risk/Quality entities
    RISK_INDICATOR = "risk_indicator"
    QUALITY_METRIC = "quality_metric"
    ANOMALY = "anomaly"

    # Generic
    ENTITY = "entity"
    CONCEPT = "concept"


class RelationType(Enum):
    """Relation types in the Knowledge Graph"""
    # Schema relations (for data silo integration)
    HAS_ATTRIBUTE = "has_attribute"
    HAS_COLUMN = "has_column"
    HOMEOMORPHIC_TO = "homeomorphic_to"
    FK_REFERENCE = "fk_reference"
    DEPENDS_ON = "depends_on"

    # Transactional
    SUPPLIES = "supplies"
    ORDERS_FROM = "orders_from"
    SHIPS_TO = "ships_to"
    CONTAINS = "contains"
    PAYS_FOR = "pays_for"

    # Hierarchical
    BELONGS_TO = "belongs_to"
    PARENT_OF = "parent_of"
    PART_OF = "part_of"

    # Associative
    RELATED_TO = "related_to"
    SIMILAR_TO = "similar_to"
    REFERENCES = "references"

    # Temporal
    PRECEDES = "precedes"
    FOLLOWS = "follows"
    CONCURRENT_WITH = "concurrent_with"

    # Risk/Quality
    HAS_RISK = "has_risk"
    AFFECTS = "affects"
    CAUSES = "causes"
    MITIGATES = "mitigates"

    # Data lineage
    DERIVED_FROM = "derived_from"
    MAPS_TO = "maps_to"
    EQUIVALENT_TO = "equivalent_to"


@dataclass
class KGEntity:
    """
    Knowledge Graph Entity

    Represents a real-world entity extracted from data.
    Includes both schema-level and instance-level information.
    """
    id: str
    entity_type: EntityType
    name: str
    properties: Dict[str, Any] = field(default_factory=dict)
    source: str = ""  # Generic source identifier
    source_table: str = ""
    source_column: str = ""
    source_row_id: Any = None
    confidence: float = 1.0
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    embeddings: Optional[List[float]] = None
    community_id: Optional[int] = None

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        if isinstance(other, KGEntity):
            return self.id == other.id
        return False

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["entity_type"] = self.entity_type.value
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "KGEntity":
        data = data.copy()
        if isinstance(data.get("entity_type"), str):
            data["entity_type"] = EntityType(data["entity_type"])
        return cls(**data)


@dataclass
class KGRelation:
    """
    Knowledge Graph Relation

    Represents a relationship between two entities.
    Includes provenance and confidence information.
    """
    source_id: str
    target_id: str
    relation_type: RelationType
    id: str = ""  # Auto-generated if not provided
    properties: Dict[str, Any] = field(default_factory=dict)
    weight: float = 1.0
    confidence: float = 1.0
    source_evidence: str = ""
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def __post_init__(self):
        if not self.id:
            # Auto-generate id from source, target, and relation type
            self.id = f"rel_{self.source_id}_{self.relation_type.value}_{self.target_id}"

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        if isinstance(other, KGRelation):
            return self.id == other.id
        return False

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["relation_type"] = self.relation_type.value
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "KGRelation":
        data = data.copy()
        if isinstance(data.get("relation_type"), str):
            data["relation_type"] = RelationType(data["relation_type"])
        return cls(**data)


@dataclass
class KGCommunity:
    """
    Knowledge Graph Community

    A cluster of related entities discovered through community detection.
    Used for hierarchical summarization (GraphRAG style).
    """
    id: int
    name: str
    description: str
    entity_ids: List[str] = field(default_factory=list)
    summary: str = ""
    key_entities: List[str] = field(default_factory=list)
    key_relations: List[str] = field(default_factory=list)
    level: int = 0  # Hierarchy level (0 = leaf, higher = more abstract)
    parent_id: Optional[int] = None
    children_ids: List[int] = field(default_factory=list)

    @property
    def size(self) -> int:
        """Number of entities in this community."""
        return len(self.entity_ids)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["size"] = self.size
        return d


class KnowledgeGraph:
    """
    Enterprise Knowledge Graph

    Core capabilities:
    - Entity and relation management
    - Multi-hop path finding
    - Subgraph extraction
    - Community detection (Louvain algorithm)
    - Graph reasoning support
    - Serialization/deserialization

    Architecture inspired by:
    - Microsoft GraphRAG
    - Palantir Foundry Ontology
    - Neo4j graph patterns
    """

    def __init__(self, name: str = "default"):
        self.name = name
        self.graph = nx.DiGraph()
        self.entities: Dict[str, KGEntity] = {}
        self.relations: Dict[str, KGRelation] = {}
        self.communities: Dict[int, KGCommunity] = {}
        self._entity_index: Dict[str, Set[str]] = defaultdict(set)  # type -> entity_ids
        self._relation_index: Dict[str, Set[str]] = defaultdict(set)  # type -> relation_ids
        self.created_at = datetime.now().isoformat()
        self.client = None  # Lazy LLM client initialization

    def _get_llm_client(self):
        if self.client is None:
            self.client = get_openai_client()
        return self.client

    # =========================================================================
    # Entity Management
    # =========================================================================

    def add_entity(self, entity: KGEntity) -> str:
        """Add an entity to the graph"""
        self.entities[entity.id] = entity
        self.graph.add_node(
            entity.id,
            entity_type=entity.entity_type.value,
            name=entity.name,
            properties=entity.properties,
            confidence=entity.confidence,
        )
        self._entity_index[entity.entity_type.value].add(entity.id)
        return entity.id

    def get_entity(self, entity_id: str) -> Optional[KGEntity]:
        """Get an entity by ID"""
        return self.entities.get(entity_id)

    def get_entities_by_type(self, entity_type: EntityType) -> List[KGEntity]:
        """Get all entities of a specific type"""
        entity_ids = self._entity_index.get(entity_type.value, set())
        return [self.entities[eid] for eid in entity_ids if eid in self.entities]

    def find_entities(
        self,
        entity_type: Optional[EntityType] = None,
        properties: Optional[Dict[str, Any]] = None,
        name_contains: Optional[str] = None,
    ) -> List[KGEntity]:
        """Find entities matching criteria"""
        results = []

        if entity_type:
            candidates = self.get_entities_by_type(entity_type)
        else:
            candidates = list(self.entities.values())

        for entity in candidates:
            if name_contains and name_contains.lower() not in entity.name.lower():
                continue

            if properties:
                match = True
                for key, value in properties.items():
                    if entity.properties.get(key) != value:
                        match = False
                        break
                if not match:
                    continue

            results.append(entity)

        return results

    # =========================================================================
    # Relation Management
    # =========================================================================

    def add_relation(self, relation: KGRelation) -> str:
        """Add a relation to the graph"""
        if relation.source_id not in self.entities:
            raise ValueError(f"Source entity {relation.source_id} not found")
        if relation.target_id not in self.entities:
            raise ValueError(f"Target entity {relation.target_id} not found")

        self.relations[relation.id] = relation
        self.graph.add_edge(
            relation.source_id,
            relation.target_id,
            relation_id=relation.id,
            relation_type=relation.relation_type.value,
            weight=relation.weight,
            confidence=relation.confidence,
        )
        self._relation_index[relation.relation_type.value].add(relation.id)
        return relation.id

    def get_relation(self, relation_id: str) -> Optional[KGRelation]:
        """Get a relation by ID"""
        return self.relations.get(relation_id)

    def get_relations_by_type(self, relation_type: RelationType) -> List[KGRelation]:
        """Get all relations of a specific type"""
        relation_ids = self._relation_index.get(relation_type.value, set())
        return [self.relations[rid] for rid in relation_ids if rid in self.relations]

    def get_entity_relations(
        self,
        entity_id: str,
        direction: str = "both"  # "outgoing", "incoming", "both"
    ) -> List[KGRelation]:
        """Get all relations connected to an entity"""
        results = []

        if direction in ("outgoing", "both"):
            for _, target, data in self.graph.out_edges(entity_id, data=True):
                rel_id = data.get("relation_id")
                if rel_id and rel_id in self.relations:
                    results.append(self.relations[rel_id])

        if direction in ("incoming", "both"):
            for source, _, data in self.graph.in_edges(entity_id, data=True):
                rel_id = data.get("relation_id")
                if rel_id and rel_id in self.relations:
                    results.append(self.relations[rel_id])

        return results

    # =========================================================================
    # Graph Traversal & Reasoning
    # =========================================================================

    def find_paths(
        self,
        source_id: str,
        target_id: str,
        max_hops: int = 3,
        relation_types: Optional[List[RelationType]] = None,
    ) -> List[List[str]]:
        """
        Find all paths between two entities.

        Multi-hop reasoning support for complex queries like:
        "How is Supplier X connected to Warehouse Y?"
        """
        if source_id not in self.graph or target_id not in self.graph:
            return []

        try:
            # Use NetworkX's all_simple_paths with cutoff
            paths = list(nx.all_simple_paths(
                self.graph,
                source=source_id,
                target=target_id,
                cutoff=max_hops
            ))

            # Filter by relation types if specified
            if relation_types:
                type_values = {rt.value for rt in relation_types}
                filtered_paths = []
                for path in paths:
                    valid = True
                    for i in range(len(path) - 1):
                        edge_data = self.graph.get_edge_data(path[i], path[i+1])
                        if edge_data and edge_data.get("relation_type") not in type_values:
                            valid = False
                            break
                    if valid:
                        filtered_paths.append(path)
                return filtered_paths

            return paths
        except nx.NetworkXNoPath:
            return []

    def get_neighbors(
        self,
        entity_id: str,
        hops: int = 1,
        relation_types: Optional[List[RelationType]] = None,
    ) -> Set[str]:
        """Get all entities within N hops of an entity"""
        if entity_id not in self.graph:
            return set()

        visited = {entity_id}
        frontier = {entity_id}

        for _ in range(hops):
            new_frontier = set()
            for node in frontier:
                # Outgoing edges
                for _, neighbor, data in self.graph.out_edges(node, data=True):
                    if relation_types:
                        if data.get("relation_type") in {rt.value for rt in relation_types}:
                            new_frontier.add(neighbor)
                    else:
                        new_frontier.add(neighbor)

                # Incoming edges
                for neighbor, _, data in self.graph.in_edges(node, data=True):
                    if relation_types:
                        if data.get("relation_type") in {rt.value for rt in relation_types}:
                            new_frontier.add(neighbor)
                    else:
                        new_frontier.add(neighbor)

            new_frontier -= visited
            visited.update(new_frontier)
            frontier = new_frontier

        visited.discard(entity_id)  # Remove starting entity
        return visited

    def extract_subgraph(
        self,
        entity_ids: List[str],
        include_neighbors: bool = True,
        neighbor_hops: int = 1,
    ) -> "KnowledgeGraph":
        """
        Extract a subgraph containing specified entities.

        Useful for:
        - Focused reasoning on relevant context
        - Reducing context window size for LLM
        - Visualization of specific relationships
        """
        subgraph = KnowledgeGraph(name=f"{self.name}_subgraph")

        # Collect all entity IDs
        all_entity_ids = set(entity_ids)

        if include_neighbors:
            for eid in entity_ids:
                neighbors = self.get_neighbors(eid, hops=neighbor_hops)
                all_entity_ids.update(neighbors)

        # Add entities
        for eid in all_entity_ids:
            if eid in self.entities:
                subgraph.add_entity(self.entities[eid])

        # Add relations between included entities
        for relation in self.relations.values():
            if relation.source_id in all_entity_ids and relation.target_id in all_entity_ids:
                subgraph.add_relation(relation)

        return subgraph

    # =========================================================================
    # Community Detection (GraphRAG style)
    # =========================================================================

    def detect_communities(self, resolution: float = 1.0) -> Dict[int, KGCommunity]:
        """
        Detect communities using Louvain algorithm.

        GraphRAG-style community detection for:
        - Hierarchical summarization
        - Topic clustering
        - Efficient global search
        """
        if not COMMUNITY_AVAILABLE:
            logger.warning("community package not available, skipping community detection")
            return {}

        if len(self.graph.nodes()) == 0:
            return {}

        # Convert to undirected for community detection
        undirected = self.graph.to_undirected()

        # Run Louvain algorithm
        partition = community_louvain.best_partition(
            undirected,
            resolution=resolution,
            random_state=42
        )

        # Create community objects
        communities_dict: Dict[int, List[str]] = defaultdict(list)
        for node_id, community_id in partition.items():
            communities_dict[community_id].append(node_id)
            # Update entity community assignment
            if node_id in self.entities:
                self.entities[node_id].community_id = community_id

        # Build KGCommunity objects
        for community_id, entity_ids in communities_dict.items():
            # Get entity names for community naming
            entity_names = [
                self.entities[eid].name
                for eid in entity_ids
                if eid in self.entities
            ][:5]

            community = KGCommunity(
                id=community_id,
                name=f"Community {community_id}",
                description=f"Cluster containing: {', '.join(entity_names)}...",
                entity_ids=entity_ids,
                key_entities=entity_ids[:10],
            )
            self.communities[community_id] = community

        logger.info(f"Detected {len(self.communities)} communities")
        return self.communities

    def summarize_community(self, community_id: int) -> str:
        """
        Generate LLM-powered summary for a community.

        GraphRAG-style hierarchical summarization.
        """
        if community_id not in self.communities:
            return ""

        community = self.communities[community_id]

        # Gather community content
        entities_info = []
        for eid in community.entity_ids[:20]:  # Limit for context
            if eid in self.entities:
                entity = self.entities[eid]
                entities_info.append({
                    "id": entity.id,
                    "type": entity.entity_type.value,
                    "name": entity.name,
                    "properties": entity.properties,
                })

        # Get internal relations
        relations_info = []
        entity_set = set(community.entity_ids)
        for relation in self.relations.values():
            if relation.source_id in entity_set and relation.target_id in entity_set:
                relations_info.append({
                    "from": self.entities[relation.source_id].name if relation.source_id in self.entities else relation.source_id,
                    "to": self.entities[relation.target_id].name if relation.target_id in self.entities else relation.target_id,
                    "type": relation.relation_type.value,
                })

        prompt = f"""다음 Knowledge Graph 커뮤니티를 분석하고 요약하세요.

엔티티 ({len(entities_info)}개):
{json.dumps(entities_info[:15], indent=2, ensure_ascii=False)}

관계 ({len(relations_info)}개):
{json.dumps(relations_info[:20], indent=2, ensure_ascii=False)}

다음 JSON 형식으로 요약하세요:
{{
    "community_name": "커뮤니티를 설명하는 이름 (예: '고위험 공급업체 그룹')",
    "summary": "이 커뮤니티의 특성을 2-3문장으로 설명",
    "key_entities": ["핵심 엔티티 ID 목록"],
    "key_findings": ["주요 발견사항 1", "주요 발견사항 2"],
    "risk_indicators": ["리스크 지표가 있다면"],
    "business_implications": "비즈니스 관점에서의 의미"
}}
"""

        try:
            response = chat_completion_with_json(
                model=DEFAULT_MODEL,
                messages=[
                    {"role": "system", "content": "당신은 Knowledge Graph 분석 전문가입니다. 엔티티와 관계를 분석하여 비즈니스 인사이트를 도출합니다."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                client=self._get_llm_client(),
            )

            result = extract_json_from_response(response)

            # Update community with summary
            community.name = result.get("community_name", community.name)
            community.summary = result.get("summary", "")
            community.key_entities = result.get("key_entities", community.key_entities)

            return community.summary

        except Exception as e:
            logger.error(f"Community summarization failed: {e}")
            return ""

    def summarize_all_communities(self) -> Dict[int, str]:
        """Summarize all communities"""
        summaries = {}
        for community_id in self.communities:
            summary = self.summarize_community(community_id)
            summaries[community_id] = summary
        return summaries

    # =========================================================================
    # Graph Statistics
    # =========================================================================

    def get_statistics(self) -> Dict[str, Any]:
        """Get graph statistics"""
        stats = {
            "name": self.name,
            "entity_count": len(self.entities),
            "relation_count": len(self.relations),
            "community_count": len(self.communities),
            "node_count": self.graph.number_of_nodes(),
            "edge_count": self.graph.number_of_edges(),
            "entities_by_type": {},
            "relations_by_type": {},
            "density": nx.density(self.graph) if self.graph.number_of_nodes() > 0 else 0,
            "is_connected": nx.is_weakly_connected(self.graph) if self.graph.number_of_nodes() > 0 else False,
        }

        # Count by type
        for entity_type, entity_ids in self._entity_index.items():
            stats["entities_by_type"][entity_type] = len(entity_ids)

        for relation_type, relation_ids in self._relation_index.items():
            stats["relations_by_type"][relation_type] = len(relation_ids)

        return stats

    # =========================================================================
    # Serialization
    # =========================================================================

    def to_dict(self) -> Dict[str, Any]:
        """Serialize graph to dictionary"""
        return {
            "name": self.name,
            "created_at": self.created_at,
            "entities": [e.to_dict() for e in self.entities.values()],
            "relations": [r.to_dict() for r in self.relations.values()],
            "communities": [c.to_dict() for c in self.communities.values()],
            "statistics": self.get_statistics(),
        }

    def to_json(self, indent: int = 2) -> str:
        """Serialize graph to JSON string"""
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False, default=str)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "KnowledgeGraph":
        """Deserialize graph from dictionary"""
        kg = cls(name=data.get("name", "imported"))
        kg.created_at = data.get("created_at", kg.created_at)

        # Load entities
        for entity_data in data.get("entities", []):
            entity = KGEntity.from_dict(entity_data)
            kg.add_entity(entity)

        # Load relations
        for relation_data in data.get("relations", []):
            relation = KGRelation.from_dict(relation_data)
            try:
                kg.add_relation(relation)
            except ValueError as e:
                logger.warning(f"Skipping relation: {e}")

        # Load communities
        for community_data in data.get("communities", []):
            community = KGCommunity(**community_data)
            kg.communities[community.id] = community

        return kg

    @classmethod
    def from_json(cls, json_str: str) -> "KnowledgeGraph":
        """Deserialize graph from JSON string"""
        data = json.loads(json_str)
        return cls.from_dict(data)


class KnowledgeGraphBuilder:
    """
    Knowledge Graph Builder

    Builds a Knowledge Graph from tabular data using:
    - Schema-based entity extraction
    - Value-based relation discovery
    - LLM-powered entity resolution

    Based on:
    - Microsoft GraphRAG extraction pipeline
    - arXiv papers on LLM-based KG construction
    """

    def __init__(self, use_llm: bool = True):
        self.use_llm = use_llm
        self.client = get_openai_client() if use_llm else None
        self.kg = KnowledgeGraph()

    def build_from_tables(
        self,
        tables_data: List[Dict[str, Any]],
        domain_context: Optional[Dict[str, Any]] = None,
    ) -> KnowledgeGraph:
        """
        Build Knowledge Graph from table data.

        Args:
            tables_data: List of table dicts with {table_name, columns, sample_data}
            domain_context: Domain context for entity type inference

        Returns:
            Constructed KnowledgeGraph
        """
        logger.info(f"Building Knowledge Graph from {len(tables_data)} tables")

        # Phase 1: Extract entities from each table
        for table_data in tables_data:
            self._extract_entities_from_table(table_data, domain_context)

        # Phase 2: Discover relations between entities
        self._discover_relations(tables_data, domain_context)

        # Phase 3: Detect communities
        if len(self.kg.entities) > 2:
            self.kg.detect_communities()

        logger.info(f"Knowledge Graph built: {self.kg.get_statistics()}")
        return self.kg

    def _extract_entities_from_table(
        self,
        table_data: Dict[str, Any],
        domain_context: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Extract entities from a single table"""
        table_name = table_data.get("table_name", "unknown")
        sample_data = table_data.get("sample_data", [])
        columns = table_data.get("columns", [])

        if not sample_data:
            return

        # Identify entity type based on table name
        entity_type = self._infer_entity_type(table_name, columns, domain_context)

        # Identify primary key column
        pk_column = self._find_primary_key(columns, sample_data)

        if not pk_column:
            logger.warning(f"No primary key found for table {table_name}")
            return

        # Extract entities from each row
        for row in sample_data:
            entity_id = row.get(pk_column)
            if not entity_id:
                continue

            # Create unique entity ID
            full_entity_id = f"{table_name}:{entity_id}"

            # Collect properties
            properties = {}
            for col_meta in columns:
                col_name = col_meta.get("name", "")
                if col_name != pk_column:
                    value = row.get(col_name)
                    if value is not None:
                        properties[col_name] = value

            # Create entity
            entity = KGEntity(
                id=full_entity_id,
                entity_type=entity_type,
                name=str(entity_id),
                properties=properties,
                source_table=table_name,
                source_column=pk_column,
                source_row_id=entity_id,
            )

            self.kg.add_entity(entity)

    def _infer_entity_type(
        self,
        table_name: str,
        columns: List[Dict],
        domain_context: Optional[Dict[str, Any]] = None,
    ) -> EntityType:
        """Infer entity type from table name"""
        table_lower = table_name.lower()

        # Pattern matching
        type_patterns = {
            EntityType.SUPPLIER: ["supplier", "vendor", "srm"],
            EntityType.CUSTOMER: ["customer", "client", "buyer"],
            EntityType.PRODUCT: ["product", "item", "sku", "goods"],
            EntityType.ORDER: ["order", "oms", "sales"],
            EntityType.PURCHASE_ORDER: ["purchase", "po", "procurement"],
            EntityType.SHIPMENT: ["shipment", "delivery", "tms", "shipping"],
            EntityType.WAREHOUSE: ["warehouse", "wms", "storage", "facility"],
            EntityType.INVENTORY: ["inventory", "stock"],
            EntityType.INVOICE: ["invoice", "billing"],
            EntityType.PAYMENT: ["payment", "transaction"],
        }

        for entity_type, patterns in type_patterns.items():
            if any(p in table_lower for p in patterns):
                return entity_type

        return EntityType.ENTITY

    def _find_primary_key(
        self,
        columns: List[Dict],
        sample_data: List[Dict],
    ) -> Optional[str]:
        """Find the primary key column"""
        # Check for explicit key candidates
        for col in columns:
            if col.get("is_key_candidate") or col.get("is_primary_key_candidate"):
                return col.get("name")

        # Look for _id suffix
        for col in columns:
            col_name = col.get("name", "").lower()
            if col_name.endswith("_id") or col_name == "id":
                # Verify uniqueness
                values = [row.get(col.get("name")) for row in sample_data]
                if len(set(values)) == len(values):
                    return col.get("name")

        # First column as fallback
        if columns:
            return columns[0].get("name")

        return None

    def _discover_relations(
        self,
        tables_data: List[Dict[str, Any]],
        domain_context: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Discover relations between entities"""

        # Build value index for relation discovery
        value_index: Dict[str, List[Tuple[str, str]]] = defaultdict(list)  # value -> [(entity_id, column)]

        for entity in self.kg.entities.values():
            # Index by source row ID
            if entity.source_row_id:
                value_index[str(entity.source_row_id)].append((entity.id, "pk"))

            # Index by FK-like columns
            for prop_name, prop_value in entity.properties.items():
                if prop_value and self._is_fk_column(prop_name):
                    value_index[str(prop_value)].append((entity.id, prop_name))

        # Find relations through value matching
        relation_counter = 0
        for value, entity_refs in value_index.items():
            if len(entity_refs) < 2:
                continue

            # Find PK entity
            pk_entities = [(eid, col) for eid, col in entity_refs if col == "pk"]
            fk_entities = [(eid, col) for eid, col in entity_refs if col != "pk"]

            # Create relations from FK to PK
            for fk_entity_id, fk_column in fk_entities:
                for pk_entity_id, _ in pk_entities:
                    if fk_entity_id == pk_entity_id:
                        continue

                    relation_type = self._infer_relation_type(fk_column)

                    relation = KGRelation(
                        id=f"REL-{relation_counter:05d}",
                        source_id=fk_entity_id,
                        target_id=pk_entity_id,
                        relation_type=relation_type,
                        properties={"join_column": fk_column, "join_value": value},
                        source_evidence=f"{fk_column} = {value}",
                    )

                    try:
                        self.kg.add_relation(relation)
                        relation_counter += 1
                    except ValueError:
                        pass  # Entity not found, skip

    def _is_fk_column(self, column_name: str) -> bool:
        """Check if column is likely a foreign key"""
        col_lower = column_name.lower()
        return (
            col_lower.endswith("_id") or
            col_lower.endswith("_key") or
            col_lower.endswith("_code") or
            col_lower.endswith("_ref") or
            "supplier" in col_lower or
            "customer" in col_lower or
            "warehouse" in col_lower or
            "order" in col_lower
        )

    def _infer_relation_type(self, fk_column: str) -> RelationType:
        """Infer relation type from FK column name"""
        col_lower = fk_column.lower()

        if "supplier" in col_lower:
            return RelationType.SUPPLIES
        elif "customer" in col_lower:
            return RelationType.ORDERS_FROM
        elif "warehouse" in col_lower or "destination" in col_lower:
            return RelationType.SHIPS_TO
        elif "parent" in col_lower:
            return RelationType.PARENT_OF
        else:
            return RelationType.REFERENCES
