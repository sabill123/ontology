"""
Graph Analysis Module

그래프 기반 관계 분석
- 관계 그래프 구축
- 경로 탐색
- 클러스터링
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Set, Tuple
from collections import defaultdict
from enum import Enum

logger = logging.getLogger(__name__)


class EdgeType(str, Enum):
    """에지 타입"""
    FOREIGN_KEY = "foreign_key"
    INDIRECT_FK = "indirect_fk"  # v7.3: Bridge Table 경유 간접 FK
    VALUE_OVERLAP = "value_overlap"
    NAME_SIMILARITY = "name_similarity"
    TYPE_COMPATIBILITY = "type_compatibility"
    SEMANTIC = "semantic"


@dataclass
class GraphEdge:
    """그래프 에지"""
    from_node: str  # table.column or table
    to_node: str
    edge_type: EdgeType
    weight: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "from": self.from_node,
            "to": self.to_node,
            "type": self.edge_type.value,
            "weight": round(self.weight, 4),
            "metadata": self.metadata,
        }


@dataclass
class GraphNode:
    """그래프 노드"""
    node_id: str  # table or table.column
    node_type: str  # table, column
    properties: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RelationshipGraph:
    """관계 그래프"""
    nodes: Dict[str, GraphNode] = field(default_factory=dict)
    edges: List[GraphEdge] = field(default_factory=list)
    adjacency: Dict[str, List[str]] = field(default_factory=lambda: defaultdict(list))

    def add_node(self, node: GraphNode):
        self.nodes[node.node_id] = node

    def add_edge(self, edge: GraphEdge):
        self.edges.append(edge)
        self.adjacency[edge.from_node].append(edge.to_node)
        self.adjacency[edge.to_node].append(edge.from_node)

    def get_neighbors(self, node_id: str) -> List[str]:
        return self.adjacency.get(node_id, [])

    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_count": len(self.nodes),
            "edge_count": len(self.edges),
            "nodes": [{"id": n.node_id, "type": n.node_type} for n in self.nodes.values()],
            "edges": [e.to_dict() for e in self.edges],
        }


class GraphAnalyzer:
    """
    그래프 분석기

    테이블/컬럼 관계를 그래프로 모델링하여 분석
    """

    def __init__(self):
        self.graph = RelationshipGraph()

    def build_relationship_graph(
        self,
        tables: Dict[str, Dict[str, Any]],
        value_overlaps: List[Dict[str, Any]],
        foreign_keys: List[Dict[str, Any]] = None,
        indirect_fks: List[Dict[str, Any]] = None,  # v7.3: Bridge Table 경유 간접 FK
    ) -> RelationshipGraph:
        """
        관계 그래프 구축

        Args:
            tables: 테이블 정보
            value_overlaps: 값 겹침 분석 결과
            foreign_keys: FK 정보
            indirect_fks: v7.3 Bridge Table 경유 간접 FK

        Returns:
            RelationshipGraph
        """
        foreign_keys = foreign_keys or []
        indirect_fks = indirect_fks or []

        self.graph = RelationshipGraph()

        # 1. 테이블 노드 추가
        for table_name, table_info in tables.items():
            self.graph.add_node(GraphNode(
                node_id=table_name,
                node_type="table",
                properties={
                    "column_count": len(table_info.get("columns", [])),
                    "row_count": table_info.get("row_count", 0),
                }
            ))

            # 컬럼 노드 추가
            for col in table_info.get("columns", []):
                col_name = col.get("name", "unknown")
                col_id = f"{table_name}.{col_name}"
                self.graph.add_node(GraphNode(
                    node_id=col_id,
                    node_type="column",
                    properties={
                        "table": table_name,
                        "dtype": col.get("dtype", "unknown"),
                        "is_key": col.get("is_key", False),
                    }
                ))

        # 2. FK 에지 추가
        for fk in foreign_keys:
            from_table = fk.get("from_table", fk.get("table"))
            from_col = fk.get("from_column", fk.get("column"))
            to_table = fk.get("to_table", fk.get("references_table"))
            to_col = fk.get("to_column", fk.get("references_column"))

            if from_table and from_col and to_table and to_col:
                self.graph.add_edge(GraphEdge(
                    from_node=f"{from_table}.{from_col}",
                    to_node=f"{to_table}.{to_col}",
                    edge_type=EdgeType.FOREIGN_KEY,
                    weight=1.0,
                    metadata={"verified": True}
                ))

        # 2.5. v7.3: Indirect FK 에지 추가 (Bridge Table 경유)
        for indirect in indirect_fks:
            # Support both naming conventions: source_table/target_table OR table_a/table_b
            source_table = indirect.get("source_table") or indirect.get("table_a")
            source_column = indirect.get("source_column") or indirect.get("column_a")
            target_table = indirect.get("target_table") or indirect.get("table_b")
            target_column = indirect.get("target_column") or indirect.get("column_b")
            bridge_table = indirect.get("bridge_table") or indirect.get("via_bridge_table", "")
            join_path = indirect.get("join_path", "")
            confidence = indirect.get("confidence", 0.8)

            if source_table and source_column and target_table and target_column:
                self.graph.add_edge(GraphEdge(
                    from_node=f"{source_table}.{source_column}",
                    to_node=f"{target_table}.{target_column}",
                    edge_type=EdgeType.INDIRECT_FK,
                    weight=confidence,
                    metadata={
                        "bridge_table": bridge_table,
                        "join_path": join_path,
                        "indirect": True,
                    }
                ))
                logger.debug(f"v7.3: Added indirect FK edge: {source_table}.{source_column} -> {target_table}.{target_column} via {bridge_table}")

        # 3. Value Overlap 에지 추가
        for overlap in value_overlaps:
            jaccard = overlap.get("jaccard_similarity", overlap.get("jaccard", 0))
            if jaccard > 0.1:
                # 다양한 키 형식 지원: table_a/column_a 또는 source_table/source_column
                from_table = overlap.get("table_a", overlap.get("source_table"))
                from_col = overlap.get("column_a", overlap.get("source_column"))
                to_table = overlap.get("table_b", overlap.get("target_table"))
                to_col = overlap.get("column_b", overlap.get("target_column"))

                if from_table and from_col and to_table and to_col:
                    self.graph.add_edge(GraphEdge(
                        from_node=f"{from_table}.{from_col}",
                        to_node=f"{to_table}.{to_col}",
                        edge_type=EdgeType.VALUE_OVERLAP,
                        weight=jaccard,
                        metadata={
                            "jaccard": jaccard,
                            "overlap_ratio_a": overlap.get("overlap_ratio_a", 0),
                            "overlap_ratio_b": overlap.get("overlap_ratio_b", 0),
                        }
                    ))

        return self.graph

    def find_connected_components(self) -> List[Set[str]]:
        """
        연결된 컴포넌트 찾기

        Returns:
            [{"table1", "table2"}, {"table3", "table4"}, ...]
        """
        visited = set()
        components = []

        def dfs(node: str, component: Set[str]):
            if node in visited:
                return
            visited.add(node)
            component.add(node)
            for neighbor in self.graph.get_neighbors(node):
                dfs(neighbor, component)

        for node_id in self.graph.nodes:
            if node_id not in visited:
                component = set()
                dfs(node_id, component)
                if component:
                    components.append(component)

        return components

    def find_bridges(self) -> List[GraphEdge]:
        """
        브릿지 에지 찾기 (제거 시 그래프가 분리되는 에지)

        데이터 사일로 간 연결점 식별에 유용
        """
        bridges = []

        # Tarjan's algorithm for bridge finding
        discovery = {}
        low = {}
        parent = {}
        time = [0]

        def dfs(u: str):
            discovery[u] = low[u] = time[0]
            time[0] += 1

            for v in self.graph.get_neighbors(u):
                if v not in discovery:
                    parent[v] = u
                    dfs(v)
                    low[u] = min(low[u], low[v])

                    # 브릿지 조건: low[v] > discovery[u]
                    if low[v] > discovery[u]:
                        # 해당 에지 찾기
                        for edge in self.graph.edges:
                            if (edge.from_node == u and edge.to_node == v) or \
                               (edge.from_node == v and edge.to_node == u):
                                bridges.append(edge)
                                break
                elif v != parent.get(u):
                    low[u] = min(low[u], discovery[v])

        for node in self.graph.nodes:
            if node not in discovery:
                parent[node] = None
                dfs(node)

        return bridges

    def find_shortest_path(
        self,
        from_node: str,
        to_node: str,
    ) -> Optional[List[str]]:
        """
        두 노드 간 최단 경로 (BFS)

        Returns:
            경로 노드 리스트 또는 None
        """
        if from_node not in self.graph.nodes or to_node not in self.graph.nodes:
            return None

        if from_node == to_node:
            return [from_node]

        from collections import deque

        visited = {from_node}
        queue = deque([(from_node, [from_node])])

        while queue:
            current, path = queue.popleft()

            for neighbor in self.graph.get_neighbors(current):
                if neighbor not in visited:
                    new_path = path + [neighbor]
                    if neighbor == to_node:
                        return new_path
                    visited.add(neighbor)
                    queue.append((neighbor, new_path))

        return None

    def find_all_paths(
        self,
        from_node: str,
        to_node: str,
        max_depth: int = 5,
    ) -> List[List[str]]:
        """
        두 노드 간 모든 경로 (깊이 제한)

        Returns:
            경로 리스트
        """
        if from_node not in self.graph.nodes or to_node not in self.graph.nodes:
            return []

        all_paths = []

        def dfs(current: str, target: str, path: List[str], visited: Set[str]):
            if len(path) > max_depth:
                return

            if current == target:
                all_paths.append(path.copy())
                return

            for neighbor in self.graph.get_neighbors(current):
                if neighbor not in visited:
                    visited.add(neighbor)
                    path.append(neighbor)
                    dfs(neighbor, target, path, visited)
                    path.pop()
                    visited.remove(neighbor)

        visited = {from_node}
        dfs(from_node, to_node, [from_node], visited)

        return all_paths

    def detect_table_relationships(self) -> List[Dict[str, Any]]:
        """
        테이블 간 관계 탐지

        Returns:
            [
                {
                    "from_table": str,
                    "to_table": str,
                    "relationship_type": str,
                    "cardinality": str,
                    "join_columns": [...],
                    "confidence": float,
                    "evidence": [...]
                }
            ]
        """
        relationships = []
        table_pairs_seen = set()

        for edge in self.graph.edges:
            # 컬럼 에지에서 테이블 추출
            if '.' in edge.from_node and '.' in edge.to_node:
                from_table = edge.from_node.split('.')[0]
                to_table = edge.to_node.split('.')[0]

                if from_table == to_table:
                    continue

                pair_key = tuple(sorted([from_table, to_table]))
                if pair_key in table_pairs_seen:
                    # 이미 본 테이블 쌍 - 기존 관계에 증거 추가
                    for rel in relationships:
                        if set([rel["from_table"], rel["to_table"]]) == set(pair_key):
                            rel["evidence"].append(edge.to_dict())
                            rel["confidence"] = min(1.0, rel["confidence"] + 0.1)
                    continue

                table_pairs_seen.add(pair_key)

                # 카디널리티 추정
                cardinality = self._estimate_cardinality(edge)

                relationships.append({
                    "from_table": from_table,
                    "to_table": to_table,
                    "relationship_type": edge.edge_type.value,
                    "cardinality": cardinality,
                    "join_columns": [
                        {
                            "from": edge.from_node.split('.')[1],
                            "to": edge.to_node.split('.')[1],
                        }
                    ],
                    "confidence": edge.weight,
                    "evidence": [edge.to_dict()],
                })

        return relationships

    def _estimate_cardinality(self, edge: GraphEdge) -> str:
        """카디널리티 추정"""
        meta = edge.metadata

        if edge.edge_type == EdgeType.FOREIGN_KEY:
            # FK는 일반적으로 N:1
            return "many_to_one"

        if edge.edge_type == EdgeType.VALUE_OVERLAP:
            overlap_a = meta.get("overlap_ratio_a", 0)
            overlap_b = meta.get("overlap_ratio_b", 0)

            if overlap_a > 0.9 and overlap_b > 0.9:
                return "one_to_one"
            elif overlap_a > 0.9:
                return "many_to_one"
            elif overlap_b > 0.9:
                return "one_to_many"
            else:
                return "many_to_many"

        return "unknown"

    def find_entity_clusters(self) -> List[Dict[str, Any]]:
        """
        테이블을 엔티티별로 클러스터링

        같은 엔티티를 나타내는 테이블들을 그룹화

        Returns:
            [
                {
                    "cluster_id": str,
                    "tables": [str],
                    "primary_table": str,
                    "confidence": float,
                    "shared_columns": [...],
                }
            ]
        """
        # 컴포넌트 기반 클러스터링
        components = self.find_connected_components()

        clusters = []
        for i, component in enumerate(components):
            # 테이블만 필터링
            tables = [n for n in component if '.' not in n]

            if len(tables) > 1:
                # 가장 많은 연결을 가진 테이블을 primary로
                table_degrees = {
                    t: len([e for e in self.graph.edges
                           if t in e.from_node or t in e.to_node])
                    for t in tables
                }
                primary = max(table_degrees, key=table_degrees.get)

                # 공유 컬럼 찾기
                shared_cols = self._find_shared_columns(tables)

                clusters.append({
                    "cluster_id": f"cluster_{i}",
                    "tables": tables,
                    "primary_table": primary,
                    "confidence": min(1.0, len(shared_cols) * 0.2 + 0.3),
                    "shared_columns": shared_cols,
                })

        return clusters

    def _find_shared_columns(self, tables: List[str]) -> List[Dict[str, Any]]:
        """테이블 간 공유 컬럼 찾기"""
        shared = []

        for edge in self.graph.edges:
            if edge.edge_type == EdgeType.VALUE_OVERLAP:
                from_table = edge.from_node.split('.')[0]
                to_table = edge.to_node.split('.')[0]

                if from_table in tables and to_table in tables:
                    shared.append({
                        "column_a": edge.from_node,
                        "column_b": edge.to_node,
                        "similarity": edge.weight,
                    })

        return shared

    def get_statistics(self) -> Dict[str, Any]:
        """그래프 통계"""
        table_nodes = [n for n in self.graph.nodes if self.graph.nodes[n].node_type == "table"]
        column_nodes = [n for n in self.graph.nodes if self.graph.nodes[n].node_type == "column"]

        edge_types = defaultdict(int)
        for edge in self.graph.edges:
            edge_types[edge.edge_type.value] += 1

        return {
            "total_nodes": len(self.graph.nodes),
            "table_count": len(table_nodes),
            "column_count": len(column_nodes),
            "edge_count": len(self.graph.edges),
            "edges_by_type": dict(edge_types),
            "connected_components": len(self.find_connected_components()),
            "density": self._compute_density(),
        }

    def _compute_density(self) -> float:
        """그래프 밀도"""
        n = len(self.graph.nodes)
        if n < 2:
            return 0.0
        max_edges = n * (n - 1) / 2
        return len(self.graph.edges) / max_edges
