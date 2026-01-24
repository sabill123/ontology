"""
Graph Embedding Module - 그래프 임베딩 및 커뮤니티 탐지

NetworkX 기반 그래프 분석과 Node2Vec 스타일 임베딩을 제공합니다.

주요 기능:
- 스키마 그래프 구축 (테이블-컬럼-관계)
- Node2Vec 스타일 그래프 임베딩
- 커뮤니티 탐지 (Louvain, Label Propagation)
- 그래프 중심성 분석 (PageRank, Betweenness, etc.)
- 그래프 유사도 계산

References:
- Grover & Leskovec, "node2vec: Scalable Feature Learning for Networks", KDD 2016
- Blondel et al., "Fast unfolding of communities in large networks", 2008
- NetworkX documentation
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Any
from enum import Enum
import numpy as np
from collections import defaultdict
import random
import hashlib


class NodeType(Enum):
    """그래프 노드 타입"""
    TABLE = "table"
    COLUMN = "column"
    VALUE = "value"
    ENTITY = "entity"


class EdgeType(Enum):
    """그래프 엣지 타입"""
    HAS_COLUMN = "has_column"           # Table -> Column
    FOREIGN_KEY = "foreign_key"         # Column -> Column
    SIMILAR_NAME = "similar_name"       # Column <-> Column
    SIMILAR_VALUE = "similar_value"     # Column <-> Column
    SAME_ENTITY = "same_entity"         # Entity <-> Entity
    VALUE_OVERLAP = "value_overlap"     # Column <-> Column


@dataclass
class GraphNode:
    """그래프 노드"""
    id: str
    node_type: NodeType
    label: str
    properties: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[np.ndarray] = None


@dataclass
class GraphEdge:
    """그래프 엣지"""
    source: str
    target: str
    edge_type: EdgeType
    weight: float = 1.0
    properties: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Community:
    """탐지된 커뮤니티"""
    id: int
    nodes: List[str]
    density: float
    modularity_contribution: float
    central_nodes: List[str]
    description: str = ""


@dataclass
class GraphEmbedding:
    """그래프 임베딩 결과"""
    node_embeddings: Dict[str, np.ndarray]
    embedding_dim: int
    method: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CentralityMetrics:
    """중심성 지표"""
    pagerank: Dict[str, float]
    betweenness: Dict[str, float]
    closeness: Dict[str, float]
    degree: Dict[str, int]
    hub_scores: Dict[str, float]
    authority_scores: Dict[str, float]


class SchemaGraph:
    """스키마 그래프 - 테이블, 컬럼, 관계를 그래프로 표현"""

    def __init__(self):
        self.nodes: Dict[str, GraphNode] = {}
        self.edges: List[GraphEdge] = []
        self.adjacency: Dict[str, List[Tuple[str, EdgeType, float]]] = defaultdict(list)

    def add_node(self, node: GraphNode):
        """노드 추가"""
        self.nodes[node.id] = node

    def add_edge(self, edge: GraphEdge):
        """엣지 추가"""
        self.edges.append(edge)
        self.adjacency[edge.source].append((edge.target, edge.edge_type, edge.weight))
        self.adjacency[edge.target].append((edge.source, edge.edge_type, edge.weight))

    def get_neighbors(self, node_id: str) -> List[Tuple[str, EdgeType, float]]:
        """이웃 노드 조회"""
        return self.adjacency.get(node_id, [])

    def get_nodes_by_type(self, node_type: NodeType) -> List[GraphNode]:
        """타입별 노드 조회"""
        return [n for n in self.nodes.values() if n.node_type == node_type]

    def to_adjacency_matrix(self) -> Tuple[np.ndarray, List[str]]:
        """인접 행렬 변환"""
        node_ids = list(self.nodes.keys())
        n = len(node_ids)
        id_to_idx = {nid: i for i, nid in enumerate(node_ids)}

        matrix = np.zeros((n, n))
        for edge in self.edges:
            if edge.source in id_to_idx and edge.target in id_to_idx:
                i, j = id_to_idx[edge.source], id_to_idx[edge.target]
                matrix[i, j] = edge.weight
                matrix[j, i] = edge.weight

        return matrix, node_ids

    def subgraph(self, node_ids: Set[str]) -> 'SchemaGraph':
        """부분 그래프 추출"""
        sub = SchemaGraph()
        for nid in node_ids:
            if nid in self.nodes:
                sub.add_node(self.nodes[nid])
        for edge in self.edges:
            if edge.source in node_ids and edge.target in node_ids:
                sub.add_edge(edge)
        return sub


class Node2VecEmbedder:
    """
    Node2Vec 스타일 그래프 임베딩

    Random walk 기반으로 노드 임베딩을 생성합니다.
    실제 Node2Vec은 Word2Vec을 사용하지만, 여기서는
    간단한 SVD 기반 임베딩을 사용합니다.

    Parameters:
    - dimensions: 임베딩 차원
    - walk_length: Random walk 길이
    - num_walks: 노드당 walk 수
    - p: Return parameter (BFS-like)
    - q: In-out parameter (DFS-like)
    """

    def __init__(
        self,
        dimensions: int = 64,
        walk_length: int = 30,
        num_walks: int = 10,
        p: float = 1.0,
        q: float = 1.0,
        seed: int = 42
    ):
        self.dimensions = dimensions
        self.walk_length = walk_length
        self.num_walks = num_walks
        self.p = p
        self.q = q
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)

    def _compute_transition_probs(
        self,
        graph: SchemaGraph
    ) -> Dict[Tuple[str, str], List[Tuple[str, float]]]:
        """전이 확률 계산 (2nd order random walk)"""
        transition_probs = {}

        for node_id in graph.nodes:
            neighbors = graph.get_neighbors(node_id)
            if not neighbors:
                continue

            for prev_id, _, _ in neighbors:
                # (prev, current) -> next 전이 확률
                probs = []
                prev_neighbors = {n[0] for n in graph.get_neighbors(prev_id)}

                for next_id, _, weight in neighbors:
                    if next_id == prev_id:
                        # 돌아가기: 1/p
                        alpha = 1.0 / self.p
                    elif next_id in prev_neighbors:
                        # 공통 이웃: 1
                        alpha = 1.0
                    else:
                        # 멀어지기: 1/q
                        alpha = 1.0 / self.q
                    probs.append((next_id, alpha * weight))

                # 정규화
                total = sum(p[1] for p in probs)
                if total > 0:
                    probs = [(p[0], p[1] / total) for p in probs]
                transition_probs[(prev_id, node_id)] = probs

        return transition_probs

    def _random_walk(
        self,
        graph: SchemaGraph,
        start_node: str,
        transition_probs: Dict
    ) -> List[str]:
        """2nd order random walk 수행"""
        walk = [start_node]

        neighbors = graph.get_neighbors(start_node)
        if not neighbors:
            return walk

        # 첫 스텝: 균등 확률
        weights = [n[2] for n in neighbors]
        total = sum(weights)
        if total > 0:
            probs = [w / total for w in weights]
            next_idx = np.random.choice(len(neighbors), p=probs)
            walk.append(neighbors[next_idx][0])

        # 이후 스텝: 2nd order 전이 확률
        while len(walk) < self.walk_length:
            cur = walk[-1]
            prev = walk[-2] if len(walk) > 1 else None

            if prev and (prev, cur) in transition_probs:
                probs_list = transition_probs[(prev, cur)]
                if probs_list:
                    nodes, probs = zip(*probs_list)
                    next_node = np.random.choice(nodes, p=probs)
                    walk.append(next_node)
                else:
                    break
            else:
                # Fallback: 균등 확률
                neighbors = graph.get_neighbors(cur)
                if neighbors:
                    weights = [n[2] for n in neighbors]
                    total = sum(weights)
                    if total > 0:
                        probs = [w / total for w in weights]
                        next_idx = np.random.choice(len(neighbors), p=probs)
                        walk.append(neighbors[next_idx][0])
                    else:
                        break
                else:
                    break

        return walk

    def _walks_to_embeddings(
        self,
        walks: List[List[str]],
        node_ids: List[str]
    ) -> Dict[str, np.ndarray]:
        """Walk 시퀀스를 임베딩으로 변환 (SVD 기반)"""
        # Co-occurrence 행렬 구축
        node_to_idx = {n: i for i, n in enumerate(node_ids)}
        n = len(node_ids)
        window_size = 5

        cooccur = np.zeros((n, n))
        for walk in walks:
            for i, node in enumerate(walk):
                if node not in node_to_idx:
                    continue
                idx_i = node_to_idx[node]

                # 윈도우 내 co-occurrence
                for j in range(max(0, i - window_size), min(len(walk), i + window_size + 1)):
                    if i != j and walk[j] in node_to_idx:
                        idx_j = node_to_idx[walk[j]]
                        cooccur[idx_i, idx_j] += 1

        # PPMI (Positive Pointwise Mutual Information)
        row_sum = cooccur.sum(axis=1, keepdims=True) + 1e-10
        col_sum = cooccur.sum(axis=0, keepdims=True) + 1e-10
        total = cooccur.sum() + 1e-10

        pmi = np.log2(cooccur * total / (row_sum * col_sum) + 1e-10)
        ppmi = np.maximum(pmi, 0)

        # SVD로 차원 축소
        try:
            U, S, Vt = np.linalg.svd(ppmi, full_matrices=False)
            dim = min(self.dimensions, len(S))
            embeddings = U[:, :dim] * np.sqrt(S[:dim])
        except np.linalg.LinAlgError:
            # SVD 실패시 랜덤 임베딩
            embeddings = np.random.randn(n, self.dimensions) * 0.1

        # 정규화
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-10
        embeddings = embeddings / norms

        return {node_ids[i]: embeddings[i] for i in range(n)}

    def fit(self, graph: SchemaGraph) -> GraphEmbedding:
        """그래프 임베딩 학습"""
        if not graph.nodes:
            return GraphEmbedding(
                node_embeddings={},
                embedding_dim=self.dimensions,
                method="node2vec_svd"
            )

        # 전이 확률 계산
        transition_probs = self._compute_transition_probs(graph)

        # Random walks 생성
        walks = []
        node_ids = list(graph.nodes.keys())
        for _ in range(self.num_walks):
            random.shuffle(node_ids)
            for node_id in node_ids:
                walk = self._random_walk(graph, node_id, transition_probs)
                walks.append(walk)

        # 임베딩 생성
        embeddings = self._walks_to_embeddings(walks, node_ids)

        # 노드에 임베딩 저장
        for node_id, emb in embeddings.items():
            if node_id in graph.nodes:
                graph.nodes[node_id].embedding = emb

        return GraphEmbedding(
            node_embeddings=embeddings,
            embedding_dim=self.dimensions,
            method="node2vec_svd",
            metadata={
                "walk_length": self.walk_length,
                "num_walks": self.num_walks,
                "p": self.p,
                "q": self.q,
                "num_nodes": len(node_ids),
                "num_walks_total": len(walks)
            }
        )


class CommunityDetector:
    """
    커뮤니티 탐지

    Louvain 알고리즘과 Label Propagation을 구현합니다.
    """

    def __init__(self, resolution: float = 1.0, seed: int = 42):
        self.resolution = resolution
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)

    def _compute_modularity(
        self,
        graph: SchemaGraph,
        communities: Dict[str, int]
    ) -> float:
        """모듈러리티 계산"""
        m = sum(e.weight for e in graph.edges)
        if m == 0:
            return 0.0

        degree = defaultdict(float)
        for edge in graph.edges:
            degree[edge.source] += edge.weight
            degree[edge.target] += edge.weight

        q = 0.0
        for edge in graph.edges:
            if communities.get(edge.source) == communities.get(edge.target):
                q += edge.weight - (degree[edge.source] * degree[edge.target]) / (2 * m)

        return q / (2 * m) if m > 0 else 0.0

    def louvain(self, graph: SchemaGraph) -> List[Community]:
        """Louvain 커뮤니티 탐지"""
        if not graph.nodes:
            return []

        # 초기화: 각 노드가 자신만의 커뮤니티
        node_ids = list(graph.nodes.keys())
        communities = {n: i for i, n in enumerate(node_ids)}

        # 엣지 가중치 합계
        m = sum(e.weight for e in graph.edges)
        if m == 0:
            # 엣지 없으면 모든 노드가 개별 커뮤니티
            return [
                Community(
                    id=i,
                    nodes=[n],
                    density=0.0,
                    modularity_contribution=0.0,
                    central_nodes=[n]
                )
                for i, n in enumerate(node_ids)
            ]

        # 노드별 차수
        degree = defaultdict(float)
        for edge in graph.edges:
            degree[edge.source] += edge.weight
            degree[edge.target] += edge.weight

        # 이웃 커뮤니티별 가중치
        def get_neighbor_communities(node: str) -> Dict[int, float]:
            neighbor_weights = defaultdict(float)
            for target, _, weight in graph.get_neighbors(node):
                comm = communities[target]
                neighbor_weights[comm] += weight
            return neighbor_weights

        # Phase 1: 로컬 최적화
        improved = True
        max_iterations = 100
        iteration = 0

        while improved and iteration < max_iterations:
            improved = False
            random.shuffle(node_ids)

            for node in node_ids:
                current_comm = communities[node]
                neighbor_weights = get_neighbor_communities(node)

                # 현재 커뮤니티에서 제거시 모듈러리티 변화
                comm_nodes = [n for n in node_ids if communities[n] == current_comm]
                current_gain = neighbor_weights.get(current_comm, 0)

                best_comm = current_comm
                best_gain = 0

                for comm, weight in neighbor_weights.items():
                    if comm == current_comm:
                        continue

                    # 해당 커뮤니티로 이동시 게인
                    gain = weight - current_gain
                    if gain > best_gain:
                        best_gain = gain
                        best_comm = comm

                if best_comm != current_comm:
                    communities[node] = best_comm
                    improved = True

            iteration += 1

        # 커뮤니티 재번호화
        unique_comms = sorted(set(communities.values()))
        comm_map = {c: i for i, c in enumerate(unique_comms)}
        communities = {n: comm_map[c] for n, c in communities.items()}

        # Community 객체 생성
        result = []
        for comm_id in set(communities.values()):
            comm_nodes = [n for n in node_ids if communities[n] == comm_id]

            # 밀도 계산
            subgraph = graph.subgraph(set(comm_nodes))
            num_edges = len(subgraph.edges)
            max_edges = len(comm_nodes) * (len(comm_nodes) - 1) / 2
            density = num_edges / max_edges if max_edges > 0 else 0

            # 중심 노드 (차수 기준)
            node_degrees = [(n, degree[n]) for n in comm_nodes]
            node_degrees.sort(key=lambda x: -x[1])
            central = [n for n, _ in node_degrees[:3]]

            result.append(Community(
                id=comm_id,
                nodes=comm_nodes,
                density=density,
                modularity_contribution=0.0,  # 개별 계산 복잡
                central_nodes=central
            ))

        return result

    def label_propagation(self, graph: SchemaGraph) -> List[Community]:
        """Label Propagation 커뮤니티 탐지"""
        if not graph.nodes:
            return []

        node_ids = list(graph.nodes.keys())
        labels = {n: i for i, n in enumerate(node_ids)}

        max_iterations = 100
        for _ in range(max_iterations):
            changed = False
            random.shuffle(node_ids)

            for node in node_ids:
                neighbors = graph.get_neighbors(node)
                if not neighbors:
                    continue

                # 이웃 라벨 집계
                label_weights = defaultdict(float)
                for target, _, weight in neighbors:
                    label_weights[labels[target]] += weight

                if label_weights:
                    # 최다 라벨 선택
                    max_weight = max(label_weights.values())
                    best_labels = [l for l, w in label_weights.items() if w == max_weight]
                    new_label = random.choice(best_labels)

                    if new_label != labels[node]:
                        labels[node] = new_label
                        changed = True

            if not changed:
                break

        # 커뮤니티 재번호화
        unique_labels = sorted(set(labels.values()))
        label_map = {l: i for i, l in enumerate(unique_labels)}
        labels = {n: label_map[l] for n, l in labels.items()}

        # 차수 계산
        degree = defaultdict(float)
        for edge in graph.edges:
            degree[edge.source] += edge.weight
            degree[edge.target] += edge.weight

        # Community 객체 생성
        result = []
        for comm_id in set(labels.values()):
            comm_nodes = [n for n in node_ids if labels[n] == comm_id]

            subgraph = graph.subgraph(set(comm_nodes))
            num_edges = len(subgraph.edges)
            max_edges = len(comm_nodes) * (len(comm_nodes) - 1) / 2
            density = num_edges / max_edges if max_edges > 0 else 0

            node_degrees = [(n, degree[n]) for n in comm_nodes]
            node_degrees.sort(key=lambda x: -x[1])
            central = [n for n, _ in node_degrees[:3]]

            result.append(Community(
                id=comm_id,
                nodes=comm_nodes,
                density=density,
                modularity_contribution=0.0,
                central_nodes=central
            ))

        return result


class CentralityAnalyzer:
    """그래프 중심성 분석"""

    def __init__(self, max_iterations: int = 100, tolerance: float = 1e-6):
        self.max_iterations = max_iterations
        self.tolerance = tolerance

    def compute_degree(self, graph: SchemaGraph) -> Dict[str, int]:
        """차수 중심성"""
        degree = defaultdict(int)
        for edge in graph.edges:
            degree[edge.source] += 1
            degree[edge.target] += 1
        return dict(degree)

    def compute_pagerank(
        self,
        graph: SchemaGraph,
        damping: float = 0.85
    ) -> Dict[str, float]:
        """PageRank 계산"""
        if not graph.nodes:
            return {}

        n = len(graph.nodes)
        node_ids = list(graph.nodes.keys())
        pr = {node_id: 1.0 / n for node_id in node_ids}

        # Out-degree 계산
        out_degree = defaultdict(float)
        for edge in graph.edges:
            out_degree[edge.source] += edge.weight
            out_degree[edge.target] += edge.weight

        for _ in range(self.max_iterations):
            new_pr = {}
            for node in node_ids:
                rank = (1 - damping) / n
                for neighbor, _, weight in graph.get_neighbors(node):
                    if out_degree[neighbor] > 0:
                        rank += damping * pr[neighbor] * weight / out_degree[neighbor]
                new_pr[node] = rank

            # 수렴 체크
            diff = sum(abs(new_pr[n] - pr[n]) for n in node_ids)
            pr = new_pr
            if diff < self.tolerance:
                break

        return pr

    def compute_betweenness(self, graph: SchemaGraph) -> Dict[str, float]:
        """Betweenness 중심성 (근사)"""
        if not graph.nodes:
            return {}

        node_ids = list(graph.nodes.keys())
        betweenness = {n: 0.0 for n in node_ids}

        # 샘플링 기반 근사 (전체 계산은 O(VE)로 비쌈)
        sample_size = min(50, len(node_ids))
        sample_nodes = random.sample(node_ids, sample_size)

        for source in sample_nodes:
            # BFS로 최단 경로 계산
            dist = {source: 0}
            pred = defaultdict(list)
            queue = [source]

            while queue:
                current = queue.pop(0)
                for neighbor, _, _ in graph.get_neighbors(current):
                    if neighbor not in dist:
                        dist[neighbor] = dist[current] + 1
                        queue.append(neighbor)
                        pred[neighbor].append(current)
                    elif dist[neighbor] == dist[current] + 1:
                        pred[neighbor].append(current)

            # 역방향 누적
            sigma = {n: 0.0 for n in node_ids}
            sigma[source] = 1.0

            for node in sorted(dist.keys(), key=lambda x: -dist.get(x, float('inf'))):
                if node == source:
                    continue
                for p in pred[node]:
                    sigma[p] += sigma[node] / len(pred[node])
                betweenness[node] += sigma[node]

        # 정규화
        scale = 1.0 / (sample_size * (len(node_ids) - 1)) if len(node_ids) > 1 else 1.0
        return {n: b * scale for n, b in betweenness.items()}

    def compute_closeness(self, graph: SchemaGraph) -> Dict[str, float]:
        """Closeness 중심성"""
        if not graph.nodes:
            return {}

        node_ids = list(graph.nodes.keys())
        closeness = {}

        for source in node_ids:
            # BFS로 거리 계산
            dist = {source: 0}
            queue = [source]

            while queue:
                current = queue.pop(0)
                for neighbor, _, _ in graph.get_neighbors(current):
                    if neighbor not in dist:
                        dist[neighbor] = dist[current] + 1
                        queue.append(neighbor)

            # Closeness = (도달 가능 노드 수 - 1) / 거리 합
            total_dist = sum(dist.values())
            reachable = len(dist) - 1

            if total_dist > 0 and reachable > 0:
                closeness[source] = reachable / total_dist
            else:
                closeness[source] = 0.0

        return closeness

    def compute_hits(
        self,
        graph: SchemaGraph
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        """HITS (Hub & Authority) 점수"""
        if not graph.nodes:
            return {}, {}

        node_ids = list(graph.nodes.keys())
        hub = {n: 1.0 for n in node_ids}
        auth = {n: 1.0 for n in node_ids}

        for _ in range(self.max_iterations):
            new_auth = {}
            new_hub = {}

            # Authority 업데이트
            for node in node_ids:
                score = 0.0
                for neighbor, _, weight in graph.get_neighbors(node):
                    score += hub[neighbor] * weight
                new_auth[node] = score

            # Hub 업데이트
            for node in node_ids:
                score = 0.0
                for neighbor, _, weight in graph.get_neighbors(node):
                    score += auth[neighbor] * weight
                new_hub[node] = score

            # 정규화
            auth_norm = np.sqrt(sum(v ** 2 for v in new_auth.values())) + 1e-10
            hub_norm = np.sqrt(sum(v ** 2 for v in new_hub.values())) + 1e-10

            auth = {n: v / auth_norm for n, v in new_auth.items()}
            hub = {n: v / hub_norm for n, v in new_hub.items()}

        return hub, auth

    def compute_all(self, graph: SchemaGraph) -> CentralityMetrics:
        """모든 중심성 지표 계산"""
        hub, auth = self.compute_hits(graph)

        return CentralityMetrics(
            pagerank=self.compute_pagerank(graph),
            betweenness=self.compute_betweenness(graph),
            closeness=self.compute_closeness(graph),
            degree=self.compute_degree(graph),
            hub_scores=hub,
            authority_scores=auth
        )


class GraphSimilarity:
    """그래프 유사도 계산"""

    @staticmethod
    def embedding_similarity(emb1: GraphEmbedding, emb2: GraphEmbedding) -> float:
        """임베딩 기반 그래프 유사도"""
        if not emb1.node_embeddings or not emb2.node_embeddings:
            return 0.0

        # 각 그래프의 평균 임베딩
        emb1_mean = np.mean(list(emb1.node_embeddings.values()), axis=0)
        emb2_mean = np.mean(list(emb2.node_embeddings.values()), axis=0)

        # 차원 맞추기
        min_dim = min(len(emb1_mean), len(emb2_mean))
        emb1_mean = emb1_mean[:min_dim]
        emb2_mean = emb2_mean[:min_dim]

        # 코사인 유사도
        norm1 = np.linalg.norm(emb1_mean)
        norm2 = np.linalg.norm(emb2_mean)

        if norm1 > 0 and norm2 > 0:
            return float(np.dot(emb1_mean, emb2_mean) / (norm1 * norm2))
        return 0.0

    @staticmethod
    def structural_similarity(graph1: SchemaGraph, graph2: SchemaGraph) -> float:
        """구조적 유사도 (Jaccard 기반)"""
        # 노드 타입 분포
        type1 = defaultdict(int)
        type2 = defaultdict(int)

        for node in graph1.nodes.values():
            type1[node.node_type] += 1
        for node in graph2.nodes.values():
            type2[node.node_type] += 1

        # Jaccard 유사도
        all_types = set(type1.keys()) | set(type2.keys())
        if not all_types:
            return 0.0

        intersection = sum(min(type1[t], type2[t]) for t in all_types)
        union = sum(max(type1[t], type2[t]) for t in all_types)

        type_sim = intersection / union if union > 0 else 0.0

        # 엣지 타입 분포
        edge_type1 = defaultdict(int)
        edge_type2 = defaultdict(int)

        for edge in graph1.edges:
            edge_type1[edge.edge_type] += 1
        for edge in graph2.edges:
            edge_type2[edge.edge_type] += 1

        all_edge_types = set(edge_type1.keys()) | set(edge_type2.keys())
        if not all_edge_types:
            edge_sim = 0.0
        else:
            e_intersection = sum(min(edge_type1[t], edge_type2[t]) for t in all_edge_types)
            e_union = sum(max(edge_type1[t], edge_type2[t]) for t in all_edge_types)
            edge_sim = e_intersection / e_union if e_union > 0 else 0.0

        return (type_sim + edge_sim) / 2


class SchemaGraphBuilder:
    """스키마 정보로부터 그래프 구축"""

    def __init__(self, include_values: bool = False):
        self.include_values = include_values

    def build_from_tables(
        self,
        tables: Dict[str, Dict],
        fk_relations: List[Dict] = None,
        value_overlaps: List[Dict] = None
    ) -> SchemaGraph:
        """
        테이블 정보로부터 그래프 구축

        Parameters:
        - tables: {table_name: {columns: [...], data: [...]}}
        - fk_relations: [{source_table, source_column, target_table, target_column}, ...]
        - value_overlaps: [{col1, col2, overlap_ratio}, ...]
        """
        graph = SchemaGraph()

        # 테이블 및 컬럼 노드 추가
        for table_name, table_info in tables.items():
            # 테이블 노드
            table_node = GraphNode(
                id=f"table:{table_name}",
                node_type=NodeType.TABLE,
                label=table_name,
                properties={
                    "row_count": len(table_info.get("data", [])),
                    "column_count": len(table_info.get("columns", []))
                }
            )
            graph.add_node(table_node)

            # 컬럼 노드
            columns = table_info.get("columns", [])
            for col in columns:
                if isinstance(col, dict):
                    col_name = col.get("name", col.get("column_name", "unknown"))
                    col_type = col.get("type", col.get("data_type", "unknown"))
                else:
                    col_name = str(col)
                    col_type = "unknown"

                col_id = f"column:{table_name}.{col_name}"
                col_node = GraphNode(
                    id=col_id,
                    node_type=NodeType.COLUMN,
                    label=col_name,
                    properties={
                        "table": table_name,
                        "data_type": col_type
                    }
                )
                graph.add_node(col_node)

                # 테이블-컬럼 엣지
                graph.add_edge(GraphEdge(
                    source=f"table:{table_name}",
                    target=col_id,
                    edge_type=EdgeType.HAS_COLUMN,
                    weight=1.0
                ))

        # FK 관계 엣지
        if fk_relations:
            for fk in fk_relations:
                source_col = f"column:{fk['source_table']}.{fk['source_column']}"
                target_col = f"column:{fk['target_table']}.{fk['target_column']}"

                if source_col in graph.nodes and target_col in graph.nodes:
                    graph.add_edge(GraphEdge(
                        source=source_col,
                        target=target_col,
                        edge_type=EdgeType.FOREIGN_KEY,
                        weight=fk.get("confidence", 1.0)
                    ))

        # 값 겹침 엣지
        if value_overlaps:
            for overlap in value_overlaps:
                col1 = overlap.get("col1", overlap.get("column1", ""))
                col2 = overlap.get("col2", overlap.get("column2", ""))
                ratio = overlap.get("overlap_ratio", overlap.get("similarity", 0.5))

                if col1 in graph.nodes and col2 in graph.nodes:
                    graph.add_edge(GraphEdge(
                        source=col1,
                        target=col2,
                        edge_type=EdgeType.VALUE_OVERLAP,
                        weight=ratio
                    ))

        return graph

    def add_name_similarity_edges(
        self,
        graph: SchemaGraph,
        threshold: float = 0.7
    ):
        """컬럼명 유사도 기반 엣지 추가"""
        columns = graph.get_nodes_by_type(NodeType.COLUMN)

        for i, col1 in enumerate(columns):
            for col2 in columns[i + 1:]:
                # 다른 테이블의 컬럼만
                if col1.properties.get("table") == col2.properties.get("table"):
                    continue

                # Jaro-Winkler 유사도
                sim = self._jaro_winkler(col1.label.lower(), col2.label.lower())

                if sim >= threshold:
                    graph.add_edge(GraphEdge(
                        source=col1.id,
                        target=col2.id,
                        edge_type=EdgeType.SIMILAR_NAME,
                        weight=sim
                    ))

    def _jaro_winkler(self, s1: str, s2: str) -> float:
        """Jaro-Winkler 유사도"""
        if s1 == s2:
            return 1.0
        if not s1 or not s2:
            return 0.0

        # Jaro 계산
        len1, len2 = len(s1), len(s2)
        match_distance = max(len1, len2) // 2 - 1

        s1_matches = [False] * len1
        s2_matches = [False] * len2

        matches = 0
        transpositions = 0

        for i in range(len1):
            start = max(0, i - match_distance)
            end = min(i + match_distance + 1, len2)

            for j in range(start, end):
                if s2_matches[j] or s1[i] != s2[j]:
                    continue
                s1_matches[i] = True
                s2_matches[j] = True
                matches += 1
                break

        if matches == 0:
            return 0.0

        k = 0
        for i in range(len1):
            if not s1_matches[i]:
                continue
            while not s2_matches[k]:
                k += 1
            if s1[i] != s2[k]:
                transpositions += 1
            k += 1

        jaro = (matches / len1 + matches / len2 +
                (matches - transpositions / 2) / matches) / 3

        # Winkler 보정
        prefix = 0
        for i in range(min(4, len1, len2)):
            if s1[i] == s2[i]:
                prefix += 1
            else:
                break

        return jaro + prefix * 0.1 * (1 - jaro)


class GraphEmbeddingAnalyzer:
    """통합 그래프 임베딩 분석기"""

    def __init__(
        self,
        embedding_dim: int = 64,
        walk_length: int = 30,
        num_walks: int = 10
    ):
        self.embedder = Node2VecEmbedder(
            dimensions=embedding_dim,
            walk_length=walk_length,
            num_walks=num_walks
        )
        self.community_detector = CommunityDetector()
        self.centrality_analyzer = CentralityAnalyzer()
        self.graph_builder = SchemaGraphBuilder()

    def analyze_schema(
        self,
        tables: Dict[str, Dict],
        fk_relations: List[Dict] = None,
        value_overlaps: List[Dict] = None
    ) -> Dict[str, Any]:
        """
        스키마 그래프 종합 분석

        Returns:
        - graph: SchemaGraph
        - embedding: GraphEmbedding
        - communities: List[Community]
        - centrality: CentralityMetrics
        - summary: Dict
        """
        # 그래프 구축
        graph = self.graph_builder.build_from_tables(
            tables, fk_relations, value_overlaps
        )
        self.graph_builder.add_name_similarity_edges(graph)

        # 임베딩
        embedding = self.embedder.fit(graph)

        # 커뮤니티 탐지
        communities = self.community_detector.louvain(graph)

        # 중심성 분석
        centrality = self.centrality_analyzer.compute_all(graph)

        # 요약 통계
        summary = {
            "num_nodes": len(graph.nodes),
            "num_edges": len(graph.edges),
            "num_tables": len(graph.get_nodes_by_type(NodeType.TABLE)),
            "num_columns": len(graph.get_nodes_by_type(NodeType.COLUMN)),
            "num_communities": len(communities),
            "avg_community_size": np.mean([len(c.nodes) for c in communities]) if communities else 0,
            "top_pagerank_nodes": sorted(
                centrality.pagerank.items(),
                key=lambda x: -x[1]
            )[:5],
            "top_betweenness_nodes": sorted(
                centrality.betweenness.items(),
                key=lambda x: -x[1]
            )[:5]
        }

        return {
            "graph": graph,
            "embedding": embedding,
            "communities": communities,
            "centrality": centrality,
            "summary": summary
        }

    def find_similar_nodes(
        self,
        embedding: GraphEmbedding,
        node_id: str,
        top_k: int = 5
    ) -> List[Tuple[str, float]]:
        """임베딩 기반 유사 노드 검색"""
        if node_id not in embedding.node_embeddings:
            return []

        target_emb = embedding.node_embeddings[node_id]
        similarities = []

        for other_id, other_emb in embedding.node_embeddings.items():
            if other_id == node_id:
                continue

            # 코사인 유사도
            norm1 = np.linalg.norm(target_emb)
            norm2 = np.linalg.norm(other_emb)

            if norm1 > 0 and norm2 > 0:
                sim = float(np.dot(target_emb, other_emb) / (norm1 * norm2))
                similarities.append((other_id, sim))

        similarities.sort(key=lambda x: -x[1])
        return similarities[:top_k]

    def get_community_insights(
        self,
        graph: SchemaGraph,
        communities: List[Community]
    ) -> List[Dict]:
        """커뮤니티별 인사이트 도출"""
        insights = []

        for comm in communities:
            # 커뮤니티 내 테이블
            tables = set()
            columns = []

            for node_id in comm.nodes:
                if node_id in graph.nodes:
                    node = graph.nodes[node_id]
                    if node.node_type == NodeType.TABLE:
                        tables.add(node.label)
                    elif node.node_type == NodeType.COLUMN:
                        columns.append(node.label)
                        tables.add(node.properties.get("table", ""))

            # 커뮤니티 특성 분석
            insight = {
                "community_id": comm.id,
                "tables": list(tables),
                "num_columns": len(columns),
                "density": comm.density,
                "central_nodes": comm.central_nodes,
                "interpretation": self._interpret_community(tables, columns, comm.density)
            }
            insights.append(insight)

        return insights

    def _interpret_community(
        self,
        tables: Set[str],
        columns: List[str],
        density: float
    ) -> str:
        """커뮤니티 해석"""
        if len(tables) == 1:
            return f"단일 테이블 '{list(tables)[0]}'의 컬럼들로 구성된 응집 그룹"
        elif len(tables) == 2:
            return f"2개 테이블 간 강한 관계: {', '.join(tables)}"
        else:
            if density > 0.5:
                return f"밀접하게 연결된 {len(tables)}개 테이블 클러스터"
            else:
                return f"느슨하게 연결된 {len(tables)}개 테이블 그룹"

    # =========================================================================
    # 편의 메서드: discovery.py 에이전트에서 사용하는 API
    # =========================================================================

    def build_schema_graph(
        self,
        tables: Dict[str, Dict[str, Any]],
        foreign_keys: List[Dict[str, Any]],
        value_overlaps: List[Dict[str, Any]] = None
    ) -> SchemaGraph:
        """
        스키마 그래프 생성 (에이전트 호출용)

        Args:
            tables: {table_name: {"columns": [...]}}
            foreign_keys: [{"source_table": ..., "source_column": ..., "target_table": ..., "target_column": ...}]
            value_overlaps: [{"table_a": ..., "column_a": ..., "table_b": ..., "column_b": ..., "jaccard": ...}]

        Returns:
            SchemaGraph 객체
        """
        return self.graph_builder.build_from_tables(tables, foreign_keys, value_overlaps)

    def compute_embeddings(self, graph: SchemaGraph) -> GraphEmbedding:
        """
        그래프 임베딩 계산 (에이전트 호출용)

        Args:
            graph: SchemaGraph 객체

        Returns:
            GraphEmbedding 객체
        """
        return self.embedder.fit(graph)

    def detect_communities(self, graph: SchemaGraph) -> List[Community]:
        """
        커뮤니티 탐지 (에이전트 호출용)

        Args:
            graph: SchemaGraph 객체

        Returns:
            Community 리스트
        """
        return self.community_detector.louvain(graph)

    def analyze_centrality(self, graph: SchemaGraph) -> CentralityMetrics:
        """
        중심성 분석 (에이전트 호출용)

        Args:
            graph: SchemaGraph 객체

        Returns:
            CentralityMetrics 객체
        """
        return self.centrality_analyzer.compute_all(graph)
