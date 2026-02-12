"""
TDA (Topological Data Analysis) Module

실제 위상수학적 데이터 분석 구현
- Betti numbers 계산
- Persistence diagrams
- Homeomorphism detection via topological signatures
"""

import logging
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple, Set
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class BettiNumbers:
    """Betti numbers 결과"""
    beta_0: int  # Connected components
    beta_1: int  # 1-dimensional holes (cycles)
    beta_2: int  # 2-dimensional voids

    @property
    def euler_characteristic(self) -> int:
        """오일러 특성: χ = β₀ - β₁ + β₂"""
        return self.beta_0 - self.beta_1 + self.beta_2

    def distance(self, other: "BettiNumbers") -> float:
        """Betti distance (L2)"""
        return np.sqrt(
            (self.beta_0 - other.beta_0) ** 2 +
            (self.beta_1 - other.beta_1) ** 2 +
            (self.beta_2 - other.beta_2) ** 2
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "beta_0": self.beta_0,
            "beta_1": self.beta_1,
            "beta_2": self.beta_2,
            "euler_characteristic": self.euler_characteristic,
        }


@dataclass
class PersistencePair:
    """Persistence diagram의 birth-death 쌍"""
    dimension: int
    birth: float
    death: float

    @property
    def persistence(self) -> float:
        """Persistence = death - birth"""
        return self.death - self.birth


@dataclass
class TDASignature:
    """테이블의 TDA 서명"""
    table_name: str
    betti_numbers: BettiNumbers
    persistence_pairs: List[PersistencePair]
    key_columns: List[str]
    structural_complexity: str  # low, medium, high

    # 추가 메트릭
    total_persistence: float = 0.0
    max_persistence: float = 0.0
    column_count: int = 0
    relationship_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "table_name": self.table_name,
            "betti_numbers": self.betti_numbers.to_dict(),
            "betti_0": self.betti_numbers.beta_0,
            "betti_1": self.betti_numbers.beta_1,
            "betti_2": self.betti_numbers.beta_2,
            "euler_characteristic": self.betti_numbers.euler_characteristic,
            "total_persistence": self.total_persistence,
            "max_persistence": self.max_persistence,
            "key_columns": self.key_columns,
            "structural_complexity": self.structural_complexity,
            "column_count": self.column_count,
            "relationship_count": self.relationship_count,
        }


class TDAAnalyzer:
    """
    TDA 분석기

    테이블 스키마를 그래프로 변환하여 위상수학적 특성 계산

    알고리즘:
    1. 스키마를 Simplicial Complex로 변환
       - 0-simplex: 컬럼
       - 1-simplex: 컬럼 간 관계 (FK, 타입 유사성)
       - 2-simplex: 삼각 관계
    2. Betti numbers 계산
    3. Persistence 계산
    """

    def __init__(self):
        self.signatures: Dict[str, TDASignature] = {}

    def compute_signature(
        self,
        table_name: str,
        columns: List[Dict[str, Any]],
        foreign_keys: List[Dict[str, Any]] = None,
        sample_data: List[Dict[str, Any]] = None,
    ) -> TDASignature:
        """
        테이블의 TDA 서명 계산

        Args:
            table_name: 테이블 이름
            columns: 컬럼 정보 리스트
            foreign_keys: FK 정보
            sample_data: 샘플 데이터

        Returns:
            TDASignature
        """
        foreign_keys = foreign_keys or []
        sample_data = sample_data or []

        # 1. 컬럼 그래프 구축
        adjacency = self._build_column_graph(columns, foreign_keys, sample_data)

        # 2. Betti numbers 계산
        betti = self._compute_betti_numbers(adjacency, len(columns))

        # 3. Persistence 계산
        persistence_pairs = self._compute_persistence(adjacency, columns)

        # 4. Key 컬럼 식별
        key_columns = self._identify_key_columns(columns)

        # 5. 복잡도 평가
        complexity = self._assess_complexity(betti, len(columns), len(adjacency))

        # 6. Persistence 메트릭 (inf 제외)
        finite_persistence = [
            p.persistence for p in persistence_pairs
            if p.death != float('inf')
        ]
        total_persistence = sum(finite_persistence) if finite_persistence else 0.0
        max_persistence = max(finite_persistence, default=0.0)

        signature = TDASignature(
            table_name=table_name,
            betti_numbers=betti,
            persistence_pairs=persistence_pairs,
            key_columns=key_columns,
            structural_complexity=complexity,
            total_persistence=total_persistence,
            max_persistence=max_persistence,
            column_count=len(columns),
            relationship_count=len(adjacency),
        )

        self.signatures[table_name] = signature
        return signature

    def _build_column_graph(
        self,
        columns: List[Any],
        foreign_keys: List[Dict[str, Any]],
        sample_data: List[Dict[str, Any]],
    ) -> List[Tuple[int, int, float]]:
        """
        컬럼 관계 그래프 구축

        Returns:
            Edge list: [(col_idx_a, col_idx_b, weight), ...]
        """
        edges = []

        # Handle both dict format and string format for columns
        if columns and isinstance(columns[0], str):
            # String format: ["col1", "col2", ...]
            col_names = columns
            columns = [{"name": c, "dtype": "unknown"} for c in columns]
        else:
            # Dict format: [{"name": "col1", "dtype": "int"}, ...]
            col_names = [c.get("name", f"col_{i}") for i, c in enumerate(columns)]

        col_index = {name: i for i, name in enumerate(col_names)}

        # 1. FK 관계
        for fk in foreign_keys:
            src_col = fk.get("column") or fk.get("from_column")
            if src_col in col_index:
                # FK는 강한 연결
                edges.append((col_index[src_col], col_index[src_col], 1.0))

        # 2. 타입 유사성
        type_groups = defaultdict(list)
        for i, col in enumerate(columns):
            dtype = col.get("dtype") or col.get("inferred_type") or "unknown"
            base_type = self._normalize_type(dtype)
            type_groups[base_type].append(i)

        # 같은 타입의 컬럼들은 연결
        for type_name, indices in type_groups.items():
            for i in range(len(indices)):
                for j in range(i + 1, len(indices)):
                    weight = 0.3 if type_name != "unknown" else 0.1
                    edges.append((indices[i], indices[j], weight))

        # 3. 이름 유사성
        for i in range(len(col_names)):
            for j in range(i + 1, len(col_names)):
                similarity = self._name_similarity(col_names[i], col_names[j])
                if similarity > 0.3:
                    edges.append((i, j, similarity * 0.5))

        # 4. 값 공존 패턴 (sample_data 기반)
        if sample_data:
            coexistence = self._compute_coexistence(sample_data, col_names)
            for (i, j), weight in coexistence.items():
                if weight > 0.5:
                    edges.append((i, j, weight * 0.4))

        return edges

    def _normalize_type(self, dtype: str) -> str:
        """타입 정규화"""
        dtype_lower = str(dtype).lower()

        if any(t in dtype_lower for t in ["int", "bigint", "smallint", "tinyint"]):
            return "integer"
        elif any(t in dtype_lower for t in ["float", "double", "decimal", "numeric", "real"]):
            return "float"
        elif any(t in dtype_lower for t in ["varchar", "char", "text", "string", "object"]):
            return "string"
        elif any(t in dtype_lower for t in ["date", "time", "timestamp", "datetime"]):
            return "datetime"
        elif any(t in dtype_lower for t in ["bool", "boolean"]):
            return "boolean"
        elif any(t in dtype_lower for t in ["uuid", "guid"]):
            return "uuid"
        else:
            return "unknown"

    def _name_similarity(self, name1: str, name2: str) -> float:
        """컬럼 이름 유사도 (Jaccard on tokens)"""
        # 토큰화
        tokens1 = set(self._tokenize_name(name1))
        tokens2 = set(self._tokenize_name(name2))

        if not tokens1 or not tokens2:
            return 0.0

        intersection = len(tokens1 & tokens2)
        union = len(tokens1 | tokens2)

        return intersection / union if union > 0 else 0.0

    def _tokenize_name(self, name: str) -> List[str]:
        """이름을 토큰으로 분리"""
        import re
        # snake_case, camelCase, PascalCase 분리
        tokens = re.split(r'[_\-\s]', name)
        result = []
        for token in tokens:
            # camelCase 분리
            parts = re.findall(r'[A-Z]?[a-z]+|[A-Z]+(?=[A-Z]|$)', token)
            result.extend(p.lower() for p in parts if p)
        return [t for t in result if len(t) > 1]

    def _compute_coexistence(
        self,
        sample_data: List[Dict[str, Any]],
        col_names: List[str],
    ) -> Dict[Tuple[int, int], float]:
        """
        값 공존 패턴 계산 — v27.6 BMM 벡터화

        Boolean Matrix Multiplication: M.T @ M (BLAS 가속)
        기존 O(R×C²) Python 루프 → O(R×C) 행렬 구축 + O(C²×R) BLAS 연산
        수학적으로 동일한 결과, ~100x 빠름
        """
        total_rows = len(sample_data)
        n_cols = len(col_names)

        if total_rows == 0 or n_cols == 0:
            return {}

        # Boolean non-null 행렬 구축 (pandas 활용)
        try:
            import pandas as pd
            df = pd.DataFrame(sample_data)
            M = df.reindex(columns=col_names).notnull().values.astype(np.float32)
        except Exception:
            # Fallback: numpy 직접 구축
            M = np.zeros((total_rows, n_cols), dtype=np.float32)
            col_index = {name: i for i, name in enumerate(col_names)}
            for r, row in enumerate(sample_data):
                for col_name in col_names:
                    if col_name in row and row[col_name] is not None:
                        M[r, col_index[col_name]] = 1.0

        # BLAS 가속 행렬 곱: coex[i,j] = 동시 non-null 비율
        coex_matrix = (M.T @ M) / total_rows

        # 상삼각 행렬에서 유의미한 항목 추출
        mask = np.triu(np.ones((n_cols, n_cols), dtype=bool), k=1)
        significant = mask & (coex_matrix > 0)
        rows, cols = np.where(significant)

        return {(int(r), int(c)): float(coex_matrix[r, c]) for r, c in zip(rows, cols)}

    def _compute_betti_numbers(
        self,
        edges: List[Tuple[int, int, float]],
        num_vertices: int,
    ) -> BettiNumbers:
        """
        Betti numbers 계산

        β₀: Connected components (Union-Find)
        β₁: 1차원 구멍 (사이클 수)
        β₂: 2차원 구멍 (clique 기반 추정)
        """
        if num_vertices == 0:
            return BettiNumbers(0, 0, 0)

        # Union-Find for connected components
        parent = list(range(num_vertices))
        rank = [0] * num_vertices

        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]

        def union(x, y):
            px, py = find(x), find(y)
            if px == py:
                return False
            if rank[px] < rank[py]:
                px, py = py, px
            parent[py] = px
            if rank[px] == rank[py]:
                rank[px] += 1
            return True

        # 에지 추가하며 컴포넌트 계산
        edge_count = 0
        cycle_count = 0

        for u, v, w in edges:
            if u < num_vertices and v < num_vertices:
                if not union(u, v):
                    # 이미 연결됨 = 사이클 발견
                    cycle_count += 1
                else:
                    edge_count += 1

        # β₀: 연결된 컴포넌트 수
        components = len(set(find(i) for i in range(num_vertices)))
        beta_0 = components

        # β₁: 사이클 수 (Euler formula: V - E + F = 2 - 2g for genus g)
        # 단순화: β₁ ≈ E - V + β₀
        beta_1 = max(0, edge_count - num_vertices + beta_0)

        # β₂: 3-clique 기반 추정 (고차원 구멍)
        adjacency_set = set()
        for u, v, w in edges:
            if u < num_vertices and v < num_vertices:
                adjacency_set.add((min(u, v), max(u, v)))

        triangles = self._count_triangles(adjacency_set, num_vertices)
        # β₂는 일반적으로 스키마에서는 0에 가까움
        beta_2 = max(0, triangles // 4)  # 휴리스틱

        return BettiNumbers(beta_0, beta_1, beta_2)

    def _count_triangles(
        self,
        edges: Set[Tuple[int, int]],
        num_vertices: int,
    ) -> int:
        """삼각형(3-clique) 개수"""
        count = 0
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)

        for u in range(num_vertices):
            for v in adj[u]:
                if v > u:
                    # u-v 에지의 공통 이웃
                    common = adj[u] & adj[v]
                    for w in common:
                        if w > v:
                            count += 1

        return count

    def _compute_persistence(
        self,
        edges: List[Tuple[int, int, float]],
        columns: List[Dict[str, Any]],
    ) -> List[PersistencePair]:
        """
        Persistence diagram 계산

        Filtration: 에지 가중치 기반
        """
        pairs = []

        # 에지를 가중치로 정렬 (오름차순)
        sorted_edges = sorted(edges, key=lambda x: x[2])

        num_vertices = len(columns)
        parent = list(range(num_vertices))
        birth_time = {i: 0.0 for i in range(num_vertices)}

        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]

        for u, v, w in sorted_edges:
            if u >= num_vertices or v >= num_vertices:
                continue

            pu, pv = find(u), find(v)
            if pu != pv:
                # 컴포넌트 병합 = 0차원 feature death
                older = pu if birth_time[pu] <= birth_time[pv] else pv
                younger = pv if older == pu else pu

                pairs.append(PersistencePair(
                    dimension=0,
                    birth=birth_time[younger],
                    death=w,
                ))

                parent[younger] = older

        # 영구적인 0차원 features (connected components)
        components = set(find(i) for i in range(num_vertices))
        for comp in components:
            pairs.append(PersistencePair(
                dimension=0,
                birth=birth_time[comp],
                death=float('inf'),
            ))

        # 1차원 persistence (사이클)
        # 사이클이 생기는 시점을 birth로, 더 큰 가중치의 에지로 닫히는 시점을 death로
        for u, v, w in sorted_edges:
            if u >= num_vertices or v >= num_vertices:
                continue
            pu, pv = find(u), find(v)
            if pu == pv:
                # 사이클 형성
                pairs.append(PersistencePair(
                    dimension=1,
                    birth=w,
                    death=w + 0.5,  # 휴리스틱
                ))

        return pairs

    def _identify_key_columns(self, columns: List[Any]) -> List[str]:
        """Key 컬럼 식별"""
        keys = []

        # Handle string format
        if columns and isinstance(columns[0], str):
            for name in columns:
                if name.lower().endswith("_id") or name.lower() == "id":
                    keys.append(name)
            return keys

        # Handle dict format
        for col in columns:
            name = col.get("name", "")
            is_key = col.get("is_key", False)
            unique_ratio = col.get("unique_ratio", 0)

            if is_key:
                keys.append(name)
            elif unique_ratio > 0.95:
                keys.append(name)
            elif name.lower().endswith("_id") or name.lower() == "id":
                keys.append(name)

        return keys

    def _assess_complexity(
        self,
        betti: BettiNumbers,
        num_columns: int,
        num_edges: int,
    ) -> str:
        """구조 복잡도 평가"""
        # 복잡도 점수 계산
        score = 0

        # 컬럼 수 기반
        if num_columns > 20:
            score += 2
        elif num_columns > 10:
            score += 1

        # 사이클 (β₁) 기반
        if betti.beta_1 > 5:
            score += 2
        elif betti.beta_1 > 2:
            score += 1

        # 연결성 기반
        edge_density = num_edges / max(1, num_columns * (num_columns - 1) / 2)
        if edge_density > 0.5:
            score += 2
        elif edge_density > 0.2:
            score += 1

        if score >= 4:
            return "high"
        elif score >= 2:
            return "medium"
        else:
            return "low"

    def compute_homeomorphism_score(
        self,
        sig_a: TDASignature,
        sig_b: TDASignature,
    ) -> Dict[str, Any]:
        """
        두 테이블 간 위상동형 점수 계산

        Returns:
            {
                "is_homeomorphic": bool,
                "structural_score": float,
                "betti_distance": float,
                "persistence_similarity": float,
                "confidence": float,
            }
        """
        # 1. Betti distance
        betti_dist = sig_a.betti_numbers.distance(sig_b.betti_numbers)

        # 정규화 (0-1)
        max_betti = max(
            sig_a.betti_numbers.beta_0 + sig_a.betti_numbers.beta_1,
            sig_b.betti_numbers.beta_0 + sig_b.betti_numbers.beta_1,
            1
        )
        normalized_dist = min(1.0, betti_dist / max_betti)
        structural_score = 1.0 - normalized_dist

        # 2. Persistence similarity
        pers_a = sig_a.total_persistence
        pers_b = sig_b.total_persistence
        if pers_a + pers_b > 0:
            persistence_sim = 1 - abs(pers_a - pers_b) / (pers_a + pers_b)
        else:
            persistence_sim = 1.0

        # 3. 컬럼 수 유사성
        col_ratio = min(sig_a.column_count, sig_b.column_count) / max(sig_a.column_count, sig_b.column_count, 1)

        # 4. 복잡도 매칭
        complexity_match = 1.0 if sig_a.structural_complexity == sig_b.structural_complexity else 0.5

        # 종합 점수
        confidence = (
            structural_score * 0.4 +
            persistence_sim * 0.2 +
            col_ratio * 0.2 +
            complexity_match * 0.2
        )

        is_homeomorphic = bool(confidence > 0.6 and betti_dist < 3)

        return {
            "is_homeomorphic": is_homeomorphic,
            "structural_score": float(round(structural_score, 4)),
            "betti_distance": float(round(betti_dist, 4)),
            "persistence_similarity": float(round(persistence_sim, 4)),
            "column_ratio": float(round(col_ratio, 4)),
            "confidence": float(round(confidence, 4)),
        }
