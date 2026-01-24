"""
Advanced TDA (Topological Data Analysis) Module

Ripser 기반 실제 Persistent Homology 계산
- Vietoris-Rips Complex
- Persistence Diagrams
- Wasserstein/Bottleneck Distance
- Schema Similarity via TDA

References:
- https://github.com/Ripser/ripser
- https://github.com/scikit-tda/persim
- Paper: "Topological Data Analysis and TDL Beyond Persistent Homology" (2024)
"""

import logging
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple, Union
from collections import defaultdict

logger = logging.getLogger(__name__)

# Optional imports with fallbacks
try:
    import ripser
    RIPSER_AVAILABLE = True
except ImportError:
    RIPSER_AVAILABLE = False
    logger.warning("ripser not available. Install with: pip install ripser")

try:
    import persim
    PERSIM_AVAILABLE = True
except ImportError:
    PERSIM_AVAILABLE = False
    logger.warning("persim not available. Install with: pip install persim")

try:
    from scipy.spatial.distance import pdist, squareform
    from scipy.cluster.hierarchy import linkage, fcluster
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


@dataclass
class PersistenceDiagram:
    """Persistence Diagram 결과"""
    dimension: int
    pairs: List[Tuple[float, float]]  # (birth, death) pairs

    @property
    def lifetimes(self) -> List[float]:
        """각 feature의 수명 (death - birth)"""
        return [d - b for b, d in self.pairs if d != np.inf]

    @property
    def total_persistence(self) -> float:
        """총 persistence (inf 제외)"""
        return sum(self.lifetimes)

    @property
    def max_persistence(self) -> float:
        """최대 persistence"""
        lifetimes = self.lifetimes
        return max(lifetimes) if lifetimes else 0.0

    @property
    def num_features(self) -> int:
        """feature 개수 (inf 포함)"""
        return len(self.pairs)

    @property
    def num_essential(self) -> int:
        """essential features (death=inf)"""
        return sum(1 for _, d in self.pairs if d == np.inf)

    def to_array(self) -> np.ndarray:
        """numpy array로 변환"""
        if not self.pairs:
            return np.array([]).reshape(0, 2)
        return np.array(self.pairs)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dimension": self.dimension,
            "pairs": self.pairs,
            "total_persistence": self.total_persistence,
            "max_persistence": self.max_persistence,
            "num_features": self.num_features,
            "num_essential": self.num_essential,
        }


@dataclass
class AdvancedTDASignature:
    """고도화된 TDA 서명"""
    table_name: str
    diagrams: List[PersistenceDiagram]  # H0, H1, H2
    betti_numbers: Tuple[int, int, int]  # β0, β1, β2

    # 추가 메트릭
    total_persistence_h0: float = 0.0
    total_persistence_h1: float = 0.0
    euler_characteristic: int = 0
    structural_complexity: str = "low"

    # Point cloud info
    num_points: int = 0
    embedding_dim: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "table_name": self.table_name,
            "betti_numbers": {
                "beta_0": self.betti_numbers[0],
                "beta_1": self.betti_numbers[1],
                "beta_2": self.betti_numbers[2],
            },
            "euler_characteristic": self.euler_characteristic,
            "total_persistence_h0": self.total_persistence_h0,
            "total_persistence_h1": self.total_persistence_h1,
            "structural_complexity": self.structural_complexity,
            "num_points": self.num_points,
            "embedding_dim": self.embedding_dim,
            "diagrams": [d.to_dict() for d in self.diagrams],
        }


@dataclass
class TDADistance:
    """두 TDA 서명 간 거리"""
    table_a: str
    table_b: str
    wasserstein_h0: float = 0.0
    wasserstein_h1: float = 0.0
    bottleneck_h0: float = 0.0
    bottleneck_h1: float = 0.0
    betti_distance: float = 0.0
    overall_similarity: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "table_a": self.table_a,
            "table_b": self.table_b,
            "wasserstein_h0": self.wasserstein_h0,
            "wasserstein_h1": self.wasserstein_h1,
            "bottleneck_h0": self.bottleneck_h0,
            "bottleneck_h1": self.bottleneck_h1,
            "betti_distance": self.betti_distance,
            "overall_similarity": self.overall_similarity,
        }


class AdvancedTDAAnalyzer:
    """
    고도화된 TDA 분석기

    Ripser를 사용한 실제 Persistent Homology 계산
    """

    def __init__(self, max_dimension: int = 2, max_edge_length: float = np.inf):
        """
        Args:
            max_dimension: 계산할 최대 homology 차원 (기본 2 = H0, H1, H2)
            max_edge_length: Rips complex 최대 edge 길이
        """
        self.max_dimension = max_dimension
        self.max_edge_length = max_edge_length
        self.signatures: Dict[str, AdvancedTDASignature] = {}

    def schema_to_point_cloud(
        self,
        columns: List[Dict[str, Any]],
        sample_data: Optional[List[Dict[str, Any]]] = None,
    ) -> np.ndarray:
        """
        스키마를 point cloud로 변환

        각 컬럼을 고차원 공간의 점으로 매핑:
        - 타입 정보 (one-hot)
        - 통계적 특성 (null_ratio, unique_ratio, ...)
        - 이름 유사성 (character n-gram embedding)
        """
        if not columns:
            return np.array([]).reshape(0, 0)

        points = []

        # 타입 매핑
        type_map = {
            "string": [1, 0, 0, 0, 0, 0],
            "integer": [0, 1, 0, 0, 0, 0],
            "float": [0, 0, 1, 0, 0, 0],
            "datetime": [0, 0, 0, 1, 0, 0],
            "boolean": [0, 0, 0, 0, 1, 0],
            "unknown": [0, 0, 0, 0, 0, 1],
        }

        for col in columns:
            # 컬럼 정보가 문자열인 경우 처리
            if isinstance(col, str):
                col = {"name": col, "dtype": "unknown"}

            features = []

            # 1. 타입 one-hot (6차원)
            dtype = self._normalize_type(col.get("dtype", "unknown"))
            features.extend(type_map.get(dtype, type_map["unknown"]))

            # 2. 통계적 특성 (4차원)
            features.append(col.get("null_ratio", 0.0))
            features.append(col.get("unique_ratio", 0.5))
            features.append(1.0 if col.get("is_key", False) else 0.0)
            features.append(1.0 if col.get("is_fk", False) else 0.0)

            # 3. 이름 기반 특성 (4차원)
            name = col.get("name", "")
            features.append(1.0 if name.lower().endswith("_id") else 0.0)
            features.append(1.0 if "date" in name.lower() or "time" in name.lower() else 0.0)
            features.append(len(name) / 50.0)  # 정규화된 이름 길이
            features.append(name.count("_") / 5.0)  # 언더스코어 개수

            points.append(features)

        return np.array(points)

    def _normalize_type(self, dtype: str) -> str:
        """타입 정규화"""
        dtype_lower = str(dtype).lower()

        if any(t in dtype_lower for t in ["int", "bigint", "smallint"]):
            return "integer"
        elif any(t in dtype_lower for t in ["float", "double", "decimal", "numeric"]):
            return "float"
        elif any(t in dtype_lower for t in ["varchar", "char", "text", "string", "object"]):
            return "string"
        elif any(t in dtype_lower for t in ["date", "time", "timestamp"]):
            return "datetime"
        elif any(t in dtype_lower for t in ["bool", "boolean"]):
            return "boolean"
        else:
            return "unknown"

    def compute_persistence(
        self,
        point_cloud: np.ndarray,
    ) -> List[PersistenceDiagram]:
        """
        Point cloud에서 Persistence Diagram 계산

        Args:
            point_cloud: (n_points, n_features) 배열

        Returns:
            각 차원의 Persistence Diagram 리스트
        """
        if point_cloud.size == 0:
            return [PersistenceDiagram(dim, []) for dim in range(self.max_dimension + 1)]

        # Ripser 사용 가능한 경우 실제 계산
        if RIPSER_AVAILABLE and point_cloud.shape[0] >= 2:
            try:
                result = ripser.ripser(
                    point_cloud,
                    maxdim=min(self.max_dimension, point_cloud.shape[0] - 1),
                    thresh=self.max_edge_length,
                )

                diagrams = []
                for dim, dgm in enumerate(result['dgms']):
                    pairs = [(float(b), float(d)) for b, d in dgm]
                    diagrams.append(PersistenceDiagram(dim, pairs))

                # 부족한 차원 채우기
                while len(diagrams) <= self.max_dimension:
                    diagrams.append(PersistenceDiagram(len(diagrams), []))

                return diagrams

            except Exception as e:
                logger.warning(f"Ripser computation failed: {e}, using fallback")

        # Fallback: 단순 근사
        return self._compute_persistence_fallback(point_cloud)

    def _compute_persistence_fallback(
        self,
        point_cloud: np.ndarray,
    ) -> List[PersistenceDiagram]:
        """Ripser 없을 때 fallback 계산"""
        diagrams = []
        n_points = point_cloud.shape[0]

        if n_points == 0:
            for dim in range(self.max_dimension + 1):
                diagrams.append(PersistenceDiagram(dim, []))
            return diagrams

        # H0: Connected components (Union-Find로 근사)
        if SCIPY_AVAILABLE and n_points >= 2:
            distances = pdist(point_cloud)
            Z = linkage(distances, method='single')

            # Birth-death pairs 추출
            h0_pairs = []
            merge_distances = Z[:, 2]
            for i, d in enumerate(merge_distances):
                h0_pairs.append((0.0, float(d)))
            # Essential feature (마지막 component)
            h0_pairs.append((0.0, np.inf))
            diagrams.append(PersistenceDiagram(0, h0_pairs))
        else:
            # 매우 단순한 근사
            diagrams.append(PersistenceDiagram(0, [(0.0, np.inf)]))

        # H1, H2: 빈 다이어그램 (정확한 계산은 Ripser 필요)
        for dim in range(1, self.max_dimension + 1):
            diagrams.append(PersistenceDiagram(dim, []))

        return diagrams

    def compute_signature(
        self,
        table_name: str,
        columns: List[Any],
        sample_data: Optional[List[Dict[str, Any]]] = None,
    ) -> AdvancedTDASignature:
        """
        테이블의 TDA 서명 계산
        """
        # Schema를 point cloud로 변환
        point_cloud = self.schema_to_point_cloud(columns, sample_data)

        # Persistence diagram 계산
        diagrams = self.compute_persistence(point_cloud)

        # Betti numbers 추출 (essential features 수)
        betti = tuple(
            dgm.num_essential if dgm else 0
            for dgm in diagrams[:3]
        )
        # 3개 미만이면 채우기
        while len(betti) < 3:
            betti = betti + (0,)

        # 복잡도 평가
        total_h0 = diagrams[0].total_persistence if diagrams else 0
        total_h1 = diagrams[1].total_persistence if len(diagrams) > 1 else 0

        if total_h1 > 1.0 or betti[1] > 2:
            complexity = "high"
        elif total_h1 > 0.5 or betti[1] > 0:
            complexity = "medium"
        else:
            complexity = "low"

        signature = AdvancedTDASignature(
            table_name=table_name,
            diagrams=diagrams,
            betti_numbers=betti,
            total_persistence_h0=total_h0,
            total_persistence_h1=total_h1,
            euler_characteristic=betti[0] - betti[1] + betti[2],
            structural_complexity=complexity,
            num_points=point_cloud.shape[0] if point_cloud.size > 0 else 0,
            embedding_dim=point_cloud.shape[1] if point_cloud.size > 0 else 0,
        )

        self.signatures[table_name] = signature
        return signature

    def compute_distance(
        self,
        sig_a: AdvancedTDASignature,
        sig_b: AdvancedTDASignature,
    ) -> TDADistance:
        """
        두 TDA 서명 간 거리 계산

        - Wasserstein distance: 최적 수송 기반
        - Bottleneck distance: 최대 이동 거리
        - Betti distance: Betti number 차이
        """
        result = TDADistance(
            table_a=sig_a.table_name,
            table_b=sig_b.table_name,
        )

        # Betti distance
        betti_diff = sum(
            abs(a - b)
            for a, b in zip(sig_a.betti_numbers, sig_b.betti_numbers)
        )
        result.betti_distance = float(betti_diff)

        # Persistence diagram distances
        if PERSIM_AVAILABLE:
            try:
                # H0 distances
                dgm_a_h0 = sig_a.diagrams[0].to_array() if sig_a.diagrams else np.array([]).reshape(0, 2)
                dgm_b_h0 = sig_b.diagrams[0].to_array() if sig_b.diagrams else np.array([]).reshape(0, 2)

                if dgm_a_h0.size > 0 and dgm_b_h0.size > 0:
                    # inf 값 처리
                    dgm_a_h0 = self._replace_inf(dgm_a_h0)
                    dgm_b_h0 = self._replace_inf(dgm_b_h0)

                    result.wasserstein_h0 = float(persim.wasserstein(dgm_a_h0, dgm_b_h0))
                    result.bottleneck_h0 = float(persim.bottleneck(dgm_a_h0, dgm_b_h0))

                # H1 distances
                if len(sig_a.diagrams) > 1 and len(sig_b.diagrams) > 1:
                    dgm_a_h1 = sig_a.diagrams[1].to_array()
                    dgm_b_h1 = sig_b.diagrams[1].to_array()

                    if dgm_a_h1.size > 0 and dgm_b_h1.size > 0:
                        dgm_a_h1 = self._replace_inf(dgm_a_h1)
                        dgm_b_h1 = self._replace_inf(dgm_b_h1)

                        result.wasserstein_h1 = float(persim.wasserstein(dgm_a_h1, dgm_b_h1))
                        result.bottleneck_h1 = float(persim.bottleneck(dgm_a_h1, dgm_b_h1))

            except Exception as e:
                logger.warning(f"Persim distance calculation failed: {e}")

        # Overall similarity (거리를 유사도로 변환)
        max_distance = max(
            result.wasserstein_h0 + result.wasserstein_h1,
            result.betti_distance,
            0.001
        )
        result.overall_similarity = 1.0 / (1.0 + max_distance)

        return result

    def _replace_inf(self, dgm: np.ndarray, replacement: float = 10.0) -> np.ndarray:
        """inf 값을 큰 유한값으로 대체"""
        dgm = dgm.copy()
        dgm[np.isinf(dgm)] = replacement
        return dgm

    def compute_pairwise_distances(
        self,
        signatures: List[AdvancedTDASignature],
    ) -> Dict[Tuple[str, str], TDADistance]:
        """모든 쌍에 대해 거리 계산"""
        distances = {}

        for i, sig_a in enumerate(signatures):
            for sig_b in signatures[i+1:]:
                dist = self.compute_distance(sig_a, sig_b)
                distances[(sig_a.table_name, sig_b.table_name)] = dist

        return distances

    def find_similar_schemas(
        self,
        threshold: float = 0.7,
    ) -> List[Tuple[str, str, float]]:
        """
        유사한 스키마 쌍 찾기

        Returns:
            (table_a, table_b, similarity) 튜플 리스트
        """
        signatures = list(self.signatures.values())
        distances = self.compute_pairwise_distances(signatures)

        similar_pairs = []
        for (table_a, table_b), dist in distances.items():
            if dist.overall_similarity >= threshold:
                similar_pairs.append((table_a, table_b, dist.overall_similarity))

        return sorted(similar_pairs, key=lambda x: -x[2])


# Convenience functions
def compute_schema_tda(
    table_name: str,
    columns: List[Any],
    sample_data: Optional[List[Dict]] = None,
) -> AdvancedTDASignature:
    """스키마의 TDA 서명 계산 (편의 함수)"""
    analyzer = AdvancedTDAAnalyzer()
    return analyzer.compute_signature(table_name, columns, sample_data)


def compute_schema_similarity_tda(
    columns_a: List[Any],
    columns_b: List[Any],
    table_a: str = "table_a",
    table_b: str = "table_b",
) -> float:
    """두 스키마 간 TDA 기반 유사도 (0~1)"""
    analyzer = AdvancedTDAAnalyzer()
    sig_a = analyzer.compute_signature(table_a, columns_a)
    sig_b = analyzer.compute_signature(table_b, columns_b)
    distance = analyzer.compute_distance(sig_a, sig_b)
    return distance.overall_similarity
