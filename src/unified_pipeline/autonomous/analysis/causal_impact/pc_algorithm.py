"""
PC Algorithm for Causal Graph Discovery

관찰 데이터로부터 DAG (Directed Acyclic Graph) 구조를 학습합니다.

References:
- Spirtes, Glymour, Scheines, "Causation, Prediction, and Search", 2000
- Colombo et al., "Learning High-Dimensional DAGs with Latent and Selection Variables", 2012
"""

from typing import Dict, List, Optional, Set, Tuple
import numpy as np

from .models import (
    PCAlgorithmResult,
    CausalGraph,
    CausalNode,
    CausalEdge,
)


class PCAlgorithm:
    """
    PC (Peter-Clark) Algorithm for Causal Discovery

    관찰 데이터로부터 DAG (Directed Acyclic Graph) 구조를 학습합니다.

    알고리즘:
    1. 완전 연결 무방향 그래프로 시작
    2. 조건부 독립성 테스트로 엣지 제거 (skeleton 학습)
    3. V-structure (충돌자) 식별로 방향 부여
    4. 추가 규칙으로 방향 전파

    References:
    - Spirtes, Glymour, Scheines, "Causation, Prediction, and Search", 2000
    - Colombo et al., "Learning High-Dimensional DAGs with Latent and Selection Variables", 2012
    """

    def __init__(
        self,
        alpha: float = 0.05,
        max_cond_set_size: Optional[int] = None,
        method: str = "partial_correlation"
    ):
        """
        Parameters:
        - alpha: 조건부 독립성 테스트 유의수준
        - max_cond_set_size: 최대 조건부 집합 크기 (None이면 변수 수 - 2)
        - method: 독립성 테스트 방법 ("partial_correlation", "fisher_z")
        """
        self.alpha = alpha
        self.max_cond_set_size = max_cond_set_size
        self.method = method
        self.n_tests = 0

    def fit(
        self,
        data: np.ndarray,
        variable_names: Optional[List[str]] = None
    ) -> PCAlgorithmResult:
        """
        PC Algorithm으로 인과 그래프 학습

        Parameters:
        - data: (n_samples, n_variables) 데이터 행렬
        - variable_names: 변수명 리스트

        Returns:
        - PCAlgorithmResult
        """
        n_samples, n_vars = data.shape

        if variable_names is None:
            variable_names = [f"V{i}" for i in range(n_vars)]

        if len(variable_names) != n_vars:
            raise ValueError("변수명 수가 데이터 열 수와 일치하지 않습니다")

        max_cond = self.max_cond_set_size
        if max_cond is None:
            max_cond = n_vars - 2

        self.n_tests = 0

        # Step 1: Skeleton 학습 (무방향 그래프)
        adjacency, sep_sets = self._learn_skeleton(data, n_vars, max_cond)

        # Skeleton을 엣지 집합으로
        skeleton = set()
        for i in range(n_vars):
            for j in range(i + 1, n_vars):
                if adjacency[i, j]:
                    skeleton.add((variable_names[i], variable_names[j]))

        # Step 2: V-structure 식별 및 방향 부여
        directed_edges = self._orient_edges(adjacency, sep_sets, variable_names)

        # Step 3: CausalGraph 객체 생성
        causal_graph = self._build_causal_graph(adjacency, directed_edges, variable_names)

        return PCAlgorithmResult(
            adjacency_matrix=adjacency.copy(),
            causal_graph=causal_graph,
            skeleton=skeleton,
            directed_edges=directed_edges,
            variable_names=variable_names,
            separation_sets=sep_sets,
            n_conditional_tests=self.n_tests,
            alpha=self.alpha
        )

    def _learn_skeleton(
        self,
        data: np.ndarray,
        n_vars: int,
        max_cond: int
    ) -> Tuple[np.ndarray, Dict[Tuple[int, int], Set[int]]]:
        """Skeleton 학습 (Phase 1)"""
        # 완전 연결 그래프로 시작
        adjacency = np.ones((n_vars, n_vars), dtype=bool)
        np.fill_diagonal(adjacency, False)

        sep_sets: Dict[Tuple[int, int], Set[int]] = {}

        # 조건부 집합 크기를 0부터 증가시키며 테스트
        for cond_size in range(max_cond + 1):
            # 인접한 쌍에 대해 테스트
            for i in range(n_vars):
                for j in range(i + 1, n_vars):
                    if not adjacency[i, j]:
                        continue

                    # i와 j를 제외한 인접 노드
                    adj_i = set(np.where(adjacency[i, :])[0]) - {j}
                    adj_j = set(np.where(adjacency[j, :])[0]) - {i}
                    possible_sep = adj_i | adj_j

                    if len(possible_sep) < cond_size:
                        continue

                    # 조건부 집합 후보 생성
                    found_sep = False
                    for cond_set in self._combinations(list(possible_sep), cond_size):
                        cond_set = set(cond_set)

                        # 조건부 독립성 테스트
                        is_independent = self._conditional_independence_test(
                            data, i, j, list(cond_set)
                        )
                        self.n_tests += 1

                        if is_independent:
                            # 엣지 제거
                            adjacency[i, j] = False
                            adjacency[j, i] = False
                            sep_sets[(i, j)] = cond_set
                            sep_sets[(j, i)] = cond_set
                            found_sep = True
                            break

                    if found_sep:
                        continue

        return adjacency, sep_sets

    def _orient_edges(
        self,
        adjacency: np.ndarray,
        sep_sets: Dict[Tuple[int, int], Set[int]],
        variable_names: List[str]
    ) -> Set[Tuple[str, str]]:
        """엣지 방향 결정 (Phase 2)"""
        n_vars = len(variable_names)
        directed = set()

        # 방향 행렬: directed_adj[i,j] = True 이면 i -> j
        directed_adj = np.zeros((n_vars, n_vars), dtype=bool)

        # V-structure 식별: i - k - j 에서 i와 j가 인접하지 않고
        # k가 sep_set(i,j)에 없으면 i -> k <- j
        for i in range(n_vars):
            for j in range(i + 1, n_vars):
                if adjacency[i, j]:
                    continue  # i와 j가 인접하면 skip

                # 공통 이웃 k 찾기
                for k in range(n_vars):
                    if k == i or k == j:
                        continue
                    if not (adjacency[i, k] and adjacency[k, j]):
                        continue

                    # k가 separation set에 없으면 V-structure
                    sep = sep_sets.get((i, j), set())
                    if k not in sep:
                        directed_adj[i, k] = True
                        directed_adj[j, k] = True
                        directed.add((variable_names[i], variable_names[k]))
                        directed.add((variable_names[j], variable_names[k]))

        # 추가 규칙으로 방향 전파
        changed = True
        while changed:
            changed = False
            for i in range(n_vars):
                for j in range(n_vars):
                    if i == j:
                        continue
                    if not adjacency[i, j]:
                        continue
                    if directed_adj[i, j] or directed_adj[j, i]:
                        continue  # 이미 방향 결정됨

                    # 규칙 1: i -> k - j 이고 i와 j가 인접하지 않으면 k -> j
                    for k in range(n_vars):
                        if k == i or k == j:
                            continue
                        if directed_adj[i, k] and adjacency[k, j] and not adjacency[i, j]:
                            if not directed_adj[j, k]:  # 사이클 방지
                                directed_adj[k, j] = True
                                directed.add((variable_names[k], variable_names[j]))
                                changed = True

                    # 규칙 2: 사이클 방지
                    # i -> ... -> j 경로가 있으면 j -> i 불가

        # 방향 미결정 엣지는 임의로 (또는 그대로 무방향으로) 처리
        for i in range(n_vars):
            for j in range(i + 1, n_vars):
                if adjacency[i, j] and not directed_adj[i, j] and not directed_adj[j, i]:
                    # 아직 방향 미결정: 임의로 i -> j
                    directed_adj[i, j] = True
                    directed.add((variable_names[i], variable_names[j]))

        return directed

    def _conditional_independence_test(
        self,
        data: np.ndarray,
        i: int,
        j: int,
        cond_set: List[int]
    ) -> bool:
        """조건부 독립성 테스트 (Partial Correlation 기반)"""
        n_samples = data.shape[0]

        if len(cond_set) == 0:
            # 무조건부 상관
            corr = np.corrcoef(data[:, i], data[:, j])[0, 1]
        else:
            # 조건부 편상관 (Partial Correlation)
            corr = self._partial_correlation(data, i, j, cond_set)

        if np.isnan(corr):
            return False

        # Fisher's Z 변환
        z = 0.5 * np.log((1 + corr + 1e-10) / (1 - corr + 1e-10))

        # 표준오차
        df = n_samples - len(cond_set) - 3
        if df <= 0:
            return False

        se = 1.0 / np.sqrt(df)

        # 양측 검정
        z_stat = abs(z) / se
        p_value = 2 * (1 - self._normal_cdf(z_stat))

        return p_value > self.alpha

    def _partial_correlation(
        self,
        data: np.ndarray,
        i: int,
        j: int,
        cond_set: List[int]
    ) -> float:
        """편상관 계수 계산"""
        # 조건부 변수로 회귀한 잔차의 상관
        if len(cond_set) == 0:
            return np.corrcoef(data[:, i], data[:, j])[0, 1]

        Z = data[:, cond_set]
        Z = np.column_stack([np.ones(len(Z)), Z])

        try:
            # X_i에서 Z의 효과 제거
            beta_i = np.linalg.lstsq(Z, data[:, i], rcond=None)[0]
            residual_i = data[:, i] - Z @ beta_i

            # X_j에서 Z의 효과 제거
            beta_j = np.linalg.lstsq(Z, data[:, j], rcond=None)[0]
            residual_j = data[:, j] - Z @ beta_j

            # 잔차 간 상관
            corr = np.corrcoef(residual_i, residual_j)[0, 1]
            return corr
        except np.linalg.LinAlgError:
            return 0.0

    def _combinations(self, items: List[int], size: int) -> List[Tuple[int, ...]]:
        """조합 생성"""
        if size == 0:
            return [()]
        if size > len(items):
            return []

        result = []
        for i, item in enumerate(items):
            for rest in self._combinations(items[i + 1:], size - 1):
                result.append((item,) + rest)
        return result

    def _normal_cdf(self, x: float) -> float:
        """표준 정규 분포 CDF 근사"""
        return 0.5 * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x ** 3)))

    def _build_causal_graph(
        self,
        adjacency: np.ndarray,
        directed_edges: Set[Tuple[str, str]],
        variable_names: List[str]
    ) -> CausalGraph:
        """CausalGraph 객체 생성"""
        graph = CausalGraph()

        # 노드 추가
        for name in variable_names:
            graph.add_node(CausalNode(
                name=name,
                node_type="observed",
                observed=True
            ))

        # 방향 엣지 추가
        for source, target in directed_edges:
            graph.add_edge(CausalEdge(
                source=source,
                target=target,
                weight=1.0,
                mechanism="discovered"
            ))

        return graph
