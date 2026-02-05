"""
Join Path Resolver (v17.0)

다중 테이블 쿼리에서 최적의 조인 경로 추론:
- BFS 기반 최단 경로 탐색
- Bridge Table 경유 간접 조인 지원
- LLM 기반 복잡한 조인 추론
"""

import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Set, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from .schema_context import SchemaContext, ForeignKeyMeta

logger = logging.getLogger(__name__)


@dataclass
class JoinPath:
    """조인 경로"""
    tables: List[str]  # 조인 순서대로의 테이블 목록
    conditions: List[str]  # JOIN 조건들 (["a.col = b.col", ...])
    join_types: List[str]  # JOIN 타입들 (["INNER", "LEFT", ...])

    # 메타데이터
    is_direct: bool = True  # 직접 FK 관계인지
    path_length: int = 0
    confidence: float = 1.0

    def to_sql_fragment(self) -> str:
        """SQL JOIN 절 생성"""
        if len(self.tables) <= 1:
            return self.tables[0] if self.tables else ""

        parts = [self.tables[0]]
        for i, (table, condition, join_type) in enumerate(
            zip(self.tables[1:], self.conditions, self.join_types)
        ):
            parts.append(f"{join_type} JOIN {table} ON {condition}")

        return "\n".join(parts)


class JoinPathResolver:
    """
    조인 경로 해결기

    FK 그래프를 탐색하여 테이블 간 최적 조인 경로를 찾습니다.
    """

    def __init__(self, schema_context: "SchemaContext"):
        """
        Args:
            schema_context: 스키마 컨텍스트
        """
        self.schema = schema_context

        # FK 그래프 구축 (양방향)
        self._graph: Dict[str, Dict[str, "ForeignKeyMeta"]] = {}
        self._build_graph()

    def _build_graph(self) -> None:
        """FK 관계를 그래프로 구축"""
        for table in self.schema.tables:
            self._graph[table] = {}

        for fk in self.schema.foreign_keys:
            # 양방향으로 추가 (조인은 양방향 가능)
            if fk.from_table not in self._graph:
                self._graph[fk.from_table] = {}
            if fk.to_table not in self._graph:
                self._graph[fk.to_table] = {}

            self._graph[fk.from_table][fk.to_table] = fk
            # 역방향도 추가 (같은 FK를 참조)
            self._graph[fk.to_table][fk.from_table] = fk

        logger.debug(f"Join graph built with {len(self._graph)} nodes")

    def find_path(
        self,
        source: str,
        target: str,
        max_hops: int = 4
    ) -> Optional[JoinPath]:
        """
        두 테이블 간 최단 조인 경로 탐색 (BFS)

        Args:
            source: 시작 테이블
            target: 목표 테이블
            max_hops: 최대 홉 수 (경유 테이블 수)

        Returns:
            JoinPath 또는 None
        """
        if source not in self._graph or target not in self._graph:
            logger.warning(f"Table not found in graph: {source} or {target}")
            return None

        if source == target:
            return JoinPath(tables=[source], conditions=[], join_types=[], path_length=0)

        # BFS
        queue: deque = deque([(source, [source], [])])  # (current, path, fks)
        visited: Set[str] = {source}

        while queue:
            current, path, fks = queue.popleft()

            if len(path) > max_hops + 1:
                continue

            for neighbor, fk in self._graph.get(current, {}).items():
                if neighbor == target:
                    # 경로 발견
                    full_path = path + [neighbor]
                    full_fks = fks + [fk]
                    return self._build_join_path(full_path, full_fks)

                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor], fks + [fk]))

        logger.info(f"No join path found between {source} and {target}")
        return None

    def _build_join_path(
        self,
        tables: List[str],
        fks: List["ForeignKeyMeta"]
    ) -> JoinPath:
        """탐색 결과로 JoinPath 객체 생성"""
        conditions = []
        join_types = []

        for i, (fk, next_table) in enumerate(zip(fks, tables[1:])):
            # 조인 조건 생성
            condition = f"{fk.from_table}.{fk.from_column} = {fk.to_table}.{fk.to_column}"
            conditions.append(condition)

            # 조인 타입 결정 (기본 INNER, N:1 관계면 LEFT 고려)
            join_types.append("INNER")

        # 신뢰도 계산 (FK 신뢰도의 평균)
        confidence = sum(fk.confidence for fk in fks) / len(fks) if fks else 1.0

        return JoinPath(
            tables=tables,
            conditions=conditions,
            join_types=join_types,
            is_direct=len(tables) == 2,
            path_length=len(tables) - 1,
            confidence=confidence,
        )

    def find_multi_table_path(
        self,
        tables: List[str],
        base_table: Optional[str] = None
    ) -> Optional[JoinPath]:
        """
        여러 테이블을 연결하는 조인 경로 탐색

        Args:
            tables: 조인할 테이블 목록
            base_table: 기준 테이블 (없으면 첫 번째 테이블)

        Returns:
            전체 조인을 커버하는 JoinPath
        """
        if len(tables) <= 1:
            return JoinPath(
                tables=tables,
                conditions=[],
                join_types=[],
                path_length=0
            )

        # 기준 테이블 결정
        if base_table and base_table in tables:
            start = base_table
            remaining = [t for t in tables if t != base_table]
        else:
            start = tables[0]
            remaining = tables[1:]

        # 각 테이블로의 경로 찾기 (Greedy)
        combined_tables = [start]
        combined_conditions = []
        combined_join_types = []
        visited = {start}

        for target in remaining:
            # 이미 포함된 테이블에서 가장 가까운 경로 찾기
            best_path = None
            best_length = float('inf')

            for source in combined_tables:
                if source == target:
                    continue
                path = self.find_path(source, target)
                if path and path.path_length < best_length:
                    best_path = path
                    best_length = path.path_length

            if best_path:
                # 경로 병합 (중복 테이블 제외)
                for i, table in enumerate(best_path.tables[1:]):
                    if table not in visited:
                        combined_tables.append(table)
                        combined_conditions.append(best_path.conditions[i])
                        combined_join_types.append(best_path.join_types[i])
                        visited.add(table)
            else:
                logger.warning(f"Cannot find path to {target}")
                return None

        return JoinPath(
            tables=combined_tables,
            conditions=combined_conditions,
            join_types=combined_join_types,
            is_direct=False,
            path_length=len(combined_conditions),
            confidence=0.8,  # 복합 경로는 신뢰도 감소
        )

    def get_all_paths(
        self,
        source: str,
        target: str,
        max_hops: int = 3
    ) -> List[JoinPath]:
        """
        가능한 모든 조인 경로 탐색 (DFS)

        여러 경로가 있을 때 선택 옵션 제공
        """
        if source not in self._graph or target not in self._graph:
            return []

        if source == target:
            return [JoinPath(tables=[source], conditions=[], join_types=[], path_length=0)]

        all_paths = []
        self._dfs_paths(source, target, [source], [], all_paths, set(), max_hops)

        # 길이순 정렬
        all_paths.sort(key=lambda p: p.path_length)

        return all_paths

    def _dfs_paths(
        self,
        current: str,
        target: str,
        path: List[str],
        fks: List["ForeignKeyMeta"],
        results: List[JoinPath],
        visited: Set[str],
        max_hops: int
    ) -> None:
        """DFS 헬퍼"""
        if len(path) > max_hops + 1:
            return

        visited.add(current)

        for neighbor, fk in self._graph.get(current, {}).items():
            if neighbor == target:
                results.append(self._build_join_path(path + [neighbor], fks + [fk]))
            elif neighbor not in visited:
                self._dfs_paths(
                    neighbor, target,
                    path + [neighbor],
                    fks + [fk],
                    results,
                    visited.copy(),
                    max_hops
                )

    def suggest_join_type(
        self,
        from_table: str,
        to_table: str,
        fk: "ForeignKeyMeta"
    ) -> str:
        """
        조인 타입 추천

        - INNER: 양쪽 모두 데이터가 있어야 할 때
        - LEFT: 왼쪽 테이블의 모든 행을 유지할 때
        - RIGHT: 오른쪽 테이블의 모든 행을 유지할 때
        """
        # FK 방향 기준으로 결정
        # FK를 가진 테이블(from)이 기준이면 LEFT가 안전
        if fk.from_table == from_table:
            # from_table이 FK 소유 → to_table은 참조 대상
            # 참조 대상이 없을 수도 있으므로 LEFT
            return "LEFT"
        else:
            # to_table이 FK 소유
            return "INNER"

    def to_json(self) -> str:
        """조인 그래프를 JSON으로 직렬화"""
        import json

        graph_data = {}
        for table, neighbors in self._graph.items():
            graph_data[table] = {
                n: {
                    "from": fk.from_table,
                    "from_col": fk.from_column,
                    "to": fk.to_table,
                    "to_col": fk.to_column,
                }
                for n, fk in neighbors.items()
            }

        return json.dumps(graph_data, indent=2, ensure_ascii=False)
