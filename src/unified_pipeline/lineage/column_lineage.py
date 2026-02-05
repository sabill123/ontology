"""
Column-level Lineage (v16.0)

세분화된 컬럼 수준 데이터 계보 추적:
- 컬럼 간 의존성 그래프
- 변환 로직 추적
- 영향도 분석
- FK 기반 자동 lineage
- 시각화 지원

SharedContext와 연동하여 FK 관계 기반 lineage 자동 생성

사용법:
    from .column_lineage import LineageTracker
    tracker = LineageTracker(shared_context)
    tracker.build_from_fks()
    tracker.get_upstream("table.column")
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Set, Tuple, TYPE_CHECKING
from enum import Enum
from collections import defaultdict, deque
import uuid
import logging

if TYPE_CHECKING:
    from ..shared_context import SharedContext

logger = logging.getLogger(__name__)


class LineageEdgeType(Enum):
    """Lineage Edge 유형"""
    DIRECT = "direct"               # 직접 복사
    TRANSFORM = "transform"         # 변환
    AGGREGATE = "aggregate"         # 집계
    FILTER = "filter"               # 필터링
    JOIN = "join"                   # 조인
    FK_REFERENCE = "fk_reference"   # FK 참조
    DERIVED = "derived"             # 파생
    INFERRED = "inferred"           # 추론


class ColumnType(Enum):
    """컬럼 유형"""
    SOURCE = "source"               # 원본 데이터
    INTERMEDIATE = "intermediate"   # 중간 단계
    TARGET = "target"               # 최종 결과
    DERIVED = "derived"             # 파생 컬럼


@dataclass
class ColumnNode:
    """Lineage 그래프의 컬럼 노드"""
    node_id: str
    table_name: str
    column_name: str
    column_type: ColumnType = ColumnType.SOURCE

    # 메타데이터
    data_type: Optional[str] = None
    description: Optional[str] = None
    is_pk: bool = False
    is_fk: bool = False
    fk_target: Optional[str] = None  # "table.column" 형식

    # 변환 정보
    transformation: Optional[str] = None
    business_logic: Optional[str] = None

    @property
    def full_name(self) -> str:
        return f"{self.table_name}.{self.column_name}"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_id": self.node_id,
            "table_name": self.table_name,
            "column_name": self.column_name,
            "full_name": self.full_name,
            "column_type": self.column_type.value,
            "data_type": self.data_type,
            "is_pk": self.is_pk,
            "is_fk": self.is_fk,
            "fk_target": self.fk_target,
            "transformation": self.transformation,
        }


@dataclass
class LineageEdge:
    """Lineage 그래프의 Edge"""
    edge_id: str
    source_node_id: str  # Upstream
    target_node_id: str  # Downstream
    edge_type: LineageEdgeType

    # 메타데이터
    confidence: float = 1.0
    transformation: Optional[str] = None
    sql_expression: Optional[str] = None
    description: Optional[str] = None

    # FK 관련
    fk_relationship: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "edge_id": self.edge_id,
            "source_node_id": self.source_node_id,
            "target_node_id": self.target_node_id,
            "edge_type": self.edge_type.value,
            "confidence": round(self.confidence, 4),
            "transformation": self.transformation,
            "fk_relationship": self.fk_relationship,
        }


@dataclass
class ImpactAnalysis:
    """영향도 분석 결과"""
    source_column: str
    upstream_columns: List[str]      # 이 컬럼에 영향을 주는 컬럼들
    downstream_columns: List[str]    # 이 컬럼이 영향을 주는 컬럼들
    direct_dependencies: int
    total_impact_count: int
    max_depth: int
    affected_tables: Set[str] = field(default_factory=set)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_column": self.source_column,
            "upstream_columns": self.upstream_columns,
            "downstream_columns": self.downstream_columns,
            "direct_dependencies": self.direct_dependencies,
            "total_impact_count": self.total_impact_count,
            "max_depth": self.max_depth,
            "affected_tables": list(self.affected_tables),
        }


class LineageTracker:
    """
    Column-level Lineage Tracker

    기능:
    1. 컬럼 노드 관리
    2. Lineage Edge 추적
    3. Upstream/Downstream 탐색
    4. 영향도 분석
    5. FK 기반 자동 lineage 생성
    6. 그래프 시각화 데이터 생성

    알고리즘:
    - BFS/DFS 기반 그래프 탐색
    - FK 관계에서 자동 edge 생성
    - 변환 로직 파싱 및 추적
    """

    def __init__(self, shared_context: Optional["SharedContext"] = None):
        self.shared_context = shared_context

        # 그래프 저장소
        self.nodes: Dict[str, ColumnNode] = {}  # node_id -> ColumnNode
        self.edges: Dict[str, LineageEdge] = {}  # edge_id -> LineageEdge

        # 인덱스
        self._name_to_node: Dict[str, str] = {}  # "table.column" -> node_id
        self._upstream_index: Dict[str, Set[str]] = defaultdict(set)  # node_id -> upstream node_ids
        self._downstream_index: Dict[str, Set[str]] = defaultdict(set)  # node_id -> downstream node_ids
        self._table_columns: Dict[str, Set[str]] = defaultdict(set)  # table -> node_ids

    def add_column(
        self,
        table_name: str,
        column_name: str,
        column_type: ColumnType = ColumnType.SOURCE,
        data_type: Optional[str] = None,
        is_pk: bool = False,
        is_fk: bool = False,
        fk_target: Optional[str] = None,
    ) -> ColumnNode:
        """컬럼 노드 추가"""
        full_name = f"{table_name}.{column_name}"

        # 이미 존재하면 반환
        if full_name in self._name_to_node:
            return self.nodes[self._name_to_node[full_name]]

        node_id = str(uuid.uuid4())
        node = ColumnNode(
            node_id=node_id,
            table_name=table_name,
            column_name=column_name,
            column_type=column_type,
            data_type=data_type,
            is_pk=is_pk,
            is_fk=is_fk,
            fk_target=fk_target,
        )

        self.nodes[node_id] = node
        self._name_to_node[full_name] = node_id
        self._table_columns[table_name].add(node_id)

        return node

    def add_edge(
        self,
        source: str,  # "table.column" 또는 node_id
        target: str,  # "table.column" 또는 node_id
        edge_type: LineageEdgeType = LineageEdgeType.DIRECT,
        confidence: float = 1.0,
        transformation: Optional[str] = None,
        fk_relationship: Optional[str] = None,
    ) -> Optional[LineageEdge]:
        """Lineage Edge 추가"""
        # 노드 ID 찾기
        source_id = self._resolve_node_id(source)
        target_id = self._resolve_node_id(target)

        if not source_id or not target_id:
            logger.warning(f"[LineageTracker] Cannot add edge: {source} -> {target}")
            return None

        # 이미 존재하는 edge 확인
        for edge in self.edges.values():
            if edge.source_node_id == source_id and edge.target_node_id == target_id:
                return edge

        edge_id = str(uuid.uuid4())
        edge = LineageEdge(
            edge_id=edge_id,
            source_node_id=source_id,
            target_node_id=target_id,
            edge_type=edge_type,
            confidence=confidence,
            transformation=transformation,
            fk_relationship=fk_relationship,
        )

        self.edges[edge_id] = edge
        self._upstream_index[target_id].add(source_id)
        self._downstream_index[source_id].add(target_id)

        return edge

    def _resolve_node_id(self, ref: str) -> Optional[str]:
        """참조를 node_id로 변환"""
        if ref in self.nodes:
            return ref
        if ref in self._name_to_node:
            return self._name_to_node[ref]
        return None

    def build_from_fks(self) -> int:
        """SharedContext의 FK 관계에서 lineage 자동 생성"""
        if not self.shared_context:
            logger.warning("[LineageTracker] No SharedContext available")
            return 0

        edge_count = 0

        # 테이블 컬럼 추가
        for table_name, table_info in self.shared_context.tables.items():
            columns = (
                table_info.columns if hasattr(table_info, 'columns')
                else table_info.get('columns', [])
            )
            primary_keys = (
                table_info.primary_keys if hasattr(table_info, 'primary_keys')
                else table_info.get('primary_keys', [])
            )

            for col in columns:
                col_name = col.get("name", "") if isinstance(col, dict) else col
                col_type = col.get("data_type", "") if isinstance(col, dict) else ""

                self.add_column(
                    table_name=table_name,
                    column_name=col_name,
                    data_type=col_type,
                    is_pk=col_name in primary_keys,
                )

        # FK 관계에서 Edge 생성
        for fk in self.shared_context.enhanced_fk_candidates:
            if isinstance(fk, dict):
                from_table = fk.get("from_table", "")
                from_col = fk.get("from_column", "")
                to_table = fk.get("to_table", "")
                to_col = fk.get("to_column", "")
                confidence = fk.get("fk_score", 0.5)

                if all([from_table, from_col, to_table, to_col]):
                    # FK 컬럼 업데이트
                    from_node = self.add_column(from_table, from_col, is_fk=True, fk_target=f"{to_table}.{to_col}")
                    to_node = self.add_column(to_table, to_col, is_pk=True)

                    # Edge 추가 (FK -> PK)
                    edge = self.add_edge(
                        source=f"{to_table}.{to_col}",
                        target=f"{from_table}.{from_col}",
                        edge_type=LineageEdgeType.FK_REFERENCE,
                        confidence=confidence,
                        fk_relationship=f"{from_table}.{from_col} -> {to_table}.{to_col}",
                    )

                    if edge:
                        edge_count += 1

        logger.info(f"[LineageTracker] Built lineage from FKs: {len(self.nodes)} nodes, {edge_count} edges")
        return edge_count

    def get_upstream(
        self,
        column_ref: str,
        max_depth: int = 10,
    ) -> List[ColumnNode]:
        """
        Upstream 컬럼 조회 (이 컬럼에 영향을 주는 컬럼들)

        Args:
            column_ref: "table.column" 또는 node_id
            max_depth: 최대 탐색 깊이

        Returns:
            Upstream ColumnNode 리스트
        """
        start_id = self._resolve_node_id(column_ref)
        if not start_id:
            return []

        visited = set()
        result = []
        queue = deque([(start_id, 0)])

        while queue:
            node_id, depth = queue.popleft()

            if node_id in visited or depth > max_depth:
                continue

            visited.add(node_id)

            if node_id != start_id:
                result.append(self.nodes[node_id])

            for upstream_id in self._upstream_index.get(node_id, set()):
                if upstream_id not in visited:
                    queue.append((upstream_id, depth + 1))

        return result

    def get_downstream(
        self,
        column_ref: str,
        max_depth: int = 10,
    ) -> List[ColumnNode]:
        """
        Downstream 컬럼 조회 (이 컬럼이 영향을 주는 컬럼들)

        Args:
            column_ref: "table.column" 또는 node_id
            max_depth: 최대 탐색 깊이

        Returns:
            Downstream ColumnNode 리스트
        """
        start_id = self._resolve_node_id(column_ref)
        if not start_id:
            return []

        visited = set()
        result = []
        queue = deque([(start_id, 0)])

        while queue:
            node_id, depth = queue.popleft()

            if node_id in visited or depth > max_depth:
                continue

            visited.add(node_id)

            if node_id != start_id:
                result.append(self.nodes[node_id])

            for downstream_id in self._downstream_index.get(node_id, set()):
                if downstream_id not in visited:
                    queue.append((downstream_id, depth + 1))

        return result

    def analyze_impact(
        self,
        column_ref: str,
    ) -> ImpactAnalysis:
        """
        영향도 분석

        Args:
            column_ref: 분석할 컬럼 ("table.column")

        Returns:
            ImpactAnalysis 결과
        """
        upstream = self.get_upstream(column_ref)
        downstream = self.get_downstream(column_ref)

        node_id = self._resolve_node_id(column_ref)
        direct_deps = len(self._upstream_index.get(node_id, set())) + len(self._downstream_index.get(node_id, set()))

        affected_tables = set()
        for node in upstream + downstream:
            affected_tables.add(node.table_name)

        # 최대 깊이 계산
        max_depth = self._calculate_max_depth(column_ref)

        return ImpactAnalysis(
            source_column=column_ref,
            upstream_columns=[n.full_name for n in upstream],
            downstream_columns=[n.full_name for n in downstream],
            direct_dependencies=direct_deps,
            total_impact_count=len(upstream) + len(downstream),
            max_depth=max_depth,
            affected_tables=affected_tables,
        )

    def _calculate_max_depth(self, column_ref: str) -> int:
        """최대 계보 깊이 계산"""
        start_id = self._resolve_node_id(column_ref)
        if not start_id:
            return 0

        # Upstream 깊이
        upstream_depth = 0
        visited = set()
        queue = deque([(start_id, 0)])

        while queue:
            node_id, depth = queue.popleft()
            if node_id in visited:
                continue
            visited.add(node_id)
            upstream_depth = max(upstream_depth, depth)

            for upstream_id in self._upstream_index.get(node_id, set()):
                queue.append((upstream_id, depth + 1))

        # Downstream 깊이
        downstream_depth = 0
        visited = set()
        queue = deque([(start_id, 0)])

        while queue:
            node_id, depth = queue.popleft()
            if node_id in visited:
                continue
            visited.add(node_id)
            downstream_depth = max(downstream_depth, depth)

            for downstream_id in self._downstream_index.get(node_id, set()):
                queue.append((downstream_id, depth + 1))

        return max(upstream_depth, downstream_depth)

    def get_table_lineage(self, table_name: str) -> Dict[str, Any]:
        """테이블 수준 lineage 요약"""
        node_ids = self._table_columns.get(table_name, set())

        all_upstream_tables = set()
        all_downstream_tables = set()

        for node_id in node_ids:
            for upstream_id in self._upstream_index.get(node_id, set()):
                all_upstream_tables.add(self.nodes[upstream_id].table_name)
            for downstream_id in self._downstream_index.get(node_id, set()):
                all_downstream_tables.add(self.nodes[downstream_id].table_name)

        # 자기 자신 제외
        all_upstream_tables.discard(table_name)
        all_downstream_tables.discard(table_name)

        return {
            "table_name": table_name,
            "column_count": len(node_ids),
            "upstream_tables": list(all_upstream_tables),
            "downstream_tables": list(all_downstream_tables),
        }

    def export_graph(self) -> Dict[str, Any]:
        """시각화용 그래프 데이터 내보내기"""
        return {
            "nodes": [n.to_dict() for n in self.nodes.values()],
            "edges": [e.to_dict() for e in self.edges.values()],
            "tables": list(self._table_columns.keys()),
            "summary": {
                "total_nodes": len(self.nodes),
                "total_edges": len(self.edges),
                "total_tables": len(self._table_columns),
            },
        }

    def get_fk_lineage_paths(self) -> List[Dict[str, Any]]:
        """FK 기반 lineage 경로 조회"""
        paths = []

        for edge in self.edges.values():
            if edge.edge_type == LineageEdgeType.FK_REFERENCE:
                source_node = self.nodes.get(edge.source_node_id)
                target_node = self.nodes.get(edge.target_node_id)

                if source_node and target_node:
                    paths.append({
                        "from": target_node.full_name,  # FK 컬럼
                        "to": source_node.full_name,    # PK 컬럼
                        "relationship": edge.fk_relationship,
                        "confidence": edge.confidence,
                    })

        return paths


# SharedContext 연동 함수
def build_lineage_from_context(
    shared_context: "SharedContext",
) -> Dict[str, Any]:
    """
    SharedContext에서 lineage 구축 후 저장

    사용법:
        from .column_lineage import build_lineage_from_context
        result = build_lineage_from_context(shared_context)
    """
    tracker = LineageTracker(shared_context)
    edge_count = tracker.build_from_fks()

    # 결과 저장
    lineage_data = tracker.export_graph()
    shared_context.set_dynamic("column_lineage", lineage_data)
    shared_context.set_dynamic("fk_lineage_paths", tracker.get_fk_lineage_paths())

    # 테이블별 요약
    table_summaries = {}
    for table_name in tracker._table_columns.keys():
        table_summaries[table_name] = tracker.get_table_lineage(table_name)

    shared_context.set_dynamic("table_lineage_summaries", table_summaries)

    logger.info(f"[LineageTracker] Built lineage: {len(tracker.nodes)} nodes, {edge_count} edges")

    # tracker 객체를 반환하여 impact analysis 등을 직접 사용 가능하게 함
    # summary dict는 이미 SharedContext에 저장됨
    return tracker


# 모듈 초기화
__all__ = [
    "LineageTracker",
    "ColumnNode",
    "LineageEdge",
    "LineageEdgeType",
    "ColumnType",
    "ImpactAnalysis",
    "build_lineage_from_context",
]
