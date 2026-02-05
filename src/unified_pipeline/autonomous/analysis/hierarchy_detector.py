"""
Hierarchy Detector (v16.0)

Self-Reference FK 100% 탐지:
- 조직도 (employees.manager_id → employees.emp_id)
- 카테고리 트리 (categories.parent_id → categories.category_id)
- BOM (parts.parent_part → parts.part_id)
- 댓글 트리 (comments.parent_comment → comments.comment_id)
- 부서 계층 (departments.parent_dept → departments.department_id)

EnhancedFKDetector와 통합되어 사용됨

References:
- Graph-based hierarchy detection
- Pattern matching + Data validation hybrid approach
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Set, Tuple, TYPE_CHECKING
import re
import logging

from .enhanced_fk_detector import FKCandidate, EnhancedFKDetector

if TYPE_CHECKING:
    from ...shared_context import SharedContext

logger = logging.getLogger(__name__)


@dataclass
class HierarchyCandidate:
    """계층 구조 후보"""
    table: str
    child_column: str   # FK (manager_id)
    parent_column: str  # PK (emp_id)

    hierarchy_type: str  # org_chart, category_tree, bom, location, comment_thread, generic
    max_depth: int = 0
    root_count: int = 0  # NULL parent 수

    # Scores
    pattern_score: float = 0.0
    value_score: float = 0.0
    structure_score: float = 0.0

    confidence: float = 0.0

    # 사이클 탐지
    has_cycle: bool = False

    def to_fk_candidate(self) -> FKCandidate:
        """FKCandidate로 변환 (하위 호환성)"""
        return FKCandidate(
            from_table=self.table,
            from_column=self.child_column,
            to_table=self.table,  # Self-reference
            to_column=self.parent_column,
            name_similarity=self.pattern_score,
            value_inclusion=self.value_score,
            fk_score=self.confidence,
            confidence="high" if self.confidence >= 0.8 else "medium" if self.confidence >= 0.6 else "low",
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "table": self.table,
            "child_column": self.child_column,
            "parent_column": self.parent_column,
            "hierarchy_type": self.hierarchy_type,
            "max_depth": self.max_depth,
            "root_count": self.root_count,
            "pattern_score": round(self.pattern_score, 4),
            "value_score": round(self.value_score, 4),
            "structure_score": round(self.structure_score, 4),
            "confidence": round(self.confidence, 4),
            "has_cycle": self.has_cycle,
            "is_self_reference": True,
        }


class HierarchyDetector:
    """
    계층 구조 (Self-Reference FK) 탐지기

    알고리즘:
    1. 패턴 매칭: parent_*, manager_*, *_parent, *_manager 등
    2. 값 분석: child 값이 parent 컬럼에 존재
    3. 구조 분석: root 노드 (NULL) 존재, 깊이 계산
    4. 사이클 검출: 순환 참조 방지
    """

    # Self-reference 패턴 (child_column → hierarchy_type)
    HIERARCHY_PATTERNS: Dict[str, List[Tuple[str, str]]] = {
        "org_chart": [
            (r"manager_?(?:id)?$", r"emp(?:loyee)?_?(?:id)?$"),
            (r"supervisor_?(?:id)?$", r"emp(?:loyee)?_?(?:id)?$"),
            (r"reports_?to$", r"emp(?:loyee)?_?(?:id)?$"),
            (r"boss_?(?:id)?$", r"emp(?:loyee)?_?(?:id)?$"),
            (r"lead(?:er)?_?(?:id)?$", r"emp(?:loyee)?_?(?:id)?$"),
        ],
        "category_tree": [
            (r"parent_?(?:cat(?:egory)?)?(?:_?id)?$", r"(?:cat(?:egory)?_?)?(?:id)?$"),
            (r"super_?(?:cat(?:egory)?)?(?:_?id)?$", r"(?:cat(?:egory)?_?)?(?:id)?$"),
        ],
        "department_tree": [
            (r"parent_?(?:dept(?:artment)?)?(?:_?id)?$", r"(?:dept(?:artment)?_?)?(?:id)?$"),
            (r"head_?(?:dept(?:artment)?)?(?:_?id)?$", r"(?:dept(?:artment)?_?)?(?:id)?$"),
        ],
        "bom": [  # Bill of Materials
            (r"parent_?(?:part|component|item)(?:_?id)?$", r"(?:part|component|item)_?(?:id)?$"),
            (r"assembly_?(?:id)?$", r"(?:part|component)_?(?:id)?$"),
        ],
        "location": [
            (r"parent_?(?:loc(?:ation)?|region|area)(?:_?id)?$", r"(?:loc(?:ation)?|region|area)_?(?:id)?$"),
        ],
        "comment_thread": [
            (r"parent_?(?:comment|post|reply)(?:_?id)?$", r"(?:comment|post)_?(?:id)?$"),
            (r"reply_?to(?:_?id)?$", r"(?:comment|post)_?(?:id)?$"),
            (r"in_?reply_?to(?:_?id)?$", r"(?:comment|post)_?(?:id)?$"),
        ],
        "generic": [
            (r"parent_?(?:id)?$", r"(?:id)$"),
            (r"parent_?(.+)$", r"\1_?(?:id)?$"),  # parent_xxx → xxx_id
        ],
    }

    # 단일 단어 Self-reference 패턴 (값 기반으로 탐지)
    SINGLE_WORD_PARENT_PATTERNS = [
        "parent", "manager", "supervisor", "boss", "lead", "head",
        "parent_dept", "parent_cat", "parent_id"
    ]

    def __init__(self, llm_enhancer=None):
        self.llm_enhancer = llm_enhancer
        self.candidates: List[HierarchyCandidate] = []

    def detect_hierarchies(
        self,
        tables: Dict[str, Dict[str, Any]],
        sample_data: Dict[str, List[Dict[str, Any]]],
    ) -> List[HierarchyCandidate]:
        """
        계층 구조 탐지

        Args:
            tables: {table_name: {"columns": [...], "row_count": int, "primary_keys": [...]}}
            sample_data: {table_name: [row_dicts]}

        Returns:
            계층 구조 후보 리스트
        """
        self.candidates = []

        for table_name, table_info in tables.items():
            columns = [c.get("name", "") for c in table_info.get("columns", [])]
            pk_cols = table_info.get("primary_keys", [])
            data = sample_data.get(table_name, [])

            # PK가 없으면 추론
            if not pk_cols:
                pk_cols = self._infer_primary_key(columns, table_name)

            if not pk_cols:
                continue

            # 각 컬럼에 대해 self-reference 검사
            for col in columns:
                if col in pk_cols:
                    continue  # PK는 스킵

                for pk_col in pk_cols:
                    candidate = self._check_hierarchy(
                        table_name, col, pk_col, columns, data
                    )
                    if candidate:
                        self.candidates.append(candidate)

        # 정렬
        self.candidates.sort(key=lambda x: x.confidence, reverse=True)

        logger.info(f"[HierarchyDetector] Found {len(self.candidates)} hierarchy candidates")
        return self.candidates

    def _infer_primary_key(self, columns: List[str], table_name: str) -> List[str]:
        """PK 추론"""
        pk_candidates = []

        # 테이블명에서 엔티티 추출
        table_base = table_name.lower().rstrip('s')  # employees → employee

        for col in columns:
            col_lower = col.lower()

            # 1. 'id' 컬럼
            if col_lower == "id":
                return [col]

            # 2. 테이블명_id 패턴 (emp_id, employee_id)
            if col_lower == f"{table_base}_id" or col_lower == f"{table_name.lower()}_id":
                pk_candidates.insert(0, col)  # 최우선

            # 3. xxx_id 패턴
            elif col_lower.endswith("_id"):
                pk_candidates.append(col)

            # 4. xxx_key, xxx_code 패턴
            elif col_lower.endswith("_key") or col_lower.endswith("_code"):
                pk_candidates.append(col)

        return pk_candidates[:1] if pk_candidates else []

    def _check_hierarchy(
        self,
        table: str,
        child_col: str,
        parent_col: str,
        all_columns: List[str],
        data: List[Dict]
    ) -> Optional[HierarchyCandidate]:
        """계층 구조 검사"""

        # 1. 패턴 매칭
        pattern_score, hierarchy_type = self._match_pattern(child_col, parent_col, table)

        # 2. 값 분석 (패턴 점수가 낮아도 값으로 보완 가능)
        value_score = self._check_value_inclusion(data, child_col, parent_col)

        # 패턴 점수와 값 점수 중 하나라도 낮으면 스킵
        if pattern_score < 0.3 and value_score < 0.5:
            return None

        # 값 점수가 높으면 패턴 점수 부족해도 진행
        if value_score < 0.5 and pattern_score < 0.5:
            return None

        # 3. 구조 분석
        root_count, max_depth, has_cycle = self._analyze_structure(
            data, child_col, parent_col
        )

        if has_cycle:
            logger.warning(f"[HierarchyDetector] Cycle detected in {table}.{child_col}")
            # 사이클이 있어도 계층 관계로 인정 (경고만)

        if root_count == 0 and len(data) > 0:
            return None  # 루트 노드 없음 = 계층 구조 아님

        # 적절한 루트 비율 (전체의 5-50% 사이가 합리적)
        if len(data) > 0:
            root_ratio = root_count / len(data)
            if root_ratio < 0.01 or root_ratio > 0.8:
                structure_score = 0.3  # 비정상적인 비율
            else:
                structure_score = min(1.0, 0.5 + root_ratio)
        else:
            structure_score = 0.5

        # 4. 최종 신뢰도
        confidence = (
            pattern_score * 0.4 +
            value_score * 0.4 +
            structure_score * 0.2
        )

        if confidence < 0.5:
            return None

        return HierarchyCandidate(
            table=table,
            child_column=child_col,
            parent_column=parent_col,
            hierarchy_type=hierarchy_type,
            max_depth=max_depth,
            root_count=root_count,
            pattern_score=pattern_score,
            value_score=value_score,
            structure_score=structure_score,
            confidence=confidence,
            has_cycle=has_cycle,
        )

    def _match_pattern(self, child_col: str, parent_col: str, table_name: str) -> Tuple[float, str]:
        """패턴 매칭"""
        child_lower = child_col.lower()
        parent_lower = parent_col.lower()
        table_lower = table_name.lower()

        # 1. 정의된 패턴 매칭
        for h_type, patterns in self.HIERARCHY_PATTERNS.items():
            for child_pattern, parent_pattern in patterns:
                if re.search(child_pattern, child_lower) and re.search(parent_pattern, parent_lower):
                    return 1.0, h_type

        # 2. 테이블명 기반 추론
        # employees 테이블의 manager_id → emp_id
        if "employee" in table_lower or "emp" in table_lower:
            if any(kw in child_lower for kw in ["manager", "supervisor", "boss", "lead", "reports"]):
                return 0.9, "org_chart"

        # departments 테이블의 parent_dept → department_id
        if "dept" in table_lower or "department" in table_lower:
            if "parent" in child_lower:
                return 0.9, "department_tree"

        # categories 테이블의 parent_cat → category_id
        if "categor" in table_lower or "cat" in table_lower:
            if "parent" in child_lower:
                return 0.9, "category_tree"

        # 3. 단일 단어 패턴
        for pattern in self.SINGLE_WORD_PARENT_PATTERNS:
            if pattern in child_lower:
                return 0.7, "generic"

        # 4. parent_ 접두사
        if child_lower.startswith("parent_"):
            return 0.6, "generic"

        # 5. _parent 접미사
        if "_parent" in child_lower:
            return 0.6, "generic"

        return 0.0, "unknown"

    def _check_value_inclusion(
        self,
        data: List[Dict],
        child_col: str,
        parent_col: str
    ) -> float:
        """값 포함도 검사 (child 값이 parent에 존재하는지)"""
        if not data:
            return 0.0

        # Parent (PK) 값 집합
        parent_values: Set[str] = set()
        for row in data:
            val = row.get(parent_col)
            if val is not None and str(val).strip():
                parent_values.add(str(val).lower().strip())

        if not parent_values:
            return 0.0

        # Child (FK) 값 중 parent에 포함된 비율
        child_values: List[str] = []
        matched = 0
        null_count = 0

        for row in data:
            val = row.get(child_col)
            if val is None or str(val).lower().strip() in ("", "null", "none", "nan"):
                null_count += 1  # 루트 노드
            else:
                child_values.append(str(val).lower().strip())
                if str(val).lower().strip() in parent_values:
                    matched += 1

        if not child_values:
            # 모두 NULL이면 루트만 있는 flat 구조
            return 0.3 if null_count > 0 else 0.0

        inclusion = matched / len(child_values)

        # NULL이 있으면 보너스 (루트 노드 존재 = 계층 구조 특성)
        if null_count > 0:
            inclusion = min(1.0, inclusion + 0.1)

        return inclusion

    def _analyze_structure(
        self,
        data: List[Dict],
        child_col: str,
        parent_col: str
    ) -> Tuple[int, int, bool]:
        """구조 분석: 루트 수, 최대 깊이, 사이클 여부"""
        if not data:
            return 0, 0, False

        # 그래프 구성
        parent_map: Dict[str, str] = {}  # node_id -> parent_id
        all_ids: Set[str] = set()

        for row in data:
            node_id = row.get(parent_col)  # PK (자신의 ID)
            parent_id = row.get(child_col)  # FK (부모의 ID)

            if node_id is not None and str(node_id).strip():
                node_str = str(node_id).lower().strip()
                all_ids.add(node_str)

                if parent_id is not None and str(parent_id).lower().strip() not in ("", "null", "none", "nan"):
                    parent_str = str(parent_id).lower().strip()
                    parent_map[node_str] = parent_str

        # 루트 노드 (parent가 없는 노드)
        root_count = len(all_ids) - len(parent_map)

        # 최대 깊이 계산 (DFS) + 사이클 검출
        max_depth = 0
        has_cycle = False

        def get_depth(node_id: str, visited: Set[str], depth: int) -> int:
            nonlocal has_cycle

            if node_id in visited:
                has_cycle = True
                return depth  # 사이클 탐지

            if node_id not in parent_map:
                return depth  # 루트 도달

            visited.add(node_id)
            result = get_depth(parent_map[node_id], visited, depth + 1)
            visited.remove(node_id)

            return result

        for node_id in all_ids:
            depth = get_depth(node_id, set(), 1)
            max_depth = max(max_depth, depth)

            if has_cycle:
                break  # 사이클 발견 시 조기 종료

        return root_count, max_depth, has_cycle


def detect_hierarchies_and_update_context(
    shared_context: "SharedContext",
) -> List[HierarchyCandidate]:
    """
    계층 구조 탐지 후 SharedContext 업데이트

    사용법 (RelationshipDetector 에이전트에서):
        from ..analysis.hierarchy_detector import detect_hierarchies_and_update_context
        hierarchies = detect_hierarchies_and_update_context(self.shared_context)

    Args:
        shared_context: 공유 컨텍스트

    Returns:
        탐지된 계층 구조 후보 리스트
    """
    # 테이블 데이터 추출
    tables: Dict[str, Dict[str, Any]] = {}
    sample_data: Dict[str, List[Dict[str, Any]]] = {}

    for table_name, table_info in shared_context.tables.items():
        tables[table_name] = {
            "columns": table_info.columns,
            "row_count": table_info.row_count,
            "primary_keys": table_info.primary_keys,
        }
        sample_data[table_name] = table_info.sample_data

    # 계층 구조 탐지
    detector = HierarchyDetector()
    hierarchies = detector.detect_hierarchies(tables, sample_data)

    # SharedContext에 추가 (FKCandidate dict 형태로)
    for h in hierarchies:
        fk_dict = h.to_fk_candidate().to_dict()
        fk_dict["is_self_reference"] = True
        fk_dict["hierarchy_type"] = h.hierarchy_type
        fk_dict["max_depth"] = h.max_depth
        fk_dict["root_count"] = h.root_count
        fk_dict["has_cycle"] = h.has_cycle
        shared_context.enhanced_fk_candidates.append(fk_dict)

    # dynamic_data에 원본 저장
    shared_context.set_dynamic("hierarchy_candidates", [
        h.to_dict() for h in hierarchies
    ])

    logger.info(f"[HierarchyDetector] Added {len(hierarchies)} hierarchies to context")
    return hierarchies
