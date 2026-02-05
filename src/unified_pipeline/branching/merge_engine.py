"""
Merge Engine (v17.0)

3-way 머지 엔진:
- Fast-forward 머지
- 3-way 머지
- 충돌 감지

사용 예시:
    engine = MergeEngine(version_manager, branch_manager)
    result = engine.merge("feature/analysis", "main")
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Set, TYPE_CHECKING
from datetime import datetime
from enum import Enum

if TYPE_CHECKING:
    from ..versioning.version_manager import VersionManager, DatasetVersion
    from .branch_manager import BranchManager

logger = logging.getLogger(__name__)


class MergeStrategy(str, Enum):
    """머지 전략"""
    FAST_FORWARD = "fast_forward"
    THREE_WAY = "three_way"
    SQUASH = "squash"
    REBASE = "rebase"


class MergeStatus(str, Enum):
    """머지 상태"""
    SUCCESS = "success"
    CONFLICTS = "conflicts"
    FAILED = "failed"
    NO_CHANGES = "no_changes"


@dataclass
class RowConflict:
    """행 단위 충돌"""
    table: str
    row_id: str
    column: str
    base_value: Any
    source_value: Any
    target_value: Any

    def to_dict(self) -> Dict[str, Any]:
        return {
            "table": self.table,
            "row_id": self.row_id,
            "column": self.column,
            "base_value": self.base_value,
            "source_value": self.source_value,
            "target_value": self.target_value,
        }


@dataclass
class MergeConflict:
    """머지 충돌"""
    conflict_id: str
    table: str
    row_conflicts: List[RowConflict] = field(default_factory=list)

    # 메타데이터
    source_branch: str = ""
    target_branch: str = ""
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())

    @property
    def conflict_count(self) -> int:
        return len(self.row_conflicts)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "conflict_id": self.conflict_id,
            "table": self.table,
            "row_conflicts": [r.to_dict() for r in self.row_conflicts],
            "source_branch": self.source_branch,
            "target_branch": self.target_branch,
            "conflict_count": self.conflict_count,
        }


@dataclass
class MergeResult:
    """머지 결과"""
    status: MergeStatus
    source_branch: str
    target_branch: str
    strategy_used: MergeStrategy

    # 결과 버전
    result_version_id: Optional[str] = None
    lca_version_id: Optional[str] = None

    # 충돌 정보
    conflicts: List[MergeConflict] = field(default_factory=list)

    # 변경 통계
    rows_added: int = 0
    rows_modified: int = 0
    rows_deleted: int = 0
    tables_affected: List[str] = field(default_factory=list)

    # 메타데이터
    merged_at: str = field(default_factory=lambda: datetime.now().isoformat())
    message: str = ""

    @property
    def has_conflicts(self) -> bool:
        return len(self.conflicts) > 0

    @property
    def is_success(self) -> bool:
        return self.status == MergeStatus.SUCCESS

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status.value,
            "source_branch": self.source_branch,
            "target_branch": self.target_branch,
            "strategy_used": self.strategy_used.value,
            "result_version_id": self.result_version_id,
            "lca_version_id": self.lca_version_id,
            "conflicts": [c.to_dict() for c in self.conflicts],
            "rows_added": self.rows_added,
            "rows_modified": self.rows_modified,
            "rows_deleted": self.rows_deleted,
            "tables_affected": self.tables_affected,
            "has_conflicts": self.has_conflicts,
            "merged_at": self.merged_at,
            "message": self.message,
        }


class MergeEngine:
    """
    3-way 머지 엔진

    기능:
    - Fast-forward 머지 (변경 없이 HEAD 이동)
    - 3-way 머지 (LCA 기반 충돌 감지)
    - 충돌 감지 및 보고
    """

    def __init__(
        self,
        version_manager: "VersionManager",
        branch_manager: "BranchManager",
    ):
        """
        Args:
            version_manager: 버전 관리자
            branch_manager: 브랜치 관리자
        """
        self.version_manager = version_manager
        self.branch_manager = branch_manager

        logger.info("MergeEngine initialized")

    def merge(
        self,
        source_branch: str,
        target_branch: str,
        strategy: Optional[MergeStrategy] = None,
        message: Optional[str] = None,
        auto_resolve: bool = False,
    ) -> MergeResult:
        """
        브랜치 머지

        Args:
            source_branch: 소스 브랜치 (머지할 브랜치)
            target_branch: 타겟 브랜치 (머지 대상)
            strategy: 머지 전략 (None이면 자동 결정)
            message: 커밋 메시지
            auto_resolve: 충돌 자동 해결 시도

        Returns:
            MergeResult: 머지 결과
        """
        # 브랜치 존재 확인
        source = self.branch_manager.get_branch(source_branch)
        target = self.branch_manager.get_branch(target_branch)

        if not source:
            return MergeResult(
                status=MergeStatus.FAILED,
                source_branch=source_branch,
                target_branch=target_branch,
                strategy_used=strategy or MergeStrategy.THREE_WAY,
                message=f"Source branch not found: {source_branch}",
            )

        if not target:
            return MergeResult(
                status=MergeStatus.FAILED,
                source_branch=source_branch,
                target_branch=target_branch,
                strategy_used=strategy or MergeStrategy.THREE_WAY,
                message=f"Target branch not found: {target_branch}",
            )

        # HEAD 가져오기
        source_head = source.head
        target_head = target.head

        if not source_head:
            return MergeResult(
                status=MergeStatus.NO_CHANGES,
                source_branch=source_branch,
                target_branch=target_branch,
                strategy_used=strategy or MergeStrategy.THREE_WAY,
                message="Source branch has no commits",
            )

        # 같은 HEAD인 경우
        if source_head == target_head:
            return MergeResult(
                status=MergeStatus.NO_CHANGES,
                source_branch=source_branch,
                target_branch=target_branch,
                strategy_used=MergeStrategy.FAST_FORWARD,
                message="Branches are already up to date",
            )

        # LCA 찾기
        lca = self.branch_manager._find_lca(
            source.version_ids,
            target.version_ids if target.version_ids else []
        )

        # 전략 결정
        if strategy is None:
            strategy = self._determine_strategy(source, target, lca)

        # 머지 실행
        if strategy == MergeStrategy.FAST_FORWARD:
            return self._fast_forward_merge(source, target, message)
        else:
            return self._three_way_merge(
                source, target, lca, message, auto_resolve
            )

    def _determine_strategy(
        self,
        source: Any,  # Branch
        target: Any,  # Branch
        lca: Optional[str],
    ) -> MergeStrategy:
        """머지 전략 자동 결정"""
        # 타겟이 LCA와 같으면 fast-forward 가능
        if target.head == lca:
            return MergeStrategy.FAST_FORWARD

        # 그 외는 3-way merge
        return MergeStrategy.THREE_WAY

    def _fast_forward_merge(
        self,
        source: Any,  # Branch
        target: Any,  # Branch
        message: Optional[str],
    ) -> MergeResult:
        """Fast-forward 머지"""
        source_head = source.head

        # 타겟 HEAD를 소스 HEAD로 이동
        target.info.head = source_head
        if source_head not in target.version_ids:
            # 소스의 새 커밋들을 타겟에 추가
            for vid in source.version_ids:
                if vid not in target.version_ids:
                    target.version_ids.append(vid)

        # VersionManager 업데이트
        self.version_manager.branch_heads[target.name] = source_head

        logger.info(
            f"Fast-forward merge: {source.name} -> {target.name} "
            f"(HEAD: {source_head})"
        )

        return MergeResult(
            status=MergeStatus.SUCCESS,
            source_branch=source.name,
            target_branch=target.name,
            strategy_used=MergeStrategy.FAST_FORWARD,
            result_version_id=source_head,
            message=message or f"Fast-forward merge {source.name} into {target.name}",
        )

    def _three_way_merge(
        self,
        source: Any,  # Branch
        target: Any,  # Branch
        lca: Optional[str],
        message: Optional[str],
        auto_resolve: bool,
    ) -> MergeResult:
        """3-way 머지"""
        source_head = source.head
        target_head = target.head

        # 스냅샷 가져오기
        source_version = self.version_manager.get_version(source_head) if source_head else None
        target_version = self.version_manager.get_version(target_head) if target_head else None
        lca_version = self.version_manager.get_version(lca) if lca else None

        source_snapshot = source_version.snapshot if source_version else {}
        target_snapshot = target_version.snapshot if target_version else {}
        lca_snapshot = lca_version.snapshot if lca_version else {}

        # 3-way 머지 수행
        merged_snapshot, conflicts = self._perform_three_way_merge(
            source_snapshot,
            target_snapshot,
            lca_snapshot,
            source.name,
            target.name,
        )

        # 충돌이 있고 자동 해결 불가
        if conflicts and not auto_resolve:
            return MergeResult(
                status=MergeStatus.CONFLICTS,
                source_branch=source.name,
                target_branch=target.name,
                strategy_used=MergeStrategy.THREE_WAY,
                lca_version_id=lca,
                conflicts=conflicts,
                message="Merge conflicts detected",
            )

        # 충돌 자동 해결 (소스 우선)
        if conflicts and auto_resolve:
            merged_snapshot = self._auto_resolve_conflicts(
                merged_snapshot, conflicts, "source"
            )
            conflicts = []

        # 변경 통계 계산
        rows_added, rows_modified, rows_deleted = self._compute_change_stats(
            target_snapshot, merged_snapshot
        )

        # 머지 커밋 생성
        # 현재 브랜치를 타겟으로 임시 전환
        original_branch = self.version_manager.current_branch
        self.version_manager.current_branch = target.name

        merge_message = message or f"Merge branch '{source.name}' into '{target.name}'"

        # 새 버전으로 커밋 (스냅샷 직접 설정 필요)
        # NOTE: 실제로는 VersionManager에 스냅샷 직접 설정 메서드 필요
        version_id = self.version_manager.commit(merge_message)

        # 원래 브랜치로 복원
        self.version_manager.current_branch = original_branch

        # 타겟 브랜치 업데이트
        target.add_commit(version_id)

        logger.info(
            f"3-way merge: {source.name} -> {target.name} "
            f"(+{rows_added} ~{rows_modified} -{rows_deleted})"
        )

        return MergeResult(
            status=MergeStatus.SUCCESS,
            source_branch=source.name,
            target_branch=target.name,
            strategy_used=MergeStrategy.THREE_WAY,
            result_version_id=version_id,
            lca_version_id=lca,
            rows_added=rows_added,
            rows_modified=rows_modified,
            rows_deleted=rows_deleted,
            tables_affected=list(merged_snapshot.keys()),
            message=merge_message,
        )

    def _perform_three_way_merge(
        self,
        source: Dict[str, Dict[str, Dict[str, Any]]],
        target: Dict[str, Dict[str, Dict[str, Any]]],
        base: Dict[str, Dict[str, Dict[str, Any]]],
        source_branch: str,
        target_branch: str,
    ) -> tuple:
        """
        3-way 머지 수행

        Returns:
            (merged_snapshot, conflicts)
        """
        merged: Dict[str, Dict[str, Dict[str, Any]]] = {}
        conflicts: List[MergeConflict] = []

        all_tables = set(source.keys()) | set(target.keys()) | set(base.keys())

        for table in all_tables:
            source_rows = source.get(table, {})
            target_rows = target.get(table, {})
            base_rows = base.get(table, {})

            merged_rows, table_conflicts = self._merge_table(
                source_rows,
                target_rows,
                base_rows,
                table,
                source_branch,
                target_branch,
            )

            merged[table] = merged_rows
            conflicts.extend(table_conflicts)

        return merged, conflicts

    def _merge_table(
        self,
        source_rows: Dict[str, Dict[str, Any]],
        target_rows: Dict[str, Dict[str, Any]],
        base_rows: Dict[str, Dict[str, Any]],
        table_name: str,
        source_branch: str,
        target_branch: str,
    ) -> tuple:
        """테이블 단위 머지"""
        merged: Dict[str, Dict[str, Any]] = {}
        conflicts: List[MergeConflict] = []

        all_row_ids = set(source_rows.keys()) | set(target_rows.keys()) | set(base_rows.keys())

        table_conflict = MergeConflict(
            conflict_id=f"conflict_{table_name}_{datetime.now().timestamp()}",
            table=table_name,
            source_branch=source_branch,
            target_branch=target_branch,
        )

        for row_id in all_row_ids:
            source_row = source_rows.get(row_id)
            target_row = target_rows.get(row_id)
            base_row = base_rows.get(row_id)

            # 케이스 분류
            if base_row is None:
                # 신규 행
                if source_row and target_row:
                    # 양쪽에서 추가됨 → 충돌 가능성
                    if source_row != target_row:
                        row_conflict = self._detect_row_conflicts(
                            table_name, row_id, None, source_row, target_row
                        )
                        table_conflict.row_conflicts.extend(row_conflict)
                    merged[row_id] = target_row  # 기본: 타겟 우선
                elif source_row:
                    merged[row_id] = source_row
                elif target_row:
                    merged[row_id] = target_row

            elif source_row is None and target_row is None:
                # 양쪽에서 삭제됨 → 삭제 유지
                pass

            elif source_row is None:
                # 소스에서 삭제됨
                if target_row == base_row:
                    # 타겟에서 변경 없음 → 삭제 적용
                    pass
                else:
                    # 타겟에서 변경됨 → 충돌
                    row_conflict = RowConflict(
                        table=table_name,
                        row_id=row_id,
                        column="*",
                        base_value=base_row,
                        source_value=None,
                        target_value=target_row,
                    )
                    table_conflict.row_conflicts.append(row_conflict)
                    merged[row_id] = target_row  # 기본: 타겟 우선

            elif target_row is None:
                # 타겟에서 삭제됨
                if source_row == base_row:
                    # 소스에서 변경 없음 → 삭제 적용
                    pass
                else:
                    # 소스에서 변경됨 → 충돌
                    row_conflict = RowConflict(
                        table=table_name,
                        row_id=row_id,
                        column="*",
                        base_value=base_row,
                        source_value=source_row,
                        target_value=None,
                    )
                    table_conflict.row_conflicts.append(row_conflict)
                    merged[row_id] = source_row  # 기본: 소스 우선

            else:
                # 양쪽 존재
                merged_row, row_conflicts = self._merge_row(
                    table_name, row_id, base_row, source_row, target_row
                )
                merged[row_id] = merged_row
                table_conflict.row_conflicts.extend(row_conflicts)

        if table_conflict.row_conflicts:
            conflicts.append(table_conflict)

        return merged, conflicts

    def _merge_row(
        self,
        table_name: str,
        row_id: str,
        base_row: Dict[str, Any],
        source_row: Dict[str, Any],
        target_row: Dict[str, Any],
    ) -> tuple:
        """행 단위 머지"""
        merged: Dict[str, Any] = {}
        conflicts: List[RowConflict] = []

        all_columns = set(base_row.keys()) | set(source_row.keys()) | set(target_row.keys())

        for col in all_columns:
            base_val = base_row.get(col)
            source_val = source_row.get(col)
            target_val = target_row.get(col)

            if source_val == target_val:
                # 같은 값
                merged[col] = source_val
            elif source_val == base_val:
                # 소스에서 변경 없음 → 타겟 변경 적용
                merged[col] = target_val
            elif target_val == base_val:
                # 타겟에서 변경 없음 → 소스 변경 적용
                merged[col] = source_val
            else:
                # 양쪽에서 다르게 변경 → 충돌
                conflicts.append(RowConflict(
                    table=table_name,
                    row_id=row_id,
                    column=col,
                    base_value=base_val,
                    source_value=source_val,
                    target_value=target_val,
                ))
                merged[col] = target_val  # 기본: 타겟 우선

        return merged, conflicts

    def _detect_row_conflicts(
        self,
        table_name: str,
        row_id: str,
        base_row: Optional[Dict[str, Any]],
        source_row: Dict[str, Any],
        target_row: Dict[str, Any],
    ) -> List[RowConflict]:
        """행 충돌 감지"""
        conflicts = []

        all_columns = set(source_row.keys()) | set(target_row.keys())

        for col in all_columns:
            source_val = source_row.get(col)
            target_val = target_row.get(col)
            base_val = base_row.get(col) if base_row else None

            if source_val != target_val:
                conflicts.append(RowConflict(
                    table=table_name,
                    row_id=row_id,
                    column=col,
                    base_value=base_val,
                    source_value=source_val,
                    target_value=target_val,
                ))

        return conflicts

    def _auto_resolve_conflicts(
        self,
        merged_snapshot: Dict[str, Dict[str, Dict[str, Any]]],
        conflicts: List[MergeConflict],
        prefer: str = "source",
    ) -> Dict[str, Dict[str, Dict[str, Any]]]:
        """충돌 자동 해결"""
        for conflict in conflicts:
            for row_conflict in conflict.row_conflicts:
                table = row_conflict.table
                row_id = row_conflict.row_id
                col = row_conflict.column

                if table in merged_snapshot and row_id in merged_snapshot[table]:
                    if col == "*":
                        # 전체 행 충돌
                        if prefer == "source" and row_conflict.source_value:
                            merged_snapshot[table][row_id] = row_conflict.source_value
                        elif prefer == "target" and row_conflict.target_value:
                            merged_snapshot[table][row_id] = row_conflict.target_value
                    else:
                        # 특정 컬럼 충돌
                        value = (
                            row_conflict.source_value if prefer == "source"
                            else row_conflict.target_value
                        )
                        merged_snapshot[table][row_id][col] = value

        return merged_snapshot

    def _compute_change_stats(
        self,
        before: Dict[str, Dict[str, Dict[str, Any]]],
        after: Dict[str, Dict[str, Dict[str, Any]]],
    ) -> tuple:
        """변경 통계 계산"""
        added = 0
        modified = 0
        deleted = 0

        all_tables = set(before.keys()) | set(after.keys())

        for table in all_tables:
            before_rows = before.get(table, {})
            after_rows = after.get(table, {})

            before_ids = set(before_rows.keys())
            after_ids = set(after_rows.keys())

            added += len(after_ids - before_ids)
            deleted += len(before_ids - after_ids)

            for row_id in before_ids & after_ids:
                if before_rows[row_id] != after_rows[row_id]:
                    modified += 1

        return added, modified, deleted
