"""
Dataset Version Manager (v17.0)

Palantir Foundry 스타일 Git-like 버전 관리:
- 커밋 (스냅샷 생성)
- 체크아웃 (버전 복원)
- 디프 (버전 비교)
- 로그 (히스토리 조회)

사용 예시:
    manager = VersionManager(context)

    # 커밋
    version_id = manager.commit("Initial data load")

    # 체크아웃
    manager.checkout(version_id)

    # 디프
    diff = manager.diff(version_a, version_b)

    # 로그
    history = manager.log()
"""

import logging
import hashlib
import json
import copy
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Set, TYPE_CHECKING
from datetime import datetime
from enum import Enum

if TYPE_CHECKING:
    from ..shared_context import SharedContext

logger = logging.getLogger(__name__)


class ChangeType(str, Enum):
    """변경 유형"""
    ADDED = "added"
    MODIFIED = "modified"
    DELETED = "deleted"
    UNCHANGED = "unchanged"


@dataclass
class CommitInfo:
    """커밋 정보"""
    version_id: str
    message: str
    author: str
    timestamp: str
    parent_version: Optional[str] = None
    branch: str = "main"

    # 메타데이터
    tables_affected: List[str] = field(default_factory=list)
    rows_added: int = 0
    rows_modified: int = 0
    rows_deleted: int = 0

    # 태그
    tags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version_id": self.version_id,
            "message": self.message,
            "author": self.author,
            "timestamp": self.timestamp,
            "parent_version": self.parent_version,
            "branch": self.branch,
            "tables_affected": self.tables_affected,
            "rows_added": self.rows_added,
            "rows_modified": self.rows_modified,
            "rows_deleted": self.rows_deleted,
            "tags": self.tags,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CommitInfo":
        return cls(
            version_id=data["version_id"],
            message=data["message"],
            author=data["author"],
            timestamp=data["timestamp"],
            parent_version=data.get("parent_version"),
            branch=data.get("branch", "main"),
            tables_affected=data.get("tables_affected", []),
            rows_added=data.get("rows_added", 0),
            rows_modified=data.get("rows_modified", 0),
            rows_deleted=data.get("rows_deleted", 0),
            tags=data.get("tags", []),
        )


@dataclass
class RowChange:
    """행 단위 변경 정보"""
    table: str
    row_id: str
    change_type: ChangeType
    old_values: Optional[Dict[str, Any]] = None
    new_values: Optional[Dict[str, Any]] = None
    changed_columns: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "table": self.table,
            "row_id": self.row_id,
            "change_type": self.change_type.value,
            "old_values": self.old_values,
            "new_values": self.new_values,
            "changed_columns": self.changed_columns,
        }


@dataclass
class TableDiff:
    """테이블 단위 디프"""
    table_name: str
    rows_added: List[RowChange] = field(default_factory=list)
    rows_modified: List[RowChange] = field(default_factory=list)
    rows_deleted: List[RowChange] = field(default_factory=list)

    # 스키마 변경
    columns_added: List[str] = field(default_factory=list)
    columns_removed: List[str] = field(default_factory=list)
    columns_modified: List[str] = field(default_factory=list)

    @property
    def has_changes(self) -> bool:
        return bool(
            self.rows_added or
            self.rows_modified or
            self.rows_deleted or
            self.columns_added or
            self.columns_removed or
            self.columns_modified
        )

    @property
    def total_changes(self) -> int:
        return (
            len(self.rows_added) +
            len(self.rows_modified) +
            len(self.rows_deleted)
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "table_name": self.table_name,
            "rows_added": [r.to_dict() for r in self.rows_added],
            "rows_modified": [r.to_dict() for r in self.rows_modified],
            "rows_deleted": [r.to_dict() for r in self.rows_deleted],
            "columns_added": self.columns_added,
            "columns_removed": self.columns_removed,
            "columns_modified": self.columns_modified,
            "has_changes": self.has_changes,
            "total_changes": self.total_changes,
        }


@dataclass
class VersionDiff:
    """버전 간 디프"""
    version_a: str
    version_b: str
    table_diffs: Dict[str, TableDiff] = field(default_factory=dict)

    # 요약
    tables_added: List[str] = field(default_factory=list)
    tables_removed: List[str] = field(default_factory=list)
    tables_modified: List[str] = field(default_factory=list)

    # 메타데이터
    computed_at: str = field(default_factory=lambda: datetime.now().isoformat())

    @property
    def has_changes(self) -> bool:
        return bool(
            self.tables_added or
            self.tables_removed or
            any(td.has_changes for td in self.table_diffs.values())
        )

    @property
    def total_row_changes(self) -> int:
        return sum(td.total_changes for td in self.table_diffs.values())

    def get_summary(self) -> Dict[str, Any]:
        """변경 요약"""
        return {
            "version_a": self.version_a,
            "version_b": self.version_b,
            "tables_added": len(self.tables_added),
            "tables_removed": len(self.tables_removed),
            "tables_modified": len(self.tables_modified),
            "total_row_changes": self.total_row_changes,
            "has_changes": self.has_changes,
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version_a": self.version_a,
            "version_b": self.version_b,
            "table_diffs": {k: v.to_dict() for k, v in self.table_diffs.items()},
            "tables_added": self.tables_added,
            "tables_removed": self.tables_removed,
            "tables_modified": self.tables_modified,
            "computed_at": self.computed_at,
            "summary": self.get_summary(),
        }


@dataclass
class DatasetVersion:
    """데이터셋 버전 (스냅샷 포함)"""
    version_id: str
    commit_info: CommitInfo

    # 스냅샷 데이터 (테이블별 행 데이터)
    # 구조: {table_name: {row_id: row_data}}
    snapshot: Dict[str, Dict[str, Dict[str, Any]]] = field(default_factory=dict)

    # 스키마 스냅샷
    # 구조: {table_name: {column_name: column_type}}
    schema_snapshot: Dict[str, Dict[str, str]] = field(default_factory=dict)

    # 체크섬 (무결성 검증)
    checksum: str = ""

    def compute_checksum(self) -> str:
        """스냅샷 체크섬 계산"""
        content = json.dumps(self.snapshot, sort_keys=True, default=str)
        return hashlib.sha256(content.encode()).hexdigest()[:12]

    def verify_integrity(self) -> bool:
        """무결성 검증"""
        return self.checksum == self.compute_checksum()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version_id": self.version_id,
            "commit_info": self.commit_info.to_dict(),
            "snapshot": self.snapshot,
            "schema_snapshot": self.schema_snapshot,
            "checksum": self.checksum,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DatasetVersion":
        return cls(
            version_id=data["version_id"],
            commit_info=CommitInfo.from_dict(data["commit_info"]),
            snapshot=data.get("snapshot", {}),
            schema_snapshot=data.get("schema_snapshot", {}),
            checksum=data.get("checksum", ""),
        )


class VersionManager:
    """
    데이터셋 버전 관리자

    Git-like 버전 관리 제공:
    - commit: 현재 상태 스냅샷 생성
    - checkout: 특정 버전으로 복원
    - diff: 버전 간 차이 비교
    - log: 히스토리 조회
    - tag: 버전에 태그 부여
    """

    def __init__(
        self,
        context: Optional["SharedContext"] = None,
        author: str = "system",
    ):
        """
        Args:
            context: SharedContext (상태 공유용)
            author: 기본 커밋 작성자
        """
        self.context = context
        self.default_author = author

        # 버전 저장소
        self.versions: Dict[str, DatasetVersion] = {}

        # 브랜치별 HEAD
        self.branch_heads: Dict[str, str] = {"main": ""}

        # 현재 브랜치
        self.current_branch: str = "main"

        # 태그
        self.tags: Dict[str, str] = {}  # tag_name -> version_id

        logger.info("VersionManager initialized")

    def commit(
        self,
        message: str,
        author: Optional[str] = None,
        tables: Optional[List[str]] = None,
        tags: Optional[List[str]] = None,
    ) -> str:
        """
        현재 상태를 커밋 (스냅샷 생성)

        Args:
            message: 커밋 메시지
            author: 작성자 (기본값 사용 가능)
            tables: 커밋할 테이블 (None이면 전체)
            tags: 태그 목록

        Returns:
            version_id: 생성된 버전 ID
        """
        author = author or self.default_author
        timestamp = datetime.now().isoformat()

        # 현재 HEAD 가져오기
        parent_version = self.branch_heads.get(self.current_branch, "")
        if not parent_version:
            parent_version = None

        # 스냅샷 생성
        snapshot, schema_snapshot = self._create_snapshot(tables)

        # 버전 ID 생성
        version_id = self._generate_version_id(timestamp, message)

        # 변경 사항 분석
        rows_added, rows_modified, rows_deleted = 0, 0, 0
        tables_affected = list(snapshot.keys())

        if parent_version and parent_version in self.versions:
            parent = self.versions[parent_version]
            diff = self._compute_diff_internal(parent.snapshot, snapshot)
            for table_diff in diff.values():
                rows_added += len(table_diff.rows_added)
                rows_modified += len(table_diff.rows_modified)
                rows_deleted += len(table_diff.rows_deleted)
        else:
            # 첫 커밋
            for table_data in snapshot.values():
                rows_added += len(table_data)

        # CommitInfo 생성
        commit_info = CommitInfo(
            version_id=version_id,
            message=message,
            author=author,
            timestamp=timestamp,
            parent_version=parent_version,
            branch=self.current_branch,
            tables_affected=tables_affected,
            rows_added=rows_added,
            rows_modified=rows_modified,
            rows_deleted=rows_deleted,
            tags=tags or [],
        )

        # DatasetVersion 생성
        version = DatasetVersion(
            version_id=version_id,
            commit_info=commit_info,
            snapshot=snapshot,
            schema_snapshot=schema_snapshot,
        )
        version.checksum = version.compute_checksum()

        # 저장
        self.versions[version_id] = version
        self.branch_heads[self.current_branch] = version_id

        # 태그 등록
        if tags:
            for tag in tags:
                self.tags[tag] = version_id

        # SharedContext 업데이트
        if self.context:
            self._update_context_on_commit(version)

        logger.info(
            f"Committed version {version_id}: {message} "
            f"(+{rows_added} ~{rows_modified} -{rows_deleted})"
        )

        return version_id

    def checkout(
        self,
        version_id: str,
        force: bool = False,
    ) -> bool:
        """
        특정 버전으로 체크아웃 (복원)

        Args:
            version_id: 복원할 버전 ID 또는 태그
            force: 변경 사항 무시하고 강제 체크아웃

        Returns:
            성공 여부
        """
        # 태그 → 버전 ID 변환
        if version_id in self.tags:
            version_id = self.tags[version_id]

        if version_id not in self.versions:
            logger.error(f"Version not found: {version_id}")
            return False

        version = self.versions[version_id]

        # 무결성 검증
        if not version.verify_integrity():
            logger.error(f"Version integrity check failed: {version_id}")
            return False

        # SharedContext에 스냅샷 복원
        if self.context:
            self._restore_snapshot_to_context(version)

        logger.info(f"Checked out version: {version_id}")

        return True

    def diff(
        self,
        version_a: str,
        version_b: str,
    ) -> VersionDiff:
        """
        두 버전 간 차이 비교

        Args:
            version_a: 기준 버전 ID
            version_b: 비교 버전 ID

        Returns:
            VersionDiff: 차이 정보
        """
        # 태그 → 버전 ID 변환
        if version_a in self.tags:
            version_a = self.tags[version_a]
        if version_b in self.tags:
            version_b = self.tags[version_b]

        if version_a not in self.versions:
            raise ValueError(f"Version not found: {version_a}")
        if version_b not in self.versions:
            raise ValueError(f"Version not found: {version_b}")

        snapshot_a = self.versions[version_a].snapshot
        snapshot_b = self.versions[version_b].snapshot

        table_diffs = self._compute_diff_internal(snapshot_a, snapshot_b)

        # 테이블 변경 분류
        tables_a = set(snapshot_a.keys())
        tables_b = set(snapshot_b.keys())

        tables_added = list(tables_b - tables_a)
        tables_removed = list(tables_a - tables_b)
        tables_modified = [
            name for name, diff in table_diffs.items()
            if diff.has_changes and name not in tables_added and name not in tables_removed
        ]

        return VersionDiff(
            version_a=version_a,
            version_b=version_b,
            table_diffs=table_diffs,
            tables_added=tables_added,
            tables_removed=tables_removed,
            tables_modified=tables_modified,
        )

    def log(
        self,
        branch: Optional[str] = None,
        limit: int = 50,
    ) -> List[CommitInfo]:
        """
        커밋 히스토리 조회

        Args:
            branch: 브랜치 (None이면 현재 브랜치)
            limit: 최대 조회 개수

        Returns:
            커밋 정보 리스트 (최신순)
        """
        branch = branch or self.current_branch

        history = []
        version_id = self.branch_heads.get(branch)

        while version_id and len(history) < limit:
            if version_id not in self.versions:
                break

            version = self.versions[version_id]
            history.append(version.commit_info)
            version_id = version.commit_info.parent_version

        return history

    def tag(self, version_id: str, tag_name: str) -> bool:
        """
        버전에 태그 부여

        Args:
            version_id: 버전 ID
            tag_name: 태그 이름

        Returns:
            성공 여부
        """
        if version_id not in self.versions:
            logger.error(f"Version not found: {version_id}")
            return False

        self.tags[tag_name] = version_id
        self.versions[version_id].commit_info.tags.append(tag_name)

        logger.info(f"Tagged {version_id} as '{tag_name}'")
        return True

    def get_version(self, version_id: str) -> Optional[DatasetVersion]:
        """버전 조회"""
        if version_id in self.tags:
            version_id = self.tags[version_id]
        return self.versions.get(version_id)

    def get_current_head(self) -> Optional[str]:
        """현재 브랜치의 HEAD 버전 ID"""
        return self.branch_heads.get(self.current_branch)

    def list_versions(self) -> List[str]:
        """모든 버전 ID 목록"""
        return list(self.versions.keys())

    def list_tags(self) -> Dict[str, str]:
        """모든 태그 목록"""
        return dict(self.tags)

    # === Private Methods ===

    def _generate_version_id(self, timestamp: str, message: str) -> str:
        """버전 ID 생성"""
        content = f"{timestamp}:{message}:{self.current_branch}"
        hash_val = hashlib.sha256(content.encode()).hexdigest()[:8]
        return f"v_{hash_val}"

    def _create_snapshot(
        self,
        tables: Optional[List[str]] = None,
    ) -> tuple:
        """현재 상태에서 스냅샷 생성"""
        snapshot: Dict[str, Dict[str, Dict[str, Any]]] = {}
        schema_snapshot: Dict[str, Dict[str, str]] = {}

        if not self.context:
            return snapshot, schema_snapshot

        # 테이블 정보에서 스냅샷 생성
        table_infos = self.context.tables

        for table_name, table_info in table_infos.items():
            if tables and table_name not in tables:
                continue

            # TableInfo dataclass 또는 dict 처리
            if hasattr(table_info, 'columns'):
                columns = table_info.columns or []
                sample_data = table_info.sample_data or []
            elif isinstance(table_info, dict):
                columns = table_info.get("columns", [])
                sample_data = table_info.get("sample_data", [])
            else:
                columns = []
                sample_data = []

            # 스키마 스냅샷 (columns가 List[Dict]인 경우)
            if columns and isinstance(columns, list):
                schema_snapshot[table_name] = {
                    col.get("name", f"col_{i}"): col.get("dtype", "unknown")
                    for i, col in enumerate(columns)
                    if isinstance(col, dict)
                }
            elif columns and isinstance(columns, dict):
                schema_snapshot[table_name] = {
                    col_name: col_info.get("dtype", "unknown")
                    for col_name, col_info in columns.items()
                }
            else:
                schema_snapshot[table_name] = {}

            # 데이터 스냅샷 (샘플 데이터 사용)
            pk_column = self._get_primary_key(table_info, columns)

            snapshot[table_name] = {}
            for idx, row in enumerate(sample_data):
                row_id = str(row.get(pk_column, idx)) if pk_column else str(idx)
                snapshot[table_name][row_id] = copy.deepcopy(row)

        return snapshot, schema_snapshot

    def _get_primary_key(self, table_info: Any, columns: Any = None) -> Optional[str]:
        """테이블의 PK 컬럼 찾기"""
        # columns 파라미터가 전달되지 않았으면 table_info에서 추출
        if columns is None:
            if hasattr(table_info, 'columns'):
                columns = table_info.columns or []
            elif isinstance(table_info, dict):
                columns = table_info.get("columns", [])
            else:
                columns = []

        # columns가 List[Dict]인 경우 (TableInfo 형식)
        if columns and isinstance(columns, list):
            for col in columns:
                if isinstance(col, dict):
                    if col.get("is_primary_key", False):
                        return col.get("name", col.get("column_name"))
            # PK 없으면 첫 번째 컬럼
            if columns and isinstance(columns[0], dict):
                return columns[0].get("name", columns[0].get("column_name"))
        # columns가 Dict인 경우 (레거시 형식)
        elif columns and isinstance(columns, dict):
            for col_name, col_info in columns.items():
                if isinstance(col_info, dict) and col_info.get("is_primary_key", False):
                    return col_name
            # PK 없으면 첫 번째 컬럼
            return next(iter(columns.keys()), None)

        # TableInfo에 primary_keys 속성이 있는 경우
        if hasattr(table_info, 'primary_keys') and table_info.primary_keys:
            return table_info.primary_keys[0] if isinstance(table_info.primary_keys, list) else table_info.primary_keys

        return None

    def _compute_diff_internal(
        self,
        snapshot_a: Dict[str, Dict[str, Dict[str, Any]]],
        snapshot_b: Dict[str, Dict[str, Dict[str, Any]]],
    ) -> Dict[str, TableDiff]:
        """내부 디프 계산"""
        table_diffs: Dict[str, TableDiff] = {}

        all_tables = set(snapshot_a.keys()) | set(snapshot_b.keys())

        for table_name in all_tables:
            rows_a = snapshot_a.get(table_name, {})
            rows_b = snapshot_b.get(table_name, {})

            table_diff = TableDiff(table_name=table_name)

            all_row_ids = set(rows_a.keys()) | set(rows_b.keys())

            for row_id in all_row_ids:
                row_a = rows_a.get(row_id)
                row_b = rows_b.get(row_id)

                if row_a is None and row_b is not None:
                    # 추가됨
                    table_diff.rows_added.append(RowChange(
                        table=table_name,
                        row_id=row_id,
                        change_type=ChangeType.ADDED,
                        new_values=row_b,
                    ))
                elif row_a is not None and row_b is None:
                    # 삭제됨
                    table_diff.rows_deleted.append(RowChange(
                        table=table_name,
                        row_id=row_id,
                        change_type=ChangeType.DELETED,
                        old_values=row_a,
                    ))
                elif row_a != row_b:
                    # 수정됨
                    changed_columns = [
                        col for col in set(row_a.keys()) | set(row_b.keys())
                        if row_a.get(col) != row_b.get(col)
                    ]
                    table_diff.rows_modified.append(RowChange(
                        table=table_name,
                        row_id=row_id,
                        change_type=ChangeType.MODIFIED,
                        old_values=row_a,
                        new_values=row_b,
                        changed_columns=changed_columns,
                    ))

            table_diffs[table_name] = table_diff

        return table_diffs

    def _update_context_on_commit(self, version: DatasetVersion) -> None:
        """커밋 시 SharedContext 업데이트"""
        branch = self.current_branch
        version_id = version.version_id

        # 버전 목록 업데이트
        if branch not in self.context.dataset_versions:
            self.context.dataset_versions[branch] = []
        self.context.dataset_versions[branch].append(version_id)

        # 현재 버전 업데이트
        self.context.current_dataset_version = version_id

        # 스냅샷 저장
        self.context.dataset_snapshots[version_id] = {
            "commit_info": version.commit_info.to_dict(),
            "checksum": version.checksum,
            "tables": list(version.snapshot.keys()),
        }

        # Evidence Chain 기록
        if self.context.evidence_chain:
            try:
                from ..autonomous.evidence_chain import EvidenceType

                evidence_type = getattr(
                    EvidenceType,
                    "VERSION_COMMIT",
                    EvidenceType.QUALITY_ASSESSMENT
                )

                self.context.evidence_chain.add_block(
                    phase="versioning",
                    agent="VersionManager",
                    evidence_type=evidence_type,
                    finding=f"Created version {version_id}: {version.commit_info.message}",
                    reasoning=f"Snapshot of {len(version.snapshot)} tables",
                    conclusion=(
                        f"+{version.commit_info.rows_added} "
                        f"~{version.commit_info.rows_modified} "
                        f"-{version.commit_info.rows_deleted} rows"
                    ),
                    metrics={
                        "version_id": version_id,
                        "tables_affected": version.commit_info.tables_affected,
                        "checksum": version.checksum,
                    },
                    confidence=0.95,
                )
            except Exception as e:
                logger.warning(f"Failed to record to evidence chain: {e}")

    def _restore_snapshot_to_context(self, version: DatasetVersion) -> None:
        """스냅샷을 SharedContext에 복원"""
        # 현재 버전 업데이트
        self.context.current_dataset_version = version.version_id

        # 테이블 데이터 복원 (sample_data에)
        for table_name, rows in version.snapshot.items():
            if table_name in self.context.tables:
                self.context.tables[table_name]["sample_data"] = list(rows.values())

        logger.info(f"Restored snapshot {version.version_id} to context")

    def get_version_summary(self) -> Dict[str, Any]:
        """버전 관리 요약"""
        return {
            "total_versions": len(self.versions),
            "current_branch": self.current_branch,
            "current_head": self.get_current_head(),
            "branches": list(self.branch_heads.keys()),
            "tags": self.list_tags(),
            "recent_commits": [
                c.to_dict() for c in self.log(limit=5)
            ],
        }
