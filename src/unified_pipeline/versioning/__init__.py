"""
Dataset Versioning Module (v17.0)

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
"""

from .version_manager import (
    VersionManager,
    DatasetVersion,
    VersionDiff,
    CommitInfo,
)

__all__ = [
    "VersionManager",
    "DatasetVersion",
    "VersionDiff",
    "CommitInfo",
]
