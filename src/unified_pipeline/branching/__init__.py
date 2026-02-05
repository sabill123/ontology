"""
Branching & Merging Module (v17.0)

Palantir Foundry 스타일 브랜치 및 머지:
- 브랜치 생성/삭제
- 3-way 머지
- 충돌 해결 (Consensus 기반)

사용 예시:
    manager = BranchManager(version_manager, context)

    # 브랜치 생성
    manager.create_branch("feature/new-analysis")

    # 브랜치 전환
    manager.switch_branch("feature/new-analysis")

    # 머지
    result = manager.merge("feature/new-analysis", "main")
"""

from .branch_manager import (
    BranchManager,
    Branch,
    BranchInfo,
)

from .merge_engine import (
    MergeEngine,
    MergeResult,
    MergeStrategy,
)

from .conflict_resolver import (
    ConflictResolver,
    Conflict,
    ConflictResolution,
    ResolutionStrategy,
)

__all__ = [
    # Branch Manager
    "BranchManager",
    "Branch",
    "BranchInfo",
    # Merge Engine
    "MergeEngine",
    "MergeResult",
    "MergeStrategy",
    # Conflict Resolver
    "ConflictResolver",
    "Conflict",
    "ConflictResolution",
    "ResolutionStrategy",
]
