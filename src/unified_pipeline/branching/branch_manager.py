"""
Branch Manager (v17.0)

Git 스타일 브랜치 관리:
- 브랜치 생성/삭제
- 브랜치 전환
- 브랜치 목록 및 상태 조회

사용 예시:
    manager = BranchManager(version_manager)
    manager.create_branch("feature/analysis")
    manager.switch_branch("feature/analysis")
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, TYPE_CHECKING
from datetime import datetime

if TYPE_CHECKING:
    from ..versioning.version_manager import VersionManager
    from ..shared_context import SharedContext

logger = logging.getLogger(__name__)


@dataclass
class BranchInfo:
    """브랜치 정보"""
    name: str
    created_at: str
    created_from: str  # 생성 시점의 버전 ID
    head: Optional[str] = None  # 현재 HEAD 버전 ID
    description: str = ""
    is_protected: bool = False  # main 등 보호 브랜치

    # 통계
    commit_count: int = 0
    ahead_of_main: int = 0
    behind_main: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "created_at": self.created_at,
            "created_from": self.created_from,
            "head": self.head,
            "description": self.description,
            "is_protected": self.is_protected,
            "commit_count": self.commit_count,
            "ahead_of_main": self.ahead_of_main,
            "behind_main": self.behind_main,
        }


@dataclass
class Branch:
    """브랜치 객체"""
    info: BranchInfo
    version_ids: List[str] = field(default_factory=list)

    @property
    def name(self) -> str:
        return self.info.name

    @property
    def head(self) -> Optional[str]:
        return self.version_ids[-1] if self.version_ids else None

    def add_commit(self, version_id: str) -> None:
        """커밋 추가"""
        self.version_ids.append(version_id)
        self.info.head = version_id
        self.info.commit_count = len(self.version_ids)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "info": self.info.to_dict(),
            "version_ids": self.version_ids,
        }


class BranchManager:
    """
    브랜치 관리자

    기능:
    - 브랜치 생성/삭제/목록
    - 브랜치 전환
    - 브랜치 비교
    """

    def __init__(
        self,
        version_manager: "VersionManager",
        context: Optional["SharedContext"] = None,
    ):
        """
        Args:
            version_manager: 버전 관리자
            context: SharedContext
        """
        self.version_manager = version_manager
        self.context = context

        # 브랜치 저장소
        self.branches: Dict[str, Branch] = {}

        # main 브랜치 초기화
        self._init_main_branch()

        logger.info("BranchManager initialized")

    def _init_main_branch(self) -> None:
        """main 브랜치 초기화"""
        main_info = BranchInfo(
            name="main",
            created_at=datetime.now().isoformat(),
            created_from="",
            description="Default branch",
            is_protected=True,
        )
        self.branches["main"] = Branch(info=main_info)

    def create_branch(
        self,
        name: str,
        from_branch: Optional[str] = None,
        from_version: Optional[str] = None,
        description: str = "",
    ) -> Branch:
        """
        새 브랜치 생성

        Args:
            name: 브랜치 이름
            from_branch: 기준 브랜치 (기본: 현재 브랜치)
            from_version: 기준 버전 (기본: HEAD)
            description: 브랜치 설명

        Returns:
            생성된 Branch 객체
        """
        if name in self.branches:
            raise ValueError(f"Branch already exists: {name}")

        # 기준점 결정
        if from_version:
            base_version = from_version
        elif from_branch:
            base_version = self.branches[from_branch].head
        else:
            current_branch = self.version_manager.current_branch
            base_version = self.version_manager.branch_heads.get(current_branch)

        # 브랜치 생성
        info = BranchInfo(
            name=name,
            created_at=datetime.now().isoformat(),
            created_from=base_version or "",
            head=base_version,
            description=description,
        )

        branch = Branch(info=info)
        if base_version:
            branch.version_ids = [base_version]

        self.branches[name] = branch

        # VersionManager에 브랜치 HEAD 등록
        self.version_manager.branch_heads[name] = base_version or ""

        # SharedContext 업데이트
        if self.context:
            if name not in self.context.branches:
                self.context.branches[name] = []
            if base_version:
                self.context.branches[name].append(base_version)

        logger.info(f"Created branch '{name}' from {base_version or 'empty'}")

        return branch

    def delete_branch(self, name: str, force: bool = False) -> bool:
        """
        브랜치 삭제

        Args:
            name: 브랜치 이름
            force: 강제 삭제 (머지되지 않은 변경 무시)

        Returns:
            성공 여부
        """
        if name not in self.branches:
            logger.error(f"Branch not found: {name}")
            return False

        branch = self.branches[name]

        # 보호 브랜치 확인
        if branch.info.is_protected:
            logger.error(f"Cannot delete protected branch: {name}")
            return False

        # 현재 브랜치 확인
        if self.version_manager.current_branch == name:
            logger.error(f"Cannot delete current branch: {name}")
            return False

        # 머지 여부 확인 (force가 아니면)
        if not force:
            main_head = self.branches.get("main", Branch(info=BranchInfo(
                name="main", created_at="", created_from=""
            ))).head

            if branch.head and branch.head != main_head:
                # TODO: 실제로는 머지 여부 확인 필요
                logger.warning(
                    f"Branch '{name}' has unmerged changes. Use force=True to delete."
                )

        # 삭제
        del self.branches[name]
        if name in self.version_manager.branch_heads:
            del self.version_manager.branch_heads[name]

        # SharedContext 업데이트
        if self.context and name in self.context.branches:
            del self.context.branches[name]

        logger.info(f"Deleted branch '{name}'")

        return True

    def switch_branch(self, name: str) -> bool:
        """
        브랜치 전환

        Args:
            name: 전환할 브랜치 이름

        Returns:
            성공 여부
        """
        if name not in self.branches:
            logger.error(f"Branch not found: {name}")
            return False

        branch = self.branches[name]

        # VersionManager 현재 브랜치 변경
        self.version_manager.current_branch = name

        # HEAD 버전으로 체크아웃
        if branch.head:
            self.version_manager.checkout(branch.head)

        # SharedContext 업데이트
        if self.context:
            self.context.current_branch = name

        logger.info(f"Switched to branch '{name}'")

        return True

    def get_branch(self, name: str) -> Optional[Branch]:
        """브랜치 조회"""
        return self.branches.get(name)

    def list_branches(self) -> List[BranchInfo]:
        """모든 브랜치 목록"""
        return [b.info for b in self.branches.values()]

    def get_current_branch(self) -> str:
        """현재 브랜치 이름"""
        return self.version_manager.current_branch

    def compare_branches(
        self,
        branch_a: str,
        branch_b: str,
    ) -> Dict[str, Any]:
        """
        두 브랜치 비교

        Args:
            branch_a: 기준 브랜치
            branch_b: 비교 브랜치

        Returns:
            비교 결과
        """
        if branch_a not in self.branches:
            raise ValueError(f"Branch not found: {branch_a}")
        if branch_b not in self.branches:
            raise ValueError(f"Branch not found: {branch_b}")

        a = self.branches[branch_a]
        b = self.branches[branch_b]

        # 공통 조상 찾기
        lca = self._find_lca(a.version_ids, b.version_ids)

        # 각 브랜치의 ahead/behind 계산
        a_ahead = 0
        b_ahead = 0

        if lca:
            try:
                a_ahead = a.version_ids.index(a.head) - a.version_ids.index(lca) if a.head and lca in a.version_ids else 0
            except (ValueError, AttributeError):
                a_ahead = 0

            try:
                b_ahead = b.version_ids.index(b.head) - b.version_ids.index(lca) if b.head and lca in b.version_ids else 0
            except (ValueError, AttributeError):
                b_ahead = 0

        # 버전 diff 계산
        version_diff = None
        if a.head and b.head:
            try:
                version_diff = self.version_manager.diff(a.head, b.head)
            except Exception:
                pass

        return {
            "branch_a": branch_a,
            "branch_b": branch_b,
            "lca": lca,
            "a_ahead": a_ahead,
            "b_ahead": b_ahead,
            "can_fast_forward": a_ahead == 0 and b_ahead > 0,
            "has_conflicts": version_diff.has_changes if version_diff else False,
            "diff_summary": version_diff.get_summary() if version_diff else None,
        }

    def _find_lca(
        self,
        versions_a: List[str],
        versions_b: List[str],
    ) -> Optional[str]:
        """
        Lowest Common Ancestor 찾기

        두 브랜치의 공통 조상 버전 ID 반환
        """
        set_a = set(versions_a)

        for version_id in reversed(versions_b):
            if version_id in set_a:
                return version_id

        return None

    def get_branch_summary(self) -> Dict[str, Any]:
        """브랜치 관리 요약"""
        return {
            "total_branches": len(self.branches),
            "current_branch": self.get_current_branch(),
            "branches": {
                name: branch.info.to_dict()
                for name, branch in self.branches.items()
            },
        }
