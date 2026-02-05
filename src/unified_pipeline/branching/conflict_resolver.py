"""
Conflict Resolver (v17.0)

Multi-agent Consensus 기반 충돌 해결:
- 다양한 해결 전략
- 에이전트 투표 기반 결정
- Evidence Chain 기록

사용 예시:
    resolver = ConflictResolver(context)
    resolution = resolver.resolve(conflict, strategy=ResolutionStrategy.CONSENSUS)
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, TYPE_CHECKING
from datetime import datetime
from enum import Enum

if TYPE_CHECKING:
    from ..shared_context import SharedContext
    from .merge_engine import MergeConflict, RowConflict

logger = logging.getLogger(__name__)


class ResolutionStrategy(str, Enum):
    """충돌 해결 전략"""
    SOURCE_WINS = "source_wins"  # 소스 브랜치 우선
    TARGET_WINS = "target_wins"  # 타겟 브랜치 우선
    LATEST_WINS = "latest_wins"  # 최신 변경 우선
    CONSENSUS = "consensus"      # 에이전트 합의
    MANUAL = "manual"            # 수동 해결
    CUSTOM = "custom"            # 커스텀 로직


@dataclass
class Conflict:
    """충돌 정보"""
    conflict_id: str
    table: str
    row_id: str
    column: str
    base_value: Any
    source_value: Any
    target_value: Any
    source_branch: str
    target_branch: str

    # 메타데이터
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    severity: str = "medium"  # low, medium, high

    def to_dict(self) -> Dict[str, Any]:
        return {
            "conflict_id": self.conflict_id,
            "table": self.table,
            "row_id": self.row_id,
            "column": self.column,
            "base_value": self.base_value,
            "source_value": self.source_value,
            "target_value": self.target_value,
            "source_branch": self.source_branch,
            "target_branch": self.target_branch,
            "severity": self.severity,
        }


@dataclass
class ConflictResolution:
    """충돌 해결 결과"""
    conflict_id: str
    resolved_value: Any
    strategy_used: ResolutionStrategy
    confidence: float = 0.0

    # 해결 근거
    reasoning: str = ""
    agent_votes: Dict[str, Any] = field(default_factory=dict)

    # 메타데이터
    resolved_at: str = field(default_factory=lambda: datetime.now().isoformat())
    resolved_by: str = "system"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "conflict_id": self.conflict_id,
            "resolved_value": self.resolved_value,
            "strategy_used": self.strategy_used.value,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "agent_votes": self.agent_votes,
            "resolved_at": self.resolved_at,
            "resolved_by": self.resolved_by,
        }


class ConflictResolver:
    """
    충돌 해결기

    기능:
    - 다양한 해결 전략 지원
    - Multi-agent Consensus 기반 결정
    - Evidence Chain 기록
    """

    def __init__(
        self,
        context: Optional["SharedContext"] = None,
    ):
        """
        Args:
            context: SharedContext (Consensus 및 Evidence Chain 연동용)
        """
        self.context = context

        # 해결 히스토리
        self.resolution_history: List[ConflictResolution] = []

        logger.info("ConflictResolver initialized")

    def resolve(
        self,
        conflict: Conflict,
        strategy: ResolutionStrategy = ResolutionStrategy.SOURCE_WINS,
        custom_value: Any = None,
    ) -> ConflictResolution:
        """
        충돌 해결

        Args:
            conflict: 충돌 정보
            strategy: 해결 전략
            custom_value: 커스텀 값 (CUSTOM/MANUAL 전략 시)

        Returns:
            ConflictResolution: 해결 결과
        """
        resolved_value = None
        reasoning = ""
        confidence = 0.0
        agent_votes = {}

        if strategy == ResolutionStrategy.SOURCE_WINS:
            resolved_value = conflict.source_value
            reasoning = f"Source branch ({conflict.source_branch}) value selected"
            confidence = 0.8

        elif strategy == ResolutionStrategy.TARGET_WINS:
            resolved_value = conflict.target_value
            reasoning = f"Target branch ({conflict.target_branch}) value selected"
            confidence = 0.8

        elif strategy == ResolutionStrategy.LATEST_WINS:
            # 최신 변경 결정 (실제로는 타임스탬프 비교 필요)
            # 여기서는 소스를 최신으로 가정
            resolved_value = conflict.source_value
            reasoning = "Latest change selected (assumed source is newer)"
            confidence = 0.7

        elif strategy == ResolutionStrategy.CONSENSUS:
            resolved_value, reasoning, confidence, agent_votes = (
                self._resolve_by_consensus(conflict)
            )

        elif strategy == ResolutionStrategy.MANUAL:
            if custom_value is None:
                raise ValueError("Manual resolution requires custom_value")
            resolved_value = custom_value
            reasoning = "Manually resolved by user"
            confidence = 1.0

        elif strategy == ResolutionStrategy.CUSTOM:
            if custom_value is None:
                raise ValueError("Custom resolution requires custom_value")
            resolved_value = custom_value
            reasoning = "Custom value provided"
            confidence = 0.9

        # 결과 생성
        resolution = ConflictResolution(
            conflict_id=conflict.conflict_id,
            resolved_value=resolved_value,
            strategy_used=strategy,
            confidence=confidence,
            reasoning=reasoning,
            agent_votes=agent_votes,
        )

        # 히스토리 기록
        self.resolution_history.append(resolution)

        # Evidence Chain 기록
        self._record_to_evidence_chain(conflict, resolution)

        logger.info(
            f"Resolved conflict {conflict.conflict_id} using {strategy.value}: "
            f"{resolved_value} (confidence: {confidence:.0%})"
        )

        return resolution

    def resolve_batch(
        self,
        conflicts: List[Conflict],
        strategy: ResolutionStrategy = ResolutionStrategy.SOURCE_WINS,
    ) -> List[ConflictResolution]:
        """
        여러 충돌 일괄 해결

        Args:
            conflicts: 충돌 목록
            strategy: 해결 전략

        Returns:
            해결 결과 목록
        """
        return [
            self.resolve(conflict, strategy)
            for conflict in conflicts
        ]

    def _resolve_by_consensus(
        self,
        conflict: Conflict,
    ) -> tuple:
        """
        Multi-agent Consensus 기반 해결

        Returns:
            (resolved_value, reasoning, confidence, agent_votes)
        """
        # Consensus Engine 사용 가능 여부 확인
        if not self.context:
            logger.warning("No context available for consensus resolution")
            return (
                conflict.source_value,
                "Fallback to source (no consensus available)",
                0.5,
                {},
            )

        # 간소화된 에이전트 투표 시뮬레이션
        # 실제 구현에서는 PipelineConsensusEngine 사용
        agent_votes = self._simulate_agent_votes(conflict)

        # 투표 집계
        source_votes = sum(
            1 for vote in agent_votes.values()
            if vote.get("choice") == "source"
        )
        target_votes = sum(
            1 for vote in agent_votes.values()
            if vote.get("choice") == "target"
        )
        total_votes = source_votes + target_votes

        if total_votes == 0:
            return (
                conflict.source_value,
                "No agent votes received, defaulting to source",
                0.5,
                agent_votes,
            )

        if source_votes > target_votes:
            resolved_value = conflict.source_value
            confidence = source_votes / total_votes
            reasoning = (
                f"Consensus: {source_votes}/{total_votes} agents chose source value"
            )
        elif target_votes > source_votes:
            resolved_value = conflict.target_value
            confidence = target_votes / total_votes
            reasoning = (
                f"Consensus: {target_votes}/{total_votes} agents chose target value"
            )
        else:
            # 동률 시 소스 우선
            resolved_value = conflict.source_value
            confidence = 0.5
            reasoning = "Tie in consensus, defaulting to source"

        return resolved_value, reasoning, confidence, agent_votes

    def _simulate_agent_votes(
        self,
        conflict: Conflict,
    ) -> Dict[str, Dict[str, Any]]:
        """
        에이전트 투표 시뮬레이션

        실제 구현에서는 각 에이전트가 conflict를 평가하고 투표
        """
        # 가상의 에이전트 투표
        agents = [
            "DataQualityAgent",
            "SchemaExpert",
            "DomainAnalyst",
            "ConsistencyChecker",
            "DataSteward",
        ]

        votes = {}

        for agent in agents:
            # 간단한 휴리스틱 기반 투표
            # 실제로는 각 에이전트가 conflict 분석
            choice = self._agent_heuristic_vote(agent, conflict)
            confidence = 0.7 + (hash(agent) % 30) / 100  # 0.7-0.99

            votes[agent] = {
                "choice": choice,
                "confidence": confidence,
                "reasoning": f"{agent} prefers {choice} based on analysis",
            }

        return votes

    def _agent_heuristic_vote(
        self,
        agent: str,
        conflict: Conflict,
    ) -> str:
        """에이전트별 휴리스틱 투표"""
        # 간단한 휴리스틱
        # DataQualityAgent: null이 아닌 값 선호
        if agent == "DataQualityAgent":
            if conflict.source_value is None:
                return "target"
            if conflict.target_value is None:
                return "source"

        # SchemaExpert: 타입 일관성 선호
        if agent == "SchemaExpert":
            source_type = type(conflict.source_value).__name__
            target_type = type(conflict.target_value).__name__
            base_type = type(conflict.base_value).__name__ if conflict.base_value else None

            if base_type and source_type == base_type:
                return "source"
            if base_type and target_type == base_type:
                return "target"

        # 기본: 해시 기반 랜덤
        combined = f"{agent}:{conflict.conflict_id}"
        return "source" if hash(combined) % 2 == 0 else "target"

    def _record_to_evidence_chain(
        self,
        conflict: Conflict,
        resolution: ConflictResolution,
    ) -> None:
        """Evidence Chain에 기록"""
        if not self.context or not self.context.evidence_chain:
            return

        try:
            from ..autonomous.evidence_chain import EvidenceType

            evidence_type = getattr(
                EvidenceType,
                "CONFLICT_RESOLUTION",
                EvidenceType.QUALITY_ASSESSMENT
            )

            self.context.evidence_chain.add_block(
                phase="merge",
                agent="ConflictResolver",
                evidence_type=evidence_type,
                finding=(
                    f"Resolved conflict in {conflict.table}.{conflict.column} "
                    f"(row {conflict.row_id})"
                ),
                reasoning=resolution.reasoning,
                conclusion=f"Resolved to: {resolution.resolved_value}",
                metrics={
                    "conflict_id": conflict.conflict_id,
                    "strategy": resolution.strategy_used.value,
                    "confidence": resolution.confidence,
                    "source_value": str(conflict.source_value),
                    "target_value": str(conflict.target_value),
                    "resolved_value": str(resolution.resolved_value),
                },
                confidence=resolution.confidence,
            )
        except Exception as e:
            logger.warning(f"Failed to record to evidence chain: {e}")

    def convert_merge_conflicts(
        self,
        merge_conflicts: List["MergeConflict"],
    ) -> List[Conflict]:
        """MergeConflict를 Conflict로 변환"""
        conflicts = []

        for mc in merge_conflicts:
            for rc in mc.row_conflicts:
                conflicts.append(Conflict(
                    conflict_id=f"{mc.conflict_id}_{rc.row_id}_{rc.column}",
                    table=rc.table,
                    row_id=rc.row_id,
                    column=rc.column,
                    base_value=rc.base_value,
                    source_value=rc.source_value,
                    target_value=rc.target_value,
                    source_branch=mc.source_branch,
                    target_branch=mc.target_branch,
                ))

        return conflicts

    def get_resolution_summary(self) -> Dict[str, Any]:
        """해결 요약"""
        strategy_counts = {}
        total_confidence = 0.0

        for resolution in self.resolution_history:
            strategy = resolution.strategy_used.value
            strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1
            total_confidence += resolution.confidence

        return {
            "total_resolutions": len(self.resolution_history),
            "strategy_counts": strategy_counts,
            "avg_confidence": (
                total_confidence / len(self.resolution_history)
                if self.resolution_history else 0.0
            ),
            "recent_resolutions": [
                r.to_dict() for r in self.resolution_history[-5:]
            ],
        }
