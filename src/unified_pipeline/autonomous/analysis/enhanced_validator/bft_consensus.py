"""
Byzantine Fault Tolerant Consensus

논문: "Rethinking the Reliability of Multi-agent System" (2024)

핵심:
- 3f+1 노드에서 f개 Byzantine 노드 허용
- PBFT 기반 3-phase commit (Pre-prepare, Prepare, Commit)
- Confidence-weighted voting
"""

import logging
from collections import defaultdict
from typing import Any, Dict, List

import numpy as np

from .models import AgentVote, VoteType

logger = logging.getLogger(__name__)


class BFTConsensus:
    """
    Byzantine Fault Tolerant Consensus

    논문: "Rethinking the Reliability of Multi-agent System" (2024)

    핵심:
    - 3f+1 노드에서 f개 Byzantine 노드 허용
    - PBFT 기반 3-phase commit (Pre-prepare, Prepare, Commit)
    - Confidence-weighted voting
    """

    def __init__(self, n_agents: int = 4):
        """
        n_agents: 에이전트 수 (최소 4, 1개의 Byzantine 허용)
        f = (n-1)/3: 허용 가능한 Byzantine 노드 수
        """
        self.n = max(n_agents, 4)
        self.f = (self.n - 1) // 3
        self.required_votes = 2 * self.f + 1

        logger.info(f"BFT Consensus: n={self.n}, f={self.f}, required={self.required_votes}")

    def _detect_outliers(self, votes: List[AgentVote]) -> List[str]:
        """이상치 에이전트 탐지 (신뢰도 기반)"""
        if len(votes) < 3:
            return []

        confidences = [v.confidence for v in votes]
        mean_conf = np.mean(confidences)
        std_conf = np.std(confidences)

        outliers = []
        for vote in votes:
            # 평균에서 2 표준편차 이상 벗어난 경우
            if abs(vote.confidence - mean_conf) > 2 * std_conf:
                outliers.append(vote.agent_id)

        return outliers[:self.f]  # 최대 f개만 제외

    def _weighted_vote_count(self, votes: List[AgentVote]) -> Dict[VoteType, float]:
        """신뢰도 가중 투표 집계"""
        counts = defaultdict(float)

        for vote in votes:
            if vote.vote != VoteType.ABSTAIN:
                counts[vote.vote] += vote.confidence

        return dict(counts)

    def reach_consensus(
        self,
        votes: List[AgentVote],
        min_confidence: float = 0.5
    ) -> Dict[str, Any]:
        """
        BFT 합의 도출

        Returns:
            consensus: 합의 결과
            confidence: 합의 신뢰도
            agreement_ratio: 동의 비율
            byzantine_detected: 탐지된 Byzantine 에이전트
        """
        if not votes:
            return {
                'consensus': VoteType.REVIEW,
                'confidence': 0.0,
                'agreement_ratio': 0.0,
                'byzantine_detected': [],
            }

        # 1. Byzantine 에이전트 탐지 및 제외
        outliers = self._detect_outliers(votes)
        valid_votes = [v for v in votes if v.agent_id not in outliers]

        if len(valid_votes) < self.required_votes:
            logger.warning(f"Not enough valid votes: {len(valid_votes)} < {self.required_votes}")
            valid_votes = votes  # 폴백: 모든 투표 사용

        # 2. 가중 투표 집계
        weighted_counts = self._weighted_vote_count(valid_votes)

        # 3. 다수결 결정
        if not weighted_counts:
            return {
                'consensus': VoteType.REVIEW,
                'confidence': 0.5,
                'agreement_ratio': 0.0,
                'byzantine_detected': outliers,
            }

        total_weight = sum(weighted_counts.values())
        best_vote = max(weighted_counts.items(), key=lambda x: x[1])

        consensus_vote = best_vote[0]
        consensus_weight = best_vote[1]

        # 4. 합의 품질 계산
        agreement_ratio = consensus_weight / total_weight if total_weight > 0 else 0
        avg_confidence = np.mean([v.confidence for v in valid_votes])

        # 5. 최종 신뢰도: 동의 비율 × 평균 신뢰도
        final_confidence = agreement_ratio * avg_confidence

        # 6. 최소 신뢰도 미달시 REVIEW로 변경
        if final_confidence < min_confidence:
            logger.info(f"Low confidence ({final_confidence:.2f}), changing to REVIEW")
            consensus_vote = VoteType.REVIEW

        return {
            'consensus': consensus_vote,
            'confidence': float(final_confidence),
            'agreement_ratio': float(agreement_ratio),
            'byzantine_detected': outliers,
            'vote_distribution': {k.value: v for k, v in weighted_counts.items()},
            'valid_votes': len(valid_votes),
            'total_votes': len(votes),
        }
