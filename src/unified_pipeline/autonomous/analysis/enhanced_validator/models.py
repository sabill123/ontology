"""
Data models and enums for the Enhanced Validation System.

Small dataclass/enum definitions used across the validator package.
"""

from dataclasses import dataclass
from enum import Enum


@dataclass
class StatisticalTestResult:
    """통계 검정 결과"""
    test_name: str
    statistic: float
    p_value: float
    passed: bool
    interpretation: str


@dataclass
class KGQualityScore:
    """KG 품질 점수"""
    metric_name: str
    score: float
    max_score: float = 1.0
    interpretation: str = ""


class VoteType(Enum):
    APPROVE = "approve"
    REJECT = "reject"
    REVIEW = "review"
    ABSTAIN = "abstain"


@dataclass
class AgentVote:
    """에이전트 투표"""
    agent_id: str
    vote: VoteType
    confidence: float
    reasoning: str = ""
    round: int = 1
