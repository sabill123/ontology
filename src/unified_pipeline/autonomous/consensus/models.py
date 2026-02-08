"""
Consensus Models

합의 시스템에서 사용하는 데이터 모델들
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Any, Optional


class ConsensusType(str, Enum):
    """합의 결과 유형"""
    ACCEPTED = "accepted"           # 완전 승인
    PROVISIONAL = "provisional"     # 임시 승인 (추후 재검토)
    REJECTED = "rejected"           # 거부
    UNDECIDED = "undecided"         # 미결정 (토론 계속)
    ESCALATED = "escalated"         # 사람 검토 필요
    NO_AGENTS = "no_agents"         # v14.0: 에이전트 없음 (기본 결과 반환)


class VoteType(str, Enum):
    """투표 유형"""
    APPROVE = "approve"
    REJECT = "reject"
    ABSTAIN = "abstain"


class ChallengeStatus(str, Enum):
    """Challenge 상태"""
    PENDING = "pending"             # 대기 중
    ACTIVE = "active"               # 진행 중
    RESOLVED = "resolved"           # 해결됨
    ESCALATED = "escalated"         # Human escalation


class EvidenceType(str, Enum):
    """증거 유형"""
    TDA_RESULT = "tda_result"       # TDA 분석 결과
    STATISTICAL = "statistical"     # 통계 분석 결과
    SEMANTIC = "semantic"           # 시맨틱 분석 결과
    HISTORICAL = "historical"       # 과거 결정 기록
    EXTERNAL = "external"           # 외부 소스


@dataclass
class AgentOpinion:
    """
    에이전트의 의견 (Soft Consensus)

    이진 투표가 아닌 신뢰도 기반 평가
    """
    agent_id: str
    agent_type: str
    agent_name: str

    # Confidence scores (0.0 - 1.0)
    confidence: float = 0.5         # 제안에 대한 신뢰도
    uncertainty: float = 0.2        # 불확실성
    doubt: float = 0.1              # 의심 수준
    evidence_weight: float = 0.5    # 근거 강도

    # 의견 내용
    reasoning: str = ""
    key_observations: List[str] = field(default_factory=list)
    concerns: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)

    # 다른 에이전트 참조
    references_to: List[str] = field(default_factory=list)    # @멘션한 에이전트들
    agrees_with: Dict[str, float] = field(default_factory=dict)  # {agent_id: 동의 정도}
    disagrees_with: Dict[str, float] = field(default_factory=dict)  # {agent_id: 반대 정도}

    # 메타데이터
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    round_number: int = 0

    @property
    def adjusted_confidence(self) -> float:
        """
        v23.0: 불확실성과 의심을 반영한 조정 신뢰도
        - v5.2에서 너무 공격적으로 감소시켜 0.07 같은 극단적 결과 발생
        - v23.0: 더 완화된 계산 + floor 보장

        기존 문제: conf=0.65, unc=0.8, doubt=0.6 → 0.27 (너무 낮음)
        v23.0: 최소 confidence의 50%는 보장 + 감쇠 factor 완화
        """
        # v23.0: uncertainty/doubt 영향을 1/3로 줄임 (기존 1/2)
        # 또한 최소값을 confidence의 40%로 보장
        adjusted = self.confidence * (1 - self.uncertainty / 3) * (1 - self.doubt / 3)
        min_floor = self.confidence * 0.4  # confidence의 40%는 최소 보장
        return max(min_floor, adjusted)

    @property
    def legacy_vote(self) -> str:
        """
        v5.2: 레거시 호환용 이진 투표 변환
        - approve 임계값: 0.6 -> 0.45 (더 관대)
        - reject 임계값: 0.3 -> 0.2 (reject 줄임)
        """
        if self.adjusted_confidence >= 0.45:
            return "approve"
        elif self.adjusted_confidence <= 0.2:
            return "reject"
        else:
            return "abstain"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "agent_type": self.agent_type,
            "agent_name": self.agent_name,
            "confidence": self.confidence,
            "uncertainty": self.uncertainty,
            "doubt": self.doubt,
            "evidence_weight": self.evidence_weight,
            "adjusted_confidence": self.adjusted_confidence,
            "reasoning": self.reasoning,
            "key_observations": self.key_observations,
            "concerns": self.concerns,
            "suggestions": self.suggestions,
            "references_to": self.references_to,
            "agrees_with": self.agrees_with,
            "disagrees_with": self.disagrees_with,
            "timestamp": self.timestamp,
            "round_number": self.round_number,
            "legacy_vote": self.legacy_vote,
        }


@dataclass
class DiscussionMessage:
    """토론 메시지"""
    agent_id: str
    agent_name: str
    content: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    round_number: int = 0
    opinion: Optional[AgentOpinion] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "agent_name": self.agent_name,
            "content": self.content,
            "timestamp": self.timestamp,
            "round_number": self.round_number,
            "opinion": self.opinion.to_dict() if self.opinion else None,
        }


@dataclass
class DiscussionRound:
    """토론 라운드"""
    round_number: int
    messages: List[DiscussionMessage] = field(default_factory=list)
    opinions: Dict[str, AgentOpinion] = field(default_factory=dict)  # {agent_id: opinion}
    combined_confidence: float = 0.0
    agreement_level: float = 0.0
    started_at: str = field(default_factory=lambda: datetime.now().isoformat())
    completed_at: Optional[str] = None

    def add_message(self, message: DiscussionMessage) -> None:
        """메시지 추가"""
        self.messages.append(message)
        if message.opinion:
            self.opinions[message.agent_id] = message.opinion

    def calculate_metrics(self) -> None:
        """라운드 메트릭 계산"""
        if not self.opinions:
            return

        # Combined confidence (가중 평균)
        total_weight = 0.0
        weighted_sum = 0.0

        for opinion in self.opinions.values():
            weight = opinion.evidence_weight
            weighted_sum += opinion.adjusted_confidence * weight
            total_weight += weight

        self.combined_confidence = weighted_sum / total_weight if total_weight > 0 else 0.0

        # Agreement level (의견 분산)
        if len(self.opinions) > 1:
            confidences = [o.adjusted_confidence for o in self.opinions.values()]
            mean_conf = sum(confidences) / len(confidences)
            variance = sum((c - mean_conf) ** 2 for c in confidences) / len(confidences)
            self.agreement_level = 1.0 - min(1.0, variance * 4)  # 분산이 0.25 이상이면 동의도 0
        else:
            self.agreement_level = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "round_number": self.round_number,
            "messages": [m.to_dict() for m in self.messages],
            "opinions": {k: v.to_dict() for k, v in self.opinions.items()},
            "combined_confidence": self.combined_confidence,
            "agreement_level": self.agreement_level,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
        }


@dataclass
class ConsensusState:
    """
    합의 상태

    전체 토론 진행 상황과 최종 결과를 관리
    """
    topic_id: str                   # 토론 주제 ID (예: todo_id)
    topic_name: str                 # 토론 주제명
    topic_type: str                 # 주제 유형 (예: "governance_decision")

    # 토론 데이터
    rounds: List[DiscussionRound] = field(default_factory=list)
    current_round: int = 0

    # 전체 의견 집계
    all_opinions: Dict[str, AgentOpinion] = field(default_factory=dict)

    # 메트릭
    combined_confidence: float = 0.0
    agreement_level: float = 0.0

    # 상태
    started_at: str = field(default_factory=lambda: datetime.now().isoformat())
    completed_at: Optional[str] = None
    result: Optional[ConsensusType] = None

    # v5.2: Thresholds 조정 - 더 관대한 승인
    acceptance_threshold: float = 0.55      # v5.2: 0.75 -> 0.55 (더 쉬운 승인)
    rejection_threshold: float = 0.2        # v5.2: 0.3 -> 0.2 (reject 줄임)
    provisional_threshold: float = 0.35     # v5.2: 0.5 -> 0.35 (provisional 범위 확대)
    min_agreement: float = 0.4              # v5.2: 0.6 -> 0.4 (동의 요건 완화)

    def start_new_round(self) -> DiscussionRound:
        """새 라운드 시작"""
        self.current_round += 1
        new_round = DiscussionRound(round_number=self.current_round)
        self.rounds.append(new_round)
        return new_round

    def add_opinion(self, opinion: AgentOpinion) -> None:
        """의견 추가"""
        opinion.round_number = self.current_round
        self.all_opinions[opinion.agent_id] = opinion

        if self.rounds:
            # 현재 라운드에도 추가
            current = self.rounds[-1]
            current.opinions[opinion.agent_id] = opinion

    def calculate_consensus(self) -> None:
        """합의 계산"""
        if not self.all_opinions:
            return

        # Combined confidence (가중 평균)
        total_weight = 0.0
        weighted_sum = 0.0

        for opinion in self.all_opinions.values():
            weight = opinion.evidence_weight
            weighted_sum += opinion.adjusted_confidence * weight
            total_weight += weight

        self.combined_confidence = weighted_sum / total_weight if total_weight > 0 else 0.0

        # Agreement level
        if len(self.all_opinions) > 1:
            confidences = [o.adjusted_confidence for o in self.all_opinions.values()]
            mean_conf = sum(confidences) / len(confidences)
            variance = sum((c - mean_conf) ** 2 for c in confidences) / len(confidences)
            self.agreement_level = 1.0 - min(1.0, variance * 4)
        else:
            self.agreement_level = 1.0

    def check_consensus(self) -> tuple[bool, ConsensusType]:
        """
        합의 도달 여부 확인

        Returns:
            (도달여부, 결과유형)
        """
        self.calculate_consensus()

        # Accepted: 높은 신뢰도 + 높은 동의
        if (self.combined_confidence >= self.acceptance_threshold and
            self.agreement_level >= self.min_agreement):
            self.result = ConsensusType.ACCEPTED
            return True, ConsensusType.ACCEPTED

        # Rejected: 낮은 신뢰도 + 높은 동의
        if (self.combined_confidence < self.rejection_threshold and
            self.agreement_level >= self.min_agreement):
            self.result = ConsensusType.REJECTED
            return True, ConsensusType.REJECTED

        # Provisional: 중간 신뢰도
        if (self.provisional_threshold <= self.combined_confidence < self.acceptance_threshold):
            self.result = ConsensusType.PROVISIONAL
            return True, ConsensusType.PROVISIONAL

        # Undecided: 합의 미도달
        return False, ConsensusType.UNDECIDED

    def get_summary(self) -> Dict[str, Any]:
        """합의 상태 요약"""
        return {
            "topic_id": self.topic_id,
            "topic_name": self.topic_name,
            "combined_confidence": self.combined_confidence,
            "agreement_level": self.agreement_level,
            "total_opinions": len(self.all_opinions),
            "rounds_completed": self.current_round,
            "result": self.result.value if self.result else None,
            "opinions": {
                agent_id: {
                    "confidence": op.confidence,
                    "adjusted": op.adjusted_confidence,
                    "vote": op.legacy_vote,
                }
                for agent_id, op in self.all_opinions.items()
            }
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "topic_id": self.topic_id,
            "topic_name": self.topic_name,
            "topic_type": self.topic_type,
            "rounds": [r.to_dict() for r in self.rounds],
            "current_round": self.current_round,
            "all_opinions": {k: v.to_dict() for k, v in self.all_opinions.items()},
            "combined_confidence": self.combined_confidence,
            "agreement_level": self.agreement_level,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "result": self.result.value if self.result else None,
        }


@dataclass
class ConsensusResult:
    """합의 결과 (v2.0 - EvidenceBundle 지원)"""
    success: bool
    result_type: ConsensusType
    combined_confidence: float
    agreement_level: float
    total_rounds: int

    final_decision: Optional[str] = None
    reasoning: str = ""
    dissenting_views: List[Dict[str, Any]] = field(default_factory=list)

    consensus_state: Optional[ConsensusState] = None

    # v2.0: EvidenceBundle 지원 (감사 추적용)
    evidence_bundle: Optional[Any] = None  # EvidenceBundle type (optional import)

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "success": self.success,
            "result_type": self.result_type.value,
            "combined_confidence": self.combined_confidence,
            "agreement_level": self.agreement_level,
            "total_rounds": self.total_rounds,
            "final_decision": self.final_decision,
            "reasoning": self.reasoning,
            "dissenting_views": self.dissenting_views,
        }

        # v2.0: EvidenceBundle 요약 포함
        if self.evidence_bundle:
            try:
                result["evidence_summary"] = self.evidence_bundle.to_summary()
            except Exception:
                result["evidence_summary"] = {"available": True}

        return result


# ============================================================
# Debate Protocol Models (Phase 3)
# ============================================================

@dataclass
class DebateEvidence:
    """
    토론에서 제시되는 증거

    TDA 결과, 통계 분석, 시맨틱 분석 등 다양한 증거 유형 지원
    """
    evidence_id: str
    evidence_type: EvidenceType
    presenter_id: str                    # 증거를 제시한 에이전트
    data: Dict[str, Any] = field(default_factory=dict)
    summary: str = ""
    weight: float = 0.5                  # 증거 가중치 (0.0 - 1.0)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    # 증거 검증 상태
    verified: bool = False
    verification_details: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "evidence_id": self.evidence_id,
            "evidence_type": self.evidence_type.value,
            "presenter_id": self.presenter_id,
            "data": self.data,
            "summary": self.summary,
            "weight": self.weight,
            "timestamp": self.timestamp,
            "verified": self.verified,
            "verification_details": self.verification_details,
        }


@dataclass
class Challenge:
    """
    토론 Challenge

    반대 의견 에이전트가 기존 투표에 이의를 제기할 때 사용
    """
    challenge_id: str
    challenger_id: str                   # Challenge를 제기한 에이전트
    challenger_name: str
    target_vote: str                     # 대상 투표 (예: "approve", "reject")
    target_agent_id: Optional[str] = None  # 특정 에이전트 투표 대상 (None이면 전체)
    reason: str = ""

    # Challenge 상태
    status: ChallengeStatus = ChallengeStatus.PENDING
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    resolved_at: Optional[str] = None
    resolution: Optional[str] = None     # 해결 결과

    # 관련 증거
    supporting_evidence: List[DebateEvidence] = field(default_factory=list)
    counter_evidence: List[DebateEvidence] = field(default_factory=list)

    # 토론 결과
    revote_result: Optional[Dict[str, Any]] = None

    def add_evidence(self, evidence: DebateEvidence, is_supporting: bool = True) -> None:
        """증거 추가"""
        if is_supporting:
            self.supporting_evidence.append(evidence)
        else:
            self.counter_evidence.append(evidence)

    def resolve(self, resolution: str, revote_result: Optional[Dict[str, Any]] = None) -> None:
        """Challenge 해결"""
        self.status = ChallengeStatus.RESOLVED
        self.resolved_at = datetime.now().isoformat()
        self.resolution = resolution
        self.revote_result = revote_result

    def escalate(self, reason: str) -> None:
        """Human escalation"""
        self.status = ChallengeStatus.ESCALATED
        self.resolved_at = datetime.now().isoformat()
        self.resolution = f"Escalated: {reason}"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "challenge_id": self.challenge_id,
            "challenger_id": self.challenger_id,
            "challenger_name": self.challenger_name,
            "target_vote": self.target_vote,
            "target_agent_id": self.target_agent_id,
            "reason": self.reason,
            "status": self.status.value,
            "created_at": self.created_at,
            "resolved_at": self.resolved_at,
            "resolution": self.resolution,
            "supporting_evidence": [e.to_dict() for e in self.supporting_evidence],
            "counter_evidence": [e.to_dict() for e in self.counter_evidence],
            "revote_result": self.revote_result,
        }


@dataclass
class DebateRound:
    """
    Debate Protocol 라운드

    일반 토론 라운드와 달리, Challenge 기반 구조화된 토론
    """
    round_id: str
    round_number: int
    challenge: Optional[Challenge] = None

    # 라운드 타임아웃 (v18.0: 10분으로 증가)
    timeout_seconds: float = 0  # v18.0: 타임아웃 비활성화 (무제한)
    started_at: str = field(default_factory=lambda: datetime.now().isoformat())
    deadline: Optional[str] = None
    completed_at: Optional[str] = None
    timed_out: bool = False

    # 참여자 및 발언
    participants: List[str] = field(default_factory=list)
    statements: List[Dict[str, Any]] = field(default_factory=list)

    # 증거 제시
    evidence_presented: List[DebateEvidence] = field(default_factory=list)

    # 재투표 결과
    revote_opinions: Dict[str, AgentOpinion] = field(default_factory=dict)
    revote_combined_confidence: float = 0.0

    def __post_init__(self):
        """Initialize deadline based on timeout (v18.0: 0 = 무제한)"""
        if self.deadline is None:
            from datetime import timedelta
            start = datetime.fromisoformat(self.started_at)
            # v18.0: timeout_seconds가 0이면 무제한 (100년 후)
            if self.timeout_seconds and self.timeout_seconds > 0:
                deadline = start + timedelta(seconds=self.timeout_seconds)
            else:
                deadline = start + timedelta(days=36500)  # 100년 = 무제한
            self.deadline = deadline.isoformat()

    def add_statement(self, agent_id: str, agent_name: str, content: str,
                      statement_type: str = "argument") -> None:
        """발언 추가"""
        self.statements.append({
            "agent_id": agent_id,
            "agent_name": agent_name,
            "content": content,
            "type": statement_type,
            "timestamp": datetime.now().isoformat(),
        })

    def add_evidence(self, evidence: DebateEvidence) -> None:
        """증거 추가"""
        self.evidence_presented.append(evidence)

    def check_timeout(self) -> bool:
        """타임아웃 체크"""
        if self.completed_at:
            return False

        now = datetime.now()
        deadline = datetime.fromisoformat(self.deadline)
        if now > deadline:
            self.timed_out = True
            self.completed_at = now.isoformat()
            return True
        return False

    def complete(self) -> None:
        """라운드 완료"""
        self.completed_at = datetime.now().isoformat()

    def calculate_revote_result(self) -> None:
        """재투표 결과 계산"""
        if not self.revote_opinions:
            return

        total_weight = 0.0
        weighted_sum = 0.0

        for opinion in self.revote_opinions.values():
            weight = opinion.evidence_weight
            weighted_sum += opinion.adjusted_confidence * weight
            total_weight += weight

        self.revote_combined_confidence = weighted_sum / total_weight if total_weight > 0 else 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "round_id": self.round_id,
            "round_number": self.round_number,
            "challenge": self.challenge.to_dict() if self.challenge else None,
            "timeout_seconds": self.timeout_seconds,
            "started_at": self.started_at,
            "deadline": self.deadline,
            "completed_at": self.completed_at,
            "timed_out": self.timed_out,
            "participants": self.participants,
            "statements": self.statements,
            "evidence_presented": [e.to_dict() for e in self.evidence_presented],
            "revote_opinions": {k: v.to_dict() for k, v in self.revote_opinions.items()},
            "revote_combined_confidence": self.revote_combined_confidence,
        }


@dataclass
class HumanEscalation:
    """
    Human Escalation 요청

    합의 실패 시 사람에게 검토를 요청
    """
    escalation_id: str
    topic_id: str
    topic_name: str
    reason: str

    # 요청 상태
    status: str = "pending"  # pending, reviewed, resolved
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    resolved_at: Optional[str] = None

    # 관련 정보
    challenges: List[Challenge] = field(default_factory=list)
    debate_history: List[DebateRound] = field(default_factory=list)
    agent_opinions: Dict[str, AgentOpinion] = field(default_factory=dict)

    # Human 결정
    human_decision: Optional[str] = None
    human_notes: Optional[str] = None

    def resolve(self, decision: str, notes: str = "") -> None:
        """Escalation 해결"""
        self.status = "resolved"
        self.resolved_at = datetime.now().isoformat()
        self.human_decision = decision
        self.human_notes = notes

    def to_dict(self) -> Dict[str, Any]:
        return {
            "escalation_id": self.escalation_id,
            "topic_id": self.topic_id,
            "topic_name": self.topic_name,
            "reason": self.reason,
            "status": self.status,
            "created_at": self.created_at,
            "resolved_at": self.resolved_at,
            "challenges": [c.to_dict() for c in self.challenges],
            "debate_history": [r.to_dict() for r in self.debate_history],
            "agent_opinions": {k: v.to_dict() for k, v in self.agent_opinions.items()},
            "human_decision": self.human_decision,
            "human_notes": self.human_notes,
        }
