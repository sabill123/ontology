"""
Debate Protocol - Evidence-based Council Debate System (v11.0)

BFT(Byzantine Fault Tolerance) 기반의 Council 토론 프로토콜:
- 각 에이전트가 근거 체인을 기반으로 주장
- Challenge/반론 메커니즘으로 실제 토론 수행
- 합의 도달 시까지 라운드 진행
- 증거 기반 의사결정

토론 흐름:
1. Proposal 제출 (안건)
2. Initial Vote (각 에이전트 의견 + 근거 제시)
3. Challenge Round (반대 의견 에이전트의 반론)
4. Defense Round (찬성 측 반박)
5. Re-vote (최종 투표)
6. Consensus Check (합의 확인)
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Any, Optional, Callable
from enum import Enum

from .evidence_chain import EvidenceBlock, EvidenceChain, EvidenceRegistry, EvidenceType

logger = logging.getLogger(__name__)


class VoteDecision(str, Enum):
    """투표 결정"""
    APPROVE = "approve"
    REJECT = "reject"
    ABSTAIN = "abstain"
    REVIEW = "review"  # 추가 검토 필요


class DebateRound(str, Enum):
    """토론 라운드"""
    INITIAL_VOTE = "initial_vote"
    CHALLENGE = "challenge"
    DEFENSE = "defense"
    FINAL_VOTE = "final_vote"


@dataclass
class AgentVoteWithEvidence:
    """근거 기반 에이전트 투표"""
    agent_name: str
    agent_role: str  # "ontologist", "risk_assessor", "business_strategist", "data_steward"
    decision: VoteDecision
    confidence: float

    # 근거 기반 주장
    reasoning: str
    cited_evidence: List[str] = field(default_factory=list)  # block_ids
    supporting_evidence: List[str] = field(default_factory=list)  # 지지 근거
    contradicting_evidence: List[str] = field(default_factory=list)  # 반박 근거

    # 토론 과정
    round: DebateRound = DebateRound.INITIAL_VOTE
    changed_from: Optional[VoteDecision] = None  # 의견 변경 시 이전 의견
    change_reason: str = ""

    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_name": self.agent_name,
            "agent_role": self.agent_role,
            "decision": self.decision.value,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "cited_evidence": self.cited_evidence,
            "supporting_evidence": self.supporting_evidence,
            "contradicting_evidence": self.contradicting_evidence,
            "round": self.round.value,
            "changed_from": self.changed_from.value if self.changed_from else None,
            "change_reason": self.change_reason,
            "timestamp": self.timestamp,
        }


@dataclass
class ChallengeArgument:
    """Challenge 주장 (반론)"""
    challenger: str  # 반론 제기 에이전트
    target_vote: AgentVoteWithEvidence  # 반론 대상
    challenge_type: str  # "evidence_weakness", "logic_flaw", "missing_consideration"
    argument: str  # 반론 내용
    counter_evidence: List[str] = field(default_factory=list)  # 반박 근거 block_ids
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class DefenseArgument:
    """Defense 주장 (반박)"""
    defender: str  # 반박 에이전트
    challenge: ChallengeArgument  # 반론 대상
    defense_type: str  # "evidence_rebuttal", "additional_evidence", "scope_clarification"
    argument: str  # 반박 내용
    additional_evidence: List[str] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class DebateResult:
    """토론 결과"""
    proposal_id: str
    proposal: str
    final_decision: VoteDecision
    consensus_reached: bool
    consensus_type: str  # "unanimous", "supermajority", "majority", "no_consensus"
    confidence: float

    # 투표 결과
    votes: Dict[str, AgentVoteWithEvidence] = field(default_factory=dict)
    vote_counts: Dict[str, int] = field(default_factory=dict)

    # 토론 과정
    rounds_conducted: int = 0
    challenges: List[ChallengeArgument] = field(default_factory=list)
    defenses: List[DefenseArgument] = field(default_factory=list)

    # 근거 체인
    evidence_cited: List[str] = field(default_factory=list)
    key_evidence_blocks: List[str] = field(default_factory=list)

    # 메타데이터
    duration_ms: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "proposal_id": self.proposal_id,
            "proposal": self.proposal,
            "final_decision": self.final_decision.value,
            "consensus_reached": self.consensus_reached,
            "consensus_type": self.consensus_type,
            "confidence": self.confidence,
            "votes": {k: v.to_dict() for k, v in self.votes.items()},
            "vote_counts": self.vote_counts,
            "rounds_conducted": self.rounds_conducted,
            "challenges": [
                {
                    "challenger": c.challenger,
                    "target": c.target_vote.agent_name,
                    "type": c.challenge_type,
                    "argument": c.argument,
                }
                for c in self.challenges
            ],
            "defenses": [
                {
                    "defender": d.defender,
                    "type": d.defense_type,
                    "argument": d.argument,
                }
                for d in self.defenses
            ],
            "evidence_cited": self.evidence_cited,
            "key_evidence_blocks": self.key_evidence_blocks,
            "duration_ms": self.duration_ms,
            "timestamp": self.timestamp,
        }


class CouncilAgent:
    """
    Council 에이전트 (토론 참여자)

    각 에이전트는 고유한 관점(role)에서 근거를 평가하고 투표
    """

    def __init__(
        self,
        name: str,
        role: str,
        evaluation_focus: List[str],
        weight: float = 1.0,
        llm_client: Optional[Any] = None,
    ):
        self.name = name
        self.role = role
        self.evaluation_focus = evaluation_focus  # ["consistency", "evidence_quality", ...]
        self.weight = weight
        self.llm_client = llm_client

    def evaluate_proposal(
        self,
        proposal: str,
        evidence_chain: EvidenceChain,
        context: Dict[str, Any] = None,
    ) -> AgentVoteWithEvidence:
        """
        안건 평가 및 투표

        Args:
            proposal: 평가할 안건
            evidence_chain: 근거 체인
            context: 추가 컨텍스트

        Returns:
            근거 기반 투표
        """
        context = context or {}

        # 관련 근거 수집
        all_evidence = list(evidence_chain.blocks.values())
        relevant_evidence = self._find_relevant_evidence(proposal, all_evidence)

        # 근거 분석
        supporting, contradicting = self._analyze_evidence(
            proposal, relevant_evidence, context
        )

        # 의사결정 (근거 기반)
        decision, confidence, reasoning = self._make_decision(
            proposal, supporting, contradicting, context
        )

        return AgentVoteWithEvidence(
            agent_name=self.name,
            agent_role=self.role,
            decision=decision,
            confidence=confidence,
            reasoning=reasoning,
            cited_evidence=[e.block_id for e in relevant_evidence[:5]],
            supporting_evidence=[e.block_id for e in supporting],
            contradicting_evidence=[e.block_id for e in contradicting],
            round=DebateRound.INITIAL_VOTE,
        )

    def _find_relevant_evidence(
        self,
        proposal: Any,
        all_evidence: List[EvidenceBlock]
    ) -> List[EvidenceBlock]:
        """안건과 관련된 근거 찾기"""
        relevant = []
        # proposal이 dict일 수 있으므로 문자열로 변환
        if isinstance(proposal, dict):
            proposal_str = " ".join(str(v) for v in proposal.values())
        else:
            proposal_str = str(proposal)
        proposal_lower = proposal_str.lower()

        for evidence in all_evidence:
            # 키워드 매칭
            text = f"{evidence.finding} {evidence.conclusion}".lower()
            if any(word in text for word in proposal_lower.split()[:5]):
                relevant.append(evidence)
            # Role별 관심 영역 매칭
            elif self._matches_focus(evidence):
                relevant.append(evidence)

        return relevant

    def _matches_focus(self, evidence: EvidenceBlock) -> bool:
        """에이전트의 관심 영역과 매칭되는지 확인"""
        focus_keywords = {
            "consistency": ["conflict", "inconsistent", "contradiction", "mismatch"],
            "risk": ["risk", "danger", "warning", "critical", "failure"],
            "business_value": ["revenue", "cost", "efficiency", "kpi", "impact"],
            "data_quality": ["quality", "accuracy", "completeness", "validity"],
            "evidence_quality": ["confidence", "statistical", "p-value", "correlation"],
        }

        evidence_text = f"{evidence.finding} {evidence.reasoning}".lower()

        for focus in self.evaluation_focus:
            keywords = focus_keywords.get(focus, [])
            if any(kw in evidence_text for kw in keywords):
                return True

        return False

    def _analyze_evidence(
        self,
        proposal: str,
        evidence: List[EvidenceBlock],
        context: Dict[str, Any]
    ) -> tuple:
        """근거를 지지/반박으로 분류"""
        supporting = []
        contradicting = []

        for e in evidence:
            # 간단한 휴리스틱: 높은 confidence면 지지, 낮으면 반박 가능성
            # 실제로는 LLM이나 더 정교한 로직 필요
            if e.confidence >= 0.7:
                supporting.append(e)
            elif e.confidence < 0.5:
                contradicting.append(e)
            else:
                # Role별 판단
                if self.role == "risk_assessor":
                    # Risk 관련 키워드가 있으면 반박 근거로
                    if "risk" in e.finding.lower() or "warning" in e.finding.lower():
                        contradicting.append(e)
                    else:
                        supporting.append(e)
                else:
                    supporting.append(e)

        return supporting, contradicting

    def _make_decision(
        self,
        proposal: str,
        supporting: List[EvidenceBlock],
        contradicting: List[EvidenceBlock],
        context: Dict[str, Any]
    ) -> tuple:
        """의사결정 수행"""
        # 가중치 계산
        support_weight = sum(e.confidence for e in supporting)
        contradict_weight = sum(e.confidence for e in contradicting)

        # Role별 기본 임계값
        role_thresholds = {
            "ontologist": {"approve": 0.6, "review": 0.3},
            "risk_assessor": {"approve": 0.75, "review": 0.5},  # 더 엄격
            "business_strategist": {"approve": 0.55, "review": 0.3},  # 더 유연
            "data_steward": {"approve": 0.65, "review": 0.4},
        }

        thresholds = role_thresholds.get(self.role, {"approve": 0.6, "review": 0.35})

        # 결정
        total_weight = support_weight + contradict_weight
        if total_weight == 0:
            return VoteDecision.ABSTAIN, 0.5, "Insufficient evidence to make a decision"

        support_ratio = support_weight / total_weight

        if support_ratio >= thresholds["approve"]:
            decision = VoteDecision.APPROVE
            confidence = min(0.95, support_ratio)
            reasoning = self._generate_reasoning(
                decision, supporting, contradicting, "Support outweighs concerns"
            )
        elif support_ratio >= thresholds["review"]:
            decision = VoteDecision.REVIEW
            confidence = 0.5 + (support_ratio - thresholds["review"]) * 0.3
            reasoning = self._generate_reasoning(
                decision, supporting, contradicting, "Mixed evidence, needs review"
            )
        else:
            decision = VoteDecision.REJECT
            confidence = min(0.9, 1 - support_ratio)
            reasoning = self._generate_reasoning(
                decision, supporting, contradicting, "Contradicting evidence outweighs support"
            )

        return decision, confidence, reasoning

    def _generate_reasoning(
        self,
        decision: VoteDecision,
        supporting: List[EvidenceBlock],
        contradicting: List[EvidenceBlock],
        summary: str
    ) -> str:
        """근거 기반 추론 생성"""
        lines = [f"[{self.role.upper()}] {summary}"]

        if supporting:
            top_support = sorted(supporting, key=lambda e: e.confidence, reverse=True)[:2]
            lines.append(f"\nSupporting evidence ({len(supporting)} blocks):")
            for e in top_support:
                lines.append(f"  - [{e.block_id}] {e.agent}: \"{e.finding[:60]}...\" (conf: {e.confidence:.0%})")

        if contradicting:
            top_contradict = sorted(contradicting, key=lambda e: e.confidence, reverse=True)[:2]
            lines.append(f"\nContradicting evidence ({len(contradicting)} blocks):")
            for e in top_contradict:
                lines.append(f"  - [{e.block_id}] {e.agent}: \"{e.finding[:60]}...\" (conf: {e.confidence:.0%})")

        return "\n".join(lines)

    def respond_to_challenge(
        self,
        challenge: ChallengeArgument,
        evidence_chain: EvidenceChain,
        original_vote: AgentVoteWithEvidence,
    ) -> tuple:
        """
        Challenge에 응답 (의견 유지/변경 결정)

        Returns:
            (new_vote, defense_argument)
        """
        # Challenge 근거 분석
        challenge_evidence = [
            evidence_chain.blocks[bid]
            for bid in challenge.counter_evidence
            if bid in evidence_chain.blocks
        ]

        # 기존 근거 재평가
        original_evidence = [
            evidence_chain.blocks[bid]
            for bid in original_vote.cited_evidence
            if bid in evidence_chain.blocks
        ]

        challenge_strength = sum(e.confidence for e in challenge_evidence)
        original_strength = sum(e.confidence for e in original_evidence)

        # 의견 변경 여부 결정
        if challenge_strength > original_strength * 1.2:  # 20% 이상 강하면 변경
            new_decision = VoteDecision.REVIEW if original_vote.decision == VoteDecision.APPROVE else VoteDecision.APPROVE
            new_vote = AgentVoteWithEvidence(
                agent_name=self.name,
                agent_role=self.role,
                decision=new_decision,
                confidence=0.6,
                reasoning=f"Revised position based on challenge evidence from {challenge.challenger}",
                cited_evidence=original_vote.cited_evidence + challenge.counter_evidence[:2],
                supporting_evidence=original_vote.supporting_evidence,
                contradicting_evidence=original_vote.contradicting_evidence + challenge.counter_evidence,
                round=DebateRound.FINAL_VOTE,
                changed_from=original_vote.decision,
                change_reason=f"Challenge evidence strength ({challenge_strength:.2f}) exceeded original ({original_strength:.2f})",
            )
            defense = None
        else:
            # 의견 유지 및 반박
            new_vote = AgentVoteWithEvidence(
                agent_name=self.name,
                agent_role=self.role,
                decision=original_vote.decision,
                confidence=original_vote.confidence,
                reasoning=original_vote.reasoning + f"\n[MAINTAINED after challenge from {challenge.challenger}]",
                cited_evidence=original_vote.cited_evidence,
                supporting_evidence=original_vote.supporting_evidence,
                contradicting_evidence=original_vote.contradicting_evidence,
                round=DebateRound.FINAL_VOTE,
            )
            defense = DefenseArgument(
                defender=self.name,
                challenge=challenge,
                defense_type="evidence_rebuttal",
                argument=f"Original evidence (strength: {original_strength:.2f}) remains stronger than challenge (strength: {challenge_strength:.2f})",
                additional_evidence=original_vote.supporting_evidence[:2],
            )

        return new_vote, defense


class DebateProtocol:
    """
    토론 프로토콜 - BFT 기반 합의 도출

    토론 흐름:
    1. Initial Vote: 각 에이전트가 근거 기반 투표
    2. Challenge: 소수 의견 에이전트가 다수 의견에 반론
    3. Defense: 다수 의견 에이전트가 반박
    4. Final Vote: 재투표
    5. Consensus: 합의 확인
    """

    def __init__(
        self,
        agents: List[CouncilAgent] = None,
        consensus_threshold: float = 0.75,  # Supermajority
        max_rounds: int = 3,
        llm_client: Optional[Any] = None,
    ):
        self.agents = agents or self._create_default_agents()
        self.consensus_threshold = consensus_threshold
        self.max_rounds = max_rounds
        self.llm_client = llm_client

    def _create_default_agents(self) -> List[CouncilAgent]:
        """기본 4명의 Council 에이전트 생성"""
        return [
            CouncilAgent(
                name="Ontologist",
                role="ontologist",
                evaluation_focus=["consistency", "evidence_quality"],
                weight=1.3,
            ),
            CouncilAgent(
                name="Risk Assessor",
                role="risk_assessor",
                evaluation_focus=["risk", "data_quality"],
                weight=1.5,
            ),
            CouncilAgent(
                name="Business Strategist",
                role="business_strategist",
                evaluation_focus=["business_value"],
                weight=1.0,
            ),
            CouncilAgent(
                name="Data Steward",
                role="data_steward",
                evaluation_focus=["data_quality", "consistency"],
                weight=1.2,
            ),
        ]

    def conduct_debate(
        self,
        proposal_id: str,
        proposal: str,
        evidence_chain: EvidenceChain,
        context: Dict[str, Any] = None,
    ) -> DebateResult:
        """
        토론 수행

        Args:
            proposal_id: 안건 ID
            proposal: 안건 내용
            evidence_chain: 근거 체인
            context: 추가 컨텍스트

        Returns:
            토론 결과
        """
        import time
        start_time = time.time()

        context = context or {}
        logger.info(f"Starting debate for proposal: {proposal_id}")

        # Phase 1: Initial Vote
        votes = {}
        for agent in self.agents:
            vote = agent.evaluate_proposal(proposal, evidence_chain, context)
            votes[agent.name] = vote
            logger.info(f"  {agent.name} votes: {vote.decision.value} (confidence: {vote.confidence:.0%})")

        # Check for quick consensus
        initial_consensus, consensus_type = self._check_consensus(votes)

        if initial_consensus:
            logger.info(f"Quick consensus reached: {consensus_type}")
            return self._create_result(
                proposal_id, proposal, votes, [], [],
                rounds=1,
                duration_ms=(time.time() - start_time) * 1000,
            )

        # Phase 2: Challenge Round
        challenges = []
        minority_agents = self._get_minority_agents(votes)

        for minority_agent in minority_agents:
            majority_votes = self._get_majority_votes(votes)
            for majority_vote in majority_votes[:1]:  # 대표 1명에게만 challenge
                challenge = self._create_challenge(
                    minority_agent,
                    majority_vote,
                    evidence_chain,
                    votes[minority_agent.name]
                )
                if challenge:
                    challenges.append(challenge)
                    logger.info(f"  Challenge: {challenge.challenger} -> {challenge.target_vote.agent_name}")

        # Phase 3: Defense Round
        defenses = []
        for challenge in challenges:
            target_agent = next(
                (a for a in self.agents if a.name == challenge.target_vote.agent_name),
                None
            )
            if target_agent:
                new_vote, defense = target_agent.respond_to_challenge(
                    challenge, evidence_chain, votes[target_agent.name]
                )
                votes[target_agent.name] = new_vote
                if defense:
                    defenses.append(defense)
                    logger.info(f"  Defense: {defense.defender} maintains position")
                else:
                    logger.info(f"  {target_agent.name} changed vote to {new_vote.decision.value}")

        # Phase 4: Final Consensus Check
        final_consensus, consensus_type = self._check_consensus(votes)

        duration_ms = (time.time() - start_time) * 1000
        logger.info(f"Debate completed in {duration_ms:.0f}ms. Consensus: {consensus_type}")

        return self._create_result(
            proposal_id, proposal, votes, challenges, defenses,
            rounds=2 if challenges else 1,
            duration_ms=duration_ms,
        )

    def _check_consensus(self, votes: Dict[str, AgentVoteWithEvidence]) -> tuple:
        """합의 확인"""
        vote_counts = {}
        weighted_counts = {}

        for agent_name, vote in votes.items():
            agent = next((a for a in self.agents if a.name == agent_name), None)
            weight = agent.weight if agent else 1.0

            decision = vote.decision.value
            vote_counts[decision] = vote_counts.get(decision, 0) + 1
            weighted_counts[decision] = weighted_counts.get(decision, 0) + weight

        total_weight = sum(weighted_counts.values())
        total_votes = len(votes)

        # 만장일치
        if len(vote_counts) == 1:
            return True, "unanimous"

        # Supermajority (75%+)
        for decision, weight in weighted_counts.items():
            if weight / total_weight >= self.consensus_threshold:
                return True, "supermajority"

        # Simple majority (50%+)
        for decision, count in vote_counts.items():
            if count / total_votes > 0.5:
                return True, "majority"

        return False, "no_consensus"

    def _get_minority_agents(self, votes: Dict[str, AgentVoteWithEvidence]) -> List[CouncilAgent]:
        """소수 의견 에이전트 찾기"""
        vote_counts = {}
        for vote in votes.values():
            decision = vote.decision.value
            vote_counts[decision] = vote_counts.get(decision, 0) + 1

        majority_decision = max(vote_counts, key=vote_counts.get)

        return [
            agent for agent in self.agents
            if votes.get(agent.name) and votes[agent.name].decision.value != majority_decision
        ]

    def _get_majority_votes(self, votes: Dict[str, AgentVoteWithEvidence]) -> List[AgentVoteWithEvidence]:
        """다수 의견 투표 가져오기"""
        vote_counts = {}
        for vote in votes.values():
            decision = vote.decision.value
            vote_counts[decision] = vote_counts.get(decision, 0) + 1

        majority_decision = max(vote_counts, key=vote_counts.get)

        return [
            vote for vote in votes.values()
            if vote.decision.value == majority_decision
        ]

    def _create_challenge(
        self,
        challenger_agent: CouncilAgent,
        target_vote: AgentVoteWithEvidence,
        evidence_chain: EvidenceChain,
        challenger_vote: AgentVoteWithEvidence,
    ) -> Optional[ChallengeArgument]:
        """Challenge 생성"""
        # 반박 근거 수집
        counter_evidence = challenger_vote.contradicting_evidence

        if not counter_evidence:
            # 근거가 없으면 새로 검색
            all_evidence = list(evidence_chain.blocks.values())
            for e in all_evidence:
                if e.confidence < 0.6:  # 낮은 신뢰도 근거
                    counter_evidence.append(e.block_id)
                if len(counter_evidence) >= 3:
                    break

        if not counter_evidence:
            return None

        return ChallengeArgument(
            challenger=challenger_agent.name,
            target_vote=target_vote,
            challenge_type="evidence_weakness",
            argument=f"Evidence cited by {target_vote.agent_name} has weaknesses. Counter-evidence shows potential issues.",
            counter_evidence=counter_evidence,
        )

    def _create_result(
        self,
        proposal_id: str,
        proposal: str,
        votes: Dict[str, AgentVoteWithEvidence],
        challenges: List[ChallengeArgument],
        defenses: List[DefenseArgument],
        rounds: int,
        duration_ms: float,
    ) -> DebateResult:
        """토론 결과 생성"""
        # 최종 결정 계산
        vote_counts = {}
        weighted_counts = {}

        for agent_name, vote in votes.items():
            agent = next((a for a in self.agents if a.name == agent_name), None)
            weight = agent.weight if agent else 1.0

            decision = vote.decision.value
            vote_counts[decision] = vote_counts.get(decision, 0) + 1
            weighted_counts[decision] = weighted_counts.get(decision, 0) + weight

        # 가중 다수결로 최종 결정
        final_decision_str = max(weighted_counts, key=weighted_counts.get)
        final_decision = VoteDecision(final_decision_str)

        total_weight = sum(weighted_counts.values())
        confidence = weighted_counts[final_decision_str] / total_weight if total_weight > 0 else 0.5

        # 합의 유형 결정
        _, consensus_type = self._check_consensus(votes)
        consensus_reached = consensus_type != "no_consensus"

        # 인용된 근거 수집
        all_cited = set()
        key_evidence = set()
        for vote in votes.values():
            all_cited.update(vote.cited_evidence)
            key_evidence.update(vote.supporting_evidence[:2])

        return DebateResult(
            proposal_id=proposal_id,
            proposal=proposal,
            final_decision=final_decision,
            consensus_reached=consensus_reached,
            consensus_type=consensus_type,
            confidence=confidence,
            votes=votes,
            vote_counts=vote_counts,
            rounds_conducted=rounds,
            challenges=challenges,
            defenses=defenses,
            evidence_cited=list(all_cited),
            key_evidence_blocks=list(key_evidence),
            duration_ms=duration_ms,
        )


def conduct_evidence_based_debate(
    proposal_id: str,
    proposal: str,
    evidence_registry: EvidenceRegistry,
    context: Dict[str, Any] = None,
    llm_client: Any = None,
) -> DebateResult:
    """
    근거 기반 토론 수행 (편의 함수)

    Args:
        proposal_id: 안건 ID
        proposal: 안건 내용
        evidence_registry: 근거 레지스트리
        context: 추가 컨텍스트
        llm_client: LLM 클라이언트 (선택)

    Returns:
        토론 결과
    """
    protocol = DebateProtocol(llm_client=llm_client)
    return protocol.conduct_debate(
        proposal_id=proposal_id,
        proposal=proposal,
        evidence_chain=evidence_registry.chain,
        context=context,
    )
