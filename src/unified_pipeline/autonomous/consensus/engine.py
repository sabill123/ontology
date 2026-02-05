"""
Pipeline Consensus Engine (v3.0 - Async & Fast-track)

멀티에이전트 토론/합의 엔진
- Async Parallel Thinking: 모든 에이전트가 동시에 사고하고 투표
- Fast-track Consensus: 만장일치 시 즉시 종료 (Step 1 Optimization)
- Evidence Integration: 알고리즘 증거 기반 토론
"""

import asyncio
import logging
import re
import json
from datetime import datetime
from typing import Dict, List, Any, Optional, AsyncGenerator, TYPE_CHECKING, Tuple

from .models import (
    ConsensusState,
    ConsensusResult,
    ConsensusType,
    AgentOpinion,
    DiscussionMessage,
    DiscussionRound,
    VoteType,
    # Debate Protocol models (Phase 3)
    Challenge,
    ChallengeStatus,
    DebateEvidence,
    DebateRound,
    EvidenceType,
    HumanEscalation,
)
from .channel import AgentCommunicationChannel
from .logger import ConversationLogger, get_conversation_logger

# v7.8: CouncilAgent 통합 (C-3 Fix)
try:
    from ..debate_protocol import CouncilAgent, VoteDecision, AgentVoteWithEvidence
    from ..evidence_chain import EvidenceChain
    COUNCIL_AGENT_AVAILABLE = True
except ImportError:
    COUNCIL_AGENT_AVAILABLE = False
    CouncilAgent = None
    VoteDecision = None
    AgentVoteWithEvidence = None
    EvidenceChain = None

# v7.8: StructuredDebate 통합 (C-4 Fix)
try:
    from ..analysis.structured_debate import StructuredDebateEngine, DebateOutcome
    STRUCTURED_DEBATE_AVAILABLE = True
except ImportError:
    STRUCTURED_DEBATE_AVAILABLE = False
    StructuredDebateEngine = None
    DebateOutcome = None

# EvidenceBundle 통합 (Optional)
try:
    from src.common.core.evidence import (
        EvidenceBundle,
        EvidenceFactory,
        EvidenceType,
        DataEvidence,
        SemanticEvidence,
        AgentAssessmentEvidence,
        CertificationLevel,
    )
    from src.common.core.models import AgentRole
    EVIDENCE_AVAILABLE = True
except ImportError:
    EVIDENCE_AVAILABLE = False
    EvidenceBundle = None
    EvidenceFactory = None

# v16.0: AgentCommunicationBus 통합 (Direct Agent-to-Agent Messaging)
try:
    from ..agent_bus import (
        AgentCommunicationBus,
        AgentMessage,
        MessageType,
        MessagePriority,
        get_agent_bus,
        create_info_message,
        create_discovery_message,
    )
    AGENT_BUS_AVAILABLE = True
except ImportError:
    AGENT_BUS_AVAILABLE = False
    AgentCommunicationBus = None
    get_agent_bus = None

# v16.0: Column-level Lineage (Impact Analysis during Consensus)
try:
    from ...lineage import LineageTracker, build_lineage_from_context, ImpactAnalysis
    LINEAGE_AVAILABLE = True
except ImportError:
    LINEAGE_AVAILABLE = False
    LineageTracker = None
    build_lineage_from_context = None

if TYPE_CHECKING:
    from ..base import AutonomousAgent
    from ...shared_context import SharedContext

logger = logging.getLogger(__name__)


# =============================================================================
# v7.8: Agent-to-Council Role Mapping (C-1 Fix)
# =============================================================================
AGENT_ROLE_MAPPING: Dict[str, str] = {
    # Phase 1: Discovery Agents
    "data_analyst": "data_steward",          # 데이터 품질/무결성 관점
    "tda_expert": "ontologist",              # 구조적/위상적 관점
    "schema_analyst": "data_steward",        # 스키마/메타데이터 관점
    "value_matcher": "data_steward",         # 값 일치성 관점
    "entity_classifier": "ontologist",       # 엔티티 분류 관점
    "relationship_detector": "ontologist",   # 관계 모델링 관점

    # Phase 2: Refinement Agents
    "ontology_architect": "ontologist",      # 온톨로지 설계 관점
    "conflict_resolver": "risk_assessor",    # 충돌/위험 평가 관점
    "quality_judge": "risk_assessor",        # 품질/위험 평가 관점
    "semantic_validator": "ontologist",      # 의미적 일관성 관점

    # Phase 3: Governance Agents
    "governance_strategist": "business_strategist",  # 비즈니스 전략 관점
    "action_prioritizer": "business_strategist",     # ROI/우선순위 관점
    "risk_assessor": "risk_assessor",                # 위험 평가 관점
    "policy_generator": "data_steward",              # 정책/규정 관점
}

# =============================================================================
# v7.8: Role-Specific Vote Prompts (C-2 Fix)
# =============================================================================
ROLE_SPECIFIC_PROMPTS: Dict[str, str] = {
    "ontologist": """## Your Role: ONTOLOGIST
You evaluate proposals from a **structural and semantic perspective**.

Focus Areas:
- Is the entity/relationship properly modeled?
- Does it follow ontology best practices (naming, cardinality)?
- Is the semantic meaning clear and unambiguous?
- Does it integrate well with existing ontology structure?

Evaluation Criteria:
- Structural Consistency: Does it fit the overall model?
- Semantic Clarity: Is the meaning well-defined?
- Reusability: Can this be extended in the future?
""",

    "risk_assessor": """## Your Role: RISK ASSESSOR
You evaluate proposals from a **risk and quality perspective**.

Focus Areas:
- What are the potential risks of accepting this?
- What is the confidence level of the evidence?
- Are there data quality issues that could affect accuracy?
- What could go wrong if this is incorrect?

Evaluation Criteria:
- Evidence Strength: How reliable is the supporting data?
- Risk Level: What's the impact if wrong? (low/medium/high)
- Quality Score: Does it meet quality thresholds?

Be conservative - when in doubt, flag for review.
""",

    "business_strategist": """## Your Role: BUSINESS STRATEGIST
You evaluate proposals from a **business value perspective**.

Focus Areas:
- Does this provide actionable business insights?
- What is the ROI potential of this information?
- How does this support business decision-making?
- Is this aligned with business objectives?

Evaluation Criteria:
- Business Relevance: Is this useful for business users?
- Actionability: Can decisions be made based on this?
- Value Potential: What opportunities does this unlock?

Prioritize practical business value over theoretical purity.
""",

    "data_steward": """## Your Role: DATA STEWARD
You evaluate proposals from a **data governance perspective**.

Focus Areas:
- Is the data lineage clear and traceable?
- Are there data quality or integrity concerns?
- Does this comply with data governance policies?
- Is the data source reliable and authoritative?

Evaluation Criteria:
- Data Quality: Is the underlying data reliable?
- Traceability: Can we trace the origin of this data?
- Compliance: Does it meet governance standards?

Ensure data integrity and trustworthiness.
""",
}

# Role-specific decision thresholds (더 엄격한/유연한 기준)
ROLE_DECISION_THRESHOLDS: Dict[str, Dict[str, float]] = {
    "ontologist": {"approve": 0.60, "review": 0.30},       # 중립적
    "risk_assessor": {"approve": 0.75, "review": 0.50},    # 더 엄격
    "business_strategist": {"approve": 0.55, "review": 0.30},  # 더 유연
    "data_steward": {"approve": 0.65, "review": 0.40},     # 약간 엄격
}


class PipelineConsensusEngine:
    """
    파이프라인 합의 엔진 (v8.1 Optimized)

    Optimization Goals:
    1. Speed: Asyncio.gather()로 병렬 처리 (3x Faster)
    2. Efficiency: Fast-track으로 쉬운 안건은 1라운드 종료
    3. Depth: 어려운 안건만 Deep Debate 진행
    4. v8.1: 강화된 Early Exit 및 효율적 합의
    """

    def __init__(
        self,
        max_rounds: int = 2,  # v7.0: 2라운드로 축소 (Fast-track 강화)
        max_messages_per_round: int = 10,  # 충분한 발언권 보장
        acceptance_threshold: float = 0.75,
        rejection_threshold: float = 0.3,
        provisional_threshold: float = 0.5,
        min_agreement: float = 0.6,
        min_agents_for_consensus: int = 3,
        round_timeout: float = 0,  # v18.0: 타임아웃 비활성화 (무제한)
        enable_debate_protocol: bool = True,  # Phase 3: Debate Protocol 활성화
        fast_track_threshold: float = 0.55,  # v8.1: Fast-track 임계값 (55%+면 즉시 종료)
        strong_consensus_threshold: float = 0.85,  # v8.1: 강한 합의 임계값
    ):
        self.max_rounds = max_rounds
        self.max_messages_per_round = max_messages_per_round
        self.acceptance_threshold = acceptance_threshold
        self.rejection_threshold = rejection_threshold
        self.provisional_threshold = provisional_threshold
        self.min_agreement = min_agreement
        self.min_agents_for_consensus = min_agents_for_consensus
        self.round_timeout = round_timeout
        self.fast_track_threshold = fast_track_threshold  # v8.1: 55%
        self.strong_consensus_threshold = strong_consensus_threshold  # v8.1: 85%

        # 통신 채널
        self.channel = AgentCommunicationChannel()
        self._agents: Dict[str, "AutonomousAgent"] = {}
        self._current_discussion: Optional[ConsensusState] = None

        # 로거
        self._conversation_logger: Optional[ConversationLogger] = None
        self._current_discussion_log: Optional[Dict[str, Any]] = None

        # Phase 3: Debate Protocol
        self._debate_protocol: Optional["DebateProtocol"] = None
        self._enable_debate_protocol = enable_debate_protocol

        # v16.0: AgentCommunicationBus 통합 (Direct Agent-to-Agent Messaging)
        self._agent_bus: Optional[AgentCommunicationBus] = None
        if AGENT_BUS_AVAILABLE and get_agent_bus:
            try:
                self._agent_bus = get_agent_bus()
                logger.debug("[v16.0] AgentCommunicationBus integrated with ConsensusEngine")
            except Exception as e:
                logger.warning(f"[v16.0] AgentCommunicationBus integration failed: {e}")

        # v16.0: Consensus topics for pub/sub
        self._consensus_topics = {
            "vote_cast": "consensus.vote_cast",
            "round_start": "consensus.round_start",
            "round_end": "consensus.round_end",
            "consensus_reached": "consensus.consensus_reached",
            "challenge_issued": "consensus.challenge_issued",
        }

        logger.info(f"PipelineConsensusEngine v3.0 initialized (Async/Fast-track enabled, timeout={round_timeout}s, bus={'enabled' if self._agent_bus else 'disabled'})")

    def enable_logging(self, log_dir: str = "./data/logs/conversations") -> None:
        self._conversation_logger = ConversationLogger(log_dir)
        self._conversation_logger.enable()
        logger.info(f"ConversationLogger enabled: {log_dir}")

    def disable_logging(self) -> None:
        if self._conversation_logger:
            self._conversation_logger.disable()

    def register_agent(self, agent: "AutonomousAgent") -> None:
        self._agents[agent.agent_id] = agent
        self.channel.register_agent(
            agent_id=agent.agent_id,
            agent_name=agent.agent_name,
            agent_type=agent.agent_type,
        )

    def unregister_agent(self, agent_id: str) -> None:
        if agent_id in self._agents:
            del self._agents[agent_id]
            self.channel.unregister_agent(agent_id)

    def clear_agents(self) -> None:
        """모든 등록된 에이전트 해제"""
        agent_ids = list(self._agents.keys())
        for agent_id in agent_ids:
            self.unregister_agent(agent_id)
        logger.debug(f"Cleared {len(agent_ids)} agents from consensus engine")

    # --------------------------------------------------------
    # v16.0: AgentCommunicationBus Integration
    # --------------------------------------------------------

    async def _publish_consensus_event(
        self,
        event_type: str,
        data: Dict[str, Any],
    ) -> int:
        """
        v16.0: 합의 이벤트를 AgentCommunicationBus를 통해 발행

        Args:
            event_type: 이벤트 유형 (vote_cast, round_start, round_end, consensus_reached)
            data: 이벤트 데이터

        Returns:
            전달된 구독자 수
        """
        if not self._agent_bus or not AGENT_BUS_AVAILABLE:
            return 0

        topic = self._consensus_topics.get(event_type, f"consensus.{event_type}")
        try:
            message = create_info_message(
                sender_id="consensus_engine",
                content={
                    "event_type": event_type,
                    "timestamp": datetime.now().isoformat(),
                    **data,
                },
                topic=topic,
            )
            count = await self._agent_bus.publish(topic, message)
            logger.debug(f"[v16.0] Consensus event published: {event_type} to {count} subscribers")
            return count
        except Exception as e:
            logger.warning(f"[v16.0] Failed to publish consensus event: {e}")
            return 0

    async def _notify_agents_via_bus(
        self,
        recipients: List[str],
        content: Dict[str, Any],
        message_type: str = "info",
    ) -> int:
        """
        v16.0: 특정 에이전트들에게 메시지 전송

        Args:
            recipients: 수신자 에이전트 ID 리스트
            content: 메시지 내용
            message_type: 메시지 유형

        Returns:
            성공적으로 전달된 수
        """
        if not self._agent_bus or not AGENT_BUS_AVAILABLE:
            return 0

        success_count = 0
        for recipient_id in recipients:
            try:
                # MessageType 매핑
                type_map = {
                    "info": MessageType.INFO,
                    "collaboration": MessageType.COLLABORATION,
                    "validation": MessageType.VALIDATION,
                    "feedback": MessageType.FEEDBACK,
                }
                msg_type = type_map.get(message_type, MessageType.INFO)

                message = AgentMessage(
                    message_id=f"consensus_{datetime.now().strftime('%Y%m%d%H%M%S')}_{recipient_id[:8]}",
                    message_type=msg_type,
                    sender_id="consensus_engine",
                    recipient_id=recipient_id,
                    content=content,
                    priority=MessagePriority.HIGH,
                )
                if await self._agent_bus.send_to(recipient_id, message):
                    success_count += 1
            except Exception as e:
                logger.warning(f"[v16.0] Failed to notify {recipient_id}: {e}")

        return success_count

    async def request_collaborative_analysis(
        self,
        requesting_agent_id: str,
        target_agents: List[str],
        analysis_type: str,
        data: Dict[str, Any],
        timeout: float = 30.0,
    ) -> List[Dict[str, Any]]:
        """
        v16.0: 다중 에이전트에 협업 분석 요청 (Parallel Request/Response)

        Args:
            requesting_agent_id: 요청 에이전트 ID
            target_agents: 대상 에이전트 ID 리스트
            analysis_type: 분석 유형
            data: 분석 데이터
            timeout: 응답 대기 시간 (초)

        Returns:
            각 에이전트의 응답 리스트
        """
        if not self._agent_bus or not AGENT_BUS_AVAILABLE:
            return []

        responses = []

        async def request_single(agent_id: str) -> Optional[Dict[str, Any]]:
            try:
                from ..agent_bus import create_validation_request
                message = create_validation_request(
                    sender_id=requesting_agent_id,
                    validation_type=analysis_type,
                    data=data,
                )
                response = await self._agent_bus.request(agent_id, message, timeout)
                if response:
                    return {
                        "agent_id": agent_id,
                        "response": response.content,
                        "success": True,
                    }
            except Exception as e:
                logger.warning(f"[v16.0] Collaborative analysis request to {agent_id} failed: {e}")
            return {"agent_id": agent_id, "response": None, "success": False}

        # 병렬 요청
        tasks = [request_single(agent_id) for agent_id in target_agents]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for result in results:
            if isinstance(result, dict):
                responses.append(result)
            else:
                logger.warning(f"[v16.0] Unexpected result: {result}")

        logger.info(f"[v16.0] Collaborative analysis: {len([r for r in responses if r.get('success')])}/{len(target_agents)} responses")
        return responses

    def get_agent_bus(self) -> Optional[AgentCommunicationBus]:
        """v16.0: AgentCommunicationBus 인스턴스 반환"""
        return self._agent_bus

    # --------------------------------------------------------
    # Phase 3: Debate Protocol Integration
    # --------------------------------------------------------

    def get_debate_protocol(self) -> "DebateProtocol":
        """
        DebateProtocol 인스턴스 반환 (Lazy initialization)
        """
        if self._debate_protocol is None:
            self._debate_protocol = DebateProtocol(
                consensus_engine=self,
                round_timeout=self.round_timeout,
                max_debate_rounds=self.max_rounds,
            )
        return self._debate_protocol

    async def initiate_debate(
        self,
        challenger_id: str,
        target_vote: str,
        reason: str,
    ) -> Dict[str, Any]:
        """
        Debate 시작 (Challenge 발의)

        Args:
            challenger_id: Challenge를 제기하는 에이전트 ID
            target_vote: 대상 투표 유형
            reason: Challenge 사유

        Returns:
            Debate 시작 결과
        """
        if not self._enable_debate_protocol:
            raise RuntimeError("Debate Protocol is disabled")

        protocol = self.get_debate_protocol()
        challenge = await protocol.initiate_challenge(
            challenger_id=challenger_id,
            target_vote=target_vote,
            reason=reason,
        )

        return {
            "type": "debate_initiated",
            "challenge_id": challenge.challenge_id,
            "challenger_id": challenger_id,
            "target_vote": target_vote,
            "reason": reason,
        }

    async def submit_debate_evidence(
        self,
        evidence_type: str,
        data: Dict[str, Any],
        presenter_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        토론 증거 제출

        Args:
            evidence_type: 증거 유형 ("tda_result", "statistical", "semantic", "historical")
            data: 증거 데이터
            presenter_id: 제출자 ID

        Returns:
            증거 제출 결과
        """
        protocol = self.get_debate_protocol()

        # EvidenceType 변환
        type_map = {
            "tda_result": EvidenceType.TDA_RESULT,
            "statistical": EvidenceType.STATISTICAL,
            "semantic": EvidenceType.SEMANTIC,
            "historical": EvidenceType.HISTORICAL,
            "external": EvidenceType.EXTERNAL,
        }
        ev_type = type_map.get(evidence_type.lower(), EvidenceType.EXTERNAL)

        evidence = await protocol.present_evidence(
            evidence_type=ev_type,
            data=data,
            presenter_id=presenter_id,
        )

        return {
            "type": "evidence_submitted",
            "evidence_id": evidence.evidence_id,
            "evidence_type": evidence_type,
            "summary": evidence.summary,
        }

    async def request_revote(
        self,
        participants: Optional[List[str]] = None,
        timeout: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        재투표 요청

        Args:
            participants: 참여자 목록 (None이면 전체)
            timeout: 타임아웃 (None이면 기본값)

        Returns:
            재투표 결과
        """
        protocol = self.get_debate_protocol()
        result = await protocol.call_revote(
            participants=participants,
            timeout=timeout,
        )

        # Escalation 필요 여부 확인
        if result.get("resolution") == "needs_escalation":
            escalation = await protocol.escalate_to_human(
                topic=self._current_discussion.topic_name if self._current_discussion else "Unknown",
                reason="Consensus failed after multiple debate rounds",
            )
            result["escalation"] = escalation.to_dict()

        return result

    async def escalate_discussion(self, reason: str) -> Dict[str, Any]:
        """
        토론을 Human에게 Escalate

        Args:
            reason: Escalation 사유

        Returns:
            Escalation 결과
        """
        protocol = self.get_debate_protocol()
        topic = self._current_discussion.topic_name if self._current_discussion else "Unknown"

        escalation = await protocol.escalate_to_human(
            topic=topic,
            reason=reason,
        )

        return {
            "type": "escalated",
            "escalation_id": escalation.escalation_id,
            "topic": topic,
            "reason": reason,
            "status": escalation.status,
        }

    # --------------------------------------------------------
    # v7.8: CouncilAgent Integration (C-3 Fix)
    # --------------------------------------------------------

    def _create_council_agents(self) -> Dict[str, "CouncilAgent"]:
        """
        등록된 에이전트들로부터 CouncilAgent 인스턴스 생성

        Returns:
            {agent_id: CouncilAgent} 매핑
        """
        if not COUNCIL_AGENT_AVAILABLE:
            logger.warning("CouncilAgent not available - skipping council creation")
            return {}

        council_agents = {}

        # 역할별 평가 초점 정의
        ROLE_EVALUATION_FOCUS = {
            "ontologist": ["semantic_consistency", "naming_conventions", "structural_integrity"],
            "risk_assessor": ["data_quality", "confidence_levels", "potential_risks"],
            "business_strategist": ["business_value", "roi_potential", "actionability"],
            "data_steward": ["data_lineage", "governance_compliance", "traceability"],
        }

        for agent_id, agent in self._agents.items():
            agent_type = getattr(agent, 'agent_type', 'unknown')
            council_role = AGENT_ROLE_MAPPING.get(agent_type, "data_steward")
            evaluation_focus = ROLE_EVALUATION_FOCUS.get(council_role, ["general_quality"])

            council_agent = CouncilAgent(
                name=agent.agent_name,
                role=council_role,
                evaluation_focus=evaluation_focus,
                weight=1.0,  # 모든 에이전트 동일 가중치
                llm_client=getattr(agent, 'llm_client', None),
            )
            council_agents[agent_id] = council_agent

        logger.info(f"Created {len(council_agents)} CouncilAgents for evidence-based voting")
        return council_agents

    async def conduct_evidence_based_vote(
        self,
        proposal: str,
        evidence_chain: "EvidenceChain",
        context: Dict[str, Any] = None,
    ) -> Dict[str, Any]:
        """
        v7.8: Evidence-based 투표 수행 (CouncilAgent 사용)

        Args:
            proposal: 평가할 제안
            evidence_chain: 근거 체인
            context: 추가 컨텍스트

        Returns:
            투표 결과 {
                "votes": [...],
                "consensus": str,
                "confidence": float,
                "role_breakdown": {...}
            }
        """
        if not COUNCIL_AGENT_AVAILABLE:
            logger.warning("CouncilAgent not available - falling back to standard voting")
            return {"fallback": True, "reason": "CouncilAgent not available"}

        council_agents = self._create_council_agents()
        if not council_agents:
            return {"fallback": True, "reason": "No council agents created"}

        # === v22.0: 투표 시작 이벤트 발행 ===
        await self._publish_consensus_event(
            event_type="round_start",
            data={
                "round_type": "evidence_based_vote",
                "proposal_summary": proposal[:100] if proposal else "",
                "council_agent_count": len(council_agents),
                "evidence_block_count": len(evidence_chain.get_all_blocks()) if evidence_chain else 0,
            },
        )

        votes: List[AgentVoteWithEvidence] = []
        role_votes: Dict[str, List[str]] = {
            "ontologist": [],
            "risk_assessor": [],
            "business_strategist": [],
            "data_steward": [],
        }

        # 각 CouncilAgent로부터 투표 수집
        for agent_id, council_agent in council_agents.items():
            try:
                vote = council_agent.evaluate_proposal(
                    proposal=proposal,
                    evidence_chain=evidence_chain,
                    context=context,
                )
                votes.append(vote)
                role_votes[council_agent.role].append(vote.decision.value)

                logger.debug(
                    f"[CouncilVote] {council_agent.name} ({council_agent.role}): "
                    f"{vote.decision.value} (conf={vote.confidence:.2f})"
                )

                # === v22.0: 개별 투표를 Agent Bus로 브로드캐스트 ===
                await self._publish_consensus_event(
                    event_type="vote_cast",
                    data={
                        "agent_id": agent_id,
                        "agent_name": council_agent.name,
                        "agent_role": council_agent.role,
                        "decision": vote.decision.value,
                        "confidence": vote.confidence,
                        "cited_evidence_count": len(vote.cited_evidence),
                    },
                )
            except Exception as e:
                logger.warning(f"CouncilAgent {council_agent.name} failed to vote: {e}")

        # 합의 결정
        if not votes:
            return {"fallback": True, "reason": "No votes collected"}

        approve_count = sum(1 for v in votes if v.decision == VoteDecision.APPROVE)
        reject_count = sum(1 for v in votes if v.decision == VoteDecision.REJECT)
        total_count = len(votes)

        approve_ratio = approve_count / total_count
        avg_confidence = sum(v.confidence for v in votes) / total_count

        # 합의 유형 결정
        if approve_ratio >= self.strong_consensus_threshold:
            consensus = "strong_approve"
        elif approve_ratio >= self.acceptance_threshold:
            consensus = "approve"
        elif approve_ratio <= self.rejection_threshold:
            consensus = "reject"
        else:
            consensus = "review"

        # 역할별 분석
        role_breakdown = {}
        for role, decisions in role_votes.items():
            if decisions:
                approve_pct = decisions.count("approve") / len(decisions) * 100
                role_breakdown[role] = {
                    "approve_pct": approve_pct,
                    "votes": decisions,
                    "count": len(decisions),
                }

        result = {
            "consensus": consensus,
            "confidence": avg_confidence,
            "approve_ratio": approve_ratio,
            "votes": [
                {
                    "agent": v.agent_name,
                    "role": v.agent_role,
                    "decision": v.decision.value,
                    "confidence": v.confidence,
                    "reasoning": v.reasoning[:200] + "..." if len(v.reasoning) > 200 else v.reasoning,
                    "cited_evidence": v.cited_evidence[:3],
                }
                for v in votes
            ],
            "role_breakdown": role_breakdown,
            "total_votes": total_count,
        }

        logger.info(
            f"[EvidenceBasedVote] {consensus} (approve={approve_ratio:.0%}, "
            f"conf={avg_confidence:.2f}, votes={total_count})"
        )

        # === v22.0: Agent Bus로 합의 결과 브로드캐스트 ===
        # Evidence-based vote 결과를 모든 에이전트에게 알림
        await self._publish_consensus_event(
            event_type="consensus_reached",
            data={
                "consensus_type": "evidence_based",
                "proposal_summary": proposal[:100] if proposal else "",
                "consensus": consensus,
                "confidence": avg_confidence,
                "approve_ratio": approve_ratio,
                "total_votes": total_count,
                "role_breakdown_summary": {
                    role: data.get("approve_pct", 0)
                    for role, data in role_breakdown.items()
                },
                "evidence_block_count": len(evidence_chain.get_all_blocks()) if evidence_chain else 0,
            },
        )

        return result

    # --------------------------------------------------------
    # v7.8: StructuredDebate Integration (C-4 Fix)
    # --------------------------------------------------------

    async def conduct_structured_debate(
        self,
        topic: str,
        context: Dict[str, Any],
        evidence: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        v7.8: 구조화된 3라운드 토론 수행

        합의에 도달하지 못한 복잡한 안건에 대해 심층 토론을 진행합니다.

        Args:
            topic: 토론 주제
            context: 컨텍스트 정보
            evidence: 증거 데이터

        Returns:
            토론 결과 {
                "outcome": DebateOutcome,
                "final_decision": str,
                "conditions": [...],
                "dissenting_views": [...]
            }
        """
        if not STRUCTURED_DEBATE_AVAILABLE:
            logger.warning("StructuredDebateEngine not available")
            return {"fallback": True, "reason": "StructuredDebate not available"}

        # 에이전트별 관점 수집
        agents_perspectives = {}
        for agent_id, agent in self._agents.items():
            agent_type = getattr(agent, 'agent_type', 'unknown')
            council_role = AGENT_ROLE_MAPPING.get(agent_type, "data_steward")

            # 역할별 초기 관점 설정
            role_perspectives = {
                "ontologist": f"From a semantic/structural perspective on '{topic}'",
                "risk_assessor": f"Assessing risks and quality concerns for '{topic}'",
                "business_strategist": f"Evaluating business value and ROI of '{topic}'",
                "data_steward": f"Checking governance compliance for '{topic}'",
            }
            agents_perspectives[agent.agent_name] = role_perspectives.get(
                council_role,
                f"General evaluation of '{topic}'"
            )

        # StructuredDebateEngine 인스턴스 생성
        debate_engine = StructuredDebateEngine(
            llm_client=None,  # 에이전트들이 자체 LLM 사용
            max_rounds=3,
            consensus_threshold=self.acceptance_threshold,
        )

        try:
            # 토론 수행
            outcome = await debate_engine.conduct_debate(
                topic=topic,
                context=context,
                agents_perspectives=agents_perspectives,
                evidence=evidence,
            )

            result = {
                "topic": topic,
                "final_decision": outcome.verdict,
                "consensus_level": outcome.rounds[-1].consensus_level if outcome.rounds else 0.0,
                "rounds_completed": len(outcome.rounds),
                "conditions": outcome.conditions,
                "dissenting_views": outcome.dissenting_views,
                "recommendations": outcome.recommendations,
                "metadata": {
                    "early_consensus": outcome.metadata.get("early_consensus", False),
                    "total_arguments": sum(len(r.arguments) for r in outcome.rounds),
                },
            }

            logger.info(
                f"[StructuredDebate] {outcome.verdict} "
                f"(consensus={outcome.rounds[-1].consensus_level:.0%}, "
                f"rounds={len(outcome.rounds)})"
            )

            return result

        except Exception as e:
            logger.error(f"StructuredDebate failed: {e}")
            return {
                "fallback": True,
                "reason": str(e),
                "topic": topic,
            }

    async def negotiate(
        self,
        topic_id: str,
        topic_name: str,
        topic_type: str,
        topic_data: Dict[str, Any],
        context: "SharedContext",
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        비동기 병렬 토론 진행 (Optimized Flow)
        """
        # 상태 초기화
        self._current_discussion = ConsensusState(
            topic_id=topic_id,
            topic_name=topic_name,
            topic_type=topic_type,
            acceptance_threshold=self.acceptance_threshold,
            rejection_threshold=self.rejection_threshold,
            provisional_threshold=self.provisional_threshold,
            min_agreement=self.min_agreement,
        )
        self.channel.clear_history()

        # 로깅 시작
        if self._conversation_logger:
            self._current_discussion_log = self._conversation_logger.start_discussion(
                topic_id=topic_id,
                topic_name=topic_name,
                topic_type=topic_type,
                topic_data=topic_data,
            )

        yield {
            "type": "discussion_started",
            "topic_id": topic_id,
            "topic_name": topic_name,
            "participating_agents": list(self._agents.keys()),
        }

        # === Round 1: Parallel Initial Vote (v15.0 - Group-think Prevention) ===
        # Agent Drift 논문: 독립적 1차 투표로 Group-think 방지
        round_num = 1
        yield {"type": "round_started", "round": round_num, "mode": "independent_parallel_vote"}

        current_round = self._current_discussion.start_new_round()
        active_agents = list(self._agents.values())

        # v15.0: 에이전트 순서 무작위화 (순서 편향 방지)
        import random
        random.shuffle(active_agents)

        # v14.0 Enhanced: 빈 에이전트 처리 - 기본 결과 반환
        if not active_agents:
            logger.warning(f"[v14.0] No agents registered for consensus on topic: {topic_id}")

            # v14.0: NO_AGENTS 타입의 기본 결과 생성
            no_agent_result = ConsensusResult(
                success=False,
                result_type=ConsensusType.NO_AGENTS,
                combined_confidence=0.0,
                agreement_level=0.0,
                total_rounds=0,
                reasoning="No agents were registered for consensus. Using default provisional acceptance.",
            )

            yield {
                "type": "no_agents",
                "error": "No agents registered for consensus",
                "result_type": ConsensusType.NO_AGENTS.value,
                "result": no_agent_result,
            }
            return

        # Async Parallel Thinking: 모든 에이전트 동시 호출
        tasks = []
        for agent in active_agents:
            tasks.append(self._get_agent_response_task(
                agent=agent,
                topic_data=topic_data,
                context=context,
                round_num=round_num,
                previous_discussion="" # 1라운드는 이전 대화 없음
            ))
        
        # 결과 수집
        responses = await asyncio.gather(*tasks, return_exceptions=True)

        for i, response in enumerate(responses):
            agent = active_agents[i]
            if isinstance(response, Exception):
                logger.error(f"Agent {agent.agent_id} failed: {response}")
                continue
            
            # 파싱 및 등록
            opinion = self._parse_opinion(agent, response, round_num)
            await self.channel.broadcast(agent.agent_id, response, opinion)
            self._current_discussion.add_opinion(opinion)
            
            yield {
                "type": "agent_opinion",
                "agent_id": agent.agent_id,
                "confidence": opinion.confidence,
                "vote": opinion.legacy_vote
            }

        # v15.0: Group-think Detection - 다양성 검사
        vote_diversity = self._check_vote_diversity()
        if vote_diversity["is_suspicious"]:
            logger.warning(
                f"[GROUP-THINK] Suspicious voting pattern detected: "
                f"diversity_score={vote_diversity['diversity_score']:.2f}, "
                f"dominant_vote={vote_diversity['dominant_vote']} ({vote_diversity['dominant_ratio']:.0%})"
            )
            yield {
                "type": "group_think_warning",
                "diversity_score": vote_diversity["diversity_score"],
                "dominant_vote": vote_diversity["dominant_vote"],
                "dominant_ratio": vote_diversity["dominant_ratio"],
            }

        # === Fast-track Check (v7.0 강화) ===
        reached, result_type = self._current_discussion.check_consensus()

        # v7.0: Fast-track 조건 완화 - 60% 이상이면 즉시 종료
        combined_conf = self._current_discussion.combined_confidence
        is_unanimous = all(op.legacy_vote == VoteType.APPROVE for op in self._current_discussion.all_opinions.values())
        is_fast_track_eligible = combined_conf >= self.fast_track_threshold  # 60%+

        # Fast-track: 만장일치 OR 60%+ 신뢰도면 즉시 종료
        if is_unanimous or is_fast_track_eligible:
            logger.info(f"Fast-track consensus: {topic_id} (conf={combined_conf:.2f}, unanimous={is_unanimous})")

            # 신뢰도에 따른 결과 유형 결정
            if combined_conf >= self.acceptance_threshold:
                result_type = ConsensusType.ACCEPTED
            elif combined_conf >= self.provisional_threshold:
                result_type = ConsensusType.PROVISIONAL
            else:
                result_type = ConsensusType.PROVISIONAL  # 60%+는 최소 PROVISIONAL

            result = self._finalize_result(result_type)
            yield {
                "type": "consensus_reached",
                "result_type": result_type.value,
                "mode": "fast_track",
                "confidence": combined_conf,
                "result": result
            }
            return

        # === Round 2: 한번만 추가 토론 (max_rounds=2) ===
        if self.max_rounds > 1:
            yield {"type": "debate_needed", "reason": "conflict_detected"}

            # Convergence Detection을 위한 이전 라운드 상태 저장
            prev_vote_distribution = self._get_vote_distribution()
            convergence_count = 0  # 연속 수렴 횟수
            CONVERGENCE_THRESHOLD = 2  # 2회 연속 변화 없으면 종료

            # 반대자/의심자 위주로 토론 진행 (max_rounds는 안전장치)
            for round_num in range(2, self.max_rounds + 1):
                yield {"type": "round_started", "round": round_num, "mode": "deep_debate"}
                current_round = self._current_discussion.start_new_round()

                # 이전 라운드 요약
                prev_summary = self.channel.get_conversation_context(limit=5)

                # Async Parallel Thinking - 모든 에이전트 동시 투표
                tasks = []
                for agent in active_agents:
                    tasks.append(self._get_agent_response_task(
                        agent=agent,
                        topic_data=topic_data,
                        context=context,
                        round_num=round_num,
                        previous_discussion=prev_summary
                    ))

                responses = await asyncio.gather(*tasks, return_exceptions=True)

                # 결과 처리
                for i, response in enumerate(responses):
                    agent = active_agents[i]
                    if isinstance(response, Exception): continue

                    opinion = self._parse_opinion(agent, response, round_num)
                    self._current_discussion.add_opinion(opinion)

                # 합의 재확인
                reached, result_type = self._current_discussion.check_consensus()
                if reached:
                    result = self._finalize_result(result_type)
                    yield {
                        "type": "consensus_reached",
                        "result_type": result_type.value,
                        "mode": "debate_resolved",
                        "round": round_num,
                        "result": result
                    }
                    return

                # === Convergence Detection ===
                # 이번 라운드 투표 분포와 이전 라운드 비교
                current_vote_distribution = self._get_vote_distribution()
                vote_change = self._calculate_vote_change(prev_vote_distribution, current_vote_distribution)

                if vote_change < 0.1:  # 10% 미만 변화 = 수렴 중
                    convergence_count += 1
                    yield {
                        "type": "convergence_detected",
                        "round": round_num,
                        "vote_change": vote_change,
                        "convergence_count": convergence_count
                    }

                    if convergence_count >= CONVERGENCE_THRESHOLD:
                        # 더 이상 의견 변화 없음 → 조기 종료
                        logger.info(f"Convergence detected after {round_num} rounds (no significant change)")
                        final_result = self._finalize_convergence()
                        yield {
                            "type": "consensus_converged",
                            "result_type": final_result.result_type.value,
                            "mode": "convergence_early_stop",
                            "round": round_num,
                            "result": final_result
                        }
                        return
                else:
                    convergence_count = 0  # 변화가 있으면 리셋

                prev_vote_distribution = current_vote_distribution

        # === Timeout (Max Rounds - 안전장치) ===
        final_result = self._finalize_timeout()
        yield {
            "type": "max_rounds_reached",
            "result_type": final_result.result_type.value,
            "result": final_result
        }

    async def _get_agent_response_task(
        self,
        agent: "AutonomousAgent",
        topic_data: Dict[str, Any],
        context: "SharedContext",
        round_num: int,
        previous_discussion: str
    ) -> str:
        """단일 에이전트 응답 생성 태스크 (Wrapper)"""
        return await self._get_agent_response(agent, topic_data, context, round_num, previous_discussion)

    async def _get_agent_response(
        self,
        agent: "AutonomousAgent",
        topic_data: Dict[str, Any],
        context: "SharedContext",
        round_num: int,
        previous_discussion: str,
    ) -> str:
        """LLM 호출 및 프롬프트 구성"""
        
        # 다른 에이전트 의견 요약 (라운드 1은 비어있음)
        other_opinions = ""
        if round_num > 1:
            for agent_id, op in self._current_discussion.all_opinions.items():
                if agent_id != agent.agent_id:
                    other_opinions += f"- {op.agent_name}: {op.legacy_vote} (conf={op.confidence:.2f})\n"

        topic_name = topic_data.get('description', topic_data.get('name', 'Discussion'))
        prompt = f"""## Topic: {topic_name}
        
## Data
{self._format_topic_data(topic_data)}

## Context
Round: {round_num}
Previous Discussion:
{previous_discussion if previous_discussion else "(None - Initial Vote)"}

## Other Opinions
{other_opinions if other_opinions else "(None yet)"}

## Task
Evaluate the proposal.
- If Round 1: Provide independent assessment based on Data.
- If Round 2+: Address conflicts and refine your stance.

## Required Output Format
[CONFIDENCE: 0.XX]
[VOTE: APPROVE | REJECT | ABSTAIN]
[REASONING]
...
"""
        # 실제 LLM 호출 (agent.call_llm은 이미 비동기라 가정)
        # 템플릿 메서드 호출
        # v8.0: API 재시도 로직 적용
        return await self._call_agent_llm_with_retry(
            agent, topic_data, other_opinions, previous_discussion, round_num
        )

    async def _call_agent_llm_with_retry(
        self,
        agent,
        topic_data,
        other_opinions,
        prev_discussion,
        round_num,
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ) -> str:
        """
        v8.0: API 호출 실패 시 재시도 로직

        - 빈 응답 감지 및 재시도
        - 지수 백오프
        - 최대 재시도 횟수 제한
        """
        last_error = None

        for attempt in range(max_retries):
            try:
                response = await self._call_agent_llm(
                    agent, topic_data, other_opinions, prev_discussion, round_num
                )

                # 빈 응답 체크
                if not response or len(response.strip()) < 20:
                    logger.warning(
                        f"Empty response from {agent.agent_name} "
                        f"(attempt {attempt + 1}/{max_retries})"
                    )
                    if attempt < max_retries - 1:
                        await asyncio.sleep(retry_delay * (2 ** attempt))
                        continue
                    else:
                        # 최대 재시도 후에도 빈 응답 → 기본 응답 반환
                        return self._generate_fallback_response(agent)

                return response

            except Exception as e:
                last_error = e
                logger.warning(
                    f"API call failed for {agent.agent_name}: {e} "
                    f"(attempt {attempt + 1}/{max_retries})"
                )
                if attempt < max_retries - 1:
                    await asyncio.sleep(retry_delay * (2 ** attempt))

        # 모든 재시도 실패
        logger.error(f"All retries failed for {agent.agent_name}: {last_error}")
        return self._generate_fallback_response(agent)

    def _generate_fallback_response(self, agent) -> str:
        """v8.0: API 실패 시 기본 응답 생성 (ABSTAIN with low confidence)"""
        return f"""[CONFIDENCE: 0.4]
[UNCERTAINTY: 0.8]
[DOUBT: 0.3]
[EVIDENCE: 0.2]

[OBSERVATIONS]
- API 호출 실패로 인해 {agent.agent_name}의 완전한 평가 불가
- 다른 에이전트의 의견에 의존 필요

[CONCERNS]
- 완전한 분석이 수행되지 않음

[SUGGESTIONS]
- 다른 에이전트의 평가 결과 우선 적용 권장
"""

    async def _call_agent_llm(self, agent, topic_data, other_opinions, prev_discussion, round_num):
        """v8.0: 에이전트 전문성을 반영한 합의 프롬프트 생성"""

        # 에이전트별 전문 역할 정의
        AGENT_CONSENSUS_ROLES = {
            "ontology_architect": {
                "focus": "온톨로지 구조, 계층 관계, 명명 규칙",
                "evaluate": "concept_type, hierarchy, naming_convention",
                "perspective": "설계 일관성과 확장성",
            },
            "quality_judge": {
                "focus": "품질 메트릭, 완전성, 정확도",
                "evaluate": "confidence_scores, evidence_strength, coverage",
                "perspective": "품질 기준 충족 여부",
            },
            "semantic_validator": {
                "focus": "의미론적 일관성, OWL/RDFS 규칙",
                "evaluate": "semantic_consistency, inference_validity",
                "perspective": "논리적 무결성",
            },
            "conflict_resolver": {
                "focus": "명명 충돌, 중복, 모호성",
                "evaluate": "naming_conflicts, duplicate_concepts",
                "perspective": "충돌 해결 우선순위",
            },
            "governance_strategist": {
                "focus": "비즈니스 영향, 구현 가능성",
                "evaluate": "business_impact, implementation_risk",
                "perspective": "조직 수용 가능성",
            },
            "risk_assessor": {
                "focus": "리스크 식별, 완화 전략",
                "evaluate": "risk_factors, mitigation_options",
                "perspective": "리스크 대비 가치",
            },
            "action_prioritizer": {
                "focus": "액션 우선순위, ROI 분석",
                "evaluate": "priority_score, dependency_order",
                "perspective": "실행 효율성",
            },
            "policy_generator": {
                "focus": "정책 규칙, 컴플라이언스",
                "evaluate": "policy_rules, compliance_requirements",
                "perspective": "정책 적용 가능성",
            },
        }

        # 에이전트 역할 정보 가져오기
        agent_type = agent.agent_type if hasattr(agent, 'agent_type') else "unknown"
        role_info = AGENT_CONSENSUS_ROLES.get(agent_type, {
            "focus": "전반적인 데이터 품질 평가",
            "evaluate": "overall_quality, consistency",
            "perspective": "일반적 관점",
        })

        # v7.8: Council 역할 및 역할별 프롬프트 통합 (C-1, C-2 Fix)
        council_role = AGENT_ROLE_MAPPING.get(agent_type, "data_steward")
        council_prompt = ROLE_SPECIFIC_PROMPTS.get(council_role, ROLE_SPECIFIC_PROMPTS["data_steward"])
        council_thresholds = ROLE_DECISION_THRESHOLDS.get(council_role, {"approve": 0.60, "review": 0.35})

        # v15.0: Group-think 방지를 위한 독립성 강조 프롬프트
        independence_notice = ""
        if round_num == 1:
            independence_notice = """
⚠️ **IMPORTANT: INDEPENDENT JUDGMENT REQUIRED**
This is Round 1. You MUST form your opinion INDEPENDENTLY based solely on:
1. The data provided below
2. Your professional expertise
3. Your own analysis

DO NOT be influenced by assumptions about what others might think.
Your unique perspective is valuable - disagreement is encouraged if warranted by evidence.
"""

        # v8.0: 전문화된 프롬프트 생성 (v7.8: Council 역할 통합)
        prompt = f"""## 합의 토론: {topic_data.get('todo_name', 'Unknown Task')}
{independence_notice}
{council_prompt}

### 당신의 역할: {agent.agent_name} (Council Role: {council_role.upper()})
- **전문 분야**: {role_info['focus']}
- **평가 관점**: {role_info['perspective']}
- **주요 평가 항목**: {role_info['evaluate']}
- **결정 기준**: APPROVE >= {council_thresholds['approve']:.0%}, REVIEW >= {council_thresholds['review']:.0%}

### 평가 대상 데이터
{self._format_topic_data_for_agent(topic_data, agent_type)}

### 라운드 {round_num}
**이전 토론 요약**:
{prev_discussion if prev_discussion else "(첫 라운드 - 독립적 평가 수행)"}

**다른 에이전트 의견**:
{other_opinions if other_opinions else "(아직 의견 없음 - 독립적으로 판단하세요)"}

### 당신의 태스크
**{agent.agent_name}** ({council_role})로서 전문 관점에서 평가하세요:

1. **{role_info['focus']}** 관점에서 데이터 분석
2. 제시된 증거의 타당성 검토
3. 다른 에이전트 의견에 대한 동의/반대 표명 (라운드 2+)

### 응답 형식 (필수)
[CONFIDENCE: 0.0-1.0]  # 제안에 대한 신뢰도
[UNCERTAINTY: 0.0-1.0]  # 불확실한 부분
[DOUBT: 0.0-1.0]  # 의심되는 부분
[EVIDENCE: 0.0-1.0]  # 근거의 강도

[OBSERVATIONS]
- {role_info['focus']} 관점에서의 핵심 관찰 (2-3개)

[CONCERNS]
- 발견된 문제점이나 리스크 (있다면)

[SUGGESTIONS]
- 개선 제안 (있다면)
"""
        return await agent.call_llm(
            user_prompt=prompt,
            temperature=0.3,
            max_tokens=1500
        )

    def _format_topic_data_for_agent(self, topic_data: Dict[str, Any], agent_type: str) -> str:
        """v8.0: 에이전트 타입에 따라 관련 데이터만 포맷팅"""
        import json

        # 기본 정보
        result_parts = []

        # 모든 에이전트에게 공통 정보
        if "ontology_summary" in topic_data:
            result_parts.append(f"온톨로지 요약: {json.dumps(topic_data['ontology_summary'], ensure_ascii=False)}")

        if "phase1_evidence" in topic_data:
            result_parts.append(f"Phase 1 증거: {json.dumps(topic_data['phase1_evidence'], ensure_ascii=False)}")

        # 에이전트별 특화 정보
        if agent_type in ["ontology_architect", "quality_judge", "semantic_validator"]:
            if "concepts_to_evaluate" in topic_data:
                concepts = topic_data["concepts_to_evaluate"][:5]  # 상위 5개
                result_parts.append(f"평가 대상 개념 ({len(concepts)}개):")
                for c in concepts:
                    result_parts.append(f"  - {c.get('name')}: type={c.get('type')}, conf={c.get('confidence', 0):.2f}")

        if agent_type == "conflict_resolver":
            if "conflicts_to_resolve" in topic_data:
                conflicts = topic_data["conflicts_to_resolve"][:5]
                result_parts.append(f"해결할 충돌 ({len(conflicts)}개):")
                for c in conflicts:
                    result_parts.append(f"  - {c.get('concept_a')} vs {c.get('concept_b')}: similarity={c.get('similarity', 0):.2f}")

        if agent_type in ["governance_strategist", "action_prioritizer"]:
            if "approved_ontology" in topic_data:
                result_parts.append(f"승인된 온톨로지: {json.dumps(topic_data['approved_ontology'], ensure_ascii=False)}")
            if "business_insights" in topic_data:
                insights = topic_data["business_insights"][:3]
                result_parts.append(f"비즈니스 인사이트 ({len(insights)}개):")
                for ins in insights:
                    result_parts.append(f"  - [{ins.get('type')}] {ins.get('description', '')[:100]}")

        if agent_type == "risk_assessor":
            if "potential_risks" in topic_data:
                risks = topic_data["potential_risks"]
                result_parts.append(f"잠재적 리스크 ({len(risks)}개):")
                for r in risks:
                    result_parts.append(f"  - [{r.get('severity')}] {r.get('type')}: {r.get('description', '')}")
            if "risk_context" in topic_data:
                result_parts.append(f"리스크 컨텍스트: {json.dumps(topic_data['risk_context'], ensure_ascii=False)}")

        # 결과가 없으면 전체 데이터 표시
        if not result_parts:
            return json.dumps(topic_data, indent=2, ensure_ascii=False, default=str)[:2000]

        return "\n".join(result_parts)

    def _format_topic_data(self, topic_data: Dict[str, Any]) -> str:
        # 기존 로직 유지 (JSON dumps 등)
        import json
        return json.dumps(topic_data, indent=2, ensure_ascii=False, default=str)[:2000]

    def _parse_opinion(
        self,
        agent: "AutonomousAgent",
        response: str,
        round_num: int,
    ) -> AgentOpinion:
        """ 의견 파싱 (기존 로직 유지) """
        confidence = self._parse_score(response, "CONFIDENCE", default=0.65)
        uncertainty = self._parse_score(response, "UNCERTAINTY", default=0.15)
        doubt = self._parse_score(response, "DOUBT", default=0.05)
        evidence = self._parse_score(response, "EVIDENCE", default=0.6)

        observations = self._parse_section(response, "OBSERVATIONS")
        concerns = self._parse_section(response, "CONCERNS")
        suggestions = self._parse_section(response, "SUGGESTIONS")
        
        # Mentions 파싱 등은 기존 로직과 동일
        
        return AgentOpinion(
            agent_id=agent.agent_id,
            agent_type=agent.agent_type,
            agent_name=agent.agent_name,
            confidence=confidence,
            uncertainty=uncertainty,
            doubt=doubt,
            evidence_weight=evidence,
            reasoning=response[:1000],
            key_observations=observations,
            concerns=concerns,
            suggestions=suggestions,
            round_number=round_num,
        )

    def _parse_score(self, text: str, tag: str, default: float) -> float:
        match = re.search(rf'\[{tag}:\s*([\d.]+)\]', text, re.IGNORECASE)
        if match:
            try:
                return float(match.group(1))
            except: pass
        return default

    def _parse_section(self, text: str, section: str) -> List[str]:
        match = re.search(rf'\[{section}\](.*?)(?:\[|$)', text, re.IGNORECASE | re.DOTALL)
        if match:
            return [line.strip('- ').strip() for line in match.group(1).split('\n') if line.strip('- ').strip()]
        return []

    def _finalize_result(self, result_type: ConsensusType) -> ConsensusResult:
        """결과 객체 생성 (EvidenceBundle 포함)"""
        state = self._current_discussion
        
        evidence_bundle = self._create_evidence_bundle(state, result_type)
        
        return ConsensusResult(
            success=True,
            result_type=result_type,
            combined_confidence=state.combined_confidence,
            agreement_level=state.agreement_level,
            total_rounds=state.current_round,
            final_decision="approve" if result_type == ConsensusType.ACCEPTED else "reject",
            reasoning=f"Consensus reached. Confidence: {state.combined_confidence:.2f}",
            dissenting_views=[],
            consensus_state=state,
            evidence_bundle=evidence_bundle
        )

    def _finalize_timeout(self) -> ConsensusResult:
        """v6.3: Max rounds 소진 시 결과 - PROVISIONAL도 성공으로 처리"""
        state = self._current_discussion
        state.calculate_consensus()

        # v6.3: PROVISIONAL은 성공으로 처리 (합의가 이루어진 것으로 간주)
        # 단, ESCALATED만 실패로 처리
        if state.combined_confidence >= self.acceptance_threshold:
            result_type = ConsensusType.ACCEPTED
            final_decision = "approve"
            success = True
        elif state.combined_confidence >= 0.4:  # provisional threshold
            result_type = ConsensusType.PROVISIONAL
            final_decision = "provisional"
            success = True  # v6.3: PROVISIONAL도 성공
        else:
            result_type = ConsensusType.ESCALATED
            final_decision = "escalate"
            success = False  # 에스컬레이션만 실패

        return ConsensusResult(
            success=success,
            result_type=result_type,
            combined_confidence=state.combined_confidence,
            agreement_level=state.agreement_level,
            total_rounds=state.current_round,
            final_decision=final_decision,
            reasoning=f"Max rounds reached. Confidence: {state.combined_confidence:.2f}",
            dissenting_views=[],
            consensus_state=state
        )

    def _finalize_convergence(self) -> ConsensusResult:
        """수렴 감지 시 결과 생성 - 현재 상태에서 최선의 결정"""
        state = self._current_discussion
        state.calculate_consensus()

        # 수렴된 상태에서 결정: confidence 기준으로 판단
        if state.combined_confidence >= self.acceptance_threshold:
            result_type = ConsensusType.ACCEPTED
            final_decision = "approve"
            success = True
        elif state.combined_confidence >= self.provisional_threshold:
            result_type = ConsensusType.PROVISIONAL
            final_decision = "provisional"
            success = True  # 수렴으로 종료되었으므로 성공으로 간주
        else:
            result_type = ConsensusType.REJECTED
            final_decision = "reject"
            success = True  # 명확한 거부도 성공적인 합의

        return ConsensusResult(
            success=success,
            result_type=result_type,
            combined_confidence=state.combined_confidence,
            agreement_level=state.agreement_level,
            total_rounds=state.current_round,
            final_decision=final_decision,
            reasoning=f"Convergence detected - no significant opinion change. Final confidence: {state.combined_confidence:.2f}",
            dissenting_views=[],
            consensus_state=state
        )

    def _check_vote_diversity(self) -> Dict[str, Any]:
        """
        v15.0: Group-think 탐지를 위한 투표 다양성 검사

        다양성 점수가 낮고 특정 투표가 지배적이면 Group-think 의심

        Returns:
            {
                "diversity_score": float,  # 0-1, 높을수록 다양
                "is_suspicious": bool,
                "dominant_vote": str,
                "dominant_ratio": float,
            }
        """
        state = self._current_discussion
        if not state or not state.all_opinions:
            return {
                "diversity_score": 1.0,
                "is_suspicious": False,
                "dominant_vote": None,
                "dominant_ratio": 0.0,
            }

        opinions = list(state.all_opinions.values())
        total = len(opinions)

        if total < 2:
            return {
                "diversity_score": 1.0,
                "is_suspicious": False,
                "dominant_vote": None,
                "dominant_ratio": 0.0,
            }

        # 투표 유형별 카운트
        vote_counts = {}
        confidence_values = []

        for op in opinions:
            vote = op.legacy_vote.value if hasattr(op.legacy_vote, 'value') else str(op.legacy_vote)
            vote_counts[vote] = vote_counts.get(vote, 0) + 1
            confidence_values.append(op.confidence)

        # 지배적 투표 찾기
        dominant_vote = max(vote_counts, key=vote_counts.get)
        dominant_count = vote_counts[dominant_vote]
        dominant_ratio = dominant_count / total

        # Simpson's Diversity Index 계산
        # D = 1 - Σ(n_i * (n_i - 1)) / (N * (N - 1))
        sum_ni = sum(n * (n - 1) for n in vote_counts.values())
        if total > 1:
            simpson_diversity = 1 - (sum_ni / (total * (total - 1)))
        else:
            simpson_diversity = 1.0

        # Confidence 분산 검사 (모두 비슷하면 의심)
        import statistics
        if len(confidence_values) > 1:
            conf_stdev = statistics.stdev(confidence_values)
        else:
            conf_stdev = 0.0

        # 종합 다양성 점수
        diversity_score = (simpson_diversity * 0.6) + (min(conf_stdev * 2, 0.4))

        # Group-think 의심 조건:
        # 1. 다양성 점수가 0.2 미만
        # 2. 한 투표가 85% 이상 지배
        # 3. Confidence 분산이 0.1 미만 (모두 비슷한 확신)
        is_suspicious = (
            diversity_score < 0.2 and
            dominant_ratio > 0.85 and
            conf_stdev < 0.1
        )

        return {
            "diversity_score": round(diversity_score, 3),
            "is_suspicious": is_suspicious,
            "dominant_vote": dominant_vote,
            "dominant_ratio": round(dominant_ratio, 3),
            "confidence_stdev": round(conf_stdev, 3),
        }

    def _get_vote_distribution(self) -> Dict[str, float]:
        """현재 투표 분포 계산"""
        state = self._current_discussion
        if not state or not state.all_opinions:
            return {"approve": 0.0, "reject": 0.0, "abstain": 0.0}

        total = len(state.all_opinions)
        if total == 0:
            return {"approve": 0.0, "reject": 0.0, "abstain": 0.0}

        approve_count = sum(1 for op in state.all_opinions.values() if op.legacy_vote == VoteType.APPROVE)
        reject_count = sum(1 for op in state.all_opinions.values() if op.legacy_vote == VoteType.REJECT)
        abstain_count = total - approve_count - reject_count

        return {
            "approve": approve_count / total,
            "reject": reject_count / total,
            "abstain": abstain_count / total,
            "avg_confidence": state.combined_confidence
        }

    def _calculate_vote_change(self, prev: Dict[str, float], current: Dict[str, float]) -> float:
        """두 라운드 간 투표 변화율 계산 (0.0 ~ 1.0)"""
        if not prev or not current:
            return 1.0  # 첫 라운드는 항상 큰 변화로 간주

        # 각 카테고리별 변화량의 평균
        changes = []
        for key in ["approve", "reject", "abstain"]:
            prev_val = prev.get(key, 0.0)
            curr_val = current.get(key, 0.0)
            changes.append(abs(curr_val - prev_val))

        # confidence 변화도 고려
        prev_conf = prev.get("avg_confidence", 0.5)
        curr_conf = current.get("avg_confidence", 0.5)
        changes.append(abs(curr_conf - prev_conf))

        return sum(changes) / len(changes) if changes else 0.0

    def _create_evidence_bundle(self, state, result_type):
        if not EVIDENCE_AVAILABLE: return None
        # 기존 EvidenceBundle 생성 로직 복원 (약식)
        try:
            bundle = EvidenceBundle(state.topic_id, state.topic_name, state.topic_type)
            bundle.semantic_evidence.semantic_match_score = state.combined_confidence
            return bundle
        except: return None


# ============================================================
# Debate Protocol (Phase 3)
# ============================================================

class DebateProtocol:
    """
    Debate Protocol - 구조화된 토론 프로토콜

    Phase 3 스펙에 따른 Debate Flow:
    1. Challenge Initiation: 반대 의견 에이전트가 challenge 발언권
    2. Evidence Presentation: TDA, 통계 결과를 증거로 제시
    3. Revote: 새로운 증거 기반 재투표
    4. Escalation: 합의 실패 시 human escalation
    """

    def __init__(
        self,
        consensus_engine: "PipelineConsensusEngine",
        round_timeout: float = 0,            # v18.0: 타임아웃 비활성화 (무제한)
        max_debate_rounds: int = 3,          # 최대 토론 라운드
        escalation_threshold: int = 2,       # escalation까지 실패 횟수
    ):
        self.consensus_engine = consensus_engine
        self.round_timeout = round_timeout
        self.max_debate_rounds = max_debate_rounds
        self.escalation_threshold = escalation_threshold

        # 상태 관리
        self._active_challenges: Dict[str, Challenge] = {}
        self._debate_rounds: List[DebateRound] = []
        self._escalations: List[HumanEscalation] = []
        self._failed_consensus_count: int = 0

        # 이벤트 핸들러 (옵션)
        self._on_challenge_initiated: Optional[callable] = None
        self._on_evidence_presented: Optional[callable] = None
        self._on_revote_completed: Optional[callable] = None
        self._on_escalated: Optional[callable] = None

        logger.info(f"DebateProtocol initialized (timeout={round_timeout}s, max_rounds={max_debate_rounds})")

    # --------------------------------------------------------
    # 1. Challenge Initiation
    # --------------------------------------------------------

    async def initiate_challenge(
        self,
        challenger_id: str,
        target_vote: str,
        reason: str,
        target_agent_id: Optional[str] = None,
    ) -> Challenge:
        """
        Challenge 발언권 획득

        반대 의견 에이전트가 기존 투표에 이의를 제기

        Args:
            challenger_id: Challenge를 제기하는 에이전트 ID
            target_vote: 대상 투표 유형 ("approve", "reject" 등)
            reason: Challenge 사유
            target_agent_id: 특정 에이전트 대상 (None이면 전체 투표 대상)

        Returns:
            생성된 Challenge 객체
        """
        import uuid

        # Challenger 정보 확인
        challenger = self.consensus_engine._agents.get(challenger_id)
        if not challenger:
            raise ValueError(f"Unknown challenger: {challenger_id}")

        challenger_name = challenger.agent_name

        # Challenge 생성
        challenge = Challenge(
            challenge_id=f"challenge_{uuid.uuid4().hex[:8]}",
            challenger_id=challenger_id,
            challenger_name=challenger_name,
            target_vote=target_vote,
            target_agent_id=target_agent_id,
            reason=reason,
            status=ChallengeStatus.ACTIVE,
        )

        self._active_challenges[challenge.challenge_id] = challenge

        # 새 Debate Round 시작
        debate_round = self._start_debate_round(challenge)

        logger.info(
            f"Challenge initiated: {challenge.challenge_id} by {challenger_name} "
            f"targeting '{target_vote}' votes. Reason: {reason[:50]}..."
        )

        # 이벤트 핸들러 호출
        if self._on_challenge_initiated:
            await self._on_challenge_initiated(challenge, debate_round)

        return challenge

    def _start_debate_round(self, challenge: Challenge) -> DebateRound:
        """새 Debate Round 시작"""
        import uuid

        round_number = len(self._debate_rounds) + 1
        participants = list(self.consensus_engine._agents.keys())

        debate_round = DebateRound(
            round_id=f"debate_{uuid.uuid4().hex[:8]}",
            round_number=round_number,
            challenge=challenge,
            timeout_seconds=self.round_timeout,
            participants=participants,
        )

        self._debate_rounds.append(debate_round)
        return debate_round

    # --------------------------------------------------------
    # 2. Evidence Presentation
    # --------------------------------------------------------

    async def present_evidence(
        self,
        evidence_type: EvidenceType,
        data: Dict[str, Any],
        presenter_id: Optional[str] = None,
        summary: str = "",
        weight: float = 0.5,
    ) -> DebateEvidence:
        """
        증거 제시

        TDA 분석 결과, 통계 분석, 시맨틱 분석 등을 증거로 제시

        Args:
            evidence_type: 증거 유형 (TDA_RESULT, STATISTICAL, SEMANTIC 등)
            data: 증거 데이터
            presenter_id: 증거를 제시하는 에이전트 ID
            summary: 증거 요약
            weight: 증거 가중치 (0.0 - 1.0)

        Returns:
            생성된 DebateEvidence 객체
        """
        import uuid

        evidence = DebateEvidence(
            evidence_id=f"evidence_{uuid.uuid4().hex[:8]}",
            evidence_type=evidence_type,
            presenter_id=presenter_id or "system",
            data=data,
            summary=summary or self._generate_evidence_summary(evidence_type, data),
            weight=weight,
        )

        # 현재 Debate Round에 증거 추가
        if self._debate_rounds:
            current_round = self._debate_rounds[-1]
            current_round.add_evidence(evidence)

            # 관련 Challenge에도 증거 추가
            if current_round.challenge:
                # 증거가 challenger를 지지하는지 판단
                is_supporting = self._is_evidence_supporting_challenge(
                    evidence, current_round.challenge
                )
                current_round.challenge.add_evidence(evidence, is_supporting)

        logger.info(
            f"Evidence presented: {evidence.evidence_id} ({evidence_type.value}) "
            f"by {presenter_id or 'system'}. Weight: {weight}"
        )

        # 이벤트 핸들러 호출
        if self._on_evidence_presented:
            await self._on_evidence_presented(evidence)

        return evidence

    def _generate_evidence_summary(
        self,
        evidence_type: EvidenceType,
        data: Dict[str, Any],
    ) -> str:
        """증거 요약 자동 생성"""
        if evidence_type == EvidenceType.TDA_RESULT:
            persistence = data.get("persistence_score", "N/A")
            holes = data.get("topological_holes", [])
            return f"TDA Analysis: persistence={persistence}, holes={len(holes)}"

        elif evidence_type == EvidenceType.STATISTICAL:
            confidence = data.get("confidence_interval", "N/A")
            p_value = data.get("p_value", "N/A")
            return f"Statistical: CI={confidence}, p={p_value}"

        elif evidence_type == EvidenceType.SEMANTIC:
            similarity = data.get("similarity_score", "N/A")
            return f"Semantic Match: similarity={similarity}"

        elif evidence_type == EvidenceType.HISTORICAL:
            past_decisions = data.get("past_decisions", [])
            return f"Historical: {len(past_decisions)} similar past decisions"

        else:
            return f"Evidence: {evidence_type.value}"

    def _is_evidence_supporting_challenge(
        self,
        evidence: DebateEvidence,
        challenge: Challenge,
    ) -> bool:
        """증거가 Challenge를 지지하는지 판단"""
        # 기본 로직: 증거 가중치가 0.5 이상이면 지지로 간주
        # 실제로는 증거 내용과 Challenge 목표를 비교해야 함
        data = evidence.data

        # TDA 결과: 높은 persistence는 보통 "approve" 지지
        if evidence.evidence_type == EvidenceType.TDA_RESULT:
            persistence = data.get("persistence_score", 0.5)
            if challenge.target_vote == "approve":
                return persistence < 0.5  # 낮은 persistence는 approve 반대
            else:
                return persistence >= 0.5

        # 통계 결과: 유의미한 p-value
        if evidence.evidence_type == EvidenceType.STATISTICAL:
            p_value = data.get("p_value", 0.5)
            return p_value < 0.05  # 유의미한 결과는 Challenge 지지

        # 기본: 가중치 기반
        return evidence.weight >= 0.5

    # --------------------------------------------------------
    # 3. Revote
    # --------------------------------------------------------

    async def call_revote(
        self,
        participants: Optional[List[str]] = None,
        timeout: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        재투표 호출

        새로운 증거를 기반으로 참여자들에게 재투표 요청

        Args:
            participants: 재투표 참여자 목록 (None이면 전체)
            timeout: 재투표 타임아웃 (None이면 기본값)

        Returns:
            재투표 결과
        """
        if not self._debate_rounds:
            raise RuntimeError("No active debate round for revote")

        current_round = self._debate_rounds[-1]
        effective_timeout = timeout or self.round_timeout

        # 참여자 결정
        if participants is None:
            participants = current_round.participants

        logger.info(f"Revote called with {len(participants)} participants, timeout={effective_timeout}s")

        # 증거 요약 생성
        evidence_summary = self._compile_evidence_summary(current_round)

        # 비동기 재투표 수집 (타임아웃 적용)
        revote_results = await self._collect_revotes(
            participants=participants,
            evidence_summary=evidence_summary,
            timeout=effective_timeout,
        )

        # 결과 기록
        current_round.revote_opinions = revote_results["opinions"]
        current_round.calculate_revote_result()
        current_round.complete()

        # Challenge 해결 여부 확인
        resolution_result = self._check_challenge_resolution(current_round)

        result = {
            "round_id": current_round.round_id,
            "participants": participants,
            "opinions": {k: v.to_dict() for k, v in revote_results["opinions"].items()},
            "combined_confidence": current_round.revote_combined_confidence,
            "timed_out": revote_results.get("timed_out", False),
            "resolution": resolution_result,
        }

        logger.info(
            f"Revote completed: combined_confidence={current_round.revote_combined_confidence:.2f}, "
            f"resolution={resolution_result}"
        )

        # 이벤트 핸들러 호출
        if self._on_revote_completed:
            await self._on_revote_completed(result)

        return result

    def _compile_evidence_summary(self, debate_round: DebateRound) -> str:
        """Debate Round의 증거 요약 컴파일"""
        summaries = []
        for evidence in debate_round.evidence_presented:
            summaries.append(f"- [{evidence.evidence_type.value}] {evidence.summary}")

        if not summaries:
            return "(No new evidence presented)"

        return "New Evidence:\n" + "\n".join(summaries)

    async def _collect_revotes(
        self,
        participants: List[str],
        evidence_summary: str,
        timeout: float,
    ) -> Dict[str, Any]:
        """재투표 수집 (타임아웃 적용)"""
        opinions: Dict[str, AgentOpinion] = {}
        timed_out = False

        # 병렬 투표 수집
        tasks = []
        valid_agent_ids = []
        for agent_id in participants:
            agent = self.consensus_engine._agents.get(agent_id)
            if agent:
                tasks.append(self._get_revote_from_agent(agent, evidence_summary))
                valid_agent_ids.append(agent_id)

        try:
            # v18.0: timeout=0이면 무제한 대기
            if timeout and timeout > 0:
                results = await asyncio.wait_for(
                    asyncio.gather(*tasks, return_exceptions=True),
                    timeout=timeout,
                )
            else:
                results = await asyncio.gather(*tasks, return_exceptions=True)

            for i, result in enumerate(results):
                agent_id = valid_agent_ids[i]
                if isinstance(result, Exception):
                    logger.warning(f"Agent {agent_id} revote failed: {result}")
                    continue
                if result:
                    opinions[agent_id] = result

        except asyncio.TimeoutError:
            logger.warning(f"Revote timed out after {timeout}s")
            timed_out = True

        return {"opinions": opinions, "timed_out": timed_out}

    async def _get_revote_from_agent(
        self,
        agent: "AutonomousAgent",
        evidence_summary: str,
    ) -> Optional[AgentOpinion]:
        """개별 에이전트로부터 재투표 수집"""
        try:
            # 현재 토론 상태
            current_state = self.consensus_engine._current_discussion
            topic_data = {"description": current_state.topic_name if current_state else "Revote"}

            # 재투표 프롬프트
            prompt = f"""## Revote Request

Based on the new evidence presented during debate, please reconsider your vote.

{evidence_summary}

## Previous Vote
Your previous vote and confidence are on record.

## Task
Reconsider and provide your updated assessment.

## Response Format (REQUIRED)
[CONFIDENCE: 0.0-1.0]
[UNCERTAINTY: 0.0-1.0]
[DOUBT: 0.0-1.0]
[EVIDENCE: 0.0-1.0]

[OBSERVATIONS]
- Point 1
...

[CONCERNS]
- Risk 1
...

[REASONING]
Explain how the new evidence affected your decision.
"""

            response = await agent.call_llm(
                user_prompt=prompt,
                temperature=0.3,
                max_tokens=1000,
            )

            # 파싱
            return self.consensus_engine._parse_opinion(agent, response, 0)

        except Exception as e:
            logger.error(f"Failed to get revote from {agent.agent_id}: {e}")
            return None

    def _check_challenge_resolution(self, debate_round: DebateRound) -> str:
        """Challenge 해결 여부 확인"""
        if not debate_round.challenge:
            return "no_challenge"

        challenge = debate_round.challenge
        combined_conf = debate_round.revote_combined_confidence

        # 재투표 결과에 따른 해결
        if combined_conf >= 0.7:
            # 높은 신뢰도로 합의 도달
            challenge.resolve(
                resolution="consensus_reached",
                revote_result={
                    "combined_confidence": combined_conf,
                    "decision": "approve" if combined_conf >= 0.7 else "reject",
                },
            )
            self._failed_consensus_count = 0
            return "resolved_approve"

        elif combined_conf <= 0.3:
            # 낮은 신뢰도로 거부 합의
            challenge.resolve(
                resolution="consensus_reject",
                revote_result={
                    "combined_confidence": combined_conf,
                    "decision": "reject",
                },
            )
            self._failed_consensus_count = 0
            return "resolved_reject"

        else:
            # 합의 실패
            self._failed_consensus_count += 1
            if self._failed_consensus_count >= self.escalation_threshold:
                return "needs_escalation"
            return "unresolved"

    # --------------------------------------------------------
    # 4. Escalation to Human
    # --------------------------------------------------------

    async def escalate_to_human(
        self,
        topic: str,
        reason: str,
    ) -> HumanEscalation:
        """
        Human Escalation

        합의 실패 시 사람에게 검토 요청

        Args:
            topic: 토론 주제
            reason: Escalation 사유

        Returns:
            HumanEscalation 객체
        """
        import uuid

        current_state = self.consensus_engine._current_discussion

        escalation = HumanEscalation(
            escalation_id=f"escalation_{uuid.uuid4().hex[:8]}",
            topic_id=current_state.topic_id if current_state else "unknown",
            topic_name=topic,
            reason=reason,
            challenges=[c for c in self._active_challenges.values()],
            debate_history=self._debate_rounds.copy(),
            agent_opinions=current_state.all_opinions.copy() if current_state else {},
        )

        self._escalations.append(escalation)

        # 모든 활성 Challenge를 escalate
        for challenge in self._active_challenges.values():
            if challenge.status == ChallengeStatus.ACTIVE:
                challenge.escalate(reason)

        logger.warning(
            f"Human escalation requested: {escalation.escalation_id}. "
            f"Topic: {topic}. Reason: {reason}"
        )

        # 이벤트 핸들러 호출
        if self._on_escalated:
            await self._on_escalated(escalation)

        return escalation

    # --------------------------------------------------------
    # Helper Methods
    # --------------------------------------------------------

    def get_active_challenges(self) -> List[Challenge]:
        """활성 Challenge 목록 반환"""
        return [
            c for c in self._active_challenges.values()
            if c.status == ChallengeStatus.ACTIVE
        ]

    def get_debate_history(self) -> List[Dict[str, Any]]:
        """Debate 히스토리 반환"""
        return [r.to_dict() for r in self._debate_rounds]

    def get_escalations(self) -> List[Dict[str, Any]]:
        """Escalation 목록 반환"""
        return [e.to_dict() for e in self._escalations]

    def reset(self) -> None:
        """상태 초기화"""
        self._active_challenges.clear()
        self._debate_rounds.clear()
        self._failed_consensus_count = 0
        logger.info("DebateProtocol state reset")

    # --------------------------------------------------------
    # Event Handler Registration
    # --------------------------------------------------------

    def on_challenge_initiated(self, handler: callable) -> None:
        """Challenge 발생 시 호출될 핸들러 등록"""
        self._on_challenge_initiated = handler

    def on_evidence_presented(self, handler: callable) -> None:
        """증거 제시 시 호출될 핸들러 등록"""
        self._on_evidence_presented = handler

    def on_revote_completed(self, handler: callable) -> None:
        """재투표 완료 시 호출될 핸들러 등록"""
        self._on_revote_completed = handler

    def on_escalated(self, handler: callable) -> None:
        """Escalation 시 호출될 핸들러 등록"""
        self._on_escalated = handler
