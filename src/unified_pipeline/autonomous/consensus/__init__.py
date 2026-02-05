"""
Consensus Module for Unified Pipeline (v16.0)

멀티에이전트 토론/합의 시스템
- 에이전트간 실시간 토론
- Confidence-based voting (0.0-1.0)
- 동의/반대 추적
- 합의 도달 판정
- v2.0: EvidenceBundle 통합, ConversationLogger 추가
- v16.0: AgentCommunicationBus 통합 - 직접 에이전트간 메시징
"""

from .engine import PipelineConsensusEngine
from .models import (
    ConsensusState,
    AgentOpinion,
    DiscussionRound,
    DiscussionMessage,
    ConsensusResult,
    ConsensusType,
)
from .channel import AgentCommunicationChannel
from .logger import (
    ConversationLogger,
    get_conversation_logger,
    enable_conversation_logging,
    disable_conversation_logging,
)

# v16.0: AgentCommunicationBus (re-export for convenience)
try:
    from ..agent_bus import (
        AgentCommunicationBus,
        AgentMessage,
        MessageType,
        MessagePriority,
        get_agent_bus,
    )
    AGENT_BUS_AVAILABLE = True
except ImportError:
    AGENT_BUS_AVAILABLE = False
    AgentCommunicationBus = None
    AgentMessage = None
    MessageType = None
    MessagePriority = None
    get_agent_bus = None

__all__ = [
    "PipelineConsensusEngine",
    "ConsensusState",
    "AgentOpinion",
    "DiscussionRound",
    "DiscussionMessage",
    "ConsensusResult",
    "ConsensusType",
    "AgentCommunicationChannel",
    # v2.0: ConversationLogger
    "ConversationLogger",
    "get_conversation_logger",
    "enable_conversation_logging",
    "disable_conversation_logging",
    # v16.0: AgentCommunicationBus
    "AgentCommunicationBus",
    "AgentMessage",
    "MessageType",
    "MessagePriority",
    "get_agent_bus",
    "AGENT_BUS_AVAILABLE",
]
