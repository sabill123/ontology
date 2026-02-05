"""
Autonomous Agent System (v17.0)

자율 에이전트 시스템
- AutonomousAgent: 자율 에이전트 베이스 클래스 (v1.0)
- AutonomousAgentV2: 팔란티어급 에이전트 (v2.0)
- AgentOrchestrator: 에이전트 오케스트레이터
- TodoContinuationEnforcer: Todo 지속 실행기
- v16.0: AgentCommunicationBus - 에이전트간 직접 통신 버스
- v17.0: Object-Centric Agent (Object→Action 패턴)
"""

from .base import AutonomousAgent, AgentState

# v17.0: Palantir-style Agent
from .base_v2 import AutonomousAgentV2, ReasoningMode, ObjectAction, ReasoningContext
from .orchestrator import AgentOrchestrator
from .continuation import TodoContinuationEnforcer

# v16.0: Agent Communication Bus
try:
    from .agent_bus import (
        AgentCommunicationBus,
        AgentMessage,
        MessageType,
        MessagePriority,
        Subscription,
        get_agent_bus,
        create_info_message,
        create_discovery_message,
        create_validation_request,
    )
    AGENT_BUS_AVAILABLE = True
except ImportError:
    AGENT_BUS_AVAILABLE = False
    AgentCommunicationBus = None
    AgentMessage = None
    MessageType = None
    MessagePriority = None
    Subscription = None
    get_agent_bus = None
    create_info_message = None
    create_discovery_message = None
    create_validation_request = None

__all__ = [
    # v1.0 Agent
    "AutonomousAgent",
    "AgentState",
    "AgentOrchestrator",
    "TodoContinuationEnforcer",
    # v17.0: Palantir-style Agent
    "AutonomousAgentV2",
    "ReasoningMode",
    "ObjectAction",
    "ReasoningContext",
    # v16.0: Agent Communication Bus
    "AgentCommunicationBus",
    "AgentMessage",
    "MessageType",
    "MessagePriority",
    "Subscription",
    "get_agent_bus",
    "create_info_message",
    "create_discovery_message",
    "create_validation_request",
    "AGENT_BUS_AVAILABLE",
]
