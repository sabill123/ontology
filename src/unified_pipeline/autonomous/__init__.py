"""
Autonomous Agent System

자율 에이전트 시스템
- AutonomousAgent: 자율 에이전트 베이스 클래스
- AgentOrchestrator: 에이전트 오케스트레이터
- TodoContinuationEnforcer: Todo 지속 실행기
- v16.0: AgentCommunicationBus - 에이전트간 직접 통신 버스
"""

from .base import AutonomousAgent, AgentState
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
    "AutonomousAgent",
    "AgentState",
    "AgentOrchestrator",
    "TodoContinuationEnforcer",
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
