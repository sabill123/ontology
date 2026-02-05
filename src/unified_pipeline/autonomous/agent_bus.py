"""
Agent Communication Bus (v16.0)

에이전트 간 직접 통신을 위한 메시지 버스:
- Direct Messaging: 에이전트 간 1:1 메시지 전송
- Broadcast: 전체 에이전트에 메시지 브로드캐스트
- Request/Response: 동기적 요청/응답 패턴
- Pub/Sub: 토픽 기반 구독/발행

멀티에이전트 협업 패턴:
- 정보 공유: FK 탐지 결과 즉시 공유
- 협력 요청: 다른 에이전트에 분석 요청
- 검증 요청: 결과 상호 검증
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Any, Optional, Callable, Set, Awaitable
from enum import Enum
from collections import defaultdict
import uuid

logger = logging.getLogger(__name__)


class MessageType(Enum):
    """메시지 유형"""
    # 정보 공유
    INFO = "info"                       # 일반 정보
    DISCOVERY = "discovery"             # 발견 결과 공유
    ALERT = "alert"                     # 중요 알림

    # 요청/응답
    REQUEST = "request"                 # 분석/검증 요청
    RESPONSE = "response"               # 요청에 대한 응답

    # 협업
    COLLABORATION = "collaboration"     # 협업 제안
    VALIDATION = "validation"           # 검증 요청
    FEEDBACK = "feedback"               # 피드백

    # 시스템
    HEARTBEAT = "heartbeat"             # 상태 확인
    STATUS = "status"                   # 상태 보고


class MessagePriority(Enum):
    """메시지 우선순위"""
    LOW = 0
    NORMAL = 1
    HIGH = 2
    URGENT = 3


@dataclass
class AgentMessage:
    """에이전트 메시지"""
    message_id: str
    message_type: MessageType
    sender_id: str
    content: Dict[str, Any]

    # 수신자 (None이면 브로드캐스트)
    recipient_id: Optional[str] = None

    # 토픽 (Pub/Sub용)
    topic: Optional[str] = None

    # 메타데이터
    priority: MessagePriority = MessagePriority.NORMAL
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    ttl_seconds: int = 300  # 메시지 만료 시간 (5분)

    # 요청/응답 연결
    correlation_id: Optional[str] = None  # 원본 요청 ID
    requires_response: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "message_id": self.message_id,
            "message_type": self.message_type.value,
            "sender_id": self.sender_id,
            "recipient_id": self.recipient_id,
            "topic": self.topic,
            "content": self.content,
            "priority": self.priority.value,
            "timestamp": self.timestamp,
            "requires_response": self.requires_response,
            "correlation_id": self.correlation_id,
        }


@dataclass
class Subscription:
    """토픽 구독"""
    subscription_id: str
    subscriber_id: str
    topic: str
    callback: Callable[[AgentMessage], Awaitable[None]]
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())


class AgentCommunicationBus:
    """
    에이전트 통신 버스 (v16.0)

    기능:
    1. Direct Messaging: send_to(agent_id, message)
    2. Broadcast: broadcast(message)
    3. Request/Response: request(agent_id, message) -> response
    4. Pub/Sub: subscribe(topic), publish(topic, message)

    사용법:
        bus = AgentCommunicationBus()

        # 직접 메시지
        await bus.send_to("agent_2", message)

        # 브로드캐스트
        await bus.broadcast(message)

        # 요청/응답
        response = await bus.request("agent_2", request_message, timeout=10)

        # Pub/Sub
        await bus.subscribe("fk_detected", callback)
        await bus.publish("fk_detected", message)
    """

    _instance: Optional["AgentCommunicationBus"] = None

    def __new__(cls):
        """싱글톤 패턴"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self._initialized = True

        # 등록된 에이전트
        self.agents: Dict[str, Any] = {}  # agent_id -> agent instance

        # 메시지 큐 (에이전트별)
        self.message_queues: Dict[str, asyncio.Queue] = defaultdict(asyncio.Queue)

        # 토픽 구독
        self.subscriptions: Dict[str, List[Subscription]] = defaultdict(list)

        # 대기 중인 응답
        self.pending_responses: Dict[str, asyncio.Future] = {}

        # 메시지 히스토리 (디버깅용)
        self.message_history: List[AgentMessage] = []
        self._history_limit = 1000

        # 통계
        self.stats = {
            "messages_sent": 0,
            "messages_broadcast": 0,
            "requests_made": 0,
            "responses_received": 0,
            "topics_active": 0,
        }

        logger.info("[v16.0] AgentCommunicationBus initialized")

    def register_agent(self, agent_id: str, agent: Any) -> None:
        """에이전트 등록"""
        self.agents[agent_id] = agent
        logger.debug(f"[AgentBus] Agent registered: {agent_id}")

    def unregister_agent(self, agent_id: str) -> None:
        """에이전트 등록 해제"""
        if agent_id in self.agents:
            del self.agents[agent_id]
            logger.debug(f"[AgentBus] Agent unregistered: {agent_id}")

    async def send_to(
        self,
        recipient_id: str,
        message: AgentMessage,
    ) -> bool:
        """
        특정 에이전트에 메시지 전송

        Args:
            recipient_id: 수신자 에이전트 ID
            message: 전송할 메시지

        Returns:
            전송 성공 여부
        """
        if recipient_id not in self.agents:
            logger.warning(f"[AgentBus] Unknown recipient: {recipient_id}")
            return False

        message.recipient_id = recipient_id
        await self.message_queues[recipient_id].put(message)

        self._record_message(message)
        self.stats["messages_sent"] += 1

        logger.debug(f"[AgentBus] Message sent: {message.sender_id} -> {recipient_id}")
        return True

    async def broadcast(
        self,
        message: AgentMessage,
        exclude: Optional[Set[str]] = None,
    ) -> int:
        """
        모든 에이전트에 메시지 브로드캐스트

        Args:
            message: 전송할 메시지
            exclude: 제외할 에이전트 ID 집합

        Returns:
            전송된 에이전트 수
        """
        exclude = exclude or set()
        sent_count = 0

        for agent_id in self.agents:
            if agent_id != message.sender_id and agent_id not in exclude:
                msg_copy = AgentMessage(
                    message_id=f"{message.message_id}_{agent_id}",
                    message_type=message.message_type,
                    sender_id=message.sender_id,
                    recipient_id=agent_id,
                    content=message.content.copy(),
                    topic=message.topic,
                    priority=message.priority,
                )
                await self.message_queues[agent_id].put(msg_copy)
                sent_count += 1

        self._record_message(message)
        self.stats["messages_broadcast"] += 1

        logger.debug(f"[AgentBus] Broadcast from {message.sender_id} to {sent_count} agents")
        return sent_count

    async def request(
        self,
        recipient_id: str,
        message: AgentMessage,
        timeout: float = 0,  # v18.0: 타임아웃 비활성화 (0 = 무제한)
    ) -> Optional[AgentMessage]:
        """
        에이전트에 요청하고 응답 대기

        Args:
            recipient_id: 수신자 에이전트 ID
            message: 요청 메시지
            timeout: 응답 대기 시간 (초) - v18.0: 5분 기본값

        Returns:
            응답 메시지 또는 None (타임아웃)
        """
        message.requires_response = True
        message.correlation_id = message.message_id

        # 응답 대기 Future 생성
        response_future: asyncio.Future = asyncio.get_event_loop().create_future()
        self.pending_responses[message.message_id] = response_future

        # 요청 전송
        await self.send_to(recipient_id, message)
        self.stats["requests_made"] += 1

        try:
            # 응답 대기 (v18.0: timeout=0이면 무제한 대기)
            if timeout and timeout > 0:
                response = await asyncio.wait_for(response_future, timeout=timeout)
            else:
                response = await response_future  # 무제한 대기
            self.stats["responses_received"] += 1
            return response
        except asyncio.TimeoutError:
            logger.warning(f"[AgentBus] Request timeout: {message.message_id}")
            return None
        finally:
            # 정리
            if message.message_id in self.pending_responses:
                del self.pending_responses[message.message_id]

    async def respond(
        self,
        original_message: AgentMessage,
        response_content: Dict[str, Any],
        sender_id: str,
    ) -> bool:
        """
        요청에 대한 응답 전송

        Args:
            original_message: 원본 요청 메시지
            response_content: 응답 내용
            sender_id: 응답자 에이전트 ID

        Returns:
            응답 전송 성공 여부
        """
        if not original_message.requires_response:
            return False

        response = AgentMessage(
            message_id=f"resp_{uuid.uuid4().hex[:8]}",
            message_type=MessageType.RESPONSE,
            sender_id=sender_id,
            recipient_id=original_message.sender_id,
            content=response_content,
            correlation_id=original_message.correlation_id,
        )

        # 대기 중인 Future에 결과 설정
        correlation_id = original_message.correlation_id or original_message.message_id
        if correlation_id in self.pending_responses:
            self.pending_responses[correlation_id].set_result(response)
            return True

        # Future가 없으면 일반 메시지로 전송
        return await self.send_to(original_message.sender_id, response)

    async def subscribe(
        self,
        topic: str,
        subscriber_id: str,
        callback: Callable[[AgentMessage], Awaitable[None]],
    ) -> str:
        """
        토픽 구독

        Args:
            topic: 구독할 토픽
            subscriber_id: 구독자 에이전트 ID
            callback: 메시지 수신 콜백 (async)

        Returns:
            구독 ID
        """
        subscription = Subscription(
            subscription_id=f"sub_{uuid.uuid4().hex[:8]}",
            subscriber_id=subscriber_id,
            topic=topic,
            callback=callback,
        )

        self.subscriptions[topic].append(subscription)
        self.stats["topics_active"] = len(self.subscriptions)

        logger.debug(f"[AgentBus] {subscriber_id} subscribed to {topic}")
        return subscription.subscription_id

    async def unsubscribe(self, subscription_id: str) -> bool:
        """토픽 구독 해제"""
        for topic, subs in self.subscriptions.items():
            for sub in subs:
                if sub.subscription_id == subscription_id:
                    subs.remove(sub)
                    logger.debug(f"[AgentBus] Unsubscribed: {subscription_id}")
                    return True
        return False

    async def publish(
        self,
        topic: str,
        message: AgentMessage,
    ) -> int:
        """
        토픽에 메시지 발행

        Args:
            topic: 발행할 토픽
            message: 메시지

        Returns:
            전달된 구독자 수
        """
        message.topic = topic
        subscribers = self.subscriptions.get(topic, [])
        delivered_count = 0

        for sub in subscribers:
            if sub.subscriber_id != message.sender_id:
                try:
                    await sub.callback(message)
                    delivered_count += 1
                except Exception as e:
                    logger.warning(f"[AgentBus] Callback error for {sub.subscriber_id}: {e}")

        self._record_message(message)
        logger.debug(f"[AgentBus] Published to {topic}: {delivered_count} subscribers")
        return delivered_count

    async def receive(
        self,
        agent_id: str,
        timeout: float = 1.0,
    ) -> Optional[AgentMessage]:
        """
        에이전트의 메시지 수신

        Args:
            agent_id: 에이전트 ID
            timeout: 대기 시간 (초)

        Returns:
            수신된 메시지 또는 None
        """
        try:
            # v18.0: timeout이 0이면 즉시 반환 (non-blocking check)
            if timeout and timeout > 0:
                message = await asyncio.wait_for(
                    self.message_queues[agent_id].get(),
                    timeout=timeout
                )
            else:
                # 메시지가 있으면 반환, 없으면 None
                if self.message_queues[agent_id].qsize() > 0:
                    message = await self.message_queues[agent_id].get()
                else:
                    return None
            return message
        except asyncio.TimeoutError:
            return None

    def get_pending_messages(self, agent_id: str) -> int:
        """에이전트의 대기 중인 메시지 수"""
        return self.message_queues[agent_id].qsize()

    def _record_message(self, message: AgentMessage) -> None:
        """메시지 히스토리 기록"""
        self.message_history.append(message)

        # 히스토리 제한
        if len(self.message_history) > self._history_limit:
            self.message_history = self.message_history[-self._history_limit:]

    def get_statistics(self) -> Dict[str, Any]:
        """통계 조회"""
        return {
            **self.stats,
            "registered_agents": len(self.agents),
            "pending_responses": len(self.pending_responses),
            "message_history_size": len(self.message_history),
        }


# 싱글톤 인스턴스 접근
def get_agent_bus() -> AgentCommunicationBus:
    """AgentCommunicationBus 싱글톤 인스턴스 반환"""
    return AgentCommunicationBus()


# === 편의 함수들 ===

def create_info_message(
    sender_id: str,
    content: Dict[str, Any],
    topic: Optional[str] = None,
) -> AgentMessage:
    """정보 메시지 생성"""
    return AgentMessage(
        message_id=f"info_{uuid.uuid4().hex[:8]}",
        message_type=MessageType.INFO,
        sender_id=sender_id,
        content=content,
        topic=topic,
    )


def create_discovery_message(
    sender_id: str,
    discovery_type: str,
    data: Dict[str, Any],
) -> AgentMessage:
    """발견 결과 메시지 생성"""
    return AgentMessage(
        message_id=f"disc_{uuid.uuid4().hex[:8]}",
        message_type=MessageType.DISCOVERY,
        sender_id=sender_id,
        content={
            "discovery_type": discovery_type,
            "data": data,
        },
        topic=f"discovery.{discovery_type}",
    )


def create_validation_request(
    sender_id: str,
    validation_type: str,
    data: Dict[str, Any],
) -> AgentMessage:
    """검증 요청 메시지 생성"""
    return AgentMessage(
        message_id=f"val_{uuid.uuid4().hex[:8]}",
        message_type=MessageType.VALIDATION,
        sender_id=sender_id,
        content={
            "validation_type": validation_type,
            "data": data,
        },
        requires_response=True,
    )


# 모듈 초기화
__all__ = [
    "AgentCommunicationBus",
    "AgentMessage",
    "MessageType",
    "MessagePriority",
    "Subscription",
    "get_agent_bus",
    "create_info_message",
    "create_discovery_message",
    "create_validation_request",
]
