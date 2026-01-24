"""
Agent Communication Channel

에이전트간 실시간 통신 채널
- @멘션 기반 호출
- 메시지 브로드캐스트
- 비동기 메시지 큐
"""

import asyncio
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Any, Optional, Callable, Awaitable

from .models import AgentOpinion, DiscussionMessage

logger = logging.getLogger(__name__)


@dataclass
class PendingMessage:
    """대기 중인 메시지"""
    from_agent: str
    to_agent: str  # "*" for broadcast
    content: str
    priority: int = 0
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


class AgentCommunicationChannel:
    """
    에이전트간 통신 채널

    핵심 기능:
    1. @멘션 파싱 - 응답에서 @agent_name 추출
    2. 메시지 라우팅 - 특정 에이전트 또는 브로드캐스트
    3. 동의/반대 추적 - 에이전트간 의견 관계 파악
    4. 발언 순서 관리 - 우선순위 기반 다음 발언자 선택
    """

    def __init__(self):
        # 등록된 에이전트
        self._agents: Dict[str, Dict[str, Any]] = {}  # {agent_id: {name, type, callback}}

        # 메시지 큐
        self._message_queue: asyncio.Queue = asyncio.Queue()
        self._message_history: List[DiscussionMessage] = []

        # 동의/반대 관계
        self._agreement_graph: Dict[str, Dict[str, float]] = {}  # {from_agent: {to_agent: score}}
        self._disagreement_graph: Dict[str, Dict[str, float]] = {}

        # 발언 통계
        self._speaking_count: Dict[str, int] = {}
        self._last_speaker: Optional[str] = None

        # 실행 상태
        self._running = False

    def register_agent(
        self,
        agent_id: str,
        agent_name: str,
        agent_type: str,
        callback: Optional[Callable[[DiscussionMessage], Awaitable[str]]] = None,
    ) -> None:
        """
        에이전트 등록

        Args:
            agent_id: 에이전트 ID
            agent_name: 에이전트 이름 (표시용, @멘션용)
            agent_type: 에이전트 타입
            callback: 메시지 수신 콜백 (비동기)
        """
        self._agents[agent_id] = {
            "name": agent_name,
            "type": agent_type,
            "callback": callback,
        }
        self._speaking_count[agent_id] = 0
        logger.debug(f"Agent registered: {agent_id} ({agent_name})")

    def unregister_agent(self, agent_id: str) -> None:
        """에이전트 등록 해제"""
        if agent_id in self._agents:
            del self._agents[agent_id]
            if agent_id in self._speaking_count:
                del self._speaking_count[agent_id]
            logger.debug(f"Agent unregistered: {agent_id}")

    # === 메시지 전송 ===

    async def broadcast(
        self,
        from_agent: str,
        content: str,
        opinion: Optional[AgentOpinion] = None,
    ) -> DiscussionMessage:
        """
        전체 브로드캐스트

        Args:
            from_agent: 발신 에이전트 ID
            content: 메시지 내용
            opinion: 첨부된 의견 (있을 경우)

        Returns:
            전송된 메시지
        """
        agent_info = self._agents.get(from_agent, {})

        message = DiscussionMessage(
            agent_id=from_agent,
            agent_name=agent_info.get("name", from_agent),
            content=content,
            opinion=opinion,
        )

        self._message_history.append(message)
        self._speaking_count[from_agent] = self._speaking_count.get(from_agent, 0) + 1
        self._last_speaker = from_agent

        # @멘션 및 동의/반대 파싱
        self._parse_references(from_agent, content)

        # 다른 에이전트들에게 알림 (비동기)
        for agent_id, agent_info in self._agents.items():
            if agent_id != from_agent and agent_info.get("callback"):
                try:
                    asyncio.create_task(agent_info["callback"](message))
                except Exception as e:
                    logger.warning(f"Callback error for {agent_id}: {e}")

        logger.debug(f"Broadcast from {from_agent}: {content[:100]}...")
        return message

    async def send_to(
        self,
        from_agent: str,
        to_agent: str,
        content: str,
    ) -> Optional[DiscussionMessage]:
        """
        특정 에이전트에게 메시지 전송

        Args:
            from_agent: 발신 에이전트 ID
            to_agent: 수신 에이전트 ID
            content: 메시지 내용

        Returns:
            전송된 메시지 (수신자가 없으면 None)
        """
        if to_agent not in self._agents:
            logger.warning(f"Agent not found: {to_agent}")
            return None

        from_info = self._agents.get(from_agent, {})
        to_info = self._agents.get(to_agent, {})

        message = DiscussionMessage(
            agent_id=from_agent,
            agent_name=from_info.get("name", from_agent),
            content=f"@{to_info.get('name', to_agent)}: {content}",
        )

        self._message_history.append(message)
        self._speaking_count[from_agent] = self._speaking_count.get(from_agent, 0) + 1
        self._last_speaker = from_agent

        # 수신자 콜백 호출
        if to_info.get("callback"):
            try:
                asyncio.create_task(to_info["callback"](message))
            except Exception as e:
                logger.warning(f"Callback error for {to_agent}: {e}")

        return message

    # === @멘션 및 참조 파싱 ===

    def _parse_references(self, from_agent: str, content: str) -> None:
        """
        응답에서 @멘션 및 동의/반대 파싱

        Args:
            from_agent: 발신 에이전트
            content: 메시지 내용
        """
        content_lower = content.lower()

        for agent_id, agent_info in self._agents.items():
            if agent_id == from_agent:
                continue

            agent_name = agent_info.get("name", "").lower()
            if not agent_name:
                continue

            # @멘션 확인
            if f"@{agent_name}" in content_lower or agent_name in content_lower:
                # 동의 패턴
                agreement_patterns = [
                    f"agree with {agent_name}",
                    f"agree with @{agent_name}",
                    f"support {agent_name}",
                    f"{agent_name} is right",
                    f"{agent_name}'s point",
                    f"concur with {agent_name}",
                    f"{agent_name} makes a good point",
                ]

                disagreement_patterns = [
                    f"disagree with {agent_name}",
                    f"disagree with @{agent_name}",
                    f"challenge {agent_name}",
                    f"{agent_name} is wrong",
                    f"contrary to {agent_name}",
                    f"differ from {agent_name}",
                    f"unlike {agent_name}",
                ]

                # 동의 확인
                for pattern in agreement_patterns:
                    if pattern in content_lower:
                        # 강도 파싱 (예: "agree 80%")
                        level_match = re.search(rf'{pattern}.*?(\d+)%', content_lower)
                        level = int(level_match.group(1)) / 100 if level_match else 0.8

                        if from_agent not in self._agreement_graph:
                            self._agreement_graph[from_agent] = {}
                        self._agreement_graph[from_agent][agent_id] = level
                        logger.debug(f"{from_agent} agrees with {agent_id} ({level:.0%})")
                        break

                # 반대 확인
                for pattern in disagreement_patterns:
                    if pattern in content_lower:
                        level_match = re.search(rf'{pattern}.*?(\d+)%', content_lower)
                        level = int(level_match.group(1)) / 100 if level_match else 0.7

                        if from_agent not in self._disagreement_graph:
                            self._disagreement_graph[from_agent] = {}
                        self._disagreement_graph[from_agent][agent_id] = level
                        logger.debug(f"{from_agent} disagrees with {agent_id} ({level:.0%})")
                        break

    def parse_mentions(self, content: str) -> List[str]:
        """
        응답에서 @멘션된 에이전트 ID 추출

        Args:
            content: 메시지 내용

        Returns:
            멘션된 에이전트 ID 목록
        """
        mentioned = []
        content_lower = content.lower()

        for agent_id, agent_info in self._agents.items():
            agent_name = agent_info.get("name", "").lower()
            if agent_name and (f"@{agent_name}" in content_lower or agent_name in content_lower):
                mentioned.append(agent_id)

        return mentioned

    # === 발언 순서 관리 ===

    def select_next_speaker(
        self,
        exclude: Optional[List[str]] = None,
        prefer_mentioned: bool = True,
        last_message: Optional[str] = None,
    ) -> Optional[str]:
        """
        다음 발언자 선택

        우선순위:
        1. 직전 발언에서 @멘션된 에이전트
        2. 아직 발언하지 않은 에이전트
        3. 발언 횟수가 적은 에이전트
        4. 직전 발언자 제외

        Args:
            exclude: 제외할 에이전트 ID 목록
            prefer_mentioned: @멘션된 에이전트 우선 선택
            last_message: 직전 메시지 (멘션 파싱용)

        Returns:
            다음 발언자 agent_id (없으면 None)
        """
        exclude = exclude or []
        if self._last_speaker:
            exclude.append(self._last_speaker)

        available = [
            agent_id for agent_id in self._agents.keys()
            if agent_id not in exclude
        ]

        if not available:
            return None

        # 1. @멘션된 에이전트 우선
        if prefer_mentioned and last_message:
            mentioned = self.parse_mentions(last_message)
            for agent_id in mentioned:
                if agent_id in available:
                    return agent_id

        # 2. 발언 횟수 기준 정렬 (적은 순)
        available.sort(key=lambda x: self._speaking_count.get(x, 0))

        return available[0] if available else None

    def get_agents_not_spoken(self) -> List[str]:
        """아직 발언하지 않은 에이전트 목록"""
        return [
            agent_id for agent_id in self._agents.keys()
            if self._speaking_count.get(agent_id, 0) == 0
        ]

    # === 관계 조회 ===

    def get_agreements_for(self, agent_id: str) -> Dict[str, float]:
        """특정 에이전트와 동의하는 에이전트들"""
        result = {}
        for from_agent, agreements in self._agreement_graph.items():
            if agent_id in agreements:
                result[from_agent] = agreements[agent_id]
        return result

    def get_disagreements_for(self, agent_id: str) -> Dict[str, float]:
        """특정 에이전트와 반대하는 에이전트들"""
        result = {}
        for from_agent, disagreements in self._disagreement_graph.items():
            if agent_id in disagreements:
                result[from_agent] = disagreements[agent_id]
        return result

    def get_cross_references(self) -> Dict[str, Any]:
        """전체 에이전트간 참조 관계"""
        return {
            "agreements": self._agreement_graph,
            "disagreements": self._disagreement_graph,
        }

    # === 메시지 히스토리 ===

    def get_message_history(self, limit: int = 100) -> List[DiscussionMessage]:
        """메시지 히스토리 조회"""
        return self._message_history[-limit:]

    def get_conversation_context(self, limit: int = 10) -> str:
        """
        LLM 컨텍스트용 대화 히스토리 포맷팅

        Args:
            limit: 최근 메시지 수

        Returns:
            포맷된 대화 히스토리
        """
        messages = self._message_history[-limit:]
        if not messages:
            return "(No previous discussion)"

        parts = []
        for msg in messages:
            confidence_str = ""
            if msg.opinion:
                confidence_str = f" [Confidence: {msg.opinion.confidence:.2f}]"
            parts.append(f"**{msg.agent_name}**{confidence_str}: {msg.content}")

        return "\n\n".join(parts)

    def clear_history(self) -> None:
        """히스토리 초기화"""
        self._message_history.clear()
        self._agreement_graph.clear()
        self._disagreement_graph.clear()
        self._speaking_count = {agent_id: 0 for agent_id in self._agents}
        self._last_speaker = None

    # === 상태 조회 ===

    def get_status(self) -> Dict[str, Any]:
        """채널 상태 조회"""
        return {
            "agents": {
                agent_id: {
                    "name": info.get("name"),
                    "type": info.get("type"),
                    "speaking_count": self._speaking_count.get(agent_id, 0),
                }
                for agent_id, info in self._agents.items()
            },
            "message_count": len(self._message_history),
            "last_speaker": self._last_speaker,
            "cross_references": self.get_cross_references(),
        }
