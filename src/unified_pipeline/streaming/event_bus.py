"""
Event Bus - 이벤트 기반 컴포넌트 통신

Pub/Sub 패턴을 사용한 느슨한 결합(loose coupling) 아키텍처:
- 컴포넌트 간 직접 의존성 제거
- 비동기 이벤트 처리
- 와일드카드 구독 지원
- 이벤트 히스토리 관리

References:
- Enterprise Integration Patterns (Hohpe & Woolf, 2003)
- Reactive Manifesto
"""

import asyncio
import logging
import uuid
import re
from dataclasses import dataclass, field
from typing import Dict, List, Callable, Any, Optional, Union, Pattern
from datetime import datetime
from enum import Enum
from functools import lru_cache

logger = logging.getLogger(__name__)


class EventPriority(int, Enum):
    """이벤트 우선순위"""
    CRITICAL = 1
    HIGH = 2
    NORMAL = 3
    LOW = 4
    BACKGROUND = 5


@dataclass
class DomainEvent:
    """도메인 이벤트"""
    event_id: str
    event_name: str
    payload: Dict[str, Any]
    source: str
    timestamp: datetime = field(default_factory=datetime.now)
    correlation_id: Optional[str] = None
    causation_id: Optional[str] = None  # 이 이벤트를 발생시킨 이벤트
    priority: EventPriority = EventPriority.NORMAL
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.event_id:
            self.event_id = f"EVT-{uuid.uuid4().hex[:12]}"
        if not self.correlation_id:
            self.correlation_id = self.event_id

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "event_name": self.event_name,
            "payload": self.payload,
            "source": self.source,
            "timestamp": self.timestamp.isoformat(),
            "correlation_id": self.correlation_id,
            "causation_id": self.causation_id,
            "priority": self.priority.name,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DomainEvent":
        """딕셔너리에서 DomainEvent 생성"""
        priority = data.get("priority", "NORMAL")
        if isinstance(priority, str):
            priority = EventPriority[priority]

        return cls(
            event_id=data.get("event_id", ""),
            event_name=data["event_name"],
            payload=data.get("payload", {}),
            source=data.get("source", "unknown"),
            timestamp=datetime.fromisoformat(data["timestamp"]) if isinstance(data.get("timestamp"), str) else datetime.now(),
            correlation_id=data.get("correlation_id"),
            causation_id=data.get("causation_id"),
            priority=priority,
            metadata=data.get("metadata", {}),
        )

    def derive(
        self,
        event_name: str,
        payload: Dict[str, Any],
        source: str,
    ) -> "DomainEvent":
        """연관 이벤트 생성 (causation chain 유지)"""
        return DomainEvent(
            event_id="",  # 자동 생성
            event_name=event_name,
            payload=payload,
            source=source,
            correlation_id=self.correlation_id,
            causation_id=self.event_id,
        )


@dataclass
class Subscription:
    """이벤트 구독"""
    subscription_id: str
    pattern: str
    handler: Callable[[DomainEvent], Any]
    priority: int = 0  # 높을수록 먼저 실행
    is_async: bool = False
    filter_fn: Optional[Callable[[DomainEvent], bool]] = None

    def __post_init__(self):
        self.is_async = asyncio.iscoroutinefunction(self.handler)


class EventBus:
    """
    이벤트 버스 - Pub/Sub 패턴

    기능:
    - 이벤트 발행/구독
    - 와일드카드 패턴 매칭 (예: "ontology.*", "*.completed")
    - 비동기 핸들러 지원
    - 이벤트 필터링
    - 히스토리 관리

    사용 예:
        bus = EventBus()

        # 구독
        bus.subscribe("ontology.concept.created", handle_concept)
        bus.subscribe("ontology.*", handle_all_ontology)

        # 발행
        await bus.publish(DomainEvent(
            event_name="ontology.concept.created",
            payload={"concept_id": "C001", "name": "Customer"},
            source="OntologyArchitect"
        ))
    """

    def __init__(
        self,
        max_history: int = 1000,
        enable_logging: bool = True,
    ):
        """
        Args:
            max_history: 최대 이벤트 히스토리 크기
            enable_logging: 이벤트 로깅 활성화
        """
        self.max_history = max_history
        self.enable_logging = enable_logging

        # 구독 관리
        self._subscriptions: Dict[str, List[Subscription]] = {}
        self._wildcard_subscriptions: List[Subscription] = []
        self._subscription_counter = 0

        # 이벤트 히스토리
        self._event_history: List[DomainEvent] = []

        # 통계
        self._stats = {
            "published": 0,
            "delivered": 0,
            "errors": 0,
        }

        # 컴파일된 패턴 캐시
        self._pattern_cache: Dict[str, Pattern] = {}

    def subscribe(
        self,
        event_pattern: str,
        handler: Callable[[DomainEvent], Any],
        priority: int = 0,
        filter_fn: Optional[Callable[[DomainEvent], bool]] = None,
    ) -> str:
        """
        이벤트 구독

        Args:
            event_pattern: 이벤트 패턴 (예: "ontology.*", "discovery.completed")
            handler: 이벤트 핸들러
            priority: 핸들러 우선순위 (높을수록 먼저 실행)
            filter_fn: 추가 필터 함수

        Returns:
            구독 ID
        """
        self._subscription_counter += 1
        subscription_id = f"SUB-{self._subscription_counter:06d}"

        subscription = Subscription(
            subscription_id=subscription_id,
            pattern=event_pattern,
            handler=handler,
            priority=priority,
            filter_fn=filter_fn,
        )

        if "*" in event_pattern:
            self._wildcard_subscriptions.append(subscription)
            # 우선순위 정렬
            self._wildcard_subscriptions.sort(key=lambda s: -s.priority)
        else:
            if event_pattern not in self._subscriptions:
                self._subscriptions[event_pattern] = []
            self._subscriptions[event_pattern].append(subscription)
            # 우선순위 정렬
            self._subscriptions[event_pattern].sort(key=lambda s: -s.priority)

        logger.debug(f"Subscribed {subscription_id} to {event_pattern}")
        return subscription_id

    def unsubscribe(self, subscription_id: str) -> bool:
        """구독 해제"""
        # 정확한 패턴 구독에서 찾기
        for pattern, subs in self._subscriptions.items():
            for sub in subs:
                if sub.subscription_id == subscription_id:
                    subs.remove(sub)
                    logger.debug(f"Unsubscribed {subscription_id}")
                    return True

        # 와일드카드 구독에서 찾기
        for sub in self._wildcard_subscriptions:
            if sub.subscription_id == subscription_id:
                self._wildcard_subscriptions.remove(sub)
                logger.debug(f"Unsubscribed {subscription_id}")
                return True

        return False

    async def publish(
        self,
        event: Union[DomainEvent, Dict[str, Any]],
        wait: bool = True,
    ) -> int:
        """
        이벤트 발행

        Args:
            event: 발행할 이벤트
            wait: 모든 핸들러 완료 대기 여부

        Returns:
            전달된 핸들러 수
        """
        if isinstance(event, dict):
            event = DomainEvent.from_dict(event)

        # 히스토리 저장
        self._event_history.append(event)
        if len(self._event_history) > self.max_history:
            self._event_history.pop(0)

        self._stats["published"] += 1

        if self.enable_logging:
            logger.info(f"Event published: {event.event_name} from {event.source}")

        # 매칭되는 핸들러 수집
        handlers = self._get_matching_handlers(event)

        if not handlers:
            return 0

        # 핸들러 실행
        delivered = 0
        tasks = []

        for subscription in handlers:
            # 필터 확인
            if subscription.filter_fn and not subscription.filter_fn(event):
                continue

            try:
                if subscription.is_async:
                    if wait:
                        await subscription.handler(event)
                    else:
                        task = asyncio.create_task(subscription.handler(event))
                        tasks.append(task)
                else:
                    subscription.handler(event)

                delivered += 1
                self._stats["delivered"] += 1

            except Exception as e:
                self._stats["errors"] += 1
                logger.error(f"Event handler error ({subscription.subscription_id}): {e}")

        # 비동기 태스크 완료 대기 (wait=False인 경우)
        if tasks and not wait:
            asyncio.gather(*tasks, return_exceptions=True)

        return delivered

    def publish_sync(self, event: Union[DomainEvent, Dict[str, Any]]) -> int:
        """동기 이벤트 발행 (동기 핸들러만 호출)"""
        if isinstance(event, dict):
            event = DomainEvent.from_dict(event)

        self._event_history.append(event)
        if len(self._event_history) > self.max_history:
            self._event_history.pop(0)

        self._stats["published"] += 1

        handlers = self._get_matching_handlers(event)
        delivered = 0

        for subscription in handlers:
            if subscription.is_async:
                continue  # 비동기 핸들러 스킵

            if subscription.filter_fn and not subscription.filter_fn(event):
                continue

            try:
                subscription.handler(event)
                delivered += 1
                self._stats["delivered"] += 1
            except Exception as e:
                self._stats["errors"] += 1
                logger.error(f"Event handler error: {e}")

        return delivered

    def _get_matching_handlers(self, event: DomainEvent) -> List[Subscription]:
        """이벤트에 매칭되는 핸들러 수집"""
        handlers = []

        # 정확한 매칭
        if event.event_name in self._subscriptions:
            handlers.extend(self._subscriptions[event.event_name])

        # 와일드카드 매칭
        for subscription in self._wildcard_subscriptions:
            if self._match_pattern(subscription.pattern, event.event_name):
                handlers.append(subscription)

        # 우선순위 정렬
        handlers.sort(key=lambda s: -s.priority)
        return handlers

    def _match_pattern(self, pattern: str, event_name: str) -> bool:
        """와일드카드 패턴 매칭"""
        if pattern == "*":
            return True

        # 캐시된 정규식 사용
        if pattern not in self._pattern_cache:
            # 와일드카드를 정규식으로 변환
            regex_pattern = pattern.replace(".", r"\.").replace("*", r"[^.]*")
            self._pattern_cache[pattern] = re.compile(f"^{regex_pattern}$")

        return bool(self._pattern_cache[pattern].match(event_name))

    def get_history(
        self,
        event_pattern: Optional[str] = None,
        source: Optional[str] = None,
        correlation_id: Optional[str] = None,
        limit: int = 100,
    ) -> List[DomainEvent]:
        """이벤트 히스토리 조회"""
        results = self._event_history

        if event_pattern:
            results = [
                e for e in results
                if self._match_pattern(event_pattern, e.event_name)
            ]

        if source:
            results = [e for e in results if e.source == source]

        if correlation_id:
            results = [e for e in results if e.correlation_id == correlation_id]

        return results[-limit:]

    def get_causation_chain(self, event_id: str) -> List[DomainEvent]:
        """이벤트의 인과 체인 조회"""
        chain = []
        current_id = event_id

        while current_id:
            event = next(
                (e for e in self._event_history if e.event_id == current_id),
                None
            )
            if event:
                chain.append(event)
                current_id = event.causation_id
            else:
                break

        return list(reversed(chain))

    def get_stats(self) -> Dict[str, Any]:
        """통계 조회"""
        return {
            **self._stats,
            "subscriptions": len(self._subscriptions) + len(self._wildcard_subscriptions),
            "history_size": len(self._event_history),
        }

    def clear_history(self):
        """히스토리 초기화"""
        self._event_history.clear()


# === 표준 이벤트 이름 정의 ===

class OntologyEvents:
    """온톨로지 관련 이벤트"""
    CONCEPT_CREATED = "ontology.concept.created"
    CONCEPT_UPDATED = "ontology.concept.updated"
    CONCEPT_DELETED = "ontology.concept.deleted"
    CONCEPT_APPROVED = "ontology.concept.approved"
    CONCEPT_REJECTED = "ontology.concept.rejected"
    RELATIONSHIP_CREATED = "ontology.relationship.created"
    HOMEOMORPHISM_DISCOVERED = "ontology.homeomorphism.discovered"


class DiscoveryEvents:
    """탐색 관련 이벤트"""
    PHASE_STARTED = "discovery.phase.started"
    PHASE_COMPLETED = "discovery.phase.completed"
    SCHEMA_ANALYZED = "discovery.schema.analyzed"
    PATTERNS_DETECTED = "discovery.patterns.detected"
    DOMAIN_DETECTED = "discovery.domain.detected"


class GovernanceEvents:
    """거버넌스 관련 이벤트"""
    DECISION_MADE = "governance.decision.made"
    DECISION_ESCALATED = "governance.decision.escalated"
    POLICY_APPLIED = "governance.policy.applied"
    ACTION_CREATED = "governance.action.created"
    ACTION_COMPLETED = "governance.action.completed"


class InsightEvents:
    """인사이트 관련 이벤트"""
    INSIGHT_GENERATED = "insight.generated"
    ANOMALY_DETECTED = "insight.anomaly.detected"
    TREND_IDENTIFIED = "insight.trend.identified"
    FORECAST_COMPLETED = "insight.forecast.completed"


class SystemEvents:
    """시스템 이벤트"""
    PIPELINE_STARTED = "system.pipeline.started"
    PIPELINE_COMPLETED = "system.pipeline.completed"
    AGENT_TASK_STARTED = "system.agent.task.started"
    AGENT_TASK_COMPLETED = "system.agent.task.completed"
    ERROR_OCCURRED = "system.error.occurred"


# === 글로벌 이벤트 버스 ===

_global_event_bus: Optional[EventBus] = None


def get_event_bus() -> EventBus:
    """글로벌 이벤트 버스 가져오기"""
    global _global_event_bus
    if _global_event_bus is None:
        _global_event_bus = EventBus()
    return _global_event_bus


def create_event_bus(
    max_history: int = 1000,
    enable_logging: bool = True,
) -> EventBus:
    """새 이벤트 버스 생성"""
    return EventBus(
        max_history=max_history,
        enable_logging=enable_logging,
    )
