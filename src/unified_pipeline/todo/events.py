"""
Todo Event Emitter

실시간 이벤트 스트리밍 시스템
- WebSocket/SSE를 통한 FE 실시간 모니터링 지원
- 이벤트 버퍼링 및 배치 처리
- 구독자 패턴 지원
"""

import asyncio
import logging
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, Dict, List, Any, Optional, Set, AsyncGenerator
from enum import Enum

from .models import TodoEvent, TodoEventType

logger = logging.getLogger(__name__)


class EventPriority(int, Enum):
    """이벤트 우선순위"""
    CRITICAL = 0   # 즉시 전송 (에러, 완료)
    HIGH = 1       # 높은 우선순위 (상태 변경)
    NORMAL = 2     # 일반 (진행률)
    LOW = 3        # 낮은 우선순위 (디버그)


@dataclass
class EventSubscription:
    """이벤트 구독"""
    subscriber_id: str
    event_types: Set[TodoEventType]
    callback: Optional[Callable[[TodoEvent], None]] = None
    async_callback: Optional[Callable[[TodoEvent], Any]] = None
    phase_filter: Optional[str] = None  # 특정 phase만 구독
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())


class TodoEventEmitter:
    """
    Todo 이벤트 이미터

    기능:
    - 동기/비동기 이벤트 발행
    - 구독자 관리
    - 이벤트 히스토리 버퍼
    - 배치 처리 지원
    """

    def __init__(
        self,
        buffer_size: int = 1000,
        batch_size: int = 10,
        batch_interval_ms: int = 100,
    ):
        """
        이벤트 이미터 초기화

        Args:
            buffer_size: 이벤트 히스토리 버퍼 크기
            batch_size: 배치 처리 크기
            batch_interval_ms: 배치 처리 간격 (ms)
        """
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.batch_interval_ms = batch_interval_ms

        # 구독자
        self._subscriptions: Dict[str, EventSubscription] = {}

        # 이벤트 히스토리 (최근 N개)
        self._event_history: deque = deque(maxlen=buffer_size)

        # 배치 버퍼
        self._batch_buffer: List[TodoEvent] = []

        # 통계
        self._stats = {
            "total_events": 0,
            "events_by_type": {},
        }

        logger.info(f"TodoEventEmitter initialized (buffer={buffer_size})")

    def subscribe(
        self,
        subscriber_id: str,
        event_types: Optional[Set[TodoEventType]] = None,
        callback: Optional[Callable[[TodoEvent], None]] = None,
        async_callback: Optional[Callable[[TodoEvent], Any]] = None,
        phase_filter: Optional[str] = None,
    ) -> EventSubscription:
        """
        이벤트 구독

        Args:
            subscriber_id: 구독자 ID
            event_types: 구독할 이벤트 타입들 (None이면 전체)
            callback: 동기 콜백
            async_callback: 비동기 콜백
            phase_filter: 특정 phase만 구독

        Returns:
            구독 정보
        """
        subscription = EventSubscription(
            subscriber_id=subscriber_id,
            event_types=event_types or set(TodoEventType),
            callback=callback,
            async_callback=async_callback,
            phase_filter=phase_filter,
        )
        self._subscriptions[subscriber_id] = subscription
        logger.debug(f"Subscriber added: {subscriber_id}")
        return subscription

    def unsubscribe(self, subscriber_id: str) -> bool:
        """
        구독 해제

        Args:
            subscriber_id: 구독자 ID

        Returns:
            성공 여부
        """
        if subscriber_id in self._subscriptions:
            del self._subscriptions[subscriber_id]
            logger.debug(f"Subscriber removed: {subscriber_id}")
            return True
        return False

    def emit(self, event: TodoEvent) -> None:
        """
        이벤트 발행 (동기)

        Args:
            event: 발행할 이벤트
        """
        # 히스토리에 추가
        self._event_history.append(event)

        # 통계 업데이트
        self._stats["total_events"] += 1
        event_type = event.event_type.value
        self._stats["events_by_type"][event_type] = \
            self._stats["events_by_type"].get(event_type, 0) + 1

        # 로깅
        log_level = self._get_log_level(event.event_type)
        logger.log(
            log_level,
            f"[{event.event_type.value}] {event.message} "
            f"(todo={event.todo_id}, agent={event.agent_id})"
        )

        # 구독자에게 전달
        for subscription in self._subscriptions.values():
            if self._should_notify(subscription, event):
                if subscription.callback:
                    try:
                        subscription.callback(event)
                    except Exception as e:
                        logger.error(
                            f"Callback error for {subscription.subscriber_id}: {e}"
                        )

    async def emit_async(self, event: TodoEvent) -> None:
        """
        이벤트 발행 (비동기)

        Args:
            event: 발행할 이벤트
        """
        # 동기 처리
        self.emit(event)

        # 비동기 콜백 처리
        for subscription in self._subscriptions.values():
            if self._should_notify(subscription, event):
                if subscription.async_callback:
                    try:
                        await subscription.async_callback(event)
                    except Exception as e:
                        logger.error(
                            f"Async callback error for {subscription.subscriber_id}: {e}"
                        )

    def emit_batch(self, events: List[TodoEvent]) -> None:
        """
        배치 이벤트 발행

        Args:
            events: 이벤트 리스트
        """
        for event in events:
            self.emit(event)

    # === 편의 메서드들 ===

    def todo_created(
        self,
        todo_id: str,
        name: str,
        phase: str,
        data: Optional[Dict[str, Any]] = None,
    ) -> TodoEvent:
        """Todo 생성 이벤트"""
        event = TodoEvent.create(
            event_type=TodoEventType.TODO_CREATED,
            message=f"Todo created: {name}",
            todo_id=todo_id,
            phase=phase,
            data=data or {"name": name},
        )
        self.emit(event)
        return event

    def todo_started(
        self,
        todo_id: str,
        agent_id: str,
        phase: str,
        name: str = "",
    ) -> TodoEvent:
        """Todo 시작 이벤트"""
        event = TodoEvent.create(
            event_type=TodoEventType.TODO_STARTED,
            message=f"Todo started: {name}" if name else f"Todo started",
            todo_id=todo_id,
            agent_id=agent_id,
            phase=phase,
            progress=0.0,
        )
        self.emit(event)
        return event

    def todo_progress(
        self,
        todo_id: str,
        progress: float,
        message: str = "",
        agent_id: Optional[str] = None,
        phase: Optional[str] = None,
    ) -> TodoEvent:
        """Todo 진행률 이벤트"""
        event = TodoEvent.create(
            event_type=TodoEventType.TODO_PROGRESS,
            message=message or f"Progress: {progress:.1%}",
            todo_id=todo_id,
            agent_id=agent_id,
            phase=phase,
            progress=progress,
        )
        self.emit(event)
        return event

    def todo_completed(
        self,
        todo_id: str,
        execution_time: float,
        phase: str,
        result_summary: Optional[Dict[str, Any]] = None,
        terminate_mode: Optional[str] = None,  # v14.0
        completion_signal: Optional[str] = None,  # v14.0
    ) -> TodoEvent:
        """Todo 완료 이벤트"""
        event = TodoEvent.create(
            event_type=TodoEventType.TODO_COMPLETED,
            message=f"Todo completed in {execution_time:.2f}s",
            todo_id=todo_id,
            phase=phase,
            progress=1.0,
            data={
                "execution_time": execution_time,
                "result_summary": result_summary or {},
            },
            terminate_mode=terminate_mode,  # v14.0
            completion_signal=completion_signal,  # v14.0
        )
        self.emit(event)
        return event

    def todo_failed(
        self,
        todo_id: str,
        error: str,
        phase: str,
        retry_count: int = 0,
    ) -> TodoEvent:
        """Todo 실패 이벤트"""
        event = TodoEvent.create(
            event_type=TodoEventType.TODO_FAILED,
            message=f"Todo failed: {error}",
            todo_id=todo_id,
            phase=phase,
            data={
                "error": error,
                "retry_count": retry_count,
            },
        )
        self.emit(event)
        return event

    def agent_thinking(
        self,
        agent_id: str,
        todo_id: str,
        thinking: str,
        phase: Optional[str] = None,
    ) -> TodoEvent:
        """에이전트 사고 과정 이벤트"""
        event = TodoEvent.create(
            event_type=TodoEventType.AGENT_THINKING,
            message=thinking[:200] + "..." if len(thinking) > 200 else thinking,
            agent_id=agent_id,
            todo_id=todo_id,
            phase=phase,
            data={"full_thinking": thinking},
        )
        self.emit(event)
        return event

    def agent_output(
        self,
        agent_id: str,
        todo_id: str,
        output: Dict[str, Any],
        phase: Optional[str] = None,
    ) -> TodoEvent:
        """에이전트 출력 이벤트"""
        event = TodoEvent.create(
            event_type=TodoEventType.AGENT_OUTPUT,
            message=f"Agent {agent_id} produced output",
            agent_id=agent_id,
            todo_id=todo_id,
            phase=phase,
            data=output,
        )
        self.emit(event)
        return event

    def phase_started(self, phase: str, todo_count: int) -> TodoEvent:
        """Phase 시작 이벤트"""
        event = TodoEvent.create(
            event_type=TodoEventType.PHASE_STARTED,
            message=f"Phase {phase} started with {todo_count} todos",
            phase=phase,
            data={"todo_count": todo_count},
            progress=0.0,
        )
        self.emit(event)
        return event

    def phase_completed(
        self,
        phase: str,
        execution_time: float,
        stats: Dict[str, Any],
    ) -> TodoEvent:
        """Phase 완료 이벤트"""
        event = TodoEvent.create(
            event_type=TodoEventType.PHASE_COMPLETED,
            message=f"Phase {phase} completed in {execution_time:.2f}s",
            phase=phase,
            progress=1.0,
            data={
                "execution_time": execution_time,
                "stats": stats,
            },
        )
        self.emit(event)
        return event

    def pipeline_started(self, scenario: str, phases: List[str]) -> TodoEvent:
        """파이프라인 시작 이벤트"""
        event = TodoEvent.create(
            event_type=TodoEventType.PIPELINE_STARTED,
            message=f"Pipeline started: {scenario}",
            data={
                "scenario": scenario,
                "phases": phases,
            },
            progress=0.0,
        )
        self.emit(event)
        return event

    def pipeline_completed(
        self,
        execution_time: float,
        stats: Dict[str, Any],
    ) -> TodoEvent:
        """파이프라인 완료 이벤트"""
        event = TodoEvent.create(
            event_type=TodoEventType.PIPELINE_COMPLETED,
            message=f"Pipeline completed in {execution_time:.2f}s",
            progress=1.0,
            data={
                "execution_time": execution_time,
                "stats": stats,
            },
        )
        self.emit(event)
        return event

    # === 유틸리티 메서드들 ===

    def _should_notify(
        self,
        subscription: EventSubscription,
        event: TodoEvent,
    ) -> bool:
        """구독자에게 알림 필요 여부"""
        # 이벤트 타입 필터
        if event.event_type not in subscription.event_types:
            return False

        # Phase 필터
        if subscription.phase_filter and event.phase != subscription.phase_filter:
            return False

        return True

    def _get_log_level(self, event_type: TodoEventType) -> int:
        """이벤트 타입별 로그 레벨"""
        critical_types = {
            TodoEventType.TODO_FAILED,
            TodoEventType.AGENT_ERROR,
            TodoEventType.PIPELINE_ERROR,
        }
        warning_types = {
            TodoEventType.TODO_BLOCKED,
            TodoEventType.TODO_CANCELLED,
        }
        info_types = {
            TodoEventType.TODO_COMPLETED,
            TodoEventType.TODO_STARTED,
            TodoEventType.PHASE_STARTED,
            TodoEventType.PHASE_COMPLETED,
            TodoEventType.PIPELINE_STARTED,
            TodoEventType.PIPELINE_COMPLETED,
        }

        if event_type in critical_types:
            return logging.ERROR
        elif event_type in warning_types:
            return logging.WARNING
        elif event_type in info_types:
            return logging.INFO
        else:
            return logging.DEBUG

    def get_history(
        self,
        limit: int = 100,
        event_types: Optional[Set[TodoEventType]] = None,
        phase: Optional[str] = None,
    ) -> List[TodoEvent]:
        """
        이벤트 히스토리 조회

        Args:
            limit: 최대 개수
            event_types: 필터링할 이벤트 타입
            phase: 필터링할 phase

        Returns:
            이벤트 리스트 (최신순)
        """
        events = list(self._event_history)

        # 필터링
        if event_types:
            events = [e for e in events if e.event_type in event_types]
        if phase:
            events = [e for e in events if e.phase == phase]

        # 최신순 정렬 후 limit 적용
        return list(reversed(events))[:limit]

    def get_stats(self) -> Dict[str, Any]:
        """통계 조회"""
        return {
            **self._stats,
            "buffer_size": len(self._event_history),
            "subscriber_count": len(self._subscriptions),
        }

    def clear_history(self) -> None:
        """히스토리 초기화"""
        self._event_history.clear()
        logger.info("Event history cleared")

    async def stream_events(
        self,
        subscriber_id: str,
        event_types: Optional[Set[TodoEventType]] = None,
    ) -> AsyncGenerator[TodoEvent, None]:
        """
        이벤트 스트리밍 (SSE/WebSocket용)

        Args:
            subscriber_id: 구독자 ID
            event_types: 구독할 이벤트 타입

        Yields:
            TodoEvent
        """
        queue: asyncio.Queue = asyncio.Queue()

        async def enqueue_event(event: TodoEvent):
            await queue.put(event)

        # 구독 등록
        self.subscribe(
            subscriber_id=subscriber_id,
            event_types=event_types,
            async_callback=enqueue_event,
        )

        try:
            while True:
                event = await queue.get()
                yield event
        finally:
            self.unsubscribe(subscriber_id)


# 전역 이벤트 이미터 (싱글톤)
_global_emitter: Optional[TodoEventEmitter] = None


def get_event_emitter() -> TodoEventEmitter:
    """전역 이벤트 이미터 가져오기"""
    global _global_emitter
    if _global_emitter is None:
        _global_emitter = TodoEventEmitter()
    return _global_emitter


def reset_event_emitter() -> None:
    """전역 이벤트 이미터 리셋"""
    global _global_emitter
    _global_emitter = None
