"""
Todo Continuation Enforcer

미완료 Todo 자동 재개 시스템
- 유휴 상태 감지
- 미완료 Todo 자동 재실행
- 복구 전략 적용
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Any, Optional, Callable, Awaitable

from ..todo.manager import TodoManager
from ..todo.models import TodoStatus, PipelineTodo
from ..todo.events import TodoEventEmitter, get_event_emitter, TodoEventType, TodoEvent

logger = logging.getLogger(__name__)


@dataclass
class ContinuationConfig:
    """Continuation 설정"""
    idle_threshold_seconds: float = 5.0    # 유휴 감지 임계값
    check_interval_seconds: float = 1.0    # 체크 간격
    max_continuation_attempts: int = 3     # 최대 재개 시도 횟수
    backoff_multiplier: float = 2.0        # 재시도 백오프 배수
    enabled: bool = True                   # 활성화 여부


@dataclass
class ContinuationStats:
    """Continuation 통계"""
    continuations_triggered: int = 0
    todos_resumed: int = 0
    todos_abandoned: int = 0
    last_check_at: Optional[str] = None
    last_continuation_at: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "continuations_triggered": self.continuations_triggered,
            "todos_resumed": self.todos_resumed,
            "todos_abandoned": self.todos_abandoned,
            "last_check_at": self.last_check_at,
            "last_continuation_at": self.last_continuation_at,
        }


class TodoContinuationEnforcer:
    """
    Todo 지속 실행 강제기

    핵심 기능:
    1. 유휴 상태 감지 - 일정 시간 동안 진행이 없으면 감지
    2. 미완료 Todo 재개 - Ready/Pending 상태 Todo 자동 실행
    3. IN_PROGRESS 복구 - 중단된 작업 복구
    4. 실패 재시도 - 실패한 Todo 재시도

    사용 예:
    ```python
    enforcer = TodoContinuationEnforcer(
        todo_manager=manager,
        resume_callback=orchestrator.resume_todo,
    )
    await enforcer.start()
    ```
    """

    def __init__(
        self,
        todo_manager: TodoManager,
        resume_callback: Callable[[PipelineTodo], Awaitable[None]],
        config: Optional[ContinuationConfig] = None,
        event_emitter: Optional[TodoEventEmitter] = None,
    ):
        """
        Enforcer 초기화

        Args:
            todo_manager: Todo 매니저
            resume_callback: Todo 재개 콜백 (에이전트 실행)
            config: 설정
            event_emitter: 이벤트 이미터
        """
        self.todo_manager = todo_manager
        self.resume_callback = resume_callback
        self.config = config or ContinuationConfig()
        self.event_emitter = event_emitter or get_event_emitter()

        # 상태
        self._running = False
        self._last_activity_time = time.time()
        self._continuation_attempts: Dict[str, int] = {}  # todo_id -> attempts

        # 통계
        self.stats = ContinuationStats()

        # 태스크
        self._monitor_task: Optional[asyncio.Task] = None

        logger.info(f"TodoContinuationEnforcer initialized (idle_threshold={self.config.idle_threshold_seconds}s)")

    async def start(self) -> None:
        """모니터링 시작"""
        if self._running:
            return

        self._running = True
        self._last_activity_time = time.time()
        self._monitor_task = asyncio.create_task(self._monitor_loop())

        logger.info("TodoContinuationEnforcer started")

    async def stop(self) -> None:
        """모니터링 중지"""
        self._running = False

        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
            self._monitor_task = None

        logger.info("TodoContinuationEnforcer stopped")

    def record_activity(self) -> None:
        """활동 기록 (외부에서 호출)"""
        self._last_activity_time = time.time()

    async def _monitor_loop(self) -> None:
        """모니터링 루프"""
        while self._running:
            try:
                await asyncio.sleep(self.config.check_interval_seconds)

                if not self.config.enabled:
                    continue

                self.stats.last_check_at = datetime.now().isoformat()

                # 유휴 상태 체크
                idle_time = time.time() - self._last_activity_time

                if idle_time >= self.config.idle_threshold_seconds:
                    await self._handle_idle_state()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Continuation monitor error: {e}", exc_info=True)
                await asyncio.sleep(1.0)

    async def _handle_idle_state(self) -> None:
        """유휴 상태 처리"""
        # 미완료 Todo 확인
        pending_todos = self._get_resumable_todos()

        if not pending_todos:
            return

        logger.info(f"Idle state detected, {len(pending_todos)} todos to resume")

        # Continuation 이벤트 발행
        self.event_emitter.emit(TodoEvent.create(
            event_type=TodoEventType.CONTINUATION_TRIGGERED,
            message=f"Resuming {len(pending_todos)} idle todos",
            data={"todo_count": len(pending_todos)},
        ))

        self.stats.continuations_triggered += 1
        self.stats.last_continuation_at = datetime.now().isoformat()

        # Todo들 재개
        for todo in pending_todos:
            await self._resume_todo(todo)

    def _get_resumable_todos(self) -> List[PipelineTodo]:
        """재개 가능한 Todo 목록"""
        resumable = []

        # READY 상태 Todo (가장 우선)
        ready_todos = self.todo_manager.get_todos_by_status(TodoStatus.READY)
        for todo in ready_todos:
            if self._can_resume(todo):
                resumable.append(todo)

        # IN_PROGRESS 상태 중 정지된 것 (복구)
        in_progress_todos = self.todo_manager.get_todos_by_status(TodoStatus.IN_PROGRESS)
        for todo in in_progress_todos:
            # 진행률이 오래 변하지 않은 경우
            if self._is_stalled(todo) and self._can_resume(todo):
                resumable.append(todo)

        # 우선순위순 정렬
        return sorted(resumable, key=lambda t: (t.priority.value, t.created_at))

    def _can_resume(self, todo: PipelineTodo) -> bool:
        """Todo 재개 가능 여부"""
        attempts = self._continuation_attempts.get(todo.todo_id, 0)
        return attempts < self.config.max_continuation_attempts

    def _is_stalled(self, todo: PipelineTodo) -> bool:
        """Todo가 정지 상태인지"""
        # 시작 후 일정 시간이 지났는데 진행률이 없으면 정지로 간주
        if not todo.started_at:
            return False

        try:
            started = datetime.fromisoformat(todo.started_at)
            elapsed = (datetime.now() - started).total_seconds()

            # 30초 이상 경과했는데 진행률이 10% 미만이면 정지
            if elapsed > 30 and todo.progress < 0.1:
                return True

            # 2분 이상 경과했는데 진행률이 50% 미만이면 정지
            if elapsed > 120 and todo.progress < 0.5:
                return True

        except Exception:
            pass

        return False

    async def _resume_todo(self, todo: PipelineTodo) -> None:
        """Todo 재개"""
        try:
            # 시도 횟수 증가
            self._continuation_attempts[todo.todo_id] = \
                self._continuation_attempts.get(todo.todo_id, 0) + 1

            attempts = self._continuation_attempts[todo.todo_id]

            logger.info(f"Resuming todo {todo.todo_id} (attempt {attempts})")

            # 백오프 적용
            if attempts > 1:
                backoff = self.config.backoff_multiplier ** (attempts - 1)
                await asyncio.sleep(min(backoff, 10.0))

            # 콜백 호출 (에이전트 실행)
            await self.resume_callback(todo)

            self.stats.todos_resumed += 1
            self._last_activity_time = time.time()

        except Exception as e:
            logger.error(f"Failed to resume todo {todo.todo_id}: {e}")

            # 최대 시도 초과 시 포기
            if self._continuation_attempts.get(todo.todo_id, 0) >= self.config.max_continuation_attempts:
                logger.warning(f"Abandoning todo {todo.todo_id} after max attempts")
                self.stats.todos_abandoned += 1

    def reset_attempts(self, todo_id: str) -> None:
        """시도 횟수 리셋 (성공 시 호출)"""
        if todo_id in self._continuation_attempts:
            del self._continuation_attempts[todo_id]

    def get_status(self) -> Dict[str, Any]:
        """상태 조회"""
        return {
            "running": self._running,
            "enabled": self.config.enabled,
            "idle_threshold": self.config.idle_threshold_seconds,
            "current_idle_time": time.time() - self._last_activity_time,
            "pending_attempts": dict(self._continuation_attempts),
            "stats": self.stats.to_dict(),
        }


class ContinuationAwareOrchestrator:
    """
    Continuation 기능이 통합된 오케스트레이터 래퍼

    AgentOrchestrator에 TodoContinuationEnforcer를 통합하여
    미완료 작업 자동 재개 기능 제공
    """

    def __init__(self, orchestrator: "AgentOrchestrator"):
        """
        래퍼 초기화

        Args:
            orchestrator: 기존 오케스트레이터
        """
        from .orchestrator import AgentOrchestrator

        self.orchestrator = orchestrator

        # Continuation Enforcer 생성
        self.enforcer = TodoContinuationEnforcer(
            todo_manager=orchestrator.todo_manager,
            resume_callback=self._resume_todo,
            event_emitter=orchestrator.event_emitter,
        )

    async def _resume_todo(self, todo: PipelineTodo) -> None:
        """Todo 재개 콜백"""
        agent = await self.orchestrator._assign_agent_to_todo(todo)
        if agent:
            logger.info(f"Todo {todo.todo_id} resumed with agent {agent.agent_id}")

    async def run_pipeline_with_continuation(
        self,
        tables_data: Dict[str, Dict[str, Any]],
        phases: Optional[List[str]] = None,
    ):
        """Continuation이 활성화된 파이프라인 실행"""
        # Enforcer 시작
        await self.enforcer.start()

        try:
            # 파이프라인 실행
            async for event in self.orchestrator.run_pipeline(tables_data, phases):
                # 활동 기록
                self.enforcer.record_activity()

                # 완료된 Todo 시도 횟수 리셋
                if event.get("type") == "todo_completed":
                    self.enforcer.reset_attempts(event.get("todo_id"))

                yield event

        finally:
            # Enforcer 중지
            await self.enforcer.stop()

    def get_status(self) -> Dict[str, Any]:
        """상태 조회"""
        return {
            "orchestrator": self.orchestrator.get_status(),
            "enforcer": self.enforcer.get_status(),
        }
