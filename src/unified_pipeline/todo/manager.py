"""
Todo Manager

DAG 기반 의존성 관리 및 작업 할당 시스템
- 의존성 그래프 관리
- 작업 큐 및 스케줄링
- 에이전트 할당
"""

import logging
import threading
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Any, Optional, Set, Callable
import heapq

from .models import (
    PipelineTodo,
    TodoStatus,
    TodoPriority,
    TodoResult,
    TodoEvent,
    TodoEventType,
    DISCOVERY_TODO_TEMPLATES,
    REFINEMENT_TODO_TEMPLATES,
    GOVERNANCE_TODO_TEMPLATES,
)
from .events import TodoEventEmitter, get_event_emitter

logger = logging.getLogger(__name__)


@dataclass
class TodoStats:
    """Todo 통계"""
    total: int = 0
    pending: int = 0
    ready: int = 0
    in_progress: int = 0
    completed: int = 0
    failed: int = 0
    blocked: int = 0
    cancelled: int = 0

    @property
    def completion_rate(self) -> float:
        """완료율"""
        if self.total == 0:
            return 0.0
        return self.completed / self.total

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total": self.total,
            "pending": self.pending,
            "ready": self.ready,
            "in_progress": self.in_progress,
            "completed": self.completed,
            "failed": self.failed,
            "blocked": self.blocked,
            "cancelled": self.cancelled,
            "completion_rate": self.completion_rate,
        }


class TodoManager:
    """
    Todo 매니저

    핵심 기능:
    1. DAG 기반 의존성 관리
    2. 우선순위 기반 작업 큐
    3. 에이전트별 작업 할당
    4. 자동 상태 전이 (PENDING -> READY)
    """

    def __init__(
        self,
        event_emitter: Optional[TodoEventEmitter] = None,
    ):
        """
        TodoManager 초기화

        Args:
            event_emitter: 이벤트 이미터 (없으면 전역 사용)
        """
        self.event_emitter = event_emitter or get_event_emitter()

        # Todo 저장소
        self._todos: Dict[str, PipelineTodo] = {}

        # 인덱스
        self._by_phase: Dict[str, Set[str]] = defaultdict(set)
        self._by_status: Dict[TodoStatus, Set[str]] = defaultdict(set)
        self._by_agent_type: Dict[str, Set[str]] = defaultdict(set)

        # 의존성 그래프
        self._dependency_graph: Dict[str, Set[str]] = defaultdict(set)  # todo -> dependencies
        self._dependent_graph: Dict[str, Set[str]] = defaultdict(set)   # todo -> dependents

        # 완료된 Todo ID들 (빠른 조회용)
        self._completed_todos: Set[str] = set()

        # 준비된 작업 큐 (우선순위 힙)
        self._ready_queue: List[tuple] = []  # (priority, created_at, todo_id)

        # 스레드 안전성
        self._lock = threading.RLock()

        # 콜백
        self._on_todo_ready: Optional[Callable[[PipelineTodo], None]] = None
        self._on_all_complete: Optional[Callable[[], None]] = None

        logger.info("TodoManager initialized")

    def set_callbacks(
        self,
        on_todo_ready: Optional[Callable[[PipelineTodo], None]] = None,
        on_all_complete: Optional[Callable[[], None]] = None,
    ) -> None:
        """콜백 설정"""
        self._on_todo_ready = on_todo_ready
        self._on_all_complete = on_all_complete

    # === Todo CRUD ===

    def add_todo(self, todo: PipelineTodo) -> str:
        """
        Todo 추가

        Args:
            todo: 추가할 Todo

        Returns:
            todo_id
        """
        with self._lock:
            if todo.todo_id in self._todos:
                raise ValueError(f"Todo already exists: {todo.todo_id}")

            # 저장
            self._todos[todo.todo_id] = todo

            # 인덱스 업데이트
            self._by_phase[todo.phase].add(todo.todo_id)
            self._by_status[todo.status].add(todo.todo_id)
            if todo.required_agent_type:
                self._by_agent_type[todo.required_agent_type].add(todo.todo_id)

            # 의존성 그래프 업데이트
            self._dependency_graph[todo.todo_id] = todo.dependencies.copy()
            for dep_id in todo.dependencies:
                self._dependent_graph[dep_id].add(todo.todo_id)
                # 의존성 Todo의 dependents도 업데이트
                if dep_id in self._todos:
                    self._todos[dep_id].dependents.add(todo.todo_id)

            # 이벤트 발행
            self.event_emitter.todo_created(
                todo_id=todo.todo_id,
                name=todo.name,
                phase=todo.phase,
                data={"task_type": todo.task_type, "dependencies": list(todo.dependencies)},
            )

            # 상태 체크 (의존성 없으면 바로 READY)
            self._check_and_update_status(todo.todo_id)

            logger.debug(f"Todo added: {todo.todo_id} ({todo.name})")
            return todo.todo_id

    def get_todo(self, todo_id: str) -> Optional[PipelineTodo]:
        """Todo 조회"""
        return self._todos.get(todo_id)

    def get_todos_by_phase(self, phase: str) -> List[PipelineTodo]:
        """Phase별 Todo 조회"""
        todo_ids = self._by_phase.get(phase, set())
        return [self._todos[tid] for tid in todo_ids if tid in self._todos]

    def get_todos_by_status(self, status: TodoStatus) -> List[PipelineTodo]:
        """상태별 Todo 조회"""
        todo_ids = self._by_status.get(status, set())
        return [self._todos[tid] for tid in todo_ids if tid in self._todos]

    def get_ready_todos(self, agent_type: Optional[str] = None) -> List[PipelineTodo]:
        """
        준비된 (실행 가능한) Todo 조회

        Args:
            agent_type: 특정 에이전트 타입 필터

        Returns:
            준비된 Todo 리스트 (우선순위순)
        """
        with self._lock:
            ready_todos = self.get_todos_by_status(TodoStatus.READY)

            if agent_type:
                ready_todos = [
                    t for t in ready_todos
                    if t.required_agent_type is None or t.required_agent_type == agent_type
                ]

            # 우선순위 순 정렬
            return sorted(ready_todos, key=lambda t: (t.priority.value, t.created_at))

    # === 작업 할당 ===

    def assign_todo(self, todo_id: str, agent_id: str) -> Optional[PipelineTodo]:
        """
        Todo를 에이전트에 할당

        Args:
            todo_id: Todo ID
            agent_id: 에이전트 ID

        Returns:
            할당된 Todo (없으면 None)
        """
        with self._lock:
            todo = self._todos.get(todo_id)
            if not todo or todo.status != TodoStatus.READY:
                return None

            # 상태 변경
            self._update_status(todo_id, TodoStatus.IN_PROGRESS)
            todo.start(agent_id)

            # 이벤트 발행
            self.event_emitter.todo_started(
                todo_id=todo_id,
                agent_id=agent_id,
                phase=todo.phase,
                name=todo.name,
            )

            logger.info(f"Todo assigned: {todo_id} -> {agent_id}")
            return todo

    def claim_next_todo(self, agent_id: str, agent_type: str) -> Optional[PipelineTodo]:
        """
        에이전트가 다음 작업을 가져감 (자율적)

        Args:
            agent_id: 에이전트 ID
            agent_type: 에이전트 타입

        Returns:
            할당된 Todo (없으면 None)
        """
        with self._lock:
            ready_todos = self.get_ready_todos(agent_type=agent_type)
            if not ready_todos:
                return None

            # 첫 번째 Todo 할당
            todo = ready_todos[0]
            return self.assign_todo(todo.todo_id, agent_id)

    # === 작업 완료/실패 ===

    def complete_todo(
        self,
        todo_id: str,
        result: TodoResult,
    ) -> bool:
        """
        Todo 완료 처리

        Args:
            todo_id: Todo ID
            result: 실행 결과

        Returns:
            성공 여부
        """
        with self._lock:
            todo = self._todos.get(todo_id)
            if not todo:
                logger.warning(f"Todo not found: {todo_id}")
                return False

            # 완료 처리
            todo.complete(result)
            self._update_status(todo_id, TodoStatus.COMPLETED)
            self._completed_todos.add(todo_id)

            # 이벤트 발행 (v14.0: terminate_mode, completion_signal 포함)
            self.event_emitter.todo_completed(
                todo_id=todo_id,
                execution_time=result.execution_time_seconds,
                phase=todo.phase,
                result_summary=result.output,
                terminate_mode=result.terminate_mode,  # v14.0
                completion_signal=result.completion_signal,  # v14.0
            )

            # 후속 Todo들의 상태 체크
            dependents = self._dependent_graph.get(todo_id, set())
            print(f"[TODO] Completed: '{todo.name}' ({todo_id}). Checking {len(dependents)} dependents: {[self._todos[d].name for d in dependents if d in self._todos]}")
            for dependent_id in dependents:
                dep_todo = self._todos.get(dependent_id)
                if dep_todo:
                    old_status = dep_todo.status.value
                    self._check_and_update_status(dependent_id)
                    new_status = dep_todo.status.value
                    if old_status != new_status:
                        print(f"[TODO] Dependent '{dep_todo.name}' changed: {old_status} -> {new_status}")
                    else:
                        # Show why it didn't change
                        unmet = dep_todo.dependencies - self._completed_todos
                        unmet_names = [self._todos[u].name for u in unmet if u in self._todos]
                        print(f"[TODO] Dependent '{dep_todo.name}' stays {old_status} (unmet deps: {unmet_names})")

            # 모든 Todo 완료 체크
            if self._check_all_complete():
                if self._on_all_complete:
                    self._on_all_complete()

            logger.info(f"Todo completed: {todo_id}")
            return True

    def fail_todo(self, todo_id: str, error: str) -> bool:
        """
        Todo 실패 처리

        Args:
            todo_id: Todo ID
            error: 에러 메시지

        Returns:
            재시도 여부
        """
        with self._lock:
            todo = self._todos.get(todo_id)
            if not todo:
                return False

            todo.fail(error)

            if todo.status == TodoStatus.FAILED:
                # 영구 실패
                self._update_status(todo_id, TodoStatus.FAILED)

                # 후속 Todo들 차단
                self._block_dependents(todo_id, f"Dependency {todo_id} failed")

                # 이벤트 발행
                self.event_emitter.todo_failed(
                    todo_id=todo_id,
                    error=error,
                    phase=todo.phase,
                    retry_count=todo.retry_count,
                )
                return False
            else:
                # 재시도 가능
                self._update_status(todo_id, TodoStatus.READY)

                # 이벤트 발행
                self.event_emitter.emit(TodoEvent.create(
                    event_type=TodoEventType.TODO_FAILED,
                    message=f"Todo failed, will retry ({todo.retry_count}/{todo.max_retries})",
                    todo_id=todo_id,
                    phase=todo.phase,
                    data={"error": error, "retry_count": todo.retry_count},
                ))
                return True

    def _block_dependents(self, todo_id: str, reason: str) -> None:
        """후속 Todo들 차단"""
        for dependent_id in self._dependent_graph.get(todo_id, set()):
            dependent = self._todos.get(dependent_id)
            if dependent and dependent.status not in {
                TodoStatus.COMPLETED, TodoStatus.CANCELLED
            }:
                dependent.block(reason)
                self._update_status(dependent_id, TodoStatus.BLOCKED)

                # 이벤트 발행
                self.event_emitter.emit(TodoEvent.create(
                    event_type=TodoEventType.TODO_BLOCKED,
                    message=f"Todo blocked: {reason}",
                    todo_id=dependent_id,
                    phase=dependent.phase,
                ))

                # 재귀적으로 후속 차단
                self._block_dependents(dependent_id, reason)

    # === 상태 관리 ===

    def _check_and_update_status(self, todo_id: str) -> None:
        """Todo 상태 체크 및 업데이트"""
        todo = self._todos.get(todo_id)
        if not todo or todo.status != TodoStatus.PENDING:
            return

        # 의존성 체크
        if todo.is_ready(self._completed_todos):
            self._update_status(todo_id, TodoStatus.READY)
            todo.status = TodoStatus.READY

            # 준비 큐에 추가
            heapq.heappush(
                self._ready_queue,
                (todo.priority.value, todo.created_at, todo_id)
            )

            # 이벤트 발행
            self.event_emitter.emit(TodoEvent.create(
                event_type=TodoEventType.TODO_READY,
                message=f"Todo ready: {todo.name}",
                todo_id=todo_id,
                phase=todo.phase,
            ))

            # 콜백 호출
            if self._on_todo_ready:
                self._on_todo_ready(todo)

            logger.debug(f"Todo ready: {todo_id}")

    def _update_status(self, todo_id: str, new_status: TodoStatus) -> None:
        """Todo 상태 업데이트 (인덱스 동기화)"""
        todo = self._todos.get(todo_id)
        if not todo:
            return

        old_status = todo.status

        # 인덱스에서 제거
        self._by_status[old_status].discard(todo_id)

        # 새 상태 설정
        todo.status = new_status

        # 인덱스에 추가
        self._by_status[new_status].add(todo_id)

    def _check_all_complete(self) -> bool:
        """모든 Todo 완료 여부 체크"""
        for todo in self._todos.values():
            if todo.status not in {
                TodoStatus.COMPLETED,
                TodoStatus.CANCELLED,
                TodoStatus.FAILED,
            }:
                return False
        return True

    # === Phase 관리 ===

    def initialize_phase(
        self,
        phase: str,
        input_data: Optional[Dict[str, Any]] = None,
    ) -> List[str]:
        """
        Phase 초기화 (Todo 템플릿으로부터 생성)

        Args:
            phase: phase 이름 ("discovery", "refinement", "governance" 또는
                   "phase1_discovery", "phase2_refinement", "phase3_governance")
            input_data: 초기 입력 데이터

        Returns:
            생성된 Todo ID 리스트
        """
        # v7.8: Phase 이름 정규화 - 양쪽 형식 모두 지원
        phase_mapping = {
            # 긴 형식 -> 짧은 형식
            "phase1_discovery": "discovery",
            "phase2_refinement": "refinement",
            "phase3_governance": "governance",
            # 짧은 형식은 그대로
            "discovery": "discovery",
            "refinement": "refinement",
            "governance": "governance",
        }
        normalized_phase = phase_mapping.get(phase, phase)

        templates = {
            "discovery": DISCOVERY_TODO_TEMPLATES,
            "refinement": REFINEMENT_TODO_TEMPLATES,
            "governance": GOVERNANCE_TODO_TEMPLATES,
        }.get(normalized_phase, [])

        if not templates:
            logger.warning(f"No templates for phase: {phase}")
            return []

        # 템플릿으로 Todo 생성
        created_ids = []
        task_type_to_id: Dict[str, str] = {}

        for template in templates:
            # 의존성 해석 (task_type -> todo_id)
            dependencies = set()
            for dep_task_type in template.get("dependencies", []):
                if dep_task_type in task_type_to_id:
                    dependencies.add(task_type_to_id[dep_task_type])

            todo = PipelineTodo.create(
                name=template["name"],
                description=template["description"],
                phase=phase,
                task_type=template["task_type"],
                dependencies=dependencies,
                required_agent_type=template.get("required_agent_type"),
                input_data=input_data or {},
            )

            self.add_todo(todo)
            created_ids.append(todo.todo_id)
            task_type_to_id[template["task_type"]] = todo.todo_id

        # Phase 시작 이벤트
        self.event_emitter.phase_started(phase, len(created_ids))

        logger.info(f"Phase {phase} initialized with {len(created_ids)} todos")
        return created_ids

    def is_phase_complete(self, phase: str) -> bool:
        """Phase 완료 여부"""
        phase_todos = self.get_todos_by_phase(phase)
        # v22.1: 디버깅 - 빈 리스트면 True 반환하는 문제 확인
        if not phase_todos:
            print(f"[DEBUG] is_phase_complete('{phase}'): NO TODOS FOUND - returning True (BUG?)")
            return True
        result = all(
            t.status in {TodoStatus.COMPLETED, TodoStatus.CANCELLED}
            for t in phase_todos
        )
        if result:
            print(f"[DEBUG] is_phase_complete('{phase}'): True - all {len(phase_todos)} todos done")
        return result

    def get_phase_stats(self, phase: str) -> TodoStats:
        """Phase별 통계"""
        stats = TodoStats()
        for todo in self.get_todos_by_phase(phase):
            stats.total += 1
            status_attr = todo.status.value
            setattr(stats, status_attr, getattr(stats, status_attr, 0) + 1)
        return stats

    def get_phase_gate_metrics(self, phase: str) -> Dict[str, Any]:
        """
        v14.0: Phase Gate 검증을 위한 상세 메트릭 수집

        Silent Failure 탐지 및 품질 검증을 위한 terminate_mode, completion_signal 분석

        Args:
            phase: Phase 이름

        Returns:
            {
                "total_todos": int,
                "completed_todos": int,
                "terminate_mode_counts": {"goal": N, "no_evidence": N, ...},
                "completion_signal_counts": {"[DONE]": N, "[NEEDS_MORE]": N, ...},
                "silent_failure_ratio": float,  # no_evidence / completed
                "confidence_analysis": {"avg": float, "min": float, "below_threshold": int},
                "gate_passed": bool,
                "gate_warnings": List[str],
            }
        """
        phase_todos = self.get_todos_by_phase(phase)
        completed_todos = [t for t in phase_todos if t.status == TodoStatus.COMPLETED and t.result]

        # 종료 모드 집계
        terminate_mode_counts: Dict[str, int] = {}
        for todo in completed_todos:
            mode = todo.result.terminate_mode or "unknown"
            terminate_mode_counts[mode] = terminate_mode_counts.get(mode, 0) + 1

        # 완료 시그널 집계
        completion_signal_counts: Dict[str, int] = {}
        for todo in completed_todos:
            signal = todo.result.completion_signal or "none"
            completion_signal_counts[signal] = completion_signal_counts.get(signal, 0) + 1

        # Silent Failure 비율 계산 (no_evidence + unknown) / total
        no_evidence_count = terminate_mode_counts.get("no_evidence", 0) + terminate_mode_counts.get("unknown", 0)
        silent_failure_ratio = no_evidence_count / len(completed_todos) if completed_todos else 0.0

        # 신뢰도 분석
        confidences = []
        for todo in completed_todos:
            if todo.result.output:
                conf = todo.result.output.get("confidence", todo.result.metadata.get("confidence"))
                if conf is not None:
                    confidences.append(float(conf))

        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
        min_confidence = min(confidences) if confidences else 0.0
        below_threshold = sum(1 for c in confidences if c < 0.5)

        # Gate 검증
        gate_passed = True
        gate_warnings = []

        # 경고 1: Silent Failure 비율이 50% 이상
        if silent_failure_ratio > 0.5:
            gate_warnings.append(f"High silent failure ratio: {silent_failure_ratio:.1%}")
            gate_passed = False

        # 경고 2: [BLOCKED] 또는 [ESCALATE] 시그널이 과반수
        blocked_escalated = completion_signal_counts.get("[BLOCKED]", 0) + completion_signal_counts.get("[ESCALATE]", 0)
        if completed_todos and blocked_escalated / len(completed_todos) > 0.5:
            gate_warnings.append(f"Too many blocked/escalated signals: {blocked_escalated}/{len(completed_todos)}")
            gate_passed = False

        # 경고 3: 평균 신뢰도가 0.3 미만
        if confidences and avg_confidence < 0.3:
            gate_warnings.append(f"Low average confidence: {avg_confidence:.2f}")
            # 경고만, 실패는 아님

        # 경고 4: 완료된 Todo가 없음
        if not completed_todos and phase_todos:
            gate_warnings.append("No completed todos in phase")
            gate_passed = False

        logger.info(
            f"[v14.0 Gate Metrics] Phase={phase}, "
            f"completed={len(completed_todos)}/{len(phase_todos)}, "
            f"silent_failure_ratio={silent_failure_ratio:.1%}, "
            f"gate_passed={gate_passed}"
        )

        return {
            "total_todos": len(phase_todos),
            "completed_todos": len(completed_todos),
            "terminate_mode_counts": terminate_mode_counts,
            "completion_signal_counts": completion_signal_counts,
            "silent_failure_ratio": silent_failure_ratio,
            "confidence_analysis": {
                "avg": avg_confidence,
                "min": min_confidence,
                "below_threshold": below_threshold,
                "sample_count": len(confidences),
            },
            "gate_passed": gate_passed,
            "gate_warnings": gate_warnings,
        }

    # === 전체 통계 ===

    def get_stats(self) -> TodoStats:
        """전체 통계"""
        stats = TodoStats()
        for todo in self._todos.values():
            stats.total += 1
            status_attr = todo.status.value
            current = getattr(stats, status_attr, 0)
            setattr(stats, status_attr, current + 1)
        return stats

    def get_progress(self) -> float:
        """전체 진행률"""
        if not self._todos:
            return 0.0
        completed = len(self._completed_todos)
        return completed / len(self._todos)

    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리 변환"""
        return {
            "stats": self.get_stats().to_dict(),
            "progress": self.get_progress(),
            "todos": {
                tid: todo.to_dict()
                for tid, todo in self._todos.items()
            },
            "phases": {
                phase: {
                    "stats": self.get_phase_stats(phase).to_dict(),
                    "todos": [t.todo_id for t in self.get_todos_by_phase(phase)],
                }
                for phase in self._by_phase.keys()
            },
        }

    # === 디버깅 ===

    def visualize_dag(self) -> str:
        """DAG 시각화 (텍스트)"""
        lines = ["=== Todo DAG ==="]
        for todo_id, todo in sorted(self._todos.items(), key=lambda x: x[1].created_at):
            deps = ", ".join(self._dependency_graph.get(todo_id, [])) or "none"
            status_emoji = {
                TodoStatus.PENDING: "⏳",
                TodoStatus.READY: "✅",
                TodoStatus.IN_PROGRESS: "🔄",
                TodoStatus.COMPLETED: "✔️",
                TodoStatus.FAILED: "❌",
                TodoStatus.BLOCKED: "🚫",
                TodoStatus.CANCELLED: "⛔",
            }.get(todo.status, "❓")
            lines.append(
                f"{status_emoji} [{todo.phase}] {todo.name} ({todo_id[:8]}...) "
                f"<- deps: [{deps}]"
            )
        return "\n".join(lines)
