"""
Workflow Orchestrator

액션 기반 워크플로우 자동화:
- 인사이트 → 액션 → 태스크 매핑
- 우선순위 기반 스케줄링
- 의존성 해결
- 진행 추적 및 결과 측정

DomainContext 워크플로우 설정 활용

References:
- Van Der Aalst "Workflow Management" (2004)
- Apache Airflow Architecture
"""

import asyncio
import logging
import uuid
import heapq
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Callable, Awaitable
from datetime import datetime, timedelta
from enum import Enum
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class TaskStatus(str, Enum):
    """태스크 상태"""
    PENDING = "pending"       # 대기 중
    SCHEDULED = "scheduled"   # 스케줄됨
    RUNNING = "running"       # 실행 중
    COMPLETED = "completed"   # 완료
    FAILED = "failed"         # 실패
    CANCELLED = "cancelled"   # 취소됨
    BLOCKED = "blocked"       # 의존성 대기
    RETRYING = "retrying"     # 재시도 중


class TaskPriority(int, Enum):
    """태스크 우선순위 (낮을수록 높은 우선순위)"""
    CRITICAL = 1
    HIGH = 2
    MEDIUM = 3
    LOW = 4
    BACKGROUND = 5


class WorkflowType(str, Enum):
    """워크플로우 타입"""
    DATA_QUALITY = "data_quality_monitoring"
    ANOMALY_RESPONSE = "anomaly_response"
    GOVERNANCE = "governance_action"
    NOTIFICATION = "notification"
    REMEDIATION = "remediation"
    ESCALATION = "escalation"
    COMPLIANCE = "compliance_check"
    CUSTOM = "custom"


@dataclass
class WorkflowTask:
    """워크플로우 태스크"""
    task_id: str
    name: str
    workflow_type: str
    priority: TaskPriority
    action_id: str  # 연결된 액션 ID

    # 상태
    status: TaskStatus = TaskStatus.PENDING

    # 의존성
    dependencies: List[str] = field(default_factory=list)

    # 파라미터
    params: Dict[str, Any] = field(default_factory=dict)

    # 타임스탬프
    created_at: datetime = field(default_factory=datetime.now)
    scheduled_at: Optional[datetime] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    # 결과
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

    # 재시도
    retry_count: int = 0
    max_retries: int = 3

    # 타임아웃 (초) - v18.0: 비활성화 (0 = 무제한)
    timeout_seconds: int = 0

    # 메타데이터
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __lt__(self, other):
        """우선순위 비교 (heapq용)"""
        if self.priority.value != other.priority.value:
            return self.priority.value < other.priority.value
        return self.created_at < other.created_at

    @property
    def duration_seconds(self) -> Optional[float]:
        """실행 시간 (초)"""
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        return None

    @property
    def is_terminal(self) -> bool:
        """종료 상태인지 확인"""
        return self.status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "name": self.name,
            "workflow_type": self.workflow_type,
            "priority": self.priority.name,
            "status": self.status.value,
            "action_id": self.action_id,
            "dependencies": self.dependencies,
            "params": self.params,
            "created_at": self.created_at.isoformat(),
            "scheduled_at": self.scheduled_at.isoformat() if self.scheduled_at else None,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "duration_seconds": self.duration_seconds,
            "result": self.result,
            "error": self.error,
            "retry_count": self.retry_count,
        }


@dataclass
class WorkflowExecution:
    """워크플로우 실행 기록"""
    execution_id: str
    workflow_type: str
    trigger_source: str  # 트리거 소스 (insight_id, manual, scheduled)

    tasks: List[str] = field(default_factory=list)
    started_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    status: str = "running"

    # 메트릭
    metrics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "execution_id": self.execution_id,
            "workflow_type": self.workflow_type,
            "trigger_source": self.trigger_source,
            "tasks": self.tasks,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "status": self.status,
            "metrics": self.metrics,
        }


class TaskHandler(ABC):
    """태스크 핸들러 추상 클래스"""

    @abstractmethod
    async def execute(self, task: WorkflowTask) -> Dict[str, Any]:
        """태스크 실행"""
        pass

    @property
    @abstractmethod
    def workflow_type(self) -> str:
        """지원하는 워크플로우 타입"""
        pass


class DataQualityHandler(TaskHandler):
    """데이터 품질 모니터링 핸들러"""

    @property
    def workflow_type(self) -> str:
        return WorkflowType.DATA_QUALITY.value

    async def execute(self, task: WorkflowTask) -> Dict[str, Any]:
        logger.info(f"Executing data quality check: {task.name}")

        # 데이터 품질 체크 시뮬레이션
        await asyncio.sleep(0.1)

        return {
            "status": "completed",
            "quality_score": 0.95,
            "issues_found": 0,
            "checked_at": datetime.now().isoformat(),
        }


class NotificationHandler(TaskHandler):
    """알림 핸들러"""

    @property
    def workflow_type(self) -> str:
        return WorkflowType.NOTIFICATION.value

    async def execute(self, task: WorkflowTask) -> Dict[str, Any]:
        recipients = task.params.get("recipients", [])
        message = task.params.get("message", "No message")

        logger.info(f"Sending notification to {len(recipients)} recipients: {message[:50]}...")

        # 알림 전송 시뮬레이션
        await asyncio.sleep(0.05)

        return {
            "status": "sent",
            "recipients_count": len(recipients),
            "sent_at": datetime.now().isoformat(),
        }


class EscalationHandler(TaskHandler):
    """에스컬레이션 핸들러"""

    def __init__(self, domain_context=None):
        self.domain_context = domain_context

    @property
    def workflow_type(self) -> str:
        return WorkflowType.ESCALATION.value

    async def execute(self, task: WorkflowTask) -> Dict[str, Any]:
        issue_type = task.params.get("issue_type", "general")

        # DomainContext에서 에스컬레이션 경로 조회
        escalation_path = []
        if self.domain_context and hasattr(self.domain_context, 'get_escalation_path'):
            escalation_path = self.domain_context.get_escalation_path(issue_type)
        else:
            escalation_path = ["Team Lead", "Manager", "Director"]

        logger.info(f"Escalating {issue_type} issue via: {escalation_path}")

        await asyncio.sleep(0.1)

        return {
            "status": "escalated",
            "escalation_path": escalation_path,
            "issue_type": issue_type,
            "escalated_at": datetime.now().isoformat(),
        }


class TaskExecutor:
    """태스크 실행기"""

    def __init__(self, domain_context=None):
        self.domain_context = domain_context
        self.handlers: Dict[str, TaskHandler] = {}

        # 기본 핸들러 등록
        self._register_default_handlers()

    def _register_default_handlers(self):
        """기본 핸들러 등록"""
        self.register_handler(DataQualityHandler())
        self.register_handler(NotificationHandler())
        self.register_handler(EscalationHandler(self.domain_context))

    def register_handler(self, handler: TaskHandler):
        """핸들러 등록"""
        self.handlers[handler.workflow_type] = handler
        logger.debug(f"Registered handler for: {handler.workflow_type}")

    def register_custom_handler(
        self,
        workflow_type: str,
        handler_func: Callable[[WorkflowTask], Awaitable[Dict[str, Any]]],
    ):
        """커스텀 핸들러 등록"""
        class CustomHandler(TaskHandler):
            @property
            def workflow_type(self) -> str:
                return workflow_type

            async def execute(self, task: WorkflowTask) -> Dict[str, Any]:
                return await handler_func(task)

        self.handlers[workflow_type] = CustomHandler()

    async def execute(self, task: WorkflowTask) -> Dict[str, Any]:
        """태스크 실행"""
        handler = self.handlers.get(task.workflow_type)

        if handler:
            return await handler.execute(task)

        # 핸들러 없으면 DomainContext 워크플로우 설정 사용
        if self.domain_context and hasattr(self.domain_context, 'get_workflow_config'):
            config = self.domain_context.get_workflow_config(task.workflow_type)
            return await self._execute_with_config(task, config)

        # 기본 실행
        return await self._default_execute(task)

    async def _execute_with_config(
        self,
        task: WorkflowTask,
        config: Dict[str, Any],
    ) -> Dict[str, Any]:
        """설정 기반 실행"""
        logger.info(f"Executing task {task.task_id} with config: {config}")

        # 설정에서 실행 시간 시뮬레이션
        delay = config.get("execution_delay_ms", 100) / 1000
        await asyncio.sleep(delay)

        return {
            "status": "completed",
            "workflow_type": task.workflow_type,
            "config_used": config,
            "execution_time_ms": delay * 1000,
        }

    async def _default_execute(self, task: WorkflowTask) -> Dict[str, Any]:
        """기본 실행"""
        logger.info(f"Default execution for task: {task.task_id}")
        await asyncio.sleep(0.1)

        return {
            "status": "completed",
            "message": "Default execution completed",
            "workflow_type": task.workflow_type,
        }


class WorkflowOrchestrator:
    """
    워크플로우 오케스트레이터

    Phase 3 Governance 액션을 실행 가능한 태스크로 변환하고
    우선순위 기반으로 스케줄링/실행

    기능:
    - 인사이트 → 액션 → 태스크 변환
    - 우선순위 기반 스케줄링
    - 의존성 해결
    - 병렬 실행
    - 재시도 및 타임아웃
    - 진행 추적
    """

    def __init__(
        self,
        domain_context=None,
        max_concurrent_tasks: int = 5,
        default_timeout: int = 0,  # v18.0: 타임아웃 비활성화 (0 = 무제한)
    ):
        """
        Args:
            domain_context: DomainContext 객체
            max_concurrent_tasks: 최대 동시 실행 태스크 수
            default_timeout: 기본 타임아웃 (초) - v18.0: 0 = 무제한
        """
        self.domain_context = domain_context
        self.max_concurrent_tasks = max_concurrent_tasks
        self.default_timeout = default_timeout

        # 실행기
        self.executor = TaskExecutor(domain_context)

        # 태스크 관리
        self.task_queue: List[WorkflowTask] = []  # 우선순위 힙
        self.tasks: Dict[str, WorkflowTask] = {}
        self.executions: Dict[str, WorkflowExecution] = {}

        # 상태
        self.running = False

        # 통계
        self._stats = {
            "total_tasks": 0,
            "completed_tasks": 0,
            "failed_tasks": 0,
            "total_executions": 0,
        }

    def create_task_from_action(
        self,
        action: Dict[str, Any],
        priority: Optional[TaskPriority] = None,
        dependencies: Optional[List[str]] = None,
    ) -> WorkflowTask:
        """
        Phase 3 액션에서 태스크 생성

        Args:
            action: 액션 딕셔너리 (governance 모듈에서 생성)
            priority: 우선순위 (None이면 자동 결정)
            dependencies: 의존성 태스크 ID 리스트

        Returns:
            생성된 WorkflowTask
        """
        task_id = f"TASK-{uuid.uuid4().hex[:8]}"

        # 워크플로우 타입 결정
        workflow_type = action.get("workflow_type")
        if not workflow_type:
            # 액션 타입에서 추론
            action_type = action.get("type", "").lower()
            if "quality" in action_type:
                workflow_type = WorkflowType.DATA_QUALITY.value
            elif "notify" in action_type or "alert" in action_type:
                workflow_type = WorkflowType.NOTIFICATION.value
            elif "escalate" in action_type:
                workflow_type = WorkflowType.ESCALATION.value
            else:
                workflow_type = WorkflowType.CUSTOM.value

        # 도메인 컨텍스트에서 권장 워크플로우 확인
        if self.domain_context:
            recommended = getattr(self.domain_context, 'recommended_workflows', [])
            if recommended and workflow_type not in recommended:
                logger.debug(f"Workflow type {workflow_type} not in recommended: {recommended}")

        # 우선순위 결정
        if priority is None:
            severity = action.get("severity", "medium").lower()
            priority_map = {
                "critical": TaskPriority.CRITICAL,
                "high": TaskPriority.HIGH,
                "medium": TaskPriority.MEDIUM,
                "low": TaskPriority.LOW,
            }
            priority = priority_map.get(severity, TaskPriority.MEDIUM)

        task = WorkflowTask(
            task_id=task_id,
            name=action.get("title", "Unnamed Action"),
            workflow_type=workflow_type,
            priority=priority,
            action_id=action.get("action_id", ""),
            dependencies=dependencies or [],
            params=action.get("params", {}),
            timeout_seconds=action.get("timeout", self.default_timeout),
            metadata={
                "source_insight": action.get("source_insight_id"),
                "action_type": action.get("type"),
            },
        )

        self.tasks[task_id] = task
        heapq.heappush(self.task_queue, task)
        self._stats["total_tasks"] += 1

        logger.info(f"Created task {task_id} from action {action.get('action_id')}")
        return task

    def create_task(
        self,
        name: str,
        workflow_type: str,
        priority: TaskPriority = TaskPriority.MEDIUM,
        params: Optional[Dict[str, Any]] = None,
        dependencies: Optional[List[str]] = None,
    ) -> WorkflowTask:
        """직접 태스크 생성"""
        task_id = f"TASK-{uuid.uuid4().hex[:8]}"

        task = WorkflowTask(
            task_id=task_id,
            name=name,
            workflow_type=workflow_type,
            priority=priority,
            action_id="direct",
            dependencies=dependencies or [],
            params=params or {},
            timeout_seconds=self.default_timeout,
        )

        self.tasks[task_id] = task
        heapq.heappush(self.task_queue, task)
        self._stats["total_tasks"] += 1

        return task

    def create_workflow(
        self,
        workflow_type: str,
        tasks: List[Dict[str, Any]],
        trigger_source: str = "manual",
    ) -> WorkflowExecution:
        """
        워크플로우 생성 (여러 태스크 그룹)

        Args:
            workflow_type: 워크플로우 타입
            tasks: 태스크 정의 리스트
            trigger_source: 트리거 소스

        Returns:
            WorkflowExecution
        """
        execution_id = f"EXEC-{uuid.uuid4().hex[:8]}"
        task_ids = []

        prev_task_id = None
        for task_def in tasks:
            # 의존성 설정 (순차 실행)
            dependencies = []
            if task_def.get("sequential", True) and prev_task_id:
                dependencies = [prev_task_id]

            task = self.create_task(
                name=task_def.get("name", "Unnamed"),
                workflow_type=task_def.get("type", workflow_type),
                priority=TaskPriority[task_def.get("priority", "MEDIUM").upper()],
                params=task_def.get("params", {}),
                dependencies=dependencies,
            )
            task_ids.append(task.task_id)
            prev_task_id = task.task_id

        execution = WorkflowExecution(
            execution_id=execution_id,
            workflow_type=workflow_type,
            trigger_source=trigger_source,
            tasks=task_ids,
        )

        self.executions[execution_id] = execution
        self._stats["total_executions"] += 1

        logger.info(f"Created workflow {execution_id} with {len(task_ids)} tasks")
        return execution

    async def run_scheduler(self):
        """
        스케줄러 메인 루프

        우선순위 기반으로 태스크를 스케줄링하고 실행
        """
        self.running = True
        running_tasks: Dict[str, asyncio.Task] = {}

        logger.info(f"Scheduler started (max_concurrent={self.max_concurrent_tasks})")

        while self.running or running_tasks:
            # 완료된 태스크 정리
            completed = [tid for tid, t in running_tasks.items() if t.done()]
            for tid in completed:
                async_task = running_tasks.pop(tid)
                try:
                    await async_task  # 예외 확인
                except Exception as e:
                    logger.error(f"Task {tid} raised exception: {e}")

            # 새 태스크 시작
            while (
                len(running_tasks) < self.max_concurrent_tasks
                and self.task_queue
                and self.running
            ):
                # 큐에서 최고 우선순위 태스크 peek
                if not self.task_queue:
                    break

                task = self.task_queue[0]

                # 의존성 확인
                if not self._check_dependencies(task):
                    # 의존성 미충족 - 큐 재정렬
                    heapq.heappop(self.task_queue)
                    task.status = TaskStatus.BLOCKED
                    heapq.heappush(self.task_queue, task)
                    break

                # 태스크 시작
                heapq.heappop(self.task_queue)
                task.status = TaskStatus.SCHEDULED
                task.scheduled_at = datetime.now()

                async_task = asyncio.create_task(
                    self._execute_task(task),
                    name=task.task_id,
                )
                running_tasks[task.task_id] = async_task

            await asyncio.sleep(0.01)

        logger.info("Scheduler stopped")

    async def _execute_task(self, task: WorkflowTask):
        """태스크 실행"""
        task.status = TaskStatus.RUNNING
        task.started_at = datetime.now()

        logger.info(f"Executing task {task.task_id}: {task.name}")

        try:
            # v18.0: timeout_seconds가 0이면 무제한 대기
            if task.timeout_seconds and task.timeout_seconds > 0:
                result = await asyncio.wait_for(
                    self.executor.execute(task),
                    timeout=task.timeout_seconds,
                )
            else:
                result = await self.executor.execute(task)  # 무제한 대기

            task.result = result
            task.status = TaskStatus.COMPLETED
            self._stats["completed_tasks"] += 1

            logger.info(f"Task {task.task_id} completed successfully")

        except asyncio.TimeoutError:
            task.error = f"Task timed out after {task.timeout_seconds}s"
            task.status = TaskStatus.FAILED
            self._stats["failed_tasks"] += 1
            logger.error(f"Task {task.task_id} timed out")

        except Exception as e:
            task.error = str(e)

            # 재시도 확인
            if task.retry_count < task.max_retries:
                task.retry_count += 1
                task.status = TaskStatus.RETRYING
                heapq.heappush(self.task_queue, task)
                logger.warning(f"Task {task.task_id} failed, retry {task.retry_count}/{task.max_retries}")
            else:
                task.status = TaskStatus.FAILED
                self._stats["failed_tasks"] += 1
                logger.error(f"Task {task.task_id} failed permanently: {e}")

        finally:
            task.completed_at = datetime.now()

            # 워크플로우 상태 업데이트
            self._update_workflow_status(task)

    def _check_dependencies(self, task: WorkflowTask) -> bool:
        """의존성 확인"""
        for dep_id in task.dependencies:
            dep_task = self.tasks.get(dep_id)
            if not dep_task:
                logger.warning(f"Dependency {dep_id} not found for task {task.task_id}")
                return False
            if dep_task.status != TaskStatus.COMPLETED:
                return False
        return True

    def _update_workflow_status(self, task: WorkflowTask):
        """워크플로우 실행 상태 업데이트"""
        for execution in self.executions.values():
            if task.task_id in execution.tasks:
                # 모든 태스크 상태 확인
                all_completed = True
                any_failed = False

                for tid in execution.tasks:
                    t = self.tasks.get(tid)
                    if not t:
                        continue
                    if t.status == TaskStatus.FAILED:
                        any_failed = True
                    if not t.is_terminal:
                        all_completed = False

                if all_completed:
                    execution.completed_at = datetime.now()
                    execution.status = "failed" if any_failed else "completed"

                    # 메트릭 계산
                    execution.metrics = self._calculate_execution_metrics(execution)

    def _calculate_execution_metrics(self, execution: WorkflowExecution) -> Dict[str, Any]:
        """워크플로우 실행 메트릭 계산"""
        tasks = [self.tasks[tid] for tid in execution.tasks if tid in self.tasks]

        total_duration = 0
        completed_count = 0
        failed_count = 0

        for task in tasks:
            if task.duration_seconds:
                total_duration += task.duration_seconds
            if task.status == TaskStatus.COMPLETED:
                completed_count += 1
            elif task.status == TaskStatus.FAILED:
                failed_count += 1

        return {
            "total_tasks": len(tasks),
            "completed_tasks": completed_count,
            "failed_tasks": failed_count,
            "total_duration_seconds": round(total_duration, 2),
            "success_rate": round(completed_count / len(tasks) * 100, 1) if tasks else 0,
        }

    def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """태스크 상태 조회"""
        task = self.tasks.get(task_id)
        if not task:
            return None
        return task.to_dict()

    def get_execution_status(self, execution_id: str) -> Optional[Dict[str, Any]]:
        """워크플로우 실행 상태 조회"""
        execution = self.executions.get(execution_id)
        if not execution:
            return None

        result = execution.to_dict()
        result["task_details"] = [
            self.tasks[tid].to_dict()
            for tid in execution.tasks
            if tid in self.tasks
        ]
        return result

    def get_pending_tasks(self) -> List[Dict[str, Any]]:
        """대기 중인 태스크 조회"""
        return [
            task.to_dict()
            for task in self.task_queue
            if task.status in [TaskStatus.PENDING, TaskStatus.BLOCKED]
        ]

    def cancel_task(self, task_id: str) -> bool:
        """태스크 취소"""
        task = self.tasks.get(task_id)
        if not task:
            return False

        if task.is_terminal:
            return False

        task.status = TaskStatus.CANCELLED
        task.completed_at = datetime.now()
        logger.info(f"Task {task_id} cancelled")
        return True

    def stop(self):
        """스케줄러 중지"""
        self.running = False

    def get_stats(self) -> Dict[str, Any]:
        """통계 조회"""
        return {
            **self._stats,
            "pending_tasks": len(self.task_queue),
            "running": self.running,
        }


# Factory function
def create_orchestrator(
    domain_context=None,
    max_concurrent_tasks: int = 5,
) -> WorkflowOrchestrator:
    """팩토리 함수"""
    return WorkflowOrchestrator(
        domain_context=domain_context,
        max_concurrent_tasks=max_concurrent_tasks,
    )
