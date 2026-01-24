"""
Todo System Core Models

Todo 시스템 핵심 데이터 모델
- PipelineTodo: 개별 Todo 아이템
- TodoStatus: Todo 상태
- TodoResult: Todo 실행 결과
- TodoEvent: 실시간 이벤트
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Any, Optional, Set
import uuid
import logging

logger = logging.getLogger(__name__)


class TodoStatus(str, Enum):
    """Todo 상태"""
    PENDING = "pending"           # 대기 중 (의존성 충족 대기)
    READY = "ready"               # 준비됨 (의존성 충족, 실행 가능)
    IN_PROGRESS = "in_progress"   # 진행 중 (에이전트가 작업 중)
    COMPLETED = "completed"       # 완료됨
    FAILED = "failed"             # 실패함
    BLOCKED = "blocked"           # 차단됨 (의존성 실패로 인해)
    CANCELLED = "cancelled"       # 취소됨


class TodoPriority(int, Enum):
    """Todo 우선순위"""
    CRITICAL = 0    # 즉시 처리 필요
    HIGH = 1        # 높은 우선순위
    MEDIUM = 2      # 보통 우선순위
    LOW = 3         # 낮은 우선순위


class TodoEventType(str, Enum):
    """Todo 이벤트 타입"""
    # Todo 라이프사이클
    TODO_CREATED = "todo_created"
    TODO_READY = "todo_ready"
    TODO_STARTED = "todo_started"
    TODO_PROGRESS = "todo_progress"
    TODO_COMPLETED = "todo_completed"
    TODO_FAILED = "todo_failed"
    TODO_BLOCKED = "todo_blocked"
    TODO_CANCELLED = "todo_cancelled"

    # 에이전트 관련
    AGENT_ASSIGNED = "agent_assigned"
    AGENT_STARTED = "agent_started"
    AGENT_THINKING = "agent_thinking"
    AGENT_OUTPUT = "agent_output"
    AGENT_COMPLETED = "agent_completed"
    AGENT_ERROR = "agent_error"

    # 파이프라인 관련
    PHASE_STARTED = "phase_started"
    PHASE_COMPLETED = "phase_completed"
    PIPELINE_STARTED = "pipeline_started"
    PIPELINE_COMPLETED = "pipeline_completed"
    PIPELINE_ERROR = "pipeline_error"

    # 시스템 관련
    DEPENDENCY_RESOLVED = "dependency_resolved"
    CONTINUATION_TRIGGERED = "continuation_triggered"


@dataclass
class TodoEvent:
    """실시간 Todo 이벤트"""
    event_id: str
    event_type: TodoEventType
    timestamp: str

    # 이벤트 대상
    todo_id: Optional[str] = None
    agent_id: Optional[str] = None
    phase: Optional[str] = None

    # 이벤트 데이터
    data: Dict[str, Any] = field(default_factory=dict)
    message: str = ""

    # 진행률 (0.0 ~ 1.0)
    progress: Optional[float] = None

    # v14.0: 종료 모드 및 완료 시그널 (이벤트 표준화)
    terminate_mode: Optional[str] = None  # AgentTerminateMode 값
    completion_signal: Optional[str] = None  # CompletionSignal 값

    @classmethod
    def create(
        cls,
        event_type: TodoEventType,
        message: str = "",
        todo_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        phase: Optional[str] = None,
        data: Optional[Dict[str, Any]] = None,
        progress: Optional[float] = None,
        terminate_mode: Optional[str] = None,  # v14.0
        completion_signal: Optional[str] = None,  # v14.0
    ) -> "TodoEvent":
        """이벤트 생성 팩토리"""
        return cls(
            event_id=str(uuid.uuid4())[:8],
            event_type=event_type,
            timestamp=datetime.now().isoformat(),
            todo_id=todo_id,
            agent_id=agent_id,
            phase=phase,
            data=data or {},
            message=message,
            progress=progress,
            terminate_mode=terminate_mode,
            completion_signal=completion_signal,
        )

    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리 변환 (FE 전송용)"""
        result = {
            "event_id": self.event_id,
            "event_type": self.event_type.value,
            "timestamp": self.timestamp,
            "todo_id": self.todo_id,
            "agent_id": self.agent_id,
            "phase": self.phase,
            "data": self.data,
            "message": self.message,
            "progress": self.progress,
        }
        # v14.0: 종료 모드 및 완료 시그널 포함 (설정된 경우만)
        if self.terminate_mode:
            result["terminate_mode"] = self.terminate_mode
        if self.completion_signal:
            result["completion_signal"] = self.completion_signal
        return result


@dataclass
class TodoResult:
    """Todo 실행 결과"""
    success: bool
    output: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None

    # 실행 정보
    execution_time_seconds: float = 0.0
    agent_id: Optional[str] = None

    # v14.0: 종료 모드 - 종료 "이유"를 명확히 기록 (Silent Failure 탐지용)
    terminate_mode: Optional[str] = None  # AgentTerminateMode 값

    # v14.0: 명시적 완료 시그널 - 분석 상태 표시 (DONE, NEEDS_MORE, BLOCKED, ESCALATE)
    completion_signal: Optional[str] = None  # CompletionSignal 값

    # 컨텍스트 업데이트 (SharedContext에 반영할 데이터)
    context_updates: Dict[str, Any] = field(default_factory=dict)

    # 생성된 후속 Todo들
    spawned_todos: List[str] = field(default_factory=list)

    # 메타데이터
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리 변환"""
        return {
            "success": self.success,
            "output": self.output,
            "error": self.error,
            "execution_time_seconds": self.execution_time_seconds,
            "agent_id": self.agent_id,
            "terminate_mode": self.terminate_mode,
            "completion_signal": self.completion_signal,  # v14.0
            "context_updates": self.context_updates,
            "spawned_todos": self.spawned_todos,
            "metadata": self.metadata,
        }


@dataclass
class PipelineTodo:
    """
    파이프라인 Todo 아이템

    핵심 특징:
    - 고유 ID로 식별
    - 의존성 기반 실행 순서 (DAG)
    - 특정 에이전트 타입 지정 가능
    - 실시간 진행률 추적
    """

    # 식별자
    todo_id: str
    name: str
    description: str

    # 소속 정보
    phase: str  # "discovery", "refinement", "governance"
    task_type: str  # "tda_analysis", "schema_analysis", "value_matching", etc.

    # 의존성 (DAG 구조)
    dependencies: Set[str] = field(default_factory=set)  # 선행 todo_id들
    dependents: Set[str] = field(default_factory=set)    # 후행 todo_id들 (자동 계산)

    # 에이전트 할당
    required_agent_type: Optional[str] = None  # 특정 에이전트 타입 지정
    assigned_agent_id: Optional[str] = None    # 실제 할당된 에이전트

    # 상태
    status: TodoStatus = TodoStatus.PENDING
    priority: TodoPriority = TodoPriority.MEDIUM

    # 진행률
    progress: float = 0.0  # 0.0 ~ 1.0

    # 입력/출력
    input_data: Dict[str, Any] = field(default_factory=dict)
    result: Optional[TodoResult] = None

    # 타임스탬프
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    started_at: Optional[str] = None
    completed_at: Optional[str] = None

    # 재시도
    retry_count: int = 0
    max_retries: int = 3

    # 메타데이터
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        name: str,
        description: str,
        phase: str,
        task_type: str,
        dependencies: Optional[Set[str]] = None,
        required_agent_type: Optional[str] = None,
        priority: TodoPriority = TodoPriority.MEDIUM,
        input_data: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "PipelineTodo":
        """Todo 생성 팩토리"""
        return cls(
            todo_id=f"todo_{uuid.uuid4().hex[:12]}",
            name=name,
            description=description,
            phase=phase,
            task_type=task_type,
            dependencies=dependencies or set(),
            required_agent_type=required_agent_type,
            priority=priority,
            input_data=input_data or {},
            metadata=metadata or {},
        )

    def is_ready(self, completed_todos: Set[str]) -> bool:
        """실행 가능 여부 확인 (모든 의존성 완료)"""
        return self.dependencies.issubset(completed_todos)

    def can_retry(self) -> bool:
        """재시도 가능 여부"""
        return self.retry_count < self.max_retries

    def start(self, agent_id: str) -> None:
        """Todo 시작"""
        self.status = TodoStatus.IN_PROGRESS
        self.assigned_agent_id = agent_id
        self.started_at = datetime.now().isoformat()
        self.progress = 0.0
        logger.info(f"Todo started: {self.todo_id} by agent {agent_id}")

    def update_progress(self, progress: float, message: str = "") -> None:
        """진행률 업데이트"""
        self.progress = max(0.0, min(1.0, progress))
        logger.debug(f"Todo progress: {self.todo_id} - {self.progress:.1%} {message}")

    def complete(self, result: TodoResult) -> None:
        """Todo 완료"""
        self.status = TodoStatus.COMPLETED
        self.result = result
        self.progress = 1.0
        self.completed_at = datetime.now().isoformat()
        logger.info(f"Todo completed: {self.todo_id} in {result.execution_time_seconds:.2f}s")

    def fail(self, error: str) -> None:
        """Todo 실패"""
        self.retry_count += 1
        if self.can_retry():
            self.status = TodoStatus.PENDING
            logger.warning(f"Todo failed (retry {self.retry_count}): {self.todo_id} - {error}")
        else:
            self.status = TodoStatus.FAILED
            self.result = TodoResult(success=False, error=error)
            self.completed_at = datetime.now().isoformat()
            logger.error(f"Todo failed permanently: {self.todo_id} - {error}")

    def block(self, reason: str) -> None:
        """Todo 차단 (의존성 실패)"""
        self.status = TodoStatus.BLOCKED
        self.metadata["block_reason"] = reason
        logger.warning(f"Todo blocked: {self.todo_id} - {reason}")

    def cancel(self, reason: str = "") -> None:
        """Todo 취소"""
        self.status = TodoStatus.CANCELLED
        self.metadata["cancel_reason"] = reason
        self.completed_at = datetime.now().isoformat()
        logger.info(f"Todo cancelled: {self.todo_id} - {reason}")

    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리 변환 (FE 전송용)"""
        return {
            "todo_id": self.todo_id,
            "name": self.name,
            "description": self.description,
            "phase": self.phase,
            "task_type": self.task_type,
            "dependencies": list(self.dependencies),
            "dependents": list(self.dependents),
            "required_agent_type": self.required_agent_type,
            "assigned_agent_id": self.assigned_agent_id,
            "status": self.status.value,
            "priority": self.priority.value,
            "progress": self.progress,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "retry_count": self.retry_count,
            "result": self.result.to_dict() if self.result else None,
        }


# 미리 정의된 Todo 템플릿들 (Phase별)
DISCOVERY_TODO_TEMPLATES = [
    {
        "name": "Data Understanding",
        "task_type": "data_understanding",
        "required_agent_type": "data_analyst",
        "description": "LLM이 전체 데이터셋을 직접 읽고 분석: 도메인 파악, 컬럼 의미, 데이터 패턴, 품질 이슈 탐지",
    },
    {
        "name": "TDA Analysis",
        "task_type": "tda_analysis",
        "required_agent_type": "tda_expert",
        "description": "Compute TDA signatures and Betti numbers for all tables",
        "dependencies": ["data_understanding"],
    },
    {
        "name": "Schema Analysis",
        "task_type": "schema_analysis",
        "required_agent_type": "schema_analyst",
        "description": "Analyze schema structures and identify potential mappings",
        "dependencies": ["tda_analysis"],
    },
    {
        "name": "Value Overlap Analysis",
        "task_type": "value_matching",
        "required_agent_type": "value_matcher",
        "description": "Detect value overlaps between tables",
        "dependencies": ["schema_analysis"],
    },
    {
        "name": "Homeomorphism Detection",
        "task_type": "homeomorphism_detection",
        "required_agent_type": "tda_expert",
        "description": "Identify homeomorphic relationships between tables",
        "dependencies": ["tda_analysis", "schema_analysis", "value_matching"],
    },
    {
        "name": "Entity Classification",
        "task_type": "entity_classification",
        "required_agent_type": "entity_classifier",
        "description": "Classify and unify entities across tables",
        "dependencies": ["homeomorphism_detection"],
    },
    {
        "name": "Relationship Detection",
        "task_type": "relationship_detection",
        "required_agent_type": "relationship_detector",
        "description": "Detect relationships between unified entities",
        "dependencies": ["entity_classification"],
    },
]

REFINEMENT_TODO_TEMPLATES = [
    {
        "name": "Ontology Proposal",
        "task_type": "ontology_proposal",
        "required_agent_type": "ontology_architect",
        "description": "Propose ontology concepts based on discovery results",
    },
    {
        "name": "Conflict Detection",
        "task_type": "conflict_detection",
        "required_agent_type": "conflict_resolver",
        "description": "Detect conflicts in proposed ontology",
        "dependencies": ["ontology_proposal"],
    },
    {
        "name": "Conflict Resolution",
        "task_type": "conflict_resolution",
        "required_agent_type": "conflict_resolver",
        "description": "Resolve detected conflicts",
        "dependencies": ["conflict_detection"],
    },
    {
        "name": "Quality Assessment",
        "task_type": "quality_assessment",
        "required_agent_type": "quality_judge",
        "description": "Assess quality of ontology concepts",
        "dependencies": ["conflict_resolution"],
    },
    {
        "name": "Semantic Validation",
        "task_type": "semantic_validation",
        "required_agent_type": "semantic_validator",
        "description": "Validate semantic consistency",
        "dependencies": ["quality_assessment"],
    },
]

GOVERNANCE_TODO_TEMPLATES = [
    {
        "name": "Governance Strategy",
        "task_type": "governance_strategy",
        "required_agent_type": "governance_strategist",
        "description": "Define governance strategy for ontology",
    },
    {
        "name": "Action Prioritization",
        "task_type": "action_prioritization",
        "required_agent_type": "action_prioritizer",
        "description": "Prioritize implementation actions",
        "dependencies": ["governance_strategy"],
    },
    {
        "name": "Risk Assessment",
        "task_type": "risk_assessment",
        "required_agent_type": "risk_assessor",
        "description": "Assess risks of proposed changes",
        "dependencies": ["action_prioritization"],
    },
    {
        "name": "Policy Generation",
        "task_type": "policy_generation",
        "required_agent_type": "policy_generator",
        "description": "Generate governance policies",
        "dependencies": ["risk_assessment"],
    },
]
