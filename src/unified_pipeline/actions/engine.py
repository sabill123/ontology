"""
Actions Engine (v16.0)

Palantir Ontology 스타일의 자동화된 워크플로우 실행 엔진:
- Action 정의 및 등록
- Trigger 기반 자동 실행
- 권한 및 승인 체계
- 실행 이력 관리
- Rollback 지원

SharedContext와 연동하여 FK 관계 기반 Action 실행

사용법:
    from .engine import ActionsEngine, Action, Trigger
    engine = ActionsEngine()
    engine.register_action(action)
    engine.execute("action_id", params)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Callable, Union, Tuple, TYPE_CHECKING
from enum import Enum
from datetime import datetime
from collections import defaultdict
import uuid
import asyncio
import logging

if TYPE_CHECKING:
    from ..shared_context import SharedContext

logger = logging.getLogger(__name__)


class ActionStatus(Enum):
    """Action 실행 상태"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    ROLLED_BACK = "rolled_back"


class TriggerType(Enum):
    """Trigger 유형"""
    MANUAL = "manual"           # 수동 실행
    SCHEDULE = "schedule"       # 스케줄 기반
    EVENT = "event"             # 이벤트 기반
    FK_CHANGE = "fk_change"     # FK 관계 변경 시
    DATA_QUALITY = "data_quality"  # 데이터 품질 이슈 시
    THRESHOLD = "threshold"     # 임계값 초과 시


class ActionType(Enum):
    """Action 유형"""
    CREATE = "create"           # 데이터 생성
    UPDATE = "update"           # 데이터 수정
    DELETE = "delete"           # 데이터 삭제
    TRANSFORM = "transform"     # 데이터 변환
    VALIDATE = "validate"       # 검증
    NOTIFY = "notify"           # 알림
    SYNC = "sync"               # 동기화
    AGGREGATE = "aggregate"     # 집계
    CHAIN = "chain"             # 연쇄 Action


@dataclass
class ActionParameter:
    """Action 파라미터 정의"""
    name: str
    param_type: str  # string, number, boolean, array, object
    required: bool = True
    default: Any = None
    description: Optional[str] = None
    validation: Optional[Callable[[Any], bool]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "type": self.param_type,
            "required": self.required,
            "default": self.default,
            "description": self.description,
        }


@dataclass
class Trigger:
    """Action Trigger 정의"""
    trigger_id: str
    trigger_type: TriggerType
    condition: Optional[Callable[["ActionContext"], bool]] = None

    # Schedule 기반
    schedule: Optional[str] = None  # cron expression

    # Event 기반
    event_source: Optional[str] = None  # 테이블/엔티티명
    event_type: Optional[str] = None    # insert, update, delete

    # Threshold 기반
    metric: Optional[str] = None
    threshold_value: Optional[float] = None
    threshold_operator: Optional[str] = None  # gt, lt, eq, gte, lte

    is_enabled: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "trigger_id": self.trigger_id,
            "trigger_type": self.trigger_type.value,
            "schedule": self.schedule,
            "event_source": self.event_source,
            "event_type": self.event_type,
            "metric": self.metric,
            "threshold_value": self.threshold_value,
            "is_enabled": self.is_enabled,
        }


@dataclass
class ActionContext:
    """Action 실행 컨텍스트"""
    execution_id: str
    action_id: str
    parameters: Dict[str, Any]
    trigger: Optional[Trigger] = None

    # 실행 환경
    shared_context: Optional["SharedContext"] = None
    user_id: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)

    # 실행 중 데이터
    data: Dict[str, Any] = field(default_factory=dict)
    affected_entities: List[str] = field(default_factory=list)
    changes: List[Dict[str, Any]] = field(default_factory=list)

    # 결과
    status: ActionStatus = ActionStatus.PENDING
    result: Any = None
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "execution_id": self.execution_id,
            "action_id": self.action_id,
            "parameters": self.parameters,
            "status": self.status.value,
            "timestamp": self.timestamp.isoformat(),
            "affected_entities": self.affected_entities,
            "error": self.error,
        }


@dataclass
class Action:
    """Action 정의"""
    action_id: str
    name: str
    action_type: ActionType

    # 실행 로직
    handler: Callable[[ActionContext], Any]

    # 파라미터
    parameters: List[ActionParameter] = field(default_factory=list)

    # Trigger
    triggers: List[Trigger] = field(default_factory=list)

    # 메타데이터
    description: Optional[str] = None
    version: str = "1.0.0"
    author: Optional[str] = None

    # 권한
    required_permissions: List[str] = field(default_factory=list)
    requires_approval: bool = False
    approvers: List[str] = field(default_factory=list)

    # Rollback
    supports_rollback: bool = False
    rollback_handler: Optional[Callable[[ActionContext], Any]] = None

    # 연쇄 Action
    chained_actions: List[str] = field(default_factory=list)

    is_enabled: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action_id": self.action_id,
            "name": self.name,
            "action_type": self.action_type.value,
            "parameters": [p.to_dict() for p in self.parameters],
            "triggers": [t.to_dict() for t in self.triggers],
            "description": self.description,
            "version": self.version,
            "required_permissions": self.required_permissions,
            "requires_approval": self.requires_approval,
            "supports_rollback": self.supports_rollback,
            "is_enabled": self.is_enabled,
        }


@dataclass
class ExecutionRecord:
    """실행 이력 레코드"""
    execution_id: str
    action_id: str
    status: ActionStatus
    started_at: datetime
    completed_at: Optional[datetime] = None
    parameters: Dict[str, Any] = field(default_factory=dict)
    result: Any = None
    error: Optional[str] = None
    user_id: Optional[str] = None
    trigger_id: Optional[str] = None
    changes: List[Dict[str, Any]] = field(default_factory=list)
    rolled_back: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "execution_id": self.execution_id,
            "action_id": self.action_id,
            "status": self.status.value,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "parameters": self.parameters,
            "error": self.error,
            "user_id": self.user_id,
            "trigger_id": self.trigger_id,
            "rolled_back": self.rolled_back,
        }


class ActionsEngine:
    """
    Actions Engine

    기능:
    1. Action 등록 및 관리
    2. Trigger 기반 자동 실행
    3. 수동 실행
    4. 실행 이력 관리
    5. Rollback 지원
    6. 연쇄 Action 실행

    SharedContext 연동:
    - FK 변경 시 자동 Action 트리거
    - 데이터 품질 이슈 시 알림 Action
    - 온톨로지 업데이트 Action
    """

    def __init__(self, shared_context: Optional["SharedContext"] = None):
        self.shared_context = shared_context
        self.actions: Dict[str, Action] = {}
        self.execution_history: List[ExecutionRecord] = []
        self._pending_approvals: Dict[str, ActionContext] = {}
        self._event_handlers: Dict[str, List[str]] = defaultdict(list)  # event -> action_ids

    def register_action(self, action: Action) -> None:
        """Action 등록"""
        self.actions[action.action_id] = action

        # 이벤트 기반 Trigger 등록
        for trigger in action.triggers:
            if trigger.trigger_type == TriggerType.EVENT and trigger.event_source:
                event_key = f"{trigger.event_source}:{trigger.event_type}"
                self._event_handlers[event_key].append(action.action_id)

        logger.info(f"[ActionsEngine] Registered action: {action.action_id}")

    def unregister_action(self, action_id: str) -> bool:
        """Action 등록 해제"""
        if action_id in self.actions:
            del self.actions[action_id]
            return True
        return False

    def execute(
        self,
        action_id: str,
        parameters: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None,
        trigger: Optional[Trigger] = None,
    ) -> ActionContext:
        """
        Action 실행

        Args:
            action_id: 실행할 Action ID
            parameters: 실행 파라미터
            user_id: 실행 사용자
            trigger: 트리거 (자동 실행 시)

        Returns:
            ActionContext
        """
        if action_id not in self.actions:
            raise ValueError(f"Action not found: {action_id}")

        action = self.actions[action_id]

        if not action.is_enabled:
            raise ValueError(f"Action is disabled: {action_id}")

        # 파라미터 검증 및 기본값 적용
        validated_params = self._validate_parameters(action, parameters or {})

        # 실행 컨텍스트 생성
        context = ActionContext(
            execution_id=str(uuid.uuid4()),
            action_id=action_id,
            parameters=validated_params,
            trigger=trigger,
            shared_context=self.shared_context,
            user_id=user_id,
        )

        # 승인 필요 확인
        if action.requires_approval:
            self._pending_approvals[context.execution_id] = context
            context.status = ActionStatus.PENDING
            logger.info(f"[ActionsEngine] Action {action_id} pending approval")
            return context

        # 실행
        return self._execute_action(action, context)

    async def execute_async(
        self,
        action_id: str,
        parameters: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None,
    ) -> ActionContext:
        """비동기 Action 실행"""
        return await asyncio.get_event_loop().run_in_executor(
            None, lambda: self.execute(action_id, parameters, user_id)
        )

    def _execute_action(
        self,
        action: Action,
        context: ActionContext,
    ) -> ActionContext:
        """실제 Action 실행"""
        # 실행 이력 시작
        record = ExecutionRecord(
            execution_id=context.execution_id,
            action_id=action.action_id,
            status=ActionStatus.RUNNING,
            started_at=datetime.now(),
            parameters=context.parameters,
            user_id=context.user_id,
            trigger_id=context.trigger.trigger_id if context.trigger else None,
        )

        context.status = ActionStatus.RUNNING

        try:
            # Handler 실행
            result = action.handler(context)
            context.result = result
            context.status = ActionStatus.COMPLETED
            record.status = ActionStatus.COMPLETED
            record.result = result

            logger.info(f"[ActionsEngine] Action {action.action_id} completed successfully")

            # 연쇄 Action 실행
            if action.chained_actions:
                for chained_id in action.chained_actions:
                    if chained_id in self.actions:
                        self.execute(
                            chained_id,
                            parameters={"parent_result": result},
                            user_id=context.user_id,
                        )

        except Exception as e:
            context.status = ActionStatus.FAILED
            context.error = str(e)
            record.status = ActionStatus.FAILED
            record.error = str(e)

            logger.error(f"[ActionsEngine] Action {action.action_id} failed: {e}")

        finally:
            record.completed_at = datetime.now()
            record.changes = context.changes
            self.execution_history.append(record)

        return context

    def _validate_parameters(
        self,
        action: Action,
        params: Dict[str, Any],
    ) -> Dict[str, Any]:
        """파라미터 검증 및 기본값 적용"""
        validated = {}

        for param_def in action.parameters:
            name = param_def.name

            if name in params:
                value = params[name]
                # 사용자 정의 검증
                if param_def.validation and not param_def.validation(value):
                    raise ValueError(f"Invalid value for parameter {name}")
                validated[name] = value
            elif param_def.default is not None:
                validated[name] = param_def.default
            elif param_def.required:
                raise ValueError(f"Missing required parameter: {name}")

        return validated

    def approve(
        self,
        execution_id: str,
        approver_id: str,
    ) -> ActionContext:
        """승인 대기 중인 Action 승인"""
        if execution_id not in self._pending_approvals:
            raise ValueError(f"No pending approval: {execution_id}")

        context = self._pending_approvals.pop(execution_id)
        action = self.actions[context.action_id]

        # 승인자 확인
        if action.approvers and approver_id not in action.approvers:
            raise PermissionError(f"User {approver_id} not authorized to approve")

        return self._execute_action(action, context)

    def reject(
        self,
        execution_id: str,
        reason: Optional[str] = None,
    ) -> None:
        """승인 대기 중인 Action 거절"""
        if execution_id in self._pending_approvals:
            context = self._pending_approvals.pop(execution_id)
            context.status = ActionStatus.CANCELLED
            context.error = f"Rejected: {reason}" if reason else "Rejected"

            self.execution_history.append(ExecutionRecord(
                execution_id=context.execution_id,
                action_id=context.action_id,
                status=ActionStatus.CANCELLED,
                started_at=context.timestamp,
                completed_at=datetime.now(),
                error=context.error,
            ))

    def rollback(
        self,
        execution_id: str,
        user_id: Optional[str] = None,
    ) -> bool:
        """실행된 Action Rollback"""
        # 실행 이력 찾기
        record = None
        for r in reversed(self.execution_history):
            if r.execution_id == execution_id:
                record = r
                break

        if not record:
            raise ValueError(f"Execution not found: {execution_id}")

        if record.status != ActionStatus.COMPLETED:
            raise ValueError(f"Cannot rollback non-completed execution")

        if record.rolled_back:
            raise ValueError(f"Already rolled back: {execution_id}")

        action = self.actions.get(record.action_id)
        if not action or not action.supports_rollback or not action.rollback_handler:
            raise ValueError(f"Action does not support rollback: {record.action_id}")

        # Rollback 컨텍스트 생성
        rollback_context = ActionContext(
            execution_id=str(uuid.uuid4()),
            action_id=f"{record.action_id}:rollback",
            parameters=record.parameters,
            shared_context=self.shared_context,
            user_id=user_id,
        )
        rollback_context.data["original_changes"] = record.changes

        try:
            action.rollback_handler(rollback_context)
            record.rolled_back = True
            record.status = ActionStatus.ROLLED_BACK
            logger.info(f"[ActionsEngine] Rolled back execution: {execution_id}")
            return True
        except Exception as e:
            logger.error(f"[ActionsEngine] Rollback failed: {e}")
            return False

    def on_event(
        self,
        event_source: str,
        event_type: str,
        event_data: Dict[str, Any],
    ) -> List[ActionContext]:
        """이벤트 발생 시 등록된 Action 트리거"""
        event_key = f"{event_source}:{event_type}"
        results = []

        for action_id in self._event_handlers.get(event_key, []):
            action = self.actions.get(action_id)
            if action and action.is_enabled:
                context = self.execute(
                    action_id,
                    parameters={"event_data": event_data},
                    trigger=Trigger(
                        trigger_id=f"event:{event_key}",
                        trigger_type=TriggerType.EVENT,
                        event_source=event_source,
                        event_type=event_type,
                    ),
                )
                results.append(context)

        return results

    def get_action(self, action_id: str) -> Optional[Action]:
        """Action 조회"""
        return self.actions.get(action_id)

    def list_actions(self) -> List[Dict[str, Any]]:
        """등록된 모든 Action 목록"""
        return [a.to_dict() for a in self.actions.values()]

    def get_history(
        self,
        action_id: Optional[str] = None,
        status: Optional[ActionStatus] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """실행 이력 조회"""
        history = self.execution_history

        if action_id:
            history = [r for r in history if r.action_id == action_id]
        if status:
            history = [r for r in history if r.status == status]

        return [r.to_dict() for r in history[-limit:]]


# 빌트인 Action 헬퍼
def create_validation_action(
    action_id: str,
    validation_fn: Callable[[ActionContext], List[Dict[str, Any]]],
    name: Optional[str] = None,
) -> Action:
    """검증 Action 생성 헬퍼"""
    def handler(ctx: ActionContext) -> Dict[str, Any]:
        violations = validation_fn(ctx)
        return {
            "violations_count": len(violations),
            "violations": violations,
            "passed": len(violations) == 0,
        }

    return Action(
        action_id=action_id,
        name=name or f"Validate: {action_id}",
        action_type=ActionType.VALIDATE,
        handler=handler,
    )


def create_notification_action(
    action_id: str,
    notification_fn: Callable[[str, Dict[str, Any]], None],
    name: Optional[str] = None,
) -> Action:
    """알림 Action 생성 헬퍼"""
    def handler(ctx: ActionContext) -> Dict[str, Any]:
        message = ctx.parameters.get("message", "")
        recipients = ctx.parameters.get("recipients", [])

        for recipient in recipients:
            notification_fn(recipient, {"message": message, "context": ctx.to_dict()})

        return {"notified": len(recipients)}

    return Action(
        action_id=action_id,
        name=name or f"Notify: {action_id}",
        action_type=ActionType.NOTIFY,
        handler=handler,
        parameters=[
            ActionParameter("message", "string", required=True),
            ActionParameter("recipients", "array", required=False, default=[]),
        ],
    )


def create_sync_action(
    action_id: str,
    source: str,
    target: str,
    sync_fn: Callable[[str, str, ActionContext], int],
    name: Optional[str] = None,
) -> Action:
    """동기화 Action 생성 헬퍼"""
    def handler(ctx: ActionContext) -> Dict[str, Any]:
        synced_count = sync_fn(source, target, ctx)
        return {
            "source": source,
            "target": target,
            "synced_records": synced_count,
        }

    return Action(
        action_id=action_id,
        name=name or f"Sync: {source} -> {target}",
        action_type=ActionType.SYNC,
        handler=handler,
    )


# SharedContext 연동
def create_fk_validation_action(shared_context: "SharedContext") -> Action:
    """FK 참조 무결성 검증 Action"""
    def handler(ctx: ActionContext) -> Dict[str, Any]:
        violations = []

        if not ctx.shared_context:
            return {"error": "No shared context"}

        # FK 후보에 대해 참조 무결성 검사
        for fk in ctx.shared_context.enhanced_fk_candidates:
            if isinstance(fk, dict):
                from_table = fk.get("from_table")
                from_col = fk.get("from_column")
                to_table = fk.get("to_table")
                to_col = fk.get("to_column")

                # 간단한 검증 로직 (실제로는 더 정교하게)
                if fk.get("confidence", "low") == "low":
                    violations.append({
                        "type": "low_confidence_fk",
                        "fk": f"{from_table}.{from_col} -> {to_table}.{to_col}",
                        "confidence": fk.get("fk_score", 0),
                    })

        return {
            "total_fks": len(ctx.shared_context.enhanced_fk_candidates),
            "violations": violations,
            "passed": len(violations) == 0,
        }

    return Action(
        action_id="validate_fk_integrity",
        name="Validate FK Integrity",
        action_type=ActionType.VALIDATE,
        handler=handler,
        description="Validates foreign key referential integrity",
    )


# Module initialization
__all__ = [
    "ActionsEngine",
    "Action",
    "ActionContext",
    "ActionParameter",
    "ActionStatus",
    "ActionType",
    "Trigger",
    "TriggerType",
    "create_validation_action",
    "create_notification_action",
    "create_sync_action",
    "create_fk_validation_action",
]
