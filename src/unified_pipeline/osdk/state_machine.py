"""
Entity State Machine (v2.0)

팔란티어 스타일 상태 전이 시스템

팔란티어 검증 기준:
✅ 상태 전이가 명시적이다 (Explicit state transitions)
✅ 상태에 따라 가능한 행동이 제한된다 (State-dependent behaviors)
✅ 워크플로우가 데이터 흐름이 아닌 개념 상태 전이

핵심 개념:
1. StateDefinition: 상태 정의 (DRAFT, PENDING, APPROVED 등)
2. TransitionDefinition: 전이 정의 (DRAFT → PENDING, guard 조건 포함)
3. StateMachine: 상태 머신 (상태 + 전이 집합)
4. StateMachineExecutor: 전이 실행기

사용 예시:
    # 상태 머신 정의
    order_sm = StateMachine(
        machine_id="order_state_machine",
        object_type_id="Order",
        state_property="status"
    )

    # 상태 추가
    order_sm.add_state(StateDefinition(
        state_id="draft",
        name="Draft",
        is_initial=True,
        allowed_behaviors=["edit", "delete", "submit"]
    ))

    # 전이 추가
    order_sm.add_transition(TransitionDefinition(
        transition_id="submit",
        from_state="draft",
        to_state="pending",
        guard_condition="self.get('total') > 0"
    ))

    # 전이 실행
    executor = StateMachineExecutor(behavior_executor)
    result = await executor.transition(order, order_sm, "submit")
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, TYPE_CHECKING
from datetime import datetime
from enum import Enum
import logging

if TYPE_CHECKING:
    from .models import ObjectInstance
    from .executor import BehaviorExecutor
    from ..evidence_chain import EvidenceChain

logger = logging.getLogger(__name__)


@dataclass
class StateDefinition:
    """
    상태 정의

    팔란티어 검증 기준:
    - 각 상태는 명시적인 의미를 가짐
    - 상태별로 허용/금지된 행동이 정의됨
    - 상태 진입/이탈 시 트리거 실행
    """
    state_id: str
    name: str
    description: str = ""

    # 상태 특성
    is_initial: bool = False        # 초기 상태
    is_terminal: bool = False       # 종료 상태 (더 이상 전이 불가)

    # 상태별 허용 행동 (Behavior Layer 연동)
    allowed_behaviors: List[str] = field(default_factory=list)    # 허용된 행동만 실행 가능
    forbidden_behaviors: List[str] = field(default_factory=list)  # 금지된 행동

    # 상태 진입/이탈 시 트리거 (ActionsEngine 연동)
    on_enter_triggers: List[str] = field(default_factory=list)  # 진입 시 실행할 action_id
    on_exit_triggers: List[str] = field(default_factory=list)   # 이탈 시 실행할 action_id

    # 상태별 속성 제약 (SHACL 스타일)
    property_constraints: Dict[str, str] = field(default_factory=dict)
    # 예: {"total": "value > 0", "status": "value == 'active'"}

    # 상태 유지 시간 제한 (초) - SLA 관리용
    max_duration_seconds: Optional[int] = None

    # UI 메타데이터
    color: Optional[str] = None  # 예: "#4CAF50" (green)
    icon: Optional[str] = None   # 예: "check_circle"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "state_id": self.state_id,
            "name": self.name,
            "description": self.description,
            "is_initial": self.is_initial,
            "is_terminal": self.is_terminal,
            "allowed_behaviors": self.allowed_behaviors,
            "forbidden_behaviors": self.forbidden_behaviors,
            "on_enter_triggers": self.on_enter_triggers,
            "on_exit_triggers": self.on_exit_triggers,
            "max_duration_seconds": self.max_duration_seconds,
            "color": self.color,
            "icon": self.icon,
        }


@dataclass
class TransitionDefinition:
    """
    상태 전이 정의

    팔란티어 검증 기준:
    - 전이 조건(guard)이 명시적
    - 전이 시 실행할 액션 정의
    - 권한 기반 전이 제어
    """
    transition_id: str
    name: str
    from_state: str
    to_state: str

    # 전이 조건 (Python 표현식)
    guard_condition: Optional[str] = None
    # 예: "self.get('risk_score') < 0.7 and self.get('total') > 0"

    # 전이 트리거 행동 (이 행동 호출 시 전이 발생)
    trigger_behavior: Optional[str] = None
    # 예: "approve" 행동 성공 시 자동 전이

    # 전이 시 실행할 액션 (ActionsEngine 연동)
    on_transition_triggers: List[str] = field(default_factory=list)

    # 권한
    required_permissions: List[str] = field(default_factory=list)
    allowed_roles: List[str] = field(default_factory=list)  # 예: ["manager", "admin"]

    # 자동 전이 여부
    is_automatic: bool = False
    # True면 guard_condition 충족 시 자동 전이

    # 우선순위 (자동 전이 시 여러 개 가능할 때)
    priority: int = 0

    # 확인 필요 여부 (Human-in-the-loop)
    requires_confirmation: bool = False

    # 설명
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "transition_id": self.transition_id,
            "name": self.name,
            "from_state": self.from_state,
            "to_state": self.to_state,
            "guard_condition": self.guard_condition,
            "trigger_behavior": self.trigger_behavior,
            "on_transition_triggers": self.on_transition_triggers,
            "required_permissions": self.required_permissions,
            "is_automatic": self.is_automatic,
            "requires_confirmation": self.requires_confirmation,
            "description": self.description,
        }


class TransitionResultStatus(str, Enum):
    """전이 결과 상태"""
    SUCCESS = "success"
    FAILED = "failed"
    PENDING_CONFIRMATION = "pending_confirmation"
    GUARD_FAILED = "guard_failed"
    INVALID_STATE = "invalid_state"
    PERMISSION_DENIED = "permission_denied"


@dataclass
class TransitionResult:
    """전이 결과"""
    success: bool
    status: TransitionResultStatus
    from_state: str
    to_state: Optional[str] = None
    error: Optional[str] = None
    triggered_actions: List[str] = field(default_factory=list)
    transitioned_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "status": self.status.value,
            "from_state": self.from_state,
            "to_state": self.to_state,
            "error": self.error,
            "triggered_actions": self.triggered_actions,
            "transitioned_at": self.transitioned_at.isoformat(),
        }


@dataclass
class StateHistory:
    """상태 이력"""
    from_state: str
    to_state: str
    transition_id: str
    transitioned_at: datetime
    transitioned_by: Optional[str] = None  # agent_id 또는 user_id
    params: Dict[str, Any] = field(default_factory=dict)
    reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "from_state": self.from_state,
            "to_state": self.to_state,
            "transition_id": self.transition_id,
            "transitioned_at": self.transitioned_at.isoformat(),
            "transitioned_by": self.transitioned_by,
            "params": self.params,
            "reason": self.reason,
        }


@dataclass
class StateMachine:
    """
    엔티티 상태 머신

    팔란티어 검증 기준:
    ✅ 업무 흐름이 개념 상태 전이로 표현됨
    ✅ 상태에 따라 가능한 행동이 제한됨
    ✅ 전이 조건이 명시적

    예시: Order 상태 머신
    DRAFT → PENDING → APPROVED → SHIPPED → DELIVERED
                ↓          ↓
            REJECTED    BLOCKED
    """
    machine_id: str
    object_type_id: str
    description: str = ""

    # 상태 및 전이 정의
    states: Dict[str, StateDefinition] = field(default_factory=dict)
    transitions: Dict[str, TransitionDefinition] = field(default_factory=dict)

    # 상태 저장 속성명
    state_property: str = "status"

    # 이력 저장 속성명 (선택)
    history_property: Optional[str] = "state_history"

    # 메타데이터
    version: str = "1.0.0"
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def add_state(self, state: StateDefinition) -> "StateMachine":
        """상태 추가 (빌더 패턴)"""
        self.states[state.state_id] = state
        return self

    def add_transition(self, transition: TransitionDefinition) -> "StateMachine":
        """전이 추가 (빌더 패턴)"""
        self.transitions[transition.transition_id] = transition
        return self

    def get_initial_state(self) -> Optional[StateDefinition]:
        """초기 상태 조회"""
        for state in self.states.values():
            if state.is_initial:
                return state
        return None

    def get_state(self, state_id: str) -> Optional[StateDefinition]:
        """상태 조회"""
        return self.states.get(state_id)

    def get_current_state(self, instance: "ObjectInstance") -> Optional[StateDefinition]:
        """인스턴스의 현재 상태 조회"""
        current_state_id = instance.get(self.state_property)
        return self.states.get(current_state_id) if current_state_id else None

    def get_available_transitions(self, current_state: str) -> List[TransitionDefinition]:
        """현재 상태에서 가능한 전이 목록"""
        return [t for t in self.transitions.values() if t.from_state == current_state]

    def get_automatic_transitions(self, current_state: str) -> List[TransitionDefinition]:
        """현재 상태에서 자동 전이 목록 (우선순위 순)"""
        transitions = [t for t in self.transitions.values()
                       if t.from_state == current_state and t.is_automatic]
        return sorted(transitions, key=lambda t: -t.priority)

    def is_behavior_allowed(
        self,
        current_state: str,
        behavior_id: str,
    ) -> bool:
        """
        현재 상태에서 행동 허용 여부

        팔란티어 검증 기준:
        - 상태에 따라 가능한 행동이 제한됨
        """
        state = self.states.get(current_state)
        if not state:
            return True  # 상태 정의 없으면 허용

        # 금지 목록 체크
        if behavior_id in state.forbidden_behaviors:
            return False

        # 허용 목록이 정의된 경우, 목록에 있어야 함
        if state.allowed_behaviors:
            return behavior_id in state.allowed_behaviors

        return True  # 기본 허용

    def get_next_states(self, current_state: str) -> List[str]:
        """현재 상태에서 이동 가능한 다음 상태 목록"""
        transitions = self.get_available_transitions(current_state)
        return list(set(t.to_state for t in transitions))

    def validate(self) -> List[str]:
        """상태 머신 유효성 검증"""
        errors = []

        # 초기 상태 확인
        initial_states = [s for s in self.states.values() if s.is_initial]
        if len(initial_states) == 0:
            errors.append("No initial state defined")
        elif len(initial_states) > 1:
            errors.append(f"Multiple initial states: {[s.state_id for s in initial_states]}")

        # 전이의 from/to 상태 존재 확인
        for t in self.transitions.values():
            if t.from_state not in self.states:
                errors.append(f"Transition {t.transition_id}: from_state '{t.from_state}' not found")
            if t.to_state not in self.states:
                errors.append(f"Transition {t.transition_id}: to_state '{t.to_state}' not found")

        # 도달 불가능 상태 확인
        reachable = set()
        if initial_states:
            queue = [initial_states[0].state_id]
            while queue:
                current = queue.pop(0)
                if current in reachable:
                    continue
                reachable.add(current)
                queue.extend(self.get_next_states(current))

        unreachable = set(self.states.keys()) - reachable
        if unreachable:
            errors.append(f"Unreachable states: {unreachable}")

        return errors

    def to_dict(self) -> Dict[str, Any]:
        return {
            "machine_id": self.machine_id,
            "object_type_id": self.object_type_id,
            "description": self.description,
            "state_property": self.state_property,
            "history_property": self.history_property,
            "states": {k: v.to_dict() for k, v in self.states.items()},
            "transitions": {k: v.to_dict() for k, v in self.transitions.items()},
            "version": self.version,
        }

    def to_mermaid(self) -> str:
        """Mermaid 다이어그램 생성"""
        lines = ["stateDiagram-v2"]

        # 초기 상태
        initial = self.get_initial_state()
        if initial:
            lines.append(f"    [*] --> {initial.state_id}")

        # 전이
        for t in self.transitions.values():
            label = t.name if t.name else t.transition_id
            lines.append(f"    {t.from_state} --> {t.to_state}: {label}")

        # 종료 상태
        for s in self.states.values():
            if s.is_terminal:
                lines.append(f"    {s.state_id} --> [*]")

        return "\n".join(lines)


class StateMachineExecutor:
    """
    상태 머신 실행기

    팔란티어 검증 기준:
    ✅ 상태 전이가 조건부로 실행됨 (guard)
    ✅ 전이 시 액션이 트리거됨
    ✅ 전이 이력이 기록됨 (audit)
    """

    def __init__(
        self,
        behavior_executor: Optional["BehaviorExecutor"] = None,
        actions_engine=None,
        evidence_chain: Optional["EvidenceChain"] = None,
    ):
        self.behavior_executor = behavior_executor
        self.actions = actions_engine
        self.evidence = evidence_chain

    async def transition(
        self,
        instance: "ObjectInstance",
        machine: StateMachine,
        transition_id: str,
        params: Dict[str, Any] = None,
        executor_id: Optional[str] = None,
        reason: Optional[str] = None,
    ) -> TransitionResult:
        """
        상태 전이 실행

        Args:
            instance: 대상 인스턴스
            machine: 상태 머신
            transition_id: 전이 ID
            params: 추가 파라미터
            executor_id: 실행자 ID (agent_id 또는 user_id)
            reason: 전이 사유

        Returns:
            TransitionResult
        """
        params = params or {}

        # 1. 전이 정의 조회
        transition = machine.transitions.get(transition_id)
        if not transition:
            return TransitionResult(
                success=False,
                status=TransitionResultStatus.FAILED,
                from_state=instance.get(machine.state_property, "unknown"),
                error=f"Transition not found: {transition_id}",
            )

        current_state = instance.get(machine.state_property)

        # 2. 현재 상태 검증
        if current_state != transition.from_state:
            return TransitionResult(
                success=False,
                status=TransitionResultStatus.INVALID_STATE,
                from_state=current_state,
                error=f"Invalid current state. Expected: {transition.from_state}, Got: {current_state}",
            )

        # 3. 종료 상태 체크
        from_state_def = machine.states.get(current_state)
        if from_state_def and from_state_def.is_terminal:
            return TransitionResult(
                success=False,
                status=TransitionResultStatus.INVALID_STATE,
                from_state=current_state,
                error=f"Cannot transition from terminal state: {current_state}",
            )

        # 4. Guard 조건 검증
        if transition.guard_condition:
            try:
                guard_context = {"self": instance, **params}
                if not eval(transition.guard_condition, {"__builtins__": {}}, guard_context):
                    return TransitionResult(
                        success=False,
                        status=TransitionResultStatus.GUARD_FAILED,
                        from_state=current_state,
                        error=f"Guard condition failed: {transition.guard_condition}",
                    )
            except Exception as e:
                return TransitionResult(
                    success=False,
                    status=TransitionResultStatus.GUARD_FAILED,
                    from_state=current_state,
                    error=f"Guard evaluation error: {str(e)}",
                )

        # 5. 확인 필요 여부 체크
        if transition.requires_confirmation and not params.get("confirmed"):
            return TransitionResult(
                success=False,
                status=TransitionResultStatus.PENDING_CONFIRMATION,
                from_state=current_state,
                to_state=transition.to_state,
                error="Transition requires confirmation",
            )

        # 6. on_exit 트리거 실행
        triggered = []
        if from_state_def:
            for trigger in from_state_def.on_exit_triggers:
                if self.actions:
                    try:
                        self.actions.execute(trigger, {
                            "instance_id": instance.instance_id,
                            "from_state": current_state,
                            "to_state": transition.to_state,
                            "transition": transition_id,
                        })
                        triggered.append(f"exit:{trigger}")
                    except Exception as e:
                        logger.warning(f"[StateMachineExecutor] Exit trigger failed: {e}")

        # 7. 전이 트리거 실행
        for trigger in transition.on_transition_triggers:
            if self.actions:
                try:
                    self.actions.execute(trigger, {
                        "instance_id": instance.instance_id,
                        "from_state": current_state,
                        "to_state": transition.to_state,
                        "params": params,
                    })
                    triggered.append(f"transition:{trigger}")
                except Exception as e:
                    logger.warning(f"[StateMachineExecutor] Transition trigger failed: {e}")

        # 8. 상태 변경
        old_state = current_state
        instance.set(machine.state_property, transition.to_state)

        # 9. 이력 저장
        if machine.history_property:
            history = instance.get(machine.history_property) or []
            history.append(StateHistory(
                from_state=old_state,
                to_state=transition.to_state,
                transition_id=transition_id,
                transitioned_at=datetime.now(),
                transitioned_by=executor_id,
                params=params,
                reason=reason,
            ).to_dict())
            instance.set(machine.history_property, history)

        # 10. on_enter 트리거 실행
        to_state_def = machine.states.get(transition.to_state)
        if to_state_def:
            for trigger in to_state_def.on_enter_triggers:
                if self.actions:
                    try:
                        self.actions.execute(trigger, {
                            "instance_id": instance.instance_id,
                            "state": transition.to_state,
                        })
                        triggered.append(f"enter:{trigger}")
                    except Exception as e:
                        logger.warning(f"[StateMachineExecutor] Enter trigger failed: {e}")

        # 11. Evidence Chain 기록
        if self.evidence:
            self.evidence.add_evidence(
                evidence_type="state_transition",
                agent_id=executor_id or "system",
                phase="runtime",
                content={
                    "instance_id": instance.instance_id,
                    "object_type": machine.object_type_id,
                    "from_state": old_state,
                    "to_state": transition.to_state,
                    "transition_id": transition_id,
                    "reason": reason,
                },
            )

        logger.info(
            f"[StateMachineExecutor] {machine.object_type_id} "
            f"transitioned: {old_state} → {transition.to_state}"
        )

        return TransitionResult(
            success=True,
            status=TransitionResultStatus.SUCCESS,
            from_state=old_state,
            to_state=transition.to_state,
            triggered_actions=triggered,
        )

    async def check_automatic_transitions(
        self,
        instance: "ObjectInstance",
        machine: StateMachine,
    ) -> Optional[TransitionResult]:
        """
        자동 전이 체크 및 실행

        조건 충족되는 자동 전이가 있으면 실행
        """
        current_state = instance.get(machine.state_property)

        for transition in machine.get_automatic_transitions(current_state):
            # Guard 조건 체크
            if transition.guard_condition:
                try:
                    if eval(transition.guard_condition, {"self": instance}):
                        # 조건 충족 → 전이 실행
                        return await self.transition(
                            instance, machine, transition.transition_id,
                            executor_id="auto_transition",
                        )
                except Exception:
                    continue

        return None

    async def initialize_state(
        self,
        instance: "ObjectInstance",
        machine: StateMachine,
    ) -> None:
        """인스턴스 초기 상태 설정"""
        initial_state = machine.get_initial_state()
        if initial_state:
            instance.set(machine.state_property, initial_state.state_id)

            # 이력 초기화
            if machine.history_property:
                instance.set(machine.history_property, [])

            # on_enter 트리거
            for trigger in initial_state.on_enter_triggers:
                if self.actions:
                    try:
                        self.actions.execute(trigger, {
                            "instance_id": instance.instance_id,
                            "state": initial_state.state_id,
                        })
                    except Exception as e:
                        logger.warning(f"[StateMachineExecutor] Initial enter trigger failed: {e}")

    def can_transition(
        self,
        instance: "ObjectInstance",
        machine: StateMachine,
        transition_id: str,
    ) -> bool:
        """전이 가능 여부 확인 (guard 체크 포함)"""
        transition = machine.transitions.get(transition_id)
        if not transition:
            return False

        current_state = instance.get(machine.state_property)
        if current_state != transition.from_state:
            return False

        if transition.guard_condition:
            try:
                if not eval(transition.guard_condition, {"self": instance}):
                    return False
            except Exception:
                return False

        return True


# 상태 머신 레지스트리
class GlobalStateMachineRegistry:
    """전역 상태 머신 레지스트리"""
    _machines: Dict[str, StateMachine] = {}

    @classmethod
    def register(cls, machine: StateMachine) -> None:
        """상태 머신 등록"""
        cls._machines[machine.object_type_id] = machine
        logger.info(f"[GlobalStateMachineRegistry] Registered state machine for {machine.object_type_id}")

    @classmethod
    def get(cls, object_type_id: str) -> Optional[StateMachine]:
        """상태 머신 조회"""
        return cls._machines.get(object_type_id)

    @classmethod
    def list_all(cls) -> List[str]:
        """등록된 모든 ObjectType ID"""
        return list(cls._machines.keys())

    @classmethod
    def clear(cls) -> None:
        """모든 레지스트리 초기화"""
        cls._machines.clear()


__all__ = [
    # State Definitions
    "StateDefinition",
    "TransitionDefinition",
    "TransitionResult",
    "TransitionResultStatus",
    "StateHistory",
    # State Machine
    "StateMachine",
    "StateMachineExecutor",
    "GlobalStateMachineRegistry",
]
