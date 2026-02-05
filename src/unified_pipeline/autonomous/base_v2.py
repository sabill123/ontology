"""
AutonomousAgentV2 - 팔란티어급 에이전트 (Object-Centric Agent)

v1.0 에이전트 확장:
- query(): OQL 기반 객체 쿼리 (SQL 없이 개념 단위)
- call_behavior(): 객체 행동 호출 (Object→Action 패턴)
- transition(): 명시적 상태 전이
- reason_with_objects(): 개념 단위 LLM reasoning

팔란티어 검증 기준 충족:
✅ 에이전트가 개념 단위로 사고 (테이블명 없이 동작)
✅ 에이전트가 ontology action만 호출
✅ Object→Action 패턴 (Order.evaluateRisk() 형태)
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Callable, TypeVar, Generic
from datetime import datetime
from enum import Enum
import asyncio
import json

from .base import AutonomousAgent
from ..osdk import (
    ObjectType,
    ObjectInstance,
    BehaviorExecutor,
    ExecutionResult,
    GlobalBehaviorRegistry,
    StateMachineExecutor,
    GlobalStateMachineRegistry,
    TransitionResult,
)
from ..oql import OQLEngine, OQLQuery
from ..shared_context import SharedContext
from ..measurement import (
    OutcomeTracker,
    GlobalOutcomeTracker,
    DecisionType,
    OutcomeStatus,
    FeedbackLoop,
)


class ReasoningMode(str, Enum):
    """에이전트 reasoning 모드"""
    CONCEPT_ONLY = "concept_only"      # 개념 단위만 (SQL 없음)
    HYBRID = "hybrid"                   # 개념 우선, 필요시 raw data
    LEGACY = "legacy"                   # v1.0 호환 모드


@dataclass
class ObjectAction:
    """객체-액션 기록"""
    object_type: str
    object_id: str
    action_name: str
    parameters: Dict[str, Any]
    result: Any
    executed_at: datetime = field(default_factory=datetime.now)
    success: bool = True
    error_message: Optional[str] = None


@dataclass
class ReasoningContext:
    """LLM reasoning 컨텍스트"""
    available_concepts: List[str]
    available_behaviors: Dict[str, List[str]]  # object_type -> behaviors
    available_transitions: Dict[str, List[str]]  # object_type -> transitions
    current_objects: List[Dict[str, Any]]
    query_history: List[str] = field(default_factory=list)
    action_history: List[ObjectAction] = field(default_factory=list)


class AutonomousAgentV2(AutonomousAgent):
    """
    팔란티어급 자율 에이전트 (v2.0)

    v1.0 기능 유지 + 다음 기능 추가:
    - 개념 단위 쿼리 (OQL)
    - 객체 행동 호출 (Behavior)
    - 명시적 상태 전이 (State Machine)
    - 개념 단위 LLM reasoning

    팔란티어 검증:
    ✅ 비즈니스 개념이 코드/권한/액션의 중심
    ✅ 데이터 소스가 바뀌어도 로직이 안 깨짐
    ✅ 상태 전이가 명시적
    ✅ 에이전트가 개념 단위로 reasoning
    """

    def __init__(
        self,
        agent_id: str,
        context: SharedContext,
        reasoning_mode: ReasoningMode = ReasoningMode.CONCEPT_ONLY,
        outcome_tracker: Optional[OutcomeTracker] = None,
        **kwargs
    ):
        super().__init__(agent_id=agent_id, context=context, **kwargs)

        self.reasoning_mode = reasoning_mode

        # v2.0 컴포넌트
        self.oql_engine = OQLEngine(context)
        self.behavior_executor = BehaviorExecutor(context)
        self.state_machine_executor = StateMachineExecutor()

        # Outcome Measurement (팔란티어 검증 #5: 의사결정 결과 측정)
        self.outcome_tracker = outcome_tracker or GlobalOutcomeTracker
        self.feedback_loop = FeedbackLoop(self.outcome_tracker)

        # Action 기록 (Outcome Measurement용)
        self.action_history: List[ObjectAction] = []

        # Reasoning 컨텍스트
        self._reasoning_context: Optional[ReasoningContext] = None

    # =========================================================================
    # OQL 기반 쿼리 (개념 단위)
    # =========================================================================

    def query(self, object_type: str) -> "OQLEngine":
        """
        개념 단위 쿼리 시작

        팔란티어 패턴:
            agent.query("Order")
                .filter(status="pending")
                .filter(amount__gt=1000)
                .include("customer", "items")
                .execute()

        SQL 없이 개념 단위로 동작
        """
        return self.oql_engine.query(object_type)

    async def ask(self, natural_language_query: str) -> List[Dict[str, Any]]:
        """
        자연어 쿼리 (LLM→OQL 변환)

        예시:
            results = await agent.ask("위험도가 높은 주문을 찾아줘")

        팔란티어 검증:
        ✅ 프롬프트에 테이블명 없이 동작
        ✅ 개념 단위 사고
        """
        return await self.oql_engine.ask(natural_language_query)

    # =========================================================================
    # 객체 행동 호출 (Object→Action)
    # =========================================================================

    async def call_behavior(
        self,
        obj: ObjectInstance,
        behavior_name: str,
        reason: Optional[str] = None,
        expected_outcome: Optional[str] = None,
        **parameters
    ) -> ExecutionResult:
        """
        객체 행동 호출

        팔란티어 패턴 (Object→Action):
            result = await agent.call_behavior(order, "evaluateRisk")
            result = await agent.call_behavior(customer, "checkCredit", threshold=1000)

        SQL이 아닌 개념 메서드 호출
        """
        # 행동 가져오기
        registry = GlobalBehaviorRegistry.get_registry(obj.object_type.name)
        if not registry:
            raise ValueError(f"No behaviors registered for {obj.object_type.name}")

        behavior = registry.get(behavior_name)
        if not behavior:
            raise ValueError(
                f"Behavior '{behavior_name}' not found for {obj.object_type.name}. "
                f"Available: {list(registry.list().keys())}"
            )

        # Outcome Measurement: 의사결정 기록
        decision = self.outcome_tracker.record_decision(
            decision_type=DecisionType.BEHAVIOR_CALL,
            object_type=obj.object_type.name,
            object_id=obj.id,
            object_state=dict(obj.properties),
            action_name=behavior_name,
            action_parameters=parameters,
            agent_id=self.agent_id,
            reason=reason,
            expected_outcome=expected_outcome
        )

        # 행동 실행
        start_time = datetime.now()
        result = await self.behavior_executor.execute(
            behavior=behavior,
            obj=obj,
            **parameters
        )
        execution_time_ms = int((datetime.now() - start_time).total_seconds() * 1000)

        # Outcome Measurement: 결과 기록
        self.outcome_tracker.record_outcome(
            decision_id=decision.decision_id,
            status=OutcomeStatus.SUCCESS if result.success else OutcomeStatus.FAILURE,
            actual_outcome={"result": result.result, "error": result.error_message},
            object_state_after=dict(obj.properties),
            execution_time_ms=execution_time_ms
        )

        # Action 기록 (로컬)
        action = ObjectAction(
            object_type=obj.object_type.name,
            object_id=obj.id,
            action_name=behavior_name,
            parameters=parameters,
            result=result.result,
            success=result.success,
            error_message=result.error_message
        )
        self.action_history.append(action)

        return result

    async def batch_call_behavior(
        self,
        objects: List[ObjectInstance],
        behavior_name: str,
        **parameters
    ) -> List[ExecutionResult]:
        """
        여러 객체에 행동 일괄 호출

        예시:
            results = await agent.batch_call_behavior(
                orders, "evaluateRisk"
            )
        """
        tasks = [
            self.call_behavior(obj, behavior_name, **parameters)
            for obj in objects
        ]
        return await asyncio.gather(*tasks)

    def get_available_behaviors(self, object_type: str) -> List[str]:
        """객체 타입의 사용 가능한 행동 목록"""
        registry = GlobalBehaviorRegistry.get_registry(object_type)
        if not registry:
            return []
        return list(registry.list().keys())

    # =========================================================================
    # 상태 전이 (Explicit State Transition)
    # =========================================================================

    async def transition(
        self,
        obj: ObjectInstance,
        to_state: str,
        reason: Optional[str] = None,
        expected_outcome: Optional[str] = None,
        **context
    ) -> TransitionResult:
        """
        명시적 상태 전이

        팔란티어 검증:
        ✅ 상태 전이가 명시적
        ✅ 전이 조건 검증 (guard)
        ✅ 전이 기록 (audit trail)

        예시:
            result = await agent.transition(
                order, "shipped",
                reason="모든 품목 준비 완료"
            )
        """
        # 현재 상태 가져오기
        current_state = obj.get("status") or obj.get("state") or "unknown"

        # 상태 머신 가져오기
        state_machine = GlobalStateMachineRegistry.get(obj.object_type.name)
        if not state_machine:
            raise ValueError(
                f"No state machine registered for {obj.object_type.name}"
            )

        # Outcome Measurement: 의사결정 기록
        decision = self.outcome_tracker.record_decision(
            decision_type=DecisionType.STATE_TRANSITION,
            object_type=obj.object_type.name,
            object_id=obj.id,
            object_state=dict(obj.properties),
            action_name=f"transition:{current_state}→{to_state}",
            action_parameters={"reason": reason, **context},
            agent_id=self.agent_id,
            reason=reason,
            expected_outcome=expected_outcome or f"state={to_state}"
        )

        # 전이 실행
        start_time = datetime.now()
        result = await self.state_machine_executor.transition(
            state_machine=state_machine,
            obj=obj,
            from_state=current_state,
            to_state=to_state,
            context={"reason": reason, **context},
            actor_id=self.agent_id
        )
        execution_time_ms = int((datetime.now() - start_time).total_seconds() * 1000)

        # Outcome Measurement: 결과 기록
        self.outcome_tracker.record_outcome(
            decision_id=decision.decision_id,
            status=OutcomeStatus.SUCCESS if result.success else OutcomeStatus.FAILURE,
            actual_outcome={
                "from_state": current_state,
                "to_state": to_state if result.success else current_state,
                "message": result.message
            },
            object_state_after=dict(obj.properties),
            execution_time_ms=execution_time_ms
        )

        # Action 기록 (로컬)
        if result.success:
            action = ObjectAction(
                object_type=obj.object_type.name,
                object_id=obj.id,
                action_name=f"transition:{current_state}→{to_state}",
                parameters={"reason": reason, **context},
                result={"from": current_state, "to": to_state},
                success=True
            )
            self.action_history.append(action)

        return result

    def get_available_transitions(
        self,
        obj: ObjectInstance
    ) -> List[str]:
        """현재 상태에서 가능한 전이 목록"""
        current_state = obj.get("status") or obj.get("state") or "unknown"

        state_machine = GlobalStateMachineRegistry.get(obj.object_type.name)
        if not state_machine:
            return []

        return state_machine.get_available_transitions(current_state)

    # =========================================================================
    # 개념 단위 LLM Reasoning
    # =========================================================================

    async def reason_with_objects(
        self,
        question: str,
        relevant_objects: Optional[List[ObjectInstance]] = None,
        max_objects: int = 10
    ) -> Dict[str, Any]:
        """
        개념 단위 LLM reasoning

        팔란티어 검증:
        ✅ 에이전트가 개념 단위로 사고
        ✅ 테이블명 없이 동작
        ✅ ontology action만 호출

        예시:
            result = await agent.reason_with_objects(
                "이 주문의 위험도를 평가하고 다음 단계를 추천해줘",
                relevant_objects=[order]
            )

        반환:
            {
                "analysis": "주문 분석 결과...",
                "recommended_actions": [
                    {"object": "Order:123", "action": "evaluateRisk"},
                    {"object": "Order:123", "action": "transition:pending→approved"}
                ],
                "reasoning": "위험도가 낮고 재고가 충분하므로..."
            }
        """
        # Reasoning 컨텍스트 구성
        context = self._build_reasoning_context(relevant_objects, max_objects)

        # LLM 프롬프트 구성 (테이블명 없이 개념 단위)
        prompt = self._build_reasoning_prompt(question, context)

        # LLM 호출
        response = await self._call_llm_for_reasoning(prompt)

        return response

    def _build_reasoning_context(
        self,
        objects: Optional[List[ObjectInstance]],
        max_objects: int
    ) -> ReasoningContext:
        """Reasoning 컨텍스트 구성"""
        # 사용 가능한 개념 목록
        available_concepts = list(GlobalBehaviorRegistry._type_registries.keys())

        # 개념별 행동 목록
        available_behaviors = {}
        for concept in available_concepts:
            registry = GlobalBehaviorRegistry.get_registry(concept)
            if registry:
                available_behaviors[concept] = list(registry.list().keys())

        # 개념별 전이 목록
        available_transitions = {}
        for concept in available_concepts:
            sm = GlobalStateMachineRegistry.get(concept)
            if sm:
                available_transitions[concept] = [
                    f"{t.from_state}→{t.to_state}"
                    for t in sm.transitions.values()
                ]

        # 현재 객체 정보
        current_objects = []
        if objects:
            for obj in objects[:max_objects]:
                current_objects.append({
                    "type": obj.object_type.name,
                    "id": obj.id,
                    "properties": dict(obj.properties),
                    "available_actions": self.get_available_behaviors(
                        obj.object_type.name
                    ),
                    "available_transitions": self.get_available_transitions(obj)
                })

        return ReasoningContext(
            available_concepts=available_concepts,
            available_behaviors=available_behaviors,
            available_transitions=available_transitions,
            current_objects=current_objects,
            action_history=self.action_history[-10:]  # 최근 10개
        )

    def _build_reasoning_prompt(
        self,
        question: str,
        context: ReasoningContext
    ) -> str:
        """
        개념 단위 reasoning 프롬프트

        팔란티어 검증:
        ✅ 테이블명/SQL 없음
        ✅ 개념과 행동만 언급
        """
        prompt = f"""당신은 온톨로지 기반 의사결정 에이전트입니다.

## 사용 가능한 개념 (Concepts)
{json.dumps(context.available_concepts, indent=2, ensure_ascii=False)}

## 개념별 행동 (Behaviors)
{json.dumps(context.available_behaviors, indent=2, ensure_ascii=False)}

## 개념별 상태 전이 (Transitions)
{json.dumps(context.available_transitions, indent=2, ensure_ascii=False)}

## 현재 객체 정보
{json.dumps(context.current_objects, indent=2, ensure_ascii=False)}

## 최근 액션 기록
{self._format_action_history(context.action_history)}

---

## 질문
{question}

---

## 지침
1. SQL이나 테이블명을 사용하지 마세요
2. 개념(Concept)과 행동(Behavior)만 사용하세요
3. 추천하는 액션은 "Object.action()" 형태로 명시하세요
4. 상태 전이가 필요하면 "Object.transition(to_state)" 형태로 명시하세요

응답 형식 (JSON):
{{
    "analysis": "분석 결과",
    "recommended_actions": [
        {{"object": "타입:ID", "action": "행동명", "reason": "이유"}}
    ],
    "reasoning": "추론 과정"
}}
"""
        return prompt

    def _format_action_history(self, history: List[ObjectAction]) -> str:
        """액션 기록 포맷팅"""
        if not history:
            return "없음"

        lines = []
        for action in history:
            status = "✓" if action.success else "✗"
            lines.append(
                f"- [{status}] {action.object_type}:{action.object_id}"
                f".{action.action_name}()"
            )
        return "\n".join(lines)

    async def _call_llm_for_reasoning(
        self,
        prompt: str
    ) -> Dict[str, Any]:
        """LLM 호출하여 reasoning 수행"""
        # SharedContext의 LLM 호출 사용
        try:
            response = await self.context.call_llm(
                prompt=prompt,
                system="You are an ontology-based decision agent. "
                       "Always respond in valid JSON format.",
                temperature=0.3
            )

            # JSON 파싱
            return json.loads(response)
        except json.JSONDecodeError:
            return {
                "analysis": response,
                "recommended_actions": [],
                "reasoning": "JSON 파싱 실패"
            }
        except Exception as e:
            return {
                "analysis": f"Error: {str(e)}",
                "recommended_actions": [],
                "reasoning": str(e)
            }

    # =========================================================================
    # 에이전트 실행 (v2.0 오버라이드)
    # =========================================================================

    async def execute_action(
        self,
        action_type: str,
        **kwargs
    ) -> Dict[str, Any]:
        """
        액션 실행 (v2.0 확장)

        v2.0 액션 타입:
        - query: OQL 쿼리 실행
        - behavior: 객체 행동 호출
        - transition: 상태 전이
        - reason: 개념 단위 reasoning
        """
        if action_type == "query":
            object_type = kwargs.get("object_type")
            filters = kwargs.get("filters", {})

            query = self.query(object_type)
            for key, value in filters.items():
                query = query.filter(**{key: value})

            return {"results": query.execute()}

        elif action_type == "behavior":
            obj = kwargs.get("object")
            behavior_name = kwargs.get("behavior")
            params = kwargs.get("parameters", {})

            result = await self.call_behavior(obj, behavior_name, **params)
            return {
                "success": result.success,
                "result": result.result,
                "error": result.error_message
            }

        elif action_type == "transition":
            obj = kwargs.get("object")
            to_state = kwargs.get("to_state")
            reason = kwargs.get("reason")

            result = await self.transition(obj, to_state, reason=reason)
            return {
                "success": result.success,
                "message": result.message
            }

        elif action_type == "reason":
            question = kwargs.get("question")
            objects = kwargs.get("objects", [])

            return await self.reason_with_objects(question, objects)

        else:
            # v1.0 폴백
            return await super().execute_action(action_type, **kwargs)

    # =========================================================================
    # 유틸리티
    # =========================================================================

    def get_action_history(
        self,
        object_type: Optional[str] = None,
        limit: int = 100
    ) -> List[ObjectAction]:
        """액션 기록 조회 (Outcome Measurement용)"""
        history = self.action_history

        if object_type:
            history = [a for a in history if a.object_type == object_type]

        return history[-limit:]

    def get_success_rate(
        self,
        object_type: Optional[str] = None
    ) -> float:
        """액션 성공률 (Outcome Measurement용)"""
        history = self.get_action_history(object_type)
        if not history:
            return 0.0

        success_count = sum(1 for a in history if a.success)
        return success_count / len(history)

    def get_outcome_metrics(self) -> Dict[str, Any]:
        """
        에이전트의 Outcome 메트릭 조회

        팔란티어 검증:
        ✅ 의사결정 결과가 측정됨
        """
        metrics = self.outcome_tracker.compute_metrics(agent_id=self.agent_id)
        return metrics.to_dict()

    def get_improvement_suggestions(self) -> List[Dict[str, Any]]:
        """
        개선 제안 조회

        팔란티어 검증:
        ✅ Feedback loop 형성
        """
        return self.feedback_loop.analyze_failures()

    def generate_outcome_report(self) -> Dict[str, Any]:
        """
        종합 Outcome 보고서 생성

        팔란티어 검증:
        ✅ 의사결정 결과 측정 보고서
        ✅ 지속적 개선을 위한 피드백
        """
        return self.feedback_loop.generate_report()

    def __repr__(self) -> str:
        return (
            f"AutonomousAgentV2("
            f"id={self.agent_id}, "
            f"mode={self.reasoning_mode.value}, "
            f"actions={len(self.action_history)})"
        )
