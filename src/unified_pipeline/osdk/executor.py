"""
Behavior Executor (v2.0)

행동 실행 엔진 - 3가지 실행 모드 지원

팔란티어 검증 기준:
✅ 로직이 SQL 쿼리가 아닌 Object → Action 구조
✅ AI/Agent가 개념 단위로 reasoning (LLM 모드)
✅ 결정 결과가 측정됨 (ExecutionResult 추적)

실행 모드:
1. PYTHON: 샌드박스에서 Python 코드 실행
2. LLM: LLM 프롬프트 실행 후 결과 파싱
3. OQL: OQL 쿼리 실행
4. HYBRID: Python 데이터 준비 + LLM 추론

사용 예시:
    executor = BehaviorExecutor(llm_client=llm, shared_context=context)

    # 행동 실행
    result = await executor.execute(order_instance, evaluate_risk_behavior, {})

    if result.success:
        print(f"Risk score: {result.result['risk_score']}")
        print(f"Execution time: {result.execution_time_ms}ms")
    else:
        print(f"Failed: {result.error}")
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, TYPE_CHECKING
from datetime import datetime
from enum import Enum
import asyncio
import hashlib
import json
import re
import logging

from .behavior import BehaviorDefinition, ExecutionMode, BehaviorParameter

if TYPE_CHECKING:
    from .models import ObjectInstance
    from ..shared_context import SharedContext

logger = logging.getLogger(__name__)


class ExecutionErrorType(str, Enum):
    """실행 오류 유형"""
    PARAMETER_VALIDATION = "parameter_validation"
    PRECONDITION_FAILED = "precondition_failed"
    POSTCONDITION_FAILED = "postcondition_failed"
    TIMEOUT = "timeout"
    EXECUTION_ERROR = "execution_error"
    PERMISSION_DENIED = "permission_denied"
    NOT_FOUND = "not_found"


@dataclass
class ExecutionResult:
    """
    행동 실행 결과

    팔란티어 검증 기준 - 결과 측정:
    ✅ 결정 시간 (execution_time_ms)
    ✅ 성공/실패 상태
    ✅ 상태 변경 추적 (state_changes)
    ✅ 연쇄 행동 추적 (triggered_behaviors)
    """
    success: bool
    result: Any = None
    error: Optional[str] = None
    error_type: Optional[ExecutionErrorType] = None

    # 실행 정보
    execution_time_ms: float = 0.0
    execution_mode: Optional[str] = None

    # 부수효과 추적
    state_changes: Dict[str, Any] = field(default_factory=dict)
    triggered_behaviors: List[str] = field(default_factory=list)

    # 감사(Audit) 정보
    executed_at: datetime = field(default_factory=datetime.now)
    executed_by: Optional[str] = None  # agent_id

    # 캐시 정보
    from_cache: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "result": self.result,
            "error": self.error,
            "error_type": self.error_type.value if self.error_type else None,
            "execution_time_ms": self.execution_time_ms,
            "execution_mode": self.execution_mode,
            "state_changes": self.state_changes,
            "triggered_behaviors": self.triggered_behaviors,
            "executed_at": self.executed_at.isoformat(),
            "executed_by": self.executed_by,
            "from_cache": self.from_cache,
        }


class BehaviorExecutor:
    """
    행동 실행기

    3가지 실행 모드 지원:
    - PYTHON: 샌드박스에서 Python 코드 실행
    - LLM: LLM 프롬프트 실행 후 결과 파싱
    - OQL: OQL 쿼리 실행 (v2.0-C에서 연동)
    - HYBRID: Python + LLM 조합

    팔란티어 검증 기준:
    ✅ Object → Action 구조 (SQL 아님)
    ✅ 개념 단위 실행 (ObjectInstance 대상)
    ✅ ActionsEngine과 연동하여 트리거 실행
    """

    def __init__(
        self,
        llm_client=None,
        oql_engine=None,
        shared_context: Optional["SharedContext"] = None,
        actions_engine=None,  # 기존 ActionsEngine 연동
    ):
        self.llm = llm_client
        self.oql = oql_engine
        self.context = shared_context
        self.actions = actions_engine

        # 결과 캐시
        self._cache: Dict[str, Dict[str, Any]] = {}

        # 실행 이력 (Outcome Measurement용)
        self._execution_history: List[ExecutionResult] = []

    async def execute(
        self,
        instance: "ObjectInstance",
        behavior: BehaviorDefinition,
        params: Dict[str, Any],
        executor_id: Optional[str] = None,
    ) -> ExecutionResult:
        """
        행동 실행

        Args:
            instance: 대상 ObjectInstance (self 역할)
            behavior: 실행할 BehaviorDefinition
            params: 파라미터 (behavior.parameters에 정의된 값들)
            executor_id: 실행자 ID (agent_id 등)

        Returns:
            ExecutionResult

        팔란티어 패턴:
            # v1.0 (SQL 스타일) - ❌
            SELECT * FROM orders WHERE risk_score > 0.8

            # v2.0 (Object → Action) - ✅
            result = await executor.execute(order, evaluateRisk, {})
        """
        start_time = datetime.now()

        # 1. 캐시 체크 (QUERY 타입에 유용)
        if behavior.cache_ttl_seconds:
            cached = self._get_cached(instance, behavior, params)
            if cached is not None:
                return ExecutionResult(
                    success=True,
                    result=cached,
                    execution_time_ms=0,
                    execution_mode="cached",
                    from_cache=True,
                    executed_by=executor_id,
                )

        # 2. 파라미터 검증
        param_error = self._validate_parameters(behavior.parameters, params)
        if param_error:
            return ExecutionResult(
                success=False,
                error=param_error,
                error_type=ExecutionErrorType.PARAMETER_VALIDATION,
                executed_by=executor_id,
            )

        # 3. 사전 조건 검증 (Design-by-Contract)
        precond_error = self._check_preconditions(instance, behavior.preconditions)
        if precond_error:
            return ExecutionResult(
                success=False,
                error=precond_error,
                error_type=ExecutionErrorType.PRECONDITION_FAILED,
                executed_by=executor_id,
            )

        # 4. 실행 (v18.0: timeout=0이면 무제한)
        try:
            if behavior.timeout_seconds and behavior.timeout_seconds > 0:
                result = await asyncio.wait_for(
                    self._execute_by_mode(instance, behavior, params),
                    timeout=behavior.timeout_seconds,
                )
            else:
                result = await self._execute_by_mode(instance, behavior, params)

        except asyncio.TimeoutError:
            await self._run_failure_triggers(instance, behavior, "Timeout")
            return ExecutionResult(
                success=False,
                error=f"Execution timeout ({behavior.timeout_seconds}s)",
                error_type=ExecutionErrorType.TIMEOUT,
                execution_time_ms=(datetime.now() - start_time).total_seconds() * 1000,
                executed_by=executor_id,
            )
        except Exception as e:
            await self._run_failure_triggers(instance, behavior, str(e))
            logger.error(f"[BehaviorExecutor] Execution error: {e}")
            return ExecutionResult(
                success=False,
                error=str(e),
                error_type=ExecutionErrorType.EXECUTION_ERROR,
                execution_time_ms=(datetime.now() - start_time).total_seconds() * 1000,
                executed_by=executor_id,
            )

        # 5. 사후 조건 검증
        postcond_error = self._check_postconditions(instance, behavior.postconditions)
        if postcond_error:
            return ExecutionResult(
                success=False,
                result=result,
                error=postcond_error,
                error_type=ExecutionErrorType.POSTCONDITION_FAILED,
                executed_by=executor_id,
            )

        # 6. 성공 트리거 실행
        triggered = await self._run_success_triggers(instance, behavior, result)

        # 7. 캐시 저장
        if behavior.cache_ttl_seconds:
            self._set_cached(instance, behavior, params, result)

        # 8. 결과 생성
        execution_time = (datetime.now() - start_time).total_seconds() * 1000

        exec_result = ExecutionResult(
            success=True,
            result=result,
            execution_time_ms=execution_time,
            execution_mode=behavior.execution_mode.value,
            triggered_behaviors=triggered,
            executed_at=start_time,
            executed_by=executor_id,
        )

        # 실행 이력 저장 (Outcome Measurement용)
        self._execution_history.append(exec_result)

        logger.debug(
            f"[BehaviorExecutor] {behavior.behavior_id} executed in {execution_time:.2f}ms"
        )

        return exec_result

    async def _execute_by_mode(
        self,
        instance: "ObjectInstance",
        behavior: BehaviorDefinition,
        params: Dict[str, Any],
    ) -> Any:
        """실행 모드에 따른 분기"""
        if behavior.execution_mode == ExecutionMode.PYTHON:
            return await self._execute_python(instance, behavior, params)
        elif behavior.execution_mode == ExecutionMode.LLM:
            return await self._execute_llm(instance, behavior, params)
        elif behavior.execution_mode == ExecutionMode.OQL:
            return await self._execute_oql(instance, behavior, params)
        elif behavior.execution_mode == ExecutionMode.HYBRID:
            return await self._execute_hybrid(instance, behavior, params)
        else:
            raise ValueError(f"Unknown execution mode: {behavior.execution_mode}")

    # ============== 실행 모드별 구현 ==============

    async def _execute_python(
        self,
        instance: "ObjectInstance",
        behavior: BehaviorDefinition,
        params: Dict[str, Any],
    ) -> Any:
        """
        Python 코드 실행

        샌드박스 환경에서 안전하게 실행:
        - self: ObjectInstance (속성 접근용)
        - params: 파라미터 딕셔너리
        - result: 결과 저장 변수
        - context: SharedContext (선택적)
        """
        if not behavior.python_code:
            raise ValueError("python_code not defined for PYTHON execution mode")

        # 안전한 실행 환경 구성
        local_vars = {
            "self": instance,
            "context": self.context,
            "params": params,
            **params,  # 파라미터 직접 접근 가능
            "result": None,
        }

        # 허용된 내장 함수만 (보안)
        safe_builtins = {
            "len": len, "str": str, "int": int, "float": float,
            "bool": bool, "list": list, "dict": dict, "set": set,
            "min": min, "max": max, "sum": sum, "abs": abs,
            "round": round, "sorted": sorted, "enumerate": enumerate,
            "zip": zip, "map": map, "filter": filter, "range": range,
            "True": True, "False": False, "None": None,
            "isinstance": isinstance, "hasattr": hasattr, "getattr": getattr,
        }

        global_vars = {"__builtins__": safe_builtins}

        # 실행
        exec(behavior.python_code, global_vars, local_vars)

        return local_vars.get("result")

    async def _execute_llm(
        self,
        instance: "ObjectInstance",
        behavior: BehaviorDefinition,
        params: Dict[str, Any],
    ) -> Any:
        """
        LLM 프롬프트 실행

        팔란티어 AIP Agent 스타일:
        - 객체 컨텍스트를 프롬프트에 포함
        - 구조화된 응답 파싱

        팔란티어 검증 기준:
        ✅ 프롬프트에 테이블명이 없어도 동작
        ✅ 에이전트가 개념 단위로 reasoning
        """
        if not behavior.llm_prompt_template:
            raise ValueError("llm_prompt_template not defined for LLM execution mode")

        if not self.llm:
            raise RuntimeError("LLM client not configured")

        # 템플릿 변수 치환
        prompt = self._render_prompt(behavior.llm_prompt_template, instance, params)

        # LLM 호출
        response = await self.llm.call(prompt)

        # 결과 파싱
        return self._parse_llm_response(response, behavior.return_type)

    async def _execute_oql(
        self,
        instance: "ObjectInstance",
        behavior: BehaviorDefinition,
        params: Dict[str, Any],
    ) -> Any:
        """
        OQL 쿼리 실행

        v2.0-C OQL Engine과 연동
        """
        if not behavior.oql_query:
            raise ValueError("oql_query not defined for OQL execution mode")

        if not self.oql:
            raise RuntimeError("OQL engine not configured")

        # self를 인스턴스 참조로 치환
        query = behavior.oql_query.replace("self", f"@{instance.instance_id}")

        # 파라미터 치환
        for key, value in params.items():
            query = query.replace(f"${key}", json.dumps(value))

        return await self.oql.execute_raw(query)

    async def _execute_hybrid(
        self,
        instance: "ObjectInstance",
        behavior: BehaviorDefinition,
        params: Dict[str, Any],
    ) -> Any:
        """
        Python + LLM 조합 실행

        1단계: Python으로 데이터 준비
        2단계: LLM으로 추론
        """
        # 1단계: Python 실행
        prepared_data = await self._execute_python(instance, behavior, params)

        # 2단계: LLM 실행 (준비된 데이터 포함)
        enhanced_params = {**params, "prepared_data": prepared_data}
        return await self._execute_llm(instance, behavior, enhanced_params)

    # ============== 헬퍼 메서드 ==============

    def _validate_parameters(
        self,
        param_defs: List[BehaviorParameter],
        params: Dict[str, Any],
    ) -> Optional[str]:
        """파라미터 검증"""
        for p in param_defs:
            # 필수 파라미터 체크
            if p.required and p.name not in params:
                if p.default is None:
                    return f"Required parameter missing: {p.name}"

            # 검증 규칙 적용
            if p.name in params and p.validation_rule:
                value = params[p.name]
                try:
                    if not eval(p.validation_rule, {"value": value, "len": len}):
                        return f"Validation failed for {p.name}: {p.validation_rule}"
                except Exception as e:
                    return f"Validation error for {p.name}: {str(e)}"

        return None

    def _check_preconditions(
        self,
        instance: "ObjectInstance",
        conditions: List[str],
    ) -> Optional[str]:
        """사전 조건 검증 (Design-by-Contract)"""
        for cond in conditions:
            try:
                if not eval(cond, {"self": instance}):
                    return f"Precondition failed: {cond}"
            except Exception as e:
                return f"Precondition error: {cond} - {str(e)}"
        return None

    def _check_postconditions(
        self,
        instance: "ObjectInstance",
        conditions: List[str],
    ) -> Optional[str]:
        """사후 조건 검증 (Design-by-Contract)"""
        for cond in conditions:
            try:
                if not eval(cond, {"self": instance}):
                    return f"Postcondition failed: {cond}"
            except Exception as e:
                return f"Postcondition error: {cond} - {str(e)}"
        return None

    def _render_prompt(
        self,
        template: str,
        instance: "ObjectInstance",
        params: Dict[str, Any],
    ) -> str:
        """
        프롬프트 템플릿 렌더링

        지원 패턴:
        - {self.property}: ObjectInstance 속성
        - {param_name}: 파라미터 값
        - {self}: 전체 인스턴스 JSON
        """
        result = template

        # {self.property} 형태 치환
        for key, value in instance.properties.items():
            result = result.replace(f"{{self.{key}}}", str(value))

        # {self} 전체 치환
        result = result.replace("{self}", json.dumps(instance.properties, ensure_ascii=False))

        # {param} 형태 치환
        for key, value in params.items():
            result = result.replace(f"{{{key}}}", str(value))

        return result

    def _parse_llm_response(self, response: str, return_type: Optional[str]) -> Any:
        """LLM 응답 파싱"""
        # JSON 추출 시도
        try:
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                return json.loads(json_match.group())
        except json.JSONDecodeError:
            pass

        # 타입에 따른 변환
        if return_type == "number":
            numbers = re.findall(r'-?\d+\.?\d*', response)
            return float(numbers[0]) if numbers else 0.0
        elif return_type == "boolean":
            lower = response.lower()
            return "true" in lower or "yes" in lower or "확인" in lower
        elif return_type == "string":
            return response.strip()

        return response

    async def _run_success_triggers(
        self,
        instance: "ObjectInstance",
        behavior: BehaviorDefinition,
        result: Any,
    ) -> List[str]:
        """
        성공 트리거 실행

        기존 ActionsEngine과 연동
        """
        triggered = []
        for trigger_id in behavior.triggers_on_success:
            if self.actions:
                try:
                    self.actions.execute(trigger_id, {
                        "instance_id": instance.instance_id,
                        "type_id": instance.type_id,
                        "behavior_id": behavior.behavior_id,
                        "result": result,
                    })
                    triggered.append(trigger_id)
                except Exception as e:
                    logger.warning(f"[BehaviorExecutor] Trigger failed: {trigger_id} - {e}")
        return triggered

    async def _run_failure_triggers(
        self,
        instance: "ObjectInstance",
        behavior: BehaviorDefinition,
        error: str,
    ) -> None:
        """실패 트리거 실행"""
        for trigger_id in behavior.triggers_on_failure:
            if self.actions:
                try:
                    self.actions.execute(trigger_id, {
                        "instance_id": instance.instance_id,
                        "type_id": instance.type_id,
                        "behavior_id": behavior.behavior_id,
                        "error": error,
                    })
                except Exception as e:
                    logger.warning(f"[BehaviorExecutor] Failure trigger failed: {trigger_id} - {e}")

    # ============== 캐싱 ==============

    def _cache_key(
        self,
        instance: "ObjectInstance",
        behavior: BehaviorDefinition,
        params: Dict[str, Any],
    ) -> str:
        """캐시 키 생성"""
        key_data = f"{instance.instance_id}:{behavior.behavior_id}:{json.dumps(params, sort_keys=True)}"
        return hashlib.md5(key_data.encode()).hexdigest()

    def _get_cached(
        self,
        instance: "ObjectInstance",
        behavior: BehaviorDefinition,
        params: Dict[str, Any],
    ) -> Optional[Any]:
        """캐시 조회"""
        key = self._cache_key(instance, behavior, params)
        cached = self._cache.get(key)
        if cached:
            # TTL 체크
            elapsed = (datetime.now() - cached["time"]).total_seconds()
            if elapsed < behavior.cache_ttl_seconds:
                return cached["result"]
            else:
                del self._cache[key]
        return None

    def _set_cached(
        self,
        instance: "ObjectInstance",
        behavior: BehaviorDefinition,
        params: Dict[str, Any],
        result: Any,
    ) -> None:
        """캐시 저장"""
        key = self._cache_key(instance, behavior, params)
        self._cache[key] = {"result": result, "time": datetime.now()}

    def clear_cache(self) -> int:
        """캐시 초기화"""
        count = len(self._cache)
        self._cache.clear()
        return count

    # ============== 메트릭 ==============

    def get_execution_history(
        self,
        limit: int = 100,
        behavior_id: Optional[str] = None,
        success_only: bool = False,
    ) -> List[ExecutionResult]:
        """
        실행 이력 조회 (Outcome Measurement용)
        """
        history = self._execution_history

        if behavior_id:
            # behavior_id로 필터링하려면 result에 저장해야 하므로 일단 전체 반환
            pass

        if success_only:
            history = [r for r in history if r.success]

        return history[-limit:]

    def get_metrics(self) -> Dict[str, Any]:
        """
        실행 메트릭 요약

        팔란티어 검증 기준:
        ✅ 결정 시간 측정
        ✅ 성공/실패 비율
        """
        if not self._execution_history:
            return {"total_executions": 0}

        total = len(self._execution_history)
        successful = sum(1 for r in self._execution_history if r.success)
        times = [r.execution_time_ms for r in self._execution_history]

        return {
            "total_executions": total,
            "successful_executions": successful,
            "failed_executions": total - successful,
            "success_rate": successful / total if total > 0 else 0,
            "avg_execution_time_ms": sum(times) / len(times) if times else 0,
            "min_execution_time_ms": min(times) if times else 0,
            "max_execution_time_ms": max(times) if times else 0,
            "cache_hits": sum(1 for r in self._execution_history if r.from_cache),
        }


__all__ = [
    "ExecutionResult",
    "ExecutionErrorType",
    "BehaviorExecutor",
]
