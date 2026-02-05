"""
Chain Executor (v17.0)

도구 체인 실행:
- 의존성 기반 순차 실행
- 병렬 실행 지원
- 결과 집계 및 에러 처리

사용 예시:
    executor = ChainExecutor(registry, context)
    result = await executor.execute(chain)

    for step in result.steps:
        print(f"{step.tool_name}: {step.result}")
"""

import logging
import asyncio
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, TYPE_CHECKING
from datetime import datetime
from enum import Enum

if TYPE_CHECKING:
    from ...shared_context import SharedContext
    from .tool_registry import ToolRegistry, ToolResult
    from .tool_selector import ToolChain, ToolCall

logger = logging.getLogger(__name__)


class ExecutionStatus(str, Enum):
    """실행 상태"""
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class ExecutionStep:
    """실행 단계"""
    step_id: int
    tool_name: str
    status: ExecutionStatus = ExecutionStatus.PENDING
    arguments: Dict[str, Any] = field(default_factory=dict)
    result: Any = None
    error: Optional[str] = None

    # 타이밍
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    execution_time_ms: float = 0.0

    # 의존성
    depends_on: List[str] = field(default_factory=list)
    dependency_results: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step_id": self.step_id,
            "tool_name": self.tool_name,
            "status": self.status.value,
            "arguments": self.arguments,
            "result": self.result,
            "error": self.error,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "execution_time_ms": self.execution_time_ms,
            "depends_on": self.depends_on,
        }


@dataclass
class ExecutionResult:
    """체인 실행 결과"""
    success: bool
    steps: List[ExecutionStep] = field(default_factory=list)
    final_result: Any = None

    # 통계
    total_steps: int = 0
    successful_steps: int = 0
    failed_steps: int = 0
    skipped_steps: int = 0

    # 타이밍
    started_at: str = field(default_factory=lambda: datetime.now().isoformat())
    completed_at: Optional[str] = None
    total_execution_time_ms: float = 0.0

    # 에러 정보
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "steps": [s.to_dict() for s in self.steps],
            "final_result": self.final_result,
            "total_steps": self.total_steps,
            "successful_steps": self.successful_steps,
            "failed_steps": self.failed_steps,
            "skipped_steps": self.skipped_steps,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "total_execution_time_ms": self.total_execution_time_ms,
            "errors": self.errors,
        }


class ChainExecutor:
    """
    도구 체인 실행기

    기능:
    - 의존성 기반 순차/병렬 실행
    - 결과 전달 (이전 도구 결과 → 다음 도구 인자)
    - 에러 처리 및 롤백
    - Evidence Chain 기록
    """

    def __init__(
        self,
        registry: "ToolRegistry",
        context: Optional["SharedContext"] = None,
        max_parallel: int = 3,
        stop_on_error: bool = False,
    ):
        """
        Args:
            registry: 도구 레지스트리
            context: SharedContext
            max_parallel: 최대 병렬 실행 수
            stop_on_error: 에러 시 중단 여부
        """
        self.registry = registry
        self.context = context
        self.max_parallel = max_parallel
        self.stop_on_error = stop_on_error

        # 실행 히스토리
        self.execution_history: List[ExecutionResult] = []

        logger.info("ChainExecutor initialized")

    async def execute(
        self,
        chain: "ToolChain",
        initial_context: Optional[Dict[str, Any]] = None,
    ) -> ExecutionResult:
        """
        도구 체인 실행

        Args:
            chain: 실행할 도구 체인
            initial_context: 초기 컨텍스트 (첫 도구에 전달)

        Returns:
            ExecutionResult: 실행 결과
        """
        start_time = datetime.now()

        result = ExecutionResult(
            success=True,
            total_steps=len(chain.calls),
        )

        # 실행 단계 초기화
        steps: Dict[str, ExecutionStep] = {}
        for i, call in enumerate(chain.calls):
            step = ExecutionStep(
                step_id=i,
                tool_name=call.tool_name,
                arguments=call.arguments.copy(),
                depends_on=call.depends_on,
            )
            steps[call.tool_name] = step
            result.steps.append(step)

        # 결과 저장소 (의존성 해결용)
        results_map: Dict[str, Any] = {}

        if initial_context:
            results_map["_initial"] = initial_context

        # 의존성 기반 실행
        executed = set()
        while len(executed) < len(chain.calls):
            # 실행 가능한 도구 찾기
            ready = []
            for call in chain.calls:
                if call.tool_name in executed:
                    continue

                # 모든 의존성이 완료되었는지 확인
                deps_met = all(
                    dep in executed for dep in call.depends_on
                )
                if deps_met:
                    ready.append(call)

            if not ready:
                # 데드락 방지
                logger.error("Deadlock detected in chain execution")
                break

            # 병렬 실행 (최대 max_parallel개)
            batch = ready[:self.max_parallel]

            tasks = [
                self._execute_step(
                    call,
                    steps[call.tool_name],
                    results_map,
                )
                for call in batch
            ]

            batch_results = await asyncio.gather(*tasks, return_exceptions=True)

            # 결과 처리
            for call, step_result in zip(batch, batch_results):
                step = steps[call.tool_name]
                executed.add(call.tool_name)

                if isinstance(step_result, Exception):
                    step.status = ExecutionStatus.FAILED
                    step.error = str(step_result)
                    result.failed_steps += 1
                    result.errors.append(f"{call.tool_name}: {step_result}")

                    if self.stop_on_error:
                        result.success = False
                        break
                else:
                    step.status = ExecutionStatus.SUCCESS
                    step.result = step_result
                    results_map[call.tool_name] = step_result
                    result.successful_steps += 1

            if not result.success:
                break

        # 최종 결과 (마지막 성공 단계의 결과)
        for step in reversed(result.steps):
            if step.status == ExecutionStatus.SUCCESS:
                result.final_result = step.result
                break

        # 실패 여부 갱신
        result.success = result.failed_steps == 0

        # 스킵된 단계 카운트
        result.skipped_steps = (
            result.total_steps - result.successful_steps - result.failed_steps
        )

        # 타이밍
        end_time = datetime.now()
        result.completed_at = end_time.isoformat()
        result.total_execution_time_ms = (end_time - start_time).total_seconds() * 1000

        # Evidence Chain 기록
        self._record_to_evidence_chain(chain, result)

        # 히스토리 기록
        self.execution_history.append(result)

        logger.info(
            f"Chain execution completed: {result.successful_steps}/{result.total_steps} "
            f"successful in {result.total_execution_time_ms:.0f}ms"
        )

        return result

    async def _execute_step(
        self,
        call: "ToolCall",
        step: ExecutionStep,
        results_map: Dict[str, Any],
    ) -> Any:
        """단일 단계 실행"""
        step.status = ExecutionStatus.RUNNING
        step.started_at = datetime.now().isoformat()

        try:
            # 의존성 결과 주입
            arguments = call.arguments.copy()
            for dep in call.depends_on:
                if dep in results_map:
                    step.dependency_results[dep] = results_map[dep]
                    # 이전 결과를 특별 인자로 전달
                    arguments["_previous_result"] = results_map[dep]

            # 도구 실행
            tool_result = await self.registry.execute(
                call.tool_name,
                **arguments,
            )

            step.completed_at = datetime.now().isoformat()

            if step.started_at:
                start = datetime.fromisoformat(step.started_at)
                end = datetime.fromisoformat(step.completed_at)
                step.execution_time_ms = (end - start).total_seconds() * 1000

            if not tool_result.success:
                raise Exception(tool_result.error or "Tool execution failed")

            return tool_result.result

        except Exception as e:
            step.completed_at = datetime.now().isoformat()
            if step.started_at:
                start = datetime.fromisoformat(step.started_at)
                end = datetime.fromisoformat(step.completed_at)
                step.execution_time_ms = (end - start).total_seconds() * 1000
            raise

    def _record_to_evidence_chain(
        self,
        chain: "ToolChain",
        result: ExecutionResult,
    ) -> None:
        """Evidence Chain에 실행 기록"""
        if not self.context or not self.context.evidence_chain:
            return

        try:
            from ...autonomous.evidence_chain import EvidenceType

            evidence_type = getattr(
                EvidenceType,
                "TOOL_EXECUTION",
                EvidenceType.QUALITY_ASSESSMENT
            )

            self.context.evidence_chain.add_block(
                phase="function_calling",
                agent="ChainExecutor",
                evidence_type=evidence_type,
                finding=f"Executed tool chain: {chain.tool_names}",
                reasoning=chain.reasoning,
                conclusion=(
                    f"Success: {result.successful_steps}/{result.total_steps} steps"
                    if result.success
                    else f"Failed: {result.errors}"
                ),
                metrics={
                    "total_steps": result.total_steps,
                    "successful_steps": result.successful_steps,
                    "failed_steps": result.failed_steps,
                    "execution_time_ms": result.total_execution_time_ms,
                },
                confidence=0.95 if result.success else 0.3,
            )
        except Exception as e:
            logger.warning(f"Failed to record to evidence chain: {e}")

    async def execute_single(
        self,
        tool_name: str,
        **arguments,
    ) -> ExecutionStep:
        """단일 도구 실행 (체인 없이)"""
        from .tool_selector import ToolCall, ToolChain

        call = ToolCall(tool_name=tool_name, arguments=arguments)
        chain = ToolChain(calls=[call])

        result = await self.execute(chain)

        return result.steps[0] if result.steps else ExecutionStep(
            step_id=0,
            tool_name=tool_name,
            status=ExecutionStatus.FAILED,
            error="Execution failed",
        )

    def get_execution_summary(self) -> Dict[str, Any]:
        """실행 요약"""
        total_executions = len(self.execution_history)
        successful = sum(1 for r in self.execution_history if r.success)
        total_steps = sum(r.total_steps for r in self.execution_history)
        total_time = sum(r.total_execution_time_ms for r in self.execution_history)

        return {
            "total_executions": total_executions,
            "successful_executions": successful,
            "success_rate": successful / total_executions if total_executions > 0 else 0,
            "total_steps_executed": total_steps,
            "avg_steps_per_execution": total_steps / total_executions if total_executions > 0 else 0,
            "total_execution_time_ms": total_time,
            "avg_execution_time_ms": total_time / total_executions if total_executions > 0 else 0,
            "recent_executions": [
                r.to_dict() for r in self.execution_history[-5:]
            ],
        }
