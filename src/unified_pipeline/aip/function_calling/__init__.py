"""
Function Calling Module (v17.0)

Palantir AIP 스타일 LLM 도구 호출:
- 도구 등록 및 관리
- LLM 기반 도구 선택
- 체인 실행 (의존성 기반)

사용 예시:
    registry = ToolRegistry()

    @registry.tool("search_customers")
    def search_customers(query: str) -> List[Dict]:
        ...

    selector = ToolSelector(registry, context)
    chain = await selector.select_tools("Find top customers by revenue")

    executor = ChainExecutor(registry)
    result = await executor.execute(chain)
"""

from .tool_registry import (
    ToolRegistry,
    Tool,
    ToolParameter,
    ToolResult,
)

from .tool_selector import (
    ToolSelector,
    ToolSelectionResult,
    ToolChain,
)

from .chain_executor import (
    ChainExecutor,
    ExecutionResult,
    ExecutionStep,
)

__all__ = [
    # Tool Registry
    "ToolRegistry",
    "Tool",
    "ToolParameter",
    "ToolResult",
    # Tool Selector
    "ToolSelector",
    "ToolSelectionResult",
    "ToolChain",
    # Chain Executor
    "ChainExecutor",
    "ExecutionResult",
    "ExecutionStep",
]
