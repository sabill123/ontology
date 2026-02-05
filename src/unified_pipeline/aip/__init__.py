"""
AIP (AI-Powered Intelligence Platform) Module (v17.0)

Palantir AIP 스타일 기능 구현:
1. NL2SQL: 자연어 → SQL 변환 및 실행
2. Function Calling: LLM 기반 도구 선택 및 체인 실행
3. Real-time Reasoning: 스트리밍 데이터 실시간 추론

핵심 개념:
- SharedContext와 완전 통합
- Evidence Chain으로 모든 결정 추적
- DSPy 기반 구조화된 LLM 호출
"""

from .nl2sql import (
    NL2SQLEngine,
    NL2SQLSignature,
    JoinPathSignature,
    SchemaContext,
    QueryResult,
)

from .function_calling import (
    ToolRegistry,
    Tool,
    ToolParameter,
    ToolResult,
    ToolSelector,
    ToolSelectionResult,
    ToolChain,
    ChainExecutor,
    ExecutionResult,
    ExecutionStep,
)

from .realtime import (
    StreamingReasoner,
    StreamEvent,
    StreamInsight,
    EventTrigger,
    ThresholdTrigger,
    PatternTrigger,
    AnomalyTrigger,
    TriggerResult,
)

__all__ = [
    # NL2SQL
    "NL2SQLEngine",
    "NL2SQLSignature",
    "JoinPathSignature",
    "SchemaContext",
    "QueryResult",
    # Function Calling
    "ToolRegistry",
    "Tool",
    "ToolParameter",
    "ToolResult",
    "ToolSelector",
    "ToolSelectionResult",
    "ToolChain",
    "ChainExecutor",
    "ExecutionResult",
    "ExecutionStep",
    # Real-time Reasoning
    "StreamingReasoner",
    "StreamEvent",
    "StreamInsight",
    "EventTrigger",
    "ThresholdTrigger",
    "PatternTrigger",
    "AnomalyTrigger",
    "TriggerResult",
]
