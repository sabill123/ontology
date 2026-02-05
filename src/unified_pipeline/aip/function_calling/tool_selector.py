"""
Tool Selector (v17.0)

LLM 기반 도구 선택:
- 자연어 요청에서 필요한 도구 선택
- 도구 체인 생성 (의존성 파악)
- DSPy 기반 구조화된 선택

사용 예시:
    selector = ToolSelector(registry, context)
    result = await selector.select_tools(
        "Find the top 10 customers by revenue and their recent orders"
    )
    # result.chain: [Tool1(customer_search), Tool2(order_lookup)]
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, TYPE_CHECKING
from datetime import datetime
from enum import Enum

try:
    import dspy
    DSPY_AVAILABLE = True
except ImportError:
    DSPY_AVAILABLE = False

if TYPE_CHECKING:
    from ...shared_context import SharedContext
    from .tool_registry import ToolRegistry, Tool

logger = logging.getLogger(__name__)


class SelectionConfidence(str, Enum):
    """선택 신뢰도"""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class ToolCall:
    """도구 호출 정보"""
    tool_name: str
    arguments: Dict[str, Any] = field(default_factory=dict)
    depends_on: List[str] = field(default_factory=list)  # 의존하는 이전 도구
    description: str = ""  # 이 호출의 목적

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tool_name": self.tool_name,
            "arguments": self.arguments,
            "depends_on": self.depends_on,
            "description": self.description,
        }


@dataclass
class ToolChain:
    """도구 체인 (실행 계획)"""
    calls: List[ToolCall] = field(default_factory=list)
    reasoning: str = ""
    estimated_steps: int = 0

    @property
    def tool_names(self) -> List[str]:
        return [call.tool_name for call in self.calls]

    @property
    def has_dependencies(self) -> bool:
        return any(call.depends_on for call in self.calls)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "calls": [call.to_dict() for call in self.calls],
            "reasoning": self.reasoning,
            "estimated_steps": self.estimated_steps,
            "tool_names": self.tool_names,
            "has_dependencies": self.has_dependencies,
        }


@dataclass
class ToolSelectionResult:
    """도구 선택 결과"""
    chain: ToolChain
    confidence: SelectionConfidence
    confidence_score: float
    alternatives: List[ToolChain] = field(default_factory=list)

    # 메타데이터
    query: str = ""
    selected_at: str = field(default_factory=lambda: datetime.now().isoformat())
    selection_method: str = "llm"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chain": self.chain.to_dict(),
            "confidence": self.confidence.value,
            "confidence_score": self.confidence_score,
            "alternatives": [a.to_dict() for a in self.alternatives],
            "query": self.query,
            "selected_at": self.selected_at,
            "selection_method": self.selection_method,
        }


# DSPy Signatures (DSPY_AVAILABLE 시)
if DSPY_AVAILABLE:
    class ToolSelectionSignature(dspy.Signature):
        """도구 선택 시그니처"""
        user_request: str = dspy.InputField(
            desc="User's natural language request"
        )
        available_tools: str = dspy.InputField(
            desc="List of available tools with descriptions"
        )
        context_info: str = dspy.InputField(
            desc="Current context (available tables, etc.)"
        )

        selected_tools: str = dspy.OutputField(
            desc="Comma-separated list of tool names to use"
        )
        reasoning: str = dspy.OutputField(
            desc="Explanation of why these tools were selected"
        )
        confidence: str = dspy.OutputField(
            desc="Confidence level: high, medium, or low"
        )

    class ToolArgumentsSignature(dspy.Signature):
        """도구 인자 결정 시그니처"""
        user_request: str = dspy.InputField(
            desc="User's natural language request"
        )
        tool_name: str = dspy.InputField(
            desc="Name of the tool to call"
        )
        tool_schema: str = dspy.InputField(
            desc="Tool parameters schema"
        )
        previous_results: str = dspy.InputField(
            desc="Results from previous tool calls (if any)"
        )

        arguments_json: str = dspy.OutputField(
            desc="JSON object of tool arguments"
        )
        reasoning: str = dspy.OutputField(
            desc="Explanation of argument choices"
        )


class ToolSelector:
    """
    LLM 기반 도구 선택기

    기능:
    - 자연어 요청 분석
    - 적합한 도구 선택
    - 도구 체인 및 의존성 파악
    - 인자 자동 결정
    """

    def __init__(
        self,
        registry: "ToolRegistry",
        context: Optional["SharedContext"] = None,
    ):
        """
        Args:
            registry: 도구 레지스트리
            context: SharedContext
        """
        self.registry = registry
        self.context = context

        # DSPy 모듈 초기화
        if DSPY_AVAILABLE:
            self._init_dspy_modules()

        # 선택 히스토리
        self.selection_history: List[ToolSelectionResult] = []

        logger.info("ToolSelector initialized")

    def _init_dspy_modules(self) -> None:
        """DSPy 모듈 초기화"""
        self.tool_selector = dspy.ChainOfThought(ToolSelectionSignature)
        self.arg_generator = dspy.ChainOfThought(ToolArgumentsSignature)

    async def select_tools(
        self,
        query: str,
        max_tools: int = 5,
        include_alternatives: bool = False,
    ) -> ToolSelectionResult:
        """
        쿼리에 맞는 도구 선택

        Args:
            query: 사용자 자연어 요청
            max_tools: 최대 도구 수
            include_alternatives: 대안 체인 포함 여부

        Returns:
            ToolSelectionResult: 선택 결과
        """
        # 도구 목록 준비
        tool_descriptions = self.registry.get_tool_descriptions()

        # 컨텍스트 정보 준비
        context_info = self._prepare_context_info()

        # DSPy 사용 가능하면 LLM 기반 선택
        if DSPY_AVAILABLE:
            result = await self._select_with_dspy(
                query, tool_descriptions, context_info, max_tools
            )
        else:
            # 폴백: 키워드 기반 선택
            result = self._select_with_keywords(query, max_tools)

        # 대안 생성
        if include_alternatives:
            result.alternatives = self._generate_alternatives(query, result.chain)

        # 히스토리 기록
        self.selection_history.append(result)

        logger.info(
            f"Selected tools for '{query[:50]}...': {result.chain.tool_names}"
        )

        return result

    async def _select_with_dspy(
        self,
        query: str,
        tool_descriptions: str,
        context_info: str,
        max_tools: int,
    ) -> ToolSelectionResult:
        """DSPy 기반 도구 선택"""
        try:
            # 도구 선택
            selection = self.tool_selector(
                user_request=query,
                available_tools=tool_descriptions,
                context_info=context_info,
            )

            # 선택된 도구 파싱
            tool_names = [
                name.strip()
                for name in selection.selected_tools.split(",")
                if name.strip()
            ][:max_tools]

            # 신뢰도 파싱
            confidence_str = selection.confidence.lower()
            if "high" in confidence_str:
                confidence = SelectionConfidence.HIGH
                confidence_score = 0.9
            elif "medium" in confidence_str:
                confidence = SelectionConfidence.MEDIUM
                confidence_score = 0.7
            else:
                confidence = SelectionConfidence.LOW
                confidence_score = 0.5

            # 도구 체인 생성
            chain = await self._build_chain(query, tool_names)
            chain.reasoning = selection.reasoning

            return ToolSelectionResult(
                chain=chain,
                confidence=confidence,
                confidence_score=confidence_score,
                query=query,
                selection_method="dspy",
            )

        except Exception as e:
            logger.error(f"DSPy selection failed: {e}")
            return self._select_with_keywords(query, max_tools)

    def _select_with_keywords(
        self,
        query: str,
        max_tools: int,
    ) -> ToolSelectionResult:
        """키워드 기반 도구 선택 (폴백)"""
        query_lower = query.lower()
        selected = []

        for tool in self.registry.list_tools():
            # 도구 이름이나 설명에 매칭
            if (
                any(word in query_lower for word in tool.name.lower().split("_")) or
                any(word in tool.description.lower() for word in query_lower.split())
            ):
                selected.append(tool.name)

                if len(selected) >= max_tools:
                    break

        # 선택된 도구가 없으면 기본 도구
        if not selected:
            tools = self.registry.list_tools()
            if tools:
                selected = [tools[0].name]

        # 체인 생성
        calls = [
            ToolCall(tool_name=name, description=f"Execute {name}")
            for name in selected
        ]

        chain = ToolChain(
            calls=calls,
            reasoning="Selected based on keyword matching",
            estimated_steps=len(calls),
        )

        return ToolSelectionResult(
            chain=chain,
            confidence=SelectionConfidence.LOW,
            confidence_score=0.4,
            query=query,
            selection_method="keyword",
        )

    async def _build_chain(
        self,
        query: str,
        tool_names: List[str],
    ) -> ToolChain:
        """도구 체인 구축 (의존성 분석)"""
        calls = []

        for i, name in enumerate(tool_names):
            tool = self.registry.get(name)
            if not tool:
                continue

            # 인자 생성
            arguments = await self._generate_arguments(query, tool, calls)

            # 의존성 파악
            depends_on = []
            if i > 0:
                # 이전 도구의 결과에 의존할 수 있음
                # 실제로는 더 정교한 의존성 분석 필요
                prev_call = calls[-1]
                depends_on = [prev_call.tool_name]

            calls.append(ToolCall(
                tool_name=name,
                arguments=arguments,
                depends_on=depends_on,
                description=f"Execute {name} for user request",
            ))

        return ToolChain(
            calls=calls,
            estimated_steps=len(calls),
        )

    async def _generate_arguments(
        self,
        query: str,
        tool: "Tool",
        previous_calls: List[ToolCall],
    ) -> Dict[str, Any]:
        """도구 인자 생성"""
        if not DSPY_AVAILABLE or not tool.parameters:
            return {}

        try:
            # 이전 결과 문자열화
            prev_results = str([c.to_dict() for c in previous_calls]) if previous_calls else "None"

            # 스키마 문자열화
            schema = str([p.to_dict() for p in tool.parameters])

            # DSPy로 인자 생성
            result = self.arg_generator(
                user_request=query,
                tool_name=tool.name,
                tool_schema=schema,
                previous_results=prev_results,
            )

            # JSON 파싱
            import json
            try:
                arguments = json.loads(result.arguments_json)
            except json.JSONDecodeError:
                arguments = {}

            return arguments

        except Exception as e:
            logger.warning(f"Failed to generate arguments for {tool.name}: {e}")
            return {}

    def _prepare_context_info(self) -> str:
        """컨텍스트 정보 준비"""
        if not self.context:
            return "No context available"

        info_parts = []

        # 테이블 정보
        if self.context.tables:
            tables = list(self.context.tables.keys())
            info_parts.append(f"Available tables: {', '.join(tables)}")

        # 현재 파이프라인 단계
        if self.context.current_phase:
            info_parts.append(f"Current phase: {self.context.current_phase}")

        return "\n".join(info_parts) if info_parts else "Empty context"

    def _generate_alternatives(
        self,
        query: str,
        main_chain: ToolChain,
    ) -> List[ToolChain]:
        """대안 체인 생성"""
        alternatives = []

        # 간단한 대안: 개별 도구 체인
        for call in main_chain.calls:
            alt_chain = ToolChain(
                calls=[call],
                reasoning=f"Single tool alternative: {call.tool_name}",
                estimated_steps=1,
            )
            alternatives.append(alt_chain)

        return alternatives[:3]  # 최대 3개

    def get_selection_summary(self) -> Dict[str, Any]:
        """선택 요약"""
        method_counts = {}
        confidence_counts = {}

        for result in self.selection_history:
            method = result.selection_method
            method_counts[method] = method_counts.get(method, 0) + 1

            conf = result.confidence.value
            confidence_counts[conf] = confidence_counts.get(conf, 0) + 1

        return {
            "total_selections": len(self.selection_history),
            "method_counts": method_counts,
            "confidence_counts": confidence_counts,
            "recent_selections": [
                r.to_dict() for r in self.selection_history[-5:]
            ],
        }
