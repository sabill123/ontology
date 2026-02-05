"""
Tool Registry (v17.0)

도구 등록 및 관리:
- @tool 데코레이터로 함수 등록
- 도구 메타데이터 관리
- 도구 실행

사용 예시:
    registry = ToolRegistry()

    @registry.tool(
        name="search_customers",
        description="Search customers by various criteria",
    )
    def search_customers(
        query: str,
        limit: int = 10,
    ) -> List[Dict]:
        ...

    result = await registry.execute("search_customers", query="top revenue")
"""

import logging
import inspect
import asyncio
from dataclasses import dataclass, field
from typing import (
    Dict, List, Any, Optional, Callable, Union, Type,
    get_type_hints, TYPE_CHECKING,
)
from datetime import datetime
from enum import Enum
import functools

if TYPE_CHECKING:
    from ...shared_context import SharedContext

logger = logging.getLogger(__name__)


class ParameterType(str, Enum):
    """파라미터 타입"""
    STRING = "string"
    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"
    ARRAY = "array"
    OBJECT = "object"
    ANY = "any"


@dataclass
class ToolParameter:
    """도구 파라미터 정의"""
    name: str
    type: ParameterType
    description: str = ""
    required: bool = True
    default: Any = None
    enum: Optional[List[Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "name": self.name,
            "type": self.type.value,
            "description": self.description,
            "required": self.required,
        }
        if self.default is not None:
            result["default"] = self.default
        if self.enum:
            result["enum"] = self.enum
        return result

    @classmethod
    def from_type_hint(
        cls,
        name: str,
        type_hint: Type,
        default: Any = inspect.Parameter.empty,
        description: str = "",
    ) -> "ToolParameter":
        """타입 힌트에서 파라미터 생성"""
        # 타입 매핑
        type_mapping = {
            str: ParameterType.STRING,
            int: ParameterType.INTEGER,
            float: ParameterType.FLOAT,
            bool: ParameterType.BOOLEAN,
            list: ParameterType.ARRAY,
            dict: ParameterType.OBJECT,
        }

        param_type = type_mapping.get(type_hint, ParameterType.ANY)
        required = default is inspect.Parameter.empty

        return cls(
            name=name,
            type=param_type,
            description=description,
            required=required,
            default=None if required else default,
        )


@dataclass
class ToolResult:
    """도구 실행 결과"""
    tool_name: str
    success: bool
    result: Any = None
    error: Optional[str] = None
    execution_time_ms: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tool_name": self.tool_name,
            "success": self.success,
            "result": self.result,
            "error": self.error,
            "execution_time_ms": self.execution_time_ms,
            "timestamp": self.timestamp,
        }


@dataclass
class Tool:
    """도구 정의"""
    name: str
    description: str
    function: Callable
    parameters: List[ToolParameter] = field(default_factory=list)

    # 메타데이터
    category: str = "general"
    tags: List[str] = field(default_factory=list)
    is_async: bool = False
    requires_context: bool = False

    # 사용 통계
    call_count: int = 0
    total_execution_time_ms: float = 0.0
    success_count: int = 0
    error_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": [p.to_dict() for p in self.parameters],
            "category": self.category,
            "tags": self.tags,
            "is_async": self.is_async,
            "requires_context": self.requires_context,
            "call_count": self.call_count,
            "avg_execution_time_ms": (
                self.total_execution_time_ms / self.call_count
                if self.call_count > 0 else 0
            ),
            "success_rate": (
                self.success_count / self.call_count
                if self.call_count > 0 else 0
            ),
        }

    def to_llm_schema(self) -> Dict[str, Any]:
        """LLM용 스키마 생성 (OpenAI function calling 형식)"""
        properties = {}
        required = []

        for param in self.parameters:
            properties[param.name] = {
                "type": param.type.value,
                "description": param.description,
            }
            if param.enum:
                properties[param.name]["enum"] = param.enum
            if param.required:
                required.append(param.name)

        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        }


class ToolRegistry:
    """
    도구 레지스트리

    기능:
    - @tool 데코레이터로 함수 등록
    - 도구 메타데이터 관리
    - 도구 실행 및 통계 추적
    """

    def __init__(
        self,
        context: Optional["SharedContext"] = None,
    ):
        """
        Args:
            context: SharedContext (도구에 전달용)
        """
        self.context = context
        self._tools: Dict[str, Tool] = {}

        logger.info("ToolRegistry initialized")

    def tool(
        self,
        name: Optional[str] = None,
        description: str = "",
        category: str = "general",
        tags: Optional[List[str]] = None,
        parameters: Optional[List[ToolParameter]] = None,
        requires_context: bool = False,
    ) -> Callable:
        """
        도구 등록 데코레이터

        Args:
            name: 도구 이름 (기본: 함수명)
            description: 설명
            category: 카테고리
            tags: 태그 목록
            parameters: 파라미터 정의 (자동 추론도 가능)
            requires_context: SharedContext 필요 여부

        Returns:
            데코레이터 함수
        """
        def decorator(func: Callable) -> Callable:
            tool_name = name or func.__name__
            tool_description = description or func.__doc__ or ""

            # 파라미터 자동 추론
            if parameters:
                tool_params = parameters
            else:
                tool_params = self._extract_parameters(func)

            # 비동기 여부 확인
            is_async = asyncio.iscoroutinefunction(func)

            # Tool 객체 생성
            tool = Tool(
                name=tool_name,
                description=tool_description.strip(),
                function=func,
                parameters=tool_params,
                category=category,
                tags=tags or [],
                is_async=is_async,
                requires_context=requires_context,
            )

            # 등록
            self._tools[tool_name] = tool

            logger.info(f"Registered tool: {tool_name}")

            # 원본 함수 반환
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                return func(*args, **kwargs)

            return wrapper

        return decorator

    def register(
        self,
        func: Callable,
        name: Optional[str] = None,
        description: str = "",
        **kwargs,
    ) -> None:
        """
        함수를 도구로 직접 등록 (데코레이터 없이)

        Args:
            func: 등록할 함수
            name: 도구 이름
            description: 설명
            **kwargs: 추가 옵션
        """
        self.tool(name=name, description=description, **kwargs)(func)

    def unregister(self, name: str) -> bool:
        """도구 등록 해제"""
        if name in self._tools:
            del self._tools[name]
            logger.info(f"Unregistered tool: {name}")
            return True
        return False

    def get(self, name: str) -> Optional[Tool]:
        """도구 조회"""
        return self._tools.get(name)

    def list_tools(
        self,
        category: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ) -> List[Tool]:
        """
        도구 목록 조회

        Args:
            category: 카테고리 필터
            tags: 태그 필터 (OR 조건)

        Returns:
            Tool 목록
        """
        tools = list(self._tools.values())

        if category:
            tools = [t for t in tools if t.category == category]

        if tags:
            tools = [
                t for t in tools
                if any(tag in t.tags for tag in tags)
            ]

        return tools

    async def execute(
        self,
        name: str,
        **kwargs,
    ) -> ToolResult:
        """
        도구 실행

        Args:
            name: 도구 이름
            **kwargs: 도구 인자

        Returns:
            ToolResult: 실행 결과
        """
        tool = self.get(name)

        if not tool:
            return ToolResult(
                tool_name=name,
                success=False,
                error=f"Tool not found: {name}",
            )

        start_time = datetime.now()

        try:
            # context 주입
            if tool.requires_context and self.context:
                kwargs["context"] = self.context

            # 실행
            if tool.is_async:
                result = await tool.function(**kwargs)
            else:
                result = tool.function(**kwargs)

            # 통계 업데이트
            execution_time = (datetime.now() - start_time).total_seconds() * 1000
            tool.call_count += 1
            tool.success_count += 1
            tool.total_execution_time_ms += execution_time

            return ToolResult(
                tool_name=name,
                success=True,
                result=result,
                execution_time_ms=execution_time,
            )

        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds() * 1000
            tool.call_count += 1
            tool.error_count += 1
            tool.total_execution_time_ms += execution_time

            logger.error(f"Tool execution failed: {name} - {e}")

            return ToolResult(
                tool_name=name,
                success=False,
                error=str(e),
                execution_time_ms=execution_time,
            )

    def _extract_parameters(self, func: Callable) -> List[ToolParameter]:
        """함수 시그니처에서 파라미터 추출"""
        params = []
        sig = inspect.signature(func)
        hints = get_type_hints(func) if hasattr(func, "__annotations__") else {}

        for name, param in sig.parameters.items():
            if name in ("self", "cls", "context"):
                continue

            type_hint = hints.get(name, Any)
            default = param.default

            tool_param = ToolParameter.from_type_hint(
                name=name,
                type_hint=type_hint,
                default=default,
            )
            params.append(tool_param)

        return params

    def get_llm_schemas(self) -> List[Dict[str, Any]]:
        """모든 도구의 LLM 스키마 반환"""
        return [tool.to_llm_schema() for tool in self._tools.values()]

    def get_tool_descriptions(self) -> str:
        """도구 설명 문자열 생성 (LLM 컨텍스트용)"""
        descriptions = []

        for tool in self._tools.values():
            param_strs = []
            for p in tool.parameters:
                req = "*" if p.required else ""
                param_strs.append(f"  - {p.name}{req}: {p.type.value} - {p.description}")

            tool_desc = f"""
Tool: {tool.name}
Description: {tool.description}
Category: {tool.category}
Parameters:
{chr(10).join(param_strs) if param_strs else "  (none)"}
"""
            descriptions.append(tool_desc.strip())

        return "\n\n".join(descriptions)

    def get_registry_summary(self) -> Dict[str, Any]:
        """레지스트리 요약"""
        categories = {}
        for tool in self._tools.values():
            if tool.category not in categories:
                categories[tool.category] = 0
            categories[tool.category] += 1

        return {
            "total_tools": len(self._tools),
            "categories": categories,
            "tools": {
                name: tool.to_dict()
                for name, tool in self._tools.items()
            },
        }


# 기본 도구들 (예시)
def create_default_registry(
    context: Optional["SharedContext"] = None,
) -> ToolRegistry:
    """기본 도구가 등록된 레지스트리 생성"""
    registry = ToolRegistry(context)

    @registry.tool(
        name="get_table_info",
        description="Get information about a table including schema and sample data",
        category="data",
        tags=["schema", "metadata"],
        requires_context=True,
    )
    def get_table_info(table_name: str, context: "SharedContext") -> Dict[str, Any]:
        """테이블 정보 조회"""
        if table_name not in context.tables:
            return {"error": f"Table not found: {table_name}"}
        return context.tables[table_name]

    @registry.tool(
        name="list_tables",
        description="List all available tables",
        category="data",
        tags=["schema", "discovery"],
        requires_context=True,
    )
    def list_tables(context: "SharedContext") -> List[str]:
        """테이블 목록 조회"""
        return list(context.tables.keys())

    @registry.tool(
        name="get_column_stats",
        description="Get statistics for a specific column",
        category="analytics",
        tags=["statistics", "profiling"],
        requires_context=True,
    )
    def get_column_stats(
        table_name: str,
        column_name: str,
        context: "SharedContext",
    ) -> Dict[str, Any]:
        """컬럼 통계 조회"""
        if table_name not in context.tables:
            return {"error": f"Table not found: {table_name}"}

        table_info = context.tables[table_name]
        columns = table_info.get("columns", {})

        if column_name not in columns:
            return {"error": f"Column not found: {column_name}"}

        return columns[column_name]

    @registry.tool(
        name="search_data",
        description="Search for data matching a pattern",
        category="search",
        tags=["query", "filter"],
    )
    def search_data(
        pattern: str,
        table: Optional[str] = None,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """데이터 검색 (예시)"""
        # 실제 구현 필요
        return [{"message": f"Search for '{pattern}' in {table or 'all tables'}"}]

    return registry
