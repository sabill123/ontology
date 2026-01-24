"""Utility modules"""

from .logger import logger, console, print_panel, print_status, log_step
from .job_logger import (
    JobLogger,
    get_job_logger,
    start_job,
    end_job,
    LogLevel,
    PhaseType,
    LLMCallRecord,
    AgentExecutionRecord,
    JobMetadata,
    # v5.1: 사고 과정 기록
    ThinkingType,
    ThinkingRecord,
    ReasoningChain,
)
from .llm import (
    set_llm_agent_context,
    get_llm_agent_context,
    clear_llm_agent_context,
    set_llm_thinking_context,
    get_llm_thinking_context,
    clear_llm_thinking_context,
)

__all__ = [
    # Legacy logger
    "logger",
    "console",
    "print_panel",
    "print_status",
    "log_step",
    # Job Logger (v5.0)
    "JobLogger",
    "get_job_logger",
    "start_job",
    "end_job",
    "LogLevel",
    "PhaseType",
    "LLMCallRecord",
    "AgentExecutionRecord",
    "JobMetadata",
    # v5.1: 사고 과정 기록
    "ThinkingType",
    "ThinkingRecord",
    "ReasoningChain",
    # LLM 컨텍스트 관리
    "set_llm_agent_context",
    "get_llm_agent_context",
    "clear_llm_agent_context",
    "set_llm_thinking_context",
    "get_llm_thinking_context",
    "clear_llm_thinking_context",
]
