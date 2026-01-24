"""
Instructor 기반 구조화된 LLM 호출

Pydantic 모델로 타입 안전한 응답을 보장합니다.
- 자동 검증 및 재시도
- 기존 llm.py와 호환

v1.0: 초기 구현
"""

import logging
import time
from typing import Type, TypeVar, Optional, List, Dict, Any

from pydantic import BaseModel

from .llm import (
    LETSUR_BASE_URL,
    LETSUR_API_KEY,
    DEFAULT_MODEL,
    get_llm_agent_context,
    get_llm_thinking_context,
    clear_llm_thinking_context,
)

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

# Instructor 클라이언트 캐시
_instructor_client = None


def get_instructor_client():
    """Instructor로 패치된 OpenAI 클라이언트 반환 (싱글톤)"""
    global _instructor_client

    if _instructor_client is None:
        try:
            import instructor
            from openai import OpenAI

            base_client = OpenAI(
                base_url=LETSUR_BASE_URL,
                api_key=LETSUR_API_KEY,
            )
            _instructor_client = instructor.from_openai(base_client)
            logger.info("Instructor client initialized successfully")
        except ImportError:
            logger.error("instructor 패키지가 설치되지 않았습니다. pip install instructor")
            raise
        except Exception as e:
            logger.error(f"Instructor 클라이언트 초기화 실패: {e}")
            raise

    return _instructor_client


def call_llm_structured(
    response_model: Type[T],
    messages: List[Dict[str, str]],
    model: str = None,
    temperature: float = 0.2,
    max_retries: int = 3,
    client=None,
    agent_name: Optional[str] = None,
) -> T:
    """
    구조화된 LLM 호출 - Pydantic 모델로 응답 보장

    Args:
        response_model: 응답 Pydantic 모델 클래스
        messages: 메시지 리스트
        model: 모델명 (기본값: DEFAULT_MODEL)
        temperature: 온도
        max_retries: 검증 실패 시 재시도 횟수
        client: Instructor 클라이언트 (선택)
        agent_name: 로깅용 에이전트 이름

    Returns:
        response_model 인스턴스

    Raises:
        ValidationError: 재시도 후에도 검증 실패 시

    Example:
        class FKAnalysis(BaseModel):
            is_relationship: bool
            confidence: float
            reasoning: str

        result = call_llm_structured(
            response_model=FKAnalysis,
            messages=[
                {"role": "system", "content": "You are a data analyst."},
                {"role": "user", "content": "Analyze this column..."}
            ]
        )
        # result.is_relationship, result.confidence 등 타입 안전하게 접근
    """
    if client is None:
        client = get_instructor_client()

    log_agent_name = agent_name or get_llm_agent_context()
    start_time = time.time()

    try:
        result = client.chat.completions.create(
            model=model or DEFAULT_MODEL,
            messages=messages,
            response_model=response_model,
            temperature=temperature,
            max_retries=max_retries,
        )

        latency_ms = (time.time() - start_time) * 1000
        logger.debug(
            f"[Instructor] {log_agent_name} -> {response_model.__name__} "
            f"(latency={latency_ms:.0f}ms, retries<={max_retries})"
        )

        # JobLogger 기록 시도
        _log_instructor_call(
            agent_name=log_agent_name,
            model=model or DEFAULT_MODEL,
            response_model_name=response_model.__name__,
            latency_ms=latency_ms,
            success=True,
        )

        return result

    except Exception as e:
        latency_ms = (time.time() - start_time) * 1000
        logger.error(f"[Instructor] {log_agent_name} failed: {e}")

        _log_instructor_call(
            agent_name=log_agent_name,
            model=model or DEFAULT_MODEL,
            response_model_name=response_model.__name__,
            latency_ms=latency_ms,
            success=False,
            error=str(e),
        )

        raise


def call_llm_structured_with_context(
    response_model: Type[T],
    instruction: str,
    context_data: Dict[str, Any],
    system_prompt: str = "You are a data analysis expert. Provide accurate and structured responses.",
    model: str = None,
    temperature: float = 0.2,
    max_retries: int = 3,
) -> T:
    """
    컨텍스트와 함께 구조화된 LLM 호출

    Args:
        response_model: 응답 Pydantic 모델
        instruction: 지시사항
        context_data: 컨텍스트 데이터
        system_prompt: 시스템 프롬프트
        model: 모델명
        temperature: 온도
        max_retries: 재시도 횟수

    Returns:
        response_model 인스턴스
    """
    import json

    # 컨텍스트 포맷팅
    context_str = json.dumps(context_data, ensure_ascii=False, indent=2, default=str)

    user_prompt = f"""## Context
{context_str}

## Task
{instruction}

Analyze carefully and provide your response in the requested format."""

    return call_llm_structured(
        response_model=response_model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        model=model,
        temperature=temperature,
        max_retries=max_retries,
    )


def _log_instructor_call(
    agent_name: str,
    model: str,
    response_model_name: str,
    latency_ms: float,
    success: bool,
    error: Optional[str] = None,
):
    """Instructor 호출을 JobLogger에 기록"""
    try:
        from .job_logger import get_job_logger

        job_logger = get_job_logger()
        if job_logger.is_active():
            thinking_ctx = get_llm_thinking_context()
            job_logger.log_llm_call(
                agent_name=agent_name,
                model=model,
                messages=[{"note": f"Instructor call -> {response_model_name}"}],
                response=f"Structured response: {response_model_name}",
                prompt_tokens=0,  # Instructor doesn't expose this easily
                completion_tokens=0,
                latency_ms=latency_ms,
                success=success,
                error=error,
                purpose=thinking_ctx.get("purpose"),
                thinking_before=thinking_ctx.get("thinking_before"),
            )
            clear_llm_thinking_context()
    except ImportError:
        pass
    except Exception:
        pass


# === 편의 함수들 ===

def structured_completion(
    prompt: str,
    response_model: Type[T],
    system_prompt: str = "You are a helpful assistant.",
    model: str = None,
    temperature: float = 0.2,
) -> T:
    """
    간단한 구조화 완성 (편의 함수)

    Args:
        prompt: 사용자 프롬프트
        response_model: 응답 모델
        system_prompt: 시스템 프롬프트
        model: 모델명
        temperature: 온도

    Returns:
        response_model 인스턴스
    """
    return call_llm_structured(
        response_model=response_model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        model=model,
        temperature=temperature,
    )
