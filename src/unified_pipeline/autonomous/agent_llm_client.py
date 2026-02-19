"""
Agent LLM Client

에이전트의 LLM 호출 책임을 분리 (SRP)

책임:
- LLM 호출 (chat_completion)
- Few-shot learning enrichment
- Agent Bus 컨텍스트 주입
- v17 서비스 컨텍스트 주입
- ASI 메트릭 기록
"""

import asyncio
import logging
import re
from typing import Dict, List, Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .base import AutonomousAgent

logger = logging.getLogger(__name__)


class AgentLLMClient:
    """
    에이전트 LLM 클라이언트

    AutonomousAgent의 LLM 호출 관련 책임을 분리합니다.
    에이전트 참조를 통해 필요한 상태에 접근합니다.
    """

    def __init__(self, agent: "AutonomousAgent"):
        self._agent = agent

    async def call_llm(
        self,
        user_prompt: str,
        temperature: float = 0.3,
        max_tokens: int = 2000,
        purpose: str = None,
        thinking_before: str = None,
        model_override: str = None,
        seed: int = 42,
    ) -> str:
        """
        LLM 호출

        Args:
            user_prompt: 사용자 프롬프트
            temperature: 온도
            max_tokens: 최대 토큰
            purpose: LLM 호출 목적
            thinking_before: 호출 전 에이전트의 사고 과정
            model_override: 모델 타입 직접 지정
            seed: 재현성 보장을 위한 시드

        Returns:
            LLM 응답
        """
        agent = self._agent

        model = self._resolve_model(model_override)

        if agent.llm_client is None:
            logger.debug(f"LLM client not available, returning empty response for {agent.agent_type}")
            return "{}"

        # v28.6: max_tokens를 모델별 최대 output 한도로 자동 설정 (비용 무관, 품질 우선)
        from ..model_config import get_model_spec
        model_output_max = get_model_spec(model).get("output_max", 32_000)
        effective_max_tokens = model_output_max

        try:
            from ...common.utils.llm import chat_completion, set_llm_thinking_context

            if purpose or thinking_before:
                set_llm_thinking_context(purpose=purpose, thinking_before=thinking_before)

            if agent.current_todo:
                agent.event_emitter.agent_thinking(
                    agent_id=agent.agent_id,
                    todo_id=agent.current_todo.todo_id,
                    thinking=user_prompt[:500],
                    phase=agent.phase,
                )

            enriched_prompt = self._enrich_with_fewshot(user_prompt)
            bus_context = await self._get_bus_context()
            v17_context = self._get_v17_context()

            final_prompt = enriched_prompt + bus_context + v17_context

            response = await asyncio.to_thread(
                chat_completion,
                model=model,
                messages=[
                    {"role": "system", "content": agent.get_full_system_prompt()},
                    {"role": "user", "content": final_prompt},
                ],
                max_tokens=effective_max_tokens,
                temperature=temperature,
                client=agent.llm_client,
                seed=seed,
            )

            agent.metrics.llm_calls += 1
            self._record_asi_metrics(response)

            return response

        except Exception as e:
            logger.warning(f"LLM call failed: {e}, returning empty response")
            return "{}"

    async def call_llm_with_context(
        self,
        instruction: str,
        context_data: Dict[str, Any],
        temperature: float = 0.3,
        max_tokens: int = 2000,
        purpose: str = None,
        thinking_before: str = None,
        request_completion_signal: bool = True,
        model_override: str = None,
    ) -> str:
        """컨텍스트와 함께 LLM 호출"""
        context_str = self._format_context(context_data)

        completion_signal_suffix = (
            self._agent._get_completion_signal_prompt_suffix()
            if request_completion_signal
            else ""
        )

        user_prompt = f"""## Context
{context_str}

---

## Task
{instruction}

## Response Format
Please provide your analysis in a structured format:

[CONFIDENCE: 0.XX] - Your confidence level (0.0-1.0)

[ANALYSIS]
Your detailed analysis here...

[KEY_FINDINGS]
- Finding 1
- Finding 2
...

[RECOMMENDATIONS]
- Recommendation 1
- Recommendation 2
...

[OUTPUT_DATA]
```json
{{
    "key": "value",
    ...
}}
```
{completion_signal_suffix}
"""
        return await self.call_llm(
            user_prompt, temperature, max_tokens,
            purpose=purpose, thinking_before=thinking_before,
            model_override=model_override,
        )

    async def call_llm_structured(
        self,
        response_model,
        user_prompt: str,
        temperature: float = 0.2,
        max_retries: int = 2,
        purpose: str = None,
    ):
        """구조화된 LLM 호출 (Instructor 기반)"""
        agent = self._agent

        try:
            from src.common.utils.llm_instructor import call_llm_structured as instructor_call
            from ..model_config import get_agent_model

            model = get_agent_model(agent.agent_type)

            messages = [
                {"role": "system", "content": agent.get_full_system_prompt()},
                {"role": "user", "content": user_prompt},
            ]

            if agent.current_todo:
                agent.event_emitter.agent_thinking(
                    agent_id=agent.agent_id,
                    todo_id=agent.current_todo.todo_id,
                    thinking=f"[Structured] {user_prompt[:300]}",
                    phase=agent.phase,
                )

            result = await asyncio.to_thread(
                instructor_call,
                response_model=response_model,
                messages=messages,
                model=model,
                temperature=temperature,
                max_retries=max_retries,
            )

            agent.metrics.llm_calls += 1
            logger.debug(f"[Instructor] {agent.agent_name} -> {response_model.__name__}")

            return result

        except ImportError:
            logger.warning("Instructor not available, falling back to regular call_llm")
            raise RuntimeError("Instructor not available. Install with: pip install instructor")

        except Exception as e:
            logger.error(f"Structured LLM call failed: {e}")
            raise

    # === Private: Prompt Enrichment ===

    def _resolve_model(self, model_override: Optional[str] = None) -> str:
        """모델 결정"""
        from ..model_config import get_agent_model, get_model, ModelType

        if model_override:
            try:
                return get_model(ModelType(model_override))
            except ValueError:
                return model_override
        else:
            return get_agent_model(self._agent.agent_type)

    def _enrich_with_fewshot(self, prompt: str) -> str:
        """Few-shot learning — 유사 경험을 프롬프트에 추가"""
        agent = self._agent

        try:
            from .learning import ExperienceType, AGENT_LEARNING_AVAILABLE as learning_ok
        except ImportError:
            return prompt

        if not agent.agent_memory or not learning_ok:
            return prompt

        try:
            few_shot = agent.agent_memory.get_few_shot_examples(
                experience_type=ExperienceType.AGENT_DECISION,
                n=2,
                only_success=True,
            )
            if few_shot:
                examples_text = "\n".join(
                    f"- Prior: {exp.agent_action} → {exp.outcome.value}" +
                    (f" (confidence: {exp.confidence_score:.2f})" if exp.confidence_score else "")
                    for exp in few_shot
                )
                return f"{prompt}\n\n[Historical context from similar tasks]\n{examples_text}"
        except Exception:
            pass

        return prompt

    async def _get_bus_context(self) -> str:
        """Agent Bus — 다른 에이전트 메시지를 컨텍스트로 변환"""
        agent = self._agent

        try:
            from .agent_bus import AGENT_BUS_AVAILABLE as bus_ok
        except ImportError:
            return ""

        if not agent.agent_bus or not bus_ok:
            return ""

        try:
            pending_count = agent.agent_bus.get_pending_messages(agent.agent_id)
            if pending_count > 0:
                messages_text = []
                for _ in range(min(pending_count, 5)):
                    msg = await agent.agent_bus.receive(agent.agent_id, timeout=0.1)
                    if msg:
                        sender = msg.sender_id
                        content_summary = str(msg.content)[:300]
                        messages_text.append(f"- From {sender}: {content_summary}")
                if messages_text:
                    return "\n\n[Inter-agent communications]\n" + "\n".join(messages_text)
        except Exception:
            pass

        return ""

    def _get_v17_context(self) -> str:
        """v17 서비스 컨텍스트 자동 주입"""
        agent = self._agent
        if not agent.shared_context:
            return ""

        try:
            v17_services = agent.shared_context.get_dynamic("_v17_services", {})
            v17_parts = []

            calibrator = v17_services.get("calibrator")
            if calibrator:
                try:
                    cal_summary = calibrator.get_calibration_summary()
                    if cal_summary.get("total_calibrations", 0) > 0:
                        v17_parts.append(f"- Calibration: {cal_summary.get('total_calibrations', 0)} calibrations tracked")
                except Exception:
                    pass

            if v17_services.get("semantic_searcher"):
                v17_parts.append("- Semantic search: available for entity lookup")

            if v17_services.get("remediation_engine"):
                v17_parts.append("- Auto-remediation: active for data quality issues")

            version_mgr = v17_services.get("version_manager")
            if version_mgr:
                try:
                    history = version_mgr.log(limit=1)
                    if history:
                        v17_parts.append(f"- Versioning: {len(history)} commits tracked")
                except Exception:
                    pass

            if v17_parts:
                return "\n\n[v17 Platform Services]\n" + "\n".join(v17_parts)
        except Exception:
            pass

        return ""

    def _record_asi_metrics(self, response: str) -> None:
        """ASI 메트릭 기록"""
        if not response:
            return

        agent = self._agent

        conf_match = re.search(r'\[CONFIDENCE:\s*([\d.]+)\]', response, re.IGNORECASE)
        confidence = float(conf_match.group(1)) if conf_match else 0.5
        confidence = max(0.0, min(1.0, confidence))

        decision = None
        if "approve" in response.lower():
            decision = "approve"
        elif "reject" in response.lower():
            decision = "reject"

        agent.metrics.record_response(
            response_length=len(response),
            confidence=confidence,
            decision=decision
        )

        drift_warning = agent.metrics.get_drift_warning()
        if drift_warning:
            logger.warning(f"[ASI] {agent.agent_id}: {drift_warning}")

    def _format_context(self, context_data: Dict[str, Any]) -> str:
        """LLM용 컨텍스트 포맷팅"""
        import json

        parts = []
        for key, value in context_data.items():
            if isinstance(value, (dict, list)):
                formatted = json.dumps(value, indent=2, ensure_ascii=False, default=str)
                parts.append(f"### {key}\n```json\n{formatted}\n```")
            else:
                parts.append(f"### {key}\n{value}")
        return "\n\n".join(parts)
