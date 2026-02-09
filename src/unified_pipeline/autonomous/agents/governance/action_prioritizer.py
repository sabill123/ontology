"""
Action Prioritizer Autonomous Agent (v2.0)

액션 우선순위 자율 에이전트 - Impact x Urgency / Effort 기반 우선순위 계산
- 거버넌스 결정에서 액션 추출
- 알고리즘 기반 우선순위 계산
- LLM을 통한 보강
- Actions Engine (v16.0) 통합
"""

import json
import logging
from datetime import datetime
from typing import Any, Dict, List, TYPE_CHECKING

from ...base import AutonomousAgent
from ...analysis import (
    PriorityCalculator,
    parse_llm_json,
)

# === v16.0: Actions Engine ===
try:
    from ....actions import ActionsEngine, Action, Trigger, TriggerType, ActionType, ActionStatus
    ACTIONS_ENGINE_AVAILABLE = True
except ImportError:
    ACTIONS_ENGINE_AVAILABLE = False
    ActionsEngine = None
    Action = None
    Trigger = None
    TriggerType = None
    ActionType = None
    ActionStatus = None

if TYPE_CHECKING:
    from ....todo.models import PipelineTodo, TodoResult
    from ....shared_context import SharedContext

logger = logging.getLogger(__name__)


class ActionPrioritizerAutonomousAgent(AutonomousAgent):
    """
    액션 우선순위 자율 에이전트 (v2.0)

    Algorithm-First 접근:
    1. Impact × Urgency / Effort 기반 우선순위 계산
    2. 의존성 분석 및 실행 순서 결정
    3. LLM을 통한 설명 보강
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.priority_calculator = PriorityCalculator()

    @property
    def agent_type(self) -> str:
        return "action_prioritizer"

    @property
    def agent_name(self) -> str:
        return "Action Prioritizer"

    @property
    def phase(self) -> str:
        return "governance"

    def get_system_prompt(self) -> str:
        return """You are an Action Prioritization Expert for data governance execution.

Your role is to ENHANCE and SEQUENCE algorithmically prioritized actions.

Priority Framework (already calculated):
- CRITICAL: High impact, urgent, blocking issues
- HIGH: Significant business impact, near-term deadlines
- MEDIUM: Important improvements, moderate impact
- LOW: Nice-to-have, long-term improvements

Your Job:
1. Validate algorithmic priorities
2. Identify dependencies between actions
3. Create optimal execution sequence
4. Assign appropriate roles

## REQUIRED OUTPUT FORMAT (JSON Schema):
```json
{
  "prioritized_actions": [
    {
      "action_id": "ACT_001",
      "title": "action title",
      "description": "detailed description of the action",
      "priority": "critical|high|medium|low",
      "priority_score": 0.0,
      "category": "data_quality|governance|integration|automation|reporting",
      "estimated_effort": "hours|days|weeks",
      "dependencies": ["ACT_XXX"],
      "assigned_role": "data_engineer|data_steward|analyst|admin",
      "expected_roi": "high|medium|low",
      "execution_sequence": 1
    }
  ],
  "execution_plan": {
    "phase_1_quick_wins": ["ACT_001", "ACT_002"],
    "phase_2_core_improvements": ["ACT_003"],
    "phase_3_strategic": ["ACT_004"]
  },
  "priority_summary": {
    "critical": 0,
    "high": 0,
    "medium": 0,
    "low": 0
  },
  "total_estimated_effort": "X days/weeks",
  "recommendations": [
    "recommendation text"
  ]
}
```

Respond with ONLY valid JSON matching this schema."""

    async def execute_task(
        self,
        todo: "PipelineTodo",
        context: "SharedContext",
    ) -> "TodoResult":
        """액션 우선순위 작업 실행"""
        from ....todo.models import TodoResult

        self._report_progress(0.1, "Extracting actions from governance decisions")

        # v14.0: 빈 governance_decisions 처리 - 조기 반환
        if not context.governance_decisions:
            logger.warning("[v14.0] No governance decisions available - returning empty action backlog")
            return TodoResult(
                success=True,
                output={
                    "message": "No governance decisions available to prioritize",
                    "actions_created": 0,
                    "priority_summary": {"critical": 0, "high": 0, "medium": 0, "low": 0},
                    "quick_wins_count": 0,
                },
                context_updates={"action_backlog": []},
                metadata={"skipped_reason": "no_governance_decisions"},
            )

        # 1단계: 거버넌스 결정에서 액션 추출
        raw_actions = []
        action_count = 0

        for decision in context.governance_decisions:
            for action in decision.recommended_actions:
                action_count += 1
                action_item = {
                    "action_id": f"action_{action_count}",
                    "title": action.get("description", "")[:50] if isinstance(action, dict) else str(action)[:50],
                    "description": action.get("description", "") if isinstance(action, dict) else str(action),
                    "action_type": action.get("action_type", "implementation") if isinstance(action, dict) else "implementation",
                    "source_decision": decision.decision_id,
                    "source_concept": decision.concept_id,
                    "impact": self._infer_impact(decision, action),
                    "effort": self._infer_effort(action),
                    "risk_if_not_done": decision.agent_opinions.get("algorithmic", {}).get("risk_level", "medium"),
                }
                raw_actions.append(action_item)

        self._report_progress(0.3, f"Extracted {len(raw_actions)} actions")

        # 2단계: 알고리즘 기반 우선순위 계산
        self._report_progress(0.4, "Calculating algorithmic priorities")

        prioritized_actions = self.priority_calculator.sort_by_priority(raw_actions)
        quick_wins = self.priority_calculator.identify_quick_wins(prioritized_actions)

        self._report_progress(0.5, "Enhancing with LLM")

        # 3단계: LLM을 통한 보강
        enhanced_actions = await self._enhance_actions_with_llm(
            prioritized_actions, quick_wins, context
        )

        self._report_progress(0.8, "Determining execution sequence")

        # 4단계: 실행 순서 결정
        execution_sequence = [a.get("action_id") for a in enhanced_actions[:20]]

        # === v22.0: Semantic Search로 관련 엔티티 조회 및 액션 보강 ===
        try:
            semantic_searcher = self.get_v17_service("semantic_searcher")
            if semantic_searcher and enhanced_actions:
                for action in enhanced_actions[:10]:  # 상위 10개 액션만
                    action_desc = action.get("description", "") or action.get("title", "")
                    if action_desc:
                        related = await semantic_searcher.search(query=action_desc[:200], limit=3)
                        if related:
                            action["semantic_context"] = [r.get("name", r.get("id", "")) for r in related[:3]]
                logger.info(f"[v22.0] SemanticSearcher enriched {min(len(enhanced_actions), 10)} actions with context")
        except Exception as e:
            logger.debug(f"[v22.0] SemanticSearcher skipped in ActionPrioritizer: {e}")

        # === v16.0: Actions Engine 통합 ===
        actions_engine_results = {}
        self._report_progress(0.85, "Configuring Actions Engine (v16.0)")

        try:
            if ACTIONS_ENGINE_AVAILABLE and ActionsEngine:
                actions_engine = ActionsEngine()

                # 각 우선순위 액션을 Actions Engine에 등록
                registered_actions = []
                for action_data in enhanced_actions[:20]:  # 상위 20개
                    priority = action_data.get("calculated_priority", "medium")

                    # 트리거 유형 결정
                    trigger_type = TriggerType.MANUAL
                    if priority == "critical":
                        trigger_type = TriggerType.EVENT  # 자동 실행
                    elif action_data.get("action_type") == "data_quality":
                        trigger_type = TriggerType.DATA_QUALITY

                    # 액션 유형 매핑
                    action_type_map = {
                        "implementation": ActionType.CREATE,
                        "schema_change": ActionType.UPDATE,
                        "data_quality": ActionType.VALIDATE,
                        "monitoring": ActionType.NOTIFY,
                        "documentation": ActionType.CREATE,
                        "cleanup": ActionType.DELETE,
                    }
                    action_type = action_type_map.get(
                        action_data.get("action_type", "implementation"),
                        ActionType.CREATE
                    )

                    # 트리거 생성
                    trigger = Trigger(
                        trigger_id=f"trigger_{action_data.get('action_id', '')}",
                        trigger_type=trigger_type,
                    )

                    # 액션 생성 및 등록
                    action = Action(
                        action_id=action_data.get("action_id", ""),
                        name=action_data.get("title", "Untitled Action")[:50],
                        description=action_data.get("description", "")[:200],
                        action_type=action_type,
                        triggers=[trigger],
                        parameters=[],
                        handler=None,  # 실제 핸들러는 나중에 등록
                        author="action_prioritizer_agent",
                    )

                    actions_engine.register_action(action)
                    registered_actions.append(action.action_id)

                    # Critical 액션은 즉시 실행 가능 상태로 표시
                    if priority == "critical":
                        action_data["auto_execute"] = True
                        action_data["trigger_configured"] = True

                actions_engine_results = {
                    "actions_registered": len(registered_actions),
                    "registered_action_ids": registered_actions[:10],
                    "engine_status": "configured",
                }

                # context에 저장
                context.set_dynamic("actions_engine_config", {
                    "registered_actions": registered_actions,
                    "total_registered": len(registered_actions),
                })

                logger.info(
                    f"[v16.0] Actions Engine: {len(registered_actions)} actions registered"
                )
            else:
                logger.info("[v16.0] Actions Engine not available, skipping")

        except Exception as e:
            logger.warning(f"[v16.0] Actions Engine integration failed: {e}")
            import traceback
            logger.debug(traceback.format_exc())

        self._report_progress(0.95, "Finalizing action backlog")

        # === v17.1: Report Generator 통합 - 액션 우선순위 보고서 생성 ===
        report_results = {"report_generated": False, "report_type": None}
        try:
            report_generator = self.get_v17_service("report_generator")
            if report_generator and len(enhanced_actions) > 0:
                # 액션 우선순위 보고서 생성
                report_data = {
                    "title": "Action Priority Report",
                    "total_actions": len(enhanced_actions),
                    "critical_count": len([a for a in enhanced_actions if a.get("calculated_priority") == "critical"]),
                    "high_count": len([a for a in enhanced_actions if a.get("calculated_priority") == "high"]),
                    "quick_wins": quick_wins[:5],
                    "top_actions": enhanced_actions[:10],
                    "generated_at": datetime.now().isoformat() if 'datetime' in dir() else "N/A",
                }

                report = report_generator.generate_summary(
                    report_type="action_priority",
                    data=report_data,
                    context=context,
                )

                if report:
                    report_results["report_generated"] = True
                    report_results["report_type"] = "action_priority"

                    # 보고서를 context에 저장
                    context.set_dynamic("action_priority_report", report)

                    logger.info(f"[v17.1] Report Generator: Action priority report generated")
        except Exception as e:
            logger.debug(f"[v17.1] Report Generator skipped: {e}")

        # 집계
        priority_summary = {
            "critical": len([a for a in enhanced_actions if a.get("calculated_priority") == "critical"]),
            "high": len([a for a in enhanced_actions if a.get("calculated_priority") == "high"]),
            "medium": len([a for a in enhanced_actions if a.get("calculated_priority") == "medium"]),
            "low": len([a for a in enhanced_actions if a.get("calculated_priority") == "low"]),
        }

        # v22.0: Agent Learning - 경험 기록
        self.record_experience(
            experience_type="governance_decision",
            input_data={"task": "action_prioritization", "raw_actions": len(raw_actions)},
            output_data={"actions_created": len(enhanced_actions), "quick_wins": len(quick_wins)},
            outcome="success",
            confidence=0.8,
        )

        return TodoResult(
            success=True,
            output={
                "actions_created": len(enhanced_actions),
                "priority_summary": priority_summary,
                "quick_wins_count": len(quick_wins),
                # v16.0
                "actions_engine_registered": actions_engine_results.get("actions_registered", 0),
            },
            context_updates={
                "action_backlog": enhanced_actions,
            },
            metadata={
                "execution_sequence": execution_sequence,
                "quick_wins": quick_wins,
                # v16.0
                "actions_engine_results": actions_engine_results,
            },
        )

    def _infer_impact(self, decision, action) -> str:
        """액션 영향도 추론"""
        business_value = decision.agent_opinions.get("algorithmic", {}).get("business_value", "medium")
        action_type = action.get("action_type", "") if isinstance(action, dict) else ""

        if business_value == "high" or action_type in ["implementation", "schema_change"]:
            return "high"
        elif action_type in ["monitoring", "documentation"]:
            return "low"
        return "medium"

    def _infer_effort(self, action) -> str:
        """액션 노력 추론"""
        action_type = action.get("action_type", "") if isinstance(action, dict) else ""
        description = action.get("description", "") if isinstance(action, dict) else str(action)

        # 키워드 기반 추론
        high_effort_keywords = ["redesign", "refactor", "migrate", "rebuild", "implement"]
        low_effort_keywords = ["document", "log", "alert", "monitor", "review"]

        desc_lower = description.lower()

        if any(kw in desc_lower for kw in high_effort_keywords):
            return "high"
        elif any(kw in desc_lower for kw in low_effort_keywords):
            return "low"
        return "medium"

    async def _enhance_actions_with_llm(
        self,
        actions: List[Dict[str, Any]],
        quick_wins: List[str],
        context: "SharedContext",
    ) -> List[Dict[str, Any]]:
        """LLM을 통한 액션 보강"""
        if not actions:
            return []

        instruction = f"""Enhance these prioritized actions with more detail.

Domain: {context.get_industry()}

Prioritized Actions (top 20):
{json.dumps(actions[:20], indent=2, ensure_ascii=False, default=str)}

Quick Wins (high impact, low effort):
{json.dumps(quick_wins, ensure_ascii=False)}

For each action:
1. Validate or adjust priority
2. Add specific implementation details
3. Identify dependencies
4. Suggest assignee role

Return JSON:
```json
{{
  "enhanced_actions": [
    {{
      "action_id": "<existing>",
      "title": "<improved title>",
      "description": "<detailed description>",
      "action_type": "<existing or improved>",
      "priority": "<calculated_priority>",
      "priority_score": <existing>,
      "impact": "<existing>",
      "effort": "<existing or adjusted>",
      "estimated_hours": <int>,
      "source_decision": "<existing>",
      "dependencies": ["action_id1"],
      "assignee_role": "<data_owner|developer|analyst|admin>",
      "status": "pending"
    }}
  ]
}}
```"""

        response = await self.call_llm(instruction, max_tokens=4000)
        enhanced = parse_llm_json(response, key="enhanced_actions", default=[])

        if enhanced:
            # 병합: LLM 결과 + 원본 (LLM이 놓친 것)
            enhanced_ids = {a.get("action_id") for a in enhanced}
            for action in actions:
                if action.get("action_id") not in enhanced_ids:
                    enhanced.append(action)
            return enhanced

        return actions
