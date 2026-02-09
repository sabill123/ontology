"""
Conflict Resolver Autonomous Agent (v2.0)

충돌 해결 자율 에이전트 - 알고리즘 기반 충돌 탐지 + LLM 해결
- 이름 유사도 기반 충돌 탐지
- 정의 중복 탐지
- LLM을 통한 해결 전략 결정
"""

import json
import logging
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from ...base import AutonomousAgent
from ...analysis import (
    ConflictDetector,
    parse_llm_json,
)

if TYPE_CHECKING:
    from ....todo.models import PipelineTodo, TodoResult
    from ....shared_context import SharedContext

logger = logging.getLogger(__name__)


class ConflictResolverAutonomousAgent(AutonomousAgent):
    """
    충돌 해결 자율 에이전트 (v2.0)

    Algorithm-First 접근:
    1. 알고리즘 기반 충돌 탐지 (이름 유사도, 정의 중복)
    2. LLM을 통한 해결 전략 결정
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conflict_detector = ConflictDetector(similarity_threshold=0.75)

    @property
    def agent_type(self) -> str:
        return "conflict_resolver"

    @property
    def agent_name(self) -> str:
        return "Conflict Resolver"

    @property
    def phase(self) -> str:
        return "refinement"

    def get_system_prompt(self) -> str:
        return """You are a Conflict Resolution Expert specialized in ontology harmonization.

Your role is to RESOLVE conflicts that have been algorithmically detected.

v5.2 Resolution Philosophy:
- MOST conflicts can be resolved by KEEPING BOTH concepts
- True conflicts are rare - different names often mean different things
- When in doubt, KEEP BOTH and let the system evolve

Resolution Strategies (in order of preference):
1. **Keep Both**: Maintain both if semantically distinct (DEFAULT)
2. **Merge**: Combine only when concepts are truly identical
3. **Rename**: Differentiate concepts with distinct names
4. **Remove One**: Delete only if clearly redundant
5. **Escalate**: Flag for human review if genuinely unresolvable

Principles:
- DEFAULT TO KEEPING concepts when data supports them
- Prefer VALUE-BASED evidence over assumptions
- Document resolution rationale
- Preserve data lineage

Respond ONLY with valid JSON."""

    async def execute_task(
        self,
        todo: "PipelineTodo",
        context: "SharedContext",
    ) -> "TodoResult":
        """충돌 탐지/해결 작업 실행"""
        from ....todo.models import TodoResult

        task_type = todo.task_type

        if task_type == "conflict_detection":
            return await self._detect_conflicts(todo, context)
        elif task_type == "conflict_resolution":
            return await self._resolve_conflicts(todo, context)
        else:
            return TodoResult(success=False, error=f"Unknown task type: {task_type}")

    async def _detect_conflicts(
        self,
        todo: "PipelineTodo",
        context: "SharedContext",
    ) -> "TodoResult":
        """충돌 탐지 (알고리즘 기반) - v17.1: data_patterns 통합"""
        from ....todo.models import TodoResult

        self._report_progress(0.1, "Preparing concepts for conflict detection")

        # === v17.1: Phase 1 data_patterns 로드 ===
        data_patterns = context.get_dynamic("data_patterns", [])
        logger.info(f"[v17.1] Loaded {len(data_patterns)} data patterns for conflict detection")

        # 개념을 딕셔너리로 변환
        concepts = [
            {
                "concept_id": c.concept_id,
                "concept_type": c.concept_type,
                "name": c.name,
                "description": c.description,
                "definition": c.definition,
                "source_tables": c.source_tables,
                "confidence": c.confidence,
            }
            for c in context.ontology_concepts
        ]

        # === v17.1: data_patterns 기반 추가 충돌 탐지 ===
        # 수학적 패턴(A+B=1, derived columns)으로 인한 중복/파생 컬럼 탐지
        pattern_based_conflicts = []
        for pattern in data_patterns:
            if isinstance(pattern, dict):
                pattern_type = pattern.get("type", pattern.get("pattern_type", ""))

                # 파생 컬럼 패턴: 중복 개념 가능성
                if pattern_type in ("derived_column", "calculated_field", "sum_to_one"):
                    columns = pattern.get("columns", pattern.get("involved_columns", []))
                    for c in concepts:
                        if any(col in str(c.get("definition", {})) for col in columns):
                            pattern_based_conflicts.append({
                                "conflict_id": f"pattern_derived_{c['concept_id']}",
                                "conflict_type": "semantic_redundancy",
                                "severity": "low",
                                "items_involved": [c["concept_id"]],
                                "description": f"Concept may include derived/calculated attributes: {pattern}",
                                "evidence": {"pattern": pattern},
                                "suggested_resolution": "Consider separating base vs derived attributes",
                                "confidence": 0.6,
                            })

        self._report_progress(0.3, "Running algorithmic conflict detection")

        # 알고리즘 기반 충돌 탐지
        detected_conflicts = self.conflict_detector.detect_all_conflicts(concepts)

        self._report_progress(0.6, f"Detected {len(detected_conflicts)} conflicts")

        # 충돌을 JSON 직렬화 가능한 형태로 변환
        conflicts_data = [
            {
                "conflict_id": c.conflict_id,
                "conflict_type": c.conflict_type,
                "severity": c.severity,
                "items_involved": c.items_involved,
                "description": c.description,
                "evidence": c.evidence,
                "suggested_resolution": c.suggested_resolution,
                "confidence": c.confidence,
            }
            for c in detected_conflicts
        ]

        # === v17.1: data_patterns 기반 충돌 병합 ===
        conflicts_data.extend(pattern_based_conflicts)
        if pattern_based_conflicts:
            logger.info(f"[v17.1] Added {len(pattern_based_conflicts)} pattern-based conflicts")

        # 충돌이 있으면 LLM에게 추가 분석 요청
        if conflicts_data:
            self._report_progress(0.7, "Enhancing conflict analysis with LLM")
            enhanced = await self._enhance_conflict_analysis(conflicts_data, concepts)
            if enhanced:
                conflicts_data = enhanced

        self._report_progress(0.95, "Finalizing conflict detection")

        # 심각도별 집계
        by_severity = {"critical": 0, "high": 0, "medium": 0, "low": 0}
        for c in conflicts_data:
            severity = c.get("severity", "medium")
            by_severity[severity] = by_severity.get(severity, 0) + 1

        # 충돌이 없는 개념 식별
        involved_ids = set()
        for c in conflicts_data:
            involved_ids.update(c.get("items_involved", []))

        clean_concepts = [
            c["concept_id"] for c in concepts
            if c["concept_id"] not in involved_ids
        ]

        # v22.0: Agent Learning - 경험 기록
        self.record_experience(
            experience_type="conflict_resolution",
            input_data={"task": "conflict_detection", "concepts": len(concepts)},
            output_data={"conflicts_detected": len(conflicts_data), "clean_concepts": len(clean_concepts)},
            outcome="success" if conflicts_data else "partial",
            confidence=0.75,
        )

        return TodoResult(
            success=True,
            output={
                "conflicts_detected": len(conflicts_data),
                "by_severity": by_severity,
                "by_type": {
                    t: len([c for c in conflicts_data if c.get("conflict_type") == t])
                    for t in ["naming", "definition", "type", "relationship"]
                },
                "clean_concepts_count": len(clean_concepts),
            },
            metadata={
                "conflicts_detected": conflicts_data,
                "clean_concepts": clean_concepts,
            },
        )

    async def _enhance_conflict_analysis(
        self,
        conflicts: List[Dict[str, Any]],
        concepts: List[Dict[str, Any]],
    ) -> Optional[List[Dict[str, Any]]]:
        """LLM을 통한 충돌 분석 보강"""
        if not conflicts:
            return None

        # 관련 개념만 추출
        involved_ids = set()
        for c in conflicts:
            involved_ids.update(c.get("items_involved", []))

        related_concepts = [
            c for c in concepts if c.get("concept_id") in involved_ids
        ]

        instruction = f"""Review these algorithmically detected conflicts and enhance the analysis.

Detected Conflicts:
{json.dumps(conflicts[:20], indent=2, ensure_ascii=False, default=str)}

Related Concepts:
{json.dumps(related_concepts[:30], indent=2, ensure_ascii=False, default=str)}

For each conflict:
1. Validate if it's a true conflict
2. Refine the severity if needed
3. Provide a more specific resolution suggestion

Return JSON:
```json
{{
  "conflicts": [
    {{
      "conflict_id": "<existing>",
      "conflict_type": "<existing or refined>",
      "severity": "<refined severity>",
      "items_involved": [<existing>],
      "description": "<enhanced description>",
      "evidence": "<enhanced evidence>",
      "suggested_resolution": "<specific resolution steps>",
      "confidence": <float>,
      "is_valid_conflict": <bool>
    }}
  ]
}}
```"""

        response = await self.call_llm(instruction, max_tokens=4000)
        result = parse_llm_json(response, key="conflicts", default=None)

        if result and isinstance(result, list):
            # 유효한 충돌만 반환
            return [c for c in result if c.get("is_valid_conflict", True)]

        return None

    async def _resolve_conflicts(
        self,
        todo: "PipelineTodo",
        context: "SharedContext",
    ) -> "TodoResult":
        """충돌 해결"""
        from ....todo.models import TodoResult

        self._report_progress(0.1, "Loading detected conflicts")

        conflicts = todo.input_data.get("conflicts_detected", [])

        if not conflicts:
            return TodoResult(
                success=True,
                output={"message": "No conflicts to resolve"},
                context_updates={"conflicts_resolved": []},
            )

        self._report_progress(0.3, f"Resolving {len(conflicts)} conflicts")

        # LLM에게 해결 요청
        instruction = f"""Resolve these detected conflicts.

Conflicts to Resolve:
{json.dumps(conflicts, indent=2, ensure_ascii=False, default=str)}

For each conflict, provide:
1. Resolution type (merge/rename/keep_both/remove_one/escalate)
2. Specific actions to take
3. Resulting concept if merging

Return JSON:
```json
{{
  "conflicts_resolved": [
    {{
      "conflict_id": "<id>",
      "resolution_type": "<merge|rename|keep_both|remove_one|escalate>",
      "resolution_action": "<specific action description>",
      "resulting_concept": {{
        "concept_id": "<new or existing id>",
        "name": "<resolved name>",
        "description": "<resolved description>"
      }},
      "changes_made": ["change1", "change2"],
      "needs_human_review": <bool>,
      "confidence": <float 0-1>
    }}
  ],
  "concepts_to_update": [
    {{
      "concept_id": "<id>",
      "updates": {{"field": "new_value"}}
    }}
  ]
}}
```"""

        response = await self.call_llm(instruction, max_tokens=4000)

        self._report_progress(0.7, "Processing resolutions")

        resolved = parse_llm_json(response, key="conflicts_resolved", default=[])
        updates = parse_llm_json(response, key="concepts_to_update", default=[])

        # v14.0: LLM이 빈 결과를 반환하고 충돌이 있는 경우 - 기본 해결 (에스컬레이션)
        if not resolved and conflicts:
            logger.warning(f"[v14.0] LLM returned empty resolutions for {len(conflicts)} conflicts - applying default escalation")
            resolved = [
                {
                    "conflict_id": c.get("conflict_id", f"conflict_{i}"),
                    "resolution_type": "escalate",
                    "resolution_action": "Escalated to human review due to LLM resolution failure",
                    "resulting_concept": None,
                    "changes_made": [],
                    "needs_human_review": True,
                    "confidence": 0.3,
                }
                for i, c in enumerate(conflicts)
            ]

        # 컨텍스트 개념에 업데이트 적용
        if updates:
            update_map = {u.get("concept_id"): u.get("updates", {}) for u in updates}
            for concept in context.ontology_concepts:
                if concept.concept_id in update_map:
                    for field, value in update_map[concept.concept_id].items():
                        if hasattr(concept, field):
                            setattr(concept, field, value)

        self._report_progress(0.95, "Finalizing resolutions")

        # === v22.0: remediation_engine으로 자동 수정 제안 생성 ===
        remediation_suggestions = []
        try:
            remediation_engine = self.get_v17_service("remediation_engine")
            if remediation_engine and resolved:
                for resolution in resolved[:5]:  # 상위 5개만
                    if resolution.get("resolution_type") not in ["escalate"]:
                        suggestion = await remediation_engine.suggest_fix(
                            issue_type="conflict",
                            issue_data={
                                "conflict_id": resolution.get("conflict_id"),
                                "resolution_type": resolution.get("resolution_type"),
                                "changes_made": resolution.get("changes_made", []),
                            },
                            context={"phase": "refinement", "agent": self.agent_name},
                        )
                        if suggestion:
                            remediation_suggestions.append({
                                "conflict_id": resolution.get("conflict_id"),
                                "auto_fix": suggestion.get("fix_script", ""),
                                "confidence": suggestion.get("confidence", 0.0),
                            })
                if remediation_suggestions:
                    logger.info(f"[v22.0] RemediationEngine generated {len(remediation_suggestions)} auto-fix suggestions")
        except Exception as e:
            logger.debug(f"[v22.0] RemediationEngine skipped: {e}")

        # v22.0: Agent Learning - 경험 기록
        self.record_experience(
            experience_type="conflict_resolution",
            input_data={"task": "conflict_resolution", "conflicts": len(conflicts)},
            output_data={"conflicts_resolved": len(resolved)},
            outcome="success" if resolved else "partial",
            confidence=0.8,
        )

        # v22.0: Agent Bus - 충돌 해결 결과 브로드캐스트
        await self.broadcast_discovery(
            discovery_type="conflicts_resolved",
            data={
                "conflicts_resolved": len(resolved),
                "needs_human_review": len([r for r in resolved if r.get("needs_human_review")]),
                "resolution_types": {
                    rt: len([r for r in resolved if r.get("resolution_type") == rt])
                    for rt in ["merge", "rename", "keep_both", "remove_one", "escalate"]
                },
            },
        )

        return TodoResult(
            success=True,
            output={
                "conflicts_resolved": len(resolved),
                "needs_human_review": len([r for r in resolved if r.get("needs_human_review")]),
                "by_resolution_type": {
                    rt: len([r for r in resolved if r.get("resolution_type") == rt])
                    for rt in ["merge", "rename", "keep_both", "remove_one", "escalate"]
                },
                "remediation_suggestions": len(remediation_suggestions),  # v22.0
            },
            context_updates={
                "conflicts_resolved": resolved,
                "remediation_suggestions": remediation_suggestions,  # v22.0
            },
        )
