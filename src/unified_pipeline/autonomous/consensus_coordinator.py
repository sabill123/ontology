"""
Consensus Coordinator

멀티에이전트 합의 조율 — AgentOrchestrator에서 추출 (SRP)

책임:
- 합의 필요 여부 판단
- 스마트 합의 (full/lightweight/skip)
- 토론 진행 및 결과 적용
- 리스크/충돌 분석
"""

import logging
import re
from typing import Dict, List, Any, Optional, AsyncGenerator, TYPE_CHECKING

from .consensus import PipelineConsensusEngine, ConsensusResult, ConsensusType

if TYPE_CHECKING:
    from .base import AutonomousAgent
    from ..shared_context import SharedContext
    from ..todo.manager import TodoManager
    from ..todo.models import PipelineTodo

logger = logging.getLogger(__name__)


class ConsensusCoordinator:
    """
    멀티에이전트 합의 조율

    AgentOrchestrator의 합의 관련 책임을 분리하여
    합의 판단, 토론 진행, 결과 적용을 담당합니다.
    """

    def __init__(
        self,
        shared_context: "SharedContext",
        consensus_engine: Optional[PipelineConsensusEngine],
        agents: Dict[str, "AutonomousAgent"],
        stats,
        todo_manager: "TodoManager",
        config=None,
    ):
        self.shared_context = shared_context
        self.consensus_engine = consensus_engine
        self._agents = agents
        self.stats = stats
        self.todo_manager = todo_manager
        self.config = config

    def register_agents(self, phase: str) -> None:
        """Phase의 에이전트들을 합의 엔진에 등록"""
        if not self.consensus_engine:
            return

        self.consensus_engine.clear_agents()

        for agent in self._agents.values():
            if agent.phase == phase:
                self.consensus_engine.register_agent(agent)
                logger.debug(f"Registered {agent.agent_id} for consensus in {phase}")

    def requires_consensus(self, todo: "PipelineTodo") -> bool:
        """
        Todo가 멀티에이전트 합의를 필요로 하는지 확인

        v7.1: 품질에 직접 영향을 주는 핵심 결정만 합의
        """
        if not self.consensus_engine:
            return False
        if self.config and not self.config.enable_consensus:
            return False

        todo_name_normalized = todo.name.lower().replace(" ", "_").replace("-", "_")

        critical_quality_decisions = [
            "quality_assessment",
            "ontology_approval",
            "ontology_proposal",
            "governance_decision",
            "governance_strategy",
        ]

        for critical in critical_quality_decisions:
            if critical in todo_name_normalized:
                return True

        return False

    async def run_consensus_discussion(
        self,
        todo: "PipelineTodo",
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        멀티에이전트 토론/합의 진행

        Args:
            todo: 토론할 Todo

        Yields:
            토론 진행 이벤트
        """
        if not self.consensus_engine:
            logger.warning("Consensus engine not available")
            return

        yield {
            "type": "consensus_discussion_started",
            "todo_id": todo.todo_id,
            "todo_name": todo.name,
            "phase": todo.phase,
        }

        self.stats.consensus_discussions += 1

        skip_mode, skip_reason, base_confidence = self._should_skip_v82(todo)

        if skip_mode == "full_skip":
            async for event in self._handle_full_skip(todo, skip_reason, base_confidence):
                yield event
            return

        elif skip_mode == "lightweight":
            async for event in self._handle_lightweight(todo, skip_reason, base_confidence):
                yield event
            return

        # Full consensus
        topic_data = self._build_topic_data(todo)

        evidence_vote_result = await self._run_evidence_prevote(todo, topic_data)

        consensus_result = await self._run_negotiation(todo, topic_data)

        if consensus_result:
            consensus_result = self._integrate_evidence_vote(consensus_result, evidence_vote_result)
            await self._apply_result(todo, consensus_result)

            yield {
                "type": "consensus_discussion_completed",
                "todo_id": todo.todo_id,
                "result_type": consensus_result.result_type.value if hasattr(consensus_result, 'result_type') else "unknown",
                "combined_confidence": consensus_result.combined_confidence if hasattr(consensus_result, 'combined_confidence') else 0,
            }
        else:
            async for event in self._fallback_structured_debate(todo, topic_data):
                yield event

    async def run_batch_consensus(
        self,
        todos: List["PipelineTodo"],
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """여러 Todo를 하나의 합의 세션에서 배치 처리"""
        if not self.consensus_engine or not todos:
            return

        yield {
            "type": "batch_consensus_started",
            "todo_count": len(todos),
            "todo_ids": [t.todo_id for t in todos],
        }

        batch_topic_data = {
            "batch_mode": True,
            "items": [],
        }

        for todo in todos:
            batch_topic_data["items"].append({
                "todo_id": todo.todo_id,
                "todo_name": todo.name,
                "description": todo.description or "",
                "phase": todo.phase,
                "priority": todo.priority,
            })

        batch_id = f"batch_{todos[0].phase}_{len(todos)}"
        consensus_result = None

        async for event in self.consensus_engine.negotiate(
            topic_id=batch_id,
            topic_name=f"Batch approval for {len(todos)} items",
            topic_type=f"{todos[0].phase}_batch_decision",
            topic_data=batch_topic_data,
            context=self.shared_context,
        ):
            yield {
                **event,
                "batch_id": batch_id,
            }

            if event.get("type") in ["consensus_reached", "max_rounds_reached", "consensus_converged"]:
                consensus_result = event.get("result")

        if consensus_result:
            for todo in todos:
                await self._apply_result(todo, consensus_result)

            yield {
                "type": "batch_consensus_completed",
                "batch_id": batch_id,
                "todo_count": len(todos),
                "result_type": consensus_result.result_type.value if hasattr(consensus_result, 'result_type') else "unknown",
            }
        else:
            yield {
                "type": "batch_consensus_failed",
                "batch_id": batch_id,
                "fallback": "individual_processing",
            }

        self.stats.consensus_discussions += 1

    def get_summary(self) -> Dict[str, Any]:
        """합의 통계 요약"""
        return {
            "enabled": self.config.enable_consensus if self.config else False,
            "total_discussions": self.stats.consensus_discussions,
            "accepted": self.stats.consensus_accepted,
            "rejected": self.stats.consensus_rejected,
            "provisional": self.stats.consensus_provisional,
            "escalated": self.stats.consensus_escalated,
            "acceptance_rate": (
                self.stats.consensus_accepted / self.stats.consensus_discussions
                if self.stats.consensus_discussions > 0 else 0
            ),
        }

    # === Private: Skip / Lightweight / Full Consensus Handlers ===

    async def _handle_full_skip(
        self, todo: "PipelineTodo", reason: str, base_confidence: float
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Full skip 모드 처리"""
        logger.info(f"Full skip consensus for {todo.todo_id}: {reason}")
        yield {
            "type": "consensus_skipped",
            "todo_id": todo.todo_id,
            "reason": reason,
            "mode": "full_skip",
        }
        default_result = ConsensusResult(
            success=True,
            result_type=ConsensusType.ACCEPTED,
            combined_confidence=base_confidence,
            agreement_level=1.0,
            total_rounds=0,
            final_decision="auto_approved",
            reasoning=f"Auto-approved: {reason}",
            dissenting_views=[],
        )
        await self._apply_result(todo, default_result)
        yield {
            "type": "consensus_discussion_completed",
            "todo_id": todo.todo_id,
            "result_type": "accepted",
            "combined_confidence": base_confidence,
            "mode": "full_skip",
        }

    async def _handle_lightweight(
        self, todo: "PipelineTodo", reason: str, base_confidence: float
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Lightweight 검증 모드 처리"""
        logger.info(f"Lightweight validation for {todo.todo_id}: {reason}")
        yield {
            "type": "consensus_lightweight_started",
            "todo_id": todo.todo_id,
            "reason": reason,
        }

        lightweight_result = await self._perform_lightweight_validation(todo, base_confidence)

        self.shared_context.last_consensus_result = {
            "todo_id": todo.todo_id,
            "result": lightweight_result.result_type.value,
            "confidence": lightweight_result.combined_confidence,
            "mode": "lightweight",
        }

        yield {
            "type": "consensus_discussion_completed",
            "todo_id": todo.todo_id,
            "result_type": lightweight_result.result_type.value,
            "combined_confidence": lightweight_result.combined_confidence,
            "mode": "lightweight",
            "skip_todo_completion": True,
        }

    async def _run_evidence_prevote(self, todo: "PipelineTodo", topic_data: dict) -> Optional[dict]:
        """Evidence-based vote 사전 검증"""
        evidence_vote_result = None
        try:
            evidence_chain = self.shared_context.get_evidence_chain()
            if evidence_chain and hasattr(evidence_chain, 'blocks') and evidence_chain.blocks:
                logger.info(f"[v7.8] Running evidence-based vote for {todo.todo_id}")
                evidence_vote_result = await self.consensus_engine.conduct_evidence_based_vote(
                    proposal=todo.name,
                    evidence_chain=evidence_chain,
                    context={"todo_id": todo.todo_id, "phase": todo.phase},
                )

                if not evidence_vote_result.get("fallback"):
                    topic_data["evidence_based_vote"] = {
                        "consensus": evidence_vote_result.get("consensus"),
                        "confidence": evidence_vote_result.get("confidence"),
                        "role_breakdown": evidence_vote_result.get("role_breakdown", {}),
                    }
        except Exception as e:
            logger.debug(f"[v7.8] Evidence-based vote skipped: {e}")

        return evidence_vote_result

    async def _run_negotiation(self, todo: "PipelineTodo", topic_data: dict) -> Optional[ConsensusResult]:
        """합의 엔진으로 negotiate 실행"""
        consensus_result = None

        async for event in self.consensus_engine.negotiate(
            topic_id=todo.todo_id,
            topic_name=todo.name,
            topic_type=f"{todo.phase}_decision",
            topic_data=topic_data,
            context=self.shared_context,
        ):
            if event.get("type") in ["consensus_reached", "max_rounds_reached"]:
                consensus_result = event.get("result")

        return consensus_result

    def _integrate_evidence_vote(
        self, consensus_result: ConsensusResult, evidence_vote_result: Optional[dict]
    ) -> ConsensusResult:
        """v23.0: Evidence-based vote 결과 통합"""
        if not evidence_vote_result or evidence_vote_result.get("fallback"):
            return consensus_result

        ev_confidence = evidence_vote_result.get("confidence", 0)
        ev_consensus = evidence_vote_result.get("consensus", "")

        if ev_confidence > consensus_result.combined_confidence:
            logger.info(
                f"[v23.0] Boosting consensus with evidence-based vote: "
                f"negotiate={consensus_result.combined_confidence:.2f} → evidence={ev_confidence:.2f}"
            )

            if ev_consensus in ["approve", "strong_approve"] and ev_confidence >= 0.6:
                return ConsensusResult(
                    success=True,
                    result_type=ConsensusType.ACCEPTED,
                    combined_confidence=ev_confidence,
                    agreement_level=evidence_vote_result.get("approve_ratio", 0.5),
                    total_rounds=consensus_result.total_rounds,
                    final_decision="approve",
                    reasoning=f"Evidence-based vote override: {ev_consensus} with {ev_confidence:.2f} confidence",
                    dissenting_views=[],
                )
            elif ev_consensus == "review" and ev_confidence >= 0.5:
                return ConsensusResult(
                    success=True,
                    result_type=ConsensusType.PROVISIONAL,
                    combined_confidence=ev_confidence,
                    agreement_level=evidence_vote_result.get("approve_ratio", 0.5),
                    total_rounds=consensus_result.total_rounds,
                    final_decision="provisional",
                    reasoning=f"Evidence-based vote boost: {ev_consensus} with {ev_confidence:.2f} confidence",
                    dissenting_views=[],
                )

        return consensus_result

    async def _fallback_structured_debate(
        self, todo: "PipelineTodo", topic_data: dict
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """negotiate 실패 시 StructuredDebate 시도"""
        logger.info(f"[v7.8] No consensus from negotiate, trying StructuredDebate for {todo.todo_id}")

        yield {
            "type": "consensus_structured_debate_started",
            "todo_id": todo.todo_id,
            "reason": "Standard negotiation failed, attempting structured debate",
        }

        try:
            debate_result = await self.consensus_engine.conduct_structured_debate(
                topic=todo.name,
                context={"todo_id": todo.todo_id, "phase": todo.phase},
                evidence=topic_data,
            )

            if not debate_result.get("fallback"):
                final_decision = debate_result.get("final_decision", "review")

                result_type_map = {
                    "approve": ConsensusType.ACCEPTED,
                    "conditional_approve": ConsensusType.PROVISIONAL,
                    "reject": ConsensusType.REJECTED,
                    "review": ConsensusType.PROVISIONAL,
                }

                consensus_result = ConsensusResult(
                    success=final_decision in ["approve", "conditional_approve"],
                    result_type=result_type_map.get(final_decision, ConsensusType.PROVISIONAL),
                    combined_confidence=debate_result.get("consensus_level", 0.5),
                    agreement_level=debate_result.get("consensus_level", 0.5),
                    total_rounds=debate_result.get("rounds_completed", 3),
                    final_decision=final_decision,
                    reasoning=f"StructuredDebate: {final_decision}",
                    dissenting_views=debate_result.get("dissenting_views", []),
                )

                await self._apply_result(todo, consensus_result)

                yield {
                    "type": "consensus_discussion_completed",
                    "todo_id": todo.todo_id,
                    "result_type": consensus_result.result_type.value,
                    "combined_confidence": consensus_result.combined_confidence,
                    "mode": "structured_debate",
                }
                return
        except Exception as e:
            logger.warning(f"[v7.8] StructuredDebate failed: {e}")

        yield {
            "type": "consensus_discussion_escalated",
            "todo_id": todo.todo_id,
            "reason": "No consensus result after structured debate",
        }
        self.stats.consensus_escalated += 1

    # === Private: Result Application ===

    async def _apply_result(
        self,
        todo: "PipelineTodo",
        result: ConsensusResult,
    ) -> None:
        """합의 결과를 Todo에 적용"""
        from ..todo.models import TodoResult

        if result.result_type == ConsensusType.ACCEPTED:
            self.stats.consensus_accepted += 1
        elif result.result_type == ConsensusType.REJECTED:
            self.stats.consensus_rejected += 1
        elif result.result_type == ConsensusType.PROVISIONAL:
            self.stats.consensus_provisional += 1
        elif result.result_type == ConsensusType.ESCALATED:
            self.stats.consensus_escalated += 1
        elif result.result_type == ConsensusType.NO_AGENTS:
            self.stats.consensus_provisional += 1
            logger.warning(f"[v14.0] Consensus skipped due to no agents: {todo.todo_id}")

        todo_result = TodoResult(
            success=result.success,
            output={
                "consensus_result": result.result_type.value,
                "combined_confidence": result.combined_confidence,
                "agreement_level": result.agreement_level,
                "total_rounds": result.total_rounds,
                "final_decision": result.final_decision,
                "reasoning": result.reasoning,
                "dissenting_views": result.dissenting_views,
            },
            context_updates={
                "last_consensus_result": {
                    "todo_id": todo.todo_id,
                    "result": result.result_type.value,
                    "confidence": result.combined_confidence,
                },
            },
        )

        if result.success or result.result_type in [ConsensusType.ACCEPTED, ConsensusType.PROVISIONAL, ConsensusType.NO_AGENTS]:
            self.todo_manager.complete_todo(todo.todo_id, todo_result)
            self.stats.total_todos_completed += 1
            logger.info(
                f"[CONSENSUS OK] {todo.todo_id}: {result.result_type.value} "
                f"(confidence: {result.combined_confidence:.2f})"
            )
        else:
            if result.result_type == ConsensusType.ESCALATED:
                self.todo_manager.fail_todo(
                    todo.todo_id,
                    f"Escalated for human review (confidence: {result.combined_confidence:.2f})"
                )
            else:
                self.todo_manager.fail_todo(
                    todo.todo_id,
                    f"Consensus rejected (confidence: {result.combined_confidence:.2f})"
                )
            self.stats.total_todos_failed += 1

        logger.info(
            f"Consensus result applied to {todo.todo_id}: "
            f"{result.result_type.value} (confidence: {result.combined_confidence:.2f})"
        )

    # === Private: Smart Consensus Decision ===

    def _should_skip_v82(self, todo: "PipelineTodo") -> tuple:
        """
        v8.2: 3단계 스마트 합의 결정

        Returns:
            (mode: str, reason: str, base_confidence: float)
            - mode: "full_skip" | "lightweight" | "full_consensus"
        """
        todo_type = todo.name.lower().replace(" ", "_").replace("-", "_")
        ctx = self.shared_context
        concepts = ctx.ontology_concepts or []
        avg_conf = self._calc_avg_confidence(concepts) if concepts else 0.5

        if concepts and avg_conf < 0.15:
            logger.info(f"[v24.0] Pre-filtered: avg confidence {avg_conf:.2f} too low for {todo.name}")
            return "full_skip", f"Pre-filtered: confidence {avg_conf:.2f} too low", avg_conf

        if todo.phase == "discovery":
            return "full_skip", "Phase 1 discovery - exploration stage", avg_conf

        if "quality" in todo_type or "assessment" in todo_type:
            # v25.2: concepts가 비어있어도 에이전트는 반드시 실행 (full_skip 제거)
            if not concepts:
                return "lightweight", "No concepts yet - agent execution required", 0.5
            if len(concepts) <= 2:
                return "lightweight", f"Few concepts ({len(concepts)}) - agent execution required", avg_conf
            if avg_conf >= 0.85:
                return "lightweight", f"High quality ({avg_conf:.0%}) - agent execution required", avg_conf
            return "lightweight", "Quality assessment - agent execution required", avg_conf

        if "conflict" in todo_type:
            conflicts = self._detect_naming_conflicts()
            if not conflicts:
                return "lightweight", "No conflicts - lightweight validation", 0.85
            if len(conflicts) <= 2:
                return "lightweight", f"Few conflicts ({len(conflicts)}) - lightweight", 0.7
            return "full_consensus", f"Multiple conflicts ({len(conflicts)})", 0.6

        if "ontology" in todo_type and "approval" in todo_type:
            pending = [c for c in concepts if c.status in ["pending", "provisional"]]
            if not pending:
                return "full_skip", "No pending concepts", 0.9
            avg_pending_conf = self._calc_avg_confidence(pending)
            if avg_pending_conf >= 0.8:
                return "lightweight", f"High confidence pending ({avg_pending_conf:.0%})", avg_pending_conf
            return "full_consensus", "Pending concepts need review", avg_pending_conf

        if "governance" in todo_type and "conflict" not in todo_type:
            if not concepts:
                return "full_skip", "No concepts for governance", 0.5
            if avg_conf >= 0.8:
                return "lightweight", f"High confidence ({avg_conf:.0%}) - lightweight governance", avg_conf
            return "full_consensus", "Governance review needed", avg_conf

        if "risk" in todo_type:
            risks = self._identify_risks()
            if not risks:
                return "lightweight", "No risks identified", 0.9
            high_risks = [r for r in risks if r.get("severity") in ["high", "critical"]]
            if not high_risks:
                return "lightweight", "Only low risks - lightweight validation", 0.8
            return "full_consensus", f"High risks ({len(high_risks)}) need review", 0.6

        return "full_consensus", "Default - full consensus", avg_conf

    async def _perform_lightweight_validation(self, todo: "PipelineTodo", base_confidence: float) -> ConsensusResult:
        """경량 검증 - 1회 LLM 호출로 품질 확보"""
        ctx = self.shared_context
        concepts = ctx.ontology_concepts or []

        validation_prompt = self._build_lightweight_prompt(todo, concepts)

        validator_agent = None
        for agent_id, agent in self._agents.items():
            if "validator" in agent_id.lower() or "judge" in agent_id.lower():
                validator_agent = agent
                break

        if not validator_agent and self._agents:
            validator_agent = next(iter(self._agents.values()))

        if not validator_agent:
            calculated_confidence = self._calculate_data_confidence(todo, concepts)
            return ConsensusResult(
                success=True,
                result_type=ConsensusType.ACCEPTED if calculated_confidence >= 0.7 else ConsensusType.PROVISIONAL,
                combined_confidence=calculated_confidence,
                agreement_level=0.9,
                total_rounds=0,
                final_decision="lightweight_auto",
                reasoning=f"Lightweight auto-validation: data_confidence={calculated_confidence:.2f}",
                dissenting_views=[],
            )

        try:
            response = await validator_agent.call_llm(
                user_prompt=validation_prompt,
                temperature=0.3,
                max_tokens=500,
            )

            validated_confidence, validation_reasoning = self._parse_lightweight_response(response, base_confidence)

            if validated_confidence >= 0.75:
                result_type = ConsensusType.ACCEPTED
            elif validated_confidence >= 0.5:
                result_type = ConsensusType.PROVISIONAL
            else:
                result_type = ConsensusType.NEEDS_REVIEW

            return ConsensusResult(
                success=True,
                result_type=result_type,
                combined_confidence=validated_confidence,
                agreement_level=0.85,
                total_rounds=1,
                final_decision="lightweight_validated",
                reasoning=validation_reasoning,
                dissenting_views=[],
            )

        except Exception as e:
            logger.warning(f"Lightweight validation failed: {e}, using base confidence")
            return ConsensusResult(
                success=True,
                result_type=ConsensusType.PROVISIONAL,
                combined_confidence=base_confidence,
                agreement_level=0.8,
                total_rounds=0,
                final_decision="lightweight_fallback",
                reasoning=f"Validation failed, using base confidence: {base_confidence:.2f}",
                dissenting_views=[],
            )

    def _build_lightweight_prompt(self, todo: "PipelineTodo", concepts: list) -> str:
        """경량 검증 프롬프트 생성"""
        avg_conf = self._calc_avg_confidence(concepts) if concepts else 0.5
        ctx = self.shared_context

        evidence_sources = len(ctx.unified_entities or [])
        relationships = len(ctx.homeomorphisms or [])
        approved = len([c for c in concepts if c.status == "approved"])

        summary = f"""## Data Governance Quality Validation

**Decision**: {todo.name}
**Phase**: {todo.phase}

## Current Quality Metrics
- Average Confidence: {avg_conf:.0%}
- Concepts: {len(concepts)} (Approved: {approved})
- Evidence Sources: {evidence_sources}
- Relationships: {relationships}

## Concepts Summary
"""
        if concepts:
            for i, c in enumerate(concepts[:5], 1):
                desc = (c.description[:60] + "...") if c.description and len(c.description) > 60 else (c.description or "N/A")
                summary += f"{i}. {c.name}: {c.confidence:.0%} ({c.status}) - {desc}\n"

        summary += f"""
## Validation Task
Assess overall quality considering:
1. Evidence strength (sources: {evidence_sources}, relationships: {relationships})
2. Confidence appropriateness for data governance
3. Business value and actionable insights
4. Risk level

Provide your assessment:

[QUALITY_SCORE: 0.XX]  (0.0-1.0, be generous if evidence is reasonable)
[DECISION: APPROVE | PROVISIONAL | NEEDS_REVIEW]
[REASONING: 1-2 sentence justification]
"""
        return summary

    def _parse_lightweight_response(self, response: str, base_confidence: float) -> tuple:
        """경량 검증 응답 파싱"""
        confidence = base_confidence
        reasoning = "Lightweight validation completed"

        quality_match = re.search(r'\[QUALITY_SCORE:\s*([\d.]+)\]', response, re.IGNORECASE)
        if quality_match:
            try:
                confidence = float(quality_match.group(1))
                confidence = max(0.0, min(1.0, confidence))
            except ValueError:
                pass
        else:
            conf_match = re.search(r'\[CONFIDENCE:\s*([\d.]+)\]', response, re.IGNORECASE)
            if conf_match:
                try:
                    confidence = float(conf_match.group(1))
                    confidence = max(0.0, min(1.0, confidence))
                except ValueError:
                    pass

        decision_match = re.search(r'\[DECISION:\s*(APPROVE|PROVISIONAL|NEEDS_REVIEW|REJECT)\]', response, re.IGNORECASE)
        if decision_match:
            decision = decision_match.group(1).upper()
            if decision == "APPROVE" and confidence < 0.75:
                confidence = max(confidence, 0.75)
            elif decision == "PROVISIONAL" and confidence < 0.5:
                confidence = max(confidence, 0.5)

        reason_match = re.search(r'\[REASONING:\s*(.+?)\]', response, re.IGNORECASE | re.DOTALL)
        if reason_match:
            reasoning = reason_match.group(1).strip()[:200]
        else:
            reasoning = response[:200] if response else reasoning

        return confidence, reasoning

    def _calculate_data_confidence(self, todo: "PipelineTodo", concepts: list) -> float:
        """데이터 기반 신뢰도 계산 (LLM 없이)"""
        if not concepts:
            return 0.5

        avg_conf = self._calc_avg_confidence(concepts)
        num_concepts = len(concepts)

        approved_count = len([c for c in concepts if c.status == "approved"])
        provisional_count = len([c for c in concepts if c.status == "provisional"])

        if num_concepts > 0:
            approval_rate = approved_count / num_concepts
            provisional_rate = provisional_count / num_concepts
        else:
            approval_rate = 0
            provisional_rate = 0

        status_bonus = (approval_rate * 0.2) + (provisional_rate * 0.1)
        calculated = min(1.0, avg_conf + status_bonus)

        return round(calculated, 2)

    async def apply_quality_status(self, lightweight_result: ConsensusResult) -> None:
        """경량 검증 시 concept.status 설정"""
        concepts = self.shared_context.ontology_concepts or []
        if not concepts:
            return

        confidence = lightweight_result.combined_confidence
        result_type = lightweight_result.result_type

        for concept in concepts:
            if concept.status not in ["approved", "rejected"]:
                if result_type == ConsensusType.ACCEPTED and confidence >= 0.7:
                    concept.status = "approved"
                elif result_type == ConsensusType.ACCEPTED or confidence >= 0.5:
                    concept.status = "provisional"
                else:
                    concept.status = "pending"

                if not hasattr(concept, 'agent_assessments') or concept.agent_assessments is None:
                    concept.agent_assessments = {}

                concept.agent_assessments["quality_judge_lightweight"] = {
                    "decision": concept.status,
                    "confidence": confidence,
                    "reasoning": f"Lightweight validation: {lightweight_result.reasoning}",
                    "mode": "lightweight",
                }

        logger.info(f"Applied quality assessment status to {len(concepts)} concepts (lightweight mode)")

    # === Private: Analysis Helpers ===

    def _build_topic_data(self, todo: "PipelineTodo") -> Dict[str, Any]:
        """토론용 topic_data 구성"""
        todo_type = todo.name.lower().replace(" ", "_").replace("-", "_")
        ctx = self.shared_context
        concepts = ctx.ontology_concepts or []

        base_data = {
            "phase": todo.phase,
            "decision_type": todo_type,
            "concepts_count": len(concepts),
            "avg_confidence": round(self._calc_avg_confidence(concepts), 2),
        }

        if "quality" in todo_type or "assessment" in todo_type:
            base_data["quality_summary"] = {
                "total": len(concepts),
                "by_status": self._count_by_status(concepts),
                "avg_confidence": round(self._calc_avg_confidence(concepts), 2),
            }
            low_conf = sorted(
                [c for c in concepts if c.confidence < 0.6],
                key=lambda x: x.confidence
            )[:3]
            if low_conf:
                base_data["low_confidence_concepts"] = [
                    {"name": c.name, "confidence": round(c.confidence, 2)}
                    for c in low_conf
                ]

        elif "conflict" in todo_type:
            conflicts = self._detect_naming_conflicts()
            base_data["conflict_summary"] = {
                "total": len(conflicts),
                "high_similarity": len([c for c in conflicts if c.get("similarity", 0) > 0.85]),
            }
            base_data["top_conflicts"] = conflicts[:5]

        elif "ontology" in todo_type and "approval" in todo_type:
            pending = [c for c in concepts if c.status in ["pending", "provisional"]]
            base_data["approval_summary"] = {
                "total_pending": len(pending),
                "avg_confidence": round(self._calc_avg_confidence(pending), 2),
            }
            base_data["top_candidates"] = [
                {"name": c.name, "confidence": round(c.confidence, 2), "status": c.status}
                for c in pending[:3]
            ]

        elif "governance" in todo_type:
            base_data["governance_summary"] = {
                "total_concepts": len(concepts),
                "approved": len([c for c in concepts if c.status == "approved"]),
                "avg_confidence": round(self._calc_avg_confidence(concepts), 2),
            }
            insights = ctx.business_insights or []
            if insights:
                base_data["key_insights"] = [
                    getattr(ins, 'description', str(ins))[:100]
                    for ins in insights[:2]
                ]

        elif "risk" in todo_type:
            risks = self._identify_risks()
            base_data["risk_summary"] = {
                "total_risks": len(risks),
                "high_severity": len([r for r in risks if r.get("severity") in ["high", "critical"]]),
            }
            high_risks = [r for r in risks if r.get("severity") in ["high", "critical"]][:3]
            if high_risks:
                base_data["high_risks"] = high_risks

        return base_data

    def _detect_naming_conflicts(self) -> List[Dict[str, Any]]:
        """명명 충돌 탐지"""
        conflicts = []
        concepts = self.shared_context.ontology_concepts or []

        for i, c1 in enumerate(concepts):
            for c2 in concepts[i+1:]:
                similarity = self._calc_name_similarity(c1.name, c2.name)
                if similarity > 0.7 and c1.name.lower() != c2.name.lower():
                    conflicts.append({
                        "type": "naming",
                        "concept_a": c1.name,
                        "concept_b": c2.name,
                        "similarity": round(similarity, 2),
                        "suggestion": "merge" if similarity > 0.85 else "review",
                    })

        return conflicts

    def _calc_name_similarity(self, name1: str, name2: str) -> float:
        """Jaccard 기반 이름 유사도 계산"""
        n1, n2 = name1.lower(), name2.lower()
        if n1 == n2:
            return 1.0

        words1 = set(n1.replace("_", " ").split())
        words2 = set(n2.replace("_", " ").split())

        if not words1 or not words2:
            return 0.0

        intersection = len(words1 & words2)
        union = len(words1 | words2)
        return intersection / union if union > 0 else 0.0

    def _count_by_type(self, concepts) -> Dict[str, int]:
        """개념 타입별 카운트"""
        counts = {}
        for c in concepts:
            t = c.concept_type if hasattr(c, 'concept_type') else "unknown"
            counts[t] = counts.get(t, 0) + 1
        return counts

    def _count_by_status(self, concepts) -> Dict[str, int]:
        """개념 상태별 카운트"""
        counts = {}
        for c in concepts:
            s = c.status if hasattr(c, 'status') else "unknown"
            counts[s] = counts.get(s, 0) + 1
        return counts

    def _calc_avg_confidence(self, concepts) -> float:
        """평균 신뢰도 계산"""
        if not concepts:
            return 0.0
        confidences = [c.confidence for c in concepts if hasattr(c, 'confidence')]
        return sum(confidences) / len(confidences) if confidences else 0.0

    def _calc_evidence_strength(self, concept) -> str:
        """증거 강도 평가"""
        conf = concept.confidence if hasattr(concept, 'confidence') else 0.5
        if conf >= 0.8:
            return "strong"
        elif conf >= 0.6:
            return "moderate"
        else:
            return "weak"

    def _identify_risks(self) -> List[Dict[str, Any]]:
        """잠재적 리스크 식별"""
        risks = []
        ctx = self.shared_context

        low_conf_concepts = [
            c for c in (ctx.ontology_concepts or [])
            if hasattr(c, 'confidence') and c.confidence < 0.5
        ]
        if low_conf_concepts:
            risks.append({
                "type": "low_confidence_concepts",
                "count": len(low_conf_concepts),
                "severity": "medium",
                "description": f"{len(low_conf_concepts)} concepts with confidence < 0.5",
            })

        conflicts = self._detect_naming_conflicts()
        if conflicts:
            risks.append({
                "type": "unresolved_conflicts",
                "count": len(conflicts),
                "severity": "high" if len(conflicts) > 5 else "medium",
                "description": f"{len(conflicts)} naming conflicts detected",
            })

        homeomorphisms = ctx.homeomorphisms or []
        if len(ctx.tables) > 1 and len(homeomorphisms) == 0:
            risks.append({
                "type": "no_table_connections",
                "severity": "high",
                "description": "Multiple tables but no homeomorphisms found",
            })

        return risks
