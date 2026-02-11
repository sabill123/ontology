"""
Agent Orchestrator

에이전트 오케스트레이터
- 에이전트 풀 관리
- 작업 분배 및 조율
- 병렬 실행 관리
- Phase 관리
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Any, Optional, Set, Type, AsyncGenerator

from .base import AutonomousAgent, AgentState
from .agents import AGENT_TYPE_MAP
from .consensus import PipelineConsensusEngine, ConsensusResult, ConsensusType
from .consensus_coordinator import ConsensusCoordinator
from ..todo.manager import TodoManager
from ..todo.models import TodoStatus, PipelineTodo
from ..todo.events import TodoEventEmitter, get_event_emitter, TodoEventType, TodoEvent
from ..shared_context import SharedContext
from ..contracts import PhaseContractValidator, validate_phase_transition
from ..exceptions import PhaseContractViolation

logger = logging.getLogger(__name__)


@dataclass
class OrchestratorConfig:
    """오케스트레이터 설정"""
    max_concurrent_agents: int = 10      # 최대 동시 실행 에이전트 수 (v17.1: 5→10 확장)
    agent_idle_timeout: float = 0        # 에이전트 유휴 타임아웃 비활성화
    phase_timeout: float = 0             # Phase 타임아웃 비활성화 (무제한)
    enable_parallel: bool = True         # 병렬 실행 활성화
    auto_scale: bool = True              # 에이전트 자동 스케일링

    # 합의 설정 (v17.1: 토론 확장)
    enable_consensus: bool = True        # 멀티에이전트 합의 활성화
    consensus_max_rounds: int = 5        # 합의 최대 라운드 (v17.1: 3→5 확장)
    consensus_acceptance_threshold: float = 0.66  # 승인 임계값 (v17.1: 0.75→0.66 완화)
    consensus_min_agents: int = 3        # 합의에 필요한 최소 에이전트 수

    # 합의가 필요한 Todo 유형들
    consensus_required_todos: List[str] = field(default_factory=lambda: [
        "governance_decision",
        "quality_assessment",
        "ontology_approval",
        "conflict_resolution",
        "risk_assessment",
    ])

    # v6.0: Phase 계약 검증 설정
    enable_contract_validation: bool = True   # 계약 검증 활성화
    strict_contract_mode: bool = False        # 엄격 모드 (RECOMMENDED도 필수화)


@dataclass
class OrchestratorStats:
    """오케스트레이터 통계"""
    total_agents_created: int = 0
    total_todos_completed: int = 0
    total_todos_failed: int = 0
    total_execution_time: float = 0.0
    phases_completed: int = 0
    started_at: Optional[str] = None
    completed_at: Optional[str] = None

    # 합의 통계
    consensus_discussions: int = 0       # 총 토론 횟수
    consensus_accepted: int = 0          # 승인된 횟수
    consensus_rejected: int = 0          # 거부된 횟수
    consensus_provisional: int = 0       # 임시 승인 횟수
    consensus_escalated: int = 0         # 사람 검토로 에스컬레이션된 횟수

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_agents_created": self.total_agents_created,
            "total_todos_completed": self.total_todos_completed,
            "total_todos_failed": self.total_todos_failed,
            "total_execution_time": self.total_execution_time,
            "phases_completed": self.phases_completed,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "consensus_discussions": self.consensus_discussions,
            "consensus_accepted": self.consensus_accepted,
            "consensus_rejected": self.consensus_rejected,
            "consensus_provisional": self.consensus_provisional,
            "consensus_escalated": self.consensus_escalated,
        }


class AgentOrchestrator:
    """
    에이전트 오케스트레이터

    핵심 기능:
    1. 에이전트 풀 관리 - 필요에 따라 에이전트 생성/삭제
    2. Todo 기반 작업 분배 - Ready 상태 Todo를 적합한 에이전트에 할당
    3. 병렬 실행 - 독립적인 Todo들 동시 실행
    4. Phase 관리 - Discovery → Refinement → Governance 순차 진행
    5. 멀티에이전트 합의 - ConsensusCoordinator에 위임
    """

    def __init__(
        self,
        shared_context: SharedContext,
        config: Optional[OrchestratorConfig] = None,
        llm_client=None,
        event_emitter: Optional[TodoEventEmitter] = None,
    ):
        self.shared_context = shared_context
        self.config = config or OrchestratorConfig()
        self.llm_client = llm_client
        self.event_emitter = event_emitter or get_event_emitter()

        # Todo 매니저
        self.todo_manager = TodoManager(event_emitter=self.event_emitter)

        # 에이전트 풀
        self._agents: Dict[str, AutonomousAgent] = {}
        self._agent_tasks: Dict[str, asyncio.Task] = {}

        # 상태
        self._running = False
        self._current_phase: Optional[str] = None
        self.stats = OrchestratorStats()

        # 합의 엔진 초기화
        consensus_engine: Optional[PipelineConsensusEngine] = None
        if self.config.enable_consensus:
            consensus_engine = PipelineConsensusEngine(
                max_rounds=self.config.consensus_max_rounds,
                acceptance_threshold=self.config.consensus_acceptance_threshold,
                min_agents_for_consensus=self.config.consensus_min_agents,
            )
            logger.info("Consensus engine enabled for multi-agent deliberation")

        self.consensus_engine = consensus_engine

        # 합의 조율자 (SRP: 합의 관련 책임 위임)
        self.consensus = ConsensusCoordinator(
            shared_context=self.shared_context,
            consensus_engine=self.consensus_engine,
            agents=self._agents,
            stats=self.stats,
            todo_manager=self.todo_manager,
            config=self.config,
        )

        # v6.0: Phase 계약 검증기 초기화
        self.contract_validator: Optional[PhaseContractValidator] = None
        if self.config.enable_contract_validation:
            self.contract_validator = PhaseContractValidator(
                strict_mode=self.config.strict_contract_mode
            )
            logger.info("Contract validator enabled for phase transitions")

        # 콜백 설정
        self.todo_manager.set_callbacks(
            on_todo_ready=self._on_todo_ready,
            on_all_complete=self._on_all_complete,
        )

        logger.info(f"AgentOrchestrator initialized (max_concurrent={self.config.max_concurrent_agents}, consensus={self.config.enable_consensus})")

    # === 파이프라인 실행 ===

    async def run_pipeline(
        self,
        tables_data: Dict[str, Dict[str, Any]],
        phases: Optional[List[str]] = None,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        전체 파이프라인 실행

        Args:
            tables_data: 테이블 데이터
            phases: 실행할 Phase 목록 (기본: discovery, refinement, governance)

        Yields:
            진행 이벤트
        """
        phases = phases or ["discovery", "refinement", "governance"]
        self._running = True
        self.stats.started_at = datetime.now().isoformat()
        start_time = time.time()

        # 파이프라인 시작 이벤트
        self.event_emitter.pipeline_started(
            scenario=self.shared_context.scenario_name,
            phases=phases,
        )

        yield {
            "type": "pipeline_started",
            "scenario": self.shared_context.scenario_name,
            "phases": phases,
        }

        try:
            # 테이블 데이터를 컨텍스트에 로드
            self._load_tables_to_context(tables_data)

            for phase_idx, phase in enumerate(phases):
                self._current_phase = phase

                # v6.0: Phase 전환 전 계약 검증
                if phase_idx > 0 and self.contract_validator:
                    prev_phase = phases[phase_idx - 1]
                    validation_result = self.contract_validator.validate_transition(
                        source_phase=prev_phase,
                        target_phase=phase,
                        context=self.shared_context,
                    )

                    yield {
                        "type": "contract_validation",
                        "source_phase": prev_phase,
                        "target_phase": phase,
                        "is_valid": validation_result.is_valid,
                        "violations": validation_result.violations,
                        "warnings": validation_result.warnings,
                    }

                    if not validation_result.is_valid:
                        raise PhaseContractViolation(
                            message=f"Phase contract violated: {', '.join(validation_result.violations)}",
                            source_phase=prev_phase,
                            target_phase=phase,
                            missing_fields=validation_result.missing_fields,
                        )

                yield {
                    "type": "phase_starting",
                    "phase": phase,
                    "phase_index": phase_idx,
                    "total_phases": len(phases),
                }

                # Phase 실행
                async for event in self._run_phase(phase):
                    yield event

                # v27.0.1: Phase 완료 후 진행 중인 에이전트 대기 (게이트 전)
                for _wait in range(1200):  # max 120s
                    active = [a for a in self._agents.values() if a.state not in {AgentState.IDLE, AgentState.ERROR}]
                    pending = list(self._agent_tasks.keys())
                    if not active and not pending:
                        break
                    await asyncio.sleep(0.1)

                # v14.0: Phase 완료 후 검증 게이트 실행 (폴백 적용)
                gate_result = None
                if phase == "discovery":
                    gate_result = self.shared_context.validate_phase1_output()
                    if gate_result.get("fallbacks_applied"):
                        logger.warning(f"[v14.0 Gate] Phase 1 fallbacks applied: {gate_result['fallbacks_applied']}")
                elif phase == "refinement":
                    gate_result = self.shared_context.validate_phase2_output()
                    if gate_result.get("fallbacks_applied"):
                        logger.warning(f"[v14.0 Gate] Phase 2 fallbacks applied: {gate_result['fallbacks_applied']}")
                    # v27.0.1: 중복 개념 제거 (architect vs fallback_gate)
                    dedup_count = self.shared_context.dedup_ontology_concepts()
                    if dedup_count > 0:
                        logger.info(f"[v27.0.1] Removed {dedup_count} duplicate concepts before governance")
                elif phase == "governance":
                    gate_result = self.shared_context.validate_phase3_output()
                    if gate_result.get("fallbacks_applied"):
                        logger.warning(f"[v17.1 Gate] Phase 3 fallbacks applied: {gate_result['fallbacks_applied']}")

                    # v17.1: 통합 연결 자동화 (Phase 3 완료 후)
                    await self._run_v17_integration()

                # v14.0 Enhanced: TodoManager에서 terminate_mode/completion_signal 메트릭 수집
                phase_gate_metrics = self.todo_manager.get_phase_gate_metrics(phase)

                if not phase_gate_metrics.get("gate_passed", True):
                    logger.warning(
                        f"[v14.0 Gate] Phase {phase} gate metrics failed: "
                        f"{phase_gate_metrics.get('gate_warnings', [])}"
                    )
                    if gate_result:
                        gate_result.setdefault("warnings", []).extend(
                            phase_gate_metrics.get("gate_warnings", [])
                        )

                if gate_result:
                    yield {
                        "type": "phase_gate_validation",
                        "phase": phase,
                        "gate_valid": gate_result.get("valid", False),
                        "warnings": gate_result.get("warnings", []),
                        "fallbacks_applied": gate_result.get("fallbacks_applied", []),
                        "terminate_mode_counts": phase_gate_metrics.get("terminate_mode_counts", {}),
                        "completion_signal_counts": phase_gate_metrics.get("completion_signal_counts", {}),
                        "silent_failure_ratio": phase_gate_metrics.get("silent_failure_ratio", 0.0),
                        "confidence_analysis": phase_gate_metrics.get("confidence_analysis", {}),
                    }

                self.stats.phases_completed += 1

                yield {
                    "type": "phase_completed",
                    "phase": phase,
                    "progress": (phase_idx + 1) / len(phases),
                }

        except Exception as e:
            logger.error(f"Pipeline error: {e}", exc_info=True)
            self.event_emitter.emit(TodoEvent.create(
                event_type=TodoEventType.PIPELINE_ERROR,
                message=str(e),
            ))
            yield {
                "type": "pipeline_error",
                "error": str(e),
            }

        finally:
            await self._cleanup()

            self._running = False
            self.stats.completed_at = datetime.now().isoformat()
            self.stats.total_execution_time = time.time() - start_time

            self.event_emitter.pipeline_completed(
                execution_time=self.stats.total_execution_time,
                stats=self.stats.to_dict(),
            )

            yield {
                "type": "pipeline_completed",
                "stats": self.stats.to_dict(),
                "context_stats": self.shared_context.get_statistics(),
            }

    async def _run_phase(self, phase: str) -> AsyncGenerator[Dict[str, Any], None]:
        """단일 Phase 실행 (v15.0 - EMC 트리거 포함)"""
        phase_start = time.time()

        # v15.0: EMC - Phase 시작 전 컨텍스트 압축 검사
        if self.shared_context.should_consolidate():
            logger.info(f"[EMC] Triggering context consolidation before phase {phase}")
            consolidation_stats = self.shared_context.consolidate_context()
            yield {
                "type": "context_consolidated",
                "phase": phase,
                "stats": consolidation_stats,
            }

        # Phase Todo 초기화
        todo_ids = self.todo_manager.initialize_phase(phase)

        yield {
            "type": "todos_initialized",
            "phase": phase,
            "todo_count": len(todo_ids),
        }

        # 에이전트 풀 준비
        self._prepare_agents_for_phase(phase)

        # 합의 엔진에 에이전트 등록
        if self.consensus_engine:
            self.consensus.register_agents(phase)

        # 작업 루프
        loop_count = 0
        max_idle_loops = 500
        idle_loop_count = 0
        max_total_loops = 2000
        consensus_completed_todos: set = set()
        while not self.todo_manager.is_phase_complete(phase):
            loop_count += 1
            if loop_count >= max_total_loops:
                logger.error(f"[SAFETY] Phase {phase} exceeded {max_total_loops} loops, forcing completion")
                print(f"[ORCHESTRATOR] Phase '{phase}' SAFETY BREAK: {loop_count} total loops")
                break
            if self.config.phase_timeout > 0 and time.time() - phase_start > self.config.phase_timeout:
                logger.warning(f"Phase {phase} timed out")
                yield {
                    "type": "phase_timeout",
                    "phase": phase,
                }
                break

            if loop_count % 100 == 1:
                all_todos = self.todo_manager.get_todos_by_phase(phase)
                status_summary = {}
                for t in all_todos:
                    s = t.status.value
                    if s not in status_summary:
                        status_summary[s] = []
                    status_summary[s].append(t.name)
                print(f"[ORCHESTRATOR] Phase '{phase}' loop #{loop_count}, elapsed={time.time()-phase_start:.1f}s")
                for s, names in status_summary.items():
                    print(f"  [{s}]: {names}")
                active_agents = [f"{a.agent_id}({a.agent_type}:{a.state.value})" for a in self._agents.values() if a.state != AgentState.IDLE]
                print(f"  Active agents: {active_agents}")
                print(f"  Pending tasks: {list(self._agent_tasks.keys())}")

            # Ready 상태 Todo 처리
            ready_todos = self.todo_manager.get_ready_todos()
            ready_todos = [t for t in ready_todos if t.todo_id not in consensus_completed_todos]

            if ready_todos and self._can_spawn_agent():
                for todo in ready_todos[:self._available_slots()]:
                    if self.consensus.requires_consensus(todo):
                        needs_agent_execution = False
                        async for event in self.consensus.run_consensus_discussion(todo):
                            yield event
                            if event.get("skip_todo_completion"):
                                needs_agent_execution = True

                        if needs_agent_execution:
                            agent = await self._assign_agent_to_todo(todo)
                            if agent:
                                yield {
                                    "type": "todo_assigned",
                                    "todo_id": todo.todo_id,
                                    "agent_id": agent.agent_id,
                                    "todo_name": todo.name,
                                    "mode": "post_lightweight_consensus",
                                }
                                # v25.2: 에이전트 할당 성공 시에만 consensus 완료 처리
                                consensus_completed_todos.add(todo.todo_id)
                            # 할당 실패 시 다음 루프에서 재시도 (consensus_completed_todos에 추가하지 않음)
                        else:
                            # consensus에서 에이전트 실행 불필요 판단 → 완료 처리
                            consensus_completed_todos.add(todo.todo_id)
                    else:
                        agent = await self._assign_agent_to_todo(todo)
                        if agent:
                            yield {
                                "type": "todo_assigned",
                                "todo_id": todo.todo_id,
                                "agent_id": agent.agent_id,
                                "todo_name": todo.name,
                            }

            # 진행률 보고
            stats = self.todo_manager.get_phase_stats(phase)
            yield {
                "type": "phase_progress",
                "phase": phase,
                "stats": stats.to_dict(),
                "progress": stats.completion_rate,
            }

            # v22.1: BLOCKED/FAILED 상태 체크
            all_todos = self.todo_manager.get_todos_by_phase(phase)
            blocked_or_failed = [t for t in all_todos if t.status in {TodoStatus.BLOCKED, TodoStatus.FAILED}]
            active_or_ready = [t for t in all_todos if t.status in {TodoStatus.READY, TodoStatus.IN_PROGRESS, TodoStatus.PENDING}]

            if blocked_or_failed and not active_or_ready:
                logger.warning(f"[v22.1] Phase {phase} cannot progress - all remaining todos blocked/failed")
                print(f"[ORCHESTRATOR] Phase '{phase}' EARLY EXIT: {len(blocked_or_failed)} todos blocked/failed, no active todos")
                yield {
                    "type": "phase_blocked",
                    "phase": phase,
                    "blocked_todos": [t.name for t in blocked_or_failed],
                }
                break

            if not ready_todos and not self._agent_tasks:
                # v27.0.1: active agent가 있으면 idle count 리셋 (continuation 대기)
                active_agents = [a for a in self._agents.values() if a.state not in {AgentState.IDLE, AgentState.ERROR}]
                if active_agents:
                    idle_loop_count = 0
                else:
                    idle_loop_count += 1
                    if idle_loop_count >= max_idle_loops:
                        logger.warning(f"[v22.1] Phase {phase} idle loop limit reached ({max_idle_loops})")
                        print(f"[ORCHESTRATOR] Phase '{phase}' IDLE TIMEOUT: {idle_loop_count} idle loops")
                        break
            else:
                idle_loop_count = 0

            await asyncio.sleep(0.1)

        # Phase 완료
        phase_time = time.time() - phase_start
        phase_stats = self.todo_manager.get_phase_stats(phase)

        self.event_emitter.phase_completed(
            phase=phase,
            execution_time=phase_time,
            stats=phase_stats.to_dict(),
        )

        yield {
            "type": "phase_stats",
            "phase": phase,
            "execution_time": phase_time,
            "stats": phase_stats.to_dict(),
        }

    # === 테이블 데이터 로드 ===

    def _load_tables_to_context(self, tables_data: Dict[str, Dict[str, Any]]) -> None:
        """테이블 데이터를 컨텍스트에 로드"""
        from ..shared_context import TableInfo

        for table_name, table_data in tables_data.items():
            raw_columns = table_data.get("columns", [])
            if isinstance(raw_columns, dict):
                columns = [
                    {"name": col_name, **col_info}
                    for col_name, col_info in raw_columns.items()
                    if isinstance(col_info, dict)
                ]
            else:
                columns = raw_columns if isinstance(raw_columns, list) else []

            table_info = TableInfo(
                name=table_name,
                source=table_data.get("source", "unknown"),
                columns=columns,
                row_count=table_data.get("row_count", 0),
                sample_data=table_data.get("sample_data", []),
                metadata=table_data.get("metadata", {}),
                primary_keys=table_data.get("primary_keys", []),
                foreign_keys=table_data.get("foreign_keys", []),
            )
            self.shared_context.add_table(table_info)

        logger.info(f"Loaded {len(tables_data)} tables to context")

    # === 에이전트 관리 ===

    def _prepare_agents_for_phase(self, phase: str) -> None:
        """Phase에 필요한 에이전트 준비"""
        phase_agent_types = {
            "discovery": ["data_analyst", "tda_expert", "schema_analyst", "value_matcher",
                         "entity_classifier", "relationship_detector"],
            "refinement": ["ontology_architect", "conflict_resolver",
                          "quality_judge", "semantic_validator"],
            "governance": ["governance_strategist", "action_prioritizer",
                          "risk_assessor", "policy_generator"],
        }

        agent_types = phase_agent_types.get(phase, [])

        for agent_type in agent_types:
            if agent_type not in self._agents:
                self._create_agent(agent_type)

    def _create_agent(self, agent_type: str) -> Optional[AutonomousAgent]:
        """에이전트 생성"""
        if agent_type not in AGENT_TYPE_MAP:
            logger.warning(f"Unknown agent type: {agent_type}")
            return None

        agent_class = AGENT_TYPE_MAP[agent_type]
        agent = agent_class(
            todo_manager=self.todo_manager,
            shared_context=self.shared_context,
            llm_client=self.llm_client,
        )

        self._agents[agent.agent_id] = agent
        self.stats.total_agents_created += 1

        logger.info(f"Agent created: {agent.agent_id} ({agent_type})")
        return agent

    async def _assign_agent_to_todo(self, todo: PipelineTodo) -> Optional[AutonomousAgent]:
        """Todo에 에이전트 할당 및 실행"""
        print(f"[ASSIGN] Looking for agent for '{todo.name}' (required_type={todo.required_agent_type})")

        agent = self._find_suitable_agent(todo)

        if not agent:
            if todo.required_agent_type:
                print(f"[ASSIGN] No idle agent found, creating new '{todo.required_agent_type}'")
                agent = self._create_agent(todo.required_agent_type)

        if not agent:
            print(f"[ASSIGN] FAILED: No suitable agent for '{todo.name}'")
            logger.warning(f"No suitable agent for todo: {todo.todo_id}")
            return None

        assigned = self.todo_manager.assign_todo(todo.todo_id, agent.agent_id)
        if not assigned:
            print(f"[ASSIGN] FAILED: Could not assign '{todo.name}' to agent {agent.agent_id}")
            return None

        print(f"[ASSIGN] SUCCESS: '{todo.name}' -> {agent.agent_type} ({agent.agent_id})")

        task = asyncio.create_task(self._run_agent_task(agent, todo))
        self._agent_tasks[todo.todo_id] = task

        return agent

    async def _run_agent_task(self, agent: AutonomousAgent, todo: PipelineTodo) -> None:
        """에이전트 작업 실행"""
        print(f"[AGENT_TASK] START: {todo.name} (type={todo.task_type}, agent={agent.agent_type})")
        try:
            result = await agent.run_single(todo.todo_id)

            if result and result.success:
                self.stats.total_todos_completed += 1
                print(f"[AGENT_TASK] SUCCESS: {todo.name} (agent={agent.agent_type})")
            else:
                self.stats.total_todos_failed += 1
                print(f"[AGENT_TASK] FAILED: {todo.name} (agent={agent.agent_type}, error={result.error if result else 'no result'})")

        except Exception as e:
            logger.error(f"Agent task error: {e}", exc_info=True)
            print(f"[AGENT_TASK] EXCEPTION: {todo.name} (agent={agent.agent_type}, error={e})")
            self.todo_manager.fail_todo(todo.todo_id, str(e))
            self.stats.total_todos_failed += 1

        finally:
            if todo.todo_id in self._agent_tasks:
                del self._agent_tasks[todo.todo_id]
            print(f"[AGENT_TASK] CLEANUP: {todo.name} removed from pending tasks")

    def _find_suitable_agent(self, todo: PipelineTodo) -> Optional[AutonomousAgent]:
        """
        Todo에 적합한 에이전트 찾기 (v15.0 - DAR: Drift-Aware Routing)

        선택 기준:
        1. 타입/Phase 매칭
        2. IDLE 상태
        3. ASI 점수 (높을수록 우선)
        """
        candidates = []

        for agent in self._agents.values():
            if agent.state != AgentState.IDLE:
                continue

            if todo.required_agent_type:
                if agent.agent_type == todo.required_agent_type:
                    candidates.append(agent)
            else:
                if agent.phase == todo.phase:
                    candidates.append(agent)

        if not candidates:
            return None

        # v15.0: DAR - ASI 점수 기반 정렬
        candidates_with_asi = []
        for agent in candidates:
            asi_score = agent.metrics.calculate_asi()
            candidates_with_asi.append((agent, asi_score))

            drift_warning = agent.metrics.get_drift_warning()
            if drift_warning:
                logger.warning(f"[DAR] {agent.agent_id}: {drift_warning}")

        candidates_with_asi.sort(key=lambda x: x[1], reverse=True)

        selected_agent, selected_asi = candidates_with_asi[0]

        if selected_asi < 0.4:
            logger.warning(
                f"[DAR] Selected agent {selected_agent.agent_id} has low ASI ({selected_asi:.2f}). "
                f"Consider agent refresh."
            )

        logger.debug(
            f"[DAR] Selected {selected_agent.agent_id} for {todo.name} "
            f"(ASI={selected_asi:.2f}, candidates={len(candidates)})"
        )

        return selected_agent

    def _can_spawn_agent(self) -> bool:
        """새 에이전트 생성 가능 여부"""
        active_count = len([a for a in self._agents.values() if a.state != AgentState.IDLE])
        return active_count < self.config.max_concurrent_agents

    def _available_slots(self) -> int:
        """사용 가능한 에이전트 슬롯 수"""
        active_count = len([a for a in self._agents.values() if a.state != AgentState.IDLE])
        return max(0, self.config.max_concurrent_agents - active_count)

    # === 콜백 ===

    def _on_todo_ready(self, todo: PipelineTodo) -> None:
        """Todo Ready 콜백"""
        logger.debug(f"Todo ready: {todo.todo_id}")

    def _on_all_complete(self) -> None:
        """모든 Todo 완료 콜백"""
        logger.info("All todos completed!")

    # === v17.1: 통합 연결 자동화 ===

    async def _run_v17_integration(self) -> None:
        """
        v17.1: 파이프라인 완료 후 통합 연결 자동화

        1. 컴포넌트 간 참조 연결 (link_components)
        2. UnifiedInsightPipeline 실행
        3. 통합 메트릭 계산
        """
        logger.info("[v17.1] Running integration automation...")

        ctx = self.shared_context

        try:
            # 1. 온톨로지 개념 ↔ 거버넌스 결정 연결
            for decision in ctx.governance_decisions:
                concept_id = decision.concept_id
                decision_id = decision.decision_id

                ctx.link_components(
                    source_id=concept_id,
                    target_id=decision_id,
                    relation_type="has_governance_decision"
                )

                for action in ctx.action_backlog:
                    if isinstance(action, dict):
                        action_concept = action.get("target_concept", action.get("concept_id", ""))
                        if action_concept == concept_id:
                            ctx.link_components(
                                source_id=decision_id,
                                target_id=action.get("action_id", ""),
                                relation_type="recommends_action"
                            )

            logger.info(f"[v17.1] Linked {len(ctx.component_references)} component references")

            # 2. UnifiedInsightPipeline 실행
            llm_client = self.llm_client

            if llm_client is None and ctx.has_v17_services():
                services = ctx.get_all_v17_services()
                for service_name in ["semantic_searcher", "whatif_analyzer", "remediation_engine"]:
                    service = services.get(service_name)
                    if service and hasattr(service, 'llm_client') and service.llm_client:
                        llm_client = service.llm_client
                        logger.info(f"[v17.1] LLM client extracted from {service_name}")
                        break

            tables = {}
            data = {}
            for table_name, table_info in ctx.tables.items():
                tables[table_name] = {
                    "columns": table_info.columns if hasattr(table_info, 'columns') else [],
                    "row_count": table_info.row_count if hasattr(table_info, 'row_count') else 0,
                }
                if hasattr(table_info, 'sample_data') and table_info.sample_data:
                    data[table_name] = table_info.sample_data

            if tables and data:
                try:
                    from .analysis.unified_insight_pipeline import UnifiedInsightPipeline

                    domain = ctx.get_industry() if hasattr(ctx, 'get_industry') else "general"
                    pipeline = UnifiedInsightPipeline(
                        context=ctx,
                        llm_client=llm_client,
                        domain=domain,
                        enable_simulation=True,
                        enable_llm=llm_client is not None,
                    )

                    correlations = []
                    for causal in (ctx.causal_relationships or []):
                        if isinstance(causal, dict):
                            correlations.append(causal)

                    result = await pipeline.run(
                        tables=tables,
                        data=data,
                        correlations=correlations[:10],
                    )

                    for insight in result.unified_insights:
                        ctx.add_unified_insight(insight.to_dict())

                    logger.info(f"[v17.1] UnifiedInsightPipeline: {len(result.unified_insights)} insights generated")

                except Exception as e:
                    logger.warning(f"[v17.1] UnifiedInsightPipeline failed: {e}")

            # 3. 통합 메트릭 계산
            connectivity = ctx.calculate_component_connectivity()
            ctx.update_integration_metrics("algorithm_confidence", 0.75)
            ctx.update_integration_metrics("pipeline_completed", 1.0)

            logger.info(f"[v17.1] Integration complete: connectivity={connectivity:.2%}")

        except Exception as e:
            logger.error(f"[v17.1] Integration automation failed: {e}", exc_info=True)

    # === 정리 ===

    async def _cleanup(self) -> None:
        """리소스 정리"""
        for agent in self._agents.values():
            agent.stop()

        for task in self._agent_tasks.values():
            task.cancel()

        if self._agent_tasks:
            await asyncio.gather(*self._agent_tasks.values(), return_exceptions=True)

        self._agents.clear()
        self._agent_tasks.clear()

        logger.info("Orchestrator cleanup completed")

    # === 상태 조회 ===

    def get_status(self) -> Dict[str, Any]:
        """오케스트레이터 상태 조회"""
        status = {
            "running": self._running,
            "current_phase": self._current_phase,
            "stats": self.stats.to_dict(),
            "agents": {
                agent_id: agent.get_status()
                for agent_id, agent in self._agents.items()
            },
            "todo_stats": self.todo_manager.get_stats().to_dict(),
            "todo_progress": self.todo_manager.get_progress(),
        }

        if self.consensus_engine:
            status["consensus_engine"] = self.consensus_engine.get_status()

        return status

    def get_dag_visualization(self) -> str:
        """DAG 시각화"""
        return self.todo_manager.visualize_dag()

    def get_consensus_summary(self) -> Dict[str, Any]:
        """합의 통계 요약"""
        return self.consensus.get_summary()
