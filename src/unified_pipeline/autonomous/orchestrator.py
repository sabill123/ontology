"""
Agent Orchestrator

에이전트 오케스트레이터
- 에이전트 풀 관리
- 작업 분배 및 조율
- 병렬 실행 관리
- 멀티에이전트 토론/합의 (Multi-Agent Deliberation)
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
    5. 멀티에이전트 합의 - 중요 결정에 대해 에이전트간 토론/합의 진행
    """

    def __init__(
        self,
        shared_context: SharedContext,
        config: Optional[OrchestratorConfig] = None,
        llm_client=None,
        event_emitter: Optional[TodoEventEmitter] = None,
    ):
        """
        오케스트레이터 초기화

        Args:
            shared_context: 공유 컨텍스트
            config: 설정
            llm_client: LLM 클라이언트
            event_emitter: 이벤트 이미터
        """
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

        # 합의 엔진 초기화 (멀티에이전트 토론 시스템)
        self.consensus_engine: Optional[PipelineConsensusEngine] = None
        if self.config.enable_consensus:
            self.consensus_engine = PipelineConsensusEngine(
                max_rounds=self.config.consensus_max_rounds,
                acceptance_threshold=self.config.consensus_acceptance_threshold,
                min_agents_for_consensus=self.config.consensus_min_agents,
            )
            logger.info("Consensus engine enabled for multi-agent deliberation")

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
                elif phase == "governance":
                    # v17.1: Phase 3 검증 게이트
                    gate_result = self.shared_context.validate_phase3_output()
                    if gate_result.get("fallbacks_applied"):
                        logger.warning(f"[v17.1 Gate] Phase 3 fallbacks applied: {gate_result['fallbacks_applied']}")

                    # v17.1: 통합 연결 자동화 (Phase 3 완료 후)
                    await self._run_v17_integration()

                # v14.0 Enhanced: TodoManager에서 terminate_mode/completion_signal 메트릭 수집
                phase_gate_metrics = self.todo_manager.get_phase_gate_metrics(phase)

                # 메트릭 기반 추가 검증
                if not phase_gate_metrics.get("gate_passed", True):
                    logger.warning(
                        f"[v14.0 Gate] Phase {phase} gate metrics failed: "
                        f"{phase_gate_metrics.get('gate_warnings', [])}"
                    )
                    # gate_result에 경고 병합
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
                        # v14.0 Enhanced: 추가 메트릭 정보
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
            # 정리
            await self._cleanup()

            self._running = False
            self.stats.completed_at = datetime.now().isoformat()
            self.stats.total_execution_time = time.time() - start_time

            # 파이프라인 완료 이벤트
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

        # 합의 엔진에 에이전트 등록 (Phase별)
        if self.consensus_engine:
            self._register_agents_for_consensus(phase)

        # 작업 루프
        loop_count = 0
        max_idle_loops = 500  # v22.1: 무한 루프 방지
        idle_loop_count = 0
        while not self.todo_manager.is_phase_complete(phase):
            loop_count += 1
            # 타임아웃 체크 (비활성화: phase_timeout이 0이면 무제한)
            if self.config.phase_timeout > 0 and time.time() - phase_start > self.config.phase_timeout:
                logger.warning(f"Phase {phase} timed out")
                yield {
                    "type": "phase_timeout",
                    "phase": phase,
                }
                break

            # 주기적 상태 출력 (10초마다)
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

            if ready_todos and self._can_spawn_agent():
                for todo in ready_todos[:self._available_slots()]:
                    # 합의가 필요한 Todo인지 확인
                    if self._requires_consensus(todo):
                        # 멀티에이전트 토론 진행
                        needs_agent_execution = False
                        async for event in self._run_consensus_discussion(todo):
                            yield event
                            # v8.4: 경량 검증 모드면 에이전트 실행 필요
                            if event.get("skip_todo_completion"):
                                needs_agent_execution = True

                        # v8.4: 경량 검증 후 에이전트 실행
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
                    else:
                        # 일반 작업 할당
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

            # v22.1: BLOCKED/FAILED 상태 체크 - 진행 불가 시 조기 종료
            all_todos = self.todo_manager.get_todos_by_phase(phase)
            blocked_or_failed = [t for t in all_todos if t.status in {TodoStatus.BLOCKED, TodoStatus.FAILED}]
            active_or_ready = [t for t in all_todos if t.status in {TodoStatus.READY, TodoStatus.IN_PROGRESS, TodoStatus.PENDING}]

            # 모든 미완료 작업이 blocked/failed면 조기 종료
            if blocked_or_failed and not active_or_ready:
                logger.warning(f"[v22.1] Phase {phase} cannot progress - all remaining todos blocked/failed")
                print(f"[ORCHESTRATOR] Phase '{phase}' EARLY EXIT: {len(blocked_or_failed)} todos blocked/failed, no active todos")
                yield {
                    "type": "phase_blocked",
                    "phase": phase,
                    "blocked_todos": [t.name for t in blocked_or_failed],
                }
                break

            # v22.1: Idle loop 감지 (ready todo 없고 active agent 없으면)
            if not ready_todos and not self._agent_tasks:
                idle_loop_count += 1
                if idle_loop_count >= max_idle_loops:
                    logger.warning(f"[v22.1] Phase {phase} idle loop limit reached ({max_idle_loops})")
                    print(f"[ORCHESTRATOR] Phase '{phase}' IDLE TIMEOUT: {idle_loop_count} idle loops")
                    break
            else:
                idle_loop_count = 0

            # 잠시 대기
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
            # columns 형식 변환: Dict[str, Dict] -> List[Dict] (TableInfo 형식)
            raw_columns = table_data.get("columns", [])
            if isinstance(raw_columns, dict):
                # 키가 컬럼명인 dict를 list로 변환
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

        # 적합한 에이전트 찾기
        agent = self._find_suitable_agent(todo)

        if not agent:
            # 필요시 새 에이전트 생성
            if todo.required_agent_type:
                print(f"[ASSIGN] No idle agent found, creating new '{todo.required_agent_type}'")
                agent = self._create_agent(todo.required_agent_type)

        if not agent:
            print(f"[ASSIGN] FAILED: No suitable agent for '{todo.name}'")
            logger.warning(f"No suitable agent for todo: {todo.todo_id}")
            return None

        # Todo 할당
        assigned = self.todo_manager.assign_todo(todo.todo_id, agent.agent_id)
        if not assigned:
            print(f"[ASSIGN] FAILED: Could not assign '{todo.name}' to agent {agent.agent_id}")
            return None

        print(f"[ASSIGN] SUCCESS: '{todo.name}' -> {agent.agent_type} ({agent.agent_id})")

        # 비동기 실행
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
            # 태스크 정리
            if todo.todo_id in self._agent_tasks:
                del self._agent_tasks[todo.todo_id]
            print(f"[AGENT_TASK] CLEANUP: {todo.name} removed from pending tasks")

    def _find_suitable_agent(self, todo: PipelineTodo) -> Optional[AutonomousAgent]:
        """
        Todo에 적합한 에이전트 찾기 (v15.0 - DAR: Drift-Aware Routing)

        Agent Drift 완화를 위해 ASI (Agent Stability Index)를 고려하여
        안정적인 에이전트를 우선 선택합니다.

        선택 기준:
        1. 타입/Phase 매칭
        2. IDLE 상태
        3. ASI 점수 (높을수록 우선)
        """
        candidates = []

        for agent in self._agents.values():
            # 상태 체크
            if agent.state != AgentState.IDLE:
                continue

            # 타입 체크
            if todo.required_agent_type:
                if agent.agent_type == todo.required_agent_type:
                    candidates.append(agent)
            else:
                # Phase 매칭
                if agent.phase == todo.phase:
                    candidates.append(agent)

        if not candidates:
            return None

        # v15.0: DAR - ASI 점수 기반 정렬 (높은 ASI 우선)
        # 불안정한 에이전트(낮은 ASI)는 후순위로 배치
        candidates_with_asi = []
        for agent in candidates:
            asi_score = agent.metrics.calculate_asi()
            candidates_with_asi.append((agent, asi_score))

            # Drift 경고 로깅
            drift_warning = agent.metrics.get_drift_warning()
            if drift_warning:
                logger.warning(f"[DAR] {agent.agent_id}: {drift_warning}")

        # ASI 내림차순 정렬
        candidates_with_asi.sort(key=lambda x: x[1], reverse=True)

        # 가장 안정적인 에이전트 선택
        selected_agent, selected_asi = candidates_with_asi[0]

        # ASI가 너무 낮으면 경고 (하지만 실행은 계속)
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

                # 개념 → 결정 연결
                ctx.link_components(
                    source_id=concept_id,
                    target_id=decision_id,
                    relation_type="has_governance_decision"
                )

                # 결정 → 액션 연결
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

            # 2. UnifiedInsightPipeline 실행 (LLM 클라이언트 있는 경우)
            # v17.1 Fix: Orchestrator 자체의 llm_client 사용 (이미 초기화됨)
            llm_client = self.llm_client

            if llm_client is None and ctx.has_v17_services():
                # Fallback: v17 서비스에서 LLM 클라이언트 가져오기 시도
                services = ctx.get_all_v17_services()
                # semantic_searcher, whatif_analyzer 등에서 LLM 클라이언트 추출
                for service_name in ["semantic_searcher", "whatif_analyzer", "remediation_engine"]:
                    service = services.get(service_name)
                    if service and hasattr(service, 'llm_client') and service.llm_client:
                        llm_client = service.llm_client
                        logger.info(f"[v17.1] LLM client extracted from {service_name}")
                        break

            # tables와 data 준비
            tables = {}
            data = {}
            for table_name, table_info in ctx.tables.items():
                tables[table_name] = {
                    "columns": table_info.columns if hasattr(table_info, 'columns') else [],
                    "row_count": table_info.row_count if hasattr(table_info, 'row_count') else 0,
                }
                # sample_data가 있으면 사용
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

                    # 상관관계 수집 (causal_relationships 사용)
                    correlations = []
                    for causal in (ctx.causal_relationships or []):
                        if isinstance(causal, dict):
                            correlations.append(causal)

                    result = await pipeline.run(
                        tables=tables,
                        data=data,
                        correlations=correlations[:10],  # 상위 10개
                    )

                    # 통합 인사이트 저장
                    for insight in result.unified_insights:
                        ctx.add_unified_insight(insight.to_dict())

                    logger.info(f"[v17.1] UnifiedInsightPipeline: {len(result.unified_insights)} insights generated")

                except Exception as e:
                    logger.warning(f"[v17.1] UnifiedInsightPipeline failed: {e}")

            # 3. 통합 메트릭 계산
            connectivity = ctx.calculate_component_connectivity()
            ctx.update_integration_metrics("algorithm_confidence", 0.75)  # 기본값
            ctx.update_integration_metrics("pipeline_completed", 1.0)

            logger.info(f"[v17.1] Integration complete: connectivity={connectivity:.2%}")

        except Exception as e:
            logger.error(f"[v17.1] Integration automation failed: {e}", exc_info=True)

    # === 정리 ===

    async def _cleanup(self) -> None:
        """리소스 정리"""
        # 에이전트 정지
        for agent in self._agents.values():
            agent.stop()

        # 태스크 취소
        for task in self._agent_tasks.values():
            task.cancel()

        # 태스크 완료 대기
        if self._agent_tasks:
            await asyncio.gather(*self._agent_tasks.values(), return_exceptions=True)

        self._agents.clear()
        self._agent_tasks.clear()

        logger.info("Orchestrator cleanup completed")

    # === 멀티에이전트 합의 시스템 ===

    def _requires_consensus(self, todo: PipelineTodo) -> bool:
        """
        Todo가 멀티에이전트 합의를 필요로 하는지 확인 (v7.1 균형 최적화)

        v7.1: 품질 유지 + 효율성 개선
        - 핵심 품질 결정은 합의 유지 (quality_assessment, ontology_approval)
        - 단, Fast-track으로 1-2라운드 내 종료
        - conflict_resolution, risk_assessment는 단일 에이전트로 처리
        """
        if not self.consensus_engine or not self.config.enable_consensus:
            return False

        todo_name_normalized = todo.name.lower().replace(" ", "_").replace("-", "_")

        # v7.1+v22.0: 품질에 직접 영향을 주는 핵심 결정만 합의
        # v22.0: 실제 todo 이름과 일치하도록 수정
        critical_quality_decisions = [
            "quality_assessment",    # 품질 평가 - 중요
            "ontology_approval",     # 온톨로지 승인
            "ontology_proposal",     # v22.0: 실제 todo 이름
            "governance_decision",   # 거버넌스 결정
            "governance_strategy",   # v22.0: 실제 todo 이름
        ]

        for critical in critical_quality_decisions:
            if critical in todo_name_normalized:
                return True

        # v7.1: 이하는 단일 에이전트로 처리 (효율성)
        # - conflict_resolution: 충돌 해결은 Conflict Resolver 단독 처리
        # - risk_assessment: 리스크 평가는 Risk Assessor 단독 처리

        return False

    def _register_agents_for_consensus(self, phase: str) -> None:
        """Phase의 에이전트들을 합의 엔진에 등록"""
        if not self.consensus_engine:
            return

        # 기존 등록 해제
        self.consensus_engine.clear_agents()

        # Phase에 맞는 에이전트만 등록
        for agent in self._agents.values():
            if agent.phase == phase:
                self.consensus_engine.register_agent(agent)
                logger.debug(f"Registered {agent.agent_id} for consensus in {phase}")

    async def _run_consensus_discussion(
        self,
        todo: PipelineTodo,
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

        # 토론 시작 알림
        yield {
            "type": "consensus_discussion_started",
            "todo_id": todo.todo_id,
            "todo_name": todo.name,
            "phase": todo.phase,
        }

        self.stats.consensus_discussions += 1

        # === v8.2: 스마트 합의 - 경량 검증 또는 Full Consensus ===
        skip_mode, skip_reason, base_confidence = self._should_skip_consensus_v82(todo)

        if skip_mode == "full_skip":
            # Phase 1: 완전 스킵 (탐색 단계)
            logger.info(f"Full skip consensus for {todo.todo_id}: {skip_reason}")
            yield {
                "type": "consensus_skipped",
                "todo_id": todo.todo_id,
                "reason": skip_reason,
                "mode": "full_skip",
            }
            default_result = ConsensusResult(
                success=True,
                result_type=ConsensusType.ACCEPTED,
                combined_confidence=base_confidence,
                agreement_level=1.0,
                total_rounds=0,
                final_decision="auto_approved",
                reasoning=f"Auto-approved: {skip_reason}",
                dissenting_views=[],
            )
            await self._apply_consensus_result(todo, default_result)
            yield {
                "type": "consensus_discussion_completed",
                "todo_id": todo.todo_id,
                "result_type": "accepted",
                "combined_confidence": base_confidence,
                "mode": "full_skip",
            }
            return

        elif skip_mode == "lightweight":
            # Phase 2/3 단순 케이스: 경량 검증 (1회 LLM 호출로 품질 확보)
            logger.info(f"Lightweight validation for {todo.todo_id}: {skip_reason}")
            yield {
                "type": "consensus_lightweight_started",
                "todo_id": todo.todo_id,
                "reason": skip_reason,
            }

            # 경량 검증 수행 (결과만 저장, Todo 완료는 나중에)
            lightweight_result = await self._perform_lightweight_validation(todo, base_confidence)

            # v8.4: 경량 검증 결과를 context에 저장 (에이전트가 참조할 수 있도록)
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
                "skip_todo_completion": True,  # v8.4: 에이전트 실행 후 완료 처리
            }
            # v8.4: return 제거 - 에이전트 실행을 위해 caller로 제어권 반환
            return

        # skip_mode == "full_consensus": Full consensus 진행

        # === v8.0: Rich Topic Data 구성 ===
        topic_data = self._build_rich_topic_data(todo)

        # === v7.8: Evidence-Based Vote 사전 검증 (I-1 Fix) ===
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
                    # Evidence-based 투표 결과를 topic_data에 추가
                    topic_data["evidence_based_vote"] = {
                        "consensus": evidence_vote_result.get("consensus"),
                        "confidence": evidence_vote_result.get("confidence"),
                        "role_breakdown": evidence_vote_result.get("role_breakdown", {}),
                    }

                    yield {
                        "type": "evidence_based_vote_completed",
                        "todo_id": todo.todo_id,
                        "consensus": evidence_vote_result.get("consensus"),
                        "confidence": evidence_vote_result.get("confidence"),
                    }
        except Exception as e:
            logger.debug(f"[v7.8] Evidence-based vote skipped: {e}")

        # 토론 진행
        consensus_result: Optional[ConsensusResult] = None

        async for event in self.consensus_engine.negotiate(
            topic_id=todo.todo_id,
            topic_name=todo.name,
            topic_type=f"{todo.phase}_decision",
            topic_data=topic_data,
            context=self.shared_context,
        ):
            # 이벤트 전달
            yield {
                **event,
                "todo_id": todo.todo_id,
            }

            # 최종 결과 캡처
            if event.get("type") in ["consensus_reached", "max_rounds_reached"]:
                consensus_result = event.get("result")

        # 결과 처리
        if consensus_result:
            await self._apply_consensus_result(todo, consensus_result)

            yield {
                "type": "consensus_discussion_completed",
                "todo_id": todo.todo_id,
                "result_type": consensus_result.result_type.value if hasattr(consensus_result, 'result_type') else "unknown",
                "combined_confidence": consensus_result.combined_confidence if hasattr(consensus_result, 'combined_confidence') else 0,
            }
        else:
            # v7.8: 결과 없음 - StructuredDebate으로 심층 토론 시도 (I-2 Fix)
            logger.info(f"[v7.8] No consensus from negotiate, trying StructuredDebate for {todo.todo_id}")

            yield {
                "type": "consensus_structured_debate_started",
                "todo_id": todo.todo_id,
                "reason": "Standard negotiation failed, attempting structured debate",
            }

            try:
                # StructuredDebate 시도
                debate_result = await self.consensus_engine.conduct_structured_debate(
                    topic=todo.name,
                    context={"todo_id": todo.todo_id, "phase": todo.phase},
                    evidence=topic_data,
                )

                if not debate_result.get("fallback"):
                    # 구조화된 토론 결과로 ConsensusResult 생성
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

                    await self._apply_consensus_result(todo, consensus_result)

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

            # 최종 에스컬레이션
            yield {
                "type": "consensus_discussion_escalated",
                "todo_id": todo.todo_id,
                "reason": "No consensus result after structured debate",
            }
            self.stats.consensus_escalated += 1

    async def _apply_consensus_result(
        self,
        todo: PipelineTodo,
        result: ConsensusResult,
    ) -> None:
        """
        합의 결과를 Todo에 적용

        Args:
            todo: 대상 Todo
            result: 합의 결과
        """
        from ..todo.models import TodoResult

        # 통계 업데이트
        if result.result_type == ConsensusType.ACCEPTED:
            self.stats.consensus_accepted += 1
        elif result.result_type == ConsensusType.REJECTED:
            self.stats.consensus_rejected += 1
        elif result.result_type == ConsensusType.PROVISIONAL:
            self.stats.consensus_provisional += 1
        elif result.result_type == ConsensusType.ESCALATED:
            self.stats.consensus_escalated += 1
        elif result.result_type == ConsensusType.NO_AGENTS:
            # v14.0: NO_AGENTS는 provisional로 카운트
            self.stats.consensus_provisional += 1
            logger.warning(f"[v14.0] Consensus skipped due to no agents: {todo.todo_id}")

        # Todo 결과 생성
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

        # v6.3: Todo 완료 처리 - PROVISIONAL도 성공으로 처리
        # v14.0: NO_AGENTS도 provisional 성공으로 처리
        # ACCEPTED, PROVISIONAL, NO_AGENTS: 성공
        # ESCALATED, REJECTED: 실패
        if result.success or result.result_type in [ConsensusType.ACCEPTED, ConsensusType.PROVISIONAL, ConsensusType.NO_AGENTS]:
            self.todo_manager.complete_todo(todo.todo_id, todo_result)
            self.stats.total_todos_completed += 1
            logger.info(
                f"[CONSENSUS OK] {todo.todo_id}: {result.result_type.value} "
                f"(confidence: {result.combined_confidence:.2f})"
            )
        else:
            if result.result_type == ConsensusType.ESCALATED:
                # 에스컬레이션된 경우 실패 처리
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

    # === v7.0: 배치 합의 처리 ===

    async def _run_batch_consensus(
        self,
        todos: List[PipelineTodo],
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        v7.0: 여러 Todo를 하나의 합의 세션에서 배치 처리

        LLM 호출을 최소화하면서 여러 결정을 한번에 처리합니다.

        Args:
            todos: 배치 처리할 Todo 목록

        Yields:
            배치 합의 진행 이벤트
        """
        if not self.consensus_engine or not todos:
            return

        yield {
            "type": "batch_consensus_started",
            "todo_count": len(todos),
            "todo_ids": [t.todo_id for t in todos],
        }

        # 모든 Todo 데이터를 하나의 토픽으로 통합
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

        # 단일 합의 세션 진행
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

        # 배치 결과를 각 Todo에 적용
        if consensus_result:
            for todo in todos:
                await self._apply_consensus_result(todo, consensus_result)

            yield {
                "type": "batch_consensus_completed",
                "batch_id": batch_id,
                "todo_count": len(todos),
                "result_type": consensus_result.result_type.value if hasattr(consensus_result, 'result_type') else "unknown",
            }
        else:
            # 배치 실패 - 개별 처리로 폴백
            yield {
                "type": "batch_consensus_failed",
                "batch_id": batch_id,
                "fallback": "individual_processing",
            }

        self.stats.consensus_discussions += 1

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

        # 합의 엔진 상태 추가
        if self.consensus_engine:
            status["consensus_engine"] = self.consensus_engine.get_status()

        return status

    def get_dag_visualization(self) -> str:
        """DAG 시각화"""
        return self.todo_manager.visualize_dag()

    def get_consensus_summary(self) -> Dict[str, Any]:
        """합의 통계 요약"""
        return {
            "enabled": self.config.enable_consensus,
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

    # === v8.2: 스마트 합의 시스템 ===

    def _should_skip_consensus_v82(self, todo: PipelineTodo) -> tuple:
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

        # === Phase 1: 완전 스킵 (탐색 단계) ===
        if todo.phase == "discovery":
            return "full_skip", "Phase 1 discovery - exploration stage", avg_conf

        # === Phase 2: Refinement ===

        # v9.0: Quality Assessment는 항상 에이전트 실행 필요
        # Quality Judge가 concept.status를 설정해야 LLM Review가 정상 동작함
        if "quality" in todo_type or "assessment" in todo_type:
            if not concepts:
                return "full_skip", "No concepts to assess", 0.5
            # v9.0: Quality Assessment는 lightweight로 처리하되 항상 에이전트 실행 보장
            # 단순 케이스도 에이전트 실행이 필수 (status 설정 필요)
            if len(concepts) <= 2:
                return "lightweight", f"Few concepts ({len(concepts)}) - agent execution required", avg_conf
            if avg_conf >= 0.85:
                return "lightweight", f"High quality ({avg_conf:.0%}) - agent execution required", avg_conf
            return "lightweight", "Quality assessment - agent execution required", avg_conf

        # Conflict Resolution
        if "conflict" in todo_type:
            conflicts = self._detect_naming_conflicts()
            if not conflicts:
                return "lightweight", "No conflicts - lightweight validation", 0.85
            if len(conflicts) <= 2:
                return "lightweight", f"Few conflicts ({len(conflicts)}) - lightweight", 0.7
            return "full_consensus", f"Multiple conflicts ({len(conflicts)})", 0.6

        # Ontology Approval
        if "ontology" in todo_type and "approval" in todo_type:
            pending = [c for c in concepts if c.status in ["pending", "provisional"]]
            if not pending:
                return "full_skip", "No pending concepts", 0.9
            avg_pending_conf = self._calc_avg_confidence(pending)
            if avg_pending_conf >= 0.8:
                return "lightweight", f"High confidence pending ({avg_pending_conf:.0%})", avg_pending_conf
            return "full_consensus", "Pending concepts need review", avg_pending_conf

        # === Phase 3: Governance ===

        # Governance Decision
        if "governance" in todo_type and "conflict" not in todo_type:
            if not concepts:
                return "full_skip", "No concepts for governance", 0.5
            if avg_conf >= 0.8:
                return "lightweight", f"High confidence ({avg_conf:.0%}) - lightweight governance", avg_conf
            return "full_consensus", "Governance review needed", avg_conf

        # Risk Assessment
        if "risk" in todo_type:
            risks = self._identify_potential_risks()
            if not risks:
                return "lightweight", "No risks identified", 0.9
            high_risks = [r for r in risks if r.get("severity") in ["high", "critical"]]
            if not high_risks:
                return "lightweight", "Only low risks - lightweight validation", 0.8
            return "full_consensus", f"High risks ({len(high_risks)}) need review", 0.6

        # 기본: Full consensus
        return "full_consensus", "Default - full consensus", avg_conf

    async def _perform_lightweight_validation(self, todo: PipelineTodo, base_confidence: float) -> "ConsensusResult":
        """
        v8.2: 경량 검증 - 1회 LLM 호출로 품질 확보

        Full consensus 대신 단일 검증으로 품질 메트릭 생성
        """
        from .consensus import ConsensusResult

        todo_type = todo.name.lower().replace(" ", "_").replace("-", "_")
        ctx = self.shared_context
        concepts = ctx.ontology_concepts or []

        # 경량 검증 프롬프트 생성
        validation_prompt = self._build_lightweight_validation_prompt(todo, concepts)

        # LLM 호출 (사용 가능한 에이전트 중 하나 사용)
        validator_agent = None

        # 1. validator/judge 에이전트 우선
        for agent_id, agent in self._agents.items():
            if "validator" in agent_id.lower() or "judge" in agent_id.lower():
                validator_agent = agent
                break

        # 2. 없으면 아무 에이전트나 사용
        if not validator_agent and self._agents:
            validator_agent = next(iter(self._agents.values()))

        if not validator_agent:
            # 에이전트 없으면 데이터 기반 confidence 계산
            calculated_confidence = self._calculate_data_based_confidence(todo, concepts)
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
            # LLM 검증 호출 (에이전트의 call_llm 메서드 사용)
            response = await validator_agent.call_llm(
                user_prompt=validation_prompt,
                temperature=0.3,
                max_tokens=500,
            )

            # 응답 파싱
            validated_confidence, validation_reasoning = self._parse_lightweight_response(response, base_confidence)

            # 결과 유형 결정
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
                agreement_level=0.85,  # 단일 검증이므로 높은 합의도
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

    def _build_lightweight_validation_prompt(self, todo: PipelineTodo, concepts: list) -> str:
        """v8.3: 강화된 경량 검증 프롬프트"""
        todo_type = todo.name.lower()
        avg_conf = self._calc_avg_confidence(concepts) if concepts else 0.5
        ctx = self.shared_context

        # 품질 지표 수집
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
        """v8.3: 강화된 경량 검증 응답 파싱"""
        import re

        confidence = base_confidence
        reasoning = "Lightweight validation completed"

        # Quality Score 파싱 (우선)
        quality_match = re.search(r'\[QUALITY_SCORE:\s*([\d.]+)\]', response, re.IGNORECASE)
        if quality_match:
            try:
                confidence = float(quality_match.group(1))
                confidence = max(0.0, min(1.0, confidence))
            except ValueError:
                pass
        else:
            # 기존 CONFIDENCE 포맷도 지원
            conf_match = re.search(r'\[CONFIDENCE:\s*([\d.]+)\]', response, re.IGNORECASE)
            if conf_match:
                try:
                    confidence = float(conf_match.group(1))
                    confidence = max(0.0, min(1.0, confidence))
                except ValueError:
                    pass

        # Decision 파싱 (보너스 적용)
        decision_match = re.search(r'\[DECISION:\s*(APPROVE|PROVISIONAL|NEEDS_REVIEW|REJECT)\]', response, re.IGNORECASE)
        if decision_match:
            decision = decision_match.group(1).upper()
            # Decision에 따른 confidence 보정
            if decision == "APPROVE" and confidence < 0.75:
                confidence = max(confidence, 0.75)  # APPROVE면 최소 75%
            elif decision == "PROVISIONAL" and confidence < 0.5:
                confidence = max(confidence, 0.5)

        # Reasoning 파싱
        reason_match = re.search(r'\[REASONING:\s*(.+?)\]', response, re.IGNORECASE | re.DOTALL)
        if reason_match:
            reasoning = reason_match.group(1).strip()[:200]
        else:
            # 전체 응답에서 추출
            reasoning = response[:200] if response else reasoning

        return confidence, reasoning

    def _calculate_data_based_confidence(self, todo: PipelineTodo, concepts: list) -> float:
        """v8.2: 데이터 기반 신뢰도 계산 (LLM 없이)"""
        if not concepts:
            return 0.5

        # 기본 메트릭
        avg_conf = self._calc_avg_confidence(concepts)
        num_concepts = len(concepts)

        # 상태별 가중치
        approved_count = len([c for c in concepts if c.status == "approved"])
        provisional_count = len([c for c in concepts if c.status == "provisional"])
        pending_count = len([c for c in concepts if c.status == "pending"])

        # 승인율 계산
        if num_concepts > 0:
            approval_rate = approved_count / num_concepts
            provisional_rate = provisional_count / num_concepts
        else:
            approval_rate = 0
            provisional_rate = 0

        # 종합 신뢰도: 평균 confidence + 상태 보너스
        status_bonus = (approval_rate * 0.2) + (provisional_rate * 0.1)
        calculated = min(1.0, avg_conf + status_bonus)

        return round(calculated, 2)

    async def _apply_quality_assessment_status(self, lightweight_result: "ConsensusResult") -> None:
        """
        v8.4: 경량 검증 시 concept.status 설정

        Quality Judge 에이전트가 스킵되어도 concept.status를 설정하여
        Phase 3의 LLM 리뷰가 정상적으로 실행되도록 함
        """
        concepts = self.shared_context.ontology_concepts or []
        if not concepts:
            return

        # 경량 검증 결과에 따른 상태 결정
        confidence = lightweight_result.combined_confidence
        result_type = lightweight_result.result_type

        for concept in concepts:
            # 기존 상태가 approved/rejected가 아니면 업데이트
            if concept.status not in ["approved", "rejected"]:
                if result_type == ConsensusType.ACCEPTED and confidence >= 0.7:
                    concept.status = "approved"
                elif result_type == ConsensusType.ACCEPTED or confidence >= 0.5:
                    concept.status = "provisional"
                else:
                    concept.status = "pending"

                # agent_assessments에 경량 검증 결과 추가
                if not hasattr(concept, 'agent_assessments') or concept.agent_assessments is None:
                    concept.agent_assessments = {}

                concept.agent_assessments["quality_judge_lightweight"] = {
                    "decision": concept.status,
                    "confidence": confidence,
                    "reasoning": f"Lightweight validation: {lightweight_result.reasoning}",
                    "mode": "lightweight",
                }

        logger.info(f"Applied quality assessment status to {len(concepts)} concepts (lightweight mode)")

    # === v8.1: Legacy skip function (deprecated) ===

    def _should_skip_consensus(self, todo: PipelineTodo) -> tuple:
        """
        v8.1: 강화된 합의 스킵 조건

        핵심 원칙: "의미 있는 consensus만 진행"
        - Phase 1: 탐색 단계로 consensus 불필요
        - Phase 2: 품질 문제나 충돌이 있을 때만 진행
        - Phase 3: 최종 결정 단계이나 높은 confidence면 자동 승인

        Returns:
            (skip: bool, reason: str)
        """
        todo_type = todo.name.lower().replace(" ", "_").replace("-", "_")
        ctx = self.shared_context
        concepts = ctx.ontology_concepts or []

        # === Phase 1: Discovery 단계는 consensus 불필요 ===
        if todo.phase == "discovery":
            return True, "Phase 1 discovery - exploration stage, no consensus needed"

        # === Phase 2: Refinement ===

        # Quality Assessment: 개념이 너무 적으면 토론 불필요
        if "quality" in todo_type or "assessment" in todo_type:
            if not concepts:
                return True, "No ontology concepts to assess"
            if len(concepts) <= 2:
                return True, f"Too few concepts ({len(concepts)}) for meaningful quality discussion"
            # 이미 높은 품질이면 스킵
            avg_conf = self._calc_avg_confidence(concepts)
            if avg_conf >= 0.8:
                return True, f"Already high quality ({avg_conf:.0%}) - auto approve"

        # Conflict Resolution: 실제 충돌이 없으면 스킵
        if "conflict" in todo_type:
            conflicts = self._detect_naming_conflicts()
            if not conflicts:
                return True, "No conflicts detected - auto approve"

        # Ontology Approval: pending 상태가 없으면 스킵
        if "ontology" in todo_type and "approval" in todo_type:
            pending = [c for c in concepts if c.status in ["pending", "provisional"]]
            if not pending:
                return True, "No pending concepts to approve"
            # 모든 pending이 높은 confidence면 자동 승인
            avg_pending_conf = self._calc_avg_confidence(pending)
            if avg_pending_conf >= 0.85:
                return True, f"All pending concepts high confidence ({avg_pending_conf:.0%}) - auto approve"

        # === Phase 3: Governance ===

        # Governance Decision: 높은 confidence면 자동 승인
        if "governance" in todo_type and "conflict" not in todo_type:
            if not concepts:
                return True, "No ontology concepts for governance"
            avg_conf = self._calc_avg_confidence(concepts)
            if avg_conf >= 0.85:
                return True, f"High overall confidence ({avg_conf:.0%}) - auto approve governance"

        # Risk Assessment: 리스크 요소가 없으면 스킵
        if "risk" in todo_type:
            potential_risks = self._identify_potential_risks()
            if not potential_risks:
                return True, "No potential risks identified - auto approve"
            # 모든 리스크가 low severity면 스킵
            high_risks = [r for r in potential_risks if r.get("severity") in ["high", "critical"]]
            if not high_risks:
                return True, "Only low-severity risks - auto approve"

        return False, ""

    def _build_rich_topic_data(self, todo: PipelineTodo) -> Dict[str, Any]:
        """
        v8.1: 경량화된 topic_data 구성

        핵심 원칙: "토론에 필요한 최소 정보만 포함"
        - 요약 메트릭 위주
        - 세부 데이터는 상위 3-5개만
        - 토큰 사용량 최소화
        """
        todo_type = todo.name.lower().replace(" ", "_").replace("-", "_")
        ctx = self.shared_context
        concepts = ctx.ontology_concepts or []

        # 기본 메트릭 (항상 포함, 경량)
        base_data = {
            "phase": todo.phase,
            "decision_type": todo_type,
            "concepts_count": len(concepts),
            "avg_confidence": round(self._calc_avg_confidence(concepts), 2),
        }

        # === Phase 2: Quality Assessment ===
        if "quality" in todo_type or "assessment" in todo_type:
            base_data["quality_summary"] = {
                "total": len(concepts),
                "by_status": self._count_concepts_by_status(concepts),
                "avg_confidence": round(self._calc_avg_confidence(concepts), 2),
            }
            # 낮은 confidence 개념만 (상위 3개)
            low_conf = sorted(
                [c for c in concepts if c.confidence < 0.6],
                key=lambda x: x.confidence
            )[:3]
            if low_conf:
                base_data["low_confidence_concepts"] = [
                    {"name": c.name, "confidence": round(c.confidence, 2)}
                    for c in low_conf
                ]

        # === Phase 2: Conflict Resolution ===
        elif "conflict" in todo_type:
            conflicts = self._detect_naming_conflicts()
            base_data["conflict_summary"] = {
                "total": len(conflicts),
                "high_similarity": len([c for c in conflicts if c.get("similarity", 0) > 0.85]),
            }
            # 상위 5개 충돌만
            base_data["top_conflicts"] = conflicts[:5]

        # === Phase 2: Ontology Approval ===
        elif "ontology" in todo_type and "approval" in todo_type:
            pending = [c for c in concepts if c.status in ["pending", "provisional"]]
            base_data["approval_summary"] = {
                "total_pending": len(pending),
                "avg_confidence": round(self._calc_avg_confidence(pending), 2),
            }
            # 상위 3개 후보만
            base_data["top_candidates"] = [
                {"name": c.name, "confidence": round(c.confidence, 2), "status": c.status}
                for c in pending[:3]
            ]

        # === Phase 3: Governance Decision ===
        elif "governance" in todo_type:
            base_data["governance_summary"] = {
                "total_concepts": len(concepts),
                "approved": len([c for c in concepts if c.status == "approved"]),
                "avg_confidence": round(self._calc_avg_confidence(concepts), 2),
            }
            # 비즈니스 인사이트 요약 (상위 2개)
            insights = ctx.business_insights or []
            if insights:
                base_data["key_insights"] = [
                    getattr(ins, 'description', str(ins))[:100]
                    for ins in insights[:2]
                ]

        # === Phase 3: Risk Assessment ===
        elif "risk" in todo_type:
            risks = self._identify_potential_risks()
            base_data["risk_summary"] = {
                "total_risks": len(risks),
                "high_severity": len([r for r in risks if r.get("severity") in ["high", "critical"]]),
            }
            # 고위험 항목만 (상위 3개)
            high_risks = [r for r in risks if r.get("severity") in ["high", "critical"]][:3]
            if high_risks:
                base_data["high_risks"] = high_risks

        return base_data

    def _detect_naming_conflicts(self) -> List[Dict[str, Any]]:
        """v8.0: 명명 충돌 탐지"""
        conflicts = []
        concepts = self.shared_context.ontology_concepts or []

        # 이름 유사도 기반 충돌 탐지
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
        """간단한 이름 유사도 계산"""
        n1, n2 = name1.lower(), name2.lower()
        if n1 == n2:
            return 1.0

        # Jaccard 유사도 (단어 기반)
        words1 = set(n1.replace("_", " ").split())
        words2 = set(n2.replace("_", " ").split())

        if not words1 or not words2:
            return 0.0

        intersection = len(words1 & words2)
        union = len(words1 | words2)
        return intersection / union if union > 0 else 0.0

    def _count_concepts_by_type(self, concepts) -> Dict[str, int]:
        """개념 타입별 카운트"""
        counts = {}
        for c in concepts:
            t = c.concept_type if hasattr(c, 'concept_type') else "unknown"
            counts[t] = counts.get(t, 0) + 1
        return counts

    def _count_concepts_by_status(self, concepts) -> Dict[str, int]:
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

    def _identify_potential_risks(self) -> List[Dict[str, Any]]:
        """잠재적 리스크 요소 식별"""
        risks = []
        ctx = self.shared_context

        # 낮은 신뢰도 개념
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

        # 충돌 미해결
        conflicts = self._detect_naming_conflicts()
        if conflicts:
            risks.append({
                "type": "unresolved_conflicts",
                "count": len(conflicts),
                "severity": "high" if len(conflicts) > 5 else "medium",
                "description": f"{len(conflicts)} naming conflicts detected",
            })

        # 테이블 연결 부족
        homeomorphisms = ctx.homeomorphisms or []
        if len(ctx.tables) > 1 and len(homeomorphisms) == 0:
            risks.append({
                "type": "no_table_connections",
                "severity": "high",
                "description": "Multiple tables but no homeomorphisms found",
            })

        return risks
