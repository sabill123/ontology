"""
Autonomous Agent Base Class (v11.0)

자율 에이전트 베이스 클래스
- 작업 루프 (Work Loop)
- Todo 자동 가져오기
- 진행률 리포팅
- LLM 통신
- 블록체인 기반 Evidence Chain 생성

v6.0 변경사항:
- EnhancedAgentMixin 통합으로 공통 기능 제공
- Audit Trail 지원
- 구조화된 LLM 응답 파싱

v6.1 변경사항:
- JobLogger 통합 - 모든 에이전트 실행/LLM 호출 자동 기록

v11.0 변경사항:
- Evidence Chain 통합 - 모든 에이전트가 블록체인 기반 근거 블록 생성
- record_evidence() 메서드 추가 - 에이전트 분석 결과를 근거 블록으로 저장
- 근거 체인을 통한 Phase 간 유기적 연결
"""

import asyncio
import logging
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Any, Optional, TYPE_CHECKING

from .mixins import EnhancedAgentMixin

# === v16.0: Agent Learning System ===
try:
    from .learning import (
        AgentMemory,
        Experience,
        ExperienceType,
        Outcome,
        create_agent_memory_with_context,
    )
    AGENT_LEARNING_AVAILABLE = True
except ImportError:
    AGENT_LEARNING_AVAILABLE = False
    AgentMemory = None
    Experience = None
    ExperienceType = None
    Outcome = None
    create_agent_memory_with_context = None

# === v16.0: Agent Communication Bus ===
try:
    from .agent_bus import (
        AgentCommunicationBus,
        AgentMessage,
        MessageType,
        MessagePriority,
        get_agent_bus,
        create_info_message,
        create_discovery_message,
        create_validation_request,
    )
    AGENT_BUS_AVAILABLE = True
except ImportError:
    AGENT_BUS_AVAILABLE = False
    AgentCommunicationBus = None
    AgentMessage = None
    MessageType = None
    MessagePriority = None
    get_agent_bus = None
    create_info_message = None
    create_discovery_message = None
    create_validation_request = None

if TYPE_CHECKING:
    from ..todo.manager import TodoManager
    from ..todo.models import PipelineTodo, TodoResult
    from ..shared_context import SharedContext
    from ..enterprise.audit_logger import AuditLogger

logger = logging.getLogger(__name__)


# === v6.1: JobLogger 통합 ===
def _get_job_logger():
    """JobLogger 인스턴스 반환 (지연 로딩)"""
    try:
        from ...common.utils.job_logger import get_job_logger
        return get_job_logger()
    except ImportError:
        return None


def _get_phase_type(phase: str):
    """Phase 문자열을 PhaseType으로 변환"""
    try:
        from ...common.utils.job_logger import PhaseType
        phase_map = {
            "discovery": PhaseType.DISCOVERY,
            "refinement": PhaseType.REFINEMENT,
            "governance": PhaseType.GOVERNANCE,
        }
        return phase_map.get(phase.lower(), PhaseType.DISCOVERY)
    except ImportError:
        return None


class AgentState(str, Enum):
    """에이전트 상태"""
    IDLE = "idle"               # 대기 중
    WORKING = "working"         # 작업 중
    THINKING = "thinking"       # 사고 중 (LLM 호출)
    PAUSED = "paused"           # 일시 정지
    STOPPED = "stopped"         # 정지됨
    ERROR = "error"             # 에러 상태


class AgentTerminateMode(str, Enum):
    """
    v14.0: 에이전트 종료 모드 - 종료 "이유"를 명확히 기록

    Gemini CLI 패턴 참고:
    - GOAL: 정상 완료 + Evidence 기록됨
    - NO_EVIDENCE: 성공했지만 Evidence 미기록 (Silent Failure 가능성)
    - TIMEOUT: 시간 초과
    - MAX_TURNS: 최대 턴 도달
    - ERROR: 에러 발생
    - CONSENSUS_FAILED: 합의 실패
    """
    GOAL = "goal"                       # 정상 완료 + Evidence 기록됨
    NO_EVIDENCE = "no_evidence"         # 성공했지만 Evidence 미기록 (Silent Failure 위험)
    TIMEOUT = "timeout"                 # 시간 초과
    MAX_TURNS = "max_turns"             # 최대 턴 도달
    ERROR = "error"                     # 에러 발생
    CONSENSUS_FAILED = "consensus_failed"  # 합의 실패
    BLOCKED = "blocked"                 # 블로커 발생 (의존성 미해결)
    ESCALATED = "escalated"             # 사람 검토 필요로 에스컬레이션


class CompletionSignal(str, Enum):
    """
    v14.0: 명시적 완료 시그널 - 에이전트가 분석 상태를 명확히 표시

    Autonomous Researcher의 [DONE] 패턴 차용:
    - 에이전트가 "충분히 분석했는지" vs "시간 부족으로 중단했는지" 구분
    - LLM 응답 끝에 시그널 포함 요청
    """
    DONE = "[DONE]"                     # 분석 완료 및 확신
    NEEDS_MORE = "[NEEDS_MORE]"         # 추가 분석 필요
    BLOCKED = "[BLOCKED]"               # 블로커 발생 (데이터 누락 등)
    ESCALATE = "[ESCALATE]"             # 사람 검토 권장


@dataclass
class AgentMetrics:
    """
    에이전트 메트릭 (v15.0 - Agent Stability Index 추가)

    Agent Drift 완화를 위한 안정성 지표 추적:
    - response_lengths: 응답 길이 히스토리 (Behavioral Drift 탐지)
    - confidence_history: 신뢰도 히스토리 (의사결정 품질 추적)
    - decision_reversals: 결정 번복 횟수 (Coordination Drift 지표)
    """
    todos_completed: int = 0
    todos_failed: int = 0
    total_execution_time: float = 0.0
    llm_calls: int = 0
    llm_tokens_used: int = 0
    started_at: Optional[str] = None
    last_activity_at: Optional[str] = None

    # v15.0: Agent Stability Index (ASI) 관련 필드
    response_lengths: List[int] = field(default_factory=list)
    confidence_history: List[float] = field(default_factory=list)
    decision_reversals: int = 0
    _last_decision: Optional[str] = field(default=None, repr=False)

    # ASI 가중치 설정
    _ASI_WEIGHT_LENGTH_VARIANCE: float = field(default=0.3, repr=False)
    _ASI_WEIGHT_CONFIDENCE_STABILITY: float = field(default=0.4, repr=False)
    _ASI_WEIGHT_DECISION_CONSISTENCY: float = field(default=0.3, repr=False)
    _ASI_HISTORY_LIMIT: int = field(default=20, repr=False)

    def record_response(self, response_length: int, confidence: float, decision: Optional[str] = None) -> None:
        """
        v15.0: 응답 메트릭 기록

        Args:
            response_length: LLM 응답 길이 (문자 수)
            confidence: 응답 신뢰도 (0.0-1.0)
            decision: 의사결정 (approve/reject 등, Optional)
        """
        # 히스토리 제한 유지
        if len(self.response_lengths) >= self._ASI_HISTORY_LIMIT:
            self.response_lengths.pop(0)
        if len(self.confidence_history) >= self._ASI_HISTORY_LIMIT:
            self.confidence_history.pop(0)

        self.response_lengths.append(response_length)
        self.confidence_history.append(confidence)

        # 결정 번복 탐지
        if decision and self._last_decision:
            if decision != self._last_decision:
                self.decision_reversals += 1
        self._last_decision = decision

    def calculate_asi(self) -> float:
        """
        v15.0: Agent Stability Index 계산

        ASI = w1 * (1 - length_variance_norm) +
              w2 * confidence_stability +
              w3 * decision_consistency

        높은 ASI = 안정적인 에이전트 (Drift 적음)
        낮은 ASI = 불안정한 에이전트 (Drift 위험)

        Returns:
            ASI 점수 (0.0-1.0, 높을수록 안정적)
        """
        import statistics

        # v24.0: 충분한 히스토리가 없으면 기본값 반환 (최소 샘플 3→8)
        if len(self.response_lengths) < 8 or len(self.confidence_history) < 8:
            return 0.7  # 데이터 부족 시 중립적 점수

        # 1. 응답 길이 분산 (낮을수록 안정적)
        try:
            length_variance = statistics.stdev(self.response_lengths)
            mean_length = statistics.mean(self.response_lengths)
            # 정규화: CV (Coefficient of Variation)
            cv = length_variance / mean_length if mean_length > 0 else 0
            # 0.5 이상 CV는 불안정으로 간주, 0-1 범위로 변환
            length_stability = max(0, 1 - min(cv, 1.0))
        except statistics.StatisticsError:
            length_stability = 0.5

        # 2. 신뢰도 안정성 (분산이 낮고 평균이 높을수록 좋음)
        try:
            conf_mean = statistics.mean(self.confidence_history)
            conf_stdev = statistics.stdev(self.confidence_history)
            # 평균 높고 분산 낮으면 좋음
            confidence_stability = conf_mean * max(0, 1 - conf_stdev * 2)
        except statistics.StatisticsError:
            confidence_stability = 0.5

        # 3. 결정 일관성 (번복이 적을수록 좋음)
        total_decisions = len(self.confidence_history)
        if total_decisions > 0:
            reversal_rate = self.decision_reversals / total_decisions
            decision_consistency = max(0, 1 - reversal_rate * 2)
        else:
            decision_consistency = 1.0

        # 가중 합산
        asi = (
            self._ASI_WEIGHT_LENGTH_VARIANCE * length_stability +
            self._ASI_WEIGHT_CONFIDENCE_STABILITY * confidence_stability +
            self._ASI_WEIGHT_DECISION_CONSISTENCY * decision_consistency
        )

        return round(max(0.0, min(1.0, asi)), 3)

    def get_drift_warning(self) -> Optional[str]:
        """
        v15.0: Drift 경고 메시지 반환

        Returns:
            경고 메시지 (문제 없으면 None)
        """
        asi = self.calculate_asi()

        # v24.0: 임계값 조정 — 자연 탐색 분산 고려
        if asi < 0.25:
            return f"CRITICAL: Agent highly unstable (ASI={asi:.2f}). Consider replacement."
        elif asi < 0.35:
            return f"WARNING: Agent showing drift signs (ASI={asi:.2f}). Monitor closely."
        elif asi < 0.45:
            return f"NOTICE: Agent stability declining (ASI={asi:.2f})."
        return None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "todos_completed": self.todos_completed,
            "todos_failed": self.todos_failed,
            "total_execution_time": self.total_execution_time,
            "llm_calls": self.llm_calls,
            "llm_tokens_used": self.llm_tokens_used,
            "started_at": self.started_at,
            "last_activity_at": self.last_activity_at,
            # v15.0: ASI 관련 필드
            "asi_score": self.calculate_asi(),
            "response_length_history_size": len(self.response_lengths),
            "confidence_history_size": len(self.confidence_history),
            "decision_reversals": self.decision_reversals,
            "drift_warning": self.get_drift_warning(),
        }


class AutonomousAgent(ABC, EnhancedAgentMixin):
    """
    자율 에이전트 베이스 클래스 (v6.0)

    핵심 특징:
    1. 자율적 작업 루프 - Todo 큐에서 작업을 가져와 실행
    2. 진행률 실시간 리포팅 - 이벤트 이미터 통해 FE 전송
    3. 컨텍스트 인식 - SharedContext와 연동
    4. 재시도 및 에러 처리

    v6.0 추가 기능 (EnhancedAgentMixin):
    - parse_llm_response(): 구조화된 LLM 응답 파싱
    - create_justification(): Decision Justification 생성
    - update_context_list/dict(): 타입 안전 컨텍스트 업데이트
    - combine_analysis_results(): 분석 결과 통합

    서브클래스 구현 필요:
    - agent_type: 에이전트 타입 (예: "tda_expert")
    - execute_task(): 실제 작업 실행 로직
    - get_system_prompt(): LLM 시스템 프롬프트
    """

    def __init__(
        self,
        todo_manager: "TodoManager",
        shared_context: "SharedContext",
        llm_client=None,
        agent_id: Optional[str] = None,
        audit_logger: Optional["AuditLogger"] = None,
    ):
        """
        에이전트 초기화

        Args:
            todo_manager: Todo 매니저
            shared_context: 공유 컨텍스트
            llm_client: LLM 클라이언트 (공유)
            agent_id: 에이전트 ID (없으면 자동 생성)
            audit_logger: 감사 로거 (v6.0)
        """
        self.todo_manager = todo_manager
        self.shared_context = shared_context
        self.llm_client = llm_client

        # 식별자
        self.agent_id = agent_id or f"{self.agent_type}_{uuid.uuid4().hex[:8]}"

        # 상태
        self.state = AgentState.IDLE
        self.current_todo: Optional["PipelineTodo"] = None
        self._stop_requested = False
        self._pause_requested = False

        # v14.0: Evidence 기록 추적 (Silent Failure 탐지용)
        self._evidence_recorded_for_current_task = False

        # 메트릭
        self.metrics = AgentMetrics()

        # 이벤트 이미터 (TodoManager에서 가져옴)
        self.event_emitter = todo_manager.event_emitter

        # v6.0: Audit Logger 설정
        if audit_logger:
            self.set_audit_logger(audit_logger)

        # LLM 클라이언트 초기화
        self._init_llm_client()

        # v16.0: Agent Learning System 초기화
        self.agent_memory: Optional[AgentMemory] = None
        self._init_agent_memory()

        # v16.0: Agent Communication Bus 초기화
        self.agent_bus: Optional[AgentCommunicationBus] = None
        self._init_agent_bus()

        # v25.1: Extracted LLM Client & Job Logger (SRP)
        from .agent_llm_client import AgentLLMClient
        from .agent_job_logger import AgentJobLogger
        self._llm_client = AgentLLMClient(self)
        self._job_logger = AgentJobLogger(self.agent_name)

        logger.info(f"Agent initialized: {self.agent_id} (type={self.agent_type})")

    def _init_agent_memory(self):
        """v16.0: Agent Memory 초기화 (Experience Replay)"""
        if AGENT_LEARNING_AVAILABLE and create_agent_memory_with_context:
            try:
                self.agent_memory = create_agent_memory_with_context(self.shared_context)
                logger.debug(f"[v16.0] Agent Memory initialized for {self.agent_id}")
            except Exception as e:
                logger.warning(f"[v16.0] Agent Memory init failed: {e}")
                self.agent_memory = None

    def _init_agent_bus(self):
        """v16.0: Agent Communication Bus 초기화"""
        if AGENT_BUS_AVAILABLE and get_agent_bus:
            try:
                self.agent_bus = get_agent_bus()
                self.agent_bus.register_agent(self.agent_id, self)
                logger.debug(f"[v16.0] Agent Bus initialized for {self.agent_id}")
            except Exception as e:
                logger.warning(f"[v16.0] Agent Bus init failed: {e}")
                self.agent_bus = None

    def record_experience(
        self,
        experience_type: str,
        input_data: Dict[str, Any],
        output_data: Dict[str, Any],
        outcome: str,
        confidence: float = 0.7,
        feedback: Optional[str] = None,
    ) -> bool:
        """
        v16.0: 에이전트 경험 기록 (Experience Replay용)

        Args:
            experience_type: 경험 유형 (fk_detection, entity_mapping 등)
            input_data: 입력 데이터
            output_data: 출력 데이터
            outcome: 결과 (success, failure, partial)
            confidence: 신뢰도 (0.0-1.0)
            feedback: 피드백 (선택)

        Returns:
            성공 여부
        """
        if not self.agent_memory or not AGENT_LEARNING_AVAILABLE:
            return False

        try:
            # ExperienceType 매핑
            exp_type_map = {
                "fk_detection": ExperienceType.FK_DETECTION,
                "entity_mapping": ExperienceType.ENTITY_MAPPING,
                "relationship_inference": ExperienceType.RELATIONSHIP_INFERENCE,
                "conflict_resolution": ExperienceType.CONFLICT_RESOLUTION,
                "quality_assessment": ExperienceType.QUALITY_ASSESSMENT,
                "governance_decision": ExperienceType.GOVERNANCE_DECISION,
            }
            exp_type = exp_type_map.get(experience_type, ExperienceType.FK_DETECTION)

            # Outcome 매핑
            outcome_map = {
                "success": Outcome.SUCCESS,
                "failure": Outcome.FAILURE,
                "partial": Outcome.PARTIAL_SUCCESS,
            }
            exp_outcome = outcome_map.get(outcome.lower(), Outcome.UNKNOWN)

            # store_experience로 저장
            self.agent_memory.store_experience(
                experience_type=exp_type,
                input_context=input_data or {},
                agent_action=experience_type,
                outcome=exp_outcome,
                output_data=output_data,
                confidence=confidence,
            )
            logger.debug(f"[v16.0] Experience recorded: {experience_type} -> {outcome}")
            return True

        except Exception as e:
            logger.warning(f"[v16.0] Failed to record experience: {e}")
            return False

    def get_similar_experiences(
        self,
        input_data: Dict[str, Any],
        limit: int = 5,
    ) -> List[Any]:
        """
        v16.0: 유사 경험 조회 (Few-shot Learning용)

        Args:
            input_data: 현재 입력 데이터
            limit: 최대 반환 수

        Returns:
            유사 경험 리스트
        """
        if not self.agent_memory or not AGENT_LEARNING_AVAILABLE:
            return []

        try:
            return self.agent_memory.recall_similar(input_data, k=limit)
        except Exception as e:
            logger.warning(f"[v16.0] Failed to get similar experiences: {e}")
            return []

    # === v16.0: Inter-Agent Communication Methods ===

    async def send_message_to_agent(
        self,
        recipient_id: str,
        content: Dict[str, Any],
        message_type: str = "info",
        requires_response: bool = False,
    ) -> bool:
        """
        v16.0: 다른 에이전트에 메시지 전송

        Args:
            recipient_id: 수신자 에이전트 ID
            content: 메시지 내용
            message_type: 메시지 유형 (info, discovery, alert, request, collaboration)
            requires_response: 응답 필요 여부

        Returns:
            전송 성공 여부
        """
        if not self.agent_bus or not AGENT_BUS_AVAILABLE:
            return False

        try:
            # MessageType 매핑
            type_map = {
                "info": MessageType.INFO,
                "discovery": MessageType.DISCOVERY,
                "alert": MessageType.ALERT,
                "request": MessageType.REQUEST,
                "collaboration": MessageType.COLLABORATION,
                "validation": MessageType.VALIDATION,
                "feedback": MessageType.FEEDBACK,
            }
            msg_type = type_map.get(message_type, MessageType.INFO)

            message = AgentMessage(
                message_id=f"msg_{self.agent_id}_{uuid.uuid4().hex[:8]}",
                message_type=msg_type,
                sender_id=self.agent_id,
                content=content,
                requires_response=requires_response,
            )

            result = await self.agent_bus.send_to(recipient_id, message)
            logger.debug(f"[v16.0] Message sent: {self.agent_id} -> {recipient_id}")
            return result

        except Exception as e:
            logger.warning(f"[v16.0] Failed to send message: {e}")
            return False

    async def broadcast_discovery(
        self,
        discovery_type: str,
        data: Dict[str, Any],
    ) -> int:
        """
        v16.0: 발견 결과를 모든 에이전트에 브로드캐스트

        Args:
            discovery_type: 발견 유형 (fk_detected, entity_found, relationship_detected 등)
            data: 발견 데이터

        Returns:
            전달된 에이전트 수
        """
        if not self.agent_bus or not AGENT_BUS_AVAILABLE or not create_discovery_message:
            return 0

        try:
            message = create_discovery_message(
                sender_id=self.agent_id,
                discovery_type=discovery_type,
                data=data,
            )
            count = await self.agent_bus.broadcast(message)
            logger.debug(f"[v16.0] Discovery broadcast: {discovery_type} to {count} agents")
            return count

        except Exception as e:
            logger.warning(f"[v16.0] Failed to broadcast discovery: {e}")
            return 0

    async def request_validation(
        self,
        recipient_id: str,
        validation_type: str,
        data: Dict[str, Any],
        timeout: float = 30.0,
    ) -> Optional[Dict[str, Any]]:
        """
        v16.0: 다른 에이전트에 검증 요청

        Args:
            recipient_id: 검증 요청할 에이전트 ID
            validation_type: 검증 유형 (fk_validation, entity_validation 등)
            data: 검증할 데이터
            timeout: 응답 대기 시간 (초)

        Returns:
            검증 결과 또는 None (타임아웃)
        """
        if not self.agent_bus or not AGENT_BUS_AVAILABLE or not create_validation_request:
            return None

        try:
            message = create_validation_request(
                sender_id=self.agent_id,
                validation_type=validation_type,
                data=data,
            )

            response = await self.agent_bus.request(recipient_id, message, timeout)
            if response:
                logger.debug(f"[v16.0] Validation response received from {recipient_id}")
                return response.content
            return None

        except Exception as e:
            logger.warning(f"[v16.0] Failed to request validation: {e}")
            return None

    async def subscribe_to_topic(
        self,
        topic: str,
        callback: callable,
    ) -> Optional[str]:
        """
        v16.0: 토픽 구독 (Pub/Sub)

        Args:
            topic: 구독할 토픽 (예: "fk_detected", "entity_discovered")
            callback: 메시지 수신 시 호출될 콜백 (async)

        Returns:
            구독 ID 또는 None
        """
        if not self.agent_bus or not AGENT_BUS_AVAILABLE:
            return None

        try:
            subscription_id = await self.agent_bus.subscribe(
                topic=topic,
                subscriber_id=self.agent_id,
                callback=callback,
            )
            logger.debug(f"[v16.0] Subscribed to topic: {topic}")
            return subscription_id

        except Exception as e:
            logger.warning(f"[v16.0] Failed to subscribe to topic: {e}")
            return None

    async def publish_to_topic(
        self,
        topic: str,
        content: Dict[str, Any],
    ) -> int:
        """
        v16.0: 토픽에 메시지 발행

        Args:
            topic: 발행할 토픽
            content: 메시지 내용

        Returns:
            전달된 구독자 수
        """
        if not self.agent_bus or not AGENT_BUS_AVAILABLE or not create_info_message:
            return 0

        try:
            message = create_info_message(
                sender_id=self.agent_id,
                content=content,
                topic=topic,
            )
            count = await self.agent_bus.publish(topic, message)
            logger.debug(f"[v16.0] Published to topic {topic}: {count} subscribers")
            return count

        except Exception as e:
            logger.warning(f"[v16.0] Failed to publish to topic: {e}")
            return 0

    async def check_messages(self, timeout: float = 0.1) -> Optional[Dict[str, Any]]:
        """
        v16.0: 대기 중인 메시지 확인

        Args:
            timeout: 대기 시간 (초)

        Returns:
            수신된 메시지 또는 None
        """
        if not self.agent_bus or not AGENT_BUS_AVAILABLE:
            return None

        try:
            message = await self.agent_bus.receive(self.agent_id, timeout)
            if message:
                return message.to_dict()
            return None

        except Exception as e:
            logger.warning(f"[v16.0] Failed to check messages: {e}")
            return None

    # === v22.0: Automatic Learning & Communication ===

    def _record_task_experience(
        self,
        todo: "PipelineTodo",
        result: "TodoResult",
        execution_time: float,
    ) -> None:
        """v22.0: 작업 완료 후 자동 경험 기록 (Agent Learning System)"""
        if not self.agent_memory or not AGENT_LEARNING_AVAILABLE:
            return

        try:
            # agent_type → experience_type 매핑
            exp_type_map = {
                "data_analyst": "fk_detection",
                "tda_expert": "relationship_inference",
                "schema_analyst": "fk_detection",
                "value_matcher": "fk_detection",
                "entity_classifier": "entity_mapping",
                "relationship_detector": "relationship_inference",
                "ontology_architect": "ontology_generation",
                "conflict_resolver": "conflict_resolution",
                "quality_judge": "quality_assessment",
                "semantic_validator": "validation",
                "governance_strategist": "governance_decision",
                "action_prioritizer": "governance_decision",
                "risk_assessor": "governance_decision",
                "policy_generator": "governance_decision",
            }
            exp_type = exp_type_map.get(self.agent_type, "agent_decision")

            # output 요약 (너무 크면 잘라냄)
            output_summary = {}
            if isinstance(result.output, dict):
                output_summary = {
                    k: v for k, v in result.output.items()
                    if not isinstance(v, (list, dict)) or (isinstance(v, list) and len(v) < 10)
                }

            outcome = "success" if result.success else "failure"

            # AgentMemory.store_experience 직접 사용 (base의 record_experience보다 정확)
            self.agent_memory.store_experience(
                experience_type=ExperienceType(exp_type) if exp_type in [e.value for e in ExperienceType] else ExperienceType.AGENT_DECISION,
                input_context={
                    "task_name": todo.name,
                    "task_description": todo.description[:200] if todo.description else "",
                    "agent_type": self.agent_type,
                },
                agent_action=f"{self.agent_type}:{todo.name}",
                outcome=Outcome(outcome) if outcome in [o.value for o in Outcome] else Outcome.UNKNOWN,
                output_data=output_summary,
                confidence=result.output.get("confidence", 0.7) if isinstance(result.output, dict) else 0.7,
            )
            logger.debug(f"[v22.0] Experience recorded: {self.agent_id} / {exp_type} / {outcome}")

        except Exception as e:
            logger.debug(f"[v22.0] Experience recording failed (non-critical): {e}")

    async def _broadcast_task_result(
        self,
        todo: "PipelineTodo",
        result: "TodoResult",
    ) -> None:
        """v22.0: 작업 완료 후 핵심 결과를 다른 에이전트에게 브로드캐스트 (Agent Bus)"""
        if not self.agent_bus or not AGENT_BUS_AVAILABLE:
            return
        if not result.success:
            return  # 실패한 작업은 브로드캐스트하지 않음

        try:
            # output에서 핵심 수치만 추출 (큰 데이터 제외)
            broadcast_data = {}
            if isinstance(result.output, dict):
                broadcast_data = {
                    k: v for k, v in result.output.items()
                    if isinstance(v, (str, int, float, bool))
                }

            # context_updates 키 목록 (무엇이 업데이트되었는지만)
            context_keys = list(result.context_updates.keys()) if result.context_updates else []

            await self.broadcast_discovery(
                discovery_type=f"{self.agent_type}_complete",
                data={
                    "agent_type": self.agent_type,
                    "task_name": todo.name,
                    "output_metrics": broadcast_data,
                    "context_updates_keys": context_keys,
                    "execution_time": result.execution_time_seconds,
                },
            )
        except Exception as e:
            logger.debug(f"[v22.0] Broadcast failed (non-critical): {e}")

    # === v17.0: Unified Services Access Methods ===

    def get_v17_service(self, service_name: str) -> Any:
        """
        v17.0: 통합 서비스 접근

        Args:
            service_name: 서비스 이름
                - calibrator: ConfidenceCalibrator
                - semantic_searcher: SemanticSearcher
                - remediation_engine: RemediationEngine
                - version_manager: VersionManager
                - branch_manager: BranchManager
                - osdk_client: OSDKClient
                - writeback_engine: WritebackEngine
                - decision_explainer: DecisionExplainer
                - whatif_analyzer: WhatIfAnalyzer
                - report_generator: ReportGenerator
                - nl2sql_engine: NL2SQLEngine
                - tool_registry: ToolRegistry
                - streaming_reasoner: StreamingReasoner

        Returns:
            서비스 인스턴스 또는 None
        """
        return self.shared_context.get_v17_service(service_name)

    def calibrate_confidence(
        self,
        raw_confidence: float,
        record_evidence: bool = True,
    ) -> Dict[str, Any]:
        """
        v17.0: 신뢰도 보정 (v5.0 호환 인터페이스)

        v17 Calibrator를 사용하되, v5.0과 동일한 반환 형식 유지.

        Args:
            raw_confidence: 원본 신뢰도 (0-1)
            record_evidence: Evidence Chain에 기록 여부

        Returns:
            Dict with 'original', 'calibrated', 'confidence_interval'
        """
        # Type coercion: handle string/None input
        try:
            raw_confidence = float(raw_confidence) if raw_confidence is not None else 0.5
        except (ValueError, TypeError):
            raw_confidence = 0.5  # Default fallback

        # Clamp to valid range
        raw_confidence = max(0.0, min(1.0, raw_confidence))

        calibrator = self.get_v17_service("calibrator")

        if calibrator is None:
            # v17 서비스 없으면 기본 반환
            return {
                "original": raw_confidence,
                "calibrated": raw_confidence,
                "temperature": 1.0,
                "confidence_interval": (
                    max(0, raw_confidence - 0.1),
                    min(1, raw_confidence + 0.1),
                ),
            }

        try:
            result = calibrator.calibrate(
                agent=self.agent_type,
                raw_confidence=raw_confidence,
                record_evidence=record_evidence,
            )

            # v5.0 호환 형식으로 변환
            return {
                "original": result.raw_confidence,
                "calibrated": result.final_confidence,
                "temperature": result.temperature_used,
                "confidence_interval": (
                    max(0, result.final_confidence - 0.1),
                    min(1, result.final_confidence + 0.1),
                ),
                # v17.0 추가 정보
                "v17_result": result.to_dict() if hasattr(result, "to_dict") else None,
            }
        except Exception as e:
            logger.warning(f"[v17.0] Calibration failed: {e}, using raw confidence")
            return {
                "original": raw_confidence,
                "calibrated": raw_confidence,
                "temperature": 1.0,
                "confidence_interval": (
                    max(0, raw_confidence - 0.1),
                    min(1, raw_confidence + 0.1),
                ),
            }

    async def semantic_search(
        self,
        query: str,
        limit: int = 10,
        tables: Optional[List[str]] = None,
        expand_query: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        v17.0: 시맨틱 검색

        Args:
            query: 검색 쿼리
            limit: 최대 결과 수
            tables: 검색할 테이블 (None이면 전체)
            expand_query: LLM으로 쿼리 확장 여부

        Returns:
            검색 결과 목록
        """
        searcher = self.get_v17_service("semantic_searcher")

        if searcher is None:
            logger.debug(f"[v17.0] SemanticSearcher not available")
            return []

        try:
            results = await searcher.search(
                query=query,
                limit=limit,
                tables=tables,
                expand_query=expand_query,
            )
            return [r.to_dict() for r in results]
        except Exception as e:
            logger.warning(f"[v17.0] Semantic search failed: {e}")
            return []

    async def auto_remediate(
        self,
        table: str,
        issue_types: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        v17.0: 데이터 품질 자동 수정

        Args:
            table: 테이블 이름
            issue_types: 처리할 이슈 유형

        Returns:
            수정 결과 요약
        """
        engine = self.get_v17_service("remediation_engine")

        if engine is None:
            logger.debug(f"[v17.0] RemediationEngine not available")
            return {"actions_taken": 0, "issues_fixed": 0}

        try:
            result = await engine.remediate(
                table=table,
                auto_approve=True,
                issue_types=issue_types,
            )
            return result.to_dict() if hasattr(result, "to_dict") else {}
        except Exception as e:
            logger.warning(f"[v17.0] Auto-remediation failed: {e}")
            return {"actions_taken": 0, "issues_fixed": 0, "error": str(e)}

    def commit_version(self, message: str) -> Optional[str]:
        """
        v17.0: 버전 커밋

        Args:
            message: 커밋 메시지

        Returns:
            버전 ID 또는 None
        """
        version_manager = self.get_v17_service("version_manager")

        if version_manager is None:
            return None

        try:
            return version_manager.commit(
                message=message,
                author=self.agent_type,
            )
        except Exception as e:
            logger.warning(f"[v17.0] Version commit failed: {e}")
            return None

    def _init_llm_client(self):
        """LLM 클라이언트 초기화"""
        if self.llm_client is None:
            try:
                from ...common.utils.llm import get_openai_client
                self.llm_client = get_openai_client()
            except Exception as e:
                logger.warning(f"LLM client init failed: {e}")

    # === 추상 메서드 (서브클래스 구현 필요) ===

    @property
    @abstractmethod
    def agent_type(self) -> str:
        """에이전트 타입 (예: "tda_expert", "schema_analyst")"""
        pass

    @property
    @abstractmethod
    def agent_name(self) -> str:
        """에이전트 이름 (표시용)"""
        pass

    @property
    @abstractmethod
    def phase(self) -> str:
        """소속 Phase ("discovery", "refinement", "governance")"""
        pass

    @abstractmethod
    def get_system_prompt(self) -> str:
        """LLM 시스템 프롬프트 (에이전트별 특화)"""
        pass

    def get_full_system_prompt(self) -> str:
        """
        프로젝트 컨텍스트가 포함된 전체 시스템 프롬프트 (v9.1)

        프로젝트 목적, 지향점, Phase별 역할 정보를 포함합니다.
        팔란티어 스타일의 출력 형식 가이드도 포함됩니다.

        Returns:
            프로젝트 컨텍스트 + 에이전트별 프롬프트
        """
        try:
            from .system_prompts import get_enhanced_system_prompt
            return get_enhanced_system_prompt(self.agent_type, self.get_system_prompt())
        except ImportError:
            # system_prompts 모듈이 없으면 기존 프롬프트만 반환
            return self.get_system_prompt()

    @abstractmethod
    async def execute_task(
        self,
        todo: "PipelineTodo",
        context: "SharedContext",
    ) -> "TodoResult":
        """
        작업 실행

        Args:
            todo: 실행할 Todo
            context: 공유 컨텍스트

        Returns:
            실행 결과
        """
        pass

    # === 작업 루프 ===

    async def run(self) -> None:
        """
        에이전트 작업 루프 실행

        - Todo 큐에서 작업을 가져와 실행
        - 완료되면 다음 작업 대기
        - stop() 호출 시 종료
        """
        self.state = AgentState.IDLE
        self.metrics.started_at = datetime.now().isoformat()
        self._stop_requested = False

        logger.info(f"Agent {self.agent_id} started work loop")

        while not self._stop_requested:
            # 일시 정지 체크
            if self._pause_requested:
                self.state = AgentState.PAUSED
                await asyncio.sleep(0.5)
                continue

            # Todo 가져오기
            todo = self.todo_manager.claim_next_todo(
                agent_id=self.agent_id,
                agent_type=self.agent_type,
            )

            if todo is None:
                # 작업 없음 - 대기
                self.state = AgentState.IDLE
                await asyncio.sleep(0.1)  # 100ms 대기
                continue

            # 작업 실행
            await self._execute_todo(todo)

        self.state = AgentState.STOPPED
        logger.info(f"Agent {self.agent_id} stopped")

    async def run_single(self, todo_id: str) -> Optional["TodoResult"]:
        """
        단일 Todo 실행 (테스트/디버깅용)

        Args:
            todo_id: 실행할 Todo ID

        Returns:
            실행 결과
        """
        # 이미 할당된 Todo 가져오기 (orchestrator에서 이미 assign됨)
        todo = self.todo_manager.get_todo(todo_id)
        if todo is None:
            logger.warning(f"Todo not found: {todo_id}")
            return None

        return await self._execute_todo(todo)

    async def _execute_todo(self, todo: "PipelineTodo") -> "TodoResult":
        """Todo 실행"""
        from ..todo.models import TodoResult

        self.current_todo = todo
        self.state = AgentState.WORKING
        self.metrics.last_activity_at = datetime.now().isoformat()

        # v14.0: Evidence 추적 플래그 리셋
        self._evidence_recorded_for_current_task = False

        start_time = time.time()

        # v6.1: JobLogger 에이전트 실행 시작 기록
        job_logger = _get_job_logger()
        phase_type = _get_phase_type(self.phase)
        if job_logger and job_logger.is_active() and phase_type:
            job_logger.start_agent_execution(
                agent_name=self.agent_name,
                agent_type=self.agent_type,
                phase=phase_type,
                todo_id=todo.todo_id,
                input_data={
                    "todo_name": todo.name,
                    "todo_type": getattr(todo, 'task_type', getattr(todo, 'todo_type', 'unknown')),
                    "context_keys": list(self.shared_context.to_dict().keys()) if hasattr(self.shared_context, 'to_dict') else [],
                },
            )

        # v6.1: LLM 호출 컨텍스트 설정
        try:
            from ...common.utils.llm import set_llm_agent_context
            set_llm_agent_context(self.agent_name)
        except ImportError:
            pass

        try:
            # 진행률 초기화
            self._report_progress(0.0, f"Starting: {todo.name}")

            # 실제 작업 실행
            self.state = AgentState.THINKING
            result = await self.execute_task(todo, self.shared_context)

            # 실행 시간 기록
            execution_time = time.time() - start_time
            result.execution_time_seconds = execution_time
            result.agent_id = self.agent_id

            # v22.0: Agent Learning — 경험 기록 (성공/실패 모두)
            self._record_task_experience(todo, result, execution_time)

            # v22.0: Agent Communication Bus — 핵심 결과 브로드캐스트
            await self._broadcast_task_result(todo, result)

            # 성공 처리
            if result.success:
                # v14.0: Evidence 기록 여부 검증 및 terminate_mode 설정
                if not self._evidence_recorded_for_current_task:
                    # Auto-record evidence from task result to prevent silent failure
                    try:
                        output_summary = str(result.output)[:300] if result.output else ""
                        self.record_evidence(
                            finding=f"Completed: {todo.name}",
                            reasoning=f"Agent {self.agent_name} completed task with output",
                            conclusion=output_summary or "Task completed successfully",
                            confidence=0.7,
                            topics=[self.phase, self.agent_type, todo.name.lower().replace(" ", "_")],
                        )
                    except Exception:
                        pass  # best-effort auto-record

                if self._evidence_recorded_for_current_task:
                    result.terminate_mode = AgentTerminateMode.GOAL.value
                else:
                    logger.warning(
                        f"[SILENT_FAILURE_RISK] Agent {self.agent_id} completed '{todo.name}' "
                        f"without recording evidence"
                    )
                    result.terminate_mode = AgentTerminateMode.NO_EVIDENCE.value

                # 컨텍스트 업데이트
                self._apply_context_updates(result.context_updates)

                # Todo 완료
                self.todo_manager.complete_todo(todo.todo_id, result)
                self.metrics.todos_completed += 1

                self._report_progress(1.0, f"Completed: {todo.name}")

                # v6.1: JobLogger 에이전트 실행 완료 기록
                if job_logger and job_logger.is_active():
                    job_logger.end_agent_execution(
                        agent_name=self.agent_name,
                        output_data=result.data if hasattr(result, 'data') else {},
                        success=True,
                    )
            else:
                # 실패 처리
                self.todo_manager.fail_todo(todo.todo_id, result.error or "Unknown error")
                self.metrics.todos_failed += 1

                # v6.1: JobLogger 에이전트 실행 실패 기록
                if job_logger and job_logger.is_active():
                    job_logger.end_agent_execution(
                        agent_name=self.agent_name,
                        success=False,
                        error=result.error,
                    )

            self.metrics.total_execution_time += execution_time

            return result

        except Exception as e:
            # 예외 처리
            import traceback
            traceback.print_exc()
            logger.error(f"Agent {self.agent_id} error: {e}", exc_info=True)

            execution_time = time.time() - start_time
            error_result = TodoResult(
                success=False,
                error=str(e),
                execution_time_seconds=execution_time,
                agent_id=self.agent_id,
                terminate_mode=AgentTerminateMode.ERROR.value,  # v14.0
            )

            self.todo_manager.fail_todo(todo.todo_id, str(e))
            self.metrics.todos_failed += 1
            self.state = AgentState.ERROR

            # v6.1: JobLogger 에이전트 실행 에러 기록
            if job_logger and job_logger.is_active():
                job_logger.end_agent_execution(
                    agent_name=self.agent_name,
                    success=False,
                    error=str(e),
                )

            return error_result

        finally:
            self.current_todo = None
            self.state = AgentState.IDLE

            # v6.1: LLM 호출 컨텍스트 초기화
            try:
                from ...common.utils.llm import clear_llm_agent_context
                clear_llm_agent_context()
            except ImportError:
                pass

    # === 컨트롤 ===

    def stop(self) -> None:
        """에이전트 정지 요청"""
        self._stop_requested = True
        logger.info(f"Agent {self.agent_id} stop requested")

    def pause(self) -> None:
        """에이전트 일시 정지"""
        self._pause_requested = True
        logger.info(f"Agent {self.agent_id} paused")

    def resume(self) -> None:
        """에이전트 재개"""
        self._pause_requested = False
        logger.info(f"Agent {self.agent_id} resumed")

    # === LLM 통신 (AgentLLMClient에 위임) ===

    async def call_llm(self, user_prompt: str, temperature: float = 0.3, max_tokens: int = 2000,
                       purpose: str = None, thinking_before: str = None, model_override: str = None, seed: int = 42) -> str:
        """LLM 호출 → AgentLLMClient에 위임"""
        return await self._llm_client.call_llm(user_prompt, temperature, max_tokens, purpose, thinking_before, model_override, seed)

    async def call_llm_with_context(self, instruction: str, context_data: Dict[str, Any],
                                     temperature: float = 0.3, max_tokens: int = 2000, purpose: str = None,
                                     thinking_before: str = None, request_completion_signal: bool = True,
                                     model_override: str = None) -> str:
        """컨텍스트와 함께 LLM 호출 → AgentLLMClient에 위임"""
        return await self._llm_client.call_llm_with_context(instruction, context_data, temperature, max_tokens,
                                                             purpose, thinking_before, request_completion_signal, model_override)

    async def call_llm_structured(self, response_model, user_prompt: str, temperature: float = 0.2,
                                   max_retries: int = 2, purpose: str = None):
        """구조화된 LLM 호출 → AgentLLMClient에 위임"""
        return await self._llm_client.call_llm_structured(response_model, user_prompt, temperature, max_retries, purpose)

    def _format_context_for_llm(self, context_data: Dict[str, Any]) -> str:
        """LLM용 컨텍스트 포맷팅 → AgentLLMClient에 위임"""
        return self._llm_client._format_context(context_data)

    # === 사고 과정 로깅 (AgentJobLogger에 위임) ===

    def log_thinking(self, thinking_type: str, content: str, confidence: float = None, **kwargs):
        self._job_logger.log_thinking(thinking_type, content, confidence, **kwargs)

    def log_analysis(self, target: str, findings: List[str], confidence: float = None, **kwargs):
        self._job_logger.log_analysis(target, findings, confidence, **kwargs)

    def log_reasoning(self, premise: str, logic: str, conclusion: str, confidence: float = None, **kwargs):
        self._job_logger.log_reasoning(premise, logic, conclusion, confidence, **kwargs)

    def log_decision(self, decision: str, options: List[str], rationale: str, confidence: float = None, **kwargs):
        self._job_logger.log_decision(decision, options, rationale, confidence, **kwargs)

    def log_insight(self, insight: str, source: str, importance: str = "medium", confidence: float = None):
        self._job_logger.log_insight(insight, source, importance, confidence)

    def log_hypothesis(self, hypothesis: str, basis: List[str], confidence: float = None):
        self._job_logger.log_hypothesis(hypothesis, basis, confidence)

    def start_reasoning_chain(self, goal: str) -> Optional[str]:
        return self._job_logger.start_reasoning_chain(goal)

    def add_reasoning_step(self, chain_id: str, thinking_type: str, content: str, confidence: float = None):
        self._job_logger.add_reasoning_step(chain_id, thinking_type, content, confidence)

    def end_reasoning_chain(self, chain_id: str, conclusion: str, confidence: float = None, success: bool = True):
        self._job_logger.end_reasoning_chain(chain_id, conclusion, confidence, success)

    def set_analysis_summary(self, summary: str):
        self._job_logger.set_analysis_summary(summary)

    # === v14.0: 완료 시그널 파싱 ===

    def _parse_completion_signal(self, response: str) -> CompletionSignal:
        """
        v14.0: LLM 응답에서 완료 시그널 파싱

        Autonomous Researcher 패턴:
        - LLM 응답 끝에 [DONE], [NEEDS_MORE], [BLOCKED], [ESCALATE] 시그널 탐지
        - 시그널이 없으면 기본적으로 NEEDS_MORE 반환 (보수적 접근)

        Args:
            response: LLM 응답 문자열

        Returns:
            CompletionSignal enum 값
        """
        if not response:
            return CompletionSignal.NEEDS_MORE

        # 응답 끝 부분에서 시그널 탐지 (마지막 200자 검사)
        response_tail = response[-200:] if len(response) > 200 else response

        for signal in CompletionSignal:
            if signal.value in response_tail:
                logger.debug(f"[v14.0] Completion signal detected: {signal.value}")
                return signal

        # 전체 응답에서 한번 더 검사 (백업)
        for signal in CompletionSignal:
            if signal.value in response:
                logger.debug(f"[v14.0] Completion signal found in full response: {signal.value}")
                return signal

        # 기본값: 추가 분석 권장 (보수적 접근)
        logger.debug("[v14.0] No completion signal found, defaulting to NEEDS_MORE")
        return CompletionSignal.NEEDS_MORE

    def _get_completion_signal_prompt_suffix(self) -> str:
        """
        v14.0: LLM 프롬프트에 추가할 완료 시그널 요청 문구

        Returns:
            시그널 요청 프롬프트 suffix
        """
        return """

---
## Completion Signal (Required)
At the END of your response, you MUST include ONE of these completion signals:

- [DONE] - Analysis is complete and I am confident in the results
- [NEEDS_MORE] - Additional analysis or data would improve results
- [BLOCKED] - Cannot proceed due to missing data or blockers
- [ESCALATE] - Human review recommended for this decision

Choose the most appropriate signal based on your analysis confidence."""

    # === 진행률 리포팅 ===

    def _report_progress(self, progress: float, message: str = "") -> None:
        """진행률 리포팅"""
        if self.current_todo:
            self.current_todo.update_progress(progress, message)
            self.event_emitter.todo_progress(
                todo_id=self.current_todo.todo_id,
                progress=progress,
                message=message,
                agent_id=self.agent_id,
                phase=self.phase,
            )

    # === 컨텍스트 업데이트 ===

    def _apply_context_updates(self, updates: Dict[str, Any]) -> None:
        """
        v13.2: 동적 컨텍스트 업데이트 적용

        LLM이 분석한 결과를 동적으로 저장합니다.
        - 기존 필드: 기존 로직대로 업데이트 (list extend, dict update)
        - 새 필드: dynamic_data 딕셔너리에 저장 (스키마 변경 없이)
        """
        if not updates:
            return

        for key, value in updates.items():
            if hasattr(self.shared_context, key):
                # 기존 필드: 기존 로직 유지
                current = getattr(self.shared_context, key)
                if isinstance(current, list) and isinstance(value, list):
                    current.extend(value)
                elif isinstance(current, dict) and isinstance(value, dict):
                    current.update(value)
                else:
                    setattr(self.shared_context, key, value)
                logger.debug(f"Context updated (existing field): {key}")
            else:
                # v13.2: 새 필드는 dynamic_data에 저장
                if not hasattr(self.shared_context, 'dynamic_data'):
                    self.shared_context.dynamic_data = {}
                self.shared_context.dynamic_data[key] = value
                logger.debug(f"Context updated (dynamic field): {key}")

    # === 유틸리티 ===

    def get_status(self) -> Dict[str, Any]:
        """에이전트 상태 조회"""
        return {
            "agent_id": self.agent_id,
            "agent_type": self.agent_type,
            "agent_name": self.agent_name,
            "phase": self.phase,
            "state": self.state.value,
            "current_todo": self.current_todo.to_dict() if self.current_todo else None,
            "metrics": self.metrics.to_dict(),
        }

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} id={self.agent_id} state={self.state.value}>"

    # === v11.0: Evidence Chain Integration ===

    def record_evidence(
        self,
        finding: str,
        reasoning: str,
        conclusion: str,
        evidence_type: str = None,
        data_references: List[str] = None,
        metrics: Dict[str, Any] = None,
        supporting_block_ids: List[str] = None,
        contradicting_block_ids: List[str] = None,
        confidence: float = 0.0,
        topics: List[str] = None,
    ) -> Optional[str]:
        """
        근거 블록 생성 및 체인에 추가 (v11.0)

        블록체인 구조를 차용하여 에이전트의 분석 결과를
        불변의 근거 블록으로 저장합니다. 각 블록은 이전 블록의
        해시를 참조하여 근거 체인을 형성합니다.

        Args:
            finding: 발견 사항 (무엇을 발견했는가)
            reasoning: 추론 근거 (왜 그렇게 판단했는가)
            conclusion: 결론 (그래서 어떻다는 것인가)
            evidence_type: 근거 유형 (자동 추론 가능)
            data_references: 데이터 참조 (예: ["freight_rates.TPT_Day_Count"])
            metrics: 측정 지표 (예: {"confidence": 0.85, "p_value": 0.02})
            supporting_block_ids: 이 결론을 지지하는 이전 블록들
            contradicting_block_ids: 이 결론과 충돌하는 블록들
            confidence: 신뢰도 (0.0-1.0)
            topics: 검색용 토픽 태그

        Returns:
            생성된 블록 ID (실패 시 None)

        Example:
            >>> block_id = self.record_evidence(
            ...     finding="freight_rates 테이블의 TPT_Day_Count 분포가 bimodal",
            ...     reasoning="Betti-1 분석 결과 0.57로 두 개의 클러스터 존재",
            ...     conclusion="빠른배송(1-3일) vs 일반배송(7-15일) 그룹으로 분류 가능",
            ...     data_references=["freight_rates.TPT_Day_Count"],
            ...     metrics={"betti_1": 0.57, "cluster_count": 2},
            ...     confidence=0.85,
            ...     topics=["transport", "clustering"]
            ... )
        """
        try:
            from .evidence_chain import EvidenceRegistry, EvidenceType

            registry = EvidenceRegistry()

            # 에이전트 타입에서 Evidence 타입 추론
            if evidence_type is None:
                evidence_type = self._infer_evidence_type()

            # Phase 문자열 변환
            phase_map = {
                "discovery": "phase1_discovery",
                "refinement": "phase2_refinement",
                "governance": "phase3_governance",
            }
            phase = phase_map.get(self.phase, f"phase_{self.phase}")

            # 근거 블록 생성
            block = registry.record_evidence(
                phase=phase,
                agent=self.agent_name,
                evidence_type=EvidenceType(evidence_type) if isinstance(evidence_type, str) else evidence_type,
                finding=finding,
                reasoning=reasoning,
                conclusion=conclusion,
                data_references=data_references or [],
                metrics=metrics or {},
                supporting_block_ids=supporting_block_ids or [],
                contradicting_block_ids=contradicting_block_ids or [],
                confidence=confidence,
                topics=topics or [],
            )

            # v14.0: Evidence 기록 플래그 설정 (Silent Failure 탐지용)
            self._evidence_recorded_for_current_task = True

            logger.debug(f"Evidence recorded: {block.block_id} ({self.agent_name})")

            # === v22.0: Agent Bus로 Evidence 브로드캐스트 ===
            # 다른 에이전트들이 새로운 근거를 실시간으로 인지할 수 있도록 함
            try:
                if hasattr(self, '_agent_bus') and self._agent_bus is not None:
                    import asyncio
                    message = {
                        "block_id": block.block_id,
                        "agent_id": self.agent_id,
                        "agent_name": self.agent_name,
                        "phase": phase,
                        "evidence_type": str(evidence_type),
                        "finding_summary": finding[:100] if finding else "",
                        "confidence": confidence,
                        "topics": topics or [],
                    }
                    # Non-blocking fire-and-forget: schedule if event loop running
                    try:
                        loop = asyncio.get_running_loop()
                        loop.create_task(self._agent_bus.publish("evidence.recorded", message))
                    except RuntimeError:
                        # No event loop - use sync queue if available
                        if hasattr(self._agent_bus, 'queue_message'):
                            self._agent_bus.queue_message("evidence.recorded", message)
            except Exception as bus_err:
                logger.debug(f"[v22.0] Evidence broadcast skipped: {bus_err}")

            return block.block_id

        except Exception as e:
            logger.warning(f"Failed to record evidence: {e}")
            return None

    def _infer_evidence_type(self) -> str:
        """에이전트 타입에서 Evidence 타입 추론"""
        type_map = {
            # Phase 1: Discovery
            "tda_expert": "tda_analysis",
            "schema_analyst": "schema_analysis",
            "value_matcher": "value_matching",
            "entity_classifier": "entity_classification",
            "relationship_detector": "relationship_detection",
            # Phase 2: Refinement
            "ontology_architect": "ontology_design",
            "conflict_resolver": "conflict_resolution",
            "quality_judge": "quality_assessment",
            "semantic_validator": "semantic_validation",
            # Phase 3: Governance
            "governance_strategist": "council_vote",
            "action_prioritizer": "council_vote",
            "risk_assessor": "council_vote",
            "policy_generator": "council_vote",
        }
        return type_map.get(self.agent_type, "tda_analysis")

    def get_evidence_for_reasoning(
        self,
        topics: List[str] = None,
        phases: List[str] = None,
        min_confidence: float = 0.0,
    ) -> List[Dict[str, Any]]:
        """
        추론을 위한 근거 조회 (v11.0)

        Council 토론이나 후속 분석을 위해 이전 Phase의
        근거 블록들을 조회합니다.

        Args:
            topics: 검색할 토픽
            phases: 검색할 Phase (예: ["phase1_discovery"])
            min_confidence: 최소 신뢰도

        Returns:
            근거 블록 목록 (딕셔너리 형태)

        Example:
            >>> evidence = self.get_evidence_for_reasoning(
            ...     topics=["transport", "freight"],
            ...     phases=["phase1_discovery"],
            ...     min_confidence=0.5
            ... )
            >>> for e in evidence:
            ...     print(f"[{e['block_id']}] {e['finding']}")
        """
        try:
            from .evidence_chain import EvidenceRegistry

            registry = EvidenceRegistry()
            results = []

            # 토픽으로 검색
            if topics:
                for topic in topics:
                    blocks = registry.find_evidence_by_topic(topic)
                    results.extend(blocks)

            # Phase로 검색
            if phases:
                for phase in phases:
                    blocks = registry.get_phase_evidence(phase)
                    results.extend(blocks)

            # 토픽도 Phase도 없으면 전체 조회
            if not topics and not phases:
                results = registry.get_all_evidence()

            # 중복 제거 및 신뢰도 필터링
            seen = set()
            filtered = []
            for block in results:
                if block.block_id not in seen and block.confidence >= min_confidence:
                    seen.add(block.block_id)
                    filtered.append(block.to_dict())

            return filtered

        except Exception as e:
            logger.warning(f"Failed to get evidence: {e}")
            return []

    def format_evidence_for_debate(
        self,
        proposal: str,
        relevant_phases: List[str] = None
    ) -> str:
        """
        토론용 근거 문서 생성 (v11.0)

        Council 토론을 위해 관련 근거들을 문서 형태로 정리합니다.

        Args:
            proposal: 토론 안건
            relevant_phases: 관련 Phase

        Returns:
            토론용 근거 문서

        Example:
            >>> doc = self.format_evidence_for_debate(
            ...     proposal="Warehouse-FreightRate 관계 승인",
            ...     relevant_phases=["phase1_discovery", "phase2_refinement"]
            ... )
        """
        try:
            from .evidence_chain import EvidenceRegistry

            registry = EvidenceRegistry()
            return registry.format_for_council_debate(
                proposal=proposal,
                relevant_phases=relevant_phases
            )

        except Exception as e:
            logger.warning(f"Failed to format evidence: {e}")
            return f"[Evidence document unavailable: {e}]"
