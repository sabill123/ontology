"""
Enterprise Audit Logger

엔터프라이즈 감사 로깅:
- 모든 의사결정 기록
- 사용자 액션 추적
- 컴플라이언스 보고서 생성 (SOX, GDPR, HIPAA)
- 데이터 리니지 추적
- 불변 감사 로그

References:
- SOX Section 404: Internal Controls
- GDPR Article 30: Records of Processing Activities
- HIPAA Security Rule: Audit Controls
"""

import json
import hashlib
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
import uuid

logger = logging.getLogger(__name__)


class AuditEventType(str, Enum):
    """감사 이벤트 타입"""
    # 시스템 이벤트
    SYSTEM_START = "system_start"
    SYSTEM_STOP = "system_stop"
    SYSTEM_ERROR = "system_error"
    CONFIG_CHANGE = "config_change"

    # 데이터 이벤트
    DATA_ACCESSED = "data_accessed"
    DATA_MODIFIED = "data_modified"
    DATA_DELETED = "data_deleted"
    DATA_EXPORTED = "data_exported"

    # 인사이트 이벤트
    INSIGHT_GENERATED = "insight_generated"
    INSIGHT_VIEWED = "insight_viewed"
    INSIGHT_DISMISSED = "insight_dismissed"

    # 의사결정 이벤트
    DECISION_MADE = "decision_made"
    DECISION_APPROVED = "decision_approved"
    DECISION_REJECTED = "decision_rejected"

    # 액션 이벤트
    ACTION_CREATED = "action_created"
    ACTION_EXECUTED = "action_executed"
    ACTION_FAILED = "action_failed"
    ACTION_CANCELLED = "action_cancelled"

    # 거버넌스 이벤트
    POLICY_APPLIED = "policy_applied"
    POLICY_VIOLATED = "policy_violated"
    COMPLIANCE_CHECK = "compliance_check"

    # 사용자 이벤트
    USER_LOGIN = "user_login"
    USER_LOGOUT = "user_logout"
    USER_ACTION = "user_action"
    PERMISSION_CHANGE = "permission_change"

    # v6.0: Agent Decision Justification 이벤트
    AGENT_DECISION = "agent_decision"
    AGENT_CONSENSUS = "agent_consensus"
    AGENT_REASONING = "agent_reasoning"
    THRESHOLD_APPLIED = "threshold_applied"
    CONTRACT_VALIDATED = "contract_validated"
    EVIDENCE_EVALUATED = "evidence_evaluated"


class ComplianceFramework(str, Enum):
    """컴플라이언스 프레임워크"""
    SOX = "sox"          # Sarbanes-Oxley (재무 보고)
    GDPR = "gdpr"        # General Data Protection Regulation
    HIPAA = "hipaa"      # Health Insurance Portability and Accountability Act
    PCI_DSS = "pci_dss"  # Payment Card Industry Data Security Standard
    ISO27001 = "iso27001"  # Information Security Management
    SOC2 = "soc2"        # Service Organization Control 2


@dataclass
class AuditEvent:
    """감사 이벤트"""
    event_id: str
    event_type: AuditEventType
    timestamp: datetime
    actor: str  # user ID or "system"
    action: str
    resource: str

    # 상세 정보
    details: Dict[str, Any] = field(default_factory=dict)
    outcome: str = "success"  # success, failure, pending

    # 컨텍스트
    ip_address: Optional[str] = None
    session_id: Optional[str] = None
    user_agent: Optional[str] = None
    correlation_id: Optional[str] = None

    # 컴플라이언스 태그
    compliance_tags: List[str] = field(default_factory=list)

    # 데이터 민감도
    sensitivity_level: str = "normal"  # normal, sensitive, highly_sensitive

    # 불변성 보장용 해시
    previous_hash: Optional[str] = None
    event_hash: Optional[str] = None

    def __post_init__(self):
        """해시 계산"""
        if self.event_hash is None:
            self.event_hash = self._calculate_hash()

    def _calculate_hash(self) -> str:
        """이벤트 해시 계산 (불변성 보장)"""
        data = {
            "event_id": self.event_id,
            "event_type": self.event_type.value if isinstance(self.event_type, AuditEventType) else self.event_type,
            "timestamp": self.timestamp.isoformat(),
            "actor": self.actor,
            "action": self.action,
            "resource": self.resource,
            "outcome": self.outcome,
            "previous_hash": self.previous_hash,
        }
        content = json.dumps(data, sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "event_type": self.event_type.value if isinstance(self.event_type, AuditEventType) else self.event_type,
            "timestamp": self.timestamp.isoformat(),
            "actor": self.actor,
            "action": self.action,
            "resource": self.resource,
            "details": self.details,
            "outcome": self.outcome,
            "ip_address": self.ip_address,
            "session_id": self.session_id,
            "correlation_id": self.correlation_id,
            "compliance_tags": self.compliance_tags,
            "sensitivity_level": self.sensitivity_level,
            "event_hash": self.event_hash,
            "previous_hash": self.previous_hash,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AuditEvent":
        """딕셔너리에서 AuditEvent 생성"""
        event_type = data.get("event_type", "user_action")
        if isinstance(event_type, str):
            try:
                event_type = AuditEventType(event_type)
            except ValueError:
                event_type = AuditEventType.USER_ACTION

        return cls(
            event_id=data.get("event_id", str(uuid.uuid4())),
            event_type=event_type,
            timestamp=datetime.fromisoformat(data["timestamp"]) if isinstance(data.get("timestamp"), str) else datetime.now(),
            actor=data.get("actor", "unknown"),
            action=data.get("action", ""),
            resource=data.get("resource", ""),
            details=data.get("details", {}),
            outcome=data.get("outcome", "success"),
            ip_address=data.get("ip_address"),
            session_id=data.get("session_id"),
            correlation_id=data.get("correlation_id"),
            compliance_tags=data.get("compliance_tags", []),
            sensitivity_level=data.get("sensitivity_level", "normal"),
            previous_hash=data.get("previous_hash"),
            event_hash=data.get("event_hash"),
        )


# === v6.0: Decision Justification Models ===

@dataclass
class EvidenceItem:
    """증거 아이템"""
    evidence_id: str
    evidence_type: str  # data_pattern, statistical, semantic, rule_based, llm_analysis
    source: str
    description: str
    weight: float = 1.0  # 증거 가중치 (0.0 - 1.0)
    confidence: float = 0.0
    raw_data: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "evidence_id": self.evidence_id,
            "evidence_type": self.evidence_type,
            "source": self.source,
            "description": self.description,
            "weight": self.weight,
            "confidence": self.confidence,
            "raw_data": self.raw_data,
        }


@dataclass
class DecisionJustification:
    """
    결정 근거 (v6.0)

    모든 에이전트 결정에 대한 완전한 추적 가능성 제공
    """
    justification_id: str
    decision_id: str
    agent_id: str
    agent_type: str

    # 결정 내용
    decision_type: str  # approve, reject, escalate, provisional
    decision_target: str  # 결정 대상 (concept_id, entity_id 등)
    final_confidence: float

    # 증거 체인
    evidence_chain: List[EvidenceItem] = field(default_factory=list)

    # 추론 과정
    reasoning_steps: List[str] = field(default_factory=list)

    # 사용된 임계값
    thresholds_applied: Dict[str, float] = field(default_factory=dict)

    # 대안 분석
    alternatives_considered: List[Dict[str, Any]] = field(default_factory=list)

    # 반대 의견
    dissenting_factors: List[str] = field(default_factory=list)

    # 메타데이터
    created_at: datetime = field(default_factory=datetime.now)
    llm_model_used: Optional[str] = None
    prompt_hash: Optional[str] = None  # LLM 프롬프트 해시 (재현성)

    def add_evidence(self, evidence: EvidenceItem) -> "DecisionJustification":
        """증거 추가 (빌더 패턴)"""
        self.evidence_chain.append(evidence)
        return self

    def add_reasoning_step(self, step: str) -> "DecisionJustification":
        """추론 단계 추가"""
        self.reasoning_steps.append(step)
        return self

    def to_dict(self) -> Dict[str, Any]:
        return {
            "justification_id": self.justification_id,
            "decision_id": self.decision_id,
            "agent_id": self.agent_id,
            "agent_type": self.agent_type,
            "decision_type": self.decision_type,
            "decision_target": self.decision_target,
            "final_confidence": self.final_confidence,
            "evidence_chain": [e.to_dict() for e in self.evidence_chain],
            "reasoning_steps": self.reasoning_steps,
            "thresholds_applied": self.thresholds_applied,
            "alternatives_considered": self.alternatives_considered,
            "dissenting_factors": self.dissenting_factors,
            "created_at": self.created_at.isoformat(),
            "llm_model_used": self.llm_model_used,
            "prompt_hash": self.prompt_hash,
        }

    def to_summary(self) -> str:
        """사람이 읽기 쉬운 요약 생성"""
        lines = [
            f"Decision: {self.decision_type} ({self.final_confidence:.2%} confidence)",
            f"Target: {self.decision_target}",
            f"Agent: {self.agent_type} ({self.agent_id})",
            "",
            "Reasoning:",
        ]
        for i, step in enumerate(self.reasoning_steps, 1):
            lines.append(f"  {i}. {step}")

        lines.append("")
        lines.append(f"Evidence Count: {len(self.evidence_chain)}")

        if self.dissenting_factors:
            lines.append("")
            lines.append("Dissenting Factors:")
            for factor in self.dissenting_factors:
                lines.append(f"  - {factor}")

        return "\n".join(lines)


@dataclass
class ConsensusJustification:
    """
    합의 근거 (v6.0)

    멀티에이전트 합의 과정에 대한 완전한 추적
    """
    consensus_id: str
    topic_id: str
    topic_name: str

    # 참여 에이전트
    participating_agents: List[str] = field(default_factory=list)

    # 각 에이전트별 결정 근거
    agent_justifications: Dict[str, DecisionJustification] = field(default_factory=dict)

    # 합의 결과
    final_result: str = "undecided"
    combined_confidence: float = 0.0
    agreement_level: float = 0.0

    # 토론 라운드
    rounds_completed: int = 0
    round_summaries: List[Dict[str, Any]] = field(default_factory=list)

    # 합의 과정
    convergence_path: List[Dict[str, Any]] = field(default_factory=list)

    # 메타데이터
    created_at: datetime = field(default_factory=datetime.now)
    total_duration_seconds: float = 0.0

    def add_agent_justification(
        self,
        agent_id: str,
        justification: DecisionJustification
    ) -> "ConsensusJustification":
        """에이전트 근거 추가"""
        self.agent_justifications[agent_id] = justification
        if agent_id not in self.participating_agents:
            self.participating_agents.append(agent_id)
        return self

    def to_dict(self) -> Dict[str, Any]:
        return {
            "consensus_id": self.consensus_id,
            "topic_id": self.topic_id,
            "topic_name": self.topic_name,
            "participating_agents": self.participating_agents,
            "agent_justifications": {
                k: v.to_dict() for k, v in self.agent_justifications.items()
            },
            "final_result": self.final_result,
            "combined_confidence": self.combined_confidence,
            "agreement_level": self.agreement_level,
            "rounds_completed": self.rounds_completed,
            "round_summaries": self.round_summaries,
            "convergence_path": self.convergence_path,
            "created_at": self.created_at.isoformat(),
            "total_duration_seconds": self.total_duration_seconds,
        }


class AuditLogger:
    """
    엔터프라이즈 감사 로거

    불변 감사 로그 관리:
    - 체인 해시로 변조 방지
    - 컴플라이언스 보고서 생성
    - 데이터 리니지 추적
    """

    def __init__(
        self,
        storage_path: str = "./audit_logs",
        retention_days: int = 365 * 7,  # 7년 보관 (SOX 요구사항)
        enable_file_logging: bool = True,
    ):
        """
        Args:
            storage_path: 감사 로그 저장 경로
            retention_days: 로그 보관 기간 (일)
            enable_file_logging: 파일 로깅 활성화
        """
        self.storage_path = Path(storage_path)
        self.retention_days = retention_days
        self.enable_file_logging = enable_file_logging

        # 메모리 버퍼
        self.events: List[AuditEvent] = []
        self.event_counter = 0

        # 해시 체인
        self._last_hash: Optional[str] = None

        # 인덱스 (빠른 조회용)
        self._by_actor: Dict[str, List[str]] = {}
        self._by_type: Dict[str, List[str]] = {}
        self._by_resource: Dict[str, List[str]] = {}
        self._by_correlation: Dict[str, List[str]] = {}

        # 저장 경로 생성
        if enable_file_logging:
            self.storage_path.mkdir(parents=True, exist_ok=True)

    def log(
        self,
        event_type: AuditEventType,
        actor: str,
        action: str,
        resource: str,
        details: Optional[Dict[str, Any]] = None,
        outcome: str = "success",
        correlation_id: Optional[str] = None,
        compliance_tags: Optional[List[str]] = None,
        sensitivity_level: str = "normal",
        ip_address: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> AuditEvent:
        """
        감사 이벤트 기록

        Args:
            event_type: 이벤트 타입
            actor: 행위자 (사용자 ID 또는 "system")
            action: 수행된 액션
            resource: 대상 리소스
            details: 상세 정보
            outcome: 결과 (success/failure/pending)
            correlation_id: 연관 ID (트랜잭션 추적용)
            compliance_tags: 컴플라이언스 태그
            sensitivity_level: 민감도 수준

        Returns:
            생성된 AuditEvent
        """
        self.event_counter += 1
        event_id = f"AUDIT-{datetime.now().strftime('%Y%m%d')}-{self.event_counter:08d}"

        event = AuditEvent(
            event_id=event_id,
            event_type=event_type,
            timestamp=datetime.now(),
            actor=actor,
            action=action,
            resource=resource,
            details=details or {},
            outcome=outcome,
            correlation_id=correlation_id or str(uuid.uuid4()),
            compliance_tags=compliance_tags or [],
            sensitivity_level=sensitivity_level,
            ip_address=ip_address,
            session_id=session_id,
            previous_hash=self._last_hash,
        )

        # 해시 체인 업데이트
        self._last_hash = event.event_hash

        # 저장
        self.events.append(event)
        self._index_event(event)

        # 파일 로깅
        if self.enable_file_logging:
            self._write_to_file(event)

        logger.info(f"Audit: [{event_type.value}] {actor} -> {action} on {resource}")

        return event

    def _index_event(self, event: AuditEvent):
        """이벤트 인덱싱"""
        # 액터 인덱스
        if event.actor not in self._by_actor:
            self._by_actor[event.actor] = []
        self._by_actor[event.actor].append(event.event_id)

        # 타입 인덱스
        type_key = event.event_type.value if isinstance(event.event_type, AuditEventType) else event.event_type
        if type_key not in self._by_type:
            self._by_type[type_key] = []
        self._by_type[type_key].append(event.event_id)

        # 리소스 인덱스
        if event.resource not in self._by_resource:
            self._by_resource[event.resource] = []
        self._by_resource[event.resource].append(event.event_id)

        # 연관 ID 인덱스
        if event.correlation_id:
            if event.correlation_id not in self._by_correlation:
                self._by_correlation[event.correlation_id] = []
            self._by_correlation[event.correlation_id].append(event.event_id)

    def _write_to_file(self, event: AuditEvent):
        """파일에 이벤트 기록"""
        date_str = event.timestamp.strftime("%Y-%m-%d")
        file_path = self.storage_path / f"audit_{date_str}.jsonl"

        try:
            with open(file_path, "a") as f:
                f.write(json.dumps(event.to_dict()) + "\n")
        except Exception as e:
            logger.error(f"Failed to write audit log: {e}")

    # === 편의 메서드 ===

    def log_insight(
        self,
        insight: Dict[str, Any],
        actor: str = "system",
    ) -> AuditEvent:
        """인사이트 생성 기록"""
        return self.log(
            AuditEventType.INSIGHT_GENERATED,
            actor=actor,
            action="generate_insight",
            resource=insight.get("insight_id", "unknown"),
            details={
                "title": insight.get("title"),
                "severity": insight.get("severity"),
                "source_table": insight.get("source_table"),
                "insight_type": insight.get("insight_type"),
            },
            compliance_tags=["analytics", "decision_support"],
        )

    def log_decision(
        self,
        decision: Dict[str, Any],
        actor: str = "system",
    ) -> AuditEvent:
        """의사결정 기록"""
        return self.log(
            AuditEventType.DECISION_MADE,
            actor=actor,
            action="make_decision",
            resource=decision.get("decision_id", "unknown"),
            details={
                "decision_type": decision.get("decision_type"),
                "confidence": decision.get("confidence"),
                "rationale": decision.get("rationale"),
                "alternatives_considered": decision.get("alternatives", []),
            },
            compliance_tags=["governance", "decision_audit"],
        )

    def log_action(
        self,
        action_data: Dict[str, Any],
        actor: str = "system",
        outcome: str = "success",
    ) -> AuditEvent:
        """액션 실행 기록"""
        return self.log(
            AuditEventType.ACTION_EXECUTED,
            actor=actor,
            action=action_data.get("action_type", "execute"),
            resource=action_data.get("action_id", "unknown"),
            details={
                "title": action_data.get("title"),
                "workflow_type": action_data.get("workflow_type"),
                "params": action_data.get("params", {}),
            },
            outcome=outcome,
            compliance_tags=["workflow", "automation"],
        )

    def log_data_access(
        self,
        actor: str,
        resource: str,
        access_type: str = "read",
        details: Optional[Dict[str, Any]] = None,
    ) -> AuditEvent:
        """데이터 접근 기록"""
        return self.log(
            AuditEventType.DATA_ACCESSED,
            actor=actor,
            action=access_type,
            resource=resource,
            details=details or {},
            compliance_tags=["data_access", "privacy"],
            sensitivity_level="sensitive" if "pii" in resource.lower() else "normal",
        )

    def log_policy_violation(
        self,
        policy_id: str,
        actor: str,
        resource: str,
        violation_details: Dict[str, Any],
    ) -> AuditEvent:
        """정책 위반 기록"""
        return self.log(
            AuditEventType.POLICY_VIOLATED,
            actor=actor,
            action="policy_violation",
            resource=resource,
            details={
                "policy_id": policy_id,
                **violation_details,
            },
            outcome="failure",
            compliance_tags=["policy", "violation", "security"],
            sensitivity_level="highly_sensitive",
        )

    # === v6.0: Decision Justification 메서드 ===

    def log_agent_decision(
        self,
        justification: DecisionJustification,
        correlation_id: Optional[str] = None,
    ) -> AuditEvent:
        """
        에이전트 결정 및 근거 기록

        Args:
            justification: 결정 근거
            correlation_id: 연관 ID

        Returns:
            AuditEvent
        """
        return self.log(
            AuditEventType.AGENT_DECISION,
            actor=justification.agent_id,
            action=f"decide_{justification.decision_type}",
            resource=justification.decision_target,
            details={
                "justification": justification.to_dict(),
                "summary": justification.to_summary(),
            },
            correlation_id=correlation_id,
            compliance_tags=["agent_decision", "justification", "audit_trail"],
        )

    def log_consensus(
        self,
        consensus: ConsensusJustification,
        correlation_id: Optional[str] = None,
    ) -> AuditEvent:
        """
        멀티에이전트 합의 기록

        Args:
            consensus: 합의 근거
            correlation_id: 연관 ID

        Returns:
            AuditEvent
        """
        return self.log(
            AuditEventType.AGENT_CONSENSUS,
            actor="multi_agent_system",
            action=f"consensus_{consensus.final_result}",
            resource=consensus.topic_id,
            details={
                "consensus": consensus.to_dict(),
                "agents": consensus.participating_agents,
                "rounds": consensus.rounds_completed,
                "confidence": consensus.combined_confidence,
                "agreement": consensus.agreement_level,
            },
            correlation_id=correlation_id,
            compliance_tags=["consensus", "multi_agent", "governance"],
        )

    def log_evidence_evaluation(
        self,
        agent_id: str,
        evidence_items: List[EvidenceItem],
        evaluation_result: str,
        target_resource: str,
        correlation_id: Optional[str] = None,
    ) -> AuditEvent:
        """
        증거 평가 기록

        Args:
            agent_id: 에이전트 ID
            evidence_items: 평가된 증거 목록
            evaluation_result: 평가 결과
            target_resource: 대상 리소스
            correlation_id: 연관 ID

        Returns:
            AuditEvent
        """
        total_weight = sum(e.weight for e in evidence_items)
        weighted_confidence = (
            sum(e.weight * e.confidence for e in evidence_items) / total_weight
            if total_weight > 0 else 0
        )

        return self.log(
            AuditEventType.EVIDENCE_EVALUATED,
            actor=agent_id,
            action="evaluate_evidence",
            resource=target_resource,
            details={
                "evidence_count": len(evidence_items),
                "evidence_types": list(set(e.evidence_type for e in evidence_items)),
                "total_weight": total_weight,
                "weighted_confidence": weighted_confidence,
                "evaluation_result": evaluation_result,
                "evidence_summary": [
                    {"id": e.evidence_id, "type": e.evidence_type, "confidence": e.confidence}
                    for e in evidence_items
                ],
            },
            correlation_id=correlation_id,
            compliance_tags=["evidence", "evaluation", "analysis"],
        )

    def log_threshold_application(
        self,
        agent_id: str,
        thresholds: Dict[str, float],
        actual_values: Dict[str, float],
        decision_outcome: str,
        target_resource: str,
        correlation_id: Optional[str] = None,
    ) -> AuditEvent:
        """
        임계값 적용 기록

        Args:
            agent_id: 에이전트 ID
            thresholds: 적용된 임계값
            actual_values: 실제 값
            decision_outcome: 결정 결과
            target_resource: 대상 리소스
            correlation_id: 연관 ID

        Returns:
            AuditEvent
        """
        # 임계값 대비 실제 값 비교
        comparisons = {}
        for key, threshold in thresholds.items():
            if key in actual_values:
                comparisons[key] = {
                    "threshold": threshold,
                    "actual": actual_values[key],
                    "passed": actual_values[key] >= threshold,
                }

        return self.log(
            AuditEventType.THRESHOLD_APPLIED,
            actor=agent_id,
            action="apply_threshold",
            resource=target_resource,
            details={
                "thresholds": thresholds,
                "actual_values": actual_values,
                "comparisons": comparisons,
                "decision_outcome": decision_outcome,
            },
            correlation_id=correlation_id,
            compliance_tags=["threshold", "decision_rule", "governance"],
        )

    def log_contract_validation(
        self,
        source_phase: str,
        target_phase: str,
        is_valid: bool,
        violations: List[str],
        warnings: List[str],
        correlation_id: Optional[str] = None,
    ) -> AuditEvent:
        """
        Phase 계약 검증 기록

        Args:
            source_phase: 소스 Phase
            target_phase: 대상 Phase
            is_valid: 검증 통과 여부
            violations: 위반 사항
            warnings: 경고 사항
            correlation_id: 연관 ID

        Returns:
            AuditEvent
        """
        return self.log(
            AuditEventType.CONTRACT_VALIDATED,
            actor="contract_validator",
            action="validate_phase_contract",
            resource=f"{source_phase}_to_{target_phase}",
            details={
                "source_phase": source_phase,
                "target_phase": target_phase,
                "is_valid": is_valid,
                "violations": violations,
                "warnings": warnings,
            },
            outcome="success" if is_valid else "failure",
            correlation_id=correlation_id,
            compliance_tags=["contract", "validation", "pipeline"],
        )

    def create_decision_justification(
        self,
        decision_id: str,
        agent_id: str,
        agent_type: str,
        decision_type: str,
        decision_target: str,
        confidence: float,
    ) -> DecisionJustification:
        """
        DecisionJustification 팩토리 메서드

        Args:
            decision_id: 결정 ID
            agent_id: 에이전트 ID
            agent_type: 에이전트 타입
            decision_type: 결정 타입
            decision_target: 결정 대상
            confidence: 신뢰도

        Returns:
            DecisionJustification 인스턴스
        """
        return DecisionJustification(
            justification_id=f"JUST-{datetime.now().strftime('%Y%m%d%H%M%S')}-{uuid.uuid4().hex[:8]}",
            decision_id=decision_id,
            agent_id=agent_id,
            agent_type=agent_type,
            decision_type=decision_type,
            decision_target=decision_target,
            final_confidence=confidence,
        )

    def create_evidence_item(
        self,
        evidence_type: str,
        source: str,
        description: str,
        weight: float = 1.0,
        confidence: float = 0.0,
        raw_data: Optional[Dict[str, Any]] = None,
    ) -> EvidenceItem:
        """
        EvidenceItem 팩토리 메서드

        Args:
            evidence_type: 증거 타입
            source: 소스
            description: 설명
            weight: 가중치
            confidence: 신뢰도
            raw_data: 원본 데이터

        Returns:
            EvidenceItem 인스턴스
        """
        return EvidenceItem(
            evidence_id=f"EV-{uuid.uuid4().hex[:12]}",
            evidence_type=evidence_type,
            source=source,
            description=description,
            weight=weight,
            confidence=confidence,
            raw_data=raw_data,
        )

    # === 조회 메서드 ===

    def get_events(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        event_types: Optional[List[AuditEventType]] = None,
        actor: Optional[str] = None,
        resource: Optional[str] = None,
        outcome: Optional[str] = None,
        limit: int = 1000,
    ) -> List[AuditEvent]:
        """이벤트 조회"""
        filtered = self.events

        if start_date:
            filtered = [e for e in filtered if e.timestamp >= start_date]

        if end_date:
            filtered = [e for e in filtered if e.timestamp <= end_date]

        if event_types:
            type_values = [t.value for t in event_types]
            filtered = [
                e for e in filtered
                if (e.event_type.value if isinstance(e.event_type, AuditEventType) else e.event_type) in type_values
            ]

        if actor:
            filtered = [e for e in filtered if e.actor == actor]

        if resource:
            filtered = [e for e in filtered if resource in e.resource]

        if outcome:
            filtered = [e for e in filtered if e.outcome == outcome]

        return filtered[:limit]

    def get_by_correlation(self, correlation_id: str) -> List[AuditEvent]:
        """연관 ID로 이벤트 조회"""
        event_ids = self._by_correlation.get(correlation_id, [])
        return [e for e in self.events if e.event_id in event_ids]

    def verify_chain_integrity(self) -> Dict[str, Any]:
        """해시 체인 무결성 검증"""
        if not self.events:
            return {"valid": True, "checked_events": 0}

        valid = True
        invalid_events = []
        prev_hash = None

        for event in self.events:
            # 이전 해시 확인
            if event.previous_hash != prev_hash:
                valid = False
                invalid_events.append({
                    "event_id": event.event_id,
                    "issue": "previous_hash_mismatch",
                })

            # 현재 해시 검증
            expected_hash = event._calculate_hash()
            if event.event_hash != expected_hash:
                valid = False
                invalid_events.append({
                    "event_id": event.event_id,
                    "issue": "hash_tampered",
                })

            prev_hash = event.event_hash

        return {
            "valid": valid,
            "checked_events": len(self.events),
            "invalid_events": invalid_events,
        }

    # === 컴플라이언스 보고서 ===

    def generate_compliance_report(
        self,
        start_date: datetime,
        end_date: datetime,
        framework: ComplianceFramework = ComplianceFramework.SOX,
    ) -> Dict[str, Any]:
        """
        컴플라이언스 보고서 생성

        Args:
            start_date: 시작일
            end_date: 종료일
            framework: 컴플라이언스 프레임워크

        Returns:
            컴플라이언스 보고서
        """
        filtered_events = [
            e for e in self.events
            if start_date <= e.timestamp <= end_date
        ]

        # 프레임워크별 보고서 생성
        if framework == ComplianceFramework.SOX:
            return self._generate_sox_report(filtered_events, start_date, end_date)
        elif framework == ComplianceFramework.GDPR:
            return self._generate_gdpr_report(filtered_events, start_date, end_date)
        elif framework == ComplianceFramework.HIPAA:
            return self._generate_hipaa_report(filtered_events, start_date, end_date)
        else:
            return self._generate_generic_report(filtered_events, start_date, end_date, framework)

    def _generate_sox_report(
        self,
        events: List[AuditEvent],
        start_date: datetime,
        end_date: datetime,
    ) -> Dict[str, Any]:
        """SOX 컴플라이언스 보고서"""
        # SOX 관련 이벤트 필터링
        financial_events = [
            e for e in events
            if any(tag in ["financial", "decision_audit", "governance"]
                   for tag in e.compliance_tags)
        ]

        # 통제 활동 분석
        control_activities = {
            "data_modifications": len([e for e in events if e.event_type == AuditEventType.DATA_MODIFIED]),
            "decisions_made": len([e for e in events if e.event_type == AuditEventType.DECISION_MADE]),
            "policy_violations": len([e for e in events if e.event_type == AuditEventType.POLICY_VIOLATED]),
            "failed_actions": len([e for e in events if e.outcome == "failure"]),
        }

        # 권한 변경 추적
        permission_changes = [
            e for e in events
            if e.event_type == AuditEventType.PERMISSION_CHANGE
        ]

        return {
            "report_type": "SOX Section 404 Compliance Report",
            "framework": "SOX",
            "period": {
                "start": start_date.isoformat(),
                "end": end_date.isoformat(),
            },
            "executive_summary": {
                "total_events": len(events),
                "financial_events": len(financial_events),
                "control_effectiveness": self._calculate_control_effectiveness(events),
            },
            "control_activities": control_activities,
            "permission_changes": len(permission_changes),
            "anomalies": self._detect_anomalies(events),
            "chain_integrity": self.verify_chain_integrity(),
            "generated_at": datetime.now().isoformat(),
            "retention_compliant": self.retention_days >= 365 * 7,
        }

    def _generate_gdpr_report(
        self,
        events: List[AuditEvent],
        start_date: datetime,
        end_date: datetime,
    ) -> Dict[str, Any]:
        """GDPR 컴플라이언스 보고서"""
        # 데이터 접근 이벤트
        data_access_events = [
            e for e in events
            if e.event_type == AuditEventType.DATA_ACCESSED
        ]

        # PII 관련 이벤트
        pii_events = [
            e for e in events
            if e.sensitivity_level in ["sensitive", "highly_sensitive"]
        ]

        # 데이터 주체 권리 요청 (삭제 등)
        deletion_events = [
            e for e in events
            if e.event_type == AuditEventType.DATA_DELETED
        ]

        return {
            "report_type": "GDPR Article 30 Processing Activities Record",
            "framework": "GDPR",
            "period": {
                "start": start_date.isoformat(),
                "end": end_date.isoformat(),
            },
            "processing_activities": {
                "total_data_access": len(data_access_events),
                "pii_access_events": len(pii_events),
                "deletion_requests": len(deletion_events),
            },
            "data_subjects_affected": len(set(e.actor for e in data_access_events)),
            "lawful_basis_verification": "consent",  # 실제로는 동적으로 결정
            "cross_border_transfers": 0,  # 실제 구현 필요
            "generated_at": datetime.now().isoformat(),
        }

    def _generate_hipaa_report(
        self,
        events: List[AuditEvent],
        start_date: datetime,
        end_date: datetime,
    ) -> Dict[str, Any]:
        """HIPAA 컴플라이언스 보고서"""
        # PHI 접근 이벤트
        phi_events = [
            e for e in events
            if "phi" in e.resource.lower() or "health" in e.resource.lower()
        ]

        return {
            "report_type": "HIPAA Security Rule Audit Report",
            "framework": "HIPAA",
            "period": {
                "start": start_date.isoformat(),
                "end": end_date.isoformat(),
            },
            "phi_access_summary": {
                "total_accesses": len(phi_events),
                "unique_users": len(set(e.actor for e in phi_events)),
            },
            "security_incidents": len([e for e in events if e.event_type == AuditEventType.POLICY_VIOLATED]),
            "audit_controls": {
                "logging_enabled": True,
                "integrity_verification": self.verify_chain_integrity()["valid"],
                "retention_days": self.retention_days,
            },
            "generated_at": datetime.now().isoformat(),
        }

    def _generate_generic_report(
        self,
        events: List[AuditEvent],
        start_date: datetime,
        end_date: datetime,
        framework: ComplianceFramework,
    ) -> Dict[str, Any]:
        """일반 컴플라이언스 보고서"""
        return {
            "report_type": f"{framework.value.upper()} Compliance Report",
            "framework": framework.value,
            "period": {
                "start": start_date.isoformat(),
                "end": end_date.isoformat(),
            },
            "summary": {
                "total_events": len(events),
                "events_by_type": self._count_by_type(events),
                "events_by_actor": self._count_by_actor(events),
                "events_by_outcome": self._count_by_outcome(events),
            },
            "failed_events": [e.to_dict() for e in events if e.outcome != "success"][:100],
            "generated_at": datetime.now().isoformat(),
        }

    def _calculate_control_effectiveness(self, events: List[AuditEvent]) -> float:
        """통제 효과성 계산"""
        if not events:
            return 100.0

        failures = len([e for e in events if e.outcome == "failure"])
        violations = len([e for e in events if e.event_type == AuditEventType.POLICY_VIOLATED])

        total_issues = failures + violations
        effectiveness = max(0, 100 - (total_issues / len(events) * 100))

        return round(effectiveness, 1)

    def _detect_anomalies(self, events: List[AuditEvent]) -> List[Dict[str, Any]]:
        """이상 활동 탐지"""
        anomalies = []

        # 빈도 분석
        actor_counts = {}
        for e in events:
            actor_counts[e.actor] = actor_counts.get(e.actor, 0) + 1

        # 이상 빈도 탐지
        if actor_counts:
            avg_count = sum(actor_counts.values()) / len(actor_counts)
            for actor, count in actor_counts.items():
                if count > avg_count * 3:  # 평균의 3배 초과
                    anomalies.append({
                        "type": "high_frequency_actor",
                        "actor": actor,
                        "count": count,
                        "average": round(avg_count, 1),
                    })

        # 실패 패턴 탐지
        failures = [e for e in events if e.outcome == "failure"]
        if len(failures) / len(events) > 0.1 if events else False:  # 10% 이상 실패
            anomalies.append({
                "type": "high_failure_rate",
                "failure_count": len(failures),
                "total_events": len(events),
                "rate": round(len(failures) / len(events) * 100, 1),
            })

        return anomalies

    def _count_by_type(self, events: List[AuditEvent]) -> Dict[str, int]:
        """타입별 카운트"""
        counts = {}
        for e in events:
            type_key = e.event_type.value if isinstance(e.event_type, AuditEventType) else e.event_type
            counts[type_key] = counts.get(type_key, 0) + 1
        return counts

    def _count_by_actor(self, events: List[AuditEvent]) -> Dict[str, int]:
        """액터별 카운트"""
        counts = {}
        for e in events:
            counts[e.actor] = counts.get(e.actor, 0) + 1
        return counts

    def _count_by_outcome(self, events: List[AuditEvent]) -> Dict[str, int]:
        """결과별 카운트"""
        counts = {}
        for e in events:
            counts[e.outcome] = counts.get(e.outcome, 0) + 1
        return counts

    def get_stats(self) -> Dict[str, Any]:
        """통계 조회"""
        return {
            "total_events": len(self.events),
            "unique_actors": len(self._by_actor),
            "event_types": len(self._by_type),
            "resources_tracked": len(self._by_resource),
            "chain_valid": self.verify_chain_integrity()["valid"],
        }


# Factory function
def create_audit_logger(
    storage_path: str = "./audit_logs",
    retention_days: int = 365 * 7,
) -> AuditLogger:
    """팩토리 함수"""
    return AuditLogger(
        storage_path=storage_path,
        retention_days=retention_days,
    )
