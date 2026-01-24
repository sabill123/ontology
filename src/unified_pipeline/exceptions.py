"""
Unified Pipeline Exception Hierarchy

v6.0: 구조화된 예외 체계
- 에러 원인 명확화
- 복구 전략 제시
- 감사 추적 지원

사용 예:
    try:
        result = detect_homeomorphism(table_a, table_b)
    except InsufficientDataError as e:
        # 전략: 더 많은 데이터 수집 또는 스킵
        logger.warning(f"Insufficient data: {e}")
    except LLMAnalysisError as e:
        # 전략: Fallback to rule-based (만약 필요하다면)
        logger.error(f"LLM failed: {e}")
"""

from typing import Optional, Dict, Any, List
from enum import Enum


class ErrorSeverity(str, Enum):
    """에러 심각도"""
    WARNING = "warning"      # 계속 진행 가능
    ERROR = "error"          # 현재 작업 실패, 다음 작업 가능
    CRITICAL = "critical"    # 파이프라인 중단 필요
    FATAL = "fatal"          # 시스템 재시작 필요


class RecoveryStrategy(str, Enum):
    """복구 전략"""
    SKIP = "skip"                    # 현재 항목 건너뛰기
    RETRY = "retry"                  # 재시도
    USE_DEFAULT = "use_default"      # 기본값 사용
    ESCALATE = "escalate"            # 상위로 에스컬레이션
    ABORT = "abort"                  # 중단


# === Base Exception ===

class OntologyPipelineError(Exception):
    """
    온톨로지 파이프라인 기본 예외

    모든 파이프라인 예외의 부모 클래스
    """

    def __init__(
        self,
        message: str,
        severity: ErrorSeverity = ErrorSeverity.ERROR,
        recovery: RecoveryStrategy = RecoveryStrategy.SKIP,
        context: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None,
    ):
        super().__init__(message)
        self.message = message
        self.severity = severity
        self.recovery = recovery
        self.context = context or {}
        self.cause = cause

    def to_dict(self) -> Dict[str, Any]:
        """감사 로그용 딕셔너리 변환"""
        return {
            "error_type": self.__class__.__name__,
            "message": self.message,
            "severity": self.severity.value,
            "recovery_strategy": self.recovery.value,
            "context": self.context,
            "cause": str(self.cause) if self.cause else None,
        }

    def __str__(self) -> str:
        return f"[{self.severity.value.upper()}] {self.message}"


# === Phase 1: Discovery Exceptions ===

class DiscoveryError(OntologyPipelineError):
    """Discovery 단계 기본 예외"""
    pass


class DataLoadError(DiscoveryError):
    """데이터 로드 실패"""

    def __init__(
        self,
        message: str,
        source_path: Optional[str] = None,
        **kwargs
    ):
        super().__init__(
            message,
            severity=ErrorSeverity.ERROR,
            recovery=RecoveryStrategy.SKIP,
            **kwargs
        )
        self.source_path = source_path
        self.context["source_path"] = source_path


class SchemaAnalysisError(DiscoveryError):
    """스키마 분석 실패"""

    def __init__(
        self,
        message: str,
        table_name: Optional[str] = None,
        **kwargs
    ):
        super().__init__(
            message,
            severity=ErrorSeverity.WARNING,
            recovery=RecoveryStrategy.USE_DEFAULT,
            **kwargs
        )
        self.table_name = table_name
        self.context["table_name"] = table_name


class HomeomorphismDetectionError(DiscoveryError):
    """위상동형 탐지 실패"""
    pass


class InsufficientDataError(HomeomorphismDetectionError):
    """데이터 부족으로 분석 불가"""

    def __init__(
        self,
        message: str,
        required_rows: int = 0,
        actual_rows: int = 0,
        **kwargs
    ):
        super().__init__(
            message,
            severity=ErrorSeverity.WARNING,
            recovery=RecoveryStrategy.SKIP,
            **kwargs
        )
        self.required_rows = required_rows
        self.actual_rows = actual_rows
        self.context.update({
            "required_rows": required_rows,
            "actual_rows": actual_rows,
        })


class StructuralMismatchError(HomeomorphismDetectionError):
    """테이블 간 구조적 불일치"""

    def __init__(
        self,
        message: str,
        table_a: str,
        table_b: str,
        mismatch_type: str = "unknown",
        **kwargs
    ):
        super().__init__(
            message,
            severity=ErrorSeverity.WARNING,
            recovery=RecoveryStrategy.SKIP,
            **kwargs
        )
        self.table_a = table_a
        self.table_b = table_b
        self.mismatch_type = mismatch_type
        self.context.update({
            "table_a": table_a,
            "table_b": table_b,
            "mismatch_type": mismatch_type,
        })


class TDAAnalysisError(DiscoveryError):
    """TDA(위상 데이터 분석) 실패"""

    def __init__(
        self,
        message: str,
        table_name: Optional[str] = None,
        **kwargs
    ):
        super().__init__(
            message,
            severity=ErrorSeverity.WARNING,
            recovery=RecoveryStrategy.USE_DEFAULT,
            **kwargs
        )
        self.table_name = table_name
        self.context["table_name"] = table_name


# === Phase 2: Refinement Exceptions ===

class RefinementError(OntologyPipelineError):
    """Refinement 단계 기본 예외"""
    pass


class ConflictResolutionError(RefinementError):
    """충돌 해결 실패"""

    def __init__(
        self,
        message: str,
        conflicting_concepts: Optional[List[str]] = None,
        conflict_type: str = "unknown",
        **kwargs
    ):
        super().__init__(
            message,
            severity=ErrorSeverity.WARNING,
            recovery=RecoveryStrategy.ESCALATE,
            **kwargs
        )
        self.conflicting_concepts = conflicting_concepts or []
        self.conflict_type = conflict_type
        self.context.update({
            "conflicting_concepts": self.conflicting_concepts,
            "conflict_type": conflict_type,
        })


class QualityAssessmentError(RefinementError):
    """품질 평가 실패"""

    def __init__(
        self,
        message: str,
        concept_id: Optional[str] = None,
        **kwargs
    ):
        super().__init__(
            message,
            severity=ErrorSeverity.WARNING,
            recovery=RecoveryStrategy.USE_DEFAULT,
            **kwargs
        )
        self.concept_id = concept_id
        self.context["concept_id"] = concept_id


class SemanticValidationError(RefinementError):
    """의미론적 검증 실패"""

    def __init__(
        self,
        message: str,
        concept_id: Optional[str] = None,
        validation_type: str = "unknown",
        **kwargs
    ):
        super().__init__(
            message,
            severity=ErrorSeverity.WARNING,
            recovery=RecoveryStrategy.SKIP,
            **kwargs
        )
        self.concept_id = concept_id
        self.validation_type = validation_type
        self.context.update({
            "concept_id": concept_id,
            "validation_type": validation_type,
        })


# === Phase 3: Governance Exceptions ===

class GovernanceError(OntologyPipelineError):
    """Governance 단계 기본 예외"""
    pass


class DecisionError(GovernanceError):
    """거버넌스 결정 실패"""

    def __init__(
        self,
        message: str,
        concept_id: Optional[str] = None,
        decision_stage: str = "unknown",
        **kwargs
    ):
        super().__init__(
            message,
            severity=ErrorSeverity.ERROR,
            recovery=RecoveryStrategy.ESCALATE,
            **kwargs
        )
        self.concept_id = concept_id
        self.decision_stage = decision_stage
        self.context.update({
            "concept_id": concept_id,
            "decision_stage": decision_stage,
        })


class ThresholdCalculationError(GovernanceError):
    """임계값 계산 실패"""

    def __init__(
        self,
        message: str,
        threshold_type: str = "unknown",
        **kwargs
    ):
        super().__init__(
            message,
            severity=ErrorSeverity.WARNING,
            recovery=RecoveryStrategy.USE_DEFAULT,
            **kwargs
        )
        self.threshold_type = threshold_type
        self.context["threshold_type"] = threshold_type


class PolicyGenerationError(GovernanceError):
    """정책 생성 실패"""

    def __init__(
        self,
        message: str,
        policy_type: str = "unknown",
        **kwargs
    ):
        super().__init__(
            message,
            severity=ErrorSeverity.WARNING,
            recovery=RecoveryStrategy.SKIP,
            **kwargs
        )
        self.policy_type = policy_type
        self.context["policy_type"] = policy_type


# === LLM Related Exceptions ===

class LLMError(OntologyPipelineError):
    """LLM 관련 기본 예외"""
    pass


class LLMAnalysisError(LLMError):
    """LLM 분석 실패"""

    def __init__(
        self,
        message: str,
        prompt_summary: Optional[str] = None,
        model_name: Optional[str] = None,
        **kwargs
    ):
        super().__init__(
            message,
            severity=ErrorSeverity.ERROR,
            recovery=RecoveryStrategy.RETRY,  # LLM은 재시도 가능
            **kwargs
        )
        self.prompt_summary = prompt_summary
        self.model_name = model_name
        self.context.update({
            "prompt_summary": prompt_summary,
            "model_name": model_name,
        })


class LLMRateLimitError(LLMError):
    """LLM API Rate Limit 초과"""

    def __init__(
        self,
        message: str,
        retry_after: Optional[int] = None,
        **kwargs
    ):
        super().__init__(
            message,
            severity=ErrorSeverity.WARNING,
            recovery=RecoveryStrategy.RETRY,
            **kwargs
        )
        self.retry_after = retry_after
        self.context["retry_after_seconds"] = retry_after


class LLMResponseParseError(LLMError):
    """LLM 응답 파싱 실패"""

    def __init__(
        self,
        message: str,
        raw_response: Optional[str] = None,
        expected_format: str = "json",
        **kwargs
    ):
        super().__init__(
            message,
            severity=ErrorSeverity.WARNING,
            recovery=RecoveryStrategy.RETRY,
            **kwargs
        )
        self.raw_response = raw_response[:500] if raw_response else None  # 너무 길면 자름
        self.expected_format = expected_format
        self.context.update({
            "raw_response_preview": self.raw_response,
            "expected_format": expected_format,
        })


# === Consensus Exceptions ===

class ConsensusError(OntologyPipelineError):
    """합의 관련 예외"""
    pass


class ConsensusTimeoutError(ConsensusError):
    """합의 타임아웃"""

    def __init__(
        self,
        message: str,
        rounds_completed: int = 0,
        max_rounds: int = 0,
        **kwargs
    ):
        super().__init__(
            message,
            severity=ErrorSeverity.WARNING,
            recovery=RecoveryStrategy.USE_DEFAULT,
            **kwargs
        )
        self.rounds_completed = rounds_completed
        self.max_rounds = max_rounds
        self.context.update({
            "rounds_completed": rounds_completed,
            "max_rounds": max_rounds,
        })


class ConsensusDeadlockError(ConsensusError):
    """합의 교착 상태"""

    def __init__(
        self,
        message: str,
        agent_votes: Optional[Dict[str, str]] = None,
        **kwargs
    ):
        super().__init__(
            message,
            severity=ErrorSeverity.WARNING,
            recovery=RecoveryStrategy.ESCALATE,
            **kwargs
        )
        self.agent_votes = agent_votes
        self.context["agent_votes"] = agent_votes


# === Validation Exceptions ===

class ValidationError(OntologyPipelineError):
    """검증 관련 예외"""
    pass


class PhaseContractViolation(ValidationError):
    """Phase 간 계약 위반"""

    def __init__(
        self,
        message: str,
        source_phase: str,
        target_phase: str,
        missing_fields: Optional[List[str]] = None,
        **kwargs
    ):
        super().__init__(
            message,
            severity=ErrorSeverity.ERROR,
            recovery=RecoveryStrategy.ABORT,
            **kwargs
        )
        self.source_phase = source_phase
        self.target_phase = target_phase
        self.missing_fields = missing_fields or []
        self.context.update({
            "source_phase": source_phase,
            "target_phase": target_phase,
            "missing_fields": self.missing_fields,
        })


class DataIntegrityError(ValidationError):
    """데이터 무결성 오류"""

    def __init__(
        self,
        message: str,
        entity_id: Optional[str] = None,
        integrity_type: str = "unknown",
        **kwargs
    ):
        super().__init__(
            message,
            severity=ErrorSeverity.ERROR,
            recovery=RecoveryStrategy.ABORT,
            **kwargs
        )
        self.entity_id = entity_id
        self.integrity_type = integrity_type
        self.context.update({
            "entity_id": entity_id,
            "integrity_type": integrity_type,
        })


# === Export/Import Exceptions ===

class ExportError(OntologyPipelineError):
    """내보내기 실패"""

    def __init__(
        self,
        message: str,
        export_format: str = "unknown",
        output_path: Optional[str] = None,
        **kwargs
    ):
        super().__init__(
            message,
            severity=ErrorSeverity.ERROR,
            recovery=RecoveryStrategy.RETRY,
            **kwargs
        )
        self.export_format = export_format
        self.output_path = output_path
        self.context.update({
            "export_format": export_format,
            "output_path": output_path,
        })


# === Helper Functions ===

def classify_exception(e: Exception) -> OntologyPipelineError:
    """
    일반 Exception을 적절한 OntologyPipelineError로 변환

    Args:
        e: 원본 예외

    Returns:
        분류된 OntologyPipelineError
    """
    error_message = str(e)
    error_type = type(e).__name__

    # LLM 관련
    if "rate limit" in error_message.lower():
        return LLMRateLimitError(error_message, cause=e)
    if "json" in error_message.lower() and "parse" in error_message.lower():
        return LLMResponseParseError(error_message, cause=e)
    if any(kw in error_message.lower() for kw in ["llm", "openai", "anthropic", "api"]):
        return LLMAnalysisError(error_message, cause=e)

    # 데이터 관련
    if any(kw in error_message.lower() for kw in ["file not found", "no such file"]):
        return DataLoadError(error_message, cause=e)
    if "insufficient" in error_message.lower() or "not enough" in error_message.lower():
        return InsufficientDataError(error_message, cause=e)

    # 기본
    return OntologyPipelineError(
        error_message,
        severity=ErrorSeverity.ERROR,
        recovery=RecoveryStrategy.SKIP,
        cause=e
    )
