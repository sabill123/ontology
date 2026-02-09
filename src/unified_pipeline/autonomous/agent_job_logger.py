"""
Agent Job Logger

에이전트의 JobLogger 관련 책임을 분리 (SRP)

책임:
- 사고 과정 로깅 (thinking, analysis, reasoning, decision)
- 인사이트/가설 기록
- 추론 체인 관리
- 분석 요약 설정

8개 중복 패턴을 _safe_log로 통합하여 Shotgun Surgery 제거
"""

import logging
from typing import List, Optional

logger = logging.getLogger(__name__)


def _get_job_logger():
    """JobLogger 인스턴스 반환 (지연 로딩)"""
    try:
        from ...common.utils.job_logger import get_job_logger
        return get_job_logger()
    except ImportError:
        return None


class AgentJobLogger:
    """
    에이전트 JobLogger 위임 클래스

    AutonomousAgent의 8개 중복 로깅 메서드를 통합합니다.
    """

    def __init__(self, agent_name: str):
        self._agent_name = agent_name

    def _safe_log(self, method_name: str, **kwargs) -> Optional[any]:
        """JobLogger 메서드를 안전하게 호출"""
        jl = _get_job_logger()
        if not jl or not jl.is_active():
            return None
        try:
            return getattr(jl, method_name)(agent_name=self._agent_name, **kwargs)
        except Exception:
            return None

    def log_thinking(self, thinking_type: str, content: str, confidence: float = None, **kwargs):
        """사고 과정 기록"""
        jl = _get_job_logger()
        if not jl or not jl.is_active():
            return
        try:
            from ...common.utils.job_logger import ThinkingType
            tt = ThinkingType(thinking_type) if thinking_type in [t.value for t in ThinkingType] else ThinkingType.REASONING
            jl.log_thinking(
                agent_name=self._agent_name,
                thinking_type=tt,
                content=content,
                confidence=confidence,
                context=kwargs.get('context'),
                evidence=kwargs.get('evidence'),
                related_data=kwargs.get('related_data'),
            )
        except Exception:
            pass

    def log_analysis(self, target: str, findings: List[str], confidence: float = None, **kwargs):
        """분석 과정 기록"""
        self._safe_log(
            "log_analysis",
            analysis_target=target,
            findings=findings,
            methodology=kwargs.get('methodology'),
            data_examined=kwargs.get('data_examined'),
            confidence=confidence,
        )

    def log_reasoning(self, premise: str, logic: str, conclusion: str, confidence: float = None, **kwargs):
        """추론 과정 기록"""
        self._safe_log(
            "log_reasoning",
            premise=premise,
            logic=logic,
            conclusion=conclusion,
            confidence=confidence,
            supporting_evidence=kwargs.get('evidence'),
        )

    def log_decision(self, decision: str, options: List[str], rationale: str, confidence: float = None, **kwargs):
        """의사결정 기록"""
        self._safe_log(
            "log_decision",
            decision=decision,
            options_considered=options,
            rationale=rationale,
            confidence=confidence,
            impact=kwargs.get('impact'),
        )

    def log_insight(self, insight: str, source: str, importance: str = "medium", confidence: float = None):
        """인사이트 도출 기록"""
        self._safe_log(
            "log_insight",
            insight=insight,
            source=source,
            importance=importance,
            confidence=confidence,
        )

    def log_hypothesis(self, hypothesis: str, basis: List[str], confidence: float = None):
        """가설 수립 기록"""
        self._safe_log(
            "log_hypothesis",
            hypothesis=hypothesis,
            basis=basis,
            confidence=confidence,
        )

    def start_reasoning_chain(self, goal: str) -> Optional[str]:
        """추론 체인 시작"""
        jl = _get_job_logger()
        if not jl or not jl.is_active():
            return None
        try:
            return jl.start_reasoning_chain(
                agent_name=self._agent_name,
                goal=goal,
            )
        except Exception:
            return None

    def add_reasoning_step(self, chain_id: str, thinking_type: str, content: str, confidence: float = None):
        """추론 체인에 단계 추가"""
        if not chain_id:
            return
        jl = _get_job_logger()
        if not jl or not jl.is_active():
            return
        try:
            from ...common.utils.job_logger import ThinkingType
            tt = ThinkingType(thinking_type) if thinking_type in [t.value for t in ThinkingType] else ThinkingType.REASONING
            jl.add_reasoning_step(
                chain_id=chain_id,
                thinking_type=tt,
                content=content,
                confidence=confidence,
            )
        except Exception:
            pass

    def end_reasoning_chain(self, chain_id: str, conclusion: str, confidence: float = None, success: bool = True):
        """추론 체인 종료"""
        if not chain_id:
            return
        self._safe_log(
            "end_reasoning_chain",
            chain_id=chain_id,
            conclusion=conclusion,
            confidence=confidence,
            success=success,
        )

    def set_analysis_summary(self, summary: str):
        """분석 요약 설정"""
        self._safe_log("set_analysis_summary", summary=summary)
