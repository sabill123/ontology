"""
Confidence Calibrator - Main Interface (v17.0)

LLM 신뢰도 보정의 메인 인터페이스:
- Temperature Scaling + Uncertainty Quantification 통합
- 에이전트별 보정 프로파일 관리
- Evidence Chain 연동
- SharedContext와 통합

사용 예시:
    calibrator = ConfidenceCalibrator(context)
    result = calibrator.calibrate(
        agent="DataAnalyst",
        raw_confidence=0.95,
        samples=[...],  # 다중 샘플링 결과 (옵션)
    )
    # result.calibrated_confidence → 보정된 신뢰도
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, TYPE_CHECKING
from datetime import datetime
from enum import Enum

from .temperature_scaler import TemperatureScaler, AgentTemperatureManager, ScalingResult
from .uncertainty_quantifier import (
    UncertaintyQuantifier,
    UncertaintyEstimate,
    UncertaintyType,
    SemanticConsistencyChecker,
)

if TYPE_CHECKING:
    from ..shared_context import SharedContext

logger = logging.getLogger(__name__)


@dataclass
class CalibrationConfig:
    """보정 설정"""
    # Temperature Scaling
    initial_temperature: float = 1.0
    learning_rate: float = 0.1
    min_samples_for_update: int = 10

    # Uncertainty Quantification
    num_uncertainty_samples: int = 5
    consistency_threshold: float = 0.8

    # 보정 정책
    apply_temperature_scaling: bool = True
    apply_uncertainty_adjustment: bool = True
    uncertainty_weight: float = 0.3  # 불확실성 반영 비중

    # 임계값
    high_confidence_threshold: float = 0.9  # 과신 경고 임계값
    low_confidence_threshold: float = 0.3  # 저신뢰 경고 임계값

    # 에이전트별 커스텀 설정
    agent_overrides: Dict[str, Dict[str, Any]] = field(default_factory=dict)


@dataclass
class CalibrationResult:
    """보정 결과"""
    # 신뢰도 값들
    raw_confidence: float  # 원본 신뢰도
    scaled_confidence: float  # Temperature Scaling 후
    uncertainty_adjusted: float  # 불확실성 조정 후
    final_confidence: float  # 최종 보정된 신뢰도

    # 보정 상세
    temperature_used: float
    uncertainty_estimate: Optional[UncertaintyEstimate]

    # 메타데이터
    agent: str
    calibration_method: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    # 경고
    warnings: List[str] = field(default_factory=list)

    @property
    def adjustment_magnitude(self) -> float:
        """조정 크기 (%)"""
        if self.raw_confidence == 0:
            return 0.0
        return abs(self.final_confidence - self.raw_confidence) / self.raw_confidence

    @property
    def was_overconfident(self) -> bool:
        """과신이었는지"""
        return self.raw_confidence > self.final_confidence

    def to_dict(self) -> Dict[str, Any]:
        return {
            "raw_confidence": self.raw_confidence,
            "scaled_confidence": self.scaled_confidence,
            "uncertainty_adjusted": self.uncertainty_adjusted,
            "final_confidence": self.final_confidence,
            "temperature_used": self.temperature_used,
            "uncertainty_estimate": (
                self.uncertainty_estimate.to_dict()
                if self.uncertainty_estimate else None
            ),
            "agent": self.agent,
            "calibration_method": self.calibration_method,
            "adjustment_magnitude": self.adjustment_magnitude,
            "was_overconfident": self.was_overconfident,
            "warnings": self.warnings,
            "timestamp": self.timestamp,
        }


class ConfidenceCalibrator:
    """
    LLM 신뢰도 보정기 (메인 인터페이스)

    기능:
    1. Temperature Scaling을 통한 확률 보정
    2. Uncertainty Quantification으로 불확실성 반영
    3. 에이전트별 보정 프로파일 관리
    4. Evidence Chain 연동 (근거 기록)
    5. 히스토리 기반 자동 튜닝
    """

    def __init__(
        self,
        context: Optional["SharedContext"] = None,
        config: Optional[CalibrationConfig] = None,
    ):
        """
        Args:
            context: SharedContext (Evidence Chain 연동용)
            config: 보정 설정
        """
        self.context = context
        self.config = config or CalibrationConfig()

        # Temperature Scaling
        self.temperature_manager = AgentTemperatureManager()

        # Uncertainty Quantification
        self.uncertainty_quantifier = UncertaintyQuantifier(
            num_samples=self.config.num_uncertainty_samples,
            consistency_threshold=self.config.consistency_threshold,
        )

        # 일관성 체커
        self.consistency_checker = SemanticConsistencyChecker()

        # 보정 히스토리
        self.calibration_history: List[CalibrationResult] = []

        # 에이전트별 통계
        self.agent_stats: Dict[str, Dict[str, Any]] = {}

        logger.info("ConfidenceCalibrator initialized")

    def calibrate(
        self,
        agent: str,
        raw_confidence: float,
        samples: Optional[List[Dict[str, Any]]] = None,
        token_probs: Optional[List[float]] = None,
        record_evidence: bool = True,
    ) -> CalibrationResult:
        """
        신뢰도 보정

        Args:
            agent: 에이전트 이름
            raw_confidence: 원본 신뢰도 (0-1)
            samples: 다중 샘플링 결과 (불확실성 추정용)
            token_probs: 토큰 확률 (불확실성 추정용)
            record_evidence: Evidence Chain에 기록할지 여부

        Returns:
            CalibrationResult: 보정 결과
        """
        warnings = []

        # 1. 입력 검증 및 클램프
        raw_confidence = max(0.0, min(1.0, raw_confidence))

        # 과신 경고
        if raw_confidence > self.config.high_confidence_threshold:
            warnings.append(f"High confidence ({raw_confidence:.0%}) may indicate overconfidence")

        # 저신뢰 경고
        if raw_confidence < self.config.low_confidence_threshold:
            warnings.append(f"Low confidence ({raw_confidence:.0%}) indicates high uncertainty")

        # 2. Temperature Scaling
        scaled_confidence = raw_confidence
        temperature_used = 1.0

        if self.config.apply_temperature_scaling:
            scaling_result = self.temperature_manager.scale_confidence(agent, raw_confidence)
            scaled_confidence = scaling_result.calibrated_confidence
            temperature_used = scaling_result.temperature

        # 3. Uncertainty Quantification
        uncertainty_estimate: Optional[UncertaintyEstimate] = None
        uncertainty_adjusted = scaled_confidence

        if self.config.apply_uncertainty_adjustment:
            if samples and len(samples) > 1:
                # 다중 샘플 기반 추정
                uncertainty_estimate = self.uncertainty_quantifier.estimate_from_samples(samples)
            else:
                # 단일 샘플 기반 추정
                uncertainty_estimate = self.uncertainty_quantifier.estimate_single(
                    raw_confidence,
                    token_probs
                )

            # 불확실성 반영
            uncertainty_penalty = uncertainty_estimate.total * self.config.uncertainty_weight
            uncertainty_adjusted = scaled_confidence * (1.0 - uncertainty_penalty)

        # 4. 최종 신뢰도
        final_confidence = max(0.0, min(1.0, uncertainty_adjusted))

        # 5. 결과 생성
        calibration_method = self._determine_method(samples, token_probs)

        result = CalibrationResult(
            raw_confidence=raw_confidence,
            scaled_confidence=scaled_confidence,
            uncertainty_adjusted=uncertainty_adjusted,
            final_confidence=final_confidence,
            temperature_used=temperature_used,
            uncertainty_estimate=uncertainty_estimate,
            agent=agent,
            calibration_method=calibration_method,
            warnings=warnings,
        )

        # 6. 히스토리 기록
        self.calibration_history.append(result)
        self._update_agent_stats(agent, result)

        # 7. Evidence Chain 기록
        if record_evidence and self.context and self.context.evidence_chain:
            self._record_to_evidence_chain(result)

        logger.debug(
            f"Calibrated {agent}: {raw_confidence:.2%} → {final_confidence:.2%} "
            f"(T={temperature_used:.2f})"
        )

        return result

    def record_outcome(
        self,
        agent: str,
        confidence: float,
        was_correct: bool
    ) -> None:
        """
        예측 결과 기록 (온라인 학습)

        실제 결과가 밝혀지면 호출하여 Temperature를 업데이트합니다.

        Args:
            agent: 에이전트 이름
            confidence: 예측 시 신뢰도
            was_correct: 실제로 맞았는지
        """
        self.temperature_manager.record_agent_outcome(agent, confidence, was_correct)

        # 에이전트 통계 업데이트
        if agent not in self.agent_stats:
            self.agent_stats[agent] = {
                "total_predictions": 0,
                "correct_predictions": 0,
                "calibrated_predictions": 0,
            }

        self.agent_stats[agent]["total_predictions"] += 1
        if was_correct:
            self.agent_stats[agent]["correct_predictions"] += 1

        logger.debug(f"Recorded outcome for {agent}: correct={was_correct}")

    def _determine_method(
        self,
        samples: Optional[List[Dict[str, Any]]],
        token_probs: Optional[List[float]]
    ) -> str:
        """보정 방법 결정"""
        methods = []

        if self.config.apply_temperature_scaling:
            methods.append("temperature_scaling")

        if self.config.apply_uncertainty_adjustment:
            if samples and len(samples) > 1:
                methods.append("multi_sample_uncertainty")
            elif token_probs:
                methods.append("token_prob_uncertainty")
            else:
                methods.append("heuristic_uncertainty")

        return "+".join(methods) if methods else "none"

    def _update_agent_stats(self, agent: str, result: CalibrationResult) -> None:
        """에이전트별 통계 업데이트"""
        if agent not in self.agent_stats:
            self.agent_stats[agent] = {
                "calibrated_predictions": 0,
                "total_adjustment": 0.0,
                "overconfident_count": 0,
                "underconfident_count": 0,
            }

        stats = self.agent_stats[agent]
        stats["calibrated_predictions"] += 1
        stats["total_adjustment"] += result.adjustment_magnitude

        if result.was_overconfident:
            stats["overconfident_count"] += 1
        elif result.raw_confidence < result.final_confidence:
            stats["underconfident_count"] += 1

    def _record_to_evidence_chain(self, result: CalibrationResult) -> None:
        """Evidence Chain에 보정 결과 기록"""
        try:
            from ..autonomous.evidence_chain import EvidenceType

            # EvidenceType에 CALIBRATION이 없으면 QUALITY_ASSESSMENT 사용
            evidence_type = getattr(
                EvidenceType,
                "CALIBRATION",
                EvidenceType.QUALITY_ASSESSMENT
            )

            self.context.evidence_chain.add_block(
                phase="calibration",
                agent="ConfidenceCalibrator",
                evidence_type=evidence_type,
                finding=f"Calibrated {result.agent} confidence: {result.raw_confidence:.0%} → {result.final_confidence:.0%}",
                reasoning=f"Applied {result.calibration_method} with T={result.temperature_used:.2f}",
                conclusion=(
                    "Overconfidence detected and adjusted"
                    if result.was_overconfident
                    else "Confidence calibrated within normal range"
                ),
                metrics={
                    "raw_confidence": result.raw_confidence,
                    "final_confidence": result.final_confidence,
                    "temperature": result.temperature_used,
                    "adjustment_magnitude": result.adjustment_magnitude,
                },
                confidence=result.final_confidence,
            )
        except Exception as e:
            logger.warning(f"Failed to record to evidence chain: {e}")

    def get_agent_calibration_profile(self, agent: str) -> Dict[str, Any]:
        """에이전트 보정 프로파일 조회"""
        scaler = self.temperature_manager.get_scaler(agent)
        scaler_summary = scaler.get_calibration_summary()

        agent_specific_stats = self.agent_stats.get(agent, {})

        return {
            "agent": agent,
            "temperature": scaler_summary["temperature"],
            "ece": scaler_summary["ece"],
            "mce": scaler_summary["mce"],
            "total_calibrated": agent_specific_stats.get("calibrated_predictions", 0),
            "avg_adjustment": (
                agent_specific_stats.get("total_adjustment", 0) /
                max(1, agent_specific_stats.get("calibrated_predictions", 1))
            ),
            "overconfident_ratio": (
                agent_specific_stats.get("overconfident_count", 0) /
                max(1, agent_specific_stats.get("calibrated_predictions", 1))
            ),
            "reliability_diagram": scaler_summary["reliability_diagram"],
        }

    def get_calibration_summary(self) -> Dict[str, Any]:
        """전체 보정 요약"""
        uncertainty_summary = self.uncertainty_quantifier.get_uncertainty_summary()

        return {
            "total_calibrations": len(self.calibration_history),
            "agents_tracked": list(self.agent_stats.keys()),
            "agent_profiles": {
                agent: self.get_agent_calibration_profile(agent)
                for agent in self.agent_stats.keys()
            },
            "uncertainty_summary": uncertainty_summary,
            "config": {
                "apply_temperature_scaling": self.config.apply_temperature_scaling,
                "apply_uncertainty_adjustment": self.config.apply_uncertainty_adjustment,
                "uncertainty_weight": self.config.uncertainty_weight,
            },
        }

    def reset(self) -> None:
        """전체 상태 초기화"""
        self.temperature_manager = AgentTemperatureManager()
        self.uncertainty_quantifier.reset()
        self.calibration_history = []
        self.agent_stats = {}
        logger.info("ConfidenceCalibrator reset")


# 편의 함수
def calibrate_confidence(
    agent: str,
    confidence: float,
    context: Optional["SharedContext"] = None,
) -> float:
    """
    간단한 신뢰도 보정 함수

    Args:
        agent: 에이전트 이름
        confidence: 원본 신뢰도
        context: SharedContext (옵션)

    Returns:
        보정된 신뢰도
    """
    calibrator = ConfidenceCalibrator(context)
    result = calibrator.calibrate(agent, confidence, record_evidence=False)
    return result.final_confidence
