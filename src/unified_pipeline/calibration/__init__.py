"""
Confidence Calibration Module (v17.0)

LLM 신뢰도 보정 시스템:
- Temperature Scaling을 통한 확률 보정
- 불확실성 정량화 (알레아토릭 + 에피스테믹)
- 히스토리 기반 신뢰도 추적
- 에이전트별 보정 매개변수 학습

핵심 개념:
1. ConfidenceCalibrator: 메인 보정 인터페이스
2. TemperatureScaler: Temperature Scaling 알고리즘
3. UncertaintyQuantifier: 불확실성 정량화
"""

from .confidence_calibrator import (
    ConfidenceCalibrator,
    CalibrationResult,
    CalibrationConfig,
)
from .temperature_scaler import (
    TemperatureScaler,
    ScalingResult,
)
from .uncertainty_quantifier import (
    UncertaintyQuantifier,
    UncertaintyEstimate,
    UncertaintyType,
)

__all__ = [
    # Main calibrator
    "ConfidenceCalibrator",
    "CalibrationResult",
    "CalibrationConfig",
    # Temperature scaling
    "TemperatureScaler",
    "ScalingResult",
    # Uncertainty
    "UncertaintyQuantifier",
    "UncertaintyEstimate",
    "UncertaintyType",
]
