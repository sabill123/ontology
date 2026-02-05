"""
Temperature Scaling for Confidence Calibration (v17.0)

Platt Scaling의 단일 파라미터 버전인 Temperature Scaling 구현:
- 모델의 logits을 temperature로 나누어 확률 보정
- ECE (Expected Calibration Error) 최소화
- 에이전트별 최적 temperature 학습

참고: Guo et al. "On Calibration of Modern Neural Networks" (ICML 2017)
"""

import logging
import math
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


@dataclass
class ScalingResult:
    """Temperature Scaling 결과"""
    original_confidence: float
    calibrated_confidence: float
    temperature: float
    adjustment_factor: float  # calibrated / original
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    @property
    def was_overconfident(self) -> bool:
        """과신이었는지 여부"""
        return self.original_confidence > self.calibrated_confidence

    @property
    def adjustment_magnitude(self) -> float:
        """조정 크기 (%)"""
        if self.original_confidence == 0:
            return 0.0
        return abs(self.calibrated_confidence - self.original_confidence) / self.original_confidence


class CalibrationBin:
    """ECE 계산을 위한 보정 빈"""

    def __init__(self, lower: float, upper: float):
        self.lower = lower
        self.upper = upper
        self.confidences: List[float] = []
        self.accuracies: List[float] = []  # 1 if correct, 0 if incorrect

    def add(self, confidence: float, is_correct: bool) -> None:
        """샘플 추가"""
        if self.lower <= confidence < self.upper:
            self.confidences.append(confidence)
            self.accuracies.append(1.0 if is_correct else 0.0)

    @property
    def count(self) -> int:
        return len(self.confidences)

    @property
    def avg_confidence(self) -> float:
        if not self.confidences:
            return 0.0
        return sum(self.confidences) / len(self.confidences)

    @property
    def avg_accuracy(self) -> float:
        if not self.accuracies:
            return 0.0
        return sum(self.accuracies) / len(self.accuracies)

    @property
    def gap(self) -> float:
        """신뢰도-정확도 갭"""
        return abs(self.avg_accuracy - self.avg_confidence)


class TemperatureScaler:
    """
    Temperature Scaling을 통한 신뢰도 보정

    LLM이 출력하는 confidence 값을 실제 정확도에 맞게 보정합니다.
    T > 1: 과신 완화 (softmax 평탄화)
    T < 1: 저신뢰 보정 (softmax 첨예화)
    T = 1: 보정 없음
    """

    def __init__(
        self,
        initial_temperature: float = 1.0,
        learning_rate: float = 0.1,
        min_samples_for_update: int = 10,
        num_bins: int = 10,
    ):
        """
        Args:
            initial_temperature: 초기 temperature 값
            learning_rate: temperature 업데이트 학습률
            min_samples_for_update: temperature 업데이트에 필요한 최소 샘플 수
            num_bins: ECE 계산용 빈 개수
        """
        self.temperature = initial_temperature
        self.learning_rate = learning_rate
        self.min_samples_for_update = min_samples_for_update
        self.num_bins = num_bins

        # 히스토리
        self.calibration_history: List[Tuple[float, bool]] = []  # (confidence, was_correct)
        self.temperature_history: List[Tuple[str, float]] = []  # (timestamp, temperature)

        # 빈 초기화
        self._reset_bins()

        logger.info(f"TemperatureScaler initialized with T={initial_temperature}")

    def _reset_bins(self) -> None:
        """보정 빈 초기화"""
        bin_width = 1.0 / self.num_bins
        self.bins = [
            CalibrationBin(i * bin_width, (i + 1) * bin_width)
            for i in range(self.num_bins)
        ]

    def scale(self, confidence: float) -> ScalingResult:
        """
        신뢰도를 temperature로 스케일링

        Args:
            confidence: 원본 신뢰도 (0-1)

        Returns:
            ScalingResult: 보정된 신뢰도 결과
        """
        # logit 변환 (0, 1 범위 보호)
        confidence = max(0.001, min(0.999, confidence))
        logit = math.log(confidence / (1 - confidence))

        # Temperature scaling
        scaled_logit = logit / self.temperature

        # sigmoid로 확률 복원
        calibrated = 1 / (1 + math.exp(-scaled_logit))

        return ScalingResult(
            original_confidence=confidence,
            calibrated_confidence=calibrated,
            temperature=self.temperature,
            adjustment_factor=calibrated / confidence if confidence > 0 else 1.0,
        )

    def record_outcome(self, confidence: float, was_correct: bool) -> None:
        """
        예측 결과 기록 (temperature 학습용)

        Args:
            confidence: 예측 시 신뢰도
            was_correct: 실제로 맞았는지 여부
        """
        self.calibration_history.append((confidence, was_correct))

        # 해당 빈에 추가
        for bin_obj in self.bins:
            bin_obj.add(confidence, was_correct)

        # 충분한 샘플이 쌓이면 temperature 업데이트
        if len(self.calibration_history) >= self.min_samples_for_update:
            self._update_temperature()

    def _update_temperature(self) -> None:
        """
        ECE 최소화를 위한 temperature 업데이트

        Grid search 방식으로 최적 temperature 탐색
        """
        best_temperature = self.temperature
        best_ece = self.compute_ece()

        # Grid search: 0.5 ~ 3.0 범위에서 탐색
        for t in [x / 10 for x in range(5, 31)]:
            # 임시로 temperature 변경
            old_temp = self.temperature
            self.temperature = t

            ece = self.compute_ece()

            if ece < best_ece:
                best_ece = ece
                best_temperature = t

            # 원복
            self.temperature = old_temp

        # 최적 temperature 적용 (급격한 변화 방지)
        delta = best_temperature - self.temperature
        self.temperature += self.learning_rate * delta

        self.temperature_history.append((datetime.now().isoformat(), self.temperature))

        logger.info(f"Temperature updated: {self.temperature:.3f} (ECE: {best_ece:.4f})")

    def compute_ece(self) -> float:
        """
        Expected Calibration Error 계산

        ECE = Σ (n_b / N) * |accuracy_b - confidence_b|
        """
        total_samples = sum(bin_obj.count for bin_obj in self.bins)
        if total_samples == 0:
            return 0.0

        ece = 0.0
        for bin_obj in self.bins:
            if bin_obj.count > 0:
                weight = bin_obj.count / total_samples
                ece += weight * bin_obj.gap

        return ece

    def compute_mce(self) -> float:
        """
        Maximum Calibration Error 계산

        MCE = max_b |accuracy_b - confidence_b|
        """
        max_gap = 0.0
        for bin_obj in self.bins:
            if bin_obj.count > 0:
                max_gap = max(max_gap, bin_obj.gap)
        return max_gap

    def get_reliability_diagram_data(self) -> Dict[str, List[float]]:
        """
        Reliability Diagram 데이터 반환

        Returns:
            {"bin_centers": [...], "accuracies": [...], "confidences": [...], "counts": [...]}
        """
        bin_width = 1.0 / self.num_bins

        return {
            "bin_centers": [
                (bin_obj.lower + bin_obj.upper) / 2 for bin_obj in self.bins
            ],
            "accuracies": [bin_obj.avg_accuracy for bin_obj in self.bins],
            "confidences": [bin_obj.avg_confidence for bin_obj in self.bins],
            "counts": [bin_obj.count for bin_obj in self.bins],
        }

    def get_calibration_summary(self) -> Dict[str, Any]:
        """보정 요약 정보"""
        return {
            "temperature": self.temperature,
            "total_samples": len(self.calibration_history),
            "ece": self.compute_ece(),
            "mce": self.compute_mce(),
            "temperature_history_count": len(self.temperature_history),
            "reliability_diagram": self.get_reliability_diagram_data(),
        }

    def reset(self) -> None:
        """상태 초기화"""
        self.temperature = 1.0
        self.calibration_history = []
        self.temperature_history = []
        self._reset_bins()
        logger.info("TemperatureScaler reset")


# 에이전트별 Temperature Scaler 관리
class AgentTemperatureManager:
    """
    에이전트별 Temperature Scaler 관리

    각 에이전트는 고유한 보정 특성을 가질 수 있으므로
    개별 TemperatureScaler를 관리합니다.
    """

    def __init__(self):
        self.scalers: Dict[str, TemperatureScaler] = {}
        self._default_scaler = TemperatureScaler()

    def get_scaler(self, agent_name: str) -> TemperatureScaler:
        """에이전트별 scaler 반환 (없으면 생성)"""
        if agent_name not in self.scalers:
            self.scalers[agent_name] = TemperatureScaler()
            logger.info(f"Created TemperatureScaler for agent: {agent_name}")
        return self.scalers[agent_name]

    def scale_confidence(self, agent_name: str, confidence: float) -> ScalingResult:
        """특정 에이전트의 신뢰도 보정"""
        scaler = self.get_scaler(agent_name)
        return scaler.scale(confidence)

    def record_agent_outcome(
        self,
        agent_name: str,
        confidence: float,
        was_correct: bool
    ) -> None:
        """에이전트 예측 결과 기록"""
        scaler = self.get_scaler(agent_name)
        scaler.record_outcome(confidence, was_correct)

    def get_all_calibration_summaries(self) -> Dict[str, Dict[str, Any]]:
        """모든 에이전트의 보정 요약"""
        return {
            agent: scaler.get_calibration_summary()
            for agent, scaler in self.scalers.items()
        }
