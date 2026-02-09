"""
Confidence Calibration

논문: "A Survey of Uncertainty in Deep Neural Networks" (2023)

핵심:
- Temperature Scaling으로 과신(overconfidence) 보정
- Expected Calibration Error (ECE) 최소화
- Reliability Diagram 기반 분석
"""

import logging
from typing import Dict, List, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class ConfidenceCalibrator:
    """
    신뢰도 보정 (Calibration)

    논문: "A Survey of Uncertainty in Deep Neural Networks" (2023)

    핵심:
    - Temperature Scaling으로 과신(overconfidence) 보정
    - Expected Calibration Error (ECE) 최소화
    - Reliability Diagram 기반 분석
    """

    def __init__(self, n_bins: int = 10):
        self.n_bins = n_bins
        self.temperature = 1.0
        self.calibration_history = []

    def temperature_scaling(self, logits: np.ndarray, temperature: float = None) -> np.ndarray:
        """
        Temperature Scaling: p = softmax(logits / T)
        T > 1: 분포를 더 균일하게 (과신 완화)
        T < 1: 분포를 더 뾰족하게 (확신 강화)
        """
        T = temperature or self.temperature
        scaled = logits / T
        # Softmax
        exp_scaled = np.exp(scaled - np.max(scaled))
        return exp_scaled / exp_scaled.sum()

    def calculate_ece(
        self,
        confidences: np.ndarray,
        accuracies: np.ndarray
    ) -> float:
        """
        Expected Calibration Error
        ECE = Σ (|Bm|/n) * |acc(Bm) - conf(Bm)|

        낮을수록 좋음 (0 = 완벽한 calibration)
        """
        bin_boundaries = np.linspace(0, 1, self.n_bins + 1)
        ece = 0.0
        n = len(confidences)

        for i in range(self.n_bins):
            lower, upper = bin_boundaries[i], bin_boundaries[i + 1]
            in_bin = (confidences >= lower) & (confidences < upper)

            if in_bin.sum() > 0:
                avg_conf = confidences[in_bin].mean()
                avg_acc = accuracies[in_bin].mean()
                bin_weight = in_bin.sum() / n
                ece += bin_weight * abs(avg_acc - avg_conf)

        return ece

    def find_optimal_temperature(
        self,
        predictions: List[Dict],
        ground_truth: List[bool],
        temp_range: Tuple[float, float] = (0.5, 3.0),
        n_steps: int = 50
    ) -> float:
        """최적 Temperature 탐색 (ECE 최소화)"""
        if not predictions or not ground_truth:
            return 1.0

        confidences = np.array([p.get('confidence', 0.5) for p in predictions])
        accuracies = np.array([1.0 if g else 0.0 for g in ground_truth])

        best_temp = 1.0
        best_ece = float('inf')

        for temp in np.linspace(temp_range[0], temp_range[1], n_steps):
            # Temperature 적용
            calibrated = np.clip(confidences / temp, 0.01, 0.99)
            ece = self.calculate_ece(calibrated, accuracies)

            if ece < best_ece:
                best_ece = ece
                best_temp = temp

        self.temperature = best_temp
        logger.info(f"Optimal temperature: {best_temp:.3f}, ECE: {best_ece:.4f}")
        return best_temp

    def calibrate(self, confidence: float) -> Dict[str, float]:
        """단일 신뢰도 보정"""
        calibrated = np.clip(confidence / self.temperature, 0.01, 0.99)

        # Bootstrap 신뢰 구간 (근사)
        std_error = np.sqrt(calibrated * (1 - calibrated))
        ci_lower = max(0, calibrated - 1.96 * std_error)
        ci_upper = min(1, calibrated + 1.96 * std_error)

        return {
            'original': confidence,
            'calibrated': float(calibrated),
            'temperature': self.temperature,
            'confidence_interval': (float(ci_lower), float(ci_upper)),
        }
