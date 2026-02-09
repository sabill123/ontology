"""
Counterfactual Analysis - 반사실적 분석 및 민감도 분석

CounterfactualAnalyzer, SensitivityAnalyzer
"""

from typing import Optional
import numpy as np

from .models import CounterfactualResult, SensitivityResult
from .estimators import OutcomeModel


class CounterfactualAnalyzer:
    """반사실적 분석"""

    def __init__(self):
        self.outcome_model = OutcomeModel()

    def analyze(
        self,
        X: np.ndarray,
        treatment: np.ndarray,
        outcome: np.ndarray,
        target_idx: int = 0
    ) -> CounterfactualResult:
        """
        특정 개체에 대한 반사실적 분석

        "만약 처리를 받았다면(받지 않았다면) 결과가 어떻게 달랐을까?"
        """
        self.outcome_model.fit(X, treatment, outcome)

        # 실제 결과
        factual = outcome[target_idx]
        actual_treatment = treatment[target_idx]

        # 반사실적 결과 예측
        x_target = X[target_idx:target_idx + 1]
        counterfactual_treatment = 1 - actual_treatment

        cf_outcome = self.outcome_model.predict(
            x_target,
            np.array([counterfactual_treatment])
        )[0]

        # 개별 처리 효과
        if actual_treatment == 1:
            ite = factual - cf_outcome  # 처리 효과 = 처리 받은 결과 - 받지 않은 결과
        else:
            ite = cf_outcome - factual

        # 신뢰도 (모델 적합도 기반)
        residuals = outcome - self.outcome_model.predict(X, treatment)
        rmse = np.sqrt(np.mean(residuals ** 2))
        confidence = max(0, 1 - rmse / np.std(outcome)) if np.std(outcome) > 0 else 0.5

        # 해석
        if actual_treatment == 1:
            if ite > 0:
                explanation = f"처리를 받음으로써 결과가 {ite:.2f} 만큼 증가했을 것으로 추정"
            else:
                explanation = f"처리를 받음으로써 결과가 {abs(ite):.2f} 만큼 감소했을 것으로 추정"
        else:
            if ite > 0:
                explanation = f"처리를 받았다면 결과가 {ite:.2f} 만큼 증가했을 것으로 추정"
            else:
                explanation = f"처리를 받았다면 결과가 {abs(ite):.2f} 만큼 감소했을 것으로 추정"

        return CounterfactualResult(
            factual_outcome=factual,
            counterfactual_outcome=cf_outcome,
            individual_effect=ite,
            confidence=confidence,
            explanation=explanation
        )


class SensitivityAnalyzer:
    """민감도 분석 - 미관측 교란에 대한 강건성 평가"""

    def analyze(
        self,
        ate_estimate: float,
        std_error: float,
        r2_y_z: float = 0.3,  # 가정: 미관측 교란이 결과 분산의 얼마를 설명하는가
        r2_t_z: float = 0.3   # 가정: 미관측 교란이 처리 분산의 얼마를 설명하는가
    ) -> SensitivityResult:
        """
        Rosenbaum bounds 스타일 민감도 분석

        미관측 교란이 있을 때 결과가 어떻게 바뀔 수 있는지 평가
        """
        # 편향 근사: bias ≈ sqrt(r2_y_z * r2_t_z) * sigma
        # 여기서 sigma는 결과의 표준편차 (std_error로 대체)
        potential_bias = np.sqrt(r2_y_z * r2_t_z) * std_error * 5

        # Robustness value: 결과를 뒤집는데 필요한 교란 정도
        # |ATE| < bias 이면 결과가 민감
        if abs(ate_estimate) < potential_bias:
            robustness = abs(ate_estimate) / (potential_bias + 1e-10)
            interpretation = f"결과가 미관측 교란에 민감함. 약 {(1-robustness)*100:.0f}%의 교란으로 결과 역전 가능"
        else:
            robustness = 1.0
            interpretation = "결과가 미관측 교란에 강건함"

        # Critical bias: ATE를 0으로 만드는 편향
        critical_bias = abs(ate_estimate)

        return SensitivityResult(
            robustness_value=robustness,
            critical_bias=critical_bias,
            interpretation=interpretation
        )
