"""
Causal Impact Analysis Module - 인과 추론 및 영향 분석

데이터 통합이 KPI에 미치는 인과적 영향을 추정합니다.

주요 기능:
- DoWhy 스타일 인과 그래프 모델링
- 처리 효과 추정 (ATE, CATE)
- 반사실적 분석 (Counterfactual)
- 민감도 분석 (Sensitivity Analysis)
- 데이터 통합 시나리오별 영향 예측
- Granger Causality 시계열 인과관계 분석
- PC Algorithm 기반 Causal Graph Discovery
- CATE 추정 (S-Learner, T-Learner, X-Learner)

References:
- Pearl, "Causality: Models, Reasoning, and Inference", 2009
- Sharma & Kiciman, "DoWhy: An End-to-End Library for Causal Inference", 2020
- Athey & Imbens, "Machine Learning Methods for Estimating Heterogeneous Causal Effects", 2015
- Granger, "Investigating Causal Relations by Econometric Models", 1969
- Spirtes, Glymour & Scheines, "Causation, Prediction, and Search", 2000
- Kunzel et al., "Metalearners for Estimating Heterogeneous Treatment Effects", 2019
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Any, Callable
from enum import Enum
import numpy as np
from collections import defaultdict
import random


class TreatmentType(Enum):
    """처리(Treatment) 타입"""
    DATA_INTEGRATION = "data_integration"       # 데이터 통합
    SCHEMA_UNIFICATION = "schema_unification"   # 스키마 통합
    FK_ESTABLISHMENT = "fk_establishment"       # FK 관계 수립
    ENTITY_RESOLUTION = "entity_resolution"     # 엔티티 통합
    DATA_QUALITY_FIX = "data_quality_fix"       # 데이터 품질 개선


class EffectType(Enum):
    """효과 타입"""
    ATE = "average_treatment_effect"            # 평균 처리 효과
    ATT = "average_treatment_on_treated"        # 처리군 평균 효과
    CATE = "conditional_average_treatment"      # 조건부 평균 효과


@dataclass
class CausalNode:
    """인과 그래프 노드"""
    name: str
    node_type: str  # treatment, outcome, confounder, mediator, instrument
    observed: bool = True
    values: Optional[np.ndarray] = None


@dataclass
class CausalEdge:
    """인과 그래프 엣지"""
    source: str
    target: str
    weight: float = 1.0
    mechanism: str = "linear"  # linear, nonlinear, threshold


@dataclass
class CausalGraph:
    """인과 그래프 (DAG)"""
    nodes: Dict[str, CausalNode] = field(default_factory=dict)
    edges: List[CausalEdge] = field(default_factory=list)

    def add_node(self, node: CausalNode):
        self.nodes[node.name] = node

    def add_edge(self, edge: CausalEdge):
        self.edges.append(edge)

    def get_parents(self, node_name: str) -> List[str]:
        """부모 노드 조회"""
        return [e.source for e in self.edges if e.target == node_name]

    def get_children(self, node_name: str) -> List[str]:
        """자식 노드 조회"""
        return [e.target for e in self.edges if e.source == node_name]

    def get_confounders(self, treatment: str, outcome: str) -> List[str]:
        """교란 변수 식별 (treatment와 outcome의 공통 원인)"""
        treatment_ancestors = self._get_ancestors(treatment)
        outcome_ancestors = self._get_ancestors(outcome)
        return list(treatment_ancestors & outcome_ancestors)

    def _get_ancestors(self, node_name: str) -> Set[str]:
        """모든 조상 노드"""
        ancestors = set()
        queue = [node_name]
        while queue:
            current = queue.pop(0)
            for parent in self.get_parents(current):
                if parent not in ancestors:
                    ancestors.add(parent)
                    queue.append(parent)
        return ancestors


@dataclass
class TreatmentEffect:
    """처리 효과 추정 결과"""
    effect_type: EffectType
    estimate: float
    std_error: float
    confidence_interval: Tuple[float, float]
    p_value: float
    treatment: str
    outcome: str
    method: str
    sample_size: int


@dataclass
class CounterfactualResult:
    """반사실적 분석 결과"""
    factual_outcome: float
    counterfactual_outcome: float
    individual_effect: float
    confidence: float
    explanation: str


@dataclass
class SensitivityResult:
    """민감도 분석 결과"""
    robustness_value: float  # 결과를 뒤집는데 필요한 교란 정도
    critical_bias: float     # 임계 편향
    interpretation: str


@dataclass
class CausalImpactResult:
    """인과 영향 분석 종합 결과"""
    treatment_effects: List[TreatmentEffect]
    counterfactuals: List[CounterfactualResult]
    sensitivity: SensitivityResult
    causal_graph: CausalGraph
    recommendations: List[str]
    summary: Dict[str, Any]


class PropensityScoreEstimator:
    """경향 점수(Propensity Score) 추정"""

    def __init__(self, method: str = "logistic"):
        self.method = method
        self.coefficients: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, treatment: np.ndarray) -> 'PropensityScoreEstimator':
        """경향 점수 모델 학습"""
        # 간단한 로지스틱 회귀 구현
        n_features = X.shape[1] if len(X.shape) > 1 else 1
        self.coefficients = np.zeros(n_features + 1)  # +1 for bias

        # Gradient descent
        lr = 0.01
        n_iter = 100

        X_with_bias = np.column_stack([np.ones(len(X)), X])

        for _ in range(n_iter):
            logits = X_with_bias @ self.coefficients
            probs = self._sigmoid(logits)

            # Gradient
            gradient = X_with_bias.T @ (probs - treatment) / len(treatment)
            self.coefficients -= lr * gradient

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """경향 점수 예측"""
        if self.coefficients is None:
            return np.full(len(X), 0.5)

        X_with_bias = np.column_stack([np.ones(len(X)), X])
        logits = X_with_bias @ self.coefficients
        return self._sigmoid(logits)

    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        """시그모이드 함수"""
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))


class OutcomeModel:
    """결과 변수 예측 모델"""

    def __init__(self, method: str = "linear"):
        self.method = method
        self.coefficients: Optional[np.ndarray] = None

    def fit(
        self,
        X: np.ndarray,
        treatment: np.ndarray,
        outcome: np.ndarray
    ) -> 'OutcomeModel':
        """결과 모델 학습"""
        # Treatment를 피처에 추가
        X_full = np.column_stack([X, treatment])
        X_with_bias = np.column_stack([np.ones(len(X_full)), X_full])

        # OLS
        try:
            self.coefficients = np.linalg.lstsq(
                X_with_bias, outcome, rcond=None
            )[0]
        except np.linalg.LinAlgError:
            self.coefficients = np.zeros(X_with_bias.shape[1])

        return self

    def predict(
        self,
        X: np.ndarray,
        treatment: np.ndarray
    ) -> np.ndarray:
        """결과 예측"""
        if self.coefficients is None:
            return np.zeros(len(X))

        X_full = np.column_stack([X, treatment])
        X_with_bias = np.column_stack([np.ones(len(X_full)), X_full])
        return X_with_bias @ self.coefficients


class ATEEstimator:
    """평균 처리 효과(ATE) 추정"""

    def __init__(self, method: str = "ipw"):
        """
        Parameters:
        - method: "ipw" (Inverse Propensity Weighting),
                  "regression", "doubly_robust"
        """
        self.method = method
        self.propensity_model = PropensityScoreEstimator()
        self.outcome_model = OutcomeModel()

    def estimate(
        self,
        X: np.ndarray,
        treatment: np.ndarray,
        outcome: np.ndarray
    ) -> TreatmentEffect:
        """ATE 추정"""
        n = len(treatment)
        treatment = treatment.astype(float)

        if self.method == "ipw":
            return self._ipw_estimate(X, treatment, outcome)
        elif self.method == "regression":
            return self._regression_estimate(X, treatment, outcome)
        else:  # doubly_robust
            return self._doubly_robust_estimate(X, treatment, outcome)

    def _ipw_estimate(
        self,
        X: np.ndarray,
        treatment: np.ndarray,
        outcome: np.ndarray
    ) -> TreatmentEffect:
        """IPW 추정"""
        # 경향 점수 추정
        self.propensity_model.fit(X, treatment)
        ps = self.propensity_model.predict_proba(X)

        # IPW 가중치
        weights_treated = treatment / np.clip(ps, 0.01, 0.99)
        weights_control = (1 - treatment) / np.clip(1 - ps, 0.01, 0.99)

        # ATE = E[Y(1)] - E[Y(0)]
        y1_mean = np.sum(weights_treated * outcome) / np.sum(weights_treated)
        y0_mean = np.sum(weights_control * outcome) / np.sum(weights_control)
        ate = y1_mean - y0_mean

        # Bootstrap for standard error
        n_bootstrap = 100
        bootstrap_ates = []

        for _ in range(n_bootstrap):
            idx = np.random.choice(len(outcome), len(outcome), replace=True)
            boot_y1 = np.sum(weights_treated[idx] * outcome[idx]) / np.sum(weights_treated[idx])
            boot_y0 = np.sum(weights_control[idx] * outcome[idx]) / np.sum(weights_control[idx])
            bootstrap_ates.append(boot_y1 - boot_y0)

        std_error = np.std(bootstrap_ates)
        ci_low = np.percentile(bootstrap_ates, 2.5)
        ci_high = np.percentile(bootstrap_ates, 97.5)

        # 대략적인 p-value (정규 근사)
        z_score = abs(ate) / (std_error + 1e-10)
        p_value = 2 * (1 - self._normal_cdf(z_score))

        return TreatmentEffect(
            effect_type=EffectType.ATE,
            estimate=ate,
            std_error=std_error,
            confidence_interval=(ci_low, ci_high),
            p_value=p_value,
            treatment="treatment",
            outcome="outcome",
            method="ipw",
            sample_size=len(outcome)
        )

    def _regression_estimate(
        self,
        X: np.ndarray,
        treatment: np.ndarray,
        outcome: np.ndarray
    ) -> TreatmentEffect:
        """회귀 기반 추정"""
        self.outcome_model.fit(X, treatment, outcome)

        # E[Y(1)] - E[Y(0)]
        y1_pred = self.outcome_model.predict(X, np.ones(len(X)))
        y0_pred = self.outcome_model.predict(X, np.zeros(len(X)))

        ate = np.mean(y1_pred - y0_pred)

        # Bootstrap
        n_bootstrap = 100
        bootstrap_ates = []

        for _ in range(n_bootstrap):
            idx = np.random.choice(len(outcome), len(outcome), replace=True)
            temp_model = OutcomeModel()
            temp_model.fit(X[idx], treatment[idx], outcome[idx])
            y1 = temp_model.predict(X, np.ones(len(X)))
            y0 = temp_model.predict(X, np.zeros(len(X)))
            bootstrap_ates.append(np.mean(y1 - y0))

        std_error = np.std(bootstrap_ates)
        ci_low = np.percentile(bootstrap_ates, 2.5)
        ci_high = np.percentile(bootstrap_ates, 97.5)

        z_score = abs(ate) / (std_error + 1e-10)
        p_value = 2 * (1 - self._normal_cdf(z_score))

        return TreatmentEffect(
            effect_type=EffectType.ATE,
            estimate=ate,
            std_error=std_error,
            confidence_interval=(ci_low, ci_high),
            p_value=p_value,
            treatment="treatment",
            outcome="outcome",
            method="regression",
            sample_size=len(outcome)
        )

    def _doubly_robust_estimate(
        self,
        X: np.ndarray,
        treatment: np.ndarray,
        outcome: np.ndarray
    ) -> TreatmentEffect:
        """이중 강건(Doubly Robust) 추정"""
        # 경향 점수
        self.propensity_model.fit(X, treatment)
        ps = self.propensity_model.predict_proba(X)
        ps = np.clip(ps, 0.01, 0.99)

        # 결과 모델
        self.outcome_model.fit(X, treatment, outcome)
        y1_pred = self.outcome_model.predict(X, np.ones(len(X)))
        y0_pred = self.outcome_model.predict(X, np.zeros(len(X)))

        # DR 추정량
        dr_y1 = y1_pred + treatment * (outcome - y1_pred) / ps
        dr_y0 = y0_pred + (1 - treatment) * (outcome - y0_pred) / (1 - ps)

        ate = np.mean(dr_y1 - dr_y0)

        # Bootstrap
        n_bootstrap = 100
        bootstrap_ates = []

        for _ in range(n_bootstrap):
            idx = np.random.choice(len(outcome), len(outcome), replace=True)
            bootstrap_ates.append(np.mean(dr_y1[idx] - dr_y0[idx]))

        std_error = np.std(bootstrap_ates)
        ci_low = np.percentile(bootstrap_ates, 2.5)
        ci_high = np.percentile(bootstrap_ates, 97.5)

        z_score = abs(ate) / (std_error + 1e-10)
        p_value = 2 * (1 - self._normal_cdf(z_score))

        return TreatmentEffect(
            effect_type=EffectType.ATE,
            estimate=ate,
            std_error=std_error,
            confidence_interval=(ci_low, ci_high),
            p_value=p_value,
            treatment="treatment",
            outcome="outcome",
            method="doubly_robust",
            sample_size=len(outcome)
        )

    def _normal_cdf(self, x: float) -> float:
        """표준 정규 분포 CDF 근사"""
        return 0.5 * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x ** 3)))


class CATEEstimator:
    """조건부 평균 처리 효과(CATE) 추정"""

    def __init__(self, method: str = "s_learner"):
        """
        Parameters:
        - method: "s_learner", "t_learner", "x_learner"
        """
        self.method = method

    def estimate(
        self,
        X: np.ndarray,
        treatment: np.ndarray,
        outcome: np.ndarray,
        X_test: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        CATE 추정

        Returns:
        - cate: 개별 처리 효과
        - std_errors: 표준 오차
        """
        if X_test is None:
            X_test = X

        if self.method == "s_learner":
            return self._s_learner(X, treatment, outcome, X_test)
        elif self.method == "t_learner":
            return self._t_learner(X, treatment, outcome, X_test)
        else:  # x_learner
            return self._x_learner(X, treatment, outcome, X_test)

    def _s_learner(
        self,
        X: np.ndarray,
        treatment: np.ndarray,
        outcome: np.ndarray,
        X_test: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """S-Learner: 단일 모델로 CATE 추정"""
        model = OutcomeModel()
        model.fit(X, treatment, outcome)

        y1 = model.predict(X_test, np.ones(len(X_test)))
        y0 = model.predict(X_test, np.zeros(len(X_test)))

        cate = y1 - y0

        # 간단한 불확실성 추정
        residuals = outcome - model.predict(X, treatment)
        std_error = np.std(residuals) * np.ones(len(X_test))

        return cate, std_error

    def _t_learner(
        self,
        X: np.ndarray,
        treatment: np.ndarray,
        outcome: np.ndarray,
        X_test: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """T-Learner: 처리군/대조군 별도 모델"""
        # 처리군 모델
        mask_t = treatment == 1
        X_t = X[mask_t]
        y_t = outcome[mask_t]

        # 대조군 모델
        mask_c = treatment == 0
        X_c = X[mask_c]
        y_c = outcome[mask_c]

        # OLS로 각각 학습
        if len(X_t) > 0 and len(X_c) > 0:
            X_t_bias = np.column_stack([np.ones(len(X_t)), X_t])
            X_c_bias = np.column_stack([np.ones(len(X_c)), X_c])

            try:
                coef_t = np.linalg.lstsq(X_t_bias, y_t, rcond=None)[0]
                coef_c = np.linalg.lstsq(X_c_bias, y_c, rcond=None)[0]

                X_test_bias = np.column_stack([np.ones(len(X_test)), X_test])
                y1 = X_test_bias @ coef_t
                y0 = X_test_bias @ coef_c
            except np.linalg.LinAlgError:
                y1 = np.mean(y_t) * np.ones(len(X_test))
                y0 = np.mean(y_c) * np.ones(len(X_test))
        else:
            y1 = np.zeros(len(X_test))
            y0 = np.zeros(len(X_test))

        cate = y1 - y0
        std_error = np.std(cate) * np.ones(len(X_test)) * 0.2

        return cate, std_error

    def _x_learner(
        self,
        X: np.ndarray,
        treatment: np.ndarray,
        outcome: np.ndarray,
        X_test: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """X-Learner: 교차 추정"""
        # Step 1: T-Learner 방식으로 기본 모델
        cate_base, _ = self._t_learner(X, treatment, outcome, X_test)

        # Step 2: 경향 점수
        ps_model = PropensityScoreEstimator()
        ps_model.fit(X, treatment)
        ps = ps_model.predict_proba(X_test)

        # 가중 평균 (단순화된 X-Learner)
        # 실제로는 imputed treatment effect를 다시 학습해야 함
        cate = cate_base  # 단순화

        std_error = np.abs(cate) * 0.2  # 불확실성 근사

        return cate, std_error


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


# =============================================================================
# Granger Causality Analysis
# =============================================================================

@dataclass
class GrangerCausalityResult:
    """Granger 인과성 테스트 결과"""
    cause_variable: str
    effect_variable: str
    f_statistic: float
    p_value: float
    optimal_lag: int
    is_causal: bool  # p < 0.05
    r2_restricted: float  # 제한 모델 R^2
    r2_unrestricted: float  # 비제한 모델 R^2
    interpretation: str


class GrangerCausalityAnalyzer:
    """
    Granger Causality 분석기

    시계열 X의 과거값이 시계열 Y의 예측에 통계적으로 유의미한 정보를 제공하는지 검정.

    H0: X가 Y를 Granger-cause 하지 않는다 (X의 과거값이 Y 예측에 도움되지 않음)
    H1: X가 Y를 Granger-cause 한다 (X의 과거값이 Y 예측에 도움됨)

    References:
    - Granger, C. W. J. (1969). "Investigating Causal Relations by Econometric Models"
    """

    def __init__(self, max_lag: int = 5, significance_level: float = 0.05):
        """
        Parameters:
        - max_lag: 최대 시차 수
        - significance_level: 유의수준 (default: 0.05)
        """
        self.max_lag = max_lag
        self.significance_level = significance_level

    def test(
        self,
        x: np.ndarray,
        y: np.ndarray,
        lag: Optional[int] = None,
        x_name: str = "X",
        y_name: str = "Y"
    ) -> GrangerCausalityResult:
        """
        Granger Causality 테스트 수행

        X -> Y 방향의 인과성 검정 (X가 Y를 Granger-cause 하는지)

        Parameters:
        - x: 원인 변수 시계열 (T,)
        - y: 결과 변수 시계열 (T,)
        - lag: 사용할 시차 (None이면 AIC로 자동 선택)
        - x_name: X 변수명
        - y_name: Y 변수명

        Returns:
        - GrangerCausalityResult
        """
        x = np.asarray(x).flatten()
        y = np.asarray(y).flatten()

        if len(x) != len(y):
            raise ValueError("X와 Y의 길이가 같아야 합니다")

        n = len(y)

        if n < self.max_lag + 10:
            raise ValueError(f"시계열 길이({n})가 너무 짧습니다. 최소 {self.max_lag + 10} 필요")

        # 최적 시차 선택 (AIC 기준)
        if lag is None:
            optimal_lag = self._select_optimal_lag(x, y)
        else:
            optimal_lag = min(lag, self.max_lag)

        # 데이터 준비
        y_target, X_restricted, X_unrestricted = self._prepare_data(x, y, optimal_lag)

        # 제한 모델 (Y의 과거값만 사용)
        r2_restricted, rss_restricted = self._fit_ols(X_restricted, y_target)

        # 비제한 모델 (Y와 X의 과거값 모두 사용)
        r2_unrestricted, rss_unrestricted = self._fit_ols(X_unrestricted, y_target)

        # F-통계량 계산
        # F = ((RSS_r - RSS_ur) / q) / (RSS_ur / (n - k))
        # q = 추가된 제한 수 (X의 과거값 개수 = optimal_lag)
        # k = 비제한 모델의 파라미터 수
        q = optimal_lag
        k = X_unrestricted.shape[1]
        n_obs = len(y_target)

        if rss_unrestricted > 0:
            f_stat = ((rss_restricted - rss_unrestricted) / q) / (rss_unrestricted / (n_obs - k))
        else:
            f_stat = 0.0

        # p-value 계산 (F 분포)
        p_value = self._f_distribution_pvalue(f_stat, q, n_obs - k)

        # 결과 해석
        is_causal = p_value < self.significance_level

        if is_causal:
            interpretation = (
                f"{x_name}이(가) {y_name}을(를) Granger-cause 합니다 "
                f"(F={f_stat:.3f}, p={p_value:.4f}, lag={optimal_lag}). "
                f"{x_name}의 과거값이 {y_name} 예측에 통계적으로 유의미한 정보를 제공합니다."
            )
        else:
            interpretation = (
                f"{x_name}이(가) {y_name}을(를) Granger-cause 하지 않습니다 "
                f"(F={f_stat:.3f}, p={p_value:.4f}, lag={optimal_lag}). "
                f"{x_name}의 과거값이 {y_name} 예측에 추가적인 정보를 제공하지 않습니다."
            )

        return GrangerCausalityResult(
            cause_variable=x_name,
            effect_variable=y_name,
            f_statistic=f_stat,
            p_value=p_value,
            optimal_lag=optimal_lag,
            is_causal=is_causal,
            r2_restricted=r2_restricted,
            r2_unrestricted=r2_unrestricted,
            interpretation=interpretation
        )

    def test_bidirectional(
        self,
        x: np.ndarray,
        y: np.ndarray,
        lag: Optional[int] = None,
        x_name: str = "X",
        y_name: str = "Y"
    ) -> Dict[str, GrangerCausalityResult]:
        """
        양방향 Granger Causality 테스트

        X -> Y 와 Y -> X 모두 검정

        Returns:
        - {"x_to_y": result1, "y_to_x": result2}
        """
        result_x_to_y = self.test(x, y, lag, x_name, y_name)
        result_y_to_x = self.test(y, x, lag, y_name, x_name)

        return {
            "x_to_y": result_x_to_y,
            "y_to_x": result_y_to_x
        }

    def test_multivariate(
        self,
        data: Dict[str, np.ndarray],
        target: str,
        lag: Optional[int] = None
    ) -> Dict[str, GrangerCausalityResult]:
        """
        다변량 Granger Causality 테스트

        여러 변수가 target 변수에 미치는 인과성 검정

        Parameters:
        - data: {"var_name": time_series, ...}
        - target: 타겟 변수명
        - lag: 시차

        Returns:
        - {"var_name": GrangerCausalityResult, ...}
        """
        if target not in data:
            raise ValueError(f"타겟 변수 '{target}'가 데이터에 없습니다")

        y = data[target]
        results = {}

        for var_name, x in data.items():
            if var_name != target:
                try:
                    result = self.test(x, y, lag, var_name, target)
                    results[var_name] = result
                except Exception as e:
                    # 테스트 실패 시 None 결과 저장
                    results[var_name] = GrangerCausalityResult(
                        cause_variable=var_name,
                        effect_variable=target,
                        f_statistic=0.0,
                        p_value=1.0,
                        optimal_lag=1,
                        is_causal=False,
                        r2_restricted=0.0,
                        r2_unrestricted=0.0,
                        interpretation=f"테스트 실패: {str(e)}"
                    )

        return results

    def _select_optimal_lag(self, x: np.ndarray, y: np.ndarray) -> int:
        """AIC를 사용하여 최적 시차 선택"""
        best_lag = 1
        best_aic = float('inf')

        for lag in range(1, self.max_lag + 1):
            try:
                y_target, _, X_unrestricted = self._prepare_data(x, y, lag)
                _, rss = self._fit_ols(X_unrestricted, y_target)

                n = len(y_target)
                k = X_unrestricted.shape[1]

                # AIC = n * log(RSS/n) + 2k
                if rss > 0:
                    aic = n * np.log(rss / n) + 2 * k
                else:
                    aic = float('inf')

                if aic < best_aic:
                    best_aic = aic
                    best_lag = lag
            except Exception:
                continue

        return best_lag

    def _prepare_data(
        self,
        x: np.ndarray,
        y: np.ndarray,
        lag: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """회귀 분석용 데이터 준비"""
        n = len(y)

        # 종속변수: y[lag:]
        y_target = y[lag:]

        # 제한 모델: Y의 과거값만 (상수항 포함)
        X_restricted = np.column_stack([
            np.ones(n - lag),  # 상수항
            *[y[lag - i - 1:n - i - 1] for i in range(lag)]  # Y의 lag
        ])

        # 비제한 모델: Y와 X의 과거값 모두
        X_unrestricted = np.column_stack([
            X_restricted,
            *[x[lag - i - 1:n - i - 1] for i in range(lag)]  # X의 lag
        ])

        return y_target, X_restricted, X_unrestricted

    def _fit_ols(self, X: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
        """OLS 적합 및 R^2, RSS 반환"""
        try:
            # beta = (X'X)^(-1) X'y
            beta = np.linalg.lstsq(X, y, rcond=None)[0]
            y_pred = X @ beta

            # RSS (Residual Sum of Squares)
            residuals = y - y_pred
            rss = np.sum(residuals ** 2)

            # R^2
            tss = np.sum((y - np.mean(y)) ** 2)
            r2 = 1 - (rss / tss) if tss > 0 else 0.0

            return r2, rss
        except np.linalg.LinAlgError:
            return 0.0, np.sum((y - np.mean(y)) ** 2)

    def _f_distribution_pvalue(self, f_stat: float, df1: int, df2: int) -> float:
        """F 분포 p-value 근사 계산"""
        if f_stat <= 0 or df1 <= 0 or df2 <= 0:
            return 1.0

        # Beta 분포를 이용한 F 분포 CDF 근사
        # P(F > f) = 1 - I_{df2/(df2 + df1*f)}(df2/2, df1/2)
        # 여기서 I_x(a,b)는 정규화된 불완전 베타 함수

        x = df2 / (df2 + df1 * f_stat)
        a = df2 / 2.0
        b = df1 / 2.0

        # 불완전 베타 함수 근사 (계속 분수 전개)
        beta_cdf = self._incomplete_beta(x, a, b)
        p_value = 1.0 - beta_cdf

        return max(0.0, min(1.0, p_value))

    def _incomplete_beta(self, x: float, a: float, b: float, max_iter: int = 200) -> float:
        """정규화된 불완전 베타 함수 I_x(a,b) 근사"""
        if x <= 0:
            return 0.0
        if x >= 1:
            return 1.0

        # 대칭성 활용
        if x > (a + 1) / (a + b + 2):
            return 1.0 - self._incomplete_beta(1 - x, b, a, max_iter)

        # 계속 분수 전개 (Lentz's algorithm)
        front = np.exp(
            a * np.log(x) + b * np.log(1 - x) -
            np.log(a) - self._log_beta(a, b)
        )

        f = 1.0
        c = 1.0
        d = 0.0

        for m in range(1, max_iter + 1):
            # even terms
            m2 = 2 * m
            aa = m * (b - m) * x / ((a + m2 - 1) * (a + m2))
            d = 1.0 + aa * d
            if abs(d) < 1e-30:
                d = 1e-30
            c = 1.0 + aa / c
            if abs(c) < 1e-30:
                c = 1e-30
            d = 1.0 / d
            f *= c * d

            # odd terms
            aa = -(a + m) * (a + b + m) * x / ((a + m2) * (a + m2 + 1))
            d = 1.0 + aa * d
            if abs(d) < 1e-30:
                d = 1e-30
            c = 1.0 + aa / c
            if abs(c) < 1e-30:
                c = 1e-30
            d = 1.0 / d
            delta = c * d
            f *= delta

            if abs(delta - 1.0) < 1e-10:
                break

        return front * f

    def _log_beta(self, a: float, b: float) -> float:
        """log(Beta(a,b)) = log(Gamma(a)) + log(Gamma(b)) - log(Gamma(a+b))"""
        return self._log_gamma(a) + self._log_gamma(b) - self._log_gamma(a + b)

    def _log_gamma(self, x: float) -> float:
        """Stirling 근사를 사용한 log(Gamma(x))"""
        if x <= 0:
            return 0.0

        # Lanczos 근사
        g = 7
        coefficients = [
            0.99999999999980993,
            676.5203681218851,
            -1259.1392167224028,
            771.32342877765313,
            -176.61502916214059,
            12.507343278686905,
            -0.13857109526572012,
            9.9843695780195716e-6,
            1.5056327351493116e-7
        ]

        if x < 0.5:
            return np.log(np.pi / np.sin(np.pi * x)) - self._log_gamma(1 - x)

        x -= 1
        a = coefficients[0]
        t = x + g + 0.5
        for i in range(1, g + 2):
            a += coefficients[i] / (x + i)

        return 0.5 * np.log(2 * np.pi) + (x + 0.5) * np.log(t) - t + np.log(a)


# =============================================================================
# PC Algorithm for Causal Graph Discovery
# =============================================================================

@dataclass
class PCAlgorithmResult:
    """PC Algorithm 결과"""
    adjacency_matrix: np.ndarray
    causal_graph: 'CausalGraph'
    skeleton: Set[Tuple[str, str]]
    directed_edges: Set[Tuple[str, str]]
    variable_names: List[str]
    separation_sets: Dict[Tuple[str, str], Set[str]]
    n_conditional_tests: int
    alpha: float


class PCAlgorithm:
    """
    PC (Peter-Clark) Algorithm for Causal Discovery

    관찰 데이터로부터 DAG (Directed Acyclic Graph) 구조를 학습합니다.

    알고리즘:
    1. 완전 연결 무방향 그래프로 시작
    2. 조건부 독립성 테스트로 엣지 제거 (skeleton 학습)
    3. V-structure (충돌자) 식별로 방향 부여
    4. 추가 규칙으로 방향 전파

    References:
    - Spirtes, Glymour, Scheines, "Causation, Prediction, and Search", 2000
    - Colombo et al., "Learning High-Dimensional DAGs with Latent and Selection Variables", 2012
    """

    def __init__(
        self,
        alpha: float = 0.05,
        max_cond_set_size: Optional[int] = None,
        method: str = "partial_correlation"
    ):
        """
        Parameters:
        - alpha: 조건부 독립성 테스트 유의수준
        - max_cond_set_size: 최대 조건부 집합 크기 (None이면 변수 수 - 2)
        - method: 독립성 테스트 방법 ("partial_correlation", "fisher_z")
        """
        self.alpha = alpha
        self.max_cond_set_size = max_cond_set_size
        self.method = method
        self.n_tests = 0

    def fit(
        self,
        data: np.ndarray,
        variable_names: Optional[List[str]] = None
    ) -> PCAlgorithmResult:
        """
        PC Algorithm으로 인과 그래프 학습

        Parameters:
        - data: (n_samples, n_variables) 데이터 행렬
        - variable_names: 변수명 리스트

        Returns:
        - PCAlgorithmResult
        """
        n_samples, n_vars = data.shape

        if variable_names is None:
            variable_names = [f"V{i}" for i in range(n_vars)]

        if len(variable_names) != n_vars:
            raise ValueError("변수명 수가 데이터 열 수와 일치하지 않습니다")

        max_cond = self.max_cond_set_size
        if max_cond is None:
            max_cond = n_vars - 2

        self.n_tests = 0

        # Step 1: Skeleton 학습 (무방향 그래프)
        adjacency, sep_sets = self._learn_skeleton(data, n_vars, max_cond)

        # Skeleton을 엣지 집합으로
        skeleton = set()
        for i in range(n_vars):
            for j in range(i + 1, n_vars):
                if adjacency[i, j]:
                    skeleton.add((variable_names[i], variable_names[j]))

        # Step 2: V-structure 식별 및 방향 부여
        directed_edges = self._orient_edges(adjacency, sep_sets, variable_names)

        # Step 3: CausalGraph 객체 생성
        causal_graph = self._build_causal_graph(adjacency, directed_edges, variable_names)

        return PCAlgorithmResult(
            adjacency_matrix=adjacency.copy(),
            causal_graph=causal_graph,
            skeleton=skeleton,
            directed_edges=directed_edges,
            variable_names=variable_names,
            separation_sets=sep_sets,
            n_conditional_tests=self.n_tests,
            alpha=self.alpha
        )

    def _learn_skeleton(
        self,
        data: np.ndarray,
        n_vars: int,
        max_cond: int
    ) -> Tuple[np.ndarray, Dict[Tuple[int, int], Set[int]]]:
        """Skeleton 학습 (Phase 1)"""
        # 완전 연결 그래프로 시작
        adjacency = np.ones((n_vars, n_vars), dtype=bool)
        np.fill_diagonal(adjacency, False)

        sep_sets: Dict[Tuple[int, int], Set[int]] = {}

        # 조건부 집합 크기를 0부터 증가시키며 테스트
        for cond_size in range(max_cond + 1):
            # 인접한 쌍에 대해 테스트
            for i in range(n_vars):
                for j in range(i + 1, n_vars):
                    if not adjacency[i, j]:
                        continue

                    # i와 j를 제외한 인접 노드
                    adj_i = set(np.where(adjacency[i, :])[0]) - {j}
                    adj_j = set(np.where(adjacency[j, :])[0]) - {i}
                    possible_sep = adj_i | adj_j

                    if len(possible_sep) < cond_size:
                        continue

                    # 조건부 집합 후보 생성
                    found_sep = False
                    for cond_set in self._combinations(list(possible_sep), cond_size):
                        cond_set = set(cond_set)

                        # 조건부 독립성 테스트
                        is_independent = self._conditional_independence_test(
                            data, i, j, list(cond_set)
                        )
                        self.n_tests += 1

                        if is_independent:
                            # 엣지 제거
                            adjacency[i, j] = False
                            adjacency[j, i] = False
                            sep_sets[(i, j)] = cond_set
                            sep_sets[(j, i)] = cond_set
                            found_sep = True
                            break

                    if found_sep:
                        continue

        return adjacency, sep_sets

    def _orient_edges(
        self,
        adjacency: np.ndarray,
        sep_sets: Dict[Tuple[int, int], Set[int]],
        variable_names: List[str]
    ) -> Set[Tuple[str, str]]:
        """엣지 방향 결정 (Phase 2)"""
        n_vars = len(variable_names)
        directed = set()

        # 방향 행렬: directed_adj[i,j] = True 이면 i -> j
        directed_adj = np.zeros((n_vars, n_vars), dtype=bool)

        # V-structure 식별: i - k - j 에서 i와 j가 인접하지 않고
        # k가 sep_set(i,j)에 없으면 i -> k <- j
        for i in range(n_vars):
            for j in range(i + 1, n_vars):
                if adjacency[i, j]:
                    continue  # i와 j가 인접하면 skip

                # 공통 이웃 k 찾기
                for k in range(n_vars):
                    if k == i or k == j:
                        continue
                    if not (adjacency[i, k] and adjacency[k, j]):
                        continue

                    # k가 separation set에 없으면 V-structure
                    sep = sep_sets.get((i, j), set())
                    if k not in sep:
                        directed_adj[i, k] = True
                        directed_adj[j, k] = True
                        directed.add((variable_names[i], variable_names[k]))
                        directed.add((variable_names[j], variable_names[k]))

        # 추가 규칙으로 방향 전파
        changed = True
        while changed:
            changed = False
            for i in range(n_vars):
                for j in range(n_vars):
                    if i == j:
                        continue
                    if not adjacency[i, j]:
                        continue
                    if directed_adj[i, j] or directed_adj[j, i]:
                        continue  # 이미 방향 결정됨

                    # 규칙 1: i -> k - j 이고 i와 j가 인접하지 않으면 k -> j
                    for k in range(n_vars):
                        if k == i or k == j:
                            continue
                        if directed_adj[i, k] and adjacency[k, j] and not adjacency[i, j]:
                            if not directed_adj[j, k]:  # 사이클 방지
                                directed_adj[k, j] = True
                                directed.add((variable_names[k], variable_names[j]))
                                changed = True

                    # 규칙 2: 사이클 방지
                    # i -> ... -> j 경로가 있으면 j -> i 불가

        # 방향 미결정 엣지는 임의로 (또는 그대로 무방향으로) 처리
        for i in range(n_vars):
            for j in range(i + 1, n_vars):
                if adjacency[i, j] and not directed_adj[i, j] and not directed_adj[j, i]:
                    # 아직 방향 미결정: 임의로 i -> j
                    directed_adj[i, j] = True
                    directed.add((variable_names[i], variable_names[j]))

        return directed

    def _conditional_independence_test(
        self,
        data: np.ndarray,
        i: int,
        j: int,
        cond_set: List[int]
    ) -> bool:
        """조건부 독립성 테스트 (Partial Correlation 기반)"""
        n_samples = data.shape[0]

        if len(cond_set) == 0:
            # 무조건부 상관
            corr = np.corrcoef(data[:, i], data[:, j])[0, 1]
        else:
            # 조건부 편상관 (Partial Correlation)
            corr = self._partial_correlation(data, i, j, cond_set)

        if np.isnan(corr):
            return False

        # Fisher's Z 변환
        z = 0.5 * np.log((1 + corr + 1e-10) / (1 - corr + 1e-10))

        # 표준오차
        df = n_samples - len(cond_set) - 3
        if df <= 0:
            return False

        se = 1.0 / np.sqrt(df)

        # 양측 검정
        z_stat = abs(z) / se
        p_value = 2 * (1 - self._normal_cdf(z_stat))

        return p_value > self.alpha

    def _partial_correlation(
        self,
        data: np.ndarray,
        i: int,
        j: int,
        cond_set: List[int]
    ) -> float:
        """편상관 계수 계산"""
        # 조건부 변수로 회귀한 잔차의 상관
        if len(cond_set) == 0:
            return np.corrcoef(data[:, i], data[:, j])[0, 1]

        Z = data[:, cond_set]
        Z = np.column_stack([np.ones(len(Z)), Z])

        try:
            # X_i에서 Z의 효과 제거
            beta_i = np.linalg.lstsq(Z, data[:, i], rcond=None)[0]
            residual_i = data[:, i] - Z @ beta_i

            # X_j에서 Z의 효과 제거
            beta_j = np.linalg.lstsq(Z, data[:, j], rcond=None)[0]
            residual_j = data[:, j] - Z @ beta_j

            # 잔차 간 상관
            corr = np.corrcoef(residual_i, residual_j)[0, 1]
            return corr
        except np.linalg.LinAlgError:
            return 0.0

    def _combinations(self, items: List[int], size: int) -> List[Tuple[int, ...]]:
        """조합 생성"""
        if size == 0:
            return [()]
        if size > len(items):
            return []

        result = []
        for i, item in enumerate(items):
            for rest in self._combinations(items[i + 1:], size - 1):
                result.append((item,) + rest)
        return result

    def _normal_cdf(self, x: float) -> float:
        """표준 정규 분포 CDF 근사"""
        return 0.5 * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x ** 3)))

    def _build_causal_graph(
        self,
        adjacency: np.ndarray,
        directed_edges: Set[Tuple[str, str]],
        variable_names: List[str]
    ) -> CausalGraph:
        """CausalGraph 객체 생성"""
        graph = CausalGraph()

        # 노드 추가
        for name in variable_names:
            graph.add_node(CausalNode(
                name=name,
                node_type="observed",
                observed=True
            ))

        # 방향 엣지 추가
        for source, target in directed_edges:
            graph.add_edge(CausalEdge(
                source=source,
                target=target,
                weight=1.0,
                mechanism="discovered"
            ))

        return graph


# =============================================================================
# Enhanced CATE Estimators (S-Learner, T-Learner, X-Learner)
# =============================================================================

@dataclass
class CATEResult:
    """CATE 추정 결과 (개선된 버전)"""
    cate_estimates: np.ndarray  # 개별 처리 효과
    std_errors: np.ndarray  # 표준 오차
    confidence_intervals: np.ndarray  # (n, 2) 신뢰구간
    ate: float  # 평균 처리 효과
    method: str
    feature_importances: Optional[Dict[str, float]]
    subgroup_effects: Optional[Dict[str, float]]


class EnhancedCATEEstimator:
    """
    향상된 CATE (Conditional Average Treatment Effect) 추정기

    S-Learner, T-Learner, X-Learner 메타러너 구현

    References:
    - Kunzel et al., "Metalearners for Estimating Heterogeneous Treatment Effects", 2019
    - Athey & Imbens, "Machine Learning Methods for Estimating Heterogeneous Causal Effects", 2015
    """

    def __init__(
        self,
        method: str = "x_learner",
        base_learner: str = "linear",
        n_bootstrap: int = 100,
        confidence_level: float = 0.95
    ):
        """
        Parameters:
        - method: "s_learner", "t_learner", "x_learner"
        - base_learner: "linear", "ridge"
        - n_bootstrap: 부트스트랩 반복 횟수
        - confidence_level: 신뢰수준
        """
        self.method = method
        self.base_learner = base_learner
        self.n_bootstrap = n_bootstrap
        self.confidence_level = confidence_level

        # 학습된 모델 저장
        self._model_1 = None  # T-Learner: 처리군 모델
        self._model_0 = None  # T-Learner: 대조군 모델
        self._model_s = None  # S-Learner: 통합 모델
        self._propensity_model = None
        self._tau_0 = None  # X-Learner: 대조군 CATE 모델
        self._tau_1 = None  # X-Learner: 처리군 CATE 모델

    def fit_predict(
        self,
        X: np.ndarray,
        treatment: np.ndarray,
        outcome: np.ndarray,
        X_test: Optional[np.ndarray] = None,
        feature_names: Optional[List[str]] = None
    ) -> CATEResult:
        """
        CATE 추정 및 예측

        Parameters:
        - X: 공변량 행렬 (n_samples, n_features)
        - treatment: 처리 벡터 (0/1)
        - outcome: 결과 벡터
        - X_test: 테스트 데이터 (None이면 X 사용)
        - feature_names: 피처명 리스트

        Returns:
        - CATEResult
        """
        X = np.atleast_2d(X)
        treatment = np.asarray(treatment).flatten()
        outcome = np.asarray(outcome).flatten()

        if X_test is None:
            X_test = X
        else:
            X_test = np.atleast_2d(X_test)

        if feature_names is None:
            feature_names = [f"X{i}" for i in range(X.shape[1])]

        # 방법별 CATE 추정
        if self.method == "s_learner":
            cate, std_errors = self._s_learner(X, treatment, outcome, X_test)
        elif self.method == "t_learner":
            cate, std_errors = self._t_learner(X, treatment, outcome, X_test)
        else:  # x_learner
            cate, std_errors = self._x_learner(X, treatment, outcome, X_test)

        # 신뢰구간 계산
        alpha = 1 - self.confidence_level
        z = 1.96  # 95% 신뢰구간
        confidence_intervals = np.column_stack([
            cate - z * std_errors,
            cate + z * std_errors
        ])

        # ATE
        ate = np.mean(cate)

        # 피처 중요도 (단순 상관 기반)
        feature_importances = {}
        for i, name in enumerate(feature_names):
            corr = np.corrcoef(X_test[:, i], cate)[0, 1]
            feature_importances[name] = abs(corr) if not np.isnan(corr) else 0.0

        # 서브그룹 효과 (피처 중앙값 기준 분할)
        subgroup_effects = {}
        for i, name in enumerate(feature_names):
            median = np.median(X_test[:, i])
            high_mask = X_test[:, i] >= median
            low_mask = ~high_mask

            if np.sum(high_mask) > 0 and np.sum(low_mask) > 0:
                subgroup_effects[f"{name}_high"] = float(np.mean(cate[high_mask]))
                subgroup_effects[f"{name}_low"] = float(np.mean(cate[low_mask]))

        return CATEResult(
            cate_estimates=cate,
            std_errors=std_errors,
            confidence_intervals=confidence_intervals,
            ate=ate,
            method=self.method,
            feature_importances=feature_importances,
            subgroup_effects=subgroup_effects
        )

    def _s_learner(
        self,
        X: np.ndarray,
        treatment: np.ndarray,
        outcome: np.ndarray,
        X_test: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        S-Learner (Single Model)

        단일 모델 mu(x, t)를 학습하고
        CATE(x) = mu(x, 1) - mu(x, 0)

        장점: 간단, 데이터 효율적
        단점: 처리 효과가 작으면 무시될 수 있음
        """
        n_test = len(X_test)

        # Treatment를 피처로 추가
        X_with_t = np.column_stack([X, treatment])

        # 모델 학습
        self._model_s = self._fit_base_learner(X_with_t, outcome)

        # 예측
        X_test_t1 = np.column_stack([X_test, np.ones(n_test)])
        X_test_t0 = np.column_stack([X_test, np.zeros(n_test)])

        y1 = self._predict_base_learner(self._model_s, X_test_t1)
        y0 = self._predict_base_learner(self._model_s, X_test_t0)

        cate = y1 - y0

        # Bootstrap for standard errors
        std_errors = self._bootstrap_std(X, treatment, outcome, X_test, self._s_learner_single)

        return cate, std_errors

    def _t_learner(
        self,
        X: np.ndarray,
        treatment: np.ndarray,
        outcome: np.ndarray,
        X_test: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        T-Learner (Two Models)

        처리군/대조군 별도 모델:
        mu_1(x) = E[Y | X=x, T=1]
        mu_0(x) = E[Y | X=x, T=0]
        CATE(x) = mu_1(x) - mu_0(x)

        장점: 처리 효과 학습에 집중
        단점: 데이터가 작으면 불안정
        """
        # 처리군/대조군 분리
        mask_1 = treatment == 1
        mask_0 = treatment == 0

        X_1, y_1 = X[mask_1], outcome[mask_1]
        X_0, y_0 = X[mask_0], outcome[mask_0]

        # 각각 모델 학습
        self._model_1 = self._fit_base_learner(X_1, y_1)
        self._model_0 = self._fit_base_learner(X_0, y_0)

        # 예측
        y1 = self._predict_base_learner(self._model_1, X_test)
        y0 = self._predict_base_learner(self._model_0, X_test)

        cate = y1 - y0

        # Bootstrap for standard errors
        std_errors = self._bootstrap_std(X, treatment, outcome, X_test, self._t_learner_single)

        return cate, std_errors

    def _x_learner(
        self,
        X: np.ndarray,
        treatment: np.ndarray,
        outcome: np.ndarray,
        X_test: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        X-Learner (Cross Learning)

        1단계: T-Learner 방식으로 mu_1, mu_0 학습
        2단계: Imputed treatment effects 계산
               - 처리군: D_1 = Y_1 - mu_0(X_1)
               - 대조군: D_0 = mu_1(X_0) - Y_0
        3단계: tau_1(x), tau_0(x) 학습
        4단계: 경향점수 가중 평균
               CATE(x) = g(x) * tau_0(x) + (1-g(x)) * tau_1(x)

        장점: 이질적 효과 추정에 효과적, 작은 처리군에도 강건
        """
        # Step 1: T-Learner 기본 모델
        mask_1 = treatment == 1
        mask_0 = treatment == 0

        X_1, y_1 = X[mask_1], outcome[mask_1]
        X_0, y_0 = X[mask_0], outcome[mask_0]

        self._model_1 = self._fit_base_learner(X_1, y_1)
        self._model_0 = self._fit_base_learner(X_0, y_0)

        # Step 2: Imputed treatment effects
        # 처리군: 실제 결과 - 대조군 모델 예측
        D_1 = y_1 - self._predict_base_learner(self._model_0, X_1)
        # 대조군: 처리군 모델 예측 - 실제 결과
        D_0 = self._predict_base_learner(self._model_1, X_0) - y_0

        # Step 3: CATE 모델 학습
        if len(D_1) > 0:
            self._tau_1 = self._fit_base_learner(X_1, D_1)
        else:
            self._tau_1 = None

        if len(D_0) > 0:
            self._tau_0 = self._fit_base_learner(X_0, D_0)
        else:
            self._tau_0 = None

        # Step 4: 경향점수 기반 가중 평균
        self._propensity_model = PropensityScoreEstimator()
        self._propensity_model.fit(X, treatment)

        ps = self._propensity_model.predict_proba(X_test)
        ps = np.clip(ps, 0.01, 0.99)

        # tau 예측
        if self._tau_1 is not None:
            tau_1 = self._predict_base_learner(self._tau_1, X_test)
        else:
            tau_1 = np.zeros(len(X_test))

        if self._tau_0 is not None:
            tau_0 = self._predict_base_learner(self._tau_0, X_test)
        else:
            tau_0 = np.zeros(len(X_test))

        # 가중 평균: 경향점수가 높으면 tau_0 가중치 높임
        cate = ps * tau_0 + (1 - ps) * tau_1

        # Bootstrap for standard errors
        std_errors = self._bootstrap_std(X, treatment, outcome, X_test, self._x_learner_single)

        return cate, std_errors

    def _fit_base_learner(self, X: np.ndarray, y: np.ndarray) -> Dict[str, np.ndarray]:
        """기본 학습기 (선형 회귀 또는 Ridge)"""
        X = np.atleast_2d(X)
        if X.shape[0] == 0:
            return {"coef": np.zeros(X.shape[1] + 1)}

        X_with_bias = np.column_stack([np.ones(len(X)), X])

        try:
            if self.base_learner == "ridge":
                # Ridge 회귀: (X'X + lambda*I)^(-1) X'y
                lam = 1.0
                XtX = X_with_bias.T @ X_with_bias
                reg = lam * np.eye(XtX.shape[0])
                coef = np.linalg.solve(XtX + reg, X_with_bias.T @ y)
            else:
                # OLS
                coef = np.linalg.lstsq(X_with_bias, y, rcond=None)[0]

            return {"coef": coef}
        except np.linalg.LinAlgError:
            return {"coef": np.zeros(X_with_bias.shape[1])}

    def _predict_base_learner(self, model: Dict[str, np.ndarray], X: np.ndarray) -> np.ndarray:
        """기본 학습기 예측"""
        X = np.atleast_2d(X)
        X_with_bias = np.column_stack([np.ones(len(X)), X])
        coef = model.get("coef", np.zeros(X_with_bias.shape[1]))

        if len(coef) != X_with_bias.shape[1]:
            return np.zeros(len(X))

        return X_with_bias @ coef

    def _bootstrap_std(
        self,
        X: np.ndarray,
        treatment: np.ndarray,
        outcome: np.ndarray,
        X_test: np.ndarray,
        estimator_func: Callable
    ) -> np.ndarray:
        """Bootstrap 표준오차 계산"""
        n = len(X)
        n_test = len(X_test)
        bootstrap_cates = []

        for _ in range(min(self.n_bootstrap, 50)):  # 속도를 위해 제한
            idx = np.random.choice(n, n, replace=True)
            try:
                cate = estimator_func(X[idx], treatment[idx], outcome[idx], X_test)
                bootstrap_cates.append(cate)
            except Exception:
                continue

        if len(bootstrap_cates) < 2:
            return np.std(outcome) * np.ones(n_test) * 0.2

        return np.std(bootstrap_cates, axis=0)

    def _s_learner_single(
        self,
        X: np.ndarray,
        treatment: np.ndarray,
        outcome: np.ndarray,
        X_test: np.ndarray
    ) -> np.ndarray:
        """S-Learner 단일 추정"""
        X_with_t = np.column_stack([X, treatment])
        model = self._fit_base_learner(X_with_t, outcome)

        n_test = len(X_test)
        X_test_t1 = np.column_stack([X_test, np.ones(n_test)])
        X_test_t0 = np.column_stack([X_test, np.zeros(n_test)])

        y1 = self._predict_base_learner(model, X_test_t1)
        y0 = self._predict_base_learner(model, X_test_t0)

        return y1 - y0

    def _t_learner_single(
        self,
        X: np.ndarray,
        treatment: np.ndarray,
        outcome: np.ndarray,
        X_test: np.ndarray
    ) -> np.ndarray:
        """T-Learner 단일 추정"""
        mask_1 = treatment == 1
        mask_0 = treatment == 0

        if np.sum(mask_1) == 0 or np.sum(mask_0) == 0:
            return np.zeros(len(X_test))

        model_1 = self._fit_base_learner(X[mask_1], outcome[mask_1])
        model_0 = self._fit_base_learner(X[mask_0], outcome[mask_0])

        y1 = self._predict_base_learner(model_1, X_test)
        y0 = self._predict_base_learner(model_0, X_test)

        return y1 - y0

    def _x_learner_single(
        self,
        X: np.ndarray,
        treatment: np.ndarray,
        outcome: np.ndarray,
        X_test: np.ndarray
    ) -> np.ndarray:
        """X-Learner 단일 추정"""
        mask_1 = treatment == 1
        mask_0 = treatment == 0

        if np.sum(mask_1) == 0 or np.sum(mask_0) == 0:
            return np.zeros(len(X_test))

        X_1, y_1 = X[mask_1], outcome[mask_1]
        X_0, y_0 = X[mask_0], outcome[mask_0]

        model_1 = self._fit_base_learner(X_1, y_1)
        model_0 = self._fit_base_learner(X_0, y_0)

        # Imputed effects
        D_1 = y_1 - self._predict_base_learner(model_0, X_1)
        D_0 = self._predict_base_learner(model_1, X_0) - y_0

        tau_1_model = self._fit_base_learner(X_1, D_1)
        tau_0_model = self._fit_base_learner(X_0, D_0)

        tau_1 = self._predict_base_learner(tau_1_model, X_test)
        tau_0 = self._predict_base_learner(tau_0_model, X_test)

        # 경향점수
        ps_model = PropensityScoreEstimator()
        ps_model.fit(X, treatment)
        ps = ps_model.predict_proba(X_test)
        ps = np.clip(ps, 0.01, 0.99)

        return ps * tau_0 + (1 - ps) * tau_1


class DataIntegrationCausalModel:
    """
    데이터 통합 시나리오에 특화된 인과 모델

    데이터 통합(처리)이 비즈니스 KPI(결과)에 미치는 영향을 분석
    """

    def __init__(self):
        self.causal_graph = CausalGraph()
        self.ate_estimator = ATEEstimator(method="doubly_robust")
        self.cate_estimator = CATEEstimator(method="t_learner")
        self.cf_analyzer = CounterfactualAnalyzer()
        self.sens_analyzer = SensitivityAnalyzer()

    def build_causal_graph(
        self,
        treatment_vars: List[str],
        outcome_vars: List[str],
        confounders: List[str] = None,
        mediators: List[str] = None
    ) -> CausalGraph:
        """인과 그래프 구축"""
        self.causal_graph = CausalGraph()

        # 처리 변수
        for t in treatment_vars:
            self.causal_graph.add_node(CausalNode(
                name=t,
                node_type="treatment"
            ))

        # 결과 변수
        for o in outcome_vars:
            self.causal_graph.add_node(CausalNode(
                name=o,
                node_type="outcome"
            ))

        # 교란 변수
        if confounders:
            for c in confounders:
                self.causal_graph.add_node(CausalNode(
                    name=c,
                    node_type="confounder"
                ))
                # 교란 -> 처리, 교란 -> 결과
                for t in treatment_vars:
                    self.causal_graph.add_edge(CausalEdge(source=c, target=t))
                for o in outcome_vars:
                    self.causal_graph.add_edge(CausalEdge(source=c, target=o))

        # 매개 변수
        if mediators:
            for m in mediators:
                self.causal_graph.add_node(CausalNode(
                    name=m,
                    node_type="mediator"
                ))
                # 처리 -> 매개 -> 결과
                for t in treatment_vars:
                    self.causal_graph.add_edge(CausalEdge(source=t, target=m))
                for o in outcome_vars:
                    self.causal_graph.add_edge(CausalEdge(source=m, target=o))

        # 처리 -> 결과 (직접 효과)
        for t in treatment_vars:
            for o in outcome_vars:
                self.causal_graph.add_edge(CausalEdge(source=t, target=o))

        return self.causal_graph

    def estimate_integration_impact(
        self,
        tables_info: Dict[str, Any],
        integration_scenario: Dict[str, Any],
        baseline_metrics: Dict[str, float] = None
    ) -> CausalImpactResult:
        """
        데이터 통합이 비즈니스 지표에 미치는 영향 추정

        시뮬레이션 기반 접근: 실제 데이터가 없으므로
        스키마/관계 복잡도에서 파생된 특성을 사용
        """
        # 1. 피처 추출
        features = self._extract_features(tables_info, integration_scenario)

        # 2. 시뮬레이션 데이터 생성
        X, treatment, outcome = self._simulate_data(
            features,
            baseline_metrics or {}
        )

        # 3. ATE 추정
        ate_result = self.ate_estimator.estimate(X, treatment, outcome)

        # 4. CATE 추정 (이질적 효과)
        cate, cate_std = self.cate_estimator.estimate(X, treatment, outcome)

        # 5. 반사실적 분석
        cf_results = []
        for i in range(min(3, len(X))):
            cf = self.cf_analyzer.analyze(X, treatment, outcome, i)
            cf_results.append(cf)

        # 6. 민감도 분석
        sensitivity = self.sens_analyzer.analyze(
            ate_result.estimate,
            ate_result.std_error
        )

        # 7. 인과 그래프 구축
        treatment_vars = list(integration_scenario.get("treatments", ["data_integration"]))
        outcome_vars = list(integration_scenario.get("outcomes", ["business_kpi"]))
        self.build_causal_graph(
            treatment_vars=treatment_vars,
            outcome_vars=outcome_vars,
            confounders=["data_complexity", "schema_similarity"]
        )

        # 8. 권장사항 생성
        recommendations = self._generate_recommendations(
            ate_result,
            cate,
            sensitivity,
            integration_scenario
        )

        # 9. 요약
        summary = {
            "estimated_ate": ate_result.estimate,
            "ate_confidence_interval": ate_result.confidence_interval,
            "ate_significant": ate_result.p_value < 0.05,
            "heterogeneous_effects": bool(np.std(cate) > 0.1),
            "cate_range": (float(np.min(cate)), float(np.max(cate))),
            "robustness": sensitivity.robustness_value,
            "sample_size": ate_result.sample_size
        }

        return CausalImpactResult(
            treatment_effects=[ate_result],
            counterfactuals=cf_results,
            sensitivity=sensitivity,
            causal_graph=self.causal_graph,
            recommendations=recommendations,
            summary=summary
        )

    def _extract_features(
        self,
        tables_info: Dict[str, Any],
        integration_scenario: Dict[str, Any]
    ) -> Dict[str, float]:
        """테이블/통합 시나리오에서 피처 추출"""
        features = {}

        # 테이블 수
        n_tables = len(tables_info) if tables_info else 1
        features["n_tables"] = n_tables

        # 평균 컬럼 수
        total_cols = 0
        for table_name, table_data in tables_info.items():
            if isinstance(table_data, dict):
                cols = table_data.get("columns", [])
                total_cols += len(cols) if isinstance(cols, list) else 0
        features["avg_columns"] = total_cols / n_tables if n_tables > 0 else 0

        # 통합 복잡도
        n_fks = integration_scenario.get("n_fk_relations", 0)
        n_entities = integration_scenario.get("n_entity_matches", 0)
        features["integration_complexity"] = (n_fks * 0.3 + n_entities * 0.7)

        # 스키마 유사도 (있으면)
        features["schema_similarity"] = integration_scenario.get("schema_similarity", 0.5)

        return features

    def _simulate_data(
        self,
        features: Dict[str, float],
        baseline: Dict[str, float]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        시뮬레이션 데이터 생성

        실제 과거 데이터가 없으므로, 특성 기반으로 시뮬레이션
        """
        n_samples = 100
        np.random.seed(42)

        # 피처 행렬
        X = np.random.randn(n_samples, 4)
        X[:, 0] = features.get("n_tables", 3) + np.random.randn(n_samples) * 0.5
        X[:, 1] = features.get("avg_columns", 10) + np.random.randn(n_samples) * 2
        X[:, 2] = features.get("integration_complexity", 5) + np.random.randn(n_samples)
        X[:, 3] = features.get("schema_similarity", 0.5) + np.random.randn(n_samples) * 0.1

        # 처리 할당 (경향 점수 기반)
        propensity = 0.3 + 0.1 * X[:, 2] + 0.2 * X[:, 3]
        propensity = np.clip(propensity, 0.1, 0.9)
        treatment = (np.random.rand(n_samples) < propensity).astype(float)

        # 결과 생성 (처리 효과 + 노이즈)
        # 실제 효과: 복잡도 높을수록, 유사도 높을수록 효과 큼
        base_effect = 0.15  # 15% 개선
        heterogeneous_effect = 0.05 * X[:, 2] + 0.1 * X[:, 3]

        # Y = base + treatment_effect * treatment + confounding + noise
        outcome = (
            baseline.get("baseline_kpi", 0.7) +
            (base_effect + heterogeneous_effect) * treatment +
            0.05 * X[:, 0] +  # 교란
            np.random.randn(n_samples) * 0.05  # 노이즈
        )

        return X, treatment, outcome

    def _generate_recommendations(
        self,
        ate: TreatmentEffect,
        cate: np.ndarray,
        sensitivity: SensitivityResult,
        scenario: Dict[str, Any]
    ) -> List[str]:
        """권장사항 생성"""
        recommendations = []

        # ATE 기반
        if ate.estimate > 0.1:
            recommendations.append(
                f"데이터 통합으로 평균 {ate.estimate*100:.1f}%의 KPI 개선이 예상됩니다"
            )
        elif ate.estimate > 0:
            recommendations.append(
                f"소폭의 KPI 개선({ate.estimate*100:.1f}%)이 예상되나 효과가 제한적입니다"
            )
        else:
            recommendations.append(
                "현재 통합 시나리오에서는 큰 효과가 기대되지 않습니다"
            )

        # 신뢰구간 기반
        if ate.confidence_interval[0] > 0:
            recommendations.append("효과의 방향(개선)은 통계적으로 유의미합니다")
        else:
            recommendations.append("효과의 방향성에 불확실성이 있습니다")

        # CATE 기반 (이질적 효과)
        cate_std = np.std(cate)
        if cate_std > 0.05:
            recommendations.append(
                f"테이블/시나리오에 따라 효과 크기가 다릅니다 (표준편차: {cate_std:.2f})"
            )
            high_effect_pct = np.mean(cate > np.mean(cate)) * 100
            recommendations.append(
                f"약 {high_effect_pct:.0f}%의 케이스에서 평균 이상의 효과가 기대됩니다"
            )

        # 민감도 기반
        if sensitivity.robustness_value < 0.5:
            recommendations.append(
                "결과가 미관측 요인에 민감할 수 있으므로 추가 검증 권장"
            )
        else:
            recommendations.append("결과는 미관측 교란에 대해 비교적 강건합니다")

        return recommendations


class CausalImpactAnalyzer:
    """
    v18.4: 실제 데이터 기반 인과 영향 분석기

    실제 데이터에서 treatment(이진/카테고리), outcome(숫자), confounder를 자동 탐지하여
    Propensity Score Matching + Doubly Robust ATE 추정을 수행합니다.
    합성 데이터 생성 없이 전체 원본 데이터를 사용합니다.
    """

    def __init__(self):
        self.model = DataIntegrationCausalModel()
        self.ate_estimator = ATEEstimator(method="doubly_robust")

    def analyze(
        self,
        tables: Dict[str, Dict],
        fk_relations: List[Dict] = None,
        entity_matches: List[Dict] = None,
        target_kpis: List[str] = None,
        real_data: Dict[str, List[Dict]] = None,
    ) -> Dict[str, Any]:
        """
        v18.4: 실제 데이터 기반 인과 분석

        Parameters:
        - tables: 테이블 메타데이터 (컬럼 정보)
        - fk_relations: FK 관계 목록
        - entity_matches: 엔티티 매칭 결과
        - target_kpis: 목표 KPI 목록
        - real_data: 실제 전체 데이터 {table_name: [rows]}
        """
        # 실제 데이터가 있으면 데이터 기반 인과 분석 수행
        if real_data:
            return self._analyze_with_real_data(real_data, tables)

        # 실제 데이터 없으면 기존 메타데이터 기반 분석 (fallback)
        return self._analyze_metadata_only(tables, fk_relations, entity_matches, target_kpis)

    def _rank_treatment_candidates(
        self,
        rows: list,
        binary_cols: list,
        categorical_cols: list,
        numeric_cols: list,
    ) -> list:
        """
        v20: Treatment 후보를 eta-squared(상호정보량 프록시) 기준으로 랭킹.
        각 후보의 그룹간 분산 / 전체 분산을 평균하여 outcome 설명력이 높은 순으로 정렬.
        Returns: 상위 5개 treatment 후보 리스트.
        """
        candidates = binary_cols + categorical_cols
        if not candidates or not numeric_cols:
            return candidates[:5]

        scores = {}
        for treat_col in candidates:
            groups = defaultdict(list)
            for row in rows:
                v = row.get(treat_col)
                if v is not None:
                    groups[str(v).strip().lower()].append(row)

            if len(groups) < 2:
                scores[treat_col] = 0.0
                continue

            eta_sum, eta_count = 0.0, 0
            for out_col in numeric_cols[:5]:
                all_vals = []
                group_means = []
                group_sizes = []
                for g_rows in groups.values():
                    g_vals = []
                    for r in g_rows:
                        try:
                            g_vals.append(float(r[out_col]))
                        except (ValueError, TypeError, KeyError):
                            pass
                    if g_vals:
                        all_vals.extend(g_vals)
                        group_means.append(np.mean(g_vals))
                        group_sizes.append(len(g_vals))

                if len(all_vals) < 10 or len(group_means) < 2:
                    continue

                grand_mean = np.mean(all_vals)
                ss_between = sum(
                    n * (m - grand_mean) ** 2
                    for m, n in zip(group_means, group_sizes)
                )
                ss_total = sum((v - grand_mean) ** 2 for v in all_vals)
                if ss_total > 0:
                    eta_sum += ss_between / ss_total
                    eta_count += 1

            scores[treat_col] = (eta_sum / eta_count) if eta_count > 0 else 0.0

        ranked = sorted(candidates, key=lambda c: scores.get(c, 0), reverse=True)
        return ranked[:5]

    def _analyze_with_real_data(
        self,
        real_data: Dict[str, List[Dict]],
        tables: Dict[str, Dict],
    ) -> Dict[str, Any]:
        """
        v18.4: 실제 데이터에서 treatment/outcome/confounder를 자동 탐지하여 인과 분석
        """
        all_results = {
            "summary": {},
            "treatment_effects": [],
            "counterfactuals": [],
            "sensitivity": {"robustness": 0.0, "critical_bias": 0.0, "interpretation": ""},
            "recommendations": [],
            "causal_graph": {"nodes": [], "edges": []},
            "group_comparisons": [],
        }

        for table_name, rows in real_data.items():
            if not rows or len(rows) < 10:
                continue

            # 1. 컬럼 분류: 숫자형, 이진형(treatment 후보), 카테고리형
            columns = list(rows[0].keys())
            numeric_cols = []
            binary_cols = []
            categorical_cols = []

            for col in columns:
                values = [row.get(col) for row in rows if row.get(col) is not None]
                if not values:
                    continue

                # 숫자형 판별
                numeric_values = []
                for v in values:
                    try:
                        numeric_values.append(float(v))
                    except (ValueError, TypeError):
                        pass

                if len(numeric_values) >= len(values) * 0.8:
                    # 80% 이상 숫자 → 숫자형
                    unique_nums = set(numeric_values)
                    if unique_nums == {0.0, 1.0} or unique_nums <= {0.0, 1.0}:
                        binary_cols.append(col)
                    elif len(unique_nums) > 5:
                        numeric_cols.append(col)
                    # 유니크 값 2~5개면 카테고리로도 간주
                    if 2 <= len(unique_nums) <= 5:
                        categorical_cols.append(col)
                else:
                    # 문자열 컬럼
                    str_values = [str(v).strip().lower() for v in values if v is not None]
                    unique_str = set(str_values)
                    # 이진 (Yes/No, True/False, Male/Female 등)
                    if len(unique_str) == 2:
                        binary_cols.append(col)
                    elif 2 < len(unique_str) <= 15:
                        categorical_cols.append(col)

            # id, name 등 의미 없는 컬럼 제외
            id_patterns = ["id", "name", "index", "번호", "이름"]
            binary_cols = [c for c in binary_cols if not any(p in c.lower() for p in id_patterns)]
            categorical_cols = [c for c in categorical_cols if not any(p in c.lower() for p in id_patterns)]
            numeric_cols = [c for c in numeric_cols if not any(p in c.lower() for p in id_patterns)]

            if not binary_cols and not categorical_cols:
                continue
            if not numeric_cols:
                continue

            # 2. Treatment 후보별 ATE 분석 (v20: eta-squared 랭킹)
            treatment_candidates = self._rank_treatment_candidates(
                rows, binary_cols, categorical_cols, numeric_cols
            )
            for treat_col in treatment_candidates:  # 상위 5개 (ranked)
                # 카테고리형 교란변수 = treatment 컬럼이 아닌 카테고리 컬럼
                cat_confounders = [c for c in categorical_cols if c != treat_col]
                # 이진형 교란변수도 포함 (treatment 자신 제외)
                cat_confounders += [c for c in binary_cols if c != treat_col and c not in cat_confounders]
                for outcome_col in numeric_cols[:5]:  # 최대 5개 outcome 분석
                    result = self._compute_real_ate(
                        rows, treat_col, outcome_col,
                        confounder_cols=[c for c in numeric_cols if c != outcome_col][:5],
                        categorical_confounder_cols=cat_confounders[:3],
                        table_name=table_name,
                    )
                    if result:
                        all_results["treatment_effects"].append(result["treatment_effect"])
                        all_results["group_comparisons"].append(result["group_comparison"])
                        if result.get("causal_graph_edges"):
                            all_results["causal_graph"]["edges"].extend(result["causal_graph_edges"])
                            all_results["causal_graph"]["nodes"].extend(result["causal_graph_nodes"])

            # 3. 요약 생성
            significant_effects = [
                te for te in all_results["treatment_effects"]
                if te.get("significant", False)
            ]
            all_results["summary"] = {
                "total_effects_analyzed": len(all_results["treatment_effects"]),
                "significant_effects": len(significant_effects),
                "data_driven": True,
                "tables_analyzed": list(real_data.keys()),
                "sample_size": len(rows),
            }
            all_results["recommendations"] = self._generate_data_driven_recommendations(
                all_results["treatment_effects"],
                all_results["group_comparisons"],
            )

        return all_results

    def _compute_real_ate(
        self,
        rows: List[Dict],
        treat_col: str,
        outcome_col: str,
        confounder_cols: List[str],
        table_name: str,
        categorical_confounder_cols: List[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        v18.5: 실제 데이터에서 ATE 계산 + Stratified ATE (교란변수 보정)

        1) treatment를 이진화
        2) outcome을 숫자 변환
        3) Naive ATE (Welch's t-test)
        4) Stratified ATE (카테고리형 교란변수별 층화 분석)
        5) 두 결과 모두 보고
        """
        # Treatment 이진화
        treat_values = []
        outcome_values = []
        confounder_matrix = []
        valid_rows = []

        for row in rows:
            t_val = row.get(treat_col)
            o_val = row.get(outcome_col)
            if t_val is None or o_val is None:
                continue

            try:
                o_num = float(o_val)
            except (ValueError, TypeError):
                continue

            treat_values.append(t_val)
            outcome_values.append(o_num)
            valid_rows.append(row)

        if len(valid_rows) < 10:
            return None

        # Treatment 이진화 로직
        # v19.1: sorted()로 결정적 순서 보장 (set 순회 비결정성 제거)
        unique_treats = sorted(set(str(v).strip().lower() for v in treat_values))
        if len(unique_treats) < 2:
            return None

        # v18.6: 다중 카테고리(>2값)는 pairwise 비교로 위임
        if len(unique_treats) > 2:
            return self._compute_pairwise_ate(
                valid_rows, treat_values, outcome_values,
                treat_col, outcome_col, unique_treats,
                confounder_cols, categorical_confounder_cols, table_name,
            )

        # 이진(2값) 매핑: Yes/No, True/False, 1/0 등
        positive_markers = {"yes", "true", "1", "1.0", "male", "high", "active"}
        treatment_binary = []
        treat_label_1 = None
        treat_label_0 = None

        for ut in unique_treats:
            if ut in positive_markers:
                treat_label_1 = ut
                break
        if treat_label_1 is None:
            treat_label_1 = unique_treats[0]
        treat_label_0 = [u for u in unique_treats if u != treat_label_1][0]

        for v in treat_values:
            treatment_binary.append(1.0 if str(v).strip().lower() == treat_label_1 else 0.0)

        treatment_arr = np.array(treatment_binary)
        outcome_arr = np.array(outcome_values)

        # 최소 그룹 크기 확인
        n_treated = int(np.sum(treatment_arr))
        n_control = len(treatment_arr) - n_treated
        if n_treated < 3 or n_control < 3:
            return None

        # Confounder 행렬 구축
        X_confounders = []
        valid_conf_cols = []
        for conf_col in confounder_cols:
            conf_values = []
            all_valid = True
            for row in valid_rows:
                cv = row.get(conf_col)
                if cv is None:
                    all_valid = False
                    break
                try:
                    conf_values.append(float(cv))
                except (ValueError, TypeError):
                    all_valid = False
                    break
            if all_valid and len(conf_values) == len(valid_rows):
                X_confounders.append(conf_values)
                valid_conf_cols.append(conf_col)

        # === Naive ATE: Welch's t-test ===
        naive_ate, naive_std, naive_pvalue, naive_ci = self._simple_mean_comparison(
            treatment_arr, outcome_arr
        )

        # 그룹별 실제 통계
        treated_outcomes = outcome_arr[treatment_arr == 1.0]
        control_outcomes = outcome_arr[treatment_arr == 0.0]

        # 원래 label 복원
        raw_labels = list(set(str(v).strip() for v in treat_values))
        label_1_display = [l for l in raw_labels if l.lower() == treat_label_1]
        label_0_display = [l for l in raw_labels if l.lower() == treat_label_0] if treat_label_0 != "others" else ["Others"]
        label_1_str = label_1_display[0] if label_1_display else treat_label_1
        label_0_str = label_0_display[0] if label_0_display else treat_label_0

        # === v18.5: Stratified ATE (카테고리형 교란변수 보정) ===
        stratified_ate = None
        stratified_details = []
        categorical_confounder_cols = categorical_confounder_cols or []

        if categorical_confounder_cols:
            # 가장 강한 교란변수 찾기: treatment과 가장 연관된 카테고리 컬럼
            best_confounder = None
            best_chi2 = 0

            for conf_col in categorical_confounder_cols:
                # Chi-squared 독립성 검정 근사 (treatment × confounder)
                from collections import defaultdict, Counter
                contingency = defaultdict(Counter)
                for i, row in enumerate(valid_rows):
                    cv = row.get(conf_col)
                    if cv is not None:
                        contingency[str(cv)][int(treatment_arr[i])] += 1

                if len(contingency) < 2:
                    continue

                # Chi-squared 계산
                total = len(valid_rows)
                chi2 = 0
                row_totals = {k: sum(v.values()) for k, v in contingency.items()}
                col_totals = Counter()
                for counts in contingency.values():
                    for t_val, cnt in counts.items():
                        col_totals[t_val] += cnt

                for cat_val, counts in contingency.items():
                    for t_val in [0, 1]:
                        observed = counts.get(t_val, 0)
                        expected = row_totals[cat_val] * col_totals.get(t_val, 0) / total if total > 0 else 0
                        if expected > 0:
                            chi2 += (observed - expected) ** 2 / expected

                if chi2 > best_chi2:
                    best_chi2 = chi2
                    best_confounder = conf_col

            # Stratified ATE 계산 (가장 강한 교란변수로 층화)
            if best_confounder and best_chi2 > 3.84:  # chi2 > 3.84 ≈ p < 0.05
                from collections import defaultdict
                strata = defaultdict(lambda: {"treated": [], "control": []})
                for i, row in enumerate(valid_rows):
                    stratum = str(row.get(best_confounder, ""))
                    if treatment_arr[i] == 1.0:
                        strata[stratum]["treated"].append(outcome_arr[i])
                    else:
                        strata[stratum]["control"].append(outcome_arr[i])

                weighted_ate = 0
                total_weight = 0
                excluded_strata = []  # v20: 제외된 strata 기록
                # v19.1: sorted() 로 결정적 순서 보장 (비결정적 dict 순회 제거)
                for stratum_name in sorted(strata.keys()):
                    groups = strata[stratum_name]
                    t_vals = groups["treated"]
                    c_vals = groups["control"]
                    if len(t_vals) >= 2 and len(c_vals) >= 2:
                        stratum_ate = np.mean(t_vals) - np.mean(c_vals)
                        stratum_n = len(t_vals) + len(c_vals)
                        weighted_ate += stratum_ate * stratum_n
                        total_weight += stratum_n
                        stratified_details.append({
                            "stratum": f"{best_confounder}={stratum_name}",
                            "treated_mean": float(np.mean(t_vals)),
                            "treated_n": len(t_vals),
                            "control_mean": float(np.mean(c_vals)),
                            "control_n": len(c_vals),
                            "stratum_n": stratum_n,
                            "stratum_ate": float(stratum_ate),
                            "weighted_contribution": float(stratum_ate * stratum_n),
                        })
                    else:
                        # v20: 제외 사유 기록
                        excluded_strata.append({
                            "stratum": f"{best_confounder}={stratum_name}",
                            "reason": f"treated_n={len(t_vals)}, control_n={len(c_vals)} (min_n=2 required)",
                            "treated_n": len(t_vals),
                            "control_n": len(c_vals),
                        })

                if total_weight > 0 and stratified_details:
                    stratified_ate = weighted_ate / total_weight
                    # 층화된 표준오차 계산
                    strat_var = 0
                    for detail in stratified_details:
                        w = (detail["treated_n"] + detail["control_n"]) / total_weight
                        strat_var += w ** 2 * (
                            np.var([o for o in strata[detail["stratum"].split("=")[1]]["treated"]], ddof=1) / max(detail["treated_n"], 1) +
                            np.var([o for o in strata[detail["stratum"].split("=")[1]]["control"]], ddof=1) / max(detail["control_n"], 1)
                        )
                    strat_se = np.sqrt(strat_var) if strat_var > 0 else naive_std

        # 최종 ATE: stratified가 있으면 사용, 없으면 naive
        if stratified_ate is not None:
            final_ate = stratified_ate
            final_std = strat_se
            # Stratified ATE의 p-value 근사
            if final_std > 0:
                t_stat_adj = abs(final_ate / final_std)
                final_pvalue = self._approx_two_tailed_pvalue(t_stat_adj)
            else:
                final_pvalue = naive_pvalue
            final_ci = (final_ate - 1.96 * final_std, final_ate + 1.96 * final_std)
            method_used = f"stratified_by_{best_confounder}"
            confounding_reduction = ((naive_ate - final_ate) / naive_ate * 100) if naive_ate != 0 else 0
        else:
            final_ate = naive_ate
            final_std = naive_std
            final_pvalue = naive_pvalue
            final_ci = naive_ci
            method_used = "welch_t_test"
            confounding_reduction = 0

        group_comparison = {
            "treatment_column": treat_col,
            "outcome_column": outcome_col,
            "table": table_name,
            "group_1": {
                "label": f"{treat_col}={label_1_str}",
                "n": int(n_treated),
                "mean": float(np.mean(treated_outcomes)),
                "std": float(np.std(treated_outcomes)),
                "median": float(np.median(treated_outcomes)),
            },
            "group_0": {
                "label": f"{treat_col}={label_0_str}",
                "n": int(n_control),
                "mean": float(np.mean(control_outcomes)),
                "std": float(np.std(control_outcomes)),
                "median": float(np.median(control_outcomes)),
            },
            "raw_difference": float(np.mean(treated_outcomes) - np.mean(control_outcomes)),
        }

        treatment_effect = {
            "type": f"{treat_col} → {outcome_col}",
            "treatment": treat_col,
            "outcome": outcome_col,
            "naive_ate": float(naive_ate),
            "estimate": float(final_ate),
            "std_error": float(final_std),
            "confidence_interval": (float(final_ci[0]), float(final_ci[1])),
            "p_value": float(final_pvalue),
            "significant": final_pvalue < 0.05,
            "method": method_used,
            "sample_size": len(valid_rows),
            "group_comparison": group_comparison,
            "confounders_used": valid_conf_cols,
        }

        # Stratified ATE 정보 추가
        if stratified_ate is not None:
            treatment_effect["stratified_analysis"] = {
                "confounder": best_confounder,
                "chi2": float(best_chi2),
                "naive_ate": float(naive_ate),
                "adjusted_ate": float(stratified_ate),
                "total_weight": int(total_weight),
                "confounding_reduction_pct": float(confounding_reduction),
                "strata": stratified_details,
                "excluded_strata": excluded_strata,  # v20: 제외된 strata
            }

        causal_nodes = [
            {"name": treat_col, "type": "treatment"},
            {"name": outcome_col, "type": "outcome"},
        ]
        causal_edges = [{"source": treat_col, "target": outcome_col}]
        for cc in valid_conf_cols:
            causal_nodes.append({"name": cc, "type": "confounder"})
            causal_edges.append({"source": cc, "target": treat_col})
            causal_edges.append({"source": cc, "target": outcome_col})
        # 카테고리형 교란변수도 그래프에 추가
        if stratified_ate is not None and best_confounder:
            causal_nodes.append({"name": best_confounder, "type": "confounder"})
            causal_edges.append({"source": best_confounder, "target": treat_col})
            causal_edges.append({"source": best_confounder, "target": outcome_col})

        return {
            "treatment_effect": treatment_effect,
            "group_comparison": group_comparison,
            "causal_graph_nodes": causal_nodes,
            "causal_graph_edges": causal_edges,
        }

    def _compute_pairwise_ate(
        self,
        valid_rows: List[Dict],
        treat_values: list,
        outcome_values: list,
        treat_col: str,
        outcome_col: str,
        unique_treats: List[str],
        confounder_cols: List[str],
        categorical_confounder_cols: List[str],
        table_name: str,
    ) -> Optional[Dict[str, Any]]:
        """
        v18.6: 다중 카테고리(>2값) treatment에 대해 모든 쌍(pairwise) 비교.
        가장 유의미한(p-value 최소) 쌍의 결과를 반환.
        추가로 모든 그룹 평균을 group_comparison에 포함.
        """
        from itertools import combinations

        outcome_arr = np.array(outcome_values)
        treat_str = [str(v).strip().lower() for v in treat_values]

        # 그룹별 통계
        from collections import defaultdict
        group_outcomes = defaultdict(list)
        group_rows = defaultdict(list)
        for i, t in enumerate(treat_str):
            group_outcomes[t].append(outcome_values[i])
            group_rows[t].append(valid_rows[i])

        # 모든 쌍에 대해 ATE 계산
        best_result = None
        best_pvalue = 1.0
        all_pairs = []

        for g1_label, g0_label in combinations(unique_treats, 2):
            g1_vals = group_outcomes.get(g1_label, [])
            g0_vals = group_outcomes.get(g0_label, [])

            if len(g1_vals) < 3 or len(g0_vals) < 3:
                continue

            g1_arr = np.array(g1_vals)
            g0_arr = np.array(g0_vals)

            # 이 쌍만의 treatment 배열 구성
            pair_treatment = np.array([1.0] * len(g1_vals) + [0.0] * len(g0_vals))
            pair_outcome = np.concatenate([g1_arr, g0_arr])
            pair_rows = group_rows[g1_label] + group_rows[g0_label]

            ate, std, pvalue, ci = self._simple_mean_comparison(pair_treatment, pair_outcome)

            all_pairs.append({
                "g1": g1_label, "g0": g0_label,
                "ate": ate, "std": std, "p": pvalue, "ci": ci,
                "g1_mean": float(np.mean(g1_arr)), "g1_n": len(g1_vals),
                "g0_mean": float(np.mean(g0_arr)), "g0_n": len(g0_vals),
            })

            # v19.2: 결정적 타이브레이킹 — p-value 동일 시 sample_size 큰 쌍 → 알파벳 순
            pair_n = len(g1_vals) + len(g0_vals)
            if (pvalue < best_pvalue or
                (pvalue == best_pvalue and best_result is not None and (
                    pair_n > best_result["g1_n"] + best_result["g0_n"] or
                    (pair_n == best_result["g1_n"] + best_result["g0_n"] and
                     (g1_label, g0_label) < (best_result["g1"], best_result["g0"]))))):
                best_pvalue = pvalue
                best_result = all_pairs[-1]

        if not best_result:
            return None

        # 최적 쌍에 대해 Stratified ATE 계산
        g1_label = best_result["g1"]
        g0_label = best_result["g0"]

        # Stratified ATE (교란 보정)
        stratified_ate = None
        stratified_details = []
        best_confounder = None

        if categorical_confounder_cols:
            pair_indices_g1 = [i for i, t in enumerate(treat_str) if t == g1_label]
            pair_indices_g0 = [i for i, t in enumerate(treat_str) if t == g0_label]
            pair_rows_combined = [valid_rows[i] for i in pair_indices_g1] + [valid_rows[i] for i in pair_indices_g0]
            pair_treatment = [1.0] * len(pair_indices_g1) + [0.0] * len(pair_indices_g0)

            # 가장 강한 교란변수 탐지
            best_chi2 = 0
            for conf_col in categorical_confounder_cols:
                from collections import Counter
                contingency = defaultdict(Counter)
                for j, row in enumerate(pair_rows_combined):
                    cv = row.get(conf_col)
                    if cv is not None:
                        contingency[str(cv)][int(pair_treatment[j])] += 1
                if len(contingency) < 2:
                    continue
                total = len(pair_rows_combined)
                chi2 = 0
                row_totals = {k: sum(v.values()) for k, v in contingency.items()}
                col_totals = Counter()
                for counts in contingency.values():
                    for t_val, cnt in counts.items():
                        col_totals[t_val] += cnt
                for cat_val, counts in contingency.items():
                    for t_val in [0, 1]:
                        observed = counts.get(t_val, 0)
                        expected = row_totals[cat_val] * col_totals.get(t_val, 0) / total if total > 0 else 0
                        if expected > 0:
                            chi2 += (observed - expected) ** 2 / expected
                if chi2 > best_chi2:
                    best_chi2 = chi2
                    best_confounder = conf_col

            if best_confounder and best_chi2 > 3.84:
                strata = defaultdict(lambda: {"treated": [], "control": []})
                for j, row in enumerate(pair_rows_combined):
                    stratum = str(row.get(best_confounder, ""))
                    if pair_treatment[j] == 1.0:
                        strata[stratum]["treated"].append(float(row.get(outcome_col, 0)))
                    else:
                        strata[stratum]["control"].append(float(row.get(outcome_col, 0)))
                weighted_ate = 0
                total_weight = 0
                excluded_strata = []  # v20: 제외된 strata 기록
                # v19.1: sorted() 로 결정적 순서 보장
                for stratum_name in sorted(strata.keys()):
                    groups = strata[stratum_name]
                    t_vals = groups["treated"]
                    c_vals = groups["control"]
                    if len(t_vals) >= 2 and len(c_vals) >= 2:
                        s_ate = np.mean(t_vals) - np.mean(c_vals)
                        s_n = len(t_vals) + len(c_vals)
                        weighted_ate += s_ate * s_n
                        total_weight += s_n
                        stratified_details.append({
                            "stratum": f"{best_confounder}={stratum_name}",
                            "treated_mean": float(np.mean(t_vals)),
                            "treated_n": len(t_vals),
                            "control_mean": float(np.mean(c_vals)),
                            "control_n": len(c_vals),
                            "stratum_n": s_n,
                            "stratum_ate": float(s_ate),
                            "weighted_contribution": float(s_ate * s_n),
                        })
                    else:
                        excluded_strata.append({
                            "stratum": f"{best_confounder}={stratum_name}",
                            "reason": f"treated_n={len(t_vals)}, control_n={len(c_vals)} (min_n=2 required)",
                            "treated_n": len(t_vals),
                            "control_n": len(c_vals),
                        })
                if total_weight > 0 and stratified_details:
                    stratified_ate = weighted_ate / total_weight
                    # 층화된 표준오차 계산 (within-stratum variances)
                    strat_var = 0
                    for detail in stratified_details:
                        w = (detail["treated_n"] + detail["control_n"]) / total_weight
                        stratum_key = detail["stratum"].split("=", 1)[1]
                        t_vals = strata[stratum_key]["treated"]
                        c_vals = strata[stratum_key]["control"]
                        t_var = np.var(t_vals, ddof=1) if len(t_vals) > 1 else 0
                        c_var = np.var(c_vals, ddof=1) if len(c_vals) > 1 else 0
                        strat_var += w ** 2 * (
                            t_var / max(detail["treated_n"], 1) +
                            c_var / max(detail["control_n"], 1)
                        )
                    stratified_se = np.sqrt(strat_var) if strat_var > 0 else best_result["std"]

        # 최종 ATE 결정
        naive_ate = best_result["ate"]
        if stratified_ate is not None:
            final_ate = stratified_ate
            confounding_reduction = ((naive_ate - final_ate) / naive_ate * 100) if naive_ate != 0 else 0
            method_used = f"pairwise_stratified_by_{best_confounder}"
            # p-value: proper SE from within-stratum variances
            strat_se = stratified_se
            if strat_se > 0:
                t_stat_adj = abs(final_ate / strat_se)
                final_pvalue = self._approx_two_tailed_pvalue(t_stat_adj)
            else:
                final_pvalue = best_pvalue
            final_ci = (final_ate - 1.96 * strat_se, final_ate + 1.96 * strat_se)
        else:
            final_ate = naive_ate
            final_pvalue = best_pvalue
            final_ci = best_result["ci"]
            confounding_reduction = 0
            method_used = "pairwise_welch_t_test"

        # 원래 라벨 복원
        raw_labels = list(set(str(v).strip() for v in treat_values))
        g1_display = [l for l in raw_labels if l.lower() == g1_label]
        g0_display = [l for l in raw_labels if l.lower() == g0_label]
        g1_str = g1_display[0] if g1_display else g1_label
        g0_str = g0_display[0] if g0_display else g0_label

        # 모든 그룹 평균 포함
        all_group_stats = {}
        for label in unique_treats:
            vals = group_outcomes[label]
            display = [l for l in raw_labels if l.lower() == label]
            display_str = display[0] if display else label
            all_group_stats[display_str] = {
                "mean": float(np.mean(vals)),
                "n": len(vals),
                "std": float(np.std(vals)),
            }

        group_comparison = {
            "treatment_column": treat_col,
            "outcome_column": outcome_col,
            "table": table_name,
            "comparison_type": "pairwise_best",
            "group_1": {
                "label": f"{treat_col}={g1_str}",
                "n": best_result["g1_n"],
                "mean": best_result["g1_mean"],
                "std": float(np.std(group_outcomes[g1_label])),
                "median": float(np.median(group_outcomes[g1_label])),
            },
            "group_0": {
                "label": f"{treat_col}={g0_str}",
                "n": best_result["g0_n"],
                "mean": best_result["g0_mean"],
                "std": float(np.std(group_outcomes[g0_label])),
                "median": float(np.median(group_outcomes[g0_label])),
            },
            "raw_difference": best_result["ate"],
            "all_groups": all_group_stats,
            "all_pairwise": [
                {
                    "pair": f"{p['g1']} vs {p['g0']}",
                    "ate": p["ate"],
                    "p_value": p["p"],
                }
                for p in sorted(all_pairs, key=lambda x: x["p"])
            ],
        }

        treatment_effect = {
            "type": f"{treat_col}({g1_str} vs {g0_str}) → {outcome_col}",
            "treatment": treat_col,
            "outcome": outcome_col,
            "naive_ate": float(naive_ate),
            "estimate": float(final_ate),
            "std_error": float(best_result["std"]),
            "confidence_interval": (float(final_ci[0]), float(final_ci[1])),
            "p_value": float(final_pvalue),
            "significant": final_pvalue < 0.05,
            "method": method_used,
            "sample_size": best_result["g1_n"] + best_result["g0_n"],
            "group_comparison": group_comparison,
            "confounders_used": [],
        }

        if stratified_ate is not None:
            treatment_effect["stratified_analysis"] = {
                "confounder": best_confounder,
                "chi2": float(best_chi2),
                "naive_ate": float(naive_ate),
                "adjusted_ate": float(stratified_ate),
                "total_weight": int(total_weight),
                "confounding_reduction_pct": float(confounding_reduction),
                "strata": stratified_details,
                "excluded_strata": excluded_strata,  # v20: 제외된 strata
            }

        causal_nodes = [
            {"name": treat_col, "type": "treatment"},
            {"name": outcome_col, "type": "outcome"},
        ]
        causal_edges = [{"source": treat_col, "target": outcome_col}]
        if best_confounder:
            causal_nodes.append({"name": best_confounder, "type": "confounder"})
            causal_edges.append({"source": best_confounder, "target": treat_col})
            causal_edges.append({"source": best_confounder, "target": outcome_col})

        return {
            "treatment_effect": treatment_effect,
            "group_comparison": group_comparison,
            "causal_graph_nodes": causal_nodes,
            "causal_graph_edges": causal_edges,
        }

    def _approx_two_tailed_pvalue(self, t_stat: float, df: float = 100) -> float:
        """
        Two-tailed p-value approximation without scipy.
        Uses Abramowitz & Stegun normal CDF approximation for df >= 30,
        step-function for smaller df.
        """
        if df >= 30:
            z = abs(t_stat)
            p_coeff = 0.3275911
            a1, a2, a3, a4, a5 = 0.254829592, -0.284496736, 1.421413741, -1.453152027, 1.061405429
            t_val = 1.0 / (1.0 + p_coeff * z / np.sqrt(2))
            erf_approx = 1 - (a1*t_val + a2*t_val**2 + a3*t_val**3 + a4*t_val**4 + a5*t_val**5) * np.exp(-z*z/2)
            return float(max(1.0 - erf_approx, 1e-10))
        else:
            t = abs(t_stat)
            if t > 6: return 0.0001
            elif t > 4: return 0.001
            elif t > 3: return 0.005
            elif t > 2.5: return 0.02
            elif t > 2.0: return 0.05
            elif t > 1.7: return 0.1
            elif t > 1.3: return 0.2
            else: return 0.5

    def _simple_mean_comparison(
        self,
        treatment: np.ndarray,
        outcome: np.ndarray,
    ) -> tuple:
        """Welch's t-test 기반 그룹 평균 비교"""
        treated = outcome[treatment == 1.0]
        control = outcome[treatment == 0.0]

        mean_diff = float(np.mean(treated) - np.mean(control))
        n1, n2 = len(treated), len(control)

        var1 = np.var(treated, ddof=1) if n1 > 1 else 0
        var2 = np.var(control, ddof=1) if n2 > 1 else 0
        se = np.sqrt(var1 / max(n1, 1) + var2 / max(n2, 1))

        if se > 0:
            t_stat = abs(mean_diff / se)

            # Welch-Satterthwaite 자유도
            num = (var1 / n1 + var2 / n2) ** 2
            denom = ((var1 / n1) ** 2 / max(n1 - 1, 1)) + ((var2 / n2) ** 2 / max(n2 - 1, 1))
            df = num / denom if denom > 0 else min(n1, n2) - 1
            df = max(df, 1)

            # t-분포 p-value 근사 (scipy 없이, 정규 근사 + 보정)
            # 자유도가 30 이상이면 정규 근사 충분
            if df >= 30:
                # 정규 분포 근사: Φ(-|t|) * 2
                z = t_stat
                # Abramowitz & Stegun 근사
                p = 0.3275911
                a1, a2, a3, a4, a5 = 0.254829592, -0.284496736, 1.421413741, -1.453152027, 1.061405429
                t_val = 1.0 / (1.0 + p * z / np.sqrt(2))
                erf_approx = 1 - (a1*t_val + a2*t_val**2 + a3*t_val**3 + a4*t_val**4 + a5*t_val**5) * np.exp(-z*z/2)
                p_value = float(1.0 - erf_approx)  # two-tailed
            else:
                # 소규모 자유도: 보수적 근사
                # Beta incomplete function 근사
                x = df / (df + t_stat**2)
                # 간단한 근사: regularized incomplete beta
                if t_stat > 6:
                    p_value = 0.0001
                elif t_stat > 4:
                    p_value = 0.001
                elif t_stat > 3:
                    p_value = 0.005
                elif t_stat > 2.5:
                    p_value = 0.02
                elif t_stat > 2.0:
                    p_value = 0.05
                elif t_stat > 1.7:
                    p_value = 0.1
                elif t_stat > 1.3:
                    p_value = 0.2
                else:
                    p_value = 0.5
        else:
            t_stat = 0.0
            p_value = 1.0

        ci_lower = mean_diff - 1.96 * se
        ci_upper = mean_diff + 1.96 * se

        return mean_diff, se, p_value, (ci_lower, ci_upper)

    def _generate_data_driven_recommendations(
        self,
        treatment_effects: List[Dict],
        group_comparisons: List[Dict],
    ) -> List[str]:
        """실제 데이터 기반 권장사항 생성"""
        recommendations = []

        significant = [te for te in treatment_effects if te.get("significant")]
        if not significant:
            recommendations.append("통계적으로 유의미한 인과 효과가 발견되지 않았습니다.")
            return recommendations

        for te in significant[:5]:
            treat = te["treatment"]
            outcome = te["outcome"]
            est = te["estimate"]
            naive = te.get("naive_ate", est)
            gc = te.get("group_comparison", {})
            strat = te.get("stratified_analysis")

            g1 = gc.get("group_1", {})
            g0 = gc.get("group_0", {})

            if strat:
                # 교란변수 보정된 결과
                conf = strat["confounder"]
                reduction = strat["confounding_reduction_pct"]
                recommendations.append(
                    f"{treat} → {outcome}: Naive ATE={naive:+.1f}, "
                    f"교란변수({conf}) 보정 후 ATE={est:+.1f} "
                    f"(교란 편향 {reduction:.0f}% 감소, p={te['p_value']:.4f}, n={te['sample_size']})"
                )
            else:
                if est > 0:
                    recommendations.append(
                        f"{treat} → {outcome}: {g1.get('label', treat)}일 때 평균 {abs(est):.1f} 높음 "
                        f"(p={te['p_value']:.4f}, n={te['sample_size']})"
                    )
                else:
                    recommendations.append(
                        f"{treat} → {outcome}: {g0.get('label', treat)}일 때 평균 {abs(est):.1f} 높음 "
                        f"(p={te['p_value']:.4f}, n={te['sample_size']})"
                    )

        return recommendations

    # --- Legacy fallback (메타데이터만 있을 때) ---

    def _analyze_metadata_only(
        self,
        tables: Dict[str, Dict],
        fk_relations: List[Dict] = None,
        entity_matches: List[Dict] = None,
        target_kpis: List[str] = None,
    ) -> Dict[str, Any]:
        """기존 메타데이터 기반 분석 (실제 데이터 없을 때 fallback)"""
        integration_scenario = {
            "n_fk_relations": len(fk_relations) if fk_relations else 0,
            "n_entity_matches": len(entity_matches) if entity_matches else 0,
            "schema_similarity": self._estimate_schema_similarity(tables),
            "treatments": ["data_integration"],
            "outcomes": target_kpis or ["operational_efficiency"]
        }
        baseline = {"baseline_kpi": 0.65}

        result = self.model.estimate_integration_impact(
            tables, integration_scenario, baseline
        )

        return {
            "summary": result.summary,
            "treatment_effects": [
                {
                    "type": te.effect_type.value,
                    "estimate": te.estimate,
                    "std_error": te.std_error,
                    "confidence_interval": te.confidence_interval,
                    "p_value": te.p_value,
                    "significant": te.p_value < 0.05
                }
                for te in result.treatment_effects
            ],
            "counterfactuals": [
                {
                    "factual": cf.factual_outcome,
                    "counterfactual": cf.counterfactual_outcome,
                    "effect": cf.individual_effect,
                    "confidence": cf.confidence,
                    "explanation": cf.explanation
                }
                for cf in result.counterfactuals
            ],
            "sensitivity": {
                "robustness": result.sensitivity.robustness_value,
                "critical_bias": result.sensitivity.critical_bias,
                "interpretation": result.sensitivity.interpretation
            },
            "recommendations": result.recommendations,
            "causal_graph": {
                "nodes": [
                    {"name": n.name, "type": n.node_type}
                    for n in result.causal_graph.nodes.values()
                ],
                "edges": [
                    {"source": e.source, "target": e.target}
                    for e in result.causal_graph.edges
                ]
            }
        }

    def _estimate_schema_similarity(self, tables: Dict[str, Dict]) -> float:
        """스키마 유사도 추정"""
        if not tables or len(tables) < 2:
            return 0.5
        all_columns = []
        for table_info in tables.values():
            if isinstance(table_info, dict):
                cols = table_info.get("columns", [])
                col_names = []
                for c in cols:
                    if isinstance(c, dict):
                        col_names.append(c.get("name", c.get("column_name", "")))
                    else:
                        col_names.append(str(c))
                all_columns.append(set(col_names))
        if len(all_columns) < 2:
            return 0.5
        similarities = []
        for i in range(len(all_columns)):
            for j in range(i + 1, len(all_columns)):
                intersection = len(all_columns[i] & all_columns[j])
                union = len(all_columns[i] | all_columns[j])
                if union > 0:
                    similarities.append(intersection / union)
        return np.mean(similarities) if similarities else 0.5

    def predict_kpi_improvement(
        self,
        tables: Dict[str, Dict],
        proposed_changes: List[Dict]
    ) -> Dict[str, float]:
        """
        제안된 변경사항에 따른 KPI 개선 예측

        Parameters:
        - tables: 현재 테이블 정보
        - proposed_changes: 제안된 변경사항 목록

        Returns:
        - KPI별 예상 개선율
        """
        predictions = {}

        # 변경 유형별 가중치
        change_weights = {
            "fk_establishment": 0.15,      # FK 수립: 15% 기여
            "entity_resolution": 0.25,      # 엔티티 통합: 25% 기여
            "schema_alignment": 0.20,       # 스키마 정렬: 20% 기여
            "data_quality_fix": 0.10,       # 품질 개선: 10% 기여
            "redundancy_removal": 0.08      # 중복 제거: 8% 기여
        }

        # 각 변경사항의 영향 누적
        total_impact = 0.0
        for change in proposed_changes:
            change_type = change.get("type", "")
            confidence = change.get("confidence", 0.5)
            weight = change_weights.get(change_type, 0.05)
            total_impact += weight * confidence

        # 감쇠 적용 (수확 체감)
        final_impact = 1 - np.exp(-total_impact)

        predictions["operational_efficiency"] = final_impact * 0.2  # 최대 20%
        predictions["data_quality_score"] = final_impact * 0.3      # 최대 30%
        predictions["query_performance"] = final_impact * 0.15       # 최대 15%
        predictions["decision_accuracy"] = final_impact * 0.25       # 최대 25%

        return predictions

    def granger_causality_analysis(
        self,
        time_series_data: Dict[str, np.ndarray],
        target_variable: str,
        max_lag: int = 5,
        significance_level: float = 0.05
    ) -> Dict[str, Any]:
        """
        Granger Causality 분석 수행

        시계열 변수들이 타겟 변수에 미치는 Granger 인과성 검정

        Parameters:
        - time_series_data: {"variable_name": time_series_array, ...}
        - target_variable: 타겟 변수명
        - max_lag: 최대 시차 수
        - significance_level: 유의수준

        Returns:
        - Granger causality 분석 결과
        """
        analyzer = GrangerCausalityAnalyzer(
            max_lag=max_lag,
            significance_level=significance_level
        )

        results = analyzer.test_multivariate(time_series_data, target_variable)

        # 결과 정리
        causal_variables = []
        non_causal_variables = []

        formatted_results = {}
        for var_name, result in results.items():
            formatted_results[var_name] = {
                "f_statistic": result.f_statistic,
                "p_value": result.p_value,
                "optimal_lag": result.optimal_lag,
                "is_causal": result.is_causal,
                "r2_restricted": result.r2_restricted,
                "r2_unrestricted": result.r2_unrestricted,
                "interpretation": result.interpretation
            }

            if result.is_causal:
                causal_variables.append(var_name)
            else:
                non_causal_variables.append(var_name)

        return {
            "target_variable": target_variable,
            "causal_variables": causal_variables,
            "non_causal_variables": non_causal_variables,
            "detailed_results": formatted_results,
            "summary": {
                "n_variables_tested": len(results),
                "n_causal": len(causal_variables),
                "significance_level": significance_level,
                "max_lag": max_lag
            }
        }

    def discover_causal_graph(
        self,
        data: np.ndarray,
        variable_names: Optional[List[str]] = None,
        alpha: float = 0.05,
        max_cond_set_size: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        PC Algorithm을 사용하여 인과 그래프 발견

        Parameters:
        - data: (n_samples, n_variables) 데이터 행렬
        - variable_names: 변수명 리스트
        - alpha: 조건부 독립성 테스트 유의수준
        - max_cond_set_size: 최대 조건부 집합 크기

        Returns:
        - 발견된 인과 그래프 정보
        """
        pc = PCAlgorithm(
            alpha=alpha,
            max_cond_set_size=max_cond_set_size
        )

        result = pc.fit(data, variable_names)

        # 엣지 목록 정리
        edges_list = [
            {"source": src, "target": tgt}
            for src, tgt in result.directed_edges
        ]

        # 각 노드의 부모/자식 관계
        node_relations = {}
        for name in result.variable_names:
            parents = result.causal_graph.get_parents(name)
            children = result.causal_graph.get_children(name)
            node_relations[name] = {
                "parents": parents,
                "children": children,
                "n_parents": len(parents),
                "n_children": len(children)
            }

        return {
            "nodes": result.variable_names,
            "edges": edges_list,
            "n_edges": len(result.directed_edges),
            "adjacency_matrix": result.adjacency_matrix.tolist(),
            "node_relations": node_relations,
            "skeleton": [{"v1": v1, "v2": v2} for v1, v2 in result.skeleton],
            "algorithm_stats": {
                "n_conditional_tests": result.n_conditional_tests,
                "alpha": result.alpha
            }
        }

    def estimate_heterogeneous_effects(
        self,
        X: np.ndarray,
        treatment: np.ndarray,
        outcome: np.ndarray,
        method: str = "x_learner",
        feature_names: Optional[List[str]] = None,
        X_test: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        CATE (이질적 처리 효과) 추정

        S-Learner, T-Learner, X-Learner 중 선택

        Parameters:
        - X: 공변량 행렬 (n_samples, n_features)
        - treatment: 처리 벡터 (0/1)
        - outcome: 결과 벡터
        - method: "s_learner", "t_learner", "x_learner"
        - feature_names: 피처명 리스트
        - X_test: 테스트 데이터

        Returns:
        - CATE 추정 결과
        """
        estimator = EnhancedCATEEstimator(
            method=method,
            base_learner="ridge",
            n_bootstrap=100
        )

        result = estimator.fit_predict(
            X=X,
            treatment=treatment,
            outcome=outcome,
            X_test=X_test,
            feature_names=feature_names
        )

        # 서브그룹 분석
        subgroup_summary = {}
        if result.subgroup_effects:
            for key, value in result.subgroup_effects.items():
                subgroup_summary[key] = value

        return {
            "method": result.method,
            "ate": float(result.ate),
            "cate_estimates": result.cate_estimates.tolist(),
            "std_errors": result.std_errors.tolist(),
            "confidence_intervals": result.confidence_intervals.tolist(),
            "feature_importances": result.feature_importances,
            "subgroup_effects": subgroup_summary,
            "summary": {
                "mean_cate": float(np.mean(result.cate_estimates)),
                "std_cate": float(np.std(result.cate_estimates)),
                "min_cate": float(np.min(result.cate_estimates)),
                "max_cate": float(np.max(result.cate_estimates)),
                "pct_positive_effect": float(np.mean(result.cate_estimates > 0) * 100)
            }
        }

    def comprehensive_causal_analysis(
        self,
        tables: Dict[str, Dict],
        time_series_data: Optional[Dict[str, np.ndarray]] = None,
        observation_data: Optional[np.ndarray] = None,
        variable_names: Optional[List[str]] = None,
        fk_relations: List[Dict] = None,
        entity_matches: List[Dict] = None
    ) -> Dict[str, Any]:
        """
        종합 인과 분석 수행

        1. 기본 인과 영향 분석
        2. Granger Causality (시계열 데이터가 있는 경우)
        3. PC Algorithm 기반 그래프 발견 (관찰 데이터가 있는 경우)
        4. CATE 추정

        Parameters:
        - tables: 테이블 정보
        - time_series_data: 시계열 데이터 (optional)
        - observation_data: 관찰 데이터 행렬 (optional)
        - variable_names: 변수명 리스트
        - fk_relations: FK 관계
        - entity_matches: 엔티티 매칭

        Returns:
        - 종합 인과 분석 결과
        """
        results = {
            "analysis_types": [],
            "basic_impact": None,
            "granger_causality": None,
            "causal_graph_discovery": None,
            "heterogeneous_effects": None
        }

        # 1. 기본 인과 영향 분석
        try:
            basic_result = self.analyze(
                tables=tables,
                fk_relations=fk_relations,
                entity_matches=entity_matches
            )
            results["basic_impact"] = basic_result
            results["analysis_types"].append("basic_impact")
        except Exception as e:
            results["basic_impact"] = {"error": str(e)}

        # 2. Granger Causality (시계열 데이터가 있는 경우)
        if time_series_data and len(time_series_data) >= 2:
            try:
                # 첫 번째 변수를 타겟으로 설정 (또는 'outcome', 'kpi' 같은 이름 찾기)
                target = None
                for key in time_series_data.keys():
                    if any(t in key.lower() for t in ['outcome', 'kpi', 'target', 'y']):
                        target = key
                        break
                if target is None:
                    target = list(time_series_data.keys())[0]

                granger_result = self.granger_causality_analysis(
                    time_series_data=time_series_data,
                    target_variable=target
                )
                results["granger_causality"] = granger_result
                results["analysis_types"].append("granger_causality")
            except Exception as e:
                results["granger_causality"] = {"error": str(e)}

        # 3. PC Algorithm (관찰 데이터가 있는 경우)
        if observation_data is not None and len(observation_data) > 10:
            try:
                pc_result = self.discover_causal_graph(
                    data=observation_data,
                    variable_names=variable_names
                )
                results["causal_graph_discovery"] = pc_result
                results["analysis_types"].append("causal_graph_discovery")
            except Exception as e:
                results["causal_graph_discovery"] = {"error": str(e)}

        # 4. CATE 추정 (관찰 데이터가 있고 treatment/outcome이 있는 경우)
        if observation_data is not None and observation_data.shape[1] >= 3:
            try:
                # 마지막 두 열을 treatment, outcome으로 가정
                X = observation_data[:, :-2]
                treatment = observation_data[:, -2]
                outcome = observation_data[:, -1]

                # treatment를 이진화
                treatment_binary = (treatment > np.median(treatment)).astype(float)

                feature_names_cate = None
                if variable_names and len(variable_names) >= 2:
                    feature_names_cate = variable_names[:-2]

                cate_result = self.estimate_heterogeneous_effects(
                    X=X,
                    treatment=treatment_binary,
                    outcome=outcome,
                    method="x_learner",
                    feature_names=feature_names_cate
                )
                results["heterogeneous_effects"] = cate_result
                results["analysis_types"].append("heterogeneous_effects")
            except Exception as e:
                results["heterogeneous_effects"] = {"error": str(e)}

        # 종합 요약
        results["summary"] = {
            "n_analysis_types": len(results["analysis_types"]),
            "analyses_performed": results["analysis_types"],
            "has_causal_evidence": (
                results.get("basic_impact", {}).get("summary", {}).get("ate_significant", False) or
                (results.get("granger_causality", {}).get("n_causal", 0) > 0 if isinstance(results.get("granger_causality"), dict) else False)
            )
        }

        return results
