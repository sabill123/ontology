"""
Causal Impact Estimators - 인과 효과 추정기

PropensityScoreEstimator, OutcomeModel, ATEEstimator, CATEEstimator, EnhancedCATEEstimator
"""

from typing import Dict, List, Optional, Tuple, Callable
import numpy as np

from .models import (
    TreatmentEffect,
    EffectType,
    CATEResult,
)


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
