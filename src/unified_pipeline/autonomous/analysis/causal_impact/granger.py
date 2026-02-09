"""
Granger Causality Analysis - 시계열 인과관계 분석

GrangerCausalityAnalyzer

References:
- Granger, C. W. J. (1969). "Investigating Causal Relations by Econometric Models"
"""

from typing import Dict, List, Optional, Tuple
import numpy as np

from .models import GrangerCausalityResult


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
