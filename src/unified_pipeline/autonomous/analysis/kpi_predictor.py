"""
KPI Predictor Module - KPI 예측 및 시계열 분석

데이터 통합 전후의 KPI 변화를 예측합니다.

주요 기능:
- 시계열 예측 (ARIMA 스타일, Exponential Smoothing)
- ROI 예측 및 불확실성 정량화
- 다중 시나리오 분석
- 민감도 분석
- 목표 도달 시간 예측

References:
- Hyndman & Athanasopoulos, "Forecasting: Principles and Practice", 3rd ed.
- Box et al., "Time Series Analysis: Forecasting and Control"
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Callable
from enum import Enum
import numpy as np
from collections import defaultdict


class KPIType(Enum):
    """KPI 유형"""
    EFFICIENCY = "efficiency"                    # 운영 효율성
    QUALITY = "quality"                          # 데이터 품질
    COST = "cost"                                # 비용
    TIME = "time"                                # 시간/속도
    ACCURACY = "accuracy"                        # 정확도
    COVERAGE = "coverage"                        # 커버리지
    SATISFACTION = "satisfaction"                # 만족도


class ScenarioType(Enum):
    """시나리오 유형"""
    CONSERVATIVE = "conservative"                # 보수적
    MODERATE = "moderate"                        # 중도
    OPTIMISTIC = "optimistic"                    # 낙관적


@dataclass
class KPIDefinition:
    """KPI 정의"""
    name: str
    kpi_type: KPIType
    unit: str                                    # %, 원, 초, 건 등
    direction: str = "higher_better"             # higher_better, lower_better
    baseline: float = 0.0
    target: float = 1.0
    weight: float = 1.0                          # 중요도 가중치


@dataclass
class Forecast:
    """예측 결과"""
    values: np.ndarray                           # 예측값
    lower_bound: np.ndarray                      # 하한
    upper_bound: np.ndarray                      # 상한
    timestamps: List[int]                        # 시점 (월 또는 분기)
    confidence_level: float = 0.95


@dataclass
class KPIForecast:
    """KPI별 예측 결과"""
    kpi_name: str
    kpi_type: KPIType
    baseline: float
    forecast: Forecast
    expected_improvement: float
    time_to_target: Optional[int]                # 목표 도달 예상 시간 (월)
    roi_estimate: float
    confidence: float


@dataclass
class ScenarioAnalysis:
    """시나리오 분석 결과"""
    scenario_type: ScenarioType
    assumptions: Dict[str, float]
    forecasts: Dict[str, KPIForecast]
    total_roi: float
    probability: float
    risk_factors: List[str]


@dataclass
class SensitivityAnalysis:
    """민감도 분석 결과"""
    kpi_name: str
    base_value: float
    sensitivities: Dict[str, float]              # 변수별 민감도
    tornado_chart: List[Tuple[str, float, float]]  # (변수명, 하한영향, 상한영향)
    critical_variables: List[str]


@dataclass
class ROIProjection:
    """ROI 프로젝션"""
    initial_investment: float
    monthly_costs: List[float]
    monthly_benefits: List[float]
    cumulative_roi: List[float]
    break_even_month: Optional[int]
    npv: float                                   # Net Present Value
    irr: float                                   # Internal Rate of Return


class ExponentialSmoothing:
    """
    지수 평활법 (Holt-Winters)

    트렌드와 계절성을 고려한 시계열 예측
    """

    def __init__(
        self,
        alpha: float = 0.3,     # 레벨 평활 계수
        beta: float = 0.1,      # 트렌드 평활 계수
        gamma: float = 0.1,     # 계절 평활 계수
        seasonal_period: int = 12
    ):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.seasonal_period = seasonal_period

        self.level = 0.0
        self.trend = 0.0
        self.seasonals: List[float] = []

    def fit(self, data: np.ndarray) -> 'ExponentialSmoothing':
        """모델 학습"""
        n = len(data)
        if n < 2:
            self.level = data[0] if n > 0 else 0
            self.trend = 0
            self.seasonals = [1.0] * self.seasonal_period
            return self

        # 초기화
        self.level = np.mean(data[:self.seasonal_period]) if n >= self.seasonal_period else np.mean(data)
        self.trend = (np.mean(data[self.seasonal_period:2*self.seasonal_period]) -
                     np.mean(data[:self.seasonal_period])) / self.seasonal_period if n >= 2*self.seasonal_period else 0

        # 계절 성분 초기화
        if n >= self.seasonal_period:
            self.seasonals = list(data[:self.seasonal_period] / self.level)
        else:
            self.seasonals = [1.0] * self.seasonal_period

        # 업데이트
        for i in range(n):
            season_idx = i % self.seasonal_period
            value = data[i]

            # 레벨 업데이트
            old_level = self.level
            self.level = self.alpha * (value / self.seasonals[season_idx]) + \
                        (1 - self.alpha) * (self.level + self.trend)

            # 트렌드 업데이트
            self.trend = self.beta * (self.level - old_level) + \
                        (1 - self.beta) * self.trend

            # 계절 성분 업데이트
            self.seasonals[season_idx] = self.gamma * (value / self.level) + \
                                         (1 - self.gamma) * self.seasonals[season_idx]

        return self

    def forecast(self, steps: int) -> np.ndarray:
        """미래 예측"""
        predictions = []
        for h in range(1, steps + 1):
            season_idx = h % self.seasonal_period
            pred = (self.level + h * self.trend) * self.seasonals[season_idx]
            predictions.append(pred)
        return np.array(predictions)


class ARIMASimple:
    """
    간단한 ARIMA(p,d,q) 구현

    자기회귀(AR) + 적분(I) + 이동평균(MA)
    """

    def __init__(self, p: int = 2, d: int = 1, q: int = 1):
        self.p = p  # AR 차수
        self.d = d  # 차분 차수
        self.q = q  # MA 차수

        self.ar_coeffs: Optional[np.ndarray] = None
        self.ma_coeffs: Optional[np.ndarray] = None
        self.residuals: Optional[np.ndarray] = None
        self.last_values: Optional[np.ndarray] = None

    def _difference(self, data: np.ndarray, d: int) -> np.ndarray:
        """차분"""
        result = data.copy()
        for _ in range(d):
            result = np.diff(result)
        return result

    def _inverse_difference(
        self,
        diff_data: np.ndarray,
        last_values: np.ndarray
    ) -> np.ndarray:
        """역차분"""
        result = diff_data.copy()
        for last_val in reversed(last_values):
            result = np.cumsum(np.concatenate([[last_val], result]))
        return result

    def fit(self, data: np.ndarray) -> 'ARIMASimple':
        """모델 학습"""
        if len(data) < self.p + self.d + 2:
            # 데이터 부족시 단순 평균
            self.ar_coeffs = np.array([1.0 / self.p] * self.p) if self.p > 0 else np.array([])
            self.ma_coeffs = np.zeros(self.q)
            self.residuals = np.zeros(self.q)
            self.last_values = data[-self.d:] if self.d > 0 else np.array([])
            return self

        # 차분
        diff_data = self._difference(data, self.d)
        self.last_values = data[-(self.d):] if self.d > 0 else np.array([])

        # AR 계수 추정 (Yule-Walker)
        if self.p > 0 and len(diff_data) > self.p:
            # 자기상관 계산
            acf = np.correlate(diff_data - np.mean(diff_data),
                              diff_data - np.mean(diff_data), 'full')
            acf = acf[len(acf)//2:] / acf[len(acf)//2]

            # Yule-Walker 방정식
            r = acf[1:self.p+1]
            R = np.array([[acf[abs(i-j)] for j in range(self.p)] for i in range(self.p)])

            try:
                self.ar_coeffs = np.linalg.solve(R, r)
            except np.linalg.LinAlgError:
                self.ar_coeffs = np.ones(self.p) / self.p
        else:
            self.ar_coeffs = np.array([])

        # AR 잔차 계산
        if len(self.ar_coeffs) > 0:
            fitted = np.zeros(len(diff_data))
            for i in range(self.p, len(diff_data)):
                fitted[i] = np.dot(self.ar_coeffs, diff_data[i-self.p:i][::-1])
            self.residuals = diff_data[self.p:] - fitted[self.p:]
        else:
            self.residuals = diff_data

        # MA 계수 (단순 추정)
        if self.q > 0 and len(self.residuals) > self.q:
            self.ma_coeffs = np.array([0.3 ** (i+1) for i in range(self.q)])
        else:
            self.ma_coeffs = np.array([])

        return self

    def forecast(self, steps: int, data: np.ndarray = None) -> np.ndarray:
        """미래 예측"""
        if data is None:
            return np.zeros(steps)

        # 차분 데이터
        diff_data = self._difference(data, self.d)

        predictions = []
        current_diff = list(diff_data[-self.p:]) if self.p > 0 else []
        current_residuals = list(self.residuals[-self.q:]) if self.q > 0 else []

        for _ in range(steps):
            # AR 부분
            ar_part = 0.0
            if len(self.ar_coeffs) > 0 and len(current_diff) >= self.p:
                ar_part = np.dot(self.ar_coeffs, current_diff[-self.p:][::-1])

            # MA 부분
            ma_part = 0.0
            if len(self.ma_coeffs) > 0 and len(current_residuals) >= self.q:
                ma_part = np.dot(self.ma_coeffs, current_residuals[-self.q:][::-1])

            pred = ar_part + ma_part
            predictions.append(pred)

            # 업데이트
            current_diff.append(pred)
            current_residuals.append(0)  # 미래 잔차는 0으로 가정

        predictions = np.array(predictions)

        # 역차분
        if self.d > 0:
            predictions = self._inverse_difference(predictions, self.last_values)

        return predictions[:steps]


class KPIModel:
    """KPI 모델링 및 예측"""

    def __init__(self, kpi_definition: KPIDefinition):
        self.kpi = kpi_definition
        self.es_model = ExponentialSmoothing()
        self.arima_model = ARIMASimple()

    def fit(self, historical_data: np.ndarray) -> 'KPIModel':
        """과거 데이터로 모델 학습"""
        if len(historical_data) >= 12:
            self.es_model.fit(historical_data)
        if len(historical_data) >= 5:
            self.arima_model.fit(historical_data)
        return self

    def forecast(
        self,
        steps: int,
        historical_data: np.ndarray = None,
        improvement_factor: float = 1.0
    ) -> Forecast:
        """KPI 예측"""
        # 기본 예측
        if historical_data is not None and len(historical_data) >= 5:
            es_pred = self.es_model.forecast(steps)
            arima_pred = self.arima_model.forecast(steps, historical_data)

            # 앙상블 (가중 평균)
            base_pred = 0.6 * es_pred + 0.4 * arima_pred
        else:
            # 데이터 부족시 선형 외삽
            base_pred = np.linspace(
                self.kpi.baseline,
                self.kpi.baseline * (1 + 0.1 * improvement_factor),
                steps
            )

        # 개선 요소 적용
        improved_pred = base_pred * improvement_factor

        # 목표 방향 조정
        if self.kpi.direction == "lower_better":
            # 비용 등 낮을수록 좋은 KPI
            improved_pred = base_pred / improvement_factor

        # 불확실성 구간
        uncertainty = np.abs(improved_pred) * 0.1 * np.sqrt(np.arange(1, steps + 1))
        lower = improved_pred - 1.96 * uncertainty
        upper = improved_pred + 1.96 * uncertainty

        return Forecast(
            values=improved_pred,
            lower_bound=lower,
            upper_bound=upper,
            timestamps=list(range(1, steps + 1)),
            confidence_level=0.95
        )


class ScenarioBuilder:
    """시나리오 구축"""

    @staticmethod
    def build_scenarios(
        base_assumptions: Dict[str, float]
    ) -> Dict[ScenarioType, Dict[str, float]]:
        """보수적/중도/낙관적 시나리오 생성"""
        scenarios = {}

        # 보수적 시나리오: 기대치의 70%
        scenarios[ScenarioType.CONSERVATIVE] = {
            k: v * 0.7 for k, v in base_assumptions.items()
        }

        # 중도 시나리오: 기대치 그대로
        scenarios[ScenarioType.MODERATE] = base_assumptions.copy()

        # 낙관적 시나리오: 기대치의 130%
        scenarios[ScenarioType.OPTIMISTIC] = {
            k: v * 1.3 for k, v in base_assumptions.items()
        }

        return scenarios


class ROICalculator:
    """ROI 계산기"""

    def __init__(self, discount_rate: float = 0.08):
        """
        Parameters:
        - discount_rate: 할인율 (연간, 기본 8%)
        """
        self.discount_rate = discount_rate
        self.monthly_rate = (1 + discount_rate) ** (1/12) - 1

    def calculate(
        self,
        initial_investment: float,
        monthly_costs: List[float],
        monthly_benefits: List[float]
    ) -> ROIProjection:
        """ROI 계산"""
        months = len(monthly_costs)

        # 월별 순현금흐름
        net_cashflows = np.array(monthly_benefits) - np.array(monthly_costs)

        # 누적 현금흐름
        cumulative = np.cumsum(net_cashflows) - initial_investment

        # NPV 계산
        discount_factors = [(1 + self.monthly_rate) ** -i for i in range(1, months + 1)]
        npv = -initial_investment + np.sum(net_cashflows * discount_factors)

        # 누적 ROI
        cumulative_roi = [(cum / initial_investment * 100) if initial_investment > 0 else 0
                         for cum in cumulative]

        # 손익분기점
        break_even = None
        for i, cum in enumerate(cumulative):
            if cum >= 0:
                break_even = i + 1
                break

        # IRR 추정 (간단한 이진 탐색)
        irr = self._estimate_irr(initial_investment, net_cashflows)

        return ROIProjection(
            initial_investment=initial_investment,
            monthly_costs=list(monthly_costs),
            monthly_benefits=list(monthly_benefits),
            cumulative_roi=cumulative_roi,
            break_even_month=break_even,
            npv=npv,
            irr=irr
        )

    def _estimate_irr(
        self,
        initial_investment: float,
        cashflows: np.ndarray,
        max_iterations: int = 100
    ) -> float:
        """IRR 추정 (이진 탐색)"""
        def npv_at_rate(rate):
            monthly_rate = (1 + rate) ** (1/12) - 1
            discount_factors = [(1 + monthly_rate) ** -i for i in range(1, len(cashflows) + 1)]
            return -initial_investment + np.sum(cashflows * discount_factors)

        low, high = -0.5, 2.0

        for _ in range(max_iterations):
            mid = (low + high) / 2
            npv = npv_at_rate(mid)

            if abs(npv) < 0.01:
                return mid
            elif npv > 0:
                low = mid
            else:
                high = mid

        return (low + high) / 2


class SensitivityCalculator:
    """민감도 분석"""

    def analyze(
        self,
        kpi_name: str,
        base_value: float,
        variables: Dict[str, float],
        impact_function: Callable[[Dict[str, float]], float]
    ) -> SensitivityAnalysis:
        """
        일원 민감도 분석

        각 변수를 ±20% 변화시켜 결과에 미치는 영향 분석
        """
        sensitivities = {}
        tornado_data = []

        for var_name, var_value in variables.items():
            # -20% 변화
            low_vars = variables.copy()
            low_vars[var_name] = var_value * 0.8
            low_result = impact_function(low_vars)

            # +20% 변화
            high_vars = variables.copy()
            high_vars[var_name] = var_value * 1.2
            high_result = impact_function(high_vars)

            # 민감도 = 결과 변화율 / 입력 변화율
            sensitivity = (high_result - low_result) / (0.4 * var_value) if var_value != 0 else 0
            sensitivities[var_name] = sensitivity

            # Tornado 데이터
            tornado_data.append((
                var_name,
                low_result - base_value,
                high_result - base_value
            ))

        # 영향력 순 정렬
        tornado_data.sort(key=lambda x: abs(x[2] - x[1]), reverse=True)

        # 주요 변수 식별
        critical = [t[0] for t in tornado_data[:3]]

        return SensitivityAnalysis(
            kpi_name=kpi_name,
            base_value=base_value,
            sensitivities=sensitivities,
            tornado_chart=tornado_data,
            critical_variables=critical
        )


class KPIPredictor:
    """통합 KPI 예측기"""

    def __init__(self):
        self.kpi_definitions: Dict[str, KPIDefinition] = {}
        self.kpi_models: Dict[str, KPIModel] = {}
        self.roi_calculator = ROICalculator()
        self.sensitivity_calculator = SensitivityCalculator()

        # 기본 KPI 정의
        self._setup_default_kpis()

    def _setup_default_kpis(self):
        """기본 KPI 정의 설정"""
        default_kpis = [
            KPIDefinition(
                name="data_quality_score",
                kpi_type=KPIType.QUALITY,
                unit="%",
                direction="higher_better",
                baseline=0.65,
                target=0.90,
                weight=1.0
            ),
            KPIDefinition(
                name="operational_efficiency",
                kpi_type=KPIType.EFFICIENCY,
                unit="%",
                direction="higher_better",
                baseline=0.60,
                target=0.85,
                weight=0.9
            ),
            KPIDefinition(
                name="query_response_time",
                kpi_type=KPIType.TIME,
                unit="초",
                direction="lower_better",
                baseline=2.5,
                target=0.5,
                weight=0.7
            ),
            KPIDefinition(
                name="data_integration_coverage",
                kpi_type=KPIType.COVERAGE,
                unit="%",
                direction="higher_better",
                baseline=0.40,
                target=0.95,
                weight=0.8
            ),
            KPIDefinition(
                name="decision_accuracy",
                kpi_type=KPIType.ACCURACY,
                unit="%",
                direction="higher_better",
                baseline=0.70,
                target=0.90,
                weight=1.0
            ),
            KPIDefinition(
                name="maintenance_cost",
                kpi_type=KPIType.COST,
                unit="원",
                direction="lower_better",
                baseline=10000000,
                target=5000000,
                weight=0.6
            )
        ]

        for kpi_def in default_kpis:
            self.kpi_definitions[kpi_def.name] = kpi_def
            self.kpi_models[kpi_def.name] = KPIModel(kpi_def)

    def predict_kpi(
        self,
        kpi_name: str,
        months: int = 12,
        improvement_factor: float = 1.15,
        historical_data: np.ndarray = None
    ) -> KPIForecast:
        """단일 KPI 예측"""
        if kpi_name not in self.kpi_definitions:
            # 새 KPI 생성
            self.kpi_definitions[kpi_name] = KPIDefinition(
                name=kpi_name,
                kpi_type=KPIType.EFFICIENCY,
                unit="%",
                baseline=0.5,
                target=0.8
            )
            self.kpi_models[kpi_name] = KPIModel(self.kpi_definitions[kpi_name])

        kpi_def = self.kpi_definitions[kpi_name]
        model = self.kpi_models[kpi_name]

        # 과거 데이터로 학습
        if historical_data is not None:
            model.fit(historical_data)

        # 예측
        forecast = model.forecast(months, historical_data, improvement_factor)

        # 기대 개선율
        if kpi_def.direction == "higher_better":
            expected_improvement = (forecast.values[-1] - kpi_def.baseline) / kpi_def.baseline
        else:
            expected_improvement = (kpi_def.baseline - forecast.values[-1]) / kpi_def.baseline

        # 목표 도달 시간
        time_to_target = None
        for i, val in enumerate(forecast.values):
            if kpi_def.direction == "higher_better" and val >= kpi_def.target:
                time_to_target = i + 1
                break
            elif kpi_def.direction == "lower_better" and val <= kpi_def.target:
                time_to_target = i + 1
                break

        # ROI 추정 (단순화)
        roi_estimate = expected_improvement * kpi_def.weight

        # 신뢰도 (예측 구간 기반)
        uncertainty_ratio = np.mean(
            (forecast.upper_bound - forecast.lower_bound) / (np.abs(forecast.values) + 1e-10)
        )
        confidence = max(0, 1 - uncertainty_ratio)

        return KPIForecast(
            kpi_name=kpi_name,
            kpi_type=kpi_def.kpi_type,
            baseline=kpi_def.baseline,
            forecast=forecast,
            expected_improvement=expected_improvement,
            time_to_target=time_to_target,
            roi_estimate=roi_estimate,
            confidence=confidence
        )

    def predict_all_kpis(
        self,
        months: int = 12,
        improvement_factors: Dict[str, float] = None
    ) -> Dict[str, KPIForecast]:
        """모든 KPI 예측"""
        if improvement_factors is None:
            improvement_factors = {k: 1.15 for k in self.kpi_definitions}

        forecasts = {}
        for kpi_name in self.kpi_definitions:
            factor = improvement_factors.get(kpi_name, 1.15)
            forecasts[kpi_name] = self.predict_kpi(kpi_name, months, factor)

        return forecasts

    def analyze_scenarios(
        self,
        months: int = 12,
        base_improvement: float = 0.15
    ) -> Dict[ScenarioType, ScenarioAnalysis]:
        """시나리오 분석"""
        base_assumptions = {
            "integration_completeness": base_improvement,
            "data_quality_improvement": base_improvement * 0.8,
            "process_automation": base_improvement * 0.6,
            "query_optimization": base_improvement * 1.2
        }

        scenarios = ScenarioBuilder.build_scenarios(base_assumptions)
        results = {}

        for scenario_type, assumptions in scenarios.items():
            # 시나리오별 개선 계수
            improvement_factors = {
                "data_quality_score": 1 + assumptions["data_quality_improvement"],
                "operational_efficiency": 1 + assumptions["process_automation"],
                "query_response_time": 1 / (1 + assumptions["query_optimization"]),
                "data_integration_coverage": 1 + assumptions["integration_completeness"],
                "decision_accuracy": 1 + assumptions["data_quality_improvement"] * 0.5,
                "maintenance_cost": 1 / (1 + assumptions["process_automation"] * 0.5)
            }

            # KPI 예측
            forecasts = {}
            for kpi_name in self.kpi_definitions:
                factor = improvement_factors.get(kpi_name, 1.0)
                forecasts[kpi_name] = self.predict_kpi(kpi_name, months, factor)

            # 총 ROI
            total_roi = sum(f.roi_estimate for f in forecasts.values())

            # 시나리오 확률 및 리스크
            if scenario_type == ScenarioType.CONSERVATIVE:
                probability = 0.7
                risk_factors = ["외부 요인 영향 최소화", "점진적 개선"]
            elif scenario_type == ScenarioType.MODERATE:
                probability = 0.6
                risk_factors = ["일부 기술적 도전", "예상 일정 준수 필요"]
            else:  # OPTIMISTIC
                probability = 0.3
                risk_factors = ["높은 기술적 리스크", "조직 변화 관리 필요", "외부 지원 필요 가능성"]

            results[scenario_type] = ScenarioAnalysis(
                scenario_type=scenario_type,
                assumptions=assumptions,
                forecasts=forecasts,
                total_roi=total_roi,
                probability=probability,
                risk_factors=risk_factors
            )

        return results

    def calculate_roi_projection(
        self,
        initial_investment: float = 100000000,  # 1억원
        monthly_operation_cost: float = 5000000,  # 월 500만원
        expected_monthly_benefit: float = 15000000,  # 월 1500만원
        months: int = 24
    ) -> ROIProjection:
        """ROI 프로젝션"""
        # 비용은 점진적 감소 (효율화)
        monthly_costs = [
            monthly_operation_cost * (1 - 0.02 * min(i, 12))
            for i in range(months)
        ]

        # 혜택은 점진적 증가 (성숙)
        monthly_benefits = [
            expected_monthly_benefit * (0.5 + 0.5 * min(i / 6, 1))
            for i in range(months)
        ]

        return self.roi_calculator.calculate(
            initial_investment,
            monthly_costs,
            monthly_benefits
        )

    def perform_sensitivity_analysis(
        self,
        kpi_name: str
    ) -> SensitivityAnalysis:
        """민감도 분석 수행"""
        if kpi_name not in self.kpi_definitions:
            return None

        kpi_def = self.kpi_definitions[kpi_name]

        # 분석 대상 변수
        variables = {
            "integration_completeness": 0.15,
            "data_quality_improvement": 0.12,
            "process_automation": 0.10,
            "adoption_rate": 0.80,
            "technical_success_rate": 0.85
        }

        # 영향 함수
        def impact_function(vars: Dict[str, float]) -> float:
            improvement = (
                vars["integration_completeness"] * 0.3 +
                vars["data_quality_improvement"] * 0.25 +
                vars["process_automation"] * 0.2 +
                vars["adoption_rate"] * 0.15 +
                vars["technical_success_rate"] * 0.1
            )
            return kpi_def.baseline * (1 + improvement)

        base_value = impact_function(variables)

        return self.sensitivity_calculator.analyze(
            kpi_name,
            base_value,
            variables,
            impact_function
        )

    def generate_summary(
        self,
        forecasts: Dict[str, KPIForecast],
        roi: ROIProjection = None
    ) -> Dict[str, Any]:
        """예측 요약 생성"""
        # 주요 KPI 개선
        improvements = {
            kpi_name: {
                "baseline": f.baseline,
                "predicted_final": float(f.forecast.values[-1]),
                "improvement": f"{f.expected_improvement * 100:.1f}%",
                "time_to_target": f"{f.time_to_target}개월" if f.time_to_target else "목표 미달성",
                "confidence": f"{f.confidence * 100:.0f}%"
            }
            for kpi_name, f in forecasts.items()
        }

        # 종합 점수
        weighted_improvement = sum(
            f.expected_improvement * self.kpi_definitions[kpi_name].weight
            for kpi_name, f in forecasts.items()
        ) / sum(d.weight for d in self.kpi_definitions.values())

        summary = {
            "kpi_improvements": improvements,
            "weighted_average_improvement": f"{weighted_improvement * 100:.1f}%",
            "highest_improvement_kpi": max(
                forecasts.items(),
                key=lambda x: x[1].expected_improvement
            )[0],
            "most_challenging_kpi": min(
                forecasts.items(),
                key=lambda x: x[1].confidence
            )[0]
        }

        if roi:
            summary["roi"] = {
                "npv": f"{roi.npv:,.0f}원",
                "irr": f"{roi.irr * 100:.1f}%",
                "break_even": f"{roi.break_even_month}개월" if roi.break_even_month else "미달성",
                "final_cumulative_roi": f"{roi.cumulative_roi[-1]:.1f}%" if roi.cumulative_roi else "N/A"
            }

        return summary

    # =========================================================================
    # 편의 메서드: governance.py 에이전트에서 사용하는 API
    # =========================================================================

    def predict_kpis(
        self,
        sample_data: Dict[str, List[Dict[str, Any]]],
        months: int = 12
    ) -> List[Dict[str, Any]]:
        """
        샘플 데이터 기반 KPI 예측 (에이전트 호출용 편의 메서드)

        Args:
            sample_data: {table_name: [row_dicts]}
            months: 예측 기간

        Returns:
            KPI 예측 결과 리스트
        """
        # 테이블 수에서 개선 계수 추정
        n_tables = len(sample_data)
        base_factor = 1.1 + 0.02 * min(n_tables, 10)

        results = []
        for kpi_name in self.kpi_definitions:
            try:
                forecast = self.predict_kpi(kpi_name, months, base_factor)
                results.append({
                    "kpi_name": forecast.kpi_name,
                    "kpi_type": forecast.kpi_type.value,
                    "baseline": forecast.baseline,
                    "predicted_final": float(forecast.forecast.values[-1]),
                    "expected_improvement": forecast.expected_improvement,
                    "time_to_target": forecast.time_to_target,
                    "confidence": forecast.confidence,
                    "roi_estimate": forecast.roi_estimate,
                })
            except Exception:
                continue

        return results

    def analyze_roi(
        self,
        governance_decisions: List[Any],
        action_backlog: List[Dict[str, Any]],
    ) -> 'ROIProjection':
        """
        거버넌스 결정 및 액션 백로그 기반 ROI 분석 (에이전트 호출용)

        Args:
            governance_decisions: 거버넌스 결정 목록
            action_backlog: 액션 백로그

        Returns:
            ROIProjection 객체
        """
        # 결정 수와 액션 수에 따른 투자/효익 추정
        n_decisions = len(governance_decisions) if governance_decisions else 1
        n_actions = len(action_backlog) if action_backlog else 0

        # 초기 투자 추정 (의사결정 당 기본 비용)
        initial_investment = 50_000_000 + n_decisions * 10_000_000  # 기본 5천만 + 결정당 1천만

        # 월간 운영 비용 (액션 수에 비례)
        monthly_cost = 3_000_000 + n_actions * 500_000

        # 월간 효익 (통합 효과)
        monthly_benefit = n_decisions * 5_000_000

        return self.calculate_roi_projection(
            initial_investment=initial_investment,
            monthly_operation_cost=monthly_cost,
            expected_monthly_benefit=monthly_benefit,
            months=24
        )

    def calculate_domain_kpis(
        self,
        sample_data: Dict[str, List[Dict[str, Any]]],
        domain: str = "general"
    ) -> List[Dict[str, Any]]:
        """
        도메인별 KPI 계산 (에이전트 호출용)

        Args:
            sample_data: {table_name: [row_dicts]}
            domain: 도메인 이름

        Returns:
            도메인 특화 KPI 리스트
        """
        # 도메인별 KPI 정의
        domain_kpi_map = {
            "healthcare": [
                {"name": "patient_data_completeness", "weight": 1.0, "baseline": 0.7, "target": 0.95},
                {"name": "clinical_data_accuracy", "weight": 0.9, "baseline": 0.8, "target": 0.98},
                {"name": "care_coordination_efficiency", "weight": 0.8, "baseline": 0.6, "target": 0.85},
            ],
            "finance": [
                {"name": "transaction_accuracy", "weight": 1.0, "baseline": 0.95, "target": 0.999},
                {"name": "risk_detection_rate", "weight": 0.9, "baseline": 0.7, "target": 0.9},
                {"name": "compliance_score", "weight": 0.95, "baseline": 0.8, "target": 0.95},
            ],
            "retail": [
                {"name": "inventory_accuracy", "weight": 1.0, "baseline": 0.85, "target": 0.98},
                {"name": "customer_data_quality", "weight": 0.8, "baseline": 0.7, "target": 0.9},
                {"name": "order_fulfillment_rate", "weight": 0.9, "baseline": 0.9, "target": 0.98},
            ],
            "manufacturing": [
                {"name": "production_data_accuracy", "weight": 1.0, "baseline": 0.8, "target": 0.95},
                {"name": "supply_chain_visibility", "weight": 0.9, "baseline": 0.6, "target": 0.85},
                {"name": "quality_control_coverage", "weight": 0.85, "baseline": 0.75, "target": 0.95},
            ],
        }

        # 도메인에 해당하는 KPI 또는 기본 KPI
        kpis = domain_kpi_map.get(domain, [
            {"name": "data_quality_score", "weight": 1.0, "baseline": 0.65, "target": 0.9},
            {"name": "operational_efficiency", "weight": 0.9, "baseline": 0.6, "target": 0.85},
            {"name": "integration_coverage", "weight": 0.8, "baseline": 0.5, "target": 0.9},
        ])

        # 샘플 데이터 기반 현재 수치 계산 (간단한 추정)
        n_tables = len(sample_data)
        total_rows = sum(len(rows) for rows in sample_data.values())
        completeness_factor = min(1.0, total_rows / 1000)  # 1000행 기준 정규화

        results = []
        for kpi in kpis:
            # 현재 값 추정
            current = kpi["baseline"] + (kpi["target"] - kpi["baseline"]) * completeness_factor * 0.3
            gap = kpi["target"] - current

            results.append({
                "name": kpi["name"],
                "domain": domain,
                "weight": kpi["weight"],
                "baseline": kpi["baseline"],
                "current_estimate": round(current, 3),
                "target": kpi["target"],
                "gap": round(gap, 3),
                "improvement_potential": round(gap / kpi["baseline"] * 100, 1) if kpi["baseline"] > 0 else 0,
            })

        return results


class IntegrationKPIAnalyzer:
    """데이터 통합 KPI 분석기"""

    def __init__(self):
        self.predictor = KPIPredictor()

    def analyze(
        self,
        tables: Dict[str, Dict],
        integration_plan: Dict[str, Any] = None,
        months: int = 12
    ) -> Dict[str, Any]:
        """
        데이터 통합에 따른 KPI 영향 분석

        Parameters:
        - tables: 테이블 정보
        - integration_plan: 통합 계획
        - months: 예측 기간 (월)

        Returns:
        - KPI 예측 및 분석 결과
        """
        # 통합 계획에서 개선 계수 추정
        improvement_factors = self._estimate_improvement_factors(
            tables,
            integration_plan or {}
        )

        # KPI 예측
        forecasts = {}
        for kpi_name in self.predictor.kpi_definitions:
            factor = improvement_factors.get(kpi_name, 1.1)
            forecasts[kpi_name] = self.predictor.predict_kpi(
                kpi_name, months, factor
            )

        # 시나리오 분석
        scenarios = self.predictor.analyze_scenarios(months)

        # ROI 프로젝션
        roi = self.predictor.calculate_roi_projection()

        # 민감도 분석
        sensitivity = self.predictor.perform_sensitivity_analysis("operational_efficiency")

        # 요약
        summary = self.predictor.generate_summary(forecasts, roi)

        return {
            "forecasts": {
                kpi_name: {
                    "values": list(f.forecast.values),
                    "lower_bound": list(f.forecast.lower_bound),
                    "upper_bound": list(f.forecast.upper_bound),
                    "expected_improvement": f.expected_improvement,
                    "time_to_target": f.time_to_target,
                    "confidence": f.confidence
                }
                for kpi_name, f in forecasts.items()
            },
            "scenarios": {
                scenario_type.value: {
                    "probability": analysis.probability,
                    "total_roi": analysis.total_roi,
                    "risk_factors": analysis.risk_factors,
                    "assumptions": analysis.assumptions
                }
                for scenario_type, analysis in scenarios.items()
            },
            "roi_projection": {
                "npv": roi.npv,
                "irr": roi.irr,
                "break_even_month": roi.break_even_month,
                "cumulative_roi": roi.cumulative_roi
            },
            "sensitivity": {
                "kpi": sensitivity.kpi_name if sensitivity else None,
                "critical_variables": sensitivity.critical_variables if sensitivity else [],
                "tornado_chart": sensitivity.tornado_chart if sensitivity else []
            },
            "summary": summary
        }

    def _estimate_improvement_factors(
        self,
        tables: Dict[str, Dict],
        plan: Dict[str, Any]
    ) -> Dict[str, float]:
        """통합 계획에서 개선 계수 추정"""
        n_tables = len(tables) if tables else 1
        n_fks = plan.get("n_fk_relations", 0)
        n_entities = plan.get("n_entity_matches", 0)

        # 복잡도에 따른 개선 잠재력
        base_improvement = min(0.05 + 0.02 * n_tables + 0.05 * n_fks + 0.03 * n_entities, 0.3)

        return {
            "data_quality_score": 1 + base_improvement * 1.2,
            "operational_efficiency": 1 + base_improvement,
            "query_response_time": 1 / (1 + base_improvement * 0.5),
            "data_integration_coverage": 1 + base_improvement * 1.5,
            "decision_accuracy": 1 + base_improvement * 0.8,
            "maintenance_cost": 1 / (1 + base_improvement * 0.6)
        }
