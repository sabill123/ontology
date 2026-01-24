"""
Time-Series Forecasting Module (Phase 2 Enhanced)

시계열 예측 모듈:
- ARIMA/SARIMA for statistical forecasting
- Auto ARIMA (pmdarima integration)
- Prophet for trend + seasonality (Holiday effects, Custom seasonality)
- LSTM for complex patterns
- Ensemble prediction with weighted stacking
- Bootstrap Confidence Intervals
- ACF-based automatic seasonality detection

References:
- Box & Jenkins "Time Series Analysis" (1970)
- Taylor & Letham "Forecasting at Scale" (Prophet, 2017)
- Hochreiter & Schmidhuber "LSTM" (1997)
- Hyndman & Khandakar "Automatic Time Series Forecasting" (2008)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from enum import Enum
import logging
import warnings
import random
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# Optional imports with availability flags
NUMPY_AVAILABLE = False
PANDAS_AVAILABLE = False
STATSMODELS_AVAILABLE = False
PROPHET_AVAILABLE = False
SCIPY_AVAILABLE = False
PMDARIMA_AVAILABLE = False
SKLEARN_AVAILABLE = False

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    np = None

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    pd = None

try:
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.stattools import adfuller, acf, pacf
    from statsmodels.tsa.seasonal import seasonal_decompose
    STATSMODELS_AVAILABLE = True
except ImportError:
    ARIMA = None
    adfuller = None
    acf = None
    pacf = None
    seasonal_decompose = None

try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    Prophet = None

try:
    from scipy import stats
    from scipy.signal import find_peaks
    SCIPY_AVAILABLE = True
except ImportError:
    stats = None
    find_peaks = None

try:
    import pmdarima as pm
    from pmdarima import auto_arima
    PMDARIMA_AVAILABLE = True
except ImportError:
    pm = None
    auto_arima = None

try:
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    Ridge = None
    StandardScaler = None


class ForecastModel(str, Enum):
    """예측 모델 종류"""
    ARIMA = "arima"
    AUTO_ARIMA = "auto_arima"  # pmdarima integration
    SARIMA = "sarima"
    PROPHET = "prophet"
    PROPHET_ADVANCED = "prophet_advanced"  # Holiday + Custom seasonality
    LSTM = "lstm"
    ENSEMBLE = "ensemble"
    WEIGHTED_ENSEMBLE = "weighted_ensemble"  # Stacking ensemble
    SIMPLE_MA = "simple_ma"  # Fallback


@dataclass
class ForecastResult:
    """예측 결과"""
    series_name: str
    horizon: int
    predictions: List[float]
    lower_bound: List[float]  # 95% CI
    upper_bound: List[float]  # 95% CI
    model_used: str
    metrics: Dict[str, float] = field(default_factory=dict)  # MAE, RMSE, MAPE
    seasonality: Optional[Dict[str, Any]] = None
    trend: Optional[str] = None  # "increasing", "decreasing", "stable"
    decomposition: Optional[Dict[str, Any]] = None
    model_params: Optional[Dict[str, Any]] = None
    bootstrap_ci: Optional[Dict[str, List[float]]] = None  # Bootstrap CI results
    ensemble_weights: Optional[Dict[str, float]] = None  # Stacking weights
    detected_periods: Optional[List[int]] = None  # ACF detected periods

    def to_dict(self) -> Dict[str, Any]:
        return {
            "series_name": self.series_name,
            "horizon": self.horizon,
            "predictions": self.predictions,
            "confidence_interval": {
                "lower": self.lower_bound,
                "upper": self.upper_bound,
            },
            "model": self.model_used,
            "metrics": self.metrics,
            "seasonality": self.seasonality,
            "trend": self.trend,
            "decomposition": self.decomposition,
            "model_params": self.model_params,
            "bootstrap_ci": self.bootstrap_ci,
            "ensemble_weights": self.ensemble_weights,
            "detected_periods": self.detected_periods,
        }

    @property
    def is_high_confidence(self) -> bool:
        """예측 신뢰도 판단"""
        if not self.metrics:
            return False
        mape = self.metrics.get("mape", 100)
        return mape < 20  # MAPE 20% 미만이면 고신뢰

    @property
    def forecast_summary(self) -> str:
        """예측 요약 텍스트"""
        direction = self.trend or "unknown"
        last_pred = self.predictions[-1] if self.predictions else 0
        conf_width = (
            (self.upper_bound[-1] - self.lower_bound[-1]) / last_pred * 100
            if self.predictions and last_pred != 0 else 0
        )
        return f"Trend: {direction}, Final prediction: {last_pred:.2f} (±{conf_width:.1f}%)"


@dataclass
class SeasonalityInfo:
    """계절성 정보"""
    has_seasonality: bool
    period: Optional[int] = None  # 7=weekly, 12=monthly, 365=yearly
    strength: float = 0.0  # 0-1
    type: str = "none"  # additive, multiplicative
    peaks: List[int] = field(default_factory=list)  # 주기 피크 위치들
    secondary_periods: List[int] = field(default_factory=list)  # 추가 발견된 주기
    acf_values: Optional[List[float]] = None  # ACF 값 (디버깅/시각화용)
    confidence: float = 0.0  # 탐지 신뢰도 (0-1)


@dataclass
class TrendInfo:
    """트렌드 정보"""
    direction: str  # "increasing", "decreasing", "stable"
    slope: float
    relative_slope: float  # 평균 대비 기울기
    r_squared: float  # 트렌드 설명력
    change_points: List[int] = field(default_factory=list)  # 변화점 인덱스


class TimeSeriesForecaster:
    """
    시계열 예측기

    도메인별 최적 모델 자동 선택:
    - Retail: Prophet (holiday effects)
    - Manufacturing: ARIMA (process control)
    - Healthcare: Ensemble (safety-critical)
    - Finance: SARIMA (seasonality + volatility)
    """

    # 도메인별 기본 모델 매핑
    DOMAIN_MODEL_MAPPING = {
        "retail": ForecastModel.PROPHET,
        "ecommerce": ForecastModel.PROPHET,
        "manufacturing": ForecastModel.ARIMA,
        "healthcare": ForecastModel.ENSEMBLE,
        "finance": ForecastModel.SARIMA,
        "banking": ForecastModel.SARIMA,
        "logistics": ForecastModel.ARIMA,
        "energy": ForecastModel.PROPHET,
        "telecom": ForecastModel.ENSEMBLE,
    }

    def __init__(
        self,
        domain_context=None,
        default_model: ForecastModel = ForecastModel.ENSEMBLE,
        confidence_level: float = 0.95,
        auto_select_model: bool = True,
    ):
        """
        Args:
            domain_context: DomainContext 객체 (도메인별 최적화용)
            default_model: 기본 예측 모델
            confidence_level: 신뢰구간 수준 (기본 95%)
            auto_select_model: 데이터 특성 기반 모델 자동 선택
        """
        self.domain_context = domain_context
        self.default_model = default_model
        self.confidence_level = confidence_level
        self.auto_select_model = auto_select_model

        # 캐시
        self._seasonality_cache: Dict[str, SeasonalityInfo] = {}
        self._trend_cache: Dict[str, TrendInfo] = {}

    def detect_seasonality(
        self,
        series: List[float],
        max_lag: Optional[int] = None,
    ) -> SeasonalityInfo:
        """
        계절성 탐지 (Enhanced ACF-based)

        ACF(자기상관함수) 분석으로 주기성 탐지
        - scipy.signal.find_peaks 활용한 정교한 피크 탐지
        - 다중 주기 감지 (primary + secondary periods)
        - 신뢰도 점수 계산
        """
        if len(series) < 30:
            return SeasonalityInfo(has_seasonality=False)

        if not STATSMODELS_AVAILABLE or not NUMPY_AVAILABLE:
            return self._detect_seasonality_simple(series)

        try:
            max_lag = max_lag or min(len(series) // 2, 365)

            # ACF 계산
            acf_values = acf(series, nlags=max_lag, fft=True)

            # scipy.signal.find_peaks 사용 (가능한 경우)
            if SCIPY_AVAILABLE and find_peaks is not None:
                peaks, peak_properties = self._find_acf_peaks_scipy(acf_values)
            else:
                peaks, peak_properties = self._find_acf_peaks_manual(acf_values)

            if not peaks:
                return SeasonalityInfo(
                    has_seasonality=False,
                    acf_values=acf_values.tolist() if hasattr(acf_values, 'tolist') else list(acf_values)
                )

            # 가장 강한 주기 선택
            best_period = peaks[0]
            strength = peak_properties.get('strengths', [0.5])[0]

            # Secondary periods (다중 계절성)
            secondary_periods = peaks[1:5] if len(peaks) > 1 else []

            # 계절성 타입 결정
            # 변동 계수(CV)가 높으면 multiplicative
            mean_val = sum(series) / len(series)
            std_val = (sum((x - mean_val) ** 2 for x in series) / len(series)) ** 0.5
            cv = std_val / mean_val if mean_val != 0 else 0

            seasonality_type = "multiplicative" if cv > 0.5 else "additive"

            # 신뢰도 계산
            confidence = self._calculate_seasonality_confidence(
                acf_values, peaks, strength, len(series)
            )

            return SeasonalityInfo(
                has_seasonality=True,
                period=best_period,
                strength=min(strength, 1.0),
                type=seasonality_type,
                peaks=peaks[:5],
                secondary_periods=secondary_periods,
                acf_values=acf_values.tolist() if hasattr(acf_values, 'tolist') else list(acf_values),
                confidence=confidence,
            )

        except Exception as e:
            logger.debug(f"Seasonality detection failed: {e}")
            return SeasonalityInfo(has_seasonality=False)

    def _find_acf_peaks_scipy(
        self,
        acf_values,
    ) -> Tuple[List[int], Dict[str, List[float]]]:
        """scipy.signal.find_peaks를 사용한 정교한 피크 탐지"""
        # numpy array로 변환
        acf_arr = np.array(acf_values) if not isinstance(acf_values, np.ndarray) else acf_values

        # find_peaks 사용 (최소 높이 0.1, 최소 거리 2)
        peak_indices, properties = find_peaks(
            acf_arr[1:],  # lag 0 제외 (항상 1.0)
            height=0.1,
            distance=2,
            prominence=0.05,
        )

        # 인덱스 보정 (lag 0 제외했으므로 +1)
        peak_indices = peak_indices + 1

        if len(peak_indices) == 0:
            return [], {}

        # 강도순 정렬
        strengths = acf_arr[peak_indices]
        sorted_indices = np.argsort(strengths)[::-1]
        peaks = peak_indices[sorted_indices].tolist()
        sorted_strengths = strengths[sorted_indices].tolist()

        return peaks, {'strengths': sorted_strengths}

    def _find_acf_peaks_manual(
        self,
        acf_values,
    ) -> Tuple[List[int], Dict[str, List[float]]]:
        """수동 피크 탐지 (scipy 없을 때)"""
        peaks = []
        for i in range(2, len(acf_values) - 1):
            if (acf_values[i] > acf_values[i-1] and
                acf_values[i] > acf_values[i+1] and
                acf_values[i] > 0.1):
                peaks.append((i, acf_values[i]))

        if not peaks:
            return [], {}

        # 강도순 정렬
        peaks.sort(key=lambda x: x[1], reverse=True)
        return [p[0] for p in peaks], {'strengths': [p[1] for p in peaks]}

    def _calculate_seasonality_confidence(
        self,
        acf_values,
        peaks: List[int],
        strength: float,
        series_length: int,
    ) -> float:
        """계절성 탐지 신뢰도 계산"""
        confidence = 0.0

        # 1. 피크 강도 기반 (40%)
        confidence += min(strength, 1.0) * 0.4

        # 2. 피크 반복성 (피크의 배수에서도 ACF가 높은가) (30%)
        if len(peaks) > 0:
            main_period = peaks[0]
            harmonic_score = 0.0
            for i in range(2, 5):
                harmonic_idx = main_period * i
                if harmonic_idx < len(acf_values):
                    harmonic_score += max(0, acf_values[harmonic_idx]) * (1 / i)
            confidence += min(harmonic_score, 1.0) * 0.3

        # 3. 데이터 충분성 (30%)
        if len(peaks) > 0:
            required_cycles = peaks[0] * 3
            if series_length >= required_cycles:
                confidence += 0.3
            else:
                confidence += 0.3 * (series_length / required_cycles)

        return round(min(confidence, 1.0), 3)

    def _detect_seasonality_simple(self, series: List[float]) -> SeasonalityInfo:
        """간단한 계절성 탐지 (numpy/statsmodels 없을 때)"""
        if len(series) < 30:
            return SeasonalityInfo(has_seasonality=False)

        # 주간 패턴 체크 (7일 주기)
        if len(series) >= 28:
            week_pattern = self._check_periodic_pattern(series, 7)
            if week_pattern > 0.3:
                return SeasonalityInfo(
                    has_seasonality=True,
                    period=7,
                    strength=week_pattern,
                    type="additive",
                )

        # 월간 패턴 체크 (30일 주기)
        if len(series) >= 90:
            month_pattern = self._check_periodic_pattern(series, 30)
            if month_pattern > 0.3:
                return SeasonalityInfo(
                    has_seasonality=True,
                    period=30,
                    strength=month_pattern,
                    type="additive",
                )

        return SeasonalityInfo(has_seasonality=False)

    def _check_periodic_pattern(self, series: List[float], period: int) -> float:
        """특정 주기의 패턴 강도 계산"""
        if len(series) < period * 2:
            return 0.0

        # 주기별 평균 계산
        period_avgs = []
        for i in range(period):
            vals = [series[j] for j in range(i, len(series), period)]
            period_avgs.append(sum(vals) / len(vals))

        # 전체 평균 대비 분산
        overall_avg = sum(series) / len(series)
        period_variance = sum((p - overall_avg) ** 2 for p in period_avgs) / period
        overall_variance = sum((x - overall_avg) ** 2 for x in series) / len(series)

        if overall_variance == 0:
            return 0.0

        return min(period_variance / overall_variance, 1.0)

    def detect_trend(self, series: List[float]) -> TrendInfo:
        """
        트렌드 탐지

        선형 회귀로 트렌드 방향 및 강도 판단
        """
        if len(series) < 10:
            return TrendInfo(
                direction="stable",
                slope=0.0,
                relative_slope=0.0,
                r_squared=0.0,
            )

        # 선형 회귀 계산
        n = len(series)
        x = list(range(n))

        sum_x = sum(x)
        sum_y = sum(series)
        sum_xy = sum(xi * yi for xi, yi in zip(x, series))
        sum_x2 = sum(xi ** 2 for xi in x)

        denom = n * sum_x2 - sum_x ** 2
        if denom == 0:
            return TrendInfo(
                direction="stable",
                slope=0.0,
                relative_slope=0.0,
                r_squared=0.0,
            )

        slope = (n * sum_xy - sum_x * sum_y) / denom
        intercept = (sum_y - slope * sum_x) / n

        # R² 계산
        mean_y = sum_y / n
        ss_tot = sum((yi - mean_y) ** 2 for yi in series)
        ss_res = sum((yi - (slope * xi + intercept)) ** 2 for xi, yi in zip(x, series))
        r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

        # 상대적 기울기 (평균 대비)
        relative_slope = slope / mean_y if mean_y != 0 else 0

        # 트렌드 방향 결정
        if relative_slope > 0.01:
            direction = "increasing"
        elif relative_slope < -0.01:
            direction = "decreasing"
        else:
            direction = "stable"

        # 변화점 탐지 (간단한 방법)
        change_points = self._detect_change_points(series)

        return TrendInfo(
            direction=direction,
            slope=slope,
            relative_slope=relative_slope,
            r_squared=max(0, min(r_squared, 1.0)),
            change_points=change_points,
        )

    def _detect_change_points(self, series: List[float], window: int = 10) -> List[int]:
        """변화점 탐지 (기울기 변화 기반)"""
        if len(series) < window * 3:
            return []

        change_points = []

        for i in range(window, len(series) - window):
            # 이전 윈도우와 이후 윈도우의 평균 비교
            prev_avg = sum(series[i-window:i]) / window
            next_avg = sum(series[i:i+window]) / window

            # 평균의 급격한 변화 감지
            if prev_avg != 0:
                change_ratio = abs(next_avg - prev_avg) / abs(prev_avg)
                if change_ratio > 0.3:  # 30% 이상 변화
                    change_points.append(i)

        # 연속된 변화점 제거 (첫 번째만 유지)
        filtered = []
        for cp in change_points:
            if not filtered or cp - filtered[-1] > window:
                filtered.append(cp)

        return filtered

    def forecast_arima(
        self,
        series: List[float],
        horizon: int,
        order: Optional[Tuple[int, int, int]] = None,
        seasonal_order: Optional[Tuple[int, int, int, int]] = None,
    ) -> ForecastResult:
        """
        ARIMA/SARIMA 예측

        Args:
            series: 시계열 데이터
            horizon: 예측 기간
            order: (p, d, q) ARIMA 파라미터
            seasonal_order: (P, D, Q, s) 계절성 파라미터
        """
        if not STATSMODELS_AVAILABLE:
            raise ImportError("statsmodels required for ARIMA forecasting")

        # 자동 order 선택
        if order is None:
            order = self._auto_select_arima_order(series)

        try:
            if seasonal_order:
                from statsmodels.tsa.statespace.sarimax import SARIMAX
                model = SARIMAX(
                    series,
                    order=order,
                    seasonal_order=seasonal_order,
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                )
                model_name = f"SARIMA{order}x{seasonal_order}"
            else:
                model = ARIMA(series, order=order)
                model_name = f"ARIMA{order}"

            fitted = model.fit()

            # 예측
            forecast_obj = fitted.get_forecast(steps=horizon)
            predictions = forecast_obj.predicted_mean.tolist()
            conf_int = forecast_obj.conf_int(alpha=1 - self.confidence_level)

            # conf_int이 DataFrame인지 numpy array인지에 따라 처리
            if hasattr(conf_int, 'iloc'):
                lower_bound = conf_int.iloc[:, 0].tolist()
                upper_bound = conf_int.iloc[:, 1].tolist()
            elif hasattr(conf_int, 'values'):
                lower_bound = conf_int.values[:, 0].tolist()
                upper_bound = conf_int.values[:, 1].tolist()
            else:
                # numpy array인 경우
                lower_bound = conf_int[:, 0].tolist()
                upper_bound = conf_int[:, 1].tolist()

            return ForecastResult(
                series_name="",
                horizon=horizon,
                predictions=predictions,
                lower_bound=lower_bound,
                upper_bound=upper_bound,
                model_used=model_name,
                model_params={
                    "order": order,
                    "seasonal_order": seasonal_order,
                    "aic": fitted.aic if hasattr(fitted, 'aic') else None,
                    "bic": fitted.bic if hasattr(fitted, 'bic') else None,
                },
            )

        except Exception as e:
            logger.warning(f"ARIMA forecasting failed: {e}")
            return self._fallback_forecast(series, horizon)

    def _auto_select_arima_order(self, series: List[float]) -> Tuple[int, int, int]:
        """AIC 기반 ARIMA order 자동 선택"""
        if not STATSMODELS_AVAILABLE:
            return (1, 1, 1)

        # 정상성 테스트
        try:
            adf_result = adfuller(series)
            d = 0 if adf_result[1] < 0.05 else 1
        except Exception:
            d = 1

        # 간단한 그리드 서치
        best_aic = float('inf')
        best_order = (1, d, 1)

        for p in range(3):
            for q in range(3):
                try:
                    model = ARIMA(series, order=(p, d, q))
                    fitted = model.fit()
                    if fitted.aic < best_aic:
                        best_aic = fitted.aic
                        best_order = (p, d, q)
                except Exception:
                    continue

        return best_order

    def forecast_auto_arima(
        self,
        series: List[float],
        horizon: int,
        seasonal: bool = True,
        m: Optional[int] = None,
        max_p: int = 5,
        max_q: int = 5,
        max_d: int = 2,
        stepwise: bool = True,
    ) -> ForecastResult:
        """
        Auto ARIMA 예측 (pmdarima 통합)

        pmdarima를 사용한 자동 ARIMA 파라미터 선택
        - AIC/BIC 기반 최적 파라미터 탐색
        - 계절성 자동 탐지
        - Stepwise 알고리즘으로 빠른 탐색

        Args:
            series: 시계열 데이터
            horizon: 예측 기간
            seasonal: 계절성 ARIMA 사용 여부
            m: 계절 주기 (None이면 자동 탐지)
            max_p: 최대 AR 차수
            max_q: 최대 MA 차수
            max_d: 최대 차분 차수
            stepwise: Stepwise 알고리즘 사용

        Returns:
            ForecastResult
        """
        if not PMDARIMA_AVAILABLE:
            logger.info("pmdarima not available, falling back to manual ARIMA")
            return self.forecast_arima(series, horizon)

        try:
            # 계절 주기 자동 탐지
            if seasonal and m is None:
                seasonality_info = self.detect_seasonality(series)
                m = seasonality_info.period if seasonality_info.has_seasonality else 1
                if m == 1:
                    seasonal = False

            # Auto ARIMA 실행
            model = auto_arima(
                series,
                start_p=0,
                start_q=0,
                max_p=max_p,
                max_q=max_q,
                max_d=max_d,
                seasonal=seasonal,
                m=m if seasonal else 1,
                stepwise=stepwise,
                suppress_warnings=True,
                error_action='ignore',
                trace=False,
                n_fits=50,  # 최대 모델 수
            )

            # 예측
            forecast, conf_int = model.predict(
                n_periods=horizon,
                return_conf_int=True,
                alpha=1 - self.confidence_level
            )

            predictions = forecast.tolist() if hasattr(forecast, 'tolist') else list(forecast)
            lower_bound = conf_int[:, 0].tolist() if hasattr(conf_int, 'tolist') else [c[0] for c in conf_int]
            upper_bound = conf_int[:, 1].tolist() if hasattr(conf_int, 'tolist') else [c[1] for c in conf_int]

            # 모델 정보 추출
            order = model.order
            seasonal_order = model.seasonal_order if seasonal else None

            model_name = f"AutoARIMA{order}"
            if seasonal_order:
                model_name += f"x{seasonal_order}"

            return ForecastResult(
                series_name="",
                horizon=horizon,
                predictions=predictions,
                lower_bound=lower_bound,
                upper_bound=upper_bound,
                model_used=model_name,
                model_params={
                    "order": order,
                    "seasonal_order": seasonal_order,
                    "aic": model.aic() if hasattr(model, 'aic') else None,
                    "bic": model.bic() if hasattr(model, 'bic') else None,
                    "auto_selected": True,
                    "stepwise": stepwise,
                },
            )

        except Exception as e:
            logger.warning(f"Auto ARIMA forecasting failed: {e}")
            return self.forecast_arima(series, horizon)

    def forecast_prophet(
        self,
        series: List[float],
        dates: List[str],
        horizon: int,
        yearly_seasonality: bool = True,
        weekly_seasonality: bool = True,
        daily_seasonality: bool = False,
        holidays_df=None,
    ) -> ForecastResult:
        """
        Prophet 예측

        Args:
            series: 시계열 데이터
            dates: 날짜 리스트 (ISO format)
            horizon: 예측 기간
            yearly_seasonality: 연간 계절성
            weekly_seasonality: 주간 계절성
            daily_seasonality: 일간 계절성
            holidays_df: 휴일 데이터프레임
        """
        if not PROPHET_AVAILABLE or not PANDAS_AVAILABLE:
            raise ImportError("prophet and pandas required for Prophet forecasting")

        try:
            # 데이터 준비
            df = pd.DataFrame({
                'ds': pd.to_datetime(dates),
                'y': series
            })

            # Prophet 모델 설정
            model = Prophet(
                yearly_seasonality=yearly_seasonality,
                weekly_seasonality=weekly_seasonality,
                daily_seasonality=daily_seasonality,
                interval_width=self.confidence_level,
            )

            # 휴일 추가
            if holidays_df is not None:
                model.add_country_holidays(country_name='US')

            # 도메인별 추가 설정
            if self.domain_context:
                industry = getattr(self.domain_context, 'industry', None)
                if industry in ['retail', 'ecommerce']:
                    model.add_seasonality(
                        name='monthly',
                        period=30.5,
                        fourier_order=5
                    )

            model.fit(df)

            # 미래 데이터프레임 생성
            future = model.make_future_dataframe(periods=horizon)
            forecast = model.predict(future)

            # 예측 결과 추출
            predictions = forecast['yhat'].tail(horizon).tolist()
            lower = forecast['yhat_lower'].tail(horizon).tolist()
            upper = forecast['yhat_upper'].tail(horizon).tolist()

            return ForecastResult(
                series_name="",
                horizon=horizon,
                predictions=predictions,
                lower_bound=lower,
                upper_bound=upper,
                model_used="Prophet",
                model_params={
                    "yearly_seasonality": yearly_seasonality,
                    "weekly_seasonality": weekly_seasonality,
                    "daily_seasonality": daily_seasonality,
                },
            )

        except Exception as e:
            logger.warning(f"Prophet forecasting failed: {e}")
            return self._fallback_forecast(series, horizon)

    def forecast_prophet_advanced(
        self,
        series: List[float],
        dates: List[str],
        horizon: int,
        country_holidays: Optional[str] = None,
        custom_holidays: Optional[List[Dict[str, Any]]] = None,
        custom_seasonalities: Optional[List[Dict[str, Any]]] = None,
        regressors: Optional[Dict[str, List[float]]] = None,
        changepoint_prior_scale: float = 0.05,
        seasonality_prior_scale: float = 10.0,
        growth: str = "linear",
        cap: Optional[float] = None,
        floor: Optional[float] = None,
    ) -> ForecastResult:
        """
        강화된 Prophet 예측

        Holiday effects와 Custom seasonality 지원

        Args:
            series: 시계열 데이터
            dates: 날짜 리스트 (ISO format)
            horizon: 예측 기간
            country_holidays: 국가 코드 (e.g., 'US', 'KR', 'JP')
            custom_holidays: 커스텀 휴일 리스트
                [{"holiday": "black_friday", "ds": "2024-11-29", "lower_window": -1, "upper_window": 1}]
            custom_seasonalities: 커스텀 계절성 리스트
                [{"name": "monthly", "period": 30.5, "fourier_order": 5}]
            regressors: 외부 회귀 변수 딕셔너리
            changepoint_prior_scale: 변화점 민감도 (높을수록 유연)
            seasonality_prior_scale: 계절성 강도
            growth: 성장 모델 ("linear" 또는 "logistic")
            cap: 로지스틱 성장의 상한
            floor: 로지스틱 성장의 하한

        Returns:
            ForecastResult

        Example:
            >>> result = forecaster.forecast_prophet_advanced(
            ...     series=sales_data,
            ...     dates=date_list,
            ...     horizon=30,
            ...     country_holidays='US',
            ...     custom_holidays=[
            ...         {"holiday": "company_anniversary", "ds": "2024-03-15"}
            ...     ],
            ...     custom_seasonalities=[
            ...         {"name": "payday", "period": 14, "fourier_order": 3}
            ...     ]
            ... )
        """
        if not PROPHET_AVAILABLE or not PANDAS_AVAILABLE:
            raise ImportError("prophet and pandas required for Prophet forecasting")

        try:
            # 데이터 준비
            df = pd.DataFrame({
                'ds': pd.to_datetime(dates),
                'y': series
            })

            # 로지스틱 성장인 경우 cap/floor 추가
            if growth == "logistic":
                if cap is None:
                    cap = max(series) * 1.5
                if floor is None:
                    floor = min(0, min(series) * 0.5)
                df['cap'] = cap
                df['floor'] = floor

            # Prophet 모델 설정
            model = Prophet(
                growth=growth,
                changepoint_prior_scale=changepoint_prior_scale,
                seasonality_prior_scale=seasonality_prior_scale,
                interval_width=self.confidence_level,
                yearly_seasonality='auto',
                weekly_seasonality='auto',
                daily_seasonality='auto',
            )

            # 국가별 휴일 추가
            if country_holidays:
                model.add_country_holidays(country_name=country_holidays)

            # 커스텀 휴일 추가
            if custom_holidays:
                holidays_df = pd.DataFrame(custom_holidays)
                holidays_df['ds'] = pd.to_datetime(holidays_df['ds'])
                # 휴일 영향 윈도우 기본값 설정
                if 'lower_window' not in holidays_df.columns:
                    holidays_df['lower_window'] = 0
                if 'upper_window' not in holidays_df.columns:
                    holidays_df['upper_window'] = 1
                model = Prophet(
                    growth=growth,
                    changepoint_prior_scale=changepoint_prior_scale,
                    seasonality_prior_scale=seasonality_prior_scale,
                    interval_width=self.confidence_level,
                    holidays=holidays_df,
                )
                # 국가 휴일 다시 추가
                if country_holidays:
                    model.add_country_holidays(country_name=country_holidays)

            # 커스텀 계절성 추가
            if custom_seasonalities:
                for seasonality in custom_seasonalities:
                    model.add_seasonality(
                        name=seasonality.get('name', 'custom'),
                        period=seasonality.get('period', 7),
                        fourier_order=seasonality.get('fourier_order', 3),
                        prior_scale=seasonality.get('prior_scale', 10.0),
                        mode=seasonality.get('mode', 'additive'),
                    )

            # 외부 회귀 변수 추가
            regressor_names = []
            if regressors:
                for reg_name, reg_values in regressors.items():
                    if len(reg_values) == len(series):
                        df[reg_name] = reg_values
                        model.add_regressor(reg_name)
                        regressor_names.append(reg_name)

            # 모델 학습
            model.fit(df)

            # 미래 데이터프레임 생성
            future = model.make_future_dataframe(periods=horizon)

            # 로지스틱 성장인 경우 cap/floor 추가
            if growth == "logistic":
                future['cap'] = cap
                future['floor'] = floor

            # 외부 회귀 변수 미래 값 (0으로 채움 - 실제 사용 시 제공 필요)
            for reg_name in regressor_names:
                if reg_name in future.columns:
                    future[reg_name] = future[reg_name].fillna(0)
                else:
                    future[reg_name] = 0

            # 예측
            forecast = model.predict(future)

            # 예측 결과 추출
            predictions = forecast['yhat'].tail(horizon).tolist()
            lower = forecast['yhat_lower'].tail(horizon).tolist()
            upper = forecast['yhat_upper'].tail(horizon).tolist()

            # 계절성 컴포넌트 추출
            seasonality_components = {}
            for col in forecast.columns:
                if col.endswith('_lower') or col.endswith('_upper'):
                    continue
                if col in ['ds', 'yhat', 'yhat_lower', 'yhat_upper', 'trend',
                           'trend_lower', 'trend_upper', 'cap', 'floor',
                           'multiplicative_terms', 'additive_terms']:
                    continue
                seasonality_components[col] = forecast[col].tail(horizon).tolist()

            return ForecastResult(
                series_name="",
                horizon=horizon,
                predictions=predictions,
                lower_bound=lower,
                upper_bound=upper,
                model_used="ProphetAdvanced",
                model_params={
                    "growth": growth,
                    "country_holidays": country_holidays,
                    "custom_holidays_count": len(custom_holidays) if custom_holidays else 0,
                    "custom_seasonalities": [s.get('name') for s in (custom_seasonalities or [])],
                    "regressors": regressor_names,
                    "changepoint_prior_scale": changepoint_prior_scale,
                    "seasonality_prior_scale": seasonality_prior_scale,
                },
                decomposition={
                    "trend": forecast['trend'].tail(horizon).tolist(),
                    "seasonality_components": seasonality_components,
                },
            )

        except Exception as e:
            logger.warning(f"Prophet Advanced forecasting failed: {e}")
            # fallback to basic Prophet
            return self.forecast_prophet(
                series, dates, horizon,
                yearly_seasonality=True,
                weekly_seasonality=True,
            )

    def _get_default_holidays(self, country: str) -> List[Dict[str, Any]]:
        """국가별 기본 휴일 목록 반환 (Prophet 없을 때 사용)"""
        # 주요 국가별 공휴일 (예시)
        holidays = {
            'US': [
                {"holiday": "new_years_day", "month": 1, "day": 1},
                {"holiday": "mlk_day", "month": 1, "day": 15},  # 3rd Monday
                {"holiday": "presidents_day", "month": 2, "day": 19},
                {"holiday": "memorial_day", "month": 5, "day": 27},  # Last Monday
                {"holiday": "independence_day", "month": 7, "day": 4},
                {"holiday": "labor_day", "month": 9, "day": 2},  # 1st Monday
                {"holiday": "thanksgiving", "month": 11, "day": 28},  # 4th Thursday
                {"holiday": "christmas", "month": 12, "day": 25},
            ],
            'KR': [
                {"holiday": "new_years_day", "month": 1, "day": 1},
                {"holiday": "seollal", "month": 2, "day": 10},  # Lunar
                {"holiday": "independence_day", "month": 3, "day": 1},
                {"holiday": "childrens_day", "month": 5, "day": 5},
                {"holiday": "memorial_day", "month": 6, "day": 6},
                {"holiday": "liberation_day", "month": 8, "day": 15},
                {"holiday": "chuseok", "month": 9, "day": 17},  # Lunar
                {"holiday": "hangul_day", "month": 10, "day": 9},
                {"holiday": "christmas", "month": 12, "day": 25},
            ],
        }
        return holidays.get(country.upper(), [])

    def forecast_ensemble(
        self,
        series: List[float],
        horizon: int,
        dates: Optional[List[str]] = None,
        weights: Optional[Dict[str, float]] = None,
    ) -> ForecastResult:
        """
        앙상블 예측 (ARIMA + Prophet 가중 평균)

        Args:
            series: 시계열 데이터
            horizon: 예측 기간
            dates: 날짜 리스트 (Prophet용)
            weights: 모델별 가중치 {"arima": 0.4, "prophet": 0.6}
        """
        results = []
        model_weights = []

        # 기본 가중치
        if weights is None:
            weights = {"arima": 0.4, "prophet": 0.6}

        # ARIMA 예측
        if STATSMODELS_AVAILABLE:
            try:
                arima_result = self.forecast_arima(series, horizon)
                results.append(arima_result)
                model_weights.append(weights.get("arima", 0.4))
            except Exception as e:
                logger.debug(f"ARIMA failed in ensemble: {e}")

        # Prophet 예측
        if PROPHET_AVAILABLE and PANDAS_AVAILABLE and dates:
            try:
                prophet_result = self.forecast_prophet(series, dates, horizon)
                results.append(prophet_result)
                model_weights.append(weights.get("prophet", 0.6))
            except Exception as e:
                logger.debug(f"Prophet failed in ensemble: {e}")

        # 결과가 없으면 fallback
        if not results:
            return self._fallback_forecast(series, horizon)

        # 가중 평균 계산
        total_weight = sum(model_weights)
        model_weights = [w / total_weight for w in model_weights]

        predictions = [0.0] * horizon
        lower = [0.0] * horizon
        upper = [0.0] * horizon

        for result, weight in zip(results, model_weights):
            for i in range(horizon):
                predictions[i] += result.predictions[i] * weight
                lower[i] += result.lower_bound[i] * weight
                upper[i] += result.upper_bound[i] * weight

        models_used = [r.model_used for r in results]

        return ForecastResult(
            series_name="",
            horizon=horizon,
            predictions=predictions,
            lower_bound=lower,
            upper_bound=upper,
            model_used=f"Ensemble({'+'.join(models_used)})",
            model_params={
                "models": models_used,
                "weights": dict(zip(models_used, model_weights)),
            },
        )

    def forecast_weighted_stacking(
        self,
        series: List[float],
        horizon: int,
        dates: Optional[List[str]] = None,
        validation_ratio: float = 0.2,
        use_auto_arima: bool = True,
    ) -> ForecastResult:
        """
        Weighted Stacking Ensemble 예측

        각 모델의 validation 성능을 기반으로 가중치를 자동 학습
        - Ridge regression을 meta-learner로 사용
        - Cross-validation으로 가중치 최적화
        - Bootstrap CI 포함

        Args:
            series: 시계열 데이터
            horizon: 예측 기간
            dates: 날짜 리스트 (Prophet용)
            validation_ratio: 검증 데이터 비율
            use_auto_arima: Auto ARIMA 사용 여부

        Returns:
            ForecastResult
        """
        val_size = max(horizon, int(len(series) * validation_ratio))
        train_series = series[:-val_size]
        val_series = series[-val_size:]

        # 날짜도 분할
        train_dates = dates[:-val_size] if dates else None
        val_dates = dates[-val_size:] if dates else None

        # 각 모델로 validation 예측 수행
        model_predictions = {}
        model_results = {}

        # 1. ARIMA 계열
        if STATSMODELS_AVAILABLE:
            try:
                if use_auto_arima and PMDARIMA_AVAILABLE:
                    arima_result = self.forecast_auto_arima(train_series, val_size)
                else:
                    arima_result = self.forecast_arima(train_series, val_size)
                model_predictions['arima'] = arima_result.predictions
                model_results['arima'] = arima_result
            except Exception as e:
                logger.debug(f"ARIMA failed in stacking: {e}")

        # 2. Prophet
        if PROPHET_AVAILABLE and PANDAS_AVAILABLE and train_dates:
            try:
                prophet_result = self.forecast_prophet(train_series, train_dates, val_size)
                model_predictions['prophet'] = prophet_result.predictions
                model_results['prophet'] = prophet_result
            except Exception as e:
                logger.debug(f"Prophet failed in stacking: {e}")

        # 3. Simple MA (항상 사용 가능)
        try:
            ma_result = self._fallback_forecast(train_series, val_size)
            model_predictions['simple_ma'] = ma_result.predictions
            model_results['simple_ma'] = ma_result
        except Exception as e:
            logger.debug(f"Simple MA failed in stacking: {e}")

        if not model_predictions:
            return self._fallback_forecast(series, horizon)

        # 가중치 학습
        weights = self._learn_stacking_weights(
            model_predictions, val_series[:len(list(model_predictions.values())[0])]
        )

        # 전체 데이터로 최종 예측
        final_predictions = {}

        if 'arima' in weights and STATSMODELS_AVAILABLE:
            try:
                if use_auto_arima and PMDARIMA_AVAILABLE:
                    final_predictions['arima'] = self.forecast_auto_arima(series, horizon)
                else:
                    final_predictions['arima'] = self.forecast_arima(series, horizon)
            except Exception:
                pass

        if 'prophet' in weights and PROPHET_AVAILABLE and dates:
            try:
                final_predictions['prophet'] = self.forecast_prophet(series, dates, horizon)
            except Exception:
                pass

        if 'simple_ma' in weights:
            try:
                final_predictions['simple_ma'] = self._fallback_forecast(series, horizon)
            except Exception:
                pass

        if not final_predictions:
            return self._fallback_forecast(series, horizon)

        # 가중 예측 계산
        predictions = [0.0] * horizon
        lower = [0.0] * horizon
        upper = [0.0] * horizon
        total_weight = sum(weights.get(k, 0) for k in final_predictions.keys())

        for model_name, result in final_predictions.items():
            w = weights.get(model_name, 0) / total_weight if total_weight > 0 else 1 / len(final_predictions)
            for i in range(horizon):
                predictions[i] += result.predictions[i] * w
                lower[i] += result.lower_bound[i] * w
                upper[i] += result.upper_bound[i] * w

        # Bootstrap CI 계산 (옵션)
        bootstrap_ci = None
        if len(series) >= 50:
            try:
                bootstrap_ci = self._compute_bootstrap_ci(
                    series, horizon, predictions, n_bootstrap=100
                )
            except Exception as e:
                logger.debug(f"Bootstrap CI failed: {e}")

        models_used = list(final_predictions.keys())

        return ForecastResult(
            series_name="",
            horizon=horizon,
            predictions=predictions,
            lower_bound=lower,
            upper_bound=upper,
            model_used=f"WeightedStacking({'+'.join(models_used)})",
            model_params={
                "models": models_used,
                "stacking_weights": weights,
                "validation_ratio": validation_ratio,
                "use_auto_arima": use_auto_arima,
            },
            ensemble_weights=weights,
            bootstrap_ci=bootstrap_ci,
        )

    def _learn_stacking_weights(
        self,
        model_predictions: Dict[str, List[float]],
        actual_values: List[float],
    ) -> Dict[str, float]:
        """
        Stacking 가중치 학습

        Ridge regression 또는 간단한 MSE 기반 가중치 계산
        """
        if not model_predictions:
            return {}

        model_names = list(model_predictions.keys())

        # sklearn 사용 가능한 경우 Ridge regression
        if SKLEARN_AVAILABLE and NUMPY_AVAILABLE:
            try:
                # 예측 행렬 구성 (rows: time, cols: models)
                X = np.column_stack([
                    model_predictions[name][:len(actual_values)]
                    for name in model_names
                ])
                y = np.array(actual_values)

                # Ridge regression으로 가중치 학습
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)

                ridge = Ridge(alpha=1.0, fit_intercept=False, positive=True)
                ridge.fit(X_scaled, y)

                # 가중치 정규화 (합이 1이 되도록)
                raw_weights = np.abs(ridge.coef_)
                total = raw_weights.sum()
                if total > 0:
                    normalized_weights = raw_weights / total
                else:
                    normalized_weights = np.ones(len(model_names)) / len(model_names)

                return {name: float(w) for name, w in zip(model_names, normalized_weights)}

            except Exception as e:
                logger.debug(f"Ridge weight learning failed: {e}")

        # Fallback: MSE 기반 역가중치
        mse_dict = {}
        for name, preds in model_predictions.items():
            truncated_preds = preds[:len(actual_values)]
            mse = sum((p - a) ** 2 for p, a in zip(truncated_preds, actual_values)) / len(actual_values)
            mse_dict[name] = mse + 1e-10  # 0 방지

        # 역수 가중치 (MSE가 낮을수록 높은 가중치)
        inv_mse = {name: 1.0 / mse for name, mse in mse_dict.items()}
        total_inv = sum(inv_mse.values())
        weights = {name: v / total_inv for name, v in inv_mse.items()}

        return weights

    def _compute_bootstrap_ci(
        self,
        series: List[float],
        horizon: int,
        point_predictions: List[float],
        n_bootstrap: int = 100,
        confidence_level: float = None,
    ) -> Dict[str, List[float]]:
        """
        Bootstrap Confidence Interval 계산

        잔차 재샘플링 방식으로 신뢰구간 추정

        Args:
            series: 원본 시계열
            horizon: 예측 기간
            point_predictions: 점 예측값
            n_bootstrap: 부트스트랩 반복 횟수
            confidence_level: 신뢰 수준 (None이면 self.confidence_level 사용)

        Returns:
            {"lower": [...], "upper": [...], "std": [...]}
        """
        if confidence_level is None:
            confidence_level = self.confidence_level

        alpha = 1 - confidence_level

        # 훈련 데이터에서 잔차 계산 (in-sample)
        # 간단한 예측 모델로 잔차 추정
        window = min(7, len(series) // 2)
        residuals = []
        for i in range(window, len(series)):
            ma_pred = sum(series[i-window:i]) / window
            residuals.append(series[i] - ma_pred)

        if not residuals:
            residuals = [0.0]

        # Bootstrap 샘플링
        bootstrap_predictions = []
        for _ in range(n_bootstrap):
            # 잔차 재샘플링
            sampled_residuals = random.choices(residuals, k=horizon)

            # 예측에 잔차 추가
            boot_pred = [
                p + r for p, r in zip(point_predictions, sampled_residuals)
            ]
            bootstrap_predictions.append(boot_pred)

        # Percentile 방식 CI 계산
        if NUMPY_AVAILABLE:
            boot_array = np.array(bootstrap_predictions)
            lower_percentile = (alpha / 2) * 100
            upper_percentile = (1 - alpha / 2) * 100

            lower = np.percentile(boot_array, lower_percentile, axis=0).tolist()
            upper = np.percentile(boot_array, upper_percentile, axis=0).tolist()
            std = np.std(boot_array, axis=0).tolist()
        else:
            # numpy 없을 때 수동 계산
            lower = []
            upper = []
            std = []
            for i in range(horizon):
                values = sorted([bp[i] for bp in bootstrap_predictions])
                n = len(values)
                lower_idx = int(n * alpha / 2)
                upper_idx = int(n * (1 - alpha / 2))
                lower.append(values[lower_idx])
                upper.append(values[min(upper_idx, n - 1)])

                mean_val = sum(values) / n
                std_val = (sum((v - mean_val) ** 2 for v in values) / n) ** 0.5
                std.append(std_val)

        return {
            "lower": lower,
            "upper": upper,
            "std": std,
            "n_bootstrap": n_bootstrap,
            "confidence_level": confidence_level,
        }

    def _fallback_forecast(self, series: List[float], horizon: int) -> ForecastResult:
        """Fallback: 단순 이동평균 예측"""
        # 최근 7개 값의 이동평균
        ma_window = min(7, len(series))
        ma = sum(series[-ma_window:]) / ma_window

        # 트렌드 반영
        trend_info = self.detect_trend(series)
        predictions = []
        for i in range(horizon):
            pred = ma + (trend_info.slope * i)
            predictions.append(pred)

        # 신뢰구간 (표준편차 기반)
        std = (sum((x - ma) ** 2 for x in series[-ma_window:]) / ma_window) ** 0.5
        margin = std * 1.96  # 95% CI

        lower = [p - margin for p in predictions]
        upper = [p + margin for p in predictions]

        return ForecastResult(
            series_name="",
            horizon=horizon,
            predictions=predictions,
            lower_bound=lower,
            upper_bound=upper,
            model_used="SimpleMA+Trend",
            model_params={
                "ma_window": ma_window,
                "trend_slope": trend_info.slope,
            },
        )

    def forecast(
        self,
        series_name: str,
        series: List[float],
        horizon: int,
        dates: Optional[List[str]] = None,
        model: Optional[ForecastModel] = None,
    ) -> ForecastResult:
        """
        시계열 예측 메인 함수

        Args:
            series_name: 시계열 이름 (예: "daily_revenue")
            series: 시계열 데이터
            horizon: 예측 기간
            dates: 날짜 리스트 (Prophet용)
            model: 사용할 모델 (None이면 자동 선택)

        Returns:
            ForecastResult

        Example:
            >>> forecaster = TimeSeriesForecaster()
            >>> result = forecaster.forecast(
            ...     series_name="daily_sales",
            ...     series=[100, 120, 115, 130, ...],
            ...     horizon=7,
            ...     dates=["2024-01-01", "2024-01-02", ...]
            ... )
            >>> print(result.predictions)
        """
        if len(series) < 10:
            raise ValueError("Need at least 10 data points for forecasting")

        # 계절성/트렌드 분석
        seasonality = self.detect_seasonality(series)
        trend_info = self.detect_trend(series)

        # 모델 자동 선택
        if model is None and self.auto_select_model:
            model = self._select_model(seasonality, trend_info)
        elif model is None:
            model = self.default_model

        # 예측 실행
        if model == ForecastModel.ARIMA and STATSMODELS_AVAILABLE:
            result = self.forecast_arima(series, horizon)
        elif model == ForecastModel.AUTO_ARIMA:
            if PMDARIMA_AVAILABLE:
                result = self.forecast_auto_arima(series, horizon)
            elif STATSMODELS_AVAILABLE:
                result = self.forecast_arima(series, horizon)
            else:
                result = self._fallback_forecast(series, horizon)
        elif model == ForecastModel.SARIMA and STATSMODELS_AVAILABLE:
            # 계절성 order 설정
            seasonal_period = seasonality.period if seasonality.has_seasonality else 7
            result = self.forecast_arima(
                series,
                horizon,
                seasonal_order=(1, 1, 1, seasonal_period)
            )
        elif model == ForecastModel.PROPHET and PROPHET_AVAILABLE and dates:
            result = self.forecast_prophet(series, dates, horizon)
        elif model == ForecastModel.PROPHET_ADVANCED and PROPHET_AVAILABLE and dates:
            result = self.forecast_prophet_advanced(series, dates, horizon)
        elif model == ForecastModel.WEIGHTED_ENSEMBLE:
            result = self.forecast_weighted_stacking(series, horizon, dates)
        else:
            result = self.forecast_ensemble(series, horizon, dates)

        # detected periods 추가
        if seasonality.has_seasonality:
            result.detected_periods = [seasonality.period] + (seasonality.secondary_periods or [])

        # 메타데이터 추가
        result.series_name = series_name
        result.seasonality = seasonality.__dict__ if seasonality.has_seasonality else None
        result.trend = trend_info.direction

        # 정확도 메트릭 계산 (holdout validation)
        if len(series) > horizon + 10:
            result.metrics = self._calculate_metrics(series, horizon)

        return result

    def _select_model(
        self,
        seasonality: SeasonalityInfo,
        trend: TrendInfo,
    ) -> ForecastModel:
        """도메인 + 데이터 특성 기반 모델 자동 선택"""
        # 1. 도메인 기반 기본값
        if self.domain_context:
            industry = getattr(self.domain_context, 'industry', None)
            if industry and industry.lower() in self.DOMAIN_MODEL_MAPPING:
                return self.DOMAIN_MODEL_MAPPING[industry.lower()]

        # 2. 데이터 특성 기반
        # 강한 계절성 → Prophet
        if seasonality.has_seasonality and seasonality.strength > 0.5:
            if PROPHET_AVAILABLE:
                return ForecastModel.PROPHET
            elif STATSMODELS_AVAILABLE:
                return ForecastModel.SARIMA

        # 뚜렷한 트렌드 + 적은 노이즈 → ARIMA
        if trend.r_squared > 0.7 and trend.direction != "stable":
            if STATSMODELS_AVAILABLE:
                return ForecastModel.ARIMA

        # 복잡한 패턴 → Ensemble
        return ForecastModel.ENSEMBLE

    def _calculate_metrics(
        self,
        series: List[float],
        horizon: int,
    ) -> Dict[str, float]:
        """예측 정확도 메트릭 계산 (holdout validation)"""
        train = series[:-horizon]
        test = series[-horizon:]

        try:
            result = self.forecast_ensemble(train, horizon)
            predictions = result.predictions

            # MAE (Mean Absolute Error)
            mae = sum(abs(p - t) for p, t in zip(predictions, test)) / horizon

            # RMSE (Root Mean Squared Error)
            mse = sum((p - t) ** 2 for p, t in zip(predictions, test)) / horizon
            rmse = mse ** 0.5

            # MAPE (Mean Absolute Percentage Error)
            mape_sum = sum(
                abs((p - t) / t)
                for p, t in zip(predictions, test)
                if t != 0
            )
            mape = (mape_sum / horizon) * 100

            # SMAPE (Symmetric MAPE)
            smape_sum = sum(
                abs(p - t) / ((abs(p) + abs(t)) / 2)
                for p, t in zip(predictions, test)
                if (abs(p) + abs(t)) > 0
            )
            smape = (smape_sum / horizon) * 100

            return {
                "mae": round(mae, 4),
                "rmse": round(rmse, 4),
                "mape": round(mape, 2),
                "smape": round(smape, 2),
            }

        except Exception as e:
            logger.debug(f"Metrics calculation failed: {e}")
            return {}

    def decompose(
        self,
        series: List[float],
        period: Optional[int] = None,
    ) -> Dict[str, Any]:
        """시계열 분해 (Trend + Seasonal + Residual)"""
        if not STATSMODELS_AVAILABLE or not PANDAS_AVAILABLE:
            return {"error": "statsmodels and pandas required"}

        if len(series) < 2 * (period or 7):
            return {"error": "Series too short for decomposition"}

        try:
            # 주기 자동 탐지
            if period is None:
                seasonality = self.detect_seasonality(series)
                period = seasonality.period or 7

            result = seasonal_decompose(
                series,
                model='additive',
                period=period,
            )

            return {
                "trend": result.trend.tolist() if hasattr(result.trend, 'tolist') else list(result.trend),
                "seasonal": result.seasonal.tolist() if hasattr(result.seasonal, 'tolist') else list(result.seasonal),
                "residual": result.resid.tolist() if hasattr(result.resid, 'tolist') else list(result.resid),
                "period": period,
            }

        except Exception as e:
            logger.error(f"Decomposition failed: {e}")
            return {"error": str(e)}

    def multi_series_forecast(
        self,
        series_dict: Dict[str, List[float]],
        horizon: int,
        dates: Optional[List[str]] = None,
    ) -> Dict[str, ForecastResult]:
        """
        다중 시계열 동시 예측

        Args:
            series_dict: {시리즈명: 데이터} 딕셔너리
            horizon: 예측 기간
            dates: 날짜 리스트

        Returns:
            {시리즈명: ForecastResult} 딕셔너리
        """
        results = {}

        for name, series in series_dict.items():
            try:
                results[name] = self.forecast(
                    series_name=name,
                    series=series,
                    horizon=horizon,
                    dates=dates,
                )
            except Exception as e:
                logger.warning(f"Forecast failed for {name}: {e}")
                continue

        return results


# Factory functions
def create_forecaster(
    domain_context=None,
    default_model: ForecastModel = ForecastModel.WEIGHTED_ENSEMBLE,
    confidence_level: float = 0.95,
    auto_select_model: bool = True,
) -> TimeSeriesForecaster:
    """
    TimeSeriesForecaster 팩토리 함수

    Args:
        domain_context: DomainContext 객체 (도메인별 최적화용)
        default_model: 기본 예측 모델 (기본값: WEIGHTED_ENSEMBLE)
        confidence_level: 신뢰구간 수준 (기본 95%)
        auto_select_model: 데이터 특성 기반 모델 자동 선택

    Returns:
        TimeSeriesForecaster
    """
    return TimeSeriesForecaster(
        domain_context=domain_context,
        default_model=default_model,
        confidence_level=confidence_level,
        auto_select_model=auto_select_model,
    )


def get_available_models() -> List[str]:
    """사용 가능한 모델 목록 반환"""
    models = ["simple_ma", "ensemble", "weighted_ensemble"]

    if STATSMODELS_AVAILABLE:
        models.extend(["arima", "sarima"])

    if PMDARIMA_AVAILABLE:
        models.append("auto_arima")

    if PROPHET_AVAILABLE and PANDAS_AVAILABLE:
        models.extend(["prophet", "prophet_advanced"])

    return models


def check_dependencies() -> Dict[str, bool]:
    """의존성 상태 확인"""
    return {
        "numpy": NUMPY_AVAILABLE,
        "pandas": PANDAS_AVAILABLE,
        "statsmodels": STATSMODELS_AVAILABLE,
        "prophet": PROPHET_AVAILABLE,
        "scipy": SCIPY_AVAILABLE,
        "pmdarima": PMDARIMA_AVAILABLE,
        "sklearn": SKLEARN_AVAILABLE,
    }


def get_dependency_recommendations() -> Dict[str, str]:
    """
    누락된 의존성에 대한 설치 권장 사항 반환

    Returns:
        {dependency_name: install_command}
    """
    recommendations = {}

    if not NUMPY_AVAILABLE:
        recommendations["numpy"] = "pip install numpy"
    if not PANDAS_AVAILABLE:
        recommendations["pandas"] = "pip install pandas"
    if not STATSMODELS_AVAILABLE:
        recommendations["statsmodels"] = "pip install statsmodels"
    if not PROPHET_AVAILABLE:
        recommendations["prophet"] = "pip install prophet"
    if not SCIPY_AVAILABLE:
        recommendations["scipy"] = "pip install scipy"
    if not PMDARIMA_AVAILABLE:
        recommendations["pmdarima"] = "pip install pmdarima"
    if not SKLEARN_AVAILABLE:
        recommendations["sklearn"] = "pip install scikit-learn"

    return recommendations


def quick_forecast(
    series: List[float],
    horizon: int,
    dates: Optional[List[str]] = None,
    model: Optional[str] = None,
) -> ForecastResult:
    """
    빠른 예측을 위한 편의 함수

    Args:
        series: 시계열 데이터
        horizon: 예측 기간
        dates: 날짜 리스트 (Prophet용)
        model: 모델 이름 (None이면 자동 선택)

    Returns:
        ForecastResult

    Example:
        >>> result = quick_forecast(
        ...     series=[100, 120, 115, 130, 125],
        ...     horizon=7
        ... )
        >>> print(result.predictions)
    """
    forecaster = create_forecaster()

    model_enum = None
    if model:
        model_lower = model.lower()
        model_mapping = {
            "arima": ForecastModel.ARIMA,
            "auto_arima": ForecastModel.AUTO_ARIMA,
            "sarima": ForecastModel.SARIMA,
            "prophet": ForecastModel.PROPHET,
            "prophet_advanced": ForecastModel.PROPHET_ADVANCED,
            "ensemble": ForecastModel.ENSEMBLE,
            "weighted_ensemble": ForecastModel.WEIGHTED_ENSEMBLE,
            "simple_ma": ForecastModel.SIMPLE_MA,
        }
        model_enum = model_mapping.get(model_lower)

    return forecaster.forecast(
        series_name="quick_forecast",
        series=series,
        horizon=horizon,
        dates=dates,
        model=model_enum,
    )


def detect_seasonality_quick(
    series: List[float],
    max_lag: Optional[int] = None,
) -> Dict[str, Any]:
    """
    빠른 계절성 탐지를 위한 편의 함수

    Args:
        series: 시계열 데이터
        max_lag: 최대 lag (None이면 자동 설정)

    Returns:
        계절성 정보 딕셔너리
    """
    forecaster = create_forecaster()
    seasonality = forecaster.detect_seasonality(series, max_lag)

    return {
        "has_seasonality": seasonality.has_seasonality,
        "period": seasonality.period,
        "strength": seasonality.strength,
        "type": seasonality.type,
        "peaks": seasonality.peaks,
        "secondary_periods": seasonality.secondary_periods,
        "confidence": seasonality.confidence,
    }


def compute_bootstrap_ci(
    series: List[float],
    predictions: List[float],
    n_bootstrap: int = 100,
    confidence_level: float = 0.95,
) -> Dict[str, Any]:
    """
    Bootstrap 신뢰구간 계산을 위한 편의 함수

    Args:
        series: 원본 시계열
        predictions: 예측값
        n_bootstrap: 부트스트랩 반복 횟수
        confidence_level: 신뢰 수준

    Returns:
        {"lower": [...], "upper": [...], "std": [...]}
    """
    forecaster = TimeSeriesForecaster(confidence_level=confidence_level)
    return forecaster._compute_bootstrap_ci(
        series=series,
        horizon=len(predictions),
        point_predictions=predictions,
        n_bootstrap=n_bootstrap,
        confidence_level=confidence_level,
    )
