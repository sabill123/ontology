"""
Anomaly Detection Module

데이터 품질 이상 탐지 및 이상 패턴 감지
- Isolation Forest
- Autoencoder (reconstruction error)
- Statistical methods (Z-score, IQR)
- Ensemble methods

References:
- https://github.com/yzhao062/pyod
- Paper: "Hybrid Autoencoder and Isolation Forest for IoT Anomaly Detection" (2024)
- Paper: "LSTM Autoencoder + Isolation Forest for Multivariate Time Series" (2020)
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple, Union
from collections import defaultdict
from enum import Enum

logger = logging.getLogger(__name__)

# Optional imports
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

try:
    from scipy import stats as scipy_stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

try:
    from sklearn.ensemble import IsolationForest
    from sklearn.preprocessing import StandardScaler
    from sklearn.covariance import EllipticEnvelope
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning("sklearn not available for advanced anomaly detection")

try:
    from pyod.models.iforest import IForest
    from pyod.models.lof import LOF
    from pyod.models.knn import KNN
    from pyod.models.hbos import HBOS
    PYOD_AVAILABLE = True
except ImportError:
    PYOD_AVAILABLE = False
    logger.warning("pyod not available. Install with: pip install pyod")


class AnomalyType(Enum):
    """이상 유형"""
    OUTLIER = "outlier"           # 통계적 이상치
    NULL_SPIKE = "null_spike"     # 갑자기 증가한 null 값
    TYPE_MISMATCH = "type_mismatch"  # 타입 불일치
    PATTERN_BREAK = "pattern_break"  # 패턴 이탈
    DUPLICATE = "duplicate"       # 중복 데이터
    RANGE_VIOLATION = "range_violation"  # 범위 위반


class AnomalySeverity(Enum):
    """이상 심각도"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class Anomaly:
    """탐지된 이상"""
    anomaly_id: str
    anomaly_type: AnomalyType
    severity: AnomalySeverity
    table_name: str
    column_name: Optional[str]
    row_indices: List[int]
    anomaly_score: float
    description: str
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "anomaly_id": self.anomaly_id,
            "anomaly_type": self.anomaly_type.value,
            "severity": self.severity.value,
            "table_name": self.table_name,
            "column_name": self.column_name,
            "row_count": len(self.row_indices),
            "sample_rows": self.row_indices[:10],
            "anomaly_score": self.anomaly_score,
            "description": self.description,
            "details": self.details,
        }


@dataclass
class AnomalyReport:
    """이상 탐지 보고서"""
    table_name: str
    total_rows: int
    total_anomalies: int
    anomalies: List[Anomaly]
    quality_score: float  # 0~1, 높을수록 양호
    summary: Dict[str, int]  # type별 카운트

    def to_dict(self) -> Dict[str, Any]:
        return {
            "table_name": self.table_name,
            "total_rows": self.total_rows,
            "total_anomalies": self.total_anomalies,
            "quality_score": self.quality_score,
            "anomalies": [a.to_dict() for a in self.anomalies],
            "summary": self.summary,
        }


@dataclass
class AnomalyDetectionResult:
    """단일 컬럼/배열 이상 탐지 결과"""
    anomaly_count: int
    anomaly_ratio: float
    anomaly_indices: List[int]
    method: str
    scores: Optional[List[float]] = None


class StatisticalDetector:
    """통계 기반 이상 탐지"""

    def __init__(self, z_threshold: float = 3.0, iqr_multiplier: float = 1.5):
        self.z_threshold = z_threshold
        self.iqr_multiplier = iqr_multiplier

    def detect_z_score_outliers(
        self,
        values: List[Any],
        column_name: str,
    ) -> List[int]:
        """Z-score 기반 이상치 탐지"""
        if not NUMPY_AVAILABLE:
            return []

        # 숫자형 값만 추출
        numeric_values = []
        indices = []
        for i, v in enumerate(values):
            try:
                if v is not None:
                    numeric_values.append(float(v))
                    indices.append(i)
            except (ValueError, TypeError):
                continue

        if len(numeric_values) < 3:
            return []

        arr = np.array(numeric_values)
        mean = np.mean(arr)
        std = np.std(arr)

        if std == 0:
            return []

        z_scores = np.abs((arr - mean) / std)
        outlier_mask = z_scores > self.z_threshold

        return [indices[i] for i, is_outlier in enumerate(outlier_mask) if is_outlier]

    def detect_iqr_outliers(
        self,
        values: List[Any],
        column_name: str,
    ) -> List[int]:
        """IQR 기반 이상치 탐지"""
        if not NUMPY_AVAILABLE:
            return []

        numeric_values = []
        indices = []
        for i, v in enumerate(values):
            try:
                if v is not None:
                    numeric_values.append(float(v))
                    indices.append(i)
            except (ValueError, TypeError):
                continue

        if len(numeric_values) < 4:
            return []

        arr = np.array(numeric_values)
        q1 = np.percentile(arr, 25)
        q3 = np.percentile(arr, 75)
        iqr = q3 - q1

        lower_bound = q1 - self.iqr_multiplier * iqr
        upper_bound = q3 + self.iqr_multiplier * iqr

        outlier_mask = (arr < lower_bound) | (arr > upper_bound)

        return [indices[i] for i, is_outlier in enumerate(outlier_mask) if is_outlier]


class MLDetector:
    """ML 기반 이상 탐지"""

    def __init__(self, contamination: float = 0.1):
        self.contamination = contamination
        self.models = {}

    def detect_isolation_forest(
        self,
        data: np.ndarray,
        column_names: Optional[List[str]] = None,
    ) -> Tuple[List[int], np.ndarray]:
        """
        Isolation Forest 기반 이상 탐지

        Returns:
            (outlier_indices, anomaly_scores)
        """
        if not SKLEARN_AVAILABLE:
            return [], np.array([])

        if data.shape[0] < 10:
            return [], np.array([])

        try:
            # 결측치 처리
            data = np.nan_to_num(data, nan=0.0)

            # 스케일링
            scaler = StandardScaler()
            data_scaled = scaler.fit_transform(data)

            # Isolation Forest
            iso_forest = IsolationForest(
                contamination=self.contamination,
                random_state=42,
                n_estimators=100,
            )
            predictions = iso_forest.fit_predict(data_scaled)
            scores = -iso_forest.score_samples(data_scaled)  # 높을수록 이상

            outlier_indices = np.where(predictions == -1)[0].tolist()
            return outlier_indices, scores

        except Exception as e:
            logger.warning(f"Isolation Forest failed: {e}")
            return [], np.array([])

    def detect_with_pyod(
        self,
        data: np.ndarray,
        method: str = "ensemble",
    ) -> Tuple[List[int], np.ndarray]:
        """
        PyOD 기반 이상 탐지

        Args:
            method: "iforest", "lof", "knn", "hbos", "ensemble"
        """
        if not PYOD_AVAILABLE:
            return [], np.array([])

        if data.shape[0] < 10:
            return [], np.array([])

        try:
            data = np.nan_to_num(data, nan=0.0)

            if method == "ensemble":
                # 앙상블: 여러 모델의 평균
                models = [
                    IForest(contamination=self.contamination),
                    LOF(contamination=self.contamination),
                    HBOS(contamination=self.contamination),
                ]
                scores_list = []
                for model in models:
                    model.fit(data)
                    scores_list.append(model.decision_scores_)

                ensemble_scores = np.mean(scores_list, axis=0)
                threshold = np.percentile(ensemble_scores, (1 - self.contamination) * 100)
                outlier_indices = np.where(ensemble_scores > threshold)[0].tolist()
                return outlier_indices, ensemble_scores

            else:
                model_map = {
                    "iforest": IForest,
                    "lof": LOF,
                    "knn": KNN,
                    "hbos": HBOS,
                }
                model_class = model_map.get(method, IForest)
                model = model_class(contamination=self.contamination)
                model.fit(data)

                outlier_indices = np.where(model.labels_ == 1)[0].tolist()
                return outlier_indices, model.decision_scores_

        except Exception as e:
            logger.warning(f"PyOD detection failed: {e}")
            return [], np.array([])


class DataQualityDetector:
    """데이터 품질 이상 탐지"""

    def detect_null_anomalies(
        self,
        values: List[Any],
        column_name: str,
        expected_null_ratio: float = 0.1,
    ) -> Optional[Anomaly]:
        """Null 비율 이상 탐지"""
        if not values:
            return None

        null_count = sum(1 for v in values if v is None or v == "" or (NUMPY_AVAILABLE and isinstance(v, float) and np.isnan(v)))
        null_ratio = null_count / len(values)

        if null_ratio > expected_null_ratio * 3:  # 예상의 3배 이상
            severity = AnomalySeverity.CRITICAL if null_ratio > 0.5 else AnomalySeverity.HIGH
            return Anomaly(
                anomaly_id=f"NULL_{column_name}",
                anomaly_type=AnomalyType.NULL_SPIKE,
                severity=severity,
                table_name="",
                column_name=column_name,
                row_indices=[i for i, v in enumerate(values) if v is None or v == ""],
                anomaly_score=null_ratio,
                description=f"High null ratio: {null_ratio:.1%} (expected: {expected_null_ratio:.1%})",
                details={"null_ratio": null_ratio, "null_count": null_count},
            )
        return None

    def detect_type_mismatches(
        self,
        values: List[Any],
        column_name: str,
        expected_type: str,
    ) -> Optional[Anomaly]:
        """타입 불일치 탐지"""
        if not values:
            return None

        mismatch_indices = []
        for i, v in enumerate(values):
            if v is None:
                continue

            is_mismatch = False
            if expected_type in ["integer", "int"]:
                try:
                    int(v)
                except (ValueError, TypeError):
                    is_mismatch = True
            elif expected_type in ["float", "numeric"]:
                try:
                    float(v)
                except (ValueError, TypeError):
                    is_mismatch = True
            elif expected_type in ["datetime", "date"]:
                # 간단한 날짜 패턴 체크
                import re
                if not re.match(r'\d{4}[-/]\d{2}[-/]\d{2}', str(v)):
                    is_mismatch = True

            if is_mismatch:
                mismatch_indices.append(i)

        mismatch_ratio = len(mismatch_indices) / len(values) if values else 0

        if mismatch_ratio > 0.05:  # 5% 이상
            return Anomaly(
                anomaly_id=f"TYPE_{column_name}",
                anomaly_type=AnomalyType.TYPE_MISMATCH,
                severity=AnomalySeverity.MEDIUM if mismatch_ratio < 0.2 else AnomalySeverity.HIGH,
                table_name="",
                column_name=column_name,
                row_indices=mismatch_indices,
                anomaly_score=mismatch_ratio,
                description=f"Type mismatch: {mismatch_ratio:.1%} values don't match expected type '{expected_type}'",
                details={"expected_type": expected_type, "mismatch_count": len(mismatch_indices)},
            )
        return None

    def detect_duplicates(
        self,
        values: List[Any],
        column_name: str,
        is_key: bool = False,
    ) -> Optional[Anomaly]:
        """중복 값 탐지 (키 컬럼의 경우)"""
        if not values or not is_key:
            return None

        value_indices = defaultdict(list)
        for i, v in enumerate(values):
            if v is not None:
                value_indices[v].append(i)

        duplicate_groups = {v: indices for v, indices in value_indices.items() if len(indices) > 1}

        if duplicate_groups:
            total_duplicates = sum(len(indices) - 1 for indices in duplicate_groups.values())
            duplicate_ratio = total_duplicates / len(values)

            return Anomaly(
                anomaly_id=f"DUP_{column_name}",
                anomaly_type=AnomalyType.DUPLICATE,
                severity=AnomalySeverity.HIGH if is_key else AnomalySeverity.MEDIUM,
                table_name="",
                column_name=column_name,
                row_indices=[idx for indices in duplicate_groups.values() for idx in indices[1:]],
                anomaly_score=duplicate_ratio,
                description=f"Duplicate values in {'key' if is_key else ''} column: {total_duplicates} duplicates",
                details={"duplicate_groups": len(duplicate_groups), "total_duplicates": total_duplicates},
            )
        return None


class AnomalyDetector:
    """
    통합 이상 탐지기

    여러 탐지 방법을 앙상블로 결합
    """

    def __init__(
        self,
        contamination: float = 0.1,
        z_threshold: float = 3.0,
        use_ml: bool = True,
    ):
        self.contamination = contamination
        self.statistical = StatisticalDetector(z_threshold=z_threshold)
        self.ml = MLDetector(contamination=contamination) if use_ml else None
        self.quality = DataQualityDetector()

        self.anomaly_counter = 0

    def _next_anomaly_id(self) -> str:
        self.anomaly_counter += 1
        return f"ANO_{self.anomaly_counter:05d}"

    def detect_column_anomalies(
        self,
        values: List[Any],
        column_name: str,
        column_info: Optional[Dict[str, Any]] = None,
    ) -> List[Anomaly]:
        """컬럼 단위 이상 탐지"""
        anomalies = []
        column_info = column_info or {}

        # 1. Null 이상
        expected_null = column_info.get("null_ratio", 0.1)
        null_anomaly = self.quality.detect_null_anomalies(values, column_name, expected_null)
        if null_anomaly:
            null_anomaly.anomaly_id = self._next_anomaly_id()
            anomalies.append(null_anomaly)

        # 2. 타입 불일치
        expected_type = column_info.get("dtype", "string")
        type_anomaly = self.quality.detect_type_mismatches(values, column_name, expected_type)
        if type_anomaly:
            type_anomaly.anomaly_id = self._next_anomaly_id()
            anomalies.append(type_anomaly)

        # 3. 중복 (키 컬럼)
        is_key = column_info.get("is_key", False)
        dup_anomaly = self.quality.detect_duplicates(values, column_name, is_key)
        if dup_anomaly:
            dup_anomaly.anomaly_id = self._next_anomaly_id()
            anomalies.append(dup_anomaly)

        # 4. 통계적 이상치 (숫자형)
        if expected_type in ["integer", "int", "float", "numeric"]:
            z_outliers = self.statistical.detect_z_score_outliers(values, column_name)
            iqr_outliers = self.statistical.detect_iqr_outliers(values, column_name)

            # 두 방법에서 모두 탐지된 것을 이상치로
            confirmed_outliers = set(z_outliers) & set(iqr_outliers)

            if confirmed_outliers:
                anomalies.append(Anomaly(
                    anomaly_id=self._next_anomaly_id(),
                    anomaly_type=AnomalyType.OUTLIER,
                    severity=AnomalySeverity.MEDIUM,
                    table_name="",
                    column_name=column_name,
                    row_indices=list(confirmed_outliers),
                    anomaly_score=len(confirmed_outliers) / len(values),
                    description=f"Statistical outliers detected: {len(confirmed_outliers)} values",
                    details={"z_outliers": len(z_outliers), "iqr_outliers": len(iqr_outliers)},
                ))

        return anomalies

    def detect_table_anomalies(
        self,
        table_name: str,
        columns: List[str],
        data: List[List[Any]],
        column_info: Optional[Dict[str, Dict]] = None,
    ) -> AnomalyReport:
        """테이블 단위 이상 탐지"""
        all_anomalies = []
        column_info = column_info or {}

        if not data:
            return AnomalyReport(
                table_name=table_name,
                total_rows=0,
                total_anomalies=0,
                anomalies=[],
                quality_score=1.0,
                summary={},
            )

        # 컬럼별 이상 탐지
        for col_idx, col_name in enumerate(columns):
            col_values = [row[col_idx] if col_idx < len(row) else None for row in data]
            col_meta = column_info.get(col_name, {})

            col_anomalies = self.detect_column_anomalies(col_values, col_name, col_meta)
            for anomaly in col_anomalies:
                anomaly.table_name = table_name
            all_anomalies.extend(col_anomalies)

        # ML 기반 다변량 이상 탐지
        if self.ml and NUMPY_AVAILABLE and len(data) >= 20:
            numeric_cols = []
            numeric_indices = []

            for col_idx, col_name in enumerate(columns):
                col_meta = column_info.get(col_name, {})
                dtype = col_meta.get("dtype", "")
                if dtype in ["integer", "int", "float", "numeric"]:
                    numeric_indices.append(col_idx)
                    numeric_cols.append(col_name)

            if len(numeric_indices) >= 2:
                # 숫자형 데이터 추출
                numeric_data = []
                for row in data:
                    row_values = []
                    for idx in numeric_indices:
                        try:
                            val = float(row[idx]) if idx < len(row) and row[idx] is not None else 0.0
                        except (ValueError, TypeError):
                            val = 0.0
                        row_values.append(val)
                    numeric_data.append(row_values)

                numeric_array = np.array(numeric_data)

                # Isolation Forest
                outlier_indices, scores = self.ml.detect_isolation_forest(numeric_array, numeric_cols)

                if outlier_indices:
                    all_anomalies.append(Anomaly(
                        anomaly_id=self._next_anomaly_id(),
                        anomaly_type=AnomalyType.PATTERN_BREAK,
                        severity=AnomalySeverity.MEDIUM,
                        table_name=table_name,
                        column_name=None,
                        row_indices=outlier_indices,
                        anomaly_score=len(outlier_indices) / len(data),
                        description=f"Multivariate outliers (Isolation Forest): {len(outlier_indices)} rows",
                        details={"method": "isolation_forest", "columns_used": numeric_cols},
                    ))

        # 품질 점수 계산
        total_issues = sum(len(a.row_indices) for a in all_anomalies)
        quality_score = max(0, 1 - (total_issues / (len(data) * len(columns) + 1)))

        # 요약
        summary = defaultdict(int)
        for a in all_anomalies:
            summary[a.anomaly_type.value] += 1

        return AnomalyReport(
            table_name=table_name,
            total_rows=len(data),
            total_anomalies=len(all_anomalies),
            anomalies=all_anomalies,
            quality_score=quality_score,
            summary=dict(summary),
        )

    # =========================================================================
    # 편의 메서드: business_insights.py 에서 사용하는 API
    # =========================================================================

    def detect(
        self,
        values: Union[List[Any], "np.ndarray"],
        method: str = "ensemble",
    ) -> AnomalyDetectionResult:
        """
        단일 컬럼/배열에 대한 이상 탐지 (business_insights.py 호출용)

        Args:
            values: 숫자 값 리스트 또는 numpy 배열
            method: "ensemble", "iforest", "zscore", "iqr"

        Returns:
            AnomalyDetectionResult
        """
        # Convert numpy array to list if needed
        if NUMPY_AVAILABLE and isinstance(values, np.ndarray):
            values = values.tolist()

        # Check for empty values
        if values is None or len(values) == 0:
            return AnomalyDetectionResult(
                anomaly_count=0,
                anomaly_ratio=0.0,
                anomaly_indices=[],
                method=method,
                scores=None,
            )

        # 숫자형 값만 추출
        numeric_values = []
        original_indices = []
        for i, v in enumerate(values):
            try:
                if v is not None:
                    numeric_values.append(float(v))
                    original_indices.append(i)
            except (ValueError, TypeError):
                continue

        if len(numeric_values) < 10:
            return AnomalyDetectionResult(
                anomaly_count=0,
                anomaly_ratio=0.0,
                anomaly_indices=[],
                method=method,
                scores=None,
            )

        anomaly_indices = []
        scores = None

        if method == "zscore":
            # Z-score만 사용
            z_outliers = self.statistical.detect_z_score_outliers(values, "values")
            anomaly_indices = z_outliers

        elif method == "iqr":
            # IQR만 사용
            iqr_outliers = self.statistical.detect_iqr_outliers(values, "values")
            anomaly_indices = iqr_outliers

        elif method == "iforest" and self.ml and NUMPY_AVAILABLE:
            # Isolation Forest만 사용
            arr = np.array(numeric_values).reshape(-1, 1)
            outlier_local_indices, scores_arr = self.ml.detect_isolation_forest(arr)
            # 로컬 인덱스를 원본 인덱스로 매핑
            anomaly_indices = [original_indices[i] for i in outlier_local_indices if i < len(original_indices)]
            scores = scores_arr.tolist() if len(scores_arr) > 0 else None

        else:  # ensemble
            # v19.1: 적응형 contamination — 통계적 이상치 비율 기반
            z_outliers_prelim = set(self.statistical.detect_z_score_outliers(values, "values"))
            iqr_outliers_prelim = set(self.statistical.detect_iqr_outliers(values, "values"))
            stat_ratio = len(z_outliers_prelim | iqr_outliers_prelim) / max(len(values), 1)
            # 통계적 이상치가 없으면 contamination을 최소값으로, 있으면 비율에 맞춤
            adaptive_contamination = max(0.01, min(0.15, stat_ratio if stat_ratio > 0 else 0.02))
            if adaptive_contamination != self.contamination and self.ml:
                self.ml.contamination = adaptive_contamination
                logger.debug(f"[v19.1] Adaptive contamination: {self.contamination} → {adaptive_contamination} (stat_ratio={stat_ratio:.3f})")

            # 통계적 방법 + ML 앙상블
            z_outliers = z_outliers_prelim
            iqr_outliers = iqr_outliers_prelim
            statistical_outliers = z_outliers | iqr_outliers

            ml_outliers = set()
            if self.ml and NUMPY_AVAILABLE:
                arr = np.array(numeric_values).reshape(-1, 1)
                outlier_local_indices, scores_arr = self.ml.detect_isolation_forest(arr)
                ml_outliers = {original_indices[i] for i in outlier_local_indices if i < len(original_indices)}
                scores = scores_arr.tolist() if len(scores_arr) > 0 else None

            # v19.1: 앙상블 — majority vote (2/3 이상 동의) 로직
            if ml_outliers:
                # 각 인덱스별 탐지 방법 수 카운트 (z, iqr, ml 중 몇 개가 탐지했는지)
                all_candidates = z_outliers | iqr_outliers | ml_outliers
                consensus_indices = []
                for idx in all_candidates:
                    vote_count = sum([
                        idx in z_outliers,
                        idx in iqr_outliers,
                        idx in ml_outliers,
                    ])
                    if vote_count >= 2:  # 3개 방법 중 2개 이상 동의
                        consensus_indices.append(idx)
                # consensus가 비어있으면, 통계적 방법 합집합만 사용 (ML 단독은 제외)
                if consensus_indices:
                    anomaly_indices = consensus_indices
                else:
                    anomaly_indices = list(statistical_outliers)
            else:
                # ML 없으면 통계적 방법 두 개 모두에서 탐지된 것만
                anomaly_indices = list(z_outliers & iqr_outliers)

        anomaly_count = len(anomaly_indices)
        anomaly_ratio = anomaly_count / len(values) if values else 0.0

        return AnomalyDetectionResult(
            anomaly_count=anomaly_count,
            anomaly_ratio=anomaly_ratio,
            anomaly_indices=anomaly_indices,
            method=method,
            scores=scores,
        )

    # =========================================================================
    # 편의 메서드: governance.py 에이전트에서 사용하는 API
    # =========================================================================

    def detect_anomalies(
        self,
        sample_data: Dict[str, List[Dict[str, Any]]]
    ) -> List[Anomaly]:
        """
        샘플 데이터에서 이상 탐지 (에이전트 호출용 편의 메서드)

        Args:
            sample_data: {table_name: [row_dicts]}

        Returns:
            Anomaly 리스트
        """
        all_anomalies = []

        for table_name, rows in sample_data.items():
            if not rows:
                continue

            # row dict 형식을 list[list] 형식으로 변환
            columns = list(rows[0].keys())
            data = [[row.get(col) for col in columns] for row in rows]

            # 컬럼 정보 생성 (타입 추론)
            column_info = {}
            for col in columns:
                sample_values = [row.get(col) for row in rows[:10] if row.get(col) is not None]
                if sample_values:
                    first_val = sample_values[0]
                    if isinstance(first_val, (int, float)):
                        dtype = "numeric"
                    elif isinstance(first_val, bool):
                        dtype = "boolean"
                    else:
                        dtype = "string"
                    column_info[col] = {"dtype": dtype, "null_ratio": 0.1}

            # 테이블 이상 탐지 실행
            try:
                report = self.detect_table_anomalies(
                    table_name=table_name,
                    columns=columns,
                    data=data,
                    column_info=column_info,
                )
                all_anomalies.extend(report.anomalies)
            except Exception as e:
                logger.warning(f"Anomaly detection failed for {table_name}: {e}")
                continue

        return all_anomalies


# Convenience function
def detect_anomalies(
    table_name: str,
    columns: List[str],
    data: List[List[Any]],
    column_info: Optional[Dict[str, Dict]] = None,
) -> Dict[str, Any]:
    """
    테이블 이상 탐지 (편의 함수)

    Returns:
        AnomalyReport as dict
    """
    detector = AnomalyDetector()
    report = detector.detect_table_anomalies(table_name, columns, data, column_info)
    return report.to_dict()
