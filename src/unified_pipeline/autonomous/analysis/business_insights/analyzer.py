"""
Business Insights Analyzer Module

데이터 값 분석을 통한 비즈니스 인사이트 도출:
1. Status Distribution Analysis - 상태값 분포 분석
2. Anomaly Detection - 이상치 탐지
3. Trend Analysis - 트렌드 분석
4. Risk Identification - 리스크 식별
5. KPI Extraction - KPI 추출

References:
- Chandola et al. "Anomaly Detection: A Survey" (2009)
- Aggarwal "Outlier Analysis" (2017)
- Cleveland "Visualizing Data" (1993)
"""

import json
import logging
from typing import Dict, List, Any, Optional, Set, Tuple
from collections import Counter, defaultdict
from datetime import datetime
import statistics

from .models import InsightType, InsightSeverity, BusinessInsight

logger = logging.getLogger(__name__)


class BusinessInsightsAnalyzer:
    """
    비즈니스 인사이트 분석기

    데이터 값 분석을 통해 Palantir 수준의 비즈니스 인사이트 도출

    v2.0 기능 추가:
    - 시계열 예측 (ARIMA + Prophet 앙상블)
    - ML 기반 이상치 탐지 (IForest, LOF, HBOS 앙상블)
    - DomainContext 연동 KPI 분석
    """

    # 상태 컬럼 패턴
    STATUS_PATTERNS = ["status", "state", "stage"]

    # 문제 상태 키워드
    PROBLEM_STATUS_KEYWORDS = {
        "delayed": InsightSeverity.HIGH,
        "failed": InsightSeverity.CRITICAL,
        "cancelled": InsightSeverity.MEDIUM,
        "returned": InsightSeverity.MEDIUM,
        "rejected": InsightSeverity.HIGH,
        "error": InsightSeverity.CRITICAL,
        "pending": InsightSeverity.LOW,  # 오래 pending이면 문제
        "overdue": InsightSeverity.HIGH,
        "expired": InsightSeverity.MEDIUM,
        "at_risk": InsightSeverity.HIGH,
        "blocked": InsightSeverity.HIGH,
    }

    # 리스크 키워드
    RISK_KEYWORDS = {
        "high": ["high", "critical", "severe", "urgent"],
        "medium": ["medium", "moderate", "warning"],
        "low": ["low", "minor", "normal"],
    }

    # KPI 관련 컬럼 패턴
    KPI_PATTERNS = {
        "rate": r".*_rate$",
        "score": r".*_score$",
        "count": r".*_count$",
        "total": r"^total.*",
        "percentage": r".*_pct$|.*_percentage$",
    }

    # v14.0: LLM 기반 도메인 특화 분석 (하드코딩된 패턴 제거)
    # LLM이 데이터 특성을 보고 직접 KPI, 리스크, 임계값을 판단

    def __init__(
        self,
        domain_context=None,
        enable_ml_anomaly: bool = True,
        enable_forecasting: bool = True,
        llm_client=None,  # v14.0: LLM 클라이언트 추가
    ):
        """
        Args:
            domain_context: DomainContext 객체 (도메인별 KPI/워크플로우 설정)
            enable_ml_anomaly: ML 기반 이상치 탐지 활성화
            enable_forecasting: 시계열 예측 활성화
            llm_client: OpenAI 호환 LLM 클라이언트 (v14.0 도메인 특화 분석용)
        """
        self.insights: List[BusinessInsight] = []
        self.insight_counter = 0
        self.domain_context = domain_context
        self.enable_ml_anomaly = enable_ml_anomaly
        self.enable_forecasting = enable_forecasting
        self.llm_client = llm_client  # v14.0

        # 시계열 예측기 초기화
        self._forecaster = None
        if enable_forecasting:
            try:
                from ..time_series_forecaster import create_forecaster
                self._forecaster = create_forecaster(domain_context)
            except ImportError:
                logger.debug("Time series forecaster not available")

        # ML 이상치 탐지기 초기화
        self._anomaly_detector = None
        if enable_ml_anomaly:
            try:
                from ..anomaly_detection import AnomalyDetector
                self._anomaly_detector = AnomalyDetector()
            except ImportError:
                logger.debug("ML anomaly detector not available")

        # v14.0: LLM 기반 도메인 특화 분석 결과 캐시
        self._detected_domain: Optional[str] = None
        self._llm_domain_analysis: Optional[Dict[str, Any]] = None

    def analyze(
        self,
        tables: Dict[str, Dict[str, Any]],
        data: Dict[str, List[Dict[str, Any]]],
        column_semantics: Optional[Dict[str, Dict[str, Any]]] = None,
        pipeline_context: Optional[Dict[str, Any]] = None,
    ) -> List[BusinessInsight]:
        """
        전체 데이터셋 분석 및 인사이트 도출

        Args:
            tables: {table_name: {"columns": [...], "row_count": int}}
            data: {table_name: [row_dicts]}
            column_semantics: LLM이 분석한 컬럼 의미론 정보 (v14.0)
                {column_name: {"business_role": "optional_input", "nullable_by_design": True, ...}}
            pipeline_context: v18.3 — Phase 2 분석 모델 결과
                {"causal_insights": {...}, "counterfactual_analysis": [...],
                 "anomaly_results": {...}, "virtual_entities": [...], ...}

        Returns:
            비즈니스 인사이트 리스트
        """
        self.insights = []
        self.insight_counter = 0
        self.column_semantics = column_semantics or {}
        self._pipeline_context = pipeline_context or {}

        # v14.0: 도메인 정보 추출 (LLM 분석용)
        self._extract_domain_info()

        for table_name, rows in data.items():
            if not rows:
                continue

            table_info = tables.get(table_name, {})
            system = self._extract_system_name(table_name)

            # v28.0: Palantir Two-pass — 진입 시 단일 DataFrame 변환
            # 이후 모든 sub-method가 self._current_df를 통해 vectorized 연산 사용
            # 정확도: pandas groupby/corr는 Python statistics와 수학적으로 동일
            try:
                import pandas as _pd
                self._current_df = _pd.DataFrame(rows) if rows else _pd.DataFrame()
            except Exception:
                self._current_df = None

            # 1. 상태 분포 분석
            self._analyze_status_distribution(table_name, rows, system)

            # 2. 수량/금액 이상치 분석
            self._analyze_quantity_anomalies(table_name, rows, table_info, system)

            # 3. 날짜 기반 인사이트 (지연, 만료 등)
            self._analyze_date_insights(table_name, rows, table_info, system)

            # 4. 리스크 레벨 분석
            self._analyze_risk_levels(table_name, rows, system)

            # 5. KPI 추출
            self._extract_kpis(table_name, rows, table_info, system)

            # 6. 데이터 품질 이슈
            self._analyze_data_quality(table_name, rows, table_info, system)

            # 7. 시계열 예측 기반 인사이트 (v2.0)
            if self._forecaster:
                self._analyze_time_series_predictions(table_name, rows, table_info, system)

            # 8. ML 앙상블 이상치 탐지 (v2.0)
            if self._anomaly_detector:
                self._analyze_ml_anomalies(table_name, rows, table_info, system)

            # 9. 도메인 KPI 분석 (v2.0)
            if self.domain_context:
                self._analyze_domain_kpis(table_name, rows, table_info, system)

            # 10. 광고/마케팅 도메인 전용 분석 (v3.0)
            self._analyze_advertising_metrics(table_name, rows, table_info, system)

            # 11. v26.1: 카테고리별 세그먼트 성과 분석 (Duration, BGM, Creator Tier, Platform 등)
            self._analyze_segment_performance(table_name, rows, table_info, system)

            # 12. v26.1: 상관관계 기반 인사이트 (음의 상관 포함)
            self._analyze_correlation_insights(table_name, rows, table_info, system)

            # 13. v26.2: ID 컬럼 중복 패턴 분석 (video_id 등 longitudinal 패턴 감지)
            self._analyze_duplicate_patterns(table_name, rows, table_info, system)

            # 14. v27.0: 카테고리 분포 인사이트 (Language Distribution 등)
            self._analyze_categorical_distributions(table_name, rows, table_info, system)

        # 15. v27.0: 인사이트 중복 제거 및 모순 감지
        self._consolidate_and_validate_insights()

        # 16. v18.3: LLM 기반 도메인 특화 인사이트 생성 (분석 모델 결과 포함)
        if self.llm_client and self.domain_context:
            self._generate_llm_domain_insights(tables, data)

        # 인사이트 정렬 (심각도 순)
        severity_order = {
            InsightSeverity.CRITICAL: 0,
            InsightSeverity.HIGH: 1,
            InsightSeverity.MEDIUM: 2,
            InsightSeverity.LOW: 3,
            InsightSeverity.INFO: 4,
        }
        self.insights.sort(key=lambda x: severity_order[x.severity])

        return self.insights

    def _analyze_status_distribution(
        self,
        table_name: str,
        rows: List[Dict],
        system: str,
    ):
        """상태 분포 분석"""

        # 상태 컬럼 찾기
        if not rows:
            return

        sample_row = rows[0]
        status_columns = [
            col for col in sample_row.keys()
            if col and any(pat in col.lower() for pat in self.STATUS_PATTERNS)
        ]

        total_rows = len(rows)

        for col in status_columns:
            # 상태값 분포 계산
            status_counts = Counter(
                str(row.get(col, "")).lower()
                for row in rows
                if row.get(col) is not None
            )

            if not status_counts:
                continue

            # 문제 상태 탐지
            for status_value, count in status_counts.items():
                for problem_keyword, severity in self.PROBLEM_STATUS_KEYWORDS.items():
                    if problem_keyword in status_value:
                        percentage = (count / total_rows) * 100

                        # 임계값 기반 인사이트 생성
                        if count >= 100 or percentage >= 5:
                            insight = self._create_insight(
                                title=f"{problem_keyword.capitalize()} {col.replace('_', ' ').title()}",
                                description=f"Found {count:,} records ({percentage:.1f}%) with '{status_value}' status in {table_name}",
                                insight_type=InsightType.STATUS_DISTRIBUTION,
                                severity=severity if percentage >= 10 else InsightSeverity.MEDIUM,
                                source_table=table_name,
                                source_column=col,
                                source_system=system,
                                metric_value=status_value,
                                metric_name=f"{col}_distribution",
                                count=count,
                                percentage=percentage,
                                recommendations=self._get_status_recommendations(problem_keyword, status_value),
                                business_impact=self._assess_status_impact(problem_keyword, count, percentage),
                            )
                            self.insights.append(insight)

    def _analyze_quantity_anomalies(
        self,
        table_name: str,
        rows: List[Dict],
        table_info: Dict,
        system: str,
    ):
        """수량/금액 이상치 분석"""

        if not rows:
            return

        sample_row = rows[0]

        # 수량/금액 관련 컬럼 찾기
        numeric_patterns = ["quantity", "qty", "amount", "total", "price", "cost", "count", "value"]
        numeric_columns = [
            col for col in sample_row.keys()
            if col and any(pat in col.lower() for pat in numeric_patterns)
        ]

        for col in numeric_columns:
            values = [
                row.get(col) for row in rows
                if row.get(col) is not None
                and isinstance(row.get(col), (int, float))
            ]

            if len(values) < 10:
                continue

            # 통계 계산
            try:
                mean_val = statistics.mean(values)
                stdev_val = statistics.stdev(values) if len(values) > 1 else 0
                max_val = max(values)
                min_val = min(values)

                # 음수 값 탐지 (수량에서는 비정상)
                if col and ("quantity" in col.lower() or "qty" in col.lower()):
                    negative_count = sum(1 for v in values if v < 0)
                    if negative_count > 0:
                        insight = self._create_insight(
                            title=f"Negative {col.replace('_', ' ').title()} Values",
                            description=f"Found {negative_count:,} records with negative values in {col}",
                            insight_type=InsightType.DATA_QUALITY,
                            severity=InsightSeverity.HIGH,
                            source_table=table_name,
                            source_column=col,
                            source_system=system,
                            metric_value=negative_count,
                            metric_name="negative_count",
                            count=negative_count,
                            percentage=(negative_count / len(values)) * 100,
                            recommendations=["Investigate data entry errors", "Review business logic for negative values"],
                        )
                        self.insights.append(insight)

                # 이상치 탐지 (3 sigma rule)
                if stdev_val > 0:
                    upper_bound = mean_val + 3 * stdev_val
                    outliers = [v for v in values if v > upper_bound]

                    if len(outliers) >= 10:
                        insight = self._create_insight(
                            title=f"High {col.replace('_', ' ').title()} Outliers",
                            description=f"Found {len(outliers):,} records with abnormally high values (>{upper_bound:.2f})",
                            insight_type=InsightType.ANOMALY,
                            severity=InsightSeverity.MEDIUM,
                            source_table=table_name,
                            source_column=col,
                            source_system=system,
                            metric_value=upper_bound,
                            metric_name="upper_threshold",
                            count=len(outliers),
                            percentage=(len(outliers) / len(values)) * 100,
                            benchmark=mean_val,
                            deviation=stdev_val,
                            sample_values=sorted(outliers, reverse=True)[:5],
                        )
                        self.insights.append(insight)

            except Exception as e:
                logger.debug(f"Error analyzing {col}: {e}")

    def _analyze_date_insights(
        self,
        table_name: str,
        rows: List[Dict],
        table_info: Dict,
        system: str,
    ):
        """날짜 기반 인사이트 분석"""

        if not rows:
            return

        sample_row = rows[0]

        # 날짜 컬럼 쌍 찾기 (예: expected vs actual)
        date_patterns = {
            "expected": ["expected", "estimated", "scheduled", "planned", "due"],
            "actual": ["actual", "completed", "delivered", "processed"],
        }

        date_columns = [col for col in sample_row.keys() if col and ("date" in col.lower() or "time" in col.lower())]

        # 지연 분석 (expected vs actual)
        for col in date_columns:
            col_lower = col.lower()

            # 예정일 컬럼 찾기
            if any(pat in col_lower for pat in date_patterns["expected"]):
                # 대응되는 실제 날짜 컬럼 찾기
                actual_col = None
                for acol in date_columns:
                    if acol and any(pat in acol.lower() for pat in date_patterns["actual"]):
                        actual_col = acol
                        break

                if actual_col:
                    delayed_count = 0
                    total_with_dates = 0

                    for row in rows:
                        expected = row.get(col)
                        actual = row.get(actual_col)

                        if expected and actual:
                            total_with_dates += 1
                            try:
                                # 문자열을 날짜로 파싱 시도
                                if isinstance(expected, str) and isinstance(actual, str):
                                    exp_date = self._parse_date(expected)
                                    act_date = self._parse_date(actual)

                                    if exp_date and act_date and act_date > exp_date:
                                        delayed_count += 1
                            except (ValueError, TypeError, KeyError):
                                pass  # 날짜 파싱 실패는 정상 (비날짜 컬럼)

                    if delayed_count >= 100 and total_with_dates > 0:
                        percentage = (delayed_count / total_with_dates) * 100

                        insight = self._create_insight(
                            title=f"Delivery Delays Detected",
                            description=f"Found {delayed_count:,} delayed deliveries ({percentage:.1f}%) where {actual_col} > {col}",
                            insight_type=InsightType.OPERATIONAL,
                            severity=InsightSeverity.HIGH if percentage >= 10 else InsightSeverity.MEDIUM,
                            source_table=table_name,
                            source_column=col,
                            source_system=system,
                            metric_value=delayed_count,
                            metric_name="delayed_count",
                            count=delayed_count,
                            percentage=percentage,
                            recommendations=[
                                "Analyze root causes of delays",
                                "Review logistics and fulfillment processes",
                                "Identify problematic suppliers/carriers",
                            ],
                            business_impact=f"Potential customer satisfaction impact affecting {delayed_count:,} orders",
                        )
                        self.insights.append(insight)

    def _analyze_risk_levels(
        self,
        table_name: str,
        rows: List[Dict],
        system: str,
    ):
        """리스크 레벨 분석"""

        if not rows:
            return

        sample_row = rows[0]

        # 리스크 관련 컬럼 찾기
        risk_columns = [
            col for col in sample_row.keys()
            if col and ("risk" in col.lower() or "rating" in col.lower() or "tier" in col.lower())
        ]

        total_rows = len(rows)

        for col in risk_columns:
            risk_counts = Counter(
                str(row.get(col, "")).lower()
                for row in rows
                if row.get(col) is not None
            )

            # 고위험 항목 탐지
            high_risk_count = 0
            for value, count in risk_counts.items():
                if any(kw in value for kw in self.RISK_KEYWORDS["high"]):
                    high_risk_count += count

            if high_risk_count >= 10:
                percentage = (high_risk_count / total_rows) * 100

                insight = self._create_insight(
                    title=f"High Risk {col.replace('_', ' ').title()} Items",
                    description=f"Found {high_risk_count:,} high-risk items ({percentage:.1f}%)",
                    insight_type=InsightType.RISK,
                    severity=InsightSeverity.HIGH if percentage >= 5 else InsightSeverity.MEDIUM,
                    source_table=table_name,
                    source_column=col,
                    source_system=system,
                    metric_value="high_risk",
                    metric_name="risk_level",
                    count=high_risk_count,
                    percentage=percentage,
                    recommendations=[
                        "Review high-risk items urgently",
                        "Implement mitigation strategies",
                        "Consider backup plans",
                    ],
                )
                self.insights.append(insight)

    def _infer_kpi_threshold(self, col: str, values: List[float]):
        """
        v26.0: 데이터 스케일 기반 적응형 KPI 벤치마크 추론.
        1순위: domain_context KPI 정의 (threshold)
        2순위: 파생 비율 컬럼 감지 (_rate, _ratio, _score 패턴)
        3순위: 데이터 분포 기반 스케일 감지
        Returns None if no meaningful threshold.
        """
        col_lower = col.lower()

        # 1순위: domain_context KPI 정의
        if self.domain_context:
            domain_kpis = getattr(self.domain_context, 'kpis', [])
            for kpi_def in domain_kpis:
                kpi_name = kpi_def.get("name", "").lower()
                if kpi_name in col_lower or col_lower in kpi_name:
                    t = kpi_def.get("threshold")
                    if t is not None:
                        return float(t)

        # 2순위: 데이터 스케일 감지
        val_max = max(values)
        if val_max <= 0:
            return None

        # v26.0: 비율/점수 컬럼 패턴 감지 — max가 100 초과여도 적용
        # like_rate, comment_rate, share_rate, engagement_rate 등
        # 이 컬럼들은 실제로 백분율이지만 이상치로 인해 max > 100 가능
        rate_patterns = ["_rate", "_ratio", "rate_", "ratio_"]
        score_patterns = ["_score", "score_"]
        is_rate_col = any(p in col_lower for p in rate_patterns)
        is_score_col = any(p in col_lower for p in score_patterns)

        if is_rate_col or is_score_col:
            import numpy as np
            median_val = float(np.median(values))
            p75 = float(np.percentile(values, 75))
            # 중위수가 100 이하이면 백분율/비율 컬럼으로 간주
            if median_val <= 100:
                # 75th percentile 기반 벤치마크 (이상치에 강건)
                return round(min(p75 * 1.2, 100.0), 1) if is_rate_col else round(p75, 1)
            # 중위수가 0-1이면 비율 스케일
            if median_val <= 1.0:
                return round(min(p75 * 1.2, 1.0), 3)

        # rate 컬럼 (0-1 스케일)
        if "rate" in col_lower and val_max <= 1.0:
            return 0.7

        # 명시적 퍼센트 컬럼
        if ("pct" in col_lower or "percent" in col_lower) and val_max <= 100:
            return 70.0

        # Likert 스케일 (1-5, 1-10 등)
        if val_max <= 10:
            return round(0.7 * val_max, 1)

        # 0-100 스케일
        if val_max <= 100:
            return 70.0

        # v26.0: 대형 스케일 — 분포 기반 (이상치 강건)
        # 범위의 70% 대신 75th percentile 사용
        import numpy as np
        p75 = float(np.percentile(values, 75))
        return round(p75, 1)

    def _extract_kpis(
        self,
        table_name: str,
        rows: List[Dict],
        table_info: Dict,
        system: str,
    ):
        """KPI 추출"""

        if not rows:
            return

        sample_row = rows[0]

        for col in sample_row.keys():
            if not col:
                continue
            col_lower = col.lower()

            # rate/score 컬럼
            if "rate" in col_lower or "score" in col_lower or "pct" in col_lower:
                values = [
                    row.get(col) for row in rows
                    if row.get(col) is not None
                    and isinstance(row.get(col), (int, float))
                ]

                if len(values) >= 10:
                    avg_value = statistics.mean(values)

                    # v20: 데이터 스케일 기반 적응형 벤치마크
                    low_threshold = self._infer_kpi_threshold(col, values)
                    if low_threshold is None:
                        continue

                    low_count = sum(1 for v in values if v < low_threshold)
                    if low_count >= 10:
                        percentage = (low_count / len(values)) * 100

                        val_min, val_max = min(values), max(values)
                        insight = self._create_insight(
                            title=f"Low {col.replace('_', ' ').title()} Performance",
                            description=f"Found {low_count:,} items ({percentage:.1f}%) with {col} below threshold ({low_threshold}) [scale: {val_min:.1f}-{val_max:.1f}]",
                            insight_type=InsightType.KPI,
                            severity=InsightSeverity.MEDIUM if percentage < 20 else InsightSeverity.HIGH,
                            source_table=table_name,
                            source_column=col,
                            source_system=system,
                            metric_value=avg_value,
                            metric_name=col,
                            count=low_count,
                            percentage=percentage,
                            benchmark=low_threshold,
                            recommendations=[
                                f"Investigate factors affecting {col}",
                                "Identify improvement opportunities",
                            ],
                        )
                        self.insights.append(insight)

    def _analyze_data_quality(
        self,
        table_name: str,
        rows: List[Dict],
        table_info: Dict,
        system: str,
    ):
        """
        데이터 품질 이슈 분석 (v14.0: LLM 분석 결과 참조)

        LLM이 DataAnalyst에서 분석한 column_semantics를 참조하여:
        - nullable_by_design=True인 컬럼 (비고, memo, notes 등)의 null은 무시
        - business_role="optional_input"인 컬럼의 null은 무시
        - 비즈니스적으로 의미있는 품질 이슈만 보고
        """

        if not rows:
            return

        total_rows = len(rows)
        sample_row = rows[0]

        for col in sample_row.keys():
            if not col:
                continue

            # v14.0: LLM이 분석한 컬럼 의미론 확인
            col_info = self.column_semantics.get(col, {})
            nullable_by_design = col_info.get("nullable_by_design", False)
            business_role = col_info.get("business_role", "")

            # LLM이 "설계상 null 허용" 또는 "선택적 입력"으로 판단한 컬럼은 건너뜀
            if nullable_by_design or business_role == "optional_input":
                logger.debug(f"[v14.0] Skipping null check for {col}: nullable_by_design={nullable_by_design}, role={business_role}")
                continue

            # Null/빈값 비율
            null_count = sum(1 for row in rows if row.get(col) is None or row.get(col) == "")

            if null_count > 0:
                null_percentage = (null_count / total_rows) * 100

                if null_percentage >= 20:  # 20% 이상 null이면 문제 (필수 컬럼에 한해)
                    insight = self._create_insight(
                        title=f"High Missing Data in {col}",
                        description=f"{null_count:,} records ({null_percentage:.1f}%) have missing values in {col}",
                        insight_type=InsightType.DATA_QUALITY,
                        severity=InsightSeverity.MEDIUM if null_percentage < 50 else InsightSeverity.HIGH,
                        source_table=table_name,
                        source_column=col,
                        source_system=system,
                        metric_value=null_count,
                        metric_name="null_count",
                        count=null_count,
                        percentage=null_percentage,
                        recommendations=[
                            "Investigate data collection process",
                            "Consider default values or imputation",
                            "Review data entry procedures",
                        ],
                    )
                    self.insights.append(insight)

    # === Helper Methods ===

    def _create_insight(self, **kwargs) -> BusinessInsight:
        """인사이트 생성"""
        self.insight_counter += 1
        kwargs["insight_id"] = f"INSIGHT_{self.insight_counter:04d}"
        return BusinessInsight(**kwargs)

    def _extract_system_name(self, table_name: str) -> str:
        """테이블명에서 시스템명 추출"""
        if "__" in table_name:
            return table_name.split("__")[0].upper()
        return "UNKNOWN"

    def _get_status_recommendations(self, problem_keyword: str, status_value: str) -> List[str]:
        """상태별 권장 조치"""
        recommendations = {
            "delayed": [
                "Investigate delay root causes",
                "Review process bottlenecks",
                "Communicate with stakeholders",
            ],
            "failed": [
                "Analyze failure reasons",
                "Implement retry mechanisms",
                "Set up alerts for failures",
            ],
            "cancelled": [
                "Track cancellation reasons",
                "Analyze customer feedback",
                "Review cancellation policies",
            ],
            "returned": [
                "Analyze return reasons",
                "Improve product quality",
                "Review packaging and shipping",
            ],
        }
        return recommendations.get(problem_keyword, ["Investigate and resolve issues"])

    def _assess_status_impact(self, problem_keyword: str, count: int, percentage: float) -> str:
        """상태의 비즈니스 영향 평가"""
        if problem_keyword in ["failed", "delayed"]:
            return f"Customer satisfaction risk affecting {count:,} transactions ({percentage:.1f}%)"
        elif problem_keyword in ["cancelled", "returned"]:
            return f"Revenue impact from {count:,} items ({percentage:.1f}%)"
        else:
            return f"Operational impact on {count:,} items"

    def _parse_date(self, date_str: str) -> Optional[datetime]:
        """날짜 문자열 파싱"""
        formats = [
            "%Y-%m-%d",
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%d %H:%M",
            "%d/%m/%Y",
            "%m/%d/%Y",
        ]

        for fmt in formats:
            try:
                return datetime.strptime(date_str[:19], fmt)
            except (ValueError, TypeError):
                continue

        return None

    # === v2.0 시계열 예측 분석 ===

    def _analyze_time_series_predictions(
        self,
        table_name: str,
        rows: List[Dict],
        table_info: Dict,
        system: str,
    ):
        """시계열 예측 기반 인사이트 (v2.0)"""
        if not self._forecaster or not rows:
            return

        sample_row = rows[0]

        # 시계열 컬럼 찾기
        date_columns = [
            c for c in sample_row.keys()
            if "date" in c.lower() or "time" in c.lower()
        ]

        # v26.0: 시계열 부적합 컬럼 제외 — 순서형/나이/파생 컬럼은 시계열이 아님
        _ts_exclude_patterns = [
            "days_since", "age_in", "age_", "_age", "elapsed",
            "duration", "scene_count", "keyframe_count", "line_count",
            "text_length", "description_length", "hashtag_count",
            "_id", "_index", "_count",
        ]
        numeric_columns = [
            c for c in sample_row.keys()
            if isinstance(sample_row.get(c), (int, float))
            and not any(p in c.lower() for p in _ts_exclude_patterns)
        ]

        if not date_columns or not numeric_columns:
            return

        # 상위 3개 수치 컬럼에 대해 예측
        for num_col in numeric_columns[:3]:
            series = []
            dates = []

            for r in rows:
                val = r.get(num_col)
                date_val = r.get(date_columns[0])
                if val is not None and isinstance(val, (int, float)):
                    series.append(val)
                    if date_val:
                        dates.append(str(date_val))

            # 최소 30개 데이터 필요
            if len(series) < 30:
                continue

            try:
                result = self._forecaster.forecast(
                    series_name=num_col,
                    series=series,
                    horizon=7,
                    dates=dates if len(dates) == len(series) else None,
                )

                # 하락 트렌드 인사이트
                if result.trend == "decreasing":
                    last_val = series[-1] if series else 0
                    pred_val = result.predictions[-1] if result.predictions else 0
                    change_pct = ((pred_val - last_val) / last_val * 100) if last_val != 0 else 0

                    insight = self._create_insight(
                        title=f"Declining Trend Forecast: {num_col}",
                        description=f"Predicted {result.trend} trend using {result.model_used}. Expected {abs(change_pct):.1f}% decrease over next 7 periods.",
                        insight_type=InsightType.TREND,
                        severity=InsightSeverity.MEDIUM if abs(change_pct) < 10 else InsightSeverity.HIGH,
                        source_table=table_name,
                        source_column=num_col,
                        source_system=system,
                        metric_value=pred_val,
                        metric_name=f"{num_col}_forecast",
                        benchmark=last_val,
                        deviation=change_pct,
                        recommendations=[
                            "Monitor trend closely over next period",
                            "Investigate potential root causes",
                            "Prepare contingency plans",
                        ],
                        business_impact=f"Projected {abs(change_pct):.1f}% decline may impact operational targets",
                    )
                    self.insights.append(insight)

                # 급증 트렌드 인사이트
                elif result.trend == "increasing":
                    last_val = series[-1] if series else 0
                    pred_val = result.predictions[-1] if result.predictions else 0
                    change_pct = ((pred_val - last_val) / last_val * 100) if last_val != 0 else 0

                    if change_pct > 20:  # 20% 이상 증가 예측
                        insight = self._create_insight(
                            title=f"Growth Forecast: {num_col}",
                            description=f"Predicted {change_pct:.1f}% increase using {result.model_used}",
                            insight_type=InsightType.TREND,
                            severity=InsightSeverity.INFO,
                            source_table=table_name,
                            source_column=num_col,
                            source_system=system,
                            metric_value=pred_val,
                            metric_name=f"{num_col}_forecast",
                            benchmark=last_val,
                            deviation=change_pct,
                            recommendations=[
                                "Prepare for increased capacity",
                                "Review resource allocation",
                            ],
                        )
                        self.insights.append(insight)

                # 계절성 탐지 인사이트
                if result.seasonality:
                    period = result.seasonality.get("period")
                    strength = result.seasonality.get("strength", 0)

                    if strength > 0.5:  # 강한 계절성
                        insight = self._create_insight(
                            title=f"Strong Seasonality Detected: {num_col}",
                            description=f"Detected {period}-period seasonality with {strength:.0%} strength",
                            insight_type=InsightType.TREND,
                            severity=InsightSeverity.INFO,
                            source_table=table_name,
                            source_column=num_col,
                            source_system=system,
                            metric_value=period,
                            metric_name="seasonality_period",
                            recommendations=[
                                "Factor seasonality into planning",
                                "Adjust forecasts for seasonal patterns",
                            ],
                        )
                        self.insights.append(insight)

            except Exception as e:
                logger.debug(f"Forecast failed for {num_col}: {e}")

    # === v2.0 ML 이상치 앙상블 분석 ===

    # v18.2: ML 이상탐지에서 제외할 기술적 컬럼 패턴
    _SKIP_ML_COLUMN_PATTERNS = {
        "id", "key", "code", "uuid", "hash", "seq", "index", "idx",
        "pk", "fk", "created", "updated", "modified", "deleted",
        "timestamp", "version", "row_num", "row_number", "record_id",
    }

    def _is_business_metric_column(self, col_name: str) -> bool:
        """비즈니스 메트릭 컬럼인지 판별 (기술적 ID/키 컬럼 제외)"""
        col_lower = col_name.lower().strip()
        # 정확히 'id'이거나 '_id'로 끝나는 컬럼 제외
        if col_lower == "id" or col_lower.endswith("_id"):
            return False
        # 기술적 패턴 매칭
        for pattern in self._SKIP_ML_COLUMN_PATTERNS:
            if col_lower == pattern or col_lower.startswith(pattern + "_") or col_lower.endswith("_" + pattern):
                return False
        return True

    def _analyze_ml_anomalies(
        self,
        table_name: str,
        rows: List[Dict],
        table_info: Dict,
        system: str,
    ):
        """ML 기반 앙상블 이상치 탐지 (v2.0, v18.2: 비즈니스 메트릭 컬럼만 분석)"""
        if not self._anomaly_detector or not rows:
            return

        sample_row = rows[0]

        # 숫자 컬럼만 선택 (v18.2: 기술적 ID/키 컬럼 제외)
        for col in sample_row.keys():
            if not col:
                continue
            if not self._is_business_metric_column(col):
                continue
            values = [
                r.get(col) for r in rows
                if r.get(col) is not None and isinstance(r.get(col), (int, float))
            ]

            # 최소 100개 데이터 필요
            if len(values) < 100:
                continue

            try:
                result = self._anomaly_detector.detect(values, method="ensemble")

                if result.anomaly_count > 0 and result.anomaly_ratio > 0.01:
                    severity = InsightSeverity.HIGH if result.anomaly_ratio > 0.05 else InsightSeverity.MEDIUM

                    # 이상치 샘플 추출
                    anomaly_samples = []
                    if hasattr(result, 'anomaly_indices'):
                        for idx in result.anomaly_indices[:10]:
                            if 0 <= idx < len(values):
                                anomaly_samples.append(values[idx])

                    insight = self._create_insight(
                        title=f"ML Anomalies Detected: {col}",
                        description=f"Detected {result.anomaly_count:,} anomalies ({result.anomaly_ratio:.1%}) using {result.method} ensemble",
                        insight_type=InsightType.ANOMALY,
                        severity=severity,
                        source_table=table_name,
                        source_column=col,
                        source_system=system,
                        metric_value=result.anomaly_count,
                        metric_name="ml_anomaly_count",
                        count=result.anomaly_count,
                        percentage=result.anomaly_ratio * 100,
                        sample_values=anomaly_samples,
                        recommendations=[
                            "Review flagged records for data quality",
                            "Investigate business logic causing outliers",
                            "Consider adjusting anomaly thresholds if false positives",
                        ],
                        business_impact=f"{result.anomaly_count:,} records flagged for review",
                    )
                    self.insights.append(insight)

            except Exception as e:
                logger.debug(f"ML anomaly detection failed for {col}: {e}")

    # === v2.0 도메인 KPI 분석 ===

    def _analyze_domain_kpis(
        self,
        table_name: str,
        rows: List[Dict],
        table_info: Dict,
        system: str,
    ):
        """도메인 컨텍스트 기반 KPI 분석 (v2.0)"""
        if not self.domain_context or not rows:
            return

        sample_row = rows[0]

        # 도메인 KPI 정의 가져오기
        domain_kpis = getattr(self.domain_context, 'kpis', [])
        if not domain_kpis:
            return

        for kpi_def in domain_kpis:
            kpi_name = kpi_def.get("name", "").lower()
            threshold = kpi_def.get("threshold")
            direction = kpi_def.get("direction", "higher_is_better")

            # 매칭되는 컬럼 찾기
            matched_col = None
            for col in sample_row.keys():
                if col and (kpi_name in col.lower() or col.lower() in kpi_name):
                    matched_col = col
                    break

            if not matched_col:
                continue

            # KPI 값 계산
            values = [
                r.get(matched_col) for r in rows
                if r.get(matched_col) is not None and isinstance(r.get(matched_col), (int, float))
            ]

            if len(values) < 10:
                continue

            try:
                avg_value = statistics.mean(values)

                # 임계값 기반 분석
                if threshold is not None:
                    if direction == "higher_is_better" and avg_value < threshold:
                        deviation = ((threshold - avg_value) / threshold) * 100

                        insight = self._create_insight(
                            title=f"Domain KPI Below Target: {kpi_def.get('name', matched_col)}",
                            description=f"Average {matched_col} ({avg_value:.2f}) is below domain threshold ({threshold})",
                            insight_type=InsightType.KPI,
                            severity=InsightSeverity.HIGH if deviation > 20 else InsightSeverity.MEDIUM,
                            source_table=table_name,
                            source_column=matched_col,
                            source_system=system,
                            metric_value=avg_value,
                            metric_name=kpi_def.get("name", matched_col),
                            benchmark=threshold,
                            deviation=deviation,
                            recommendations=kpi_def.get("recommendations", [
                                f"Improve {kpi_def.get('name', matched_col)} by {deviation:.1f}%",
                                "Review operational processes",
                            ]),
                            business_impact=kpi_def.get("impact", f"KPI underperformance by {deviation:.1f}%"),
                        )
                        self.insights.append(insight)

                    elif direction == "lower_is_better" and avg_value > threshold:
                        deviation = ((avg_value - threshold) / threshold) * 100

                        insight = self._create_insight(
                            title=f"Domain KPI Above Target: {kpi_def.get('name', matched_col)}",
                            description=f"Average {matched_col} ({avg_value:.2f}) exceeds domain threshold ({threshold})",
                            insight_type=InsightType.KPI,
                            severity=InsightSeverity.HIGH if deviation > 20 else InsightSeverity.MEDIUM,
                            source_table=table_name,
                            source_column=matched_col,
                            source_system=system,
                            metric_value=avg_value,
                            metric_name=kpi_def.get("name", matched_col),
                            benchmark=threshold,
                            deviation=deviation,
                            recommendations=kpi_def.get("recommendations", [
                                f"Reduce {kpi_def.get('name', matched_col)} by {deviation:.1f}%",
                            ]),
                            business_impact=kpi_def.get("impact", f"KPI exceeds target by {deviation:.1f}%"),
                        )
                        self.insights.append(insight)

            except Exception as e:
                logger.debug(f"Domain KPI analysis failed for {matched_col}: {e}")

    # === v3.0 광고/마케팅 도메인 분석 ===

    # 광고 KPI 벤치마크 (업계 평균 기준)
    AD_BENCHMARKS = {
        "ctr": {"avg": 0.035, "good": 0.05, "excellent": 0.08},  # 3.5% 평균
        "conversion_rate": {"avg": 0.025, "good": 0.04, "excellent": 0.06},  # 2.5% 평균
        "cpc": {"avg": 2500, "good": 2000, "excellent": 1500},  # 2500원 평균 (lower is better)
        "cpm": {"avg": 10000, "good": 8000, "excellent": 5000},  # 10000원 평균 (lower is better)
        "roas": {"avg": 3.0, "good": 4.0, "excellent": 6.0},  # 300% 평균
        "quality_score": {"avg": 6, "good": 7, "excellent": 9},  # 6점 평균
        "frequency": {"avg": 2.0, "warning": 4.0, "critical": 7.0},  # 2회 평균
    }

    # 광고 메트릭 컬럼 패턴
    AD_METRIC_PATTERNS = {
        "impressions": ["impressions", "imps", "impression_count"],
        "clicks": ["clicks", "click_count"],
        "ctr": ["ctr", "click_through_rate", "clickthrough"],
        "conversions": ["conversions", "conversion_count", "cvr_count"],
        "conversion_rate": ["conversion_rate", "cvr"],
        "spend": ["spend", "cost", "ad_spend", "cost_micros"],
        "cpc": ["cpc", "cost_per_click", "average_cpc", "average_cpc_micros"],
        "cpm": ["cpm", "cost_per_mille", "cost_per_impression"],
        "roas": ["roas", "return_on_ad_spend"],
        "reach": ["reach", "unique_reach"],
        "frequency": ["frequency", "avg_frequency"],
        "quality_score": ["quality_score", "relevance_score", "ad_quality"],
    }

    def _analyze_advertising_metrics(
        self,
        table_name: str,
        rows: List[Dict],
        table_info: Dict,
        system: str,
    ):
        """
        광고/마케팅 도메인 전용 분석 (v3.0)

        Palantir 수준의 광고 인사이트 도출:
        1. CTR/CVR 벤치마크 비교
        2. CPC/CPM 트렌드 분석
        3. ROAS 평가
        4. 플랫폼 간 성과 비교
        5. 캠페인 이상치 탐지
        6. 예산 효율성 분석
        """
        if not rows:
            return

        sample_row = rows[0]
        table_lower = table_name.lower()

        # 광고 테이블인지 확인
        ad_table_signals = ["campaign", "ad", "metric", "insight", "performance"]
        is_ad_table = any(sig in table_lower for sig in ad_table_signals)

        # 광고 메트릭 컬럼이 있는지 확인
        has_ad_metrics = any(
            col and any(pattern in col.lower() for pattern in patterns)
            for col in sample_row.keys()
            for patterns in self.AD_METRIC_PATTERNS.values()
        )

        if not is_ad_table and not has_ad_metrics:
            return

        # 플랫폼 추출
        platform = self._extract_ad_platform(table_name)

        # 1. CTR 분석
        self._analyze_ctr(table_name, rows, platform, system)

        # 2. CPC 분석
        self._analyze_cpc(table_name, rows, platform, system)

        # 3. Conversion Rate 분석
        self._analyze_conversion_rate(table_name, rows, platform, system)

        # 4. ROAS 분석
        self._analyze_roas(table_name, rows, platform, system)

        # 5. Frequency 분석 (광고 피로도)
        self._analyze_ad_frequency(table_name, rows, platform, system)

        # 6. Quality Score 분석
        self._analyze_quality_score(table_name, rows, platform, system)

        # 7. 예산 효율성 분석
        self._analyze_budget_efficiency(table_name, rows, platform, system)

        # 8. 캠페인 성과 이상치 탐지
        self._analyze_campaign_anomalies(table_name, rows, platform, system)

    def _extract_ad_platform(self, table_name: str) -> str:
        """테이블명에서 광고 플랫폼 추출"""
        table_lower = table_name.lower()
        if "google" in table_lower:
            return "Google Ads"
        elif "meta" in table_lower or "facebook" in table_lower:
            return "Meta Ads"
        elif "naver" in table_lower:
            return "Naver Ads"
        elif "kakao" in table_lower:
            return "Kakao Ads"
        else:
            return "Unknown Platform"

    def _find_ad_column(self, sample_row: Dict, metric_type: str) -> Optional[str]:
        """광고 메트릭 컬럼 찾기"""
        patterns = self.AD_METRIC_PATTERNS.get(metric_type, [])
        for col in sample_row.keys():
            if not col:
                continue
            col_lower = col.lower()
            if any(pat in col_lower for pat in patterns):
                return col
        return None

    def _analyze_ctr(self, table_name: str, rows: List[Dict], platform: str, system: str):
        """CTR(클릭률) 분석"""
        if not rows:
            return

        sample_row = rows[0]
        ctr_col = self._find_ad_column(sample_row, "ctr")

        if not ctr_col:
            # CTR 직접 계산 시도
            clicks_col = self._find_ad_column(sample_row, "clicks")
            impressions_col = self._find_ad_column(sample_row, "impressions")

            if clicks_col and impressions_col:
                ctrs = []
                for row in rows:
                    clicks = row.get(clicks_col, 0)
                    impressions = row.get(impressions_col, 0)
                    if impressions and impressions > 0:
                        ctrs.append(clicks / impressions)

                if ctrs:
                    avg_ctr = statistics.mean(ctrs)
                    self._generate_ctr_insight(table_name, avg_ctr, len(ctrs), platform, system, "calculated")
            return

        # CTR 컬럼이 있는 경우
        ctrs = [
            row.get(ctr_col) for row in rows
            if row.get(ctr_col) is not None
            and isinstance(row.get(ctr_col), (int, float))
        ]

        if len(ctrs) < 3:
            return

        # CTR이 백분율로 저장된 경우 (예: 4.5 = 4.5%)
        avg_ctr = statistics.mean(ctrs)
        if avg_ctr > 1:  # 백분율로 저장된 경우
            avg_ctr = avg_ctr / 100

        self._generate_ctr_insight(table_name, avg_ctr, len(ctrs), platform, system, ctr_col)

    def _generate_ctr_insight(
        self,
        table_name: str,
        avg_ctr: float,
        sample_count: int,
        platform: str,
        system: str,
        source_column: str,
    ):
        """CTR 인사이트 생성"""
        benchmark = self.AD_BENCHMARKS["ctr"]

        if avg_ctr >= benchmark["excellent"]:
            severity = InsightSeverity.INFO
            title = f"Excellent CTR Performance - {platform}"
            description = f"평균 CTR {avg_ctr*100:.2f}%로 업계 평균({benchmark['avg']*100:.1f}%) 대비 {((avg_ctr/benchmark['avg'])-1)*100:.0f}% 우수"
            recommendations = [
                "현재 크리에이티브 전략 유지",
                "성공 요인 분석하여 다른 캠페인에 적용",
                "예산 증액 고려"
            ]
            business_impact = f"높은 CTR로 광고 효율성 극대화, 예상 클릭 비용 절감"
        elif avg_ctr >= benchmark["good"]:
            severity = InsightSeverity.INFO
            title = f"Good CTR Performance - {platform}"
            description = f"평균 CTR {avg_ctr*100:.2f}%로 업계 평균 상회"
            recommendations = ["A/B 테스트로 추가 최적화 시도"]
            business_impact = f"안정적인 CTR 성과 유지 중"
        elif avg_ctr >= benchmark["avg"]:
            severity = InsightSeverity.LOW
            title = f"Average CTR - {platform}"
            description = f"평균 CTR {avg_ctr*100:.2f}%로 업계 평균 수준"
            recommendations = [
                "헤드라인 및 설명 최적화",
                "타겟팅 정교화 검토",
                "크리에이티브 A/B 테스트 진행"
            ]
            business_impact = f"CTR 개선 시 클릭당 비용 절감 가능"
        else:
            severity = InsightSeverity.HIGH
            deviation = ((benchmark["avg"] - avg_ctr) / benchmark["avg"]) * 100
            title = f"Low CTR Alert - {platform}"
            description = f"평균 CTR {avg_ctr*100:.2f}%로 업계 평균({benchmark['avg']*100:.1f}%) 대비 {deviation:.0f}% 하락"
            recommendations = [
                "즉시 광고 크리에이티브 검토 필요",
                "타겟 오디언스 재설정",
                "경쟁사 광고 분석",
                "키워드 관련성 검토"
            ]
            business_impact = f"낮은 CTR로 인한 광고 효율성 저하, CPC 상승 위험"

        insight = self._create_insight(
            title=title,
            description=description,
            insight_type=InsightType.KPI,
            severity=severity,
            source_table=table_name,
            source_column=source_column,
            source_system=system,
            metric_value=avg_ctr,
            metric_name="CTR",
            count=sample_count,
            percentage=avg_ctr * 100,
            benchmark=benchmark["avg"],
            deviation=((avg_ctr / benchmark["avg"]) - 1) * 100 if benchmark["avg"] > 0 else 0,
            recommendations=recommendations,
            business_impact=business_impact,
        )
        self.insights.append(insight)

    def _analyze_cpc(self, table_name: str, rows: List[Dict], platform: str, system: str):
        """CPC(클릭당 비용) 분석"""
        if not rows:
            return

        sample_row = rows[0]
        cpc_col = self._find_ad_column(sample_row, "cpc")

        if not cpc_col:
            return

        cpcs = [
            row.get(cpc_col) for row in rows
            if row.get(cpc_col) is not None
            and isinstance(row.get(cpc_col), (int, float))
            and row.get(cpc_col) > 0
        ]

        if len(cpcs) < 3:
            return

        avg_cpc = statistics.mean(cpcs)

        # micros 단위인 경우 변환 (1,000,000 micros = 1 currency unit)
        if cpc_col and ("micros" in cpc_col.lower() or avg_cpc > 100000):
            avg_cpc = avg_cpc / 1000000

        benchmark = self.AD_BENCHMARKS["cpc"]

        if avg_cpc <= benchmark["excellent"]:
            severity = InsightSeverity.INFO
            title = f"Excellent CPC - {platform}"
            savings = (benchmark["avg"] - avg_cpc) / benchmark["avg"] * 100
            description = f"평균 CPC {avg_cpc:,.0f}원으로 업계 평균 대비 {savings:.0f}% 절감"
            recommendations = ["현재 입찰 전략 유지", "예산 확대 고려"]
            business_impact = f"효율적인 CPC로 동일 예산 대비 더 많은 클릭 확보"
        elif avg_cpc <= benchmark["good"]:
            severity = InsightSeverity.INFO
            title = f"Good CPC - {platform}"
            description = f"평균 CPC {avg_cpc:,.0f}원으로 양호한 수준"
            recommendations = ["추가 최적화로 비용 절감 가능"]
            business_impact = f"CPC 효율 양호"
        elif avg_cpc <= benchmark["avg"]:
            severity = InsightSeverity.LOW
            title = f"Average CPC - {platform}"
            description = f"평균 CPC {avg_cpc:,.0f}원으로 업계 평균 수준"
            recommendations = [
                "품질 점수 개선으로 CPC 절감",
                "롱테일 키워드 활용 검토"
            ]
            business_impact = f"CPC 최적화 시 비용 효율성 개선 가능"
        else:
            severity = InsightSeverity.HIGH
            overspend = (avg_cpc - benchmark["avg"]) / benchmark["avg"] * 100
            title = f"High CPC Alert - {platform}"
            description = f"평균 CPC {avg_cpc:,.0f}원으로 업계 평균 대비 {overspend:.0f}% 초과"
            recommendations = [
                "입찰 전략 즉시 검토",
                "품질 점수 개선 필요",
                "키워드 경쟁도 분석",
                "자동 입찰 전략 고려"
            ]
            business_impact = f"높은 CPC로 인한 광고 예산 비효율, 월간 약 {overspend:.0f}% 추가 비용 발생"

        insight = self._create_insight(
            title=title,
            description=description,
            insight_type=InsightType.KPI,
            severity=severity,
            source_table=table_name,
            source_column=cpc_col,
            source_system=system,
            metric_value=avg_cpc,
            metric_name="CPC",
            count=len(cpcs),
            benchmark=benchmark["avg"],
            recommendations=recommendations,
            business_impact=business_impact,
        )
        self.insights.append(insight)

    def _analyze_conversion_rate(self, table_name: str, rows: List[Dict], platform: str, system: str):
        """전환율 분석"""
        if not rows:
            return

        sample_row = rows[0]
        cvr_col = self._find_ad_column(sample_row, "conversion_rate")

        if not cvr_col:
            # 전환율 계산 시도
            conversions_col = self._find_ad_column(sample_row, "conversions")
            clicks_col = self._find_ad_column(sample_row, "clicks")

            if conversions_col and clicks_col:
                cvrs = []
                for row in rows:
                    conversions = row.get(conversions_col, 0)
                    clicks = row.get(clicks_col, 0)
                    if clicks and clicks > 0:
                        cvrs.append(conversions / clicks)

                if cvrs and len(cvrs) >= 3:
                    avg_cvr = statistics.mean(cvrs)
                    self._generate_cvr_insight(table_name, avg_cvr, len(cvrs), platform, system, "calculated")
            return

        cvrs = [
            row.get(cvr_col) for row in rows
            if row.get(cvr_col) is not None
            and isinstance(row.get(cvr_col), (int, float))
        ]

        if len(cvrs) < 3:
            return

        avg_cvr = statistics.mean(cvrs)
        if avg_cvr > 1:  # 백분율로 저장된 경우
            avg_cvr = avg_cvr / 100

        self._generate_cvr_insight(table_name, avg_cvr, len(cvrs), platform, system, cvr_col)

    def _generate_cvr_insight(
        self,
        table_name: str,
        avg_cvr: float,
        sample_count: int,
        platform: str,
        system: str,
        source_column: str,
    ):
        """전환율 인사이트 생성"""
        benchmark = self.AD_BENCHMARKS["conversion_rate"]

        if avg_cvr >= benchmark["excellent"]:
            severity = InsightSeverity.INFO
            title = f"Excellent Conversion Rate - {platform}"
            description = f"평균 전환율 {avg_cvr*100:.2f}%로 업계 최고 수준"
            recommendations = ["성공 요인 분석 및 확산"]
            business_impact = "높은 전환율로 ROI 극대화"
        elif avg_cvr >= benchmark["good"]:
            severity = InsightSeverity.INFO
            title = f"Good Conversion Rate - {platform}"
            description = f"평균 전환율 {avg_cvr*100:.2f}%로 양호"
            recommendations = ["랜딩 페이지 A/B 테스트로 추가 개선"]
            business_impact = "안정적인 전환 성과"
        elif avg_cvr >= benchmark["avg"]:
            severity = InsightSeverity.LOW
            title = f"Average Conversion Rate - {platform}"
            description = f"평균 전환율 {avg_cvr*100:.2f}%로 업계 평균 수준"
            recommendations = [
                "랜딩 페이지 최적화",
                "CTA 개선",
                "타겟팅 정교화"
            ]
            business_impact = "전환율 개선 시 매출 증대 기대"
        else:
            severity = InsightSeverity.HIGH
            title = f"Low Conversion Rate Alert - {platform}"
            description = f"평균 전환율 {avg_cvr*100:.2f}%로 업계 평균 미달"
            recommendations = [
                "랜딩 페이지 UX 전면 검토",
                "광고-랜딩 메시지 일치성 확인",
                "전환 퍼널 분석",
                "타겟 오디언스 재검토"
            ]
            business_impact = f"낮은 전환율로 광고 투자 대비 매출 저조"

        insight = self._create_insight(
            title=title,
            description=description,
            insight_type=InsightType.KPI,
            severity=severity,
            source_table=table_name,
            source_column=source_column,
            source_system=system,
            metric_value=avg_cvr,
            metric_name="Conversion Rate",
            count=sample_count,
            percentage=avg_cvr * 100,
            benchmark=benchmark["avg"],
            recommendations=recommendations,
            business_impact=business_impact,
        )
        self.insights.append(insight)

    def _analyze_roas(self, table_name: str, rows: List[Dict], platform: str, system: str):
        """ROAS(광고 투자 수익률) 분석"""
        if not rows:
            return

        sample_row = rows[0]
        roas_col = self._find_ad_column(sample_row, "roas")

        if not roas_col:
            # ROAS 계산 시도
            conversion_value_col = None
            spend_col = self._find_ad_column(sample_row, "spend")

            for col in sample_row.keys():
                if col and ("conversion_value" in col.lower() or "revenue" in col.lower()):
                    conversion_value_col = col
                    break

            if conversion_value_col and spend_col:
                roas_values = []
                for row in rows:
                    value = row.get(conversion_value_col, 0)
                    spend = row.get(spend_col, 0)
                    if spend and spend > 0:
                        roas_values.append(value / spend)

                if roas_values and len(roas_values) >= 3:
                    avg_roas = statistics.mean(roas_values)
                    self._generate_roas_insight(table_name, avg_roas, len(roas_values), platform, system, "calculated")
            return

        roas_values = [
            row.get(roas_col) for row in rows
            if row.get(roas_col) is not None
            and isinstance(row.get(roas_col), (int, float))
            and row.get(roas_col) > 0
        ]

        if len(roas_values) < 3:
            return

        avg_roas = statistics.mean(roas_values)
        self._generate_roas_insight(table_name, avg_roas, len(roas_values), platform, system, roas_col)

    def _generate_roas_insight(
        self,
        table_name: str,
        avg_roas: float,
        sample_count: int,
        platform: str,
        system: str,
        source_column: str,
    ):
        """ROAS 인사이트 생성"""
        benchmark = self.AD_BENCHMARKS["roas"]

        if avg_roas >= benchmark["excellent"]:
            severity = InsightSeverity.INFO
            title = f"Excellent ROAS - {platform}"
            description = f"ROAS {avg_roas:.1f}x로 광고비 1원당 {avg_roas:.1f}원 매출"
            recommendations = ["예산 증액으로 성과 확대"]
            business_impact = f"우수한 광고 ROI, 예산 대비 {avg_roas*100:.0f}% 매출 창출"
        elif avg_roas >= benchmark["good"]:
            severity = InsightSeverity.INFO
            title = f"Good ROAS - {platform}"
            description = f"ROAS {avg_roas:.1f}x로 양호한 수익률"
            recommendations = ["고성과 캠페인 분석 및 확산"]
            business_impact = f"안정적인 광고 수익률"
        elif avg_roas >= benchmark["avg"]:
            severity = InsightSeverity.LOW
            title = f"Average ROAS - {platform}"
            description = f"ROAS {avg_roas:.1f}x로 손익분기 상회"
            recommendations = [
                "저성과 캠페인 예산 재분배",
                "고가치 오디언스 타겟팅 강화"
            ]
            business_impact = f"ROAS 개선 시 수익성 향상 가능"
        elif avg_roas >= 1.0:
            severity = InsightSeverity.MEDIUM
            title = f"Low ROAS Warning - {platform}"
            description = f"ROAS {avg_roas:.1f}x로 손익분기 수준"
            recommendations = [
                "캠페인별 성과 분석 후 저효율 캠페인 중단 검토",
                "타겟팅 및 크리에이티브 전면 검토",
                "입찰 전략 변경 고려"
            ]
            business_impact = f"마진 거의 없음, 개선 필요"
        else:
            severity = InsightSeverity.CRITICAL
            title = f"Negative ROAS Alert - {platform}"
            description = f"ROAS {avg_roas:.1f}x로 광고비 대비 손실 발생"
            recommendations = [
                "즉시 캠페인 검토 및 저효율 캠페인 중단",
                "타겟팅 전면 재설정",
                "전환 추적 정확성 확인",
                "경쟁사 대비 가격 경쟁력 검토"
            ]
            business_impact = f"광고비 손실 발생, 즉각적인 조치 필요"

        insight = self._create_insight(
            title=title,
            description=description,
            insight_type=InsightType.KPI,
            severity=severity,
            source_table=table_name,
            source_column=source_column,
            source_system=system,
            metric_value=avg_roas,
            metric_name="ROAS",
            count=sample_count,
            benchmark=benchmark["avg"],
            recommendations=recommendations,
            business_impact=business_impact,
        )
        self.insights.append(insight)

    def _analyze_ad_frequency(self, table_name: str, rows: List[Dict], platform: str, system: str):
        """광고 빈도(피로도) 분석"""
        if not rows:
            return

        sample_row = rows[0]
        freq_col = self._find_ad_column(sample_row, "frequency")

        if not freq_col:
            return

        frequencies = [
            row.get(freq_col) for row in rows
            if row.get(freq_col) is not None
            and isinstance(row.get(freq_col), (int, float))
            and row.get(freq_col) > 0
        ]

        if len(frequencies) < 3:
            return

        avg_freq = statistics.mean(frequencies)
        max_freq = max(frequencies)
        benchmark = self.AD_BENCHMARKS["frequency"]

        if max_freq >= benchmark["critical"]:
            severity = InsightSeverity.CRITICAL
            title = f"Critical Ad Fatigue - {platform}"
            description = f"최대 빈도 {max_freq:.1f}회로 광고 피로도 위험 수준"
            recommendations = [
                "즉시 빈도 캡 설정",
                "오디언스 확장 필요",
                "크리에이티브 로테이션 추가"
            ]
            business_impact = "심각한 광고 피로로 효율 급락 및 브랜드 이미지 손상 위험"
        elif avg_freq >= benchmark["warning"]:
            severity = InsightSeverity.HIGH
            title = f"High Ad Frequency Warning - {platform}"
            description = f"평균 빈도 {avg_freq:.1f}회로 광고 피로 우려"
            recommendations = [
                "빈도 캡 설정 권장",
                "새로운 크리에이티브 준비",
                "타겟 오디언스 확장 검토"
            ]
            business_impact = "광고 피로로 인한 CTR/전환율 하락 예상"
        elif avg_freq >= benchmark["avg"]:
            severity = InsightSeverity.LOW
            title = f"Normal Ad Frequency - {platform}"
            description = f"평균 빈도 {avg_freq:.1f}회로 적정 수준"
            recommendations = ["현재 수준 유지 권장"]
            business_impact = "적정 광고 노출 빈도 유지 중"
        else:
            severity = InsightSeverity.INFO
            title = f"Low Ad Frequency - {platform}"
            description = f"평균 빈도 {avg_freq:.1f}회로 노출 기회 여유 있음"
            recommendations = ["예산 증액으로 도달 확대 가능"]
            business_impact = "추가 노출 여력 존재"

        insight = self._create_insight(
            title=title,
            description=description,
            insight_type=InsightType.OPERATIONAL,
            severity=severity,
            source_table=table_name,
            source_column=freq_col,
            source_system=system,
            metric_value=avg_freq,
            metric_name="Frequency",
            count=len(frequencies),
            benchmark=benchmark["avg"],
            recommendations=recommendations,
            business_impact=business_impact,
        )
        self.insights.append(insight)

    def _analyze_quality_score(self, table_name: str, rows: List[Dict], platform: str, system: str):
        """품질 점수 분석"""
        if not rows:
            return

        sample_row = rows[0]
        qs_col = self._find_ad_column(sample_row, "quality_score")

        if not qs_col:
            return

        scores = [
            row.get(qs_col) for row in rows
            if row.get(qs_col) is not None
            and isinstance(row.get(qs_col), (int, float))
        ]

        if len(scores) < 3:
            return

        avg_score = statistics.mean(scores)
        low_score_count = sum(1 for s in scores if s < 6)
        benchmark = self.AD_BENCHMARKS["quality_score"]

        if avg_score >= benchmark["excellent"]:
            severity = InsightSeverity.INFO
            title = f"Excellent Quality Score - {platform}"
            description = f"평균 품질 점수 {avg_score:.1f}점으로 최상위 수준"
            recommendations = ["현재 키워드-광고 관련성 유지"]
            business_impact = "높은 품질 점수로 CPC 절감 효과"
        elif avg_score >= benchmark["good"]:
            severity = InsightSeverity.INFO
            title = f"Good Quality Score - {platform}"
            description = f"평균 품질 점수 {avg_score:.1f}점으로 양호"
            recommendations = ["랜딩 페이지 경험 개선으로 추가 향상 가능"]
            business_impact = "안정적인 품질 점수 유지"
        elif avg_score >= benchmark["avg"]:
            severity = InsightSeverity.LOW
            title = f"Average Quality Score - {platform}"
            description = f"평균 품질 점수 {avg_score:.1f}점으로 개선 여지 있음"
            recommendations = [
                "광고 관련성 개선",
                "랜딩 페이지 로딩 속도 개선",
                "키워드-광고 문구 일치성 강화"
            ]
            business_impact = "품질 점수 개선 시 CPC 10-15% 절감 가능"
        else:
            severity = InsightSeverity.HIGH
            title = f"Low Quality Score Alert - {platform}"
            description = f"평균 품질 점수 {avg_score:.1f}점, 낮은 점수 광고 {low_score_count}개"
            recommendations = [
                "키워드 관련성 즉시 검토",
                "광고 문구 재작성",
                "랜딩 페이지 전면 개선",
                "낮은 품질 키워드 제거 또는 개선"
            ]
            business_impact = f"낮은 품질 점수로 CPC 20-50% 추가 지불, {low_score_count}개 광고 개선 필요"

        insight = self._create_insight(
            title=title,
            description=description,
            insight_type=InsightType.KPI,
            severity=severity,
            source_table=table_name,
            source_column=qs_col,
            source_system=system,
            metric_value=avg_score,
            metric_name="Quality Score",
            count=len(scores),
            benchmark=benchmark["avg"],
            recommendations=recommendations,
            business_impact=business_impact,
        )
        self.insights.append(insight)

    def _analyze_budget_efficiency(self, table_name: str, rows: List[Dict], platform: str, system: str):
        """예산 효율성 분석"""
        if not rows:
            return

        sample_row = rows[0]
        spend_col = self._find_ad_column(sample_row, "spend")
        conversions_col = self._find_ad_column(sample_row, "conversions")

        if not spend_col or not conversions_col:
            return

        total_spend = 0
        total_conversions = 0

        for row in rows:
            spend = row.get(spend_col, 0)
            conversions = row.get(conversions_col, 0)

            if isinstance(spend, (int, float)):
                total_spend += spend
            if isinstance(conversions, (int, float)):
                total_conversions += conversions

        if total_spend <= 0 or total_conversions <= 0:
            return

        cpa = total_spend / total_conversions

        # micros 단위 변환
        if spend_col and ("micros" in spend_col.lower() or cpa > 1000000):
            cpa = cpa / 1000000

        # 동적 벤치마크 (업종별로 다름, 일반적으로 CPC의 10-20배)
        cpc_benchmark = self.AD_BENCHMARKS["cpc"]["avg"]
        cpa_benchmark = cpc_benchmark * 15  # 평균 전환율 기준

        if cpa <= cpa_benchmark * 0.5:
            severity = InsightSeverity.INFO
            title = f"Excellent CPA - {platform}"
            description = f"CPA {cpa:,.0f}원으로 매우 효율적인 전환 비용"
            recommendations = ["예산 증액으로 전환 확대"]
            business_impact = "우수한 전환 비용 효율"
        elif cpa <= cpa_benchmark:
            severity = InsightSeverity.LOW
            title = f"Good CPA - {platform}"
            description = f"CPA {cpa:,.0f}원으로 적정 수준"
            recommendations = ["고효율 캠페인 분석으로 추가 최적화"]
            business_impact = "안정적인 전환 비용"
        else:
            severity = InsightSeverity.HIGH
            overspend = (cpa - cpa_benchmark) / cpa_benchmark * 100
            title = f"High CPA Alert - {platform}"
            description = f"CPA {cpa:,.0f}원으로 목표 대비 {overspend:.0f}% 초과"
            recommendations = [
                "저효율 캠페인 예산 삭감",
                "전환율 개선 우선",
                "고가치 오디언스 타겟팅 강화"
            ]
            business_impact = f"높은 CPA로 수익성 저하, 전환당 약 {cpa-cpa_benchmark:,.0f}원 추가 비용"

        insight = self._create_insight(
            title=title,
            description=description,
            insight_type=InsightType.KPI,
            severity=severity,
            source_table=table_name,
            source_column=f"{spend_col}/{conversions_col}",
            source_system=system,
            metric_value=cpa,
            metric_name="CPA",
            count=int(total_conversions),
            benchmark=cpa_benchmark,
            recommendations=recommendations,
            business_impact=business_impact,
        )
        self.insights.append(insight)

    def _analyze_campaign_anomalies(self, table_name: str, rows: List[Dict], platform: str, system: str):
        """캠페인 성과 이상치 탐지"""
        if not rows or len(rows) < 5:
            return

        sample_row = rows[0]

        # 캠페인 ID 컬럼 찾기
        campaign_col = None
        for col in sample_row.keys():
            if col and "campaign" in col.lower() and "id" in col.lower():
                campaign_col = col
                break

        if not campaign_col:
            return

        # 캠페인별 성과 집계
        campaign_metrics = defaultdict(lambda: {"impressions": 0, "clicks": 0, "spend": 0, "conversions": 0})

        impressions_col = self._find_ad_column(sample_row, "impressions")
        clicks_col = self._find_ad_column(sample_row, "clicks")
        spend_col = self._find_ad_column(sample_row, "spend")
        conversions_col = self._find_ad_column(sample_row, "conversions")

        for row in rows:
            campaign_id = row.get(campaign_col)
            if not campaign_id:
                continue

            if impressions_col:
                val = row.get(impressions_col, 0)
                if isinstance(val, (int, float)):
                    campaign_metrics[campaign_id]["impressions"] += val
            if clicks_col:
                val = row.get(clicks_col, 0)
                if isinstance(val, (int, float)):
                    campaign_metrics[campaign_id]["clicks"] += val
            if spend_col:
                val = row.get(spend_col, 0)
                if isinstance(val, (int, float)):
                    campaign_metrics[campaign_id]["spend"] += val
            if conversions_col:
                val = row.get(conversions_col, 0)
                if isinstance(val, (int, float)):
                    campaign_metrics[campaign_id]["conversions"] += val

        if len(campaign_metrics) < 2:
            return

        # CTR 계산 및 이상치 탐지
        ctrs = []
        for metrics in campaign_metrics.values():
            if metrics["impressions"] > 0:
                ctrs.append(metrics["clicks"] / metrics["impressions"])

        if len(ctrs) >= 3:
            avg_ctr = statistics.mean(ctrs)
            std_ctr = statistics.stdev(ctrs) if len(ctrs) > 1 else 0

            # 이상치 캠페인 찾기
            anomaly_campaigns = []
            for campaign_id, metrics in campaign_metrics.items():
                if metrics["impressions"] > 0:
                    campaign_ctr = metrics["clicks"] / metrics["impressions"]
                    if std_ctr > 0 and abs(campaign_ctr - avg_ctr) > 2 * std_ctr:
                        anomaly_campaigns.append({
                            "campaign_id": campaign_id,
                            "ctr": campaign_ctr,
                            "deviation": (campaign_ctr - avg_ctr) / avg_ctr * 100 if avg_ctr > 0 else 0
                        })

            if anomaly_campaigns:
                # 성과 급락 캠페인
                low_performers = [c for c in anomaly_campaigns if c["deviation"] < 0]
                high_performers = [c for c in anomaly_campaigns if c["deviation"] > 0]

                if low_performers:
                    worst = min(low_performers, key=lambda x: x["deviation"])
                    insight = self._create_insight(
                        title=f"Campaign Performance Drop - {platform}",
                        description=f"캠페인 {worst['campaign_id']}의 CTR이 평균 대비 {abs(worst['deviation']):.0f}% 하락",
                        insight_type=InsightType.ANOMALY,
                        severity=InsightSeverity.HIGH,
                        source_table=table_name,
                        source_column=campaign_col,
                        source_system=system,
                        metric_value=worst["ctr"],
                        metric_name="Campaign CTR",
                        count=len(low_performers),
                        deviation=worst["deviation"],
                        recommendations=[
                            f"캠페인 {worst['campaign_id']} 즉시 점검 필요",
                            "크리에이티브 피로도 확인",
                            "타겟팅 변경 여부 확인",
                            "경쟁 환경 변화 분석"
                        ],
                        business_impact=f"{len(low_performers)}개 캠페인 성과 하락, 긴급 점검 필요",
                    )
                    self.insights.append(insight)

                if high_performers:
                    best = max(high_performers, key=lambda x: x["deviation"])
                    insight = self._create_insight(
                        title=f"Top Performing Campaign - {platform}",
                        description=f"캠페인 {best['campaign_id']}의 CTR이 평균 대비 {best['deviation']:.0f}% 상회",
                        insight_type=InsightType.KPI,
                        severity=InsightSeverity.INFO,
                        source_table=table_name,
                        source_column=campaign_col,
                        source_system=system,
                        metric_value=best["ctr"],
                        metric_name="Campaign CTR",
                        count=len(high_performers),
                        deviation=best["deviation"],
                        recommendations=[
                            f"캠페인 {best['campaign_id']} 성공 요인 분석",
                            "해당 캠페인 예산 증액 검토",
                            "유사 전략 다른 캠페인에 적용"
                        ],
                        business_impact=f"{len(high_performers)}개 고성과 캠페인 발견, 확산 전략 수립 권장",
                    )
                    self.insights.append(insight)

    # === v26.1: 카테고리별 세그먼트 성과 분석 ===

    # 세그먼트 분석 시 제외할 컬럼 — 토큰 정확 매칭 (underscore 구분)
    _SEGMENT_SKIP_TOKENS = {
        "id", "key", "uuid", "url", "title", "description",
        "name", "text", "suggestion", "factor",
    }
    # 컬럼명 정확 매칭 (전체 이름)
    _SEGMENT_SKIP_EXACT = {
        "s3_key", "username", "channel_name", "hashtags",
    }

    def _should_skip_segment_col(self, col: str) -> bool:
        """토큰 기반 매칭으로 분석 제외 컬럼 판별 (hash→hashtag 오탐 방지)"""
        col_lower = col.lower()
        if col_lower in self._SEGMENT_SKIP_EXACT:
            return True
        tokens = set(col_lower.replace("-", "_").split("_"))
        return bool(tokens & self._SEGMENT_SKIP_TOKENS)

    def _analyze_segment_performance(
        self,
        table_name: str,
        rows: List[Dict],
        table_info: Dict,
        system: str,
    ):
        """
        v26.1: 카테고리 컬럼별 세그먼트 성과 비교 (generic, 도메인 무관)

        - 카테고리 컬럼 자동 감지 (cardinality 2-20, string/bool)
        - 숫자 KPI 컬럼 자동 감지
        - 그룹별 평균 비교 → 유의미한 차이(2x+) 보고
        - Duration 버킷 분석 (duration 컬럼 존재 시)
        """
        if not rows or len(rows) < 30:
            return

        sample_row = rows[0]
        total_rows = len(rows)

        # v28.0: Palantir Two-pass — pre-computed statistics from table_info
        # unified_main.py에서 이미 계산된 unique_count, null_count, sample_values 재사용
        # Python 루프로 rows를 재스캔하지 않음 → 297컬럼 × 1000행 = 297K 반복 제거
        col_info_map = {c["name"]: c for c in table_info.get("columns", [])}

        # --- 1. 카테고리 컬럼 감지 (Two-pass: pre-computed stats 활용) ---
        categorical_cols = []
        df = getattr(self, '_current_df', None)

        if col_info_map and df is not None and not df.empty:
            # v28.0 fast path: table_info 통계 기반 O(cols) 감지 (rows 스캔 없음)
            for col in sample_row.keys():
                if not col or self._should_skip_segment_col(col):
                    continue
                cinfo = col_info_map.get(col, {})
                unique_count = cinfo.get("unique_count", -1)
                null_count = cinfo.get("null_count", 0)
                if unique_count < 2 or unique_count > 50:
                    continue
                if (total_rows - null_count) < total_rows * 0.5:
                    continue
                sample_vals = cinfo.get("sample_values", [])
                sample_val = str(sample_vals[0]) if sample_vals else ""
                if not self._is_numeric_string(sample_val) or unique_count <= 6:
                    categorical_cols.append((col, unique_count, None))
        else:
            # fallback: 기존 rows 스캔 방식
            for col in sample_row.keys():
                if not col or self._should_skip_segment_col(col):
                    continue
                values = [str(row.get(col, "")) for row in rows if row.get(col) is not None and str(row.get(col)).strip()]
                if len(values) < total_rows * 0.5:
                    continue
                unique_vals = set(values)
                cardinality = len(unique_vals)
                if 2 <= cardinality <= 50:
                    sample_val = next(iter(unique_vals))
                    if not self._is_numeric_string(sample_val) or cardinality <= 6:
                        categorical_cols.append((col, cardinality, unique_vals))

        # --- 2. 숫자 KPI 컬럼 감지 (Two-pass: dtype + null_count 활용) ---
        numeric_kpi_cols = []
        if col_info_map and df is not None and not df.empty:
            # v28.0 fast path: pandas dtype 기반 O(cols) 감지
            import pandas as _pd
            for col in sample_row.keys():
                if not col or self._should_skip_segment_col(col) or not self._is_business_metric_column(col):
                    continue
                cinfo = col_info_map.get(col, {})
                col_dtype = cinfo.get("type", "object")
                null_count = cinfo.get("null_count", 0)
                # numeric dtype이고 non-null 50% 이상
                if any(t in col_dtype for t in ("int", "float")):
                    if (total_rows - null_count) >= total_rows * 0.5:
                        numeric_kpi_cols.append(col)
                elif col in df.columns and _pd.api.types.is_numeric_dtype(df[col]):
                    non_null = df[col].count()
                    if non_null >= total_rows * 0.5:
                        numeric_kpi_cols.append(col)
        else:
            # fallback: 기존 rows 스캔 방식
            for col in sample_row.keys():
                if not col or self._should_skip_segment_col(col) or not self._is_business_metric_column(col):
                    continue
                values = [
                    row.get(col) for row in rows
                    if row.get(col) is not None and isinstance(row.get(col), (int, float))
                ]
                if len(values) >= total_rows * 0.5:
                    numeric_kpi_cols.append(col)

        if not categorical_cols or not numeric_kpi_cols:
            return

        # --- 3. KPI 선택: 모든 비즈니스 숫자 컬럼 사용 (최대 15개) ---
        # CV 기반 제한은 view_count 같은 핵심 KPI를 누락시킴.
        # _is_business_metric_column 필터가 이미 ID/키 컬럼을 제외하므로 전부 사용.
        target_kpis = numeric_kpi_cols[:15]

        # --- 4. v27.0: 세그먼트별 통합 분석 (카테고리별 top 3 gap 통합) ---
        for cat_col, cardinality, unique_vals in categorical_cols:
            self._compare_segments_consolidated(table_name, rows, cat_col, target_kpis, system)

        # --- 5. 연속형 숫자 컬럼 버킷 분석 (분위수 기반) ---
        # v27.4: KPI 컬럼이 아닌 feature 컬럼 우선 + 후보 8개로 확대
        kpi_set = set(target_kpis)
        feature_candidates = []
        kpi_candidates = []
        for col in sample_row.keys():
            if not col or not self._is_business_metric_column(col):
                continue
            if self._should_skip_segment_col(col):
                continue
            if any(col == cc[0] for cc in categorical_cols):
                continue
            vals = [row.get(col) for row in rows if row.get(col) is not None and isinstance(row.get(col), (int, float))]
            if len(vals) >= total_rows * 0.5:
                unique_ratio = len(set(vals)) / len(vals)
                if unique_ratio > 0.05:  # v27.4.3: 0.1→0.05 (duration=0.097 탈락 방지)
                    if col in kpi_set:
                        kpi_candidates.append((col, unique_ratio))
                    else:
                        feature_candidates.append((col, unique_ratio))

        # v27.4.2: feature 컬럼 전체 + KPI 전체 분석 (duration 포함 보장)
        feature_candidates.sort(key=lambda x: -x[1])
        kpi_candidates.sort(key=lambda x: -x[1])
        continuous_candidates = feature_candidates + kpi_candidates
        for cont_col, _ in continuous_candidates:
            self._analyze_numeric_buckets(table_name, rows, cont_col, target_kpis, system)

    def _is_numeric_string(self, val: str) -> bool:
        """문자열이 숫자인지 판별"""
        try:
            float(val)
            return True
        except (ValueError, TypeError):
            return False

    @staticmethod
    def _robust_mean(vals: list) -> float:
        """v27.4.3: CV 기반 적응적 평균.
        - CV > 10 (극단 분포: engagement_rate 12M%): median 사용
        - otherwise: 일반 mean (합법적 스큐 보존: view_count CV=6.6 등)
        """
        if len(vals) < 5:
            return statistics.mean(vals)
        mean_val = statistics.mean(vals)
        if mean_val == 0:
            return 0.0
        stdev = statistics.stdev(vals) if len(vals) > 1 else 0
        cv = abs(stdev / mean_val) if mean_val != 0 else 0

        if cv > 10:
            return statistics.median(vals)
        return mean_val

    def _compare_segments_consolidated(
        self,
        table_name: str,
        rows: List[Dict],
        cat_col: str,
        kpi_cols: List[str],
        system: str,
    ):
        """v27.0: 카테고리 컬럼 당 모든 KPI gap을 수집, top 3만 하나의 인사이트로 통합."""
        significant_gaps = []
        non_effect_kpis = []  # v27.4.1: gap < 1.2x인 KPI 추적
        tested_kpi_count = 0
        max_observed_ratio = 1.0

        # v28.0: pandas groupby vectorized — O(kpis × rows) Python 루프 → C-level 연산
        # 정확도: groupby mean/median은 statistics.mean/median과 수학적으로 동일
        import pandas as _pd
        import numpy as _np
        df = getattr(self, '_current_df', None)
        use_df = (df is not None and not df.empty
                  and cat_col in df.columns
                  and all(k in df.columns for k in kpi_cols))

        for kpi_col in kpi_cols:
            if use_df:
                # v28.0 fast path: pandas groupby (C-level, 100x faster)
                try:
                    df_sub = df[[cat_col, kpi_col]].copy()
                    df_sub[cat_col] = df_sub[cat_col].astype(str)
                    df_sub[kpi_col] = _pd.to_numeric(df_sub[kpi_col], errors='coerce')
                    df_sub = df_sub.dropna()
                    grp_counts = df_sub.groupby(cat_col)[kpi_col].count()
                    valid_cats = grp_counts[grp_counts >= 10].index
                    if len(valid_cats) < 2:
                        continue
                    df_valid = df_sub[df_sub[cat_col].isin(valid_cats)]
                    all_kpi_vals_arr = df_valid[kpi_col].values
                    kpi_mean_all = float(_np.mean(all_kpi_vals_arr))
                    if kpi_mean_all == 0:
                        continue
                    kpi_cv_all = float(_np.std(all_kpi_vals_arr) / abs(kpi_mean_all)) if len(all_kpi_vals_arr) > 1 else 0
                    use_median = kpi_cv_all > 10
                    agg_func = 'median' if use_median else 'mean'
                    grp_agg = df_valid.groupby(cat_col)[kpi_col].agg(['mean', 'median', 'count'])
                    group_stats = {
                        str(idx): {"n": int(row['count']), "mean": float(row[agg_func])}
                        for idx, row in grp_agg.iterrows()
                    }
                    valid_groups = group_stats  # 이미 count >= 10 필터 적용됨
                except Exception:
                    use_df = False  # 오류 시 fallback으로 전환
                    df = None

            if not use_df:
                # fallback: 기존 Python 루프 방식
                groups = defaultdict(list)
                for row in rows:
                    cat_val = row.get(cat_col)
                    kpi_val = row.get(kpi_col)
                    if cat_val is not None and kpi_val is not None and isinstance(kpi_val, (int, float)):
                        groups[str(cat_val)].append(kpi_val)
                valid_groups = {k: v for k, v in groups.items() if len(v) >= 10}
                if len(valid_groups) < 2:
                    continue
                all_kpi_vals = [v for vals in valid_groups.values() for v in vals]
                kpi_mean_all = statistics.mean(all_kpi_vals) if all_kpi_vals else 0
                if kpi_mean_all == 0:
                    continue
                kpi_cv_all = abs(statistics.stdev(all_kpi_vals) / kpi_mean_all) if len(all_kpi_vals) > 1 else 0
                use_median = kpi_cv_all > 10
                group_stats = {}
                for grp, vals in valid_groups.items():
                    agg = statistics.median(vals) if use_median else statistics.mean(vals)
                    group_stats[grp] = {"n": len(vals), "mean": agg}
                valid_groups = group_stats

            means = [s["mean"] for s in group_stats.values()]
            max_mean = max(means)
            min_mean = min(means)

            if min_mean <= 0:
                continue

            tested_kpi_count += 1
            ratio = max_mean / min_mean
            max_observed_ratio = max(max_observed_ratio, ratio)

            if ratio < 1.2:
                # v27.4.1: 거의 동일한 KPI 추적 (null finding)
                non_effect_kpis.append((kpi_col, ratio))

            if ratio < 2.0:
                continue

            best_group = max(group_stats, key=lambda g: group_stats[g]["mean"])
            worst_group = min(group_stats, key=lambda g: group_stats[g]["mean"])
            significant_gaps.append((kpi_col, ratio, best_group, worst_group, group_stats))

        # v27.0 Fix 6: Null-Finding 감지 — 효과 없는 카테고리 보고
        if not significant_gaps:
            if tested_kpi_count >= 3 and max_observed_ratio < 1.5:
                insight = self._create_insight(
                    title=f"No Significant Effect: {cat_col}",
                    description=(
                        f"{cat_col} shows no significant performance gap across {tested_kpi_count} KPIs tested "
                        f"(max ratio {max_observed_ratio:.2f}x). This variable does not meaningfully drive outcomes."
                    ),
                    insight_type=InsightType.KPI,
                    severity=InsightSeverity.LOW,
                    source_table=table_name,
                    source_column=cat_col,
                    source_system=system,
                    metric_value=max_observed_ratio,
                    metric_name=f"null_effect_{cat_col}",
                    recommendations=[
                        f"Deprioritize {cat_col} as a segmentation variable",
                        f"Focus optimization efforts on other dimensions with larger impact",
                    ],
                    business_impact=f"{cat_col} is not a meaningful performance driver (max gap {max_observed_ratio:.2f}x)",
                )
                self.insights.append(insight)
            return

        # top 3 유의미한 gap만 하나의 인사이트로 통합
        significant_gaps.sort(key=lambda x: -x[1])
        top_gaps = significant_gaps[:3]

        gap_details = []
        for kpi_col, ratio, best_grp, worst_grp, stats in top_gaps:
            gap_details.append(
                f"{kpi_col}: {best_grp}({stats[best_grp]['mean']:,.1f}) vs "
                f"{worst_grp}({stats[worst_grp]['mean']:,.1f}) = {ratio:.1f}x"
            )

        max_ratio = top_gaps[0][1]

        # v27.4.1: non-effect KPI 보고 (gap 있는데 특정 KPI는 영향 없음 → null finding)
        non_effect_note = ""
        if non_effect_kpis and len(non_effect_kpis) >= 1:
            ne_names = [f"{k}({r:.2f}x)" for k, r in non_effect_kpis[:5]]
            non_effect_note = f" Notable non-effects: {', '.join(ne_names)} show no meaningful variation by {cat_col}."

        insight = self._create_insight(
            title=f"Segment Performance: {cat_col} ({len(significant_gaps)} KPI gaps)",
            description=(
                f"{cat_col} shows significant performance gaps across {len(significant_gaps)} KPIs. "
                f"Top gaps: {'; '.join(gap_details)}"
                f"{non_effect_note}"
            ),
            insight_type=InsightType.KPI,
            severity=InsightSeverity.HIGH if max_ratio >= 5.0 else InsightSeverity.MEDIUM,
            source_table=table_name,
            source_column=cat_col,
            source_system=system,
            metric_value=max_ratio,
            metric_name=f"segment_gap_{cat_col}",
            count=sum(top_gaps[0][4][g]["n"] for g in top_gaps[0][4]),
            percentage=max_ratio,
            recommendations=[
                f"Investigate why {top_gaps[0][2]} consistently outperforms across {len(significant_gaps)} KPIs",
                f"Consider stratified analysis by {cat_col} for targeted optimization",
            ],
            business_impact=f"{cat_col} drives up to {max_ratio:.1f}x gap across {len(significant_gaps)} metrics",
        )
        self.insights.append(insight)

    def _compare_segments(
        self,
        table_name: str,
        rows: List[Dict],
        cat_col: str,
        kpi_col: str,
        system: str,
    ):
        """카테고리 컬럼의 각 그룹별 KPI 평균 비교"""
        groups = defaultdict(list)
        for row in rows:
            cat_val = row.get(cat_col)
            kpi_val = row.get(kpi_col)
            if cat_val is not None and kpi_val is not None and isinstance(kpi_val, (int, float)):
                groups[str(cat_val)].append(kpi_val)

        # 최소 2그룹, 각 그룹 최소 10건
        valid_groups = {k: v for k, v in groups.items() if len(v) >= 10}
        if len(valid_groups) < 2:
            return

        # 그룹별 평균 계산
        group_stats = {}
        for grp, vals in valid_groups.items():
            group_stats[grp] = {
                "n": len(vals),
                "mean": statistics.mean(vals),
                "median": statistics.median(vals),
            }

        means = [s["mean"] for s in group_stats.values()]
        max_mean = max(means)
        min_mean = min(means)

        # 유의미한 차이: max/min >= 2x (0인 경우 제외)
        if min_mean <= 0 or max_mean / min_mean < 2.0:
            return

        ratio = max_mean / min_mean
        best_group = max(group_stats, key=lambda g: group_stats[g]["mean"])
        worst_group = min(group_stats, key=lambda g: group_stats[g]["mean"])

        # 그룹별 통계 문자열
        details = "; ".join(
            f"{grp}(n={s['n']}, avg={s['mean']:,.1f})"
            for grp, s in sorted(group_stats.items(), key=lambda x: -x[1]["mean"])
        )

        insight = self._create_insight(
            title=f"Segment Gap: {cat_col} → {kpi_col}",
            description=(
                f"{best_group} has {ratio:.1f}x higher avg {kpi_col} than {worst_group}. "
                f"[{details}]"
            ),
            insight_type=InsightType.KPI,
            severity=InsightSeverity.HIGH if ratio >= 5.0 else InsightSeverity.MEDIUM,
            source_table=table_name,
            source_column=cat_col,
            source_system=system,
            metric_value=max_mean,
            metric_name=f"{cat_col}→{kpi_col}",
            count=sum(s["n"] for s in group_stats.values()),
            percentage=ratio,
            recommendations=[
                f"Investigate why {best_group} outperforms {worst_group} by {ratio:.1f}x on {kpi_col}",
                f"Consider stratified analysis by {cat_col} for deeper insights",
            ],
            business_impact=f"{cat_col} segment drives {ratio:.1f}x performance gap in {kpi_col}",
        )
        self.insights.append(insight)

    def _analyze_categorical_distributions(
        self,
        table_name: str,
        rows: List[Dict],
        table_info: Dict,
        system: str,
    ):
        """v27.0: 카테고리 컬럼의 분포 인사이트 (Language Distribution 등)."""
        if not rows or len(rows) < 50:
            return

        from collections import Counter
        total_rows = len(rows)
        sample_row = rows[0]

        for col in sample_row.keys():
            if not col:
                continue
            if self._should_skip_segment_col(col):
                continue

            values = [
                str(row.get(col, "")).strip() for row in rows
                if row.get(col) is not None and str(row.get(col)).strip()
            ]
            if len(values) < total_rows * 0.5:
                continue

            counter = Counter(values)
            cardinality = len(counter)

            if cardinality < 5 or cardinality > 100:
                continue

            top_val = counter.most_common(1)[0][0]
            if self._is_numeric_string(top_val):
                continue

            top_count = counter.most_common(1)[0][1]
            if top_count / len(values) < 0.2:
                continue

            top5 = counter.most_common(5)
            dist_str = ", ".join(f"{v}({c})" for v, c in top5)
            remaining = cardinality - 5
            if remaining > 0:
                dist_str += f", +{remaining} others"

            insight = self._create_insight(
                title=f"Distribution: {col} ({cardinality} categories)",
                description=(
                    f"{col} has {cardinality} unique values across {len(values):,} records. "
                    f"Top: {dist_str}"
                ),
                insight_type=InsightType.STATUS_DISTRIBUTION,
                severity=InsightSeverity.INFO,
                source_table=table_name,
                source_column=col,
                source_system=system,
                metric_value=cardinality,
                metric_name=f"distribution_{col}",
                count=len(values),
                percentage=top_count / len(values) * 100,
                recommendations=[
                    f"Consider {col} as a segmentation dimension",
                    f"Top category '{top5[0][0]}' represents {top_count/len(values):.0%} of data",
                ],
                business_impact=f"{col} distribution reveals {cardinality} distinct categories",
            )
            self.insights.append(insight)

    def _consolidate_and_validate_insights(self):
        """v27.0: 인사이트 중복 제거 및 모순 감지."""
        if not self.insights:
            return

        # 1. ML Anomaly 인사이트 상위 5개로 제한
        ml_anomalies = [i for i in self.insights if i.title.startswith("ML Anomalies Detected:")]
        if len(ml_anomalies) > 5:
            ml_anomalies.sort(key=lambda x: -(x.percentage or 0))
            keep = set(id(i) for i in ml_anomalies[:5])
            dropped = len(ml_anomalies) - 5
            self.insights = [
                i for i in self.insights
                if not i.title.startswith("ML Anomalies Detected:") or id(i) in keep
            ]
            logger.info(f"[v27.10] ML Anomaly 인사이트 {dropped}개 제거 (상위 5개만 유지)")

        # 2. 같은 컬럼에 대한 positive/negative 방향 모순 감지
        column_directions = defaultdict(list)
        for insight in self.insights:
            col = insight.source_column
            if not col:
                continue
            if "Segment Gap:" in insight.title or "Segment Performance:" in insight.title:
                column_directions[col].append({
                    "insight_id": insight.insight_id,
                    "direction": "positive",
                    "title": insight.title,
                })
            elif "Negative Correlation:" in insight.title or "Negative Driver:" in insight.title:
                column_directions[col].append({
                    "insight_id": insight.insight_id,
                    "direction": "negative",
                    "title": insight.title,
                })

        # v27.10: O(n²) → O(n) dict lookup으로 contradiction 감지
        insight_by_id = {i.insight_id: i for i in self.insights}
        for col, entries in column_directions.items():
            directions = set(e["direction"] for e in entries)
            if len(directions) > 1:
                contradiction_note = (
                    f" [CONTRADICTION] {col} shows conflicting signals: "
                    + "; ".join(e["title"] for e in entries)
                )
                for entry in entries:
                    insight = insight_by_id.get(entry["insight_id"])
                    if insight:
                        insight.description += contradiction_note
                        if "Validate with controlled experiments" not in (insight.recommendations or []):
                            insight.recommendations.append(
                                "Validate with controlled experiments — conflicting signals detected"
                            )

    def _analyze_numeric_buckets(
        self,
        table_name: str,
        rows: List[Dict],
        bucket_col: str,
        kpi_cols: List[str],
        system: str,
    ):
        """
        연속형 숫자 컬럼을 분위수 기반 버킷으로 나누어 KPI 비교.
        도메인 무관하게 5-quantile(Q1~Q5) 사용.
        """
        import numpy as np

        raw_vals = [
            row.get(bucket_col) for row in rows
            if row.get(bucket_col) is not None and isinstance(row.get(bucket_col), (int, float))
        ]
        if len(raw_vals) < 50:
            return

        arr = np.array(raw_vals, dtype=float)
        quantiles = [0, 20, 40, 60, 80, 100]
        edges = [float(np.percentile(arr, q)) for q in quantiles]

        # 버킷 생성 (중복 edge 제거)
        buckets = {}
        for i in range(len(edges) - 1):
            lo, hi = edges[i], edges[i + 1]
            if lo == hi:
                continue
            bname = f"{lo:.1f}-{hi:.1f}"
            buckets[bname] = (lo, hi)

        if len(buckets) < 3:
            return

        # v27.0: 모든 KPI를 한번에 분석하여 통합 인사이트 생성 (카테시안 곱 제거)
        # v27.4: bucket_col 자기 자신을 KPI로 측정하는 순환 분석 제거
        # v27.5: 동일 KPI 내 모든 버킷에 같은 집계 방법 사용 (mean vs median 혼합 비교 방지)
        significant_kpis = []
        for kpi_col in kpi_cols:
            if kpi_col == bucket_col:
                continue
            # 먼저 KPI 전체의 CV를 계산하여 집계 방법을 결정
            all_kpi_vals = [
                row[kpi_col] for row in rows
                if row.get(kpi_col) is not None and isinstance(row.get(kpi_col), (int, float))
            ]
            if len(all_kpi_vals) < 50:
                continue
            kpi_mean = statistics.mean(all_kpi_vals)
            if kpi_mean == 0:
                continue
            kpi_cv = abs(statistics.stdev(all_kpi_vals) / kpi_mean) if len(all_kpi_vals) > 1 else 0
            use_median = kpi_cv > 10  # KPI 전체 기준으로 한 번만 판단

            bucket_stats = {}
            for bname, (lo, hi) in buckets.items():
                vals = [
                    row[kpi_col] for row in rows
                    if row.get(bucket_col) is not None
                    and isinstance(row.get(bucket_col), (int, float))
                    and lo <= row[bucket_col] < (hi + 0.001 if bname == list(buckets.keys())[-1] else hi)
                    and row.get(kpi_col) is not None
                    and isinstance(row.get(kpi_col), (int, float))
                ]
                if len(vals) >= 10:
                    agg = statistics.median(vals) if use_median else statistics.mean(vals)
                    bucket_stats[bname] = {"n": len(vals), "mean": agg}

            if len(bucket_stats) < 3:
                continue

            means = [s["mean"] for s in bucket_stats.values()]
            max_mean = max(means)
            min_mean = min(means)
            if min_mean <= 0 or max_mean / min_mean < 2.0:
                continue

            best_bucket = max(bucket_stats, key=lambda b: bucket_stats[b]["mean"])
            ratio = max_mean / min_mean
            significant_kpis.append({
                "kpi": kpi_col,
                "ratio": ratio,
                "best_bucket": best_bucket,
                "best_mean": bucket_stats[best_bucket]["mean"],
                "stats": bucket_stats,
            })

        if not significant_kpis:
            return

        # 상위 3개 KPI만 하나의 인사이트로 통합
        significant_kpis.sort(key=lambda x: x["ratio"], reverse=True)
        top_kpis = significant_kpis[:3]

        kpi_details = "; ".join(
            f"{k['kpi']}({k['ratio']:.1f}x, best={k['best_bucket']})"
            for k in top_kpis
        )
        best = top_kpis[0]
        bucket_detail = "; ".join(
            f"{b}(n={s['n']}, avg={s['mean']:,.1f})"
            for b, s in best["stats"].items()
        )

        insight = self._create_insight(
            title=f"{bucket_col} Sweet Spot ({len(top_kpis)} KPIs)",
            description=(
                f"{bucket_col} range {best['best_bucket']} is the sweet spot. "
                f"Top effects: [{kpi_details}]. "
                f"Detail for {best['kpi']}: [{bucket_detail}]"
            ),
            insight_type=InsightType.KPI,
            severity=InsightSeverity.MEDIUM,
            source_table=table_name,
            source_column=bucket_col,
            source_system=system,
            metric_value=best["best_mean"],
            metric_name=f"{bucket_col}→{best['kpi']}",
            count=sum(s["n"] for s in best["stats"].values()),
            recommendations=[
                f"Target {bucket_col} range {best['best_bucket']} for maximum impact",
                f"Affects {len(top_kpis)} KPIs with up to {best['ratio']:.1f}x variation",
            ],
            business_impact=f"{bucket_col} optimization affects {len(significant_kpis)} KPIs, up to {best['ratio']:.1f}x improvement",
        )
        self.insights.append(insight)

    # === v26.1: 상관관계 기반 인사이트 (음의 상관 포함) ===

    def _analyze_correlation_insights(
        self,
        table_name: str,
        rows: List[Dict],
        table_info: Dict,
        system: str,
    ):
        """
        v26.1: 숫자 컬럼 간 Pearson 상관관계 분석

        - 강한 양의 상관 (r > 0.3): 드라이버 관계 보고
        - 주목할 음의 상관 (r < -0.1): 역설/트레이드오프 보고
        - 파생 컬럼(r ≈ 1.0) 제외
        """
        if not rows or len(rows) < 30:
            return

        import numpy as np

        sample_row = rows[0]

        # v28.0: pandas 기반 numeric 컬럼 감지 + corr matrix
        import pandas as _pd
        import numpy as _np
        df = getattr(self, '_current_df', None)

        # 숫자 컬럼 수집
        numeric_cols = []
        if df is not None and not df.empty:
            # v28.0 fast path: pandas dtype 기반 O(cols) 감지 (rows 재스캔 없음)
            col_info_map = {c["name"]: c for c in table_info.get("columns", [])}
            total_rows_local = len(rows)
            for col in sample_row.keys():
                if not col or self._should_skip_segment_col(col) or not self._is_business_metric_column(col):
                    continue
                if col not in df.columns:
                    continue
                if _pd.api.types.is_numeric_dtype(df[col]):
                    non_null = df[col].count()
                    if non_null >= total_rows_local * 0.5:
                        numeric_cols.append(col)
        else:
            # fallback: 기존 rows 스캔 방식
            for col in sample_row.keys():
                if not col or self._should_skip_segment_col(col) or not self._is_business_metric_column(col):
                    continue
                vals = [row.get(col) for row in rows if row.get(col) is not None and isinstance(row.get(col), (int, float))]
                if len(vals) >= len(rows) * 0.5:
                    numeric_cols.append(col)

        if len(numeric_cols) < 2:
            return

        # KPI 타겟 컬럼 — CV 상위 + _score/_rate 패턴
        col_cv = []
        if df is not None and not df.empty and numeric_cols:
            # v28.0: pandas vectorized CV 계산
            numeric_df = df[numeric_cols].apply(_pd.to_numeric, errors='coerce')
            means = numeric_df.mean()
            stds = numeric_df.std()
            for col in numeric_cols:
                m = means.get(col, 0)
                s = stds.get(col, 0)
                if m != 0:
                    col_cv.append((col, s / abs(m)))
        else:
            for col in numeric_cols:
                vals = [row.get(col) for row in rows if row.get(col) is not None and isinstance(row.get(col), (int, float))]
                if vals and len(vals) > 1:
                    m = statistics.mean(vals)
                    if m != 0:
                        col_cv.append((col, statistics.stdev(vals) / abs(m)))
        col_cv.sort(key=lambda x: -x[1])
        cv_targets = [c for c, _ in col_cv[:5]] if col_cv else numeric_cols[:3]
        # _score/_rate/_index 패턴 컬럼도 타겟에 추가 (합성 KPI는 모든 도메인에서 중요)
        kpi_pattern_targets = [
            c for c in numeric_cols
            if c not in cv_targets
            and any(p in c.lower() for p in ("_score", "_rate", "_index", "score_", "rate_"))
        ]
        target_cols = cv_targets + kpi_pattern_targets[:3]

        # v28.0: df.corr() — 전체 correlation matrix를 C-level 단일 연산으로 계산
        # 기존: target × col 쌍마다 rows를 재순회 O(targets × cols × rows)
        # 변경: df[numeric_cols].corr() → O(cols²) C-level 연산 → 100x 이상 빠름
        reported = set()

        # Correlation matrix pre-computation (df 있을 때)
        _corr_matrix = None
        if df is not None and not df.empty and numeric_cols:
            try:
                numeric_df_for_corr = df[[c for c in numeric_cols if c in df.columns]].apply(_pd.to_numeric, errors='coerce')
                _corr_matrix = numeric_df_for_corr.corr(method='pearson')
            except Exception:
                _corr_matrix = None

        for target in target_cols:
            correlations = []

            if _corr_matrix is not None and target in _corr_matrix.columns:
                # v28.0 fast path: corr matrix에서 O(1) 조회
                for col in numeric_cols:
                    if col == target or col not in _corr_matrix.columns:
                        continue
                    r = _corr_matrix.loc[target, col]
                    if np.isnan(r) or abs(r) > 0.95:
                        continue
                    # 유효 샘플 수 확인 (30 이상)
                    valid_n = df[[target, col]].dropna().shape[0] if df is not None else len(rows)
                    if valid_n < 30:
                        continue
                    correlations.append((col, float(r)))
            else:
                # fallback: 기존 row-by-row 방식
                target_vals = np.array([row.get(target, np.nan) for row in rows], dtype=float)
                target_valid = ~np.isnan(target_vals)
                for col in numeric_cols:
                    if col == target:
                        continue
                    col_vals = np.array([row.get(col, np.nan) for row in rows], dtype=float)
                    valid = target_valid & ~np.isnan(col_vals)
                    if valid.sum() < 30:
                        continue
                    try:
                        r = float(np.corrcoef(target_vals[valid], col_vals[valid])[0, 1])
                    except Exception:
                        continue
                    if np.isnan(r) or abs(r) > 0.95:
                        continue
                    correlations.append((col, r))

            if not correlations:
                continue

            # 상관 순으로 정렬
            correlations.sort(key=lambda x: -abs(x[1]))

            # 양의 상관 드라이버 (top 3, r > 0.15)
            positive = [(c, r) for c, r in correlations if r > 0.15][:3]
            # v27.5: 음의 상관 임계값 강화 (r < -0.1)
            # 이전: r < -0.08 → r≈-0.08 수준의 사실상 무상관도 보고
            # 이제: |r| >= 0.1 이상만 보고 (통계적 의미 있는 약한 상관)
            negative = [(c, r) for c, r in correlations if r < -0.1]

            # 양의 드라이버 보고
            if positive:
                pair_key = f"pos_{target}"
                if pair_key not in reported:
                    reported.add(pair_key)
                    driver_details = ", ".join(f"{c}({r:.3f})" for c, r in positive)
                    insight = self._create_insight(
                        title=f"Top Drivers of {target}",
                        description=f"Strongest positive correlations with {target}: {driver_details}",
                        insight_type=InsightType.KPI,
                        severity=InsightSeverity.MEDIUM,
                        source_table=table_name,
                        source_column=target,
                        source_system=system,
                        metric_value=positive[0][1],
                        metric_name=f"correlation_{target}",
                        recommendations=[
                            f"Focus on {positive[0][0]} to improve {target}",
                            "Validate correlation with A/B testing before causal claims",
                        ],
                    )
                    self.insights.append(insight)

            # 음의 상관 (paradox / tradeoff) 보고
            for col, r in negative:
                pair_key = f"neg_{target}_{col}"
                if pair_key in reported:
                    continue
                reported.add(pair_key)

                insight = self._create_insight(
                    title=f"Negative Correlation: {col} vs {target}",
                    description=(
                        f"{col} shows negative correlation with {target} (r={r:.3f}). "
                        f"Increasing {col} is associated with LOWER {target}."
                    ),
                    insight_type=InsightType.KPI,
                    severity=InsightSeverity.MEDIUM,
                    source_table=table_name,
                    source_column=col,
                    source_system=system,
                    metric_value=r,
                    metric_name=f"neg_corr_{col}_vs_{target}",
                    recommendations=[
                        f"Avoid over-optimizing {col} — diminishing/negative returns on {target}",
                        f"Investigate confounding variables between {col} and {target}",
                    ],
                    business_impact=f"Counter-intuitive: more {col} → lower {target}",
                )
                self.insights.append(insight)

    # === v26.2: ID 컬럼 중복 패턴 분석 ===

    def _analyze_duplicate_patterns(
        self,
        table_name: str,
        rows: List[Dict],
        table_info: Dict,
        system: str,
    ):
        """
        v26.2: ID성 컬럼의 중복 비율 분석.
        동일 ID에 여러 행이 존재하면 longitudinal/snapshot 패턴 가능성 보고.
        """
        if not rows or len(rows) < 20:
            return

        total_rows = len(rows)
        sample_row = rows[0]

        # ID성 컬럼 감지: "_id" 접미사 또는 cardinality 비율이 50-95% (고유하지만 중복 있음)
        id_patterns = ("_id", "_key", "_code")

        for col in sample_row.keys():
            if not col:
                continue
            col_lower = col.lower()

            # ID성 컬럼 판별
            is_id_col = any(col_lower.endswith(p) for p in id_patterns)
            if not is_id_col:
                continue

            # 값 수집
            values = [str(row.get(col, "")) for row in rows if row.get(col) is not None and str(row.get(col)).strip()]
            if len(values) < total_rows * 0.8:
                continue

            unique_count = len(set(values))
            uniqueness_ratio = unique_count / len(values)

            # 중복 패턴: 고유값이 50-95% (일부 중복 있음)
            if 0.50 <= uniqueness_ratio <= 0.95:
                dup_count = len(values) - unique_count
                # 중복 빈도 계산
                from collections import Counter
                freq = Counter(values)
                dup_ids = {k: v for k, v in freq.items() if v > 1}
                max_freq = max(freq.values())
                avg_freq = len(values) / unique_count

                insight = self._create_insight(
                    title=f"Duplicate Pattern: {col}",
                    description=(
                        f"{unique_count:,} unique {col} out of {len(values):,} rows "
                        f"({uniqueness_ratio:.1%} unique). "
                        f"{len(dup_ids):,} IDs have duplicates (max {max_freq}x, avg {avg_freq:.1f}x). "
                        f"Possible longitudinal/time-series analysis pattern."
                    ),
                    insight_type=InsightType.DATA_QUALITY,
                    severity=InsightSeverity.MEDIUM,
                    source_table=table_name,
                    source_column=col,
                    source_system=system,
                    metric_value=float(unique_count),
                    recommendations=[
                        f"Verify if multiple rows per {col} represent time-series snapshots",
                        f"Consider deduplication (use latest record) for cross-sectional analysis",
                        f"Explore temporal patterns across duplicate {col} entries",
                    ],
                    business_impact=f"{len(dup_ids):,} {col}s have repeated entries — may indicate re-analysis or versioning",
                )
                self.insights.append(insight)

    # === v14.0: LLM 기반 도메인 특화 분석 (하드코딩 제거) ===

    def _extract_domain_info(self):
        """
        v14.0: DomainContext에서 도메인 정보 추출

        LLM이 탐지한 도메인 정보를 저장하여 이후 LLM 프롬프트에서 활용
        """
        if not self.domain_context:
            self._detected_domain = "general"
            return

        # DomainContext에서 industry 추출
        if hasattr(self.domain_context, 'industry'):
            self._detected_domain = self.domain_context.industry
        elif isinstance(self.domain_context, dict):
            self._detected_domain = self.domain_context.get('industry', 'general')
        else:
            self._detected_domain = str(self.domain_context)

        self._detected_domain = self._detected_domain.lower().replace('-', '_').replace(' ', '_')
        logger.info(f"[v14.0] Domain for LLM insight analysis: {self._detected_domain}")

    def _generate_llm_domain_insights(
        self,
        tables: Dict[str, Dict[str, Any]],
        data: Dict[str, List[Dict[str, Any]]],
    ):
        """
        v18.3: LLM 기반 도메인 특화 인사이트 생성

        데이터 요약 + 파이프라인 분석 모델 결과(인과분석, 시뮬레이션, 이상탐지 등)를
        종합하여 LLM이 고수준 비즈니스 인사이트를 도출합니다.
        """
        if not self.llm_client:
            logger.debug("[v14.0] LLM client not available, skipping domain-specific insights")
            return

        try:
            from ..utils import parse_llm_json
        except ImportError:
            logger.warning("[v14.0] parse_llm_json not available")
            return

        # 데이터 요약 생성
        data_summary = self._create_data_summary_for_llm(tables, data)

        # v18.3: 파이프라인 분석 모델 결과 요약 생성
        analysis_summary = self._create_analysis_results_summary()

        # v18.3: 자체 ML 분석에서 이미 발견한 이상탐지 결과 요약
        ml_findings = self._summarize_current_insights()

        prompt = f"""You are a senior business intelligence analyst at a Palantir-class analytics firm, specializing in **{self._detected_domain}**.

You have access to THREE sources of evidence. Synthesize ALL of them to produce executive-level insights.

## Source 1: Raw Dataset Summary
{data_summary}

## Source 2: Predictive & Causal Model Results (from pipeline analysis modules)
{analysis_summary if analysis_summary else "(No model results available — rely on data summary only)"}

## Source 3: ML Anomaly Detection Findings (already computed)
{ml_findings if ml_findings else "(No ML findings yet)"}

## Your Task

Produce insights that SYNTHESIZE the model results with the data. You MUST reference specific model outputs when available. Examples of model-grounded insights:

- "Causal impact analysis confirms data integration yields 20.3% ATE improvement (p<0.001, robust). Combined with the 35% high-turnover-risk workforce, prioritizing integrated HR analytics could reduce attrition cost by $467K annually."
- "Counterfactual simulation: improving data quality from 75%→90% for Employee entity is projected to [impact]. Given salary anomalies in 10% of records, this quality gap directly affects compensation analytics reliability."
- "ML ensemble detected 10% salary anomalies (IsolationForest + LOF). Cross-referencing with the causal graph edge salary→turnover_risk suggests pay inequity is a statistically validated driver of attrition."

DO NOT simply restate the raw data statistics. Every insight MUST either:
1. Reference a specific model result (ATE, p-value, simulation scenario, anomaly rate) AND connect it to business impact, OR
2. Synthesize across multiple analysis results to surface a non-obvious pattern

## Output Format (JSON) — v19.0 Palantir-Style Prescriptive Insights

Return a JSON array where each insight follows hypothesis → evidence → validation → prescriptive actions:

```json
[
  {{
    "title": "[{self._detected_domain}] Concise executive-level title",
    "description": "2-3 sentence executive summary grounded in model results with specific numbers",

    "hypothesis": "Testable business claim (e.g., 'Junior employees with high tenure are underpaid, driving turnover')",

    "evidence": [
      {{
        "model": "CausalImpact|AnomalyDetection|CorrelationAnalysis|TimeSeries|Simulation",
        "finding": "Specific result with numbers (ATE=-4827, p<0.001, anomaly_rate=10%)",
        "metrics": {{"key": "value"}}
      }}
    ],

    "validation_summary": "How evidence confirms hypothesis (synthesis, NOT restatement)",

    "prescriptive_actions": [
      {{
        "action": "Specific actionable step",
        "expected_impact": "Quantified outcome (e.g., 15% reduction in turnover)",
        "estimated_savings": "Dollar or KPI improvement",
        "priority": "immediate|short_term|long_term"
      }}
    ],

    "insight_type": "kpi|risk|anomaly|operational|prediction",
    "severity": "critical|high|medium|low",
    "source_table": "table name",
    "source_column": "key columns",
    "metric_value": null,
    "recommendations": ["Action 1", "Action 2"],
    "business_impact": "Quantified overall business impact",
    "confidence_breakdown": {{"algorithm": 0.XX, "simulation": 0.XX, "llm": 0.XX}},
    "evidence_models": ["CausalImpact", "AnomalyDetection"]
  }}
]
```

CRITICAL: Every insight MUST have hypothesis, evidence[] (min 2 models), validation_summary, prescriptive_actions[] with quantified impact.
DO NOT create shallow insights like "X% anomalies detected" — synthesize across multiple analysis sources.

## v26.0 ANTI-HALLUCINATION RULES
1. NEVER invent specific numbers, counts, or statistics not present in the data summary above
2. NEVER generate dollar amounts ($) unless cost/revenue data explicitly appears in the dataset
3. When referencing categorical groups (e.g., "unknown tier"), report ONLY the statistics shown in the data summary — do NOT extrapolate or assume hidden values
4. If the data summary says avg=0.0 for a group, that group has ZERO average — do NOT claim otherwise
5. Distinguish CORRELATION from CAUSATION — always note "correlation observed" not "X causes Y" unless causal model confirms it
6. For rate/ratio columns, verify the scale (0-1 vs 0-100 vs unbounded) before setting thresholds

Generate 4-6 prescriptive insights. Prioritize by business impact.
Respond ONLY with valid JSON array."""

        try:
            # v18.2: Letsur AI Gateway 모델 사용 (하드코딩 제거)
            from src.unified_pipeline.model_config import get_model, ModelType, get_model_spec
            llm_model = get_model(ModelType.BALANCED)
            response = self.llm_client.chat.completions.create(
                model=llm_model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=get_model_spec(llm_model).get("output_max", 32_000),
                temperature=0.3,
            )

            result_text = response.choices[0].message.content
            llm_insights = parse_llm_json(result_text, default=[])

            if not isinstance(llm_insights, list):
                llm_insights = [llm_insights] if llm_insights else []

            # LLM 인사이트를 BusinessInsight 객체로 변환
            for llm_insight in llm_insights:
                if not isinstance(llm_insight, dict):
                    continue

                insight_type_str = llm_insight.get("insight_type", "operational")
                insight_type_map = {
                    "kpi": InsightType.KPI,
                    "risk": InsightType.RISK,
                    "anomaly": InsightType.ANOMALY,
                    "operational": InsightType.OPERATIONAL,
                    "data_quality": InsightType.DATA_QUALITY,
                }
                insight_type = insight_type_map.get(insight_type_str, InsightType.OPERATIONAL)

                severity_str = llm_insight.get("severity", "medium")
                severity_map = {
                    "critical": InsightSeverity.CRITICAL,
                    "high": InsightSeverity.HIGH,
                    "medium": InsightSeverity.MEDIUM,
                    "low": InsightSeverity.LOW,
                    "info": InsightSeverity.INFO,
                }
                severity = severity_map.get(severity_str, InsightSeverity.MEDIUM)

                insight = self._create_insight(
                    title=f"[{self._detected_domain}] {llm_insight.get('title', 'Domain Insight')}",
                    description=llm_insight.get("description", ""),
                    insight_type=insight_type,
                    severity=severity,
                    source_table=llm_insight.get("source_table", "unknown"),
                    source_column=llm_insight.get("source_column", ""),
                    source_system=self._detected_domain,
                    metric_value=llm_insight.get("metric_value"),
                    recommendations=llm_insight.get("recommendations", []),
                    business_impact=llm_insight.get("business_impact", ""),
                )
                self.insights.append(insight)

            logger.info(f"[v14.0] Generated {len(llm_insights)} LLM-based domain insights for {self._detected_domain}")

        except Exception as e:
            logger.warning(f"[v18.3] LLM domain insight generation failed: {e}")

    def _create_data_summary_for_llm(
        self,
        tables: Dict[str, Dict[str, Any]],
        data: Dict[str, List[Dict[str, Any]]],
    ) -> str:
        """
        v14.0: LLM 프롬프트용 데이터 요약 생성
        """
        summary_parts = []

        for table_name, rows in data.items():
            if not rows:
                continue

            table_info = tables.get(table_name, {})
            sample_row = rows[0]
            columns = list(sample_row.keys())

            # 숫자형 컬럼 통계 — v18.3: 전체 데이터 사용
            numeric_stats = []
            for col in columns:
                numeric_values = []
                for row in rows:
                    val = row.get(col)
                    if val is not None:
                        try:
                            numeric_values.append(float(val))
                        except (ValueError, TypeError):
                            pass

                if len(numeric_values) >= 5:
                    avg = sum(numeric_values) / len(numeric_values)
                    min_val = min(numeric_values)
                    max_val = max(numeric_values)
                    numeric_stats.append(f"  - {col}: avg={avg:.2f}, min={min_val:.2f}, max={max_val:.2f} (n={len(numeric_values)})")

            # 카테고리형 컬럼 분포 — v18.3: 전체 데이터 사용
            categorical_stats = []
            for col in columns:
                values = [str(row.get(col, "")) for row in rows if row.get(col)]
                if values and len(set(values)) <= 20:  # 유니크 값이 적으면 카테고리
                    value_counts = Counter(values)
                    top_3 = value_counts.most_common(3)
                    dist = ", ".join([f"'{v}': {c}" for v, c in top_3])
                    categorical_stats.append(f"  - {col}: {dist}")

            # v18.4: 카테고리 x 숫자 그룹별 교차 분석
            cross_analysis = []
            from collections import defaultdict

            # 카테고리 컬럼 식별: 문자열 이진(2~5값)을 우선, 숫자형 소수값은 후순위
            cat_cols_priority = []  # (priority, col) — 낮은 값이 높은 우선순위
            for col in columns:
                str_values = [str(row.get(col, "")) for row in rows if row.get(col) is not None]
                if not str_values:
                    continue
                unique = set(str_values)
                n_unique = len(unique)
                if n_unique < 2 or n_unique > 15:
                    continue
                # 숫자형인지 확인
                is_numeric = True
                for u in list(unique)[:5]:
                    try:
                        float(u)
                    except (ValueError, TypeError):
                        is_numeric = False
                        break
                # id 컬럼 제외
                if any(p in col.lower() for p in ["id", "index", "번호"]):
                    continue
                # 문자열 이진 (Yes/No 등) → 최고 우선순위
                if not is_numeric and n_unique == 2:
                    cat_cols_priority.append((0, col))
                elif not is_numeric and n_unique <= 5:
                    cat_cols_priority.append((1, col))
                elif not is_numeric:
                    cat_cols_priority.append((2, col))
                elif n_unique <= 5:
                    cat_cols_priority.append((3, col))

            cat_cols_priority.sort(key=lambda x: x[0])
            cat_cols_for_cross = [c for _, c in cat_cols_priority[:6]]

            # 숫자 컬럼 식별 (id 제외)
            num_cols_for_cross = []
            for col in columns:
                if any(p in col.lower() for p in ["id", "index", "번호"]):
                    continue
                n_vals = []
                for row in rows:
                    v = row.get(col)
                    if v is not None:
                        try:
                            n_vals.append(float(v))
                        except (ValueError, TypeError):
                            pass
                if len(n_vals) >= 5 and len(set(n_vals)) > 5:
                    num_cols_for_cross.append(col)

            for cat_col in cat_cols_for_cross:
                for num_col in num_cols_for_cross[:6]:
                    if cat_col == num_col:
                        continue
                    groups = defaultdict(list)
                    for row in rows:
                        cat_val = row.get(cat_col)
                        num_val = row.get(num_col)
                        if cat_val is not None and num_val is not None:
                            try:
                                groups[str(cat_val)].append(float(num_val))
                            except (ValueError, TypeError):
                                pass

                    if len(groups) >= 2:
                        group_avgs = {}
                        for k, v in groups.items():
                            if len(v) >= 3:
                                group_avgs[k] = sum(v) / len(v)
                        if len(group_avgs) >= 2:
                            parts = ", ".join(f"{k}: avg={v:.1f} (n={len(groups[k])})" for k, v in sorted(group_avgs.items()))
                            cross_analysis.append(f"  - {num_col} by {cat_col}: {parts}")

            summary_parts.append(f"""
### Table: {table_name}
- Rows: {len(rows)}
- Columns: {', '.join(columns[:10])}{'...' if len(columns) > 10 else ''}

Numeric columns:
{chr(10).join(numeric_stats[:5]) if numeric_stats else '  (none)'}

Categorical columns:
{chr(10).join(categorical_stats[:5]) if categorical_stats else '  (none)'}

Group-level analysis (numeric by category):
{chr(10).join(cross_analysis[:10]) if cross_analysis else '  (none)'}
""")

        # v26.0: 비용/매출 데이터 존재 여부 명시 (LLM 환각 방지)
        has_cost_data = False
        cost_keywords = ["cost", "salary", "price", "revenue", "budget", "expense",
                         "income", "profit", "fee", "amount", "payment", "wage"]
        for table_name_check, tinfo in tables.items():
            cols = tinfo.get("columns", [])
            for col in cols:
                cname = (col.get("name", str(col)) if isinstance(col, dict) else str(col)).lower()
                if any(kw in cname for kw in cost_keywords):
                    has_cost_data = True
                    break

        if not has_cost_data:
            summary_parts.append("""
### ⚠️ DATA CONSTRAINTS
- **NO cost/revenue/salary/price data** exists in this dataset
- You MUST NOT generate monetary estimates ($, savings, cost reduction, revenue impact)
- Use only relative metrics (%, x-fold, ratio) for impact quantification
""")

        return "\n".join(summary_parts[:5])  # 최대 5개 테이블

    def _create_analysis_results_summary(self) -> str:
        """
        v18.3: 파이프라인 분석 모델 결과를 LLM 프롬프트용 텍스트로 요약

        Phase 2(Refinement)에서 실행된 CausalImpact, Granger, Counterfactual,
        VirtualEntity 등의 결과를 구조화된 텍스트로 변환합니다.
        """
        ctx = self._pipeline_context
        if not ctx:
            return ""

        parts = []

        # 1. Causal Impact Analysis (인과 영향 분석)
        causal = ctx.get("causal_insights", {})
        if isinstance(causal, dict) and causal:
            impact = causal.get("impact_analysis", {})
            if isinstance(impact, dict) and impact:
                summary = impact.get("summary", {})

                # v18.5: data_driven 결과 (실제 데이터 기반) vs 기존 합성 데이터 결과
                if summary.get("data_driven"):
                    # 실제 데이터 기반 인과 분석 결과
                    effects = impact.get("treatment_effects", [])
                    recs = impact.get("recommendations", [])
                    group_comps = impact.get("group_comparisons", [])

                    effect_lines = []
                    for te in effects[:8]:
                        strat = te.get("stratified_analysis")
                        gc = te.get("group_comparison", {})
                        g1 = gc.get("group_1", {})
                        g0 = gc.get("group_0", {})

                        if strat:
                            conf = strat.get("confounder", "?")
                            effect_lines.append(
                                f"  - {te.get('type', '?')}: "
                                f"Naive ATE={te.get('naive_ate', 0):+.1f} "
                                f"({g1.get('label','?')}={g1.get('mean',0):.1f} vs {g0.get('label','?')}={g0.get('mean',0):.1f}), "
                                f"Adjusted ATE={te.get('estimate', 0):+.1f} "
                                f"(confounding by {conf} removed {strat.get('confounding_reduction_pct', 0):.0f}%), "
                                f"p={te.get('p_value', 1):.4f}, sig={'YES' if te.get('significant') else 'no'}"
                            )
                            # 층별 세부 정보
                            for s in strat.get("strata", [])[:4]:
                                effect_lines.append(
                                    f"    └ {s['stratum']}: "
                                    f"treated={s['treated_mean']:.1f}(n={s['treated_n']}), "
                                    f"control={s['control_mean']:.1f}(n={s['control_n']}), "
                                    f"stratum ATE={s['stratum_ate']:+.1f}"
                                )
                        else:
                            effect_lines.append(
                                f"  - {te.get('type', '?')}: "
                                f"ATE={te.get('estimate', 0):+.1f}, "
                                f"p={te.get('p_value', 1):.4f}, "
                                f"sig={'YES' if te.get('significant') else 'no'}"
                            )

                    parts.append(f"""### CausalImpactAnalyzer (실제 데이터 기반)
- Data-driven: True
- Total effects analyzed: {summary.get('total_effects_analyzed', 0)}
- Statistically significant: {summary.get('significant_effects', 0)}
- Sample size: {summary.get('sample_size', 0)}

IMPORTANT - Treatment Effects (교란변수 보정 포함):
{chr(10).join(effect_lines) if effect_lines else '  (none)'}

Recommendations:
{chr(10).join('  - ' + r for r in recs[:5]) if recs else '  (none)'}

CRITICAL NOTE: When interpreting causal effects, ALWAYS use the Adjusted ATE (not Naive ATE).
Naive ATE includes confounding bias. Adjusted ATE controls for confounding variables.
If Naive ATE and Adjusted ATE differ significantly, the confounding variable is important.""")

                else:
                    # 기존 합성 데이터 결과 (fallback)
                    ate = summary.get("estimated_ate")
                    ci = summary.get("ate_confidence_interval")
                    sig = summary.get("ate_significant")
                    robust = summary.get("robustness")
                    sample = summary.get("sample_size")
                    effects = impact.get("treatment_effects", [])
                    recs = impact.get("recommendations", [])
                    parts.append(f"""### CausalImpactAnalyzer (metadata-based estimate)
- Average Treatment Effect (ATE): {ate}
- Confidence Interval: {ci}
- Statistically Significant: {sig}
- Robustness Score: {robust}
- Sample Size: {sample}
- Treatment Effects: {len(effects)} detected
- Recommendations: {'; '.join(recs[:3]) if recs else 'N/A'}
NOTE: This is a metadata-based estimate, not based on actual data values.""")

            # Granger Causality
            granger = causal.get("granger_causality", [])
            if isinstance(granger, list) and granger:
                pairs = []
                for g in granger[:5]:
                    if isinstance(g, dict):
                        pairs.append(f"{g.get('cause', '?')} → {g.get('effect', '?')} (p={g.get('p_value', '?')})")
                if pairs:
                    parts.append(f"""### Granger Causality (시계열 인과관계)
- {len(granger)} causal links detected
- Key links: {', '.join(pairs)}""")

            # Causal Graph
            nodes = causal.get("causal_graph_nodes", [])
            edges = causal.get("causal_graph_edges", [])
            if nodes or edges:
                edge_strs = []
                for e in edges[:6]:
                    if isinstance(e, dict):
                        edge_strs.append(f"{e.get('source', '?')} → {e.get('target', '?')}")
                parts.append(f"""### Causal Graph (인과 DAG)
- {len(nodes)} nodes, {len(edges)} edges
- Key edges: {', '.join(edge_strs)}""")

        # 2. Counterfactual Analysis (반사실 시뮬레이션)
        counterfactual = ctx.get("counterfactual_analysis", [])
        if isinstance(counterfactual, list) and counterfactual:
            cf_lines = []
            for cf in counterfactual[:3]:
                if isinstance(cf, dict):
                    scenario = cf.get("scenario", "?")
                    rec = cf.get("recommendation", "")
                    cf_lines.append(f"  - Scenario: {scenario} → {rec}")
            if cf_lines:
                parts.append(f"""### Counterfactual Simulation (반사실 시뮬레이션)
- {len(counterfactual)} scenarios analyzed
{chr(10).join(cf_lines)}""")

        # 3. Virtual Entities (가상 시나리오 엔티티)
        virtual = ctx.get("virtual_entities", [])
        if isinstance(virtual, list) and virtual:
            ve_names = [v.get("name", "?") for v in virtual[:5] if isinstance(v, dict)]
            if ve_names:
                parts.append(f"""### Virtual Entity Simulation
- {len(virtual)} virtual entities generated: {', '.join(ve_names)}""")

        # 4. Causal Relationships
        causal_rels = ctx.get("causal_relationships", [])
        if isinstance(causal_rels, list) and causal_rels:
            rel_strs = []
            for r in causal_rels[:5]:
                if isinstance(r, dict):
                    rel_strs.append(f"{r.get('source', '?')} → {r.get('target', '?')} (type={r.get('relationship_type', '?')}, conf={r.get('confidence', '?')})")
            if rel_strs:
                parts.append(f"""### Causal Relationships
- {len(causal_rels)} relationships discovered
- {'; '.join(rel_strs)}""")

        # 5. Anomaly Results (if available from governance)
        anomaly = ctx.get("anomaly_results", {})
        if isinstance(anomaly, dict) and anomaly:
            parts.append(f"""### Anomaly Detection Results
- {json.dumps(anomaly, default=str)[:300]}""")

        return "\n\n".join(parts) if parts else ""

    def _summarize_current_insights(self) -> str:
        """
        v18.3: 현재까지 이 Analyzer가 발견한 ML 기반 인사이트를 요약

        LLM 프롬프트에서 이미 발견된 이상탐지/리스크 결과를 참조할 수 있도록 합니다.
        중복 인사이트를 방지하고, 모델 결과를 cross-reference 하도록 유도합니다.
        """
        if not self.insights:
            return ""

        lines = []
        for ins in self.insights[:15]:
            lines.append(f"- [{ins.severity.value}] {ins.title}: {ins.description[:120]}")

        return f"""Already detected {len(self.insights)} findings:
{chr(10).join(lines)}

IMPORTANT: Do NOT repeat these findings. Instead, SYNTHESIZE them with the model results above to produce higher-order insights."""
