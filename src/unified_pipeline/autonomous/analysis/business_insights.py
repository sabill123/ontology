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

import logging
import re
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Set, Tuple
from collections import Counter, defaultdict
from datetime import datetime
from enum import Enum
import statistics

logger = logging.getLogger(__name__)


class InsightType(str, Enum):
    """인사이트 유형"""
    STATUS_DISTRIBUTION = "status_distribution"
    ANOMALY = "anomaly"
    TREND = "trend"
    RISK = "risk"
    KPI = "kpi"
    DATA_QUALITY = "data_quality"
    OPERATIONAL = "operational"


class InsightSeverity(str, Enum):
    """인사이트 심각도"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


@dataclass
class BusinessInsight:
    """비즈니스 인사이트"""
    insight_id: str
    title: str
    description: str
    insight_type: InsightType
    severity: InsightSeverity

    # 소스
    source_table: str
    source_column: str
    source_system: str

    # 데이터
    metric_value: Any = None
    metric_name: str = ""
    count: int = 0
    percentage: float = 0.0

    # 컨텍스트
    benchmark: Optional[float] = None
    deviation: Optional[float] = None
    sample_values: List[Any] = field(default_factory=list)

    # 권장 조치
    recommendations: List[str] = field(default_factory=list)
    business_impact: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "insight_id": self.insight_id,
            "title": self.title,
            "description": self.description,
            "insight_type": self.insight_type.value,
            "severity": self.severity.value,
            "source_table": self.source_table,
            "source_column": self.source_column,
            "source_system": self.source_system,
            "metric_value": self.metric_value,
            "metric_name": self.metric_name,
            "count": self.count,
            "percentage": round(self.percentage, 2),
            "benchmark": self.benchmark,
            "deviation": round(self.deviation, 2) if self.deviation else None,
            "sample_values": self.sample_values[:10],
            "recommendations": self.recommendations,
            "business_impact": self.business_impact,
        }


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
                from .time_series_forecaster import create_forecaster
                self._forecaster = create_forecaster(domain_context)
            except ImportError:
                logger.debug("Time series forecaster not available")

        # ML 이상치 탐지기 초기화
        self._anomaly_detector = None
        if enable_ml_anomaly:
            try:
                from .anomaly_detection import AnomalyDetector
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
    ) -> List[BusinessInsight]:
        """
        전체 데이터셋 분석 및 인사이트 도출

        Args:
            tables: {table_name: {"columns": [...], "row_count": int}}
            data: {table_name: [row_dicts]}
            column_semantics: LLM이 분석한 컬럼 의미론 정보 (v14.0)
                {column_name: {"business_role": "optional_input", "nullable_by_design": True, ...}}

        Returns:
            비즈니스 인사이트 리스트
        """
        self.insights = []
        self.insight_counter = 0
        self.column_semantics = column_semantics or {}

        # v14.0: 도메인 정보 추출 (LLM 분석용)
        self._extract_domain_info()

        for table_name, rows in data.items():
            if not rows:
                continue

            table_info = tables.get(table_name, {})
            system = self._extract_system_name(table_name)

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

        # 11. v14.0: LLM 기반 도메인 특화 인사이트 생성 (전체 데이터 한 번에 분석)
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
                            except:
                                pass

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

                    # 성과 지표 저하 탐지
                    low_threshold = 0.7 if "rate" in col_lower else 70  # rate는 0-1, score는 0-100

                    low_count = sum(1 for v in values if v < low_threshold)
                    if low_count >= 10:
                        percentage = (low_count / len(values)) * 100

                        insight = self._create_insight(
                            title=f"Low {col.replace('_', ' ').title()} Performance",
                            description=f"Found {low_count:,} items ({percentage:.1f}%) with {col} below threshold ({low_threshold})",
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
            except:
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
        numeric_columns = [
            c for c in sample_row.keys()
            if isinstance(sample_row.get(c), (int, float))
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

    def _analyze_ml_anomalies(
        self,
        table_name: str,
        rows: List[Dict],
        table_info: Dict,
        system: str,
    ):
        """ML 기반 앙상블 이상치 탐지 (v2.0)"""
        if not self._anomaly_detector or not rows:
            return

        sample_row = rows[0]

        # 숫자 컬럼만 선택
        for col in sample_row.keys():
            if not col:
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
        v14.0: LLM 기반 도메인 특화 인사이트 생성

        LLM이 데이터를 분석하여 도메인에 맞는 KPI, 리스크, 권장 조치를 직접 판단합니다.
        하드코딩된 패턴 없이 LLM의 지식을 활용합니다.
        """
        if not self.llm_client:
            logger.debug("[v14.0] LLM client not available, skipping domain-specific insights")
            return

        try:
            from .utils import parse_llm_json
        except ImportError:
            logger.warning("[v14.0] parse_llm_json not available")
            return

        # 데이터 요약 생성
        data_summary = self._create_data_summary_for_llm(tables, data)

        prompt = f"""You are a business analyst expert in the **{self._detected_domain}** domain.

Analyze the following dataset and generate domain-specific business insights.

## Dataset Summary
{data_summary}

## Your Task

Based on your knowledge of the **{self._detected_domain}** industry, identify:

1. **Key KPIs**: Which columns represent important KPIs for this domain? Are there any concerning values?
2. **Risk Indicators**: What potential risks do you see in this data?
3. **Concentration Risks**: Is there over-reliance on specific entities (e.g., single supplier, carrier, plant)?
4. **Operational Issues**: Any patterns suggesting inefficiencies or problems?

## Output Format (JSON)

Return a JSON array of insights. Each insight should have:
```json
[
  {{
    "title": "Short descriptive title",
    "description": "Detailed explanation of the finding",
    "insight_type": "kpi" | "risk" | "anomaly" | "operational",
    "severity": "critical" | "high" | "medium" | "low" | "info",
    "source_table": "table name",
    "source_column": "column name",
    "metric_value": <number or null>,
    "recommendations": ["action 1", "action 2"],
    "business_impact": "Impact description"
  }}
]
```

Generate 2-5 domain-specific insights. Focus on actionable findings relevant to **{self._detected_domain}** operations.
Respond ONLY with valid JSON array."""

        try:
            response = self.llm_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=2000,
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
            logger.warning(f"[v14.0] LLM domain insight generation failed: {e}")

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

            # 숫자형 컬럼 통계
            numeric_stats = []
            for col in columns:
                numeric_values = []
                for row in rows[:100]:  # 샘플링
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
                    numeric_stats.append(f"  - {col}: avg={avg:.2f}, min={min_val:.2f}, max={max_val:.2f}")

            # 카테고리형 컬럼 분포
            categorical_stats = []
            for col in columns:
                values = [str(row.get(col, "")) for row in rows[:100] if row.get(col)]
                if values and len(set(values)) <= 20:  # 유니크 값이 적으면 카테고리
                    value_counts = Counter(values)
                    top_3 = value_counts.most_common(3)
                    dist = ", ".join([f"'{v}': {c}" for v, c in top_3])
                    categorical_stats.append(f"  - {col}: {dist}")

            summary_parts.append(f"""
### Table: {table_name}
- Rows: {len(rows)}
- Columns: {', '.join(columns[:10])}{'...' if len(columns) > 10 else ''}

Numeric columns:
{chr(10).join(numeric_stats[:5]) if numeric_stats else '  (none)'}

Categorical columns:
{chr(10).join(categorical_stats[:5]) if categorical_stats else '  (none)'}
""")

        return "\n".join(summary_parts[:5])  # 최대 5개 테이블
