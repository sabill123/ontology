"""
Data Quality Analyzer - 데이터 품질 분석기 (v1.0)

Status/Enum 컬럼의 이상 값, Boolean 플래그, Outlier를 탐지합니다.

주요 기능:
1. Status/Enum 컬럼 탐지 및 이상 상태값 분석 (CANCELLED, GROUNDED 등)
2. Boolean 플래그 컬럼 분석 (damage_reported=TRUE 등)
3. Outlier 탐지 (수치형 컬럼)

사용 예:
    analyzer = DataQualityAnalyzer()
    report = analyzer.analyze(tables_data)
    llm_context = analyzer.generate_llm_context()
"""

import re
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Set, Tuple
from enum import Enum
from collections import Counter


class StatusCategory(Enum):
    """상태값 카테고리"""
    NORMAL = "normal"        # 정상 상태
    WARNING = "warning"      # 경고 (주의 필요)
    ERROR = "error"          # 오류/실패
    ANOMALY = "anomaly"      # 이상 상태


@dataclass
class StatusValueAnalysis:
    """Status/Enum 컬럼 분석 결과"""
    table: str
    column: str
    is_status_column: bool
    total_count: int
    unique_values: List[str]
    value_distribution: Dict[str, int]
    anomalous_values: List[str]      # 문제를 나타내는 값들
    normal_values: List[str]         # 정상 상태 값들
    warning_values: List[str]        # 경고 상태 값들
    anomaly_rate: float              # 이상 상태 비율
    business_impact: str             # 비즈니스 영향 설명


@dataclass
class BooleanFlagAnalysis:
    """Boolean 플래그 컬럼 분석 결과"""
    table: str
    column: str
    true_count: int
    false_count: int
    null_count: int
    total_count: int
    true_ratio: float
    flag_semantic: str               # 플래그의 의미
    is_problem_flag: bool            # 문제를 나타내는 플래그인지
    business_implication: str        # 비즈니스 영향


@dataclass
class OutlierAnalysis:
    """Outlier 분석 결과"""
    table: str
    column: str
    outlier_count: int
    outlier_ratio: float
    outlier_values: List[Any]
    lower_bound: float
    upper_bound: float
    mean: float
    std: float


@dataclass
class DataQualityReport:
    """데이터 품질 분석 종합 보고서"""
    status_analyses: List[StatusValueAnalysis]
    boolean_analyses: List[BooleanFlagAnalysis]
    outlier_analyses: List[OutlierAnalysis]
    summary: Dict[str, Any]


class StatusEnumAnalyzer:
    """
    Status/Enum 컬럼 분석기

    이상 상태값(CANCELLED, FAILED, GROUNDED 등)을 자동 탐지합니다.
    """

    # 이상 상태를 나타내는 패턴 (대소문자 무관)
    ANOMALOUS_PATTERNS = [
        r"cancel",
        r"fail",
        r"error",
        r"reject",
        r"damage",
        r"delay",
        r"ground",
        r"suspend",
        r"block",
        r"expire",
        r"invalid",
        r"abort",
        r"timeout",
        r"denied",
        r"refused",
        r"missing",
        r"lost",
        r"broken",
        r"overdue",
        r"late",
    ]

    # 경고 상태를 나타내는 패턴
    WARNING_PATTERNS = [
        r"pending",
        r"wait",
        r"hold",
        r"review",
        r"manual",
        r"check",
        r"attention",
        r"warn",
    ]

    # 정상 상태를 나타내는 패턴
    NORMAL_PATTERNS = [
        r"active",
        r"complete",
        r"success",
        r"approved",
        r"confirm",
        r"delivered",
        r"paid",
        r"valid",
        r"ok",
        r"done",
        r"finished",
        r"ready",
        r"open",
        r"available",
        r"on.?time",
        r"scheduled",
    ]

    # Status 컬럼을 나타내는 이름 패턴
    STATUS_COLUMN_PATTERNS = [
        r"status",
        r"state",
        r"stat",
        r"sts",
        r"condition",
        r"result",
        r"outcome",
    ]

    def detect_status_columns(
        self,
        table_data: Dict[str, List[Any]]
    ) -> List[str]:
        """
        테이블에서 Status/Enum 컬럼 탐지

        Args:
            table_data: {column_name: [values]}

        Returns:
            List[str]: Status 컬럼 목록
        """
        status_columns = []

        for column_name, values in table_data.items():
            # 1. 컬럼 이름으로 판단
            col_lower = column_name.lower()
            if any(re.search(p, col_lower) for p in self.STATUS_COLUMN_PATTERNS):
                status_columns.append(column_name)
                continue

            # 2. 값의 특성으로 판단
            if self._is_likely_status_column(values):
                status_columns.append(column_name)

        return status_columns

    def analyze_status_column(
        self,
        table: str,
        column: str,
        values: List[Any]
    ) -> StatusValueAnalysis:
        """
        Status 컬럼 분석

        Args:
            table: 테이블명
            column: 컬럼명
            values: 컬럼 값 목록

        Returns:
            StatusValueAnalysis: 분석 결과
        """
        # 문자열로 변환 및 정리
        str_values = [str(v).strip() for v in values if v is not None and str(v).strip()]
        total_count = len(str_values)

        if total_count == 0:
            return StatusValueAnalysis(
                table=table,
                column=column,
                is_status_column=False,
                total_count=0,
                unique_values=[],
                value_distribution={},
                anomalous_values=[],
                normal_values=[],
                warning_values=[],
                anomaly_rate=0.0,
                business_impact="No data available",
            )

        # 값 분포 계산
        distribution = Counter(str_values)
        unique_values = list(distribution.keys())

        # 값 카테고리화
        categorized = self.categorize_values(set(unique_values))

        anomalous_values = categorized.get(StatusCategory.ANOMALY, []) + \
                          categorized.get(StatusCategory.ERROR, [])
        warning_values = categorized.get(StatusCategory.WARNING, [])
        normal_values = categorized.get(StatusCategory.NORMAL, [])

        # 이상 상태 비율 계산
        anomaly_count = sum(distribution[v] for v in anomalous_values)
        anomaly_rate = anomaly_count / total_count if total_count > 0 else 0.0

        # 비즈니스 영향 평가
        business_impact = self._assess_business_impact(
            column, anomalous_values, anomaly_rate, distribution
        )

        return StatusValueAnalysis(
            table=table,
            column=column,
            is_status_column=True,
            total_count=total_count,
            unique_values=unique_values,
            value_distribution=dict(distribution),
            anomalous_values=anomalous_values,
            normal_values=normal_values,
            warning_values=warning_values,
            anomaly_rate=anomaly_rate,
            business_impact=business_impact,
        )

    def categorize_values(
        self,
        values: Set[str]
    ) -> Dict[StatusCategory, List[str]]:
        """
        상태값 카테고리화

        Args:
            values: 고유 값 집합

        Returns:
            Dict[StatusCategory, List[str]]: 카테고리별 값 목록
        """
        categorized = {
            StatusCategory.NORMAL: [],
            StatusCategory.WARNING: [],
            StatusCategory.ERROR: [],
            StatusCategory.ANOMALY: [],
        }

        for value in values:
            value_lower = value.lower()

            # 이상 상태 체크
            if any(re.search(p, value_lower) for p in self.ANOMALOUS_PATTERNS):
                categorized[StatusCategory.ANOMALY].append(value)
            # 경고 상태 체크
            elif any(re.search(p, value_lower) for p in self.WARNING_PATTERNS):
                categorized[StatusCategory.WARNING].append(value)
            # 정상 상태 체크
            elif any(re.search(p, value_lower) for p in self.NORMAL_PATTERNS):
                categorized[StatusCategory.NORMAL].append(value)
            # 분류 불가
            else:
                # 기본적으로 정상으로 분류
                categorized[StatusCategory.NORMAL].append(value)

        return categorized

    def _is_likely_status_column(self, values: List[Any]) -> bool:
        """값의 특성으로 Status 컬럼 여부 판단"""
        str_values = [str(v) for v in values if v is not None]
        if not str_values:
            return False

        unique_values = set(str_values)

        # 고유값이 너무 많으면 Status 컬럼이 아님
        if len(unique_values) > 20:
            return False

        # 고유값 중 Status 패턴이 있는지 확인
        status_pattern_count = 0
        for value in unique_values:
            value_lower = value.lower()
            if any(re.search(p, value_lower) for p in
                   self.ANOMALOUS_PATTERNS + self.WARNING_PATTERNS + self.NORMAL_PATTERNS):
                status_pattern_count += 1

        # 30% 이상이 Status 패턴이면 Status 컬럼
        return status_pattern_count / len(unique_values) >= 0.3 if unique_values else False

    def _assess_business_impact(
        self,
        column: str,
        anomalous_values: List[str],
        anomaly_rate: float,
        distribution: Counter
    ) -> str:
        """비즈니스 영향 평가"""
        if not anomalous_values:
            return "No anomalous values detected"

        impact_parts = []

        if anomaly_rate > 0.1:  # 10% 이상
            impact_parts.append(f"HIGH: {anomaly_rate:.1%} anomaly rate")
        elif anomaly_rate > 0.05:  # 5% 이상
            impact_parts.append(f"MEDIUM: {anomaly_rate:.1%} anomaly rate")
        else:
            impact_parts.append(f"LOW: {anomaly_rate:.1%} anomaly rate")

        # 주요 이상값 나열
        top_anomalies = sorted(
            [(v, distribution[v]) for v in anomalous_values],
            key=lambda x: x[1],
            reverse=True
        )[:3]

        if top_anomalies:
            details = ", ".join([f"{v}: {c}" for v, c in top_anomalies])
            impact_parts.append(f"Top issues: {details}")

        return "; ".join(impact_parts)


class BooleanFlagDetector:
    """
    Boolean 플래그 컬럼 분석기

    damage_reported=TRUE, is_delayed=TRUE 등의 플래그를 탐지합니다.
    """

    # Boolean 플래그 컬럼 이름 패턴
    FLAG_COLUMN_PATTERNS = [
        r"^is_",
        r"^has_",
        r"^was_",
        r"^can_",
        r"^should_",
        r"_flag$",
        r"_flg$",
        r"_reported$",
        r"_enabled$",
        r"_active$",
        r"_confirmed$",
        r"_verified$",
        r"_required$",
        r"_completed$",
    ]

    # 문제를 나타내는 플래그 패턴
    PROBLEM_FLAG_PATTERNS = [
        r"damage",
        r"error",
        r"fail",
        r"reject",
        r"cancel",
        r"delay",
        r"overdue",
        r"late",
        r"missing",
        r"lost",
        r"invalid",
        r"expired",
    ]

    # Boolean 값 패턴
    TRUE_VALUES = {"true", "1", "yes", "y", "t", "on", "enabled", "active"}
    FALSE_VALUES = {"false", "0", "no", "n", "f", "off", "disabled", "inactive"}

    def detect_boolean_columns(
        self,
        table_data: Dict[str, List[Any]]
    ) -> List[str]:
        """
        테이블에서 Boolean 컬럼 탐지

        Args:
            table_data: {column_name: [values]}

        Returns:
            List[str]: Boolean 컬럼 목록
        """
        boolean_columns = []

        for column_name, values in table_data.items():
            # 1. 컬럼 이름으로 판단
            col_lower = column_name.lower()
            if any(re.search(p, col_lower) for p in self.FLAG_COLUMN_PATTERNS):
                boolean_columns.append(column_name)
                continue

            # 2. 값의 특성으로 판단
            if self._is_likely_boolean_column(values):
                boolean_columns.append(column_name)

        return boolean_columns

    def analyze_flag_column(
        self,
        table: str,
        column: str,
        values: List[Any]
    ) -> BooleanFlagAnalysis:
        """
        Boolean 플래그 컬럼 분석

        Args:
            table: 테이블명
            column: 컬럼명
            values: 컬럼 값 목록

        Returns:
            BooleanFlagAnalysis: 분석 결과
        """
        true_count = 0
        false_count = 0
        null_count = 0

        for v in values:
            if v is None or str(v).strip() == "":
                null_count += 1
            elif str(v).lower().strip() in self.TRUE_VALUES:
                true_count += 1
            elif str(v).lower().strip() in self.FALSE_VALUES:
                false_count += 1
            else:
                # 숫자형 Boolean 처리
                try:
                    if float(v) != 0:
                        true_count += 1
                    else:
                        false_count += 1
                except (ValueError, TypeError):
                    null_count += 1

        total_count = len(values)
        non_null_count = true_count + false_count
        true_ratio = true_count / non_null_count if non_null_count > 0 else 0.0

        # 플래그 의미 추론
        flag_semantic = self._infer_flag_semantic(column)

        # 문제 플래그 여부
        is_problem_flag = any(
            re.search(p, column.lower()) for p in self.PROBLEM_FLAG_PATTERNS
        )

        # 비즈니스 영향 평가
        business_implication = self._assess_flag_impact(
            column, is_problem_flag, true_ratio, true_count
        )

        return BooleanFlagAnalysis(
            table=table,
            column=column,
            true_count=true_count,
            false_count=false_count,
            null_count=null_count,
            total_count=total_count,
            true_ratio=true_ratio,
            flag_semantic=flag_semantic,
            is_problem_flag=is_problem_flag,
            business_implication=business_implication,
        )

    def _is_likely_boolean_column(self, values: List[Any]) -> bool:
        """값의 특성으로 Boolean 컬럼 여부 판단"""
        if not values:
            return False

        str_values = [str(v).lower().strip() for v in values if v is not None]
        unique_values = set(str_values)

        # 고유값이 2개 이하이고 Boolean 패턴과 일치하면
        if len(unique_values) <= 3:  # True, False, (None)
            bool_values = self.TRUE_VALUES | self.FALSE_VALUES | {"", "null", "none"}
            if unique_values.issubset(bool_values):
                return True

        return False

    def _infer_flag_semantic(self, column: str) -> str:
        """플래그의 의미 추론"""
        col_lower = column.lower()

        if "damage" in col_lower:
            return "Indicates damage occurrence"
        elif "delay" in col_lower:
            return "Indicates delay occurrence"
        elif "cancel" in col_lower:
            return "Indicates cancellation"
        elif "error" in col_lower or "fail" in col_lower:
            return "Indicates error/failure"
        elif "active" in col_lower:
            return "Indicates active status"
        elif "complete" in col_lower:
            return "Indicates completion status"
        elif "verified" in col_lower or "confirm" in col_lower:
            return "Indicates verification status"
        elif "required" in col_lower:
            return "Indicates requirement flag"
        else:
            return f"Boolean flag for {column}"

    def _assess_flag_impact(
        self,
        column: str,
        is_problem_flag: bool,
        true_ratio: float,
        true_count: int
    ) -> str:
        """플래그의 비즈니스 영향 평가"""
        if not is_problem_flag:
            return f"{true_ratio:.1%} TRUE ratio"

        # 문제 플래그인 경우
        if true_ratio > 0.1:  # 10% 이상
            return f"CRITICAL: {true_ratio:.1%} ({true_count} cases) have {column}=TRUE"
        elif true_ratio > 0.05:  # 5% 이상
            return f"WARNING: {true_ratio:.1%} ({true_count} cases) have {column}=TRUE"
        elif true_ratio > 0:
            return f"NOTICE: {true_ratio:.1%} ({true_count} cases) have {column}=TRUE"
        else:
            return "No issues detected"


class DataQualityAnalyzer:
    """
    데이터 품질 종합 분석기

    Status/Enum, Boolean 플래그, Outlier를 통합 분석합니다.
    """

    def __init__(self):
        self.status_analyzer = StatusEnumAnalyzer()
        self.flag_detector = BooleanFlagDetector()
        self._last_report: Optional[DataQualityReport] = None

    def analyze(
        self,
        tables_data: Dict[str, Dict[str, List[Any]]]
    ) -> DataQualityReport:
        """
        전체 데이터 품질 분석

        Args:
            tables_data: {table_name: {column_name: [values]}}

        Returns:
            DataQualityReport: 종합 분석 보고서
        """
        status_analyses = []
        boolean_analyses = []
        outlier_analyses = []

        for table_name, table_data in tables_data.items():
            # 1. Status 컬럼 분석
            status_columns = self.status_analyzer.detect_status_columns(table_data)
            for col in status_columns:
                analysis = self.status_analyzer.analyze_status_column(
                    table_name, col, table_data[col]
                )
                if analysis.anomalous_values:  # 이상값이 있는 경우만 포함
                    status_analyses.append(analysis)

            # 2. Boolean 플래그 분석
            boolean_columns = self.flag_detector.detect_boolean_columns(table_data)
            for col in boolean_columns:
                analysis = self.flag_detector.analyze_flag_column(
                    table_name, col, table_data[col]
                )
                if analysis.is_problem_flag and analysis.true_count > 0:
                    boolean_analyses.append(analysis)

            # 3. Outlier 분석 (수치형 컬럼)
            for col, values in table_data.items():
                if col in status_columns or col in boolean_columns:
                    continue
                outlier_analysis = self._analyze_outliers(table_name, col, values)
                if outlier_analysis and outlier_analysis.outlier_ratio > 0.01:
                    outlier_analyses.append(outlier_analysis)

        # 요약 생성
        summary = self._generate_summary(
            status_analyses, boolean_analyses, outlier_analyses
        )

        self._last_report = DataQualityReport(
            status_analyses=status_analyses,
            boolean_analyses=boolean_analyses,
            outlier_analyses=outlier_analyses,
            summary=summary,
        )

        return self._last_report

    def generate_llm_context(self) -> str:
        """
        LLM에게 전달할 데이터 품질 컨텍스트 생성

        Returns:
            str: LLM 프롬프트용 컨텍스트
        """
        if not self._last_report:
            return "No data quality analysis available."

        lines = ["## DATA QUALITY ANALYSIS RESULTS", ""]

        # Status/Enum 이슈
        if self._last_report.status_analyses:
            lines.append("### Status/Enum Anomalies Detected")
            for analysis in self._last_report.status_analyses:
                lines.append(
                    f"- **{analysis.table}.{analysis.column}**: "
                    f"{analysis.anomaly_rate:.1%} anomaly rate"
                )
                lines.append(f"  - Anomalous values: {', '.join(analysis.anomalous_values[:5])}")
                lines.append(f"  - Distribution: {dict(list(analysis.value_distribution.items())[:5])}")
                lines.append(f"  - Impact: {analysis.business_impact}")
            lines.append("")

        # Boolean 플래그 이슈
        if self._last_report.boolean_analyses:
            lines.append("### Problem Flags Detected")
            for analysis in self._last_report.boolean_analyses:
                lines.append(
                    f"- **{analysis.table}.{analysis.column}**: "
                    f"{analysis.true_ratio:.1%} TRUE ({analysis.true_count} cases)"
                )
                lines.append(f"  - Meaning: {analysis.flag_semantic}")
                lines.append(f"  - Impact: {analysis.business_implication}")
            lines.append("")

        # 요약
        if self._last_report.summary:
            lines.append("### Summary")
            lines.append(f"- Total status issues: {self._last_report.summary.get('total_status_issues', 0)}")
            lines.append(f"- Total flag issues: {self._last_report.summary.get('total_flag_issues', 0)}")
            lines.append(f"- Critical items: {self._last_report.summary.get('critical_count', 0)}")

        return "\n".join(lines)

    def _analyze_outliers(
        self,
        table: str,
        column: str,
        values: List[Any]
    ) -> Optional[OutlierAnalysis]:
        """Outlier 분석 (IQR 방식)"""
        # 숫자로 변환 가능한 값만 추출
        numeric_values = []
        for v in values:
            try:
                if v is not None:
                    numeric_values.append(float(v))
            except (ValueError, TypeError):
                continue

        if len(numeric_values) < 10:
            return None

        import statistics

        try:
            mean = statistics.mean(numeric_values)
            std = statistics.stdev(numeric_values) if len(numeric_values) > 1 else 0
            sorted_values = sorted(numeric_values)

            # IQR 계산
            n = len(sorted_values)
            q1_idx = n // 4
            q3_idx = (3 * n) // 4
            q1 = sorted_values[q1_idx]
            q3 = sorted_values[q3_idx]
            iqr = q3 - q1

            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr

            # Outlier 탐지
            outliers = [v for v in numeric_values if v < lower_bound or v > upper_bound]
            outlier_ratio = len(outliers) / len(numeric_values)

            return OutlierAnalysis(
                table=table,
                column=column,
                outlier_count=len(outliers),
                outlier_ratio=outlier_ratio,
                outlier_values=outliers[:10],
                lower_bound=lower_bound,
                upper_bound=upper_bound,
                mean=mean,
                std=std,
            )
        except Exception:
            return None

    def _generate_summary(
        self,
        status_analyses: List[StatusValueAnalysis],
        boolean_analyses: List[BooleanFlagAnalysis],
        outlier_analyses: List[OutlierAnalysis]
    ) -> Dict[str, Any]:
        """요약 생성"""
        critical_count = 0

        # 높은 이상율 카운트
        for a in status_analyses:
            if a.anomaly_rate > 0.1:
                critical_count += 1

        for a in boolean_analyses:
            if a.true_ratio > 0.1:
                critical_count += 1

        return {
            "total_status_issues": len(status_analyses),
            "total_flag_issues": len(boolean_analyses),
            "total_outlier_issues": len(outlier_analyses),
            "critical_count": critical_count,
            "tables_with_issues": len(set(
                [a.table for a in status_analyses] +
                [a.table for a in boolean_analyses]
            )),
        }


# === Convenience Functions ===

def analyze_data_quality(
    tables_data: Dict[str, Dict[str, List[Any]]]
) -> DataQualityReport:
    """데이터 품질 분석 (편의 함수)"""
    return DataQualityAnalyzer().analyze(tables_data)
