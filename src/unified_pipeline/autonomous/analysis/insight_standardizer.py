"""
Insight Standardizer Module (v1.0)

인사이트 타이틀 및 형식 표준화
- 검증 가능한 표준 형식으로 변환
- 중복 제거
- 심각도 검증
"""

import logging
import re
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class StandardizedInsight:
    """표준화된 인사이트"""
    insight_id: str
    title: str
    standardized_title: str  # 검증용 표준 타이틀
    insight_type: str
    severity: str
    source_table: str
    source_column: str
    description: str
    metric_value: Any
    metric_name: str
    count: int
    percentage: float
    benchmark: Optional[float]
    deviation: Optional[float]
    recommendations: List[str]
    business_impact: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "insight_id": self.insight_id,
            "title": self.standardized_title,  # 표준화된 타이틀 사용
            "original_title": self.title,
            "insight_type": self.insight_type,
            "severity": self.severity,
            "source_table": self.source_table,
            "source_column": self.source_column,
            "description": self.description,
            "metric_value": self.metric_value,
            "metric_name": self.metric_name,
            "count": self.count,
            "percentage": self.percentage,
            "benchmark": self.benchmark,
            "deviation": self.deviation,
            "recommendations": self.recommendations,
            "business_impact": self.business_impact,
        }


class InsightStandardizer:
    """
    인사이트 표준화기

    다양한 형식의 인사이트를 검증 가능한 표준 형식으로 변환
    """

    # 표준 타이틀 매핑 (원본 키워드 → 표준 형식)
    TITLE_MAPPINGS = {
        # CTR 관련
        "ctr": {
            "low": "Low CTR Alert",
            "high": "High CTR Alert",
            "excellent": "Excellent CTR",
            "average": "Average CTR",
            "good": "Good CTR",
            "anomaly": "CTR Anomaly",
        },
        # CPC 관련
        "cpc": {
            "low": "Low CPC",
            "high": "High CPC Alert",
            "excellent": "Excellent CPC",
            "average": "Average CPC",
            "good": "Good CPC",
        },
        # ROAS 관련
        "roas": {
            "low": "Low ROAS Alert",
            "high": "High ROAS",
            "excellent": "Excellent ROAS",
            "average": "Average ROAS",
            "good": "Good ROAS",
            "negative": "Negative ROAS Alert",
        },
        # Conversion Rate 관련
        "conversion": {
            "low": "Low Conversion Rate Alert",
            "high": "High Conversion Rate",
            "excellent": "Excellent Conversion Rate",
            "average": "Average Conversion Rate",
            "good": "Good Conversion Rate",
        },
        # Quality Score 관련
        "quality": {
            "low": "Low Quality Score Alert",
            "high": "High Quality Score",
            "excellent": "Excellent Quality Score",
            "average": "Average Quality Score",
            "good": "Good Quality Score",
        },
        # Data Quality 관련
        "missing": {
            "default": "Missing Data",
        },
        "null": {
            "default": "NULL Values Detected",
        },
    }

    # 심각도별 키워드
    SEVERITY_KEYWORDS = {
        "critical": ["critical", "negative", "urgent", "immediate"],
        "high": ["high", "alert", "warning", "drop", "low"],  # Low CTR = High severity
        "medium": ["medium", "moderate", "average"],
        "low": ["low", "minor"],  # context-dependent
        "info": ["excellent", "good", "top", "best", "info"],
    }

    def __init__(self):
        self._seen_insights: set = set()

    def standardize(self, insights: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        인사이트 리스트 표준화

        Args:
            insights: 원본 인사이트 리스트

        Returns:
            표준화된 인사이트 리스트
        """
        self._seen_insights = set()
        standardized = []

        for insight in insights:
            std_insight = self._standardize_single(insight)
            if std_insight:
                # 중복 체크
                key = self._get_insight_key(std_insight)
                if key not in self._seen_insights:
                    self._seen_insights.add(key)
                    standardized.append(std_insight)

        return standardized

    def _standardize_single(self, insight: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """단일 인사이트 표준화"""
        original_title = insight.get("title", "")
        metric_name = insight.get("metric_name", "").lower()
        severity = insight.get("severity", "info")

        # 표준 타이틀 생성
        standardized_title = self._generate_standard_title(original_title, metric_name, severity)

        # 심각도 검증 및 조정
        verified_severity = self._verify_severity(standardized_title, severity, insight)

        # 표준화된 인사이트 반환
        result = insight.copy()
        result["title"] = standardized_title
        result["original_title"] = original_title
        result["severity"] = verified_severity

        return result

    def _generate_standard_title(self, original_title: str, metric_name: str, severity: str) -> str:
        """표준 타이틀 생성"""
        title_lower = original_title.lower()

        # 메트릭별 표준화
        for metric_key, title_map in self.TITLE_MAPPINGS.items():
            if metric_key in title_lower or metric_key in metric_name:
                # 상태 키워드 찾기
                for status_key, standard_title in title_map.items():
                    if status_key in title_lower:
                        # 소스 테이블/플랫폼 정보 추출
                        platform = self._extract_platform(original_title)
                        if platform:
                            return f"{standard_title}: {platform}"
                        return standard_title

                # 기본값 사용
                if "default" in title_map:
                    return title_map["default"]

        # 매핑 없으면 원본 사용 (일부 정리)
        return self._clean_title(original_title)

    def _extract_platform(self, title: str) -> Optional[str]:
        """타이틀에서 플랫폼 정보 추출"""
        platforms = ["Google Ads", "Meta Ads", "Naver Ads", "Facebook", "Instagram"]
        for platform in platforms:
            if platform.lower() in title.lower():
                return platform

        # 패턴으로 추출 시도
        match = re.search(r'-\s*(.+)$', title)
        if match:
            return match.group(1).strip()

        return None

    def _clean_title(self, title: str) -> str:
        """타이틀 정리"""
        # 불필요한 접두사/접미사 제거
        title = re.sub(r'^(Alert:|Warning:|Info:)\s*', '', title, flags=re.IGNORECASE)
        return title.strip()

    def _verify_severity(self, title: str, original_severity: str, insight: Dict) -> str:
        """심각도 검증 및 조정"""
        title_lower = title.lower()

        # 특정 키워드가 있으면 심각도 조정
        if any(kw in title_lower for kw in ["critical", "negative roas", "urgent"]):
            return "critical"

        if any(kw in title_lower for kw in ["low ctr alert", "high cpc alert", "low roas alert", "low quality"]):
            return "high"

        if any(kw in title_lower for kw in ["low conversion", "missing data", "null"]):
            # deviation 기반으로 세분화
            deviation = insight.get("deviation")
            if deviation and abs(deviation) > 50:
                return "high"
            return "medium"

        if any(kw in title_lower for kw in ["excellent", "good", "top performing"]):
            return "info"

        # 원본 유지
        return original_severity

    def _get_insight_key(self, insight: Dict[str, Any]) -> str:
        """중복 체크용 키 생성"""
        return f"{insight.get('source_table', '')}_{insight.get('source_column', '')}_{insight.get('title', '')}_{insight.get('metric_name', '')}"


def standardize_insights(insights: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """편의 함수: 인사이트 표준화"""
    standardizer = InsightStandardizer()
    return standardizer.standardize(insights)
