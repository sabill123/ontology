"""
Business Insights Data Models

InsightType, InsightSeverity, BusinessInsight dataclass.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
from enum import Enum


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
        result = {
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
        # v19.4: Palantir-style fields (UnifiedInsightPipeline에서 설정)
        for attr in ("_hypothesis", "_evidence", "_validation_summary",
                      "_prescriptive_actions", "_confidence_breakdown",
                      "_algorithm_findings", "_simulation_evidence", "_phases_completed"):
            val = getattr(self, attr, None)
            if val is not None:
                result[attr.lstrip("_")] = val
        return result
