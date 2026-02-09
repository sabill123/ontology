"""
Business Insights Package

Re-exports all public symbols for backward compatibility.
Previously this was a single business_insights.py module.
"""

from .models import InsightType, InsightSeverity, BusinessInsight
from .analyzer import BusinessInsightsAnalyzer

__all__ = [
    "InsightType",
    "InsightSeverity",
    "BusinessInsight",
    "BusinessInsightsAnalyzer",
]
