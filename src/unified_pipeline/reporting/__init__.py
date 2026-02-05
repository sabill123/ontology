"""
NL Reports Module (v17.0)

자연어 보고서 자동 생성:
- 데이터 분석 요약
- 인사이트 내러티브
- Executive Summary
- 다국어 지원

사용 예시:
    generator = ReportGenerator(context)

    # 보고서 생성
    report = await generator.generate(
        report_type="executive_summary",
        data=analysis_results,
        language="ko",
    )

    print(report.content)
"""

from .report_generator import (
    ReportGenerator,
    Report,
    ReportType,
    ReportSection,
)

__all__ = [
    "ReportGenerator",
    "Report",
    "ReportType",
    "ReportSection",
]
