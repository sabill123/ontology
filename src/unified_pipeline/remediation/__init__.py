"""
Auto-remediation Module (v17.0)

데이터 품질 자동 수정:
- 이상 탐지 기반 수정
- 수정 전략 관리
- 영향 분석
- 승인 워크플로우

사용 예시:
    engine = RemediationEngine(context)

    # 이상치 탐지 및 수정
    result = await engine.remediate(
        table="customers",
        issue_type="missing_values",
        auto_approve=False,
    )
"""

from .remediation_engine import (
    RemediationEngine,
    RemediationAction,
    RemediationResult,
    IssueType,
)

__all__ = [
    "RemediationEngine",
    "RemediationAction",
    "RemediationResult",
    "IssueType",
]
