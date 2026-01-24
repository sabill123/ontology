"""
Domain Module - Auto Domain Detection and Context

자동 도메인 탐지 및 컨텍스트 관리 모듈
- DomainContext: 도메인 정보 저장
- DomainDetector: LLM 기반 도메인 자동 탐지
- InsightPattern: Phase 2용 인사이트 패턴
"""

from .domain_context import (
    DomainContext,
    InsightPattern,
    IndustryDomain,
    BusinessStage,
    DataType,
    DEFAULT_DOMAIN_CONTEXT,
)
from .domain_detector import (
    DomainDetector,
    create_domain_detector,
    DOMAIN_SIGNALS,
    INDUSTRY_RECOMMENDATIONS,
)

__all__ = [
    # Domain Context
    "DomainContext",
    "InsightPattern",
    "IndustryDomain",
    "BusinessStage",
    "DataType",
    "DEFAULT_DOMAIN_CONTEXT",
    # Domain Detector
    "DomainDetector",
    "create_domain_detector",
    "DOMAIN_SIGNALS",
    "INDUSTRY_RECOMMENDATIONS",
]
