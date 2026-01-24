"""
DSPy Modules for Ontoloty

프롬프트를 선언적으로 정의하고 자동 최적화하는 DSPy 기반 모듈입니다.

사용 예시:
    from src.unified_pipeline.dspy_modules import configure_dspy
    from src.unified_pipeline.dspy_modules import FKRelationshipAnalyzer

    # 초기화 (앱 시작 시 1회)
    configure_dspy(model="gpt-5.1")

    # 모듈 사용
    analyzer = FKRelationshipAnalyzer()
    result = analyzer(
        source_table="orders",
        source_column="customer_id",
        ...
    )
"""

from .config import configure_dspy, get_dspy_lm
from .modules import (
    FKRelationshipAnalyzer,
    ColumnSemanticClassifier,
    EntityAnalyzer,
    DynamicThresholdDecider,
    ConsensusVoter,
)
from .signatures import (
    FKRelationshipSignature,
    ColumnSemanticSignature,
    EntityAnalysisSignature,
    ThresholdDecisionSignature,
    ConsensusVoteSignature,
)

__all__ = [
    # Config
    "configure_dspy",
    "get_dspy_lm",
    # Modules
    "FKRelationshipAnalyzer",
    "ColumnSemanticClassifier",
    "EntityAnalyzer",
    "DynamicThresholdDecider",
    "ConsensusVoter",
    # Signatures
    "FKRelationshipSignature",
    "ColumnSemanticSignature",
    "EntityAnalysisSignature",
    "ThresholdDecisionSignature",
    "ConsensusVoteSignature",
]
