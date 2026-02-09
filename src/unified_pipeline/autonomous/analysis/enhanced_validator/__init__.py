"""
Enhanced Validation System v5.0

Research-based validation framework integrating:
1. Statistical Hypothesis Testing (KS, Welch, F-test)
2. Dempster-Shafer Evidence Theory for multi-source fusion
3. Confidence Calibration (Temperature Scaling, ECE)
4. Knowledge Graph Quality Metrics
5. Byzantine Fault Tolerant Consensus

References:
- "A Survey of Uncertainty in Deep Neural Networks" (arXiv:2107.03342)
- "Structural Quality Metrics to Evaluate Knowledge Graphs" (arXiv:2211.10011)
- "Dempster-Shafer Evidence Theory" (ScienceDirect)
- "Statistical Validation of Column Matching" (arXiv:2407.09885)
- "Rethinking the Reliability of Multi-agent System" (arXiv:2511.10400)
"""

# Re-export all public classes for backward compatibility
from .models import StatisticalTestResult, KGQualityScore, VoteType, AgentVote
from .statistical import StatisticalValidator
from .dempster_shafer import MassFunction, BPA, DempsterShaferFusion, DempsterShaferEngine
from .calibration import ConfidenceCalibrator
from .kg_validator import KnowledgeGraphQualityValidator
from .bft_consensus import BFTConsensus
from .validator import EnhancedValidator

__all__ = [
    # Models / Enums
    'StatisticalTestResult',
    'KGQualityScore',
    'VoteType',
    'AgentVote',
    # Statistical
    'StatisticalValidator',
    # Dempster-Shafer
    'MassFunction',
    'BPA',
    'DempsterShaferFusion',
    'DempsterShaferEngine',
    # Calibration
    'ConfidenceCalibrator',
    # KG Quality
    'KnowledgeGraphQualityValidator',
    # BFT Consensus
    'BFTConsensus',
    # Main orchestrator
    'EnhancedValidator',
]
