"""
Causal Impact Analysis Package - 인과 추론 및 영향 분석

데이터 통합이 KPI에 미치는 인과적 영향을 추정합니다.

주요 기능:
- DoWhy 스타일 인과 그래프 모델링
- 처리 효과 추정 (ATE, CATE)
- 반사실적 분석 (Counterfactual)
- 민감도 분석 (Sensitivity Analysis)
- 데이터 통합 시나리오별 영향 예측
- Granger Causality 시계열 인과관계 분석
- PC Algorithm 기반 Causal Graph Discovery
- CATE 추정 (S-Learner, T-Learner, X-Learner)

References:
- Pearl, "Causality: Models, Reasoning, and Inference", 2009
- Sharma & Kiciman, "DoWhy: An End-to-End Library for Causal Inference", 2020
- Athey & Imbens, "Machine Learning Methods for Estimating Heterogeneous Causal Effects", 2015
- Granger, "Investigating Causal Relations by Econometric Models", 1969
- Spirtes, Glymour & Scheines, "Causation, Prediction, and Search", 2000
- Kunzel et al., "Metalearners for Estimating Heterogeneous Treatment Effects", 2019
"""

# Models (Enums + Dataclasses)
from .models import (
    TreatmentType,
    EffectType,
    CausalNode,
    CausalEdge,
    CausalGraph,
    TreatmentEffect,
    CounterfactualResult,
    SensitivityResult,
    CausalImpactResult,
    GrangerCausalityResult,
    PCAlgorithmResult,
    CATEResult,
)

# Estimators
from .estimators import (
    PropensityScoreEstimator,
    OutcomeModel,
    ATEEstimator,
    CATEEstimator,
    EnhancedCATEEstimator,
)

# Counterfactual & Sensitivity
from .counterfactual import (
    CounterfactualAnalyzer,
    SensitivityAnalyzer,
)

# Granger Causality
from .granger import GrangerCausalityAnalyzer

# PC Algorithm
from .pc_algorithm import PCAlgorithm

# Data Integration Causal Model
from .integration import DataIntegrationCausalModel

# Main Orchestrator
from .analyzer import CausalImpactAnalyzer

__all__ = [
    # Models
    'TreatmentType',
    'EffectType',
    'CausalNode',
    'CausalEdge',
    'CausalGraph',
    'TreatmentEffect',
    'CounterfactualResult',
    'SensitivityResult',
    'CausalImpactResult',
    'GrangerCausalityResult',
    'PCAlgorithmResult',
    'CATEResult',
    # Estimators
    'PropensityScoreEstimator',
    'OutcomeModel',
    'ATEEstimator',
    'CATEEstimator',
    'EnhancedCATEEstimator',
    # Counterfactual & Sensitivity
    'CounterfactualAnalyzer',
    'SensitivityAnalyzer',
    # Granger
    'GrangerCausalityAnalyzer',
    # PC Algorithm
    'PCAlgorithm',
    # Integration
    'DataIntegrationCausalModel',
    # Main Orchestrator
    'CausalImpactAnalyzer',
]
