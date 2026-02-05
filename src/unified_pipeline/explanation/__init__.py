"""
Explanation (XAI) Module (v17.0)

설명 가능한 AI (Explainable AI):
- Evidence Chain 기반 결정 설명
- Feature Attribution
- Counterfactual 생성
- 자연어 설명 생성

사용 예시:
    explainer = DecisionExplainer(context)

    # 결정 설명
    explanation = await explainer.explain_decision(decision_id)

    # 반사실적 설명
    counterfactuals = await explainer.generate_counterfactuals(decision_id)
"""

from .decision_explainer import (
    DecisionExplainer,
    Explanation,
    FeatureAttribution,
    Counterfactual,
)

__all__ = [
    "DecisionExplainer",
    "Explanation",
    "FeatureAttribution",
    "Counterfactual",
]
