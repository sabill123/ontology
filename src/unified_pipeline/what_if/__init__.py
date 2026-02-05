"""
What-If Analysis Module (v17.0)

가상 시나리오 시뮬레이션:
- 시나리오 정의
- 인과 그래프 기반 영향 예측
- Monte Carlo 시뮬레이션
- 의사결정 지원

사용 예시:
    analyzer = WhatIfAnalyzer(context)

    # 시나리오 정의
    scenario = analyzer.create_scenario(
        name="Price Increase",
        changes={"product.price": 1.1},  # 10% 인상
    )

    # 시뮬레이션 실행
    result = await analyzer.simulate(scenario)

    # 영향 분석
    print(result.impact_summary)
"""

from .what_if_analyzer import (
    WhatIfAnalyzer,
    Scenario,
    SimulationResult,
    ImpactPrediction,
)

__all__ = [
    "WhatIfAnalyzer",
    "Scenario",
    "SimulationResult",
    "ImpactPrediction",
]
