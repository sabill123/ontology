"""
Data Integration Causal Model - 데이터 통합 시나리오 인과 모델

DataIntegrationCausalModel: 데이터 통합(처리)이 비즈니스 KPI(결과)에 미치는 영향을 분석
"""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np

from .models import (
    CausalGraph,
    CausalNode,
    CausalEdge,
    TreatmentEffect,
    CausalImpactResult,
    SensitivityResult,
)
from .estimators import ATEEstimator, CATEEstimator
from .counterfactual import CounterfactualAnalyzer, SensitivityAnalyzer


class DataIntegrationCausalModel:
    """
    데이터 통합 시나리오에 특화된 인과 모델

    데이터 통합(처리)이 비즈니스 KPI(결과)에 미치는 영향을 분석
    """

    def __init__(self):
        self.causal_graph = CausalGraph()
        self.ate_estimator = ATEEstimator(method="doubly_robust")
        self.cate_estimator = CATEEstimator(method="t_learner")
        self.cf_analyzer = CounterfactualAnalyzer()
        self.sens_analyzer = SensitivityAnalyzer()

    def build_causal_graph(
        self,
        treatment_vars: List[str],
        outcome_vars: List[str],
        confounders: List[str] = None,
        mediators: List[str] = None
    ) -> CausalGraph:
        """인과 그래프 구축"""
        self.causal_graph = CausalGraph()

        # 처리 변수
        for t in treatment_vars:
            self.causal_graph.add_node(CausalNode(
                name=t,
                node_type="treatment"
            ))

        # 결과 변수
        for o in outcome_vars:
            self.causal_graph.add_node(CausalNode(
                name=o,
                node_type="outcome"
            ))

        # 교란 변수
        if confounders:
            for c in confounders:
                self.causal_graph.add_node(CausalNode(
                    name=c,
                    node_type="confounder"
                ))
                # 교란 -> 처리, 교란 -> 결과
                for t in treatment_vars:
                    self.causal_graph.add_edge(CausalEdge(source=c, target=t))
                for o in outcome_vars:
                    self.causal_graph.add_edge(CausalEdge(source=c, target=o))

        # 매개 변수
        if mediators:
            for m in mediators:
                self.causal_graph.add_node(CausalNode(
                    name=m,
                    node_type="mediator"
                ))
                # 처리 -> 매개 -> 결과
                for t in treatment_vars:
                    self.causal_graph.add_edge(CausalEdge(source=t, target=m))
                for o in outcome_vars:
                    self.causal_graph.add_edge(CausalEdge(source=m, target=o))

        # 처리 -> 결과 (직접 효과)
        for t in treatment_vars:
            for o in outcome_vars:
                self.causal_graph.add_edge(CausalEdge(source=t, target=o))

        return self.causal_graph

    def estimate_integration_impact(
        self,
        tables_info: Dict[str, Any],
        integration_scenario: Dict[str, Any],
        baseline_metrics: Dict[str, float] = None
    ) -> CausalImpactResult:
        """
        데이터 통합이 비즈니스 지표에 미치는 영향 추정

        시뮬레이션 기반 접근: 실제 데이터가 없으므로
        스키마/관계 복잡도에서 파생된 특성을 사용
        """
        # 1. 피처 추출
        features = self._extract_features(tables_info, integration_scenario)

        # 2. 시뮬레이션 데이터 생성
        X, treatment, outcome = self._simulate_data(
            features,
            baseline_metrics or {}
        )

        # 3. ATE 추정
        ate_result = self.ate_estimator.estimate(X, treatment, outcome)

        # 4. CATE 추정 (이질적 효과)
        cate, cate_std = self.cate_estimator.estimate(X, treatment, outcome)

        # 5. 반사실적 분석
        cf_results = []
        for i in range(min(3, len(X))):
            cf = self.cf_analyzer.analyze(X, treatment, outcome, i)
            cf_results.append(cf)

        # 6. 민감도 분석
        sensitivity = self.sens_analyzer.analyze(
            ate_result.estimate,
            ate_result.std_error
        )

        # 7. 인과 그래프 구축
        treatment_vars = list(integration_scenario.get("treatments", ["data_integration"]))
        outcome_vars = list(integration_scenario.get("outcomes", ["business_kpi"]))
        self.build_causal_graph(
            treatment_vars=treatment_vars,
            outcome_vars=outcome_vars,
            confounders=["data_complexity", "schema_similarity"]
        )

        # 8. 권장사항 생성
        recommendations = self._generate_recommendations(
            ate_result,
            cate,
            sensitivity,
            integration_scenario
        )

        # 9. 요약
        summary = {
            "estimated_ate": ate_result.estimate,
            "ate_confidence_interval": ate_result.confidence_interval,
            "ate_significant": ate_result.p_value < 0.05,
            "heterogeneous_effects": bool(np.std(cate) > 0.1),
            "cate_range": (float(np.min(cate)), float(np.max(cate))),
            "robustness": sensitivity.robustness_value,
            "sample_size": ate_result.sample_size
        }

        return CausalImpactResult(
            treatment_effects=[ate_result],
            counterfactuals=cf_results,
            sensitivity=sensitivity,
            causal_graph=self.causal_graph,
            recommendations=recommendations,
            summary=summary
        )

    def _extract_features(
        self,
        tables_info: Dict[str, Any],
        integration_scenario: Dict[str, Any]
    ) -> Dict[str, float]:
        """테이블/통합 시나리오에서 피처 추출"""
        features = {}

        # 테이블 수
        n_tables = len(tables_info) if tables_info else 1
        features["n_tables"] = n_tables

        # 평균 컬럼 수
        total_cols = 0
        for table_name, table_data in tables_info.items():
            if isinstance(table_data, dict):
                cols = table_data.get("columns", [])
                total_cols += len(cols) if isinstance(cols, list) else 0
        features["avg_columns"] = total_cols / n_tables if n_tables > 0 else 0

        # 통합 복잡도
        n_fks = integration_scenario.get("n_fk_relations", 0)
        n_entities = integration_scenario.get("n_entity_matches", 0)
        features["integration_complexity"] = (n_fks * 0.3 + n_entities * 0.7)

        # 스키마 유사도 (있으면)
        features["schema_similarity"] = integration_scenario.get("schema_similarity", 0.5)

        return features

    def _simulate_data(
        self,
        features: Dict[str, float],
        baseline: Dict[str, float]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        시뮬레이션 데이터 생성

        실제 과거 데이터가 없으므로, 특성 기반으로 시뮬레이션
        """
        n_samples = 100
        np.random.seed(42)

        # 피처 행렬
        X = np.random.randn(n_samples, 4)
        X[:, 0] = features.get("n_tables", 3) + np.random.randn(n_samples) * 0.5
        X[:, 1] = features.get("avg_columns", 10) + np.random.randn(n_samples) * 2
        X[:, 2] = features.get("integration_complexity", 5) + np.random.randn(n_samples)
        X[:, 3] = features.get("schema_similarity", 0.5) + np.random.randn(n_samples) * 0.1

        # 처리 할당 (경향 점수 기반)
        propensity = 0.3 + 0.1 * X[:, 2] + 0.2 * X[:, 3]
        propensity = np.clip(propensity, 0.1, 0.9)
        treatment = (np.random.rand(n_samples) < propensity).astype(float)

        # 결과 생성 (처리 효과 + 노이즈)
        # 실제 효과: 복잡도 높을수록, 유사도 높을수록 효과 큼
        base_effect = 0.15  # 15% 개선
        heterogeneous_effect = 0.05 * X[:, 2] + 0.1 * X[:, 3]

        # Y = base + treatment_effect * treatment + confounding + noise
        outcome = (
            baseline.get("baseline_kpi", 0.7) +
            (base_effect + heterogeneous_effect) * treatment +
            0.05 * X[:, 0] +  # 교란
            np.random.randn(n_samples) * 0.05  # 노이즈
        )

        return X, treatment, outcome

    def _generate_recommendations(
        self,
        ate: TreatmentEffect,
        cate: np.ndarray,
        sensitivity: SensitivityResult,
        scenario: Dict[str, Any]
    ) -> List[str]:
        """권장사항 생성"""
        recommendations = []

        # ATE 기반
        if ate.estimate > 0.1:
            recommendations.append(
                f"데이터 통합으로 평균 {ate.estimate*100:.1f}%의 KPI 개선이 예상됩니다"
            )
        elif ate.estimate > 0:
            recommendations.append(
                f"소폭의 KPI 개선({ate.estimate*100:.1f}%)이 예상되나 효과가 제한적입니다"
            )
        else:
            recommendations.append(
                "현재 통합 시나리오에서는 큰 효과가 기대되지 않습니다"
            )

        # 신뢰구간 기반
        if ate.confidence_interval[0] > 0:
            recommendations.append("효과의 방향(개선)은 통계적으로 유의미합니다")
        else:
            recommendations.append("효과의 방향성에 불확실성이 있습니다")

        # CATE 기반 (이질적 효과)
        cate_std = np.std(cate)
        if cate_std > 0.05:
            recommendations.append(
                f"테이블/시나리오에 따라 효과 크기가 다릅니다 (표준편차: {cate_std:.2f})"
            )
            high_effect_pct = np.mean(cate > np.mean(cate)) * 100
            recommendations.append(
                f"약 {high_effect_pct:.0f}%의 케이스에서 평균 이상의 효과가 기대됩니다"
            )

        # 민감도 기반
        if sensitivity.robustness_value < 0.5:
            recommendations.append(
                "결과가 미관측 요인에 민감할 수 있으므로 추가 검증 권장"
            )
        else:
            recommendations.append("결과는 미관측 교란에 대해 비교적 강건합니다")

        return recommendations
