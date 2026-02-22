"""
What-If Analyzer (v17.0)

가상 시나리오 시뮬레이션:
- 시나리오 정의 및 관리
- 인과 그래프 기반 영향 분석
- Monte Carlo 시뮬레이션
- 시나리오 비교

사용 예시:
    analyzer = WhatIfAnalyzer(context)

    # 시나리오 생성
    scenario = analyzer.create_scenario(
        name="Revenue Increase",
        changes={"sales.amount": {"multiply": 1.2}},
    )

    # 시뮬레이션
    result = await analyzer.simulate(scenario)

    # 시나리오 비교
    comparison = await analyzer.compare_scenarios([scenario1, scenario2])
"""

import logging
import random
import copy
import json
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple, TYPE_CHECKING
from datetime import datetime
from enum import Enum

if TYPE_CHECKING:
    from ..shared_context import SharedContext

logger = logging.getLogger(__name__)


class ChangeType(str, Enum):
    """변경 유형"""
    SET = "set"  # 절대값 설정
    ADD = "add"  # 더하기
    MULTIPLY = "multiply"  # 곱하기
    PERCENTAGE = "percentage"  # 퍼센트 변경


@dataclass
class Change:
    """시나리오 변경"""
    target: str  # table.column 형식
    change_type: ChangeType
    value: Any

    def apply(self, original: Any) -> Any:
        """변경 적용"""
        if self.change_type == ChangeType.SET:
            return self.value
        elif self.change_type == ChangeType.ADD:
            return original + self.value
        elif self.change_type == ChangeType.MULTIPLY:
            return original * self.value
        elif self.change_type == ChangeType.PERCENTAGE:
            return original * (1 + self.value / 100)
        return original

    def to_dict(self) -> Dict[str, Any]:
        return {
            "target": self.target,
            "change_type": self.change_type.value,
            "value": self.value,
        }


@dataclass
class Scenario:
    """시나리오 정의"""
    scenario_id: str
    name: str
    description: str = ""
    changes: List[Change] = field(default_factory=list)

    # 설정
    num_simulations: int = 100  # Monte Carlo 반복 횟수
    confidence_level: float = 0.95

    # 메타데이터
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    created_by: str = "user"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "name": self.name,
            "description": self.description,
            "changes": [c.to_dict() for c in self.changes],
            "num_simulations": self.num_simulations,
            "created_at": self.created_at,
        }


@dataclass
class ImpactPrediction:
    """영향 예측"""
    target: str
    original_value: Any
    predicted_value: Any
    change_percent: float
    confidence_interval: Tuple[float, float]
    impact_type: str  # positive, negative, neutral

    def to_dict(self) -> Dict[str, Any]:
        return {
            "target": self.target,
            "original_value": self.original_value,
            "predicted_value": self.predicted_value,
            "change_percent": self.change_percent,
            "confidence_interval": list(self.confidence_interval),
            "impact_type": self.impact_type,
        }


@dataclass
class SimulationResult:
    """시뮬레이션 결과"""
    scenario_id: str
    success: bool
    num_iterations: int = 0

    # 영향 예측
    impacts: List[ImpactPrediction] = field(default_factory=list)

    # 통계
    mean_outcomes: Dict[str, float] = field(default_factory=dict)
    std_outcomes: Dict[str, float] = field(default_factory=dict)

    # 리스크 분석
    risk_score: float = 0.0  # 0-1, 높을수록 위험
    opportunity_score: float = 0.0  # 0-1, 높을수록 기회

    # 권장 사항
    recommendations: List[str] = field(default_factory=list)

    # 메타데이터
    simulated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    execution_time_ms: float = 0.0

    @property
    def impact_summary(self) -> str:
        """영향 요약"""
        positive = sum(1 for i in self.impacts if i.impact_type == "positive")
        negative = sum(1 for i in self.impacts if i.impact_type == "negative")
        neutral = sum(1 for i in self.impacts if i.impact_type == "neutral")

        return (
            f"총 {len(self.impacts)}개 영향 분석: "
            f"긍정 {positive}, 부정 {negative}, 중립 {neutral}"
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "success": self.success,
            "num_iterations": self.num_iterations,
            "impacts": [i.to_dict() for i in self.impacts],
            "mean_outcomes": self.mean_outcomes,
            "std_outcomes": self.std_outcomes,
            "risk_score": self.risk_score,
            "opportunity_score": self.opportunity_score,
            "recommendations": self.recommendations,
            "impact_summary": self.impact_summary,
            "simulated_at": self.simulated_at,
            "execution_time_ms": self.execution_time_ms,
        }


class WhatIfAnalyzer:
    """
    What-If 분석기 (v17.0 - LLM 통합)

    기능:
    - 시나리오 정의 및 관리
    - 인과 그래프 기반 영향 분석
    - Monte Carlo 시뮬레이션
    - 시나리오 비교
    - LLM 기반 권장 사항 및 인사이트 생성
    """

    def __init__(
        self,
        context: Optional["SharedContext"] = None,
        llm_client=None,
    ):
        """
        Args:
            context: SharedContext
            llm_client: LLM 클라이언트 (Anthropic 등)
        """
        self.context = context
        self.llm_client = llm_client

        # 시나리오 저장소
        self.scenarios: Dict[str, Scenario] = {}
        self.results: Dict[str, SimulationResult] = {}

        # 인과 그래프 (FK 관계 기반)
        self.causal_graph: Dict[str, List[str]] = {}

        # 초기화
        self._build_causal_graph()

        logger.info(f"WhatIfAnalyzer initialized (LLM: {'enabled' if llm_client else 'disabled'})")

    def _build_causal_graph(self) -> None:
        """FK 관계 기반 인과 그래프 구축"""
        if not self.context:
            return

        for table_name, table_info in self.context.tables.items():
            # TableInfo dataclass 또는 dict 지원
            if hasattr(table_info, 'foreign_keys'):
                fk_info = table_info.foreign_keys or []
            elif isinstance(table_info, dict):
                fk_info = table_info.get("foreign_keys", [])
            else:
                fk_info = []

            # foreign_keys가 list인 경우 dict로 변환
            if isinstance(fk_info, list):
                fk_dict = {}
                for fk in fk_info:
                    if isinstance(fk, dict):
                        to_table = fk.get("to_table", "")
                        if to_table:
                            fk_dict[to_table] = fk
                fk_info = fk_dict

            for ref_table, fk_details in fk_info.items():
                # 인과 관계: ref_table → table_name
                if ref_table not in self.causal_graph:
                    self.causal_graph[ref_table] = []
                if table_name not in self.causal_graph[ref_table]:
                    self.causal_graph[ref_table].append(table_name)

    def create_scenario(
        self,
        name: str,
        changes: Dict[str, Any],
        description: str = "",
        num_simulations: int = 100,
    ) -> Scenario:
        """
        시나리오 생성

        Args:
            name: 시나리오 이름
            changes: 변경 사항 {target: value 또는 {change_type: value}}
            description: 설명
            num_simulations: 시뮬레이션 횟수

        Returns:
            Scenario: 생성된 시나리오
        """
        scenario_id = f"scenario_{uuid.uuid4().hex[:8]}"

        # 변경 사항 파싱
        parsed_changes = []
        for target, value in changes.items():
            if isinstance(value, dict):
                # {change_type: value} 형식
                for change_type, change_value in value.items():
                    parsed_changes.append(Change(
                        target=target,
                        change_type=ChangeType(change_type),
                        value=change_value,
                    ))
            else:
                # 단순 값 → SET으로 처리
                parsed_changes.append(Change(
                    target=target,
                    change_type=ChangeType.SET,
                    value=value,
                ))

        scenario = Scenario(
            scenario_id=scenario_id,
            name=name,
            description=description,
            changes=parsed_changes,
            num_simulations=num_simulations,
        )

        self.scenarios[scenario_id] = scenario

        # SharedContext에 저장
        if self.context:
            self.context.simulation_scenarios.append(scenario.to_dict())

        logger.info(f"Created scenario: {name} ({scenario_id})")

        return scenario

    async def simulate(
        self,
        scenario: Scenario,
    ) -> SimulationResult:
        """
        시나리오 시뮬레이션

        Args:
            scenario: 시뮬레이션할 시나리오

        Returns:
            SimulationResult: 시뮬레이션 결과
        """
        start_time = datetime.now()

        result = SimulationResult(
            scenario_id=scenario.scenario_id,
            success=False,
        )

        try:
            # 1. 원본 데이터 스냅샷
            original_data = self._snapshot_data()

            # 2. Monte Carlo 시뮬레이션
            outcomes = []
            for _ in range(scenario.num_simulations):
                # 변경 적용 (노이즈 추가)
                modified_data = self._apply_changes_with_noise(
                    original_data, scenario.changes
                )

                # 인과 전파
                propagated_data = self._propagate_causally(modified_data, scenario.changes)

                # 결과 수집
                outcomes.append(propagated_data)

            result.num_iterations = len(outcomes)

            # 3. 통계 분석
            result.mean_outcomes, result.std_outcomes = self._compute_statistics(outcomes)

            # 4. 영향 예측
            result.impacts = self._predict_impacts(
                original_data, result.mean_outcomes, result.std_outcomes
            )

            # 5. 리스크/기회 분석
            result.risk_score, result.opportunity_score = self._analyze_risk_opportunity(
                result.impacts
            )

            # 6. 권장 사항 (v18.0: asyncio.to_thread로 블로킹 호출 방지)
            import asyncio
            result.recommendations = await asyncio.to_thread(
                self._generate_recommendations,
                scenario, result.impacts
            )

            result.success = True

        except Exception as e:
            logger.error(f"Simulation failed: {e}")
            result.recommendations = [f"시뮬레이션 실패: {e}"]

        # 타이밍
        end_time = datetime.now()
        result.execution_time_ms = (end_time - start_time).total_seconds() * 1000

        self.results[scenario.scenario_id] = result

        # v17.1: SharedContext에 결과 저장
        if self.context:
            self.context.record_scenario_result(
                scenario_id=scenario.scenario_id,
                result=result.to_dict(),
                impact_summary=result.impact_summary,
            )

        # Evidence Chain 기록
        self._record_to_evidence_chain(scenario, result)

        return result

    async def compare_scenarios(
        self,
        scenarios: List[Scenario],
    ) -> Dict[str, Any]:
        """
        시나리오 비교

        Args:
            scenarios: 비교할 시나리오 목록

        Returns:
            비교 결과
        """
        results = []

        for scenario in scenarios:
            result = await self.simulate(scenario)
            results.append({
                "scenario": scenario.to_dict(),
                "result": result.to_dict(),
            })

        # 비교 분석
        comparison = {
            "scenarios": [s.name for s in scenarios],
            "results": results,
            "best_opportunity": None,
            "lowest_risk": None,
            "recommendation": "",
        }

        # 최고 기회
        best_opp = max(results, key=lambda r: r["result"]["opportunity_score"])
        comparison["best_opportunity"] = best_opp["scenario"]["name"]

        # 최저 리스크
        lowest_risk = min(results, key=lambda r: r["result"]["risk_score"])
        comparison["lowest_risk"] = lowest_risk["scenario"]["name"]

        # 권장
        if best_opp == lowest_risk:
            comparison["recommendation"] = (
                f"'{best_opp['scenario']['name']}' 시나리오가 "
                f"기회와 리스크 모두에서 최적입니다."
            )
        else:
            comparison["recommendation"] = (
                f"기회 최대화: '{best_opp['scenario']['name']}', "
                f"리스크 최소화: '{lowest_risk['scenario']['name']}'"
            )

        return comparison

    def _safe_deepcopy(self, data: Any) -> Any:
        """
        v28.15: pandas nullable dtype 안전한 deepcopy.

        copy.deepcopy는 pandas nullable dtype (Int64, Float64 등)의 내부
        fractions 구조를 복사할 수 없어서 'numerator' AttributeError 발생.
        JSON serialize/deserialize로 Python native 타입으로 변환.
        """
        try:
            # JSON round-trip으로 pandas 타입 제거 + deepcopy
            return json.loads(json.dumps(data, default=str))
        except (TypeError, ValueError):
            # JSON 직렬화 실패 시 기본 deepcopy 시도
            return copy.deepcopy(data)

    def _snapshot_data(self) -> Dict[str, List[Dict[str, Any]]]:
        """현재 데이터 스냅샷"""
        snapshot = {}

        if not self.context:
            return snapshot

        for table_name, table_info in self.context.tables.items():
            # TableInfo dataclass 또는 dict 지원
            if hasattr(table_info, 'sample_data'):
                sample_data = table_info.sample_data or []
            elif isinstance(table_info, dict):
                sample_data = table_info.get("sample_data", [])
            else:
                sample_data = []
            snapshot[table_name] = self._safe_deepcopy(sample_data)

        return snapshot

    def _apply_changes_with_noise(
        self,
        data: Dict[str, List[Dict[str, Any]]],
        changes: List[Change],
        noise_factor: float = 0.05,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """변경 적용 (노이즈 포함)"""
        modified = self._safe_deepcopy(data)

        for change in changes:
            parts = change.target.split(".")
            if len(parts) != 2:
                continue

            table, column = parts
            if table not in modified:
                continue

            for row in modified[table]:
                if column in row:
                    original = row[column]

                    # 변경 적용
                    new_value = change.apply(original)

                    # 노이즈 추가
                    if isinstance(new_value, (int, float)):
                        noise = random.gauss(0, noise_factor * abs(new_value))
                        new_value += noise

                    row[column] = new_value

        return modified

    def _propagate_causally(
        self,
        data: Dict[str, List[Dict[str, Any]]],
        changes: List[Change],
    ) -> Dict[str, List[Dict[str, Any]]]:
        """인과 그래프 기반 변경 전파"""
        propagated = self._safe_deepcopy(data)

        # 변경된 테이블 식별
        changed_tables = set()
        for change in changes:
            parts = change.target.split(".")
            if len(parts) == 2:
                changed_tables.add(parts[0])

        # BFS로 인과 전파
        visited = set()
        queue = list(changed_tables)

        while queue:
            table = queue.pop(0)
            if table in visited:
                continue
            visited.add(table)

            # 하위 테이블에 영향 전파
            downstream = self.causal_graph.get(table, [])
            for down_table in downstream:
                if down_table not in visited:
                    queue.append(down_table)

                    # 간단한 전파 로직: 상위 테이블 변경 → 하위 테이블에 영향
                    if down_table in propagated:
                        for row in propagated[down_table]:
                            # 랜덤하게 일부 값 조정
                            for key, value in row.items():
                                if isinstance(value, (int, float)):
                                    row[key] = value * random.uniform(0.95, 1.05)

        return propagated

    def _compute_statistics(
        self,
        outcomes: List[Dict[str, List[Dict[str, Any]]]],
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        """통계 계산"""
        import statistics

        means = {}
        stds = {}

        # 각 테이블.컬럼의 평균/표준편차
        all_values: Dict[str, List[float]] = {}

        for outcome in outcomes:
            for table, rows in outcome.items():
                for row in rows:
                    for col, value in row.items():
                        if isinstance(value, (int, float)):
                            key = f"{table}.{col}"
                            if key not in all_values:
                                all_values[key] = []
                            all_values[key].append(value)

        for key, values in all_values.items():
            if len(values) > 1:
                means[key] = statistics.mean(values)
                stds[key] = statistics.stdev(values)
            elif values:
                means[key] = values[0]
                stds[key] = 0.0

        return means, stds

    def _predict_impacts(
        self,
        original: Dict[str, List[Dict[str, Any]]],
        means: Dict[str, float],
        stds: Dict[str, float],
    ) -> List[ImpactPrediction]:
        """영향 예측"""
        impacts = []

        # 원본 평균 계산
        original_means: Dict[str, float] = {}
        for table, rows in original.items():
            for row in rows:
                for col, value in row.items():
                    if isinstance(value, (int, float)):
                        key = f"{table}.{col}"
                        if key not in original_means:
                            original_means[key] = []
                        original_means[key].append(value)

        for key, values in original_means.items():
            original_means[key] = sum(values) / len(values) if values else 0

        # 영향 계산
        for key, predicted_mean in means.items():
            original_mean = original_means.get(key, 0)

            if original_mean == 0:
                change_percent = 0.0
            else:
                change_percent = ((predicted_mean - original_mean) / original_mean) * 100

            std = stds.get(key, 0)
            ci_lower = predicted_mean - 1.96 * std
            ci_upper = predicted_mean + 1.96 * std

            if change_percent > 5:
                impact_type = "positive"
            elif change_percent < -5:
                impact_type = "negative"
            else:
                impact_type = "neutral"

            impacts.append(ImpactPrediction(
                target=key,
                original_value=round(original_mean, 2),
                predicted_value=round(predicted_mean, 2),
                change_percent=round(change_percent, 2),
                confidence_interval=(round(ci_lower, 2), round(ci_upper, 2)),
                impact_type=impact_type,
            ))

        # 변화량 기준 정렬
        impacts.sort(key=lambda x: abs(x.change_percent), reverse=True)

        return impacts[:20]  # 상위 20개

    def _analyze_risk_opportunity(
        self,
        impacts: List[ImpactPrediction],
    ) -> Tuple[float, float]:
        """리스크/기회 분석"""
        if not impacts:
            return 0.5, 0.5

        negative_impacts = [i for i in impacts if i.impact_type == "negative"]
        positive_impacts = [i for i in impacts if i.impact_type == "positive"]

        # 리스크: 부정적 영향의 크기
        risk_score = 0.0
        if negative_impacts:
            risk_score = min(1.0, sum(
                abs(i.change_percent) for i in negative_impacts
            ) / 100)

        # 기회: 긍정적 영향의 크기
        opportunity_score = 0.0
        if positive_impacts:
            opportunity_score = min(1.0, sum(
                abs(i.change_percent) for i in positive_impacts
            ) / 100)

        return round(risk_score, 2), round(opportunity_score, 2)

    def _generate_recommendations(
        self,
        scenario: Scenario,
        impacts: List[ImpactPrediction],
    ) -> List[str]:
        """권장 사항 생성 (LLM 사용 가능 시 LLM 기반)"""
        # LLM 클라이언트가 있으면 LLM 기반 권장 사항 생성
        if self.llm_client:
            try:
                return self._generate_recommendations_with_llm(scenario, impacts)
            except Exception as e:
                logger.warning(f"LLM recommendations failed, falling back: {e}")

        # 기본 규칙 기반 권장 사항
        return self._generate_recommendations_rule_based(scenario, impacts)

    def _generate_recommendations_rule_based(
        self,
        scenario: Scenario,
        impacts: List[ImpactPrediction],
    ) -> List[str]:
        """규칙 기반 권장 사항 생성"""
        recommendations = []

        positive = [i for i in impacts if i.impact_type == "positive"]
        negative = [i for i in impacts if i.impact_type == "negative"]

        if positive:
            top_positive = positive[0]
            recommendations.append(
                f"긍정적 영향: {top_positive.target}이(가) "
                f"{top_positive.change_percent:+.1f}% 변화 예상"
            )

        if negative:
            top_negative = negative[0]
            recommendations.append(
                f"주의 필요: {top_negative.target}이(가) "
                f"{top_negative.change_percent:+.1f}% 변화 예상"
            )

        if len(positive) > len(negative):
            recommendations.append(f"'{scenario.name}' 시나리오는 전반적으로 긍정적 영향이 예상됩니다.")
        elif len(negative) > len(positive):
            recommendations.append(f"'{scenario.name}' 시나리오는 신중한 검토가 필요합니다.")
        else:
            recommendations.append(f"'{scenario.name}' 시나리오의 영향은 균형적입니다.")

        return recommendations

    def _generate_recommendations_with_llm(
        self,
        scenario: Scenario,
        impacts: List[ImpactPrediction],
    ) -> List[str]:
        """LLM 기반 권장 사항 생성 (AI Gateway 사용)"""
        from ..model_config import get_v17_service_model

        # 도메인 컨텍스트 가져오기
        domain = "unknown"
        if self.context:
            domain_ctx = self.context.get_domain()
            domain = domain_ctx.industry if domain_ctx else "unknown"

        # 영향 요약 구성
        impacts_text = "\n".join([
            f"- {i.target}: {i.change_percent:+.1f}% ({i.impact_type})"
            for i in impacts
        ])

        prompt = f"""You are a business analyst providing strategic recommendations.

## Scenario
- Name: {scenario.name}
- Description: {scenario.description or 'N/A'}
- Domain: {domain}

## Changes Applied
{chr(10).join([f'- {c.target}: {c.change_type.value} {c.value}' for c in scenario.changes])}

## Predicted Impacts
{impacts_text}

## Task
Based on the above scenario and predicted impacts, provide 3-5 actionable recommendations in Korean.
Each recommendation should:
1. Be specific and actionable
2. Consider both risks and opportunities
3. Include mitigation strategies for negative impacts

Respond with a JSON array of recommendation strings:
["recommendation 1", "recommendation 2", ...]"""

        try:
            import json as json_module
            from ...common.utils.llm import chat_completion

            # AI Gateway 호출 (모델 자동 선택)
            model = get_v17_service_model("what_if_analyzer")

            response = chat_completion(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
            )

            # 응답 파싱 (chat_completion은 문자열 반환)
            content = response if isinstance(response, str) else str(response)

            # JSON 배열 추출
            start = content.find('[')
            end = content.rfind(']') + 1
            if start >= 0 and end > start:
                recommendations = json_module.loads(content[start:end])
                return recommendations

        except Exception as e:
            logger.warning(f"LLM recommendation generation failed: {e}")

        # 실패 시 규칙 기반으로 폴백
        return self._generate_recommendations_rule_based(scenario, impacts)

    def _record_to_evidence_chain(
        self,
        scenario: Scenario,
        result: SimulationResult,
    ) -> None:
        """Evidence Chain에 기록"""
        if not self.context or not self.context.evidence_chain:
            return

        try:
            from ..autonomous.evidence_chain import EvidenceType

            evidence_type = getattr(
                EvidenceType,
                "WHAT_IF_SIMULATION",
                EvidenceType.QUALITY_ASSESSMENT
            )

            self.context.evidence_chain.add_block(
                phase="what_if",
                agent="WhatIfAnalyzer",
                evidence_type=evidence_type,
                finding=f"Simulated scenario: {scenario.name}",
                reasoning=scenario.description or f"{len(scenario.changes)} changes applied",
                conclusion=result.impact_summary,
                metrics={
                    "scenario_id": scenario.scenario_id,
                    "num_iterations": result.num_iterations,
                    "risk_score": result.risk_score,
                    "opportunity_score": result.opportunity_score,
                },
                confidence=1.0 - result.risk_score,
            )
        except Exception as e:
            logger.warning(f"Failed to record to evidence chain: {e}")

    def get_analyzer_summary(self) -> Dict[str, Any]:
        """분석기 요약"""
        return {
            "total_scenarios": len(self.scenarios),
            "total_simulations": len(self.results),
            "causal_graph_nodes": len(self.causal_graph),
            "recent_scenarios": [
                s.to_dict() for s in list(self.scenarios.values())[-5:]
            ],
        }
