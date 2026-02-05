"""
Decision Explainer (v17.0)

AI 결정 설명 시스템:
- Evidence Chain 기반 결정 추적
- Feature Attribution (SHAP 스타일)
- Counterfactual 생성
- 자연어 설명 생성

사용 예시:
    explainer = DecisionExplainer(context)

    # 결정 설명
    explanation = await explainer.explain_decision(decision_id)
    print(explanation.narrative)

    # 반사실적 설명
    counterfactuals = await explainer.generate_counterfactuals(decision_id)
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, TYPE_CHECKING
from datetime import datetime
from enum import Enum

try:
    import dspy
    DSPY_AVAILABLE = True
except ImportError:
    DSPY_AVAILABLE = False

if TYPE_CHECKING:
    from ..shared_context import SharedContext

logger = logging.getLogger(__name__)


class ExplanationType(str, Enum):
    """설명 유형"""
    FEATURE_ATTRIBUTION = "feature_attribution"
    COUNTERFACTUAL = "counterfactual"
    RULE_BASED = "rule_based"
    EVIDENCE_CHAIN = "evidence_chain"
    NARRATIVE = "narrative"


@dataclass
class FeatureAttribution:
    """Feature Attribution (기여도)"""
    feature: str
    value: Any
    attribution: float  # -1 to 1, 양수면 긍정적 기여
    importance_rank: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "feature": self.feature,
            "value": self.value,
            "attribution": self.attribution,
            "importance_rank": self.importance_rank,
        }


@dataclass
class Counterfactual:
    """반사실적 설명"""
    original_outcome: str
    counterfactual_outcome: str
    changes_required: Dict[str, Dict[str, Any]]  # {feature: {from: x, to: y}}
    feasibility_score: float  # 변경 실현 가능성

    def to_dict(self) -> Dict[str, Any]:
        return {
            "original_outcome": self.original_outcome,
            "counterfactual_outcome": self.counterfactual_outcome,
            "changes_required": self.changes_required,
            "feasibility_score": self.feasibility_score,
        }

    def to_narrative(self) -> str:
        """자연어 설명"""
        changes = []
        for feature, change in self.changes_required.items():
            changes.append(
                f"{feature}를 {change['from']}에서 {change['to']}으로 변경"
            )
        return (
            f"만약 {', '.join(changes)}했다면, "
            f"결과가 '{self.original_outcome}'에서 "
            f"'{self.counterfactual_outcome}'으로 바뀌었을 것입니다."
        )


@dataclass
class Explanation:
    """결정 설명"""
    explanation_id: str
    decision_id: str
    explanation_type: ExplanationType

    # 설명 내용
    summary: str
    narrative: str
    confidence: float

    # Feature Attribution
    feature_attributions: List[FeatureAttribution] = field(default_factory=list)

    # Evidence Chain 기반
    evidence_blocks: List[Dict[str, Any]] = field(default_factory=list)
    reasoning_chain: List[str] = field(default_factory=list)

    # Counterfactuals
    counterfactuals: List[Counterfactual] = field(default_factory=list)

    # 메타데이터
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "explanation_id": self.explanation_id,
            "decision_id": self.decision_id,
            "explanation_type": self.explanation_type.value,
            "summary": self.summary,
            "narrative": self.narrative,
            "confidence": self.confidence,
            "feature_attributions": [f.to_dict() for f in self.feature_attributions],
            "evidence_blocks": self.evidence_blocks,
            "reasoning_chain": self.reasoning_chain,
            "counterfactuals": [c.to_dict() for c in self.counterfactuals],
            "timestamp": self.timestamp,
        }


# DSPy Signatures
if DSPY_AVAILABLE:
    class NarrativeGenerationSignature(dspy.Signature):
        """설명 내러티브 생성"""
        decision_context: str = dspy.InputField(
            desc="Context of the decision being explained"
        )
        evidence_summary: str = dspy.InputField(
            desc="Summary of evidence that led to the decision"
        )
        feature_attributions: str = dspy.InputField(
            desc="Feature attributions and their contributions"
        )

        narrative: str = dspy.OutputField(
            desc="Natural language explanation of the decision"
        )
        confidence: str = dspy.OutputField(
            desc="Confidence in the explanation (0.0-1.0)"
        )


class DecisionExplainer:
    """
    결정 설명 시스템

    기능:
    - Evidence Chain 기반 결정 추적
    - Feature Attribution (SHAP 스타일)
    - Counterfactual 생성
    - 자연어 내러티브 생성
    """

    def __init__(
        self,
        context: Optional["SharedContext"] = None,
    ):
        """
        Args:
            context: SharedContext (Evidence Chain 접근용)
        """
        self.context = context

        # DSPy 모듈
        if DSPY_AVAILABLE:
            self.narrative_generator = dspy.ChainOfThought(NarrativeGenerationSignature)

        # 설명 히스토리
        self.explanations: Dict[str, Explanation] = {}

        logger.info("DecisionExplainer initialized")

    async def explain_decision(
        self,
        decision_id: str,
        include_counterfactuals: bool = False,
    ) -> Explanation:
        """
        결정 설명 생성

        Args:
            decision_id: 설명할 결정의 ID
            include_counterfactuals: 반사실적 설명 포함 여부

        Returns:
            Explanation: 설명 객체
        """
        import uuid

        # Evidence Chain에서 관련 블록 추출
        evidence_blocks = self._extract_evidence_blocks(decision_id)

        # Feature Attribution 계산
        attributions = self._compute_feature_attributions(evidence_blocks)

        # Reasoning Chain 구성
        reasoning_chain = self._build_reasoning_chain(evidence_blocks)

        # 요약 생성
        summary = self._generate_summary(evidence_blocks)

        # 내러티브 생성
        narrative = await self._generate_narrative(
            decision_id, evidence_blocks, attributions
        )

        # Counterfactuals
        counterfactuals = []
        if include_counterfactuals:
            counterfactuals = await self._generate_counterfactuals(
                decision_id, evidence_blocks, attributions
            )

        explanation = Explanation(
            explanation_id=f"exp_{uuid.uuid4().hex[:8]}",
            decision_id=decision_id,
            explanation_type=ExplanationType.EVIDENCE_CHAIN,
            summary=summary,
            narrative=narrative,
            confidence=self._calculate_confidence(evidence_blocks),
            feature_attributions=attributions,
            evidence_blocks=evidence_blocks,
            reasoning_chain=reasoning_chain,
            counterfactuals=counterfactuals,
        )

        self.explanations[explanation.explanation_id] = explanation

        # Evidence Chain에 기록
        self._record_to_evidence_chain(explanation)

        return explanation

    async def generate_counterfactuals(
        self,
        decision_id: str,
        num_counterfactuals: int = 3,
    ) -> List[Counterfactual]:
        """
        반사실적 설명 생성

        Args:
            decision_id: 결정 ID
            num_counterfactuals: 생성할 counterfactual 수

        Returns:
            Counterfactual 목록
        """
        evidence_blocks = self._extract_evidence_blocks(decision_id)
        attributions = self._compute_feature_attributions(evidence_blocks)

        return await self._generate_counterfactuals(
            decision_id, evidence_blocks, attributions, num_counterfactuals
        )

    def _extract_evidence_blocks(
        self,
        decision_id: str,
    ) -> List[Dict[str, Any]]:
        """Evidence Chain에서 관련 블록 추출"""
        if not self.context or not self.context.evidence_chain:
            return []

        blocks = []

        try:
            # Evidence Chain의 블록들을 순회
            chain = self.context.evidence_chain
            for block in getattr(chain, "blocks", []):
                # 관련 블록 필터링 (decision_id 또는 관련 키워드)
                block_dict = block if isinstance(block, dict) else getattr(block, "__dict__", {})

                # 모든 블록을 수집 (실제로는 decision_id로 필터링)
                blocks.append(block_dict)

        except Exception as e:
            logger.warning(f"Failed to extract evidence blocks: {e}")

        return blocks[-10:]  # 최근 10개

    def _compute_feature_attributions(
        self,
        evidence_blocks: List[Dict[str, Any]],
    ) -> List[FeatureAttribution]:
        """Feature Attribution 계산"""
        attributions = []

        # Evidence blocks에서 메트릭 추출
        all_metrics = {}
        for block in evidence_blocks:
            metrics = block.get("metrics", {})
            for key, value in metrics.items():
                if key not in all_metrics:
                    all_metrics[key] = []
                all_metrics[key].append(value)

        # 각 feature의 기여도 계산 (간단한 휴리스틱)
        for i, (feature, values) in enumerate(all_metrics.items()):
            if not values:
                continue

            # 평균 값
            try:
                avg_value = sum(v for v in values if isinstance(v, (int, float))) / len(values)
            except (TypeError, ZeroDivisionError):
                avg_value = values[0] if values else 0

            # 기여도 계산 (confidence 기반)
            attribution = 0.5  # 기본값
            if "confidence" in feature.lower():
                attribution = avg_value - 0.5  # 0.5 기준으로 정규화
            elif "count" in feature.lower() or "total" in feature.lower():
                attribution = min(1.0, avg_value / 100) if isinstance(avg_value, (int, float)) else 0.5

            attributions.append(FeatureAttribution(
                feature=feature,
                value=avg_value,
                attribution=round(attribution, 3),
                importance_rank=i + 1,
            ))

        # 중요도 순 정렬
        attributions.sort(key=lambda x: abs(x.attribution), reverse=True)
        for i, attr in enumerate(attributions):
            attr.importance_rank = i + 1

        return attributions[:10]  # 상위 10개

    def _build_reasoning_chain(
        self,
        evidence_blocks: List[Dict[str, Any]],
    ) -> List[str]:
        """Reasoning Chain 구성"""
        chain = []

        for block in evidence_blocks:
            finding = block.get("finding", "")
            reasoning = block.get("reasoning", "")
            conclusion = block.get("conclusion", "")

            if finding:
                chain.append(f"발견: {finding}")
            if reasoning:
                chain.append(f"추론: {reasoning}")
            if conclusion:
                chain.append(f"결론: {conclusion}")

        return chain

    def _generate_summary(
        self,
        evidence_blocks: List[Dict[str, Any]],
    ) -> str:
        """요약 생성"""
        if not evidence_blocks:
            return "결정에 대한 증거가 없습니다."

        # 마지막 블록의 결론 사용
        last_block = evidence_blocks[-1]
        conclusion = last_block.get("conclusion", "")
        finding = last_block.get("finding", "")

        return conclusion or finding or "결정이 기록되었습니다."

    async def _generate_narrative(
        self,
        decision_id: str,
        evidence_blocks: List[Dict[str, Any]],
        attributions: List[FeatureAttribution],
    ) -> str:
        """자연어 내러티브 생성"""
        if DSPY_AVAILABLE:
            try:
                # 컨텍스트 준비
                decision_context = f"Decision: {decision_id}"
                evidence_summary = "\n".join([
                    f"- {b.get('finding', '')}: {b.get('conclusion', '')}"
                    for b in evidence_blocks
                ])
                attr_summary = "\n".join([
                    f"- {a.feature}: {a.value} (기여도: {a.attribution:+.2f})"
                    for a in attributions[:5]
                ])

                result = self.narrative_generator(
                    decision_context=decision_context,
                    evidence_summary=evidence_summary,
                    feature_attributions=attr_summary,
                )

                return result.narrative

            except Exception as e:
                logger.warning(f"LLM narrative generation failed: {e}")

        # 폴백: 템플릿 기반 생성
        return self._generate_template_narrative(evidence_blocks, attributions)

    def _generate_template_narrative(
        self,
        evidence_blocks: List[Dict[str, Any]],
        attributions: List[FeatureAttribution],
    ) -> str:
        """템플릿 기반 내러티브"""
        parts = []

        # 도입
        parts.append("이 결정은 다음과 같은 근거를 바탕으로 이루어졌습니다.")

        # 주요 발견
        if evidence_blocks:
            parts.append("\n주요 발견:")
            for block in evidence_blocks[-3:]:
                finding = block.get("finding", "")
                if finding:
                    parts.append(f"  • {finding}")

        # 주요 요인
        if attributions:
            parts.append("\n결정에 영향을 미친 주요 요인:")
            for attr in attributions[:3]:
                direction = "긍정적" if attr.attribution > 0 else "부정적"
                parts.append(f"  • {attr.feature}: {direction} 영향 ({attr.attribution:+.2f})")

        # 결론
        if evidence_blocks:
            conclusion = evidence_blocks[-1].get("conclusion", "")
            if conclusion:
                parts.append(f"\n최종 결론: {conclusion}")

        return "\n".join(parts)

    async def _generate_counterfactuals(
        self,
        decision_id: str,
        evidence_blocks: List[Dict[str, Any]],
        attributions: List[FeatureAttribution],
        num_counterfactuals: int = 3,
    ) -> List[Counterfactual]:
        """Counterfactual 생성"""
        counterfactuals = []

        # 현재 결과 추출
        original_outcome = "현재 결정"
        if evidence_blocks:
            original_outcome = evidence_blocks[-1].get("conclusion", "현재 결정")

        # 상위 attribution features를 변경하여 counterfactual 생성
        for i, attr in enumerate(attributions[:num_counterfactuals]):
            if attr.attribution == 0:
                continue

            # 반대 방향 변경 제안
            original_value = attr.value
            if isinstance(original_value, (int, float)):
                if attr.attribution > 0:
                    # 긍정적 기여 → 낮추면 결과 변경
                    proposed_value = original_value * 0.5
                else:
                    # 부정적 기여 → 높이면 결과 변경
                    proposed_value = original_value * 1.5
            else:
                proposed_value = f"NOT {original_value}"

            counterfactuals.append(Counterfactual(
                original_outcome=original_outcome,
                counterfactual_outcome=f"대안적 결과 {i+1}",
                changes_required={
                    attr.feature: {
                        "from": original_value,
                        "to": proposed_value,
                    }
                },
                feasibility_score=0.7 - (i * 0.1),  # 첫 번째가 가장 실현 가능
            ))

        return counterfactuals

    def _calculate_confidence(
        self,
        evidence_blocks: List[Dict[str, Any]],
    ) -> float:
        """설명 신뢰도 계산"""
        if not evidence_blocks:
            return 0.0

        # 평균 confidence
        confidences = []
        for block in evidence_blocks:
            conf = block.get("confidence", 0.5)
            if isinstance(conf, (int, float)):
                confidences.append(conf)

        return sum(confidences) / len(confidences) if confidences else 0.5

    def _record_to_evidence_chain(self, explanation: Explanation) -> None:
        """Evidence Chain에 기록"""
        if not self.context or not self.context.evidence_chain:
            return

        try:
            from ..autonomous.evidence_chain import EvidenceType

            evidence_type = getattr(
                EvidenceType,
                "DECISION_EXPLANATION",
                EvidenceType.QUALITY_ASSESSMENT
            )

            self.context.evidence_chain.add_block(
                phase="explanation",
                agent="DecisionExplainer",
                evidence_type=evidence_type,
                finding=f"Generated explanation for {explanation.decision_id}",
                reasoning=explanation.summary,
                conclusion=explanation.narrative[:200],
                metrics={
                    "explanation_id": explanation.explanation_id,
                    "num_attributions": len(explanation.feature_attributions),
                    "num_counterfactuals": len(explanation.counterfactuals),
                },
                confidence=explanation.confidence,
            )
        except Exception as e:
            logger.warning(f"Failed to record to evidence chain: {e}")

    def get_explanation_summary(self) -> Dict[str, Any]:
        """설명 요약"""
        return {
            "total_explanations": len(self.explanations),
            "recent_explanations": [
                e.to_dict()
                for e in list(self.explanations.values())[-5:]
            ],
        }
