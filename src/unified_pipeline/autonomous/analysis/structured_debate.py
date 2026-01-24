"""
Structured Debate Module - 구조화된 에이전트 토론 시스템

Phase 3 Governance를 위한 라운드 기반 토론 메커니즘입니다.

핵심 원칙:
- LLM-Centric: 알고리즘은 LLM의 판단을 지원하는 도구
- Deep Debate: 단순 투표가 아닌 깊은 논증과 반론
- Round-Based: 라운드별로 입장 정리 → 반론 → 합의 도출
- Evidence-Driven: 데이터 증거 기반 논증

토론 구조:
1. Round 1 - Position Statement: 각 에이전트가 입장 제시
2. Round 2 - Cross-Examination: 상호 질문과 반론
3. Round 3 - Synthesis: 합의점 도출 및 최종 결정

References:
- Multi-Agent Debate (Du et al., 2023)
- Constitutional AI의 토론 메커니즘
- Palantir의 Human-in-the-Loop 철학
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable
from enum import Enum
import logging
import json

logger = logging.getLogger(__name__)


class DebateRole(Enum):
    """토론 역할"""
    PROPOSER = "proposer"           # 제안자 (찬성)
    OPPONENT = "opponent"           # 반대자 (반대/우려)
    ANALYST = "analyst"             # 분석가 (중립/데이터)
    JUDGE = "judge"                 # 심판 (최종 결정)


class DebatePhase(Enum):
    """토론 단계"""
    POSITION_STATEMENT = "position_statement"   # 입장 제시
    CROSS_EXAMINATION = "cross_examination"     # 상호 질문
    REBUTTAL = "rebuttal"                       # 반론
    SYNTHESIS = "synthesis"                     # 합의 도출
    FINAL_DECISION = "final_decision"           # 최종 결정


class VerdictType(Enum):
    """결정 유형"""
    APPROVED = "approved"
    REJECTED = "rejected"
    CONDITIONAL = "conditional"
    ESCALATE = "escalate"
    DEFERRED = "deferred"


@dataclass
class DebateArgument:
    """토론 논증"""
    role: DebateRole
    phase: DebatePhase
    position: str                   # 입장 (support/oppose/neutral)
    argument: str                   # 논증 내용
    evidence: List[Dict[str, Any]]  # 증거 데이터
    confidence: float               # 신뢰도 (0-1)
    counter_arguments: List[str] = field(default_factory=list)  # 반론
    strengths: List[str] = field(default_factory=list)
    weaknesses: List[str] = field(default_factory=list)


@dataclass
class DebateRound:
    """토론 라운드"""
    round_number: int
    phase: DebatePhase
    arguments: List[DebateArgument]
    consensus_level: float          # 합의 수준 (0-1)
    key_points: List[str]
    unresolved_issues: List[str]


@dataclass
class DebateOutcome:
    """토론 결과"""
    topic: str
    verdict: VerdictType
    confidence: float
    rounds: List[DebateRound]
    final_arguments: Dict[str, str]
    conditions: List[str]           # 조건부 승인 시 조건들
    dissenting_views: List[str]     # 소수 의견
    recommendations: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)


class StructuredDebateEngine:
    """
    구조화된 토론 엔진

    다중 에이전트 간의 구조화된 토론을 통해 거버넌스 결정을 도출합니다.
    """

    def __init__(
        self,
        llm_client: Optional[Any] = None,
        max_rounds: int = 3,
        consensus_threshold: float = 0.7,
    ):
        """
        Args:
            llm_client: LLM 클라이언트 (선택)
            max_rounds: 최대 라운드 수
            consensus_threshold: 합의 임계값
        """
        self.llm_client = llm_client
        self.max_rounds = max_rounds
        self.consensus_threshold = consensus_threshold
        self.debate_history: List[DebateOutcome] = []

    async def conduct_debate(
        self,
        topic: str,
        context: Dict[str, Any],
        agents_perspectives: Dict[str, str],
        evidence: Dict[str, Any],
    ) -> DebateOutcome:
        """
        토론 수행

        Args:
            topic: 토론 주제
            context: 컨텍스트 정보
            agents_perspectives: 각 에이전트의 초기 관점
            evidence: 증거 데이터

        Returns:
            토론 결과
        """
        rounds = []
        current_consensus = 0.0

        # Round 1: Position Statement
        round1 = await self._run_position_statement_round(
            topic, context, agents_perspectives, evidence
        )
        rounds.append(round1)
        current_consensus = round1.consensus_level

        # 합의에 도달했으면 조기 종료
        if current_consensus >= self.consensus_threshold:
            return self._create_outcome(
                topic, rounds, evidence, early_consensus=True
            )

        # Round 2: Cross-Examination
        round2 = await self._run_cross_examination_round(
            topic, round1, evidence
        )
        rounds.append(round2)
        current_consensus = round2.consensus_level

        if current_consensus >= self.consensus_threshold:
            return self._create_outcome(
                topic, rounds, evidence, early_consensus=True
            )

        # Round 3: Synthesis
        round3 = await self._run_synthesis_round(
            topic, rounds, evidence
        )
        rounds.append(round3)

        return self._create_outcome(topic, rounds, evidence)

    async def _run_position_statement_round(
        self,
        topic: str,
        context: Dict[str, Any],
        perspectives: Dict[str, str],
        evidence: Dict[str, Any],
    ) -> DebateRound:
        """Round 1: 입장 제시"""
        arguments = []

        # Proposer (찬성)
        proposer_arg = DebateArgument(
            role=DebateRole.PROPOSER,
            phase=DebatePhase.POSITION_STATEMENT,
            position="support",
            argument=self._generate_proposer_argument(topic, context, evidence),
            evidence=self._extract_supporting_evidence(evidence),
            confidence=0.7,
            strengths=self._identify_strengths(evidence),
        )
        arguments.append(proposer_arg)

        # Opponent (반대/우려)
        opponent_arg = DebateArgument(
            role=DebateRole.OPPONENT,
            phase=DebatePhase.POSITION_STATEMENT,
            position="oppose",
            argument=self._generate_opponent_argument(topic, context, evidence),
            evidence=self._extract_counter_evidence(evidence),
            confidence=0.6,
            weaknesses=self._identify_weaknesses(evidence),
        )
        arguments.append(opponent_arg)

        # Analyst (중립/데이터)
        analyst_arg = DebateArgument(
            role=DebateRole.ANALYST,
            phase=DebatePhase.POSITION_STATEMENT,
            position="neutral",
            argument=self._generate_analyst_argument(topic, context, evidence),
            evidence=list(evidence.items())[:5] if isinstance(evidence, dict) else [],
            confidence=0.8,
            strengths=self._identify_strengths(evidence),
            weaknesses=self._identify_weaknesses(evidence),
        )
        arguments.append(analyst_arg)

        # 합의 수준 계산
        consensus = self._calculate_consensus(arguments)

        return DebateRound(
            round_number=1,
            phase=DebatePhase.POSITION_STATEMENT,
            arguments=arguments,
            consensus_level=consensus,
            key_points=self._extract_key_points(arguments),
            unresolved_issues=self._identify_unresolved_issues(arguments),
        )

    async def _run_cross_examination_round(
        self,
        topic: str,
        previous_round: DebateRound,
        evidence: Dict[str, Any],
    ) -> DebateRound:
        """Round 2: 상호 질문 및 반론"""
        arguments = []

        # 이전 라운드의 논점 수집
        prev_arguments = {arg.role: arg for arg in previous_round.arguments}

        # Proposer가 Opponent에게 반론
        proposer_rebuttal = DebateArgument(
            role=DebateRole.PROPOSER,
            phase=DebatePhase.REBUTTAL,
            position="support",
            argument=self._generate_rebuttal(
                prev_arguments.get(DebateRole.OPPONENT),
                "proposer",
                evidence
            ),
            evidence=self._extract_supporting_evidence(evidence),
            confidence=0.65,
            counter_arguments=self._generate_counter_arguments(
                prev_arguments.get(DebateRole.OPPONENT)
            ),
        )
        arguments.append(proposer_rebuttal)

        # Opponent가 Proposer에게 반론
        opponent_rebuttal = DebateArgument(
            role=DebateRole.OPPONENT,
            phase=DebatePhase.REBUTTAL,
            position="oppose",
            argument=self._generate_rebuttal(
                prev_arguments.get(DebateRole.PROPOSER),
                "opponent",
                evidence
            ),
            evidence=self._extract_counter_evidence(evidence),
            confidence=0.55,
            counter_arguments=self._generate_counter_arguments(
                prev_arguments.get(DebateRole.PROPOSER)
            ),
        )
        arguments.append(opponent_rebuttal)

        # Analyst가 양측 분석
        analyst_review = DebateArgument(
            role=DebateRole.ANALYST,
            phase=DebatePhase.CROSS_EXAMINATION,
            position="neutral",
            argument=self._generate_analyst_review(
                proposer_rebuttal,
                opponent_rebuttal,
                evidence
            ),
            evidence=list(evidence.items())[:5] if isinstance(evidence, dict) else [],
            confidence=0.75,
        )
        arguments.append(analyst_review)

        consensus = self._calculate_consensus(arguments)

        return DebateRound(
            round_number=2,
            phase=DebatePhase.CROSS_EXAMINATION,
            arguments=arguments,
            consensus_level=consensus,
            key_points=self._extract_key_points(arguments),
            unresolved_issues=self._identify_unresolved_issues(arguments),
        )

    async def _run_synthesis_round(
        self,
        topic: str,
        previous_rounds: List[DebateRound],
        evidence: Dict[str, Any],
    ) -> DebateRound:
        """Round 3: 합의 도출"""
        arguments = []

        # 모든 이전 논점 수집
        all_key_points = []
        all_issues = []
        for round_ in previous_rounds:
            all_key_points.extend(round_.key_points)
            all_issues.extend(round_.unresolved_issues)

        # Judge의 최종 분석
        judge_synthesis = DebateArgument(
            role=DebateRole.JUDGE,
            phase=DebatePhase.SYNTHESIS,
            position="neutral",
            argument=self._generate_judge_synthesis(
                all_key_points,
                all_issues,
                evidence
            ),
            evidence=list(evidence.items())[:5] if isinstance(evidence, dict) else [],
            confidence=0.85,
            strengths=list(set(all_key_points[:3])),
            weaknesses=list(set(all_issues[:3])),
        )
        arguments.append(judge_synthesis)

        # 각 역할의 최종 입장
        final_positions = self._gather_final_positions(previous_rounds)

        # 합의 수준 계산 (Judge 중심)
        consensus = judge_synthesis.confidence

        return DebateRound(
            round_number=3,
            phase=DebatePhase.SYNTHESIS,
            arguments=arguments,
            consensus_level=consensus,
            key_points=all_key_points[:5],
            unresolved_issues=all_issues[:3],
        )

    def _generate_proposer_argument(
        self,
        topic: str,
        context: Dict[str, Any],
        evidence: Dict[str, Any],
    ) -> str:
        """찬성 논증 생성"""
        strengths = self._identify_strengths(evidence)
        return (
            f"I support the approval of {topic}. "
            f"Key strengths: {'; '.join(strengths[:3])}. "
            f"The evidence shows positive indicators with acceptable quality metrics."
        )

    def _generate_opponent_argument(
        self,
        topic: str,
        context: Dict[str, Any],
        evidence: Dict[str, Any],
    ) -> str:
        """반대/우려 논증 생성"""
        weaknesses = self._identify_weaknesses(evidence)
        return (
            f"I have concerns about {topic}. "
            f"Key risks: {'; '.join(weaknesses[:3])}. "
            f"Further validation may be needed before full approval."
        )

    def _generate_analyst_argument(
        self,
        topic: str,
        context: Dict[str, Any],
        evidence: Dict[str, Any],
    ) -> str:
        """분석가 논증 생성"""
        metrics = self._extract_metrics(evidence)
        return (
            f"Data analysis for {topic}: "
            f"Metrics summary - {json.dumps(metrics, default=str)[:200]}. "
            f"Recommend data-driven decision based on these indicators."
        )

    def _generate_rebuttal(
        self,
        target_argument: Optional[DebateArgument],
        rebuttal_role: str,
        evidence: Dict[str, Any],
    ) -> str:
        """반론 생성"""
        if not target_argument:
            return f"{rebuttal_role} has no specific target to rebut."

        if rebuttal_role == "proposer":
            return (
                f"Addressing concerns raised: {target_argument.argument[:100]}... "
                f"The weaknesses identified can be mitigated through iterative improvement. "
                f"Initial acceptance allows for progressive refinement."
            )
        else:
            return (
                f"Challenging claims: {target_argument.argument[:100]}... "
                f"The strengths mentioned need to be weighed against potential risks. "
                f"Quality thresholds should be met before full adoption."
            )

    def _generate_analyst_review(
        self,
        proposer_arg: DebateArgument,
        opponent_arg: DebateArgument,
        evidence: Dict[str, Any],
    ) -> str:
        """분석가 검토 생성"""
        return (
            f"Cross-examination analysis: "
            f"Proposer confidence: {proposer_arg.confidence:.2f}, "
            f"Opponent confidence: {opponent_arg.confidence:.2f}. "
            f"Both perspectives have valid points. "
            f"Recommend conditional approval with monitoring."
        )

    def _generate_judge_synthesis(
        self,
        key_points: List[str],
        issues: List[str],
        evidence: Dict[str, Any],
    ) -> str:
        """심판 합의 생성"""
        return (
            f"Synthesis of debate: "
            f"Key agreements: {'; '.join(key_points[:3])}. "
            f"Remaining concerns: {'; '.join(issues[:2])}. "
            f"Final recommendation: Conditional approval with defined review cycle."
        )

    def _generate_counter_arguments(
        self,
        target: Optional[DebateArgument],
    ) -> List[str]:
        """반론 목록 생성"""
        if not target:
            return []

        counters = []
        if target.position == "support":
            counters = [
                "Confidence levels may be inflated",
                "Edge cases not fully addressed",
                "Long-term sustainability unclear",
            ]
        else:
            counters = [
                "Risks are manageable with proper governance",
                "Iterative improvement is preferable to rejection",
                "Waiting for perfection delays value delivery",
            ]

        return counters

    def _calculate_consensus(self, arguments: List[DebateArgument]) -> float:
        """합의 수준 계산"""
        if not arguments:
            return 0.5

        # 역할별 가중치
        role_weights = {
            DebateRole.PROPOSER: 1.0,
            DebateRole.OPPONENT: 1.0,
            DebateRole.ANALYST: 1.2,  # 데이터 분석가에게 더 높은 가중치
            DebateRole.JUDGE: 1.5,
        }

        # 입장별 점수
        position_scores = {
            "support": 1.0,
            "neutral": 0.5,
            "oppose": 0.0,
        }

        total_weight = 0.0
        weighted_score = 0.0

        for arg in arguments:
            weight = role_weights.get(arg.role, 1.0) * arg.confidence
            score = position_scores.get(arg.position, 0.5)
            weighted_score += weight * score
            total_weight += weight

        if total_weight == 0:
            return 0.5

        return weighted_score / total_weight

    def _extract_supporting_evidence(self, evidence: Dict[str, Any]) -> List[Dict[str, Any]]:
        """긍정적 증거 추출"""
        supporting = []

        if isinstance(evidence, dict):
            # 품질 점수가 높은 항목
            quality = evidence.get("quality_score", evidence.get("confidence", 0))
            if quality > 0.5:
                supporting.append({"type": "quality", "value": quality})

            # 데이터 완전성
            completeness = evidence.get("completeness", 0)
            if completeness > 0.6:
                supporting.append({"type": "completeness", "value": completeness})

            # 비즈니스 가치
            business_value = evidence.get("business_value", 0)
            if business_value > 0.5:
                supporting.append({"type": "business_value", "value": business_value})

        return supporting

    def _extract_counter_evidence(self, evidence: Dict[str, Any]) -> List[Dict[str, Any]]:
        """부정적/우려 증거 추출"""
        counter = []

        if isinstance(evidence, dict):
            # 낮은 품질 점수
            quality = evidence.get("quality_score", evidence.get("confidence", 1))
            if quality < 0.7:
                counter.append({"type": "quality_concern", "value": 1 - quality})

            # 리스크
            risk = evidence.get("risk_level", evidence.get("risk_score", 0))
            if risk > 0.3:
                counter.append({"type": "risk", "value": risk})

            # 미해결 이슈
            issues = evidence.get("issues", evidence.get("unresolved", []))
            if issues:
                counter.append({"type": "unresolved_issues", "count": len(issues)})

        return counter

    def _identify_strengths(self, evidence: Dict[str, Any]) -> List[str]:
        """강점 식별"""
        strengths = []

        if isinstance(evidence, dict):
            if evidence.get("quality_score", 0) > 0.6:
                strengths.append("Good data quality metrics")
            if evidence.get("confidence", 0) > 0.7:
                strengths.append("High confidence from analysis")
            if evidence.get("data_sources", []):
                strengths.append(f"Multiple data sources: {len(evidence.get('data_sources', []))}")
            if evidence.get("business_alignment", False):
                strengths.append("Aligned with business objectives")

        return strengths or ["Data presence indicates potential value"]

    def _identify_weaknesses(self, evidence: Dict[str, Any]) -> List[str]:
        """약점 식별"""
        weaknesses = []

        if isinstance(evidence, dict):
            if evidence.get("quality_score", 1) < 0.5:
                weaknesses.append("Quality metrics below threshold")
            if evidence.get("completeness", 1) < 0.7:
                weaknesses.append("Incomplete data coverage")
            if evidence.get("risk_level", 0) > 0.5:
                weaknesses.append("Elevated risk indicators")
            issues = evidence.get("issues", [])
            if issues:
                weaknesses.append(f"Unresolved issues: {len(issues)}")

        return weaknesses or ["Minor improvements recommended"]

    def _extract_metrics(self, evidence: Dict[str, Any]) -> Dict[str, Any]:
        """메트릭 추출"""
        metrics = {}

        if isinstance(evidence, dict):
            for key in ["quality_score", "confidence", "completeness", "risk_level"]:
                if key in evidence:
                    metrics[key] = evidence[key]

        return metrics

    def _extract_key_points(self, arguments: List[DebateArgument]) -> List[str]:
        """핵심 논점 추출"""
        key_points = []

        for arg in arguments:
            key_points.extend(arg.strengths[:1])
            if arg.position == "support":
                key_points.append(f"{arg.role.value}: Supports approval")
            elif arg.position == "oppose":
                key_points.append(f"{arg.role.value}: Raises concerns")

        return list(set(key_points))[:5]

    def _identify_unresolved_issues(self, arguments: List[DebateArgument]) -> List[str]:
        """미해결 이슈 식별"""
        issues = []

        for arg in arguments:
            issues.extend(arg.weaknesses[:1])
            issues.extend(arg.counter_arguments[:1])

        return list(set(issues))[:3]

    def _gather_final_positions(
        self,
        rounds: List[DebateRound],
    ) -> Dict[str, str]:
        """최종 입장 수집"""
        positions = {}

        for round_ in rounds:
            for arg in round_.arguments:
                positions[arg.role.value] = arg.position

        return positions

    def _create_outcome(
        self,
        topic: str,
        rounds: List[DebateRound],
        evidence: Dict[str, Any],
        early_consensus: bool = False,
    ) -> DebateOutcome:
        """토론 결과 생성"""
        # 최종 합의 수준
        final_consensus = rounds[-1].consensus_level if rounds else 0.5

        # 결정 유형 결정
        if final_consensus >= 0.75:
            verdict = VerdictType.APPROVED
        elif final_consensus >= 0.55:
            verdict = VerdictType.CONDITIONAL
        elif final_consensus >= 0.35:
            verdict = VerdictType.ESCALATE
        else:
            verdict = VerdictType.DEFERRED

        # 최종 논증 수집
        final_arguments = {}
        if rounds:
            for arg in rounds[-1].arguments:
                final_arguments[arg.role.value] = arg.argument[:200]

        # 조건 (조건부 승인 시)
        conditions = []
        if verdict == VerdictType.CONDITIONAL:
            conditions = [
                "Implement monitoring for identified risks",
                "Schedule review in 30 days",
                "Address high-priority issues",
            ]

        # 소수 의견
        dissenting = []
        for round_ in rounds:
            for arg in round_.arguments:
                if arg.position == "oppose" and arg.confidence > 0.5:
                    dissenting.append(f"{arg.role.value}: {arg.argument[:100]}")

        # 권장사항
        recommendations = [
            "Continue iterative improvement",
            "Monitor quality metrics",
            "Engage stakeholders for feedback",
        ]

        outcome = DebateOutcome(
            topic=topic,
            verdict=verdict,
            confidence=final_consensus,
            rounds=rounds,
            final_arguments=final_arguments,
            conditions=conditions,
            dissenting_views=dissenting[:2],
            recommendations=recommendations,
            metadata={
                "total_rounds": len(rounds),
                "early_consensus": early_consensus,
            },
        )

        self.debate_history.append(outcome)
        return outcome


def create_debate_summary(outcome: DebateOutcome) -> Dict[str, Any]:
    """토론 결과 요약 생성"""
    return {
        "topic": outcome.topic,
        "verdict": outcome.verdict.value,
        "confidence": outcome.confidence,
        "total_rounds": len(outcome.rounds),
        "conditions": outcome.conditions,
        "dissenting_views_count": len(outcome.dissenting_views),
        "recommendations": outcome.recommendations[:3],
        "final_positions": outcome.final_arguments,
    }
