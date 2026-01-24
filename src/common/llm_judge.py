"""
LLM-as-Judge: Multi-Agent Parallel Evaluation System

Implements a robust evaluation system using 4 specialized LLM judges
running in parallel to assess analysis quality from multiple perspectives.

Judge Council Pattern:
1. AccuracyJudge - 정확성: 수치/사실 정확도 평가
2. ActionabilityJudge - 실행가능성: 권고사항의 실행 가능 여부 평가
3. BusinessValueJudge - 비즈니스 가치: 인사이트의 사업적 가치 평가
4. EvidenceJudge - 증거 품질: 인사이트 뒷받침 근거의 충분성 평가

각 Judge는 독립적으로 평가하고, 결과는 앙상블로 통합됩니다.
"""

import json
import logging
import concurrent.futures
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass, field, asdict
from enum import Enum
from datetime import datetime

from .utils.llm import (
    chat_completion_with_json,
    get_openai_client,
    extract_json_from_response,
    DEFAULT_MODEL,
)

logger = logging.getLogger(__name__)

# Instructor 통합 (선택적 - 가용 시 구조화된 출력 사용)
_INSTRUCTOR_AVAILABLE = False
try:
    from .utils.llm_instructor import call_llm_structured
    from src.unified_pipeline.llm_schemas import JudgeEvaluationResult
    _INSTRUCTOR_AVAILABLE = True
    logger.info("Instructor integration available for LLM Judge")
except ImportError as e:
    logger.debug(f"Instructor not available for LLM Judge: {e}")


class JudgeRole(Enum):
    """평가자 역할"""
    ACCURACY = "accuracy"           # 정확성 평가
    ACTIONABILITY = "actionability" # 실행가능성 평가
    BUSINESS_VALUE = "business_value"  # 비즈니스 가치 평가
    EVIDENCE = "evidence"           # 증거 품질 평가


@dataclass
class JudgeScore:
    """개별 Judge의 평가 결과"""
    judge_role: str
    score: float  # 0.0 - 1.0
    reasoning: str
    strengths: List[str]
    weaknesses: List[str]
    improvement_suggestions: List[str]
    confidence: float  # Judge 자신의 평가 확신도


@dataclass
class EnsembleJudgment:
    """앙상블 최종 판정"""
    overall_score: float
    individual_scores: Dict[str, float]
    weighted_scores: Dict[str, float]
    consensus_level: float  # Judge 간 합의 수준
    final_verdict: str  # "excellent" | "good" | "acceptable" | "needs_improvement" | "poor"
    key_strengths: List[str]
    critical_weaknesses: List[str]
    priority_improvements: List[str]
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


def _safe_float(value: Any, default: float = 0.5) -> float:
    """안전한 float 변환 - LLM 응답에서 수치 파싱 시 사용"""
    if value is None:
        return default
    try:
        result = float(value)
        # 0-1 범위 강제
        return max(0.0, min(1.0, result))
    except (ValueError, TypeError):
        return default


class BaseJudge:
    """평가자 기본 클래스"""

    def __init__(self, role: JudgeRole, model: str = None):
        self.role = role
        self.model = model or DEFAULT_MODEL
        self.client = get_openai_client()

    @property
    def persona(self) -> str:
        """Judge 페르소나"""
        raise NotImplementedError

    @property
    def evaluation_criteria(self) -> str:
        """평가 기준"""
        raise NotImplementedError

    def evaluate(self, insights: List[Dict], context: Dict[str, Any]) -> JudgeScore:
        """인사이트 평가"""
        raise NotImplementedError

    def _call_llm(self, prompt: str, system_prompt: str = None) -> Dict[str, Any]:
        """LLM 호출 - Instructor 가용 시 구조화된 출력 사용"""
        try:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})

            # Instructor 사용 (가용 시)
            if _INSTRUCTOR_AVAILABLE:
                try:
                    result = call_llm_structured(
                        response_model=JudgeEvaluationResult,
                        messages=messages,
                        model=self.model,
                        temperature=0.2,
                        max_retries=2,
                    )
                    return {
                        "score": result.score,
                        "reasoning": result.reasoning,
                        "strengths": result.strengths,
                        "weaknesses": result.weaknesses,
                        "improvement_suggestions": result.improvement_suggestions,
                        "confidence": result.confidence,
                    }
                except Exception as e:
                    logger.warning(f"Instructor call failed, falling back to regex: {e}")

            # Fallback: 기존 방식 (regex 기반 JSON 파싱)
            response = chat_completion_with_json(
                model=self.model,
                messages=messages,
                temperature=0.2,
                client=self.client,
            )
            return extract_json_from_response(response)
        except Exception as e:
            logger.error(f"LLM call failed in {self.role.value}: {e}")
            return {}


class AccuracyJudge(BaseJudge):
    """
    정확성 평가 Judge

    평가 항목:
    - 수치의 정확성 (실제 데이터와 일치 여부)
    - 통계 계산의 정확성
    - 사실 진술의 정확성
    - 인과관계 주장의 타당성
    """

    def __init__(self, model: str = None):
        super().__init__(JudgeRole.ACCURACY, model)

    @property
    def persona(self) -> str:
        return """당신은 **정확성 평가 전문가**입니다.

데이터 분석 인사이트의 정확성을 엄격하게 평가합니다:
- 인용된 수치가 실제 데이터와 일치하는가?
- 통계 계산(평균, 백분율, 상관계수)이 정확한가?
- 사실 진술이 데이터에 의해 뒷받침되는가?
- 인과관계 주장이 논리적으로 타당한가?

엄격하지만 공정하게 평가하세요. 수치적 오류는 무관용 원칙으로 처리합니다."""

    @property
    def evaluation_criteria(self) -> str:
        return """
## 정확성 평가 기준 (Accuracy Criteria)

### 1. 수치 정확성 (40%)
- 인용된 모든 수치가 정확한가?
- 백분율, 비율, 평균값이 올바르게 계산되었는가?
- 반올림/표기가 적절한가?

### 2. 통계 계산 (25%)
- 상관계수, 표준편차 등 통계치가 정확한가?
- IQR, 백분위수 계산이 올바른가?
- 샘플 크기가 적절히 고려되었는가?

### 3. 사실 진술 (20%)
- 데이터에서 직접 추론 가능한 사실만 진술했는가?
- 과장이나 축소 없이 객관적인가?

### 4. 논리적 일관성 (15%)
- 인사이트 간 모순이 없는가?
- 인과관계 추론이 논리적인가?
"""

    def evaluate(self, insights: List[Dict], context: Dict[str, Any]) -> JudgeScore:
        """정확성 평가"""

        insights_summary = self._summarize_insights(insights)
        data_context = json.dumps(context.get("data_summary", {}), ensure_ascii=False, indent=2)

        prompt = f"""{self.evaluation_criteria}

## 평가 대상 인사이트
{insights_summary}

## 데이터 컨텍스트 (검증용)
{data_context[:3000]}

## 평가 요청
위 인사이트들의 정확성을 엄격히 평가하세요.

JSON 형식으로 반환:
{{
    "score": 0.0-1.0,
    "reasoning": "평가 근거 설명 (한글)",
    "strengths": ["정확한 부분들"],
    "weaknesses": ["부정확한 부분들"],
    "improvement_suggestions": ["개선 방법들"],
    "confidence": 0.0-1.0
}}
"""

        result = self._call_llm(prompt, self.persona)

        return JudgeScore(
            judge_role=self.role.value,
            score=_safe_float(result.get("score"), 0.5),
            reasoning=result.get("reasoning", "평가 실패"),
            strengths=result.get("strengths", []),
            weaknesses=result.get("weaknesses", []),
            improvement_suggestions=result.get("improvement_suggestions", []),
            confidence=_safe_float(result.get("confidence"), 0.5)
        )

    def _summarize_insights(self, insights: List[Dict]) -> str:
        """인사이트 요약"""
        lines = []
        for i, insight in enumerate(insights[:15], 1):
            title = insight.get("title", insight.get("explanation", "")[:50])
            metrics = insight.get("metrics", {})
            lines.append(f"{i}. {title}")
            if metrics:
                lines.append(f"   수치: {json.dumps(metrics, ensure_ascii=False)}")
        return "\n".join(lines)


class ActionabilityJudge(BaseJudge):
    """
    실행가능성 평가 Judge

    평가 항목:
    - 권고사항의 구체성
    - 실행 가능성 (현실적인가?)
    - 우선순위 명확성
    - 담당자 지정 가능성
    """

    def __init__(self, model: str = None):
        super().__init__(JudgeRole.ACTIONABILITY, model)

    @property
    def persona(self) -> str:
        return """당신은 **실행가능성 평가 전문가**입니다.

컨설팅 및 운영 전문가로서, 인사이트가 실제로 실행 가능한지 평가합니다:
- 권고사항이 구체적이고 명확한가?
- 현실적으로 실행 가능한가?
- 우선순위가 명확한가?
- 담당자가 즉시 행동할 수 있는가?

"그래서 뭘 하라는 거야?"라는 질문에 명확히 답할 수 있어야 합니다."""

    @property
    def evaluation_criteria(self) -> str:
        return """
## 실행가능성 평가 기준 (Actionability Criteria)

### 1. 구체성 (35%)
- 권고사항이 구체적인가? (누가, 무엇을, 어떻게)
- 모호한 표현 없이 명확한가?
- 측정 가능한 목표가 있는가?

### 2. 현실성 (30%)
- 현재 리소스로 실행 가능한가?
- 기술적/조직적 제약을 고려했는가?
- 비용 대비 효과가 합리적인가?

### 3. 우선순위 (20%)
- 긴급도와 중요도가 구분되는가?
- 선후관계가 명확한가?
- 리스크 기반 우선순위인가?

### 4. 책임 소재 (15%)
- 담당 부서/역할이 명확한가?
- 즉시 실행 개시 가능한가?
"""

    def evaluate(self, insights: List[Dict], context: Dict[str, Any]) -> JudgeScore:
        """실행가능성 평가"""

        # 권고사항 추출
        recommendations = []
        for insight in insights[:15]:
            rec = insight.get("recommendation", insight.get("recommended_action", ""))
            if rec:
                recommendations.append({
                    "insight": insight.get("title", insight.get("explanation", "")[:50]),
                    "recommendation": rec,
                    "priority": insight.get("severity", insight.get("action_priority", "medium"))
                })

        prompt = f"""{self.evaluation_criteria}

## 평가 대상 권고사항
{json.dumps(recommendations, ensure_ascii=False, indent=2)}

## 조직 컨텍스트
산업: {context.get("domain_context", {}).get("industry", "unknown")}
부서: {context.get("domain_context", {}).get("department", "unknown")}

## 평가 요청
위 권고사항들의 실행가능성을 평가하세요.

JSON 형식으로 반환:
{{
    "score": 0.0-1.0,
    "reasoning": "평가 근거 설명 (한글)",
    "strengths": ["실행 가능한 부분들"],
    "weaknesses": ["실행 어려운 부분들"],
    "improvement_suggestions": ["더 실행 가능하게 만드는 방법들"],
    "confidence": 0.0-1.0
}}
"""

        result = self._call_llm(prompt, self.persona)

        return JudgeScore(
            judge_role=self.role.value,
            score=_safe_float(result.get("score"), 0.5),
            reasoning=result.get("reasoning", "평가 실패"),
            strengths=result.get("strengths", []),
            weaknesses=result.get("weaknesses", []),
            improvement_suggestions=result.get("improvement_suggestions", []),
            confidence=_safe_float(result.get("confidence"), 0.5)
        )


class BusinessValueJudge(BaseJudge):
    """
    비즈니스 가치 평가 Judge

    평가 항목:
    - KPI 영향도
    - 비용 절감 / 수익 증대 잠재력
    - 전략적 가치
    - 경쟁 우위 기여도
    """

    def __init__(self, model: str = None):
        super().__init__(JudgeRole.BUSINESS_VALUE, model)

    @property
    def persona(self) -> str:
        return """당신은 **비즈니스 가치 평가 전문가**입니다.

경영 컨설턴트로서, 인사이트의 비즈니스 가치를 평가합니다:
- 이 인사이트가 수익에 어떤 영향을 미치는가?
- 비용 절감 효과가 있는가?
- 전략적으로 중요한가?
- 경쟁 우위를 제공하는가?

CFO나 CEO에게 보고할 가치가 있는지 판단하세요."""

    @property
    def evaluation_criteria(self) -> str:
        return """
## 비즈니스 가치 평가 기준 (Business Value Criteria)

### 1. 재무적 영향 (35%)
- 수익 증대 잠재력이 있는가?
- 비용 절감 효과가 있는가?
- ROI가 합리적인가?

### 2. 전략적 가치 (30%)
- 핵심 비즈니스 목표에 부합하는가?
- 장기적 경쟁력에 기여하는가?
- 시장 포지셔닝에 영향이 있는가?

### 3. 운영 효율성 (20%)
- 프로세스 개선 효과가 있는가?
- 생산성 향상에 기여하는가?
- 리스크 감소에 도움이 되는가?

### 4. 의사결정 지원 (15%)
- 경영진 의사결정에 도움이 되는가?
- 새로운 기회를 발견했는가?
- 데이터 기반 문화에 기여하는가?
"""

    def evaluate(self, insights: List[Dict], context: Dict[str, Any]) -> JudgeScore:
        """비즈니스 가치 평가"""

        # KPI 영향 추출
        kpi_impacts = []
        for insight in insights[:15]:
            impact = insight.get("kpi_impact", insight.get("estimated_kpi_impact", {}))
            if impact:
                kpi_impacts.append({
                    "insight": insight.get("title", insight.get("explanation", "")[:50]),
                    "kpi_impact": impact,
                    "severity": insight.get("severity", "medium")
                })

        domain = context.get("domain_context", {})

        prompt = f"""{self.evaluation_criteria}

## 평가 대상 인사이트와 KPI 영향
{json.dumps(kpi_impacts, ensure_ascii=False, indent=2)}

## 비즈니스 컨텍스트
산업: {domain.get("industry", "unknown")}
비즈니스 단계: {domain.get("business_stage", "unknown")}
핵심 KPI: {domain.get("key_metrics", [])}

## 평가 요청
위 인사이트들의 비즈니스 가치를 평가하세요.

JSON 형식으로 반환:
{{
    "score": 0.0-1.0,
    "reasoning": "평가 근거 설명 (한글)",
    "strengths": ["가치 높은 부분들"],
    "weaknesses": ["가치 낮은 부분들"],
    "improvement_suggestions": ["가치를 높이는 방법들"],
    "confidence": 0.0-1.0
}}
"""

        result = self._call_llm(prompt, self.persona)

        return JudgeScore(
            judge_role=self.role.value,
            score=_safe_float(result.get("score"), 0.5),
            reasoning=result.get("reasoning", "평가 실패"),
            strengths=result.get("strengths", []),
            weaknesses=result.get("weaknesses", []),
            improvement_suggestions=result.get("improvement_suggestions", []),
            confidence=_safe_float(result.get("confidence"), 0.5)
        )


class EvidenceJudge(BaseJudge):
    """
    증거 품질 평가 Judge

    평가 항목:
    - 증거의 충분성
    - 데이터 출처 명시성
    - 샘플 크기 적절성
    - 통계적 유의성
    """

    def __init__(self, model: str = None):
        super().__init__(JudgeRole.EVIDENCE, model)

    @property
    def persona(self) -> str:
        return """당신은 **증거 품질 평가 전문가**입니다.

과학적 연구 방법론 전문가로서, 인사이트의 증거 품질을 평가합니다:
- 주장을 뒷받침하는 충분한 증거가 있는가?
- 데이터 출처가 명확한가?
- 샘플 크기가 통계적으로 유의한가?
- 편향 없이 데이터를 분석했는가?

"이 주장을 믿을 수 있는가?"라는 질문에 답하세요."""

    @property
    def evaluation_criteria(self) -> str:
        return """
## 증거 품질 평가 기준 (Evidence Quality Criteria)

### 1. 증거 충분성 (35%)
- 각 주장에 데이터 증거가 있는가?
- 증거가 주장을 충분히 뒷받침하는가?
- 반례나 예외가 고려되었는가?

### 2. 출처 명시성 (25%)
- 데이터 출처(테이블, 컬럼)가 명확한가?
- 데이터 시점/기간이 명시되었는가?
- 데이터 수집 방법이 투명한가?

### 3. 통계적 적절성 (25%)
- 샘플 크기가 충분한가?
- 신뢰구간/유의수준이 고려되었는가?
- 이상치가 적절히 처리되었는가?

### 4. 객관성 (15%)
- 확증 편향 없이 분석했는가?
- 상반된 증거도 고려했는가?
- 한계점을 인정했는가?
"""

    def evaluate(self, insights: List[Dict], context: Dict[str, Any]) -> JudgeScore:
        """증거 품질 평가"""

        # 증거 추출
        evidence_list = []
        for insight in insights[:15]:
            evidence_list.append({
                "insight": insight.get("title", insight.get("explanation", "")[:50]),
                "evidence": insight.get("evidence", []),
                "affected_tables": insight.get("affected_tables", []),
                "metrics": insight.get("metrics", {}),
                "confidence": insight.get("confidence", 0.5)
            })

        prompt = f"""{self.evaluation_criteria}

## 평가 대상 인사이트와 증거
{json.dumps(evidence_list, ensure_ascii=False, indent=2)}

## 데이터 컨텍스트
총 테이블 수: {len(context.get("tables", []))}
평균 행 수: {context.get("avg_row_count", "unknown")}

## 평가 요청
위 인사이트들의 증거 품질을 평가하세요.

JSON 형식으로 반환:
{{
    "score": 0.0-1.0,
    "reasoning": "평가 근거 설명 (한글)",
    "strengths": ["증거가 충분한 부분들"],
    "weaknesses": ["증거가 부족한 부분들"],
    "improvement_suggestions": ["증거를 강화하는 방법들"],
    "confidence": 0.0-1.0
}}
"""

        result = self._call_llm(prompt, self.persona)

        return JudgeScore(
            judge_role=self.role.value,
            score=_safe_float(result.get("score"), 0.5),
            reasoning=result.get("reasoning", "평가 실패"),
            strengths=result.get("strengths", []),
            weaknesses=result.get("weaknesses", []),
            improvement_suggestions=result.get("improvement_suggestions", []),
            confidence=_safe_float(result.get("confidence"), 0.5)
        )


class LLMJudgeOrchestrator:
    """
    LLM-as-Judge 오케스트레이터

    4개의 전문 Judge를 병렬로 실행하고 결과를 앙상블합니다.
    """

    # 각 Judge의 가중치 (총합 = 1.0)
    WEIGHTS = {
        JudgeRole.ACCURACY.value: 0.30,      # 정확성 30%
        JudgeRole.ACTIONABILITY.value: 0.25, # 실행가능성 25%
        JudgeRole.BUSINESS_VALUE.value: 0.25, # 비즈니스 가치 25%
        JudgeRole.EVIDENCE.value: 0.20,      # 증거 품질 20%
    }

    def __init__(self, model: str = None, parallel: bool = True, max_workers: int = 4):
        self.model = model or DEFAULT_MODEL
        self.parallel = parallel
        self.max_workers = max_workers

        # Judge 초기화
        self.judges = {
            JudgeRole.ACCURACY: AccuracyJudge(model),
            JudgeRole.ACTIONABILITY: ActionabilityJudge(model),
            JudgeRole.BUSINESS_VALUE: BusinessValueJudge(model),
            JudgeRole.EVIDENCE: EvidenceJudge(model),
        }

        logger.info(f"LLMJudgeOrchestrator initialized with {len(self.judges)} judges, parallel={parallel}")

    def evaluate(
        self,
        insights: List[Dict],
        context: Dict[str, Any]
    ) -> EnsembleJudgment:
        """
        전체 평가 수행

        Args:
            insights: 평가할 인사이트 목록
            context: 평가 컨텍스트 (domain_context, tables 등)

        Returns:
            EnsembleJudgment: 앙상블 판정 결과
        """
        if not insights:
            return EnsembleJudgment(
                overall_score=0.0,
                individual_scores={},
                weighted_scores={},
                consensus_level=1.0,
                final_verdict="no_data",
                key_strengths=[],
                critical_weaknesses=["평가할 인사이트가 없습니다"],
                priority_improvements=[]
            )

        logger.info(f"Starting evaluation of {len(insights)} insights with {len(self.judges)} judges")

        judge_scores: Dict[str, JudgeScore] = {}

        if self.parallel:
            # 병렬 실행
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = {
                    executor.submit(judge.evaluate, insights, context): role
                    for role, judge in self.judges.items()
                }

                for future in concurrent.futures.as_completed(futures):
                    role = futures[future]
                    try:
                        score = future.result()
                        judge_scores[role.value] = score
                        logger.info(f"Judge {role.value} completed: score={score.score:.2f}")
                    except Exception as e:
                        logger.error(f"Judge {role.value} failed: {e}")
                        judge_scores[role.value] = JudgeScore(
                            judge_role=role.value,
                            score=0.5,
                            reasoning=f"평가 실패: {str(e)}",
                            strengths=[],
                            weaknesses=[],
                            improvement_suggestions=[],
                            confidence=0.0
                        )
        else:
            # 순차 실행
            for role, judge in self.judges.items():
                try:
                    score = judge.evaluate(insights, context)
                    judge_scores[role.value] = score
                except Exception as e:
                    logger.error(f"Judge {role.value} failed: {e}")
                    judge_scores[role.value] = JudgeScore(
                        judge_role=role.value,
                        score=0.5,
                        reasoning=f"평가 실패: {str(e)}",
                        strengths=[],
                        weaknesses=[],
                        improvement_suggestions=[],
                        confidence=0.0
                    )

        # 앙상블 판정
        return self._ensemble_judgment(judge_scores)

    def _ensemble_judgment(self, judge_scores: Dict[str, JudgeScore]) -> EnsembleJudgment:
        """앙상블 판정 수행"""

        # 개별 점수 추출
        individual_scores = {role: js.score for role, js in judge_scores.items()}

        # 가중 점수 계산
        weighted_scores = {}
        total_weight = 0
        weighted_sum = 0

        for role, score in individual_scores.items():
            weight = self.WEIGHTS.get(role, 0.25)
            weighted_scores[role] = score * weight
            weighted_sum += score * weight
            total_weight += weight

        overall_score = weighted_sum / total_weight if total_weight > 0 else 0.5

        # 합의 수준 계산 (표준편차 기반)
        scores = list(individual_scores.values())
        if len(scores) > 1:
            mean_score = sum(scores) / len(scores)
            variance = sum((s - mean_score) ** 2 for s in scores) / len(scores)
            std_dev = variance ** 0.5
            consensus_level = max(0, 1 - std_dev * 2)  # 표준편차가 낮을수록 합의 수준 높음
        else:
            consensus_level = 1.0

        # 최종 판정
        if overall_score >= 0.85:
            verdict = "excellent"
        elif overall_score >= 0.70:
            verdict = "good"
        elif overall_score >= 0.55:
            verdict = "acceptable"
        elif overall_score >= 0.40:
            verdict = "needs_improvement"
        else:
            verdict = "poor"

        # 강점/약점/개선사항 통합
        all_strengths = []
        all_weaknesses = []
        all_improvements = []

        for js in judge_scores.values():
            all_strengths.extend(js.strengths[:2])  # 각 Judge에서 상위 2개
            all_weaknesses.extend(js.weaknesses[:2])
            all_improvements.extend(js.improvement_suggestions[:2])

        # 중복 제거 및 상위 항목 선택
        key_strengths = list(dict.fromkeys(all_strengths))[:5]
        critical_weaknesses = list(dict.fromkeys(all_weaknesses))[:5]
        priority_improvements = list(dict.fromkeys(all_improvements))[:5]

        return EnsembleJudgment(
            overall_score=round(overall_score, 3),
            individual_scores=individual_scores,
            weighted_scores={k: round(v, 3) for k, v in weighted_scores.items()},
            consensus_level=round(consensus_level, 3),
            final_verdict=verdict,
            key_strengths=key_strengths,
            critical_weaknesses=critical_weaknesses,
            priority_improvements=priority_improvements
        )


def evaluate_insights(
    insights: List[Dict],
    context: Dict[str, Any] = None,
    parallel: bool = True,
    model: str = None
) -> Dict[str, Any]:
    """
    인사이트 평가 편의 함수

    Args:
        insights: 평가할 인사이트 목록
        context: 평가 컨텍스트
        parallel: 병렬 실행 여부
        model: LLM 모델

    Returns:
        평가 결과 딕셔너리
    """
    orchestrator = LLMJudgeOrchestrator(model=model, parallel=parallel)
    judgment = orchestrator.evaluate(insights, context or {})
    return asdict(judgment)
