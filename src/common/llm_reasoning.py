"""
LLM-Powered Dynamic Reasoning Engine

하드코딩된 임계값과 규칙 대신 LLM을 사용하여 동적으로 결정합니다.

핵심 기능:
1. DynamicThresholdEngine - 데이터 특성에 맞는 임계값 동적 결정
2. ReasoningChain - 다단계 추론 체인으로 깊은 인사이트 도출
3. ContextualAnalyzer - 도메인 컨텍스트 기반 적응형 분석

"하드코딩된 0.3, 0.5 같은 매직 넘버 대신 LLM이 데이터를 보고 결정"
"""

import json
import logging
from typing import List, Dict, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field, asdict
from enum import Enum

from .utils.llm import (
    chat_completion_with_json,
    get_openai_client,
    extract_json_from_response,
    DEFAULT_MODEL,
)

logger = logging.getLogger(__name__)


class ThresholdType(Enum):
    """임계값 유형"""
    NULL_RATIO = "null_ratio"           # 결측치 비율
    OUTLIER_RATIO = "outlier_ratio"     # 이상치 비율
    CORRELATION = "correlation"          # 상관관계
    DISTINCTNESS = "distinctness"        # 고유값 비율
    QUALITY_SCORE = "quality_score"      # 품질 점수
    CONFIDENCE = "confidence"            # 신뢰도
    SIMILARITY = "similarity"            # 유사도


@dataclass
class DynamicThreshold:
    """동적 임계값"""
    threshold_type: ThresholdType
    recommended_value: float
    reasoning: str
    context_factors: List[str]  # 결정에 영향을 미친 요소들
    confidence: float
    alternative_values: Dict[str, float] = None  # {"conservative": 0.2, "aggressive": 0.5}


@dataclass
class ReasoningStep:
    """추론 단계"""
    step_number: int
    question: str
    analysis: str
    findings: List[str]
    confidence: float
    next_question: Optional[str] = None


@dataclass
class ReasoningChainResult:
    """추론 체인 결과"""
    initial_question: str
    steps: List[ReasoningStep]
    final_conclusion: str
    insights_discovered: List[Dict[str, Any]]
    confidence: float
    reasoning_depth: int


class DynamicThresholdEngine:
    """
    동적 임계값 결정 엔진

    데이터 특성과 도메인 컨텍스트를 분석하여
    하드코딩 대신 적절한 임계값을 LLM이 결정합니다.
    """

    def __init__(self, model: str = None):
        self.model = model or DEFAULT_MODEL
        self.client = get_openai_client()
        self._cache: Dict[str, DynamicThreshold] = {}

    def get_threshold(
        self,
        threshold_type: ThresholdType,
        data_characteristics: Dict[str, Any],
        domain_context: Dict[str, Any] = None,
        use_cache: bool = True
    ) -> DynamicThreshold:
        """
        데이터 특성에 맞는 임계값 동적 결정

        Args:
            threshold_type: 임계값 유형
            data_characteristics: 데이터 특성 (통계, 분포 등)
            domain_context: 도메인 컨텍스트
            use_cache: 캐시 사용 여부

        Returns:
            DynamicThreshold: 동적으로 결정된 임계값
        """
        # 캐시 키 생성
        cache_key = f"{threshold_type.value}:{hash(json.dumps(data_characteristics, sort_keys=True, default=str))}"

        if use_cache and cache_key in self._cache:
            return self._cache[cache_key]

        threshold = self._determine_threshold(threshold_type, data_characteristics, domain_context)

        if use_cache:
            self._cache[cache_key] = threshold

        return threshold

    def _determine_threshold(
        self,
        threshold_type: ThresholdType,
        data_characteristics: Dict[str, Any],
        domain_context: Dict[str, Any] = None
    ) -> DynamicThreshold:
        """LLM을 사용하여 임계값 결정"""

        threshold_descriptions = {
            ThresholdType.NULL_RATIO: "결측치 비율 임계값 (이 이상이면 데이터 품질 경고)",
            ThresholdType.OUTLIER_RATIO: "이상치 비율 임계값 (이 이상이면 이상 탐지 트리거)",
            ThresholdType.CORRELATION: "상관관계 임계값 (이 이상이면 강한 관계로 판단)",
            ThresholdType.DISTINCTNESS: "고유값 비율 임계값 (이 이상이면 잠재적 식별자)",
            ThresholdType.QUALITY_SCORE: "데이터 품질 점수 임계값 (이 이하면 품질 문제)",
            ThresholdType.CONFIDENCE: "신뢰도 임계값 (이 이상이면 신뢰 가능)",
            ThresholdType.SIMILARITY: "유사도 임계값 (이 이상이면 동형 관계)",
        }

        domain_info = ""
        if domain_context:
            domain_info = f"""
도메인 컨텍스트:
- 산업: {domain_context.get('industry', 'unknown')}
- 비즈니스 단계: {domain_context.get('business_stage', 'unknown')}
- 핵심 메트릭: {domain_context.get('key_metrics', [])}
"""

        prompt = f"""데이터 분석을 위한 최적의 임계값을 결정하세요.

## 임계값 유형
{threshold_type.value}: {threshold_descriptions.get(threshold_type, threshold_type.value)}

## 데이터 특성
{json.dumps(data_characteristics, ensure_ascii=False, indent=2)}

{domain_info}

## 요청
이 데이터의 특성을 고려하여 적절한 임계값을 결정하세요.

고려 사항:
1. 데이터의 분포와 특성 (편향, 분산)
2. 산업/도메인 표준
3. 거짓 양성(False Positive) vs 거짓 음성(False Negative) 트레이드오프
4. 데이터 품질 수준
5. 비즈니스 리스크 허용도

JSON 형식으로 반환:
{{
    "recommended_value": 0.0-1.0,
    "reasoning": "이 값을 선택한 이유 (한글)",
    "context_factors": ["결정에 영향을 미친 요소들"],
    "confidence": 0.0-1.0,
    "alternative_values": {{
        "conservative": 0.XX,
        "moderate": 0.XX,
        "aggressive": 0.XX
    }}
}}
"""

        try:
            response = chat_completion_with_json(
                model=self.model,
                messages=[
                    {"role": "system", "content": "당신은 데이터 분석 및 통계 전문가입니다. 데이터 특성에 맞는 최적의 임계값을 결정합니다."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                client=self.client,
            )
            result = extract_json_from_response(response)

            return DynamicThreshold(
                threshold_type=threshold_type,
                recommended_value=float(result.get("recommended_value", 0.5)),
                reasoning=result.get("reasoning", "기본값 사용"),
                context_factors=result.get("context_factors", []),
                confidence=float(result.get("confidence", 0.5)),
                alternative_values=result.get("alternative_values")
            )

        except Exception as e:
            logger.warning(f"Dynamic threshold determination failed: {e}, using default")
            return self._get_default_threshold(threshold_type)

    def _get_default_threshold(self, threshold_type: ThresholdType) -> DynamicThreshold:
        """기본 임계값 반환"""
        defaults = {
            ThresholdType.NULL_RATIO: 0.3,
            ThresholdType.OUTLIER_RATIO: 0.05,
            ThresholdType.CORRELATION: 0.5,
            ThresholdType.DISTINCTNESS: 0.95,
            ThresholdType.QUALITY_SCORE: 0.7,
            ThresholdType.CONFIDENCE: 0.7,
            ThresholdType.SIMILARITY: 0.8,
        }

        return DynamicThreshold(
            threshold_type=threshold_type,
            recommended_value=defaults.get(threshold_type, 0.5),
            reasoning="LLM 결정 실패로 기본값 사용",
            context_factors=["fallback"],
            confidence=0.3
        )

    def batch_determine_thresholds(
        self,
        data_characteristics: Dict[str, Any],
        domain_context: Dict[str, Any] = None
    ) -> Dict[ThresholdType, DynamicThreshold]:
        """모든 임계값을 한 번에 결정"""

        prompt = f"""데이터 분석을 위한 모든 임계값을 한 번에 결정하세요.

## 데이터 특성
{json.dumps(data_characteristics, ensure_ascii=False, indent=2)}

## 도메인 컨텍스트
{json.dumps(domain_context or {}, ensure_ascii=False, indent=2)}

## 결정해야 할 임계값들
1. null_ratio: 결측치 비율 임계값 (이 이상이면 데이터 품질 경고)
2. outlier_ratio: 이상치 비율 임계값 (이 이상이면 이상 탐지)
3. correlation: 상관관계 임계값 (이 이상이면 강한 관계)
4. distinctness: 고유값 비율 임계값 (이 이상이면 식별자 후보)
5. quality_score: 품질 점수 임계값 (이 이하면 문제)
6. confidence: 신뢰도 임계값
7. similarity: 유사도/동형성 임계값

JSON 형식으로 반환:
{{
    "thresholds": {{
        "null_ratio": {{"value": 0.XX, "reasoning": "..."}},
        "outlier_ratio": {{"value": 0.XX, "reasoning": "..."}},
        "correlation": {{"value": 0.XX, "reasoning": "..."}},
        "distinctness": {{"value": 0.XX, "reasoning": "..."}},
        "quality_score": {{"value": 0.XX, "reasoning": "..."}},
        "confidence": {{"value": 0.XX, "reasoning": "..."}},
        "similarity": {{"value": 0.XX, "reasoning": "..."}}
    }},
    "overall_confidence": 0.0-1.0
}}
"""

        try:
            response = chat_completion_with_json(
                model=self.model,
                messages=[
                    {"role": "system", "content": "당신은 데이터 분석 전문가입니다."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                client=self.client,
            )
            result = extract_json_from_response(response)

            thresholds = {}
            for t_type in ThresholdType:
                t_data = result.get("thresholds", {}).get(t_type.value, {})
                thresholds[t_type] = DynamicThreshold(
                    threshold_type=t_type,
                    recommended_value=float(t_data.get("value", 0.5)),
                    reasoning=t_data.get("reasoning", ""),
                    context_factors=[],
                    confidence=float(result.get("overall_confidence", 0.7))
                )

            return thresholds

        except Exception as e:
            logger.warning(f"Batch threshold determination failed: {e}")
            return {t: self._get_default_threshold(t) for t in ThresholdType}


class ReasoningChainEngine:
    """
    다단계 추론 체인 엔진

    깊은 인사이트 발굴을 위해 연속적인 질문-분석-발견 단계를 수행합니다.
    "왜?" → "그래서?" → "어떻게?" 체인
    """

    def __init__(self, model: str = None, max_depth: int = 5):
        self.model = model or DEFAULT_MODEL
        self.client = get_openai_client()
        self.max_depth = max_depth

    def run_reasoning_chain(
        self,
        initial_question: str,
        data_context: Dict[str, Any],
        domain_context: Dict[str, Any] = None,
        min_confidence: float = 0.6
    ) -> ReasoningChainResult:
        """
        다단계 추론 체인 실행

        Args:
            initial_question: 초기 분석 질문
            data_context: 데이터 컨텍스트
            domain_context: 도메인 컨텍스트
            min_confidence: 최소 신뢰도 (이 이하면 추론 중단)

        Returns:
            ReasoningChainResult: 추론 체인 결과
        """
        steps = []
        current_question = initial_question
        accumulated_findings = []
        insights_discovered = []

        for depth in range(1, self.max_depth + 1):
            step = self._execute_reasoning_step(
                step_number=depth,
                question=current_question,
                data_context=data_context,
                domain_context=domain_context,
                previous_findings=accumulated_findings
            )

            steps.append(step)
            accumulated_findings.extend(step.findings)

            # 인사이트 추출
            if step.findings:
                for finding in step.findings:
                    insights_discovered.append({
                        "finding": finding,
                        "discovered_at_step": depth,
                        "confidence": step.confidence
                    })

            # 중단 조건 확인
            if step.confidence < min_confidence:
                logger.info(f"Reasoning chain stopped at step {depth}: confidence too low")
                break

            if not step.next_question:
                logger.info(f"Reasoning chain completed at step {depth}: no follow-up question")
                break

            current_question = step.next_question

        # 최종 결론 생성
        final_conclusion = self._generate_final_conclusion(
            initial_question, steps, domain_context
        )

        return ReasoningChainResult(
            initial_question=initial_question,
            steps=steps,
            final_conclusion=final_conclusion,
            insights_discovered=insights_discovered,
            confidence=sum(s.confidence for s in steps) / len(steps) if steps else 0,
            reasoning_depth=len(steps)
        )

    def _execute_reasoning_step(
        self,
        step_number: int,
        question: str,
        data_context: Dict[str, Any],
        domain_context: Dict[str, Any],
        previous_findings: List[str]
    ) -> ReasoningStep:
        """단일 추론 단계 실행"""

        previous_context = ""
        if previous_findings:
            previous_context = f"""
## 이전 발견 사항
{chr(10).join(f'- {f}' for f in previous_findings[-5:])}
"""

        domain_info = ""
        if domain_context:
            domain_info = f"""
## 도메인 컨텍스트
산업: {domain_context.get('industry', 'unknown')}
핵심 엔티티: {domain_context.get('key_entities', [])}
"""

        prompt = f"""## 분석 질문 (Step {step_number})
{question}

## 데이터 컨텍스트
{json.dumps(data_context, ensure_ascii=False, indent=2)[:2000]}

{domain_info}
{previous_context}

## 요청
위 질문에 대해 깊이 분석하고:
1. 데이터에서 발견한 사실들
2. 그 사실이 의미하는 바
3. 후속으로 탐구해야 할 질문

JSON 형식으로 반환:
{{
    "analysis": "분석 내용 (한글)",
    "findings": ["발견한 사실들 (구체적 수치 포함)"],
    "confidence": 0.0-1.0,
    "next_question": "후속 질문 (더 탐구 필요 없으면 null)"
}}
"""

        try:
            response = chat_completion_with_json(
                model=self.model,
                messages=[
                    {"role": "system", "content": "당신은 데이터 분석 전문가입니다. 깊이 있는 추론과 인사이트 발굴을 수행합니다."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                client=self.client,
            )
            result = extract_json_from_response(response)

            return ReasoningStep(
                step_number=step_number,
                question=question,
                analysis=result.get("analysis", ""),
                findings=result.get("findings", []),
                confidence=float(result.get("confidence", 0.5)),
                next_question=result.get("next_question")
            )

        except Exception as e:
            logger.error(f"Reasoning step {step_number} failed: {e}")
            return ReasoningStep(
                step_number=step_number,
                question=question,
                analysis=f"분석 실패: {str(e)}",
                findings=[],
                confidence=0.0,
                next_question=None
            )

    def _generate_final_conclusion(
        self,
        initial_question: str,
        steps: List[ReasoningStep],
        domain_context: Dict[str, Any]
    ) -> str:
        """최종 결론 생성"""

        if not steps:
            return "추론 단계가 없어 결론을 도출할 수 없습니다."

        all_findings = []
        for step in steps:
            all_findings.extend(step.findings)

        prompt = f"""## 초기 질문
{initial_question}

## 추론 과정에서 발견된 사항들
{chr(10).join(f'- {f}' for f in all_findings)}

## 요청
위 발견 사항들을 종합하여 최종 결론을 도출하세요.

결론은:
1. 명확하고 실행 가능해야 합니다
2. 구체적인 수치와 근거를 포함해야 합니다
3. 비즈니스 관점에서 의미 있어야 합니다

2-3문장으로 결론을 작성하세요. (한글)
"""

        try:
            response = chat_completion_with_json(
                model=self.model,
                messages=[
                    {"role": "system", "content": "당신은 비즈니스 분석 전문가입니다."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                client=self.client,
            )
            result = extract_json_from_response(response)
            return result.get("conclusion", response) if isinstance(result, dict) else response

        except Exception as e:
            logger.error(f"Final conclusion generation failed: {e}")
            return f"추론 완료. {len(all_findings)}개의 발견 사항이 있습니다."


class ContextualAnalyzer:
    """
    컨텍스트 기반 적응형 분석기

    도메인, 데이터 특성, 비즈니스 상황에 따라
    분석 방법과 기준을 동적으로 조정합니다.
    """

    def __init__(self, model: str = None):
        self.model = model or DEFAULT_MODEL
        self.client = get_openai_client()
        self.threshold_engine = DynamicThresholdEngine(model)
        self.reasoning_engine = ReasoningChainEngine(model)

    def analyze_adaptively(
        self,
        data: Dict[str, Any],
        analysis_goal: str,
        domain_context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        적응형 분석 수행

        Args:
            data: 분석할 데이터
            analysis_goal: 분석 목표
            domain_context: 도메인 컨텍스트

        Returns:
            분석 결과
        """
        # 1. 데이터 특성 파악
        data_characteristics = self._characterize_data(data)

        # 2. 동적 임계값 결정
        thresholds = self.threshold_engine.batch_determine_thresholds(
            data_characteristics, domain_context
        )

        # 3. 분석 전략 결정
        analysis_strategy = self._determine_analysis_strategy(
            data_characteristics, domain_context, analysis_goal
        )

        # 4. 추론 체인 실행
        reasoning_result = self.reasoning_engine.run_reasoning_chain(
            initial_question=analysis_goal,
            data_context=data,
            domain_context=domain_context
        )

        return {
            "data_characteristics": data_characteristics,
            "thresholds_used": {t.value: asdict(th) if hasattr(th, '__dataclass_fields__') else th
                               for t, th in thresholds.items()},
            "analysis_strategy": analysis_strategy,
            "reasoning_result": {
                "steps": [asdict(s) if hasattr(s, '__dataclass_fields__') else s
                         for s in reasoning_result.steps],
                "conclusion": reasoning_result.final_conclusion,
                "insights": reasoning_result.insights_discovered,
                "confidence": reasoning_result.confidence,
                "depth": reasoning_result.reasoning_depth
            }
        }

    def _characterize_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """데이터 특성 파악"""
        characteristics = {
            "num_tables": 0,
            "total_columns": 0,
            "total_rows": 0,
            "avg_null_ratio": 0,
            "data_types": {},
            "complexity": "unknown"
        }

        tables = data.get("tables", [])
        characteristics["num_tables"] = len(tables)

        null_ratios = []
        for table in tables:
            columns = table.get("columns", [])
            characteristics["total_columns"] += len(columns)
            characteristics["total_rows"] += table.get("row_count", 0)

            for col in columns:
                dtype = col.get("inferred_type", "unknown")
                characteristics["data_types"][dtype] = characteristics["data_types"].get(dtype, 0) + 1
                null_ratios.append(col.get("null_ratio", 0))

        if null_ratios:
            characteristics["avg_null_ratio"] = sum(null_ratios) / len(null_ratios)

        # 복잡도 결정
        if characteristics["num_tables"] <= 2:
            characteristics["complexity"] = "simple"
        elif characteristics["num_tables"] <= 5:
            characteristics["complexity"] = "moderate"
        else:
            characteristics["complexity"] = "complex"

        return characteristics

    def _determine_analysis_strategy(
        self,
        data_characteristics: Dict[str, Any],
        domain_context: Dict[str, Any],
        analysis_goal: str
    ) -> Dict[str, Any]:
        """분석 전략 결정"""

        prompt = f"""데이터 분석을 위한 최적의 전략을 결정하세요.

## 분석 목표
{analysis_goal}

## 데이터 특성
{json.dumps(data_characteristics, ensure_ascii=False, indent=2)}

## 도메인 컨텍스트
{json.dumps(domain_context or {}, ensure_ascii=False, indent=2)}

## 요청
이 상황에 맞는 분석 전략을 결정하세요.

JSON 형식으로 반환:
{{
    "primary_approach": "correlation|anomaly|profiling|pattern|relationship",
    "focus_areas": ["집중할 분석 영역들"],
    "expected_insights": ["예상되는 인사이트 유형들"],
    "risk_factors": ["주의해야 할 위험 요소들"],
    "recommended_depth": "shallow|moderate|deep",
    "reasoning": "이 전략을 선택한 이유"
}}
"""

        try:
            response = chat_completion_with_json(
                model=self.model,
                messages=[
                    {"role": "system", "content": "당신은 데이터 분석 전략 전문가입니다."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                client=self.client,
            )
            return extract_json_from_response(response)

        except Exception as e:
            logger.error(f"Analysis strategy determination failed: {e}")
            return {
                "primary_approach": "profiling",
                "focus_areas": ["데이터 품질", "기본 통계"],
                "expected_insights": ["데이터 품질 인사이트"],
                "risk_factors": [],
                "recommended_depth": "moderate",
                "reasoning": "기본 전략 사용"
            }


# 편의 함수들
def get_dynamic_threshold(
    threshold_type: str,
    data_characteristics: Dict[str, Any],
    domain_context: Dict[str, Any] = None
) -> float:
    """동적 임계값 가져오기 편의 함수"""
    # 기본값 매핑
    defaults = {
        "null_ratio": 0.3,
        "outlier_ratio": 0.05,
        "correlation": 0.5,
        "distinctness": 0.95,
        "quality_score": 0.7,
        "confidence": 0.7,
        "similarity": 0.8,
    }

    try:
        t_type = ThresholdType(threshold_type)
    except ValueError:
        logger.warning(f"Invalid threshold type: {threshold_type}, using default")
        return defaults.get(threshold_type, 0.5)

    try:
        engine = DynamicThresholdEngine()
        threshold = engine.get_threshold(t_type, data_characteristics, domain_context)
        return threshold.recommended_value
    except Exception as e:
        logger.warning(f"Dynamic threshold failed for {threshold_type}: {e}")
        return defaults.get(threshold_type, 0.5)


def run_deep_analysis(
    question: str,
    data_context: Dict[str, Any],
    domain_context: Dict[str, Any] = None,
    max_depth: int = 5
) -> Dict[str, Any]:
    """깊은 추론 분석 편의 함수"""
    engine = ReasoningChainEngine(max_depth=max_depth)
    result = engine.run_reasoning_chain(question, data_context, domain_context)
    return {
        "conclusion": result.final_conclusion,
        "insights": result.insights_discovered,
        "steps": len(result.steps),
        "confidence": result.confidence
    }
