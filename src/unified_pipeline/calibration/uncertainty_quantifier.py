"""
Uncertainty Quantification for LLM Outputs (v17.0)

불확실성 유형:
1. Aleatoric Uncertainty (데이터 노이즈): 데이터 자체의 고유한 불확실성
2. Epistemic Uncertainty (모델 무지): 모델이 모르는 것에 대한 불확실성

LLM에서의 적용:
- 다중 샘플링을 통한 불확실성 추정
- 토큰 확률 분포 분석
- 의미적 일관성 검사
"""

import logging
import math
import statistics
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class UncertaintyType(str, Enum):
    """불확실성 유형"""
    ALEATORIC = "aleatoric"  # 데이터 노이즈
    EPISTEMIC = "epistemic"  # 모델 무지
    TOTAL = "total"  # 전체 불확실성


@dataclass
class UncertaintyEstimate:
    """불확실성 추정 결과"""
    aleatoric: float  # 데이터 불확실성 (0-1)
    epistemic: float  # 모델 불확실성 (0-1)
    total: float  # 전체 불확실성 (0-1)

    # 추가 메트릭
    entropy: float = 0.0  # 엔트로피
    variance: float = 0.0  # 분산
    consistency_score: float = 1.0  # 일관성 점수

    # 메타데이터
    num_samples: int = 1
    estimation_method: str = "single_pass"
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    @property
    def is_high_uncertainty(self) -> bool:
        """고불확실성 여부 (임계값: 0.5)"""
        return self.total > 0.5

    @property
    def dominant_type(self) -> UncertaintyType:
        """지배적인 불확실성 유형"""
        if self.aleatoric > self.epistemic:
            return UncertaintyType.ALEATORIC
        return UncertaintyType.EPISTEMIC

    @property
    def calibrated_confidence(self) -> float:
        """불확실성 반영 신뢰도"""
        return max(0.0, 1.0 - self.total)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "aleatoric": self.aleatoric,
            "epistemic": self.epistemic,
            "total": self.total,
            "entropy": self.entropy,
            "variance": self.variance,
            "consistency_score": self.consistency_score,
            "num_samples": self.num_samples,
            "estimation_method": self.estimation_method,
            "is_high_uncertainty": self.is_high_uncertainty,
            "dominant_type": self.dominant_type.value,
            "calibrated_confidence": self.calibrated_confidence,
        }


class UncertaintyQuantifier:
    """
    LLM 출력의 불확실성 정량화

    방법론:
    1. Monte Carlo Dropout (MC Dropout) 시뮬레이션
    2. 다중 샘플링 기반 분산 추정
    3. 토큰 확률 엔트로피 분석
    4. 의미적 일관성 점수
    """

    def __init__(
        self,
        num_samples: int = 5,
        temperature_range: tuple = (0.7, 1.3),
        consistency_threshold: float = 0.8,
    ):
        """
        Args:
            num_samples: 불확실성 추정을 위한 샘플링 횟수
            temperature_range: 샘플링 temperature 범위
            consistency_threshold: 일관성 판단 임계값
        """
        self.num_samples = num_samples
        self.temperature_range = temperature_range
        self.consistency_threshold = consistency_threshold

        # 히스토리
        self.estimation_history: List[UncertaintyEstimate] = []

        logger.info(f"UncertaintyQuantifier initialized (samples={num_samples})")

    def estimate_from_samples(
        self,
        samples: List[Dict[str, Any]],
        confidence_key: str = "confidence",
        answer_key: str = "answer",
    ) -> UncertaintyEstimate:
        """
        다중 샘플로부터 불확실성 추정

        Args:
            samples: LLM 응답 샘플 리스트 (각각 confidence와 answer 포함)
            confidence_key: 신뢰도 필드명
            answer_key: 답변 필드명

        Returns:
            UncertaintyEstimate: 불확실성 추정 결과
        """
        if not samples:
            return UncertaintyEstimate(
                aleatoric=1.0,
                epistemic=1.0,
                total=1.0,
                estimation_method="no_samples"
            )

        confidences = [s.get(confidence_key, 0.5) for s in samples]
        answers = [s.get(answer_key, "") for s in samples]

        # 1. Epistemic Uncertainty: 신뢰도 분산
        epistemic = self._compute_epistemic(confidences)

        # 2. Aleatoric Uncertainty: 답변 일관성
        aleatoric, consistency = self._compute_aleatoric(answers)

        # 3. Total Uncertainty: 결합
        total = self._combine_uncertainties(aleatoric, epistemic)

        # 4. 엔트로피 계산
        entropy = self._compute_entropy(confidences)

        # 5. 분산
        variance = statistics.variance(confidences) if len(confidences) > 1 else 0.0

        estimate = UncertaintyEstimate(
            aleatoric=aleatoric,
            epistemic=epistemic,
            total=total,
            entropy=entropy,
            variance=variance,
            consistency_score=consistency,
            num_samples=len(samples),
            estimation_method="multi_sample",
        )

        self.estimation_history.append(estimate)
        return estimate

    def estimate_single(
        self,
        confidence: float,
        token_probs: Optional[List[float]] = None,
    ) -> UncertaintyEstimate:
        """
        단일 응답으로부터 불확실성 추정 (근사)

        Args:
            confidence: 모델이 출력한 신뢰도
            token_probs: 토큰별 확률 (있으면 더 정확한 추정)

        Returns:
            UncertaintyEstimate: 불확실성 추정 결과
        """
        # 단일 샘플에서는 분산 추정이 어려움
        # 휴리스틱: 극단적 신뢰도는 과신일 가능성

        # Epistemic: 신뢰도 기반 추정
        # 과신 패널티: 0.99 이상이면 epistemic 증가
        if confidence > 0.95:
            epistemic = 0.3 + (confidence - 0.95) * 2  # 과신 패널티
        elif confidence < 0.2:
            epistemic = 0.4  # 매우 낮은 신뢰도도 불확실
        else:
            epistemic = 0.2  # 기본 불확실성

        # Aleatoric: 토큰 확률 기반 (있으면)
        if token_probs:
            entropy = self._compute_entropy(token_probs)
            aleatoric = min(1.0, entropy / 2)  # 정규화
        else:
            # 신뢰도 기반 근사
            aleatoric = 1.0 - confidence

        # Total
        total = self._combine_uncertainties(aleatoric, epistemic)

        estimate = UncertaintyEstimate(
            aleatoric=aleatoric,
            epistemic=epistemic,
            total=total,
            entropy=self._compute_entropy([confidence]) if not token_probs else self._compute_entropy(token_probs),
            variance=0.0,  # 단일 샘플은 분산 없음
            consistency_score=1.0,  # 단일 샘플은 일관성 평가 불가
            num_samples=1,
            estimation_method="single_pass",
        )

        self.estimation_history.append(estimate)
        return estimate

    def _compute_epistemic(self, confidences: List[float]) -> float:
        """
        Epistemic Uncertainty 계산

        신뢰도의 분산이 높으면 모델이 "모른다"는 의미
        """
        if len(confidences) <= 1:
            return 0.3  # 기본값

        # 분산 기반
        variance = statistics.variance(confidences)

        # 표준편차로 정규화 (최대 0.5 → 불확실성 1.0)
        std = math.sqrt(variance)
        epistemic = min(1.0, std * 2)

        return epistemic

    def _compute_aleatoric(self, answers: List[str]) -> tuple:
        """
        Aleatoric Uncertainty 계산

        답변의 의미적 일관성으로 데이터 노이즈 추정

        Returns:
            (aleatoric_uncertainty, consistency_score)
        """
        if len(answers) <= 1:
            return 0.2, 1.0  # 기본값

        # 답변 일관성 (간단한 exact match)
        # TODO: 의미적 유사도 (embedding) 사용 가능
        unique_answers = set(answers)
        consistency = 1.0 / len(unique_answers)

        # 불일치 비율이 높으면 aleatoric 높음
        aleatoric = 1.0 - consistency

        return aleatoric, consistency

    def _combine_uncertainties(
        self,
        aleatoric: float,
        epistemic: float,
        method: str = "max"
    ) -> float:
        """
        불확실성 결합

        Args:
            aleatoric: 데이터 불확실성
            epistemic: 모델 불확실성
            method: 결합 방법 ("max", "sum", "sqrt")

        Returns:
            총 불확실성
        """
        if method == "max":
            return max(aleatoric, epistemic)
        elif method == "sum":
            # 클램프
            return min(1.0, aleatoric + epistemic)
        elif method == "sqrt":
            # 제곱합의 제곱근 (기하 평균 비슷)
            return min(1.0, math.sqrt(aleatoric**2 + epistemic**2))
        else:
            return max(aleatoric, epistemic)

    def _compute_entropy(self, probs: List[float]) -> float:
        """
        엔트로피 계산

        H = -Σ p * log(p)
        """
        if not probs:
            return 0.0

        entropy = 0.0
        for p in probs:
            if p > 0:
                entropy -= p * math.log(p + 1e-10)

        return entropy

    def get_uncertainty_summary(self) -> Dict[str, Any]:
        """불확실성 추정 요약"""
        if not self.estimation_history:
            return {
                "total_estimations": 0,
                "avg_total_uncertainty": 0.0,
                "avg_epistemic": 0.0,
                "avg_aleatoric": 0.0,
            }

        total_uncerts = [e.total for e in self.estimation_history]
        epistemic_uncerts = [e.epistemic for e in self.estimation_history]
        aleatoric_uncerts = [e.aleatoric for e in self.estimation_history]

        return {
            "total_estimations": len(self.estimation_history),
            "avg_total_uncertainty": statistics.mean(total_uncerts),
            "avg_epistemic": statistics.mean(epistemic_uncerts),
            "avg_aleatoric": statistics.mean(aleatoric_uncerts),
            "high_uncertainty_count": sum(1 for e in self.estimation_history if e.is_high_uncertainty),
            "dominant_type_counts": self._count_dominant_types(),
        }

    def _count_dominant_types(self) -> Dict[str, int]:
        """지배적 불확실성 유형별 카운트"""
        counts = {UncertaintyType.ALEATORIC.value: 0, UncertaintyType.EPISTEMIC.value: 0}
        for e in self.estimation_history:
            counts[e.dominant_type.value] += 1
        return counts

    def reset(self) -> None:
        """히스토리 초기화"""
        self.estimation_history = []
        logger.info("UncertaintyQuantifier history reset")


class SemanticConsistencyChecker:
    """
    의미적 일관성 검사기

    다중 샘플의 의미적 유사도를 평가하여
    더 정확한 aleatoric uncertainty 추정
    """

    def __init__(self, embedding_fn: Optional[Callable[[str], List[float]]] = None):
        """
        Args:
            embedding_fn: 텍스트 → 임베딩 함수 (없으면 간단한 휴리스틱 사용)
        """
        self.embedding_fn = embedding_fn

    def compute_consistency(self, texts: List[str]) -> float:
        """
        텍스트 리스트의 의미적 일관성 점수

        Returns:
            0-1 사이 점수 (1이 완전 일관)
        """
        if len(texts) <= 1:
            return 1.0

        if self.embedding_fn:
            return self._embedding_consistency(texts)
        else:
            return self._heuristic_consistency(texts)

    def _embedding_consistency(self, texts: List[str]) -> float:
        """임베딩 기반 일관성"""
        embeddings = [self.embedding_fn(t) for t in texts]

        # 코사인 유사도 평균
        similarities = []
        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                sim = self._cosine_similarity(embeddings[i], embeddings[j])
                similarities.append(sim)

        return statistics.mean(similarities) if similarities else 1.0

    def _heuristic_consistency(self, texts: List[str]) -> float:
        """휴리스틱 기반 일관성 (임베딩 없을 때)"""
        # 단어 집합 유사도
        word_sets = [set(t.lower().split()) for t in texts]

        similarities = []
        for i in range(len(word_sets)):
            for j in range(i + 1, len(word_sets)):
                intersection = word_sets[i] & word_sets[j]
                union = word_sets[i] | word_sets[j]
                jaccard = len(intersection) / len(union) if union else 0
                similarities.append(jaccard)

        return statistics.mean(similarities) if similarities else 1.0

    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """코사인 유사도"""
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return dot / (norm_a * norm_b)
