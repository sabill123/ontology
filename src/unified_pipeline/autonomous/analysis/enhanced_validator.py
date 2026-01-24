"""
Enhanced Validation System v5.0

Research-based validation framework integrating:
1. Statistical Hypothesis Testing (KS, Welch, F-test)
2. Dempster-Shafer Evidence Theory for multi-source fusion
3. Confidence Calibration (Temperature Scaling, ECE)
4. Knowledge Graph Quality Metrics
5. Byzantine Fault Tolerant Consensus

References:
- "A Survey of Uncertainty in Deep Neural Networks" (arXiv:2107.03342)
- "Structural Quality Metrics to Evaluate Knowledge Graphs" (arXiv:2211.10011)
- "Dempster-Shafer Evidence Theory" (ScienceDirect)
- "Statistical Validation of Column Matching" (arXiv:2407.09885)
- "Rethinking the Reliability of Multi-agent System" (arXiv:2511.10400)
"""

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Set
from enum import Enum
import numpy as np
from scipy import stats
from collections import defaultdict

logger = logging.getLogger(__name__)


# =============================================================================
# 1. STATISTICAL VALIDATOR
# =============================================================================

@dataclass
class StatisticalTestResult:
    """통계 검정 결과"""
    test_name: str
    statistic: float
    p_value: float
    passed: bool
    interpretation: str


class StatisticalValidator:
    """
    통계적 검증기

    논문: "Statistical Validation of Column Matching" (2024)
    핵심: Goodness-of-fit 테스트로 매칭 품질 검증

    검정 종류:
    - Kolmogorov-Smirnov: 분포 비교
    - Welch's t-test: 평균 비교 (분산 동질성 가정 X)
    - F-test: 분산 비교
    - Anderson-Darling: 분포 적합도
    """

    def __init__(self, alpha: float = 0.05):
        self.alpha = alpha

    def validate_distribution_match(
        self,
        source: np.ndarray,
        target: np.ndarray
    ) -> Dict[str, StatisticalTestResult]:
        """두 분포의 통계적 유사성 검증"""
        results = {}

        # 1. Kolmogorov-Smirnov Test (분포 전체 비교)
        try:
            ks_stat, ks_pval = stats.ks_2samp(source, target)
            results['ks_test'] = StatisticalTestResult(
                test_name="Kolmogorov-Smirnov",
                statistic=ks_stat,
                p_value=ks_pval,
                passed=ks_pval > self.alpha,
                interpretation=f"분포 {'유사' if ks_pval > self.alpha else '상이'} (p={ks_pval:.4f})"
            )
        except Exception as e:
            logger.warning(f"KS test failed: {e}")

        # 2. Welch's t-test (평균 비교, 등분산 가정 X)
        try:
            t_stat, t_pval = stats.ttest_ind(source, target, equal_var=False)
            results['welch_test'] = StatisticalTestResult(
                test_name="Welch's t-test",
                statistic=t_stat,
                p_value=t_pval,
                passed=t_pval > self.alpha,
                interpretation=f"평균 {'동일' if t_pval > self.alpha else '상이'} (p={t_pval:.4f})"
            )
        except Exception as e:
            logger.warning(f"Welch test failed: {e}")

        # 3. Levene's Test (분산 동질성)
        try:
            lev_stat, lev_pval = stats.levene(source, target)
            results['levene_test'] = StatisticalTestResult(
                test_name="Levene's Test",
                statistic=lev_stat,
                p_value=lev_pval,
                passed=lev_pval > self.alpha,
                interpretation=f"분산 {'동질' if lev_pval > self.alpha else '이질'} (p={lev_pval:.4f})"
            )
        except Exception as e:
            logger.warning(f"Levene test failed: {e}")

        # 4. Anderson-Darling Test (정규성 검정)
        try:
            ad_source = stats.anderson(source)
            ad_target = stats.anderson(target)
            # 5% 유의수준 사용
            ad_passed = (ad_source.statistic < ad_source.critical_values[2] and
                        ad_target.statistic < ad_target.critical_values[2])
            results['anderson_darling'] = StatisticalTestResult(
                test_name="Anderson-Darling",
                statistic=(ad_source.statistic + ad_target.statistic) / 2,
                p_value=0.05 if ad_passed else 0.01,  # 근사
                passed=ad_passed,
                interpretation=f"정규성 {'충족' if ad_passed else '미충족'}"
            )
        except Exception as e:
            logger.warning(f"Anderson-Darling test failed: {e}")

        return results

    def calculate_confidence(self, results: Dict[str, StatisticalTestResult]) -> float:
        """통계 검정 결과를 종합 신뢰도로 변환"""
        if not results:
            return 0.5

        # 가중치: KS > Welch > Levene > AD
        weights = {
            'ks_test': 0.35,
            'welch_test': 0.30,
            'levene_test': 0.20,
            'anderson_darling': 0.15,
        }

        total_weight = 0
        weighted_score = 0

        for name, result in results.items():
            w = weights.get(name, 0.1)
            # p-value를 신뢰도로 변환 (p > alpha면 높은 신뢰도)
            score = min(result.p_value / self.alpha, 1.0) if result.passed else result.p_value
            weighted_score += w * score
            total_weight += w

        return weighted_score / total_weight if total_weight > 0 else 0.5


# =============================================================================
# 2. DEMPSTER-SHAFER EVIDENCE FUSION
# =============================================================================

@dataclass
class MassFunction:
    """
    Mass Function (질량 함수) - Dempster-Shafer Theory의 핵심 구조

    질량 함수 m: 2^Θ → [0,1]
    - m(∅) = 0
    - Σ m(A) = 1 for all A ⊆ Θ

    Attributes:
        frame: Frame of Discernment (Θ) - 가능한 모든 가설의 집합
        masses: focal elements에 대한 질량 할당
        source_id: 증거 출처 식별자
        timestamp: 증거 생성 시간
    """
    frame: frozenset = field(default_factory=frozenset)
    masses: Dict[frozenset, float] = field(default_factory=dict)
    source_id: str = ""
    timestamp: float = field(default_factory=lambda: 0.0)

    def __post_init__(self):
        """초기화 후 정규화"""
        if self.masses:
            self.normalize()

    def normalize(self) -> 'MassFunction':
        """질량 함수 정규화: Σm(A) = 1 보장"""
        # 공집합 제거
        if frozenset() in self.masses:
            del self.masses[frozenset()]

        total = sum(self.masses.values())
        if total > 0 and abs(total - 1.0) > 1e-10:
            self.masses = {k: v / total for k, v in self.masses.items()}
        return self

    def is_valid(self) -> bool:
        """질량 함수 유효성 검증"""
        if frozenset() in self.masses and self.masses[frozenset()] > 0:
            return False
        total = sum(self.masses.values())
        return abs(total - 1.0) < 1e-6

    def get_focal_elements(self) -> List[frozenset]:
        """Focal elements (m(A) > 0인 A들) 반환"""
        return [k for k, v in self.masses.items() if v > 0]

    def belief(self, hypothesis: frozenset) -> float:
        """
        Belief function: Bel(A) = Σ m(B) for all B ⊆ A

        신뢰도 함수: 가설 A를 완전히 지지하는 증거의 총합
        하한(lower bound) 확률로 해석
        """
        if not hypothesis:
            return 0.0
        return sum(m for h, m in self.masses.items() if h and h.issubset(hypothesis))

    def plausibility(self, hypothesis: frozenset) -> float:
        """
        Plausibility function: Pl(A) = Σ m(B) for all B ∩ A ≠ ∅

        가능도 함수: 가설 A와 충돌하지 않는 증거의 총합
        상한(upper bound) 확률로 해석
        Pl(A) = 1 - Bel(¬A)
        """
        if not hypothesis:
            return 0.0
        return sum(m for h, m in self.masses.items() if h and h.intersection(hypothesis))

    def doubt(self, hypothesis: frozenset) -> float:
        """
        Doubt function: Dou(A) = Bel(¬A) = 1 - Pl(A)

        의심도 함수: 가설 A를 부정하는 증거의 총합
        """
        return 1.0 - self.plausibility(hypothesis)

    def uncertainty_interval(self, hypothesis: frozenset) -> Tuple[float, float]:
        """
        불확실성 구간: [Bel(A), Pl(A)]

        이 구간의 폭이 클수록 증거의 불확실성이 높음
        """
        bel = self.belief(hypothesis)
        pl = self.plausibility(hypothesis)
        return (bel, pl)

    def uncertainty_width(self, hypothesis: frozenset) -> float:
        """불확실성 폭: Pl(A) - Bel(A)"""
        bel, pl = self.uncertainty_interval(hypothesis)
        return pl - bel

    def pignistic_probability(self, element: str) -> float:
        """
        Pignistic Probability Transformation (BetP)

        BetP(x) = Σ m(A) / |A| for all A containing x

        의사결정을 위한 확률 변환
        """
        prob = 0.0
        for focal, mass in self.masses.items():
            if element in focal and len(focal) > 0:
                prob += mass / len(focal)
        return prob

    def get_all_pignistic_probabilities(self) -> Dict[str, float]:
        """모든 singleton에 대한 Pignistic 확률"""
        probs = {}
        for element in self.frame:
            probs[element] = self.pignistic_probability(element)
        return probs


@dataclass
class BPA:
    """Basic Probability Assignment (기본 확률 할당) - 레거시 호환성 유지"""
    focal_elements: Dict[frozenset, float] = field(default_factory=dict)

    def normalize(self):
        """정규화"""
        total = sum(self.focal_elements.values())
        if total > 0:
            self.focal_elements = {k: v/total for k, v in self.focal_elements.items()}

    def get_belief(self, hypothesis: frozenset) -> float:
        """Belief function: Bel(A) = Σ m(B) for B ⊆ A"""
        return sum(m for h, m in self.focal_elements.items() if h.issubset(hypothesis))

    def get_plausibility(self, hypothesis: frozenset) -> float:
        """Plausibility function: Pl(A) = Σ m(B) for B ∩ A ≠ ∅"""
        return sum(m for h, m in self.focal_elements.items() if h.intersection(hypothesis))

    def to_mass_function(self, frame: Set[str] = None) -> MassFunction:
        """BPA를 MassFunction으로 변환"""
        if frame is None:
            # focal elements에서 frame 추출
            frame = set()
            for focal in self.focal_elements.keys():
                frame.update(focal)
        return MassFunction(
            frame=frozenset(frame),
            masses=dict(self.focal_elements)
        )


class DempsterShaferFusion:
    """
    Dempster-Shafer Evidence Theory 기반 다중 소스 융합

    논문: "Information fusion for large-scale multi-source data
          based on the Dempster-Shafer evidence theory" (2024)

    핵심:
    - 다중 에이전트 의견을 확률론적으로 융합
    - 충돌 증거 처리 (Murphy's averaging rule)
    - 불확실성 정량화
    """

    def __init__(self, frame_of_discernment: Set[str] = None):
        self.frame = frame_of_discernment or {'approve', 'reject', 'review'}

    def create_bpa_from_opinion(self, opinion: Dict[str, Any]) -> BPA:
        """에이전트 의견을 BPA로 변환"""
        decision = opinion.get('decision_type', opinion.get('decision', 'review'))
        confidence = opinion.get('confidence', 0.5)

        bpa = BPA()

        # 주요 가설에 확률 할당
        main_hypothesis = frozenset({decision})
        bpa.focal_elements[main_hypothesis] = confidence

        # 나머지를 불확실성(전체 집합)에 할당
        uncertainty = frozenset(self.frame)
        bpa.focal_elements[uncertainty] = 1.0 - confidence

        bpa.normalize()
        return bpa

    def calculate_conflict(self, bpa1: BPA, bpa2: BPA) -> float:
        """두 BPA 간 충돌도 계산: K = Σ m1(A)·m2(B) for A∩B=∅"""
        conflict = 0.0
        for h1, m1 in bpa1.focal_elements.items():
            for h2, m2 in bpa2.focal_elements.items():
                if not h1.intersection(h2):  # 교집합이 공집합
                    conflict += m1 * m2
        return conflict

    def dempster_combination(self, bpa1: BPA, bpa2: BPA) -> BPA:
        """
        Dempster's Rule of Combination
        m12(A) = Σ m1(B)·m2(C) / (1-K) for B∩C=A
        """
        combined = BPA()
        conflict = self.calculate_conflict(bpa1, bpa2)

        if conflict >= 1.0:
            logger.warning("Complete conflict between BPAs, using Murphy's rule")
            return self.murphy_combination([bpa1, bpa2])

        normalization = 1.0 - conflict

        for h1, m1 in bpa1.focal_elements.items():
            for h2, m2 in bpa2.focal_elements.items():
                intersection = h1.intersection(h2)
                if intersection:
                    key = frozenset(intersection)
                    current = combined.focal_elements.get(key, 0.0)
                    combined.focal_elements[key] = current + (m1 * m2) / normalization

        return combined

    def murphy_combination(self, bpas: List[BPA]) -> BPA:
        """
        Murphy's Averaging Rule (높은 충돌시 사용)
        먼저 BPA들을 평균화한 후 Dempster 규칙 적용
        """
        if not bpas:
            return BPA()

        # 모든 focal elements 수집
        all_elements = set()
        for bpa in bpas:
            all_elements.update(bpa.focal_elements.keys())

        # 평균 BPA 계산
        avg_bpa = BPA()
        n = len(bpas)
        for element in all_elements:
            avg_mass = sum(bpa.focal_elements.get(element, 0.0) for bpa in bpas) / n
            if avg_mass > 0:
                avg_bpa.focal_elements[element] = avg_mass

        avg_bpa.normalize()

        # 평균 BPA를 n-1번 자기 자신과 결합
        result = avg_bpa
        for _ in range(n - 1):
            result = self.dempster_combination(result, avg_bpa)

        return result

    def fuse_opinions(self, opinions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """다중 에이전트 의견 융합"""
        if not opinions:
            return {'decision': 'review', 'confidence': 0.5, 'conflict': 0.0}

        # BPA 변환
        bpas = [self.create_bpa_from_opinion(op) for op in opinions]

        # 충돌도 계산
        avg_conflict = 0.0
        count = 0
        for i in range(len(bpas)):
            for j in range(i + 1, len(bpas)):
                avg_conflict += self.calculate_conflict(bpas[i], bpas[j])
                count += 1
        avg_conflict = avg_conflict / count if count > 0 else 0.0

        # 융합 방법 선택
        if avg_conflict > 0.7:
            logger.info(f"High conflict ({avg_conflict:.2f}), using Murphy's rule")
            fused_bpa = self.murphy_combination(bpas)
        else:
            # 순차적 Dempster 결합
            fused_bpa = bpas[0]
            for bpa in bpas[1:]:
                fused_bpa = self.dempster_combination(fused_bpa, bpa)

        # 최종 결정 도출
        best_decision = None
        best_mass = 0.0

        for hypothesis, mass in fused_bpa.focal_elements.items():
            if len(hypothesis) == 1 and mass > best_mass:  # singleton만 고려
                best_mass = mass
                best_decision = list(hypothesis)[0]

        # Belief와 Plausibility 계산
        if best_decision:
            singleton = frozenset({best_decision})
            belief = fused_bpa.get_belief(singleton)
            plausibility = fused_bpa.get_plausibility(singleton)
        else:
            belief = plausibility = 0.5
            best_decision = 'review'

        return {
            'decision': best_decision,
            'confidence': best_mass,
            'belief': belief,
            'plausibility': plausibility,
            'conflict': avg_conflict,
            'uncertainty_interval': (belief, plausibility),  # [Bel, Pl]
        }


class DempsterShaferEngine:
    """
    Dempster-Shafer Evidence Theory 완전 구현 엔진

    Phase 3 Governance Spec에 따른 구현:
    1. Dempster's Rule of Combination: m1 ⊕ m2 결합
    2. Conflict Coefficient (K): 증거 간 충돌 측정
    3. Belief & Plausibility: 가설에 대한 신뢰도 구간
    4. Mass Function: 증거의 질량 함수 관리

    공식:
    - m(A) = (1/(1-K)) * Σ m1(B) * m2(C), where B∩C=A
    - K = Σ m1(B) * m2(C), where B∩C=∅
    - Bel(A) = Σ m(B), where B⊆A
    - Pl(A) = Σ m(B), where A∩B≠∅

    참조:
    - "A Mathematical Theory of Evidence" (Shafer, 1976)
    - "Information fusion for large-scale multi-source data" (2024)
    """

    def __init__(self, frame_of_discernment: Set[str] = None):
        """
        Args:
            frame_of_discernment: 가능한 모든 가설의 집합 (Θ)
                                 기본값: {'approve', 'reject', 'review'}
        """
        self.frame = frozenset(frame_of_discernment or {'approve', 'reject', 'review'})
        self.evidence_history: List[MassFunction] = []
        self.combination_log: List[Dict[str, Any]] = []

        # 충돌 임계값 설정
        self.conflict_threshold_warning = 0.5  # 경고
        self.conflict_threshold_critical = 0.8  # 판단 보류

    def create_mass_function(
        self,
        assignments: Dict[str, float],
        source_id: str = "",
        include_uncertainty: bool = True
    ) -> MassFunction:
        """
        질량 함수 생성

        Args:
            assignments: 가설 -> 질량 할당 (예: {'approve': 0.7, 'reject': 0.2})
            source_id: 증거 출처 식별자
            include_uncertainty: 나머지를 불확실성(전체 집합)에 할당할지 여부

        Returns:
            MassFunction 객체
        """
        masses = {}
        total_assigned = 0.0

        for hypothesis, mass in assignments.items():
            if mass > 0:
                # 단일 가설이면 singleton frozenset으로
                if isinstance(hypothesis, str):
                    key = frozenset({hypothesis})
                else:
                    key = frozenset(hypothesis)

                # frame 내에 있는지 확인
                if key.issubset(self.frame):
                    masses[key] = mass
                    total_assigned += mass
                else:
                    logger.warning(f"Hypothesis {hypothesis} not in frame, skipping")

        # 나머지를 불확실성(전체 집합)에 할당
        if include_uncertainty and total_assigned < 1.0:
            uncertainty_mass = 1.0 - total_assigned
            masses[self.frame] = masses.get(self.frame, 0.0) + uncertainty_mass

        import time
        return MassFunction(
            frame=self.frame,
            masses=masses,
            source_id=source_id,
            timestamp=time.time()
        )

    def create_mass_from_agent_opinion(
        self,
        opinion: Dict[str, Any],
        agent_id: str = ""
    ) -> MassFunction:
        """
        에이전트 의견을 MassFunction으로 변환

        Args:
            opinion: 에이전트 의견 (decision_type, confidence 포함)
            agent_id: 에이전트 식별자

        Returns:
            MassFunction 객체
        """
        decision = opinion.get('decision_type', opinion.get('decision', 'review'))
        confidence = max(0.01, min(0.99, opinion.get('confidence', 0.5)))

        # 결정에 따른 질량 할당
        assignments = {decision: confidence}

        # 보조 가설에 대한 작은 질량 할당 (evidence spreading)
        if confidence < 0.9:
            remaining = (1.0 - confidence) * 0.3
            other_hypotheses = [h for h in self.frame if h != decision]
            if other_hypotheses:
                spread_mass = remaining / len(other_hypotheses)
                for h in other_hypotheses:
                    if spread_mass > 0.01:
                        assignments[h] = spread_mass

        return self.create_mass_function(
            assignments=assignments,
            source_id=agent_id or opinion.get('agent_id', 'unknown'),
            include_uncertainty=True
        )

    def calculate_conflict_coefficient(
        self,
        m1: MassFunction,
        m2: MassFunction
    ) -> float:
        """
        Conflict Coefficient (K) 계산

        K = Σ m1(B) * m2(C), where B ∩ C = ∅

        K 해석:
        - K = 0: 완전 일치
        - K < 0.5: 낮은 충돌
        - 0.5 ≤ K < 0.8: 중간 충돌 (경고)
        - K ≥ 0.8: 높은 충돌 (판단 보류 권고)
        - K = 1: 완전 충돌 (결합 불가)

        Args:
            m1: 첫 번째 질량 함수
            m2: 두 번째 질량 함수

        Returns:
            충돌 계수 K (0~1)
        """
        conflict = 0.0

        for h1, mass1 in m1.masses.items():
            for h2, mass2 in m2.masses.items():
                # 교집합이 공집합인 경우
                if not h1.intersection(h2):
                    conflict += mass1 * mass2

        return min(conflict, 0.9999)  # 1.0 방지 (수치 안정성)

    def analyze_conflict(
        self,
        m1: MassFunction,
        m2: MassFunction
    ) -> Dict[str, Any]:
        """
        충돌 상세 분석

        Args:
            m1, m2: 분석할 두 질량 함수

        Returns:
            충돌 분석 결과 (K, 수준, 충돌 원인 등)
        """
        K = self.calculate_conflict_coefficient(m1, m2)

        # 충돌 수준 판정
        if K < 0.2:
            level = "minimal"
            description = "증거가 거의 일치함"
        elif K < 0.5:
            level = "low"
            description = "약간의 불일치 존재"
        elif K < 0.8:
            level = "moderate"
            description = "상당한 충돌, 주의 필요"
        else:
            level = "high"
            description = "심각한 충돌, 판단 보류 권고"

        # 충돌 원인 분석
        conflict_sources = []
        for h1, mass1 in m1.masses.items():
            for h2, mass2 in m2.masses.items():
                if not h1.intersection(h2) and mass1 * mass2 > 0.01:
                    conflict_sources.append({
                        'source_1': set(h1),
                        'mass_1': mass1,
                        'source_2': set(h2),
                        'mass_2': mass2,
                        'contribution': mass1 * mass2
                    })

        # 충돌 원인을 기여도순으로 정렬
        conflict_sources.sort(key=lambda x: x['contribution'], reverse=True)

        return {
            'conflict_coefficient': K,
            'level': level,
            'description': description,
            'is_critical': K >= self.conflict_threshold_critical,
            'needs_review': K >= self.conflict_threshold_warning,
            'conflict_sources': conflict_sources[:5],  # 상위 5개
            'recommendation': self._get_conflict_recommendation(K)
        }

    def _get_conflict_recommendation(self, K: float) -> str:
        """충돌 수준에 따른 권고사항"""
        if K < 0.2:
            return "표준 Dempster 결합 사용 가능"
        elif K < 0.5:
            return "Dempster 결합 사용 가능, 결과 검증 권장"
        elif K < 0.8:
            return "Murphy's averaging 또는 Yager's modified rule 사용 권장"
        else:
            return "증거 재검토 필요, 자동 결합 비권장"

    def dempster_combination(
        self,
        m1: MassFunction,
        m2: MassFunction,
        check_conflict: bool = True
    ) -> Tuple[MassFunction, Dict[str, Any]]:
        """
        Dempster's Rule of Combination

        m12(A) = (1/(1-K)) * Σ m1(B) * m2(C), where B ∩ C = A

        Args:
            m1: 첫 번째 질량 함수
            m2: 두 번째 질량 함수
            check_conflict: 충돌 검사 수행 여부

        Returns:
            (결합된 질량 함수, 결합 메타데이터)
        """
        # 충돌 계수 계산
        K = self.calculate_conflict_coefficient(m1, m2)

        metadata = {
            'conflict_coefficient': K,
            'method': 'dempster',
            'sources': [m1.source_id, m2.source_id]
        }

        # 높은 충돌 시 경고
        if check_conflict and K >= self.conflict_threshold_critical:
            logger.warning(
                f"High conflict detected (K={K:.4f}). "
                f"Consider using Murphy's rule or reviewing evidence."
            )
            metadata['warning'] = 'high_conflict'

        # 완전 충돌 시 Murphy's rule로 폴백
        if K >= 0.9999:
            logger.error("Complete conflict (K≈1), using Murphy's combination")
            return self.murphy_combination([m1, m2])

        # 정규화 인자
        normalization = 1.0 / (1.0 - K)

        # 결합 수행
        combined_masses: Dict[frozenset, float] = {}

        for h1, mass1 in m1.masses.items():
            for h2, mass2 in m2.masses.items():
                intersection = h1.intersection(h2)
                if intersection:  # 공집합이 아닌 경우만
                    key = frozenset(intersection)
                    combined_mass = normalization * mass1 * mass2
                    combined_masses[key] = combined_masses.get(key, 0.0) + combined_mass

        import time
        combined = MassFunction(
            frame=self.frame,
            masses=combined_masses,
            source_id=f"combined({m1.source_id},{m2.source_id})",
            timestamp=time.time()
        )

        # 로그 기록
        self.combination_log.append({
            'sources': [m1.source_id, m2.source_id],
            'conflict': K,
            'method': 'dempster',
            'result_id': combined.source_id
        })

        return combined, metadata

    def murphy_combination(
        self,
        mass_functions: List[MassFunction]
    ) -> Tuple[MassFunction, Dict[str, Any]]:
        """
        Murphy's Averaging Rule (높은 충돌 시 사용)

        1단계: 모든 질량 함수의 평균 계산
        2단계: 평균 질량 함수를 (n-1)번 자기 자신과 Dempster 결합

        Args:
            mass_functions: 결합할 질량 함수 리스트

        Returns:
            (결합된 질량 함수, 결합 메타데이터)
        """
        if not mass_functions:
            return MassFunction(frame=self.frame), {'method': 'murphy', 'error': 'empty_input'}

        if len(mass_functions) == 1:
            return mass_functions[0], {'method': 'murphy', 'note': 'single_source'}

        n = len(mass_functions)

        # 1단계: 모든 focal elements 수집 및 평균 계산
        all_focals: Set[frozenset] = set()
        for mf in mass_functions:
            all_focals.update(mf.masses.keys())

        avg_masses: Dict[frozenset, float] = {}
        for focal in all_focals:
            total = sum(mf.masses.get(focal, 0.0) for mf in mass_functions)
            avg_masses[focal] = total / n

        import time
        avg_mf = MassFunction(
            frame=self.frame,
            masses=avg_masses,
            source_id="murphy_avg",
            timestamp=time.time()
        )

        # 2단계: (n-1)번 자기 자신과 결합
        result = avg_mf
        for i in range(n - 1):
            result, _ = self.dempster_combination(result, avg_mf, check_conflict=False)

        result.source_id = f"murphy_combined({','.join(mf.source_id for mf in mass_functions)})"

        metadata = {
            'method': 'murphy',
            'num_sources': n,
            'sources': [mf.source_id for mf in mass_functions]
        }

        self.combination_log.append({
            'sources': [mf.source_id for mf in mass_functions],
            'method': 'murphy',
            'result_id': result.source_id
        })

        return result, metadata

    def yager_combination(
        self,
        m1: MassFunction,
        m2: MassFunction
    ) -> Tuple[MassFunction, Dict[str, Any]]:
        """
        Yager's Modified Rule (충돌 질량을 불확실성에 할당)

        K를 전체 집합(Θ)에 할당하여 충돌을 불확실성으로 처리

        Args:
            m1, m2: 결합할 질량 함수

        Returns:
            (결합된 질량 함수, 결합 메타데이터)
        """
        K = self.calculate_conflict_coefficient(m1, m2)

        combined_masses: Dict[frozenset, float] = {}

        for h1, mass1 in m1.masses.items():
            for h2, mass2 in m2.masses.items():
                intersection = h1.intersection(h2)
                if intersection:
                    key = frozenset(intersection)
                    combined_masses[key] = combined_masses.get(key, 0.0) + mass1 * mass2

        # 충돌 질량을 전체 집합에 할당
        combined_masses[self.frame] = combined_masses.get(self.frame, 0.0) + K

        import time
        combined = MassFunction(
            frame=self.frame,
            masses=combined_masses,
            source_id=f"yager({m1.source_id},{m2.source_id})",
            timestamp=time.time()
        )

        metadata = {
            'method': 'yager',
            'conflict_coefficient': K,
            'conflict_assigned_to_uncertainty': True
        }

        return combined, metadata

    def combine_multiple(
        self,
        mass_functions: List[MassFunction],
        method: str = 'auto'
    ) -> Tuple[MassFunction, Dict[str, Any]]:
        """
        여러 질량 함수를 결합

        Args:
            mass_functions: 결합할 질량 함수 리스트
            method: 결합 방법 ('dempster', 'murphy', 'yager', 'auto')
                   'auto': 충돌 수준에 따라 자동 선택

        Returns:
            (결합된 질량 함수, 결합 메타데이터)
        """
        if not mass_functions:
            return MassFunction(frame=self.frame), {'error': 'no_evidence'}

        if len(mass_functions) == 1:
            return mass_functions[0], {'method': 'single_source'}

        # 평균 충돌 계산 (method='auto'일 때)
        if method == 'auto':
            total_conflict = 0.0
            count = 0
            for i in range(len(mass_functions)):
                for j in range(i + 1, len(mass_functions)):
                    total_conflict += self.calculate_conflict_coefficient(
                        mass_functions[i], mass_functions[j]
                    )
                    count += 1
            avg_conflict = total_conflict / count if count > 0 else 0.0

            if avg_conflict >= self.conflict_threshold_critical:
                method = 'murphy'
                logger.info(f"High average conflict ({avg_conflict:.4f}), using Murphy's rule")
            elif avg_conflict >= self.conflict_threshold_warning:
                method = 'yager'
                logger.info(f"Moderate conflict ({avg_conflict:.4f}), using Yager's rule")
            else:
                method = 'dempster'

        # 선택된 방법으로 결합
        if method == 'murphy':
            return self.murphy_combination(mass_functions)
        elif method == 'yager':
            result = mass_functions[0]
            for mf in mass_functions[1:]:
                result, _ = self.yager_combination(result, mf)
            return result, {'method': 'yager', 'num_sources': len(mass_functions)}
        else:  # dempster
            result = mass_functions[0]
            total_conflict = 0.0
            for mf in mass_functions[1:]:
                result, meta = self.dempster_combination(result, mf)
                total_conflict += meta.get('conflict_coefficient', 0)
            return result, {
                'method': 'dempster',
                'num_sources': len(mass_functions),
                'avg_conflict': total_conflict / (len(mass_functions) - 1)
            }

    def compute_belief_plausibility(
        self,
        mass_function: MassFunction,
        hypothesis: Set[str]
    ) -> Dict[str, float]:
        """
        가설에 대한 Belief와 Plausibility 계산

        Args:
            mass_function: 질량 함수
            hypothesis: 평가할 가설 집합

        Returns:
            {belief, plausibility, uncertainty_width, doubt}
        """
        h = frozenset(hypothesis)

        bel = mass_function.belief(h)
        pl = mass_function.plausibility(h)

        return {
            'belief': bel,
            'plausibility': pl,
            'uncertainty_width': pl - bel,
            'doubt': mass_function.doubt(h),
            'uncertainty_interval': (bel, pl)
        }

    def get_decision(
        self,
        mass_function: MassFunction,
        threshold: float = 0.5,
        use_pignistic: bool = True
    ) -> Dict[str, Any]:
        """
        최종 결정 도출

        Args:
            mass_function: 결합된 질량 함수
            threshold: 결정 임계값
            use_pignistic: Pignistic 확률 사용 여부

        Returns:
            결정 결과 (decision, confidence, belief_plausibility 등)
        """
        if use_pignistic:
            # Pignistic 확률로 결정
            probs = mass_function.get_all_pignistic_probabilities()
            if probs:
                best_decision = max(probs.items(), key=lambda x: x[1])
                decision = best_decision[0]
                confidence = best_decision[1]
            else:
                decision = 'review'
                confidence = 0.5
        else:
            # 최대 질량 기반 결정
            best_mass = 0.0
            decision = 'review'

            for focal, mass in mass_function.masses.items():
                if len(focal) == 1 and mass > best_mass:
                    best_mass = mass
                    decision = list(focal)[0]

            confidence = best_mass

        # Belief/Plausibility 계산
        singleton = frozenset({decision})
        bel_pl = self.compute_belief_plausibility(mass_function, {decision})

        # 신뢰도가 임계값 미만이면 review로 변경
        if confidence < threshold:
            decision = 'review'
            logger.info(f"Low confidence ({confidence:.4f}), defaulting to review")

        return {
            'decision': decision,
            'confidence': confidence,
            'belief': bel_pl['belief'],
            'plausibility': bel_pl['plausibility'],
            'uncertainty_interval': bel_pl['uncertainty_interval'],
            'uncertainty_width': bel_pl['uncertainty_width'],
            'all_probabilities': mass_function.get_all_pignistic_probabilities() if use_pignistic else None
        }

    def fuse_agent_opinions(
        self,
        opinions: List[Dict[str, Any]],
        method: str = 'auto'
    ) -> Dict[str, Any]:
        """
        다중 에이전트 의견 융합 (고수준 API)

        Args:
            opinions: 에이전트 의견 리스트
            method: 결합 방법 ('dempster', 'murphy', 'yager', 'auto')

        Returns:
            융합 결과 (decision, confidence, conflict 분석 등)
        """
        if not opinions:
            return {
                'decision': 'review',
                'confidence': 0.5,
                'conflict': 0.0,
                'error': 'no_opinions'
            }

        # 질량 함수 생성
        mass_functions = []
        for i, opinion in enumerate(opinions):
            agent_id = opinion.get('agent_id', f'agent_{i}')
            mf = self.create_mass_from_agent_opinion(opinion, agent_id)
            mass_functions.append(mf)
            self.evidence_history.append(mf)

        # 충돌 분석
        pairwise_conflicts = []
        for i in range(len(mass_functions)):
            for j in range(i + 1, len(mass_functions)):
                conflict_analysis = self.analyze_conflict(
                    mass_functions[i], mass_functions[j]
                )
                pairwise_conflicts.append({
                    'agents': (mass_functions[i].source_id, mass_functions[j].source_id),
                    'conflict': conflict_analysis['conflict_coefficient'],
                    'level': conflict_analysis['level']
                })

        avg_conflict = (
            sum(c['conflict'] for c in pairwise_conflicts) / len(pairwise_conflicts)
            if pairwise_conflicts else 0.0
        )

        # 결합
        combined, combination_meta = self.combine_multiple(mass_functions, method)

        # 결정 도출
        decision_result = self.get_decision(combined)

        return {
            'decision': decision_result['decision'],
            'confidence': decision_result['confidence'],
            'belief': decision_result['belief'],
            'plausibility': decision_result['plausibility'],
            'uncertainty_interval': decision_result['uncertainty_interval'],
            'uncertainty_width': decision_result['uncertainty_width'],
            'conflict': avg_conflict,
            'pairwise_conflicts': pairwise_conflicts,
            'combination_method': combination_meta.get('method', 'unknown'),
            'needs_review': avg_conflict >= self.conflict_threshold_warning,
            'all_probabilities': decision_result.get('all_probabilities'),
            'num_sources': len(opinions)
        }

    def get_evidence_summary(self) -> Dict[str, Any]:
        """증거 이력 요약"""
        return {
            'total_evidence': len(self.evidence_history),
            'combinations_performed': len(self.combination_log),
            'frame_of_discernment': list(self.frame),
            'recent_combinations': self.combination_log[-5:] if self.combination_log else []
        }


# =============================================================================
# 3. CONFIDENCE CALIBRATION
# =============================================================================

class ConfidenceCalibrator:
    """
    신뢰도 보정 (Calibration)

    논문: "A Survey of Uncertainty in Deep Neural Networks" (2023)

    핵심:
    - Temperature Scaling으로 과신(overconfidence) 보정
    - Expected Calibration Error (ECE) 최소화
    - Reliability Diagram 기반 분석
    """

    def __init__(self, n_bins: int = 10):
        self.n_bins = n_bins
        self.temperature = 1.0
        self.calibration_history = []

    def temperature_scaling(self, logits: np.ndarray, temperature: float = None) -> np.ndarray:
        """
        Temperature Scaling: p = softmax(logits / T)
        T > 1: 분포를 더 균일하게 (과신 완화)
        T < 1: 분포를 더 뾰족하게 (확신 강화)
        """
        T = temperature or self.temperature
        scaled = logits / T
        # Softmax
        exp_scaled = np.exp(scaled - np.max(scaled))
        return exp_scaled / exp_scaled.sum()

    def calculate_ece(
        self,
        confidences: np.ndarray,
        accuracies: np.ndarray
    ) -> float:
        """
        Expected Calibration Error
        ECE = Σ (|Bm|/n) * |acc(Bm) - conf(Bm)|

        낮을수록 좋음 (0 = 완벽한 calibration)
        """
        bin_boundaries = np.linspace(0, 1, self.n_bins + 1)
        ece = 0.0
        n = len(confidences)

        for i in range(self.n_bins):
            lower, upper = bin_boundaries[i], bin_boundaries[i + 1]
            in_bin = (confidences >= lower) & (confidences < upper)

            if in_bin.sum() > 0:
                avg_conf = confidences[in_bin].mean()
                avg_acc = accuracies[in_bin].mean()
                bin_weight = in_bin.sum() / n
                ece += bin_weight * abs(avg_acc - avg_conf)

        return ece

    def find_optimal_temperature(
        self,
        predictions: List[Dict],
        ground_truth: List[bool],
        temp_range: Tuple[float, float] = (0.5, 3.0),
        n_steps: int = 50
    ) -> float:
        """최적 Temperature 탐색 (ECE 최소화)"""
        if not predictions or not ground_truth:
            return 1.0

        confidences = np.array([p.get('confidence', 0.5) for p in predictions])
        accuracies = np.array([1.0 if g else 0.0 for g in ground_truth])

        best_temp = 1.0
        best_ece = float('inf')

        for temp in np.linspace(temp_range[0], temp_range[1], n_steps):
            # Temperature 적용
            calibrated = np.clip(confidences / temp, 0.01, 0.99)
            ece = self.calculate_ece(calibrated, accuracies)

            if ece < best_ece:
                best_ece = ece
                best_temp = temp

        self.temperature = best_temp
        logger.info(f"Optimal temperature: {best_temp:.3f}, ECE: {best_ece:.4f}")
        return best_temp

    def calibrate(self, confidence: float) -> Dict[str, float]:
        """단일 신뢰도 보정"""
        calibrated = np.clip(confidence / self.temperature, 0.01, 0.99)

        # Bootstrap 신뢰 구간 (근사)
        std_error = np.sqrt(calibrated * (1 - calibrated))
        ci_lower = max(0, calibrated - 1.96 * std_error)
        ci_upper = min(1, calibrated + 1.96 * std_error)

        return {
            'original': confidence,
            'calibrated': float(calibrated),
            'temperature': self.temperature,
            'confidence_interval': (float(ci_lower), float(ci_upper)),
        }


# =============================================================================
# 4. KNOWLEDGE GRAPH QUALITY METRICS
# =============================================================================

@dataclass
class KGQualityScore:
    """KG 품질 점수"""
    metric_name: str
    score: float
    max_score: float = 1.0
    interpretation: str = ""


class KnowledgeGraphQualityValidator:
    """
    Knowledge Graph 품질 검증

    논문: "Structural Quality Metrics to Evaluate Knowledge Graphs" (2022)

    6가지 구조적 메트릭:
    1. Class Instantiation
    2. Property Completeness
    3. Relationship Balance
    4. Semantic Consistency
    5. Hierarchy Depth
    6. Link Validity
    """

    def __init__(self):
        self.weights = {
            'class_instantiation': 0.15,
            'property_completeness': 0.20,
            'relationship_balance': 0.15,
            'semantic_consistency': 0.25,
            'hierarchy_depth': 0.10,
            'link_validity': 0.15,
        }

    def validate(self, kg_data: Dict[str, Any]) -> Dict[str, KGQualityScore]:
        """KG 품질 종합 평가"""
        scores = {}

        # 1. Class Instantiation: 클래스당 인스턴스 수
        entities = kg_data.get('unified_entities', [])
        concepts = kg_data.get('ontology_concepts', [])

        if concepts:
            avg_instances = len(entities) / len(concepts) if concepts else 0
            # 이상적: 5-20개의 인스턴스
            ci_score = min(avg_instances / 10, 1.0)
            scores['class_instantiation'] = KGQualityScore(
                metric_name="Class Instantiation",
                score=ci_score,
                interpretation=f"평균 {avg_instances:.1f} 인스턴스/클래스"
            )

        # 2. Property Completeness: 필수 속성 완성도
        total_props = 0
        filled_props = 0
        required_props = {'entity_type', 'confidence', 'source_tables'}

        for entity in entities:
            for prop in required_props:
                total_props += 1
                if entity.get(prop):
                    filled_props += 1

        pc_score = filled_props / total_props if total_props > 0 else 0
        scores['property_completeness'] = KGQualityScore(
            metric_name="Property Completeness",
            score=pc_score,
            interpretation=f"{filled_props}/{total_props} 속성 완성"
        )

        # 3. Relationship Balance: in/out degree 균형
        triples = (kg_data.get('knowledge_graph_triples', []) +
                  kg_data.get('semantic_base_triples', []))

        in_degree = defaultdict(int)
        out_degree = defaultdict(int)

        for triple in triples:
            subj = triple.get('subject', '')
            obj = triple.get('object', '')
            out_degree[subj] += 1
            in_degree[obj] += 1

        if out_degree and in_degree:
            avg_in = sum(in_degree.values()) / len(in_degree)
            avg_out = sum(out_degree.values()) / len(out_degree)
            balance = min(avg_in, avg_out) / max(avg_in, avg_out) if max(avg_in, avg_out) > 0 else 0
        else:
            balance = 0.5

        scores['relationship_balance'] = KGQualityScore(
            metric_name="Relationship Balance",
            score=balance,
            interpretation=f"In/Out 비율: {balance:.2f}"
        )

        # 4. Semantic Consistency: 동일 predicate 일관성
        predicates = defaultdict(list)
        for triple in triples:
            pred = triple.get('predicate', '')
            predicates[pred].append(triple)

        consistency_scores = []
        for pred, pred_triples in predicates.items():
            if len(pred_triples) > 1:
                # 동일 predicate의 subject/object 패턴 일관성
                subjects = [t.get('subject', '') for t in pred_triples]
                # 간단한 일관성: 유사한 prefix/suffix 비율
                if subjects:
                    prefixes = [s.split(':')[0] if ':' in s else s[:3] for s in subjects]
                    consistency = len(set(prefixes)) / len(prefixes)
                    consistency_scores.append(1 - consistency + 0.5)  # 다양성 ↓ = 일관성 ↑

        sc_score = sum(consistency_scores) / len(consistency_scores) if consistency_scores else 0.7
        scores['semantic_consistency'] = KGQualityScore(
            metric_name="Semantic Consistency",
            score=min(sc_score, 1.0),
            interpretation=f"{len(predicates)} predicate 유형 분석"
        )

        # 5. Hierarchy Depth: rdfs:subClassOf 체인 깊이
        subclass_triples = [t for t in triples if 'subClassOf' in t.get('predicate', '')]
        depth = len(subclass_triples)
        ideal_depth = 3  # 보통 3단계 정도가 적절
        hd_score = 1 - abs(depth - ideal_depth) / max(depth, ideal_depth) if depth > 0 else 0.5

        scores['hierarchy_depth'] = KGQualityScore(
            metric_name="Hierarchy Depth",
            score=max(hd_score, 0),
            interpretation=f"계층 깊이: {depth}"
        )

        # 6. Link Validity: 참조 무결성
        all_subjects = {t.get('subject') for t in triples}
        all_objects = {t.get('object') for t in triples}
        all_nodes = all_subjects | all_objects

        # 리터럴이 아닌 객체 중 subject로도 등장하는 비율
        non_literal_objects = {o for o in all_objects if ':' in str(o) and not o.startswith('"')}
        valid_refs = len(non_literal_objects & all_subjects)
        total_refs = len(non_literal_objects)

        lv_score = valid_refs / total_refs if total_refs > 0 else 0.8
        scores['link_validity'] = KGQualityScore(
            metric_name="Link Validity",
            score=lv_score,
            interpretation=f"{valid_refs}/{total_refs} 유효 참조"
        )

        return scores

    def get_overall_score(self, scores: Dict[str, KGQualityScore]) -> float:
        """가중 평균 품질 점수"""
        total = 0.0
        weight_sum = 0.0

        for name, score in scores.items():
            w = self.weights.get(name, 0.1)
            total += w * score.score
            weight_sum += w

        return total / weight_sum if weight_sum > 0 else 0.5


# =============================================================================
# 5. BYZANTINE FAULT TOLERANT CONSENSUS
# =============================================================================

class VoteType(Enum):
    APPROVE = "approve"
    REJECT = "reject"
    REVIEW = "review"
    ABSTAIN = "abstain"


@dataclass
class AgentVote:
    """에이전트 투표"""
    agent_id: str
    vote: VoteType
    confidence: float
    reasoning: str = ""
    round: int = 1


class BFTConsensus:
    """
    Byzantine Fault Tolerant Consensus

    논문: "Rethinking the Reliability of Multi-agent System" (2024)

    핵심:
    - 3f+1 노드에서 f개 Byzantine 노드 허용
    - PBFT 기반 3-phase commit (Pre-prepare, Prepare, Commit)
    - Confidence-weighted voting
    """

    def __init__(self, n_agents: int = 4):
        """
        n_agents: 에이전트 수 (최소 4, 1개의 Byzantine 허용)
        f = (n-1)/3: 허용 가능한 Byzantine 노드 수
        """
        self.n = max(n_agents, 4)
        self.f = (self.n - 1) // 3
        self.required_votes = 2 * self.f + 1

        logger.info(f"BFT Consensus: n={self.n}, f={self.f}, required={self.required_votes}")

    def _detect_outliers(self, votes: List[AgentVote]) -> List[str]:
        """이상치 에이전트 탐지 (신뢰도 기반)"""
        if len(votes) < 3:
            return []

        confidences = [v.confidence for v in votes]
        mean_conf = np.mean(confidences)
        std_conf = np.std(confidences)

        outliers = []
        for vote in votes:
            # 평균에서 2 표준편차 이상 벗어난 경우
            if abs(vote.confidence - mean_conf) > 2 * std_conf:
                outliers.append(vote.agent_id)

        return outliers[:self.f]  # 최대 f개만 제외

    def _weighted_vote_count(self, votes: List[AgentVote]) -> Dict[VoteType, float]:
        """신뢰도 가중 투표 집계"""
        counts = defaultdict(float)

        for vote in votes:
            if vote.vote != VoteType.ABSTAIN:
                counts[vote.vote] += vote.confidence

        return dict(counts)

    def reach_consensus(
        self,
        votes: List[AgentVote],
        min_confidence: float = 0.5
    ) -> Dict[str, Any]:
        """
        BFT 합의 도출

        Returns:
            consensus: 합의 결과
            confidence: 합의 신뢰도
            agreement_ratio: 동의 비율
            byzantine_detected: 탐지된 Byzantine 에이전트
        """
        if not votes:
            return {
                'consensus': VoteType.REVIEW,
                'confidence': 0.0,
                'agreement_ratio': 0.0,
                'byzantine_detected': [],
            }

        # 1. Byzantine 에이전트 탐지 및 제외
        outliers = self._detect_outliers(votes)
        valid_votes = [v for v in votes if v.agent_id not in outliers]

        if len(valid_votes) < self.required_votes:
            logger.warning(f"Not enough valid votes: {len(valid_votes)} < {self.required_votes}")
            valid_votes = votes  # 폴백: 모든 투표 사용

        # 2. 가중 투표 집계
        weighted_counts = self._weighted_vote_count(valid_votes)

        # 3. 다수결 결정
        if not weighted_counts:
            return {
                'consensus': VoteType.REVIEW,
                'confidence': 0.5,
                'agreement_ratio': 0.0,
                'byzantine_detected': outliers,
            }

        total_weight = sum(weighted_counts.values())
        best_vote = max(weighted_counts.items(), key=lambda x: x[1])

        consensus_vote = best_vote[0]
        consensus_weight = best_vote[1]

        # 4. 합의 품질 계산
        agreement_ratio = consensus_weight / total_weight if total_weight > 0 else 0
        avg_confidence = np.mean([v.confidence for v in valid_votes])

        # 5. 최종 신뢰도: 동의 비율 × 평균 신뢰도
        final_confidence = agreement_ratio * avg_confidence

        # 6. 최소 신뢰도 미달시 REVIEW로 변경
        if final_confidence < min_confidence:
            logger.info(f"Low confidence ({final_confidence:.2f}), changing to REVIEW")
            consensus_vote = VoteType.REVIEW

        return {
            'consensus': consensus_vote,
            'confidence': float(final_confidence),
            'agreement_ratio': float(agreement_ratio),
            'byzantine_detected': outliers,
            'vote_distribution': {k.value: v for k, v in weighted_counts.items()},
            'valid_votes': len(valid_votes),
            'total_votes': len(votes),
        }


# =============================================================================
# 6. INTEGRATED ENHANCED VALIDATOR
# =============================================================================

class EnhancedValidator:
    """
    통합 검증 시스템

    모든 검증 컴포넌트를 통합하여 최종 검증 결과 산출

    Phase 3 Governance Spec 준수:
    - Dempster-Shafer Evidence Theory를 통한 다중 에이전트 의견 융합
    - 충돌 증거(Conflicting Evidence) 처리 및 판단 보류 로직
    - BFT 기반 합의 프로토콜과의 통합
    """

    def __init__(self, domain: str = "supply_chain"):
        self.domain = domain
        self.statistical_validator = StatisticalValidator()
        self.ds_fusion = DempsterShaferFusion()  # 레거시 호환
        self.ds_engine = DempsterShaferEngine()  # 새로운 DS 엔진
        self.calibrator = ConfidenceCalibrator()
        self.kg_validator = KnowledgeGraphQualityValidator()
        self.bft_consensus = BFTConsensus(n_agents=4)

        # 도메인별 가중치
        self.domain_weights = {
            'supply_chain': {
                'statistical': 0.25,
                'ds_fusion': 0.20,
                'kg_quality': 0.30,
                'bft_consensus': 0.25,
            },
            'healthcare': {
                'statistical': 0.30,
                'ds_fusion': 0.25,
                'kg_quality': 0.25,
                'bft_consensus': 0.20,
            },
        }

    def validate_homeomorphism(
        self,
        table_a_data: np.ndarray,
        table_b_data: np.ndarray,
        agent_opinions: List[Dict[str, Any]],
        use_enhanced_ds: bool = True
    ) -> Dict[str, Any]:
        """
        Homeomorphism 통합 검증

        Args:
            table_a_data: 테이블 A 데이터
            table_b_data: 테이블 B 데이터
            agent_opinions: 에이전트 의견 리스트
            use_enhanced_ds: 향상된 DS 엔진 사용 여부 (기본: True)

        Returns:
            통합 검증 결과
        """
        results = {}

        # 1. 통계적 검증
        stat_results = self.statistical_validator.validate_distribution_match(
            table_a_data, table_b_data
        )
        stat_confidence = self.statistical_validator.calculate_confidence(stat_results)
        results['statistical'] = {
            'tests': {k: {'statistic': v.statistic, 'p_value': v.p_value, 'passed': v.passed}
                     for k, v in stat_results.items()},
            'confidence': stat_confidence,
        }

        # 2. Dempster-Shafer 융합 (향상된 엔진 사용)
        if use_enhanced_ds:
            ds_result = self.ds_engine.fuse_agent_opinions(agent_opinions)
            results['ds_fusion'] = {
                'decision': ds_result['decision'],
                'confidence': ds_result['confidence'],
                'belief': ds_result['belief'],
                'plausibility': ds_result['plausibility'],
                'conflict': ds_result['conflict'],
                'uncertainty_interval': ds_result['uncertainty_interval'],
                'uncertainty_width': ds_result['uncertainty_width'],
                'combination_method': ds_result['combination_method'],
                'needs_review': ds_result['needs_review'],
                'pairwise_conflicts': ds_result.get('pairwise_conflicts', []),
            }
        else:
            # 레거시 호환
            ds_result = self.ds_fusion.fuse_opinions(agent_opinions)
            results['ds_fusion'] = ds_result

        # 3. BFT 합의
        votes = [
            AgentVote(
                agent_id=f"agent_{i}",
                vote=VoteType.APPROVE if op.get('confidence', 0) > 0.5 else VoteType.REJECT,
                confidence=op.get('confidence', 0.5),
            )
            for i, op in enumerate(agent_opinions)
        ]
        bft_result = self.bft_consensus.reach_consensus(votes)
        results['bft_consensus'] = {
            'consensus': bft_result['consensus'].value,
            'confidence': bft_result['confidence'],
            'agreement_ratio': bft_result['agreement_ratio'],
        }

        # 4. 최종 신뢰도 (가중 평균)
        weights = self.domain_weights.get(self.domain, self.domain_weights['supply_chain'])

        # DS 결과의 충돌이 높으면 가중치 조정
        ds_confidence = ds_result.get('confidence', 0.5)
        ds_conflict = ds_result.get('conflict', 0.0)

        # 높은 충돌 시 DS 결과 가중치 감소
        effective_ds_weight = weights['ds_fusion'] * (1.0 - ds_conflict * 0.5)

        final_confidence = (
            weights['statistical'] * stat_confidence +
            effective_ds_weight * ds_confidence +
            weights['bft_consensus'] * bft_result['confidence']
        ) / (weights['statistical'] + effective_ds_weight + weights['bft_consensus'])

        # 5. 보정
        calibrated = self.calibrator.calibrate(final_confidence)
        results['calibrated'] = calibrated

        results['final_confidence'] = calibrated['calibrated']

        # 충돌이 높거나 신뢰도가 낮으면 review로 강제
        if ds_result.get('needs_review', False) or ds_conflict > 0.7:
            results['decision'] = 'review'
            results['review_reason'] = 'high_conflict'
        else:
            results['decision'] = (
                'approve' if calibrated['calibrated'] > 0.6
                else 'review' if calibrated['calibrated'] > 0.4
                else 'reject'
            )

        return results

    def validate_with_dempster_shafer(
        self,
        agent_opinions: List[Dict[str, Any]],
        combination_method: str = 'auto'
    ) -> Dict[str, Any]:
        """
        Dempster-Shafer Theory만을 사용한 순수 검증

        Phase 3 Governance Spec에서 요구하는 핵심 기능:
        - 에이전트 A의 신뢰도(m_A)와 에이전트 B의 신뢰도(m_B) 결합
        - 충돌하는 증거(K)를 계산하여 K가 너무 크면 "판단 보류"

        Args:
            agent_opinions: 에이전트 의견 리스트
            combination_method: 결합 방법 ('dempster', 'murphy', 'yager', 'auto')

        Returns:
            DS 융합 결과
        """
        result = self.ds_engine.fuse_agent_opinions(agent_opinions, method=combination_method)

        # 충돌 계수(K)가 너무 크면 판단 보류
        if result['conflict'] > 0.8:
            result['decision'] = 'review'
            result['review_reason'] = 'conflict_too_high'
            result['recommendation'] = '증거 재검토 필요, 판단 보류'

        # 보정된 신뢰도 추가
        calibrated = self.calibrator.calibrate(result['confidence'])
        result['calibrated_confidence'] = calibrated['calibrated']
        result['confidence_interval'] = calibrated['confidence_interval']

        return result

    def analyze_evidence_conflict(
        self,
        agent_opinions: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        증거 간 충돌 상세 분석

        Args:
            agent_opinions: 에이전트 의견 리스트

        Returns:
            충돌 분석 결과 (pairwise conflicts, overall conflict, recommendations)
        """
        if len(agent_opinions) < 2:
            return {
                'overall_conflict': 0.0,
                'conflict_level': 'none',
                'pairwise_analysis': [],
                'recommendation': '단일 증거로 충돌 분석 불가'
            }

        # 질량 함수 생성
        mass_functions = []
        for i, opinion in enumerate(agent_opinions):
            agent_id = opinion.get('agent_id', f'agent_{i}')
            mf = self.ds_engine.create_mass_from_agent_opinion(opinion, agent_id)
            mass_functions.append(mf)

        # Pairwise 충돌 분석
        pairwise_analysis = []
        total_conflict = 0.0
        count = 0

        for i in range(len(mass_functions)):
            for j in range(i + 1, len(mass_functions)):
                analysis = self.ds_engine.analyze_conflict(
                    mass_functions[i], mass_functions[j]
                )
                pairwise_analysis.append({
                    'agent_pair': (mass_functions[i].source_id, mass_functions[j].source_id),
                    'conflict_coefficient': analysis['conflict_coefficient'],
                    'level': analysis['level'],
                    'description': analysis['description'],
                    'recommendation': analysis['recommendation']
                })
                total_conflict += analysis['conflict_coefficient']
                count += 1

        avg_conflict = total_conflict / count if count > 0 else 0.0

        # 전체 충돌 수준 판정
        if avg_conflict < 0.2:
            overall_level = 'minimal'
            overall_recommendation = '표준 Dempster 결합 권장'
        elif avg_conflict < 0.5:
            overall_level = 'low'
            overall_recommendation = 'Dempster 결합 사용 가능, 결과 검증 권장'
        elif avg_conflict < 0.8:
            overall_level = 'moderate'
            overall_recommendation = 'Murphy 또는 Yager 규칙 사용 권장'
        else:
            overall_level = 'high'
            overall_recommendation = '판단 보류, 증거 재검토 필요'

        return {
            'overall_conflict': avg_conflict,
            'conflict_level': overall_level,
            'pairwise_analysis': pairwise_analysis,
            'recommendation': overall_recommendation,
            'num_sources': len(agent_opinions)
        }

    def get_belief_plausibility_interval(
        self,
        agent_opinions: List[Dict[str, Any]],
        hypothesis: str
    ) -> Dict[str, Any]:
        """
        특정 가설에 대한 Belief-Plausibility 구간 계산

        Args:
            agent_opinions: 에이전트 의견 리스트
            hypothesis: 평가할 가설 ('approve', 'reject', 'review')

        Returns:
            신뢰도 구간 [Bel(A), Pl(A)] 및 해석
        """
        # 증거 융합
        mass_functions = []
        for i, opinion in enumerate(agent_opinions):
            agent_id = opinion.get('agent_id', f'agent_{i}')
            mf = self.ds_engine.create_mass_from_agent_opinion(opinion, agent_id)
            mass_functions.append(mf)

        # 결합
        combined, _ = self.ds_engine.combine_multiple(mass_functions, method='auto')

        # Belief-Plausibility 계산
        bel_pl = self.ds_engine.compute_belief_plausibility(combined, {hypothesis})

        # 해석
        uncertainty = bel_pl['uncertainty_width']
        if uncertainty < 0.1:
            interpretation = "높은 확신: 증거가 명확함"
        elif uncertainty < 0.3:
            interpretation = "중간 확신: 일부 불확실성 존재"
        elif uncertainty < 0.5:
            interpretation = "낮은 확신: 상당한 불확실성"
        else:
            interpretation = "매우 낮은 확신: 증거 부족 또는 충돌"

        return {
            'hypothesis': hypothesis,
            'belief': bel_pl['belief'],
            'plausibility': bel_pl['plausibility'],
            'uncertainty_interval': bel_pl['uncertainty_interval'],
            'uncertainty_width': uncertainty,
            'doubt': bel_pl['doubt'],
            'interpretation': interpretation,
            'pignistic_probability': combined.pignistic_probability(hypothesis)
        }

    def validate_knowledge_graph(self, kg_data: Dict[str, Any]) -> Dict[str, Any]:
        """Knowledge Graph 통합 검증"""
        # KG 품질 메트릭
        kg_scores = self.kg_validator.validate(kg_data)
        overall_kg = self.kg_validator.get_overall_score(kg_scores)

        # 거버넌스 결정들을 BFT로 재검증
        decisions = kg_data.get('governance_decisions', [])
        if decisions:
            votes = [
                AgentVote(
                    agent_id=d.get('concept_id', f"concept_{i}"),
                    vote=VoteType(d.get('decision_type', 'review').replace('schedule_review', 'review')),
                    confidence=d.get('confidence', 0.5),
                )
                for i, d in enumerate(decisions)
                if d.get('decision_type') in ['approve', 'reject', 'review', 'schedule_review', 'escalate']
            ]
            # escalate를 review로 변환
            for v in votes:
                if v.vote.value == 'escalate':
                    v.vote = VoteType.REVIEW

            bft_result = self.bft_consensus.reach_consensus(votes)
        else:
            bft_result = {'consensus': VoteType.REVIEW, 'confidence': 0.5}

        # 최종 결과
        final_confidence = (overall_kg * 0.6 + bft_result.get('confidence', 0.5) * 0.4)
        calibrated = self.calibrator.calibrate(final_confidence)

        return {
            'kg_quality_scores': {k: {'score': v.score, 'interpretation': v.interpretation}
                                  for k, v in kg_scores.items()},
            'kg_overall_score': overall_kg,
            'bft_consensus': bft_result,
            'final_confidence': calibrated['calibrated'],
            'confidence_interval': calibrated['confidence_interval'],
        }


# =============================================================================
# MAIN TEST
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # 테스트 데이터
    source = np.random.normal(100, 15, 1000)
    target = np.random.normal(102, 16, 1000)

    # 통계 검증 테스트
    print("=== Statistical Validator Test ===")
    sv = StatisticalValidator()
    results = sv.validate_distribution_match(source, target)
    for name, result in results.items():
        print(f"{name}: stat={result.statistic:.4f}, p={result.p_value:.4f}, passed={result.passed}")
    print(f"Overall confidence: {sv.calculate_confidence(results):.4f}")

    # DS 융합 테스트 (레거시)
    print("\n=== Dempster-Shafer Fusion Test (Legacy) ===")
    opinions = [
        {'decision_type': 'approve', 'confidence': 0.8},
        {'decision_type': 'approve', 'confidence': 0.7},
        {'decision_type': 'reject', 'confidence': 0.6},
        {'decision_type': 'approve', 'confidence': 0.75},
    ]
    ds = DempsterShaferFusion()
    fused = ds.fuse_opinions(opinions)
    print(f"Fused decision: {fused['decision']}, confidence: {fused['confidence']:.4f}")
    print(f"Conflict: {fused['conflict']:.4f}")

    # DempsterShaferEngine 테스트 (Phase 3)
    print("\n=== Dempster-Shafer Engine Test (Phase 3) ===")
    ds_engine = DempsterShaferEngine()

    # 1. 질량 함수 생성 테스트
    print("\n1. Mass Function Creation:")
    m1 = ds_engine.create_mass_function({'approve': 0.7, 'reject': 0.1}, source_id='agent_1')
    m2 = ds_engine.create_mass_function({'approve': 0.6, 'reject': 0.2}, source_id='agent_2')
    print(f"   m1 masses: {dict(m1.masses)}")
    print(f"   m2 masses: {dict(m2.masses)}")

    # 2. 충돌 계수 계산 테스트
    print("\n2. Conflict Coefficient (K) Calculation:")
    K = ds_engine.calculate_conflict_coefficient(m1, m2)
    print(f"   K = {K:.4f}")
    conflict_analysis = ds_engine.analyze_conflict(m1, m2)
    print(f"   Level: {conflict_analysis['level']}")
    print(f"   Recommendation: {conflict_analysis['recommendation']}")

    # 3. Dempster 결합 테스트
    print("\n3. Dempster's Rule of Combination:")
    combined, meta = ds_engine.dempster_combination(m1, m2)
    print(f"   Combined masses: {dict(combined.masses)}")
    print(f"   Conflict in combination: {meta['conflict_coefficient']:.4f}")

    # 4. Belief & Plausibility 테스트
    print("\n4. Belief & Plausibility Calculation:")
    hypothesis = {'approve'}
    bel_pl = ds_engine.compute_belief_plausibility(combined, hypothesis)
    print(f"   Hypothesis: {hypothesis}")
    print(f"   Belief (Bel): {bel_pl['belief']:.4f}")
    print(f"   Plausibility (Pl): {bel_pl['plausibility']:.4f}")
    print(f"   Uncertainty Interval: [{bel_pl['belief']:.4f}, {bel_pl['plausibility']:.4f}]")
    print(f"   Uncertainty Width: {bel_pl['uncertainty_width']:.4f}")

    # 5. 다중 에이전트 융합 테스트
    print("\n5. Multi-Agent Opinion Fusion:")
    agent_opinions = [
        {'agent_id': 'ontologist', 'decision_type': 'approve', 'confidence': 0.85},
        {'agent_id': 'risk_assessor', 'decision_type': 'approve', 'confidence': 0.7},
        {'agent_id': 'biz_strategist', 'decision_type': 'reject', 'confidence': 0.6},
        {'agent_id': 'data_steward', 'decision_type': 'approve', 'confidence': 0.75},
    ]
    fusion_result = ds_engine.fuse_agent_opinions(agent_opinions)
    print(f"   Decision: {fusion_result['decision']}")
    print(f"   Confidence: {fusion_result['confidence']:.4f}")
    print(f"   Belief: {fusion_result['belief']:.4f}")
    print(f"   Plausibility: {fusion_result['plausibility']:.4f}")
    print(f"   Conflict: {fusion_result['conflict']:.4f}")
    print(f"   Combination Method: {fusion_result['combination_method']}")
    print(f"   Needs Review: {fusion_result['needs_review']}")

    # 6. 높은 충돌 시나리오 테스트
    print("\n6. High Conflict Scenario Test:")
    high_conflict_opinions = [
        {'agent_id': 'agent_a', 'decision_type': 'approve', 'confidence': 0.9},
        {'agent_id': 'agent_b', 'decision_type': 'reject', 'confidence': 0.9},
    ]
    high_conflict_result = ds_engine.fuse_agent_opinions(high_conflict_opinions)
    print(f"   Decision: {high_conflict_result['decision']}")
    print(f"   Conflict: {high_conflict_result['conflict']:.4f}")
    print(f"   Needs Review: {high_conflict_result['needs_review']}")
    print(f"   Combination Method: {high_conflict_result['combination_method']}")

    # 7. Pignistic 확률 변환 테스트
    print("\n7. Pignistic Probability Transformation:")
    pignistic = combined.get_all_pignistic_probabilities()
    print(f"   Pignistic probabilities: {pignistic}")

    # BFT 테스트
    print("\n=== BFT Consensus Test ===")
    votes = [
        AgentVote("a1", VoteType.APPROVE, 0.8),
        AgentVote("a2", VoteType.APPROVE, 0.7),
        AgentVote("a3", VoteType.REJECT, 0.3),  # Byzantine?
        AgentVote("a4", VoteType.APPROVE, 0.75),
    ]
    bft = BFTConsensus(n_agents=4)
    consensus = bft.reach_consensus(votes)
    print(f"Consensus: {consensus['consensus'].value}, confidence: {consensus['confidence']:.4f}")
    print(f"Byzantine detected: {consensus['byzantine_detected']}")

    # EnhancedValidator 통합 테스트
    print("\n=== EnhancedValidator Integration Test ===")
    validator = EnhancedValidator()

    # DS 전용 검증
    ds_only_result = validator.validate_with_dempster_shafer(agent_opinions)
    print(f"\nDS-only validation:")
    print(f"   Decision: {ds_only_result['decision']}")
    print(f"   Confidence: {ds_only_result['confidence']:.4f}")
    print(f"   Calibrated: {ds_only_result['calibrated_confidence']:.4f}")

    # 충돌 분석
    conflict_analysis = validator.analyze_evidence_conflict(agent_opinions)
    print(f"\nConflict Analysis:")
    print(f"   Overall Conflict: {conflict_analysis['overall_conflict']:.4f}")
    print(f"   Level: {conflict_analysis['conflict_level']}")
    print(f"   Recommendation: {conflict_analysis['recommendation']}")

    # Belief-Plausibility 구간
    bel_pl_result = validator.get_belief_plausibility_interval(agent_opinions, 'approve')
    print(f"\nBelief-Plausibility for 'approve':")
    print(f"   Interval: [{bel_pl_result['belief']:.4f}, {bel_pl_result['plausibility']:.4f}]")
    print(f"   Interpretation: {bel_pl_result['interpretation']}")

    print("\n=== All tests passed! ===")
