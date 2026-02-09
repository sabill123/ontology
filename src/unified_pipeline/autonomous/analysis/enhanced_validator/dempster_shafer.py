"""
Dempster-Shafer Evidence Theory

Includes:
- MassFunction: Core mass function data structure
- BPA: Basic Probability Assignment (legacy compatibility)
- DempsterShaferFusion: Multi-source opinion fusion
- DempsterShaferEngine: Full DS theory implementation for Phase 3 Governance

References:
- "A Mathematical Theory of Evidence" (Shafer, 1976)
- "Information fusion for large-scale multi-source data
   based on the Dempster-Shafer evidence theory" (2024)
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Set, Tuple

logger = logging.getLogger(__name__)


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
