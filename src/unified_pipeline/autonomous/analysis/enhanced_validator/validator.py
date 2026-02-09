"""
Enhanced Validator - Integrated validation orchestrator

통합 검증 시스템: 모든 검증 컴포넌트를 통합하여 최종 검증 결과 산출

Phase 3 Governance Spec 준수:
- Dempster-Shafer Evidence Theory를 통한 다중 에이전트 의견 융합
- 충돌 증거(Conflicting Evidence) 처리 및 판단 보류 로직
- BFT 기반 합의 프로토콜과의 통합
"""

import logging
from typing import Any, Dict, List

import numpy as np

from .models import AgentVote, VoteType
from .statistical import StatisticalValidator
from .dempster_shafer import DempsterShaferFusion, DempsterShaferEngine
from .calibration import ConfidenceCalibrator
from .kg_validator import KnowledgeGraphQualityValidator
from .bft_consensus import BFTConsensus

logger = logging.getLogger(__name__)


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
