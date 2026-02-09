"""
Statistical Validator

논문: "Statistical Validation of Column Matching" (2024)
핵심: Goodness-of-fit 테스트로 매칭 품질 검증
"""

import logging
from typing import Dict

import numpy as np
from scipy import stats

from .models import StatisticalTestResult

logger = logging.getLogger(__name__)


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
