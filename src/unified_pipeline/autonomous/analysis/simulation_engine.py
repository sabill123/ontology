"""
Simulation Engine for Data-Driven Insights (v10.1)

시뮬레이션 엔진: LLM이 더 정확한 판단을 내리기 위한 내부 도구
- 실제 데이터로 what-if 분석 수행
- 시뮬레이션 결과는 LLM에게만 전달 (사용자에게 노출 X)
- LLM은 시뮬레이션 결과를 바탕으로 비즈니스 인사이트 도출
"""

import logging
import uuid
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Any, Optional, Tuple
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class SimulationExperiment:
    """하나의 시뮬레이션 실험 결과"""
    experiment_id: str
    experiment_type: str  # "segmentation", "threshold", "bootstrap", "regression"

    # 실험 대상
    source_table: str
    source_column: str
    target_table: str
    target_column: str

    # 실험 조건
    condition: Dict[str, Any]

    # 측정 결과
    group_a: Dict[str, Any]  # e.g., {"label": "high", "n": 45, "mean": 234.5, "std": 12.3}
    group_b: Dict[str, Any]  # e.g., {"label": "low", "n": 52, "mean": 178.2, "std": 15.1}

    # 계산된 효과
    observed_effect: float  # 실제 관측된 차이 (%)
    effect_direction: str   # "increase" or "decrease"
    statistical_significance: float  # p-value
    confidence_interval: Tuple[float, float] = (0.0, 0.0)

    # 추가 분석
    effect_size: float = 0.0  # Cohen's d
    practical_significance: str = ""  # "small", "medium", "large"

    # 샘플 증거 (실제 데이터 예시)
    sample_evidence: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class SimulationReport:
    """전체 시뮬레이션 리포트 (LLM에게만 전달)"""
    simulation_id: str
    total_experiments: int

    # 모든 실험 결과
    experiments: List[SimulationExperiment] = field(default_factory=list)

    # 주요 발견
    significant_findings: List[Dict[str, Any]] = field(default_factory=list)

    # 예상치 못한 발견
    unexpected_findings: List[Dict[str, Any]] = field(default_factory=list)

    # 데이터 품질 노트
    data_quality_notes: List[str] = field(default_factory=list)


class SimulationEngine:
    """
    데이터 기반 시뮬레이션 엔진

    LLM이 정확한 판단을 내리기 위한 도구로서,
    실제 데이터로 다양한 what-if 시나리오를 실행하고 결과를 측정
    """

    def __init__(self, tables_data: Dict[str, Any]):
        """
        Args:
            tables_data: 테이블명 -> DataFrame 매핑
        """
        import pandas as pd
        self.tables_data = tables_data
        self._cache = {}

    def run_all_experiments(
        self,
        correlations: List[Any],  # List[CorrelationResult]
        max_experiments_per_correlation: int = 5
    ) -> SimulationReport:
        """
        모든 상관관계에 대해 시뮬레이션 실험 수행

        Args:
            correlations: 발견된 상관관계 목록
            max_experiments_per_correlation: 상관관계당 최대 실험 수

        Returns:
            SimulationReport: 전체 시뮬레이션 결과
        """
        experiments = []
        data_quality_notes = []

        for corr in correlations[:10]:  # 상위 10개 상관관계만
            try:
                # 1. Segmentation Analysis (구간 분석)
                seg_experiments = self._run_segmentation_analysis(corr)
                experiments.extend(seg_experiments)

                # 2. Threshold Analysis (임계값 분석)
                threshold_experiments = self._run_threshold_analysis(corr)
                experiments.extend(threshold_experiments)

                # 3. Bootstrap Confidence Estimation
                bootstrap_exp = self._run_bootstrap_analysis(corr)
                if bootstrap_exp:
                    experiments.append(bootstrap_exp)

            except Exception as e:
                src_col = corr.get("source_column", "unknown") if isinstance(corr, dict) else getattr(corr, "source_column", "unknown")
                tgt_col = corr.get("target_column", "unknown") if isinstance(corr, dict) else getattr(corr, "target_column", "unknown")
                logger.warning(f"Experiment failed for {src_col} -> {tgt_col}: {e}")
                data_quality_notes.append(f"분석 실패: {src_col} -> {tgt_col}")

        # 유의미한 발견 추출
        significant = self._extract_significant_findings(experiments)
        unexpected = self._find_unexpected_patterns(experiments)

        report = SimulationReport(
            simulation_id=str(uuid.uuid4())[:8],
            total_experiments=len(experiments),
            experiments=experiments,
            significant_findings=significant,
            unexpected_findings=unexpected,
            data_quality_notes=data_quality_notes,
        )

        logger.info(f"Simulation complete: {len(experiments)} experiments, "
                   f"{len(significant)} significant findings")

        return report

    def _run_segmentation_analysis(
        self,
        corr: Any,
        percentiles: List[Tuple[int, int]] = [(0, 25), (75, 100), (0, 50), (50, 100)]
    ) -> List[SimulationExperiment]:
        """
        구간 분석: 데이터를 구간별로 나누어 실제 차이 측정

        예: Daily_Capacity 상위 25% vs 하위 25%에서 Max_Weight_Quant 차이
        """
        import pandas as pd
        from scipy import stats

        experiments = []

        source_table = corr.source_table
        target_table = corr.target_table
        source_col = corr.source_column
        target_col = corr.target_column

        # 데이터 준비
        if source_table not in self.tables_data or target_table not in self.tables_data:
            return experiments

        df_source = self.tables_data[source_table]
        df_target = self.tables_data[target_table]

        if source_col not in df_source.columns or target_col not in df_target.columns:
            return experiments

        # 데이터 정렬 (행 수 맞추기)
        min_rows = min(len(df_source), len(df_target))
        source_values = df_source[source_col].iloc[:min_rows].dropna()
        target_values = df_target[target_col].iloc[:min_rows].dropna()

        # 최소 샘플 수: 10개면 분석 가능 (기존 20에서 완화)
        if len(source_values) < 10:
            logger.debug(f"Skipping segmentation: insufficient samples ({len(source_values)})")
            return experiments

        # 구간별 분석
        for low_pct, high_pct in percentiles:
            try:
                # 구간 경계 계산
                low_threshold = np.percentile(source_values, low_pct)
                high_threshold = np.percentile(source_values, high_pct)

                if low_pct == 0:
                    # 하위 구간
                    mask = source_values <= high_threshold
                    group_label = f"하위 {high_pct}%"
                elif high_pct == 100:
                    # 상위 구간
                    mask = source_values >= low_threshold
                    group_label = f"상위 {100-low_pct}%"
                else:
                    mask = (source_values >= low_threshold) & (source_values <= high_threshold)
                    group_label = f"{low_pct}-{high_pct}%"

                # 상위/하위 구간에서의 target 값 비교
                if low_pct == 0 and high_pct == 25:
                    # 하위 25% vs 상위 25% 비교
                    low_mask = source_values <= np.percentile(source_values, 25)
                    high_mask = source_values >= np.percentile(source_values, 75)

                    # 인덱스 맞추기
                    common_idx = source_values.index.intersection(target_values.index)
                    source_aligned = source_values.loc[common_idx]
                    target_aligned = target_values.loc[common_idx]

                    low_mask_aligned = source_aligned <= np.percentile(source_aligned, 25)
                    high_mask_aligned = source_aligned >= np.percentile(source_aligned, 75)

                    low_group = target_aligned[low_mask_aligned]
                    high_group = target_aligned[high_mask_aligned]

                    if len(low_group) < 2 or len(high_group) < 2:
                        continue

                    # 통계 계산
                    low_mean = low_group.mean()
                    high_mean = high_group.mean()
                    low_std = low_group.std()
                    high_std = high_group.std()

                    # 효과 크기 계산
                    if low_mean != 0:
                        effect_pct = ((high_mean - low_mean) / abs(low_mean)) * 100
                    else:
                        effect_pct = 0

                    # 통계적 유의성 (t-test)
                    try:
                        t_stat, p_value = stats.ttest_ind(high_group, low_group)
                    except Exception:
                        p_value = 1.0

                    # Cohen's d
                    pooled_std = np.sqrt((low_std**2 + high_std**2) / 2)
                    cohens_d = (high_mean - low_mean) / pooled_std if pooled_std > 0 else 0

                    # 실제 데이터 샘플
                    sample_evidence = []
                    for idx in high_group.head(3).index:
                        if idx in source_aligned.index:
                            sample_evidence.append({
                                source_col: float(source_aligned.loc[idx]),
                                target_col: float(target_aligned.loc[idx]),
                                "group": "high"
                            })
                    for idx in low_group.head(3).index:
                        if idx in source_aligned.index:
                            sample_evidence.append({
                                source_col: float(source_aligned.loc[idx]),
                                target_col: float(target_aligned.loc[idx]),
                                "group": "low"
                            })

                    experiment = SimulationExperiment(
                        experiment_id=f"SEG_{str(uuid.uuid4())[:6]}",
                        experiment_type="segmentation",
                        source_table=source_table,
                        source_column=source_col,
                        target_table=target_table,
                        target_column=target_col,
                        condition={
                            "type": "percentile_comparison",
                            "low_percentile": 25,
                            "high_percentile": 75,
                        },
                        group_a={
                            "label": f"{source_col} 상위 25%",
                            "n": len(high_group),
                            "mean": float(high_mean),
                            "std": float(high_std),
                            "min": float(high_group.min()),
                            "max": float(high_group.max()),
                        },
                        group_b={
                            "label": f"{source_col} 하위 25%",
                            "n": len(low_group),
                            "mean": float(low_mean),
                            "std": float(low_std),
                            "min": float(low_group.min()),
                            "max": float(low_group.max()),
                        },
                        observed_effect=float(effect_pct),
                        effect_direction="increase" if effect_pct > 0 else "decrease",
                        statistical_significance=float(p_value),
                        effect_size=float(cohens_d),
                        practical_significance=self._interpret_effect_size(cohens_d),
                        sample_evidence=sample_evidence,
                    )

                    experiments.append(experiment)

            except Exception as e:
                logger.debug(f"Segmentation analysis failed: {e}")
                continue

        return experiments

    def _run_threshold_analysis(
        self,
        corr: Any,
    ) -> List[SimulationExperiment]:
        """
        임계값 분석: 다양한 임계값에서의 효과 측정

        예: Daily_Capacity > 500 vs <= 500 에서의 결과 차이
        """
        import pandas as pd
        from scipy import stats

        experiments = []

        source_table = corr.source_table
        target_table = corr.target_table
        source_col = corr.source_column
        target_col = corr.target_column

        if source_table not in self.tables_data or target_table not in self.tables_data:
            return experiments

        df_source = self.tables_data[source_table]
        df_target = self.tables_data[target_table]

        if source_col not in df_source.columns or target_col not in df_target.columns:
            return experiments

        min_rows = min(len(df_source), len(df_target))
        source_values = df_source[source_col].iloc[:min_rows].dropna()
        target_values = df_target[target_col].iloc[:min_rows].dropna()

        if len(source_values) < 10:
            return experiments

        # 다양한 임계값 자동 탐지
        thresholds = [
            np.percentile(source_values, 25),   # Q1
            np.percentile(source_values, 50),   # Median
            np.percentile(source_values, 75),   # Q3
            source_values.mean(),               # Mean
            source_values.mean() + source_values.std(),  # Mean + 1 STD
        ]

        # 중복 제거 및 정렬
        thresholds = sorted(set([round(t, 2) for t in thresholds]))

        for threshold in thresholds[:3]:  # 상위 3개만
            try:
                common_idx = source_values.index.intersection(target_values.index)
                source_aligned = source_values.loc[common_idx]
                target_aligned = target_values.loc[common_idx]

                above_mask = source_aligned > threshold
                below_mask = source_aligned <= threshold

                above_group = target_aligned[above_mask]
                below_group = target_aligned[below_mask]

                if len(above_group) < 2 or len(below_group) < 2:
                    continue

                above_mean = above_group.mean()
                below_mean = below_group.mean()

                if below_mean != 0:
                    effect_pct = ((above_mean - below_mean) / abs(below_mean)) * 100
                else:
                    effect_pct = 0

                try:
                    t_stat, p_value = stats.ttest_ind(above_group, below_group)
                except Exception:
                    p_value = 1.0

                # 부트스트랩으로 신뢰구간 추정
                ci_low, ci_high = self._bootstrap_ci(above_group, below_group)

                experiment = SimulationExperiment(
                    experiment_id=f"THR_{str(uuid.uuid4())[:6]}",
                    experiment_type="threshold",
                    source_table=source_table,
                    source_column=source_col,
                    target_table=target_table,
                    target_column=target_col,
                    condition={
                        "type": "threshold",
                        "threshold_value": float(threshold),
                        "comparison": "above_vs_below",
                    },
                    group_a={
                        "label": f"{source_col} > {threshold:.1f}",
                        "n": len(above_group),
                        "mean": float(above_mean),
                        "std": float(above_group.std()),
                    },
                    group_b={
                        "label": f"{source_col} <= {threshold:.1f}",
                        "n": len(below_group),
                        "mean": float(below_mean),
                        "std": float(below_group.std()),
                    },
                    observed_effect=float(effect_pct),
                    effect_direction="increase" if effect_pct > 0 else "decrease",
                    statistical_significance=float(p_value),
                    confidence_interval=(float(ci_low), float(ci_high)),
                )

                experiments.append(experiment)

            except Exception as e:
                logger.debug(f"Threshold analysis failed: {e}")
                continue

        return experiments

    def _run_bootstrap_analysis(
        self,
        corr: Any,
        n_iterations: int = 500
    ) -> Optional[SimulationExperiment]:
        """
        부트스트랩 분석: 리샘플링으로 효과의 신뢰구간 추정
        """
        import pandas as pd

        source_table = corr.source_table
        target_table = corr.target_table
        source_col = corr.source_column
        target_col = corr.target_column

        if source_table not in self.tables_data or target_table not in self.tables_data:
            return None

        df_source = self.tables_data[source_table]
        df_target = self.tables_data[target_table]

        if source_col not in df_source.columns or target_col not in df_target.columns:
            return None

        min_rows = min(len(df_source), len(df_target))
        source_values = df_source[source_col].iloc[:min_rows].dropna()
        target_values = df_target[target_col].iloc[:min_rows].dropna()

        if len(source_values) < 10:
            return None

        try:
            common_idx = source_values.index.intersection(target_values.index)
            source_aligned = source_values.loc[common_idx]
            target_aligned = target_values.loc[common_idx]

            # 부트스트랩 상관계수 분포
            bootstrap_corrs = []

            for _ in range(n_iterations):
                # 리샘플링
                sample_idx = np.random.choice(len(common_idx), size=len(common_idx), replace=True)
                source_sample = source_aligned.iloc[sample_idx]
                target_sample = target_aligned.iloc[sample_idx]

                # 상관계수 계산
                if source_sample.std() > 0 and target_sample.std() > 0:
                    corr_val = source_sample.corr(target_sample)
                    if not np.isnan(corr_val):
                        bootstrap_corrs.append(corr_val)

            if len(bootstrap_corrs) < 100:
                return None

            # 신뢰구간 계산
            ci_low = np.percentile(bootstrap_corrs, 2.5)
            ci_high = np.percentile(bootstrap_corrs, 97.5)
            mean_corr = np.mean(bootstrap_corrs)
            std_corr = np.std(bootstrap_corrs)

            experiment = SimulationExperiment(
                experiment_id=f"BOOT_{str(uuid.uuid4())[:6]}",
                experiment_type="bootstrap",
                source_table=source_table,
                source_column=source_col,
                target_table=target_table,
                target_column=target_col,
                condition={
                    "type": "bootstrap",
                    "n_iterations": n_iterations,
                    "confidence_level": 0.95,
                },
                group_a={
                    "label": "bootstrap_distribution",
                    "n": len(bootstrap_corrs),
                    "mean": float(mean_corr),
                    "std": float(std_corr),
                },
                group_b={
                    "label": "original",
                    "n": len(common_idx),
                    "mean": float(corr.correlation),
                    "std": 0,
                },
                observed_effect=float(mean_corr * 100),  # 상관계수를 % 효과로 해석
                effect_direction="positive" if mean_corr > 0 else "negative",
                statistical_significance=0.05 if ci_low * ci_high > 0 else 0.5,  # CI가 0을 포함하지 않으면 유의
                confidence_interval=(float(ci_low), float(ci_high)),
            )

            return experiment

        except Exception as e:
            logger.debug(f"Bootstrap analysis failed: {e}")
            return None

    def _bootstrap_ci(
        self,
        group_a: Any,
        group_b: Any,
        n_iterations: int = 500,
        ci_level: float = 0.95
    ) -> Tuple[float, float]:
        """두 그룹 차이의 신뢰구간을 부트스트랩으로 추정"""
        differences = []

        for _ in range(n_iterations):
            sample_a = np.random.choice(group_a, size=len(group_a), replace=True)
            sample_b = np.random.choice(group_b, size=len(group_b), replace=True)

            mean_a = np.mean(sample_a)
            mean_b = np.mean(sample_b)

            if mean_b != 0:
                diff_pct = ((mean_a - mean_b) / abs(mean_b)) * 100
                differences.append(diff_pct)

        if not differences:
            return (0.0, 0.0)

        alpha = (1 - ci_level) / 2
        ci_low = np.percentile(differences, alpha * 100)
        ci_high = np.percentile(differences, (1 - alpha) * 100)

        return (ci_low, ci_high)

    def _interpret_effect_size(self, cohens_d: float) -> str:
        """Cohen's d 해석"""
        d = abs(cohens_d)
        if d < 0.2:
            return "negligible"
        elif d < 0.5:
            return "small"
        elif d < 0.8:
            return "medium"
        else:
            return "large"

    def _extract_significant_findings(
        self,
        experiments: List[SimulationExperiment]
    ) -> List[Dict[str, Any]]:
        """통계적으로 유의미한 발견 추출"""
        significant = []

        for exp in experiments:
            # p-value < 0.05이고 효과가 10% 이상인 경우
            if exp.statistical_significance < 0.05 and abs(exp.observed_effect) > 10:
                significant.append({
                    "experiment_id": exp.experiment_id,
                    "type": exp.experiment_type,
                    "source": f"{exp.source_table}.{exp.source_column}",
                    "target": f"{exp.target_table}.{exp.target_column}",
                    "effect": f"{exp.observed_effect:+.1f}%",
                    "direction": exp.effect_direction,
                    "p_value": exp.statistical_significance,
                    "condition": exp.condition,
                    "group_a": exp.group_a,
                    "group_b": exp.group_b,
                })

        # 효과 크기로 정렬
        significant.sort(key=lambda x: abs(float(x["effect"].rstrip("%"))), reverse=True)

        return significant

    def _find_unexpected_patterns(
        self,
        experiments: List[SimulationExperiment]
    ) -> List[Dict[str, Any]]:
        """예상치 못한 패턴 발견"""
        unexpected = []

        for exp in experiments:
            # 1. 상관관계와 반대 방향의 효과
            # 2. 매우 큰 효과 (> 50%)
            # 3. 특정 임계값에서만 급격한 변화

            if abs(exp.observed_effect) > 50 and exp.statistical_significance < 0.05:
                unexpected.append({
                    "type": "large_effect",
                    "description": f"{exp.source_column}의 변화가 {exp.target_column}에 "
                                  f"{abs(exp.observed_effect):.1f}%의 큰 영향",
                    "experiment": asdict(exp),
                })

            # 임계값 분석에서 급격한 변화
            if exp.experiment_type == "threshold":
                if abs(exp.observed_effect) > 30:
                    threshold = exp.condition.get("threshold_value", 0)
                    unexpected.append({
                        "type": "threshold_effect",
                        "description": f"{exp.source_column}이 {threshold:.1f}를 넘으면 "
                                      f"{exp.target_column}이 급격히 변화 ({exp.observed_effect:+.1f}%)",
                        "threshold": threshold,
                        "experiment": asdict(exp),
                    })

        return unexpected

    def format_for_llm(self, report: SimulationReport) -> str:
        """
        시뮬레이션 결과를 LLM에게 전달할 형식으로 포맷팅

        주의: 이 결과는 LLM의 분석용이며, 최종 사용자에게는 노출되지 않음
        """
        sections = []

        sections.append("# 시뮬레이션 분석 결과 (내부 분석용)")
        sections.append(f"\n총 {report.total_experiments}개 실험 수행\n")

        # 유의미한 발견이 있으면 표시
        if report.significant_findings:
            sections.append("## 주요 발견 (통계적으로 유의미)")
            for i, finding in enumerate(report.significant_findings[:10], 1):
                sections.append(f"\n### 발견 {i}")
                sections.append(f"- 관계: {finding['source']} → {finding['target']}")
                sections.append(f"- 효과: {finding['effect']} ({finding['direction']})")
                sections.append(f"- 통계적 유의성: p={finding['p_value']:.4f}")

                if finding.get('condition', {}).get('type') == 'threshold':
                    threshold = finding['condition'].get('threshold_value', 0)
                    sections.append(f"- 임계값: {threshold:.1f}")

                group_a = finding.get('group_a', {})
                group_b = finding.get('group_b', {})
                sections.append(f"- 그룹 A ({group_a.get('label', 'high')}): "
                              f"n={group_a.get('n', 0)}, mean={group_a.get('mean', 0):.2f}")
                sections.append(f"- 그룹 B ({group_b.get('label', 'low')}): "
                              f"n={group_b.get('n', 0)}, mean={group_b.get('mean', 0):.2f}")

        # 유의미한 발견이 없어도 모든 실험 결과 표시
        if not report.significant_findings and report.experiments:
            sections.append("## 모든 실험 결과 (분석용)")
            for i, exp in enumerate(report.experiments[:15], 1):
                sections.append(f"\n### 실험 {i}: {exp.experiment_type}")
                sections.append(f"- 관계: {exp.source_table}.{exp.source_column} → {exp.target_table}.{exp.target_column}")
                sections.append(f"- 관측된 효과: {exp.observed_effect:+.1f}% ({exp.effect_direction})")
                sections.append(f"- p-value: {exp.statistical_significance:.4f}")

                if exp.experiment_type == "threshold":
                    threshold = exp.condition.get("threshold_value", 0)
                    sections.append(f"- 임계값: {threshold:.1f}")

                sections.append(f"- 그룹 A ({exp.group_a.get('label', '')}): "
                              f"n={exp.group_a.get('n', 0)}, mean={exp.group_a.get('mean', 0):.2f}")
                sections.append(f"- 그룹 B ({exp.group_b.get('label', '')}): "
                              f"n={exp.group_b.get('n', 0)}, mean={exp.group_b.get('mean', 0):.2f}")

        # 예상치 못한 패턴
        if report.unexpected_findings:
            sections.append("\n## 예상치 못한 패턴")
            for finding in report.unexpected_findings[:5]:
                sections.append(f"- [{finding['type']}] {finding['description']}")

        # 데이터 품질 노트
        if report.data_quality_notes:
            sections.append("\n## 데이터 품질 참고사항")
            for note in report.data_quality_notes:
                sections.append(f"- {note}")

        return "\n".join(sections)
