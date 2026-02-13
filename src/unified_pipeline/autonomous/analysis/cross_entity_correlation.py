"""
Cross-Entity Correlation Analyzer for Palantir-Style Insights (v3.0 / v12.0)

v3.0 핵심 변경: LLM 자율 판단 강화
- 수학적 제약조건 자동 감지 (A+B=1, A+B=100 등)
- 동일 데이터 컬럼 감지 (모바일_세션수 == 전체_세션수)
- 카테고리/세그먼트 기반 분석 추가
- LLM이 무의미한 상관관계를 스스로 필터링

v2.0 핵심 변경: LLM 자율 분석
- 하드코딩된 규칙/템플릿 제거
- LLM이 데이터를 보고 자율적으로 시나리오 탐지, 관계 발견, 시뮬레이션 설계
- 실제 데이터 기반 what-if 시뮬레이션 및 검증

팔란티어 Foundry 스타일의 Cross-Entity 예측 인사이트 생성:
- "Gate 1 승객 30% 증가 → 도쿄행 비행기 지연 확률 20% 상승"
- "Plant_A 재고 20% 감소 → Carrier_X 운송비용 35% 증가"
- "TPT_Day_Count가 7일 초과시 → 배송 지연 확률 25% 상승"
"""

import logging
import json
import uuid
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from enum import Enum

from ...model_config import get_service_model

import numpy as np

logger = logging.getLogger(__name__)


class InsightType(str, Enum):
    """팔란티어 인사이트 타입"""
    CROSS_ENTITY_IMPACT = "cross_entity_impact"
    THRESHOLD_ALERT = "threshold_alert"
    PREDICTIVE_CORRELATION = "predictive_correlation"
    CHAIN_EFFECT = "chain_effect"  # A→B→C 체인 효과
    ANOMALY_TRIGGER = "anomaly_trigger"  # 이상치 발생 시 영향


@dataclass
class CorrelationResult:
    """테이블 간 상관관계 결과"""
    source_table: str
    source_column: str
    target_table: str
    target_column: str
    correlation: float
    p_value: float
    sample_size: int
    relationship_direction: str
    # v2.0: 추가 통계 정보
    source_stats: Dict[str, float] = field(default_factory=dict)
    target_stats: Dict[str, float] = field(default_factory=dict)


@dataclass
class SimulationResult:
    """시뮬레이션 결과"""
    scenario_id: str
    scenario_description: str
    source_change: Dict[str, Any]  # {"column": "X", "from": 100, "to": 120, "change_pct": 20}
    predicted_impacts: List[Dict[str, Any]]  # [{"column": "Y", "predicted_change": 15, "confidence": 0.85}]
    validation_score: float  # 실제 데이터로 검증한 정확도
    sample_evidence: List[Dict[str, Any]]  # 실제 데이터에서 찾은 증거


@dataclass
class PalantirInsight:
    """팔란티어 스타일 예측 인사이트 (v2.0)"""
    insight_id: str
    insight_type: str

    # 원인 측
    source_table: str
    source_column: str
    source_condition: str

    # 결과 측
    target_table: str
    target_column: str
    predicted_impact: str

    # 통계적 근거
    correlation_coefficient: float
    p_value: float
    confidence: float
    sample_size: int

    # v2.0: LLM 생성 콘텐츠
    business_interpretation: str = ""
    recommended_action: str = ""
    risk_assessment: str = ""
    opportunity_assessment: str = ""

    # v2.0: 시뮬레이션 결과
    simulation_results: List[Dict[str, Any]] = field(default_factory=list)
    validation_evidence: List[Dict[str, Any]] = field(default_factory=list)

    # 메타데이터
    source_entity: str = ""
    target_entity: str = ""
    chain_effects: List[str] = field(default_factory=list)  # A→B→C 체인

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class CrossEntityCorrelationAnalyzer:
    """
    Cross-Entity 상관관계 분석기 (v2.0 - LLM 자율 분석)

    v2.0 핵심 변경:
    - LLM이 자율적으로 시나리오 탐지/설계
    - 실제 데이터 기반 시뮬레이션
    - 하드코딩 규칙 제거
    """

    def __init__(
        self,
        data_dir: str,
        correlation_threshold: float = 0.3,
        p_value_threshold: float = 0.05,
        llm_client=None,
    ):
        self.data_dir = Path(data_dir)
        self.correlation_threshold = correlation_threshold
        self.p_value_threshold = p_value_threshold
        self.llm_client = llm_client
        self._tables_data: Dict[str, Any] = {}
        self._column_stats: Dict[str, Dict[str, Dict[str, float]]] = {}  # table -> column -> stats

    def load_table_data(self, table_name: str) -> Optional[Any]:
        """CSV 파일에서 테이블 데이터 로딩"""
        if table_name in self._tables_data:
            return self._tables_data[table_name]

        try:
            import pandas as pd

            csv_path = self.data_dir / f"{table_name}.csv"
            if not csv_path.exists():
                for path in self.data_dir.rglob(f"{table_name}.csv"):
                    csv_path = path
                    break

            if not csv_path.exists():
                logger.warning(f"CSV file not found: {table_name}")
                return None

            df = pd.read_csv(csv_path)
            self._tables_data[table_name] = df

            # 컬럼별 통계 계산 및 캐시
            self._compute_column_stats(table_name, df)

            logger.debug(f"Loaded table {table_name}: {len(df)} rows, {len(df.columns)} columns")
            return df

        except Exception as e:
            logger.warning(f"Failed to load table {table_name}: {e}")
            return None

    def _compute_column_stats(self, table_name: str, df: Any) -> None:
        """컬럼별 통계 계산 (LLM에게 제공할 컨텍스트)"""
        import pandas as pd

        if table_name not in self._column_stats:
            self._column_stats[table_name] = {}

        # v12.0: 데이터 품질 이슈 캐시
        if not hasattr(self, '_data_quality_issues'):
            self._data_quality_issues = []
        if not hasattr(self, '_mathematical_constraints'):
            self._mathematical_constraints = []
        if not hasattr(self, '_duplicate_columns'):
            self._duplicate_columns = []

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        for col in numeric_cols:
            series = df[col].dropna()
            if len(series) > 0:
                self._column_stats[table_name][col] = {
                    "mean": float(series.mean()),
                    "std": float(series.std()) if len(series) > 1 else 0.0,
                    "min": float(series.min()),
                    "max": float(series.max()),
                    "median": float(series.median()),
                    "q25": float(series.quantile(0.25)),
                    "q75": float(series.quantile(0.75)),
                    "count": int(len(series)),
                    "unique_ratio": float(series.nunique() / len(series)) if len(series) > 0 else 0,
                }

    def get_numeric_columns(self, df: Any) -> List[str]:
        """숫자형 컬럼 목록 반환"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        filtered = [c for c in numeric_cols if not c.lower().endswith('_id')
                   and 'id' not in c.lower().split('_')]
        return filtered if filtered else numeric_cols[:10]

    # ============================================================
    # v12.0: 데이터 품질 및 수학적 제약조건 감지
    # ============================================================

    def _detect_mathematical_constraints(self, df: Any, table_name: str) -> List[Dict[str, Any]]:
        """
        v12.0: 수학적 제약조건 감지 (A+B=1, A+B=100 등)

        예: 신규방문비율 + 재방문비율 = 1.0
        예: 완료율 + 미완료율 = 100

        이런 관계는 상관관계 분석에서 제외해야 함
        """
        import itertools

        constraints = []
        numeric_cols = self.get_numeric_columns(df)

        for col1, col2 in itertools.combinations(numeric_cols, 2):
            try:
                series1 = df[col1].dropna()
                series2 = df[col2].dropna()

                # 같은 인덱스만 사용
                common_idx = series1.index.intersection(series2.index)
                if len(common_idx) < 10:
                    continue

                s1 = series1.loc[common_idx]
                s2 = series2.loc[common_idx]

                # 합이 상수인지 확인
                sum_values = s1 + s2
                sum_std = sum_values.std()
                sum_mean = sum_values.mean()

                # 표준편차가 매우 작으면 (< 평균의 1%) 수학적 제약
                if sum_mean != 0 and sum_std / abs(sum_mean) < 0.01:
                    # 1.0 또는 100 근처인지 확인
                    is_unit_constraint = abs(sum_mean - 1.0) < 0.05
                    is_percent_constraint = abs(sum_mean - 100) < 5

                    if is_unit_constraint or is_percent_constraint:
                        constraint_type = "비율 보완 (합=1)" if is_unit_constraint else "퍼센트 보완 (합=100)"
                        constraints.append({
                            "table": table_name,
                            "column_1": col1,
                            "column_2": col2,
                            "constraint_type": constraint_type,
                            "sum_value": float(sum_mean),
                            "reason": f"{col1} + {col2} = {sum_mean:.2f} (수학적 정의 관계)"
                        })
                        logger.info(f"[v12.0] Mathematical constraint detected: {col1} + {col2} = {sum_mean:.2f}")

            except Exception as e:
                logger.debug(f"Constraint check failed for {col1}, {col2}: {e}")

        return constraints

    def _detect_duplicate_columns(self, df: Any, table_name: str) -> List[Dict[str, Any]]:
        """
        v12.0: 동일 데이터 컬럼 감지

        예: 모바일_세션수 == 전체_세션수 (100% 동일)

        이런 경우 데이터 품질 이슈로 보고해야 함
        """
        import itertools

        duplicates = []
        numeric_cols = self.get_numeric_columns(df)

        for col1, col2 in itertools.combinations(numeric_cols, 2):
            try:
                series1 = df[col1].dropna()
                series2 = df[col2].dropna()

                common_idx = series1.index.intersection(series2.index)
                if len(common_idx) < 10:
                    continue

                s1 = series1.loc[common_idx]
                s2 = series2.loc[common_idx]

                # 완전 일치 확인
                match_rate = (s1 == s2).mean()

                if match_rate > 0.99:  # 99% 이상 일치
                    duplicates.append({
                        "table": table_name,
                        "column_1": col1,
                        "column_2": col2,
                        "match_rate": float(match_rate),
                        "severity": "warning",
                        "reason": f"'{col1}'과 '{col2}'가 {match_rate*100:.1f}% 동일합니다. 데이터 수집 오류 또는 중복 컬럼 가능성."
                    })
                    logger.warning(f"[v12.0] Duplicate columns detected: {col1} == {col2} ({match_rate*100:.1f}%)")

            except Exception as e:
                logger.debug(f"Duplicate check failed for {col1}, {col2}: {e}")

        return duplicates

    def _detect_derived_columns(self, df: Any, table_name: str) -> List[Dict[str, Any]]:
        """
        v26.0: 파생(계산) 컬럼 감지

        col_c ≈ col_a / col_b * K (K=1 or 100) 형태의 관계를 자동 탐지.
        예: like_rate = like_count / view_count * 100
        예: views_per_follower = view_count / follower_count
        예: avg_scene_duration = duration / scene_count
        """
        import itertools

        derived = []
        numeric_cols = self.get_numeric_columns(df)

        if len(numeric_cols) < 3:
            return derived

        for target_col in numeric_cols:
            target = df[target_col].dropna()
            if len(target) < 20:
                continue

            for col_a, col_b in itertools.combinations(numeric_cols, 2):
                if col_a == target_col or col_b == target_col:
                    continue

                try:
                    sa = df[col_a].loc[target.index].dropna()
                    sb = df[col_b].loc[target.index].dropna()
                    common = target.index.intersection(sa.index).intersection(sb.index)
                    if len(common) < 20:
                        continue

                    t = target.loc[common]
                    a = sa.loc[common]
                    b = sb.loc[common]

                    # b != 0 인 행만 사용 (division by zero 방지)
                    nonzero_mask = b != 0
                    if nonzero_mask.sum() < 20:
                        continue

                    t_nz = t[nonzero_mask]
                    a_nz = a[nonzero_mask]
                    b_nz = b[nonzero_mask]

                    # a/b와 비교
                    ratio = a_nz / b_nz

                    # K=1: target ≈ a / b
                    for multiplier, label in [(1.0, ""), (100.0, " * 100")]:
                        computed = ratio * multiplier
                        # 상대 오차 계산 (t가 0인 경우 절대 오차 사용)
                        abs_diff = (t_nz - computed).abs()
                        abs_target = t_nz.abs().clip(lower=1e-10)
                        rel_error = (abs_diff / abs_target).median()

                        if rel_error < 0.02:  # 중위 상대오차 2% 미만
                            match_rate = float((abs_diff / abs_target < 0.05).mean())
                            if match_rate >= 0.95:  # 95% 이상 일치
                                formula = f"{col_a} / {col_b}{label}"
                                derived.append({
                                    "table": table_name,
                                    "derived_column": target_col,
                                    "formula": formula,
                                    "source_columns": [col_a, col_b],
                                    "match_rate": match_rate,
                                    "sample_size": int(nonzero_mask.sum()),
                                    "severity": "info",
                                    "reason": f"'{target_col}' = {formula} (일치율 {match_rate*100:.1f}%, 파생 컬럼)",
                                })
                                logger.info(f"[v26.0] Derived column detected: {target_col} = {formula} ({match_rate*100:.1f}% match)")
                                break  # 이 target_col에 대해 첫 번째 매칭만 사용
                    else:
                        continue
                    break  # target_col에 대한 첫 매칭을 찾았으므로 다음 target으로

                except Exception as e:
                    logger.debug(f"Derived column check failed for {target_col} = {col_a}/{col_b}: {e}")

        return derived

    def _detect_categorical_columns(self, df: Any) -> List[str]:
        """v12.0: 카테고리/세그먼트 컬럼 감지"""
        import pandas as pd
        categorical_cols = []

        for col in df.columns:
            # 문자열 컬럼이면서 고유값이 적당히 있는 경우
            if pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_string_dtype(df[col]):
                nunique = df[col].nunique()
                if 2 <= nunique <= 100:  # 2~100개 고유값
                    categorical_cols.append(col)

        return categorical_cols

    def detect_confounding_variables(
        self, df: Any, treatment_col: str, outcome_col: str,
        candidate_confounders: List[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        v26.0: 교란변수(confounding variable) 탐지 — Simpson's Paradox 검출

        treatment_col이 outcome_col에 미치는 효과가 confounder를 통제하면
        사라지거나 반전되는지 검사합니다.

        Returns: 교란변수 후보 리스트 [{confounder, raw_effect, adjusted_effect, is_confounded}]
        """
        from scipy import stats

        confounders_found = []

        if candidate_confounders is None:
            candidate_confounders = self.get_numeric_columns(df)

        # 원시 상관관계
        try:
            mask = df[[treatment_col, outcome_col]].notna().all(axis=1)
            raw_corr, raw_p = stats.pearsonr(
                df.loc[mask, treatment_col], df.loc[mask, outcome_col]
            )
        except Exception:
            return confounders_found

        for confounder in candidate_confounders:
            if confounder in (treatment_col, outcome_col):
                continue

            try:
                # 3변수 모두 존재하는 행만 사용
                valid = df[[treatment_col, outcome_col, confounder]].dropna()
                if len(valid) < 30:
                    continue

                # 편상관(partial correlation): treatment-outcome 상관에서 confounder 효과 제거
                # rAB.C = (rAB - rAC * rBC) / sqrt((1 - rAC^2)(1 - rBC^2))
                r_to = raw_corr
                r_tc, _ = stats.pearsonr(valid[treatment_col], valid[confounder])
                r_oc, _ = stats.pearsonr(valid[outcome_col], valid[confounder])

                denom = ((1 - r_tc**2) * (1 - r_oc**2)) ** 0.5
                if denom < 1e-10:
                    continue
                partial_corr = (r_to - r_tc * r_oc) / denom

                # 교란 판정: 원시 상관이 유의미하지만 편상관이 50% 이상 감소
                raw_abs = abs(raw_corr)
                partial_abs = abs(partial_corr)
                reduction = (raw_abs - partial_abs) / raw_abs if raw_abs > 0.05 else 0

                if reduction > 0.4 and raw_abs > 0.05:
                    # Simpson's paradox 체크: 카테고리형 treatment일 때 층화 분석
                    is_simpson = False
                    if df[treatment_col].nunique() <= 5:
                        # 층화 분석 — confounder를 quantile로 나누어 각 층에서 효과 확인
                        try:
                            valid["_conf_bin"] = np.digitize(
                                valid[confounder],
                                bins=np.quantile(valid[confounder].dropna(), [0.33, 0.67]),
                            )
                            strata_effects = []
                            for _, stratum in valid.groupby("_conf_bin"):
                                if len(stratum) < 10:
                                    continue
                                s_corr, _ = stats.pearsonr(
                                    stratum[treatment_col], stratum[outcome_col]
                                )
                                strata_effects.append(s_corr)
                            # 층화 효과의 부호가 원시와 반대 → Simpson's Paradox
                            if strata_effects and all(
                                (e * raw_corr) < 0 for e in strata_effects if abs(e) > 0.02
                            ):
                                is_simpson = True
                        except Exception:
                            pass

                    confounders_found.append({
                        "confounder": confounder,
                        "raw_correlation": round(float(raw_corr), 4),
                        "partial_correlation": round(float(partial_corr), 4),
                        "reduction_pct": round(float(reduction * 100), 1),
                        "r_treatment_confounder": round(float(r_tc), 4),
                        "r_outcome_confounder": round(float(r_oc), 4),
                        "is_confounded": True,
                        "is_simpsons_paradox": is_simpson,
                        "note": (
                            f"'{confounder}'를 통제하면 {treatment_col}→{outcome_col} "
                            f"상관이 {raw_abs:.3f}→{partial_abs:.3f}로 {reduction*100:.0f}% 감소"
                            + (" [Simpson's Paradox 감지]" if is_simpson else "")
                        ),
                    })
                    logger.info(
                        f"[v26.0] Confounding detected: {confounder} confounds "
                        f"{treatment_col}→{outcome_col} (reduction={reduction*100:.0f}%"
                        f"{', Simpson!' if is_simpson else ''})"
                    )

            except Exception as e:
                logger.debug(f"Confounding check failed for {confounder}: {e}")

        return confounders_found

    def _analyze_categorical_insights(self, df: Any, table_name: str) -> List[Dict[str, Any]]:
        """
        v12.2: 카테고리/세그먼트 기반 비즈니스 인사이트 분석 (강화)

        예: 트래픽 소스별 세션수, 체류시간, 신규방문비율 분석
        """
        insights = []
        categorical_cols = self._detect_categorical_columns(df)
        numeric_cols = self.get_numeric_columns(df)

        logger.info(f"[v12.2] Analyzing categorical columns: {categorical_cols}")
        logger.info(f"[v12.2] Numeric columns for segment analysis: {numeric_cols[:5]}")

        # 핵심 지표 우선순위 (세션수, 체류시간, 비율 등)
        priority_patterns = ['세션', 'session', '체류', 'time', 'duration', '방문', 'visit']

        # 우선순위 기반 정렬
        def get_priority(col):
            col_lower = col.lower()
            for i, pattern in enumerate(priority_patterns):
                if pattern in col_lower:
                    return i
            return len(priority_patterns)

        sorted_numeric = sorted(numeric_cols, key=get_priority)

        for cat_col in categorical_cols[:3]:  # v12.2: 최대 3개 카테고리
            for num_col in sorted_numeric[:5]:  # v12.2: 최대 5개 지표
                try:
                    # 카테고리별 집계
                    agg = df.groupby(cat_col)[num_col].agg(['sum', 'mean', 'count', 'std'])

                    if len(agg) < 2:
                        continue

                    # TOP 5 분석 (sum 기준)
                    top_5_sum = agg['sum'].nlargest(5)
                    total = agg['sum'].sum()

                    if total == 0:
                        continue

                    top_categories = []
                    for cat, value in top_5_sum.items():
                        pct = (value / total) * 100
                        cat_mean = agg.loc[cat, 'mean']
                        cat_count = agg.loc[cat, 'count']
                        top_categories.append({
                            "category": str(cat),
                            "value": float(value),
                            "percentage": float(pct),
                            "mean": float(cat_mean),
                            "count": int(cat_count)
                        })

                    concentration = sum(c["percentage"] for c in top_categories)

                    # v12.2: 평균 기준 TOP 5도 분석 (체류시간 등에 유용)
                    top_5_mean = agg['mean'].nlargest(5)
                    top_by_mean = []
                    for cat, mean_val in top_5_mean.items():
                        if agg.loc[cat, 'count'] >= 3:  # 최소 3개 샘플
                            top_by_mean.append({
                                "category": str(cat),
                                "mean": float(mean_val),
                                "count": int(agg.loc[cat, 'count'])
                            })

                    insights.append({
                        "table": table_name,
                        "category_column": cat_col,
                        "metric_column": num_col,
                        "total_categories": int(df[cat_col].nunique()),
                        "top_categories": top_categories,
                        "top_by_mean": top_by_mean[:5],  # 평균 기준 TOP 5
                        "top5_concentration": float(concentration),
                        "overall_mean": float(agg['mean'].mean()),
                        "overall_std": float(agg['mean'].std()) if len(agg) > 1 else 0.0,
                        "insight_type": "category_distribution"
                    })

                    logger.debug(f"[v12.2] Segment insight: {cat_col} x {num_col}, "
                                f"top={top_categories[0]['category'] if top_categories else 'N/A'}, "
                                f"concentration={concentration:.1f}%")

                except Exception as e:
                    logger.debug(f"Categorical analysis failed for {cat_col} x {num_col}: {e}")

        logger.info(f"[v12.2] Generated {len(insights)} categorical insights for {table_name}")
        return insights

    def compute_correlation(
        self,
        series_a: Any,
        series_b: Any,
    ) -> Tuple[float, float, int]:
        """두 시리즈 간 Pearson 상관계수 계산"""
        from scipy import stats
        import pandas as pd

        combined = pd.concat([series_a, series_b], axis=1).dropna()

        if len(combined) < 10:
            return 0.0, 1.0, 0

        try:
            corr, p_value = stats.pearsonr(combined.iloc[:, 0], combined.iloc[:, 1])
            return float(corr), float(p_value), len(combined)
        except Exception:
            return 0.0, 1.0, 0

    def compute_cross_table_correlations(
        self,
        table_names: List[str],
    ) -> List[CorrelationResult]:
        """테이블 간 모든 숫자형 컬럼 쌍의 상관계수 계산"""
        import pandas as pd

        correlations = []
        tables_data = {}

        for name in table_names:
            df = self.load_table_data(name)
            if df is not None and len(df) > 0:
                tables_data[name] = df

        if len(tables_data) < 2:
            # v24.0: WARNING → DEBUG (단일 테이블에서는 정상 동작)
            logger.debug("Need at least 2 tables for cross-table correlation")
            return correlations

        table_list = list(tables_data.keys())

        for i, table_a in enumerate(table_list):
            df_a = tables_data[table_a]
            cols_a = self.get_numeric_columns(df_a)

            for table_b in table_list[i+1:]:
                df_b = tables_data[table_b]
                cols_b = self.get_numeric_columns(df_b)

                min_rows = min(len(df_a), len(df_b))

                # v8.1: skip if row counts too different (spurious correlation risk)
                if min_rows < 10 or min_rows / max(len(df_a), len(df_b), 1) < 0.8:
                    continue

                for col_a in cols_a[:10]:
                    for col_b in cols_b[:10]:
                        if col_a == col_b:
                            continue

                        series_a = df_a[col_a].iloc[:min_rows].reset_index(drop=True)
                        series_b = df_b[col_b].iloc[:min_rows].reset_index(drop=True)

                        corr, p_value, sample_size = self.compute_correlation(series_a, series_b)

                        if abs(corr) >= self.correlation_threshold and p_value <= self.p_value_threshold:
                            # v2.0: 통계 정보 추가
                            source_stats = self._column_stats.get(table_a, {}).get(col_a, {})
                            target_stats = self._column_stats.get(table_b, {}).get(col_b, {})

                            correlations.append(CorrelationResult(
                                source_table=table_a,
                                source_column=col_a,
                                target_table=table_b,
                                target_column=col_b,
                                correlation=corr,
                                p_value=p_value,
                                sample_size=sample_size,
                                relationship_direction="positive" if corr > 0 else "negative",
                                source_stats=source_stats,
                                target_stats=target_stats,
                            ))

        # 상관관계 강도로 정렬
        correlations.sort(key=lambda x: abs(x.correlation), reverse=True)

        logger.info(f"Found {len(correlations)} significant cross-table correlations")
        return correlations

    def _prepare_data_summary_for_llm(self, tables_data: Dict[str, Any]) -> str:
        """LLM에게 전달할 데이터 요약 생성 (v12.0: 데이터 품질 이슈 포함)"""
        summary_parts = []

        # v12.0: 데이터 품질 이슈 수집
        all_constraints = []
        all_duplicates = []
        all_categorical = []
        all_derived = []  # v26.0: 파생 컬럼

        for table_name, df in tables_data.items():
            numeric_cols = self.get_numeric_columns(df)
            stats = self._column_stats.get(table_name, {})

            col_info = []
            for col in numeric_cols[:8]:  # 주요 컬럼만
                col_stats = stats.get(col, {})
                if col_stats:
                    col_info.append(
                        f"  - {col}: mean={col_stats.get('mean', 0):.2f}, "
                        f"std={col_stats.get('std', 0):.2f}, "
                        f"range=[{col_stats.get('min', 0):.2f}, {col_stats.get('max', 0):.2f}]"
                    )

            summary_parts.append(
                f"테이블: {table_name} ({len(df)} rows)\n"
                f"숫자형 컬럼:\n" + "\n".join(col_info)
            )

            # v12.0: 각 테이블에 대해 품질 검사 수행
            constraints = self._detect_mathematical_constraints(df, table_name)
            duplicates = self._detect_duplicate_columns(df, table_name)
            categorical = self._analyze_categorical_insights(df, table_name)
            # v26.0: 파생 컬럼 감지
            derived = self._detect_derived_columns(df, table_name)

            all_constraints.extend(constraints)
            all_duplicates.extend(duplicates)
            all_categorical.extend(categorical)
            all_derived.extend(derived)

        # 캐시에 저장 (나중에 LLM에게 전달)
        self._mathematical_constraints = all_constraints
        self._duplicate_columns = all_duplicates
        self._categorical_insights = all_categorical
        self._derived_columns = all_derived

        return "\n\n".join(summary_parts)

    def _prepare_data_quality_warnings_for_llm(self) -> str:
        """v12.0: LLM에게 전달할 데이터 품질 경고 메시지"""
        warnings = []

        # 수학적 제약조건
        if hasattr(self, '_mathematical_constraints') and self._mathematical_constraints:
            warnings.append("## ⚠️ 수학적 제약조건 (분석에서 제외 필요)")
            for c in self._mathematical_constraints:
                warnings.append(f"  - {c['reason']}")
                warnings.append(f"    → 이 두 컬럼의 상관관계는 인과관계가 아닌 수학적 정의입니다. 인사이트에서 제외하세요.")

        # 동일 데이터 컬럼
        if hasattr(self, '_duplicate_columns') and self._duplicate_columns:
            warnings.append("\n## ⚠️ 동일 데이터 컬럼 (데이터 품질 이슈)")
            for d in self._duplicate_columns:
                warnings.append(f"  - {d['reason']}")
                warnings.append(f"    → 이 두 컬럼은 사실상 동일한 데이터입니다. 분석 가치가 없으며, 데이터 품질 이슈로 보고해야 합니다.")

        # v26.0: 파생 컬럼
        if hasattr(self, '_derived_columns') and self._derived_columns:
            warnings.append("\n## 📐 파생(계산) 컬럼 감지")
            for d in self._derived_columns:
                warnings.append(f"  - {d['reason']}")
                warnings.append(f"    → 이 컬럼은 다른 컬럼에서 계산된 파생 값입니다. KPI 분석 시 원본 컬럼과의 상관관계는 무의미합니다.")

        # v26.0: 교란변수 경고
        if hasattr(self, '_confounding_results') and self._confounding_results:
            warnings.append("\n## 🔍 교란변수 감지 (Simpson's Paradox 위험)")
            for cf in self._confounding_results[:5]:
                warnings.append(f"  - {cf['note']}")
                if cf.get('is_simpsons_paradox'):
                    warnings.append(f"    ⚠️ Simpson's Paradox 확인 — 층화 분석에서 효과가 반전됩니다!")
                warnings.append(f"    → 이 상관관계를 인과관계로 보고하지 마세요. 교란변수를 통제한 편상관: {cf['partial_correlation']:.4f}")

        return "\n".join(warnings) if warnings else ""

    def _prepare_categorical_summary_for_llm(self) -> str:
        """v12.0: LLM에게 전달할 카테고리 분석 결과"""
        if not hasattr(self, '_categorical_insights') or not self._categorical_insights:
            return ""

        summary = ["## 📊 카테고리/세그먼트 분석 결과"]

        for cat in self._categorical_insights[:5]:  # 최대 5개
            summary.append(f"\n### {cat['category_column']}별 {cat['metric_column']} 분석")
            summary.append(f"  - 총 {cat['total_categories']}개 카테고리")
            summary.append(f"  - 상위 5개가 {cat['top5_concentration']:.1f}% 차지")
            summary.append("  - TOP 5:")
            for top in cat['top_categories'][:5]:
                summary.append(f"    • {top['category']}: {top['value']:.0f} ({top['percentage']:.1f}%)")

        return "\n".join(summary)

    def _filter_invalid_correlations(self, correlations: List[CorrelationResult]) -> List[CorrelationResult]:
        """
        v12.1: 무효한 상관관계 필터링 (수학적 제약조건, 동일 데이터)

        LLM에 의존하지 않고 코드 레벨에서 직접 필터링
        - 수학적 제약조건 컬럼 쌍 (A+B=1, A+B=100)
        - 동일 데이터 컬럼 쌍 (100% 일치)
        - 상관계수 절대값이 0.99 이상 (거의 완벽한 상관관계 = 수학적 관계 의심)
        """
        if not correlations:
            return correlations

        # 필터링 대상 컬럼 쌍 수집
        excluded_pairs = set()

        # 1. 수학적 제약조건 컬럼 쌍 수집
        if hasattr(self, '_mathematical_constraints') and self._mathematical_constraints:
            for constraint in self._mathematical_constraints:
                col1 = constraint.get('column_1', '')
                col2 = constraint.get('column_2', '')
                table = constraint.get('table', '')
                if col1 and col2:
                    # 양방향 추가
                    excluded_pairs.add((table, col1, table, col2))
                    excluded_pairs.add((table, col2, table, col1))
                    # 다른 테이블에서도 같은 이름의 컬럼이면 제외
                    excluded_pairs.add(('*', col1, '*', col2))
                    excluded_pairs.add(('*', col2, '*', col1))
                    logger.debug(f"[v12.1] Excluding math constraint pair: {col1} <-> {col2}")

        # 2. 동일 데이터 컬럼 쌍 수집
        if hasattr(self, '_duplicate_columns') and self._duplicate_columns:
            for dup in self._duplicate_columns:
                col1 = dup.get('column_1', '')
                col2 = dup.get('column_2', '')
                table = dup.get('table', '')
                if col1 and col2:
                    excluded_pairs.add((table, col1, table, col2))
                    excluded_pairs.add((table, col2, table, col1))
                    excluded_pairs.add(('*', col1, '*', col2))
                    excluded_pairs.add(('*', col2, '*', col1))
                    logger.debug(f"[v12.1] Excluding duplicate pair: {col1} <-> {col2}")

        # 3. 필터링 수행
        filtered = []
        removed_count = 0

        for corr in correlations:
            # 상관계수가 거의 완벽한 경우 (|r| >= 0.99) - 수학적 관계 의심
            if abs(corr.correlation) >= 0.99:
                # 이름 패턴 기반 추가 검증
                src_col_lower = corr.source_column.lower()
                tgt_col_lower = corr.target_column.lower()

                # "비율" 컬럼 쌍이면서 -1.0 상관계수면 A+B=1 관계
                if abs(corr.correlation + 1.0) < 0.01:  # -1.0에 가까움
                    if ('비율' in src_col_lower and '비율' in tgt_col_lower) or \
                       ('rate' in src_col_lower and 'rate' in tgt_col_lower) or \
                       ('ratio' in src_col_lower and 'ratio' in tgt_col_lower) or \
                       ('percent' in src_col_lower and 'percent' in tgt_col_lower):
                        logger.warning(f"[v12.1] REMOVED (math constraint pattern): "
                                      f"{corr.source_column} <-> {corr.target_column} (r={corr.correlation:.3f})")
                        removed_count += 1
                        continue

                # 이름이 매우 유사한 경우도 필터링 (예: 모바일_세션수 vs 전체_세션수)
                if abs(corr.correlation - 1.0) < 0.01:  # +1.0에 가까움
                    # 공통 부분 확인 (예: "세션수"가 둘 다 포함)
                    common_parts = set(src_col_lower.split('_')) & set(tgt_col_lower.split('_'))
                    if common_parts - {'', '전체', '모바일', 'pc', 'total', 'mobile'}:
                        logger.warning(f"[v12.1] REMOVED (duplicate pattern): "
                                      f"{corr.source_column} <-> {corr.target_column} (r={corr.correlation:.3f})")
                        removed_count += 1
                        continue

            # 명시적으로 제외된 쌍인지 확인
            pair1 = (corr.source_table, corr.source_column, corr.target_table, corr.target_column)
            pair2 = ('*', corr.source_column, '*', corr.target_column)

            if pair1 in excluded_pairs or pair2 in excluded_pairs:
                logger.warning(f"[v12.1] REMOVED (explicit exclusion): "
                              f"{corr.source_table}.{corr.source_column} <-> "
                              f"{corr.target_table}.{corr.target_column}")
                removed_count += 1
                continue

            # 유효한 상관관계만 추가
            filtered.append(corr)

        if removed_count > 0:
            logger.info(f"[v12.1] Filtered out {removed_count} invalid correlations")

        return filtered

    def _create_data_quality_insights_only(self) -> List[PalantirInsight]:
        """
        v13.0: Phase 1 분석 결과 기반 데이터 품질 + 카테고리 인사이트 반환

        v13.0 변경:
        - Phase 1 DataAnalystAgent의 분석 결과(data_patterns, data_quality_issues)를 활용
        - recommended_analysis의 key_dimensions, key_metrics 기반 세그먼트 분석
        - LLM이 이미 탐지한 패턴을 인사이트로 변환

        모든 상관관계가 수학적 제약조건이나 동일 데이터로 필터링된 경우에도,
        카테고리/세그먼트 분석 결과는 유효한 비즈니스 인사이트이므로 포함합니다.
        """
        insights = []
        insight_id = 1

        # v13.0: Phase 1에서 LLM이 탐지한 data_patterns를 인사이트로 변환
        if hasattr(self, '_phase1_analysis') and self._phase1_analysis:
            data_patterns = self._phase1_analysis.get('data_patterns', [])
            for pattern in data_patterns[:3]:  # 최대 3개
                pattern_type = pattern.get('pattern_type', '')
                description = pattern.get('description', '')
                implication = pattern.get('implication', '')
                columns = pattern.get('columns_involved', [])

                if pattern_type == 'mathematical_constraint':
                    # 수학적 제약조건 패턴
                    insights.append(PalantirInsight(
                        insight_id=f"DQ_{insight_id:03d}",
                        insight_type="data_quality_issue",
                        source_table="",
                        source_column=columns[0] if columns else "",
                        source_condition=f"LLM 탐지: {description}",
                        target_table="",
                        target_column=columns[1] if len(columns) > 1 else "",
                        predicted_impact=implication,
                        correlation_coefficient=-1.0,
                        p_value=0.0,
                        confidence=1.0,
                        sample_size=0,
                        business_interpretation=f"[LLM 분석] {description}. {implication}",
                        recommended_action="이 컬럼 쌍은 상관관계 분석에서 제외됩니다.",
                        risk_assessment="수학적 관계를 인과관계로 오해하지 마세요.",
                        opportunity_assessment="세그먼트/카테고리 분석에 집중하세요.",
                    ))
                    insight_id += 1
                elif pattern_type == 'duplicate_column':
                    # 중복 컬럼 패턴
                    insights.append(PalantirInsight(
                        insight_id=f"DQ_{insight_id:03d}",
                        insight_type="data_quality_issue",
                        source_table="",
                        source_column=columns[0] if columns else "",
                        source_condition=f"LLM 탐지: {description}",
                        target_table="",
                        target_column=columns[1] if len(columns) > 1 else "",
                        predicted_impact=implication,
                        correlation_coefficient=1.0,
                        p_value=0.0,
                        confidence=1.0,
                        sample_size=0,
                        business_interpretation=f"[LLM 분석] {description}",
                        recommended_action="데이터 수집 프로세스를 검토하세요.",
                        risk_assessment="중복 데이터로 인해 분석 결과가 왜곡될 수 있습니다.",
                        opportunity_assessment="데이터 품질 개선을 통해 더 정확한 분석이 가능합니다.",
                    ))
                    insight_id += 1

            # v13.0: Phase 1에서 탐지한 data_quality_issues를 인사이트로 변환
            quality_issues = self._phase1_analysis.get('data_quality_issues', [])
            for issue in quality_issues[:2]:  # 최대 2개
                insights.append(PalantirInsight(
                    insight_id=f"DQ_{insight_id:03d}",
                    insight_type="data_quality_issue",
                    source_table="",
                    source_column=issue.get('columns', [''])[0] if issue.get('columns') else "",
                    source_condition=f"LLM 탐지: {issue.get('issue_type', '')}",
                    target_table="",
                    target_column="",
                    predicted_impact=issue.get('description', ''),
                    correlation_coefficient=0.0,
                    p_value=0.0,
                    confidence=0.9,
                    sample_size=0,
                    business_interpretation=f"[LLM 분석] {issue.get('description', '')}",
                    recommended_action="데이터 품질 개선이 필요합니다.",
                    risk_assessment=f"심각도: {issue.get('severity', 'unknown')}",
                    opportunity_assessment="품질 개선으로 분석 정확도 향상 가능",
                ))
                insight_id += 1

            logger.info(f"[v13.0] Created {insight_id - 1} insights from Phase 1 analysis")

        # v12.2: 카테고리/세그먼트 인사이트를 먼저 추가 (가장 가치있는 인사이트)
        # (Phase 1 분석 결과가 없거나 카테고리 인사이트가 추가로 필요한 경우)
        if hasattr(self, '_categorical_insights') and self._categorical_insights:
            for cat in self._categorical_insights[:5]:  # 최대 5개
                top_cats = cat.get('top_categories', [])
                if not top_cats:
                    continue

                # TOP 카테고리 요약
                top_summary = ", ".join([f"{c['category']}({c['percentage']:.1f}%)" for c in top_cats[:3]])

                # 집중도 분석
                concentration = cat.get('top5_concentration', 0)
                concentration_level = "매우 높음" if concentration > 80 else "높음" if concentration > 60 else "보통"

                insights.append(PalantirInsight(
                    insight_id=f"SEG_{insight_id:03d}",
                    insight_type="segment_analysis",
                    source_table=cat.get('table', ''),
                    source_column=cat.get('category_column', ''),
                    source_condition=f"{cat.get('category_column', '')}별 {cat.get('metric_column', '')} 분석",
                    target_table=cat.get('table', ''),
                    target_column=cat.get('metric_column', ''),
                    predicted_impact=f"상위 5개 {cat.get('category_column', '')}가 전체의 {concentration:.1f}%를 차지 ({concentration_level} 집중도)",
                    correlation_coefficient=concentration / 100,  # 집중도를 0-1 스케일로
                    p_value=0.0,
                    confidence=0.9,
                    sample_size=cat.get('total_categories', 0),
                    business_interpretation=f"{cat.get('metric_column', '')} 기준 상위 유입경로: {top_summary}. "
                                           f"총 {cat.get('total_categories', 0)}개 {cat.get('category_column', '')} 중 "
                                           f"상위 5개가 {concentration:.1f}%를 차지합니다.",
                    recommended_action=f"상위 {cat.get('category_column', '')}({top_cats[0]['category'] if top_cats else ''})에 대한 "
                                      f"마케팅 투자 강화 및 하위 채널 효율성 검토를 권장합니다.",
                    risk_assessment=f"{'소수 채널 의존도가 높아 리스크 분산 필요' if concentration > 70 else '채널 분산이 양호합니다'}",
                    opportunity_assessment=f"{'고집중 채널 최적화로 ROI 극대화 가능' if concentration > 70 else '다양한 채널에서 성장 기회 존재'}",
                ))
                insight_id += 1

        # 수학적 제약조건 인사이트 (데이터 품질)
        if hasattr(self, '_mathematical_constraints') and self._mathematical_constraints:
            for c in self._mathematical_constraints[:2]:  # 최대 2개 (카테고리 인사이트 우선)
                insights.append(PalantirInsight(
                    insight_id=f"DQ_{insight_id:03d}",
                    insight_type="data_quality_issue",
                    source_table=c.get('table', ''),
                    source_column=c.get('column_1', ''),
                    source_condition=f"수학적 제약조건: {c.get('column_1', '')} + {c.get('column_2', '')} = {c.get('sum_value', 0):.2f}",
                    target_table=c.get('table', ''),
                    target_column=c.get('column_2', ''),
                    predicted_impact="이 두 컬럼의 상관관계는 수학적 정의 관계이며, 비즈니스 인과관계가 아닙니다.",
                    correlation_coefficient=-1.0,
                    p_value=0.0,
                    confidence=1.0,
                    sample_size=0,
                    business_interpretation=f"{c.get('column_1', '')}과 {c.get('column_2', '')}는 합이 항상 {c.get('sum_value', 0):.2f}인 보완 관계입니다. 분석 가치가 없습니다.",
                    recommended_action="이 컬럼 쌍은 상관관계 분석에서 제외됩니다.",
                    risk_assessment="수학적 관계를 인과관계로 오해하지 마세요.",
                    opportunity_assessment="세그먼트/카테고리 분석에 집중하세요.",
                ))
                insight_id += 1

        # 동일 데이터 컬럼 인사이트 (데이터 품질)
        if hasattr(self, '_duplicate_columns') and self._duplicate_columns:
            for d in self._duplicate_columns[:2]:  # 최대 2개
                insights.append(PalantirInsight(
                    insight_id=f"DQ_{insight_id:03d}",
                    insight_type="data_quality_issue",
                    source_table=d.get('table', ''),
                    source_column=d.get('column_1', ''),
                    source_condition=f"동일 데이터: {d.get('column_1', '')}과 {d.get('column_2', '')}가 {d.get('match_rate', 0)*100:.1f}% 일치",
                    target_table=d.get('table', ''),
                    target_column=d.get('column_2', ''),
                    predicted_impact="두 컬럼이 사실상 동일한 데이터입니다.",
                    correlation_coefficient=1.0,
                    p_value=0.0,
                    confidence=1.0,
                    sample_size=0,
                    business_interpretation=f"'{d.get('column_1', '')}'과 '{d.get('column_2', '')}'가 동일합니다. 데이터 수집 오류 또는 의도적 중복일 수 있습니다.",
                    recommended_action="데이터 수집 프로세스를 검토하세요.",
                    risk_assessment="중복 데이터로 인해 분석 결과가 왜곡될 수 있습니다.",
                    opportunity_assessment="데이터 품질 개선을 통해 더 정확한 분석이 가능합니다.",
                ))
                insight_id += 1

        # 인사이트가 없으면 기본 인사이트 생성
        if not insights:
            insights.append(PalantirInsight(
                insight_id="INFO_001",
                insight_type="no_insights",
                source_table="",
                source_column="",
                source_condition="분석 가능한 패턴을 찾지 못함",
                target_table="",
                target_column="",
                predicted_impact="추가 데이터 소스가 필요합니다.",
                correlation_coefficient=0.0,
                p_value=1.0,
                confidence=0.0,
                sample_size=0,
                business_interpretation="현재 데이터에서 유의미한 인사이트를 찾지 못했습니다.",
                recommended_action="외부 데이터 소스 연동, 시계열 데이터 추가를 고려하세요.",
                risk_assessment="분석 가능한 데이터 부족",
                opportunity_assessment="데이터 확장을 통한 인사이트 발굴",
            ))

        logger.info(f"[v12.2] Created {len(insights)} insights (including segment analysis)")
        return insights

    def _prepare_correlation_summary_for_llm(self, correlations: List[CorrelationResult]) -> str:
        """LLM에게 전달할 상관관계 요약"""
        if not correlations:
            return "유의미한 상관관계가 발견되지 않았습니다."

        lines = []
        for i, corr in enumerate(correlations[:10], 1):
            direction = "양의" if corr.correlation > 0 else "음의"
            strength = "강한" if abs(corr.correlation) > 0.7 else "중간" if abs(corr.correlation) > 0.5 else "약한"

            lines.append(
                f"{i}. {corr.source_table}.{corr.source_column} ↔ {corr.target_table}.{corr.target_column}\n"
                f"   상관계수: {corr.correlation:.3f} ({strength} {direction} 상관관계)\n"
                f"   p-value: {corr.p_value:.4f}, 샘플 크기: {corr.sample_size}"
            )

        return "\n".join(lines)

    async def _llm_discover_scenarios(
        self,
        data_summary: str,
        correlation_summary: str,
        domain_context: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        v2.0: LLM이 자율적으로 분석 시나리오 탐지/설계

        LLM이 데이터를 보고:
        1. 어떤 관계가 비즈니스적으로 중요한지 판단
        2. 어떤 what-if 시나리오를 테스트해야 하는지 설계
        3. 임계값, 변화율 등을 데이터 기반으로 자동 결정
        """
        if not self.llm_client:
            logger.warning("LLM client not available, falling back to basic analysis")
            return []

        domain_hint = f"\n도메인: {domain_context}" if domain_context else ""

        prompt = f"""당신은 팔란티어 Foundry 스타일의 데이터 분석 전문가입니다.
아래 데이터와 발견된 상관관계를 분석하여, 비즈니스에 가치있는 예측 시나리오를 설계해주세요.

## 데이터 요약
{data_summary}

## 발견된 상관관계
{correlation_summary}
{domain_hint}

## 요청사항

다음 형식의 JSON 배열로 분석 시나리오를 설계해주세요:

```json
[
  {{
    "scenario_id": "S001",
    "scenario_type": "threshold_impact" | "percentage_change" | "trend_projection" | "anomaly_effect",
    "priority": "high" | "medium" | "low",
    "source": {{
      "table": "테이블명",
      "column": "컬럼명",
      "condition_type": "exceeds" | "drops_below" | "increases_by" | "decreases_by" | "reaches",
      "threshold_value": 숫자 또는 null,
      "change_percentage": 숫자 또는 null,
      "rationale": "이 조건을 선택한 이유 (데이터 기반)"
    }},
    "expected_impacts": [
      {{
        "table": "영향받는 테이블",
        "column": "영향받는 컬럼",
        "impact_type": "increase" | "decrease" | "volatility",
        "estimated_magnitude": "예상 변화량 또는 비율",
        "confidence": 0.0-1.0
      }}
    ],
    "business_significance": "이 시나리오가 비즈니스에 중요한 이유",
    "recommended_monitoring": "모니터링해야 할 지표",
    "risk_level": "high" | "medium" | "low",
    "opportunity_potential": "high" | "medium" | "low"
  }}
]
```

## 중요 가이드라인
1. 데이터의 실제 분포(평균, 표준편차, 범위)를 고려하여 현실적인 임계값 설정
2. 단순 상관관계가 아닌 비즈니스 인과관계 관점에서 해석
3. 체인 효과 (A→B→C) 가 있다면 함께 분석
4. 리스크와 기회를 모두 고려
5. 최소 3개, 최대 8개의 시나리오 제안

JSON만 출력하세요. 다른 텍스트는 포함하지 마세요."""

        try:
            response = await self.llm_client.chat.completions.create(
                model=get_service_model("cross_entity_scenario"),
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=4000,
            )

            content = response.choices[0].message.content.strip()

            # JSON 추출
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]

            scenarios = json.loads(content)
            logger.info(f"LLM designed {len(scenarios)} analysis scenarios")
            return scenarios

        except Exception as e:
            logger.warning(f"LLM scenario discovery failed: {e}")
            return []

    async def _llm_run_simulation(
        self,
        scenario: Dict[str, Any],
        tables_data: Dict[str, Any],
        correlation: Optional[CorrelationResult] = None,
    ) -> SimulationResult:
        """
        v2.0: LLM 기반 시뮬레이션 실행

        실제 데이터를 사용하여 what-if 시나리오 테스트:
        1. 조건에 맞는 데이터 서브셋 추출
        2. 해당 조건에서 타겟 변수의 실제 변화 분석
        3. 예측 정확도 검증
        """
        import pandas as pd

        source_info = scenario.get("source", {})
        source_table = source_info.get("table", "")
        source_column = source_info.get("column", "")

        if source_table not in tables_data or source_column not in tables_data[source_table].columns:
            return SimulationResult(
                scenario_id=scenario.get("scenario_id", "unknown"),
                scenario_description="Invalid scenario",
                source_change={},
                predicted_impacts=[],
                validation_score=0.0,
                sample_evidence=[],
            )

        df_source = tables_data[source_table]
        source_series = df_source[source_column].dropna()

        # 조건에 따른 데이터 분할
        condition_type = source_info.get("condition_type", "")
        threshold = source_info.get("threshold_value")
        change_pct = source_info.get("change_percentage")

        # 실제 데이터에서 증거 수집
        evidence = []
        predicted_impacts = []
        validation_score = 0.0

        if threshold is not None and condition_type in ["exceeds", "drops_below", "reaches"]:
            # 임계값 기반 분석
            if condition_type == "exceeds":
                high_mask = source_series > threshold
                low_mask = source_series <= threshold
            elif condition_type == "drops_below":
                high_mask = source_series >= threshold
                low_mask = source_series < threshold
            else:
                # reaches: 근처 값
                range_pct = 0.1 * threshold if threshold != 0 else 1
                high_mask = abs(source_series - threshold) <= range_pct
                low_mask = ~high_mask

            if high_mask.sum() > 5 and low_mask.sum() > 5:
                # 각 타겟 컬럼에 대해 영향 분석
                for impact in scenario.get("expected_impacts", []):
                    target_table = impact.get("table", "")
                    target_column = impact.get("column", "")

                    if target_table in tables_data and target_column in tables_data[target_table].columns:
                        df_target = tables_data[target_table]
                        min_rows = min(len(df_source), len(df_target))

                        target_series = df_target[target_column].iloc[:min_rows]
                        source_mask_trimmed = high_mask.iloc[:min_rows] if len(high_mask) > min_rows else high_mask
                        low_mask_trimmed = low_mask.iloc[:min_rows] if len(low_mask) > min_rows else low_mask

                        try:
                            high_group_mean = target_series[source_mask_trimmed].mean()
                            low_group_mean = target_series[low_mask_trimmed].mean()

                            if low_group_mean != 0:
                                actual_change_pct = ((high_group_mean - low_group_mean) / abs(low_group_mean)) * 100

                                predicted_impacts.append({
                                    "table": target_table,
                                    "column": target_column,
                                    "baseline_mean": float(low_group_mean),
                                    "condition_mean": float(high_group_mean),
                                    "actual_change_pct": float(actual_change_pct),
                                    "high_group_size": int(source_mask_trimmed.sum()),
                                    "low_group_size": int(low_mask_trimmed.sum()),
                                })

                                evidence.append({
                                    "description": f"{source_column}이 {threshold}를 {'초과' if condition_type == 'exceeds' else '미만'}할 때, "
                                                   f"{target_column}의 평균이 {low_group_mean:.2f}에서 {high_group_mean:.2f}로 변화 ({actual_change_pct:+.1f}%)",
                                    "sample_count": int(source_mask_trimmed.sum()),
                                })

                                validation_score = min(1.0, abs(actual_change_pct) / 100 * 0.5 + 0.5)
                        except Exception as e:
                            logger.debug(f"Impact analysis failed for {target_column}: {e}")

        elif change_pct is not None:
            # 변화율 기반 분석 (상위/하위 분위수 비교)
            q_low = source_series.quantile(0.25)
            q_high = source_series.quantile(0.75)

            low_mask = source_series <= q_low
            high_mask = source_series >= q_high

            if high_mask.sum() > 5 and low_mask.sum() > 5:
                for impact in scenario.get("expected_impacts", []):
                    target_table = impact.get("table", "")
                    target_column = impact.get("column", "")

                    if target_table in tables_data and target_column in tables_data[target_table].columns:
                        df_target = tables_data[target_table]
                        min_rows = min(len(df_source), len(df_target))

                        target_series = df_target[target_column].iloc[:min_rows]
                        high_mask_trimmed = high_mask.iloc[:min_rows]
                        low_mask_trimmed = low_mask.iloc[:min_rows]

                        try:
                            high_group_mean = target_series[high_mask_trimmed].mean()
                            low_group_mean = target_series[low_mask_trimmed].mean()

                            if low_group_mean != 0:
                                actual_change_pct = ((high_group_mean - low_group_mean) / abs(low_group_mean)) * 100

                                predicted_impacts.append({
                                    "table": target_table,
                                    "column": target_column,
                                    "q25_mean": float(low_group_mean),
                                    "q75_mean": float(high_group_mean),
                                    "actual_change_pct": float(actual_change_pct),
                                })

                                evidence.append({
                                    "description": f"{source_column} 상위 25%와 하위 25% 비교 시, "
                                                   f"{target_column}이 {actual_change_pct:+.1f}% 차이",
                                    "sample_count": int(high_mask_trimmed.sum()) + int(low_mask_trimmed.sum()),
                                })

                                validation_score = min(1.0, abs(actual_change_pct) / 100 * 0.5 + 0.5)
                        except Exception as e:
                            logger.debug(f"Quartile analysis failed: {e}")

        return SimulationResult(
            scenario_id=scenario.get("scenario_id", str(uuid.uuid4())[:8]),
            scenario_description=scenario.get("business_significance", ""),
            source_change={
                "table": source_table,
                "column": source_column,
                "condition_type": condition_type,
                "threshold": threshold,
                "change_pct": change_pct,
            },
            predicted_impacts=predicted_impacts,
            validation_score=validation_score,
            sample_evidence=evidence,
        )

    async def _llm_generate_insights(
        self,
        scenarios: List[Dict[str, Any]],
        simulation_results: List[SimulationResult],
        correlations: List[CorrelationResult],
        domain_context: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        v2.0: LLM이 시뮬레이션 결과를 바탕으로 최종 인사이트 생성

        단순 템플릿이 아닌, 실제 데이터 증거와 비즈니스 맥락을 종합한 인사이트
        """
        if not self.llm_client:
            return []

        # 시뮬레이션 결과 요약
        sim_summary = []
        for sim in simulation_results:
            if sim.predicted_impacts:
                impacts_str = "; ".join([
                    f"{imp.get('column')}: {imp.get('actual_change_pct', 0):+.1f}%"
                    for imp in sim.predicted_impacts
                ])
                sim_summary.append(
                    f"시나리오 {sim.scenario_id}: {sim.scenario_description}\n"
                    f"  조건: {sim.source_change}\n"
                    f"  실제 영향: {impacts_str}\n"
                    f"  검증 점수: {sim.validation_score:.2f}"
                )

        # 상관관계 요약
        corr_summary = self._prepare_correlation_summary_for_llm(correlations[:5])

        domain_hint = f"\n도메인: {domain_context}" if domain_context else ""

        prompt = f"""당신은 팔란티어 스타일의 비즈니스 인텔리전스 전문가입니다.
아래 시뮬레이션 결과와 상관관계 분석을 바탕으로 실행 가능한 예측 인사이트를 생성해주세요.

## 시뮬레이션 결과 (실제 데이터 검증됨)
{chr(10).join(sim_summary) if sim_summary else "시뮬레이션 데이터 없음"}

## 발견된 상관관계
{corr_summary}
{domain_hint}

## 요청사항

다음 형식으로 비즈니스 인사이트를 생성해주세요:

```json
[
  {{
    "insight_id": "PI_001",
    "insight_type": "cross_entity_impact" | "threshold_alert" | "predictive_correlation" | "chain_effect",
    "headline": "한 문장으로 된 핵심 인사이트 (예: 'Daily_Capacity가 1000을 초과하면 운송비용이 15% 증가합니다')",
    "source_condition": "조건 설명 (구체적인 숫자 포함)",
    "predicted_impact": "예측되는 영향 (구체적인 숫자 포함)",
    "confidence_level": 0.0-1.0,
    "evidence_summary": "이 예측을 뒷받침하는 데이터 증거 요약",
    "business_interpretation": {{
      "what_it_means": "비즈니스적으로 이것이 의미하는 바",
      "why_it_matters": "왜 이것이 중요한지",
      "potential_root_cause": "가능한 근본 원인"
    }},
    "risk_assessment": {{
      "risk_level": "high" | "medium" | "low",
      "potential_negative_outcomes": ["부정적 결과 목록"],
      "mitigation_strategies": ["위험 완화 전략"]
    }},
    "opportunity_assessment": {{
      "opportunity_level": "high" | "medium" | "low",
      "potential_benefits": ["잠재적 이점 목록"],
      "action_items": ["실행 항목"]
    }},
    "recommended_actions": [
      {{
        "action": "구체적인 액션",
        "priority": "immediate" | "short_term" | "long_term",
        "expected_outcome": "예상 결과"
      }}
    ],
    "monitoring_kpis": ["모니터링해야 할 KPI 목록"],
    "chain_effects": ["A→B→C 형태의 체인 효과가 있다면 기술"]
  }}
]
```

## 중요 가이드라인
1. 실제 시뮬레이션 데이터를 기반으로 현실적인 수치 사용
2. 구체적이고 실행 가능한 권장사항 제시
3. 리스크와 기회를 균형있게 분석
4. 불확실성을 솔직하게 표현 (confidence_level 반영)
5. 체인 효과(A가 B에 영향, B가 C에 영향)가 있다면 명시

JSON만 출력하세요."""

        try:
            response = await self.llm_client.chat.completions.create(
                model=get_service_model("cross_entity_interpret"),
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=4000,
            )

            content = response.choices[0].message.content.strip()

            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]

            insights = json.loads(content)
            logger.info(f"LLM generated {len(insights)} business insights")
            return insights

        except Exception as e:
            logger.warning(f"LLM insight generation failed: {e}")
            return []

    async def analyze(
        self,
        table_names: List[str],
        fk_relations: Optional[List[Dict]] = None,
        entity_matches: Optional[List[Dict]] = None,
        domain_context: Optional[str] = None,
    ) -> List[PalantirInsight]:
        """
        v2.0: 전체 분석 파이프라인 (LLM 자율 분석)

        1. 데이터 로딩 및 통계 계산
        2. 상관관계 분석
        3. LLM 시나리오 탐지/설계
        4. 시뮬레이션 실행
        5. LLM 인사이트 생성
        """
        import pandas as pd

        # 1. 테이블 데이터 로드
        tables_data: Dict[str, pd.DataFrame] = {}
        for table_name in table_names:
            df = self.load_table_data(table_name)
            if df is not None and len(df) > 0:
                tables_data[table_name] = df

        if not tables_data:
            logger.warning("No valid tables loaded")
            return []

        logger.info(f"Loaded {len(tables_data)} tables for analysis")

        # 2. 상관관계 분석
        correlations = self.compute_cross_table_correlations(table_names)

        # 3. LLM 시나리오 탐지/설계
        data_summary = self._prepare_data_summary_for_llm(tables_data)
        correlation_summary = self._prepare_correlation_summary_for_llm(correlations)

        scenarios = await self._llm_discover_scenarios(
            data_summary, correlation_summary, domain_context
        )

        # 4. 시뮬레이션 실행
        simulation_results = []
        for scenario in scenarios:
            sim_result = await self._llm_run_simulation(
                scenario, tables_data,
                correlations[0] if correlations else None
            )
            if sim_result.predicted_impacts:
                simulation_results.append(sim_result)

        logger.info(f"Completed {len(simulation_results)} simulations with valid results")

        # 5. LLM 인사이트 생성
        llm_insights = await self._llm_generate_insights(
            scenarios, simulation_results, correlations, domain_context
        )

        # PalantirInsight 객체로 변환
        insights = []
        for i, llm_insight in enumerate(llm_insights):
            # 관련 상관관계 찾기
            corr = correlations[i] if i < len(correlations) else None
            sim = simulation_results[i] if i < len(simulation_results) else None

            insight = PalantirInsight(
                insight_id=llm_insight.get("insight_id", f"PI_{i+1:03d}"),
                insight_type=llm_insight.get("insight_type", "cross_entity_impact"),
                source_table=corr.source_table if corr else "",
                source_column=corr.source_column if corr else "",
                source_condition=llm_insight.get("source_condition", ""),
                target_table=corr.target_table if corr else "",
                target_column=corr.target_column if corr else "",
                predicted_impact=llm_insight.get("predicted_impact", ""),
                correlation_coefficient=corr.correlation if corr else 0.0,
                p_value=corr.p_value if corr else 1.0,
                confidence=llm_insight.get("confidence_level", 0.5),
                sample_size=corr.sample_size if corr else 0,
                business_interpretation=json.dumps(
                    llm_insight.get("business_interpretation", {}),
                    ensure_ascii=False
                ),
                recommended_action=json.dumps(
                    llm_insight.get("recommended_actions", []),
                    ensure_ascii=False
                ),
                risk_assessment=json.dumps(
                    llm_insight.get("risk_assessment", {}),
                    ensure_ascii=False
                ),
                opportunity_assessment=json.dumps(
                    llm_insight.get("opportunity_assessment", {}),
                    ensure_ascii=False
                ),
                simulation_results=[asdict(sim)] if sim else [],
                validation_evidence=sim.sample_evidence if sim else [],
                source_entity=corr.source_table.replace("_", " ").title() if corr else "",
                target_entity=corr.target_table.replace("_", " ").title() if corr else "",
                chain_effects=llm_insight.get("chain_effects", []),
            )
            insights.append(insight)

        # 상관관계는 있지만 LLM 인사이트가 없는 경우 기본 인사이트 추가
        if correlations and not insights:
            insights = self._create_basic_insights_from_correlations(correlations)

        logger.info(f"Generated {len(insights)} total Palantir-style insights")
        return insights

    def _create_basic_insights_from_correlations(
        self,
        correlations: List[CorrelationResult],
    ) -> List[PalantirInsight]:
        """LLM 없이 상관관계에서 기본 인사이트 생성 (폴백)"""
        insights = []

        for i, corr in enumerate(correlations[:10]):
            # 데이터 기반 동적 시나리오 생성
            source_stats = corr.source_stats

            # 표준편차 기반 변화량 계산
            if source_stats:
                std = source_stats.get("std", 0)
                mean = source_stats.get("mean", 1)
                if mean != 0:
                    # 1 표준편차 변화가 몇 %인지 계산
                    one_std_pct = (std / abs(mean)) * 100
                    change_pct = min(50, max(10, one_std_pct))  # 10-50% 범위로 제한
                else:
                    change_pct = 20
            else:
                change_pct = 20

            # 예측 영향 계산
            predicted_change = abs(corr.correlation * change_pct)
            direction = "증가" if corr.correlation > 0 else "감소"

            insight = PalantirInsight(
                insight_id=f"PI_{i+1:03d}",
                insight_type="cross_entity_impact",
                source_table=corr.source_table,
                source_column=corr.source_column,
                source_condition=f"{change_pct:.0f}% 변화 시",
                target_table=corr.target_table,
                target_column=corr.target_column,
                predicted_impact=f"{predicted_change:.1f}% {direction}",
                correlation_coefficient=corr.correlation,
                p_value=corr.p_value,
                confidence=min(0.95, (1 - corr.p_value) * 0.9),
                sample_size=corr.sample_size,
                business_interpretation=f"{corr.source_column}와 {corr.target_column} 사이에 "
                                        f"{'강한' if abs(corr.correlation) > 0.7 else '중간 수준의'} "
                                        f"{'양의' if corr.correlation > 0 else '음의'} 상관관계가 있습니다.",
                recommended_action=f"{corr.source_column} 변화 시 {corr.target_column} 모니터링 권장",
                source_entity=corr.source_table.replace("_", " ").title(),
                target_entity=corr.target_table.replace("_", " ").title(),
            )
            insights.append(insight)

        return insights

    def analyze_sync(
        self,
        table_names: List[str],
        fk_relations: Optional[List[Dict]] = None,
        entity_matches: Optional[List[Dict]] = None,
        domain_context: Optional[str] = None,
        phase1_analysis: Optional[Dict[str, Any]] = None,  # v13.0: Phase 1 분석 결과
    ) -> List[PalantirInsight]:
        """
        v13.0: LLM-First 분석 결과 활용 + 시뮬레이션 기반 분석

        v13.0 핵심 변경:
        - Phase 1 DataAnalystAgent의 분석 결과(data_patterns, recommended_analysis)를 활용
        - LLM이 이미 탐지한 패턴/품질이슈를 중복으로 계산하지 않음
        - 추천된 분석 방향(key_dimensions, key_metrics)에 따라 인사이트 생성

        1. Phase 1 분석 결과 활용
        2. 상관관계 분석 (Phase 1 패턴 제외)
        3. 시뮬레이션 엔진으로 실제 데이터 실험 수행
        4. LLM이 시뮬레이션 결과를 바탕으로 인사이트 도출
        5. Phase 1 추천 분석 방향에 따른 인사이트 보완
        """
        import pandas as pd
        from .simulation_engine import SimulationEngine

        # v13.0: Phase 1 분석 결과 저장
        self._phase1_analysis = phase1_analysis or {}
        if phase1_analysis:
            logger.info(f"[v13.0] Received Phase 1 analysis: "
                       f"domain={phase1_analysis.get('data_understanding', {}).get('domain', {}).get('detected', 'unknown')}, "
                       f"patterns={len(phase1_analysis.get('data_patterns', []))}, "
                       f"quality_issues={len(phase1_analysis.get('data_quality_issues', []))}")

            # Phase 1에서 탐지한 key dimensions/metrics 사용
            recommended = phase1_analysis.get('recommended_analysis', {})
            self._key_dimensions = recommended.get('key_dimensions', [])
            self._key_metrics = recommended.get('key_metrics', [])
            logger.info(f"[v13.0] Key dimensions: {self._key_dimensions}, Key metrics: {self._key_metrics}")

        # 1. 테이블 데이터 로드
        tables_data: Dict[str, pd.DataFrame] = {}
        for table_name in table_names:
            df = self.load_table_data(table_name)
            if df is not None and len(df) > 0:
                tables_data[table_name] = df

        if not tables_data:
            logger.warning("No valid tables loaded for correlation analysis")
            return []

        logger.info(f"Loaded {len(tables_data)} tables for Cross-Entity Correlation Analysis")

        # v12.0: 데이터 품질 검사 수행 (수학적 제약조건, 동일 데이터 감지)
        logger.info("[v12.0] Running data quality checks...")
        _ = self._prepare_data_summary_for_llm(tables_data)  # 이 함수가 품질 검사 수행

        # 감지된 이슈 로깅
        if hasattr(self, '_mathematical_constraints') and self._mathematical_constraints:
            for c in self._mathematical_constraints:
                logger.warning(f"[v12.0] Mathematical constraint: {c['reason']}")

        if hasattr(self, '_duplicate_columns') and self._duplicate_columns:
            for d in self._duplicate_columns:
                logger.warning(f"[v12.0] Duplicate columns: {d['reason']}")

        if hasattr(self, '_categorical_insights') and self._categorical_insights:
            logger.info(f"[v12.0] Found {len(self._categorical_insights)} categorical insights")

        # v26.0: 파생 컬럼 로깅
        if hasattr(self, '_derived_columns') and self._derived_columns:
            for d in self._derived_columns:
                logger.info(f"[v26.0] Derived column: {d['reason']}")

        # v26.0: 교란변수 탐지 (단일 테이블 내)
        self._confounding_results = []
        for table_name, df in tables_data.items():
            numeric_cols = self.get_numeric_columns(df)
            # 주요 상관 쌍에 대해 교란변수 체크
            for i, col_a in enumerate(numeric_cols[:8]):
                for col_b in numeric_cols[i+1:]:
                    try:
                        from scipy import stats
                        mask = df[[col_a, col_b]].notna().all(axis=1)
                        if mask.sum() < 30:
                            continue
                        corr_val, p_val = stats.pearsonr(df.loc[mask, col_a], df.loc[mask, col_b])
                        if abs(corr_val) > 0.3 and p_val < 0.05:
                            confounders = self.detect_confounding_variables(
                                df, col_a, col_b, numeric_cols
                            )
                            if confounders:
                                self._confounding_results.extend(confounders)
                    except Exception as e:
                        logger.debug(f"Confounding analysis failed for {col_a} vs {col_b}: {e}")

        if self._confounding_results:
            logger.info(f"[v26.0] Found {len(self._confounding_results)} confounding relationships")

        # 2. 상관관계 분석
        correlations = self.compute_cross_table_correlations(table_names)

        if not correlations:
            logger.info("No significant correlations found - returning data quality insights")
            return self._create_data_quality_insights_only()

        logger.info(f"Found {len(correlations)} significant correlations (before filtering)")

        # v12.1: 수학적 제약조건 및 동일 데이터 컬럼 쌍을 필터링 (LLM 의존 X, 코드에서 직접 제거)
        original_count = len(correlations)
        correlations = self._filter_invalid_correlations(correlations)
        filtered_count = original_count - len(correlations)
        logger.info(f"[v12.1] After filtering: {len(correlations)} valid correlations (removed {filtered_count})")

        # v12.1: 필터링 후에도 체크 - 유효한 상관관계가 없으면 데이터 품질 인사이트만 반환
        if not correlations:
            logger.info("[v12.1] All correlations filtered out - returning data quality insights only")
            return self._create_data_quality_insights_only()

        # 3. 시뮬레이션 엔진 실행 (v10.1 NEW!)
        logger.info("Running simulation engine...")
        sim_engine = SimulationEngine(tables_data)
        simulation_report = sim_engine.run_all_experiments(correlations)

        logger.info(f"Simulation complete: {simulation_report.total_experiments} experiments, "
                   f"{len(simulation_report.significant_findings)} significant findings")

        # 4. LLM에게 시뮬레이션 결과 전달하여 인사이트 생성
        insights = []

        # 실험이 있으면 (유의미한 발견이 없더라도) LLM에게 전달
        if self.llm_client and simulation_report.total_experiments > 0:
            logger.info("LLM analyzing simulation results...")
            try:
                # 시뮬레이션 결과를 LLM에게 전달
                simulation_summary = sim_engine.format_for_llm(simulation_report)
                insights = self._llm_analyze_simulation_results(
                    simulation_summary, correlations, domain_context
                )
                if insights:
                    logger.info(f"LLM generated {len(insights)} insights from simulation")
                    return insights
            except Exception as e:
                logger.warning(f"LLM simulation analysis failed: {e}")
                import traceback
                logger.debug(traceback.format_exc())

        # 5. 폴백: 기본 인사이트 생성
        insights = self._create_basic_insights_from_correlations(correlations)

        logger.info(f"Generated {len(insights)} Palantir-style insights (sync mode)")
        return insights

    def _sync_llm_analysis(
        self,
        tables_data: Dict[str, Any],
        correlations: List[CorrelationResult],
        domain_context: Optional[str] = None,
    ) -> List[PalantirInsight]:
        """동기 방식의 LLM 분석"""
        if not self.llm_client:
            return []

        # 데이터 요약 준비
        data_summary = self._prepare_data_summary_for_llm(tables_data)
        corr_summary = self._prepare_correlation_summary_for_llm(correlations)

        domain_hint = f"\n도메인: {domain_context}" if domain_context else ""

        # 단일 통합 프롬프트로 시나리오 + 인사이트 생성
        prompt = f"""당신은 팔란티어 Foundry 스타일의 데이터 분석 전문가입니다.
아래 데이터와 상관관계를 분석하여 비즈니스에 가치있는 예측 인사이트를 생성해주세요.

## 데이터 요약
{data_summary}

## 발견된 상관관계
{corr_summary}
{domain_hint}

## 요청사항

각 상관관계에 대해 다음 형식으로 인사이트를 생성해주세요:

```json
[
  {{
    "insight_id": "PI_001",
    "insight_type": "cross_entity_impact",
    "source_table": "소스 테이블명",
    "source_column": "소스 컬럼명",
    "target_table": "타겟 테이블명",
    "target_column": "타겟 컬럼명",
    "source_condition": "구체적인 조건 (데이터 분포 기반, 예: 'Daily_Capacity가 평균 대비 1 표준편차 증가 시')",
    "predicted_impact": "구체적인 영향 예측 (예: 'Max_Weight_Quant가 약 12% 증가')",
    "confidence": 0.0-1.0,
    "business_interpretation": "비즈니스 관점의 해석",
    "risk_assessment": "리스크 평가",
    "opportunity_assessment": "기회 평가",
    "recommended_actions": ["구체적인 권장 액션 목록"],
    "chain_effects": ["A→B→C 형태의 연쇄 효과 (있다면)"],
    "simulation_scenarios": [
      {{
        "scenario": "시나리오 설명",
        "condition": "조건",
        "expected_outcome": "예상 결과"
      }}
    ]
  }}
]
```

## 중요 가이드라인
1. 데이터의 실제 통계(평균, 표준편차, 범위)를 사용하여 현실적인 시나리오 설계
2. 20%만 사용하지 말고, 데이터 분포에 맞는 다양한 변화율 사용
3. 임계값 시나리오도 포함 (예: "X가 Y를 초과하면...")
4. 체인 효과 탐지 (A→B→C)
5. 리스크와 기회를 균형있게 분석
6. 구체적이고 실행 가능한 권장사항

JSON만 출력하세요."""

        try:
            response = self.llm_client.chat.completions.create(
                model=get_service_model("cross_entity_scenario"),
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=4000,
            )

            content = response.choices[0].message.content.strip()

            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]

            llm_insights = json.loads(content)

            # PalantirInsight 객체로 변환
            insights = []
            for llm_insight in llm_insights:
                # 매칭되는 상관관계 찾기
                corr = None
                for c in correlations:
                    if (c.source_table == llm_insight.get("source_table") and
                        c.source_column == llm_insight.get("source_column")):
                        corr = c
                        break

                insight = PalantirInsight(
                    insight_id=llm_insight.get("insight_id", f"PI_{len(insights)+1:03d}"),
                    insight_type=llm_insight.get("insight_type", "cross_entity_impact"),
                    source_table=llm_insight.get("source_table", ""),
                    source_column=llm_insight.get("source_column", ""),
                    source_condition=llm_insight.get("source_condition", ""),
                    target_table=llm_insight.get("target_table", ""),
                    target_column=llm_insight.get("target_column", ""),
                    predicted_impact=llm_insight.get("predicted_impact", ""),
                    correlation_coefficient=corr.correlation if corr else 0.0,
                    p_value=corr.p_value if corr else 1.0,
                    confidence=llm_insight.get("confidence", 0.5),
                    sample_size=corr.sample_size if corr else 0,
                    business_interpretation=llm_insight.get("business_interpretation", ""),
                    recommended_action=json.dumps(
                        llm_insight.get("recommended_actions", []),
                        ensure_ascii=False
                    ),
                    risk_assessment=llm_insight.get("risk_assessment", ""),
                    opportunity_assessment=llm_insight.get("opportunity_assessment", ""),
                    simulation_results=llm_insight.get("simulation_scenarios", []),
                    source_entity=llm_insight.get("source_table", "").replace("_", " ").title(),
                    target_entity=llm_insight.get("target_table", "").replace("_", " ").title(),
                    chain_effects=llm_insight.get("chain_effects", []),
                )
                insights.append(insight)

            logger.info(f"Generated {len(insights)} LLM-powered insights")
            return insights

        except Exception as e:
            logger.warning(f"Sync LLM analysis failed: {e}")
            return []

    def _llm_analyze_simulation_results(
        self,
        simulation_summary: str,
        correlations: List[CorrelationResult],
        domain_context: Optional[str] = None,
    ) -> List[PalantirInsight]:
        """
        v12.0: LLM이 시뮬레이션 결과를 분석하여 인사이트 생성 (강화된 자율 판단)

        핵심 원칙:
        - 시뮬레이션 결과는 LLM의 분석 도구일 뿐
        - 최종 출력에는 비즈니스 인사이트만 포함 (시뮬레이션 상세 X)
        - 사용자가 볼 내용: headline, business_interpretation, risk, opportunity, actions
        - v12.0: LLM이 스스로 무의미한 상관관계를 필터링
        """
        if not self.llm_client:
            return []

        domain_hint = f"\n도메인: {domain_context}" if domain_context else ""

        # v12.0: 데이터 품질 경고 추가
        data_quality_warnings = self._prepare_data_quality_warnings_for_llm()
        categorical_summary = self._prepare_categorical_summary_for_llm()

        prompt = f"""당신은 비즈니스 인텔리전스 전문가입니다.
아래는 실제 데이터로 수행한 시뮬레이션 분석 결과입니다. 이 결과를 바탕으로 경영진이 이해할 수 있는 비즈니스 인사이트를 도출해주세요.

{simulation_summary}
{domain_hint}

{data_quality_warnings}

{categorical_summary}

## 🚨 v12.0 핵심 지침 - 무의미한 인사이트 필터링

당신은 다음 유형의 "가짜 인사이트"를 반드시 제외해야 합니다:

### 1. 수학적 제약조건 (Mathematical Constraints)
- A + B = 1 또는 A + B = 100 관계는 **인과관계가 아닙니다**
- 예: "신규방문비율이 증가하면 재방문비율이 감소" → ❌ 제외 (둘의 합은 항상 1)
- 예: "완료율이 높으면 미완료율이 낮다" → ❌ 제외 (자명한 사실)
- **상관계수가 -1.0 또는 1.0에 가깝다면 수학적 관계일 가능성이 높습니다**

### 2. 동일 데이터 (Duplicate Columns)
- 두 컬럼이 100% 동일하면 **분석 가치가 없습니다**
- 예: "모바일_세션수와 전체_세션수가 상관관계가 있다" → ❌ 제외 (같은 데이터)
- 대신 **"데이터 품질 이슈: 두 컬럼이 동일함"**으로 보고하세요

### 3. 자명한 동어반복 (Tautology)
- "X가 높으면 X_관련지표도 높다" → ❌ 제외
- 컬럼 이름이 비슷하거나 같은 개념을 측정하는 경우 주의

### 4. 가치 없는 상관관계
- 비즈니스적 의미가 없거나 실행 불가능한 인사이트 제외
- "상관관계가 있다"만으로는 인사이트가 아닙니다

## ✅ 좋은 인사이트의 조건
1. **실행 가능**: 누군가가 이 정보로 행동을 취할 수 있어야 함
2. **비즈니스 가치**: 매출, 비용, 리스크, 효율성에 영향을 줘야 함
3. **구체적**: "X가 Y를 초과하면 Z가 15% 증가"처럼 수치 포함
4. **카테고리 기반**: 세그먼트/트래픽 소스/지역별 차이 분석은 매우 유용

## 출력 형식

```json
[
  {{
    "insight_id": "PI_001",
    "headline": "핵심 발견을 한 문장으로 (예: '대용량 창고에서 운송 지연 위험 2배 높음')",
    "insight_type": "actionable_insight" | "data_quality_issue" | "category_analysis",
    "business_interpretation": "이것이 비즈니스에 의미하는 바와 원인 분석",
    "why_this_matters": "왜 이것이 비즈니스에 중요한지 (자명한 관계가 아닌 이유)",
    "risk_level": "high" | "medium" | "low",
    "risk_description": "구체적인 위험 요소와 잠재적 손실",
    "opportunity": "이 발견을 활용할 수 있는 기회",
    "recommended_actions": [
      "즉시 실행할 구체적인 액션 1",
      "단기적으로 검토할 사항 2"
    ],
    "affected_entities": ["영향받는 구체적인 엔티티 목록"],
    "urgency": "immediate" | "short_term" | "long_term",
    "source_table": "관련 소스 테이블",
    "source_column": "관련 소스 컬럼",
    "target_table": "관련 타겟 테이블",
    "target_column": "관련 타겟 컬럼",
    "excluded_false_insights": ["제외한 가짜 인사이트와 그 이유 (선택)"]
  }}
]
```

## 최종 체크리스트
- [ ] 수학적 제약조건(A+B=1)을 인과관계로 해석하지 않았는가?
- [ ] 동일 데이터 컬럼을 상관관계로 보고하지 않았는가?
- [ ] 각 인사이트가 실행 가능하고 비즈니스 가치가 있는가?
- [ ] 카테고리/세그먼트 분석이 포함되었는가?

뻔하지 않은, 실제로 유용한 인사이트만 생성하세요. JSON만 출력하세요."""

        try:
            response = self.llm_client.chat.completions.create(
                model=get_service_model("cross_entity_interpret"),
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=4000,
            )

            content = response.choices[0].message.content.strip()

            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]

            llm_insights = json.loads(content)

            # PalantirInsight 객체로 변환
            insights = []
            for llm_insight in llm_insights:
                # 매칭되는 상관관계 찾기
                corr = None
                for c in correlations:
                    if (c.source_table == llm_insight.get("source_table") and
                        c.source_column == llm_insight.get("source_column")):
                        corr = c
                        break

                insight = PalantirInsight(
                    insight_id=llm_insight.get("insight_id", f"PI_{len(insights)+1:03d}"),
                    insight_type="simulation_based",  # v10.1: 시뮬레이션 기반임을 명시
                    source_table=llm_insight.get("source_table", ""),
                    source_column=llm_insight.get("source_column", ""),
                    source_condition=llm_insight.get("headline", ""),  # headline을 condition으로 사용
                    target_table=llm_insight.get("target_table", ""),
                    target_column=llm_insight.get("target_column", ""),
                    predicted_impact=llm_insight.get("headline", ""),
                    correlation_coefficient=corr.correlation if corr else 0.0,
                    p_value=corr.p_value if corr else 1.0,
                    confidence=0.85 if llm_insight.get("risk_level") == "high" else 0.7,
                    sample_size=corr.sample_size if corr else 0,
                    business_interpretation=llm_insight.get("business_interpretation", ""),
                    recommended_action=json.dumps(
                        llm_insight.get("recommended_actions", []),
                        ensure_ascii=False
                    ),
                    risk_assessment=llm_insight.get("risk_description", ""),
                    opportunity_assessment=llm_insight.get("opportunity", ""),
                    # v10.1: simulation_results는 비워둠 (사용자에게 노출 X)
                    simulation_results=[],
                    source_entity=llm_insight.get("source_table", "").replace("_", " ").title(),
                    target_entity=llm_insight.get("target_table", "").replace("_", " ").title(),
                    chain_effects=llm_insight.get("affected_entities", []),
                )
                insights.append(insight)

            logger.info(f"Generated {len(insights)} simulation-based insights")
            return insights

        except Exception as e:
            logger.warning(f"LLM simulation analysis failed: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return []
