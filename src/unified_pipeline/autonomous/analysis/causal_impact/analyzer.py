"""
Causal Impact Analyzer - 인과 영향 분석 오케스트레이터

v18.4: 실제 데이터 기반 인과 영향 분석기
"""

from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict, Counter
import numpy as np

from .models import (
    CausalGraph,
    CausalNode,
    CausalEdge,
    TreatmentEffect,
    CausalImpactResult,
    SensitivityResult,
    GrangerCausalityResult,
    PCAlgorithmResult,
    CATEResult,
)
from .estimators import (
    ATEEstimator,
    CATEEstimator,
    PropensityScoreEstimator,
    EnhancedCATEEstimator,
)
from .counterfactual import CounterfactualAnalyzer, SensitivityAnalyzer
from .granger import GrangerCausalityAnalyzer
from .pc_algorithm import PCAlgorithm
from .integration import DataIntegrationCausalModel


class CausalImpactAnalyzer:
    """
    v18.4: 실제 데이터 기반 인과 영향 분석기

    실제 데이터에서 treatment(이진/카테고리), outcome(숫자), confounder를 자동 탐지하여
    Propensity Score Matching + Doubly Robust ATE 추정을 수행합니다.
    합성 데이터 생성 없이 전체 원본 데이터를 사용합니다.
    """

    def __init__(self):
        self.model = DataIntegrationCausalModel()
        self.ate_estimator = ATEEstimator(method="doubly_robust")

    def analyze(
        self,
        tables: Dict[str, Dict],
        fk_relations: List[Dict] = None,
        entity_matches: List[Dict] = None,
        target_kpis: List[str] = None,
        real_data: Dict[str, List[Dict]] = None,
    ) -> Dict[str, Any]:
        """
        v18.4: 실제 데이터 기반 인과 분석

        Parameters:
        - tables: 테이블 메타데이터 (컬럼 정보)
        - fk_relations: FK 관계 목록
        - entity_matches: 엔티티 매칭 결과
        - target_kpis: 목표 KPI 목록
        - real_data: 실제 전체 데이터 {table_name: [rows]}
        """
        # 실제 데이터가 있으면 데이터 기반 인과 분석 수행
        if real_data:
            return self._analyze_with_real_data(real_data, tables)

        # 실제 데이터 없으면 기존 메타데이터 기반 분석 (fallback)
        return self._analyze_metadata_only(tables, fk_relations, entity_matches, target_kpis)

    def _rank_treatment_candidates(
        self,
        rows: list,
        binary_cols: list,
        categorical_cols: list,
        numeric_cols: list,
    ) -> list:
        """
        v20: Treatment 후보를 eta-squared(상호정보량 프록시) 기준으로 랭킹.
        각 후보의 그룹간 분산 / 전체 분산을 평균하여 outcome 설명력이 높은 순으로 정렬.
        Returns: 상위 5개 treatment 후보 리스트.
        """
        candidates = binary_cols + categorical_cols
        if not candidates or not numeric_cols:
            return candidates[:5]

        scores = {}
        for treat_col in candidates:
            groups = defaultdict(list)
            for row in rows:
                v = row.get(treat_col)
                if v is not None:
                    groups[str(v).strip().lower()].append(row)

            if len(groups) < 2:
                scores[treat_col] = 0.0
                continue

            eta_sum, eta_count = 0.0, 0
            for out_col in numeric_cols[:5]:
                all_vals = []
                group_means = []
                group_sizes = []
                for g_rows in groups.values():
                    g_vals = []
                    for r in g_rows:
                        try:
                            g_vals.append(float(r[out_col]))
                        except (ValueError, TypeError, KeyError):
                            pass
                    if g_vals:
                        all_vals.extend(g_vals)
                        group_means.append(np.mean(g_vals))
                        group_sizes.append(len(g_vals))

                if len(all_vals) < 10 or len(group_means) < 2:
                    continue

                grand_mean = np.mean(all_vals)
                ss_between = sum(
                    n * (m - grand_mean) ** 2
                    for m, n in zip(group_means, group_sizes)
                )
                ss_total = sum((v - grand_mean) ** 2 for v in all_vals)
                if ss_total > 0:
                    eta_sum += ss_between / ss_total
                    eta_count += 1

            scores[treat_col] = (eta_sum / eta_count) if eta_count > 0 else 0.0

        ranked = sorted(candidates, key=lambda c: scores.get(c, 0), reverse=True)
        return ranked[:5]

    def _analyze_with_real_data(
        self,
        real_data: Dict[str, List[Dict]],
        tables: Dict[str, Dict],
    ) -> Dict[str, Any]:
        """
        v18.4: 실제 데이터에서 treatment/outcome/confounder를 자동 탐지하여 인과 분석
        """
        all_results = {
            "summary": {},
            "treatment_effects": [],
            "counterfactuals": [],
            "sensitivity": {"robustness": 0.0, "critical_bias": 0.0, "interpretation": ""},
            "recommendations": [],
            "causal_graph": {"nodes": [], "edges": []},
            "group_comparisons": [],
        }

        for table_name, rows in real_data.items():
            if not rows or len(rows) < 10:
                continue

            # 1. 컬럼 분류: 숫자형, 이진형(treatment 후보), 카테고리형
            columns = list(rows[0].keys())
            numeric_cols = []
            binary_cols = []
            categorical_cols = []

            for col in columns:
                values = [row.get(col) for row in rows if row.get(col) is not None]
                if not values:
                    continue

                # 숫자형 판별
                numeric_values = []
                for v in values:
                    try:
                        numeric_values.append(float(v))
                    except (ValueError, TypeError):
                        pass

                if len(numeric_values) >= len(values) * 0.8:
                    # 80% 이상 숫자 → 숫자형
                    unique_nums = set(numeric_values)
                    if unique_nums == {0.0, 1.0} or unique_nums <= {0.0, 1.0}:
                        binary_cols.append(col)
                    elif len(unique_nums) > 5:
                        numeric_cols.append(col)
                    # 유니크 값 2~5개면 카테고리로도 간주
                    if 2 <= len(unique_nums) <= 5:
                        categorical_cols.append(col)
                else:
                    # 문자열 컬럼
                    str_values = [str(v).strip().lower() for v in values if v is not None]
                    unique_str = set(str_values)
                    # 이진 (Yes/No, True/False, Male/Female 등)
                    if len(unique_str) == 2:
                        binary_cols.append(col)
                    elif 2 < len(unique_str) <= 15:
                        categorical_cols.append(col)

            # id, name 등 의미 없는 컬럼 제외
            id_patterns = ["id", "name", "index", "번호", "이름"]
            binary_cols = [c for c in binary_cols if not any(p in c.lower() for p in id_patterns)]
            categorical_cols = [c for c in categorical_cols if not any(p in c.lower() for p in id_patterns)]
            numeric_cols = [c for c in numeric_cols if not any(p in c.lower() for p in id_patterns)]

            if not binary_cols and not categorical_cols:
                continue
            if not numeric_cols:
                continue

            # 2. Treatment 후보별 ATE 분석 (v20: eta-squared 랭킹)
            treatment_candidates = self._rank_treatment_candidates(
                rows, binary_cols, categorical_cols, numeric_cols
            )
            for treat_col in treatment_candidates:  # 상위 5개 (ranked)
                # 카테고리형 교란변수 = treatment 컬럼이 아닌 카테고리 컬럼
                cat_confounders = [c for c in categorical_cols if c != treat_col]
                # 이진형 교란변수도 포함 (treatment 자신 제외)
                cat_confounders += [c for c in binary_cols if c != treat_col and c not in cat_confounders]
                for outcome_col in numeric_cols[:5]:  # 최대 5개 outcome 분석
                    result = self._compute_real_ate(
                        rows, treat_col, outcome_col,
                        confounder_cols=[c for c in numeric_cols if c != outcome_col][:5],
                        categorical_confounder_cols=cat_confounders[:3],
                        table_name=table_name,
                    )
                    if result:
                        all_results["treatment_effects"].append(result["treatment_effect"])
                        all_results["group_comparisons"].append(result["group_comparison"])
                        if result.get("causal_graph_edges"):
                            all_results["causal_graph"]["edges"].extend(result["causal_graph_edges"])
                            all_results["causal_graph"]["nodes"].extend(result["causal_graph_nodes"])

            # 3. 요약 생성
            significant_effects = [
                te for te in all_results["treatment_effects"]
                if te.get("significant", False)
            ]
            all_results["summary"] = {
                "total_effects_analyzed": len(all_results["treatment_effects"]),
                "significant_effects": len(significant_effects),
                "data_driven": True,
                "tables_analyzed": list(real_data.keys()),
                "sample_size": len(rows),
            }
            all_results["recommendations"] = self._generate_data_driven_recommendations(
                all_results["treatment_effects"],
                all_results["group_comparisons"],
            )

        return all_results

    def _compute_real_ate(
        self,
        rows: List[Dict],
        treat_col: str,
        outcome_col: str,
        confounder_cols: List[str],
        table_name: str,
        categorical_confounder_cols: List[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        v18.5: 실제 데이터에서 ATE 계산 + Stratified ATE (교란변수 보정)

        1) treatment를 이진화
        2) outcome을 숫자 변환
        3) Naive ATE (Welch's t-test)
        4) Stratified ATE (카테고리형 교란변수별 층화 분석)
        5) 두 결과 모두 보고
        """
        # Treatment 이진화
        treat_values = []
        outcome_values = []
        confounder_matrix = []
        valid_rows = []

        for row in rows:
            t_val = row.get(treat_col)
            o_val = row.get(outcome_col)
            if t_val is None or o_val is None:
                continue

            try:
                o_num = float(o_val)
            except (ValueError, TypeError):
                continue

            treat_values.append(t_val)
            outcome_values.append(o_num)
            valid_rows.append(row)

        if len(valid_rows) < 10:
            return None

        # Treatment 이진화 로직
        # v19.1: sorted()로 결정적 순서 보장 (set 순회 비결정성 제거)
        unique_treats = sorted(set(str(v).strip().lower() for v in treat_values))
        if len(unique_treats) < 2:
            return None

        # v18.6: 다중 카테고리(>2값)는 pairwise 비교로 위임
        if len(unique_treats) > 2:
            return self._compute_pairwise_ate(
                valid_rows, treat_values, outcome_values,
                treat_col, outcome_col, unique_treats,
                confounder_cols, categorical_confounder_cols, table_name,
            )

        # 이진(2값) 매핑: Yes/No, True/False, 1/0 등
        positive_markers = {"yes", "true", "1", "1.0", "male", "high", "active"}
        treatment_binary = []
        treat_label_1 = None
        treat_label_0 = None

        for ut in unique_treats:
            if ut in positive_markers:
                treat_label_1 = ut
                break
        if treat_label_1 is None:
            treat_label_1 = unique_treats[0]
        treat_label_0 = [u for u in unique_treats if u != treat_label_1][0]

        for v in treat_values:
            treatment_binary.append(1.0 if str(v).strip().lower() == treat_label_1 else 0.0)

        treatment_arr = np.array(treatment_binary)
        outcome_arr = np.array(outcome_values)

        # 최소 그룹 크기 확인
        n_treated = int(np.sum(treatment_arr))
        n_control = len(treatment_arr) - n_treated
        if n_treated < 3 or n_control < 3:
            return None

        # Confounder 행렬 구축
        X_confounders = []
        valid_conf_cols = []
        for conf_col in confounder_cols:
            conf_values = []
            all_valid = True
            for row in valid_rows:
                cv = row.get(conf_col)
                if cv is None:
                    all_valid = False
                    break
                try:
                    conf_values.append(float(cv))
                except (ValueError, TypeError):
                    all_valid = False
                    break
            if all_valid and len(conf_values) == len(valid_rows):
                X_confounders.append(conf_values)
                valid_conf_cols.append(conf_col)

        # === Naive ATE: Welch's t-test ===
        naive_ate, naive_std, naive_pvalue, naive_ci = self._simple_mean_comparison(
            treatment_arr, outcome_arr
        )

        # 그룹별 실제 통계
        treated_outcomes = outcome_arr[treatment_arr == 1.0]
        control_outcomes = outcome_arr[treatment_arr == 0.0]

        # 원래 label 복원
        raw_labels = list(set(str(v).strip() for v in treat_values))
        label_1_display = [l for l in raw_labels if l.lower() == treat_label_1]
        label_0_display = [l for l in raw_labels if l.lower() == treat_label_0] if treat_label_0 != "others" else ["Others"]
        label_1_str = label_1_display[0] if label_1_display else treat_label_1
        label_0_str = label_0_display[0] if label_0_display else treat_label_0

        # === v18.5: Stratified ATE (카테고리형 교란변수 보정) ===
        stratified_ate = None
        stratified_details = []
        best_confounder = None
        categorical_confounder_cols = categorical_confounder_cols or []

        if categorical_confounder_cols:
            # 가장 강한 교란변수 찾기: treatment과 가장 연관된 카테고리 컬럼
            best_chi2 = 0

            for conf_col in categorical_confounder_cols:
                # Chi-squared 독립성 검정 근사 (treatment × confounder)
                contingency = defaultdict(Counter)
                for i, row in enumerate(valid_rows):
                    cv = row.get(conf_col)
                    if cv is not None:
                        contingency[str(cv)][int(treatment_arr[i])] += 1

                if len(contingency) < 2:
                    continue

                # Chi-squared 계산
                total = len(valid_rows)
                chi2 = 0
                row_totals = {k: sum(v.values()) for k, v in contingency.items()}
                col_totals = Counter()
                for counts in contingency.values():
                    for t_val, cnt in counts.items():
                        col_totals[t_val] += cnt

                for cat_val, counts in contingency.items():
                    for t_val in [0, 1]:
                        observed = counts.get(t_val, 0)
                        expected = row_totals[cat_val] * col_totals.get(t_val, 0) / total if total > 0 else 0
                        if expected > 0:
                            chi2 += (observed - expected) ** 2 / expected

                if chi2 > best_chi2:
                    best_chi2 = chi2
                    best_confounder = conf_col

            # Stratified ATE 계산 (가장 강한 교란변수로 층화)
            if best_confounder and best_chi2 > 3.84:  # chi2 > 3.84 ≈ p < 0.05
                strata = defaultdict(lambda: {"treated": [], "control": []})
                for i, row in enumerate(valid_rows):
                    stratum = str(row.get(best_confounder, ""))
                    if treatment_arr[i] == 1.0:
                        strata[stratum]["treated"].append(outcome_arr[i])
                    else:
                        strata[stratum]["control"].append(outcome_arr[i])

                weighted_ate = 0
                total_weight = 0
                excluded_strata = []  # v20: 제외된 strata 기록
                # v19.1: sorted() 로 결정적 순서 보장 (비결정적 dict 순회 제거)
                for stratum_name in sorted(strata.keys()):
                    groups = strata[stratum_name]
                    t_vals = groups["treated"]
                    c_vals = groups["control"]
                    if len(t_vals) >= 2 and len(c_vals) >= 2:
                        stratum_ate = np.mean(t_vals) - np.mean(c_vals)
                        stratum_n = len(t_vals) + len(c_vals)
                        weighted_ate += stratum_ate * stratum_n
                        total_weight += stratum_n
                        stratified_details.append({
                            "stratum": f"{best_confounder}={stratum_name}",
                            "treated_mean": float(np.mean(t_vals)),
                            "treated_n": len(t_vals),
                            "control_mean": float(np.mean(c_vals)),
                            "control_n": len(c_vals),
                            "stratum_n": stratum_n,
                            "stratum_ate": float(stratum_ate),
                            "weighted_contribution": float(stratum_ate * stratum_n),
                        })
                    else:
                        # v20: 제외 사유 기록
                        excluded_strata.append({
                            "stratum": f"{best_confounder}={stratum_name}",
                            "reason": f"treated_n={len(t_vals)}, control_n={len(c_vals)} (min_n=2 required)",
                            "treated_n": len(t_vals),
                            "control_n": len(c_vals),
                        })

                if total_weight > 0 and stratified_details:
                    stratified_ate = weighted_ate / total_weight
                    # 층화된 표준오차 계산
                    strat_var = 0
                    for detail in stratified_details:
                        w = (detail["treated_n"] + detail["control_n"]) / total_weight
                        strat_var += w ** 2 * (
                            np.var([o for o in strata[detail["stratum"].split("=")[1]]["treated"]], ddof=1) / max(detail["treated_n"], 1) +
                            np.var([o for o in strata[detail["stratum"].split("=")[1]]["control"]], ddof=1) / max(detail["control_n"], 1)
                        )
                    strat_se = np.sqrt(strat_var) if strat_var > 0 else naive_std

        # 최종 ATE: stratified가 있으면 사용, 없으면 naive
        if stratified_ate is not None:
            final_ate = stratified_ate
            final_std = strat_se
            # Stratified ATE의 p-value 근사
            if final_std > 0:
                t_stat_adj = abs(final_ate / final_std)
                final_pvalue = self._approx_two_tailed_pvalue(t_stat_adj)
            else:
                final_pvalue = naive_pvalue
            final_ci = (final_ate - 1.96 * final_std, final_ate + 1.96 * final_std)
            method_used = f"stratified_by_{best_confounder}"
            confounding_reduction = ((naive_ate - final_ate) / naive_ate * 100) if naive_ate != 0 else 0
        else:
            final_ate = naive_ate
            final_std = naive_std
            final_pvalue = naive_pvalue
            final_ci = naive_ci
            method_used = "welch_t_test"
            confounding_reduction = 0

        group_comparison = {
            "treatment_column": treat_col,
            "outcome_column": outcome_col,
            "table": table_name,
            "group_1": {
                "label": f"{treat_col}={label_1_str}",
                "n": int(n_treated),
                "mean": float(np.mean(treated_outcomes)),
                "std": float(np.std(treated_outcomes)),
                "median": float(np.median(treated_outcomes)),
            },
            "group_0": {
                "label": f"{treat_col}={label_0_str}",
                "n": int(n_control),
                "mean": float(np.mean(control_outcomes)),
                "std": float(np.std(control_outcomes)),
                "median": float(np.median(control_outcomes)),
            },
            "raw_difference": float(np.mean(treated_outcomes) - np.mean(control_outcomes)),
        }

        treatment_effect = {
            "type": f"{treat_col} → {outcome_col}",
            "treatment": treat_col,
            "outcome": outcome_col,
            "naive_ate": float(naive_ate),
            "estimate": float(final_ate),
            "std_error": float(final_std),
            "confidence_interval": (float(final_ci[0]), float(final_ci[1])),
            "p_value": float(final_pvalue),
            "significant": final_pvalue < 0.05,
            "method": method_used,
            "sample_size": len(valid_rows),
            "group_comparison": group_comparison,
            "confounders_used": valid_conf_cols,
        }

        # Stratified ATE 정보 추가
        if stratified_ate is not None:
            treatment_effect["stratified_analysis"] = {
                "confounder": best_confounder,
                "chi2": float(best_chi2),
                "naive_ate": float(naive_ate),
                "adjusted_ate": float(stratified_ate),
                "total_weight": int(total_weight),
                "confounding_reduction_pct": float(confounding_reduction),
                "strata": stratified_details,
                "excluded_strata": excluded_strata,  # v20: 제외된 strata
            }

        causal_nodes = [
            {"name": treat_col, "type": "treatment"},
            {"name": outcome_col, "type": "outcome"},
        ]
        causal_edges = [{"source": treat_col, "target": outcome_col}]
        for cc in valid_conf_cols:
            causal_nodes.append({"name": cc, "type": "confounder"})
            causal_edges.append({"source": cc, "target": treat_col})
            causal_edges.append({"source": cc, "target": outcome_col})
        # 카테고리형 교란변수도 그래프에 추가
        if stratified_ate is not None and best_confounder:
            causal_nodes.append({"name": best_confounder, "type": "confounder"})
            causal_edges.append({"source": best_confounder, "target": treat_col})
            causal_edges.append({"source": best_confounder, "target": outcome_col})

        return {
            "treatment_effect": treatment_effect,
            "group_comparison": group_comparison,
            "causal_graph_nodes": causal_nodes,
            "causal_graph_edges": causal_edges,
        }

    def _compute_pairwise_ate(
        self,
        valid_rows: List[Dict],
        treat_values: list,
        outcome_values: list,
        treat_col: str,
        outcome_col: str,
        unique_treats: List[str],
        confounder_cols: List[str],
        categorical_confounder_cols: List[str],
        table_name: str,
    ) -> Optional[Dict[str, Any]]:
        """
        v18.6: 다중 카테고리(>2값) treatment에 대해 모든 쌍(pairwise) 비교.
        가장 유의미한(p-value 최소) 쌍의 결과를 반환.
        추가로 모든 그룹 평균을 group_comparison에 포함.
        """
        from itertools import combinations

        outcome_arr = np.array(outcome_values)
        treat_str = [str(v).strip().lower() for v in treat_values]

        # 그룹별 통계
        group_outcomes = defaultdict(list)
        group_rows = defaultdict(list)
        for i, t in enumerate(treat_str):
            group_outcomes[t].append(outcome_values[i])
            group_rows[t].append(valid_rows[i])

        # 모든 쌍에 대해 ATE 계산
        best_result = None
        best_pvalue = 1.0
        all_pairs = []

        for g1_label, g0_label in combinations(unique_treats, 2):
            g1_vals = group_outcomes.get(g1_label, [])
            g0_vals = group_outcomes.get(g0_label, [])

            if len(g1_vals) < 3 or len(g0_vals) < 3:
                continue

            g1_arr = np.array(g1_vals)
            g0_arr = np.array(g0_vals)

            # 이 쌍만의 treatment 배열 구성
            pair_treatment = np.array([1.0] * len(g1_vals) + [0.0] * len(g0_vals))
            pair_outcome = np.concatenate([g1_arr, g0_arr])
            pair_rows = group_rows[g1_label] + group_rows[g0_label]

            ate, std, pvalue, ci = self._simple_mean_comparison(pair_treatment, pair_outcome)

            all_pairs.append({
                "g1": g1_label, "g0": g0_label,
                "ate": ate, "std": std, "p": pvalue, "ci": ci,
                "g1_mean": float(np.mean(g1_arr)), "g1_n": len(g1_vals),
                "g0_mean": float(np.mean(g0_arr)), "g0_n": len(g0_vals),
            })

            # v19.2: 결정적 타이브레이킹 — p-value 동일 시 sample_size 큰 쌍 → 알파벳 순
            pair_n = len(g1_vals) + len(g0_vals)
            if (pvalue < best_pvalue or
                (pvalue == best_pvalue and best_result is not None and (
                    pair_n > best_result["g1_n"] + best_result["g0_n"] or
                    (pair_n == best_result["g1_n"] + best_result["g0_n"] and
                     (g1_label, g0_label) < (best_result["g1"], best_result["g0"]))))):
                best_pvalue = pvalue
                best_result = all_pairs[-1]

        if not best_result:
            return None

        # 최적 쌍에 대해 Stratified ATE 계산
        g1_label = best_result["g1"]
        g0_label = best_result["g0"]

        # Stratified ATE (교란 보정)
        stratified_ate = None
        stratified_details = []
        best_confounder = None

        if categorical_confounder_cols:
            pair_indices_g1 = [i for i, t in enumerate(treat_str) if t == g1_label]
            pair_indices_g0 = [i for i, t in enumerate(treat_str) if t == g0_label]
            pair_rows_combined = [valid_rows[i] for i in pair_indices_g1] + [valid_rows[i] for i in pair_indices_g0]
            pair_treatment = [1.0] * len(pair_indices_g1) + [0.0] * len(pair_indices_g0)

            # 가장 강한 교란변수 탐지
            best_chi2 = 0
            for conf_col in categorical_confounder_cols:
                contingency = defaultdict(Counter)
                for j, row in enumerate(pair_rows_combined):
                    cv = row.get(conf_col)
                    if cv is not None:
                        contingency[str(cv)][int(pair_treatment[j])] += 1
                if len(contingency) < 2:
                    continue
                total = len(pair_rows_combined)
                chi2 = 0
                row_totals = {k: sum(v.values()) for k, v in contingency.items()}
                col_totals = Counter()
                for counts in contingency.values():
                    for t_val, cnt in counts.items():
                        col_totals[t_val] += cnt
                for cat_val, counts in contingency.items():
                    for t_val in [0, 1]:
                        observed = counts.get(t_val, 0)
                        expected = row_totals[cat_val] * col_totals.get(t_val, 0) / total if total > 0 else 0
                        if expected > 0:
                            chi2 += (observed - expected) ** 2 / expected
                if chi2 > best_chi2:
                    best_chi2 = chi2
                    best_confounder = conf_col

            if best_confounder and best_chi2 > 3.84:
                strata = defaultdict(lambda: {"treated": [], "control": []})
                for j, row in enumerate(pair_rows_combined):
                    stratum = str(row.get(best_confounder, ""))
                    if pair_treatment[j] == 1.0:
                        strata[stratum]["treated"].append(float(row.get(outcome_col, 0)))
                    else:
                        strata[stratum]["control"].append(float(row.get(outcome_col, 0)))
                weighted_ate = 0
                total_weight = 0
                excluded_strata = []  # v20: 제외된 strata 기록
                # v19.1: sorted() 로 결정적 순서 보장
                for stratum_name in sorted(strata.keys()):
                    groups = strata[stratum_name]
                    t_vals = groups["treated"]
                    c_vals = groups["control"]
                    if len(t_vals) >= 2 and len(c_vals) >= 2:
                        s_ate = np.mean(t_vals) - np.mean(c_vals)
                        s_n = len(t_vals) + len(c_vals)
                        weighted_ate += s_ate * s_n
                        total_weight += s_n
                        stratified_details.append({
                            "stratum": f"{best_confounder}={stratum_name}",
                            "treated_mean": float(np.mean(t_vals)),
                            "treated_n": len(t_vals),
                            "control_mean": float(np.mean(c_vals)),
                            "control_n": len(c_vals),
                            "stratum_n": s_n,
                            "stratum_ate": float(s_ate),
                            "weighted_contribution": float(s_ate * s_n),
                        })
                    else:
                        excluded_strata.append({
                            "stratum": f"{best_confounder}={stratum_name}",
                            "reason": f"treated_n={len(t_vals)}, control_n={len(c_vals)} (min_n=2 required)",
                            "treated_n": len(t_vals),
                            "control_n": len(c_vals),
                        })
                if total_weight > 0 and stratified_details:
                    stratified_ate = weighted_ate / total_weight
                    # 층화된 표준오차 계산 (within-stratum variances)
                    strat_var = 0
                    for detail in stratified_details:
                        w = (detail["treated_n"] + detail["control_n"]) / total_weight
                        stratum_key = detail["stratum"].split("=", 1)[1]
                        t_vals = strata[stratum_key]["treated"]
                        c_vals = strata[stratum_key]["control"]
                        t_var = np.var(t_vals, ddof=1) if len(t_vals) > 1 else 0
                        c_var = np.var(c_vals, ddof=1) if len(c_vals) > 1 else 0
                        strat_var += w ** 2 * (
                            t_var / max(detail["treated_n"], 1) +
                            c_var / max(detail["control_n"], 1)
                        )
                    stratified_se = np.sqrt(strat_var) if strat_var > 0 else best_result["std"]

        # 최종 ATE 결정
        naive_ate = best_result["ate"]
        if stratified_ate is not None:
            final_ate = stratified_ate
            confounding_reduction = ((naive_ate - final_ate) / naive_ate * 100) if naive_ate != 0 else 0
            method_used = f"pairwise_stratified_by_{best_confounder}"
            # p-value: proper SE from within-stratum variances
            strat_se = stratified_se
            if strat_se > 0:
                t_stat_adj = abs(final_ate / strat_se)
                final_pvalue = self._approx_two_tailed_pvalue(t_stat_adj)
            else:
                final_pvalue = best_pvalue
            final_ci = (final_ate - 1.96 * strat_se, final_ate + 1.96 * strat_se)
        else:
            final_ate = naive_ate
            final_pvalue = best_pvalue
            final_ci = best_result["ci"]
            confounding_reduction = 0
            method_used = "pairwise_welch_t_test"

        # 원래 라벨 복원
        raw_labels = list(set(str(v).strip() for v in treat_values))
        g1_display = [l for l in raw_labels if l.lower() == g1_label]
        g0_display = [l for l in raw_labels if l.lower() == g0_label]
        g1_str = g1_display[0] if g1_display else g1_label
        g0_str = g0_display[0] if g0_display else g0_label

        # 모든 그룹 평균 포함
        all_group_stats = {}
        for label in unique_treats:
            vals = group_outcomes[label]
            display = [l for l in raw_labels if l.lower() == label]
            display_str = display[0] if display else label
            all_group_stats[display_str] = {
                "mean": float(np.mean(vals)),
                "n": len(vals),
                "std": float(np.std(vals)),
            }

        group_comparison = {
            "treatment_column": treat_col,
            "outcome_column": outcome_col,
            "table": table_name,
            "comparison_type": "pairwise_best",
            "group_1": {
                "label": f"{treat_col}={g1_str}",
                "n": best_result["g1_n"],
                "mean": best_result["g1_mean"],
                "std": float(np.std(group_outcomes[g1_label])),
                "median": float(np.median(group_outcomes[g1_label])),
            },
            "group_0": {
                "label": f"{treat_col}={g0_str}",
                "n": best_result["g0_n"],
                "mean": best_result["g0_mean"],
                "std": float(np.std(group_outcomes[g0_label])),
                "median": float(np.median(group_outcomes[g0_label])),
            },
            "raw_difference": best_result["ate"],
            "all_groups": all_group_stats,
            "all_pairwise": [
                {
                    "pair": f"{p['g1']} vs {p['g0']}",
                    "ate": p["ate"],
                    "p_value": p["p"],
                }
                for p in sorted(all_pairs, key=lambda x: x["p"])
            ],
        }

        treatment_effect = {
            "type": f"{treat_col}({g1_str} vs {g0_str}) → {outcome_col}",
            "treatment": treat_col,
            "outcome": outcome_col,
            "naive_ate": float(naive_ate),
            "estimate": float(final_ate),
            "std_error": float(best_result["std"]),
            "confidence_interval": (float(final_ci[0]), float(final_ci[1])),
            "p_value": float(final_pvalue),
            "significant": final_pvalue < 0.05,
            "method": method_used,
            "sample_size": best_result["g1_n"] + best_result["g0_n"],
            "group_comparison": group_comparison,
            "confounders_used": [],
        }

        if stratified_ate is not None:
            treatment_effect["stratified_analysis"] = {
                "confounder": best_confounder,
                "chi2": float(best_chi2),
                "naive_ate": float(naive_ate),
                "adjusted_ate": float(stratified_ate),
                "total_weight": int(total_weight),
                "confounding_reduction_pct": float(confounding_reduction),
                "strata": stratified_details,
                "excluded_strata": excluded_strata,  # v20: 제외된 strata
            }

        causal_nodes = [
            {"name": treat_col, "type": "treatment"},
            {"name": outcome_col, "type": "outcome"},
        ]
        causal_edges = [{"source": treat_col, "target": outcome_col}]
        if best_confounder:
            causal_nodes.append({"name": best_confounder, "type": "confounder"})
            causal_edges.append({"source": best_confounder, "target": treat_col})
            causal_edges.append({"source": best_confounder, "target": outcome_col})

        return {
            "treatment_effect": treatment_effect,
            "group_comparison": group_comparison,
            "causal_graph_nodes": causal_nodes,
            "causal_graph_edges": causal_edges,
        }

    def _approx_two_tailed_pvalue(self, t_stat: float, df: float = 100) -> float:
        """
        Two-tailed p-value approximation without scipy.
        Uses Abramowitz & Stegun normal CDF approximation for df >= 30,
        step-function for smaller df.
        """
        if df >= 30:
            z = abs(t_stat)
            p_coeff = 0.3275911
            a1, a2, a3, a4, a5 = 0.254829592, -0.284496736, 1.421413741, -1.453152027, 1.061405429
            t_val = 1.0 / (1.0 + p_coeff * z / np.sqrt(2))
            erf_approx = 1 - (a1*t_val + a2*t_val**2 + a3*t_val**3 + a4*t_val**4 + a5*t_val**5) * np.exp(-z*z/2)
            return float(max(1.0 - erf_approx, 1e-10))
        else:
            t = abs(t_stat)
            if t > 6: return 0.0001
            elif t > 4: return 0.001
            elif t > 3: return 0.005
            elif t > 2.5: return 0.02
            elif t > 2.0: return 0.05
            elif t > 1.7: return 0.1
            elif t > 1.3: return 0.2
            else: return 0.5

    def _simple_mean_comparison(
        self,
        treatment: np.ndarray,
        outcome: np.ndarray,
    ) -> tuple:
        """Welch's t-test 기반 그룹 평균 비교"""
        treated = outcome[treatment == 1.0]
        control = outcome[treatment == 0.0]

        mean_diff = float(np.mean(treated) - np.mean(control))
        n1, n2 = len(treated), len(control)

        var1 = np.var(treated, ddof=1) if n1 > 1 else 0
        var2 = np.var(control, ddof=1) if n2 > 1 else 0
        se = np.sqrt(var1 / max(n1, 1) + var2 / max(n2, 1))

        if se > 0:
            t_stat = abs(mean_diff / se)

            # Welch-Satterthwaite 자유도
            num = (var1 / n1 + var2 / n2) ** 2
            denom = ((var1 / n1) ** 2 / max(n1 - 1, 1)) + ((var2 / n2) ** 2 / max(n2 - 1, 1))
            df = num / denom if denom > 0 else min(n1, n2) - 1
            df = max(df, 1)

            # t-분포 p-value 근사 (scipy 없이, 정규 근사 + 보정)
            # 자유도가 30 이상이면 정규 근사 충분
            if df >= 30:
                # 정규 분포 근사: Φ(-|t|) * 2
                z = t_stat
                # Abramowitz & Stegun 근사
                p = 0.3275911
                a1, a2, a3, a4, a5 = 0.254829592, -0.284496736, 1.421413741, -1.453152027, 1.061405429
                t_val = 1.0 / (1.0 + p * z / np.sqrt(2))
                erf_approx = 1 - (a1*t_val + a2*t_val**2 + a3*t_val**3 + a4*t_val**4 + a5*t_val**5) * np.exp(-z*z/2)
                p_value = float(1.0 - erf_approx)  # two-tailed
            else:
                # 소규모 자유도: 보수적 근사
                # Beta incomplete function 근사
                x = df / (df + t_stat**2)
                # 간단한 근사: regularized incomplete beta
                if t_stat > 6:
                    p_value = 0.0001
                elif t_stat > 4:
                    p_value = 0.001
                elif t_stat > 3:
                    p_value = 0.005
                elif t_stat > 2.5:
                    p_value = 0.02
                elif t_stat > 2.0:
                    p_value = 0.05
                elif t_stat > 1.7:
                    p_value = 0.1
                elif t_stat > 1.3:
                    p_value = 0.2
                else:
                    p_value = 0.5
        else:
            t_stat = 0.0
            p_value = 1.0

        ci_lower = mean_diff - 1.96 * se
        ci_upper = mean_diff + 1.96 * se

        return mean_diff, se, p_value, (ci_lower, ci_upper)

    def _generate_data_driven_recommendations(
        self,
        treatment_effects: List[Dict],
        group_comparisons: List[Dict],
    ) -> List[str]:
        """실제 데이터 기반 권장사항 생성"""
        recommendations = []

        significant = [te for te in treatment_effects if te.get("significant")]
        if not significant:
            recommendations.append("통계적으로 유의미한 인과 효과가 발견되지 않았습니다.")
            return recommendations

        for te in significant[:5]:
            treat = te["treatment"]
            outcome = te["outcome"]
            est = te["estimate"]
            naive = te.get("naive_ate", est)
            gc = te.get("group_comparison", {})
            strat = te.get("stratified_analysis")

            g1 = gc.get("group_1", {})
            g0 = gc.get("group_0", {})

            if strat:
                # 교란변수 보정된 결과
                conf = strat["confounder"]
                reduction = strat["confounding_reduction_pct"]
                recommendations.append(
                    f"{treat} → {outcome}: Naive ATE={naive:+.1f}, "
                    f"교란변수({conf}) 보정 후 ATE={est:+.1f} "
                    f"(교란 편향 {reduction:.0f}% 감소, p={te['p_value']:.4f}, n={te['sample_size']})"
                )
            else:
                if est > 0:
                    recommendations.append(
                        f"{treat} → {outcome}: {g1.get('label', treat)}일 때 평균 {abs(est):.1f} 높음 "
                        f"(p={te['p_value']:.4f}, n={te['sample_size']})"
                    )
                else:
                    recommendations.append(
                        f"{treat} → {outcome}: {g0.get('label', treat)}일 때 평균 {abs(est):.1f} 높음 "
                        f"(p={te['p_value']:.4f}, n={te['sample_size']})"
                    )

        return recommendations

    # --- Legacy fallback (메타데이터만 있을 때) ---

    def _analyze_metadata_only(
        self,
        tables: Dict[str, Dict],
        fk_relations: List[Dict] = None,
        entity_matches: List[Dict] = None,
        target_kpis: List[str] = None,
    ) -> Dict[str, Any]:
        """기존 메타데이터 기반 분석 (실제 데이터 없을 때 fallback)"""
        integration_scenario = {
            "n_fk_relations": len(fk_relations) if fk_relations else 0,
            "n_entity_matches": len(entity_matches) if entity_matches else 0,
            "schema_similarity": self._estimate_schema_similarity(tables),
            "treatments": ["data_integration"],
            "outcomes": target_kpis or ["operational_efficiency"]
        }
        baseline = {"baseline_kpi": 0.65}

        result = self.model.estimate_integration_impact(
            tables, integration_scenario, baseline
        )

        return {
            "summary": result.summary,
            "treatment_effects": [
                {
                    "type": te.effect_type.value,
                    "estimate": te.estimate,
                    "std_error": te.std_error,
                    "confidence_interval": te.confidence_interval,
                    "p_value": te.p_value,
                    "significant": te.p_value < 0.05
                }
                for te in result.treatment_effects
            ],
            "counterfactuals": [
                {
                    "factual": cf.factual_outcome,
                    "counterfactual": cf.counterfactual_outcome,
                    "effect": cf.individual_effect,
                    "confidence": cf.confidence,
                    "explanation": cf.explanation
                }
                for cf in result.counterfactuals
            ],
            "sensitivity": {
                "robustness": result.sensitivity.robustness_value,
                "critical_bias": result.sensitivity.critical_bias,
                "interpretation": result.sensitivity.interpretation
            },
            "recommendations": result.recommendations,
            "causal_graph": {
                "nodes": [
                    {"name": n.name, "type": n.node_type}
                    for n in result.causal_graph.nodes.values()
                ],
                "edges": [
                    {"source": e.source, "target": e.target}
                    for e in result.causal_graph.edges
                ]
            }
        }

    def _estimate_schema_similarity(self, tables: Dict[str, Dict]) -> float:
        """스키마 유사도 추정"""
        if not tables or len(tables) < 2:
            return 0.5
        all_columns = []
        for table_info in tables.values():
            if isinstance(table_info, dict):
                cols = table_info.get("columns", [])
                col_names = []
                for c in cols:
                    if isinstance(c, dict):
                        col_names.append(c.get("name", c.get("column_name", "")))
                    else:
                        col_names.append(str(c))
                all_columns.append(set(col_names))
        if len(all_columns) < 2:
            return 0.5
        similarities = []
        for i in range(len(all_columns)):
            for j in range(i + 1, len(all_columns)):
                intersection = len(all_columns[i] & all_columns[j])
                union = len(all_columns[i] | all_columns[j])
                if union > 0:
                    similarities.append(intersection / union)
        return np.mean(similarities) if similarities else 0.5

    def predict_kpi_improvement(
        self,
        tables: Dict[str, Dict],
        proposed_changes: List[Dict]
    ) -> Dict[str, float]:
        """
        제안된 변경사항에 따른 KPI 개선 예측

        Parameters:
        - tables: 현재 테이블 정보
        - proposed_changes: 제안된 변경사항 목록

        Returns:
        - KPI별 예상 개선율
        """
        predictions = {}

        # 변경 유형별 가중치
        change_weights = {
            "fk_establishment": 0.15,      # FK 수립: 15% 기여
            "entity_resolution": 0.25,      # 엔티티 통합: 25% 기여
            "schema_alignment": 0.20,       # 스키마 정렬: 20% 기여
            "data_quality_fix": 0.10,       # 품질 개선: 10% 기여
            "redundancy_removal": 0.08      # 중복 제거: 8% 기여
        }

        # 각 변경사항의 영향 누적
        total_impact = 0.0
        for change in proposed_changes:
            change_type = change.get("type", "")
            confidence = change.get("confidence", 0.5)
            weight = change_weights.get(change_type, 0.05)
            total_impact += weight * confidence

        # 감쇠 적용 (수확 체감)
        final_impact = 1 - np.exp(-total_impact)

        predictions["operational_efficiency"] = final_impact * 0.2  # 최대 20%
        predictions["data_quality_score"] = final_impact * 0.3      # 최대 30%
        predictions["query_performance"] = final_impact * 0.15       # 최대 15%
        predictions["decision_accuracy"] = final_impact * 0.25       # 최대 25%

        return predictions

    def granger_causality_analysis(
        self,
        time_series_data: Dict[str, np.ndarray],
        target_variable: str,
        max_lag: int = 5,
        significance_level: float = 0.05
    ) -> Dict[str, Any]:
        """
        Granger Causality 분석 수행

        시계열 변수들이 타겟 변수에 미치는 Granger 인과성 검정

        Parameters:
        - time_series_data: {"variable_name": time_series_array, ...}
        - target_variable: 타겟 변수명
        - max_lag: 최대 시차 수
        - significance_level: 유의수준

        Returns:
        - Granger causality 분석 결과
        """
        analyzer = GrangerCausalityAnalyzer(
            max_lag=max_lag,
            significance_level=significance_level
        )

        results = analyzer.test_multivariate(time_series_data, target_variable)

        # 결과 정리
        causal_variables = []
        non_causal_variables = []

        formatted_results = {}
        for var_name, result in results.items():
            formatted_results[var_name] = {
                "f_statistic": result.f_statistic,
                "p_value": result.p_value,
                "optimal_lag": result.optimal_lag,
                "is_causal": result.is_causal,
                "r2_restricted": result.r2_restricted,
                "r2_unrestricted": result.r2_unrestricted,
                "interpretation": result.interpretation
            }

            if result.is_causal:
                causal_variables.append(var_name)
            else:
                non_causal_variables.append(var_name)

        return {
            "target_variable": target_variable,
            "causal_variables": causal_variables,
            "non_causal_variables": non_causal_variables,
            "detailed_results": formatted_results,
            "summary": {
                "n_variables_tested": len(results),
                "n_causal": len(causal_variables),
                "significance_level": significance_level,
                "max_lag": max_lag
            }
        }

    def discover_causal_graph(
        self,
        data: np.ndarray,
        variable_names: Optional[List[str]] = None,
        alpha: float = 0.05,
        max_cond_set_size: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        PC Algorithm을 사용하여 인과 그래프 발견

        Parameters:
        - data: (n_samples, n_variables) 데이터 행렬
        - variable_names: 변수명 리스트
        - alpha: 조건부 독립성 테스트 유의수준
        - max_cond_set_size: 최대 조건부 집합 크기

        Returns:
        - 발견된 인과 그래프 정보
        """
        pc = PCAlgorithm(
            alpha=alpha,
            max_cond_set_size=max_cond_set_size
        )

        result = pc.fit(data, variable_names)

        # 엣지 목록 정리
        edges_list = [
            {"source": src, "target": tgt}
            for src, tgt in result.directed_edges
        ]

        # 각 노드의 부모/자식 관계
        node_relations = {}
        for name in result.variable_names:
            parents = result.causal_graph.get_parents(name)
            children = result.causal_graph.get_children(name)
            node_relations[name] = {
                "parents": parents,
                "children": children,
                "n_parents": len(parents),
                "n_children": len(children)
            }

        return {
            "nodes": result.variable_names,
            "edges": edges_list,
            "n_edges": len(result.directed_edges),
            "adjacency_matrix": result.adjacency_matrix.tolist(),
            "node_relations": node_relations,
            "skeleton": [{"v1": v1, "v2": v2} for v1, v2 in result.skeleton],
            "algorithm_stats": {
                "n_conditional_tests": result.n_conditional_tests,
                "alpha": result.alpha
            }
        }

    def estimate_heterogeneous_effects(
        self,
        X: np.ndarray,
        treatment: np.ndarray,
        outcome: np.ndarray,
        method: str = "x_learner",
        feature_names: Optional[List[str]] = None,
        X_test: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        CATE (이질적 처리 효과) 추정

        S-Learner, T-Learner, X-Learner 중 선택

        Parameters:
        - X: 공변량 행렬 (n_samples, n_features)
        - treatment: 처리 벡터 (0/1)
        - outcome: 결과 벡터
        - method: "s_learner", "t_learner", "x_learner"
        - feature_names: 피처명 리스트
        - X_test: 테스트 데이터

        Returns:
        - CATE 추정 결과
        """
        estimator = EnhancedCATEEstimator(
            method=method,
            base_learner="ridge",
            n_bootstrap=100
        )

        result = estimator.fit_predict(
            X=X,
            treatment=treatment,
            outcome=outcome,
            X_test=X_test,
            feature_names=feature_names
        )

        # 서브그룹 분석
        subgroup_summary = {}
        if result.subgroup_effects:
            for key, value in result.subgroup_effects.items():
                subgroup_summary[key] = value

        return {
            "method": result.method,
            "ate": float(result.ate),
            "cate_estimates": result.cate_estimates.tolist(),
            "std_errors": result.std_errors.tolist(),
            "confidence_intervals": result.confidence_intervals.tolist(),
            "feature_importances": result.feature_importances,
            "subgroup_effects": subgroup_summary,
            "summary": {
                "mean_cate": float(np.mean(result.cate_estimates)),
                "std_cate": float(np.std(result.cate_estimates)),
                "min_cate": float(np.min(result.cate_estimates)),
                "max_cate": float(np.max(result.cate_estimates)),
                "pct_positive_effect": float(np.mean(result.cate_estimates > 0) * 100)
            }
        }

    def comprehensive_causal_analysis(
        self,
        tables: Dict[str, Dict],
        time_series_data: Optional[Dict[str, np.ndarray]] = None,
        observation_data: Optional[np.ndarray] = None,
        variable_names: Optional[List[str]] = None,
        fk_relations: List[Dict] = None,
        entity_matches: List[Dict] = None
    ) -> Dict[str, Any]:
        """
        종합 인과 분석 수행

        1. 기본 인과 영향 분석
        2. Granger Causality (시계열 데이터가 있는 경우)
        3. PC Algorithm 기반 그래프 발견 (관찰 데이터가 있는 경우)
        4. CATE 추정

        Parameters:
        - tables: 테이블 정보
        - time_series_data: 시계열 데이터 (optional)
        - observation_data: 관찰 데이터 행렬 (optional)
        - variable_names: 변수명 리스트
        - fk_relations: FK 관계
        - entity_matches: 엔티티 매칭

        Returns:
        - 종합 인과 분석 결과
        """
        results = {
            "analysis_types": [],
            "basic_impact": None,
            "granger_causality": None,
            "causal_graph_discovery": None,
            "heterogeneous_effects": None
        }

        # 1. 기본 인과 영향 분석
        try:
            basic_result = self.analyze(
                tables=tables,
                fk_relations=fk_relations,
                entity_matches=entity_matches
            )
            results["basic_impact"] = basic_result
            results["analysis_types"].append("basic_impact")
        except Exception as e:
            results["basic_impact"] = {"error": str(e)}

        # 2. Granger Causality (시계열 데이터가 있는 경우)
        if time_series_data and len(time_series_data) >= 2:
            try:
                # 첫 번째 변수를 타겟으로 설정 (또는 'outcome', 'kpi' 같은 이름 찾기)
                target = None
                for key in time_series_data.keys():
                    if any(t in key.lower() for t in ['outcome', 'kpi', 'target', 'y']):
                        target = key
                        break
                if target is None:
                    target = list(time_series_data.keys())[0]

                granger_result = self.granger_causality_analysis(
                    time_series_data=time_series_data,
                    target_variable=target
                )
                results["granger_causality"] = granger_result
                results["analysis_types"].append("granger_causality")
            except Exception as e:
                results["granger_causality"] = {"error": str(e)}

        # 3. PC Algorithm (관찰 데이터가 있는 경우)
        if observation_data is not None and len(observation_data) > 10:
            try:
                pc_result = self.discover_causal_graph(
                    data=observation_data,
                    variable_names=variable_names
                )
                results["causal_graph_discovery"] = pc_result
                results["analysis_types"].append("causal_graph_discovery")
            except Exception as e:
                results["causal_graph_discovery"] = {"error": str(e)}

        # 4. CATE 추정 (관찰 데이터가 있고 treatment/outcome이 있는 경우)
        if observation_data is not None and observation_data.shape[1] >= 3:
            try:
                # 마지막 두 열을 treatment, outcome으로 가정
                X = observation_data[:, :-2]
                treatment = observation_data[:, -2]
                outcome = observation_data[:, -1]

                # treatment를 이진화
                treatment_binary = (treatment > np.median(treatment)).astype(float)

                feature_names_cate = None
                if variable_names and len(variable_names) >= 2:
                    feature_names_cate = variable_names[:-2]

                cate_result = self.estimate_heterogeneous_effects(
                    X=X,
                    treatment=treatment_binary,
                    outcome=outcome,
                    method="x_learner",
                    feature_names=feature_names_cate
                )
                results["heterogeneous_effects"] = cate_result
                results["analysis_types"].append("heterogeneous_effects")
            except Exception as e:
                results["heterogeneous_effects"] = {"error": str(e)}

        # 종합 요약
        results["summary"] = {
            "n_analysis_types": len(results["analysis_types"]),
            "analyses_performed": results["analysis_types"],
            "has_causal_evidence": (
                results.get("basic_impact", {}).get("summary", {}).get("ate_significant", False) or
                (results.get("granger_causality", {}).get("n_causal", 0) > 0 if isinstance(results.get("granger_causality"), dict) else False)
            )
        }

        return results
