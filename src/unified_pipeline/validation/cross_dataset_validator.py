# validation/cross_dataset_validator.py
"""
Solution 4: Cross-Dataset Validation Framework

다중 데이터셋 검증으로 overfitting 탐지 및 일반화 성능 측정
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any, Callable
from pathlib import Path
import json
import statistics
import logging
import re

logger = logging.getLogger(__name__)


@dataclass
class GroundTruthFK:
    """Ground Truth FK 정의"""
    from_table: str
    from_column: str
    to_table: str
    to_column: str
    description: str = ""

    def to_key(self) -> str:
        """비교용 키 생성"""
        return f"{self.from_table}.{self.from_column}->{self.to_table}.{self.to_column}"


@dataclass
class DatasetConfig:
    """검증 데이터셋 설정"""
    name: str
    path: str
    domain: str
    difficulty: str  # easy, medium, hard
    ground_truth: List[GroundTruthFK] = field(default_factory=list)
    description: str = ""


@dataclass
class ValidationResult:
    """단일 데이터셋 검증 결과"""
    dataset: str
    precision: float
    recall: float
    f1: float
    true_positives: int
    false_positives: int
    false_negatives: int
    tp_details: List[str] = field(default_factory=list)
    fp_details: List[str] = field(default_factory=list)
    fn_details: List[str] = field(default_factory=list)
    execution_time_ms: float = 0.0


class CrossDatasetValidator:
    """
    다중 데이터셋 검증 프레임워크

    목표:
    - 여러 도메인에서 일관된 성능 확인
    - Overfitting 탐지 (특정 데이터셋만 잘되면 문제)
    - Generalization 성능 측정
    """

    def __init__(self):
        self.datasets: List[DatasetConfig] = []
        self._load_default_datasets()

    def _load_default_datasets(self):
        """기본 검증 데이터셋 로드"""
        # Marketing Silo Dataset
        self.datasets.append(DatasetConfig(
            name="marketing_silo",
            path="data/marketing_silo",
            domain="marketing",
            difficulty="hard",
            description="E-commerce marketing data with 11 tables",
            ground_truth=[
                GroundTruthFK("orders", "buyer_id", "customers", "customer_id",
                             "Order placed by customer"),
                GroundTruthFK("orders", "prod_code", "products", "product_id",
                             "Order contains product"),
                GroundTruthFK("web_sessions", "user_id", "customers", "customer_id",
                             "Web session belongs to customer"),
                GroundTruthFK("email_sends", "cust_no", "customers", "customer_id",
                             "Email sent to customer"),
                GroundTruthFK("email_sends", "cmp_id", "campaigns", "campaign_id",
                             "Email part of campaign"),
                GroundTruthFK("email_events", "email_send_id", "email_sends", "send_id",
                             "Event for email send"),
                GroundTruthFK("leads", "cust_id", "customers", "customer_id",
                             "Lead converted to customer"),
                GroundTruthFK("conversions", "order_ref", "orders", "order_id",
                             "Conversion resulted in order"),
                GroundTruthFK("conversions", "attributed_cmp_id", "campaigns", "campaign_id",
                             "Conversion attributed to campaign"),
                GroundTruthFK("conversions", "attributed_ad_id", "ad_campaigns", "ad_campaign_id",
                             "Conversion attributed to ad"),
                GroundTruthFK("ad_campaigns", "marketing_campaign_ref", "campaigns", "campaign_id",
                             "Ad campaign part of marketing campaign"),
                GroundTruthFK("ad_performance", "ad_cmp_id", "ad_campaigns", "ad_campaign_id",
                             "Performance for ad campaign"),
            ]
        ))

        # Airport Silo Dataset (if exists)
        airport_path = Path("data/airport_silo")
        if airport_path.exists():
            self.datasets.append(DatasetConfig(
                name="airport_silo",
                path="data/airport_silo",
                domain="aviation",
                difficulty="medium",
                description="Airport operations data",
                ground_truth=[
                    # Add ground truth when available
                ]
            ))

    def add_dataset(self, config: DatasetConfig):
        """데이터셋 추가"""
        self.datasets.append(config)

    def validate_single(
        self,
        detector_fn: Callable[[str], List[Dict]],
        dataset: DatasetConfig,
    ) -> ValidationResult:
        """
        단일 데이터셋 검증

        Args:
            detector_fn: FK 탐지 함수 (path -> List[{from_table, from_column, to_table, to_column}])
            dataset: 검증 데이터셋 설정
        """
        import time

        if not dataset.ground_truth:
            logger.warning(f"No ground truth for dataset: {dataset.name}")
            return ValidationResult(
                dataset=dataset.name,
                precision=0.0,
                recall=0.0,
                f1=0.0,
                true_positives=0,
                false_positives=0,
                false_negatives=0,
            )

        # FK 탐지 실행
        start = time.time()
        try:
            detected = detector_fn(dataset.path)
        except Exception as e:
            logger.error(f"Detection failed for {dataset.name}: {e}")
            return ValidationResult(
                dataset=dataset.name,
                precision=0.0,
                recall=0.0,
                f1=0.0,
                true_positives=0,
                false_positives=0,
                false_negatives=len(dataset.ground_truth),
            )
        elapsed_ms = (time.time() - start) * 1000

        # 메트릭 계산
        tp, fp, fn, tp_list, fp_list, fn_list = self._compute_confusion(
            detected, dataset.ground_truth
        )

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        return ValidationResult(
            dataset=dataset.name,
            precision=precision,
            recall=recall,
            f1=f1,
            true_positives=tp,
            false_positives=fp,
            false_negatives=fn,
            tp_details=tp_list,
            fp_details=fp_list,
            fn_details=fn_list,
            execution_time_ms=elapsed_ms,
        )

    def validate_all(
        self,
        detector_fn: Callable[[str], List[Dict]],
    ) -> Dict:
        """모든 데이터셋에서 검증"""
        results = []

        for dataset in self.datasets:
            if not dataset.ground_truth:
                logger.info(f"Skipping {dataset.name} (no ground truth)")
                continue

            logger.info(f"Validating on {dataset.name}...")
            result = self.validate_single(detector_fn, dataset)
            results.append(result)

            logger.info(f"  Precision: {result.precision:.2%}")
            logger.info(f"  Recall: {result.recall:.2%}")
            logger.info(f"  F1: {result.f1:.2%}")

        return self._aggregate_results(results)

    def _compute_confusion(
        self,
        detected: List[Dict],
        ground_truth: List[GroundTruthFK],
    ) -> Tuple[int, int, int, List[str], List[str], List[str]]:
        """Confusion matrix 계산"""

        # Ground Truth를 set으로 변환
        gt_set = set()
        for gt in ground_truth:
            gt_set.add(gt.to_key())

        # Detected를 set으로 변환
        detected_set = set()
        for d in detected:
            key = f"{d.get('from_table', '')}.{d.get('from_column', '')}->{d.get('to_table', '')}.{d.get('to_column', '')}"
            detected_set.add(key)

        # TP: 둘 다 있음
        tp_set = detected_set & gt_set
        # FP: detected에만 있음
        fp_set = detected_set - gt_set
        # FN: gt에만 있음
        fn_set = gt_set - detected_set

        return (
            len(tp_set),
            len(fp_set),
            len(fn_set),
            list(tp_set),
            list(fp_set),
            list(fn_set),
        )

    def _aggregate_results(self, results: List[ValidationResult]) -> Dict:
        """결과 종합"""
        if not results:
            return {"error": "No validation results"}

        precisions = [r.precision for r in results]
        recalls = [r.recall for r in results]
        f1s = [r.f1 for r in results]

        return {
            "per_dataset": {r.dataset: {
                "precision": r.precision,
                "recall": r.recall,
                "f1": r.f1,
                "true_positives": r.true_positives,
                "false_positives": r.false_positives,
                "false_negatives": r.false_negatives,
                "execution_time_ms": r.execution_time_ms,
            } for r in results},

            "aggregate": {
                "avg_precision": statistics.mean(precisions),
                "avg_recall": statistics.mean(recalls),
                "avg_f1": statistics.mean(f1s),
                "std_precision": statistics.stdev(precisions) if len(precisions) > 1 else 0,
                "std_recall": statistics.stdev(recalls) if len(recalls) > 1 else 0,
                "std_f1": statistics.stdev(f1s) if len(f1s) > 1 else 0,
                "min_f1": min(f1s),
                "max_f1": max(f1s),
            },

            "generalization_score": self._compute_generalization_score(results),

            "overfitting_analysis": self._detect_overfitting(results),

            "detailed_results": [
                {
                    "dataset": r.dataset,
                    "precision": r.precision,
                    "recall": r.recall,
                    "f1": r.f1,
                    "tp": r.tp_details,
                    "fp": r.fp_details,
                    "fn": r.fn_details,
                }
                for r in results
            ]
        }

    def _compute_generalization_score(self, results: List[ValidationResult]) -> float:
        """
        일반화 점수 계산

        높은 평균 + 낮은 분산 = 좋은 일반화
        """
        if not results:
            return 0.0

        f1s = [r.f1 for r in results]
        avg = statistics.mean(f1s)
        std = statistics.stdev(f1s) if len(f1s) > 1 else 0

        # 일반화 점수 = 평균 - 표준편차
        # (일관되게 높은 성능이면 높은 점수)
        return max(0, avg - std)

    def _detect_overfitting(self, results: List[ValidationResult]) -> Dict:
        """
        Overfitting 탐지

        특정 데이터셋만 F1이 높고 나머지는 낮으면 overfitting
        """
        if len(results) < 2:
            return {
                "is_overfitting": False,
                "message": "Need at least 2 datasets to detect overfitting",
            }

        f1s = [r.f1 for r in results]
        avg = statistics.mean(f1s)

        outliers = []
        for r in results:
            deviation = r.f1 - avg
            if abs(deviation) > 0.2:  # 평균에서 20% 이상 벗어나면
                outliers.append({
                    "dataset": r.dataset,
                    "f1": r.f1,
                    "deviation": deviation,
                    "direction": "above" if deviation > 0 else "below"
                })

        is_overfitting = False
        recommendation = "Good generalization across datasets"

        if outliers:
            above = [o for o in outliers if o["direction"] == "above"]
            below = [o for o in outliers if o["direction"] == "below"]

            if above and below:
                is_overfitting = True
                recommendation = f"Possible overfitting to: {[o['dataset'] for o in above]}. Poor on: {[o['dataset'] for o in below]}"
            elif above:
                recommendation = f"Good on {[o['dataset'] for o in above]}, average on others"
            elif below:
                is_overfitting = True
                recommendation = f"Poor performance on: {[o['dataset'] for o in below]}"

        return {
            "is_overfitting": is_overfitting,
            "outliers": outliers,
            "recommendation": recommendation,
            "average_f1": avg,
        }

    def generate_report(self, results: Dict, output_path: Optional[str] = None) -> str:
        """검증 리포트 생성"""
        lines = []
        lines.append("=" * 60)
        lines.append("FK DETECTION CROSS-DATASET VALIDATION REPORT")
        lines.append("=" * 60)
        lines.append("")

        # Aggregate Results
        agg = results.get("aggregate", {})
        lines.append("## Aggregate Results")
        lines.append(f"  Average Precision: {agg.get('avg_precision', 0):.2%}")
        lines.append(f"  Average Recall:    {agg.get('avg_recall', 0):.2%}")
        lines.append(f"  Average F1:        {agg.get('avg_f1', 0):.2%}")
        lines.append(f"  F1 Std Dev:        {agg.get('std_f1', 0):.2%}")
        lines.append(f"  F1 Range:          {agg.get('min_f1', 0):.2%} - {agg.get('max_f1', 0):.2%}")
        lines.append("")

        # Generalization
        gen_score = results.get("generalization_score", 0)
        lines.append("## Generalization")
        lines.append(f"  Score: {gen_score:.2%}")
        if gen_score > 0.7:
            lines.append("  Status: GOOD - Consistent performance across datasets")
        elif gen_score > 0.5:
            lines.append("  Status: MODERATE - Some variance across datasets")
        else:
            lines.append("  Status: POOR - High variance, possible overfitting")
        lines.append("")

        # Overfitting Analysis
        overfit = results.get("overfitting_analysis", {})
        lines.append("## Overfitting Analysis")
        lines.append(f"  Is Overfitting: {overfit.get('is_overfitting', False)}")
        lines.append(f"  Recommendation: {overfit.get('recommendation', 'N/A')}")
        lines.append("")

        # Per-Dataset Results
        lines.append("## Per-Dataset Results")
        for name, metrics in results.get("per_dataset", {}).items():
            lines.append(f"\n### {name}")
            lines.append(f"  Precision: {metrics.get('precision', 0):.2%}")
            lines.append(f"  Recall:    {metrics.get('recall', 0):.2%}")
            lines.append(f"  F1:        {metrics.get('f1', 0):.2%}")
            lines.append(f"  TP: {metrics.get('true_positives', 0)}, "
                        f"FP: {metrics.get('false_positives', 0)}, "
                        f"FN: {metrics.get('false_negatives', 0)}")

        report = "\n".join(lines)

        if output_path:
            with open(output_path, "w") as f:
                f.write(report)
            logger.info(f"Report saved to: {output_path}")

        return report


# Utility function for quick validation
def validate_fk_detector(
    detector_fn: Callable[[str], List[Dict]],
    datasets: Optional[List[DatasetConfig]] = None,
) -> Dict:
    """
    FK detector 검증 유틸리티

    Args:
        detector_fn: FK 탐지 함수 (path -> List[{from_table, from_column, to_table, to_column}])
        datasets: 검증 데이터셋 (None이면 기본 데이터셋 사용)

    Returns:
        검증 결과 딕셔너리
    """
    validator = CrossDatasetValidator()

    if datasets:
        validator.datasets = datasets

    return validator.validate_all(detector_fn)
