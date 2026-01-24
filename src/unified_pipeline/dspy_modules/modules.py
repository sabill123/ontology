"""
DSPy Modules - 재사용 가능한 LLM 호출 모듈

각 모듈은 Signature를 사용하여 구조화된 입출력을 정의하고,
ChainOfThought를 통해 추론 과정을 포함합니다.

사용 예시:
    from dspy_modules import configure_dspy, FKRelationshipAnalyzer

    configure_dspy()  # 초기화

    analyzer = FKRelationshipAnalyzer()
    result = analyzer(
        source_table="orders",
        source_column="customer_id",
        source_samples=["C001", "C002"],
        target_table="customers",
        target_column="id",
        target_samples=["C001", "C002", "C003"],
        overlap_ratio=0.67,
    )
    print(result.is_relationship)  # True
    print(result.confidence)       # 0.85
"""

import json
import logging
from typing import List, Dict, Any, Optional

try:
    import dspy
except ImportError:
    raise ImportError("dspy-ai 패키지가 필요합니다. pip install dspy-ai")

from .signatures import (
    FKRelationshipSignature,
    FKDirectionSignature,
    ColumnSemanticSignature,
    ColumnTypeSignature,
    EntityAnalysisSignature,
    ThresholdDecisionSignature,
    ConsensusVoteSignature,
    CrossSiloRelationshipSignature,
    ReasoningStepSignature,
)

logger = logging.getLogger(__name__)


# ============================================================
# FK Detection Modules
# ============================================================

class FKRelationshipAnalyzer(dspy.Module):
    """
    FK 관계 분석 모듈

    두 컬럼 간의 FK 관계를 분석하고 관계 유형, 카디널리티,
    신뢰도를 반환합니다.

    Example:
        analyzer = FKRelationshipAnalyzer()
        result = analyzer(
            source_table="orders",
            source_column="customer_id",
            source_samples=["C001", "C002", "C003"],
            target_table="customers",
            target_column="id",
            target_samples=["C001", "C002", "C003", "C004"],
            overlap_ratio=0.75,
        )
        if result.is_relationship and result.confidence > 0.7:
            print(f"FK found: {result.cardinality}")
    """

    def __init__(self):
        super().__init__()
        self.analyze = dspy.ChainOfThought(FKRelationshipSignature)

    def forward(
        self,
        source_table: str,
        source_column: str,
        source_samples: List[str],
        target_table: str,
        target_column: str,
        target_samples: List[str],
        overlap_ratio: float,
    ):
        return self.analyze(
            source_table=source_table,
            source_column=source_column,
            source_samples=source_samples[:15],
            target_table=target_table,
            target_column=target_column,
            target_samples=target_samples[:15],
            overlap_ratio=overlap_ratio,
        )


class FKDirectionAnalyzer(dspy.Module):
    """
    FK 방향 분석 모듈

    두 테이블 간의 FK 관계에서 방향(source -> target)을 결정합니다.
    """

    def __init__(self):
        super().__init__()
        self.analyze = dspy.ChainOfThought(FKDirectionSignature)

    def forward(
        self,
        table_a: str,
        column_a: str,
        table_a_samples: List[str],
        table_b: str,
        column_b: str,
        table_b_samples: List[str],
    ):
        return self.analyze(
            table_a=table_a,
            column_a=column_a,
            table_a_samples=table_a_samples[:15],
            table_b=table_b,
            column_b=column_b,
            table_b_samples=table_b_samples[:15],
        )


# ============================================================
# Column Analysis Modules
# ============================================================

class ColumnSemanticClassifier(dspy.Module):
    """
    컬럼 의미역 분류 모듈

    컬럼의 비즈니스 의미역(IDENTIFIER, DIMENSION, MEASURE 등)을
    분류합니다.

    Example:
        classifier = ColumnSemanticClassifier()
        result = classifier(
            column_name="total_amount",
            table_name="orders",
            data_type="float",
            sample_values=["150.00", "299.99", "50.00"],
            distinct_ratio=0.95,
        )
        print(result.semantic_role)  # "MEASURE"
    """

    def __init__(self):
        super().__init__()
        self.classify = dspy.ChainOfThought(ColumnSemanticSignature)

    def forward(
        self,
        column_name: str,
        table_name: str,
        data_type: str,
        sample_values: List[str],
        distinct_ratio: Optional[float] = None,
    ):
        return self.classify(
            column_name=column_name,
            table_name=table_name,
            data_type=data_type,
            sample_values=sample_values[:20],
            distinct_ratio=distinct_ratio,
        )


class ColumnTypeAnalyzer(dspy.Module):
    """
    컬럼 타입 분석 모듈

    데이터 타입, 의미론적 타입, PK/FK 후보 여부를 분석합니다.
    """

    def __init__(self):
        super().__init__()
        self.analyze = dspy.ChainOfThought(ColumnTypeSignature)

    def forward(
        self,
        column_name: str,
        table_name: str,
        sample_values: List[str],
    ):
        return self.analyze(
            column_name=column_name,
            table_name=table_name,
            sample_values=sample_values[:20],
        )


# ============================================================
# Entity Analysis Modules
# ============================================================

class EntityAnalyzer(dspy.Module):
    """
    엔티티 분석 모듈

    테이블의 비즈니스 의미와 엔티티 유형을 분석합니다.

    Example:
        analyzer = EntityAnalyzer()
        result = analyzer(
            table_name="customers",
            columns=["id", "name", "email", "created_at"],
            sample_data=[{"id": 1, "name": "John", ...}],
            source_context="CRM System",
        )
        print(result.entity_type)  # "master_data"
    """

    def __init__(self):
        super().__init__()
        self.analyze = dspy.ChainOfThought(EntityAnalysisSignature)

    def forward(
        self,
        table_name: str,
        columns: List[str],
        sample_data: List[Dict],
        source_context: str,
    ):
        return self.analyze(
            table_name=table_name,
            columns=columns,
            sample_data=json.dumps(sample_data[:5], default=str, ensure_ascii=False),
            source_context=source_context,
        )


# ============================================================
# Threshold Decision Modules
# ============================================================

class DynamicThresholdDecider(dspy.Module):
    """
    동적 임계값 결정 모듈

    하드코딩된 매직 넘버 대신 데이터 특성에 맞는 임계값을
    동적으로 결정합니다.

    Example:
        decider = DynamicThresholdDecider()
        result = decider(
            threshold_type="correlation",
            data_characteristics={"num_tables": 5, "avg_null_ratio": 0.1},
            domain_context={"industry": "finance"},
        )
        print(result.recommended_value)  # 0.65
    """

    def __init__(self):
        super().__init__()
        self.decide = dspy.ChainOfThought(ThresholdDecisionSignature)

    def forward(
        self,
        threshold_type: str,
        data_characteristics: Dict[str, Any],
        domain_context: Dict[str, Any] = None,
    ):
        return self.decide(
            threshold_type=threshold_type,
            data_characteristics=json.dumps(data_characteristics, default=str, ensure_ascii=False),
            domain_context=json.dumps(domain_context or {}, default=str, ensure_ascii=False),
        )


# ============================================================
# Consensus Modules
# ============================================================

class ConsensusVoter(dspy.Module):
    """
    컨센서스 투표 모듈

    특정 역할의 관점에서 온톨로지 제안에 대해 투표합니다.

    Example:
        voter = ConsensusVoter()
        result = voter(
            proposal="Add Customer-Order relationship",
            evidence="Value overlap 95%, naming convention matches",
            role="ontologist",
            previous_votes=[],
        )
        print(result.vote)  # "approve"
    """

    def __init__(self):
        super().__init__()
        self.vote = dspy.ChainOfThought(ConsensusVoteSignature)

    def forward(
        self,
        proposal: str,
        evidence: str,
        role: str,
        previous_votes: List[Dict] = None,
    ):
        return self.vote(
            proposal=proposal,
            evidence=evidence,
            role=role,
            previous_votes=json.dumps(previous_votes or [], default=str, ensure_ascii=False),
        )


# ============================================================
# Cross-Silo Analysis Modules
# ============================================================

class CrossSiloAnalyzer(dspy.Module):
    """
    크로스 사일로 분석 모듈

    여러 데이터 소스 간의 관계를 분석합니다.
    """

    def __init__(self):
        super().__init__()
        self.analyze = dspy.ChainOfThought(CrossSiloRelationshipSignature)

    def forward(self, sources_metadata: List[Dict[str, Any]]):
        return self.analyze(
            sources_metadata=json.dumps(sources_metadata, default=str, ensure_ascii=False)
        )


# ============================================================
# Reasoning Chain Modules
# ============================================================

class ReasoningChainRunner(dspy.Module):
    """
    다단계 추론 체인 모듈

    깊은 인사이트 발굴을 위해 연속적인 질문-분석-발견 체인을
    실행합니다.

    Example:
        runner = ReasoningChainRunner(max_depth=5)
        results = runner(
            initial_question="Why is the data quality poor?",
            data_context={"tables": [...], "quality_metrics": {...}},
        )
        for step in results:
            print(f"Step {step.step_number}: {step.analysis}")
    """

    def __init__(self, max_depth: int = 5, min_confidence: float = 0.5):
        super().__init__()
        self.step = dspy.ChainOfThought(ReasoningStepSignature)
        self.max_depth = max_depth
        self.min_confidence = min_confidence

    def forward(
        self,
        initial_question: str,
        data_context: Dict[str, Any],
    ) -> List:
        """
        추론 체인 실행

        Returns:
            List of reasoning step results
        """
        steps = []
        current_question = initial_question
        previous_findings = []

        for depth in range(1, self.max_depth + 1):
            result = self.step(
                question=current_question,
                data_context=json.dumps(data_context, default=str, ensure_ascii=False)[:2000],
                previous_findings=previous_findings[-5:],  # 최근 5개만
            )

            steps.append({
                "step_number": depth,
                "question": current_question,
                "analysis": result.analysis,
                "findings": result.findings,
                "confidence": float(result.confidence) if result.confidence else 0.5,
                "next_question": result.next_question,
            })

            if result.findings:
                previous_findings.extend(result.findings)

            # 중단 조건
            confidence = float(result.confidence) if result.confidence else 0.5
            if confidence < self.min_confidence:
                logger.info(f"Reasoning stopped at step {depth}: low confidence")
                break

            if not result.next_question:
                logger.info(f"Reasoning completed at step {depth}")
                break

            current_question = result.next_question

        return steps


# ============================================================
# Optimization Utilities
# ============================================================

def compile_module(
    module: dspy.Module,
    trainset: List[dspy.Example],
    metric=None,
    teacher_settings: Dict = None,
) -> dspy.Module:
    """
    모듈 프롬프트 최적화

    Few-shot 예시를 자동 선택하여 프롬프트를 최적화합니다.

    Args:
        module: 최적화할 DSPy 모듈
        trainset: 훈련 예시 리스트 (dspy.Example)
        metric: 평가 함수 (선택)
        teacher_settings: Teacher LM 설정 (선택)

    Returns:
        최적화된 모듈

    Example:
        # 훈련 데이터 준비
        trainset = [
            dspy.Example(
                source_table="orders",
                source_column="customer_id",
                ...,
                is_relationship=True,
                confidence=0.9,
            ).with_inputs("source_table", "source_column", ...),
        ]

        # 최적화
        optimized = compile_module(analyzer, trainset)

        # 저장
        optimized.save("optimized_analyzer.json")
    """
    from dspy.teleprompt import BootstrapFewShot

    optimizer = BootstrapFewShot(
        metric=metric,
        teacher_settings=teacher_settings or {},
    )

    return optimizer.compile(module, trainset=trainset)


def save_module(module: dspy.Module, path: str) -> None:
    """최적화된 모듈 저장"""
    module.save(path)
    logger.info(f"Module saved to {path}")


def load_module(module_class, path: str) -> dspy.Module:
    """
    최적화된 모듈 로드

    Args:
        module_class: 모듈 클래스 (예: FKRelationshipAnalyzer)
        path: 저장 경로

    Returns:
        로드된 모듈

    Example:
        analyzer = load_module(FKRelationshipAnalyzer, "optimized_analyzer.json")
    """
    module = module_class()
    module.load(path)
    logger.info(f"Module loaded from {path}")
    return module
