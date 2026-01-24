# analysis/llm_semantic_enhancer.py
"""
LLM Semantic Enhancer for FK Detection

Rule-based SemanticEntityResolver가 놓친 semantic role 기반 FK 관계를
LLM을 통해 탐지합니다.

예시 (rule-based로 탐지 불가):
- diagnosed_by → doctors.doctor_id (진단한 의사)
- prescribing_doc → doctors.doctor_id (처방한 의사)
- ordering_physician → doctors.doctor_id (오더한 의사)

v1.0: 초기 구현
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# LLM 가용성 체크
_LLM_AVAILABLE = False
_DSPY_AVAILABLE = False

try:
    from src.unified_pipeline.llm_schemas import FKAnalysisResult, Cardinality
    from src.common.utils.llm_instructor import call_llm_structured
    _LLM_AVAILABLE = True
    logger.info("Instructor integration available for LLM Semantic Enhancer")
except ImportError as e:
    logger.debug(f"Instructor not available: {e}")

try:
    import dspy
    from src.unified_pipeline.dspy_modules import configure_dspy, FKRelationshipAnalyzer
    _DSPY_AVAILABLE = True
    logger.info("DSPy integration available for LLM Semantic Enhancer")
except ImportError as e:
    logger.debug(f"DSPy not available: {e}")


@dataclass
class SemanticFKResult:
    """LLM 기반 Semantic FK 분석 결과"""
    is_relationship: bool = False
    confidence: float = 0.0
    cardinality: str = "UNKNOWN"
    semantic_relationship: str = ""
    reasoning: str = ""
    method: str = "none"  # instructor, dspy, none

    def to_dict(self) -> Dict:
        return {
            "is_relationship": self.is_relationship,
            "confidence": self.confidence,
            "cardinality": self.cardinality,
            "semantic_relationship": self.semantic_relationship,
            "reasoning": self.reasoning,
            "method": self.method,
        }


class LLMSemanticEnhancer:
    """
    LLM 기반 Semantic FK 탐지 강화

    Rule-based resolver가 탐지 못한 semantic role 패턴을 LLM으로 보완합니다.

    지원 패턴:
    - Verb+Entity: diagnosed_by, created_by, assigned_to
    - Role+Entity: prescribing_doc, attending_doc, ordering_physician
    - Domain-specific: carrier, assigned_airline, member_id

    Example:
        enhancer = LLMSemanticEnhancer()
        result = enhancer.analyze_semantic_fk(
            fk_column="diagnosed_by",
            fk_table="diagnoses",
            pk_column="doctor_id",
            pk_table="doctors",
            fk_samples=["DOC001", "DOC002"],
            pk_samples=["DOC001", "DOC002", "DOC003"],
        )
        if result.is_relationship:
            print(f"FK found: {result.semantic_relationship}")
    """

    def __init__(
        self,
        use_instructor: bool = True,
        use_dspy: bool = False,
        min_confidence: float = 0.6,
        llm_client=None,
    ):
        """
        Args:
            use_instructor: Instructor 사용 여부 (기본값: True)
            use_dspy: DSPy 사용 여부 (기본값: False)
            min_confidence: 최소 신뢰도 임계값
            llm_client: 외부 LLM 클라이언트 (선택)
        """
        self.use_instructor = use_instructor and _LLM_AVAILABLE
        self.use_dspy = use_dspy and _DSPY_AVAILABLE
        self.min_confidence = min_confidence
        self.llm_client = llm_client

        # DSPy 초기화
        if self.use_dspy:
            try:
                configure_dspy()
                self._dspy_analyzer = FKRelationshipAnalyzer()
            except Exception as e:
                logger.warning(f"DSPy initialization failed: {e}")
                self.use_dspy = False
                self._dspy_analyzer = None
        else:
            self._dspy_analyzer = None

        # Semantic role 힌트 패턴 (LLM 호출 전 필터링용)
        self._semantic_role_patterns = [
            "_by",  # diagnosed_by, created_by, assigned_by
            "_doc",  # prescribing_doc, attending_doc
            "_physician",  # ordering_physician
            "_nurse",  # attending_nurse
            "_staff",  # assigned_staff
            "_manager",  # account_manager
            "_owner",  # record_owner
            "_handler",  # case_handler
            "_agent",  # sales_agent
            "assigned_",  # assigned_airline, assigned_gate
            "primary_",  # primary_doctor, primary_contact
            "secondary_",  # secondary_approver
            "carrier",  # airline carrier
            "member",  # insurance member = patient
        ]

    def should_use_llm(
        self,
        fk_column: str,
        rule_confidence: float,
    ) -> bool:
        """
        LLM 사용 여부 결정

        Rule-based confidence가 낮고, semantic role 패턴이 감지되면 LLM 사용
        """
        # Rule-based가 이미 높은 confidence면 LLM 불필요
        if rule_confidence >= 0.7:
            return False

        # Semantic role 패턴 체크
        col_lower = fk_column.lower()
        for pattern in self._semantic_role_patterns:
            if pattern in col_lower:
                return True

        # Rule-based confidence가 매우 낮으면(애매하면) LLM 시도
        if 0.3 <= rule_confidence < 0.6:
            return True

        return False

    def analyze_semantic_fk(
        self,
        fk_column: str,
        fk_table: str,
        pk_column: str,
        pk_table: str,
        fk_samples: List[str] = None,
        pk_samples: List[str] = None,
        overlap_ratio: float = 0.0,
        domain_context: str = "",
    ) -> SemanticFKResult:
        """
        LLM 기반 Semantic FK 분석

        Args:
            fk_column: FK 후보 컬럼명
            fk_table: FK 후보 테이블명
            pk_column: PK 후보 컬럼명
            pk_table: PK 후보 테이블명
            fk_samples: FK 컬럼 샘플 값
            pk_samples: PK 컬럼 샘플 값
            overlap_ratio: 값 오버랩 비율
            domain_context: 도메인 컨텍스트 (예: "healthcare")

        Returns:
            SemanticFKResult
        """
        fk_samples = fk_samples or []
        pk_samples = pk_samples or []

        # Instructor 우선 시도
        if self.use_instructor:
            try:
                return self._analyze_with_instructor(
                    fk_column, fk_table, pk_column, pk_table,
                    fk_samples, pk_samples, overlap_ratio, domain_context
                )
            except Exception as e:
                logger.warning(f"Instructor analysis failed: {e}")

        # DSPy fallback
        if self.use_dspy and self._dspy_analyzer:
            try:
                return self._analyze_with_dspy(
                    fk_column, fk_table, pk_column, pk_table,
                    fk_samples, pk_samples, overlap_ratio
                )
            except Exception as e:
                logger.warning(f"DSPy analysis failed: {e}")

        # LLM 사용 불가
        return SemanticFKResult(
            is_relationship=False,
            confidence=0.0,
            reasoning="LLM not available",
            method="none",
        )

    def _analyze_with_instructor(
        self,
        fk_column: str,
        fk_table: str,
        pk_column: str,
        pk_table: str,
        fk_samples: List[str],
        pk_samples: List[str],
        overlap_ratio: float,
        domain_context: str,
    ) -> SemanticFKResult:
        """Instructor 기반 분석"""

        system_prompt = """You are a database relationship expert specializing in identifying foreign key relationships.

Analyze if there is a semantic relationship between two columns, especially looking for:
1. Role-based references (diagnosed_by → doctor, prescribing_doc → doctor)
2. Verb patterns (created_by, assigned_to, managed_by → person/entity)
3. Domain-specific synonyms (member_id in insurance context = patient_id)
4. Abbreviated references (carrier = airline, assigned_airline = airline)

Consider column naming conventions, semantic meaning, and domain context."""

        user_prompt = f"""Analyze if there is a foreign key relationship:

## Source (Potential FK)
- Table: {fk_table}
- Column: {fk_column}
- Sample values: {fk_samples[:10]}

## Target (Potential PK)
- Table: {pk_table}
- Column: {pk_column}
- Sample values: {pk_samples[:10]}

## Context
- Value overlap ratio: {overlap_ratio:.2%}
- Domain: {domain_context or 'general'}

Determine if {fk_table}.{fk_column} references {pk_table}.{pk_column}.
Focus on SEMANTIC meaning - does the FK column name suggest it references the target entity?"""

        result = call_llm_structured(
            response_model=FKAnalysisResult,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.1,
            max_retries=2,
        )

        return SemanticFKResult(
            is_relationship=result.is_relationship,
            confidence=result.confidence,
            cardinality=result.cardinality.value if result.cardinality else "UNKNOWN",
            semantic_relationship=result.semantic_relationship,
            reasoning=result.reasoning,
            method="instructor",
        )

    def _analyze_with_dspy(
        self,
        fk_column: str,
        fk_table: str,
        pk_column: str,
        pk_table: str,
        fk_samples: List[str],
        pk_samples: List[str],
        overlap_ratio: float,
    ) -> SemanticFKResult:
        """DSPy 기반 분석"""

        result = self._dspy_analyzer(
            source_table=fk_table,
            source_column=fk_column,
            source_samples=[str(s) for s in fk_samples[:15]],
            target_table=pk_table,
            target_column=pk_column,
            target_samples=[str(s) for s in pk_samples[:15]],
            overlap_ratio=overlap_ratio,
        )

        # DSPy 결과 파싱
        is_rel = bool(result.is_relationship) if hasattr(result, 'is_relationship') else False
        conf = float(result.confidence) if hasattr(result, 'confidence') else 0.0
        card = str(result.cardinality) if hasattr(result, 'cardinality') else "UNKNOWN"
        reasoning = str(result.reasoning) if hasattr(result, 'reasoning') else ""

        return SemanticFKResult(
            is_relationship=is_rel,
            confidence=conf,
            cardinality=card,
            reasoning=reasoning,
            method="dspy",
        )

    def batch_analyze(
        self,
        candidates: List[Dict],
        tables_data: Dict[str, Dict[str, List]],
        domain_context: str = "",
    ) -> List[Dict]:
        """
        배치 분석 - 여러 FK 후보에 대해 LLM 분석 수행

        Args:
            candidates: FK 후보 리스트 (from_table, from_column, to_table, to_column 포함)
            tables_data: 테이블별 컬럼 데이터
            domain_context: 도메인 컨텍스트

        Returns:
            강화된 FK 후보 리스트
        """
        enhanced = []

        for candidate in candidates:
            fk_table = candidate.get("from_table", "")
            fk_column = candidate.get("from_column", "")
            pk_table = candidate.get("to_table", "")
            pk_column = candidate.get("to_column", "")

            # 기존 confidence
            rule_confidence = candidate.get("semantic_confidence", 0.0)

            # LLM 사용 여부 결정
            if not self.should_use_llm(fk_column, rule_confidence):
                enhanced.append(candidate)
                continue

            # 샘플 데이터 추출
            fk_samples = tables_data.get(fk_table, {}).get(fk_column, [])[:15]
            pk_samples = tables_data.get(pk_table, {}).get(pk_column, [])[:15]

            # 오버랩 계산
            fk_set = set(str(v) for v in fk_samples if v)
            pk_set = set(str(v) for v in pk_samples if v)
            overlap = len(fk_set & pk_set) / len(fk_set) if fk_set else 0.0

            # LLM 분석
            llm_result = self.analyze_semantic_fk(
                fk_column=fk_column,
                fk_table=fk_table,
                pk_column=pk_column,
                pk_table=pk_table,
                fk_samples=[str(v) for v in fk_samples],
                pk_samples=[str(v) for v in pk_samples],
                overlap_ratio=overlap,
                domain_context=domain_context,
            )

            # 결과 병합
            if llm_result.is_relationship and llm_result.confidence >= self.min_confidence:
                # LLM이 FK로 판단하고 confidence가 높으면 업데이트
                enhanced_candidate = candidate.copy()
                enhanced_candidate["llm_analysis"] = llm_result.to_dict()
                enhanced_candidate["llm_confidence"] = llm_result.confidence
                enhanced_candidate["llm_reasoning"] = llm_result.reasoning

                # semantic confidence를 LLM 결과로 업데이트
                if llm_result.confidence > rule_confidence:
                    enhanced_candidate["semantic_confidence"] = llm_result.confidence
                    enhanced_candidate["semantic_analysis"]["llm_enhanced"] = True
                    enhanced_candidate["semantic_analysis"]["llm_relationship"] = llm_result.semantic_relationship

                enhanced.append(enhanced_candidate)
                logger.info(
                    f"LLM enhanced: {fk_table}.{fk_column} → {pk_table}.{pk_column} "
                    f"(rule:{rule_confidence:.2f} → llm:{llm_result.confidence:.2f})"
                )
            else:
                # LLM도 FK가 아니라고 판단하거나 confidence가 낮으면 원본 유지
                enhanced.append(candidate)

        return enhanced


# Factory function
def create_llm_enhancer(**kwargs) -> LLMSemanticEnhancer:
    """LLM Semantic Enhancer 생성"""
    return LLMSemanticEnhancer(**kwargs)


# Quick analysis function
def analyze_semantic_fk(
    fk_column: str,
    fk_table: str,
    pk_column: str,
    pk_table: str,
    **kwargs,
) -> SemanticFKResult:
    """
    빠른 Semantic FK 분석

    Example:
        result = analyze_semantic_fk(
            fk_column="diagnosed_by",
            fk_table="diagnoses",
            pk_column="doctor_id",
            pk_table="doctors",
        )
    """
    enhancer = LLMSemanticEnhancer()
    return enhancer.analyze_semantic_fk(
        fk_column=fk_column,
        fk_table=fk_table,
        pk_column=pk_column,
        pk_table=pk_table,
        **kwargs,
    )
