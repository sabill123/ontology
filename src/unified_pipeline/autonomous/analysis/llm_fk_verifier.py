"""
LLM FK Direction Verifier v1.0

Rule-based FK Detection의 방향 오류를 LLM으로 검증/교정합니다.

문제:
- Rule-based: order_id 패턴 → orders.order_id와 shipping.order_no 둘 다 매칭
- Direction Analyzer: unique ratio로 PK 추정 → shipping.order_no를 PK로 오인
- 결과: orders.order_id → shipping.order_no (역방향)

해결:
- LLM이 테이블명, 컬럼명, 샘플 데이터를 분석하여 올바른 방향 판단
- CERTAIN 포함 전수 검증으로 오탐 방지
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from ...model_config import get_service_model

logger = logging.getLogger(__name__)

# Try to import Instructor
try:
    import instructor
    from pydantic import BaseModel, Field
    _INSTRUCTOR_AVAILABLE = True
except ImportError:
    _INSTRUCTOR_AVAILABLE = False
    logger.warning("Instructor not available for FK Direction Verifier")

# Try to import DSPy
try:
    import dspy
    _DSPY_AVAILABLE = True
except ImportError:
    _DSPY_AVAILABLE = False


class VerificationResult(Enum):
    """FK 검증 결과"""
    CORRECT = "correct"           # 방향 정확
    REVERSED = "reversed"         # 방향 반대로 교정 필요
    INVALID = "invalid"           # FK 관계 아님
    UNCERTAIN = "uncertain"       # 판단 불가


if _INSTRUCTOR_AVAILABLE:
    class FKDirectionVerification(BaseModel):
        """FK 방향 검증 결과 (Instructor용)"""
        is_correct_direction: bool = Field(
            description="현재 제안된 FK 방향이 올바른지 여부"
        )
        correct_source_table: str = Field(
            description="올바른 FK 소스 테이블 (FK를 가진 테이블)"
        )
        correct_source_column: str = Field(
            description="올바른 FK 컬럼"
        )
        correct_target_table: str = Field(
            description="올바른 PK 타겟 테이블 (참조되는 테이블)"
        )
        correct_target_column: str = Field(
            description="올바른 PK 컬럼"
        )
        confidence: float = Field(
            ge=0.0, le=1.0,
            description="검증 신뢰도 (0.0-1.0)"
        )
        reasoning: str = Field(
            description="방향 판단 이유"
        )

    class PKIdentification(BaseModel):
        """PK 식별 결과"""
        table_name: str = Field(description="테이블명")
        primary_key_column: str = Field(description="Primary Key 컬럼명")
        confidence: float = Field(ge=0.0, le=1.0, description="신뢰도")
        reasoning: str = Field(description="PK 판단 이유")


if _DSPY_AVAILABLE:
    class FKDirectionVerifierSignature(dspy.Signature):
        """FK 관계의 방향이 올바른지 검증합니다."""

        proposed_fk: str = dspy.InputField(
            desc="제안된 FK 관계: source_table.source_col → target_table.target_col"
        )
        source_table_info: str = dspy.InputField(
            desc="소스 테이블 정보 (컬럼, 샘플 데이터)"
        )
        target_table_info: str = dspy.InputField(
            desc="타겟 테이블 정보 (컬럼, 샘플 데이터)"
        )
        domain_context: str = dspy.InputField(
            desc="도메인 컨텍스트 (e.g., ecommerce, healthcare)"
        )

        is_correct_direction: bool = dspy.OutputField(
            desc="현재 방향이 올바른지 여부"
        )
        correct_direction: str = dspy.OutputField(
            desc="올바른 FK 방향 (source.col → target.col)"
        )
        reasoning: str = dspy.OutputField(
            desc="방향 판단 이유"
        )

    class PKIdentifierSignature(dspy.Signature):
        """테이블의 Primary Key를 식별합니다."""

        table_name: str = dspy.InputField(desc="테이블명")
        columns: str = dspy.InputField(desc="컬럼 목록")
        sample_data: str = dspy.InputField(desc="샘플 데이터")
        domain_context: str = dspy.InputField(desc="도메인 컨텍스트")

        primary_key: str = dspy.OutputField(desc="Primary Key 컬럼명")
        confidence: float = dspy.OutputField(desc="신뢰도 (0.0-1.0)")
        reasoning: str = dspy.OutputField(desc="PK 판단 이유")


@dataclass
class VerifiedFK:
    """검증된 FK 관계"""
    source_table: str
    source_column: str
    target_table: str
    target_column: str
    original_confidence: float
    verified_confidence: float
    was_corrected: bool
    correction_reason: Optional[str] = None
    verification_method: str = "llm_direction_verifier"


class LLMFKDirectionVerifier:
    """
    LLM 기반 FK 방향 검증기

    Rule-based FK Detection의 방향 오류를 검증하고 교정합니다.

    Features:
    - PK 사전 식별: 각 테이블의 진짜 PK를 먼저 식별
    - 방향 검증: 제안된 FK 방향이 올바른지 확인
    - 전수 검증: CERTAIN 포함 모든 FK 후보 검증
    """

    def __init__(
        self,
        use_instructor: bool = True,
        use_dspy: bool = False,
        min_confidence: float = 0.7,
        llm_client = None,
    ):
        self.use_instructor = use_instructor and _INSTRUCTOR_AVAILABLE
        self.use_dspy = use_dspy and _DSPY_AVAILABLE
        self.min_confidence = min_confidence
        self.llm_client = llm_client

        # PK cache per table
        self._pk_cache: Dict[str, str] = {}

        if self.use_instructor and llm_client:
            self._instructor_client = instructor.from_openai(llm_client)
        else:
            self._instructor_client = None

        if self.use_dspy:
            self._dspy_verifier = dspy.ChainOfThought(FKDirectionVerifierSignature)
            self._dspy_pk_identifier = dspy.ChainOfThought(PKIdentifierSignature)
        else:
            self._dspy_verifier = None
            self._dspy_pk_identifier = None

        logger.info(
            f"LLMFKDirectionVerifier initialized: "
            f"instructor={self.use_instructor}, dspy={self.use_dspy}"
        )

    def identify_pk(
        self,
        table_name: str,
        columns: List[str],
        sample_data: List[Dict[str, Any]],
        domain_context: str = "",
    ) -> Tuple[str, float]:
        """
        테이블의 Primary Key를 식별합니다.

        Args:
            table_name: 테이블명
            columns: 컬럼 목록
            sample_data: 샘플 데이터
            domain_context: 도메인 컨텍스트

        Returns:
            (pk_column, confidence) 튜플
        """
        # Cache check
        if table_name in self._pk_cache:
            return self._pk_cache[table_name], 0.95

        # Heuristic PK detection first
        pk_candidate = self._heuristic_pk_detection(table_name, columns)
        if pk_candidate:
            self._pk_cache[table_name] = pk_candidate
            return pk_candidate, 0.9

        # LLM fallback
        if self.use_instructor and self._instructor_client:
            try:
                result = self._identify_pk_with_instructor(
                    table_name, columns, sample_data, domain_context
                )
                if result:
                    self._pk_cache[table_name] = result[0]
                    return result
            except Exception as e:
                logger.warning(f"Instructor PK identification failed: {e}")

        # Default: first column ending with _id
        for col in columns:
            if col.lower().endswith('_id'):
                self._pk_cache[table_name] = col
                return col, 0.5

        return columns[0] if columns else "", 0.3

    def _heuristic_pk_detection(
        self,
        table_name: str,
        columns: List[str],
    ) -> Optional[str]:
        """휴리스틱 기반 PK 탐지"""
        table_lower = table_name.lower()
        table_singular = table_lower.rstrip('s')

        # Common PK patterns
        pk_patterns = [
            f"{table_singular}_id",      # orders -> order_id
            f"{table_lower}_id",          # orders -> orders_id
            f"{table_singular}_code",     # products -> product_code, prod_code
            "id",                          # just 'id'
        ]

        # Also check abbreviated forms
        if len(table_singular) > 4:
            abbrev = table_singular[:4]
            pk_patterns.append(f"{abbrev}_id")      # products -> prod_id
            pk_patterns.append(f"{abbrev}_code")    # products -> prod_code

        for pattern in pk_patterns:
            for col in columns:
                if col.lower() == pattern:
                    return col

        return None

    def _identify_pk_with_instructor(
        self,
        table_name: str,
        columns: List[str],
        sample_data: List[Dict[str, Any]],
        domain_context: str,
    ) -> Optional[Tuple[str, float]]:
        """Instructor를 사용한 PK 식별"""
        if not self._instructor_client:
            return None

        # Format sample data
        sample_str = ""
        for i, row in enumerate(sample_data[:3]):
            sample_str += f"Row {i+1}: {row}\n"

        prompt = f"""Identify the Primary Key column for this table.

Table: {table_name}
Columns: {columns}
Domain: {domain_context}

Sample Data:
{sample_str}

The Primary Key is the unique identifier column that:
1. Has unique values for each row
2. Usually ends with _id, _code, or similar
3. Matches the table name pattern (e.g., orders table -> order_id)
"""

        try:
            result = self._instructor_client.chat.completions.create(
                model=get_service_model("fk_verification"),
                response_model=PKIdentification,
                messages=[
                    {"role": "system", "content": "You are a database schema expert."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500,
            )
            return (result.primary_key_column, result.confidence)
        except Exception as e:
            logger.warning(f"Instructor PK identification error: {e}")
            return None

    def verify_fk_direction(
        self,
        source_table: str,
        source_column: str,
        target_table: str,
        target_column: str,
        source_table_info: Dict[str, Any],
        target_table_info: Dict[str, Any],
        domain_context: str = "",
    ) -> VerifiedFK:
        """
        FK 방향을 검증하고 필요시 교정합니다.

        Args:
            source_table: 제안된 FK 소스 테이블
            source_column: 제안된 FK 컬럼
            target_table: 제안된 PK 타겟 테이블
            target_column: 제안된 PK 컬럼
            source_table_info: 소스 테이블 정보
            target_table_info: 타겟 테이블 정보
            domain_context: 도메인 컨텍스트

        Returns:
            VerifiedFK 객체
        """
        # Step 1: Identify true PKs for both tables
        source_pk, _ = self.identify_pk(
            source_table,
            source_table_info.get('columns', []),
            source_table_info.get('sample_data', []),
            domain_context
        )

        target_pk, _ = self.identify_pk(
            target_table,
            target_table_info.get('columns', []),
            target_table_info.get('sample_data', []),
            domain_context
        )

        # Step 2: Check if direction is obviously wrong
        # If source_column is the PK of source_table, direction is wrong
        is_source_col_pk = (source_column.lower() == source_pk.lower())
        is_target_col_pk = (target_column.lower() == target_pk.lower())

        if is_source_col_pk and not is_target_col_pk:
            # Source column is a PK, target column is not
            # This is reversed! The FK should be target → source
            logger.info(
                f"Direction correction: {source_table}.{source_column} is PK, "
                f"reversing to {target_table}.{target_column} → {source_table}.{source_column}"
            )
            return VerifiedFK(
                source_table=target_table,
                source_column=target_column,
                target_table=source_table,
                target_column=source_column,
                original_confidence=0.0,  # Will be set by caller
                verified_confidence=0.85,
                was_corrected=True,
                correction_reason=f"{source_column} is the PK of {source_table}, not an FK",
            )

        if not is_source_col_pk and is_target_col_pk:
            # Correct direction: source references target PK
            return VerifiedFK(
                source_table=source_table,
                source_column=source_column,
                target_table=target_table,
                target_column=target_column,
                original_confidence=0.0,
                verified_confidence=0.9,
                was_corrected=False,
            )

        # Step 3: Ambiguous case - use LLM for deeper analysis
        if self.use_instructor and self._instructor_client:
            return self._verify_with_instructor(
                source_table, source_column,
                target_table, target_column,
                source_table_info, target_table_info,
                domain_context
            )

        # Default: trust the original direction
        return VerifiedFK(
            source_table=source_table,
            source_column=source_column,
            target_table=target_table,
            target_column=target_column,
            original_confidence=0.0,
            verified_confidence=0.6,
            was_corrected=False,
            correction_reason="Could not verify, using original direction",
        )

    def _verify_with_instructor(
        self,
        source_table: str,
        source_column: str,
        target_table: str,
        target_column: str,
        source_table_info: Dict[str, Any],
        target_table_info: Dict[str, Any],
        domain_context: str,
    ) -> VerifiedFK:
        """Instructor를 사용한 FK 방향 검증"""

        # Format table info
        source_cols = source_table_info.get('columns', [])
        target_cols = target_table_info.get('columns', [])

        source_sample = source_table_info.get('sample_data', [])[:3]
        target_sample = target_table_info.get('sample_data', [])[:3]

        prompt = f"""Verify the direction of this Foreign Key relationship.

Proposed FK: {source_table}.{source_column} → {target_table}.{target_column}

SOURCE TABLE: {source_table}
Columns: {source_cols}
Sample: {source_sample}

TARGET TABLE: {target_table}
Columns: {target_cols}
Sample: {target_sample}

Domain: {domain_context}

Rules for FK direction:
1. FK column references PK column
2. FK table has many rows, PK table has fewer (usually)
3. FK column name often includes target table name (e.g., customer_id references customers)
4. PK column usually matches table name pattern (e.g., orders.order_id)

Is the proposed direction correct? If not, what is the correct direction?
"""

        try:
            result = self._instructor_client.chat.completions.create(
                model=get_service_model("fk_verification"),
                response_model=FKDirectionVerification,
                messages=[
                    {"role": "system", "content": "You are a database schema expert specializing in FK relationships."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=800,
            )

            if result.is_correct_direction:
                return VerifiedFK(
                    source_table=source_table,
                    source_column=source_column,
                    target_table=target_table,
                    target_column=target_column,
                    original_confidence=0.0,
                    verified_confidence=result.confidence,
                    was_corrected=False,
                    correction_reason=result.reasoning,
                )
            else:
                return VerifiedFK(
                    source_table=result.correct_source_table,
                    source_column=result.correct_source_column,
                    target_table=result.correct_target_table,
                    target_column=result.correct_target_column,
                    original_confidence=0.0,
                    verified_confidence=result.confidence,
                    was_corrected=True,
                    correction_reason=result.reasoning,
                )

        except Exception as e:
            logger.warning(f"Instructor FK verification error: {e}")
            return VerifiedFK(
                source_table=source_table,
                source_column=source_column,
                target_table=target_table,
                target_column=target_column,
                original_confidence=0.0,
                verified_confidence=0.5,
                was_corrected=False,
                correction_reason=f"Verification failed: {e}",
            )

    def verify_all_fks(
        self,
        fk_candidates: List[Dict[str, Any]],
        tables_data: Dict[str, Any],
        domain_context: str = "",
    ) -> List[Dict[str, Any]]:
        """
        모든 FK 후보를 검증합니다 (CERTAIN 포함 전수 검증).

        Args:
            fk_candidates: FK 후보 리스트
            tables_data: 테이블 데이터
            domain_context: 도메인 컨텍스트

        Returns:
            검증/교정된 FK 리스트
        """
        verified_fks = []
        corrections_made = 0

        for fk in fk_candidates:
            source_table = fk.get('source_table', fk.get('fk_table', ''))
            source_column = fk.get('source_column', fk.get('fk_column', ''))
            target_table = fk.get('target_table', fk.get('pk_table', ''))
            target_column = fk.get('target_column', fk.get('pk_column', ''))
            original_conf = fk.get('confidence', fk.get('fk_score', 0))

            # Get table info
            source_info = tables_data.get(source_table, {})
            target_info = tables_data.get(target_table, {})

            # Verify direction
            verified = self.verify_fk_direction(
                source_table, source_column,
                target_table, target_column,
                source_info, target_info,
                domain_context
            )

            # Build verified FK dict
            verified_fk = {
                'source_table': verified.source_table,
                'source_column': verified.source_column,
                'target_table': verified.target_table,
                'target_column': verified.target_column,
                'confidence': verified.verified_confidence,
                'original_confidence': original_conf if isinstance(original_conf, (int, float)) else 0.8,
                'was_corrected': verified.was_corrected,
                'correction_reason': verified.correction_reason,
                'detection_method': fk.get('detection_method', 'rule_based'),
                'verification_method': 'llm_fk_direction_verifier',
            }

            # Copy other fields
            for key in ['semantic_relationship', 'reasoning', 'confidence_level']:
                if key in fk:
                    verified_fk[key] = fk[key]

            verified_fks.append(verified_fk)

            if verified.was_corrected:
                corrections_made += 1
                logger.info(
                    f"FK Direction Corrected: "
                    f"{source_table}.{source_column} → {target_table}.{target_column} "
                    f"changed to {verified.source_table}.{verified.source_column} → "
                    f"{verified.target_table}.{verified.target_column}"
                )

        logger.info(
            f"FK Direction Verification complete: "
            f"{len(verified_fks)} verified, {corrections_made} corrected"
        )

        return verified_fks


def create_fk_verifier(llm_client=None) -> LLMFKDirectionVerifier:
    """FK Direction Verifier 생성 헬퍼"""
    return LLMFKDirectionVerifier(
        use_instructor=True,
        use_dspy=False,
        llm_client=llm_client,
    )
