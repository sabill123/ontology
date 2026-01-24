"""
LLM-Augmented FK Validator (v1.0)

LLM을 활용한 FK 후보 검증 모듈
- False Positive 필터링
- 의미론적 FK 관계 검증
- Cross-Platform FK 탐지
- Self-Verification 패턴

핵심 원칙: "Algorithm as Evidence, LLM as Judge"
"""

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Set, Tuple
from enum import Enum

logger = logging.getLogger(__name__)


class FKRelationshipType(str, Enum):
    """FK 관계 유형"""
    STRONG_FK = "strong_fk"           # 명확한 FK (campaign_id → campaign_id)
    WEAK_FK = "weak_fk"               # 약한 FK (naming convention 차이)
    CROSS_PLATFORM = "cross_platform" # 크로스 플랫폼 FK
    CORRELATION = "correlation"       # 상관관계 (FK 아님)
    FALSE_POSITIVE = "false_positive" # 오탐


@dataclass
class FKValidationResult:
    """FK 검증 결과"""
    is_valid_fk: bool
    confidence: float
    relationship_type: FKRelationshipType
    reasoning: str
    concerns: List[str] = field(default_factory=list)
    evidence_summary: Dict[str, Any] = field(default_factory=dict)

    # Self-verification 결과
    verified: bool = False
    verified_confidence: float = 0.0
    verification_notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_valid_fk": self.is_valid_fk,
            "confidence": self.confidence,
            "relationship_type": self.relationship_type.value,
            "reasoning": self.reasoning,
            "concerns": self.concerns,
            "evidence_summary": self.evidence_summary,
            "verified": self.verified,
            "verified_confidence": self.verified_confidence,
        }


class FalsePositiveFilter:
    """
    FK False Positive 필터

    메타데이터, 공통 컬럼 등 FK가 아닌 관계를 필터링
    """

    # Platform prefixes for isolation
    PLATFORM_PREFIXES = ["google_ads_", "meta_ads_", "naver_ads_", "facebook_ads_"]

    # False Positive 패턴 (확장 가능)
    FP_COLUMN_PATTERNS = {
        # 메타데이터 컬럼
        "metadata": {
            "status", "state", "flag", "is_active", "is_deleted",
            "created_at", "created_time", "created_date", "reg_time", "reg_date",
            "updated_at", "updated_time", "modified_at", "modified_time", "edit_time",
            "deleted_at", "deleted_time",
        },
        # 공통 속성 컬럼 (FK가 아닌 속성)
        "common_attributes": {
            "name", "title", "description", "type", "category",
            "value", "amount", "count", "total",
            "start_date", "end_date", "date", "start_time", "stop_time",
            "budget", "daily_budget", "spend", "cost",
            "currency", "timezone", "language", "country",
        },
        # 기술적 컬럼
        "technical": {
            "version", "revision", "sequence", "order", "sort_order",
            "hash", "checksum", "token",
        },
        # 광고 플랫폼 공통 속성 (FK가 아님)
        # NOTE: customer_id, user_id는 광고 플랫폼에서는 공통 속성이지만
        # e-commerce, marketing 등 다른 도메인에서는 유효한 FK이므로 제외
        "ad_common": {
            "account_id",  # 광고 계정 ID (보통 플랫폼별 별도)
            "bid_amount", "bid_strategy", "delivery_method",
            "effective_status", "configured_status",
        }
    }

    # FK로 보이지만 실제로는 아닌 패턴
    FP_RELATIONSHIP_PATTERNS = [
        # 동일 컬럼명이지만 FK가 아닌 경우
        ("status", "status"),
        ("type", "type"),
        ("name", "name"),
        ("category", "category"),
        ("created_time", "created_time"),
        ("updated_time", "updated_time"),
        ("created_at", "created_at"),
        ("updated_at", "updated_at"),
        ("date", "date"),
        ("start_time", "start_time"),
        ("end_time", "end_time"),
    ]

    # 값이 겹치더라도 FK가 아닌 경우 (enumerated values)
    ENUM_VALUE_PATTERNS = {
        "status": {"active", "inactive", "paused", "deleted", "enabled", "disabled"},
        "type": {"standard", "custom", "default", "premium"},
        "gender": {"male", "female", "other", "unknown"},
    }

    def __init__(self):
        # Flatten all FP columns for quick lookup
        self._fp_columns: Set[str] = set()
        for category_cols in self.FP_COLUMN_PATTERNS.values():
            self._fp_columns.update(category_cols)

    def is_likely_false_positive(
        self,
        from_column: str,
        to_column: str,
        sample_values: List[Any] = None,
        from_table: str = "",
        to_table: str = "",
    ) -> Tuple[bool, str]:
        """
        False Positive 가능성 판단

        Returns:
            (is_fp, reason)
        """
        from_col_lower = from_column.lower()
        to_col_lower = to_column.lower()
        from_table_lower = from_table.lower()
        to_table_lower = to_table.lower()

        # 0. Platform Isolation 체크 (다른 플랫폼 간 직접 FK는 불가)
        if from_table and to_table:
            from_platform = self._extract_platform(from_table_lower)
            to_platform = self._extract_platform(to_table_lower)

            # cross_platform_mapping은 예외
            if from_platform and to_platform and from_platform != to_platform:
                if "cross_platform" not in from_table_lower and "cross_platform" not in to_table_lower:
                    return True, f"Cross-platform FK not allowed: {from_platform} → {to_platform}"

        # 1. 명시적 FP 패턴 체크
        if (from_col_lower, to_col_lower) in self.FP_RELATIONSHIP_PATTERNS:
            return True, f"Known FP pattern: {from_column} ↔ {to_column}"

        # 2. 메타데이터 컬럼 체크
        if from_col_lower in self._fp_columns and to_col_lower in self._fp_columns:
            return True, f"Both columns are metadata/common attributes"

        # 3. ID 패턴이 아닌 동일 컬럼명
        if from_col_lower == to_col_lower:
            if not self._is_id_column(from_col_lower):
                return True, f"Same column name but not ID pattern: {from_column}"

        # 4. Enum 값 체크
        if sample_values:
            for col_pattern, enum_values in self.ENUM_VALUE_PATTERNS.items():
                if col_pattern in from_col_lower or col_pattern in to_col_lower:
                    sample_set = set(str(v).lower() for v in sample_values if v)
                    if sample_set.issubset(enum_values):
                        return True, f"Values appear to be enumerated, not FK references"

        # 5. 광고 공통 속성 컬럼 체크 (account_id, customer_id 등)
        ad_common_cols = self.FP_COLUMN_PATTERNS.get("ad_common", set())
        if from_col_lower in ad_common_cols or to_col_lower in ad_common_cols:
            # campaign_id, ad_id 등은 제외 (실제 FK)
            if not self._is_valid_fk_id(from_col_lower) and not self._is_valid_fk_id(to_col_lower):
                return True, f"Common attribute column, not FK: {from_column} or {to_column}"

        # 6. ID 타입 불일치 체크 (ad_id → campaign_id 등은 불가)
        from_id_type = self._extract_id_type(from_col_lower)
        to_id_type = self._extract_id_type(to_col_lower)
        if from_id_type and to_id_type and from_id_type != to_id_type:
            return True, f"ID type mismatch: {from_id_type} → {to_id_type}"

        # 7. FK 방향 검증 (하위 테이블 → 상위 테이블만 허용)
        if from_table and to_table:
            if self._is_invalid_fk_direction(from_table_lower, to_table_lower):
                return True, f"Invalid FK direction: {from_table} → {to_table}"

        # 8. Cross-platform FK 검증: campaigns 테이블로만 연결 가능
        if from_table and to_table:
            if "cross_platform" in from_table_lower:
                # cross_platform에서 나가는 FK는 campaigns 테이블로만 가야 함
                if "campaign" in from_col_lower and "campaigns" not in to_table_lower:
                    return True, f"Cross-platform FK must point to campaigns table: {to_table}"
            if "cross_platform" in to_table_lower:
                # cross_platform으로 들어오는 FK는 잘못됨 (역방향)
                return True, f"Invalid FK direction to cross_platform_mapping: {from_table}"

        # 9. 시간/금액 컬럼은 FK가 아님
        time_money_patterns = ["time", "date", "amount", "bid", "cost", "spend", "cpc", "cpm"]
        for pattern in time_money_patterns:
            if pattern in from_col_lower or pattern in to_col_lower:
                if from_col_lower != to_col_lower:  # 같은 이름이 아니면
                    return True, f"Time/money columns are not FK: {from_column} → {to_column}"

        # 10. FK는 PK를 참조해야 함 (campaign_id는 campaigns 테이블만, ad_group_id는 ad_groups만)
        if from_table and to_table:
            # 광고 플랫폼 FK → PK 매핑
            valid_fk_targets = {
                "campaign_id": ["campaigns"],
                "ad_group_id": ["ad_groups"],
                "adgroup_id": ["adgroups"],
                "adset_id": ["adsets"],
                "ad_id": ["ads"],
                "keyword_id": ["keywords"],
            }
            if from_col_lower in valid_fk_targets:
                expected_targets = valid_fk_targets[from_col_lower]
                # 타겟 테이블이 expected_targets 중 하나를 포함해야 함
                if not any(target in to_table_lower for target in expected_targets):
                    return True, f"FK {from_column} must reference {expected_targets} table, not {to_table}"

        return False, ""

    def _extract_id_type(self, col_name: str) -> Optional[str]:
        """컬럼명에서 ID 타입 추출 (campaign_id → campaign, ad_id → ad)"""
        id_patterns = {
            "campaign_id": "campaign",
            "ad_id": "ad",
            "ad_group_id": "ad_group",
            "adgroup_id": "adgroup",
            "adset_id": "adset",
            "keyword_id": "keyword",
            "creative_id": "creative",
            "metric_id": "metric",
            "stat_id": "stat",
            "insight_id": "insight",
        }
        return id_patterns.get(col_name)

    def _is_invalid_fk_direction(self, from_table: str, to_table: str) -> bool:
        """FK 방향이 잘못되었는지 확인 (상위 → 하위는 잘못됨)"""
        # 광고 플랫폼 계층 구조: campaigns > ad_groups > ads > keywords/metrics/stats/insights
        # 숫자가 작을수록 상위 테이블
        hierarchy_levels = {
            "campaigns": 1,
            "ad_groups": 2, "adgroups": 2, "adsets": 2,
            "ads": 3,
            "keywords": 4, "metrics": 4, "stats": 4, "insights": 4,
        }

        from_level = 0
        to_level = 0

        for entity, level in hierarchy_levels.items():
            if entity in from_table:
                from_level = max(from_level, level)  # 가장 높은 레벨 사용
            if entity in to_table:
                to_level = max(to_level, level)

        # FK는 하위 → 상위로 가야 함 (from_level > to_level)
        # 예: ads(3) → campaigns(1) = OK, campaigns(1) → ads(3) = Invalid
        if from_level > 0 and to_level > 0:
            if from_level < to_level:  # 상위에서 하위로 가는 FK는 잘못됨
                return True

        return False

    def _extract_platform(self, table_name: str) -> Optional[str]:
        """테이블명에서 플랫폼 추출"""
        for prefix in self.PLATFORM_PREFIXES:
            if table_name.startswith(prefix):
                return prefix.rstrip("_")
        return None

    def _is_valid_fk_id(self, col_name: str) -> bool:
        """유효한 FK ID 컬럼인지 확인 (campaign_id, ad_id 등)"""
        valid_fk_patterns = [
            "campaign_id", "ad_id", "adgroup_id", "ad_group_id",
            "adset_id", "keyword_id", "creative_id",
        ]
        return col_name in valid_fk_patterns

    def _is_id_column(self, col_name: str) -> bool:
        """ID 관련 컬럼인지 확인"""
        id_patterns = ["_id", "id_", "_key", "_code", "_ref", "uuid", "guid"]
        return any(p in col_name for p in id_patterns) or col_name == "id"

    def get_fp_category(self, column_name: str) -> Optional[str]:
        """컬럼이 속한 FP 카테고리 반환"""
        col_lower = column_name.lower()
        for category, columns in self.FP_COLUMN_PATTERNS.items():
            if col_lower in columns:
                return category
        return None


class LLMFKValidator:
    """
    LLM 기반 FK 검증기

    알고리즘이 생성한 FK 후보를 LLM이 검증
    """

    def __init__(self, llm_client=None):
        self.llm_client = llm_client
        self.fp_filter = FalsePositiveFilter()

        # 도메인별 FK 패턴 힌트
        self.domain_hints = {
            "advertising": {
                "entity_hierarchy": ["account", "campaign", "adgroup", "ad", "keyword"],
                "common_fks": ["campaign_id", "adgroup_id", "ad_id", "account_id"],
                "cross_platform_patterns": [
                    ("google_campaign_id", "campaign_id"),
                    ("meta_campaign_id", "campaign_id"),
                    ("naver_campaign_id", "campaign_id"),
                ],
            },
            "ecommerce": {
                "entity_hierarchy": ["store", "category", "product", "order", "customer"],
                "common_fks": ["product_id", "order_id", "customer_id", "category_id"],
            },
        }

    async def validate_fk_candidate(
        self,
        candidate: Dict[str, Any],
        source_table_info: Dict[str, Any],
        target_table_info: Dict[str, Any],
        domain: str = "general",
    ) -> FKValidationResult:
        """
        FK 후보 검증

        Args:
            candidate: FK 후보 정보
            source_table_info: 소스 테이블 정보 (columns, sample_data)
            target_table_info: 타겟 테이블 정보
            domain: 도메인 힌트

        Returns:
            FKValidationResult
        """
        from_table = candidate.get("from_table", candidate.get("table_a", ""))
        from_column = candidate.get("from_column", candidate.get("column_a", ""))
        to_table = candidate.get("to_table", candidate.get("table_b", ""))
        to_column = candidate.get("to_column", candidate.get("column_b", ""))

        # 1. Rule-based False Positive 체크
        is_fp, fp_reason = self.fp_filter.is_likely_false_positive(
            from_column, to_column,
            candidate.get("sample_matches", [])
        )

        if is_fp:
            return FKValidationResult(
                is_valid_fk=False,
                confidence=0.95,
                relationship_type=FKRelationshipType.FALSE_POSITIVE,
                reasoning=fp_reason,
                concerns=[fp_reason],
                evidence_summary={"fp_filter": "rule_based"},
            )

        # 2. LLM 검증 (LLM 클라이언트가 있는 경우)
        if self.llm_client:
            llm_result = await self._validate_with_llm(
                candidate, source_table_info, target_table_info, domain
            )

            # 3. Self-verification (높은 신뢰도가 필요한 경우)
            if 0.5 <= llm_result.confidence < 0.9:
                llm_result = await self._self_verify(llm_result, candidate)

            return llm_result

        # LLM 없이 알고리즘 결과만 사용
        return self._fallback_validation(candidate)

    async def _validate_with_llm(
        self,
        candidate: Dict[str, Any],
        source_info: Dict[str, Any],
        target_info: Dict[str, Any],
        domain: str,
    ) -> FKValidationResult:
        """LLM을 사용한 FK 검증"""

        prompt = self._build_validation_prompt(candidate, source_info, target_info, domain)

        try:
            response = await self.llm_client.generate(prompt, temperature=0.1)
            return self._parse_llm_response(response)
        except Exception as e:
            logger.warning(f"LLM validation failed: {e}")
            return self._fallback_validation(candidate)

    def _build_validation_prompt(
        self,
        candidate: Dict[str, Any],
        source_info: Dict[str, Any],
        target_info: Dict[str, Any],
        domain: str,
    ) -> str:
        """LLM 검증 프롬프트 생성"""

        from_table = candidate.get("from_table", candidate.get("table_a", ""))
        from_column = candidate.get("from_column", candidate.get("column_a", ""))
        to_table = candidate.get("to_table", candidate.get("table_b", ""))
        to_column = candidate.get("to_column", candidate.get("column_b", ""))

        # 도메인 힌트
        domain_context = ""
        if domain in self.domain_hints:
            hints = self.domain_hints[domain]
            domain_context = f"""
### Domain Context: {domain}
- Entity Hierarchy: {' → '.join(hints.get('entity_hierarchy', []))}
- Common FK patterns: {', '.join(hints.get('common_fks', []))}
"""

        prompt = f"""## FK Candidate Validation

You are validating a potential Foreign Key relationship.

### Candidate Details
- Source: `{from_table}`.`{from_column}`
- Target: `{to_table}`.`{to_column}`

### Algorithm Evidence
- Name Similarity: {candidate.get('name_similarity', 0):.2f}
- Value Inclusion: {candidate.get('value_inclusion', 0):.2f}
- Cardinality Ratio: {candidate.get('cardinality_ratio', 0):.2f}
- Sample Matches: {candidate.get('sample_matches', [])[:5]}
- Unmatched Count: {candidate.get('unmatched_count', 0)} / {candidate.get('total_count', 0)}

### Source Table Context
- Table: {from_table}
- Columns: {source_info.get('columns', [])}
- Sample Data (first 3 rows): {json.dumps(source_info.get('sample_data', [])[:3], default=str, ensure_ascii=False)}

### Target Table Context
- Table: {to_table}
- Columns: {target_info.get('columns', [])}
- Sample Data (first 3 rows): {json.dumps(target_info.get('sample_data', [])[:3], default=str, ensure_ascii=False)}
{domain_context}
### Validation Criteria
1. **Semantic Relationship**: Does the source column semantically reference the target table?
2. **Referential Integrity**: Do the values indicate a real FK relationship?
3. **Domain Logic**: Is this relationship meaningful in the business domain?
4. **Naming Convention**: Do the column names follow FK naming patterns?

### False Positive Indicators (IMPORTANT)
- Both columns are metadata (status, created_at, updated_at, type, category)
- Values are coincidental matches (common strings like "active", sequential numbers starting from 1)
- Columns represent attributes, not references (name, description, amount)
- Same column name but values are enumerated/categorical, not IDs

### Response Format (JSON only, no explanation outside JSON)
```json
{{
    "is_valid_fk": true or false,
    "confidence": 0.0 to 1.0,
    "relationship_type": "strong_fk" or "weak_fk" or "cross_platform" or "correlation" or "false_positive",
    "reasoning": "brief explanation",
    "concerns": ["list of concerns if any"]
}}
```
"""
        return prompt

    def _parse_llm_response(self, response: str) -> FKValidationResult:
        """LLM 응답 파싱"""
        try:
            # JSON 블록 추출
            json_match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                # JSON 블록 없이 직접 JSON인 경우
                json_str = response

            data = json.loads(json_str)

            return FKValidationResult(
                is_valid_fk=data.get("is_valid_fk", False),
                confidence=float(data.get("confidence", 0.5)),
                relationship_type=FKRelationshipType(data.get("relationship_type", "correlation")),
                reasoning=data.get("reasoning", ""),
                concerns=data.get("concerns", []),
                evidence_summary={"source": "llm_validation"},
            )
        except Exception as e:
            logger.warning(f"Failed to parse LLM response: {e}")
            return FKValidationResult(
                is_valid_fk=False,
                confidence=0.3,
                relationship_type=FKRelationshipType.CORRELATION,
                reasoning=f"Failed to parse LLM response: {e}",
                concerns=["LLM response parsing failed"],
            )

    async def _self_verify(
        self,
        initial_result: FKValidationResult,
        candidate: Dict[str, Any],
    ) -> FKValidationResult:
        """Self-verification 패턴"""

        if not self.llm_client:
            return initial_result

        prompt = f"""## Self-Verification

You previously assessed this FK candidate:

### Previous Assessment
- Is Valid FK: {initial_result.is_valid_fk}
- Confidence: {initial_result.confidence:.2f}
- Relationship Type: {initial_result.relationship_type.value}
- Reasoning: {initial_result.reasoning}
- Concerns: {initial_result.concerns}

### Original Evidence
- From: {candidate.get('from_table', '')}.{candidate.get('from_column', '')}
- To: {candidate.get('to_table', '')}.{candidate.get('to_column', '')}
- Value Inclusion: {candidate.get('value_inclusion', 0):.2f}

### Verification Questions
1. Is the confidence level justified by the evidence?
2. Could this be a false positive that was missed?
3. Are there alternative interpretations?

### Task
Re-evaluate and provide verified confidence. Be MORE CONSERVATIVE if uncertain.

### Response Format (JSON only)
```json
{{
    "verified_confidence": 0.0 to 1.0,
    "is_valid_fk": true or false,
    "verification_notes": "brief notes on what changed or confirmed"
}}
```
"""

        try:
            response = await self.llm_client.generate(prompt, temperature=0.1)

            json_match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group(1))
            else:
                data = json.loads(response)

            initial_result.verified = True
            initial_result.verified_confidence = float(data.get("verified_confidence", initial_result.confidence))
            initial_result.verification_notes = data.get("verification_notes", "")

            # 검증 후 confidence 업데이트
            if initial_result.verified_confidence < initial_result.confidence:
                initial_result.confidence = initial_result.verified_confidence

            # 검증 후 is_valid_fk 업데이트
            if "is_valid_fk" in data:
                initial_result.is_valid_fk = data["is_valid_fk"]

        except Exception as e:
            logger.warning(f"Self-verification failed: {e}")
            initial_result.verified = False

        return initial_result

    def _fallback_validation(self, candidate: Dict[str, Any]) -> FKValidationResult:
        """LLM 없이 알고리즘 결과만으로 검증"""

        score = candidate.get("fk_score", candidate.get("confidence", 0.5))
        value_inclusion = candidate.get("value_inclusion", 0)
        name_similarity = candidate.get("name_similarity", 0)

        # 높은 value_inclusion + 높은 name_similarity = 강한 FK
        if value_inclusion >= 0.9 and name_similarity >= 0.8:
            return FKValidationResult(
                is_valid_fk=True,
                confidence=min(0.9, score),
                relationship_type=FKRelationshipType.STRONG_FK,
                reasoning="High value inclusion and name similarity",
                evidence_summary={"source": "algorithm_only"},
            )
        elif value_inclusion >= 0.7:
            return FKValidationResult(
                is_valid_fk=True,
                confidence=min(0.7, score),
                relationship_type=FKRelationshipType.WEAK_FK,
                reasoning="Moderate value inclusion",
                evidence_summary={"source": "algorithm_only"},
            )
        else:
            return FKValidationResult(
                is_valid_fk=False,
                confidence=score,
                relationship_type=FKRelationshipType.CORRELATION,
                reasoning="Insufficient evidence for FK relationship",
                evidence_summary={"source": "algorithm_only"},
            )


class CrossPlatformFKDetector:
    """
    Cross-Platform FK 탐지기

    컬럼명이 다르지만 의미론적으로 연결된 FK 관계 탐지
    예: google_campaign_id → campaign_id
    """

    # 알려진 Cross-Platform 패턴
    KNOWN_PATTERNS = [
        # Platform prefix patterns
        (r"^google_(.+)$", r"^\1$"),  # google_xxx → xxx
        (r"^meta_(.+)$", r"^\1$"),    # meta_xxx → xxx
        (r"^naver_(.+)$", r"^\1$"),   # naver_xxx → xxx
        (r"^fb_(.+)$", r"^\1$"),      # fb_xxx → xxx

        # Legacy/Modern naming
        (r"^(.+)_code$", r"^(.+)_id$"),  # xxx_code → xxx_id
        (r"^(.+)_no$", r"^(.+)_id$"),    # xxx_no → xxx_id
        (r"^(.+)_key$", r"^(.+)_id$"),   # xxx_key → xxx_id
    ]

    def __init__(self, llm_client=None):
        self.llm_client = llm_client

    async def detect_cross_platform_fks(
        self,
        tables: Dict[str, Dict[str, Any]],
        sample_data: Dict[str, List[Dict]],
        domain: str = "general",
    ) -> List[Dict[str, Any]]:
        """
        Cross-Platform FK 탐지

        Returns:
            탐지된 Cross-Platform FK 리스트
        """
        candidates = []

        # 1. 패턴 기반 탐지
        pattern_candidates = self._detect_by_patterns(tables)
        candidates.extend(pattern_candidates)

        # 2. LLM 기반 탐지 (LLM이 있는 경우)
        if self.llm_client:
            llm_candidates = await self._detect_with_llm(tables, sample_data, domain)

            # 중복 제거하며 병합
            existing_keys = {(c["from_table"], c["from_column"], c["to_table"], c["to_column"])
                           for c in candidates}
            for c in llm_candidates:
                key = (c["from_table"], c["from_column"], c["to_table"], c["to_column"])
                if key not in existing_keys:
                    candidates.append(c)

        # 3. 값 겹침 검증
        validated = []
        for candidate in candidates:
            if self._validate_by_values(candidate, sample_data):
                validated.append(candidate)

        return validated

    def detect_cross_platform_fks_sync(
        self,
        table_names: List[str],
        tables_columns: Dict[str, List[str]],
    ) -> List[Dict[str, Any]]:
        """
        Cross-Platform FK 탐지 (Sync 버전 - 패턴 기반만)

        Discovery Agent에서 async 없이 호출 가능

        Args:
            table_names: 테이블 이름 리스트
            tables_columns: 테이블별 컬럼 이름 리스트

        Returns:
            탐지된 Cross-Platform FK 리스트
        """
        candidates = []

        # 플랫폼 → 테이블 매핑 (플랫폼별 primary 테이블)
        PLATFORM_PRIMARY_TABLES = {
            "google": ["google_ads_campaigns", "google_ads_ad_groups", "google_ads_ads"],
            "meta": ["meta_ads_campaigns", "meta_ads_adsets", "meta_ads_ads"],
            "naver": ["naver_ads_campaigns", "naver_ads_adgroups", "naver_ads_ads"],
            "fb": ["facebook_campaigns", "fb_campaigns", "meta_ads_campaigns"],
        }

        # 모든 컬럼 수집
        all_columns = {}  # {column_name: [table_name, ...]}
        for table_name in table_names:
            for col_name in tables_columns.get(table_name, []):
                if col_name not in all_columns:
                    all_columns[col_name] = []
                all_columns[col_name].append(table_name)

        # 패턴 매칭
        for from_col_name in all_columns:
            for pattern_from, pattern_to in self.KNOWN_PATTERNS:
                match = re.match(pattern_from, from_col_name.lower())
                if match:
                    # 플랫폼 추출 (예: google_campaign_id → google)
                    platform_prefix = None
                    if match.groups():
                        # 예: google_campaign_id에서 campaign_id 추출하면 google이 prefix
                        full_match = match.group(0)
                        extracted = match.group(1)
                        if full_match.startswith("google_"):
                            platform_prefix = "google"
                        elif full_match.startswith("meta_"):
                            platform_prefix = "meta"
                        elif full_match.startswith("naver_"):
                            platform_prefix = "naver"
                        elif full_match.startswith("fb_"):
                            platform_prefix = "fb"

                    # 매칭되는 타겟 컬럼 찾기
                    expected_target = re.sub(pattern_from, pattern_to, from_col_name.lower())

                    # 유사한 이름의 컬럼 찾기
                    for to_col_name in all_columns:
                        if re.match(expected_target, to_col_name.lower()):
                            # 서로 다른 테이블인 경우만
                            for from_table in all_columns[from_col_name]:
                                for to_table in all_columns[to_col_name]:
                                    if from_table != to_table:
                                        # 플랫폼 기반 필터링 적용
                                        confidence = 0.6
                                        if platform_prefix:
                                            preferred_tables = PLATFORM_PRIMARY_TABLES.get(platform_prefix, [])
                                            if any(pref.lower() in to_table.lower() for pref in preferred_tables):
                                                confidence = 0.85  # 플랫폼 매칭 시 높은 신뢰도
                                            else:
                                                confidence = 0.3  # 다른 플랫폼은 낮은 신뢰도

                                        candidates.append({
                                            "from_table": from_table,
                                            "from_column": from_col_name,
                                            "to_table": to_table,
                                            "to_column": to_col_name,
                                            "detection_method": "cross_platform_pattern",
                                            "pattern": f"{pattern_from} → {pattern_to}",
                                            "confidence": confidence,
                                            "platform_hint": platform_prefix,
                                        })

        # 중복 제거
        seen = set()
        unique_candidates = []
        for c in candidates:
            key = (c["from_table"], c["from_column"], c["to_table"], c["to_column"])
            if key not in seen:
                seen.add(key)
                unique_candidates.append(c)

        # 신뢰도 기준으로 정렬
        unique_candidates.sort(key=lambda x: x.get("confidence", 0), reverse=True)

        return unique_candidates

    def _detect_by_patterns(self, tables: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
        """패턴 기반 Cross-Platform FK 탐지"""
        candidates = []

        # 모든 컬럼 수집
        all_columns = {}  # {column_name: [(table_name, column_info), ...]}
        for table_name, table_info in tables.items():
            for col in table_info.get("columns", []):
                col_name = col.get("name", "")
                if col_name not in all_columns:
                    all_columns[col_name] = []
                all_columns[col_name].append((table_name, col))

        # 패턴 매칭
        for from_col_name in all_columns:
            for pattern_from, pattern_to in self.KNOWN_PATTERNS:
                match = re.match(pattern_from, from_col_name.lower())
                if match:
                    # 매칭되는 타겟 컬럼 찾기
                    expected_target = re.sub(pattern_from, pattern_to, from_col_name.lower())

                    # 유사한 이름의 컬럼 찾기
                    for to_col_name in all_columns:
                        if re.match(expected_target, to_col_name.lower()):
                            # 서로 다른 테이블인 경우만
                            for from_table, from_col_info in all_columns[from_col_name]:
                                for to_table, to_col_info in all_columns[to_col_name]:
                                    if from_table != to_table:
                                        candidates.append({
                                            "from_table": from_table,
                                            "from_column": from_col_name,
                                            "to_table": to_table,
                                            "to_column": to_col_name,
                                            "detection_method": "pattern",
                                            "pattern": f"{pattern_from} → {pattern_to}",
                                            "confidence": 0.7,
                                        })

        return candidates

    async def _detect_with_llm(
        self,
        tables: Dict[str, Dict[str, Any]],
        sample_data: Dict[str, List[Dict]],
        domain: str,
    ) -> List[Dict[str, Any]]:
        """LLM을 사용한 Cross-Platform FK 탐지"""

        # 테이블 요약 생성
        table_summaries = []
        for table_name, table_info in tables.items():
            cols = [c.get("name", "") for c in table_info.get("columns", [])]
            sample = sample_data.get(table_name, [])[:2]
            table_summaries.append(f"- {table_name}: columns={cols}, sample={sample}")

        prompt = f"""## Cross-Platform FK Detection

Domain: {domain}

### Available Tables
{chr(10).join(table_summaries)}

### Task
Identify Foreign Key relationships where column names are DIFFERENT but semantically related.

Focus on:
1. Platform-prefixed columns (google_xxx, meta_xxx, naver_xxx) → base columns
2. Different naming conventions (xxx_code, xxx_no) → (xxx_id)
3. Legacy vs modern naming patterns

### Response Format (JSON only)
```json
{{
    "cross_platform_fks": [
        {{
            "from_table": "table_name",
            "from_column": "column_name",
            "to_table": "target_table",
            "to_column": "target_column",
            "confidence": 0.0 to 1.0,
            "reasoning": "brief explanation"
        }}
    ]
}}
```
"""

        try:
            response = await self.llm_client.generate(prompt, temperature=0.1)

            json_match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group(1))
            else:
                data = json.loads(response)

            candidates = data.get("cross_platform_fks", [])
            for c in candidates:
                c["detection_method"] = "llm"

            return candidates

        except Exception as e:
            logger.warning(f"LLM cross-platform detection failed: {e}")
            return []

    def _validate_by_values(
        self,
        candidate: Dict[str, Any],
        sample_data: Dict[str, List[Dict]],
    ) -> bool:
        """값 겹침으로 검증"""
        from_table = candidate["from_table"]
        from_column = candidate["from_column"]
        to_table = candidate["to_table"]
        to_column = candidate["to_column"]

        from_data = sample_data.get(from_table, [])
        to_data = sample_data.get(to_table, [])

        if not from_data or not to_data:
            return True  # 데이터 없으면 패스

        from_values = set()
        for row in from_data:
            if from_column in row and row[from_column] is not None:
                from_values.add(str(row[from_column]))

        to_values = set()
        for row in to_data:
            if to_column in row and row[to_column] is not None:
                to_values.add(str(row[to_column]))

        if not from_values or not to_values:
            return True

        # 최소 30% 겹침 필요
        overlap = len(from_values & to_values)
        inclusion = overlap / len(from_values) if from_values else 0

        candidate["value_inclusion"] = inclusion

        return inclusion >= 0.3
