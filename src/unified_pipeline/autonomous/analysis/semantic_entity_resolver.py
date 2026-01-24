# analysis/semantic_entity_resolver.py
"""
Solution 3: Semantic Entity Resolver

컬럼명/테이블명에서 entity를 추출하고
synonym 관계를 통해 FK 관계 추론
"""

from typing import Dict, List, Set, Tuple
import re
import logging

logger = logging.getLogger(__name__)


class SemanticEntityResolver:
    """
    의미론적 Entity 해석기

    컬럼명/테이블명에서 entity를 추출하고
    synonym 관계를 통해 FK 관계 추론
    """

    # Domain-agnostic synonym groups
    UNIVERSAL_SYNONYMS = {
        "person": {"customer", "buyer", "user", "client", "member", "person",
                   "cust", "usr", "consumer", "subscriber", "owner", "creator"},
        "product": {"product", "item", "goods", "sku", "article", "prod",
                    "merchandise", "commodity"},
        "order": {"order", "purchase", "transaction", "txn", "sale", "booking"},
        "organization": {"company", "org", "organization", "vendor", "supplier",
                        "merchant", "business", "firm", "enterprise"},
        "location": {"location", "loc", "address", "place", "site", "region",
                     "area", "zone", "territory"},
        "time": {"date", "time", "datetime", "timestamp", "created", "updated",
                 "modified", "when"},
        "identifier": {"id", "code", "key", "no", "num", "number", "ref",
                       "reference", "pk", "fk"},
        "campaign": {"campaign", "cmp", "promotion", "marketing",
                     "advertisement", "promo"},
        "ad_campaign": {"ad_campaign", "ad_campaigns", "advertisement_campaign",
                        "ad_cmp", "advertising_campaign"},
        "event": {"event", "activity", "action", "interaction", "occurrence"},
        "session": {"session", "visit", "browsing", "web_session"},
        "lead": {"lead", "prospect", "potential", "opportunity"},
        "email": {"email", "mail", "message", "notification", "send"},
        "account": {"account", "acct", "profile", "registration"},
    }

    # Common abbreviation expansions
    ABBREVIATIONS = {
        "cust": "customer",
        "prod": "product",
        "txn": "transaction",
        "cmp": "campaign",
        "emp": "employee",
        "dept": "department",
        "org": "organization",
        "loc": "location",
        "acct": "account",
        "inv": "invoice",
        "qty": "quantity",
        "amt": "amount",
        "no": "number",
        "num": "number",
        "ref": "reference",
        "attr": "attribute",
        "desc": "description",
        "addr": "address",
        "msg": "message",
        "usr": "user",
        "sess": "session",
        "req": "request",
        "resp": "response",
        "src": "source",
        "dst": "destination",
        "tgt": "target",
        # Additional abbreviations for FK detection
        "ad": "advertisement",
        "conv": "conversion",
        "perf": "performance",
        "attrib": "attributed",
        "mkt": "marketing",
        "promo": "promotion",
    }

    # Compound entity patterns (for FK columns like attributed_cmp_id)
    # Maps compound prefix to the target entity for FK relationship
    COMPOUND_ENTITY_PATTERNS = {
        "attributed_cmp": "campaign",
        "attributed_ad": "ad_campaign",
        "attributed_campaign": "campaign",
        "marketing_campaign": "campaign",
        "ad_cmp": "ad_campaign",
        "order_ref": "order",
    }

    # Suffix patterns to remove
    SUFFIXES = ['_id', '_code', '_key', '_no', '_num', '_ref', '_fk', '_pk']

    def extract_entity(self, name: str) -> Tuple[str, float]:
        """
        이름에서 entity 추출

        Returns:
            (entity_name, confidence)
        """
        # 1. 정규화
        normalized = self._normalize(name)

        # 2. Suffix 제거
        entity = self._remove_suffixes(normalized)

        # 3. Compound pattern 체크 (attributed_cmp -> campaign)
        compound_entity = self._check_compound_pattern(entity)
        if compound_entity:
            return (compound_entity, 0.85)  # compound pattern matched

        # 4. 약어 확장 (단일 단어)
        expanded = self._expand_abbreviation(entity)

        # 5. 약어 확장 (복합 단어 내 약어)
        if expanded == entity and '_' in entity:
            expanded = self._expand_compound_abbreviations(entity)

        # 6. Confidence 계산
        # - 변환이 많을수록 불확실성 증가
        confidence = 1.0
        if entity != normalized:
            confidence -= 0.1  # suffix 제거됨
        if expanded != entity:
            confidence -= 0.1  # 약어 확장됨

        return (expanded, max(confidence, 0.7))

    def are_semantically_related(
        self,
        entity_a: str,
        entity_b: str,
    ) -> Tuple[bool, float, str]:
        """
        두 entity가 의미적으로 관련있는지 판단

        Returns:
            (is_related, confidence, relation_type)
        """
        a_normalized = self._expand_abbreviation(entity_a.lower())
        b_normalized = self._expand_abbreviation(entity_b.lower())

        # 1. 동일
        if a_normalized == b_normalized:
            return (True, 1.0, "identical")

        # 2. 복수형/단수형 먼저 체크 (정규화 용도)
        a_singular = self._to_singular(a_normalized)
        b_singular = self._to_singular(b_normalized)

        if a_singular == b_singular:
            return (True, 0.95, "plural_singular")

        # 3. Synonym 그룹 체크 (단수형으로 비교)
        for group_name, synonyms in self.UNIVERSAL_SYNONYMS.items():
            a_in = a_singular in synonyms or a_normalized in synonyms
            b_in = b_singular in synonyms or b_normalized in synonyms
            if a_in and b_in:
                return (True, 0.9, f"synonym:{group_name}")

        # 4. 부분 일치 (customer_id → customer, 단 최소 길이 제한)
        if len(a_singular) > 3 and len(b_singular) > 3:
            if a_singular in b_singular or b_singular in a_singular:
                return (True, 0.7, "partial_match")
            # underscore 포함 복합어 부분 일치
            if '_' in a_singular or '_' in b_singular:
                a_parts = set(a_singular.split('_'))
                b_parts = set(b_singular.split('_'))
                if a_parts & b_parts:  # 공통 부분이 있으면
                    return (True, 0.65, "partial_match")

        # 5. Fuzzy matching (Levenshtein-like)
        similarity = self._string_similarity(a_singular, b_singular)
        if similarity > 0.8:
            return (True, similarity * 0.8, "fuzzy_match")

        return (False, 0.0, "none")

    def infer_fk_relationship(
        self,
        fk_column: str,
        fk_table: str,
        pk_column: str,
        pk_table: str,
    ) -> Dict:
        """
        FK 관계 의미론적 추론
        """
        # FK 컬럼에서 entity 추출
        fk_entity, fk_conf = self.extract_entity(fk_column)

        # PK 테이블에서 entity 추출
        pk_entity, pk_conf = self.extract_entity(pk_table)

        # PK 컬럼에서도 entity 추출
        pk_col_entity, pk_col_conf = self.extract_entity(pk_column)

        # 의미적 관련성 체크 (FK 컬럼 vs PK 테이블)
        is_related_table, rel_conf_table, rel_type_table = self.are_semantically_related(
            fk_entity, pk_entity
        )

        # 의미적 관련성 체크 (FK 컬럼 vs PK 컬럼)
        is_related_col, rel_conf_col, rel_type_col = self.are_semantically_related(
            fk_entity, pk_col_entity
        )

        # 종합
        is_related = is_related_table or is_related_col
        rel_conf = max(rel_conf_table, rel_conf_col)
        rel_type = rel_type_table if rel_conf_table >= rel_conf_col else rel_type_col

        # 종합 점수
        semantic_score = 0.0
        if is_related:
            semantic_score = (fk_conf + pk_conf + rel_conf) / 3

        return {
            "fk_entity": fk_entity,
            "pk_entity": pk_entity,
            "pk_col_entity": pk_col_entity,
            "is_semantically_related": is_related,
            "relation_type": rel_type,
            "semantic_confidence": semantic_score,
            "reasoning": self._generate_reasoning(
                fk_column, fk_entity, pk_table, pk_entity, rel_type
            ),
            "details": {
                "fk_confidence": fk_conf,
                "pk_confidence": pk_conf,
                "relation_confidence": rel_conf,
                "table_relation": rel_type_table,
                "column_relation": rel_type_col,
            }
        }

    def find_potential_references(
        self,
        column_name: str,
        all_tables: List[str],
    ) -> List[Tuple[str, float, str]]:
        """
        컬럼이 참조할 수 있는 잠재적 테이블 찾기

        Returns:
            List of (table_name, confidence, relation_type)
        """
        entity, _ = self.extract_entity(column_name)
        matches = []

        for table in all_tables:
            table_entity, _ = self.extract_entity(table)
            is_related, conf, rel_type = self.are_semantically_related(entity, table_entity)
            if is_related:
                matches.append((table, conf, rel_type))

        # confidence 순 정렬
        return sorted(matches, key=lambda x: x[1], reverse=True)

    def _normalize(self, name: str) -> str:
        """이름 정규화"""
        # camelCase → snake_case
        name = re.sub(r'([a-z])([A-Z])', r'\1_\2', name)
        return name.lower()

    def _remove_suffixes(self, name: str) -> str:
        """ID suffix 제거"""
        for suffix in self.SUFFIXES:
            if name.endswith(suffix):
                return name[:-len(suffix)]
        return name

    def _expand_abbreviation(self, word: str) -> str:
        """약어 확장"""
        return self.ABBREVIATIONS.get(word, word)

    def _check_compound_pattern(self, entity: str) -> str:
        """
        Compound pattern 체크

        attributed_cmp -> campaign
        ad_cmp -> ad_campaign
        """
        # 정확히 일치하는 compound pattern 체크
        if entity in self.COMPOUND_ENTITY_PATTERNS:
            return self.COMPOUND_ENTITY_PATTERNS[entity]

        # prefix로 시작하는 compound pattern 체크
        for pattern, target in self.COMPOUND_ENTITY_PATTERNS.items():
            if entity.startswith(pattern + "_") or entity == pattern:
                return target

        return ""

    def _expand_compound_abbreviations(self, entity: str) -> str:
        """
        복합 단어 내 약어 확장

        attributed_cmp -> attributed_campaign

        Note: 'ad' is NOT expanded when part of semantic compound like 'ad_campaign'
        """
        # Preserve known semantic compounds
        PRESERVE_PATTERNS = {'ad_campaign', 'ad_campaigns', 'ad_performance'}
        if entity in PRESERVE_PATTERNS:
            return entity

        parts = entity.split('_')
        expanded_parts = []

        for i, part in enumerate(parts):
            # Don't expand 'ad' if followed by campaign-related words
            if part == 'ad' and i + 1 < len(parts):
                next_part = parts[i + 1]
                if next_part in {'campaign', 'campaigns', 'cmp', 'performance'}:
                    expanded_parts.append(part)  # Keep as 'ad'
                    continue

            expanded = self.ABBREVIATIONS.get(part, part)
            expanded_parts.append(expanded)

        return '_'.join(expanded_parts)

    def _is_plural_match(self, a: str, b: str) -> bool:
        """복수형/단수형 일치"""
        if a + 's' == b or b + 's' == a:
            return True
        if a + 'es' == b or b + 'es' == a:
            return True
        if a.rstrip('s') == b or b.rstrip('s') == a:
            return True
        # ies -> y
        if a.endswith('ies') and a[:-3] + 'y' == b:
            return True
        if b.endswith('ies') and b[:-3] + 'y' == a:
            return True
        return False

    def _to_singular(self, word: str) -> str:
        """복수형을 단수형으로 변환"""
        if not word:
            return word
        # ies -> y (categories -> category)
        if word.endswith('ies') and len(word) > 4:
            return word[:-3] + 'y'
        # es -> (boxes -> box)
        if word.endswith('es') and len(word) > 3:
            return word[:-2]
        # s -> (customers -> customer)
        if word.endswith('s') and len(word) > 2:
            return word[:-1]
        return word

    def _string_similarity(self, a: str, b: str) -> float:
        """문자열 유사도 계산 (간단한 Jaccard)"""
        if not a or not b:
            return 0.0

        # Character-level Jaccard
        set_a = set(a)
        set_b = set(b)
        intersection = len(set_a & set_b)
        union = len(set_a | set_b)

        return intersection / union if union > 0 else 0.0

    def _generate_reasoning(
        self, fk_col, fk_entity, pk_table, pk_entity, rel_type
    ) -> str:
        """추론 근거 생성"""
        if rel_type == "identical":
            return f"'{fk_col}' references '{pk_table}' (entity '{fk_entity}' matches exactly)"
        elif rel_type.startswith("synonym"):
            group = rel_type.split(":")[1] if ":" in rel_type else "unknown"
            return f"'{fk_col}' ({fk_entity}) is synonym of '{pk_table}' ({pk_entity}) in group '{group}'"
        elif rel_type == "partial_match":
            return f"'{fk_col}' ({fk_entity}) partially matches '{pk_table}' ({pk_entity})"
        elif rel_type == "plural_singular":
            return f"'{fk_entity}' is singular/plural form of '{pk_entity}'"
        elif rel_type == "fuzzy_match":
            return f"'{fk_entity}' fuzzy matches '{pk_entity}'"
        return "No semantic relationship detected"


# Utility function for quick semantic check
def quick_semantic_check(fk_col: str, pk_table: str) -> Tuple[bool, float]:
    """
    빠른 의미적 관계 체크

    Returns:
        (is_related, confidence)
    """
    resolver = SemanticEntityResolver()
    result = resolver.infer_fk_relationship(
        fk_column=fk_col,
        fk_table="",  # not used for quick check
        pk_column="id",
        pk_table=pk_table,
    )
    return (result["is_semantically_related"], result["semantic_confidence"])
