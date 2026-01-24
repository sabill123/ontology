"""
Column Normalizer - 컬럼명 정규화 및 의미적 매칭 (v1.0)

컬럼명의 약어를 확장하고, 토큰화하여 의미적으로 동일한 컬럼을 찾습니다.
예: flt_no ↔ flight_number, carr_cd ↔ carrier_code

사용 예:
    normalizer = ColumnNormalizer()
    result = normalizer.normalize("flt_no")
    # NormalizedColumn(original="flt_no", normalized="flight_number", ...)

    similarity = normalizer.compute_semantic_similarity(col_a, col_b)
    # 0.95 (높을수록 유사)
"""

import re
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Set, Tuple
from difflib import SequenceMatcher


@dataclass
class NormalizedColumn:
    """정규화된 컬럼 정보"""
    original: str                           # 원본 컬럼명
    normalized: str                         # 정규화된 컬럼명
    tokens: List[str]                       # 토큰 목록
    semantic_entity: Optional[str] = None   # 참조하는 엔티티 (있는 경우)
    is_fk_pattern: bool = False             # FK 패턴 여부 (_id, _code 등)
    abbreviations_expanded: Dict[str, str] = field(default_factory=dict)  # 확장된 약어


@dataclass
class ColumnMatch:
    """컬럼 매칭 결과"""
    column_a: str
    column_b: str
    normalized_a: NormalizedColumn
    normalized_b: NormalizedColumn
    similarity: float
    match_type: str  # "exact", "abbreviation", "semantic", "partial"


class ColumnNormalizer:
    """
    컬럼명 정규화 및 의미적 매칭

    기능:
    1. 약어 확장 (flt → flight)
    2. 케이스 정규화 (camelCase → snake_case)
    3. 토큰 분리 (flight_number → [flight, number])
    4. 의미적 유사도 계산
    """

    # 범용 약어 매핑 (도메인 무관)
    ABBREVIATIONS = {
        # === 항공/공항 ===
        "flt": "flight",
        "pax": "passenger",
        "apt": "airport",
        "arr": "arrival",
        "dep": "departure",
        "sched": "scheduled",
        "orig": "origin",
        "dest": "destination",
        "carr": "carrier",
        "aln": "airline",
        "acft": "aircraft",
        "dly": "delay",
        "gate": "gate",
        "term": "terminal",
        "rwy": "runway",

        # === 물류/운송 ===
        "shp": "shipment",
        "dlvr": "delivery",
        "wh": "warehouse",
        "veh": "vehicle",
        "trk": "truck",
        "rte": "route",
        "eta": "estimated_arrival",
        "etd": "estimated_departure",
        "pkg": "package",
        "frt": "freight",
        "inv": "inventory",

        # === 일반 비즈니스 ===
        "cust": "customer",
        "custs": "customers",
        "prod": "product",
        "prods": "products",
        "acct": "account",
        "txn": "transaction",
        "emp": "employee",
        "dept": "department",
        "org": "organization",
        "cat": "category",
        "qty": "quantity",
        "amt": "amount",
        "prc": "price",

        # === 상태/메타데이터 ===
        "stat": "status",
        "sts": "status",
        "rpt": "report",
        "dmg": "damage",
        "maint": "maintenance",
        "cfg": "configuration",
        "conf": "configuration",
        "ref": "reference",
        "seq": "sequence",
        "ver": "version",
        "rev": "revision",
        "msg": "message",
        "req": "request",
        "res": "response",
        "resp": "response",

        # === 식별자 관련 ===
        "id": "identifier",
        "num": "number",
        "no": "number",
        "cd": "code",
        "typ": "type",
        "nm": "name",
        "desc": "description",
        "lbl": "label",
        "idx": "index",
        "key": "key",

        # === 수치/통계 ===
        "cnt": "count",
        "avg": "average",
        "min": "minimum",
        "max": "maximum",
        "tot": "total",
        "sum": "sum",
        "pct": "percent",
        "val": "value",
        "flg": "flag",
        "ind": "indicator",

        # === 시간 관련 ===
        "dt": "date",
        "tm": "time",
        "ts": "timestamp",
        "yr": "year",
        "mo": "month",
        "dy": "day",
        "hr": "hour",
        "mn": "minute",
        "sec": "second",
        "dur": "duration",

        # === 위치/주소 ===
        "addr": "address",
        "loc": "location",
        "lat": "latitude",
        "lng": "longitude",
        "lon": "longitude",
        "cty": "city",
        "st": "state",
        "ctry": "country",
        "rgn": "region",
        "zip": "zipcode",
        "pst": "postal",

        # === 기타 ===
        "src": "source",
        "tgt": "target",
        "dst": "destination",
        "lvl": "level",
        "grp": "group",
        "cls": "class",
        "prty": "priority",
        "attr": "attribute",
        "prop": "property",
        "param": "parameter",
    }

    # FK 패턴 (엔티티 참조를 나타내는 접미사)
    FK_SUFFIXES = ["_id", "_code", "_cd", "_no", "_num", "_key", "_ref", "_fk"]

    # 무시할 접두사/접미사
    IGNORABLE_PREFIXES = ["tbl_", "t_", "tb_", "dim_", "fact_", "stg_", "raw_", "src_", "dw_"]
    IGNORABLE_SUFFIXES = ["_tbl", "_table", "_dim", "_fact"]

    def __init__(self):
        # 역방향 매핑 생성 (flight → flt)
        self.reverse_abbreviations = {v: k for k, v in self.ABBREVIATIONS.items()}

    def normalize(self, column_name: str) -> NormalizedColumn:
        """
        컬럼명 정규화

        Args:
            column_name: 원본 컬럼명

        Returns:
            NormalizedColumn: 정규화된 컬럼 정보
        """
        original = column_name

        # 1. 소문자 변환
        name = column_name.lower()

        # 2. camelCase → snake_case
        name = self._camel_to_snake(name)

        # 3. 무시할 접두사/접미사 제거
        name = self._remove_ignorable_affixes(name)

        # 4. 토큰 분리
        tokens = self._tokenize(name)

        # 5. 약어 확장
        expanded_tokens, abbreviations_expanded = self._expand_abbreviations(tokens)

        # 6. 정규화된 이름 생성
        normalized = "_".join(expanded_tokens)

        # 7. FK 패턴 검사
        is_fk_pattern, semantic_entity = self._detect_fk_pattern(normalized, tokens)

        return NormalizedColumn(
            original=original,
            normalized=normalized,
            tokens=expanded_tokens,
            semantic_entity=semantic_entity,
            is_fk_pattern=is_fk_pattern,
            abbreviations_expanded=abbreviations_expanded,
        )

    def compute_semantic_similarity(
        self,
        col_a: NormalizedColumn,
        col_b: NormalizedColumn
    ) -> float:
        """
        두 컬럼 간 의미적 유사도 계산 (0.0 ~ 1.0)

        Args:
            col_a: 첫 번째 정규화된 컬럼
            col_b: 두 번째 정규화된 컬럼

        Returns:
            float: 유사도 (0.0 ~ 1.0)
        """
        # 1. 정규화된 이름이 완전히 같으면 1.0
        if col_a.normalized == col_b.normalized:
            return 1.0

        # 2. 토큰 기반 Jaccard 유사도
        tokens_a = set(col_a.tokens)
        tokens_b = set(col_b.tokens)

        if not tokens_a or not tokens_b:
            return 0.0

        intersection = tokens_a & tokens_b
        union = tokens_a | tokens_b
        jaccard = len(intersection) / len(union) if union else 0.0

        # 3. 순서 고려한 시퀀스 매칭
        sequence_ratio = SequenceMatcher(
            None, col_a.normalized, col_b.normalized
        ).ratio()

        # 4. 엔티티 참조가 같으면 보너스
        entity_bonus = 0.0
        if col_a.semantic_entity and col_a.semantic_entity == col_b.semantic_entity:
            entity_bonus = 0.2

        # 5. 가중 평균
        similarity = (jaccard * 0.5) + (sequence_ratio * 0.3) + entity_bonus

        # 6. FK 패턴 매칭 보너스
        if col_a.is_fk_pattern and col_b.is_fk_pattern:
            # 둘 다 FK 패턴이고 토큰이 겹치면 추가 보너스
            if intersection:
                similarity += 0.1

        return min(similarity, 1.0)

    def find_matching_columns(
        self,
        columns_a: List[str],
        columns_b: List[str],
        threshold: float = 0.7
    ) -> List[ColumnMatch]:
        """
        두 컬럼 목록에서 매칭되는 쌍 찾기

        Args:
            columns_a: 첫 번째 테이블의 컬럼 목록
            columns_b: 두 번째 테이블의 컬럼 목록
            threshold: 최소 유사도 임계값

        Returns:
            List[ColumnMatch]: 매칭된 컬럼 쌍 목록
        """
        matches = []

        # 정규화
        normalized_a = {col: self.normalize(col) for col in columns_a}
        normalized_b = {col: self.normalize(col) for col in columns_b}

        # 모든 쌍에 대해 유사도 계산
        for col_a, norm_a in normalized_a.items():
            for col_b, norm_b in normalized_b.items():
                similarity = self.compute_semantic_similarity(norm_a, norm_b)

                if similarity >= threshold:
                    # 매치 타입 결정
                    match_type = self._determine_match_type(norm_a, norm_b, similarity)

                    matches.append(ColumnMatch(
                        column_a=col_a,
                        column_b=col_b,
                        normalized_a=norm_a,
                        normalized_b=norm_b,
                        similarity=similarity,
                        match_type=match_type,
                    ))

        # 유사도 높은 순으로 정렬
        matches.sort(key=lambda m: m.similarity, reverse=True)

        return matches

    def generate_llm_context(
        self,
        matches: List[ColumnMatch],
        table_a: str,
        table_b: str
    ) -> str:
        """
        LLM에게 전달할 컬럼 매칭 컨텍스트 생성

        Args:
            matches: 매칭된 컬럼 쌍 목록
            table_a: 첫 번째 테이블명
            table_b: 두 번째 테이블명

        Returns:
            str: LLM 프롬프트용 컨텍스트
        """
        if not matches:
            return f"No semantic column matches found between {table_a} and {table_b}."

        lines = [
            f"## Semantic Column Matches: {table_a} ↔ {table_b}",
            "",
        ]

        for match in matches[:10]:  # 상위 10개만
            abbrev_info = ""
            if match.normalized_a.abbreviations_expanded:
                abbrev_info = f" (expanded: {match.normalized_a.abbreviations_expanded})"

            lines.append(
                f"- {table_a}.{match.column_a} ↔ {table_b}.{match.column_b} "
                f"(similarity: {match.similarity:.2f}, type: {match.match_type}){abbrev_info}"
            )

        return "\n".join(lines)

    # === Private Methods ===

    def _camel_to_snake(self, name: str) -> str:
        """camelCase를 snake_case로 변환"""
        # 대문자 앞에 _ 삽입
        s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
        return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()

    def _remove_ignorable_affixes(self, name: str) -> str:
        """무시할 접두사/접미사 제거"""
        for prefix in self.IGNORABLE_PREFIXES:
            if name.startswith(prefix):
                name = name[len(prefix):]
                break

        for suffix in self.IGNORABLE_SUFFIXES:
            if name.endswith(suffix):
                name = name[:-len(suffix)]
                break

        return name

    def _tokenize(self, name: str) -> List[str]:
        """컬럼명을 토큰으로 분리"""
        # 구분자로 분리 (_, -, .)
        tokens = re.split(r'[_\-\.]+', name)
        # 빈 토큰 제거
        return [t for t in tokens if t]

    def _expand_abbreviations(self, tokens: List[str]) -> Tuple[List[str], Dict[str, str]]:
        """약어를 전체 단어로 확장"""
        expanded = []
        expansions = {}

        for token in tokens:
            if token in self.ABBREVIATIONS:
                expanded_token = self.ABBREVIATIONS[token]
                expanded.append(expanded_token)
                expansions[token] = expanded_token
            else:
                expanded.append(token)

        return expanded, expansions

    def _detect_fk_pattern(
        self,
        normalized: str,
        tokens: List[str]
    ) -> Tuple[bool, Optional[str]]:
        """FK 패턴 감지 및 참조 엔티티 추출"""
        is_fk = False
        entity = None

        for suffix in self.FK_SUFFIXES:
            if normalized.endswith(suffix):
                is_fk = True
                # 엔티티 이름 추출 (예: customer_id → customer)
                entity_part = normalized[:-len(suffix)]
                if entity_part:
                    entity = entity_part
                break

        return is_fk, entity

    def _determine_match_type(
        self,
        norm_a: NormalizedColumn,
        norm_b: NormalizedColumn,
        similarity: float
    ) -> str:
        """매치 타입 결정"""
        if norm_a.normalized == norm_b.normalized:
            return "exact"

        if norm_a.abbreviations_expanded or norm_b.abbreviations_expanded:
            # 약어 확장 후 일치하면
            if norm_a.normalized == norm_b.normalized:
                return "abbreviation"
            # 약어 확장이 포함된 유사 매칭
            return "abbreviation_semantic"

        if similarity >= 0.9:
            return "semantic_high"
        elif similarity >= 0.7:
            return "semantic_medium"
        else:
            return "partial"


# === Convenience Functions ===

def normalize_column(column_name: str) -> NormalizedColumn:
    """컬럼명 정규화 (편의 함수)"""
    return ColumnNormalizer().normalize(column_name)


def find_column_matches(
    columns_a: List[str],
    columns_b: List[str],
    threshold: float = 0.7
) -> List[ColumnMatch]:
    """컬럼 매칭 찾기 (편의 함수)"""
    return ColumnNormalizer().find_matching_columns(columns_a, columns_b, threshold)
