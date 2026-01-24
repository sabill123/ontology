"""
Universal FK Detection Algorithm (v7.4)

도메인 무관 범용 FK 탐지 알고리즘
- 어떤 네이밍 컨벤션이든 처리 가능
- 값 기반 검증으로 실제 관계 확인
- LLM이 최종 판단

핵심 원리:
1. 네이밍 정규화: 다양한 컨벤션을 통일된 형태로 변환
2. 패턴 탐지: 도메인 무관 FK 패턴 인식
3. 값 기반 검증: Jaccard, Containment로 실제 관계 확인
4. 계층 구조 추론: 공통 prefix/suffix로 그룹 탐지

v7.4 변경사항 (Containment 강화):
- Jaccard 낮아도 Containment 95%+ 면 FK 후보로 포함
- 컬럼명이 달라도 값 기반으로 FK 관계 탐지 (예: airline_code → iata_code)
- Containment 가중치 대폭 상향 (20% → 40%)
- detection_reason 필드 추가로 탐지 방법 추적
"""

import re
import logging
from typing import Dict, List, Any, Set, Tuple, Optional
from dataclasses import dataclass, field
from collections import defaultdict

logger = logging.getLogger(__name__)


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class NormalizedName:
    """정규화된 이름"""
    original: str
    normalized: str  # snake_case로 통일
    tokens: List[str]  # 분리된 토큰들
    prefix: Optional[str] = None  # 공통 prefix (예: google_ads, tbl, T)
    entity: Optional[str] = None  # 엔티티명 (예: campaigns, products)
    suffix: Optional[str] = None  # suffix (예: _id, _code)


@dataclass
class FKCandidate:
    """FK 후보"""
    from_table: str
    from_column: str
    to_table: str
    to_column: str

    # 알고리즘 계산 결과 (LLM 판단을 위한 데이터)
    jaccard_similarity: float = 0.0
    containment_fk_in_pk: float = 0.0  # FK 값이 PK에 포함된 비율
    containment_pk_in_fk: float = 0.0  # PK 값이 FK에 포함된 비율

    # 카디널리티 분석
    fk_unique_ratio: float = 0.0  # FK 컬럼의 고유값 비율 (낮을수록 FK 가능성)
    pk_unique_ratio: float = 0.0  # PK 컬럼의 고유값 비율 (높을수록 PK 가능성)

    # 네이밍 분석
    name_similarity: float = 0.0  # 정규화된 이름 유사도
    same_normalized_name: bool = False  # 정규화 후 같은 이름인지

    # 구조 분석
    same_prefix_group: bool = False  # 같은 prefix 그룹인지
    hierarchy_pattern: Optional[str] = None  # 계층 패턴 (예: "parent->child")

    # 샘플 데이터
    sample_overlapping_values: List[str] = field(default_factory=list)
    sample_fk_only_values: List[str] = field(default_factory=list)  # FK에만 있는 값

    # 메타데이터
    detection_method: str = ""  # 탐지 방법
    confidence_factors: Dict[str, float] = field(default_factory=dict)


@dataclass
class TableGroup:
    """테이블 그룹 (공통 prefix/도메인)"""
    prefix: str
    tables: List[str]
    hierarchy: Dict[str, List[str]] = field(default_factory=dict)  # parent -> children


# =============================================================================
# Naming Normalizer
# =============================================================================

class NamingNormalizer:
    """
    네이밍 정규화기

    다양한 네이밍 컨벤션을 통일된 형태로 변환:
    - PascalCase, camelCase → snake_case
    - 접두사 분리: tbl_product → (tbl, product)
    - 축약어 확장: prod → product, acct → account
    """

    # 공통 테이블 접두사 (제거 대상)
    TABLE_PREFIXES = [
        "tbl_", "t_", "tb_", "table_",
        "vw_", "view_",
        "dim_", "fact_",
        "stg_", "staging_",
        "raw_", "src_",
        "dw_", "dwh_",
        "mes_", "erp_", "crm_",
    ]

    # 축약어 매핑 (v7.0: 대폭 확장)
    ABBREVIATIONS = {
        # === 항공/공항 (Aviation/Airport) ===
        "flt": "flight",
        "flts": "flights",
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
        "iata": "iata",
        "icao": "icao",

        # === 물류/운송 (Logistics/Transport) ===
        "shp": "shipment",
        "shpmt": "shipment",
        "dlvr": "delivery",
        "wh": "warehouse",
        "veh": "vehicle",
        "trk": "truck",
        "rte": "route",
        "eta": "estimated_arrival",
        "etd": "estimated_departure",
        "pkg": "package",
        "frt": "freight",

        # === 일반 비즈니스 ===
        "prod": "product",
        "prods": "products",
        "cust": "customer",
        "custs": "customers",
        "acct": "account",
        "accts": "accounts",
        "txn": "transaction",
        "txns": "transactions",
        "inv": "inventory",
        "wo": "work_order",
        "po": "purchase_order",
        "so": "sales_order",
        "emp": "employee",
        "emps": "employees",
        "dept": "department",
        "depts": "departments",
        "org": "organization",
        "orgs": "organizations",
        "loc": "location",
        "locs": "locations",
        "cat": "category",
        "cats": "categories",

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
        "cd": "code",
        "nm": "name",
        "no": "number",
        "num": "number",
        "desc": "description",
        "lbl": "label",
        "idx": "index",
        "typ": "type",
        "key": "key",

        # === 수치/통계 ===
        "qty": "quantity",
        "amt": "amount",
        "cnt": "count",
        "avg": "average",
        "min": "minimum",
        "max": "maximum",
        "tot": "total",
        "sum": "sum",
        "pct": "percent",
        "val": "value",
        "prc": "price",
        "flg": "flag",
        "ind": "indicator",

        # === 시간 관련 ===
        "dt": "date",
        "tm": "time",
        "ts": "timestamp",
        "yr": "year",
        "mo": "month",
        "dy": "day",
        "wk": "week",
        "hr": "hour",
        "mn": "minute",
        "sec": "second",
        "dur": "duration",

        # === 위치/주소 ===
        "addr": "address",
        "lat": "latitude",
        "lng": "longitude",
        "lon": "longitude",
        "cty": "city",
        "st": "state",
        "ctry": "country",
        "rgn": "region",
        "zip": "zipcode",
        "pst": "postal",

        # === 사람/조직 ===
        "mgr": "manager",
        "supv": "supervisor",
        "tel": "telephone",
        "fax": "fax",

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

    # FK 컬럼 패턴 (정규표현식)
    FK_PATTERNS = [
        (r"^(.+)_id$", "entity_id"),           # campaign_id
        (r"^(.+)_ID$", "entity_ID"),           # Campaign_ID
        (r"^(.+)ID$", "entityID"),             # CampaignID
        (r"^(.+)_cd$", "entity_cd"),           # prod_cd
        (r"^(.+)_code$", "entity_code"),       # product_code
        (r"^(.+)_CODE$", "entity_CODE"),       # PRODUCT_CODE
        (r"^(.+)_no$", "entity_no"),           # acct_no
        (r"^(.+)_num$", "entity_num"),         # order_num
        (r"^(.+)_number$", "entity_number"),   # order_number
        (r"^(.+)_key$", "entity_key"),         # order_key
        (r"^(.+)_KEY$", "entity_KEY"),         # ORDER_KEY
        (r"^fk_(.+)$", "fk_entity"),           # fk_customer
        (r"^FK_(.+)$", "FK_entity"),           # FK_CUSTOMER
        (r"^(.+)_fk$", "entity_fk"),           # customer_fk
        (r"^(.+)_ref$", "entity_ref"),         # customer_ref
        (r"^id_(.+)$", "id_entity"),           # id_customer
    ]

    def normalize_table_name(self, name: str) -> NormalizedName:
        """테이블명 정규화"""
        original = name
        normalized = name.lower()

        # 접두사 분리
        prefix = None
        for pfx in self.TABLE_PREFIXES:
            if normalized.startswith(pfx):
                prefix = pfx.rstrip("_")
                normalized = normalized[len(pfx):]
                break

        # 도메인 prefix 탐지 (예: google_ads_, meta_ads_)
        domain_prefix_match = re.match(r"^([a-z]+_[a-z]+)_(.+)$", normalized)
        if domain_prefix_match:
            prefix = domain_prefix_match.group(1)
            normalized = domain_prefix_match.group(2)

        # PascalCase/camelCase → snake_case
        normalized = self._to_snake_case(normalized)

        # 토큰 분리
        tokens = normalized.split("_")

        # 엔티티명 추출 (마지막 의미있는 토큰)
        entity = tokens[-1] if tokens else None

        return NormalizedName(
            original=original,
            normalized=normalized,
            tokens=tokens,
            prefix=prefix,
            entity=entity,
        )

    def normalize_column_name(self, name: str) -> NormalizedName:
        """컬럼명 정규화"""
        original = name
        normalized = name.lower()

        # PascalCase/camelCase → snake_case
        normalized = self._to_snake_case(normalized)

        # FK 패턴 매칭
        suffix = None
        entity = None
        for pattern, pattern_name in self.FK_PATTERNS:
            match = re.match(pattern, original, re.IGNORECASE)
            if match:
                entity = match.group(1).lower()
                entity = self._to_snake_case(entity)
                # suffix 추출 (_id, _code 등)
                suffix_match = re.search(r"(_id|_code|_cd|_no|_num|_key|_ref|_fk)$", normalized, re.IGNORECASE)
                if suffix_match:
                    suffix = suffix_match.group(1).lower()
                break

        # 토큰 분리
        tokens = normalized.split("_")

        return NormalizedName(
            original=original,
            normalized=normalized,
            tokens=tokens,
            entity=entity,
            suffix=suffix,
        )

    def expand_abbreviation(self, token: str) -> str:
        """축약어 확장"""
        return self.ABBREVIATIONS.get(token.lower(), token)

    def _to_snake_case(self, name: str) -> str:
        """PascalCase/camelCase를 snake_case로 변환"""
        # 이미 snake_case면 그대로
        if "_" in name and name.islower():
            return name

        # PascalCase → snake_case
        result = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", name)
        result = re.sub(r"([a-z\d])([A-Z])", r"\1_\2", result)
        return result.lower()

    def compute_name_similarity(self, name1: NormalizedName, name2: NormalizedName) -> float:
        """두 정규화된 이름의 유사도 계산"""
        # 정규화된 이름이 같으면 1.0
        if name1.normalized == name2.normalized:
            return 1.0

        # 엔티티명이 같으면 높은 점수
        if name1.entity and name2.entity:
            if name1.entity == name2.entity:
                return 0.9
            # 복수형/단수형 매칭
            if name1.entity.rstrip("s") == name2.entity.rstrip("s"):
                return 0.85

        # 토큰 Jaccard
        tokens1 = set(name1.tokens)
        tokens2 = set(name2.tokens)
        if tokens1 and tokens2:
            intersection = tokens1 & tokens2
            union = tokens1 | tokens2
            return len(intersection) / len(union) * 0.7

        return 0.0


# =============================================================================
# Value-Based FK Validator
# =============================================================================

class ValueBasedFKValidator:
    """
    값 기반 FK 검증기

    네이밍과 무관하게 실제 데이터 값으로 FK 관계 검증:
    - Jaccard Similarity: 값 집합의 교집합/합집합
    - Containment: FK 값이 PK 값의 부분집합인지
    - Cardinality: PK는 unique, FK는 중복 허용
    """

    def validate_fk_relationship(
        self,
        fk_values: List[Any],
        pk_values: List[Any],
    ) -> Dict[str, Any]:
        """
        FK-PK 관계 검증

        Returns:
            검증 결과 딕셔너리
        """
        # None 제거 및 문자열 변환
        fk_set = set(str(v) for v in fk_values if v is not None)
        pk_set = set(str(v) for v in pk_values if v is not None)

        if not fk_set or not pk_set:
            return {
                "jaccard": 0.0,
                "containment_fk_in_pk": 0.0,
                "containment_pk_in_fk": 0.0,
                "fk_unique_ratio": 0.0,
                "pk_unique_ratio": 0.0,
                "is_valid_fk": False,
                "sample_overlapping": [],
                "sample_fk_only": [],
            }

        # Jaccard Similarity
        intersection = fk_set & pk_set
        union = fk_set | pk_set
        jaccard = len(intersection) / len(union) if union else 0.0

        # Containment
        containment_fk_in_pk = len(intersection) / len(fk_set) if fk_set else 0.0
        containment_pk_in_fk = len(intersection) / len(pk_set) if pk_set else 0.0

        # Cardinality (unique ratio)
        fk_unique_ratio = len(fk_set) / len(fk_values) if fk_values else 0.0
        pk_unique_ratio = len(pk_set) / len(pk_values) if pk_values else 0.0

        # FK에만 있는 값 (referential integrity 위반 후보)
        fk_only = fk_set - pk_set

        # 샘플 값
        sample_overlapping = list(intersection)[:10]
        sample_fk_only = list(fk_only)[:5]

        # FK 유효성 판단 (기본 휴리스틱)
        # - FK 값의 80% 이상이 PK에 있어야 함
        # - PK는 unique해야 함 (unique ratio > 0.9)
        is_valid_fk = (
            containment_fk_in_pk >= 0.8 and
            pk_unique_ratio >= 0.9
        )

        return {
            "jaccard": jaccard,
            "containment_fk_in_pk": containment_fk_in_pk,
            "containment_pk_in_fk": containment_pk_in_fk,
            "fk_unique_ratio": fk_unique_ratio,
            "pk_unique_ratio": pk_unique_ratio,
            "is_valid_fk": is_valid_fk,
            "sample_overlapping": sample_overlapping,
            "sample_fk_only": sample_fk_only,
        }

    def compute_all_overlaps(
        self,
        tables_data: Dict[str, Dict[str, List[Any]]],
        min_jaccard: float = 0.3,  # v7.9: 0.1→0.3 상향 (False Positive 감소)
        min_containment: float = 0.8,  # v7.9: 0.95→0.8 (AND 조건이므로 하향)
    ) -> List[Dict[str, Any]]:
        """
        모든 테이블 쌍의 컬럼 간 값 오버랩 계산

        v7.9: Precision 개선
        - min_jaccard 0.1→0.3 상향: 10%→30% 이상 오버랩 필요
        - min_containment 0.95→0.8: AND 조건 적용으로 하향 조정
        - OR→AND 조건: Jaccard >= 0.3 AND Containment >= 0.8

        Args:
            tables_data: {table_name: {column_name: [values]}}
            min_jaccard: 최소 Jaccard threshold (v7.9: 0.3)
            min_containment: 최소 Containment threshold (v7.9: 0.8, AND 조건)

        Returns:
            오버랩 결과 리스트
        """
        overlaps = []
        table_names = list(tables_data.keys())

        for i, table_a in enumerate(table_names):
            for table_b in table_names[i+1:]:
                cols_a = tables_data[table_a]
                cols_b = tables_data[table_b]

                for col_a, values_a in cols_a.items():
                    for col_b, values_b in cols_b.items():
                        result = self.validate_fk_relationship(values_a, values_b)

                        # v7.9: Jaccard AND Containment 기반 필터링 (OR→AND 변경)
                        # - Jaccard >= 0.3: 30% 이상 값 오버랩 필요
                        # - Containment >= 0.8: FK 컬럼 값의 80%가 PK에 존재
                        meets_jaccard = result["jaccard"] >= min_jaccard
                        meets_containment = (
                            result["containment_fk_in_pk"] >= min_containment or
                            result["containment_pk_in_fk"] >= min_containment
                        )

                        # v7.9: AND 조건으로 변경 - 둘 다 만족해야 FK 후보
                        if meets_jaccard and meets_containment:
                            # v7.4: 탐지 방법 기록
                            detection_reason = []
                            if meets_jaccard:
                                detection_reason.append(f"jaccard:{result['jaccard']:.3f}")
                            if meets_containment:
                                if result["containment_fk_in_pk"] >= min_containment:
                                    detection_reason.append(f"containment_a_in_b:{result['containment_fk_in_pk']:.3f}")
                                if result["containment_pk_in_fk"] >= min_containment:
                                    detection_reason.append(f"containment_b_in_a:{result['containment_pk_in_fk']:.3f}")

                            overlaps.append({
                                "table_a": table_a,
                                "column_a": col_a,
                                "table_b": table_b,
                                "column_b": col_b,
                                "detection_reason": "|".join(detection_reason),  # v7.4
                                **result,
                            })

        # v7.4: Containment 우선, Jaccard 보조로 정렬
        # (Containment 높은 것이 FK 관계일 가능성 높음)
        overlaps.sort(
            key=lambda x: (
                max(x["containment_fk_in_pk"], x["containment_pk_in_fk"]),
                x["jaccard"]
            ),
            reverse=True
        )
        return overlaps


# =============================================================================
# Universal FK Detector
# =============================================================================

class UniversalFKDetector:
    """
    범용 FK 탐지기

    도메인 지식 없이 테이블 구조와 데이터만으로 FK 관계 탐지

    탐지 전략:
    1. 네이밍 정규화로 다양한 컨벤션 통일
    2. 공통 prefix로 테이블 그룹 탐지
    3. 값 기반 검증으로 실제 관계 확인
    4. 계층 구조 추론
    """

    def __init__(self):
        self.normalizer = NamingNormalizer()
        self.value_validator = ValueBasedFKValidator()

    def detect_fk_candidates(
        self,
        tables_columns: Dict[str, List[str]],
        tables_data: Optional[Dict[str, Dict[str, List[Any]]]] = None,
    ) -> List[FKCandidate]:
        """
        FK 후보 탐지

        Args:
            tables_columns: {table_name: [column_names]}
            tables_data: {table_name: {column_name: [values]}} (optional)

        Returns:
            FK 후보 리스트 (LLM 판단용)
        """
        candidates = []

        # 1. 테이블명 정규화
        normalized_tables = {
            table: self.normalizer.normalize_table_name(table)
            for table in tables_columns.keys()
        }

        # 2. 컬럼명 정규화
        normalized_columns = {}
        for table, columns in tables_columns.items():
            normalized_columns[table] = {
                col: self.normalizer.normalize_column_name(col)
                for col in columns
            }

        # 3. 테이블 그룹 탐지 (공통 prefix)
        table_groups = self._detect_table_groups(normalized_tables)

        # 4. FK 패턴 기반 후보 탐지
        pattern_candidates = self._detect_by_column_pattern(
            tables_columns, normalized_tables, normalized_columns, table_groups
        )
        candidates.extend(pattern_candidates)

        # 5. 값 기반 검증 (데이터가 있는 경우)
        if tables_data:
            candidates = self._enrich_with_value_analysis(candidates, tables_data)

            # 추가: 값 기반으로 새 후보 탐지
            value_candidates = self._detect_by_value_overlap(
                tables_data, normalized_tables, normalized_columns
            )
            # 중복 제거 후 병합
            existing_keys = {
                (c.from_table, c.from_column, c.to_table, c.to_column)
                for c in candidates
            }
            for vc in value_candidates:
                key = (vc.from_table, vc.from_column, vc.to_table, vc.to_column)
                if key not in existing_keys:
                    candidates.append(vc)

        # 6. 신뢰도 점수 계산
        for candidate in candidates:
            self._compute_confidence_factors(candidate)

        # === v7.10: Post-Processing for Precision Improvement ===

        # 7. 역방향 FK 필터링 (Cardinality 기반 방향 판단)
        if tables_data:
            candidates = self._filter_reverse_fks(candidates, tables_data)

        # 8. 간접 연결 필터링 (Transitive Closure 제거)
        candidates = self._filter_transitive_connections(candidates)

        # 신뢰도 순 정렬
        candidates.sort(
            key=lambda x: sum(x.confidence_factors.values()),
            reverse=True
        )

        return candidates

    def _filter_reverse_fks(
        self,
        candidates: List[FKCandidate],
        tables_data: Dict[str, Dict[str, List[Any]]],
    ) -> List[FKCandidate]:
        """
        v7.10: 역방향 FK 필터링 (Cardinality 기반 방향 판단)

        FK-PK 관계에서:
        - PK: unique_ratio ≈ 1.0 (모든 값 고유)
        - FK: unique_ratio < 1.0 (중복 허용)

        A→B와 B→A 둘 다 있으면 올바른 방향만 유지
        """
        # Step 1: 양방향 쌍 찾기
        pair_map = {}  # frozenset -> [candidates]
        for c in candidates:
            pair = frozenset([
                (c.from_table, c.from_column),
                (c.to_table, c.to_column)
            ])
            if pair not in pair_map:
                pair_map[pair] = []
            pair_map[pair].append(c)

        filtered = []
        processed_pairs = set()

        for c in candidates:
            pair = frozenset([
                (c.from_table, c.from_column),
                (c.to_table, c.to_column)
            ])

            if pair in processed_pairs:
                continue

            pair_candidates = pair_map[pair]

            if len(pair_candidates) == 1:
                # 단방향만 있음 - 그대로 유지
                filtered.append(c)
            else:
                # 양방향 존재 - Cardinality로 방향 판단
                best = self._determine_fk_direction_by_cardinality(
                    pair_candidates, tables_data
                )
                if best:
                    filtered.append(best)

            processed_pairs.add(pair)

        logger.info(f"[v7.10] Reverse FK filter: {len(candidates)} → {len(filtered)} "
                   f"(removed {len(candidates) - len(filtered)} reverse FKs)")

        return filtered

    def _determine_fk_direction_by_cardinality(
        self,
        pair_candidates: List[FKCandidate],
        tables_data: Dict[str, Dict[str, List[Any]]],
    ) -> Optional[FKCandidate]:
        """
        Cardinality 기반 FK 방향 결정

        FK쪽은 N:1 관계에서 N쪽 (중복 많음)
        PK쪽은 1쪽 (unique)
        """
        best_candidate = None
        best_score = -1

        for c in pair_candidates:
            # from 컬럼의 unique ratio
            from_values = tables_data.get(c.from_table, {}).get(c.from_column, [])
            to_values = tables_data.get(c.to_table, {}).get(c.to_column, [])

            if not from_values or not to_values:
                continue

            from_unique = len(set(from_values))
            to_unique = len(set(to_values))
            from_ratio = from_unique / len(from_values) if from_values else 0
            to_ratio = to_unique / len(to_values) if to_values else 0

            # FK → PK 방향이 올바름:
            # - to (PK)가 더 unique해야 함 (to_ratio > from_ratio)
            # - 이상적: to_ratio ≈ 1.0, from_ratio < 1.0
            direction_score = to_ratio - from_ratio

            # PK가 unique하고 FK가 중복이면 높은 점수
            if to_ratio > 0.9 and from_ratio < 0.9:
                direction_score += 0.5

            if direction_score > best_score:
                best_score = direction_score
                best_candidate = c

        return best_candidate

    def _filter_transitive_connections(
        self,
        candidates: List[FKCandidate],
        confidence_threshold: float = 0.85,
    ) -> List[FKCandidate]:
        """
        v7.10: 간접 연결 필터링 (Transitive Closure 제거)

        A → C, B → C 가 있을 때 A → B 제거
        (같은 PK를 참조하는 FK들끼리의 연결)
        """
        # Step 1: 높은 confidence FK로 target → sources 매핑
        target_to_sources = defaultdict(set)

        for c in candidates:
            conf = sum(c.confidence_factors.values())
            if conf >= confidence_threshold:
                target_key = f"{c.to_table}.{c.to_column}"
                source_key = f"{c.from_table}.{c.from_column}"
                target_to_sources[target_key].add(source_key)

        # Step 2: 같은 target을 공유하는 source들 간의 연결 제거
        filtered = []
        removed_count = 0

        for c in candidates:
            source_key = f"{c.from_table}.{c.from_column}"
            target_key = f"{c.to_table}.{c.to_column}"

            is_transitive = False
            for common_target, sources in target_to_sources.items():
                # source와 target 모두 같은 common_target을 참조하면 간접 연결
                if source_key in sources and target_key in sources:
                    is_transitive = True
                    break

            if not is_transitive:
                filtered.append(c)
            else:
                removed_count += 1

        logger.info(f"[v7.10] Transitive filter: removed {removed_count} indirect connections")

        return filtered

    def _detect_table_groups(
        self,
        normalized_tables: Dict[str, NormalizedName],
    ) -> Dict[str, TableGroup]:
        """테이블 그룹 탐지 (공통 prefix)"""
        groups = defaultdict(list)

        for table, norm in normalized_tables.items():
            if norm.prefix:
                groups[norm.prefix].append(table)
            else:
                # prefix 없으면 "default" 그룹
                groups["default"].append(table)

        return {
            prefix: TableGroup(prefix=prefix, tables=tables)
            for prefix, tables in groups.items()
            if len(tables) >= 1
        }

    def _detect_by_column_pattern(
        self,
        tables_columns: Dict[str, List[str]],
        normalized_tables: Dict[str, NormalizedName],
        normalized_columns: Dict[str, Dict[str, NormalizedName]],
        table_groups: Dict[str, TableGroup],
    ) -> List[FKCandidate]:
        """컬럼 패턴 기반 FK 탐지"""
        candidates = []
        table_names = list(tables_columns.keys())

        # 테이블명 정규화 맵 (역방향 조회용)
        norm_to_tables = defaultdict(list)
        for table, norm in normalized_tables.items():
            norm_to_tables[norm.normalized].append(table)
            # 복수형도 등록
            if not norm.normalized.endswith("s"):
                norm_to_tables[norm.normalized + "s"].append(table)
            else:
                norm_to_tables[norm.normalized.rstrip("s")].append(table)

        for source_table, columns in tables_columns.items():
            source_norm = normalized_tables[source_table]

            for col in columns:
                col_norm = normalized_columns[source_table][col]

                # FK 패턴 컬럼인지 확인
                if not col_norm.entity or not col_norm.suffix:
                    continue

                entity = col_norm.entity

                # 1. 같은 prefix 그룹에서 타겟 테이블 찾기
                target_tables = []

                if source_norm.prefix:
                    # 같은 prefix + 엔티티명 조합
                    potential_targets = [
                        f"{source_norm.prefix}_{entity}",
                        f"{source_norm.prefix}_{entity}s",
                    ]
                    for pt in potential_targets:
                        for table in table_names:
                            if normalized_tables[table].normalized == pt or table.lower() == pt:
                                if table != source_table:
                                    target_tables.append(table)

                # 2. 전역에서 엔티티명으로 찾기
                for norm_name, tables in norm_to_tables.items():
                    if entity in norm_name or norm_name in entity:
                        for table in tables:
                            if table != source_table and table not in target_tables:
                                target_tables.append(table)

                # 3. 타겟 테이블에서 매칭되는 PK 컬럼 찾기
                for target_table in target_tables:
                    target_cols = tables_columns[target_table]

                    for target_col in target_cols:
                        target_col_norm = normalized_columns[target_table][target_col]

                        # 같은 정규화된 이름이면 FK 후보
                        name_sim = self.normalizer.compute_name_similarity(col_norm, target_col_norm)

                        if name_sim >= 0.7:
                            candidate = FKCandidate(
                                from_table=source_table,
                                from_column=col,
                                to_table=target_table,
                                to_column=target_col,
                                name_similarity=name_sim,
                                same_normalized_name=(col_norm.normalized == target_col_norm.normalized),
                                same_prefix_group=(source_norm.prefix == normalized_tables[target_table].prefix),
                                detection_method="column_pattern",
                            )
                            candidates.append(candidate)

        return candidates

    def _detect_by_value_overlap(
        self,
        tables_data: Dict[str, Dict[str, List[Any]]],
        normalized_tables: Dict[str, NormalizedName],
        normalized_columns: Dict[str, Dict[str, NormalizedName]],
    ) -> List[FKCandidate]:
        """
        값 오버랩 기반 FK 탐지

        v7.4: Containment 강화
        - Jaccard 낮아도 Containment 95%+ 면 FK 후보로 포함
        - 컬럼명 다르더라도 값 기반으로 관계 탐지 가능
        """
        candidates = []

        # v7.4: Jaccard 0.05로 낮추고 (실질적으로 무시), Containment 95%로 탐지
        overlaps = self.value_validator.compute_all_overlaps(
            tables_data,
            min_jaccard=0.05,         # v7.4: 낮은 Jaccard도 허용
            min_containment=0.95,     # v7.4: Containment 95%+ 면 포함
        )

        for overlap in overlaps:
            table_a = overlap["table_a"]
            table_b = overlap["table_b"]
            col_a = overlap["column_a"]
            col_b = overlap["column_b"]

            # v7.4: Containment 기반 방향 결정
            # containment_fk_in_pk: A의 값이 B에 포함된 비율 (A → B FK)
            # containment_pk_in_fk: B의 값이 A에 포함된 비율 (B → A FK)
            containment_a_to_b = overlap["containment_fk_in_pk"]
            containment_b_to_a = overlap["containment_pk_in_fk"]

            # 어느 방향이 FK인지 결정:
            # 1. Containment 높은 쪽이 FK (FK 값은 PK에 포함되어야 함)
            # 2. Unique ratio 낮은 쪽이 FK (FK는 중복 허용)
            if containment_a_to_b >= containment_b_to_a:
                # A → B (A가 FK, B가 PK)
                from_table, from_col = table_a, col_a
                to_table, to_col = table_b, col_b
                containment = containment_a_to_b
            else:
                # B → A (B가 FK, A가 PK)
                from_table, from_col = table_b, col_b
                to_table, to_col = table_a, col_a
                containment = containment_b_to_a

            # 컬럼 정규화 정보 가져오기
            if from_table in normalized_columns and from_col in normalized_columns[from_table]:
                from_col_norm = normalized_columns[from_table][from_col]
            else:
                from_col_norm = self.normalizer.normalize_column_name(from_col)

            if to_table in normalized_columns and to_col in normalized_columns[to_table]:
                to_col_norm = normalized_columns[to_table][to_col]
            else:
                to_col_norm = self.normalizer.normalize_column_name(to_col)

            name_sim = self.normalizer.compute_name_similarity(from_col_norm, to_col_norm)

            # v7.4: detection_method에 탐지 이유 포함
            detection_reason = overlap.get("detection_reason", "value_overlap")
            detection_method = f"value_overlap({detection_reason})"

            candidate = FKCandidate(
                from_table=from_table,
                from_column=from_col,
                to_table=to_table,
                to_column=to_col,
                jaccard_similarity=overlap["jaccard"],
                containment_fk_in_pk=containment,
                containment_pk_in_fk=overlap["containment_pk_in_fk"] if containment == containment_a_to_b else overlap["containment_fk_in_pk"],
                fk_unique_ratio=overlap["fk_unique_ratio"],
                pk_unique_ratio=overlap["pk_unique_ratio"],
                name_similarity=name_sim,
                same_normalized_name=(from_col_norm.normalized == to_col_norm.normalized),
                sample_overlapping_values=overlap["sample_overlapping"],
                sample_fk_only_values=overlap["sample_fk_only"],
                detection_method=detection_method,
            )
            candidates.append(candidate)

        return candidates

    def _enrich_with_value_analysis(
        self,
        candidates: List[FKCandidate],
        tables_data: Dict[str, Dict[str, List[Any]]],
    ) -> List[FKCandidate]:
        """패턴 기반 후보에 값 분석 결과 추가"""
        for candidate in candidates:
            from_table = candidate.from_table
            from_col = candidate.from_column
            to_table = candidate.to_table
            to_col = candidate.to_column

            # 데이터 가져오기
            fk_values = tables_data.get(from_table, {}).get(from_col, [])
            pk_values = tables_data.get(to_table, {}).get(to_col, [])

            if fk_values and pk_values:
                result = self.value_validator.validate_fk_relationship(fk_values, pk_values)

                candidate.jaccard_similarity = result["jaccard"]
                candidate.containment_fk_in_pk = result["containment_fk_in_pk"]
                candidate.containment_pk_in_fk = result["containment_pk_in_fk"]
                candidate.fk_unique_ratio = result["fk_unique_ratio"]
                candidate.pk_unique_ratio = result["pk_unique_ratio"]
                candidate.sample_overlapping_values = result["sample_overlapping"]
                candidate.sample_fk_only_values = result["sample_fk_only"]

        return candidates

    def _compute_confidence_factors(self, candidate: FKCandidate) -> None:
        """
        신뢰도 요소 계산

        v7.4: Containment 가중치 대폭 상향
        - 컬럼명이 달라도 값 기반으로 FK 관계 탐지
        - Containment 높으면 FK 관계 가능성 매우 높음
        """
        factors = {}

        # 1. 이름 유사도 (0-0.2) - v7.4: 가중치 하향 (컬럼명 다를 수 있음)
        factors["name_similarity"] = candidate.name_similarity * 0.2

        # 2. 값 오버랩 Jaccard (0-0.15) - v7.4: 가중치 하향
        if candidate.jaccard_similarity > 0:
            factors["value_overlap"] = min(candidate.jaccard_similarity, 1.0) * 0.15

        # 3. Containment (0-0.4) - v7.4: 가중치 대폭 상향 (핵심 지표)
        # FK 값이 PK에 포함된 비율이 높으면 FK 관계 가능성 높음
        if candidate.containment_fk_in_pk > 0:
            # 95%+ containment에 보너스
            containment_score = candidate.containment_fk_in_pk * 0.35
            if candidate.containment_fk_in_pk >= 0.95:
                containment_score += 0.05  # 95%+ 보너스
            factors["containment"] = min(containment_score, 0.4)

        # 4. 카디널리티 패턴 (0-0.15) - v7.4: 가중치 상향
        # PK는 unique해야 하고, FK는 중복 가능
        if candidate.pk_unique_ratio > 0.9 and candidate.fk_unique_ratio < 0.9:
            factors["cardinality_pattern"] = 0.15

        # 5. 같은 prefix 그룹 (0-0.1)
        if candidate.same_prefix_group:
            factors["same_group"] = 0.1

        candidate.confidence_factors = factors

    def format_for_llm(self, candidates: List[FKCandidate]) -> str:
        """LLM 판단을 위한 포맷팅"""
        output = []

        for i, c in enumerate(candidates[:30], 1):  # 상위 30개
            confidence = sum(c.confidence_factors.values())

            output.append(f"""
## FK Candidate #{i}
- **From**: {c.from_table}.{c.from_column}
- **To**: {c.to_table}.{c.to_column}

### Algorithm Results (for your judgment):
| Metric | Value | Interpretation |
|--------|-------|----------------|
| Jaccard Similarity | {c.jaccard_similarity:.3f} | How much values overlap |
| FK values in PK | {c.containment_fk_in_pk:.3f} | Should be high (>0.8) for valid FK |
| FK Unique Ratio | {c.fk_unique_ratio:.3f} | FK can have duplicates (low is OK) |
| PK Unique Ratio | {c.pk_unique_ratio:.3f} | PK should be unique (high >0.9) |
| Name Similarity | {c.name_similarity:.3f} | Normalized name match |

### Evidence:
- Same column name (normalized): {c.same_normalized_name}
- Same table group: {c.same_prefix_group}
- Sample overlapping values: {c.sample_overlapping_values[:5]}
- Values only in FK (potential violations): {c.sample_fk_only_values[:3]}

### Detection Method: {c.detection_method}
### Algorithm Confidence: {confidence:.2f}
""")

        return "\n".join(output)
