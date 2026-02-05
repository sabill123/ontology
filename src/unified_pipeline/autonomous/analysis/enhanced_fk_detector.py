"""
Enhanced Foreign Key Detection Module (v7.8)

연구 기반 강화된 FK 탐지 알고리즘:
1. Column Name Matching - 동일/유사 컬럼명 탐지
2. Value Inclusion Dependency (IND) - 포함 종속성
3. Cardinality Analysis - 기수성 분석
4. Data Type Compatibility - 데이터 타입 호환성
5. Statistical FK Scoring - 통계적 FK 점수화
6. Matched Value Prefix Analysis (v7.7) - 값 패턴 기반 FK 추론
7. Adaptive Threshold (v7.8) - 데이터 품질 기반 동적 threshold
8. Cardinality-Aware MVP (v7.8) - many-to-one 관계 검증

v7.8 변경사항:
- FK Score > 1.0 버그 수정: min(1.0, score) 상한선 추가
- ABBREVIATION_MAP 하드코딩 제거: buyer→customer, user→customer 삭제 (MVP 대체)
- Adaptive Threshold: 데이터 품질(NULL/empty 비율)에 따라 min_score 동적 조정
- MVP Cardinality Awareness: FK unique <= PK unique 검증으로 false positive 감소

v7.7 변경사항:
- Matched Value Prefix Analysis 추가: 컬럼명이 다르더라도 값의 prefix 패턴이
  일치하면 FK 관계로 추론 (buyer_id → customer_id 감지 가능)
- ABBREVIATION_MAP 하드코딩 의존도 감소
- Dirty data에 대한 강건성 향상 (매칭된 값만 분석)

v7.6 변경사항:
- FK/PK 패턴 확장: _number, _no, _num, _ref 등 추가
- PK 패턴에 _code 추가 (iata_code, icao_code 등 도메인 코드 지원)
- 값 기반 폴백 탐지: 패턴 매칭 실패해도 Containment 95%+ 면 FK 후보로 포함
- 시맨틱 유사도: 축약어 처리 (flt→flight, reg→registration 등)

References:
- Papenbrock et al. "A Hybrid Approach to Functional Dependency Discovery" (2015)
- Abedjan et al. "Profiling Relational Data: A Survey" (2015)
- Koutras et al. "Valentine: Evaluating Matching Techniques for Dataset Discovery" (2021)
"""

import re
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Set, Tuple
from collections import defaultdict
import hashlib

logger = logging.getLogger(__name__)


@dataclass
class FKCandidate:
    """FK 후보"""
    from_table: str
    from_column: str
    to_table: str
    to_column: str

    # Scores
    name_similarity: float = 0.0
    value_inclusion: float = 0.0  # IND score
    cardinality_ratio: float = 0.0
    type_compatibility: float = 0.0

    # Evidence
    sample_matches: List[Any] = field(default_factory=list)
    unmatched_count: int = 0
    total_count: int = 0

    # Final score
    fk_score: float = 0.0
    confidence: str = "low"  # low, medium, high, certain

    def to_dict(self) -> Dict[str, Any]:
        return {
            "from_table": self.from_table,
            "from_column": self.from_column,
            "to_table": self.to_table,
            "to_column": self.to_column,
            "name_similarity": round(self.name_similarity, 4),
            "value_inclusion": round(self.value_inclusion, 4),
            "cardinality_ratio": round(self.cardinality_ratio, 4),
            "type_compatibility": round(self.type_compatibility, 4),
            "fk_score": round(self.fk_score, 4),
            "confidence": self.confidence,
            "sample_matches": self.sample_matches[:5],
            "unmatched_count": self.unmatched_count,
            "total_count": self.total_count,
        }


class EnhancedFKDetector:
    """
    강화된 FK 탐지기

    알고리즘:
    1. Column Name Matching (weighted score)
    2. Value Inclusion Dependency (IND)
    3. Cardinality-based filtering
    4. Type compatibility check
    5. Combined scoring with confidence levels
    """

    # FK 컬럼명 패턴 (v7.6 확장)
    FK_PATTERNS = [
        r'^(.+)_id$',           # xxx_id
        r'^(.+)_key$',          # xxx_key
        r'^(.+)_code$',         # xxx_code
        r'^fk_(.+)$',           # fk_xxx
        r'^(.+)_fk$',           # xxx_fk
        r'^id_(.+)$',           # id_xxx
        r'^ref_(.+)$',          # ref_xxx
        # v7.6: 추가 패턴
        r'^(.+)_number$',       # xxx_number (flight_number, flt_number)
        r'^(.+)_num$',          # xxx_num (flight_num)
        r'^(.+)_no$',           # xxx_no (flt_no, registration_no)
        r'^(.+)_ref$',          # xxx_ref (booking_ref)
        r'^(.+)_reference$',    # xxx_reference (booking_reference)
        r'^(.+)_reg$',          # xxx_reg (aircraft_reg)
    ]

    # PK 컬럼명 패턴 (v7.6 확장)
    PK_PATTERNS = [
        r'^id$',
        r'^(.+)_id$',
        r'^pk$',
        r'^(.+)_pk$',
        r'^(.+)_key$',
        # v7.6: 추가 패턴
        r'^(.+)_code$',         # xxx_code (iata_code, icao_code - 도메인 고유 식별자)
        r'^(.+)_number$',       # xxx_number (flight_number - PK로도 사용 가능)
        r'^(.+)_no$',           # xxx_no (registration_no)
    ]

    # v7.6: 도메인 축약어 매핑 (시맨틱 유사도용)
    # v1: 마케팅 도메인 축약어 추가 (cust, user, buyer → customer)
    ABBREVIATION_MAP = {
        "flt": "flight",
        "reg": "registration",
        "pax": "passenger",
        "maint": "maintenance",
        "emp": "employee",
        "dept": "department",
        "acft": "aircraft",
        "arr": "arrival",
        "dep": "departure",
        "sched": "scheduled",
        "act": "actual",
        # v7.8: 일반적 축약어만 유지 (도메인 특정 추측 제거 - MVP가 대체)
        "cust": "customer",
        # "user": "customer",  # v7.8: 삭제 - user != customer (false positive 위험)
        # "buyer": "customer", # v7.8: 삭제 - MVP 알고리즘이 값 패턴으로 감지
        "prod": "product",
        "cat": "category",
        "ord": "order",
        "inv": "inventory",
        "promo": "promotion",
        "camp": "campaign",
        # v22.0: 의료/헬스케어 축약어 추가
        "doc": "doctor",
        "dr": "doctor",
        "pt": "patient",
        "pat": "patient",
        "appt": "appointment",
        "rx": "prescription",
        "dx": "diagnosis",
        "diag": "diagnosis",
        "med": "medication",
        "meds": "medications",
        "proc": "procedure",
        "phys": "physician",
        "surg": "surgery",
        "hosp": "hospital",
        "spec": "specialty",
    }

    # v22.0: 의미적 역할 → 테이블 매핑 (semantic role → target table)
    # 컬럼명이 _doc, _physician, _by 등으로 끝나면 특정 테이블을 참조
    SEMANTIC_ROLE_MAP = {
        # _doc, _dr, _doctor 접미사 → doctors 테이블
        "_doc": "doctor",
        "_dr": "doctor",
        "_doctor": "doctor",  # head_doctor, primary_doctor 등
        # _physician 접미사 → doctors 테이블
        "_physician": "doctor",
        # _by 접미사 (의료 도메인에서 doctors 가능성 높음)
        "diagnosed_by": "doctor",
        "prescribing_": "doctor",
        "ordering_": "doctor",
        "attending_": "doctor",
        "referring_": "doctor",
        "head_": "doctor",  # head_doctor, head_of_dept
        "primary_": "doctor",  # primary_doctor
        "created_by": "user",
        "updated_by": "user",
        "assigned_to": "user",
        # patient 관련
        "_patient": "patient",
        "patient_": "patient",
    }

    def __init__(self):
        self.candidates: List[FKCandidate] = []

        # Weights for scoring
        self.weights = {
            "name_exact": 0.35,      # 정확한 이름 매칭
            "name_similar": 0.20,    # 유사 이름 매칭
            "value_inclusion": 0.30, # 값 포함도
            "cardinality": 0.10,     # 기수성
            "type_compat": 0.05,     # 타입 호환성
        }

    def _compute_adaptive_threshold(
        self,
        sample_data: Dict[str, List[Dict[str, Any]]],
        base_threshold: float = 0.3,
    ) -> float:
        """
        v7.8: Adaptive Threshold 알고리즘

        데이터 품질에 따라 동적으로 min_score threshold를 조정합니다.
        - Clean data (품질 > 90%): threshold 유지 (정확도 중시)
        - Moderate data (품질 70-90%): threshold 약간 낮춤
        - Dirty data (품질 < 70%): threshold 더 낮춤 (recall 중시)

        알고리즘:
        1. 샘플 데이터의 NULL/empty 비율 측정
        2. ID 패턴 컬럼의 값 유효성 검사
        3. 품질 점수 기반 threshold 조정

        Args:
            sample_data: {table_name: [row_dicts]}
            base_threshold: 기본 threshold (default: 0.3)

        Returns:
            조정된 threshold (0.15 ~ 0.4)
        """
        if not sample_data:
            return base_threshold

        total_values = 0
        invalid_values = 0

        for table_name, rows in sample_data.items():
            if not rows:
                continue

            for row in rows:
                for col_name, value in row.items():
                    # ID 패턴 컬럼만 검사
                    if not any(p in col_name.lower() for p in ['id', '_id', 'key', 'no', 'num']):
                        continue

                    total_values += 1

                    # Invalid 조건: NULL, empty, 비정상 패턴
                    if value is None:
                        invalid_values += 1
                    elif str(value).strip() == '':
                        invalid_values += 1
                    elif str(value).lower() in ['null', 'none', 'n/a', 'nan']:
                        invalid_values += 1

        if total_values == 0:
            return base_threshold

        # 데이터 품질 점수 (0.0 ~ 1.0)
        quality_score = 1.0 - (invalid_values / total_values)

        # Adaptive threshold 계산
        # 품질 좋음 (> 0.9): threshold 유지/증가 → 정확도 중시
        # 품질 나쁨 (< 0.7): threshold 낮춤 → recall 중시
        if quality_score > 0.9:
            adjusted = base_threshold + 0.05  # 정확도 향상
        elif quality_score > 0.7:
            adjusted = base_threshold  # 유지
        elif quality_score > 0.5:
            adjusted = base_threshold - 0.05  # 약간 낮춤
        else:
            adjusted = base_threshold - 0.1  # 더 낮춤 (dirty data 대응)

        # 범위 제한 (0.15 ~ 0.4)
        adjusted = max(0.15, min(0.4, adjusted))

        logger.debug(
            f"[Adaptive Threshold] quality={quality_score:.2f}, "
            f"base={base_threshold}, adjusted={adjusted}"
        )

        return adjusted

    def detect_foreign_keys(
        self,
        tables: Dict[str, Dict[str, Any]],
        sample_data: Dict[str, List[Dict[str, Any]]] = None,
        min_score: float = 0.3,
    ) -> List[FKCandidate]:
        """
        FK 관계 탐지

        Args:
            tables: {table_name: {"columns": [...], "row_count": int}}
            sample_data: {table_name: [row_dicts]}
            min_score: 최소 FK 점수

        Returns:
            FK 후보 리스트
        """
        sample_data = sample_data or {}
        self.candidates = []

        # v7.8: Adaptive Threshold - 데이터 품질에 따라 동적 조정
        effective_min_score = self._compute_adaptive_threshold(sample_data, min_score)
        if effective_min_score != min_score:
            logger.info(f"[Adaptive] min_score adjusted: {min_score} -> {effective_min_score}")
        min_score = effective_min_score

        # 1. 모든 컬럼 분류
        pk_columns = {}  # {table: [pk_cols]}
        fk_columns = {}  # {table: [fk_cols]}

        for table_name, table_info in tables.items():
            columns = table_info.get("columns", [])
            pk_columns[table_name] = []
            fk_columns[table_name] = []

            for col in columns:
                col_name = col.get("name", "")
                if self._is_pk_pattern(col_name):
                    pk_columns[table_name].append(col)
                if self._is_fk_pattern(col_name):
                    fk_columns[table_name].append(col)

        # 2. FK 후보 생성
        for from_table, fk_cols in fk_columns.items():
            for fk_col in fk_cols:
                fk_name = fk_col.get("name", "")

                # FK 이름에서 참조 대상 추론
                base_name = self._extract_base_name(fk_name)

                for to_table, pk_cols in pk_columns.items():
                    if from_table == to_table:
                        continue

                    for pk_col in pk_cols:
                        pk_name = pk_col.get("name", "")

                        # 후보 생성
                        candidate = FKCandidate(
                            from_table=from_table,
                            from_column=fk_name,
                            to_table=to_table,
                            to_column=pk_name,
                        )

                        # 점수 계산
                        self._score_candidate(
                            candidate,
                            fk_col,
                            pk_col,
                            tables[from_table],
                            tables[to_table],
                            sample_data.get(from_table, []),
                            sample_data.get(to_table, []),
                        )

                        if candidate.fk_score >= min_score:
                            self.candidates.append(candidate)

        # 3. 동일 컬럼명 직접 매칭 (cross-table)
        self._detect_same_name_fks(tables, sample_data, min_score)

        # v7.6: 4. 값 기반 폴백 탐지 (패턴 매칭 실패해도 Containment 95%+ 면 포함)
        self._detect_by_value_inclusion(tables, sample_data, min_containment=0.95)

        # 5. 중복 제거 및 정렬
        self.candidates = self._deduplicate_candidates(self.candidates)
        self.candidates.sort(key=lambda x: x.fk_score, reverse=True)

        return self.candidates

    def _detect_by_value_inclusion(
        self,
        tables: Dict[str, Dict[str, Any]],
        sample_data: Dict[str, List[Dict[str, Any]]],
        min_containment: float = 0.95,
    ):
        """
        v7.6: 값 기반 폴백 FK 탐지

        패턴 매칭이 실패해도 Containment >= 95%면 FK 후보로 추가.
        이는 flt_no -> flight_number 같은 축약어 케이스를 잡기 위함.
        """
        # 이미 탐지된 FK 쌍 기록 (중복 방지)
        existing_pairs = set()
        for c in self.candidates:
            existing_pairs.add((c.from_table, c.from_column, c.to_table, c.to_column))
            existing_pairs.add((c.to_table, c.to_column, c.from_table, c.from_column))

        table_names = list(tables.keys())

        for i, from_table in enumerate(table_names):
            from_cols = tables[from_table].get("columns", [])
            from_data = sample_data.get(from_table, [])

            for to_table in table_names[i+1:]:
                to_cols = tables[to_table].get("columns", [])
                to_data = sample_data.get(to_table, [])

                for from_col in from_cols:
                    from_col_name = from_col.get("name", "")

                    # ID-like 컬럼만 체크 (최소한의 필터)
                    if not self._is_id_like_column(from_col_name):
                        continue

                    for to_col in to_cols:
                        to_col_name = to_col.get("name", "")

                        if not self._is_id_like_column(to_col_name):
                            continue

                        # 이미 탐지된 쌍이면 스킵
                        if (from_table, from_col_name, to_table, to_col_name) in existing_pairs:
                            continue

                        # 값 추출
                        from_values = self._get_column_values(from_data, from_col_name)
                        to_values = self._get_column_values(to_data, to_col_name)

                        if not from_values or not to_values:
                            continue

                        # 양방향 Containment 계산
                        inclusion_from_to = self._compute_inclusion(from_values, to_values)
                        inclusion_to_from = self._compute_inclusion(to_values, from_values)

                        # v7.6: 축약어 확장 후 시맨틱 유사도 계산
                        from_base = self._extract_base_name(from_col_name)
                        to_base = self._extract_base_name(to_col_name)
                        from_expanded = self._expand_abbreviation(from_base)
                        to_expanded = self._expand_abbreviation(to_base)

                        # 확장 후 같으면 시맨틱 매칭
                        semantic_match = (from_expanded == to_expanded)

                        # Containment 95%+ 또는 시맨틱 매칭이면 FK 후보로
                        if inclusion_from_to >= min_containment or (semantic_match and inclusion_from_to >= 0.8):
                            # from -> to (from이 FK, to가 PK)
                            candidate = FKCandidate(
                                from_table=from_table,
                                from_column=from_col_name,
                                to_table=to_table,
                                to_column=to_col_name,
                                name_similarity=1.0 if semantic_match else self._levenshtein_similarity(from_expanded, to_expanded),
                                value_inclusion=inclusion_from_to,
                                cardinality_ratio=len(set(to_values)) / len(set(from_values)) if len(set(from_values)) > 0 else 0,
                                type_compatibility=1.0,
                            )
                            candidate.fk_score = self._compute_final_score(candidate)
                            candidate.confidence = self._determine_confidence(candidate)
                            candidate.sample_matches = list(set(from_values) & set(to_values))[:10]
                            candidate.total_count = len(from_values)
                            candidate.unmatched_count = len(set(from_values) - set(to_values))

                            self.candidates.append(candidate)
                            existing_pairs.add((from_table, from_col_name, to_table, to_col_name))

                            logger.debug(
                                f"v7.6 Value-based FK: {from_table}.{from_col_name} -> {to_table}.{to_col_name} "
                                f"(containment={inclusion_from_to:.2f}, semantic={semantic_match})"
                            )

                        if inclusion_to_from >= min_containment or (semantic_match and inclusion_to_from >= 0.8):
                            # to -> from (to가 FK, from이 PK)
                            if (to_table, to_col_name, from_table, from_col_name) not in existing_pairs:
                                candidate = FKCandidate(
                                    from_table=to_table,
                                    from_column=to_col_name,
                                    to_table=from_table,
                                    to_column=from_col_name,
                                    name_similarity=1.0 if semantic_match else self._levenshtein_similarity(from_expanded, to_expanded),
                                    value_inclusion=inclusion_to_from,
                                    cardinality_ratio=len(set(from_values)) / len(set(to_values)) if len(set(to_values)) > 0 else 0,
                                    type_compatibility=1.0,
                                )
                                candidate.fk_score = self._compute_final_score(candidate)
                                candidate.confidence = self._determine_confidence(candidate)
                                candidate.sample_matches = list(set(from_values) & set(to_values))[:10]
                                candidate.total_count = len(to_values)
                                candidate.unmatched_count = len(set(to_values) - set(from_values))

                                self.candidates.append(candidate)
                                existing_pairs.add((to_table, to_col_name, from_table, from_col_name))

                                logger.debug(
                                    f"v7.6 Value-based FK: {to_table}.{to_col_name} -> {from_table}.{from_col_name} "
                                    f"(containment={inclusion_to_from:.2f}, semantic={semantic_match})"
                                )

    def _detect_same_name_fks(
        self,
        tables: Dict[str, Dict[str, Any]],
        sample_data: Dict[str, List[Dict[str, Any]]],
        min_score: float,
    ):
        """동일 컬럼명 FK 탐지 (핵심 개선)"""

        # 컬럼명별로 테이블 그룹화
        column_tables: Dict[str, List[Tuple[str, Dict]]] = defaultdict(list)

        for table_name, table_info in tables.items():
            for col in table_info.get("columns", []):
                col_name = col.get("name", "")
                column_tables[col_name].append((table_name, col))

        # 동일 컬럼명이 2개 이상 테이블에 있는 경우
        for col_name, table_list in column_tables.items():
            if len(table_list) < 2:
                continue

            # ID 패턴인 경우만 FK 후보로
            if not self._is_id_like_column(col_name):
                continue

            # 모든 테이블 쌍 비교
            for i, (table_a, col_a) in enumerate(table_list):
                for table_b, col_b in table_list[i+1:]:
                    # 값 포함도 계산
                    values_a = self._get_column_values(sample_data.get(table_a, []), col_name)
                    values_b = self._get_column_values(sample_data.get(table_b, []), col_name)

                    if not values_a or not values_b:
                        continue

                    # A가 B를 참조하는지 (A의 값이 B에 포함되는지)
                    inclusion_a_in_b = self._compute_inclusion(values_a, values_b)
                    inclusion_b_in_a = self._compute_inclusion(values_b, values_a)

                    # 방향 결정 (더 작은 unique가 FK)
                    unique_a = len(set(values_a))
                    unique_b = len(set(values_b))

                    if inclusion_a_in_b >= 0.5 and unique_a >= unique_b:
                        # A -> B (A가 B를 참조)
                        candidate = FKCandidate(
                            from_table=table_a,
                            from_column=col_name,
                            to_table=table_b,
                            to_column=col_name,
                            name_similarity=1.0,  # 동일 이름
                            value_inclusion=inclusion_a_in_b,
                            cardinality_ratio=unique_b / unique_a if unique_a > 0 else 0,
                            type_compatibility=1.0,  # 동일 이름이므로 타입도 호환
                        )
                        candidate.fk_score = self._compute_final_score(candidate)
                        candidate.confidence = self._determine_confidence(candidate)
                        candidate.sample_matches = list(set(values_a) & set(values_b))[:10]
                        candidate.total_count = len(values_a)
                        candidate.unmatched_count = len(set(values_a) - set(values_b))

                        if candidate.fk_score >= min_score:
                            self.candidates.append(candidate)

                    if inclusion_b_in_a >= 0.5 and unique_b >= unique_a:
                        # B -> A (B가 A를 참조)
                        candidate = FKCandidate(
                            from_table=table_b,
                            from_column=col_name,
                            to_table=table_a,
                            to_column=col_name,
                            name_similarity=1.0,
                            value_inclusion=inclusion_b_in_a,
                            cardinality_ratio=unique_a / unique_b if unique_b > 0 else 0,
                            type_compatibility=1.0,
                        )
                        candidate.fk_score = self._compute_final_score(candidate)
                        candidate.confidence = self._determine_confidence(candidate)
                        candidate.sample_matches = list(set(values_a) & set(values_b))[:10]
                        candidate.total_count = len(values_b)
                        candidate.unmatched_count = len(set(values_b) - set(values_a))

                        if candidate.fk_score >= min_score:
                            self.candidates.append(candidate)

    def _score_candidate(
        self,
        candidate: FKCandidate,
        fk_col: Dict,
        pk_col: Dict,
        from_table: Dict,
        to_table: Dict,
        from_data: List[Dict],
        to_data: List[Dict],
    ):
        """FK 후보 점수 계산 (v7.7: Matched Value Prefix Analysis 통합)"""

        # 1. 이름 유사도
        candidate.name_similarity = self._compute_name_similarity(
            candidate.from_column,
            candidate.to_column,
            candidate.to_table,
        )

        # 2. 값 포함도 (IND)
        fk_values = self._get_column_values(from_data, candidate.from_column)
        pk_values = self._get_column_values(to_data, candidate.to_column)

        if fk_values and pk_values:
            candidate.value_inclusion = self._compute_inclusion(fk_values, pk_values)
            candidate.sample_matches = list(set(fk_values) & set(pk_values))[:10]
            candidate.total_count = len(fk_values)
            candidate.unmatched_count = len(set(fk_values) - set(pk_values))

            # v7.7: Matched Value Prefix Analysis
            # name_similarity가 낮아도 값의 prefix 패턴이 일치하면 보완
            if candidate.name_similarity < 0.7:
                prefix_score = self._compute_matched_value_prefix(fk_values, pk_values)

                if prefix_score > 0.7:
                    # prefix 패턴이 강하게 일치하면 name_similarity 보완
                    boosted_similarity = max(candidate.name_similarity, prefix_score * 0.85)
                    logger.debug(
                        f"[MVP] Boosting name_similarity: "
                        f"{candidate.from_table}.{candidate.from_column} -> "
                        f"{candidate.to_table}.{candidate.to_column}: "
                        f"{candidate.name_similarity:.2f} -> {boosted_similarity:.2f} "
                        f"(prefix_score={prefix_score:.2f})"
                    )
                    candidate.name_similarity = boosted_similarity

        # 3. 기수성 비율
        fk_unique = len(set(fk_values)) if fk_values else 0
        pk_unique = len(set(pk_values)) if pk_values else 0

        if fk_unique > 0 and pk_unique > 0:
            # FK unique <= PK unique 이면 valid
            candidate.cardinality_ratio = min(1.0, pk_unique / fk_unique)

        # 4. 타입 호환성
        candidate.type_compatibility = self._compute_type_compatibility(
            fk_col.get("dtype", ""),
            pk_col.get("dtype", ""),
        )

        # 5. 최종 점수
        candidate.fk_score = self._compute_final_score(candidate)
        candidate.confidence = self._determine_confidence(candidate)

    def _compute_name_similarity(self, fk_name: str, pk_name: str, pk_table: str) -> float:
        """컬럼명 유사도 계산 (v7.6 시맨틱 유사도 추가, v1 PK 테이블 우선순위 추가, v22.0 시맨틱 역할 추가)"""
        fk_lower = fk_name.lower()
        pk_lower = pk_name.lower()
        pk_table_lower = pk_table.lower()

        # 복수형 제거하여 테이블 기본명 추출 (customers → customer)
        table_base = pk_table_lower.rstrip('s')
        if table_base.endswith('e'):
            # "es" 복수형 처리 (classes → class)
            pass
        else:
            table_base = pk_table_lower.rstrip('s').rstrip('es')

        # 정확히 같은 이름
        if fk_lower == pk_lower:
            return 1.0

        # v22.0: 시맨틱 역할 매핑 (ordering_physician → doctors.doctor_id)
        for pattern, target_entity in self.SEMANTIC_ROLE_MAP.items():
            if pattern.startswith("_"):
                # 접미사 패턴 (_doc, _physician 등)
                if fk_lower.endswith(pattern):
                    if target_entity in table_base or table_base in target_entity:
                        return 0.9  # 높은 유사도
            elif pattern.endswith("_"):
                # 접두사 패턴 (attending_, ordering_ 등)
                if fk_lower.startswith(pattern):
                    if target_entity in table_base or table_base in target_entity:
                        return 0.9
            else:
                # 전체 매칭 (diagnosed_by 등)
                if fk_lower == pattern or fk_lower.replace("_", "") == pattern.replace("_", ""):
                    if target_entity in table_base or table_base in target_entity:
                        return 0.9

        # v7.6: 축약어 확장 후 비교
        fk_base = self._extract_base_name(fk_name)
        pk_base = self._extract_base_name(pk_name)
        fk_expanded = self._expand_abbreviation(fk_base)
        pk_expanded = self._expand_abbreviation(pk_base)

        # v1: PK 테이블명 매칭 보너스
        # FK 확장형이 PK 테이블명과 일치하면 "진짜" PK 테이블로 판단
        # e.g., cust_id(→customer) → customers 테이블 = 마스터 테이블 우선
        table_match_bonus = 0.0
        if fk_expanded == table_base:
            table_match_bonus = 0.05

        # 확장 후 같으면 높은 유사도 (+ 테이블 매칭 보너스)
        if fk_expanded == pk_expanded:
            return min(1.0, 0.95 + table_match_bonus)

        # base가 같으면 높은 점수 (+ 테이블 매칭 보너스)
        if fk_base == pk_base:
            return min(1.0, 0.9 + table_match_bonus)

        # FK base가 PK 테이블명과 관련있는지
        # e.g., order_id -> orders 테이블
        if fk_base == table_base:
            return 0.85

        # v1: 축약어 확장 후 테이블명과 비교
        # e.g., cust_id -> customers 테이블 (cust -> customer == customer)
        if fk_expanded == table_base:
            return 0.85

        if fk_base in pk_table_lower or pk_table_lower in fk_base:
            return 0.7

        # v1: 확장된 FK base가 테이블명에 포함되는지 확인
        if fk_expanded in pk_table_lower or pk_table_lower in fk_expanded:
            return 0.7

        # Levenshtein 기반 유사도
        return self._levenshtein_similarity(fk_base, pk_base) * 0.5

    def _compute_inclusion(self, fk_values: List, pk_values: List) -> float:
        """값 포함도 (Inclusion Dependency) 계산"""
        if not fk_values:
            return 0.0

        fk_set = set(str(v).strip().lower() for v in fk_values if v is not None)
        pk_set = set(str(v).strip().lower() for v in pk_values if v is not None)

        if not fk_set:
            return 0.0

        # FK 값 중 PK에 포함된 비율
        matched = fk_set & pk_set
        return len(matched) / len(fk_set)

    def _compute_matched_value_prefix(self, fk_values: List, pk_values: List) -> float:
        """
        v7.7: Matched Value Prefix Analysis

        컬럼명이 아닌 실제 값의 prefix 패턴으로 FK 관계를 추론합니다.
        이 방식은 dirty data와 축약어에 강건합니다.

        알고리즘:
        1. FK ∩ PK 교집합 계산 (실제 매칭된 값)
        2. 매칭된 값들의 공통 prefix 추출
        3. PK 값들의 공통 prefix 추출
        4. prefix 일치 + match ratio로 점수 계산

        예시:
        - FK: ["CUST000001", "CUST000002", "INVALID001"]  (buyer_id)
        - PK: ["CUST000001", "CUST000002", "CUST000003"]  (customer_id)
        - matched: ["CUST000001", "CUST000002"]
        - matched_prefix: "CUST", pk_prefix: "CUST" → match!
        - match_ratio: 2/3 = 0.67
        - final_score: 0.8 + 0.2 * 0.67 = 0.93

        Args:
            fk_values: FK 컬럼 값 리스트
            pk_values: PK 컬럼 값 리스트

        Returns:
            0.0~1.0 점수 (높을수록 FK 관계 가능성 높음)
        """
        if not fk_values or not pk_values:
            return 0.0

        # 문자열로 정규화
        fk_set = set(str(v).strip() for v in fk_values if v is not None and str(v).strip())
        pk_set = set(str(v).strip() for v in pk_values if v is not None and str(v).strip())

        if not fk_set or not pk_set:
            return 0.0

        # 1. 매칭된 값 찾기
        matched = fk_set & pk_set

        if not matched:
            return 0.0

        # 2. prefix 추출 함수
        def extract_common_prefix(values: set) -> str:
            """값들의 공통 prefix 추출"""
            if not values:
                return ""

            # 문자열만 필터링
            str_values = [str(v) for v in values if v]
            if not str_values:
                return ""

            # 가장 짧은 문자열을 기준으로
            min_len = min(len(s) for s in str_values)
            if min_len == 0:
                return ""

            # 공통 prefix 찾기
            prefix = ""
            for i in range(min_len):
                chars = set(s[i] for s in str_values)
                if len(chars) == 1:
                    prefix += str_values[0][i]
                else:
                    break

            # 알파벳/숫자 경계에서 자르기 (CUST000 → CUST)
            alpha_prefix = ""
            for c in prefix:
                if c.isalpha():
                    alpha_prefix += c
                elif c.isdigit() and alpha_prefix:
                    # 알파벳 다음 숫자가 나오면 중단
                    break
                else:
                    alpha_prefix += c

            return alpha_prefix.upper() if alpha_prefix else prefix[:4].upper()

        # 3. prefix 추출
        matched_prefix = extract_common_prefix(matched)
        pk_prefix = extract_common_prefix(pk_set)

        # 최소 2자 이상의 prefix 필요
        if len(matched_prefix) < 2 or len(pk_prefix) < 2:
            return 0.0

        # 4. prefix 매칭 검사
        if matched_prefix == pk_prefix:
            # 완전 일치
            match_ratio = len(matched) / len(fk_set)
            base_score = 0.8 + 0.2 * match_ratio

            # v7.8: Cardinality Awareness - FK는 many-to-one
            # FK unique <= PK unique이면 valid FK 패턴
            cardinality_bonus = 0.0
            if len(fk_set) <= len(pk_set):
                cardinality_bonus = 0.05  # valid FK 패턴 보너스
            elif len(fk_set) > len(pk_set) * 2:
                base_score *= 0.9  # FK unique >> PK unique이면 패널티

            score = min(1.0, base_score + cardinality_bonus)
            logger.debug(
                f"[MVP] Prefix match: matched_prefix={matched_prefix}, "
                f"pk_prefix={pk_prefix}, match_ratio={match_ratio:.2f}, "
                f"cardinality_bonus={cardinality_bonus}, score={score:.2f}"
            )
            return score
        elif pk_prefix.startswith(matched_prefix) or matched_prefix.startswith(pk_prefix):
            # 부분 일치 (CUST vs CUSTOMER)
            match_ratio = len(matched) / len(fk_set)
            base_score = 0.6 + 0.2 * match_ratio

            # v7.8: Cardinality Awareness
            cardinality_bonus = 0.0
            if len(fk_set) <= len(pk_set):
                cardinality_bonus = 0.05

            score = min(1.0, base_score + cardinality_bonus)
            logger.debug(
                f"[MVP] Partial prefix match: matched_prefix={matched_prefix}, "
                f"pk_prefix={pk_prefix}, match_ratio={match_ratio:.2f}, "
                f"cardinality_bonus={cardinality_bonus}, score={score:.2f}"
            )
            return score

        return 0.0

    def _compute_type_compatibility(self, fk_type: str, pk_type: str) -> float:
        """데이터 타입 호환성"""
        fk_type = fk_type.lower()
        pk_type = pk_type.lower()

        if fk_type == pk_type:
            return 1.0

        # 숫자 타입 호환
        numeric_types = {"int", "int64", "int32", "float", "float64", "double", "number"}
        if any(t in fk_type for t in numeric_types) and any(t in pk_type for t in numeric_types):
            return 0.9

        # 문자열 타입 호환
        string_types = {"str", "string", "object", "varchar", "text", "char"}
        if any(t in fk_type for t in string_types) and any(t in pk_type for t in string_types):
            return 0.9

        return 0.3

    def _compute_final_score(self, candidate: FKCandidate) -> float:
        """최종 FK 점수 계산"""

        # 가중 평균
        if candidate.name_similarity >= 0.9:
            # 이름이 거의 같으면 name_exact 가중치 사용
            name_score = candidate.name_similarity * self.weights["name_exact"]
        else:
            name_score = candidate.name_similarity * self.weights["name_similar"]

        inclusion_score = candidate.value_inclusion * self.weights["value_inclusion"]
        cardinality_score = candidate.cardinality_ratio * self.weights["cardinality"]
        type_score = candidate.type_compatibility * self.weights["type_compat"]

        total_weight = sum(self.weights.values())

        # v7.8: 정규화 후 상한선 적용 (score > 1.0 버그 수정)
        raw_score = (name_score + inclusion_score + cardinality_score + type_score) / total_weight * (1 / 0.7)
        return min(1.0, raw_score)

    def _determine_confidence(self, candidate: FKCandidate) -> str:
        """신뢰도 결정"""
        score = candidate.fk_score

        if score >= 0.9 and candidate.value_inclusion >= 0.95:
            return "certain"
        elif score >= 0.7 and candidate.value_inclusion >= 0.8:
            return "high"
        elif score >= 0.5 and candidate.value_inclusion >= 0.5:
            return "medium"
        else:
            return "low"

    def _is_pk_pattern(self, col_name: str) -> bool:
        """PK 패턴 확인"""
        col_lower = col_name.lower()

        for pattern in self.PK_PATTERNS:
            if re.match(pattern, col_lower):
                return True

        return False

    def _is_fk_pattern(self, col_name: str) -> bool:
        """FK 패턴 확인 (v22.0: 시맨틱 역할 패턴 추가)"""
        col_lower = col_name.lower()

        # 기존 FK 패턴 확인
        for pattern in self.FK_PATTERNS:
            if re.match(pattern, col_lower):
                return True

        # v22.0: 시맨틱 역할 패턴 확인
        # _doc, _physician 등으로 끝나거나, diagnosed_by, attending_ 등의 패턴
        for pattern in self.SEMANTIC_ROLE_MAP.keys():
            if pattern.startswith("_"):
                # 접미사 패턴 (_doc, _physician 등)
                if col_lower.endswith(pattern):
                    return True
            elif pattern.endswith("_"):
                # 접두사 패턴 (attending_, ordering_ 등)
                if col_lower.startswith(pattern):
                    return True
            else:
                # 전체 매칭 (diagnosed_by 등)
                if col_lower == pattern:
                    return True

        return False

    def _is_id_like_column(self, col_name: str) -> bool:
        """ID 유사 컬럼 확인 (v7.6 확장)"""
        col_lower = col_name.lower()

        # v7.6: 확장된 ID 지시자 목록
        id_indicators = [
            "_id", "id_", "_key", "_code", "_ref", "uuid", "guid",
            "_number", "_num", "_no", "_reference", "_reg",  # v7.6 추가
        ]

        return any(ind in col_lower for ind in id_indicators) or col_lower == "id"

    def _extract_base_name(self, col_name: str) -> str:
        """컬럼명에서 base name 추출 (v7.6 확장)"""
        col_lower = col_name.lower()

        # v7.6: 확장된 접미사 목록 (긴 것부터 처리)
        suffixes = [
            "_reference", "_number", "_id", "_key", "_code", "_fk", "_pk",
            "_ref", "_reg", "_num", "_no",
        ]
        for suffix in suffixes:
            if col_lower.endswith(suffix):
                return col_lower[:-len(suffix)]

        # fk_, id_ 등 접두사 제거
        for prefix in ["fk_", "id_", "ref_", "pk_"]:
            if col_lower.startswith(prefix):
                return col_lower[len(prefix):]

        return col_lower

    def _expand_abbreviation(self, name: str) -> str:
        """v7.6: 축약어를 원래 형태로 확장"""
        name_lower = name.lower()
        for abbrev, full in self.ABBREVIATION_MAP.items():
            if name_lower == abbrev:
                return full
            # 부분 치환 (flt_number -> flight_number)
            if name_lower.startswith(abbrev + "_"):
                return full + name_lower[len(abbrev):]
        return name_lower

    def _get_column_values(self, data: List[Dict], col_name: str) -> List:
        """데이터에서 컬럼 값 추출"""
        values = []
        for row in data:
            if col_name in row:
                val = row[col_name]
                if val is not None:
                    values.append(val)
        return values

    def _levenshtein_similarity(self, s1: str, s2: str) -> float:
        """Levenshtein 유사도"""
        if not s1 or not s2:
            return 0.0

        if s1 == s2:
            return 1.0

        len1, len2 = len(s1), len(s2)

        # 간단한 DP 구현
        dp = [[0] * (len2 + 1) for _ in range(len1 + 1)]

        for i in range(len1 + 1):
            dp[i][0] = i
        for j in range(len2 + 1):
            dp[0][j] = j

        for i in range(1, len1 + 1):
            for j in range(1, len2 + 1):
                if s1[i-1] == s2[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                else:
                    dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])

        distance = dp[len1][len2]
        return 1 - distance / max(len1, len2)

    def _deduplicate_candidates(self, candidates: List[FKCandidate]) -> List[FKCandidate]:
        """중복 후보 제거 (더 높은 점수 유지)"""
        seen = {}

        for c in candidates:
            key = (c.from_table, c.from_column, c.to_table, c.to_column)
            if key not in seen or c.fk_score > seen[key].fk_score:
                seen[key] = c

        return list(seen.values())
