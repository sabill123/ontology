"""
Schema Analysis Module

스키마 분석 유틸리티
- 컬럼 프로파일링
- 키 후보 탐지
- 타입 추론
"""

import logging
import re
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Set, Tuple
from collections import defaultdict
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class ColumnProfile:
    """컬럼 프로파일"""
    table_name: str
    column_name: str

    # 기본 정보
    dtype: str
    inferred_type: str

    # 통계
    total_count: int
    null_count: int
    unique_count: int
    null_ratio: float
    unique_ratio: float

    # 샘플
    sample_values: List[Any] = field(default_factory=list)
    min_value: Any = None
    max_value: Any = None

    # 패턴 분석
    patterns: List[str] = field(default_factory=list)
    semantic_type: str = "unknown"  # id, name, date, email, phone, etc.

    # 키 후보
    is_key_candidate: bool = False
    key_confidence: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "table_name": self.table_name,
            "column_name": self.column_name,
            "dtype": self.dtype,
            "inferred_type": self.inferred_type,
            "total_count": self.total_count,
            "null_count": self.null_count,
            "unique_count": self.unique_count,
            "null_ratio": round(self.null_ratio, 4),
            "unique_ratio": round(self.unique_ratio, 4),
            "sample_values": self.sample_values[:5],
            "semantic_type": self.semantic_type,
            "is_key_candidate": self.is_key_candidate,
            "key_confidence": round(self.key_confidence, 4),
            "patterns": self.patterns[:3],
        }


@dataclass
class KeyCandidate:
    """키 후보"""
    table_name: str
    column_names: List[str]  # 단일 또는 복합 키
    key_type: str  # primary, unique, foreign
    confidence: float
    reasoning: List[str] = field(default_factory=list)

    # FK의 경우
    references_table: Optional[str] = None
    references_column: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "table_name": self.table_name,
            "column_names": self.column_names,
            "key_type": self.key_type,
            "confidence": round(self.confidence, 4),
            "reasoning": self.reasoning,
        }
        if self.references_table:
            result["references_table"] = self.references_table
            result["references_column"] = self.references_column
        return result


class SchemaAnalyzer:
    """
    스키마 분석기

    테이블 스키마의 상세 분석
    """

    def __init__(self):
        self.profiles: Dict[str, Dict[str, ColumnProfile]] = {}
        self.key_candidates: Dict[str, List[KeyCandidate]] = {}

        # 패턴 정규식
        self.patterns = {
            "email": re.compile(r'^[\w\.-]+@[\w\.-]+\.\w+$'),
            "phone": re.compile(r'^[\d\-\+\(\)\s]+$'),
            "uuid": re.compile(r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$', re.I),
            "date_iso": re.compile(r'^\d{4}-\d{2}-\d{2}'),
            "url": re.compile(r'^https?://'),
            "numeric_id": re.compile(r'^\d+$'),
            "code": re.compile(r'^[A-Z]{2,5}\d{2,}$'),  # CODE123 형태
            "postal_code": re.compile(r'^\d{5}(-\d{4})?$'),
        }

        # 의미론적 타입 키워드
        self.semantic_keywords = {
            "id": ["id", "key", "pk", "code", "num", "no"],
            "name": ["name", "title", "label", "description"],
            "date": ["date", "time", "created", "updated", "at", "on"],
            "amount": ["amount", "price", "cost", "total", "sum", "fee"],
            "count": ["count", "qty", "quantity", "number"],
            "status": ["status", "state", "flag", "active", "enabled"],
            "email": ["email", "mail"],
            "phone": ["phone", "tel", "mobile", "fax"],
            "address": ["address", "addr", "street", "city", "zip", "postal"],
        }

    def profile_column(
        self,
        table_name: str,
        column_name: str,
        values: List[Any],
        declared_dtype: str = None,
    ) -> ColumnProfile:
        """
        컬럼 프로파일링

        Args:
            table_name: 테이블 이름
            column_name: 컬럼 이름
            values: 컬럼 값 리스트
            declared_dtype: 선언된 타입

        Returns:
            ColumnProfile
        """
        total_count = len(values)
        non_null_values = [v for v in values if v is not None and v != '']

        null_count = total_count - len(non_null_values)
        unique_values = set(str(v) for v in non_null_values)
        unique_count = len(unique_values)

        null_ratio = null_count / total_count if total_count > 0 else 0.0
        unique_ratio = unique_count / len(non_null_values) if non_null_values else 0.0

        # 타입 추론
        inferred_type = self._infer_type(non_null_values)

        # 패턴 탐지
        patterns = self._detect_patterns(non_null_values)

        # 의미론적 타입
        semantic_type = self._infer_semantic_type(column_name, patterns, non_null_values)

        # 키 후보 여부
        is_key_candidate, key_confidence = self._assess_key_potential(
            column_name, unique_ratio, null_ratio, semantic_type
        )

        # 샘플 및 min/max
        sample_values = list(unique_values)[:10]
        min_val, max_val = self._get_min_max(non_null_values, inferred_type)

        profile = ColumnProfile(
            table_name=table_name,
            column_name=column_name,
            dtype=declared_dtype or "unknown",
            inferred_type=inferred_type,
            total_count=total_count,
            null_count=null_count,
            unique_count=unique_count,
            null_ratio=null_ratio,
            unique_ratio=unique_ratio,
            sample_values=sample_values,
            min_value=min_val,
            max_value=max_val,
            patterns=patterns,
            semantic_type=semantic_type,
            is_key_candidate=is_key_candidate,
            key_confidence=key_confidence,
        )

        # 저장
        if table_name not in self.profiles:
            self.profiles[table_name] = {}
        self.profiles[table_name][column_name] = profile

        return profile

    def _infer_type(self, values: List[Any]) -> str:
        """타입 추론"""
        if not values:
            return "unknown"

        # 샘플링
        sample = values[:100]

        type_counts = defaultdict(int)
        for v in sample:
            if v is None:
                continue
            elif isinstance(v, bool):
                type_counts["boolean"] += 1
            elif isinstance(v, int):
                type_counts["integer"] += 1
            elif isinstance(v, float):
                type_counts["float"] += 1
            elif isinstance(v, str):
                # 문자열 세부 분류
                if self._is_date_string(v):
                    type_counts["datetime"] += 1
                elif self._is_numeric_string(v):
                    type_counts["numeric_string"] += 1
                else:
                    type_counts["string"] += 1
            else:
                type_counts["object"] += 1

        if not type_counts:
            return "unknown"

        # 가장 많은 타입
        return max(type_counts, key=type_counts.get)

    def _is_date_string(self, s: str) -> bool:
        """날짜 문자열 여부"""
        date_patterns = [
            r'^\d{4}-\d{2}-\d{2}',  # ISO
            r'^\d{2}/\d{2}/\d{4}',  # US
            r'^\d{2}-\d{2}-\d{4}',  # EU
        ]
        return any(re.match(p, s) for p in date_patterns)

    def _is_numeric_string(self, s: str) -> bool:
        """숫자 문자열 여부"""
        try:
            float(s.replace(',', ''))
            return True
        except (ValueError, AttributeError):
            return False

    def _detect_patterns(self, values: List[Any]) -> List[str]:
        """패턴 탐지"""
        patterns_found = []

        if not values:
            return patterns_found

        # 문자열 값만
        str_values = [str(v) for v in values if v is not None][:100]

        if not str_values:
            return patterns_found

        # 각 패턴 검사
        for pattern_name, regex in self.patterns.items():
            match_count = sum(1 for v in str_values if regex.match(v))
            if match_count / len(str_values) > 0.8:
                patterns_found.append(pattern_name)

        return patterns_found

    def _infer_semantic_type(
        self,
        column_name: str,
        patterns: List[str],
        values: List[Any],
    ) -> str:
        """의미론적 타입 추론"""
        col_lower = column_name.lower()

        # 패턴 기반
        if "email" in patterns:
            return "email"
        if "phone" in patterns:
            return "phone"
        if "uuid" in patterns:
            return "uuid"
        if "date_iso" in patterns:
            return "date"
        if "url" in patterns:
            return "url"

        # 이름 기반
        for sem_type, keywords in self.semantic_keywords.items():
            for keyword in keywords:
                if keyword in col_lower:
                    return sem_type

        # 값 기반 추가 추론
        str_values = [str(v) for v in values if v is not None][:20]
        if str_values:
            avg_len = sum(len(v) for v in str_values) / len(str_values)
            if avg_len > 50:
                return "text"

        return "unknown"

    def _assess_key_potential(
        self,
        column_name: str,
        unique_ratio: float,
        null_ratio: float,
        semantic_type: str,
    ) -> Tuple[bool, float]:
        """키 후보 가능성 평가"""
        confidence = 0.0
        reasons = []

        # 고유성
        if unique_ratio >= 0.99:
            confidence += 0.4
            reasons.append("high_uniqueness")
        elif unique_ratio >= 0.95:
            confidence += 0.3
            reasons.append("good_uniqueness")

        # Null 없음
        if null_ratio == 0:
            confidence += 0.2
            reasons.append("no_nulls")
        elif null_ratio < 0.01:
            confidence += 0.1

        # 이름 패턴
        col_lower = column_name.lower()
        if col_lower.endswith("_id") or col_lower == "id":
            confidence += 0.2
            reasons.append("id_naming")
        elif "key" in col_lower or "pk" in col_lower:
            confidence += 0.15
            reasons.append("key_naming")

        # 의미론적 타입
        if semantic_type in ["id", "uuid"]:
            confidence += 0.1

        is_candidate = confidence >= 0.5
        return is_candidate, min(1.0, confidence)

    def _get_min_max(
        self,
        values: List[Any],
        inferred_type: str,
    ) -> Tuple[Any, Any]:
        """최소/최대값"""
        if not values:
            return None, None

        try:
            if inferred_type in ["integer", "float", "numeric_string"]:
                numeric_values = []
                for v in values:
                    try:
                        numeric_values.append(float(str(v).replace(',', '')))
                    except (ValueError, TypeError):
                        pass
                if numeric_values:
                    return min(numeric_values), max(numeric_values)

            elif inferred_type == "datetime":
                return min(values), max(values)

            elif inferred_type == "string":
                str_values = [str(v) for v in values]
                return min(str_values), max(str_values)

        except Exception:
            pass

        return None, None

    def analyze_table(
        self,
        table_name: str,
        columns: List[Dict[str, Any]],
        sample_data: List[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        테이블 전체 분석

        Args:
            table_name: 테이블 이름
            columns: 컬럼 메타데이터
            sample_data: 샘플 데이터

        Returns:
            {
                "table_name": str,
                "column_profiles": [...],
                "key_candidates": [...],
                "data_quality_score": float,
                "normalization_level": str,
            }
        """
        sample_data = sample_data or []

        # 컬럼별 값 추출
        column_values = defaultdict(list)
        for row in sample_data:
            for col in columns:
                col_name = col.get("name", "")
                if col_name in row:
                    column_values[col_name].append(row[col_name])

        # 각 컬럼 프로파일링
        profiles = []
        for col in columns:
            col_name = col.get("name", "")
            values = column_values.get(col_name, [])

            # 메타데이터에서 기존 값 활용
            if not values and "sample_values" in col:
                values = col.get("sample_values", [])

            profile = self.profile_column(
                table_name=table_name,
                column_name=col_name,
                values=values,
                declared_dtype=col.get("dtype"),
            )
            profiles.append(profile)

        # 키 후보 탐지
        key_candidates = self._detect_key_candidates(table_name, profiles)

        # 데이터 품질 점수
        quality_score = self._compute_quality_score(profiles)

        # 정규화 수준 평가
        normalization = self._assess_normalization(profiles, key_candidates)

        return {
            "table_name": table_name,
            "column_profiles": [p.to_dict() for p in profiles],
            "key_candidates": [k.to_dict() for k in key_candidates],
            "data_quality_score": round(quality_score, 4),
            "normalization_level": normalization,
            "column_count": len(profiles),
        }

    def _detect_key_candidates(
        self,
        table_name: str,
        profiles: List[ColumnProfile],
    ) -> List[KeyCandidate]:
        """키 후보 탐지"""
        candidates = []

        # Primary Key 후보
        pk_candidates = [p for p in profiles if p.is_key_candidate]
        pk_candidates.sort(key=lambda p: p.key_confidence, reverse=True)

        if pk_candidates:
            best = pk_candidates[0]
            candidates.append(KeyCandidate(
                table_name=table_name,
                column_names=[best.column_name],
                key_type="primary",
                confidence=best.key_confidence,
                reasoning=[
                    f"unique_ratio: {best.unique_ratio:.2%}",
                    f"null_ratio: {best.null_ratio:.2%}",
                    f"semantic_type: {best.semantic_type}",
                ],
            ))

        # Foreign Key 후보 (_id로 끝나는 컬럼 중 PK가 아닌 것)
        for p in profiles:
            if p.column_name.lower().endswith("_id") and not p.is_key_candidate:
                # 다른 테이블 참조 가능성
                referenced_table = p.column_name[:-3]  # _id 제거

                candidates.append(KeyCandidate(
                    table_name=table_name,
                    column_names=[p.column_name],
                    key_type="foreign",
                    confidence=0.6,
                    reasoning=[f"naming_pattern: ends with _id"],
                    references_table=referenced_table,
                    references_column="id",
                ))

        self.key_candidates[table_name] = candidates
        return candidates

    def _compute_quality_score(self, profiles: List[ColumnProfile]) -> float:
        """데이터 품질 점수"""
        if not profiles:
            return 0.0

        scores = []
        for p in profiles:
            col_score = 1.0

            # Null 패널티
            col_score -= p.null_ratio * 0.3

            # 유일성 보너스 (ID 컬럼인 경우)
            if p.semantic_type == "id" and p.unique_ratio < 0.95:
                col_score -= 0.3

            # 패턴 매칭 보너스
            if p.patterns:
                col_score += 0.1

            scores.append(max(0.0, min(1.0, col_score)))

        return sum(scores) / len(scores)

    def _assess_normalization(
        self,
        profiles: List[ColumnProfile],
        key_candidates: List[KeyCandidate],
    ) -> str:
        """정규화 수준 평가"""
        # 간단한 휴리스틱

        has_pk = any(k.key_type == "primary" for k in key_candidates)
        has_fk = any(k.key_type == "foreign" for k in key_candidates)

        redundant_patterns = 0
        for p in profiles:
            # 반복 패턴 체크 (예: address1, address2)
            if re.match(r'.+\d$', p.column_name):
                redundant_patterns += 1

        if not has_pk:
            return "0NF"
        elif redundant_patterns > 2:
            return "1NF"
        elif not has_fk and len(profiles) > 10:
            return "2NF"
        else:
            return "3NF"

    def find_cross_table_mappings(
        self,
        tables: Dict[str, Dict[str, ColumnProfile]],
    ) -> List[Dict[str, Any]]:
        """
        테이블 간 컬럼 매핑 찾기

        Returns:
            [
                {
                    "table_a": str,
                    "column_a": str,
                    "table_b": str,
                    "column_b": str,
                    "mapping_type": "exact" | "similar" | "semantic",
                    "confidence": float,
                }
            ]
        """
        mappings = []
        table_names = list(tables.keys())

        for i, table_a in enumerate(table_names):
            for table_b in table_names[i + 1:]:
                cols_a = tables[table_a]
                cols_b = tables[table_b]

                for col_a_name, prof_a in cols_a.items():
                    for col_b_name, prof_b in cols_b.items():
                        mapping = self._compare_columns(
                            table_a, prof_a,
                            table_b, prof_b,
                        )
                        if mapping:
                            mappings.append(mapping)

        # 신뢰도 순 정렬
        mappings.sort(key=lambda x: x["confidence"], reverse=True)
        return mappings

    def _compare_columns(
        self,
        table_a: str,
        prof_a: ColumnProfile,
        table_b: str,
        prof_b: ColumnProfile,
    ) -> Optional[Dict[str, Any]]:
        """두 컬럼 비교"""
        confidence = 0.0
        mapping_type = "none"

        col_a = prof_a.column_name.lower()
        col_b = prof_b.column_name.lower()

        # 정확히 같은 이름
        if col_a == col_b:
            confidence = 0.8
            mapping_type = "exact"
        # 접미사/접두사 제거 후 비교
        elif self._normalize_column_name(col_a) == self._normalize_column_name(col_b):
            confidence = 0.6
            mapping_type = "similar"

        # 타입 호환성
        if prof_a.inferred_type == prof_b.inferred_type:
            confidence += 0.1

        # 의미론적 타입
        if prof_a.semantic_type == prof_b.semantic_type and prof_a.semantic_type != "unknown":
            confidence += 0.1
            if mapping_type == "none":
                mapping_type = "semantic"
                confidence = max(confidence, 0.4)

        if confidence >= 0.4:
            return {
                "table_a": table_a,
                "column_a": prof_a.column_name,
                "table_b": table_b,
                "column_b": prof_b.column_name,
                "mapping_type": mapping_type,
                "confidence": round(confidence, 4),
            }

        return None

    def _normalize_column_name(self, name: str) -> str:
        """컬럼 이름 정규화"""
        # 공통 접미사 제거
        suffixes = ["_id", "_key", "_code", "_num", "_no", "_date", "_time"]
        for suffix in suffixes:
            if name.endswith(suffix):
                name = name[:-len(suffix)]
                break

        # 공통 접두사 제거
        prefixes = ["fk_", "pk_", "src_", "tgt_"]
        for prefix in prefixes:
            if name.startswith(prefix):
                name = name[len(prefix):]
                break

        return name
