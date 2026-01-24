"""
Value Pattern Analyzer (v1.0)

도메인 무관 범용 값 패턴 분석기
- 하드코딩된 도메인 지식 없음
- 순수 통계적/패턴 분석만 수행
- LLM이 결과를 해석하여 의미 부여

핵심 원리:
1. 값의 형태적 특성만 추출 (길이, 문자타입, 구조)
2. 같은 테이블 내 1:1 매핑 컬럼 탐지 (Bridge Table 후보)
3. 다른 테이블 간 패턴 유사성 분석
4. LLM에게 충분한 컨텍스트 제공
"""

import re
import logging
from typing import Dict, List, Any, Set, Tuple, Optional
from dataclasses import dataclass, field
from collections import Counter, defaultdict

logger = logging.getLogger(__name__)


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class ValuePattern:
    """단일 컬럼의 값 패턴 분석 결과"""
    table: str
    column: str

    # 기본 통계
    total_count: int
    unique_count: int
    null_count: int
    unique_ratio: float

    # 패턴 분포 (도메인 무관)
    length_distribution: Dict[int, int]  # {길이: 개수}
    dominant_length: int
    length_consistency: float  # 0-1, 1이면 모든 값이 같은 길이

    # 문자 타입 분포
    char_type_distribution: Dict[str, int]  # {"UPPER": n, "LOWER": n, ...}
    dominant_char_type: str

    # 구조 패턴
    structure_patterns: Dict[str, int]  # {"AAA": n, "AA9": n, "A9A9": n, ...}
    dominant_structure: str

    # 샘플
    sample_values: List[str] = field(default_factory=list)

    # 메타
    is_likely_code: bool = False  # 짧고 일관된 패턴 = 코드성 데이터
    is_likely_identifier: bool = False  # unique ratio 높고 패턴 일관

    def to_dict(self) -> Dict[str, Any]:
        return {
            "table": self.table,
            "column": self.column,
            "total_count": self.total_count,
            "unique_count": self.unique_count,
            "unique_ratio": round(self.unique_ratio, 4),
            "dominant_length": self.dominant_length,
            "length_consistency": round(self.length_consistency, 4),
            "dominant_char_type": self.dominant_char_type,
            "dominant_structure": self.dominant_structure,
            "sample_values": self.sample_values[:10],
            "is_likely_code": self.is_likely_code,
            "is_likely_identifier": self.is_likely_identifier,
        }


@dataclass
class SameTableMapping:
    """같은 테이블 내 1:1 매핑 컬럼 쌍 (Bridge Table 후보)"""
    table: str
    column_a: str
    column_b: str

    # 패턴 정보
    pattern_a: str  # "2_UPPER"
    pattern_b: str  # "3_UPPER"

    # 매핑 정보
    mapping_count: int  # 매핑된 쌍의 수
    is_bijective: bool  # 양방향 1:1인지

    # 샘플 매핑
    sample_mappings: List[Tuple[str, str]] = field(default_factory=list)

    # 신뢰도
    confidence: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "table": self.table,
            "column_a": self.column_a,
            "column_b": self.column_b,
            "pattern_a": self.pattern_a,
            "pattern_b": self.pattern_b,
            "mapping_count": self.mapping_count,
            "is_bijective": self.is_bijective,
            "sample_mappings": self.sample_mappings[:10],
            "confidence": round(self.confidence, 4),
        }


@dataclass
class CrossTablePatternMatch:
    """다른 테이블 간 같은 패턴을 가진 컬럼 쌍"""
    table_a: str
    column_a: str
    table_b: str
    column_b: str

    # 패턴 유사도
    pattern_match: str  # 공통 패턴 (예: "2_UPPER")
    length_match: bool
    char_type_match: bool
    structure_match: bool

    # 값 겹침 (직접 매칭 여부)
    has_value_overlap: bool
    overlap_ratio: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "table_a": self.table_a,
            "column_a": self.column_a,
            "table_b": self.table_b,
            "column_b": self.column_b,
            "pattern_match": self.pattern_match,
            "length_match": self.length_match,
            "char_type_match": self.char_type_match,
            "structure_match": self.structure_match,
            "has_value_overlap": self.has_value_overlap,
            "overlap_ratio": round(self.overlap_ratio, 4),
        }


# =============================================================================
# Value Pattern Analyzer
# =============================================================================

class ValuePatternAnalyzer:
    """
    범용 값 패턴 분석기

    도메인 지식 없이 순수 통계/패턴 분석만 수행
    LLM이 결과를 해석하여 의미 부여
    """

    def analyze_all_columns(
        self,
        tables_data: Dict[str, Dict[str, List[Any]]]
    ) -> Dict[str, ValuePattern]:
        """모든 테이블의 모든 컬럼 패턴 분석"""
        patterns = {}

        for table_name, columns_data in tables_data.items():
            for column_name, values in columns_data.items():
                key = f"{table_name}.{column_name}"
                patterns[key] = self.analyze_column(table_name, column_name, values)

        return patterns

    def analyze_column(
        self,
        table: str,
        column: str,
        values: List[Any]
    ) -> ValuePattern:
        """단일 컬럼 패턴 분석"""
        # None 제거 및 문자열 변환
        str_values = [str(v).strip() for v in values if v is not None and str(v).strip()]

        total_count = len(values)
        null_count = len(values) - len(str_values)
        unique_values = set(str_values)
        unique_count = len(unique_values)
        unique_ratio = unique_count / len(str_values) if str_values else 0

        # 길이 분포
        length_dist = Counter(len(v) for v in str_values)
        dominant_length = length_dist.most_common(1)[0][0] if length_dist else 0
        length_consistency = length_dist[dominant_length] / len(str_values) if str_values else 0

        # 문자 타입 분포
        char_types = Counter(self._classify_char_type(v) for v in str_values)
        dominant_char_type = char_types.most_common(1)[0][0] if char_types else "UNKNOWN"

        # 구조 패턴 분포
        structures = Counter(self._extract_structure(v) for v in str_values)
        dominant_structure = structures.most_common(1)[0][0] if structures else "UNKNOWN"

        # 코드성 데이터 판단 (짧고 일관된 패턴)
        is_likely_code = (
            dominant_length <= 10 and
            length_consistency >= 0.8 and
            dominant_char_type in ("UPPER", "ALNUM_UPPER", "NUMERIC")
        )

        # 식별자 판단 (unique하고 패턴 일관)
        is_likely_identifier = (
            unique_ratio >= 0.9 and
            length_consistency >= 0.7
        )

        return ValuePattern(
            table=table,
            column=column,
            total_count=total_count,
            unique_count=unique_count,
            null_count=null_count,
            unique_ratio=unique_ratio,
            length_distribution=dict(length_dist),
            dominant_length=dominant_length,
            length_consistency=length_consistency,
            char_type_distribution=dict(char_types),
            dominant_char_type=dominant_char_type,
            structure_patterns=dict(structures.most_common(10)),
            dominant_structure=dominant_structure,
            sample_values=list(unique_values)[:20],
            is_likely_code=is_likely_code,
            is_likely_identifier=is_likely_identifier,
        )

    def _classify_char_type(self, value: str) -> str:
        """문자 타입 분류 (도메인 무관)"""
        if not value:
            return "EMPTY"

        if value.isdigit():
            return "NUMERIC"
        elif value.isalpha():
            if value.isupper():
                return "UPPER"
            elif value.islower():
                return "LOWER"
            else:
                return "MIXED_CASE"
        elif value.isalnum():
            if value[0].isalpha():
                if all(c.isupper() or c.isdigit() for c in value):
                    return "ALNUM_UPPER"
                elif all(c.islower() or c.isdigit() for c in value):
                    return "ALNUM_LOWER"
                return "ALNUM_MIXED"
            else:
                return "NUM_ALPHA"
        else:
            # 특수문자 포함
            if '-' in value or '_' in value:
                return "WITH_SEPARATOR"
            elif ' ' in value:
                return "WITH_SPACE"
            else:
                return "SPECIAL"

    def _extract_structure(self, value: str) -> str:
        """
        값의 구조 패턴 추출 (도메인 무관)

        예:
        - "KE" → "AA"
        - "KE123" → "AA999"
        - "2024-01-15" → "9999-99-99"
        - "AB-123-XY" → "AA-999-AA"
        """
        if not value:
            return "EMPTY"

        pattern = []
        for char in value:
            if char.isupper():
                pattern.append('A')
            elif char.islower():
                pattern.append('a')
            elif char.isdigit():
                pattern.append('9')
            else:
                pattern.append(char)  # 구분자는 그대로 유지

        # 연속된 같은 패턴 압축 (선택적)
        # "AAAA" → "A+" 형태로 압축하지 않음 - 길이 정보 유지가 더 유용
        return ''.join(pattern)


# =============================================================================
# Same-Table Mapping Detector (Bridge Table 후보)
# =============================================================================

class SameTableMappingDetector:
    """
    같은 테이블 내 1:1 매핑 컬럼 탐지

    Bridge Table(매핑 테이블) 후보 발견
    예: airlines 테이블의 iata_code ↔ icao_code
    """

    def __init__(self, pattern_analyzer: ValuePatternAnalyzer):
        self.pattern_analyzer = pattern_analyzer

    def detect_mappings(
        self,
        tables_data: Dict[str, Dict[str, List[Any]]],
        patterns: Dict[str, ValuePattern]
    ) -> List[SameTableMapping]:
        """모든 테이블에서 1:1 매핑 컬럼 쌍 탐지"""
        mappings = []

        for table_name, columns_data in tables_data.items():
            table_mappings = self._detect_in_table(table_name, columns_data, patterns)
            mappings.extend(table_mappings)

        return mappings

    def _detect_in_table(
        self,
        table_name: str,
        columns_data: Dict[str, List[Any]],
        patterns: Dict[str, ValuePattern]
    ) -> List[SameTableMapping]:
        """단일 테이블 내 매핑 탐지"""
        mappings = []
        column_names = list(columns_data.keys())

        for i, col_a in enumerate(column_names):
            pattern_a = patterns.get(f"{table_name}.{col_a}")
            if not pattern_a:
                continue

            # 코드성 데이터만 대상
            if not pattern_a.is_likely_code and not pattern_a.is_likely_identifier:
                continue

            for col_b in column_names[i+1:]:
                pattern_b = patterns.get(f"{table_name}.{col_b}")
                if not pattern_b:
                    continue

                if not pattern_b.is_likely_code and not pattern_b.is_likely_identifier:
                    continue

                # 1:1 매핑 확인
                mapping_result = self._check_bijective_mapping(
                    columns_data[col_a],
                    columns_data[col_b]
                )

                if mapping_result["is_mapping"]:
                    mappings.append(SameTableMapping(
                        table=table_name,
                        column_a=col_a,
                        column_b=col_b,
                        pattern_a=f"{pattern_a.dominant_length}_{pattern_a.dominant_char_type}",
                        pattern_b=f"{pattern_b.dominant_length}_{pattern_b.dominant_char_type}",
                        mapping_count=mapping_result["mapping_count"],
                        is_bijective=mapping_result["is_bijective"],
                        sample_mappings=mapping_result["sample_mappings"],
                        confidence=mapping_result["confidence"],
                    ))

        return mappings

    def _check_bijective_mapping(
        self,
        values_a: List[Any],
        values_b: List[Any]
    ) -> Dict[str, Any]:
        """두 컬럼이 1:1 매핑인지 확인"""
        # 같은 row의 값 쌍으로 매핑 구축
        forward_map = {}  # a → b
        reverse_map = {}  # b → a

        valid_pairs = 0
        sample_mappings = []

        for va, vb in zip(values_a, values_b):
            if va is None or vb is None:
                continue

            sa = str(va).strip()
            sb = str(vb).strip()

            if not sa or not sb:
                continue

            valid_pairs += 1

            # Forward 매핑 확인
            if sa in forward_map:
                if forward_map[sa] != sb:
                    # 1:N 관계 - 매핑 아님
                    return {"is_mapping": False}
            else:
                forward_map[sa] = sb
                if len(sample_mappings) < 20:
                    sample_mappings.append((sa, sb))

            # Reverse 매핑 확인
            if sb in reverse_map:
                if reverse_map[sb] != sa:
                    # N:1 관계 - 단방향 매핑
                    pass  # 계속 진행
            else:
                reverse_map[sb] = sa

        if len(forward_map) < 3:
            return {"is_mapping": False}

        # Bijective 확인 (양방향 1:1)
        is_bijective = len(forward_map) == len(reverse_map)

        # 신뢰도 계산
        confidence = min(len(forward_map) / 20, 1.0) * (0.95 if is_bijective else 0.7)

        return {
            "is_mapping": True,
            "mapping_count": len(forward_map),
            "is_bijective": is_bijective,
            "sample_mappings": sample_mappings,
            "confidence": confidence,
        }


# =============================================================================
# Cross-Table Pattern Matcher
# =============================================================================

class CrossTablePatternMatcher:
    """
    다른 테이블 간 패턴 유사성 분석

    값이 직접 겹치지 않아도 패턴이 같으면
    같은 도메인의 다른 코드 체계일 수 있음
    """

    def find_pattern_matches(
        self,
        patterns: Dict[str, ValuePattern],
        tables_data: Dict[str, Dict[str, List[Any]]]
    ) -> List[CrossTablePatternMatch]:
        """패턴이 유사한 크로스 테이블 컬럼 쌍 찾기"""
        matches = []

        # 코드성 컬럼만 추출
        code_columns = [
            (key, pattern) for key, pattern in patterns.items()
            if pattern.is_likely_code or pattern.is_likely_identifier
        ]

        for i, (key_a, pattern_a) in enumerate(code_columns):
            table_a, col_a = key_a.split(".", 1)

            for key_b, pattern_b in code_columns[i+1:]:
                table_b, col_b = key_b.split(".", 1)

                # 같은 테이블은 스킵 (SameTableMappingDetector가 처리)
                if table_a == table_b:
                    continue

                # 패턴 비교
                length_match = pattern_a.dominant_length == pattern_b.dominant_length
                char_type_match = pattern_a.dominant_char_type == pattern_b.dominant_char_type
                structure_match = pattern_a.dominant_structure == pattern_b.dominant_structure

                # 값 겹침 확인
                values_a = set(str(v).strip() for v in tables_data[table_a][col_a] if v)
                values_b = set(str(v).strip() for v in tables_data[table_b][col_b] if v)

                intersection = values_a & values_b
                overlap_ratio = len(intersection) / min(len(values_a), len(values_b)) if values_a and values_b else 0

                # 패턴이 유사하면 기록 (값이 겹치든 안 겹치든)
                if char_type_match and (length_match or structure_match):
                    matches.append(CrossTablePatternMatch(
                        table_a=table_a,
                        column_a=col_a,
                        table_b=table_b,
                        column_b=col_b,
                        pattern_match=f"{pattern_a.dominant_length}_{pattern_a.dominant_char_type}",
                        length_match=length_match,
                        char_type_match=char_type_match,
                        structure_match=structure_match,
                        has_value_overlap=overlap_ratio > 0.1,
                        overlap_ratio=overlap_ratio,
                    ))

        return matches


# =============================================================================
# LLM Context Generator (핵심!)
# =============================================================================

class PatternAnalysisContextGenerator:
    """
    LLM을 위한 패턴 분석 컨텍스트 생성

    알고리즘 분석 결과를 LLM이 해석할 수 있는 형태로 포맷팅
    도메인 특화 지식 없이, LLM이 추론하도록 유도
    """

    def generate_context(
        self,
        patterns: Dict[str, ValuePattern],
        same_table_mappings: List[SameTableMapping],
        cross_table_matches: List[CrossTablePatternMatch],
    ) -> str:
        """LLM용 전체 컨텍스트 생성"""
        sections = []

        # 1. 패턴 기반 컬럼 그룹
        sections.append(self._generate_pattern_groups(patterns))

        # 2. Bridge Table 후보 (같은 테이블 내 매핑)
        sections.append(self._generate_bridge_table_section(same_table_mappings))

        # 3. 크로스 테이블 패턴 매칭
        sections.append(self._generate_cross_table_section(cross_table_matches))

        # 4. 분석 지침
        sections.append(self._generate_analysis_instructions())

        return "\n\n".join(sections)

    def _generate_pattern_groups(self, patterns: Dict[str, ValuePattern]) -> str:
        """패턴별 컬럼 그룹"""
        output = ["## 📊 Value Pattern Analysis"]
        output.append("각 컬럼의 값 패턴을 분석했습니다. 같은 패턴은 같은 도메인일 수 있습니다.\n")

        # 패턴별로 그룹화
        groups = defaultdict(list)
        for key, pattern in patterns.items():
            if pattern.is_likely_code or pattern.is_likely_identifier:
                group_key = f"{pattern.dominant_length}_{pattern.dominant_char_type}"
                groups[group_key].append((key, pattern))

        for group_key, members in sorted(groups.items(), key=lambda x: -len(x[1])):
            output.append(f"### Pattern: `{group_key}`")
            output.append(f"이 패턴을 가진 컬럼들 ({len(members)}개):\n")

            for key, pattern in members:
                output.append(f"- **{key}**")
                output.append(f"  - 샘플: {pattern.sample_values[:5]}")
                output.append(f"  - 구조: `{pattern.dominant_structure}`")
                output.append(f"  - Unique ratio: {pattern.unique_ratio:.2f}")
            output.append("")

        return "\n".join(output)

    def _generate_bridge_table_section(self, mappings: List[SameTableMapping]) -> str:
        """Bridge Table 후보 섹션"""
        output = ["## 🔗 Bridge Table Candidates (같은 테이블 내 1:1 매핑)"]

        if not mappings:
            output.append("발견된 Bridge Table 후보가 없습니다.")
            return "\n".join(output)

        output.append("다음 테이블들은 서로 다른 코드 체계를 매핑하는 역할을 할 수 있습니다.\n")

        for m in mappings:
            output.append(f"### {m.table}")
            output.append(f"- **{m.column_a}** (`{m.pattern_a}`) ↔ **{m.column_b}** (`{m.pattern_b}`)")
            output.append(f"- 매핑 수: {m.mapping_count}, Bijective: {m.is_bijective}")
            output.append(f"- 신뢰도: {m.confidence:.2f}")
            output.append(f"- 샘플 매핑:")
            for sa, sb in m.sample_mappings[:5]:
                output.append(f"  - `{sa}` ↔ `{sb}`")
            output.append("")

        return "\n".join(output)

    def _generate_cross_table_section(self, matches: List[CrossTablePatternMatch]) -> str:
        """크로스 테이블 패턴 매칭 섹션"""
        output = ["## 🔀 Cross-Table Pattern Matches"]

        if not matches:
            output.append("패턴이 유사한 크로스 테이블 컬럼이 없습니다.")
            return "\n".join(output)

        output.append("다른 테이블에서 유사한 패턴을 가진 컬럼들입니다.\n")

        # 값이 겹치는 것과 안 겹치는 것 분리
        with_overlap = [m for m in matches if m.has_value_overlap]
        without_overlap = [m for m in matches if not m.has_value_overlap]

        if with_overlap:
            output.append("### ✅ 값이 직접 겹치는 쌍 (Direct FK 후보)")
            for m in with_overlap:
                output.append(f"- {m.table_a}.{m.column_a} ↔ {m.table_b}.{m.column_b}")
                output.append(f"  - 패턴: `{m.pattern_match}`, Overlap: {m.overlap_ratio:.2%}")
            output.append("")

        if without_overlap:
            output.append("### ⚠️ 값은 안 겹치지만 패턴이 유사한 쌍 (간접 관계 가능)")
            output.append("**이 컬럼들은 같은 도메인의 다른 코드 체계일 수 있습니다.**")
            output.append("**Bridge Table을 통해 연결 가능한지 확인하세요.**\n")

            for m in without_overlap:
                output.append(f"- {m.table_a}.{m.column_a} (pattern: `{m.pattern_match}`)")
                output.append(f"  ↔ {m.table_b}.{m.column_b} (pattern: `{m.pattern_match}`)")
                output.append(f"  - Length match: {m.length_match}, Structure match: {m.structure_match}")
            output.append("")

        return "\n".join(output)

    def _generate_analysis_instructions(self) -> str:
        """LLM 분석 지침"""
        return """## 📝 분석 지침

위 패턴 분석 결과를 바탕으로 다음을 판단해주세요:

### 1. 같은 도메인의 다른 코드 체계 식별
- 패턴이 유사하지만 값이 다른 컬럼들이 있나요?
- 예: 2글자 대문자 vs 3글자 대문자가 같은 엔티티의 다른 표현일 수 있습니다
- Bridge Table의 매핑을 통해 연결 가능한지 확인하세요

### 2. 의미적 컬럼 그룹화
- 컬럼명과 값 패턴을 종합하여 같은 개념을 참조하는 컬럼들을 그룹화하세요
- 컬럼명이 달라도 같은 엔티티를 가리킬 수 있습니다

### 3. 간접 FK 관계 발견
- 직접 값이 매칭되지 않아도 Bridge Table을 통해 연결되는 관계가 있나요?
- Join Path를 제시해주세요

### 출력 형식
```json
{
    "semantic_column_groups": [
        {
            "concept": "식별된 개념 (예: 항공사 코드)",
            "columns": ["table1.col1", "table2.col2", ...],
            "code_systems": {
                "system_a": ["columns..."],
                "system_b": ["columns..."]
            },
            "bridge_table": "매핑 테이블명 또는 null",
            "reasoning": "판단 근거"
        }
    ],
    "indirect_relationships": [
        {
            "from": "table1.column1",
            "to": "table2.column2",
            "via": "bridge_table",
            "join_path": "table1.col1 → bridge.col_a ↔ bridge.col_b ← table2.col2",
            "reasoning": "판단 근거"
        }
    ]
}
```"""


# =============================================================================
# Main Analyzer (통합 진입점)
# =============================================================================

class EnhancedValuePatternAnalysis:
    """
    향상된 값 패턴 분석 (통합 진입점)

    사용법:
        analyzer = EnhancedValuePatternAnalysis()
        context = analyzer.analyze(tables_data)
        # context를 LLM 프롬프트에 포함
    """

    def __init__(self):
        self.pattern_analyzer = ValuePatternAnalyzer()
        self.mapping_detector = SameTableMappingDetector(self.pattern_analyzer)
        self.cross_matcher = CrossTablePatternMatcher()
        self.context_generator = PatternAnalysisContextGenerator()

    def analyze(
        self,
        tables_data: Dict[str, Dict[str, List[Any]]]
    ) -> Dict[str, Any]:
        """
        전체 분석 수행

        Returns:
            {
                "patterns": Dict[str, ValuePattern],
                "same_table_mappings": List[SameTableMapping],
                "cross_table_matches": List[CrossTablePatternMatch],
                "llm_context": str,  # LLM 프롬프트에 포함할 텍스트
            }
        """
        # 1. 모든 컬럼 패턴 분석
        patterns = self.pattern_analyzer.analyze_all_columns(tables_data)

        # 2. 같은 테이블 내 매핑 탐지 (Bridge Table 후보)
        same_table_mappings = self.mapping_detector.detect_mappings(tables_data, patterns)

        # 3. 크로스 테이블 패턴 매칭
        cross_table_matches = self.cross_matcher.find_pattern_matches(patterns, tables_data)

        # 4. LLM 컨텍스트 생성
        llm_context = self.context_generator.generate_context(
            patterns, same_table_mappings, cross_table_matches
        )

        return {
            "patterns": patterns,
            "same_table_mappings": same_table_mappings,
            "cross_table_matches": cross_table_matches,
            "llm_context": llm_context,
        }

    def get_patterns_summary(self, patterns: Dict[str, ValuePattern]) -> Dict[str, List[str]]:
        """패턴별 컬럼 요약"""
        summary = defaultdict(list)
        for key, pattern in patterns.items():
            if pattern.is_likely_code or pattern.is_likely_identifier:
                group = f"{pattern.dominant_length}_{pattern.dominant_char_type}"
                summary[group].append(key)
        return dict(summary)
