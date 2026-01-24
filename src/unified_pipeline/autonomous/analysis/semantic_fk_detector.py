"""
Semantic FK Detector v8.0 (LLM-First Discovery)

This module addresses the fundamental issues in FK detection:
1. LLM Schema Analyzer - LLM-First discovery of ALL FK candidates (v8.0 NEW)
2. Column Name Semantic Tokenizer - Extract table references from column names
3. Bridge Table Detector - Identify mapping tables with multiple code systems
4. Dynamic Abbreviation Learning - Learn abbreviation patterns from data
5. Value Pattern Correlator - Detect same-entity columns with different values
6. Enhanced LLM Prompt Generator - Provide comprehensive context to LLM
7. LLM Semantic Enhancer - LLM-based semantic role detection (v7.3)
8. LLM FK Direction Verifier - Verify/correct FK direction (v7.4)

v8.0 Changes (LLM-First Discovery Architecture):
- LLM Schema Analyzer: 전체 스키마를 LLM이 먼저 분석
- Self-Reference FK Detection: manager_id → emp_id, parent_dept → dept_id
- Single Noun FK Detection: trainer → employees, participant → employees
- Semantic Role FK Detection: head_of_dept → employees
- Rule-based 전에 LLM이 후보 발굴 → Rule-based가 검증

v7.4 Changes:
- LLM FK Direction Verifier: 전수 검증으로 방향 오류 교정
- CERTAIN 포함 모든 FK 후보 검증
- PK 사전 식별로 direction confusion 방지

v7.3 Changes:
- LLM Semantic Enhancer integration for universal FK detection
- Handles semantic roles like diagnosed_by, prescribing_doc, ordering_physician
- Optional LLM enhancement (enable_llm parameter)

Author: Ontoloty Team
Version: 8.0.0
"""

import re
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Any, TYPE_CHECKING
from collections import defaultdict
import difflib

logger = logging.getLogger(__name__)

# LLM Enhancer (optional)
try:
    from .llm_semantic_enhancer import LLMSemanticEnhancer
    _LLM_ENHANCER_AVAILABLE = True
except ImportError:
    _LLM_ENHANCER_AVAILABLE = False
    logger.debug("LLM Semantic Enhancer not available")

# LLM FK Direction Verifier (v7.4)
try:
    from .llm_fk_verifier import LLMFKDirectionVerifier
    _LLM_VERIFIER_AVAILABLE = True
except ImportError:
    _LLM_VERIFIER_AVAILABLE = False
    logger.debug("LLM FK Direction Verifier not available")

# LLM Schema Analyzer (v8.0 - LLM-First Discovery)
try:
    from .llm_schema_analyzer import LLMSchemaAnalyzer, DiscoveredFK, FKType
    _LLM_SCHEMA_ANALYZER_AVAILABLE = True
except ImportError:
    _LLM_SCHEMA_ANALYZER_AVAILABLE = False
    logger.debug("LLM Schema Analyzer not available")


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class SemanticToken:
    """Represents a semantic token extracted from a column name."""
    original: str
    normalized: str
    is_fk_suffix: bool = False
    potential_table_refs: List[str] = field(default_factory=list)
    potential_entity: Optional[str] = None


@dataclass
class ColumnSemantics:
    """Complete semantic analysis of a column name."""
    table_name: str
    column_name: str
    tokens: List[SemanticToken]
    potential_table_refs: List[str]
    potential_entity: Optional[str]
    fk_suffix: Optional[str]
    confidence: float


@dataclass
class BridgeTable:
    """Represents a bridge/mapping table that connects multiple code systems."""
    table_name: str
    code_columns: List[str]
    code_systems: Dict[str, List[str]]  # column -> sample values
    cardinality: int
    confidence: float
    description: str


@dataclass
class LearnedAbbreviation:
    """An abbreviation pattern learned from data."""
    short_form: str
    long_form: str
    evidence_pairs: List[Tuple[str, str]]  # (col_a, col_b) pairs
    confidence: float


@dataclass
class ValueCorrelation:
    """Correlation between two columns based on value patterns."""
    table_a: str
    column_a: str
    table_b: str
    column_b: str
    jaccard: float
    containment_a_in_b: float
    containment_b_in_a: float
    cardinality_a: int
    cardinality_b: int
    pattern_similarity: float
    is_strong_match: bool
    samples: List[str]


@dataclass
class SemanticFKCandidate:
    """A semantically-detected FK candidate with full context."""
    source_table: str
    source_column: str
    target_table: str
    target_column: str
    detection_method: str  # 'semantic', 'value_overlap', 'bridge', 'learned'
    confidence: float
    evidence: Dict[str, Any]
    llm_context: str


# =============================================================================
# Column Name Semantic Tokenizer
# =============================================================================

class ColumnSemanticTokenizer:
    """
    Extracts semantic meaning from column names.

    Addresses Issue #5: Column Name Table Reference Recognition

    Example:
        "assigned_airline" -> tokens: ["assigned", "airline"]
                          -> potential_table_refs: ["airlines", "airline"]
                          -> potential_entity: "airline"
    """

    # FK suffix patterns
    FK_SUFFIXES = {'_id', '_code', '_no', '_number', '_key', '_ref', '_fk'}

    # Common prefixes to strip
    PREFIXES_TO_STRIP = {'fk_', 'pk_', 'src_', 'dst_', 'from_', 'to_'}

    # Verb prefixes that indicate relationship
    RELATIONSHIP_VERBS = {
        'assigned', 'owned', 'managed', 'created', 'updated', 'handled',
        'processed', 'approved', 'reported', 'performed', 'belongs'
    }

    # Pluralization rules
    PLURAL_MAPPINGS = {
        'airlines': 'airline',
        'employees': 'employee',
        'passengers': 'passenger',
        'aircraft': 'aircraft',
        'gates': 'gate',
        'flights': 'flight',
        'delays': 'delay',
        'records': 'record',
        'maintenance': 'maintenance',
    }

    def __init__(self, table_names: List[str]):
        """
        Initialize with available table names for reference matching.

        Args:
            table_names: List of table names in the schema
        """
        self.table_names = set(table_names)
        self.table_name_variants = self._build_table_variants()

    def _build_table_variants(self) -> Dict[str, str]:
        """Build variants of table names for matching."""
        variants = {}
        for table in self.table_names:
            # Original
            variants[table.lower()] = table
            # Without underscores
            variants[table.lower().replace('_', '')] = table
            # Singular form
            singular = self._to_singular(table.lower())
            variants[singular] = table
            variants[singular.replace('_', '')] = table
        return variants

    def _to_singular(self, word: str) -> str:
        """Convert plural to singular."""
        for plural, singular in self.PLURAL_MAPPINGS.items():
            if word.endswith(plural):
                return word[:-len(plural)] + singular
        if word.endswith('ies'):
            return word[:-3] + 'y'
        if word.endswith('es'):
            return word[:-2]
        if word.endswith('s') and not word.endswith('ss'):
            return word[:-1]
        return word

    def tokenize(self, table_name: str, column_name: str) -> ColumnSemantics:
        """
        Tokenize a column name and extract semantic meaning.

        Args:
            table_name: The table containing this column
            column_name: The column name to analyze

        Returns:
            ColumnSemantics with full semantic analysis
        """
        # Normalize column name
        col_lower = column_name.lower()

        # Check for FK suffix
        fk_suffix = None
        base_name = col_lower
        for suffix in self.FK_SUFFIXES:
            if col_lower.endswith(suffix):
                fk_suffix = suffix
                base_name = col_lower[:-len(suffix)]
                break

        # Strip common prefixes
        for prefix in self.PREFIXES_TO_STRIP:
            if base_name.startswith(prefix):
                base_name = base_name[len(prefix):]
                break

        # Split into tokens
        raw_tokens = re.split(r'[_\s]+', base_name)

        # Analyze each token
        tokens = []
        potential_table_refs = []
        potential_entity = None
        relationship_verb = None

        for token in raw_tokens:
            if not token:
                continue

            semantic_token = SemanticToken(
                original=token,
                normalized=token,
                is_fk_suffix=(token in {'id', 'code', 'no', 'number', 'key', 'ref', 'fk'})
            )

            # Check if token matches a table name
            if token in self.table_name_variants:
                matched_table = self.table_name_variants[token]
                semantic_token.potential_table_refs = [matched_table]
                potential_table_refs.append(matched_table)
                if not potential_entity:
                    potential_entity = token

            # Check if token is a relationship verb
            if token in self.RELATIONSHIP_VERBS:
                relationship_verb = token

            tokens.append(semantic_token)

        # If no direct table match, try combining tokens
        if not potential_table_refs and len(tokens) >= 2:
            # Try "assigned_airline" -> "airline"
            for token in tokens:
                if token.original not in self.RELATIONSHIP_VERBS:
                    singular = self._to_singular(token.original)
                    if singular in self.table_name_variants:
                        potential_table_refs.append(self.table_name_variants[singular])
                        potential_entity = singular

        # Calculate confidence
        confidence = self._calculate_confidence(
            fk_suffix is not None,
            len(potential_table_refs) > 0,
            relationship_verb is not None
        )

        return ColumnSemantics(
            table_name=table_name,
            column_name=column_name,
            tokens=tokens,
            potential_table_refs=potential_table_refs,
            potential_entity=potential_entity,
            fk_suffix=fk_suffix,
            confidence=confidence
        )

    def _calculate_confidence(
        self,
        has_fk_suffix: bool,
        has_table_ref: bool,
        has_relationship_verb: bool
    ) -> float:
        """Calculate confidence score for semantic analysis."""
        score = 0.0
        if has_fk_suffix:
            score += 0.4
        if has_table_ref:
            score += 0.4
        if has_relationship_verb:
            score += 0.2
        return min(score, 1.0)

    def find_semantic_fk_candidates(
        self,
        all_columns: Dict[str, List[str]]
    ) -> List[SemanticFKCandidate]:
        """
        Find FK candidates based on column name semantics.

        Args:
            all_columns: Dict mapping table_name -> list of column names

        Returns:
            List of semantic FK candidates
        """
        candidates = []

        for table_name, columns in all_columns.items():
            for column in columns:
                semantics = self.tokenize(table_name, column)

                if semantics.potential_table_refs and semantics.confidence >= 0.4:
                    for target_table in semantics.potential_table_refs:
                        if target_table == table_name:
                            continue

                        # Find likely target column
                        target_columns = all_columns.get(target_table, [])
                        target_column = self._find_best_target_column(
                            semantics, target_columns
                        )

                        if target_column:
                            candidate = SemanticFKCandidate(
                                source_table=table_name,
                                source_column=column,
                                target_table=target_table,
                                target_column=target_column,
                                detection_method='semantic',
                                confidence=semantics.confidence,
                                evidence={
                                    'tokens': [t.original for t in semantics.tokens],
                                    'potential_entity': semantics.potential_entity,
                                    'fk_suffix': semantics.fk_suffix
                                },
                                llm_context=self._generate_llm_context(semantics, target_table, target_column)
                            )
                            candidates.append(candidate)

        return candidates

    def _find_best_target_column(
        self,
        semantics: ColumnSemantics,
        target_columns: List[str]
    ) -> Optional[str]:
        """Find the most likely target column in the reference table."""
        # Priority order: _id, _code, primary-looking columns
        priority_suffixes = ['_id', '_code', '_key', '_no', '_number']

        for suffix in priority_suffixes:
            for col in target_columns:
                if col.lower().endswith(suffix):
                    return col

        # Fallback: first column (often the PK)
        if target_columns:
            return target_columns[0]

        return None

    def _generate_llm_context(
        self,
        semantics: ColumnSemantics,
        target_table: str,
        target_column: str
    ) -> str:
        """Generate context string for LLM prompt."""
        return (
            f"Column '{semantics.column_name}' in table '{semantics.table_name}' "
            f"contains semantic reference to entity '{semantics.potential_entity}'. "
            f"This likely references '{target_table}.{target_column}'. "
            f"Evidence: tokens={[t.original for t in semantics.tokens]}, "
            f"fk_suffix={semantics.fk_suffix}"
        )


# =============================================================================
# Bridge Table Detector
# =============================================================================

class BridgeTableDetector:
    """
    Detects bridge/mapping tables that connect multiple code systems.

    Addresses Issue #4: No Bridge Table Detection

    Example:
        airlines table with iata_code and icao_code columns
        -> This is a bridge connecting IATA and ICAO code systems
    """

    # Code-like column patterns
    CODE_PATTERNS = [
        r'.*_code$',
        r'.*_id$',
        r'^code_.*',
        r'^id_.*',
        r'.*_key$'
    ]

    def __init__(self):
        self.code_pattern_regex = [re.compile(p, re.IGNORECASE) for p in self.CODE_PATTERNS]

    def detect_bridge_tables(
        self,
        tables_data: Dict[str, Any]
    ) -> List[BridgeTable]:
        """
        Detect bridge tables in the schema.

        Args:
            tables_data: Dict with table data including columns and sample values

        Returns:
            List of detected bridge tables
        """
        bridges = []

        for table_name, table_info in tables_data.items():
            columns = table_info.get('columns', [])
            data = table_info.get('data', [])

            if not columns or not data:
                continue

            # Find code-like columns
            code_columns = self._find_code_columns(columns, data)

            # A bridge table has 2+ code columns
            if len(code_columns) >= 2:
                # Extract sample values for each code column
                code_systems = {}
                for col in code_columns:
                    col_idx = columns.index(col) if col in columns else -1
                    if col_idx >= 0:
                        values = list(set(
                            str(row[col_idx]) for row in data[:100]
                            if row[col_idx] is not None
                        ))[:10]
                        code_systems[col] = values

                # Calculate confidence based on column characteristics
                confidence = self._calculate_bridge_confidence(
                    code_columns, code_systems, len(data)
                )

                if confidence >= 0.6:
                    bridge = BridgeTable(
                        table_name=table_name,
                        code_columns=code_columns,
                        code_systems=code_systems,
                        cardinality=len(data),
                        confidence=confidence,
                        description=self._generate_description(table_name, code_columns, code_systems)
                    )
                    bridges.append(bridge)

        return bridges

    def _find_code_columns(
        self,
        columns: List[str],
        data: List[Any]
    ) -> List[str]:
        """Find columns that look like code/identifier columns."""
        code_columns = []

        for i, col in enumerate(columns):
            # Check if column name matches code pattern
            is_code_pattern = any(p.match(col) for p in self.code_pattern_regex)

            # Check value characteristics
            if data:
                values = [str(row[i]) for row in data[:100] if row[i] is not None]
                if values:
                    avg_length = sum(len(v) for v in values) / len(values)
                    unique_ratio = len(set(values)) / len(values)
                    is_short_string = avg_length <= 10
                    is_high_unique = unique_ratio > 0.5

                    if is_code_pattern or (is_short_string and is_high_unique):
                        code_columns.append(col)

        return code_columns

    def _calculate_bridge_confidence(
        self,
        code_columns: List[str],
        code_systems: Dict[str, List[str]],
        cardinality: int
    ) -> float:
        """Calculate confidence that this is a bridge table."""
        score = 0.0

        # More code columns = higher confidence
        if len(code_columns) >= 2:
            score += 0.3
        if len(code_columns) >= 3:
            score += 0.1

        # Low cardinality = likely lookup/bridge table
        if cardinality <= 50:
            score += 0.3
        elif cardinality <= 200:
            score += 0.2

        # Consistent value lengths across code columns
        lengths = []
        for col, values in code_systems.items():
            if values:
                avg_len = sum(len(v) for v in values) / len(values)
                lengths.append(avg_len)

        if lengths and max(lengths) - min(lengths) <= 3:
            score += 0.2

        return min(score, 1.0)

    def _generate_description(
        self,
        table_name: str,
        code_columns: List[str],
        code_systems: Dict[str, List[str]]
    ) -> str:
        """Generate human-readable description of the bridge table."""
        code_desc = []
        for col, values in code_systems.items():
            sample = ', '.join(values[:3])
            code_desc.append(f"{col}=[{sample}...]")

        return (
            f"Table '{table_name}' is a bridge/mapping table connecting "
            f"{len(code_columns)} code systems: {', '.join(code_desc)}"
        )

    def generate_llm_context(self, bridges: List[BridgeTable]) -> str:
        """Generate LLM context for all detected bridge tables."""
        if not bridges:
            return ""

        context = "## BRIDGE/MAPPING TABLES DETECTED\n\n"
        context += "These tables connect multiple code systems. "
        context += "Consider INDIRECT relationships through these bridges:\n\n"

        for bridge in bridges:
            context += f"### {bridge.table_name} (Confidence: {bridge.confidence:.0%})\n"
            context += f"Code columns: {', '.join(bridge.code_columns)}\n"

            for col, values in bridge.code_systems.items():
                context += f"  - {col}: [{', '.join(values[:5])}]\n"

            context += f"\nUsage: If table A uses one code system and table B uses another, "
            context += f"they may be connected through '{bridge.table_name}'.\n\n"

        return context


# =============================================================================
# Dynamic Abbreviation Learner
# =============================================================================

class DynamicAbbreviationLearner:
    """
    Learns abbreviation patterns from data by finding columns with matching values.

    Addresses Issue #3: Abbreviation/Synonym Expansion Insufficient

    Example:
        passengers.booking_ref and baggage_handling.booking_reference
        have 100% value overlap -> learn that "ref" = "reference"
    """

    # Known abbreviation patterns (fallback)
    KNOWN_PATTERNS = {
        'ref': 'reference',
        'no': 'number',
        'num': 'number',
        'desc': 'description',
        'qty': 'quantity',
        'amt': 'amount',
        'dt': 'date',
        'tm': 'time',
        'addr': 'address',
        'tel': 'telephone',
        'info': 'information',
        'config': 'configuration',
        'msg': 'message',
        'stat': 'status',
    }

    def __init__(self):
        self.learned_patterns: Dict[str, str] = dict(self.KNOWN_PATTERNS)

    def learn_from_value_overlaps(
        self,
        high_overlap_pairs: List[ValueCorrelation]
    ) -> List[LearnedAbbreviation]:
        """
        Learn abbreviation patterns from columns with high value overlap.

        Args:
            high_overlap_pairs: List of column pairs with Jaccard >= 0.9

        Returns:
            List of newly learned abbreviation patterns
        """
        learned = []

        for pair in high_overlap_pairs:
            if pair.jaccard < 0.9:
                continue

            # Compare column names
            col_a_parts = self._split_column_name(pair.column_a)
            col_b_parts = self._split_column_name(pair.column_b)

            # Find differing parts
            new_patterns = self._find_abbreviation_patterns(col_a_parts, col_b_parts)

            for short, long in new_patterns:
                if short not in self.learned_patterns:
                    self.learned_patterns[short] = long

                    abbreviation = LearnedAbbreviation(
                        short_form=short,
                        long_form=long,
                        evidence_pairs=[
                            (f"{pair.table_a}.{pair.column_a}",
                             f"{pair.table_b}.{pair.column_b}")
                        ],
                        confidence=pair.jaccard
                    )
                    learned.append(abbreviation)

        return learned

    def _split_column_name(self, name: str) -> List[str]:
        """Split column name into tokens."""
        return [t.lower() for t in re.split(r'[_\s]+', name) if t]

    def _find_abbreviation_patterns(
        self,
        parts_a: List[str],
        parts_b: List[str]
    ) -> List[Tuple[str, str]]:
        """Find abbreviation patterns between two column name part lists."""
        patterns = []

        # Use sequence matcher to align parts
        matcher = difflib.SequenceMatcher(None, parts_a, parts_b)

        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == 'replace' and (i2 - i1) == 1 and (j2 - j1) == 1:
                # Single word replacement - potential abbreviation
                short = parts_a[i1]
                long = parts_b[j1]

                # Short should be shorter than long
                if len(short) < len(long) and long.startswith(short[:2]):
                    patterns.append((short, long))
                elif len(long) < len(short) and short.startswith(long[:2]):
                    patterns.append((long, short))

        return patterns

    def expand_column_name(self, column_name: str) -> str:
        """Expand abbreviations in a column name."""
        parts = self._split_column_name(column_name)
        expanded = []

        for part in parts:
            if part in self.learned_patterns:
                expanded.append(self.learned_patterns[part])
            else:
                expanded.append(part)

        return '_'.join(expanded)

    def get_all_patterns(self) -> Dict[str, str]:
        """Get all known abbreviation patterns."""
        return dict(self.learned_patterns)


# =============================================================================
# Value Pattern Correlator
# =============================================================================

class ValuePatternCorrelator:
    """
    Correlates columns based on value patterns, even without direct overlap.

    Addresses Issue #2: Value Overlap Detection Issues
    """

    def compute_correlations(
        self,
        tables_data: Dict[str, Any],
        min_jaccard: float = 0.3
    ) -> List[ValueCorrelation]:
        """
        Compute value correlations between all column pairs.

        Args:
            tables_data: Dict with table data
            min_jaccard: Minimum Jaccard similarity to report

        Returns:
            List of value correlations
        """
        correlations = []

        # Extract column values
        column_values = {}
        for table_name, table_info in tables_data.items():
            columns = table_info.get('columns', [])
            data = table_info.get('data', [])

            for i, col in enumerate(columns):
                values = set(
                    str(row[i]) for row in data
                    if row[i] is not None and str(row[i]).strip()
                )
                if values:
                    column_values[(table_name, col)] = values

        # Compare all pairs
        keys = list(column_values.keys())
        for i, (table_a, col_a) in enumerate(keys):
            values_a = column_values[(table_a, col_a)]

            for table_b, col_b in keys[i+1:]:
                if table_a == table_b:
                    continue

                values_b = column_values[(table_b, col_b)]

                # Compute metrics
                intersection = values_a & values_b
                union = values_a | values_b

                if not union:
                    continue

                jaccard = len(intersection) / len(union)
                containment_a = len(intersection) / len(values_a) if values_a else 0
                containment_b = len(intersection) / len(values_b) if values_b else 0

                if jaccard >= min_jaccard or max(containment_a, containment_b) >= 0.8:
                    pattern_sim = self._compute_pattern_similarity(values_a, values_b)

                    correlation = ValueCorrelation(
                        table_a=table_a,
                        column_a=col_a,
                        table_b=table_b,
                        column_b=col_b,
                        jaccard=jaccard,
                        containment_a_in_b=containment_a,
                        containment_b_in_a=containment_b,
                        cardinality_a=len(values_a),
                        cardinality_b=len(values_b),
                        pattern_similarity=pattern_sim,
                        is_strong_match=(jaccard >= 0.8 or max(containment_a, containment_b) >= 0.9),
                        samples=list(intersection)[:5]
                    )
                    correlations.append(correlation)

        # Sort by strength
        correlations.sort(key=lambda x: (x.is_strong_match, x.jaccard), reverse=True)

        return correlations

    def _compute_pattern_similarity(
        self,
        values_a: Set[str],
        values_b: Set[str]
    ) -> float:
        """Compute pattern similarity between two value sets."""
        if not values_a or not values_b:
            return 0.0

        # Compare value length distributions
        lens_a = [len(v) for v in values_a]
        lens_b = [len(v) for v in values_b]

        avg_len_a = sum(lens_a) / len(lens_a)
        avg_len_b = sum(lens_b) / len(lens_b)

        len_sim = 1 - abs(avg_len_a - avg_len_b) / max(avg_len_a, avg_len_b, 1)

        # Compare character type distributions
        def char_types(values):
            alpha, digit, other = 0, 0, 0
            for v in values:
                for c in v:
                    if c.isalpha():
                        alpha += 1
                    elif c.isdigit():
                        digit += 1
                    else:
                        other += 1
            total = alpha + digit + other
            if total == 0:
                return (0, 0, 0)
            return (alpha/total, digit/total, other/total)

        types_a = char_types(values_a)
        types_b = char_types(values_b)

        type_sim = 1 - sum(abs(a - b) for a, b in zip(types_a, types_b)) / 2

        return (len_sim + type_sim) / 2

    def generate_llm_context(
        self,
        correlations: List[ValueCorrelation],
        top_n: int = 20
    ) -> str:
        """Generate LLM context for value correlations."""
        if not correlations:
            return ""

        context = "## STRONG VALUE MATCHES (Likely FK Relationships)\n\n"
        context += "The following column pairs have high value overlap:\n\n"

        strong = [c for c in correlations if c.is_strong_match][:top_n]

        for i, corr in enumerate(strong, 1):
            context += f"### Match {i}: {corr.table_a}.{corr.column_a} ↔ {corr.table_b}.{corr.column_b}\n"
            context += f"- **Jaccard Similarity**: {corr.jaccard:.1%}\n"
            context += f"- **Containment A→B**: {corr.containment_a_in_b:.1%}\n"
            context += f"- **Containment B→A**: {corr.containment_b_in_a:.1%}\n"
            context += f"- **Sample overlapping values**: {corr.samples}\n"
            context += f"- **This is a STRONG candidate for FK relationship.**\n\n"

        # Also show moderate matches
        moderate = [c for c in correlations if not c.is_strong_match][:10]
        if moderate:
            context += "\n## MODERATE VALUE MATCHES (Investigate Further)\n\n"
            for corr in moderate:
                context += f"- {corr.table_a}.{corr.column_a} ↔ {corr.table_b}.{corr.column_b} "
                context += f"(Jaccard: {corr.jaccard:.1%})\n"

        return context


# =============================================================================
# Integrated Semantic FK Detector
# =============================================================================

@dataclass
class EntitySynonym:
    """Represents entity synonyms discovered from value overlap."""
    entity_a: str
    entity_b: str
    evidence_table_a: str
    evidence_column_a: str
    evidence_table_b: str
    evidence_column_b: str
    jaccard: float
    explanation: str


@dataclass
class IndirectRelationship:
    """Represents an indirect FK relationship through a bridge table."""
    source_table: str
    source_column: str
    source_code_system: str
    target_table: str
    target_column: str
    target_code_system: str
    bridge_table: str
    join_path: str
    confidence: float


class SemanticFKDetector:
    """
    Main class integrating all v7.1/v7.2/v7.3 FK detection improvements.

    v7.3 Changes:
    - LLM Semantic Enhancer integration for universal FK detection
    - Handles semantic roles (diagnosed_by, prescribing_doc, etc.)
    - Optional LLM enhancement (enable_llm parameter)

    v7.2 Changes:
    - Prescriptive LLM prompts (MANDATORY FK instructions)
    - Entity Synonym Detection
    - Bridge Table Indirect Relationship Hints
    - Confidence Boost with Value Evidence
    """

    def __init__(
        self,
        tables_data: Dict[str, Any],
        enable_llm: bool = True,
        llm_min_confidence: float = 0.6,
        domain_context: str = "",
    ):
        """
        Initialize with schema and data.

        Args:
            tables_data: Dict with table data (columns, data, etc.)
            enable_llm: Enable LLM semantic enhancement (v7.3)
            llm_min_confidence: Minimum confidence for LLM results
            domain_context: Domain context (e.g., "healthcare", "aviation")
        """
        self.tables_data = tables_data
        self.table_names = list(tables_data.keys())
        self.domain_context = domain_context

        # Initialize components
        self.tokenizer = ColumnSemanticTokenizer(self.table_names)
        self.bridge_detector = BridgeTableDetector()
        self.abbreviation_learner = DynamicAbbreviationLearner()
        self.value_correlator = ValuePatternCorrelator()

        # v7.3: LLM Semantic Enhancer (optional)
        self.enable_llm = enable_llm and _LLM_ENHANCER_AVAILABLE
        self.llm_min_confidence = llm_min_confidence
        self._llm_enhancer = None

        if self.enable_llm:
            try:
                self._llm_enhancer = LLMSemanticEnhancer(
                    use_instructor=True,
                    use_dspy=False,
                    min_confidence=llm_min_confidence,
                )
                logger.info("SemanticFKDetector: LLM Enhancer initialized")
            except Exception as e:
                logger.warning(f"SemanticFKDetector: LLM Enhancer init failed: {e}")
                self.enable_llm = False

        # v7.4: LLM FK Direction Verifier (전수 검증)
        self._fk_verifier = None
        if self.enable_llm and _LLM_VERIFIER_AVAILABLE:
            try:
                self._fk_verifier = LLMFKDirectionVerifier(
                    use_instructor=True,
                    use_dspy=False,
                )
                logger.info("SemanticFKDetector: FK Direction Verifier initialized")
            except Exception as e:
                logger.warning(f"SemanticFKDetector: FK Verifier init failed: {e}")

        # v8.0: LLM Schema Analyzer (LLM-First Discovery)
        self._schema_analyzer = None
        if self.enable_llm and _LLM_SCHEMA_ANALYZER_AVAILABLE:
            try:
                self._schema_analyzer = LLMSchemaAnalyzer(
                    use_instructor=True,
                    use_dspy=False,
                    min_confidence=llm_min_confidence,
                )
                logger.info("SemanticFKDetector: Schema Analyzer initialized (LLM-First Discovery)")
            except Exception as e:
                logger.warning(f"SemanticFKDetector: Schema Analyzer init failed: {e}")

        # Results
        self.semantic_candidates: List[SemanticFKCandidate] = []
        self.bridge_tables: List[BridgeTable] = []
        self.value_correlations: List[ValueCorrelation] = []
        self.learned_abbreviations: List[LearnedAbbreviation] = []

        # v7.2: Additional results
        self.entity_synonyms: List[EntitySynonym] = []
        self.indirect_relationships: List[IndirectRelationship] = []
        self.mandatory_fks: List[Dict[str, Any]] = []

        # v7.3: LLM enhanced FKs
        self.llm_enhanced_fks: List[Dict[str, Any]] = []

        # v8.0: LLM-First Discovery results
        self.llm_discovered_fks: List[Dict[str, Any]] = []
        self.hierarchy_tables: Dict[str, Dict[str, Any]] = {}

    def analyze(self) -> Dict[str, Any]:
        """
        Run full semantic FK analysis.

        Returns:
            Dict with all analysis results
        """
        # v8.0 Step 0: LLM-First Discovery (전체 스키마 LLM 분석)
        if self._schema_analyzer:
            self.llm_discovered_fks, self.hierarchy_tables = self._llm_first_discovery()
            logger.info(f"LLM-First Discovery: {len(self.llm_discovered_fks)} FK candidates found")

        # Step 1: Detect bridge tables
        self.bridge_tables = self.bridge_detector.detect_bridge_tables(self.tables_data)

        # Step 2: Compute value correlations
        self.value_correlations = self.value_correlator.compute_correlations(
            self.tables_data, min_jaccard=0.3
        )

        # Step 3: Learn abbreviation patterns from high-overlap pairs
        high_overlap = [c for c in self.value_correlations if c.jaccard >= 0.9]
        self.learned_abbreviations = self.abbreviation_learner.learn_from_value_overlaps(
            high_overlap
        )

        # Step 4: Find semantic FK candidates from column names
        all_columns = {}
        for table_name, table_info in self.tables_data.items():
            all_columns[table_name] = table_info.get('columns', [])

        self.semantic_candidates = self.tokenizer.find_semantic_fk_candidates(all_columns)

        # v7.2 Step 5: Detect entity synonyms from value overlap
        self.entity_synonyms = self._detect_entity_synonyms()

        # v7.2 Step 6: Find indirect relationships through bridge tables
        self.indirect_relationships = self._find_indirect_relationships()

        # v7.2 Step 7: Boost semantic candidate confidence with value evidence
        self._boost_confidence_with_value_evidence()

        # v7.2 Step 8: Build mandatory FK list
        self.mandatory_fks = self._build_mandatory_fk_list()

        # v7.3 Step 9: LLM Semantic Enhancement
        if self.enable_llm and self._llm_enhancer:
            self.llm_enhanced_fks = self._enhance_with_llm()
            logger.info(f"LLM enhanced {len(self.llm_enhanced_fks)} FK candidates")

        # v7.4 Step 10: LLM FK Direction Verification (전수 검증)
        self.verified_fks = []
        if self._fk_verifier and self.mandatory_fks:
            self.verified_fks = self._verify_fk_directions()
            logger.info(f"Verified {len(self.verified_fks)} FK candidates")

        return {
            'bridge_tables': self.bridge_tables,
            'value_correlations': self.value_correlations,
            'learned_abbreviations': self.learned_abbreviations,
            'semantic_candidates': self.semantic_candidates,
            'entity_synonyms': self.entity_synonyms,
            'indirect_relationships': self.indirect_relationships,
            'mandatory_fks': self.mandatory_fks,
            'llm_enhanced_fks': self.llm_enhanced_fks,  # v7.3
            'verified_fks': self.verified_fks,  # v7.4
            'llm_discovered_fks': self.llm_discovered_fks,  # v8.0
            'hierarchy_tables': self.hierarchy_tables,  # v8.0
        }

    def _llm_first_discovery(self) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, Any]]]:
        """
        v8.0: LLM-First Discovery - 전체 스키마를 LLM이 먼저 분석

        Rule-based 탐지 전에 LLM이 전체 스키마를 분석하여:
        1. Self-Reference FK: manager_id → emp_id
        2. Single Noun FK: trainer → employees
        3. Semantic Role FK: head_of_dept → employees

        Returns:
            (discovered_fks, hierarchy_tables) 튜플
        """
        discovered_fks = []
        hierarchy_tables = {}

        if not self._schema_analyzer:
            return discovered_fks, hierarchy_tables

        try:
            # Step 1: Discover all FK candidates using LLM
            raw_discoveries = self._schema_analyzer.analyze_schema(
                self.tables_data,
                domain_context=self.domain_context,
            )

            # Convert DiscoveredFK to dict format
            for fk in raw_discoveries:
                fk_dict = {
                    'source_table': fk.source_table,
                    'source_column': fk.source_column,
                    'target_table': fk.target_table,
                    'target_column': fk.target_column,
                    'confidence': fk.confidence,
                    'fk_type': fk.fk_type.value if hasattr(fk.fk_type, 'value') else str(fk.fk_type),
                    'reasoning': fk.reasoning,
                    'semantic_relationship': fk.semantic_relationship,
                    'method': 'llm_first_discovery',
                    'is_validated': fk.is_validated,
                }
                discovered_fks.append(fk_dict)

                # Log self-reference and single-noun discoveries
                if fk.fk_type in [FKType.SELF_REFERENCE, FKType.SINGLE_NOUN, FKType.HIERARCHICAL]:
                    logger.info(
                        f"LLM-First Discovery [{fk.fk_type.value}]: "
                        f"{fk.source_table}.{fk.source_column} → "
                        f"{fk.target_table}.{fk.target_column} "
                        f"(conf: {fk.confidence:.2f})"
                    )

            # Step 2: Detect hierarchy structures
            hierarchy_tables = self._schema_analyzer.detect_hierarchies(self.tables_data)

            if hierarchy_tables:
                logger.info(f"Detected {len(hierarchy_tables)} hierarchical tables: {list(hierarchy_tables.keys())}")

        except Exception as e:
            logger.warning(f"LLM-First Discovery failed: {e}")

        return discovered_fks, hierarchy_tables

    def _enhance_with_llm(self) -> List[Dict[str, Any]]:
        """
        v7.3: LLM-based semantic FK enhancement.

        Finds FKs that rule-based methods missed, especially:
        - Semantic role patterns: diagnosed_by, prescribing_doc, attending_doc
        - Domain-specific synonyms: carrier→airline, member→patient

        Returns:
            List of LLM-enhanced FK candidates
        """
        enhanced_fks = []

        # Get all FK-like columns that might need LLM analysis
        # v7.4: Extended for multi-domain support (healthcare, ecommerce, aviation, etc.)
        semantic_role_patterns = [
            # Healthcare
            "_by", "_doc", "_physician", "_nurse", "_staff",
            "_manager", "_owner", "_handler", "_agent",
            "assigned_", "primary_", "secondary_",
            "attending_", "ordering_", "prescribing_", "diagnosed_",
            # Ecommerce
            "buyer_", "seller_", "vendor_", "customer_",
            "reviewed_", "shipped_", "purchased_", "applied_",
            "parent_",  # for hierarchical categories
            # Aviation
            "carrier", "member", "pilot_", "crew_",
            # General
            "_ref", "_code", "_no",  # common FK suffixes
        ]

        for table_name, table_info in self.tables_data.items():
            columns = table_info.get('columns', [])
            if isinstance(columns, list) and columns and isinstance(columns[0], dict):
                col_names = [c.get('name', '') for c in columns]
            else:
                col_names = columns if isinstance(columns, list) else []

            for col_name in col_names:
                col_lower = col_name.lower()

                # Check if column matches semantic role patterns
                needs_llm = any(pattern in col_lower for pattern in semantic_role_patterns)

                # Also check if already detected by rule-based methods
                already_detected = any(
                    c.source_column == col_name and c.source_table == table_name
                    for c in self.semantic_candidates
                    if c.confidence >= 0.7
                )

                if needs_llm and not already_detected:
                    # Find potential target tables (entities like doctors, patients, etc.)
                    potential_targets = self._find_potential_targets_for_column(col_name)

                    for target_table, target_column in potential_targets:
                        # Skip self-references (same table/column)
                        if table_name == target_table and col_name == target_column:
                            continue

                        try:
                            # Get sample values
                            fk_samples = self._get_column_samples(table_name, col_name)
                            pk_samples = self._get_column_samples(target_table, target_column)

                            if not fk_samples or not pk_samples:
                                continue

                            # Calculate overlap
                            fk_set = set(str(v) for v in fk_samples if v)
                            pk_set = set(str(v) for v in pk_samples if v)
                            overlap = len(fk_set & pk_set) / len(fk_set) if fk_set else 0.0

                            # Skip if no value overlap at all
                            if overlap < 0.1:
                                continue

                            # LLM analysis
                            result = self._llm_enhancer.analyze_semantic_fk(
                                fk_column=col_name,
                                fk_table=table_name,
                                pk_column=target_column,
                                pk_table=target_table,
                                fk_samples=[str(v) for v in fk_samples[:15]],
                                pk_samples=[str(v) for v in pk_samples[:15]],
                                overlap_ratio=overlap,
                                domain_context=self.domain_context,
                            )

                            if result.is_relationship and result.confidence >= self.llm_min_confidence:
                                enhanced_fks.append({
                                    'source_table': table_name,
                                    'source_column': col_name,
                                    'target_table': target_table,
                                    'target_column': target_column,
                                    'confidence': result.confidence,
                                    'cardinality': result.cardinality,
                                    'semantic_relationship': result.semantic_relationship,
                                    'reasoning': result.reasoning,
                                    'method': 'llm_semantic_enhancer',
                                    'value_overlap': overlap,
                                })

                                logger.info(
                                    f"LLM detected FK: {table_name}.{col_name} → "
                                    f"{target_table}.{target_column} (conf: {result.confidence:.2f})"
                                )

                        except Exception as e:
                            logger.warning(f"LLM analysis failed for {table_name}.{col_name}: {e}")

        return enhanced_fks

    def _verify_fk_directions(self) -> List[Dict[str, Any]]:
        """
        v7.4: LLM 기반 FK 방향 검증.

        Rule-based FK Detection의 방향 오류를 전수 검증하고 교정합니다.
        CERTAIN 포함 모든 FK 후보를 검증합니다.

        문제 해결:
        - orders.order_id → shipping.order_no (역방향)
        - shipping.order_no → orders.order_id (올바른 방향)

        Returns:
            검증/교정된 FK 리스트
        """
        if not self._fk_verifier:
            return []

        # Combine all FK candidates for verification
        all_fks = []

        # Add mandatory FKs
        for fk in self.mandatory_fks:
            # Convert to standard format
            if isinstance(fk, dict):
                # Format 1: source/target as "table.column" strings
                if 'source' in fk and '.' in str(fk.get('source', '')):
                    source_parts = fk['source'].split('.')
                    target_parts = fk['target'].split('.')
                    if len(source_parts) == 2 and len(target_parts) == 2:
                        all_fks.append({
                            'source_table': source_parts[0],
                            'source_column': source_parts[1],
                            'target_table': target_parts[0],
                            'target_column': target_parts[1],
                            'confidence': 0.9 if fk.get('priority') == 'CRITICAL' else 0.7,
                            'confidence_level': fk.get('priority', 'unknown'),
                            'detection_method': 'rule_based',
                        })
                        continue

                # Format 2: explicit table/column fields
                all_fks.append({
                    'source_table': fk.get('fk_table', fk.get('source_table', '')),
                    'source_column': fk.get('fk_column', fk.get('source_column', '')),
                    'target_table': fk.get('pk_table', fk.get('target_table', '')),
                    'target_column': fk.get('pk_column', fk.get('target_column', '')),
                    'confidence': fk.get('fk_score', fk.get('confidence', 0)),
                    'confidence_level': fk.get('confidence_level', 'unknown'),
                    'detection_method': 'rule_based',
                })

        # Add LLM enhanced FKs (already in correct format)
        for fk in self.llm_enhanced_fks:
            fk_copy = dict(fk)
            fk_copy['detection_method'] = 'llm_semantic_enhancer'
            all_fks.append(fk_copy)

        if not all_fks:
            return []

        # Verify all FK directions
        verified = self._fk_verifier.verify_all_fks(
            fk_candidates=all_fks,
            tables_data=self.tables_data,
            domain_context=self.domain_context,
        )

        # Log corrections
        corrections = [fk for fk in verified if fk.get('was_corrected')]
        if corrections:
            logger.info(f"FK Direction Verifier corrected {len(corrections)} FKs:")
            for fk in corrections:
                logger.info(
                    f"  Corrected: {fk['source_table']}.{fk['source_column']} → "
                    f"{fk['target_table']}.{fk['target_column']} "
                    f"(reason: {fk.get('correction_reason', 'unknown')[:50]})"
                )

        return verified

    def _find_potential_targets_for_column(self, col_name: str) -> List[Tuple[str, str]]:
        """Find potential target tables for a column based on naming hints.

        v7.4: Extended for multi-domain support (healthcare, ecommerce, aviation, etc.)
        """
        targets = []
        col_lower = col_name.lower()

        # Semantic role hints → target entities (multi-domain)
        role_hints = {
            # Healthcare entities
            'doctor': ['doctors', 'physician', 'staff'],
            'patient': ['patients', 'member', 'customer'],
            'nurse': ['nurses', 'staff'],
            'staff': ['staff', 'employee'],
            # Ecommerce entities
            'customer': ['customers', 'users', 'members', 'buyers'],
            'vendor': ['vendors', 'sellers', 'suppliers', 'merchants'],
            'buyer': ['customers', 'users', 'buyers'],
            'seller': ['vendors', 'sellers', 'suppliers'],
            'product': ['products', 'items', 'goods'],
            'category': ['categories', 'types', 'groups'],
            'order': ['orders', 'purchases'],
            'promo': ['promotions', 'promos', 'discounts', 'coupons'],
            # Aviation entities
            'airline': ['airlines', 'carrier'],
            'flight': ['flights'],
            'pilot': ['pilots', 'crew', 'staff'],
            'passenger': ['passengers', 'customers', 'travelers'],
        }

        # Check which entity the column might reference
        for entity, table_hints in role_hints.items():
            if entity in col_lower or any(h in col_lower for h in table_hints):
                # Find matching tables
                for table_name in self.table_names:
                    table_lower = table_name.lower()
                    if any(h in table_lower for h in table_hints) or entity in table_lower:
                        # Find PK column in target table
                        pk_col = self._find_pk_column(table_name)
                        if pk_col:
                            targets.append((table_name, pk_col))

        # Role-based patterns for common suffixes
        role_suffix_hints = {
            # _by suffix typically references person/actor entities
            '_by': ['customers', 'vendors', 'users', 'doctors', 'staff', 'employees'],
            '_doc': ['doctors', 'physicians'],
            '_physician': ['doctors', 'physicians'],
        }

        for suffix, table_hints in role_suffix_hints.items():
            if suffix in col_lower:
                for table_name in self.table_names:
                    table_lower = table_name.lower()
                    if any(h in table_lower for h in table_hints):
                        pk_col = self._find_pk_column(table_name)
                        if pk_col and (table_name, pk_col) not in targets:
                            targets.append((table_name, pk_col))

        # General _ref, _code, _no patterns - search all tables
        if '_ref' in col_lower or '_code' in col_lower or '_no' in col_lower:
            # Extract entity hint from column name (e.g., ord_ref -> order)
            entity_hint = col_lower.replace('_ref', '').replace('_code', '').replace('_no', '')
            for table_name in self.table_names:
                if entity_hint[:3] in table_name.lower():  # Fuzzy match
                    pk_col = self._find_pk_column(table_name)
                    if pk_col and (table_name, pk_col) not in targets:
                        targets.append((table_name, pk_col))

        return targets

    def _find_pk_column(self, table_name: str) -> Optional[str]:
        """Find the primary key column of a table."""
        table_info = self.tables_data.get(table_name, {})
        columns = table_info.get('columns', [])

        if isinstance(columns, list) and columns and isinstance(columns[0], dict):
            col_names = [c.get('name', '') for c in columns]
        else:
            col_names = columns if isinstance(columns, list) else []

        # Common PK patterns
        table_singular = table_name.rstrip('s')
        pk_patterns = [
            f'{table_singular}_id',
            f'{table_name}_id',
            'id',
            f'{table_singular}_code',
        ]

        for pattern in pk_patterns:
            for col in col_names:
                if col.lower() == pattern.lower():
                    return col

        # Fallback: first column ending with _id
        for col in col_names:
            if col.lower().endswith('_id'):
                return col

        return None

    def _get_column_samples(self, table_name: str, col_name: str) -> List[Any]:
        """Get sample values from a column."""
        table_info = self.tables_data.get(table_name, {})

        # Try different data formats

        # Format 1: List of dicts (sample_data)
        sample_data = table_info.get('sample_data', [])
        if isinstance(sample_data, list) and sample_data:
            if isinstance(sample_data[0], dict):
                return [row.get(col_name) for row in sample_data[:50] if row.get(col_name)]

        # Format 2: columns + data (list of lists/tuples)
        columns = table_info.get('columns', [])
        data = table_info.get('data', [])

        if columns and data:
            # Get column names (handle both string list and dict list)
            if isinstance(columns, list) and columns:
                if isinstance(columns[0], dict):
                    col_names = [c.get('name', '') for c in columns]
                else:
                    col_names = columns

                # Find column index
                col_lower = col_name.lower()
                col_idx = None
                for idx, cn in enumerate(col_names):
                    if cn.lower() == col_lower:
                        col_idx = idx
                        break

                if col_idx is not None and isinstance(data, list):
                    samples = []
                    for row in data[:50]:
                        if isinstance(row, (list, tuple)) and col_idx < len(row):
                            val = row[col_idx]
                            if val is not None and str(val).strip():
                                samples.append(val)
                    return samples

        return []

    def _extract_entity_name(self, column_name: str) -> str:
        """Extract entity name from column name."""
        # Remove common suffixes
        col = column_name.lower()
        for suffix in ['_id', '_code', '_no', '_number', '_key', '_ref']:
            if col.endswith(suffix):
                col = col[:-len(suffix)]
                break
        # Get last meaningful token
        tokens = [t for t in re.split(r'[_\s]+', col) if t]
        return tokens[-1] if tokens else col

    def _detect_entity_synonyms(self) -> List[EntitySynonym]:
        """
        v7.2: Detect entity synonyms from high value overlap.

        When two columns have 90%+ Jaccard but different names,
        they likely refer to the same entity with different terminology.
        """
        synonyms = []

        for corr in self.value_correlations:
            if corr.jaccard < 0.9:
                continue

            entity_a = self._extract_entity_name(corr.column_a)
            entity_b = self._extract_entity_name(corr.column_b)

            # Different names but same values = synonyms
            if entity_a != entity_b:
                explanation = (
                    f"'{entity_a}' and '{entity_b}' are SYNONYMS - "
                    f"100% value overlap ({corr.samples[:3]}). "
                    f"They refer to the SAME entity using different terminology."
                )

                synonym = EntitySynonym(
                    entity_a=entity_a,
                    entity_b=entity_b,
                    evidence_table_a=corr.table_a,
                    evidence_column_a=corr.column_a,
                    evidence_table_b=corr.table_b,
                    evidence_column_b=corr.column_b,
                    jaccard=corr.jaccard,
                    explanation=explanation
                )
                synonyms.append(synonym)

        return synonyms

    def _find_indirect_relationships(self) -> List[IndirectRelationship]:
        """
        v7.2: Find indirect relationships through bridge tables.

        If column A matches Bridge.code1 and column B matches Bridge.code2,
        A and B are related through the bridge.
        """
        indirect = []

        for bridge in self.bridge_tables:
            # Map each code column to tables that use it
            code_to_tables: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
            code_values = {col: set(bridge.code_systems.get(col, [])) for col in bridge.code_columns}

            for corr in self.value_correlations:
                sample_set = set(corr.samples) if corr.samples else set()

                for code_col, values in code_values.items():
                    if sample_set and sample_set & values:
                        # This column matches this code system
                        code_to_tables[code_col].append((corr.table_a, corr.column_a))
                        code_to_tables[code_col].append((corr.table_b, corr.column_b))

            # Find pairs connected through different code systems
            code_cols = list(code_to_tables.keys())
            for i, code_a in enumerate(code_cols):
                for code_b in code_cols[i+1:]:
                    for table_a, col_a in code_to_tables[code_a]:
                        for table_b, col_b in code_to_tables[code_b]:
                            if table_a != table_b and table_a != bridge.table_name and table_b != bridge.table_name:
                                join_path = (
                                    f"{table_a}.{col_a} → {bridge.table_name}.{code_a} "
                                    f"↔ {bridge.table_name}.{code_b} ← {table_b}.{col_b}"
                                )

                                rel = IndirectRelationship(
                                    source_table=table_a,
                                    source_column=col_a,
                                    source_code_system=code_a,
                                    target_table=table_b,
                                    target_column=col_b,
                                    target_code_system=code_b,
                                    bridge_table=bridge.table_name,
                                    join_path=join_path,
                                    confidence=0.8
                                )
                                indirect.append(rel)

        return indirect

    def _boost_confidence_with_value_evidence(self):
        """
        v7.2: Boost semantic candidate confidence when value evidence exists.
        """
        for cand in self.semantic_candidates:
            for corr in self.value_correlations:
                # Check if this semantic candidate has value overlap evidence
                if ((cand.source_table == corr.table_a and cand.target_table == corr.table_b) or
                    (cand.source_table == corr.table_b and cand.target_table == corr.table_a)):
                    # Value evidence exists - boost confidence!
                    boost = min(0.3, corr.jaccard * 0.4)
                    cand.confidence = min(1.0, cand.confidence + boost)
                    cand.evidence['value_boost'] = corr.jaccard
                    cand.evidence['boosted'] = True

    def _build_mandatory_fk_list(self) -> List[Dict[str, Any]]:
        """
        v8.0: Build list of MANDATORY FK relationships based on strong evidence.

        Includes:
        - v8.0 LLM-First Discovery results (self-reference, single noun, etc.)
        - Strong value matches (Jaccard >= 80%)
        - High-confidence semantic candidates
        """
        mandatory = []
        seen = set()

        # v8.0: Add LLM-First Discovery FKs (highest priority)
        for fk in self.llm_discovered_fks:
            key = (fk['source_table'], fk['source_column'], fk['target_table'], fk['target_column'])
            if key not in seen:
                seen.add(key)
                fk_type = fk.get('fk_type', 'standard')
                priority = 'CRITICAL' if fk_type in ['self_reference', 'hierarchical'] else 'HIGH'

                mandatory.append({
                    'source': f"{fk['source_table']}.{fk['source_column']}",
                    'target': f"{fk['target_table']}.{fk['target_column']}",
                    'evidence': f"LLM-First Discovery ({fk_type}): {fk.get('reasoning', '')[:50]}",
                    'priority': priority,
                    'type': 'llm_first_discovery',
                    'fk_type': fk_type,
                })

        # Strong value matches (Jaccard >= 80%)
        for corr in self.value_correlations:
            if corr.jaccard >= 0.8:
                # Skip boolean/flag columns
                if set(corr.samples) <= {'True', 'False', 'true', 'false', '0', '1'}:
                    continue

                key = (corr.table_a, corr.column_a, corr.table_b, corr.column_b)
                if key not in seen:
                    seen.add(key)
                    mandatory.append({
                        'source': f"{corr.table_a}.{corr.column_a}",
                        'target': f"{corr.table_b}.{corr.column_b}",
                        'evidence': f"Value Jaccard={corr.jaccard:.0%}",
                        'priority': 'CRITICAL' if corr.jaccard >= 0.95 else 'HIGH',
                        'samples': corr.samples[:3],
                        'type': 'value_based'
                    })

        # High-confidence semantic candidates (boosted)
        for cand in self.semantic_candidates:
            if cand.confidence >= 0.6 or cand.evidence.get('boosted'):
                key = (cand.source_table, cand.source_column, cand.target_table, cand.target_column)
                if key not in seen:
                    seen.add(key)
                    mandatory.append({
                        'source': f"{cand.source_table}.{cand.source_column}",
                        'target': f"{cand.target_table}.{cand.target_column}",
                        'evidence': f"Semantic match (entity='{cand.evidence.get('potential_entity')}')",
                        'priority': 'HIGH' if cand.confidence >= 0.7 else 'MEDIUM',
                        'type': 'semantic'
                    })

        return mandatory

    def generate_llm_context(self) -> str:
        """
        v7.2: Generate PRESCRIPTIVE LLM context for FK detection.

        Changed from "informative" to "directive" - tells LLM what to DO, not just what was found.

        Returns:
            Formatted string with mandatory FK instructions
        """
        context_parts = []

        # =====================================================
        # SECTION 1: MANDATORY FK RELATIONSHIPS (PRESCRIPTIVE)
        # =====================================================
        if self.mandatory_fks:
            mandatory_context = "## 🔴 MANDATORY FK RELATIONSHIPS - YOU MUST CREATE THESE\n\n"
            mandatory_context += "Based on STRONG evidence (Jaccard ≥ 80% or semantic match), "
            mandatory_context += "the following FK relationships are VERIFIED. **CREATE ALL OF THEM.**\n\n"
            mandatory_context += "| # | Source | Target | Evidence | Priority |\n"
            mandatory_context += "|---|--------|--------|----------|----------|\n"

            for i, fk in enumerate(self.mandatory_fks, 1):
                mandatory_context += f"| {i} | `{fk['source']}` | `{fk['target']}` | {fk['evidence']} | **{fk['priority']}** |\n"

            mandatory_context += "\n**⚠️ DO NOT SKIP ANY OF THESE.** The algorithm has verified the relationship validity.\n"
            mandatory_context += "If you skip any, explain WHY with specific evidence.\n"
            context_parts.append(mandatory_context)

        # =====================================================
        # SECTION 2: ENTITY SYNONYMS (CRITICAL INSIGHT)
        # =====================================================
        if self.entity_synonyms:
            synonym_context = "## 🔵 ENTITY SYNONYMS DETECTED\n\n"
            synonym_context += "**CRITICAL**: The following terms refer to the SAME entity:\n\n"

            for syn in self.entity_synonyms:
                synonym_context += f"### '{syn.entity_a}' = '{syn.entity_b}'\n"
                synonym_context += f"- Evidence: `{syn.evidence_table_a}.{syn.evidence_column_a}` ↔ "
                synonym_context += f"`{syn.evidence_table_b}.{syn.evidence_column_b}`\n"
                synonym_context += f"- Jaccard: {syn.jaccard:.0%}\n"
                synonym_context += f"- **Implication**: Any column named '{syn.entity_a}' or '{syn.entity_b}' "
                synonym_context += f"likely refers to the same entity!\n\n"

            context_parts.append(synonym_context)

        # =====================================================
        # SECTION 3: INDIRECT RELATIONSHIPS VIA BRIDGE TABLES
        # =====================================================
        if self.indirect_relationships:
            indirect_context = "## 🟡 INDIRECT FK RELATIONSHIPS (Via Bridge Tables)\n\n"
            indirect_context += "These tables are connected INDIRECTLY through a mapping table:\n\n"

            for rel in self.indirect_relationships:
                indirect_context += f"### {rel.source_table} ↔ {rel.target_table}\n"
                indirect_context += f"- **Join Path**: `{rel.join_path}`\n"
                indirect_context += f"- **Bridge Table**: `{rel.bridge_table}`\n"
                indirect_context += f"- **Code Systems**: {rel.source_code_system} ↔ {rel.target_code_system}\n"
                indirect_context += f"- **Action**: Create FK relationship with bridge table join\n\n"

            context_parts.append(indirect_context)

        # =====================================================
        # SECTION 4: BRIDGE TABLES
        # =====================================================
        bridge_context = self.bridge_detector.generate_llm_context(self.bridge_tables)
        if bridge_context:
            context_parts.append(bridge_context)

        # =====================================================
        # SECTION 5: STRONG VALUE MATCHES (Supporting Evidence)
        # =====================================================
        value_context = self.value_correlator.generate_llm_context(
            self.value_correlations, top_n=20
        )
        if value_context:
            context_parts.append(value_context)

        # =====================================================
        # SECTION 6: SEMANTIC CANDIDATES
        # =====================================================
        if self.semantic_candidates:
            sem_context = "## SEMANTIC FK CANDIDATES (From Column Names)\n\n"
            sem_context += "These FK relationships were inferred from column name semantics:\n\n"

            for cand in self.semantic_candidates:
                boosted = " ⬆️ BOOSTED" if cand.evidence.get('boosted') else ""
                sem_context += f"- **{cand.source_table}.{cand.source_column}** → "
                sem_context += f"**{cand.target_table}.{cand.target_column}**{boosted}\n"
                sem_context += f"  - Confidence: {cand.confidence:.0%}\n"
                sem_context += f"  - Evidence: {cand.evidence}\n\n"

            context_parts.append(sem_context)

        # =====================================================
        # SECTION 7: LEARNED ABBREVIATIONS
        # =====================================================
        if self.learned_abbreviations:
            abbr_context = "## LEARNED ABBREVIATION PATTERNS\n\n"
            abbr_context += "These abbreviations were learned from value overlap analysis:\n\n"

            for abbr in self.learned_abbreviations:
                abbr_context += f"- '{abbr.short_form}' = '{abbr.long_form}' "
                abbr_context += f"(learned from {abbr.evidence_pairs[0]})\n"

            context_parts.append(abbr_context)

        return "\n\n".join(context_parts)

    def get_all_fk_candidates(self) -> List[SemanticFKCandidate]:
        """
        Get all FK candidates from all detection methods.

        Returns:
            Combined list of all FK candidates
        """
        all_candidates = list(self.semantic_candidates)

        # Add value-based candidates
        for corr in self.value_correlations:
            if corr.is_strong_match:
                candidate = SemanticFKCandidate(
                    source_table=corr.table_a,
                    source_column=corr.column_a,
                    target_table=corr.table_b,
                    target_column=corr.column_b,
                    detection_method='value_overlap',
                    confidence=corr.jaccard,
                    evidence={
                        'jaccard': corr.jaccard,
                        'containment_a': corr.containment_a_in_b,
                        'containment_b': corr.containment_b_in_a,
                        'samples': corr.samples
                    },
                    llm_context=f"Strong value overlap ({corr.jaccard:.0%}) between these columns"
                )
                all_candidates.append(candidate)

        # Deduplicate
        seen = set()
        unique = []
        for cand in all_candidates:
            key = (cand.source_table, cand.source_column, cand.target_table, cand.target_column)
            if key not in seen:
                seen.add(key)
                unique.append(cand)

        return unique


# =============================================================================
# Helper Functions
# =============================================================================

def create_semantic_fk_detector(tables_data: Dict[str, Any]) -> SemanticFKDetector:
    """Factory function to create a SemanticFKDetector."""
    return SemanticFKDetector(tables_data)


def analyze_fk_semantics(tables_data: Dict[str, Any]) -> Dict[str, Any]:
    """Convenience function for quick FK semantic analysis."""
    detector = SemanticFKDetector(tables_data)
    return detector.analyze()
