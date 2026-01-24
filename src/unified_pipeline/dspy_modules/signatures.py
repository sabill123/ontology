"""
DSPy Signatures - 프롬프트를 선언적으로 정의

Signature는 입출력 스펙을 정의하며, DSPy가 이를 기반으로
최적의 프롬프트를 자동 생성합니다.

사용 예시:
    class MySignature(dspy.Signature):
        '''Docstring이 시스템 프롬프트가 됩니다.'''
        input_field = dspy.InputField(desc="입력 설명")
        output_field = dspy.OutputField(desc="출력 설명")
"""

try:
    import dspy
except ImportError:
    raise ImportError("dspy-ai 패키지가 필요합니다. pip install dspy-ai")

from typing import List, Optional


# ============================================================
# FK Detection Signatures
# ============================================================

class FKRelationshipSignature(dspy.Signature):
    """Analyze if there's a foreign key relationship between two database columns.

    Consider column names, value patterns, overlap ratios, and semantic meaning.
    Be precise about relationship type and cardinality.
    """

    # 입력 필드
    source_table: str = dspy.InputField(desc="Name of the source table")
    source_column: str = dspy.InputField(desc="Name of the source column")
    source_samples: List[str] = dspy.InputField(desc="Sample values from source column")
    target_table: str = dspy.InputField(desc="Name of the target table")
    target_column: str = dspy.InputField(desc="Name of the target column")
    target_samples: List[str] = dspy.InputField(desc="Sample values from target column")
    overlap_ratio: float = dspy.InputField(desc="Value overlap ratio between columns (0.0-1.0)")

    # 출력 필드
    is_relationship: bool = dspy.OutputField(desc="Whether a FK relationship exists")
    relationship_type: str = dspy.OutputField(
        desc="Type: 'foreign_key', 'cross_silo_reference', 'semantic_link', or 'none'"
    )
    cardinality: str = dspy.OutputField(
        desc="Cardinality: 'ONE_TO_ONE', 'ONE_TO_MANY', 'MANY_TO_ONE', 'MANY_TO_MANY'"
    )
    confidence: float = dspy.OutputField(desc="Confidence score between 0.0 and 1.0")
    reasoning: str = dspy.OutputField(desc="Detailed explanation of the decision")


class FKDirectionSignature(dspy.Signature):
    """Determine the direction of a foreign key relationship.

    The FK column references the PK column. Identify which table contains the FK
    (referencing table) and which contains the PK (referenced table).
    """

    table_a: str = dspy.InputField(desc="First table name")
    column_a: str = dspy.InputField(desc="Column in first table")
    table_a_samples: List[str] = dspy.InputField(desc="Sample values from first column")
    table_b: str = dspy.InputField(desc="Second table name")
    column_b: str = dspy.InputField(desc="Column in second table")
    table_b_samples: List[str] = dspy.InputField(desc="Sample values from second column")

    source_table: str = dspy.OutputField(desc="Table containing the FK (referencing table)")
    source_column: str = dspy.OutputField(desc="FK column name")
    target_table: str = dspy.OutputField(desc="Table containing the PK (referenced table)")
    target_column: str = dspy.OutputField(desc="PK column name")
    confidence: float = dspy.OutputField(desc="Direction confidence 0.0-1.0")
    reasoning: str = dspy.OutputField(desc="Explanation of direction determination")


# ============================================================
# Column Analysis Signatures
# ============================================================

class ColumnSemanticSignature(dspy.Signature):
    """Classify a database column's semantic role in business context.

    Semantic roles:
    - IDENTIFIER: Primary/Foreign keys (customer_id, order_id)
    - DIMENSION: Categories for grouping (type, status, region)
    - MEASURE: Numbers for aggregation (amount, balance, quantity)
    - TEMPORAL: Dates/times (created_at, birth_date)
    - DESCRIPTOR: Text descriptions (name, address, email)
    - FLAG: Boolean indicators (is_active, has_verified)
    - REFERENCE: External system references (external_id, legacy_code)
    """

    column_name: str = dspy.InputField(desc="Name of the column")
    table_name: str = dspy.InputField(desc="Table name for context")
    data_type: str = dspy.InputField(desc="Data type of the column")
    sample_values: List[str] = dspy.InputField(desc="Sample values from the column")
    distinct_ratio: Optional[float] = dspy.InputField(
        desc="Ratio of unique values (0.0-1.0), None if unknown"
    )

    semantic_role: str = dspy.OutputField(
        desc="Role: IDENTIFIER, DIMENSION, MEASURE, TEMPORAL, DESCRIPTOR, FLAG, REFERENCE"
    )
    business_name: str = dspy.OutputField(desc="Human-readable business display name")
    domain_category: str = dspy.OutputField(
        desc="Business domain: finance, customer, time, location, product, risk, marketing, healthcare, operations, other"
    )
    confidence: float = dspy.OutputField(desc="Classification confidence 0.0-1.0")
    reasoning: str = dspy.OutputField(desc="Explanation of classification decision")


class ColumnTypeSignature(dspy.Signature):
    """Analyze and infer the data type and semantic meaning of a database column.

    Check for patterns like:
    - Leading zeros (preserve as string)
    - Codes/identifiers (A001, SAP123)
    - Date formats
    - Email, phone, UUID patterns
    - Primary/Foreign key indicators
    """

    column_name: str = dspy.InputField(desc="Column name")
    table_name: str = dspy.InputField(desc="Table name for context")
    sample_values: List[str] = dspy.InputField(desc="Sample values (first 20 non-null)")

    inferred_type: str = dspy.OutputField(
        desc="Type: string, integer, float, boolean, date, datetime, email, phone, uuid, code, id"
    )
    semantic_type: str = dspy.OutputField(desc="Semantic meaning (e.g., customer_identifier)")
    preserve_as_string: bool = dspy.OutputField(
        desc="Whether to preserve as string (leading zeros, codes)"
    )
    is_identifier: bool = dspy.OutputField(desc="Whether this is an identifier column")
    is_pk_candidate: bool = dspy.OutputField(desc="Primary key candidate")
    is_fk_candidate: bool = dspy.OutputField(desc="Foreign key candidate")
    referenced_entity: Optional[str] = dspy.OutputField(
        desc="If FK candidate, the entity it likely references"
    )
    confidence: float = dspy.OutputField(desc="Analysis confidence 0.0-1.0")


# ============================================================
# Entity Analysis Signatures
# ============================================================

class EntityAnalysisSignature(dspy.Signature):
    """Infer the semantic meaning of a database table/entity.

    Determine:
    - What business entity does this represent?
    - Is it master data, transaction, reference, or junction table?
    - What business domain does it belong to?
    - What relationships might exist?
    """

    table_name: str = dspy.InputField(desc="Name of the table")
    columns: List[str] = dspy.InputField(desc="List of column names")
    sample_data: str = dspy.InputField(desc="JSON string of sample data (first 5 rows)")
    source_context: str = dspy.InputField(desc="Data source context (e.g., CRM, ERP)")

    entity_name: str = dspy.OutputField(desc="PascalCase semantic entity name")
    display_name: str = dspy.OutputField(desc="Human-readable display name")
    description: str = dspy.OutputField(desc="Description of what this entity represents")
    entity_type: str = dspy.OutputField(
        desc="Type: master_data, transaction, reference, junction"
    )
    business_domain: str = dspy.OutputField(desc="Business domain (sales, customer_management, etc.)")
    key_attributes: List[str] = dspy.OutputField(desc="List of key attribute column names")


# ============================================================
# Threshold Decision Signatures
# ============================================================

class ThresholdDecisionSignature(dspy.Signature):
    """Determine the optimal threshold for data analysis based on data characteristics.

    Instead of hardcoded magic numbers (0.3, 0.5), analyze the data and context
    to recommend appropriate thresholds.

    Consider:
    - Data distribution and characteristics
    - Industry/domain standards
    - False positive vs false negative tradeoffs
    - Business risk tolerance
    """

    threshold_type: str = dspy.InputField(
        desc="Type: null_ratio, outlier_ratio, correlation, distinctness, quality_score, confidence, similarity"
    )
    data_characteristics: str = dspy.InputField(desc="JSON string of data characteristics")
    domain_context: str = dspy.InputField(desc="JSON string of domain context (industry, stage, etc.)")

    recommended_value: float = dspy.OutputField(desc="Recommended threshold between 0.0 and 1.0")
    reasoning: str = dspy.OutputField(desc="Why this value was chosen")
    confidence: float = dspy.OutputField(desc="Recommendation confidence 0.0-1.0")


# ============================================================
# Consensus Vote Signatures
# ============================================================

class ConsensusVoteSignature(dspy.Signature):
    """Vote on an ontology proposal from a specific role perspective.

    Roles and their focus:
    - ontologist: Semantic correctness, ontology best practices
    - risk_assessor: Data quality risks, operational concerns
    - business_strategist: Business value, strategic alignment
    - data_steward: Governance compliance, data standards
    """

    proposal: str = dspy.InputField(desc="The ontology proposal to vote on")
    evidence: str = dspy.InputField(desc="Supporting evidence for the proposal")
    role: str = dspy.InputField(
        desc="Voter role: ontologist, risk_assessor, business_strategist, data_steward"
    )
    previous_votes: str = dspy.InputField(
        desc="JSON string of other agents' votes (if any)"
    )

    vote: str = dspy.OutputField(desc="Decision: approve, reject, or abstain")
    confidence: float = dspy.OutputField(desc="Vote confidence 0.0-1.0")
    key_observations: List[str] = dspy.OutputField(desc="Key observations supporting the vote")
    concerns: List[str] = dspy.OutputField(desc="Any concerns or risks identified")
    reasoning: str = dspy.OutputField(desc="Detailed reasoning for the vote")


# ============================================================
# Cross-Silo Analysis Signatures
# ============================================================

class CrossSiloRelationshipSignature(dspy.Signature):
    """Identify potential relationships across different data silos/sources.

    Look for:
    - Columns referencing external systems (sap_customer_id in CRM)
    - Entity matches across sources (same customer in multiple systems)
    - Naming patterns suggesting cross-system references
    """

    sources_metadata: str = dspy.InputField(
        desc="JSON string of all data sources with their tables and columns"
    )

    relationships: List[str] = dspy.OutputField(
        desc="JSON strings describing each cross-silo relationship found"
    )
    entity_mappings: List[str] = dspy.OutputField(
        desc="JSON strings describing entity mappings across systems"
    )


# ============================================================
# Reasoning Chain Signatures
# ============================================================

class ReasoningStepSignature(dspy.Signature):
    """Execute one step in a multi-step reasoning chain.

    Analyze deeply and determine if further investigation is needed.
    """

    question: str = dspy.InputField(desc="The current analysis question")
    data_context: str = dspy.InputField(desc="JSON string of relevant data context")
    previous_findings: List[str] = dspy.InputField(desc="Findings from previous steps")

    analysis: str = dspy.OutputField(desc="Detailed analysis of the question")
    findings: List[str] = dspy.OutputField(desc="Specific findings with concrete evidence")
    confidence: float = dspy.OutputField(desc="Confidence in findings 0.0-1.0")
    next_question: Optional[str] = dspy.OutputField(
        desc="Follow-up question if more investigation needed, None if done"
    )
