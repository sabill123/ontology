"""
NL2SQL DSPy Signatures (v17.0)

자연어 → SQL 변환을 위한 DSPy Signature 정의:
- NL2SQLSignature: 메인 변환 시그니처
- JoinPathSignature: 조인 경로 추론
- SQLValidationSignature: SQL 검증
- QueryIntentSignature: 쿼리 의도 분석
"""

try:
    import dspy
    _DSPY_AVAILABLE = True
except ImportError:
    _DSPY_AVAILABLE = False

    # Stub base class so module can be imported without dspy
    class _StubSignature:
        pass

    class _DspyStub:
        Signature = _StubSignature
        InputField = lambda *a, **kw: None
        OutputField = lambda *a, **kw: None

    dspy = _DspyStub()

from typing import List, Optional


class QueryIntentSignature(dspy.Signature):
    """Analyze the intent of a natural language query.

    Determine:
    - Query type (select, aggregate, filter, join, etc.)
    - Time range if mentioned
    - Entities/tables likely involved
    - Aggregation functions needed
    - Sort/limit requirements
    """

    natural_query: str = dspy.InputField(desc="Natural language query in any language")
    available_tables: List[str] = dspy.InputField(desc="List of available table names")
    schema_summary: str = dspy.InputField(desc="Brief summary of schema relationships")

    query_type: str = dspy.OutputField(
        desc="Type: select, aggregate, filter, comparison, trend, top_n, distribution"
    )
    target_tables: List[str] = dspy.OutputField(desc="Tables that should be queried")
    time_range: Optional[str] = dspy.OutputField(
        desc="Time range mentioned (e.g., 'last_month', '2024-01', None if not specified)"
    )
    aggregations: List[str] = dspy.OutputField(
        desc="Aggregation functions needed (SUM, AVG, COUNT, etc.)"
    )
    filters: List[str] = dspy.OutputField(desc="Filter conditions in natural language")
    sort_by: Optional[str] = dspy.OutputField(desc="Sort column and direction if applicable")
    limit: Optional[int] = dspy.OutputField(desc="Result limit if specified (e.g., TOP 10)")
    reasoning: str = dspy.OutputField(desc="Analysis of query intent")


class NL2SQLSignature(dspy.Signature):
    """Convert a natural language query to SQL.

    Consider:
    - Use the exact table and column names from the schema
    - Apply appropriate JOINs based on foreign key relationships
    - Handle date/time filters correctly
    - Use proper aggregation and grouping
    - Apply appropriate sorting and limits

    Output valid SQL that can be executed directly.
    """

    natural_query: str = dspy.InputField(desc="Natural language query in any language")
    schema_context: str = dspy.InputField(
        desc="JSON schema with tables, columns, types, and relationships"
    )
    query_intent: str = dspy.InputField(desc="Analyzed query intent (type, tables, filters)")
    join_paths: str = dspy.InputField(
        desc="Available join paths between tables as JSON"
    )
    dialect: str = dspy.InputField(
        desc="SQL dialect: sqlite, postgresql, mysql, etc."
    )
    conversation_context: Optional[str] = dspy.InputField(
        desc="Previous queries in this conversation for context"
    )

    sql_query: str = dspy.OutputField(desc="Valid SQL query")
    explanation: str = dspy.OutputField(desc="Explanation of how the SQL answers the question")
    tables_used: List[str] = dspy.OutputField(desc="Tables used in the query")
    columns_used: List[str] = dspy.OutputField(desc="Columns used (table.column format)")
    confidence: float = dspy.OutputField(desc="Confidence score 0.0-1.0")


class JoinPathSignature(dspy.Signature):
    """Determine the optimal join path between tables.

    Given a set of tables that need to be joined:
    1. Find all possible paths through foreign key relationships
    2. Select the shortest/most direct path
    3. Determine join type (INNER, LEFT, etc.)
    4. Specify join conditions

    Consider bridge tables for indirect relationships.
    """

    source_table: str = dspy.InputField(desc="Starting table")
    target_table: str = dspy.InputField(desc="Target table to join")
    foreign_keys: str = dspy.InputField(
        desc="JSON of all FK relationships: [{from_table, from_col, to_table, to_col}, ...]"
    )
    tables_in_query: List[str] = dspy.InputField(desc="All tables that need to be in the query")

    join_path: List[str] = dspy.OutputField(
        desc="Ordered list of tables in join path"
    )
    join_conditions: List[str] = dspy.OutputField(
        desc="JOIN conditions: ['tableA.col = tableB.col', ...]"
    )
    join_types: List[str] = dspy.OutputField(
        desc="JOIN types for each step: ['INNER', 'LEFT', ...]"
    )
    is_direct: bool = dspy.OutputField(desc="Whether this is a direct FK relationship")
    reasoning: str = dspy.OutputField(desc="Explanation of the chosen join path")


class SQLValidationSignature(dspy.Signature):
    """Validate and improve a generated SQL query.

    Check for:
    - Syntax errors
    - Missing JOINs
    - Ambiguous column references
    - Performance issues (missing indexes, full scans)
    - Security issues (injection risks)

    Suggest fixes if problems found.
    """

    sql_query: str = dspy.InputField(desc="SQL query to validate")
    schema_context: str = dspy.InputField(desc="Schema information for validation")
    dialect: str = dspy.InputField(desc="SQL dialect")

    is_valid: bool = dspy.OutputField(desc="Whether the SQL is valid")
    issues: List[str] = dspy.OutputField(desc="List of issues found")
    suggestions: List[str] = dspy.OutputField(desc="Suggested improvements")
    corrected_sql: Optional[str] = dspy.OutputField(
        desc="Corrected SQL if issues found, None otherwise"
    )
    performance_notes: List[str] = dspy.OutputField(desc="Performance considerations")


class QueryClarificationSignature(dspy.Signature):
    """Generate clarifying questions when query is ambiguous.

    When the natural language query is unclear:
    - Identify what's ambiguous
    - Generate specific questions
    - Provide options where applicable
    """

    natural_query: str = dspy.InputField(desc="The ambiguous natural language query")
    schema_context: str = dspy.InputField(desc="Schema information")
    ambiguity_type: str = dspy.InputField(
        desc="Type: column_choice, time_range, aggregation, entity, filter_value"
    )

    clarifying_questions: List[str] = dspy.OutputField(
        desc="Questions to ask the user"
    )
    options: List[str] = dspy.OutputField(
        desc="Possible options to present"
    )
    default_interpretation: str = dspy.OutputField(
        desc="Default interpretation if user doesn't clarify"
    )


class SQLExplanationSignature(dspy.Signature):
    """Generate a natural language explanation of SQL query results.

    Translate query results into human-readable insights.
    """

    sql_query: str = dspy.InputField(desc="The SQL query that was executed")
    query_results: str = dspy.InputField(desc="JSON of query results")
    original_question: str = dspy.InputField(desc="Original natural language question")
    row_count: int = dspy.InputField(desc="Number of rows returned")

    summary: str = dspy.OutputField(desc="1-2 sentence summary of results")
    insights: List[str] = dspy.OutputField(desc="Key insights from the data")
    follow_up_suggestions: List[str] = dspy.OutputField(
        desc="Suggested follow-up queries"
    )
