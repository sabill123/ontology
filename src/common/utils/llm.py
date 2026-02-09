"""
LLM Client for the Ontology Platform
Uses Letsur AI Gateway (OpenAI-compatible)

Enhanced for universal data analysis without hardcoded patterns.
v5.0: JobLogger 통합 - 모든 LLM 호출 자동 기록
"""

import os
import json
import re
import time
import threading
from typing import Optional, List, Dict, Any, Union
import openai
from dotenv import load_dotenv

load_dotenv()

# === LLM Call Logging Context ===
# 현재 에이전트 이름과 사고 과정을 저장하는 thread-local storage
_llm_context = threading.local()


def set_llm_agent_context(agent_name: str):
    """LLM 호출 컨텍스트에 에이전트 이름 설정"""
    _llm_context.agent_name = agent_name


def get_llm_agent_context() -> str:
    """현재 LLM 호출 컨텍스트의 에이전트 이름 반환"""
    return getattr(_llm_context, "agent_name", "unknown")


def clear_llm_agent_context():
    """LLM 호출 컨텍스트 초기화"""
    _llm_context.agent_name = "unknown"
    _llm_context.purpose = None
    _llm_context.thinking_before = None


def set_llm_thinking_context(purpose: str = None, thinking_before: str = None):
    """
    LLM 호출 전 사고 컨텍스트 설정

    Args:
        purpose: LLM 호출 목적 (예: "entity classification", "relationship detection")
        thinking_before: 호출 전 에이전트의 사고 과정
    """
    _llm_context.purpose = purpose
    _llm_context.thinking_before = thinking_before


def get_llm_thinking_context() -> Dict[str, Optional[str]]:
    """현재 LLM 사고 컨텍스트 반환"""
    return {
        "purpose": getattr(_llm_context, "purpose", None),
        "thinking_before": getattr(_llm_context, "thinking_before", None),
    }


def clear_llm_thinking_context():
    """LLM 사고 컨텍스트만 초기화"""
    _llm_context.purpose = None
    _llm_context.thinking_before = None

# Letsur AI Gateway 설정
LETSUR_BASE_URL = "https://gateway.letsur.ai/v1"
LETSUR_API_KEY = os.environ.get("LETSUR_API_KEY", "")

# Default model for data analysis (align with multi-agent configuration)
DEFAULT_MODEL = "gpt-5.1"


def get_openai_client() -> openai.OpenAI:
    """Get OpenAI client configured for Letsur Gateway."""
    return openai.OpenAI(
        base_url=LETSUR_BASE_URL,
        api_key=LETSUR_API_KEY,
        timeout=300.0,  # 5 min timeout for multi-agent pipeline (v21.0)
    )


def chat_completion(
    model: str,
    messages: List[Dict[str, str]],
    max_tokens: Optional[int] = None,
    temperature: float = 0.7,
    client: Optional[openai.OpenAI] = None,
    agent_name: Optional[str] = None,
    seed: Optional[int] = None,
) -> str:
    """
    Send a chat completion request.

    Args:
        model: Model name (e.g., "gpt-5.2", "claude-opus-4-5-20251101", "gemini-3-pro-preview")
        messages: List of message dicts with 'role' and 'content'
        max_tokens: Maximum tokens in response (None for no limit)
        temperature: Sampling temperature
        client: Optional pre-configured client
        agent_name: Optional agent name for logging (overrides context)

    Returns:
        Response content string
    """
    if client is None:
        client = get_openai_client()

    # 로깅용 에이전트 이름 결정
    log_agent_name = agent_name or get_llm_agent_context()
    start_time = time.time()
    success = True
    error_msg = None
    response_content = ""
    prompt_tokens = 0
    completion_tokens = 0

    try:
        kwargs = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
        }
        # v25.0: seed 파라미터로 재현성 보장
        if seed is not None:
            kwargs["seed"] = seed
        # max_tokens intentionally omitted — let the model use its default limit

        # Terminal logging for debugging
        print(f"[LLM] Calling {model} (agent: {log_agent_name})...")

        response = client.chat.completions.create(**kwargs)
        response_content = response.choices[0].message.content or ""

        # Terminal logging for response
        print(f"[LLM] Response received: {len(response_content)} chars in {(time.time() - start_time):.2f}s")

        # 토큰 사용량 추출
        if hasattr(response, "usage") and response.usage:
            prompt_tokens = response.usage.prompt_tokens or 0
            completion_tokens = response.usage.completion_tokens or 0

    except Exception as e:
        success = False
        error_msg = str(e)
        print(f"[LLM] ERROR: {e}")
        raise RuntimeError(f"LLM request failed: {e}")
    finally:
        # JobLogger에 기록
        latency_ms = (time.time() - start_time) * 1000
        _log_llm_call(
            agent_name=log_agent_name,
            model=model,
            messages=messages,
            response=response_content,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            latency_ms=latency_ms,
            success=success,
            error=error_msg,
        )

    return response_content


def _log_llm_call(
    agent_name: str,
    model: str,
    messages: List[Dict[str, str]],
    response: str,
    prompt_tokens: int,
    completion_tokens: int,
    latency_ms: float,
    success: bool,
    error: Optional[str],
    purpose: Optional[str] = None,
    thinking_before: Optional[str] = None,
    thinking_after: Optional[str] = None,
    extracted_insights: Optional[List[str]] = None,
):
    """LLM 호출을 JobLogger에 기록 (내부 함수)"""
    try:
        from .job_logger import get_job_logger

        job_logger = get_job_logger()
        if job_logger.is_active():
            # 컨텍스트에서 사고 과정 정보 가져오기
            thinking_ctx = get_llm_thinking_context()
            final_purpose = purpose or thinking_ctx.get("purpose")
            final_thinking_before = thinking_before or thinking_ctx.get("thinking_before")

            job_logger.log_llm_call(
                agent_name=agent_name,
                model=model,
                messages=messages,
                response=response,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                latency_ms=latency_ms,
                success=success,
                error=error,
                purpose=final_purpose,
                thinking_before=final_thinking_before,
                thinking_after=thinking_after,
                extracted_insights=extracted_insights,
            )

            # 사고 컨텍스트 초기화 (사용 후)
            clear_llm_thinking_context()
    except ImportError:
        pass  # JobLogger가 없으면 무시
    except Exception:
        pass  # 로깅 실패해도 메인 로직에 영향 없음


def chat_completion_with_json(
    model: str,
    messages: List[Dict[str, str]],
    max_tokens: Optional[int] = None,
    temperature: float = 0.2,
    client: Optional[openai.OpenAI] = None,
    agent_name: Optional[str] = None,
    seed: Optional[int] = None,
) -> str:
    """
    Send a chat completion request expecting JSON response.
    Uses lower temperature for more consistent output.
    """
    if client is None:
        client = get_openai_client()

    # 로깅용 에이전트 이름 결정
    log_agent_name = agent_name or get_llm_agent_context()
    start_time = time.time()
    success = True
    error_msg = None
    response_content = "{}"
    prompt_tokens = 0
    completion_tokens = 0

    try:
        kwargs = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
        }
        # v25.0: seed 파라미터로 재현성 보장
        if seed is not None:
            kwargs["seed"] = seed
        # max_tokens intentionally omitted — let the model use its default limit

        # Enable JSON mode for supported models
        if "gpt" in model.lower():
            kwargs["response_format"] = {"type": "json_object"}

        response = client.chat.completions.create(**kwargs)
        response_content = response.choices[0].message.content or "{}"

        # 토큰 사용량 추출
        if hasattr(response, "usage") and response.usage:
            prompt_tokens = response.usage.prompt_tokens or 0
            completion_tokens = response.usage.completion_tokens or 0

    except Exception as e:
        success = False
        error_msg = str(e)
        print(f"[LLM] ERROR: {e}")
        raise RuntimeError(f"LLM request failed: {e}")
    finally:
        # JobLogger에 기록
        latency_ms = (time.time() - start_time) * 1000
        _log_llm_call(
            agent_name=log_agent_name,
            model=model,
            messages=messages,
            response=response_content,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            latency_ms=latency_ms,
            success=success,
            error=error_msg,
        )

    return response_content


def extract_json_from_response(response: str) -> Dict[str, Any]:
    """
    Extract JSON from LLM response, handling various formats.

    Args:
        response: Raw LLM response string

    Returns:
        Parsed JSON dictionary
    """
    # Try direct JSON parse first
    try:
        return json.loads(response)
    except json.JSONDecodeError:
        pass

    # Try to find JSON in markdown code blocks
    json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', response)
    if json_match:
        try:
            return json.loads(json_match.group(1))
        except json.JSONDecodeError:
            pass

    # Try to find JSON object pattern
    json_match = re.search(r'\{[\s\S]*\}', response)
    if json_match:
        try:
            return json.loads(json_match.group(0))
        except json.JSONDecodeError:
            pass

    # Return empty dict if nothing works
    return {}


# Available models (aligned with production agents) — v22.1: 환경변수 오버라이드
MODELS = {
    "fast": os.environ.get("MODEL_FAST", "gpt-5.1"),
    "balanced": os.environ.get("MODEL_BALANCED", "claude-opus-4-5-20251101"),
    "creative": os.environ.get("MODEL_CREATIVE", "gemini-3-pro-preview"),
    "high_context": os.environ.get("MODEL_HIGH_CONTEXT", "gemini-3-pro-preview"),
}


def get_model(model_type: str = "balanced") -> str:
    """Get model name by type."""
    return MODELS.get(model_type, MODELS["balanced"])


# ============================================================
# Data Analysis LLM Functions
# ============================================================

def analyze_column_type(
    column_name: str,
    sample_values: List[Any],
    table_context: Optional[str] = None,
    client: Optional[openai.OpenAI] = None,
) -> Dict[str, Any]:
    """
    Use LLM to intelligently infer column data type and semantic meaning.

    Returns:
        {
            "inferred_type": "string" | "integer" | "float" | "boolean" | "date" | "datetime" | "email" | "phone" | "uuid" | "code" | "id",
            "semantic_type": str,  # e.g., "customer_identifier", "transaction_date", "email_address"
            "should_preserve_as_string": bool,  # True if leading zeros, codes, etc.
            "is_identifier": bool,
            "is_foreign_key_candidate": bool,
            "pattern_description": str,
            "confidence": float
        }
    """
    if client is None:
        client = get_openai_client()

    # Prepare sample values for prompt
    samples = [str(v) for v in sample_values[:20] if v is not None and str(v).strip()]

    prompt = f"""Analyze this database column and determine its data type and semantic meaning.

Column Name: {column_name}
Table Context: {table_context or "Unknown"}
Sample Values (first 20 non-null): {samples}

Analyze the values carefully:
1. Check for patterns like leading zeros (e.g., "001", "0123") - these should be preserved as strings
2. Check for codes/identifiers (e.g., "A001", "SAP123", "CRM-456")
3. Check for dates in various formats (YYYY-MM-DD, DD/MM/YYYY, timestamps)
4. Check for email addresses, phone numbers, UUIDs
5. Determine if this is likely a primary key or foreign key based on:
   - Column name patterns (contains 'id', 'key', 'code', 'ref', 'fk')
   - Value patterns (unique identifiers vs repeated values)
   - Relationship hints in the name (e.g., "customer_id" suggests FK to customers)

Return JSON with:
{{
    "inferred_type": "string|integer|float|boolean|date|datetime|email|phone|uuid|code|id",
    "semantic_type": "description of what this column represents",
    "should_preserve_as_string": true/false,
    "is_identifier": true/false,
    "is_primary_key_candidate": true/false,
    "is_foreign_key_candidate": true/false,
    "referenced_entity": "entity name if FK candidate, else null",
    "pattern_description": "description of detected pattern",
    "confidence": 0.0-1.0
}}

Return only valid JSON."""

    try:
        response = chat_completion_with_json(
            model=DEFAULT_MODEL,
            messages=[
                {"role": "system", "content": "You are a data analyst expert. Analyze column data types precisely. Return only valid JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            client=client,
        )
        return extract_json_from_response(response)
    except Exception as e:
        # Fallback to basic inference
        return {
            "inferred_type": "string",
            "semantic_type": column_name,
            "should_preserve_as_string": True,
            "is_identifier": False,
            "is_primary_key_candidate": False,
            "is_foreign_key_candidate": False,
            "referenced_entity": None,
            "pattern_description": f"LLM analysis failed: {e}",
            "confidence": 0.0
        }


def detect_foreign_key_relationship(
    source_column: Dict[str, Any],
    target_column: Dict[str, Any],
    source_values: List[Any],
    target_values: List[Any],
    source_context: str,
    target_context: str,
    client: Optional[openai.OpenAI] = None,
) -> Dict[str, Any]:
    """
    Use LLM to determine if there's a foreign key relationship between columns.

    Args:
        source_column: {"name": str, "table": str, "source": str}
        target_column: {"name": str, "table": str, "source": str}
        source_values: Sample values from source column
        target_values: Sample values from target column
        source_context: Description of source table/source
        target_context: Description of target table/source

    Returns:
        {
            "is_relationship": bool,
            "relationship_type": "foreign_key" | "cross_silo_reference" | "none",
            "confidence": float,
            "cardinality": "ONE_TO_ONE" | "ONE_TO_MANY" | "MANY_TO_ONE" | "MANY_TO_MANY",
            "semantic_relationship": str,  # e.g., "customer places orders"
            "reasoning": str
        }
    """
    if client is None:
        client = get_openai_client()

    # Calculate value overlap
    src_set = set(str(v) for v in source_values if v is not None)
    tgt_set = set(str(v) for v in target_values if v is not None)
    overlap = len(src_set & tgt_set)
    overlap_ratio = overlap / len(src_set) if src_set else 0

    prompt = f"""Analyze if there's a foreign key relationship between these two columns.

SOURCE:
- Data Source: {source_context}
- Table: {source_column['table']}
- Column: {source_column['name']}
- Sample Values: {[str(v) for v in source_values[:15]]}

TARGET:
- Data Source: {target_context}
- Table: {target_column['table']}
- Column: {target_column['name']}
- Sample Values: {[str(v) for v in target_values[:15]]}

VALUE OVERLAP ANALYSIS:
- Overlapping values: {overlap}
- Overlap ratio: {overlap_ratio:.2%}

Consider:
1. Column name semantics - do they suggest a relationship?
2. Value overlap - do source values appear in target?
3. Cross-silo patterns - source column name might reference external system
   (e.g., "sap_customer_id" in CRM might reference SAP's customer table)
4. Naming conventions - "_id", "_key", "_ref", "_code" suffixes
5. Entity relationships - does the semantic meaning suggest a relationship?

Return JSON:
{{
    "is_relationship": true/false,
    "relationship_type": "foreign_key|cross_silo_reference|semantic_link|none",
    "confidence": 0.0-1.0,
    "cardinality": "ONE_TO_ONE|ONE_TO_MANY|MANY_TO_ONE|MANY_TO_MANY",
    "semantic_relationship": "description like 'customer places orders'",
    "reasoning": "explanation of why this is/isn't a relationship"
}}

Return only valid JSON."""

    try:
        response = chat_completion_with_json(
            model=DEFAULT_MODEL,
            messages=[
                {"role": "system", "content": "You are a database relationship expert. Analyze foreign key relationships precisely. Return only valid JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            client=client,
        )
        return extract_json_from_response(response)
    except Exception as e:
        return {
            "is_relationship": False,
            "relationship_type": "none",
            "confidence": 0.0,
            "cardinality": "UNKNOWN",
            "semantic_relationship": "",
            "reasoning": f"LLM analysis failed: {e}"
        }


def analyze_cross_silo_relationships(
    sources_metadata: List[Dict[str, Any]],
    client: Optional[openai.OpenAI] = None,
) -> List[Dict[str, Any]]:
    """
    Use LLM to analyze potential cross-silo relationships across multiple data sources.

    Args:
        sources_metadata: List of source metadata with tables and columns

    Returns:
        List of potential cross-silo relationships
    """
    if client is None:
        client = get_openai_client()

    prompt = f"""Analyze these data sources and identify potential cross-silo relationships.

DATA SOURCES:
{json.dumps(sources_metadata, indent=2, default=str)}

For each data source, I've provided:
- Source name and type
- Tables with their columns
- Sample column values where available

Identify relationships where:
1. A column in one source might reference a table in another source
2. Column names suggest cross-system references (e.g., "external_customer_id", "sap_order_ref")
3. Business entities span multiple systems (e.g., customers in both CRM and ERP)
4. Value patterns match across sources

For each potential relationship, provide:
{{
    "from_source": "source name",
    "from_table": "table name",
    "from_column": "column name",
    "to_source": "target source name",
    "to_table": "target table name",
    "to_column": "target column name",
    "relationship_type": "cross_silo_fk|entity_match|semantic_link",
    "confidence": 0.0-1.0,
    "reasoning": "explanation"
}}

Return JSON:
{{
    "relationships": [array of relationships],
    "entity_mappings": [
        {{
            "entity_name": "Customer",
            "occurrences": [
                {{"source": "...", "table": "...", "key_column": "..."}}
            ]
        }}
    ]
}}

Return only valid JSON."""

    try:
        response = chat_completion_with_json(
            model=DEFAULT_MODEL,
            messages=[
                {"role": "system", "content": "You are an enterprise data integration expert. Analyze cross-system data relationships. Return only valid JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            client=client,
        )
        result = extract_json_from_response(response)
        return result.get("relationships", [])
    except Exception as e:
        return []


def classify_semantic_role(
    column_name: str,
    sample_values: List[Any],
    data_type: str,
    table_context: str,
    statistics: Optional[Dict[str, Any]] = None,
    client: Optional[openai.OpenAI] = None,
) -> Dict[str, Any]:
    """
    Use LLM to classify a column's semantic role for ontology generation.

    Semantic roles determine how the column relates to real-world concepts:
    - IDENTIFIER: Primary/Foreign keys linking entities (account_id, customer_id)
    - DIMENSION: Categorical attributes for grouping/filtering (customer_type, region)
    - MEASURE: Numeric values for aggregation/calculation (amount, balance, score)
    - TEMPORAL: Date/time fields (created_at, transaction_date, birth_date)
    - DESCRIPTOR: Descriptive text/names (first_name, description, address)
    - FLAG: Boolean indicators (is_active, has_premium, verified)
    - REFERENCE: External system references (external_id, source_system)

    Args:
        column_name: Name of the column
        sample_values: Sample values from the column
        data_type: Inferred data type (string, integer, float, etc.)
        table_context: Table name or description for context
        statistics: Optional statistics (distinct_count, null_ratio, etc.)
        client: Optional pre-configured OpenAI client

    Returns:
        {
            "semantic_role": "IDENTIFIER|DIMENSION|MEASURE|TEMPORAL|DESCRIPTOR|FLAG|REFERENCE|UNKNOWN",
            "confidence": 0.0-1.0,
            "business_name": "Human readable name",
            "business_description": "What this column represents in business terms",
            "domain_category": "finance|customer|time|location|product|risk|marketing|...",
            "value_domain": {...},  # constraints/allowed values
            "reasoning": "Why this classification was chosen"
        }
    """
    if client is None:
        client = get_openai_client()

    # Prepare sample values
    samples = [str(v) for v in sample_values[:20] if v is not None and str(v).strip()]

    # Prepare statistics info
    stats_info = ""
    if statistics:
        stats_info = f"""
Statistics:
- Distinct count: {statistics.get('distinct_count', 'N/A')}
- Null ratio: {statistics.get('null_ratio', 'N/A')}
- Min: {statistics.get('min', 'N/A')}
- Max: {statistics.get('max', 'N/A')}
- Mean: {statistics.get('mean', 'N/A')}"""

    prompt = f"""Analyze this database column and determine its semantic role in the business context.

Column Name: {column_name}
Table: {table_context}
Data Type: {data_type}
Sample Values: {samples[:15]}
{stats_info}

SEMANTIC ROLES (choose the most appropriate):
1. IDENTIFIER - Primary/Foreign keys that uniquely identify entities or link tables
   - Examples: account_id, customer_id, order_number, transaction_id
   - Key indicators: "_id", "_key", "_code", high uniqueness, used for joins

2. DIMENSION - Categorical attributes used for grouping, filtering, slicing
   - Examples: customer_type, region, product_category, status, tier
   - Key indicators: Low cardinality, repeated values, used in GROUP BY

3. MEASURE - Numeric values meant for aggregation and calculation
   - Examples: amount, balance, score, quantity, price, count
   - Key indicators: Numeric type, continuous values, used in SUM/AVG

4. TEMPORAL - Date/time fields representing when events occurred
   - Examples: created_at, transaction_date, birth_date, start_time
   - Key indicators: Date/time values, "_at", "_date", "_time" suffixes

5. DESCRIPTOR - Descriptive text providing human-readable information
   - Examples: first_name, last_name, description, address, email
   - Key indicators: High cardinality strings, human-readable content

6. FLAG - Boolean/binary indicators representing yes/no states
   - Examples: is_active, has_verified, is_premium, approved
   - Key indicators: "is_", "has_", "can_" prefixes, binary values

7. REFERENCE - External system references or codes
   - Examples: external_id, source_system, sap_code, legacy_ref
   - Key indicators: "external_", "ref_", references to other systems

8. UNKNOWN - Cannot determine with confidence

Consider the REAL-WORLD MEANING of this data:
- What business concept does this column represent?
- How would analysts use this column?
- What questions would this column help answer?

Return JSON:
{{
    "semantic_role": "IDENTIFIER|DIMENSION|MEASURE|TEMPORAL|DESCRIPTOR|FLAG|REFERENCE|UNKNOWN",
    "confidence": 0.0-1.0,
    "business_name": "Human readable display name",
    "business_description": "What this column represents in business terms",
    "domain_category": "finance|customer|time|location|product|risk|marketing|healthcare|operations|null",
    "value_domain": {{
        "type": "enum|range|pattern|free_text",
        "allowed_values": ["list if enum"],
        "min": "if range",
        "max": "if range"
    }},
    "reasoning": "Brief explanation of classification decision"
}}

Return only valid JSON."""

    try:
        response = chat_completion_with_json(
            model=DEFAULT_MODEL,
            messages=[
                {"role": "system", "content": "You are a data analyst expert specializing in semantic data modeling and ontology design. Classify columns based on their real-world meaning and business purpose, not just technical patterns. Return only valid JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            client=client,
        )
        result = extract_json_from_response(response)

        # Ensure required fields exist
        if "semantic_role" not in result:
            result["semantic_role"] = "UNKNOWN"
        if "confidence" not in result:
            result["confidence"] = 0.5

        return result
    except Exception as e:
        return {
            "semantic_role": "UNKNOWN",
            "confidence": 0.0,
            "business_name": column_name.replace("_", " ").title(),
            "business_description": f"Column: {column_name}",
            "domain_category": None,
            "value_domain": None,
            "reasoning": f"LLM classification failed: {e}"
        }


def classify_table_columns_batch(
    table_name: str,
    columns: List[Dict[str, Any]],
    client: Optional[openai.OpenAI] = None,
) -> Dict[str, Dict[str, Any]]:
    """
    Batch classify semantic roles for all columns in a table.

    More efficient than individual calls - provides table-level context.

    Args:
        table_name: Name of the table
        columns: List of column info dicts with 'name', 'sample_values', 'data_type', 'statistics'
        client: Optional pre-configured OpenAI client

    Returns:
        Dict mapping column names to classification results
    """
    if client is None:
        client = get_openai_client()

    # Prepare column summaries
    column_summaries = []
    for col in columns:
        samples = [str(v) for v in col.get('sample_values', [])[:10] if v is not None]
        stats = col.get('statistics', {})
        summary = {
            "name": col['name'],
            "data_type": col.get('data_type', 'unknown'),
            "samples": samples[:5],
            "distinct_count": stats.get('distinct_count'),
            "null_ratio": stats.get('null_ratio')
        }
        column_summaries.append(summary)

    prompt = f"""Analyze all columns in this table and classify their semantic roles.

Table: {table_name}

Columns:
{json.dumps(column_summaries, indent=2, default=str)}

SEMANTIC ROLES:
- IDENTIFIER: Keys (account_id, customer_id)
- DIMENSION: Categories (type, status, region)
- MEASURE: Numbers for aggregation (amount, balance)
- TEMPORAL: Dates/times (created_at, birth_date)
- DESCRIPTOR: Text descriptions (name, address)
- FLAG: Boolean indicators (is_active, verified)
- REFERENCE: External refs (external_id, source_system)

For each column, determine:
1. Its semantic role based on business meaning
2. A human-readable business name
3. The business domain category

Return JSON:
{{
    "classifications": {{
        "column_name_1": {{
            "semantic_role": "IDENTIFIER|DIMENSION|MEASURE|...",
            "confidence": 0.0-1.0,
            "business_name": "Display Name",
            "domain_category": "finance|customer|time|..."
        }},
        "column_name_2": {{...}}
    }}
}}

Return only valid JSON."""

    try:
        response = chat_completion_with_json(
            model=DEFAULT_MODEL,
            messages=[
                {"role": "system", "content": "You are a semantic data modeling expert. Classify database columns by their business meaning. Return only valid JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            client=client,
        )
        result = extract_json_from_response(response)
        return result.get("classifications", {})
    except Exception as e:
        # Return empty dict on failure - caller should handle
        return {}


def infer_entity_semantics(
    table_name: str,
    columns: List[Dict[str, Any]],
    sample_data: List[Dict[str, Any]],
    source_context: str,
    client: Optional[openai.OpenAI] = None,
) -> Dict[str, Any]:
    """
    Use LLM to infer the semantic meaning of a table/entity.

    Returns:
        {
            "entity_name": str,  # PascalCase semantic name
            "display_name": str,  # Human-readable name
            "description": str,
            "entity_type": "master_data" | "transaction" | "reference" | "junction",
            "business_domain": str,  # e.g., "sales", "customer_management", "inventory"
            "key_attributes": List[str],
            "relationships_hint": List[str]  # Suggested relationships based on columns
        }
    """
    if client is None:
        client = get_openai_client()

    prompt = f"""Analyze this database table and infer its semantic meaning.

Data Source Context: {source_context}
Table Name: {table_name}

Columns:
{json.dumps(columns, indent=2, default=str)}

Sample Data (first 5 rows):
{json.dumps(sample_data[:5], indent=2, default=str)}

Determine:
1. What business entity does this table represent?
2. What is its role (master data, transaction, reference, junction table)?
3. What business domain does it belong to?
4. What are the key attributes that identify this entity?
5. What relationships might exist based on column names?

Return JSON:
{{
    "entity_name": "PascalCase name like 'Customer' or 'SalesOrder'",
    "display_name": "Human readable name (can be in Korean if appropriate)",
    "description": "What this entity represents",
    "entity_type": "master_data|transaction|reference|junction",
    "business_domain": "domain like 'sales', 'customer_management', 'inventory'",
    "key_attributes": ["list", "of", "important", "columns"],
    "relationships_hint": ["Customer -> Orders", "Product referenced by OrderItems"]
}}

Return only valid JSON."""

    try:
        response = chat_completion_with_json(
            model=DEFAULT_MODEL,
            messages=[
                {"role": "system", "content": "You are a business analyst expert in data modeling. Analyze tables semantically. Return only valid JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            client=client,
        )
        return extract_json_from_response(response)
    except Exception as e:
        # Fallback
        return {
            "entity_name": table_name.replace("_", " ").title().replace(" ", ""),
            "display_name": table_name,
            "description": f"Entity from table {table_name}",
            "entity_type": "unknown",
            "business_domain": "unknown",
            "key_attributes": [],
            "relationships_hint": []
        }
