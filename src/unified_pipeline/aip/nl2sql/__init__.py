"""
NL2SQL Module (v17.0)

자연어를 SQL로 변환하는 Palantir AIP 스타일 엔진:
- 온톨로지 기반 스키마 컨텍스트
- 자동 조인 경로 추론
- 멀티턴 대화 지원
- Evidence Chain 연동

사용 예시:
    engine = NL2SQLEngine(context)
    result = await engine.query("지난 달 매출 TOP 10 고객")
    # result.sql → "SELECT customer_name, SUM(amount)..."
    # result.data → [{"customer_name": "...", ...}]
"""

from .signatures import NL2SQLSignature, JoinPathSignature, SQLValidationSignature
from .schema_context import SchemaContext, TableMeta, ColumnMeta
from .join_path_resolver import JoinPathResolver, JoinPath
from .query_executor import QueryExecutor, QueryResult
from .engine import NL2SQLEngine

__all__ = [
    # Signatures
    "NL2SQLSignature",
    "JoinPathSignature",
    "SQLValidationSignature",
    # Schema Context
    "SchemaContext",
    "TableMeta",
    "ColumnMeta",
    # Join Path
    "JoinPathResolver",
    "JoinPath",
    # Query Execution
    "QueryExecutor",
    "QueryResult",
    # Main Engine
    "NL2SQLEngine",
]
