"""
NL2SQL Engine (v17.0)

자연어 → SQL 변환의 메인 엔진:
- DSPy 기반 구조화된 LLM 호출
- 온톨로지 기반 스키마 컨텍스트
- 자동 조인 경로 추론
- 멀티턴 대화 지원
- Evidence Chain 연동

사용 예시:
    engine = NL2SQLEngine(context)
    result = await engine.query("지난 달 매출 TOP 10 고객")
"""

import json
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, TYPE_CHECKING
from datetime import datetime

try:
    import dspy
except ImportError:
    dspy = None  # Optional dependency

from .signatures import (
    NL2SQLSignature,
    QueryIntentSignature,
    JoinPathSignature,
    SQLValidationSignature,
    SQLExplanationSignature,
    QueryClarificationSignature,
)
from .schema_context import SchemaContext
from .join_path_resolver import JoinPathResolver, JoinPath
from .query_executor import QueryExecutor, QueryResult

if TYPE_CHECKING:
    from ...shared_context import SharedContext

logger = logging.getLogger(__name__)


@dataclass
class NL2SQLConfig:
    """NL2SQL 엔진 설정"""
    dialect: str = "sqlite"
    max_join_hops: int = 4
    enable_validation: bool = True
    enable_explanation: bool = True
    auto_execute: bool = True
    max_retries: int = 2
    confidence_threshold: float = 0.7


@dataclass
class ConversationTurn:
    """대화 턴"""
    natural_query: str
    sql_query: str
    result_preview: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


class NL2SQLEngine:
    """
    자연어 → SQL 변환 엔진

    Palantir AIP 스타일의 NL2SQL 기능 구현:
    1. 쿼리 의도 분석
    2. 스키마 컨텍스트 기반 SQL 생성
    3. 조인 경로 자동 추론
    4. SQL 검증 및 수정
    5. 결과 설명 생성
    """

    def __init__(
        self,
        context: Optional["SharedContext"] = None,
        config: Optional[NL2SQLConfig] = None,
    ):
        """
        Args:
            context: SharedContext
            config: 엔진 설정
        """
        self.context = context
        self.config = config or NL2SQLConfig()

        # 스키마 컨텍스트
        self.schema_context = SchemaContext(context)

        # 조인 경로 해결기
        self.join_resolver = JoinPathResolver(self.schema_context)

        # 쿼리 실행기
        self.executor = QueryExecutor(
            context=context,
            dialect=self.config.dialect,
        )

        # 대화 이력
        self.conversation_history: List[ConversationTurn] = []

        # DSPy 모듈 초기화
        self._intent_module = None
        self._nl2sql_module = None
        self._validation_module = None
        self._explanation_module = None

        if dspy:
            self._init_dspy_modules()

        logger.info("NL2SQLEngine initialized")

    def _init_dspy_modules(self) -> None:
        """DSPy 모듈 초기화"""
        try:
            self._intent_module = dspy.ChainOfThought(QueryIntentSignature)
            self._nl2sql_module = dspy.ChainOfThought(NL2SQLSignature)
            self._validation_module = dspy.Predict(SQLValidationSignature)
            self._explanation_module = dspy.Predict(SQLExplanationSignature)
            logger.debug("DSPy modules initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize DSPy modules: {e}")

    async def query(
        self,
        natural_query: str,
        execute: Optional[bool] = None,
        include_explanation: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """
        자연어 쿼리 처리

        Args:
            natural_query: 자연어 쿼리
            execute: SQL 실행 여부 (None이면 config 따름)
            include_explanation: 결과 설명 포함 여부

        Returns:
            {
                "sql": str,
                "result": QueryResult (if executed),
                "explanation": str (if requested),
                "confidence": float,
                "tables_used": List[str],
            }
        """
        execute = execute if execute is not None else self.config.auto_execute
        include_explanation = include_explanation if include_explanation is not None else self.config.enable_explanation

        response = {
            "natural_query": natural_query,
            "sql": None,
            "result": None,
            "explanation": None,
            "confidence": 0.0,
            "tables_used": [],
            "success": False,
            "error": None,
        }

        try:
            # 1. 쿼리 의도 분석
            intent = await self._analyze_intent(natural_query)
            response["intent"] = intent

            # 2. 조인 경로 결정
            join_paths = self._resolve_join_paths(intent.get("target_tables", []))
            response["join_paths"] = [jp.tables for jp in join_paths.values()] if join_paths else []

            # 3. SQL 생성
            sql_result = await self._generate_sql(natural_query, intent, join_paths)
            response["sql"] = sql_result.get("sql")
            response["confidence"] = sql_result.get("confidence", 0.0)
            response["tables_used"] = sql_result.get("tables_used", [])

            # 4. SQL 검증
            if self.config.enable_validation and response["sql"]:
                validation = await self._validate_sql(response["sql"])
                if not validation.get("is_valid"):
                    # 수정된 SQL 사용
                    if validation.get("corrected_sql"):
                        response["sql"] = validation["corrected_sql"]
                        response["validation_issues"] = validation.get("issues", [])

            # 5. 실행
            if execute and response["sql"]:
                result = self.executor.execute(response["sql"])
                response["result"] = result.to_dict()
                response["success"] = result.success

                # 6. 설명 생성
                if include_explanation and result.success:
                    explanation = await self._generate_explanation(
                        response["sql"],
                        result,
                        natural_query
                    )
                    response["explanation"] = explanation

                # 대화 이력 추가
                self.conversation_history.append(ConversationTurn(
                    natural_query=natural_query,
                    sql_query=response["sql"],
                    result_preview=result.get_preview(3) if result.success else None,
                ))

                # SharedContext에 기록
                if self.context:
                    self.context.add_nl2sql_query(
                        natural_query=natural_query,
                        sql_query=response["sql"],
                        result_preview=result.to_dict() if result.success else None,
                        confidence=response["confidence"],
                    )

            else:
                response["success"] = True  # SQL만 생성해도 성공

            # Evidence Chain 기록
            self._record_evidence(response)

        except Exception as e:
            logger.error(f"NL2SQL query failed: {e}")
            response["error"] = str(e)
            response["success"] = False

        return response

    async def _analyze_intent(self, natural_query: str) -> Dict[str, Any]:
        """쿼리 의도 분석"""
        if self._intent_module:
            try:
                result = self._intent_module(
                    natural_query=natural_query,
                    available_tables=self.schema_context.get_table_names(),
                    schema_summary=self._get_schema_summary(),
                )
                return {
                    "query_type": getattr(result, "query_type", "select"),
                    "target_tables": getattr(result, "target_tables", []),
                    "time_range": getattr(result, "time_range", None),
                    "aggregations": getattr(result, "aggregations", []),
                    "filters": getattr(result, "filters", []),
                    "sort_by": getattr(result, "sort_by", None),
                    "limit": getattr(result, "limit", None),
                    "reasoning": getattr(result, "reasoning", ""),
                }
            except Exception as e:
                logger.warning(f"Intent analysis failed: {e}")

        # 폴백: 간단한 휴리스틱
        return self._heuristic_intent(natural_query)

    def _heuristic_intent(self, natural_query: str) -> Dict[str, Any]:
        """휴리스틱 기반 의도 분석 (DSPy 없을 때)"""
        query_lower = natural_query.lower()

        intent = {
            "query_type": "select",
            "target_tables": [],
            "time_range": None,
            "aggregations": [],
            "filters": [],
            "sort_by": None,
            "limit": None,
        }

        # 집계 감지
        if any(word in query_lower for word in ["합계", "총", "sum", "total"]):
            intent["aggregations"].append("SUM")
            intent["query_type"] = "aggregate"
        if any(word in query_lower for word in ["평균", "avg", "average"]):
            intent["aggregations"].append("AVG")
            intent["query_type"] = "aggregate"
        if any(word in query_lower for word in ["개수", "count", "몇 개"]):
            intent["aggregations"].append("COUNT")
            intent["query_type"] = "aggregate"

        # TOP N 감지
        import re
        top_match = re.search(r"top\s*(\d+)|상위\s*(\d+)", query_lower)
        if top_match:
            intent["limit"] = int(top_match.group(1) or top_match.group(2))
            intent["query_type"] = "top_n"

        # 시간 범위 감지
        if "지난 달" in query_lower or "last month" in query_lower:
            intent["time_range"] = "last_month"
        elif "이번 달" in query_lower or "this month" in query_lower:
            intent["time_range"] = "this_month"
        elif "오늘" in query_lower or "today" in query_lower:
            intent["time_range"] = "today"

        # 테이블 추측 (테이블 이름이 쿼리에 포함되어 있는지)
        for table_name in self.schema_context.get_table_names():
            if table_name.lower() in query_lower:
                intent["target_tables"].append(table_name)

        return intent

    def _get_schema_summary(self) -> str:
        """스키마 요약 문자열"""
        tables = self.schema_context.get_table_names()
        fk_count = len(self.schema_context.foreign_keys)
        return f"{len(tables)} tables, {fk_count} relationships"

    def _resolve_join_paths(
        self,
        tables: List[str]
    ) -> Dict[str, JoinPath]:
        """조인 경로 해결"""
        if len(tables) <= 1:
            return {}

        paths = {}
        base_table = tables[0]

        for target in tables[1:]:
            path = self.join_resolver.find_path(base_table, target)
            if path:
                paths[f"{base_table}->{target}"] = path

        return paths

    async def _generate_sql(
        self,
        natural_query: str,
        intent: Dict[str, Any],
        join_paths: Dict[str, JoinPath]
    ) -> Dict[str, Any]:
        """SQL 생성"""
        if self._nl2sql_module:
            try:
                # 대화 컨텍스트
                conv_context = None
                if self.conversation_history:
                    conv_context = json.dumps([
                        {"q": t.natural_query, "sql": t.sql_query}
                        for t in self.conversation_history[-3:]
                    ])

                # 조인 경로 JSON
                join_paths_json = json.dumps({
                    key: {"tables": jp.tables, "conditions": jp.conditions}
                    for key, jp in join_paths.items()
                })

                result = self._nl2sql_module(
                    natural_query=natural_query,
                    schema_context=self.schema_context.to_compact_json(),
                    query_intent=json.dumps(intent),
                    join_paths=join_paths_json,
                    dialect=self.config.dialect,
                    conversation_context=conv_context,
                )

                return {
                    "sql": getattr(result, "sql_query", ""),
                    "explanation": getattr(result, "explanation", ""),
                    "tables_used": getattr(result, "tables_used", []),
                    "columns_used": getattr(result, "columns_used", []),
                    "confidence": getattr(result, "confidence", 0.5),
                }
            except Exception as e:
                logger.warning(f"SQL generation with DSPy failed: {e}")

        # 폴백: 간단한 SQL 생성
        return self._heuristic_sql(natural_query, intent)

    def _heuristic_sql(
        self,
        natural_query: str,
        intent: Dict[str, Any]
    ) -> Dict[str, Any]:
        """휴리스틱 기반 SQL 생성 (DSPy 없을 때)"""
        tables = intent.get("target_tables", [])
        if not tables:
            tables = self.schema_context.get_table_names()[:1]

        if not tables:
            return {"sql": "", "confidence": 0.0, "tables_used": []}

        table = tables[0]
        sql = f'SELECT * FROM "{table}"'

        limit = intent.get("limit")
        if limit:
            sql += f" LIMIT {limit}"

        return {
            "sql": sql,
            "confidence": 0.3,
            "tables_used": [table],
            "columns_used": [],
        }

    async def _validate_sql(self, sql: str) -> Dict[str, Any]:
        """SQL 검증"""
        if self._validation_module:
            try:
                result = self._validation_module(
                    sql_query=sql,
                    schema_context=self.schema_context.to_compact_json(),
                    dialect=self.config.dialect,
                )
                return {
                    "is_valid": getattr(result, "is_valid", True),
                    "issues": getattr(result, "issues", []),
                    "suggestions": getattr(result, "suggestions", []),
                    "corrected_sql": getattr(result, "corrected_sql", None),
                }
            except Exception as e:
                logger.warning(f"SQL validation failed: {e}")

        # 폴백: 실행기의 validate_sql 사용
        return self.executor.validate_sql(sql)

    async def _generate_explanation(
        self,
        sql: str,
        result: "QueryResult",
        original_question: str
    ) -> str:
        """결과 설명 생성"""
        if self._explanation_module:
            try:
                explain_result = self._explanation_module(
                    sql_query=sql,
                    query_results=json.dumps(result.data[:10], default=str),
                    original_question=original_question,
                    row_count=result.row_count,
                )
                return getattr(explain_result, "summary", "")
            except Exception as e:
                logger.warning(f"Explanation generation failed: {e}")

        # 폴백
        return f"Query returned {result.row_count} rows."

    def _record_evidence(self, response: Dict[str, Any]) -> None:
        """Evidence Chain에 기록"""
        if not self.context or not self.context.evidence_chain:
            return

        try:
            from ...autonomous.evidence_chain import EvidenceType

            self.context.evidence_chain.add_block(
                phase="aip_nl2sql",
                agent="NL2SQLEngine",
                evidence_type=EvidenceType.NL2SQL_TRANSLATION,
                finding=f"Translated: {response['natural_query'][:100]}",
                reasoning=f"Generated SQL for {len(response.get('tables_used', []))} tables",
                conclusion="Query executed successfully" if response["success"] else f"Failed: {response.get('error')}",
                data_references=response.get("tables_used", []),
                metrics={
                    "confidence": response.get("confidence", 0),
                    "tables_count": len(response.get("tables_used", [])),
                    "success": response["success"],
                },
                confidence=response.get("confidence", 0.5),
            )
        except Exception as e:
            logger.debug(f"Failed to record evidence: {e}")

    def get_conversation_context(self, limit: int = 5) -> List[Dict[str, Any]]:
        """최근 대화 컨텍스트 반환"""
        return [
            {
                "query": turn.natural_query,
                "sql": turn.sql_query,
                "preview": turn.result_preview,
            }
            for turn in self.conversation_history[-limit:]
        ]

    def clear_conversation(self) -> None:
        """대화 이력 초기화"""
        self.conversation_history = []
        logger.info("Conversation history cleared")

    def get_statistics(self) -> Dict[str, Any]:
        """엔진 통계"""
        return {
            "total_queries": len(self.conversation_history),
            "executor_stats": self.executor.get_statistics(),
            "schema_tables": len(self.schema_context.tables),
            "schema_fks": len(self.schema_context.foreign_keys),
        }
