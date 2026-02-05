"""
Query Executor for NL2SQL (v17.0)

SQL 쿼리 실행 및 결과 관리:
- SQLite/PostgreSQL/MySQL 지원
- CSV 파일 기반 쿼리 지원
- 결과 포매팅 및 페이지네이션
- 실행 통계 및 프로파일링
"""

import json
import logging
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, TYPE_CHECKING
from datetime import datetime

if TYPE_CHECKING:
    from ...shared_context import SharedContext

logger = logging.getLogger(__name__)


@dataclass
class QueryResult:
    """쿼리 실행 결과"""
    # 데이터
    data: List[Dict[str, Any]] = field(default_factory=list)
    columns: List[str] = field(default_factory=list)
    row_count: int = 0
    total_rows: int = 0  # 전체 행 수 (LIMIT 전)

    # 쿼리 정보
    sql: str = ""
    execution_time_ms: float = 0.0

    # 상태
    success: bool = True
    error: Optional[str] = None

    # 메타데이터
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "data": self.data[:100],  # 최대 100행
            "columns": self.columns,
            "row_count": self.row_count,
            "total_rows": self.total_rows,
            "sql": self.sql,
            "execution_time_ms": self.execution_time_ms,
            "success": self.success,
            "error": self.error,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False, default=str)

    def get_preview(self, max_rows: int = 5) -> str:
        """결과 미리보기 문자열"""
        if not self.data:
            return "No results"

        lines = [f"Columns: {', '.join(self.columns)}"]
        lines.append("-" * 50)

        for row in self.data[:max_rows]:
            values = [str(row.get(col, ""))[:30] for col in self.columns]
            lines.append(" | ".join(values))

        if self.row_count > max_rows:
            lines.append(f"... and {self.row_count - max_rows} more rows")

        return "\n".join(lines)


class QueryExecutor:
    """
    SQL 쿼리 실행기

    여러 데이터 소스(SQLite, CSV, PostgreSQL)를 지원합니다.
    """

    def __init__(
        self,
        context: Optional["SharedContext"] = None,
        dialect: str = "sqlite",
        data_directory: Optional[str] = None,
    ):
        """
        Args:
            context: SharedContext (테이블 정보 소스)
            dialect: SQL 방언 (sqlite, postgresql, mysql)
            data_directory: CSV 파일 디렉토리 (SQLite 미사용 시)
        """
        self.context = context
        self.dialect = dialect
        self.data_directory = data_directory or (
            context.data_directory if context else None
        )

        # SQLite 인메모리 DB (CSV 로드용)
        self._conn: Optional[sqlite3.Connection] = None
        self._tables_loaded: bool = False

        # 실행 통계
        self.execution_stats = {
            "total_queries": 0,
            "successful_queries": 0,
            "failed_queries": 0,
            "total_time_ms": 0.0,
        }

    def _ensure_connection(self) -> sqlite3.Connection:
        """SQLite 연결 확보"""
        if self._conn is None:
            self._conn = sqlite3.connect(":memory:")
            self._conn.row_factory = sqlite3.Row
            logger.info("SQLite in-memory connection created")

        if not self._tables_loaded and self.context:
            self._load_tables_from_context()

        return self._conn

    def _load_tables_from_context(self) -> None:
        """SharedContext의 테이블 데이터를 SQLite에 로드"""
        if self._tables_loaded:
            return

        conn = self._conn
        if not conn:
            return

        cursor = conn.cursor()

        for table_name, table_info in self.context.tables.items():
            try:
                # 테이블 생성
                columns_def = []
                for col in table_info.columns:
                    col_name = col.get("name", "")
                    col_type = self._map_type_to_sqlite(
                        col.get("data_type", col.get("type", "TEXT"))
                    )
                    columns_def.append(f'"{col_name}" {col_type}')

                if columns_def:
                    create_sql = f'CREATE TABLE IF NOT EXISTS "{table_name}" ({", ".join(columns_def)})'
                    cursor.execute(create_sql)

                    # 데이터 삽입 (CSV 파일에서)
                    self._load_csv_data(cursor, table_name, table_info)

            except Exception as e:
                logger.warning(f"Failed to load table {table_name}: {e}")

        conn.commit()
        self._tables_loaded = True
        logger.info(f"Loaded {len(self.context.tables)} tables into SQLite")

    def _load_csv_data(self, cursor, table_name: str, table_info: Any) -> None:
        """CSV 파일에서 데이터 로드"""
        import csv

        # 파일 경로 찾기
        file_path = None
        if hasattr(table_info, "metadata") and table_info.metadata:
            file_path = table_info.metadata.get("file_path")

        if not file_path and self.data_directory:
            potential_path = Path(self.data_directory) / f"{table_name}.csv"
            if potential_path.exists():
                file_path = str(potential_path)

        if not file_path:
            # sample_data 사용
            if table_info.sample_data:
                columns = [col.get("name") for col in table_info.columns]
                placeholders = ", ".join(["?" for _ in columns])
                insert_sql = f'INSERT INTO "{table_name}" VALUES ({placeholders})'

                for row in table_info.sample_data[:1000]:  # 최대 1000행
                    values = [row.get(col, None) for col in columns]
                    cursor.execute(insert_sql, values)
            return

        # CSV 파일 로드
        try:
            with open(file_path, "r", encoding="utf-8-sig") as f:
                reader = csv.DictReader(f)
                columns = reader.fieldnames or []

                if not columns:
                    return

                placeholders = ", ".join(["?" for _ in columns])
                cols_quoted = ", ".join([f'"{c}"' for c in columns])
                insert_sql = f'INSERT INTO "{table_name}" ({cols_quoted}) VALUES ({placeholders})'

                batch = []
                for row in reader:
                    values = [row.get(col, None) for col in columns]
                    batch.append(values)

                    if len(batch) >= 1000:
                        cursor.executemany(insert_sql, batch)
                        batch = []

                if batch:
                    cursor.executemany(insert_sql, batch)

                logger.debug(f"Loaded CSV data for {table_name} from {file_path}")

        except Exception as e:
            logger.warning(f"Failed to load CSV {file_path}: {e}")

    def _map_type_to_sqlite(self, data_type: str) -> str:
        """데이터 타입을 SQLite 타입으로 매핑"""
        data_type = data_type.lower()

        if "int" in data_type:
            return "INTEGER"
        elif "float" in data_type or "double" in data_type or "decimal" in data_type:
            return "REAL"
        elif "bool" in data_type:
            return "INTEGER"
        elif "date" in data_type or "time" in data_type:
            return "TEXT"  # SQLite는 날짜를 문자열로
        else:
            return "TEXT"

    def execute(
        self,
        sql: str,
        params: Optional[tuple] = None,
        limit: int = 1000,
    ) -> QueryResult:
        """
        SQL 쿼리 실행

        Args:
            sql: 실행할 SQL
            params: 바인딩 파라미터
            limit: 최대 결과 행 수

        Returns:
            QueryResult
        """
        self.execution_stats["total_queries"] += 1
        start_time = time.time()

        try:
            conn = self._ensure_connection()
            cursor = conn.cursor()

            # 쿼리 실행
            if params:
                cursor.execute(sql, params)
            else:
                cursor.execute(sql)

            # 결과 가져오기
            rows = cursor.fetchmany(limit)
            columns = [description[0] for description in cursor.description] if cursor.description else []

            # dict로 변환
            data = [dict(zip(columns, row)) for row in rows]

            # 전체 행 수 확인 (SELECT만)
            total_rows = len(data)
            if sql.strip().upper().startswith("SELECT"):
                # 대략적인 전체 행 수
                remaining = cursor.fetchall()
                total_rows += len(remaining)

            execution_time = (time.time() - start_time) * 1000
            self.execution_stats["successful_queries"] += 1
            self.execution_stats["total_time_ms"] += execution_time

            return QueryResult(
                data=data,
                columns=columns,
                row_count=len(data),
                total_rows=total_rows,
                sql=sql,
                execution_time_ms=execution_time,
                success=True,
            )

        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            self.execution_stats["failed_queries"] += 1
            self.execution_stats["total_time_ms"] += execution_time

            logger.error(f"Query execution failed: {e}\nSQL: {sql}")

            return QueryResult(
                sql=sql,
                execution_time_ms=execution_time,
                success=False,
                error=str(e),
            )

    def validate_sql(self, sql: str) -> Dict[str, Any]:
        """
        SQL 검증 (실행 없이)

        Returns:
            {"valid": bool, "error": Optional[str], "tables": List[str]}
        """
        try:
            conn = self._ensure_connection()

            # EXPLAIN으로 검증
            explain_sql = f"EXPLAIN {sql}"
            cursor = conn.cursor()
            cursor.execute(explain_sql)

            # 사용된 테이블 추출 (간단한 파싱)
            tables = []
            sql_upper = sql.upper()
            for keyword in ["FROM", "JOIN"]:
                idx = 0
                while True:
                    idx = sql_upper.find(keyword, idx)
                    if idx == -1:
                        break
                    # 키워드 다음 단어가 테이블명
                    rest = sql[idx + len(keyword):].strip()
                    parts = rest.split()
                    if parts:
                        table_name = parts[0].strip('"').strip("'").strip("`")
                        if table_name and table_name.upper() not in ["ON", "WHERE", "LEFT", "RIGHT", "INNER", "OUTER"]:
                            tables.append(table_name)
                    idx += len(keyword)

            return {
                "valid": True,
                "error": None,
                "tables": list(set(tables)),
            }

        except Exception as e:
            return {
                "valid": False,
                "error": str(e),
                "tables": [],
            }

    def get_table_preview(
        self,
        table_name: str,
        limit: int = 5
    ) -> QueryResult:
        """테이블 미리보기"""
        sql = f'SELECT * FROM "{table_name}" LIMIT {limit}'
        return self.execute(sql)

    def get_statistics(self) -> Dict[str, Any]:
        """실행 통계 반환"""
        stats = self.execution_stats.copy()
        if stats["total_queries"] > 0:
            stats["avg_time_ms"] = stats["total_time_ms"] / stats["total_queries"]
            stats["success_rate"] = stats["successful_queries"] / stats["total_queries"]
        return stats

    def close(self) -> None:
        """연결 종료"""
        if self._conn:
            self._conn.close()
            self._conn = None
            self._tables_loaded = False
            logger.info("SQLite connection closed")
