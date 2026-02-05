"""
OSDK Query Builder (v17.0)

타입 안전 쿼리 빌더:
- 체이닝 API
- 타입 검증
- SQL 생성
- 결과 변환

사용 예시:
    result = (
        QueryBuilder(client, "Customer")
        .filter({"status": "active"})
        .select(["id", "name", "email"])
        .order_by("created_at", desc=True)
        .limit(10)
        .execute()
    )
"""

import json
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Union, Callable, TYPE_CHECKING
from datetime import datetime
from enum import Enum

if TYPE_CHECKING:
    from .models import ObjectType, ObjectInstance

logger = logging.getLogger(__name__)


class FilterOperator(str, Enum):
    """필터 연산자"""
    EQ = "eq"  # ==
    NE = "ne"  # !=
    GT = "gt"  # >
    GTE = "gte"  # >=
    LT = "lt"  # <
    LTE = "lte"  # <=
    IN = "in"  # IN (...)
    NOT_IN = "not_in"  # NOT IN (...)
    LIKE = "like"  # LIKE
    ILIKE = "ilike"  # ILIKE (case-insensitive)
    IS_NULL = "is_null"  # IS NULL
    IS_NOT_NULL = "is_not_null"  # IS NOT NULL
    BETWEEN = "between"  # BETWEEN


@dataclass
class FilterCondition:
    """필터 조건"""
    property_name: str
    operator: FilterOperator
    value: Any

    def to_sql(self, table_alias: str = "") -> str:
        """SQL WHERE 절 생성"""
        prefix = f"{table_alias}." if table_alias else ""
        col = f'{prefix}"{self.property_name}"'

        if self.operator == FilterOperator.EQ:
            return f"{col} = ?"
        elif self.operator == FilterOperator.NE:
            return f"{col} != ?"
        elif self.operator == FilterOperator.GT:
            return f"{col} > ?"
        elif self.operator == FilterOperator.GTE:
            return f"{col} >= ?"
        elif self.operator == FilterOperator.LT:
            return f"{col} < ?"
        elif self.operator == FilterOperator.LTE:
            return f"{col} <= ?"
        elif self.operator == FilterOperator.IN:
            placeholders = ", ".join(["?" for _ in self.value])
            return f"{col} IN ({placeholders})"
        elif self.operator == FilterOperator.NOT_IN:
            placeholders = ", ".join(["?" for _ in self.value])
            return f"{col} NOT IN ({placeholders})"
        elif self.operator == FilterOperator.LIKE:
            return f"{col} LIKE ?"
        elif self.operator == FilterOperator.ILIKE:
            return f"LOWER({col}) LIKE LOWER(?)"
        elif self.operator == FilterOperator.IS_NULL:
            return f"{col} IS NULL"
        elif self.operator == FilterOperator.IS_NOT_NULL:
            return f"{col} IS NOT NULL"
        elif self.operator == FilterOperator.BETWEEN:
            return f"{col} BETWEEN ? AND ?"
        else:
            return f"{col} = ?"

    def get_params(self) -> List[Any]:
        """SQL 파라미터 반환"""
        if self.operator in [FilterOperator.IS_NULL, FilterOperator.IS_NOT_NULL]:
            return []
        elif self.operator == FilterOperator.IN or self.operator == FilterOperator.NOT_IN:
            return list(self.value)
        elif self.operator == FilterOperator.BETWEEN:
            return list(self.value)  # [start, end]
        else:
            return [self.value]


@dataclass
class QueryResult:
    """쿼리 결과"""
    objects: List["ObjectInstance"] = field(default_factory=list)
    total_count: int = 0
    page: int = 1
    page_size: int = 100

    # 메타데이터
    query_time_ms: float = 0.0
    sql_generated: Optional[str] = None

    def first(self) -> Optional["ObjectInstance"]:
        """첫 번째 결과"""
        return self.objects[0] if self.objects else None

    def to_list(self) -> List[Dict[str, Any]]:
        """딕셔너리 리스트로 변환"""
        return [obj.to_dict() for obj in self.objects]

    def to_json(self) -> str:
        """JSON 문자열로 변환"""
        return json.dumps(self.to_list(), indent=2, default=str)


class QueryBuilder:
    """
    타입 안전 쿼리 빌더

    체이닝 API로 쿼리를 구성하고 실행합니다.
    """

    def __init__(
        self,
        client: Any,  # OSDKClient
        type_id: str,
    ):
        """
        Args:
            client: OSDK 클라이언트
            type_id: 조회할 ObjectType ID
        """
        self.client = client
        self.type_id = type_id

        # 쿼리 상태
        self._select_properties: Optional[List[str]] = None
        self._filters: List[FilterCondition] = []
        self._order_by: List[tuple] = []  # [(property, desc), ...]
        self._limit: Optional[int] = None
        self._offset: Optional[int] = None
        self._include_links: List[str] = []

        # ObjectType 참조
        self._object_type: Optional["ObjectType"] = None

    def select(self, properties: List[str]) -> "QueryBuilder":
        """
        조회할 속성 지정

        Args:
            properties: 속성 이름 리스트

        Returns:
            self (체이닝)
        """
        self._select_properties = properties
        return self

    def filter(
        self,
        conditions: Union[Dict[str, Any], FilterCondition, List[FilterCondition]]
    ) -> "QueryBuilder":
        """
        필터 조건 추가

        Args:
            conditions: 딕셔너리 또는 FilterCondition

        Returns:
            self (체이닝)

        Examples:
            .filter({"status": "active"})
            .filter({"age": {"$gt": 18}})
            .filter([FilterCondition("name", FilterOperator.LIKE, "%John%")])
        """
        if isinstance(conditions, FilterCondition):
            self._filters.append(conditions)
        elif isinstance(conditions, list):
            self._filters.extend(conditions)
        elif isinstance(conditions, dict):
            for key, value in conditions.items():
                if isinstance(value, dict):
                    # 연산자 지정: {"age": {"$gt": 18}}
                    for op, v in value.items():
                        operator = self._parse_operator(op)
                        self._filters.append(FilterCondition(key, operator, v))
                else:
                    # 기본 등호: {"status": "active"}
                    self._filters.append(FilterCondition(key, FilterOperator.EQ, value))
        return self

    def _parse_operator(self, op: str) -> FilterOperator:
        """연산자 문자열 파싱"""
        op_map = {
            "$eq": FilterOperator.EQ,
            "$ne": FilterOperator.NE,
            "$gt": FilterOperator.GT,
            "$gte": FilterOperator.GTE,
            "$lt": FilterOperator.LT,
            "$lte": FilterOperator.LTE,
            "$in": FilterOperator.IN,
            "$nin": FilterOperator.NOT_IN,
            "$like": FilterOperator.LIKE,
            "$ilike": FilterOperator.ILIKE,
            "$null": FilterOperator.IS_NULL,
            "$notnull": FilterOperator.IS_NOT_NULL,
            "$between": FilterOperator.BETWEEN,
        }
        return op_map.get(op, FilterOperator.EQ)

    def where(
        self,
        property_name: str,
        operator: FilterOperator,
        value: Any
    ) -> "QueryBuilder":
        """
        명시적 필터 조건 추가

        Args:
            property_name: 속성 이름
            operator: 연산자
            value: 값

        Returns:
            self (체이닝)
        """
        self._filters.append(FilterCondition(property_name, operator, value))
        return self

    def order_by(
        self,
        property_name: str,
        desc: bool = False
    ) -> "QueryBuilder":
        """
        정렬 조건 추가

        Args:
            property_name: 정렬할 속성
            desc: 내림차순 여부

        Returns:
            self (체이닝)
        """
        self._order_by.append((property_name, desc))
        return self

    def limit(self, count: int) -> "QueryBuilder":
        """
        결과 개수 제한

        Args:
            count: 최대 결과 수

        Returns:
            self (체이닝)
        """
        self._limit = count
        return self

    def offset(self, count: int) -> "QueryBuilder":
        """
        결과 오프셋

        Args:
            count: 건너뛸 결과 수

        Returns:
            self (체이닝)
        """
        self._offset = count
        return self

    def page(self, page_num: int, page_size: int = 100) -> "QueryBuilder":
        """
        페이지네이션

        Args:
            page_num: 페이지 번호 (1부터 시작)
            page_size: 페이지당 결과 수

        Returns:
            self (체이닝)
        """
        self._offset = (page_num - 1) * page_size
        self._limit = page_size
        return self

    def include(self, link_names: Union[str, List[str]]) -> "QueryBuilder":
        """
        관련 객체 포함

        Args:
            link_names: 포함할 링크 이름

        Returns:
            self (체이닝)
        """
        if isinstance(link_names, str):
            link_names = [link_names]
        self._include_links.extend(link_names)
        return self

    def build_sql(self) -> tuple:
        """
        SQL 쿼리 생성

        Returns:
            (sql_string, params_list)
        """
        # 테이블 이름 (소문자)
        table_name = self.type_id

        # SELECT 절
        if self._select_properties:
            cols = ", ".join([f'"{p}"' for p in self._select_properties])
        else:
            cols = "*"

        sql = f'SELECT {cols} FROM "{table_name}"'
        params = []

        # WHERE 절
        if self._filters:
            where_clauses = []
            for f in self._filters:
                where_clauses.append(f.to_sql())
                params.extend(f.get_params())
            sql += " WHERE " + " AND ".join(where_clauses)

        # ORDER BY 절
        if self._order_by:
            order_clauses = []
            for prop, desc in self._order_by:
                direction = "DESC" if desc else "ASC"
                order_clauses.append(f'"{prop}" {direction}')
            sql += " ORDER BY " + ", ".join(order_clauses)

        # LIMIT / OFFSET
        if self._limit is not None:
            sql += f" LIMIT {self._limit}"
        if self._offset is not None:
            sql += f" OFFSET {self._offset}"

        return sql, params

    def execute(self) -> QueryResult:
        """
        쿼리 실행

        Returns:
            QueryResult
        """
        import time
        start_time = time.time()

        sql, params = self.build_sql()
        logger.debug(f"Executing OSDK query: {sql} with params: {params}")

        # 클라이언트를 통해 실행
        result = self.client._execute_query(sql, params, self.type_id)

        result.query_time_ms = (time.time() - start_time) * 1000
        result.sql_generated = sql

        return result

    def count(self) -> int:
        """결과 개수만 조회"""
        table_name = self.type_id
        sql = f'SELECT COUNT(*) as cnt FROM "{table_name}"'
        params = []

        if self._filters:
            where_clauses = []
            for f in self._filters:
                where_clauses.append(f.to_sql())
                params.extend(f.get_params())
            sql += " WHERE " + " AND ".join(where_clauses)

        result = self.client._execute_raw_query(sql, params)
        if result and len(result) > 0:
            return result[0].get("cnt", 0)
        return 0

    def exists(self) -> bool:
        """결과 존재 여부"""
        return self.count() > 0

    def first(self) -> Optional["ObjectInstance"]:
        """첫 번째 결과만 조회"""
        self._limit = 1
        result = self.execute()
        return result.first()

    def to_list(self) -> List[Dict[str, Any]]:
        """딕셔너리 리스트로 결과 반환"""
        return self.execute().to_list()
