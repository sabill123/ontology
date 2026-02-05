"""
OQL Query (v2.0)

Object Query Language의 쿼리 표현 및 빌더

팔란티어 검증 기준:
✅ SQL 대신 개념 수준 쿼리
✅ 플루언트 API로 타입 안전 쿼리 작성
✅ 관계 기반 eager loading 지원

사용 예시:
    # 플루언트 빌더
    orders = await oql.query("Order") \\
        .filter("status", "==", "pending") \\
        .filter("risk_score", ">", 0.8) \\
        .include("customer", "line_items") \\
        .select("id", "total", "customer.name") \\
        .order_by("created_at", desc=True) \\
        .limit(100) \\
        .execute()

    # 첫 번째 결과만
    order = await oql.query("Order") \\
        .filter("order_id", "==", "ORD-001") \\
        .first()

    # 존재 여부
    exists = await oql.query("Order") \\
        .filter("customer_id", "==", "CUST-001") \\
        .exists()
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Union, TYPE_CHECKING
from enum import Enum

if TYPE_CHECKING:
    from .engine import OQLEngine
    from ..osdk.models import ObjectInstance


class FilterOperator(Enum):
    """필터 연산자"""
    # 비교
    EQ = "=="
    NE = "!="
    GT = ">"
    GTE = ">="
    LT = "<"
    LTE = "<="

    # 집합
    IN = "in"
    NOT_IN = "not_in"

    # 문자열
    CONTAINS = "contains"
    STARTS_WITH = "starts_with"
    ENDS_WITH = "ends_with"
    MATCHES = "matches"  # regex

    # NULL
    IS_NULL = "is_null"
    IS_NOT_NULL = "is_not_null"

    # 범위
    BETWEEN = "between"


@dataclass
class FilterClause:
    """
    필터 조건

    지원하는 속성 접근:
    - 단순: "status"
    - 중첩: "customer.name" (관계 통한 접근)
    - 집합: "tags" (배열 속성)
    """
    property: str                    # 속성명 (점 표기법 지원)
    operator: FilterOperator
    value: Any = None                # 값 (IS_NULL 등은 None)
    value2: Any = None               # BETWEEN용 두 번째 값

    def to_dict(self) -> Dict[str, Any]:
        return {
            "property": self.property,
            "operator": self.operator.value,
            "value": self.value,
            "value2": self.value2,
        }


@dataclass
class SortClause:
    """정렬 조건"""
    property: str
    descending: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "property": self.property,
            "descending": self.descending,
        }


@dataclass
class OQLQuery:
    """
    OQL 쿼리 표현

    SQL 없이 개념 수준 쿼리를 표현:
    - object_type: 대상 ObjectType (테이블이 아님!)
    - filters: 조건
    - includes: 관계 (eager loading)
    - selects: 속성 선택 (projection)
    """
    object_type: str

    # 필터
    filters: List[FilterClause] = field(default_factory=list)

    # 관계 포함 (eager loading)
    includes: List[str] = field(default_factory=list)

    # 속성 선택
    selects: List[str] = field(default_factory=list)

    # 정렬
    order_by: List[SortClause] = field(default_factory=list)

    # 페이징
    limit: Optional[int] = None
    offset: Optional[int] = None

    # 집계
    group_by: List[str] = field(default_factory=list)
    aggregations: Dict[str, str] = field(default_factory=dict)
    # 예: {"total_amount": "sum(amount)", "count": "count(*)"}

    # 고급 옵션
    distinct: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "object_type": self.object_type,
            "filters": [f.to_dict() for f in self.filters],
            "includes": self.includes,
            "selects": self.selects,
            "order_by": [s.to_dict() for s in self.order_by],
            "limit": self.limit,
            "offset": self.offset,
            "group_by": self.group_by,
            "aggregations": self.aggregations,
            "distinct": self.distinct,
        }


class OQLQueryBuilder:
    """
    플루언트 쿼리 빌더

    팔란티어 검증 기준:
    ✅ SQL 대신 개념 수준 쿼리
    ✅ 타입 안전한 플루언트 API
    ✅ 테이블명 없이 개념으로 쿼리

    사용 예:
    ```python
    orders = await oql.query("Order") \\
        .filter("status", "==", "pending") \\
        .filter("risk_score", ">", 0.8) \\
        .include("customer", "line_items") \\
        .select("id", "total", "customer.name") \\
        .order_by("created_at", desc=True) \\
        .limit(100) \\
        .execute()
    ```
    """

    def __init__(self, engine: "OQLEngine", object_type: str):
        self.engine = engine
        self.query = OQLQuery(object_type=object_type)

    def filter(
        self,
        property: str,
        operator: Union[str, FilterOperator],
        value: Any = None,
        value2: Any = None,
    ) -> "OQLQueryBuilder":
        """
        필터 추가

        Args:
            property: 속성명 (점 표기법 지원: "customer.name")
            operator: 연산자 ("==", ">", "in" 등)
            value: 비교 값
            value2: BETWEEN용 두 번째 값

        예:
            .filter("status", "==", "pending")
            .filter("risk_score", ">", 0.8)
            .filter("created_at", "between", "2024-01-01", "2024-12-31")
            .filter("tags", "contains", "urgent")
        """
        if isinstance(operator, str):
            operator = FilterOperator(operator)

        self.query.filters.append(FilterClause(
            property=property,
            operator=operator,
            value=value,
            value2=value2,
        ))
        return self

    def where(self, property: str, value: Any) -> "OQLQueryBuilder":
        """
        간단한 동등 필터 (filter의 축약)

        예: .where("status", "pending") == .filter("status", "==", "pending")
        """
        return self.filter(property, FilterOperator.EQ, value)

    def where_in(self, property: str, values: List[Any]) -> "OQLQueryBuilder":
        """IN 필터"""
        return self.filter(property, FilterOperator.IN, values)

    def where_not_null(self, property: str) -> "OQLQueryBuilder":
        """NOT NULL 필터"""
        return self.filter(property, FilterOperator.IS_NOT_NULL)

    def where_null(self, property: str) -> "OQLQueryBuilder":
        """NULL 필터"""
        return self.filter(property, FilterOperator.IS_NULL)

    def where_between(self, property: str, min_value: Any, max_value: Any) -> "OQLQueryBuilder":
        """BETWEEN 필터"""
        return self.filter(property, FilterOperator.BETWEEN, min_value, max_value)

    def include(self, *relations: str) -> "OQLQueryBuilder":
        """
        관계 포함 (eager loading)

        예: .include("customer", "line_items")
        """
        self.query.includes.extend(relations)
        return self

    def select(self, *properties: str) -> "OQLQueryBuilder":
        """
        속성 선택 (projection)

        예: .select("id", "total", "customer.name")
        """
        self.query.selects.extend(properties)
        return self

    def order_by(self, property: str, desc: bool = False) -> "OQLQueryBuilder":
        """
        정렬

        예: .order_by("created_at", desc=True)
        """
        self.query.order_by.append(SortClause(property=property, descending=desc))
        return self

    def limit(self, n: int) -> "OQLQueryBuilder":
        """결과 수 제한"""
        self.query.limit = n
        return self

    def offset(self, n: int) -> "OQLQueryBuilder":
        """오프셋"""
        self.query.offset = n
        return self

    def page(self, page_number: int, page_size: int = 20) -> "OQLQueryBuilder":
        """페이지네이션 (편의 메서드)"""
        self.query.offset = (page_number - 1) * page_size
        self.query.limit = page_size
        return self

    def distinct(self) -> "OQLQueryBuilder":
        """중복 제거"""
        self.query.distinct = True
        return self

    def group_by(self, *properties: str) -> "OQLQueryBuilder":
        """그룹화"""
        self.query.group_by.extend(properties)
        return self

    def aggregate(self, name: str, expression: str) -> "OQLQueryBuilder":
        """
        집계 추가

        예:
            .group_by("status")
            .aggregate("total_amount", "sum(amount)")
            .aggregate("order_count", "count(*)")
        """
        self.query.aggregations[name] = expression
        return self

    # ============== 실행 메서드 ==============

    async def execute(self) -> List["ObjectInstance"]:
        """쿼리 실행"""
        return await self.engine.execute(self.query)

    async def first(self) -> Optional["ObjectInstance"]:
        """첫 번째 결과 (없으면 None)"""
        self.query.limit = 1
        results = await self.engine.execute(self.query)
        return results[0] if results else None

    async def count(self) -> int:
        """결과 수"""
        return await self.engine.count(self.query)

    async def exists(self) -> bool:
        """결과 존재 여부"""
        return await self.count() > 0

    async def all(self) -> List["ObjectInstance"]:
        """모든 결과 (limit 무시)"""
        self.query.limit = None
        self.query.offset = None
        return await self.engine.execute(self.query)

    # ============== 유틸리티 ==============

    def to_dict(self) -> Dict[str, Any]:
        """쿼리를 딕셔너리로 변환"""
        return self.query.to_dict()

    def clone(self) -> "OQLQueryBuilder":
        """빌더 복제"""
        import copy
        new_builder = OQLQueryBuilder(self.engine, self.query.object_type)
        new_builder.query = copy.deepcopy(self.query)
        return new_builder


__all__ = [
    "FilterOperator",
    "FilterClause",
    "SortClause",
    "OQLQuery",
    "OQLQueryBuilder",
]
