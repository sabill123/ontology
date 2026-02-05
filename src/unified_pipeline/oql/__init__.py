"""
OQL - Object Query Language (v2.0)

팔란티어 스타일 개념 수준 쿼리 언어

팔란티어 검증 기준:
✅ 프롬프트에 테이블명이 없어도 동작
✅ 에이전트가 ontology action만 호출
✅ SQL이 아닌 개념 중심 쿼리

두 가지 쿼리 방식:
1. 빌더 패턴 (구조화된 쿼리)
   orders = await oql.query("Order") \\
       .filter("status", "==", "pending") \\
       .filter("risk_score", ">", 0.8) \\
       .include("customer") \\
       .execute()

2. 자연어 (LLM으로 OQL 변환)
   orders = await oql.ask("리스크가 0.8 이상인 대기 중인 주문")

핵심 클래스:
- OQLQuery: 쿼리 표현
- OQLQueryBuilder: 플루언트 빌더
- OQLEngine: 실행 엔진
"""

from .query import (
    FilterOperator,
    FilterClause,
    SortClause,
    OQLQuery,
    OQLQueryBuilder,
)
from .engine import OQLEngine

__all__ = [
    # Query
    "FilterOperator",
    "FilterClause",
    "SortClause",
    "OQLQuery",
    "OQLQueryBuilder",
    # Engine
    "OQLEngine",
]
