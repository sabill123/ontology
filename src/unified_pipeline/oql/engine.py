"""
OQL Engine (v2.1)

Object Query Language 실행 엔진

팔란티어 검증 기준:
✅ 프롬프트에 테이블명이 없어도 동작
✅ 에이전트가 개념 단위로 reasoning
✅ 자연어 → OQL 변환 지원
✅ 데이터 소스 변경이 로직을 깨지 않음 (DataAdapter)

두 가지 쿼리 방식:
1. 빌더 패턴 (구조화된 쿼리)
   orders = await oql.query("Order").filter(...).execute()

2. 자연어 (LLM으로 OQL 변환)
   orders = await oql.ask("리스크가 높은 대기 중인 주문")

아키텍처 (v2.1):
    OQLEngine → DataAdapter → [SharedContext, DB, API, ...]
                     ↓
              ObjectProxy (통합 인터페이스)

사용 예시:
    engine = OQLEngine(shared_context, llm_client=llm)

    # 빌더 방식
    orders = await engine.query("Order") \\
        .filter("status", "==", "pending") \\
        .execute()

    # 자연어 방식
    orders = await engine.ask("리스크 0.8 이상인 주문")
"""

from typing import List, Dict, Any, Optional, TYPE_CHECKING
import json
import re
import logging

from .query import (
    OQLQuery,
    OQLQueryBuilder,
    FilterClause,
    FilterOperator,
    SortClause,
)
from ..osdk.adapter import (
    DataAdapter,
    SharedContextAdapter,
    ObjectProxy,
    GlobalAdapterRegistry,
    create_adapter_for_context,
)

if TYPE_CHECKING:
    from ..shared_context import SharedContext
    from ..osdk.models import ObjectInstance, TypeRegistry

logger = logging.getLogger(__name__)


class OQLEngine:
    """
    Object Query Language 엔진 (v2.1)

    팔란티어 검증 기준:
    ✅ SQL 대신 개념 수준 쿼리 실행
    ✅ 데이터 소스 변경이 로직을 깨지 않음 (DataAdapter)
    ✅ 자연어 쿼리 지원 (LLM)

    아키텍처:
        OQLEngine → DataAdapter → [SharedContext, DB, API, ...]
                         ↓
                  ObjectProxy (통합 인터페이스)

    두 가지 쿼리 방식:
    1. 빌더 패턴 (구조화된 쿼리)
    2. 자연어 (LLM으로 OQL 변환)
    """

    def __init__(
        self,
        shared_context: Optional["SharedContext"] = None,
        type_registry: Optional["TypeRegistry"] = None,
        llm_client=None,
        adapter: Optional[DataAdapter] = None,
    ):
        self.context = shared_context
        self.registry = type_registry
        self.llm = llm_client

        # DataAdapter 설정 (v2.1)
        if adapter:
            self.adapter = adapter
        elif shared_context:
            # SharedContext용 어댑터 자동 생성
            self.adapter = create_adapter_for_context(shared_context)
        else:
            # 전역 레지스트리 사용
            self.adapter = GlobalAdapterRegistry.get()

    # ============== 빌더 API ==============

    def query(self, object_type: str) -> OQLQueryBuilder:
        """
        쿼리 빌더 시작

        예:
            orders = await oql.query("Order") \\
                .filter("status", "==", "pending") \\
                .execute()
        """
        return OQLQueryBuilder(self, object_type)

    # ============== 자연어 API ==============

    async def ask(self, question: str) -> List:
        """
        자연어 쿼리

        팔란티어 검증 기준:
        ✅ 프롬프트에 테이블명이 없어도 동작
        ✅ 에이전트가 개념 단위로 reasoning

        예:
            orders = await oql.ask("리스크가 0.8 이상인 대기 중인 주문")
            customers = await oql.ask("최근 30일간 주문이 없는 고객")
        """
        if not self.llm:
            raise RuntimeError("LLM client not configured for natural language queries")

        # 1. 스키마 컨텍스트 생성
        schema_context = self._build_schema_context()

        # 2. LLM으로 OQL 변환
        prompt = f"""
다음 자연어 질의를 OQL(Object Query Language)로 변환하세요.

=== 스키마 정보 ===
{schema_context}

=== 질의 ===
{question}

=== 출력 형식 (JSON) ===
{{
    "object_type": "대상 ObjectType명",
    "filters": [
        {{"property": "속성명", "operator": "연산자", "value": "값"}}
    ],
    "includes": ["관계명"],
    "selects": ["속성명"],
    "order_by": [{{"property": "속성명", "descending": false}}],
    "limit": null
}}

=== 연산자 목록 ===
==, !=, >, >=, <, <=, in, not_in, contains, starts_with, ends_with, is_null, is_not_null, between

=== 규칙 ===
1. object_type은 테이블명이 아닌 비즈니스 개념명 사용
2. property는 컬럼명이 아닌 속성명 사용
3. 관계는 includes로 표현 (JOIN 아님)
"""

        response = await self.llm.call(prompt)
        oql_dict = self._parse_oql_json(response)

        # 3. OQLQuery 생성
        query = self._dict_to_query(oql_dict)

        logger.debug(f"[OQLEngine] NL→OQL: {question} → {query.to_dict()}")

        # 4. 실행
        return await self.execute(query)

    # ============== 실행 ==============

    async def execute(self, query: OQLQuery) -> List:
        """OQL 쿼리 실행"""

        # 1. 대상 인스턴스 가져오기
        instances = self._get_instances_by_type(query.object_type)

        logger.debug(f"[OQLEngine] Found {len(instances)} instances of {query.object_type}")

        # 2. 필터 적용
        for filter_clause in query.filters:
            instances = self._apply_filter(instances, filter_clause)

        # 3. 중복 제거
        if query.distinct:
            instances = self._deduplicate(instances)

        # 4. 정렬
        for sort in reversed(query.order_by):
            instances = self._apply_sort(instances, sort)

        # 5. 페이징
        if query.offset:
            instances = instances[query.offset:]
        if query.limit:
            instances = instances[:query.limit]

        # 6. 관계 로딩 (eager loading)
        if query.includes:
            instances = await self._load_relations(instances, query.includes)

        # 7. 속성 선택 (projection)
        if query.selects:
            instances = self._project(instances, query.selects)

        logger.debug(f"[OQLEngine] Returning {len(instances)} results")

        return instances

    async def count(self, query: OQLQuery) -> int:
        """쿼리 결과 수"""
        instances = self._get_instances_by_type(query.object_type)
        for filter_clause in query.filters:
            instances = self._apply_filter(instances, filter_clause)
        return len(instances)

    async def execute_raw(self, oql_string: str) -> List:
        """
        OQL 문자열 직접 실행

        Behavior에서 사용:
            oql_query: "Order.filter(status == 'pending')"
        """
        query = self._parse_oql_string(oql_string)
        return await self.execute(query)

    # ============== 내부 메서드 ==============

    def _get_instances_by_type(self, object_type: str) -> List[ObjectProxy]:
        """
        타입별 인스턴스 조회 (v2.1 - DataAdapter 사용)

        팔란티어 검증:
        ✅ 데이터 소스 변경이 로직을 깨지 않음
        """
        import asyncio

        # DataAdapter를 통한 조회 (추상화)
        if self.adapter:
            try:
                # 동기 컨텍스트에서 비동기 호출
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # 이미 실행 중인 루프가 있으면 동기적으로 fallback
                    return self._get_instances_sync_fallback(object_type)
                else:
                    return loop.run_until_complete(
                        self.adapter.get_instances(object_type)
                    )
            except RuntimeError:
                # 이벤트 루프가 없는 경우
                return self._get_instances_sync_fallback(object_type)

        return []

    def _get_instances_sync_fallback(self, object_type: str) -> List[ObjectProxy]:
        """
        동기 fallback (이벤트 루프 내에서 호출 시)

        Note: 가능하면 async 메서드 사용 권장
        """
        instances = []

        if not self.context:
            return instances

        # SharedContextAdapter와 동일한 로직 (fallback)
        # unified_entities에서 조회
        if hasattr(self.context, 'unified_entities') and self.context.unified_entities:
            for idx, entity in enumerate(self.context.unified_entities):
                entity_type = ""
                if isinstance(entity, dict):
                    entity_type = entity.get('entity_type', '') or entity.get('type', '')
                else:
                    entity_type = getattr(entity, 'entity_type', '') or getattr(entity, 'type', '')

                if entity_type == object_type or object_type.lower() in entity_type.lower():
                    instances.append(self._to_proxy(entity, object_type))

        # ontology_concepts에서 조회
        if hasattr(self.context, 'ontology_concepts') and self.context.ontology_concepts:
            for concept_id, concept in self.context.ontology_concepts.items():
                concept_type = getattr(concept, 'concept_type', None) if hasattr(concept, 'concept_type') else None
                if concept_type == object_type or object_type == "OntologyConcept":
                    instances.append(self._concept_to_proxy(concept, concept_id))

        # fk_candidates에서 Relationship 조회
        if object_type == "Relationship" and hasattr(self.context, 'enhanced_fk_candidates'):
            for idx, fk in enumerate(self.context.enhanced_fk_candidates or []):
                if isinstance(fk, dict):
                    instances.append(ObjectProxy(
                        instance_id=f"rel_{idx}",
                        object_type="Relationship",
                        properties=fk,
                    ))

        return instances

    def _to_proxy(self, entity: Any, object_type: str) -> ObjectProxy:
        """Entity를 ObjectProxy로 변환"""
        if isinstance(entity, dict):
            return ObjectProxy(
                instance_id=entity.get('id', str(hash(frozenset(entity.items())))),
                object_type=object_type,
                properties=entity,
            )
        else:
            return ObjectProxy(
                instance_id=getattr(entity, 'instance_id', str(id(entity))),
                object_type=object_type,
                properties=getattr(entity, 'properties', entity.__dict__),
            )

    def _concept_to_proxy(self, concept: Any, concept_id: str) -> ObjectProxy:
        """OntologyConcept를 ObjectProxy로 변환"""
        if isinstance(concept, dict):
            return ObjectProxy(
                instance_id=concept_id,
                object_type="OntologyConcept",
                properties=concept,
            )
        else:
            props = {
                'concept_id': concept_id,
                'name': getattr(concept, 'name', ''),
                'concept_type': getattr(concept, 'concept_type', ''),
                'description': getattr(concept, 'description', ''),
                'properties': getattr(concept, 'properties', []),
                'relationships': getattr(concept, 'relationships', []),
                'confidence': getattr(concept, 'confidence', 0.0),
            }
            return ObjectProxy(
                instance_id=concept_id,
                object_type="OntologyConcept",
                properties=props,
            )

    def _apply_filter(
        self,
        instances: List[ObjectProxy],
        filter_clause: FilterClause,
    ) -> List[ObjectProxy]:
        """필터 적용"""
        result = []

        for instance in instances:
            value = self._get_nested_property(instance, filter_clause.property)

            if self._evaluate_filter(value, filter_clause):
                result.append(instance)

        return result

    def _evaluate_filter(self, value: Any, filter_clause: FilterClause) -> bool:
        """필터 조건 평가"""
        op = filter_clause.operator
        filter_value = filter_clause.value

        try:
            if op == FilterOperator.EQ:
                return value == filter_value
            elif op == FilterOperator.NE:
                return value != filter_value
            elif op == FilterOperator.GT:
                return value is not None and value > filter_value
            elif op == FilterOperator.GTE:
                return value is not None and value >= filter_value
            elif op == FilterOperator.LT:
                return value is not None and value < filter_value
            elif op == FilterOperator.LTE:
                return value is not None and value <= filter_value
            elif op == FilterOperator.IN:
                return value in filter_value
            elif op == FilterOperator.NOT_IN:
                return value not in filter_value
            elif op == FilterOperator.CONTAINS:
                return filter_value in str(value) if value else False
            elif op == FilterOperator.STARTS_WITH:
                return str(value).startswith(str(filter_value)) if value else False
            elif op == FilterOperator.ENDS_WITH:
                return str(value).endswith(str(filter_value)) if value else False
            elif op == FilterOperator.MATCHES:
                return bool(re.match(str(filter_value), str(value))) if value else False
            elif op == FilterOperator.IS_NULL:
                return value is None
            elif op == FilterOperator.IS_NOT_NULL:
                return value is not None
            elif op == FilterOperator.BETWEEN:
                return filter_value <= value <= filter_clause.value2
        except Exception:
            pass

        return False

    def _get_nested_property(self, instance: ObjectProxy, property_path: str) -> Any:
        """
        점 표기법으로 중첩 속성 조회

        예: "customer.name" → instance.properties["customer"]["name"]
        """
        parts = property_path.split(".")
        value = instance.properties

        for part in parts:
            if isinstance(value, dict):
                value = value.get(part)
            elif hasattr(value, 'get'):
                value = value.get(part)
            elif hasattr(value, part):
                value = getattr(value, part)
            else:
                return None

            if value is None:
                return None

        return value

    def _apply_sort(
        self,
        instances: List[ObjectProxy],
        sort: SortClause,
    ) -> List[ObjectProxy]:
        """정렬 적용"""
        def sort_key(x):
            val = self._get_nested_property(x, sort.property)
            # None 처리
            if val is None:
                return (1, "")  # None은 뒤로
            return (0, val)

        return sorted(
            instances,
            key=sort_key,
            reverse=sort.descending,
        )

    def _deduplicate(self, instances: List[ObjectProxy]) -> List[ObjectProxy]:
        """중복 제거"""
        seen = set()
        result = []
        for inst in instances:
            if inst.instance_id not in seen:
                seen.add(inst.instance_id)
                result.append(inst)
        return result

    async def _load_relations(
        self,
        instances: List[ObjectProxy],
        relations: List[str],
    ) -> List[ObjectProxy]:
        """
        관계 로딩 (eager loading)

        관계명으로 연결된 객체를 미리 로드
        """
        for instance in instances:
            for relation in relations:
                # 관계 데이터 로드
                related = await self._fetch_related(instance, relation)
                instance.properties[f"_loaded_{relation}"] = related

        return instances

    async def _fetch_related(
        self,
        instance: ObjectProxy,
        relation: str,
    ) -> List[ObjectProxy]:
        """
        관계 데이터 조회

        LinkType 기반으로 관계 조회
        """
        # 간단한 구현: FK 스타일 관계 조회
        fk_value = instance.properties.get(f"{relation}_id")
        if fk_value and self.context:
            # 해당 ID를 가진 객체 찾기
            related_type = relation.title()  # 예: "customer" → "Customer"
            all_instances = self._get_instances_by_type(related_type)
            return [i for i in all_instances if i.properties.get("id") == fk_value]

        return []

    def _project(
        self,
        instances: List[ObjectProxy],
        selects: List[str],
    ) -> List[Dict[str, Any]]:
        """
        속성 선택 (projection)

        선택된 속성만 포함하는 딕셔너리 반환
        """
        result = []
        for instance in instances:
            projected = {"_id": instance.instance_id, "_type": instance.object_type}
            for prop in selects:
                projected[prop] = self._get_nested_property(instance, prop)
            result.append(projected)
        return result

    def _build_schema_context(self) -> str:
        """
        NL→OQL용 스키마 컨텍스트 생성

        LLM이 쿼리를 변환할 수 있도록 스키마 정보 제공
        """
        schema_lines = []

        # ObjectType 정보
        if self.registry:
            for type_id, obj_type in self.registry._object_types.items():
                props = list(obj_type.properties.keys()) if obj_type.properties else []
                schema_lines.append(f"ObjectType: {type_id}")
                schema_lines.append(f"  Properties: {', '.join(props[:10])}")  # 최대 10개

        # Context 기반 정보
        if self.context:
            # Entity 타입
            if hasattr(self.context, 'unified_entities') and self.context.unified_entities:
                entity_types = set()
                for e in self.context.unified_entities[:50]:  # 최대 50개 샘플
                    et = getattr(e, 'entity_type', None) or (e.get('entity_type', '') if isinstance(e, dict) else '')
                    if et:
                        entity_types.add(et)
                if entity_types:
                    schema_lines.append(f"Entity Types: {', '.join(entity_types)}")

            # Concept 정보
            if hasattr(self.context, 'ontology_concepts') and self.context.ontology_concepts:
                concept_names = list(self.context.ontology_concepts.keys())[:10]
                schema_lines.append(f"Concepts: {', '.join(concept_names)}")

        if not schema_lines:
            schema_lines.append("(스키마 정보 없음)")

        return "\n".join(schema_lines)

    def _parse_oql_json(self, response: str) -> Dict:
        """LLM 응답에서 OQL JSON 파싱"""
        # JSON 블록 추출
        json_match = re.search(r'\{[\s\S]*\}', response)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass
        return {"object_type": "", "filters": []}

    def _dict_to_query(self, d: Dict) -> OQLQuery:
        """딕셔너리 → OQLQuery 변환"""
        query = OQLQuery(object_type=d.get("object_type", ""))

        for f in d.get("filters", []):
            try:
                query.filters.append(FilterClause(
                    property=f["property"],
                    operator=FilterOperator(f["operator"]),
                    value=f.get("value"),
                    value2=f.get("value2"),
                ))
            except (KeyError, ValueError):
                continue

        query.includes = d.get("includes", [])
        query.selects = d.get("selects", [])

        for s in d.get("order_by", []):
            query.order_by.append(SortClause(
                property=s.get("property", ""),
                descending=s.get("descending", False),
            ))

        query.limit = d.get("limit")
        query.offset = d.get("offset")

        return query

    def _parse_oql_string(self, oql_string: str) -> OQLQuery:
        """
        간단한 OQL 문자열 파싱

        형식: "ObjectType.filter(prop op value).filter(...)"
        """
        # 간단한 파서 (추후 확장)
        parts = oql_string.split(".")
        if not parts:
            return OQLQuery(object_type="")

        object_type = parts[0]
        query = OQLQuery(object_type=object_type)

        for part in parts[1:]:
            if part.startswith("filter("):
                # filter(status == 'pending') 파싱
                inner = part[7:-1]  # "status == 'pending'"
                # 간단한 파싱
                for op_str in ["==", "!=", ">=", "<=", ">", "<"]:
                    if op_str in inner:
                        prop, val = inner.split(op_str, 1)
                        query.filters.append(FilterClause(
                            property=prop.strip(),
                            operator=FilterOperator(op_str),
                            value=val.strip().strip("'\""),
                        ))
                        break

        return query


__all__ = ["OQLEngine"]
