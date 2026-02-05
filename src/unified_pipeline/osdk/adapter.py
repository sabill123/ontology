"""
DataAdapter - 데이터 소스 추상화 레이어

팔란티어 검증 기준 #2:
✅ 데이터 소스가 바뀌어도 로직이 안 깨짐

문제점 (기존):
- OQLEngine이 SharedContext의 특정 필드에 직접 접근
- unified_entities, ontology_concepts 등 하드코딩

해결책:
- DataAdapter가 모든 데이터 소스를 추상화
- ObjectType 기준으로 통합 인터페이스 제공
- 데이터 소스 변경 시 Adapter만 수정

아키텍처:
    OQLEngine → DataAdapter → [SharedContext, DB, API, File, ...]
                     ↓
              ObjectProxy (통합 인터페이스)
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Callable, Type, TypeVar
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class DataSourceType(str, Enum):
    """데이터 소스 유형"""
    SHARED_CONTEXT = "shared_context"   # SharedContext (메모리)
    DATABASE = "database"               # DB (SQL/NoSQL)
    API = "api"                         # 외부 API
    FILE = "file"                       # 파일 시스템
    CACHE = "cache"                     # 캐시
    CUSTOM = "custom"                   # 커스텀


@dataclass
class ObjectProxy:
    """
    ObjectInstance 프록시

    모든 데이터 소스의 데이터를 통합 인터페이스로 접근
    """
    instance_id: str
    object_type: str
    properties: Dict[str, Any]
    source_type: DataSourceType = DataSourceType.SHARED_CONTEXT
    source_id: Optional[str] = None  # 원본 소스 식별자

    def get(self, key: str, default: Any = None) -> Any:
        """속성 조회"""
        return self.properties.get(key, default)

    def set(self, key: str, value: Any) -> None:
        """속성 설정"""
        self.properties[key] = value

    def has(self, key: str) -> bool:
        """속성 존재 여부"""
        return key in self.properties

    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리 변환"""
        return {
            "_id": self.instance_id,
            "_type": self.object_type,
            "_source": self.source_type.value,
            **self.properties
        }

    def __repr__(self) -> str:
        return f"ObjectProxy({self.object_type}:{self.instance_id})"


@dataclass
class TypeMapping:
    """
    ObjectType ↔ 데이터 소스 매핑

    어떤 ObjectType이 어떤 데이터 소스에서 오는지 정의
    """
    object_type: str
    source_type: DataSourceType
    source_config: Dict[str, Any] = field(default_factory=dict)
    # 예: {"collection": "orders", "id_field": "order_id"}

    # 필드 매핑 (데이터 소스 필드 → ObjectType 속성)
    field_mapping: Dict[str, str] = field(default_factory=dict)
    # 예: {"orderId": "id", "orderStatus": "status"}

    # 역필드 매핑 (자동 생성)
    reverse_mapping: Dict[str, str] = field(default_factory=dict)


class DataAdapter(ABC):
    """
    데이터 어댑터 추상 클래스

    모든 데이터 소스는 이 인터페이스를 구현
    """

    @property
    @abstractmethod
    def source_type(self) -> DataSourceType:
        """데이터 소스 유형"""
        pass

    @abstractmethod
    async def get_instances(
        self,
        object_type: str,
        filters: Optional[Dict[str, Any]] = None,
        limit: Optional[int] = None,
        offset: int = 0
    ) -> List[ObjectProxy]:
        """
        인스턴스 조회

        Args:
            object_type: ObjectType 이름
            filters: 필터 조건 {"property": value}
            limit: 최대 개수
            offset: 시작 위치

        Returns:
            ObjectProxy 리스트
        """
        pass

    @abstractmethod
    async def get_instance_by_id(
        self,
        object_type: str,
        instance_id: str
    ) -> Optional[ObjectProxy]:
        """ID로 인스턴스 조회"""
        pass

    @abstractmethod
    async def count(
        self,
        object_type: str,
        filters: Optional[Dict[str, Any]] = None
    ) -> int:
        """인스턴스 수 조회"""
        pass

    async def save_instance(
        self,
        proxy: ObjectProxy
    ) -> ObjectProxy:
        """인스턴스 저장 (선택적)"""
        raise NotImplementedError("This adapter is read-only")

    async def delete_instance(
        self,
        object_type: str,
        instance_id: str
    ) -> bool:
        """인스턴스 삭제 (선택적)"""
        raise NotImplementedError("This adapter is read-only")


class SharedContextAdapter(DataAdapter):
    """
    SharedContext 어댑터

    SharedContext의 다양한 필드를 ObjectType으로 매핑
    """

    def __init__(self, context: Any):
        self.context = context
        self._type_mappings: Dict[str, TypeMapping] = {}
        self._setup_default_mappings()

    @property
    def source_type(self) -> DataSourceType:
        return DataSourceType.SHARED_CONTEXT

    def _setup_default_mappings(self):
        """기본 타입 매핑 설정"""
        # unified_entities → Entity
        self.register_mapping(TypeMapping(
            object_type="Entity",
            source_type=DataSourceType.SHARED_CONTEXT,
            source_config={"field": "unified_entities"},
            field_mapping={"entity_type": "type", "entity_id": "id"}
        ))

        # ontology_concepts → OntologyConcept
        self.register_mapping(TypeMapping(
            object_type="OntologyConcept",
            source_type=DataSourceType.SHARED_CONTEXT,
            source_config={"field": "ontology_concepts", "is_dict": True},
            field_mapping={}
        ))

        # enhanced_fk_candidates → Relationship
        self.register_mapping(TypeMapping(
            object_type="Relationship",
            source_type=DataSourceType.SHARED_CONTEXT,
            source_config={"field": "enhanced_fk_candidates"},
            field_mapping={}
        ))

        # domain_contexts → Domain
        self.register_mapping(TypeMapping(
            object_type="Domain",
            source_type=DataSourceType.SHARED_CONTEXT,
            source_config={"field": "domain_contexts", "is_dict": True},
            field_mapping={}
        ))

    def register_mapping(self, mapping: TypeMapping):
        """타입 매핑 등록"""
        # 역매핑 생성
        mapping.reverse_mapping = {v: k for k, v in mapping.field_mapping.items()}
        self._type_mappings[mapping.object_type] = mapping

    def get_supported_types(self) -> List[str]:
        """지원하는 ObjectType 목록"""
        return list(self._type_mappings.keys())

    async def get_instances(
        self,
        object_type: str,
        filters: Optional[Dict[str, Any]] = None,
        limit: Optional[int] = None,
        offset: int = 0
    ) -> List[ObjectProxy]:
        """인스턴스 조회"""
        mapping = self._type_mappings.get(object_type)

        # 매핑이 없으면 동적 탐색
        if not mapping:
            return self._dynamic_search(object_type, filters, limit, offset)

        # 소스 데이터 가져오기
        source_field = mapping.source_config.get("field", "")
        is_dict = mapping.source_config.get("is_dict", False)

        raw_data = getattr(self.context, source_field, None)
        if raw_data is None:
            return []

        # ObjectProxy로 변환
        proxies = []
        items = raw_data.items() if is_dict else enumerate(raw_data)

        for key, item in items:
            proxy = self._to_proxy(item, object_type, str(key), mapping)
            if proxy:
                # 필터 적용
                if filters and not self._matches_filters(proxy, filters):
                    continue
                proxies.append(proxy)

        # 페이징
        if offset:
            proxies = proxies[offset:]
        if limit:
            proxies = proxies[:limit]

        return proxies

    async def get_instance_by_id(
        self,
        object_type: str,
        instance_id: str
    ) -> Optional[ObjectProxy]:
        """ID로 인스턴스 조회"""
        instances = await self.get_instances(
            object_type,
            filters={"_id": instance_id},
            limit=1
        )
        return instances[0] if instances else None

    async def count(
        self,
        object_type: str,
        filters: Optional[Dict[str, Any]] = None
    ) -> int:
        """인스턴스 수"""
        instances = await self.get_instances(object_type, filters)
        return len(instances)

    def _to_proxy(
        self,
        item: Any,
        object_type: str,
        default_id: str,
        mapping: TypeMapping
    ) -> Optional[ObjectProxy]:
        """아이템을 ObjectProxy로 변환"""
        if item is None:
            return None

        # 딕셔너리 또는 객체에서 속성 추출
        if isinstance(item, dict):
            props = dict(item)
            instance_id = props.get("id") or props.get("_id") or default_id
        else:
            props = {}
            for attr in dir(item):
                if not attr.startswith("_"):
                    try:
                        val = getattr(item, attr)
                        if not callable(val):
                            props[attr] = val
                    except Exception:
                        pass
            instance_id = getattr(item, "id", None) or getattr(item, "instance_id", None) or default_id

        # 필드 매핑 적용
        mapped_props = {}
        for src_key, value in props.items():
            dest_key = mapping.field_mapping.get(src_key, src_key)
            mapped_props[dest_key] = value

        return ObjectProxy(
            instance_id=str(instance_id),
            object_type=object_type,
            properties=mapped_props,
            source_type=DataSourceType.SHARED_CONTEXT
        )

    def _dynamic_search(
        self,
        object_type: str,
        filters: Optional[Dict[str, Any]],
        limit: Optional[int],
        offset: int
    ) -> List[ObjectProxy]:
        """매핑 없는 타입의 동적 탐색"""
        proxies = []

        # unified_entities에서 entity_type으로 검색
        if hasattr(self.context, 'unified_entities') and self.context.unified_entities:
            for idx, entity in enumerate(self.context.unified_entities):
                entity_type = ""
                if isinstance(entity, dict):
                    entity_type = entity.get('entity_type', '') or entity.get('type', '')
                else:
                    entity_type = getattr(entity, 'entity_type', '') or getattr(entity, 'type', '')

                if object_type.lower() in entity_type.lower() or entity_type.lower() in object_type.lower():
                    proxy = ObjectProxy(
                        instance_id=str(idx),
                        object_type=object_type,
                        properties=entity if isinstance(entity, dict) else entity.__dict__,
                        source_type=DataSourceType.SHARED_CONTEXT
                    )
                    if not filters or self._matches_filters(proxy, filters):
                        proxies.append(proxy)

        # 페이징
        if offset:
            proxies = proxies[offset:]
        if limit:
            proxies = proxies[:limit]

        return proxies

    def _matches_filters(
        self,
        proxy: ObjectProxy,
        filters: Dict[str, Any]
    ) -> bool:
        """필터 조건 매칭"""
        for key, value in filters.items():
            if key == "_id":
                if proxy.instance_id != str(value):
                    return False
            else:
                prop_value = proxy.get(key)
                if prop_value != value:
                    return False
        return True


class CompositeAdapter(DataAdapter):
    """
    복합 어댑터

    여러 데이터 소스를 통합하여 하나의 인터페이스로 제공
    """

    def __init__(self):
        self._adapters: Dict[str, DataAdapter] = {}  # object_type -> adapter
        self._default_adapter: Optional[DataAdapter] = None

    @property
    def source_type(self) -> DataSourceType:
        return DataSourceType.CUSTOM

    def register_adapter(
        self,
        object_type: str,
        adapter: DataAdapter
    ):
        """특정 ObjectType에 대한 어댑터 등록"""
        self._adapters[object_type] = adapter

    def set_default_adapter(self, adapter: DataAdapter):
        """기본 어댑터 설정"""
        self._default_adapter = adapter

    def get_adapter(self, object_type: str) -> Optional[DataAdapter]:
        """ObjectType에 대한 어댑터 반환"""
        return self._adapters.get(object_type) or self._default_adapter

    async def get_instances(
        self,
        object_type: str,
        filters: Optional[Dict[str, Any]] = None,
        limit: Optional[int] = None,
        offset: int = 0
    ) -> List[ObjectProxy]:
        adapter = self.get_adapter(object_type)
        if not adapter:
            logger.warning(f"No adapter found for {object_type}")
            return []
        return await adapter.get_instances(object_type, filters, limit, offset)

    async def get_instance_by_id(
        self,
        object_type: str,
        instance_id: str
    ) -> Optional[ObjectProxy]:
        adapter = self.get_adapter(object_type)
        if not adapter:
            return None
        return await adapter.get_instance_by_id(object_type, instance_id)

    async def count(
        self,
        object_type: str,
        filters: Optional[Dict[str, Any]] = None
    ) -> int:
        adapter = self.get_adapter(object_type)
        if not adapter:
            return 0
        return await adapter.count(object_type, filters)


# 전역 어댑터 레지스트리
class GlobalAdapterRegistry:
    """전역 어댑터 레지스트리"""
    _instance: Optional[CompositeAdapter] = None

    @classmethod
    def get(cls) -> CompositeAdapter:
        if cls._instance is None:
            cls._instance = CompositeAdapter()
        return cls._instance

    @classmethod
    def register(cls, object_type: str, adapter: DataAdapter):
        cls.get().register_adapter(object_type, adapter)

    @classmethod
    def set_default(cls, adapter: DataAdapter):
        cls.get().set_default_adapter(adapter)

    @classmethod
    def reset(cls):
        cls._instance = None


class SupabaseAdapter(DataAdapter):
    """
    Supabase Postgres 어댑터

    Supabase DB에서 ObjectType별 데이터 조회

    설정:
        adapter = SupabaseAdapter(
            supabase_url="https://xxx.supabase.co",
            supabase_key="your-anon-key"
        )

    또는 환경변수:
        SUPABASE_URL, SUPABASE_KEY
    """

    # ObjectType → Supabase 테이블 매핑
    DEFAULT_TYPE_MAPPINGS = {
        "PipelineRun": TypeMapping(
            object_type="PipelineRun",
            source_type=DataSourceType.DATABASE,
            source_config={"schema": "public", "table": "pipeline_runs"},
            field_mapping={"scenario_name": "name", "domain_context": "domain"}
        ),
        "SourceTable": TypeMapping(
            object_type="SourceTable",
            source_type=DataSourceType.DATABASE,
            source_config={"schema": "discovery", "table": "source_tables"},
            field_mapping={}
        ),
        "UnifiedEntity": TypeMapping(
            object_type="UnifiedEntity",
            source_type=DataSourceType.DATABASE,
            source_config={"schema": "discovery", "table": "unified_entities"},
            field_mapping={"entity_id": "id", "entity_type": "type"}
        ),
        "OntologyConcept": TypeMapping(
            object_type="OntologyConcept",
            source_type=DataSourceType.DATABASE,
            source_config={"schema": "refinement", "table": "ontology_concepts"},
            field_mapping={"concept_id": "id", "concept_type": "type"}
        ),
        "Relationship": TypeMapping(
            object_type="Relationship",
            source_type=DataSourceType.DATABASE,
            source_config={"schema": "refinement", "table": "concept_relationships"},
            field_mapping={}
        ),
        "GovernanceDecision": TypeMapping(
            object_type="GovernanceDecision",
            source_type=DataSourceType.DATABASE,
            source_config={"schema": "governance", "table": "governance_decisions"},
            field_mapping={}
        ),
        "ActionBacklog": TypeMapping(
            object_type="ActionBacklog",
            source_type=DataSourceType.DATABASE,
            source_config={"schema": "governance", "table": "action_backlog"},
            field_mapping={}
        ),
        "EvidenceBlock": TypeMapping(
            object_type="EvidenceBlock",
            source_type=DataSourceType.DATABASE,
            source_config={"schema": "evidence", "table": "evidence_blocks"},
            field_mapping={}
        ),
    }

    def __init__(
        self,
        supabase_url: Optional[str] = None,
        supabase_key: Optional[str] = None,
        supabase_client: Any = None,
    ):
        import os

        self.supabase_url = supabase_url or os.getenv("SUPABASE_URL")
        self.supabase_key = supabase_key or os.getenv("SUPABASE_KEY")

        # 클라이언트 초기화
        self._client = supabase_client
        self._type_mappings = dict(self.DEFAULT_TYPE_MAPPINGS)

    @property
    def source_type(self) -> DataSourceType:
        return DataSourceType.DATABASE

    @property
    def type_mappings(self) -> Dict[str, TypeMapping]:
        """등록된 타입 매핑 조회"""
        return self._type_mappings

    @property
    def client(self):
        """Lazy Supabase client initialization"""
        if self._client is None:
            try:
                from supabase import create_client
                self._client = create_client(self.supabase_url, self.supabase_key)
            except ImportError:
                raise RuntimeError("supabase-py not installed. Run: pip install supabase")
            except Exception as e:
                raise RuntimeError(f"Failed to create Supabase client: {e}")
        return self._client

    def register_mapping(self, mapping: TypeMapping):
        """타입 매핑 등록"""
        mapping.reverse_mapping = {v: k for k, v in mapping.field_mapping.items()}
        self._type_mappings[mapping.object_type] = mapping

    async def get_instances(
        self,
        object_type: str,
        filters: Optional[Dict[str, Any]] = None,
        limit: Optional[int] = None,
        offset: int = 0
    ) -> List[ObjectProxy]:
        """Supabase에서 인스턴스 조회"""
        mapping = self._type_mappings.get(object_type)
        if not mapping:
            logger.warning(f"No mapping for {object_type}, trying table name directly")
            # 테이블명 직접 시도
            table_name = object_type.lower() + "s"  # 복수형
            schema = "public"
        else:
            schema = mapping.source_config.get("schema", "public")
            table_name = mapping.source_config["table"]

        try:
            # 쿼리 빌드
            query = self.client.schema(schema).table(table_name).select("*")

            # 필터 적용
            if filters:
                for key, value in filters.items():
                    # 매핑된 필드명으로 변환
                    db_key = key
                    if mapping and key in mapping.reverse_mapping:
                        db_key = mapping.reverse_mapping[key]
                    query = query.eq(db_key, value)

            # 페이징
            if limit:
                query = query.limit(limit)
            if offset:
                query = query.offset(offset)

            # 실행
            result = query.execute()

            # ObjectProxy로 변환
            proxies = []
            for row in result.data:
                proxy = self._row_to_proxy(row, object_type, mapping)
                proxies.append(proxy)

            return proxies

        except Exception as e:
            logger.error(f"Supabase query failed: {e}")
            return []

    async def get_instance_by_id(
        self,
        object_type: str,
        instance_id: str
    ) -> Optional[ObjectProxy]:
        """ID로 인스턴스 조회"""
        instances = await self.get_instances(
            object_type,
            filters={"id": instance_id},
            limit=1
        )
        return instances[0] if instances else None

    async def query(
        self,
        object_type: str,
        filters: Optional[Dict[str, Any]] = None,
        limit: Optional[int] = None,
        offset: int = 0
    ) -> List[ObjectProxy]:
        """쿼리 메서드 (get_instances 별칭)"""
        return await self.get_instances(object_type, filters, limit, offset)

    async def count(
        self,
        object_type: str,
        filters: Optional[Dict[str, Any]] = None
    ) -> int:
        """인스턴스 수 조회"""
        mapping = self._type_mappings.get(object_type)
        if not mapping:
            return 0

        schema = mapping.source_config.get("schema", "public")
        table_name = mapping.source_config["table"]

        try:
            query = self.client.schema(schema).table(table_name).select("*", count="exact")

            if filters:
                for key, value in filters.items():
                    db_key = key
                    if key in mapping.reverse_mapping:
                        db_key = mapping.reverse_mapping[key]
                    query = query.eq(db_key, value)

            result = query.execute()
            return result.count or 0

        except Exception as e:
            logger.error(f"Supabase count failed: {e}")
            return 0

    async def save_instance(self, proxy: ObjectProxy) -> ObjectProxy:
        """인스턴스 저장 (upsert)"""
        mapping = self._type_mappings.get(proxy.object_type)
        if not mapping:
            raise ValueError(f"No mapping for {proxy.object_type}")

        schema = mapping.source_config.get("schema", "public")
        table_name = mapping.source_config["table"]

        # 필드 매핑 역변환
        db_data = {}
        for key, value in proxy.properties.items():
            db_key = mapping.reverse_mapping.get(key, key)
            db_data[db_key] = value

        try:
            result = self.client.schema(schema).table(table_name).upsert(db_data).execute()
            if result.data:
                return self._row_to_proxy(result.data[0], proxy.object_type, mapping)
            return proxy

        except Exception as e:
            logger.error(f"Supabase save failed: {e}")
            raise

    def _row_to_proxy(
        self,
        row: Dict[str, Any],
        object_type: str,
        mapping: Optional[TypeMapping]
    ) -> ObjectProxy:
        """DB 행을 ObjectProxy로 변환"""
        instance_id = str(row.get("id", ""))

        # 필드 매핑 적용
        props = dict(row)
        if mapping and mapping.field_mapping:
            mapped_props = {}
            for db_key, value in row.items():
                obj_key = mapping.field_mapping.get(db_key, db_key)
                mapped_props[obj_key] = value
            props = mapped_props

        return ObjectProxy(
            instance_id=instance_id,
            object_type=object_type,
            properties=props,
            source_type=DataSourceType.DATABASE,
            source_id=f"supabase:{mapping.source_config.get('schema', 'public')}.{mapping.source_config['table']}" if mapping else None
        )


def create_adapter_for_context(context: Any) -> SharedContextAdapter:
    """
    SharedContext용 어댑터 생성 및 등록

    사용 예:
        adapter = create_adapter_for_context(shared_context)
        GlobalAdapterRegistry.set_default(adapter)
    """
    adapter = SharedContextAdapter(context)
    GlobalAdapterRegistry.set_default(adapter)
    return adapter


def create_supabase_adapter(
    supabase_url: Optional[str] = None,
    supabase_key: Optional[str] = None
) -> SupabaseAdapter:
    """
    Supabase 어댑터 생성 및 등록

    사용 예:
        adapter = create_supabase_adapter()  # 환경변수 사용
        GlobalAdapterRegistry.set_default(adapter)
    """
    adapter = SupabaseAdapter(
        supabase_url=supabase_url,
        supabase_key=supabase_key
    )
    GlobalAdapterRegistry.set_default(adapter)
    return adapter


def create_hybrid_adapter(
    context: Any,
    supabase_url: Optional[str] = None,
    supabase_key: Optional[str] = None
) -> CompositeAdapter:
    """
    하이브리드 어댑터 생성 (SharedContext + Supabase)

    SharedContext 우선, 없으면 Supabase 조회

    사용 예:
        adapter = create_hybrid_adapter(context)
        GlobalAdapterRegistry.set_default(adapter)
    """
    composite = CompositeAdapter()

    # SharedContext 어댑터 (기본)
    context_adapter = SharedContextAdapter(context)
    composite.set_default_adapter(context_adapter)

    # Supabase 어댑터 (DB 타입용)
    try:
        supabase_adapter = SupabaseAdapter(
            supabase_url=supabase_url,
            supabase_key=supabase_key
        )
        # DB 전용 타입 등록
        for object_type in SupabaseAdapter.DEFAULT_TYPE_MAPPINGS:
            composite.register_adapter(object_type, supabase_adapter)
    except Exception as e:
        logger.warning(f"Supabase adapter not available: {e}")

    GlobalAdapterRegistry._instance = composite
    return composite


__all__ = [
    "DataSourceType",
    "ObjectProxy",
    "TypeMapping",
    "DataAdapter",
    "SharedContextAdapter",
    "SupabaseAdapter",
    "CompositeAdapter",
    "GlobalAdapterRegistry",
    "create_adapter_for_context",
    "create_supabase_adapter",
    "create_hybrid_adapter",
]
