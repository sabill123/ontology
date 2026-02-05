"""
OSDK Client (v17.0)

Ontology SDK 통합 클라이언트:
- SharedContext와 동기화
- ObjectType/LinkType CRUD
- 타입 안전 쿼리
- Writeback 지원

사용 예시:
    client = OSDKClient(context)

    # 타입 조회
    customer_type = client.get_object_type("customer")

    # 쿼리 빌더
    customers = (
        client.query("customer")
        .filter({"status": "active"})
        .limit(10)
        .execute()
    )

    # 인스턴스 생성
    new_customer = client.create("customer", {
        "name": "John Doe",
        "email": "john@example.com"
    })
"""

import logging
import sqlite3
from typing import Dict, List, Any, Optional, TYPE_CHECKING
from datetime import datetime

from .models import (
    ObjectType,
    LinkType,
    Property,
    PropertyType,
    Cardinality,
    ObjectInstance,
    TypeRegistry,
)
from .query_builder import QueryBuilder, QueryResult

if TYPE_CHECKING:
    from ..shared_context import SharedContext

logger = logging.getLogger(__name__)


class OSDKClient:
    """
    OSDK 통합 클라이언트

    Palantir Ontology SDK 스타일의 타입 안전 API 제공
    """

    def __init__(
        self,
        context: Optional["SharedContext"] = None,
        auto_sync: bool = True,
    ):
        """
        Args:
            context: SharedContext
            auto_sync: SharedContext에서 자동 동기화 여부
        """
        self.context = context

        # Type Registry
        self.registry = TypeRegistry()

        # SQLite 연결 (쿼리 실행용)
        self._conn: Optional[sqlite3.Connection] = None
        self._tables_loaded: bool = False

        # 인스턴스 캐시
        self._instance_cache: Dict[str, Dict[str, ObjectInstance]] = {}

        if auto_sync and context:
            self.sync_from_context()

        logger.info("OSDKClient initialized")

    def sync_from_context(self) -> None:
        """
        SharedContext에서 온톨로지 정보 동기화

        - 테이블 → ObjectType
        - FK → LinkType
        - 엔티티 → ObjectType 메타데이터
        """
        if not self.context:
            logger.warning("No context to sync from")
            return

        logger.info("Syncing OSDK from SharedContext...")

        # 1. 테이블에서 ObjectType 생성
        for table_name, table_info in self.context.tables.items():
            # TableInfo dataclass 또는 dict에서 columns 추출
            if hasattr(table_info, 'columns'):
                columns = table_info.columns or []
                primary_keys = getattr(table_info, 'primary_keys', []) or []
            elif isinstance(table_info, dict):
                columns = table_info.get("columns", [])
                primary_keys = table_info.get("primary_keys", []) or []
            else:
                columns = []
                primary_keys = []

            # columns가 dict인 경우 (키가 컬럼명) → List[Dict] 형식으로 변환
            if isinstance(columns, dict):
                columns = [
                    {"name": col_name, **col_info}
                    for col_name, col_info in columns.items()
                    if isinstance(col_info, dict)
                ]

            obj_type = ObjectType.from_table_info(
                table_name=table_name,
                columns=columns,
                primary_keys=primary_keys,
            )

            # 엔티티 정보 연결
            for entity in self.context.unified_entities:
                if table_name in entity.source_tables:
                    obj_type.display_name = entity.canonical_name
                    obj_type.entity_type = entity.entity_type
                    break

            self.registry.register_type(obj_type)

        # 2. FK에서 LinkType 생성
        for fk in self.context.enhanced_fk_candidates:
            if isinstance(fk, dict):
                from_table = fk.get("from_table", "")
                from_column = fk.get("from_column", "")
                to_table = fk.get("to_table", "")
                to_column = fk.get("to_column", "")
                confidence = fk.get("confidence", 0.5)
            else:
                from_table = getattr(fk, "from_table", "")
                from_column = getattr(fk, "from_column", "")
                to_table = getattr(fk, "to_table", "")
                to_column = getattr(fk, "to_column", "")
                confidence = getattr(fk, "confidence", 0.5)

            # 낮은 신뢰도 FK 제외
            if confidence < 0.3:
                continue

            link_id = f"{from_table}_{from_column}_to_{to_table}"
            link = LinkType(
                link_id=link_id,
                name=f"{from_table}_to_{to_table}",
                display_name=f"{from_table} → {to_table}",
                from_type=from_table.lower(),
                to_type=to_table.lower(),
                from_property=from_column,
                to_property=to_column,
                cardinality=Cardinality.MANY_TO_ONE,
                source_fk=fk if isinstance(fk, dict) else fk.__dict__,
            )

            self.registry.register_link(link)

        # 3. SharedContext의 OSDK 필드 업데이트
        self.context.osdk_object_types = {
            t.type_id: t.to_dict() for t in self.registry.get_all_types()
        }
        self.context.osdk_link_types = {
            l.link_id: l.to_dict() for l in self.registry.get_all_links()
        }

        logger.info(
            f"OSDK synced: {len(self.registry.types)} types, "
            f"{len(self.registry.links)} links"
        )

    def sync_from_shared_context(self) -> None:
        """sync_from_context 별칭 (호환성)"""
        self.sync_from_context()

    # ==================== Type Management ====================

    def get_object_type(self, type_id: str) -> Optional[ObjectType]:
        """ObjectType 조회"""
        return self.registry.get_type(type_id.lower())

    def get_link_type(self, link_id: str) -> Optional[LinkType]:
        """LinkType 조회"""
        return self.registry.get_link(link_id)

    def list_object_types(self) -> List[ObjectType]:
        """모든 ObjectType 목록"""
        return self.registry.get_all_types()

    def list_link_types(self) -> List[LinkType]:
        """모든 LinkType 목록"""
        return self.registry.get_all_links()

    def register_object_type(self, obj_type: ObjectType) -> None:
        """ObjectType 등록"""
        self.registry.register_type(obj_type)

        if self.context:
            self.context.osdk_object_types[obj_type.type_id] = obj_type.to_dict()

    def register_link_type(self, link: LinkType) -> None:
        """LinkType 등록"""
        self.registry.register_link(link)

        if self.context:
            self.context.osdk_link_types[link.link_id] = link.to_dict()

    # ==================== Query API ====================

    def query(self, type_id: str) -> QueryBuilder:
        """
        쿼리 빌더 생성

        Args:
            type_id: ObjectType ID

        Returns:
            QueryBuilder 인스턴스
        """
        return QueryBuilder(self, type_id.lower())

    def get_by_id(
        self,
        type_id: str,
        instance_id: Any
    ) -> Optional[ObjectInstance]:
        """
        ID로 인스턴스 조회

        Args:
            type_id: ObjectType ID
            instance_id: 인스턴스 ID

        Returns:
            ObjectInstance 또는 None
        """
        obj_type = self.get_object_type(type_id)
        if not obj_type or not obj_type.primary_key:
            return None

        result = (
            self.query(type_id)
            .filter({obj_type.primary_key: instance_id})
            .first()
        )
        return result

    def get_linked_objects(
        self,
        instance: ObjectInstance,
        link_name: str,
        limit: int = 100
    ) -> List[ObjectInstance]:
        """
        연결된 객체 조회

        Args:
            instance: 소스 인스턴스
            link_name: 링크 이름
            limit: 최대 결과 수

        Returns:
            연결된 ObjectInstance 리스트
        """
        # 캐시 확인
        cached = instance.get_linked(link_name)
        if cached:
            return cached

        # ObjectType의 링크 찾기
        obj_type = self.get_object_type(instance.type_id)
        if not obj_type:
            return []

        # Outgoing 링크 확인
        link = None
        for l in obj_type.outgoing_links.values():
            if l.link_id == link_name or l.name == link_name:
                link = l
                break

        # Incoming 링크 확인
        if not link:
            for l in obj_type.incoming_links.values():
                if l.link_id == link_name or l.name == link_name:
                    link = l
                    break

        if not link:
            logger.warning(f"Link not found: {link_name}")
            return []

        # 쿼리 실행
        if link.from_type == instance.type_id:
            # Outgoing: instance.from_property = target.to_property
            fk_value = instance.get(link.from_property)
            if fk_value is None:
                return []

            result = (
                self.query(link.to_type)
                .filter({link.to_property: fk_value})
                .limit(limit)
                .execute()
            )
        else:
            # Incoming: target.from_property = instance.to_property
            pk_value = instance.get(link.to_property)
            if pk_value is None:
                return []

            result = (
                self.query(link.from_type)
                .filter({link.from_property: pk_value})
                .limit(limit)
                .execute()
            )

        # 캐시 저장
        for obj in result.objects:
            instance.add_linked(link_name, obj)

        return result.objects

    # ==================== CRUD Operations ====================

    def create(
        self,
        type_id: str,
        properties: Dict[str, Any]
    ) -> Optional[ObjectInstance]:
        """
        새 인스턴스 생성

        Args:
            type_id: ObjectType ID
            properties: 속성 값들

        Returns:
            생성된 ObjectInstance
        """
        obj_type = self.get_object_type(type_id)
        if not obj_type:
            logger.error(f"ObjectType not found: {type_id}")
            return None

        # 타입 검증
        for prop_name, value in properties.items():
            prop = obj_type.get_property(prop_name)
            if prop and prop.required and value is None:
                logger.error(f"Required property missing: {prop_name}")
                return None

        # SQL INSERT 생성
        columns = list(properties.keys())
        placeholders = ", ".join(["?" for _ in columns])
        cols_quoted = ", ".join([f'"{c}"' for c in columns])
        sql = f'INSERT INTO "{type_id}" ({cols_quoted}) VALUES ({placeholders})'

        try:
            conn = self._ensure_connection()
            cursor = conn.cursor()
            cursor.execute(sql, list(properties.values()))
            conn.commit()

            # ID 가져오기
            instance_id = cursor.lastrowid or properties.get(obj_type.primary_key)

            instance = ObjectInstance(
                instance_id=str(instance_id),
                type_id=type_id,
                properties=properties,
            )

            # Writeback 기록
            if self.context:
                self.context.queue_writeback(
                    action_type="create",
                    target_type=type_id,
                    target_id=str(instance_id),
                    changes=properties,
                    reason="OSDK create operation"
                )

            logger.info(f"Created {type_id} instance: {instance_id}")
            return instance

        except Exception as e:
            logger.error(f"Create failed: {e}")
            return None

    def update(
        self,
        type_id: str,
        instance_id: Any,
        changes: Dict[str, Any]
    ) -> bool:
        """
        인스턴스 업데이트

        Args:
            type_id: ObjectType ID
            instance_id: 인스턴스 ID
            changes: 변경할 속성들

        Returns:
            성공 여부
        """
        obj_type = self.get_object_type(type_id)
        if not obj_type or not obj_type.primary_key:
            return False

        # SQL UPDATE 생성
        set_clauses = [f'"{k}" = ?' for k in changes.keys()]
        sql = f'UPDATE "{type_id}" SET {", ".join(set_clauses)} WHERE "{obj_type.primary_key}" = ?'

        try:
            conn = self._ensure_connection()
            cursor = conn.cursor()
            cursor.execute(sql, list(changes.values()) + [instance_id])
            conn.commit()

            # Writeback 기록
            if self.context:
                self.context.queue_writeback(
                    action_type="update",
                    target_type=type_id,
                    target_id=str(instance_id),
                    changes=changes,
                    reason="OSDK update operation"
                )

            logger.info(f"Updated {type_id} instance: {instance_id}")
            return True

        except Exception as e:
            logger.error(f"Update failed: {e}")
            return False

    def delete(self, type_id: str, instance_id: Any) -> bool:
        """
        인스턴스 삭제

        Args:
            type_id: ObjectType ID
            instance_id: 인스턴스 ID

        Returns:
            성공 여부
        """
        obj_type = self.get_object_type(type_id)
        if not obj_type or not obj_type.primary_key:
            return False

        sql = f'DELETE FROM "{type_id}" WHERE "{obj_type.primary_key}" = ?'

        try:
            conn = self._ensure_connection()
            cursor = conn.cursor()
            cursor.execute(sql, [instance_id])
            conn.commit()

            # Writeback 기록
            if self.context:
                self.context.queue_writeback(
                    action_type="delete",
                    target_type=type_id,
                    target_id=str(instance_id),
                    changes={},
                    reason="OSDK delete operation"
                )

            logger.info(f"Deleted {type_id} instance: {instance_id}")
            return True

        except Exception as e:
            logger.error(f"Delete failed: {e}")
            return False

    # ==================== Internal Methods ====================

    def _ensure_connection(self) -> sqlite3.Connection:
        """SQLite 연결 확보"""
        if self._conn is None:
            self._conn = sqlite3.connect(":memory:")
            self._conn.row_factory = sqlite3.Row

        if not self._tables_loaded and self.context:
            self._load_tables()

        return self._conn

    def _load_tables(self) -> None:
        """테이블 데이터 로드"""
        if self._tables_loaded:
            return

        # NL2SQL의 QueryExecutor와 유사한 로직
        from ..aip.nl2sql.query_executor import QueryExecutor

        executor = QueryExecutor(context=self.context)
        executor._ensure_connection()

        # 연결 공유
        self._conn = executor._conn
        self._tables_loaded = True

        logger.info("OSDK tables loaded")

    def _execute_query(
        self,
        sql: str,
        params: List[Any],
        type_id: str
    ) -> QueryResult:
        """쿼리 실행 및 ObjectInstance 변환"""
        conn = self._ensure_connection()
        cursor = conn.cursor()

        try:
            cursor.execute(sql, params)
            rows = cursor.fetchall()

            obj_type = self.get_object_type(type_id)
            pk = obj_type.primary_key if obj_type else "id"

            objects = [
                ObjectInstance.from_row(dict(row), type_id, pk)
                for row in rows
            ]

            return QueryResult(
                objects=objects,
                total_count=len(objects),
            )

        except Exception as e:
            logger.error(f"Query execution failed: {e}")
            return QueryResult()

    def _execute_raw_query(
        self,
        sql: str,
        params: List[Any]
    ) -> List[Dict[str, Any]]:
        """원시 쿼리 실행"""
        conn = self._ensure_connection()
        cursor = conn.cursor()

        try:
            cursor.execute(sql, params)
            rows = cursor.fetchall()
            return [dict(row) for row in rows]
        except Exception as e:
            logger.error(f"Raw query failed: {e}")
            return []

    def get_schema_summary(self) -> Dict[str, Any]:
        """스키마 요약 정보"""
        return {
            "object_types": [
                {
                    "type_id": t.type_id,
                    "name": t.name,
                    "properties_count": len(t.properties),
                    "links_count": len(t.get_all_links()),
                }
                for t in self.registry.get_all_types()
            ],
            "link_types": [
                {
                    "link_id": l.link_id,
                    "from": l.from_type,
                    "to": l.to_type,
                    "cardinality": l.cardinality.value,
                }
                for l in self.registry.get_all_links()
            ],
            "total_types": len(self.registry.types),
            "total_links": len(self.registry.links),
        }
