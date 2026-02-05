"""
OSDK Models (v17.0)

온톨로지 SDK의 핵심 모델:
- ObjectType: 엔티티 유형 정의
- LinkType: 관계 유형 정의
- Property: 속성 정의
- ObjectInstance: 객체 인스턴스
"""

import logging
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Any, Optional, Union, Callable
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class PropertyType(str, Enum):
    """속성 데이터 타입"""
    STRING = "string"
    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"
    DATE = "date"
    DATETIME = "datetime"
    TIMESTAMP = "timestamp"
    ARRAY = "array"
    OBJECT = "object"
    REFERENCE = "reference"  # 다른 ObjectType 참조


class Cardinality(str, Enum):
    """관계 카디널리티"""
    ONE_TO_ONE = "one_to_one"
    ONE_TO_MANY = "one_to_many"
    MANY_TO_ONE = "many_to_one"
    MANY_TO_MANY = "many_to_many"


@dataclass
class Property:
    """속성 정의"""
    name: str
    property_type: PropertyType
    display_name: Optional[str] = None
    description: Optional[str] = None

    # 제약조건
    required: bool = False
    unique: bool = False
    indexed: bool = False
    default_value: Optional[Any] = None

    # 참조 (PropertyType.REFERENCE인 경우)
    references_type: Optional[str] = None  # ObjectType ID

    # 메타데이터
    source_column: Optional[str] = None  # 원본 DB 컬럼
    source_table: Optional[str] = None  # 원본 DB 테이블

    # Derived Property (계산된 속성)
    is_derived: bool = False
    derivation_formula: Optional[str] = None  # e.g., "SUM(orders.amount)"

    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result["property_type"] = self.property_type.value
        return result

    @classmethod
    def from_column(
        cls,
        column_name: str,
        data_type: str,
        table_name: str,
        **kwargs
    ) -> "Property":
        """DB 컬럼에서 Property 생성"""
        # 데이터 타입 매핑
        type_map = {
            "int": PropertyType.INTEGER,
            "integer": PropertyType.INTEGER,
            "bigint": PropertyType.INTEGER,
            "float": PropertyType.FLOAT,
            "double": PropertyType.FLOAT,
            "decimal": PropertyType.FLOAT,
            "bool": PropertyType.BOOLEAN,
            "boolean": PropertyType.BOOLEAN,
            "date": PropertyType.DATE,
            "datetime": PropertyType.DATETIME,
            "timestamp": PropertyType.TIMESTAMP,
            "text": PropertyType.STRING,
            "varchar": PropertyType.STRING,
            "char": PropertyType.STRING,
        }

        property_type = PropertyType.STRING
        for key, ptype in type_map.items():
            if key in data_type.lower():
                property_type = ptype
                break

        return cls(
            name=column_name,
            property_type=property_type,
            source_column=column_name,
            source_table=table_name,
            **kwargs
        )


@dataclass
class LinkType:
    """관계 유형 정의"""
    link_id: str
    name: str
    display_name: Optional[str] = None
    description: Optional[str] = None

    # 관계 정의
    from_type: str = ""  # Source ObjectType ID
    to_type: str = ""  # Target ObjectType ID
    cardinality: Cardinality = Cardinality.MANY_TO_ONE

    # 조인 조건
    from_property: str = ""  # FK 속성
    to_property: str = ""  # PK 속성

    # 메타데이터
    bidirectional: bool = True  # 양방향 탐색 가능 여부
    cascade_delete: bool = False

    # 생성 정보
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    source_fk: Optional[Dict[str, Any]] = None  # 원본 FK 정보

    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result["cardinality"] = self.cardinality.value
        return result

    def get_inverse_name(self) -> str:
        """역방향 관계 이름"""
        # e.g., "orders" → "customer"
        if self.cardinality == Cardinality.ONE_TO_MANY:
            return f"{self.from_type.lower()}"
        return f"{self.from_type.lower()}s"


@dataclass
class ObjectType:
    """
    객체 유형 정의

    Palantir Ontology의 ObjectType에 해당:
    - 속성(Property) 집합
    - 관계(LinkType) 집합
    - 타입 안전 접근 API
    """
    type_id: str
    name: str
    display_name: Optional[str] = None
    description: Optional[str] = None

    # 속성 및 관계
    properties: Dict[str, Property] = field(default_factory=dict)
    outgoing_links: Dict[str, LinkType] = field(default_factory=dict)  # 나가는 관계
    incoming_links: Dict[str, LinkType] = field(default_factory=dict)  # 들어오는 관계

    # Primary Key
    primary_key: Optional[str] = None  # Property name

    # 메타데이터
    source_tables: List[str] = field(default_factory=list)
    entity_type: str = "master_data"  # master_data, transaction, reference, junction

    # 생성 정보
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def add_property(self, prop: Property) -> None:
        """속성 추가"""
        self.properties[prop.name] = prop
        self.updated_at = datetime.now().isoformat()

    def add_outgoing_link(self, link: LinkType) -> None:
        """나가는 관계 추가"""
        self.outgoing_links[link.link_id] = link
        self.updated_at = datetime.now().isoformat()

    def add_incoming_link(self, link: LinkType) -> None:
        """들어오는 관계 추가"""
        self.incoming_links[link.link_id] = link
        self.updated_at = datetime.now().isoformat()

    def get_property(self, name: str) -> Optional[Property]:
        """속성 조회"""
        return self.properties.get(name)

    def get_property_names(self) -> List[str]:
        """모든 속성 이름"""
        return list(self.properties.keys())

    def get_all_links(self) -> Dict[str, LinkType]:
        """모든 관계 (나가는 + 들어오는)"""
        all_links = {}
        all_links.update(self.outgoing_links)
        all_links.update(self.incoming_links)
        return all_links

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type_id": self.type_id,
            "name": self.name,
            "display_name": self.display_name,
            "description": self.description,
            "properties": {k: v.to_dict() for k, v in self.properties.items()},
            "outgoing_links": {k: v.to_dict() for k, v in self.outgoing_links.items()},
            "incoming_links": {k: v.to_dict() for k, v in self.incoming_links.items()},
            "primary_key": self.primary_key,
            "source_tables": self.source_tables,
            "entity_type": self.entity_type,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_table_info(
        cls,
        table_name: str,
        columns: List[Dict[str, Any]],
        primary_keys: List[str] = None,
        entity_name: Optional[str] = None,
    ) -> "ObjectType":
        """테이블 정보에서 ObjectType 생성"""
        type_id = table_name.lower().replace(" ", "_")
        name = entity_name or table_name.replace("_", " ").title().replace(" ", "")

        obj_type = cls(
            type_id=type_id,
            name=name,
            display_name=name,
            source_tables=[table_name],
            primary_key=primary_keys[0] if primary_keys else None,
        )

        # 속성 추가
        for col in columns:
            col_name = col.get("name", "")
            data_type = col.get("data_type", col.get("type", "string"))

            prop = Property.from_column(
                column_name=col_name,
                data_type=data_type,
                table_name=table_name,
                required=not col.get("nullable", True),
                unique=col_name in (primary_keys or []),
            )

            if col_name in (primary_keys or []):
                prop.indexed = True

            obj_type.add_property(prop)

        return obj_type


@dataclass
class ObjectInstance:
    """
    객체 인스턴스

    ObjectType의 실제 데이터 인스턴스
    """
    instance_id: str
    type_id: str
    properties: Dict[str, Any] = field(default_factory=dict)

    # 관계 캐시
    _linked_objects: Dict[str, List["ObjectInstance"]] = field(
        default_factory=dict, repr=False
    )

    # 메타데이터
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    source_record: Optional[Dict[str, Any]] = None

    def get(self, property_name: str, default: Any = None) -> Any:
        """속성 값 조회"""
        return self.properties.get(property_name, default)

    def set(self, property_name: str, value: Any) -> None:
        """속성 값 설정"""
        self.properties[property_name] = value
        self.updated_at = datetime.now().isoformat()

    def get_linked(self, link_name: str) -> List["ObjectInstance"]:
        """연결된 객체 조회"""
        return self._linked_objects.get(link_name, [])

    def add_linked(self, link_name: str, instance: "ObjectInstance") -> None:
        """연결된 객체 추가"""
        if link_name not in self._linked_objects:
            self._linked_objects[link_name] = []
        self._linked_objects[link_name].append(instance)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "instance_id": self.instance_id,
            "type_id": self.type_id,
            "properties": self.properties,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_row(
        cls,
        row: Dict[str, Any],
        type_id: str,
        id_property: str = "id"
    ) -> "ObjectInstance":
        """DB 행에서 ObjectInstance 생성"""
        instance_id = str(row.get(id_property, hash(frozenset(row.items()))))

        return cls(
            instance_id=instance_id,
            type_id=type_id,
            properties=row.copy(),
            source_record=row,
        )


# Type Registry for validation
class TypeRegistry:
    """ObjectType 레지스트리"""

    def __init__(self):
        self.types: Dict[str, ObjectType] = {}
        self.links: Dict[str, LinkType] = {}

    def register_type(self, obj_type: ObjectType) -> None:
        """ObjectType 등록"""
        self.types[obj_type.type_id] = obj_type
        logger.debug(f"Registered ObjectType: {obj_type.type_id}")

    def register_link(self, link: LinkType) -> None:
        """LinkType 등록"""
        self.links[link.link_id] = link

        # 양쪽 ObjectType에 관계 추가
        if link.from_type in self.types:
            self.types[link.from_type].add_outgoing_link(link)
        if link.to_type in self.types:
            self.types[link.to_type].add_incoming_link(link)

        logger.debug(f"Registered LinkType: {link.link_id}")

    def get_type(self, type_id: str) -> Optional[ObjectType]:
        """ObjectType 조회"""
        return self.types.get(type_id)

    def get_link(self, link_id: str) -> Optional[LinkType]:
        """LinkType 조회"""
        return self.links.get(link_id)

    def get_all_types(self) -> List[ObjectType]:
        """모든 ObjectType 반환"""
        return list(self.types.values())

    def get_all_links(self) -> List[LinkType]:
        """모든 LinkType 반환"""
        return list(self.links.values())
