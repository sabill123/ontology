"""
Enhanced Entity Extraction Module

컬럼 기반 엔티티 추출 알고리즘:
1. Column-based Entity Detection - ID 컬럼에서 숨은 엔티티 탐지
2. Cross-silo Entity Unification - 여러 사일로에 걸친 공통 엔티티
3. Entity Hierarchy Detection - 엔티티 계층 구조 탐지
4. Semantic Entity Classification - 의미론적 엔티티 분류

References:
- Deng et al. "Schema Matching and Mapping" (2015)
- Rahm & Bernstein "A Survey of Approaches to Automatic Schema Matching" (2001)
- Lehmberg et al. "A Large Public Corpus of Web Tables" (2016)
"""

import re
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Set, Tuple
from collections import defaultdict, Counter
from enum import Enum

logger = logging.getLogger(__name__)


class EntityType(str, Enum):
    """엔티티 유형"""
    TABLE_ENTITY = "table_entity"       # 테이블에서 직접 파생
    COLUMN_ENTITY = "column_entity"     # ID 컬럼에서 추출 (숨은 엔티티)
    CROSS_SILO = "cross_silo"           # 여러 사일로에 걸친 공통 엔티티
    ENUMERATION = "enumeration"         # 열거형 값
    HIERARCHY = "hierarchy"             # 계층적 엔티티


@dataclass
class ExtractedEntity:
    """추출된 엔티티"""
    entity_id: str
    name: str
    entity_type: EntityType
    source_tables: List[str]
    source_columns: List[str]
    source_systems: List[str]

    # 메타데이터
    unique_count: int = 0
    total_references: int = 0
    sample_values: List[Any] = field(default_factory=list)

    # 속성
    inferred_attributes: List[str] = field(default_factory=list)
    relationships: List[Dict] = field(default_factory=list)

    # 신뢰도
    confidence: float = 0.0
    evidence: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "entity_id": self.entity_id,
            "name": self.name,
            "entity_type": self.entity_type.value,
            "source_tables": self.source_tables,
            "source_columns": self.source_columns,
            "source_systems": self.source_systems,
            "unique_count": self.unique_count,
            "total_references": self.total_references,
            "sample_values": self.sample_values[:10],
            "inferred_attributes": self.inferred_attributes,
            "relationships": self.relationships,
            "confidence": round(self.confidence, 4),
            "evidence": self.evidence,
        }


class EntityExtractor:
    """
    강화된 엔티티 추출기

    핵심 기능:
    1. 테이블에서 명시적 엔티티 추출
    2. ID 컬럼에서 숨은 엔티티 추출 (customer_id -> Customer)
    3. 여러 테이블에 걸친 공통 엔티티 통합 (warehouse_id)
    4. 엔티티 속성 및 관계 추론
    """

    # 엔티티 후보 컬럼 패턴
    ID_PATTERNS = [
        (r'^(.+)_id$', 1),           # xxx_id -> xxx
        (r'^id_(.+)$', 1),           # id_xxx -> xxx
        (r'^(.+)_key$', 1),          # xxx_key -> xxx
        (r'^(.+)_code$', 1),         # xxx_code -> xxx
        (r'^(.+)_uuid$', 1),         # xxx_uuid -> xxx
        (r'^(.+)_guid$', 1),         # xxx_guid -> xxx
    ]

    # 시스템명 패턴 (테이블명에서 추출)
    SYSTEM_PATTERNS = [
        r'^([a-z]+)__',              # oms__, tms__, srm__
        r'^([a-z]+)_',               # 단일 언더스코어
    ]

    # 엔티티 도메인 키워드
    DOMAIN_KEYWORDS = {
        "customer": ["customer", "client", "buyer", "consumer", "account"],
        "product": ["product", "item", "sku", "goods", "merchandise"],
        "order": ["order", "purchase", "transaction", "sale"],
        "supplier": ["supplier", "vendor", "provider", "manufacturer"],
        "warehouse": ["warehouse", "wh", "storage", "depot", "facility"],
        "shipment": ["shipment", "delivery", "transport", "freight"],
        "vehicle": ["vehicle", "truck", "car", "fleet"],
        "driver": ["driver", "operator", "courier"],
        "employee": ["employee", "staff", "worker", "user"],
        "location": ["location", "address", "zone", "region"],
        # v4.7: Supply Chain 추가 도메인 키워드
        "carrier": ["carrier", "shipper", "transporter", "logistics_provider"],
        "port": ["port", "terminal", "harbor", "dock"],
        "plant": ["plant", "factory", "manufacturing_site"],
        "rate": ["rate", "freight_rate", "shipping_rate", "tariff"],
    }

    # v4.7: 단순 컬럼명에서 엔티티 추출 (ID 접미사 없이도 감지)
    SIMPLE_ENTITY_COLUMNS = {
        "carrier": "Carrier",
        "customer": "Customer",
        "supplier": "Supplier",
        "product": "Product",
        "plant": "Plant",
        "port": "Port",
        "warehouse": "Warehouse",
    }

    def __init__(self):
        self.entities: Dict[str, ExtractedEntity] = {}
        self.column_entity_map: Dict[str, str] = {}  # column -> entity_id

    def extract_entities(
        self,
        tables: Dict[str, Dict[str, Any]],
        sample_data: Dict[str, List[Dict[str, Any]]] = None,
    ) -> List[ExtractedEntity]:
        """
        모든 엔티티 추출

        Args:
            tables: {table_name: {"columns": [...], "row_count": int}}
            sample_data: {table_name: [row_dicts]}

        Returns:
            추출된 엔티티 리스트
        """
        sample_data = sample_data or {}
        self.entities = {}

        # 1. 테이블 기반 엔티티 추출
        table_entities = self._extract_table_entities(tables, sample_data)

        # 2. 컬럼 기반 숨은 엔티티 추출
        column_entities = self._extract_column_entities(tables, sample_data)

        # 3. Cross-silo 공통 엔티티 탐지
        cross_silo_entities = self._detect_cross_silo_entities(tables, sample_data)

        # 4. 열거형 엔티티 추출
        enum_entities = self._extract_enumeration_entities(tables, sample_data)

        # 5. 모든 엔티티 통합
        all_entities = table_entities + column_entities + cross_silo_entities + enum_entities

        # 6. 중복 병합 및 강화
        merged_entities = self._merge_and_enrich_entities(all_entities, tables, sample_data)

        return merged_entities

    def _extract_table_entities(
        self,
        tables: Dict[str, Dict[str, Any]],
        sample_data: Dict[str, List[Dict[str, Any]]],
    ) -> List[ExtractedEntity]:
        """테이블에서 명시적 엔티티 추출"""
        entities = []

        for table_name, table_info in tables.items():
            # 시스템명 추출
            system = self._extract_system_name(table_name)

            # 엔티티명 추출 (시스템 접두사 제거)
            entity_name = self._extract_entity_name(table_name)

            # 모든 컬럼을 속성으로
            columns = table_info.get("columns", [])
            col_names = [c.get("name", "") for c in columns]

            # ID 컬럼 찾기
            id_columns = [c for c in col_names if self._is_primary_key_pattern(c)]

            entity = ExtractedEntity(
                entity_id=f"ENTITY_{entity_name.upper()}",
                name=self._humanize_name(entity_name),
                entity_type=EntityType.TABLE_ENTITY,
                source_tables=[table_name],
                source_columns=id_columns,
                source_systems=[system] if system else [],
                unique_count=table_info.get("row_count", 0),
                total_references=table_info.get("row_count", 0),
                inferred_attributes=col_names,
                confidence=0.95,
                evidence=[f"Direct extraction from table '{table_name}'"],
            )

            # 샘플 값 추가
            if id_columns and table_name in sample_data:
                pk_col = id_columns[0]
                entity.sample_values = [
                    row.get(pk_col) for row in sample_data[table_name][:10]
                    if pk_col in row
                ]

            entities.append(entity)
            self.entities[entity.entity_id] = entity

        return entities

    def _extract_column_entities(
        self,
        tables: Dict[str, Dict[str, Any]],
        sample_data: Dict[str, List[Dict[str, Any]]],
    ) -> List[ExtractedEntity]:
        """ID 컬럼에서 숨은 엔티티 추출 (핵심 기능)"""
        entities = []
        column_analysis: Dict[str, Dict] = defaultdict(lambda: {
            "tables": [],
            "systems": [],
            "values": [],
            "unique_count": 0,
            "total_count": 0,
        })

        # 모든 테이블의 ID 컬럼 분석
        for table_name, table_info in tables.items():
            system = self._extract_system_name(table_name)

            for col in table_info.get("columns", []):
                col_name = col.get("name", "")

                # ID 패턴 매칭
                base_entity = self._extract_entity_from_column(col_name)

                # v4.7: 단순 컬럼명에서도 엔티티 추출 (Carrier, Customer 등)
                if not base_entity:
                    col_lower = col_name.lower()
                    for simple_col, entity_name in self.SIMPLE_ENTITY_COLUMNS.items():
                        if col_lower == simple_col:
                            base_entity = simple_col
                            break

                if base_entity:
                    # 해당 테이블의 PK가 아닌 경우만 (FK로 추정)
                    if not self._is_likely_pk(col_name, table_name):
                        analysis = column_analysis[base_entity]
                        analysis["tables"].append(table_name)

                        if system:
                            analysis["systems"].append(system)

                        # 샘플 값 수집
                        if table_name in sample_data:
                            values = [
                                row.get(col_name) for row in sample_data[table_name]
                                if col_name in row and row[col_name] is not None
                            ]
                            analysis["values"].extend(values)
                            analysis["total_count"] += len(values)

        # 분석 결과로 엔티티 생성
        for base_entity, analysis in column_analysis.items():
            if len(analysis["tables"]) == 0:
                continue

            # 이미 테이블 엔티티로 존재하면 스킵
            entity_id = f"ENTITY_{base_entity.upper()}"
            if entity_id in self.entities:
                continue

            unique_values = set(str(v) for v in analysis["values"] if v is not None)

            entity = ExtractedEntity(
                entity_id=entity_id,
                name=self._humanize_name(base_entity),
                entity_type=EntityType.COLUMN_ENTITY,
                source_tables=list(set(analysis["tables"])),
                source_columns=[f"{base_entity}_id"],
                source_systems=list(set(analysis["systems"])),
                unique_count=len(unique_values),
                total_references=analysis["total_count"],
                sample_values=list(unique_values)[:10],
                confidence=self._calculate_column_entity_confidence(analysis),
                evidence=[
                    f"Extracted from ID column in {len(analysis['tables'])} tables",
                    f"Referenced {analysis['total_count']} times",
                    f"Found {len(unique_values)} unique values",
                ],
            )

            entities.append(entity)
            self.entities[entity_id] = entity

        return entities

    def _detect_cross_silo_entities(
        self,
        tables: Dict[str, Dict[str, Any]],
        sample_data: Dict[str, List[Dict[str, Any]]],
    ) -> List[ExtractedEntity]:
        """여러 사일로에 걸친 공통 엔티티 탐지"""
        entities = []

        # 시스템별 컬럼 수집
        system_columns: Dict[str, Dict[str, Set[str]]] = defaultdict(lambda: defaultdict(set))

        for table_name, table_info in tables.items():
            system = self._extract_system_name(table_name)
            if not system:
                continue

            for col in table_info.get("columns", []):
                col_name = col.get("name", "")
                if self._is_id_like_column(col_name):
                    # 정규화된 이름
                    base = self._normalize_column_name(col_name)
                    system_columns[base][system].add(table_name)

        # 2개 이상 시스템에 존재하는 컬럼 = cross-silo entity
        for base_col, systems_tables in system_columns.items():
            if len(systems_tables) >= 2:
                # 이미 존재하는 엔티티 확인
                base_entity = self._extract_entity_from_column(base_col)
                if not base_entity:
                    base_entity = base_col.replace("_id", "").replace("id_", "")

                entity_id = f"CROSSSILO_{base_entity.upper()}"

                # 기존 엔티티와 병합 대상인지 확인
                existing_id = f"ENTITY_{base_entity.upper()}"
                if existing_id in self.entities:
                    # 기존 엔티티에 cross-silo 정보 추가
                    existing = self.entities[existing_id]
                    for sys_name, tables_set in systems_tables.items():
                        if sys_name not in existing.source_systems:
                            existing.source_systems.append(sys_name)
                        existing.source_tables.extend(list(tables_set))
                    existing.source_tables = list(set(existing.source_tables))
                    existing.evidence.append(f"Cross-silo entity across {list(systems_tables.keys())}")
                    continue

                all_tables = []
                all_systems = []
                for sys_name, tables_set in systems_tables.items():
                    all_tables.extend(list(tables_set))
                    all_systems.append(sys_name)

                # 값 수집
                all_values = []
                for table in all_tables:
                    if table in sample_data:
                        for row in sample_data[table]:
                            if base_col in row and row[base_col] is not None:
                                all_values.append(row[base_col])

                entity = ExtractedEntity(
                    entity_id=entity_id,
                    name=self._humanize_name(base_entity),
                    entity_type=EntityType.CROSS_SILO,
                    source_tables=list(set(all_tables)),
                    source_columns=[base_col],
                    source_systems=list(set(all_systems)),
                    unique_count=len(set(str(v) for v in all_values)),
                    total_references=len(all_values),
                    sample_values=list(set(str(v) for v in all_values))[:10],
                    confidence=0.9,
                    evidence=[
                        f"Cross-silo entity across {len(all_systems)} systems: {all_systems}",
                        f"Found in {len(all_tables)} tables",
                    ],
                )

                entities.append(entity)
                self.entities[entity_id] = entity

        return entities

    def _extract_enumeration_entities(
        self,
        tables: Dict[str, Dict[str, Any]],
        sample_data: Dict[str, List[Dict[str, Any]]],
    ) -> List[ExtractedEntity]:
        """열거형 값에서 엔티티 추출"""
        entities = []

        for table_name, table_info in tables.items():
            for col in table_info.get("columns", []):
                col_name = col.get("name", "")

                # status, type, category 패턴
                if not any(pat in col_name.lower() for pat in ["status", "type", "category", "state", "mode"]):
                    continue

                # 샘플 데이터에서 unique 값 수집
                if table_name not in sample_data:
                    continue

                values = [
                    row.get(col_name) for row in sample_data[table_name]
                    if col_name in row and row[col_name] is not None
                ]

                unique_values = set(str(v) for v in values)

                # 10개 이하의 unique 값만 열거형으로 취급
                if 2 <= len(unique_values) <= 15:
                    entity = ExtractedEntity(
                        entity_id=f"ENUM_{table_name.upper()}_{col_name.upper()}",
                        name=f"{self._humanize_name(col_name)} Values",
                        entity_type=EntityType.ENUMERATION,
                        source_tables=[table_name],
                        source_columns=[col_name],
                        source_systems=[self._extract_system_name(table_name) or "unknown"],
                        unique_count=len(unique_values),
                        total_references=len(values),
                        sample_values=list(unique_values),
                        confidence=0.85,
                        evidence=[
                            f"Enumeration with {len(unique_values)} values",
                            f"Values: {list(unique_values)[:5]}",
                        ],
                    )

                    entities.append(entity)

        return entities

    def _merge_and_enrich_entities(
        self,
        entities: List[ExtractedEntity],
        tables: Dict[str, Dict[str, Any]],
        sample_data: Dict[str, List[Dict[str, Any]]],
    ) -> List[ExtractedEntity]:
        """엔티티 병합 및 강화"""

        # ID 기반 그룹화
        entity_groups: Dict[str, List[ExtractedEntity]] = defaultdict(list)

        for entity in entities:
            # 정규화된 base name
            base = entity.name.lower().replace(" ", "_")
            entity_groups[base].append(entity)

        merged = []
        seen_ids = set()

        for base_name, group in entity_groups.items():
            if len(group) == 1:
                merged.append(group[0])
                seen_ids.add(group[0].entity_id)
                continue

            # 여러 엔티티 병합 (가장 높은 confidence 기준)
            group.sort(key=lambda x: x.confidence, reverse=True)
            primary = group[0]

            for secondary in group[1:]:
                if secondary.entity_id in seen_ids:
                    continue

                # 소스 정보 병합
                primary.source_tables = list(set(primary.source_tables + secondary.source_tables))
                primary.source_columns = list(set(primary.source_columns + secondary.source_columns))
                primary.source_systems = list(set(primary.source_systems + secondary.source_systems))
                primary.evidence.extend(secondary.evidence)

                # 카운트 합산
                primary.total_references += secondary.total_references
                primary.unique_count = max(primary.unique_count, secondary.unique_count)

            merged.append(primary)
            seen_ids.add(primary.entity_id)

        # 관계 추론
        self._infer_relationships(merged, tables)

        return merged

    def _infer_relationships(
        self,
        entities: List[ExtractedEntity],
        tables: Dict[str, Dict[str, Any]],
    ):
        """엔티티 간 관계 추론"""

        entity_map = {e.name.lower(): e for e in entities}

        for entity in entities:
            for table in entity.source_tables:
                if table not in tables:
                    continue

                for col in tables[table].get("columns", []):
                    col_name = col.get("name", "")
                    base = self._extract_entity_from_column(col_name)

                    if base and base.lower() in entity_map:
                        target = entity_map[base.lower()]

                        if target.entity_id != entity.entity_id:
                            rel = {
                                "from_entity": entity.entity_id,
                                "to_entity": target.entity_id,
                                "relationship_type": "references",
                                "via_column": col_name,
                                "in_table": table,
                            }

                            if rel not in entity.relationships:
                                entity.relationships.append(rel)

    # === Helper Methods ===

    def _extract_system_name(self, table_name: str) -> Optional[str]:
        """테이블명에서 시스템명 추출"""
        for pattern in self.SYSTEM_PATTERNS:
            match = re.match(pattern, table_name.lower())
            if match:
                return match.group(1).upper()
        return None

    def _extract_entity_name(self, table_name: str) -> str:
        """테이블명에서 엔티티명 추출"""
        name = table_name.lower()

        # 시스템 접두사 제거
        for pattern in self.SYSTEM_PATTERNS:
            name = re.sub(pattern, '', name)

        return name

    def _extract_entity_from_column(self, col_name: str) -> Optional[str]:
        """컬럼명에서 엔티티명 추출"""
        col_lower = col_name.lower()

        for pattern, group in self.ID_PATTERNS:
            match = re.match(pattern, col_lower)
            if match:
                return match.group(group)

        return None

    def _is_primary_key_pattern(self, col_name: str) -> bool:
        """PK 패턴 확인"""
        col_lower = col_name.lower()
        return col_lower == "id" or col_lower.endswith("_id") and not col_lower.startswith("fk_")

    def _is_likely_pk(self, col_name: str, table_name: str) -> bool:
        """해당 테이블의 PK인지 추정"""
        col_lower = col_name.lower()
        table_lower = table_name.lower()

        # 테이블명과 연관된 ID인지
        entity_name = self._extract_entity_name(table_name)

        if col_lower == "id":
            return True

        if col_lower == f"{entity_name}_id":
            return True

        if col_lower.endswith("_id"):
            base = col_lower[:-3]
            if base in table_lower:
                return True

        return False

    def _is_id_like_column(self, col_name: str) -> bool:
        """ID 유사 컬럼 확인"""
        col_lower = col_name.lower()
        return "_id" in col_lower or col_lower.startswith("id_") or col_lower == "id"

    def _normalize_column_name(self, col_name: str) -> str:
        """컬럼명 정규화"""
        return col_name.lower().strip()

    def _humanize_name(self, name: str) -> str:
        """이름을 사람이 읽기 좋게 변환"""
        # 언더스코어를 공백으로, 첫 글자 대문자
        words = name.replace("_", " ").split()
        return " ".join(word.capitalize() for word in words)

    def _calculate_column_entity_confidence(self, analysis: Dict) -> float:
        """컬럼 기반 엔티티의 신뢰도 계산"""
        confidence = 0.5

        # 여러 테이블에서 참조되면 높은 신뢰도
        if len(analysis["tables"]) >= 3:
            confidence += 0.3
        elif len(analysis["tables"]) >= 2:
            confidence += 0.2

        # 여러 시스템에 걸쳐있으면 더 높은 신뢰도
        unique_systems = len(set(analysis["systems"]))
        if unique_systems >= 2:
            confidence += 0.15

        # 많은 참조가 있으면 높은 신뢰도
        if analysis["total_count"] >= 10000:
            confidence += 0.1
        elif analysis["total_count"] >= 1000:
            confidence += 0.05

        return min(0.95, confidence)
