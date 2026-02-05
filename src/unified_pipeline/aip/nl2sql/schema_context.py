"""
Schema Context for NL2SQL (v17.0)

온톨로지 기반 스키마 컨텍스트:
- SharedContext에서 스키마 정보 추출
- LLM 친화적인 형식으로 변환
- 테이블/컬럼 메타데이터 관리
- FK 관계 인덱싱
"""

import json
import logging
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ...shared_context import SharedContext

logger = logging.getLogger(__name__)


@dataclass
class ColumnMeta:
    """컬럼 메타데이터"""
    name: str
    data_type: str
    is_nullable: bool = True
    is_primary_key: bool = False
    is_foreign_key: bool = False
    references_table: Optional[str] = None
    references_column: Optional[str] = None

    # 의미론적 정보
    semantic_type: Optional[str] = None  # IDENTIFIER, DIMENSION, MEASURE, etc.
    business_name: Optional[str] = None  # 비즈니스 친화적 이름
    description: Optional[str] = None

    # 통계 정보
    distinct_count: Optional[int] = None
    null_ratio: Optional[float] = None
    sample_values: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_llm_string(self) -> str:
        """LLM 친화적 문자열 표현"""
        parts = [f"{self.name} ({self.data_type})"]

        if self.is_primary_key:
            parts.append("PRIMARY KEY")
        if self.is_foreign_key and self.references_table:
            parts.append(f"FK → {self.references_table}.{self.references_column}")
        if self.business_name and self.business_name != self.name:
            parts.append(f"[{self.business_name}]")

        return " ".join(parts)


@dataclass
class TableMeta:
    """테이블 메타데이터"""
    name: str
    columns: List[ColumnMeta] = field(default_factory=list)

    # 의미론적 정보
    entity_name: Optional[str] = None  # 비즈니스 엔티티 이름
    entity_type: Optional[str] = None  # master_data, transaction, etc.
    description: Optional[str] = None

    # 통계 정보
    row_count: Optional[int] = None
    source: Optional[str] = None  # CRM, ERP, etc.

    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result["columns"] = [c.to_dict() for c in self.columns]
        return result

    def get_column(self, name: str) -> Optional[ColumnMeta]:
        """컬럼 조회"""
        for col in self.columns:
            if col.name == name:
                return col
        return None

    def get_primary_key(self) -> Optional[ColumnMeta]:
        """PK 컬럼 반환"""
        for col in self.columns:
            if col.is_primary_key:
                return col
        return None

    def get_foreign_keys(self) -> List[ColumnMeta]:
        """FK 컬럼 목록 반환"""
        return [col for col in self.columns if col.is_foreign_key]

    def to_llm_string(self) -> str:
        """LLM 친화적 문자열 표현"""
        lines = [f"TABLE: {self.name}"]
        if self.entity_name:
            lines[0] += f" ({self.entity_name})"
        if self.description:
            lines.append(f"  Description: {self.description}")
        lines.append("  Columns:")
        for col in self.columns:
            lines.append(f"    - {col.to_llm_string()}")
        return "\n".join(lines)


@dataclass
class ForeignKeyMeta:
    """FK 관계 메타데이터"""
    from_table: str
    from_column: str
    to_table: str
    to_column: str
    confidence: float = 1.0
    relationship_type: str = "foreign_key"  # foreign_key, semantic_link, etc.

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class SchemaContext:
    """
    NL2SQL을 위한 스키마 컨텍스트

    SharedContext에서 테이블/컬럼/FK 정보를 추출하여
    LLM이 SQL을 생성할 때 참조할 수 있는 형태로 관리합니다.
    """

    def __init__(self, context: Optional["SharedContext"] = None):
        self.tables: Dict[str, TableMeta] = {}
        self.foreign_keys: List[ForeignKeyMeta] = []

        # 인덱스
        self._fk_by_table: Dict[str, List[ForeignKeyMeta]] = {}  # table -> outgoing FKs
        self._fk_to_table: Dict[str, List[ForeignKeyMeta]] = {}  # table -> incoming FKs

        if context:
            self.sync_from_context(context)

    def sync_from_context(self, context: "SharedContext") -> None:
        """
        SharedContext에서 스키마 정보 동기화

        Args:
            context: SharedContext 인스턴스
        """
        logger.info("Syncing schema context from SharedContext...")

        # 1. 테이블 정보 추출
        for table_name, table_info in context.tables.items():
            columns = []

            for col_info in table_info.columns:
                col_name = col_info.get("name", "")
                col = ColumnMeta(
                    name=col_name,
                    data_type=col_info.get("data_type", col_info.get("type", "unknown")),
                    is_nullable=col_info.get("nullable", True),
                    is_primary_key=col_name in table_info.primary_keys,
                )
                columns.append(col)

            table_meta = TableMeta(
                name=table_name,
                columns=columns,
                row_count=table_info.row_count,
                source=table_info.source,
            )

            # 엔티티 정보 연결
            for entity in context.unified_entities:
                if table_name in entity.source_tables:
                    table_meta.entity_name = entity.canonical_name
                    table_meta.entity_type = entity.entity_type
                    break

            self.tables[table_name] = table_meta

        # 2. FK 정보 추출
        self._extract_foreign_keys(context)

        # 3. 의미론적 정보 보강 (dynamic_data에서)
        column_semantics = context.get_dynamic("column_semantics", {})
        for col_key, semantics in column_semantics.items():
            # col_key는 "table.column" 또는 "column" 형식
            parts = col_key.split(".")
            if len(parts) == 2:
                table_name, col_name = parts
            else:
                col_name = parts[0]
                table_name = None

            # 해당 컬럼 찾아서 업데이트
            for table in self.tables.values():
                if table_name and table.name != table_name:
                    continue
                col = table.get_column(col_name)
                if col:
                    col.semantic_type = semantics.get("semantic_role")
                    col.business_name = semantics.get("business_name")

        logger.info(
            f"Schema context synced: {len(self.tables)} tables, "
            f"{len(self.foreign_keys)} foreign keys"
        )

    def _extract_foreign_keys(self, context: "SharedContext") -> None:
        """FK 관계 추출"""
        self.foreign_keys = []
        self._fk_by_table = {}
        self._fk_to_table = {}

        # enhanced_fk_candidates에서 추출
        for fk in context.enhanced_fk_candidates:
            # dict 또는 object 형태 모두 지원
            if isinstance(fk, dict):
                fk_meta = ForeignKeyMeta(
                    from_table=fk.get("from_table", ""),
                    from_column=fk.get("from_column", ""),
                    to_table=fk.get("to_table", ""),
                    to_column=fk.get("to_column", ""),
                    confidence=fk.get("confidence", 0.5),
                    relationship_type=fk.get("relationship_type", "foreign_key"),
                )
            else:
                fk_meta = ForeignKeyMeta(
                    from_table=getattr(fk, "from_table", ""),
                    from_column=getattr(fk, "from_column", ""),
                    to_table=getattr(fk, "to_table", ""),
                    to_column=getattr(fk, "to_column", ""),
                    confidence=getattr(fk, "confidence", 0.5),
                    relationship_type=getattr(fk, "relationship_type", "foreign_key"),
                )

            self.foreign_keys.append(fk_meta)

            # 인덱스 업데이트
            if fk_meta.from_table not in self._fk_by_table:
                self._fk_by_table[fk_meta.from_table] = []
            self._fk_by_table[fk_meta.from_table].append(fk_meta)

            if fk_meta.to_table not in self._fk_to_table:
                self._fk_to_table[fk_meta.to_table] = []
            self._fk_to_table[fk_meta.to_table].append(fk_meta)

            # 테이블 메타데이터에 FK 표시
            if fk_meta.from_table in self.tables:
                col = self.tables[fk_meta.from_table].get_column(fk_meta.from_column)
                if col:
                    col.is_foreign_key = True
                    col.references_table = fk_meta.to_table
                    col.references_column = fk_meta.to_column

    def add_table(self, table: TableMeta) -> None:
        """테이블 추가"""
        self.tables[table.name] = table

    def add_foreign_key(self, fk: ForeignKeyMeta) -> None:
        """FK 관계 추가"""
        self.foreign_keys.append(fk)

        if fk.from_table not in self._fk_by_table:
            self._fk_by_table[fk.from_table] = []
        self._fk_by_table[fk.from_table].append(fk)

        if fk.to_table not in self._fk_to_table:
            self._fk_to_table[fk.to_table] = []
        self._fk_to_table[fk.to_table].append(fk)

    def get_table(self, name: str) -> Optional[TableMeta]:
        """테이블 조회"""
        return self.tables.get(name)

    def get_related_tables(self, table_name: str) -> List[str]:
        """관련 테이블 목록 (FK로 연결된)"""
        related = set()

        # Outgoing FKs
        for fk in self._fk_by_table.get(table_name, []):
            related.add(fk.to_table)

        # Incoming FKs
        for fk in self._fk_to_table.get(table_name, []):
            related.add(fk.from_table)

        return list(related)

    def get_join_condition(self, table_a: str, table_b: str) -> Optional[str]:
        """
        두 테이블 간 조인 조건 반환

        Returns:
            "tableA.col = tableB.col" 형식, 없으면 None
        """
        # A → B 방향
        for fk in self._fk_by_table.get(table_a, []):
            if fk.to_table == table_b:
                return f"{fk.from_table}.{fk.from_column} = {fk.to_table}.{fk.to_column}"

        # B → A 방향
        for fk in self._fk_by_table.get(table_b, []):
            if fk.to_table == table_a:
                return f"{fk.from_table}.{fk.from_column} = {fk.to_table}.{fk.to_column}"

        return None

    def to_llm_string(self, max_tables: int = 20) -> str:
        """
        LLM 친화적 전체 스키마 문자열

        Args:
            max_tables: 포함할 최대 테이블 수
        """
        lines = ["=== DATABASE SCHEMA ===\n"]

        # 테이블 정보
        for i, (name, table) in enumerate(self.tables.items()):
            if i >= max_tables:
                lines.append(f"\n... and {len(self.tables) - max_tables} more tables")
                break
            lines.append(table.to_llm_string())
            lines.append("")

        # FK 관계
        lines.append("\n=== FOREIGN KEY RELATIONSHIPS ===\n")
        for fk in self.foreign_keys[:50]:  # 최대 50개
            lines.append(
                f"  {fk.from_table}.{fk.from_column} → "
                f"{fk.to_table}.{fk.to_column} "
                f"(confidence: {fk.confidence:.0%})"
            )

        return "\n".join(lines)

    def to_json(self) -> str:
        """JSON 형식으로 변환"""
        return json.dumps({
            "tables": {name: table.to_dict() for name, table in self.tables.items()},
            "foreign_keys": [fk.to_dict() for fk in self.foreign_keys],
        }, indent=2, ensure_ascii=False)

    def to_compact_json(self) -> str:
        """컴팩트한 JSON (LLM 컨텍스트 최적화)"""
        compact = {
            "tables": {},
            "fks": []
        }

        for name, table in self.tables.items():
            compact["tables"][name] = {
                "cols": [
                    {
                        "n": c.name,
                        "t": c.data_type,
                        "pk": c.is_primary_key,
                        "fk": c.is_foreign_key,
                    }
                    for c in table.columns
                ],
                "entity": table.entity_name,
            }

        for fk in self.foreign_keys:
            compact["fks"].append({
                "from": f"{fk.from_table}.{fk.from_column}",
                "to": f"{fk.to_table}.{fk.to_column}",
            })

        return json.dumps(compact, ensure_ascii=False)

    def get_table_names(self) -> List[str]:
        """테이블 이름 목록"""
        return list(self.tables.keys())

    def get_column_names(self, table_name: str) -> List[str]:
        """특정 테이블의 컬럼 이름 목록"""
        table = self.tables.get(table_name)
        if table:
            return [c.name for c in table.columns]
        return []
