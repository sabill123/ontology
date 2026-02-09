"""
Entity Registry

Unified entities, homeomorphism pairs, extracted entities,
FK candidates, value overlaps, cross-table mappings.
Extracted from SharedContext to reduce God Object complexity.
"""

from typing import Dict, List, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from .shared_context import UnifiedEntity, HomeomorphismPair


class EntityRegistry:
    """
    Entity lifecycle management.

    Manages:
    - unified_entities
    - homeomorphisms
    - extracted_entities
    - enhanced_fk_candidates
    - value_overlaps
    - indirect_fks
    - cross_table_mappings
    - schema_analysis
    - tda_signatures
    """

    def __init__(self) -> None:
        self.unified_entities: List["UnifiedEntity"] = []
        self.homeomorphisms: List["HomeomorphismPair"] = []
        self.extracted_entities: List[Dict[str, Any]] = []
        self.enhanced_fk_candidates: List[Dict[str, Any]] = []
        self.value_overlaps: List[Dict[str, Any]] = []
        self.indirect_fks: List[Dict[str, Any]] = []
        self.cross_table_mappings: List[Dict[str, Any]] = []
        self.schema_analysis: Dict[str, Any] = {}
        self.tda_signatures: Dict[str, Dict[str, Any]] = {}

    # ------------------------------------------------------------------
    # Bind helpers
    # ------------------------------------------------------------------

    def bind(
        self,
        unified_entities: List["UnifiedEntity"],
        homeomorphisms: List["HomeomorphismPair"],
        extracted_entities: List[Dict[str, Any]],
        enhanced_fk_candidates: List[Dict[str, Any]],
        value_overlaps: List[Dict[str, Any]],
        indirect_fks: List[Dict[str, Any]],
        cross_table_mappings: List[Dict[str, Any]],
        schema_analysis: Dict[str, Any],
        tda_signatures: Dict[str, Dict[str, Any]],
    ) -> None:
        """Bind to the dataclass-owned containers."""
        self.unified_entities = unified_entities
        self.homeomorphisms = homeomorphisms
        self.extracted_entities = extracted_entities
        self.enhanced_fk_candidates = enhanced_fk_candidates
        self.value_overlaps = value_overlaps
        self.indirect_fks = indirect_fks
        self.cross_table_mappings = cross_table_mappings
        self.schema_analysis = schema_analysis
        self.tda_signatures = tda_signatures

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_unified_entity(self, entity: "UnifiedEntity") -> None:
        self.unified_entities.append(entity)

    def add_homeomorphism(self, pair: "HomeomorphismPair") -> None:
        self.homeomorphisms.append(pair)

    def get_tables_for_entity(self, entity_type: str) -> List[str]:
        tables: List[str] = []
        for entity in self.unified_entities:
            if entity.entity_type == entity_type:
                tables.extend(entity.source_tables)
        return list(set(tables))

    def get_homeomorphisms_for_table(self, table_name: str) -> List["HomeomorphismPair"]:
        return [
            h for h in self.homeomorphisms
            if h.table_a == table_name or h.table_b == table_name
        ]
