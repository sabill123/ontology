"""
Ontology Registry

Ontology concepts, concept relationships, and knowledge-graph triples.
Extracted from SharedContext to reduce God Object complexity.
"""

from typing import Dict, List, Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .shared_context import OntologyConcept


class OntologyRegistry:
    """
    Ontology concept and knowledge-graph triple management.

    Manages:
    - ontology_concepts
    - concept_relationships
    - knowledge_graph_triples  (OntologyArchitect asserted)
    - semantic_base_triples    (SemanticValidator base)
    - inferred_triples         (reasoned)
    - governance_triples       (GovernanceStrategist)
    """

    def __init__(self) -> None:
        self.ontology_concepts: List["OntologyConcept"] = []
        self.concept_relationships: List[Dict[str, Any]] = []
        self.knowledge_graph_triples: List[Dict[str, Any]] = []
        self.semantic_base_triples: List[Dict[str, Any]] = []
        self.inferred_triples: List[Dict[str, Any]] = []
        self.governance_triples: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Bind helpers
    # ------------------------------------------------------------------

    def bind(
        self,
        ontology_concepts: List["OntologyConcept"],
        concept_relationships: List[Dict[str, Any]],
        knowledge_graph_triples: List[Dict[str, Any]],
        semantic_base_triples: List[Dict[str, Any]],
        inferred_triples: List[Dict[str, Any]],
        governance_triples: List[Dict[str, Any]],
    ) -> None:
        """Bind to the dataclass-owned containers."""
        self.ontology_concepts = ontology_concepts
        self.concept_relationships = concept_relationships
        self.knowledge_graph_triples = knowledge_graph_triples
        self.semantic_base_triples = semantic_base_triples
        self.inferred_triples = inferred_triples
        self.governance_triples = governance_triples

    # ------------------------------------------------------------------
    # Concept management
    # ------------------------------------------------------------------

    def add_concept(self, concept: "OntologyConcept") -> None:
        self.ontology_concepts.append(concept)

    def get_concepts_by_type(self, concept_type: str) -> List["OntologyConcept"]:
        return [c for c in self.ontology_concepts if c.concept_type == concept_type]

    def get_approved_concepts(self) -> List["OntologyConcept"]:
        return [c for c in self.ontology_concepts if c.status == "approved"]

    # ------------------------------------------------------------------
    # Knowledge-graph triples
    # ------------------------------------------------------------------

    def add_triple(
        self,
        subject: str,
        predicate: str,
        obj: str,
        confidence: float = 1.0,
        source: str = "asserted",
    ) -> None:
        self.knowledge_graph_triples.append({
            "subject": subject,
            "predicate": predicate,
            "object": obj,
            "confidence": confidence,
            "source": source,
        })

    def add_inferred_triple(
        self,
        subject: str,
        predicate: str,
        obj: str,
        confidence: float = 1.0,
        rule_applied: Optional[str] = None,
    ) -> None:
        self.inferred_triples.append({
            "subject": subject,
            "predicate": predicate,
            "object": obj,
            "confidence": confidence,
            "source": "inferred",
            "rule_applied": rule_applied,
        })

    def get_all_triples(self) -> List[Dict[str, Any]]:
        return (
            self.knowledge_graph_triples
            + self.semantic_base_triples
            + self.inferred_triples
            + self.governance_triples
        )

    def get_triple_count(self) -> Dict[str, int]:
        return {
            "ontology_asserted": len(self.knowledge_graph_triples),
            "semantic_base": len(self.semantic_base_triples),
            "inferred": len(self.inferred_triples),
            "governance": len(self.governance_triples),
            "total": (
                len(self.knowledge_graph_triples)
                + len(self.semantic_base_triples)
                + len(self.inferred_triples)
                + len(self.governance_triples)
            ),
        }

    def build_knowledge_graph(self, context: Any) -> Any:
        """
        Build full knowledge graph using KnowledgeGraphBuilder.

        Args:
            context: SharedContext instance (needed by builder).
        """
        from .autonomous.analysis.knowledge_graph import KnowledgeGraphBuilder
        builder = KnowledgeGraphBuilder()
        return builder.build_from_context(context)
