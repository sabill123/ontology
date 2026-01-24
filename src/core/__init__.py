"""
Core Infrastructure for Ontology OS

Enterprise-grade components:
- Knowledge Graph Engine (NetworkX-based)
- LangGraph StateGraph Agent Orchestration
- Graph Reasoning (Multi-hop, Pattern Matching, Causal Inference)
- Community Detection (GraphRAG/Louvain style)
"""

from .knowledge_graph import (
    KnowledgeGraph,
    KGEntity,
    KGRelation,
    KGCommunity,
    KnowledgeGraphBuilder,
    EntityType,
    RelationType,
)

from .graph_state import (
    OntologyGraphState,
    OntologyOrchestrator,
    PhaseStatus,
    AgentType,
    AgentMessage,
    create_initial_state,
    build_ontology_graph,
)

from .graph_reasoning import (
    GraphReasoningAgent,
    ReasoningQuery,
    ReasoningResult,
    ReasoningType,
)

__all__ = [
    # Knowledge Graph
    "KnowledgeGraph",
    "KGEntity",
    "KGRelation",
    "KGCommunity",
    "KnowledgeGraphBuilder",
    "EntityType",
    "RelationType",
    # LangGraph Orchestration
    "OntologyGraphState",
    "OntologyOrchestrator",
    "PhaseStatus",
    "AgentType",
    "AgentMessage",
    "create_initial_state",
    "build_ontology_graph",
    # Graph Reasoning
    "GraphReasoningAgent",
    "ReasoningQuery",
    "ReasoningResult",
    "ReasoningType",
]
