"""
LangGraph StateGraph-based Agent Orchestrator

Microsoft GraphRAG + LangGraph 아키텍처를 적용한 에이전트 오케스트레이션.
각 Phase에서 공유 상태를 통해 Knowledge Graph를 점진적으로 구축.
"""

from typing import TypedDict, Annotated, Sequence, Optional, Any, Dict, List
from enum import Enum
from dataclasses import dataclass, field, asdict
from datetime import datetime
import json
import operator

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from src.core.knowledge_graph import (
    KnowledgeGraph,
    KnowledgeGraphBuilder,
    KGEntity,
    KGRelation,
    EntityType,
    RelationType
)


class PhaseStatus(str, Enum):
    """Phase execution status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class AgentType(str, Enum):
    """Agent types in the system."""
    SCHEMA_ANALYZER = "schema_analyzer"
    DATA_PROFILER = "data_profiler"
    ENTITY_EXTRACTOR = "entity_extractor"
    RELATION_DISCOVERER = "relation_discoverer"
    TOPOLOGY_VALIDATOR = "topology_validator"
    INSIGHT_GENERATOR = "insight_generator"
    GRAPH_REASONER = "graph_reasoner"
    GOVERNANCE_EVALUATOR = "governance_evaluator"
    ACTION_PLANNER = "action_planner"
    LLM_JUDGE = "llm_judge"


@dataclass
class AgentMessage:
    """Message from an agent."""
    agent_type: AgentType
    content: str
    data: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    confidence: float = 0.0


@dataclass
class PhaseResult:
    """Result from a phase execution."""
    phase: str
    status: PhaseStatus
    outputs: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, float] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    duration_ms: float = 0.0


class OntologyGraphState(TypedDict):
    """
    Shared state across all phases.

    LangGraph의 StateGraph에서 사용하는 공유 상태.
    모든 노드(에이전트)가 이 상태를 읽고 수정할 수 있음.
    """
    # === Scenario & Session Info ===
    scenario_name: str
    session_id: str
    timestamp: str

    # === Knowledge Graph (핵심) ===
    knowledge_graph: Optional[Dict[str, Any]]  # Serialized KG

    # === Phase Status ===
    phase1_status: PhaseStatus
    phase2_status: PhaseStatus
    phase3_status: PhaseStatus
    current_phase: str

    # === Phase 1 Outputs ===
    data_sources: List[Dict[str, Any]]
    schema_profiles: List[Dict[str, Any]]
    homeomorphic_mappings: List[Dict[str, Any]]
    canonical_objects: List[str]
    topology_score: float

    # === Phase 1 Data Analysis ===
    data_profiles: List[Dict[str, Any]]
    correlations: List[Dict[str, Any]]
    anomalies: List[Dict[str, Any]]
    business_metrics: Dict[str, Any]

    # === Phase 2 Outputs ===
    insights: List[Dict[str, Any]]
    insight_evidence: List[Dict[str, Any]]
    graph_reasoning_results: List[Dict[str, Any]]
    community_summaries: List[Dict[str, Any]]

    # === Phase 3 Outputs ===
    governance_decisions: List[Dict[str, Any]]
    actions: List[Dict[str, Any]]
    kpi_predictions: Dict[str, Any]

    # === Agent Messages (Append-only) ===
    messages: Annotated[Sequence[Dict[str, Any]], operator.add]

    # === Validation Scores ===
    llm_judge_scores: Dict[str, float]
    overall_confidence: float

    # === Errors ===
    errors: List[str]


def create_initial_state(scenario_name: str, session_id: str) -> OntologyGraphState:
    """Create initial state for a new session."""
    return OntologyGraphState(
        scenario_name=scenario_name,
        session_id=session_id,
        timestamp=datetime.now().isoformat(),

        knowledge_graph=None,

        phase1_status=PhaseStatus.PENDING,
        phase2_status=PhaseStatus.PENDING,
        phase3_status=PhaseStatus.PENDING,
        current_phase="",

        data_sources=[],
        schema_profiles=[],
        homeomorphic_mappings=[],
        canonical_objects=[],
        topology_score=0.0,

        data_profiles=[],
        correlations=[],
        anomalies=[],
        business_metrics={},

        insights=[],
        insight_evidence=[],
        graph_reasoning_results=[],
        community_summaries=[],

        governance_decisions=[],
        actions=[],
        kpi_predictions={},

        messages=[],

        llm_judge_scores={},
        overall_confidence=0.0,

        errors=[]
    )


# ==============================================================================
# Phase 1 Nodes
# ==============================================================================

def schema_analysis_node(state: OntologyGraphState) -> Dict[str, Any]:
    """
    스키마 분석 에이전트 노드.

    데이터 소스의 스키마를 분석하고 프로파일링 결과를 상태에 추가.
    """
    message = AgentMessage(
        agent_type=AgentType.SCHEMA_ANALYZER,
        content="Analyzing data source schemas",
        data={"sources_count": len(state.get("data_sources", []))},
        confidence=0.9
    )

    return {
        "messages": [asdict(message)],
        "current_phase": "phase1"
    }


def entity_extraction_node(state: OntologyGraphState) -> Dict[str, Any]:
    """
    엔티티 추출 에이전트 노드.

    데이터에서 엔티티를 추출하고 Knowledge Graph에 추가.
    """
    # Initialize or load KG
    kg_data = state.get("knowledge_graph")
    if kg_data:
        kg = KnowledgeGraph.from_dict(kg_data)
    else:
        kg = KnowledgeGraph(name=f"KG_{state['scenario_name']}")

    # Extract entities from schema profiles
    entities_added = 0
    for profile in state.get("schema_profiles", []):
        table_name = profile.get("table_name", "unknown")

        # Add table as entity
        table_entity = KGEntity(
            id=f"table_{table_name}",
            name=table_name,
            entity_type=EntityType.TABLE,
            properties={"columns": profile.get("columns", [])},
            source=profile.get("source_file", "")
        )
        kg.add_entity(table_entity)
        entities_added += 1

        # Add columns as entities
        for col in profile.get("columns", []):
            col_entity = KGEntity(
                id=f"col_{table_name}_{col['name']}",
                name=col["name"],
                entity_type=EntityType.COLUMN,
                properties={
                    "data_type": col.get("dtype", "unknown"),
                    "nullable": col.get("nullable", True)
                },
                source=table_name
            )
            kg.add_entity(col_entity)
            entities_added += 1

            # Add relation: table HAS column
            rel = KGRelation(
                source_id=table_entity.id,
                target_id=col_entity.id,
                relation_type=RelationType.HAS_ATTRIBUTE,
                properties={"position": col.get("position", 0)}
            )
            kg.add_relation(rel)

    message = AgentMessage(
        agent_type=AgentType.ENTITY_EXTRACTOR,
        content=f"Extracted {entities_added} entities from schemas",
        data={"entities_added": entities_added},
        confidence=0.85
    )

    return {
        "knowledge_graph": kg.to_dict(),
        "messages": [asdict(message)]
    }


def relation_discovery_node(state: OntologyGraphState) -> Dict[str, Any]:
    """
    관계 발견 에이전트 노드.

    FK 관계, 의미적 관계를 발견하고 Knowledge Graph에 추가.
    """
    kg_data = state.get("knowledge_graph")
    if not kg_data:
        return {"errors": state.get("errors", []) + ["KG not initialized"]}

    kg = KnowledgeGraph.from_dict(kg_data)
    relations_added = 0

    # Process homeomorphic mappings as relations
    for mapping in state.get("homeomorphic_mappings", []):
        source_table = mapping.get("source_a", "")
        target_table = mapping.get("source_b", "")

        if source_table and target_table:
            rel = KGRelation(
                source_id=f"table_{source_table}",
                target_id=f"table_{target_table}",
                relation_type=RelationType.HOMEOMORPHIC_TO,
                confidence=mapping.get("confidence", 0.8),
                properties={
                    "mapping_type": mapping.get("mapping_type", ""),
                    "betti_match": mapping.get("betti_match", False),
                    "tda_distance": mapping.get("tda_distance", 0.0)
                }
            )
            kg.add_relation(rel)
            relations_added += 1

    message = AgentMessage(
        agent_type=AgentType.RELATION_DISCOVERER,
        content=f"Discovered {relations_added} relations",
        data={"relations_added": relations_added},
        confidence=0.8
    )

    return {
        "knowledge_graph": kg.to_dict(),
        "messages": [asdict(message)]
    }


def topology_validation_node(state: OntologyGraphState) -> Dict[str, Any]:
    """
    위상 검증 에이전트 노드.

    TDA 기반으로 위상 구조를 검증.
    """
    kg_data = state.get("knowledge_graph")
    if not kg_data:
        return {"errors": state.get("errors", []) + ["KG not initialized"]}

    kg = KnowledgeGraph.from_dict(kg_data)

    # Calculate topology metrics
    stats = kg.get_stats()

    # Simple topology score based on graph connectivity
    entity_count = stats["entity_count"]
    relation_count = stats["relation_count"]

    if entity_count > 0:
        connectivity = min(1.0, relation_count / (entity_count * 2))
    else:
        connectivity = 0.0

    topology_score = connectivity * 0.5 + 0.5  # Base score + connectivity bonus

    message = AgentMessage(
        agent_type=AgentType.TOPOLOGY_VALIDATOR,
        content=f"Topology validation: score={topology_score:.2f}",
        data={
            "topology_score": topology_score,
            "entity_count": entity_count,
            "relation_count": relation_count
        },
        confidence=topology_score
    )

    return {
        "topology_score": topology_score,
        "messages": [asdict(message)],
        "phase1_status": PhaseStatus.COMPLETED
    }


# ==============================================================================
# Phase 2 Nodes
# ==============================================================================

def graph_reasoning_node(state: OntologyGraphState) -> Dict[str, Any]:
    """
    그래프 추론 에이전트 노드.

    Knowledge Graph에서 multi-hop 추론을 수행하여 인사이트 발견.
    """
    kg_data = state.get("knowledge_graph")
    if not kg_data:
        return {"errors": state.get("errors", []) + ["KG not initialized"]}

    kg = KnowledgeGraph.from_dict(kg_data)
    reasoning_results = []

    # 1. Find connected components (potential data silos)
    components = []
    visited = set()

    for entity_id in kg.entities:
        if entity_id not in visited:
            component = set()
            stack = [entity_id]
            while stack:
                current = stack.pop()
                if current not in visited:
                    visited.add(current)
                    component.add(current)
                    neighbors = kg.get_neighbors(current)
                    stack.extend([n for n in neighbors if n not in visited])
            components.append(list(component))

    reasoning_results.append({
        "type": "connected_components",
        "count": len(components),
        "sizes": [len(c) for c in components],
        "insight": f"Found {len(components)} data silo clusters"
    })

    # 2. Find bridge entities (entities connecting different clusters)
    bridge_entities = []
    for entity_id in kg.entities:
        neighbors = kg.get_neighbors(entity_id)
        neighbor_components = set()
        for neighbor in neighbors:
            for i, comp in enumerate(components):
                if neighbor in comp:
                    neighbor_components.add(i)
        if len(neighbor_components) > 1:
            bridge_entities.append({
                "entity_id": entity_id,
                "connects_clusters": list(neighbor_components)
            })

    if bridge_entities:
        reasoning_results.append({
            "type": "bridge_entities",
            "entities": bridge_entities,
            "insight": f"Found {len(bridge_entities)} bridge entities connecting data silos"
        })

    # 3. Detect potential data quality issues from anomalies
    anomalies = state.get("anomalies", [])
    if anomalies:
        reasoning_results.append({
            "type": "anomaly_patterns",
            "count": len(anomalies),
            "insight": f"Detected {len(anomalies)} potential data quality issues"
        })

    message = AgentMessage(
        agent_type=AgentType.GRAPH_REASONER,
        content=f"Graph reasoning completed with {len(reasoning_results)} findings",
        data={"results_count": len(reasoning_results)},
        confidence=0.75
    )

    return {
        "graph_reasoning_results": reasoning_results,
        "messages": [asdict(message)],
        "current_phase": "phase2"
    }


def community_detection_node(state: OntologyGraphState) -> Dict[str, Any]:
    """
    커뮤니티 탐지 에이전트 노드.

    GraphRAG 스타일의 커뮤니티 탐지 및 요약.
    """
    kg_data = state.get("knowledge_graph")
    if not kg_data:
        return {"errors": state.get("errors", []) + ["KG not initialized"]}

    kg = KnowledgeGraph.from_dict(kg_data)

    # Detect communities
    communities = kg.detect_communities()

    # Create community summaries (without LLM for now - can be enhanced)
    summaries = []
    for community in communities:
        entity_types = {}
        for entity_id in community.entity_ids:
            entity = kg.entities.get(entity_id)
            if entity:
                etype = entity.entity_type.value
                entity_types[etype] = entity_types.get(etype, 0) + 1

        summary = {
            "community_id": community.id,
            "size": community.size,
            "entity_type_distribution": entity_types,
            "summary": community.summary or f"Community with {community.size} entities"
        }
        summaries.append(summary)

    message = AgentMessage(
        agent_type=AgentType.GRAPH_REASONER,
        content=f"Detected {len(communities)} communities",
        data={"community_count": len(communities)},
        confidence=0.8
    )

    return {
        "knowledge_graph": kg.to_dict(),  # Updated with communities
        "community_summaries": summaries,
        "messages": [asdict(message)]
    }


def insight_generation_node(state: OntologyGraphState) -> Dict[str, Any]:
    """
    인사이트 생성 에이전트 노드.

    그래프 추론 결과와 데이터 분석 결과를 종합하여 인사이트 생성.
    """
    insights = []

    # Combine graph reasoning with data analysis
    reasoning_results = state.get("graph_reasoning_results", [])
    community_summaries = state.get("community_summaries", [])
    anomalies = state.get("anomalies", [])
    correlations = state.get("correlations", [])

    # Generate insights from graph reasoning
    for result in reasoning_results:
        insight = {
            "id": f"INS-GR-{len(insights)+1:03d}",
            "type": "graph_reasoning",
            "source": result.get("type"),
            "description": result.get("insight", ""),
            "confidence": 0.75,
            "evidence": result
        }
        insights.append(insight)

    # Generate insights from communities
    for comm in community_summaries:
        if comm.get("size", 0) > 3:  # Significant communities
            insight = {
                "id": f"INS-CM-{len(insights)+1:03d}",
                "type": "community_pattern",
                "source": "community_detection",
                "description": f"Identified data cluster: {comm.get('summary')}",
                "confidence": 0.8,
                "evidence": comm
            }
            insights.append(insight)

    # Generate insights from correlations
    for corr in correlations:
        if abs(corr.get("correlation", 0)) > 0.7:  # Strong correlations
            insight = {
                "id": f"INS-CR-{len(insights)+1:03d}",
                "type": "correlation",
                "source": "data_analysis",
                "description": f"Strong correlation found: {corr.get('description', '')}",
                "confidence": abs(corr.get("correlation", 0)),
                "evidence": corr
            }
            insights.append(insight)

    # Generate insights from anomalies
    for anomaly in anomalies:
        if anomaly.get("severity", "low") in ["high", "critical"]:
            insight = {
                "id": f"INS-AN-{len(insights)+1:03d}",
                "type": "anomaly",
                "source": "anomaly_detection",
                "description": anomaly.get("description", "Anomaly detected"),
                "confidence": 0.85,
                "severity": anomaly.get("severity"),
                "evidence": anomaly
            }
            insights.append(insight)

    message = AgentMessage(
        agent_type=AgentType.INSIGHT_GENERATOR,
        content=f"Generated {len(insights)} insights",
        data={"insight_count": len(insights)},
        confidence=0.8
    )

    return {
        "insights": insights,
        "messages": [asdict(message)],
        "phase2_status": PhaseStatus.COMPLETED
    }


# ==============================================================================
# Phase 3 Nodes
# ==============================================================================

def governance_evaluation_node(state: OntologyGraphState) -> Dict[str, Any]:
    """
    거버넌스 평가 에이전트 노드.

    인사이트를 평가하고 승인/거부/에스컬레이션 결정.
    """
    insights = state.get("insights", [])
    decisions = []

    for insight in insights:
        confidence = insight.get("confidence", 0.5)
        severity = insight.get("severity", "medium")

        # Decision logic
        if confidence >= 0.8:
            decision_type = "approve"
        elif confidence >= 0.6:
            decision_type = "schedule_review"
        elif severity in ["high", "critical"]:
            decision_type = "escalate"
        else:
            decision_type = "reject"

        decision = {
            "decision_id": f"DEC-{len(decisions)+1:03d}",
            "insight_id": insight.get("id"),
            "decision_type": decision_type,
            "confidence": confidence,
            "reasoning": f"Based on confidence {confidence:.2f} and severity {severity}",
            "timestamp": datetime.now().isoformat()
        }
        decisions.append(decision)

    message = AgentMessage(
        agent_type=AgentType.GOVERNANCE_EVALUATOR,
        content=f"Evaluated {len(decisions)} insights",
        data={
            "approved": len([d for d in decisions if d["decision_type"] == "approve"]),
            "escalated": len([d for d in decisions if d["decision_type"] == "escalate"]),
            "rejected": len([d for d in decisions if d["decision_type"] == "reject"])
        },
        confidence=0.85
    )

    return {
        "governance_decisions": decisions,
        "messages": [asdict(message)],
        "current_phase": "phase3"
    }


def action_planning_node(state: OntologyGraphState) -> Dict[str, Any]:
    """
    액션 계획 에이전트 노드.

    승인된 인사이트에 대한 실행 계획 수립.
    """
    decisions = state.get("governance_decisions", [])
    insights = {i.get("id"): i for i in state.get("insights", [])}
    actions = []

    for decision in decisions:
        if decision.get("decision_type") == "approve":
            insight_id = decision.get("insight_id")
            insight = insights.get(insight_id, {})

            action = {
                "action_id": f"ACT-{len(actions)+1:03d}",
                "insight_id": insight_id,
                "decision_id": decision.get("decision_id"),
                "title": f"Action for: {insight.get('description', '')[:50]}",
                "description": f"Implement changes based on {insight.get('type')} insight",
                "priority": "high" if insight.get("severity") in ["high", "critical"] else "medium",
                "status": "backlog",
                "estimated_impact": {
                    "confidence": decision.get("confidence", 0.5),
                    "insight_type": insight.get("type")
                }
            }
            actions.append(action)

    message = AgentMessage(
        agent_type=AgentType.ACTION_PLANNER,
        content=f"Created {len(actions)} action items",
        data={"action_count": len(actions)},
        confidence=0.8
    )

    return {
        "actions": actions,
        "messages": [asdict(message)],
        "phase3_status": PhaseStatus.COMPLETED
    }


def llm_judge_node(state: OntologyGraphState) -> Dict[str, Any]:
    """
    LLM Judge 에이전트 노드.

    전체 파이프라인 결과를 평가.
    """
    # Calculate scores based on state
    insights = state.get("insights", [])
    actions = state.get("actions", [])
    kg_data = state.get("knowledge_graph", {})

    scores = {
        "topology_fidelity": state.get("topology_score", 0.0),
        "insight_coverage": min(1.0, len(insights) / 10) if insights else 0.0,
        "action_alignment": min(1.0, len(actions) / len(insights)) if insights else 0.0,
        "graph_completeness": 0.8 if kg_data else 0.0,
        "evidence_quality": sum(i.get("confidence", 0) for i in insights) / len(insights) if insights else 0.0
    }

    overall_confidence = sum(scores.values()) / len(scores) if scores else 0.0

    message = AgentMessage(
        agent_type=AgentType.LLM_JUDGE,
        content=f"Pipeline evaluation complete: {overall_confidence:.2f}",
        data=scores,
        confidence=overall_confidence
    )

    return {
        "llm_judge_scores": scores,
        "overall_confidence": overall_confidence,
        "messages": [asdict(message)]
    }


# ==============================================================================
# Graph Builder
# ==============================================================================

def build_ontology_graph() -> StateGraph:
    """
    Build the complete LangGraph StateGraph for the Ontology pipeline.

    Returns:
        Configured StateGraph
    """
    graph = StateGraph(OntologyGraphState)

    # Add Phase 1 nodes
    graph.add_node("schema_analysis", schema_analysis_node)
    graph.add_node("entity_extraction", entity_extraction_node)
    graph.add_node("relation_discovery", relation_discovery_node)
    graph.add_node("topology_validation", topology_validation_node)

    # Add Phase 2 nodes
    graph.add_node("graph_reasoning", graph_reasoning_node)
    graph.add_node("community_detection", community_detection_node)
    graph.add_node("insight_generation", insight_generation_node)

    # Add Phase 3 nodes
    graph.add_node("governance_evaluation", governance_evaluation_node)
    graph.add_node("action_planning", action_planning_node)
    graph.add_node("llm_judge", llm_judge_node)

    # Define edges for Phase 1
    graph.set_entry_point("schema_analysis")
    graph.add_edge("schema_analysis", "entity_extraction")
    graph.add_edge("entity_extraction", "relation_discovery")
    graph.add_edge("relation_discovery", "topology_validation")

    # Define edges for Phase 2
    graph.add_edge("topology_validation", "graph_reasoning")
    graph.add_edge("graph_reasoning", "community_detection")
    graph.add_edge("community_detection", "insight_generation")

    # Define edges for Phase 3
    graph.add_edge("insight_generation", "governance_evaluation")
    graph.add_edge("governance_evaluation", "action_planning")
    graph.add_edge("action_planning", "llm_judge")

    # End
    graph.add_edge("llm_judge", END)

    return graph


class OntologyOrchestrator:
    """
    Orchestrator for the Ontology pipeline using LangGraph.

    Manages the complete Phase 1 → 2 → 3 pipeline with:
    - Shared state management
    - Checkpointing for resume
    - Knowledge Graph accumulation
    """

    def __init__(self, scenario_name: str, enable_checkpointing: bool = True):
        """
        Initialize orchestrator.

        Args:
            scenario_name: Name of the scenario
            enable_checkpointing: Enable state checkpointing
        """
        self.scenario_name = scenario_name
        self.graph = build_ontology_graph()

        if enable_checkpointing:
            self.memory = MemorySaver()
            self.app = self.graph.compile(checkpointer=self.memory)
        else:
            self.app = self.graph.compile()

        self.session_id = f"{scenario_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    def run_full_pipeline(
        self,
        data_sources: List[Dict[str, Any]],
        schema_profiles: List[Dict[str, Any]],
        homeomorphic_mappings: List[Dict[str, Any]],
        data_analysis_results: Optional[Dict[str, Any]] = None
    ) -> OntologyGraphState:
        """
        Run the complete pipeline.

        Args:
            data_sources: List of data source info
            schema_profiles: List of schema profiles
            homeomorphic_mappings: List of homeomorphic mappings
            data_analysis_results: Optional data analysis results

        Returns:
            Final state after pipeline completion
        """
        # Initialize state
        initial_state = create_initial_state(self.scenario_name, self.session_id)

        # Add input data
        initial_state["data_sources"] = data_sources
        initial_state["schema_profiles"] = schema_profiles
        initial_state["homeomorphic_mappings"] = homeomorphic_mappings

        # Add data analysis results if provided
        if data_analysis_results:
            initial_state["data_profiles"] = data_analysis_results.get("profiles", [])
            initial_state["correlations"] = data_analysis_results.get("correlations", [])
            initial_state["anomalies"] = data_analysis_results.get("anomalies", [])
            initial_state["business_metrics"] = data_analysis_results.get("business_metrics", {})

        # Run pipeline
        config = {"configurable": {"thread_id": self.session_id}}
        final_state = self.app.invoke(initial_state, config)

        return final_state

    def run_phase1_only(
        self,
        data_sources: List[Dict[str, Any]],
        schema_profiles: List[Dict[str, Any]],
        homeomorphic_mappings: List[Dict[str, Any]]
    ) -> OntologyGraphState:
        """Run only Phase 1 nodes."""
        initial_state = create_initial_state(self.scenario_name, self.session_id)
        initial_state["data_sources"] = data_sources
        initial_state["schema_profiles"] = schema_profiles
        initial_state["homeomorphic_mappings"] = homeomorphic_mappings

        # Run until topology_validation
        config = {"configurable": {"thread_id": self.session_id}}

        # Execute nodes sequentially
        state = initial_state
        for node_name in ["schema_analysis", "entity_extraction", "relation_discovery", "topology_validation"]:
            node_fn = self.graph.nodes[node_name]
            updates = node_fn(state)
            state = {**state, **updates}

        return state

    def get_knowledge_graph(self, state: OntologyGraphState) -> Optional[KnowledgeGraph]:
        """Extract Knowledge Graph from state."""
        kg_data = state.get("knowledge_graph")
        if kg_data:
            return KnowledgeGraph.from_dict(kg_data)
        return None

    def get_summary(self, state: OntologyGraphState) -> Dict[str, Any]:
        """Get summary of pipeline execution."""
        return {
            "scenario": state.get("scenario_name"),
            "session_id": state.get("session_id"),
            "phases": {
                "phase1": state.get("phase1_status", PhaseStatus.PENDING).value,
                "phase2": state.get("phase2_status", PhaseStatus.PENDING).value,
                "phase3": state.get("phase3_status", PhaseStatus.PENDING).value
            },
            "metrics": {
                "topology_score": state.get("topology_score", 0.0),
                "insights_count": len(state.get("insights", [])),
                "actions_count": len(state.get("actions", [])),
                "overall_confidence": state.get("overall_confidence", 0.0)
            },
            "llm_judge_scores": state.get("llm_judge_scores", {}),
            "message_count": len(state.get("messages", [])),
            "errors": state.get("errors", [])
        }
