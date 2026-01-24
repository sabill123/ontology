"""
Standard Data Schemas for Phase 1 → 2 → 3 Pipeline

Defines the canonical data structures that ensure continuity across all phases.
"""

from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional
from datetime import datetime
from enum import Enum
import json


# ============================================================================
# PHASE 1: Topology & Silo Integration
# ============================================================================

@dataclass
class HomeomorphicMapping:
    """A homeomorphic mapping between two data sources."""
    source_a: str  # e.g., "sap_mrp.production_orders"
    source_b: str  # e.g., "mes_quality.line_quality_events"
    mapping_type: str  # e.g., "FK_MATCH", "SEMANTIC_EQUIV"
    confidence: float
    evidence: List[str]
    canonical_object: str  # e.g., "WorkcellTopology"
    
    # Topology evidence
    betti_numbers_match: bool
    tda_distance: float
    
    # Column mappings
    column_mappings: Dict[str, str]  # {col_a: col_b}
    
    # Metadata
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    phase: str = "phase1"


@dataclass
class Phase1Output:
    """Complete output from Phase 1."""
    scenario: str
    timestamp: str
    
    # Homeomorphic mappings discovered
    mappings: List[HomeomorphicMapping]
    
    # Canonical objects created
    canonical_objects: List[Dict[str, Any]]
    
    # Cross-silo links
    cross_silo_links: List[Dict[str, Any]]
    
    # Statistics
    total_silos: int
    total_mappings: int
    topology_validation_score: float
    
    # Evidence bundle
    evidence: Dict[str, Any]
    
    def to_dict(self):
        """Convert to serializable dict."""
        return {
            "scenario": self.scenario,
            "timestamp": self.timestamp,
            "mappings": [asdict(m) for m in self.mappings],
            "canonical_objects": self.canonical_objects,
            "cross_silo_links": self.cross_silo_links,
            "total_silos": self.total_silos,
            "total_mappings": self.total_mappings,
            "topology_validation_score": self.topology_validation_score,
            "evidence": self.evidence,
            "phase": "phase1"
        }
    
    def save(self, filepath: str):
        """Save to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, filepath: str):
        """Load from JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        # Reconstruct mappings
        mappings = [HomeomorphicMapping(**m) for m in data.get("mappings", [])]
        
        return cls(
            scenario=data["scenario"],
            timestamp=data["timestamp"],
            mappings=mappings,
            canonical_objects=data.get("canonical_objects", []),
            cross_silo_links=data.get("cross_silo_links", []),
            total_silos=data.get("total_silos", 0),
            total_mappings=data.get("total_mappings", 0),
            topology_validation_score=data.get("topology_validation_score", 0.0),
            evidence=data.get("evidence", {})
        )


# ============================================================================
# PHASE 2: Insight Generation
# ============================================================================

@dataclass
class InsightEvidence:
    """Evidence supporting an insight."""
    source_table: str
    source_columns: List[str]
    data_snapshot: Dict[str, Any]
    
    # Link to Phase 1 mapping
    homeomorphic_mapping_id: Optional[str] = None
    canonical_object: Optional[str] = None


@dataclass
class Insight:
    """A validated business insight from Phase 2."""
    insight_id: str
    scenario: str
    
    # Core content
    explanation: str
    confidence: float
    severity: str  # "critical", "high", "medium", "low"
    
    # Evidence (links to Phase 1)
    evidence: List[InsightEvidence]
    source_signatures: List[str]  # Phase 1 mappings used
    canonical_objects: List[str]  # Phase 1 objects referenced
    
    # Cross-silo validation
    cross_silo_coverage: int  # Number of silos involved
    homeomorphic_links: List[str]  # IDs of Phase 1 mappings
    
    # Metrics
    kpis: Dict[str, float]  # Business KPIs affected
    estimated_impact: Optional[Dict[str, Any]] = None
    
    # LLM evaluation scores
    llm_eval_scores: Optional[Dict[str, float]] = None
    
    # Metadata
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    validated_at: Optional[str] = None
    phase: str = "phase2"
    
    # Original source (for traceability)
    source_file: Optional[str] = None
    source_row: Optional[Dict[str, Any]] = None


@dataclass
class Phase2Output:
    """Complete output from Phase 2."""
    scenario: str
    timestamp: str
    
    # Links to Phase 1
    phase1_file: str
    phase1_mappings_used: List[str]
    
    # Insights generated
    insights: List[Insight]
    
    # Validation scores
    topology_fidelity: float
    semantic_faithfulness: float
    insight_soundness: float
    action_alignment: float
    overall_score: float
    
    # LLM evaluation
    llm_judge_score: Optional[float] = None
    
    def to_dict(self):
        """Convert to serializable dict."""
        return {
            "scenario": self.scenario,
            "timestamp": self.timestamp,
            "phase1_file": self.phase1_file,
            "phase1_mappings_used": self.phase1_mappings_used,
            "insights": [asdict(ins) for ins in self.insights],
            "validation_scores": {
                "topology_fidelity": self.topology_fidelity,
                "semantic_faithfulness": self.semantic_faithfulness,
                "insight_soundness": self.insight_soundness,
                "action_alignment": self.action_alignment,
                "overall_score": self.overall_score
            },
            "llm_judge_score": self.llm_judge_score,
            "phase": "phase2"
        }
    
    def save(self, filepath: str):
        """Save to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, filepath: str):
        """Load from JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        # Reconstruct insights
        insights = []
        for ins_data in data.get("insights", []):
            # Reconstruct evidence
            evidence = [InsightEvidence(**e) for e in ins_data.get("evidence", [])]
            ins_data["evidence"] = evidence
            insights.append(Insight(**ins_data))
        
        scores = data.get("validation_scores", {})
        
        return cls(
            scenario=data["scenario"],
            timestamp=data["timestamp"],
            phase1_file=data["phase1_file"],
            phase1_mappings_used=data.get("phase1_mappings_used", []),
            insights=insights,
            topology_fidelity=scores.get("topology_fidelity", 0.0),
            semantic_faithfulness=scores.get("semantic_faithfulness", 0.0),
            insight_soundness=scores.get("insight_soundness", 0.0),
            action_alignment=scores.get("action_alignment", 0.0),
            overall_score=scores.get("overall_score", 0.0),
            llm_judge_score=data.get("llm_judge_score")
        )


# ============================================================================
# PHASE 3: Governance & Actions
# ============================================================================

class GovernanceDecisionType(Enum):
    """Types of governance decisions."""
    APPROVE = "approve"
    REJECT = "reject"
    SCHEDULE_REVIEW = "schedule_review"
    ESCALATE = "escalate"
    REQUEST_MORE_DATA = "request_more_data"


class ActionStatus(Enum):
    """Status of an action."""
    BACKLOG = "backlog"
    PLANNED = "planned"
    IN_PROGRESS = "in_progress"
    DONE = "done"
    CANCELLED = "cancelled"


class ActionPriority(Enum):
    """Priority levels."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class GovernanceDecision:
    """A governance decision on an insight."""
    decision_id: str
    
    # Link to Phase 2 insight
    insight_id: str
    phase2_file: str
    
    # Decision
    decision_type: GovernanceDecisionType
    confidence: float
    reasoning: str
    
    # Evidence (links to Phase 1 & 2)
    evidence: List[str]
    phase1_mappings_referenced: List[str]
    phase2_validation_scores: Dict[str, float]
    
    # Workflow
    next_review_date: Optional[str] = None
    escalation_path: Optional[List[str]] = None
    
    # Agent opinions (if multi-agent)
    agent_opinions: Optional[List[Dict[str, Any]]] = None
    
    # Metadata
    made_by: str = "GovernanceEngine"
    made_at: str = field(default_factory=lambda: datetime.now().isoformat())
    phase: str = "phase3"


@dataclass
class Action:
    """An actionable task."""
    action_id: str
    
    # Link to Phase 2 insight and Phase 3 decision
    insight_id: str
    decision_id: str
    phase2_file: str
    
    # Action details
    title: str
    description: str
    priority: ActionPriority
    status: ActionStatus
    
    # Assignment
    owner: Optional[str] = None
    due_date: Optional[str] = None
    
    # Impact estimation (required for approval)
    estimated_kpi_delta: Dict[str, float] = field(default_factory=dict)
    estimated_impact: Optional[Dict[str, Any]] = None
    simulation_results: Optional[Dict[str, Any]] = None
    
    # Evidence chain (full traceability)
    evidence_chain: Dict[str, Any] = field(default_factory=dict)
    
    # Metadata
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    completed_at: Optional[str] = None
    phase: str = "phase3"


@dataclass
class Phase3Output:
    """Complete output from Phase 3."""
    scenario: str
    timestamp: str
    
    # Links to previous phases
    phase1_file: str
    phase2_file: str
    
    # Governance decisions
    decisions: List[GovernanceDecision]
    
    # Actions generated
    actions: List[Action]
    
    # Statistics
    total_insights_processed: int
    approval_rate: float
    escalation_rate: float
    
    def to_dict(self):
        """Convert to serializable dict."""
        return {
            "scenario": self.scenario,
            "timestamp": self.timestamp,
            "phase1_file": self.phase1_file,
            "phase2_file": self.phase2_file,
            "decisions": [
                {
                    **asdict(d),
                    "decision_type": d.decision_type.value
                }
                for d in self.decisions
            ],
            "actions": [
                {
                    **asdict(a),
                    "priority": a.priority.value,
                    "status": a.status.value
                }
                for a in self.actions
            ],
            "statistics": {
                "total_insights_processed": self.total_insights_processed,
                "approval_rate": self.approval_rate,
                "escalation_rate": self.escalation_rate
            },
            "phase": "phase3"
        }
    
    def save(self, filepath: str):
        """Save to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, filepath: str):
        """Load from JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        # Reconstruct decisions
        decisions = []
        for d in data.get("decisions", []):
            d["decision_type"] = GovernanceDecisionType(d["decision_type"])
            decisions.append(GovernanceDecision(**d))
        
        # Reconstruct actions
        actions = []
        for a in data.get("actions", []):
            a["priority"] = ActionPriority(a["priority"])
            a["status"] = ActionStatus(a["status"])
            actions.append(Action(**a))
        
        stats = data.get("statistics", {})
        
        return cls(
            scenario=data["scenario"],
            timestamp=data["timestamp"],
            phase1_file=data["phase1_file"],
            phase2_file=data["phase2_file"],
            decisions=decisions,
            actions=actions,
            total_insights_processed=stats.get("total_insights_processed", 0),
            approval_rate=stats.get("approval_rate", 0.0),
            escalation_rate=stats.get("escalation_rate", 0.0)
        )


# ============================================================================
# Evidence Chain Tracking
# ============================================================================

@dataclass
class EvidenceChain:
    """Complete evidence chain from Phase 1 to Phase 3."""
    
    # Identifiers
    action_id: str
    insight_id: str
    
    # Phase 1: Topology
    homeomorphic_mappings: List[str]
    canonical_objects: List[str]
    topology_score: float
    
    # Phase 2: Insight
    insight_confidence: float
    insight_evidence: List[str]
    llm_eval_scores: Dict[str, float]
    
    # Phase 3: Decision
    decision_type: str
    decision_confidence: float
    governance_reasoning: str
    
    # Full traceability
    phase1_file: str
    phase2_file: str
    phase3_file: str
    
    def validate_continuity(self) -> bool:
        """Validate that evidence chain is complete."""
        required_fields = [
            self.action_id,
            self.insight_id,
            self.homeomorphic_mappings,
            self.canonical_objects,
            self.phase1_file,
            self.phase2_file,
            self.phase3_file
        ]
        return all(required_fields)
