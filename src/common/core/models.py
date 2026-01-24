"""
Core Data Models for the Ontology Platform

Defines the fundamental data structures for:
- Concepts (Object Types, Properties, Links, Aliases)
- Ontology Blocks (confirmed decisions)
- Shared State (system-wide state)
- Metadata (data source information)
"""

import hashlib
from datetime import datetime
from enum import Enum
from typing import Optional, List, Dict, Any, TYPE_CHECKING
from dataclasses import dataclass, field
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from .evidence import EvidenceBundle


# =============================================================================
# ENUMS
# =============================================================================

class ConceptType(str, Enum):
    """Types of ontology concepts"""
    OBJECT_TYPE = "object_type"      # Entity type (Customer, Order)
    PROPERTY = "property"            # Attribute (name, email)
    LINK_TYPE = "link_type"          # Relationship (placed, belongs_to)
    ALIAS = "alias"                  # Term mapping (customer_id = client_id)
    CONSTRAINT = "constraint"        # Constraint (unique, required)


class ConceptStatus(str, Enum):
    """Status of a concept in the pipeline"""
    PENDING = "pending"              # In mempool, waiting
    IN_DISCUSSION = "in_discussion"  # Currently being debated
    APPROVED = "approved"            # Consensus reached
    REJECTED = "rejected"            # Rejected by consensus
    NEEDS_REVISION = "needs_revision"  # Needs more work


class AgentRole(str, Enum):
    """Roles of agents in the consensus system"""
    DATA_ANALYST = "data_analyst"
    SEMANTIC_ARCHITECT = "semantic_architect"
    LINK_RESOLVER = "link_resolver"
    RISK_ANALYST = "risk_analyst"
    PROCESS_ARCHITECT = "process_architect"


class VoteType(str, Enum):
    """Types of votes (legacy - used for final decision mapping)"""
    APPROVE = "APPROVE"
    NEEDS_WORK = "NEEDS_WORK"
    REJECT = "REJECT"


class ConfidenceLevel(str, Enum):
    """Confidence level categories for soft consensus"""
    VERY_HIGH = "very_high"      # 0.9 - 1.0
    HIGH = "high"                # 0.75 - 0.9
    MODERATE = "moderate"        # 0.5 - 0.75
    LOW = "low"                  # 0.25 - 0.5
    VERY_LOW = "very_low"        # 0.0 - 0.25

    @classmethod
    def from_score(cls, score: float) -> "ConfidenceLevel":
        """Convert numeric score to confidence level"""
        if score >= 0.9:
            return cls.VERY_HIGH
        elif score >= 0.75:
            return cls.HIGH
        elif score >= 0.5:
            return cls.MODERATE
        elif score >= 0.25:
            return cls.LOW
        else:
            return cls.VERY_LOW


class UncertaintyType(str, Enum):
    """Types of uncertainty in agent assessments"""
    EPISTEMIC = "epistemic"          # Lack of knowledge (reducible)
    ALEATORY = "aleatory"            # Inherent randomness (irreducible)
    SEMANTIC = "semantic"            # Ambiguity in meaning
    CONFLICTING_EVIDENCE = "conflicting_evidence"  # Evidence contradicts


class SemanticRole(str, Enum):
    """Semantic role of a column - describes its business meaning"""
    IDENTIFIER = "identifier"        # Primary/Foreign key (account_id, customer_id)
    DIMENSION = "dimension"          # Categorical attribute for grouping (customer_type, tier, status)
    MEASURE = "measure"              # Numeric value for aggregation (amount, balance, score)
    TEMPORAL = "temporal"            # Date/time column (created_at, txn_ts)
    DESCRIPTOR = "descriptor"        # Descriptive text (name, description, notes)
    FLAG = "flag"                    # Boolean indicator (is_active, verified)
    REFERENCE = "reference"          # External reference (external_id, source_system)
    UNKNOWN = "unknown"              # Not yet classified


# =============================================================================
# SOFT CONSENSUS MODELS
# =============================================================================

class ConfidenceScore(BaseModel):
    """
    Confidence score with uncertainty quantification.
    Used in Soft/Graded Consensus system.
    """
    # Core confidence (0.0 to 1.0)
    value: float = Field(ge=0.0, le=1.0, description="Primary confidence score")

    # Uncertainty components
    uncertainty: float = Field(ge=0.0, le=1.0, default=0.1, description="Uncertainty level")
    uncertainty_type: Optional[UncertaintyType] = None

    # Evidence strength
    evidence_weight: float = Field(ge=0.0, le=1.0, default=0.5, description="Strength of supporting evidence")

    # Doubt factor (suspicion about the proposal)
    doubt: float = Field(ge=0.0, le=1.0, default=0.0, description="Level of doubt/suspicion")

    @property
    def level(self) -> ConfidenceLevel:
        """Get confidence level category"""
        return ConfidenceLevel.from_score(self.value)

    @property
    def adjusted_confidence(self) -> float:
        """Get confidence adjusted for uncertainty and doubt"""
        # Penalize for high uncertainty and doubt
        adjustment = (1 - self.uncertainty * 0.3) * (1 - self.doubt * 0.5)
        return max(0.0, min(1.0, self.value * adjustment))

    def to_vote(self) -> VoteType:
        """Convert confidence to legacy vote type"""
        adjusted = self.adjusted_confidence
        if adjusted >= 0.7:
            return VoteType.APPROVE
        elif adjusted >= 0.4:
            return VoteType.NEEDS_WORK
        else:
            return VoteType.REJECT


class AgentAssessment(BaseModel):
    """
    An agent's graded assessment of a concept.
    Replaces binary voting with confidence-based evaluation.
    """
    agent: AgentRole
    confidence: ConfidenceScore

    # Detailed reasoning
    reasoning: str = ""
    key_observations: List[str] = Field(default_factory=list)
    concerns: List[str] = Field(default_factory=list)
    suggestions: List[str] = Field(default_factory=list)

    # References to other agents' points
    references_to: List[str] = Field(default_factory=list)  # Agent names mentioned
    agrees_with: Dict[str, float] = Field(default_factory=dict)  # Agent -> agreement level
    disagrees_with: Dict[str, float] = Field(default_factory=dict)  # Agent -> disagreement level

    # Evidence and context
    evidence: Dict[str, Any] = Field(default_factory=dict)

    # Timestamps
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    updated_at: Optional[str] = None

    @property
    def legacy_vote(self) -> VoteType:
        """Get legacy vote type for compatibility"""
        return self.confidence.to_vote()

    def update_confidence(self, new_confidence: float, reason: str = ""):
        """Update confidence based on new information"""
        self.confidence.value = max(0.0, min(1.0, new_confidence))
        self.updated_at = datetime.now().isoformat()
        if reason:
            self.key_observations.append(f"[UPDATE] {reason}")


class SoftConsensusState(BaseModel):
    """
    State of soft consensus for a concept.
    Tracks probability distribution across agents.
    """
    concept_id: str

    # Agent assessments
    assessments: Dict[str, AgentAssessment] = Field(default_factory=dict)

    # Combined scores (computed from assessments)
    combined_confidence: float = 0.0
    combined_uncertainty: float = 1.0
    agreement_level: float = 0.0  # How much agents agree with each other

    # History of confidence updates
    confidence_history: List[Dict[str, Any]] = Field(default_factory=list)

    # Final decision
    is_decided: bool = False
    decision: Optional[str] = None  # "accepted", "rejected", "provisional"
    decision_threshold: float = 0.75  # Minimum combined confidence for acceptance

    # Provisional acceptance tracking
    is_provisional: bool = False
    provisional_confidence: Optional[float] = None

    def add_assessment(self, assessment: AgentAssessment):
        """Add or update an agent's assessment"""
        self.assessments[assessment.agent.value] = assessment
        self._recalculate_combined_scores()
        self._record_history()

    def _recalculate_combined_scores(self):
        """Recalculate combined scores using weighted averaging"""
        if not self.assessments:
            return

        # Weighted combination based on evidence strength
        total_weight = 0.0
        weighted_confidence = 0.0
        weighted_uncertainty = 0.0

        confidences = []

        for assessment in self.assessments.values():
            weight = assessment.confidence.evidence_weight
            conf = assessment.confidence.adjusted_confidence

            weighted_confidence += conf * weight
            weighted_uncertainty += assessment.confidence.uncertainty * weight
            total_weight += weight
            confidences.append(conf)

        if total_weight > 0:
            self.combined_confidence = weighted_confidence / total_weight
            self.combined_uncertainty = weighted_uncertainty / total_weight

        # Calculate agreement level (inverse of variance)
        if len(confidences) > 1:
            mean_conf = sum(confidences) / len(confidences)
            variance = sum((c - mean_conf) ** 2 for c in confidences) / len(confidences)
            self.agreement_level = max(0.0, 1.0 - variance * 4)  # Scale variance to 0-1
        else:
            self.agreement_level = 1.0

    def _record_history(self):
        """Record current state in history"""
        self.confidence_history.append({
            "timestamp": datetime.now().isoformat(),
            "combined_confidence": self.combined_confidence,
            "combined_uncertainty": self.combined_uncertainty,
            "agreement_level": self.agreement_level,
            "assessments": {
                k: v.confidence.value for k, v in self.assessments.items()
            }
        })

    def check_consensus(self) -> tuple[bool, str]:
        """
        Check if soft consensus is reached.

        Returns:
            (reached, result) where result is 'accepted', 'rejected', 'provisional', or 'undecided'
        """
        if len(self.assessments) < 3:
            return False, "insufficient_assessments"

        # High confidence + high agreement = accepted
        if (self.combined_confidence >= self.decision_threshold and
            self.agreement_level >= 0.6 and
            self.combined_uncertainty <= 0.4):
            self.is_decided = True
            self.decision = "accepted"
            return True, "accepted"

        # Low confidence + high agreement = rejected
        if (self.combined_confidence < 0.3 and
            self.agreement_level >= 0.6):
            self.is_decided = True
            self.decision = "rejected"
            return True, "rejected"

        # Moderate confidence or low agreement = provisional
        if (self.combined_confidence >= 0.5 and
            len(self.assessments) >= 3):
            self.is_provisional = True
            self.provisional_confidence = self.combined_confidence
            return True, "provisional"

        return False, "undecided"

    def get_summary(self) -> Dict[str, Any]:
        """Get summary of consensus state"""
        return {
            "concept_id": self.concept_id,
            "combined_confidence": round(self.combined_confidence, 3),
            "combined_uncertainty": round(self.combined_uncertainty, 3),
            "agreement_level": round(self.agreement_level, 3),
            "assessments": {
                agent: {
                    "confidence": round(a.confidence.value, 3),
                    "adjusted": round(a.confidence.adjusted_confidence, 3),
                    "legacy_vote": a.legacy_vote.value,
                }
                for agent, a in self.assessments.items()
            },
            "is_decided": self.is_decided,
            "decision": self.decision,
            "is_provisional": self.is_provisional,
        }


# =============================================================================
# METADATA MODELS (Data Source Information)
# =============================================================================

class ColumnMetadata(BaseModel):
    """Metadata for a single column"""
    name: str
    inferred_type: str
    nullable: bool = True
    unique: bool = False
    sample_values: List[Any] = Field(default_factory=list)
    null_ratio: float = 0.0
    distinct_count: int = 0
    is_key_candidate: bool = False
    is_foreign_key_candidate: bool = False
    potential_references: List[Dict[str, Any]] = Field(default_factory=list)
    statistics: Dict[str, Any] = Field(default_factory=dict)
    # CRITICAL: 키 컬럼의 모든 유니크 값을 저장 (크로스 사일로 FK 매칭용)
    # 일반 컬럼은 비어 있고, 키/FK 후보 컬럼만 전체 값을 저장
    all_key_values: List[str] = Field(default_factory=list)

    # Semantic information - captures business meaning of the column
    semantic_role: Optional[str] = None  # SemanticRole enum value
    business_name: Optional[str] = None  # Human-readable business name (e.g., "Risk Score")
    business_description: Optional[str] = None  # Description of business meaning
    domain_category: Optional[str] = None  # Domain category (e.g., "finance", "customer", "time")
    value_domain: Optional[Dict[str, Any]] = None  # Allowed values or range (e.g., {"min": 0, "max": 100} or {"values": ["A", "B", "C"]})


class TableMetadata(BaseModel):
    """Metadata for a database table"""
    name: str
    source_id: str
    row_count: int = 0
    columns: List[ColumnMetadata] = Field(default_factory=list)
    primary_key_candidates: List[str] = Field(default_factory=list)
    foreign_key_candidates: List[Dict[str, Any]] = Field(default_factory=list)
    quality_score: Dict[str, float] = Field(default_factory=dict)
    sample_data: List[Dict[str, Any]] = Field(default_factory=list)


class DataSource(BaseModel):
    """Information about a data source"""
    id: str
    name: str
    source_type: str  # "csv", "json", "postgresql", "mysql", etc.
    connection_info: Dict[str, Any] = Field(default_factory=dict)
    tables: List[TableMetadata] = Field(default_factory=list)
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())


# =============================================================================
# CONCEPT MODELS
# =============================================================================

class Concept(BaseModel):
    """
    A concept to be added to the ontology.
    This is the unit of discussion and consensus.
    """
    id: str
    type: ConceptType
    name: str
    display_name: str = ""
    description: str = ""
    definition: Dict[str, Any] = Field(default_factory=dict)
    source_evidence: Dict[str, Any] = Field(default_factory=dict)
    proposed_by: str = ""  # Agent role or "system"
    status: ConceptStatus = ConceptStatus.PENDING
    priority: int = 0
    dependencies: List[str] = Field(default_factory=list)
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())

    def to_context_string(self) -> str:
        """Format concept for LLM context"""
        parts = [
            f"**{self.type.value.upper()}**: {self.name}",
            f"Display Name: {self.display_name}" if self.display_name else "",
            f"Description: {self.description}" if self.description else "",
        ]

        if self.definition:
            parts.append(f"Definition: {self.definition}")

        if self.source_evidence:
            parts.append(f"Evidence: {self.source_evidence}")

        return "\n".join([p for p in parts if p])


# =============================================================================
# VOTE AND BLOCK MODELS
# =============================================================================

class AgentVote(BaseModel):
    """An agent's vote on a concept"""
    agent: AgentRole
    vote: VoteType
    reasoning: str
    evidence: Dict[str, Any] = Field(default_factory=dict)
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())


class OntologyBlock(BaseModel):
    """
    A confirmed block in the ontology chain.
    Immutable record of a consensus decision.

    Supports both legacy votes and soft consensus assessments.
    """
    id: int
    concept: Concept

    # Legacy votes (for compatibility)
    votes: List[AgentVote] = Field(default_factory=list)

    # Soft Consensus data
    consensus_state: Optional[SoftConsensusState] = None
    combined_confidence: float = 0.0
    is_provisional: bool = False

    discussion_summary: str = ""
    dissenting_views: List[Dict[str, Any]] = Field(default_factory=list)

    # Evidence information - stores data evidence metrics for audit
    evidence_info: Dict[str, Any] = Field(default_factory=dict)
    # Example: {
    #   "match_rate": 0.85,
    #   "sample_overlap": 42,
    #   "evidence_status": "strong_evidence",
    #   "needs_verification": False,
    #   "from_source": "manufacturing",
    #   "to_source": "marketing"
    # }

    # Full Evidence Bundle (optional) - comprehensive evidence package
    # Stored as dict for JSON serialization, reconstructed as EvidenceBundle when needed
    evidence_bundle: Optional[Dict[str, Any]] = None

    # Blockchain properties
    previous_hash: str = "0" * 64
    hash: str = ""
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())

    def calculate_hash(self) -> str:
        """Calculate SHA-256 hash of the block"""
        payload = (
            f"{self.id}"
            f"{self.concept.id}"
            f"{self.concept.type.value}"
            f"{self.concept.name}"
            f"{str(self.concept.definition)}"
            f"{self.combined_confidence}"
            f"{self.timestamp}"
            f"{self.previous_hash}"
        )
        return hashlib.sha256(payload.encode()).hexdigest()

    def model_post_init(self, __context: Any) -> None:
        """Calculate hash after initialization if not set"""
        if not self.hash:
            self.hash = self.calculate_hash()

    @property
    def confidence_level(self) -> ConfidenceLevel:
        """Get confidence level of this block"""
        return ConfidenceLevel.from_score(self.combined_confidence)

    def get_evidence_bundle(self) -> Optional["EvidenceBundle"]:
        """
        Reconstruct EvidenceBundle from stored dict.

        Returns:
            EvidenceBundle instance if evidence_bundle is set, None otherwise.
        """
        if not self.evidence_bundle:
            return None

        # Import here to avoid circular import
        from .evidence import EvidenceBundle as EB
        try:
            return EB.model_validate(self.evidence_bundle)
        except Exception:
            return None


# =============================================================================
# MESSAGE MODELS
# =============================================================================

class AgentMessage(BaseModel):
    """A message from an agent during discussion"""
    agent: str
    role: AgentRole
    content: str
    concept_id: str = ""
    vote: Optional[VoteType] = None
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())


# =============================================================================
# SHARED STATE
# =============================================================================

class OntologySharedState(BaseModel):
    """
    The shared state of the ontology construction system.
    This is the central data structure that all agents operate on.

    Supports both legacy voting and soft consensus systems.
    """
    # Session info
    session_id: str = Field(default_factory=lambda: datetime.now().strftime("%Y%m%d_%H%M%S"))

    # Data sources
    data_sources: List[DataSource] = Field(default_factory=list)

    # Concept mempool (pending concepts)
    concept_queue: List[Concept] = Field(default_factory=list)

    # Ontology chain (confirmed concepts)
    ontology_chain: List[OntologyBlock] = Field(default_factory=list)

    # Current discussion
    current_discussion: Optional[Dict[str, Any]] = None
    current_votes: Dict[str, AgentVote] = Field(default_factory=dict)

    # Soft Consensus State (NEW)
    current_consensus: Optional[SoftConsensusState] = None
    current_assessments: Dict[str, AgentAssessment] = Field(default_factory=dict)

    # Provisional concepts (accepted with lower confidence, subject to update)
    provisional_concepts: List[Dict[str, Any]] = Field(default_factory=list)

    # Message history
    messages: List[AgentMessage] = Field(default_factory=list)

    # Counters
    turn_count: int = 0
    total_concepts_processed: int = 0

    # Consensus mode
    use_soft_consensus: bool = True  # Toggle between legacy and soft consensus

    def get_pending_concepts(self) -> List[Concept]:
        """Get concepts waiting to be processed"""
        return [c for c in self.concept_queue if c.status == ConceptStatus.PENDING]

    def get_in_discussion_concept(self) -> Optional[Concept]:
        """Get the concept currently being discussed"""
        for c in self.concept_queue:
            if c.status == ConceptStatus.IN_DISCUSSION:
                return c
        return None

    def get_confirmed_ontology(self) -> Dict[str, Dict]:
        """Get the confirmed ontology as a dictionary"""
        ontology = {
            "object_types": {},
            "properties": {},
            "link_types": {},
            "aliases": {},
            "constraints": {},
        }

        for block in self.ontology_chain:
            concept = block.concept
            category = concept.type.value + "s"  # e.g., "object_types"
            if category in ontology:
                ontology[category][concept.name] = {
                    "display_name": concept.display_name,
                    "description": concept.description,
                    "definition": concept.definition,
                    "block_id": block.id,
                    "approved_at": block.timestamp,
                    # Soft consensus additions
                    "confidence": block.combined_confidence,
                    "confidence_level": block.confidence_level.value if block.combined_confidence > 0 else "unknown",
                    "is_provisional": block.is_provisional,
                }

        return ontology

    def get_latest_hash(self) -> str:
        """Get the hash of the latest block"""
        if self.ontology_chain:
            return self.ontology_chain[-1].hash
        return "0" * 64

    def add_concept(self, concept: Concept) -> None:
        """Add a concept to the queue"""
        self.concept_queue.append(concept)

    def add_block(self, block: OntologyBlock) -> None:
        """Add a confirmed block to the chain"""
        self.ontology_chain.append(block)
        self.total_concepts_processed += 1

    def add_message(self, message: AgentMessage) -> None:
        """Add a message to history"""
        self.messages.append(message)
        self.turn_count += 1

    # =========================================================================
    # SOFT CONSENSUS METHODS
    # =========================================================================

    def init_soft_consensus(self, concept_id: str) -> SoftConsensusState:
        """Initialize soft consensus state for a concept"""
        self.current_consensus = SoftConsensusState(concept_id=concept_id)
        self.current_assessments = {}
        return self.current_consensus

    def add_assessment(self, assessment: AgentAssessment) -> None:
        """Add an agent's assessment to current consensus"""
        if self.current_consensus:
            self.current_consensus.add_assessment(assessment)
            self.current_assessments[assessment.agent.value] = assessment

    def get_consensus_summary(self) -> Optional[Dict[str, Any]]:
        """Get summary of current consensus state"""
        if self.current_consensus:
            return self.current_consensus.get_summary()
        return None

    def update_provisional_concept(
        self,
        concept_id: str,
        new_confidence: float,
        reason: str = ""
    ) -> bool:
        """
        Update a provisional concept's confidence based on new information.

        Returns True if confidence was updated.
        """
        for block in self.ontology_chain:
            if block.concept.id == concept_id and block.is_provisional:
                old_confidence = block.combined_confidence
                block.combined_confidence = new_confidence

                # If confidence dropped too low, mark for review
                if new_confidence < 0.3:
                    block.concept.status = ConceptStatus.NEEDS_REVISION

                # Record the update in provisional concepts list
                self.provisional_concepts.append({
                    "concept_id": concept_id,
                    "old_confidence": old_confidence,
                    "new_confidence": new_confidence,
                    "reason": reason,
                    "timestamp": datetime.now().isoformat(),
                })

                return True
        return False

    def get_confidence_distribution(self) -> Dict[str, int]:
        """Get distribution of confidence levels across all blocks"""
        distribution = {level.value: 0 for level in ConfidenceLevel}

        for block in self.ontology_chain:
            if block.combined_confidence > 0:
                level = block.confidence_level.value
                distribution[level] += 1

        return distribution


# =============================================================================
# AGENT CONFIGURATION
# =============================================================================

@dataclass
class AgentConfig:
    """Configuration for an agent"""
    role: AgentRole
    name: str
    model: str
    expertise: str
    color: str
    expertise_keywords: List[str] = field(default_factory=list)


# Default agent configurations
AGENT_CONFIGS = {
    AgentRole.DATA_ANALYST: AgentConfig(
        role=AgentRole.DATA_ANALYST,
        name="Data Analyst",
        model="gpt-5.1",
        expertise="Schema Analysis, Data Quality, Statistics",
        color="blue",
        expertise_keywords=[
            "schema", "column", "table", "type", "null", "distribution",
            "cardinality", "sample", "statistics", "quality", "profile",
            "data", "row", "count", "unique", "duplicate"
        ]
    ),
    AgentRole.SEMANTIC_ARCHITECT: AgentConfig(
        role=AgentRole.SEMANTIC_ARCHITECT,
        name="Semantic Architect",
        model="claude-opus-4-5-20251101",
        expertise="Concept Definition, Semantic Modeling, Terminology",
        color="green",
        expertise_keywords=[
            "object", "entity", "concept", "property", "meaning", "semantic",
            "definition", "alias", "terminology", "domain", "ontology",
            "model", "type", "attribute", "name"
        ]
    ),
    AgentRole.LINK_RESOLVER: AgentConfig(
        role=AgentRole.LINK_RESOLVER,
        name="Link Resolver",
        model="gemini-3-pro-preview",
        expertise="Relationships, Entity Resolution, Foreign Keys",
        color="magenta",
        expertise_keywords=[
            "relationship", "foreign", "key", "link", "reference", "join",
            "entity", "resolution", "matching", "cardinality", "one-to-many",
            "many-to-many", "connect", "relate"
        ]
    ),
    AgentRole.RISK_ANALYST: AgentConfig(
        role=AgentRole.RISK_ANALYST,
        name="Risk Analyst",
        model="gpt-5.1",
        expertise="Risk Assessment, Compliance, Data Governance",
        color="red",
        expertise_keywords=[
            "risk", "compliance", "governance", "security", "audit",
            "privacy", "regulation", "vulnerability", "exposure", "mitigation",
            "threat", "impact", "probability", "control", "breach"
        ]
    ),
    AgentRole.PROCESS_ARCHITECT: AgentConfig(
        role=AgentRole.PROCESS_ARCHITECT,
        name="Process Architect",
        model="claude-opus-4-5-20251101",
        expertise="Business Process, Workflow, Integration Design",
        color="cyan",
        expertise_keywords=[
            "process", "workflow", "automation", "integration", "pipeline",
            "orchestration", "flow", "step", "stage", "sequence",
            "trigger", "event", "action", "business", "operation"
        ]
    ),
}
