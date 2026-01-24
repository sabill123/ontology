"""
Evidence Bundle and Certification System

Provides comprehensive evidence management for ontology concepts:
- EvidenceBundle: Multi-dimensional evidence package
- Certification: Trust levels and human sign-off workflow
- Evidence scoring and aggregation

This system enables auditable, traceable ontology decisions
with clear evidence lineage.
"""

import hashlib
from datetime import datetime, timedelta
from enum import Enum
from typing import Optional, List, Dict, Any, Callable
from dataclasses import dataclass, field
from pydantic import BaseModel, Field, validator

from .models import AgentRole, ConfidenceLevel


# =============================================================================
# ENUMS
# =============================================================================

class CertificationLevel(str, Enum):
    """
    Certification levels for ontology concepts.

    Concepts progress through these levels based on confidence
    and human review status.
    """
    AUTO = "auto"                    # Confidence ≥ 0.85, full data support
    PROVISIONAL = "provisional"       # 0.5 ≤ confidence < 0.85
    HUMAN_REQUIRED = "human_required" # Confidence < 0.5 or conflicts
    CERTIFIED = "certified"          # Human sign-off completed
    DEPRECATED = "deprecated"        # Marked for removal


class EvidenceType(str, Enum):
    """Types of evidence that support ontology decisions"""
    DATA_MATCH = "data_match"          # Statistical data matching
    SCHEMA_INFERENCE = "schema_inference"  # Schema-based inference
    SEMANTIC_ANALYSIS = "semantic_analysis"  # LLM semantic analysis
    BUSINESS_RULE = "business_rule"    # Business rule validation
    USER_FEEDBACK = "user_feedback"    # Human input/correction
    CROSS_SILO = "cross_silo"          # Cross-silo relationship evidence
    HISTORICAL = "historical"          # Historical pattern evidence
    TOPOLOGY_MAPPING = "topology_mapping"  # Structure-preserving isomorphism evidence
    CANONICAL_MATCH = "canonical_match"    # Canonical form matching evidence


class EvidenceStrength(str, Enum):
    """Strength classification for evidence"""
    STRONG = "strong"          # score ≥ 0.8
    MODERATE = "moderate"      # 0.5 ≤ score < 0.8
    WEAK = "weak"              # 0.3 ≤ score < 0.5
    INSUFFICIENT = "insufficient"  # score < 0.3

    @classmethod
    def from_score(cls, score: float) -> "EvidenceStrength":
        if score >= 0.8:
            return cls.STRONG
        elif score >= 0.5:
            return cls.MODERATE
        elif score >= 0.3:
            return cls.WEAK
        else:
            return cls.INSUFFICIENT


# =============================================================================
# EVIDENCE COMPONENTS
# =============================================================================

class DataEvidence(BaseModel):
    """
    Statistical and data-driven evidence.

    Captures metrics from data analysis that support the concept.
    """
    # Match statistics
    match_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    sample_overlap: int = 0
    total_checked: int = 0

    # Data quality metrics
    null_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    duplicate_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    consistency_score: float = Field(default=0.0, ge=0.0, le=1.0)

    # Column/table statistics
    source_table: Optional[str] = None
    target_table: Optional[str] = None
    matched_columns: List[str] = Field(default_factory=list)

    # Sample data (for audit)
    sample_matches: List[Dict[str, Any]] = Field(default_factory=list)

    # Timestamp
    analyzed_at: str = Field(default_factory=lambda: datetime.now().isoformat())

    @property
    def score(self) -> float:
        """Calculate overall data evidence score"""
        # Weighted combination
        if self.total_checked == 0:
            return 0.0

        match_weight = 0.6
        quality_weight = 0.4

        quality_score = (
            (1 - self.null_rate) * 0.3 +
            (1 - self.duplicate_rate) * 0.3 +
            self.consistency_score * 0.4
        )

        return self.match_rate * match_weight + quality_score * quality_weight

    @property
    def strength(self) -> EvidenceStrength:
        return EvidenceStrength.from_score(self.score)


class SemanticEvidence(BaseModel):
    """
    LLM-based semantic analysis evidence.

    Captures reasoning and semantic understanding from LLM analysis.
    """
    model_config = {"protected_namespaces": ()}

    # LLM analysis results
    semantic_match_score: float = Field(default=0.0, ge=0.0, le=1.0)
    reasoning: str = ""
    key_observations: List[str] = Field(default_factory=list)

    # Domain knowledge
    domain_relevance: float = Field(default=0.0, ge=0.0, le=1.0)
    terminology_accuracy: float = Field(default=0.0, ge=0.0, le=1.0)

    # Entity resolution
    entity_similarity: float = Field(default=0.0, ge=0.0, le=1.0)
    name_matching_confidence: float = Field(default=0.0, ge=0.0, le=1.0)

    # Model metadata
    model_used: Optional[str] = None
    analyzed_at: str = Field(default_factory=lambda: datetime.now().isoformat())

    @property
    def score(self) -> float:
        """Calculate overall semantic evidence score"""
        return (
            self.semantic_match_score * 0.4 +
            self.domain_relevance * 0.2 +
            self.terminology_accuracy * 0.2 +
            self.entity_similarity * 0.2
        )

    @property
    def strength(self) -> EvidenceStrength:
        return EvidenceStrength.from_score(self.score)


class BusinessEvidence(BaseModel):
    """
    Business context and rule-based evidence.

    Captures business validation and contextual relevance.
    """
    # Business rule validation
    rules_passed: List[str] = Field(default_factory=list)
    rules_failed: List[str] = Field(default_factory=list)
    rule_compliance_rate: float = Field(default=1.0, ge=0.0, le=1.0)

    # Use case relevance
    use_cases: List[str] = Field(default_factory=list)
    use_case_coverage: float = Field(default=0.0, ge=0.0, le=1.0)

    # Stakeholder input
    stakeholder_approvals: List[str] = Field(default_factory=list)
    stakeholder_concerns: List[str] = Field(default_factory=list)

    # Business impact
    business_criticality: str = "medium"  # low, medium, high, critical
    integration_priority: int = 0

    @property
    def score(self) -> float:
        """Calculate overall business evidence score"""
        # Base score from rule compliance
        base_score = self.rule_compliance_rate * 0.5

        # Use case coverage
        use_case_score = self.use_case_coverage * 0.3

        # Stakeholder alignment (approvals vs concerns)
        total_feedback = len(self.stakeholder_approvals) + len(self.stakeholder_concerns)
        if total_feedback > 0:
            alignment = len(self.stakeholder_approvals) / total_feedback
        else:
            alignment = 0.5  # Neutral if no feedback

        return base_score + use_case_score + alignment * 0.2

    @property
    def strength(self) -> EvidenceStrength:
        return EvidenceStrength.from_score(self.score)


class AgentAssessmentEvidence(BaseModel):
    """
    Evidence from agent consensus process.

    Captures individual agent evaluations.
    """
    agent_role: AgentRole
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str = ""
    concerns: List[str] = Field(default_factory=list)
    suggestions: List[str] = Field(default_factory=list)
    evidence_weight: float = Field(default=0.5, ge=0.0, le=1.0)

    # Agreement with other agents
    agrees_with: Dict[str, float] = Field(default_factory=dict)
    disagrees_with: Dict[str, float] = Field(default_factory=dict)

    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())


class TopologyEvidence(BaseModel):
    """
    Topological structure-preserving evidence.

    Captures evidence from isomorphism detection and canonical form matching.
    This enables structure-based ontology integration independent of naming.
    """
    # Isomorphism mapping info
    has_isomorphism: bool = False
    isomorphism_id: Optional[str] = None
    structural_similarity: float = Field(default=0.0, ge=0.0, le=1.0)
    semantic_similarity: float = Field(default=0.0, ge=0.0, le=1.0)
    mapping_confidence: float = Field(default=0.0, ge=0.0, le=1.0)

    # Canonical form info
    canonical_id: Optional[str] = None
    canonical_name: Optional[str] = None
    canonical_match_score: float = Field(default=0.0, ge=0.0, le=1.0)

    # Preservation metrics
    completeness_score: float = Field(default=0.0, ge=0.0, le=1.0)
    fidelity_score: float = Field(default=0.0, ge=0.0, le=1.0)
    is_exact_isomorphism: bool = False
    is_partial_match: bool = False

    # Source traceability
    source_graph_id: Optional[str] = None
    source_silo_id: Optional[str] = None
    original_columns: Dict[str, str] = Field(default_factory=dict)
    # canonical_property -> original_column

    # Topology signature
    topology_signature_hash: Optional[str] = None
    node_count: int = 0
    edge_count: int = 0

    # Matched silos (for cross-silo canonical forms)
    matched_silos: List[str] = Field(default_factory=list)

    # Timestamp
    analyzed_at: str = Field(default_factory=lambda: datetime.now().isoformat())

    @property
    def score(self) -> float:
        """Calculate overall topology evidence score"""
        if not self.has_isomorphism:
            return 0.0

        structure_weight = 0.35
        semantic_weight = 0.25
        preservation_weight = 0.40

        preservation_score = (self.completeness_score + self.fidelity_score) / 2

        return (
            self.structural_similarity * structure_weight +
            self.semantic_similarity * semantic_weight +
            preservation_score * preservation_weight
        )

    @property
    def strength(self) -> "EvidenceStrength":
        return EvidenceStrength.from_score(self.score)

    def is_structure_preserving(self) -> bool:
        """Check if the mapping is truly structure-preserving"""
        return (
            self.has_isomorphism and
            self.completeness_score >= 0.8 and
            self.fidelity_score >= 0.8
        )


# =============================================================================
# CERTIFICATION
# =============================================================================

class CertificationInfo(BaseModel):
    """
    Certification status and metadata.

    Tracks the trust level and human review status of a concept.
    """
    level: CertificationLevel = CertificationLevel.PROVISIONAL

    # Auto-certification metadata
    auto_certified: bool = False
    auto_certification_score: Optional[float] = None
    auto_certification_reason: Optional[str] = None

    # Human certification
    certified_by: Optional[str] = None  # user_id
    certified_at: Optional[str] = None
    certification_notes: Optional[str] = None

    # Review history
    review_history: List[Dict[str, Any]] = Field(default_factory=list)

    # Expiry and re-review
    expiry: Optional[str] = None  # ISO datetime string
    requires_re_review: bool = False
    re_review_reason: Optional[str] = None

    # Deprecation
    deprecated_at: Optional[str] = None
    deprecated_by: Optional[str] = None
    deprecation_reason: Optional[str] = None

    def certify(
        self,
        user_id: str,
        notes: Optional[str] = None,
        expiry_days: int = 365
    ) -> None:
        """Human certification sign-off"""
        self.level = CertificationLevel.CERTIFIED
        self.certified_by = user_id
        self.certified_at = datetime.now().isoformat()
        self.certification_notes = notes
        self.expiry = (datetime.now() + timedelta(days=expiry_days)).isoformat()

        self.review_history.append({
            "action": "certified",
            "by": user_id,
            "at": self.certified_at,
            "notes": notes,
        })

    def deprecate(self, user_id: str, reason: str) -> None:
        """Mark concept as deprecated"""
        self.level = CertificationLevel.DEPRECATED
        self.deprecated_by = user_id
        self.deprecated_at = datetime.now().isoformat()
        self.deprecation_reason = reason

        self.review_history.append({
            "action": "deprecated",
            "by": user_id,
            "at": self.deprecated_at,
            "reason": reason,
        })

    def mark_for_review(self, reason: str) -> None:
        """Mark for re-review"""
        self.requires_re_review = True
        self.re_review_reason = reason

        self.review_history.append({
            "action": "marked_for_review",
            "at": datetime.now().isoformat(),
            "reason": reason,
        })

    def is_expired(self) -> bool:
        """Check if certification has expired"""
        if not self.expiry:
            return False
        return datetime.now() > datetime.fromisoformat(self.expiry)

    def is_valid(self) -> bool:
        """Check if certification is valid (certified and not expired)"""
        return (
            self.level == CertificationLevel.CERTIFIED and
            not self.is_expired() and
            not self.requires_re_review
        )


# =============================================================================
# EVIDENCE BUNDLE
# =============================================================================

class EvidenceBundle(BaseModel):
    """
    Complete evidence package for an ontology concept.

    Aggregates all types of evidence with weighted scoring
    for transparent decision-making.
    """
    # Identity
    concept_id: str
    concept_name: str
    concept_type: str  # "object_type", "link_type", etc.

    # Timestamp
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    updated_at: Optional[str] = None

    # Evidence components
    data_evidence: DataEvidence = Field(default_factory=DataEvidence)
    semantic_evidence: SemanticEvidence = Field(default_factory=SemanticEvidence)
    business_evidence: BusinessEvidence = Field(default_factory=BusinessEvidence)
    topology_evidence: TopologyEvidence = Field(default_factory=TopologyEvidence)

    # Agent assessments
    agent_assessments: List[AgentAssessmentEvidence] = Field(default_factory=list)

    # Certification
    certification: CertificationInfo = Field(default_factory=CertificationInfo)

    # Evidence weights (customizable per domain)
    weights: Dict[str, float] = Field(default_factory=lambda: {
        "data": 0.35,
        "semantic": 0.25,
        "business": 0.20,
        "topology": 0.20,
    })

    # Audit trail
    evidence_log: List[Dict[str, Any]] = Field(default_factory=list)

    # Cross-references
    related_concepts: List[str] = Field(default_factory=list)
    source_references: List[Dict[str, str]] = Field(default_factory=list)

    def combined_score(self) -> float:
        """
        Calculate weighted combined evidence score.

        Returns:
            Combined score between 0.0 and 1.0
        """
        data_score = self.data_evidence.score * self.weights.get("data", 0.35)
        semantic_score = self.semantic_evidence.score * self.weights.get("semantic", 0.25)
        business_score = self.business_evidence.score * self.weights.get("business", 0.20)
        topology_score = self.topology_evidence.score * self.weights.get("topology", 0.20)

        return min(1.0, data_score + semantic_score + business_score + topology_score)

    @property
    def overall_strength(self) -> EvidenceStrength:
        """Get overall evidence strength"""
        return EvidenceStrength.from_score(self.combined_score())

    @property
    def confidence_level(self) -> ConfidenceLevel:
        """Get confidence level from combined score"""
        return ConfidenceLevel.from_score(self.combined_score())

    def get_agent_consensus(self) -> Dict[str, Any]:
        """
        Calculate agent consensus metrics.

        Returns:
            Dictionary with consensus information
        """
        if not self.agent_assessments:
            return {
                "has_consensus": False,
                "agent_count": 0,
                "average_confidence": 0.0,
                "agreement_level": 0.0,
            }

        confidences = [a.confidence for a in self.agent_assessments]
        avg_confidence = sum(confidences) / len(confidences)

        # Calculate agreement (inverse of variance)
        if len(confidences) > 1:
            mean = avg_confidence
            variance = sum((c - mean) ** 2 for c in confidences) / len(confidences)
            agreement = max(0.0, 1.0 - variance * 4)
        else:
            agreement = 1.0

        return {
            "has_consensus": len(self.agent_assessments) >= 3 and agreement >= 0.6,
            "agent_count": len(self.agent_assessments),
            "average_confidence": round(avg_confidence, 3),
            "agreement_level": round(agreement, 3),
            "assessments": {
                a.agent_role.value: {
                    "confidence": a.confidence,
                    "has_concerns": len(a.concerns) > 0,
                }
                for a in self.agent_assessments
            }
        }

    def determine_certification_level(self) -> CertificationLevel:
        """
        Determine appropriate certification level based on evidence.

        Returns:
            Recommended CertificationLevel
        """
        score = self.combined_score()
        consensus = self.get_agent_consensus()

        # Strong evidence + consensus = AUTO
        if (score >= 0.85 and
            consensus["has_consensus"] and
            consensus["average_confidence"] >= 0.8):
            return CertificationLevel.AUTO

        # Moderate evidence = PROVISIONAL
        if score >= 0.5:
            return CertificationLevel.PROVISIONAL

        # Weak evidence = HUMAN_REQUIRED
        return CertificationLevel.HUMAN_REQUIRED

    def auto_certify_if_eligible(self) -> bool:
        """
        Attempt automatic certification if evidence is strong enough.

        Returns:
            True if auto-certified, False otherwise
        """
        level = self.determine_certification_level()

        if level == CertificationLevel.AUTO:
            self.certification.level = CertificationLevel.AUTO
            self.certification.auto_certified = True
            self.certification.auto_certification_score = self.combined_score()
            self.certification.auto_certification_reason = (
                f"Strong evidence (score: {self.combined_score():.2f}) "
                f"with agent consensus"
            )
            self.certification.review_history.append({
                "action": "auto_certified",
                "at": datetime.now().isoformat(),
                "score": self.combined_score(),
            })
            return True

        self.certification.level = level
        return False

    def add_evidence(
        self,
        evidence_type: EvidenceType,
        data: Dict[str, Any],
        source: str = "system"
    ) -> None:
        """
        Add evidence and log the addition.

        Args:
            evidence_type: Type of evidence being added
            data: Evidence data
            source: Source of the evidence
        """
        self.evidence_log.append({
            "type": evidence_type.value,
            "data": data,
            "source": source,
            "timestamp": datetime.now().isoformat(),
        })

        # Update the appropriate evidence component
        if evidence_type == EvidenceType.DATA_MATCH:
            for key, value in data.items():
                if hasattr(self.data_evidence, key):
                    setattr(self.data_evidence, key, value)

        elif evidence_type == EvidenceType.SEMANTIC_ANALYSIS:
            for key, value in data.items():
                if hasattr(self.semantic_evidence, key):
                    setattr(self.semantic_evidence, key, value)

        elif evidence_type == EvidenceType.BUSINESS_RULE:
            for key, value in data.items():
                if hasattr(self.business_evidence, key):
                    setattr(self.business_evidence, key, value)

        elif evidence_type in (EvidenceType.TOPOLOGY_MAPPING, EvidenceType.CANONICAL_MATCH):
            for key, value in data.items():
                if hasattr(self.topology_evidence, key):
                    setattr(self.topology_evidence, key, value)

        self.updated_at = datetime.now().isoformat()

    def add_agent_assessment(self, assessment: AgentAssessmentEvidence) -> None:
        """Add an agent's assessment"""
        # Remove existing assessment from same agent if present
        self.agent_assessments = [
            a for a in self.agent_assessments
            if a.agent_role != assessment.agent_role
        ]
        self.agent_assessments.append(assessment)
        self.updated_at = datetime.now().isoformat()

    def to_summary(self) -> Dict[str, Any]:
        """Get a summary of the evidence bundle"""
        return {
            "concept_id": self.concept_id,
            "concept_name": self.concept_name,
            "concept_type": self.concept_type,
            "combined_score": round(self.combined_score(), 3),
            "overall_strength": self.overall_strength.value,
            "confidence_level": self.confidence_level.value,
            "component_scores": {
                "data": round(self.data_evidence.score, 3),
                "semantic": round(self.semantic_evidence.score, 3),
                "business": round(self.business_evidence.score, 3),
                "topology": round(self.topology_evidence.score, 3),
            },
            "topology_info": {
                "has_isomorphism": self.topology_evidence.has_isomorphism,
                "canonical_id": self.topology_evidence.canonical_id,
                "canonical_name": self.topology_evidence.canonical_name,
                "is_structure_preserving": self.topology_evidence.is_structure_preserving(),
                "matched_silos": self.topology_evidence.matched_silos,
            } if self.topology_evidence.has_isomorphism else None,
            "certification": {
                "level": self.certification.level.value,
                "is_valid": self.certification.is_valid(),
                "certified_by": self.certification.certified_by,
            },
            "agent_consensus": self.get_agent_consensus(),
        }

    def calculate_hash(self) -> str:
        """Calculate hash of evidence for integrity verification"""
        import json
        content = json.dumps({
            "concept_id": self.concept_id,
            "data_score": self.data_evidence.score,
            "semantic_score": self.semantic_evidence.score,
            "business_score": self.business_evidence.score,
            "combined_score": self.combined_score(),
            "certification_level": self.certification.level.value,
        }, sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()[:16]


# =============================================================================
# EVIDENCE FACTORY
# =============================================================================

class EvidenceFactory:
    """
    Factory for creating and managing evidence bundles.

    Provides convenient methods for common evidence scenarios.
    """

    @staticmethod
    def create_for_object_type(
        concept_id: str,
        name: str,
        source_table: str,
        match_rate: float = 0.0,
        sample_count: int = 0,
    ) -> EvidenceBundle:
        """Create evidence bundle for an object type"""
        bundle = EvidenceBundle(
            concept_id=concept_id,
            concept_name=name,
            concept_type="object_type",
            data_evidence=DataEvidence(
                match_rate=match_rate,
                total_checked=sample_count,
                source_table=source_table,
            ),
        )
        return bundle

    @staticmethod
    def create_for_link_type(
        concept_id: str,
        name: str,
        from_type: str,
        to_type: str,
        match_rate: float,
        sample_overlap: int,
        total_checked: int,
        is_cross_silo: bool = False,
    ) -> EvidenceBundle:
        """Create evidence bundle for a link type"""
        bundle = EvidenceBundle(
            concept_id=concept_id,
            concept_name=name,
            concept_type="link_type",
            data_evidence=DataEvidence(
                match_rate=match_rate,
                sample_overlap=sample_overlap,
                total_checked=total_checked,
                source_table=from_type,
                target_table=to_type,
            ),
            source_references=[
                {"type": "from_object", "name": from_type},
                {"type": "to_object", "name": to_type},
            ],
        )

        if is_cross_silo:
            bundle.add_evidence(
                EvidenceType.CROSS_SILO,
                {"is_cross_silo": True, "integration_type": "data_silo_bridge"},
                source="system"
            )

        return bundle

    @staticmethod
    def create_from_agent_votes(
        concept_id: str,
        concept_name: str,
        concept_type: str,
        assessments: List[Dict[str, Any]],
    ) -> EvidenceBundle:
        """Create evidence bundle from agent assessment results"""
        bundle = EvidenceBundle(
            concept_id=concept_id,
            concept_name=concept_name,
            concept_type=concept_type,
        )

        for assessment in assessments:
            agent_evidence = AgentAssessmentEvidence(
                agent_role=AgentRole(assessment.get("agent", "data_analyst")),
                confidence=assessment.get("confidence", 0.5),
                reasoning=assessment.get("reasoning", ""),
                concerns=assessment.get("concerns", []),
                suggestions=assessment.get("suggestions", []),
            )
            bundle.add_agent_assessment(agent_evidence)

        return bundle

    @staticmethod
    def create_with_topology(
        concept_id: str,
        concept_name: str,
        concept_type: str,
        canonical_id: str,
        canonical_name: str,
        isomorphism_mapping: Dict[str, Any],
        source_silo_id: str,
        original_columns: Dict[str, str],
    ) -> EvidenceBundle:
        """
        Create evidence bundle with topology mapping information.

        Args:
            concept_id: Concept identifier
            concept_name: Concept name
            concept_type: Type of concept (object_type, link_type)
            canonical_id: ID of the canonical form
            canonical_name: Name of the canonical form
            isomorphism_mapping: Mapping details from IsomorphismMapping
            source_silo_id: ID of the source silo
            original_columns: Mapping from canonical properties to original columns

        Returns:
            EvidenceBundle with topology evidence populated
        """
        bundle = EvidenceBundle(
            concept_id=concept_id,
            concept_name=concept_name,
            concept_type=concept_type,
            topology_evidence=TopologyEvidence(
                has_isomorphism=True,
                isomorphism_id=isomorphism_mapping.get("id"),
                structural_similarity=isomorphism_mapping.get("structural_similarity", 0.0),
                semantic_similarity=isomorphism_mapping.get("semantic_similarity", 0.0),
                mapping_confidence=isomorphism_mapping.get("confidence", 0.0),
                canonical_id=canonical_id,
                canonical_name=canonical_name,
                canonical_match_score=isomorphism_mapping.get("confidence", 0.0),
                completeness_score=isomorphism_mapping.get("completeness_score", 0.0),
                fidelity_score=isomorphism_mapping.get("fidelity_score", 0.0),
                is_exact_isomorphism=isomorphism_mapping.get("is_exact_isomorphism", False),
                is_partial_match=isomorphism_mapping.get("is_partial", False),
                source_silo_id=source_silo_id,
                original_columns=original_columns,
                matched_silos=isomorphism_mapping.get("matched_silos", [source_silo_id]),
            ),
        )

        bundle.add_evidence(
            EvidenceType.TOPOLOGY_MAPPING,
            {
                "canonical_id": canonical_id,
                "canonical_name": canonical_name,
                "source_silo": source_silo_id,
            },
            source="topology_engine"
        )

        return bundle

    @staticmethod
    def create_for_canonical_match(
        concept_id: str,
        concept_name: str,
        concept_type: str,
        canonical_id: str,
        canonical_name: str,
        matched_silos: List[str],
        match_confidence: float,
        structural_similarity: float,
        preservation_score: float,
    ) -> EvidenceBundle:
        """
        Create evidence bundle for a canonical form match across silos.

        Args:
            concept_id: Concept identifier
            concept_name: Concept name
            concept_type: Type of concept
            canonical_id: ID of the matched canonical form
            canonical_name: Name of the canonical form
            matched_silos: List of silo IDs that map to this canonical
            match_confidence: Overall match confidence
            structural_similarity: Structural similarity score
            preservation_score: Data preservation score

        Returns:
            EvidenceBundle for canonical match
        """
        bundle = EvidenceBundle(
            concept_id=concept_id,
            concept_name=concept_name,
            concept_type=concept_type,
            topology_evidence=TopologyEvidence(
                has_isomorphism=True,
                canonical_id=canonical_id,
                canonical_name=canonical_name,
                canonical_match_score=match_confidence,
                structural_similarity=structural_similarity,
                completeness_score=preservation_score,
                fidelity_score=preservation_score,
                matched_silos=matched_silos,
            ),
        )

        bundle.add_evidence(
            EvidenceType.CANONICAL_MATCH,
            {
                "canonical_id": canonical_id,
                "canonical_name": canonical_name,
                "matched_silos": matched_silos,
                "silo_count": len(matched_silos),
            },
            source="canonical_resolver"
        )

        return bundle


# =============================================================================
# REVIEW QUEUE
# =============================================================================

class ReviewQueueItem(BaseModel):
    """Item in the human review queue"""
    evidence_bundle: EvidenceBundle
    priority: int = 0
    queued_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    assigned_to: Optional[str] = None
    due_date: Optional[str] = None
    status: str = "pending"  # pending, in_review, completed, skipped

    def assign(self, user_id: str, due_days: int = 7) -> None:
        """Assign to a reviewer"""
        self.assigned_to = user_id
        self.status = "in_review"
        self.due_date = (datetime.now() + timedelta(days=due_days)).isoformat()

    def complete(self, approved: bool, notes: Optional[str] = None) -> None:
        """Complete the review"""
        self.status = "completed"
        if approved:
            self.evidence_bundle.certification.certify(
                self.assigned_to or "system",
                notes
            )


class HumanReviewQueue:
    """
    Queue for concepts requiring human review.

    Manages the workflow for human certification.
    """

    def __init__(self):
        self.queue: List[ReviewQueueItem] = []

    def add(
        self,
        evidence_bundle: EvidenceBundle,
        priority: int = 0,
    ) -> ReviewQueueItem:
        """Add item to review queue"""
        item = ReviewQueueItem(
            evidence_bundle=evidence_bundle,
            priority=priority,
        )
        self.queue.append(item)
        self.queue.sort(key=lambda x: -x.priority)  # Higher priority first
        return item

    def get_pending(self, limit: int = 10) -> List[ReviewQueueItem]:
        """Get pending review items"""
        return [
            item for item in self.queue
            if item.status == "pending"
        ][:limit]

    def get_by_assignee(self, user_id: str) -> List[ReviewQueueItem]:
        """Get items assigned to a specific user"""
        return [
            item for item in self.queue
            if item.assigned_to == user_id and item.status == "in_review"
        ]

    def get_overdue(self) -> List[ReviewQueueItem]:
        """Get overdue review items"""
        now = datetime.now()
        return [
            item for item in self.queue
            if item.due_date and datetime.fromisoformat(item.due_date) < now
            and item.status == "in_review"
        ]

    def get_statistics(self) -> Dict[str, Any]:
        """Get queue statistics"""
        return {
            "total": len(self.queue),
            "pending": len([i for i in self.queue if i.status == "pending"]),
            "in_review": len([i for i in self.queue if i.status == "in_review"]),
            "completed": len([i for i in self.queue if i.status == "completed"]),
            "overdue": len(self.get_overdue()),
        }
