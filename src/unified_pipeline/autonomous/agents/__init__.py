"""
Autonomous Agents

자율 에이전트 구현체들

Discovery Phase:
- TDAExpertAutonomousAgent
- SchemaAnalystAutonomousAgent
- ValueMatcherAutonomousAgent
- EntityClassifierAutonomousAgent
- RelationshipDetectorAutonomousAgent

Refinement Phase:
- OntologyArchitectAutonomousAgent
- ConflictResolverAutonomousAgent
- QualityJudgeAutonomousAgent
- SemanticValidatorAutonomousAgent

Governance Phase:
- GovernanceStrategistAutonomousAgent
- ActionPrioritizerAutonomousAgent
- RiskAssessorAutonomousAgent
- PolicyGeneratorAutonomousAgent
"""

from .discovery import (
    DataAnalystAutonomousAgent,  # v13.0: LLM-First 데이터 분석
    TDAExpertAutonomousAgent,
    SchemaAnalystAutonomousAgent,
    ValueMatcherAutonomousAgent,
    EntityClassifierAutonomousAgent,
    RelationshipDetectorAutonomousAgent,
)
from .refinement import (
    OntologyArchitectAutonomousAgent,
    ConflictResolverAutonomousAgent,
    QualityJudgeAutonomousAgent,
    SemanticValidatorAutonomousAgent,
)
from .governance import (
    GovernanceStrategistAutonomousAgent,
    ActionPrioritizerAutonomousAgent,
    RiskAssessorAutonomousAgent,
    PolicyGeneratorAutonomousAgent,
)

# 에이전트 타입별 클래스 매핑
AGENT_TYPE_MAP = {
    # Discovery
    "data_analyst": DataAnalystAutonomousAgent,  # v13.0: LLM-First
    "tda_expert": TDAExpertAutonomousAgent,
    "schema_analyst": SchemaAnalystAutonomousAgent,
    "value_matcher": ValueMatcherAutonomousAgent,
    "entity_classifier": EntityClassifierAutonomousAgent,
    "relationship_detector": RelationshipDetectorAutonomousAgent,
    # Refinement
    "ontology_architect": OntologyArchitectAutonomousAgent,
    "conflict_resolver": ConflictResolverAutonomousAgent,
    "quality_judge": QualityJudgeAutonomousAgent,
    "semantic_validator": SemanticValidatorAutonomousAgent,
    # Governance
    "governance_strategist": GovernanceStrategistAutonomousAgent,
    "action_prioritizer": ActionPrioritizerAutonomousAgent,
    "risk_assessor": RiskAssessorAutonomousAgent,
    "policy_generator": PolicyGeneratorAutonomousAgent,
}

__all__ = [
    # Discovery
    "DataAnalystAutonomousAgent",
    "TDAExpertAutonomousAgent",
    "SchemaAnalystAutonomousAgent",
    "ValueMatcherAutonomousAgent",
    "EntityClassifierAutonomousAgent",
    "RelationshipDetectorAutonomousAgent",
    # Refinement
    "OntologyArchitectAutonomousAgent",
    "ConflictResolverAutonomousAgent",
    "QualityJudgeAutonomousAgent",
    "SemanticValidatorAutonomousAgent",
    # Governance
    "GovernanceStrategistAutonomousAgent",
    "ActionPrioritizerAutonomousAgent",
    "RiskAssessorAutonomousAgent",
    "PolicyGeneratorAutonomousAgent",
    # Map
    "AGENT_TYPE_MAP",
]
