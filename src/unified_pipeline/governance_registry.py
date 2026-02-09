"""
Governance Registry

Governance decisions, action backlog, policy rules, and business insights.
Extracted from SharedContext to reduce God Object complexity.
"""

from typing import Dict, List, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from .shared_context import GovernanceDecision


class GovernanceRegistry:
    """
    Governance decision and action management.

    Manages:
    - governance_decisions
    - action_backlog
    - policy_rules
    - business_insights
    """

    def __init__(self) -> None:
        self.governance_decisions: List["GovernanceDecision"] = []
        self.action_backlog: List[Dict[str, Any]] = []
        self.policy_rules: List[Dict[str, Any]] = []
        self.business_insights: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Bind helpers
    # ------------------------------------------------------------------

    def bind(
        self,
        governance_decisions: List["GovernanceDecision"],
        action_backlog: List[Dict[str, Any]],
        policy_rules: List[Dict[str, Any]],
        business_insights: List[Dict[str, Any]],
    ) -> None:
        """Bind to the dataclass-owned containers."""
        self.governance_decisions = governance_decisions
        self.action_backlog = action_backlog
        self.policy_rules = policy_rules
        self.business_insights = business_insights

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_governance_decision(self, decision: "GovernanceDecision") -> None:
        self.governance_decisions.append(decision)

    def get_pending_actions(self) -> List[Dict[str, Any]]:
        return [a for a in self.action_backlog if a.get("status") == "pending"]
