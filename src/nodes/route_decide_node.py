"""AgentCore Platform v1.0"""

# Node contract:
#  - Extend FunctionNode; implement execute(state) -> dict
#  - Return ONLY the fields this node changes (never the full state)
#  - Return AgentStatus enum constants — never plain strings
#  - Never import from mediator/, api/, or other agents
#
# INS-C2-053 — RouteDecideNode
# Fourth domain node and the reason the template exists: it maps
# (category x severity) onto a compliance officer, a team and an SLA, and
# audit-logs the decision. The routing decision is the loggable artefact the
# complaint-handling obligation asks for, so this node emits it explicitly
# rather than relying on the framework's lifecycle events.
#
# Input state keys:
#   out_of_scope, is_complaint : bool
#   complaint_category         : member of COMPLAINT_CATEGORIES
#   severity_level             : member of SEVERITY_LEVELS
#   severity_score             : int
#   channel, agency_id, case_ref : caller context, already inert
#
# Output state keys (partial dict):
#   routing_target_officer_id : member of OFFICER_IDS, or None
#   routing_target_team       : member of ROUTING_TEAMS — always set
#   routing_sla_hours         : int within the declared SLA range
#   routing_rationale         : assembled from RATIONALE_TEMPLATES only
#   status                    : AgentStatus.SUCCESS
#
# The routing table is a compiled-in constant, not configuration
# ---------------------------------------------------------------------------
# It used to be advertised as overridable through an execute(state, config)
# argument the framework never supplies, so no override could ever apply. It is
# not simply re-plumbed to the constructor, because a team or officer name
# arriving from a YAML file would be a string outside the closed vocabulary the
# output gate enforces — the gate would then have to either reject a valid
# operator configuration or stop enforcing the vocabulary. Only the fallback
# SLA, a bounded integer, is configurable.

import logging
from typing import Any, ClassVar, Dict, List, Mapping, Optional

from framework.nodes.function_node import FunctionNode
from framework.schemas.agent_status import AgentStatus
from framework.schemas.trust_level import TrustLevel

from shared.utils.audit_logger import emit_trace_event

from src.schemas.vocabulary import (
    RATIONALE_TEMPLATES,
    SLA_HOURS_MAX,
    SLA_HOURS_MIN,
)
from src.services.service import bounded_config_int, config_section

logger = logging.getLogger(__name__)

TEMPLATE_ID = "INS-C2-053"

# ---------------------------------------------------------------------------
# Routing table — first match wins, exact severity before the catch-all
# ---------------------------------------------------------------------------

ROUTING_RULES: List[Dict[str, Any]] = [
    # claim_handling
    {
        "category": "claim_handling",
        "severity": "critical",
        "officer_id": "compliance_chief",
        "team": "claims_escalation",
        "sla_hours": 4,
    },
    {
        "category": "claim_handling",
        "severity": "high",
        "officer_id": "compliance_officer_a",
        "team": "claims_team",
        "sla_hours": 24,
    },
    {"category": "claim_handling", "severity": "*", "officer_id": None, "team": "claims_team", "sla_hours": 72},
    # solicitation_conduct — conduct findings go to the chief at any severity
    # above the catch-all, because the obligation is on the conduct itself.
    {
        "category": "solicitation_conduct",
        "severity": "critical",
        "officer_id": "compliance_chief",
        "team": "solicitation_review",
        "sla_hours": 4,
    },
    {
        "category": "solicitation_conduct",
        "severity": "high",
        "officer_id": "compliance_chief",
        "team": "solicitation_review",
        "sla_hours": 4,
    },
    {
        "category": "solicitation_conduct",
        "severity": "*",
        "officer_id": "compliance_officer_b",
        "team": "solicitation_review",
        "sla_hours": 48,
    },
    # contract_explanation — SLA scales with severity
    {
        "category": "contract_explanation",
        "severity": "critical",
        "officer_id": None,
        "team": "contract_support",
        "sla_hours": 8,
    },
    {
        "category": "contract_explanation",
        "severity": "high",
        "officer_id": None,
        "team": "contract_support",
        "sla_hours": 24,
    },
    {
        "category": "contract_explanation",
        "severity": "*",
        "officer_id": None,
        "team": "contract_support",
        "sla_hours": 72,
    },
    # premium_billing
    {
        "category": "premium_billing",
        "severity": "critical",
        "officer_id": "compliance_officer_a",
        "team": "billing_team",
        "sla_hours": 48,
    },
    {"category": "premium_billing", "severity": "*", "officer_id": None, "team": "billing_team", "sla_hours": 48},
    # policy_cancellation
    {
        "category": "policy_cancellation",
        "severity": "*",
        "officer_id": None,
        "team": "contract_support",
        "sla_hours": 48,
    },
    # customer_service
    {
        "category": "customer_service",
        "severity": "*",
        "officer_id": None,
        "team": "customer_relations",
        "sla_hours": 72,
    },
]

# Used when no rule matches at all — a complaint with no route is the failure
# this template exists to prevent, so there is always a route.
FALLBACK_TEAM = "compliance_general"
FALLBACK_OFFICER_ID = None

# Non-complaint / unclassifiable input goes to the general enquiry desk.
GENERAL_DESK_TEAM = "general_inquiry_desk"
GENERAL_DESK_OFFICER_ID = None

DEFAULT_FALLBACK_SLA_HOURS = 72


def match_rule(category: str, severity: str) -> Optional[Dict[str, Any]]:
    """Return the first rule for (category, severity), exact severity first."""
    for rule in ROUTING_RULES:
        if rule["category"] == category and rule["severity"] == severity:
            return rule
    for rule in ROUTING_RULES:
        if rule["category"] == category and rule["severity"] == "*":
            return rule
    return None


class RouteDecideNode(FunctionNode):
    """Routing-table lookup plus the audit record of the decision.

    Output (partial dict): routing_target_officer_id, routing_target_team,
    routing_sla_hours, routing_rationale, status.
    """

    required_trust_level: ClassVar[TrustLevel] = TrustLevel.ANONYMOUS

    def __init__(self, config: Optional[Mapping[str, Any]] = None) -> None:
        routing = config_section(config, "routing")
        self._fallback_sla_hours = bounded_config_int(
            routing,
            "fallback_sla_hours",
            DEFAULT_FALLBACK_SLA_HOURS,
            SLA_HOURS_MIN,
            SLA_HOURS_MAX,
        )

    @property
    def fallback_sla_hours(self) -> int:
        """Resolved fallback SLA — exposed so a test can assert it moved."""
        return self._fallback_sla_hours

    def _audit(self, state: Dict[str, Any], payload: Dict[str, Any]) -> None:
        """Write the routing decision to the audit trail.

        The payload carries the decision and the closed-set values it was made
        from. It does NOT carry the complaint text: this record is the routing
        artefact, and the text is neither needed to reconstruct the decision nor
        appropriate to duplicate into a second store.
        """
        emit_trace_event("routing_decision", {"template_id": TEMPLATE_ID, **payload}, state)

    def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        agency_id = state.get("agency_id")
        case_ref = state.get("case_ref")
        channel = state.get("channel")

        if state.get("out_of_scope"):
            team = GENERAL_DESK_TEAM
            officer_id = GENERAL_DESK_OFFICER_ID
            sla = self._fallback_sla_hours
            category = state.get("complaint_category") or "unclassifiable"
            rationale = RATIONALE_TEMPLATES["out_of_scope"].format(category=category, team=team, sla=sla)
            self._audit(
                state,
                {
                    "event": "routing_decision",
                    "rule": "out_of_scope",
                    "category": category,
                    "severity_level": state.get("severity_level"),
                    "channel": channel,
                    "agency_id": agency_id,
                    "case_ref": case_ref,
                    "routing_target_team": team,
                    "routing_target_officer_id": officer_id,
                    "sla_hours": sla,
                },
            )
            return {
                "routing_target_officer_id": officer_id,
                "routing_target_team": team,
                "routing_sla_hours": sla,
                "routing_rationale": rationale,
                "status": AgentStatus.SUCCESS,
            }

        category = state.get("complaint_category") or "unclassifiable"
        severity = state.get("severity_level") or "low"
        rule = match_rule(category, severity)

        if rule is not None:
            officer_id = rule["officer_id"]
            team = rule["team"]
            sla = int(rule["sla_hours"])
            rule_label = "matched_rule"
        else:
            officer_id = FALLBACK_OFFICER_ID
            team = FALLBACK_TEAM
            sla = self._fallback_sla_hours
            rule_label = "fallback_rule"

        rationale = RATIONALE_TEMPLATES[rule_label].format(category=category, severity=severity, team=team, sla=sla)
        logger.info("RouteDecideNode: %s", rationale)
        self._audit(
            state,
            {
                "event": "routing_decision",
                "rule": rule_label,
                "category": category,
                "severity_level": severity,
                "severity_score": state.get("severity_score"),
                "channel": channel,
                "agency_id": agency_id,
                "case_ref": case_ref,
                "routing_target_team": team,
                "routing_target_officer_id": officer_id,
                "sla_hours": sla,
            },
        )
        return {
            "routing_target_officer_id": officer_id,
            "routing_target_team": team,
            "routing_sla_hours": sla,
            "routing_rationale": rationale,
            "status": AgentStatus.SUCCESS,
        }
