"""AgentCore Platform v1.0"""

# The closed vocabularies of the routing decision.
#
# The output invariant this template states — and the output gate enforces — is:
#
#   every string in the routing decision is a member of one of the sets below,
#   or a caller-supplied inert [a-z0-9_]{1,32} identifier; every number is an
#   integer inside its declared range.
#
# That is what makes the decision safe to render and safe to log. It is also
# what a single shared definition buys: the classifier, the router and the gate
# read the same constants, so "the gate allows what the router produces" is
# true by construction rather than by two lists agreeing today.
#
# Adding a category or a team means adding it here, once. A value the router
# could emit that is not here is a gate violation, which is the correct and
# visible failure — not a silent widening of what ships.

from typing import Dict, FrozenSet

# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

# Complaint categories. The first six are complaint categories proper; the
# seventh is the non-complaint inquiry class and the eighth is the explicit
# "recognised as text, not recognised as a category" outcome.
COMPLAINT_CATEGORIES: FrozenSet[str] = frozenset(
    {
        "claim_handling",
        "contract_explanation",
        "solicitation_conduct",
        "premium_billing",
        "policy_cancellation",
        "customer_service",
        "general_inquiry",
        "unclassifiable",
    }
)

CONFIDENCE_LEVELS: FrozenSet[str] = frozenset({"high", "medium", "low"})

# ---------------------------------------------------------------------------
# Severity
# ---------------------------------------------------------------------------

SEVERITY_LEVELS: FrozenSet[str] = frozenset({"critical", "high", "medium", "low"})

SEVERITY_SCORE_MIN = 0
SEVERITY_SCORE_MAX = 100

# Names of the additive scoring rules. Only these can appear in the decision's
# modifier list, so the list cannot become a channel for arbitrary text.
SEVERITY_MODIFIERS: FrozenSet[str] = frozenset(
    {"regulatory", "legal_threat", "urgency", "counter_channel", "repeat_complaint"}
)

# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------

ROUTING_TEAMS: FrozenSet[str] = frozenset(
    {
        "claims_escalation",
        "claims_team",
        "solicitation_review",
        "contract_support",
        "billing_team",
        "customer_relations",
        "compliance_general",
        "general_inquiry_desk",
    }
)

OFFICER_IDS: FrozenSet[str] = frozenset({"compliance_chief", "compliance_officer_a", "compliance_officer_b"})

SLA_HOURS_MIN = 1
SLA_HOURS_MAX = 720

# ---------------------------------------------------------------------------
# Rationale grammar
# ---------------------------------------------------------------------------
#
# The rationale is assembled from a fixed set of templates and closed-set
# substitutions, never by concatenating caller text. A caller therefore cannot
# add a line to the record, and a newline cannot manufacture one.

RATIONALE_TEMPLATES: Dict[str, str] = {
    "matched_rule": "category={category} severity={severity} team={team} sla_hours={sla}",
    "fallback_rule": "category={category} severity={severity} team={team} sla_hours={sla} rule=fallback",
    "out_of_scope": "category={category} team={team} sla_hours={sla} rule=non_complaint_inquiry",
}

RATIONALE_RULE_LABELS: FrozenSet[str] = frozenset({"matched_rule", "fallback_rule", "out_of_scope"})

# ---------------------------------------------------------------------------
# Refusal labels
# ---------------------------------------------------------------------------
#
# Every refusal anywhere in the pipeline reports one of these. A closed set
# means a refusal reason can be shown to a caller without re-checking whether
# it happens to embed something the caller sent.

# The single key of the refusal notice that replaces a withheld decision. It is
# defined here, once, because three different places produce that notice — the
# caller-contract node, the subgraph boundary and the output gate — and a
# caller should not have to recognise three spellings of "no decision".
#
# The notice is a non-empty mapping on purpose: the invocation envelope falls
# back to the raw, un-gated result whenever the formatted output is falsy, so a
# refusal that returned an empty string or an empty dict would re-open exactly
# the channel it is closing.
WITHHELD_NOTICE_KEY = "routing_decision_withheld"

REJECTION_REASONS: FrozenSet[str] = frozenset(
    {
        "text_absent_or_empty",
        "text_too_long",
        "text_fully_redacted",
        "instruction_detected",
        "credential_shaped_value",
        "context_not_an_object",
        "context_too_many_fields",
        "channel_not_recognised",
        "identifier_not_inert",
        "value_not_finite_or_out_of_range",
        "routing_incomplete",
        "output_not_on_closed_vocabulary",
        "output_credential_detected",
        "output_gate_error",
        "upstream_error",
    }
)
