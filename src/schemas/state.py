"""AgentCore Platform v1.0"""

# ADR-005: State must be a flat TypedDict — never a Pydantic BaseModel.
# LangGraph checkpoints use msgpack serialization; Pydantic objects cause
# silent corruption. Extend AgentState with agent-specific fields only.
# Do NOT add credentials, secrets, tokens, or Pydantic models.
#
# INS-C2-053 — Insurance Agency Complaint & Inquiry Routing Classification Agent
# Two-layer nested Cat 2 (Chat / Classify) graph: outer backbone
# (AgentBaseGraph) + inner domain workflow (BaseGraph). Fields below cover
# both layers.
#
# ── Why this file declares NO framework field ────────────────────────────────
# A subclass may only ADD keys. Re-declaring an inherited key with a different
# annotation is not a documentation nicety — LangGraph builds one channel per
# TypedDict key and raises at compile time:
#
#     ValueError: Channel 'error_log' already exists with a different type
#
# AgentState declares `error_log` and `node_history` as
# `Annotated[list[str], operator.add]` (accumulating channels) and `status` as
# an `AgentStatus`. Re-stating them as plain `List` / `Optional[str]` — which
# this file used to do — makes the inner graph impossible to compile, so the
# whole domain pipeline was unreachable. Inherit them; never restate them.
#
# Regulatory note: every complaint must produce a routing decision with a
# human-readable routing_rationale, which is audit-logged in RouteDecideNode.
# The routing decision is the loggable artefact.
#
# Confidentiality / PII note: raw_input_text is ALWAYS the MASKED version —
# InputValidateNode masks phone / email / name / policy-number PII before
# writing it to State. The pre-mask original text is NEVER persisted in State
# and NEVER passed to emit_trace_event.
#
# Secrets note: no credentials, tokens, or internal system passwords are stored
# in any State key.

from typing import Any, Dict, List, Optional

from framework.schemas.agent_state import AgentState


class State(AgentState):
    """Flat TypedDict for INS-C2-053.

    All shared fields (user_input, input_context, validated_input, status,
    session_id, node_history, error_log, trace_id, hitl_*, ...) are inherited
    from AgentState and are deliberately NOT restated here.

    Cross-node state-key contract (producer -> consumer):

    | Key                         | Producer                            | Consumer                                   |
    |-----------------------------|-------------------------------------|--------------------------------------------|
    | validated_context           | PreProcessNode (outer)              | DomainWorkflowGraphNode.extract_input      |
    | raw_input_text              | InputValidateNode (masked)          | ComplaintClassifyNode, SeverityScoreNode   |
    | channel                     | inner _extra_initial_state (bridge) | SeverityScoreNode, RouteDecideNode         |
    | agency_id / case_ref        | inner _extra_initial_state (bridge) | RouteDecideNode, OutputValidateNode        |
    | prior_complaint_count       | inner _extra_initial_state (bridge) | SeverityScoreNode                          |
    | is_valid_input              | InputValidateNode                   | ComplaintClassifyNode                      |
    | complaint_category          | ComplaintClassifyNode               | RouteDecideNode, OutputValidateNode        |
    | classification_confidence   | ComplaintClassifyNode               | OutputValidateNode                         |
    | is_complaint                | ComplaintClassifyNode               | SeverityScoreNode, RouteDecideNode, Output |
    | severity_level              | SeverityScoreNode                   | RouteDecideNode, OutputValidateNode        |
    | severity_score              | SeverityScoreNode                   | RouteDecideNode, OutputValidateNode        |
    | severity_modifiers          | SeverityScoreNode                   | OutputValidateNode                         |
    | routing_target_officer_id   | RouteDecideNode                     | OutputValidateNode                         |
    | routing_target_team         | RouteDecideNode                     | OutputValidateNode                         |
    | routing_sla_hours           | RouteDecideNode                     | OutputValidateNode                         |
    | routing_rationale           | RouteDecideNode                     | OutputValidateNode                         |
    | out_of_scope                | ComplaintClassifyNode               | SeverityScore, RouteDecide, OutputValidate |
    | rejection_reason            | any node                            | OutputValidateNode (closed-set label only) |
    """

    # ------------------------------------------------------------------
    # PreProcessNode (outer backbone) — the caller-data contract
    # ------------------------------------------------------------------

    # The caller fields that survived validation, already bounded and inert.
    # Carried to the inner graph through src/graph/context_bridge.py because
    # GraphNode.execute() forwards only a single string to the subgraph.
    validated_context: Optional[Dict[str, Any]]

    # ------------------------------------------------------------------
    # InputValidateNode — produces the sanitized (masked) text + validity
    # ------------------------------------------------------------------

    # Sanitized complaint / inquiry text with all PII masked. InputValidateNode
    # masks phone / email / name / policy numbers and stores ONLY the masked
    # result here. The pre-mask original is never persisted.
    raw_input_text: Optional[str]

    # Channel metadata: "counter" / "phone" / "online".
    channel: Optional[str]

    # Caller-supplied inert identifiers, rendered into the routing record.
    agency_id: Optional[str]
    case_ref: Optional[str]

    # Number of prior complaints already on file for this policyholder (0-1000).
    # A bounded caller-supplied integer that raises the severity score.
    prior_complaint_count: Optional[int]

    # True when input passed sanitization + structural validation.
    is_valid_input: bool

    # ------------------------------------------------------------------
    # ComplaintClassifyNode — deterministic keyword classification
    # ------------------------------------------------------------------

    # Classified complaint category (None when out_of_scope).
    complaint_category: Optional[str]

    # Classification confidence: "high" / "medium" / "low".
    classification_confidence: Optional[str]

    # True for a genuine complaint; False routes via out_of_scope=True.
    is_complaint: bool

    # ------------------------------------------------------------------
    # SeverityScoreNode — deterministic additive rule scoring
    # ------------------------------------------------------------------

    # Severity level: "critical" / "high" / "medium" / "low" (None if oos).
    severity_level: Optional[str]

    # Numeric severity score 0-100 (None if out_of_scope).
    severity_score: Optional[int]

    # Names of the scoring modifiers that fired, from a closed set.
    severity_modifiers: Optional[List[str]]

    # ------------------------------------------------------------------
    # RouteDecideNode — routing-table lookup + audit
    # ------------------------------------------------------------------

    # Compliance officer ID the complaint is routed to (may be None).
    routing_target_officer_id: Optional[str]

    # Routing team identifier (always set, even on the out_of_scope path).
    routing_target_team: Optional[str]

    # SLA in hours for the resolved routing decision.
    routing_sla_hours: Optional[int]

    # Machine-readable routing rationale, assembled from closed-set tokens only
    # so no caller-controlled text can reach the rendered decision.
    routing_rationale: Optional[str]

    # ------------------------------------------------------------------
    # Control / routing flags
    # ------------------------------------------------------------------

    # True for a non-complaint inquiry or an unclassifiable input.
    # Out-of-scope is NOT a separate status — status stays SUCCESS and the
    # out_of_scope path still produces a routing decision (general desk).
    out_of_scope: bool

    # Closed-set failure label (never caller text, never a matched value).
    rejection_reason: Optional[str]
