"""AgentCore Platform v1.0"""

# Node contract:
#  - Extend FunctionNode; implement execute(state) -> dict
#  - Return ONLY the fields this node changes (never the full state)
#  - Return AgentStatus enum constants — never plain strings
#  - Never import from mediator/, api/, or other agents
#
# INS-C2-053 — OutputValidateNode
# Final node of the inner domain graph. It owns ONE duty: the routing decision
# is complete before it is assembled. Every field a downstream handler needs to
# act on the complaint is present, or the decision does not exist.
#
# What this node deliberately does NOT do
# ---------------------------------------
# It does not enforce the closed-vocabulary or credential invariant. That is
# the outer PostProcessNode's single duty, and duplicating it here would make
# both copies unfalsifiable: removing either one alone would leave every test
# green, which is a way of buying less assurance with more code. The split is
# completeness here, boundary invariant there.
#
# ⚠️ S-3: _security_gate_output is @final on FunctionNode and overriding it
# raises TypeError at class-definition time. The extension hook is
# _extra_security_gate_output(self, output: dict) -> dict.
#
# Output state keys (partial dict):
#   result           : the assembled routing decision, or None on failure
#   out_of_scope     : bool
#   rejection_reason : closed-set label on failure
#   status           : AgentStatus.SUCCESS or AgentStatus.ERROR

import logging
from typing import Any, ClassVar, Dict, List, Mapping, Optional

from framework.nodes.function_node import FunctionNode
from framework.schemas.agent_status import AgentStatus
from framework.schemas.trust_level import TrustLevel

from shared.utils.audit_logger import emit_trace_event

logger = logging.getLogger(__name__)

# Fields the complaint path cannot produce a decision without.
REQUIRED_COMPLAINT_FIELDS = (
    "complaint_category",
    "severity_level",
    "routing_target_team",
    "routing_sla_hours",
    "routing_rationale",
)

# The out-of-scope path still produces a routing decision; it just needs less.
REQUIRED_OUT_OF_SCOPE_FIELDS = (
    "routing_target_team",
    "routing_sla_hours",
    "routing_rationale",
)


class OutputValidateNode(FunctionNode):
    """Assemble the routing decision, and refuse to assemble an incomplete one.

    Output (partial dict): result, out_of_scope, rejection_reason, status.
    """

    required_trust_level: ClassVar[TrustLevel] = TrustLevel.ANONYMOUS

    def __init__(self, config: Optional[Mapping[str, Any]] = None) -> None:
        self._config = dict(config or {})

    def _fail(self, state: Dict[str, Any], reason: str, missing: List[str]) -> Dict[str, Any]:
        """Return the ERROR partial, with the decision explicitly cleared.

        Clearing matters because the invocation envelope falls back to
        ``state["result"]`` when no formatted output is present — an error that
        leaves a half-built decision in state still ships it.
        """
        emit_trace_event(
            "routing_decision_incomplete",
            {"reason": reason, "missing_fields": missing},
            state,
        )
        logger.error("OutputValidateNode: %s (missing=%s)", reason, missing)
        return {
            "result": None,
            "rejection_reason": reason,
            "status": AgentStatus.ERROR,
            "error_log": [f"OutputValidateNode: {reason}"],
        }

    def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        # An upstream refusal is carried through as-is. Its reason is already a
        # closed-set label, so it can be reported without re-inspection.
        upstream_reason = state.get("rejection_reason")
        if upstream_reason:
            return self._fail(state, str(upstream_reason), [])

        out_of_scope = bool(state.get("out_of_scope"))
        required = REQUIRED_OUT_OF_SCOPE_FIELDS if out_of_scope else REQUIRED_COMPLAINT_FIELDS
        missing = [name for name in required if state.get(name) in (None, "")]
        if missing:
            return self._fail(state, "routing_incomplete", missing)

        decision: Dict[str, Any] = {
            "classification": {
                "category": state.get("complaint_category"),
                "confidence": state.get("classification_confidence"),
                "is_complaint": bool(state.get("is_complaint")),
            },
            "severity": {
                "level": state.get("severity_level"),
                "score": state.get("severity_score"),
                "modifiers": list(state.get("severity_modifiers") or []),
            },
            "routing": {
                "officer_id": state.get("routing_target_officer_id"),
                "team": state.get("routing_target_team"),
                "sla_hours": state.get("routing_sla_hours"),
                "rationale": state.get("routing_rationale"),
            },
            "case": {
                "agency_id": state.get("agency_id"),
                "case_ref": state.get("case_ref"),
                "channel": state.get("channel"),
            },
            "out_of_scope": out_of_scope,
        }

        emit_trace_event(
            "routing_decision_assembled",
            {
                "category": decision["classification"]["category"],
                "team": decision["routing"]["team"],
                "out_of_scope": out_of_scope,
            },
            state,
        )
        logger.info(
            "OutputValidateNode: decision assembled — category=%s team=%s oos=%s",
            decision["classification"]["category"],
            decision["routing"]["team"],
            out_of_scope,
        )
        return {
            "result": decision,
            "out_of_scope": out_of_scope,
            "rejection_reason": None,
            "status": AgentStatus.SUCCESS,
        }
