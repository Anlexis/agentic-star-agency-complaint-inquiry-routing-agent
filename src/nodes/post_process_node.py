"""AgentCore Platform v1.0"""

# Node contract:
#  - Extend FunctionNode; implement execute(state) -> dict
#  - Return ONLY the fields this node changes (never the full state)
#  - Return AgentStatus enum constants — never plain strings
#  - Never import from mediator/, api/, or other agents
#
# INS-C2-053 — PostProcessNode (outer backbone slot): the output boundary.
#
# The invariant it enforces
# -------------------------
# This template renders no monetary aggregate, so the precision grid other
# templates enforce does not apply here and is documented as not applicable in
# docs/02_design.md. The invariant that does apply is the one this template
# actually states:
#
#   every string in the routing decision is either a member of a closed
#   vocabulary (src/schemas/vocabulary.py) or a caller-supplied inert
#   [a-z0-9_]{1,32} identifier; every number is an integer inside its declared
#   range; no other key exists.
#
# That is what makes the decision safe to render, forward and store. It also
# closes the caller-controlled-output class by construction: there is no field
# through which free text can reach the record, so no newline can manufacture a
# line in it.
#
# Why the clearing, and not a raise
# ---------------------------------
# The invocation envelope is built as
# ``{"output": state["formatted_output"] or state["result"], ...}`` — and the
# fallback applies on the error path too. A gate that merely raises, or returns
# ERROR without clearing, therefore still ships the un-gated inner decision
# inside the error envelope. On violation this node returns ERROR, clears every
# output-bearing field, and puts a TRUTHY refusal notice in formatted_output: a
# falsy replacement would re-open the very fallback being closed.
#
# Why the credential screen is a union
# ------------------------------------
# The framework's @final output gate scans every value of whatever this node
# returns and RAISES on a credential pattern. A raise from there is caught by
# the node wrapper, which returns a bare error partial — discarding this node's
# clearing entirely. So a local pattern set narrower than the framework's is not
# a smaller net, it is a containment bypass. The screen therefore calls the
# framework's own detector as its floor and adds the shapes the framework does
# not carry. Deleting the local half would be the same bug pointing the other
# way.

import logging
from typing import Any, ClassVar, Dict, FrozenSet, List, Mapping, Optional, Tuple

from framework.nodes.function_node import FunctionNode
from framework.schemas.agent_status import AgentStatus
from framework.schemas.trust_level import TrustLevel

from shared.utils.audit_logger import emit_trace_event

from src.schemas.vocabulary import (
    COMPLAINT_CATEGORIES,
    CONFIDENCE_LEVELS,
    OFFICER_IDS,
    ROUTING_TEAMS,
    SEVERITY_LEVELS,
    SEVERITY_MODIFIERS,
    SEVERITY_SCORE_MAX,
    SEVERITY_SCORE_MIN,
    SLA_HOURS_MAX,
    SLA_HOURS_MIN,
    WITHHELD_NOTICE_KEY,
)
from src.services.service import (
    bounded_int,
    detect_credentials_in_structure,
    is_inert_token,
)

logger = logging.getLogger(__name__)

VALID_CHANNELS: FrozenSet[str] = frozenset({"counter", "phone", "online"})

# The rationale grammar: space-separated key=value pairs over an alphabet with
# no whitespace, no punctuation and no newline. Checking the SHAPE rather than
# re-deriving the exact string keeps the gate independent of the router's
# formatting while still making a free-text rationale impossible.
_RATIONALE_MAX_PAIRS = 8
_RATIONALE_KEY_CHARS = "abcdefghijklmnopqrstuvwxyz_"
_RATIONALE_VALUE_CHARS = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_*")


def _rationale_is_well_formed(value: object) -> bool:
    """True when *value* is a space-separated run of inert ``key=value`` pairs."""
    if not isinstance(value, str) or not value:
        return False
    pairs = value.split(" ")
    if not 1 <= len(pairs) <= _RATIONALE_MAX_PAIRS:
        return False
    for pair in pairs:
        key, sep, val = pair.partition("=")
        if not sep or not key or not val:
            return False
        if any(character not in _RATIONALE_KEY_CHARS for character in key):
            return False
        if any(character not in _RATIONALE_VALUE_CHARS for character in val):
            return False
    return True


def _in_closed_set(value: object, allowed: FrozenSet[str], optional: bool = False) -> bool:
    if value is None:
        return optional
    return isinstance(value, str) and value in allowed


def _bounded_or_none(value: object, low: int, high: int, optional: bool) -> bool:
    if value is None:
        return optional
    return bounded_int(value, low, high) is not None


class PostProcessNode(FunctionNode):
    """Enforce the output invariant, or withhold the decision entirely.

    Output (partial dict): formatted_output, result, rejection_reason, status.
    """

    # The outer boundary already admitted the caller at pre_process. Re-asserting
    # a higher level here would deny the very callers the entry contract accepts,
    # on the last node before the answer — which reads as a silent failure, not
    # as a boundary.
    required_trust_level: ClassVar[TrustLevel] = TrustLevel.ANONYMOUS

    def __init__(self, config: Optional[Mapping[str, Any]] = None) -> None:
        self._config = dict(config or {})

    # ------------------------------------------------------------------
    # Vocabulary check
    # ------------------------------------------------------------------

    def _vocabulary_violations(self, decision: object) -> List[str]:
        """Return the field paths that fall outside the closed vocabulary."""
        if not isinstance(decision, Mapping):
            return ["decision"]

        bad: List[str] = []
        expected_top = {"classification", "severity", "routing", "case", "out_of_scope"}
        extra = set(decision) - expected_top
        bad.extend(sorted(f"decision.{name}" for name in extra))
        missing = expected_top - set(decision)
        bad.extend(sorted(f"decision.{name}(missing)" for name in missing))
        if bad:
            return bad

        sections: Tuple[Tuple[str, Dict[str, Any]], ...] = tuple(
            (name, decision[name]) for name in ("classification", "severity", "routing", "case")
        )
        for name, section in sections:
            if not isinstance(section, Mapping):
                bad.append(f"decision.{name}")
        if bad:
            return bad

        classification = decision["classification"]
        severity = decision["severity"]
        routing = decision["routing"]
        case = decision["case"]

        if set(classification) != {"category", "confidence", "is_complaint"}:
            bad.append("decision.classification(shape)")
        if set(severity) != {"level", "score", "modifiers"}:
            bad.append("decision.severity(shape)")
        if set(routing) != {"officer_id", "team", "sla_hours", "rationale"}:
            bad.append("decision.routing(shape)")
        if set(case) != {"agency_id", "case_ref", "channel"}:
            bad.append("decision.case(shape)")
        if bad:
            return bad

        if not _in_closed_set(classification["category"], COMPLAINT_CATEGORIES, optional=True):
            bad.append("decision.classification.category")
        if not _in_closed_set(classification["confidence"], CONFIDENCE_LEVELS, optional=True):
            bad.append("decision.classification.confidence")
        if not isinstance(classification["is_complaint"], bool):
            bad.append("decision.classification.is_complaint")

        if not _in_closed_set(severity["level"], SEVERITY_LEVELS, optional=True):
            bad.append("decision.severity.level")
        if not _bounded_or_none(severity["score"], SEVERITY_SCORE_MIN, SEVERITY_SCORE_MAX, optional=True):
            bad.append("decision.severity.score")
        modifiers = severity["modifiers"]
        if not isinstance(modifiers, list) or any(not _in_closed_set(item, SEVERITY_MODIFIERS) for item in modifiers):
            bad.append("decision.severity.modifiers")

        if not _in_closed_set(routing["officer_id"], OFFICER_IDS, optional=True):
            bad.append("decision.routing.officer_id")
        if not _in_closed_set(routing["team"], ROUTING_TEAMS):
            bad.append("decision.routing.team")
        if not _bounded_or_none(routing["sla_hours"], SLA_HOURS_MIN, SLA_HOURS_MAX, optional=False):
            bad.append("decision.routing.sla_hours")
        if not _rationale_is_well_formed(routing["rationale"]):
            bad.append("decision.routing.rationale")

        for name in ("agency_id", "case_ref"):
            value = case[name]
            if value is not None and not is_inert_token(value):
                bad.append(f"decision.case.{name}")
        if not _in_closed_set(case["channel"], VALID_CHANNELS, optional=True):
            bad.append("decision.case.channel")

        if not isinstance(decision["out_of_scope"], bool):
            bad.append("decision.out_of_scope")

        return bad

    # ------------------------------------------------------------------
    # Withholding
    # ------------------------------------------------------------------

    def _withhold(self, state: Dict[str, Any], reason: str, detail: List[str]) -> Dict[str, Any]:
        """Return the contained ERROR partial.

        Every output-bearing field is cleared and the notice that replaces them
        is truthy. `detail` is a list of FIELD PATHS — never a value, never a
        detector's matched text, never a traceback.
        """
        emit_trace_event(
            "output_gate_withheld",
            {"reason": reason, "fields": sorted(detail)[:8]},
            state,
        )
        logger.error("PostProcessNode: withheld routing decision (%s)", reason)
        return {
            "formatted_output": {WITHHELD_NOTICE_KEY: reason},
            "result": None,
            "rejection_reason": reason,
            "status": AgentStatus.ERROR,
            "error_log": [f"PostProcessNode: {reason}"],
        }

    # ------------------------------------------------------------------
    # Node body
    # ------------------------------------------------------------------

    def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        # Total by construction: an unexpected exception inside the gate would
        # otherwise be caught by the node wrapper, which returns a bare error
        # partial and leaves state["result"] untouched — exactly the fallback
        # this node exists to close.
        try:
            decision = state.get("result")
            if decision is None:
                return self._withhold(state, "routing_incomplete", ["decision(absent)"])

            violations = self._vocabulary_violations(decision)
            if violations:
                return self._withhold(state, "output_not_on_closed_vocabulary", violations)

            credential_labels = detect_credentials_in_structure(decision)
            if credential_labels:
                return self._withhold(state, "output_credential_detected", credential_labels)

            emit_trace_event(
                "output_gate_passed",
                {
                    "team": decision["routing"]["team"],
                    "out_of_scope": decision["out_of_scope"],
                },
                state,
            )
            # result is replaced with the gated object rather than left as the
            # inner one, so the envelope's fallback can only ever serve content
            # this gate has already passed.
            return {
                "formatted_output": decision,
                "result": decision,
                "rejection_reason": None,
                "status": AgentStatus.SUCCESS,
            }
        except Exception as error:  # noqa: BLE001 — containment must be total
            logger.exception("PostProcessNode: gate raised (%s)", type(error).__name__)
            return self._withhold(state, "output_gate_error", ["gate"])
