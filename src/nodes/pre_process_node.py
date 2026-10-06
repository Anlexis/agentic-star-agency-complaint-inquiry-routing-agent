"""AgentCore Platform v1.0"""

# Node contract:
#  - Extend FunctionNode; implement execute(state) -> dict
#  - Return ONLY the fields this node changes (never the full state)
#  - Return AgentStatus enum constants — never plain strings
#  - Read input_context via state.get("input_context", {}) — read-only
#  - Never import from mediator/, api/, or other agents
#
# INS-C2-053 — PreProcessNode (outer backbone slot)
#
# This node owns the caller-data contract. Everything a caller can influence is
# either accepted here against an explicit bound, or refused here with a
# closed-set reason — nothing reaches the domain pipeline unexamined.
#
# The contract is a CLOSED set of fields:
#
#   complaint text          the request body's `input`
#   channel                 counter | phone | online
#   agency_id               [a-z0-9_]{1,32}, optional, rendered into the record
#   case_ref                [a-z0-9_]{1,32}, optional, rendered into the record
#   prior_complaint_count   integer 0-1000, optional, raises the severity score
#
# Two things about that list are load-bearing:
#
#   * every rendered caller string is an inert identifier. Free text never
#     reaches the routing record, so a caller cannot manufacture a line in it.
#   * prior_complaint_count is a number, so it goes through the finite+bounded
#     parser. NaN and Infinity parse cleanly through float() and arrive intact
#     through raw JSON, and every comparison against NaN is False — accepting
#     one would fail OPEN on a severity decision.
#
# An unknown key is dropped rather than ignored. Ignoring leaves it in
# input_context, where the framework's output gate scans it on the first node's
# result and turns a stray value into an opaque node-1 traceback.

import logging
from typing import Any, ClassVar, Dict, FrozenSet, Mapping, Optional

from framework.nodes.function_node import FunctionNode
from framework.schemas.agent_status import AgentStatus
from framework.schemas.trust_level import TrustLevel

from shared.utils.audit_logger import emit_trace_event

from src.schemas.vocabulary import WITHHELD_NOTICE_KEY
from src.services.service import (
    bounded_config_int,
    bounded_int,
    config_section,
    detect_credentials_in_structure,
    detect_instructions,
    detect_output_credentials,
    is_inert_token,
    screen_payload_for_instructions,
)

logger = logging.getLogger(__name__)

# Delivery channel of the complaint. A closed set: the value reaches the
# severity score and the routing record, so it cannot be free text.
VALID_CHANNELS: FrozenSet[str] = frozenset({"counter", "phone", "online"})
DEFAULT_CHANNEL = "online"

# The caller contract, as data. Anything not named here is dropped.
CONTEXT_FIELDS: FrozenSet[str] = frozenset({"channel", "agency_id", "case_ref", "prior_complaint_count"})

# Bounds for prior_complaint_count. The upper bound is a real-world ceiling on
# complaints filed against one policy, not a type limit.
PRIOR_COUNT_MIN = 0
PRIOR_COUNT_MAX = 1000

# Structural cap on the context mapping itself, so an oversized object is
# refused before any field is parsed.
MAX_CONTEXT_FIELDS = 16

# Default text bound; overridable via config/config.yaml `limits.max_input_chars`.
DEFAULT_MAX_INPUT_CHARS = 4000
MIN_ALLOWED_INPUT_CHARS = 200
MAX_ALLOWED_INPUT_CHARS = 20000


class PreProcessNode(FunctionNode):
    """Validate the caller contract before the domain pipeline sees any of it.

    Output (partial dict): validated_input, validated_context, rejection_reason,
    status.
    """

    # The external trust boundary sits here, matching the manifest's declared
    # entry level. The inner domain nodes deliberately do NOT re-assert it — see
    # the note in src/graph/domain_workflow_graph.py.
    required_trust_level: ClassVar[TrustLevel] = TrustLevel.VERIFIED_EXTERNAL

    def __init__(self, config: Optional[Mapping[str, Any]] = None) -> None:
        """Take the runtime config at construction time.

        The framework calls ``execute(state)`` with one argument, so a
        ``config`` parameter on ``execute`` is never supplied and any value read
        from it is dead. Configuration therefore arrives through the
        constructor, which is the only path that actually carries a value.
        """
        limits = config_section(config, "limits")
        self._max_input_chars = bounded_config_int(
            limits,
            "max_input_chars",
            DEFAULT_MAX_INPUT_CHARS,
            MIN_ALLOWED_INPUT_CHARS,
            MAX_ALLOWED_INPUT_CHARS,
        )

    @property
    def max_input_chars(self) -> int:
        """The resolved text bound — exposed so a test can assert it moved."""
        return self._max_input_chars

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _reject(self, reason: str, field: str, state: Dict[str, Any]) -> Dict[str, Any]:
        """Return the ERROR partial for a refused request.

        The reason is a closed-set label and the field is a fixed name. Neither
        the offending value nor a detector's matched text is recorded anywhere:
        the point of refusing a value is not to copy it into the audit trail.
        """
        emit_trace_event(
            "caller_contract_rejected",
            {"reason": reason, "field": field},
            state,
        )
        logger.warning("PreProcessNode: rejected request (%s on %s)", reason, field)
        return {
            "validated_input": None,
            "validated_context": None,
            "rejection_reason": reason,
            # A refused request never reaches post_process — the backbone routes
            # a non-success status straight to finalize — so the notice has to
            # be set here or the caller receives an empty envelope with no
            # reason at all. It is a closed-set label in a truthy container:
            # truthy because the envelope falls back to the raw result whenever
            # the formatted output is falsy.
            "formatted_output": {WITHHELD_NOTICE_KEY: reason},
            "result": None,
            "status": AgentStatus.ERROR,
            "error_log": [f"PreProcessNode: {reason} ({field})"],
        }

    def _validate_context(self, raw: object, state: Dict[str, Any]) -> Any:
        """Return the validated context mapping, or an ERROR partial.

        Unknown keys are DROPPED, not carried. A validator that merely ignores
        an undeclared key leaves it in input_context, and the framework's output
        gate scans every value of the first node's result — so a stray value
        detonates below this node rather than being refused at it.
        """
        if raw is None:
            raw = {}
        if not isinstance(raw, Mapping):
            return self._reject("context_not_an_object", "input_context", state)
        if len(raw) > MAX_CONTEXT_FIELDS:
            return self._reject("context_too_many_fields", "input_context", state)

        validated: Dict[str, Any] = {}

        channel = raw.get("channel", DEFAULT_CHANNEL)
        if not isinstance(channel, str) or channel.strip().lower() not in VALID_CHANNELS:
            return self._reject("channel_not_recognised", "input_context.channel", state)
        validated["channel"] = channel.strip().lower()

        for name in ("agency_id", "case_ref"):
            value = raw.get(name)
            if value is None:
                validated[name] = None
                continue
            if not is_inert_token(value):
                return self._reject("identifier_not_inert", f"input_context.{name}", state)
            validated[name] = value

        prior = raw.get("prior_complaint_count", 0)
        resolved = bounded_int(prior, PRIOR_COUNT_MIN, PRIOR_COUNT_MAX)
        if resolved is None:
            return self._reject(
                "value_not_finite_or_out_of_range",
                "input_context.prior_complaint_count",
                state,
            )
        validated["prior_complaint_count"] = resolved

        return validated

    # ------------------------------------------------------------------
    # Node body
    # ------------------------------------------------------------------

    def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        raw_context = state.get("input_context", {})  # read-only
        user_input = state.get("user_input", "")

        if not isinstance(user_input, str) or not user_input.strip():
            return self._reject("text_absent_or_empty", "input", state)
        if len(user_input) > self._max_input_chars:
            return self._reject("text_too_long", "input", state)

        # The template owns this refusal. Asserting only that "the framework's
        # gate refused" would make the guarantee conditional on a gate that can
        # be absent or configured off, and the payload would then reach the
        # answer path and return success.
        if detect_instructions(user_input):
            return self._reject("instruction_detected", "input", state)
        if detect_output_credentials(user_input):
            return self._reject("credential_shaped_value", "input", state)

        # Keys are caller data too, and a \u-escape is already decoded by the
        # time the payload is a Python object — so the screen runs after the
        # parse, over keys as well as values.
        if screen_payload_for_instructions(raw_context):
            return self._reject("instruction_detected", "input_context", state)
        if detect_credentials_in_structure(raw_context):
            return self._reject("credential_shaped_value", "input_context", state)

        validated_context = self._validate_context(raw_context, state)
        if not isinstance(validated_context, dict) or "channel" not in validated_context:
            # _validate_context already produced the ERROR partial.
            return validated_context

        text = user_input.strip()
        emit_trace_event(
            "caller_contract_accepted",
            {
                "channel": validated_context["channel"],
                "agency_id": validated_context.get("agency_id"),
                "case_ref": validated_context.get("case_ref"),
                "prior_complaint_count": validated_context["prior_complaint_count"],
                "text_length": len(text),
                # The text itself is deliberately absent: it is the complaint,
                # and this event is written before any masking has happened.
            },
            state,
        )
        return {
            "validated_input": text,
            "validated_context": validated_context,
            "rejection_reason": None,
            "status": AgentStatus.SUCCESS,
        }
