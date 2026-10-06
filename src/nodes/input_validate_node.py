"""AgentCore Platform v1.0"""

# Node contract:
#  - Extend FunctionNode; implement execute(state) -> dict
#  - Return ONLY the fields this node changes (never the full state)
#  - Return AgentStatus enum constants — never plain strings
#  - Never import from mediator/, api/, or other agents
#
# INS-C2-053 — InputValidateNode
# First domain node in the Classify -> Route -> Log pipeline.
#
# Responsibilities:
#   - mask the residual personal data the platform's own filter does not reach
#   - refuse a text that survives masking as nothing but redaction sentinels
#   - re-assert the instruction screen at the inner boundary
#
# Input state keys:
#   raw_input_text / user_input : the complaint text. The inner graph seeds
#       raw_input_text from the bridge-published context when present;
#       BaseGraph.invoke() always writes the boundary string to user_input, so
#       both are read and the first non-empty one wins.
#   channel : seeded by DomainWorkflowGraph._extra_initial_state()
#
# Output state keys (partial dict):
#   raw_input_text  : MASKED text — overwrites whatever was read
#   is_valid_input  : True on validation pass
#   rejection_reason: closed-set label on failure
#   status          : AgentStatus.SUCCESS or AgentStatus.ERROR
#
# Why a second masking layer exists
# ---------------------------------
# The platform masks personal-data shapes in user_input before any template
# code runs, but its word-boundary anchors are computed over \w, which includes
# Kana and Kanji — so a Japanese number written without spaces around it, which
# is the normal way to write it, does not match. This complaint corpus is
# Japanese. The patterns below therefore close the gap the platform filter
# leaves rather than duplicating what it already does.

import logging
import re
from typing import Any, ClassVar, Dict, Mapping, Optional

from framework.nodes.function_node import FunctionNode
from framework.schemas.agent_status import AgentStatus
from framework.schemas.trust_level import TrustLevel

from shared.utils.audit_logger import emit_trace_event

from src.services.service import (
    detect_instructions,
    detect_output_credentials,
    is_redaction_sentinel,
    sentinel_ratio,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Residual personal-data patterns (Japanese insurance-complaint context)
# ---------------------------------------------------------------------------
# Order matters: the more specific pattern masks first so a later, broader one
# cannot consume half of an already-masked span.

_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")

# Japanese telephone numbers: 090-1234-5678 / 03-1234-5678 / 0120-123-456 /
# +81-... Lookarounds are non-digit rather than \b, because \b is computed over
# \w and Kanji counts as a word character — so "電話090-1234-5678です" would not
# match a \b-anchored pattern at all.
_PHONE_RE = re.compile(r"(?<!\d)(?:\+81[-\s]?)?0\d{1,4}[-\s]?\d{1,4}[-\s]?\d{3,4}(?!\d)")

# Policy / certificate numbers, either behind their label or as a bare long
# identifier run.
_POLICY_LABEL_RE = re.compile(r"(?:証券番号|保険証券番号|証券No\.?|契約番号)[\s:：]*[A-Za-z0-9\-]+")
_POLICY_NUM_RE = re.compile(r"(?<![A-Za-z0-9])[A-Za-z]{0,3}\d{8,}(?![A-Za-z0-9])")

# Individual number (12 digits, written without separators in running text).
_MY_NUMBER_RE = re.compile(r"(?<!\d)\d{12}(?!\d)")

# Names introduced by an explicit label. An unlabelled honorific is deliberately
# NOT matched: titles and role phrases (部長, 取締役, 東京都知事) have exactly the
# same shape as a surname, and no denylist closes an open vocabulary.
_NAME_LABEL_RE = re.compile(r"(?:氏名|お名前|名前|契約者名)[\s:：]*[一-鿿゠-ヿA-Za-z]{1,16}")
_NAME_INTRO_RE = re.compile(r"[一-鿿゠-ヿ]{2,16}(?=と申します|と言います)")

_MASK = "[MASKED]"

# A text that is more than this fraction redaction sentinel carries no
# classifiable content. Certifying a classification of it would be certifying
# the redaction, not the complaint.
MAX_SENTINEL_RATIO = 0.6


def mask_residual_pii(text: str) -> str:
    """Return *text* with residual personal data masked.

    Applied before the masked text is written to State and before any audit
    payload is built, so nothing downstream and nothing in the audit trail ever
    holds the pre-mask form.
    """
    masked = _EMAIL_RE.sub(_MASK, text)
    masked = _POLICY_LABEL_RE.sub(_MASK, masked)
    masked = _NAME_LABEL_RE.sub(_MASK, masked)
    masked = _MY_NUMBER_RE.sub(_MASK, masked)
    masked = _PHONE_RE.sub(_MASK, masked)
    masked = _POLICY_NUM_RE.sub(_MASK, masked)
    masked = _NAME_INTRO_RE.sub(_MASK, masked)
    return masked


class InputValidateNode(FunctionNode):
    """Mask residual personal data and validate the complaint text.

    Output (partial dict): raw_input_text (masked), is_valid_input,
    rejection_reason, status.
    """

    # Inner domain nodes admit any caller the outer boundary already admitted.
    # See the trust note in src/graph/domain_workflow_graph.py — a stricter
    # level here does not add a boundary, it removes the pipeline.
    required_trust_level: ClassVar[TrustLevel] = TrustLevel.ANONYMOUS

    def __init__(self, config: Optional[Mapping[str, Any]] = None) -> None:
        """Accept a config mapping for symmetry with the other domain nodes.

        This node has no tunable, but the inner graph constructs every node the
        same way; a node that refused a config argument would make that uniform
        construction a special case.
        """
        self._config = dict(config or {})

    def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        raw_text = state.get("raw_input_text") or state.get("user_input") or ""

        if not isinstance(raw_text, str) or not raw_text.strip():
            emit_trace_event(
                "input_validation_failed",
                {"reason": "text_absent_or_empty"},
                state,
            )
            return {
                "is_valid_input": False,
                "rejection_reason": "text_absent_or_empty",
                "status": AgentStatus.ERROR,
                "error_log": ["InputValidateNode: text_absent_or_empty"],
            }

        # Re-assert the instruction screen at the inner boundary. The outer node
        # already ran it, but this node is also reachable directly (unit tests,
        # and any future caller of the inner graph), and a guarantee that only
        # holds when something upstream ran is not a guarantee.
        instruction_labels = detect_instructions(raw_text)
        if instruction_labels:
            emit_trace_event(
                "input_validation_failed",
                {"reason": "instruction_detected", "labels": instruction_labels},
                state,
            )
            return {
                "is_valid_input": False,
                "rejection_reason": "instruction_detected",
                "status": AgentStatus.ERROR,
                "error_log": ["InputValidateNode: instruction_detected"],
            }

        if detect_output_credentials(raw_text):
            emit_trace_event(
                "input_validation_failed",
                {"reason": "credential_shaped_value"},
                state,
            )
            return {
                "is_valid_input": False,
                "rejection_reason": "credential_shaped_value",
                "status": AgentStatus.ERROR,
                "error_log": ["InputValidateNode: credential_shaped_value"],
            }

        masked_text = mask_residual_pii(raw_text).strip()

        # A text that masks away to nothing but sentinels has no complaint left
        # in it. Classifying it would produce a confident category for a value
        # that is only a redaction marker.
        if not masked_text or is_redaction_sentinel(masked_text) or sentinel_ratio(masked_text) > MAX_SENTINEL_RATIO:
            emit_trace_event(
                "input_validation_failed",
                {
                    "reason": "text_fully_redacted",
                    "sentinel_ratio": round(sentinel_ratio(masked_text), 3),
                },
                state,
            )
            return {
                # Store the masked form so nothing pre-mask lingers downstream.
                "raw_input_text": masked_text or _MASK,
                "is_valid_input": False,
                "rejection_reason": "text_fully_redacted",
                "status": AgentStatus.ERROR,
                "error_log": ["InputValidateNode: text_fully_redacted"],
            }

        logger.info(
            "InputValidateNode: validated channel=%s (text masked, len=%d)",
            state.get("channel"),
            len(masked_text),
        )
        emit_trace_event(
            "input_validated",
            {
                "channel": state.get("channel"),
                "text_length": len(masked_text),
                "sentinel_ratio": round(sentinel_ratio(masked_text), 3),
                # The text is not in the payload: an audit record of a complaint
                # does not need the complaint, and this one is written to a sink
                # with a different retention policy from the routing record.
            },
            state,
        )

        return {
            "raw_input_text": masked_text,
            "is_valid_input": True,
            "rejection_reason": None,
            "status": AgentStatus.SUCCESS,
        }
