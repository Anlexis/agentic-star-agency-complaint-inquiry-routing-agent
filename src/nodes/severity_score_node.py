"""AgentCore Platform v1.0"""

# Node contract:
#  - Extend FunctionNode; implement execute(state) -> dict
#  - Return ONLY the fields this node changes (never the full state)
#  - Return AgentStatus enum constants — never plain strings
#  - Never import from mediator/, api/, or other agents
#
# INS-C2-053 — SeverityScoreNode
# Third domain node. Additive rule-based severity scoring for a classified
# complaint: a category base score, plus fixed deltas for the escalation
# signals present in the masked text, the delivery channel, and the number of
# complaints already on file.
#
# Input state keys:
#   is_complaint, out_of_scope : skip and pass through when the input is
#                                out of scope
#   complaint_category         : the base score's key
#   raw_input_text             : MASKED text, scanned for escalation signals
#   channel                    : counter / phone / online
#   prior_complaint_count      : caller-supplied integer 0-1000
#
# Output state keys (partial dict):
#   severity_level     : critical / high / medium / low, or None when skipped
#   severity_score     : integer 0-100, or None when skipped
#   severity_modifiers : the names of the rules that fired, closed set
#   status             : AgentStatus.SUCCESS
#
# Two configurable deltas, both bounded
# -------------------------------------
# `severity.counter_channel_delta` and `severity.repeat_complaint_delta` come
# from config/config.yaml through the constructor. A declared value outside its
# range is dropped and the documented default applies — clamping would invent a
# number the operator never wrote and hide the mistake.

import logging
from typing import Any, ClassVar, Dict, List, Mapping, Optional

from framework.nodes.function_node import FunctionNode
from framework.schemas.agent_status import AgentStatus
from framework.schemas.trust_level import TrustLevel

from shared.utils.audit_logger import emit_trace_event

from src.schemas.vocabulary import SEVERITY_SCORE_MAX, SEVERITY_SCORE_MIN
from src.services.service import bounded_config_int, bounded_int, config_section

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Scoring table
# ---------------------------------------------------------------------------

BASE_SCORES: Dict[str, int] = {
    "claim_handling": 60,
    "solicitation_conduct": 70,
    "contract_explanation": 50,
    "premium_billing": 40,
    "policy_cancellation": 40,
    "customer_service": 20,
    "general_inquiry": 10,
    "unclassifiable": 30,
}

# Additive keyword rules. `name` is a member of SEVERITY_MODIFIERS, so a rule
# name reaching the rendered decision is already on the closed vocabulary.
KEYWORD_MODIFIERS: List[Dict[str, Any]] = [
    {"name": "regulatory", "keywords": ["金融庁", "行政", "苦情申出", "監督官庁"], "delta": 20},
    {"name": "legal_threat", "keywords": ["訴訟", "弁護士", "裁判", "法的措置"], "delta": 25},
    {"name": "urgency", "keywords": ["至急", "即日", "今すぐ", "緊急"], "delta": 10},
]

# A complaint delivered face to face reaches a person already at the counter.
DEFAULT_COUNTER_CHANNEL_DELTA = 5
COUNTER_DELTA_MIN = 0
COUNTER_DELTA_MAX = 50

# Each prior complaint on file adds this much, up to REPEAT_COUNT_CAP of them.
DEFAULT_REPEAT_COMPLAINT_DELTA = 5
REPEAT_DELTA_MIN = 0
REPEAT_DELTA_MAX = 30
REPEAT_COUNT_CAP = 3

# Severity level thresholds (inclusive lower bounds).
LEVEL_CRITICAL = 85
LEVEL_HIGH = 65
LEVEL_MEDIUM = 40


def level_from_score(score: int) -> str:
    """Map a 0-100 score onto the severity ladder."""
    if score >= LEVEL_CRITICAL:
        return "critical"
    if score >= LEVEL_HIGH:
        return "high"
    if score >= LEVEL_MEDIUM:
        return "medium"
    return "low"


class SeverityScoreNode(FunctionNode):
    """Additive severity scoring for a classified complaint.

    Output (partial dict): severity_level, severity_score, severity_modifiers,
    status.
    """

    required_trust_level: ClassVar[TrustLevel] = TrustLevel.ANONYMOUS

    def __init__(self, config: Optional[Mapping[str, Any]] = None) -> None:
        severity = config_section(config, "severity")
        self._counter_channel_delta = bounded_config_int(
            severity,
            "counter_channel_delta",
            DEFAULT_COUNTER_CHANNEL_DELTA,
            COUNTER_DELTA_MIN,
            COUNTER_DELTA_MAX,
        )
        self._repeat_complaint_delta = bounded_config_int(
            severity,
            "repeat_complaint_delta",
            DEFAULT_REPEAT_COMPLAINT_DELTA,
            REPEAT_DELTA_MIN,
            REPEAT_DELTA_MAX,
        )

    @property
    def counter_channel_delta(self) -> int:
        """Resolved counter-channel delta — exposed so a test can assert it moved."""
        return self._counter_channel_delta

    @property
    def repeat_complaint_delta(self) -> int:
        """Resolved repeat-complaint delta — exposed so a test can assert it moved."""
        return self._repeat_complaint_delta

    def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        if state.get("out_of_scope") or not state.get("is_complaint"):
            logger.info("SeverityScoreNode: out of scope — no score produced")
            emit_trace_event(
                "severity_scored",
                {"scored": False, "reason": "out_of_scope"},
                state,
            )
            return {
                "severity_level": None,
                "severity_score": None,
                "severity_modifiers": [],
                "status": AgentStatus.SUCCESS,
            }

        category = state.get("complaint_category") or "unclassifiable"
        text = state.get("raw_input_text") or ""
        channel = (state.get("channel") or "").strip().lower()

        score = int(BASE_SCORES.get(category, BASE_SCORES["unclassifiable"]))
        applied: List[str] = []

        for modifier in KEYWORD_MODIFIERS:
            if any(keyword in text for keyword in modifier["keywords"]):
                score += int(modifier["delta"])
                applied.append(str(modifier["name"]))

        if channel == "counter" and self._counter_channel_delta:
            score += self._counter_channel_delta
            applied.append("counter_channel")

        # prior_complaint_count reaches here already bounded by the caller
        # contract, but it is re-parsed rather than trusted: this node is also
        # reachable directly, and a number that arrives from anywhere else must
        # meet the same bound. An unparseable value contributes nothing instead
        # of raising — the complaint still gets routed.
        prior = bounded_int(state.get("prior_complaint_count"), 0, 1000)
        if prior and self._repeat_complaint_delta:
            score += min(prior, REPEAT_COUNT_CAP) * self._repeat_complaint_delta
            applied.append("repeat_complaint")

        score = max(SEVERITY_SCORE_MIN, min(SEVERITY_SCORE_MAX, score))
        level = level_from_score(score)

        logger.info(
            "SeverityScoreNode: category=%s score=%d level=%s modifiers=%s",
            category,
            score,
            level,
            applied,
        )
        emit_trace_event(
            "severity_scored",
            {
                "scored": True,
                "category": category,
                "score": score,
                "level": level,
                "modifiers": applied,
                "channel": channel or None,
            },
            state,
        )
        return {
            "severity_level": level,
            "severity_score": score,
            "severity_modifiers": applied,
            "status": AgentStatus.SUCCESS,
        }
