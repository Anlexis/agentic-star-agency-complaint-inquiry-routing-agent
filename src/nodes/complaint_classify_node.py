"""AgentCore Platform v1.0"""

# Node contract:
#  - Extend FunctionNode; implement execute(state) -> dict
#  - Return ONLY the fields this node changes (never the full state)
#  - Return AgentStatus enum constants — never plain strings
#  - Never import from mediator/, api/, or other agents
#
# INS-C2-053 — ComplaintClassifyNode
# Second domain node. Deterministic keyword classification — no model is
# invoked, which is why config/agent.yaml declares generation_mode:
# deterministic. It sorts the masked text into one of the complaint categories
# and decides whether the input is a complaint at all.
#
# Input state keys:
#   raw_input_text : masked text from InputValidateNode
#   is_valid_input : skip classification when False
#
# Output state keys (partial dict):
#   complaint_category        : member of COMPLAINT_CATEGORIES
#   classification_confidence : high / medium / low
#   is_complaint              : bool
#   out_of_scope              : bool
#   status                    : AgentStatus.SUCCESS — out-of-scope is not an error
#
# The keyword tables live in code rather than in config/config.yaml. They used
# to be advertised as operator-overridable through an execute(state, config)
# argument that the framework never supplies, so the override was unreachable;
# and a category name arriving from configuration would be a string outside the
# closed vocabulary the output gate enforces. Keeping them here makes both
# problems go away instead of moving them.

import logging
from typing import Any, ClassVar, Dict, FrozenSet, List, Mapping, Optional, Tuple

from framework.nodes.function_node import FunctionNode
from framework.schemas.agent_status import AgentStatus
from framework.schemas.trust_level import TrustLevel

from shared.utils.audit_logger import emit_trace_event

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Category keyword tables
# ---------------------------------------------------------------------------

CATEGORY_KEYWORDS: Dict[str, List[str]] = {
    "claim_handling": ["保険金", "支払い", "査定", "給付金", "支払拒否", "免責"],
    "contract_explanation": ["重要事項", "説明不備", "契約内容", "説明義務", "聞いていない"],
    "solicitation_conduct": ["勧誘", "募集", "強引", "不適切", "押し売り", "営業"],
    "premium_billing": ["保険料", "請求", "引き落とし", "二重請求", "口座", "未払い"],
    "policy_cancellation": ["解約", "払戻", "解約返戻金", "クーリングオフ", "途中解約"],
    "customer_service": ["窓口", "対応", "態度", "サービス", "電話", "待たされた"],
    "general_inquiry": ["教えて", "知りたい", "確認したい", "問い合わせ", "案内", "手続き方法"],
}

# Words that mark a genuine complaint. Their presence overrides a dominant
# general-inquiry match.
COMPLAINT_SIGNALS: List[str] = [
    "苦情",
    "クレーム",
    "不満",
    "改善",
    "謝罪",
    "納得できない",
    "おかしい",
    "ひどい",
    "困っている",
]

# Categories that, dominant and without a complaint signal, are not complaints.
NON_COMPLAINT_CATEGORIES: FrozenSet[str] = frozenset({"general_inquiry"})

UNCLASSIFIABLE = "unclassifiable"


def _score_categories(text: str) -> Dict[str, int]:
    """Return ``{category: hit_count}`` for every category with at least one hit."""
    scores: Dict[str, int] = {}
    for category, keywords in CATEGORY_KEYWORDS.items():
        hits = sum(1 for keyword in keywords if keyword in text)
        if hits > 0:
            scores[category] = hits
    return scores


class ComplaintClassifyNode(FunctionNode):
    """Deterministic keyword classification of the masked complaint text.

    Output (partial dict): complaint_category, classification_confidence,
    is_complaint, out_of_scope, status.
    """

    required_trust_level: ClassVar[TrustLevel] = TrustLevel.ANONYMOUS

    def __init__(self, config: Optional[Mapping[str, Any]] = None) -> None:
        self._config = dict(config or {})

    def _emit(
        self,
        state: Dict[str, Any],
        category: Optional[str],
        confidence: Optional[str],
        is_complaint: bool,
        out_of_scope: bool,
        hits: int,
    ) -> None:
        """Record the classification outcome. Closed-set values only, no text."""
        emit_trace_event(
            "complaint_classified",
            {
                "category": category,
                "confidence": confidence,
                "is_complaint": is_complaint,
                "out_of_scope": out_of_scope,
                "keyword_hits": hits,
            },
            state,
        )

    def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        if not state.get("is_valid_input"):
            # Upstream refused the text. Classification of a refused input would
            # be an answer about something that was never accepted.
            logger.info("ComplaintClassifyNode: is_valid_input=False — out of scope")
            self._emit(state, None, None, False, True, 0)
            return {
                "complaint_category": None,
                "classification_confidence": None,
                "is_complaint": False,
                "out_of_scope": True,
                "status": AgentStatus.SUCCESS,
            }

        text = state.get("raw_input_text") or ""
        scores = _score_categories(text)
        has_complaint_signal = any(signal in text for signal in COMPLAINT_SIGNALS)

        if not scores:
            # Recognised as text, not recognised as a category. Either way it
            # still gets a routing decision — the general desk — because an
            # unrouted complaint is the failure this template exists to prevent.
            logger.info("ComplaintClassifyNode: no keyword match — unclassifiable")
            self._emit(state, UNCLASSIFIABLE, "low", False, True, 0)
            return {
                "complaint_category": UNCLASSIFIABLE,
                "classification_confidence": "low",
                "is_complaint": False,
                "out_of_scope": True,
                "status": AgentStatus.SUCCESS,
            }

        ranked: List[Tuple[str, int]] = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
        top_category, top_hits = ranked[0]
        distinct_categories = len(scores)

        # Confidence: a single matching category, or a strict winner, is high;
        # a tie between categories is medium. Low is reserved for the
        # no-category branch above, so the three levels stay distinguishable.
        if distinct_categories == 1 or top_hits > ranked[1][1]:
            confidence = "high"
        else:
            confidence = "medium"

        if top_category in NON_COMPLAINT_CATEGORIES and not has_complaint_signal:
            logger.info("ComplaintClassifyNode: general inquiry, no complaint signal")
            self._emit(state, "general_inquiry", confidence, False, True, top_hits)
            return {
                "complaint_category": "general_inquiry",
                "classification_confidence": confidence,
                "is_complaint": False,
                "out_of_scope": True,
                "status": AgentStatus.SUCCESS,
            }

        # A general-inquiry top with a complaint signal present falls through to
        # the strongest non-general category, if there is one.
        if top_category == "general_inquiry" and has_complaint_signal:
            non_general = [name for name, _ in ranked if name != "general_inquiry"]
            if non_general:
                top_category = non_general[0]

        logger.info(
            "ComplaintClassifyNode: category=%s confidence=%s (hits=%d, cats=%d)",
            top_category,
            confidence,
            top_hits,
            distinct_categories,
        )
        self._emit(state, top_category, confidence, True, False, top_hits)
        return {
            "complaint_category": top_category,
            "classification_confidence": confidence,
            "is_complaint": True,
            "out_of_scope": False,
            "status": AgentStatus.SUCCESS,
        }
