# Unit tests for ComplaintClassifyNode — keyword classification.
#
# Two properties matter beyond "the right category comes out": every category it
# can emit is on the closed vocabulary the output gate enforces, and a
# non-complaint enquiry is not an error — it still has to reach a routing
# decision, because a message that is recognised and then not routed is the
# outcome this template exists to prevent.

import pytest

from framework.schemas.agent_status import AgentStatus
from framework.schemas.trust_level import TrustLevel
from src.nodes.complaint_classify_node import (
    CATEGORY_KEYWORDS,
    COMPLAINT_SIGNALS,
    UNCLASSIFIABLE,
    ComplaintClassifyNode,
)
from src.schemas.vocabulary import COMPLAINT_CATEGORIES, CONFIDENCE_LEVELS


@pytest.fixture(autouse=True)
def _mute_audit(monkeypatch):
    monkeypatch.setattr("src.nodes.complaint_classify_node.emit_trace_event", lambda *a, **k: None)


def _state(text, valid=True):
    return {"raw_input_text": text, "is_valid_input": valid}


class TestVocabularyIsClosed:
    def test_every_table_category_is_in_the_vocabulary(self):
        """The gate rejects a category it does not know, so the two move together."""
        assert set(CATEGORY_KEYWORDS) <= COMPLAINT_CATEGORIES

    def test_the_unclassifiable_label_is_in_the_vocabulary(self):
        assert UNCLASSIFIABLE in COMPLAINT_CATEGORIES

    def test_trust_declaration(self):
        assert ComplaintClassifyNode.required_trust_level is TrustLevel.ANONYMOUS


class TestClassification:
    @pytest.mark.parametrize(
        "text,category",
        [
            ("保険金の支払拒否について苦情です", "claim_handling"),
            ("重要事項の説明不備について苦情", "contract_explanation"),
            ("強引な勧誘を受けました。苦情です", "solicitation_conduct"),
            ("保険料の二重請求です。苦情", "premium_billing"),
            ("解約返戻金がおかしい", "policy_cancellation"),
            ("窓口の態度がひどい", "customer_service"),
        ],
    )
    def test_categories(self, text, category):
        result = ComplaintClassifyNode().execute(_state(text))
        assert result["complaint_category"] == category
        assert result["is_complaint"] is True
        assert result["out_of_scope"] is False
        assert result["status"] == AgentStatus.SUCCESS
        assert result["classification_confidence"] in CONFIDENCE_LEVELS

    def test_a_plain_enquiry_is_not_a_complaint(self):
        result = ComplaintClassifyNode().execute(_state("手続き方法を教えてください"))
        assert result["complaint_category"] == "general_inquiry"
        assert result["is_complaint"] is False
        assert result["out_of_scope"] is True
        # Out of scope is not an error — the message still gets a decision.
        assert result["status"] == AgentStatus.SUCCESS

    def test_an_enquiry_carrying_a_complaint_signal_falls_through(self):
        result = ComplaintClassifyNode().execute(_state("保険料の引き落としについて教えてください。納得できない。"))
        assert result["is_complaint"] is True
        assert result["complaint_category"] == "premium_billing"

    def test_unrecognised_text_still_reaches_a_decision(self):
        result = ComplaintClassifyNode().execute(_state("あああ ＸＹＺ ???"))
        assert result["complaint_category"] == UNCLASSIFIABLE
        assert result["classification_confidence"] == "low"
        assert result["out_of_scope"] is True
        assert result["status"] == AgentStatus.SUCCESS

    def test_a_refused_input_is_not_classified(self):
        result = ComplaintClassifyNode().execute(_state("保険金の苦情", valid=False))
        assert result["complaint_category"] is None
        assert result["classification_confidence"] is None
        assert result["out_of_scope"] is True


class TestConfidence:
    def test_a_single_matching_category_is_high(self):
        result = ComplaintClassifyNode().execute(_state("保険金の支払拒否"))
        assert result["classification_confidence"] == "high"

    def test_a_tie_between_categories_is_medium(self):
        # One hit for claim_handling (保険金) and one for policy_cancellation (解約).
        result = ComplaintClassifyNode().execute(_state("保険金と解約について苦情"))
        assert result["classification_confidence"] == "medium"

    def test_a_strict_winner_is_high(self):
        result = ComplaintClassifyNode().execute(_state("保険金の支払拒否と査定、解約も"))
        assert result["classification_confidence"] == "high"

    def test_ranking_is_deterministic_on_a_tie(self):
        """Same input, same answer — the decision has to be reproducible."""
        text = "保険金と解約について苦情"
        answers = {ComplaintClassifyNode().execute(_state(text))["complaint_category"] for _ in range(20)}
        assert len(answers) == 1


class TestSignals:
    @pytest.mark.parametrize("signal", COMPLAINT_SIGNALS)
    def test_every_signal_promotes_an_enquiry(self, signal):
        result = ComplaintClassifyNode().execute(_state(f"保険料について教えて。{signal}。"))
        assert result["is_complaint"] is True
