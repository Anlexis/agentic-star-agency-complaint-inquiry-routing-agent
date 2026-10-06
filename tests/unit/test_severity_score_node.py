# Unit tests for SeverityScoreNode — additive rule scoring.
#
# The property that matters most here is that the number MOVES with the input.
# A score that is structurally the same for a small complaint and a severe one
# is a score nobody can act on, and it is the failure mode that hides best: the
# pipeline runs, the envelope looks right, and every severity band below the
# constant one is unreachable.

import pytest

from framework.schemas.agent_status import AgentStatus
from framework.schemas.trust_level import TrustLevel
from src.nodes.severity_score_node import (
    BASE_SCORES,
    DEFAULT_COUNTER_CHANNEL_DELTA,
    DEFAULT_REPEAT_COMPLAINT_DELTA,
    KEYWORD_MODIFIERS,
    REPEAT_COUNT_CAP,
    SeverityScoreNode,
    level_from_score,
)
from src.schemas.vocabulary import (
    SEVERITY_LEVELS,
    SEVERITY_MODIFIERS,
    SEVERITY_SCORE_MAX,
    SEVERITY_SCORE_MIN,
)


@pytest.fixture(autouse=True)
def _mute_audit(monkeypatch):
    monkeypatch.setattr("src.nodes.severity_score_node.emit_trace_event", lambda *a, **k: None)


def _state(category="claim_handling", text="", channel="online", prior=0):
    return {
        "is_complaint": True,
        "out_of_scope": False,
        "complaint_category": category,
        "raw_input_text": text,
        "channel": channel,
        "prior_complaint_count": prior,
    }


class TestVocabularyIsClosed:
    def test_every_modifier_name_is_in_the_vocabulary(self):
        names = {modifier["name"] for modifier in KEYWORD_MODIFIERS}
        names |= {"counter_channel", "repeat_complaint"}
        assert names <= SEVERITY_MODIFIERS

    def test_every_base_score_category_is_scorable(self):
        for score in BASE_SCORES.values():
            assert SEVERITY_SCORE_MIN <= score <= SEVERITY_SCORE_MAX

    def test_every_level_is_in_the_vocabulary(self):
        assert {level_from_score(score) for score in range(0, 101)} <= SEVERITY_LEVELS

    def test_trust_declaration(self):
        assert SeverityScoreNode.required_trust_level is TrustLevel.ANONYMOUS


class TestTheNumberMoves:
    def test_two_very_different_inputs_produce_different_scores(self):
        low = SeverityScoreNode().execute(_state(category="customer_service"))
        high = SeverityScoreNode().execute(
            _state(
                category="solicitation_conduct",
                text="弁護士に相談し、金融庁に苦情申出します。至急。",
                channel="counter",
                prior=3,
            )
        )
        assert low["severity_score"] < high["severity_score"]
        assert low["severity_level"] != high["severity_level"]

    def test_every_severity_band_is_reachable(self):
        """A band no input can reach is a band that does not exist."""
        reached = set()
        for category in BASE_SCORES:
            for text in ("", "至急", "弁護士に相談", "金融庁に苦情申出し、弁護士に相談。至急。"):
                for channel in ("online", "counter"):
                    for prior in (0, 3):
                        result = SeverityScoreNode().execute(
                            _state(category=category, text=text, channel=channel, prior=prior)
                        )
                        reached.add(result["severity_level"])
        assert reached == SEVERITY_LEVELS

    @pytest.mark.parametrize("modifier", KEYWORD_MODIFIERS)
    def test_each_keyword_modifier_raises_the_score(self, modifier):
        base = SeverityScoreNode().execute(_state(category="customer_service"))
        bumped = SeverityScoreNode().execute(_state(category="customer_service", text=modifier["keywords"][0]))
        assert bumped["severity_score"] == base["severity_score"] + modifier["delta"]
        assert modifier["name"] in bumped["severity_modifiers"]

    def test_the_counter_channel_raises_the_score(self):
        online = SeverityScoreNode().execute(_state(channel="online"))
        counter = SeverityScoreNode().execute(_state(channel="counter"))
        assert counter["severity_score"] - online["severity_score"] == DEFAULT_COUNTER_CHANNEL_DELTA
        assert "counter_channel" in counter["severity_modifiers"]

    def test_prior_complaints_raise_the_score_up_to_the_cap(self):
        none = SeverityScoreNode().execute(_state(prior=0))["severity_score"]
        one = SeverityScoreNode().execute(_state(prior=1))["severity_score"]
        capped = SeverityScoreNode().execute(_state(prior=REPEAT_COUNT_CAP))["severity_score"]
        beyond = SeverityScoreNode().execute(_state(prior=1000))["severity_score"]
        assert one == none + DEFAULT_REPEAT_COMPLAINT_DELTA
        assert capped == none + REPEAT_COUNT_CAP * DEFAULT_REPEAT_COMPLAINT_DELTA
        assert beyond == capped

    def test_the_score_is_bounded_at_both_ends(self):
        maxed = SeverityScoreNode().execute(
            _state(
                category="solicitation_conduct",
                text="金融庁に苦情申出、弁護士に相談、訴訟も検討。至急。",
                channel="counter",
                prior=3,
            )
        )
        assert maxed["severity_score"] == SEVERITY_SCORE_MAX
        assert maxed["severity_level"] == "critical"


class TestNonFiniteInput:
    @pytest.mark.parametrize(
        "prior", ["NaN", "Infinity", "-Infinity", float("nan"), float("inf"), True, 2.5, -1, 5000, "many"]
    )
    def test_an_unusable_count_contributes_nothing_and_still_routes(self, prior):
        """Re-parsed rather than trusted — this node is reachable directly too.

        The complaint still gets a score: refusing to score it would leave the
        message unrouted, which is worse than scoring it without the modifier.
        """
        result = SeverityScoreNode().execute(_state(prior=prior))
        baseline = SeverityScoreNode().execute(_state(prior=0))
        assert result["severity_score"] == baseline["severity_score"]
        assert "repeat_complaint" not in result["severity_modifiers"]
        assert result["status"] == AgentStatus.SUCCESS


class TestOutOfScope:
    @pytest.mark.parametrize(
        "state",
        [
            {"out_of_scope": True, "is_complaint": False},
            {"out_of_scope": False, "is_complaint": False},
        ],
    )
    def test_no_score_is_produced(self, state):
        result = SeverityScoreNode().execute(state)
        assert result["severity_level"] is None
        assert result["severity_score"] is None
        assert result["severity_modifiers"] == []
        assert result["status"] == AgentStatus.SUCCESS


class TestDeclaredConfigurationIsLive:
    def test_declared_deltas_change_the_score(self):
        node = SeverityScoreNode({"severity": {"counter_channel_delta": 20, "repeat_complaint_delta": 1}})
        assert node.counter_channel_delta == 20
        assert node.repeat_complaint_delta == 1
        default = SeverityScoreNode().execute(_state(channel="counter", prior=2))
        tuned = node.execute(_state(channel="counter", prior=2))
        assert tuned["severity_score"] != default["severity_score"]
        assert tuned["severity_score"] == BASE_SCORES["claim_handling"] + 20 + 2

    @pytest.mark.parametrize("value", [-1, 999, "NaN", float("inf"), True, None])
    def test_an_out_of_range_declaration_is_dropped_not_clamped(self, value):
        node = SeverityScoreNode({"severity": {"counter_channel_delta": value}})
        assert node.counter_channel_delta == DEFAULT_COUNTER_CHANNEL_DELTA

    def test_a_zero_delta_disables_the_rule_without_naming_it(self):
        node = SeverityScoreNode({"severity": {"counter_channel_delta": 0}})
        result = node.execute(_state(channel="counter"))
        assert "counter_channel" not in result["severity_modifiers"]
