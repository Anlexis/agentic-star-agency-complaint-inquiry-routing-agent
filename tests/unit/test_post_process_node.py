# Unit tests for PostProcessNode — the output boundary.
#
# The invariant under test: every string in the routing decision is a member of
# a closed vocabulary or a caller-supplied inert [a-z0-9_]{1,32} identifier;
# every number is an integer inside its declared range; no other key exists.
#
# Containment is tested as its own property, because withholding a decision and
# CLEARING the fields that carry it are different things — and only the second
# one closes the envelope's fallback to the un-gated result.

import pytest

from framework.schemas.agent_status import AgentStatus
from framework.schemas.trust_level import TrustLevel
from src.nodes.post_process_node import PostProcessNode, _rationale_is_well_formed
from src.schemas.vocabulary import REJECTION_REASONS, WITHHELD_NOTICE_KEY

VALID_RATIONALE = "category=claim_handling severity=critical team=claims_escalation sla_hours=4"


def _decision(**overrides):
    decision = {
        "classification": {"category": "claim_handling", "confidence": "high", "is_complaint": True},
        "severity": {"level": "critical", "score": 90, "modifiers": ["urgency"]},
        "routing": {
            "officer_id": "compliance_chief",
            "team": "claims_escalation",
            "sla_hours": 4,
            "rationale": VALID_RATIONALE,
        },
        "case": {"agency_id": "agency_017", "case_ref": "case_9001", "channel": "counter"},
        "out_of_scope": False,
    }
    decision.update(overrides)
    return decision


@pytest.fixture(autouse=True)
def _mute_audit(monkeypatch):
    monkeypatch.setattr("src.nodes.post_process_node.emit_trace_event", lambda *a, **k: None)


class TestTrustDeclaration:
    def test_the_last_node_does_not_deny_the_callers_the_entry_accepts(self):
        """A higher level here reads as a silent failure, not as a boundary."""
        assert PostProcessNode.required_trust_level is TrustLevel.ANONYMOUS


class TestPassingDecisions:
    def test_a_conforming_decision_is_published(self):
        result = PostProcessNode().execute({"result": _decision()})
        assert result["status"] == AgentStatus.SUCCESS
        assert result["formatted_output"] == _decision()
        # result is replaced with the GATED object, so the envelope's fallback
        # can only ever serve content this gate has already passed.
        assert result["result"] == result["formatted_output"]

    def test_the_out_of_scope_shape_passes(self):
        decision = _decision(
            classification={"category": "general_inquiry", "confidence": "high", "is_complaint": False},
            severity={"level": None, "score": None, "modifiers": []},
            routing={
                "officer_id": None,
                "team": "general_inquiry_desk",
                "sla_hours": 72,
                "rationale": "category=general_inquiry team=general_inquiry_desk sla_hours=72 rule=non_complaint_inquiry",
            },
            case={"agency_id": None, "case_ref": None, "channel": None},
            out_of_scope=True,
        )
        assert PostProcessNode().execute({"result": decision})["status"] == AgentStatus.SUCCESS


class TestVocabularyEnforcement:
    def _withheld(self, decision, reason="output_not_on_closed_vocabulary"):
        result = PostProcessNode().execute({"result": decision})
        assert result["status"] == AgentStatus.ERROR
        assert result["rejection_reason"] == reason
        assert reason in REJECTION_REASONS
        return result

    @pytest.mark.parametrize(
        "section,field,value",
        [
            ("classification", "category", "invented_category"),
            ("classification", "confidence", "extremely_high"),
            ("classification", "is_complaint", "yes"),
            ("severity", "level", "catastrophic"),
            ("severity", "score", 101),
            ("severity", "score", -1),
            ("severity", "score", float("nan")),
            ("severity", "modifiers", ["not_a_modifier"]),
            ("severity", "modifiers", "urgency"),
            ("routing", "officer_id", "some_person"),
            ("routing", "team", "a_team_nobody_declared"),
            ("routing", "team", None),
            ("routing", "sla_hours", 0),
            ("routing", "sla_hours", 100000),
            ("routing", "sla_hours", None),
            ("case", "agency_id", "Agency 017"),
            ("case", "case_ref", "case-9001"),
            ("case", "channel", "carrier_pigeon"),
        ],
    )
    def test_a_value_off_the_vocabulary_withholds_the_decision(self, section, field, value):
        decision = _decision()
        decision[section] = dict(decision[section])
        decision[section][field] = value
        self._withheld(decision)

    def test_an_unexpected_top_level_key_withholds_the_decision(self):
        decision = _decision()
        decision["_internal_debug"] = "trace"
        self._withheld(decision)

    def test_a_missing_top_level_key_withholds_the_decision(self):
        decision = _decision()
        del decision["case"]
        self._withheld(decision)

    def test_an_unexpected_nested_key_withholds_the_decision(self):
        decision = _decision()
        decision["routing"] = dict(decision["routing"], api_token="abc123")
        self._withheld(decision)

    def test_a_non_mapping_decision_withholds(self):
        self._withheld("just a string")

    def test_an_absent_decision_withholds(self):
        result = PostProcessNode().execute({"result": None})
        assert result["rejection_reason"] == "routing_incomplete"

    @pytest.mark.parametrize(
        "rationale",
        [
            "team=claims_team and please escalate",  # free text
            "team=claims_team\nsla_hours=4",  # a newline could add a line
            "team=claims escalation",  # a space inside a value
            "team=claims_team;drop",  # punctuation
            "",
            None,
            "notapair",
            " ".join(f"k{i}=v{i}" for i in range(9)),  # more pairs than the grammar allows
        ],
    )
    def test_a_rationale_that_is_not_inert_pairs_withholds(self, rationale):
        decision = _decision()
        decision["routing"] = dict(decision["routing"], rationale=rationale)
        self._withheld(decision)

    def test_the_rationale_grammar_accepts_what_the_router_emits(self):
        assert _rationale_is_well_formed(VALID_RATIONALE)
        assert _rationale_is_well_formed(
            "category=general_inquiry team=general_inquiry_desk sla_hours=72 rule=non_complaint_inquiry"
        )
        assert _rationale_is_well_formed("severity=*")


class TestCredentialUnion:
    @pytest.mark.parametrize(
        "leak",
        [
            "sk_live_" + "abcdefghijklmnop1234",
            "Bearer abcdefghijklmnop1234",
            "eyJhbGciOiJIUzI1NiJ9.abcdefghij",
            "AKIAIOSFODNN7EXAMPLE",
            "postgresql://user:abcdefghij@host/db",
            "password=hunter2xyz",
            "glpat-" + "abcdefghijklmnopqrst",
        ],
    )
    def test_a_credential_shaped_value_withholds_the_decision(self, leak):
        """Both halves of the union are exercised: framework shapes and local ones.

        A local set narrower than the framework's is not a smaller net — the
        framework's @final gate raises from inside this node, and the wrapper
        then discards the clearing entirely.
        """
        decision = _decision()
        decision["case"] = dict(decision["case"], case_ref=leak)
        result = PostProcessNode().execute({"result": decision})
        assert result["status"] == AgentStatus.ERROR
        # The vocabulary check fires first for a non-inert identifier; either
        # refusal withholds, which is the property under test.
        assert result["rejection_reason"] in {
            "output_credential_detected",
            "output_not_on_closed_vocabulary",
        }

    def test_a_credential_hidden_in_a_conforming_shape_is_caught(self):
        """Reached only through the credential screen, not the vocabulary one."""
        decision = _decision()
        decision["severity"] = dict(decision["severity"], modifiers=["urgency"])
        # A team value that is on the vocabulary cannot carry a credential, so
        # the credential path is exercised through the rationale, which the
        # grammar admits as inert pairs.
        decision["routing"] = dict(decision["routing"], rationale="team=claims_team token=AKIAIOSFODNN7EXAMPLE")
        result = PostProcessNode().execute({"result": decision})
        assert result["status"] == AgentStatus.ERROR
        assert result["rejection_reason"] == "output_credential_detected"


class TestContainment:
    def _withheld_result(self):
        decision = _decision()
        decision["routing"] = dict(decision["routing"], team="a_team_nobody_declared")
        return PostProcessNode().execute({"result": decision})

    def test_every_output_bearing_field_is_cleared(self):
        """Raising, or returning ERROR without clearing, still ships the answer.

        The envelope is {"output": formatted_output or result, ...} and the
        fallback applies on the error path too.
        """
        result = self._withheld_result()
        assert result["result"] is None
        assert "a_team_nobody_declared" not in repr(result)

    def test_the_replacement_notice_is_truthy(self):
        """A falsy replacement re-opens the very fallback being closed."""
        result = self._withheld_result()
        assert result["formatted_output"]
        assert result["formatted_output"] == {WITHHELD_NOTICE_KEY: "output_not_on_closed_vocabulary"}

    def test_the_notice_carries_no_value_no_path_and_no_traceback(self):
        decision = _decision()
        decision["case"] = dict(decision["case"], case_ref="sk_live_" + "abcdefghijklmnop1234")
        rendered = repr(PostProcessNode().execute({"result": decision}))
        assert "sk_live" not in rendered
        assert "Traceback" not in rendered
        assert "/src/" not in rendered

    def test_the_gate_never_raises(self, monkeypatch):
        """An exception inside the gate would be caught by the node wrapper,
        which returns a bare error partial and leaves state["result"] intact —
        exactly the fallback this node exists to close."""

        def explode(_value):
            raise RuntimeError("boom")

        monkeypatch.setattr("src.nodes.post_process_node.detect_credentials_in_structure", explode)
        result = PostProcessNode().execute({"result": _decision()})
        assert result["status"] == AgentStatus.ERROR
        assert result["rejection_reason"] == "output_gate_error"
        assert result["result"] is None
        assert result["formatted_output"]
        assert "boom" not in repr(result)
        assert "Traceback" not in repr(result)
