# Unit tests for OutputValidateNode — completeness of the routing decision.
#
# This node has one duty, and the tests hold it to exactly that. The closed
# vocabulary and credential invariant is PostProcessNode's duty; asserting it
# here as well would make both copies unfalsifiable — removing either alone
# would leave every test green.

import pytest

from framework.schemas.agent_status import AgentStatus
from framework.schemas.trust_level import TrustLevel
from src.nodes.output_validate_node import (
    REQUIRED_COMPLAINT_FIELDS,
    REQUIRED_OUT_OF_SCOPE_FIELDS,
    OutputValidateNode,
)
from src.schemas.vocabulary import REJECTION_REASONS


@pytest.fixture(autouse=True)
def _mute_audit(monkeypatch):
    monkeypatch.setattr("src.nodes.output_validate_node.emit_trace_event", lambda *a, **k: None)


def _complete(**extra):
    state = {
        "out_of_scope": False,
        "is_complaint": True,
        "complaint_category": "claim_handling",
        "classification_confidence": "high",
        "severity_level": "critical",
        "severity_score": 90,
        "severity_modifiers": ["urgency"],
        "routing_target_officer_id": "compliance_chief",
        "routing_target_team": "claims_escalation",
        "routing_sla_hours": 4,
        "routing_rationale": "category=claim_handling severity=critical team=claims_escalation sla_hours=4",
        "agency_id": "agency_017",
        "case_ref": "case_9001",
        "channel": "counter",
    }
    state.update(extra)
    return state


class TestTrustDeclaration:
    def test_inner_node_admits_whoever_the_entry_admitted(self):
        assert OutputValidateNode.required_trust_level is TrustLevel.ANONYMOUS


class TestAssembly:
    def test_a_complete_decision_is_assembled(self):
        result = OutputValidateNode().execute(_complete())
        assert result["status"] == AgentStatus.SUCCESS
        decision = result["result"]
        assert decision["classification"] == {
            "category": "claim_handling",
            "confidence": "high",
            "is_complaint": True,
        }
        assert decision["severity"] == {"level": "critical", "score": 90, "modifiers": ["urgency"]}
        assert decision["routing"]["team"] == "claims_escalation"
        assert decision["case"] == {
            "agency_id": "agency_017",
            "case_ref": "case_9001",
            "channel": "counter",
        }
        assert decision["out_of_scope"] is False

    def test_the_out_of_scope_path_needs_less_and_still_assembles(self):
        result = OutputValidateNode().execute(
            {
                "out_of_scope": True,
                "is_complaint": False,
                "complaint_category": "general_inquiry",
                "classification_confidence": "high",
                "routing_target_team": "general_inquiry_desk",
                "routing_sla_hours": 72,
                "routing_rationale": "category=general_inquiry team=general_inquiry_desk sla_hours=72 rule=non_complaint_inquiry",
            }
        )
        assert result["status"] == AgentStatus.SUCCESS
        assert result["result"]["out_of_scope"] is True
        assert result["result"]["severity"] == {"level": None, "score": None, "modifiers": []}

    def test_the_shape_is_exactly_the_declared_one(self):
        """The gate downstream rejects an unexpected key, so the shape is fixed."""
        decision = OutputValidateNode().execute(_complete())["result"]
        assert set(decision) == {"classification", "severity", "routing", "case", "out_of_scope"}
        assert set(decision["classification"]) == {"category", "confidence", "is_complaint"}
        assert set(decision["severity"]) == {"level", "score", "modifiers"}
        assert set(decision["routing"]) == {"officer_id", "team", "sla_hours", "rationale"}
        assert set(decision["case"]) == {"agency_id", "case_ref", "channel"}


class TestCompleteness:
    @pytest.mark.parametrize("field", REQUIRED_COMPLAINT_FIELDS)
    def test_a_missing_complaint_field_withholds_the_decision(self, field):
        result = OutputValidateNode().execute(_complete(**{field: None}))
        assert result["status"] == AgentStatus.ERROR
        assert result["rejection_reason"] == "routing_incomplete"
        # Cleared, not merely absent: the envelope falls back to state["result"]
        # whenever no formatted output is present, so a half-built decision left
        # in state still ships.
        assert result["result"] is None

    @pytest.mark.parametrize("field", REQUIRED_OUT_OF_SCOPE_FIELDS)
    def test_a_missing_out_of_scope_field_withholds_the_decision(self, field):
        state = {
            "out_of_scope": True,
            "routing_target_team": "general_inquiry_desk",
            "routing_sla_hours": 72,
            "routing_rationale": "category=general_inquiry team=general_inquiry_desk sla_hours=72 rule=non_complaint_inquiry",
        }
        state[field] = None
        result = OutputValidateNode().execute(state)
        assert result["status"] == AgentStatus.ERROR
        assert result["result"] is None

    def test_an_empty_string_counts_as_missing(self):
        result = OutputValidateNode().execute(_complete(routing_rationale=""))
        assert result["status"] == AgentStatus.ERROR
        assert result["result"] is None

    def test_an_upstream_refusal_is_carried_through_unchanged(self):
        result = OutputValidateNode().execute(_complete(rejection_reason="instruction_detected"))
        assert result["status"] == AgentStatus.ERROR
        assert result["rejection_reason"] == "instruction_detected"
        assert result["result"] is None

    def test_every_reason_it_can_emit_is_on_the_closed_set(self):
        assert "routing_incomplete" in REJECTION_REASONS


class TestFinalGateIsNotOverridden:
    def test_the_framework_output_gate_is_not_shadowed(self):
        """Overriding it raises TypeError at class definition; assert the shape."""
        assert "_security_gate_output" not in OutputValidateNode.__dict__
        assert "_security_gate_input" not in OutputValidateNode.__dict__
