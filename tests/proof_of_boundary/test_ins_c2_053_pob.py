# INS-C2-053 — Proof-of-Boundary: mandatory security / routing boundary scenarios
#
# PB-1  S-3 output gate enforced : PostProcessNode WITHHOLDS the whole routing
#                                  decision when it carries an internal
#                                  (_internal_*) key, a credential-bearing key,
#                                  or a credential-shaped value.
#
#                                  This replaces an earlier form of PB-1 that
#                                  asserted OutputValidateNode._extra_security_
#                                  gate_output STRIPPED such keys and shipped
#                                  the remainder. Stripping is sanitising, not
#                                  refusing: it forwards an output that has
#                                  already been shown to be off-contract, and
#                                  it cannot express "this decision is not
#                                  safe to release". The invariant now lives in
#                                  one place — the outer PostProcessNode — as a
#                                  closed-vocabulary allowlist, so an unexpected
#                                  key is a violation by construction rather
#                                  than by a denylist keeping pace. The gate
#                                  clears every output-bearing field, because
#                                  the envelope falls back to state["result"].
#
#                                  The assertions below are therefore STRICTER
#                                  than the ones they replace, not looser.
#
# PB-2  Routing correctness      : (complaint_category x severity_level) maps to
#                                  the documented compliance team for every main
#                                  combination, end-to-end through RouteDecideNode.
#
# PB-3  Out-of-scope -> SUCCESS  : a non-complaint inquiry and an unclassifiable
#                                  input both flow through the domain nodes to
#                                  out_of_scope=True + AgentStatus.SUCCESS with a
#                                  routing team set (the general inquiry desk) —
#                                  NOT an error boundary. The nodes are chained in
#                                  designed pipeline order (input_validate ->
#                                  complaint_classify -> severity_score ->
#                                  route_decide -> output_validate), which routes
#                                  the out-of-scope case to general_inquiry_desk
#                                  inside route_decide and surfaces SUCCESS.
#
# PB-4  PII not leaked in State  : after InputValidateNode the masked text in
#                                  State no longer contains the raw phone digits.
#
# Asserted against the MERGED develop implementation (7194fa8f, Wave-1 green).
# Audit free functions are muted at each node module via an autouse fixture
# (NEVER stub shared.* in sys.modules — the CI wheel ships a real shared package).
# The audit-mask check inspects the PAYLOAD arg (call.args[1]) only.

from unittest.mock import MagicMock

import pytest

from src.nodes.input_validate_node import InputValidateNode, _MASK
from src.nodes.complaint_classify_node import ComplaintClassifyNode
from src.nodes.severity_score_node import SeverityScoreNode
from src.nodes.route_decide_node import RouteDecideNode
from src.nodes.output_validate_node import OutputValidateNode
from src.nodes.post_process_node import PostProcessNode, _rationale_is_well_formed
from src.schemas.vocabulary import WITHHELD_NOTICE_KEY
from framework.schemas.agent_status import AgentStatus


@pytest.fixture(autouse=True)
def _mute_audit(monkeypatch):
    """Silence the S-4 audit free functions at every node module under test."""
    monkeypatch.setattr("src.nodes.input_validate_node.emit_trace_event", lambda *a, **k: None)
    monkeypatch.setattr("src.nodes.route_decide_node.emit_trace_event", lambda *a, **k: None)
    monkeypatch.setattr("src.nodes.post_process_node.emit_trace_event", lambda *a, **k: None)


def _run_pipeline(raw_text, channel):
    """Chain the 5 domain nodes in designed pipeline order, returning the state.

    Mirrors the inner DomainWorkflowGraph node sequence; each node returns a
    partial-dict update that is merged into the running state.
    """
    state = {"raw_input_text": raw_text, "channel": channel}
    for node in (
        InputValidateNode(),
        ComplaintClassifyNode(),
        SeverityScoreNode(),
        RouteDecideNode(),
        OutputValidateNode(),
    ):
        state.update(node.execute(state))
    return state


def _clean_decision():
    """A routing decision entirely on the closed vocabulary."""
    return {
        "classification": {
            "category": "claim_handling",
            "confidence": "high",
            "is_complaint": True,
        },
        "severity": {"level": "high", "score": 70, "modifiers": ["regulatory"]},
        "routing": {
            "officer_id": "compliance_officer_a",
            "team": "claims_team",
            "sla_hours": 8,
            "rationale": "category=claim_handling severity=high",
        },
        "case": {"agency_id": "ag_01", "case_ref": "cr_01", "channel": "phone"},
        "out_of_scope": False,
    }


class TestPB1S3OutputGateWithholds:
    """PB-1: the S-3 output gate withholds an off-contract routing decision."""

    def setup_method(self):
        self.node = PostProcessNode()

    def _run(self, decision):
        return self.node.execute({"result": decision})

    def test_a_clean_decision_is_released(self):
        """The gate is not vacuous: the on-contract decision does ship."""
        decision = _clean_decision()
        result = self._run(decision)

        assert result["status"] == AgentStatus.SUCCESS
        assert result["formatted_output"] == decision
        assert result["result"] == decision

    def _assert_withheld(self, result, reason):
        """Withholding means: ERROR, truthy notice, and NOTHING released.

        `result` is cleared as well as `formatted_output`, because
        AgentBaseGraph.get_output falls back to state["result"] — a gate that
        replaces only the formatted output still ships the inner object.
        """
        assert result["status"] == AgentStatus.ERROR
        assert result["result"] is None
        assert result["formatted_output"] == {WITHHELD_NOTICE_KEY: reason}
        assert result["formatted_output"], "the notice must be truthy or the fallback re-opens"
        assert result["rejection_reason"] == reason

    def test_an_internal_key_withholds_the_whole_decision(self):
        """A top-level _internal_* key is off the closed vocabulary."""
        decision = _clean_decision()
        decision["_internal_debug"] = "secret"

        result = self._run(decision)
        self._assert_withheld(result, "output_not_on_closed_vocabulary")
        assert "secret" not in repr(result)

    def test_a_nested_internal_key_withholds_the_whole_decision(self):
        """Section shapes are exact-set equality, so nesting does not evade."""
        decision = _clean_decision()
        decision["severity"]["_internal_trace"] = "x"

        result = self._run(decision)
        self._assert_withheld(result, "output_not_on_closed_vocabulary")

    def test_credential_bearing_keys_withhold_the_whole_decision(self):
        """token / password / secret keys are extra keys — refused, not stripped."""
        decision = _clean_decision()
        decision["api_token"] = "abcdefghijklmnop1234"
        decision["db_password"] = "hunter2xyz"
        decision["client_secret"] = "shhhhhhh"

        result = self._run(decision)
        self._assert_withheld(result, "output_not_on_closed_vocabulary")
        for leaked in ("abcdefghijklmnop1234", "hunter2xyz", "shhhhhhh"):
            assert leaked not in repr(result)

    def test_a_credential_shaped_value_on_a_valid_key_is_withheld(self):
        """The vocabulary check PASSES this one — only the credential screen catches it.

        The value is deliberately spelled with characters the rationale grammar
        allows ([A-Za-z0-9_*]), so `token=sk_live_…` is a well-formed pair and
        the closed-vocabulary check has no complaint. A credential with a `-`
        in it (a GitLab PAT, say) would be refused by the grammar first and
        would prove nothing about the screen. This is the one place a
        credential can ride inside an otherwise on-contract decision.
        """
        leak = "sk_live_" + "abcdefghijklmnop1234"
        decision = _clean_decision()
        decision["routing"]["rationale"] = f"token={leak}"
        # The grammar really does accept it — otherwise this tests the wrong layer.
        assert _rationale_is_well_formed(f"token={leak}")

        result = self._run(decision)
        self._assert_withheld(result, "output_credential_detected")
        assert leak not in repr(result)


class TestPB2RoutingTableCoverage:
    """PB-2: routing-table correctness for all main (category, severity) combos."""

    def setup_method(self):
        self.node = RouteDecideNode()

    @pytest.mark.parametrize(
        "category,severity,expected_team",
        [
            ("claim_handling", "critical", "claims_escalation"),
            ("claim_handling", "high", "claims_team"),
            ("claim_handling", "low", "claims_team"),
            ("solicitation_conduct", "critical", "solicitation_review"),
            ("solicitation_conduct", "high", "solicitation_review"),
            ("solicitation_conduct", "low", "solicitation_review"),
            ("contract_explanation", "critical", "contract_support"),
            ("contract_explanation", "high", "contract_support"),
            ("premium_billing", "critical", "billing_team"),
            ("premium_billing", "low", "billing_team"),
            ("policy_cancellation", "medium", "contract_support"),
            ("customer_service", "low", "customer_relations"),
        ],
    )
    def test_routing_table_coverage(self, category, severity, expected_team):
        """Each (category, severity) maps to its documented compliance team."""
        result = self.node.execute(
            {
                "out_of_scope": False,
                "is_complaint": True,
                "complaint_category": category,
                "severity_level": severity,
                "channel": "phone",
            }
        )

        assert result["status"] == AgentStatus.SUCCESS
        assert result["routing_target_team"] == expected_team
        assert result["routing_rationale"].strip()

    def test_claim_critical_escalates_to_officer(self):
        """The (claim_handling, critical) escalation names a compliance officer."""
        result = self.node.execute(
            {
                "out_of_scope": False,
                "is_complaint": True,
                "complaint_category": "claim_handling",
                "severity_level": "critical",
                "channel": "counter",
            }
        )
        assert result["routing_target_officer_id"] == "compliance_chief"
        assert result["routing_sla_hours"] == 4


class TestPB3OutOfScopeIsSuccess:
    """PB-3: out-of-scope inputs flow to SUCCESS + general inquiry desk."""

    def test_out_of_scope_end_to_end(self):
        """A non-complaint inquiry -> out_of_scope=True, SUCCESS, team set."""
        state = _run_pipeline("手続き方法を教えてください。", "online")

        assert state["status"] == AgentStatus.SUCCESS
        assert state["out_of_scope"] is True
        assert state["routing_target_team"] == "general_inquiry_desk"
        assert state["result"]["out_of_scope"] is True

    def test_unclassifiable_end_to_end(self):
        """Gibberish input -> out_of_scope=True, SUCCESS, routed to general desk."""
        state = _run_pipeline("あああ ＸＹＺ ???", "phone")

        assert state["status"] == AgentStatus.SUCCESS
        assert state["out_of_scope"] is True
        assert state["routing_target_team"] == "general_inquiry_desk"


class TestPB4PIINotLeakedInState:
    """PB-4: PII is masked out of State after InputValidateNode."""

    def test_pii_not_in_state_after_input_validate(self):
        """The masked raw_input_text no longer contains the raw phone digits.

        Asserted as behaviour, not as a sentinel spelling. The earlier form of
        this test required the literal "[PHONE]"; the template now emits the
        platform's own "[MASKED]" so that one sentinel vocabulary reaches
        State whether the platform's S-2 filter masked a span first or this
        node's residual pass did. Pinning the label made the test a statement
        about wording — what PB-4 has to prove is that no digit of the number
        survives into State, which is asserted digit-by-digit below.
        """
        node = InputValidateNode()
        raw = "090-1234-5678の田中と申します。保険金の件で。"
        result = node.execute({"raw_input_text": raw, "channel": "phone"})
        masked = result["raw_input_text"]

        assert result["status"] == AgentStatus.SUCCESS
        assert "090-1234-5678" not in masked
        # Stricter than the substring check: no fragment of the number survives.
        for fragment in ("090", "1234", "5678"):
            assert fragment not in masked
        # Something was redacted rather than the text being silently dropped.
        assert _MASK in masked
        # The complaint itself still reaches the classifier.
        assert "保険金" in masked

    def test_routing_audit_payload_carries_no_pii(self):
        """The S-4 routing-decision audit PAYLOAD never carries raw PII."""
        spy = MagicMock()
        import src.nodes.route_decide_node as mod

        original = mod.emit_trace_event
        mod.emit_trace_event = spy
        try:
            RouteDecideNode().execute(
                {
                    "out_of_scope": False,
                    "is_complaint": True,
                    "complaint_category": "claim_handling",
                    "severity_level": "critical",
                    "severity_score": 90,
                    "channel": "phone",
                    "raw_input_text": "090-1234-5678の田中です。保険金の件。",
                }
            )
        finally:
            mod.emit_trace_event = original

        assert spy.called
        for call in spy.call_args_list:
            payload = call.args[1]
            assert "raw_input_text" not in payload
            assert "090-1234-5678" not in repr(payload)
