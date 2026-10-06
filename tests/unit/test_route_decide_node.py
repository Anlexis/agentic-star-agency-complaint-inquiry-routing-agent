# Unit tests for RouteDecideNode — the routing decision and its audit record.
#
# Three properties: every routable pair reaches a team, every value the node can
# emit is on the closed vocabulary the output gate enforces, and the audit
# payload never carries the complaint text.

import pytest

from framework.schemas.agent_status import AgentStatus
from framework.schemas.trust_level import TrustLevel
from src.nodes.route_decide_node import (
    DEFAULT_FALLBACK_SLA_HOURS,
    FALLBACK_TEAM,
    GENERAL_DESK_TEAM,
    ROUTING_RULES,
    RouteDecideNode,
    match_rule,
)
from src.schemas.vocabulary import (
    COMPLAINT_CATEGORIES,
    OFFICER_IDS,
    ROUTING_TEAMS,
    SEVERITY_LEVELS,
    SLA_HOURS_MAX,
    SLA_HOURS_MIN,
)


@pytest.fixture(autouse=True)
def _mute_audit(monkeypatch):
    monkeypatch.setattr("src.nodes.route_decide_node.emit_trace_event", lambda *a, **k: None)


def _state(category="claim_handling", severity="high", **extra):
    base = {
        "out_of_scope": False,
        "is_complaint": True,
        "complaint_category": category,
        "severity_level": severity,
        "channel": "phone",
    }
    base.update(extra)
    return base


class TestVocabularyIsClosed:
    def test_every_rule_names_a_known_team_and_officer(self):
        for rule in ROUTING_RULES:
            assert rule["category"] in COMPLAINT_CATEGORIES
            assert rule["severity"] == "*" or rule["severity"] in SEVERITY_LEVELS
            assert rule["team"] in ROUTING_TEAMS
            assert rule["officer_id"] is None or rule["officer_id"] in OFFICER_IDS
            assert SLA_HOURS_MIN <= rule["sla_hours"] <= SLA_HOURS_MAX

    def test_the_fallback_and_general_desk_are_known_teams(self):
        assert FALLBACK_TEAM in ROUTING_TEAMS
        assert GENERAL_DESK_TEAM in ROUTING_TEAMS

    def test_trust_declaration(self):
        assert RouteDecideNode.required_trust_level is TrustLevel.ANONYMOUS


class TestRoutingTable:
    @pytest.mark.parametrize(
        "category,severity,team",
        [
            ("claim_handling", "critical", "claims_escalation"),
            ("claim_handling", "high", "claims_team"),
            ("claim_handling", "low", "claims_team"),
            ("solicitation_conduct", "critical", "solicitation_review"),
            ("solicitation_conduct", "high", "solicitation_review"),
            ("solicitation_conduct", "low", "solicitation_review"),
            ("contract_explanation", "critical", "contract_support"),
            ("contract_explanation", "high", "contract_support"),
            ("contract_explanation", "medium", "contract_support"),
            ("premium_billing", "critical", "billing_team"),
            ("premium_billing", "low", "billing_team"),
            ("policy_cancellation", "medium", "contract_support"),
            ("customer_service", "low", "customer_relations"),
        ],
    )
    def test_each_pair_reaches_its_documented_team(self, category, severity, team):
        result = RouteDecideNode().execute(_state(category, severity))
        assert result["status"] == AgentStatus.SUCCESS
        assert result["routing_target_team"] == team

    def test_exact_severity_wins_over_the_catch_all(self):
        assert match_rule("claim_handling", "critical")["sla_hours"] == 4
        assert match_rule("claim_handling", "low")["sla_hours"] == 72

    def test_a_critical_claim_escalates_to_the_chief(self):
        result = RouteDecideNode().execute(_state("claim_handling", "critical"))
        assert result["routing_target_officer_id"] == "compliance_chief"
        assert result["routing_sla_hours"] == 4

    def test_the_service_level_tightens_as_severity_rises(self):
        slas = [
            RouteDecideNode().execute(_state("contract_explanation", level))["routing_sla_hours"]
            for level in ("critical", "high", "low")
        ]
        assert slas == sorted(slas)

    def test_an_unknown_category_still_reaches_a_team(self):
        """A complaint with no route is the failure this template prevents."""
        result = RouteDecideNode().execute(_state("unclassifiable", "high"))
        assert result["routing_target_team"] == FALLBACK_TEAM
        assert result["routing_sla_hours"] == DEFAULT_FALLBACK_SLA_HOURS
        assert result["status"] == AgentStatus.SUCCESS

    def test_out_of_scope_goes_to_the_general_desk(self):
        result = RouteDecideNode().execute(
            {"out_of_scope": True, "complaint_category": "general_inquiry", "channel": "online"}
        )
        assert result["routing_target_team"] == GENERAL_DESK_TEAM
        assert result["status"] == AgentStatus.SUCCESS


class TestRationale:
    @pytest.mark.parametrize(
        "state",
        [
            _state("claim_handling", "critical"),
            _state("unclassifiable", "low"),
            {"out_of_scope": True, "complaint_category": "general_inquiry", "channel": "online"},
        ],
    )
    def test_the_rationale_is_inert_key_value_pairs(self, state):
        """No caller text, no newline, no punctuation — it cannot carry a line."""
        rationale = RouteDecideNode().execute(state)["routing_rationale"]
        assert "\n" not in rationale
        for pair in rationale.split(" "):
            key, sep, value = pair.partition("=")
            assert sep == "="
            assert key.replace("_", "").isalpha()
            assert value.replace("_", "").isalnum()

    def test_caller_identifiers_never_enter_the_rationale(self):
        result = RouteDecideNode().execute(_state(agency_id="agency_017", case_ref="case_9001"))
        assert "agency_017" not in result["routing_rationale"]
        assert "case_9001" not in result["routing_rationale"]


class TestAuditRecord:
    def test_the_decision_is_audited_on_both_paths(self, monkeypatch):
        seen = []
        monkeypatch.setattr(
            "src.nodes.route_decide_node.emit_trace_event",
            lambda event, payload, state: seen.append((event, payload)),
        )
        RouteDecideNode().execute(_state())
        RouteDecideNode().execute({"out_of_scope": True, "channel": "online"})
        assert [event for event, _ in seen] == ["routing_decision", "routing_decision"]
        for _event, payload in seen:
            assert payload["template_id"] == "INS-C2-053"
            assert payload["routing_target_team"] in ROUTING_TEAMS

    def test_the_audit_payload_never_carries_the_complaint(self, monkeypatch):
        seen = []
        monkeypatch.setattr(
            "src.nodes.route_decide_node.emit_trace_event",
            lambda event, payload, state: seen.append(payload),
        )
        RouteDecideNode().execute(_state(raw_input_text="090-1234-5678の田中です。保険金の件。"))
        assert seen
        for payload in seen:
            assert "raw_input_text" not in payload
            assert "090-1234-5678" not in repr(payload)


class TestDeclaredConfigurationIsLive:
    def test_the_declared_fallback_service_level_is_used(self):
        node = RouteDecideNode({"routing": {"fallback_sla_hours": 12}})
        assert node.fallback_sla_hours == 12
        assert node.execute(_state("unclassifiable", "low"))["routing_sla_hours"] == 12
        assert node.execute({"out_of_scope": True, "channel": "online"})["routing_sla_hours"] == 12

    @pytest.mark.parametrize("value", [0, 5000, "NaN", float("inf"), True, None, 2.5])
    def test_an_out_of_range_declaration_is_dropped_not_clamped(self, value):
        node = RouteDecideNode({"routing": {"fallback_sla_hours": value}})
        assert node.fallback_sla_hours == DEFAULT_FALLBACK_SLA_HOURS
