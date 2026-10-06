# Unit tests for the two graphs and the bridge between them.
#
# The properties here are the ones a node-level test structurally cannot see:
# whether the graphs COMPILE, whether every backbone slot is filled, whether the
# caller fields actually cross the subgraph boundary, and whether the routing
# callable sees the fields it branches on.

import pytest
from langgraph.graph import END, START

from framework.graph.agent_base_graph import AgentBaseGraph
from framework.nodes.graph_node import GraphNode
from framework.schemas.agent_status import AgentStatus
from framework.schemas.invocation_context import InvocationContext
from framework.schemas.trust_level import TrustLevel
from src.graph.context_bridge import (
    clear_domain_context,
    get_domain_context,
    set_domain_context,
)
from src.graph.domain_workflow_graph import BRIDGED_FIELDS, DomainWorkflowGraph
from src.graph.graph import (
    DomainWorkflowGraphNode,
    Graph,
    InsuranceComplaintRoutingAgent,
    load_runtime_config,
)
from src.schemas.state import State
from src.schemas.vocabulary import WITHHELD_NOTICE_KEY

COMPLAINT = "保険金の支払拒否について納得できない。苦情です。"


@pytest.fixture(autouse=True)
def _clean_bridge():
    clear_domain_context()
    yield
    clear_domain_context()


class TestOuterGraphCompiles:
    def test_every_backbone_slot_is_filled(self):
        """compile() raises MissingNodeError when a domain slot is left empty.

        This template registered only `main`, so Graph().compile() raised and
        src/api/server.py could not be imported at all — while the unit suite
        stayed green, because nothing in it compiled the outer graph.
        """
        agent = Graph()
        agent.compile()
        assert set(agent._nodes) == {
            "initialize",
            "pre_process",
            "main",
            "post_process",
            "finalize",
        }
        assert all(node is not None for node in agent._nodes.values())

    def test_it_inherits_the_l1_base_directly(self):
        assert issubclass(InsuranceComplaintRoutingAgent, AgentBaseGraph)
        assert isinstance(Graph()._nodes.get("main", None) or DomainWorkflowGraphNode(), GraphNode)

    def test_the_backbone_wiring_is_not_overridden(self):
        assert "add_edges" not in InsuranceComplaintRoutingAgent.__dict__

    def test_the_envelope_is_not_overridden(self):
        """An override blanking output on error would mask post_process's clearing."""
        assert "get_output" not in InsuranceComplaintRoutingAgent.__dict__

    def test_the_state_schema_is_the_templates_own(self):
        assert Graph().state_schema is State

    def test_the_name_matches_the_manifest_id(self):
        assert Graph().name == "INS-C2-053"


class TestInnerGraphCompiles:
    def test_the_inner_graph_compiles(self):
        """A State that re-declares an inherited channel makes this impossible.

        AgentState declares error_log as an accumulating channel; restating it
        as a plain list raised
        ValueError: Channel 'error_log' already exists with a different type.
        """
        inner = DomainWorkflowGraph()
        inner.compile()
        assert set(inner._nodes) == {
            "input_validate",
            "complaint_classify",
            "severity_score",
            "route_decide",
            "output_validate",
        }

    def test_it_registers_no_backbone_node(self):
        inner = DomainWorkflowGraph()
        inner.register_nodes()
        assert "initialize" not in inner._nodes
        assert "finalize" not in inner._nodes

    def test_every_registered_node_is_wired(self):
        inner = DomainWorkflowGraph()
        inner.compile()
        wired = set(inner._compiled.get_graph().nodes) - {START, END, "__start__", "__end__"}
        assert set(inner._nodes) <= wired


class TestRouteAnnotation:
    def test_the_path_callable_is_annotated_with_the_graphs_own_state(self):
        """LangGraph projects away every field the annotation does not name.

        Annotated AgentState, this template's own fields would be absent inside
        the callable and a branch depending on one would silently never be
        taken — with the unit suite green, because a unit test calls the method
        directly and never goes through the projection.
        """
        assert DomainWorkflowGraph.route.__annotations__["state"] is State

    def test_an_error_short_circuits_to_the_assembly_node(self):
        inner = DomainWorkflowGraph()
        assert inner.route({"status": AgentStatus.ERROR.value}) == "output_validate"

    def test_out_of_scope_is_not_short_circuited(self):
        """It must reach route_decide — the only node that assigns a team."""
        inner = DomainWorkflowGraph()
        assert inner.route({"status": AgentStatus.SUCCESS.value, "out_of_scope": True}) == "severity_score"


class TestContextBridge:
    def test_publish_then_read(self):
        set_domain_context({"channel": "counter", "agency_id": "agency_017"})
        assert get_domain_context() == {"channel": "counter", "agency_id": "agency_017"}

    def test_publication_replaces_rather_than_merges(self):
        """A field absent from this request cannot be inherited from the last."""
        set_domain_context({"channel": "counter", "agency_id": "agency_017"})
        set_domain_context({"channel": "online"})
        assert get_domain_context() == {"channel": "online"}

    def test_a_returned_copy_cannot_mutate_the_published_mapping(self):
        set_domain_context({"channel": "counter"})
        borrowed = get_domain_context()
        borrowed["channel"] = "tampered"
        assert get_domain_context()["channel"] == "counter"

    def test_the_initial_state_takes_only_the_declared_fields(self):
        set_domain_context({"channel": "phone", "smuggled": "anything"})
        seeded = DomainWorkflowGraph()._extra_initial_state()
        assert set(seeded) == set(BRIDGED_FIELDS)
        assert seeded["channel"] == "phone"
        assert "smuggled" not in seeded

    def test_an_empty_bridge_falls_back_to_the_documented_defaults(self):
        clear_domain_context()
        assert DomainWorkflowGraph()._extra_initial_state() == dict(BRIDGED_FIELDS)


class TestGraphNodeContracts:
    def test_extract_input_returns_a_string_and_publishes_the_context(self):
        """GraphNode forwards exactly one string; everything else needs the bridge."""
        node = DomainWorkflowGraphNode()
        text = node.extract_input({"validated_input": COMPLAINT, "validated_context": {"channel": "counter"}})
        assert isinstance(text, str)
        assert text == COMPLAINT
        assert get_domain_context() == {"channel": "counter"}

    def test_extract_input_prefers_the_validated_text(self):
        node = DomainWorkflowGraphNode()
        assert node.extract_input({"validated_input": "a", "user_input": "b"}) == "a"
        assert node.extract_input({"user_input": "b"}) == "b"
        assert node.extract_input({}) == ""

    def test_merge_output_returns_only_changed_keys(self):
        node = DomainWorkflowGraphNode()
        merged = node.merge_output(
            {},
            {
                "result": {"routing": {}},
                "status": AgentStatus.SUCCESS,
                "out_of_scope": False,
                "rejection_reason": None,
                "trace_id": "t",
                "error_log": ["noise"],
            },
        )
        assert set(merged) == {"result", "status", "out_of_scope", "rejection_reason"}

    def test_the_subgraph_is_built_with_the_agents_config(self):
        node = DomainWorkflowGraphNode({"severity": {"counter_channel_delta": 11}})
        inner = node.get_subgraph()
        assert isinstance(inner, DomainWorkflowGraph)
        assert inner.config["severity"]["counter_channel_delta"] == 11

    def test_the_error_strategy_is_handled_not_propagated(self):
        """propagate reduces an inner failure to a traceback with no reason,
        and leaves the output-bearing fields untouched on the one path that
        never reaches post_process."""
        assert DomainWorkflowGraphNode.error_strategy != "propagate"

    def test_on_subgraph_error_contains_and_explains(self):
        node = DomainWorkflowGraphNode()
        contained = node.on_subgraph_error(
            {"rejection_reason": "routing_incomplete", "result": {"leak": "x"}},
            RuntimeError("inner detail that must not ship"),
        )
        assert contained["status"] == AgentStatus.ERROR.value
        assert contained["result"] is None
        assert contained["formatted_output"] == {WITHHELD_NOTICE_KEY: "routing_incomplete"}
        assert contained["formatted_output"]
        assert "inner detail" not in repr(contained)
        assert "Traceback" not in repr(contained)

    def test_an_unnamed_inner_failure_still_reports_a_closed_set_reason(self):
        contained = DomainWorkflowGraphNode().on_subgraph_error({}, ValueError("x"))
        assert contained["formatted_output"] == {WITHHELD_NOTICE_KEY: "upstream_error"}


class TestRuntimeConfiguration:
    def test_the_declared_file_is_loaded_when_none_is_passed(self):
        """Graph() with no argument used to get an empty config, which made
        every declaration in config/config.yaml a comment."""
        declared = load_runtime_config()
        assert declared, "config/config.yaml should not be empty"
        assert Graph().config == declared

    def test_an_explicit_config_wins(self):
        agent = Graph({"max_retry": 1, "severity": {"counter_channel_delta": 9}})
        assert agent.config["max_retry"] == 1

    def test_the_declared_values_reach_the_nodes(self):
        agent = Graph({"severity": {"counter_channel_delta": 17}, "limits": {"max_input_chars": 500}})
        agent.compile()
        assert agent._nodes["pre_process"].max_input_chars == 500
        inner = agent._nodes["main"].get_subgraph()
        inner.register_nodes()
        assert inner._nodes["severity_score"].counter_channel_delta == 17


class TestInnerGraphEndToEnd:
    def test_the_inner_graph_answers_on_its_own(self):
        set_domain_context(
            {"channel": "counter", "agency_id": "agency_017", "case_ref": "case_9001", "prior_complaint_count": 1}
        )
        inner = DomainWorkflowGraph()
        inner.compile()
        ctx = InvocationContext(session_id="inner", caller_trust_level=TrustLevel.VERIFIED_EXTERNAL)
        result = inner.invoke(COMPLAINT, ctx=ctx)
        assert result["status"] == AgentStatus.SUCCESS.value
        assert result["result"]["routing"]["team"] == "claims_team"
        assert result["result"]["case"]["channel"] == "counter"
        assert result["result"]["case"]["agency_id"] == "agency_017"
