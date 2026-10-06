"""AgentCore Platform v1.0"""

# INS-C2-053 — DomainWorkflowGraph (inner BaseGraph)
#
# The inner graph of the two-layer nested Cat 2 architecture. It holds the whole
# complaint classification and routing workflow:
#
#   START -> input_validate -> complaint_classify -> severity_score
#         -> route_decide -> output_validate -> END
#
# Out-of-scope handling is NOT a topology short-circuit. A non-complaint or
# unclassifiable input flows through every node: complaint_classify sets
# out_of_scope, severity_score returns no score, and route_decide sends it to
# the general enquiry desk. route_decide is the only node that sets a routing
# team, and output_validate requires one on every path — so the flow has to
# pass through it. An input that produces no route is the failure this template
# exists to prevent, and the topology is what makes that structural.
#
# Called by DomainWorkflowGraphNode.get_subgraph() in graph.py; get_output()
# shapes the dict that node's merge_output() consumes.
#
# ── Trust levels on the inner nodes ─────────────────────────────────────────
# Every domain node below declares ANONYMOUS, explicitly. That is not a relaxed
# boundary; it is where the boundary actually is.
#
# GraphNode.execute() passes the outer InvocationContext into the subgraph
# unchanged — there is no escalation across the boundary, so an inner node
# demanding a HIGHER level than the entry contract cannot admit anyone the
# entry did not, and can only deny people the entry did. Trust orders
# ANONYMOUS < VERIFIED_EXTERNAL < INTERNAL, and the manifest's entry level is
# VERIFIED_EXTERNAL, so inner nodes marked INTERNAL — which is what this
# template shipped — denied every real external caller at S-1. The subgraph
# then errored, the backbone skipped post_process, and the caller received an
# empty envelope. The gate was not protecting anything; it was the failure.
#
# The external boundary lives on the backbone's pre_process node, which
# declares VERIFIED_EXTERNAL and owns the caller contract.
#
# Rules enforced:
#   - inherits BaseGraph (fully custom topology, no forced backbone)
#   - implements all 7 BaseGraph abstract methods
#   - register_nodes() does NOT call super() — it is abstract in BaseGraph
#   - does NOT register initialize / finalize (outer backbone concerns)
#   - add_edges() lives here, not on the outer graph
#   - no platform SDK imports

from typing import Any, Dict, Mapping, Optional

from langgraph.graph import END, START

from framework.graph.base_graph import BaseGraph
from framework.schemas.agent_status import AgentStatus
from src.graph.context_bridge import get_domain_context
from src.nodes.complaint_classify_node import ComplaintClassifyNode
from src.nodes.input_validate_node import InputValidateNode
from src.nodes.output_validate_node import OutputValidateNode
from src.nodes.route_decide_node import RouteDecideNode
from src.nodes.severity_score_node import SeverityScoreNode
from src.schemas.state import State

# The caller fields the bridge carries across the subgraph boundary, with the
# value each falls back to when the bridge is empty (a direct inner-graph
# invocation, or a test). Listing them here rather than copying whatever the
# bridge holds keeps the inner graph's input surface closed.
BRIDGED_FIELDS: Dict[str, Any] = {
    "channel": "online",
    "agency_id": None,
    "case_ref": None,
    "prior_complaint_count": 0,
}


class DomainWorkflowGraph(BaseGraph):
    """Inner domain workflow graph for INS-C2-053.

    Pipeline:
        START
          -> input_validate     (residual masking + instruction screen)
          -> complaint_classify (keyword classification)
          -> severity_score     (additive rule scoring; skips out-of-scope)
          -> route_decide       (routing table + audit; general desk when oos)
          -> output_validate    (completeness + assembly)
          -> END

    All nodes are FunctionNode subclasses returning partial-dict state updates.
    initialize / finalize are outer backbone concerns and are not registered.
    """

    def __init__(self, config: Optional[Mapping[str, Any]] = None) -> None:
        """Take the runtime configuration and hand each node its own slice.

        The framework calls ``execute(state)`` with a single argument, so a
        ``config`` parameter on a node's ``execute`` is never supplied. Passing
        configuration at construction time is the only route that carries a
        value, which is why the nodes below are built with one.
        """
        super().__init__(dict(config or {}))

    # ── Identity ──────────────────────────────────────────────────────────────

    @property
    def name(self) -> str:
        """Unique identifier for this inner graph."""
        return "ins_c2_053_domain_workflow"

    @property
    def state_schema(self) -> type:
        """TypedDict shared by the inner and outer graph."""
        return State

    def get_state_class(self) -> type:
        """Return the State TypedDict used by both graphs."""
        return State

    # ── Config validation ─────────────────────────────────────────────────────

    def _validate_config(self) -> None:
        """No mandatory keys.

        Every tunable is optional and individually bounds-checked by the node
        that reads it, which drops an out-of-range value in favour of its
        documented default rather than clamping it. There is therefore no
        configuration that can be missing here, and none that can be wrong in a
        way this method could catch earlier than its reader does.
        """
        return None

    # ── Initial state ─────────────────────────────────────────────────────────

    def _extra_initial_state(self) -> Dict[str, Any]:
        """Seed the caller fields the subgraph boundary does not forward.

        ``GraphNode.execute()`` calls ``subgraph.invoke(user_input, ...)`` and
        passes nothing else — not ``input_context``, not any other outer state
        key. Without this hook the channel, the case identifiers and the prior
        complaint count are absent on every request, whatever the caller sent,
        and the declared contract exists only above the boundary.

        The outer node publishes the validated fields immediately before
        invoking; this reads them back. Only the fields named in BRIDGED_FIELDS
        are taken, so nothing else can ride across.
        """
        published = get_domain_context()
        return {name: published.get(name, default) for name, default in BRIDGED_FIELDS.items()}

    # ── Node registration ─────────────────────────────────────────────────────

    def register_nodes(self) -> None:
        """Register the five domain nodes in pipeline order.

        No super() call — BaseGraph.register_nodes() is abstract. initialize and
        finalize belong to the outer backbone. Every key registered here is
        referenced in add_edges().
        """
        self._nodes["input_validate"] = InputValidateNode(self.config)
        self._nodes["complaint_classify"] = ComplaintClassifyNode(self.config)
        self._nodes["severity_score"] = SeverityScoreNode(self.config)
        self._nodes["route_decide"] = RouteDecideNode(self.config)
        self._nodes["output_validate"] = OutputValidateNode(self.config)

    # ── Edge wiring ───────────────────────────────────────────────────────────

    def add_edges(self) -> None:
        """Wire the classify -> score -> route -> assemble topology.

        The only branch is an error short-circuit after classification, so a
        refused input reaches the assembly node directly and is reported rather
        than scored. Out-of-scope input is not short-circuited: it must pass
        through route_decide, the only node that assigns a routing team.
        """
        self._sg.add_edge(START, "input_validate")
        self._sg.add_edge("input_validate", "complaint_classify")
        self._sg.add_conditional_edges(
            "complaint_classify",
            self.route,
            {
                "output_validate": "output_validate",
                "severity_score": "severity_score",
            },
        )
        self._sg.add_edge("severity_score", "route_decide")
        self._sg.add_edge("route_decide", "output_validate")
        self._sg.add_edge("output_validate", END)

    # ── Routing ───────────────────────────────────────────────────────────────

    def route(self, state: State) -> str:
        """Branch after complaint_classify.

        ⚠️ The annotation on this method is the graph's OWN State, and that is
        load-bearing rather than cosmetic. LangGraph reads a path callable's
        annotation as its input schema and PROJECTS AWAY every field the schema
        does not name. Annotated ``AgentState`` — which is what this method
        carried — the fields this template adds would be absent here on every
        call, and any branch that depended on one would silently never be
        taken, with the unit suite still green because a unit test calls the
        method directly and never goes through the projection.

        Only a classify-time error short-circuits. Out-of-scope input is not
        short-circuited: it has to reach route_decide, the only node that
        assigns a routing team, and it ends in SUCCESS at the general desk
        rather than in an error.
        """
        if state.get("status") == AgentStatus.ERROR.value:
            return "output_validate"
        return "severity_score"

    # ── Output shape ──────────────────────────────────────────────────────────

    def get_output(self, state: State) -> Dict[str, Any]:
        """Shape the dict returned to the outer graph as sub_result.

        Read by DomainWorkflowGraphNode.merge_output(); the two are written
        together so the field names cannot drift apart.
        """
        return {
            "result": state.get("result"),
            "status": state.get("status"),
            "out_of_scope": state.get("out_of_scope"),
            "rejection_reason": state.get("rejection_reason"),
            "error_log": state.get("error_log", []),
            "trace_id": state.get("trace_id"),
        }
