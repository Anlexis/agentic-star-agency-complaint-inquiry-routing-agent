"""AgentCore Platform v1.0"""

# INS-C2-053 — Outer graph (AgentBaseGraph; Cat 2 two-layer nested architecture)
#
# Architecture:
#
#   Outer backbone (fixed — never override add_edges()):
#     START -> initialize -> pre_process -> main -> post_process -> finalize -> END
#
#   pre_process  owns the caller contract (src/nodes/pre_process_node.py)
#   main         is a GraphNode delegating to the inner DomainWorkflowGraph
#   post_process owns the output boundary (src/nodes/post_process_node.py)
#
# All three backbone slots must be filled. AgentBaseGraph.compile() raises
# MissingNodeError when pre_process, main or post_process is None — this
# template registered only `main`, so Graph().compile() raised, src/api/server.py
# could not even be imported, and the deployed agent could not answer anything.
# The unit suite never noticed because no test compiled the outer graph.
#
# Directory layout:
#   src/graph/graph.py                 <- outer graph (this file)
#   src/graph/domain_workflow_graph.py <- inner graph (classify -> route -> assemble)
#   src/graph/context_bridge.py        <- carries caller fields across the boundary

import logging
import os
from typing import Any, ClassVar, Dict, Mapping, Optional

from framework.graph.agent_base_graph import AgentBaseGraph
from framework.nodes.graph_node import GraphNode
from framework.schemas.agent_state import AgentState
from framework.schemas.agent_status import AgentStatus
from framework.schemas.trust_level import TrustLevel

from shared.utils.audit_logger import emit_trace_event

from src.graph.context_bridge import set_domain_context
from src.nodes.post_process_node import PostProcessNode
from src.nodes.pre_process_node import PreProcessNode
from src.schemas.state import State
from src.schemas.vocabulary import WITHHELD_NOTICE_KEY

logger = logging.getLogger(__name__)

# Runtime parameters live at config/config.yaml, three levels up from this file
# (src/graph/graph.py -> src/graph -> src -> repository root). config/agent.yaml
# is the flat discovery manifest and deliberately carries no runtime block.
CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "config",
    "config.yaml",
)


def load_runtime_config() -> Dict[str, Any]:
    """Return the runtime parameters declared in config/config.yaml.

    Best-effort by design: a missing or unparseable file yields an empty
    mapping, graph construction still succeeds, and every consumer falls back to
    its documented default. PyYAML is imported on demand because it is a
    framework runtime dependency rather than one this template declares.

    This function is what makes the declarations in config/config.yaml real on
    the standalone path. Constructing the graph with no argument leaves
    ``self.config`` empty, and every declared value is then read from a mapping
    that was never populated — the file looks like configuration and behaves
    like a comment.
    """
    try:
        import yaml

        with open(CONFIG_PATH, "r", encoding="utf-8") as handle:
            loaded = yaml.safe_load(handle) or {}
        return loaded if isinstance(loaded, dict) else {}
    except Exception:  # noqa: BLE001 — a config read must never break construction
        logger.warning("config/config.yaml could not be read; using defaults")
        return {}


class DomainWorkflowGraphNode(GraphNode):
    """The `main` slot: delegates the domain workflow to the inner graph.

    Contracts:
      get_subgraph()  — build and return DomainWorkflowGraph with this agent's config
      extract_input() — the single string the inner graph receives, plus the
                        bridge publication that carries everything else
      merge_output()  — map sub_result into the outer state delta (changed keys only)
      on_subgraph_error() — the contained failure partial for the path that skips
                        post_process entirely
    """

    # "handle", not "propagate". With "propagate" an inner validation failure
    # became a raised SubgraphError, which the node wrapper turns into a bare
    # error partial carrying a traceback in error_log and no reason a caller
    # could act on — and, critically, without clearing the output-bearing state
    # fields. Handling it here is what makes the failure both readable and
    # contained on the one path that never reaches post_process.
    error_strategy: ClassVar[str] = "handle"

    # HITL interrupts stay inside the inner graph.
    propagate_hitl: ClassVar[bool] = False

    def __init__(self, config: Optional[Mapping[str, Any]] = None) -> None:
        self._config: Dict[str, Any] = dict(config or {})

    def _extra_security_gate_input(self, state: AgentState) -> AgentState:
        """Short-circuit the subgraph when the caller contract already failed.

        The backbone wires pre_process -> main unconditionally, so a refused
        request would otherwise still be classified and routed. Returning the
        state with its ERROR status intact makes the node wrapper skip execute()
        before the subgraph is ever built.
        """
        return state

    def get_subgraph(self) -> Any:
        """Build the inner domain workflow graph, with this agent's config.

        Imported inside the method to keep module import order independent of
        the inner graph, matching the nested Cat 2 pattern.
        """
        from src.graph.domain_workflow_graph import DomainWorkflowGraph

        return DomainWorkflowGraph(self._config)

    def extract_input(self, state: AgentState) -> str:
        """Return the single string the inner graph is invoked with.

        ``GraphNode.execute()`` forwards exactly this string and nothing else —
        not ``input_context``, not any other outer state key. The validated
        caller fields are therefore published on the context bridge here, one
        statement before the invocation that reads them back in the inner
        graph's ``_extra_initial_state()``.

        The publication REPLACES whatever was there rather than merging into
        it, so a field absent from this request cannot be inherited from a
        previous one.
        """
        validated_context = state.get("validated_context") or {}
        set_domain_context(validated_context)
        text = state.get("validated_input") or state.get("user_input") or ""
        return str(text)

    def merge_output(self, state: AgentState, sub_result: Dict[str, Any]) -> Dict[str, Any]:
        """Map the inner sub_result into the outer state delta (changed keys only).

        Written together with DomainWorkflowGraph.get_output(), which emits
        result / status / out_of_scope / rejection_reason / error_log / trace_id.
        """
        return {
            "result": sub_result.get("result"),
            "status": sub_result.get("status"),
            "out_of_scope": sub_result.get("out_of_scope"),
            "rejection_reason": sub_result.get("rejection_reason"),
        }

    def on_subgraph_error(self, state: AgentState, error: Exception) -> Dict[str, Any]:
        """Report an inner failure without shipping anything from it.

        This is the one path that never reaches post_process — the backbone
        routes a non-success status straight to finalize — so containment has to
        happen here or not at all. Every output-bearing field is cleared, the
        replacement notice is truthy (a falsy one re-opens the envelope's
        fallback to the un-gated inner result), and the reason is a closed-set
        label rather than the exception text, which would otherwise carry a
        traceback into the caller's envelope.
        """
        reason = str(state.get("rejection_reason") or "upstream_error")
        emit_trace_event(
            "subgraph_failed",
            {"reason": reason, "error_type": type(error).__name__},
            state,
        )
        logger.error(
            "DomainWorkflowGraphNode: inner graph failed (%s / %s)",
            reason,
            type(error).__name__,
        )
        return {
            "formatted_output": {WITHHELD_NOTICE_KEY: reason},
            "result": None,
            "rejection_reason": reason,
            "status": AgentStatus.ERROR.value,
        }


class InsuranceComplaintRoutingAgent(AgentBaseGraph):
    """Outer graph for INS-C2-053 (Cat 2 — classification and routing).

    Inherits AgentBaseGraph directly (L1 Base). The domain logic lives entirely
    in DomainWorkflowGraphNode's inner graph; this class only fills the backbone
    slots and hands each of them the runtime configuration.

    Backbone (fixed):
        START -> initialize -> pre_process -> main -> post_process -> finalize -> END

    add_edges() is NOT overridden — backbone wiring belongs to the framework.
    get_output() is NOT overridden either: post_process publishes the whole
    routing decision as formatted_output, so the framework's envelope already
    carries the complete product. An override that blanked the output on a
    non-success status would additionally mask post_process's own clearing,
    making that clearing impossible to falsify.
    """

    def __init__(self, config: Optional[Mapping[str, Any]] = None) -> None:
        """Resolve the runtime configuration once, for both load paths.

        The registry constructs the agent with the parsed config/config.yaml.
        A standalone construction passes nothing, and the declared values would
        then be read from an empty mapping — so the file is loaded here instead
        of being assumed present. Either way ``self.config`` holds the same
        settings, and the nodes below receive them at construction time.
        """
        super().__init__(dict(config) if config else load_runtime_config())

    @property
    def name(self) -> str:
        """Agent identifier registered with the agent registry."""
        return "INS-C2-053"

    @property
    def state_schema(self) -> type:
        return State

    def register_nodes(self) -> None:
        """Fill every backbone slot.

        super().register_nodes() must be called first — it injects the
        framework's InitializeNode and FinalizeNode and sets the three domain
        slots to None. All three are then filled: compile() raises
        MissingNodeError if any is left empty.

        Each node receives self.config, which the registry loads from
        config/config.yaml. That is the only path a declared runtime value can
        travel: the framework invokes a node as execute(state), so a `config`
        parameter on execute is never supplied and anything read from it is
        dead.
        """
        super().register_nodes()  # fills initialize + finalize
        self._nodes["pre_process"] = PreProcessNode(self.config)
        self._nodes["main"] = DomainWorkflowGraphNode(self.config)
        self._nodes["post_process"] = PostProcessNode(self.config)

    # add_edges() is NOT overridden — backbone wiring belongs to the framework.


# The manifest points at src.graph.graph.InsuranceComplaintRoutingAgent, and the
# package re-exports the class as well (see src/graph/__init__.py), so a
# registry load resolves whichever form it is given.
Graph = InsuranceComplaintRoutingAgent

# The declared entry trust level, kept next to the graph it applies to so the
# adapter and the manifest cannot drift apart silently.
ENTRY_TRUST_LEVEL: TrustLevel = TrustLevel.VERIFIED_EXTERNAL
