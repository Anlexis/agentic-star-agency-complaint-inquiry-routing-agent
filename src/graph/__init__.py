"""AgentCore Platform v1.0"""

# Re-export the agent class at package level.
#
# The registry resolves a manifest's dotted `class:` by importing the module
# part and reading the attribute off it. A manifest written as
# `src.graph.<Class>` therefore looks the class up on THIS module — and an
# empty __init__.py has no such attribute, so the registry raises and the agent
# never loads, while every direct `from src.graph.graph import ...` in CI and
# in the tests keeps working. The failure is invisible until a real
# registry-backed load.
#
# config/agent.yaml uses the fully-qualified `src.graph.graph.<Class>` form.
# The re-export below makes the shorter form resolve too, so neither spelling
# can be the reason the agent does not start.

from src.graph.graph import (
    DomainWorkflowGraphNode,
    Graph,
    InsuranceComplaintRoutingAgent,
)

__all__ = [
    "DomainWorkflowGraphNode",
    "Graph",
    "InsuranceComplaintRoutingAgent",
]
