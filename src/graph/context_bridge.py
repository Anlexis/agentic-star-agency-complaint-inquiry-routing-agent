"""AgentCore Platform v1.0"""

# Caller-context bridge between the outer backbone and the inner domain graph.
#
# Why this file exists
# --------------------
# ``GraphNode.execute()`` calls ``subgraph.invoke(user_input, session_id=..., ctx=...)``
# and nothing else. One string crosses the boundary. Neither ``input_context``
# nor any other outer state key is forwarded, so an inner node reading
# ``state["channel"]`` gets ``None`` on every request no matter what the caller
# sent — the declared caller contract exists only above the boundary.
#
# Two ways across are available. Serialising the fields into the single string
# and re-parsing them in the first inner node works, but it puts structured data
# back into the field the platform treats as free user text, where the input
# gate masks and screens it. A ContextVar keeps them out of that channel.
#
# The outer ``extract_input()`` stashes the validated fields here immediately
# before ``invoke()``; the inner graph's ``_extra_initial_state()`` reads them
# back while building its initial state. Both run in the same thread and the
# same context, one directly after the other.
#
# What makes this safe against request bleed
# ------------------------------------------
# A ContextVar is per-context, and each request already runs in its own context
# (the adapter dispatches the synchronous invoke through a worker thread, which
# copies the caller's context rather than sharing it). Every stash REPLACES the
# whole mapping rather than merging into it, so a field absent from this
# request cannot be inherited from the last one even if a context were reused.
# ``clear_domain_context()`` is provided for tests that want to prove exactly
# that.

from contextvars import ContextVar
from typing import Any, Dict, Mapping

_DOMAIN_CONTEXT: ContextVar[Dict[str, Any]] = ContextVar("ins_c2_053_domain_context", default={})


def set_domain_context(context: Mapping[str, Any]) -> None:
    """Publish the validated caller fields for the inner graph to pick up.

    Stores a copy: the caller keeps its mapping, and a later mutation of the
    original cannot reach into a request that has already started.
    """
    _DOMAIN_CONTEXT.set(dict(context))


def get_domain_context() -> Dict[str, Any]:
    """Return a copy of the validated caller fields, or an empty mapping."""
    return dict(_DOMAIN_CONTEXT.get())


def clear_domain_context() -> None:
    """Drop any published fields — used by tests to prove no value carries over."""
    _DOMAIN_CONTEXT.set({})
