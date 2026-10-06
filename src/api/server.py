"""AgentCore Platform v1.0"""

# Standalone HTTP entry point. Adapters carry no business logic — for
# platform-level routing the gateway calls agent.invoke() directly and this
# module is not used.
#
# Five things this adapter owns, each because of what happened without it.
#
#   1. Caller authentication. The agent's trust boundary admits a
#      VERIFIED_EXTERNAL caller; an unauthenticated one is ANONYMOUS. Nothing
#      here set the trust level at all, so every request was refused at the
#      first node that asked for one and answered 200 with an empty result — a
#      deployment that looks healthy and serves nothing. Both deployment
#      credentials are accepted: the ordinary bearer, and the separate
#      STG-runner credential the sign-off harness presents when an entry
#      contract is declared INTERNAL, so a later change to the declared level
#      cannot leave the harness unauthenticated.
#
#   2. The runtime configuration. `Graph()` with no argument gets an empty
#      config, and every declared value in config/config.yaml is then dead. The
#      agent loads the file itself, so the standalone path and the registry
#      path resolve the same settings.
#
#   3. Dropping undeclared context fields, before invoke(). A validator that
#      merely ignores an unknown key leaves it in input_context — and the first
#      node returns input_context verbatim in its result, where the framework's
#      output gate scans every value and raises. One stray field therefore ends
#      the request at node one with a traceback the caller cannot act on.
#      Dropping is what makes the declared contract an actual boundary.
#
#   4. Credential screening, before invoke(), for the same reason: the request
#      cannot succeed either way, so a 400 naming the field beats an opaque
#      node-one failure. The screen's floor is the framework's own detector, so
#      what is refused here is exactly what would be blocked one layer later.
#
#   5. The declared request budget. `timeout_s` in config/config.yaml is
#      enforced here as a wall-clock limit on the invocation; a declared value
#      with no reader would be worse than no declaration.
#
# 400 is used rather than 422: pydantic owns 422 and returns a list of error
# objects there, so reusing it would make client handling ambiguous.

import asyncio
import logging
import os
import secrets
from typing import Any, Dict, cast
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from framework.schemas.invocation_context import InvocationContext
from framework.schemas.trust_level import TrustLevel
from framework.secrets.context import bound_secrets
from shared.secrets import factory as secrets_factory

from src.graph.graph import ENTRY_TRUST_LEVEL, Graph, load_runtime_config
from src.nodes.pre_process_node import CONTEXT_FIELDS, MAX_CONTEXT_FIELDS
from src.services.service import (
    bounded_config_int,
    config_section,
    detect_output_credentials,
    first_credential_field,
    is_inert_token,
    is_session_token,
)

logger = logging.getLogger(__name__)

app = FastAPI(title="Agent")

_RUNTIME_CONFIG: Dict[str, Any] = load_runtime_config()

agent = Graph(_RUNTIME_CONFIG)
agent.compile()
agent.provision_secrets(secrets_factory(namespace="ins", agent_name="InsuranceAgencyComplaintRoutingAgent"))

# Mirrors the bound PreProcessNode enforces, so an oversized body is refused
# before it is parsed into state rather than after.
MAX_INPUT_CHARS: int = bounded_config_int(
    config_section(_RUNTIME_CONFIG, "limits"), "max_input_chars", 4000, 200, 20000
)

# Wall-clock budget for one invocation. The pipeline is deterministic and does
# no I/O, so exceeding this means something is structurally wrong rather than
# slow; the request is answered 504 and the worker thread is left to finish.
INVOKE_TIMEOUT_S: int = bounded_config_int(_RUNTIME_CONFIG, "timeout_s", 30, 1, 600)


class InvokeRequest(BaseModel):
    """The request contract.

    `input` is the complaint or enquiry text. `input_context` is the closed set
    of structured fields declared in docs/02_design.md; anything else in it is
    dropped rather than forwarded.
    """

    input: str
    session_id: str = ""
    input_context: Dict[str, Any] = Field(default_factory=dict)


def _resolve_trust(request: Request) -> TrustLevel:
    """Authenticate the caller and return the trust level it earns.

    Raises 503 when the deployment has no credential configured at all — with
    nothing able to authenticate a caller, nothing this endpoint returns could
    be an answer, and saying so once at the door beats refusing silently inside
    the graph on every request.
    """
    trust = getattr(request.state, "trust_level", TrustLevel.ANONYMOUS)
    if trust is not TrustLevel.ANONYMOUS:
        return cast(TrustLevel, trust)

    external_token = os.environ.get("INVOKE_AUTH_TOKEN", "")
    internal_token = os.environ.get("STG_INTERNAL_RUNNER_TOKEN", "")
    if not external_token and not internal_token:
        raise HTTPException(
            status_code=503,
            detail="Caller authentication is not configured on this deployment.",
        )

    supplied = request.headers.get("authorization", "")
    # Compare bytes: compare_digest raises TypeError on non-ASCII str input
    # (headers decode as latin-1), which would 500 instead of the generic 401.
    supplied_bytes = supplied.encode("utf-8", "replace")
    if internal_token and secrets.compare_digest(supplied_bytes, f"Bearer {internal_token}".encode()):
        return TrustLevel.INTERNAL
    if external_token and secrets.compare_digest(supplied_bytes, f"Bearer {external_token}".encode()):
        return TrustLevel.VERIFIED_EXTERNAL

    # Generic body on purpose — do not leak whether the token was absent,
    # malformed, or simply wrong.
    raise HTTPException(status_code=401, detail="Token is invalid or expired.")


def _declared_context(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Return only the declared context fields, dropping everything else."""
    if len(raw) > MAX_CONTEXT_FIELDS:
        raise HTTPException(
            status_code=400,
            detail=f"Field 'input_context' carries more than {MAX_CONTEXT_FIELDS} fields.",
        )
    return {name: value for name, value in raw.items() if name in CONTEXT_FIELDS}


@app.post("/invoke")
async def invoke(req: InvokeRequest, request: Request) -> Dict[str, Any]:
    trust = _resolve_trust(request)

    if len(req.input) > MAX_INPUT_CHARS:
        raise HTTPException(
            status_code=400,
            detail=f"Field 'input' exceeds the {MAX_INPUT_CHARS}-character limit.",
        )

    if req.session_id and not is_session_token(req.session_id):
        raise HTTPException(
            status_code=400,
            detail="Field 'session_id' must be 1-64 characters of [A-Za-z0-9_-].",
        )

    for field, value in (("input", req.input), ("session_id", req.session_id)):
        if detect_output_credentials(value):
            # Name the field, never the value and never the pattern's match.
            raise HTTPException(
                status_code=400,
                detail=f"Field '{field}' carries a credential-shaped value and was refused.",
            )

    context = _declared_context(req.input_context)
    offending = first_credential_field(context)
    if offending is not None:
        # Echo the field name only when the name is itself an inert token; a
        # caller-chosen name is caller data like any other.
        label = offending if is_inert_token(offending) else "an undeclared position"
        raise HTTPException(
            status_code=400,
            detail=(f"Field 'input_context.{label}' carries a credential-shaped value " "and was refused."),
        )

    with bound_secrets(agent._secrets_provider):
        ctx = InvocationContext(
            session_id=req.session_id or str(uuid4()),
            caller_trust_level=trust,
            caller_id=getattr(request.state, "caller_id", ""),
        )
        try:
            # The framework wheel ships no type information, so invoke() is Any.
            return cast(
                Dict[str, Any],
                await asyncio.wait_for(
                    asyncio.to_thread(agent.invoke, req.input, ctx=ctx, input_context=context),
                    timeout=INVOKE_TIMEOUT_S,
                ),
            )
        except asyncio.TimeoutError:
            logger.error("invoke exceeded the declared %ss budget", INVOKE_TIMEOUT_S)
            raise HTTPException(
                status_code=504,
                detail=f"The request exceeded the {INVOKE_TIMEOUT_S}-second budget.",
            )


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok", "agent": "InsuranceAgencyComplaintRoutingAgent"}


__all__ = [
    "ENTRY_TRUST_LEVEL",
    "INVOKE_TIMEOUT_S",
    "MAX_INPUT_CHARS",
    "agent",
    "app",
]
