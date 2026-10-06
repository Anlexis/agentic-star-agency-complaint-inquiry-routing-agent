# Design Specification — INS-C2-053

## What the agent does

It turns an incoming complaint or enquiry into a routing decision: a category, a severity level and
score, a team and compliance officer, a service level in hours, and a machine-readable rationale.
Non-complaint enquiries also get a decision — the general enquiry desk — because a message that is
recognised and then not routed is the failure the record is meant to prevent.

Everything is deterministic. Classification is keyword matching, severity is an additive rule
table, routing is a table lookup. No model is invoked anywhere, which is what `generation_mode:
deterministic` in `config/agent.yaml` declares.

## Position in the architecture

| Aspect | Value |
|---|---|
| Agent class | `src.graph.graph.InsuranceComplaintRoutingAgent` |
| L1 Base (framework base class) | `AgentBaseGraph` — direct framework inheritance |
| Category | Cat 2 — a domain pipeline, two-layer nested |
| Entry trust level | `VERIFIED_EXTERNAL` |
| State | flat `TypedDict` extending `AgentState` (no Pydantic — msgpack incompatible) |
| Node | `FunctionNode` subclasses overriding `execute(self, state) -> dict` |
| Graph | composition via `register_nodes()` |

### Two layers

The outer graph is the framework's fixed backbone. The `main` slot holds a `GraphNode` that
delegates to an inner `BaseGraph` carrying the whole domain workflow.

```
outer   START → initialize → pre_process → main → {route} → post_process → finalize → END
                                            │
inner                                       └→ START → input_validate → complaint_classify
                                                     → {route} → severity_score → route_decide
                                                     → output_validate → END
```

`add_edges()` is not overridden on the outer graph — backbone wiring belongs to the framework.
All three domain slots are filled: `compile()` raises `MissingNodeError` if `pre_process`, `main`
or `post_process` is left empty.

### Nodes

| Node | Responsibility | Reads | Writes |
|---|---|---|---|
| `initialize` | framework default | — | framework fields |
| `pre_process` (`PreProcessNode`) | the caller contract: bounds, screens, and the closed context field set | `user_input`, `input_context` | `validated_input`, `validated_context`, `rejection_reason` |
| `main` (`DomainWorkflowGraphNode`) | delegate to the inner graph; publish the caller fields on the bridge | `validated_input`, `validated_context` | `result`, `out_of_scope`, `rejection_reason` |
| `post_process` (`PostProcessNode`) | the output boundary: closed vocabulary, credential union, containment | `result` | `formatted_output`, `result` |
| `finalize` | framework default | — | `response_metadata` |
| `input_validate` (inner) | residual personal-data masking; instruction screen at the inner boundary | `user_input` / `raw_input_text` | `raw_input_text` (masked), `is_valid_input` |
| `complaint_classify` (inner) | keyword classification, complaint vs enquiry | `raw_input_text` | `complaint_category`, `classification_confidence`, `is_complaint`, `out_of_scope` |
| `severity_score` (inner) | additive rule scoring | `complaint_category`, `raw_input_text`, `channel`, `prior_complaint_count` | `severity_level`, `severity_score`, `severity_modifiers` |
| `route_decide` (inner) | routing table lookup and the audit record of the decision | `complaint_category`, `severity_level` | `routing_target_*`, `routing_sla_hours`, `routing_rationale` |
| `output_validate` (inner) | completeness of the decision, then assembly | all of the above | `result` |

### Why the caller fields cross the boundary on a bridge

`GraphNode.execute()` invokes the subgraph as `subgraph.invoke(user_input, session_id=..., ctx=...)`.
One string crosses. Neither `input_context` nor any other outer state key is forwarded, so an inner
node reading `state["channel"]` receives `None` on every request no matter what the caller sent.

`src/graph/context_bridge.py` is a `ContextVar`. The outer node's `extract_input()` publishes the
validated fields immediately before invoking; the inner graph's `_extra_initial_state()` reads them
back while building its initial state. Both run in the same thread and the same context, one
statement apart. Each publication replaces the whole mapping rather than merging into it, so a
field absent from one request cannot be inherited from the previous one.

The alternative — serialising the fields into the single string and re-parsing them in the first
inner node — works, but it puts structured data back into the field the platform treats as free
user text, where the input gate masks and screens it.

### Conditional-route annotation

`DomainWorkflowGraph.route()` is annotated with the graph's own `State`, not with `AgentState`.
LangGraph reads a path callable's annotation as its input schema and projects away every field the
schema does not name, so an `AgentState` annotation would make this template's own fields absent
inside the callable — and a branch depending on one would silently never be taken while the unit
suite stayed green, because a unit test calls the method directly and never goes through the
projection.

## State

Only the fields this template adds are declared in `src/schemas/state.py`. Framework fields are
inherited and never restated: LangGraph builds one channel per key and raises
`ValueError: Channel 'error_log' already exists with a different type` when a subclass re-declares
an accumulating channel as a plain list.

| Field | Type | Purpose |
|---|---|---|
| `validated_context` | `dict` | the caller fields that survived validation |
| `raw_input_text` | `str` | the complaint text, masked |
| `channel` | `str` | `counter` / `phone` / `online` |
| `agency_id`, `case_ref` | `str` | caller-supplied inert identifiers |
| `prior_complaint_count` | `int` | 0–1000, raises the severity score |
| `is_valid_input` | `bool` | passed masking and validation |
| `complaint_category` | `str` | member of `COMPLAINT_CATEGORIES` |
| `classification_confidence` | `str` | `high` / `medium` / `low` |
| `is_complaint` | `bool` | complaint or enquiry |
| `severity_level` | `str` | `critical` / `high` / `medium` / `low` |
| `severity_score` | `int` | 0–100 |
| `severity_modifiers` | `list` | members of `SEVERITY_MODIFIERS` |
| `routing_target_officer_id` | `str` | member of `OFFICER_IDS` |
| `routing_target_team` | `str` | member of `ROUTING_TEAMS` |
| `routing_sla_hours` | `int` | 1–720 |
| `routing_rationale` | `str` | assembled from `RATIONALE_TEMPLATES` |
| `out_of_scope` | `bool` | non-complaint enquiry |
| `rejection_reason` | `str` | member of `REJECTION_REASONS` |

State constraints (mandatory): flat TypedDict of JSON-serialisable primitives; no credentials,
tokens or keys; no Pydantic models, dataclasses or arbitrary objects.

## Configuration

`config/agent.yaml` is the flat discovery manifest, read at root level: `id`, `name`, `namespace`,
`version`, `enabled`, `category`, `generation_mode`, `industry`, `base_type`, a single dotted
`class`, `required_trust_level`, and `requires.secrets` / `requires.extras` (both empty — this
agent requires no secret and constructs no model client).

`config/config.yaml` holds the runtime parameters and is loaded separately. Every value has a
reader, and every reader bounds-checks it; a value outside its range is dropped in favour of the
documented default rather than clamped.

| Key | Range | Read by |
|---|---|---|
| `max_retry` | 0–9 | `AgentBaseGraph.route()` |
| `timeout_s` | 1–600 | the HTTP adapter, as a wall-clock request budget |
| `limits.max_input_chars` | 200–20000 | adapter and `PreProcessNode` |
| `severity.counter_channel_delta` | 0–50 | `SeverityScoreNode` |
| `severity.repeat_complaint_delta` | 0–30 | `SeverityScoreNode` |
| `routing.fallback_sla_hours` | 1–720 | `RouteDecideNode` |

The keyword tables, the routing table and the severity thresholds are **not** configuration. They
were previously advertised as overridable through an `execute(state, config)` argument the
framework never supplies — `BaseNode.__call__` calls `self.execute(state)` — so no override could
apply. Moving them into a YAML file would also mean a category, team or officer name arriving from
outside the closed vocabulary the output gate enforces, and the gate would then have to either
reject a valid operator configuration or stop enforcing the vocabulary. They live in code.

Configuration reaches the nodes through constructors: the agent resolves `config/config.yaml` once
and hands each node its slice at construction. That is the only path that carries a value.

## Security model

### Input

Every caller-controlled value is bounded at `PreProcessNode`, which owns the contract:

* the text is length-bounded and screened for model-directed instructions and credential shapes;
* `input_context` is a **closed set** of fields — anything else is dropped, not ignored. Ignoring
  leaves the field in `input_context`, and `InitializeNode` returns `input_context` verbatim in its
  result where the framework's output gate scans every value and raises; one undeclared field then
  ends the request at the first node with a traceback the caller cannot act on;
* `prior_complaint_count` goes through a finite-and-bounded parser. `NaN` and `Infinity` parse
  cleanly through `float()` and arrive intact through raw JSON, and every comparison against `NaN`
  is False — accepting one would fail open on a severity decision;
* `agency_id` and `case_ref` are locked to `[a-z0-9_]{1,32}` because they are rendered into the
  decision. No caller-supplied free text is rendered anywhere, so no newline can manufacture a line
  in the record;
* a rejected value is never echoed. The refusal names the field and a closed-set reason.

The instruction screen is the template's own, not a claim about the framework's. It screens the raw
text and the markup-stripped text, because stripping markup can convert a detectable token attack
into undetectable plain text and can also re-assemble a directive split by tags. It covers
chat-template control tokens as a class — `<|…|>`, `[INST]`, `<<SYS>>` — and the Japanese directive
forms; measured against the framework's own detector, `<<SYS>>`, a tag-spliced `ignore previous
instructions`, and every Japanese form score no high-confidence framework finding at all.

Both directions are calibrated. `担当者が私の指示を無視した` — "the agent ignored my instruction" —
is an ordinary conduct complaint and exactly what this template exists to route, so the Japanese
patterns require the directive to address the system rather than matching the phrase.

### Output

The invariant: **every string in the routing decision is a member of a closed vocabulary
(`src/schemas/vocabulary.py`) or a caller-supplied inert `[a-z0-9_]{1,32}` identifier; every number
is an integer inside its declared range; no other key exists.**

The monetary precision grid other templates in this line enforce is **not applicable** here: this
agent renders no monetary aggregate. Severity scores and service levels are small bounded integers
with fixed meanings, and rewriting them would destroy the decision rather than protect it. The
invariant above is the one this template states, and it is what `PostProcessNode` enforces.

On violation the gate returns ERROR, clears every output-bearing field, and replaces them with a
truthy refusal notice. Clearing is the point: the invocation envelope is
`{"output": state["formatted_output"] or state["result"], ...}` and the fallback applies on the
error path too, so a gate that merely raised would still ship the un-gated decision inside the
error envelope. A falsy replacement would re-open the same fallback.

The credential screen is the union of the framework's own `detect_credentials` and the local
patterns. Narrower than the framework means a value the framework catches and the template misses
raises inside the gate, and the node wrapper then discards this node's clearing — a detector gap is
a containment bypass. Deleting the local half is the same bug reversed: the framework's patterns
describe credential *formats* and match nothing of the `password=…` shape.

### Layers and what each one alone proves

Responsibilities are split rather than duplicated, so each is falsifiable on its own:

| Layer | Sole duty |
|---|---|
| `PreProcessNode` | the caller contract |
| `InputValidateNode` | residual personal-data masking and the inner-boundary screen |
| `OutputValidateNode` | completeness of the decision |
| `PostProcessNode` | the closed-vocabulary and credential invariant, and containment |
| `DomainWorkflowGraphNode.on_subgraph_error` | containment on the path that skips `post_process` |

A second copy of the output invariant inside `OutputValidateNode` would make both copies
unfalsifiable — removing either alone would leave every test green. More defence, less assurance.

### Trust levels

The manifest declares `VERIFIED_EXTERNAL`, and `PreProcessNode` declares the same: that is the
external boundary. Every inner domain node declares `ANONYMOUS`, explicitly.

That is where the boundary actually is. `GraphNode.execute()` passes the outer `InvocationContext`
into the subgraph unchanged, so there is no escalation across it; trust orders
`ANONYMOUS < VERIFIED_EXTERNAL < INTERNAL`, and an inner node demanding a *higher* level than the
entry contract cannot admit anyone the entry did not — it can only deny people the entry did. The
inner nodes previously declared `INTERNAL`, which denied every real external caller at the first
domain node; the subgraph then errored, the backbone routed past `post_process`, and the caller
received an empty envelope.

`PostProcessNode` declares `ANONYMOUS` for the same reason: a higher level on the last node before
the answer denies the callers the entry contract accepts, and reads as a silent failure.

### Audit

Every boundary node emits a domain event through `emit_trace_event`: the accepted or rejected
caller contract, the classification outcome, the severity outcome, the routing decision, the
assembly, and the output gate's verdict. `BaseNode.__call__` already emits the
`node_start` / `node_complete` / `node_error` lifecycle, which the template does not duplicate.

No audit payload carries the complaint text. The routing record is the artefact; the text is
neither needed to reconstruct the decision nor appropriate to duplicate into a second store with a
different retention policy.

## Framework utilisation

- [x] `InvocationContext` — correlation, session, caller trust level
- [x] `SecurityViolationError` / framework error types
- [x] S-2 `_extra_security_gate_input()` — used on the `main` slot to short-circuit the subgraph
      when the caller contract has already failed (the backbone wires `pre_process → main`
      unconditionally, so a refused request would otherwise still be classified and routed)
- [x] S-3 `_extra_security_gate_output()` — the framework's `@final` gate is never overridden
- [x] S-4 `emit_trace_event()` — at least one domain event inside every `execute()`

> S-2/S-3 gate behaviour by node type: a `FunctionNode` subclass gets the framework's `@final`
> gates automatically and extends them through the `_extra_*` hooks only; `GraphNode` and
> `RemoteAgentNode` are deliberate no-ops at that boundary; a custom `BaseNode` subclass must
> implement both directly.

## Import isolation

- [x] No platform SDK import — `framework.*` and `shared.*` only
- [x] No import from `mediator/`, `api/`, or another agent

## Design decisions

| Decision | Options | Chosen | Rationale |
|---|---|---|---|
| L1 base type | `AgentBaseGraph` / `AutonomousBaseGraph` | `AgentBaseGraph` | fixed pipeline, no autonomous loop, no cost ceiling |
| Composition | flat / nested `GraphNode` | nested | Cat 2: the domain workflow is its own graph with its own topology |
| Error propagation | `propagate` / `handle` | `handle` | `propagate` turns an inner validation failure into a raised error the wrapper reduces to a traceback with no caller-actionable reason, and without clearing the output-bearing fields on the one path that never reaches `post_process` |
| Caller fields across the boundary | JSON in `user_input` / `ContextVar` bridge | bridge | keeps structured data out of the field the platform treats as free user text |
| Keyword and routing tables | code / `config.yaml` | code | the `execute(state, config)` override path does not exist, and a value from outside the closed vocabulary would disarm the output gate |
| Output invariant | monetary precision grid / closed vocabulary | closed vocabulary | no monetary aggregate is rendered; the grid is not applicable |
