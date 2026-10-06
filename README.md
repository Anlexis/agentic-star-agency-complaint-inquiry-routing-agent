# Agency Complaint & Inquiry Routing Agent

AI agent for classifying and routing insurance agency complaints and inquiries, built with Agentic Star.

> **Category**: Cat 2 (a domain pipeline: classify an incoming message, score it, and route it)
> **Industry**: Insurance
> **Template ID**: INS-C2-053

## Overview

Insurance agencies are required to record what happened to every complaint they receive: which
category it fell into, who it went to, how quickly, and on what grounds. In practice the message
arrives as free text at a counter, on the phone or through a web form, and a person decides where
it goes. That decision is where the record either exists or does not.

This agent makes that decision reproducible. It takes the text of a complaint or enquiry together
with a small set of structured fields — the channel it arrived on, the agency and case reference,
and how many complaints are already on file for the policyholder — and returns a routing decision:
a category, a severity level and score, the team and compliance officer it goes to, the service
level in hours, and a machine-readable rationale for why. Non-complaint enquiries are not rejected;
they are routed to the general enquiry desk, because an unrouted message is the outcome the record
is meant to prevent.

Everything it does is deterministic. Classification is keyword matching, severity is an additive
rule table, and routing is a lookup — no model is invoked, so the same input always yields the same
decision and the decision can be explained line by line. Personal data in the incoming text is
masked before anything else reads it, and the decision itself is built entirely from a closed
vocabulary: no text a caller sends can appear in the record.

The keyword tables, the scoring rules and the routing table are the parts you are meant to replace.
They encode one agency's categories and escalation policy; yours will differ.

## Requirements

**This template does not run standalone.** It requires:

| Requirement | Notes |
|---|---|
| **AGENTIC STAR platform** | The agent is loaded and invoked by the platform. Deployment guides and API documentation: [AGENTIC STAR Developers](https://developers.fd.agenticstar.tm.softbank.jp/) |
| **AgentCore Framework** (`agenticstar-agentcore`) | Installed from PyPI as a dependency. |
| Python | >=3.11 |

```bash
pip install -e .
```

### Behaviour without the platform

The framework is designed to run on AGENTIC STAR and there is no degraded mode. Without it, the
framework packages the agent imports (`framework.*`, `shared.*`) are simply not installed, so
start-up fails at import rather than starting in a partially working state. This is intentional —
a half-running agent is worse than one that refuses to start.

## Quick Start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
python -m pytest tests/ -v
```

Tests run without a platform connection. Running the agent itself does not.

## Calling it

`POST /invoke` takes the complaint text and the declared context fields. Any field not in the list
below is dropped rather than forwarded.

```json
{
  "input": "保険金の支払拒否について納得できない。至急対応してほしい。苦情です。",
  "session_id": "case-intake-001",
  "input_context": {
    "channel": "counter",
    "agency_id": "agency_017",
    "case_ref": "case_9001",
    "prior_complaint_count": 2
  }
}
```

| Context field | Accepted values |
|---|---|
| `channel` | `counter`, `phone`, `online` |
| `agency_id` | 1–32 characters of `[a-z0-9_]`, optional |
| `case_ref` | 1–32 characters of `[a-z0-9_]`, optional |
| `prior_complaint_count` | integer 0–1000, optional |

The response carries the routing decision under `output`. When a request is refused — an
unrecognised channel, a value that is not a finite number, text that carries an instruction aimed
at a language model — `output` is `{"routing_decision_withheld": "<reason>"}` and `status` is
`error`. The reason comes from a fixed vocabulary, so it is safe to display and to log.

## Project Structure

```
src/api/            HTTP adapter — authentication, the caller contract, request budget
src/graph/          outer backbone graph, inner domain graph, and the bridge between them
src/nodes/          the pipeline: validate, classify, score, route, assemble, gate
src/schemas/        the state TypedDict and the closed vocabularies
src/services/       shared validators (bounded numbers, instruction and credential screens)
tests/              unit, integration and boundary tests
config/             agent.yaml (discovery manifest) and config.yaml (runtime parameters)
docs/               design and test specification
```

The two documents that ship with this template are `docs/02_design.md` (architecture, state
contract, security model) and `docs/03_test_spec.md` (what the tests cover and why).

## Customising

1. Replace the keyword tables in `src/nodes/complaint_classify_node.py` with your own categories,
   and add them to `COMPLAINT_CATEGORIES` in `src/schemas/vocabulary.py` — the output gate rejects
   any value that is not in the vocabulary, so the two have to move together.
2. Replace the routing table in `src/nodes/route_decide_node.py` with your own teams, officers and
   service levels, and add those to the vocabulary in the same way.
3. Adjust the scoring rules in `src/nodes/severity_score_node.py` and the tunable deltas in
   `config/config.yaml`.
4. Re-run the test suite.

## License

MIT — see [LICENSE](LICENSE).

## Status of this repository

This template is published **as is**, by its individual author, under the MIT license. It carries
**no warranty and no support commitment**, and no organisation stands behind its behaviour or
fitness for any purpose. Issues and pull requests may or may not receive a response; that is at
the sole discretion of the repository owner.
