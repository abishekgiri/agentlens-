# AgentLens

**[agentlens.run](https://agentlens.run) · [GitHub](https://github.com/agentlens-hq/agentlens)**



**When your AI agent breaks, AgentLens tells you exactly which decision caused it — and what to change.**

Not logs. Not traces. The answer.

```
ROOT CAUSE:   tool_selection
FAILED AT:    Step 2 (search_web)
WHY:          Both tools had identical descriptions — the agent treated them as
              interchangeable and picked the wrong one.
FIX:          Rewrite tool descriptions so search_web is clearly for external
              lookup and query_db is clearly for local records.
CONFIDENCE:   0.90
IMPACT:       wasted ~430 tokens / $0.0021 on the failed path
```

It also tells you **what the failure cost** — tokens and dollars burned on the
broken path — so you can rank failures by impact, not just count them.

Native support: **Anthropic · OpenAI · LangGraph · raw API**
Also captures **CrewAI · AutoGen · PydanticAI** agents through their underlying Anthropic/OpenAI calls (framework-level spans on the roadmap).

> Validated live on OpenAI and Anthropic across all six failure categories — sync, async, and streaming — with **zero false positives** on healthy runs.

---

## Install

```bash
pip install runlens
agentlens demo        # see a full diagnosis in 10 seconds — no API key needed
```

Or from source:

```bash
git clone https://github.com/agentlens-hq/agentlens.git
cd agentlens
pip install -e ".[anthropic,openai]"
```

Optional: `pip install 'runlens[pii]'` adds spaCy-based name redaction on top of
the built-in PII scrubber (for sharing or hosted sync).

---

## Quickstart — 2 lines

Add AgentLens before your existing provider client. No other changes.

```python
import agentlens
agentlens.init()                        # patches Anthropic + OpenAI automatically

import anthropic
client = anthropic.Anthropic()          # captured from here on

@agentlens.run(name="my_agent")         # groups everything into one run
def run_agent(query):
    response = client.messages.create(
        model="claude-3-5-sonnet-latest",
        max_tokens=512,
        messages=[{"role": "user", "content": query}]
    )
    return response
```

Async agents work exactly the same — `agentlens.init()` patches `AsyncAnthropic` and `AsyncOpenAI` too.

Run your agent, then:

```bash
agentlens runs list
agentlens diagnose <run_id>
```

---

## Framework examples

**LangGraph**

```python
import agentlens
agentlens.init()
agentlens.patch_langgraph()             # call before graph.compile()

from langgraph.graph import StateGraph

graph = StateGraph(MyState)
graph.add_node("planner", planner_fn)
graph.add_node("executor", executor_fn)
app = graph.compile()                   # automatically wrapped — all nodes traced

@agentlens.run(name="langgraph_agent")
async def run(input):
    return await app.ainvoke({"messages": input})
```

**OpenAI async**

```python
import agentlens
agentlens.init()

from openai import AsyncOpenAI
client = AsyncOpenAI()                  # captured automatically

@agentlens.run(name="openai_agent")
async def run_agent(query):
    response = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": query}],
        tools=[],
    )
    return response
```

**Multi-agent tracing**

```python
# Parent agent
ctx = agentlens.get_trace_context()

# Child agent (different process / service)
agentlens.init(parent_context=ctx)      # stitches child trace into parent
```

---

## What AgentLens catches

Six failure categories, detected automatically:

| Category | What it means |
|---|---|
| `tool_selection` | Agent picked the wrong tool — usually because descriptions were too similar |
| `loop` | Agent repeated the same tool call with the same inputs without exit |
| `cascade` | A tool returned bad/stale data and a downstream step used it and failed |
| `context_pollution` | Contradictory instructions in the prompt diluted the agent's goal |
| `state_drift` | Agent abandoned its original goal mid-run |
| `overflow` | Critical context was pushed out of the context window before the key decision |

Plus **hallucination detection** — invented tool parameters, missing required fields, LLM output that contradicts what the tool actually returned.

---

## CLI reference

```bash
# Try it
agentlens demo                          # captures + diagnoses a broken agent, opens the timeline
agentlens watch                         # live mode — spans stream as your agents run

# Runs
agentlens runs list                     # recent runs with execution + diagnosis status
agentlens upload prepare                # write anonymized-only payloads for sharing/sync
agentlens runs show <run_id>            # full span detail for one run
agentlens runs view <run_id>            # open visual timeline in browser
agentlens runs prompt <run_id>          # print exact LLM prompts sent at each step
agentlens runs prompt <run_id> --step 2 # prompt for a specific LLM call only
agentlens runs replay <run_id>          # interactive step-by-step playback (ENTER to advance)
agentlens runs stitch <run_id>          # show multi-agent trace tree rooted at this run

# Diagnosis
agentlens diagnose <run_id>             # root cause analysis + hallucination report
agentlens similar <run_id>              # find historically similar failures
agentlens similar <run_id> --top 10     # top-N matches
agentlens clusters                      # failure clusters across all runs + top fix

# Stats & cost
agentlens stats                         # token usage, latency, cost across all runs
agentlens stats <run_id>                # per-run breakdown

# Utilities
agentlens anonymize <run_id>            # redact secrets before sharing
agentlens feedback-template <run_id>    # structured feedback form
agentlens evaluate                      # accuracy check against fixtures + real cases
agentlens doctor                        # system health check
```

---

## Real example output

```
AgentLens Diagnosis
===================

ROOT CAUSE:
  cascade

FAILED AT:
  Step 3 (get_user_profile)

WHY:
  Step 3 produced bad or corrupted output that caused a failure at step 6.
  get_user_profile returned {"email": null, "warning": "stale cache entry"}
  and send_email downstream tried to use the null email field.

FIX:
  Validate the output from 'get_user_profile' before using it downstream;
  if it is stale, empty, or malformed, stop and recover instead of feeding
  it into the next step.

SECONDARY:
  None

CONFIDENCE: 0.90

IMPACT:
  Wasted ~1,240 tokens / $0.003100 across 3 redundant tool call(s) from the
  failure point onward.
  Run total: 1,630 tokens / $0.004050

HALLUCINATIONS DETECTED:
  [HIGH] step 5 — invented param: 'send_email' was called with 'priority'
  which is not in its schema. Valid params: ['to', 'body'].
```

---

## How it works

`agentlens.init()` monkeypatches your provider clients at import time — no changes to existing code. Every LLM call, tool call, error, and memory snapshot is captured as a span and saved locally to `.agentlens/runs/<run_id>.json`.

`agentlens diagnose` runs the trace through a preprocessing pipeline, then either an LLM-powered classifier (if `ANTHROPIC_API_KEY` or `OPENAI_API_KEY` is set) or a fast local heuristic fallback. The local fallback works offline with no API key required.

All data stays on your machine. No cloud. No signup. No account.

---

## Add a real-world test case

```
real_world_cases/my-broken-agent/
├── trace.json              # anonymized run from .agentlens/runs/
├── expected_diagnosis.json # {"root_cause_category": "loop", "failed_at_step": 4}
└── notes.md                # what the agent was doing and what actually broke
```

Run `agentlens evaluate` to score the diagnosis engine against your case.

---

## Local-first by default

No signup. No account. No cloud. Every run stays on your machine in
`.agentlens/runs/`, and diagnosis works fully offline with no API key.

This is a local developer tool first: when your agent breaks, run one command and
get the answer in under 30 seconds.

**Optional self-host dashboard.** If you want a shared view across runs, the repo
ships a small FastAPI backend (`server/`) that ingests **anonymized-only**
payloads from `agentlens upload prepare`, diagnoses them server-side, and serves a
dashboard with failure rate, root-cause breakdown, and wasted-spend per run:

```bash
agentlens upload prepare                          # anonymized payloads -> .agentlens/upload/
uvicorn server.app:app                            # open http://localhost:8000/dashboard
```

Raw runs never leave your machine — only redacted payloads are uploaded, and the
backend refuses any payload that still contains PII.

---

## Feedback

If AgentLens finds (or misses) a real bug in your agent, we want to know.

```bash
agentlens anonymize <run_id>            # redact secrets
agentlens feedback-template <run_id>    # fill this in and send it
```
