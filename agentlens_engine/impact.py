"""Quantify the cost impact of a diagnosed failure.

A diagnosis says what broke and where. Impact says why it matters: how many
tokens and dollars the run spent, and how much of that was wasted on the failed
or redundant work from the failure point onward. This turns a diagnosis into a
business case and gives a severity axis to rank runs by.

Cost and token usage come from the captured llm_call spans (validated accurate
against provider pricing), so this is computed offline with no API calls.
"""

from __future__ import annotations

from typing import Any


def _span_tokens(span: dict[str, Any]) -> int:
    usage = span.get("usage")
    if not isinstance(usage, dict):
        return 0
    if usage.get("total_tokens"):
        return int(usage["total_tokens"])
    # OpenAI uses prompt/completion; Anthropic uses input/output.
    prompt = usage.get("prompt_tokens") or usage.get("input_tokens") or 0
    completion = usage.get("completion_tokens") or usage.get("output_tokens") or 0
    return int(prompt) + int(completion)


def compute_impact(spans: list[dict[str, Any]], diagnosis: dict[str, Any]) -> dict[str, Any]:
    """Return token/cost totals and the portion wasted from the failure point on.

    "Wasted" = work spent at or after ``failed_at_step`` — the redundant loop
    iterations, the cascade of corrupted steps, etc. For a healthy run
    (failed_at_step 0) nothing is counted as wasted.
    """
    failed_step = int(diagnosis.get("failed_at_step") or 0)

    total_tokens = 0
    total_cost = 0.0
    wasted_tokens = 0
    wasted_cost = 0.0
    redundant_tool_calls = 0

    for index, span in enumerate(spans, start=1):
        if not isinstance(span, dict):
            continue
        tokens = _span_tokens(span)
        cost = float(span.get("cost_usd") or 0.0)
        total_tokens += tokens
        total_cost += cost
        if failed_step and index >= failed_step:
            wasted_tokens += tokens
            wasted_cost += cost
            if span.get("type") == "tool_call":
                redundant_tool_calls += 1

    # If the failure step bore no token cost — it's a tool_call/error and every
    # llm_call precedes it (e.g. a single wrong-tool choice or a drift) — attribute
    # the driving llm_call: the decision that caused the failure. Without this, a
    # one-call failure shows zero wasted spend even though the wrong decision cost
    # real tokens. Loops keep their later-iteration cost and are unaffected.
    if failed_step and wasted_tokens == 0:
        for index in range(min(failed_step, len(spans)), 0, -1):
            span = spans[index - 1]
            if isinstance(span, dict) and span.get("type") == "llm_call":
                wasted_tokens += _span_tokens(span)
                wasted_cost += float(span.get("cost_usd") or 0.0)
                break

    return {
        "total_tokens": total_tokens,
        "total_cost_usd": round(total_cost, 6),
        "wasted_tokens": wasted_tokens,
        "wasted_cost_usd": round(wasted_cost, 6),
        "redundant_tool_calls": redundant_tool_calls,
    }


def impact_summary(impact: dict[str, Any]) -> str:
    """One-line human summary, e.g. for the CLI diagnosis output."""
    wasted_cost = impact.get("wasted_cost_usd", 0.0)
    wasted_tokens = impact.get("wasted_tokens", 0)
    calls = impact.get("redundant_tool_calls", 0)
    if not wasted_tokens and not wasted_cost and not calls:
        return "No measurable wasted spend (no token usage recorded, or run was healthy)."
    return (
        f"Wasted ~{wasted_tokens} tokens / ${wasted_cost:.6f} "
        f"across {calls} redundant tool call(s) from the failure point onward."
    )
