"""Failure classification prompt and evidence-based fallback classifier."""

from __future__ import annotations

import json
from typing import Any


FAILURE_CATEGORIES = {
    "tool_selection": "Wrong tool chosen due to ambiguous descriptions",
    "context_pollution": "Contradictory instructions diluted goal",
    "loop": "Agent repeated steps without exit",
    "state_drift": "Agent lost original goal mid-run",
    "cascade": "Bad tool output corrupted later steps",
    "overflow": "Important context pushed out of window",
}

SYSTEM_PROMPT = """You are an expert AI agent debugger.
You receive a structured agent run trace as JSON.

Your job:
1. Identify EXACT failure step
2. Identify WHY it happened
3. Classify into ONE category
4. Suggest ONE concrete fix

Output ONLY valid JSON.

If confidence < 0.6, say so.

Never guess. Only use evidence from trace."""

REQUIRED_FIELDS = {
    "root_cause_category",
    "confidence",
    "failed_at_step",
    "failed_at_tool",
    "explanation",
    "fix",
    "secondary_issues",
}


def build_user_prompt(compact_run: dict[str, Any]) -> str:
    payload = {
        "failure_categories": FAILURE_CATEGORIES,
        "trace": compact_run,
        "required_output": {
            "root_cause_category": "one of failure_categories",
            "confidence": "number between 0.0 and 1.0",
            "failed_at_step": "integer",
            "failed_at_tool": "string or null",
            "explanation": "concrete evidence from the trace",
            "fix": "one actionable fix under 5 minutes",
            "secondary_issues": [],
        },
    }
    return json.dumps(payload, indent=2)


def parse_diagnosis(raw_output: str) -> dict[str, Any]:
    return json.loads(raw_output)


def validate_diagnosis(diagnosis: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    missing = REQUIRED_FIELDS - set(diagnosis)
    if missing:
        errors.append(f"missing fields: {', '.join(sorted(missing))}")
    if diagnosis.get("root_cause_category") not in FAILURE_CATEGORIES:
        errors.append("root_cause_category is unknown")
    confidence = diagnosis.get("confidence")
    if not isinstance(confidence, (int, float)) or not 0 <= confidence <= 1:
        errors.append("confidence must be a number between 0.0 and 1.0")
    if not isinstance(diagnosis.get("failed_at_step"), int):
        errors.append("failed_at_step must be an integer")
    if not isinstance(diagnosis.get("secondary_issues"), list):
        errors.append("secondary_issues must be a list")
    return errors


def classify_from_evidence(compact_run: dict[str, Any]) -> dict[str, Any]:
    warnings = compact_run.get("trace_warnings", [])
    if "no usable spans" in warnings or "missing spans" in warnings:
        return _low_confidence(compact_run, ["tool_selection", "state_drift"])

    detectors = [
        _detect_loop,
        _detect_overflow,
        _detect_context_pollution,
        _detect_state_drift,
        _detect_cascade,
        _detect_tool_selection,
    ]
    candidates = [candidate for detector in detectors if (candidate := detector(compact_run))]
    if not candidates:
        return _low_confidence(compact_run, ["tool_selection", "state_drift"])

    diagnosis = max(candidates, key=lambda item: item["confidence"])
    secondary = [
        item["root_cause_category"]
        for item in sorted(candidates, key=lambda item: item["confidence"], reverse=True)
        if item["root_cause_category"] != diagnosis["root_cause_category"]
    ]
    diagnosis["secondary_issues"] = secondary[:2]
    return diagnosis


def _detect_tool_selection(compact_run: dict[str, Any]) -> dict[str, Any] | None:
    tools = compact_run.get("tool_definitions", [])
    ambiguous = _has_ambiguous_tools(tools)

    for step in compact_run.get("diagnostic_steps", []):
        if step.get("type") != "tool_call":
            continue

        tool = step.get("tool_name")
        expected = step.get("expected_tool")
        output = step.get("output")
        output_text = _text(output) + " " + _text(compact_run.get("error"))

        # Explicit expected_tool mismatch — highest confidence
        if expected and expected != tool:
            return _diagnosis(
                "tool_selection",
                0.94,
                step["step"],
                tool,
                f"Step {step['step']} chose '{tool}' but the trace marks '{expected}' as the expected tool.",
            )

        # Ambiguous tools + error message explicitly says the right tool is elsewhere
        if ambiguous and _contains(output_text, _WRONG_TOOL_SIGNALS):
            return _diagnosis(
                "tool_selection",
                0.90,
                step["step"],
                tool,
                f"Step {step['step']} chose '{tool}' after ambiguous tool descriptions, "
                f"and the error confirms the data was only available in a different tool.",
            )

        # Structural fallback: tools are similar AND this tool call errored — even
        # without explicit "use other tool" language in the error message.
        # Lower confidence since we have no keyword confirmation.
        tool_errored = (
            isinstance(output, dict) and (
                output.get("status") == "error" or bool(output.get("error"))
            )
        )
        if ambiguous and tool_errored:
            return _diagnosis(
                "tool_selection",
                0.82,
                step["step"],
                tool,
                f"Step {step['step']} chose '{tool}' which then errored. "
                f"Multiple tools have similar descriptions — the agent likely selected "
                f"the wrong one because the descriptions were not distinct enough.",
            )

        # Behavioral fallback: real traces often have distinct tool descriptions,
        # so textual ambiguity is not always the failure signal. If a local/private
        # records task is routed to a web/search tool while a db/query tool exists
        # and remains unused, classify conservatively as wrong-tool selection.
        behavioral_explanation = _behavioral_wrong_tool_signal(compact_run, step, tools)
        if behavioral_explanation:
            return _diagnosis(
                "tool_selection",
                0.70,
                step["step"],
                tool,
                behavioral_explanation,
            )

    return None


# Phrases in tool output/error text that signal the wrong tool was called.
# Broad enough to catch varied phrasings across frameworks.
_WRONG_TOOL_SIGNALS = [
    # classic
    "wrong tool", "not the right tool", "incorrect tool",
    # "only available in X" patterns
    "only available", "only available in", "not available on the web",
    "only in", "only found in", "records only in",
    # "X only handles Y" patterns
    "only handles", "handles only", "handles tickets only",
    "only for", "is not for",
    # "use X instead" patterns
    "should use", "use instead", "try instead",
    # "available in X" without "only"
    "available in", "stored in",
]


def _detect_context_pollution(compact_run: dict[str, Any]) -> dict[str, Any] | None:
    prompt_text = _all_input_text(compact_run)
    if _contains(prompt_text, ["contradictory", "conflicting", "ignore the user", "also do the opposite"]):
        step = _first_llm_step(compact_run)
        return _diagnosis(
            "context_pollution",
            0.88,
            step.get("step", compact_run.get("failure_step_hint", 0)),
            step.get("tool_name"),
            "The input messages contain contradictory instructions, so the agent diluted the actual user goal before acting.",
        )
    return None


def _tool_call_signature(step: dict[str, Any]) -> str:
    """Stable string key for a tool call, used to detect repeated tool+input pairs."""
    return json.dumps(
        {"tool": step.get("tool_name"), "input": step.get("input")},
        sort_keys=True,
        default=str,
    )


def _detect_loop(compact_run: dict[str, Any]) -> dict[str, Any] | None:
    seen: dict[str, int] = {}
    for step in compact_run.get("diagnostic_steps", []):
        if step.get("type") != "tool_call":
            continue
        if step.get("output") is None:
            continue
        signature = _tool_call_signature(step)
        if signature in seen:
            # Report the FIRST occurrence — where the loop began — not the repeat
            # that confirms it. On a long trace the origin is the actionable step
            # to intervene at; the repeat just proves the agent never broke out.
            first_step = seen[signature]
            return _diagnosis(
                "loop",
                0.95,
                first_step,
                step.get("tool_name"),
                f"Step {first_step} begins a loop: the same tool and input recur at "
                f"step {step['step']} with no exit.",
            )
        seen[signature] = step["step"]
    return None


def _detect_state_drift(compact_run: dict[str, Any]) -> dict[str, Any] | None:
    for step in compact_run.get("diagnostic_steps", []):
        step_text = _text(step)
        if _contains(step_text, [
            "weather", "restaurant", "unrelated", "lost original goal", "switched topics",
            "off-topic", "different task", "forgot the original", "original objective",
            "changed the subject", "irrelevant",
        ]):
            return _diagnosis(
                "state_drift",
                0.89,
                step["step"],
                step.get("tool_name"),
                f"Step {step['step']} diverges from the original goal and switches to unrelated content.",
            )
    return None


def _detect_cascade(compact_run: dict[str, Any]) -> dict[str, Any] | None:
    steps = compact_run.get("diagnostic_steps", [])

    # Pass 1: find the FIRST tool_call that produced suspicious/corrupted output
    # (status can still be ok — the data is silently bad)
    bad_output_step: dict[str, Any] | None = None
    bad_output_index: int = -1
    BAD_OUTPUT_KEYWORDS = ["stale", "malformed", "corrupted", "invalid id", "NaN", "warning"]

    for i, step in enumerate(steps):
        if step.get("type") != "tool_call":
            continue
        output = step.get("output")
        output_text = _text(output)
        # Flag as bad if: keywords present AND step status is NOT itself an error
        # (an erroring step is a symptom, not the source)
        is_source_of_bad_data = (
            _contains(output_text, BAD_OUTPUT_KEYWORDS)
            and not (isinstance(output, dict) and output.get("status") == "error")
        )
        if is_source_of_bad_data:
            bad_output_step = step
            bad_output_index = i
            break  # stop at FIRST bad source, don't overwrite with downstream symptoms

    if bad_output_step is None:
        return None

    # Pass 2: check whether a LATER step failed because of that bad data
    for i, step in enumerate(steps):
        if i <= bad_output_index:
            continue

        later_text = _text(step) + " " + _text(step.get("input")) + " " + _text(step.get("output"))

        # Explicit cascade keywords in the downstream step
        explicit_cascade = _contains(later_text, ["used stale", "corrupted later", "invalid id", "downstream"])

        # Structural signal: downstream step errored (output status=error or error span follows)
        output = step.get("output")
        later_errored = (
            step.get("type") == "error"
            or (isinstance(output, dict) and output.get("status") == "error")
        )

        if explicit_cascade or later_errored:
            return _diagnosis(
                "cascade",
                0.90,
                bad_output_step["step"],
                bad_output_step.get("tool_name"),
                f"Step {bad_output_step['step']} produced bad or corrupted output "
                f"that caused a failure at step {step['step']}.",
            )

    return None


def _detect_overflow(compact_run: dict[str, Any]) -> dict[str, Any] | None:
    joined = _text(compact_run)
    # Textual path: the trace explicitly admits context was dropped.
    if _contains(joined, _OVERFLOW_SIGNALS):
        failure_step = compact_run.get("failure_step_hint") or _last_step(compact_run).get("step", 0)
        return _diagnosis(
            "overflow",
            0.87,
            int(failure_step),
            _last_step(compact_run).get("tool_name"),
            "The trace says important earlier context was truncated or pushed out before the failed decision.",
        )

    # Structural path: the agent evicted earlier messages from its own context
    # (capped/sliding-window history) AND a later step then lacked information.
    # Gated on the missing-info signal so a healthy agent that simply caps history
    # but still answers correctly is NOT flagged — this must not regress the
    # zero-false-positive result on healthy runs.
    evicted_step = _evicted_context_step(compact_run)
    if evicted_step is not None and _missing_info_after(compact_run, evicted_step):
        return _diagnosis(
            "overflow",
            0.72,
            evicted_step,
            None,
            f"Earlier conversation context was dropped from the model's input before step "
            f"{evicted_step}, which then lacked information it had been given earlier.",
        )
    return None


_OVERFLOW_SIGNALS = [
    "context window", "pushed out", "truncated", "forgot earlier", "lost from context",
    "out of context", "dropped earlier", "exceeds the context", "too long to fit",
    "earlier messages were removed", "history was trimmed",
]

# Phrases that signal the model lacked information it should have had.
_MISSING_INFO_SIGNALS = [
    "don't have", "do not have", "no information", "not provided", "wasn't provided",
    "was not provided", "cannot find", "can't find", "i don't know", "do not know",
    "no record of", "not sure what", "could you remind", "you haven't told me",
    "earlier", "previously mentioned",
]


def _message_fingerprints(input_messages: Any) -> set[str]:
    """Stable per-message keys (role + content prefix), ignoring system messages."""
    prints: set[str] = set()
    for message in input_messages or []:
        if not isinstance(message, dict):
            continue
        role = message.get("role")
        if role == "system":
            continue
        content = message.get("content")
        prints.add(f"{role}:{_text(content)[:80]}")
    return prints


def _evicted_context_step(compact_run: dict[str, Any]) -> int | None:
    """Return the step where an earlier message disappeared from the model's input.

    Real agents cap message history; when an early message present in one llm_call
    is absent from a later llm_call, the agent evicted it. Returns the later step.
    """
    seen: set[str] = set()
    for step in compact_run.get("diagnostic_steps", []):
        if step.get("type") != "llm_call":
            continue
        current = _message_fingerprints(step.get("input_messages"))
        if not current:
            continue
        if seen and (seen - current):
            return int(step.get("step", 0))
        seen |= current
    return None


def _missing_info_after(compact_run: dict[str, Any], step_number: int) -> bool:
    for step in compact_run.get("diagnostic_steps", []):
        if int(step.get("step", 0)) < step_number:
            continue
        if step.get("type") == "error":
            return True
        if _contains(_text(step.get("response_text")), _MISSING_INFO_SIGNALS):
            return True
    return False


def _diagnosis(
    category: str,
    confidence: float,
    step: int,
    tool: str | None,
    explanation: str,
) -> dict[str, Any]:
    return {
        "root_cause_category": category,
        "confidence": confidence,
        "failed_at_step": int(step),
        "failed_at_tool": tool,
        "explanation": explanation,
        "fix": "",
        "secondary_issues": [],
    }


def _low_confidence(compact_run: dict[str, Any], likely: list[str]) -> dict[str, Any]:
    warning_text = "; ".join(compact_run.get("trace_warnings", []))
    explanation = "We detected an issue but cannot confidently determine root cause from the trace evidence."
    if warning_text:
        explanation += f" Trace warnings: {warning_text}."

    # Use trace signals to suggest more relevant categories instead of always
    # defaulting to tool_selection + state_drift regardless of trace content
    likely = _infer_likely_categories(compact_run) or likely

    return {
        "root_cause_category": likely[0],
        "confidence": 0.45,
        "failed_at_step": int(compact_run.get("failure_step_hint") or 0),
        "failed_at_tool": None,
        "explanation": explanation,
        "fix": "Collect more detailed tool inputs, tool outputs, and the final model response before diagnosing.",
        "secondary_issues": likely[1:],
    }


def _infer_likely_categories(compact_run: dict[str, Any]) -> list[str]:
    """Rank categories by trace evidence when no high-confidence detector fired."""
    scores: dict[str, int] = {}
    steps = compact_run.get("diagnostic_steps", [])
    all_text = _text(compact_run).lower()

    # Error span present → upstream tool likely produced bad data
    if any(s.get("type") == "error" for s in steps):
        scores["cascade"] = scores.get("cascade", 0) + 2

    # Multiple tool_calls with same tool name → loop
    tool_names = [s.get("tool_name") for s in steps if s.get("type") == "tool_call" and s.get("tool_name")]
    if len(tool_names) != len(set(tool_names)):
        scores["loop"] = scores.get("loop", 0) + 3

    # Ambiguous tool descriptions present → tool_selection
    if _has_ambiguous_tools(compact_run.get("tool_definitions", [])):
        scores["tool_selection"] = scores.get("tool_selection", 0) + 2

    # Contradiction / conflict keywords → context_pollution
    if _contains(all_text, ["contradictory", "conflicting", "ignore", "also do"]):
        scores["context_pollution"] = scores.get("context_pollution", 0) + 2

    # Context window / truncation keywords → overflow
    if _contains(all_text, ["context window", "truncated", "pushed out", "forgot earlier"]):
        scores["overflow"] = scores.get("overflow", 0) + 2

    if not scores:
        return ["tool_selection", "state_drift"]

    ranked = sorted(scores, key=lambda k: scores[k], reverse=True)
    if len(ranked) < 2:
        fallback = [c for c in ["tool_selection", "state_drift", "loop"] if c not in ranked]
        ranked += fallback
    return ranked[:2]


def _has_ambiguous_tools(tools: list[dict[str, Any]]) -> bool:
    """Return True if any two tools have descriptions that are identical or highly similar.

    Uses Jaccard word-overlap similarity with a 0.6 threshold so that
    "look up customer records" and "look up customer information" are
    both caught — not just exact duplicates.
    """
    descriptions = [
        _tool_description(t).strip().lower()
        for t in tools
        if _tool_description(t).strip()
    ]
    if len(descriptions) < 2:
        return False

    for i in range(len(descriptions)):
        for j in range(i + 1, len(descriptions)):
            if _description_similarity(descriptions[i], descriptions[j]) >= 0.6:
                return True
    return False


def _behavioral_wrong_tool_signal(
    compact_run: dict[str, Any],
    step: dict[str, Any],
    tools: list[dict[str, Any]],
) -> str | None:
    tool = str(step.get("tool_name") or "")
    tool_names = [_tool_name(t) for t in tools]
    tool_names = [name for name in tool_names if name]
    if len(tool_names) < 2 or tool not in tool_names:
        return None

    used_tools = {
        str(s.get("tool_name"))
        for s in compact_run.get("diagnostic_steps", [])
        if s.get("type") == "tool_call" and s.get("tool_name")
    }
    unused_alternatives = [
        name for name in tool_names if name != tool and name not in used_tools
    ]
    if not unused_alternatives:
        return None

    # Mid-flow guard: only fire if this is the final tool call in the run. A correct
    # agent that calls a web/search tool first and intends to call the db tool next
    # would otherwise be mislabeled as wrong-tool before it gets there. Requiring this
    # to be the last tool call means the alternative was genuinely never reached.
    tool_steps = [
        s for s in compact_run.get("diagnostic_steps", []) if s.get("type") == "tool_call"
    ]
    if tool_steps and step.get("step") != tool_steps[-1].get("step"):
        return None

    prompt_text = _all_input_text(compact_run)
    output_text = _text(step.get("output")).lower()

    local_records_intent = _contains(
        prompt_text,
        ["local records", "customer record", "customer records", "private records", "database"],
    )
    web_tool_used = _contains(tool, ["web", "search"])
    db_alternative_unused = any(
        _contains(name, ["db", "database", "query"]) for name in unused_alternatives
    )
    public_only_output = _contains(
        output_text,
        ["public", "not private", "not local", "not private renewal records"],
    )

    if local_records_intent and web_tool_used and db_alternative_unused:
        suffix = " and the output itself indicates public/non-private data" if public_only_output else ""
        return (
            f"Step {step['step']} routed a local/private records task to '{tool}' "
            f"while an unused database/query tool was available{suffix}."
        )

    return None


def _tool_name(tool: dict[str, Any]) -> str:
    if "name" in tool:
        return str(tool["name"])
    function = tool.get("function")
    if isinstance(function, dict):
        return str(function.get("name", ""))
    return ""


def _description_similarity(a: str, b: str) -> float:
    """Jaccard similarity between the word sets of two tool descriptions."""
    words_a = set(a.split())
    words_b = set(b.split())
    if not words_a or not words_b:
        return 0.0
    return len(words_a & words_b) / len(words_a | words_b)


def _tool_description(tool: dict[str, Any]) -> str:
    if "description" in tool:
        return str(tool["description"])
    function = tool.get("function")
    if isinstance(function, dict):
        return str(function.get("description", ""))
    return ""


def _all_input_text(compact_run: dict[str, Any]) -> str:
    pieces = [str(compact_run.get("system_prompt") or "")]
    for step in compact_run.get("diagnostic_steps", []):
        pieces.append(_text(step.get("input_messages")))
    return " ".join(pieces).lower()


def _first_llm_step(compact_run: dict[str, Any]) -> dict[str, Any]:
    for step in compact_run.get("diagnostic_steps", []):
        if step.get("type") == "llm_call":
            return step
    return {}


def _last_step(compact_run: dict[str, Any]) -> dict[str, Any]:
    steps = compact_run.get("diagnostic_steps", [])
    return steps[-1] if steps else {}


def _contains(text: str, needles: list[str]) -> bool:
    lowered = text.lower()
    return any(needle in lowered for needle in needles)


def _text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, sort_keys=True, default=str)
