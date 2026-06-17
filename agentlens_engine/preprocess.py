"""Preprocess captured runs into compact diagnosis input."""

from __future__ import annotations

import json
from typing import Any

from .classifier import _tool_call_signature

MAX_STEPS = 200
MAX_TEXT_CHARS = 4000


def preprocess_run(spans: list[dict[str, Any]], run_json: dict[str, Any] | None = None) -> dict[str, Any]:
    run_json = run_json or {}
    spans = _dedupe_spans(_safe_spans(spans))[-MAX_STEPS:]
    steps = [_span_to_step(index, span) for index, span in enumerate(spans, start=1)]
    failure_step = _find_failure_step(steps)
    start_index = max(failure_step - 4, 0) if failure_step else max(len(steps) - 3, 0)

    return {
        "run_id": run_json.get("run_id"),
        "name": run_json.get("name"),
        "status": run_json.get("status"),
        "started_at": run_json.get("started_at"),
        "ended_at": run_json.get("ended_at"),
        "system_prompt": _truncate(_find_system_prompt(spans)),
        "tool_definitions": _find_tool_definitions(spans),
        "final_output": _truncate(_find_final_output(spans)),
        "error": _truncate(_find_error(steps)),
        "failure_step_hint": failure_step,
        "trace_warnings": _trace_warnings(run_json, spans),
        "earlier_steps_summary": [_summarize_step(step) for step in steps[:start_index]],
        "last_3_steps_full_detail": steps[start_index:],
        "diagnostic_steps": steps,
    }


def _span_to_step(index: int, span: dict[str, Any]) -> dict[str, Any]:
    step = {
        "step": index,
        "type": span.get("type"),
        "provider": span.get("provider"),
        "model": span.get("model"),
        "latency_ms": span.get("latency_ms"),
        "tool_name": span.get("tool_name"),
        "tool_use_id": span.get("tool_use_id"),
        "input": span.get("input"),
        "output": span.get("output"),
        "error": span.get("error"),
        "context": span.get("context"),
        "usage": span.get("usage"),
        "expected_tool": span.get("expected_tool"),
        "notes": span.get("notes"),
    }

    if span.get("type") == "llm_call":
        step["input_messages"] = span.get("input_messages")
        step["tools"] = span.get("tools")
        step["response_content"] = _truncate(span.get("response_content"))
        step["response_text"] = _truncate(_response_text(span.get("response_content")))
        step["stop_reason"] = span.get("stop_reason")

    return {key: value for key, value in step.items() if value is not None}


def _find_failure_step(steps: list[dict[str, Any]]) -> int:
    for step in steps:
        if step.get("type") == "error":
            return int(step["step"])

    seen: dict[str, int] = {}
    for step in steps:
        if step.get("type") != "tool_call":
            continue
        if step.get("output") is None:
            continue
        signature = _tool_call_signature(step)
        if signature in seen:
            return int(step["step"])
        seen[signature] = int(step["step"])

    return 0


def _find_system_prompt(spans: list[dict[str, Any]]) -> str | None:
    for span in spans:
        for message in span.get("input_messages", []) or []:
            if not isinstance(message, dict):
                continue
            if message.get("role") == "system":
                return str(message.get("content", ""))
    return None


def _find_tool_definitions(spans: list[dict[str, Any]]) -> list[dict[str, Any]]:
    for span in spans:
        tools = span.get("tools")
        if tools:
            return tools
    return []


def _find_final_output(spans: list[dict[str, Any]]) -> Any:
    for span in reversed(spans):
        if span.get("type") == "llm_call":
            return span.get("response_content")
    return None


def _find_error(steps: list[dict[str, Any]]) -> str | None:
    for step in steps:
        if step.get("type") == "error":
            return str(step.get("error"))
    return None


def _summarize_step(step: dict[str, Any]) -> dict[str, Any]:
    summary = {
        "step": step.get("step"),
        "type": step.get("type"),
        "provider": step.get("provider"),
        "model": step.get("model"),
        "tool_name": step.get("tool_name"),
        "error": step.get("error"),
        "response_text": step.get("response_text"),
    }
    return {key: value for key, value in summary.items() if value is not None}


def _response_text(response_content: Any) -> str:
    parts: list[str] = []
    if isinstance(response_content, str):
        return response_content
    if isinstance(response_content, dict):
        response_content = [response_content]
    if isinstance(response_content, list):
        for block in response_content:
            if isinstance(block, dict):
                text = block.get("text") or block.get("content")
                if text:
                    parts.append(str(text))
                message = block.get("message")
                if isinstance(message, dict) and message.get("content"):
                    parts.append(str(message["content"]))
                # OpenAI shape: {choices: [{message: {content: ...}}]} — dig in so
                # assistant text is readable for streaming and non-streaming alike.
                for choice in block.get("choices") or []:
                    if isinstance(choice, dict):
                        choice_message = choice.get("message")
                        if isinstance(choice_message, dict) and choice_message.get("content"):
                            parts.append(str(choice_message["content"]))
    return " ".join(parts)


def _safe_spans(spans: Any) -> list[dict[str, Any]]:
    if not isinstance(spans, list):
        return []
    return [span for span in spans if isinstance(span, dict)]


def _dedupe_spans(spans: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    deduped = []
    for span in spans:
        key = str(span.get("id")) if span.get("id") else json.dumps(span, sort_keys=True, default=str)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(span)
    return deduped


def _trace_warnings(run_json: dict[str, Any], spans: list[dict[str, Any]]) -> list[str]:
    warnings = []
    if "spans" not in run_json:
        warnings.append("missing spans")
    if not spans:
        warnings.append("no usable spans")
    if run_json.get("status") == "running":
        warnings.append("partial run")
    if len(spans) >= MAX_STEPS:
        warnings.append("huge trace truncated")
    if _find_system_prompt(spans) is None:
        warnings.append("missing system prompt")
    return warnings


def _truncate(value: Any) -> Any:
    if isinstance(value, str) and len(value) > MAX_TEXT_CHARS:
        return value[:MAX_TEXT_CHARS] + "...[truncated]"
    if isinstance(value, list):
        return [_truncate(item) for item in value]
    if isinstance(value, dict):
        return {key: _truncate(item) for key, item in value.items()}
    return value
