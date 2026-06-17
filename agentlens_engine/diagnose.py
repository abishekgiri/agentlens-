"""Diagnosis pipeline for captured AgentLens runs."""

from __future__ import annotations

import json
import os
import re
from typing import Any

from .classifier import (
    SYSTEM_PROMPT,
    build_user_prompt,
    classify_from_evidence,
    parse_diagnosis,
    validate_diagnosis,
)
from .fixes import generate_fix, likely_fixes
from .hallucination import detect_hallucinations
from .impact import compute_impact
from .preprocess import preprocess_run


def diagnose_run(run_json: dict[str, Any], use_llm: bool = True) -> dict[str, Any]:
    spans = run_json.get("spans", [])
    if not isinstance(spans, list):
        spans = []

    compact = preprocess_run(spans, run_json=run_json)
    diagnosis = _diagnose_with_llm(compact) if use_llm else None
    diagnosis_source = "llm" if diagnosis is not None else "heuristic"
    if diagnosis is None:
        diagnosis = classify_from_evidence(compact)

    diagnosis["diagnosis_source"] = diagnosis_source
    diagnosis["fix"] = generate_fix(diagnosis["root_cause_category"], compact, diagnosis)

    # Hallucination detection — always runs offline, results appended to diagnosis
    tool_defs = compact.get("tool_definitions") or []
    hallucinations = detect_hallucinations(spans, tool_defs)
    diagnosis["hallucinations"] = hallucinations

    # Cost impact — how much the run spent, and how much was wasted from the
    # failure point onward. Gives every diagnosis a severity / "so what".
    diagnosis["impact"] = compute_impact(spans, diagnosis)

    if diagnosis["confidence"] < 0.6:
        causes = [diagnosis["root_cause_category"], *diagnosis.get("secondary_issues", [])]
        diagnosis["low_confidence_message"] = (
            "We found possible causes but cannot confidently isolate root cause."
        )
        diagnosis["likely_causes"] = causes[:2]
        diagnosis["likely_fixes"] = likely_fixes(causes, compact)

    errors = validate_diagnosis(diagnosis)
    if errors:
        raise ValueError(f"Invalid diagnosis: {'; '.join(errors)}")
    return diagnosis


def _diagnose_with_llm(compact_run: dict[str, Any]) -> dict[str, Any] | None:
    if os.getenv("OPENAI_API_KEY"):
        return _diagnose_with_openai(compact_run)
    if os.getenv("ANTHROPIC_API_KEY"):
        return _diagnose_with_anthropic(compact_run)
    return None


def _diagnose_with_openai(compact_run: dict[str, Any]) -> dict[str, Any] | None:
    try:
        from openai import OpenAI
        client = OpenAI()
        user_prompt = build_user_prompt(compact_run)
        for _ in range(2):
            response = client.chat.completions.create(
                model=os.getenv("AGENTLENS_OPENAI_MODEL", "gpt-4.1-mini"),
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                response_format={"type": "json_object"},
            )
            raw = response.choices[0].message.content
            try:
                diagnosis = parse_diagnosis(raw)
            except (TypeError, json.JSONDecodeError):
                user_prompt += "\nYour previous output was invalid JSON. Return only valid JSON."
                continue
            if not validate_diagnosis(diagnosis):
                return diagnosis
            user_prompt += "\nYour previous output was invalid. Return only the required JSON fields."
        return None
    except Exception:
        return None


def _diagnose_with_anthropic(compact_run: dict[str, Any]) -> dict[str, Any] | None:
    try:
        import anthropic
        client = anthropic.Anthropic()
        user_prompt = build_user_prompt(compact_run)
        for _ in range(2):
            response = client.messages.create(
                model=os.getenv("AGENTLENS_ANTHROPIC_MODEL", "claude-3-5-sonnet-latest"),
                max_tokens=1200,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_prompt}],
            )
            raw = _anthropic_text(response)
            try:
                diagnosis = parse_diagnosis(raw)
            except (TypeError, json.JSONDecodeError):
                user_prompt += "\nYour previous output was invalid JSON. Return only valid JSON."
                continue
            if not validate_diagnosis(diagnosis):
                return diagnosis
            user_prompt += "\nYour previous output was invalid. Return only the required JSON fields."
        return None
    except Exception:
        return None


def _anthropic_text(response: Any) -> str:
    pieces: list[str] = []
    for block in getattr(response, "content", []) or []:
        text = getattr(block, "text", None)
        if text is None and isinstance(block, dict):
            text = block.get("text")
        if text:
            pieces.append(str(text))
    raw = "".join(pieces).strip()
    if raw.startswith("```"):
        raw = re.sub(r'^```[a-z]*\n?', '', raw.strip()).rstrip('`').strip()
    return raw


def diagnosis_to_json(diagnosis: dict[str, Any]) -> str:
    return json.dumps(diagnosis, indent=2)
