"""Derive separated execution and diagnosis status for a run.

A run's old single ``status`` field conflated two unrelated questions:
  - Did the wrapped process complete, or crash?  (execution)
  - Did the agent behave correctly, or fail?     (diagnosis)

This let a looping run — which completed without raising — show a green
"success" even though the agent never accomplished its task. This module
splits the two so a run list can show "completed but failure_detected"
instead of lying with "success".

Both the local CLI (lazy, at render time) and a hosted backend
(compute-once at ingest, store as indexed columns) call ``run_status``.
The offline classifier is sub-millisecond, so lazy use is effectively free;
a backend persists the result so it can be filtered, sorted, and aggregated.
"""

from __future__ import annotations

from typing import Any

from .diagnose import diagnose_run

# A run is a failure when a real detector fires. Every real detector returns
# confidence >= 0.70; the low-confidence fallback returns 0.45. So 0.6 cleanly
# separates "a detector found something" from "no concrete evidence".
_FAILURE_CONFIDENCE = 0.6

EXECUTION_COMPLETED = "completed"
EXECUTION_ERROR = "error"

DIAGNOSIS_HEALTHY = "healthy"
DIAGNOSIS_FAILURE = "failure_detected"
DIAGNOSIS_UNCERTAIN = "uncertain"
DIAGNOSIS_UNKNOWN = "unknown"


def _execution_status(run: dict[str, Any]) -> str:
    """Map the legacy ``status`` field onto completed | error.

    Back-compat: old runs only have ``status`` (success | error | running).
    success/running -> completed (the process did not crash); error/failure -> error.
    """
    raw = str(run.get("status") or "").lower()
    return EXECUTION_ERROR if raw in ("error", "failure") else EXECUTION_COMPLETED


def _has_error_span(run: dict[str, Any]) -> bool:
    spans = run.get("spans")
    if not isinstance(spans, list):
        return False
    return any(isinstance(s, dict) and s.get("type") == "error" for s in spans)


def run_status(run: dict[str, Any]) -> dict[str, Any]:
    """Return separated status for a run.

    Returns a dict with:
      execution_status: completed | error
      diagnosis_status: healthy | failure_detected | uncertain | unknown
      root_cause:       category string when failure_detected, else None
      confidence:       diagnosis confidence (0.0 when not computable)
    """
    execution_status = _execution_status(run)

    spans = run.get("spans")
    if not isinstance(spans, list) or not spans:
        # Nothing to diagnose — don't pretend it's healthy.
        return {
            "execution_status": execution_status,
            "diagnosis_status": DIAGNOSIS_UNKNOWN,
            "root_cause": None,
            "confidence": 0.0,
        }

    try:
        diagnosis = diagnose_run(run, use_llm=False)
        confidence = float(diagnosis.get("confidence") or 0.0)
        category = diagnosis.get("root_cause_category")
    except Exception:
        return {
            "execution_status": execution_status,
            "diagnosis_status": DIAGNOSIS_UNKNOWN,
            "root_cause": None,
            "confidence": 0.0,
        }

    has_failure_evidence = execution_status == EXECUTION_ERROR or _has_error_span(run)

    if confidence >= _FAILURE_CONFIDENCE:
        diagnosis_status = DIAGNOSIS_FAILURE
        root_cause = category
    elif has_failure_evidence:
        # Something errored, but the heuristics can't confidently categorize it.
        diagnosis_status = DIAGNOSIS_UNCERTAIN
        root_cause = None
    else:
        # Clean completion, no error spans, only the low-confidence default fired.
        diagnosis_status = DIAGNOSIS_HEALTHY
        root_cause = None

    return {
        "execution_status": execution_status,
        "diagnosis_status": diagnosis_status,
        "root_cause": root_cause,
        "confidence": confidence,
    }
