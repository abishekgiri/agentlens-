"""Simple local diagnosis evaluation for fixture and real-world cases."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from .diagnose import diagnose_run

_PACKAGE_ROOT = Path(__file__).resolve().parents[1]


def _resolve_dir(relative: str) -> Path:
    """Return CWD-relative path if it exists, otherwise fall back to package root."""
    cwd_path = Path.cwd() / relative
    if cwd_path.exists():
        return cwd_path
    return _PACKAGE_ROOT / relative


FIXTURE_EXPECTED = {
    "phase2_tool_selection": ("tool_selection", 2),
    "phase2_context_pollution": ("context_pollution", 1),
    "phase2_loop": ("loop", 2),
    "phase2_state_drift": ("state_drift", 1),
    "phase2_cascade": ("cascade", 2),
    "phase2_overflow": ("overflow", 4),
}


def evaluate_cases(
    fixture_dir: Path | None = None,
    real_world_dir: Path | None = None,
) -> dict[str, Any]:
    if fixture_dir is None:
        fixture_dir = _resolve_dir("tests/phase2_runs")
    if real_world_dir is None:
        real_world_dir = _resolve_dir("real_world_cases")
    results: list[dict[str, Any]] = []
    results.extend(_evaluate_directory(fixture_dir, FIXTURE_EXPECTED, source="fixture"))
    results.extend(_evaluate_directory(real_world_dir, {}, source="real_world"))
    return _summarize(results)


def print_evaluation(report: dict[str, Any]) -> None:
    print("AgentLens Diagnosis Evaluation")
    print()
    print(f"Fixture cases: {report['fixture_cases']}")
    print(f"Fixture accuracy: {report['fixture_accuracy']:.0%}")
    print(f"Real-world cases: {report['real_world_cases']}")
    print(f"Real-world scored cases: {report['real_world_scored_cases']}")
    print(f"Real-world accuracy: {report['real_world_accuracy']:.0%}")
    print(f"Low-confidence rate: {report['low_confidence_rate']:.0%}")
    print(f"Average runtime: {report['average_latency_ms']:.2f} ms")
    print(f"Most failed category: {report['most_failed_category']}")
    print()
    print("Fixture accuracy by category:")
    for category, accuracy in sorted(report["accuracy_by_category"].items()):
        print(f"- {category}: {accuracy:.0%}")
    if report["real_world_accuracy_by_category"]:
        print()
        print("Real-world accuracy by category:")
        for category, accuracy in sorted(report["real_world_accuracy_by_category"].items()):
            print(f"- {category}: {accuracy:.0%}")
    if report["unscored_real_world_cases"]:
        print()
        print("Unscored real-world cases:")
        for case in report["unscored_real_world_cases"]:
            print(f"- {case}")


def _evaluate_directory(
    directory: Path, expected: dict[str, tuple[str, int]], source: str
) -> list[dict[str, Any]]:
    if not directory.exists():
        return []

    results = []
    for path in sorted(directory.rglob("*.json")):
        if path.name in {"actual_diagnosis.json", "expected_diagnosis.json", "diagnosis.json"}:
            continue
        run = _load_json(path)
        if run is None:
            continue
        run_id = str(run.get("run_id") or path.stem)
        expected_category, expected_step = _expected_for(
            run, expected.get(run_id), path.parent / "expected_diagnosis.json"
        )

        started = time.perf_counter()
        diagnosis = diagnose_run(run, use_llm=False)
        latency_ms = (time.perf_counter() - started) * 1000

        scored = expected_category is not None and expected_step is not None
        correct = (
            scored
            and diagnosis["root_cause_category"] == expected_category
            and diagnosis["failed_at_step"] == expected_step
        )
        results.append(
            {
                "source": source,
                "path": str(path),
                "run_id": run_id,
                "expected_category": expected_category,
                "expected_step": expected_step,
                "actual_category": diagnosis["root_cause_category"],
                "actual_step": diagnosis["failed_at_step"],
                "confidence": diagnosis["confidence"],
                "latency_ms": latency_ms,
                "scored": scored,
                "correct": bool(correct),
            }
        )
    return results


def _summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    scored = [result for result in results if result["scored"]]
    fixture = [result for result in results if result["source"] == "fixture"]
    real_world = [result for result in results if result["source"] == "real_world"]
    fixture_scored = [result for result in fixture if result["scored"]]
    real_world_scored = [result for result in real_world if result["scored"]]
    low_confidence = [result for result in results if result["confidence"] < 0.6]
    latency_values = [result["latency_ms"] for result in results]
    by_category: dict[str, list[bool]] = {}
    real_by_category: dict[str, list[bool]] = {}
    failed_categories: dict[str, int] = {}
    for result in fixture_scored:
        by_category.setdefault(result["expected_category"], []).append(result["correct"])
    for result in real_world_scored:
        real_by_category.setdefault(result["expected_category"], []).append(result["correct"])
    for result in scored:
        if not result["correct"]:
            category = result["expected_category"] or result["actual_category"]
            failed_categories[category] = failed_categories.get(category, 0) + 1

    return {
        "total_cases": len(results),
        "scored_cases": len(scored),
        "overall_accuracy": _rate([result["correct"] for result in scored]),
        "fixture_cases": len(fixture),
        "fixture_accuracy": _rate([result["correct"] for result in fixture_scored]),
        "real_world_cases": len(real_world),
        "real_world_scored_cases": len(real_world_scored),
        "real_world_accuracy": _rate([result["correct"] for result in real_world_scored]),
        "low_confidence_rate": len(low_confidence) / len(results) if results else 0.0,
        "average_latency_ms": sum(latency_values) / len(latency_values) if latency_values else 0.0,
        "most_failed_category": _most_failed_category(failed_categories),
        "accuracy_by_category": {
            category: _rate(values) for category, values in by_category.items()
        },
        "real_world_accuracy_by_category": {
            category: _rate(values) for category, values in real_by_category.items()
        },
        "unscored_real_world_cases": [
            result["path"]
            for result in results
            if result["source"] == "real_world" and not result["scored"]
        ],
    }


def _expected_for(
    run: dict[str, Any],
    fallback: tuple[str, int] | None,
    expected_path: Path | None = None,
) -> tuple[str | None, int | None]:
    expected = run.get("expected_diagnosis") or run.get("expected")
    if isinstance(expected, dict):
        return expected.get("root_cause_category"), expected.get("failed_at_step")
    if expected_path and expected_path.exists():
        expected = _load_json(expected_path)
        if expected:
            return expected.get("root_cause_category"), expected.get("failed_at_step")
    if fallback:
        return fallback
    return None, None


def _load_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def _rate(values: list[bool]) -> float:
    return sum(1 for value in values if value) / len(values) if values else 0.0


def _most_failed_category(failed_categories: dict[str, int]) -> str:
    if not failed_categories:
        return "none"
    return max(failed_categories.items(), key=lambda item: item[1])[0]
