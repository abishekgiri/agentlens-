"""Public AgentLens SDK and CLI entrypoint."""

from __future__ import annotations

import argparse
from datetime import datetime
import importlib
import json
import re
from pathlib import Path
from typing import Any

from agentlens_engine.clustering import cluster_failures, print_clusters
from agentlens_engine.diagnose import diagnose_run
from agentlens_engine.evaluate import evaluate_cases, print_evaluation
from agentlens_engine.hallucination import hallucination_summary
from agentlens_engine.impact import impact_summary
from agentlens_engine.similarity import find_similar_failures
from agentlens_engine.status import run_status
from agentlens_engine.timeline import generate_html
from agentlens_sdk import (
    AgentLensClient,
    get_trace_context,
    init,
    load_run,
    load_runs,
    patch_langgraph,
    record_memory_snapshot,
    record_tool_result,
    run,
    save_run,
)
from agentlens_sdk.collector import RUNS_DIR, AmbiguousRunIdError

__all__ = [
    "AgentLensClient",
    "get_trace_context",
    "init",
    "patch_langgraph",
    "record_memory_snapshot",
    "record_tool_result",
    "run",
    "save_run",
]

CLI_COMMANDS = {
    "runs",
    "diagnose",
    "anonymize",
    "feedback-template",
    "evaluate",
    "doctor",
    "stats",
    "demo",
    "watch",
}


def main() -> None:
    parser = argparse.ArgumentParser(prog="agentlens")
    subparsers = parser.add_subparsers(dest="command")

    runs_parser = subparsers.add_parser("runs")
    runs_subparsers = runs_parser.add_subparsers(dest="runs_command")
    runs_subparsers.add_parser("list")
    show_parser = runs_subparsers.add_parser("show")
    show_parser.add_argument("run_id")
    view_parser = runs_subparsers.add_parser("view")
    view_parser.add_argument("run_id")
    prompt_parser = runs_subparsers.add_parser("prompt")
    prompt_parser.add_argument("run_id")
    prompt_parser.add_argument("--step", type=int, default=None, help="Show only this LLM call step (1-indexed)")
    replay_parser = runs_subparsers.add_parser("replay")
    replay_parser.add_argument("run_id")
    stitch_parser = runs_subparsers.add_parser("stitch")
    stitch_parser.add_argument("run_id")

    diagnose_parser = subparsers.add_parser("diagnose")
    diagnose_parser.add_argument("run_id")
    similar_parser = subparsers.add_parser("similar")
    similar_parser.add_argument("run_id")
    similar_parser.add_argument("--top", type=int, default=5)
    subparsers.add_parser("clusters")
    anonymize_parser = subparsers.add_parser("anonymize")
    anonymize_parser.add_argument("run_id")
    feedback_parser = subparsers.add_parser("feedback-template")
    feedback_parser.add_argument("run_id")
    stats_parser = subparsers.add_parser("stats")
    stats_parser.add_argument("run_id", nargs="?")
    subparsers.add_parser("evaluate")
    subparsers.add_parser("doctor")
    demo_parser = subparsers.add_parser("demo")
    demo_parser.add_argument("--no-browser", action="store_true", help="Skip opening the timeline in a browser")
    watch_parser = subparsers.add_parser("watch")
    watch_parser.add_argument("--interval", type=float, default=0.5, help="Poll interval in seconds")

    args = parser.parse_args()

    if args.command == "demo":
        _run_demo(open_browser=not args.no_browser)
        return

    if args.command == "watch":
        _watch_runs(interval=args.interval)
        return

    if args.command == "runs" and args.runs_command == "list":
        _print_runs_list()
        return

    if args.command == "runs" and args.runs_command == "show":
        _print_run_detail(args.run_id)
        return

    if args.command == "runs" and args.runs_command == "view":
        _open_timeline(args.run_id)
        return

    if args.command == "runs" and args.runs_command == "prompt":
        _print_prompt_viewer(args.run_id, step=args.step)
        return

    if args.command == "runs" and args.runs_command == "replay":
        _replay_run(args.run_id)
        return

    if args.command == "runs" and args.runs_command == "stitch":
        _print_stitch(args.run_id)
        return

    if args.command == "diagnose":
        _print_diagnosis(args.run_id)
        return

    if args.command == "similar":
        _print_similar(args.run_id, top_n=args.top)
        return

    if args.command == "clusters":
        _print_clusters()
        return

    if args.command == "anonymize":
        _anonymize_run(args.run_id)
        return

    if args.command == "feedback-template":
        _print_feedback_template(args.run_id)
        return

    if args.command == "stats":
        _print_stats(args.run_id)
        return

    if args.command == "evaluate":
        print_evaluation(evaluate_cases())
        return

    if args.command == "doctor":
        _print_doctor()
        return

    parser.print_help()


def _print_runs_list() -> None:
    runs = load_runs()
    if not runs:
        print("No AgentLens runs found in .agentlens/runs/")
        return

    print("AgentLens Runs")
    print()
    print(
        f"{'run_id':36}  {'name':24}  {'execution':10}  "
        f"{'diagnosis':18}  {'timestamp':25}  spans"
    )
    for item in runs[:20]:
        status = run_status(item)
        # When a failure is detected, show the root cause — it's the actionable
        # part. Otherwise show the status word (healthy / uncertain / unknown).
        diagnosis = status["root_cause"] or status["diagnosis_status"]
        print(
            f"{item.get('run_id', ''):36}  "
            f"{item.get('name', '')[:24]:24}  "
            f"{status['execution_status']:10}  "
            f"{diagnosis[:18]:18}  "
            f"{item.get('started_at', '')[:25]:25}  "
            f"{len(item.get('spans', []))}"
        )


def _print_run_detail(run_id: str) -> None:
    item = _load_run_or_report(run_id)
    if item is None:
        return

    print("AgentLens Run Detail")
    print()
    status = run_status(item)
    print(f"Run ID: {item.get('run_id')}")
    print(f"Name: {item.get('name')}")
    print(f"Execution status: {status['execution_status']}")
    diagnosis_line = status["diagnosis_status"]
    if status["root_cause"]:
        diagnosis_line += f" ({status['root_cause']}, confidence {status['confidence']:.2f})"
    print(f"Diagnosis status: {diagnosis_line}")
    print(f"Started: {item.get('started_at')}")
    print(f"Ended: {item.get('ended_at')}")
    spans = item.get("spans", [])
    if not isinstance(spans, list):
        spans = []
    print(f"Spans: {len(spans)}")
    print()

    for index, span in enumerate(spans, start=1):
        if not isinstance(span, dict):
            print(f"[{index}] malformed_span")
            print(_compact(span))
            print()
            continue

        span_type = span.get("type")
        print(f"[{index}] {span_type}")
        if span_type == "llm_call":
            print(f"Provider: {span.get('provider')}")
            print(f"Model: {span.get('model')}")
            print(f"Latency: {span.get('latency_ms')} ms")
            print(f"Stop reason: {span.get('stop_reason')}")
            print(f"Usage: {_compact(span.get('usage'))}")
            print(f"Input messages: {_compact(span.get('input_messages'))}")
            print(f"Response: {_compact(span.get('response_content'))}")
        elif span_type == "tool_call":
            print(f"Tool: {span.get('tool_name')}")
            print(f"Input: {_compact(span.get('input'))}")
            print(f"Output: {_compact(span.get('output'))}")
        elif span_type == "error":
            print(f"Error: {span.get('error')}")
            print(f"Context: {_compact(span.get('context'))}")
        else:
            print(_compact(span))
        print()


def _compact(value: Any) -> str:
    text = json.dumps(value, ensure_ascii=True, default=str)
    return text if len(text) <= 500 else text[:497] + "..."


def _print_diagnosis(run_id: str) -> None:
    item = _load_run_or_report(run_id)
    if item is None:
        return

    try:
        diagnosis = diagnose_run(item)
    except ValueError as exc:
        print(f"Diagnosis failed: {exc}")
        return
    if diagnosis.get("confidence", 0) < 0.6:
        print("AgentLens Diagnosis")
        print("===================")
        print()
        print("SOURCE:")
        print(f"  {_diagnosis_source_label(diagnosis)}")
        print()
        print("LOW CONFIDENCE")
        print()
        print(diagnosis.get("low_confidence_message"))
        print("We are not treating this as a final root cause yet.")
        print()
        print("LIKELY CAUSES:")
        for cause in diagnosis.get("likely_causes", [])[:2]:
            print(f"- {cause}")
        print()
        print("SUGGESTED FIXES:")
        for fix in diagnosis.get("likely_fixes", [])[:2]:
            print(f"- {fix}")
        return

    print("AgentLens Diagnosis")
    print("===================")
    print()
    print("SOURCE:")
    print(f"  {_diagnosis_source_label(diagnosis)}")
    print()
    print("ROOT CAUSE:")
    print(f"  {diagnosis['root_cause_category']}")
    print()
    print("FAILED AT:")
    tool = diagnosis.get("failed_at_tool") or "unknown tool"
    print(f"  Step {diagnosis['failed_at_step']} ({tool})")
    print()
    print("WHY:")
    print(f"  {diagnosis['explanation']}")
    print()
    print("FIX:")
    print(f"  {diagnosis['fix']}")
    print()
    print("SECONDARY:")
    secondary = diagnosis.get("secondary_issues") or []
    if secondary:
        for issue in secondary:
            print(f"- {issue}")
    else:
        print("  None")
    print()
    print(f"CONFIDENCE: {diagnosis['confidence']:.2f}")

    impact = diagnosis.get("impact")
    if impact:
        print()
        print("IMPACT:")
        print(f"  {impact_summary(impact)}")
        print(
            f"  Run total: {impact['total_tokens']} tokens / "
            f"${impact['total_cost_usd']:.6f}"
        )

    hallucinations = diagnosis.get("hallucinations") or []
    if hallucinations:
        print()
        print("HALLUCINATIONS DETECTED:")
        for h in hallucinations:
            sev = h.get("severity", "?").upper()
            print(f"  [{sev}] {h.get('detail', '')}")


def _anonymize_run(run_id: str) -> None:
    item = _load_run_or_report(run_id)
    if item is None:
        return

    anonymized = _anonymize_value(item)
    output_path = f"{item.get('run_id', run_id)}.anonymized.json"
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(anonymized, handle, indent=2)
    print(f"Wrote anonymized trace to {output_path}")
    print("Review it before sharing. AgentLens removes obvious secrets, but you know your data best.")


def _print_feedback_template(run_id: str) -> None:
    print(f"# AgentLens Feedback: {run_id}")
    print()
    print("## What broke?")
    print()
    print("- ")
    print()
    print("## Was diagnosis correct?")
    print()
    print("- Yes / No / Partially:")
    print("- Why:")
    print()
    print("## Was diagnosis useful?")
    print()
    print("- Yes / No / Partially:")
    print("- Did it save debugging time:")
    print()
    print("## Did the suggested fix work?")
    print()
    print("- Yes / No / Not tried:")
    print("- Notes:")
    print()
    print("## What was confusing?")
    print()
    print("- Install:")
    print("- Setup:")
    print("- CLI output:")
    print("- Diagnosis wording:")
    print()
    print("## Would you use this again?")
    print()
    print("- Yes / No / Maybe:")
    print("- Why:")
    print()
    print("## Would you pay for this?")
    print()
    print("- Yes / No / Maybe:")
    print("- What would need to improve:")


def _print_stats(run_id: str | None) -> None:
    if run_id:
        item = _load_run_or_report(run_id)
        if item is None:
            return
        _print_run_stats(item)
        return

    runs = load_runs()
    if not runs:
        print("No AgentLens runs found in .agentlens/runs/")
        return

    summaries = [_summarize_run(item) for item in runs[:20]]
    totals = _merge_stats(summaries)

    print("AgentLens Stats")
    print()
    print(f"Runs analyzed: {len(summaries)}")
    print(f"LLM calls: {totals['llm_calls']}")
    print(f"Tool calls: {totals['tool_calls']}")
    print(f"Errors: {totals['errors']}")
    print(f"Input tokens: {totals['input_tokens']}")
    print(f"Output tokens: {totals['output_tokens']}")
    print(f"Total tokens: {totals['total_tokens']}")
    print(f"Captured latency: {_format_ms(totals['latency_ms'])}")
    print(f"Captured cost: {_format_cost(totals['cost_usd'])}")
    print()
    print(f"{'run_id':36}  {'name':24}  {'status':8}  {'llm':>3}  {'tool':>4}  {'tokens':>8}  latency")
    for summary in summaries:
        print(
            f"{summary['run_id'][:36]:36}  "
            f"{summary['name'][:24]:24}  "
            f"{summary['status'][:8]:8}  "
            f"{summary['llm_calls']:>3}  "
            f"{summary['tool_calls']:>4}  "
            f"{summary['total_tokens']:>8}  "
            f"{_format_ms(summary['latency_ms'])}"
        )


def _print_run_stats(item: dict[str, Any]) -> None:
    summary = _summarize_run(item)

    print("AgentLens Run Stats")
    print()
    print(f"Run ID: {summary['run_id']}")
    print(f"Name: {summary['name']}")
    print(f"Status: {summary['status']}")
    print(f"Started: {item.get('started_at')}")
    print(f"Ended: {item.get('ended_at')}")
    print(f"Duration: {_format_ms(summary['duration_ms'])}")
    print()
    print("Calls:")
    print(f"  LLM calls: {summary['llm_calls']}")
    print(f"  Tool calls: {summary['tool_calls']}")
    print(f"  Errors: {summary['errors']}")
    print()
    print("Tokens:")
    print(f"  Input tokens: {summary['input_tokens']}")
    print(f"  Output tokens: {summary['output_tokens']}")
    print(f"  Total tokens: {summary['total_tokens']}")
    print()
    print("Performance:")
    print(f"  Captured latency: {_format_ms(summary['latency_ms'])}")
    if summary["slowest_step"]:
        print(f"  Slowest step: {summary['slowest_step']}")
    else:
        print("  Slowest step: unavailable")
    print()
    print("Cost:")
    print(f"  Captured cost: {_format_cost(summary['cost_usd'])}")
    if summary["cost_usd"] == 0:
        print("  Note: provider billing cost is not captured unless traces include cost_usd.")
    print()
    print("Providers:")
    _print_count_map(summary["providers"])
    print()
    print("Models:")
    _print_count_map(summary["models"])


def _print_similar(run_id: str, top_n: int = 5) -> None:
    """Find historically similar failures and print them."""
    item = _load_run_or_report(run_id)
    if item is None:
        return

    try:
        diagnosis = diagnose_run(item, use_llm=False)
    except ValueError as exc:
        print(f"Could not diagnose run: {exc}")
        return

    all_runs = load_runs()
    similar = find_similar_failures(diagnosis, item, all_runs, top_n=top_n)

    print(f"Similar Failures — {item.get('name', run_id)}")
    print(f"Root cause: {diagnosis.get('root_cause_category')}  ·  failed tool: {diagnosis.get('failed_at_tool') or 'n/a'}")
    print("=" * 60)
    print()

    if not similar:
        print("No similar historical failures found.")
        print("Run more agents to build up a failure library.")
        return

    for i, match in enumerate(similar, start=1):
        pct = int(match["similarity"] * 100)
        print(f"#{i}  {match['name'] or match['run_id'][:8]}  —  {pct}% similar")
        print(f"     Match: {match['match_reason']}")
        print(f"     Category: {match['category']}  ·  Tool: {match['failed_at_tool'] or 'n/a'}")
        started = (match.get("started_at") or "")[:19].replace("T", " ")
        if started:
            print(f"     When: {started} UTC")
        if match.get("fix"):
            print(f"     Fix used: {match['fix'][:120]}{'…' if len(match['fix']) > 120 else ''}")
        print()


def _print_clusters() -> None:
    """Show failure clusters across all runs."""
    all_runs = load_runs()
    error_runs = [r for r in all_runs if r.get("status") in ("error", "failure")]
    clusters = cluster_failures(all_runs)
    print_clusters(clusters, total_runs=len(all_runs))
    if clusters:
        top = clusters[0]
        pct = int(top["percentage"] * 100)
        print(f"Top fix opportunity: fix '{top['category']}' on '{top['failed_tool'] or 'any tool'}' "
              f"to eliminate {pct}% of failures.")


def _open_timeline(run_id: str) -> None:
    """Generate a self-contained HTML timeline and open it in the default browser."""
    import tempfile
    import webbrowser

    item = _load_run_or_report(run_id)
    if item is None:
        return
    html = generate_html(item, diagnosis=_load_diagnosis_for(item))
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".html", delete=False, encoding="utf-8"
    ) as tmp:
        tmp.write(html)
        tmp_path = tmp.name
    webbrowser.open(f"file://{tmp_path}")
    print(f"Timeline opened in browser: {tmp_path}")


def _load_diagnosis_for(item: dict[str, Any]) -> dict[str, Any] | None:
    """Find a diagnosis for this run: saved file first, else offline diagnosis for failed runs."""
    run_id = str(item.get("run_id") or "")
    diag_path = Path(".agentlens") / "diagnoses" / f"{run_id}.json"
    if diag_path.exists():
        try:
            return json.loads(diag_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass
    if item.get("status") in ("error", "failure"):
        try:
            return diagnose_run(item, use_llm=False)
        except Exception:
            return None
    return None


def _print_prompt_viewer(run_id: str, step: int | None = None) -> None:
    """Print the exact LLM prompt(s) sent during a run in readable form."""
    item = _load_run_or_report(run_id)
    if item is None:
        return

    spans = [s for s in (item.get("spans") or []) if isinstance(s, dict) and s.get("type") == "llm_call"]
    if not spans:
        print("No LLM calls found in this run.")
        return

    if step is not None:
        idx = step - 1
        if idx < 0 or idx >= len(spans):
            print(f"Step {step} out of range (run has {len(spans)} LLM call(s)).")
            return
        spans = [spans[idx]]
        start_idx = idx
    else:
        start_idx = 0

    print(f"AgentLens LLM Prompt Viewer — {item.get('name', run_id)}")
    print("=" * 60)

    for offset, span in enumerate(spans):
        step_num = start_idx + offset + 1
        model = span.get("model") or "unknown"
        provider = span.get("provider") or ""
        print(f"\nStep {step_num}: {provider}/{model}")
        print("─" * 60)

        messages = span.get("input_messages") or []
        if messages:
            for msg in messages:
                if not isinstance(msg, dict):
                    continue
                role = (msg.get("role") or "unknown").upper()
                content = msg.get("content")
                if isinstance(content, str):
                    body = content
                elif isinstance(content, list):
                    parts = []
                    for block in content:
                        if isinstance(block, dict):
                            if block.get("type") == "text":
                                parts.append(block.get("text", ""))
                            elif block.get("type") == "tool_result":
                                parts.append(f"[tool_result: {block.get('tool_use_id')}]")
                            elif block.get("type") == "tool_use":
                                parts.append(f"[tool_use: {block.get('name')}]")
                            else:
                                parts.append(json.dumps(block))
                        else:
                            parts.append(str(block))
                    body = "\n".join(parts)
                else:
                    body = json.dumps(content, default=str)
                print(f"[{role}]")
                print(body)
                print()
        else:
            print("(no messages captured)")

        tools = span.get("tools") or []
        if tools:
            print(f"Tools available ({len(tools)}):")
            for t in tools:
                if isinstance(t, dict):
                    name = t.get("name") or (t.get("function") or {}).get("name") or "?"
                    desc = t.get("description") or (t.get("function") or {}).get("description") or ""
                    print(f"  • {name}" + (f" — {desc}" if desc else ""))
            print()

        resp = span.get("response_content")
        if resp is not None:
            print("Response:")
            if isinstance(resp, list):
                for block in resp:
                    if isinstance(block, dict):
                        btype = block.get("type")
                        if btype == "text":
                            print(f"  [text] {block.get('text', '')}")
                        elif btype == "tool_use":
                            print(f"  [tool_use] {block.get('name')}({json.dumps(block.get('input', {}), default=str)})")
                        else:
                            print(f"  {json.dumps(block, default=str)}")
                    else:
                        print(f"  {block}")
            else:
                print(f"  {_compact(resp)}")
            stop = span.get("stop_reason")
            if stop:
                print(f"  Stop reason: {stop}")

        usage = span.get("usage")
        cost = span.get("cost_usd") or 0
        if isinstance(usage, dict) and (usage.get("input_tokens") or usage.get("prompt_tokens")):
            inp = (usage.get("input_tokens") or 0) + (usage.get("prompt_tokens") or 0)
            out = (usage.get("output_tokens") or 0) + (usage.get("completion_tokens") or 0)
            cost_str = f"  Cost: ${cost:.6f}" if cost > 0 else ""
            print(f"\n  Tokens: {inp} in / {out} out{cost_str}")


def _replay_run(run_id: str) -> None:
    """Interactive step-by-step replay of a run. Press ENTER to advance."""
    item = _load_run_or_report(run_id)
    if item is None:
        return

    spans = [s for s in (item.get("spans") or []) if isinstance(s, dict)]
    if not spans:
        print("No spans in this run.")
        return

    print(f"AgentLens Session Replay — {item.get('name', run_id)}")
    print("=" * 60)
    print(f"  {len(spans)} span(s)  ·  status: {item.get('status', '?')}")
    print()
    print("Press ENTER to advance through each span. Ctrl+C to quit.")

    for i, span in enumerate(spans, start=1):
        try:
            input(f"\n[Press ENTER for step {i}/{len(spans)}]")
        except (KeyboardInterrupt, EOFError):
            print("\nReplay stopped.")
            return

        stype = span.get("type", "unknown")
        print(f"\n{'━' * 60}")
        print(f"Step {i}/{len(spans)}  ·  {stype.upper()}")
        print('━' * 60)

        if stype == "llm_call":
            print(f"Provider : {span.get('provider', '?')}")
            print(f"Model    : {span.get('model', '?')}")
            lat = span.get("latency_ms")
            if lat:
                print(f"Latency  : {_format_ms(lat)}")
            cost = span.get("cost_usd") or 0
            if cost > 0:
                print(f"Cost     : ${cost:.6f}")

            messages = span.get("input_messages") or []
            if messages:
                print(f"\nWhat the agent knows ({len(messages)} message(s)):")
                for msg in messages[-3:]:  # Show last 3 to keep it concise
                    if not isinstance(msg, dict):
                        continue
                    role = (msg.get("role") or "?").upper()
                    content = msg.get("content", "")
                    body = content if isinstance(content, str) else json.dumps(content, default=str)
                    print(f"  [{role}] {body[:200]}{'…' if len(body) > 200 else ''}")

            tools = span.get("tools") or []
            if tools:
                names = [
                    t.get("name") or (t.get("function") or {}).get("name") or "?"
                    for t in tools if isinstance(t, dict)
                ]
                print(f"\nTools available: {', '.join(names)}")

            resp = span.get("response_content")
            if resp:
                stop = span.get("stop_reason", "")
                print(f"\n  ↳ Stop reason: {stop}")
                if isinstance(resp, list):
                    for block in resp:
                        if isinstance(block, dict):
                            if block.get("type") == "tool_use":
                                print(f"  ↳ Called tool: {block.get('name')}({json.dumps(block.get('input', {}), default=str)[:100]})")
                            elif block.get("type") == "text":
                                text = (block.get("text") or "")[:150]
                                print(f"  ↳ Response text: {text}{'…' if len(block.get('text',''))>150 else ''}")

        elif stype == "tool_call":
            print(f"Tool   : {span.get('tool_name', '?')}")
            inp = span.get("input")
            out = span.get("output")
            if inp is not None:
                print(f"Input  : {_compact(inp)}")
            if out is not None:
                is_err = isinstance(out, dict) and (out.get("status") == "error" or out.get("error"))
                prefix = "⚠ Output (ERROR)" if is_err else "Output"
                print(f"{prefix}: {_compact(out)}")

        elif stype == "error":
            print(f"⚠ Error: {span.get('error', 'unknown')}")
            ctx = span.get("context")
            if ctx:
                print(f"  Context: {_compact(ctx)}")

        elif stype == "memory_snapshot":
            label = span.get("label", "")
            print(f"Label : {label}")
            state = span.get("state")
            if state:
                print(f"State : {_compact(state)}")

        else:
            print(_compact(span))

    print(f"\n{'═' * 60}")
    print(f"End of replay  ·  status: {item.get('status', '?')}")
    llm_n = sum(1 for s in spans if s.get("type") == "llm_call")
    tool_n = sum(1 for s in spans if s.get("type") == "tool_call")
    err_n = sum(1 for s in spans if s.get("type") == "error")
    print(f"Summary: {llm_n} LLM call(s), {tool_n} tool call(s), {err_n} error(s)")


def _print_stitch(run_id: str) -> None:
    """Show a multi-agent trace tree rooted at this run."""
    root = _load_run_or_report(run_id)
    if root is None:
        return

    all_runs = load_runs()

    def children_of(rid: str) -> list[dict[str, Any]]:
        return [r for r in all_runs if r.get("parent_run_id") == rid]

    def print_tree(r: dict[str, Any], prefix: str = "", is_last: bool = True) -> None:
        connector = "└── " if is_last else "├── "
        spans = r.get("spans") or []
        status = r.get("status", "?")
        status_icon = {"success": "✓", "error": "✗", "running": "○"}.get(status, "?")
        name = r.get("name", "?")[:30]
        rid_short = (r.get("run_id") or "")[:8]
        n_spans = len([s for s in spans if isinstance(s, dict)])
        print(f"{prefix}{connector}{rid_short}  {name:30}  [{n_spans} spans]  {status_icon} {status}")
        kids = children_of(r.get("run_id", ""))
        child_prefix = prefix + ("    " if is_last else "│   ")
        for j, child in enumerate(kids):
            print_tree(child, child_prefix, is_last=(j == len(kids) - 1))

    print(f"AgentLens Multi-Agent Trace Tree — {root.get('name', run_id)}")
    print("=" * 60)
    print()
    # Print root without connector
    spans = root.get("spans") or []
    n_spans = len([s for s in spans if isinstance(s, dict)])
    status = root.get("status", "?")
    status_icon = {"success": "✓", "error": "✗", "running": "○"}.get(status, "?")
    print(f"{(root.get('run_id') or '')[:8]}  {root.get('name','?')[:30]:30}  [{n_spans} spans]  {status_icon} {status}  (root)")
    kids = children_of(root.get("run_id", ""))
    if not kids:
        print("\n  No child runs found. Child runs must be started with parent_context set.")
        print("  Example: agentlens.init(parent_context=agentlens.get_trace_context())")
        return
    for j, child in enumerate(kids):
        print_tree(child, "", is_last=(j == len(kids) - 1))
    print()


def _run_demo(open_browser: bool = True) -> None:
    """One-command demo: capture a broken agent run, diagnose it, open the timeline.

    Works offline, requires no API key — the agent is simulated through the real
    capture pipeline so the saved run is identical in shape to a live capture.
    """
    from agentlens_sdk.collector import _current_run, _finalize_run, append_span, start_run

    print("AgentLens Demo")
    print("=" * 60)
    print()
    print("Simulating a broken customer-support agent...")
    print("  Two tools with near-identical descriptions: search_web, query_db")
    print("  The task needs local records. Watch what the agent picks.")
    print()

    run_data = start_run(name="demo_customer_support_agent")
    token = _current_run.set(run_data)
    try:
        tools = [
            {
                "name": "search_web",
                "description": "find info about a topic",
                "input_schema": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                },
            },
            {
                "name": "query_db",
                "description": "find info about a topic",
                "input_schema": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                },
            },
        ]
        append_span(
            {
                "type": "llm_call",
                "provider": "anthropic",
                "ts": datetime.now().astimezone().isoformat(),
                "latency_ms": 412.6,
                "model": "claude-3-5-sonnet-latest",
                "input_messages": [
                    {
                        "role": "user",
                        "content": "Find the renewal status for customer:alex using local records.",
                    }
                ],
                "tools": tools,
                "response_content": [
                    {
                        "type": "text",
                        "text": "Both tools say they find info about a topic. I will use search_web.",
                    },
                    {
                        "type": "tool_use",
                        "id": "toolu_demo_001",
                        "name": "search_web",
                        "input": {"query": "customer:alex renewal status"},
                    },
                ],
                "stop_reason": "tool_use",
                "usage": {"input_tokens": 158, "output_tokens": 42},
                "cost_usd": 0.001104,
            }
        )
        record_tool_result(
            tool_name="search_web",
            input={"query": "customer:alex renewal status"},
            output={
                "status": "error",
                "error": "Network access disabled. Customer records are only available in query_db.",
            },
            tool_use_id="toolu_demo_001",
        )
        run_data["status"] = "error"
        run_data["error"] = "agent picked the wrong tool"
    finally:
        _finalize_run(run_data)
        _current_run.reset(token)

    run_id = run_data["run_id"]
    print(f"Run captured: {run_id[:8]}  ({len(run_data['spans'])} spans)")
    print()
    print("Diagnosing...")
    print()
    _print_diagnosis(run_id)
    print()
    print("=" * 60)
    print("Next steps:")
    print(f"  agentlens runs view {run_id[:8]}     # visual timeline")
    print(f"  agentlens runs prompt {run_id[:8]}   # exact prompts sent")
    print()
    print("Use it on your own agent — two lines:")
    print("  import agentlens")
    print("  agentlens.init()   # before creating your Anthropic/OpenAI client")
    if open_browser:
        print()
        print("Opening the timeline viewer...")
        _open_timeline(run_id)


def _watch_runs(interval: float = 0.5) -> None:
    """Live mode: tail .agentlens/runs/ and print spans as agents execute."""
    import time as _time

    colors = {
        "llm_call": "\033[34m",       # blue
        "tool_call": "\033[32m",      # green
        "error": "\033[31m",          # red
        "memory_snapshot": "\033[33m",  # amber
        "langgraph_node": "\033[35m",   # purple
        "langgraph_run": "\033[35m",
    }
    reset, dim = "\033[0m", "\033[2m"

    def describe(span: dict[str, Any]) -> str:
        stype = span.get("type", "?")
        color = colors.get(stype, "")
        if stype == "llm_call":
            u = span.get("usage") or {}
            tok = (u.get("input_tokens", 0) or u.get("prompt_tokens", 0)) + (
                u.get("output_tokens", 0) or u.get("completion_tokens", 0)
            )
            extra = f"  {tok} tok" if tok else ""
            lat = span.get("latency_ms")
            extra += f"  {lat:.0f}ms" if isinstance(lat, (int, float)) and lat else ""
            return f"{color}llm_call{reset}   {span.get('model', '?')}{dim}{extra}{reset}"
        if stype == "tool_call":
            out = span.get("output")
            bad = isinstance(out, dict) and (out.get("status") == "error" or out.get("error"))
            mark = f"  {colors['error']}error result{reset}" if bad else ""
            return f"{color}tool_call{reset}  {span.get('tool_name', '?')}{mark}"
        if stype == "error":
            return f"{color}error{reset}      {str(span.get('error', ''))[:80]}"
        return f"{color}{stype}{reset}  {span.get('tool_name') or span.get('label') or ''}"

    print(f"Watching {RUNS_DIR}/ for agent activity... (Ctrl+C to stop)", flush=True)
    print()

    seen: dict[str, int] = {}  # path -> span count already printed
    announced: set[str] = set()
    try:
        while True:
            if RUNS_DIR.exists():
                for path in sorted(RUNS_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime):
                    try:
                        run_data = json.loads(path.read_text(encoding="utf-8"))
                    except (json.JSONDecodeError, OSError):
                        continue
                    spans = [s for s in (run_data.get("spans") or []) if isinstance(s, dict)]
                    key = str(path)
                    if key not in seen:
                        # Runs that existed before watch started are skipped;
                        # only freshly written runs stream from span 0.
                        seen[key] = 0 if _is_new(path) else len(spans)
                    if seen[key] < len(spans):
                        rid = (run_data.get("run_id") or path.stem)[:8]
                        if key not in announced:
                            status = run_data.get("status", "?")
                            print(f"{dim}── run {rid}  {run_data.get('name', '')}{reset}", flush=True)
                            announced.add(key)
                        for i in range(seen[key], len(spans)):
                            print(f"  {rid}  [{i + 1}] {describe(spans[i])}", flush=True)
                        seen[key] = len(spans)
                        if run_data.get("status") in ("error", "failure"):
                            print(f"  {rid}  {colors['error']}run FAILED{reset} → agentlens diagnose {rid}", flush=True)
            _time.sleep(interval)
    except KeyboardInterrupt:
        print("\nStopped watching.", flush=True)


def _is_new(path: Path) -> bool:
    """True if the file was modified in the last 5 seconds."""
    import time as _time

    try:
        return (_time.time() - path.stat().st_mtime) < 5
    except OSError:
        return False


def _load_run_or_report(run_id: str) -> dict[str, Any] | None:
    try:
        item = load_run(run_id)
    except AmbiguousRunIdError as exc:
        print(f"Multiple runs match '{exc.prefix}'. Please use a longer run_id.")
        for match in exc.matches[:10]:
            print(f"- {match}")
        if len(exc.matches) > 10:
            print(f"...and {len(exc.matches) - 10} more")
        return None

    if item is None:
        print(f"Run not found: {run_id}")
    return item


def _diagnosis_source_label(diagnosis: dict[str, Any]) -> str:
    if diagnosis.get("diagnosis_source") == "llm":
        return "LLM diagnosis"
    return "Heuristic fallback"


def _summarize_run(item: dict[str, Any]) -> dict[str, Any]:
    spans = item.get("spans", [])
    if not isinstance(spans, list):
        spans = []
    safe_spans = [span for span in spans if isinstance(span, dict)]

    providers: dict[str, int] = {}
    models: dict[str, int] = {}
    input_tokens = 0
    output_tokens = 0
    total_tokens = 0
    latency_ms = 0.0
    cost_usd = 0.0
    slowest_latency = -1.0
    slowest_step = ""

    for index, span in enumerate(safe_spans, start=1):
        provider = span.get("provider")
        model = span.get("model")
        if provider:
            providers[str(provider)] = providers.get(str(provider), 0) + 1
        if model:
            models[str(model)] = models.get(str(model), 0) + 1

        usage = span.get("usage") if isinstance(span.get("usage"), dict) else {}
        usage_input = _number(usage.get("input_tokens")) + _number(usage.get("prompt_tokens"))
        usage_output = _number(usage.get("output_tokens")) + _number(usage.get("completion_tokens"))
        usage_total = _number(usage.get("total_tokens"))
        if usage_total == 0 and (usage_input or usage_output):
            usage_total = usage_input + usage_output
        input_tokens += int(usage_input)
        output_tokens += int(usage_output)
        total_tokens += int(usage_total)

        span_latency = _number(span.get("latency_ms"))
        latency_ms += span_latency
        if span_latency > 0 and span_latency > slowest_latency:
            slowest_latency = span_latency
            slowest_step = _describe_span(index, span, span_latency)

        cost_usd += _extract_cost_usd(span)

    return {
        "run_id": str(item.get("run_id") or ""),
        "name": str(item.get("name") or ""),
        "status": str(item.get("status") or ""),
        "llm_calls": sum(1 for span in safe_spans if span.get("type") == "llm_call"),
        "tool_calls": _count_tool_invocations(safe_spans),
        "errors": sum(1 for span in safe_spans if span.get("type") == "error"),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "latency_ms": latency_ms,
        "duration_ms": _duration_ms(item.get("started_at"), item.get("ended_at")),
        "cost_usd": cost_usd,
        "slowest_step": slowest_step if slowest_latency >= 0 else "",
        "providers": providers,
        "models": models,
    }


def _merge_stats(summaries: list[dict[str, Any]]) -> dict[str, Any]:
    totals = {
        "llm_calls": 0,
        "tool_calls": 0,
        "errors": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "latency_ms": 0.0,
        "cost_usd": 0.0,
    }
    for summary in summaries:
        for key in totals:
            totals[key] += summary[key]
    return totals


def _count_tool_invocations(spans: list[dict[str, Any]]) -> int:
    invocations: set[str] = set()
    for index, span in enumerate(spans):
        if span.get("type") != "tool_call":
            continue
        tool_use_id = span.get("tool_use_id")
        if tool_use_id:
            key = f"id:{tool_use_id}"
        else:
            key = json.dumps(
                {"tool_name": span.get("tool_name"), "input": span.get("input")},
                sort_keys=True,
                default=str,
            )
            if key == '{"input": null, "tool_name": null}':
                key = f"span:{index}"
        invocations.add(key)
    return len(invocations)


def _extract_cost_usd(value: Any) -> float:
    if isinstance(value, dict):
        total = 0.0
        for key, item in value.items():
            lowered = str(key).lower()
            if lowered in {"cost_usd", "total_cost_usd"}:
                total += _number(item)
            elif lowered in {"cost", "total_cost"} and "token" not in lowered:
                total += _number(item)
            elif isinstance(item, (dict, list)):
                total += _extract_cost_usd(item)
        return total
    if isinstance(value, list):
        return sum(_extract_cost_usd(item) for item in value)
    return 0.0


def _duration_ms(started_at: Any, ended_at: Any) -> float:
    if not isinstance(started_at, str) or not isinstance(ended_at, str):
        return 0.0
    try:
        started = datetime.fromisoformat(started_at.replace("Z", "+00:00"))
        ended = datetime.fromisoformat(ended_at.replace("Z", "+00:00"))
    except ValueError:
        return 0.0
    return max((ended - started).total_seconds() * 1000, 0.0)


def _describe_span(index: int, span: dict[str, Any], latency_ms: float) -> str:
    label = str(span.get("type") or "unknown")
    if span.get("tool_name"):
        label += f" ({span['tool_name']})"
    elif span.get("model"):
        label += f" ({span['model']})"
    return f"Step {index}: {label} at {_format_ms(latency_ms)}"


def _print_count_map(values: dict[str, int]) -> None:
    if not values:
        print("  None captured")
        return
    for name, count in sorted(values.items()):
        print(f"  {name}: {count}")


def _format_ms(value: float) -> str:
    if value <= 0:
        return "unavailable"
    if value < 1000:
        return f"{value:.2f} ms"
    return f"{value / 1000:.2f} s"


def _format_cost(value: float) -> str:
    if value <= 0:
        return "not captured"
    return f"${value:.6f}"


def _number(value: Any) -> float:
    if isinstance(value, bool):
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return 0.0
    return 0.0


def _print_doctor() -> None:
    checks = [
        _doctor_check("imports", _doctor_imports),
        _doctor_check("local storage", _doctor_local_storage),
        _doctor_check("diagnosis fixture", _doctor_diagnosis_fixture),
        _doctor_check("messy trace handling", _doctor_messy_trace_handling),
        _doctor_check("anonymization", _doctor_anonymization),
        _doctor_check("evaluation", _doctor_evaluation),
    ]

    print("AgentLens Doctor")
    print()
    for check in checks:
        line = f"{check['status']:<5} {check['name']}"
        if check["message"]:
            line += f" - {check['message']}"
        print(line)
    print()

    if any(check["status"] == "FAIL" for check in checks):
        print("Result: needs attention")
    elif any(check["status"] == "WARN" for check in checks):
        print("Result: healthy with warnings")
    else:
        print("Result: healthy")


def _doctor_check(name: str, check: Any) -> dict[str, str]:
    try:
        status, message = check()
    except Exception as exc:  # Doctor should report broken states, not crash.
        status, message = "FAIL", str(exc)
    return {"name": name, "status": status, "message": message}


def _doctor_imports() -> tuple[str, str]:
    modules = [
        "agentlens_sdk",
        "agentlens_sdk.collector",
        "agentlens_engine.classifier",
        "agentlens_engine.preprocess",
        "agentlens_engine.diagnose",
        "agentlens_engine.evaluate",
        "agentlens_engine.fixes",
    ]
    for module in modules:
        importlib.import_module(module)

    # Verify the public SDK surface is intact (not a self-referential import)
    import agentlens as _al
    if not callable(getattr(_al, "init", None)):
        return "FAIL", "agentlens.init is missing or not callable"

    if not CLI_COMMANDS.issuperset({"doctor", "diagnose", "evaluate"}):
        return "FAIL", "required CLI commands are missing"
    return "PASS", ""


def _doctor_local_storage() -> tuple[str, str]:
    run_id = "agentlens_doctor_storage_check"
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    path = RUNS_DIR / f"{run_id}.json"
    payload = {
        "run_id": run_id,
        "name": "doctor",
        "started_at": "2026-05-18T00:00:00+00:00",
        "ended_at": "2026-05-18T00:00:01+00:00",
        "status": "success",
        "spans": [],
    }
    try:
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        loaded = load_run(run_id)
        if not loaded or loaded.get("run_id") != run_id:
            return "FAIL", "run JSON could not be read back"
    finally:
        try:
            path.unlink()
        except FileNotFoundError:
            pass
    return "PASS", ""


def _doctor_diagnosis_fixture() -> tuple[str, str]:
    diagnosis = diagnose_run(_doctor_tool_selection_run(), use_llm=False)
    required = {
        "root_cause_category",
        "failed_at_step",
        "confidence",
        "explanation",
        "fix",
    }
    missing = required - set(diagnosis)
    if missing:
        return "FAIL", f"diagnosis missing {', '.join(sorted(missing))}"
    if not diagnosis["root_cause_category"]:
        return "FAIL", "diagnosis did not return a category"
    if not isinstance(diagnosis["failed_at_step"], int):
        return "FAIL", "diagnosis did not return a failed step"
    if not isinstance(diagnosis["confidence"], (int, float)):
        return "FAIL", "diagnosis did not return confidence"
    if not diagnosis["explanation"] or not diagnosis["fix"]:
        return "FAIL", "diagnosis output was not readable"
    return "PASS", ""


def _doctor_messy_trace_handling() -> tuple[str, str]:
    messy_run = {
        "run_id": "agentlens_doctor_messy",
        "name": "doctor_messy",
        "status": "running",
        "spans": [
            "malformed span",
            {
                "id": "partial",
                "type": "llm_call",
                "provider": "openai",
                "input_messages": [],
                "response_content": {"unexpected": ["partial", None]},
            },
        ],
    }
    diagnosis = diagnose_run(messy_run, use_llm=False)
    if diagnosis.get("confidence", 1.0) >= 0.6:
        return "WARN", "messy trace produced medium/high confidence"
    if not diagnosis.get("low_confidence_message"):
        return "FAIL", "messy trace did not include low-confidence messaging"
    return "PASS", ""


def _doctor_anonymization() -> tuple[str, str]:
    raw = {
        "email": "alex@example.com",
        "api_key": "sk-test123456789abcdef",
        "headers": {"Authorization": "Bearer testBearerToken123456789"},
        "text": "password=hunter2 secret=supersecretvalue token=tok_live_1234567890abcdef",
        "usage": {"input_tokens": 123, "output_tokens": 45, "total_tokens": 168},
    }
    cleaned = _anonymize_value(raw)
    cleaned_text = json.dumps(cleaned, sort_keys=True)
    leaked = [
        value
        for value in [
            "alex@example.com",
            "sk-test123456789abcdef",
            "testBearerToken123456789",
            "hunter2",
            "supersecretvalue",
            "tok_live_1234567890abcdef",
        ]
        if value in cleaned_text
    ]
    if leaked:
        return "FAIL", f"secret leaked: {leaked[0]}"
    usage = cleaned.get("usage", {})
    if usage.get("input_tokens") != 123 or usage.get("output_tokens") != 45:
        return "FAIL", "token counts were redacted"
    return "PASS", ""


def _doctor_evaluation() -> tuple[str, str]:
    report = evaluate_cases()
    required = {"fixture_accuracy", "low_confidence_rate", "fixture_cases"}
    missing = required - set(report)
    if missing:
        return "FAIL", f"evaluation missing {', '.join(sorted(missing))}"
    if report["fixture_cases"] == 0:
        return "FAIL", "no fixture cases found"
    if report["fixture_accuracy"] < 1.0:
        return "FAIL", f"fixture accuracy {report['fixture_accuracy']:.0%}"
    return "PASS", ""


def _doctor_tool_selection_run() -> dict[str, Any]:
    # Prefer the real fixture so the doctor tests the same data as evaluate.
    _fixture_path = Path(__file__).resolve().parent / "tests" / "phase2_runs" / "phase2_tool_selection.json"
    if _fixture_path.exists():
        try:
            return json.loads(_fixture_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            pass
    # Fallback inline fixture when tests/ directory is not present.
    return {
        "run_id": "agentlens_doctor_tool_selection",
        "name": "doctor_tool_selection",
        "status": "error",
        "spans": [
            {
                "type": "llm_call",
                "provider": "anthropic",
                "model": "claude-3-5-sonnet-latest",
                "input_messages": [
                    {
                        "role": "user",
                        "content": "Find the renewal status for customer:alex using local records.",
                    }
                ],
                "tools": [
                    {"name": "search_web", "description": "find info about a topic"},
                    {"name": "query_db", "description": "find info about a topic"},
                ],
                "response_content": [
                    {
                        "type": "text",
                        "text": "Both tools look similar, so I will use search_web.",
                    },
                    {
                        "type": "tool_use",
                        "name": "search_web",
                        "input": {"query": "customer:alex renewal status"},
                    },
                ],
                "usage": {"input_tokens": 100, "output_tokens": 30},
            },
            {
                "type": "tool_call",
                "tool_name": "search_web",
                "input": {"query": "customer:alex renewal status"},
                "output": {
                    "status": "error",
                    "error": "Customer records are only available in query_db.",
                },
            },
        ],
    }


def _anonymize_value(value: Any) -> Any:
    if isinstance(value, str):
        return _anonymize_string(value)
    if isinstance(value, list):
        return [_anonymize_value(item) for item in value]
    if isinstance(value, dict):
        cleaned = {}
        for key, item in value.items():
            if _secret_key_name(str(key)):
                cleaned[key] = "[REDACTED]"
            elif _pii_value_key(str(key)):
                cleaned[key] = "[PII]"
            else:
                cleaned[key] = _anonymize_value(item)
        return cleaned
    return value


def _secret_key_name(key: str) -> bool:
    lowered = key.lower()
    # Auth credentials whose VALUE must be redacted by key name. "token" is matched
    # only as a real auth-token field, not as a substring — otherwise legitimate
    # token-COUNT fields (prompt_tokens, max_tokens, num_tokens, ...) get nuked.
    auth_token_keys = {
        "token", "access_token", "refresh_token", "auth_token",
        "bearer_token", "session_token", "id_token", "auth", "bearer",
    }
    if lowered in auth_token_keys:
        return True
    markers = ("api_key", "apikey", "secret", "password", "authorization", "cookie")
    return any(marker in lowered for marker in markers)


def _pii_value_key(key: str) -> bool:
    """Keys whose VALUE is personal data — redact the value, keep the key.

    Deliberately specific so it never collides with diagnostic fields like
    ``tool_name``, the run ``name``, or a tool definition's ``name``. Catches
    personal data that arrives in labeled fields (DB rows, structured tool
    output), which is where freeform names usually live.
    """
    return key.lower() in {
        "full_name", "first_name", "last_name", "customer_name", "fullname",
        "patient_name", "person_name", "street", "street_address", "home_address",
        "mailing_address", "dob", "date_of_birth", "ssn", "social_security",
        "phone_number", "credit_card", "card_number", "account_number",
        "routing_number", "passport", "license_number",
    }


# PII patterns applied to every captured string. Order matters: longer/more
# specific number patterns (card, SSN) run before shorter ones (phone) so they
# do not partially match. IDs keep their prefix word so diagnostic signal
# ("customer", "account") survives while the identifying value is removed.
_PII_PATTERNS = [
    # credentials / keys
    (r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", "[EMAIL]"),
    (r"\bsk-[A-Za-z0-9_-]{12,}\b", "[API_KEY]"),
    (r"\bsk-ant-[A-Za-z0-9_-]{12,}\b", "[API_KEY]"),
    (r"\b(xox[baprs]-[A-Za-z0-9-]{10,})\b", "[TOKEN]"),
    (r"\b(al_[A-Za-z0-9_-]{8,})\b", "[AGENTLENS_KEY]"),
    (r"(?i)(bearer\s+)[A-Za-z0-9._-]{12,}", r"\1[TOKEN]"),
    (r"(?i)(api[_-]?key\s*[:=]\s*)[A-Za-z0-9._-]{8,}", r"\1[API_KEY]"),
    (r"(?i)(token\s*[:=]\s*)[A-Za-z0-9._-]{8,}", r"\1[TOKEN]"),
    (r"(?i)(password\s*[:=]\s*)\S+", r"\1[PASSWORD]"),
    (r"(?i)(secret\s*[:=]\s*)[A-Za-z0-9._-]{8,}", r"\1[SECRET]"),
    # structured PII — specific before general
    (r"\b(?:\d[ -]?){15,16}\b", "[CARD]"),                       # credit card (16 digits)
    (r"\b\d{3}-\d{2}-\d{4}\b", "[SSN]"),                         # SSN
    (r"(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]\d{3}[-.\s]\d{4}\b", "[PHONE]"),  # phone
    (r"\b\d{1,3}(?:\.\d{1,3}){3}\b", "[IP]"),                    # IPv4
    (r"\$\s?\d{1,3}(?:,\d{3})*(?:\.\d{2})?", "[AMOUNT]"),        # money
    # identifiers — keep the leading word, drop the value
    (r"(?i)\b(customer)\s*[:=]\s*\S+", r"\1:[ID]"),
    (r"(?i)\b(acct|account)[_:\s-]*\w*\d{3,}\w*", r"\1_[ID]"),
    (r"(?i)\b(user|users)\.(id)\s*[:=]\s*\d+", r"\1.\2=[ID]"),
    # US street address
    (r"\b\d{1,6}\s+[A-Z][\w.]*(?:\s+[A-Z][\w.]*)*\s+"
     r"(?:St|Street|Ave|Avenue|Rd|Road|Blvd|Boulevard|Ln|Lane|Dr|Drive|Way|Ct|Court)\b"
     r"(?:,?\s+[A-Z][a-zA-Z]+)*(?:,?\s+[A-Z]{2})?(?:\s+\d{5}(?:-\d{4})?)?", "[ADDRESS]"),
]


_NER_MODEL: Any = None
_NER_CHECKED = False


def _get_ner_model() -> Any:
    """Load and cache a spaCy NER model for person-name detection, if available.

    Optional dependency: install with `pip install spacy && python -m spacy
    download en_core_web_sm` (or `pip install 'agentlens[pii]'`). If unavailable,
    name redaction degrades to regex-only — we warn once and never crash.
    """
    global _NER_MODEL, _NER_CHECKED
    if _NER_CHECKED:
        return _NER_MODEL
    _NER_CHECKED = True
    try:
        import spacy

        _NER_MODEL = spacy.load("en_core_web_sm", disable=["parser", "lemmatizer"])
    except Exception:
        _NER_MODEL = None
        print(
            "Note: spaCy/en_core_web_sm not installed — freeform person names are NOT "
            "redacted (structured PII still is). Enable full name redaction with: "
            "pip install spacy && python -m spacy download en_core_web_sm"
        )
    return _NER_MODEL


def _redact_person_names(text: str) -> str:
    """Replace spaCy-detected PERSON entities with [NAME]. No-op without spaCy."""
    if not text or not any(ch.isalpha() for ch in text):
        return text
    model = _get_ner_model()
    if model is None:
        return text
    doc = model(text)
    result = text
    # Replace from the end so earlier character offsets stay valid.
    for ent in sorted(
        (e for e in doc.ents if e.label_ == "PERSON"),
        key=lambda e: e.start_char,
        reverse=True,
    ):
        result = result[: ent.start_char] + "[NAME]" + result[ent.end_char :]
    return result


def _anonymize_string(value: str) -> str:
    cleaned = value
    for pattern, replacement in _PII_PATTERNS:
        cleaned = re.sub(pattern, replacement, cleaned)
    cleaned = _redact_person_names(cleaned)
    return cleaned


if __name__ == "__main__":
    main()
