"""AgentLens hosted backend.

Receives anonymized run uploads, diagnoses them server-side with the validated
engine (compute-at-ingest, never trust the client), stores them in SQLite with
indexed columns so the dashboard can filter/sort/aggregate, and serves a
dashboard plus a JSON API.

Run it:  uvicorn server.app:app --reload   (then open http://localhost:8000/dashboard)

This is the server side of AgentLens. It consumes the output of
`agentlens upload prepare` (anonymized-only payloads). It is intentionally
single-tenant and unauthenticated for local/self-host use; auth, Postgres, and
multi-tenancy are the production follow-ups.
"""

from __future__ import annotations

import json
import os
import re
import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from html import escape
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, ConfigDict

from agentlens_engine.diagnose import diagnose_run
from agentlens_engine.status import run_status

DB_PATH = Path(os.environ.get("AGENTLENS_DB", ".agentlens_server.db"))
ENGINE_VERSION = "1"  # bump to invalidate stored diagnoses; drives /reclassify

# Last-line PII gate at the ingest boundary. A leak here is the one that lands
# in the hosted store, so ingest fails closed if any of these survive.
_PII_PATTERNS = {
    "email": r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}",
    "ssn": r"\b\d{3}-\d{2}-\d{4}\b",
    "credit_card": r"\b(?:\d[ -]?){15,16}\b",
    "phone": r"(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]\d{3}[-.\s]\d{4}\b",
    "api_key": r"\bsk-[A-Za-z0-9_-]{12,}\b",
}


def _residual_pii(payload: dict[str, Any]) -> list[str]:
    blob = json.dumps(payload, default=str)
    return [kind for kind, pattern in _PII_PATTERNS.items() if re.search(pattern, blob)]


SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id            TEXT PRIMARY KEY,
    created_at        TEXT NOT NULL,
    ingested_at       TEXT NOT NULL,
    name              TEXT,
    execution_status  TEXT,
    diagnosis_status  TEXT,
    root_cause        TEXT,
    confidence        REAL,
    failed_at_step    INTEGER,
    wasted_tokens     INTEGER,
    wasted_cost_usd   REAL,
    total_cost_usd    REAL,
    span_count        INTEGER,
    engine_version    TEXT,
    diagnosis_json    TEXT,
    run_json          TEXT
);
CREATE INDEX IF NOT EXISTS idx_diagnosis_status ON runs (diagnosis_status);
CREATE INDEX IF NOT EXISTS idx_root_cause       ON runs (root_cause);
CREATE INDEX IF NOT EXISTS idx_created_at        ON runs (created_at);
"""


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    with closing(_connect()) as conn:
        conn.executescript(SCHEMA)
        conn.commit()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _diagnose_and_store(run: dict[str, Any], created_at: str) -> dict[str, Any]:
    """Run the validated engine and upsert the result. Returns {diagnosis, status}."""
    diagnosis = diagnose_run(run, use_llm=False)
    status = run_status(run)
    impact = diagnosis.get("impact") or {}
    row = {
        "run_id": str(run.get("run_id")),
        "created_at": created_at,
        "ingested_at": _now(),
        "name": run.get("name"),
        "execution_status": status["execution_status"],
        "diagnosis_status": status["diagnosis_status"],
        "root_cause": status["root_cause"],
        "confidence": float(diagnosis.get("confidence") or 0.0),
        "failed_at_step": int(diagnosis.get("failed_at_step") or 0),
        "wasted_tokens": int(impact.get("wasted_tokens") or 0),
        "wasted_cost_usd": float(impact.get("wasted_cost_usd") or 0.0),
        "total_cost_usd": float(impact.get("total_cost_usd") or 0.0),
        "span_count": len(run.get("spans") or []),
        "engine_version": ENGINE_VERSION,
        "diagnosis_json": json.dumps(diagnosis, default=str),
        "run_json": json.dumps(run, default=str),
    }
    cols = ", ".join(row)
    placeholders = ", ".join(f":{c}" for c in row)
    updates = ", ".join(f"{c}=excluded.{c}" for c in row if c not in ("run_id", "created_at"))
    with closing(_connect()) as conn:
        conn.execute(
            f"INSERT INTO runs ({cols}) VALUES ({placeholders}) "
            f"ON CONFLICT(run_id) DO UPDATE SET {updates}",
            row,
        )
        conn.commit()
    return {"diagnosis": diagnosis, "status": status}


app = FastAPI(title="AgentLens", version="0.1.0")

# Ensure the schema exists no matter how the app is loaded (uvicorn, tests, etc.).
init_db()


class IngestBody(BaseModel):
    # Accept the full anonymized run shape; extra fields are kept.
    model_config = ConfigDict(extra="allow")
    run_id: str
    spans: list[dict[str, Any]] = []
    name: str | None = None
    status: str | None = None
    started_at: str | None = None
    metadata: dict[str, Any] = {}


@app.get("/health")
def health() -> dict[str, Any]:
    with closing(_connect()) as conn:
        count = conn.execute("SELECT COUNT(*) AS n FROM runs").fetchone()["n"]
    return {"status": "ok", "runs": count, "engine_version": ENGINE_VERSION}


@app.post("/ingest")
def ingest(body: IngestBody) -> dict[str, Any]:
    payload = body.model_dump()
    leaks = _residual_pii(payload)
    if leaks:
        # Fail closed: refuse to store data that still contains PII.
        raise HTTPException(status_code=422, detail={"error": "residual_pii", "kinds": leaks})

    run = {
        "run_id": body.run_id,
        "name": body.name,
        "status": body.status,
        "started_at": body.started_at,
        "spans": body.spans,
    }
    created_at = body.started_at or _now()
    result = _diagnose_and_store(run, created_at)
    status = result["status"]
    return {
        "status": "ok",
        "run_url": f"/runs/{body.run_id}",
        "diagnosis": {
            "execution_status": status["execution_status"],
            "diagnosis_status": status["diagnosis_status"],
            "root_cause": status["root_cause"],
            "confidence": status["confidence"],
        },
    }


@app.get("/runs/{run_id}")
def get_run(run_id: str) -> dict[str, Any]:
    with closing(_connect()) as conn:
        row = conn.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,)).fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="run not found")
    return {
        "run_id": row["run_id"],
        "name": row["name"],
        "created_at": row["created_at"],
        "execution_status": row["execution_status"],
        "diagnosis_status": row["diagnosis_status"],
        "run": json.loads(row["run_json"]),
        "diagnosis": json.loads(row["diagnosis_json"]),
    }


@app.get("/api/runs")
def api_runs(
    limit: int = Query(20, ge=1, le=200),
    diagnosis_status: str | None = None,
    root_cause: str | None = None,
) -> dict[str, Any]:
    clauses, params = [], []
    if diagnosis_status:
        clauses.append("diagnosis_status = ?")
        params.append(diagnosis_status)
    if root_cause:
        clauses.append("root_cause = ?")
        params.append(root_cause)
    where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
    with closing(_connect()) as conn:
        rows = conn.execute(
            "SELECT run_id, created_at, name, execution_status, diagnosis_status, "
            "root_cause, confidence, failed_at_step, wasted_cost_usd, span_count "
            f"FROM runs{where} ORDER BY created_at DESC LIMIT ?",
            (*params, limit),
        ).fetchall()
    return {"runs": [dict(r) for r in rows]}


@app.get("/api/stats")
def api_stats() -> dict[str, Any]:
    with closing(_connect()) as conn:
        total = conn.execute("SELECT COUNT(*) AS n FROM runs").fetchone()["n"]
        failures = conn.execute(
            "SELECT COUNT(*) AS n FROM runs WHERE diagnosis_status = 'failure_detected'"
        ).fetchone()["n"]
        wasted = conn.execute("SELECT COALESCE(SUM(wasted_cost_usd), 0) AS s FROM runs").fetchone()["s"]
        by_cause = conn.execute(
            "SELECT root_cause, COUNT(*) AS n FROM runs "
            "WHERE root_cause IS NOT NULL GROUP BY root_cause ORDER BY n DESC"
        ).fetchall()
    return {
        "total_runs": total,
        "failures": failures,
        "failure_rate": (failures / total) if total else 0.0,
        "total_wasted_cost_usd": round(wasted, 6),
        "by_root_cause": {r["root_cause"]: r["n"] for r in by_cause},
    }


@app.post("/reclassify")
def reclassify() -> dict[str, Any]:
    """Re-diagnose every stored run with the current engine.

    Persisted diagnoses go stale when the engine improves; this refreshes them so
    dashboard badges never lie after a deploy.
    """
    with closing(_connect()) as conn:
        rows = conn.execute("SELECT run_id, run_json, created_at FROM runs").fetchall()
    for row in rows:
        _diagnose_and_store(json.loads(row["run_json"]), row["created_at"])
    return {"status": "ok", "reclassified": len(rows), "engine_version": ENGINE_VERSION}


_BADGE = {
    "healthy": "#1f8b4c",
    "failure_detected": "#c0392b",
    "uncertain": "#b9770e",
    "unknown": "#555",
}


@app.get("/dashboard", response_class=HTMLResponse)
def dashboard() -> str:
    stats = api_stats()
    runs = api_runs(limit=50)["runs"]
    rows_html = []
    for r in runs:
        status = r["diagnosis_status"] or "unknown"
        color = _BADGE.get(status, "#555")
        cause = r["root_cause"] or "—"
        conf = f"{r['confidence']:.2f}" if r["confidence"] else "—"
        wasted = f"${r['wasted_cost_usd']:.6f}" if r["wasted_cost_usd"] else "—"
        rows_html.append(
            f"<tr>"
            f"<td><a href='/runs/{escape(r['run_id'])}'>{escape(r['run_id'][:18])}</a></td>"
            f"<td>{escape(r['name'] or '')}</td>"
            f"<td>{escape(r['execution_status'] or '')}</td>"
            f"<td><span style='background:{color};color:#fff;padding:2px 8px;border-radius:10px;font-size:12px'>{escape(status)}</span></td>"
            f"<td>{escape(cause)}</td>"
            f"<td>{conf}</td>"
            f"<td>{r['failed_at_step'] or '—'}</td>"
            f"<td>{wasted}</td>"
            f"<td>{escape((r['created_at'] or '')[:19])}</td>"
            f"</tr>"
        )
    causes = " · ".join(f"{k}: {v}" for k, v in stats["by_root_cause"].items()) or "none"
    return f"""<!doctype html><html><head><meta charset='utf-8'><title>AgentLens</title>
<style>
  body {{ font: 14px -apple-system, system-ui, sans-serif; margin: 32px; color: #1a1a1a; }}
  h1 {{ margin: 0 0 4px; }}
  .sub {{ color: #666; margin-bottom: 20px; }}
  .cards {{ display: flex; gap: 16px; margin-bottom: 24px; }}
  .card {{ background: #f5f5f7; border-radius: 12px; padding: 16px 20px; min-width: 140px; }}
  .card .v {{ font-size: 26px; font-weight: 600; }}
  .card .k {{ color: #666; font-size: 12px; text-transform: uppercase; }}
  table {{ border-collapse: collapse; width: 100%; }}
  th, td {{ text-align: left; padding: 8px 10px; border-bottom: 1px solid #eee; }}
  th {{ color: #666; font-size: 12px; text-transform: uppercase; }}
  a {{ color: #2563eb; text-decoration: none; }}
</style></head><body>
<h1>AgentLens</h1>
<div class='sub'>Hosted run dashboard · engine v{ENGINE_VERSION}</div>
<div class='cards'>
  <div class='card'><div class='v'>{stats['total_runs']}</div><div class='k'>Runs</div></div>
  <div class='card'><div class='v'>{stats['failures']}</div><div class='k'>Failures detected</div></div>
  <div class='card'><div class='v'>{stats['failure_rate']:.0%}</div><div class='k'>Failure rate</div></div>
  <div class='card'><div class='v'>${stats['total_wasted_cost_usd']:.4f}</div><div class='k'>Wasted spend</div></div>
</div>
<div class='sub'>By root cause: {escape(causes)}</div>
<table>
  <tr><th>run</th><th>name</th><th>execution</th><th>diagnosis</th><th>root cause</th>
      <th>conf</th><th>step</th><th>wasted</th><th>time</th></tr>
  {''.join(rows_html) or "<tr><td colspan='9'>No runs yet — POST to /ingest.</td></tr>"}
</table>
</body></html>"""
