# AgentLens hosted backend

The server side of AgentLens. It receives anonymized run uploads (the output of
`agentlens upload prepare`), diagnoses them server-side with the validated engine,
stores them in SQLite with indexed columns, and serves a dashboard + JSON API.

## Run

```bash
pip install -e '.[server]'
uvicorn server.app:app --reload          # http://localhost:8000
open http://localhost:8000/dashboard
```

DB path defaults to `.agentlens_server.db`; override with `AGENTLENS_DB`.

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| POST | `/ingest` | Accept an (anonymized) run; **rejects residual PII with 422**; diagnoses it server-side; stores it. Returns `run_url`. |
| GET | `/runs/{run_id}` | Full stored run + diagnosis. |
| GET | `/api/runs` | Recent runs as JSON; filter by `diagnosis_status`, `root_cause`; `limit`. |
| GET | `/api/stats` | Totals: run count, failures, failure rate, total wasted spend, counts by root cause. |
| POST | `/reclassify` | Re-diagnose every stored run with the current engine (run after an engine upgrade so stored badges don't go stale). |
| GET | `/dashboard` | HTML dashboard: stat cards + run table with status badges, root cause, confidence, wasted spend. |
| GET | `/health` | Liveness + run count. |

## Design notes

- **Compute-at-ingest.** Diagnosis and status are computed by the server from the
  spans (`diagnose_run`, `run_status`) — the client's diagnosis is never trusted.
  Results are stored in indexed columns so the dashboard can filter/sort/aggregate.
- **Fails closed on PII.** A last-line PII scan runs on every payload; anything
  that still contains an email/SSN/card/phone/key pattern is rejected, not stored.
- **Single-tenant, no auth.** Intended for local / self-host. Auth, Postgres,
  multi-tenancy, and a scheduled `/reclassify` are the production follow-ups.
