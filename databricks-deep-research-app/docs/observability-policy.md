# Agent Designer Observability Policy

Version: V1.5  
Last updated: 2026-04-28  
Scope: Agent Designer V4 (tree-of-blocks editor, server-driven registry, ETag concurrency)

This document defines the complete observability contract for Agent Designer V1, specifies which signals are deferred to V1.5, documents the pre-deploy data check procedure, and lists the remaining V1.5 migration items.

---

## 1. Server-side Signals (V1)

Six signals are emitted server-side in V1. Five are new; `run_duration_seconds` is included to track workflows approaching the OAuth token TTL boundary.

| Signal Name | Type | Emission Point | Alert Threshold | Owner |
|---|---|---|---|---|
| `agent_designer.registry_fetch_ms` | histogram | Server — every `GET /registry` response | p99 > 500 ms over 5 min → investigate | backend |
| `agent_designer.validation_error` | counter (by `error_kind`) | Server — `POST /validate` on schema rejection | Rate > 5% over 5 min → page on-call | backend |
| `agent_designer.save_etag_conflict` | counter | Server — 409 response from `PATCH /agents/v2/{id}` | Rate > 1% over 15 min → investigate concurrent-edit UX | backend |
| `agent_designer.chat_mutation` | structured log | Server — every chat tool call that mutates agent state | Any error outcome in prod → create ticket | backend |
| `agent_designer.token_refresh_attempt` | counter (by `outcome`) | Server — OBO token refresh attempt | Sustained rate > 0.5/min → investigate session length | backend |
| `agent_designer.token_refresh_failure` | counter (by `error_kind`) | Server — OBO token refresh failure | Rate > 0.1 / 5 min → page on-call | backend |
| `agent_designer.run_duration_seconds` | histogram | Server — workflow run completion | p95 > 3000 s (50 min) → surface editor banner | backend |

**Rationale by signal:**

- `registry_fetch_ms` — The registry endpoint is on the critical path for every editor load. A latency spike here degrades all users before any frontend error is visible.
- `validation_error` — Schema validation failures indicate either a frontend bug sending malformed payloads or a registry/schema drift. A spike rate triggers on-call to disambiguate.
- `save_etag_conflict` — Concurrent edits produce 409s. A sustained conflict rate means the ETag concurrency model is being thrashed and the EtagConflictModal UX may need tuning.
- `chat_mutation` — Every LLM-driven mutation of agent state is audited as a structured log entry (agent_id, tool_name, outcome, user_id). This provides a tamper-evident trail without a separate audit store in V1.
- `token_refresh_attempt` — V1.5 ships server-side OBO token refresh. Tracking attempt frequency (labelled by `outcome`: success / failure / noop) establishes a baseline and surfaces unexpected refresh loops.
- `token_refresh_failure` — Counts refresh attempts that ended in an error, labelled by `error_kind` (expired_refresh / network / permission). A rate spike triggers on-call to investigate the token rotation path.
- `run_duration_seconds` — Runs exceeding 50 minutes are approaching the OAuth token TTL (~60 min). The histogram drives the editor banner that warns users before the session expires.

**Dashboard:** Add an "Agent Designer V4" panel to the existing Grafana board. Include one row per registered agent for per-agent `validation_error` and `save_etag_conflict` breakdowns.

---

## 2. Deferred Client-side Signals (V1.5)

The following three signals require browser-side metrics infrastructure (Web Vitals, `performance.mark`, or an equivalent client metrics SDK) that V1 does not ship. They are logged as best-effort structured console events in V1 but are not ingested into the metrics backend until V1.5.

| Signal Name | Type | Emission Point | Alert Threshold | Owner |
|---|---|---|---|---|
| `agent_designer.block_render_count` | client metric | Frontend — block-stack render cycle | Per-agent p95 render count regression vs. baseline → ticket | frontend |
| `agent_designer.dnd_drop_failed` | error log | Frontend — dnd-kit drop handler on path resolution failure | Any occurrence in prod → frontend bug; create ticket | frontend |
| `agent_designer.widget_fallback` | warning log | Frontend — thin JSON-Schema walker when `x-widget` key is absent from client registry | Count > 0 in production → frontend bug; create ticket | frontend |

**Why deferred:** V1 does not include a client metrics pipeline. Instrumenting these signals requires either a Web Vitals integration or a lightweight `performance.mark` → backend ingest path. Shipping placeholder instrumentation without an ingestion backend would produce silent data loss. V1.5 adds the pipeline first, then enables ingestion for these three signals.

**V1.5 promotion:** `token_refresh_attempt` and `token_refresh_failure` were originally listed here as client-side deferred signals. Both are now emitted server-side as of V1.5 and have been promoted to §1.

**V1 interim behaviour:** `dnd_drop_failed` and `widget_fallback` are emitted as `console.error` / `console.warn` structured objects in V1. They are visible in browser DevTools and in any log-forwarding setup, but they do not count toward SLOs until V1.5 ingestion lands.

---

## 3. Pre-deploy Data Check

### Purpose

Migration 024 drops the `custom_agents` table. Before applying this migration in any environment, run `scripts/preflight_v1_data_check.py` to detect rows that would be permanently deleted.

### When to run

Run this script **before** applying migration 024 to any non-development environment. The deploy pipeline must gate on exit code 0.

### CLI arguments

| Argument | Type | Description |
|---|---|---|
| `--connection-string DSN` | string, optional | asyncpg-compatible PostgreSQL DSN. Overrides `DATABASE_URL` env var. Use a secret-manager value rather than an inline literal. |
| `--export-path PATH` | path, optional | Write existing rows to this file as JSONL before the drop. Required when the table is non-empty and `--check-only` is not set. |
| `--check-only` | flag | Count rows and report only — never export. Exits 2 if the table is non-empty regardless of `--export-path`. |

### Exit codes

| Code | Meaning |
|---|---|
| 0 | Table is empty (safe to drop), OR rows were exported successfully to `--export-path` |
| 1 | Unexpected error — connection failure, permission denied, etc. |
| 2 | Table has rows and no `--export-path` was given (or `--check-only` was set) — deploy is BLOCKED |

### Example invocations

**1. Safe-to-drop check using `DATABASE_URL`:**
```bash
python scripts/preflight_v1_data_check.py
```
Exits 0 with `OK: custom_agents is empty; safe to drop` if the table is empty.

**2. Export rows before dropping (production):**
```bash
python scripts/preflight_v1_data_check.py \
  --connection-string "$PROD_DATABASE_URL" \
  --export-path /tmp/custom_agents_backup.jsonl
```
Exits 0 with `EXPORTED N row(s) to /tmp/custom_agents_backup.jsonl; safe to drop after backup verified`.

**3. Check only — no export, block if non-empty:**
```bash
python scripts/preflight_v1_data_check.py --check-only
```
Exits 2 with an error message on stderr if the table has rows. Use this in CI where you want an explicit failure rather than a silent export.

### Connection string resolution

The script resolves the connection string in this order:
1. `--connection-string` CLI flag
2. `DATABASE_URL` environment variable (strips `postgresql+asyncpg://` prefix automatically)

If neither is set, the script exits 1 with an actionable error message.

---

## 4. V1.5 Migration Plan — Deferred Work

The following items are explicitly out of scope for V1 and tracked for V1.5.

- **YAML import/export endpoints** — `POST /agent-designer/import-yaml` and `GET /agents/v2/{id}/yaml`. Allows round-trip agent definitions as human-readable YAML, enabling GitOps workflows and bulk migrations. Owner: backend. Estimated effort: 2-3 days. Deferred because V1 focuses on the JSON-native tree editor; YAML serialisation adds a schema mapping layer that needs separate design review.

- **Mermaid export** — `GET /agents/v2/{id}/mermaid`. Returns a Mermaid diagram definition for the agent's block tree, enabling embedded documentation. Owner: backend. Estimated effort: 1-2 days. Deferred because the Mermaid rendering library is already included for read-only display in V1; the export endpoint is a low-risk add but is not on the V1 critical path.

- **EtagConflictModal diff path** — Three-way merge UI in `EtagConflictModal` that shows a side-by-side diff between the local version, the server version, and a suggested merge. Owner: frontend. Estimated effort: 2-3 days. V1 ships a simpler modal that lets the user choose "keep mine" or "take server version"; the full diff view requires a JSON-diff library integration and additional UX design.

- **OBO token refresh** — Server-side OAuth token refresh during long-running workflows that exceed the ~60-minute token TTL. Owner: backend. Estimated effort: 3-5 days. V1 bounds workflows by the OAuth TTL and surfaces a 50-minute editor banner (driven by `run_duration_seconds`). Automatic refresh was explicitly deferred per Codex finding M2 because it requires a secure token rotation path that needs a separate security review.

- **Frontend client metrics pipeline** — Web Vitals integration or equivalent `performance.mark` → backend ingest path required to promote the four client-side signals in Section 2 from best-effort console logs to first-class SLO signals. Owner: frontend platform. Estimated effort: 2-3 days. Deferred because standing up a new client metrics ingestion pipeline is a platform-level change affecting all frontend features, not only Agent Designer.

---

## 5. Compliance Gates

The following gates must be cleared before V1 ships to production. They are organisational or process requirements, not purely technical.

- **SECCOM signoff** — Agent Designer V4 introduces new API endpoints (`/registry`, `/validate`, `PATCH /agents/v2/{id}`), a new OBO token flow path, and direct LLM-driven mutations of stored agent configuration. A SECCOM review is required before production launch. This is an organisational process gate; it does not block staging deployments.

- **Accessibility (a11y) audit** — A Playwright-based a11y audit (`make e2e-a11y`) covering the block-stack editor, drag-and-drop interactions, and EtagConflictModal is targeted for V1.5 alongside the broader e2e infrastructure expansion. V1 ships with manual keyboard-navigation testing only. The `make e2e-a11y` target does not yet exist and will be added in V1.5.

- **Observability policy review** — This document (`docs/observability-policy.md`) constitutes the observability policy review artefact. It must be reviewed and approved by the on-call engineering lead before migration 024 is applied in production. Updates to signal names or alert thresholds must be reflected here before the change is deployed.
