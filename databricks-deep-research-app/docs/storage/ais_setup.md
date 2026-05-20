# AIS workspace setup for storage integration tests

This runbook provisions Unity Catalog and SQL Warehouse resources needed to
run `tests/contract/storage` and `tests/integration/storage` against real
Lakebase and SQL Warehouse backends in the AIS workspace.

---

## Quick reference — reset + deploy the chat-document schema

The storage layer now defaults to the cached chat-document schema
(`STORAGE_SERVICE_IMPL=cached`, `STORAGE_BACKEND=lakebase`). All tooling is
wired through the Makefile.

### Local dev

```bash
make db-local            # start local Docker Postgres (one-time)
make db-reset            # drop + alembic + apply chat-document DDL
make dev                 # boot backend :8000 + frontend :5173
```

### First-time AIS deploy

```bash
make db-reset TARGET=ais                     # drops remote Lakebase, creates schema
make deploy TARGET=ais BRAVE_SCOPE=msh       # uploads wheels + deploys app
```

### Iterative AIS deploy (schema unchanged)

```bash
make deploy TARGET=ais BRAVE_SCOPE=msh
```

The app's lifespan re-runs the chat-document `migrate()` on boot; because
the DDL is `IF NOT EXISTS`, this is a safe no-op.

### Nuke-and-redeploy AIS

```bash
make db-reset TARGET=ais                     # wipes data, keeps schema fresh
make deploy TARGET=ais BRAVE_SCOPE=msh
```

### Verify it worked

```bash
make test-storage-ais-lakebase   # full end-to-end lifecycle against AIS Lakebase
```

### Rollback to legacy ORM

Add `STORAGE_SERVICE_IMPL=sqlalchemy_legacy` to `../.env.ais` (or the
`databricks.yml` `config:` block) and redeploy. The new tables stay in
place but are unused by the legacy path.

---

## One-time Unity Catalog provisioning

1. **Pick a catalog.** We use `main`. Any catalog the service principal can
   `CREATE SCHEMA` in will do.
2. **Create a dedicated schema for test runs.** Each test run picks a
   unique sub-schema (`deep_research_test_<uuid>`) so parallel CI lanes
   don't collide.

   ```sql
   -- As the service principal or an admin:
   CREATE CATALOG IF NOT EXISTS main;
   -- Grant CREATE SCHEMA to the test principal.
   GRANT CREATE SCHEMA ON CATALOG main TO `<service-principal-id>`;
   ```

3. **Warehouse.** Any SQL Warehouse in AIS works. Capture its ID — the
   easiest way is via the Databricks UI:
   **SQL → Warehouses → (click the warehouse) → "Connection details"** →
   copy the warehouse ID from the URL or the JDBC URL.

   Set it in the environment:

   ```
   export STORAGE_WAREHOUSE_ID=<warehouse-id>
   ```

4. **Lakebase (optional, only for `STORAGE_TEST_LAKEBASE=1`).**
   The existing `Settings.lakebase_*` flow — provisioned instance via
   `LAKEBASE_INSTANCE_NAME` or autoscaling via `ENDPOINT_NAME` — is reused.
   No additional setup.

## `.env.integration` template

Copy to `.env.integration` and source it before running the tests:

```bash
# Lakebase (optional; set one of the two)
LAKEBASE_INSTANCE_NAME=<instance-id>
# or:
# ENDPOINT_NAME=projects/<pid>/branches/<bid>/endpoints/<eid>

# SQL Warehouse (required for STORAGE_TEST_WAREHOUSE=1)
STORAGE_WAREHOUSE_ID=<warehouse-id>
STORAGE_CATALOG=main
STORAGE_SCHEMA_PREFIX=deep_research_test

# Gate flags — pick what you want to run.
STORAGE_TEST_LAKEBASE=1
STORAGE_TEST_WAREHOUSE=1
STORAGE_INTEGRATION=1

# Databricks auth — the app already supports the usual config profiles.
DATABRICKS_CONFIG_PROFILE=<profile>
```

## Running the tests

```bash
# Contract suite (FakeBackend always; real backends per env flag).
uv run pytest tests/contract/storage -v

# Integration suite (real DBs only).
uv run pytest tests/integration/storage -v
```

## Teardown

Integration-test fixtures drop their temp schemas in a `try/finally`. If a
test crashes mid-run and leaves an orphan, run:

```bash
uv run python -m deep_research.storage.cleanup_test_schemas \
    --catalog main --prefix deep_research_test_ --older-than 1d
```

(That script is not shipped in v1 — manually `DROP SCHEMA CASCADE` via SQL
for now.)

## Permissions cheat sheet

Minimum grants for the test principal:

| Object | Privilege | Why |
|---|---|---|
| Catalog `main` | `CREATE SCHEMA` | Per-run temp schemas. |
| Warehouse | `CAN_USE` | Execute statements. |
| Any temp schema | `ALL PRIVILEGES` (via ownership) | DDL + DML. |

For Lakebase, the existing service-principal OAuth path already has full
DB privileges on the app's database (`deep_research`); no additional grants
needed.
