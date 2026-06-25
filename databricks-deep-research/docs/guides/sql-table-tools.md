# SQL / Table Tools — Researching over Delta Tables

> Structured research over Delta tables: discover, search, read, page, and aggregate rows
> with parameterized SQL — without natural-language-to-SQL.

## Overview

The `table_*` tool family lets a research agent answer questions **from the rows of a Delta
table** — the OfficeQA-style "the answer is a cell / a row / a group total in this table" use
case. Instead of free-text generation, the agent navigates a known table: it lists what tables
are exposed, substring-searches a content column, reads filtered/projected/paginated rows,
walks to neighboring rows, materializes specific rows for downstream compute, and runs simple
aggregates.

This is **not** Genie. Genie (`kind: genie`, `SourceKind.sql_analytics`) translates a natural
language question into arbitrary SQL via an LLM. The table tools take the opposite approach: the
table identity, the column roles, and the predicate grammar are all fixed up front, and the only
SQL the agent can cause is a bounded, parameterized `SELECT` built by the framework's own
compiler. Every value the LLM supplies is bound as a statement parameter — never concatenated.

All six tools map to the **`SourceKind.text_table`** value (`tools/protocol.py`,
`_TOOL_KIND_TO_SOURCE_KIND`). They implement the standard `ResearchTool` protocol
(`definition` / `validate_arguments` / `execute`) and additionally expose a
`to_compute_callable()` so the same logic is reachable from inside the `compute` sandbox.

| Kind | Tool class | Purpose |
|------|-----------|---------|
| `table_discovery` | `TableDiscoveryTool` | List exposed tables; register DISCOVERED bindings |
| `table_search` | `TableSearchTool` | Substring (`LIKE`) search over a binding's content column |
| `table_read` | `TableReadTool` | Filter / project / order / paginate rows |
| `table_neighbors` | `TableNeighborsTool` | Sibling rows around an anchor by partition + order |
| `table_load` | `TableLoadTool` | Materialize specific row(s) into the compute namespace |
| `table_aggregate` | `TableAggregateTool` | `count`/`sum`/`avg`/`min`/`max` with optional `GROUP BY` |

All six kinds are in `DATABRICKS_BOUND_TOOL_KINDS`: they reach UC-gated data and must run under
the user identity (OBO) to behave correctly in a deployed app.

---

## The table tool family

Every tool addresses a table by its **binding name** (not its raw FQN). A binding is declared in
YAML (BOUND) or discovered at runtime (DISCOVERED). When a tool declares `config.fqn`, the
factory registers a declaration-local BOUND binding and uses it as the tool's `default_binding`,
so the agent may omit the `binding` argument. Minimal decl: `{name, kind, config}` — see the
[Configuration example](#configuration-example). Common params (`binding`, `where`, `columns`,
`limit`/`offset`, and a `roles` override for DISCOVERED bindings) recur across tools.

### `table_discovery`

Lists tables exposed to the agent and registers each as a DISCOVERED binding for later `table_*`
calls. Needs a `TableDiscoveryProvider`; with none wired, every call returns a
`discovery_unavailable` error result (no exception).

| Parameter | Type | Description |
|-----------|------|-------------|
| `name_pattern` | string | Optional case-insensitive substring filter on table names |
| `detail` | enum | `basic` (name + description), `schema` (+ column types), `full` (+ one redacted sample row) |

`detail` of `schema`/`full` needs a wired `SchemaCache`; `full` also needs a `sql_executor`.
Sampled values are PII-redacted (email / SSN / phone) and truncated.

### `table_search`

Runs a parameterized, case-insensitive `LIKE` over the binding's **content column** (after any
`where` narrowing) and returns matching rows with a deterministic snippet. The `score` field is a
placeholder `1.0` per match — real BM25/vector ranking is a future enhancement.

| Parameter | Type | Description |
|-----------|------|-------------|
| `binding` | string | Registered binding name (omit if a default is configured) |
| `query` | string | Substring to match (case-insensitive, max 500 chars) |
| `where` | object | Optional `TableFilter` (see DSL below) to pre-narrow rows |
| `columns` | list[string] | Extra columns to project alongside id + content |
| `limit` / `offset` | integer | Pagination (limit clamped to the per-statement cap) |
| `roles` | object | Per-call role override for DISCOVERED bindings, e.g. `{id: ..., content: ...}` |

### `table_read`

Reads rows with an optional filter, column projection, ordering, and pagination — use it when the
agent already knows which rows it needs.

| Parameter | Type | Description |
|-----------|------|-------------|
| `binding` | string | Registered binding name |
| `where` | object | Optional `TableFilter` |
| `columns` | list[string] | Column projection (`None` → `SELECT *`) |
| `order_by` | list[string] | Sort keys; prefix a name with `-` for `DESC` |
| `limit` / `offset` | integer | Pagination (default limit 50, clamped to the cap) |

### `table_neighbors`

For an anchor row (by id), returns sibling rows in the **same partition** whose **order** value
falls in `[order - before, order + after]`. The binding must define `id`, `partition`, and
`order` roles, and `order` must be integer-coercible (else `neighbor_config_missing`).

| Parameter | Type | Description |
|-----------|------|-------------|
| `binding` | string | Registered binding name |
| `id` | scalar | Anchor row id |
| `before` / `after` | integer | Window size each side (default 1) |
| `roles` | object | Optional role override (must include `partition` + `order`) |

### `table_load`

Materializes one or more rows by id into the compute namespace as `Table` objects (see the Table
API below), and always returns the rows as JSON. With `as_var` the row is injected under that
name; otherwise it is exposed as `last_table` and appended to `tables`. Compute injection is
optional — with no namespace wired, the tool just returns JSON.

| Parameter | Type | Description |
|-----------|------|-------------|
| `binding` | string | Registered binding name |
| `id` | string \| list[string] | One or more row ids |
| `as_var` | string | Optional variable name (must be a valid Python identifier) |
| `columns` | list[string] | Column projection |

### `table_aggregate`

Computes `count` / `sum` / `avg` / `min` / `max` with an optional `WHERE` and `GROUP BY`. Any
non-`count` op requires a `column`; the numeric ops (`sum`/`avg`/`min`/`max`) require that column
to appear in the binding's `numeric_columns` allowlist. `HAVING` is accepted by the schema but
not yet implemented (returns `invalid_filter`).

| Parameter | Type | Description |
|-----------|------|-------------|
| `binding` | string | Registered binding name |
| `op` | enum | `count`, `sum`, `avg`, `min`, `max` |
| `column` | string | Required for `op != count`; numeric ops require membership in `numeric_columns` |
| `group_by` | list[string] | Optional grouping columns |
| `where` | object | Optional `TableFilter` |
| `limit` | integer | Row cap (default 100) |

---

## How it works under the hood — the `text_table` package

Implementation lives in
`tools/builtins/text_table/`. The public surface is re-exported from its `__init__.py`.

### Table API (`table_api.py`)

`Table` is a typed read surface wrapping the parsed `table_json` dict (headers + rows) of a
single cell-grid table. It is what `table_load` injects into the compute namespace. Beyond
dict-compatible access (`table["rows"]`, `.get()`), it offers typed extractors: `cell(row_label,
column, as_float=...)`, `row_dict(...)`, `series(column)`, `column_values(column)`,
`find_rows`/`find_columns`, and `to_dataframe()`. Row/column lookups use exact match with a fuzzy
fallback (difflib cutoff 0.6). The module-level `to_float()` normalizes messy cell strings
(thousands commas, parenthetical negatives, footnote markers, "n/a" sentinels) to `float`/`nan`.

### TableFilter DSL (`filter_dsl.py`)

The `where` parameter is a small recursive predicate DSL compiled to parameterized SQL by
`compile_filter()`. Leaf shape is `FlatTableFilter` with named operators:

- `eq`, `ne`, `gt`, `gte`, `lt`, `lte` — each `{column: value}`
- `is_null`, `is_not_null`, `in_columns` — lists of column names

Leaves combine with composite operators `{and: [...]}`, `{or: [...]}`, `{not: {...}}`. A bare
`{column: value}` mapping is accepted and coerced to `eq` (`coerce_flat_filter_shape`). Every
value is emitted as a bound parameter (`:p_<col>_<n>`); column names come from the schema, not
from filter values. DoS guards: nesting depth ≤ **8**, total leaves ≤ **64**. (Empty `and` → `TRUE`,
empty `or` → `FALSE` by monoid convention.) Compilation to the final `SELECT` (FQN quoting, column
validation against the schema, `ORDER BY`, `LIMIT`/`OFFSET`, and the `LIKE` text-search predicate)
is handled by `compile_select()` in `sql_compiler.py`.

### Budget3D + per-statement caps (`budgets.py`)

Two layers protect cost, row volume, and output size.

**Per-statement hard caps** — always enforced, never user-tunable:

| Constant | Value | Guards |
|----------|-------|--------|
| `PER_STMT_LIMIT_ROWS` | 5,000 | `LIMIT` is hard-clamped to this |
| `PER_STMT_LIMIT_BYTES` | 8 MiB | result payload size |
| `PER_STMT_LIMIT_GROUPS` | 1,000 | `GROUP BY` cardinality (overflow → `group_cardinality_exceeded`) |

**`Budget3D`** — a per-compute-turn allowance across three dimensions: `max_calls` (default 30),
`max_rows` (default 50,000), and `max_wall_clock_s` (default 30.0). Each tool call `tick()`s the
budget; exceeding any dimension raises `BudgetExceeded` (a `ToolErrorException` with
`ErrorCode.BUDGET_EXCEEDED`). The three dimensions exist because a single statement can be cheap
yet an agent can still rack up cost by issuing many of them or scanning huge row counts.

### Two-tier schema cache (`schema_cache.py`)

`SchemaCache` avoids re-`DESCRIBE`-ing tables. **Tier 1** is a per-step dict cleared on every
`begin_step()` / `end_step()` boundary, so one agent step never re-fetches a schema. **Tier 2** is
a process-wide LRU (default size 256, TTL 600 s) keyed by `(fqn, sha256(token)[:16])` — the
plaintext OBO token is never held in memory and entries expire under load. `Schema` /
`SchemaColumn` are frozen dataclasses with an immutable `column_map`.

> **Not thread-safe.** The cache is designed for single-threaded asyncio contexts only;
> concurrent calls from threads can corrupt the `OrderedDict` LRU state.

### Errors (`error_codes.py`)

`ErrorCode` is a `StrEnum` (`discovery_unavailable`, `invalid_column`, `invalid_filter`,
`budget_exceeded`, `group_cardinality_exceeded`, `neighbor_config_missing`, `invalid_binding`,
`inference_failed`, `schema_fetch_failed`, …). `ToolError` is a **frozen** dataclass
(`error_code`, `message`, optional `binding`/`hint`, and a `details` mapping frozen via
`MappingProxyType`) with `to_dict()`; `ToolErrorException` wraps one for raising. Each tool's
async `execute()` catches `ToolErrorException` and returns a failed `ToolResult` with the
serialized error; the in-compute callables raise directly instead.

### Bindings / role inference (`binding.py`, `role_inference.py`, `registry.py`)

A `BindingInfo` ties a binding `name` to a 3-part `fqn`, a `BindingSource` (`BOUND` or
`DISCOVERED`), an optional `RoleMap`, and `numeric_columns`. `RoleMap` names the semantic columns:
required `id_column` + `content_column`, and optional `order_column`, `partition_column`,
`label_column`, `type_column`, `date_column`.

`TableBindingRegistry` keeps them: `register_bound` rejects duplicate BOUND names
(`invalid_binding`); `register_discovered` lets a BOUND entry win a name collision and namespaces
the loser as `discovered.<name>` with a `duplicate_binding` warning.

DISCOVERED bindings start role-less. On first use, `infer_roles()` scores every column against
each role from a sample (name-pattern + type + null-rate + distinctness features), and picks the
top scorer above a threshold. Required roles that clear no column raise `inference_failed` with
the top-3 candidates; optional roles fall back to `None`. A caller can always override with an
explicit `roles={...}` argument, which is validated and cached back onto the binding.

---

## SQL warehouse wiring

The `table_*` tools are framework code, but they execute against a **Databricks SQL warehouse**.
The shared adapter is `StatementExecutionTableSQL` (`runtime_wiring.py`), a sync callable that
runs each compiled `SELECT` through `workspace_client.statement_execution.execute_statement(...)`,
polls to completion (cancelling on timeout, default 30 s), and returns rows as dicts.

`wire_statement_execution_text_table_context(ctx, warehouse_id=...)` populates the four
`ToolFactoryContext` text-table fields in one place: a fresh `TableBindingRegistry` (per workflow
run, so bound/discovered names never leak across runs), the `sql_executor`, a `SchemaCache` whose
fetcher runs `DESCRIBE TABLE`, and an optional `table_discovery_provider`. The warehouse id comes
from the explicit argument or the `TABLE_TOOLS_WAREHOUSE_ID` / `STORAGE_WAREHOUSE_ID` env vars; if
none resolves, table tools fail strict resolution with a clear warning.

**Identity (OBO).** Auth is baked into the `WorkspaceClient` at construction. The host decides
OBO-vs-service-principal per request (via `build_databricks_workflow_runner` /
`resolve_workspace_client`) and bakes the chosen client into the executor, so the per-call
`user_token` threaded through the executor is an intentional no-op. The token still keys the
schema cache so cached schemas are never shared across identities. Because these kinds are in
`DATABRICKS_BOUND_TOOL_KINDS`, a deployed app should run them under the end user's identity — the
service principal typically lacks the UC grants and would silently return permission errors or
empty results.

---

## Configuration example

A `table_search` tool bound to a Delta table and a warehouse. The `fqn` registers a BOUND binding
(named via `binding`, else the tool's `name`); `roles` map semantic columns; `numeric_columns`
allowlists aggregate targets. Inference fills any role you omit on first use.

```yaml
# Warehouse id is supplied to the host wiring (env or explicit):
#   export TABLE_TOOLS_WAREHOUSE_ID=0123456789abcdef

tools:
  - name: filings_search
    kind: table_search
    description: "SEC filing chunks — research corpus"
    config:
      fqn: main.research.sec_filing_chunks   # catalog.schema.table (aliases: table_name, full_name)
      binding: filings                       # binding name (aliases: binding_name, as_var)
      roles:
        id: chunk_id
        content: text
        partition: file_name
        order: chunk_index
        date: filing_date
      numeric_columns: [amount_usd]          # required for sum/avg/min/max
      columns: [chunk_id, text, file_name, chunk_index]   # default projection
```

The same `fqn` + `binding` can be reused across `table_read`, `table_neighbors`, `table_load`,
and `table_aggregate` declarations so one corpus is reachable through every access pattern. Role
keys also accept synonyms (e.g. `primary_key` for `id`, `body`/`text` for `content`,
`source_column` for `partition`).

---

## See also

- [Builtin Tools](builtin-tools.md) — the web / file tool family and the `ResearchTool` protocol.
- [Enterprise Data Sources](enterprise-data-sources.md) — Vector Search, Genie (NL→SQL), Knowledge Assistant.
- [Tool Protocol Reference](../reference/tool-protocol-reference.md) — `SourceKind`, `ToolKind`, `ToolDefinition`, `ToolResult`.
