# Deterministic Functions in Agentic Workflows

Mix deterministic computation with agents: express fixed transformations (SMA/
technical indicators, reshaping, forecasting glue) as code, and spend LLM calls
only where models excel (planning, interpretation, synthesis).

## Three function sources × two invocation surfaces

| Kind | Runs where | Trust model |
|---|---|---|
| `uc_function` | Databricks' own runtime via the workspace **managed MCP** server (`managed_target: functions/{catalog}/{schema}`), under the caller's identity (OBO) | Unity Catalog governance; no app-side sandbox needed |
| `python_function` | The run's persistent **SandboxSession** subprocess (default) or, gated, in-process | Untrusted-capable: OS boundary (rlimits, scrubbed env, SIGKILL) |
| `registered` | In-process | Operator-curated catalog; dict lookup by `config.key` — stored workflows can never import code |

Every kind is a normal `ResearchTool`, so one declaration serves BOTH surfaces:

- **Agent tools** — bind via `config.tools: [name]`; the LLM calls them mid-ReAct.
- **DAG tool nodes** — `type: tool` steps run them deterministically (no LLM):
  `ref: {name: ...}`, `input_mapping` (state keys), `input_literals`
  (constants), `output_key` (content), `output_data_key` (structured
  `ToolResult.data` + `success`/`error` for conditional branching),
  `bind_namespace` (also inject the result into the run scratchpad),
  `fail_on_error`, `enforce_output_schema`/`output_schema` (required-keys
  check). Tool nodes resolve through the ToolResolver, so per-request MCP
  overrides (`ref: {type: mcp, name: fn}`) and declared tools both work.

`uc_function` is authoring sugar: hosts normalize it into an `mcp_servers`
entry (the app's Designer does this on save); a surviving declaration fails
workflow validation loudly.

## The sandbox session (MemEx-style run scratchpad)

`python_function` code executes inside ONE hardened subprocess per workflow
run — a persistent REPL:

- Variables persist across calls: one function's `bind_result` feeds another's
  `reads_namespace` (bridged from the in-process compute scratchpad when
  JSON-able). Agents see the combined scratchpad via `{compute_namespace}`.
- The parent keeps a JSON-able **shadow** (per-exec deltas) used for prompt
  rendering and crash rehydration. On a per-call wall-timeout the process
  group is SIGKILLed and respawned; JSON values are re-injected, live objects
  are declared lost (the result carries a note).
- Scoping matches the compute namespace: per run, and `isolate` subworkflows
  get a fresh session. The session closes at run end.
- Concurrency: commands are lock-serialized; parallel branches interleave
  call-by-call. Don't bind the same variable from parallel branches.
- Checkpoint contract: the scratchpad does **not** survive resume. The
  primitives exist (`SandboxSession.shadow()/restore_shadow()`,
  `PythonComputeTool.export_namespace_jsonable()/restore_namespace()`, JSON
  only — never pickle) for hosts that persist checkpoints; wiring them into a
  host's checkpoint store is host work.

## Security model (read this before changing defaults)

- **Subprocess profile**: `python -I -B`, `sanitize_subprocess_env` (no OBO
  token, no `DATABRICKS_*`, no cloud creds), rlimits (cumulative CPU, AS on
  Linux, FSIZE, NOFILE, NPROC — fork-bomb cap), `setsid` + group SIGKILL,
  fresh tempdir cwd, JSON-over-pipes with size caps, session-count semaphore
  (`DDR_SANDBOX_MAX_SESSIONS`), never pooled across runs/users.
- **In-child policy**: AST allowlist (stdlib-math set + opted-in `pandas`/
  `numpy`), restricted builtins, module facades (no live module objects).
  Data libs default to **facade** view (top-level API; IO/eval primitives
  removed; pickle/DB method names AST-blocked). `data_lib_mode: live` and
  `backend: restricted` require the host trust switch (in the app:
  `execution.allow_inprocess_python_function`) — in-process CPython is NOT a
  hard boundary (un-killable threads, no memory cap).
- **Network egress is not blockable** without namespaces: the allowlist is the
  network gate, and the credential scrub means any residual egress is
  unauthenticated. Platform-level egress policy is the ops backstop.
- **`decorated` is import-time code execution.** The default executor factory
  chain constructs it FAIL-CLOSED (deny-all). Hosts whose YAML is authored at
  import time may pass `DecoratedToolFactory(allowed_import_prefixes=None)`;
  hosts running stored definitions should use `registered` (catalog lookup)
  or a tight prefix allowlist.

## Data plane

Citeable results (`citeable: true` on `python_function`; default-on for
`uc_function`) are admitted through the same evidence gate as agent tool
calls and land in the `sources` pool with `function://` / `mcp://` URLs —
synthesis can cite computed artifacts. A `table_json` (`headers` + `rows`) in
`ToolResult.data` is registered into the run's TableRegistry
(`source_label=function://<name>`), addressable by `table_*` tools; tool
nodes surface the index as `table_index` in their `output_data_key` payload.

## Extras contract (which keys tools may rely on)

- Agent-surface `ToolContext.extras`: `_framework_user_id`,
  `_framework_approval_broker`, `_framework_runtime_store`, host `_framework_*`
  entries.
- Tool-node surface: the resolver's factory extras (`_resolver_cache`,
  `_framework_vfs`, `_skill_store`, `_sandbox_session`, ...) plus
  `user_token`.
- Tools must NOT assume keys from the other surface. Substrate access
  (compute namespace, sandbox session) is bound at factory **create** time via
  closures over `extras["_resolver_cache"]` / `extras["_sandbox_session"]` —
  the `table_load` idiom — precisely so both surfaces behave identically.
  `isolate` subworkflows drop both keys and therefore get fresh scratchpads.

## Design note: why no in-process MCP server for internal tools

Internal tools already implement the `ResearchTool` protocol agents consume;
wrapping them in an in-process MCP server would add a serialization hop and a
fake boundary with zero isolation gain. MCP is used where it earns its keep —
at real trust boundaries: consuming Databricks **managed** MCP (UC functions,
genie, vector search; OBO) and **external** MCP servers (UC connections; SP).

See `examples/technical_indicators.yaml` for a runnable
deterministic-plus-agentic workflow.
