# Agent Designer Tool UX Harmonization Plan

## Status

Revision 2 — rewritten after a code-verified adversarial review (2026-07-07).
Do not implement from this file until re-reviewed.

Verified against three concurrent lines of work (all based on `b258929`):

- **Line A** — the uncommitted "unified tool declarations" work in this checkout
  (`ToolDeclarationDialog`, declaration kinds, tool-step form, resolver changes).
- **Line B** — `../.worktrees/feat-deterministic-functions`, commit `231eb13`
  *"deterministic function tools (uc_function, python_function, registered)"*.
  Deployed to AIS 2026-07-06.
- **Line C** — `../.worktrees/feat-function-picker-ui`, commit `b0918c0`
  *"UC-function picker UI (cascade browse + signature-driven params)"* plus
  uncommitted changes (the SHOW/DESCRIBE browse fix). Deployed to AIS 2026-07-06.

### Review changelog (what changed vs. revision 1)

1. Revision 1 treated Line A as "the current implementation" and specced UC
   discovery, a UC picker, signature-driven parameter mapping, and inline-Python
   gating as future work. **Lines B and C already ship most of that** (deployed).
   The plan is now an *integration + UX* plan, not a greenfield UX plan.
2. Corrected the `uc_function` config contract: the shipped kind uses
   `config.function` (+ auto-introspected `params`, `citeable`) — not
   `config.function_name` as revision 1 assumed from Line A.
3. Replaced invented endpoints (`GET /uc/catalogs` …) with the shipped surface
   (`GET /resources?kinds=uc_catalog|uc_schema|uc_function&parent=&query=` and
   `GET /resources/uc-functions/{fqn}/signature`). Revision 1 recommended the
   dedicated-endpoints alternative; the generic-resources alternative already won.
4. Inline Python is **not** hypothetical: `python_function` (hardened sandbox
   session) is a shipped, designer-authorable kind. The "only if runtime supports
   it" gating is replaced by UX work over the real kind. `registered` covers the
   operator-curated package story.
5. Documented a landmine revision 1 missed: Line A's resolver routes
   `uc_function`/`uc_tool` declarations to the legacy external registry, where
   **nothing ever registers UC functions** — those declarations are UI-only and
   cannot execute in the app. Line B's factory path must win in the merge.
6. Global cross-catalog UC *search* (revision 1's centerpiece CUJ) is
   transport-constrained: browse uses `SHOW` statements (BROWSE privilege) that
   require per-schema calls. Search is re-scoped with an explicit fan-out budget;
   cross-catalog search moves to v2.
7. Added the missing kinds and mechanics: `python_function`, `registered`,
   `compute`, DB-backed custom tool defs, the `mcp` → `mcp_servers` normalizer
   lift, table-valued functions, and the browse ≠ run permission split.

## Problem Statement

Three parallel workstreams touched Designer tool authoring at once:

- Line A built one shared *declaration-first* UI: declare a workflow tool, bind
  it to agents, call it from tool steps.
- Line B built the *runtime*: first-class `uc_function` (OBO SQL),
  `python_function` (sandboxed inline code), `registered` (operator catalog),
  and a rich deterministic tool-node config.
- Line C built the *UC picker*: catalog→schema→function cascade browse,
  signature introspection, and a signature-driven parameter editor for tool
  nodes.

None of the three sees the others. Harmonization means two things, in order:

1. **Integrate the three lines** without losing any shipped behavior — resolve
   the kind-schema collision, the two competing tool-step editors, and the
   dead runtime paths.
2. **Fix the user journey.** Users are not trying to choose an internal `kind`,
   create an alias, then fill an abstract JSON-schema form. Their jobs are:
   - Find a real function, index, Genie space, table, built-in tool, Python
     capability, or MCP tool.
   - Add it to the workflow with the fewest possible clicks.
   - Bind it to an agent or call it from a tool step.
   - Understand what inputs it needs.
   - Understand whether it will actually run (permissions, runtime support).

The declaration model stays. The user-facing flow becomes a search-first,
target-first picker reused everywhere.

## Ground Truth (code-verified 2026-07-07)

### Line A — declaration-first authoring UI (this checkout, uncommitted)

- `frontend/src/components/agentDesigner/toolPicker/ToolDeclarationDialog.tsx`:
  Family/source dropdown → Tool kind dropdown → **Local tool name** (defaults to
  the kind string, so `uc_function` collides on second use; `declareTool`
  returns `false` on duplicates and the dialog shows an error) → `SchemaField`
  config form. `AddToolDialog.tsx` is a thin wrapper around it.
- `toolPicker/toolKindFamilies.ts`: `FAMILY_BY_KIND` maps kinds → families
  (builtin / databricks / python / mcp_external / other). **Gap:** it has no
  entries for `python_function` or `registered`; after the merge those fall
  through the layer heuristic (`layer C → 'builtin'`, `layer D → 'other'`) and
  land in the wrong family.
- `ConfigPanel.tsx`: `AddToolRequest` intents already exist —
  `{mode: 'workspace'}`, `{mode: 'bind-agent', blockPath}` (declare + auto-bind
  via `bindToolToBlock`), `{mode: 'select-tool-step', blockPath}` (declare + set
  `ref: {type: 'builtin', name}`). Post-declaration side effects live in
  `ConfigPanel.handleToolDeclared`, not in the dialog.
- `ConfigPanel.tsx` `ToolStepForm`: declared-tool `<select>`; direct-ref
  (`uc_function`/`uc_tool`/`enterprise`) editor already collapsed under
  "Advanced direct reference"; raw `input_mapping` rows (`parameter` →
  `state_key` text inputs); `output_key`; an always-visible `output_schema`
  JSON widget. The "Convert to workflow tool" button opens the declare dialog
  **without prefilling** the direct ref's FQN.
- `BindToolDialog.tsx`: checkbox list of already-declared tools for an agent.
- `agent_designer/registry.py`: `_DECLARATION_TOOL_KIND_META` adds `decorated`
  ("Python Function", `import` module:attr + description override +
  requires_confirmation), `uc_function` (**`function_name`**, bare string, no
  picker hints), `uc_tool` (`tool_name`), `enterprise` (`tool_name`).
  Existing discovery machinery: `x-widget: resource-select` +
  `x-source-kind`/`x-value-field`/`x-label-field`/`x-allow-manual`, used by
  `vector_search`, `genie`, `knowledge_assistant`, `table_*`.
- `agent_designer/ast_normalizer.py` `_lift_mcp_servers`: `kind: mcp` tool
  declarations are authoring sugar — the normalizer moves them into the
  top-level `mcp_servers` list on save (dedup by name, drop unnamed).
- Framework `tools/resolver.py`: `_EXTERNAL_DECLARATION_FIELDS` maps
  `uc_function`/`uc_tool`/`enterprise` declarations to
  `legacy.resolve(ToolRef(type, name))` **before** trying factories.
- Framework `tools/registry.py`: `ToolRef(type in {uc_function, uc_tool,
  enterprise})` resolves only from `_external`, populated exclusively by
  `register_external` — called for app enterprise tools
  (`workflow/executor.py:424`) and the Designer's own orchestration tools
  (`agent_designer/framework_tools.py:register_designer_tools`). **Nothing
  registers UC function FQNs.** Consequence: Line A's `uc_function` and
  `uc_tool` declarations parse and render but can never execute in the app.
- Framework `agents/config.py` `ToolNodeConfig` (Line A):
  `ref` / `input_mapping` / `output_key` / `output_schema` only.

### Line B — deterministic function runtime (`231eb13`, deployed)

- Framework builtins: `tools/builtins/uc_function.py` (`UCFunctionTool` — OBO
  SQL scalar invocation), `tools/builtins/python_function.py` (fixed
  design-time code in a hardened per-run sandbox session),
  `tools/code_executor.py`, `tools/factories/{builtin,decorated,registered}.py`,
  `ToolKind` additions in `tools/protocol.py`.
- `ToolNodeConfig` superset: adds `input_literals` (validator enforces
  disjointness with `input_mapping`), `output_data_key`, `bind_namespace`,
  `fail_on_error`, `enforce_output_schema`. `output_key` default `tool_result`
  is unchanged (no conflict with Line A).
- Designer schemas in `agent_designer/registry.py`:
  - `uc_function` = `{function (FQN), params (auto-discovered on save via
    uc_function_introspect.py; explicit = override), citeable (default true)}`.
  - `python_function` = `{code (x-widget: code), params, backend
    (subprocess|restricted), timeout_seconds, extra_allowed_modules
    (pandas/numpy facade), data_lib_mode, reads_namespace, bind_result,
    citeable (default false)}`.
  - `registered` = `{key}` — operator-curated catalog
    (`app.yaml tools.registered_tools`), save-time key validation, no imports
    from stored definitions.
- `semantic_validation.py` additions (FQN shape, registered-key checks),
  designer-chat authoring guidance in `designer_architect.yaml`, docs
  (`docs/deterministic-functions.md`), example
  (`examples/technical_indicators.yaml`), unit + e2e tests
  (`e2e/tests/deterministic-functions.spec.ts`).

### Line C — UC function picker UI (`b0918c0` + uncommitted, deployed)

- `agent_designer/uc_metadata.py`: single home for OBO-SQL UC metadata reads.
  The uncommitted (deployed) revision uses `SHOW CATALOGS/SCHEMAS/USER
  FUNCTIONS` + `DESCRIBE FUNCTION` — these need only **BROWSE** privilege —
  instead of `information_schema` (needs USE CATALOG, which 403'd on
  browse-only catalogs). `check_use_catalog` deliberately probes
  `information_schema` as a *run-readiness* litmus (invoking needs USE CATALOG
  + USE SCHEMA + EXECUTE). Identifier guards: `IDENT_RE`/`FQN_RE`; hyphenated
  catalogs filtered (v1 limit) so "what you can pick" == "what you can run".
- API: `GET /resources` gains `parent` ("catalog" or "catalog.schema") +
  `query` (server-side name-prefix) params and kinds
  `uc_catalog`/`uc_schema`/`uc_function`;
  `GET /resources/uc-functions/{fqn}/signature` returns
  `{function, params, scalar, warning}` — fail-soft (degrades to empty params +
  warning, never 500s the picker).
- Frontend: `UcFunctionPicker.tsx` (three dependent datalist comboboxes,
  paste-FQN back-fills all three, signature fetch → `onSignature`, emits
  `{function, params, returns_table}` — table-valued functions supported);
  `FunctionParamsEditor.tsx` (per-param **state ref vs. literal**, disjoint,
  mirroring the backend validator); `FunctionToolNodeEditor.tsx` (tool-node
  inspector: "Unity Catalog function" mode ensures a `uc_function` declaration
  exists — dedup by FQN — and points `ref`; "Existing tool" mode binds any
  declared tool); `SchemaField` gains `uc-function-picker` and `hidden`
  widgets, so the picker also renders in Add-tool and declaration editors.
- Tool-node `default_config` fixed to `{"ref": {"name": ""}}` (a bare string
  fails `ToolNodeConfig` validation); `summary_template: "tool: {{ref.name}}"`.
- Tests: `tests/unit/agent_designer/test_uc_metadata.py`,
  `test_discovery_uc_browse.py`.

### What is still genuinely missing (the net-new work)

1. **Search-first entry.** All three lines start from classification (Line A:
   family/kind dropdowns; Line C: catalog cascade). Nobody offers "type
   `pct_change`, get actionable results across declared tools, built-ins,
   resources, UC functions."
2. **One picker.** Line A's `ToolDeclarationDialog` and Line C's
   `UcFunctionPicker`/`FunctionToolNodeEditor` are separate surfaces with
   separate declaration flows.
3. **One tool-step editor.** Line A's `ToolStepForm` (declaration select + raw
   mapping) and Line C's `FunctionToolNodeEditor` (signature-driven) both edit
   the same node type.
4. **Local-name ergonomics.** Line A defaults the name to the kind string and
   hard-fails on collisions; names should derive from the target and
   auto-dedup.
5. **Python taxonomy clarity.** Three real kinds (`python_function` inline,
   `decorated` import, `registered` operator key) surface with confusing labels
   ("Python Function" = the *import* kind today).
6. **MCP progressive disclosure.** `_MCP_SERVER_SCHEMA` renders 13 flat fields.
7. **Runtime honesty in the merge.** Line A's `uc_function`/`uc_tool`
   declaration kinds and its resolver rows must not survive in a form that
   shadows Line B's working factory path.

## Design Principles

1. **Target First, Kind Second.** The primary UI object is the thing the user
   wants: `main.finance.pct_change`, `web_search`, `sales_docs_index`, inline
   Python, or a discovered MCP tool. `kind` is metadata derived from the
   selected target.
2. **Same Component, Different Intent.** One picker for: add to workflow, add
   and bind to agent, add and select for a tool step, convert a direct ref.
   The intent plumbing already exists (`AddToolRequest` in `ConfigPanel`); it
   moves into the picker so callers stop duplicating post-declaration logic.
3. **Search Before Browse — within permission and transport reality.** Search
   is the lowest-click path, but UC browse runs on `SHOW` statements with
   per-schema granularity. Search must be scoped (catalog, or catalog+schema)
   with an explicit fan-out budget; cascade browse and paste-FQN remain
   first-class.
4. **Progressive Disclosure.** Local aliases, direct refs, output-schema JSON,
   raw input mapping, sandbox knobs, MCP transport/allow/deny — all advanced.
5. **Listable ≠ Executable (Runtime Honesty).** BROWSE lets you *see* a
   function you cannot *run* (needs USE CATALOG + USE SCHEMA + EXECUTE), and a
   declaration kind may have no runtime path at all (Line A's `uc_function`).
   The UI must surface run-readiness (`check_use_catalog`, `/probe-tools`) and
   never offer an authoring path that cannot execute.
6. **Reuse Shipped Machinery.** Extend `resource-select`/`uc-function-picker`
   widgets, `/resources` discovery, `uc_metadata.py`, `FunctionParamsEditor`,
   and save-time introspection. New abstractions only where a real gap exists
   (the aggregated search list).

## Decisions

### Settled by shipped code (do not relitigate)

| Question | Decision | Where |
| --- | --- | --- |
| Inline Python in scope? | **Yes** — `python_function` (sandbox session) is shipped and designer-authorable | Line B |
| UC function config contract | `config.function` + auto-introspected `params` + `citeable` | Line B/C registry |
| UC discovery API shape | Generic `/resources` with `parent`/`query` + dedicated signature route | Line C |
| Browse transport | `SHOW`/`DESCRIBE` (BROWSE-ok), not `information_schema` | Line C `uc_metadata.py` |
| Signature-driven params on tool nodes | `FunctionParamsEditor` (state ref vs. literal, disjoint) | Line C |
| Operator package story | `registered` catalog kind (no user wheel upload) | Line B |
| MCP servers' home | Top-level `mcp_servers` (picker card is sugar; normalizer lifts) | `_lift_mcp_servers` |

### Open decisions (recommendations)

1. **`uc_tool` fate.** No discovery, no runtime registration path, redundant
   with managed MCP (`functions/{catalog}/{schema}`). → **Retire from the
   picker.** Keep YAML/direct-ref parsing for imported workflows; semantic
   validation warns it cannot execute.
2. **`decorated` exposure.** Works only when the module is importable in the
   deployed app environment; `registered` is the operator-blessed equivalent
   with save-time validation. → **Keep, advanced-only**, relabeled "Python
   import (deployed package)" with honest deployment copy. Steer users to
   `python_function` (inline) or `registered`.
3. **Python import probe endpoint.** Importing executes module top-level code.
   → **No import probe in v1.** Client-side `module:attr` syntax validation
   only; runtime resolvability via the existing `/probe-tools` flow.
4. **Direct references.** → Import compatibility only: visible, collapsed,
   convert-to-declaration (with prefill). Not a normal authoring path.
5. **Cross-catalog global function search.** → **v2.** v1 search is scoped to a
   selected catalog (or catalog+schema); paste-FQN bypasses scoping.

## Target UX

### Unified Picker

Replace the first-screen family/kind dropdowns with a search-first list.

```text
[ Search tools, functions, indexes, spaces…            ]
All | Built-in | Databricks | Python & code | MCP / External
```

Typing filters/queries; an empty query shows curated groups. Advanced filters
(collapsed): tool kind, catalog, schema.

Result groups, in order:

1. **Existing workflow tools** (frontend-local, from `ast.tools` +
   `mcp_servers`) — selecting one applies the current intent and closes.
2. **Unity Catalog functions** — appears once a catalog scope is chosen or the
   query looks like a (partial) FQN; each result: short name, FQN,
   comment (when available), param count, `scalar`/`table` badge, and a
   run-readiness hint. Includes a `Browse Unity Catalog…` row that opens the
   existing `UcFunctionPicker` cascade.
3. **Built-in tools** — from `registry.tool_kinds` (web_search, web_research,
   academic_search, file_search, compute, table_*, …) plus DB-backed custom
   tool defs (`tool_kinds_payload_with_custom`).
4. **Databricks resources** — vector indexes, Genie spaces, Knowledge
   Assistants, Delta tables via existing `/resources` kinds; selecting one
   pre-fills the corresponding kind's config (`index_name`, `space_id`, …).
5. **Python & code** — `python_function` ("Inline Python function" → code
   editor step), `registered` (searchable operator catalog keys), `decorated`
   (advanced).
6. **MCP / External** — "Add MCP server" (authored as an `mcp` card; the save
   normalizer lifts it to `mcp_servers`), `enterprise` (advanced; runtime
   registration required).

### UC Function Fast Path (primary CUJ)

1. User clicks `Add tool` (from agent Tools tab or a tool step).
2. Types `pct_change` (optionally after picking catalog scope) **or pastes**
   `main.metrics.pct_change`.
3. Clicks the result. The app creates:

   ```yaml
   tools:
     - name: pct_change
       kind: uc_function
       config:
         function: main.metrics.pct_change
         # params auto-filled by save-time introspection; live signature
         # fetch pre-fills them for immediate param mapping
   ```

4. Depending on intent, the app also binds the tool to the agent, or points the
   tool step's `ref` at it (`ref: {name: pct_change}`) and renders
   signature-driven inputs.

No local-name prompt in the default path; no kind selection anywhere.

### UC Function Browse Path

Already shipped (`UcFunctionPicker`): searchable catalog → schema → function
datalist cascade, paste-FQN back-fill, disabled children until parent chosen,
fail-soft signature fetch, manual FQN entry as fallback. The picker embeds
unchanged inside the unified picker's Databricks group.

### Python UX (three real modes, all shipped at runtime)

| UI label | Kind | Default fields | Advanced fields |
| --- | --- | --- | --- |
| Inline Python function | `python_function` | Code editor, Parameters (`FunctionParamsEditor`-style) | backend, timeout_seconds, extra_allowed_modules, data_lib_mode, reads_namespace, bind_result, citeable |
| Registered Python tool | `registered` | Catalog key (searchable from `app.yaml tools.registered_tools`) | — |
| Python import (deployed package) — *advanced* | `decorated` | Import path `module:attr` | Description override, requires_confirmation |

Helper copy for `decorated`:

```text
The function must already be importable in the deployed app runtime
(a package or wheel shipped with the app). To author code in the
workflow itself, use an Inline Python function instead.
```

Validation: client-side `module:attr` syntax check; no server import probe
(see Decisions #3). Inline code keeps the sandbox defaults
(`backend: subprocess`, facade data libs) unless the user opens Advanced.

### Tool Step UX

One inspector (merge `ToolStepForm` + `FunctionToolNodeEditor`):

```text
Tool
[ search/select declared tool  | Add tool… ]

Inputs                      (from declaration params where known)
current   ( state ref ▾ )  [ current_value ]
previous  ( literal   ▾ )  [ 100.0 ]

Output
Save result as: [ pct_change_result ]
[ ] Fail step on tool error        (fail_on_error)
```

- Known signature (uc_function params, python_function declared params) →
  signature rows via `FunctionParamsEditor`; required params first; missing
  required mappings flagged.
- Unknown signature (most built-ins today) → keep raw mapping rows.
- Advanced (collapsed): raw `input_mapping`/`input_literals` editors,
  `output_data_key`, `bind_namespace`, `enforce_output_schema` +
  `output_schema` JSON, and the direct-reference compatibility section.
- "Convert to workflow tool" **prefills** the picker from the direct ref
  (kind + FQN/name) instead of opening blank.
- Canonical persistence stays `ToolNodeConfig` (Line B superset). No schema
  changes.

### Agent Tool Binding UX

Agent Tools tab shows two sections — **Enabled for this agent** (with remove)
and **Available workflow tools** (searchable) — plus one primary action
`Add or find tool` opening the unified picker with the `bind-agent` intent
(declare + bind in one step; this intent already exists). `BindToolDialog`'s
bulk checkbox apply remains for multi-select of existing tools.

### Workspace Tool Registry UX

`ToolsPanel` (no-selection inspector view) remains the advanced registry:
name, kind, target summary, probe status (`/probe-tools`), used-by (agents /
tool steps referencing the name), edit/remove. Never a required starting point.

## Information Architecture

### Internal kind mapping (corrected)

| User target | Kind | Primary config | Runtime path |
| --- | --- | --- | --- |
| Web Search / Web Research | `web_search` / `web_research` | provider fields, result caps | builtin factory |
| Vector Search index | `vector_search` | `index_name` | builtin factory |
| Genie space | `genie` | `space_id` | builtin factory |
| Knowledge Assistant | `knowledge_assistant` | `endpoint_name` | builtin factory |
| UC function | `uc_function` | **`function`**, `params` (auto), `citeable` | `UCFunctionTool` (OBO SQL) via factory |
| Inline Python function | `python_function` | `code`, `params`, sandbox knobs | sandbox session via factory |
| Registered Python tool | `registered` | `key` | operator catalog factory |
| Python import (advanced) | `decorated` | `import` | in-process import factory |
| MCP server | `mcp` (lifted to `mcp_servers` on save) | server config | MCPToolset per request |
| Runtime enterprise tool (advanced) | `enterprise` | `tool_name` | legacy `register_external` |
| ~~UC tool~~ | `uc_tool` | — | **retired from picker** (parse-only) |

`toolKindFamilies.ts` must add explicit `python_function`, `registered`,
`uc_function` (already present), and `compute*` entries so nothing falls
through the layer heuristic into the wrong family.

### Picker result model

Do not invent a parallel taxonomy. A slim frontend union over existing types:

```ts
type ToolTarget =
  | { source: 'workflow'; decl: ToolDecl }
  | { source: 'kind'; spec: ToolKindSpec }                 // built-ins, python_function, registered, mcp, …
  | { source: 'resource'; kind: string; resource: ResourceInfo }  // vector_index, genie_space, …
  | { source: 'uc_function'; fqn: string; signature?: UcFunctionSignatureInfo };
```

plus one pure helper `targetToToolDecl(target, existing): ToolDecl` that owns
kind mapping, config seeding, and name generation. Signature shape reuses
`FunctionSignatureResponse` / `UcFunctionParam` — no new backend model.

## Backend Plan

### 0. Land the branches (prerequisite, ordered)

1. Merge Line B (`231eb13`).
2. Commit Line C's worktree changes (the SHOW/DESCRIBE browse fix is deployed
   but **uncommitted**) and merge (`b0918c0` + fix).
3. Rebase Line A's uncommitted work on top; resolve per §1–2 below.

### 1. Resolver reconciliation (framework)

`tools/resolver.py` `_EXTERNAL_DECLARATION_FIELDS` currently short-circuits
`uc_function`/`uc_tool`/`enterprise` to the legacy external registry **before**
the factory loop. Post-merge that would shadow Line B's working `uc_function`
factory with a dead path.

- Drop `uc_function` and `uc_tool` rows; keep `enterprise` (its only real path
  is `register_external`).
- Verify Line B's executor tool-node path for direct
  `ref: {type: uc_function, name: fqn}` refs during integration; if it does not
  synthesize a runtime tool, semantic validation must flag such refs as
  import-compat-only (convert to declaration to run).
- Tests: extend `databricks-deep-research/tests/test_tool_declarations.py` —
  a `uc_function` declaration resolves via the factory chain (not legacy), and
  an `enterprise` declaration still requires the external registry.

### 2. Registry/kind reconciliation (app)

- One `uc_function` designer entry: Line B/C schema (`function` +
  `x-widget: uc-function-picker`, hidden auto-`params`, `citeable`). Delete
  Line A's `_UC_FUNCTION_TOOL_SCHEMA` (`function_name`) and the `uc_tool`
  picker entry.
- Keep Line A's `decorated`/`enterprise` declaration entries (relabeled;
  `x-advanced` where supported by the picker grouping).
- Migration shim: `ast_normalizer` rewrites any persisted
  `uc_function.config.function_name` → `function` (Line A shipped nowhere, but
  drafts may exist in Lakebase).

### 3. Scoped UC function search

Extend the shipped browse rather than adding new endpoints:

- `GET /resources?kinds=uc_function&parent={catalog}&query={prefix}` (schema
  omitted) — catalog-scoped search:
  - If `check_use_catalog` passes → one `information_schema.routines` query.
  - Else → `SHOW SCHEMAS` + per-schema `SHOW USER FUNCTIONS` fan-out with hard
    budget: ≤ 24 statements, concurrency 4, ~4s wall budget; on truncation
    return partial results + `warning: "narrow to a schema"`.
- Short-TTL (60s) per-`(user, scope, prefix)` cache in `uc_metadata.py`.
- Paste-FQN and `catalog.schema`-scoped prefix search already work; keep.
- Cross-catalog search: v2 (would multiply the fan-out by catalog count).

### 4. Validation additions (`semantic_validation.py`)

Building on Line B's checks (FQN shape, registered keys):

- Tool step whose declaration has known `params`: error when a required param
  is in neither `input_mapping` nor `input_literals`.
- Declared kind with no runtime path (`uc_tool`, unregistered `enterprise`
  names when detectable): warning with remediation copy.
- `decorated` import path failing `module:attr` syntax: error.

### 5. Explicitly not doing (v1)

- No `GET /uc/catalogs|schemas|functions` dedicated endpoints (shipped
  `/resources` shape wins).
- No Python import probe endpoint.
- No backend aggregated `tool-targets` endpoint — the frontend composes
  registry + resources + UC search hooks; revisit only if ranking/pagination
  outgrows the client.

## Frontend Plan

### 1. `ToolTargetPicker` (evolve `ToolDeclarationDialog`)

- Search input (autofocus) + family filter chips + grouped results; family/kind
  dropdowns are demoted to an "All kinds" browse group at the list bottom.
- Owns the intent: absorb `ConfigPanel.handleToolDeclared` so
  `declare` / `bind-agent` / `select-tool-step` behavior lives in one place
  (`AddToolRequest` type moves next to the picker). `BindToolDialog` and
  `ToolsPanel` call sites updated.
- Selecting a config-complete target (UC function via signature fetch, resource
  with pre-filled id, zero-required-field kinds) applies immediately; targets
  needing input (inline Python code, MCP server, import path) open a second
  "configure" step inside the same dialog with only default-tier fields.

### 2. Name generation

```ts
function suggestedToolName(target: ToolTarget): string  // FQN → last segment; module:attr → attr; kind → kind
function uniqueToolName(base: string, existing: ToolDecl[]): string  // pct_change, pct_change_2, …
```

Local name moves under Advanced (pre-filled). `declareTool`'s false-on-dup
contract is unchanged — the picker passes a pre-deduped name. Line C's
dedup-by-FQN behavior is preserved: an existing declaration for the same target
is *selected*, not re-declared.

### 3. Unified tool-step inspector

Merge `FunctionToolNodeEditor` (base: kind-aware, signature-driven,
`FunctionParamsEditor`, fail_on_error/enforce_output_schema) with
`ToolStepForm`'s remaining pieces (declared-tool select with unresolved-ref
warning, advanced direct-ref section with prefit "Convert to workflow tool",
advanced raw-mapping and `output_schema` editors). One component; delete the
other. `output_schema` becomes advanced-only.

### 4. Family map + labels

- `toolKindFamilies.ts`: add `python_function: 'python'`,
  `registered: 'python'`, `compute: 'python'`, `compute_namespace: 'python'`.
- Registry labels: `decorated` → "Python import (deployed package)";
  `python_function` → "Inline Python function"; `registered` → "Registered
  Python tool".

### 5. MCP basic/advanced split

Default fields: `name`, `client_kind`, `url` **or** `connection_name` /
`managed_target` (conditional on client_kind), `auth_type`, `secret_ref`.
Advanced: `transport`, `api_key_header`, `allow`, `deny`, `name_prefix`,
`strategy`, `citeable`. Requires either `x-advanced` support in the tool-config
`SchemaField` path (it already exists for agent schemas) or a grouped wrapper.
Keep the "authored as a card, saved into `mcp_servers`" behavior; the picker's
Existing-workflow-tools group must therefore also list `mcp_servers` entries.

## Click-Count Goals

| Journey | Current (verified) | Target |
| --- | --- | --- |
| Add UC function to agent | Add tool → family → kind → fix name → type full FQN blind → Add (auto-binds) — 6 actions, FQN memorized | Add or find tool → type/pick function → done (3) |
| Add UC function to tool step | same 6 via `select-tool-step`, then raw param rows by hand | Choose tool → type/pick → confirm suggested mappings (3) |
| Add existing workflow tool to agent | BindToolDialog checkbox → Apply (2–3) ✓ keep | same |
| Inline Python function | not offered (Line A) / kind-first (post-merge) | Add or find tool → Inline Python → paste code + params → Add (4) |

## Phased Implementation

### Phase 0 — Integration (prerequisite; mostly merge work)

Land B, then C (incl. uncommitted fix), rebase A; apply resolver + registry
reconciliation (§Backend 1–2); fix `toolKindFamilies.ts`; keep all three
lines' tests green, including `e2e/tests/deterministic-functions.spec.ts`.

Acceptance: a `uc_function` declaration authored in the Designer **executes**
via `UCFunctionTool` in a dev-workspace run; `uc_tool` no longer offered;
no remaining `function_name` producers.

### Phase 1 — Unified search-first picker

`ToolTargetPicker` + `targetToToolDecl` + name generation + intent absorption;
catalog-scoped UC search endpoint work (§Backend 3).

Acceptance: same picker opens from workspace tools, agent Tools tab, and tool
step; typing `pct` with a catalog scope surfaces UC functions; paste-FQN works
scope-free; every declared name is auto-generated and collision-free;
`ConfigPanel.test.tsx` / `ToolsPanel.test.tsx` / `agentEditorStore.test.ts`
updated.

### Phase 2 — Tool-step unification

Single tool-step inspector; signature rows for uc_function/python_function
declarations; required-param validation (§Backend 4); direct-ref convert
prefill; `output_schema`/raw-mapping demoted to advanced.

Acceptance: UC function with two required params renders two mapped rows;
reopening preserves mappings; missing required mapping is visibly flagged in
inspector + validation.

### Phase 3 — Python & MCP clarity

Labels/copy per §Frontend 4; python_function config step in the picker; MCP
basic/advanced split; `decorated` moved to advanced with deployment copy +
syntax validation.

Acceptance: no path lets a user believe `decorated` accepts pasted code; MCP
default view shows ≤ 6 fields; inline Python authorable end-to-end from the
picker.

### Phase 4 — Registry polish

ToolsPanel used-by column, probe status chip, edit-in-picker; run-readiness
hint (browse-only warning) on UC results using `check_use_catalog`.

## Test Plan

Unit (frontend): `targetToToolDecl` per source; `uniqueToolName` collisions;
picker applies each intent; python import syntax accept/reject; unified
tool-step inspector renders signature params and preserves unknown-signature
raw mapping; direct-ref convert prefill.

Unit (backend): extend `test_uc_metadata.py` (scoped search: information_schema
path, SHOW fan-out path, budget truncation + warning, cache);
`test_discovery_uc_browse.py` (query param); resolver reconciliation tests
(framework); semantic-validation required-param and dead-kind cases; registry
payload: single uc_function entry with picker widgets, relabeled python kinds.

Integration: declare-uc-function → bind → run (framework executor, mocked SQL);
save/load round-trip preserves declarations, `mcp_servers` lift, tool-node
config superset fields.

E2E (extend `deterministic-functions.spec.ts` + designer specs): search-add UC
function to agent; browse-add to tool step with param mapping; inline Python
CUJ; discovery-failure manual FQN fallback; imported direct-ref convert.

Accessibility: search input focused on open; grouped results keyboard
navigable; filter chips are buttons; comboboxes labeled; loading/error states
announced.

## Data and State Model (canonical, post-merge)

```yaml
tools:
  - name: pct_change
    kind: uc_function
    config:
      function: main.metrics.pct_change     # NOT function_name
      # params: auto-filled on save (uc_function_introspect); explicit = override
      citeable: true

  - name: normalize_scores
    kind: python_function
    config:
      code: |
        result = [round(x / max(values), 3) for x in values]
      params:
        - {name: values, type: array, required: true}

nodes:
  - id: compute_pct
    type: tool
    config:
      ref: { name: pct_change }             # declared-tool path (type defaults builtin)
      input_mapping: { current: latest_close }
      input_literals: { previous: 100.0 }   # disjoint from input_mapping (validated)
      output_key: pct_change_result
      fail_on_error: true
```

Agent references stay local names (`config.tools: [pct_change]`). MCP cards are
lifted to `mcp_servers` on save. Direct refs
(`ref: {type: uc_function, name: main.metrics.pct_change}`) stay parseable for
imported YAML but are flagged by validation and steered to conversion.

## UX Copy Guidelines

Primary labels: `Unity Catalog function`, `Inline Python function`,
`Registered Python tool`, `Python import (deployed package)` (advanced),
`Existing workflow tool`, `Add and enable`, `Add and call`.

Avoid in primary UI: `decorated`, `builtin` (as a ref type), `ToolKind`,
`Direct reference`, `Local tool name` (advanced only), `kind:` values as labels.

Run-readiness copy (UC result the user can browse but not run):

```text
Visible via BROWSE. Running it requires USE CATALOG, USE SCHEMA and
EXECUTE on main.metrics — the run will fail without them.
```

## Risks

1. **Three-way merge regressions** (highest). Line A's resolver rows shadow
   Line B's factory; two tool-step editors race to own the same config; kind
   schema collision (`function_name` vs `function`). Mitigation: Phase 0 is
   its own gated deliverable with all three suites green + one live
   dev-workspace uc_function run.
2. **UC search fan-out cost.** `SHOW`-based search is per-schema. Mitigation:
   scope requirement, statement budget, partial-result warnings, short-TTL
   cache; cross-catalog deferred.
3. **Browse ≠ run surprises.** Users pick functions they cannot execute.
   Mitigation: `check_use_catalog` hint at pick time; `/probe-tools` surfacing
   in ToolsPanel; runtime error already actionable (OBO SQL error text).
4. **Sandbox knob misuse.** `python_function` advanced knobs
   (`backend: restricted`, `data_lib_mode: live`) are operator-gated; the
   picker must not present them as ordinary options. Mitigation: advanced tier
   + operator-switch copy.
5. **Signature metadata gaps.** TVFs, overloads, comment-less params.
   Mitigation: fail-soft signature route already degrades to manual mapping;
   keep raw-mapping advanced editor.
6. **Picker over-smartness.** Mitigation: slim `ToolTarget` union, manual
   FQN/import fallbacks everywhere, cascade browse preserved.

## Non-Goals (v1)

- Cross-catalog global UC function search.
- User-managed wheel/dependency upload (operator `registered` catalog only).
- Python import probe endpoint.
- Full MCP marketplace UX.
- Semantic (beyond exact/prefix) state-to-parameter matching.
- Removing direct-ref YAML compatibility.

## Definition of Done

- Phase 0: one `uc_function` kind (Line B contract) that **executes** from a
  Designer-authored workflow; resolver has no dead declaration routes;
  `uc_tool` retired from authoring; all pre-existing suites green.
- One picker serves workspace tools, agent binding, and tool steps, applying
  the launch intent in one place.
- UC functions: searchable within a catalog scope, browsable by cascade,
  paste-FQN always works, run-readiness hinted.
- Adding a UC function to an agent or tool step ≤ 3 primary actions; local
  names auto-generated, collision-free, advanced-only.
- Tool steps render signature-driven inputs for uc_function/python_function
  declarations; required-param gaps are validated.
- Python kinds are honestly labeled; `decorated` cannot be mistaken for inline
  code authoring; MCP default form ≤ 6 fields.
- Direct refs remain loadable, are flagged, and convert with prefill.
- Critical journeys covered by unit + integration + e2e tests listed above.
