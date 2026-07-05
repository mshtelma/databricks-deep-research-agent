/**
 * TypeScript mirror of the Python surface schema.
 *
 * Source of truth: ../src/deep_research/surface/schema.py and catalog.py
 * snake_case field names are intentional — surface JSON is stored verbatim.
 */

// ---------------------------------------------------------------------------
// JSON-Pointer subset (same regex as Python POINTER_PATTERN)
// ---------------------------------------------------------------------------

/** Pattern for valid surface JSON Pointers: one or more /segment parts. */
export const POINTER_PATTERN = /^(\/[A-Za-z0-9_]+)+$/;

/** Pattern for valid identifiers (component ids, binding actions, input keys). */
export const IDENTIFIER_PATTERN = /^[A-Za-z_][A-Za-z0-9_]*$/;

/** Returns true when s matches the surface JSON-Pointer subset. */
export function isValidPointer(s: string): boolean {
  return POINTER_PATTERN.test(s);
}

// ---------------------------------------------------------------------------
// PathRef
// ---------------------------------------------------------------------------

/** A JSON-Pointer reference into the surface data model. */
export interface PathRef {
  path: string;
}

/** Type guard: value is a PathRef `{path: string}` object. */
export function isPathRef(v: unknown): v is PathRef {
  return (
    typeof v === 'object' &&
    v !== null &&
    !Array.isArray(v) &&
    typeof (v as Record<string, unknown>)['path'] === 'string'
  );
}

// ---------------------------------------------------------------------------
// DynamicValue
// ---------------------------------------------------------------------------

/** A dynamic value is either a literal scalar, a data-model reference, or null. */
export type DynamicValue = PathRef | boolean | number | string | null;

// ---------------------------------------------------------------------------
// RunReference — produced at runtime when a binding fires
// ---------------------------------------------------------------------------

/** One evidence source; a slot item's `source_refs` index into these by `ref`. */
export interface SurfaceSourceRef {
  ref: string;
  url: string;
  title?: string | null;
}

/** Per-slot fill status carried alongside the run's structured data. */
export interface SurfaceSlotMeta {
  status: string;
  error?: string;
}

/** Run state reference written into the data model by the runtime after a binding fires. */
export interface RunReference {
  status: 'running' | 'completed' | 'failed' | 'cancelled';
  session_id?: string;
  message_id?: string;
  /**
   * Structured-output payload for this run's binding, keyed by slot name
   * (components bind `<output.target>/data/<slot>`). Enriched by the host
   * (ChatPage / Preview) from the persisted message's structuredOutput —
   * results-by-reference, never persisted in surface_state.
   */
  data?: Record<string, unknown>;
  /**
   * Evidence legend for this run: slot items carry `source_refs` (index
   * strings) that chips resolve to a URL/title via this list. Enriched
   * alongside `data`; never persisted in surface_state.
   */
  sources?: SurfaceSourceRef[];
  /**
   * Per-slot status (pending/ok/empty/failed) so a component can render a
   * skeleton, a retry affordance, or its data. Enriched alongside `data`.
   */
  slotsMeta?: Record<string, SurfaceSlotMeta>;
  /**
   * True while a completed message is expected to receive a structured-output
   * envelope, but polling has not observed settled slot data yet.
   */
  pendingStructuredOutput?: boolean;
}

// ---------------------------------------------------------------------------
// ActionBinding
// ---------------------------------------------------------------------------

/** Per-run options a binding may set. */
export interface RunOptions {
  research_depth?: DynamicValue;
  verify_sources?: DynamicValue;
  query_mode?: DynamicValue;
  source_scope?: DynamicValue;
  enable_plan_review?: DynamicValue;
  turn_intent?: DynamicValue;
  tone?: DynamicValue;
  output_language?: DynamicValue;
  enable_cross_session_memory?: DynamicValue;
  allow_live_search?: DynamicValue;
}

/** Where the run result lands in the data model. */
export interface OutputTarget {
  target: string;
  mode: 'report';
}

/** Maps a UI action (Button press) to an agent run. */
export interface ActionBinding {
  action: string;
  kind: 'run_agent';
  inputs: Record<string, DynamicValue>;
  options: RunOptions;
  output: OutputTarget;
  concurrency: 'replace';
}

// ---------------------------------------------------------------------------
// SurfaceComponent
// ---------------------------------------------------------------------------

/** One node of the flat component list; tree structure via id references. */
export interface SurfaceComponent {
  id: string;
  component: string;
  props: Record<string, unknown>;
  children: string[];
}

// ---------------------------------------------------------------------------
// Host runtime controls / layout metadata
// ---------------------------------------------------------------------------

export type SurfaceControlPolicy = 'show' | 'hide' | 'locked' | 'advanced';

export interface SurfaceRuntimeControls {
  effort?: SurfaceControlPolicy;
  sources?: SurfaceControlPolicy;
  verify_sources?: SurfaceControlPolicy;
  plan_review?: SurfaceControlPolicy;
  report_style?: SurfaceControlPolicy;
  cross_session_memory?: SurfaceControlPolicy;
  live_search?: SurfaceControlPolicy;
}

export type SurfaceSectionRole = 'inputs' | 'results' | 'custom';
export type SurfaceDefaultOpen =
  | 'before_first_run'
  | 'during_run'
  | 'after_run'
  | 'always'
  | 'never';

export interface SurfaceSectionLayout {
  id: string;
  title: string;
  role: SurfaceSectionRole;
  children: string[];
  default_open?: SurfaceDefaultOpen;
}

export interface SurfaceLayout {
  sections?: SurfaceSectionLayout[];
  actions?: 'inline' | 'host_bar';
}

// ---------------------------------------------------------------------------
// Surface (root type)
// ---------------------------------------------------------------------------

/** The full declarative UI carried at definition["surface"]. */
export interface Surface {
  version: 1;
  components: SurfaceComponent[];
  data_model: Record<string, unknown>;
  bindings: ActionBinding[];
  runtime_controls?: SurfaceRuntimeControls;
  layout?: SurfaceLayout;
}

// ---------------------------------------------------------------------------
// RESERVED_INPUT_KEYS
//
// Mirror of RESERVED_INPUT_KEYS in:
//   ../src/deep_research/surface/validation.py
// Keep this list in sync with the Python constant — it is the source of truth.
// ---------------------------------------------------------------------------

/**
 * Binding input keys that collide with pipeline-owned state / template variables.
 * "query" is deliberately NOT here: it is special-cased into the job's query field.
 *
 * Source of truth: deep_research.surface.validation.RESERVED_INPUT_KEYS
 */
export const RESERVED_INPUT_KEYS: ReadonlySet<string> = new Set<string>([
  'tool_catalog',
  // selector-shadowed
  'background_summary',
  'data_landscape',
  'discovered_sources',
  'plan',
  'findings',
  'observation',
  'all_observations',
  'sources_count',
  'current_step',
  'step_title',
  'claims',
  'verification_summary',
  'analysis_summary',
  'verification_details',
  // orchestrator-seeded
  'conversation_history',
  'existing_sources',
  'prior_sources_for_seed',
  'seed_prior_sources',
  // harness auto-injected
  'current_date',
  'current_iso_datetime',
  'current_timezone',
  'compute_namespace',
  'revision_block_md',
  'source_quality',
  'chat_memory_appendix',
]);
