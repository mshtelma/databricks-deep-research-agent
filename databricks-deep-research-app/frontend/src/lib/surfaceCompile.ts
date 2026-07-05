/**
 * Compile an ActionBinding + current data model into a concrete job submission.
 *
 * This is pure/side-effect-free: it reads the binding and data model, performs
 * pointer resolution and template substitution, and returns a CompiledSubmission
 * ready to hand off to the job API.
 */

import type { ActionBinding, DynamicValue, Surface, SurfaceComponent } from '@/types/surface';
import { RESERVED_INPUT_KEYS, IDENTIFIER_PATTERN, isPathRef } from '@/types/surface';
import { getAtPointer, resolveDynamic } from '@/lib/surfaceState';
import {
  buildQuerySubmission,
  mergeRunContext,
  type RunContext,
} from '@/lib/runContext';
import type { QuerySubmission } from '@/types/querySubmission';

// ---------------------------------------------------------------------------
// CompiledSubmission
// ---------------------------------------------------------------------------

/** Concrete submission payload produced by compileBinding. */
export interface CompiledSubmission {
  query: string;
  /** Provenance of `query`: the author's bound field, a composition of filled
   *  free-text inputs, or empty (nothing to run on). Drives the run-gate + logging. */
  querySource?: QuerySource;
  surfaceInputs: Record<string, string | number | boolean>;
  researchDepth?: string;
  verifySources?: boolean;
}

export interface CompiledSurfaceSubmission extends CompiledSubmission {
  submission: QuerySubmission;
  binding: ActionBinding;
  usedPointers: string[];
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/** `{/json/pointer}` placeholder pattern — only brace groups starting with `/`. */
const TEMPLATE_POINTER_RE = /\{(\/[^}]*)\}/g;

/**
 * Substitute every `{/json/pointer}` in a string template.
 * Missing pointers resolve to empty string.
 */
function substituteTemplate(template: string, data: Record<string, unknown>): string {
  return template.replace(TEMPLATE_POINTER_RE, (_match, pointer: string) => {
    const resolved = getAtPointer(data, pointer);
    return resolved !== undefined && resolved !== null ? String(resolved) : '';
  });
}

/** Resolve a DynamicValue to a query string (PathRef or literal string + template). */
function resolveQuery(value: DynamicValue, data: Record<string, unknown>): string {
  if (isPathRef(value)) {
    const resolved = getAtPointer(data, value.path);
    return resolved !== undefined && resolved !== null ? String(resolved) : '';
  }
  if (typeof value === 'string') {
    // Tolerate a bare-string pointer authored without `{path}` (e.g. "/inputs/topic"):
    // if it is a pure pointer that resolves in the data model, use the resolved value
    // rather than the literal string. A `{/ptr}` template is handled below.
    if (value.startsWith('/') && !value.includes('{')) {
      const resolved = getAtPointer(data, value);
      if (resolved !== undefined) {
        return resolved !== null ? String(resolved) : '';
      }
    }
    return substituteTemplate(value, data);
  }
  // non-string scalar used as query — coerce
  if (value !== null && value !== undefined) {
    return String(value);
  }
  return '';
}

/**
 * Resolve a binding-input value, tolerating a bare-string pointer.
 *
 * - PathRef → the data-model value at the pointer.
 * - A bare string that is a pure data-model pointer ('/…' that resolves) → the
 *   resolved value. Bindings are sometimes authored with bare-string pointers
 *   instead of `{path}`; without this the literal pointer would leak as a prompt
 *   var (e.g. surfaceInputs.ticker === "/inputs/ticker" instead of "AAPL").
 * - Anything else → returned as-is (literal scalar / null).
 */
function resolveBindingInput(value: DynamicValue, data: Record<string, unknown>): unknown {
  if (isPathRef(value)) {
    return getAtPointer(data, value.path);
  }
  if (typeof value === 'string' && value.startsWith('/') && !value.includes('{')) {
    const resolved = getAtPointer(data, value);
    if (resolved !== undefined) {
      return resolved;
    }
  }
  return value;
}

/** Returns true if key is a safe non-reserved identifier. */
function isSafeInputKey(key: string): boolean {
  return IDENTIFIER_PATTERN.test(key) && !RESERVED_INPUT_KEYS.has(key);
}

/** Coerce a resolved value to a scalar for surfaceInputs. Returns null when not coercible. */
function toScalar(
  key: string,
  value: unknown,
): string | number | boolean | null {
  if (typeof value === 'string' || typeof value === 'number' || typeof value === 'boolean') {
    return value;
  }
  if (value !== null && value !== undefined) {
    console.warn(
      `[surfaceCompile] input '${key}' resolved to an object/array; skipping.`,
    );
  }
  return null;
}

function resolvedString(
  value: DynamicValue | undefined,
  dataModel: Record<string, unknown>,
): string | undefined {
  if (value === undefined || value === null) return undefined;
  const resolved = resolveDynamic(value, dataModel);
  return typeof resolved === 'string' && resolved.length > 0 ? resolved : undefined;
}

function resolvedBoolean(
  value: DynamicValue | undefined,
  dataModel: Record<string, unknown>,
): boolean | undefined {
  if (value === undefined || value === null) return undefined;
  const resolved = resolveDynamic(value, dataModel);
  return typeof resolved === 'boolean' ? resolved : undefined;
}

function actionOptionsToRunContext(
  binding: ActionBinding,
  dataModel: Record<string, unknown>,
): RunContext {
  const options = binding.options ?? {};
  const out: RunContext = {};
  const researchDepth = resolvedString(options.research_depth, dataModel);
  const verifySources = resolvedBoolean(options.verify_sources, dataModel);
  const queryMode = resolvedString(options.query_mode, dataModel);
  const sourceScope = resolvedString(options.source_scope, dataModel);
  const enablePlanReview = resolvedBoolean(options.enable_plan_review, dataModel);
  const turnIntent = resolvedString(options.turn_intent, dataModel);
  const tone = resolvedString(options.tone, dataModel);
  const outputLanguage = resolvedString(options.output_language, dataModel);
  const enableCrossSessionMemory = resolvedBoolean(
    options.enable_cross_session_memory,
    dataModel,
  );
  const allowLiveSearch = resolvedBoolean(options.allow_live_search, dataModel);

  if (researchDepth) out.researchDepth = researchDepth as RunContext['researchDepth'];
  if (verifySources !== undefined) out.verifySources = verifySources;
  if (queryMode) out.queryMode = queryMode as RunContext['queryMode'];
  if (sourceScope) out.sourceScope = sourceScope as RunContext['sourceScope'];
  if (enablePlanReview !== undefined) out.enablePlanReview = enablePlanReview;
  if (turnIntent) out.turnIntent = turnIntent as RunContext['turnIntent'];
  if (tone) out.tone = tone;
  if (outputLanguage) out.outputLanguage = outputLanguage;
  if (enableCrossSessionMemory !== undefined) {
    out.enableCrossSessionMemory = enableCrossSessionMemory;
  }
  if (allowLiveSearch !== undefined) out.allowLiveSearch = allowLiveSearch;
  return out;
}

// ---------------------------------------------------------------------------
// Effective query composition (tolerant of partially-filled forms)
// ---------------------------------------------------------------------------

/** Content-bearing input components whose value can seed the research query when
 *  the binding has no non-empty `query`. Includes free-text fields AND Select —
 *  a chosen Select value is often the subject itself (e.g. a ticker picker), not a
 *  filter, so selecting it must be enough to run. Checkbox is excluded: a boolean
 *  is a filter, not query text (it still flows as a surfaceInput/prompt var). The
 *  effective query is composed from these so a run proceeds with whatever the user
 *  actually chose or typed. */
const QUERY_INPUT_COMPONENTS: ReadonlySet<string> = new Set(['TextField', 'TextArea', 'Select']);

/** Cap on the composed query — matches the backend `query` max_length (jobs.py). */
const QUERY_MAX_LENGTH = 10_000;

/** Provenance of the effective query, for the run-gate + observability. */
export type QuerySource = 'bound' | 'composed' | 'empty';

export interface EffectiveQuery {
  /** Query to submit (bound or composed); trimmed + capped to QUERY_MAX_LENGTH. */
  query: string;
  source: QuerySource;
  /** Data-model pointers that contributed (debug/observability). */
  usedPointers: string[];
}

/** The `props.value` PathRef path of a component, or null when it is not a
 *  two-way (PathRef-bound) input. */
function valuePointer(comp: SurfaceComponent): string | null {
  const raw = comp.props['value'];
  return isPathRef(raw) ? raw.path : null;
}

/** Content-bearing inputs (QUERY_INPUT_COMPONENTS with a PathRef value), doc order. */
function queryInputs(surface: Surface): Array<{ pointer: string; label?: string }> {
  const out: Array<{ pointer: string; label?: string }> = [];
  for (const comp of surface.components) {
    if (!QUERY_INPUT_COMPONENTS.has(comp.component)) continue;
    const pointer = valuePointer(comp);
    if (pointer === null) continue;
    const label =
      typeof comp.props['label'] === 'string' ? comp.props['label'] : undefined;
    out.push({ pointer, label });
  }
  return out;
}

/** Truncate at a word boundary near the cap (avoids a mid-word cut / a 422). */
function capQuery(text: string): string {
  if (text.length <= QUERY_MAX_LENGTH) return text;
  const cut = text.slice(0, QUERY_MAX_LENGTH);
  const lastSpace = cut.lastIndexOf(' ');
  return (lastSpace > QUERY_MAX_LENGTH * 0.8 ? cut.slice(0, lastSpace) : cut).trimEnd();
}

/**
 * Derive the effective research query for a binding, tolerating a partially-filled
 * form. Pure + deterministic.
 *
 * - Use the author's bound `query` input when it resolves non-empty (`'bound'`).
 * - Otherwise compose from the content inputs the user filled — free-text fields
 *   AND a chosen Select value (the subject, e.g. a ticker) — excluding the
 *   bound-query pointer when it is a PathRef (a `{/ptr}` template has no single
 *   pointer to exclude, and already resolved empty). One contributor → its value;
 *   several → newline-joined `label: value` (`'composed'`).
 * - Nothing filled → `'empty'` (the caller blocks the run gracefully).
 *
 * Checkbox values are filters, never the query (they still flow as surfaceInputs).
 */
export function deriveEffectiveQuery(
  binding: ActionBinding,
  dataModel: Record<string, unknown>,
  surface?: Surface,
): EffectiveQuery {
  const rawQuery = binding.inputs['query'] ?? null;
  const bound = resolveQuery(rawQuery as DynamicValue, dataModel).trim();
  if (bound) {
    return {
      query: capQuery(bound),
      source: 'bound',
      usedPointers: isPathRef(rawQuery) ? [rawQuery.path] : [],
    };
  }
  if (!surface) {
    return { query: '', source: 'empty', usedPointers: [] };
  }
  const excluded = isPathRef(rawQuery) ? rawQuery.path : null;
  const contributions: Array<{ pointer: string; label?: string; value: string }> = [];
  for (const { pointer, label } of queryInputs(surface)) {
    if (pointer === excluded) continue;
    const resolved = getAtPointer(dataModel, pointer);
    const value =
      resolved !== undefined && resolved !== null ? String(resolved).trim() : '';
    if (value) contributions.push({ pointer, label, value });
  }
  if (contributions.length === 0) {
    return { query: '', source: 'empty', usedPointers: [] };
  }
  const composed =
    contributions.length === 1
      ? contributions.map((c) => c.value).join('')
      : contributions
          .map((c) => (c.label ? `${c.label}: ${c.value}` : c.value))
          .join('\n');
  return {
    query: capQuery(composed),
    source: 'composed',
    usedPointers: contributions.map((c) => c.pointer),
  };
}

// ---------------------------------------------------------------------------
// compileBinding
// ---------------------------------------------------------------------------

/**
 * Compile a binding + data model into a concrete job submission.
 *
 * - binding.inputs["query"]: PathRef → resolve; string → substitute `{/ptr}`.
 * - every other key: resolve DynamicValue; skip null/undefined/objects;
 *   skip RESERVED_INPUT_KEYS and non-identifier keys.
 * - options.research_depth / options.verify_sources: only included when
 *   non-empty string / boolean respectively.
 */
export function compileBinding(
  binding: ActionBinding,
  dataModel: Record<string, unknown>,
  surface?: Surface,
): CompiledSubmission {
  // --- query (tolerant: composed from filled inputs when the bound field is empty) ---
  const effective = deriveEffectiveQuery(binding, dataModel, surface);
  const query = effective.query;

  // --- surfaceInputs -------------------------------------------------------
  const surfaceInputs: Record<string, string | number | boolean> = {};
  for (const [key, value] of Object.entries(binding.inputs)) {
    if (key === 'query') continue;
    if (!isSafeInputKey(key)) continue;
    const resolved = resolveBindingInput(value, dataModel);
    if (resolved === null || resolved === undefined) continue;
    const scalar = toScalar(key, resolved);
    if (scalar !== null) {
      surfaceInputs[key] = scalar;
    }
  }

  // --- options -------------------------------------------------------------
  const result: CompiledSubmission = {
    query,
    querySource: effective.source,
    surfaceInputs,
  };

  const options = binding.options ?? {};
  if (options.research_depth !== undefined && options.research_depth !== null) {
    const rd = resolveDynamic(options.research_depth, dataModel);
    if (typeof rd === 'string' && rd.length > 0) {
      result.researchDepth = rd;
    }
  }

  if (options.verify_sources !== undefined && options.verify_sources !== null) {
    const vs = resolveDynamic(options.verify_sources, dataModel);
    if (typeof vs === 'boolean') {
      result.verifySources = vs;
    }
  }

  return result;
}

export function compileSurfaceAction({
  surface,
  binding,
  dataModel,
  runContext,
  selectedAgentId,
}: {
  surface: Surface;
  binding: ActionBinding;
  dataModel: Record<string, unknown>;
  runContext?: RunContext;
  selectedAgentId?: string;
}): CompiledSurfaceSubmission {
  const compiled = compileBinding(binding, dataModel, surface);
  const actionOverrides = actionOptionsToRunContext(binding, dataModel);
  const mergedRunContext = mergeRunContext(
    {
      queryMode: 'deep_research',
      turnIntent: 'research',
      agentId: selectedAgentId,
    },
    runContext,
    actionOverrides,
    { agentId: selectedAgentId ?? runContext?.agentId },
  );
  const submission = buildQuerySubmission({
    message: compiled.query,
    runContext: mergedRunContext,
    surfaceInputs: compiled.surfaceInputs,
    surfaceAction: binding.action,
  });

  return {
    ...compiled,
    submission,
    binding,
    usedPointers: deriveEffectiveQuery(binding, dataModel, surface).usedPointers,
  };
}
