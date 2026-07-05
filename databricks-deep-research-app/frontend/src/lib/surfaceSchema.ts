/**
 * Lenient, non-lossy normalizer for a raw Surface definition.
 *
 * The surface carried at `definition.surface` (agents_v2) / the shell `/api/config`
 * response / a designer AST is stored as **raw JSON** and is NOT re-validated through
 * the Python Pydantic `Surface` model on read (whose `children`/`props` default to
 * `[]`/`{}`). So a component authored/persisted/LLM-emitted without a `children` key
 * reaches the renderer with `children === undefined`, and layout code that does
 * `component.children.length` (`lib/surfaceLayout.ts`) throws
 * `Cannot read properties of undefined (reading 'length')` — which, in an unprotected
 * region, bubbles to the top-level `<ErrorBoundary name="App">` and blanks the whole app.
 *
 * `normalizeSurface` is the single boundary that restores the `Surface` type's runtime
 * invariants (every component has `children: string[]` + `props: object`, containers are
 * the right kind) so every downstream consumer can trust the shape. It is:
 * - lenient   — coerces missing/null/wrong-typed fields to safe defaults, never throws;
 * - non-lossy — preserves unknown/future keys (zod `.passthrough()`), so a round-trip
 *               through the schema editor keeps server-only fields;
 * - idempotent — normalize(normalize(x)) deep-equals normalize(x).
 * Returns `null` only when the input is not surface-like at the top level
 * (`version !== 1` or `components` is not an array).
 *
 * Defaults mirror the Python source of truth:
 *   framework `databricks_deep_research/surface/schema.py`
 *   (SurfaceComponent.children = Field(default_factory=list), props = Field(default_factory=dict)).
 */
import { z } from 'zod';

import type { Surface } from '@/types/surface';

/** Coerce any value into a clean `string[]` — missing/null/non-array/mixed all collapse
 * to a filtered array of strings, preserving valid ids. Never throws. */
const idArray = z.preprocess(
  (v) =>
    Array.isArray(v) ? v.filter((x): x is string => typeof x === 'string') : [],
  z.array(z.string()),
);

/** A permissive record — a non-object (null/array/scalar) or missing value becomes `{}`.
 * A valid object passes through unchanged (non-lossy). */
const looseRecord = z.record(z.string(), z.unknown()).catch({});

const isPlainObject = (x: unknown): boolean =>
  x !== null && typeof x === 'object' && !Array.isArray(x);

/** An array whose non-object elements are dropped, but which **fails** on a non-array
 * input (so a malformed top-level `components` makes the whole surface not-surface-like →
 * `null`). */
const strictObjectArray = (element: z.ZodTypeAny): z.ZodTypeAny =>
  z.preprocess(
    (v) => (Array.isArray(v) ? v.filter(isPlainObject) : v),
    z.array(element),
  );

/** An array whose non-object elements are dropped and whose non-array input collapses to
 * `[]` (for optional/secondary arrays like `layout.sections` — never nulls the surface). */
const looseObjectArray = (element: z.ZodTypeAny): z.ZodTypeAny =>
  z.preprocess(
    (v) => (Array.isArray(v) ? v.filter(isPlainObject) : []),
    z.array(element),
  );

const ComponentSchema = z
  .object({
    id: z.string().catch(''),
    component: z.string().catch(''),
    props: looseRecord,
    children: idArray,
  })
  .passthrough();

const SectionSchema = z
  .object({
    // Only `children` is crash-critical (surfaceLayout iterates it); id/title/role/
    // default_open pass through untouched.
    children: idArray,
  })
  .passthrough();

const LayoutSchema = z
  .object({
    sections: looseObjectArray(SectionSchema).optional(),
  })
  .passthrough();

const SurfaceSchema = z
  .object({
    version: z.literal(1),
    components: strictObjectArray(ComponentSchema),
    data_model: looseRecord,
    bindings: z.array(z.unknown()).catch([]),
    layout: LayoutSchema.optional(),
  })
  .passthrough();

/**
 * Parse + normalize a raw surface value. Returns a fully-normalized `Surface`, or `null`
 * when the input is not surface-like (no `version: 1` / `components` not an array).
 */
export function normalizeSurface(raw: unknown): Surface | null {
  const parsed = SurfaceSchema.safeParse(raw);
  if (!parsed.success) return null;
  return parsed.data as unknown as Surface;
}
