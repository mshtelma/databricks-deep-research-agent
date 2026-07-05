/**
 * Pure data-model helpers for surface state management.
 *
 * All functions are pure / side-effect-free — safe to use in reducers or
 * anywhere immutability is required.
 */

import { isPathRef, type DynamicValue } from '@/types/surface';

// ---------------------------------------------------------------------------
// getAtPointer
// ---------------------------------------------------------------------------

/**
 * Resolve a JSON-Pointer (e.g. "/topic/query") against a data object.
 *
 * Returns `undefined` when any segment is missing or the current node is not
 * a plain object. Never throws.
 */
export function getAtPointer(data: Record<string, unknown>, pointer: string): unknown {
  const segments = pointer.replace(/^\//, '').split('/');
  let current: unknown = data;
  for (const seg of segments) {
    if (typeof current !== 'object' || current === null || Array.isArray(current)) {
      return undefined;
    }
    current = (current as Record<string, unknown>)[seg];
  }
  return current;
}

// ---------------------------------------------------------------------------
// setAtPointer
// ---------------------------------------------------------------------------

/**
 * Immutably set a value at a JSON-Pointer path.
 *
 * Clones every object along the path; creates missing intermediate objects.
 * Returns a new top-level record without mutating any existing node.
 */
export function setAtPointer(
  data: Record<string, unknown>,
  pointer: string,
  value: unknown,
): Record<string, unknown> {
  const segments = pointer.replace(/^\//, '').split('/');

  function setIn(current: unknown, path: string[]): unknown {
    const [head, ...tail] = path;
    if (head === undefined) return value;

    const obj: Record<string, unknown> =
      typeof current === 'object' && current !== null && !Array.isArray(current)
        ? { ...(current as Record<string, unknown>) }
        : {};

    obj[head] = setIn(obj[head], tail);
    return obj;
  }

  return setIn(data, segments) as Record<string, unknown>;
}

// ---------------------------------------------------------------------------
// resolveDynamic
// ---------------------------------------------------------------------------

/**
 * Resolve a DynamicValue against the current data model.
 *
 * - PathRef → `getAtPointer(data, ref.path)`
 * - Anything else → returned as-is (scalar or null)
 */
export function resolveDynamic(value: DynamicValue, data: Record<string, unknown>): unknown {
  if (isPathRef(value)) {
    return getAtPointer(data, value.path);
  }
  return value;
}

// ---------------------------------------------------------------------------
// mergeDataModel
// ---------------------------------------------------------------------------

function isPlainObject(v: unknown): v is Record<string, unknown> {
  return typeof v === 'object' && v !== null && !Array.isArray(v);
}

/**
 * Deep-merge a persisted form state (`overrides`) onto the current surface's
 * defaults, so restoring a saved form keeps the user's entries for fields that
 * still exist while filling missing/renamed fields from the surface defaults.
 *
 * Form values nest under a namespace object (e.g. `/form` or `/inputs`), so the
 * merge is recursive, NOT top-level (a top-level merge would just replace the one
 * namespace key). Plain-object nodes merge recursively; scalars/arrays in
 * `overrides` win; nodes present on only one side are kept (stale persisted keys
 * are harmless — no component reads them). Pure; never mutates either input.
 */
export function mergeDataModel(
  defaults: Record<string, unknown>,
  overrides: Record<string, unknown> | undefined | null,
): Record<string, unknown> {
  if (!overrides || Object.keys(overrides).length === 0) return defaults;
  const out: Record<string, unknown> = { ...defaults };
  for (const [key, ov] of Object.entries(overrides)) {
    const dv = out[key];
    out[key] = isPlainObject(dv) && isPlainObject(ov) ? mergeDataModel(dv, ov) : ov;
  }
  return out;
}
