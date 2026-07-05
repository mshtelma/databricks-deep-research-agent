import type { Surface } from '@/types/surface';

import { normalizeSurface } from './surfaceSchema';

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

export function isSurfaceLike(value: unknown): value is Surface {
  if (!isRecord(value)) return false;
  return (
    value['version'] === 1 &&
    Array.isArray(value['components']) &&
    isRecord(value['data_model']) &&
    Array.isArray(value['bindings'])
  );
}

/**
 * Extract the surface from an agent definition (or a surface-like definition during
 * shell export) and **normalize** it so every consumer gets the `Surface` type's runtime
 * invariants (see `normalizeSurface`). The returned object is a normalized copy, not the
 * raw reference. Returns `null` when neither the `surface` sub-object nor the definition
 * itself is surface-like.
 */
export function extractSurfaceFromAgentDefinition(
  definition: unknown,
): Surface | null {
  if (!isRecord(definition)) return null;
  return (
    normalizeSurface(definition['surface']) ?? normalizeSurface(definition)
  );
}
