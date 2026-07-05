import { describe, it, expect } from 'vitest';

import {
  extractSurfaceFromAgentDefinition,
  isSurfaceLike,
} from '../agentSurface';
import type { Surface } from '@/types/surface';

function surface(): Surface {
  return {
    version: 1,
    components: [{ id: 'root', component: 'Column', props: {}, children: [] }],
    data_model: {},
    bindings: [],
  };
}

describe('agentSurface helpers', () => {
  it('extracts definition.surface', () => {
    const s = surface();
    // Returns a normalized copy (not the raw reference) — structurally equal for a
    // well-formed surface.
    expect(extractSurfaceFromAgentDefinition({ surface: s })).toEqual(s);
  });

  it('extracts a surface-like definition', () => {
    const s = surface();
    expect(extractSurfaceFromAgentDefinition(s)).toEqual(s);
  });

  it('rejects malformed definitions', () => {
    expect(extractSurfaceFromAgentDefinition(null)).toBeNull();
    expect(
      extractSurfaceFromAgentDefinition({ surface: { components: [] } }),
    ).toBeNull();
    expect(isSurfaceLike({ version: 1, components: [], data_model: {} })).toBe(
      false,
    );
  });

  it('normalizes a component missing children to []', () => {
    // Root cause of the `[App]` crash: a stored/LLM/legacy surface whose
    // component has no `children` key reaches surfaceLayout's
    // `component.children.length` and throws. Extraction must guarantee the
    // `children: string[]` invariant.
    const raw = {
      version: 1,
      components: [{ id: 'root', component: 'Column', props: {} }],
      data_model: {},
      bindings: [],
    };
    const result = extractSurfaceFromAgentDefinition(raw);
    expect(result?.components[0]?.children).toEqual([]);
  });
});
