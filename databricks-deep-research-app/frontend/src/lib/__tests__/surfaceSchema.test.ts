import { describe, it, expect } from 'vitest';

import { normalizeSurface } from '../surfaceSchema';
import {
  deriveSurfaceLayout,
  legacyRunOptionComponentIds,
} from '../surfaceLayout';
import type { Surface } from '@/types/surface';

describe('normalizeSurface', () => {
  it('fills a component missing children with []', () => {
    const s = normalizeSurface({
      version: 1,
      components: [{ id: 'root', component: 'Column', props: {} }],
      data_model: {},
      bindings: [],
    });
    expect(s?.components[0]?.children).toEqual([]);
  });

  it('coerces null / non-array / mixed children to a clean string[]', () => {
    const s = normalizeSurface({
      version: 1,
      components: [
        { id: 'a', component: 'X', props: {}, children: null },
        { id: 'b', component: 'X', props: {}, children: 'nope' },
        {
          id: 'c',
          component: 'X',
          props: {},
          children: ['ok', 2, null, 'also'],
        },
      ],
      data_model: {},
      bindings: [],
    });
    expect(s?.components[0]?.children).toEqual([]);
    expect(s?.components[1]?.children).toEqual([]);
    expect(s?.components[2]?.children).toEqual(['ok', 'also']);
  });

  it('defaults missing / non-object props to {}', () => {
    const s = normalizeSurface({
      version: 1,
      components: [
        { id: 'a', component: 'X', children: [] },
        { id: 'b', component: 'X', props: null, children: [] },
      ],
      data_model: {},
      bindings: [],
    });
    expect(s?.components[0]?.props).toEqual({});
    expect(s?.components[1]?.props).toEqual({});
  });

  it('normalizes section children and tolerates missing layout/bindings/data_model', () => {
    const s = normalizeSurface({
      version: 1,
      components: [
        { id: 'root', component: 'Column', props: {}, children: [] },
      ],
      layout: { sections: [{ id: 's', title: 'S', role: 'results' }] },
    });
    expect(s?.bindings).toEqual([]);
    expect(s?.data_model).toEqual({});
    expect(s?.layout?.sections?.[0]?.children).toEqual([]);
    // role preserved via passthrough
    expect((s?.layout?.sections?.[0] as { role?: string }).role).toBe(
      'results',
    );
  });

  it('preserves unknown / future keys (non-lossy)', () => {
    const s = normalizeSurface({
      version: 1,
      components: [
        {
          id: 'root',
          component: 'Column',
          props: { a: 1 },
          children: [],
          _x: 7,
        },
      ],
      data_model: { foo: 'bar' },
      bindings: [],
      _future: 'keep-me',
    });
    expect((s as unknown as Record<string, unknown>)._future).toBe('keep-me');
    expect((s?.components[0] as unknown as Record<string, unknown>)._x).toBe(7);
    expect(s?.components[0]?.props).toEqual({ a: 1 });
  });

  it('returns null when not surface-like (no version:1 / components not array)', () => {
    expect(normalizeSurface(null)).toBeNull();
    expect(normalizeSurface({ components: [] })).toBeNull();
    expect(normalizeSurface({ version: 2, components: [] })).toBeNull();
    expect(normalizeSurface({ version: 1, components: 'nope' })).toBeNull();
  });

  it('drops non-object component entries instead of throwing', () => {
    const s = normalizeSurface({
      version: 1,
      components: [
        { id: 'root', component: 'Column', props: {}, children: [] },
        'garbage',
        null,
      ],
      data_model: {},
      bindings: [],
    });
    expect(s?.components).toHaveLength(1);
    expect(s?.components[0]?.id).toBe('root');
  });

  it('is idempotent', () => {
    const raw = {
      version: 1,
      components: [{ id: 'root', component: 'Column', props: {} }],
      data_model: {},
      bindings: [],
    };
    const once = normalizeSurface(raw);
    const twice = normalizeSurface(once);
    expect(twice).toEqual(once);
  });
});

describe('surfaceLayout no longer crashes on a childless component (regression)', () => {
  // Before the fix, a component with `children === undefined` threw
  // "Cannot read properties of undefined (reading 'length')" in surfaceLayout,
  // which blanked the whole app via the top-level <ErrorBoundary name="App">.
  const s = normalizeSurface({
    version: 1,
    components: [
      { id: 'root', component: 'Column', props: {}, children: ['field'] },
      { id: 'field', component: 'TextField', props: {} }, // leaf, no children key
    ],
    data_model: {},
    bindings: [],
  }) as Surface;

  it('deriveSurfaceLayout does not throw', () => {
    expect(() => deriveSurfaceLayout(s)).not.toThrow();
  });

  it('legacyRunOptionComponentIds does not throw', () => {
    expect(() => legacyRunOptionComponentIds(s)).not.toThrow();
  });
});
