import { describe, it, expect } from 'vitest';
import {
  getAtPointer,
  setAtPointer,
  resolveDynamic,
  mergeDataModel,
} from '../surfaceState';

describe('getAtPointer', () => {
  it('resolves a top-level key', () => {
    expect(getAtPointer({ topic: 'AI' }, '/topic')).toBe('AI');
  });

  it('resolves a nested key', () => {
    expect(getAtPointer({ a: { b: { c: 42 } } }, '/a/b/c')).toBe(42);
  });

  it('returns undefined for missing segment', () => {
    expect(getAtPointer({ a: 1 }, '/b')).toBeUndefined();
  });

  it('returns undefined when intermediate node is not an object', () => {
    expect(getAtPointer({ a: 'string' }, '/a/b')).toBeUndefined();
  });

  it('returns undefined for array intermediates', () => {
    expect(getAtPointer({ a: [1, 2] }, '/a/0')).toBeUndefined();
  });

  it('resolves a falsy value (0)', () => {
    expect(getAtPointer({ count: 0 }, '/count')).toBe(0);
  });

  it('resolves a null value', () => {
    expect(getAtPointer({ x: null }, '/x')).toBeNull();
  });
});

describe('setAtPointer', () => {
  it('sets a top-level key', () => {
    const data = { topic: 'AI' };
    const next = setAtPointer(data, '/topic', 'ML');
    expect(next['topic']).toBe('ML');
  });

  it('does not mutate the original object', () => {
    const data = { topic: 'AI' };
    setAtPointer(data, '/topic', 'ML');
    expect(data['topic']).toBe('AI');
  });

  it('sets a nested key, cloning along the path', () => {
    const data = { a: { b: 1 } };
    const next = setAtPointer(data, '/a/b', 99);
    expect(next['a']).not.toBe(data['a']); // cloned
    expect((next['a'] as Record<string, unknown>)['b']).toBe(99);
    expect((data['a'] as Record<string, unknown>)['b']).toBe(1); // original unchanged
  });

  it('creates missing intermediate objects', () => {
    const data: Record<string, unknown> = {};
    const next = setAtPointer(data, '/x/y/z', 'hello');
    expect((next['x'] as Record<string, unknown>)?.['y']).toEqual({ z: 'hello' });
  });

  it('overwrites a non-object intermediate with a new object', () => {
    const data = { a: 'old' };
    const next = setAtPointer(data, '/a/b', 'new');
    expect((next['a'] as Record<string, unknown>)['b']).toBe('new');
  });

  it('preserves sibling keys', () => {
    const data = { a: { x: 1, y: 2 } };
    const next = setAtPointer(data, '/a/x', 99);
    expect((next['a'] as Record<string, unknown>)['y']).toBe(2);
  });

  it('sets a boolean value', () => {
    const next = setAtPointer({}, '/flag', false);
    expect(next['flag']).toBe(false);
  });
});

describe('resolveDynamic', () => {
  const data = { query: 'hello', nested: { val: 42 } };

  it('resolves a PathRef', () => {
    expect(resolveDynamic({ path: '/query' }, data)).toBe('hello');
  });

  it('resolves a nested PathRef', () => {
    expect(resolveDynamic({ path: '/nested/val' }, data)).toBe(42);
  });

  it('returns undefined for a PathRef pointing to a missing key', () => {
    expect(resolveDynamic({ path: '/missing' }, data)).toBeUndefined();
  });

  it('returns literal string as-is', () => {
    expect(resolveDynamic('literal', data)).toBe('literal');
  });

  it('returns literal number as-is', () => {
    expect(resolveDynamic(7, data)).toBe(7);
  });

  it('returns literal boolean as-is', () => {
    expect(resolveDynamic(true, data)).toBe(true);
  });

  it('returns null as-is', () => {
    expect(resolveDynamic(null, data)).toBeNull();
  });
});

describe('mergeDataModel', () => {
  it('returns defaults unchanged when overrides is empty/undefined', () => {
    const defaults = { inputs: { a: 'x' } };
    expect(mergeDataModel(defaults, undefined)).toBe(defaults);
    expect(mergeDataModel(defaults, {})).toBe(defaults);
  });

  it('deep-merges a nested namespace: keeps persisted, fills missing from defaults', () => {
    const defaults = { inputs: { ticker: 'AAPL', depth: 'deep' } };
    const persisted = { inputs: { ticker: 'MSFT' } };
    expect(mergeDataModel(defaults, persisted)).toEqual({
      inputs: { ticker: 'MSFT', depth: 'deep' },
    });
  });

  it('fills a renamed/new field from defaults (the incident) + keeps stale keys', () => {
    const defaults = { inputs: { ticker_choice: 'AAPL' } }; // new surface
    const persisted = { inputs: { query: 'old text' } }; // stale old-surface key
    expect(mergeDataModel(defaults, persisted)).toEqual({
      inputs: { ticker_choice: 'AAPL', query: 'old text' },
    });
  });

  it('lets a persisted empty string override a default (user cleared it)', () => {
    const merged = mergeDataModel({ inputs: { q: 'default' } }, { inputs: { q: '' } });
    expect((merged.inputs as Record<string, unknown>).q).toBe('');
  });

  it('is pointer-deep, not top-level (works for /form and /inputs namespaces)', () => {
    expect(mergeDataModel({ form: { a: '1' } }, { form: { b: '2' } })).toEqual({
      form: { a: '1', b: '2' },
    });
  });
});
