/**
 * Tests for useAstMerge hook.
 */

import { describe, it, expect } from 'vitest';
import { renderHook, act } from '@testing-library/react';
import { useAstMerge } from '../useAstMerge';
import { createDraftWorkflow } from '@/lib/workflowAst';
import type { AST } from '@/types/ast';

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function makeAst(overrides: Partial<AST> = {}): AST {
  return {
    ...createDraftWorkflow('Test Workflow'),
    schema_version: 1,
    root: {
      id: 'root-id',
      type: 'sequence',
      label: 'root',
      config: {},
      children: [],
    },
    ...overrides,
  };
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe('useAstMerge', () => {
  it('test_idle_to_merging — calling selectField transitions state to merging', () => {
    const local = makeAst({ schema_version: 1 });
    const server = makeAst({ schema_version: 2 });

    const { result } = renderHook(() => useAstMerge(local, server));

    expect(result.current.state).toBe('idle');

    act(() => {
      result.current.selectField('schema_version', 'local');
    });

    expect(result.current.state).toBe('merging');
  });

  it('test_select_field_changes_selection — selecting twice for same path overwrites', () => {
    const local = makeAst({ schema_version: 1 });
    const server = makeAst({ schema_version: 2 });

    const { result } = renderHook(() => useAstMerge(local, server));

    act(() => {
      result.current.selectField('schema_version', 'local');
    });

    expect(result.current.selections.get('schema_version')).toBe('local');

    act(() => {
      result.current.selectField('schema_version', 'server');
    });

    expect(result.current.selections.get('schema_version')).toBe('server');
  });

  it('test_apply_merge_produces_correct_ast — mixed selections produce correct merged AST', () => {
    const local = makeAst({
      schema_version: 1,
      root: {
        id: 'root-id',
        type: 'sequence',
        label: 'local-label',
        config: {},
        children: [],
      },
    });
    const server = makeAst({
      schema_version: 2,
      root: {
        id: 'root-id',
        type: 'sequence',
        label: 'server-label',
        config: {},
        children: [],
      },
    });

    const { result } = renderHook(() => useAstMerge(local, server));

    // Pick local schema_version, server label
    act(() => {
      result.current.selectField('schema_version', 'local');
      result.current.selectField('root.label', 'server');
    });

    let merged!: AST;
    act(() => {
      merged = result.current.applyMerge();
    });

    expect(merged.schema_version).toBe(1);        // from local
    expect(merged.root.label).toBe('server-label');     // from server
  });

  it('test_identical_side_detection — all local → hasRealMerge false; mixed → true', () => {
    const local = makeAst({
      schema_version: 1,
      root: {
        id: 'root-id',
        type: 'sequence',
        label: 'local-label',
        config: {},
        children: [],
      },
    });
    const server = makeAst({
      schema_version: 2,
      root: {
        id: 'root-id',
        type: 'sequence',
        label: 'server-label',
        config: {},
        children: [],
      },
    });

    const { result } = renderHook(() => useAstMerge(local, server));

    // All local — not a real merge
    act(() => {
      result.current.selectField('schema_version', 'local');
      result.current.selectField('root.label', 'local');
    });

    expect(result.current.hasRealMerge()).toBe(false);

    // Now pick one from server → real merge
    act(() => {
      result.current.selectField('root.label', 'server');
    });

    expect(result.current.hasRealMerge()).toBe(true);
  });

  it('test_reset_clears_state — reset() sets state to idle and clears selections', () => {
    const local = makeAst({ schema_version: 1 });
    const server = makeAst({ schema_version: 2 });

    const { result } = renderHook(() => useAstMerge(local, server));

    act(() => {
      result.current.selectField('schema_version', 'local');
    });

    expect(result.current.state).toBe('merging');
    expect(result.current.selections.size).toBe(1);

    act(() => {
      result.current.reset();
    });

    expect(result.current.state).toBe('idle');
    expect(result.current.selections.size).toBe(0);
  });

  it('conflicts list correctly identifies differing leaf paths', () => {
    const local = makeAst({
      schema_version: 1,
      root: {
        id: 'root-id',
        type: 'sequence',
        label: 'local-label',
        config: { key: 'a' },
        children: [],
      },
    });
    const server = makeAst({
      schema_version: 1,   // same
      root: {
        id: 'root-id',
        type: 'sequence',
        label: 'server-label',  // different
        config: { key: 'b' },  // different
        children: [],
      },
    });

    const { result } = renderHook(() => useAstMerge(local, server));

    // schema_version is identical → not in conflicts
    const paths = result.current.conflicts.map((c) => c.path);
    expect(paths).not.toContain('schema_version');
    expect(paths).toContain('root.label');
    expect(paths).toContain('root.config.key');
  });
});
