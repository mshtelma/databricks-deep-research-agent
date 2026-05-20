/**
 * useAstMerge — state machine hook for three-way AST merge.
 *
 * Walks two ASTs (local vs server), identifies field-level differences,
 * and lets the user pick per-field which source wins.
 */

import { useState, useCallback } from 'react';
import type { AST } from '@/types/ast';

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export type MergeState = 'idle' | 'merging' | 'submitting' | 'resolved';

export interface AstFieldConflict {
  /** Dot-notation path into the AST, e.g. "root.label" or "tools.0.config.endpoint" */
  path: string;
  localValue: unknown;
  serverValue: unknown;
}

export interface UseAstMergeResult {
  state: MergeState;
  conflicts: AstFieldConflict[];
  selections: Map<string, 'local' | 'server'>;
  selectField(path: string, source: 'local' | 'server'): void;
  applyMerge(): AST;
  /** Returns false when every selected field picks the same side (no real merge). */
  hasRealMerge(): boolean;
  reset(): void;
}

// ---------------------------------------------------------------------------
// Deep diff — finds paths where a !== b (leaf-level comparison)
// ---------------------------------------------------------------------------

function diffValues(
  a: unknown,
  b: unknown,
  path: string,
  out: AstFieldConflict[],
): void {
  if (a === b) return;

  const aIsObj = a !== null && typeof a === 'object';
  const bIsObj = b !== null && typeof b === 'object';

  if (aIsObj && bIsObj) {
    const aRec = a as Record<string, unknown>;
    const bRec = b as Record<string, unknown>;
    const keys = new Set([...Object.keys(aRec), ...Object.keys(bRec)]);
    for (const key of keys) {
      diffValues(aRec[key], bRec[key], path ? `${path}.${key}` : key, out);
    }
    return;
  }

  // Primitive or mixed types — this is a conflict leaf
  out.push({ path, localValue: a, serverValue: b });
}

function computeConflicts(localAst: AST, serverAst: AST): AstFieldConflict[] {
  const out: AstFieldConflict[] = [];
  diffValues(localAst as unknown, serverAst as unknown, '', out);
  return out;
}

// ---------------------------------------------------------------------------
// Deep set — sets a dot-path value in a (deep-cloned) object
// ---------------------------------------------------------------------------

function deepClone<T>(val: T): T {
  return JSON.parse(JSON.stringify(val)) as T;
}

function setAtPath(obj: Record<string, unknown>, path: string, value: unknown): void {
  const parts = path.split('.');
  let cur: Record<string, unknown> = obj;
  for (let i = 0; i < parts.length - 1; i++) {
    const part = parts[i]!;
    if (cur[part] === undefined || cur[part] === null || typeof cur[part] !== 'object') {
      cur[part] = {};
    }
    cur = cur[part] as Record<string, unknown>;
  }
  const last = parts[parts.length - 1]!;
  cur[last] = value;
}

// ---------------------------------------------------------------------------
// Hook
// ---------------------------------------------------------------------------

export function useAstMerge(localAst: AST, serverAst: AST): UseAstMergeResult {
  const [state, setState] = useState<MergeState>('idle');
  const [selections, setSelections] = useState<Map<string, 'local' | 'server'>>(new Map());

  // Conflicts are computed once from props (stable reference from callsite).
  const conflicts = computeConflicts(localAst, serverAst);

  const selectField = useCallback((path: string, source: 'local' | 'server'): void => {
    setSelections((prev) => {
      const next = new Map(prev);
      next.set(path, source);
      return next;
    });
    setState((prev) => (prev === 'idle' ? 'merging' : prev));
  }, []);

  const applyMerge = useCallback((): AST => {
    // Start from server AST as baseline, override with selections.
    const merged = deepClone(serverAst) as unknown as Record<string, unknown>;
    for (const conflict of conflicts) {
      const chosen = selections.get(conflict.path) ?? 'server';
      const value = chosen === 'local' ? conflict.localValue : conflict.serverValue;
      setAtPath(merged, conflict.path, value);
    }
    setState('resolved');
    return merged as unknown as AST;
  }, [serverAst, conflicts, selections]);

  const hasRealMerge = useCallback((): boolean => {
    if (selections.size === 0) return false;
    const sides = new Set(selections.values());
    // A "real" merge means at least one field picks local AND at least one picks server.
    return sides.has('local') && sides.has('server');
  }, [selections]);

  const reset = useCallback((): void => {
    setState('idle');
    setSelections(new Map());
  }, []);

  return { state, conflicts, selections, selectField, applyMerge, hasRealMerge, reset };
}
