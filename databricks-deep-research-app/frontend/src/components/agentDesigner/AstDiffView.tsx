/**
 * AstDiffView — renders per-field conflict rows for the three-way merge UI.
 *
 * Lazy-loaded via React.lazy() in EtagConflictModal so it doesn't bloat
 * the initial agent-designer chunk until the user actually opens the diff.
 */

import * as React from 'react';
import type { AstFieldConflict } from '@/hooks/useAstMerge';

// ---------------------------------------------------------------------------
// Props
// ---------------------------------------------------------------------------

export interface AstDiffViewProps {
  conflicts: AstFieldConflict[];
  selections: Map<string, 'local' | 'server'>;
  onSelect(path: string, source: 'local' | 'server'): void;
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function displayValue(v: unknown): string {
  if (v === undefined) return '(undefined)';
  if (v === null) return '(null)';
  if (typeof v === 'string') return v;
  return JSON.stringify(v);
}

// ---------------------------------------------------------------------------
// ConflictRow
// ---------------------------------------------------------------------------

interface ConflictRowProps {
  conflict: AstFieldConflict;
  selected: 'local' | 'server' | undefined;
  onSelect(source: 'local' | 'server'): void;
  rowIndex: number;
}

function ConflictRow({ conflict, selected, onSelect, rowIndex }: ConflictRowProps): React.ReactElement {
  // Keyboard nav: Enter → local, Shift+Enter → server.
  function handleKeyDown(e: React.KeyboardEvent<HTMLDivElement>): void {
    if (e.key === 'Enter') {
      if (e.shiftKey) {
        onSelect('server');
      } else {
        onSelect('local');
      }
    }
  }

  const localSelected = selected === 'local';
  const serverSelected = selected === 'server';

  return (
    <div
      role="group"
      aria-label={`Conflict at ${conflict.path}`}
      data-testid={`conflict-row-${rowIndex}`}
      className="mb-3 rounded border border-slate-200 overflow-hidden focus-within:ring-2 focus-within:ring-blue-400 outline-none"
      tabIndex={0}
      onKeyDown={handleKeyDown}
    >
      {/* Path label */}
      <div className="bg-slate-100 px-3 py-1 text-xs font-mono text-slate-700 border-b border-slate-200">
        {conflict.path || '(root)'}
      </div>

      {/* Side-by-side values */}
      <div className="flex divide-x divide-slate-200">
        {/* Local */}
        <button
          type="button"
          className={`flex-1 p-3 text-left text-xs transition-colors cursor-pointer focus:outline-none focus-visible:ring-2 focus-visible:ring-blue-400 ${
            localSelected
              ? 'bg-blue-100 ring-2 ring-inset ring-blue-400'
              : 'bg-blue-50 hover:bg-blue-100'
          }`}
          aria-pressed={localSelected}
          aria-label={`Select local value for ${conflict.path}`}
          onClick={() => onSelect('local')}
        >
          <div className="mb-1 font-semibold text-blue-700 text-xs uppercase tracking-wide">
            Local {localSelected && '✓'}
          </div>
          <div className="font-mono break-all text-slate-800">
            {displayValue(conflict.localValue)}
          </div>
        </button>

        {/* Server */}
        <button
          type="button"
          className={`flex-1 p-3 text-left text-xs transition-colors cursor-pointer focus:outline-none focus-visible:ring-2 focus-visible:ring-green-400 ${
            serverSelected
              ? 'bg-green-100 ring-2 ring-inset ring-green-400'
              : 'bg-green-50 hover:bg-green-100'
          }`}
          aria-pressed={serverSelected}
          aria-label={`Select server value for ${conflict.path}`}
          onClick={() => onSelect('server')}
        >
          <div className="mb-1 font-semibold text-green-700 text-xs uppercase tracking-wide">
            Server {serverSelected && '✓'}
          </div>
          <div className="font-mono break-all text-slate-800">
            {displayValue(conflict.serverValue)}
          </div>
        </button>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// AstDiffView (default export for React.lazy)
// ---------------------------------------------------------------------------

export default function AstDiffView({
  conflicts,
  selections,
  onSelect,
}: AstDiffViewProps): React.ReactElement {
  if (conflicts.length === 0) {
    return (
      <p className="text-sm text-slate-500 italic py-4 text-center">
        No field-level conflicts detected.
      </p>
    );
  }

  return (
    <div
      role="list"
      aria-label="Field conflicts"
      className="mt-4 max-h-64 overflow-y-auto pr-1"
    >
      {conflicts.map((conflict, i) => (
        <ConflictRow
          key={conflict.path}
          conflict={conflict}
          selected={selections.get(conflict.path)}
          onSelect={(source) => onSelect(conflict.path, source)}
          rowIndex={i}
        />
      ))}
    </div>
  );
}
