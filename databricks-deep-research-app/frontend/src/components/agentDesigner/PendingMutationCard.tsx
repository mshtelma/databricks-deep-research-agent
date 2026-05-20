/**
 * PendingMutationCard — Databricks-styled card for a proposed AST mutation
 * awaiting user approval.
 *
 * Visual:
 *   - White card with status-coloured border (navy-300 pending / green-300
 *     applied — applied state is short-lived since the parent removes the
 *     mutation on approval, so we always render in pending mode here).
 *   - Header strip in oat-light with a mono uppercase status label and the
 *     mutation operation kind.
 *   - Body: title (description) + summary + collapsible AST delta + validation
 *     errors + Apply/Reject buttons.
 */

import * as React from 'react';
import { GitBranch, Check, X, ChevronDown, ChevronUp } from 'lucide-react';
import type { PendingMutation } from '@/hooks/useChatSession';
import type { Block } from '@/types/ast';
import { NormalizationFixPill } from './NormalizationFixPill';

// ---------------------------------------------------------------------------
// Props
// ---------------------------------------------------------------------------

export interface PendingMutationCardProps {
  mutation: PendingMutation;
  onApply: (id: string) => void;
  onReject: (id: string) => void;
  /**
   * When false, the Designer auto-repair pill collapses to a single "!"
   * compact indicator. Controlled by the user-preference toggle.
   * Default: true (show full pill + expandable detail panel).
   */
  showAutoRepairDetails?: boolean;
}

// ---------------------------------------------------------------------------
// Diff lines (simple textual representation)
// ---------------------------------------------------------------------------

interface DiffLine {
  /** ' ', '+', '-' */
  marker: ' ' | '+' | '-';
  text: string;
  depth: number;
}

function flattenBlock(block: Block | undefined, depth = 0): Array<{ id: string; line: string; depth: number }> {
  if (!block) return [];
  const out: Array<{ id: string; line: string; depth: number }> = [];
  const labelText = block.label ? ` ${block.label}` : '';
  out.push({ id: block.id, line: `${block.type}:${labelText}`, depth });
  for (const child of block.children ?? []) {
    out.push(...flattenBlock(child, depth + 1));
  }
  // plan_and_execute body
  const body = (block.config as Record<string, unknown>)['body'] as Block | undefined;
  if (body) out.push(...flattenBlock(body, depth + 1));
  return out;
}

function buildDiff(oldRoot: Block | undefined, newRoot: Block | undefined): DiffLine[] {
  const oldFlat = flattenBlock(oldRoot);
  const newFlat = flattenBlock(newRoot);
  const oldIds = new Set(oldFlat.map((b) => b.id));
  const newIds = new Set(newFlat.map((b) => b.id));

  // Walk the new tree; mark added rows; insert removed rows where their parent
  // would have placed them. For our purposes — a simple linear merge based on
  // position is good enough.
  const result: DiffLine[] = [];
  for (const row of newFlat) {
    const marker: DiffLine['marker'] = oldIds.has(row.id) ? ' ' : '+';
    result.push({ marker, text: row.line, depth: row.depth });
  }
  for (const row of oldFlat) {
    if (!newIds.has(row.id)) {
      result.push({ marker: '-', text: row.line, depth: row.depth });
    }
  }
  return result;
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export function PendingMutationCard({
  mutation,
  onApply,
  onReject,
  showAutoRepairDetails = true,
}: PendingMutationCardProps): React.ReactElement {
  const hasErrors = mutation.validationErrors.length > 0;
  const [showDiff, setShowDiff] = React.useState(false);

  const diff = React.useMemo(
    () => buildDiff(mutation.oldAst?.root, mutation.newAst?.root),
    [mutation.oldAst, mutation.newAst],
  );

  const op = (mutation.mutationKind || 'PROPOSE')
    .toString()
    .toUpperCase()
    .replace(/[^A-Z_]/g, '_');

  return (
    <div
      className="overflow-hidden rounded-db-md border bg-white shadow-db-md"
      style={{
        borderColor: hasErrors ? 'var(--db-lava-300)' : 'var(--db-navy-300)',
      }}
    >
      {/* Status strip */}
      <div className="flex items-center gap-2 border-b border-db-gray-lines bg-db-oat-light px-3 py-2.5">
        <GitBranch
          size={13}
          strokeWidth={2.2}
          className={hasErrors ? 'text-db-lava-700' : 'text-db-navy-800'}
        />
        <span className="font-db-mono text-[11px] font-medium tracking-[0.02em] text-db-navy-800">
          {hasErrors ? 'INVALID MUTATION' : 'PROPOSED MUTATION'}
        </span>
        <span className="ml-auto truncate font-db-mono text-[10px] text-db-gray-text">{op}</span>
      </div>

      {/* Body */}
      <div className="p-3">
        <div className="mb-1 text-[13px] font-medium text-db-navy-800">{mutation.description}</div>

        {mutation.normalizationFixes.length > 0 && (
          <div className="mt-2">
            <NormalizationFixPill
              fixes={mutation.normalizationFixes}
              compact={!showAutoRepairDetails}
            />
          </div>
        )}

        {hasErrors && (
          <ul
            aria-label="Validation errors"
            className="mt-2 list-disc space-y-0.5 pl-5 text-[11px] text-db-lava-700"
          >
            {mutation.validationErrors.map((err, idx) => (
              <li key={idx}>{err.message}</li>
            ))}
          </ul>
        )}

        {/* Collapsible diff */}
        {diff.length > 0 && (
          <div className="mt-3">
            <button
              type="button"
              onClick={() => setShowDiff((v) => !v)}
              aria-expanded={showDiff}
              className="inline-flex items-center gap-1 text-[11px] font-medium text-db-gray-text transition-colors hover:text-db-navy-800"
            >
              {showDiff ? <ChevronUp size={11} /> : <ChevronDown size={11} />}
              {showDiff ? 'Hide change' : `View change (${diff.filter((d) => d.marker !== ' ').length} edits)`}
            </button>
            {showDiff && (
              <pre
                className="mt-2 max-h-48 overflow-auto rounded-[5px] px-2.5 py-2 font-db-mono text-[11px] leading-[1.55]"
                style={{ background: 'var(--db-navy-900)' }}
              >
                {diff.map((d, idx) => {
                  const color =
                    d.marker === '+'
                      ? 'var(--db-green-300)'
                      : d.marker === '-'
                      ? 'var(--db-lava-400)'
                      : 'var(--db-navy-300)';
                  const indent = '  '.repeat(d.depth);
                  return (
                    <div key={idx} style={{ color }}>
                      {d.marker} {indent}
                      {d.text}
                    </div>
                  );
                })}
              </pre>
            )}
          </div>
        )}

        {/* Actions */}
        <div className="mt-3 flex gap-1.5">
          <button
            type="button"
            disabled={hasErrors}
            onClick={() => onApply(mutation.id)}
            aria-label="Apply mutation"
            className="inline-flex flex-1 items-center justify-center gap-1.5 rounded-db-md bg-db-lava-600 px-3 py-1.5 text-[13px] font-medium text-white transition-colors hover:bg-db-lava-700 disabled:cursor-not-allowed disabled:opacity-55"
          >
            <Check size={12} strokeWidth={2.5} /> Apply
          </button>
          <button
            type="button"
            onClick={() => onReject(mutation.id)}
            aria-label="Reject mutation"
            className="inline-flex flex-1 items-center justify-center gap-1.5 rounded-db-md border border-db-gray-lines bg-white px-3 py-1.5 text-[13px] font-medium text-db-navy-800 transition-colors hover:bg-db-oat-medium"
          >
            <X size={12} /> Reject
          </button>
        </div>
      </div>
    </div>
  );
}
