/**
 * PendingMutationCard — Databricks-styled card for a proposed AST mutation
 * awaiting user approval.
 *
 * Change preview (Fix #1): an AST-AWARE semantic diff (`computeAstFieldDiff`)
 * that surfaces config / prompt / model-tier / tool-binding changes — not just
 * the node type:label tree. Prompt-only edits now read "1 edit", not "0 edits".
 *
 * Also renders:
 *   - a stale badge + disabled Apply when a newer change was applied (Fix #2)
 *   - a removed-nodes warning when the proposal drops nodes from the current
 *     workflow (Fix #5, detection-only)
 */

import * as React from 'react';
import { GitBranch, Check, X, ChevronDown, ChevronUp, AlertTriangle, Clock } from 'lucide-react';
import type { PendingMutation } from '@/hooks/useChatSession';
import { NormalizationFixPill } from './NormalizationFixPill';
import {
  computeAstFieldDiff,
  formatDiffValue,
  isNoiseField,
  type AstFieldChange,
} from '@/lib/astFieldDiff';

// ---------------------------------------------------------------------------
// Props
// ---------------------------------------------------------------------------

export interface PendingMutationCardProps {
  mutation: PendingMutation;
  onApply: (id: string) => void;
  onReject: (id: string) => void;
  /** Disable Apply while a previous apply is resolving (async, Fix #3). */
  applyInFlight?: boolean;
  /**
   * When false, the Designer auto-repair pill collapses to a single "!"
   * compact indicator. Controlled by the user-preference toggle.
   * Default: true (show full pill + expandable detail panel).
   */
  showAutoRepairDetails?: boolean;
}

// ---------------------------------------------------------------------------
// Field change row
// ---------------------------------------------------------------------------

const KIND_COLOR: Record<AstFieldChange['kind'], string> = {
  added: 'var(--db-green-700)',
  removed: 'var(--db-lava-700)',
  modified: 'var(--db-navy-800)',
};

const KIND_GLYPH: Record<AstFieldChange['kind'], string> = {
  added: '+',
  removed: '−',
  modified: '~',
};

function FieldChangeRow({ change }: { change: AstFieldChange }): React.ReactElement {
  const [expanded, setExpanded] = React.useState(false);
  const oldStr = formatDiffValue(change.oldValue, expanded ? 4000 : 120);
  const newStr = formatDiffValue(change.newValue, expanded ? 4000 : 120);
  const long =
    formatDiffValue(change.oldValue, 4000).length > 120 ||
    formatDiffValue(change.newValue, 4000).length > 120;

  return (
    <li className="rounded-[5px] border border-db-gray-lines bg-white px-2 py-1.5">
      <div className="flex items-center gap-1.5">
        <span
          className="font-db-mono text-[11px] font-bold"
          style={{ color: KIND_COLOR[change.kind] }}
          aria-hidden
        >
          {KIND_GLYPH[change.kind]}
        </span>
        <span className="text-[12px] font-medium text-db-navy-800">{change.field}</span>
        {long && (
          <button
            type="button"
            onClick={() => setExpanded((v) => !v)}
            className="ml-auto text-[10px] font-medium text-db-gray-text hover:text-db-navy-800"
          >
            {expanded ? 'Show less' : 'View full'}
          </button>
        )}
      </div>
      {change.kind === 'added' ? (
        <div className="mt-1 break-words font-db-mono text-[11px] text-db-green-700">{newStr}</div>
      ) : change.kind === 'removed' ? (
        <div className="mt-1 break-words font-db-mono text-[11px] text-db-lava-700 line-through">
          {oldStr}
        </div>
      ) : (
        <div className="mt-1 space-y-0.5 font-db-mono text-[11px]">
          <div className="break-words text-db-lava-700">
            <span className="text-db-gray-text">− </span>
            {oldStr}
          </div>
          <div className="break-words text-db-green-700">
            <span className="text-db-gray-text">+ </span>
            {newStr}
          </div>
        </div>
      )}
    </li>
  );
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export function PendingMutationCard({
  mutation,
  onApply,
  onReject,
  applyInFlight = false,
  showAutoRepairDetails = true,
}: PendingMutationCardProps): React.ReactElement {
  const hasErrors = mutation.validationErrors.length > 0;
  const isStale = mutation.isStale === true;
  const [showDiff, setShowDiff] = React.useState(false);

  const diff = React.useMemo(
    () => computeAstFieldDiff(mutation.oldAst, mutation.newAst),
    [mutation.oldAst, mutation.newAst],
  );

  // Field changes auto-applied by the Layer-2 normalizer are shown via the
  // NormalizationFixPill, not the user-meaningful list (Fix #1 segregation).
  const autoRepairPaths = React.useMemo(
    () => new Set(mutation.normalizationFixes.map((f) => f.path)),
    [mutation.normalizationFixes],
  );
  const userChanges = React.useMemo(
    () => diff.fieldChanges.filter((c) => !isNoiseField(c.rawPath) && !autoRepairPaths.has(c.rawPath)),
    [diff.fieldChanges, autoRepairPaths],
  );

  const editCount = userChanges.length + diff.addedNodeCount + diff.removedNodeCount;

  const op = (mutation.mutationKind || 'PROPOSE')
    .toString()
    .toUpperCase()
    .replace(/[^A-Z_]/g, '_');

  const applyDisabled = hasErrors || isStale || applyInFlight;

  return (
    <div
      className="overflow-hidden rounded-db-md border bg-white shadow-db-md"
      style={{
        borderColor: hasErrors
          ? 'var(--db-lava-300)'
          : isStale
          ? 'var(--db-gray-lines)'
          : 'var(--db-navy-300)',
        opacity: isStale ? 0.8 : 1,
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
          {hasErrors ? 'INVALID MUTATION' : isStale ? 'OUT OF DATE' : 'PROPOSED MUTATION'}
        </span>
        {isStale && (
          <span
            className="inline-flex items-center gap-1 rounded-sm bg-db-oat-medium px-1.5 py-0.5 font-db-mono text-[9px] font-semibold uppercase tracking-[0.04em] text-db-gray-text"
            title="A newer change was applied; regenerate this one to apply it."
          >
            <Clock size={9} /> stale
          </span>
        )}
        <span className="ml-auto truncate font-db-mono text-[10px] text-db-gray-text">{op}</span>
      </div>

      {/* Body */}
      <div className="p-3">
        <div className="mb-1 text-[13px] font-medium text-db-navy-800">{mutation.description}</div>

        {isStale && (
          <p className="mt-1.5 rounded-db-md border border-db-gray-lines bg-db-oat-light px-2.5 py-1.5 text-[11px] text-db-gray-text">
            A newer change was applied to the workflow, so this proposal is out of date. Reject it and
            ask again to regenerate against the current workflow.
          </p>
        )}

        {/* Removed-nodes warning (Fix #5 — detection only) */}
        {diff.removedNodeCount > 0 && (
          <div
            role="alert"
            className="mt-2 flex items-start gap-1.5 rounded-db-md border border-db-yellow-300 bg-db-yellow-300/30 px-2.5 py-1.5 text-[11px] text-db-yellow-800"
          >
            <AlertTriangle size={12} className="mt-0.5 shrink-0" />
            <span>
              Removes {diff.removedNodeCount} node{diff.removedNodeCount === 1 ? '' : 's'} from your
              current workflow
              {': '}
              {diff.structural
                .filter((s) => s.kind === 'node_removed')
                .map((s) => s.label)
                .join(', ')}
              .
            </span>
          </div>
        )}

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

        {/* Added-nodes summary */}
        {diff.addedNodeCount > 0 && (
          <p className="mt-2 text-[11px] text-db-green-700">
            Adds {diff.addedNodeCount} node{diff.addedNodeCount === 1 ? '' : 's'}:{' '}
            {diff.structural
              .filter((s) => s.kind === 'node_added')
              .map((s) => `${s.label} (${s.nodeType})`)
              .join(', ')}
          </p>
        )}

        {/* Collapsible field-change diff */}
        {(userChanges.length > 0 || editCount > 0) && (
          <div className="mt-3">
            <button
              type="button"
              onClick={() => setShowDiff((v) => !v)}
              aria-expanded={showDiff}
              className="inline-flex items-center gap-1 text-[11px] font-medium text-db-gray-text transition-colors hover:text-db-navy-800"
            >
              {showDiff ? <ChevronUp size={11} /> : <ChevronDown size={11} />}
              {showDiff ? 'Hide changes' : `View changes (${editCount} edit${editCount === 1 ? '' : 's'})`}
            </button>
            {showDiff && userChanges.length > 0 && (
              <ul className="mt-2 space-y-1.5" data-testid="field-change-list">
                {userChanges.map((c) => (
                  <FieldChangeRow key={`${c.rawPath}:${c.kind}`} change={c} />
                ))}
              </ul>
            )}
            {showDiff && userChanges.length === 0 && editCount > 0 && (
              <p className="mt-2 text-[11px] text-db-gray-text">
                Structural changes only — see the added/removed summary above.
              </p>
            )}
          </div>
        )}

        {/* Actions */}
        <div className="mt-3 flex gap-1.5">
          <button
            type="button"
            disabled={applyDisabled}
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
