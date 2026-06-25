/**
 * MutationConflictModal — surfaced when a chat-applied mutation can't be applied
 * cleanly because the user edited the canvas while the chat turn was streaming
 * (Fix #3). Distinct from EtagConflictModal: there is NO etag / remote-save
 * concept here — the two sides are the user's current canvas and the proposed
 * chat change.
 *
 * Sides: "Your canvas" (local) vs "Proposed change" (server, in useAstMerge
 * terms). "Apply proposed" always wins to the proposal; "Keep my canvas"
 * discards it; "Show diff" opens a per-field merge (reusing AstDiffView).
 */

import * as React from 'react';
import * as Dialog from '@radix-ui/react-dialog';
import { Button } from '@/components/ui/button';
import { useAstMerge } from '@/hooks/useAstMerge';
import { createDraftWorkflow } from '@/lib/workflowAst';
import type { AST } from '@/types/ast';
import type { AstDiffViewProps } from './AstDiffView';

const AstDiffView = React.lazy(
  () => import('./AstDiffView') as Promise<{ default: React.ComponentType<AstDiffViewProps> }>,
);

export interface MutationConflictModalProps {
  open: boolean;
  onOpenChange(open: boolean): void;
  /** The user's current canvas AST (local side). */
  localAst?: AST;
  /** The proposed mutation AST from chat (server side). */
  serverAst?: AST;
  /** Overwrite the canvas with the proposed AST. */
  onApplyProposed(): void;
  /** Keep the canvas as-is and discard the proposal. */
  onKeepCanvas(): void;
  /** Apply a per-field merged AST (optional advanced path). */
  onSaveMerge?(merged: AST): void;
}

export function MutationConflictModal({
  open,
  onOpenChange,
  localAst,
  serverAst,
  onApplyProposed,
  onKeepCanvas,
  onSaveMerge,
}: MutationConflictModalProps): React.ReactElement {
  const [showDiff, setShowDiff] = React.useState(false);

  const mergeHook = useAstMerge(
    localAst ?? createDraftWorkflow(),
    serverAst ?? createDraftWorkflow(),
  );

  React.useEffect(() => {
    if (!open) {
      setShowDiff(false);
      mergeHook.reset();
    }
    // mergeHook.reset is stable (useCallback)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [open]);

  const canShowDiff = Boolean(localAst && serverAst && onSaveMerge);
  const hasConflicts = mergeHook.conflicts.length > 0;

  function handleApplyProposed(): void {
    onApplyProposed();
    onOpenChange(false);
  }

  function handleKeepCanvas(): void {
    onKeepCanvas();
    onOpenChange(false);
  }

  function handleSaveMerge(): void {
    if (!onSaveMerge) return;
    // Unselected conflicts default to the proposed (server) side in
    // applyMerge, so saving with no selection == applying the proposal.
    const merged = mergeHook.applyMerge();
    onSaveMerge(merged);
    onOpenChange(false);
  }

  return (
    <Dialog.Root open={open} onOpenChange={onOpenChange}>
      <Dialog.Portal>
        <Dialog.Overlay className="fixed inset-0 z-40 bg-db-navy-900/30 backdrop-blur-[2px]" />
        <Dialog.Content
          className="db-root fixed left-1/2 top-1/2 z-50 w-full max-w-lg -translate-x-1/2 -translate-y-1/2 rounded-db-lg border border-db-gray-lines bg-white p-5 font-db-sans shadow-db-xl focus:outline-none"
          aria-describedby="mutation-conflict-description"
        >
          <Dialog.Title className="text-[16px] font-medium text-db-navy-800">
            You edited the canvas during this change
          </Dialog.Title>

          <Dialog.Description
            id="mutation-conflict-description"
            className="mt-1.5 text-[13px] leading-[1.55] text-db-gray-text"
          >
            You changed the workflow while this chat suggestion was being prepared, so it can&apos;t
            be applied automatically. Apply the proposed change (overwriting your canvas edits), keep
            your canvas, or open the diff to merge field-by-field.
          </Dialog.Description>

          {showDiff && canShowDiff && (
            <>
              <p className="mt-3 text-[11px] text-db-gray-text">
                <span className="font-semibold text-blue-700">Local</span> = your current canvas ·{' '}
                <span className="font-semibold text-green-700">Server</span> = the proposed change
              </p>
              <React.Suspense
                fallback={
                  <div className="mt-4 flex items-center justify-center py-6 text-sm text-slate-500">
                    Loading diff…
                  </div>
                }
              >
                {hasConflicts ? (
                  <AstDiffView
                    conflicts={mergeHook.conflicts}
                    selections={mergeHook.selections}
                    onSelect={mergeHook.selectField}
                  />
                ) : (
                  <p className="mt-4 text-sm text-slate-500 italic">
                    No field-level differences found.
                  </p>
                )}
              </React.Suspense>
            </>
          )}

          <div className="mt-5 flex flex-col-reverse gap-2 sm:flex-row sm:justify-end">
            <Button
              variant="outline"
              size="sm"
              onClick={handleKeepCanvas}
              aria-label="Keep my canvas and discard the proposed change"
            >
              Keep my canvas
            </Button>

            {canShowDiff && !showDiff && (
              <Button
                variant="outline"
                size="sm"
                onClick={() => setShowDiff(true)}
                aria-label="Show diff between your canvas and the proposed change"
              >
                Show diff
              </Button>
            )}

            {showDiff && canShowDiff && (
              <Button
                variant="default"
                size="sm"
                className="bg-purple-600 hover:bg-purple-700 text-white"
                onClick={handleSaveMerge}
                aria-label="Save merged version"
              >
                Save merge
              </Button>
            )}

            <Button
              variant="default"
              size="sm"
              className="bg-db-lava-600 hover:bg-db-lava-700 text-white"
              onClick={handleApplyProposed}
              aria-label="Apply the proposed change"
            >
              Apply proposed
            </Button>
          </div>
        </Dialog.Content>
      </Dialog.Portal>
    </Dialog.Root>
  );
}
