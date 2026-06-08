/**
 * EtagConflictModal — Radix Dialog surfaced when a PATCH returns 409.
 *
 * V1:   Reload + Force overwrite + Cancel buttons.
 * V1.5: Adds Show Diff button that opens an interactive three-way merge UI.
 *       AstDiffView is lazy-loaded so it doesn't inflate the initial chunk.
 */

import * as React from 'react';
import * as Dialog from '@radix-ui/react-dialog';
import { Button } from '@/components/ui/button';
import { useAstMerge } from '@/hooks/useAstMerge';
import { createDraftWorkflow } from '@/lib/workflowAst';
import type { AST } from '@/types/ast';
import type { AstDiffViewProps } from './AstDiffView';

// ---------------------------------------------------------------------------
// Lazy-load AstDiffView — keeps agent-designer chunk lean until needed.
// ---------------------------------------------------------------------------

const AstDiffView = React.lazy(
  () => import('./AstDiffView') as Promise<{ default: React.ComponentType<AstDiffViewProps> }>,
);

// ---------------------------------------------------------------------------
// Props
// ---------------------------------------------------------------------------

export interface EtagConflictModalProps {
  open: boolean;
  onOpenChange(open: boolean): void;
  currentEtag: string | null;
  onReload(): void;
  onForceOverwrite(): void;
  /** Present when V1.5 merge path is enabled. */
  localAst?: AST;
  serverAst?: AST;
  /** Called with the merged AST + the server's current etag. */
  onSaveMerge?(mergedAst: AST, etag: string): void;
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export function EtagConflictModal({
  open,
  onOpenChange,
  currentEtag,
  onReload,
  onForceOverwrite,
  localAst,
  serverAst,
  onSaveMerge,
}: EtagConflictModalProps): React.ReactElement {
  // Sub-step for the "Are you sure?" confirm on force overwrite.
  const [confirming, setConfirming] = React.useState(false);
  // Whether the diff panel is visible.
  const [showDiff, setShowDiff] = React.useState(false);

  // Merge state machine — only active when both ASTs are provided.
  const mergeHook = useAstMerge(
    localAst ?? createDraftWorkflow(),
    serverAst ?? createDraftWorkflow(),
  );

  // Reset all sub-state when the modal opens/closes.
  React.useEffect(() => {
    if (!open) {
      setConfirming(false);
      setShowDiff(false);
      mergeHook.reset();
    }
    // mergeHook.reset is stable (useCallback)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [open]);

  // V1 paths (unchanged) -------------------------------------------------------

  function handleReload(): void {
    onReload();
    onOpenChange(false);
  }

  function handleForceOverwriteClick(): void {
    if (!confirming) {
      setConfirming(true);
    } else {
      onForceOverwrite();
      onOpenChange(false);
    }
  }

  function handleCancel(): void {
    onOpenChange(false);
  }

  // V1.5 diff path -------------------------------------------------------------

  function handleShowDiff(): void {
    setShowDiff(true);
  }

  function handleSaveMerge(): void {
    if (!onSaveMerge || !currentEtag) return;

    // Short-circuit: if no real merge (all selections on same side), just reload.
    if (!mergeHook.hasRealMerge()) {
      onReload();
      onOpenChange(false);
      return;
    }

    const merged = mergeHook.applyMerge();
    onSaveMerge(merged, currentEtag);
    onOpenChange(false);
  }

  const canShowDiff = Boolean(localAst && serverAst && onSaveMerge);
  const hasConflicts = mergeHook.conflicts.length > 0;
  const allSelected = hasConflicts && mergeHook.selections.size === mergeHook.conflicts.length;

  return (
    <Dialog.Root open={open} onOpenChange={onOpenChange}>
      <Dialog.Portal>
        <Dialog.Overlay className="fixed inset-0 z-40 bg-db-navy-900/30 backdrop-blur-[2px]" />
        <Dialog.Content
          className="db-root fixed left-1/2 top-1/2 z-50 w-full max-w-lg -translate-x-1/2 -translate-y-1/2 rounded-db-lg border border-db-gray-lines bg-white p-5 font-db-sans shadow-db-xl focus:outline-none"
          aria-describedby="etag-conflict-description"
        >
          <Dialog.Title className="text-[16px] font-medium text-db-navy-800">
            Agent was modified elsewhere
          </Dialog.Title>

          <Dialog.Description
            id="etag-conflict-description"
            className="mt-1.5 text-[13px] leading-[1.55] text-db-gray-text"
          >
            Another user or session modified this agent since you last loaded it. Choose how to
            proceed:
          </Dialog.Description>

          {confirming && (
            <p className="mt-3 rounded-db-md border border-db-lava-300 bg-db-lava-100 px-3 py-2 text-[12px] text-db-lava-700">
              Are you sure? This will overwrite the remote version with your local changes.
            </p>
          )}

          {/* Diff panel — lazy-loaded, only visible after Show Diff */}
          {showDiff && canShowDiff && (
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
                  No field-level differences found between local and server versions.
                </p>
              )}
            </React.Suspense>
          )}

          <div className="mt-5 flex flex-col-reverse gap-2 sm:flex-row sm:justify-end">
            <Button
              variant="outline"
              size="sm"
              onClick={handleCancel}
              aria-label="Cancel and keep local changes"
            >
              Cancel
            </Button>

            <Button
              variant="destructive"
              size="sm"
              onClick={handleForceOverwriteClick}
              aria-label={confirming ? 'Confirm force overwrite' : 'Force overwrite remote agent'}
            >
              {confirming ? 'Are you sure?' : 'Force overwrite'}
            </Button>

            <Button
              variant="default"
              size="sm"
              className="bg-blue-600 hover:bg-blue-700 text-white"
              onClick={handleReload}
              aria-label="Reload agent from server"
            >
              Reload
            </Button>

            {/* V1.5: Show Diff button — only when ASTs are provided */}
            {canShowDiff && !showDiff && (
              <Button
                variant="outline"
                size="sm"
                onClick={handleShowDiff}
                aria-label="Show diff between local and server versions"
              >
                Show Diff
              </Button>
            )}

            {/* V1.5: Save Merge button — only visible in diff mode */}
            {showDiff && canShowDiff && (
              <Button
                variant="default"
                size="sm"
                className="bg-purple-600 hover:bg-purple-700 text-white"
                onClick={handleSaveMerge}
                disabled={hasConflicts && !allSelected}
                aria-label="Save merged version"
                title={
                  hasConflicts && !allSelected
                    ? 'Select a value for every conflict to save'
                    : undefined
                }
              >
                Save Merge
              </Button>
            )}
          </div>
        </Dialog.Content>
      </Dialog.Portal>
    </Dialog.Root>
  );
}
