/**
 * ChatOptionsPanel — a single "Options" dropdown that consolidates the
 * secondary, per-run composer controls so the main bar stays uncluttered (P2).
 *
 * Hosts: report style (tone + language), verify-sources, review-plan, and two
 * run-level overrides surfaced for the first time — cross-session memory and the
 * live-web-search follow-up escape hatch. Each maps to an existing
 * `OrchestrationConfig`/`SubmitJobRequest` field threaded from `MessageInput`.
 *
 * Overlay: a portaled, edge-flipping Radix Popover (the `DeployDropdown`
 * pattern). The composer is pinned to the bottom of an `h-screen` column, so a
 * naive `absolute mt-1` panel opened downward and spilled past the viewport,
 * forcing the toolbar to reflow. Portaling to the document root (out of the
 * flex flow) + `side="top"` + a height capped to the available space keeps the
 * panel on-screen and stops the reflow.
 */

import * as React from 'react';
import * as Popover from '@radix-ui/react-popover';
import { ReportStyleSelector } from './ReportStyleSelector';

export interface ChatOptionsPanelProps {
  // Report style (tone + output language)
  showReportStyle?: boolean;
  reportStyleDisabled?: boolean;
  tone: string;
  outputLanguage: string;
  onToneChange: (tone: string) => void;
  onLanguageChange: (language: string) => void;
  // Verify citations with NLI (slower) — citations always render; this toggles
  // the expensive per-claim verification overlay. Shown only when applicable.
  showVerify: boolean;
  verifyDisabled?: boolean;
  verifySources: boolean;
  onVerifyChange: (value: boolean) => void;
  // Plan review — shown only for deep research
  showPlanReview: boolean;
  planReviewDisabled?: boolean;
  enablePlanReview: boolean;
  onPlanReviewChange: (value: boolean) => void;
  // Run-level overrides (P2)
  showCrossSessionMemory?: boolean;
  crossSessionMemoryDisabled?: boolean;
  enableCrossSessionMemory: boolean;
  onCrossSessionMemoryChange: (value: boolean) => void;
  showLiveSearch?: boolean;
  liveSearchDisabled?: boolean;
  allowLiveSearch: boolean;
  onAllowLiveSearchChange: (value: boolean) => void;
  disabled?: boolean;
}

export function ChatOptionsPanel({
  showReportStyle = true,
  reportStyleDisabled = false,
  tone,
  outputLanguage,
  onToneChange,
  onLanguageChange,
  showVerify,
  verifyDisabled = false,
  verifySources,
  onVerifyChange,
  showPlanReview,
  planReviewDisabled = false,
  enablePlanReview,
  onPlanReviewChange,
  showCrossSessionMemory = true,
  crossSessionMemoryDisabled = false,
  enableCrossSessionMemory,
  onCrossSessionMemoryChange,
  showLiveSearch = true,
  liveSearchDisabled = false,
  allowLiveSearch,
  onAllowLiveSearchChange,
  disabled = false,
}: ChatOptionsPanelProps): React.ReactElement {
  const [open, setOpen] = React.useState(false);
  const showRunOptions =
    showVerify || showPlanReview || showCrossSessionMemory || showLiveSearch;

  // Badge count: how many options diverge from their defaults (a quick "you've
  // customized something" hint without opening the panel).
  const activeCount =
    (showReportStyle && tone ? 1 : 0) +
    (showReportStyle && outputLanguage ? 1 : 0) +
    (showVerify && !verifySources ? 1 : 0) +
    (showPlanReview && enablePlanReview ? 1 : 0) +
    (showCrossSessionMemory && enableCrossSessionMemory ? 1 : 0) +
    (showLiveSearch && allowLiveSearch ? 1 : 0);

  return (
    <Popover.Root open={open} onOpenChange={setOpen}>
      <Popover.Trigger asChild>
        <button
          type="button"
          disabled={disabled}
          className="inline-flex items-center gap-1 rounded-db-md border border-db-gray-lines px-2 py-1 text-[12px] text-db-navy-800 hover:bg-db-oat-light disabled:opacity-50"
        >
          Options
          {activeCount > 0 && (
            <span className="ml-1 rounded-full bg-db-navy-800 px-1.5 text-[10px] text-white">
              {activeCount}
            </span>
          )}
        </button>
      </Popover.Trigger>
      <Popover.Portal>
        <Popover.Content
          side="top"
          align="start"
          sideOffset={6}
          collisionPadding={8}
          style={{ maxHeight: 'var(--radix-popover-content-available-height)' }}
          className="z-50 w-72 overflow-auto rounded-db-md border border-db-gray-lines bg-white p-3 shadow-lg"
        >
          {showReportStyle && (
            <>
              <p className="mb-1 font-db-mono text-[10px] font-medium uppercase tracking-[0.06em] text-db-gray-text">
                Report style
              </p>
              <ReportStyleSelector
                tone={tone}
                outputLanguage={outputLanguage}
                onToneChange={onToneChange}
                onLanguageChange={onLanguageChange}
                disabled={disabled || reportStyleDisabled}
              />
            </>
          )}

          {showRunOptions && (
            <>
              <p
                className={`mb-1 font-db-mono text-[10px] font-medium uppercase tracking-[0.06em] text-db-gray-text ${
                  showReportStyle ? 'mt-3' : ''
                }`}
              >
                This run
              </p>
              {showVerify && (
                <label className="flex items-center gap-2 py-0.5 text-[12px] text-db-navy-800">
                  <input
                    type="checkbox"
                    checked={verifySources}
                    disabled={disabled || verifyDisabled}
                    onChange={(e) => onVerifyChange(e.target.checked)}
                  />
                  Verify citations with NLI (slower)
                </label>
              )}
              {showPlanReview && (
                <label className="flex items-center gap-2 py-0.5 text-[12px] text-db-navy-800">
                  <input
                    type="checkbox"
                    checked={enablePlanReview}
                    disabled={disabled || planReviewDisabled}
                    onChange={(e) => onPlanReviewChange(e.target.checked)}
                  />
                  Review plan before research
                </label>
              )}
              {showCrossSessionMemory && (
                <label className="flex items-center gap-2 py-0.5 text-[12px] text-db-navy-800">
                  <input
                    type="checkbox"
                    checked={enableCrossSessionMemory}
                    disabled={disabled || crossSessionMemoryDisabled}
                    onChange={(e) => onCrossSessionMemoryChange(e.target.checked)}
                  />
                  Recall facts from my prior chats
                </label>
              )}
              {showLiveSearch && (
                <label className="flex items-center gap-2 py-0.5 text-[12px] text-db-navy-800">
                  <input
                    type="checkbox"
                    checked={allowLiveSearch}
                    disabled={disabled || liveSearchDisabled}
                    onChange={(e) => onAllowLiveSearchChange(e.target.checked)}
                  />
                  Allow live web search on follow-ups
                </label>
              )}
              <p className="mt-1 text-[10px] italic text-db-gray-text">
                Overrides for this query only.
              </p>
            </>
          )}
        </Popover.Content>
      </Popover.Portal>
    </Popover.Root>
  );
}
