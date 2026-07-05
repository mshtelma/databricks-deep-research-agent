/**
 * Designer "Preview" tab — renders the agent's declarative UI (definition.surface)
 * against a live, editable data model.
 *
 * Clicking an action SIMULATES a run: a dry-run card shows the exact submission
 * the chat would send, while the results region plays running → completed with a
 * clearly-watermarked sample report (real rendering path, fake content).
 * "Run for real" hands the compiled submission to the page-level
 * useSurfacePreviewRun controller (real chat, real job, live stream back into
 * the region). "Try in chat" reuses the Test-run flow (save → open chat bound
 * to this agent).
 */

import React from 'react';
import { Link } from 'react-router-dom';
import { useAgentEditorStore } from '@/stores/agentEditorStore';
import { scaffoldSurface } from '@/api/agentDesigner';
import { compileBinding, type CompiledSubmission } from '@/lib/surfaceCompile';
import { setAtPointer } from '@/lib/surfaceState';
import { buildSamplePayload, buildSampleReport } from '@/lib/sampleReport';
import { buildCitationDataMap } from '@/lib/citations';
import { normalizeSurface } from '@/lib/surfaceSchema';
import { MarkdownRenderer } from '@/components/common';
import { useChatFull } from '@/hooks/useChatFull';
import { messagesApi } from '@/api/client';
import type {
  PreviewRunReference,
  SurfacePreviewRunApi,
} from '@/hooks/useSurfacePreviewRun';
import {
  SurfaceRenderer,
  useSurfaceDataModel,
} from '@/components/surface/SurfaceRenderer';
import type { RunReference, Surface } from '@/types/surface';
import type { AST } from '@/types/ast';

/** Normalize the AST's surface so it satisfies the `Surface` runtime invariants (every
 * component has `children: string[]`, etc.); full validation happens at save on the
 * backend. Shares the single normalizer with the main-chat / shell ingestion paths. */
function surfaceOf(ast: AST | null): Surface | null {
  if (!ast) return null;
  return normalizeSurface((ast as Record<string, unknown>).surface);
}

interface SampleRun {
  ref: PreviewRunReference;
  compiled: CompiledSubmission;
}

// ---------------------------------------------------------------------------
// Dry-run card (simulated submission + Run-for-real entry point)
// ---------------------------------------------------------------------------

function DryRunCard({
  action,
  compiled,
  onDismiss,
  onRunForReal,
  runForRealDisabled,
}: {
  action: string;
  compiled: CompiledSubmission;
  onDismiss: () => void;
  onRunForReal?: () => void;
  runForRealDisabled?: boolean;
}): React.ReactElement {
  return (
    <div className="mt-3 rounded-md border border-db-gray-lines bg-db-gray-50 p-3">
      <div className="mb-1 flex items-center justify-between">
        <span className="text-[11px] font-semibold uppercase tracking-wide text-db-navy-800">
          Simulated run — action “{action}”
        </span>
        <button
          type="button"
          className="text-[11px] text-db-gray-text hover:text-db-navy-800"
          onClick={onDismiss}
        >
          Dismiss
        </button>
      </div>
      <p className="mb-2 text-[11px] text-db-gray-text">
        Nothing was submitted. In chat, this button would start a run with:
      </p>
      <pre className="max-h-56 overflow-auto rounded bg-white p-2 text-[11px] leading-[1.5] text-db-navy-800">
        {JSON.stringify(compiled, null, 2)}
      </pre>
      {onRunForReal && (
        <div className="mt-2 flex items-center justify-end gap-2">
          <span className="text-[11px] text-db-gray-text">
            Execute the saved agent and stream the real report here:
          </span>
          <button
            type="button"
            data-testid="surface-preview-run-real"
            disabled={runForRealDisabled}
            onClick={onRunForReal}
            className="rounded-md bg-db-navy-800 px-2.5 py-1 text-[11px] font-medium text-white hover:bg-db-navy-700 disabled:cursor-not-allowed disabled:opacity-50"
          >
            Run for real
          </button>
        </div>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Region renderers
// ---------------------------------------------------------------------------

function Spinner(): React.ReactElement {
  return (
    <span className="inline-block h-3 w-3 animate-spin rounded-full border-2 border-db-navy-800 border-t-transparent" />
  );
}

function SampleRegion({
  status,
  agentName,
  compiled,
}: {
  status: RunReference['status'];
  agentName: string;
  compiled: CompiledSubmission;
}): React.ReactElement {
  const report = React.useMemo(
    () => buildSampleReport(agentName, compiled),
    [agentName, compiled],
  );
  if (status === 'running') {
    return (
      <div className="flex items-center gap-2 text-[12px] text-db-gray-text">
        <Spinner />
        Simulating…
      </div>
    );
  }
  return (
    <div data-testid="surface-preview-sample">
      <p className="mb-2 rounded border border-amber-200 bg-amber-50 px-2 py-1 text-[11px] font-medium text-amber-800">
        Sample output — illustrative only, not generated by this agent.
      </p>
      <div className="max-h-[40vh] overflow-auto">
        <MarkdownRenderer
          content={report.markdown}
          enableCitations
          citationMode="numeric"
          citationData={report.citationData}
        />
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// SurfacePreviewPanel
// ---------------------------------------------------------------------------

export interface SurfacePreviewPanelProps {
  ast: AST | null;
  /** Agent display name — used in the sample report copy. */
  agentName?: string;
  /** Part B: reuse of the Test-run flow (save-if-dirty → chat bound to agent). */
  onTryInChat?: () => void;
  tryInChatPending?: boolean;
  tryInChatDisabled?: boolean;
  /** Part C: page-level real-run controller (survives tab switches). */
  previewRun?: SurfacePreviewRunApi;
}

export function SurfacePreviewPanel({
  ast,
  agentName = 'This agent',
  onTryInChat,
  tryInChatPending = false,
  tryInChatDisabled = false,
  previewRun,
}: SurfacePreviewPanelProps): React.ReactElement {
  const surface = surfaceOf(ast);
  const isDirty = useAgentEditorStore((s) => s.isDirty);
  const [dataModel, setValue, reset] = useSurfaceDataModel(
    surface?.data_model ?? {},
  );
  const [dryRun, setDryRun] = React.useState<{
    action: string;
    compiled: CompiledSubmission;
  } | null>(null);
  const [scaffolding, setScaffolding] = React.useState(false);
  const [scaffoldError, setScaffoldError] = React.useState<string | null>(null);

  // Part A: per-action simulated runs (running → completed sample).
  const [sampleRuns, setSampleRuns] = React.useState<Record<string, SampleRun>>(
    {},
  );
  const sampleTimerRef = React.useRef<ReturnType<typeof setTimeout> | null>(
    null,
  );
  const clearSampleTimer = React.useCallback(() => {
    if (sampleTimerRef.current) {
      clearTimeout(sampleTimerRef.current);
      sampleTimerRef.current = null;
    }
  }, []);
  React.useEffect(() => clearSampleTimer, [clearSampleTimer]);

  // Part C: resolve the final report of the preview chat (enabled only once
  // the controller has created it).
  const { data: previewChatFull, refetch: refetchPreviewFull } = useChatFull(
    previewRun?.previewChatId ?? undefined,
  );

  // Failed-slot retry for real preview runs: POST restructure then poll the
  // preview chat (every 5s, ≤3 min) until the slots leave "pending".
  const previewPollsRef = React.useRef<Set<ReturnType<typeof setInterval>>>(
    new Set(),
  );
  React.useEffect(() => {
    const timers = previewPollsRef.current;
    return () => {
      timers.forEach((t) => clearInterval(t));
      timers.clear();
    };
  }, []);
  const retryStructuring = React.useCallback(
    (messageId: string, slots: string[]) => {
      const chatId = previewRun?.previewChatId;
      if (!chatId) return;
      messagesApi
        .restructure(chatId, messageId, slots)
        .then(() => {
          void refetchPreviewFull();
          let ticks = 0;
          const timer = setInterval(() => {
            ticks += 1;
            void refetchPreviewFull();
            if (ticks >= 36) {
              clearInterval(timer);
              previewPollsRef.current.delete(timer);
            }
          }, 5000);
          previewPollsRef.current.add(timer);
        })
        .catch(() => {
          /* fail-soft: Retry stays available */
        });
    },
    [previewRun?.previewChatId, refetchPreviewFull],
  );

  // Re-seed the preview data model whenever the AST's surface changes
  // (designer chat edits, Generate default, revision restore). Clears samples
  // but deliberately NOT previewRun.runState — real refs describe live jobs.
  const surfaceKey = React.useMemo(
    () => (surface ? JSON.stringify(surface.data_model) : ''),
    [surface],
  );
  React.useEffect(() => {
    reset(surface?.data_model ?? {});
    setDryRun(null);
    setSampleRuns({});
    clearSampleTimer();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [surfaceKey]);

  const handleAction = React.useCallback(
    (action: string) => {
      if (!surface) return;
      const binding = surface.bindings.find((b) => b.action === action);
      if (!binding) return;
      const compiled = compileBinding(binding, dataModel, surface);
      setDryRun({ action, compiled });

      // Simulate: running → completed sample after a short beat, so the
      // StatusBadge/ReportRegion visibly transition like a real run. The
      // completed ref carries deterministic sample slot data so structured
      // output components (Table/Metrics/Findings/Chart) render populated.
      const samplePayload = buildSamplePayload(surface, action);
      clearSampleTimer();
      setSampleRuns((s) => {
        const next: Record<string, SampleRun> = {};
        // Any prior still-"running" sample completes instantly.
        for (const [key, run] of Object.entries(s)) {
          next[key] =
            run.ref.status === 'running'
              ? { ...run, ref: { ...run.ref, status: 'completed' } }
              : run;
        }
        next[action] = {
          ref: { status: 'running', preview: 'sample', action },
          compiled,
        };
        return next;
      });
      sampleTimerRef.current = setTimeout(() => {
        setSampleRuns((s) => {
          const run = s[action];
          if (!run || run.ref.status !== 'running') return s;
          return {
            ...s,
            [action]: {
              ...run,
              ref: {
                ...run.ref,
                status: 'completed',
                // 'sample' routes resolveCitations to the fake sample map.
                message_id: 'sample',
                ...(samplePayload ? { data: samplePayload } : {}),
              },
            },
          };
        });
      }, 700);
    },
    [surface, dataModel, clearSampleTimer],
  );

  const handleRunForReal = React.useCallback(() => {
    if (!dryRun || !previewRun) return;
    previewRun.start(dryRun.action, dryRun.compiled);
  }, [dryRun, previewRun]);

  const handleRetry = React.useCallback(
    (action: string) => {
      if (!surface || !previewRun) return;
      const binding = surface.bindings.find((b) => b.action === action);
      if (!binding) return;
      // Intentionally recompiles from the CURRENT form values.
      previewRun.start(action, compileBinding(binding, dataModel, surface));
    },
    [surface, previewRun, dataModel],
  );

  const handleGenerateDefault = React.useCallback(async () => {
    if (!ast) return;
    setScaffolding(true);
    setScaffoldError(null);
    try {
      const { surface: generated } = await scaffoldSurface(ast);
      useAgentEditorStore.getState().setAst({
        ...(ast as Record<string, unknown>),
        surface: generated,
      } as unknown as AST);
    } catch (err) {
      setScaffoldError(
        err instanceof Error
          ? err.message
          : 'Failed to generate the default UI',
      );
    } finally {
      setScaffolding(false);
    }
  }, [ast]);

  // Read-only overlay (mirror of AgentSurfacePanel): run refs are layered onto
  // the data model at each binding's output target — real run wins over sample.
  // Completed REAL refs are enriched with the persisted message's structured
  // output (results-by-reference) so slot components render populated.
  const renderedDataModel = React.useMemo<Record<string, unknown>>(() => {
    if (!surface) return dataModel;
    let model = dataModel;
    for (const binding of surface.bindings) {
      let ref =
        previewRun?.runState[binding.action] ??
        sampleRuns[binding.action]?.ref ??
        undefined;
      if (
        ref &&
        ref.status === 'completed' &&
        ref.message_id &&
        ref.message_id !== 'sample' &&
        !ref.data
      ) {
        const messageId = ref.message_id;
        const msg = previewChatFull?.messages?.find((m) => m.id === messageId);
        const structured = msg?.structuredOutput;
        if (structured && structured.binding === binding.action) {
          ref = {
            ...ref,
            data: structured.data,
            sources: structured.meta?.sources,
            slotsMeta: structured.meta?.slots,
          };
        }
      }
      if (ref !== undefined && binding.output?.target) {
        model = setAtPointer(model, binding.output.target, ref);
      }
    }
    return model;
  }, [
    surface,
    dataModel,
    sampleRuns,
    previewRun?.runState,
    previewChatFull?.messages,
  ]);

  // Citation data for structured-output cells: the 'sample' message id maps
  // to the deterministic fake citations; real ids resolve from the preview
  // chat's persisted claims — the exact machinery chat uses.
  const sampleCitations = React.useMemo(
    () =>
      buildSampleReport(agentName, { query: '', surfaceInputs: {} })
        .citationData,
    [agentName],
  );
  const resolveCitations = React.useCallback(
    (messageId: string) => {
      if (messageId === 'sample') return sampleCitations;
      const msg = previewChatFull?.messages?.find((m) => m.id === messageId);
      if (!msg || msg.claims.length === 0) return undefined;
      return buildCitationDataMap(msg.claims);
    },
    [sampleCitations, previewChatFull?.messages],
  );

  const resolveRunReference = React.useCallback(
    (ref: RunReference | null): React.ReactNode => {
      if (!ref) return null;
      const pref = ref as PreviewRunReference;

      if (pref.preview === 'sample') {
        const run = sampleRuns[pref.action];
        if (!run) return null;
        return (
          <SampleRegion
            status={pref.status}
            agentName={agentName}
            compiled={run.compiled}
          />
        );
      }

      // Real run branches (Part C).
      if (pref.status === 'running') {
        return (
          <div data-testid="surface-preview-real-running">
            <div className="mb-2 flex items-center gap-2 text-[12px] text-db-gray-text">
              <Spinner />
              <span className="capitalize">
                {previewRun?.agentStatus ?? 'running'}…
              </span>
              <button
                type="button"
                className="ml-auto rounded-md border border-db-gray-lines px-2 py-0.5 text-[11px] font-medium text-db-navy-800 hover:bg-db-gray-50"
                onClick={() => previewRun?.stop()}
              >
                Stop
              </button>
            </div>
            {isDirty && (
              <p className="mb-2 text-[11px] text-amber-700">
                Canvas has unsaved changes — this run reflects the last saved
                version.
              </p>
            )}
            {previewRun && previewRun.streamingContent.length > 0 && (
              <div className="max-h-[40vh] overflow-auto">
                <MarkdownRenderer content={previewRun.streamingContent} />
              </div>
            )}
          </div>
        );
      }

      if (pref.status === 'completed') {
        const chatId = previewRun?.previewChatId ?? null;
        const msg = pref.message_id
          ? previewChatFull?.messages?.find((m) => m.id === pref.message_id)
          : undefined;
        const content =
          msg?.content ??
          (previewRun && previewRun.streamingContent.length > 0
            ? previewRun.streamingContent
            : null);
        return (
          <div data-testid="surface-preview-real-completed">
            {content ? (
              <div className="max-h-[40vh] overflow-auto">
                <MarkdownRenderer content={content} />
              </div>
            ) : (
              <p className="text-[12px] text-db-gray-text">
                Report is loading…
              </p>
            )}
            {chatId && (
              <p className="mt-1 text-[11px]">
                <Link
                  to={`/chat/${chatId}`}
                  className="font-medium text-db-navy-800 underline hover:no-underline"
                >
                  Open full report in chat →
                </Link>
              </p>
            )}
          </div>
        );
      }

      if (pref.status === 'failed') {
        return (
          <div data-testid="surface-preview-real-failed">
            <p className="text-[12px] text-db-lava-700">
              Run failed
              {previewRun?.errorDetails
                ? `: ${previewRun.errorDetails.error.message}`
                : ''}
            </p>
            <button
              type="button"
              className="mt-1 rounded-md border border-db-gray-lines px-2 py-0.5 text-[11px] font-medium text-db-navy-800 hover:bg-db-gray-50"
              onClick={() => handleRetry(pref.action)}
            >
              Retry
            </button>
          </div>
        );
      }

      // cancelled
      return (
        <div data-testid="surface-preview-real-cancelled">
          <p className="text-[12px] text-db-gray-text">Run stopped.</p>
          <button
            type="button"
            className="mt-1 rounded-md border border-db-gray-lines px-2 py-0.5 text-[11px] font-medium text-db-navy-800 hover:bg-db-gray-50"
            onClick={() => handleRetry(pref.action)}
          >
            Retry
          </button>
        </div>
      );
    },
    [
      sampleRuns,
      agentName,
      previewRun,
      previewChatFull?.messages,
      isDirty,
      handleRetry,
    ],
  );

  if (!surface) {
    return (
      <div className="rounded-lg border border-dashed border-db-gray-lines p-8 text-center">
        <h3 className="mb-1 text-sm font-semibold text-db-navy-800">
          No UI yet
        </h3>
        <p className="mx-auto mb-4 max-w-md text-[12px] leading-[1.5] text-db-gray-text">
          This agent has no user interface. Generate the default (a form over
          the workflow’s inputs with a Run button and a results region), or ask
          the designer chat for a custom one.
        </p>
        <div className="flex items-center justify-center gap-2">
          <button
            type="button"
            disabled={!ast || scaffolding}
            onClick={handleGenerateDefault}
            className="rounded-md bg-db-navy-800 px-3 py-1.5 text-[12px] font-medium text-white hover:bg-db-navy-700 disabled:opacity-50"
          >
            {scaffolding ? 'Generating…' : 'Generate default UI'}
          </button>
          <button
            type="button"
            onClick={() =>
              useAgentEditorStore
                .getState()
                .setPendingChatSeed(
                  'Add a UI to this agent: describe the form fields and results you want.',
                )
            }
            className="rounded-md border border-db-gray-lines px-3 py-1.5 text-[12px] font-medium text-db-navy-800 hover:bg-db-gray-50"
          >
            Ask the designer
          </button>
        </div>
        {scaffoldError && (
          <p className="mt-3 text-[12px] text-db-lava-700">{scaffoldError}</p>
        )}
      </div>
    );
  }

  return (
    <div>
      <div className="mb-3 flex items-center justify-between gap-3">
        <p className="text-[12px] text-db-gray-text">
          Interactive preview — actions simulate a run and show the submission
          they would send. Nothing executes unless you use “Run for real”.
          Unsaved edits are included.
        </p>
        <div className="flex shrink-0 items-center gap-2">
          {onTryInChat && (
            <button
              type="button"
              aria-label="Try in chat"
              data-testid="surface-preview-try-in-chat"
              disabled={tryInChatPending || tryInChatDisabled}
              title={
                tryInChatDisabled
                  ? 'Name the agent before running'
                  : 'Save and open a new chat session bound to this agent'
              }
              onClick={onTryInChat}
              className="rounded-md bg-db-navy-800 px-2.5 py-1 text-[11px] font-medium text-white hover:bg-db-navy-700 disabled:cursor-not-allowed disabled:opacity-50"
            >
              {tryInChatPending ? 'Starting…' : 'Try in chat'}
            </button>
          )}
          <button
            type="button"
            className="rounded-md border border-db-gray-lines px-2.5 py-1 text-[11px] font-medium text-db-navy-800 hover:bg-db-gray-50"
            onClick={() => {
              reset(surface.data_model ?? {});
              setDryRun(null);
              setSampleRuns({});
              clearSampleTimer();
            }}
          >
            Reset values
          </button>
        </div>
      </div>
      <div className="rounded-lg border border-db-gray-lines bg-white p-4">
        <SurfaceRenderer
          surface={surface}
          dataModel={renderedDataModel}
          onDataModelChange={setValue}
          onAction={handleAction}
          actionDisabled={false}
          resolveRunReference={resolveRunReference}
          resolveCitations={resolveCitations}
          retryStructuring={retryStructuring}
        />
      </div>
      {dryRun && (
        <DryRunCard
          action={dryRun.action}
          compiled={dryRun.compiled}
          onDismiss={() => setDryRun(null)}
          onRunForReal={previewRun ? handleRunForReal : undefined}
          runForRealDisabled={previewRun?.isActive}
        />
      )}
    </div>
  );
}

export default SurfacePreviewPanel;
