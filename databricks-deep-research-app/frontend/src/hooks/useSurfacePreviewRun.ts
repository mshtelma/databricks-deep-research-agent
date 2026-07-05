/**
 * Page-level controller for "Run for real" from the Designer Preview tab.
 *
 * Executes the SAVED agent inside a real, visibly-titled chat
 * ("Preview run — <agent>") and streams the report back into the preview's
 * ReportRegion. Lives in DesignerInner (NOT the Preview tab panel): the tab
 * content unmounts on every tab switch and useStreamingQuery closes its SSE
 * on unmount — the page-level hook keeps the stream alive across tabs.
 * Leaving /designer abandons the stream only; the job continues server-side
 * and remains fully visible at /chat/{previewChatId}.
 *
 * State machine (single-flight):
 *   IDLE → start() → SAVING (ensureSavedAgentId) → ENSURING_CHAT (create once,
 *   reused per visit) → SUBMIT_PENDING (staged; consumed by effect once
 *   useStreamingQuery is bound to the chat id) → SUBMITTING → STREAMING →
 *   completed | failed | cancelled (kept per-action in runState).
 */

import { useState, useCallback, useRef, useEffect, useMemo } from 'react';
import { useQueryClient } from '@tanstack/react-query';
import { chatsApi } from '../api/client';
import * as clientMetrics from '@/lib/clientMetrics';
import { useStreamingQuery, type ErrorDetails } from './useStreamingQuery';
import { CHAT_FULL_KEY } from './useChatFull';
import type { CompiledSubmission } from '@/lib/surfaceCompile';
import type { RunReference } from '@/types/surface';
import type { ResearchDepth } from '@/components/chat/ResearchDepthSelector';

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/**
 * A RunReference branded with its origin so a single resolveRunReference
 * callback can discriminate sample vs real refs. Extra fields are inert for
 * the catalog components (they only read `.status`).
 */
export interface PreviewRunReference extends RunReference {
  preview: 'sample' | 'real';
  /** The binding action this ref belongs to — routes Retry/Stop. */
  action: string;
}

export interface SurfacePreviewRunOptions {
  /** Route agent id; null when the designer is on /designer/new. */
  agentId: string | null;
  /** Agent display name — used for the preview chat title. */
  agentName: string;
  /**
   * Save-first gate, owned by DesignerInner (reuses the Test-run idiom):
   * saves when new/dirty and resolves the saved agent id, or null when the
   * save failed (banners/modals already shown by saveMutation's onError).
   */
  ensureSavedAgentId: () => Promise<string | null>;
}

export interface SurfacePreviewRunApi {
  /** Per-action real-run refs (running/completed/failed/cancelled). */
  runState: Record<string, PreviewRunReference | null>;
  /** True while a real run is staged, submitting, or streaming. */
  isActive: boolean;
  /** Live report text while streaming (and final-report fallback). */
  streamingContent: string;
  /** Coarse agent status label (idle|planning|researching|...|complete|error). */
  agentStatus: string;
  /** Submission/stream error details for the failed-state UI. */
  errorDetails: ErrorDetails | null;
  /** The real chat backing preview runs (created lazily, reused per visit). */
  previewChatId: string | null;
  /** Kick off a real run for a compiled binding. Errors land in runState. */
  start: (action: string, compiled: CompiledSubmission) => void;
  /** Cancel the in-flight run (also cancels the server-side job). */
  stop: () => void;
}

interface PendingSubmission {
  token: number;
  agentId: string;
  action: string;
  compiled: CompiledSubmission;
}

// ---------------------------------------------------------------------------
// Hook
// ---------------------------------------------------------------------------

export function useSurfacePreviewRun(
  opts: SurfacePreviewRunOptions,
): SurfacePreviewRunApi {
  const queryClient = useQueryClient();

  // Options via ref: start() closures always see current name/save gate.
  const optsRef = useRef(opts);
  optsRef.current = opts;

  const [previewChatId, setPreviewChatId] = useState<string | null>(null);
  const previewChatIdRef = useRef<string | null>(null);

  const [runState, setRunState] = useState<
    Record<string, PreviewRunReference | null>
  >({});
  const [pending, setPending] = useState<PendingSubmission | null>(null);
  /** Chat-create failures happen before the streaming hook is involved. */
  const [localError, setLocalError] = useState<ErrorDetails | null>(null);

  // Synchronous re-entrancy guard for the click → setState gap.
  const startInFlightRef = useRef(false);
  // Action currently streaming; nulled on terminal transitions.
  const inFlightActionRef = useRef<string | null>(null);
  // Last started action; survives terminal transitions so a late
  // persistence_completed can still enrich the ref with message/session ids.
  const lastActionRef = useRef<string | null>(null);
  // One-shot guards (refs survive StrictMode's simulated remount).
  const firedTokenRef = useRef<number | null>(null);
  const tokenCounterRef = useRef(0);
  const handledPersistenceRef = useRef<unknown>(null);

  const markFailed = useCallback((action: string | null) => {
    if (!action) return;
    setRunState((s) => ({
      ...s,
      [action]: { status: 'failed', preview: 'real', action },
    }));
    inFlightActionRef.current = null;
  }, []);

  const onJobSubmissionError = useCallback(
    (_error: Error) => {
      // errorDetails from the streaming hook carries the classified message
      // (e.g. 429 MAX_CONCURRENT_JOBS); we only flip the ref here.
      markFailed(inFlightActionRef.current ?? lastActionRef.current);
    },
    [markFailed],
  );

  const {
    sendQuery,
    stopStream,
    isStreaming,
    streamingContent,
    agentStatus,
    persistenceResult,
    errorDetails: streamErrorDetails,
  } = useStreamingQuery(previewChatId ?? undefined, { onJobSubmissionError });

  const isActive = useMemo(
    () =>
      pending !== null ||
      isStreaming ||
      Object.values(runState).some(
        (r) => r?.preview === 'real' && r.status === 'running',
      ),
    [pending, isStreaming, runState],
  );
  const isActiveRef = useRef(isActive);
  isActiveRef.current = isActive;

  // -------------------------------------------------------------------------
  // start(): save-first → ensure chat → stage submission
  // -------------------------------------------------------------------------

  const start = useCallback(
    (action: string, compiled: CompiledSubmission) => {
      if (startInFlightRef.current || isActiveRef.current) return;
      startInFlightRef.current = true;
      setLocalError(null);
      void (async () => {
        try {
          const agentId = await optsRef.current.ensureSavedAgentId();
          if (!agentId) return; // save failed — banners already visible

          let chatId = previewChatIdRef.current;
          if (!chatId) {
            try {
              const chat = await chatsApi.create({
                title: `Preview run — ${optsRef.current.agentName}`,
              });
              chatId = chat.id;
              previewChatIdRef.current = chatId;
              setPreviewChatId(chatId);
              void queryClient.invalidateQueries({ queryKey: ['chats'] });
            } catch (err) {
              setLocalError({
                error:
                  err instanceof Error
                    ? err
                    : new Error('Failed to create the preview chat'),
              });
              markFailed(action);
              return;
            }
          }

          clientMetrics.emit('surface_preview_real_run', undefined, {
            agent_id: agentId,
          });

          inFlightActionRef.current = action;
          lastActionRef.current = action;
          setRunState((s) => ({
            ...s,
            [action]: { status: 'running', preview: 'real', action },
          }));
          tokenCounterRef.current += 1;
          setPending({
            token: tokenCounterRef.current,
            agentId,
            action,
            compiled,
          });
        } finally {
          startInFlightRef.current = false;
        }
      })();
    },
    [queryClient, markFailed],
  );

  // Consume the staged submission once useStreamingQuery is bound to the
  // preview chat (sendQuery bails without a chatId). The token ref makes the
  // send one-shot under StrictMode's double-invoked effects.
  useEffect(() => {
    if (!pending || !previewChatId) return;
    if (firedTokenRef.current === pending.token) return;
    firedTokenRef.current = pending.token;
    const { agentId, compiled } = pending;
    setPending(null); // runState 'running' keeps isActive true through the gap
    void sendQuery({
      message: compiled.query,
      queryMode: 'deep_research',
      agentId,
      surfaceInputs: compiled.surfaceInputs,
      researchDepth: compiled.researchDepth as ResearchDepth | undefined,
      verifySources: compiled.verifySources,
      turnIntent: 'research',
    });
  }, [pending, previewChatId, sendQuery]);

  // -------------------------------------------------------------------------
  // Terminal transitions (mirrors ChatPage's persistence/error effect pair)
  // -------------------------------------------------------------------------

  // persistence_completed → completed ref enriched with message/session ids.
  // Keyed on lastActionRef (not inFlightActionRef) so a completed-without-ids
  // fallback that already fired can still be enriched by late persistence.
  useEffect(() => {
    if (!persistenceResult) return;
    if (handledPersistenceRef.current === persistenceResult) return;
    handledPersistenceRef.current = persistenceResult;
    const action = inFlightActionRef.current ?? lastActionRef.current;
    if (!action) return;
    setRunState((s) => ({
      ...s,
      [action]: {
        status: 'completed',
        preview: 'real',
        action,
        message_id: persistenceResult.messageId,
        session_id: persistenceResult.researchSessionId,
      },
    }));
    inFlightActionRef.current = null;
    const cid = persistenceResult.chatId;
    void queryClient.invalidateQueries({ queryKey: ['messages', cid] });
    void queryClient.invalidateQueries({ queryKey: [...CHAT_FULL_KEY, cid] });
    void queryClient.invalidateQueries({ queryKey: ['chats'] });
  }, [persistenceResult, queryClient]);

  // Status fallbacks: complete without persistence (dropped event) and error.
  useEffect(() => {
    const action = inFlightActionRef.current;
    if (!action) return;
    if (agentStatus === 'complete' && !persistenceResult) {
      // Keep lastActionRef so late persistence can still enrich with ids.
      setRunState((s) => ({
        ...s,
        [action]: { status: 'completed', preview: 'real', action },
      }));
      inFlightActionRef.current = null;
    } else if (agentStatus === 'error') {
      markFailed(action);
    }
  }, [agentStatus, persistenceResult, markFailed]);

  // -------------------------------------------------------------------------
  // stop()
  // -------------------------------------------------------------------------

  const stop = useCallback(() => {
    stopStream();
    setPending(null); // staged-but-unsent runs are cancelled too
    const action = inFlightActionRef.current ?? lastActionRef.current;
    if (action) {
      setRunState((s) => ({
        ...s,
        [action]: { status: 'cancelled', preview: 'real', action },
      }));
    }
    inFlightActionRef.current = null;
  }, [stopStream]);

  return {
    runState,
    isActive,
    streamingContent,
    agentStatus,
    errorDetails: localError ?? streamErrorDetails,
    previewChatId,
    start,
    stop,
  };
}
