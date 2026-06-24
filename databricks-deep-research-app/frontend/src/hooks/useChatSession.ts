/**
 * useChatSession — manages a streaming chat session with the Agent Designer.
 *
 * Consumes the SSE chatStream and maintains:
 *   - messages: accumulated ChatMessages (user + assistant + tool)
 *   - pendingMutations: mutations proposed by the LLM, awaiting accept/reject
 *   - isStreaming / error state
 *
 * The hook reads current AST from useAgentEditorStore.getState().ast once at
 * send time (no subscription — avoids re-render coupling during streaming).
 */

import { useReducer, useRef, useCallback, useEffect } from 'react'
import { chatStream, reconnectChatStream, type DesignerStreamChunk } from '@/api/agentDesigner'
import { ApiError } from '@/api/client'
import { useAgentEditorStore } from '@/stores/agentEditorStore'
import { astHash } from '@/lib/astHash'
import {
  createDraftWorkflow,
  isWorkflowEmpty,
  normalizeValidationErrors,
  normalizeWorkflowAst,
} from '@/lib/workflowAst'
import { applyBootstrapAgentName } from '@/lib/agentNaming'
import { clearTranscript, loadTranscript, saveTranscript } from '@/lib/designerChatPersistence'
import type {
  ChatMessage,
  DesignerAsset,
  DesignerSSEEvent,
  NormalizationFix,
  ToolCall,
} from '@/types/agentDesigner'
import type { AST, ValidationError } from '@/types/ast'

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

export interface PendingMutation {
  id: string
  /**
   * Groups all mutation events produced by a single chat turn. Multiple
   * mutating tool calls in one turn coalesce into ONE pending card keyed by
   * this id (Fix #4), so applying once yields the whole turn's net change and
   * there are no stale siblings.
   */
  turnId: string
  toolCallId: string | null
  description: string
  mutationKind: string
  oldAst: AST
  newAst: AST
  validationErrors: ValidationError[]
  /**
   * Canonical SHA-1 of the AST as it stood when the user sent the chat
   * message that produced this mutation. On Apply, we compare against
   * the *current* store AST's hash; if they diverge, the user edited
   * the canvas during streaming and the apply would silently overwrite
   * their edits. The caller surfaces a conflict prompt in that case
   * (W14 — codex-flagged designer concurrency race).
   *
   * `null` only in the legacy code path before W14 wired up hashing —
   * always populated for mutations produced after this PR ships.
   */
  baseHash: string | null
  /** Set to true when a prior apply made this mutation's base stale. */
  isStale: boolean
  /**
   * Designer auto-repair (Layer 2) records the normalizer applied to the
   * architect-emitted AST before it reached the runner. Empty when nothing
   * was rewritten — that's the common case. See `NormalizationFix` in
   * `@/types/agentDesigner` and `NormalizationFixPill` for the UI surface.
   */
  normalizationFixes: NormalizationFix[]
}

/** Result of {@link ChatSession.applyPendingMutation}. */
export type ApplyMutationResult =
  | { kind: 'applied' }
  | { kind: 'no_op' }  // mutation not found / already removed
  | {
      kind: 'conflict'
      localAst: AST
      serverAst: AST
    }

/**
 * Transient streaming status (current designer agent node, or architect-critic
 * loop iteration). Shown only while {@link ChatSession.isStreaming}; cleared
 * when the turn ends. Never persisted to the transcript.
 */
export interface ProgressInfo {
  label: string
  iteration: number | null
  total: number | null
}

export interface ChatSession {
  messages: ChatMessage[]
  pendingMutations: PendingMutation[]
  isStreaming: boolean
  /** Latest transient progress while streaming, or null. */
  progress: ProgressInfo | null
  error: string | null
  sendMessage(text: string): Promise<void>
  applyPendingMutation(id: string): Promise<ApplyMutationResult>
  rejectPendingMutation(id: string): void
  cancel(): void
  /**
   * Reset the session: abort any in-flight stream, clear the transcript +
   * pending mutations + error, and wipe the persisted transcript for this
   * session. The explicit escape hatch from any wedged state.
   */
  clearChat(): void
}

// ---------------------------------------------------------------------------
// Reducer
// ---------------------------------------------------------------------------

interface ChatState {
  messages: ChatMessage[]
  pendingMutations: PendingMutation[]
  isStreaming: boolean
  progress: ProgressInfo | null
  error: string | null
}

type ChatAction =
  | { type: 'SEND_USER'; message: ChatMessage }
  | { type: 'HYDRATE_MESSAGES'; messages: ChatMessage[] }
  | { type: 'OPEN_ASSISTANT' }
  | { type: 'APPEND_CONTENT'; delta: string }
  | { type: 'APPEND_TOOL_CALL'; toolCall: ToolCall }
  | { type: 'ADD_TOOL_RESULT'; message: ChatMessage }
  | { type: 'UPSERT_PENDING_MUTATION'; mutation: PendingMutation }
  | { type: 'REMOVE_PENDING_MUTATION'; id: string }
  | { type: 'MARK_MUTATIONS_STALE'; baseHash: string; exceptId: string }
  | { type: 'SET_ERROR'; message: string }
  | { type: 'SET_STREAMING'; value: boolean }
  | { type: 'SET_PROGRESS'; progress: ProgressInfo | null }
  | { type: 'RESET' }

function reducer(state: ChatState, action: ChatAction): ChatState {
  switch (action.type) {
    case 'SEND_USER':
      return { ...state, messages: [...state.messages, action.message] }

    case 'HYDRATE_MESSAGES':
      // Only restore into a fresh session — never clobber an in-progress one.
      if (state.messages.length > 0) return state
      return { ...state, messages: action.messages }

    case 'OPEN_ASSISTANT': {
      const assistantMsg: ChatMessage = {
        role: 'assistant',
        content: '',
        tool_calls: [],
      }
      return { ...state, messages: [...state.messages, assistantMsg] }
    }

    case 'APPEND_CONTENT': {
      const msgs = state.messages.slice()
      const last = msgs[msgs.length - 1]
      if (!last || last.role !== 'assistant') return state
      msgs[msgs.length - 1] = { ...last, content: last.content + action.delta }
      return { ...state, messages: msgs }
    }

    case 'APPEND_TOOL_CALL': {
      const msgs = state.messages.slice()
      const last = msgs[msgs.length - 1]
      if (!last || last.role !== 'assistant') return state
      const existing = last.tool_calls ?? []
      msgs[msgs.length - 1] = {
        ...last,
        tool_calls: [...existing, action.toolCall],
      }
      return { ...state, messages: msgs }
    }

    case 'ADD_TOOL_RESULT':
      return { ...state, messages: [...state.messages, action.message] }

    case 'UPSERT_PENDING_MUTATION': {
      // Coalesce all mutation events from one turn into a single card (Fix #4).
      // The first event seeds the card (id, turnId, base oldAst, baseHash,
      // description); later same-turn events only advance the net result
      // (newAst is a full cumulative snapshot from the backend's _ast_cache),
      // refresh validation against that final AST, and union auto-repair fixes.
      const idx = state.pendingMutations.findIndex((m) => m.turnId === action.mutation.turnId)
      if (idx === -1) {
        return { ...state, pendingMutations: [...state.pendingMutations, action.mutation] }
      }
      const next = state.pendingMutations.slice()
      const existing = next[idx]!
      const mergedFixes = [...existing.normalizationFixes]
      for (const fix of action.mutation.normalizationFixes) {
        if (!mergedFixes.some((f) => f.path === fix.path && f.kind === fix.kind)) {
          mergedFixes.push(fix)
        }
      }
      next[idx] = {
        ...existing,
        newAst: action.mutation.newAst,
        validationErrors: action.mutation.validationErrors,
        normalizationFixes: mergedFixes,
        // keep existing.id / turnId / oldAst (turn base) / baseHash / description
      }
      return { ...state, pendingMutations: next }
    }

    case 'REMOVE_PENDING_MUTATION':
      return {
        ...state,
        pendingMutations: state.pendingMutations.filter((m) => m.id !== action.id),
      }

    case 'MARK_MUTATIONS_STALE':
      // W14 sequential-mutation semantics: after Apply lands a mutation
      // with baseHash X, every OTHER pending mutation with that same
      // baseHash is now generated against an out-of-date snapshot. Flag
      // them stale so the UI disables Apply and prompts regenerate.
      return {
        ...state,
        pendingMutations: state.pendingMutations.map((m) =>
          m.id !== action.exceptId && m.baseHash === action.baseHash
            ? { ...m, isStale: true }
            : m,
        ),
      }

    case 'SET_ERROR':
      return { ...state, error: action.message, isStreaming: false, progress: null }

    case 'SET_STREAMING':
      // Progress is meaningful only while streaming; clear it when the turn ends.
      return { ...state, isStreaming: action.value, progress: action.value ? state.progress : null }

    case 'SET_PROGRESS':
      return { ...state, progress: action.progress }

    case 'RESET':
      return { messages: [], pendingMutations: [], isStreaming: false, progress: null, error: null }

    default:
      return state
  }
}

const initialChatState: ChatState = {
  messages: [],
  pendingMutations: [],
  isStreaming: false,
  progress: null,
  error: null,
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/**
 * Convert a camelCase/snake_case mutationKind to a short human label.
 * E.g. "add_loop_block" → "Add loop block", "proposeWorkflow" → "Propose workflow"
 */
function kindToDescription(kind: string): string {
  const spaced = kind
    .replace(/_/g, ' ')
    .replace(/([a-z])([A-Z])/g, '$1 $2')
    .toLowerCase()
  return spaced.charAt(0).toUpperCase() + spaced.slice(1)
}

const EMPTY_AST: AST = createDraftWorkflow()

// ---------------------------------------------------------------------------
// Hook
// ---------------------------------------------------------------------------

type DesignerAssetsProvider = DesignerAsset[] | (() => DesignerAsset[])

// Reconnect tuning for the designer chat. The turn runs decoupled on the server
// and survives the gateway's absolute ~4-min connection cap; we reconnect across
// it. A turn is ≤~10 min at ≤~4 min/connection, so ~3 reconnects suffice — 8 is
// safe headroom before giving up with a never-silent error.
const MAX_DESIGNER_RECONNECTS = 8
const _sleep = (ms: number): Promise<void> => new Promise((resolve) => setTimeout(resolve, ms))
function _reconnectBackoffMs(attempt: number): number {
  // Reconnect fast after a cap-cut (the turn is still running); back off only if
  // failures repeat (transient gateway/network blip). Capped at 8s.
  return Math.min(500 * 2 ** (attempt - 1), 8000)
}

export function useChatSession(opts?: {
  sessionId?: string | null
  assets?: DesignerAssetsProvider
  // Skill -> Workflow (P5): skill names to compile into the drafted workflow.
  skillNames?: string[] | (() => string[])
}): ChatSession {
  const [state, dispatch] = useReducer(reducer, initialChatState)
  // Keep a live ref to pendingMutations so applyPendingMutation can read
  // the latest list without being a stale closure over state.
  const pendingMutationsRef = useRef<PendingMutation[]>([])
  pendingMutationsRef.current = state.pendingMutations

  const abortRef = useRef<AbortController | null>(null)
  const sessionId = opts?.sessionId
  const assetsRef = useRef<DesignerAssetsProvider | undefined>(opts?.assets)
  assetsRef.current = opts?.assets
  const skillNamesRef = useRef<string[] | (() => string[]) | undefined>(opts?.skillNames)
  skillNamesRef.current = opts?.skillNames

  // Keep a live ref to isStreaming and messages for the sendMessage guard
  const isStreamingRef = useRef(false)
  isStreamingRef.current = state.isStreaming
  const messagesRef = useRef<ChatMessage[]>([])
  messagesRef.current = state.messages

  // Fix #6: restore the transcript for this session on mount / session change.
  // Only messages are restored — never pending mutations (they would be
  // applyable against a possibly-changed AST).
  useEffect(() => {
    const restored = loadTranscript(sessionId)
    if (restored && restored.length > 0) {
      dispatch({ type: 'HYDRATE_MESSAGES', messages: restored })
    }
  }, [sessionId])

  // Persist the transcript when it changes, DEBOUNCED so a burst of streaming
  // token deltas coalesces into a single localStorage write instead of one
  // synchronous write per token (F2 — per-token writes janked long turns). The
  // trailing timer fires ~400ms after the last change; the final post-turn
  // state is always captured because no later change resets the timer. The
  // timer reads the latest messages via messagesRef at fire time.
  const saveTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  useEffect(() => {
    if (saveTimerRef.current !== null) clearTimeout(saveTimerRef.current)
    saveTimerRef.current = setTimeout(() => {
      saveTimerRef.current = null
      saveTranscript(sessionId, messagesRef.current)
    }, 400)
    return () => {
      if (saveTimerRef.current !== null) {
        clearTimeout(saveTimerRef.current)
        saveTimerRef.current = null
      }
    }
  }, [sessionId, state.messages])

  // Abort any in-flight stream when the session changes or the hook unmounts,
  // so a previous session's SSE events can't land in the new session's reducer
  // (F3 — cross-session bleed). sendMessage's consume loop sees the AbortError
  // and returns cleanly; its finally clears streaming state.
  useEffect(() => {
    return () => {
      abortRef.current?.abort()
      abortRef.current = null
    }
  }, [sessionId])

  const sendMessage = useCallback(
    async (text: string): Promise<void> => {
      if (isStreamingRef.current) return
      // Claim the streaming slot SYNCHRONOUSLY, before the first `await` below,
      // so a second rapid sendMessage call can't slip past the guard and spawn
      // a duplicate stream / clobber abortRef (F1). The ref is also re-synced
      // from state on each render; this just closes the pre-render window.
      isStreamingRef.current = true

      // Snapshot current AST without subscribing to the store, and compute
      // its canonical hash (W14). Mutations produced during this turn are
      // stamped with this hash so Apply can detect divergence later.
      // Null store-state hashes to a fixed sentinel rather than crashing
      // (sendMessage may fire from a brand-new draft before the store has
      // been seeded).
      const currentAst = useAgentEditorStore.getState().ast
      const baseHash = currentAst === null ? '' : await astHash(currentAst)
      const autoApplyInitialWorkflow = isWorkflowEmpty(currentAst)

      // One id for every mutation event this turn produces, so they coalesce
      // into a single pending card (Fix #4).
      const turnId = crypto.randomUUID()

      // Empty-canvas path (Fix #4 / Codex #3): instead of auto-applying the
      // FIRST valid propose_workflow event (which left later same-turn events
      // stranded with the stale empty baseHash), accumulate the latest valid
      // AST and apply it ONCE when the turn completes.
      let initialApplyAst: AST | null = null

      // Append user message
      const userMessage: ChatMessage = { role: 'user', content: text }
      dispatch({ type: 'SEND_USER', message: userMessage })

      // Open assistant placeholder
      dispatch({ type: 'OPEN_ASSISTANT' })
      dispatch({ type: 'SET_STREAMING', value: true })

      // Build the messages list to send: current messages + new user message.
      // (The OPEN_ASSISTANT message is local UI state only — not sent to the API.)
      const allMessages: ChatMessage[] = [...messagesRef.current, userMessage]
      const assetSource = assetsRef.current
      const assets = typeof assetSource === 'function' ? assetSource() : assetSource
      const skillSource = skillNamesRef.current
      const skillNames = typeof skillSource === 'function' ? skillSource() : skillSource

      const controller = new AbortController()
      abortRef.current = controller

      // The turn runs decoupled on the server (designer_turn_registry); the
      // gateway severs a single streamed connection at an absolute ~4-min cap, so
      // a long turn may need one or more RECONNECTS to finish. turn_started gives
      // the server turn_id; each event's sequence id advances lastSeq so a resume
      // (GET …/events?since=lastSeq+1) picks up with no dup/gap.
      let serverTurnId: string | null = null
      let lastSeq: number | null = null
      let sawDone = false
      // A server-sent `error` is terminal (the turn failed); the error is already
      // surfaced, so we must NOT reconnect (it would clobber the message and
      // resume a dead turn). The backend's error path also emits `done`.
      let sawError = false

      const consume = async (stream: AsyncIterable<DesignerStreamChunk>): Promise<void> => {
        for await (const { event, seq } of stream) {
          if (event.type === 'turn_started') {
            serverTurnId = event.turn_id
            continue
          }
          if (seq !== null) lastSeq = seq
          if (event.type === 'done') sawDone = true
          if (event.type === 'error') sawError = true
          processEvent(event, dispatch, baseHash, {
            turnId,
            autoApplyInitialWorkflow,
            bootstrapPrompt: text,
            onInitialAst: (ast) => {
              initialApplyAst = ast
            },
          })
        }
      }

      try {
        let stream: AsyncIterable<DesignerStreamChunk> = chatStream({
          messages: allMessages,
          current_ast: currentAst,
          session_id: sessionId,
          assets,
          skill_names: skillNames,
          signal: controller.signal,
        })
        let reconnects = 0
        for (;;) {
          try {
            await consume(stream)
          } catch (err: unknown) {
            if (err instanceof Error && err.name === 'AbortError') return // user cancel
            if (err instanceof ApiError && err.status === 404) {
              dispatch({
                type: 'SET_ERROR',
                message: 'This designer turn expired. Please resend your message.',
              })
              return
            }
            if (err instanceof ApiError && err.status === 413) {
              // 413 is only thrown by the initial POST (it cannot be resumed).
              dispatch({ type: 'SET_ERROR', message: err.message })
              return
            }
            // Otherwise (severed socket / transient): fall through to reconnect.
          }
          if (sawDone || sawError) break
          if (
            serverTurnId === null ||
            reconnects >= MAX_DESIGNER_RECONNECTS ||
            controller.signal.aborted
          ) {
            dispatch({
              type: 'SET_ERROR',
              message: 'Lost connection to the designer. Please resend your message.',
            })
            return
          }
          reconnects += 1
          dispatch({
            type: 'SET_PROGRESS',
            progress: { label: 'Reconnecting…', iteration: null, total: null },
          })
          await _sleep(_reconnectBackoffMs(reconnects))
          if (controller.signal.aborted) return
          stream = reconnectChatStream({
            turnId: serverTurnId,
            since: (lastSeq ?? -1) + 1,
            signal: controller.signal,
          })
        }
        // Empty-canvas auto-apply happens once, after the turn's final AST is
        // known — never mid-stream (Codex #3).
        if (autoApplyInitialWorkflow && initialApplyAst) {
          useAgentEditorStore.getState().setAst(initialApplyAst)
        }
      } finally {
        dispatch({ type: 'SET_STREAMING', value: false })
        isStreamingRef.current = false
        // Only clear the shared abort ref if it still points at THIS turn's
        // controller — a newer turn or a session-change abort (F3) may have
        // already replaced it (F1/F3 hardening).
        if (abortRef.current === controller) abortRef.current = null
      }
    },
    [sessionId],
  )

  const cancel = useCallback((): void => {
    abortRef.current?.abort()
  }, [])

  const clearChat = useCallback((): void => {
    // Abort any in-flight stream, drop all local state, and wipe the persisted
    // transcript so the next send starts a clean (small) conversation. This is
    // the explicit escape hatch from any wedged session.
    abortRef.current?.abort()
    abortRef.current = null
    dispatch({ type: 'RESET' })
    clearTranscript(sessionId)
  }, [sessionId])

  const applyPendingMutation = useCallback(
    async (id: string): Promise<ApplyMutationResult> => {
      const mutation = pendingMutationsRef.current.find((m) => m.id === id)
      if (!mutation) {
        return { kind: 'no_op' }
      }

      // W14: only enforce the divergence check when the mutation actually
      // carries a baseHash. Legacy mutations from before this PR have
      // baseHash === null and fall through to the original behavior so
      // upgrading clients don't see ghost-conflict prompts.
      if (mutation.baseHash !== null) {
        const currentAst = useAgentEditorStore.getState().ast
        // When the store has no AST (e.g., brand-new draft cleared by the
        // user), there's nothing to diverge from — skip the check.
        if (currentAst !== null) {
          const currentHash = await astHash(currentAst)
          if (currentHash !== mutation.baseHash) {
            // The canvas diverged during streaming. Do NOT overwrite the
            // user's edits — let the caller surface the conflict modal so
            // they can pick local vs server (W15) or regenerate the
            // mutation against the fresh base.
            return {
              kind: 'conflict',
              localAst: currentAst,
              serverAst: mutation.newAst,
            }
          }
        }
      }

      useAgentEditorStore.getState().setAst(mutation.newAst)
      dispatch({ type: 'REMOVE_PENDING_MUTATION', id })
      // Sequential-mutation semantics: invalidate every other pending
      // mutation that was generated against the same base — they would
      // overwrite our just-applied change. Architect raised this in the
      // Phase-4 plan review; documented as no-forward-patching.
      if (mutation.baseHash !== null) {
        dispatch({
          type: 'MARK_MUTATIONS_STALE',
          baseHash: mutation.baseHash,
          exceptId: id,
        })
      }
      return { kind: 'applied' }
    },
    [],
  )

  const rejectPendingMutation = useCallback((id: string): void => {
    dispatch({ type: 'REMOVE_PENDING_MUTATION', id })
  }, [])

  return {
    messages: state.messages,
    pendingMutations: state.pendingMutations,
    isStreaming: state.isStreaming,
    progress: state.progress,
    error: state.error,
    sendMessage,
    applyPendingMutation,
    rejectPendingMutation,
    cancel,
    clearChat,
  }
}

// ---------------------------------------------------------------------------
// Event processor
// ---------------------------------------------------------------------------

function processEvent(
  event: DesignerSSEEvent,
  dispatch: React.Dispatch<ChatAction>,
  baseHash: string,
  options: {
    turnId: string
    autoApplyInitialWorkflow?: boolean
    bootstrapPrompt?: string
    /** Empty-canvas path: receives the latest valid AST to apply at turn end. */
    onInitialAst?: (ast: AST) => void
  },
): void {
  switch (event.type) {
    case 'message':
      dispatch({ type: 'APPEND_CONTENT', delta: event.content })
      break

    case 'tool_call': {
      const toolCall: ToolCall = {
        id: event.tool_call_id,
        type: 'function',
        function: {
          name: event.tool_name,
          arguments: JSON.stringify(event.args),
        },
      }
      dispatch({ type: 'APPEND_TOOL_CALL', toolCall })
      break
    }

    case 'mutation_proposed': {
      const rawNewAst = normalizeWorkflowAst(event.new_ast)
      const newAst =
        options.autoApplyInitialWorkflow === true && options.bootstrapPrompt
          ? applyBootstrapAgentName(rawNewAst, options.bootstrapPrompt)
          : rawNewAst
      const validationErrors = normalizeValidationErrors(event.validation_errors)
      const mutationKind = event.tool_name ?? 'mutation'
      if (options.autoApplyInitialWorkflow === true) {
        // Defer empty-canvas application to turn end (Codex #3). Every emit is
        // a full cumulative snapshot, so keeping the latest valid one is right.
        if (validationErrors.length === 0) {
          options.onInitialAst?.(newAst)
        }
        break
      }
      const mutation: PendingMutation = {
        id: crypto.randomUUID(),
        turnId: options.turnId,
        toolCallId: event.tool_call_id,
        description: kindToDescription(mutationKind),
        mutationKind,
        oldAst: event.old_ast ? normalizeWorkflowAst(event.old_ast) : EMPTY_AST,
        newAst,
        validationErrors,
        baseHash,
        isStale: false,
        // Older backends may omit `normalization_fixes` entirely — treat
        // undefined as empty so the UI never renders a stale pill.
        normalizationFixes: Array.isArray(event.normalization_fixes)
          ? event.normalization_fixes
          : [],
      }
      dispatch({ type: 'UPSERT_PENDING_MUTATION', mutation })
      break
    }

    case 'tool_result': {
      const toolResultMsg: ChatMessage = {
        role: 'tool',
        content: JSON.stringify(event.result),
        tool_call_id: event.tool_call_id,
        tool_name: event.tool_name,
      }
      dispatch({ type: 'ADD_TOOL_RESULT', message: toolResultMsg })
      break
    }

    case 'progress':
      dispatch({
        type: 'SET_PROGRESS',
        progress: {
          label: event.label,
          iteration: event.iteration ?? null,
          total: event.total ?? null,
        },
      })
      break

    case 'error':
      dispatch({ type: 'SET_ERROR', message: event.message })
      break

    case 'done':
      dispatch({ type: 'SET_STREAMING', value: false })
      break
  }
}
