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

import { useReducer, useRef, useCallback } from 'react'
import { chatStream } from '@/api/agentDesigner'
import { useAgentEditorStore } from '@/stores/agentEditorStore'
import { astHash } from '@/lib/astHash'
import {
  createDraftWorkflow,
  isWorkflowEmpty,
  normalizeValidationErrors,
  normalizeWorkflowAst,
} from '@/lib/workflowAst'
import { applyBootstrapAgentName } from '@/lib/agentNaming'
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

export interface ChatSession {
  messages: ChatMessage[]
  pendingMutations: PendingMutation[]
  isStreaming: boolean
  error: string | null
  sendMessage(text: string): Promise<void>
  applyPendingMutation(id: string): Promise<ApplyMutationResult>
  rejectPendingMutation(id: string): void
  cancel(): void
}

// ---------------------------------------------------------------------------
// Reducer
// ---------------------------------------------------------------------------

interface ChatState {
  messages: ChatMessage[]
  pendingMutations: PendingMutation[]
  isStreaming: boolean
  error: string | null
}

type ChatAction =
  | { type: 'SEND_USER'; message: ChatMessage }
  | { type: 'OPEN_ASSISTANT' }
  | { type: 'APPEND_CONTENT'; delta: string }
  | { type: 'APPEND_TOOL_CALL'; toolCall: ToolCall }
  | { type: 'ADD_TOOL_RESULT'; message: ChatMessage }
  | { type: 'ADD_PENDING_MUTATION'; mutation: PendingMutation }
  | { type: 'REMOVE_PENDING_MUTATION'; id: string }
  | { type: 'MARK_MUTATIONS_STALE'; baseHash: string; exceptId: string }
  | { type: 'SET_ERROR'; message: string }
  | { type: 'SET_STREAMING'; value: boolean }

function reducer(state: ChatState, action: ChatAction): ChatState {
  switch (action.type) {
    case 'SEND_USER':
      return { ...state, messages: [...state.messages, action.message] }

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

    case 'ADD_PENDING_MUTATION':
      return {
        ...state,
        pendingMutations: [...state.pendingMutations, action.mutation],
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
      return { ...state, error: action.message, isStreaming: false }

    case 'SET_STREAMING':
      return { ...state, isStreaming: action.value }

    default:
      return state
  }
}

const initialChatState: ChatState = {
  messages: [],
  pendingMutations: [],
  isStreaming: false,
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

export function useChatSession(opts?: {
  sessionId?: string | null
  assets?: DesignerAssetsProvider
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

  // Keep a live ref to isStreaming and messages for the sendMessage guard
  const isStreamingRef = useRef(false)
  isStreamingRef.current = state.isStreaming
  const messagesRef = useRef<ChatMessage[]>([])
  messagesRef.current = state.messages

  const sendMessage = useCallback(
    async (text: string): Promise<void> => {
      if (isStreamingRef.current) return

      // Snapshot current AST without subscribing to the store, and compute
      // its canonical hash (W14). Mutations produced during this turn are
      // stamped with this hash so Apply can detect divergence later.
      // Null store-state hashes to a fixed sentinel rather than crashing
      // (sendMessage may fire from a brand-new draft before the store has
      // been seeded).
      const currentAst = useAgentEditorStore.getState().ast
      const baseHash = currentAst === null ? '' : await astHash(currentAst)
      const autoApplyInitialWorkflow = isWorkflowEmpty(currentAst)

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

      const controller = new AbortController()
      abortRef.current = controller

      try {
        const stream = chatStream({
          messages: allMessages,
          current_ast: currentAst,
          session_id: sessionId,
          assets,
          signal: controller.signal,
        })

        for await (const event of stream) {
          processEvent(event, dispatch, baseHash, {
            autoApplyInitialWorkflow,
            bootstrapPrompt: text,
          })
        }
      } catch (err: unknown) {
        if (err instanceof Error && err.name === 'AbortError') {
          // Cancelled — just fall through to finally
        } else {
          const msg = err instanceof Error ? err.message : String(err)
          dispatch({ type: 'SET_ERROR', message: msg })
          return
        }
      } finally {
        dispatch({ type: 'SET_STREAMING', value: false })
        abortRef.current = null
      }
    },
    [sessionId],
  )

  const cancel = useCallback((): void => {
    abortRef.current?.abort()
  }, [])

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
    error: state.error,
    sendMessage,
    applyPendingMutation,
    rejectPendingMutation,
    cancel,
  }
}

// ---------------------------------------------------------------------------
// Event processor
// ---------------------------------------------------------------------------

function processEvent(
  event: DesignerSSEEvent,
  dispatch: React.Dispatch<ChatAction>,
  baseHash: string,
  options: { autoApplyInitialWorkflow?: boolean; bootstrapPrompt?: string } = {},
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
      if (
        options.autoApplyInitialWorkflow === true &&
        mutationKind === 'propose_workflow' &&
        validationErrors.length === 0
      ) {
        useAgentEditorStore.getState().setAst(newAst)
        break
      }
      const mutation: PendingMutation = {
        id: crypto.randomUUID(),
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
      dispatch({ type: 'ADD_PENDING_MUTATION', mutation })
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

    case 'error':
      dispatch({ type: 'SET_ERROR', message: event.message })
      break

    case 'done':
      dispatch({ type: 'SET_STREAMING', value: false })
      break
  }
}
