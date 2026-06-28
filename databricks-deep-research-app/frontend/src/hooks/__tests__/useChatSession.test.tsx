/**
 * Tests for useChatSession hook.
 *
 * chatStream is mocked via vi.mock so tests drive a controlled async generator.
 * agentEditorStore is reset in beforeEach.
 *
 * Semantics note (Fix #4): on a NON-empty workflow the hook proposes pending
 * mutation cards (one COALESCED card per chat turn). On an EMPTY workflow it
 * auto-applies the turn's final AST at `done`. Mechanics tests therefore seed a
 * non-empty workflow via `makeNonEmptyAst`; empty-canvas behavior is tested
 * explicitly.
 */

import { describe, it, expect, vi, beforeEach } from 'vitest'
import { renderHook, act, waitFor } from '@testing-library/react'
import { useChatSession } from '../useChatSession'
import { ApiError } from '@/api/client'
import { useAgentEditorStore, initialState } from '@/stores/agentEditorStore'
import { createDraftWorkflow } from '@/lib/workflowAst'
import type { DesignerSSEEvent } from '@/types/agentDesigner'
import type { AST } from '@/types/ast'

// ---------------------------------------------------------------------------
// Mock chatStream
// ---------------------------------------------------------------------------

vi.mock('@/api/agentDesigner', () => ({
  chatStream: vi.fn(),
  reconnectChatStream: vi.fn(),
}))

import { chatStream, reconnectChatStream } from '@/api/agentDesigner'
import type { DesignerStreamChunk } from '@/api/agentDesigner'
const mockChatStream = vi.mocked(chatStream)
const mockReconnect = vi.mocked(reconnectChatStream)

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

async function* fakeEvents(events: DesignerSSEEvent[]): AsyncIterable<DesignerStreamChunk> {
  let seq = 0
  for (const e of events) {
    // turn_started carries no sequence id; buffered events advance seq.
    yield { event: e, seq: e.type === 'turn_started' ? null : seq++ }
  }
}

/** A chatStream mock that throws (e.g. an HTTP ApiError) instead of yielding. */
async function* throwingStream(err: unknown): AsyncGenerator<DesignerStreamChunk> {
  if (err) throw err
  yield { event: { type: 'done' }, seq: 0 }
}

/** Yields the given chunks, then throws — simulates a severed connection
 *  mid-turn (the gateway's absolute cap). */
async function* chunksThenThrow(
  chunks: DesignerStreamChunk[],
  err: unknown,
): AsyncGenerator<DesignerStreamChunk> {
  for (const c of chunks) yield c
  throw err
}

/** Yields explicit chunks (with chosen sequence ids) — for resume streams. */
async function* fakeChunks(chunks: DesignerStreamChunk[]): AsyncIterable<DesignerStreamChunk> {
  for (const c of chunks) yield c
}

/** Empty-sequence workflow — `isWorkflowEmpty` returns true (auto-apply path). */
function makeAst(label = 'root'): AST {
  return {
    ...createDraftWorkflow('Test Workflow'),
    root: { id: 'r1', type: 'sequence', label, config: {}, children: [] },
  }
}

/** Non-empty workflow — drives the pending-card (edit) path. */
function makeNonEmptyAst(label = 'seed'): AST {
  return {
    ...createDraftWorkflow('Seeded Workflow'),
    root: {
      id: 'r1',
      type: 'sequence',
      label,
      config: {},
      children: [{ id: 'a1', type: 'agent', label: 'Agent', config: {} }],
    },
  }
}

function seedStore(ast: AST = makeNonEmptyAst()): void {
  act(() => {
    useAgentEditorStore.setState({ ast, isDirty: false })
  })
}

// ---------------------------------------------------------------------------
// Setup
// ---------------------------------------------------------------------------

beforeEach(() => {
  // Reset store to clean slate before each test
  useAgentEditorStore.setState(initialState)
  vi.clearAllMocks()
})

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe('useChatSession', () => {
  it('sendMessage adds the user message immediately and opens an assistant message', async () => {
    mockChatStream.mockReturnValue(fakeEvents([{ type: 'done' }]))

    const { result } = renderHook(() => useChatSession())

    await act(async () => {
      await result.current.sendMessage('Hello')
    })

    const messages = result.current.messages
    expect(messages).toHaveLength(2)
    expect(messages[0]).toMatchObject({ role: 'user', content: 'Hello' })
    expect(messages[1]).toMatchObject({ role: 'assistant', content: '' })
  })

  it('streamed message events accumulate into the assistant message content', async () => {
    mockChatStream.mockReturnValue(
      fakeEvents([
        { type: 'message', content: 'Hello' },
        { type: 'message', content: ', world' },
        { type: 'message', content: '!' },
        { type: 'done' },
      ]),
    )

    const { result } = renderHook(() => useChatSession())

    await act(async () => {
      await result.current.sendMessage('Hi')
    })

    await waitFor(() => {
      expect(result.current.isStreaming).toBe(false)
    })

    const assistant = result.current.messages.find((m) => m.role === 'assistant')
    expect(assistant?.content).toBe('Hello, world!')
  })

  it('progress events are transient: never persisted to messages and cleared at turn end', async () => {
    mockChatStream.mockReturnValue(
      fakeEvents([
        { type: 'progress', label: 'Workflow Architect (Opus)', iteration: null, total: null },
        { type: 'progress', label: 'Refining', iteration: 2, total: 4 },
        { type: 'message', content: 'Here is the proposal.' },
        { type: 'done' },
      ]),
    )

    const { result } = renderHook(() => useChatSession())

    await act(async () => {
      await result.current.sendMessage('go')
    })
    await waitFor(() => expect(result.current.isStreaming).toBe(false))

    // Cleared when the turn ends.
    expect(result.current.progress).toBeNull()
    // Never entered the transcript (must not bloat the resent payload).
    expect(result.current.messages).toHaveLength(2) // user + assistant only
    const assistant = result.current.messages.find((m) => m.role === 'assistant')
    expect(assistant?.content).toBe('Here is the proposal.')
    expect(
      result.current.messages.some((m) => (m.content ?? '').includes('Workflow Architect')),
    ).toBe(false)
  })

  it('a progress event sets transient progress while the turn is still streaming', async () => {
    let release!: () => void
    const gate = new Promise<void>((res) => {
      release = res
    })
    async function* gated(): AsyncGenerator<DesignerStreamChunk> {
      yield {
        event: { type: 'progress', label: 'Designer Critic (GPT-5)', iteration: 3, total: 4 },
        seq: 0,
      }
      await gate
      yield { event: { type: 'message', content: 'final' }, seq: 1 }
      yield { event: { type: 'done' }, seq: 2 }
    }
    mockChatStream.mockReturnValue(gated())

    const { result } = renderHook(() => useChatSession())

    let sendPromise!: Promise<void>
    act(() => {
      sendPromise = result.current.sendMessage('go')
    })

    await waitFor(() => {
      expect(result.current.progress).toEqual({
        label: 'Designer Critic (GPT-5)',
        iteration: 3,
        total: 4,
      })
    })
    expect(result.current.isStreaming).toBe(true)
    expect(
      result.current.messages.some((m) => (m.content ?? '').includes('Designer Critic')),
    ).toBe(false)

    await act(async () => {
      release()
      await sendPromise
    })

    expect(result.current.progress).toBeNull()
    expect(result.current.isStreaming).toBe(false)
    expect(result.current.messages.find((m) => m.role === 'assistant')?.content).toBe('final')
  })

  it('tool_result events preserve tool_name for UI rendering', async () => {
    mockChatStream.mockReturnValue(
      fakeEvents([
        {
          type: 'tool_result',
          tool_name: 'prompt_grounding',
          tool_call_id: 'prompt_grounding:init',
          result: { schema: 'prompt_grounding.v1', mentions_count: 1 },
        },
        { type: 'done' },
      ]),
    )

    const { result } = renderHook(() => useChatSession())

    await act(async () => {
      await result.current.sendMessage('Use the OfficeQA corpus')
    })

    await waitFor(() => {
      expect(result.current.isStreaming).toBe(false)
    })

    const toolMessage = result.current.messages.find((m) => m.role === 'tool')
    expect(toolMessage).toMatchObject({
      role: 'tool',
      tool_call_id: 'prompt_grounding:init',
      tool_name: 'prompt_grounding',
    })
    expect(JSON.parse(toolMessage!.content)).toEqual({
      schema: 'prompt_grounding.v1',
      mentions_count: 1,
    })
  })

  it('mutation_proposed events queue into pendingMutations with correct shape', async () => {
    seedStore()
    const newAst = makeAst('new')
    const oldAst = makeAst('old')

    mockChatStream.mockReturnValue(
      fakeEvents([
        {
          type: 'mutation_proposed',
          tool_call_id: 'tc-1',
          old_ast: oldAst,
          new_ast: newAst,
          validation_errors: [],
          summary: null,
        },
        { type: 'done' },
      ]),
    )

    const { result } = renderHook(() => useChatSession())

    await act(async () => {
      await result.current.sendMessage('Add a loop')
    })

    await waitFor(() => {
      expect(result.current.pendingMutations).toHaveLength(1)
    })

    const mutation = result.current.pendingMutations[0]
    expect(mutation).toBeDefined()
    expect(mutation!.toolCallId).toBe('tc-1')
    expect(mutation!.newAst).toEqual(newAst)
    expect(mutation!.oldAst).toEqual(oldAst)
    expect(mutation!.validationErrors).toEqual([])
    expect(typeof mutation!.id).toBe('string')
    expect(typeof mutation!.turnId).toBe('string')
    expect(mutation!.description).toBeTruthy()
  })

  it('surfaces an HTTP error (e.g. 413) from chatStream as session.error (never silent)', async () => {
    seedStore()
    mockChatStream.mockReturnValue(
      throwingStream(
        new ApiError(413, 'request_too_large', 'messages exceeds 20 turns (got 23)'),
      ),
    )

    const { result } = renderHook(() => useChatSession())

    await act(async () => {
      await result.current.sendMessage('use Best of N')
    })

    await waitFor(() => {
      expect(result.current.isStreaming).toBe(false)
    })
    expect(result.current.error).toBe('messages exceeds 20 turns (got 23)')
  })

  it('clearChat resets transcript + pending mutations + error', async () => {
    // Persistence clearing delegates to the pre-existing clearTranscript(); here
    // we assert the user-visible contract — the session is fully reset.
    seedStore()
    mockChatStream.mockReturnValue(
      fakeEvents([
        {
          type: 'mutation_proposed',
          tool_call_id: 'tc-1',
          old_ast: makeAst('old'),
          new_ast: makeAst('new'),
          validation_errors: [],
          summary: null,
        },
        { type: 'done' },
      ]),
    )

    const { result } = renderHook(() => useChatSession({ sessionId: 'sess-clear' }))

    await act(async () => {
      await result.current.sendMessage('Add a loop')
    })
    await waitFor(() => {
      expect(result.current.pendingMutations).toHaveLength(1)
    })
    expect(result.current.messages.length).toBeGreaterThan(0)

    act(() => {
      result.current.clearChat()
    })

    expect(result.current.messages).toHaveLength(0)
    expect(result.current.pendingMutations).toHaveLength(0)
    expect(result.current.error).toBeNull()
  })

  it('coalesces multiple same-turn mutation events into ONE card (latest AST wins)', async () => {
    seedStore()
    const v1 = makeAst('v1')
    const v2 = makeAst('v2')

    mockChatStream.mockReturnValue(
      fakeEvents([
        {
          type: 'mutation_proposed',
          tool_name: 'update_block',
          tool_call_id: 'tc-a',
          old_ast: makeNonEmptyAst(),
          new_ast: v1,
          validation_errors: [],
          summary: null,
        },
        {
          type: 'mutation_proposed',
          tool_name: 'set_model_tier',
          tool_call_id: 'tc-b',
          old_ast: v1,
          new_ast: v2,
          validation_errors: [],
          summary: null,
        },
        { type: 'done' },
      ]),
    )

    const { result } = renderHook(() => useChatSession())
    await act(async () => {
      await result.current.sendMessage('Tweak the researcher prompt and tier')
    })

    await waitFor(() => expect(result.current.pendingMutations).toHaveLength(1))
    const card = result.current.pendingMutations[0]!
    // Latest AST wins; base stays the turn's first old_ast.
    expect(card.newAst).toEqual(v2)
    expect(card.oldAst.root.children).toHaveLength(1)
  })

  it('applyPendingMutation calls setAst(newAst) and removes the entry', async () => {
    seedStore()
    const newAst = makeAst('proposed')

    mockChatStream.mockReturnValue(
      fakeEvents([
        {
          type: 'mutation_proposed',
          tool_call_id: 'tc-2',
          old_ast: makeNonEmptyAst(),
          new_ast: newAst,
          validation_errors: [],
          summary: null,
        },
        { type: 'done' },
      ]),
    )

    const setAstSpy = vi.spyOn(useAgentEditorStore.getState(), 'setAst')

    const { result } = renderHook(() => useChatSession())

    await act(async () => {
      await result.current.sendMessage('Propose something')
    })

    await waitFor(() => {
      expect(result.current.pendingMutations).toHaveLength(1)
    })

    const mutationId = result.current.pendingMutations[0]!.id

    await act(async () => {
      await result.current.applyPendingMutation(mutationId)
    })

    await waitFor(() => {
      expect(result.current.pendingMutations).toHaveLength(0)
    })

    expect(setAstSpy).toHaveBeenCalledWith(newAst)
  })

  it('auto-applies a valid initial propose_workflow mutation on an empty draft', async () => {
    const draft = createDraftWorkflow()
    const newAst = makeAst('generated')
    act(() => {
      useAgentEditorStore.setState({ ast: draft, isDirty: false })
    })

    mockChatStream.mockReturnValue(
      fakeEvents([
        {
          type: 'mutation_proposed',
          tool_name: 'propose_workflow',
          tool_call_id: 'tc-auto',
          old_ast: draft,
          new_ast: newAst,
          validation_errors: [],
          summary: null,
        },
        { type: 'done' },
      ]),
    )

    const { result } = renderHook(() => useChatSession())

    await act(async () => {
      await result.current.sendMessage('Build an investment research agent')
    })

    await waitFor(() => {
      expect(result.current.pendingMutations).toHaveLength(0)
    })
    expect(useAgentEditorStore.getState().ast?.root.label).toBe('generated')
    expect(useAgentEditorStore.getState().isDirty).toBe(true)
  })

  it('empty-draft turn applies only the FINAL coalesced AST at done (no mid-stream apply)', async () => {
    const draft = createDraftWorkflow()
    act(() => {
      useAgentEditorStore.setState({ ast: draft, isDirty: false })
    })
    const v1 = makeAst('first')
    const v2 = makeAst('final')

    const setAstSpy = vi.spyOn(useAgentEditorStore.getState(), 'setAst')

    mockChatStream.mockReturnValue(
      fakeEvents([
        {
          type: 'mutation_proposed',
          tool_name: 'propose_workflow',
          tool_call_id: 'tc-1',
          old_ast: draft,
          new_ast: v1,
          validation_errors: [],
          summary: null,
        },
        {
          type: 'mutation_proposed',
          tool_name: 'update_block',
          tool_call_id: 'tc-2',
          old_ast: v1,
          new_ast: v2,
          validation_errors: [],
          summary: null,
        },
        { type: 'done' },
      ]),
    )

    const { result } = renderHook(() => useChatSession())
    await act(async () => {
      await result.current.sendMessage('Build then refine')
    })

    await waitFor(() => expect(result.current.pendingMutations).toHaveLength(0))
    // Applied exactly once, with the final AST.
    expect(setAstSpy).toHaveBeenCalledTimes(1)
    expect(useAgentEditorStore.getState().ast?.root.label).toBe('final')
  })

  it('auto-applied bootstrap mutations replace prompt-like names with short prompt-derived names', async () => {
    const draft = createDraftWorkflow()
    const prompt =
      'Use main OfficeQA benchmark treasury chunks vector search create deep treseaury documetns'
    const newAst = {
      ...makeAst('generated'),
      name: prompt,
    }
    act(() => {
      useAgentEditorStore.setState({ ast: draft, isDirty: false })
    })

    mockChatStream.mockReturnValue(
      fakeEvents([
        {
          type: 'mutation_proposed',
          tool_name: 'propose_workflow',
          tool_call_id: 'tc-auto-name',
          old_ast: draft,
          new_ast: newAst,
          validation_errors: [],
          summary: null,
        },
        { type: 'done' },
      ]),
    )

    const { result } = renderHook(() => useChatSession())

    await act(async () => {
      await result.current.sendMessage(prompt)
    })

    expect(useAgentEditorStore.getState().ast?.name).toBe(
      'OfficeQA Treasury Documents Agent',
    )
  })

  it('rejectPendingMutation removes the entry without calling setAst', async () => {
    seedStore()
    const newAst = makeAst('proposed')

    mockChatStream.mockReturnValue(
      fakeEvents([
        {
          type: 'mutation_proposed',
          tool_call_id: 'tc-3',
          old_ast: makeNonEmptyAst(),
          new_ast: newAst,
          validation_errors: [],
          summary: null,
        },
        { type: 'done' },
      ]),
    )

    const setAstSpy = vi.spyOn(useAgentEditorStore.getState(), 'setAst')

    const { result } = renderHook(() => useChatSession())

    await act(async () => {
      await result.current.sendMessage('Propose something')
    })

    await waitFor(() => {
      expect(result.current.pendingMutations).toHaveLength(1)
    })

    const mutationId = result.current.pendingMutations[0]!.id

    act(() => {
      result.current.rejectPendingMutation(mutationId)
    })

    await waitFor(() => {
      expect(result.current.pendingMutations).toHaveLength(0)
    })

    expect(setAstSpy).not.toHaveBeenCalled()
  })

  it('cancel aborts the controller and isStreaming becomes false', async () => {
    // Simulate a stream that never resolves (we cancel it)
    let resolveStream!: () => void
    const neverDone = new Promise<void>((res) => {
      resolveStream = res
    })

    async function* hangingStream(): AsyncIterable<DesignerStreamChunk> {
      await neverDone
      yield { event: { type: 'done' }, seq: 0 }
    }

    mockChatStream.mockReturnValue(hangingStream())

    const { result } = renderHook(() => useChatSession())

    // Start streaming without awaiting (it hangs)
    let sendDone = false
    act(() => {
      void result.current.sendMessage('Hang').then(() => {
        sendDone = true
      })
    })

    // Give the hook a tick to set isStreaming = true
    await waitFor(() => {
      expect(result.current.isStreaming).toBe(true)
    })

    act(() => {
      result.current.cancel()
    })

    // Resolve the hanging promise so the generator can propagate abort
    resolveStream()

    await waitFor(() => {
      expect(result.current.isStreaming).toBe(false)
    })

    expect(sendDone).toBe(true)
  })

  it('error event sets error state and stops streaming', async () => {
    mockChatStream.mockReturnValue(
      fakeEvents([
        { type: 'message', content: 'partial' },
        { type: 'error', message: 'Something went wrong', tool_call_id: null },
      ]),
    )

    const { result } = renderHook(() => useChatSession())

    await act(async () => {
      await result.current.sendMessage('Trigger error')
    })

    await waitFor(() => {
      expect(result.current.isStreaming).toBe(false)
    })

    expect(result.current.error).toBe('Something went wrong')
    // Partial content still accumulated
    const assistant = result.current.messages.find((m) => m.role === 'assistant')
    expect(assistant?.content).toBe('partial')
  })

  it('does not auto-apply an empty-canvas mutation when the turn ends with error', async () => {
    const draft = createDraftWorkflow()
    act(() => {
      useAgentEditorStore.setState({ ast: draft, isDirty: false })
    })
    const setAstSpy = vi.spyOn(useAgentEditorStore.getState(), 'setAst')

    mockChatStream.mockReturnValue(
      fakeEvents([
        {
          type: 'mutation_proposed',
          tool_name: 'propose_workflow',
          tool_call_id: 'tc-rejected',
          old_ast: draft,
          new_ast: makeAst('rejected'),
          validation_errors: [],
          summary: null,
        },
        {
          type: 'error',
          message: 'The generated workflow did not pass review.',
          tool_call_id: null,
        },
        { type: 'done' },
      ]),
    )

    const { result } = renderHook(() => useChatSession())

    await act(async () => {
      await result.current.sendMessage('Build an agent')
    })

    expect(result.current.error).toBe('The generated workflow did not pass review.')
    expect(setAstSpy).not.toHaveBeenCalled()
    expect(useAgentEditorStore.getState().ast).toBe(draft)
  })

  // -------------------------------------------------------------------------
  // W14: chat-driven mutation race detection
  // -------------------------------------------------------------------------

  it('W14: pending mutations carry baseHash + isStale=false', async () => {
    // The base hash is derived from the store's current AST at sendMessage
    // time. Seed a non-empty workflow so the edit path runs (and hash is set).
    seedStore()

    mockChatStream.mockReturnValue(
      fakeEvents([
        {
          type: 'mutation_proposed',
          tool_call_id: 'tc-w14a',
          old_ast: null,
          new_ast: makeAst('proposed'),
          validation_errors: [],
          summary: null,
        },
        { type: 'done' },
      ]),
    )

    const { result } = renderHook(() => useChatSession())
    await act(async () => {
      await result.current.sendMessage('Propose')
    })
    await waitFor(() => expect(result.current.pendingMutations).toHaveLength(1))

    const m = result.current.pendingMutations[0]!
    expect(typeof m.baseHash).toBe('string')
    expect(m.baseHash!.length).toBe(40) // SHA-1 hex
    expect(m.isStale).toBe(false)
  })

  it('W14: apply returns conflict when canvas diverges during streaming', async () => {
    seedStore()
    mockChatStream.mockReturnValue(
      fakeEvents([
        {
          type: 'mutation_proposed',
          tool_call_id: 'tc-w14b',
          old_ast: null,
          new_ast: makeAst('proposed'),
          validation_errors: [],
          summary: null,
        },
        { type: 'done' },
      ]),
    )

    const { result } = renderHook(() => useChatSession())
    await act(async () => {
      await result.current.sendMessage('Propose')
    })
    await waitFor(() => expect(result.current.pendingMutations).toHaveLength(1))
    const mutationId = result.current.pendingMutations[0]!.id

    // Simulate the user editing the canvas during streaming.
    act(() => {
      useAgentEditorStore.getState().setAst(makeAst('user-edited'))
    })

    let outcome
    await act(async () => {
      outcome = await result.current.applyPendingMutation(mutationId)
    })

    expect(outcome).toMatchObject({ kind: 'conflict' })
    // Mutation must NOT be removed on conflict — caller decides whether
    // to keep it for retry or drop after the user resolves.
    expect(result.current.pendingMutations).toHaveLength(1)
  })

  it('W14: applying one turn marks a later turn (same baseHash) stale', async () => {
    // Coalescing means same-turn events merge into one card, so the stale
    // scenario now spans TWO turns that share a send-time snapshot.
    seedStore()
    const newAstA = makeAst('A')
    const newAstB = makeAst('B')

    mockChatStream.mockReturnValueOnce(
      fakeEvents([
        {
          type: 'mutation_proposed',
          tool_call_id: 'tc-A',
          old_ast: null,
          new_ast: newAstA,
          validation_errors: [],
          summary: null,
        },
        { type: 'done' },
      ]),
    )
    mockChatStream.mockReturnValueOnce(
      fakeEvents([
        {
          type: 'mutation_proposed',
          tool_call_id: 'tc-B',
          old_ast: null,
          new_ast: newAstB,
          validation_errors: [],
          summary: null,
        },
        { type: 'done' },
      ]),
    )

    const { result } = renderHook(() => useChatSession())
    await act(async () => {
      await result.current.sendMessage('First change')
    })
    await act(async () => {
      await result.current.sendMessage('Second change')
    })
    await waitFor(() => expect(result.current.pendingMutations).toHaveLength(2))

    // Both turns share the same baseHash (store unchanged between sends).
    expect(result.current.pendingMutations[0]!.baseHash).toBe(
      result.current.pendingMutations[1]!.baseHash,
    )

    const firstId = result.current.pendingMutations[0]!.id
    let outcome
    await act(async () => {
      outcome = await result.current.applyPendingMutation(firstId)
    })
    expect(outcome).toEqual({ kind: 'applied' })

    await waitFor(() => {
      expect(result.current.pendingMutations).toHaveLength(1)
      expect(result.current.pendingMutations[0]!.isStale).toBe(true)
    })
  })

  // -- Reconnect across the gateway's absolute connection cap ----------------

  it('reconnects after a severed connection and resumes the turn from lastSeq+1', async () => {
    seedStore()
    // Connection 1: turn_started (turn_id) + a message at seq 0, then the socket dies.
    mockChatStream.mockReturnValue(
      chunksThenThrow(
        [
          { event: { type: 'turn_started', turn_id: 't-123' }, seq: null },
          { event: { type: 'message', content: 'first ' }, seq: 0 },
        ],
        new TypeError('network error'),
      ),
    )
    // Reconnect delivers the rest of the buffered turn (seq 1) + done.
    mockReconnect.mockReturnValue(
      fakeChunks([
        { event: { type: 'message', content: 'second' }, seq: 1 },
        { event: { type: 'done' }, seq: 2 },
      ]),
    )

    const { result } = renderHook(() => useChatSession())
    await act(async () => {
      await result.current.sendMessage('go')
    })

    await waitFor(() => expect(result.current.isStreaming).toBe(false))
    expect(result.current.error).toBeNull()
    expect(mockReconnect).toHaveBeenCalledTimes(1)
    expect(mockReconnect).toHaveBeenCalledWith(
      expect.objectContaining({ turnId: 't-123', since: 1 }),
    )
    const assistant = result.current.messages.find((m) => m.role === 'assistant')
    expect(assistant?.content).toBe('first second')
  })

  it('shows a graceful error when the turn has expired on reconnect (404)', async () => {
    seedStore()
    mockChatStream.mockReturnValue(
      chunksThenThrow(
        [{ event: { type: 'turn_started', turn_id: 't-x' }, seq: null }],
        new TypeError('network error'),
      ),
    )
    mockReconnect.mockReturnValue(
      throwingStream(new ApiError(404, 'turn_not_found', 'Designer turn not found or expired.')),
    )

    const { result } = renderHook(() => useChatSession())
    await act(async () => {
      await result.current.sendMessage('go')
    })
    await waitFor(() => expect(result.current.isStreaming).toBe(false))
    expect(result.current.error).toMatch(/expired/i)
  })

  it('does not reconnect when the turn completes on the first connection', async () => {
    seedStore()
    mockChatStream.mockReturnValue(
      fakeEvents([
        { type: 'turn_started', turn_id: 't-1' },
        { type: 'message', content: 'done quickly' },
        { type: 'done' },
      ]),
    )
    const { result } = renderHook(() => useChatSession())
    await act(async () => {
      await result.current.sendMessage('go')
    })
    await waitFor(() => expect(result.current.isStreaming).toBe(false))
    expect(mockReconnect).not.toHaveBeenCalled()
    expect(result.current.error).toBeNull()
  })

  // -------------------------------------------------------------------------
  // F1: double-submit race — a second concurrent send must be ignored.
  // -------------------------------------------------------------------------
  it('ignores a second concurrent sendMessage while streaming (F1: one stream)', async () => {
    seedStore()
    // Fresh generator per call so a duplicate (unguarded) call wouldn't reuse a
    // spent iterator — we assert purely on the call COUNT.
    mockChatStream.mockImplementation(() =>
      fakeEvents([
        { type: 'turn_started', turn_id: 't-f1' },
        { type: 'message', content: 'hi' },
        { type: 'done' },
      ]),
    )
    const { result } = renderHook(() => useChatSession())
    await act(async () => {
      const p1 = result.current.sendMessage('first')
      const p2 = result.current.sendMessage('second') // must hit the guard
      await Promise.all([p1, p2])
    })
    // Without the synchronous isStreamingRef claim, both calls slip past the
    // guard (the ref only re-syncs from state on the next render) and chatStream
    // is invoked twice.
    expect(mockChatStream).toHaveBeenCalledTimes(1)
  })

  // -------------------------------------------------------------------------
  // F3: a sessionId change mid-stream must abort the in-flight stream so the
  // previous session's events cannot bleed into the new session.
  // -------------------------------------------------------------------------
  it('aborts the in-flight stream when sessionId changes (F3: no cross-session bleed)', async () => {
    seedStore()
    let capturedSignal: AbortSignal | undefined
    async function* hanging(): AsyncIterable<DesignerStreamChunk> {
      yield { event: { type: 'turn_started', turn_id: 't-f3' }, seq: null }
      yield { event: { type: 'message', content: 'partial' }, seq: 0 }
      await new Promise<void>(() => {}) // never resolves — turn stays in flight
    }
    mockChatStream.mockImplementation((args) => {
      capturedSignal = args.signal
      return hanging()
    })
    const { result, rerender } = renderHook(
      ({ sid }: { sid: string }) => useChatSession({ sessionId: sid }),
      { initialProps: { sid: 's1' } },
    )
    act(() => {
      void result.current.sendMessage('hi')
    })
    await waitFor(() => expect(capturedSignal).toBeDefined())
    expect(capturedSignal!.aborted).toBe(false)
    rerender({ sid: 's2' }) // session change → effect cleanup aborts the stream
    expect(capturedSignal!.aborted).toBe(true)
  })
})
