/**
 * Tests for useChatSession hook.
 *
 * chatStream is mocked via vi.mock so tests drive a controlled async generator.
 * agentEditorStore is reset in beforeEach.
 */

import { describe, it, expect, vi, beforeEach } from 'vitest'
import { renderHook, act, waitFor } from '@testing-library/react'
import { useChatSession } from '../useChatSession'
import { useAgentEditorStore, initialState } from '@/stores/agentEditorStore'
import { createDraftWorkflow } from '@/lib/workflowAst'
import type { DesignerSSEEvent } from '@/types/agentDesigner'
import type { AST } from '@/types/ast'

// ---------------------------------------------------------------------------
// Mock chatStream
// ---------------------------------------------------------------------------

vi.mock('@/api/agentDesigner', () => ({
  chatStream: vi.fn(),
}))

import { chatStream } from '@/api/agentDesigner'
const mockChatStream = vi.mocked(chatStream)

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

async function* fakeEvents(events: DesignerSSEEvent[]): AsyncIterable<DesignerSSEEvent> {
  for (const e of events) {
    yield e
  }
}

function makeAst(label = 'root'): AST {
  return {
    ...createDraftWorkflow('Test Workflow'),
    root: { id: 'r1', type: 'sequence', label, config: {}, children: [] },
  }
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
    expect(mutation!.description).toBeTruthy()
  })

  it('applyPendingMutation calls setAst(newAst) and removes the entry', async () => {
    const newAst = makeAst('proposed')

    mockChatStream.mockReturnValue(
      fakeEvents([
        {
          type: 'mutation_proposed',
          tool_call_id: 'tc-2',
          old_ast: null,
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
      result.current.applyPendingMutation(mutationId)
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
    const newAst = makeAst('proposed')

    mockChatStream.mockReturnValue(
      fakeEvents([
        {
          type: 'mutation_proposed',
          tool_call_id: 'tc-3',
          old_ast: null,
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

    async function* hangingStream(): AsyncIterable<DesignerSSEEvent> {
      await neverDone
      yield { type: 'done' }
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

  // -------------------------------------------------------------------------
  // W14: chat-driven mutation race detection
  // -------------------------------------------------------------------------

  it('W14: pending mutations carry baseHash + isStale=false', async () => {
    // The base hash is derived from the store's current AST at sendMessage
    // time. Seed the store so the hash is non-empty.
    act(() => {
      useAgentEditorStore.getState().setAst(makeAst('seed'))
    })

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
    expect(m.baseHash!.length).toBe(40)  // SHA-1 hex
    expect(m.isStale).toBe(false)
  })

  it('W14: apply returns conflict when canvas diverges during streaming', async () => {
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

  it('W14: apply marks sibling mutations as stale after success', async () => {
    const newAstA = makeAst('A')
    const newAstB = makeAst('B')
    mockChatStream.mockReturnValue(
      fakeEvents([
        {
          type: 'mutation_proposed',
          tool_call_id: 'tc-A',
          old_ast: null,
          new_ast: newAstA,
          validation_errors: [],
          summary: null,
        },
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
      await result.current.sendMessage('Two proposals')
    })
    await waitFor(() => expect(result.current.pendingMutations).toHaveLength(2))

    // Both proposals share the same baseHash (same send-time snapshot).
    expect(result.current.pendingMutations[0]!.baseHash).toBe(
      result.current.pendingMutations[1]!.baseHash,
    )

    const firstId = result.current.pendingMutations[0]!.id
    let outcome
    await act(async () => {
      outcome = await result.current.applyPendingMutation(firstId)
    })
    expect(outcome).toEqual({ kind: 'applied' })

    // The remaining mutation is flagged stale — applying it would
    // overwrite the just-landed edit (no forward-patching scope).
    await waitFor(() => {
      expect(result.current.pendingMutations).toHaveLength(1)
      expect(result.current.pendingMutations[0]!.isStale).toBe(true)
    })
  })
})
