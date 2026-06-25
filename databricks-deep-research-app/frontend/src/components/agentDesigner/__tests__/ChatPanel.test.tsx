/**
 * Tests for ChatPanel and PendingMutationCard components.
 *
 * useChatSession is mocked via vi.mock so each test can inject its own session state.
 */

import { render, screen, fireEvent } from '@testing-library/react';
import { describe, it, expect, vi, beforeEach } from 'vitest';
import '@testing-library/jest-dom';

import { ChatPanel } from '../ChatPanel';
import { createDraftWorkflow } from '@/lib/workflowAst';
import type { ChatSession, PendingMutation } from '@/hooks/useChatSession';
import type { ChatMessage } from '@/types/agentDesigner';
import type { AST } from '@/types/ast';

// ---------------------------------------------------------------------------
// Mock useChatSession
// ---------------------------------------------------------------------------

vi.mock('@/hooks/useChatSession', () => ({
  useChatSession: vi.fn(),
}));

// We import after vi.mock so the module binding is the mock.
import { useChatSession } from '@/hooks/useChatSession';

const mockUseChatSession = vi.mocked(useChatSession);

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

const EMPTY_AST: AST = createDraftWorkflow();

function fakeSession(overrides: Partial<ChatSession> = {}): ChatSession {
  return {
    messages: [],
    pendingMutations: [],
    isStreaming: false,
    progress: null,
    error: null,
    sendMessage: vi.fn(),
    applyPendingMutation: vi.fn(),
    rejectPendingMutation: vi.fn(),
    cancel: vi.fn(),
    clearChat: vi.fn(),
    ...overrides,
  };
}

function makeMutation(overrides: Partial<PendingMutation> = {}): PendingMutation {
  return {
    id: 'mut-1',
    turnId: 'turn-1',
    toolCallId: 'tc-1',
    description: 'Add loop block',
    mutationKind: 'add_loop_block',
    oldAst: EMPTY_AST,
    newAst: EMPTY_AST,
    validationErrors: [],
    // W14 fields — keep `baseHash: null` so legacy fixtures bypass the
    // race-detection check by default; individual tests can override.
    baseHash: null,
    isStale: false,
    // Layer 2 auto-repair — default empty list so the NormalizationFixPill
    // does NOT render for unrelated mutation tests.
    normalizationFixes: [],
    ...overrides,
  };
}

function makeUserMessage(content: string): ChatMessage {
  return { role: 'user', content };
}

function makeAssistantMessage(content: string): ChatMessage {
  return { role: 'assistant', content, tool_calls: [] };
}

function makeToolMessage(toolName: string, payload: Record<string, unknown>): ChatMessage {
  return {
    role: 'tool',
    content: JSON.stringify(payload),
    tool_call_id: `${toolName}:init`,
    tool_name: toolName,
  };
}

beforeEach(() => {
  mockUseChatSession.mockReset();
});

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe('ChatPanel', () => {
  it('renders empty transcript when messages is empty', () => {
    mockUseChatSession.mockReturnValue(fakeSession());
    render(<ChatPanel />);

    expect(screen.getByText('Designer Chat')).toBeInTheDocument();
    // No message bubbles
    expect(screen.queryByRole('alert')).not.toBeInTheDocument();
  });

  it('renders user messages on the right and assistant messages on the left', () => {
    mockUseChatSession.mockReturnValue(
      fakeSession({
        messages: [
          makeUserMessage('Hello from user'),
          makeAssistantMessage('Hello from assistant'),
        ],
      }),
    );

    render(<ChatPanel />);

    const userContainer = screen.getByText('Hello from user').closest('[data-role="user"]');
    expect(userContainer).toHaveClass('justify-end');

    const assistantContainer = screen
      .getByText('Hello from assistant')
      .closest('[data-role="assistant"]');
    expect(assistantContainer).toHaveClass('justify-start');
  });

  it('shows streaming indicator while isStreaming is true', () => {
    mockUseChatSession.mockReturnValue(fakeSession({ isStreaming: true }));
    render(<ChatPanel />);

    expect(screen.getByTestId('streaming-indicator')).toBeInTheDocument();
  });

  it('does not show streaming indicator when isStreaming is false', () => {
    mockUseChatSession.mockReturnValue(fakeSession({ isStreaming: false }));
    render(<ChatPanel />);

    expect(screen.queryByTestId('streaming-indicator')).not.toBeInTheDocument();
  });

  it('renders the live progress indicator (label + iteration) while streaming', () => {
    mockUseChatSession.mockReturnValue(
      fakeSession({ isStreaming: true, progress: { label: 'Refining', iteration: 2, total: 4 } }),
    );
    render(<ChatPanel />);

    const indicator = screen.getByTestId('progress-indicator');
    expect(indicator).toBeInTheDocument();
    expect(indicator).toHaveTextContent('Refining · 2/4');
  });

  it('does not render the progress indicator once streaming ends', () => {
    mockUseChatSession.mockReturnValue(
      fakeSession({ isStreaming: false, progress: { label: 'Refining', iteration: 2, total: 4 } }),
    );
    render(<ChatPanel />);

    expect(screen.queryByTestId('progress-indicator')).not.toBeInTheDocument();
  });

  it('renders PendingMutationCard for each entry in pendingMutations', () => {
    const mutations = [
      makeMutation({ id: 'mut-1', description: 'Add loop block' }),
      makeMutation({ id: 'mut-2', description: 'Add agent block' }),
    ];
    mockUseChatSession.mockReturnValue(fakeSession({ pendingMutations: mutations }));

    render(<ChatPanel />);

    expect(screen.getByText('Add loop block')).toBeInTheDocument();
    expect(screen.getByText('Add agent block')).toBeInTheDocument();
  });

  it('PendingMutationCard Apply button is disabled when validationErrors.length > 0', () => {
    const mutation = makeMutation({
      validationErrors: [{ message: 'Invalid node', path: null, line: null, kind: 'validation' }],
    });
    mockUseChatSession.mockReturnValue(fakeSession({ pendingMutations: [mutation] }));

    render(<ChatPanel />);

    const applyBtn = screen.getByRole('button', { name: 'Apply mutation' });
    expect(applyBtn).toBeDisabled();
  });

  it('PendingMutationCard Apply click calls applyPendingMutation with the mutation id', () => {
    const session = fakeSession({ pendingMutations: [makeMutation({ id: 'mut-42' })] });
    mockUseChatSession.mockReturnValue(session);

    render(<ChatPanel />);

    fireEvent.click(screen.getByRole('button', { name: 'Apply mutation' }));
    expect(session.applyPendingMutation).toHaveBeenCalledWith('mut-42');
  });

  it('PendingMutationCard Reject click calls rejectPendingMutation with the mutation id', () => {
    const session = fakeSession({ pendingMutations: [makeMutation({ id: 'mut-99' })] });
    mockUseChatSession.mockReturnValue(session);

    render(<ChatPanel />);

    fireEvent.click(screen.getByRole('button', { name: 'Reject mutation' }));
    expect(session.rejectPendingMutation).toHaveBeenCalledWith('mut-99');
  });

  it('shows error banner when session.error is non-null', () => {
    mockUseChatSession.mockReturnValue(fakeSession({ error: 'Something went wrong' }));
    render(<ChatPanel />);

    const banner = screen.getByRole('alert');
    expect(banner).toBeInTheDocument();
    expect(banner).toHaveTextContent('Something went wrong');
  });

  it('Clear chat button calls clearChat when there is something to clear', () => {
    const clearChat = vi.fn();
    mockUseChatSession.mockReturnValue(
      fakeSession({ messages: [makeUserMessage('hi')], clearChat }),
    );
    render(<ChatPanel />);

    const btn = screen.getByTestId('clear-chat-button');
    expect(btn).not.toBeDisabled();
    fireEvent.click(btn);
    expect(clearChat).toHaveBeenCalledTimes(1);
  });

  it('Clear chat button is disabled when there is nothing to clear', () => {
    mockUseChatSession.mockReturnValue(fakeSession());
    render(<ChatPanel />);

    expect(screen.getByTestId('clear-chat-button')).toBeDisabled();
  });

  it('send on Enter calls sendMessage with the input text', () => {
    const session = fakeSession();
    mockUseChatSession.mockReturnValue(session);

    render(<ChatPanel />);

    const textarea = screen.getByRole('textbox', { name: 'Chat input' });
    fireEvent.change(textarea, { target: { value: 'update the workflow' } });
    fireEvent.keyDown(textarea, { key: 'Enter', shiftKey: false });

    expect(session.sendMessage).toHaveBeenCalledWith('update the workflow');
  });

  it('Shift+Enter does NOT submit the form', () => {
    const session = fakeSession();
    mockUseChatSession.mockReturnValue(session);

    render(<ChatPanel />);

    const textarea = screen.getByRole('textbox', { name: 'Chat input' });
    fireEvent.change(textarea, { target: { value: 'multiline text' } });
    fireEvent.keyDown(textarea, { key: 'Enter', shiftKey: true });

    expect(session.sendMessage).not.toHaveBeenCalled();
  });

  it('summarizes prompt grounding results as source-check progress', () => {
    mockUseChatSession.mockReturnValue(
      fakeSession({
        messages: [
          makeToolMessage('prompt_grounding', {
            schema: 'prompt_grounding.v1',
            mentions_count: 1,
            resolved_assets_count: 1,
            resolved_resources: [
              {
                kind: 'vector_index',
                identity: 'officeqa_benchmark.treasury.chunks',
                usage: 'required',
              },
            ],
            resource_kinds: { vector_index: 1 },
            ready_tool_kinds: ['vector_search'],
            safe_to_build_blueprint: true,
            diagnostics: [
              {
                severity: 'warning',
                code: 'missing_semantics',
                message: 'Run semantic lookup on the grounded vector index.',
                blocking: false,
              },
            ],
          }),
        ],
      }),
    );

    render(<ChatPanel />);

    expect(screen.getByText('Checked selected sources')).toBeInTheDocument();
    expect(
      screen.getByText('Found 1 grounded vector index for this workflow.'),
    ).toBeInTheDocument();
    expect(screen.getByText('1 mention')).toBeInTheDocument();
    expect(screen.getByText('1 source')).toBeInTheDocument();
    expect(screen.getByText('Vector search ready')).toBeInTheDocument();
    expect(screen.getByText('Run semantic lookup on the grounded vector index.')).toBeInTheDocument();
    expect(screen.queryByText(/prompt_grounding\.v1/)).not.toBeInTheDocument();
  });

  it('keeps raw Designer JSON hidden until technical details are expanded', () => {
    mockUseChatSession.mockReturnValue(
      fakeSession({
        messages: [
          makeToolMessage('resolved_tool_contract', {
            schema: 'resolved_tool_contract.v1',
            available: true,
            evidence_policy: 'corpus_only',
            resources_count: 1,
            required_capabilities: ['vector_search'],
            ready_tool_kinds: ['vector_search'],
            required_terms: ['officeqa', 'treasury'],
            planner_obligations: ['Run semantic lookup on the grounded vector index.'],
            diagnostics: [],
          }),
        ],
      }),
    );

    render(<ChatPanel />);

    expect(screen.getByText('Planned evidence access')).toBeInTheDocument();
    expect(
      screen.getByText('Designer will answer from the named corpus using vector search.'),
    ).toBeInTheDocument();
    expect(screen.getByText('Corpus-only evidence')).toBeInTheDocument();
    expect(screen.queryByText(/resolved_tool_contract\.v1/)).not.toBeInTheDocument();

    fireEvent.click(screen.getByRole('button', { name: /technical details/i }));
    expect(screen.getByText(/resolved_tool_contract\.v1/)).toBeInTheDocument();
  });

  it('renders unavailable resource semantics as neutral process context', () => {
    mockUseChatSession.mockReturnValue(
      fakeSession({
        messages: [
          makeToolMessage('resource_semantics', {
            schema: 'resource_semantics.v1',
            available: false,
          }),
        ],
      }),
    );

    render(<ChatPanel />);

    expect(screen.getByText('Checked data semantics')).toBeInTheDocument();
    expect(
      screen.getByText(
        'No extra semantic profile was available, so Designer will use the grounded source metadata.',
      ),
    ).toBeInTheDocument();
    expect(screen.getByText('Source metadata')).toBeInTheDocument();
    expect(screen.queryByText(/resource_semantics\.v1/)).not.toBeInTheDocument();
  });

  it('opens the conflict modal when apply returns a conflict (Fix #3)', async () => {
    const localAst = createDraftWorkflow();
    const serverAst = createDraftWorkflow();
    const session = fakeSession({
      pendingMutations: [makeMutation({ id: 'mc1' })],
      applyPendingMutation: vi.fn().mockResolvedValue({ kind: 'conflict', localAst, serverAst }),
    });
    mockUseChatSession.mockReturnValue(session);

    render(<ChatPanel />);
    fireEvent.click(screen.getByRole('button', { name: 'Apply mutation' }));

    expect(session.applyPendingMutation).toHaveBeenCalledWith('mc1');
    expect(await screen.findByText('You edited the canvas during this change')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Apply the proposed change' })).toBeInTheDocument();
  });

  it('shows a no-op notice when apply returns no_op (Fix #3)', async () => {
    const session = fakeSession({
      pendingMutations: [makeMutation({ id: 'mc2' })],
      applyPendingMutation: vi.fn().mockResolvedValue({ kind: 'no_op' }),
    });
    mockUseChatSession.mockReturnValue(session);

    render(<ChatPanel />);
    fireEvent.click(screen.getByRole('button', { name: 'Apply mutation' }));

    expect(await screen.findByTestId('apply-noop-notice')).toBeInTheDocument();
  });
});
