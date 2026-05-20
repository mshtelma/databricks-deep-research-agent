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
    error: null,
    sendMessage: vi.fn(),
    applyPendingMutation: vi.fn(),
    rejectPendingMutation: vi.fn(),
    cancel: vi.fn(),
    ...overrides,
  };
}

function makeMutation(overrides: Partial<PendingMutation> = {}): PendingMutation {
  return {
    id: 'mut-1',
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
});
