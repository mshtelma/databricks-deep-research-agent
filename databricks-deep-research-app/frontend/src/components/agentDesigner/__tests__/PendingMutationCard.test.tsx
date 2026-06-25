/**
 * Tests for the AST-aware diff preview, stale badge, and removed-node banner
 * in PendingMutationCard (Fixes #1, #2, #5).
 */

import { render, screen, fireEvent } from '@testing-library/react';
import { describe, it, expect, vi } from 'vitest';
import '@testing-library/jest-dom';

import { PendingMutationCard } from '../PendingMutationCard';
import type { PendingMutation } from '@/hooks/useChatSession';
import { createDraftWorkflow } from '@/lib/workflowAst';
import type { AST, Block } from '@/types/ast';

function agentBlock(id: string, label: string, config: Record<string, unknown>): Block {
  return { id, type: 'agent', label, config };
}

function astWith(children: Block[], tools: AST['tools'] = []): AST {
  return {
    ...createDraftWorkflow('T'),
    id: 'wf',
    root: { id: 'root', type: 'sequence', label: 'Workflow', config: {}, children },
    tools,
  };
}

function makeMutation(overrides: Partial<PendingMutation> = {}): PendingMutation {
  return {
    id: 'mut-1',
    turnId: 'turn-1',
    toolCallId: 'tc-1',
    description: 'Update researcher',
    mutationKind: 'update_block',
    oldAst: astWith([agentBlock('a1', 'Researcher', { system_prompt: 'old' })]),
    newAst: astWith([agentBlock('a1', 'Researcher', { system_prompt: 'old' })]),
    validationErrors: [],
    baseHash: 'h1',
    isStale: false,
    normalizationFixes: [],
    ...overrides,
  };
}

describe('PendingMutationCard diff preview', () => {
  it('shows a non-zero edit count and a labeled row for a system_prompt rewrite', () => {
    const mutation = makeMutation({
      newAst: astWith([agentBlock('a1', 'Researcher', { system_prompt: 'a brand new prompt' })]),
    });
    render(<PendingMutationCard mutation={mutation} onApply={vi.fn()} onReject={vi.fn()} />);

    // Headline reflects the edit (NOT "0 edits").
    const toggle = screen.getByRole('button', { name: /View changes \(1 edit\)/ });
    fireEvent.click(toggle);
    expect(screen.getByText('System prompt')).toBeInTheDocument();
  });

  it('renders a stale badge and disables Apply when isStale', () => {
    const mutation = makeMutation({ isStale: true });
    render(<PendingMutationCard mutation={mutation} onApply={vi.fn()} onReject={vi.fn()} />);

    expect(screen.getByText('OUT OF DATE')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Apply mutation' })).toBeDisabled();
    // Reject stays enabled.
    expect(screen.getByRole('button', { name: 'Reject mutation' })).not.toBeDisabled();
  });

  it('warns when the proposal removes nodes from the current workflow', () => {
    const mutation = makeMutation({
      oldAst: astWith([agentBlock('a1', 'Keep', {}), agentBlock('a2', 'Reflector', {})]),
      newAst: astWith([agentBlock('a1', 'Keep', {})]),
    });
    render(<PendingMutationCard mutation={mutation} onApply={vi.fn()} onReject={vi.fn()} />);

    const alert = screen.getByRole('alert');
    expect(alert).toHaveTextContent('Removes 1 node');
    expect(alert).toHaveTextContent('Reflector');
  });

  it('disables Apply on validation errors', () => {
    const mutation = makeMutation({
      validationErrors: [{ message: 'bad', path: null, line: null, kind: 'validation' }],
    });
    render(<PendingMutationCard mutation={mutation} onApply={vi.fn()} onReject={vi.fn()} />);
    expect(screen.getByRole('button', { name: 'Apply mutation' })).toBeDisabled();
  });

  it('disables Apply while a previous apply is in flight', () => {
    render(
      <PendingMutationCard
        mutation={makeMutation()}
        onApply={vi.fn()}
        onReject={vi.fn()}
        applyInFlight
      />,
    );
    expect(screen.getByRole('button', { name: 'Apply mutation' })).toBeDisabled();
  });

  it('calls onApply / onReject with the mutation id', () => {
    const onApply = vi.fn();
    const onReject = vi.fn();
    render(
      <PendingMutationCard mutation={makeMutation({ id: 'm9' })} onApply={onApply} onReject={onReject} />,
    );
    fireEvent.click(screen.getByRole('button', { name: 'Apply mutation' }));
    fireEvent.click(screen.getByRole('button', { name: 'Reject mutation' }));
    expect(onApply).toHaveBeenCalledWith('m9');
    expect(onReject).toHaveBeenCalledWith('m9');
  });
});
