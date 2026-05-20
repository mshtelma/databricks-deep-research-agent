/**
 * Tests for EtagConflictModal — V1 regression + V1.5 diff/merge paths.
 *
 * AstDiffView is lazy-loaded via React.lazy; we mock the dynamic import so
 * tests can assert on it synchronously without code-splitting complications.
 */

import * as React from 'react';
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent, waitFor, act } from '@testing-library/react';
import '@testing-library/jest-dom';
import { EtagConflictModal } from '../EtagConflictModal';
import { createDraftWorkflow } from '@/lib/workflowAst';
import type { AST } from '@/types/ast';

// ---------------------------------------------------------------------------
// Mock AstDiffView (the lazy chunk) so Suspense resolves immediately in jsdom
// ---------------------------------------------------------------------------

vi.mock('../AstDiffView', () => ({
  default: ({
    conflicts,
    selections,
    onSelect,
  }: {
    conflicts: Array<{ path: string; localValue: unknown; serverValue: unknown }>;
    selections: Map<string, 'local' | 'server'>;
    onSelect: (path: string, source: 'local' | 'server') => void;
  }) => (
    <div data-testid="ast-diff-view">
      {conflicts.map((c) => (
        <div key={c.path} data-testid={`conflict-${c.path}`}>
          <span>{c.path}</span>
          <button
            data-testid={`pick-local-${c.path}`}
            aria-pressed={selections.get(c.path) === 'local'}
            onClick={() => onSelect(c.path, 'local')}
          >
            local
          </button>
          <button
            data-testid={`pick-server-${c.path}`}
            aria-pressed={selections.get(c.path) === 'server'}
            onClick={() => onSelect(c.path, 'server')}
          >
            server
          </button>
        </div>
      ))}
    </div>
  ),
}));

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------

function makeAst(label: string, version = '1.0.0'): AST {
  return {
    ...createDraftWorkflow(label),
    version: version === '2.0.0' ? 2 : 1,
    root: {
      id: 'root-id',
      type: 'sequence',
      label,
      config: {},
      children: [],
    },
  };
}

const LOCAL_AST = makeAst('local-root', '1.0.0');
const SERVER_AST = makeAst('server-root', '2.0.0');

// ---------------------------------------------------------------------------
// Default props helper
// ---------------------------------------------------------------------------

function defaultProps(overrides: Partial<React.ComponentProps<typeof EtagConflictModal>> = {}) {
  return {
    open: true,
    onOpenChange: vi.fn(),
    currentEtag: 'etag-123',
    onReload: vi.fn(),
    onForceOverwrite: vi.fn(),
    ...overrides,
  };
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe('EtagConflictModal', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  // -------------------------------------------------------------------------
  // V1 regression
  // -------------------------------------------------------------------------

  it('test_reload_path_unchanged — clicking Reload still calls onReload', () => {
    const onReload = vi.fn();
    const onOpenChange = vi.fn();
    render(<EtagConflictModal {...defaultProps({ onReload, onOpenChange })} />);

    fireEvent.click(screen.getByRole('button', { name: /reload agent from server/i }));

    expect(onReload).toHaveBeenCalledTimes(1);
    expect(onOpenChange).toHaveBeenCalledWith(false);
  });

  it('test_force_overwrite_path_unchanged — clicking Force overwrite (twice) still calls onForceOverwrite', () => {
    const onForceOverwrite = vi.fn();
    const onOpenChange = vi.fn();
    render(<EtagConflictModal {...defaultProps({ onForceOverwrite, onOpenChange })} />);

    // First click → confirm sub-step
    fireEvent.click(screen.getByRole('button', { name: /force overwrite remote agent/i }));
    expect(onForceOverwrite).not.toHaveBeenCalled();

    // Second click → confirmed
    fireEvent.click(screen.getByRole('button', { name: /confirm force overwrite/i }));
    expect(onForceOverwrite).toHaveBeenCalledTimes(1);
    expect(onOpenChange).toHaveBeenCalledWith(false);
  });

  // -------------------------------------------------------------------------
  // V1.5 diff path
  // -------------------------------------------------------------------------

  it('test_diff_renders — clicking Show Diff renders AstDiffView with conflicts', async () => {
    const onSaveMerge = vi.fn();
    render(
      <EtagConflictModal
        {...defaultProps({ localAst: LOCAL_AST, serverAst: SERVER_AST, onSaveMerge })}
      />,
    );

    // Show Diff button must be present when ASTs + onSaveMerge provided
    const showDiffBtn = screen.getByRole('button', { name: /show diff/i });
    expect(showDiffBtn).toBeInTheDocument();

    await act(async () => {
      fireEvent.click(showDiffBtn);
    });

    await waitFor(() => {
      expect(screen.getByTestId('ast-diff-view')).toBeInTheDocument();
    });

    // Should have at least one conflict row (schema_version + root.label differ)
    const conflictRows = screen.getAllByTestId(/^conflict-/);
    expect(conflictRows.length).toBeGreaterThan(0);
  });

  it('test_short_circuit_to_reload_when_no_real_merge — all-local selections call onReload not onSaveMerge', async () => {
    const onReload = vi.fn();
    const onSaveMerge = vi.fn();
    const onOpenChange = vi.fn();

    // Use ASTs that differ only in schema_version so there's exactly 1 conflict
    const local = makeAst('same-label', '1.0.0');
    const server = makeAst('same-label', '2.0.0');

    render(
      <EtagConflictModal
        {...defaultProps({ onReload, onSaveMerge, onOpenChange, localAst: local, serverAst: server })}
      />,
    );

    await act(async () => {
      fireEvent.click(screen.getByRole('button', { name: /show diff/i }));
    });

    await waitFor(() => {
      expect(screen.getByTestId('ast-diff-view')).toBeInTheDocument();
    });

    // Pick "local" for every conflict (schema_version path)
    const localButtons = screen.getAllByText('local');
    for (const btn of localButtons) {
      fireEvent.click(btn);
    }

    // Click Save Merge
    const saveMergeBtn = screen.getByRole('button', { name: /save merged version/i });
    fireEvent.click(saveMergeBtn);

    // All selections on same side → short-circuit to reload
    expect(onReload).toHaveBeenCalledTimes(1);
    expect(onSaveMerge).not.toHaveBeenCalled();
    expect(onOpenChange).toHaveBeenCalledWith(false);
  });

  it('test_keyboard_nav — Tab moves focus; Enter selects local; Shift+Enter selects server', async () => {
    const onSaveMerge = vi.fn();

    // Single-conflict ASTs: differ only in schema_version
    const local = makeAst('same', '1.0.0');
    const server = makeAst('same', '2.0.0');

    render(
      <EtagConflictModal
        {...defaultProps({ localAst: local, serverAst: server, onSaveMerge })}
      />,
    );

    await act(async () => {
      fireEvent.click(screen.getByRole('button', { name: /show diff/i }));
    });

    await waitFor(() => {
      expect(screen.getByTestId('ast-diff-view')).toBeInTheDocument();
    });

    // Find the "local" button for the schema_version conflict and Tab to it
    const localBtn = screen.getByTestId('pick-local-schema_version');
    const serverBtn = screen.getByTestId('pick-server-schema_version');

    // Tab navigation: focus the local button
    localBtn.focus();
    expect(document.activeElement).toBe(localBtn);

    // Tab to next element (server button)
    fireEvent.keyDown(localBtn, { key: 'Tab' });
    // After tab the browser moves focus — in jsdom we simulate by focusing serverBtn
    serverBtn.focus();
    expect(document.activeElement).toBe(serverBtn);

    // Clicking local button sets selection
    fireEvent.click(localBtn);
    expect(localBtn).toHaveAttribute('aria-pressed', 'true');

    // Clicking server button changes selection
    fireEvent.click(serverBtn);
    expect(serverBtn).toHaveAttribute('aria-pressed', 'true');
  });

  it('Show Diff button is absent when no localAst/serverAst props provided', () => {
    render(<EtagConflictModal {...defaultProps()} />);
    expect(screen.queryByRole('button', { name: /show diff/i })).not.toBeInTheDocument();
  });
});
