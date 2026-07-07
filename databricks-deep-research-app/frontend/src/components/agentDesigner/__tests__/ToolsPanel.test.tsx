/**
 * Tests for ToolsPanel + AddToolDialog + BindToolDialog.
 *
 * Radix Dialog portals render into document.body in jsdom — use
 * screen.getByRole('dialog') to assert dialog presence after trigger clicks.
 *
 * Store is reset in beforeEach via setState with initialState merged so that
 * action functions remain intact (no second `true` arg to setState).
 */

import { describe, it, expect, beforeEach, vi } from 'vitest';
import * as React from 'react';
import { render, screen, fireEvent } from '@testing-library/react';
import '@testing-library/jest-dom';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';

import { ToolsPanel } from '../ToolsPanel';
import { useAgentEditorStore, initialState } from '@/stores/agentEditorStore';
import { createDraftWorkflow } from '@/lib/workflowAst';
import type { RegistryResponse } from '@/types/agentDesigner';
import type { AST } from '@/types/ast';

vi.mock('@/api/agentDesigner', async (importOriginal) => {
  const actual = await importOriginal<typeof import('@/api/agentDesigner')>();
  return {
    ...actual,
    listDesignerResources: vi.fn().mockResolvedValue({
      resources: [
        {
          kind: 'vector_index',
          source_id: 'idx-1',
          name: 'Customer Index',
          full_name: 'main.sales.customer_index',
          description: 'Customer vector index',
          status: 'ready',
          capabilities: [],
          metadata: { index_name: 'main.sales.customer_index' },
        },
      ],
      total: 1,
    }),
  };
});

// ---------------------------------------------------------------------------
// Fixture registry — 3 tool_kinds across 3 layers (A, B, C)
// ---------------------------------------------------------------------------

const FIXTURE_REGISTRY: RegistryResponse = {
  node_types: [
    {
      type: 'agent',
      label: 'Agent',
      icon: '🤖',
      category: 'core',
      is_composite: false,
      config_schema: null,
    },
    {
      type: 'sequence',
      label: 'Sequence',
      icon: '🔢',
      category: 'control_flow',
      is_composite: true,
      config_schema: null,
    },
  ],
  agent_subtypes: [],
  tool_kinds: [
    // Layer A (index 0)
    { kind: 'web_search', label: 'Web Search', icon: '🔍' },
    // Layer B (index 3)
    {
      kind: 'vector_search',
      label: 'Vector Search',
      icon: '🗂️',
      config_schema: {
        type: 'object',
        properties: {
          index_name: {
            type: 'string',
            title: 'Vector Search Index',
            'x-widget': 'resource-select',
            'x-source-kind': 'vector_index',
            'x-value-field': 'full_name',
          },
          num_results: { type: 'integer', default: 10 },
        },
        required: ['index_name'],
      },
    },
    // Layer C (index 6)
    { kind: 'genie', label: 'Genie', icon: '🧞' },
  ],
  model_tiers: ['simple', 'analytical', 'complex'],
  version: '1.0.0',
};

// ---------------------------------------------------------------------------
// AST helpers
// ---------------------------------------------------------------------------

function makeAst(overrides: Partial<AST> = {}): AST {
  return {
    ...createDraftWorkflow('Test Workflow'),
    root: {
      id: 'root-id',
      type: 'sequence',
      label: 'Root',
      config: {},
      children: [],
    },
    ...overrides,
  };
}

function makeAstWithAgent(): AST {
  return makeAst({
    root: {
      id: 'root-id',
      type: 'sequence',
      label: 'Root',
      config: {},
      children: [
        {
          id: 'agent-1',
          type: 'agent',
          label: 'My Agent',
          config: {},
          children: [],
        },
      ],
    },
  });
}

function renderWithQuery(ui: React.ReactElement) {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false }, mutations: { retry: false } },
  });
  return render(<QueryClientProvider client={client}>{ui}</QueryClientProvider>);
}

// ---------------------------------------------------------------------------
// Store reset
// ---------------------------------------------------------------------------

beforeEach(() => {
  useAgentEditorStore.setState({ ...initialState });
});

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe('ToolsPanel', () => {
  // 1. Empty state
  it('renders empty state when no tools are declared', () => {
    useAgentEditorStore.setState({ ast: makeAst(), selectedPath: null });
    render(<ToolsPanel registry={FIXTURE_REGISTRY} />);
    expect(screen.getByText('No tools declared yet')).toBeInTheDocument();
  });

  // 2. Tool list with layer-coloured badges
  it('renders tool list with kind badges for each declared tool', () => {
    const ast = makeAst({
      tools: [
        { kind: 'web_search', name: 'ws1', config: {} },
        { kind: 'vector_search', name: 'vs1', config: {} },
      ],
    });
    useAgentEditorStore.setState({ ast, selectedPath: null });
    render(<ToolsPanel registry={FIXTURE_REGISTRY} />);

    // Both tool names appear
    expect(screen.getByText('ws1')).toBeInTheDocument();
    expect(screen.getByText('vs1')).toBeInTheDocument();

    // Badges with kind label text
    const badges = screen.getAllByText('web_search');
    expect(badges.length).toBeGreaterThanOrEqual(1);
    // web_search is index 0 → layer A → bg-db-blue-100
    const wsBadge = badges[0]!;
    expect(wsBadge.className).toMatch(/bg-db-blue-100/);

    // vector_search is index 1 → still layer A (index 1, tier=0) → bg-db-blue-100
    const vsBadge = screen.getByText('vector_search');
    expect(vsBadge.className).toMatch(/bg-db-blue-100/);
  });

  // 3. Add Tool button opens AddToolDialog
  it('Add Tool button opens AddToolDialog', () => {
    useAgentEditorStore.setState({ ast: makeAst(), selectedPath: null });
    render(<ToolsPanel registry={FIXTURE_REGISTRY} />);

    fireEvent.click(screen.getByRole('button', { name: /add tool/i }));

    // Radix Dialog renders to document.body portal
    const dialog = screen.getByRole('dialog');
    expect(dialog).toBeInTheDocument();
    // Dialog title is rendered as an h2 inside the dialog
    expect(dialog.querySelector('h2')).toHaveTextContent('Add tool');
  });

  // 4. AddToolDialog submit calls declareTool with correct args
  it('AddToolDialog submit calls declareTool with selected kind and name', () => {
    useAgentEditorStore.setState({ ast: makeAst(), selectedPath: null });
    const declareSpy = vi.spyOn(useAgentEditorStore.getState(), 'declareTool');

    render(<ToolsPanel registry={FIXTURE_REGISTRY} />);
    fireEvent.click(screen.getByRole('button', { name: /add tool/i }));

    // Dialog opens on the shared declaration picker; Web Search is the default kind.
    expect(screen.getByRole('combobox', { name: /tool family/i })).toBeInTheDocument();
    expect(screen.getByRole('combobox', { name: /tool kind/i })).toHaveValue('web_search');

    // Name field is auto-filled with "web_search"
    const nameInput = screen.getByRole('textbox');
    expect((nameInput as HTMLInputElement).value).toBe('web_search');

    // Submit via the dialog's footer button (not the panel's "Add Tool" trigger)
    const dialog = screen.getByRole('dialog');
    // Find the enabled submit button inside the dialog footer
    const footerBtns = Array.from(dialog.querySelectorAll('button'));
    const addBtn = footerBtns.find((b) => b.textContent?.trim() === 'Add tool' && !b.disabled);
    fireEvent.click(addBtn!);

    expect(declareSpy).toHaveBeenCalledWith('web_search', 'web_search', {});
    declareSpy.mockRestore();
  });

  it('AddToolDialog submits vector search with configured index and defaults', () => {
    useAgentEditorStore.setState({ ast: makeAst(), selectedPath: null });
    const declareSpy = vi.spyOn(useAgentEditorStore.getState(), 'declareTool');

    renderWithQuery(<ToolsPanel registry={FIXTURE_REGISTRY} />);
    fireEvent.click(screen.getByRole('button', { name: /add tool/i }));
    fireEvent.change(screen.getByRole('combobox', { name: /tool family/i }), {
      target: { value: 'databricks' },
    });
    fireEvent.change(screen.getByRole('combobox', { name: /tool kind/i }), {
      target: { value: 'vector_search' },
    });

    fireEvent.change(screen.getByRole('combobox', { name: /vector search index/i }), {
      target: { value: 'main.sales.customer_index' },
    });

    const dialog = screen.getByRole('dialog');
    const addBtn = Array.from(dialog.querySelectorAll('button')).find(
      (b) => b.textContent?.trim() === 'Add tool' && !b.disabled,
    );
    fireEvent.click(addBtn!);

    expect(declareSpy).toHaveBeenCalledWith('vector_search', 'vector_search', {
      index_name: 'main.sales.customer_index',
      num_results: 10,
    });
    declareSpy.mockRestore();
  });

  it('AddToolDialog requires vector search index before submit', () => {
    useAgentEditorStore.setState({ ast: makeAst(), selectedPath: null });
    const declareSpy = vi.spyOn(useAgentEditorStore.getState(), 'declareTool');

    renderWithQuery(<ToolsPanel registry={FIXTURE_REGISTRY} />);
    fireEvent.click(screen.getByRole('button', { name: /add tool/i }));
    fireEvent.change(screen.getByRole('combobox', { name: /tool family/i }), {
      target: { value: 'databricks' },
    });
    fireEvent.change(screen.getByRole('combobox', { name: /tool kind/i }), {
      target: { value: 'vector_search' },
    });

    const dialog = screen.getByRole('dialog');
    const addBtn = Array.from(dialog.querySelectorAll('button')).find(
      (b) => b.textContent?.trim() === 'Add tool' && !b.disabled,
    );
    fireEvent.click(addBtn!);

    expect(screen.getByText('Required')).toBeInTheDocument();
    expect(declareSpy).not.toHaveBeenCalled();
    declareSpy.mockRestore();
  });

  // 5. AddToolDialog rejects duplicate names
  it('AddToolDialog shows error and does not call declareTool again for duplicate name', () => {
    const ast = makeAst({
      tools: [{ kind: 'web_search', name: 'web_search', config: {} }],
    });
    useAgentEditorStore.setState({ ast, selectedPath: null });
    const declareSpy = vi.spyOn(useAgentEditorStore.getState(), 'declareTool');

    render(<ToolsPanel registry={FIXTURE_REGISTRY} />);
    fireEvent.click(screen.getByRole('button', { name: /add tool/i }));

    // Web Search is the default kind; name auto-fills to "web_search" (duplicate).
    expect(screen.getByRole('combobox', { name: /tool kind/i })).toHaveValue('web_search');

    // Submit via the dialog's enabled footer "Add Tool" button
    const dialog = screen.getByRole('dialog');
    const footerBtns = Array.from(dialog.querySelectorAll('button'));
    const addBtn = footerBtns.find((b) => b.textContent?.trim() === 'Add tool' && !b.disabled);
    fireEvent.click(addBtn!);

    // Error message must be visible
    expect(screen.getByRole('alert')).toHaveTextContent('Tool name already exists');

    // declareTool was called once (it returned false), not called a second time
    expect(declareSpy).toHaveBeenCalledTimes(1);
    declareSpy.mockRestore();
  });

  // 6. Bind Tools is disabled when no agent is selected
  it('Bind Tools button is disabled when no agent block is selected', () => {
    useAgentEditorStore.setState({ ast: makeAst(), selectedPath: null });
    render(<ToolsPanel registry={FIXTURE_REGISTRY} />);

    const bindBtn = screen.getByRole('button', { name: /bind tools/i });
    expect(bindBtn).toBeDisabled();
  });

  // 7. Bind Tools opens BindToolDialog when agent is selected
  it('Bind Tools button opens BindToolDialog when an agent block is selected', () => {
    const ast = makeAstWithAgent();
    useAgentEditorStore.setState({ ast, selectedPath: 'root.children.0' });
    render(<ToolsPanel registry={FIXTURE_REGISTRY} />);

    const bindBtn = screen.getByRole('button', { name: /bind tools/i });
    expect(bindBtn).not.toBeDisabled();

    fireEvent.click(bindBtn);

    expect(screen.getByRole('dialog')).toBeInTheDocument();
    expect(screen.getByText('Bind tools to agent')).toBeInTheDocument();
  });

  // 8. BindToolDialog submit calls updateBlock with selected tool names
  it('BindToolDialog submit calls updateBlock with the checked tool names', () => {
    const ast = makeAst({
      tools: [
        { kind: 'web_search', name: 'ws1', config: {} },
        { kind: 'vector_search', name: 'vs1', config: {} },
      ],
      root: {
        id: 'root-id',
        type: 'sequence',
        label: 'Root',
        config: {},
        children: [
          {
            id: 'agent-1',
            type: 'agent',
            label: 'My Agent',
            config: {},
            children: [],
          },
        ],
      },
    });
    useAgentEditorStore.setState({ ast, selectedPath: 'root.children.0' });
    const updateSpy = vi.spyOn(useAgentEditorStore.getState(), 'updateBlock');

    render(<ToolsPanel registry={FIXTURE_REGISTRY} />);
    fireEvent.click(screen.getByRole('button', { name: /bind tools/i }));

    // Dialog is open — check "ws1"
    const ws1Checkbox = screen.getByRole('checkbox', { name: /ws1/i });
    fireEvent.click(ws1Checkbox);

    // Submit
    fireEvent.click(screen.getByRole('button', { name: /apply/i }));

    expect(updateSpy).toHaveBeenCalledWith(
      'root.children.0',
      expect.objectContaining({
        config: expect.objectContaining({ tools: expect.arrayContaining(['ws1']) }),
      }),
    );
    updateSpy.mockRestore();
  });
});
