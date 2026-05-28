/**
 * Tests for ConfigPanel + SchemaField.
 *
 * Uses a minimal fixture registry with one node type whose config_schema
 * covers text, number, boolean, enum, and x-widget=unknown cases.
 */

import { describe, it, expect, beforeEach, vi } from 'vitest';
import * as React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import '@testing-library/jest-dom';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { ConfigPanel } from '../ConfigPanel';
import { useAgentEditorStore, initialState } from '@/stores/agentEditorStore';
import { createDraftWorkflow } from '@/lib/workflowAst';
import { listDesignerResources, startDesignerSqlWarehouse } from '@/api/agentDesigner';
import type { RegistryResponse } from '@/types/agentDesigner';
import type { AST } from '@/types/ast';

vi.mock('@/api/agentDesigner', async (importOriginal) => {
  const actual = await importOriginal<typeof import('@/api/agentDesigner')>();
  return {
    ...actual,
    listDesignerResources: vi.fn().mockResolvedValue({
      resources: [],
      total: 0,
    }),
    startDesignerSqlWarehouse: vi.fn().mockResolvedValue({
      kind: 'sql_warehouse',
      source_id: 'wh-default',
      name: 'Default Warehouse',
      full_name: 'Default Warehouse',
      description: null,
      status: 'STARTING',
      capabilities: ['sql'],
      metadata: { warehouse_id: 'wh-default', state: 'STARTING' },
    }),
  };
});

// ---------------------------------------------------------------------------
// Fixture registry
// ---------------------------------------------------------------------------

const FIXTURE_REGISTRY: RegistryResponse = {
  node_types: [
    {
      type: 'agent',
      label: 'Agent',
      icon: 'agent',
      category: 'core',
      is_composite: false,
      config_schema: {
        type: 'object',
        properties: {
          name: {
            type: 'string',
            title: 'Name',
            description: 'Agent display name',
            minLength: 1,
          },
          max_steps: {
            type: 'integer',
            title: 'Max Steps',
          },
          enabled: {
            type: 'boolean',
            title: 'Enabled',
          },
          mode: {
            type: 'string',
            title: 'Mode',
            enum: ['classic', 'react', 'chain'],
          },
          prompt_text: {
            type: 'string',
            title: 'Prompt',
            'x-widget': 'unknown-widget-xyz',
          },
          system_prompt: {
            type: 'string',
            title: 'System Prompt',
            'x-widget': 'prompt',
          },
        },
        required: ['name'],
      },
    },
    {
      type: 'plan_and_execute',
      label: 'Plan & Execute',
      icon: 'plan',
      category: 'control_flow',
      is_composite: true,
      config_schema: {
        type: 'object',
        properties: {
          planner: {
            type: 'object',
            title: 'Planner',
            properties: {
              system_prompt: { type: 'string', title: 'System Prompt', 'x-widget': 'prompt' },
            },
          },
          body: {
            type: 'object',
            title: 'Body',
            properties: {
              id: { type: 'string', title: 'ID' },
            },
          },
          max_iterations: {
            type: 'integer',
            title: 'Max Iterations',
          },
        },
      },
    },
  ],
  agent_subtypes: [],
  tool_kinds: [
    {
      kind: 'vector_search',
      label: 'Vector Search',
      icon: 'tool',
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
          num_results: { type: 'integer', title: 'Result Count', default: 10 },
        },
        required: ['index_name'],
      },
    },
    {
      kind: 'table_search',
      label: 'Table Search',
      icon: 'table',
      config_schema: {
        type: 'object',
        properties: {
          warehouse_id: {
            type: 'string',
            title: 'SQL Warehouse',
            'x-widget': 'resource-select',
            'x-source-kind': 'sql_warehouse',
            'x-value-field': 'warehouse_id',
          },
        },
        required: ['warehouse_id'],
      },
    },
  ],
  model_tiers: [],
  version: '1.0',
};

// ---------------------------------------------------------------------------
// Helper: build a minimal AST with an agent block selected at root.children.0
// ---------------------------------------------------------------------------

function makeAst(config: Record<string, unknown> = {}): AST {
  return {
    ...createDraftWorkflow('Test Workflow'),
    root: {
      id: 'root-id',
      type: 'sequence',
      label: 'root',
      config: {},
      children: [
        {
          id: 'agent-1',
          type: 'agent',
          label: 'My Agent',
          config,
          children: [],
        },
      ],
    },
  };
}

function makePlanAst(config: Record<string, unknown> = {}): AST {
  return {
    ...createDraftWorkflow('Test Workflow'),
    root: {
      id: 'root-id',
      type: 'sequence',
      label: 'root',
      config: {},
      children: [
        {
          id: 'plan-1',
          type: 'plan_and_execute',
          label: 'Plan',
          config,
          children: [],
        },
      ],
    },
  };
}

function renderWithQuery(ui: React.ReactElement) {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false }, mutations: { retry: false } },
  });
  return render(<QueryClientProvider client={client}>{ui}</QueryClientProvider>);
}

// ---------------------------------------------------------------------------
// Reset store before each test
// ---------------------------------------------------------------------------

beforeEach(() => {
  useAgentEditorStore.setState(initialState);
});

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe('ConfigPanel', () => {
  it('renders Workspace tools view when no block is selected', () => {
    // selectedPath is null by default from initialState — Inspector renders the
    // Workspace tools view (registry of declared tools available to agents).
    render(<ConfigPanel registry={FIXTURE_REGISTRY} />);
    expect(screen.getByText('Workspace tools')).toBeInTheDocument();
    expect(
      screen.getByText(/Select an agent block to bind tools to it/i),
    ).toBeInTheDocument();
  });

  it('edits an existing workspace tool declaration config', () => {
    const ast: AST = {
      ...createDraftWorkflow('Test Workflow'),
      tools: [
        {
          kind: 'vector_search',
          name: 'customer_vector',
          config: { index_name: 'old.index', num_results: 10 },
        },
      ],
    };
    useAgentEditorStore.setState({ ast, selectedPath: null });
    const updateSpy = vi.spyOn(useAgentEditorStore.getState(), 'updateTool');

    renderWithQuery(<ConfigPanel registry={FIXTURE_REGISTRY} />);
    fireEvent.click(screen.getByText('customer_vector'));
    fireEvent.change(screen.getByRole('combobox', { name: /vector search index/i }), {
      target: { value: 'main.sales.customer_index' },
    });

    expect(updateSpy).toHaveBeenCalledWith('customer_vector', {
      config: { index_name: 'main.sales.customer_index', num_results: 10 },
    });
    updateSpy.mockRestore();
  });

  it('starts a stopped SQL warehouse when selected for a table tool', async () => {
    vi.mocked(listDesignerResources).mockResolvedValueOnce({
      resources: [
        {
          kind: 'sql_warehouse',
          source_id: 'wh-stopped',
          name: 'Starter Warehouse',
          full_name: 'Starter Warehouse',
          description: null,
          status: 'STOPPED',
          capabilities: ['sql'],
          metadata: { warehouse_id: 'wh-stopped', state: 'STOPPED' },
        },
      ],
      total: 1,
    });
    vi.mocked(startDesignerSqlWarehouse).mockResolvedValueOnce({
      kind: 'sql_warehouse',
      source_id: 'wh-stopped',
      name: 'Starter Warehouse',
      full_name: 'Starter Warehouse',
      description: null,
      status: 'STARTING',
      capabilities: ['sql'],
      metadata: { warehouse_id: 'wh-stopped', state: 'STARTING' },
    });
    const ast: AST = {
      ...createDraftWorkflow('Test Workflow'),
      tools: [
        {
          kind: 'table_search',
          name: 'table_search',
          config: { warehouse_id: '' },
        },
      ],
    };
    useAgentEditorStore.setState({ ast, selectedPath: null });
    const updateSpy = vi.spyOn(useAgentEditorStore.getState(), 'updateTool');

    const { container } = renderWithQuery(<ConfigPanel registry={FIXTURE_REGISTRY} />);
    fireEvent.click(screen.getByText('table_search'));
    await waitFor(() => {
      expect(container.querySelector('option[value="wh-stopped"]')).toBeInTheDocument();
    });

    fireEvent.change(screen.getByRole('combobox', { name: /sql warehouse/i }), {
      target: { value: 'wh-stopped' },
    });

    expect(updateSpy).toHaveBeenCalledWith('table_search', {
      config: { warehouse_id: 'wh-stopped' },
    });
    await waitFor(() => {
      expect(vi.mocked(startDesignerSqlWarehouse).mock.calls[0]?.[0]).toBe('wh-stopped');
    });
    updateSpy.mockRestore();
  });

  it('renders a text input for a string field', () => {
    const ast = makeAst({ name: 'hello' });
    useAgentEditorStore.setState({ ast, selectedPath: 'root.children.0' });
    render(<ConfigPanel registry={FIXTURE_REGISTRY} />);

    const input = screen.getByRole('textbox', { name: /name/i });
    expect(input).toBeInTheDocument();
    expect(input).toHaveAttribute('type', 'text');
    expect((input as HTMLInputElement).value).toBe('hello');
  });

  it('renders a number input for an integer field', () => {
    const ast = makeAst({ max_steps: 5 });
    useAgentEditorStore.setState({ ast, selectedPath: 'root.children.0' });
    render(<ConfigPanel registry={FIXTURE_REGISTRY} />);

    const input = screen.getByRole('spinbutton');
    expect(input).toBeInTheDocument();
    expect(input).toHaveAttribute('type', 'number');
    expect((input as HTMLInputElement).value).toBe('5');
  });

  it('renders a checkbox for a boolean field', () => {
    const ast = makeAst({ enabled: true });
    useAgentEditorStore.setState({ ast, selectedPath: 'root.children.0' });
    render(<ConfigPanel registry={FIXTURE_REGISTRY} />);

    const checkbox = screen.getByRole('checkbox');
    expect(checkbox).toBeInTheDocument();
    expect((checkbox as HTMLInputElement).checked).toBe(true);
  });

  it('renders a Radix Select trigger showing the current value for an enum field', () => {
    const ast = makeAst({ mode: 'react' });
    useAgentEditorStore.setState({ ast, selectedPath: 'root.children.0' });
    render(<ConfigPanel registry={FIXTURE_REGISTRY} />);

    // The Radix Select trigger renders its current value as visible text
    const trigger = screen.getByRole('combobox', { name: /mode/i });
    expect(trigger).toBeInTheDocument();
    expect(trigger).toHaveTextContent('react');
  });

  it('calls store.updateBlock when a text input is edited', () => {
    const ast = makeAst({ name: '' });
    useAgentEditorStore.setState({ ast, selectedPath: 'root.children.0' });
    const updateBlockSpy = vi.spyOn(useAgentEditorStore.getState(), 'updateBlock');

    render(<ConfigPanel registry={FIXTURE_REGISTRY} />);

    const input = screen.getByRole('textbox', { name: /name/i });
    fireEvent.change(input, { target: { value: 'new-name' } });

    expect(updateBlockSpy).toHaveBeenCalledWith(
      'root.children.0',
      expect.objectContaining({ config: expect.objectContaining({ name: 'new-name' }) }),
    );
  });

  it('shows a validation error when an invalid value is entered for a field', () => {
    // Start with a valid name value, then clear it to violate minLength: 1
    const ast = makeAst({ name: 'valid-name' });
    useAgentEditorStore.setState({ ast, selectedPath: 'root.children.0' });
    render(<ConfigPanel registry={FIXTURE_REGISTRY} />);

    // Clear the name field — this violates minLength: 1
    const nameInput = screen.getByRole('textbox', { name: /name/i });
    fireEvent.change(nameInput, { target: { value: '' } });

    // AJV minLength error should appear (message: "must NOT have fewer than 1 characters")
    expect(
      screen.getByText(/must NOT have fewer than 1 characters/i),
    ).toBeInTheDocument();
  });

  it('falls back to default widget and logs console.warn for unknown x-widget', () => {
    const warnSpy = vi.spyOn(console, 'warn').mockImplementation(() => undefined);

    const ast = makeAst({ prompt_text: 'hello' });
    useAgentEditorStore.setState({ ast, selectedPath: 'root.children.0' });
    render(<ConfigPanel registry={FIXTURE_REGISTRY} />);

    // Should have called console.warn with the unknown widget name
    expect(warnSpy).toHaveBeenCalledWith(
      expect.stringContaining("unknown widget 'unknown-widget-xyz'"),
    );

    // Should still render a text input (fallback for type=string)
    // The prompt_text field has type=string so fallback is text input
    const input = screen.getByDisplayValue('hello');
    expect(input).toBeInTheDocument();

    warnSpy.mockRestore();
  });

  it('renders prompt widgets as textareas with persisted values', () => {
    const ast = makeAst({ system_prompt: 'Be precise and cite sources.' });
    useAgentEditorStore.setState({ ast, selectedPath: 'root.children.0' });
    render(<ConfigPanel registry={FIXTURE_REGISTRY} />);

    const input = screen.getByRole('textbox', { name: /system prompt/i });
    expect(input.tagName).toBe('TEXTAREA');
    expect(input).toHaveValue('Be precise and cite sources.');
  });

  it('renders nested planner config but hides plan body raw object editing', () => {
    const ast = makePlanAst({
      planner: { system_prompt: 'Plan carefully.' },
      body: { id: 'body', type: 'sequence', label: 'Body', config: {}, children: [] },
      max_iterations: 3,
    });
    useAgentEditorStore.setState({ ast, selectedPath: 'root.children.0' });
    render(<ConfigPanel registry={FIXTURE_REGISTRY} />);

    expect(screen.getByRole('textbox', { name: /system prompt/i })).toHaveValue('Plan carefully.');
    expect(screen.queryByText('Body')).not.toBeInTheDocument();
    expect(screen.getByRole('spinbutton')).toHaveValue(3);
  });

  it('marks the required field label with a lava asterisk', () => {
    const ast = makeAst({ name: 'test' });
    useAgentEditorStore.setState({ ast, selectedPath: 'root.children.0' });
    const { container } = render(<ConfigPanel registry={FIXTURE_REGISTRY} />);

    // The "name" field is required — should have a db-lava-600 * marker
    const lavaStar = container.querySelector('.text-db-lava-600');
    expect(lavaStar).toBeInTheDocument();
    expect(lavaStar?.textContent).toBe('*');
  });
});
