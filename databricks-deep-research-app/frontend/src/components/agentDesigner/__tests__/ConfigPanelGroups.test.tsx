/**
 * Tests for the grouped (collapsible) agent inspector + the json widget (P1c).
 *
 * The flat-render path is covered by ConfigPanel.test.tsx (fixtures without
 * x-group). These fixtures carry x-group/x-advanced/x-widget so the grouped
 * accordion path renders.
 */

import { describe, it, expect, beforeEach, vi } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import '@testing-library/jest-dom';
import { ConfigPanel } from '../ConfigPanel';
import { useAgentEditorStore, initialState } from '@/stores/agentEditorStore';
import { createDraftWorkflow } from '@/lib/workflowAst';
import type { RegistryResponse } from '@/types/agentDesigner';
import type { AST } from '@/types/ast';

vi.mock('@/api/agentDesigner', async (importOriginal) => {
  const actual = await importOriginal<typeof import('@/api/agentDesigner')>();
  return {
    ...actual,
    listDesignerResources: vi.fn().mockResolvedValue({ resources: [], total: 0 }),
    getDesignerCapabilities: vi.fn().mockResolvedValue({
      skill_scripts_global: false,
      cross_session_memory_global: false,
      live_search_global: false,
    }),
  };
});

function registryWith(properties: Record<string, Record<string, unknown>>, required: string[] = []): RegistryResponse {
  return {
    node_types: [
      {
        type: 'agent',
        label: 'Agent',
        icon: 'agent',
        category: 'core',
        is_composite: false,
        config_schema: { type: 'object', properties, required },
      },
    ],
    agent_subtypes: [],
    tool_kinds: [],
    model_tiers: [],
    version: '1.0',
  };
}

function makeAst(config: Record<string, unknown> = {}): AST {
  return {
    ...createDraftWorkflow('Test Workflow'),
    root: {
      id: 'root-id',
      type: 'sequence',
      label: 'root',
      config: {},
      children: [{ id: 'agent-1', type: 'agent', label: 'My Agent', config, children: [] }],
    },
  };
}

const GROUPED_SCHEMA = {
  subtype: { type: 'string', title: 'Subtype', enum: ['researcher'], 'x-group': 'Basics', 'x-order': 10, 'x-advanced': false },
  action_mode: {
    type: 'string',
    title: 'Action Mode',
    enum: ['tools', 'code'],
    'x-group': 'Execution',
    'x-order': 10,
    'x-advanced': true,
  },
};

beforeEach(() => {
  useAgentEditorStore.setState(initialState);
});

describe('ConfigPanel grouped inspector', () => {
  it('renders collapsible group sections with their names', () => {
    useAgentEditorStore.setState({ ast: makeAst({ subtype: 'researcher' }), selectedPath: 'root.children.0' });
    render(<ConfigPanel registry={registryWith(GROUPED_SCHEMA)} />);
    expect(screen.getByText('Basics')).toBeInTheDocument();
    expect(screen.getByText('Execution')).toBeInTheDocument();
  });

  it('expands non-advanced groups and collapses advanced groups by default', () => {
    useAgentEditorStore.setState({ ast: makeAst({ subtype: 'researcher' }), selectedPath: 'root.children.0' });
    render(<ConfigPanel registry={registryWith(GROUPED_SCHEMA)} />);
    const basics = screen.getByText('Basics').closest('details') as HTMLDetailsElement;
    const execution = screen.getByText('Execution').closest('details') as HTMLDetailsElement;
    expect(basics.open).toBe(true); // not advanced => open
    expect(execution.open).toBe(false); // advanced => collapsed
  });

  it('toggles a group open on summary click', () => {
    useAgentEditorStore.setState({ ast: makeAst({ subtype: 'researcher' }), selectedPath: 'root.children.0' });
    render(<ConfigPanel registry={registryWith(GROUPED_SCHEMA)} />);
    const summary = screen.getByText('Execution');
    const execution = summary.closest('details') as HTMLDetailsElement;
    expect(execution.open).toBe(false);
    fireEvent.click(summary);
    expect(execution.open).toBe(true);
  });

  it('force-opens an advanced group and shows an error badge when a field is invalid', () => {
    // Basics field is visible; the advanced Execution field violates minLength
    // once validation runs. Editing the visible field triggers validate() over
    // the whole config, surfacing the hidden field's error.
    const schema = {
      name: { type: 'string', title: 'Name', minLength: 1, 'x-group': 'Basics', 'x-order': 10, 'x-advanced': false },
      code: {
        type: 'string',
        title: 'Code',
        minLength: 1,
        'x-group': 'Execution',
        'x-order': 10,
        'x-advanced': true,
      },
    };
    useAgentEditorStore.setState({ ast: makeAst({ name: 'ok', code: '' }), selectedPath: 'root.children.0' });
    render(<ConfigPanel registry={registryWith(schema)} />);

    const execution = screen.getByText('Execution').closest('details') as HTMLDetailsElement;
    expect(execution.open).toBe(false); // collapsed before validation runs

    // Edit the visible Basics field to trigger validation over the whole config.
    fireEvent.change(screen.getByRole('textbox', { name: /name/i }), { target: { value: 'still-ok' } });

    expect(execution.open).toBe(true); // force-opened so the hidden error is visible
    expect(screen.getByLabelText(/1 validation error/i)).toBeInTheDocument(); // badge
  });

  it('warns when allow_skill_scripts is on but globally disabled (never-silent)', async () => {
    const schema = {
      allow_skill_scripts: {
        type: 'boolean',
        title: 'Allow Skill Scripts',
        'x-group': 'Tools & Sources',
        'x-order': 10,
        'x-advanced': false,
      },
    };
    useAgentEditorStore.setState({
      ast: makeAst({ allow_skill_scripts: true }),
      selectedPath: 'root.children.0',
    });
    render(<ConfigPanel registry={registryWith(schema)} />);
    // capabilities mock => skill_scripts_global: false, so the gated-knob banner appears.
    expect(await screen.findByText(/no effect until an admin/i)).toBeInTheDocument();
  });
});

describe('SchemaField json widget', () => {
  const JSON_SCHEMA = {
    per_tool_limits: {
      type: 'object',
      title: 'Per Tool Limits',
      'x-widget': 'json',
      'x-group': 'Basics',
      'x-order': 10,
      'x-advanced': false,
    },
  };

  it('renders a textarea pre-filled with the JSON value', () => {
    useAgentEditorStore.setState({ ast: makeAst({ per_tool_limits: { web_search: 5 } }), selectedPath: 'root.children.0' });
    render(<ConfigPanel registry={registryWith(JSON_SCHEMA)} />);
    const ta = screen.getByRole('textbox', { name: /per tool limits/i }) as HTMLTextAreaElement;
    expect(ta.tagName).toBe('TEXTAREA');
    expect(JSON.parse(ta.value)).toEqual({ web_search: 5 });
  });

  it('emits the parsed object on valid JSON', () => {
    useAgentEditorStore.setState({ ast: makeAst({ per_tool_limits: {} }), selectedPath: 'root.children.0' });
    const updateSpy = vi.spyOn(useAgentEditorStore.getState(), 'updateBlock');
    render(<ConfigPanel registry={registryWith(JSON_SCHEMA)} />);
    const ta = screen.getByRole('textbox', { name: /per tool limits/i });
    fireEvent.change(ta, { target: { value: '{"web_search": 7}' } });
    expect(updateSpy).toHaveBeenCalledWith(
      'root.children.0',
      expect.objectContaining({ config: expect.objectContaining({ per_tool_limits: { web_search: 7 } }) }),
    );
    updateSpy.mockRestore();
  });

  it('shows an inline parse error and does not emit on invalid JSON', () => {
    useAgentEditorStore.setState({ ast: makeAst({ per_tool_limits: {} }), selectedPath: 'root.children.0' });
    const updateSpy = vi.spyOn(useAgentEditorStore.getState(), 'updateBlock');
    render(<ConfigPanel registry={registryWith(JSON_SCHEMA)} />);
    const ta = screen.getByRole('textbox', { name: /per tool limits/i });
    fireEvent.change(ta, { target: { value: '{not valid' } });
    expect(screen.getByText(/invalid json/i)).toBeInTheDocument();
    expect(updateSpy).not.toHaveBeenCalled();
    updateSpy.mockRestore();
  });
});
