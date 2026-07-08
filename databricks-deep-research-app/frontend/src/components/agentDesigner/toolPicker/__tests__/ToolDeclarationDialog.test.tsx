/**
 * Search-first picker tests: UC function live search + declaration, pasted-FQN
 * fast path, partial-result warning surfacing, and existing-tool selection.
 */

import { describe, it, expect, beforeEach, vi } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import '@testing-library/jest-dom';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';

import { ToolDeclarationDialog } from '../ToolDeclarationDialog';
import { useAgentEditorStore, initialState } from '@/stores/agentEditorStore';
import { createDraftWorkflow } from '@/lib/workflowAst';
import { getUcFunctionSignature, listDesignerResources } from '@/api/agentDesigner';
import type { RegistryResponse } from '@/types/agentDesigner';
import type { AST } from '@/types/ast';

vi.mock('@/api/agentDesigner', async (importOriginal) => {
  const actual = await importOriginal<typeof import('@/api/agentDesigner')>();
  return {
    ...actual,
    listDesignerResources: vi.fn(async (kinds?: string[]) => {
      if (kinds?.[0] === 'uc_catalog') {
        return {
          resources: [
            {
              kind: 'uc_catalog',
              source_id: 'main',
              name: 'main',
              full_name: 'main',
              description: null,
              status: null,
              capabilities: [],
              metadata: {},
            },
          ],
          total: 1,
        };
      }
      if (kinds?.[0] === 'uc_function') {
        return {
          resources: [
            {
              kind: 'uc_function',
              source_id: 'main.metrics.pct_change',
              name: 'pct_change',
              full_name: 'main.metrics.pct_change',
              description: null,
              status: null,
              capabilities: [],
              metadata: {},
            },
          ],
          total: 1,
          warning: "Partial results: searched the first 22 of 30 schemas in 'main'.",
        };
      }
      return { resources: [], total: 0 };
    }),
    getUcFunctionSignature: vi.fn().mockResolvedValue({
      function: 'main.metrics.pct_change',
      params: [
        { name: 'current', type: 'number', required: true },
        { name: 'previous', type: 'number', required: true },
      ],
      scalar: true,
      returns_table: false,
      run_ready: true,
      warning: null,
    }),
  };
});

const REGISTRY: RegistryResponse = {
  node_types: [],
  agent_subtypes: [],
  tool_kinds: [
    { kind: 'web_search', label: 'Web Search', icon: 'tool' },
    {
      kind: 'uc_function',
      label: 'Unity Catalog Function',
      icon: 'tool',
      layer: 'B',
      config_schema: {
        type: 'object',
        properties: { function: { type: 'string', title: 'UC Function' } },
        required: ['function'],
      },
    },
  ],
  model_tiers: [],
  version: '1.0',
} as unknown as RegistryResponse;

function makeAst(): AST {
  return {
    ...createDraftWorkflow('Test Workflow'),
    root: { id: 'root-id', type: 'sequence', label: 'root', config: {}, children: [] },
  };
}

function renderDialog(onDeclared = vi.fn()) {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  render(
    <QueryClientProvider client={client}>
      <ToolDeclarationDialog
        registry={REGISTRY}
        open
        onOpenChange={vi.fn()}
        onDeclared={onDeclared}
      />
    </QueryClientProvider>,
  );
  return onDeclared;
}

beforeEach(() => {
  vi.clearAllMocks();
  useAgentEditorStore.setState({ ...initialState, ast: makeAst() });
});

describe('ToolDeclarationDialog (search-first)', () => {
  it('searches UC functions for a catalog-scoped query and declares with signature', async () => {
    const declareSpy = vi.spyOn(useAgentEditorStore.getState(), 'declareTool');
    renderDialog();

    fireEvent.change(screen.getByRole('textbox', { name: /search tools/i }), {
      target: { value: 'main.pct' },
    });

    const row = await screen.findByRole('button', { name: /pct_change/i });
    fireEvent.click(row);

    await waitFor(() =>
      expect(declareSpy).toHaveBeenCalledWith('uc_function', 'pct_change', {
        function: 'main.metrics.pct_change',
        params: [
          { name: 'current', type: 'number', required: true },
          { name: 'previous', type: 'number', required: true },
        ],
      }),
    );
    expect(vi.mocked(getUcFunctionSignature)).toHaveBeenCalledWith(
      'main.metrics.pct_change',
    );
    // The catalog-scoped search was issued with parent=catalog, query=prefix.
    expect(vi.mocked(listDesignerResources)).toHaveBeenCalledWith(['uc_function'], {
      parent: 'main',
      query: 'pct',
    });
    declareSpy.mockRestore();
  });

  it('surfaces the partial-result warning from the search response', async () => {
    renderDialog();
    fireEvent.change(screen.getByRole('textbox', { name: /search tools/i }), {
      target: { value: 'main.pct' },
    });
    expect(await screen.findByText(/searched the first 22 of 30 schemas/i)).toBeInTheDocument();
  });

  it('offers a pasted full FQN as a direct declaration row', async () => {
    const declareSpy = vi.spyOn(useAgentEditorStore.getState(), 'declareTool');
    renderDialog();

    fireEvent.change(screen.getByRole('textbox', { name: /search tools/i }), {
      target: { value: 'main.metrics.pct_change' },
    });

    const useRow = await screen.findByRole('button', {
      name: /use main\.metrics\.pct_change/i,
    });
    fireEvent.click(useRow);

    await waitFor(() =>
      expect(declareSpy).toHaveBeenCalledWith(
        'uc_function',
        'pct_change',
        expect.objectContaining({ function: 'main.metrics.pct_change' }),
      ),
    );
    declareSpy.mockRestore();
  });

  it('applies an existing workflow tool without re-declaring it', () => {
    const ast = makeAst();
    ast.tools = [{ kind: 'web_search', name: 'ws1', config: {} }];
    useAgentEditorStore.setState({ ast });
    const declareSpy = vi.spyOn(useAgentEditorStore.getState(), 'declareTool');
    const onDeclared = renderDialog();

    fireEvent.click(screen.getByRole('button', { name: /ws1/i }));

    expect(onDeclared).toHaveBeenCalledWith({
      kind: 'web_search',
      name: 'ws1',
      config: {},
    });
    expect(declareSpy).not.toHaveBeenCalled();
    declareSpy.mockRestore();
  });
});
