import { describe, it, expect, beforeEach } from 'vitest';
import { act, fireEvent, render, screen, waitFor } from '@testing-library/react';

import { SchemaEditorPanel } from '../SchemaEditorPanel';
import { extractSurfaceFromAgentDefinition } from '@/lib/agentSurface';
import { surfaceToSchema } from '@/lib/schemaModel';
import { initialState, useAgentEditorStore } from '@/stores/agentEditorStore';
import type { AST } from '@/types/ast';
import type { Surface, SurfaceComponent } from '@/types/surface';

function makeSurface(slotNames: string[]): Surface {
  const children = slotNames.map((slot) => `${slot}_table`);
  const components: SurfaceComponent[] = [
    {
      id: 'root',
      component: 'Column',
      props: {},
      children: ['results_card'],
    },
    {
      id: 'results_card',
      component: 'Card',
      props: { title: 'Results' },
      children,
    },
    ...slotNames.map(
      (slot): SurfaceComponent => ({
        id: `${slot}_table`,
        component: 'Table',
        props: {
          source: { path: `/results/run/data/${slot}` },
          columns: [{ key: 'name', label: 'Name', type: 'string' }],
        },
        children: [],
      }),
    ),
  ];

  return {
    version: 1,
    components,
    data_model: { results: { run: null } },
    bindings: [
      {
        action: 'run',
        kind: 'run_agent',
        inputs: {},
        options: {},
        output: { target: '/results/run', mode: 'report' },
        concurrency: 'replace',
      },
    ],
  };
}

function makeAst(surface: Surface): AST {
  return {
    id: 'agent',
    name: 'Agent',
    version: 1,
    root: {
      id: 'root-node',
      type: 'sequence',
      label: 'Root',
      config: {},
      children: [],
    },
    tools: [],
    sources: [{ name: 'docs', kind: 'vector_search' }],
    surface,
  };
}

function currentSurfaceSchema() {
  const surface = extractSurfaceFromAgentDefinition(useAgentEditorStore.getState().ast);
  return surfaceToSchema(surface);
}

describe('SchemaEditorPanel', () => {
  beforeEach(() => {
    useAgentEditorStore.setState(initialState);
  });

  it('re-derives visible schema when the underlying surface changes externally', async () => {
    act(() => {
      useAgentEditorStore.setState({ ast: makeAst(makeSurface(['comparison'])) });
    });
    render(<SchemaEditorPanel />);

    expect(screen.getByDisplayValue('comparison')).toBeTruthy();

    act(() => {
      useAgentEditorStore.getState().setAst(
        makeAst(makeSurface(['comparison', 'risks'])),
      );
    });

    expect(await screen.findByDisplayValue('risks')).toBeTruthy();
  });

  it('preserves externally-added slots when editing after an external update', async () => {
    act(() => {
      useAgentEditorStore.setState({ ast: makeAst(makeSurface(['comparison'])) });
    });
    render(<SchemaEditorPanel />);

    act(() => {
      useAgentEditorStore.getState().setAst(
        makeAst(makeSurface(['comparison', 'risks'])),
      );
    });
    await screen.findByDisplayValue('risks');

    fireEvent.change(screen.getByDisplayValue('comparison'), {
      target: { value: 'vendors' },
    });

    await waitFor(() => {
      const names = currentSurfaceSchema().slots.map((slot) => slot.name).sort();
      expect(names).toEqual(['risks', 'vendors']);
    });
  });

  it('keeps section-name focus stable while renaming a slot', async () => {
    act(() => {
      useAgentEditorStore.setState({ ast: makeAst(makeSurface(['comparison'])) });
    });
    render(<SchemaEditorPanel />);

    const input = screen.getByDisplayValue('comparison') as HTMLInputElement;
    input.focus();
    fireEvent.change(input, { target: { value: 'vendors' } });

    await waitFor(() => {
      expect(screen.getByDisplayValue('vendors')).toBe(input);
    });
    expect(document.activeElement).toBe(input);
  });
});
