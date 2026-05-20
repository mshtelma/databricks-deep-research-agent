/**
 * Tests for BlockNode and AddBlockMenu components.
 *
 * DnD context requirement: useSortable requires a <DndContext> ancestor.
 * We wrap every render in <TestDndWrapper>.
 *
 * Radix Popover portals: jsdom supports portals, so we can use
 * screen.getByRole('menuitem') after clicking the trigger.
 */

import * as React from 'react';
import { render, screen, fireEvent } from '@testing-library/react';
import { describe, it, expect, vi, beforeEach } from 'vitest';
import '@testing-library/jest-dom';
import { DndContext } from '@dnd-kit/core';
import { SortableContext } from '@dnd-kit/sortable';

import { BlockNode } from '../BlockNode';
import { AddBlockMenu } from '../AddBlockMenu';
import { useAgentEditorStore, initialState } from '@/stores/agentEditorStore';
import { createDraftWorkflow } from '@/lib/workflowAst';
import type { Block, BlockPath, NodeType } from '@/types/ast';
import type { NodeTypeSpec, RegistryResponse } from '@/types/agentDesigner';

// ---------------------------------------------------------------------------
// Test DnD wrapper
// ---------------------------------------------------------------------------

function TestDndWrapper({ children, items = [] }: { children: React.ReactNode; items?: string[] }): React.ReactElement {
  return (
    <DndContext>
      <SortableContext items={items}>
        {children}
      </SortableContext>
    </DndContext>
  );
}

// ---------------------------------------------------------------------------
// Sample registry
// ---------------------------------------------------------------------------

const sampleNodeTypes: NodeTypeSpec[] = [
  { type: 'sequence', label: 'Sequence', icon: '🔢', category: 'control_flow', is_composite: true, config_schema: null },
  { type: 'agent', label: 'Agent', icon: '🤖', category: 'agent', is_composite: false, config_schema: null },
  { type: 'tool', label: 'Tool', icon: '🔧', category: 'tool', is_composite: false, config_schema: null },
];

const sampleRegistry: RegistryResponse = {
  node_types: sampleNodeTypes,
  agent_subtypes: [],
  tool_kinds: [],
  model_tiers: ['simple', 'analytical', 'complex'],
  version: '1.0.0',
};

// ---------------------------------------------------------------------------
// Sample blocks
// ---------------------------------------------------------------------------

function makeBlock(
  nodeType: NodeType,
  overrides: Partial<Block> = {},
): Block {
  return {
    id: crypto.randomUUID(),
    type: nodeType,
    label: `Test ${nodeType}`,
    config: {},
    children: [],
    ...overrides,
  };
}

// ---------------------------------------------------------------------------
// Render helper
// ---------------------------------------------------------------------------

function renderBlock(block: Block, path: BlockPath = 'root', parentNodeType?: NodeType): ReturnType<typeof render> {
  return render(
    <TestDndWrapper items={[path]}>
      <BlockNode
        block={block}
        path={path}
        registry={sampleRegistry}
        parentNodeType={parentNodeType}
      />
    </TestDndWrapper>,
  );
}

// ---------------------------------------------------------------------------
// Store reset
// ---------------------------------------------------------------------------

beforeEach(() => {
  // Merge state reset (do NOT pass true as second arg — that would replace
  // the action functions and break vi.spyOn calls on getState()).
  useAgentEditorStore.setState({
    ...initialState,
    ast: {
      ...createDraftWorkflow('Test Workflow'),
      root: makeBlock('sequence', {
        id: 'root-id',
        label: 'Root',
        children: [],
      }),
    },
  });
});

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe('BlockNode', () => {
  it('renders root sequence with two children recursively', () => {
    const child1 = makeBlock('agent', { id: 'c1', label: 'Alpha Agent' });
    const child2 = makeBlock('tool', { id: 'c2', label: 'Beta Tool' });
    const root = makeBlock('sequence', {
      id: 'root-id',
      label: 'Root Sequence',
      children: [child1, child2],
    });

    renderBlock(root, 'root');

    expect(screen.getByText('Root Sequence')).toBeInTheDocument();
    expect(screen.getByText('Alpha Agent')).toBeInTheDocument();
    expect(screen.getByText('Beta Tool')).toBeInTheDocument();
  });

  it('renders agent block with model_tier summary line', () => {
    const block = makeBlock('agent', {
      config: { model_tier: 'complex' },
      label: 'My Agent',
    });

    renderBlock(block, 'root.children.0');

    expect(screen.getByText('My Agent')).toBeInTheDocument();
    expect(screen.getByText('Agent · complex')).toBeInTheDocument();
  });

  it('renders loop block with max_iterations in summary', () => {
    const block = makeBlock('loop', {
      config: { max_iterations: 5 },
      label: 'My Loop',
    });

    renderBlock(block, 'root.children.0');

    expect(screen.getByText('Loop × max 5')).toBeInTheDocument();
  });

  it('renders conditional block showing branch count', () => {
    const branch1 = makeBlock('agent', { id: 'b1', label: 'Branch A' });
    const branch2 = makeBlock('agent', { id: 'b2', label: 'Branch B' });
    const block = makeBlock('conditional', {
      label: 'My Conditional',
      children: [branch1, branch2],
    });

    renderBlock(block, 'root.children.0');

    expect(screen.getByText('Conditional · 2 branches')).toBeInTheDocument();
  });

  it('click on the block calls select(path)', () => {
    const selectSpy = vi.spyOn(useAgentEditorStore.getState(), 'select');

    const block = makeBlock('agent', { label: 'Click Me' });
    renderBlock(block, 'root.children.0');

    // Click the block card (not the drag handle)
    const label = screen.getByText('Click Me');
    fireEvent.click(label);

    expect(selectSpy).toHaveBeenCalledWith('root.children.0');
    selectSpy.mockRestore();
  });

  it('Delete keypress on block calls deleteBlock(path)', () => {
    const deleteSpy = vi.spyOn(useAgentEditorStore.getState(), 'deleteBlock');

    const block = makeBlock('agent', { label: 'Delete Me' });
    const { container } = renderBlock(block, 'root.children.0');

    const card = container.firstElementChild as HTMLElement;
    fireEvent.keyDown(card, { key: 'Delete' });

    expect(deleteSpy).toHaveBeenCalledWith('root.children.0');
    deleteSpy.mockRestore();
  });

  it('renders drag handle with correct test-id', () => {
    const block = makeBlock('agent', { label: 'Drag Me' });
    renderBlock(block, 'root.children.0');

    const handle = screen.getByTestId('block-drag-handle-root.children.0');
    expect(handle).toBeInTheDocument();
  });

  it('AddBlockMenu shows registry node types when trigger is clicked', async () => {
    render(
      <TestDndWrapper items={['root']}>
        <AddBlockMenu
          parentPath="root"
          parentNodeType="sequence"
          registry={sampleRegistry}
        >
          <button>Add</button>
        </AddBlockMenu>
      </TestDndWrapper>,
    );

    // Open the popover
    fireEvent.click(screen.getByText('Add'));

    // All 3 sample node types should appear as menuitems
    const items = screen.getAllByRole('menuitem');
    expect(items.length).toBeGreaterThanOrEqual(3);
    expect(items.some((el) => el.textContent?.includes('Sequence'))).toBe(true);
    expect(items.some((el) => el.textContent?.includes('Agent'))).toBe(true);
    expect(items.some((el) => el.textContent?.includes('Tool'))).toBe(true);
  });

  it('AddBlockMenu selecting a node type calls addBlock on the store', () => {
    const addBlockSpy = vi.spyOn(useAgentEditorStore.getState(), 'addBlock');

    render(
      <TestDndWrapper items={['root']}>
        <AddBlockMenu
          parentPath="root"
          parentNodeType="sequence"
          registry={sampleRegistry}
        >
          <button>Add</button>
        </AddBlockMenu>
      </TestDndWrapper>,
    );

    fireEvent.click(screen.getByText('Add'));

    const agentItem = screen.getAllByRole('menuitem').find((el) =>
      el.textContent?.includes('Agent'),
    );
    expect(agentItem).toBeDefined();
    fireEvent.click(agentItem!);

    expect(addBlockSpy).toHaveBeenCalledWith('root', 'agent', 'Agent', {});
    addBlockSpy.mockRestore();
  });
});

// ---------------------------------------------------------------------------
// Malformed-block guard tests
//
// Regression: an AST that arrives with a plan_and_execute block missing its
// `config` field (e.g. config === undefined) previously crashed render with
// "Cannot read properties of undefined (reading 'body')", taking down the
// whole app via the top-level ErrorBoundary. The guards in
// BlockEditor.tsx:65, BlockNode.tsx:68, and BlockNode.tsx:108 must accept
// missing config without throwing.
// ---------------------------------------------------------------------------

describe('BlockNode malformed config guards', () => {
  it('renders plan_and_execute with config: undefined without throwing', () => {
    const root = makeBlock('plan_and_execute', {
      id: 'pae',
      label: 'Plan and Execute',
      // Force config to be undefined despite the type contract.
      config: undefined as unknown as Record<string, unknown>,
    });

    expect(() => renderBlock(root, 'root')).not.toThrow();
    expect(screen.getByText('Plan and Execute')).toBeInTheDocument();
  });

  it('renders plan_and_execute with empty config without throwing', () => {
    const root = makeBlock('plan_and_execute', {
      id: 'pae',
      label: 'Plan and Execute Empty',
      config: {},
    });

    expect(() => renderBlock(root, 'root')).not.toThrow();
    expect(screen.getByText('Plan and Execute Empty')).toBeInTheDocument();
  });

  it('subdescription handles agent with config: undefined', () => {
    const root = makeBlock('agent', {
      id: 'a1',
      label: 'Naked Agent',
      config: undefined as unknown as Record<string, unknown>,
    });

    expect(() => renderBlock(root, 'root')).not.toThrow();
    expect(screen.getByText('Naked Agent')).toBeInTheDocument();
  });
});
