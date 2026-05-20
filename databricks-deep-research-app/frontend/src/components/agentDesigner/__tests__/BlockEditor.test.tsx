/**
 * Tests for BlockEditor.
 *
 * DnD drag-end testing strategy:
 *   @dnd-kit does not expose a way to programmatically fire a drag sequence in
 *   jsdom. We capture the onDragEnd prop by mocking @dnd-kit/core so that
 *   DndContext immediately stores its props via a module-level ref, then we
 *   invoke the captured handler directly with a synthetic DragEndEvent.
 *
 * Store isolation:
 *   Each test resets the store via useAgentEditorStore.setState(initialState)
 *   in beforeEach so tests do not bleed state into each other.
 */

import * as React from 'react';
import { render, screen, fireEvent } from '@testing-library/react';
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import '@testing-library/jest-dom';

import { useAgentEditorStore, initialState } from '@/stores/agentEditorStore';
import { createDraftWorkflow } from '@/lib/workflowAst';
import type { Block, AST, NodeType } from '@/types/ast';
import type { RegistryResponse, NodeTypeSpec } from '@/types/agentDesigner';
import type { DragEndEvent } from '@dnd-kit/core';

// ---------------------------------------------------------------------------
// DndContext capture mock
//
// We replace @dnd-kit/core's DndContext with a thin shim that:
//  1. Stores the latest onDragEnd callback in capturedOnDragEnd.
//  2. Renders its children unchanged so BlockNode still mounts correctly.
//
// All other exports (useSensor, useSensors, PointerSensor, KeyboardSensor,
// closestCorners) are left as real implementations so BlockEditor's import
// of those symbols continues to work.
// ---------------------------------------------------------------------------

let capturedOnDragEnd: ((event: DragEndEvent) => void) | undefined;

vi.mock('@dnd-kit/core', async (importOriginal) => {
  const actual = await importOriginal<typeof import('@dnd-kit/core')>();
  return {
    ...actual,
    DndContext: (props: {
      onDragEnd?: (event: DragEndEvent) => void;
      children?: React.ReactNode;
    }) => {
      capturedOnDragEnd = props.onDragEnd;
      return <>{props.children}</>;
    },
  };
});

// ---------------------------------------------------------------------------
// Import BlockEditor AFTER mock is set up
// ---------------------------------------------------------------------------

import { BlockEditor } from '../BlockEditor';

// ---------------------------------------------------------------------------
// Sample registry
// ---------------------------------------------------------------------------

const sampleNodeTypes: NodeTypeSpec[] = [
  {
    type: 'sequence',
    label: 'Sequence',
    icon: '',
    category: 'control_flow',
    is_composite: true,
    config_schema: null,
  },
  {
    type: 'agent',
    label: 'Agent',
    icon: '',
    category: 'agent',
    is_composite: false,
    config_schema: null,
  },
];

const sampleRegistry: RegistryResponse = {
  node_types: sampleNodeTypes,
  agent_subtypes: [],
  tool_kinds: [],
  model_tiers: ['simple', 'analytical', 'complex'],
  version: '1.0.0',
};

// ---------------------------------------------------------------------------
// Helper: build blocks
// ---------------------------------------------------------------------------

function makeBlock(nodeType: NodeType, overrides: Partial<Block> = {}): Block {
  return {
    id: crypto.randomUUID(),
    type: nodeType,
    label: `Test ${nodeType}`,
    config: {},
    children: [],
    ...overrides,
  };
}

function makeAst(root: Block): AST {
  return { ...createDraftWorkflow('Test Workflow'), root };
}

// ---------------------------------------------------------------------------
// Store reset
// ---------------------------------------------------------------------------

beforeEach(() => {
  capturedOnDragEnd = undefined;
  useAgentEditorStore.setState(initialState);
});

afterEach(() => {
  vi.restoreAllMocks();
});

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe('BlockEditor', () => {
  // 1. Empty state renders placeholder + "Add Root" button
  it('renders empty placeholder and Add Root button when ast is null', () => {
    useAgentEditorStore.setState({ ast: null });
    render(<BlockEditor registry={sampleRegistry} />);

    expect(screen.getByText('No workflow yet.')).toBeInTheDocument();
    expect(screen.getByTestId('add-root-button')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /add root/i })).toBeInTheDocument();
  });

  // 2. "Add Root" button initializes the AST
  it('Add Root click initializes the AST with a sequence root', () => {
    useAgentEditorStore.setState({ ast: null });
    render(<BlockEditor registry={sampleRegistry} />);

    expect(useAgentEditorStore.getState().ast).toBeNull();

    fireEvent.click(screen.getByTestId('add-root-button'));

    const ast = useAgentEditorStore.getState().ast;
    expect(ast).not.toBeNull();
    expect(ast?.root.type).toBe('sequence');
    expect(ast?.root.label).toBe('Workflow');
    expect(ast?.version).toBe(1);
  });

  // 3. Renders root sequence with one child
  it('renders root sequence with one child block', () => {
    const child = makeBlock('agent', { id: 'child-1', label: 'My Agent' });
    const root = makeBlock('sequence', {
      id: 'root-id',
      label: 'Root Workflow',
      children: [child],
    });
    useAgentEditorStore.setState({ ast: makeAst(root) });

    render(<BlockEditor registry={sampleRegistry} />);

    expect(screen.getByText('Root Workflow')).toBeInTheDocument();
    expect(screen.getByText('My Agent')).toBeInTheDocument();
  });

  // 4. DnD drag-end calls moveBlock(from, to)
  it('drag-end calls moveBlock with active and over paths', () => {
    const child1 = makeBlock('agent', { id: 'c1', label: 'Child One' });
    const child2 = makeBlock('agent', { id: 'c2', label: 'Child Two' });
    const root = makeBlock('sequence', {
      id: 'root-id',
      label: 'Root',
      children: [child1, child2],
    });
    useAgentEditorStore.setState({ ast: makeAst(root) });

    const moveBlockSpy = vi.spyOn(useAgentEditorStore.getState(), 'moveBlock');

    render(<BlockEditor registry={sampleRegistry} />);

    expect(capturedOnDragEnd).toBeDefined();

    const fakeEvent = {
      active: { id: 'root.children.0' },
      over: { id: 'root.children.1' },
    } as unknown as DragEndEvent;

    capturedOnDragEnd!(fakeEvent);

    expect(moveBlockSpy).toHaveBeenCalledWith('root.children.0', 'root.children.1');
  });

  // 5. Drag-to-descendant is rejected — moveBlock NOT called
  it('drag-end onto a descendant does not call moveBlock', () => {
    const grandchild = makeBlock('agent', { id: 'gc1', label: 'Grandchild' });
    const child = makeBlock('sequence', {
      id: 'c1',
      label: 'Child Sequence',
      children: [grandchild],
    });
    const root = makeBlock('sequence', {
      id: 'root-id',
      label: 'Root',
      children: [child],
    });
    useAgentEditorStore.setState({ ast: makeAst(root) });

    const moveBlockSpy = vi.spyOn(useAgentEditorStore.getState(), 'moveBlock');

    render(<BlockEditor registry={sampleRegistry} />);

    expect(capturedOnDragEnd).toBeDefined();

    // Dragging 'root.children.0' onto its own descendant 'root.children.0.children.0'
    const fakeEvent = {
      active: { id: 'root.children.0' },
      over: { id: 'root.children.0.children.0' },
    } as unknown as DragEndEvent;

    capturedOnDragEnd!(fakeEvent);

    expect(moveBlockSpy).not.toHaveBeenCalled();
  });

  // 6. 30-block tree (5 levels deep) renders without errors
  it('renders a 30-block tree (5 levels deep) without errors', () => {
    /**
     * Build a tree where each sequence has 2 children, depth 5.
     * Level counts: 1 root + 2 + 4 + 8 + 16 = 31 blocks total.
     * We cap at depth 5, leaf nodes are agent blocks.
     */
    function buildTree(depth: number, idPrefix: string): Block {
      if (depth === 0) {
        return makeBlock('agent', {
          id: `${idPrefix}-leaf`,
          label: `Leaf ${idPrefix}`,
        });
      }
      const left = buildTree(depth - 1, `${idPrefix}-L`);
      const right = buildTree(depth - 1, `${idPrefix}-R`);
      return makeBlock('sequence', {
        id: `${idPrefix}-seq`,
        label: `Seq ${idPrefix}`,
        children: [left, right],
      });
    }

    const bigRoot = buildTree(4, 'n');
    useAgentEditorStore.setState({ ast: makeAst(bigRoot) });

    // Should render without throwing
    const { container } = render(<BlockEditor registry={sampleRegistry} />);

    // Count all block cards rendered: each block renders a drag-handle button
    // with data-testid="block-drag-handle-<path>"
    const handles = container.querySelectorAll('[data-testid^="block-drag-handle-"]');
    // buildTree(4,...) gives 1+2+4+8+16 = 31 nodes
    expect(handles.length).toBe(31);
  });

  // Regression: BlockEditor.collectPaths walks the entire AST before render
  // and reads `block.config.body` on every plan_and_execute node. A block
  // arriving with `config === undefined` previously crashed the entire app
  // via the top-level ErrorBoundary with
  // "Cannot read properties of undefined (reading 'body')". The optional
  // chaining at BlockEditor.tsx:65 must accept this gracefully.
  it('does not throw when a plan_and_execute child has config: undefined', () => {
    const malformed = makeBlock('plan_and_execute', {
      id: 'pae-bad',
      label: 'Malformed Plan',
      // Force `config` undefined despite the type contract.
      config: undefined as unknown as Record<string, unknown>,
    });
    const root = makeBlock('sequence', {
      id: 'root-id',
      label: 'Root',
      children: [malformed],
    });
    useAgentEditorStore.setState({ ast: makeAst(root) });

    expect(() =>
      render(<BlockEditor registry={sampleRegistry} />),
    ).not.toThrow();
    expect(screen.getByText('Malformed Plan')).toBeInTheDocument();
  });
});
