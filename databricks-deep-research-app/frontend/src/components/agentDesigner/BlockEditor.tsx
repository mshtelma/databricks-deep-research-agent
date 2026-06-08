/**
 * BlockEditor — root tree component for the Agent Designer workflow editor.
 *
 * Responsibilities:
 *  1. Empty-state: renders a placeholder with an "Add Root" button when ast is null.
 *  2. Non-empty state: wraps the recursive BlockNode tree in a <DndContext> +
 *     <SortableContext> to enable drag-and-drop reordering via @dnd-kit.
 *
 * SortableContext note:
 *   @dnd-kit/sortable requires every sortable item's id to be listed in its
 *   ancestor <SortableContext>. BlockNode registers each block with useSortable({ id: path }).
 *   We provide a single top-level SortableContext containing ALL paths from the
 *   entire flattened tree (collectPaths). Individual composite containers inside
 *   BlockNode can each add their own SortableContext for ordered sorting — this
 *   top-level one ensures every item is always registered so useSortable never
 *   throws about a missing context.
 */

import * as React from 'react';
import {
  DndContext,
  closestCorners,
  useSensor,
  useSensors,
  PointerSensor,
  KeyboardSensor,
  type DragEndEvent,
} from '@dnd-kit/core';
import {
  SortableContext,
  sortableKeyboardCoordinates,
} from '@dnd-kit/sortable';
import { useAgentEditorStore } from '@/stores/agentEditorStore';
import { isDescendant } from '@/lib/blockPath';
import { createDraftWorkflow } from '@/lib/workflowAst';
import { BlockNode } from './BlockNode';
import type { Block, BlockPath } from '@/types/ast';
import type { RegistryResponse } from '@/types/agentDesigner';

// ---------------------------------------------------------------------------
// Props
// ---------------------------------------------------------------------------

export interface BlockEditorProps {
  registry: RegistryResponse;
}

// ---------------------------------------------------------------------------
// Helper: collect every block path in the tree (DFS)
//
// Walks the Block tree starting at `block` with `path` and returns a flat
// list of all BlockPaths. The list is passed to the top-level SortableContext
// so that useSortable inside every BlockNode is registered correctly.
//
// ---------------------------------------------------------------------------

function collectPaths(block: Block, path: BlockPath): BlockPath[] {
  const paths: BlockPath[] = [path];
  const children = block.children ?? [];
  children.forEach((child, idx) => {
    const childPath: BlockPath = `${path}.children.${idx}`;
    paths.push(...collectPaths(child, childPath));
  });
  if (block.type === 'plan_and_execute') {
    const body = block.config?.body as Block | undefined;
    const bodyPath = `${path}.config.body`;
    if (body) {
      if (body.type === 'sequence') {
        for (const [idx, child] of (body.children ?? []).entries()) {
          paths.push(...collectPaths(child, `${bodyPath}.children.${idx}`));
        }
      } else {
        paths.push(...collectPaths(body, bodyPath));
      }
    }
  }
  return paths;
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export function BlockEditor({ registry }: BlockEditorProps): React.ReactElement {
  const ast = useAgentEditorStore((s) => s.ast);

  // -------------------------------------------------------------------------
  // DnD sensors (pointer + keyboard for a11y)
  // -------------------------------------------------------------------------

  const sensors = useSensors(
    useSensor(PointerSensor),
    useSensor(KeyboardSensor, { coordinateGetter: sortableKeyboardCoordinates }),
  );

  // -------------------------------------------------------------------------
  // Drag end handler
  // -------------------------------------------------------------------------

  function handleDragEnd(event: DragEndEvent): void {
    const activeId = event.active.id as BlockPath;
    const overId = event.over?.id as BlockPath | undefined;

    if (!overId || activeId === overId) return;

    // Reject cycles: dragging a block onto one of its own descendants
    if (isDescendant(activeId, overId)) {
      console.warn(
        `[BlockEditor] Drag rejected: "${overId}" is a descendant of "${activeId}".`,
      );
      return;
    }

    useAgentEditorStore.getState().moveBlock(activeId, overId);
  }

  // -------------------------------------------------------------------------
  // Empty state
  // -------------------------------------------------------------------------

  if (ast === null) {
    return (
      <div className="flex h-full flex-col items-center justify-center gap-3.5 text-db-gray-text">
        <p className="text-[13px]">No workflow yet.</p>
        <button
          data-testid="add-root-button"
          onClick={() => {
            useAgentEditorStore.setState({ ast: createDraftWorkflow() });
          }}
          className="inline-flex items-center gap-1.5 rounded-db-md bg-db-lava-600 px-3.5 py-1.5 text-[13px] font-medium text-white transition-colors hover:bg-db-lava-700 focus:outline-none focus:ring-2 focus:ring-db-lava-600 focus:ring-offset-1"
        >
          Add Root
        </button>
      </div>
    );
  }

  // -------------------------------------------------------------------------
  // Populated state
  // -------------------------------------------------------------------------

  const allPaths = collectPaths(ast.root, 'root');

  return (
    <DndContext
      sensors={sensors}
      collisionDetection={closestCorners}
      onDragEnd={handleDragEnd}
    >
      {/*
       * Top-level SortableContext lists every path in the tree so that
       * useSortable({ id: path }) inside BlockNode always finds a registered
       * SortableContext ancestor, regardless of nesting depth.
       */}
      <SortableContext items={allPaths}>
        <div className="flex flex-col gap-2.5">
          <BlockNode block={ast.root} path="root" registry={registry} />
        </div>
      </SortableContext>
    </DndContext>
  );
}
