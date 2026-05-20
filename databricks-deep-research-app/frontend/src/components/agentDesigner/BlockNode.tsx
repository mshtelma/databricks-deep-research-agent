/**
 * BlockNode — renders a single Block from the workflow AST.
 *
 * Composite types (sequence, parallel, loop, conditional, plan_and_execute)
 * recursively render their children and expose an AddBlockMenu at the bottom.
 *
 * plan_and_execute stores its body at block.config.body per mutations.py
 * semantics. When config.body is a sequence wrapper, its children live at
 * config.body.children. This component handles that special case.
 *
 * Visual style — Databricks Agentic Designer:
 *  - Each block is a white card with a 3px left rail in the type's brand color
 *  - Type pill (mono badge), label, optional badges (subtype/model/tools/HITL)
 *  - Container blocks render a dashed colored guide + indented children
 *  - Selection: 3px ring matching the type's bg tint
 */

import * as React from 'react';
import { SortableContext, useSortable } from '@dnd-kit/sortable';
import { CSS } from '@dnd-kit/utilities';
import { GripVertical, Plus, Brain, Wrench, Lock } from 'lucide-react';
import { cn } from '@/lib/utils';
import { useAgentEditorStore } from '@/stores/agentEditorStore';
import { appendChildPath } from '@/lib/blockPath';
import { getBlockTypeMeta } from '@/lib/blockTypeMeta';
import { AddBlockMenu } from './AddBlockMenu';
import { TypePill, Badge } from './atoms';
import type { Block, BlockPath, NodeType } from '@/types/ast';
import type { RegistryResponse } from '@/types/agentDesigner';

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface BlockNodeProps {
  block: Block;
  path: BlockPath;
  registry: RegistryResponse;
  /** NodeType of the parent block — used by AddBlockMenu filtering. */
  parentNodeType?: NodeType;
}

// ---------------------------------------------------------------------------
// Composite types that can host children
// ---------------------------------------------------------------------------

const COMPOSITE_TYPES: ReadonlySet<NodeType> = new Set<NodeType>([
  'sequence',
  'parallel',
  'loop',
  'conditional',
  'plan_and_execute',
]);

function isComposite(nodeType: NodeType): boolean {
  return COMPOSITE_TYPES.has(nodeType);
}

// ---------------------------------------------------------------------------
// plan_and_execute body children resolver
// ---------------------------------------------------------------------------

function planBodyChildren(
  block: Block,
  path: BlockPath,
): { entries: Array<{ child: Block; path: BlockPath; label?: string }>; addPath: BlockPath } {
  const bodyPath = `${path}.config.body`;
  const body = (block.config as Record<string, unknown> | undefined)?.['body'] as Block | undefined;

  if (!body) {
    return { entries: [], addPath: bodyPath };
  }

  if (body.type === 'sequence' && Array.isArray(body.children)) {
    return {
      entries: body.children.map((child, idx) => ({
        child,
        path: `${bodyPath}.children.${idx}`,
        label: 'Body',
      })),
      addPath: bodyPath,
    };
  }

  return { entries: [{ child: body, path: bodyPath, label: 'Body' }], addPath: bodyPath };
}

function conditionalChildEntries(
  block: Block,
  path: BlockPath,
): Array<{ child: Block; path: BlockPath; label?: string }> {
  const conditions = Array.isArray(block.config.conditions) ? block.config.conditions : [];
  const defaultBranch =
    typeof block.config.default_branch === 'number'
      ? block.config.default_branch
      : conditions.length;
  return (block.children ?? []).map((child, idx) => ({
    child,
    path: appendChildPath(path, idx),
    label: idx === defaultBranch ? 'Default' : `Branch ${idx + 1}`,
  }));
}

// ---------------------------------------------------------------------------
// Subdescription
// ---------------------------------------------------------------------------

function subdescription(block: Block, registry: RegistryResponse): string {
  const cfg = (block.config ?? {}) as Record<string, unknown>;
  switch (block.type) {
    case 'loop':
      return `Loop × max ${(cfg['max_iterations'] as number | undefined) ?? '?'}`;
    case 'conditional':
      return `Conditional · ${block.children?.length ?? 0} branches`;
    case 'agent': {
      const instructions = (cfg['system_prompt'] ?? cfg['instructions']) as string | undefined;
      if (typeof instructions === 'string') {
        const firstLine = instructions.split('\n')[0]?.trim();
        if (firstLine) return firstLine;
      }
      return `Agent · ${(cfg['model_tier'] as string | undefined) ?? 'analytical'}`;
    }
    case 'tool':
      return `Tool · ${(cfg['tool_name'] as string | undefined) ?? '<unbound>'}`;
    case 'parallel':
      return `Parallel · ${block.children?.length ?? 0} children`;
    case 'sequence':
      return `Sequence · ${block.children?.length ?? 0} steps`;
    case 'subworkflow':
      return `Subworkflow · ${(cfg['subworkflow_id'] as string | undefined) ?? '<missing>'}`;
    case 'plan_and_execute': {
      const body = (cfg['body'] as Block | undefined);
      let count = 0;
      if (body?.type === 'sequence') count = body.children?.length ?? 0;
      else if (body) count = 1;
      return `Plan & Execute · ${count} ${count === 1 ? 'step' : 'steps'}`;
    }
    default:
      // Fall back to registry category description / block type
      void registry;
      return block.type;
  }
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export function BlockNode({
  block,
  path,
  registry,
  parentNodeType: _parentNodeType,
}: BlockNodeProps): React.ReactElement {
  const selectedPath = useAgentEditorStore((s) => s.selectedPath);
  const isSelected = selectedPath === path;

  const { attributes, listeners, setNodeRef, transform, transition, isDragging } = useSortable({
    id: path,
  });

  const meta = getBlockTypeMeta(block.type);

  const style: React.CSSProperties = {
    transform: CSS.Transform.toString(transform),
    transition,
    opacity: isDragging ? 0.5 : 1,
    borderLeftColor: meta.color,
    borderLeftWidth: 3,
    borderLeftStyle: 'solid',
    boxShadow: isSelected ? `0 0 0 3px ${meta.bg}` : 'var(--db-shadow-xs)',
    borderTopColor: isSelected ? meta.color : undefined,
    borderRightColor: isSelected ? meta.color : undefined,
    borderBottomColor: isSelected ? meta.color : undefined,
  };

  const [addMenuOpen, setAddMenuOpen] = React.useState(false);

  function handleKeyDown(e: React.KeyboardEvent<HTMLDivElement>): void {
    if (e.key === 'Delete') {
      e.preventDefault();
      useAgentEditorStore.getState().deleteBlock(path);
    }
    if (e.key === 'Enter' && isComposite(block.type)) {
      e.preventDefault();
      setAddMenuOpen(true);
    }
  }

  // -------------------------------------------------------------------------
  // Resolve children for composite types
  // -------------------------------------------------------------------------

  let compositeChildren: Array<{ child: Block; path: BlockPath; label?: string }> = [];
  let addPath: BlockPath = path;

  if (isComposite(block.type)) {
    if (block.type === 'plan_and_execute') {
      const resolved = planBodyChildren(block, path);
      compositeChildren = resolved.entries;
      addPath = resolved.addPath;
    } else if (block.type === 'conditional') {
      compositeChildren = conditionalChildEntries(block, path);
      addPath = path;
    } else {
      compositeChildren = (block.children ?? []).map((child, idx) => ({
        child,
        path: appendChildPath(path, idx),
      }));
      addPath = path;
    }
  }

  // -------------------------------------------------------------------------
  // Decoration badges
  // -------------------------------------------------------------------------

  const cfg = (block.config ?? {}) as Record<string, unknown>;
  const subtype = (cfg['subtype'] as string | undefined) ?? undefined;
  const modelTier = (cfg['model_tier'] as string | undefined) ?? (cfg['model'] as string | undefined);
  const boundTools = Array.isArray(cfg['tools']) ? (cfg['tools'] as string[]) : [];

  // HITL: surface when the block has explicitly opted in via config (e.g. an
  // agent's `requires_approval` / `hitl_enabled` flag). Tool-level approval
  // flags live on the workflow's ToolDecl (ast.tools), which we don't have in
  // this scope — surface them on the Approvals tab inside the Inspector.
  const hasGatedTool =
    cfg['requires_approval'] === true ||
    cfg['hitl_enabled'] === true ||
    cfg['approval_required'] === true;

  // -------------------------------------------------------------------------
  // Render
  // -------------------------------------------------------------------------

  return (
    <>
      <div
        ref={setNodeRef}
        style={style}
        tabIndex={0}
        onKeyDown={handleKeyDown}
        onClick={(e) => {
          e.stopPropagation();
          useAgentEditorStore.getState().select(path);
        }}
        className={cn(
          'db-anim-blockIn group relative cursor-pointer rounded-db-md border border-db-gray-lines bg-white transition-[box-shadow,border-color] focus:outline-none',
        )}
      >
        {/* Header row */}
        <div className="flex items-center gap-2.5 py-3 pl-3 pr-3.5">
          <button
            data-testid={`block-drag-handle-${path}`}
            aria-label="Drag to reorder"
            className="shrink-0 cursor-grab text-db-navy-300 transition-colors hover:text-db-navy-600 focus:outline-none active:cursor-grabbing"
            {...listeners}
            {...attributes}
            onClick={(e) => e.stopPropagation()}
          >
            <GripVertical size={14} strokeWidth={2.5} />
          </button>

          <TypePill type={block.type} />

          <span className="truncate text-[14px] font-medium text-db-navy-800">{block.label}</span>

          {subtype && (
            <Badge mono className="hidden sm:inline-flex">
              {subtype}
            </Badge>
          )}
          {modelTier && (
            <Badge
              color="var(--db-blue-700)"
              bg="var(--db-blue-100)"
              icon={<Brain size={10} />}
              className="hidden sm:inline-flex"
            >
              {modelTier}
            </Badge>
          )}
          {boundTools.length > 0 && (
            <Badge
              color="var(--db-navy-700)"
              bg="var(--db-oat-medium)"
              icon={<Wrench size={10} />}
              className="hidden md:inline-flex"
              title={boundTools.join(', ')}
            >
              {boundTools.length} {boundTools.length === 1 ? 'tool' : 'tools'}
            </Badge>
          )}
          {hasGatedTool && (
            <Badge
              color="var(--db-yellow-800)"
              bg="var(--db-yellow-300)"
              icon={<Lock size={10} />}
              title="Has tools that require human approval"
            >
              HITL
            </Badge>
          )}
        </div>

        {/* Subdescription line — indented to align with TypePill */}
        <div className="pb-3 pl-9 pr-3.5 text-[12px] text-db-gray-text">
          <span className="line-clamp-1">{subdescription(block, registry)}</span>
        </div>
      </div>

      {/* Children — composite container */}
      {isComposite(block.type) && (
        <div
          className="relative ml-[18px] mt-2.5 pl-[18px]"
          style={{
            borderLeftWidth: 1.5,
            borderLeftStyle: 'dashed',
            borderLeftColor: `${meta.color}40`,
          }}
        >
          {block.type === 'plan_and_execute' && (
            <div
              className="absolute -left-1.5 -top-1.5 bg-db-oat-light px-1.5 font-db-mono text-[9px] font-semibold tracking-[0.06em]"
              style={{ color: meta.color }}
            >
              BODY
            </div>
          )}
          {compositeChildren.length > 0 ? (
            <SortableContext items={compositeChildren.map((entry) => entry.path)}>
              <div className="flex flex-col gap-2.5">
                {compositeChildren.map(({ child, path: childPath, label }) => (
                  <div key={child.id} className="flex flex-col gap-1">
                    {label && label !== 'Body' && (
                      <span className="font-db-mono text-[10px] font-semibold uppercase tracking-[0.06em] text-db-navy-400">
                        {label}
                      </span>
                    )}
                    <BlockNode
                      block={child}
                      path={childPath}
                      registry={registry}
                      parentNodeType={block.type}
                    />
                  </div>
                ))}
              </div>
            </SortableContext>
          ) : null}

          {/* Add block button */}
          <div className="mt-2.5">
            <AddBlockMenu
              parentPath={addPath}
              parentNodeType={block.type}
              registry={registry}
              open={addMenuOpen}
              onOpenChange={setAddMenuOpen}
            >
              <button
                aria-label="Add block"
                className="inline-flex items-center gap-1.5 self-start rounded-db-md border border-dashed border-db-gray-lines bg-transparent px-3 py-1.5 font-db-sans text-[12px] font-medium text-db-gray-text transition-colors hover:border-[var(--db-navy-300)] hover:text-db-navy-800 focus:outline-none"
                onClick={(e) => e.stopPropagation()}
              >
                <Plus size={11} />
                Add block
              </button>
            </AddBlockMenu>
          </div>
        </div>
      )}
    </>
  );
}
