/**
 * AddBlockMenu — popover that lets the user pick a node type to insert as a
 * child of the given parent block.
 *
 * Filtering strategy:
 *   The NodeTypeSpec type returned by the registry does NOT include an
 *   `allowed_children` field (as of US-304 registry shape). We therefore
 *   apply a built-in rule:
 *     - Leaf types (agent, tool, subworkflow) never accept children.
 *     - A `tool` block cannot be nested directly inside another `tool`.
 *     - A `subworkflow` block cannot be nested inside a `subworkflow`.
 *     - All other node types are permitted as children of composite parents.
 *
 *   If the registry ever gains an `allowed_children` array on NodeTypeSpec,
 *   this component should prefer that field and fall back to the built-in
 *   rules only when it is absent.
 */

import * as React from 'react';
import { Popover, PopoverContent, PopoverTrigger } from '@/components/ui/popover';
import { useAgentEditorStore } from '@/stores/agentEditorStore';
import { getBlockTypeMeta } from '@/lib/blockTypeMeta';
import { BlockTypeIcon } from './atoms';
import type { BlockPath, NodeType } from '@/types/ast';
import type { NodeTypeSpec, RegistryResponse } from '@/types/agentDesigner';

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface AddBlockMenuProps {
  parentPath: BlockPath;
  parentNodeType: NodeType;
  registry: RegistryResponse;
  children: React.ReactNode;
  /** Optional controlled open state (e.g. driven by keyboard shortcut). */
  open?: boolean;
  onOpenChange?: (open: boolean) => void;
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/** Leaf types — cannot host children under any circumstances. */
const LEAF_TYPES: ReadonlySet<string> = new Set(['agent', 'tool', 'subworkflow']);

/**
 * Return the node types from the registry that are allowed as children of a
 * block whose type is `parentNodeType`.
 *
 * Rules (see JSDoc above for rationale):
 *   1. If the parent itself is a leaf, return an empty list.
 *   2. If NodeTypeSpec carries `allowed_children`, use it.
 *   3. Otherwise, allow all non-leaf types plus `agent` and `tool`, but
 *      exclude `tool`-under-`tool` and `subworkflow`-under-`subworkflow`.
 */
function filterAllowedChildren(
  registry: RegistryResponse,
  parentNodeType: NodeType,
): NodeTypeSpec[] {
  if (LEAF_TYPES.has(parentNodeType)) return [];

  return registry.node_types.filter((spec) => {
    // If the registry provides explicit allowed_children, respect it.
    const specWithAllowed = spec as NodeTypeSpec & { allowed_children?: string[] };
    if (Array.isArray(specWithAllowed.allowed_children)) {
      return specWithAllowed.allowed_children.includes(parentNodeType);
    }

    // Built-in fallback rules:
    // Don't allow nesting tool under tool or subworkflow under subworkflow.
    if (spec.type === 'tool' && parentNodeType === 'tool') return false;
    if (spec.type === 'subworkflow' && parentNodeType === 'subworkflow') return false;

    return true;
  });
}

/** Group an array of NodeTypeSpec by their `category` field. */
function groupByCategory(specs: NodeTypeSpec[]): Map<string, NodeTypeSpec[]> {
  const map = new Map<string, NodeTypeSpec[]>();
  for (const spec of specs) {
    const cat = spec.category || 'other';
    const existing = map.get(cat);
    if (existing) {
      existing.push(spec);
    } else {
      map.set(cat, [spec]);
    }
  }
  return map;
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export function AddBlockMenu({
  parentPath,
  parentNodeType,
  registry,
  children,
  open: controlledOpen,
  onOpenChange: controlledOnOpenChange,
}: AddBlockMenuProps): React.ReactElement {
  const [internalOpen, setInternalOpen] = React.useState(false);
  const open = controlledOpen !== undefined ? controlledOpen : internalOpen;
  const setOpen = controlledOnOpenChange ?? setInternalOpen;

  const allowed = React.useMemo(
    () => filterAllowedChildren(registry, parentNodeType),
    [registry, parentNodeType],
  );

  const grouped = React.useMemo(() => groupByCategory(allowed), [allowed]);

  function handleSelect(spec: NodeTypeSpec): void {
    useAgentEditorStore.getState().addBlock(
      parentPath,
      spec.type as NodeType,
      spec.label,
      spec.default_config ?? {},
    );
    setOpen(false);
  }

  return (
    <Popover open={open} onOpenChange={setOpen}>
      <PopoverTrigger asChild>{children}</PopoverTrigger>
      <PopoverContent
        className="db-root w-[360px] rounded-db-md border border-db-gray-lines bg-white p-3 font-db-sans shadow-db-md"
        align="start"
        sideOffset={6}
      >
        <div className="mb-2 px-1">
          <div className="text-[13px] font-medium text-db-navy-800">Add block</div>
          <div className="text-[11px] text-db-gray-text">Pick a node type to insert.</div>
        </div>
        {allowed.length === 0 ? (
          <p className="px-2 py-2 text-[12px] text-db-gray-text">No allowed children</p>
        ) : (
          <div className="flex flex-col gap-3">
            {Array.from(grouped.entries()).map(([category, specs]) => (
              <div key={category}>
                <p className="mb-1.5 px-1 font-db-sans text-[10px] font-semibold uppercase tracking-[0.06em] text-db-navy-400">
                  {category}
                </p>
                <div className="grid grid-cols-2 gap-1.5">
                  {specs.map((spec) => {
                    const meta = getBlockTypeMeta(spec.type as NodeType);
                    return (
                      <button
                        key={spec.type}
                        role="menuitem"
                        onClick={() => handleSelect(spec)}
                        className="flex flex-col gap-1.5 rounded-db-md border border-db-gray-lines bg-white p-2.5 text-left transition-colors hover:bg-db-oat-light hover:shadow-db-xs focus:outline-none"
                        style={{
                          borderLeftWidth: 3,
                          borderLeftStyle: 'solid',
                          borderLeftColor: meta.color,
                        }}
                      >
                        <div className="flex items-center gap-1.5">
                          <BlockTypeIcon
                            type={spec.type as NodeType}
                            size={12}
                            color={meta.color}
                          />
                          <span className="text-[12px] font-medium text-db-navy-800">
                            {spec.label}
                          </span>
                        </div>
                        <div className="text-[11px] leading-[1.45] text-db-gray-text">
                          {meta.description}
                        </div>
                      </button>
                    );
                  })}
                </div>
              </div>
            ))}
          </div>
        )}
      </PopoverContent>
    </Popover>
  );
}
