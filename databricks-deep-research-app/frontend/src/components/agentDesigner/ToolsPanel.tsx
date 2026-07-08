/**
 * ToolsPanel — left/top panel showing declared tools for the current workflow.
 *
 * Behaviour:
 * - Reads ast.tools and selectedPath from useAgentEditorStore.
 * - "Add Tool" button opens AddToolDialog.
 * - Each tool row shows: layer-coloured kind badge, tool name, brief summary,
 *   and a remove button (with window.confirm guard).
 * - When selectedPath resolves to an agent block, "Bind Tools" button is enabled
 *   and opens BindToolDialog for that agent.
 * - Empty state: "No tools declared yet".
 *
 * Layer colour for badges is driven by registry.tool_kinds[].layer, with an
 * index-based fallback for older registry payloads.
 */

import * as React from 'react';
import { useAgentEditorStore } from '@/stores/agentEditorStore';
import { resolveBlock } from '@/lib/blockPath';
import { AddToolDialog } from './AddToolDialog';
import { BindToolDialog } from './BindToolDialog';
import type { RegistryResponse } from '@/types/agentDesigner';
import type { Block } from '@/types/ast';

/**
 * How many places in the workflow reference a declared tool: agent bindings
 * (config.tools, incl. plan_and_execute planner/evaluator) plus tool-node refs.
 */
function usedByCount(root: Block | undefined, name: string): number {
  if (!root) return 0;
  let count = 0;
  const visit = (block: Block): void => {
    const cfg = (block.config ?? {}) as Record<string, unknown>;
    const bound = cfg['tools'];
    if (Array.isArray(bound) && bound.includes(name)) count += 1;
    if (block.type === 'tool') {
      const ref = cfg['ref'];
      const refName =
        typeof ref === 'string'
          ? ref
          : ref && typeof ref === 'object'
            ? (ref as { name?: unknown }).name
            : undefined;
      if (refName === name) count += 1;
    }
    for (const nestedKey of ['planner', 'evaluator']) {
      const nested = cfg[nestedKey];
      if (nested && typeof nested === 'object' && !Array.isArray(nested)) {
        const nestedTools = (nested as Record<string, unknown>)['tools'];
        if (Array.isArray(nestedTools) && nestedTools.includes(name)) count += 1;
      }
    }
    const body = cfg['body'];
    if (body && typeof body === 'object' && !Array.isArray(body) && 'type' in body) {
      visit(body as Block);
    }
    for (const child of block.children ?? []) visit(child);
  };
  visit(root);
  return count;
}

// ---------------------------------------------------------------------------
// Layer badge styles (matches AddToolDialog layer mapping) — drawn from the
// Databricks Agentic Designer palette so the panel stays visually coherent
// with the rest of the designer surface.
// ---------------------------------------------------------------------------

const LAYER_BADGE_STYLES = [
  'bg-db-blue-100 text-db-blue-700',
  'bg-db-green-300 text-db-green-700',
  'bg-db-maroon-300 text-db-maroon-700',
  'bg-db-yellow-300 text-db-yellow-700',
] as const;

function layerBadgeStyle(layer: string | undefined, kindIndex: number): string {
  if (layer === 'A') return LAYER_BADGE_STYLES[0];
  if (layer === 'B') return LAYER_BADGE_STYLES[1];
  if (layer === 'C') return LAYER_BADGE_STYLES[2];
  if (layer === 'D') return LAYER_BADGE_STYLES[3];
  const tier = Math.floor(kindIndex / 3);
  return LAYER_BADGE_STYLES[Math.min(tier, LAYER_BADGE_STYLES.length - 1)] ?? LAYER_BADGE_STYLES[0];
}

// ---------------------------------------------------------------------------
// Props
// ---------------------------------------------------------------------------

export interface ToolsPanelProps {
  registry: RegistryResponse;
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export function ToolsPanel({ registry }: ToolsPanelProps): React.ReactElement {
  const ast = useAgentEditorStore((s) => s.ast);
  const selectedPath = useAgentEditorStore((s) => s.selectedPath);

  const [addOpen, setAddOpen] = React.useState(false);
  const [bindOpen, setBindOpen] = React.useState(false);

  const tools = ast?.tools ?? [];

  // Check if the selected block is an agent
  const selectedBlock = React.useMemo(() => {
    if (!ast || !selectedPath) return null;
    return resolveBlock(ast, selectedPath);
  }, [ast, selectedPath]);

  const isAgentSelected = selectedBlock?.type === 'agent';

  // Build a kind→metadata map for badge colouring
  const kindMetaMap = React.useMemo(() => {
    const map = new Map<string, { index: number; layer?: string }>();
    registry.tool_kinds.forEach((tk, idx) => {
      map.set(tk.kind, { index: idx, layer: tk.layer });
    });
    return map;
  }, [registry.tool_kinds]);

  const handleRemove = React.useCallback(
    (name: string) => {
      if (window.confirm(`Remove tool "${name}"?`)) {
        useAgentEditorStore.getState().removeTool(name);
      }
    },
    [],
  );

  return (
    <div className="flex w-72 flex-col gap-3 overflow-y-auto border-r border-db-gray-lines bg-white p-4 font-db-sans">
      {/* Header */}
      <div className="flex items-center justify-between">
        <h2 className="text-[13px] font-medium text-db-navy-800">Tools</h2>
        <button
          type="button"
          onClick={() => setAddOpen(true)}
          className="inline-flex items-center gap-1 rounded-db-md bg-db-lava-600 px-2.5 py-1 text-[12px] font-medium text-white transition-colors hover:bg-db-lava-700"
        >
          Add Tool
        </button>
      </div>

      {/* Bind Tools button — enabled only when an agent block is selected */}
      <button
        type="button"
        onClick={() => setBindOpen(true)}
        disabled={!isAgentSelected}
        className="rounded-db-md border border-db-gray-lines bg-white px-3 py-1.5 text-[12px] font-medium text-db-navy-800 transition-colors hover:border-db-navy-300 hover:bg-db-oat-medium disabled:cursor-not-allowed disabled:opacity-40"
      >
        Bind Tools to Selected Agent
      </button>

      {/* Tool list */}
      {tools.length === 0 ? (
        <p className="text-[12px] text-db-gray-text">No tools declared yet</p>
      ) : (
        <ul className="space-y-2">
          {tools.map((tool) => {
            const kindMeta = kindMetaMap.get(tool.kind);
            const badgeStyle = layerBadgeStyle(kindMeta?.layer, kindMeta?.index ?? 0);
            const target =
              tool.config['function'] ??
              tool.config['index_name'] ??
              tool.config['space_id'] ??
              tool.config['endpoint_name'] ??
              tool.config['endpoint'] ??
              tool.config['import'] ??
              tool.config['key'] ??
              tool.config['tool_name'];
            const usage = usedByCount(ast?.root, tool.name);
            const summary = `${typeof target === 'string' && target ? target : '—'} · ${
              usage > 0 ? `used by ${usage}` : 'unused'
            }`;

            return (
              <li
                key={tool.name}
                className="flex items-center gap-2 rounded-db-md border border-db-gray-lines bg-white p-2 shadow-db-xs"
              >
                {/* Kind badge */}
                <span
                  className={[
                    'shrink-0 rounded-sm px-1.5 py-0.5 font-db-mono text-[10px] font-semibold uppercase tracking-[0.04em]',
                    badgeStyle,
                  ].join(' ')}
                >
                  {tool.kind}
                </span>

                {/* Name + summary */}
                <div className="min-w-0 flex-1">
                  <p className="truncate font-db-mono text-[12px] font-medium text-db-navy-800">{tool.name}</p>
                  <p className="truncate text-[11px] text-db-gray-text">{summary}</p>
                </div>

                {/* Remove button */}
                <button
                  type="button"
                  onClick={() => handleRemove(tool.name)}
                  className="shrink-0 rounded-sm p-1 text-db-gray-text transition-colors hover:bg-db-lava-300 hover:text-db-lava-800"
                  aria-label={`Remove tool ${tool.name}`}
                >
                  ✕
                </button>
              </li>
            );
          })}
        </ul>
      )}

      {/* Dialogs */}
      <AddToolDialog
        registry={registry}
        open={addOpen}
        onOpenChange={setAddOpen}
      />

      {isAgentSelected && selectedPath !== null && (
        <BindToolDialog
          blockPath={selectedPath}
          open={bindOpen}
          onOpenChange={setBindOpen}
        />
      )}
    </div>
  );
}
