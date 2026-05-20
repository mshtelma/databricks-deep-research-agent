/**
 * Shared metadata for the 8 workflow node types in the Agentic Designer UI.
 *
 * Used by BlockNode (rail color, type pill), ConfigPanel/Inspector (header pill),
 * and AddBlockMenu (type-color tiles). Colors come from the Databricks design
 * tokens defined in `globals.css`.
 */

import type { NodeType } from '@/types/ast';

export interface BlockTypeMeta {
  /** Uppercase mono label rendered in the TypePill (e.g. "AGENT"). */
  label: string;
  /** CSS color (e.g. "var(--db-lava-700)") used for the left rail and pill text. */
  color: string;
  /** CSS background tint used for the pill and selected-block ring. */
  bg: string;
  /** Lucide icon name (kept as a string so consumers can resolve it themselves). */
  icon: string;
  /** Short description used in inspector/help text and AddBlockMenu tiles. */
  description: string;
}

export const BLOCK_TYPE_META: Record<NodeType, BlockTypeMeta> = {
  sequence: {
    label: 'SEQUENCE',
    color: '#1B3139', // db-navy-800
    bg: '#E5E8EA',
    icon: 'sequence',
    description: 'Run children in order',
  },
  parallel: {
    label: 'PARALLEL',
    color: '#00875C', // db-green-700
    bg: '#D6EFE5',
    icon: 'parallel',
    description: 'Fan-out concurrent execution',
  },
  loop: {
    label: 'LOOP',
    color: '#7D5319', // db-yellow-800
    bg: '#FAEBCA',
    icon: 'loop',
    description: 'Repeat children until a condition or max iterations',
  },
  conditional: {
    label: 'CONDITIONAL',
    color: '#4A121A', // db-maroon-800
    bg: '#EDD8DA',
    icon: 'branch',
    description: 'Branch based on a predicate',
  },
  agent: {
    label: 'AGENT',
    color: '#BD2B26', // db-lava-700
    bg: '#FFE5E0', // db-lava-100
    icon: 'agent',
    description: 'Single ReAct agent',
  },
  tool: {
    label: 'TOOL',
    color: '#0E538B', // db-blue-700
    bg: '#DDEBF7', // db-blue-100
    icon: 'tool',
    description: 'A direct tool invocation',
  },
  subworkflow: {
    label: 'SUBWORKFLOW',
    color: '#4A121A', // db-maroon-800
    bg: '#EDD8DA',
    icon: 'subagent',
    description: 'Delegated workflow',
  },
  plan_and_execute: {
    label: 'PLAN_AND_EXECUTE',
    color: '#0E538B', // db-blue-700
    bg: '#DDEBF7',
    icon: 'branch',
    description: 'Planner loop with body steps',
  },
};

export function getBlockTypeMeta(type: NodeType): BlockTypeMeta {
  return BLOCK_TYPE_META[type] ?? BLOCK_TYPE_META.agent;
}
