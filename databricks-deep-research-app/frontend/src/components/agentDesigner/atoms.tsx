/**
 * Atoms for the Agentic Designer UI — TypePill, Badge, LayerChip.
 *
 * These are intentionally tiny presentational primitives so they can be reused
 * across the BlockNode tree, the Inspector, and the AddBlock / AddTool dialogs
 * without coupling them to any one consumer's layout decisions.
 */

import * as React from 'react';
import { cn } from '@/lib/utils';
import { getBlockTypeMeta } from '@/lib/blockTypeMeta';
import type { NodeType } from '@/types/ast';
import {
  Layers,
  GitBranch,
  Bot,
  Box,
  Repeat,
  Network,
  Wrench,
  Workflow,
  type LucideIcon,
} from 'lucide-react';

// ---------------------------------------------------------------------------
// Icon resolution for block type meta
// ---------------------------------------------------------------------------

const ICON_MAP: Record<string, LucideIcon> = {
  sequence: Layers,
  parallel: Network,
  loop: Repeat,
  branch: GitBranch,
  agent: Bot,
  tool: Wrench,
  subagent: Box,
  team: Box,
  workflow: Workflow,
};

export function BlockTypeIcon({
  type,
  size = 11,
  className,
  color,
}: {
  type: NodeType;
  size?: number;
  className?: string;
  color?: string;
}): React.ReactElement {
  const meta = getBlockTypeMeta(type);
  const Cmp = ICON_MAP[meta.icon] ?? Box;
  return <Cmp size={size} className={className} style={color ? { color } : undefined} />;
}

// ---------------------------------------------------------------------------
// TypePill — uppercase mono badge with the node type label + icon
// ---------------------------------------------------------------------------

export function TypePill({ type }: { type: NodeType }): React.ReactElement {
  const meta = getBlockTypeMeta(type);
  return (
    <span
      className="inline-flex items-center gap-1 rounded-sm px-2 py-px font-db-mono text-[10px] font-semibold leading-3 tracking-[0.04em]"
      style={{ background: meta.bg, color: meta.color }}
    >
      <BlockTypeIcon type={type} size={11} />
      {meta.label}
    </span>
  );
}

// ---------------------------------------------------------------------------
// Badge — small pill-shaped chip used for subtype/model/tools/HITL labels
// ---------------------------------------------------------------------------

export interface BadgeProps {
  children: React.ReactNode;
  /** Foreground colour (text + leading icon) — accepts CSS values. */
  color?: string;
  /** Background colour. */
  bg?: string;
  /** Optional leading lucide icon. */
  icon?: React.ReactNode;
  mono?: boolean;
  className?: string;
  title?: string;
}

export function Badge({
  children,
  color = 'var(--db-navy-700)',
  bg = 'var(--db-oat-medium)',
  icon,
  mono = false,
  className,
  title,
}: BadgeProps): React.ReactElement {
  return (
    <span
      title={title}
      className={cn(
        'inline-flex items-center gap-1 whitespace-nowrap rounded-db-pill px-1.5 py-px text-[10px] font-medium leading-[14px]',
        mono ? 'font-db-mono tracking-[0.02em]' : 'font-db-sans',
        className,
      )}
      style={{ background: bg, color }}
    >
      {icon}
      {children}
    </span>
  );
}

// ---------------------------------------------------------------------------
// LayerChip — letter chip used to indicate tool layer (A/B/C/D/E)
// ---------------------------------------------------------------------------

const LAYER_CMAP: Record<string, [string, string]> = {
  A: ['#FF3621', '#FFE5E0'], // Web (lava)
  B: ['#0E538B', '#DDEBF7'], // Knowledge (blue)
  C: ['#00875C', '#D6EFE5'], // Data (green)
  D: ['#7D5319', '#FAEBCA'], // Filesystem (yellow)
  E: ['#4A121A', '#EDD8DA'], // MCP / Custom (maroon)
};

export function LayerChip({ layer }: { layer: string }): React.ReactElement {
  const tuple = LAYER_CMAP[layer] ?? LAYER_CMAP.A ?? ['#1B3139', '#EEEDE9'];
  const color = tuple[0];
  const bg = tuple[1];
  return (
    <span
      className="inline-flex h-4 w-4 items-center justify-center rounded-[3px] font-db-mono text-[9px] font-bold"
      style={{ background: bg, color }}
    >
      {layer}
    </span>
  );
}
