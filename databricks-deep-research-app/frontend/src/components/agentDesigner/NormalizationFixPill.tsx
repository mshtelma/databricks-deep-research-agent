/**
 * NormalizationFixPill — collapsible amber pill that surfaces the Designer's
 * Layer 2 auto-repair (see `.omc/plans/designer-hardening.md`, Layer 2 and
 * `.omc/plans/frontend-normalization-fix-surface.md` for the full spec).
 *
 * When the architect emits an AST the framework cannot run as-is
 * (`subtype: lane_researcher`, `model_tier: standard`, researcher with no
 * retrieval tools, etc.), the backend normalizer deterministically rewrites
 * the AST AND emits a `NormalizationFix` record per rewrite. The UI surfaces
 * those records here so the user can see exactly what was repaired —
 * "nothing silent" is the guiding principle.
 *
 * Behavior:
 *   - Zero fixes → component renders nothing.
 *   - 1+ fixes → renders a single-line pill with a chevron toggle. Clicking
 *     expands a panel with one row per fix (kind, path, before→after,
 *     rationale).
 *   - Settings: when `compact` is true (controlled by the parent's
 *     "Show designer auto-repair details" toggle), the pill collapses to a
 *     single "!" indicator without the count or expandable details.
 *   - Unknown `kind` values render with a generic icon + raw kind label —
 *     no JS error, forward-compatible with future normalizer rules.
 */

import * as React from 'react';
import {
  AlertTriangle,
  Braces,
  ChevronDown,
  ChevronUp,
  Cpu,
  Database,
  GitMerge,
  Hammer,
  Layers,
  Plus,
  Wrench,
  type LucideIcon,
} from 'lucide-react';
import type { NormalizationFix } from '@/types/agentDesigner';

// ---------------------------------------------------------------------------
// Public props
// ---------------------------------------------------------------------------

export interface NormalizationFixPillProps {
  fixes: NormalizationFix[];
  /**
   * When true, render the compact "!" indicator instead of count + expandable
   * panel. Reflects the user-preference toggle "Show designer auto-repair
   * details". Default: false (full pill).
   */
  compact?: boolean;
}

// ---------------------------------------------------------------------------
// Kind → icon + label registry
// ---------------------------------------------------------------------------

interface KindMeta {
  /** Lucide icon component for this fix kind. */
  icon: LucideIcon;
  /** Short user-facing label for the row. */
  label: string;
}

const KIND_META: Record<string, KindMeta> = {
  subtype_rewrite: { icon: Cpu, label: 'Subtype rewritten' },
  tier_rewrite: { icon: Layers, label: 'Model tier rewritten' },
  tool_kind_rewrite: { icon: Hammer, label: 'Tool kind rewritten' },
  auto_bind_retrieval: { icon: Plus, label: 'Retrieval tools auto-bound' },
  auto_declare_pool: { icon: Database, label: 'Pool auto-declared' },
  set_minimum_max_tool_calls: {
    icon: Wrench,
    label: 'Tool-call budget raised',
  },
  // Added in the second pass — search/crawl tools merged into web_research.
  tool_consolidation: { icon: GitMerge, label: 'Tools consolidated' },
  // Added when literal { } in JSON-shaped prompts get Jinja-escaped to {{ }}.
  brace_escape: { icon: Braces, label: 'Prompt braces escaped' },
};

function metaFor(kind: string): KindMeta {
  return (
    KIND_META[kind] ?? {
      icon: AlertTriangle,
      label: kind,
    }
  );
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/** Render an arbitrary before/after value as a compact string. */
function formatValue(v: unknown): string {
  if (v === null || v === undefined) {
    return '∅';
  }
  if (typeof v === 'string') {
    return JSON.stringify(v);
  }
  if (typeof v === 'number' || typeof v === 'boolean') {
    return String(v);
  }
  try {
    const out = JSON.stringify(v);
    return out.length > 120 ? `${out.slice(0, 117)}…` : out;
  } catch {
    return String(v);
  }
}

async function copyToClipboard(text: string): Promise<boolean> {
  try {
    if (typeof navigator !== 'undefined' && navigator.clipboard?.writeText) {
      await navigator.clipboard.writeText(text);
      return true;
    }
  } catch {
    // fall through
  }
  return false;
}

// ---------------------------------------------------------------------------
// Detail row
// ---------------------------------------------------------------------------

interface FixRowProps {
  fix: NormalizationFix;
}

function FixRow({ fix }: FixRowProps): React.ReactElement {
  const [copied, setCopied] = React.useState(false);
  const meta = metaFor(fix.kind);
  const Icon = meta.icon;

  const onCopyPath = React.useCallback(async () => {
    const ok = await copyToClipboard(fix.path);
    if (ok) {
      setCopied(true);
      window.setTimeout(() => setCopied(false), 1500);
    }
  }, [fix.path]);

  return (
    <li className="flex gap-2 py-1.5">
      <Icon
        size={12}
        strokeWidth={2.2}
        className="mt-[2px] shrink-0 text-db-yellow-700"
        aria-hidden
      />
      <div className="min-w-0 flex-1">
        <div className="flex items-center gap-1.5 text-[11px] font-medium text-db-navy-800">
          <span>{meta.label}</span>
          <span className="text-db-gray-text" aria-hidden>
            ·
          </span>
          <button
            type="button"
            onClick={onCopyPath}
            title={copied ? 'Copied!' : 'Copy AST path'}
            className="truncate font-db-mono text-[10px] font-normal text-db-gray-text transition-colors hover:text-db-navy-800"
          >
            {fix.path}
          </button>
          {copied && (
            <span
              aria-live="polite"
              className="font-db-mono text-[10px] text-db-green-700"
            >
              Copied
            </span>
          )}
        </div>
        <div className="mt-0.5 font-db-mono text-[10px] leading-snug text-db-gray-text">
          <span className="text-db-lava-700">{formatValue(fix.before)}</span>
          <span className="mx-1">→</span>
          <span className="text-db-green-700">{formatValue(fix.after)}</span>
        </div>
        {fix.rationale && (
          <div className="mt-0.5 text-[11px] leading-snug text-db-navy-800">
            {fix.rationale}
          </div>
        )}
      </div>
    </li>
  );
}

// ---------------------------------------------------------------------------
// Pill (zero-fix => null)
// ---------------------------------------------------------------------------

export function NormalizationFixPill({
  fixes,
  compact = false,
}: NormalizationFixPillProps): React.ReactElement | null {
  const [expanded, setExpanded] = React.useState(false);

  if (!fixes || fixes.length === 0) {
    return null;
  }

  const count = fixes.length;
  const ariaLabel = `Auto-repair: ${count} ${count === 1 ? 'issue' : 'issues'} fixed`;

  // Compact mode: single "!" indicator. No count, no expand panel.
  if (compact) {
    return (
      <span
        role="status"
        aria-label={ariaLabel}
        title={ariaLabel}
        className="inline-flex h-5 w-5 items-center justify-center rounded-full bg-db-yellow-300 text-[10px] font-semibold text-db-yellow-800"
      >
        !
      </span>
    );
  }

  return (
    <div
      className="rounded-db-md border border-db-yellow-600/40 bg-db-yellow-300/30"
      data-testid="normalization-fix-pill"
    >
      <button
        type="button"
        onClick={() => setExpanded((v) => !v)}
        aria-expanded={expanded}
        aria-label={ariaLabel}
        className="inline-flex w-full items-center gap-1.5 px-2.5 py-1.5 text-left text-[11px] font-medium text-db-yellow-800 transition-colors hover:bg-db-yellow-300/60"
      >
        <Wrench size={12} strokeWidth={2.4} aria-hidden />
        <span>
          Designer · auto-fixed {count} {count === 1 ? 'issue' : 'issues'}
        </span>
        <span className="ml-auto inline-flex items-center text-db-yellow-700">
          {expanded ? <ChevronUp size={12} /> : <ChevronDown size={12} />}
        </span>
      </button>
      {expanded && (
        <ul
          aria-label="Designer auto-repair details"
          className="border-t border-db-yellow-600/30 px-2.5 py-1.5"
        >
          {fixes.map((fix, idx) => (
            <FixRow key={`${fix.kind}-${fix.path}-${idx}`} fix={fix} />
          ))}
        </ul>
      )}
    </div>
  );
}
