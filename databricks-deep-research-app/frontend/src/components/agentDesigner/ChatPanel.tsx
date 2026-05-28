/**
 * ChatPanel — Designer Chat (co-pilot) sidebar for the Agent Designer.
 *
 * Streams transcript, pending mutations, error banner, and input. Visual
 * language: Databricks oat-light aside, lava-600 sparkle avatar, navy user
 * bubbles, dark-navy diff blocks inside PendingMutationCard.
 *
 * Collapsible: 340 px expanded → 44 px rail with sparkle button + pending
 * mutation count badge. Auto-collapses below 1280 px viewport.
 *
 * Scroll hardening — every flex-col ancestor in the chat aside has min-h-0
 * so that long transcripts don't push the aside past its allocated cross-axis
 * height. Without min-h-0 a `flex-1` child in a flex column defaults to
 * `min-height: auto` (its intrinsic content size), which defeats the
 * `overflow-auto` on the transcript and silently breaks scrolling once the
 * conversation grows past the viewport.
 *
 * Event rendering — tool_call args and tool_result payloads are parsed back
 * into structured JSON trees instead of dumping raw stringified JSON into a
 * text bubble. Tool results look up their originating tool_call by id so the
 * card can surface the tool name and link the request/response pair.
 */

import * as React from 'react';
import {
  Sparkles,
  Send,
  ChevronLeft,
  ChevronRight,
  ChevronDown,
  ChevronRight as ChevronRightIcon,
  Wrench,
  CheckCircle2,
  AlertCircle,
  AlertTriangle,
  Database,
} from 'lucide-react';
import { useChatSession } from '@/hooks/useChatSession';
import { useDesignerSettings } from '@/hooks/useDesignerSettings';
import { PendingMutationCard } from './PendingMutationCard';
import type { ChatMessage, DesignerAsset } from '@/types/agentDesigner';

// ---------------------------------------------------------------------------
// Props
// ---------------------------------------------------------------------------

export interface ChatPanelProps {
  sessionId?: string | null;
  assets?: DesignerAsset[] | (() => DesignerAsset[]);
}

// ---------------------------------------------------------------------------
// Markdown-ish renderer for assistant messages.
//
// Handles: bold (**...**), inline code (`...`), paragraph breaks (\n\n),
// soft line breaks (\n), bulleted lists (lines starting with "- " or "• "),
// and inserts an implicit paragraph break before emoji-led section markers
// like "🏗️ Architect:" / "🔍 Critic:" / "✅ Revision:" so single-blob LLM
// replies render with the same visual rhythm the model intended.
// ---------------------------------------------------------------------------

// Symbol/pictograph ranges — covers the emoji bands the model tends to use
// to flag sections (construction worker, magnifying glass, checkmark, etc.).
const SECTION_EMOJI_RANGE =
  '[\\u{1F300}-\\u{1FAFF}\\u{2600}-\\u{27BF}\\u{1F1E6}-\\u{1F1FF}\\u{1F900}-\\u{1F9FF}\\u{2700}-\\u{27BF}\\u2705\\u2728]';

function renderMarkdownHTML(text: string): string {
  // Escape HTML first so we never inject user-controlled markup.
  let s = text.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');

  // Promote in-line section-marker emojis to fresh paragraphs when they
  // follow sentence-ending punctuation, so "...dangerous). 🔍 Critic:..."
  // renders as two paragraphs instead of a wall of text.
  s = s.replace(
    new RegExp(`([.!?\\)])\\s+(${SECTION_EMOJI_RANGE})`, 'gu'),
    '$1\n\n$2',
  );

  // Inline formatting (bold, inline code).
  s = s
    .replace(/\*\*(.+?)\*\*/g, '<strong style="font-weight:600">$1</strong>')
    .replace(
      /`([^`]+)`/g,
      '<code style="font-family:var(--font-mono-db); font-size:12px; background:var(--db-oat-medium); padding:1px 5px; border-radius:3px">$1</code>',
    );

  // Split into paragraphs on blank lines, then handle each block.
  const blocks = s
    .split(/\n{2,}/)
    .map((b) => b.trim())
    .filter((b) => b.length > 0);

  const lastIdx = blocks.length - 1;
  return blocks
    .map((block, i) => {
      const margin = i === lastIdx ? '0' : '0 0 6px 0';

      // List block: every line starts with "- " or "• ".
      const lines = block.split('\n');
      const isList = lines.every((ln) => /^\s*(?:-|•)\s+/.test(ln));
      if (isList && lines.length > 0) {
        const items = lines
          .map((ln) => ln.replace(/^\s*(?:-|•)\s+/, ''))
          .map((it) => `<li style="margin:0">${it}</li>`)
          .join('');
        return `<ul style="margin:${margin};padding-left:1.1em;list-style:disc">${items}</ul>`;
      }

      // Default paragraph: soft line breaks become <br>.
      const inner = lines.join('<br>');
      return `<p style="margin:${margin}">${inner}</p>`;
    })
    .join('');
}

// ---------------------------------------------------------------------------
// JSON parsing
// ---------------------------------------------------------------------------

/** Safely parse JSON; returns the original string when parsing fails. */
function tryParseJson(raw: string): unknown {
  if (!raw) return raw;
  const trimmed = raw.trim();
  if (!trimmed || (trimmed[0] !== '{' && trimmed[0] !== '[' && trimmed[0] !== '"')) {
    return raw;
  }
  try {
    return JSON.parse(raw);
  } catch {
    return raw;
  }
}

/** Compact one-line summary of an object/array for collapsed previews. */
function summarizeValue(value: unknown): string {
  if (Array.isArray(value)) {
    return `Array(${value.length})`;
  }
  if (value !== null && typeof value === 'object') {
    const keys = Object.keys(value as Record<string, unknown>);
    if (keys.length === 0) return '{}';
    return `{ ${keys.slice(0, 3).join(', ')}${keys.length > 3 ? `, +${keys.length - 3}` : ''} }`;
  }
  return String(value);
}

// ---------------------------------------------------------------------------
// JsonTree — recursive renderer for structured event payloads
// ---------------------------------------------------------------------------

interface JsonTreeProps {
  value: unknown;
  depth?: number;
  /** Whether this subtree starts expanded. Top-level defaults to open. */
  defaultOpen?: boolean;
}

function JsonTree({ value, depth = 0, defaultOpen }: JsonTreeProps): React.ReactElement {
  // Primitives — render inline with a type color.
  if (value === null || value === undefined) {
    return <span className="font-db-mono text-[11px] text-db-navy-400">null</span>;
  }
  if (typeof value === 'boolean') {
    return (
      <span className="font-db-mono text-[11px] text-db-blue-700">{String(value)}</span>
    );
  }
  if (typeof value === 'number') {
    return (
      <span className="font-db-mono text-[11px] text-db-maroon-700">{String(value)}</span>
    );
  }
  if (typeof value === 'string') {
    // Long strings collapse so a single sprawling URL doesn't blow up the row.
    const display = value.length > 220 ? value.slice(0, 220) + '…' : value;
    return (
      <span
        className="break-words font-db-mono text-[11px] text-db-green-700"
        title={value.length > 220 ? value : undefined}
      >
        “{display}”
      </span>
    );
  }

  // Arrays + objects — collapsible.
  return <CollapsibleNode value={value} depth={depth} defaultOpen={defaultOpen} />;
}

function CollapsibleNode({
  value,
  depth,
  defaultOpen,
}: {
  value: unknown;
  depth: number;
  defaultOpen?: boolean;
}): React.ReactElement {
  const isArr = Array.isArray(value);
  const entries: Array<[string, unknown]> = isArr
    ? (value as unknown[]).map((v, i) => [String(i), v])
    : Object.entries(value as Record<string, unknown>);

  // Auto-expand the first two levels so the user immediately sees structure;
  // deeper levels stay collapsed to avoid wall-of-JSON.
  const initialOpen = defaultOpen ?? depth < 2;
  const [open, setOpen] = React.useState(initialOpen);

  if (entries.length === 0) {
    return (
      <span className="font-db-mono text-[11px] text-db-navy-400">
        {isArr ? '[]' : '{}'}
      </span>
    );
  }

  return (
    <div className="font-db-mono text-[11px]">
      <button
        type="button"
        onClick={(e) => {
          e.stopPropagation();
          setOpen((v) => !v);
        }}
        className="inline-flex items-center gap-1 rounded-sm px-1 py-px text-left text-db-gray-text transition-colors hover:bg-db-oat-medium hover:text-db-navy-800 focus:outline-none"
        aria-expanded={open}
      >
        {open ? <ChevronDown size={10} /> : <ChevronRightIcon size={10} />}
        <span className="text-db-navy-400">
          {isArr ? `[${entries.length}]` : `{${entries.length}}`}
        </span>
        {!open && (
          <span className="ml-1 truncate text-db-gray-text">{summarizeValue(value)}</span>
        )}
      </button>
      {open && (
        <ul className="mt-1 space-y-1 border-l border-db-gray-lines pl-3">
          {entries.map(([k, v]) => (
            <li key={k} className="flex flex-wrap items-start gap-1">
              <span className="font-medium text-db-navy-800">{k}</span>
              <span className="text-db-navy-300">:</span>
              <span className="min-w-0 flex-1">
                <JsonTree value={v} depth={depth + 1} />
              </span>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// ToolEventCard — wraps a tool call or tool result in a styled card.
// ---------------------------------------------------------------------------

interface ToolEventCardProps {
  kind: 'call' | 'result' | 'error';
  toolName: string;
  payload: unknown;
  /** Optional one-line headline (e.g. "3 sources added"). */
  headline?: string;
  /** Initial expanded state — calls collapse, results expand. */
  defaultOpen?: boolean;
}

function ToolEventCard({
  kind,
  toolName,
  payload,
  headline,
  defaultOpen,
}: ToolEventCardProps): React.ReactElement {
  const [open, setOpen] = React.useState(defaultOpen ?? kind === 'result');

  const kindMeta: Record<
    ToolEventCardProps['kind'],
    { color: string; bg: string; icon: React.ReactNode; label: string }
  > = {
    call: {
      color: 'text-db-blue-700',
      bg: 'bg-db-blue-100',
      icon: <Wrench size={11} />,
      label: 'Tool call',
    },
    result: {
      color: 'text-db-green-700',
      bg: 'bg-db-green-300',
      icon: <CheckCircle2 size={11} />,
      label: 'Tool result',
    },
    error: {
      color: 'text-db-lava-700',
      bg: 'bg-db-lava-100',
      icon: <AlertCircle size={11} />,
      label: 'Tool error',
    },
  };
  const meta = kindMeta[kind];

  return (
    <div className="overflow-hidden rounded-db-md border border-db-gray-lines bg-white">
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        aria-expanded={open}
        className="flex w-full items-center gap-2 px-2.5 py-1.5 text-left transition-colors hover:bg-db-oat-light focus:outline-none"
      >
        <span
          className={`inline-flex items-center gap-1 rounded-sm px-1.5 py-0.5 font-db-mono text-[10px] font-semibold uppercase tracking-[0.04em] ${meta.bg} ${meta.color}`}
        >
          {meta.icon}
          {meta.label}
        </span>
        <span className="truncate font-db-mono text-[11px] font-medium text-db-navy-800">
          {toolName}
        </span>
        {headline && (
          <span className="ml-1 truncate text-[11px] text-db-gray-text">
            · {headline}
          </span>
        )}
        <span className="ml-auto text-db-navy-300">
          {open ? <ChevronDown size={12} /> : <ChevronRightIcon size={12} />}
        </span>
      </button>
      {open && (
        <div className="border-t border-db-gray-lines bg-db-oat-light px-2.5 py-2">
          <JsonTree value={payload} defaultOpen />
        </div>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Heuristic headline for tool results — surfaces the most useful fact when
// the payload is the framework's standard shape.
// ---------------------------------------------------------------------------

function deriveResultHeadline(payload: unknown): string | undefined {
  if (!payload || typeof payload !== 'object' || Array.isArray(payload)) return undefined;
  const obj = payload as Record<string, unknown>;

  if (typeof obj['summary'] === 'string') return obj['summary'];
  if (typeof obj['message'] === 'string') return obj['message'];

  const fragments: string[] = [];
  if (typeof obj['node_count'] === 'number') fragments.push(`${obj['node_count']} nodes`);
  if (typeof obj['tool_count'] === 'number') fragments.push(`${obj['tool_count']} tools`);
  if (typeof obj['source_count'] === 'number') fragments.push(`${obj['source_count']} sources`);
  if (Array.isArray(obj['sources'])) fragments.push(`${(obj['sources'] as unknown[]).length} sources`);
  if (Array.isArray(obj['nodes'])) fragments.push(`${(obj['nodes'] as unknown[]).length} nodes`);
  if (obj['ok'] === true || obj['success'] === true) fragments.unshift('ok');
  if (obj['ok'] === false || obj['success'] === false) fragments.unshift('failed');

  return fragments.length > 0 ? fragments.join(' · ') : undefined;
}

// ---------------------------------------------------------------------------
// DesignerActivityCard — user-facing summaries for Designer init/progress
// payloads. Raw JSON stays available behind "Technical details".
// ---------------------------------------------------------------------------

type ActivityStatus = 'complete' | 'warning' | 'blocked' | 'neutral';

interface DesignerActivitySummary {
  status: ActivityStatus;
  title: string;
  headline: string;
  chips: string[];
  details?: string[];
  warnings?: string[];
}

function asRecord(value: unknown): Record<string, unknown> | null {
  if (!value || typeof value !== 'object' || Array.isArray(value)) return null;
  return value as Record<string, unknown>;
}

function countFrom(value: unknown): number | null {
  return typeof value === 'number' && Number.isFinite(value) ? value : null;
}

function arrayLength(value: unknown): number | null {
  return Array.isArray(value) ? value.length : null;
}

function plural(count: number, singular: string, pluralLabel = `${singular}s`): string {
  return `${count} ${count === 1 ? singular : pluralLabel}`;
}

function titleCaseToken(value: string): string {
  return value
    .replace(/[_-]+/g, ' ')
    .replace(/\s+/g, ' ')
    .trim()
    .replace(/\b\w/g, (ch) => ch.toUpperCase());
}

function compactPolicy(value: unknown): string | null {
  if (typeof value !== 'string' || value.length === 0) return null;
  if (value === 'corpus_only') return 'Corpus-only evidence';
  return titleCaseToken(value);
}

function compactToolKind(value: unknown): string | null {
  if (typeof value !== 'string' || value.length === 0) return null;
  if (value === 'vector_search') return 'Vector search';
  if (value === 'table_search') return 'Table search';
  if (value === 'table_read') return 'Table read';
  if (value === 'web_research') return 'Web research';
  return titleCaseToken(value);
}

function compactResourceKind(value: unknown): string | null {
  if (typeof value !== 'string' || value.length === 0) return null;
  if (value === 'vector_index') return 'vector index';
  if (value === 'delta_table') return 'Delta table';
  if (value === 'genie_space') return 'Genie space';
  if (value === 'knowledge_assistant') return 'knowledge assistant';
  return titleCaseToken(value).toLowerCase();
}

function firstString(values: unknown[]): string | null {
  for (const value of values) {
    if (typeof value === 'string' && value.trim().length > 0) return value.trim();
  }
  return null;
}

function stringList(value: unknown): string[] {
  return Array.isArray(value)
    ? value.filter((item): item is string => typeof item === 'string' && item.trim().length > 0)
    : [];
}

function diagnosticSummary(payload: Record<string, unknown>): {
  messages: string[];
  blocking: boolean;
} {
  const diagnostics = Array.isArray(payload['diagnostics']) ? payload['diagnostics'] : [];
  const messages: string[] = [];
  let blocking = false;
  for (const item of diagnostics) {
    const diagnostic = asRecord(item);
    if (!diagnostic) continue;
    const severity = typeof diagnostic['severity'] === 'string' ? diagnostic['severity'] : '';
    if (diagnostic['blocking'] === true || severity === 'error') blocking = true;
    const message = firstString([diagnostic['message'], diagnostic['code']]);
    if (message) messages.push(message);
  }
  return { messages, blocking };
}

function inferStatus(
  payload: Record<string, unknown>,
  fallback: ActivityStatus = 'complete',
): ActivityStatus {
  const diagnostics = diagnosticSummary(payload);
  if (diagnostics.blocking) return 'blocked';
  if (diagnostics.messages.length > 0) return 'warning';
  return fallback;
}

function summarizePromptGrounding(payload: Record<string, unknown>): DesignerActivitySummary {
  const mentions = countFrom(payload['mentions_count']);
  const resourceCount =
    countFrom(payload['resolved_assets_count']) ??
    arrayLength(payload['resolved_resources']) ??
    countFrom(payload['resources_count']) ??
    0;
  const readyTools = stringList(payload['ready_tool_kinds']).map(compactToolKind).filter(Boolean) as string[];
  const resourceKindsObj = asRecord(payload['resource_kinds']);
  const resourceKinds = resourceKindsObj
    ? Object.keys(resourceKindsObj).map(compactResourceKind).filter(Boolean) as string[]
    : [];
  const resourceKind = resourceKinds[0] ?? 'source';
  const diagnostics = diagnosticSummary(payload);
  const chips = [
    mentions !== null ? plural(mentions, 'mention') : null,
    plural(resourceCount, 'source'),
    ...readyTools.map((tool) => `${tool} ready`),
  ].filter((item): item is string => Boolean(item));

  return {
    status: inferStatus(payload, resourceCount > 0 ? 'complete' : 'neutral'),
    title: 'Checked selected sources',
    headline:
      resourceCount > 0
        ? `Found ${resourceCount} grounded ${resourceKind}${resourceCount === 1 ? '' : 's'} for this workflow.`
        : 'No grounded sources were found for this workflow yet.',
    chips,
    details: readyTools.length > 0 ? [`Ready capability: ${readyTools.join(', ')}`] : undefined,
    warnings: diagnostics.messages,
  };
}

function summarizeResourceSemantics(payload: Record<string, unknown>): DesignerActivitySummary {
  const available = payload['available'] === true;
  const resourceCount = countFrom(payload['resources_count']) ?? arrayLength(payload['resources']) ?? 0;
  const domainTerms = stringList(payload['task_domain_terms']);

  if (!available) {
    return {
      status: inferStatus(payload, 'neutral'),
      title: 'Checked data semantics',
      headline: 'No extra semantic profile was available, so Designer will use the grounded source metadata.',
      chips: ['Source metadata'],
    };
  }

  return {
    status: inferStatus(payload),
    title: 'Checked data semantics',
    headline: `Interpreted data needs for ${plural(resourceCount, 'source')}.`,
    chips: [plural(resourceCount, 'source'), ...domainTerms.slice(0, 2)],
    details:
      domainTerms.length > 0
        ? [`Detected terms: ${domainTerms.slice(0, 8).join(', ')}`]
        : undefined,
  };
}

function summarizeResolvedToolContract(payload: Record<string, unknown>): DesignerActivitySummary {
  const available = payload['available'] === true;
  const policy = compactPolicy(payload['evidence_policy']);
  const resourceCount = countFrom(payload['resources_count']) ?? arrayLength(payload['resources']) ?? 0;
  const readyTools = stringList(payload['ready_tool_kinds']).map(compactToolKind).filter(Boolean) as string[];
  const requiredCapabilities = stringList(payload['required_capabilities'])
    .map(compactToolKind)
    .filter(Boolean) as string[];
  const requiredTerms = stringList(payload['required_terms']);
  const obligations = stringList(payload['planner_obligations']);
  const diagnostics = diagnosticSummary(payload);
  const primaryTool = readyTools[0] ?? requiredCapabilities[0] ?? 'configured tools';

  if (!available) {
    return {
      status: inferStatus(payload, 'neutral'),
      title: 'Planned evidence access',
      headline: 'Designer has not resolved an evidence access plan yet.',
      chips: policy ? [policy] : [],
      warnings: diagnostics.messages,
    };
  }

  return {
    status: inferStatus(payload),
    title: 'Planned evidence access',
    headline:
      policy === 'Corpus-only evidence' && primaryTool === 'Vector search'
        ? 'Designer will answer from the named corpus using vector search.'
        : `Designer planned evidence access with ${primaryTool.toLowerCase()}.`,
    chips: [
      policy,
      resourceCount > 0 ? plural(resourceCount, 'resource') : null,
      ...readyTools,
    ].filter((item): item is string => Boolean(item)),
    details: [
      obligations.length > 0 ? `Planner must: ${obligations.join(' ')}` : null,
      requiredTerms.length > 0 ? `Required terms: ${requiredTerms.slice(0, 8).join(', ')}` : null,
    ].filter((item): item is string => Boolean(item)),
    warnings: diagnostics.messages,
  };
}

function summarizeDesignerToolResult(
  toolName: string,
  payload: unknown,
): DesignerActivitySummary | null {
  const obj = asRecord(payload);
  if (!obj) return null;
  const schema = typeof obj['schema'] === 'string' ? obj['schema'] : '';

  if (toolName === 'prompt_grounding' || schema === 'prompt_grounding.v1') {
    return summarizePromptGrounding(obj);
  }
  if (toolName === 'resource_semantics' || schema === 'resource_semantics.v1') {
    return summarizeResourceSemantics(obj);
  }
  if (toolName === 'resolved_tool_contract' || schema === 'resolved_tool_contract.v1') {
    return summarizeResolvedToolContract(obj);
  }
  return null;
}

function DesignerActivityCard({
  summary,
  payload,
}: {
  summary: DesignerActivitySummary;
  payload: unknown;
}): React.ReactElement {
  const [open, setOpen] = React.useState(false);
  const statusMeta: Record<
    ActivityStatus,
    { color: string; bg: string; border: string; icon: React.ReactNode; label: string }
  > = {
    complete: {
      color: 'text-db-green-700',
      bg: 'bg-db-green-300/40',
      border: 'border-db-green-300',
      icon: <CheckCircle2 size={13} />,
      label: 'Complete',
    },
    warning: {
      color: 'text-db-yellow-800',
      bg: 'bg-db-yellow-300/40',
      border: 'border-db-yellow-300',
      icon: <AlertTriangle size={13} />,
      label: 'Needs attention',
    },
    blocked: {
      color: 'text-db-lava-700',
      bg: 'bg-db-lava-100',
      border: 'border-db-lava-300',
      icon: <AlertCircle size={13} />,
      label: 'Blocked',
    },
    neutral: {
      color: 'text-db-blue-700',
      bg: 'bg-db-blue-100',
      border: 'border-db-gray-lines',
      icon: <Database size={13} />,
      label: 'Checked',
    },
  };
  const meta = statusMeta[summary.status];

  return (
    <div
      className={`overflow-hidden rounded-db-md border bg-white shadow-db-xs ${meta.border}`}
      aria-label={`${summary.title}, ${meta.label.toLowerCase()}`}
    >
      <div className="px-3 py-2.5">
        <div className="mb-1 flex items-start gap-2">
          <span
            className={`mt-0.5 inline-flex h-5 w-5 shrink-0 items-center justify-center rounded-[5px] ${meta.bg} ${meta.color}`}
          >
            {meta.icon}
          </span>
          <div className="min-w-0 flex-1">
            <div className="flex min-w-0 items-center gap-1.5">
              <span className="truncate text-[12px] font-semibold text-db-navy-800">
                {summary.title}
              </span>
              <span
                className={`rounded-sm px-1.5 py-0.5 font-db-mono text-[9px] font-semibold uppercase tracking-[0.04em] ${meta.bg} ${meta.color}`}
              >
                {meta.label}
              </span>
            </div>
            <p className="mt-0.5 text-[12px] leading-[1.45] text-db-gray-text">
              {summary.headline}
            </p>
          </div>
        </div>

        {summary.chips.length > 0 && (
          <div className="mt-2 flex flex-wrap gap-1">
            {summary.chips.map((chip) => (
              <span
                key={chip}
                className="rounded-db-pill border border-db-gray-lines bg-db-oat-light px-2 py-0.5 text-[10px] font-medium text-db-navy-800"
              >
                {chip}
              </span>
            ))}
          </div>
        )}

        {summary.warnings && summary.warnings.length > 0 && (
          <ul className="mt-2 list-disc space-y-0.5 pl-5 text-[11px] leading-[1.4] text-db-yellow-800">
            {summary.warnings.map((warning, idx) => (
              <li key={`${warning}-${idx}`}>{warning}</li>
            ))}
          </ul>
        )}

        {summary.details && summary.details.length > 0 && (
          <ul className="mt-2 space-y-0.5 text-[11px] leading-[1.45] text-db-gray-text">
            {summary.details.map((detail, idx) => (
              <li key={`${detail}-${idx}`}>{detail}</li>
            ))}
          </ul>
        )}

        <button
          type="button"
          onClick={() => setOpen((v) => !v)}
          aria-expanded={open}
          className="mt-2 inline-flex items-center gap-1 text-[11px] font-medium text-db-gray-text transition-colors hover:text-db-navy-800 focus:outline-none"
        >
          {open ? <ChevronDown size={11} /> : <ChevronRightIcon size={11} />}
          Technical details
        </button>
      </div>
      {open && (
        <div className="border-t border-db-gray-lines bg-db-oat-light px-2.5 py-2">
          <JsonTree value={payload} defaultOpen />
        </div>
      )}
    </div>
  );
}

function latestDesignerActivityTitle(messages: ChatMessage[]): string | null {
  for (let i = messages.length - 1; i >= 0; i -= 1) {
    const message = messages[i];
    if (!message || message.role !== 'tool') continue;
    const parsed = tryParseJson(message.content);
    const summary = summarizeDesignerToolResult(message.tool_name ?? 'tool', parsed);
    if (summary) return summary.title;
  }
  return null;
}

// ---------------------------------------------------------------------------
// Message row
// ---------------------------------------------------------------------------

interface MessageRowProps {
  message: ChatMessage;
  /** Whole transcript so a tool result can find its originating tool_call. */
  messages: ChatMessage[];
}

function MessageRow({ message, messages }: MessageRowProps): React.ReactElement | null {
  const isUser = message.role === 'user';
  const isTool = message.role === 'tool';

  if (isTool) {
    // Look up tool_name from the assistant message that issued this call so
    // the result card can label itself meaningfully instead of showing a raw
    // UUID. Falls back to "tool" when the call is missing (e.g., the user
    // joined a session mid-stream and the OPEN_ASSISTANT event was dropped).
    const callId = message.tool_call_id ?? null;
    let toolName = message.tool_name ?? 'tool';
    if (callId) {
      for (const m of messages) {
        if (m.role !== 'assistant' || !m.tool_calls) continue;
        const tc = m.tool_calls.find((c) => c.id === callId);
        if (tc) {
          toolName = tc.function.name;
          break;
        }
      }
    }
    const parsed = tryParseJson(message.content);
    const activitySummary = summarizeDesignerToolResult(toolName, parsed);
    if (activitySummary) {
      return (
        <div className="flex justify-start" data-role="tool">
          <div className="w-full min-w-0 max-w-full">
            <DesignerActivityCard summary={activitySummary} payload={parsed} />
          </div>
        </div>
      );
    }
    const headline = deriveResultHeadline(parsed);
    return (
      <div className="flex justify-start" data-role="tool">
        <div className="w-full min-w-0 max-w-full">
          <ToolEventCard
            kind="result"
            toolName={toolName}
            payload={parsed}
            headline={headline}
          />
        </div>
      </div>
    );
  }

  if (isUser) {
    return (
      <div className="flex justify-end" data-role="user">
        <div className="max-w-[280px]">
          <div className="rounded-db-md bg-db-navy-800 px-3 py-2 text-[13px] leading-[1.5] text-white">
            {message.content}
          </div>
        </div>
      </div>
    );
  }

  // assistant
  const hasContent = message.content && message.content.trim().length > 0;
  const hasToolCalls = message.tool_calls && message.tool_calls.length > 0;
  if (!hasContent && !hasToolCalls) {
    // Skip empty placeholder bubbles (OPEN_ASSISTANT before any delta) — the
    // streaming dot in the header already signals activity.
    return null;
  }

  return (
    <div className="flex min-w-0 items-start justify-start gap-2" data-role="assistant">
      <div className="mt-0.5 flex h-6 w-6 shrink-0 items-center justify-center rounded-[5px] bg-db-lava-600">
        <Sparkles size={12} className="text-white" />
      </div>
      <div className="min-w-0 flex-1 space-y-1.5">
        {hasContent && (
          <div
            className="break-words text-[13px] leading-[1.55] text-db-navy-800"
            dangerouslySetInnerHTML={{ __html: renderMarkdownHTML(message.content) }}
          />
        )}
        {hasToolCalls && (
          <div className="space-y-1">
            {message.tool_calls!.map((tc) => {
              const argsPayload = tryParseJson(tc.function.arguments);
              return (
                <ToolEventCard
                  key={tc.id}
                  kind="call"
                  toolName={tc.function.name}
                  payload={argsPayload}
                />
              );
            })}
          </div>
        )}
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// ChatPanel
// ---------------------------------------------------------------------------

const SUGGESTIONS = [
  'Add a reflector sub-agent',
  'Wire vector_search to the Researcher',
  'Make web_crawl HITL-gated',
];

export function ChatPanel({ sessionId, assets }: ChatPanelProps): React.ReactElement {
  const session = useChatSession({ sessionId, assets });
  const { settings, setShowAutoRepairDetails } = useDesignerSettings();

  const [inputText, setInputText] = React.useState('');
  const [collapsed, setCollapsed] = React.useState<boolean>(false);
  const bottomRef = React.useRef<HTMLDivElement>(null);
  const scrollRef = React.useRef<HTMLDivElement>(null);
  const latestActivity = React.useMemo(
    () => latestDesignerActivityTitle(session.messages),
    [session.messages],
  );

  React.useEffect(() => {
    const node = scrollRef.current;
    if (node) node.scrollTop = node.scrollHeight;
  }, [session.messages.length, session.pendingMutations.length]);

  React.useEffect(() => {
    void bottomRef;
  }, []);

  function handleSend(text?: string): void {
    const value = (text ?? inputText).trim();
    if (!value || session.isStreaming) return;
    setInputText('');
    void session.sendMessage(value);
  }

  function handleKeyDown(e: React.KeyboardEvent<HTMLTextAreaElement>): void {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  }

  // -------------------------------------------------------------------------
  // Collapsed rail
  // -------------------------------------------------------------------------

  if (collapsed) {
    return (
      <aside className="db-root flex w-11 shrink-0 flex-col items-center gap-2 border-l border-db-gray-lines bg-white pt-2.5 font-db-sans">
        <button
          type="button"
          aria-label="Expand designer chat"
          title="Expand designer chat"
          onClick={() => setCollapsed(false)}
          className="rounded p-1.5 text-db-gray-text hover:bg-db-oat-medium hover:text-db-navy-800"
        >
          <ChevronLeft size={14} />
        </button>
        <button
          type="button"
          aria-label="Designer Chat"
          title="Designer Chat"
          onClick={() => setCollapsed(false)}
          className="relative flex h-7 w-7 items-center justify-center rounded-[5px] bg-db-lava-600 text-white shadow-db-xs"
        >
          <Sparkles size={13} />
          {session.pendingMutations.length > 0 && (
            <span className="absolute -right-1 -top-1 flex h-4 min-w-[16px] items-center justify-center rounded-full bg-db-navy-800 px-1 font-db-mono text-[9px] font-bold text-white">
              {session.pendingMutations.length}
            </span>
          )}
        </button>
      </aside>
    );
  }

  // -------------------------------------------------------------------------
  // Expanded panel
  //
  // Layout invariants (so long transcripts scroll instead of pushing the
  // aside past the viewport):
  //   - aside has `min-h-0` so its parent's row constraint flows in
  //   - transcript wrapper has `min-h-0 flex-1 overflow-auto`; without
  //     `min-h-0` the wrapper's intrinsic min-height equals its content
  //     height and `overflow-auto` never engages
  // -------------------------------------------------------------------------

  return (
    <aside className="db-root flex h-full min-h-0 w-[340px] shrink-0 flex-col border-l border-db-gray-lines bg-db-oat-light font-db-sans">
      {/* Header */}
      <div className="flex shrink-0 items-center gap-2 border-b border-db-gray-lines bg-white px-4 py-3.5">
        <div className="flex h-6 w-6 items-center justify-center rounded-[5px] bg-db-lava-600">
          <Sparkles size={13} className="text-white" />
        </div>
        <span className="text-[13px] font-medium text-db-navy-800">Designer Chat</span>
        <span className="rounded-db-pill bg-db-blue-100 px-2 py-0.5 font-db-mono text-[10px] font-medium tracking-[0.02em] text-db-blue-700">
          co-pilot
        </span>
        {session.isStreaming && (
          <span className="ml-1 inline-flex min-w-0 items-center gap-1 text-[11px] text-db-blue-700">
            <span
              className="h-2 w-2 shrink-0 animate-pulse rounded-full bg-db-blue-700"
              aria-label="Streaming"
              data-testid="streaming-indicator"
            />
            <span className="max-w-[92px] truncate">{latestActivity ?? 'Working'}</span>
          </span>
        )}
        <button
          type="button"
          role="switch"
          aria-checked={settings.showAutoRepairDetails}
          aria-label="Show designer auto-repair details"
          title={
            settings.showAutoRepairDetails
              ? 'Hide designer auto-repair details'
              : 'Show designer auto-repair details'
          }
          onClick={() =>
            setShowAutoRepairDetails(!settings.showAutoRepairDetails)
          }
          className={
            'ml-auto rounded p-1 transition-colors ' +
            (settings.showAutoRepairDetails
              ? 'text-db-yellow-700 hover:bg-db-yellow-300/40'
              : 'text-db-gray-text hover:bg-db-oat-medium hover:text-db-navy-800')
          }
        >
          <Wrench size={13} />
        </button>
        <button
          type="button"
          aria-label="Collapse designer chat"
          title="Collapse designer chat"
          onClick={() => setCollapsed(true)}
          className="rounded p-1 text-db-gray-text hover:bg-db-oat-medium hover:text-db-navy-800"
        >
          <ChevronRight size={14} />
        </button>
      </div>

      {/* Transcript */}
      <div ref={scrollRef} className="min-h-0 flex-1 overflow-auto overscroll-contain px-3.5 py-4">
        <div className="flex min-w-0 flex-col gap-3.5">
          {/* Welcome state */}
          {session.messages.length === 0 && session.pendingMutations.length === 0 && !session.error && (
            <div className="flex items-start gap-2" data-role="assistant">
              <div className="mt-0.5 flex h-6 w-6 shrink-0 items-center justify-center rounded-[5px] bg-db-lava-600">
                <Sparkles size={12} className="text-white" />
              </div>
              <div className="text-[13px] leading-[1.55] text-db-navy-800">
                I can rewire blocks, add agents, bind tools, or set HITL gates. What change should
                I propose?
              </div>
            </div>
          )}

          {session.error && (
            <div
              role="alert"
              data-testid="error-banner"
              className="rounded-db-md border border-db-lava-300 bg-db-lava-100 px-3 py-2 text-[13px] text-db-lava-700"
            >
              {session.error}
            </div>
          )}

          {session.messages.map((msg, idx) => (
            <MessageRow key={idx} message={msg} messages={session.messages} />
          ))}

          {session.pendingMutations.map((mutation) => (
            <PendingMutationCard
              key={mutation.id}
              mutation={mutation}
              onApply={session.applyPendingMutation}
              onReject={session.rejectPendingMutation}
              showAutoRepairDetails={settings.showAutoRepairDetails}
            />
          ))}

          <div ref={bottomRef} />
        </div>
      </div>

      {/* Composer */}
      <div className="shrink-0 border-t border-db-gray-lines bg-white p-3">
        {session.messages.length === 0 && (
          <div className="mb-2 flex flex-wrap gap-1.5">
            {SUGGESTIONS.map((s) => (
              <button
                key={s}
                type="button"
                onClick={() => handleSend(s)}
                disabled={session.isStreaming}
                className="inline-flex items-center gap-1 rounded-db-pill border border-db-gray-lines bg-white px-2.5 py-1 text-[11px] font-medium text-db-navy-800 transition-colors hover:border-db-navy-300 hover:bg-db-oat-medium disabled:opacity-55"
              >
                <Sparkles size={10} className="text-db-lava-600" />
                {s}
              </button>
            ))}
          </div>
        )}
        <div className="flex items-end gap-1.5 rounded-db-md border border-db-gray-lines bg-db-oat-light p-2 transition-colors focus-within:border-db-navy-400 focus-within:shadow-db-focus">
          <textarea
            className="flex-1 resize-none border-0 bg-transparent p-0 font-db-sans text-[13px] leading-[1.45] text-db-navy-800 outline-none placeholder:text-db-gray-text disabled:opacity-55"
            rows={2}
            placeholder="Describe how to update the workflow…"
            value={inputText}
            onChange={(e) => setInputText(e.target.value)}
            onKeyDown={handleKeyDown}
            disabled={session.isStreaming}
            aria-label="Chat input"
          />
          {session.isStreaming ? (
            <button
              type="button"
              onClick={session.cancel}
              aria-label="Cancel streaming"
              className="rounded-db-md border border-db-gray-lines bg-white px-2.5 py-1.5 text-[12px] font-medium text-db-navy-800 transition-colors hover:bg-db-oat-medium"
            >
              Cancel
            </button>
          ) : (
            <button
              type="button"
              onClick={() => handleSend()}
              disabled={!inputText.trim()}
              aria-label="Send message"
              className="inline-flex items-center justify-center rounded-db-md bg-db-navy-800 p-1.5 text-white transition-colors hover:bg-db-navy-900 disabled:opacity-55"
            >
              <Send size={12} />
            </button>
          )}
        </div>
        <div className="mt-1.5 text-center text-[10px] text-db-gray-text">
          Mutations are previewed before they touch the workflow.
        </div>
      </div>
    </aside>
  );
}
