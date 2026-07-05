/* eslint-disable react-refresh/only-export-components */
/**
 * Surface component catalog — one React component per catalog entry.
 *
 * Every component receives `SurfaceComponentProps` (comp + ctx).
 * Unknown component names render an inline red error chip (never silent, never crash).
 *
 * Mirrors catalog.py exactly; see src/deep_research/surface/catalog.py for props/docs.
 */

import * as React from 'react';
import type { ComponentType, ReactNode } from 'react';
import * as SelectPrimitive from '@radix-ui/react-select';
import * as TabsPrimitive from '@radix-ui/react-tabs';
import { ChevronDown } from 'lucide-react';
import type { SurfaceComponent, RunReference, SurfaceSourceRef } from '@/types/surface';
import { isPathRef } from '@/types/surface';
import { getAtPointer, resolveDynamic } from '@/lib/surfaceState';
import { MarkdownRenderer, type CitationContext } from '@/components/common';
import { cn } from '@/lib/utils';

// ---------------------------------------------------------------------------
// Context types
// ---------------------------------------------------------------------------

export interface SurfaceRenderContext {
  dataModel: Record<string, unknown>;
  setValue: (pointer: string, value: unknown) => void;
  onAction: (action: string) => void;
  actionDisabled: boolean;
  renderChildren: (ids: string[]) => ReactNode;
  resolveRunReference?: (ref: RunReference | null) => ReactNode;
  /**
   * Per-message citation data for structured-output cells carrying [Key]
   * markers (built by the host from the persisted message's claims).
   */
  resolveCitations?: (messageId: string) => Map<string, CitationContext> | undefined;
  /** Catalog lookup by component id (Tabs reads its TabPane children's labels). */
  getComponent?: (id: string) => SurfaceComponent | undefined;
  /**
   * Re-run structured-output wires for the given message + slots (the retry
   * affordance on a failed slot). Host wires this to the restructure endpoint.
   */
  retryStructuring?: (messageId: string, slots: string[]) => void;
}

export interface SurfaceComponentProps {
  comp: SurfaceComponent;
  ctx: SurfaceRenderContext;
}

// ---------------------------------------------------------------------------
// Shared style tokens (matching Databricks designer system from SchemaField.tsx)
// ---------------------------------------------------------------------------

const FIELD_INPUT_CLASS =
  'w-full rounded-db-md border border-db-gray-lines bg-white px-2.5 py-1.5 font-db-sans text-[13px] leading-[1.4] text-db-navy-800 outline-none transition-colors placeholder:text-db-navy-300 focus:border-db-navy-400 focus:shadow-db-focus';

const FIELD_TEXTAREA_CLASS =
  'w-full resize-y rounded-db-md border border-db-gray-lines bg-white px-3 py-2 font-db-sans text-[13px] leading-[1.55] text-db-navy-800 outline-none transition-colors placeholder:text-db-navy-300 focus:border-db-navy-400 focus:shadow-db-focus';

const LABEL_CLASS = 'mb-1 block font-db-sans text-[12px] font-medium text-db-navy-800';

// ---------------------------------------------------------------------------
// Helper: read a PathRef prop from a component and get the current value
// ---------------------------------------------------------------------------

function readBoundValue(
  comp: SurfaceComponent,
  propKey: string,
  dataModel: Record<string, unknown>,
): unknown {
  const ref = comp.props[propKey];
  if (isPathRef(ref)) {
    return getAtPointer(dataModel, ref.path);
  }
  return undefined;
}

function writeBoundValue(
  comp: SurfaceComponent,
  propKey: string,
  value: unknown,
  ctx: SurfaceRenderContext,
): void {
  const ref = comp.props[propKey];
  if (isPathRef(ref)) {
    ctx.setValue(ref.path, value);
  }
}

// ---------------------------------------------------------------------------
// Gap sizes for containers
// ---------------------------------------------------------------------------

const GAP_CLASS: Record<string, string> = {
  sm: 'gap-1',
  md: 'gap-3',
  lg: 'gap-5',
};

function gapClass(gap: unknown): string {
  return GAP_CLASS[typeof gap === 'string' ? gap : ''] ?? 'gap-3';
}

// ---------------------------------------------------------------------------
// Error chip — never silent, never crash
// ---------------------------------------------------------------------------

function ErrorChip({ message }: { message: string }): React.ReactElement {
  return (
    <span className="inline-flex items-center rounded bg-db-lava-300 px-2 py-0.5 font-db-mono text-[11px] text-db-lava-800">
      {message}
    </span>
  );
}

// ---------------------------------------------------------------------------
// Container components
// ---------------------------------------------------------------------------

function Column({ comp, ctx }: SurfaceComponentProps): React.ReactElement {
  return (
    <div className={cn('flex flex-col', gapClass(comp.props['gap']))}>
      {ctx.renderChildren(comp.children)}
    </div>
  );
}

function Row({ comp, ctx }: SurfaceComponentProps): React.ReactElement {
  return (
    <div className={cn('flex flex-row flex-wrap', gapClass(comp.props['gap']))}>
      {ctx.renderChildren(comp.children)}
    </div>
  );
}

function Card({ comp, ctx }: SurfaceComponentProps): React.ReactElement {
  const title = typeof comp.props['title'] === 'string' ? comp.props['title'] : undefined;
  return (
    <div className="rounded-db-md border border-db-gray-lines bg-white p-4">
      {title && (
        <p className="mb-2 font-db-sans text-[13px] font-semibold text-db-navy-800">{title}</p>
      )}
      <div className="flex flex-col gap-3">{ctx.renderChildren(comp.children)}</div>
    </div>
  );
}

function Divider(): React.ReactElement {
  return <hr className="my-1 border-db-gray-lines" />;
}

// ---------------------------------------------------------------------------
// Static content
// ---------------------------------------------------------------------------

function Heading({ comp, ctx }: SurfaceComponentProps): React.ReactElement {
  const level = typeof comp.props['level'] === 'number' ? comp.props['level'] : 2;
  const rawText = comp.props['text'];
  const text = typeof rawText === 'object' && rawText !== null
    ? String(resolveDynamic(rawText as Parameters<typeof resolveDynamic>[0], ctx.dataModel) ?? '')
    : typeof rawText === 'string'
    ? rawText
    : '';

  const cls = cn('font-db-sans font-semibold text-db-navy-800', {
    'text-[18px]': level === 1,
    'text-[15px]': level === 2,
    'text-[13px]': level === 3,
  });

  if (level === 1) return <h1 className={cls}>{text}</h1>;
  if (level === 3) return <h3 className={cls}>{text}</h3>;
  return <h2 className={cls}>{text}</h2>;
}

function Text({ comp, ctx }: SurfaceComponentProps): React.ReactElement {
  const rawText = comp.props['text'];
  const text = typeof rawText === 'object' && rawText !== null
    ? String(resolveDynamic(rawText as Parameters<typeof resolveDynamic>[0], ctx.dataModel) ?? '')
    : typeof rawText === 'string'
    ? rawText
    : '';
  return <p className="font-db-sans text-[13px] text-db-navy-800">{text}</p>;
}

function Markdown({ comp, ctx }: SurfaceComponentProps): React.ReactElement {
  const rawContent = comp.props['content'];
  const content = typeof rawContent === 'object' && rawContent !== null
    ? String(resolveDynamic(rawContent as Parameters<typeof resolveDynamic>[0], ctx.dataModel) ?? '')
    : typeof rawContent === 'string'
    ? rawContent
    : '';
  return <MarkdownRenderer content={content} enableCitations={false} />;
}

// ---------------------------------------------------------------------------
// Input components (two-way bound)
// ---------------------------------------------------------------------------

function TextField({ comp, ctx }: SurfaceComponentProps): React.ReactElement {
  const label = typeof comp.props['label'] === 'string' ? comp.props['label'] : undefined;
  const placeholder =
    typeof comp.props['placeholder'] === 'string' ? comp.props['placeholder'] : undefined;
  const currentValue = readBoundValue(comp, 'value', ctx.dataModel);
  const strValue = typeof currentValue === 'string' ? currentValue : '';
  const fieldId = `surface-field-${comp.id}`;

  return (
    <div className="mb-2">
      {label && (
        <label htmlFor={fieldId} className={LABEL_CLASS}>
          {label}
        </label>
      )}
      <input
        id={fieldId}
        type="text"
        value={strValue}
        placeholder={placeholder}
        onChange={(e) => writeBoundValue(comp, 'value', e.target.value, ctx)}
        className={FIELD_INPUT_CLASS}
      />
    </div>
  );
}

function TextArea({ comp, ctx }: SurfaceComponentProps): React.ReactElement {
  const label = typeof comp.props['label'] === 'string' ? comp.props['label'] : undefined;
  const placeholder =
    typeof comp.props['placeholder'] === 'string' ? comp.props['placeholder'] : undefined;
  const rows = typeof comp.props['rows'] === 'number' ? comp.props['rows'] : 4;
  const currentValue = readBoundValue(comp, 'value', ctx.dataModel);
  const strValue = typeof currentValue === 'string' ? currentValue : '';
  const fieldId = `surface-field-${comp.id}`;

  return (
    <div className="mb-2">
      {label && (
        <label htmlFor={fieldId} className={LABEL_CLASS}>
          {label}
        </label>
      )}
      <textarea
        id={fieldId}
        value={strValue}
        placeholder={placeholder}
        rows={rows}
        onChange={(e) => writeBoundValue(comp, 'value', e.target.value, ctx)}
        className={FIELD_TEXTAREA_CLASS}
      />
    </div>
  );
}

function Select({ comp, ctx }: SurfaceComponentProps): React.ReactElement {
  const label = typeof comp.props['label'] === 'string' ? comp.props['label'] : undefined;
  const rawOptions = comp.props['options'];
  const options: Array<{ label: string; value: string }> = Array.isArray(rawOptions)
    ? rawOptions.filter(
        (o): o is { label: string; value: string } =>
          typeof o === 'object' &&
          o !== null &&
          typeof (o as { label?: unknown }).label === 'string' &&
          typeof (o as { value?: unknown }).value === 'string',
      )
    : [];

  const currentValue = readBoundValue(comp, 'value', ctx.dataModel);
  const strValue = typeof currentValue === 'string' ? currentValue : '';
  const fieldId = `surface-select-${comp.id}`;

  // Find current label (fall back to raw value for custom entries)
  const currentLabel = options.find((o) => o.value === strValue)?.label ?? strValue;

  return (
    <div className="mb-2">
      {label && (
        <label htmlFor={fieldId} className={LABEL_CLASS}>
          {label}
        </label>
      )}
      <SelectPrimitive.Root
        value={strValue}
        onValueChange={(v) => writeBoundValue(comp, 'value', v, ctx)}
      >
        <SelectPrimitive.Trigger
          id={fieldId}
          className={cn('flex items-center justify-between', FIELD_INPUT_CLASS)}
          aria-label={label}
        >
          <SelectPrimitive.Value placeholder="Select…">
            {currentLabel || 'Select…'}
          </SelectPrimitive.Value>
          <SelectPrimitive.Icon className="ml-2 text-db-gray-text">
            <ChevronDown size={14} />
          </SelectPrimitive.Icon>
        </SelectPrimitive.Trigger>
        <SelectPrimitive.Portal>
          <SelectPrimitive.Content className="z-50 min-w-[8rem] overflow-hidden rounded-db-md border border-db-gray-lines bg-white font-db-sans shadow-db-md">
            <SelectPrimitive.Viewport className="p-1">
              {options.map((opt) => (
                <SelectPrimitive.Item
                  key={opt.value}
                  value={opt.value}
                  className="relative flex cursor-pointer select-none items-center rounded px-2 py-1.5 text-[13px] text-db-navy-800 outline-none transition-colors hover:bg-db-oat-medium focus:bg-db-oat-medium"
                >
                  <SelectPrimitive.ItemText>{opt.label}</SelectPrimitive.ItemText>
                </SelectPrimitive.Item>
              ))}
            </SelectPrimitive.Viewport>
          </SelectPrimitive.Content>
        </SelectPrimitive.Portal>
      </SelectPrimitive.Root>
    </div>
  );
}

function Checkbox({ comp, ctx }: SurfaceComponentProps): React.ReactElement {
  const label = typeof comp.props['label'] === 'string' ? comp.props['label'] : undefined;
  const currentValue = readBoundValue(comp, 'value', ctx.dataModel);
  const boolValue = typeof currentValue === 'boolean' ? currentValue : Boolean(currentValue);
  const fieldId = `surface-checkbox-${comp.id}`;

  return (
    <div className="mb-2">
      <label
        htmlFor={fieldId}
        className="flex cursor-pointer items-center gap-2 font-db-sans text-[12px] font-medium text-db-navy-800"
      >
        <input
          id={fieldId}
          type="checkbox"
          checked={boolValue}
          onChange={(e) => writeBoundValue(comp, 'value', e.target.checked, ctx)}
          className="h-4 w-4 rounded-sm border-db-gray-lines text-db-lava-600 outline-none focus:shadow-db-focus"
        />
        {label}
      </label>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Action
// ---------------------------------------------------------------------------

function Button({ comp, ctx }: SurfaceComponentProps): React.ReactElement {
  const label = typeof comp.props['label'] === 'string' ? comp.props['label'] : 'Run';
  const action = typeof comp.props['action'] === 'string' ? comp.props['action'] : '';
  const variant = comp.props['variant'] === 'secondary' ? 'secondary' : 'primary';

  const primaryClass =
    'inline-flex items-center justify-center rounded-db-md bg-db-lava-600 px-4 py-2 font-db-sans text-[13px] font-medium text-white transition-colors hover:bg-db-lava-700 disabled:opacity-50 focus:outline-none focus:shadow-db-focus';
  const secondaryClass =
    'inline-flex items-center justify-center rounded-db-md border border-db-gray-lines bg-white px-4 py-2 font-db-sans text-[13px] font-medium text-db-navy-800 transition-colors hover:bg-db-oat-medium disabled:opacity-50 focus:outline-none focus:shadow-db-focus';

  return (
    <button
      type="button"
      data-testid={action ? `surface-action-${action}` : undefined}
      disabled={ctx.actionDisabled}
      onClick={() => action && ctx.onAction(action)}
      className={variant === 'primary' ? primaryClass : secondaryClass}
    >
      {label}
    </button>
  );
}

// ---------------------------------------------------------------------------
// Results
// ---------------------------------------------------------------------------

function ReportRegion({ comp, ctx }: SurfaceComponentProps): React.ReactElement {
  const rawSource = comp.props['source'];
  const emptyText =
    typeof comp.props['empty_text'] === 'string' ? comp.props['empty_text'] : 'No results yet.';

  const resolvedSource = isPathRef(rawSource)
    ? (getAtPointer(ctx.dataModel, rawSource.path) as RunReference | null | undefined)
    : null;

  const ref: RunReference | null =
    resolvedSource != null && typeof resolvedSource === 'object'
      ? (resolvedSource as RunReference)
      : null;

  if (ctx.resolveRunReference) {
    const resolved = ctx.resolveRunReference(ref);
    // A null/undefined resolution falls through to the static branches below
    // (notably empty_text for a null ref) instead of blanking the region.
    if (resolved !== null && resolved !== undefined) {
      return <div>{resolved}</div>;
    }
  }

  if (ref === null) {
    return (
      <p className="font-db-sans text-[13px] italic text-db-gray-text">{emptyText}</p>
    );
  }

  return (
    <p className="font-db-sans text-[13px] text-db-navy-500">
      Run status: {ref.status}
    </p>
  );
}

const STATUS_BADGE_CLASS: Record<string, string> = {
  running:
    'inline-flex items-center rounded-full bg-db-yellow-100 px-2.5 py-0.5 font-db-sans text-[12px] font-medium text-db-yellow-800',
  completed:
    'inline-flex items-center rounded-full bg-db-green-100 px-2.5 py-0.5 font-db-sans text-[12px] font-medium text-db-green-800',
  failed:
    'inline-flex items-center rounded-full bg-db-lava-100 px-2.5 py-0.5 font-db-sans text-[12px] font-medium text-db-lava-800',
  cancelled:
    'inline-flex items-center rounded-full bg-db-gray-200 px-2.5 py-0.5 font-db-sans text-[12px] font-medium text-db-gray-600',
  idle: 'inline-flex items-center rounded-full bg-db-oat-light px-2.5 py-0.5 font-db-sans text-[12px] font-medium text-db-navy-500',
};

function StatusBadge({ comp, ctx }: SurfaceComponentProps): React.ReactElement {
  const rawSource = comp.props['source'];
  const idleLabel =
    typeof comp.props['label'] === 'string' ? comp.props['label'] : 'Idle';

  const resolvedSource = isPathRef(rawSource)
    ? (getAtPointer(ctx.dataModel, rawSource.path) as RunReference | null | undefined)
    : null;

  const ref: RunReference | null =
    resolvedSource != null && typeof resolvedSource === 'object'
      ? (resolvedSource as RunReference)
      : null;

  const status = ref?.status ?? 'idle';
  const badgeClass = STATUS_BADGE_CLASS[status] ?? STATUS_BADGE_CLASS['idle'];
  const label = ref !== null ? status.charAt(0).toUpperCase() + status.slice(1) : idleLabel;

  return <span className={badgeClass}>{label}</span>;
}

// ---------------------------------------------------------------------------
// Structured output (model-filled slots)
//
// Output components bind a SLOT pointer `<binding output target>/data/<slot>`.
// The host overlays the enriched RunReference (incl. `data`) at the binding
// target, so the slot value resolves with plain object traversal; the run
// reference itself sits at the pointer prefix before "/data/".
// ---------------------------------------------------------------------------

// Alpha keys ([Arxiv], [Key-2]) plus legacy numeric keys ([1]). Over-matching
// only costs mounting MarkdownRenderer for a cell — never correctness.
const CELL_MARKER_RE = /\[[A-Za-z0-9][A-Za-z0-9-]*\]/;

interface SlotBinding {
  runRef: RunReference | null;
  value: unknown;
  /** Slot name (segment after `/data/`), used for per-slot status + retry. */
  slotName: string | null;
}

function readSlotBinding(
  comp: SurfaceComponent,
  pointerProp: string,
  ctx: SurfaceRenderContext,
): SlotBinding {
  const raw = comp.props[pointerProp];
  if (!isPathRef(raw)) return { runRef: null, value: undefined, slotName: null };
  const pointer = raw.path;
  const value = getAtPointer(ctx.dataModel, pointer);
  const [refPointer, slotTail] = pointer.split('/data/');
  const slotName = slotTail && !slotTail.includes('/') ? slotTail : null;
  let runRef: RunReference | null = null;
  if (refPointer && refPointer !== pointer) {
    const rawRef = getAtPointer(ctx.dataModel, refPointer);
    if (rawRef !== null && typeof rawRef === 'object') {
      runRef = rawRef as RunReference;
    }
  }
  return { runRef, value, slotName };
}

// ---------------------------------------------------------------------------
// Source chips — resolve an item's `source_refs` (index strings) against the
// run's evidence legend (runRef.sources) to numbered, clickable links.
// ---------------------------------------------------------------------------

/** Read an item's `source_refs` (verbatim snake_case in the slot payload). */
function itemSourceRefs(item: unknown): string[] {
  if (typeof item !== 'object' || item === null) return [];
  const raw = (item as { source_refs?: unknown }).source_refs;
  if (!Array.isArray(raw)) return [];
  return raw.filter((r): r is string => typeof r === 'string');
}

function SourceChips({
  refs,
  runRef,
}: {
  refs: string[];
  runRef: RunReference | null;
}): React.ReactElement | null {
  const legend = runRef?.sources;
  if (!legend || legend.length === 0 || refs.length === 0) return null;
  const byRef = new Map<string, SurfaceSourceRef>(legend.map((s) => [s.ref, s]));
  const resolved = refs
    .map((ref) => byRef.get(ref))
    .filter((s): s is SurfaceSourceRef => s !== undefined);
  if (resolved.length === 0) return null;

  return (
    <span className="ml-1 inline-flex flex-wrap gap-1 align-middle">
      {resolved.map((src) => (
        <a
          key={src.ref}
          href={src.url}
          target="_blank"
          rel="noreferrer"
          title={src.title ?? src.url}
          className="inline-flex items-center rounded border border-db-navy-200 bg-db-oat-light px-1.5 py-0.5 font-db-mono text-[10px] leading-none text-db-navy-600 transition-colors hover:bg-db-oat-medium hover:text-db-navy-800"
          data-testid={`surface-source-chip-${src.ref}`}
        >
          {src.ref}
        </a>
      ))}
    </span>
  );
}

/** Normalize a string-slot item to {text, sourceRefs} — accepts the legacy v1
 * bare-string shape AND the v2 `{text, source_refs}` object shape. */
function normalizeStringItem(
  item: unknown,
): { text: string; sourceRefs: string[] } | null {
  if (typeof item === 'string') return { text: item, sourceRefs: [] };
  if (typeof item === 'object' && item !== null) {
    const text = (item as { text?: unknown }).text;
    if (typeof text === 'string') {
      return { text, sourceRefs: itemSourceRefs(item) };
    }
  }
  return null;
}

/** Cell text: plain span unless it carries [Key] markers — then the real
 * citation path (MarkdownRenderer + citationData) renders interactive chips. */
function CellText({
  text,
  runRef,
  ctx,
}: {
  text: string;
  runRef: RunReference | null;
  ctx: SurfaceRenderContext;
}): React.ReactElement {
  if (!CELL_MARKER_RE.test(text)) return <>{text}</>;
  const citationData = runRef?.message_id
    ? ctx.resolveCitations?.(runRef.message_id)
    : undefined;
  return (
    <MarkdownRenderer
      content={text}
      enableCitations
      citationMode="numeric"
      citationData={citationData}
      className="prose prose-sm max-w-none text-[13px] [&_p]:m-0 [&_p]:inline"
    />
  );
}

/** Waiting skeleton shown while a slot's wire is running (run- or slot-level). */
function SlotSkeleton(): React.ReactElement {
  return (
    <div
      aria-label="Waiting for results"
      className="h-6 w-2/3 animate-pulse rounded bg-db-gray-200"
    />
  );
}

function SlotEmpty({
  runRef,
  slotName,
  emptyText,
  ctx,
}: {
  runRef: RunReference | null;
  slotName: string | null;
  emptyText: string;
  ctx: SurfaceRenderContext;
}): React.ReactElement {
  // The run itself is still streaming — nothing persisted yet.
  if (runRef?.status === 'running') return <SlotSkeleton />;
  if (runRef?.pendingStructuredOutput) return <SlotSkeleton />;

  // Per-slot state machine (stub-first: pending → ok/empty/failed), attached
  // by the host from the persisted envelope's meta.slots.
  const slotStatus = slotName ? runRef?.slotsMeta?.[slotName]?.status : undefined;
  if (slotStatus === 'pending') return <SlotSkeleton />;
  if (slotStatus === 'failed') {
    const canRetry = Boolean(ctx.retryStructuring && runRef?.message_id && slotName);
    return (
      <div className="flex items-center gap-2">
        <p className="font-db-sans text-[13px] text-db-lava-700">
          Couldn’t structure this section.
        </p>
        {canRetry && (
          <button
            type="button"
            data-testid={`surface-slot-retry-${slotName}`}
            onClick={() =>
              ctx.retryStructuring?.(runRef!.message_id!, [slotName!])
            }
            className="inline-flex items-center rounded-db-md border border-db-gray-lines bg-white px-2 py-0.5 font-db-sans text-[12px] font-medium text-db-navy-800 transition-colors hover:bg-db-oat-medium focus:outline-none focus:shadow-db-focus"
          >
            Retry
          </button>
        )}
      </div>
    );
  }

  // Completed with no data for this slot (ok/empty), vs. never run.
  if (runRef?.status === 'completed' || slotStatus === 'ok' || slotStatus === 'empty') {
    return (
      <p className="font-db-sans text-[13px] italic text-db-gray-text">
        No structured data for this section.
      </p>
    );
  }
  return (
    <p className="font-db-sans text-[13px] italic text-db-gray-text">{emptyText}</p>
  );
}

function emptyTextOf(comp: SurfaceComponent): string {
  return typeof comp.props['empty_text'] === 'string'
    ? comp.props['empty_text']
    : 'No results yet.';
}

interface TableColumn {
  key: string;
  label: string;
  type: string;
}

function tableColumns(comp: SurfaceComponent): TableColumn[] {
  const raw = comp.props['columns'];
  if (!Array.isArray(raw)) return [];
  return raw.filter(
    (c): c is TableColumn =>
      typeof c === 'object' &&
      c !== null &&
      typeof (c as { key?: unknown }).key === 'string' &&
      typeof (c as { label?: unknown }).label === 'string' &&
      typeof (c as { type?: unknown }).type === 'string',
  );
}

function Table({ comp, ctx }: SurfaceComponentProps): React.ReactElement {
  const columns = tableColumns(comp);
  const { runRef, value, slotName } = readSlotBinding(comp, 'source', ctx);
  const rows = Array.isArray(value)
    ? (value.filter((r) => typeof r === 'object' && r !== null) as Record<
        string,
        unknown
      >[])
    : null;

  if (!rows || rows.length === 0 || columns.length === 0) {
    return (
      <SlotEmpty
        runRef={runRef}
        slotName={slotName}
        emptyText={emptyTextOf(comp)}
        ctx={ctx}
      />
    );
  }

  // Only add the trailing Sources column when we can actually resolve chips.
  const hasSourceChips =
    (runRef?.sources?.length ?? 0) > 0 &&
    rows.some((row) => itemSourceRefs(row).length > 0);

  return (
    <div className="overflow-x-auto" data-testid={`surface-table-${comp.id}`}>
      <table className="w-full border-collapse font-db-sans text-[13px]">
        <thead>
          <tr className="border-b border-db-gray-lines">
            {columns.map((col) => (
              <th
                key={col.key}
                className={cn(
                  'px-2 py-1.5 font-medium text-db-navy-800',
                  col.type === 'number' ? 'text-right' : 'text-left',
                )}
              >
                {col.label}
              </th>
            ))}
            {hasSourceChips && (
              <th className="px-2 py-1.5 text-left font-medium text-db-gray-text">
                Sources
              </th>
            )}
          </tr>
        </thead>
        <tbody>
          {rows.map((row, index) => (
            <tr key={index} className="border-b border-db-gray-lines/60 align-top">
              {columns.map((col) => {
                const cell = row[col.key];
                return (
                  <td
                    key={col.key}
                    className={cn(
                      'px-2 py-1.5 text-db-navy-800',
                      col.type === 'number' && 'text-right font-db-mono text-[12px]',
                    )}
                  >
                    {col.type === 'number' ? (
                      String(cell ?? '')
                    ) : (
                      <CellText text={String(cell ?? '')} runRef={runRef} ctx={ctx} />
                    )}
                  </td>
                );
              })}
              {hasSourceChips && (
                <td className="px-2 py-1.5">
                  <SourceChips refs={itemSourceRefs(row)} runRef={runRef} />
                </td>
              )}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

interface MetricItem {
  label: string;
  value: string;
  unit?: string | null;
  delta?: string | null;
  source_refs?: string[];
}

function MetricGrid({ comp, ctx }: SurfaceComponentProps): React.ReactElement {
  const { runRef, value, slotName } = readSlotBinding(comp, 'source', ctx);
  const metrics = Array.isArray(value)
    ? (value.filter(
        (m): m is MetricItem =>
          typeof m === 'object' &&
          m !== null &&
          typeof (m as { label?: unknown }).label === 'string' &&
          typeof (m as { value?: unknown }).value === 'string',
      ) as MetricItem[])
    : null;

  if (!metrics || metrics.length === 0) {
    return (
      <SlotEmpty
        runRef={runRef}
        slotName={slotName}
        emptyText={emptyTextOf(comp)}
        ctx={ctx}
      />
    );
  }

  return (
    <div
      className="grid grid-cols-2 gap-2 sm:grid-cols-3"
      data-testid={`surface-metrics-${comp.id}`}
    >
      {metrics.map((metric, index) => (
        <div
          key={index}
          className="rounded-db-md border border-db-gray-lines bg-white p-3"
        >
          <p className="font-db-sans text-[11px] font-medium uppercase tracking-wide text-db-gray-text">
            {metric.label}
          </p>
          <p className="font-db-sans text-[18px] font-semibold text-db-navy-800">
            {metric.value}
            {metric.unit ? (
              <span className="ml-0.5 text-[12px] font-normal text-db-gray-text">
                {metric.unit}
              </span>
            ) : null}
          </p>
          {metric.delta ? (
            <p className="font-db-sans text-[11px] text-db-navy-500">{metric.delta}</p>
          ) : null}
          <SourceChips refs={itemSourceRefs(metric)} runRef={runRef} />
        </div>
      ))}
    </div>
  );
}

function KeyFindings({ comp, ctx }: SurfaceComponentProps): React.ReactElement {
  const maxItems =
    typeof comp.props['max_items'] === 'number' ? comp.props['max_items'] : undefined;
  const { runRef, value, slotName } = readSlotBinding(comp, 'source', ctx);
  let items = Array.isArray(value)
    ? value
        .map(normalizeStringItem)
        .filter((i): i is { text: string; sourceRefs: string[] } => i !== null)
    : null;
  if (items && maxItems !== undefined) items = items.slice(0, Math.max(0, maxItems));

  if (!items || items.length === 0) {
    return (
      <SlotEmpty
        runRef={runRef}
        slotName={slotName}
        emptyText={emptyTextOf(comp)}
        ctx={ctx}
      />
    );
  }

  return (
    <ul
      className="list-disc space-y-1 pl-5 font-db-sans text-[13px] text-db-navy-800"
      data-testid={`surface-findings-${comp.id}`}
    >
      {items.map((item, index) => (
        <li key={index}>
          <CellText text={item.text} runRef={runRef} ctx={ctx} />
          <SourceChips refs={item.sourceRefs} runRef={runRef} />
        </li>
      ))}
    </ul>
  );
}

function List({ comp, ctx }: SurfaceComponentProps): React.ReactElement {
  const ordered = comp.props['ordered'] === true;
  const { runRef, value, slotName } = readSlotBinding(comp, 'items', ctx);
  const items = Array.isArray(value)
    ? value
        .map(normalizeStringItem)
        .filter((i): i is { text: string; sourceRefs: string[] } => i !== null)
    : null;

  if (!items || items.length === 0) {
    return (
      <SlotEmpty
        runRef={runRef}
        slotName={slotName}
        emptyText={emptyTextOf(comp)}
        ctx={ctx}
      />
    );
  }
  const itemNodes = items.map((item, index) => (
    <li key={index}>
      <CellText text={item.text} runRef={runRef} ctx={ctx} />
      <SourceChips refs={item.sourceRefs} runRef={runRef} />
    </li>
  ));
  const listClass =
    'space-y-1 pl-5 font-db-sans text-[13px] text-db-navy-800';
  return ordered ? (
    <ol className={cn('list-decimal', listClass)}>{itemNodes}</ol>
  ) : (
    <ul className={cn('list-disc', listClass)}>{itemNodes}</ul>
  );
}

const LazySurfaceChart = React.lazy(() => import('./SurfaceChart'));

function Chart({ comp, ctx }: SurfaceComponentProps): React.ReactElement {
  const kind = comp.props['kind'] === 'line' ? 'line' : 'bar';
  const xKey = typeof comp.props['x_key'] === 'string' ? comp.props['x_key'] : '';
  const rawYKeys = comp.props['y_keys'];
  const yKeys = Array.isArray(rawYKeys)
    ? rawYKeys.filter((k): k is string => typeof k === 'string')
    : [];
  const height =
    typeof comp.props['height'] === 'number' ? comp.props['height'] : undefined;
  const { runRef, value, slotName } = readSlotBinding(comp, 'source', ctx);
  const rows = Array.isArray(value)
    ? (value.filter((r) => typeof r === 'object' && r !== null) as Record<
        string,
        unknown
      >[])
    : null;

  if (!rows || rows.length === 0 || !xKey || yKeys.length === 0) {
    return (
      <SlotEmpty
        runRef={runRef}
        slotName={slotName}
        emptyText={emptyTextOf(comp)}
        ctx={ctx}
      />
    );
  }

  return (
    <React.Suspense
      fallback={
        <div
          className="animate-pulse rounded bg-db-gray-200"
          style={{ height: height ?? 240 }}
          aria-label="Loading chart"
        />
      }
    >
      <LazySurfaceChart rows={rows} kind={kind} xKey={xKey} yKeys={yKeys} height={height} />
    </React.Suspense>
  );
}

// ---------------------------------------------------------------------------
// Tabs
// ---------------------------------------------------------------------------

function Tabs({ comp, ctx }: SurfaceComponentProps): React.ReactElement {
  const panes = comp.children
    .map((id) => ctx.getComponent?.(id))
    .filter(
      (child): child is SurfaceComponent =>
        child !== undefined && child.component === 'TabPane',
    );
  const [active, setActive] = React.useState<string>(panes[0]?.id ?? '');

  if (panes.length === 0) {
    return <ErrorChip message={`Tabs "${comp.id}" has no TabPane children`} />;
  }

  return (
    <TabsPrimitive.Root
      value={panes.some((p) => p.id === active) ? active : panes[0]!.id}
      onValueChange={setActive}
    >
      <TabsPrimitive.List className="mb-3 flex gap-1 border-b border-db-gray-lines">
        {panes.map((pane) => {
          const label =
            typeof pane.props['label'] === 'string' ? pane.props['label'] : pane.id;
          return (
            <TabsPrimitive.Trigger
              key={pane.id}
              value={pane.id}
              className="-mb-px border-b-2 border-transparent px-3 py-1.5 font-db-sans text-[13px] font-medium text-db-gray-text transition-colors hover:text-db-navy-800 data-[state=active]:border-db-lava-600 data-[state=active]:text-db-navy-800"
            >
              {label}
            </TabsPrimitive.Trigger>
          );
        })}
      </TabsPrimitive.List>
      {panes.map((pane) => (
        <TabsPrimitive.Content key={pane.id} value={pane.id}>
          {ctx.renderChildren([pane.id])}
        </TabsPrimitive.Content>
      ))}
    </TabsPrimitive.Root>
  );
}

function TabPane({ comp, ctx }: SurfaceComponentProps): React.ReactElement {
  return <div className="flex flex-col gap-3">{ctx.renderChildren(comp.children)}</div>;
}

// ---------------------------------------------------------------------------
// Unknown component fallback
// ---------------------------------------------------------------------------

function UnknownComponent({ comp }: Pick<SurfaceComponentProps, 'comp'>): React.ReactElement {
  return (
    <ErrorChip message={`Unknown component: "${comp.component}" (id: ${comp.id})`} />
  );
}

// ---------------------------------------------------------------------------
// SURFACE_CATALOG
// ---------------------------------------------------------------------------

export const SURFACE_CATALOG: Record<string, ComponentType<SurfaceComponentProps>> = {
  Column,
  Row,
  Card,
  Divider: ({ comp: _comp, ctx: _ctx }) => <Divider />,
  Heading,
  Text,
  Markdown,
  TextField,
  TextArea,
  Select,
  Checkbox,
  Button,
  ReportRegion,
  StatusBadge,
  Table,
  MetricGrid,
  KeyFindings,
  Chart,
  List,
  Tabs,
  TabPane,
};

/** Render a single SurfaceComponent, falling back to an error chip for unknowns. */
export function renderComponent(
  comp: SurfaceComponent,
  ctx: SurfaceRenderContext,
): React.ReactElement {
  const Comp = SURFACE_CATALOG[comp.component];
  if (!Comp) {
    return <UnknownComponent key={comp.id} comp={comp} />;
  }
  return <Comp key={comp.id} comp={comp} ctx={ctx} />;
}
