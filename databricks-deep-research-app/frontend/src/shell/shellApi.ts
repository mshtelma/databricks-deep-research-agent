/**
 * Shell-app API surface. The standalone deployed app exposes a much smaller
 * API than the main app (no chats/agents-v2/persistence): a bootstrap config
 * endpoint + a single SSE run endpoint with reconnect/resume. These helpers +
 * the `useShellRun` hook adapt that to the SAME surface components the main
 * app renders, so the deployed dashboard is pixel-identical with zero
 * renderer duplication.
 */

import type { StructuredOutputEnvelope } from '@/types';
import type { RunReference, Surface } from '@/types/surface';
import { normalizeSurface } from '@/lib/surfaceSchema';

export interface ShellConfig {
  agent_name: string;
  /** null for a plain research agent (no form/dashboard, just query + report). */
  surface: Surface | null;
}

export const SHELL_TEMPLATE_VERSION = '2026-07-03.1';

export async function fetchShellConfig(): Promise<ShellConfig> {
  const resp = await fetch('/api/config', {
    headers: { Accept: 'application/json' },
    cache: 'no-store',
  });
  if (!resp.ok) throw new Error(`config request failed: HTTP ${resp.status}`);
  const cfg = (await resp.json()) as ShellConfig;
  // Normalize the raw surface so the deployed dashboard gets the same `Surface`
  // invariants as the main app (a component missing `children` would otherwise crash
  // AgentSurfacePanel on load). Shares the single normalizer with the main-chat path.
  return { ...cfg, surface: normalizeSurface(cfg.surface) };
}

/**
 * Overlay a structured-output envelope onto a RunReference — the same
 * `data` / `sources` / `slotsMeta` fields `lib/surfaceEnrichment` attaches in
 * the main app, so the catalog renders slots identically. The envelope is
 * verbatim snake_case (never camelized), matching `StructuredOutputEnvelope`.
 */
export function applyEnvelopeToRef(
  ref: RunReference,
  env: StructuredOutputEnvelope,
): RunReference {
  return {
    ...ref,
    data: env.data,
    sources: env.meta?.sources,
    slotsMeta: env.meta?.slots,
  };
}

// ---------------------------------------------------------------------------
// SSE frame parsing (the shell backend emits `event:`/`data:`/`id:` frames)
// ---------------------------------------------------------------------------

export interface SseFrame {
  event: string;
  data: string;
  id: number | null;
}

/** Parse one raw SSE frame block into {event, data, id}. */
export function parseSseFrame(frame: string): SseFrame | null {
  const lines = frame.split('\n');
  const eventLine = lines.find((l) => l.startsWith('event:'));
  const idLine = lines.find((l) => l.startsWith('id:'));
  const dataLines = lines
    .filter((l) => l.startsWith('data:'))
    .map((l) => l.slice(5).replace(/^ /, ''));
  if (dataLines.length === 0) return null; // keepalive / comment
  let id: number | null = null;
  if (idLine) {
    const n = parseInt(idLine.slice(3).trim(), 10);
    if (!Number.isNaN(n)) id = n;
  }
  return {
    event: eventLine ? eventLine.slice(6).trim() : 'data',
    data: dataLines.join('\n'),
    id,
  };
}

export function safeJsonParse(text: string): Record<string, unknown> {
  try {
    return JSON.parse(text) as Record<string, unknown>;
  } catch {
    return { raw: text };
  }
}
