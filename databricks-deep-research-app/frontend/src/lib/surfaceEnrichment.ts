/**
 * Structured-output enrichment: attach a completed run's persisted structured
 * output (data + evidence legend + per-slot status) onto its RunReference, so
 * surface output components can render `<target>/data/<slot>` by reference
 * without the payload ever living in surface_state.
 *
 * The envelope is a verbatim snake_case pass-through (see
 * `StructuredOutputEnvelope`), so this reads snake_case keys.
 */

import type { FullMessage } from '@/types';
import type { RunReference, SurfaceSlotMeta } from '@/types/surface';

/** A slot pending past this age is shown as failed (offer Retry, not a stuck
 * skeleton) — the process likely died between the stub write and the wires. */
const STALE_PENDING_MS = 10 * 60 * 1000;

/** Downgrade `pending` slots to `failed` once the envelope is older than the
 * stale threshold. Returns the same reference when nothing changes. */
function withStaleDowngrade(
  slotsMeta: Record<string, SurfaceSlotMeta> | undefined,
  generatedAt: string | undefined,
  now: number,
): Record<string, SurfaceSlotMeta> | undefined {
  if (!slotsMeta || typeof generatedAt !== 'string') return slotsMeta;
  if (now - Date.parse(generatedAt) <= STALE_PENDING_MS) return slotsMeta;
  let changed = false;
  const next: Record<string, SurfaceSlotMeta> = {};
  for (const [slot, meta] of Object.entries(slotsMeta)) {
    if (meta.status === 'pending') {
      next[slot] = { ...meta, status: 'failed', error: meta.error ?? 'timed out' };
      changed = true;
    } else {
      next[slot] = meta;
    }
  }
  return changed ? next : slotsMeta;
}

/**
 * True when `msg` is at least as recent as `prev` for latest-per-binding
 * selection. Forward-scan friendly: on an empty/unparseable `createdAt` (NaN)
 * we prefer the later array element, so the caller iterates oldest→newest
 * without sorting. `chatFull` returns messages oldest→newest.
 */
function atLeastAsRecent(msg: FullMessage, prev: FullMessage): boolean {
  const tMsg = Date.parse(msg.createdAt);
  const tPrev = Date.parse(prev.createdAt);
  if (Number.isNaN(tMsg) || Number.isNaN(tPrev)) return true;
  return tMsg >= tPrev;
}

/**
 * Project the run-state a surface renders from the persisted messages.
 *
 * Two passes:
 *
 * 1. **Backfill.** For each binding that has structured output on a message but
 *    no live/persisted run-state entry (e.g. a regular chat run under a surface
 *    agent, which never wrote `surface_state`), synthesize a completed ref
 *    `{ status:'completed', message_id }` pointing at the latest such message.
 *    A live/persisted entry is authoritative and never clobbered; a persisted
 *    completed entry that lacks a `message_id` has one attached.
 * 2. **Enrich.** A completed ref whose message carries an envelope for the SAME
 *    binding gains `data`, the `sources` legend, and `slotsMeta` (with
 *    stale-pending downgraded to failed).
 *
 * The original object identity is preserved when nothing is backfilled or
 * enriched.
 */
export function enrichSurfaceRunState(
  surfaceRunState: Record<string, RunReference | null>,
  messages: readonly FullMessage[] | undefined,
  now: number = Date.now(),
  pendingMessageIds?: ReadonlySet<string>,
): Record<string, RunReference | null> {
  if (!messages || messages.length === 0) {
    if (!pendingMessageIds || pendingMessageIds.size === 0) return surfaceRunState;
    let changed = false;
    const next: Record<string, RunReference | null> = {};
    for (const [action, ref] of Object.entries(surfaceRunState)) {
      if (
        ref?.status === 'completed' &&
        ref.message_id &&
        pendingMessageIds.has(ref.message_id)
      ) {
        next[action] = { ...ref, pendingStructuredOutput: true };
        changed = true;
      } else {
        next[action] = ref;
      }
    }
    return changed ? next : surfaceRunState;
  }

  // --- Pass 1: backfill -----------------------------------------------------
  // Latest message per binding (forward scan, no sort; NaN createdAt → later wins).
  const latestByBinding = new Map<string, FullMessage>();
  for (const msg of messages) {
    const binding = msg.structuredOutput?.binding;
    if (!binding) continue;
    const prev = latestByBinding.get(binding);
    if (prev === undefined || atLeastAsRecent(msg, prev)) {
      latestByBinding.set(binding, msg);
    }
  }

  // Lazily clone so identity is preserved when nothing is backfilled.
  let base: Record<string, RunReference | null> = surfaceRunState;
  let cloned = false;
  const cloneOnce = (): void => {
    if (!cloned) {
      base = { ...surfaceRunState };
      cloned = true;
    }
  };
  for (const [binding, msg] of latestByBinding) {
    const existing = base[binding];
    if (existing === undefined || existing === null) {
      cloneOnce();
      base[binding] = { status: 'completed', message_id: msg.id };
    } else if (existing.status === 'completed' && existing.message_id === undefined) {
      cloneOnce();
      base[binding] = { ...existing, message_id: msg.id };
    }
    // else: a live/persisted entry (running/failed, or completed with a
    // message_id) is authoritative — leave it untouched.
  }

  // --- Pass 2: enrich (over the possibly-backfilled map) --------------------
  let enrichChanged = false;
  const enriched: Record<string, RunReference | null> = {};
  for (const [action, ref] of Object.entries(base)) {
    if (ref?.status === 'completed' && ref.message_id) {
      const msg = messages.find((m) => m.id === ref.message_id);
      const structured = msg?.structuredOutput;
      if (structured && structured.binding === action) {
        enriched[action] = {
          ...ref,
          data: structured.data,
          sources: structured.meta?.sources,
          slotsMeta: withStaleDowngrade(
            structured.meta?.slots,
            structured.generated_at,
            now,
          ),
        };
        enrichChanged = true;
        continue;
      }
      if (pendingMessageIds?.has(ref.message_id)) {
        enriched[action] = { ...ref, pendingStructuredOutput: true };
        enrichChanged = true;
        continue;
      }
    }
    enriched[action] = ref;
  }

  return cloned || enrichChanged ? enriched : surfaceRunState;
}
