/**
 * Live-fill trigger selection: which message ids have a structured-output
 * envelope that is still being generated and should be polled for.
 *
 * The structured-output wires persist the envelope ~30-40s AFTER the research
 * message lands, and `useChatFull` has no refetchInterval — so without a poll
 * the surface stays empty until a manual refresh. This selects the ids to poll
 * from two triggers, so BOTH explicit surface actions AND regular composer runs
 * under a surface agent are covered (the latter never seed `surfaceRunState`).
 */

import type { FullMessage } from '@/types';
import type { RunReference } from '@/types/surface';

/** A message is "settled" when its envelope exists and no slot is pending. */
function slotsSettled(msg: FullMessage | undefined): boolean {
  const slots = msg?.structuredOutput?.meta?.slots;
  if (!msg?.structuredOutput || !slots) return false;
  return !Object.values(slots).some((s) => s.status === 'pending');
}

/**
 * Message ids whose structured-output envelope is still pending and should be
 * polled for. Two triggers, unioned:
 *   1. Explicit surface-action runs: completed refs in `surfaceRunState`
 *      (parity with the previous behavior — a ref whose message isn't in
 *      `messages` yet is treated as unsettled, so the poll still starts).
 *   2. Regular composer runs under a surface agent: agent messages that ran a
 *      research session (envelope expected) — these never seed `surfaceRunState`.
 *      Gated on `m.researchSession` because the backend sources
 *      `structured_output` from `research_session.verification_data`
 *      (api/v1/chats.py serializer), so an envelope is ALWAYS accompanied by a
 *      research session; the gate is a strict superset of "has envelope".
 *
 * Ids whose message is already settled (envelope present, no pending slot) are
 * excluded, so loaded/old chats schedule no polls.
 */
export function messageIdsNeedingLiveFill(
  surfaceRunState: Record<string, RunReference | null>,
  messages: readonly FullMessage[] | undefined,
  hasSurface: boolean,
): string[] {
  const ids = new Set<string>();

  // Trigger 1: explicit surface-action runs.
  for (const ref of Object.values(surfaceRunState)) {
    if (ref && ref.status === 'completed' && ref.message_id) {
      ids.add(ref.message_id);
    }
  }

  // Trigger 2: regular runs under a surface agent expecting an envelope.
  if (hasSurface && messages) {
    for (const m of messages) {
      if (m.role === 'agent' && m.researchSession) ids.add(m.id);
    }
  }

  const byId = new Map<string, FullMessage>();
  for (const m of messages ?? []) byId.set(m.id, m);

  return [...ids].filter((id) => !slotsSettled(byId.get(id)));
}

// ---------------------------------------------------------------------------
// Live-fill poll tick accounting (pure)
// ---------------------------------------------------------------------------
//
// The live-fill poll (ChatPage.scheduleLiveFillPoll) runs one setInterval per
// message. Its ONLY job is to refetch chatFull until the structured-output
// envelope settles. The subtle bug this accounting fixes: the poll is scheduled
// as soon as the research message appears (during `classifying`, before any
// envelope exists), so a fixed budget counted from schedule time would be
// consumed by a long research run and expire BEFORE structuring even begins —
// and the poll's permanent per-message dedup guard would never let it restart,
// leaving the slots stuck on "pending" until a manual reload.
//
// The fix: the bounded structuring budget only advances once the envelope stub
// actually exists (`hasEnvelope`). Before that, the single long-lived timer
// stays alive without burning the budget so it is still ticking when the
// envelope finally lands. Three phases, each with its own cap:

/** Structuring window: 120 ticks × 5s = 10 min. Matches STALE_PENDING_MS in
 * surfaceEnrichment (when the UI downgrades pending → failed and offers Retry,
 * which spins its own poll), so the poll retires exactly as that takes over. */
export const LIVE_FILL_STRUCTURING_MAX_TICKS = 120;

/** Research still running, no envelope yet: keep the timer alive up to ~3h so it
 * outlives SSE drops on very long runs. */
export const LIVE_FILL_RESEARCH_IDLE_MAX_TICKS = 2160;

/** Run finished but no envelope yet: short post-completion grace (12 × 5s = 60s)
 * for the persisted stub to surface. Measured from completion (a separate
 * counter), not cumulative idle — else a long run would trip it instantly. A
 * genuinely never-structured run (failed / web_search-mode / historical) then
 * retires in 60s instead of polling for hours. */
export const LIVE_FILL_TERMINAL_IDLE_MAX_TICKS = 12;

/** During research idle, only invalidate chatFull every 6th tick (~30s) — a
 * cheap self-discovery of the stub without hammering the backend. */
export const LIVE_FILL_IDLE_INVALIDATE_EVERY = 6;

export interface LiveFillPollState {
  /** 5s ticks while research runs and no envelope exists yet. */
  idleTicks: number;
  /** 5s ticks once the envelope stub exists (the bounded structuring window). */
  structuringTicks: number;
  /** 5s ticks after the run is terminal but still no envelope (grace window). */
  terminalIdleTicks: number;
}

export interface LiveFillPollDecision {
  /** Whether the caller should invalidate the chatFull query this tick. */
  invalidate: boolean;
  /** Whether the caller should clear the interval. */
  stop: boolean;
  next: LiveFillPollState;
}

export const initialLiveFillPollState = (): LiveFillPollState => ({
  idleTicks: 0,
  structuringTicks: 0,
  terminalIdleTicks: 0,
});

/**
 * One tick of the live-fill poll.
 *
 * @param hasEnvelope  the message's structured-output envelope (meta.slots) exists
 * @param settled      envelope exists AND no slot is still "pending"
 * @param runTerminal  the run is no longer in progress (agentStatus complete/error/idle)
 */
export function liveFillPollTick(
  state: LiveFillPollState,
  hasEnvelope: boolean,
  settled: boolean,
  runTerminal: boolean,
): LiveFillPollDecision {
  // Envelope fully filled — nothing left to poll.
  if (settled) {
    return { invalidate: false, stop: true, next: state };
  }

  // Structuring window: envelope stub present, poll actively for the fill.
  if (hasEnvelope) {
    const structuringTicks = state.structuringTicks + 1;
    return {
      invalidate: true,
      stop: structuringTicks >= LIVE_FILL_STRUCTURING_MAX_TICKS,
      next: { ...state, structuringTicks },
    };
  }

  // Run finished but no envelope yet: short grace for the stub to surface,
  // polled every tick and measured from completion.
  if (runTerminal) {
    const terminalIdleTicks = state.terminalIdleTicks + 1;
    return {
      invalidate: true,
      stop: terminalIdleTicks >= LIVE_FILL_TERMINAL_IDLE_MAX_TICKS,
      next: { ...state, terminalIdleTicks },
    };
  }

  // Research in progress, no envelope yet: keep the timer alive WITHOUT burning
  // the structuring budget; re-check chatFull ~every 30s to self-discover the stub.
  const idleTicks = state.idleTicks + 1;
  return {
    invalidate: idleTicks % LIVE_FILL_IDLE_INVALIDATE_EVERY === 0,
    stop: idleTicks >= LIVE_FILL_RESEARCH_IDLE_MAX_TICKS,
    next: { ...state, idleTicks },
  };
}
