import { describe, it, expect } from 'vitest';

import {
  messageIdsNeedingLiveFill,
  liveFillPollTick,
  initialLiveFillPollState,
  LIVE_FILL_STRUCTURING_MAX_TICKS,
  LIVE_FILL_RESEARCH_IDLE_MAX_TICKS,
  LIVE_FILL_TERMINAL_IDLE_MAX_TICKS,
  LIVE_FILL_IDLE_INVALIDATE_EVERY,
} from '../surfaceLiveFill';
import type { FullMessage, StructuredOutputEnvelope } from '@/types';
import type { RunReference } from '@/types/surface';

// A non-null research-session stub — the helper only checks truthiness.
const RESEARCH = { id: 'rs1' } as unknown as NonNullable<FullMessage['researchSession']>;

/**
 * Local message factory (do NOT reuse surfaceEnrichment.test's — it hardcodes
 * role:'agent'/researchSession:null; several cases here override both).
 */
function message(
  id: string,
  opts: {
    role?: FullMessage['role'];
    researchSession?: FullMessage['researchSession'];
    structuredOutput?: StructuredOutputEnvelope | null;
  } = {},
): FullMessage {
  return {
    id,
    chatId: 'c1',
    role: opts.role ?? 'agent',
    content: '',
    createdAt: '',
    isEdited: false,
    researchSession: opts.researchSession ?? null,
    claims: [],
    verificationSummary: null,
    structuredOutput: opts.structuredOutput ?? null,
  };
}

function envelope(over: Partial<StructuredOutputEnvelope> = {}): StructuredOutputEnvelope {
  return {
    version: 2,
    binding: 'run',
    generated_at: '2026-07-03T21:53:00Z',
    data: {},
    meta: { slots: { s1: { status: 'ok' } }, sources: [] },
    ...over,
  };
}

const pendingEnvelope = (): StructuredOutputEnvelope =>
  envelope({ meta: { slots: { s1: { status: 'pending' } }, sources: [] } });

describe('messageIdsNeedingLiveFill', () => {
  it('1. regular live run: agent+researchSession, no envelope, surface → [msgId]', () => {
    const out = messageIdsNeedingLiveFill(
      {},
      [message('m1', { researchSession: RESEARCH, structuredOutput: null })],
      true,
    );
    expect(out).toEqual(['m1']);
  });

  it('2. settled research message (all slots ok) → [] (no poll on load)', () => {
    const out = messageIdsNeedingLiveFill(
      {},
      [message('m1', { researchSession: RESEARCH, structuredOutput: envelope() })],
      true,
    );
    expect(out).toEqual([]);
  });

  it('3. a pending slot → [msgId]', () => {
    const out = messageIdsNeedingLiveFill(
      {},
      [message('m1', { researchSession: RESEARCH, structuredOutput: pendingEnvelope() })],
      true,
    );
    expect(out).toEqual(['m1']);
  });

  it('4. non-surface agent: research message ignored by trigger 2 → []', () => {
    const out = messageIdsNeedingLiveFill(
      {},
      [message('m1', { researchSession: RESEARCH, structuredOutput: null })],
      false,
    );
    expect(out).toEqual([]);
  });

  it('5. simple response under surface (no researchSession) → [] (no false poll)', () => {
    const out = messageIdsNeedingLiveFill(
      {},
      [message('m1', { researchSession: null, structuredOutput: null })],
      true,
    );
    expect(out).toEqual([]);
  });

  it('6. surface action whose message is not yet in the list → [msgId] (parity)', () => {
    const surfaceRunState: Record<string, RunReference | null> = {
      run: { status: 'completed', message_id: 'm1' },
    };
    // Trigger 1 is independent of hasSurface.
    expect(messageIdsNeedingLiveFill(surfaceRunState, [], false)).toEqual(['m1']);
    expect(messageIdsNeedingLiveFill(surfaceRunState, undefined, true)).toEqual(['m1']);
  });

  it('7. surface action whose message is settled → []', () => {
    const out = messageIdsNeedingLiveFill(
      { run: { status: 'completed', message_id: 'm1' } },
      [message('m1', { structuredOutput: envelope() })],
      true,
    );
    expect(out).toEqual([]);
  });

  it('8. dedup: same message from both triggers appears once', () => {
    const out = messageIdsNeedingLiveFill(
      { run: { status: 'completed', message_id: 'm1' } },
      [message('m1', { researchSession: RESEARCH, structuredOutput: null })],
      true,
    );
    expect(out).toEqual(['m1']);
  });

  it('9. multi-turn: only the unsettled research message is polled', () => {
    const out = messageIdsNeedingLiveFill(
      {},
      [
        message('m1', { researchSession: RESEARCH, structuredOutput: envelope() }),
        message('m2', { researchSession: RESEARCH, structuredOutput: pendingEnvelope() }),
      ],
      true,
    );
    expect(out).toEqual(['m2']);
  });

  it('10. user message is never a trigger-2 candidate', () => {
    const out = messageIdsNeedingLiveFill(
      {},
      [message('u1', { role: 'user', researchSession: RESEARCH, structuredOutput: null })],
      true,
    );
    expect(out).toEqual([]);
  });

  it('11. a running surface ref is not polled (trigger 1 requires completed)', () => {
    const out = messageIdsNeedingLiveFill({ run: { status: 'running' } }, [], true);
    expect(out).toEqual([]);
  });

  it('12. a completed surface ref without a message_id is not polled', () => {
    const out = messageIdsNeedingLiveFill({ run: { status: 'completed' } }, [], true);
    expect(out).toEqual([]);
  });
});

describe('liveFillPollTick', () => {
  const S = initialLiveFillPollState;

  it('initial state is all-zero', () => {
    expect(S()).toEqual({
      idleTicks: 0,
      structuringTicks: 0,
      terminalIdleTicks: 0,
    });
  });

  it('settled takes precedence → stop, no invalidate, state unchanged', () => {
    const st = { idleTicks: 3, structuringTicks: 4, terminalIdleTicks: 0 };
    const d = liveFillPollTick(st, true, true, true);
    expect(d).toEqual({ invalidate: false, stop: true, next: st });
  });

  it('envelope present + not settled → advance structuring budget + invalidate', () => {
    const d = liveFillPollTick(S(), true, false, false);
    expect(d.invalidate).toBe(true);
    expect(d.stop).toBe(false);
    expect(d.next.structuringTicks).toBe(1);
    expect(d.next.idleTicks).toBe(0);
  });

  it('structuring budget stops at LIVE_FILL_STRUCTURING_MAX_TICKS', () => {
    const below = liveFillPollTick(
      {
        idleTicks: 0,
        structuringTicks: LIVE_FILL_STRUCTURING_MAX_TICKS - 2,
        terminalIdleTicks: 0,
      },
      true,
      false,
      false,
    );
    expect(below.stop).toBe(false);
    const at = liveFillPollTick(
      {
        idleTicks: 0,
        structuringTicks: LIVE_FILL_STRUCTURING_MAX_TICKS - 1,
        terminalIdleTicks: 0,
      },
      true,
      false,
      false,
    );
    expect(at.stop).toBe(true);
    expect(at.next.structuringTicks).toBe(LIVE_FILL_STRUCTURING_MAX_TICKS);
  });

  it('research idle: advance idle only, invalidate every 6th tick, never touch structuring', () => {
    const t1 = liveFillPollTick(S(), false, false, false);
    expect(t1.next.idleTicks).toBe(1);
    expect(t1.next.structuringTicks).toBe(0);
    expect(t1.invalidate).toBe(false);
    expect(t1.stop).toBe(false);
    const t6 = liveFillPollTick(
      {
        idleTicks: LIVE_FILL_IDLE_INVALIDATE_EVERY - 1,
        structuringTicks: 0,
        terminalIdleTicks: 0,
      },
      false,
      false,
      false,
    );
    expect(t6.invalidate).toBe(true);
  });

  it('research idle stops at the ~3h absolute cap', () => {
    const d = liveFillPollTick(
      {
        idleTicks: LIVE_FILL_RESEARCH_IDLE_MAX_TICKS - 1,
        structuringTicks: 0,
        terminalIdleTicks: 0,
      },
      false,
      false,
      false,
    );
    expect(d.stop).toBe(true);
  });

  it('terminal idle: poll every tick, stop at the 60s post-completion grace cap', () => {
    const t1 = liveFillPollTick(S(), false, false, true);
    expect(t1.next.terminalIdleTicks).toBe(1);
    expect(t1.invalidate).toBe(true);
    expect(t1.stop).toBe(false);
    const at = liveFillPollTick(
      {
        idleTicks: 0,
        structuringTicks: 0,
        terminalIdleTicks: LIVE_FILL_TERMINAL_IDLE_MAX_TICKS - 1,
      },
      false,
      false,
      true,
    );
    expect(at.stop).toBe(true);
  });

  // THE root-cause regression: a long research run must NOT consume the bounded
  // structuring budget. Pre-fix, the fixed 3-min budget counted from schedule
  // time (during research), expired mid-run, and the permanent dedup guard
  // blocked any restart → structured-output slots stuck until manual reload.
  it('REGRESSION: 200 research-idle ticks leave the structuring budget intact', () => {
    let st = S();
    for (let i = 0; i < 200; i++) {
      st = liveFillPollTick(st, false, false, false).next;
    }
    expect(st.structuringTicks).toBe(0);
    expect(st.idleTicks).toBe(200);
    const first = liveFillPollTick(st, true, false, false); // envelope finally lands
    expect(first.next.structuringTicks).toBe(1);
    expect(first.stop).toBe(false);
  });

  // The 60s terminal grace is measured FROM completion, not from cumulative idle
  // ticks — otherwise a long run would trip the grace cap the instant it ends.
  it('REGRESSION: terminal grace starts from completion after a long research run', () => {
    let st = S();
    for (let i = 0; i < 200; i++) {
      st = liveFillPollTick(st, false, false, false).next;
    }
    const firstTerminal = liveFillPollTick(st, false, false, true);
    expect(firstTerminal.stop).toBe(false);
    expect(firstTerminal.next.terminalIdleTicks).toBe(1);
    let s2 = firstTerminal.next;
    for (let i = 1; i < LIVE_FILL_TERMINAL_IDLE_MAX_TICKS - 1; i++) {
      s2 = liveFillPollTick(s2, false, false, true).next;
    }
    expect(liveFillPollTick(s2, false, false, true).stop).toBe(true);
  });
});
