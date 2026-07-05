import { describe, it, expect } from 'vitest';

import { enrichSurfaceRunState } from '../surfaceEnrichment';
import type { FullMessage, StructuredOutputEnvelope } from '@/types';
import type { RunReference } from '@/types/surface';

function message(
  id: string,
  structuredOutput: StructuredOutputEnvelope | null,
  createdAt = '',
): FullMessage {
  return {
    id,
    chatId: 'c1',
    role: 'agent',
    content: '',
    createdAt,
    isEdited: false,
    researchSession: null,
    claims: [],
    verificationSummary: null,
    structuredOutput,
  };
}

const NOW = Date.parse('2026-07-02T12:00:00Z');

function envelope(
  over: Partial<StructuredOutputEnvelope> = {},
): StructuredOutputEnvelope {
  return {
    version: 2,
    binding: 'run',
    generated_at: '2026-07-02T11:59:00Z',
    data: { comparison: [{ item: 'A', source_refs: ['1'] }] },
    meta: {
      slots: { comparison: { status: 'ok' } },
      sources: [{ ref: '1', url: 'https://a', title: 'A' }],
    },
    ...over,
  };
}

describe('enrichSurfaceRunState', () => {
  const completed: RunReference = { status: 'completed', message_id: 'm1' };

  it('attaches data, sources legend, and slotsMeta for a matching binding', () => {
    const out = enrichSurfaceRunState(
      { run: completed },
      [message('m1', envelope())],
      NOW,
    );
    expect(out.run?.data).toEqual({
      comparison: [{ item: 'A', source_refs: ['1'] }],
    });
    expect(out.run?.sources).toEqual([{ ref: '1', url: 'https://a', title: 'A' }]);
    expect(out.run?.slotsMeta).toEqual({ comparison: { status: 'ok' } });
  });

  it('does not enrich when the envelope binding differs (guard)', () => {
    const state = { run: completed };
    const out = enrichSurfaceRunState(
      state,
      [message('m1', envelope({ binding: 'other' }))],
      NOW,
    );
    expect(out.run?.data).toBeUndefined();
    expect(out.run).toBe(completed); // ref passed through untouched
  });

  it('preserves identity when nothing is enriched', () => {
    const state = { run: { status: 'running' as const } };
    expect(enrichSurfaceRunState(state, [], NOW)).toBe(state);
  });

  it('enriches even with empty data as long as slot meta exists (pending stub)', () => {
    const stub = envelope({
      data: {},
      generated_at: '2026-07-02T11:59:30Z',
      meta: { slots: { comparison: { status: 'pending' } }, sources: [] },
    });
    const out = enrichSurfaceRunState({ run: completed }, [message('m1', stub)], NOW);
    expect(out.run?.slotsMeta).toEqual({ comparison: { status: 'pending' } });
  });

  it('downgrades stale pending slots to failed (offer Retry, not a stuck skeleton)', () => {
    const stale = envelope({
      data: {},
      generated_at: '2026-07-02T11:40:00Z', // 20 min old
      meta: {
        slots: { comparison: { status: 'pending' }, metrics: { status: 'ok' } },
        sources: [],
      },
    });
    const out = enrichSurfaceRunState({ run: completed }, [message('m1', stale)], NOW);
    expect(out.run?.slotsMeta?.comparison?.status).toBe('failed');
    expect(out.run?.slotsMeta?.metrics?.status).toBe('ok'); // untouched
  });

  it('keeps recent pending slots pending', () => {
    const recent = envelope({
      data: {},
      generated_at: '2026-07-02T11:58:00Z', // 2 min old
      meta: { slots: { comparison: { status: 'pending' } }, sources: [] },
    });
    const out = enrichSurfaceRunState({ run: completed }, [message('m1', recent)], NOW);
    expect(out.run?.slotsMeta?.comparison?.status).toBe('pending');
  });

  it('leaves non-completed refs untouched', () => {
    const running: RunReference = { status: 'running' };
    const out = enrichSurfaceRunState(
      { run: running },
      [message('m1', envelope())],
      NOW,
    );
    expect(out.run).toBe(running);
  });

  it('marks completed refs as pending while live-fill awaits a missing envelope', () => {
    const out = enrichSurfaceRunState(
      { run: completed },
      [message('m1', null)],
      NOW,
      new Set(['m1']),
    );
    expect(out.run?.pendingStructuredOutput).toBe(true);
    expect(out.run?.data).toBeUndefined();
  });

  it('marks completed refs as pending even before the message list catches up', () => {
    const out = enrichSurfaceRunState(
      { run: completed },
      [],
      NOW,
      new Set(['m1']),
    );
    expect(out.run?.pendingStructuredOutput).toBe(true);
  });
});

describe('enrichSurfaceRunState — backfill from messages (no surface_state)', () => {
  it('backfills + enriches a completed ref from a message when run-state is empty', () => {
    const out = enrichSurfaceRunState({}, [message('m1', envelope())], NOW);
    expect(out.run?.status).toBe('completed');
    expect(out.run?.message_id).toBe('m1');
    expect(out.run?.data).toEqual({ comparison: [{ item: 'A', source_refs: ['1'] }] });
    expect(out.run?.slotsMeta).toEqual({ comparison: { status: 'ok' } });
  });

  it('picks the newest message per binding by createdAt (distinct dates)', () => {
    const out = enrichSurfaceRunState(
      {},
      [
        message('old', envelope(), '2026-07-02T10:00:00Z'),
        message('new', envelope(), '2026-07-02T11:00:00Z'),
      ],
      NOW,
    );
    expect(out.run?.message_id).toBe('new');
  });

  it('is robust to empty/unparseable createdAt — later array element wins (NaN guard)', () => {
    const out = enrichSurfaceRunState(
      {},
      [message('first', envelope()), message('second', envelope())], // both createdAt ''
      NOW,
    );
    expect(out.run?.message_id).toBe('second');
  });

  it('does not backfill or clobber a live running entry', () => {
    const state = { run: { status: 'running' as const } };
    const out = enrichSurfaceRunState(state, [message('m1', envelope())], NOW);
    expect(out).toBe(state); // identity preserved
    expect(out.run?.status).toBe('running');
  });

  it('does not overwrite a persisted completed entry that already has a message_id', () => {
    const state = { run: { status: 'completed' as const, message_id: 'old' } };
    const out = enrichSurfaceRunState(
      state,
      [message('old', envelope()), message('new', envelope())],
      NOW,
    );
    expect(out.run?.message_id).toBe('old'); // not replaced by the newer 'new'
  });

  it('attaches a message_id to a persisted completed entry that lacks one, then enriches', () => {
    const state = { run: { status: 'completed' as const } };
    const out = enrichSurfaceRunState(state, [message('m1', envelope())], NOW);
    expect(out.run?.message_id).toBe('m1');
    expect(out.run?.data).toEqual({ comparison: [{ item: 'A', source_refs: ['1'] }] });
  });

  it('preserves identity when messages carry no structured output', () => {
    const state = { run: { status: 'running' as const } };
    const out = enrichSurfaceRunState(state, [message('u1', null)], NOW);
    expect(out).toBe(state);
  });

  it('backfill is message-driven: a stray binding is created+enriched, run untouched (F3)', () => {
    const completed: RunReference = { status: 'completed', message_id: 'm1' };
    const out = enrichSurfaceRunState(
      { run: completed },
      [message('m1', envelope({ binding: 'other' }))],
      NOW,
    );
    // `run` has no matching envelope → untouched input ref (no data).
    expect(out.run).toBe(completed);
    expect(out.run?.data).toBeUndefined();
    // `other` is backfilled from the message and enriched. Harmless stray entry:
    // AgentSurfacePanel's overlay only reads runState[b.action] for current bindings.
    expect(out.other?.status).toBe('completed');
    expect(out.other?.message_id).toBe('m1');
    expect(out.other?.data).toEqual({ comparison: [{ item: 'A', source_refs: ['1'] }] });
  });
});
