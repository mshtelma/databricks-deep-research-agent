import { describe, it, expect } from 'vitest';

import {
  mapJobStatusToRunStatus,
  toPersistedActionRun,
  actionRunsNeedingReconcile,
  surfaceRunsNeedingLiveReconcile,
  computeCaptureRun,
  surfaceRunScopeKey,
  surfaceRunScopeMatches,
  surfaceRunStateFromPersistedActionRuns,
} from '../surfaceRunReconcile';
import type { PersistedActionRun } from '@/types';
import type { RunReference } from '@/types/surface';

describe('mapJobStatusToRunStatus', () => {
  it('maps terminal + in-progress statuses', () => {
    expect(mapJobStatusToRunStatus('completed')).toBe('completed');
    expect(mapJobStatusToRunStatus('failed')).toBe('failed');
    expect(mapJobStatusToRunStatus('cancelled')).toBe('cancelled');
    expect(mapJobStatusToRunStatus('in_progress')).toBe('running');
  });

  it('treats an unknown status as not-yet-terminal (running), never mis-terminal', () => {
    expect(mapJobStatusToRunStatus('queued')).toBe('running');
    expect(mapJobStatusToRunStatus('')).toBe('running');
  });
});

describe('toPersistedActionRun', () => {
  it('keeps only persistable fields and stamps updated_at', () => {
    const ref: RunReference = { status: 'completed', session_id: 's1', message_id: 'm1' };
    expect(toPersistedActionRun(ref, '2026-07-04T00:00:00Z')).toEqual({
      status: 'completed',
      session_id: 's1',
      message_id: 'm1',
      updated_at: '2026-07-04T00:00:00Z',
    });
  });

  it('drops enrichment payload (data/sources/slotsMeta) to avoid surface_state bloat', () => {
    const ref: RunReference = {
      status: 'completed',
      session_id: 's1',
      message_id: 'm1',
      data: { comparison: [{ item: 'A' }] },
      sources: [{ ref: '1', url: 'https://a', title: 'A' }],
      slotsMeta: { comparison: { status: 'ok' } },
    };
    const out = toPersistedActionRun(ref, 'now');
    expect(out).toEqual({ status: 'completed', session_id: 's1', message_id: 'm1', updated_at: 'now' });
    expect('data' in out).toBe(false);
    expect('sources' in out).toBe(false);
    expect('slotsMeta' in out).toBe(false);
  });

  it('omits absent optional ids', () => {
    expect(toPersistedActionRun({ status: 'failed' }, 'now')).toEqual({ status: 'failed', updated_at: 'now' });
  });
});

describe('actionRunsNeedingReconcile', () => {
  it('returns [] for undefined', () => {
    expect(actionRunsNeedingReconcile(undefined)).toEqual([]);
  });

  it('includes running + failed entries that carry a session_id (with the full run)', () => {
    const actionRuns: Record<string, PersistedActionRun> = {
      run: { status: 'failed', session_id: 's-run', message_id: 'm-run' },
      compare: { status: 'running', session_id: 's-cmp' },
    };
    const out = actionRunsNeedingReconcile(actionRuns);
    expect(out).toEqual(
      expect.arrayContaining([
        { action: 'run', sessionId: 's-run', run: actionRuns.run },
        { action: 'compare', sessionId: 's-cmp', run: actionRuns.compare },
      ]),
    );
    expect(out).toHaveLength(2);
    // message_id is preserved via the returned `run` (Job payload has none).
    expect(out.find((e) => e.action === 'run')?.run.message_id).toBe('m-run');
  });

  it('excludes a failed entry with NO session_id (cannot be verified — pre-fix chats)', () => {
    expect(actionRunsNeedingReconcile({ run: { status: 'failed' } })).toEqual([]);
  });

  it('excludes completed and cancelled entries', () => {
    const out = actionRunsNeedingReconcile({
      run: { status: 'completed', session_id: 's1' },
      other: { status: 'cancelled', session_id: 's2' },
    });
    expect(out).toEqual([]);
  });

  it('returns only the eligible entries from a mixed set', () => {
    const out = actionRunsNeedingReconcile({
      a: { status: 'completed', session_id: 'sa' }, // excluded
      b: { status: 'failed', session_id: 'sb' }, // included
      c: { status: 'failed' }, // excluded (no session)
      d: { status: 'running', session_id: 'sd' }, // included
    });
    expect(out.map((e) => e.action).sort()).toEqual(['b', 'd']);
  });
});

describe('computeCaptureRun (L0 overwrite-race guard, FIX 1)', () => {
  it('stamps running+session when no entry exists yet', () => {
    expect(computeCaptureRun(undefined, 's1')).toEqual({ status: 'running', session_id: 's1' });
    expect(computeCaptureRun(null, 's1')).toEqual({ status: 'running', session_id: 's1' });
  });

  it('re-stamps a running entry that has a different (or no) session', () => {
    expect(computeCaptureRun({ status: 'running' }, 's1')).toEqual({ status: 'running', session_id: 's1' });
    expect(computeCaptureRun({ status: 'running', session_id: 'old' }, 's1')).toEqual({
      status: 'running',
      session_id: 's1',
    });
  });

  it('is a no-op when the running entry already carries this session', () => {
    expect(computeCaptureRun({ status: 'running', session_id: 's1' }, 's1')).toBeNull();
  });

  it('NEVER flips a terminal entry back to running (the overwrite race)', () => {
    // A post-completion reconnect re-fire must not clobber a resolved run.
    expect(computeCaptureRun({ status: 'completed', message_id: 'm1', session_id: 's1' }, 's1')).toBeNull();
    expect(computeCaptureRun({ status: 'failed', session_id: 's1' }, 's1')).toBeNull();
    expect(computeCaptureRun({ status: 'cancelled', session_id: 's1' }, 's2')).toBeNull();
  });
});

describe('surface run scope helpers', () => {
  it('distinguishes chat+agent scopes so common action names cannot leak', () => {
    const oldScope = surfaceRunScopeKey('chat-a', 'agent-a');
    const nextScope = surfaceRunScopeKey('chat-b', 'agent-b');

    expect(oldScope).toBe('chat-a:agent-a');
    expect(nextScope).toBe('chat-b:agent-b');
    expect(oldScope).not.toBe(nextScope);
    expect(
      surfaceRunScopeMatches(
        { chatId: 'chat-a', agentId: 'agent-a' },
        'chat-b',
        'agent-b',
      ),
    ).toBe(false);
  });

  it('projects the new scope persisted runs after stale state is reset', () => {
    const stale: Record<string, RunReference | null> = {
      run: { status: 'completed', message_id: 'old-message' },
    };
    const seeded = surfaceRunStateFromPersistedActionRuns({
      run: { status: 'completed', message_id: 'new-message', session_id: 'new-session' },
    });

    expect(seeded).toEqual({
      run: {
        status: 'completed',
        message_id: 'new-message',
        session_id: 'new-session',
      },
    });
    expect(seeded.run).not.toEqual(stale.run);
  });

  it('guards in-flight completions to the originating chat and agent', () => {
    const inflight = { chatId: 'chat-a', agentId: 'agent-a' };

    expect(surfaceRunScopeMatches(inflight, 'chat-a', 'agent-a')).toBe(true);
    expect(surfaceRunScopeMatches(inflight, 'chat-a', 'agent-b')).toBe(false);
    expect(surfaceRunScopeMatches(inflight, 'chat-b', 'agent-a')).toBe(false);
  });
});

describe('surfaceRunsNeedingLiveReconcile (live completion heal — L1.5)', () => {
  it('returns [] for an empty map', () => {
    expect(surfaceRunsNeedingLiveReconcile({})).toEqual([]);
  });

  it('selects running refs that carry a session_id (in-memory twin of L3)', () => {
    const state: Record<string, RunReference | null> = {
      run: { status: 'running', session_id: 's-run' },
      compare: { status: 'running', session_id: 's-cmp', message_id: 'm-cmp' },
    };
    const out = surfaceRunsNeedingLiveReconcile(state);
    expect(out).toEqual(
      expect.arrayContaining([
        { action: 'run', sessionId: 's-run' },
        { action: 'compare', sessionId: 's-cmp' },
      ]),
    );
    expect(out).toHaveLength(2);
  });

  it('excludes a running ref with NO session_id (cannot be verified)', () => {
    expect(surfaceRunsNeedingLiveReconcile({ run: { status: 'running' } })).toEqual([]);
  });

  it('excludes terminal refs and null entries (heal only stuck running refs)', () => {
    const out = surfaceRunsNeedingLiveReconcile({
      a: { status: 'completed', session_id: 'sa', message_id: 'ma' },
      b: { status: 'failed', session_id: 'sb' },
      c: { status: 'cancelled', session_id: 'sc' },
      d: null,
    });
    expect(out).toEqual([]);
  });

  it('returns only the eligible entries from a mixed set', () => {
    const out = surfaceRunsNeedingLiveReconcile({
      a: { status: 'completed', session_id: 'sa' }, // excluded (terminal)
      b: { status: 'running', session_id: 'sb' }, // included
      c: { status: 'running' }, // excluded (no session)
      d: null, // excluded
    });
    expect(out.map((e) => e.action)).toEqual(['b']);
  });
});
