import { describe, expect, it } from 'vitest';

import type { StructuredOutputEnvelope } from '@/types';
import type { RunReference } from '@/types/surface';

import { applyEnvelopeToRef, parseSseFrame, safeJsonParse } from '../shellApi';

describe('parseSseFrame', () => {
  it('parses event + data + id', () => {
    const f = parseSseFrame('event: complete\ndata: {"output":"hi"}\nid: 7');
    expect(f).toEqual({ event: 'complete', data: '{"output":"hi"}', id: 7 });
  });

  it('joins multi-line data and defaults event to "data"', () => {
    const f = parseSseFrame('data: line1\ndata: line2');
    expect(f?.event).toBe('data');
    expect(f?.data).toBe('line1\nline2');
    expect(f?.id).toBeNull();
  });

  it('returns null for a keepalive/comment frame (no data lines)', () => {
    expect(parseSseFrame(': keepalive')).toBeNull();
  });
});

describe('safeJsonParse', () => {
  it('falls back to {raw} on invalid JSON', () => {
    expect(safeJsonParse('not json')).toEqual({ raw: 'not json' });
  });
});

describe('applyEnvelopeToRef', () => {
  it('overlays data + sources + slotsMeta from the envelope, preserving status', () => {
    const ref: RunReference = { status: 'running', message_id: 'm1' };
    const env: StructuredOutputEnvelope = {
      version: 2,
      binding: 'run',
      data: { findings: [{ text: 'f', source_refs: ['1'] }] },
      meta: {
        sources: [{ ref: '1', url: 'https://a', title: 'A' }],
        slots: { findings: { status: 'ok' } },
      },
    };
    const out = applyEnvelopeToRef(ref, env);
    expect(out.status).toBe('running'); // preserved until `complete`
    expect(out.message_id).toBe('m1');
    expect(out.data).toBe(env.data);
    expect(out.sources).toEqual([{ ref: '1', url: 'https://a', title: 'A' }]);
    expect(out.slotsMeta).toEqual({ findings: { status: 'ok' } });
  });
});
