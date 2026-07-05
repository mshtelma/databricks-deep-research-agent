import { describe, expect, it } from 'vitest';

import { buildSamplePayload, buildSampleReport } from '../sampleReport';

describe('buildSampleReport', () => {
  it('is deterministic and echoes the query and inputs', () => {
    const compiled = {
      query: 'Compare the last two quarterly filings',
      surfaceInputs: { region: 'emea', year: 2026 },
      researchDepth: 'light',
    };
    const a = buildSampleReport('My Agent', compiled);
    const b = buildSampleReport('My Agent', compiled);

    expect(a.markdown).toBe(b.markdown);
    expect(a.markdown).toContain('Compare the last two quarterly filings');
    expect(a.markdown).toContain('**region:** emea');
    expect(a.markdown).toContain('**year:** 2026');
    expect(a.markdown).toContain('My Agent');
    // Citation markers present and backed by citationData entries.
    expect(a.markdown).toContain('[1]');
    expect(a.markdown).toContain('[2]');
    expect(a.citationData.get('1')?.verdict).toBe('supported');
    expect(a.citationData.get('2')?.claim.citationKey).toBe('2');
  });

  it('handles empty query, empty inputs, and blank agent name', () => {
    const report = buildSampleReport('   ', { query: '', surfaceInputs: {} });
    expect(report.markdown).toContain('Sample findings');
    expect(report.markdown).not.toContain('**Request.**');
    expect(report.markdown).not.toContain('**Inputs**');
    expect(report.markdown).toContain('This agent');
  });

  it('truncates very long queries', () => {
    const longQuery = 'x'.repeat(500);
    const report = buildSampleReport('A', {
      query: longQuery,
      surfaceInputs: {},
    });
    expect(report.markdown).toContain('…');
    expect(report.markdown).not.toContain(longQuery);
  });
});

describe('buildSamplePayload', () => {
  const surface = {
    version: 1,
    components: [
      {
        id: 'root',
        component: 'Column',
        props: {},
        children: ['tbl', 'cht', 'metrics', 'findings', 'static'],
      },
      {
        id: 'tbl',
        component: 'Table',
        props: {
          source: { path: '/results/run/data/comparison' },
          columns: [
            { key: 'item', label: 'Item', type: 'string' },
            { key: 'score', label: 'Score', type: 'number' },
            { key: 'as_of', label: 'As of', type: 'date' },
          ],
        },
        children: [],
      },
      {
        id: 'cht',
        component: 'Chart',
        props: {
          source: { path: '/results/run/data/comparison' },
          kind: 'bar',
          x_key: 'item',
          y_keys: ['score'],
        },
        children: [],
      },
      {
        id: 'metrics',
        component: 'MetricGrid',
        props: { source: { path: '/results/run/data/headline_metrics' } },
        children: [],
      },
      {
        id: 'findings',
        component: 'KeyFindings',
        props: { source: { path: '/results/run/data/key_findings' } },
        children: [],
      },
      {
        id: 'static',
        component: 'List',
        props: { items: { path: '/static_items' } },
        children: [],
      },
    ],
    data_model: { query: '', static_items: [] },
    bindings: [
      {
        action: 'run',
        kind: 'run_agent',
        inputs: { query: { path: '/query' } },
        options: {},
        output: { target: '/results/run', mode: 'report' },
        concurrency: 'replace',
      },
    ],
  } as unknown as import('@/types/surface').Surface;

  it('fills every slot deterministically (Table wins a shared Chart slot)', () => {
    const payload = buildSamplePayload(surface, 'run');
    expect(payload).not.toBeNull();
    const rows = payload!['comparison'] as Record<string, unknown>[];
    expect(rows).toHaveLength(3);
    // Table columns drive the shared slot: markers on the first string column.
    expect(rows[0]!['item']).toContain('[1]');
    expect(typeof rows[0]!['score']).toBe('number');
    expect(rows[0]!['as_of']).toBe('2026-01-01');
    expect(payload!['headline_metrics']).toBeInstanceOf(Array);
    expect((payload!['key_findings'] as string[])[0]).toContain('[1]');
    // Non-slot List reads static data — never sampled.
    expect(payload!['static_items']).toBeUndefined();
    // Deterministic
    expect(buildSamplePayload(surface, 'run')).toEqual(payload);
  });

  it('returns null for an unknown action or slotless surface', () => {
    expect(buildSamplePayload(surface, 'missing')).toBeNull();
    const bare = {
      ...surface,
      components: [
        { id: 'root', component: 'Column', props: {}, children: [] },
      ],
    } as unknown as import('@/types/surface').Surface;
    expect(buildSamplePayload(bare, 'run')).toBeNull();
  });
});
