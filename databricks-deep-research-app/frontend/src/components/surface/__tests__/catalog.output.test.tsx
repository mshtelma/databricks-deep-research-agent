import '@testing-library/jest-dom';
import { describe, it, expect, vi } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';

import { SurfaceRenderer } from '../SurfaceRenderer';
import { sanitizeRows } from '../SurfaceChart';
import type { Surface, RunReference } from '@/types/surface';
import type { CitationContext } from '@/components/common';
import type { Claim } from '@/types/citation';

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------

function makeSurface(componentIds: string[], extra: Surface['components']): Surface {
  return {
    version: 1,
    components: [
      { id: 'root', component: 'Column', props: {}, children: componentIds },
      ...extra,
    ],
    data_model: { query: '' },
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
  };
}

const TABLE_COMP: Surface['components'][number] = {
  id: 'tbl',
  component: 'Table',
  props: {
    source: { path: '/results/run/data/comparison' },
    columns: [
      { key: 'item', label: 'Item', type: 'string' },
      { key: 'score', label: 'Score', type: 'number' },
    ],
    empty_text: 'Nothing here yet.',
  },
  children: [],
};

function refModel(ref: RunReference | null): Record<string, unknown> {
  return { query: '', results: { run: ref } };
}

const COMPLETED_REF: RunReference = {
  status: 'completed',
  message_id: 'm1',
  data: {
    comparison: [
      { item: 'Option A [K1]', score: 8.1 },
      { item: 'Option B', score: 5 },
    ],
    headline_metrics: [
      { label: 'Coverage', value: '87', unit: '%', delta: '+4 vs prior' },
    ],
    key_findings: ['First finding [K1].', 'Second finding.', 'Third finding.'],
  },
};

function fakeCitations(): Map<string, CitationContext> {
  const claim = {
    id: 'c1',
    claimText: 'x',
    claimType: 'general',
    confidenceLevel: null,
    positionStart: 0,
    positionEnd: 1,
    verificationVerdict: 'supported',
    verificationReasoning: null,
    abstained: false,
    citations: [],
    corrections: [],
    numericDetail: null,
    citationKey: 'K1',
    citationKeys: ['K1'],
  } as unknown as Claim;
  return new Map([['K1', { claim, verdict: 'supported' }]]);
}

function renderSurface(
  surface: Surface,
  dataModel: Record<string, unknown>,
  resolveCitations?: (id: string) => Map<string, CitationContext> | undefined,
  retryStructuring?: (messageId: string, slots: string[]) => void,
) {
  return render(
    <SurfaceRenderer
      surface={surface}
      dataModel={dataModel}
      onDataModelChange={vi.fn()}
      onAction={vi.fn()}
      resolveCitations={resolveCitations}
      retryStructuring={retryStructuring}
    />,
  );
}

// ---------------------------------------------------------------------------
// Table
// ---------------------------------------------------------------------------

describe('Table', () => {
  it('renders typed rows with citation chips in marker cells', () => {
    const resolve = vi.fn(() => fakeCitations());
    renderSurface(makeSurface(['tbl'], [TABLE_COMP]), refModel(COMPLETED_REF), resolve);

    expect(screen.getByText('Item')).toBeInTheDocument();
    expect(screen.getByText('Score')).toBeInTheDocument();
    expect(screen.getByText('8.1')).toBeInTheDocument();
    expect(screen.getByText('Option B')).toBeInTheDocument();
    // The marker cell rendered through the citation path.
    expect(screen.getByTestId('citation-marker-K1')).toBeInTheDocument();
    expect(resolve).toHaveBeenCalledWith('m1');
  });

  it('shows empty_text before any run and a slot-specific message after', () => {
    const { rerender } = renderSurface(
      makeSurface(['tbl'], [TABLE_COMP]),
      refModel(null),
    );
    expect(screen.getByText('Nothing here yet.')).toBeInTheDocument();

    rerender(
      <SurfaceRenderer
        surface={makeSurface(['tbl'], [TABLE_COMP])}
        dataModel={refModel({ status: 'completed', message_id: 'm1', data: {} })}
        onDataModelChange={vi.fn()}
        onAction={vi.fn()}
      />,
    );
    expect(
      screen.getByText('No structured data for this section.'),
    ).toBeInTheDocument();
  });

  it('shows a loading placeholder while the run is in flight', () => {
    renderSurface(makeSurface(['tbl'], [TABLE_COMP]), refModel({ status: 'running' }));
    expect(screen.getByLabelText('Waiting for results')).toBeInTheDocument();
  });
});

// ---------------------------------------------------------------------------
// MetricGrid + KeyFindings + List
// ---------------------------------------------------------------------------

describe('MetricGrid / KeyFindings / List', () => {
  it('renders metric cards with unit and delta', () => {
    const comp: Surface['components'][number] = {
      id: 'metrics',
      component: 'MetricGrid',
      props: { source: { path: '/results/run/data/headline_metrics' } },
      children: [],
    };
    renderSurface(makeSurface(['metrics'], [comp]), refModel(COMPLETED_REF));
    expect(screen.getByText('Coverage')).toBeInTheDocument();
    expect(screen.getByText('87')).toBeInTheDocument();
    expect(screen.getByText('%')).toBeInTheDocument();
    expect(screen.getByText('+4 vs prior')).toBeInTheDocument();
  });

  it('renders findings and honors max_items', () => {
    const comp: Surface['components'][number] = {
      id: 'findings',
      component: 'KeyFindings',
      props: {
        source: { path: '/results/run/data/key_findings' },
        max_items: 2,
      },
      children: [],
    };
    renderSurface(makeSurface(['findings'], [comp]), refModel(COMPLETED_REF));
    expect(screen.getByText(/First finding/)).toBeInTheDocument();
    expect(screen.getByText('Second finding.')).toBeInTheDocument();
    expect(screen.queryByText('Third finding.')).not.toBeInTheDocument();
  });

  it('renders a List over a static data-model array', () => {
    const comp: Surface['components'][number] = {
      id: 'lst',
      component: 'List',
      props: { items: { path: '/static_items' }, ordered: true },
      children: [],
    };
    const surface = makeSurface(['lst'], [comp]);
    renderSurface(surface, { query: '', static_items: ['alpha', 'beta'] });
    expect(screen.getByText('alpha')).toBeInTheDocument();
    expect(screen.getByText('beta')).toBeInTheDocument();
  });
});

// ---------------------------------------------------------------------------
// Tabs
// ---------------------------------------------------------------------------

describe('Tabs', () => {
  it('renders pane labels and switches content', () => {
    const extra: Surface['components'] = [
      { id: 'tabs', component: 'Tabs', props: {}, children: ['p1', 'p2'] },
      { id: 'p1', component: 'TabPane', props: { label: 'Overview' }, children: ['t1'] },
      { id: 'p2', component: 'TabPane', props: { label: 'Details' }, children: ['t2'] },
      { id: 't1', component: 'Text', props: { text: 'overview body' }, children: [] },
      { id: 't2', component: 'Text', props: { text: 'details body' }, children: [] },
    ];
    renderSurface(makeSurface(['tabs'], extra), { query: '' });

    expect(screen.getByText('overview body')).toBeInTheDocument();
    expect(screen.queryByText('details body')).not.toBeInTheDocument();

    // Radix Tabs triggers activate on mousedown (not click) in jsdom.
    fireEvent.mouseDown(screen.getByText('Details'));
    expect(screen.getByText('details body')).toBeInTheDocument();
    expect(screen.queryByText('overview body')).not.toBeInTheDocument();
  });
});

// ---------------------------------------------------------------------------
// Chart
// ---------------------------------------------------------------------------

describe('Chart', () => {
  const CHART_COMP: Surface['components'][number] = {
    id: 'cht',
    component: 'Chart',
    props: {
      source: { path: '/results/run/data/comparison' },
      kind: 'bar',
      x_key: 'item',
      y_keys: ['score'],
      empty_text: 'No chart data.',
    },
    children: [],
  };

  it('shows the empty state without rows', () => {
    renderSurface(makeSurface(['cht'], [CHART_COMP]), refModel(null));
    expect(screen.getByText('No chart data.')).toBeInTheDocument();
  });

  it('mounts the lazy chart when rows are present', async () => {
    renderSurface(makeSurface(['cht'], [CHART_COMP]), refModel(COMPLETED_REF));
    await waitFor(() => {
      expect(screen.getByTestId('surface-chart')).toBeInTheDocument();
    });
  });

  it('sanitizeRows coerces numbers and strips markers from x labels', () => {
    const rows = sanitizeRows(
      [
        { item: 'Alpha [K1]', score: '7' },
        { item: 'Beta', score: 'not-a-number' },
      ],
      'item',
      ['score'],
    );
    expect(rows[0]).toEqual({ item: 'Alpha', score: 7 });
    expect(rows[1]).toEqual({ item: 'Beta', score: null });
  });
});

// ---------------------------------------------------------------------------
// v2 — source chips, both-shape findings, per-slot pending/failed states
// ---------------------------------------------------------------------------

const V2_REF: RunReference = {
  status: 'completed',
  message_id: 'm1',
  data: {
    comparison: [
      { item: 'Option A', score: 8.1, source_refs: ['1', '2'] },
      { item: 'Option B', score: 5, source_refs: [] },
    ],
    key_findings: [
      { text: 'Objects render [K1].', source_refs: ['1'] },
      'Legacy string still renders.',
    ],
  },
  sources: [
    { ref: '1', url: 'https://example.com/a', title: 'Source A' },
    { ref: '2', url: 'https://example.com/b', title: 'Source B' },
  ],
  slotsMeta: {
    comparison: { status: 'ok' },
    key_findings: { status: 'ok' },
  },
};

describe('source chips (v2)', () => {
  it('renders resolved chips as links in a Sources column', () => {
    renderSurface(makeSurface(['tbl'], [TABLE_COMP]), refModel(V2_REF));
    expect(screen.getByText('Sources')).toBeInTheDocument();
    const chip1 = screen.getByTestId('surface-source-chip-1');
    expect(chip1).toHaveAttribute('href', 'https://example.com/a');
    expect(chip1).toHaveAttribute('target', '_blank');
    expect(screen.getByTestId('surface-source-chip-2')).toBeInTheDocument();
  });

  it('KeyFindings renders v2 objects (text + chip) AND v1 strings', () => {
    const comp: Surface['components'][number] = {
      id: 'findings',
      component: 'KeyFindings',
      props: { source: { path: '/results/run/data/key_findings' } },
      children: [],
    };
    const resolve = vi.fn(() => fakeCitations());
    renderSurface(makeSurface(['findings'], [comp]), refModel(V2_REF), resolve);
    expect(screen.getByText(/Objects render/)).toBeInTheDocument();
    expect(screen.getByText('Legacy string still renders.')).toBeInTheDocument();
    expect(screen.getByTestId('surface-source-chip-1')).toBeInTheDocument();
  });

  it('omits the Sources column when no legend resolves', () => {
    const noLegend: RunReference = { ...V2_REF, sources: [] };
    renderSurface(makeSurface(['tbl'], [TABLE_COMP]), refModel(noLegend));
    expect(screen.queryByText('Sources')).not.toBeInTheDocument();
  });
});

describe('per-slot state machine (v2)', () => {
  const pendingRef: RunReference = {
    status: 'completed',
    message_id: 'm1',
    data: {},
    slotsMeta: { comparison: { status: 'pending' } },
  };

  it('shows a skeleton while a slot is pending (even after run completes)', () => {
    renderSurface(makeSurface(['tbl'], [TABLE_COMP]), refModel(pendingRef));
    expect(screen.getByLabelText('Waiting for results')).toBeInTheDocument();
  });

  it('shows a skeleton while a completed run is waiting for structured output', () => {
    renderSurface(
      makeSurface(['tbl'], [TABLE_COMP]),
      refModel({
        status: 'completed',
        message_id: 'm1',
        data: {},
        pendingStructuredOutput: true,
      }),
    );
    expect(screen.getByLabelText('Waiting for results')).toBeInTheDocument();
    expect(
      screen.queryByText('No structured data for this section.'),
    ).not.toBeInTheDocument();
  });

  it('shows a Retry button on a failed slot and calls retryStructuring', () => {
    const failedRef: RunReference = {
      status: 'completed',
      message_id: 'm1',
      data: {},
      slotsMeta: { comparison: { status: 'failed', error: 'boom' } },
    };
    const retry = vi.fn();
    renderSurface(
      makeSurface(['tbl'], [TABLE_COMP]),
      refModel(failedRef),
      undefined,
      retry,
    );
    const btn = screen.getByTestId('surface-slot-retry-comparison');
    fireEvent.click(btn);
    expect(retry).toHaveBeenCalledWith('m1', ['comparison']);
  });
});
