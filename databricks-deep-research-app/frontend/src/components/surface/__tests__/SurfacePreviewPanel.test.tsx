import '@testing-library/jest-dom';
import { describe, it, expect, vi, afterEach } from 'vitest';
import { render, screen, fireEvent, act, within } from '@testing-library/react';
import { MemoryRouter } from 'react-router-dom';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import type { ReactElement } from 'react';

import { SurfacePreviewPanel } from '../SurfacePreviewPanel';
import { CHAT_FULL_KEY } from '@/hooks/useChatFull';
import type { SurfacePreviewRunApi } from '@/hooks/useSurfacePreviewRun';
import type { Surface } from '@/types/surface';
import type { AST } from '@/types/ast';

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------

/** Scaffold-shaped surface: TextField(/query) + Button(run) + StatusBadge + ReportRegion. */
function makeSurface(overrides: Partial<Surface> = {}): Surface {
  return {
    version: 1,
    components: [
      {
        id: 'root',
        component: 'Column',
        props: {},
        children: ['q_field', 'run_btn', 'status', 'results'],
      },
      {
        id: 'q_field',
        component: 'TextField',
        props: { label: 'Query', value: { path: '/query' }, placeholder: 'Enter topic' },
        children: [],
      },
      { id: 'run_btn', component: 'Button', props: { label: 'Run', action: 'run' }, children: [] },
      {
        id: 'status',
        component: 'StatusBadge',
        props: { source: { path: '/results/run' } },
        children: [],
      },
      {
        id: 'results',
        component: 'ReportRegion',
        props: { source: { path: '/results/run' }, empty_text: 'No results yet.' },
        children: [],
      },
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
    ...overrides,
  };
}

function makeAst(surface: Surface | null = makeSurface()): AST {
  return { ...(surface ? { surface } : {}) } as unknown as AST;
}

function makePreviewRun(
  overrides: Partial<SurfacePreviewRunApi> = {},
): SurfacePreviewRunApi {
  return {
    runState: {},
    isActive: false,
    streamingContent: '',
    agentStatus: 'idle',
    errorDetails: null,
    previewChatId: null,
    start: vi.fn(),
    stop: vi.fn(),
    ...overrides,
  };
}

function renderPanel(ui: ReactElement, client?: QueryClient) {
  const qc =
    client ??
    new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(
    <QueryClientProvider client={qc}>
      <MemoryRouter>{ui}</MemoryRouter>
    </QueryClientProvider>,
  );
}

afterEach(() => {
  vi.useRealTimers();
});

// ---------------------------------------------------------------------------
// Part A — simulate
// ---------------------------------------------------------------------------

describe('SurfacePreviewPanel — simulate (sample output)', () => {
  it('plays running → completed sample with watermark and citation chips', () => {
    vi.useFakeTimers();
    renderPanel(<SurfacePreviewPanel ast={makeAst()} agentName="My Agent" />);

    fireEvent.change(screen.getByPlaceholderText('Enter topic'), {
      target: { value: 'What is 2+2?' },
    });
    fireEvent.click(screen.getByTestId('surface-action-run'));

    // Dry-run card + running phase.
    expect(screen.getByText(/Simulated run — action/)).toBeInTheDocument();
    expect(screen.getByText('Simulating…')).toBeInTheDocument();
    expect(screen.getByText('Running')).toBeInTheDocument(); // StatusBadge

    act(() => {
      vi.advanceTimersByTime(700);
    });

    const sample = screen.getByTestId('surface-preview-sample');
    expect(sample).toBeInTheDocument();
    expect(
      screen.getByText(/Sample output — illustrative only/),
    ).toBeInTheDocument();
    // Scoped: the query also appears in the dry-run card's JSON payload.
    expect(within(sample).getByText(/What is 2\+2\?/)).toBeInTheDocument();
    expect(screen.getByText('Completed')).toBeInTheDocument(); // StatusBadge
    // Fake citation chips render through the real citation path.
    expect(screen.getByTestId('citation-marker-1')).toBeInTheDocument();
  });

  it('Reset values clears the sample and the dry-run card', () => {
    vi.useFakeTimers();
    renderPanel(<SurfacePreviewPanel ast={makeAst()} />);

    fireEvent.click(screen.getByTestId('surface-action-run'));
    act(() => {
      vi.advanceTimersByTime(700);
    });
    expect(screen.getByTestId('surface-preview-sample')).toBeInTheDocument();

    fireEvent.click(screen.getByText('Reset values'));
    expect(screen.queryByTestId('surface-preview-sample')).not.toBeInTheDocument();
    expect(screen.queryByText(/Simulated run — action/)).not.toBeInTheDocument();
    expect(screen.getByText('No results yet.')).toBeInTheDocument();
  });
});

// ---------------------------------------------------------------------------
// Part B — Try in chat
// ---------------------------------------------------------------------------

describe('SurfacePreviewPanel — Try in chat', () => {
  it('renders the button when the callback is provided and forwards clicks', () => {
    const onTryInChat = vi.fn();
    renderPanel(
      <SurfacePreviewPanel ast={makeAst()} onTryInChat={onTryInChat} />,
    );
    fireEvent.click(screen.getByTestId('surface-preview-try-in-chat'));
    expect(onTryInChat).toHaveBeenCalledTimes(1);
  });

  it('is hidden without the callback and disabled while pending', () => {
    const { rerender } = renderPanel(<SurfacePreviewPanel ast={makeAst()} />);
    expect(
      screen.queryByTestId('surface-preview-try-in-chat'),
    ).not.toBeInTheDocument();

    rerender(
      <QueryClientProvider
        client={new QueryClient({ defaultOptions: { queries: { retry: false } } })}
      >
        <MemoryRouter>
          <SurfacePreviewPanel
            ast={makeAst()}
            onTryInChat={vi.fn()}
            tryInChatPending
          />
        </MemoryRouter>
      </QueryClientProvider>,
    );
    const btn = screen.getByTestId('surface-preview-try-in-chat');
    expect(btn).toBeDisabled();
    expect(btn).toHaveTextContent('Starting…');
  });
});

// ---------------------------------------------------------------------------
// Part C — Run for real
// ---------------------------------------------------------------------------

describe('SurfacePreviewPanel — Run for real', () => {
  it('hands the compiled submission to previewRun.start and disables while active', () => {
    const previewRun = makePreviewRun();
    const { rerender } = renderPanel(
      <SurfacePreviewPanel ast={makeAst()} previewRun={previewRun} />,
    );

    fireEvent.change(screen.getByPlaceholderText('Enter topic'), {
      target: { value: 'real question' },
    });
    fireEvent.click(screen.getByTestId('surface-action-run'));
    fireEvent.click(screen.getByTestId('surface-preview-run-real'));

    expect(previewRun.start).toHaveBeenCalledWith(
      'run',
      expect.objectContaining({ query: 'real question' }),
    );

    rerender(
      <QueryClientProvider
        client={new QueryClient({ defaultOptions: { queries: { retry: false } } })}
      >
        <MemoryRouter>
          <SurfacePreviewPanel
            ast={makeAst()}
            previewRun={makePreviewRun({ isActive: true })}
          />
        </MemoryRouter>
      </QueryClientProvider>,
    );
    fireEvent.click(screen.getByTestId('surface-action-run'));
    expect(screen.getByTestId('surface-preview-run-real')).toBeDisabled();
  });

  it('renders the live stream while running (real ref overlays the region)', () => {
    const previewRun = makePreviewRun({
      isActive: true,
      agentStatus: 'researching',
      streamingContent: '# Partial findings',
      runState: {
        run: { status: 'running', preview: 'real', action: 'run' },
      },
    });
    renderPanel(<SurfacePreviewPanel ast={makeAst()} previewRun={previewRun} />);

    expect(
      screen.getByTestId('surface-preview-real-running'),
    ).toBeInTheDocument();
    expect(screen.getByText('Partial findings')).toBeInTheDocument();
    expect(screen.getByText('Running')).toBeInTheDocument(); // StatusBadge
    fireEvent.click(screen.getByText('Stop'));
    expect(previewRun.stop).toHaveBeenCalledTimes(1);
  });

  it('resolves the completed report from the chatFull cache with a chat link', () => {
    const client = new QueryClient({
      defaultOptions: { queries: { retry: false } },
    });
    client.setQueryData([...CHAT_FULL_KEY, 'preview-chat-1'], {
      messages: [{ id: 'm1', content: '# Real report body' }],
    });
    const previewRun = makePreviewRun({
      previewChatId: 'preview-chat-1',
      runState: {
        run: {
          status: 'completed',
          preview: 'real',
          action: 'run',
          message_id: 'm1',
        },
      },
    });
    renderPanel(
      <SurfacePreviewPanel ast={makeAst()} previewRun={previewRun} />,
      client,
    );

    expect(
      screen.getByTestId('surface-preview-real-completed'),
    ).toBeInTheDocument();
    expect(screen.getByText('Real report body')).toBeInTheDocument();
    const link = screen.getByText('Open full report in chat →');
    expect(link).toHaveAttribute('href', '/chat/preview-chat-1');
  });

  it('real run supersedes the sample for the same action', () => {
    vi.useFakeTimers();
    const { rerender } = renderPanel(
      <SurfacePreviewPanel ast={makeAst()} previewRun={makePreviewRun()} />,
    );
    fireEvent.click(screen.getByTestId('surface-action-run'));
    act(() => {
      vi.advanceTimersByTime(700);
    });
    expect(screen.getByTestId('surface-preview-sample')).toBeInTheDocument();

    rerender(
      <QueryClientProvider
        client={new QueryClient({ defaultOptions: { queries: { retry: false } } })}
      >
        <MemoryRouter>
          <SurfacePreviewPanel
            ast={makeAst()}
            previewRun={makePreviewRun({
              isActive: true,
              streamingContent: 'live text',
              runState: {
                run: { status: 'running', preview: 'real', action: 'run' },
              },
            })}
          />
        </MemoryRouter>
      </QueryClientProvider>,
    );
    expect(
      screen.queryByTestId('surface-preview-sample'),
    ).not.toBeInTheDocument();
    expect(
      screen.getByTestId('surface-preview-real-running'),
    ).toBeInTheDocument();
  });

  it('failed run shows the error and Retry recompiles from the current form', () => {
    const previewRun = makePreviewRun({
      errorDetails: {
        error: new Error('Maximum concurrent jobs reached.'),
        errorCode: 'MAX_CONCURRENT_JOBS',
      },
      runState: {
        run: { status: 'failed', preview: 'real', action: 'run' },
      },
    });
    renderPanel(<SurfacePreviewPanel ast={makeAst()} previewRun={previewRun} />);

    expect(
      screen.getByTestId('surface-preview-real-failed'),
    ).toBeInTheDocument();
    expect(
      screen.getByText(/Maximum concurrent jobs reached/),
    ).toBeInTheDocument();

    fireEvent.change(screen.getByPlaceholderText('Enter topic'), {
      target: { value: 'second attempt' },
    });
    fireEvent.click(screen.getByText('Retry'));
    expect(previewRun.start).toHaveBeenCalledWith(
      'run',
      expect.objectContaining({ query: 'second attempt' }),
    );
  });

  it('does not crash when a run ref exists but its binding was removed', () => {
    const previewRun = makePreviewRun({
      runState: {
        run: { status: 'completed', preview: 'real', action: 'run' },
      },
    });
    const surface = makeSurface({ bindings: [] });
    renderPanel(
      <SurfacePreviewPanel ast={makeAst(surface)} previewRun={previewRun} />,
    );
    // No binding → nothing overlaid; the region falls back to its empty text
    // (no resolver output) and nothing throws.
    expect(screen.getByTestId('surface-action-run')).toBeInTheDocument();
  });
});

// ---------------------------------------------------------------------------
// Structured-output sample payloads
// ---------------------------------------------------------------------------

describe('SurfacePreviewPanel — structured sample payloads', () => {
  function makeStructuredSurface(): Surface {
    const base = makeSurface();
    return {
      ...base,
      components: [
        {
          id: 'root',
          component: 'Column',
          props: {},
          children: ['q_field', 'run_btn', 'status', 'tbl', 'findings'],
        },
        ...base.components.filter((c) =>
          ['q_field', 'run_btn', 'status'].includes(c.id),
        ),
        {
          id: 'tbl',
          component: 'Table',
          props: {
            source: { path: '/results/run/data/comparison' },
            columns: [
              { key: 'item', label: 'Item', type: 'string' },
              { key: 'score', label: 'Score', type: 'number' },
            ],
          },
          children: [],
        },
        {
          id: 'findings',
          component: 'KeyFindings',
          props: { source: { path: '/results/run/data/key_findings' } },
          children: [],
        },
      ],
    };
  }

  it('simulate fills tables and findings with sample data + citation chips', () => {
    vi.useFakeTimers();
    renderPanel(
      <SurfacePreviewPanel ast={makeAst(makeStructuredSurface())} agentName="My Agent" />,
    );

    fireEvent.click(screen.getByTestId('surface-action-run'));
    act(() => {
      vi.advanceTimersByTime(700);
    });

    // Sample table rows echo the column labels; markers render as chips
    // through the REAL citation path using the sample citation map.
    expect(screen.getByText('Item sample 3')).toBeInTheDocument();
    expect(screen.getByTestId('surface-table-tbl')).toBeInTheDocument();
    expect(screen.getByText(/First sample finding/)).toBeInTheDocument();
    expect(screen.getAllByTestId('citation-marker-1').length).toBeGreaterThan(0);
    expect(screen.getByText('Completed')).toBeInTheDocument();
  });
});
