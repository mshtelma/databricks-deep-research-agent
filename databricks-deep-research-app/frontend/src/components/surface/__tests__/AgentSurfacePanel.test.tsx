import '@testing-library/jest-dom';
import { describe, it, expect, vi } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { AgentSurfacePanel } from '../AgentSurfacePanel';
import type { Surface, RunReference } from '@/types/surface';

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------

/** Minimal surface: TextField for /query + Button action "run". */
function makeSurface(withReportRegion = false): Surface {
  const components: Surface['components'] = [
    {
      id: 'root',
      component: 'Column',
      props: {},
      children: withReportRegion
        ? ['q_field', 'run_btn', 'results']
        : ['q_field', 'run_btn'],
    },
    {
      id: 'q_field',
      component: 'TextField',
      props: { label: 'Query', value: { path: '/query' }, placeholder: 'Enter topic' },
      children: [],
    },
    {
      id: 'run_btn',
      // catalog name is "Button", not "RunButton"
      component: 'Button',
      props: { label: 'Run', action: 'run' },
      children: [],
    },
  ];

  if (withReportRegion) {
    components.push({
      id: 'results',
      component: 'ReportRegion',
      props: { source: { path: '/result' }, empty_text: 'No results yet.' },
      children: [],
    });
  }

  return {
    version: 1,
    components,
    data_model: { query: '' },
    bindings: [
      {
        action: 'run',
        kind: 'run_agent',
        inputs: { query: { path: '/query' } },
        options: {},
        output: { target: '/result', mode: 'report' },
        concurrency: 'replace',
      },
    ],
  };
}

/** "Pick-or-custom" surface: the query is bound to a Select (empty by default)
 *  with a sibling free-text "Custom ticker" field — the shape that used to block. */
function makeCustomSurface(): Surface {
  return {
    version: 1,
    components: [
      {
        id: 'root',
        component: 'Column',
        props: {},
        children: ['ticker_select', 'custom_input', 'run_btn'],
      },
      {
        id: 'ticker_select',
        component: 'Select',
        props: { label: 'Ticker', value: { path: '/inputs/ticker' }, options: [] },
        children: [],
      },
      {
        id: 'custom_input',
        component: 'TextField',
        props: { label: 'Custom', value: { path: '/inputs/custom' }, placeholder: 'Custom ticker' },
        children: [],
      },
      { id: 'run_btn', component: 'Button', props: { label: 'Run', action: 'run' }, children: [] },
    ],
    data_model: { inputs: { ticker: '', custom: '' } },
    bindings: [
      {
        action: 'run',
        kind: 'run_agent',
        inputs: { query: { path: '/inputs/ticker' }, custom_query: { path: '/inputs/custom' } },
        options: {},
        output: { target: '/results/run', mode: 'report' },
        concurrency: 'replace',
      },
    ],
  };
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe('AgentSurfacePanel', () => {
  it('renders the agent name and surface components', () => {
    render(
      <AgentSurfacePanel
        agentName="My Research Agent"
        surface={makeSurface()}
        onRunAction={vi.fn()}
        runDisabled={false}
        runState={{}}
      />,
    );

    expect(screen.getByText('My Research Agent')).toBeInTheDocument();
    expect(screen.getByText('Agent UI')).toBeInTheDocument();
    expect(screen.getByPlaceholderText('Enter topic')).toBeInTheDocument();
  });

  it('renders host-owned surface run controls', () => {
    render(
      <AgentSurfacePanel
        agentName="Agent"
        surface={makeSurface()}
        selectedAgentId="agent-1"
        onRunAction={vi.fn()}
        runDisabled={false}
        runState={{}}
      />,
    );

    expect(screen.getByTestId('surface-run-controls')).toBeInTheDocument();
    expect(screen.getByTestId('surface-effort-chip')).toHaveTextContent('Effort');
    expect(screen.getByTestId('surface-sources-chip')).toHaveTextContent('Sources');
    expect(screen.getByRole('button', { name: /options/i })).toBeInTheDocument();
  });

  it('shows inline error when Run is clicked with empty query', () => {
    const onRunAction = vi.fn();
    render(
      <AgentSurfacePanel
        agentName="Agent"
        surface={makeSurface()}
        onRunAction={onRunAction}
        runDisabled={false}
        runState={{}}
      />,
    );

    // Click Run without filling the field
    const runButton = screen.getByRole('button', { name: 'Run' });
    fireEvent.click(runButton);

    expect(
      screen.getByText('Enter your request or fill at least one field to run.'),
    ).toBeInTheDocument();
    expect(onRunAction).not.toHaveBeenCalled();
  });

  it('calls onRunAction with compiled query and surfaceInputs when query is filled', () => {
    const onRunAction = vi.fn();
    render(
      <AgentSurfacePanel
        agentName="Agent"
        surface={makeSurface()}
        onRunAction={onRunAction}
        runDisabled={false}
        runState={{}}
      />,
    );

    const input = screen.getByPlaceholderText('Enter topic');
    fireEvent.change(input, { target: { value: 'AI safety research' } });

    const runButton = screen.getByRole('button', { name: 'Run' });
    fireEvent.click(runButton);

    expect(onRunAction).toHaveBeenCalledOnce();
    const [compiled] = onRunAction.mock.calls[0] as [{ query: string; surfaceInputs: Record<string, unknown> }];
    expect(compiled.query).toBe('AI safety research');
    expect(typeof compiled.surfaceInputs).toBe('object');
    expect(compiled).toMatchObject({
      submission: {
        message: 'AI safety research',
        queryMode: 'deep_research',
        turnIntent: 'research',
      },
    });
  });

  it('sends selected effort in the full surface submission', () => {
    const onRunAction = vi.fn();
    render(
      <AgentSurfacePanel
        agentName="Agent"
        surface={makeSurface()}
        selectedAgentId="agent-1"
        onRunAction={onRunAction}
        runDisabled={false}
        runState={{}}
      />,
    );

    fireEvent.click(screen.getByTestId('surface-effort-chip'));
    fireEvent.click(screen.getByRole('button', { name: 'Deep' }));
    fireEvent.change(screen.getByPlaceholderText('Enter topic'), {
      target: { value: 'Lakehouse performance' },
    });
    fireEvent.click(screen.getByRole('button', { name: 'Run' }));

    const [compiled] = onRunAction.mock.calls[0] as [
      { submission: { researchDepth?: string; agentId?: string } },
    ];
    expect(compiled.submission.researchDepth).toBe('extended');
    expect(compiled.submission.agentId).toBe('agent-1');
  });

  it('includes disabled source ids in the full surface submission', () => {
    const onRunAction = vi.fn();
    render(
      <AgentSurfacePanel
        agentName="Agent"
        surface={makeSurface()}
        selectedAgentId="agent-1"
        availableSources={[
          {
            id: 'web',
            name: 'Web',
            type: 'web_search',
            description: null,
            isEnabled: true,
          },
          {
            id: 'docs',
            name: 'Docs',
            type: 'vector_search',
            description: null,
            isEnabled: false,
          },
        ]}
        disabledSourceIds={['docs']}
        onRunAction={onRunAction}
        runDisabled={false}
        runState={{}}
      />,
    );

    fireEvent.change(screen.getByPlaceholderText('Enter topic'), {
      target: { value: 'Lakebase docs' },
    });
    fireEvent.click(screen.getByRole('button', { name: 'Run' }));

    const [compiled] = onRunAction.mock.calls[0] as [
      { submission: { enabledSources?: string[]; disabledSources?: string[] } },
    ];
    expect(compiled.submission.enabledSources).toEqual(['web']);
    expect(compiled.submission.disabledSources).toEqual(['docs']);
  });

  it('omits source routing when the selected agent owns its source config', () => {
    const onRunAction = vi.fn();
    render(
      <AgentSurfacePanel
        agentName="Agent"
        surface={makeSurface()}
        selectedAgentId="agent-1"
        availableSources={[
          {
            id: 'web',
            name: 'Web',
            type: 'web_search',
            description: null,
            isEnabled: true,
          },
        ]}
        disabledSourceIds={['web']}
        agentDefinesSources
        onRunAction={onRunAction}
        runDisabled={false}
        runState={{}}
      />,
    );

    expect(screen.queryByTestId('surface-sources-chip')).not.toBeInTheDocument();
    fireEvent.change(screen.getByPlaceholderText('Enter topic'), {
      target: { value: 'Use agent defaults' },
    });
    fireEvent.click(screen.getByRole('button', { name: 'Run' }));

    const [compiled] = onRunAction.mock.calls[0] as [
      {
        submission: {
          sourceScope?: string;
          enabledSources?: string[];
          disabledSources?: string[];
        };
      },
    ];
    expect(compiled.submission.sourceScope).toBeUndefined();
    expect(compiled.submission.enabledSources).toBeUndefined();
    expect(compiled.submission.disabledSources).toBeUndefined();
  });

  it('enforces runtime_controls hide and locked policies', () => {
    const surface: Surface = {
      ...makeSurface(),
      runtime_controls: {
        effort: 'locked',
        sources: 'hide',
        verify_sources: 'locked',
        plan_review: 'hide',
        report_style: 'hide',
        cross_session_memory: 'hide',
        live_search: 'hide',
      },
    };
    render(
      <AgentSurfacePanel
        agentName="Agent"
        surface={surface}
        onRunAction={vi.fn()}
        runDisabled={false}
        runState={{}}
      />,
    );

    expect(screen.queryByTestId('surface-sources-chip')).not.toBeInTheDocument();
    expect(screen.getByTestId('surface-effort-chip')).toBeDisabled();
    fireEvent.click(screen.getByRole('button', { name: /options/i }));
    expect(screen.getByLabelText(/verify citations/i)).toBeDisabled();
    expect(screen.queryByText(/review plan before research/i)).not.toBeInTheDocument();
    expect(screen.queryByText(/recall facts from my prior chats/i)).not.toBeInTheDocument();
  });

  it('blocks invalid enterprise-only source states with an actionable message', async () => {
    render(
      <AgentSurfacePanel
        agentName="Agent"
        surface={makeSurface()}
        availableSources={[]}
        onRunAction={vi.fn()}
        runDisabled={false}
        runState={{}}
      />,
    );

    fireEvent.click(screen.getByTestId('surface-sources-chip'));
    fireEvent.click(screen.getByTestId('surface-source-web'));
    fireEvent.click(screen.getByTestId('surface-source-mcp'));

    await waitFor(() => {
      expect(screen.getByTestId('surface-source-validation')).toHaveTextContent(
        'No enterprise data sources available',
      );
    });
    expect(screen.getByRole('button', { name: 'Run' })).toBeDisabled();
  });

  it('disables the Run button when runDisabled is true', () => {
    render(
      <AgentSurfacePanel
        agentName="Agent"
        surface={makeSurface()}
        onRunAction={vi.fn()}
        runDisabled={true}
        runState={{}}
      />,
    );

    const runButton = screen.getByRole('button', { name: 'Run' });
    expect(runButton).toBeDisabled();
    expect(screen.getByText('A run is already in progress')).toBeInTheDocument();
  });

  it('calls resolveRunReference when runState has a completed ref', () => {
    const completedRef: RunReference = {
      status: 'completed',
      message_id: 'msg-1',
      session_id: 'sess-1',
    };
    const resolveRunReference = vi.fn(() => <span>Report loaded</span>);

    // Use a surface with a ReportRegion so resolveRunReference gets invoked
    const surface = makeSurface(true);
    render(
      <AgentSurfacePanel
        agentName="Agent"
        surface={surface}
        onRunAction={vi.fn()}
        runDisabled={false}
        runState={{ run: completedRef }}
        resolveRunReference={resolveRunReference}
      />,
    );

    // The ReportRegion reads /result from the overlay data model and calls resolveRunReference
    expect(resolveRunReference).toHaveBeenCalledWith(completedRef);
    expect(screen.getByText('Report loaded')).toBeInTheDocument();
  });

  it('calls onClose when the close button is clicked', () => {
    const onClose = vi.fn();
    render(
      <AgentSurfacePanel
        agentName="Agent"
        surface={makeSurface()}
        onRunAction={vi.fn()}
        runDisabled={false}
        runState={{}}
        onClose={onClose}
      />,
    );

    const closeBtn = screen.getByRole('button', { name: /close agent ui panel/i });
    fireEvent.click(closeBtn);

    expect(onClose).toHaveBeenCalledOnce();
  });

  it('seeds the form with initialDataModel when provided and non-empty', () => {
    const surface = makeSurface();
    render(
      <AgentSurfacePanel
        agentName="Agent"
        surface={surface}
        onRunAction={vi.fn()}
        runDisabled={false}
        runState={{}}
        initialDataModel={{ query: 'pre-filled topic' }}
      />,
    );

    // The TextField should show the pre-filled value from initialDataModel.
    const input = screen.getByPlaceholderText('Enter topic') as HTMLInputElement;
    expect(input.value).toBe('pre-filled topic');
  });

  it('adopts late initialDataModel when the current form is still clean', () => {
    const surface = makeSurface();
    const { rerender } = render(
      <AgentSurfacePanel
        agentName="Agent"
        surface={surface}
        surfaceIdentity="agent-a"
        onRunAction={vi.fn()}
        runDisabled={false}
        runState={{}}
        initialDataModel={{ query: 'old persisted topic' }}
      />,
    );

    rerender(
      <AgentSurfacePanel
        agentName="Agent"
        surface={surface}
        surfaceIdentity="agent-a"
        onRunAction={vi.fn()}
        runDisabled={false}
        runState={{}}
        initialDataModel={{ query: 'new persisted topic' }}
      />,
    );

    const input = screen.getByPlaceholderText('Enter topic') as HTMLInputElement;
    expect(input.value).toBe('new persisted topic');
  });

  it('does not overwrite user edits when initialDataModel changes for the same surface', () => {
    const surface = makeSurface();
    const { rerender } = render(
      <AgentSurfacePanel
        agentName="Agent"
        surface={surface}
        surfaceIdentity="agent-a"
        onRunAction={vi.fn()}
        runDisabled={false}
        runState={{}}
        initialDataModel={{ query: 'persisted topic' }}
      />,
    );

    const input = screen.getByPlaceholderText('Enter topic') as HTMLInputElement;
    fireEvent.change(input, { target: { value: 'typed topic' } });

    rerender(
      <AgentSurfacePanel
        agentName="Agent"
        surface={surface}
        surfaceIdentity="agent-a"
        onRunAction={vi.fn()}
        runDisabled={false}
        runState={{}}
        initialDataModel={{ query: 'late persisted topic' }}
      />,
    );

    expect(input.value).toBe('typed topic');
  });

  it('resets dirty form state when the surface identity changes', () => {
    const surface = makeSurface();
    const { rerender } = render(
      <AgentSurfacePanel
        agentName="Agent"
        surface={surface}
        surfaceIdentity="agent-a"
        onRunAction={vi.fn()}
        runDisabled={false}
        runState={{}}
        initialDataModel={{ query: 'agent a topic' }}
      />,
    );

    const input = screen.getByPlaceholderText('Enter topic') as HTMLInputElement;
    fireEvent.change(input, { target: { value: 'dirty agent a edit' } });

    rerender(
      <AgentSurfacePanel
        agentName="Agent"
        surface={surface}
        surfaceIdentity="agent-b"
        onRunAction={vi.fn()}
        runDisabled={false}
        runState={{}}
        initialDataModel={{ query: 'agent b topic' }}
      />,
    );

    expect(input.value).toBe('agent b topic');
  });

  it('preserves active edits when only runState changes', () => {
    const surface = makeSurface();
    const { rerender } = render(
      <AgentSurfacePanel
        agentName="Agent"
        surface={surface}
        surfaceIdentity="agent-a"
        onRunAction={vi.fn()}
        runDisabled={false}
        runState={{}}
      />,
    );

    const input = screen.getByPlaceholderText('Enter topic') as HTMLInputElement;
    fireEvent.change(input, { target: { value: 'still editing' } });

    rerender(
      <AgentSurfacePanel
        agentName="Agent"
        surface={surface}
        surfaceIdentity="agent-a"
        onRunAction={vi.fn()}
        runDisabled={false}
        runState={{ run: { status: 'running' } }}
      />,
    );

    expect(input.value).toBe('still editing');
  });

  it('collapses Inputs and expands Results when a run starts', async () => {
    const surface = makeSurface(true);
    const { rerender } = render(
      <AgentSurfacePanel
        agentName="Agent"
        surface={surface}
        onRunAction={vi.fn()}
        runDisabled={false}
        runState={{}}
      />,
    );

    expect(screen.getByPlaceholderText('Enter topic')).toBeInTheDocument();

    rerender(
      <AgentSurfacePanel
        agentName="Agent"
        surface={surface}
        onRunAction={vi.fn()}
        runDisabled={true}
        runState={{ run: { status: 'running' } }}
      />,
    );

    await waitFor(() => {
      expect(screen.queryByPlaceholderText('Enter topic')).not.toBeInTheDocument();
    });
    expect(screen.getByText(/Running/i)).toBeInTheDocument();
  });

  it.each(['completed', 'failed'] as const)(
    'collapses Inputs and expands Results when a %s run is rehydrated',
    async (status) => {
      const surface = makeSurface(true);
      render(
        <AgentSurfacePanel
          agentName="Agent"
          surface={surface}
          onRunAction={vi.fn()}
          runDisabled={false}
          runState={{ run: { status } }}
          resolveRunReference={() => <span>{status} result</span>}
        />,
      );

      await waitFor(() => {
        expect(screen.queryByPlaceholderText('Enter topic')).not.toBeInTheDocument();
      });
      expect(screen.getByText(`${status} result`)).toBeInTheDocument();
    },
  );

  it('calls onFormStateChange with the new data model after a user edit', () => {
    const onFormStateChange = vi.fn();
    render(
      <AgentSurfacePanel
        agentName="Agent"
        surface={makeSurface()}
        onRunAction={vi.fn()}
        runDisabled={false}
        runState={{}}
        onFormStateChange={onFormStateChange}
      />,
    );

    const input = screen.getByPlaceholderText('Enter topic');
    fireEvent.change(input, { target: { value: 'new topic' } });

    expect(onFormStateChange).toHaveBeenCalledOnce();
    const [model] = onFormStateChange.mock.calls[0] as [Record<string, unknown>];
    expect(model).toMatchObject({ query: 'new topic' });
  });

  it('runs with a composed query when the query-bound dropdown is empty but a free-text field is filled', () => {
    const onRunAction = vi.fn();
    render(
      <AgentSurfacePanel
        agentName="Agent"
        surface={makeCustomSurface()}
        onRunAction={onRunAction}
        runDisabled={false}
        runState={{}}
      />,
    );

    fireEvent.change(screen.getByPlaceholderText('Custom ticker'), {
      target: { value: 'Tesla' },
    });
    fireEvent.click(screen.getByRole('button', { name: 'Run' }));

    expect(onRunAction).toHaveBeenCalledOnce();
    const [compiled] = onRunAction.mock.calls[0] as [
      { query: string; querySource?: string; surfaceInputs: Record<string, unknown> },
    ];
    expect(compiled.query).toBe('Tesla');
    expect(compiled.querySource).toBe('composed');
    expect(compiled.surfaceInputs.custom_query).toBe('Tesla');
  });

  it('blocks with an inline message only when the whole form is empty', () => {
    const onRunAction = vi.fn();
    render(
      <AgentSurfacePanel
        agentName="Agent"
        surface={makeCustomSurface()}
        onRunAction={onRunAction}
        runDisabled={false}
        runState={{}}
      />,
    );

    fireEvent.click(screen.getByRole('button', { name: 'Run' }));

    expect(
      screen.getByText('Enter your request or fill at least one field to run.'),
    ).toBeInTheDocument();
    expect(onRunAction).not.toHaveBeenCalled();
  });
});
