import '@testing-library/jest-dom';
import { describe, it, expect, vi } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import { SurfaceRenderer } from '../SurfaceRenderer';
import type { Surface } from '@/types/surface';

// ---------------------------------------------------------------------------
// Minimal surface fixtures
// ---------------------------------------------------------------------------

function makeSurface(overrides: Partial<Surface> = {}): Surface {
  return {
    version: 1,
    components: [
      { id: 'root', component: 'Column', props: {}, children: [] },
    ],
    data_model: {},
    bindings: [],
    ...overrides,
  };
}

// ---------------------------------------------------------------------------
// TextField updates data model
// ---------------------------------------------------------------------------

describe('SurfaceRenderer — TextField', () => {
  it('renders a TextField and fires onDataModelChange when typed', () => {
    const onDataModelChange = vi.fn();
    const surface = makeSurface({
      components: [
        {
          id: 'root',
          component: 'Column',
          props: {},
          children: ['q_field'],
        },
        {
          id: 'q_field',
          component: 'TextField',
          props: { label: 'Query', value: { path: '/query' }, placeholder: 'Enter query' },
          children: [],
        },
      ],
      data_model: { query: '' },
    });

    render(
      <SurfaceRenderer
        surface={surface}
        dataModel={{ query: '' }}
        onDataModelChange={onDataModelChange}
        onAction={vi.fn()}
      />,
    );

    const input = screen.getByRole('textbox');
    fireEvent.change(input, { target: { value: 'AI safety' } });

    expect(onDataModelChange).toHaveBeenCalledOnce();
    // onDataModelChange is called as (pointer, value)
    expect(onDataModelChange).toHaveBeenCalledWith('/query', 'AI safety');
  });
});

// ---------------------------------------------------------------------------
// Button fires onAction and is disabled
// ---------------------------------------------------------------------------

describe('SurfaceRenderer — Button', () => {
  it('calls onAction with the button action name', () => {
    const onAction = vi.fn();
    const surface = makeSurface({
      components: [
        { id: 'root', component: 'Column', props: {}, children: ['btn'] },
        {
          id: 'btn',
          component: 'Button',
          props: { label: 'Run', action: 'submit', variant: 'primary' },
          children: [],
        },
      ],
    });

    render(
      <SurfaceRenderer
        surface={surface}
        dataModel={{}}
        onDataModelChange={vi.fn()}
        onAction={onAction}
      />,
    );

    fireEvent.click(screen.getByRole('button', { name: /run/i }));
    expect(onAction).toHaveBeenCalledWith('submit');
  });

  it('disables the button when actionDisabled is true', () => {
    const surface = makeSurface({
      components: [
        { id: 'root', component: 'Column', props: {}, children: ['btn'] },
        {
          id: 'btn',
          component: 'Button',
          props: { label: 'Go', action: 'go' },
          children: [],
        },
      ],
    });

    render(
      <SurfaceRenderer
        surface={surface}
        dataModel={{}}
        onDataModelChange={vi.fn()}
        onAction={vi.fn()}
        actionDisabled={true}
      />,
    );

    expect(screen.getByRole('button', { name: /go/i })).toBeDisabled();
  });
});

// ---------------------------------------------------------------------------
// Unknown component renders error chip
// ---------------------------------------------------------------------------

describe('SurfaceRenderer — unknown component', () => {
  it('shows an error chip for an unknown component name', () => {
    const surface = makeSurface({
      components: [
        { id: 'root', component: 'Column', props: {}, children: ['bad'] },
        { id: 'bad', component: 'FlyingPig', props: {}, children: [] },
      ],
    });

    render(
      <SurfaceRenderer
        surface={surface}
        dataModel={{}}
        onDataModelChange={vi.fn()}
        onAction={vi.fn()}
      />,
    );

    expect(screen.getByText(/unknown component.*flyingpig/i)).toBeInTheDocument();
  });
});

// ---------------------------------------------------------------------------
// Select two-way binding
// ---------------------------------------------------------------------------

describe('SurfaceRenderer — Select', () => {
  it('renders Select with options and reflects current value', () => {
    const surface = makeSurface({
      components: [
        { id: 'root', component: 'Column', props: {}, children: ['sel'] },
        {
          id: 'sel',
          component: 'Select',
          props: {
            label: 'Depth',
            value: { path: '/depth' },
            options: [
              { label: 'Light', value: 'light' },
              { label: 'Extended', value: 'extended' },
            ],
          },
          children: [],
        },
      ],
      data_model: { depth: 'light' },
    });

    render(
      <SurfaceRenderer
        surface={surface}
        dataModel={{ depth: 'light' }}
        onDataModelChange={vi.fn()}
        onAction={vi.fn()}
      />,
    );

    // The Radix Select trigger shows the current value label
    expect(screen.getByText('Light')).toBeInTheDocument();
  });
});

// ---------------------------------------------------------------------------
// Checkbox two-way binding
// ---------------------------------------------------------------------------

describe('SurfaceRenderer — Checkbox', () => {
  it('renders checked state from data model and fires onChange', () => {
    const onDataModelChange = vi.fn();
    const surface = makeSurface({
      components: [
        { id: 'root', component: 'Column', props: {}, children: ['chk'] },
        {
          id: 'chk',
          component: 'Checkbox',
          props: { label: 'Verify', value: { path: '/verify' } },
          children: [],
        },
      ],
      data_model: { verify: false },
    });

    render(
      <SurfaceRenderer
        surface={surface}
        dataModel={{ verify: false }}
        onDataModelChange={onDataModelChange}
        onAction={vi.fn()}
      />,
    );

    const checkbox = screen.getByRole('checkbox', { name: /verify/i });
    expect(checkbox).not.toBeChecked();

    fireEvent.click(checkbox);
    expect(onDataModelChange).toHaveBeenCalledOnce();
    // onDataModelChange is called as (pointer, value)
    expect(onDataModelChange).toHaveBeenCalledWith('/verify', true);
  });
});

// ---------------------------------------------------------------------------
// ReportRegion shows empty_text when ref is null
// ---------------------------------------------------------------------------

describe('SurfaceRenderer — ReportRegion', () => {
  it('shows empty_text when the source pointer resolves to null', () => {
    const surface = makeSurface({
      components: [
        { id: 'root', component: 'Column', props: {}, children: ['report'] },
        {
          id: 'report',
          component: 'ReportRegion',
          props: { source: { path: '/run_ref' }, empty_text: 'No results yet.' },
          children: [],
        },
      ],
      data_model: { run_ref: null },
    });

    render(
      <SurfaceRenderer
        surface={surface}
        dataModel={{ run_ref: null }}
        onDataModelChange={vi.fn()}
        onAction={vi.fn()}
      />,
    );

    expect(screen.getByText('No results yet.')).toBeInTheDocument();
  });
});

// ---------------------------------------------------------------------------
// No root → error chip
// ---------------------------------------------------------------------------

describe('SurfaceRenderer — missing root', () => {
  it('shows an error when no component has id "root"', () => {
    const surface: Surface = {
      version: 1,
      components: [{ id: 'not_root', component: 'Column', props: {}, children: [] }],
      data_model: {},
      bindings: [],
    };

    render(
      <SurfaceRenderer
        surface={surface}
        dataModel={{}}
        onDataModelChange={vi.fn()}
        onAction={vi.fn()}
      />,
    );

    expect(screen.getByText(/surface error.*root/i)).toBeInTheDocument();
  });
});

describe('SurfaceRenderer — host section rendering', () => {
  it('renders a rootIds subset and suppresses host-owned components', () => {
    const surface: Surface = {
      version: 1,
      components: [
        { id: 'root', component: 'Column', props: {}, children: ['field', 'legacy'] },
        {
          id: 'field',
          component: 'TextField',
          props: { label: 'Query', value: { path: '/query' } },
          children: [],
        },
        {
          id: 'legacy',
          component: 'Checkbox',
          props: { label: 'Verify sources', value: { path: '/options/verify_sources' } },
          children: [],
        },
      ],
      data_model: { query: '', options: { verify_sources: true } },
      bindings: [],
    };

    render(
      <SurfaceRenderer
        surface={surface}
        dataModel={surface.data_model}
        onDataModelChange={vi.fn()}
        onAction={vi.fn()}
        rootIds={['root']}
        suppressComponentIds={new Set(['legacy'])}
      />,
    );

    expect(screen.getByLabelText('Query')).toBeInTheDocument();
    expect(screen.queryByLabelText('Verify sources')).not.toBeInTheDocument();
  });
});
