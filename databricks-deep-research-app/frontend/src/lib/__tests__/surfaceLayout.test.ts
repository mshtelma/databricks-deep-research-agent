import { describe, expect, it } from 'vitest';

import {
  deriveSurfaceLayout,
  legacyRunOptionComponentIds,
  legacyRunOptionDefaults,
  surfaceInputSummary,
} from '../surfaceLayout';
import type {
  Surface,
  SurfaceComponent,
  SurfaceSectionLayout,
} from '@/types/surface';

function legacySurface(): Surface {
  return {
    version: 1,
    components: [
      { id: 'root', component: 'Column', props: {}, children: ['form_card', 'results_card'] },
      { id: 'form_card', component: 'Card', props: {}, children: ['field_query', 'options_row', 'run_button'] },
      { id: 'field_query', component: 'TextField', props: { label: 'Query', value: { path: '/form/query' } }, children: [] },
      { id: 'options_row', component: 'Row', props: {}, children: ['depth_select', 'verify_checkbox'] },
      { id: 'depth_select', component: 'Select', props: { label: 'Depth', value: { path: '/options/research_depth' } }, children: [] },
      { id: 'verify_checkbox', component: 'Checkbox', props: { label: 'Verify', value: { path: '/options/verify_sources' } }, children: [] },
      { id: 'run_button', component: 'Button', props: { label: 'Run', action: 'run' }, children: [] },
      { id: 'results_card', component: 'Card', props: {}, children: ['report'] },
      { id: 'report', component: 'ReportRegion', props: { source: { path: '/results/run' } }, children: [] },
    ],
    data_model: {
      form: { query: 'Databricks AI' },
      options: { research_depth: 'extended', verify_sources: false },
      results: { run: null },
    },
    bindings: [
      {
        action: 'run',
        kind: 'run_agent',
        inputs: { query: { path: '/form/query' } },
        options: {
          research_depth: { path: '/options/research_depth' },
          verify_sources: { path: '/options/verify_sources' },
        },
        output: { target: '/results/run', mode: 'report' },
        concurrency: 'replace',
      },
    ],
  };
}

describe('surfaceLayout helpers', () => {
  it('detects legacy run option controls and their shallow container', () => {
    expect([...legacyRunOptionComponentIds(legacySurface())].sort()).toEqual([
      'depth_select',
      'options_row',
      'verify_checkbox',
    ]);
  });

  it('reads legacy run option defaults without mutating the surface', () => {
    expect(legacyRunOptionDefaults(legacySurface())).toEqual({
      researchDepth: 'extended',
      verifySources: false,
    });
  });

  it('infers Inputs and Results sections from root children', () => {
    const layout = deriveSurfaceLayout(legacySurface());
    expect(layout.inputs.children).toEqual(['form_card']);
    expect(layout.results.children).toEqual(['results_card']);
  });

  it('summarizes non-platform input values', () => {
    expect(surfaceInputSummary(legacySurface(), legacySurface().data_model)).toBe(
      'Query: Databricks AI',
    );
  });
});

describe('deriveSurfaceLayout — explicit layout.sections', () => {
  // root → [form_card(Select+TextArea), results_tabs(Tabs→TabPane→ReportRegion)]
  const componentsAIS: SurfaceComponent[] = [
    { id: 'root', component: 'Column', props: {}, children: ['form_card', 'results_tabs'] },
    { id: 'form_card', component: 'Card', props: {}, children: ['ticker_select', 'instructions_field'] },
    { id: 'ticker_select', component: 'Select', props: { value: { path: '/form/ticker' } }, children: [] },
    { id: 'instructions_field', component: 'TextArea', props: { value: { path: '/form/query' } }, children: [] },
    { id: 'results_tabs', component: 'Tabs', props: {}, children: ['tab_report'] },
    { id: 'tab_report', component: 'TabPane', props: { label: 'Report' }, children: ['report_region'] },
    { id: 'report_region', component: 'ReportRegion', props: { source: { path: '/results/run' } }, children: [] },
  ];

  function make(
    components: SurfaceComponent[],
    sections: SurfaceSectionLayout[] | undefined,
  ): Surface {
    return {
      version: 1,
      components,
      data_model: { form: {}, results: { run: null } },
      bindings: [
        {
          action: 'run',
          kind: 'run_agent',
          inputs: { query: { path: '/form/query' } },
          options: {},
          output: { target: '/results/run', mode: 'report' },
          concurrency: 'replace',
        },
      ],
      ...(sections ? { layout: { actions: 'host_bar' as const, sections } } : {}),
    };
  }

  it('fills empty explicit sections from the component tree (the AIS bug)', () => {
    const layout = deriveSurfaceLayout(
      make(componentsAIS, [
        { id: 'form', role: 'inputs', title: 'Inputs', children: [] },
        { id: 'results', role: 'results', title: 'Results', children: [] },
      ]),
    );
    expect(layout.inputs.children).toEqual(['form_card']);
    expect(layout.results.children).toEqual(['results_tabs']);
  });

  it('preserves explicit section children when provided', () => {
    const layout = deriveSurfaceLayout(
      make(componentsAIS, [
        { id: 'form', role: 'inputs', title: 'Inputs', children: ['form_card'] },
        { id: 'results', role: 'results', title: 'Results', children: ['results_tabs'] },
      ]),
    );
    expect(layout.inputs.children).toEqual(['form_card']);
    expect(layout.results.children).toEqual(['results_tabs']);
  });

  it('preserves explicit section title and default_open through derivation', () => {
    const layout = deriveSurfaceLayout(
      make(componentsAIS, [
        { id: 'form', role: 'inputs', title: 'Custom Inputs', children: [], default_open: 'always' },
        { id: 'results', role: 'results', title: 'Results', children: [] },
      ]),
    );
    expect(layout.inputs.title).toBe('Custom Inputs');
    expect(layout.inputs.default_open).toBe('always');
    expect(layout.inputs.children).toEqual(['form_card']);
  });

  it('keeps a results-only surface empty in Inputs (no whole-tree duplication)', () => {
    const resultsOnly: SurfaceComponent[] = [
      { id: 'root', component: 'Column', props: {}, children: ['results_tabs'] },
      { id: 'results_tabs', component: 'Tabs', props: {}, children: ['tab_report'] },
      { id: 'tab_report', component: 'TabPane', props: { label: 'Report' }, children: ['report_region'] },
      { id: 'report_region', component: 'ReportRegion', props: { source: { path: '/results/run' } }, children: [] },
    ];
    const layout = deriveSurfaceLayout(
      make(resultsOnly, [
        { id: 'form', role: 'inputs', title: 'Inputs', children: [] },
        { id: 'results', role: 'results', title: 'Results', children: [] },
      ]),
    );
    expect(layout.inputs.children).toEqual([]);
    expect(layout.results.children).toEqual(['results_tabs']);
  });

  it('derives inputs when only a results section is declared', () => {
    const layout = deriveSurfaceLayout(
      make(componentsAIS, [{ id: 'results', role: 'results', title: 'Results', children: [] }]),
    );
    expect(layout.inputs.children).toEqual(['form_card']);
    expect(layout.results.children).toEqual(['results_tabs']);
  });

  it('classifies an input container that also holds a StatusBadge as Inputs', () => {
    const mixed: SurfaceComponent[] = [
      { id: 'root', component: 'Column', props: {}, children: ['form_card', 'results_tabs'] },
      { id: 'form_card', component: 'Card', props: {}, children: ['q', 'status'] },
      { id: 'q', component: 'TextArea', props: { value: { path: '/form/query' } }, children: [] },
      { id: 'status', component: 'StatusBadge', props: { source: { path: '/results/run' } }, children: [] },
      { id: 'results_tabs', component: 'Tabs', props: {}, children: ['tp'] },
      { id: 'tp', component: 'TabPane', props: { label: 'R' }, children: ['rr'] },
      { id: 'rr', component: 'ReportRegion', props: { source: { path: '/results/run' } }, children: [] },
    ];
    const layout = deriveSurfaceLayout(make(mixed, undefined));
    expect(layout.inputs.children).toContain('form_card');
    expect(layout.results.children).toEqual(['results_tabs']);
  });

  it('does not hang on a component cycle', () => {
    const cyclic: SurfaceComponent[] = [
      { id: 'root', component: 'Column', props: {}, children: ['a'] },
      { id: 'a', component: 'Card', props: {}, children: ['b'] },
      { id: 'b', component: 'Card', props: {}, children: ['a'] },
    ];
    expect(() => deriveSurfaceLayout(make(cyclic, undefined))).not.toThrow();
  });
});
