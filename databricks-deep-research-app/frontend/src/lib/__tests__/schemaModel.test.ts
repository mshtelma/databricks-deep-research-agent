import { describe, it, expect } from 'vitest';

import {
  surfaceToSchema,
  applySchemaToSurface,
  type EditableSchema,
} from '../schemaModel';
import type { Surface, SurfaceComponent } from '@/types/surface';

function surface(): Surface {
  const components: SurfaceComponent[] = [
    { id: 'root', component: 'Column', props: {}, children: ['form_card', 'results_card'] },
    { id: 'form_card', component: 'Card', props: { title: 'Run' }, children: ['field_query', 'run_button'] },
    { id: 'field_query', component: 'TextArea', props: { label: 'Query', value: { path: '/form/query' } }, children: [] },
    { id: 'run_button', component: 'Button', props: { label: 'Run', action: 'run' }, children: [] },
    { id: 'results_card', component: 'Card', props: { title: 'Results' }, children: ['comparison_table', 'findings'] },
    {
      id: 'comparison_table',
      component: 'Table',
      props: {
        source: { path: '/results/run/data/comparison' },
        columns: [
          { key: 'vendor', label: 'Vendor', type: 'string' },
          { key: 'share', label: 'Share', type: 'number' },
        ],
      },
      children: [],
    },
    { id: 'findings', component: 'KeyFindings', props: { source: { path: '/results/run/data/key_findings' } }, children: [] },
  ];
  return {
    version: 1,
    components,
    data_model: { form: { query: '' }, results: { run: null } },
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
  } as unknown as Surface;
}

function cloneSurface(value: Surface): Surface {
  return JSON.parse(JSON.stringify(value)) as Surface;
}

function surfaceWithComparisonChart(): Surface {
  const s = cloneSurface(surface());
  const results = s.components.find((c) => c.id === 'results_card')!;
  results.children.push('comparison_chart');
  s.components.push({
    id: 'comparison_chart',
    component: 'Chart',
    props: {
      source: { path: '/results/run/data/comparison' },
      kind: 'bar',
      x_key: 'vendor',
      y_keys: ['share'],
      height: 320,
    },
    children: [],
  });
  return s;
}

function surfaceWithFindingsList(): Surface {
  const s = cloneSurface(surface());
  const findings = s.components.find((c) => c.id === 'findings')!;
  findings.component = 'List';
  findings.props = {
    items: { path: '/results/run/data/key_findings' },
    ordered: true,
    empty_text: 'No findings.',
  };
  return s;
}

describe('surfaceToSchema', () => {
  it('derives action/target, inputs, and output slots', () => {
    const s = surfaceToSchema(surface());
    expect(s.action).toBe('run');
    expect(s.target).toBe('/results/run');
    expect(s.inputs).toEqual([
      { id: 'field_query', component: 'TextArea', label: 'Query', key: 'query' },
    ]);
    const names = s.slots.map((x) => x.name).sort();
    expect(names).toEqual(['comparison', 'key_findings']);
    const comparison = s.slots.find((x) => x.name === 'comparison')!;
    expect(comparison.kind).toBe('table');
    expect(comparison.columns).toEqual([
      { key: 'vendor', label: 'Vendor', type: 'string' },
      { key: 'share', label: 'Share', type: 'number' },
    ]);
    const kf = s.slots.find((x) => x.name === 'key_findings')!;
    expect(kf.kind).toBe('findings');
  });

  it('returns empty schema for a null/component-less surface', () => {
    expect(surfaceToSchema(null)).toEqual({
      action: '',
      target: '',
      runControls: {},
      actions: [],
      inputs: [],
      slots: [],
    });
  });
});

describe('applySchemaToSurface round-trip', () => {
  it('is stable: re-deriving after apply yields the same schema', () => {
    const s0 = surface();
    const schema = surfaceToSchema(s0);
    const s1 = applySchemaToSurface(s0, schema);
    const schema2 = surfaceToSchema(s1);
    expect(schema2.inputs).toEqual(schema.inputs);
    expect(schema2.slots.map((x) => ({ name: x.name, kind: x.kind, columns: x.columns })))
      .toEqual(schema.slots.map((x) => ({ name: x.name, kind: x.kind, columns: x.columns })));
    // Non-schema components (the Run button) are preserved.
    expect(s1.components.find((c) => c.id === 'run_button')).toBeTruthy();
  });

  it('round-trips run-control metadata and action labels', () => {
    const s0 = surface();
    s0.runtime_controls = {
      effort: 'show',
      sources: 'advanced',
      verify_sources: 'hide',
    };
    const schema = surfaceToSchema(s0);
    expect(schema.runControls).toMatchObject({
      effort: 'show',
      sources: 'advanced',
      verify_sources: 'hide',
    });
    expect(schema.actions).toEqual([
      {
        action: 'run',
        label: 'Run',
        target: '/results/run',
        buttonId: 'run_button',
      },
    ]);

    schema.runControls.live_search = 'locked';
    schema.actions[0] = { ...schema.actions[0]!, label: 'Analyze' };
    const s1 = applySchemaToSurface(s0, schema);
    expect(s1.runtime_controls).toMatchObject({ live_search: 'locked' });
    expect(s1.components.find((c) => c.id === 'run_button')?.props.label).toBe('Analyze');
  });

  it('adds a new output slot as a component wired under the results container', () => {
    const s0 = surface();
    const schema = surfaceToSchema(s0);
    schema.slots.push({
      name: 'risks',
      kind: 'table',
      columns: [{ key: 'risk', label: 'Risk', type: 'string' }],
      componentIds: [],
      hasChart: false,
    });
    const s1 = applySchemaToSurface(s0, schema);
    const added = s1.components.find(
      (c) =>
        c.component === 'Table' &&
        (c.props.source as { path?: string })?.path === '/results/run/data/risks',
    );
    expect(added).toBeTruthy();
    const resultsCard = s1.components.find((c) => c.id === 'results_card')!;
    expect(resultsCard.children).toContain(added!.id);
    // Re-derivable.
    expect(surfaceToSchema(s1).slots.map((x) => x.name).sort()).toEqual(
      ['comparison', 'key_findings', 'risks'],
    );
  });

  it('re-applying a schema does not duplicate a newly-created slot component', () => {
    const s0 = surface();
    const schema = surfaceToSchema(s0);
    schema.slots.push({
      name: 'risks',
      kind: 'table',
      columns: [{ key: 'risk', label: 'Risk', type: 'string' }],
      componentIds: [],
      hasChart: false,
    });
    // Commit twice against successive surfaces, as the panel does across edits.
    const s1 = applySchemaToSurface(s0, schema);
    const s2 = applySchemaToSurface(s1, schema);
    const risksComps = s2.components.filter(
      (c) => (c.props.source as { path?: string })?.path === '/results/run/data/risks',
    );
    expect(risksComps).toHaveLength(1);
  });

  it('removes a dropped slot and its child reference', () => {
    const s0 = surface();
    const schema = surfaceToSchema(s0);
    schema.slots = schema.slots.filter((x) => x.name !== 'key_findings');
    const s1 = applySchemaToSurface(s0, schema);
    expect(s1.components.find((c) => c.id === 'findings')).toBeUndefined();
    const resultsCard = s1.components.find((c) => c.id === 'results_card')!;
    expect(resultsCard.children).not.toContain('findings');
    expect(surfaceToSchema(s1).slots.map((x) => x.name).sort()).toEqual(['comparison']);
  });

  it('edits table columns in place', () => {
    const s0 = surface();
    const schema = surfaceToSchema(s0);
    const comparison = schema.slots.find((x) => x.name === 'comparison')!;
    comparison.columns = [{ key: 'vendor', label: 'Company', type: 'string' }];
    const s1 = applySchemaToSurface(s0, schema);
    const table = s1.components.find((c) => c.id === 'comparison_table')!;
    expect(table.props.columns).toEqual([{ key: 'vendor', label: 'Company', type: 'string' }]);
  });

  it('retargets Table and Chart pointers when a shared slot is renamed', () => {
    const s0 = surfaceWithComparisonChart();
    const schema = surfaceToSchema(s0);
    const comparison = schema.slots.find((x) => x.name === 'comparison')!;
    comparison.name = 'vendors';

    const s1 = applySchemaToSurface(s0, schema);
    const table = s1.components.find((c) => c.id === 'comparison_table')!;
    const chart = s1.components.find((c) => c.id === 'comparison_chart')!;
    const resultsCard = s1.components.find((c) => c.id === 'results_card')!;

    expect((table.props.source as { path?: string }).path).toBe(
      '/results/run/data/vendors',
    );
    expect((chart.props.source as { path?: string }).path).toBe(
      '/results/run/data/vendors',
    );
    expect(chart.props.kind).toBe('bar');
    expect(chart.props.x_key).toBe('vendor');
    expect(chart.props.y_keys).toEqual(['share']);
    expect(resultsCard.children).toContain('comparison_chart');
    expect(
      s1.components.filter(
        (c) => (c.props.source as { path?: string } | undefined)?.path === '/results/run/data/vendors',
      ),
    ).toHaveLength(2);
  });

  it('preserves a List findings renderer and retargets its items pointer', () => {
    const s0 = surfaceWithFindingsList();
    const schema = surfaceToSchema(s0);
    const findings = schema.slots.find((x) => x.name === 'key_findings')!;
    findings.name = 'takeaways';

    const s1 = applySchemaToSurface(s0, schema);
    const list = s1.components.find((c) => c.id === 'findings')!;

    expect(list.component).toBe('List');
    expect((list.props.items as { path?: string }).path).toBe(
      '/results/run/data/takeaways',
    );
    expect(list.props.ordered).toBe(true);
    expect(list.props.empty_text).toBe('No findings.');
    expect(list.props.source).toBeUndefined();
  });

  it('removes a dropped input and its data_model + binding wiring', () => {
    const s0 = surface();
    const schema: EditableSchema = surfaceToSchema(s0);
    schema.inputs = [];
    const s1 = applySchemaToSurface(s0, schema);
    expect(s1.components.find((c) => c.id === 'field_query')).toBeUndefined();
    expect((s1.data_model as { form?: Record<string, unknown> }).form).toEqual({});
    expect(s1.bindings[0]!.inputs).toEqual({});
  });
});
