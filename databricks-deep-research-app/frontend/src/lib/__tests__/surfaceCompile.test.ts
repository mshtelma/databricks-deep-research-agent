import { describe, it, expect } from 'vitest';
import { compileBinding, compileSurfaceAction, deriveEffectiveQuery } from '../surfaceCompile';
import type { ActionBinding, Surface, SurfaceComponent } from '@/types/surface';

function makeSurface(comps: Array<Partial<SurfaceComponent>>): Surface {
  return {
    version: 1,
    components: comps.map((c) => ({
      id: c.id ?? 'c',
      component: c.component ?? 'TextField',
      props: c.props ?? {},
      children: c.children ?? [],
    })),
    data_model: {},
    bindings: [],
  };
}

function makeBinding(overrides: Partial<ActionBinding> = {}): ActionBinding {
  return {
    action: 'run',
    kind: 'run_agent',
    inputs: {},
    options: {},
    output: { target: '/result', mode: 'report' },
    concurrency: 'replace',
    ...overrides,
  };
}

describe('compileBinding — query', () => {
  it('resolves a PathRef query', () => {
    const binding = makeBinding({ inputs: { query: { path: '/topic' } } });
    const { query } = compileBinding(binding, { topic: 'AI safety' });
    expect(query).toBe('AI safety');
  });

  it('returns empty string when PathRef points to missing key', () => {
    const binding = makeBinding({ inputs: { query: { path: '/missing' } } });
    const { query } = compileBinding(binding, {});
    expect(query).toBe('');
  });

  it('substitutes {/pointer} placeholders in a string query', () => {
    const binding = makeBinding({
      inputs: { query: 'Research {/topic} in {/year}' },
    });
    const { query } = compileBinding(binding, { topic: 'ML', year: 2026 });
    expect(query).toBe('Research ML in 2026');
  });

  it('substitutes missing pointer to empty string', () => {
    const binding = makeBinding({
      inputs: { query: 'Hello {/missing} world' },
    });
    const { query } = compileBinding(binding, {});
    expect(query).toBe('Hello  world');
  });

  it('does not substitute plain {word} braces', () => {
    const binding = makeBinding({ inputs: { query: 'Lookup {topic} not {/ptr}' } });
    const { query } = compileBinding(binding, { ptr: 'X' });
    expect(query).toBe('Lookup {topic} not X');
  });

  it('returns empty string when query is absent', () => {
    const binding = makeBinding({ inputs: {} });
    const { query } = compileBinding(binding, {});
    expect(query).toBe('');
  });
});

describe('compileBinding — surfaceInputs', () => {
  it('includes a simple string input', () => {
    const binding = makeBinding({
      inputs: { query: 'q', region: 'us-west' },
    });
    const { surfaceInputs } = compileBinding(binding, {});
    expect(surfaceInputs['region']).toBe('us-west');
  });

  it('resolves a PathRef input', () => {
    const binding = makeBinding({
      inputs: { query: 'q', depth: { path: '/config/depth' } },
    });
    const { surfaceInputs } = compileBinding(binding, { config: { depth: 'extended' } });
    expect(surfaceInputs['depth']).toBe('extended');
  });

  it('skips null resolved values', () => {
    const binding = makeBinding({
      inputs: { query: 'q', gone: null },
    });
    const { surfaceInputs } = compileBinding(binding, {});
    expect('gone' in surfaceInputs).toBe(false);
  });

  it('skips reserved keys', () => {
    const binding = makeBinding({
      inputs: { query: 'q', plan: 'step1', tool_catalog: 'x', current_date: 'd' },
    });
    const { surfaceInputs } = compileBinding(binding, {});
    expect('plan' in surfaceInputs).toBe(false);
    expect('tool_catalog' in surfaceInputs).toBe(false);
    expect('current_date' in surfaceInputs).toBe(false);
  });

  it('skips non-identifier keys', () => {
    const binding = makeBinding({
      inputs: {
        query: 'q',
        'bad-key': 'v1',
        '1bad': 'v2',
        'also bad': 'v3',
      },
    });
    const { surfaceInputs } = compileBinding(binding, {});
    expect('bad-key' in surfaceInputs).toBe(false);
    expect('1bad' in surfaceInputs).toBe(false);
    expect('also bad' in surfaceInputs).toBe(false);
  });

  it('skips undefined resolved values', () => {
    const binding = makeBinding({
      inputs: { query: 'q', missing: { path: '/nowhere' } },
    });
    const { surfaceInputs } = compileBinding(binding, {});
    expect('missing' in surfaceInputs).toBe(false);
  });

  it('includes numeric and boolean inputs', () => {
    const binding = makeBinding({
      inputs: { query: 'q', count: 3, flag: true },
    });
    const { surfaceInputs } = compileBinding(binding, {});
    expect(surfaceInputs['count']).toBe(3);
    expect(surfaceInputs['flag']).toBe(true);
  });
});

describe('compileBinding — options', () => {
  it('maps research_depth string option', () => {
    const binding = makeBinding({
      inputs: {},
      options: { research_depth: 'extended' },
    });
    expect(compileBinding(binding, {}).researchDepth).toBe('extended');
  });

  it('resolves research_depth PathRef', () => {
    const binding = makeBinding({
      inputs: {},
      options: { research_depth: { path: '/depth' } },
    });
    expect(compileBinding(binding, { depth: 'light' }).researchDepth).toBe('light');
  });

  it('omits research_depth when empty string', () => {
    const binding = makeBinding({ inputs: {}, options: { research_depth: '' } });
    expect(compileBinding(binding, {}).researchDepth).toBeUndefined();
  });

  it('omits research_depth when null', () => {
    const binding = makeBinding({ inputs: {}, options: { research_depth: null } });
    expect(compileBinding(binding, {}).researchDepth).toBeUndefined();
  });

  it('maps verify_sources boolean option', () => {
    const binding = makeBinding({
      inputs: {},
      options: { verify_sources: true },
    });
    expect(compileBinding(binding, {}).verifySources).toBe(true);
  });

  it('omits verify_sources when null', () => {
    const binding = makeBinding({ inputs: {}, options: { verify_sources: null } });
    expect(compileBinding(binding, {}).verifySources).toBeUndefined();
  });
});

describe('compileSurfaceAction — full QuerySubmission', () => {
  it('merges host run context and action overrides into a full submission', () => {
    const surface = makeSurface([
      {
        id: 'field_query',
        component: 'TextField',
        props: { label: 'Query', value: { path: '/inputs/query' } },
      },
    ]);
    const binding = makeBinding({
      inputs: {
        query: { path: '/inputs/query' },
        ticker: { path: '/inputs/ticker' },
      },
      options: {
        research_depth: 'extended',
        verify_sources: false,
        tone: 'concise',
        allow_live_search: true,
      },
    });

    const compiled = compileSurfaceAction({
      surface,
      binding,
      dataModel: { inputs: { query: 'Research NVDA', ticker: 'NVDA' } },
      selectedAgentId: 'agent-1',
      runContext: {
        queryMode: 'deep_research',
        researchDepth: 'light',
        verifySources: true,
        sourceScope: 'all',
        enabledSources: ['web', 'mcp:tavily'],
        enabledMcpServers: ['tavily'],
        enablePlanReview: true,
        enableCrossSessionMemory: false,
        allowLiveSearch: false,
        outputLanguage: 'Spanish',
      },
    });

    expect(compiled.submission).toMatchObject({
      message: 'Research NVDA',
      queryMode: 'deep_research',
      agentId: 'agent-1',
      surfaceAction: 'run',
      surfaceInputs: { ticker: 'NVDA' },
      researchDepth: 'extended',
      verifySources: false,
      sourceScope: 'all',
      enabledSources: ['web', 'mcp:tavily'],
      enabledMcpServers: ['tavily'],
      enablePlanReview: true,
      enableCrossSessionMemory: false,
      allowLiveSearch: true,
      tone: 'concise',
      outputLanguage: 'Spanish',
      turnIntent: 'research',
    });
  });

  it('defaults surface actions to deep research and research turn intent', () => {
    const binding = makeBinding({ inputs: { query: 'Investigate Lakebase' } });
    const compiled = compileSurfaceAction({
      surface: makeSurface([]),
      binding,
      dataModel: {},
      runContext: { verifySources: false },
    });

    expect(compiled.submission.queryMode).toBe('deep_research');
    expect(compiled.submission.turnIntent).toBe('research');
    expect(compiled.submission.verifySources).toBe(false);
  });
});

describe('deriveEffectiveQuery — tolerant composition', () => {
  it('uses the bound query when non-empty (source=bound)', () => {
    const surface = makeSurface([
      { id: 'f', component: 'TextField', props: { value: { path: '/inputs/ticker' } } },
    ]);
    const binding = makeBinding({ inputs: { query: { path: '/inputs/ticker' } } });
    expect(deriveEffectiveQuery(binding, { inputs: { ticker: 'AAPL' } }, surface)).toMatchObject({
      query: 'AAPL',
      source: 'bound',
    });
  });

  it('composes from a filled free-text input when the bound query is empty', () => {
    const surface = makeSurface([
      { id: 'sel', component: 'Select', props: { value: { path: '/inputs/ticker' } } },
      { id: 'custom', component: 'TextField', props: { value: { path: '/inputs/custom' } } },
    ]);
    const binding = makeBinding({ inputs: { query: { path: '/inputs/ticker' } } });
    const eq = deriveEffectiveQuery(binding, { inputs: { ticker: '', custom: 'TSLA' } }, surface);
    expect(eq).toMatchObject({ query: 'TSLA', source: 'composed' });
    expect(eq.usedPointers).toEqual(['/inputs/custom']);
  });

  it('labels + newline-joins multiple free-text contributors in document order', () => {
    const surface = makeSurface([
      { id: 'a', component: 'TextField', props: { label: 'Company', value: { path: '/inputs/a' } } },
      { id: 'b', component: 'TextArea', props: { label: 'Question', value: { path: '/inputs/b' } } },
    ]);
    const binding = makeBinding({ inputs: { query: { path: '/inputs/missing' } } });
    const eq = deriveEffectiveQuery(binding, { inputs: { a: 'X', b: 'Y' } }, surface);
    expect(eq.source).toBe('composed');
    expect(eq.query).toBe('Company: X\nQuestion: Y');
  });

  it('excludes the bound-query pointer from composition', () => {
    const surface = makeSurface([
      { id: 'q', component: 'TextArea', props: { value: { path: '/inputs/q' } } },
    ]);
    const binding = makeBinding({ inputs: { query: { path: '/inputs/q' } } });
    const eq = deriveEffectiveQuery(binding, { inputs: { q: '' } }, surface);
    expect(eq.source).toBe('empty');
    expect(eq.query).toBe('');
  });

  it('composes from a chosen Select value (the subject) but never a Checkbox', () => {
    const surface = makeSurface([
      { id: 'ticker', component: 'Select', props: { label: 'Ticker', value: { path: '/inputs/ticker' } } },
      { id: 'verify', component: 'Checkbox', props: { label: 'Verify', value: { path: '/inputs/verify' } } },
    ]);
    const binding = makeBinding({ inputs: {} }); // no bound query
    const eq = deriveEffectiveQuery(
      binding,
      { inputs: { ticker: 'NVDA', verify: true } },
      surface,
    );
    // Select value becomes the query; the Checkbox boolean stays a filter.
    expect(eq).toMatchObject({ query: 'NVDA', source: 'composed' });
    expect(eq.query).not.toContain('true');
  });

  it('runs a Select-primary agent authored with bare-string pointers + no query key (IRA regression)', () => {
    const surface = makeSurface([
      { id: 'ticker', component: 'Select', props: { label: 'Ticker', value: { path: '/inputs/ticker' } } },
      { id: 'notes', component: 'TextArea',
        props: { label: 'Additional Instructions', value: { path: '/inputs/additional_instructions' } } },
    ]);
    // Bare-string pointers (not {path}) and NO `query` key — the exact deployed shape.
    const binding = makeBinding({
      inputs: {
        ticker: '/inputs/ticker',
        additional_instructions: '/inputs/additional_instructions',
      },
    });

    // Ticker picked, instructions empty (the reported failing case) → composes from the Select.
    const a = compileBinding(binding, { inputs: { ticker: 'NVDA', additional_instructions: '' } }, surface);
    expect(a.query).toBe('NVDA');
    expect(a.querySource).toBe('composed');
    expect(a.surfaceInputs['ticker']).toBe('NVDA'); // bare-string pointer resolved, not the literal

    // Both filled → labeled query + resolved prompt vars.
    const b = compileBinding(
      binding,
      { inputs: { ticker: 'NVDA', additional_instructions: 'focus on earnings' } },
      surface,
    );
    expect(b.query).toBe('Ticker: NVDA\nAdditional Instructions: focus on earnings');
    expect(b.surfaceInputs['additional_instructions']).toBe('focus on earnings');

    // Nothing filled → still blocks gracefully.
    const c = compileBinding(binding, { inputs: { ticker: '', additional_instructions: '' } }, surface);
    expect(c.query).toBe('');
    expect(c.querySource).toBe('empty');
  });

  it('resolves a bare-string pointer used directly as the query (authored without {path})', () => {
    const binding = makeBinding({ inputs: { query: '/inputs/topic' } });
    const compiled = compileBinding(binding, { inputs: { topic: 'quantum computing' } });
    expect(compiled.query).toBe('quantum computing');
    expect(compiled.querySource).toBe('bound');
  });

  it('composes for an empty pure-pointer {/ptr} template query (skips exclusion)', () => {
    const surface = makeSurface([
      { id: 'c', component: 'TextField', props: { value: { path: '/inputs/custom' } } },
    ]);
    const binding = makeBinding({ inputs: { query: '{/inputs/topic}' } }); // resolves empty
    expect(
      deriveEffectiveQuery(binding, { inputs: { custom: 'Nvidia' } }, surface),
    ).toMatchObject({ query: 'Nvidia', source: 'composed' });
  });

  it('caps an over-long composed query to <= 10000 chars', () => {
    const surface = makeSurface([
      { id: 'c', component: 'TextArea', props: { value: { path: '/inputs/c' } } },
    ]);
    const binding = makeBinding({ inputs: { query: { path: '/inputs/q' } } });
    const eq = deriveEffectiveQuery(binding, { inputs: { c: 'x'.repeat(11000) } }, surface);
    expect(eq.query.length).toBeLessThanOrEqual(10000);
  });

  it('compileBinding stays backward-compatible without a surface arg', () => {
    const binding = makeBinding({ inputs: { query: { path: '/inputs/q' } } });
    const compiled = compileBinding(binding, { inputs: { q: 'hi' } });
    expect(compiled.query).toBe('hi');
    expect(compiled.querySource).toBe('bound');
    expect(compileBinding(binding, { inputs: { q: '' } }).query).toBe('');
  });

  it('compileBinding threads surface into composition + sets querySource', () => {
    const surface = makeSurface([
      { id: 'sel', component: 'Select', props: { value: { path: '/inputs/ticker' } } },
      { id: 'custom', component: 'TextField', props: { value: { path: '/inputs/custom' } } },
    ]);
    const binding = makeBinding({
      inputs: { query: { path: '/inputs/ticker' }, custom_query: { path: '/inputs/custom' } },
    });
    const compiled = compileBinding(binding, { inputs: { ticker: '', custom: 'Tesla' } }, surface);
    expect(compiled.query).toBe('Tesla');
    expect(compiled.querySource).toBe('composed');
    expect(compiled.surfaceInputs.custom_query).toBe('Tesla'); // extra input still flows
  });
});
