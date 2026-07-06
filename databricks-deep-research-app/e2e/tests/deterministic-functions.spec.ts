/**
 * Deterministic-functions e2e — DEPLOYED-ONLY.
 *
 * Exercises the deterministic-function feature set against a real deployed app
 * (framework orchestrator + tool-node execution + sandbox subprocess + UC
 * functions via the OBO SQL executor). Runs custom agents_v2 workflows and
 * asserts on the persisted terminal status and synthesized report.
 *
 * Correctness is enforced structurally, so a terminal status of "completed" is
 * itself strong proof:
 *   - every tool node sets fail_on_error:true (a runtime error fails the job);
 *   - python_function bodies embed asserts (a wrong value fails the job);
 *   - the chain's second function reads a variable bound by the first, so
 *     "completed" proves the per-run SandboxSession carried state across nodes;
 *   - uc_function nodes execute real UC functions via SQL under OBO — a wrong
 *     value or a permission/SQL failure fails the node.
 * The report text additionally shows the synthesizer integrated the outputs.
 *
 * Constellations:
 *  1. python_function alone (numpy SMA in the run's sandbox session)
 *  2. python_function chain — bind_result -> reads_namespace (session persistence)
 *  3. uc_function (plain UDF msh.dre_e2e.pct_change) via OBO SQL
 *  4. uc_function wrapping a scalar ai_* builtin (msh.dre_e2e.sentiment) via OBO SQL
 *  5. mixed: python_function + uc_function + agent synthesis
 *  6. pure tool-node workflow (NO synthesizer) — proves report-less completion
 *     writes a non-empty assistant message (no persistence_transition_missing).
 *
 * `registered` tools need a server-side catalog entry this deployment does not
 * configure — they are covered by framework/app unit tests.
 *
 * Requires E2E_BASE_URL + E2E_BEARER_TOKEN (see `make e2e-deployed`).
 */
import { test, expect } from '@playwright/test';
import { createAgent, createChat, deleteAgent, runAgent } from '../utils/deterministic-fn-api';

const DEPLOYED = !!process.env.E2E_BASE_URL;
const UC_SCHEMA = process.env.E2E_UC_SCHEMA ?? 'msh.dre_e2e';

type Node = Record<string, unknown>;

function toolNode(id: string, ref: Node, config: Node): Node {
  return { id, type: 'tool', label: id, config: { ref, ...config } };
}

function synthNode(inputKeys: string[], template: string): Node {
  return {
    id: 'synth',
    type: 'agent',
    label: 'synth',
    config: {
      subtype: 'synthesizer',
      output_key: 'output',
      input_keys: ['query', ...inputKeys],
      user_prompt_template: template,
    },
  };
}

function definition(opts: {
  id: string;
  tools?: Node[];
  children: Node[];
}): Record<string, unknown> {
  return {
    id: opts.id,
    name: opts.id,
    version: 1,
    required_inputs: ['query'],
    output_keys: ['output'],
    tools: opts.tools ?? [],
    mcp_servers: [],
    pools: [{ name: 'sources' }],
    sources: [],
    models: {},
    root: { id: 'seq', type: 'sequence', label: 'seq', children: opts.children },
  };
}

// A first-class uc_function tool declaration: invoked via SQL under OBO. We pin
// explicit params (the runtime path); save-time introspection is authoring-only
// and covered by unit tests.
function ucFunction(name: string, fn: string, params: Node[]): Node {
  return {
    name,
    kind: 'uc_function',
    config: { function: `${UC_SCHEMA}.${fn}`, params, citeable: true },
  };
}

// SMA([10,20,30,40,50], w=3) = [20, 30, 40]; the body asserts that itself.
const SMA_FN: Node = {
  name: 'sma_fn',
  kind: 'python_function',
  description: '3-window SMA over a numeric series',
  config: {
    extra_allowed_modules: ['numpy'],
    citeable: true,
    params: [{ name: 'prices', type: 'array', required: true }],
    code:
      'import numpy as np\n' +
      'arr = np.asarray(prices, dtype=float)\n' +
      "r = np.convolve(arr, np.ones(3) / 3, mode='valid').tolist()\n" +
      'assert abs(r[0] - 20) < 1e-6 and abs(r[2] - 40) < 1e-6, r\n' +
      'result = r',
  },
};

// percentage change between two values; the body asserts the expected result.
const PCT_FN: Node = {
  name: 'pct_fn',
  kind: 'python_function',
  description: 'percentage change between two values',
  config: {
    citeable: true,
    params: [
      { name: 'old_value', type: 'number', required: true },
      { name: 'new_value', type: 'number', required: true },
    ],
    code:
      'r = (new_value - old_value) / old_value * 100.0\n' +
      'assert abs(r - 40.0) < 1e-6, r\n' +
      'result = r',
  },
};

test.describe('deterministic functions (deployed)', () => {
  test.skip(!DEPLOYED, 'deployed-only suite — set E2E_BASE_URL (make e2e-deployed)');
  test.describe.configure({ timeout: 300_000, mode: 'serial' });

  test('python_function computes SMA in the sandbox session', async ({ request }) => {
    const chatId = await createChat(request, 'e2e pyfn sma');
    const agentId = await createAgent(
      request,
      'e2e-pyfn-sma',
      definition({
        id: 'e2e-pyfn-sma',
        tools: [SMA_FN],
        children: [
          toolNode('compute_sma', { name: 'sma_fn' }, {
            input_literals: { prices: [10, 20, 30, 40, 50] },
            output_key: 'sma_summary',
            output_data_key: 'sma_data',
            fail_on_error: true,
          }),
          synthNode(['sma_summary'], 'Q: {query}\nSMA: {sma_summary}\nReport the moving-average values.'),
        ],
      }),
    );
    try {
      const out = await runAgent(request, { chatId, agentId, query: 'moving average of the sales series' });
      expect(out.status, `report: ${out.reportText.slice(0, 300)}`).toBe('completed');
      expect(out.reportText.length).toBeGreaterThan(0);
      expect(out.reportText).toContain('40');
    } finally {
      await deleteAgent(request, agentId);
    }
  });

  test('python_function chain shares state across nodes (session persistence)', async ({
    request,
  }) => {
    const chatId = await createChat(request, 'e2e pyfn chain');
    const agentId = await createAgent(
      request,
      'e2e-pyfn-chain',
      definition({
        id: 'e2e-pyfn-chain',
        tools: [
          {
            name: 'make_series',
            kind: 'python_function',
            config: { params: [], bind_result: 'series', code: 'result = [11, 22, 33]' },
          },
          {
            name: 'sum_series',
            kind: 'python_function',
            config: {
              citeable: true,
              params: [],
              reads_namespace: ['series'],
              // Reads `series` bound by make_series in the SAME run session.
              code: 'assert sum(series) == 66, series\nresult = sum(series)',
            },
          },
        ],
        children: [
          toolNode('n_make', { name: 'make_series' }, { output_key: 'make_out', fail_on_error: true }),
          toolNode('n_sum', { name: 'sum_series' }, {
            output_key: 'sum_out',
            output_data_key: 'sum_data',
            fail_on_error: true,
          }),
          synthNode(['sum_out'], 'Q: {query}\nSum: {sum_out}\nReport the total.'),
        ],
      }),
    );
    try {
      const out = await runAgent(request, { chatId, agentId, query: 'sum the series values' });
      // "completed" here proves the second function saw `series` from the first.
      expect(out.status, `report: ${out.reportText.slice(0, 300)}`).toBe('completed');
      expect(out.reportText).toContain('66');
    } finally {
      await deleteAgent(request, agentId);
    }
  });

  test('uc_function (plain UDF) executes via OBO SQL', async ({ request }) => {
    const chatId = await createChat(request, 'e2e uc pct');
    const agentId = await createAgent(
      request,
      'e2e-uc-pct',
      definition({
        id: 'e2e-uc-pct',
        tools: [
          ucFunction('uc_pct', 'pct_change', [
            { name: 'old_value', type: 'number', required: true },
            { name: 'new_value', type: 'number', required: true },
          ]),
        ],
        children: [
          toolNode('pct', { name: 'uc_pct' }, {
            input_literals: { old_value: 100, new_value: 140 },
            output_key: 'pct_summary',
            output_data_key: 'pct_data',
            fail_on_error: true,
          }),
          synthNode(['pct_summary'], 'Q: {query}\nResult: {pct_summary}\nReport the percentage change.'),
        ],
      }),
    );
    try {
      const out = await runAgent(request, { chatId, agentId, query: 'percent change from 100 to 140' });
      expect(out.status, `report: ${out.reportText.slice(0, 300)}`).toBe('completed');
      expect(out.reportText).toContain('40');
    } finally {
      await deleteAgent(request, agentId);
    }
  });

  test('uc_function wrapping a scalar ai_* builtin executes via OBO SQL', async ({
    request,
  }) => {
    const chatId = await createChat(request, 'e2e uc sentiment');
    const agentId = await createAgent(
      request,
      'e2e-uc-sentiment',
      definition({
        id: 'e2e-uc-sentiment',
        tools: [ucFunction('uc_sent', 'sentiment', [{ name: 'content', type: 'string', required: true }])],
        children: [
          toolNode('sent', { name: 'uc_sent' }, {
            input_literals: {
              content: 'This was an outstanding, fantastic quarter with record growth.',
            },
            output_key: 'sent_summary',
            output_data_key: 'sent_data',
            fail_on_error: true,
          }),
          synthNode(['sent_summary'], 'Q: {query}\nSentiment: {sent_summary}\nReport the sentiment label.'),
        ],
      }),
    );
    try {
      const out = await runAgent(request, { chatId, agentId, query: 'sentiment of the earnings comment' });
      expect(out.status, `report: ${out.reportText.slice(0, 300)}`).toBe('completed');
      expect(out.reportText.toLowerCase()).toContain('positive');
    } finally {
      await deleteAgent(request, agentId);
    }
  });

  test('mixed: python_function + uc_function + agent synthesis', async ({ request }) => {
    // A python_function transform + a UC-function KPI (OBO SQL) feed a synthesizer.
    const chatId = await createChat(request, 'e2e mixed');
    const agentId = await createAgent(
      request,
      'e2e-mixed',
      definition({
        id: 'e2e-mixed',
        tools: [
          SMA_FN,
          ucFunction('uc_pct', 'pct_change', [
            { name: 'old_value', type: 'number', required: true },
            { name: 'new_value', type: 'number', required: true },
          ]),
        ],
        children: [
          toolNode('compute_sma', { name: 'sma_fn' }, {
            input_literals: { prices: [10, 20, 30, 40, 50] },
            output_key: 'sma_summary',
            fail_on_error: true,
          }),
          toolNode('compute_pct', { name: 'uc_pct' }, {
            input_literals: { old_value: 100, new_value: 140 },
            output_key: 'pct_summary',
            fail_on_error: true,
          }),
          synthNode(
            ['sma_summary', 'pct_summary'],
            'Q: {query}\nSMA: {sma_summary}\nPct change: {pct_summary}\n' +
              'Summarize both computed results for the user.',
          ),
        ],
      }),
    );
    try {
      const out = await runAgent(request, {
        chatId,
        agentId,
        query: 'Summarize the moving average and the percent change.',
      });
      expect(out.status, `report: ${out.reportText.slice(0, 300)}`).toBe('completed');
      // SMA window yields 30, uc pct_change yields 40 — both must appear.
      expect(out.reportText).toContain('30');
      expect(out.reportText).toContain('40');
    } finally {
      await deleteAgent(request, agentId);
    }
  });

  test('pure tool-node workflow (no synthesizer) completes with a non-empty message', async ({
    request,
  }) => {
    // No LLM node at all: a single deterministic tool node whose output is the
    // workflow's terminal output_key. Proves the report-less completion fix —
    // the run reaches COMPLETED AND writes a non-empty assistant message
    // (previously it failed with persistence_transition_missing).
    const chatId = await createChat(request, 'e2e no-llm');
    const agentId = await createAgent(
      request,
      'e2e-no-llm',
      definition({
        id: 'e2e-no-llm',
        tools: [PCT_FN],
        children: [
          toolNode('compute_pct', { name: 'pct_fn' }, {
            input_literals: { old_value: 100, new_value: 140 },
            output_key: 'output', // terminal output_key — NO synthesizer
            fail_on_error: true,
          }),
        ],
      }),
    );
    try {
      const out = await runAgent(request, { chatId, agentId, query: 'percent change from 100 to 140' });
      expect(out.status, `report: ${out.reportText.slice(0, 300)}`).toBe('completed');
      // Fix C: a report-less run writes a non-empty message, not a NULL turn.
      expect(out.reportText.trim().length).toBeGreaterThan(0);
      expect(out.reportText).toContain('40');
    } finally {
      await deleteAgent(request, agentId);
    }
  });
});
