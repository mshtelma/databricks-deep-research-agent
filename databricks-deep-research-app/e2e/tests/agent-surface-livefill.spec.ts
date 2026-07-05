/**
 * Agent UI Surfaces — LIVE structured-output fill after a silent SSE drop.
 *
 * Deterministic reproduction (no deploy, no real LLM) of the production bug:
 * a surface Run's SSE (EventSource) drops silently during the long
 * structured-output "wires" phase, so the client never receives
 * `phase_completed(structured_output)`. The report persists and the wires fill
 * the envelope server-side, but the surface's Results (a Table) never render the
 * filled data LIVE — only a page reload fixes it.
 *
 * We mock the run-time endpoints to script exactly that sequence:
 *   - POST /research/jobs           → fake job (no real run)
 *   - GET  …/{chat}/{sess}/stream   → emit persistence_completed, then close +
 *                                     keep dropping reconnects → give-up (error)
 *   - GET  …/{chat}/{sess}          → job status 'completed' (give-up heal path)
 *   - GET  …/chat/{chat}/active     → in_progress → null
 *   - GET  /chats/{chat}/full       → pending-stub envelope first, then FILLED
 *                                     (wires "finishing" server-side)
 *   - PUT  /chats/{chat}/surface-state → 200 no-op
 *
 * Expected: RED on current code (Table stays empty), GREEN after the fix.
 *
 * Run with:
 *   RUN_INTEGRATION_TESTS=1 E2E_BASE_URL=http://localhost:5173 \
 *     npx playwright test agent-surface-livefill --reporter=list
 */

import { test, expect, type Page, type Route } from '@playwright/test';

const AGENTS_V2_BASE = '/api/v1/agents-v2';
const DESIGNER_BASE = '/api/v1/agent-designer';

// Fixed ids for the mocked run (message_id deliberately ABSENT from surface_state,
// mirroring the production L2/heal signature).
const SESSION_ID = '5ac2f7aa-1111-4111-8111-111111111111';
const MESSAGE_ID = '63d604ae-2222-4222-8222-222222222222';
const RS_ID = 'a0a0a0a0-3333-4333-8333-333333333333';

// ---------------------------------------------------------------------------
// Agent fixture helpers (from agent-surface.spec.ts)
// ---------------------------------------------------------------------------

function minimalDefinition(name: string): Record<string, unknown> {
  return {
    id: `workflow-${Date.now()}`,
    name,
    description: 'Agent-surface livefill e2e fixture',
    version: 1,
    root: {
      id: 'root-0', type: 'sequence', label: 'Workflow', config: {},
      children: [
        { id: 'agent-0', type: 'agent', label: 'Synth', config: { subtype: 'synthesizer', model_tier: 'analytical', output_key: 'output' }, children: [] },
      ],
    },
    tools: [], pools: [], sources: [], models: {},
    required_inputs: ['query'], output_keys: ['output'],
    token_budget: 0, timeout_seconds: 1800,
  };
}

async function scaffoldSurface(page: Page, definition: Record<string, unknown>) {
  const resp = await page.request.post(`${DESIGNER_BASE}/scaffold-surface`, { data: { definition } });
  const body = await resp.text();
  expect(resp.ok(), `scaffold-surface failed (${resp.status()}): ${body}`).toBe(true);
  return (JSON.parse(body) as { surface: Record<string, unknown> }).surface;
}

/** Create an agents_v2 agent whose surface declares a structured-output Table (facts). */
async function createStructuredAgent(page: Page, name: string): Promise<string> {
  const definition = minimalDefinition(name);
  const surface = await scaffoldSurface(page, definition);
  const components = surface.components as Array<Record<string, unknown>>;
  const root = components.find((c) => c.id === 'root') as { children: string[] };
  components.push({
    id: 'facts_table',
    component: 'Table',
    props: {
      source: { path: '/results/run/data/facts' },
      columns: [
        { key: 'fact', label: 'Fact', type: 'string' },
        { key: 'value', label: 'Value', type: 'number' },
      ],
    },
    children: [],
  });
  root.children.push('facts_table');
  definition.surface = surface;
  const resp = await page.request.post(`${AGENTS_V2_BASE}?force=true`, {
    data: { name, description: 'livefill e2e fixture', definition },
  });
  const body = await resp.text();
  expect(resp.ok(), `agent create failed (${resp.status()}): ${body}`).toBe(true);
  return (JSON.parse(body) as { id: string }).id;
}

async function deleteAgent(page: Page, id: string): Promise<void> {
  try { await page.request.delete(`${AGENTS_V2_BASE}/${id}`); } catch { /* best-effort */ }
}

/** Pre-select the agent via localStorage (the composer restores it on mount),
 *  avoiding the mode-selector UI which is hidden in the e2e config. */
async function preselectAgent(page: Page, agentId: string): Promise<void> {
  await page.addInitScript((id) => {
    try { localStorage.setItem('deep-research-selected-agent', id as string); } catch { /* ignore */ }
  }, agentId);
}

// ---------------------------------------------------------------------------
// Mock payload builders (camelCase at model level; envelope inner keys snake)
// ---------------------------------------------------------------------------

function envelope(filled: boolean) {
  return {
    binding: 'run',
    version: 2,
    agent_id: 'agent',
    surface_etag: 'etag',
    generated_at: new Date().toISOString(),
    data: filled ? { facts: [{ fact: 'Alpha', value: 1 }, { fact: 'Beta', value: 2 }] } : {},
    meta: { slots: { facts: { status: filled ? 'ok' : 'pending' } } },
  };
}

function chatFull(chatId: string, agentId: string, filled: boolean) {
  const now = new Date().toISOString();
  return {
    id: chatId,
    title: 'Livefill Test',
    status: 'active',
    chatType: 'regular',
    createdAt: now,
    updatedAt: now,
    messageCount: 2,
    // snake passthrough; NO message_id (mirrors the production heal signature)
    surfaceState: { [agentId]: { action_runs: { run: { status: 'completed', session_id: SESSION_ID } } } },
    messages: [
      { id: 'user-msg-0', chatId, role: 'user', content: 'Give me facts.', createdAt: now, isEdited: false, researchSession: null, claims: [], verificationSummary: null },
      {
        id: MESSAGE_ID, chatId, role: 'agent', content: 'Report body [1].', createdAt: now, isEdited: false,
        researchSession: { id: RS_ID, messageId: MESSAGE_ID, status: 'completed', createdAt: now, completedAt: now, sources: [] },
        claims: [], verificationSummary: null,
        structuredOutput: envelope(filled),
      },
    ],
  };
}

// ---------------------------------------------------------------------------

test.describe('Agent surface — live structured-output fill after SSE drop', () => {
  test.setTimeout(120_000);
  test.skip(!process.env.RUN_INTEGRATION_TESTS, 'requires RUN_INTEGRATION_TESTS=1');

  test('fills the Table live after a silent SSE drop (no reload)', async ({ page }) => {
    const name = `E2E Livefill ${Date.now()}`;
    page.on('console', (m) => {
      const t = m.text();
      if (t.includes('[surface-dbg]')) console.log(t);
    });
    const agentId = await createStructuredAgent(page, name);

    // Mutable mock state.
    let streamCalls = 0;
    let filled = false;

    // POST /research/jobs → fake job (no real run).
    await page.route('**/api/v1/research/jobs', async (route: Route) => {
      if (route.request().method() !== 'POST') return route.fallback();
      const chatId = (route.request().postDataJSON() as { chat_id: string }).chat_id;
      await route.fulfill({
        status: 200, contentType: 'application/json',
        body: JSON.stringify({ sessionId: SESSION_ID, chatId, status: 'in_progress', createdAt: new Date().toISOString() }),
      });
    });

    // Everything else under /research/jobs/* — branch by URL/method.
    await page.route('**/api/v1/research/jobs/**', async (route: Route) => {
      const url = route.request().url();
      if (/\/stream(\?|$)/.test(url)) {
        streamCalls += 1;
        if (streamCalls === 1) {
          const m = url.match(/jobs\/([^/]+)\//);
          const chatId = m ? m[1] : '';
          const body =
            'retry: 200\n\n' +
            `data: ${JSON.stringify({ eventType: 'persistence_completed', sequenceNumber: 1, payload: { chatId, messageId: MESSAGE_ID, researchSessionId: RS_ID, chatTitle: 'Livefill Test', wasDraft: true, counts: {} } })}\n\n`;
          // wires "finish" ~0.8s after persistence.
          setTimeout(() => { filled = true; }, 800);
          await route.fulfill({ status: 200, contentType: 'text/event-stream', headers: { 'cache-control': 'no-cache' }, body });
        } else {
          // reconnect attempts → immediate close → drive give-up (>=3 rapid errors)
          await route.fulfill({ status: 200, contentType: 'text/event-stream', headers: { 'cache-control': 'no-cache' }, body: 'retry: 200\n\n' });
        }
        return;
      }
      if (/\/chat\/[^/]+\/active(\?|$)/.test(url)) {
        // in-progress until wires finish, then null (transition → invalidate).
        const active = filled ? null : { sessionId: SESSION_ID, status: 'in_progress' };
        await route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify(active) });
        return;
      }
      // job get (give-up heal path + resolveRunFromServer): completed.
      await route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify({ sessionId: SESSION_ID, status: 'completed' }) });
    });

    // /chats/{id}/full → pending stub, then filled. Other /chats/* → real backend.
    await page.route('**/api/v1/chats/**', async (route: Route) => {
      const url = route.request().url();
      const method = route.request().method();
      const fullMatch = url.match(/chats\/([^/]+)\/full/);
      if (fullMatch && method === 'GET') {
        await route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify(chatFull(fullMatch[1], agentId, filled)) });
        return;
      }
      if (/\/surface-state(\?|$)/.test(url) && method === 'PUT') {
        await route.fulfill({ status: 200, contentType: 'application/json', body: '{}' });
        return;
      }
      return route.fallback();
    });

    try {
      await preselectAgent(page, agentId);
      await page.goto('/');
      const panel = page.getByTestId('agent-surface-panel');
      await expect(panel).toBeVisible({ timeout: 15_000 });

      await panel.locator('textarea').first().fill('Give me facts.');
      await page.getByTestId('surface-host-action-run').click();

      // The Table must fill LIVE (no reload) once the wires finish server-side.
      const rows = panel.locator('[data-testid="surface-table-facts_table"] tbody tr');
      await expect(rows.first()).toBeVisible({ timeout: 30_000 });
      expect(await rows.count()).toBeGreaterThan(0);
    } finally {
      await deleteAgent(page, agentId);
    }
  });
});
