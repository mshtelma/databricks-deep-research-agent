/**
 * Agent UI Surfaces — chat panel end-to-end.
 *
 * A designer-built (agents_v2) agent carrying a declarative UI at
 * `definition.surface` renders a pinned Agent UI panel in chat: a form bound
 * to a data model, a Run action that submits a job with `surface_inputs`,
 * and a results region resolved by reference. Per-chat panel state persists
 * via `PUT /chats/{id}/surface-state` and rehydrates through `GET /full`.
 *
 * Covers:
 *   1. scaffold-surface API returns a valid default surface
 *   2. selecting a surface agent shows the panel with the scaffolded form
 *   3. filling the form + Run starts a job (submission path incl. surface_inputs)
 *   4. panel form state survives a reload (surface-state persistence)
 *
 * Run with:
 *   RUN_INTEGRATION_TESTS=1 npx playwright test agent-surface --reporter=list
 */

import { test, expect, type Page } from '@playwright/test';

const AGENTS_V2_BASE = '/api/v1/agents-v2';
const DESIGNER_BASE = '/api/v1/agent-designer';

// ---------------------------------------------------------------------------
// Helpers (agents_v2 — pattern from designer-block-create.spec.ts)
// ---------------------------------------------------------------------------

async function isAgentsV2Available(page: Page): Promise<boolean> {
  try {
    const resp = await page.request.get(AGENTS_V2_BASE);
    return resp.ok();
  } catch {
    return false;
  }
}

function minimalDefinition(name: string): Record<string, unknown> {
  return {
    id: `workflow-${Date.now()}`,
    name,
    description: 'Agent-surface e2e fixture',
    version: 1,
    root: {
      id: 'root-0',
      type: 'sequence',
      label: 'Workflow',
      config: {},
      children: [],
    },
    tools: [],
    pools: [],
    sources: [],
    models: {},
    required_inputs: ['query'],
    output_keys: ['output'],
    token_budget: 0,
    timeout_seconds: 1800,
  };
}

/** Scaffold the default surface for a definition via the stateless endpoint. */
async function scaffoldSurface(
  page: Page,
  definition: Record<string, unknown>,
): Promise<Record<string, unknown>> {
  const resp = await page.request.post(`${DESIGNER_BASE}/scaffold-surface`, {
    data: { definition },
  });
  const body = await resp.text();
  expect(resp.ok(), `scaffold-surface failed (${resp.status()}): ${body}`).toBe(true);
  return (JSON.parse(body) as { surface: Record<string, unknown> }).surface;
}

/** Create an agents_v2 agent whose definition carries a surface. */
async function createSurfaceAgent(page: Page, name: string): Promise<string> {
  const definition = minimalDefinition(name);
  definition.surface = await scaffoldSurface(page, definition);
  const resp = await page.request.post(AGENTS_V2_BASE, {
    data: { name, description: 'Agent-surface e2e fixture', definition },
  });
  const body = await resp.text();
  expect(resp.ok(), `agent create failed (${resp.status()}): ${body}`).toBe(true);
  return (JSON.parse(body) as { id: string }).id;
}

async function deleteAgent(page: Page, id: string): Promise<void> {
  try {
    await page.request.delete(`${AGENTS_V2_BASE}/${id}`);
  } catch {
    // best-effort cleanup
  }
}

/** Open the composer agent picker and select the agent by name. */
async function selectAgentInChat(page: Page, name: string): Promise<void> {
  await page.getByTestId('agent-selector-trigger').click();
  await page.locator(`[data-testid^="agent-option-"]`, { hasText: name }).first().click();
}

// ---------------------------------------------------------------------------

test.describe('Agent UI Surfaces', () => {
  test.setTimeout(180_000);

  test.skip(
    !process.env.RUN_INTEGRATION_TESTS,
    'Agent surface tests require RUN_INTEGRATION_TESTS=1',
  );

  test.beforeAll(async ({ browser }) => {
    const page = await browser.newPage();
    try {
      const available = await isAgentsV2Available(page);
      test.skip(!available, 'agents-v2 API not available');
    } finally {
      await page.close();
    }
  });

  test('scaffold-surface returns a valid default surface', async ({ page }) => {
    const surface = await scaffoldSurface(page, minimalDefinition('E2E Scaffold Probe'));
    const components = surface.components as Array<{ id: string; component: string }>;
    expect(components.some((c) => c.id === 'root')).toBe(true);
    expect(components.some((c) => c.component === 'Button')).toBe(true);
    expect(components.some((c) => c.component === 'ReportRegion')).toBe(true);
    const bindings = surface.bindings as Array<{ action: string }>;
    expect(bindings[0]?.action).toBe('run');
  });

  test('selecting a surface agent shows the pinned Agent UI panel', async ({ page }) => {
    const name = `E2E Surface Agent ${Date.now()}`;
    const agentId = await createSurfaceAgent(page, name);
    try {
      await page.goto('/');
      await selectAgentInChat(page, name);

      const panel = page.getByTestId('agent-surface-panel');
      await expect(panel).toBeVisible({ timeout: 15_000 });
      // Scaffolded form: query textarea + host-owned run controls/action.
      await expect(panel.getByText('Research request')).toBeVisible();
      await expect(page.getByTestId('surface-host-action-run')).toBeVisible();
    } finally {
      await deleteAgent(page, agentId);
    }
  });

  test('filling the form and clicking Run starts a job with surface inputs', async ({
    page,
  }) => {
    const name = `E2E Surface Run ${Date.now()}`;
    const agentId = await createSurfaceAgent(page, name);
    try {
      await page.goto('/');
      await selectAgentInChat(page, name);
      const panel = page.getByTestId('agent-surface-panel');
      await expect(panel).toBeVisible({ timeout: 15_000 });

      // Fill the scaffolded query field.
      await panel.locator('textarea').first().fill('What is 2+2? Answer briefly.');

      // Intercept the job submission to assert the payload carries the query.
      const submission = page.waitForRequest(
        (req) => req.url().includes('/research/jobs') && req.method() === 'POST',
        { timeout: 20_000 },
      );
      await page.getByTestId('surface-host-action-run').click();
      const req = await submission;
      const body = req.postDataJSON() as { query?: string; agent_id?: string };
      expect(body.query).toContain('2+2');
      expect(body.agent_id).toBe(agentId);

      // The run appears as a normal chat turn: the user message shows up and
      // the Run button disables while the job is active.
      await expect(page.getByText('What is 2+2? Answer briefly.').first()).toBeVisible({
        timeout: 20_000,
      });
      await expect(page.getByTestId('surface-host-action-run')).toBeDisabled();
    } finally {
      await deleteAgent(page, agentId);
    }
  });

  test('panel form state survives a reload (surface-state persistence)', async ({
    page,
  }) => {
    const name = `E2E Surface Persist ${Date.now()}`;
    const agentId = await createSurfaceAgent(page, name);
    try {
      await page.goto('/');
      await selectAgentInChat(page, name);
      const panel = page.getByTestId('agent-surface-panel');
      await expect(panel).toBeVisible({ timeout: 15_000 });

      // Persistence requires a REAL chat: send a trivial composer message
      // first so the draft chat materializes, then type into the panel.
      await page.getByTestId('message-input').fill('hello');
      await page.keyboard.press('Enter');
      await page.waitForURL(/\/chat\//, { timeout: 30_000 });

      const marker = `persisted-${Date.now()}`;
      await panel.locator('textarea').first().fill(marker);
      // Debounced PUT — wait for the surface-state write to fire.
      await page.waitForRequest(
        (req) => req.url().includes('/surface-state') && req.method() === 'PUT',
        { timeout: 10_000 },
      );

      await page.reload();
      await expect(page.getByTestId('agent-surface-panel')).toBeVisible({
        timeout: 20_000,
      });
      await expect(
        page.getByTestId('agent-surface-panel').locator('textarea').first(),
      ).toHaveValue(marker, { timeout: 15_000 });
    } finally {
      await deleteAgent(page, agentId);
    }
  });

  test('designer Preview simulates output and Try in chat opens the bound chat', async ({
    page,
  }) => {
    const name = `E2E Preview Simulate ${Date.now()}`;
    const agentId = await createSurfaceAgent(page, name);
    try {
      await page.goto(`/designer/${agentId}`);
      await page.getByRole('tab', { name: 'Preview' }).click();

      // Simulate: dry-run card + watermarked sample report in the region.
      await page.getByTestId('surface-action-run').click();
      await expect(page.getByText(/Simulated run — action/)).toBeVisible();
      await expect(page.getByTestId('surface-preview-sample')).toBeVisible({
        timeout: 10_000,
      });
      await expect(
        page.getByText(/Sample output — illustrative only/),
      ).toBeVisible();

      // Try in chat: lands on /chat with the pinned panel bound to the agent.
      await page.getByTestId('surface-preview-try-in-chat').click();
      await page.waitForURL(/\/chat/, { timeout: 20_000 });
      await expect(page.getByTestId('agent-surface-panel')).toBeVisible({
        timeout: 15_000,
      });
    } finally {
      await deleteAgent(page, agentId);
    }
  });

  test('structured output fills a Table after a real run', async ({ page }) => {
    test.skip(
      !process.env.RUN_SLOW_TESTS,
      'Structured-output run requires RUN_SLOW_TESTS=1',
    );
    const name = `E2E Structured Output ${Date.now()}`;
    // Agent whose surface declares structured-output slots (Table + findings).
    const definition = minimalDefinition(name);
    const surface = await scaffoldSurface(page, definition);
    const components = surface.components as Array<Record<string, unknown>>;
    const root = components.find((c) => c.id === 'root') as {
      children: string[];
    };
    components.push(
      {
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
      },
      {
        id: 'findings',
        component: 'KeyFindings',
        props: { source: { path: '/results/run/data/highlights' } },
        children: [],
      },
    );
    root.children.push('facts_table', 'findings');
    definition.surface = surface;
    const resp = await page.request.post(AGENTS_V2_BASE, {
      data: { name, description: 'Structured-output e2e fixture', definition },
    });
    expect(resp.ok(), await resp.text()).toBe(true);
    const agentId = (JSON.parse(await resp.text()) as { id: string }).id;

    try {
      await page.goto('/');
      await selectAgentInChat(page, name);
      const panel = page.getByTestId('agent-surface-panel');
      await expect(panel).toBeVisible({ timeout: 15_000 });

      await panel
        .locator('textarea')
        .first()
        .fill('How many sides does a triangle have? Answer briefly.');
      await page.getByTestId('surface-host-action-run').click();

      // The structuring pass runs post-synthesis; the table populates from
      // the chatFull refetch after persistence. Real runs take minutes.
      await expect(panel.getByTestId('surface-table-facts_table')).toBeVisible({
        timeout: 240_000,
      });
      const rows = panel.locator(
        '[data-testid="surface-table-facts_table"] tbody tr',
      );
      expect(await rows.count()).toBeGreaterThan(0);
    } finally {
      await deleteAgent(page, agentId);
    }
  });

  test('Run for real executes the agent from the Preview tab', async ({ page }) => {
    test.skip(
      !process.env.RUN_SLOW_TESTS,
      'Preview real-run requires RUN_SLOW_TESTS=1',
    );
    const name = `E2E Preview Real Run ${Date.now()}`;
    const agentId = await createSurfaceAgent(page, name);
    try {
      await page.goto(`/designer/${agentId}`);
      await page.getByRole('tab', { name: 'Preview' }).click();

      // The scaffolded query field is the first textarea in the preview canvas.
      await page.locator('textarea').first().fill('What is 2+2? Answer briefly.');

      const submission = page.waitForRequest(
        (req) => req.url().includes('/research/jobs') && req.method() === 'POST',
        { timeout: 30_000 },
      );
      await page.getByTestId('surface-action-run').click();
      await page.getByTestId('surface-preview-run-real').click();
      const req = await submission;
      const body = req.postDataJSON() as { agent_id?: string };
      expect(body.agent_id).toBe(agentId);

      // Live region while streaming, then a terminal state (report or graceful
      // failure) — a real run can take minutes.
      await expect(page.getByTestId('surface-preview-real-running')).toBeVisible({
        timeout: 30_000,
      });
      await expect(
        page
          .getByTestId('surface-preview-real-completed')
          .or(page.getByTestId('surface-preview-real-failed')),
      ).toBeVisible({ timeout: 150_000 });
    } finally {
      await deleteAgent(page, agentId);
    }
  });
});
