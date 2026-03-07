/**
 * Custom Agent CRUD Tests — API-level create, read, update, delete operations.
 *
 * Tests the full lifecycle of custom agents and preset steps via the REST API.
 * All tests are gated by RUN_INTEGRATION_TESTS + feature availability check.
 *
 * Run with:
 *   RUN_INTEGRATION_TESTS=1 npx playwright test custom-agent-crud --reporter=list
 */

import { test, expect } from '../fixtures/custom-agent.fixture';
import { AgentApiHelper } from '../utils/custom-agent-api';
import {
  makeMinimalAgent,
  makeEnterpriseAgent,
  makeWebFilteredAgent,
  makeModelOverrideAgent,
  makeWorkspaceAgent,
  makeStaleEndpointAgent,
  makePresetSteps,
  makeSourceScopedStep,
  AGENT_TIMEOUTS,
} from '../utils/custom-agent-test-data';

test.describe('Custom Agent CRUD', () => {
  test.setTimeout(60_000);

  // Gate: require explicit opt-in via environment variable
  test.skip(
    !process.env.RUN_INTEGRATION_TESTS,
    'Custom agent CRUD tests require RUN_INTEGRATION_TESTS=1',
  );

  // Gate: skip entire suite if the backend feature is not implemented
  test.beforeAll(async ({ browser }) => {
    const page = await browser.newPage();
    try {
      const available = await AgentApiHelper.isFeatureAvailable(page);
      test.skip(!available, 'Custom agents API not available (feature not implemented)');
    } finally {
      await page.close();
    }
  });

  // =========================================================================
  // Create
  // =========================================================================

  test.describe('Create', () => {
    test('creates agent with minimal defaults', async ({ agentApi }) => {
      const config = makeMinimalAgent();
      const agent = await agentApi.create(config);

      expect(agent.id).toBeTruthy();
      expect(agent.name).toBe(config.name);
      // Verify defaults are applied
      expect(agent.visibility).toBe('private');
      expect(agent.defaultDepth).toBe('medium');
      expect(agent.defaultMode).toBe('planner');
      expect(agent.usePlanner).toBe(true);
      expect(agent.sourceScope).toBe('all');
      expect(agent.domainFilterMode).toBeNull();
    });

    test('creates enterprise-only agent with enabled sources', async ({ agentApi }) => {
      const config = makeEnterpriseAgent();
      const agent = await agentApi.create(config);

      expect(agent.sourceScope).toBe('enterprise_only');
      expect(agent.enabledSources).toEqual(config.enabled_sources);
      expect(agent.defaultDepth).toBe('light');
      expect(agent.description).toBe(config.description);
    });

    test('creates web-filtered agent with domain include list', async ({ agentApi }) => {
      const config = makeWebFilteredAgent();
      const agent = await agentApi.create(config);

      expect(agent.sourceScope).toBe('web_only');
      expect(agent.domainFilterMode).toBe('include');
      expect(agent.includeDomains).toEqual(['*.gov', '*.edu']);
    });

    test('creates agent with model overrides', async ({ agentApi }) => {
      const config = makeModelOverrideAgent();
      const agent = await agentApi.create(config);

      expect(agent.modelOverrides).toEqual(config.model_overrides);
    });

    test('creates agent with workspace visibility', async ({ agentApi }) => {
      const config = makeWorkspaceAgent();
      const agent = await agentApi.create(config);

      expect(agent.visibility).toBe('workspace');
    });
  });

  // =========================================================================
  // Read
  // =========================================================================

  test.describe('Read', () => {
    test('gets agent by ID with full config', async ({ agentApi }) => {
      const config = makeEnterpriseAgent();
      const created = await agentApi.create(config);

      const fetched = await agentApi.get(created.id);

      expect(fetched.id).toBe(created.id);
      expect(fetched.name).toBe(config.name);
      expect(fetched.sourceScope).toBe('enterprise_only');
      expect(fetched.enabledSources).toEqual(config.enabled_sources);
      expect(fetched.createdAt).toBeTruthy();
      expect(fetched.updatedAt).toBeTruthy();
    });

    test('lists agents with category counts', async ({ agentApi }) => {
      // Create at least one agent so the list is non-empty
      await agentApi.create(makeMinimalAgent());

      const list = await agentApi.list();

      expect(list.agents).toBeInstanceOf(Array);
      expect(list.total).toBeGreaterThanOrEqual(1);
      expect(typeof list.userAgents).toBe('number');
      expect(typeof list.workspaceAgents).toBe('number');
      expect(typeof list.systemAgents).toBe('number');
      expect(list.total).toBe(
        list.userAgents + list.workspaceAgents + list.systemAgents,
      );
    });

    test('returns stale endpoint warnings for nonexistent endpoint', async ({
      agentApi,
    }) => {
      const config = makeStaleEndpointAgent();
      const agent = await agentApi.create(config);
      const fetched = await agentApi.get(agent.id);

      // The response should include warnings about the nonexistent endpoint
      expect(fetched.modelOverrideWarnings).toBeDefined();
      expect(fetched.modelOverrideWarnings!.length).toBeGreaterThanOrEqual(1);
      expect(fetched.modelOverrideWarnings![0].tier).toBe('complex');
      expect(fetched.modelOverrideWarnings![0].endpoint).toBe(
        'nonexistent-endpoint-for-testing',
      );
    });
  });

  // =========================================================================
  // Update
  // =========================================================================

  test.describe('Update', () => {
    test('updates agent name', async ({ agentApi }) => {
      const agent = await agentApi.create(makeMinimalAgent());
      const newName = `Updated ${agent.name}`;

      const updated = await agentApi.update(agent.id, { name: newName });

      expect(updated.name).toBe(newName);
      // Other fields remain unchanged
      expect(updated.visibility).toBe(agent.visibility);
    });

    test('updates source scope', async ({ agentApi }) => {
      const agent = await agentApi.create(makeMinimalAgent());

      const updated = await agentApi.update(agent.id, {
        source_scope: 'enterprise_only',
      });

      expect(updated.sourceScope).toBe('enterprise_only');
    });

    test('updates domain filter', async ({ agentApi }) => {
      const agent = await agentApi.create(makeMinimalAgent());

      const updated = await agentApi.update(agent.id, {
        source_scope: 'web_only',
        domain_filter_mode: 'exclude',
        exclude_domains: ['example.com'],
      });

      expect(updated.domainFilterMode).toBe('exclude');
      expect(updated.excludeDomains).toEqual(['example.com']);
    });

    test('updates workflow mode', async ({ agentApi }) => {
      const agent = await agentApi.create(makeMinimalAgent());

      const updated = await agentApi.update(agent.id, {
        default_mode: 'manual',
        use_planner: false,
      });

      expect(updated.defaultMode).toBe('manual');
      expect(updated.usePlanner).toBe(false);
    });
  });

  // =========================================================================
  // Delete
  // =========================================================================

  test.describe('Delete', () => {
    test('deletes agent successfully, re-fetch returns 404', async ({ agentApi }) => {
      const agent = await agentApi.create(makeMinimalAgent());
      await agentApi.delete(agent.id);

      const response = await agentApi.getRaw(agent.id);
      expect(response.status()).toBe(404);
    });

    test('deleting non-existent agent returns 404', async ({ agentApi }) => {
      const response = await agentApi.deleteRaw('00000000-0000-0000-0000-000000000000');
      expect(response.status()).toBe(404);
    });
  });

  // =========================================================================
  // Error Cases
  // =========================================================================

  test.describe('Error Cases', () => {
    test('duplicate name returns 409 Conflict', async ({ agentApi }) => {
      const config = makeMinimalAgent();
      await agentApi.create(config);

      // Attempt to create another agent with the same name
      const response = await agentApi.createRaw(config);
      expect(response.status()).toBe(409);
    });

    test('empty name returns 422 Unprocessable Entity', async ({ agentApi }) => {
      const response = await agentApi.createRaw({ name: '' });
      expect(response.status()).toBe(422);
    });

    test('domain filter mode without domains returns 422', async ({ agentApi }) => {
      const response = await agentApi.createRaw({
        name: `E2E Invalid ${Date.now()}`,
        domain_filter_mode: 'include',
        // Missing include_domains
      });
      expect(response.status()).toBe(422);
    });
  });

  // =========================================================================
  // Preset Steps
  // =========================================================================

  test.describe('Preset Steps', () => {
    test('creates multiple steps and lists them in order', async ({ agentApi }) => {
      const steps = makePresetSteps();
      const { agent, stepIds } = await agentApi.createWithSteps(
        makeMinimalAgent(),
        steps,
      );

      expect(stepIds).toHaveLength(3);

      const listed = await agentApi.listSteps(agent.id);
      expect(listed).toHaveLength(3);
      // Verify ordering
      expect(listed[0].order).toBe(1);
      expect(listed[1].order).toBe(2);
      expect(listed[2].order).toBe(3);
      expect(listed[0].title).toBe(steps[0].title);
    });

    test('reorders steps', async ({ agentApi }) => {
      const steps = makePresetSteps();
      const { agent, stepIds } = await agentApi.createWithSteps(
        makeMinimalAgent(),
        steps,
      );

      // Reverse the order: [3, 2, 1]
      const reversed = [...stepIds].reverse();
      await agentApi.reorderSteps(agent.id, reversed);

      const listed = await agentApi.listSteps(agent.id);
      expect(listed[0].id).toBe(reversed[0]);
      expect(listed[1].id).toBe(reversed[1]);
      expect(listed[2].id).toBe(reversed[2]);
    });

    test('deletes a step and updates count', async ({ agentApi }) => {
      const steps = makePresetSteps();
      const { agent, stepIds } = await agentApi.createWithSteps(
        makeMinimalAgent(),
        steps,
      );

      await agentApi.deleteStep(agent.id, stepIds[1]); // Delete middle step

      const listed = await agentApi.listSteps(agent.id);
      expect(listed).toHaveLength(2);
      expect(listed.find((s) => s.id === stepIds[1])).toBeUndefined();
    });

    test('creates step with per-step source scope', async ({ agentApi }) => {
      const agent = await agentApi.create(makeMinimalAgent());
      const step = makeSourceScopedStep(1, 'enterprise_only');

      const created = await agentApi.createStep(agent.id, step);

      expect(created.sourceScope).toBe('enterprise_only');
      expect(created.sourceHints).toBeTruthy();
    });

    test('lists steps for agent with no steps returns empty array', async ({
      agentApi,
    }) => {
      const agent = await agentApi.create(makeMinimalAgent());

      const steps = await agentApi.listSteps(agent.id);

      expect(steps).toEqual([]);
    });
  });
});
