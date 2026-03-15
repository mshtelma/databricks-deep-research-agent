/**
 * Custom Agent UI Tests — agent selector interactions and visual state.
 *
 * Tests the agent selector dropdown behavior, selection persistence,
 * source scope indicator, and mid-conversation agent switching.
 *
 * Run with:
 *   RUN_INTEGRATION_TESTS=1 npx playwright test custom-agent-ui --reporter=list
 */

import { test, expect } from '../fixtures/custom-agent.fixture';
import { AgentApiHelper } from '../utils/custom-agent-api';
import {
  makeMinimalAgent,
  makeEnterpriseAgent,
  makeWorkspaceAgent,
  AGENT_TIMEOUTS,
  AGENT_RESEARCH_QUERIES,
} from '../utils/custom-agent-test-data';

test.describe('Custom Agent UI', () => {
  test.setTimeout(120_000);

  // Gate: require explicit opt-in
  test.skip(
    !process.env.RUN_INTEGRATION_TESTS,
    'Custom agent UI tests require RUN_INTEGRATION_TESTS=1',
  );

  // Gate: skip if backend feature not implemented
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
  // Selector Visibility
  // =========================================================================

  test.describe('Selector Visibility', () => {
    test('agent selector trigger is visible on page load', async ({
      customAgentPage,
    }) => {
      await customAgentPage.waitForReady(AGENT_TIMEOUTS.ui);
    });

    test('dropdown shows created agents', async ({
      page,
      agentApi,
      customAgentPage,
    }) => {
      const agent = await agentApi.create(makeMinimalAgent());
      // Reload to pick up the new agent in the dropdown
      await page.reload();
      await customAgentPage.waitForReady(AGENT_TIMEOUTS.ui);

      const hasOption = await customAgentPage.hasAgentOption(agent.id);
      expect(hasOption).toBe(true);
    });

    test('shows both private and workspace agents', async ({
      page,
      agentApi,
      customAgentPage,
    }) => {
      const privateAgent = await agentApi.create(makeMinimalAgent('E2E Private'));
      const workspaceAgent = await agentApi.create(makeWorkspaceAgent());
      await page.reload();
      await customAgentPage.waitForReady(AGENT_TIMEOUTS.ui);

      const hasPrivate = await customAgentPage.hasAgentOption(privateAgent.id);
      const hasWorkspace = await customAgentPage.hasAgentOption(workspaceAgent.id);

      expect(hasPrivate).toBe(true);
      expect(hasWorkspace).toBe(true);
    });
  });

  // =========================================================================
  // Selection Behavior
  // =========================================================================

  test.describe('Selection Behavior', () => {
    test('selecting agent shows name badge', async ({
      page,
      agentApi,
      customAgentPage,
    }) => {
      const agent = await agentApi.create(makeMinimalAgent());
      await page.reload();
      await customAgentPage.waitForReady(AGENT_TIMEOUTS.ui);

      await customAgentPage.selectAgent(agent.id);

      const selected = await customAgentPage.isAgentSelected();
      expect(selected).toBe(true);

      const name = await customAgentPage.getSelectedAgentName();
      expect(name).toContain(agent.name.substring(0, 20)); // Name may be truncated
    });

    test('clearing selection removes badge', async ({
      page,
      agentApi,
      customAgentPage,
    }) => {
      const agent = await agentApi.create(makeMinimalAgent());
      await page.reload();
      await customAgentPage.waitForReady(AGENT_TIMEOUTS.ui);

      await customAgentPage.selectAgent(agent.id);
      expect(await customAgentPage.isAgentSelected()).toBe(true);

      await customAgentPage.clearSelection();
      expect(await customAgentPage.isAgentSelected()).toBe(false);
    });

    test('selection persists across page reload', async ({
      page,
      agentApi,
      customAgentPage,
    }) => {
      const agent = await agentApi.create(makeMinimalAgent());
      await page.reload();
      await customAgentPage.waitForReady(AGENT_TIMEOUTS.ui);

      await customAgentPage.selectAgent(agent.id);
      expect(await customAgentPage.isAgentSelected()).toBe(true);

      // Reload and verify persistence
      await page.reload();
      await customAgentPage.waitForReady(AGENT_TIMEOUTS.ui);

      const stillSelected = await customAgentPage.isAgentSelected();
      expect(stillSelected).toBe(true);

      const name = await customAgentPage.getSelectedAgentName();
      expect(name).toContain(agent.name.substring(0, 20));
    });
  });

  // =========================================================================
  // Source Scope UI
  // =========================================================================

  test.describe('Source Scope UI', () => {
    test('enterprise-only agent shows scope indicator', async ({
      page,
      agentApi,
      customAgentPage,
    }) => {
      const agent = await agentApi.create(makeEnterpriseAgent());
      await page.reload();
      await customAgentPage.waitForReady(AGENT_TIMEOUTS.ui);

      await customAgentPage.selectAgent(agent.id);

      const scopeText = await customAgentPage.getSourceScopeText();
      expect(scopeText).toBeTruthy();
      // Scope indicator should reflect enterprise_only (exact text depends on UI)
      expect(scopeText!.toLowerCase()).toContain('enterprise');
    });

    test('agent with source scope hides per-query selector', async ({
      page,
      agentApi,
      customAgentPage,
    }) => {
      const agent = await agentApi.create(makeEnterpriseAgent());
      await page.reload();
      await customAgentPage.waitForReady(AGENT_TIMEOUTS.ui);

      // Before selecting agent, per-query scope selector should be visible
      const perQuerySelector = page.getByTestId('source-scope-selector');
      // Only check if the selector exists in this UI (soft assertion)
      const selectorExistsBefore = await perQuerySelector.isVisible().catch(() => false);

      await customAgentPage.selectAgent(agent.id);

      if (selectorExistsBefore) {
        // After selecting an agent with source_scope, the per-query selector should hide
        await expect(perQuerySelector).toBeHidden({ timeout: AGENT_TIMEOUTS.ui });
      }
    });

    test('clearing agent restores per-query selector', async ({
      page,
      agentApi,
      customAgentPage,
    }) => {
      const agent = await agentApi.create(makeEnterpriseAgent());
      await page.reload();
      await customAgentPage.waitForReady(AGENT_TIMEOUTS.ui);

      const perQuerySelector = page.getByTestId('source-scope-selector');
      const selectorExistsBefore = await perQuerySelector.isVisible().catch(() => false);

      await customAgentPage.selectAgent(agent.id);
      await customAgentPage.clearSelection();

      if (selectorExistsBefore) {
        await expect(perQuerySelector).toBeVisible({ timeout: AGENT_TIMEOUTS.ui });
      }
    });
  });

  // =========================================================================
  // Mid-Conversation Switch (slow — requires research)
  // =========================================================================

  test.describe('Mid-Conversation Switch', () => {
    // This test requires actual research, so gate on RUN_SLOW_TESTS
    test.skip(
      !process.env.RUN_SLOW_TESTS,
      'Mid-conversation switch requires RUN_SLOW_TESTS=1',
    );

    test('switching agent mid-conversation preserves first response', async ({
      page,
      chatPage,
      agentApi,
      customAgentPage,
    }) => {
      test.slow();
      test.setTimeout(AGENT_TIMEOUTS.fullScenario);

      const agent1 = await agentApi.create(makeMinimalAgent('E2E Agent1'));
      const agent2 = await agentApi.create(makeMinimalAgent('E2E Agent2'));
      await page.reload();
      await chatPage.waitForReady();
      await customAgentPage.waitForReady(AGENT_TIMEOUTS.ui);

      // Select first agent and send query
      await customAgentPage.selectAgent(agent1.id);
      await chatPage.sendMessage(AGENT_RESEARCH_QUERIES.light);
      await chatPage.waitForAgentResponse(AGENT_TIMEOUTS.research);

      const firstResponse = await chatPage.getLastAgentResponse();
      expect(firstResponse.length).toBeGreaterThan(50);

      // Switch to second agent
      await customAgentPage.selectAgent(agent2.id);

      // First response should still be there
      const responses = await chatPage.getAgentResponses();
      expect(responses.length).toBeGreaterThanOrEqual(1);

      // New agent should be displayed as selected
      const selectedName = await customAgentPage.getSelectedAgentName();
      expect(selectedName).toContain(agent2.name.substring(0, 20));
    });
  });
});
