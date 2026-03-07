/**
 * Custom Agent Research Tests — full research pipeline with agent configuration.
 *
 * Tests that agent-configured research (enterprise sources, domain filtering,
 * preset steps, stale endpoint fallback) produces valid results.
 *
 * All assertions are structural only — no content matching on LLM outputs.
 *
 * Run with:
 *   RUN_SLOW_TESTS=1 npx playwright test custom-agent-research --reporter=list
 */

import { test, expect } from '../fixtures/custom-agent.fixture';
import { AgentApiHelper } from '../utils/custom-agent-api';
import {
  makeEnterpriseAgent,
  makeKAOnlyAgent,
  makeWebFilteredAgent,
  makeExcludeDomainAgent,
  makeManualWorkflowAgent,
  makeHybridWorkflowAgent,
  makeStaleEndpointAgent,
  makePresetSteps,
  AGENT_TIMEOUTS,
  AGENT_RESEARCH_QUERIES,
} from '../utils/custom-agent-test-data';

test.describe('Custom Agent Research', () => {
  // Research tests are inherently slow
  test.slow();
  test.setTimeout(AGENT_TIMEOUTS.fullScenario);

  // Gate: require explicit opt-in for slow tests
  test.skip(
    !process.env.RUN_SLOW_TESTS,
    'Custom agent research tests require RUN_SLOW_TESTS=1',
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
  // Enterprise Sources
  // =========================================================================

  test.describe('Enterprise Sources', () => {
    test('enterprise-only agent produces a response', async ({
      page,
      chatPage,
      agentApi,
      customAgentPage,
    }) => {
      const agent = await agentApi.create(makeEnterpriseAgent());
      await page.reload();
      await chatPage.waitForReady();
      await customAgentPage.waitForReady(AGENT_TIMEOUTS.ui);

      await customAgentPage.selectAgent(agent.id);
      await chatPage.sendMessage(AGENT_RESEARCH_QUERIES.enterprise);
      await chatPage.waitForAgentResponse(AGENT_TIMEOUTS.lightResearch);

      const response = await chatPage.getLastAgentResponse();
      expect(response.length).toBeGreaterThan(0);
    });

    test('KA-only agent produces a response', async ({
      page,
      chatPage,
      agentApi,
      customAgentPage,
    }) => {
      const agent = await agentApi.create(makeKAOnlyAgent());
      await page.reload();
      await chatPage.waitForReady();
      await customAgentPage.waitForReady(AGENT_TIMEOUTS.ui);

      await customAgentPage.selectAgent(agent.id);
      await chatPage.sendMessage(AGENT_RESEARCH_QUERIES.enterprise);
      await chatPage.waitForAgentResponse(AGENT_TIMEOUTS.lightResearch);

      const response = await chatPage.getLastAgentResponse();
      expect(response.length).toBeGreaterThan(0);
    });
  });

  // =========================================================================
  // Domain Filtering
  // =========================================================================

  test.describe('Domain Filtering', () => {
    test('include-domain agent completes research', async ({
      page,
      chatPage,
      agentApi,
      customAgentPage,
    }) => {
      const agent = await agentApi.create(makeWebFilteredAgent());
      await page.reload();
      await chatPage.waitForReady();
      await customAgentPage.waitForReady(AGENT_TIMEOUTS.ui);

      await customAgentPage.selectAgent(agent.id);
      await chatPage.sendMessage(AGENT_RESEARCH_QUERIES.domainFiltered);
      await chatPage.waitForAgentResponse(AGENT_TIMEOUTS.lightResearch);

      const response = await chatPage.getLastAgentResponse();
      // Config was accepted and research completed — structural assertion only.
      // We do NOT try to parse citations to verify domains client-side.
      expect(response.length).toBeGreaterThan(0);
    });

    test('exclude-domain agent completes research', async ({
      page,
      chatPage,
      agentApi,
      customAgentPage,
    }) => {
      const agent = await agentApi.create(makeExcludeDomainAgent());
      await page.reload();
      await chatPage.waitForReady();
      await customAgentPage.waitForReady(AGENT_TIMEOUTS.ui);

      await customAgentPage.selectAgent(agent.id);
      await chatPage.sendMessage(AGENT_RESEARCH_QUERIES.web);
      await chatPage.waitForAgentResponse(AGENT_TIMEOUTS.lightResearch);

      const response = await chatPage.getLastAgentResponse();
      expect(response.length).toBeGreaterThan(0);
    });
  });

  // =========================================================================
  // Preset Steps
  // =========================================================================

  test.describe('Preset Steps', () => {
    test('manual mode agent executes and produces response', async ({
      page,
      chatPage,
      agentApi,
      customAgentPage,
      researchPage,
    }) => {
      const steps = makePresetSteps();
      const { agent } = await agentApi.createWithSteps(
        makeManualWorkflowAgent(),
        steps,
      );
      await page.reload();
      await chatPage.waitForReady();
      await customAgentPage.waitForReady(AGENT_TIMEOUTS.ui);

      await customAgentPage.selectAgent(agent.id);
      await chatPage.sendMessage(AGENT_RESEARCH_QUERIES.light);
      await chatPage.waitForAgentResponse(AGENT_TIMEOUTS.research);

      const response = await chatPage.getLastAgentResponse();
      expect(response.length).toBeGreaterThan(0);

      // Soft check: reasoning panel may show step titles (warn, don't fail)
      const isReasoningVisible = await researchPage.isReasoningVisible();
      if (isReasoningVisible) {
        const reasoningSteps = await researchPage.getReasoningSteps();
        if (reasoningSteps.length === 0) {
          console.warn(
            'Reasoning panel visible but no steps found — preset step titles may not appear in UI yet',
          );
        }
      }
    });

    test('hybrid mode agent executes and produces response', async ({
      page,
      chatPage,
      agentApi,
      customAgentPage,
    }) => {
      const steps = makePresetSteps().slice(0, 2); // Use first 2 steps only
      const { agent } = await agentApi.createWithSteps(
        makeHybridWorkflowAgent(),
        steps,
      );
      await page.reload();
      await chatPage.waitForReady();
      await customAgentPage.waitForReady(AGENT_TIMEOUTS.ui);

      await customAgentPage.selectAgent(agent.id);
      await chatPage.sendMessage(AGENT_RESEARCH_QUERIES.light);
      await chatPage.waitForAgentResponse(AGENT_TIMEOUTS.research);

      const response = await chatPage.getLastAgentResponse();
      expect(response.length).toBeGreaterThan(0);
    });
  });

  // =========================================================================
  // Stale Endpoint Fallback
  // =========================================================================

  test.describe('Stale Endpoint Fallback', () => {
    test('agent with nonexistent model override still completes', async ({
      page,
      chatPage,
      agentApi,
      customAgentPage,
    }) => {
      const agent = await agentApi.create(makeStaleEndpointAgent());
      await page.reload();
      await chatPage.waitForReady();
      await customAgentPage.waitForReady(AGENT_TIMEOUTS.ui);

      await customAgentPage.selectAgent(agent.id);
      await chatPage.sendMessage(AGENT_RESEARCH_QUERIES.light);
      await chatPage.waitForAgentResponse(AGENT_TIMEOUTS.lightResearch);

      // Research should complete using the fallback default endpoint
      const response = await chatPage.getLastAgentResponse();
      expect(response.length).toBeGreaterThan(0);
    });
  });

  // =========================================================================
  // Job Submission with agent_id
  // =========================================================================

  test.describe('Job Submission', () => {
    // These are fast API-only tests, so use a shorter gate
    test.skip(
      !process.env.RUN_INTEGRATION_TESTS && !process.env.RUN_SLOW_TESTS,
      'Job submission tests require RUN_INTEGRATION_TESTS=1 or RUN_SLOW_TESTS=1',
    );

    test('POST job with valid agent_id returns success', async ({
      page,
      agentApi,
    }) => {
      const agent = await agentApi.create(makeEnterpriseAgent());

      // Create a chat first (required field for job submission)
      const chatResponse = await page.request.post('/api/v1/chats', {
        data: { title: 'Agent test chat' },
      });
      expect(chatResponse.ok()).toBe(true);
      const chat = await chatResponse.json();

      const response = await page.request.post('/api/v1/research/jobs', {
        data: {
          chat_id: chat.id,
          query: AGENT_RESEARCH_QUERIES.enterprise,
          agent_id: agent.id,
        },
      });

      // Accept 200 or 201 (job created/started)
      expect(
        response.status() === 200 || response.status() === 201,
        `Expected 200 or 201, got ${response.status()}: ${await response.text()}`,
      ).toBe(true);
    });

    test('POST job with invalid agent_id returns 404', async ({ page }) => {
      // Create a chat first (required field for job submission)
      const chatResponse = await page.request.post('/api/v1/chats', {
        data: { title: 'Agent test chat' },
      });
      expect(chatResponse.ok()).toBe(true);
      const chat = await chatResponse.json();

      const response = await page.request.post('/api/v1/research/jobs', {
        data: {
          chat_id: chat.id,
          query: AGENT_RESEARCH_QUERIES.light,
          agent_id: '00000000-0000-0000-0000-000000000000',
        },
      });

      expect(response.status()).toBe(404);
    });
  });
});
