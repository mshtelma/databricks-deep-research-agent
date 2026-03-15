import { test, expect } from '../fixtures';
import { FOLLOW_UP_QUERIES, CONTEXT_PATTERNS } from '../utils/test-data';

/**
 * Follow-up Questions Tests - Validate context-aware responses to follow-up questions.
 *
 * User Story 9.2: Follow-up questions
 * - Create initial research conversation
 * - Submit follow-up question
 * - Verify context-aware response
 *
 * IMPLEMENTATION STATUS:
 * - Frontend now tracks conversation history in-memory within a session
 * - Backend stream endpoint loads history from database OR accepts history via POST
 * - The real orchestrator is now connected and uses conversation_history for context
 *
 * NOTE: These tests require:
 * - Backend running with valid Databricks LLM credentials
 * - Valid Brave Search API key
 *
 * Run these tests only when you have a fully configured environment.
 * For CI without credentials, these remain skipped.
 */
test.describe('Follow-up Questions', () => {
  test.setTimeout(120000); // 2 minutes

  // Enable these tests when running against a fully configured environment
  const runWithCredentials = process.env.RUN_INTEGRATION_TESTS === 'true';

  test.skip(!runWithCredentials, 'responds using context from previous exchange', async ({
    chatPage,
  }) => {
    // Given: initial conversation about machine learning
    await chatPage.sendMessage('What is machine learning?');
    await chatPage.waitForAgentResponse(60000);

    // When: submit follow-up question
    await chatPage.sendMessage(FOLLOW_UP_QUERIES[0].text); // "Can you give me an example?"
    await chatPage.waitForAgentResponse(30000);

    // Then: verify context-aware response
    const response = await chatPage.getLastAgentResponse();

    // Response should reference machine learning context
    const lowerResponse = response.toLowerCase();
    const hasContext =
      CONTEXT_PATTERNS.machineLearning.test(lowerResponse) ||
      CONTEXT_PATTERNS.general.test(lowerResponse);
    expect(hasContext).toBe(true);
  });

  test.skip(!runWithCredentials, 'follow-up about performance maintains context', async ({
    chatPage,
  }) => {
    // Initial question about Python
    await chatPage.sendMessage('What is Python used for?');
    await chatPage.waitForAgentResponse(60000);

    // Follow-up about performance
    await chatPage.sendMessage(FOLLOW_UP_QUERIES[1].text); // "What about the performance implications?"
    await chatPage.waitForAgentResponse(30000);

    // Response should still relate to Python
    const response = await chatPage.getLastAgentResponse();
    const lowerResponse = response.toLowerCase();

    // Should mention Python or performance-related terms
    const hasContext =
      lowerResponse.includes('python') ||
      lowerResponse.includes('performance') ||
      lowerResponse.includes('speed') ||
      lowerResponse.includes('fast');
    expect(hasContext).toBe(true);
  });

  test.skip(!runWithCredentials, 'comparison follow-up provides relevant alternatives', async ({
    chatPage,
  }) => {
    // Initial question
    await chatPage.sendMessage('What is React?');
    await chatPage.waitForAgentResponse(60000);

    // Follow-up asking for comparison
    await chatPage.sendMessage(FOLLOW_UP_QUERIES[2].text); // "How does this compare to alternatives?"
    await chatPage.waitForAgentResponse(30000);

    // Response should provide comparisons
    const response = await chatPage.getLastAgentResponse();
    const lowerResponse = response.toLowerCase();

    // Should mention React or comparison terms
    const hasComparison =
      lowerResponse.includes('react') ||
      lowerResponse.includes('vue') ||
      lowerResponse.includes('angular') ||
      lowerResponse.includes('compare') ||
      lowerResponse.includes('alternative') ||
      lowerResponse.includes('versus');
    expect(hasComparison).toBe(true);
  });

  test.skip(
    !runWithCredentials,
    'maintains conversation history across multiple exchanges',
    async ({ chatPage }) => {
      // First message
      await chatPage.sendMessage('Tell me about TypeScript');
      await chatPage.waitForAgentResponse(60000);

      // Second message - follow up
      await chatPage.sendMessage('What are its advantages?');
      await chatPage.waitForAgentResponse(30000);

      // Third message - another follow up
      await chatPage.sendMessage('How do I get started?');
      await chatPage.waitForAgentResponse(30000);

      // Verify all messages are in the conversation
      const userMessages = await chatPage.getUserMessages();
      expect(userMessages.length).toBe(3);

      // Verify responses exist for all
      const agentResponses = await chatPage.getAgentResponses();
      expect(agentResponses.length).toBe(3);

      // Last response should still be TypeScript-related
      const lastResponse = agentResponses[2].toLowerCase();
      const hasContext =
        lastResponse.includes('typescript') ||
        lastResponse.includes('type') ||
        lastResponse.includes('install') ||
        lastResponse.includes('start');
      expect(hasContext).toBe(true);
    }
  );
});
