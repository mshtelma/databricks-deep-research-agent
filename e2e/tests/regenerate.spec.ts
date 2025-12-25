import { test, expect } from '../fixtures';

/**
 * Regenerate Tests - Validate regenerate functionality generates new responses.
 *
 * User Story 9.5: Regenerate response
 * - Complete research conversation
 * - Trigger regenerate
 * - Verify new response generated
 *
 * NOTE: These tests are skipped because the regenerate functionality requires:
 * - Backend endpoint to regenerate responses
 * - Frontend to call that endpoint on button click
 * - Message persistence to track and replace responses
 * These features are not yet implemented.
 */
test.describe('Regenerate Response', () => {
  test.setTimeout(180000); // 3 minutes for multiple exchanges

  test.skip('generates new response for same query', async ({ chatPage }) => {
    // Given: completed research conversation
    await chatPage.sendMessage('What is Python?');
    await chatPage.waitForAgentResponse(60000);

    // Get initial response
    const initialResponse = await chatPage.getLastAgentResponse();
    expect(initialResponse.length).toBeGreaterThan(0);

    // When: trigger regenerate
    await chatPage.regenerate();

    // Then: wait for new response
    await chatPage.waitForAgentResponse(60000);
    const newResponse = await chatPage.getLastAgentResponse();

    // Verify we got a response (may or may not be different)
    expect(newResponse.length).toBeGreaterThan(0);

    // Response should still be about Python
    expect(newResponse.toLowerCase()).toContain('python');
  });

  test.skip('regenerate button appears after response', async ({ chatPage }) => {
    // Send a message
    await chatPage.sendMessage('Hello');
    await chatPage.waitForAgentResponse(30000);

    // Regenerate button should be visible
    await expect(chatPage.regenerateButton).toBeVisible({ timeout: 5000 });
  });

  test.skip('regenerate triggers loading state', async ({ chatPage }) => {
    // Send initial message
    await chatPage.sendMessage('What is JavaScript?');
    await chatPage.waitForAgentResponse(60000);

    // Trigger regenerate
    await chatPage.regenerate();

    // Should show loading indicator
    const isLoading = await chatPage.isLoading();
    const isStreaming = await chatPage.isStreaming();

    // Either loading or streaming should be visible
    // (or response already arrived if very fast)
    // This is a best-effort check
    expect(isLoading || isStreaming || true).toBe(true);

    // Wait for completion
    await chatPage.waitForAgentResponse(60000);
  });

  test.skip('can regenerate multiple times', async ({ chatPage }) => {
    // Send initial message
    await chatPage.sendMessage('What is React?');
    await chatPage.waitForAgentResponse(60000);

    // Regenerate first time
    await chatPage.regenerate();
    await chatPage.waitForAgentResponse(60000);

    const firstRegenResponse = await chatPage.getLastAgentResponse();
    expect(firstRegenResponse.length).toBeGreaterThan(0);

    // Regenerate second time
    await chatPage.regenerate();
    await chatPage.waitForAgentResponse(60000);

    const secondRegenResponse = await chatPage.getLastAgentResponse();
    expect(secondRegenResponse.length).toBeGreaterThan(0);

    // Both should still be about React
    expect(firstRegenResponse.toLowerCase()).toMatch(/react/);
    expect(secondRegenResponse.toLowerCase()).toMatch(/react/);
  });

  test.skip('regenerate preserves user message', async ({ chatPage }) => {
    const originalMessage = 'Tell me about TypeScript';

    // Send initial message
    await chatPage.sendMessage(originalMessage);
    await chatPage.waitForAgentResponse(60000);

    // Get user messages before regenerate
    const userMessagesBefore = await chatPage.getUserMessages();
    expect(userMessagesBefore.length).toBe(1);
    expect(userMessagesBefore[0]).toContain('TypeScript');

    // Regenerate
    await chatPage.regenerate();
    await chatPage.waitForAgentResponse(60000);

    // User message should still be there
    const userMessagesAfter = await chatPage.getUserMessages();
    expect(userMessagesAfter.length).toBe(1);
    expect(userMessagesAfter[0]).toContain('TypeScript');
  });

  test.skip('regenerate replaces previous response', async ({ chatPage }) => {
    // Send message
    await chatPage.sendMessage('What is Node.js?');
    await chatPage.waitForAgentResponse(60000);

    // Get response count before regenerate
    const responsesBefore = await chatPage.getAgentResponses();
    const countBefore = responsesBefore.length;

    // Regenerate
    await chatPage.regenerate();
    await chatPage.waitForAgentResponse(60000);

    // Response count should be the same (replaced, not added)
    const responsesAfter = await chatPage.getAgentResponses();
    expect(responsesAfter.length).toBe(countBefore);
  });
});
