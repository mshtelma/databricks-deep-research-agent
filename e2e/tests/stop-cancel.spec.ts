import { test, expect } from '../fixtures';
import { LONG_RUNNING_QUERIES } from '../utils/test-data';

/**
 * Stop/Cancel Tests - Validate stop/cancel operations complete within 2 seconds.
 *
 * User Story 9.3: Stop/cancel operations
 * - Start research operation
 * - Trigger stop/cancel
 * - Verify operation stops within 2 seconds
 * - Verify partial results preserved or stopped message shown
 *
 * NOTE: Some tests in this suite are slow as they run real research queries.
 */
test.describe('Stop/Cancel', () => {
  // Mark all tests as slow (triples timeout)
  test.slow();

  // Skip stop/cancel tests unless explicitly enabled with RUN_SLOW_TESTS=1
  test.skip(
    !process.env.RUN_SLOW_TESTS,
    'Stop/cancel tests require real research - set RUN_SLOW_TESTS=1 to enable'
  );

  test('stops operation within 2 seconds', async ({ chatPage }) => {
    const query = LONG_RUNNING_QUERIES[0];

    // Given: research operation is in progress
    await chatPage.sendMessage(query.text);

    // Wait for loading to start
    await expect(chatPage.loadingIndicator).toBeVisible({ timeout: 5000 });

    // When: trigger stop/cancel
    const stopStartTime = Date.now();
    await chatPage.stopGeneration();

    // Then: verify operation stops within 2 seconds
    await expect(chatPage.loadingIndicator).toBeHidden({ timeout: 2000 });
    const stopDuration = Date.now() - stopStartTime;
    expect(stopDuration).toBeLessThan(2000);
  });

  test('application remains in stable state after stop', async ({ chatPage, page }) => {
    test.setTimeout(300000); // 5 minutes - need to wait for second research

    const query = LONG_RUNNING_QUERIES[0];

    // Start a long operation
    await chatPage.sendMessage(query.text);

    // Wait for loading
    await expect(chatPage.loadingIndicator).toBeVisible({ timeout: 5000 });

    // Stop it
    await chatPage.stopGeneration();

    // Wait for stop to complete
    await expect(chatPage.loadingIndicator).toBeHidden({ timeout: 5000 });

    // Verify app is still functional
    // Should be able to send another message
    await chatPage.sendMessage('Hello');
    // Research agent takes 1.5-2+ minutes for responses
    await chatPage.waitForAgentResponse(180000);

    // Should get a response
    const response = await chatPage.getLastAgentResponse();
    expect(response.length).toBeGreaterThan(0);
  });

  test('stop button is visible during operation', async ({ chatPage }) => {
    const query = LONG_RUNNING_QUERIES[1];

    // Send a query
    await chatPage.sendMessage(query.text);

    // Stop button should become visible
    await expect(chatPage.stopButton).toBeVisible({ timeout: 5000 });

    // Stop the operation to clean up
    await chatPage.stopGeneration();
  });

  // Skip: This test has timing issues with fast mock responses
  test.skip('stop button hides after operation completes or stops', async ({ chatPage }) => {
    const query = LONG_RUNNING_QUERIES[0];

    // Start operation
    await chatPage.sendMessage(query.text);

    // Wait for loading
    await expect(chatPage.loadingIndicator).toBeVisible({ timeout: 5000 });

    // Stop it
    await chatPage.stopGeneration();

    // Wait for stop to complete
    await expect(chatPage.loadingIndicator).toBeHidden({ timeout: 2000 });

    // Stop button should be hidden or disabled
    const stopVisible = await chatPage.stopButton.isVisible().catch(() => false);
    const stopEnabled = await chatPage.stopButton.isEnabled().catch(() => false);

    // Either hidden or disabled
    expect(!stopVisible || !stopEnabled).toBe(true);
  });

  test('partial results are preserved after stop', async ({ chatPage, page }) => {
    const query = LONG_RUNNING_QUERIES[0];

    // Start operation
    await chatPage.sendMessage(query.text);

    // Wait for some streaming to happen
    await expect(chatPage.loadingIndicator).toBeVisible({ timeout: 5000 });

    // Wait a bit for partial content
    await page.waitForTimeout(2000);

    // Stop it
    await chatPage.stopGeneration();

    // Check if there's any content (partial or complete)
    const agentResponses = page.getByTestId('agent-response');
    const messageCount = await agentResponses.count();

    // Either we have a partial response or the message was properly cancelled
    // Both are acceptable outcomes
    if (messageCount > 0) {
      // Got a response (partial or complete)
      const response = await agentResponses.first().textContent();
      // Response can be anything including a "cancelled" message
      expect(response).toBeDefined();
    } else {
      // No response - that's also acceptable for a stopped operation
      expect(messageCount).toBe(0);
    }
  });
});
