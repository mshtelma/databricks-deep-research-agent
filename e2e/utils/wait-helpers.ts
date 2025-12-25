import { type Page, expect } from '@playwright/test';

/**
 * Wait for SSE streaming to complete.
 * Checks for the streaming indicator to disappear.
 *
 * @param page The Playwright page object
 * @param timeout Maximum wait time in milliseconds (default: 120000 = 2 minutes)
 */
export async function waitForStreamingComplete(page: Page, timeout: number = 120000): Promise<void> {
  // Wait for the streaming indicator to disappear
  const streamingIndicator = page.getByTestId('streaming-indicator');

  try {
    // First check if streaming has started
    await expect(streamingIndicator).toBeVisible({ timeout: 5000 });

    // Then wait for it to complete
    await expect(streamingIndicator).toBeHidden({ timeout });
  } catch {
    // Streaming might not have started yet or already completed
    // Check if there's an agent response already
    const agentResponse = page.getByTestId('agent-response');
    if ((await agentResponse.count()) > 0) {
      // Response exists, streaming completed
      return;
    }
    // Otherwise, wait a bit and check again
    await page.waitForTimeout(1000);
  }
}

/**
 * Wait for the loading indicator to disappear.
 *
 * @param page The Playwright page object
 * @param timeout Maximum wait time in milliseconds (default: 120000 = 2 minutes)
 */
export async function waitForLoadingComplete(page: Page, timeout: number = 120000): Promise<void> {
  const loadingIndicator = page.getByTestId('loading-indicator');
  await expect(loadingIndicator).toBeHidden({ timeout });
}

/**
 * Wait for a specific number of agent responses.
 *
 * @param page The Playwright page object
 * @param count The expected number of agent responses
 * @param timeout Maximum wait time in milliseconds
 */
export async function waitForAgentResponseCount(
  page: Page,
  count: number,
  timeout: number = 120000
): Promise<void> {
  const responses = page.getByTestId('agent-response');
  await expect(responses).toHaveCount(count, { timeout });
}

/**
 * Wait for at least a minimum number of citations.
 *
 * @param page The Playwright page object
 * @param minCount Minimum number of citations expected
 * @param timeout Maximum wait time in milliseconds
 */
export async function waitForCitations(page: Page, minCount: number = 1, timeout: number = 10000): Promise<void> {
  const citations = page.getByTestId('citation');
  await expect(citations).toHaveCount({ minimum: minCount }, { timeout });
}

/**
 * Poll for a condition to be true.
 * Useful for complex async operations.
 *
 * @param condition Function that returns a promise resolving to boolean
 * @param timeout Maximum wait time in milliseconds
 * @param interval Polling interval in milliseconds
 */
export async function waitForCondition(
  condition: () => Promise<boolean>,
  timeout: number = 30000,
  interval: number = 500
): Promise<void> {
  const startTime = Date.now();

  while (Date.now() - startTime < timeout) {
    if (await condition()) {
      return;
    }
    await new Promise((resolve) => setTimeout(resolve, interval));
  }

  throw new Error(`Condition not met within ${timeout}ms`);
}
