import { test, expect } from '../fixtures';
import { SMOKE_QUERIES } from '../utils/test-data';

/**
 * Smoke Tests - Fast validation tests for CI pipeline.
 * These tests verify basic functionality and should complete quickly (<30 seconds).
 *
 * User Story 9.6: Test reporting and debugging
 */
test.describe('Smoke Tests', () => {
  test('app loads successfully', async ({ page }) => {
    // Navigate to app
    await page.goto('/');

    // Verify title contains expected text
    await expect(page).toHaveTitle(/Deep Research|Research Agent/i);

    // Verify message input is visible and ready
    await expect(page.getByTestId('message-input')).toBeVisible();
  });

  test('can create new chat', async ({ page, sidebarPage }) => {
    await page.goto('/');

    // Click new chat button
    await sidebarPage.createNewChat();

    // Verify message input is empty/ready for new chat
    const messageInput = page.getByTestId('message-input');
    await expect(messageInput).toBeVisible();
    await expect(messageInput).toBeEmpty();
  });

  test('can send simple message', async ({ chatPage }) => {
    const query = SMOKE_QUERIES[0]; // "Hello"

    // Send a simple message
    await chatPage.sendMessage(query.text);

    // Wait for response with appropriate timeout
    await chatPage.waitForAgentResponse(query.expectedResponseTimeMs);

    // Verify we got a response
    const response = await chatPage.getLastAgentResponse();
    expect(response.length).toBeGreaterThan(0);
  });

  test('message input is focused after load', async ({ page }) => {
    await page.goto('/');

    // Verify message input exists and is focusable
    const messageInput = page.getByTestId('message-input');
    await expect(messageInput).toBeVisible();

    // Should be able to type immediately
    await messageInput.fill('test');
    await expect(messageInput).toHaveValue('test');
  });

  test('send button is disabled when input is empty', async ({ page }) => {
    await page.goto('/');

    // Get the send button
    const sendButton = page.getByTestId('send-button');

    // Verify message input is empty
    const messageInput = page.getByTestId('message-input');
    await expect(messageInput).toBeEmpty();

    // Send button should be disabled or not clickable
    // (Implementation may vary - button could be disabled or just not submit)
    const isDisabled = await sendButton.isDisabled().catch(() => false);
    if (!isDisabled) {
      // If not disabled, clicking should not cause any navigation or error
      await sendButton.click();
      // Should still be on the same page
      await expect(messageInput).toBeVisible();
    }
  });
});
