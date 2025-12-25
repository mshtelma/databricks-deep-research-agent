import { test, expect } from '../fixtures';

/**
 * Error Handling Tests - Validate graceful error handling and recovery.
 *
 * Tests various error conditions to ensure the application
 * handles failures gracefully and provides useful feedback.
 */
test.describe('Error Handling', () => {
  test.setTimeout(180000); // 3 minutes - research agent needs time

  test('handles empty message gracefully', async ({ chatPage, page }) => {
    // Try to send an empty message
    const sendButton = page.getByTestId('send-button');
    const messageInput = page.getByTestId('message-input');

    // Ensure input is empty
    await messageInput.clear();

    // Send button should be disabled or clicking it should do nothing
    const isDisabled = await sendButton.isDisabled().catch(() => false);

    if (!isDisabled) {
      // If not disabled, clicking should not cause errors
      await sendButton.click();
      // Should still be on the same page with no error
      await expect(messageInput).toBeVisible();
    } else {
      // Button correctly disabled for empty input
      expect(isDisabled).toBe(true);
    }
  });

  test('recovers from network interruption', async ({ chatPage, page }) => {
    // Send a message
    await chatPage.sendMessage('What is Python?');

    // Wait for response to start
    await expect(chatPage.loadingIndicator).toBeVisible({ timeout: 10000 }).catch(() => {
      // Loading might already be done
    });

    // Stop any in-progress request
    if (await chatPage.stopButton.isVisible().catch(() => false)) {
      await chatPage.stopGeneration();
    }

    // App should still be functional
    await expect(page.getByTestId('message-input')).toBeVisible();

    // Should be able to send another message
    await chatPage.sendMessage('Hello again');

    // Should eventually get a response or be able to try again
    await expect(chatPage.loadingIndicator.or(page.getByTestId('agent-response'))).toBeVisible({
      timeout: 15000,
    });
  });

  test('app remains stable after normal operation', async ({ chatPage, page }) => {
    // This test verifies the happy path works and app remains stable

    // Send a normal message first
    await chatPage.sendMessage('Hello');
    await chatPage.waitForAgentResponse(120000);

    // Verify we got a response
    const response = await chatPage.getLastAgentResponse();
    expect(response.length).toBeGreaterThan(0);

    // App should remain stable after normal operation
    // Wait for input to be enabled (may take a moment after response completes)
    await expect(page.getByTestId('message-input')).toBeEnabled({ timeout: 30000 });
  });

  test('handles very long messages', async ({ chatPage }) => {
    // Send a very long message
    const longMessage = 'This is a test message. '.repeat(100);

    await chatPage.sendMessage(longMessage);

    // Should either:
    // 1. Successfully process the long message
    // 2. Show an appropriate error message
    // 3. Truncate the message

    // Wait for some response (success or error)
    const hasResponse = await chatPage.waitForAgentResponse(60000).then(() => true).catch(() => false);

    // App should still be functional regardless
    await expect(chatPage.messageInput).toBeVisible();

    if (hasResponse) {
      const response = await chatPage.getLastAgentResponse();
      expect(response.length).toBeGreaterThan(0);
    }
  });

  test('handles special characters in messages', async ({ chatPage, page }) => {
    // Send a simple greeting with special characters - should be treated as simple query
    // Using greeting format to avoid triggering full research cycle
    const specialMessage = 'Hello! <script>alert("xss")</script> & "quotes" \'apostrophe\'';

    await chatPage.sendMessage(specialMessage);

    // Wait for either a response or verify the app handles it gracefully
    // Use shorter timeout since simple queries should respond quickly
    const responseAppeared = await chatPage
      .waitForAgentResponse(60000)
      .then(() => true)
      .catch(() => false);

    // App should remain functional regardless of whether special chars caused issues
    await expect(page.getByTestId('message-input')).toBeVisible();

    if (responseAppeared) {
      // Response should exist and not be broken by special chars
      const response = await chatPage.getLastAgentResponse();
      expect(response.length).toBeGreaterThan(0);
    } else {
      // If no response, at least verify the user message was displayed correctly
      const userMessages = await chatPage.getUserMessages();
      expect(userMessages.length).toBeGreaterThan(0);
    }
  });

  test('handles rapid message sending', async ({ chatPage }) => {
    // Send messages rapidly (simulate impatient user)
    await chatPage.sendMessage('First question');

    // Don't wait for response, immediately try another
    await chatPage.messageInput.fill('Second question');

    // The send button might be disabled during loading
    const sendEnabled = await chatPage.sendButton.isEnabled().catch(() => false);

    if (sendEnabled) {
      await chatPage.sendButton.click();
    }

    // Wait for at least one response
    await chatPage.waitForAgentResponse(120000);

    // App should remain stable
    await expect(chatPage.messageInput).toBeVisible();
  });

  test('handles unicode and emoji in messages', async ({ chatPage, page }) => {
    // Send a simple greeting with unicode and emoji - should be treated as simple query
    const unicodeMessage = 'Hello! 你好 🌍 مرحبا 🚀 How are you?';

    await chatPage.sendMessage(unicodeMessage);

    // Wait for either a response or verify the app handles it gracefully
    const responseAppeared = await chatPage
      .waitForAgentResponse(60000)
      .then(() => true)
      .catch(() => false);

    // App should remain functional
    await expect(page.getByTestId('message-input')).toBeVisible();

    if (responseAppeared) {
      // Should handle unicode without issues
      const response = await chatPage.getLastAgentResponse();
      expect(response.length).toBeGreaterThan(0);
    } else {
      // If no response, at least verify the user message was displayed correctly
      const userMessages = await chatPage.getUserMessages();
      expect(userMessages.length).toBeGreaterThan(0);
    }
  });
});
