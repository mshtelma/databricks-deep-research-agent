import { type Locator, type Page, expect } from '@playwright/test';

/**
 * Page Object Model for the Chat interface.
 * Encapsulates selectors and common actions for chat interactions.
 */
export class ChatPage {
  readonly page: Page;
  readonly messageInput: Locator;
  readonly sendButton: Locator;
  readonly stopButton: Locator;
  readonly loadingIndicator: Locator;
  readonly streamingIndicator: Locator;
  readonly messageList: Locator;
  readonly regenerateButton: Locator;

  constructor(page: Page) {
    this.page = page;
    this.messageInput = page.getByTestId('message-input');
    this.sendButton = page.getByTestId('send-button');
    this.stopButton = page.getByTestId('stop-button');
    this.loadingIndicator = page.getByTestId('loading-indicator');
    this.streamingIndicator = page.getByTestId('streaming-indicator');
    this.messageList = page.getByTestId('message-list');
    this.regenerateButton = page.getByTestId('regenerate-response');
  }

  /**
   * Navigate to the chat page.
   */
  async goto(): Promise<void> {
    await this.page.goto('/');
  }

  /**
   * Send a message in the chat.
   * @param text The message text to send
   */
  async sendMessage(text: string): Promise<void> {
    await this.messageInput.fill(text);
    await this.sendButton.click();
  }

  /**
   * Wait for the agent to complete its response.
   * @param timeout Maximum wait time in milliseconds (default: 120000 = 2 minutes)
   */
  async waitForAgentResponse(timeout: number = 120000): Promise<void> {
    // Wait for loading indicator to disappear
    await expect(this.loadingIndicator).toBeHidden({ timeout });

    // Wait for streaming indicator to disappear (if present)
    try {
      await expect(this.streamingIndicator).toBeHidden({ timeout });
    } catch {
      // Streaming indicator might not be present for simple queries
    }

    // Verify at least one agent response exists
    await expect(this.page.getByTestId('agent-response').first()).toBeVisible({ timeout });
  }

  /**
   * Get the text content of the last agent response.
   * @returns The text content of the most recent agent response
   */
  async getLastAgentResponse(): Promise<string> {
    const responses = this.page.getByTestId('agent-response');
    const count = await responses.count();
    if (count === 0) {
      throw new Error('No agent responses found');
    }
    const lastResponse = responses.nth(count - 1);
    return (await lastResponse.textContent()) ?? '';
  }

  /**
   * Get all user messages in the chat.
   * @returns Array of user message text content
   */
  async getUserMessages(): Promise<string[]> {
    const messages = this.page.getByTestId('user-message');
    const count = await messages.count();
    const texts: string[] = [];
    for (let i = 0; i < count; i++) {
      const text = await messages.nth(i).textContent();
      texts.push(text ?? '');
    }
    return texts;
  }

  /**
   * Get all agent responses in the chat.
   * @returns Array of agent response text content
   */
  async getAgentResponses(): Promise<string[]> {
    const responses = this.page.getByTestId('agent-response');
    const count = await responses.count();
    const texts: string[] = [];
    for (let i = 0; i < count; i++) {
      const text = await responses.nth(i).textContent();
      texts.push(text ?? '');
    }
    return texts;
  }

  /**
   * Edit a previous message by index.
   * @param index The index of the message to edit (0-based)
   * @param newText The new text for the message
   */
  async editMessage(index: number, newText: string): Promise<void> {
    const userMessages = this.page.getByTestId('user-message');
    const message = userMessages.nth(index);

    // Click the edit button for this message
    const editButton = message.getByTestId(`edit-message-${index}`);
    await editButton.click();

    // Clear and fill the new text
    await this.messageInput.clear();
    await this.messageInput.fill(newText);
    await this.sendButton.click();
  }

  /**
   * Click the regenerate button to get a new response.
   */
  async regenerate(): Promise<void> {
    await this.regenerateButton.click();
  }

  /**
   * Click the stop button to cancel the current operation.
   */
  async stopGeneration(): Promise<void> {
    await this.stopButton.click();
  }

  /**
   * Check if the chat is currently loading/processing.
   * @returns True if loading indicator is visible
   */
  async isLoading(): Promise<boolean> {
    return this.loadingIndicator.isVisible();
  }

  /**
   * Check if the chat is currently streaming a response.
   * @returns True if streaming indicator is visible
   */
  async isStreaming(): Promise<boolean> {
    return this.streamingIndicator.isVisible();
  }

  /**
   * Get the count of citations in the last agent response.
   * @returns Number of citations found
   */
  async getCitationCount(): Promise<number> {
    const citations = this.page.getByTestId('citation');
    return citations.count();
  }
}
