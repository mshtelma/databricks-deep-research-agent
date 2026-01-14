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
   * Navigate to the chat page and wait for it to be ready.
   */
  async goto(): Promise<void> {
    await this.page.goto('/');
    await this.waitForReady();
  }

  /**
   * Wait for the chat page to be fully loaded and ready for interaction.
   * This means the message input is visible and enabled (not loading).
   */
  async waitForReady(timeout: number = 30000): Promise<void> {
    // Wait for message input to be visible
    await this.messageInput.waitFor({ state: 'visible', timeout });
    // Wait for input to be enabled (not disabled during loading)
    await this.page.waitForFunction(
      () => {
        const input = document.querySelector('[data-testid="message-input"]');
        return input && !input.hasAttribute('disabled');
      },
      { timeout }
    );
  }

  /**
   * Send a message in the chat.
   * @param text The message text to send
   */
  async sendMessage(text: string): Promise<void> {
    await this.messageInput.fill(text);
    // Wait for Send button to become enabled (React state update)
    await this.page.waitForFunction(
      () => {
        const button = document.querySelector('[data-testid="send-button"]');
        return button && !button.hasAttribute('disabled');
      },
      { timeout: 5000 }
    );
    await this.sendButton.click();
  }

  /**
   * Wait for the agent to complete its response.
   * @param timeout Maximum wait time in milliseconds (default: 120000 = 2 minutes)
   *
   * NOTE: The loading indicator shows during the ENTIRE research phase (before synthesis).
   * We need to wait for EITHER the loading/streaming indicators to hide OR for an
   * agent-response to appear (which indicates completion).
   */
  async waitForAgentResponse(timeout: number = 120000): Promise<void> {
    // Strategy: Two-phase polling
    // Phase 1: Wait for research to START (loading or streaming indicator appears)
    // Phase 2: Wait for research to COMPLETE (agent response appears)

    const agentResponse = this.page.getByTestId('agent-response').first();
    const startTime = Date.now();
    const pollInterval = 1000; // Check every second

    // PHASE 1: Wait for research to start
    // At the beginning, both loading and streaming indicators may be hidden.
    // We need to wait for one of them to appear, OR for an early response.
    let researchStarted = false;
    while (Date.now() - startTime < timeout && !researchStarted) {
      // Check if agent response already exists (maybe cached or very fast)
      const responseVisible = await agentResponse.isVisible().catch(() => false);
      if (responseVisible) {
        return; // Success! Response already appeared
      }

      // Check if research has started (either indicator visible)
      const loadingVisible = await this.loadingIndicator.isVisible().catch(() => false);
      const streamingVisible = await this.streamingIndicator.isVisible().catch(() => false);

      if (loadingVisible || streamingVisible) {
        researchStarted = true;
        break;
      }

      // Wait a bit before checking again
      await this.page.waitForTimeout(pollInterval);
    }

    // PHASE 2: Wait for research to complete
    // Now poll until agent response appears, or indicators disappear after having been visible
    while (Date.now() - startTime < timeout) {
      // Check if agent response exists
      const responseVisible = await agentResponse.isVisible().catch(() => false);
      if (responseVisible) {
        return; // Success! Agent response appeared
      }

      // Check current indicator state
      const loadingVisible = await this.loadingIndicator.isVisible().catch(() => false);
      const streamingVisible = await this.streamingIndicator.isVisible().catch(() => false);

      // If research started but now both indicators are hidden, check for response
      if (researchStarted && !loadingVisible && !streamingVisible) {
        // Both indicators hidden after research started - check for response
        const finalCheck = await agentResponse.isVisible().catch(() => false);
        if (finalCheck) {
          return;
        }
        // Wait a bit more for potential late response (DOM update lag)
        await this.page.waitForTimeout(2000);
        const lastCheck = await agentResponse.isVisible().catch(() => false);
        if (lastCheck) {
          return;
        }
        // Still no response after indicators gone - might be a failure, but don't break
        // Keep polling in case of a slow DOM update
      }

      // Still waiting - sleep before next poll
      await this.page.waitForTimeout(pollInterval);
    }

    // Final assertion - if we get here, either timeout or unexpected state
    await expect(agentResponse).toBeVisible({ timeout: 5000 });
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

  /**
   * Select a research depth option.
   * @param depth The depth level: 'auto' | 'light' | 'medium' | 'extended'
   */
  async selectResearchDepth(depth: 'auto' | 'light' | 'medium' | 'extended'): Promise<void> {
    const depthLabels: Record<string, string> = {
      auto: 'Auto',
      light: 'Light',
      medium: 'Medium',
      extended: 'Extended',
    };
    const depthButton = this.page.getByRole('button', { name: depthLabels[depth] });
    await depthButton.click();
  }

  /**
   * Select a query mode.
   * @param mode The query mode: 'simple' | 'web_search' | 'deep_research'
   */
  async selectQueryMode(mode: 'simple' | 'web_search' | 'deep_research'): Promise<void> {
    const modeButton = this.page.getByTestId(`mode-${mode}`);
    await modeButton.click();
  }

  /**
   * Send a message with a specific query mode.
   * @param text The message text to send
   * @param mode The query mode to use
   */
  async sendMessageWithMode(
    text: string,
    mode: 'simple' | 'web_search' | 'deep_research'
  ): Promise<void> {
    await this.selectQueryMode(mode);
    await this.sendMessage(text);
  }
}
