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
  readonly errorAlert: Locator;

  constructor(page: Page) {
    this.page = page;
    this.messageInput = page.getByTestId('message-input');
    this.sendButton = page.getByTestId('send-button');
    this.stopButton = page.getByTestId('stop-button');
    this.loadingIndicator = page.getByTestId('loading-indicator');
    this.streamingIndicator = page.getByTestId('streaming-indicator');
    this.messageList = page.getByTestId('message-list');
    this.regenerateButton = page.getByTestId('regenerate-response');
    this.errorAlert = page.getByTestId('research-error');
  }

  /**
   * Check if an agent-response element has meaningful text content.
   * The backend creates agent messages with NULL content at research start,
   * which renders as an empty agent-response div. We must exclude these
   * placeholders to avoid returning prematurely from waitForAgentResponse.
   */
  private async hasAgentResponseWithContent(minLength: number = 20): Promise<boolean> {
    const responses = this.page.getByTestId('agent-response');
    const count = await responses.count();
    for (let i = 0; i < count; i++) {
      const text = (await responses.nth(i).textContent().catch(() => '')) ?? '';
      if (text.trim().length >= minLength) return true;
    }
    return false;
  }

  /**
   * Check if a persisted (non-streaming) agent response is visible WITH content.
   *
   * During streaming, AgentMessage renders inside streaming-indicator with
   * the same data-testid="agent-response". This method excludes it by checking
   * that streaming-indicator is NOT visible before accepting agent-response.
   *
   * Also excludes empty placeholder messages (backend creates agent message
   * with NULL content at research start, before synthesis).
   *
   * The "completed-stream bridge" (MessageList.tsx:181) also renders agent-response
   * outside streaming-indicator — this is intentionally detected as persisted
   * since it has full content and streaming is complete.
   */
  private async isPersistedResponseVisible(): Promise<boolean> {
    const responseVisible = await this.page
      .getByTestId('agent-response')
      .first()
      .isVisible()
      .catch(() => false);
    if (!responseVisible) return false;
    const streamingActive = await this.streamingIndicator.isVisible().catch(() => false);
    if (streamingActive) return false;
    // Verify at least one response has actual content (not an empty placeholder)
    return this.hasAgentResponseWithContent();
  }

  /**
   * Get the count of persisted (non-streaming) agent responses WITH content.
   *
   * During streaming, one agent-response element exists inside streaming-indicator.
   * This method subtracts 1 from the raw count when streaming is active to exclude it.
   * Also excludes empty placeholder messages (NULL content from backend).
   */
  private async getPersistedResponseCount(): Promise<number> {
    const responses = this.page.getByTestId('agent-response');
    const rawCount = await responses.count();
    const streamingActive = await this.streamingIndicator.isVisible().catch(() => false);

    // Count only responses with actual content
    let contentCount = 0;
    for (let i = 0; i < rawCount; i++) {
      const text = (await responses.nth(i).textContent().catch(() => '')) ?? '';
      if (text.trim().length >= 20) contentCount++;
    }

    return streamingActive ? Math.max(0, contentCount - 1) : contentCount;
  }

  /**
   * Check if a research error alert is visible.
   * The error banner appears when the backend sends an error event.
   */
  private async isErrorVisible(): Promise<boolean> {
    return this.errorAlert.isVisible().catch(() => false);
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
  /**
   * Check if the research status shows "Complete" (green badge in the UI).
   * This indicates the backend finished processing even if the response isn't rendered.
   */
  private async isResearchStatusComplete(): Promise<boolean> {
    // The status indicator shows "Complete" text when research finishes
    const completeIndicator = this.page.locator('text=Complete').first();
    return completeIndicator.isVisible().catch(() => false);
  }

  async waitForAgentResponse(timeout: number = 120000): Promise<void> {
    const startTime = Date.now();
    const pollInterval = 1000;
    let hasReloaded = false;

    // PHASE 1: Wait for research to START (indicator appears) or instant response.
    let researchStarted = false;
    while (Date.now() - startTime < timeout && !researchStarted) {
      if (await this.isPersistedResponseVisible()) {
        return; // Cached or instant response (no streaming involved)
      }
      const loadingVisible = await this.loadingIndicator.isVisible().catch(() => false);
      const streamingVisible = await this.streamingIndicator.isVisible().catch(() => false);
      if (loadingVisible || streamingVisible) {
        researchStarted = true;
        break;
      }
      if (await this.isErrorVisible()) {
        researchStarted = true;
        break;
      }
      // Check if research already completed (fast simple queries can finish
      // before the test catches loading/streaming indicators)
      if (await this.isResearchStatusComplete()) {
        researchStarted = true;
        break;
      }
      await this.page.waitForTimeout(pollInterval);
    }

    // PHASE 2: Wait for research to COMPLETE (persisted response appears).
    let errorFirstSeenAt: number | null = null;
    let stallStartedAt: number | null = null;

    while (Date.now() - startTime < timeout) {
      if (await this.isPersistedResponseVisible()) {
        return; // Streaming finished, persisted or bridge response visible
      }

      const loadingVisible = await this.loadingIndicator.isVisible().catch(() => false);
      const streamingVisible = await this.streamingIndicator.isVisible().catch(() => false);
      const errorVisible = await this.isErrorVisible();

      // Terminal: error visible, no active indicators, no response
      if (errorVisible && !loadingVisible && !streamingVisible) {
        // Brief wait for DOM transition (bridge may appear)
        await this.page.waitForTimeout(2000);
        if (await this.isPersistedResponseVisible()) {
          return; // Partial response available via bridge
        }
        throw new Error(
          'Research failed — error alert visible but no agent response rendered. ' +
            'Backend may have errored before synthesis started.'
        );
      }

      // Defensive: error visible WITH streaming still active (Layer 1 should prevent this,
      // but handles recoverable errors or if Layer 1 fix doesn't apply)
      if (errorVisible && streamingVisible) {
        if (!errorFirstSeenAt) errorFirstSeenAt = Date.now();
        if (Date.now() - errorFirstSeenAt > 30000) {
          throw new Error(
            'Research error with streaming stuck for 30s+ — streaming-indicator ' +
              'still visible alongside error alert.'
          );
        }
      } else {
        errorFirstSeenAt = null; // Reset if error disappears or streaming stops
      }

      // Stall detection: indicators gone, no response, no error.
      // This happens when the backend completed but the frontend didn't render the response
      // (e.g., SSE closed before persistence_completed, TanStack Query cache stale).
      // Fix: reload the page to force fresh data fetch from the API.
      if (researchStarted && !loadingVisible && !streamingVisible && !errorVisible) {
        if (!stallStartedAt) {
          stallStartedAt = Date.now();
        }

        // Wait 2s for DOM transition first
        await this.page.waitForTimeout(2000);
        if (await this.isPersistedResponseVisible()) {
          return;
        }

        // After 10s of stall (or if "Complete" status visible), reload page to force refetch
        const stallDuration = Date.now() - stallStartedAt;
        const statusComplete = await this.isResearchStatusComplete();
        if (!hasReloaded && (stallDuration > 10000 || statusComplete)) {
          hasReloaded = true;
          await this.page.reload({ waitUntil: 'networkidle' });
          await this.page.waitForTimeout(3000); // Wait for React hydration + data fetch
          if (await this.isPersistedResponseVisible()) {
            return;
          }
        }
      } else {
        stallStartedAt = null; // Reset if indicators come back
      }

      await this.page.waitForTimeout(pollInterval);
    }

    // Timeout — produce actionable error
    const streamingActive = await this.streamingIndicator.isVisible().catch(() => false);
    const errorVisible = await this.isErrorVisible();
    if (streamingActive) {
      throw new Error(
        `Agent response still streaming after ${timeout}ms timeout. ` +
          `streaming-indicator is visible, error=${errorVisible} — research may need more time.`
      );
    }
    await expect(this.page.getByTestId('agent-response').first()).toBeVisible({ timeout: 5000 });
  }

  /**
   * Get the text content of the last agent response.
   * Prefers the content-specific locator (excludes timestamps, buttons, sections).
   *
   * Includes retry logic to handle the bridge→DB message DOM transition race:
   * when streaming completes, the bridge message may briefly disappear as the
   * DB-persisted message renders, causing textContent to return empty during
   * the React re-render cycle.
   *
   * @param maxRetries Maximum number of retry attempts (default: 10)
   * @param retryInterval Milliseconds between retries (default: 1000)
   * @returns The text content of the most recent agent response
   */
  async getLastAgentResponse(maxRetries: number = 10, retryInterval: number = 1000): Promise<string> {
    for (let attempt = 0; attempt <= maxRetries; attempt++) {
      const responses = this.page.getByTestId('agent-response');
      const count = await responses.count();

      if (count === 0) {
        if (attempt < maxRetries) {
          await this.page.waitForTimeout(retryInterval);
          continue;
        }
        throw new Error('No agent responses found after retries');
      }

      const lastResponse = responses.nth(count - 1);

      // Prefer the content-specific locator (excludes timestamps, buttons, sections)
      const contentLocator = lastResponse.getByTestId('agent-response-content');
      let text = '';
      if (await contentLocator.count() > 0) {
        text = (await contentLocator.textContent()) ?? '';
      } else {
        text = (await lastResponse.textContent()) ?? '';
      }

      // If we got non-empty content, return it immediately
      if (text.trim().length > 0) {
        return text;
      }

      // Content was empty — likely a DOM transition (bridge→DB message). Retry.
      if (attempt < maxRetries) {
        await this.page.waitForTimeout(retryInterval);
      }
    }

    // All retries exhausted — return whatever we have (may be empty)
    const responses = this.page.getByTestId('agent-response');
    const count = await responses.count();
    if (count === 0) {
      throw new Error('No agent responses found after retries');
    }
    const lastResponse = responses.nth(count - 1);
    const contentLocator = lastResponse.getByTestId('agent-response-content');
    if (await contentLocator.count() > 0) {
      return (await contentLocator.textContent()) ?? '';
    }
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
   * Prefers the content-specific locator (excludes timestamps, buttons, sections).
   * Includes a brief wait if any response has empty content (DOM transition).
   * @returns Array of agent response text content
   */
  async getAgentResponses(): Promise<string[]> {
    // Allow a brief retry window for DOM transitions
    for (let attempt = 0; attempt < 5; attempt++) {
      const responses = this.page.getByTestId('agent-response');
      const count = await responses.count();
      const texts: string[] = [];
      let hasEmpty = false;

      for (let i = 0; i < count; i++) {
        const response = responses.nth(i);
        const contentLocator = response.getByTestId('agent-response-content');
        let text = '';
        if (await contentLocator.count() > 0) {
          text = (await contentLocator.textContent()) ?? '';
        } else {
          text = (await response.textContent()) ?? '';
        }
        texts.push(text);
        if (text.trim().length === 0) hasEmpty = true;
      }

      if (!hasEmpty || attempt === 4) {
        return texts;
      }
      await this.page.waitForTimeout(1000);
    }

    // Fallback (shouldn't reach here)
    return [];
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

  // ==================== Parallel Testing Methods ====================

  /**
   * Extract the chat ID from the current URL.
   * @returns The chat ID if present, null otherwise
   */
  async getChatIdFromUrl(): Promise<string | null> {
    const url = this.page.url();
    // URL pattern: /chat/:chatId or /c/:chatId
    const match = url.match(/\/(?:chat|c)\/([a-zA-Z0-9-]+)/);
    return match ? match[1] : null;
  }

  /**
   * Get the count of user and agent messages in the chat.
   * @returns Object with user and agent message counts
   */
  async getMessageCount(): Promise<{ user: number; agent: number }> {
    const userMessages = this.page.getByTestId('user-message');
    const agentResponses = this.page.getByTestId('agent-response');

    const userCount = await userMessages.count();
    const agentCount = await agentResponses.count();

    return { user: userCount, agent: agentCount };
  }

  /**
   * Wait for a specific number of agent responses.
   * Useful for follow-up scenarios where we expect multiple responses.
   * @param count The expected number of agent responses
   * @param timeout Maximum wait time in milliseconds
   */
  async waitForAgentResponseCount(count: number, timeout: number = 120000): Promise<void> {
    const startTime = Date.now();
    const pollInterval = 1000;
    let stallStartedAt: number | null = null;
    let hasReloaded = false;

    while (Date.now() - startTime < timeout) {
      const currentCount = await this.getPersistedResponseCount();
      if (currentCount >= count) {
        return;
      }
      // Error terminal check
      const errorVisible = await this.isErrorVisible();
      const streamingVisible = await this.streamingIndicator.isVisible().catch(() => false);
      const loadingVisible = await this.loadingIndicator.isVisible().catch(() => false);
      if (errorVisible && !loadingVisible && !streamingVisible) {
        await this.page.waitForTimeout(2000);
        const finalCount = await this.getPersistedResponseCount();
        if (finalCount >= count) return;
        throw new Error(
          `Research failed — error visible, expected ${count} responses, have ${finalCount}.`
        );
      }

      // Stall detection: no indicators, no response, research likely complete.
      // Use only time-based threshold — do NOT check isResearchStatusComplete() as
      // it may detect the previous research's "Complete" badge and trigger premature reload.
      if (!loadingVisible && !streamingVisible && !errorVisible) {
        if (!stallStartedAt) stallStartedAt = Date.now();
        const stallDuration = Date.now() - stallStartedAt;
        if (!hasReloaded && stallDuration > 15000) {
          hasReloaded = true;
          await this.page.reload({ waitUntil: 'networkidle' });
          await this.page.waitForTimeout(3000);
          const reloadCount = await this.getPersistedResponseCount();
          if (reloadCount >= count) return;
        }
      } else {
        stallStartedAt = null;
      }

      await this.page.waitForTimeout(pollInterval);
    }

    const finalCount = await this.getPersistedResponseCount();
    throw new Error(
      `Expected ${count} persisted agent responses within ${timeout}ms, got ${finalCount}`
    );
  }

  /**
   * Wait for the Nth agent response to appear (1-indexed).
   * @param n The response number to wait for (1 = first, 2 = second, etc.)
   * @param timeout Maximum wait time in milliseconds
   */
  async waitForNthAgentResponse(n: number, timeout: number = 120000): Promise<void> {
    const startTime = Date.now();
    const pollInterval = 1000;
    let researchStarted = false;
    let errorFirstSeenAt: number | null = null;
    let stallStartedAt: number | null = null;
    let hasReloaded = false;

    while (Date.now() - startTime < timeout) {
      const currentCount = await this.getPersistedResponseCount();
      if (currentCount >= n) {
        return;
      }
      const loadingVisible = await this.loadingIndicator.isVisible().catch(() => false);
      const streamingVisible = await this.streamingIndicator.isVisible().catch(() => false);
      const errorVisible = await this.isErrorVisible();

      if (loadingVisible || streamingVisible) {
        researchStarted = true;
      }

      // NOTE: Do NOT use isResearchStatusComplete() here — it detects the PREVIOUS
      // research's "Complete" badge, which would trigger premature stall detection
      // for follow-up queries. Only detect research started via actual indicators.

      // Error terminal: no indicators, error visible, count not reached
      if (errorVisible && !loadingVisible && !streamingVisible) {
        await this.page.waitForTimeout(2000);
        const finalCount = await this.getPersistedResponseCount();
        if (finalCount >= n) return;
        throw new Error(
          `Research failed — error alert visible, expected ${n} responses but have ${finalCount}.`
        );
      }

      // Streaming + error timeout
      if (errorVisible && streamingVisible) {
        if (!errorFirstSeenAt) errorFirstSeenAt = Date.now();
        if (Date.now() - errorFirstSeenAt > 30000) {
          const finalCount = await this.getPersistedResponseCount();
          throw new Error(
            `Error with streaming stuck 30s+, expected ${n} responses, have ${finalCount}.`
          );
        }
      } else {
        errorFirstSeenAt = null;
      }

      // Stall detection with page reload fallback.
      // Only activate when we've seen indicators for the CURRENT request go away.
      // For follow-ups (n > 1), also use elapsed time as a fallback signal since
      // fast queries may complete before the first poll catches indicators.
      const elapsed = Date.now() - startTime;
      const timeFallback = !researchStarted && elapsed > 30000;
      if ((researchStarted || timeFallback) && !loadingVisible && !streamingVisible && !errorVisible) {
        if (!stallStartedAt) stallStartedAt = Date.now();

        await this.page.waitForTimeout(2000);
        const finalCount = await this.getPersistedResponseCount();
        if (finalCount >= n) return;

        const stallDuration = Date.now() - stallStartedAt;
        // Use 15s stall threshold (not isResearchStatusComplete which sees old status)
        if (!hasReloaded && stallDuration > 15000) {
          hasReloaded = true;
          await this.page.reload({ waitUntil: 'networkidle' });
          await this.page.waitForTimeout(3000);
          const reloadCount = await this.getPersistedResponseCount();
          if (reloadCount >= n) return;
        }
      } else {
        stallStartedAt = null;
      }

      await this.page.waitForTimeout(pollInterval);
    }

    // Timeout — produce actionable error
    const finalCount = await this.getPersistedResponseCount();
    const rawCount = await this.page.getByTestId('agent-response').count();
    const streamingActive = await this.streamingIndicator.isVisible().catch(() => false);
    const errorVisible = await this.isErrorVisible();
    throw new Error(
      `Expected ${n} persisted agent responses within ${timeout}ms, ` +
        `got ${finalCount} (raw: ${rawCount}, streaming: ${streamingActive}, error: ${errorVisible})`
    );
  }
}
