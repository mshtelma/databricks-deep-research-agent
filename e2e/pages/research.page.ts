import { type Locator, type Page, expect } from '@playwright/test';

/**
 * Page Object Model for the Research/Reasoning Panel.
 * Encapsulates selectors and common actions for research progress interactions.
 */
export class ResearchPage {
  readonly page: Page;
  readonly reasoningPanel: Locator;

  constructor(page: Page) {
    this.page = page;
    this.reasoningPanel = page.getByTestId('reasoning-panel');
  }

  /**
   * Check if the reasoning panel is visible.
   * @returns True if the reasoning panel is visible
   */
  async isReasoningVisible(): Promise<boolean> {
    return this.reasoningPanel.isVisible();
  }

  /**
   * Wait for the reasoning panel to appear.
   * @param timeout Maximum wait time in milliseconds
   */
  async waitForReasoning(timeout: number = 30000): Promise<void> {
    await expect(this.reasoningPanel).toBeVisible({ timeout });
  }

  /**
   * Get the reasoning steps displayed in the panel.
   * @returns Array of reasoning step text content
   */
  async getReasoningSteps(): Promise<string[]> {
    const steps = this.reasoningPanel.locator('[data-testid^="reasoning-step-"]');
    const count = await steps.count();
    const texts: string[] = [];
    for (let i = 0; i < count; i++) {
      const text = await steps.nth(i).textContent();
      texts.push(text ?? '');
    }
    return texts;
  }

  /**
   * Expand the reasoning panel if it's collapsed.
   */
  async expandReasoning(): Promise<void> {
    const expandButton = this.reasoningPanel.getByRole('button', { name: /expand|show/i });
    if (await expandButton.isVisible()) {
      await expandButton.click();
    }
  }

  /**
   * Collapse the reasoning panel if it's expanded.
   */
  async collapseReasoning(): Promise<void> {
    const collapseButton = this.reasoningPanel.getByRole('button', { name: /collapse|hide/i });
    if (await collapseButton.isVisible()) {
      await collapseButton.click();
    }
  }

  /**
   * Get the current research status (e.g., "Searching...", "Analyzing...", "Complete").
   * @returns The status text if visible
   */
  async getResearchStatus(): Promise<string | null> {
    const status = this.reasoningPanel.locator('[data-testid="research-status"]');
    if (await status.isVisible()) {
      return status.textContent();
    }
    return null;
  }

  /**
   * Get the count of sources/citations found during research.
   * @returns Number of sources found
   */
  async getSourceCount(): Promise<number> {
    const sources = this.page.getByTestId('citation');
    return sources.count();
  }
}
