import { type Locator, type Page, expect } from '@playwright/test';

/**
 * Page Object Model for the Custom Agent selector UI.
 *
 * Defines the behavioral contract for agent selection. All locators use
 * data-testid attributes that the frontend must implement. If the UI
 * pattern changes (dropdown → dialog), only this page object changes.
 */
export class CustomAgentPage {
  readonly page: Page;

  // Agent selector trigger and dropdown
  readonly selectorTrigger: Locator;
  readonly selectorDropdown: Locator;

  // Selection state indicators
  readonly selectedName: Locator;
  readonly selectedBadge: Locator;
  readonly clearButton: Locator;

  // Source scope indicator
  readonly sourceScopeIndicator: Locator;

  constructor(page: Page) {
    this.page = page;

    this.selectorTrigger = page.getByTestId('agent-selector-trigger');
    this.selectorDropdown = page.getByTestId('agent-selector-dropdown');
    this.selectedName = page.getByTestId('agent-selected-name');
    this.selectedBadge = page.getByTestId('agent-selected-badge');
    this.clearButton = page.getByTestId('agent-clear-button');
    this.sourceScopeIndicator = page.getByTestId('agent-source-scope-indicator');
  }

  /**
   * Wait for the agent selector to be ready (visible in the UI).
   * Provides a diagnostic message on timeout to help identify missing data-testid attrs.
   */
  async waitForReady(timeout: number = 10_000): Promise<void> {
    // Phase 1: Wait for chat app to fully load (input visible + enabled)
    await this.page.waitForFunction(
      () => {
        const input = document.querySelector('[data-testid="message-input"]');
        if (!input) return false;
        const style = window.getComputedStyle(input);
        return style.display !== 'none' && style.visibility !== 'hidden'
          && !input.hasAttribute('disabled');
      },
      { timeout }
    );

    // Phase 2: Ensure deep_research mode is active (agent selector only renders in this mode)
    // The mode selector may be hidden by plugin config — in that case, defaultQueryMode
    // is 'deep_research' (MessageInput.tsx:109) and agent selector is already visible.
    const modeButton = this.page.getByTestId('mode-deep_research');
    const modeButtonVisible = await modeButton.isVisible().catch(() => false);
    if (modeButtonVisible) {
      await modeButton.click();
      // click() persists to localStorage via useQueryMode hook,
      // so subsequent page.reload() calls keep deep_research mode
    }

    // Phase 3: Verify agent selector trigger appeared
    await expect(
      this.selectorTrigger,
      'Agent selector trigger (data-testid="agent-selector-trigger") not found — ' +
        'ensure queryMode is "deep_research" and the frontend renders this element',
    ).toBeVisible({ timeout });
  }

  // ---------------------------------------------------------------------------
  // Dropdown interactions
  // ---------------------------------------------------------------------------

  /**
   * Close the agent selector dropdown using a resilient multi-strategy approach.
   * 1. Try Escape key (works when frontend dismiss handler is present)
   * 2. Fall back to clicking the trigger button (toggles closed)
   * 3. If still open, give up gracefully (avoid cascading failures)
   */
  private async closeSelector(): Promise<void> {
    const isVisible = await this.selectorDropdown.isVisible().catch(() => false);
    if (!isVisible) return;

    // Primary: Escape key
    await this.page.keyboard.press('Escape');
    try {
      await expect(this.selectorDropdown).toBeHidden({ timeout: 2_000 });
      return;
    } catch { /* Escape didn't work, try fallback */ }

    // Fallback: click trigger to toggle closed
    await this.selectorTrigger.click();
    await expect(this.selectorDropdown).toBeHidden({ timeout: 2_000 }).catch(() => {
      console.warn('[closeSelector] Dropdown still visible after Escape + trigger click');
    });
  }

  /** Open the agent selector dropdown. */
  async openSelector(): Promise<void> {
    // Wait for any previous portal to be fully removed from DOM
    // (not just CSS-hidden — ensures Radix unmount + focus cleanup is complete)
    await this.selectorDropdown.waitFor({ state: 'detached', timeout: 2_000 }).catch(() => {});
    await this.selectorTrigger.click();
    await expect(this.selectorDropdown).toBeVisible({ timeout: 5_000 });
  }

  /** Select an agent by its ID from the dropdown. */
  async selectAgent(agentId: string): Promise<void> {
    await this.openSelector();
    const option = this.page.getByTestId(`agent-option-${agentId}`);
    await expect(option).toBeVisible({ timeout: 15_000 });
    await option.click();
    // Wait for dropdown to close
    await expect(this.selectorDropdown).toBeHidden({ timeout: 5_000 });
  }

  /** Select an agent by name (searches visible text in dropdown options). */
  async selectAgentByName(name: string): Promise<void> {
    await this.openSelector();
    const option = this.selectorDropdown.locator(`text="${name}"`);
    await expect(option).toBeVisible({ timeout: 5_000 });
    await option.click();
    await expect(this.selectorDropdown).toBeHidden({ timeout: 5_000 });
  }

  /** Clear the current agent selection. */
  async clearSelection(): Promise<void> {
    if (await this.clearButton.isVisible()) {
      await this.clearButton.click();
      // Wait for React to re-render and remove the badge
      await expect(this.selectedBadge).toBeHidden({ timeout: 5_000 });
    }
  }

  // ---------------------------------------------------------------------------
  // Read-only state queries
  // ---------------------------------------------------------------------------

  /** Get the display name of the currently selected agent, or null if none. */
  async getSelectedAgentName(): Promise<string | null> {
    if (await this.selectedName.isVisible()) {
      return this.selectedName.textContent();
    }
    return null;
  }

  /** Check whether any agent is currently selected. */
  async isAgentSelected(): Promise<boolean> {
    try {
      await expect(this.selectedBadge).toBeVisible({ timeout: 15_000 });
      return true;
    } catch {
      return false;
    }
  }

  /** Get the source scope indicator text (e.g., "enterprise_only"). */
  async getSourceScopeText(): Promise<string | null> {
    if (await this.sourceScopeIndicator.isVisible()) {
      return this.sourceScopeIndicator.textContent();
    }
    return null;
  }

  // ---------------------------------------------------------------------------
  // Dropdown enumeration
  // ---------------------------------------------------------------------------

  /** Get all agent option elements from the open dropdown. */
  async getAgentOptions(): Promise<Locator[]> {
    await this.openSelector();
    const options = this.selectorDropdown.locator('[data-testid^="agent-option-"]');
    const count = await options.count();
    const result: Locator[] = [];
    for (let i = 0; i < count; i++) {
      result.push(options.nth(i));
    }
    return result;
  }

  /** Check whether a specific agent is listed in the dropdown. */
  async hasAgentOption(agentId: string): Promise<boolean> {
    await this.openSelector();
    const option = this.page.getByTestId(`agent-option-${agentId}`);
    try {
      await expect(option).toBeVisible({ timeout: 15_000 });
      await this.closeSelector();
      return true;
    } catch {
      // Diagnostic: log what's actually in the dropdown
      const html = await this.selectorDropdown.innerHTML().catch(() => 'DROPDOWN_NOT_IN_DOM');
      console.log(`[hasAgentOption] agent-option-${agentId} NOT FOUND. Dropdown: ${html.substring(0, 500)}`);
      await this.closeSelector();
      return false;
    }
  }

  /** Get the count of agent options in the dropdown. */
  async getAgentOptionCount(): Promise<number> {
    await this.openSelector();
    const options = this.selectorDropdown.locator('[data-testid^="agent-option-"]');
    const count = await options.count();
    await this.closeSelector();
    return count;
  }
}
