import { type Locator, type Page, expect } from '@playwright/test';

/**
 * Page Object Model for Citation interactions.
 * Encapsulates selectors and common actions for citation verification features.
 */
export class CitationsPage {
  readonly page: Page;

  // Citation markers
  readonly citationMarkers: Locator;

  // Evidence card elements
  readonly evidenceCard: Locator;
  readonly evidenceCardClose: Locator;
  readonly evidenceQuote: Locator;
  readonly evidenceQuoteExpand: Locator;
  readonly sourceMetadata: Locator;
  readonly sourceMetadataUrl: Locator;

  // Numeric claims
  readonly numericChips: Locator;
  readonly numericDetails: Locator;
  readonly numericDetailsClose: Locator;

  // Verification summary
  readonly verificationSection: Locator;
  readonly verificationToggle: Locator;
  readonly verificationBadges: Locator;
  readonly verificationWarning: Locator;
  readonly claimsList: Locator;

  constructor(page: Page) {
    this.page = page;

    // Citation markers - uses prefix selector
    this.citationMarkers = page.locator('[data-testid^="citation-marker-"]');

    // Evidence card elements
    this.evidenceCard = page.getByTestId('evidence-card');
    this.evidenceCardClose = page.getByTestId('evidence-card-close');
    this.evidenceQuote = page.getByTestId('evidence-quote');
    this.evidenceQuoteExpand = page.getByTestId('evidence-quote-expand');
    this.sourceMetadata = page.getByTestId('source-metadata');
    this.sourceMetadataUrl = page.getByTestId('source-metadata-url');

    // Numeric claims
    this.numericChips = page.getByTestId('numeric-chip');
    this.numericDetails = page.getByTestId('numeric-details');
    this.numericDetailsClose = page.getByTestId('numeric-details-close');

    // Verification summary
    this.verificationSection = page.getByTestId('verification-summary-section');
    this.verificationToggle = page.getByTestId('verification-summary-toggle');
    this.verificationBadges = page.getByTestId('verification-summary-badges');
    this.verificationWarning = page.getByTestId('verification-summary-warning');
    this.claimsList = page.getByTestId('claims-list');
  }

  // ==================== Citation Marker Methods ====================

  /**
   * Get the count of citation markers in the page.
   */
  async getCitationMarkerCount(): Promise<number> {
    return this.citationMarkers.count();
  }

  /**
   * Get a specific citation marker by index (1-based).
   */
  getCitationMarker(index: number): Locator {
    return this.page.getByTestId(`citation-marker-${index}`);
  }

  /**
   * Click a citation marker by index (1-based).
   */
  async clickCitationMarker(index: number): Promise<void> {
    const marker = this.getCitationMarker(index);
    await marker.click();
  }

  /**
   * Hover over a citation marker by index (1-based).
   */
  async hoverCitationMarker(index: number): Promise<void> {
    const marker = this.getCitationMarker(index);
    await marker.hover();
  }

  /**
   * Focus a citation marker by index (1-based).
   */
  async focusCitationMarker(index: number): Promise<void> {
    const marker = this.getCitationMarker(index);
    await marker.focus();
  }

  /**
   * Press a key while focused on a citation marker.
   */
  async pressKeyOnCitationMarker(index: number, key: string): Promise<void> {
    await this.focusCitationMarker(index);
    await this.page.keyboard.press(key);
  }

  // ==================== Evidence Card Methods ====================

  /**
   * Wait for the evidence card to be visible.
   */
  async waitForEvidenceCard(timeout: number = 10000): Promise<void> {
    await expect(this.evidenceCard).toBeVisible({ timeout });
  }

  /**
   * Check if evidence card is visible.
   */
  async isEvidenceCardVisible(): Promise<boolean> {
    return this.evidenceCard.isVisible();
  }

  /**
   * Close the evidence card.
   */
  async closeEvidenceCard(): Promise<void> {
    await this.evidenceCardClose.click();
  }

  /**
   * Get the evidence card content.
   */
  async getEvidenceCardContent(): Promise<{
    hasQuote: boolean;
    hasMetadata: boolean;
    quoteText: string | null;
  }> {
    const hasQuote = await this.evidenceQuote.isVisible().catch(() => false);
    const hasMetadata = await this.sourceMetadata.isVisible().catch(() => false);
    const quoteText = hasQuote ? await this.evidenceQuote.textContent() : null;

    return { hasQuote, hasMetadata, quoteText };
  }

  /**
   * Click the source URL link in evidence card.
   */
  async clickSourceLink(): Promise<void> {
    await this.sourceMetadataUrl.click();
  }

  /**
   * Get the source URL href from evidence card.
   */
  async getSourceUrl(): Promise<string | null> {
    return this.sourceMetadataUrl.getAttribute('href');
  }

  /**
   * Expand the evidence quote (show more).
   */
  async expandEvidenceQuote(): Promise<void> {
    const isVisible = await this.evidenceQuoteExpand.isVisible().catch(() => false);
    if (isVisible) {
      await this.evidenceQuoteExpand.click();
    }
  }

  // ==================== Numeric Chip Methods ====================

  /**
   * Get the count of numeric chips.
   */
  async getNumericChipCount(): Promise<number> {
    return this.numericChips.count();
  }

  /**
   * Click a numeric chip by index (0-based).
   */
  async clickNumericChip(index: number = 0): Promise<void> {
    await this.numericChips.nth(index).click();
  }

  /**
   * Get numeric chip text by index (0-based).
   */
  async getNumericChipText(index: number = 0): Promise<string> {
    return (await this.numericChips.nth(index).textContent()) ?? '';
  }

  /**
   * Wait for numeric details panel to be visible.
   */
  async waitForNumericDetails(timeout: number = 5000): Promise<void> {
    await expect(this.numericDetails).toBeVisible({ timeout });
  }

  /**
   * Check if numeric details panel is visible.
   */
  async isNumericDetailsVisible(): Promise<boolean> {
    return this.numericDetails.isVisible();
  }

  /**
   * Close numeric details panel.
   */
  async closeNumericDetails(): Promise<void> {
    await this.numericDetailsClose.click();
  }

  /**
   * Get numeric details content.
   */
  async getNumericDetailsContent(): Promise<string> {
    return (await this.numericDetails.textContent()) ?? '';
  }

  // ==================== Verification Summary Methods ====================

  /**
   * Check if verification section is visible.
   */
  async isVerificationSectionVisible(): Promise<boolean> {
    return this.verificationSection.isVisible();
  }

  /**
   * Wait for verification section to be visible.
   */
  async waitForVerificationSection(timeout: number = 10000): Promise<void> {
    await expect(this.verificationSection).toBeVisible({ timeout });
  }

  /**
   * Expand/toggle the verification section.
   */
  async toggleVerificationSection(): Promise<void> {
    await this.verificationToggle.click();
  }

  /**
   * Check if verification section is expanded.
   */
  async isVerificationExpanded(): Promise<boolean> {
    const expanded = await this.verificationToggle.getAttribute('aria-expanded');
    return expanded === 'true';
  }

  /**
   * Expand verification section if not already expanded.
   */
  async expandVerificationSection(): Promise<void> {
    const isExpanded = await this.isVerificationExpanded();
    if (!isExpanded) {
      await this.toggleVerificationSection();
    }
  }

  /**
   * Collapse verification section if expanded.
   */
  async collapseVerificationSection(): Promise<void> {
    const isExpanded = await this.isVerificationExpanded();
    if (isExpanded) {
      await this.toggleVerificationSection();
    }
  }

  /**
   * Check if verification warning is visible.
   */
  async hasVerificationWarning(): Promise<boolean> {
    return this.verificationWarning.isVisible();
  }

  /**
   * Get verification badges text.
   */
  async getVerificationBadgesText(): Promise<string> {
    return (await this.verificationBadges.textContent()) ?? '';
  }

  // ==================== Claims List Methods ====================

  /**
   * Get count of claims in the list.
   */
  async getClaimCount(): Promise<number> {
    return this.page.locator('[data-testid^="claim-item-"]').count();
  }

  /**
   * Get a specific claim item by index (1-based).
   */
  getClaimItem(index: number): Locator {
    return this.page.getByTestId(`claim-item-${index}`);
  }

  /**
   * Click a claim item by index (1-based).
   */
  async clickClaim(index: number): Promise<void> {
    const claim = this.getClaimItem(index);
    await claim.click();
  }

  /**
   * Get claim item text by index (1-based).
   */
  async getClaimText(index: number): Promise<string> {
    const claim = this.getClaimItem(index);
    return (await claim.textContent()) ?? '';
  }

  /**
   * Check if a verification badge with specific verdict exists.
   */
  async hasVerificationBadge(verdict: 'supported' | 'partial' | 'unsupported' | 'contradicted'): Promise<boolean> {
    return this.page.getByTestId(`verification-badge-${verdict}`).isVisible();
  }

  // ==================== Keyboard Navigation Methods ====================

  /**
   * Press Escape to close any open popover.
   */
  async pressEscape(): Promise<void> {
    await this.page.keyboard.press('Escape');
  }

  /**
   * Press Enter while focused on an element.
   */
  async pressEnter(): Promise<void> {
    await this.page.keyboard.press('Enter');
  }

  // ==================== Grey Reference Detection Methods ====================

  /**
   * Check if a specific citation marker is grey (unresolved).
   * Grey markers have opacity-60 and text-gray-400 classes.
   */
  async isCitationMarkerGrey(citationKey: string): Promise<boolean> {
    const marker = this.page.getByTestId(`citation-marker-${citationKey}`);
    const classes = (await marker.getAttribute('class')) || '';
    return classes.includes('opacity-60') || classes.includes('text-gray-400');
  }

  /**
   * Get all grey (unresolved) citation markers.
   * Returns array of citation keys that are rendered as grey.
   */
  async getGreyCitationMarkers(): Promise<string[]> {
    const markers = await this.citationMarkers.all();
    const greyMarkers: string[] = [];

    for (const marker of markers) {
      const classes = (await marker.getAttribute('class')) || '';
      if (classes.includes('opacity-60') || classes.includes('text-gray-400')) {
        const testId = (await marker.getAttribute('data-testid')) || '';
        const citationKey = testId.replace('citation-marker-', '');
        if (citationKey) {
          greyMarkers.push(citationKey);
        }
      }
    }

    return greyMarkers;
  }

  /**
   * Get citation resolution statistics.
   * Returns count of resolved (colored) and grey (unresolved) markers.
   */
  async getCitationResolutionStats(): Promise<{ resolved: number; grey: number }> {
    const markers = await this.citationMarkers.all();
    let resolved = 0;
    let grey = 0;

    for (const marker of markers) {
      const classes = (await marker.getAttribute('class')) || '';
      if (classes.includes('opacity-60') || classes.includes('text-gray-400')) {
        grey++;
      } else {
        resolved++;
      }
    }

    return { resolved, grey };
  }

  /**
   * Wait until all citation markers are resolved (non-grey).
   * Useful for waiting for claims API to load.
   */
  async waitForAllCitationsResolved(timeout: number = 30000): Promise<void> {
    await expect(async () => {
      const stats = await this.getCitationResolutionStats();
      expect(stats.grey).toBe(0);
    }).toPass({ timeout });
  }
}
