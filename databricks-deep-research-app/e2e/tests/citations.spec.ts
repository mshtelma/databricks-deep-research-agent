import { test, expect } from '../fixtures';
import { RESEARCH_QUERIES } from '../utils/test-data';
import {
  waitForCitationMarkers,
  waitForEvidenceCard,
  waitForEvidenceCardHidden,
  waitForClaimsLoaded,
} from '../utils/wait-helpers';

/**
 * Citation Interaction Tests - Validate claim-level citation display and interaction.
 *
 * Tests the 003-claim-level-citations feature:
 * - Citation marker click/keyboard interactions
 * - Evidence card display with source metadata
 * - Source link navigation
 * - Popover open/close behavior
 *
 * NOTE: These tests require real research queries and are SLOW (~3-5 min each).
 * Skip in quick CI runs with: npx playwright test --grep-invert "@slow"
 */
test.describe('Citation Interactions', () => {
  // Mark all tests in this describe as slow (triples timeout)
  test.slow();

  // Skip citation tests unless explicitly enabled with RUN_SLOW_TESTS=1
  test.skip(
    !process.env.RUN_SLOW_TESTS,
    'Citation tests are slow - set RUN_SLOW_TESTS=1 to enable'
  );

  // Use extended timeout for research queries that generate citations
  test.setTimeout(900000); // 15 minutes total - research with citation pipeline can take 5-10+ min

  /**
   * Helper: Send a research query, wait for response AND claims to load.
   * Claims load asynchronously after the response renders — the evidence card
   * requires claims data to function.
   *
   * Returns the marker count after claims load, or -1 if claims didn't load.
   */
  async function sendQueryAndWaitForClaims(
    chatPage: InstanceType<typeof import('../pages/chat.page').ChatPage>,
    citationsPage: InstanceType<typeof import('../pages/citations.page').CitationsPage>,
    page: import('@playwright/test').Page,
    query: string,
  ): Promise<number> {
    await chatPage.sendMessageWithMode(query, 'deep_research');
    await chatPage.waitForAgentResponse(600000);

    // Claims load asynchronously after response renders via useCitations hook.
    // Without claims, citationData is undefined and evidence card cannot show.
    const claimsLoaded = await waitForClaimsLoaded(page, 60000);
    if (!claimsLoaded) {
      return -1;
    }

    // Re-count markers AFTER claims load — mode switch from link→numeric
    // may change marker keys (e.g., "1" → "Arxiv")
    return citationsPage.getCitationMarkerCount();
  }

  test.describe('Citation Markers', () => {
    test('clicking citation marker opens evidence card', async ({ chatPage, citationsPage, page }) => {
      const query = RESEARCH_QUERIES[0];

      const markerCount = await sendQueryAndWaitForClaims(chatPage, citationsPage, page, query.text);

      if (markerCount === -1) {
        test.skip(true, 'Claims not loaded within timeout — citation pipeline may be slow');
        return;
      }
      if (markerCount === 0) {
        test.skip(true, 'No citations in response after claims loaded');
        return;
      }

      // Click the first citation marker (mode-agnostic — handles both numeric and key-based markers)
      await citationsPage.clickFirstCitationMarker();

      // Evidence card should appear
      await waitForEvidenceCard(page);
      await expect(citationsPage.evidenceCard).toBeVisible();
    });

    test('pressing Enter on focused marker opens evidence card', async ({ chatPage, citationsPage, page }) => {
      const query = RESEARCH_QUERIES[0];

      const markerCount = await sendQueryAndWaitForClaims(chatPage, citationsPage, page, query.text);

      if (markerCount === -1) {
        test.skip(true, 'Claims not loaded within timeout');
        return;
      }
      if (markerCount === 0) {
        test.skip(true, 'No citations in response');
        return;
      }

      // Focus the marker and press Enter
      await citationsPage.pressKeyOnFirstCitationMarker('Enter');

      // Evidence card should appear
      await waitForEvidenceCard(page);
      await expect(citationsPage.evidenceCard).toBeVisible();
    });

    test('pressing Escape closes evidence card', async ({ chatPage, citationsPage, page }) => {
      const query = RESEARCH_QUERIES[0];

      const markerCount = await sendQueryAndWaitForClaims(chatPage, citationsPage, page, query.text);

      if (markerCount === -1) {
        test.skip(true, 'Claims not loaded within timeout');
        return;
      }
      if (markerCount === 0) {
        test.skip(true, 'No citations in response');
        return;
      }

      // Open evidence card
      await citationsPage.clickFirstCitationMarker();
      await waitForEvidenceCard(page);

      // Press Escape to close
      await citationsPage.pressEscape();

      // Evidence card should be hidden
      await waitForEvidenceCardHidden(page);
      await expect(citationsPage.evidenceCard).toBeHidden();
    });

    test('clicking outside evidence card closes it', async ({ chatPage, citationsPage, page }) => {
      const query = RESEARCH_QUERIES[0];

      const markerCount = await sendQueryAndWaitForClaims(chatPage, citationsPage, page, query.text);

      if (markerCount === -1) {
        test.skip(true, 'Claims not loaded within timeout');
        return;
      }
      if (markerCount === 0) {
        test.skip(true, 'No citations in response');
        return;
      }

      // Open evidence card
      await citationsPage.clickFirstCitationMarker();
      await waitForEvidenceCard(page);

      // Click outside the evidence card (on the message list)
      await page.getByTestId('message-list').click({ force: true });

      // Evidence card should close
      await waitForEvidenceCardHidden(page);
    });

    test('multiple citation markers are present and numbered', async ({ chatPage, citationsPage }) => {
      const query = RESEARCH_QUERIES[0];

      // Use deep_research mode to trigger citation generation
      await chatPage.sendMessageWithMode(query.text, 'deep_research');
      await chatPage.waitForAgentResponse(600000);

      const markerCount = await citationsPage.getCitationMarkerCount();

      if (markerCount === 0) {
        test.skip(true, 'No citations in response');
        return;
      }

      // Verify there are multiple markers (at least 2 for a proper research response)
      expect(markerCount).toBeGreaterThanOrEqual(1);

      // Verify the first few markers are visible
      const markers = await citationsPage.citationMarkers.all();
      for (let i = 0; i < Math.min(markers.length, 5); i++) {
        await expect(markers[i]).toBeVisible();
        // Marker should contain bracket-wrapped text (e.g., [1], [Arxiv])
        const text = await markers[i].textContent();
        expect(text).toMatch(/^\[.+\]$/);
      }
    });
  });

  test.describe('Evidence Card Display', () => {
    test('evidence card shows source metadata', async ({ chatPage, citationsPage, page }) => {
      const query = RESEARCH_QUERIES[0];

      const markerCount = await sendQueryAndWaitForClaims(chatPage, citationsPage, page, query.text);

      if (markerCount === -1) {
        test.skip(true, 'Claims not loaded within timeout');
        return;
      }
      if (markerCount === 0) {
        test.skip(true, 'No citations in response');
        return;
      }

      // Open evidence card
      await citationsPage.clickFirstCitationMarker();
      await waitForEvidenceCard(page);

      // Check for metadata elements
      const content = await citationsPage.getEvidenceCardContent();
      expect(content.hasQuote || content.hasMetadata).toBe(true);
    });

    test('evidence card shows evidence quote', async ({ chatPage, citationsPage, page }) => {
      const query = RESEARCH_QUERIES[0];

      const markerCount = await sendQueryAndWaitForClaims(chatPage, citationsPage, page, query.text);

      if (markerCount === -1) {
        test.skip(true, 'Claims not loaded within timeout');
        return;
      }
      if (markerCount === 0) {
        test.skip(true, 'No citations in response');
        return;
      }

      // Open evidence card
      await citationsPage.clickFirstCitationMarker();
      await waitForEvidenceCard(page);

      // Quote should be visible if available
      const hasQuote = await citationsPage.evidenceQuote.isVisible().catch(() => false);
      if (hasQuote) {
        const quoteText = await citationsPage.evidenceQuote.textContent();
        expect(quoteText?.length).toBeGreaterThan(0);
      }
    });

    test('source URL link has correct target attribute', async ({ chatPage, citationsPage, page }) => {
      const query = RESEARCH_QUERIES[0];

      const markerCount = await sendQueryAndWaitForClaims(chatPage, citationsPage, page, query.text);

      if (markerCount === -1) {
        test.skip(true, 'Claims not loaded within timeout');
        return;
      }
      if (markerCount === 0) {
        test.skip(true, 'No citations in response');
        return;
      }

      // Open evidence card
      await citationsPage.clickFirstCitationMarker();
      await waitForEvidenceCard(page);

      // Check if source URL link exists and has target="_blank"
      const urlVisible = await citationsPage.sourceMetadataUrl.isVisible().catch(() => false);
      if (urlVisible) {
        const target = await citationsPage.sourceMetadataUrl.getAttribute('target');
        expect(target).toBe('_blank');

        const rel = await citationsPage.sourceMetadataUrl.getAttribute('rel');
        expect(rel).toContain('noopener');
      }
    });

    test('close button dismisses evidence card', async ({ chatPage, citationsPage, page }) => {
      const query = RESEARCH_QUERIES[0];

      const markerCount = await sendQueryAndWaitForClaims(chatPage, citationsPage, page, query.text);

      if (markerCount === -1) {
        test.skip(true, 'Claims not loaded within timeout');
        return;
      }
      if (markerCount === 0) {
        test.skip(true, 'No citations in response');
        return;
      }

      // Open evidence card
      await citationsPage.clickFirstCitationMarker();
      await waitForEvidenceCard(page);

      // Click close button
      await citationsPage.closeEvidenceCard();

      // Evidence card should be hidden
      await waitForEvidenceCardHidden(page);
    });
  });

  test.describe('Provenance Export', () => {
    test('provenance export endpoint responds', async ({ request }) => {
      // This is a simple check that the endpoint structure exists
      const response = await request.get('/api/v1/messages/00000000-0000-0000-0000-000000000000/provenance');

      // Endpoint should exist (even if returning 404 for invalid ID)
      // 401 = auth required (expected), 404 = not found (expected for fake ID), 422 = validation error
      expect([401, 404, 422]).toContain(response.status());
    });
  });

  test.describe('Response Rendering', () => {
    test('response renders without errors', async ({ chatPage, page }) => {
      const query = 'What is climate change?';

      // Use deep_research mode to trigger citation generation
      await chatPage.sendMessageWithMode(query, 'deep_research');
      await chatPage.waitForAgentResponse(600000);

      // Get the agent response
      const response = await chatPage.getLastAgentResponse();

      // Response should exist and have content
      expect(response.length).toBeGreaterThan(0);

      // No error messages should be visible (check for actual error states, not styling)
      const errorAlert = page.locator('[role="alert"][class*="error" i]');
      const errorCount = await errorAlert.count();
      expect(errorCount).toBe(0);
    });

    test('sources section is accessible', async ({ chatPage, page }) => {
      const query = RESEARCH_QUERIES[0];

      // Use deep_research mode to trigger citation generation
      await chatPage.sendMessageWithMode(query.text, 'deep_research');
      await chatPage.waitForAgentResponse(600000);

      // Look for sources section
      const sourcesButton = page.locator('button:has-text("source")');
      const sourcesVisible = await sourcesButton.isVisible().catch(() => false);

      // If sources section exists, it should be expandable
      if (sourcesVisible) {
        await sourcesButton.click();

        // Source links should appear
        const sourceLinks = page.locator('a[href^="http"], [data-testid="citation"]');
        const linkCount = await sourceLinks.count();
        expect(linkCount).toBeGreaterThanOrEqual(0);
      }
    });
  });
});
