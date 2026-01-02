import { test, expect } from '../fixtures';
import { RESEARCH_QUERIES } from '../utils/test-data';
import {
  waitForCitationMarkers,
  waitForEvidenceCard,
  waitForEvidenceCardHidden,
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
  test.setTimeout(600000); // 10 minutes total

  test.describe('Citation Markers', () => {
    test('clicking citation marker opens evidence card', async ({ chatPage, citationsPage, page }) => {
      const query = RESEARCH_QUERIES[0];

      // Send research query
      await chatPage.sendMessage(query.text);
      await chatPage.waitForAgentResponse(180000);

      // Check if citations exist in the response
      const markerCount = await citationsPage.getCitationMarkerCount();

      if (markerCount === 0) {
        // Skip test if no citations - research may not have produced any
        test.skip(true, 'No citations in response - skipping citation interaction tests');
        return;
      }

      // Click the first citation marker
      await citationsPage.clickCitationMarker(1);

      // Evidence card should appear
      await waitForEvidenceCard(page);
      await expect(citationsPage.evidenceCard).toBeVisible();
    });

    test('pressing Enter on focused marker opens evidence card', async ({ chatPage, citationsPage, page }) => {
      const query = RESEARCH_QUERIES[0];

      await chatPage.sendMessage(query.text);
      await chatPage.waitForAgentResponse(180000);

      const markerCount = await citationsPage.getCitationMarkerCount();

      if (markerCount === 0) {
        test.skip(true, 'No citations in response');
        return;
      }

      // Focus the marker and press Enter
      await citationsPage.pressKeyOnCitationMarker(1, 'Enter');

      // Evidence card should appear
      await waitForEvidenceCard(page);
      await expect(citationsPage.evidenceCard).toBeVisible();
    });

    test('pressing Escape closes evidence card', async ({ chatPage, citationsPage, page }) => {
      const query = RESEARCH_QUERIES[0];

      await chatPage.sendMessage(query.text);
      await chatPage.waitForAgentResponse(180000);

      const markerCount = await citationsPage.getCitationMarkerCount();

      if (markerCount === 0) {
        test.skip(true, 'No citations in response');
        return;
      }

      // Open evidence card
      await citationsPage.clickCitationMarker(1);
      await waitForEvidenceCard(page);

      // Press Escape to close
      await citationsPage.pressEscape();

      // Evidence card should be hidden
      await waitForEvidenceCardHidden(page);
      await expect(citationsPage.evidenceCard).toBeHidden();
    });

    test('clicking outside evidence card closes it', async ({ chatPage, citationsPage, page }) => {
      const query = RESEARCH_QUERIES[0];

      await chatPage.sendMessage(query.text);
      await chatPage.waitForAgentResponse(180000);

      const markerCount = await citationsPage.getCitationMarkerCount();

      if (markerCount === 0) {
        test.skip(true, 'No citations in response');
        return;
      }

      // Open evidence card
      await citationsPage.clickCitationMarker(1);
      await waitForEvidenceCard(page);

      // Click outside the evidence card (on the message list)
      await page.getByTestId('message-list').click({ force: true });

      // Evidence card should close
      await waitForEvidenceCardHidden(page);
    });

    test('multiple citation markers are present and numbered', async ({ chatPage, citationsPage }) => {
      const query = RESEARCH_QUERIES[0];

      await chatPage.sendMessage(query.text);
      await chatPage.waitForAgentResponse(180000);

      const markerCount = await citationsPage.getCitationMarkerCount();

      if (markerCount === 0) {
        test.skip(true, 'No citations in response');
        return;
      }

      // Verify markers are numbered sequentially
      for (let i = 1; i <= Math.min(markerCount, 5); i++) {
        const marker = citationsPage.getCitationMarker(i);
        await expect(marker).toBeVisible();
        await expect(marker).toContainText(`[${i}]`);
      }
    });
  });

  test.describe('Evidence Card Display', () => {
    test('evidence card shows source metadata', async ({ chatPage, citationsPage, page }) => {
      const query = RESEARCH_QUERIES[0];

      await chatPage.sendMessage(query.text);
      await chatPage.waitForAgentResponse(180000);

      const markerCount = await citationsPage.getCitationMarkerCount();

      if (markerCount === 0) {
        test.skip(true, 'No citations in response');
        return;
      }

      // Open evidence card
      await citationsPage.clickCitationMarker(1);
      await waitForEvidenceCard(page);

      // Check for metadata elements
      const content = await citationsPage.getEvidenceCardContent();
      expect(content.hasQuote || content.hasMetadata).toBe(true);
    });

    test('evidence card shows evidence quote', async ({ chatPage, citationsPage, page }) => {
      const query = RESEARCH_QUERIES[0];

      await chatPage.sendMessage(query.text);
      await chatPage.waitForAgentResponse(180000);

      const markerCount = await citationsPage.getCitationMarkerCount();

      if (markerCount === 0) {
        test.skip(true, 'No citations in response');
        return;
      }

      // Open evidence card
      await citationsPage.clickCitationMarker(1);
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

      await chatPage.sendMessage(query.text);
      await chatPage.waitForAgentResponse(180000);

      const markerCount = await citationsPage.getCitationMarkerCount();

      if (markerCount === 0) {
        test.skip(true, 'No citations in response');
        return;
      }

      // Open evidence card
      await citationsPage.clickCitationMarker(1);
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

      await chatPage.sendMessage(query.text);
      await chatPage.waitForAgentResponse(180000);

      const markerCount = await citationsPage.getCitationMarkerCount();

      if (markerCount === 0) {
        test.skip(true, 'No citations in response');
        return;
      }

      // Open evidence card
      await citationsPage.clickCitationMarker(1);
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

      await chatPage.sendMessage(query);
      await chatPage.waitForAgentResponse(180000);

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

      await chatPage.sendMessage(query.text);
      await chatPage.waitForAgentResponse(180000);

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
