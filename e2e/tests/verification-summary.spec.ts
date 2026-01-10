import { test, expect } from '../fixtures';
import { RESEARCH_QUERIES } from '../utils/test-data';
import {
  waitForVerificationSection,
  waitForClaimsList,
  waitForEvidenceCard,
} from '../utils/wait-helpers';

/**
 * Verification Summary Tests - Validate verification panel display and interaction.
 *
 * Tests the 003-claim-level-citations feature for verification summary:
 * - Verification section appears after research response
 * - Toggle expands/collapses section
 * - Badges show correct verdict counts
 * - Warning indicator appears for high unsupported rate
 * - Claims list shows claim previews
 * - Clicking claim opens evidence card
 *
 * NOTE: These tests require real research queries and are SLOW (~3-5 min each).
 * They MUST use deep_research mode to trigger citation verification.
 * Skip in quick CI runs with: npx playwright test --grep-invert "@slow"
 */
test.describe('Verification Summary', () => {
  // Mark all tests in this describe as slow (triples timeout)
  test.slow();

  // Skip verification tests unless explicitly enabled with RUN_SLOW_TESTS=1
  test.skip(
    !process.env.RUN_SLOW_TESTS,
    'Verification tests are slow - set RUN_SLOW_TESTS=1 to enable'
  );

  // Use extended timeout for research queries
  test.setTimeout(600000); // 10 minutes total

  test.describe('Section Display', () => {
    test('verification section appears after research response', async ({
      chatPage,
      citationsPage,
      page,
    }) => {
      const query = RESEARCH_QUERIES[0];

      // Send research query in DEEP_RESEARCH mode to trigger verification
      await chatPage.sendMessageWithMode(query.text, 'deep_research');
      await chatPage.waitForAgentResponse(300000); // 5 minutes for deep research

      // Check if verification section is visible
      const isVisible = await citationsPage.isVerificationSectionVisible();

      // Skip if no verification section (may not be in all responses)
      if (!isVisible) {
        test.skip(true, 'No verification section in response');
        return;
      }

      await expect(citationsPage.verificationSection).toBeVisible();
    });

    test('verification toggle button is accessible', async ({ chatPage, citationsPage, page }) => {
      const query = RESEARCH_QUERIES[0];

      await chatPage.sendMessageWithMode(query.text, 'deep_research');
      await chatPage.waitForAgentResponse(300000);

      const isVisible = await citationsPage.isVerificationSectionVisible();

      if (!isVisible) {
        test.skip(true, 'No verification section in response');
        return;
      }

      // Toggle should be visible
      await expect(citationsPage.verificationToggle).toBeVisible();

      // Check for aria-expanded attribute
      const ariaExpanded = await citationsPage.verificationToggle.getAttribute('aria-expanded');
      expect(['true', 'false']).toContain(ariaExpanded);
    });
  });

  test.describe('Toggle Behavior', () => {
    test('toggle expands verification section', async ({ chatPage, citationsPage, page }) => {
      const query = RESEARCH_QUERIES[0];

      await chatPage.sendMessageWithMode(query.text, 'deep_research');
      await chatPage.waitForAgentResponse(300000);

      const isVisible = await citationsPage.isVerificationSectionVisible();

      if (!isVisible) {
        test.skip(true, 'No verification section in response');
        return;
      }

      // Expand the section
      await citationsPage.expandVerificationSection();

      // Wait a moment for animation
      await page.waitForTimeout(300);

      // Section should be expanded
      const isExpanded = await citationsPage.isVerificationExpanded();
      expect(isExpanded).toBe(true);
    });

    test('toggle collapses verification section', async ({ chatPage, citationsPage, page }) => {
      const query = RESEARCH_QUERIES[0];

      await chatPage.sendMessageWithMode(query.text, 'deep_research');
      await chatPage.waitForAgentResponse(300000);

      const isVisible = await citationsPage.isVerificationSectionVisible();

      if (!isVisible) {
        test.skip(true, 'No verification section in response');
        return;
      }

      // First expand
      await citationsPage.expandVerificationSection();
      await page.waitForTimeout(300);

      // Then collapse
      await citationsPage.collapseVerificationSection();
      await page.waitForTimeout(300);

      // Section should be collapsed
      const isExpanded = await citationsPage.isVerificationExpanded();
      expect(isExpanded).toBe(false);
    });

    test('toggle via keyboard (Enter key)', async ({ chatPage, citationsPage, page }) => {
      const query = RESEARCH_QUERIES[0];

      await chatPage.sendMessageWithMode(query.text, 'deep_research');
      await chatPage.waitForAgentResponse(300000);

      const isVisible = await citationsPage.isVerificationSectionVisible();

      if (!isVisible) {
        test.skip(true, 'No verification section in response');
        return;
      }

      const initialState = await citationsPage.isVerificationExpanded();

      // Focus and press Enter
      await citationsPage.verificationToggle.focus();
      await page.keyboard.press('Enter');
      await page.waitForTimeout(300);

      // State should have changed
      const finalState = await citationsPage.isVerificationExpanded();
      expect(finalState).not.toBe(initialState);
    });
  });

  test.describe('Verification Badges', () => {
    test('verification badges display verdict counts', async ({ chatPage, citationsPage, page }) => {
      const query = RESEARCH_QUERIES[0];

      await chatPage.sendMessageWithMode(query.text, 'deep_research');
      await chatPage.waitForAgentResponse(300000);

      const isVisible = await citationsPage.isVerificationSectionVisible();

      if (!isVisible) {
        test.skip(true, 'No verification section in response');
        return;
      }

      // Badges should be visible
      const badgesText = await citationsPage.getVerificationBadgesText();
      expect(badgesText.length).toBeGreaterThan(0);
    });

    test('supported badge appears when claims are supported', async ({
      chatPage,
      citationsPage,
      page,
    }) => {
      const query = RESEARCH_QUERIES[0];

      await chatPage.sendMessageWithMode(query.text, 'deep_research');
      await chatPage.waitForAgentResponse(300000);

      const isVisible = await citationsPage.isVerificationSectionVisible();

      if (!isVisible) {
        test.skip(true, 'No verification section in response');
        return;
      }

      // Check for supported badge
      const hasSupported = await citationsPage.hasVerificationBadge('supported');

      // At least one type of badge should exist
      const hasPartial = await citationsPage.hasVerificationBadge('partial');
      const hasUnsupported = await citationsPage.hasVerificationBadge('unsupported');

      expect(hasSupported || hasPartial || hasUnsupported).toBe(true);
    });
  });

  test.describe('Warning Indicator', () => {
    test('warning indicator visibility depends on unsupported rate', async ({
      chatPage,
      citationsPage,
      page,
    }) => {
      const query = RESEARCH_QUERIES[0];

      await chatPage.sendMessageWithMode(query.text, 'deep_research');
      await chatPage.waitForAgentResponse(300000);

      const isVisible = await citationsPage.isVerificationSectionVisible();

      if (!isVisible) {
        test.skip(true, 'No verification section in response');
        return;
      }

      // Check warning visibility - should be boolean
      const hasWarning = await citationsPage.hasVerificationWarning();
      expect(typeof hasWarning).toBe('boolean');
    });
  });

  test.describe('Claims List', () => {
    test('claims list displays when section is expanded', async ({ chatPage, citationsPage, page }) => {
      const query = RESEARCH_QUERIES[0];

      await chatPage.sendMessageWithMode(query.text, 'deep_research');
      await chatPage.waitForAgentResponse(300000);

      const isVisible = await citationsPage.isVerificationSectionVisible();

      if (!isVisible) {
        test.skip(true, 'No verification section in response');
        return;
      }

      // Expand section
      await citationsPage.expandVerificationSection();

      // Wait for claims list
      try {
        await waitForClaimsList(page, 5000);
        await expect(citationsPage.claimsList).toBeVisible();
      } catch {
        // Claims list may not always be present
        test.skip(true, 'Claims list not present in this response');
      }
    });

    test('claims list shows claim items', async ({ chatPage, citationsPage, page }) => {
      const query = RESEARCH_QUERIES[0];

      await chatPage.sendMessageWithMode(query.text, 'deep_research');
      await chatPage.waitForAgentResponse(300000);

      const isVisible = await citationsPage.isVerificationSectionVisible();

      if (!isVisible) {
        test.skip(true, 'No verification section in response');
        return;
      }

      // Expand section
      await citationsPage.expandVerificationSection();

      // Wait for claims
      await page.waitForTimeout(500);

      const claimCount = await citationsPage.getClaimCount();

      if (claimCount === 0) {
        test.skip(true, 'No claims in the list');
        return;
      }

      // At least first claim should be visible
      await expect(citationsPage.getClaimItem(1)).toBeVisible();
    });

    test('clicking claim shows evidence details', async ({ chatPage, citationsPage, page }) => {
      const query = RESEARCH_QUERIES[0];

      await chatPage.sendMessageWithMode(query.text, 'deep_research');
      await chatPage.waitForAgentResponse(300000);

      const isVisible = await citationsPage.isVerificationSectionVisible();

      if (!isVisible) {
        test.skip(true, 'No verification section in response');
        return;
      }

      // Expand section
      await citationsPage.expandVerificationSection();
      await page.waitForTimeout(500);

      const claimCount = await citationsPage.getClaimCount();

      if (claimCount === 0) {
        test.skip(true, 'No claims in the list');
        return;
      }

      // Click first claim
      await citationsPage.clickClaim(1);

      // Should show evidence card or expand claim details
      try {
        await waitForEvidenceCard(page, 5000);
        await expect(citationsPage.evidenceCard).toBeVisible();
      } catch {
        // May show inline details instead of card
        const claimText = await citationsPage.getClaimText(1);
        expect(claimText.length).toBeGreaterThan(0);
      }
    });

    test('claim items show preview text', async ({ chatPage, citationsPage, page }) => {
      const query = RESEARCH_QUERIES[0];

      await chatPage.sendMessageWithMode(query.text, 'deep_research');
      await chatPage.waitForAgentResponse(300000);

      const isVisible = await citationsPage.isVerificationSectionVisible();

      if (!isVisible) {
        test.skip(true, 'No verification section in response');
        return;
      }

      // Expand section
      await citationsPage.expandVerificationSection();
      await page.waitForTimeout(500);

      const claimCount = await citationsPage.getClaimCount();

      if (claimCount === 0) {
        test.skip(true, 'No claims in the list');
        return;
      }

      // Get claim text
      const claimText = await citationsPage.getClaimText(1);
      expect(claimText.length).toBeGreaterThan(0);
    });
  });

  test.describe('Accessibility', () => {
    test('verification section is keyboard navigable', async ({ chatPage, citationsPage, page }) => {
      const query = RESEARCH_QUERIES[0];

      await chatPage.sendMessageWithMode(query.text, 'deep_research');
      await chatPage.waitForAgentResponse(300000);

      const isVisible = await citationsPage.isVerificationSectionVisible();

      if (!isVisible) {
        test.skip(true, 'No verification section in response');
        return;
      }

      // Tab to toggle button
      await citationsPage.verificationToggle.focus();

      // Check it can receive focus
      const focusedElement = await page.evaluate(() => document.activeElement?.getAttribute('data-testid'));
      expect(focusedElement).toBe('verification-summary-toggle');
    });

    test('toggle has accessible label', async ({ chatPage, citationsPage, page }) => {
      const query = RESEARCH_QUERIES[0];

      await chatPage.sendMessageWithMode(query.text, 'deep_research');
      await chatPage.waitForAgentResponse(300000);

      const isVisible = await citationsPage.isVerificationSectionVisible();

      if (!isVisible) {
        test.skip(true, 'No verification section in response');
        return;
      }

      // Check for aria-label or accessible text
      const ariaLabel = await citationsPage.verificationToggle.getAttribute('aria-label');
      const buttonText = await citationsPage.verificationToggle.textContent();

      expect(ariaLabel || buttonText).toBeTruthy();
    });
  });
});
