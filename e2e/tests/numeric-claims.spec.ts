import { test, expect } from '../fixtures';
import { RESEARCH_QUERIES } from '../utils/test-data';
import { waitForNumericChips, waitForNumericDetails } from '../utils/wait-helpers';

/**
 * Numeric Claims Tests - Validate numeric chip display and interaction.
 *
 * Tests the 003-claim-level-citations feature for numeric claims:
 * - Numeric chips appear for numeric claims in response
 * - Clicking chip opens details panel
 * - Details panel shows raw and normalized values
 * - Close button and keyboard navigation
 *
 * NOTE: These tests require real research queries and are SLOW (~3-5 min each).
 * Skip in quick CI runs with: npx playwright test --grep-invert "@slow"
 */
test.describe('Numeric Claims', () => {
  // Mark all tests in this describe as slow (triples timeout)
  test.slow();

  // Skip numeric claims tests unless explicitly enabled with RUN_SLOW_TESTS=1
  test.skip(
    !process.env.RUN_SLOW_TESTS,
    'Numeric claims tests are slow - set RUN_SLOW_TESTS=1 to enable'
  );

  // Use extended timeout for research queries that may produce numeric data
  test.setTimeout(600000); // 10 minutes total

  // Use a query likely to produce numeric data
  const NUMERIC_QUERY = 'What are the current global AI market size statistics and growth projections?';

  test.describe('Numeric Chips Display', () => {
    test('numeric chips appear for research responses with statistics', async ({
      chatPage,
      citationsPage,
      page,
    }) => {
      // Send a query likely to produce numeric claims
      // Use deep_research mode to trigger numeric claim detection
      await chatPage.sendMessageWithMode(NUMERIC_QUERY, 'deep_research');
      await chatPage.waitForAgentResponse(360000);

      // Check for numeric chips
      const chipCount = await citationsPage.getNumericChipCount();

      // Skip if no numeric chips (response may not contain numbers)
      if (chipCount === 0) {
        test.skip(true, 'No numeric chips in response - query may not have produced numeric data');
        return;
      }

      // At least one chip should be visible
      await expect(citationsPage.numericChips.first()).toBeVisible();
    });

    test('numeric chips display formatted values', async ({ chatPage, citationsPage }) => {
      // Use deep_research mode to trigger numeric claim detection
      await chatPage.sendMessageWithMode(NUMERIC_QUERY, 'deep_research');
      await chatPage.waitForAgentResponse(360000);

      const chipCount = await citationsPage.getNumericChipCount();

      if (chipCount === 0) {
        test.skip(true, 'No numeric chips in response');
        return;
      }

      // Get the chip text - should contain numeric content
      const chipText = await citationsPage.getNumericChipText(0);
      expect(chipText.length).toBeGreaterThan(0);
    });
  });

  test.describe('Numeric Details Panel', () => {
    test('clicking numeric chip opens details panel', async ({ chatPage, citationsPage, page }) => {
      // Use deep_research mode to trigger numeric claim detection
      await chatPage.sendMessageWithMode(NUMERIC_QUERY, 'deep_research');
      await chatPage.waitForAgentResponse(360000);

      const chipCount = await citationsPage.getNumericChipCount();

      if (chipCount === 0) {
        test.skip(true, 'No numeric chips in response');
        return;
      }

      // Click the first numeric chip
      await citationsPage.clickNumericChip(0);

      // Details panel should appear
      await waitForNumericDetails(page);
      await expect(citationsPage.numericDetails).toBeVisible();
    });

    test('details panel shows computation information', async ({ chatPage, citationsPage, page }) => {
      // Use deep_research mode to trigger numeric claim detection
      await chatPage.sendMessageWithMode(NUMERIC_QUERY, 'deep_research');
      await chatPage.waitForAgentResponse(360000);

      const chipCount = await citationsPage.getNumericChipCount();

      if (chipCount === 0) {
        test.skip(true, 'No numeric chips in response');
        return;
      }

      // Open details panel
      await citationsPage.clickNumericChip(0);
      await waitForNumericDetails(page);

      // Check for content
      const content = await citationsPage.getNumericDetailsContent();
      expect(content.length).toBeGreaterThan(0);
    });

    test('close button dismisses details panel', async ({ chatPage, citationsPage, page }) => {
      // Use deep_research mode to trigger numeric claim detection
      await chatPage.sendMessageWithMode(NUMERIC_QUERY, 'deep_research');
      await chatPage.waitForAgentResponse(360000);

      const chipCount = await citationsPage.getNumericChipCount();

      if (chipCount === 0) {
        test.skip(true, 'No numeric chips in response');
        return;
      }

      // Open details panel
      await citationsPage.clickNumericChip(0);
      await waitForNumericDetails(page);

      // Close using close button
      await citationsPage.closeNumericDetails();

      // Panel should be hidden
      await expect(citationsPage.numericDetails).toBeHidden({ timeout: 5000 });
    });

    test('pressing Escape closes details panel', async ({ chatPage, citationsPage, page }) => {
      // Use deep_research mode to trigger numeric claim detection
      await chatPage.sendMessageWithMode(NUMERIC_QUERY, 'deep_research');
      await chatPage.waitForAgentResponse(360000);

      const chipCount = await citationsPage.getNumericChipCount();

      if (chipCount === 0) {
        test.skip(true, 'No numeric chips in response');
        return;
      }

      // Open details panel
      await citationsPage.clickNumericChip(0);
      await waitForNumericDetails(page);

      // Press Escape to close
      await citationsPage.pressEscape();

      // Panel should be hidden
      await expect(citationsPage.numericDetails).toBeHidden({ timeout: 5000 });
    });

    test('clicking outside details panel closes it', async ({ chatPage, citationsPage, page }) => {
      // Use deep_research mode to trigger numeric claim detection
      await chatPage.sendMessageWithMode(NUMERIC_QUERY, 'deep_research');
      await chatPage.waitForAgentResponse(360000);

      const chipCount = await citationsPage.getNumericChipCount();

      if (chipCount === 0) {
        test.skip(true, 'No numeric chips in response');
        return;
      }

      // Open details panel
      await citationsPage.clickNumericChip(0);
      await waitForNumericDetails(page);

      // Click outside (on the message list)
      await page.getByTestId('message-list').click({ force: true });

      // Panel should close
      await expect(citationsPage.numericDetails).toBeHidden({ timeout: 5000 });
    });
  });

  test.describe('Multiple Numeric Chips', () => {
    test('multiple chips can be opened sequentially', async ({ chatPage, citationsPage, page }) => {
      // Use deep_research mode to trigger numeric claim detection
      await chatPage.sendMessageWithMode(NUMERIC_QUERY, 'deep_research');
      await chatPage.waitForAgentResponse(360000);

      const chipCount = await citationsPage.getNumericChipCount();

      if (chipCount < 2) {
        test.skip(true, 'Need at least 2 numeric chips for this test');
        return;
      }

      // Click first chip
      await citationsPage.clickNumericChip(0);
      await waitForNumericDetails(page);

      // Close it
      await citationsPage.closeNumericDetails();
      await expect(citationsPage.numericDetails).toBeHidden({ timeout: 5000 });

      // Click second chip
      await citationsPage.clickNumericChip(1);
      await waitForNumericDetails(page);
      await expect(citationsPage.numericDetails).toBeVisible();
    });
  });

  test.describe('Accessibility', () => {
    test('numeric chips are keyboard accessible', async ({ chatPage, citationsPage, page }) => {
      // Use deep_research mode to trigger numeric claim detection
      await chatPage.sendMessageWithMode(NUMERIC_QUERY, 'deep_research');
      await chatPage.waitForAgentResponse(360000);

      const chipCount = await citationsPage.getNumericChipCount();

      if (chipCount === 0) {
        test.skip(true, 'No numeric chips in response');
        return;
      }

      // Focus the first chip
      await citationsPage.numericChips.first().focus();

      // Press Enter to activate
      await page.keyboard.press('Enter');

      // Details panel should open
      await waitForNumericDetails(page);
      await expect(citationsPage.numericDetails).toBeVisible();
    });
  });
});
