import { test, expect } from '../fixtures';
import { waitForCitationMarkers } from '../utils/wait-helpers';

/**
 * Grey Reference Detection Tests
 *
 * These tests verify that NO grey/unresolved citations exist after research
 * with the verification pipeline. Grey references are citation markers that:
 * - Appear visually as grey/faded (text-gray-400, opacity-60)
 * - Have no matching Claim row in the database
 * - Have Claim but no linked Citation rows
 *
 * Detection is done by checking CSS classes on citation markers.
 * Grey markers have: opacity-60 and text-gray-400 classes.
 *
 * NOTE: These tests run real research queries using the ultra-light test config
 * (config/app.test.yaml): 1-2 steps, minimal iterations (~60-90 seconds each).
 *
 * Run with: RUN_SLOW_TESTS=1 npx playwright test grey-references.spec.ts
 */
test.describe('Grey Reference Detection', () => {
  // Tests require real research queries - skip unless explicitly enabled
  test.skip(
    !process.env.RUN_SLOW_TESTS,
    'Grey reference tests require real research - set RUN_SLOW_TESTS=1 to enable'
  );

  // Use extended timeout for research queries (3 min max for ultra-light)
  test.setTimeout(180000);

  test('no grey citations after research completes', async ({ chatPage, citationsPage, page }) => {
    // Select light research mode for fastest execution
    await chatPage.selectResearchDepth('light');

    // Run a simple factual research query with deep_research mode to trigger citations
    await chatPage.sendMessageWithMode('What is the capital of France?', 'deep_research');

    // Wait for agent response (2 min max for ultra-light research)
    await chatPage.waitForAgentResponse(120000);

    // Wait for citation markers to appear
    try {
      await waitForCitationMarkers(page, 1, 30000);
    } catch {
      // No citation markers found - skip test as there's nothing to verify
      test.skip(true, 'No citation markers in response - cannot verify grey references');
      return;
    }

    // Wait additional time for claims API to complete (TanStack Query)
    await page.waitForTimeout(5000);

    // Check for grey/unresolved markers
    const greyMarkers = await citationsPage.getGreyCitationMarkers();

    // Log findings for debugging
    if (greyMarkers.length > 0) {
      console.log('GREY REFERENCES FOUND:');
      greyMarkers.forEach((key) => console.log(`  - [${key}]`));

      // Get resolution stats for more context
      const stats = await citationsPage.getCitationResolutionStats();
      console.log(`Resolution stats: ${stats.resolved} resolved, ${stats.grey} grey`);
    }

    // Assert no grey references exist
    expect(greyMarkers, `Grey references found: [${greyMarkers.join('], [')}]`).toHaveLength(0);
  });

  test('all citations resolve within timeout', async ({ chatPage, citationsPage, page }) => {
    // Select light research mode
    await chatPage.selectResearchDepth('light');

    // Run research query with deep_research mode to trigger citations
    await chatPage.sendMessageWithMode('What programming language did Guido van Rossum create?', 'deep_research');
    await chatPage.waitForAgentResponse(120000);

    // Check if citations exist
    const markerCount = await citationsPage.getCitationMarkerCount();
    if (markerCount === 0) {
      test.skip(true, 'No citations in response - cannot verify resolution');
      return;
    }

    // Wait for all citations to resolve (should happen within 30s after response)
    try {
      await citationsPage.waitForAllCitationsResolved(30000);
    } catch {
      // If timeout, get the grey markers for error message
      const stats = await citationsPage.getCitationResolutionStats();
      const greyMarkers = await citationsPage.getGreyCitationMarkers();

      expect.fail(
        `Citations failed to resolve within timeout.\n` +
          `Stats: ${stats.resolved} resolved, ${stats.grey} grey\n` +
          `Grey markers: [${greyMarkers.join('], [')}]`
      );
    }

    // Final verification - all should be resolved
    const finalStats = await citationsPage.getCitationResolutionStats();
    expect(finalStats.grey).toBe(0);
    expect(finalStats.resolved).toBeGreaterThan(0);
  });

  test('citation resolution stats are accurate', async ({ chatPage, citationsPage, page }) => {
    // Select light research mode
    await chatPage.selectResearchDepth('light');

    // Run research with deep_research mode to trigger citations
    await chatPage.sendMessageWithMode('What year was Python first released?', 'deep_research');
    await chatPage.waitForAgentResponse(120000);

    // Wait for citations
    try {
      await waitForCitationMarkers(page, 1, 30000);
    } catch {
      test.skip(true, 'No citations in response');
      return;
    }

    // Wait for claims API
    await page.waitForTimeout(5000);

    // Get stats
    const stats = await citationsPage.getCitationResolutionStats();
    const markerCount = await citationsPage.getCitationMarkerCount();

    // Stats should match total marker count
    expect(stats.resolved + stats.grey).toBe(markerCount);

    // Log stats for debugging
    console.log(
      `Citation stats: ${markerCount} total, ${stats.resolved} resolved, ${stats.grey} grey`
    );

    // All should be resolved (no grey)
    expect(stats.grey).toBe(0);
  });

  test('individual citation marker grey detection works', async ({
    chatPage,
    citationsPage,
    page,
  }) => {
    // Select light research mode
    await chatPage.selectResearchDepth('light');

    // Run research with deep_research mode to trigger citations
    await chatPage.sendMessageWithMode('Who founded Microsoft?', 'deep_research');
    await chatPage.waitForAgentResponse(120000);

    // Wait for citations
    try {
      await waitForCitationMarkers(page, 1, 30000);
    } catch {
      test.skip(true, 'No citations in response');
      return;
    }

    // Wait for claims API to resolve
    await page.waitForTimeout(5000);

    // Get all markers and check each one
    const markers = await page.locator('[data-testid^="citation-marker-"]').all();

    for (const marker of markers) {
      const testId = (await marker.getAttribute('data-testid')) || '';
      const citationKey = testId.replace('citation-marker-', '');

      if (citationKey) {
        const isGrey = await citationsPage.isCitationMarkerGrey(citationKey);

        // Each marker should NOT be grey
        expect(isGrey, `Citation [${citationKey}] should not be grey`).toBe(false);
      }
    }
  });
});
