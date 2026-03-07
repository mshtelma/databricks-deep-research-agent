import { test, expect } from '../fixtures';
import { RESEARCH_QUERIES } from '../utils/test-data';
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
 * IMPORTANT: Queries must be research-worthy (not simple factual questions).
 * The coordinator classifies simple queries (e.g. "What is the capital of France?")
 * as simple and bypasses the research pipeline entirely, producing no citations.
 *
 * Run with: RUN_SLOW_TESTS=1 npx playwright test grey-references.spec.ts
 */
test.describe('Grey Reference Detection', () => {
  // Tests require real research queries - skip unless explicitly enabled
  test.skip(
    !process.env.RUN_SLOW_TESTS,
    'Grey reference tests require real research - set RUN_SLOW_TESTS=1 to enable'
  );

  // Use extended timeout for research queries (research with real LLM + citation pipeline takes 5-10+ min)
  test.setTimeout(900000);

  test('no grey citations after research completes', async ({ chatPage, citationsPage, page }) => {
    // Use a research-worthy query that triggers the full deep research pipeline with citations.
    // Simple factual queries get classified as "simple" by the coordinator, bypassing research.
    const query = RESEARCH_QUERIES[0]; // Complex research topic
    await chatPage.sendMessageWithMode(query.text, 'deep_research');

    // Wait for agent response (10 min - research with real LLM + citation verification takes 3-7+ min)
    await chatPage.waitForAgentResponse(600000);

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
    // Use a research query that triggers deep research with citations
    const query = RESEARCH_QUERIES[1]; // Comparison research
    await chatPage.sendMessageWithMode(query.text, 'deep_research');
    await chatPage.waitForAgentResponse(600000);

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
    // Use a research query that triggers deep research with citations
    const query = RESEARCH_QUERIES[2]; // Current events research
    await chatPage.sendMessageWithMode(query.text, 'deep_research');
    await chatPage.waitForAgentResponse(600000);

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
    // Use a research query that triggers deep research with citations
    const query = RESEARCH_QUERIES[0]; // Complex research topic
    await chatPage.sendMessageWithMode(query.text, 'deep_research');
    await chatPage.waitForAgentResponse(600000);

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
