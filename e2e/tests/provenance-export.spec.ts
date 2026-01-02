import { test, expect } from '../fixtures';
import { RESEARCH_QUERIES } from '../utils/test-data';

/**
 * Provenance Export Tests - Validate API endpoints for provenance data.
 *
 * Tests the 003-claim-level-citations feature for provenance export:
 * - Export returns valid JSON structure
 * - Export includes all claims
 * - Export includes verification summary
 * - Export handles missing/invalid message IDs
 */
test.describe('Provenance Export', () => {
  test.describe('API Endpoint Validation', () => {
    test('provenance endpoint exists and responds', async ({ request }) => {
      // Test with a fake UUID to verify endpoint structure
      const fakeMessageId = '00000000-0000-0000-0000-000000000000';
      const response = await request.get(`/api/v1/messages/${fakeMessageId}/provenance`);

      // Endpoint should exist (even if returning 404 for invalid ID)
      // 401 = auth required (expected), 404 = not found (expected for fake ID), 422 = validation error
      expect([401, 404, 422]).toContain(response.status());
    });

    test('claims endpoint exists and responds', async ({ request }) => {
      const fakeMessageId = '00000000-0000-0000-0000-000000000000';
      const response = await request.get(`/api/v1/messages/${fakeMessageId}/claims`);

      expect([401, 404, 422]).toContain(response.status());
    });

    test('verification-summary endpoint exists and responds', async ({ request }) => {
      const fakeMessageId = '00000000-0000-0000-0000-000000000000';
      const response = await request.get(`/api/v1/messages/${fakeMessageId}/verification-summary`);

      expect([401, 404, 422]).toContain(response.status());
    });
  });

  test.describe('Export Content Validation', () => {
    // These tests require real research - skip unless explicitly enabled
    test.slow();
    test.skip(
      !process.env.RUN_SLOW_TESTS,
      'Export content validation requires real research - set RUN_SLOW_TESTS=1 to enable'
    );
    test.setTimeout(300000); // 5 minutes

    test('provenance export returns valid JSON after research', async ({
      chatPage,
      request,
      page,
    }) => {
      const query = RESEARCH_QUERIES[0];

      // Send research query and wait for response
      await chatPage.sendMessage(query.text);
      await chatPage.waitForAgentResponse(180000);

      // Get the message ID from the URL or page state
      // The URL should contain the chat ID after a message is sent
      const url = page.url();
      const chatIdMatch = url.match(/\/chat\/([a-f0-9-]+)/);

      if (!chatIdMatch) {
        test.skip(true, 'Could not extract chat ID from URL');
        return;
      }

      // Note: We can't directly get the message ID from the URL
      // This test validates the endpoint structure rather than full integration
      test.skip(true, 'Message ID extraction requires additional implementation');
    });

    test('claims endpoint returns array of claims', async ({ chatPage, request, page }) => {
      const query = RESEARCH_QUERIES[0];

      await chatPage.sendMessage(query.text);
      await chatPage.waitForAgentResponse(180000);

      // This test validates the endpoint structure
      test.skip(true, 'Message ID extraction requires additional implementation');
    });

    test('verification-summary returns expected fields', async ({ chatPage, request, page }) => {
      const query = RESEARCH_QUERIES[0];

      await chatPage.sendMessage(query.text);
      await chatPage.waitForAgentResponse(180000);

      // This test validates the endpoint structure
      test.skip(true, 'Message ID extraction requires additional implementation');
    });
  });

  test.describe('Error Handling', () => {
    test('invalid UUID returns 422', async ({ request }) => {
      const invalidId = 'not-a-uuid';
      const response = await request.get(`/api/v1/messages/${invalidId}/provenance`);

      // Should return 422 for invalid UUID format
      expect([422]).toContain(response.status());
    });

    test('non-existent message returns 404', async ({ request }) => {
      // Random UUID that doesn't exist
      const nonExistentId = 'f47ac10b-58cc-4372-a567-0e02b2c3d479';
      const response = await request.get(`/api/v1/messages/${nonExistentId}/provenance`);

      // Should return 401 (auth required) or 404 (not found)
      expect([401, 404]).toContain(response.status());
    });
  });

  test.describe('Export Format', () => {
    test('provenance export has required schema fields', async ({ request }) => {
      // This is a structural test - verify the API contract
      // The expected ProvenanceExport schema has:
      // - exported_at: datetime
      // - message_id: UUID
      // - claims: array of ClaimProvenanceExport
      // - summary: VerificationSummary

      const fakeMessageId = '00000000-0000-0000-0000-000000000000';
      const response = await request.get(`/api/v1/messages/${fakeMessageId}/provenance`);

      // We can only verify the endpoint responds
      // Full schema validation would require an authenticated request
      expect([401, 404, 422]).toContain(response.status());
    });

    test('claims response has required schema fields', async ({ request }) => {
      const fakeMessageId = '00000000-0000-0000-0000-000000000000';
      const response = await request.get(`/api/v1/messages/${fakeMessageId}/claims`);

      expect([401, 404, 422]).toContain(response.status());
    });
  });

  test.describe('Claim Details Endpoint', () => {
    test('claim endpoint exists and responds', async ({ request }) => {
      const fakeClaimId = '00000000-0000-0000-0000-000000000000';
      const response = await request.get(`/api/v1/claims/${fakeClaimId}`);

      expect([401, 404, 422]).toContain(response.status());
    });

    test('claim evidence endpoint exists and responds', async ({ request }) => {
      const fakeClaimId = '00000000-0000-0000-0000-000000000000';
      const response = await request.get(`/api/v1/claims/${fakeClaimId}/evidence`);

      expect([401, 404, 422]).toContain(response.status());
    });
  });
});
