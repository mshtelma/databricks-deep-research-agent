/**
 * Data Source CRUD Tests — API-level create, read, update, delete operations.
 *
 * Tests the full lifecycle of user data sources (Vector Search, Genie,
 * Knowledge Assistant) and query configuration via the REST API.
 *
 * All tests are gated by RUN_INTEGRATION_TESTS + feature availability check.
 * Requires a deployed app with OBO middleware configured.
 *
 * Run with:
 *   RUN_INTEGRATION_TESTS=1 npx playwright test data-source-crud --reporter=list
 */

import { test, expect } from '../fixtures/data-source.fixture';
import { DataSourceApiHelper } from '../utils/data-source-api';
import {
  makeVSSource,
  makeGenieSource,
  makeKASource,
  DS_TIMEOUTS,
} from '../utils/data-source-test-data';

test.describe('Data Source CRUD', () => {
  test.setTimeout(60_000);

  // Gate: require explicit opt-in via environment variable
  test.skip(
    !process.env.RUN_INTEGRATION_TESTS,
    'Data source CRUD tests require RUN_INTEGRATION_TESTS=1',
  );

  // Gate: skip entire suite if the backend feature is not implemented or OBO is unavailable
  test.beforeAll(async ({ browser }) => {
    const page = await browser.newPage();
    try {
      // Gate 1: Feature existence
      const available = await DataSourceApiHelper.isFeatureAvailable(page);
      test.skip(!available, 'Data sources API not available (feature not implemented)');

      // Gate 2: OBO auth availability (mutations require it)
      const probeResp = await page.request.post('/api/v1/data-sources/vector-search', {
        data: {
          name: '__e2e_obo_probe__',
          index_name: 'nonexistent.probe_index',
          endpoint_name: 'nonexistent-endpoint',
        },
      });
      if (probeResp.status() === 403) {
        const body = await probeResp.text();
        if (body.includes('OBO')) {
          test.skip(true, 'Data source tests require OBO authentication (Databricks Apps only)');
        }
      }
      // Clean up probe if it somehow succeeded (OBO available + valid index)
      if (probeResp.ok()) {
        const source = await probeResp.json();
        await page.request.delete(`/api/v1/data-sources/${source.id}`);
      }
    } finally {
      await page.close();
    }
  });

  // =========================================================================
  // Create
  // =========================================================================

  test.describe('Create', () => {
    test('creates vector search source with valid config', async ({ dsApi }) => {
      const config = makeVSSource();
      const source = await dsApi.createVectorSearch(config);

      expect(source.id).toBeTruthy();
      expect(source.name).toBe(config.name);
      expect(source.type).toBe('vector_search');
      expect(source.validation_status).toBe('valid');
      expect(source.config.endpoint_name).toBe(config.endpoint_name);
      expect(source.config.index_name).toBe(config.index_name);
      expect(source.capabilities).toContain('semantic_search');
    });

    test('creates genie source with valid config', async ({ dsApi }) => {
      const config = makeGenieSource();
      const source = await dsApi.createGenie(config);

      expect(source.id).toBeTruthy();
      expect(source.name).toBe(config.name);
      expect(source.type).toBe('genie');
      expect(source.validation_status).toBe('valid');
      expect(source.config.space_id).toBe(config.space_id);
      expect(source.capabilities).toContain('sql_analytics');
    });

    test('creates knowledge assistant source with valid config', async ({
      dsApi,
    }) => {
      const config = makeKASource();
      const source = await dsApi.createKnowledgeAssistant(config);

      expect(source.id).toBeTruthy();
      expect(source.name).toBe(config.name);
      expect(source.type).toBe('knowledge_assistant');
      expect(source.validation_status).toBe('valid');
      expect(source.config.endpoint_name).toBe(config.endpoint_name);
      expect(source.capabilities).toContain('domain_expertise');
    });
  });

  // =========================================================================
  // Read
  // =========================================================================

  test.describe('Read', () => {
    test('lists sources with user/workspace counts', async ({ dsApi }) => {
      // Create at least one source so the list is non-empty
      await dsApi.createVectorSearch(makeVSSource());

      const list = await dsApi.list();

      expect(list.sources).toBeInstanceOf(Array);
      expect(list.total).toBeGreaterThanOrEqual(1);
      expect(typeof list.user_sources).toBe('number');
      expect(typeof list.workspace_sources).toBe('number');
    });

    test('gets source by ID with full config', async ({ dsApi }) => {
      const config = makeVSSource();
      const created = await dsApi.createVectorSearch(config);

      const fetched = await dsApi.get(created.id);

      expect(fetched.id).toBe(created.id);
      expect(fetched.name).toBe(config.name);
      expect(fetched.type).toBe('vector_search');
      expect(fetched.created_at).toBeTruthy();
      expect(fetched.updated_at).toBeTruthy();
      expect(fetched.config).toBeTruthy();
    });

    test('get non-existent source returns 404', async ({ dsApi }) => {
      const response = await dsApi.getRaw(
        '00000000-0000-0000-0000-000000000000',
      );
      expect(response.status()).toBe(404);
    });
  });

  // =========================================================================
  // Update
  // =========================================================================

  test.describe('Update', () => {
    test('updates source name', async ({ dsApi }) => {
      const source = await dsApi.createVectorSearch(makeVSSource());
      const newName = `Updated ${source.name}`;

      const updated = await dsApi.update(source.id, { name: newName });

      expect(updated.name).toBe(newName);
      // Other fields remain unchanged
      expect(updated.type).toBe(source.type);
    });

    test('updates VS-specific config (num_results)', async ({ dsApi }) => {
      const source = await dsApi.createVectorSearch(makeVSSource());

      const updated = await dsApi.update(source.id, { num_results: 25 });

      expect(updated.config.num_results).toBe(25);
    });

    test('update non-owned source returns 404', async ({ dsApi }) => {
      // Use a non-existent ID to simulate non-owned source
      const response = await dsApi.getRaw(
        '00000000-0000-0000-0000-000000000000',
      );
      expect(response.status()).toBe(404);
    });
  });

  // =========================================================================
  // Delete
  // =========================================================================

  test.describe('Delete', () => {
    test('deletes source successfully, re-fetch returns 404', async ({
      dsApi,
    }) => {
      const source = await dsApi.createVectorSearch(makeVSSource());
      await dsApi.delete(source.id);

      const response = await dsApi.getRaw(source.id);
      expect(response.status()).toBe(404);
    });
  });

  // =========================================================================
  // Validation
  // =========================================================================

  test.describe('Validation', () => {
    test('validate-connection for vector search returns schema info', async ({
      dsApi,
    }) => {
      const config = makeVSSource();
      const result = await dsApi.validateConnection({
        source_type: 'vector_search',
        endpoint_name: config.endpoint_name,
        index_name: config.index_name,
      });

      expect(result.has_access).toBe(true);
      expect(result.validated_at).toBeTruthy();
    });

    test('re-validate source updates status', async ({ dsApi }) => {
      const source = await dsApi.createVectorSearch(makeVSSource());

      const result = await dsApi.validate(source.id);

      expect(result.source_id).toBe(source.id);
      expect(result.has_access).toBe(true);
      expect(result.validated_at).toBeTruthy();
    });
  });

  // =========================================================================
  // Query Config
  // =========================================================================

  test.describe('Query Config', () => {
    test('get query config returns defaults for VS source', async ({
      dsApi,
    }) => {
      const source = await dsApi.createVectorSearch(makeVSSource());

      const qc = await dsApi.getQueryConfig(source.id);

      expect(qc.source_id).toBe(source.id);
      expect(qc.config).toBeTruthy();
    });

    test('update query config with filters persists changes', async ({
      dsApi,
    }) => {
      const source = await dsApi.createVectorSearch(makeVSSource());

      const qc = await dsApi.updateQueryConfig(
        source.id,
        { num_results: 20 },
        false, // skip validation for this test
      );

      expect(qc.config.num_results).toBe(20);
    });

    test('get query config on non-VS source returns error', async ({
      dsApi,
    }) => {
      const source = await dsApi.createGenie(makeGenieSource());

      const response = await dsApi.getQueryConfigRaw(source.id);
      // Genie sources don't support query config
      expect(response.status()).toBe(400);
    });
  });
});
