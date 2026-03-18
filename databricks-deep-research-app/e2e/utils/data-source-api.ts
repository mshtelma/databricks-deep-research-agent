/**
 * DataSourceApiHelper — centralized API helper for data source CRUD operations.
 *
 * Tracks all created sources per test instance and provides cleanupAll()
 * to prevent orphaned sources if a test crashes mid-execution.
 */

import { type APIResponse, type Page, expect } from '@playwright/test';
import type {
  CreateGenieRequest,
  CreateKARequest,
  CreateVSRequest,
  DataSourceListResponse,
  DataSourceResponse,
  DataSourceValidationResponse,
  QueryConfigResponse,
  UpdateDataSourceRequest,
  UpdateQueryConfigRequest,
} from './data-source-test-data';

const API_BASE = '/api/v1/data-sources';

export class DataSourceApiHelper {
  private readonly page: Page;
  private readonly createdSourceIds: string[] = [];

  constructor(page: Page) {
    this.page = page;
  }

  // ---------------------------------------------------------------------------
  // Feature availability probe
  // ---------------------------------------------------------------------------

  /**
   * Check whether the data sources API is available.
   * Returns false if the endpoint returns 404 (feature not implemented).
   */
  static async isFeatureAvailable(page: Page): Promise<boolean> {
    const resp = await page.request.get(API_BASE);
    return resp.ok();
  }

  // ---------------------------------------------------------------------------
  // CRUD — tracked (auto-cleanup on cleanupAll)
  // ---------------------------------------------------------------------------

  /** Create a Vector Search source and track it for cleanup. */
  async createVectorSearch(config: CreateVSRequest): Promise<DataSourceResponse> {
    const response = await this.page.request.post(`${API_BASE}/vector-search`, {
      data: config,
    });
    const body = await response.text();
    expect(
      response.ok(),
      `Create VS source failed (${response.status()}): ${body}`,
    ).toBe(true);

    const source: DataSourceResponse = JSON.parse(body);
    this.createdSourceIds.push(source.id);
    return source;
  }

  /** Create a Genie source and track it for cleanup. */
  async createGenie(config: CreateGenieRequest): Promise<DataSourceResponse> {
    const response = await this.page.request.post(`${API_BASE}/genie`, {
      data: config,
    });
    const body = await response.text();
    expect(
      response.ok(),
      `Create Genie source failed (${response.status()}): ${body}`,
    ).toBe(true);

    const source: DataSourceResponse = JSON.parse(body);
    this.createdSourceIds.push(source.id);
    return source;
  }

  /** Create a Knowledge Assistant source and track it for cleanup. */
  async createKnowledgeAssistant(
    config: CreateKARequest,
  ): Promise<DataSourceResponse> {
    const response = await this.page.request.post(
      `${API_BASE}/knowledge-assistant`,
      { data: config },
    );
    const body = await response.text();
    expect(
      response.ok(),
      `Create KA source failed (${response.status()}): ${body}`,
    ).toBe(true);

    const source: DataSourceResponse = JSON.parse(body);
    this.createdSourceIds.push(source.id);
    return source;
  }

  /** Get a data source by ID. */
  async get(sourceId: string): Promise<DataSourceResponse> {
    const response = await this.page.request.get(`${API_BASE}/${sourceId}`);
    const body = await response.text();
    expect(
      response.ok(),
      `Get source failed (${response.status()}): ${body}`,
    ).toBe(true);
    return JSON.parse(body);
  }

  /** List all accessible data sources. */
  async list(params?: {
    source_type?: string;
    only_valid?: boolean;
    limit?: number;
    offset?: number;
  }): Promise<DataSourceListResponse> {
    const query = new URLSearchParams();
    if (params?.source_type) query.set('source_type', params.source_type);
    if (params?.only_valid !== undefined)
      query.set('only_valid', String(params.only_valid));
    if (params?.limit !== undefined) query.set('limit', String(params.limit));
    if (params?.offset !== undefined) query.set('offset', String(params.offset));

    const url = query.toString() ? `${API_BASE}?${query}` : API_BASE;
    const response = await this.page.request.get(url);
    const body = await response.text();
    expect(
      response.ok(),
      `List sources failed (${response.status()}): ${body}`,
    ).toBe(true);
    return JSON.parse(body);
  }

  /** Update a data source. */
  async update(
    sourceId: string,
    updates: UpdateDataSourceRequest,
  ): Promise<DataSourceResponse> {
    const response = await this.page.request.patch(`${API_BASE}/${sourceId}`, {
      data: updates,
    });
    const body = await response.text();
    expect(
      response.ok(),
      `Update source failed (${response.status()}): ${body}`,
    ).toBe(true);
    return JSON.parse(body);
  }

  /** Delete a data source. Removes it from the tracked list. */
  async delete(sourceId: string): Promise<void> {
    const response = await this.page.request.delete(`${API_BASE}/${sourceId}`);
    expect(
      response.ok(),
      `Delete source failed (${response.status()})`,
    ).toBe(true);

    const idx = this.createdSourceIds.indexOf(sourceId);
    if (idx !== -1) {
      this.createdSourceIds.splice(idx, 1);
    }
  }

  // ---------------------------------------------------------------------------
  // Raw responses (no assertions, no tracking — for error-case testing)
  // ---------------------------------------------------------------------------

  /** Create a VS source returning the raw APIResponse (no assertion). */
  async createVectorSearchRaw(config: CreateVSRequest): Promise<APIResponse> {
    return this.page.request.post(`${API_BASE}/vector-search`, { data: config });
  }

  /** Get a source returning the raw APIResponse (no assertion). */
  async getRaw(sourceId: string): Promise<APIResponse> {
    return this.page.request.get(`${API_BASE}/${sourceId}`);
  }

  /** Delete a source returning the raw APIResponse (no assertion). */
  async deleteRaw(sourceId: string): Promise<APIResponse> {
    return this.page.request.delete(`${API_BASE}/${sourceId}`);
  }

  // ---------------------------------------------------------------------------
  // Validation
  // ---------------------------------------------------------------------------

  /** Re-validate a data source's OBO access. */
  async validate(sourceId: string): Promise<DataSourceValidationResponse> {
    const response = await this.page.request.post(
      `${API_BASE}/${sourceId}/validate`,
    );
    const body = await response.text();
    expect(
      response.ok(),
      `Validate source failed (${response.status()}): ${body}`,
    ).toBe(true);
    return JSON.parse(body);
  }

  /** Test connection before creating a source. */
  async validateConnection(params: {
    source_type: string;
    endpoint_name?: string;
    index_name?: string;
    space_id?: string;
  }): Promise<DataSourceValidationResponse> {
    const query = new URLSearchParams({ source_type: params.source_type });
    if (params.endpoint_name) query.set('endpoint_name', params.endpoint_name);
    if (params.index_name) query.set('index_name', params.index_name);
    if (params.space_id) query.set('space_id', params.space_id);

    const response = await this.page.request.post(
      `${API_BASE}/validate-connection?${query}`,
    );
    const body = await response.text();
    expect(
      response.ok(),
      `Validate connection failed (${response.status()}): ${body}`,
    ).toBe(true);
    return JSON.parse(body);
  }

  // ---------------------------------------------------------------------------
  // Query Config
  // ---------------------------------------------------------------------------

  /** Get query config for a Vector Search source. */
  async getQueryConfig(
    sourceId: string,
    validate = false,
  ): Promise<QueryConfigResponse> {
    const url = validate
      ? `${API_BASE}/${sourceId}/query-config?validate=true`
      : `${API_BASE}/${sourceId}/query-config`;
    const response = await this.page.request.get(url);
    const body = await response.text();
    expect(
      response.ok(),
      `Get query config failed (${response.status()}): ${body}`,
    ).toBe(true);
    return JSON.parse(body);
  }

  /** Get query config returning raw APIResponse (for error testing). */
  async getQueryConfigRaw(sourceId: string): Promise<APIResponse> {
    return this.page.request.get(`${API_BASE}/${sourceId}/query-config`);
  }

  /** Update query config for a Vector Search source. */
  async updateQueryConfig(
    sourceId: string,
    config: UpdateQueryConfigRequest,
    validate = true,
  ): Promise<QueryConfigResponse> {
    const url = `${API_BASE}/${sourceId}/query-config?validate=${validate}`;
    const response = await this.page.request.put(url, { data: config });
    const body = await response.text();
    expect(
      response.ok(),
      `Update query config failed (${response.status()}): ${body}`,
    ).toBe(true);
    return JSON.parse(body);
  }

  // ---------------------------------------------------------------------------
  // Cleanup
  // ---------------------------------------------------------------------------

  /**
   * Delete ALL tracked sources. Safe to call multiple times —
   * ignores 404 responses (source already deleted).
   */
  async cleanupAll(): Promise<void> {
    const ids = [...this.createdSourceIds];
    this.createdSourceIds.length = 0;

    for (const id of ids) {
      const response = await this.page.request.delete(`${API_BASE}/${id}`);
      if (!response.ok() && response.status() !== 404) {
        console.warn(
          `Cleanup: failed to delete data source ${id} (${response.status()})`,
        );
      }
    }
  }

  /** Number of sources currently tracked for cleanup. */
  get trackedCount(): number {
    return this.createdSourceIds.length;
  }
}
