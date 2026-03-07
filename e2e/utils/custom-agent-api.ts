/**
 * AgentApiHelper — centralized API helper for custom agent CRUD operations.
 *
 * Tracks all created agents per test instance and provides cleanupAll()
 * to prevent orphaned agents if a test crashes mid-execution.
 */

import { type APIResponse, type Page, expect } from '@playwright/test';
import type {
  AgentListResponse,
  AgentResponse,
  CustomAgentConfig,
  PresetStepConfig,
  StepResponse,
} from './custom-agent-test-data';

const API_BASE = '/api/v1/custom-agents';

export class AgentApiHelper {
  private readonly page: Page;
  private readonly createdAgentIds: string[] = [];

  constructor(page: Page) {
    this.page = page;
  }

  // ---------------------------------------------------------------------------
  // Feature availability probe
  // ---------------------------------------------------------------------------

  /**
   * Check whether the custom agents API is available.
   * Returns false if the endpoint returns 404 (feature not implemented).
   */
  static async isFeatureAvailable(page: Page): Promise<boolean> {
    const resp = await page.request.get(API_BASE);
    return resp.ok();
  }

  // ---------------------------------------------------------------------------
  // CRUD — tracked (auto-cleanup on cleanupAll)
  // ---------------------------------------------------------------------------

  /** Create an agent and track it for cleanup. */
  async create(config: CustomAgentConfig): Promise<AgentResponse> {
    const response = await this.page.request.post(API_BASE, { data: config });
    const body = await response.text();
    expect(response.ok(), `Create agent failed (${response.status()}): ${body}`).toBe(true);

    const agent: AgentResponse = JSON.parse(body);
    this.createdAgentIds.push(agent.id);
    return agent;
  }

  /** Get an agent by ID. */
  async get(agentId: string): Promise<AgentResponse> {
    const response = await this.page.request.get(`${API_BASE}/${agentId}`);
    const body = await response.text();
    expect(response.ok(), `Get agent failed (${response.status()}): ${body}`).toBe(true);
    return JSON.parse(body);
  }

  /** List all accessible agents. */
  async list(): Promise<AgentListResponse> {
    const response = await this.page.request.get(API_BASE);
    const body = await response.text();
    expect(response.ok(), `List agents failed (${response.status()}): ${body}`).toBe(true);
    return JSON.parse(body);
  }

  /** Update an agent by ID. */
  async update(
    agentId: string,
    updates: Partial<CustomAgentConfig>,
  ): Promise<AgentResponse> {
    const response = await this.page.request.patch(`${API_BASE}/${agentId}`, {
      data: updates,
    });
    const body = await response.text();
    expect(response.ok(), `Update agent failed (${response.status()}): ${body}`).toBe(true);
    return JSON.parse(body);
  }

  /** Delete an agent by ID. Removes it from the tracked list. */
  async delete(agentId: string): Promise<void> {
    const response = await this.page.request.delete(`${API_BASE}/${agentId}`);
    const body = await response.text();
    expect(response.ok(), `Delete agent failed (${response.status()}): ${body}`).toBe(true);

    const idx = this.createdAgentIds.indexOf(agentId);
    if (idx !== -1) {
      this.createdAgentIds.splice(idx, 1);
    }
  }

  // ---------------------------------------------------------------------------
  // Raw responses (no assertions, no tracking — for error-case testing)
  // ---------------------------------------------------------------------------

  /** Create an agent returning the raw APIResponse (no assertion). */
  async createRaw(config: CustomAgentConfig): Promise<APIResponse> {
    return this.page.request.post(API_BASE, { data: config });
  }

  /** Delete an agent returning the raw APIResponse (no assertion). */
  async deleteRaw(agentId: string): Promise<APIResponse> {
    return this.page.request.delete(`${API_BASE}/${agentId}`);
  }

  /** Get an agent returning the raw APIResponse (no assertion). */
  async getRaw(agentId: string): Promise<APIResponse> {
    return this.page.request.get(`${API_BASE}/${agentId}`);
  }

  // ---------------------------------------------------------------------------
  // Preset Steps
  // ---------------------------------------------------------------------------

  /** Create a preset step for an agent. */
  async createStep(agentId: string, step: PresetStepConfig): Promise<StepResponse> {
    const response = await this.page.request.post(`${API_BASE}/${agentId}/steps`, {
      data: step,
    });
    const body = await response.text();
    expect(
      response.ok(),
      `Create step failed (${response.status()}): ${body}`,
    ).toBe(true);
    return JSON.parse(body);
  }

  /** List all preset steps for an agent. */
  async listSteps(agentId: string): Promise<StepResponse[]> {
    const response = await this.page.request.get(`${API_BASE}/${agentId}/steps`);
    const body = await response.text();
    expect(
      response.ok(),
      `List steps failed (${response.status()}): ${body}`,
    ).toBe(true);
    return JSON.parse(body);
  }

  /** Delete a preset step. */
  async deleteStep(agentId: string, stepId: string): Promise<void> {
    const response = await this.page.request.delete(
      `${API_BASE}/${agentId}/steps/${stepId}`,
    );
    const body = await response.text();
    expect(
      response.ok(),
      `Delete step failed (${response.status()}): ${body}`,
    ).toBe(true);
  }

  /** Reorder preset steps by providing an ordered list of step IDs. */
  async reorderSteps(agentId: string, stepIds: string[]): Promise<void> {
    const response = await this.page.request.post(
      `${API_BASE}/${agentId}/steps/reorder`,
      { data: stepIds },
    );
    const body = await response.text();
    expect(
      response.ok(),
      `Reorder steps failed (${response.status()}): ${body}`,
    ).toBe(true);
  }

  // ---------------------------------------------------------------------------
  // Convenience
  // ---------------------------------------------------------------------------

  /**
   * Create an agent with preset steps in one call.
   * Steps are created sequentially to preserve ordering.
   */
  async createWithSteps(
    config: CustomAgentConfig,
    steps: PresetStepConfig[],
  ): Promise<{ agent: AgentResponse; stepIds: string[] }> {
    const agent = await this.create(config);
    const stepIds: string[] = [];

    for (const step of steps) {
      const created = await this.createStep(agent.id, step);
      stepIds.push(created.id);
    }

    return { agent, stepIds };
  }

  // ---------------------------------------------------------------------------
  // Cleanup
  // ---------------------------------------------------------------------------

  /**
   * Delete ALL tracked agents. Safe to call multiple times —
   * ignores 404 responses (agent already deleted).
   */
  async cleanupAll(): Promise<void> {
    const ids = [...this.createdAgentIds];
    this.createdAgentIds.length = 0;

    for (const id of ids) {
      const response = await this.page.request.delete(`${API_BASE}/${id}`);
      // Ignore 404 (already deleted) and 2xx (success)
      if (!response.ok() && response.status() !== 404) {
        console.warn(`Cleanup: failed to delete agent ${id} (${response.status()})`);
      }
    }
  }

  /** Number of agents currently tracked for cleanup. */
  get trackedCount(): number {
    return this.createdAgentIds.length;
  }
}
