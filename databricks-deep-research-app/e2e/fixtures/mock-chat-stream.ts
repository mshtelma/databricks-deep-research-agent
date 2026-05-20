/**
 * mock-chat-stream.ts — SSE mock helpers for Agent Designer chat E2E tests.
 *
 * Provides:
 *  - MockSSEEvent interface
 *  - buildSseBody(events)  — serialises events to wire-format SSE text
 *  - mockChatStream(page, events)  — installs a Playwright route interceptor
 *
 * Wire format (mirrors _format_sse in agent_designer.py):
 *   event: <type>\n
 *   data: <json of payload with "type" omitted>\n
 *   \n
 */

import type { Page } from '@playwright/test';

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface MockSSEEvent {
  /** DesignerSSEEvent type string: message | tool_call | mutation_proposed | tool_result | error | done */
  type: string;
  /** All fields of the event payload (excluding "type" — it is carried by the SSE "event:" line). */
  payload: Record<string, unknown>;
}

// ---------------------------------------------------------------------------
// Builder
// ---------------------------------------------------------------------------

/**
 * Serialise an array of MockSSEEvents to a complete SSE body string.
 *
 * Each event becomes:
 *   event: <type>\n
 *   data: <JSON of payload>\n
 *   \n
 */
export function buildSseBody(events: MockSSEEvent[]): string {
  return events
    .map((e) => `event: ${e.type}\ndata: ${JSON.stringify(e.payload)}\n\n`)
    .join('');
}

// ---------------------------------------------------------------------------
// Playwright route helper
// ---------------------------------------------------------------------------

/**
 * Register a Playwright route that intercepts ALL POST requests to the
 * agent-designer chat endpoint and fulfils them with a synthetic SSE stream.
 *
 * Must be called **before** the action that triggers the POST (e.g. typing and
 * clicking Send).
 *
 * @param page   Playwright Page instance.
 * @param events Events to emit in order.
 */
export async function mockChatStream(page: Page, events: MockSSEEvent[]): Promise<void> {
  await page.route('**/api/v1/agent-designer/chat', async (route) => {
    await route.fulfill({
      status: 200,
      contentType: 'text/event-stream',
      body: buildSseBody(events),
    });
  });
}

// ---------------------------------------------------------------------------
// Registry mock helper
// ---------------------------------------------------------------------------

/**
 * Minimal RegistryResponse payload — sufficient for AgentDesignerPage to render
 * without a live server.  Includes just enough node_types / agent_subtypes /
 * tool_kinds for the BlockEditor to categorise agent blocks correctly.
 */
export const MINIMAL_REGISTRY = {
  node_types: [
    { type: 'sequence', label: 'Sequence', icon: 'list', category: 'control_flow', is_composite: true, config_schema: null },
    { type: 'parallel', label: 'Parallel', icon: 'layers', category: 'control_flow', is_composite: true, config_schema: null },
    { type: 'loop', label: 'Loop', icon: 'repeat', category: 'control_flow', is_composite: true, config_schema: null },
    { type: 'conditional', label: 'Conditional', icon: 'git-branch', category: 'control_flow', is_composite: true, config_schema: null },
    { type: 'agent', label: 'Agent', icon: 'bot', category: 'agent', is_composite: false, config_schema: null },
    { type: 'tool', label: 'Tool', icon: 'wrench', category: 'tool', is_composite: false, config_schema: null },
    { type: 'subworkflow', label: 'Subworkflow', icon: 'workflow', category: 'control_flow', is_composite: false, config_schema: null },
    { type: 'plan_and_execute', label: 'Plan & Execute', icon: 'map', category: 'plan_and_execute', is_composite: true, config_schema: null },
  ],
  agent_subtypes: [
    { id: 'researcher', label: 'Researcher', icon: 'search' },
    { id: 'synthesizer', label: 'Synthesizer', icon: 'merge' },
  ],
  tool_kinds: [
    { kind: 'web_search', label: 'Web Search', icon: 'globe' },
  ],
  model_tiers: ['simple', 'analytical', 'complex', 'synthesis'],
  version: '1.0.0',
};

/**
 * Install a Playwright route that serves the minimal registry response.
 * Intercepts GET /api/v1/agent-designer/registry.
 */
export async function mockRegistry(page: Page): Promise<void> {
  await page.route('**/api/v1/agent-designer/registry', async (route) => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify(MINIMAL_REGISTRY),
    });
  });
}
