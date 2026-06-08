/**
 * designer-chat-create.spec.ts
 *
 * E2E tests for the Agent Designer /designer/new chat-create flow.
 *
 * All chat LLM calls are intercepted by page.route() — no live LLM is needed.
 *
 * Test 1: mutation_proposed → Apply → editor shows the new agent block.
 * Test 2: mutation_proposed → Reject → editor is unchanged.
 */

import { test, expect } from '@playwright/test';
import { DesignerPage } from '../pages/designer.page';
import { mockChatStream, mockRegistry, type MockSSEEvent } from '../fixtures/mock-chat-stream';

// ---------------------------------------------------------------------------
// AST shape — inlined to avoid dependency on frontend path aliases.
// ---------------------------------------------------------------------------

interface BlockShape {
  id: string;
  type: string;
  label: string;
  config: Record<string, unknown>;
  children?: BlockShape[];
}

interface ASTShape {
  id: string;
  name: string;
  version: number;
  root: BlockShape;
  tools: unknown[];
  pools: unknown[];
  sources: unknown[];
  models: Record<string, unknown>;
  required_inputs: string[];
  output_keys: string[];
  token_budget: number;
  timeout_seconds: number;
}

// ---------------------------------------------------------------------------
// Deterministic AST used as the new_ast in mutation_proposed
// ---------------------------------------------------------------------------

const NEW_AST: ASTShape = {
  id: 'mock-workflow',
  name: 'Mock Workflow',
  version: 1,
  root: {
    id: 'root-seq-id',
    type: 'sequence',
    label: 'Workflow',
    config: {},
    children: [
      {
        id: 'agent-block-id',
        type: 'agent',
        label: 'My Research Agent',
        config: { model_tier: 'analytical' },
        children: [],
      },
    ],
  },
  tools: [],
  pools: [],
  sources: [],
  models: {},
  required_inputs: ['query'],
  output_keys: ['output'],
  token_budget: 0,
  timeout_seconds: 1800,
};

// ---------------------------------------------------------------------------
// Mock SSE events: message → tool_call → mutation_proposed → done
// ---------------------------------------------------------------------------

function buildChatEvents(): MockSSEEvent[] {
  return [
    {
      type: 'message',
      payload: { content: 'Adding a researcher agent to your workflow.' },
    },
    {
      type: 'tool_call',
      payload: {
        tool_name: 'propose_workflow',
        tool_call_id: 'tc-001',
        args: { description: 'add researcher agent' },
      },
    },
    {
      type: 'mutation_proposed',
      payload: {
        tool_call_id: 'tc-001',
        old_ast: null,
        new_ast: NEW_AST,
        validation_errors: [],
        summary: { node_count: 2, tool_count: 0, source_count: 0 },
      },
    },
    {
      type: 'done',
      payload: {},
    },
  ];
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

test.describe('Designer chat-create flow (mocked SSE)', () => {
  // No live LLM or database needed — all network calls are mocked.
  // The webServer in playwright.config.ts is used when available; these tests
  // also run without one because the registry and chat endpoints are fully mocked.

  test('Apply mutation_proposed updates the block editor with the new agent block', async ({
    page,
  }) => {
    const designer = new DesignerPage(page);

    // Install registry + chat mocks before navigation so they intercept the
    // initial registry fetch and any subsequent chat POST.
    await mockRegistry(page);
    await mockChatStream(page, buildChatEvents());

    await designer.navigateToNew();

    // Wait for the designer header to confirm the page rendered.
    await expect(designer.nameInput).toBeVisible({ timeout: 15000 });

    // Type a prompt and send.
    const chatInput = page.getByLabel('Chat input');
    await chatInput.fill('Add a researcher agent to the workflow');
    await page.getByRole('button', { name: 'Send message' }).click();

    // Wait for the PendingMutationCard to appear (Apply + Reject buttons).
    const applyButton = page.getByRole('button', { name: 'Apply mutation' });
    await expect(applyButton).toBeVisible({ timeout: 10000 });
    await expect(page.getByRole('button', { name: 'Reject mutation' })).toBeVisible();

    // Apply the mutation.
    await applyButton.click();

    // The PendingMutationCard should disappear.
    await expect(applyButton).toBeHidden({ timeout: 5000 });

    // BlockEditor must now show the new agent block label.
    await expect(page.getByText('My Research Agent')).toBeVisible({ timeout: 5000 });
  });

  test('Reject mutation_proposed leaves the block editor unchanged', async ({ page }) => {
    const designer = new DesignerPage(page);

    await mockRegistry(page);
    await mockChatStream(page, buildChatEvents());

    await designer.navigateToNew();
    await expect(designer.nameInput).toBeVisible({ timeout: 15000 });

    // Type a prompt and send.
    const chatInput = page.getByLabel('Chat input');
    await chatInput.fill('Add a researcher agent to the workflow');
    await page.getByRole('button', { name: 'Send message' }).click();

    // Wait for the PendingMutationCard.
    const rejectButton = page.getByRole('button', { name: 'Reject mutation' });
    await expect(rejectButton).toBeVisible({ timeout: 10000 });

    // Reject the mutation.
    await rejectButton.click();

    // PendingMutationCard should disappear.
    await expect(rejectButton).toBeHidden({ timeout: 5000 });

    // The new agent block label must NOT appear — AST was not applied.
    await expect(page.getByText('My Research Agent')).toBeHidden({ timeout: 3000 });

    // The original root sequence label should still be visible, but the proposed child is absent.
    await expect(page.getByText('Workflow', { exact: true }).first()).toBeVisible();
    await expect(page.getByText('My Research Agent')).toBeHidden();
  });
});
