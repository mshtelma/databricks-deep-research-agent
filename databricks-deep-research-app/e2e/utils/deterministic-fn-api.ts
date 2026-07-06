/**
 * API helpers for the deterministic-functions e2e suite.
 *
 * Drives the DEPLOYED app over HTTP (no browser): create a custom agents_v2
 * agent from a full workflow AST, run it as a research job, and read the
 * PERSISTED outcome. Requests are snake_case (the API accepts populate-by-name);
 * responses are camelCase.
 *
 * Assertion surface (deliberate): we poll the persisted chat rather than the
 * SSE stream — these deterministic jobs finish in ~1s and the stream does not
 * replay events for an already-terminal session. Correctness is enforced two
 * ways: tool nodes use fail_on_error:true (a runtime failure fails the job),
 * and python_function bodies embed asserts (a wrong value fails the job). So a
 * terminal status of "completed" already proves the deterministic tools ran and
 * produced the expected values; the report text additionally shows the agent
 * integrated those outputs.
 */
import type { APIRequestContext } from '@playwright/test';
import { expect } from '@playwright/test';

export interface RunOutcome {
  chatId: string;
  agentId: string;
  sessionId: string;
  status: string; // completed | failed | cancelled
  reportText: string;
  sourceUrls: string[];
}

const POLL_INTERVAL_MS = 3_000;
const POLL_MAX_MS = 240_000;
const TERMINAL = new Set(['completed', 'failed', 'cancelled']);

export async function createChat(request: APIRequestContext, title: string): Promise<string> {
  const r = await request.post('/api/v1/chats', { data: { title } });
  expect(r.ok(), `create chat (${r.status()}): ${await r.text()}`).toBeTruthy();
  return (await r.json()).id;
}

export async function createAgent(
  request: APIRequestContext,
  name: string,
  definition: Record<string, unknown>,
): Promise<string> {
  const r = await request.post('/api/v1/agents-v2?validation_mode=advisory', {
    data: { name, definition },
  });
  expect(r.ok(), `create agent (${r.status()}): ${await r.text()}`).toBeTruthy();
  return (await r.json()).id;
}

export async function deleteAgent(request: APIRequestContext, agentId: string): Promise<void> {
  await request.delete(`/api/v1/agents-v2/${agentId}`).catch(() => undefined);
}

/**
 * Submit a research job against a saved agent and poll the persisted chat until
 * the assistant message reaches a terminal state. Returns the terminal status,
 * the report text, and any persisted citation source URLs.
 */
export async function runAgent(
  request: APIRequestContext,
  opts: { chatId: string; agentId: string; query: string },
): Promise<RunOutcome> {
  const submit = await request.post('/api/v1/research/jobs', {
    data: {
      chat_id: opts.chatId,
      agent_id: opts.agentId,
      query: opts.query,
      query_mode: 'deep_research',
      research_depth: 'light',
      verify_sources: false,
    },
  });
  expect(submit.ok(), `submit job (${submit.status()}): ${await submit.text()}`).toBeTruthy();
  const sessionId = (await submit.json()).sessionId as string;

  const deadline = Date.now() + POLL_MAX_MS;
  let status = 'unknown';
  let reportText = '';
  const sourceUrls: string[] = [];
  while (Date.now() < deadline) {
    await new Promise((res) => setTimeout(res, POLL_INTERVAL_MS));
    const full = await request.get(`/api/v1/chats/${opts.chatId}/full`);
    if (!full.ok()) continue;
    const data = await full.json();
    const assistant = (data.messages ?? []).filter(
      (m: Record<string, unknown>) => m.role === 'assistant' || m.role === 'agent',
    );
    if (assistant.length === 0) continue;
    const m = assistant[assistant.length - 1];
    const rs = (m.researchSession ?? m.research_session ?? {}) as Record<string, unknown>;
    const st = String(rs.status ?? '');
    if (TERMINAL.has(st)) {
      status = st;
      reportText = String(m.content ?? '');
      for (const s of (rs.sources as Array<Record<string, unknown>>) ?? []) {
        if (s.url) sourceUrls.push(String(s.url));
      }
      break;
    }
  }

  return {
    chatId: opts.chatId,
    agentId: opts.agentId,
    sessionId,
    status,
    reportText,
    sourceUrls,
  };
}
