/**
 * designerChatPersistence — client-side (localStorage) persistence for the
 * Designer chat transcript so it survives a page reload (Fix #6).
 *
 * Hardening (per Codex review): versioned schema, key namespaced by session
 * (agent), TTL expiry, try/catch around quota + corrupt JSON, message cap, and
 * redaction/truncation of user text + tool payloads. Pending mutations are NEVER
 * persisted (they would be applyable against a possibly-changed AST).
 */

import type { ChatMessage } from '@/types/agentDesigner';

const SCHEMA_VERSION = 1;
const KEY_PREFIX = `dr.designer.chat.v${SCHEMA_VERSION}`;
const TTL_MS = 1000 * 60 * 60 * 24 * 7; // 7 days
const MAX_MESSAGES = 60;
const MAX_TEXT = 4000;
const MAX_TOOL_CONTENT = 800;
const MAX_TOOL_ARGS = 800;

interface StoredTranscript {
  v: number;
  savedAt: number;
  messages: ChatMessage[];
}

/** Namespace the key by session (agent) id. */
function keyFor(sessionId: string): string {
  return `${KEY_PREFIX}.${sessionId}`;
}

function truncate(s: string, max: number): string {
  return s.length > max ? s.slice(0, max) + '…[truncated]' : s;
}

/** Redact/summarize a message before persisting (caps user text + tool payloads). */
function redactMessage(msg: ChatMessage): ChatMessage {
  const out: ChatMessage = {
    role: msg.role,
    content:
      msg.role === 'tool'
        ? truncate(msg.content ?? '', MAX_TOOL_CONTENT)
        : truncate(msg.content ?? '', MAX_TEXT),
  };
  if (msg.tool_call_id) out.tool_call_id = msg.tool_call_id;
  if (msg.tool_name) out.tool_name = msg.tool_name;
  if (msg.tool_calls && msg.tool_calls.length > 0) {
    out.tool_calls = msg.tool_calls.map((tc) => ({
      id: tc.id,
      type: tc.type,
      function: {
        name: tc.function.name,
        arguments: truncate(tc.function.arguments ?? '', MAX_TOOL_ARGS),
      },
    }));
  }
  return out;
}

export function saveTranscript(sessionId: string | null | undefined, messages: ChatMessage[]): void {
  if (!sessionId || typeof localStorage === 'undefined') return;
  try {
    const trimmed = messages.slice(-MAX_MESSAGES).map(redactMessage);
    const payload: StoredTranscript = { v: SCHEMA_VERSION, savedAt: Date.now(), messages: trimmed };
    localStorage.setItem(keyFor(sessionId), JSON.stringify(payload));
  } catch {
    // Quota exceeded / storage disabled / serialization error — persistence is
    // best-effort and must never break the chat.
  }
}

export function loadTranscript(sessionId: string | null | undefined): ChatMessage[] | null {
  if (!sessionId || typeof localStorage === 'undefined') return null;
  try {
    const raw = localStorage.getItem(keyFor(sessionId));
    if (!raw) return null;
    const parsed = JSON.parse(raw) as StoredTranscript;
    if (!parsed || parsed.v !== SCHEMA_VERSION || !Array.isArray(parsed.messages)) {
      clearTranscript(sessionId);
      return null;
    }
    if (typeof parsed.savedAt === 'number' && Date.now() - parsed.savedAt > TTL_MS) {
      clearTranscript(sessionId);
      return null;
    }
    return parsed.messages;
  } catch {
    // Corrupt JSON — drop it so we don't keep failing.
    clearTranscript(sessionId);
    return null;
  }
}

export function clearTranscript(sessionId: string | null | undefined): void {
  if (!sessionId || typeof localStorage === 'undefined') return;
  try {
    localStorage.removeItem(keyFor(sessionId));
  } catch {
    // ignore
  }
}
