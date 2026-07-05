/**
 * Server-truth reconciliation for surface run status.
 *
 * A surface Run-button run can outlive its SSE stream; a transient client-side
 * `agentStatus='error'` must NOT become a terminal surface `failed`. These pure
 * helpers back the three-layer heal in ChatPage (capture session_id live →
 * inline resolve on error → reconcile on load): all three defer the final run
 * status to the server job status, never to a transient client signal.
 */

import type { PersistedActionRun } from '@/types';
import type { RunReference } from '@/types/surface';

const ALLOWED_PERSISTED_STATUSES = new Set([
  'running',
  'completed',
  'failed',
  'cancelled',
]);

export interface SurfaceRunScope {
  chatId: string;
  agentId: string;
}

export function surfaceRunScopeKey(
  chatId: string | null | undefined,
  agentId: string | null | undefined,
): string | null {
  if (!agentId) return null;
  return `${chatId ?? 'no-chat'}:${agentId}`;
}

export function surfaceRunScopeMatches(
  scope: SurfaceRunScope | null | undefined,
  chatId: string | null | undefined,
  agentId: string | null | undefined,
): boolean {
  return Boolean(scope && agentId && scope.chatId === chatId && scope.agentId === agentId);
}

export function surfaceRunStateFromPersistedActionRuns(
  actionRuns: Record<string, PersistedActionRun> | undefined,
): Record<string, RunReference | null> {
  const seeded: Record<string, RunReference | null> = {};
  if (!actionRuns) return seeded;
  for (const [action, run] of Object.entries(actionRuns)) {
    const rawStatus = run.status ?? '';
    if (!ALLOWED_PERSISTED_STATUSES.has(rawStatus)) continue;
    seeded[action] = {
      status: rawStatus as RunReference['status'],
      ...(run.session_id ? { session_id: run.session_id } : {}),
      ...(run.message_id ? { message_id: run.message_id } : {}),
    };
  }
  return seeded;
}

/**
 * Map a server `Job.status` to a RunReference status. Only `in_progress` (and
 * any unexpected value) is treated as non-terminal → `running` (re-checked
 * later); a known status is never coerced to a blanket `completed`/`failed`.
 * Param is `string` on purpose — the server value is untyped at the boundary.
 */
export function mapJobStatusToRunStatus(jobStatus: string): RunReference['status'] {
  switch (jobStatus) {
    case 'completed':
      return 'completed';
    case 'failed':
      return 'failed';
    case 'cancelled':
      return 'cancelled';
    case 'in_progress':
      return 'running';
    default:
      // Unknown/unexpected status: treat as not-yet-terminal so a later
      // reconcile re-checks it, rather than mislabeling it terminal.
      return 'running';
  }
}

/**
 * Project a run reference to ONLY the fields persisted in `surface_state`
 * (`status`, `session_id?`, `message_id?`, `updated_at`), dropping the
 * enrichment payload (`data`/`sources`/`slotsMeta`) so we never bloat
 * `surface_state` toward the 128 KB PUT cap.
 */
export function toPersistedActionRun(
  ref: RunReference,
  updatedAt: string,
): PersistedActionRun {
  return {
    status: ref.status,
    ...(ref.session_id ? { session_id: ref.session_id } : {}),
    ...(ref.message_id ? { message_id: ref.message_id } : {}),
    updated_at: updatedAt,
  };
}

/**
 * Decide the run-state write when the live stream first provides a `sessionId`
 * (L0 capture). Returns the `running` ref to write, or `null` for "leave the
 * current entry untouched". Order-independent overwrite guard: a run already
 * resolved to a terminal status (completed/failed/cancelled) is NEVER flipped
 * back to `running` — so a post-completion reconnect re-fire can't clobber it —
 * and a `running` entry that already carries this `sessionId` is a no-op.
 */
export function computeCaptureRun(
  current: RunReference | null | undefined,
  sessionId: string,
): RunReference | null {
  if (current && current.status !== 'running') return null; // terminal already resolved
  if (current?.status === 'running' && current.session_id === sessionId) return null; // already stamped
  return { status: 'running', session_id: sessionId };
}

/**
 * Select the persisted action-run entries that should be re-checked against the
 * server: a stale `running`/`failed` entry that carries a `session_id` (needed
 * for `jobsApi.get`). The full `run` is returned so the caller keeps
 * `run.message_id` (the server `Job` payload carries none). Entries without a
 * `session_id` are excluded — they cannot be verified and are left untouched.
 */
export function actionRunsNeedingReconcile(
  actionRuns: Record<string, PersistedActionRun> | undefined,
): Array<{ action: string; sessionId: string; run: PersistedActionRun }> {
  if (!actionRuns) return [];
  const out: Array<{ action: string; sessionId: string; run: PersistedActionRun }> = [];
  for (const [action, run] of Object.entries(actionRuns)) {
    const status = run.status ?? '';
    if ((status === 'running' || status === 'failed') && run.session_id) {
      out.push({ action, sessionId: run.session_id, run });
    }
  }
  return out;
}

/**
 * Live twin of {@link actionRunsNeedingReconcile}: select the IN-MEMORY run
 * refs still `running` (carrying a `session_id`) that must be reconciled against
 * server truth the moment the stream completes.
 *
 * Why this exists: the ONLY live `running → completed` transition is the L1
 * effect, which silently self-cancels on a scope/chat-id mismatch or a missed
 * `persistence_completed` event; the L3 heal that recovers a stuck ref runs only
 * on mount (once per chat+agent). Without a live heal, a run whose L1 missed
 * leaves the ref stuck `running` — and slot enrichment is hard-gated on
 * `completed` — so the surface renders skeletons until a manual page reload.
 *
 * Only `running` is selected (not `failed`): this fires on a SUCCESSFUL stream
 * completion, where a stuck ref is `running`; a `failed` ref would come from the
 * error-path heal (L2), which the mount L3 already reconciles.
 */
export function surfaceRunsNeedingLiveReconcile(
  surfaceRunState: Record<string, RunReference | null>,
): Array<{ action: string; sessionId: string }> {
  const out: Array<{ action: string; sessionId: string }> = [];
  for (const [action, ref] of Object.entries(surfaceRunState)) {
    if (ref && ref.status === 'running' && ref.session_id) {
      out.push({ action, sessionId: ref.session_id });
    }
  }
  return out;
}
