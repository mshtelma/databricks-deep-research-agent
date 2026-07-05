/**
 * Best-effort client-side error reporter (Pillar 3 observability).
 *
 * Ships uncaught errors (React error-boundary catches, window.onerror,
 * unhandledrejection) to the backend so a browser crash — e.g. the `[App]` blank
 * screen — lands in the server logs instead of being invisible to `make logs`.
 *
 * Contract: NEVER throws, NEVER blocks, self-rate-limits (<=5/min), dedupes
 * identical message+boundary within 5s, truncates large stacks, and is
 * endpoint-configurable so the main app and the deployed shell app each point at
 * their own backend route.
 */

export type ClientErrorKind = 'render' | 'window_error' | 'unhandled_rejection';

export interface ClientErrorInput {
  kind: ClientErrorKind;
  message: string;
  stack?: string | null;
  componentStack?: string | null;
  boundaryName?: string | null;
}

const MAX_PER_MIN = 5;
const DEDUPE_MS = 5000;
const MESSAGE_CAP = 2048;
const STACK_CAP = 8192;
const UA_CAP = 256;

let endpoint: string | null = null;
let bundleId: string | null = null;
const sentTimes: number[] = [];
const recent = new Map<string, number>();
let reporting = false;

/** Point the reporter at a backend route. Called once at app bootstrap. */
export function configureClientErrorReporting(opts: {
  endpoint: string;
  bundleId?: string | null;
}): void {
  endpoint = opts.endpoint;
  bundleId = opts.bundleId ?? null;
}

/** Test-only reset of module state. */
export function _resetClientErrorReporting(): void {
  endpoint = null;
  bundleId = null;
  sentTimes.length = 0;
  recent.clear();
  reporting = false;
}

function truncate(
  v: string | null | undefined,
  cap: number,
): string | undefined {
  if (!v) return undefined;
  return v.length <= cap ? v : `${v.slice(0, cap)}…[truncated]`;
}

/** Report one client error. Best-effort: swallows all failures. */
export function reportClientError(input: ClientErrorInput): void {
  try {
    if (reporting || !endpoint) return;
    const now = Date.now();

    // Dedupe identical message+boundary within the window (render loops).
    const key = `${input.boundaryName ?? ''}::${input.message}`;
    const last = recent.get(key);
    if (last !== undefined && now - last < DEDUPE_MS) return;

    // Per-minute cap (never storm the backend).
    while (sentTimes.length > 0 && now - (sentTimes[0] ?? 0) > 60_000) {
      sentTimes.shift();
    }
    if (sentTimes.length >= MAX_PER_MIN) return;
    sentTimes.push(now);
    recent.set(key, now);

    reporting = true;
    const payload = {
      kind: input.kind,
      message: truncate(input.message, MESSAGE_CAP) ?? 'unknown error',
      stack: truncate(input.stack, STACK_CAP),
      component_stack: truncate(input.componentStack, STACK_CAP),
      boundary_name: input.boundaryName ?? undefined,
      route:
        typeof location !== 'undefined' ? location.pathname : undefined,
      bundle_id: bundleId ?? undefined,
      user_agent:
        typeof navigator !== 'undefined'
          ? navigator.userAgent.slice(0, UA_CAP)
          : undefined,
    };
    void fetch(endpoint, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
      keepalive: true,
    }).catch(() => {
      /* best-effort */
    });
  } catch {
    /* reporting must never throw (would recurse via window.onerror) */
  } finally {
    reporting = false;
  }
}

/** Install window-level handlers. Returns a disposer (used by tests). */
export function installGlobalErrorReporting(): () => void {
  const onError = (e: ErrorEvent): void => {
    const err = e.error;
    reportClientError({
      kind: 'window_error',
      message: e.message || 'window error',
      stack: err instanceof Error ? (err.stack ?? null) : null,
    });
  };
  const onRejection = (e: PromiseRejectionEvent): void => {
    const reason: unknown = e.reason;
    reportClientError({
      kind: 'unhandled_rejection',
      message:
        reason instanceof Error
          ? reason.message
          : String(reason ?? 'unhandled rejection'),
      stack: reason instanceof Error ? (reason.stack ?? null) : null,
    });
  };
  window.addEventListener('error', onError);
  window.addEventListener('unhandledrejection', onRejection);
  return () => {
    window.removeEventListener('error', onError);
    window.removeEventListener('unhandledrejection', onRejection);
  };
}
