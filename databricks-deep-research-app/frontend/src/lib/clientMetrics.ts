/**
 * Client-side metrics pipeline for agent designer signals.
 *
 * Events are buffered locally and flushed either at FLUSH_THRESHOLD (16) or
 * every FLUSH_INTERVAL_MS (2000 ms), whichever comes first. The pipeline is
 * started once at app startup via startClientMetricsPipeline() and is gated
 * behind the VITE_AGENT_DESIGNER_CLIENT_METRICS_ENABLED flag so the bundle
 * cost is zero when the feature is off.
 */

export const CLIENT_SIGNAL_NAMES = [
  'block_render_count',
  'dnd_drop_failed',
  'widget_fallback',
  'token_refresh_attempts',
  'token_refresh_failures',
  'revisions_tab_opened',
  'agent_run_clicked',
  'agent_visibility_changed',
] as const

export type ClientSignalName = (typeof CLIENT_SIGNAL_NAMES)[number]

interface QueuedEvent {
  name: ClientSignalName
  value?: number
  labels?: Record<string, string>
  timestamp_ms: number
}

const queue: QueuedEvent[] = []
const FLUSH_THRESHOLD = 16
const FLUSH_INTERVAL_MS = 2000

export function emit(
  name: ClientSignalName,
  value?: number,
  labels?: Record<string, string>,
): void {
  if (import.meta.env.VITE_AGENT_DESIGNER_CLIENT_METRICS_ENABLED !== '1') return
  queue.push({ name, value, labels, timestamp_ms: Date.now() })
  if (queue.length >= FLUSH_THRESHOLD) {
    void flush()
  }
}

export async function flush(): Promise<void> {
  if (queue.length === 0) return
  const batch = queue.splice(0, queue.length)
  try {
    await fetch('/api/v1/metrics/client', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ events: batch }),
      keepalive: true, // ship on page unload
    })
  } catch {
    // drop silently — metrics are best-effort
  }
}

export function startClientMetricsPipeline(): () => void {
  const id = setInterval(() => {
    void flush()
  }, FLUSH_INTERVAL_MS)
  window.addEventListener('beforeunload', flush)
  return () => {
    clearInterval(id)
    window.removeEventListener('beforeunload', flush)
  }
}
