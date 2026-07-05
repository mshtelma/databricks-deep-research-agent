/**
 * Unit tests for clientMetrics.ts
 *
 * The module reads import.meta.env.VITE_AGENT_DESIGNER_CLIENT_METRICS_ENABLED
 * at call time (not module load time), so we set/unset it per test.
 */

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

// We need access to the internal queue; the module exports flush() so we can
// test it directly. We also re-import after each env mutation via dynamic
// import to keep tests isolated.

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Setup
// ---------------------------------------------------------------------------

// Stub global fetch before any test runs.
const fetchMock = vi.fn().mockResolvedValue(new Response('{}', { status: 200 }))
vi.stubGlobal('fetch', fetchMock)

// We import the module once; its `queue` is module-level state.
// To reset between tests we call flush() to drain.
import {
  CLIENT_SIGNAL_NAMES,
  emit,
  flush,
  startClientMetricsPipeline,
} from '../clientMetrics'

beforeEach(() => {
  fetchMock.mockClear()
  // Drain any leftover events from a prior test.
  void flush()
})

afterEach(() => {
  // Remove env flag so tests are isolated.
  delete (import.meta.env as Record<string, unknown>).VITE_AGENT_DESIGNER_CLIENT_METRICS_ENABLED
})

// ---------------------------------------------------------------------------
// Helper: enable the flag
// ---------------------------------------------------------------------------
function enableFlag(): void {
  const env = import.meta.env as Record<string, unknown>
  env.VITE_AGENT_DESIGNER_CLIENT_METRICS_ENABLED = '1'
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe('emit()', () => {
  it('test_emit_queues_event — emit() flushes on threshold, confirming queue growth', async () => {
    enableFlag()
    // Emit 15 events — threshold is 16, so no flush yet.
    for (let i = 0; i < 15; i++) {
      emit('block_render_count', i)
    }
    // No fetch should have been called yet.
    expect(fetchMock).not.toHaveBeenCalled()
    // Drain so next test starts clean.
    await flush()
  })

  it('test_flush_at_threshold — emitting 16 events triggers an automatic flush', async () => {
    enableFlag()
    for (let i = 0; i < 16; i++) {
      emit('block_render_count', i)
    }
    // flush is async but emit() calls void flush() — give microtasks a tick.
    await Promise.resolve()
    await Promise.resolve()
    expect(fetchMock).toHaveBeenCalledTimes(1)
    const [url, init] = fetchMock.mock.calls[0] as [string, RequestInit]
    expect(url).toBe('/api/v1/metrics/client')
    expect(init.method).toBe('POST')
    const body = JSON.parse(init.body as string) as { events: unknown[] }
    expect(body.events).toHaveLength(16)
    expect(init.keepalive).toBe(true)
  })

  it('test_flush_at_timer — flush() drains queue even with <16 events', async () => {
    enableFlag()
    emit('dnd_drop_failed')
    emit('widget_fallback', 1, { component: 'sidebar' })
    // Manually flush (simulates timer firing).
    await flush()
    expect(fetchMock).toHaveBeenCalledTimes(1)
    const [, init] = fetchMock.mock.calls[0] as [string, RequestInit]
    const body = JSON.parse(init.body as string) as { events: unknown[] }
    expect(body.events).toHaveLength(2)
  })

  it('test_no_op_when_flag_off — emit() is a no-op when env flag is not set', async () => {
    // Flag is not set (afterEach deletes it, beforeEach does not set it).
    emit('block_render_count', 1)
    emit('dnd_drop_failed')
    // Manual flush should find nothing to send.
    await flush()
    expect(fetchMock).not.toHaveBeenCalled()
  })

  it('test_keepalive_on_unload — beforeunload event triggers flush with keepalive=true', async () => {
    enableFlag()
    emit('revisions_tab_opened')
    // Dispatch beforeunload — startClientMetricsPipeline registers the handler.
    // For this test, we call flush() directly (as the listener would).
    const stop = startClientMetricsPipeline()
    // Queue an event after the pipeline starts.
    emit('revisions_tab_opened')
    window.dispatchEvent(new Event('beforeunload'))
    // Allow the async flush to settle.
    await Promise.resolve()
    await Promise.resolve()
    expect(fetchMock).toHaveBeenCalled()
    // Confirm keepalive was set.
    const calls = fetchMock.mock.calls as Array<[string, RequestInit]>
    const initWithKeepAlive = calls.find(([, init]) => init.keepalive === true)
    expect(initWithKeepAlive).toBeDefined()
    stop()
  })

  it('test_signal_name_type_safety — CLIENT_SIGNAL_NAMES contains all expected names', () => {
    // This is a runtime guard that mirrors the compile-time type constraint.
    const expected = [
      'block_render_count',
      'dnd_drop_failed',
      'widget_fallback',
      'token_refresh_attempts',
      'token_refresh_failures',
      'revisions_tab_opened',
      'agent_run_clicked',
      'agent_visibility_changed',
      'surface_preview_real_run',
    ] as const

    for (const name of expected) {
      expect(CLIENT_SIGNAL_NAMES).toContain(name)
    }
    // Ensure the tuple length matches — no extra names sneak in.
    expect(CLIENT_SIGNAL_NAMES).toHaveLength(expected.length)
  })
})
