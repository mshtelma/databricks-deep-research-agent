import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';

import {
  reportClientError,
  configureClientErrorReporting,
  _resetClientErrorReporting,
} from '../clientErrors';

const ENDPOINT = '/api/v1/observability/client-errors';

function lastBody(fetchMock: ReturnType<typeof vi.fn>): Record<string, unknown> {
  const call = fetchMock.mock.calls.at(-1);
  return JSON.parse((call?.[1] as RequestInit).body as string);
}

describe('reportClientError', () => {
  let fetchMock: ReturnType<typeof vi.fn>;

  beforeEach(() => {
    _resetClientErrorReporting();
    fetchMock = vi.fn().mockResolvedValue({ ok: true });
    vi.stubGlobal('fetch', fetchMock);
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it('does nothing until configured', () => {
    reportClientError({ kind: 'render', message: 'x' });
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it('POSTs a well-shaped payload to the configured endpoint', () => {
    configureClientErrorReporting({ endpoint: ENDPOINT, bundleId: 'b1' });
    reportClientError({
      kind: 'render',
      message: 'boom',
      stack: 'stack-trace',
      componentStack: 'at <Surface>',
      boundaryName: 'Surface',
    });
    expect(fetchMock).toHaveBeenCalledTimes(1);
    expect(fetchMock.mock.calls[0]?.[0]).toBe(ENDPOINT);
    const init = fetchMock.mock.calls[0]?.[1] as RequestInit;
    expect(init.method).toBe('POST');
    expect(init.keepalive).toBe(true);
    const body = lastBody(fetchMock);
    expect(body).toMatchObject({
      kind: 'render',
      message: 'boom',
      stack: 'stack-trace',
      component_stack: 'at <Surface>',
      boundary_name: 'Surface',
      bundle_id: 'b1',
    });
    expect(body.route).toBeDefined();
  });

  it('dedupes identical message+boundary within the window', () => {
    configureClientErrorReporting({ endpoint: ENDPOINT });
    for (let i = 0; i < 3; i++) {
      reportClientError({ kind: 'render', message: 'same', boundaryName: 'A' });
    }
    expect(fetchMock).toHaveBeenCalledTimes(1);
  });

  it('caps at 5 reports per minute', () => {
    configureClientErrorReporting({ endpoint: ENDPOINT });
    for (let i = 0; i < 8; i++) {
      // distinct messages so dedupe does not mask the cap
      reportClientError({ kind: 'render', message: `msg-${i}` });
    }
    expect(fetchMock).toHaveBeenCalledTimes(5);
  });

  it('truncates oversized stacks', () => {
    configureClientErrorReporting({ endpoint: ENDPOINT });
    reportClientError({
      kind: 'render',
      message: 'boom',
      stack: 'x'.repeat(20000),
    });
    const body = lastBody(fetchMock);
    expect((body.stack as string).length).toBeLessThan(20000);
    expect((body.stack as string)).toContain('truncated');
  });

  it('never throws even if fetch throws synchronously', () => {
    configureClientErrorReporting({ endpoint: ENDPOINT });
    vi.stubGlobal(
      'fetch',
      vi.fn(() => {
        throw new Error('network down');
      }),
    );
    expect(() =>
      reportClientError({ kind: 'window_error', message: 'boom' }),
    ).not.toThrow();
  });
});
