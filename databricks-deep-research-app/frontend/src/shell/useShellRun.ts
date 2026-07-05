/**
 * useShellRun — drives one research run over the shell-app SSE API and adapts
 * its frames to the surface run-state the catalog renders.
 *
 * Ports the proven reconnect/resume protocol from the vanilla shell page: the
 * Databricks Apps gateway imposes an absolute ~4-min cap on a single streamed
 * response, so on a mid-run cut we reconnect via
 * `GET /api/chat/{runId}/events?since=N` and replay from the last delivered id.
 */

import { useCallback, useRef, useState } from 'react';

import type { StructuredOutputEnvelope } from '@/types';
import type { RunReference } from '@/types/surface';

import {
  SHELL_TEMPLATE_VERSION,
  applyEnvelopeToRef,
  parseSseFrame,
  safeJsonParse,
} from './shellApi';

export interface ActivityItem {
  seq: number;
  event: string;
  title: string;
  detail: string;
  status: 'running' | 'done' | 'warning' | 'error' | 'neutral';
}

export interface ShellRun {
  running: boolean;
  report: string;
  activity: ActivityItem[];
  runState: Record<string, RunReference | null>;
  error: string | null;
  banner: 'reconnect' | 'rerun' | null;
  run: (query: string, action: string) => Promise<void>;
  retry: () => void;
}

const RECONNECT_MAX_ATTEMPTS = 6;
const sleep = (ms: number): Promise<void> =>
  new Promise((resolve) => setTimeout(resolve, ms));

function humanize(event: string): string {
  const s = event.replace(/_/g, ' ');
  return s.charAt(0).toUpperCase() + s.slice(1);
}

function eventStatus(event: string): ActivityItem['status'] {
  if (event === 'error') return 'error';
  if (event.endsWith('_completed') || event === 'tool_result') return 'done';
  if (event.endsWith('_started') || event === 'tool_call') return 'running';
  return 'neutral';
}

function eventDetail(payload: Record<string, unknown>): string {
  const raw = payload['raw'];
  if (typeof raw === 'string') return raw.slice(0, 200);
  for (const key of ['label', 'node_id', 'tool_name', 'output_key', 'message']) {
    const v = payload[key];
    if (typeof v === 'string' && v) return v.slice(0, 160);
  }
  return '';
}

export function useShellRun(): ShellRun {
  const [running, setRunning] = useState(false);
  const [report, setReport] = useState('');
  const [activity, setActivity] = useState<ActivityItem[]>([]);
  const [runState, setRunState] = useState<Record<string, RunReference | null>>({});
  const [error, setError] = useState<string | null>(null);
  const [banner, setBanner] = useState<'reconnect' | 'rerun' | null>(null);

  const runIdRef = useRef<string | null>(null);
  const lastSeqRef = useRef<number>(-1);
  const terminalRef = useRef<boolean>(false);
  const expiredRef = useRef<boolean>(false);
  const actionRef = useRef<string>('run');
  const bufferRef = useRef<string>('');
  const lastQueryRef = useRef<string>('');
  const seqRef = useRef<number>(0);

  const setRefStatus = useCallback(
    (action: string, status: RunReference['status']) => {
      setRunState((prev) => ({
        ...prev,
        [action]: { ...(prev[action] ?? { status: 'running' }), status },
      }));
    },
    [],
  );

  const pushActivity = useCallback((event: string, payload: Record<string, unknown>) => {
    seqRef.current += 1;
    const item: ActivityItem = {
      seq: seqRef.current,
      event,
      title: humanize(event),
      detail: eventDetail(payload),
      status: eventStatus(event),
    };
    setActivity((prev) => [...prev, item]);
  }, []);

  const handleFrame = useCallback(
    (event: string, payload: Record<string, unknown>) => {
      if (event === 'workflow_completed') {
        const fr = payload['final_report'];
        if (typeof fr === 'string' && fr) setReport(fr);
        pushActivity(event, payload);
        return;
      }
      if (event === 'structured_output') {
        const env = payload as unknown as StructuredOutputEnvelope;
        const action = typeof env.binding === 'string' ? env.binding : actionRef.current;
        setRunState((prev) => ({
          ...prev,
          [action]: applyEnvelopeToRef(prev[action] ?? { status: 'running' }, env),
        }));
        return;
      }
      if (event === 'complete') {
        terminalRef.current = true;
        const out = payload['output'];
        if (typeof out === 'string' && out) setReport(out);
        setRefStatus(actionRef.current, 'completed');
        return;
      }
      if (event === 'error') {
        terminalRef.current = true;
        const kind = payload['error_kind'];
        const code = payload['code'];
        if (kind === 'expired' || code === 'run_expired') {
          expiredRef.current = true;
          return;
        }
        const message = payload['message'];
        setError(typeof message === 'string' ? message : 'The run failed.');
        setRefStatus(actionRef.current, 'failed');
        pushActivity('error', payload);
        return;
      }
      pushActivity(event, payload);
    },
    [pushActivity, setRefStatus],
  );

  const drainAndHandle = useCallback(
    (chunk: string, flush: boolean) => {
      bufferRef.current = (bufferRef.current + chunk)
        .replace(/\r\n/g, '\n')
        .replace(/\r/g, '\n');
      const frames = bufferRef.current.split(/\n\n+/);
      bufferRef.current = flush ? '' : (frames.pop() ?? '');
      for (const frame of frames) {
        if (!frame.trim()) continue;
        const parsed = parseSseFrame(frame);
        if (!parsed) continue;
        if (parsed.id !== null) lastSeqRef.current = parsed.id;
        handleFrame(parsed.event, safeJsonParse(parsed.data));
      }
    },
    [handleFrame],
  );

  const pump = useCallback(
    async (resp: Response): Promise<'done' | 'cut'> => {
      bufferRef.current = '';
      if (!resp.body) return 'cut';
      const reader = resp.body.getReader();
      const decoder = new TextDecoder();
      try {
        for (;;) {
          const { value, done } = await reader.read();
          if (done) {
            drainAndHandle(decoder.decode(), true);
            break;
          }
          drainAndHandle(decoder.decode(value, { stream: true }), false);
        }
      } catch {
        return 'cut';
      }
      return terminalRef.current ? 'done' : 'cut';
    },
    [drainAndHandle],
  );

  const driveReconnect = useCallback(async () => {
    let attempts = 0;
    while (!terminalRef.current && runIdRef.current && attempts < RECONNECT_MAX_ATTEMPTS) {
      attempts += 1;
      await sleep(Math.min(500 * 2 ** (attempts - 1), 4000));
      try {
        const resp = await fetch(
          `/api/chat/${encodeURIComponent(runIdRef.current)}/events?since=${lastSeqRef.current + 1}`,
          {
            method: 'GET',
            cache: 'no-store',
            headers: {
              Accept: 'text/event-stream',
              'X-Shell-App-Template-Version': SHELL_TEMPLATE_VERSION,
            },
          },
        );
        if (!resp.ok || !resp.body) continue;
        await pump(resp);
      } catch {
        /* retry */
      }
    }
    if (expiredRef.current) setBanner('rerun');
    else if (!terminalRef.current) setBanner(runIdRef.current ? 'reconnect' : 'rerun');
  }, [pump]);

  const run = useCallback(
    async (query: string, action: string) => {
      if (!query.trim()) return;
      lastQueryRef.current = query;
      actionRef.current = action;
      runIdRef.current = null;
      lastSeqRef.current = -1;
      terminalRef.current = false;
      expiredRef.current = false;
      bufferRef.current = '';
      seqRef.current = 0;
      setRunning(true);
      setError(null);
      setBanner(null);
      setReport('');
      setActivity([]);
      setRunState({ [action]: { status: 'running' } });
      try {
        let outcome: 'done' | 'cut' = 'cut';
        try {
          const resp = await fetch('/api/chat', {
            method: 'POST',
            cache: 'no-store',
            headers: {
              Accept: 'text/event-stream',
              'Content-Type': 'application/json',
              'X-Shell-App-Template-Version': SHELL_TEMPLATE_VERSION,
            },
            body: JSON.stringify({ query }),
          });
          runIdRef.current = resp.headers.get('x-shell-app-request-id');
          if (!resp.ok || !resp.body) {
            setError(`Request failed: HTTP ${resp.status}`);
            setRefStatus(action, 'failed');
            return;
          }
          outcome = await pump(resp);
        } catch {
          outcome = 'cut';
        }
        if (outcome === 'cut' && !terminalRef.current && runIdRef.current) {
          await driveReconnect();
          return;
        }
        if (!terminalRef.current) {
          setBanner(runIdRef.current ? 'reconnect' : 'rerun');
        }
      } finally {
        setRunning(false);
      }
    },
    [pump, driveReconnect, setRefStatus],
  );

  const retry = useCallback(() => {
    if (banner === 'reconnect' && runIdRef.current) {
      setBanner(null);
      setRunning(true);
      void driveReconnect().finally(() => setRunning(false));
    } else {
      void run(lastQueryRef.current, actionRef.current);
    }
  }, [banner, driveReconnect, run]);

  return { running, report, activity, runState, error, banner, run, retry };
}
