/**
 * ShellApp — the deployed standalone agent UI.
 *
 * Reuses the EXACT surface components the main app renders
 * (`AgentSurfacePanel` → `SurfaceRenderer` → catalog + charts) plus
 * `MarkdownRenderer`, wired to the shell-app's simpler SSE API via
 * `useShellRun`. No TanStack/router/persistence — a single agent, no chat
 * history. A `[UI | Chat]` toggle mirrors the main app: UI = form + dashboard,
 * Chat = the full markdown report + live activity.
 */

import * as React from 'react';

import { MarkdownRenderer } from '@/components/common';
import { ErrorBoundary } from '@/components/common/ErrorBoundary';
import { AgentSurfacePanel } from '@/components/surface/AgentSurfacePanel';
import type { CompiledSurfaceSubmission } from '@/lib/surfaceCompile';
import type { RunReference } from '@/types/surface';

import { fetchShellConfig, type ShellConfig } from './shellApi';
import { useShellRun } from './useShellRun';

const DEFAULT_ACTION = 'run';

function ActivityFeed({
  activity,
}: {
  activity: ReturnType<typeof useShellRun>['activity'];
}): React.ReactElement | null {
  if (activity.length === 0) return null;
  const dot: Record<string, string> = {
    running: 'bg-db-blue-500',
    done: 'bg-db-green-600',
    warning: 'bg-db-yellow-700',
    error: 'bg-db-lava-700',
    neutral: 'bg-db-gray-400',
  };
  return (
    <ol className="mt-3 max-h-[420px] space-y-2 overflow-auto rounded-db-md border border-db-gray-lines bg-white p-3">
      {activity.map((a) => (
        <li key={a.seq} className="flex items-start gap-2">
          <span
            aria-hidden
            className={`mt-1.5 h-2 w-2 shrink-0 rounded-full ${dot[a.status] ?? dot['neutral']}`}
          />
          <div className="min-w-0">
            <p className="font-db-sans text-[13px] font-medium text-db-navy-800">
              {a.title}
            </p>
            {a.detail && (
              <p className="truncate font-db-sans text-[12px] text-db-gray-text">
                {a.detail}
              </p>
            )}
          </div>
          <span className="ml-auto shrink-0 font-db-mono text-[10px] text-db-gray-text">
            {a.event}
          </span>
        </li>
      ))}
    </ol>
  );
}

export function ShellApp(): React.ReactElement {
  const [config, setConfig] = React.useState<ShellConfig | null>(null);
  const [configError, setConfigError] = React.useState<string | null>(null);
  const [viewMode, setViewMode] = React.useState<'ui' | 'chat'>('ui');
  const [query, setQuery] = React.useState('');

  const { running, report, activity, runState, error, banner, run, retry } =
    useShellRun();

  React.useEffect(() => {
    fetchShellConfig()
      .then(setConfig)
      .catch((e: unknown) =>
        setConfigError(e instanceof Error ? e.message : String(e)),
      );
  }, []);

  const surface = config?.surface ?? null;
  React.useEffect(() => {
    setViewMode(surface ? 'ui' : 'chat');
  }, [surface]);

  const onRunAction = React.useCallback(
    (compiled: CompiledSurfaceSubmission) => {
      void run(compiled.query, compiled.binding.action);
    },
    [run],
  );

  const resolveRunReference = React.useCallback(
    (ref: RunReference | null): React.ReactNode => {
      if (!ref) return null;
      if (ref.status === 'running') {
        return (
          <div className="flex items-center gap-2 text-[12px] text-db-gray-text">
            <span className="inline-block h-3 w-3 animate-spin rounded-full border-2 border-db-navy-800 border-t-transparent" />
            Running…
          </div>
        );
      }
      if (ref.status === 'failed') {
        return <p className="text-[12px] text-db-lava-700">Run failed</p>;
      }
      if (ref.status === 'completed' && report) {
        return (
          <div className="max-h-[40vh] overflow-auto">
            <MarkdownRenderer content={report} />
          </div>
        );
      }
      return null;
    },
    [report],
  );

  const agentName = config?.agent_name ?? 'Deep Research Agent';

  const runBanner = banner && (
    <div className="mb-3 flex items-center justify-between gap-3 rounded-db-md border border-db-yellow-300 bg-db-yellow-50 px-3 py-2">
      <span className="font-db-sans text-[13px] text-db-yellow-900">
        {banner === 'reconnect'
          ? 'The connection timed out. Your research may still be running — reconnect to continue.'
          : 'The connection was lost. Retry to run your query again.'}
      </span>
      <button
        type="button"
        onClick={retry}
        className="shrink-0 rounded-db-md bg-db-navy-800 px-3 py-1 font-db-sans text-[13px] font-medium text-white hover:bg-db-navy-900"
      >
        Retry
      </button>
    </div>
  );

  if (configError) {
    return (
      <div className="mx-auto max-w-3xl p-6">
        <p className="font-db-sans text-[14px] text-db-lava-700">
          Failed to load the agent: {configError}
        </p>
      </div>
    );
  }

  return (
    <div className="mx-auto flex min-h-screen max-w-5xl flex-col gap-4 p-6">
      <header className="flex items-center justify-between gap-3">
        <div className="min-w-0">
          <p className="font-db-mono text-[10px] font-semibold uppercase tracking-wide text-db-gray-text">
            Deep Research
          </p>
          <h1 className="truncate font-db-sans text-[18px] font-semibold text-db-navy-800">
            {agentName}
          </h1>
        </div>
        {surface && (
          <div
            role="tablist"
            className="inline-flex rounded-db-md border border-db-gray-lines p-0.5"
          >
            {(['ui', 'chat'] as const).map((mode) => (
              <button
                key={mode}
                role="tab"
                aria-selected={viewMode === mode}
                onClick={() => setViewMode(mode)}
                className={`rounded px-3 py-1 font-db-sans text-[13px] font-medium capitalize transition-colors ${
                  viewMode === mode
                    ? 'bg-db-navy-800 text-white'
                    : 'text-db-navy-800 hover:bg-db-oat-medium'
                }`}
              >
                {mode}
              </button>
            ))}
          </div>
        )}
      </header>

      {runBanner}

      {surface && viewMode === 'ui' ? (
        <div className="flex-1 rounded-db-md border border-db-gray-lines bg-white">
          <ErrorBoundary
            name="Surface"
            fallback={
              <div className="p-4 text-sm">
                <p className="mb-1 font-medium text-db-lava-700">
                  This form couldn&apos;t be displayed.
                </p>
                <p className="mb-3 text-db-gray-text">
                  There was a problem rendering the UI — you can still ask in
                  chat.
                </p>
                <button
                  type="button"
                  onClick={() => setViewMode('chat')}
                  className="rounded-db-md bg-db-navy-800 px-3 py-1.5 font-medium text-white hover:bg-db-navy-900"
                >
                  Switch to Chat
                </button>
              </div>
            }
          >
            <AgentSurfacePanel
              agentName={agentName}
              surface={surface}
              onRunAction={onRunAction}
              runDisabled={running}
              runState={runState}
              resolveRunReference={resolveRunReference}
              onClose={() => setViewMode('chat')}
            />
          </ErrorBoundary>
        </div>
      ) : (
        <div className="flex-1">
          {!surface && (
            <div className="mb-3 flex gap-2">
              <input
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault();
                    void run(query, DEFAULT_ACTION);
                  }
                }}
                placeholder="Ask a research question…"
                className="min-w-0 flex-1 rounded-db-md border border-db-gray-lines px-3 py-2 font-db-sans text-[14px] focus:outline-none focus:shadow-db-focus"
              />
              <button
                type="button"
                disabled={running}
                onClick={() => void run(query, DEFAULT_ACTION)}
                className="shrink-0 rounded-db-md bg-db-navy-800 px-4 py-2 font-db-sans text-[14px] font-medium text-white hover:bg-db-navy-900 disabled:opacity-50"
              >
                Send
              </button>
            </div>
          )}
          {error && (
            <p className="mb-3 font-db-sans text-[13px] text-db-lava-700">
              {error}
            </p>
          )}
          {report && (
            <article className="rounded-db-md border border-db-gray-lines bg-white p-5">
              <MarkdownRenderer content={report} />
            </article>
          )}
          <ActivityFeed activity={activity} />
        </div>
      )}
    </div>
  );
}

export default ShellApp;
