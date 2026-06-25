/**
 * WorkflowSettingsPanel — workflow-level (not per-node) settings, shown in the
 * Designer's no-node-selected inspector (P4).
 *
 * Surfaces the workflow's top-level `mcp_servers`, which are otherwise invisible
 * after save: an "mcp" tool card is lifted into `definition.mcp_servers` by the
 * save normalizer (B2), so post-save they no longer appear in the tools list.
 * This read-only view restores that visibility (each is bound to an agent via the
 * agent inspector's MCP-servers field). Renders nothing when the workflow has no
 * MCP servers, so it never clutters a plain workflow.
 */

import * as React from 'react';
import { useAgentEditorStore } from '@/stores/agentEditorStore';

interface McpServerSummary {
  name?: string;
  client_kind?: string;
  connection_name?: string;
  managed_target?: string;
  url?: string;
}

function describeTarget(server: McpServerSummary): string {
  if (server.client_kind === 'databricks') {
    if (server.connection_name) return `databricks · ${server.connection_name}`;
    if (server.managed_target) return `databricks · ${server.managed_target}`;
    return 'databricks';
  }
  return server.url || 'http';
}

export function WorkflowSettingsPanel(): React.ReactElement | null {
  const ast = useAgentEditorStore((s) => s.ast);
  const raw = ast?.['mcp_servers'];
  const servers: McpServerSummary[] = Array.isArray(raw) ? (raw as McpServerSummary[]) : [];

  if (servers.length === 0) return null;

  return (
    <div className="border-t border-db-gray-lines p-3">
      <p className="mb-1 font-db-mono text-[10px] font-medium uppercase tracking-[0.06em] text-db-gray-text">
        MCP servers ({servers.length})
      </p>
      <p className="mb-2 text-[11px] leading-[1.5] text-db-gray-text">
        Workflow-level MCP servers. Add via the “mcp” tool; bind to an agent in its inspector.
      </p>
      <ul className="space-y-1">
        {servers.map((server, idx) => (
          <li
            key={`${server.name ?? 'mcp'}-${idx}`}
            className="flex items-center justify-between gap-2 rounded-db-md border border-db-gray-lines px-2 py-1 text-[12px] text-db-navy-800"
          >
            <span className="truncate font-medium">{server.name ?? '(unnamed)'}</span>
            <span className="shrink-0 font-db-mono text-[10px] text-db-gray-text">
              {describeTarget(server)}
            </span>
          </li>
        ))}
      </ul>
    </div>
  );
}
