import { describe, it, expect, beforeEach } from 'vitest';
import { render, screen } from '@testing-library/react';
import '@testing-library/jest-dom';
import { WorkflowSettingsPanel } from '../WorkflowSettingsPanel';
import { useAgentEditorStore, initialState } from '@/stores/agentEditorStore';
import { createDraftWorkflow } from '@/lib/workflowAst';
import type { AST } from '@/types/ast';

function astWith(mcpServers?: unknown): AST {
  const base = createDraftWorkflow('Test Workflow');
  return { ...base, ...(mcpServers !== undefined ? { mcp_servers: mcpServers } : {}) } as AST;
}

beforeEach(() => {
  useAgentEditorStore.setState(initialState);
});

describe('WorkflowSettingsPanel', () => {
  it('renders nothing when the workflow has no MCP servers', () => {
    useAgentEditorStore.setState({ ast: astWith(undefined) });
    const { container } = render(<WorkflowSettingsPanel />);
    expect(container).toBeEmptyDOMElement();
  });

  it('lists workflow-level MCP servers with their target', () => {
    useAgentEditorStore.setState({
      ast: astWith([
        { name: 'uc-tools', client_kind: 'databricks', connection_name: 'weather_conn' },
        { name: 'managed-fns', client_kind: 'databricks', managed_target: 'functions/main/default' },
        { name: 'third-party', client_kind: 'http', url: 'https://example.com/mcp' },
      ]),
    });
    render(<WorkflowSettingsPanel />);
    expect(screen.getByText(/MCP servers \(3\)/i)).toBeInTheDocument();
    expect(screen.getByText('uc-tools')).toBeInTheDocument();
    expect(screen.getByText(/databricks · weather_conn/)).toBeInTheDocument();
    expect(screen.getByText(/databricks · functions\/main\/default/)).toBeInTheDocument();
    expect(screen.getByText('https://example.com/mcp')).toBeInTheDocument();
  });

  it('tolerates a non-array mcp_servers value', () => {
    useAgentEditorStore.setState({ ast: astWith('oops') });
    const { container } = render(<WorkflowSettingsPanel />);
    expect(container).toBeEmptyDOMElement();
  });
});
