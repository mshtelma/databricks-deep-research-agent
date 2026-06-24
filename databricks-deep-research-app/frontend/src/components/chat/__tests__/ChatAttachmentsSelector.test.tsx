import { describe, it, expect, beforeEach, vi } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import '@testing-library/jest-dom';
import { ChatAttachmentsSelector } from '../ChatAttachmentsSelector';
import { listDesignerResources } from '@/api/agentDesigner';
import { skillFoldersApi } from '@/api/client';

vi.mock('@/api/agentDesigner', () => ({
  listDesignerResources: vi.fn(),
}));
vi.mock('@/api/client', () => ({
  skillFoldersApi: { add: vi.fn() },
}));

const makeRes = () => ({
  resources: [
    { kind: 'skill', source_id: 's1', name: 'market-research', metadata: {} },
    { kind: 'mcp_server', source_id: 'm1', name: 'weather', metadata: {} },
  ],
  total: 2,
});

beforeEach(() => {
  vi.mocked(listDesignerResources).mockResolvedValue(makeRes() as never);
  vi.mocked(skillFoldersApi.add).mockResolvedValue({
    id: 'f1',
    path: '/Workspace/x',
    kind: 'workspace',
  } as never);
});

describe('ChatAttachmentsSelector', () => {
  it('discovers and lists skills + MCP servers when opened', async () => {
    render(
      <ChatAttachmentsSelector
        selectedSkills={[]}
        selectedMcpServers={[]}
        onChange={() => {}}
      />,
    );
    fireEvent.click(screen.getByRole('button', { name: /skills & mcp/i }));
    await waitFor(() =>
      expect(vi.mocked(listDesignerResources)).toHaveBeenCalledWith([
        'skill',
        'mcp_server',
      ]),
    );
    expect(await screen.findByText('market-research')).toBeInTheDocument();
    expect(screen.getByText('weather')).toBeInTheDocument();
  });

  it('emits the toggled skill via onChange', async () => {
    const onChange = vi.fn();
    render(
      <ChatAttachmentsSelector
        selectedSkills={[]}
        selectedMcpServers={[]}
        onChange={onChange}
      />,
    );
    fireEvent.click(screen.getByRole('button', { name: /skills & mcp/i }));
    const skillBox = await screen.findByText('market-research');
    fireEvent.click(skillBox.querySelector('input')!);
    expect(onChange).toHaveBeenCalledWith({
      skills: ['market-research'],
      mcpServers: [],
    });
  });

  it('registers a skill folder via the API', async () => {
    render(
      <ChatAttachmentsSelector
        selectedSkills={[]}
        selectedMcpServers={[]}
        onChange={() => {}}
      />,
    );
    fireEvent.click(screen.getByRole('button', { name: /skills & mcp/i }));
    const input = await screen.findByLabelText(/skill folder path/i);
    fireEvent.change(input, { target: { value: '/Volumes/c/s/v/skills' } });
    fireEvent.click(screen.getByRole('button', { name: /^add$/i }));
    await waitFor(() =>
      expect(vi.mocked(skillFoldersApi.add)).toHaveBeenCalledWith(
        '/Volumes/c/s/v/skills',
        'volume',
      ),
    );
  });
});
