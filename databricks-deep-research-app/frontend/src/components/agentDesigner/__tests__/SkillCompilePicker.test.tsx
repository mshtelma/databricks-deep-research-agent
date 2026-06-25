import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import '@testing-library/jest-dom';
import { SkillCompilePicker } from '../SkillCompilePicker';
import { listDesignerResources } from '@/api/agentDesigner';

vi.mock('@/api/agentDesigner', () => ({ listDesignerResources: vi.fn() }));

beforeEach(() => {
  vi.mocked(listDesignerResources).mockResolvedValue({
    resources: [
      { kind: 'skill', source_id: 's1', name: 'market-research', metadata: {} },
      { kind: 'skill', source_id: 's2', name: 'competitor-scan', metadata: {} },
    ],
    total: 2,
  } as never);
});

describe('SkillCompilePicker', () => {
  it('lists skills (skill kind only) when opened', async () => {
    render(<SkillCompilePicker selected={[]} onChange={() => {}} />);
    fireEvent.click(screen.getByRole('button', { name: /compile skill/i }));
    await waitFor(() =>
      expect(vi.mocked(listDesignerResources)).toHaveBeenCalledWith(['skill']),
    );
    expect(await screen.findByText('market-research')).toBeInTheDocument();
    expect(screen.getByText('competitor-scan')).toBeInTheDocument();
  });

  it('emits the toggled skill name', async () => {
    const onChange = vi.fn();
    render(<SkillCompilePicker selected={[]} onChange={onChange} />);
    fireEvent.click(screen.getByRole('button', { name: /compile skill/i }));
    const row = await screen.findByText('market-research');
    fireEvent.click(row.querySelector('input')!);
    expect(onChange).toHaveBeenCalledWith(['market-research']);
  });

  it('shows the selected count in the trigger', () => {
    render(<SkillCompilePicker selected={['a', 'b']} onChange={() => {}} />);
    expect(screen.getByRole('button', { name: /compile skill/i })).toHaveTextContent('2');
  });
});
