import { describe, it, expect, vi } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import '@testing-library/jest-dom';
import { ChatOptionsPanel } from '../ChatOptionsPanel';

function renderPanel(overrides: Partial<React.ComponentProps<typeof ChatOptionsPanel>> = {}) {
  const props: React.ComponentProps<typeof ChatOptionsPanel> = {
    tone: '',
    outputLanguage: '',
    onToneChange: vi.fn(),
    onLanguageChange: vi.fn(),
    showVerify: true,
    verifySources: true,
    onVerifyChange: vi.fn(),
    showPlanReview: true,
    enablePlanReview: false,
    onPlanReviewChange: vi.fn(),
    enableCrossSessionMemory: false,
    onCrossSessionMemoryChange: vi.fn(),
    allowLiveSearch: false,
    onAllowLiveSearchChange: vi.fn(),
    ...overrides,
  };
  return { props, ...render(<ChatOptionsPanel {...props} />) };
}

describe('ChatOptionsPanel', () => {
  it('renders an Options button and hides the panel until opened', () => {
    renderPanel();
    expect(screen.getByRole('button', { name: /options/i })).toBeInTheDocument();
    expect(screen.queryByText(/recall facts from my prior chats/i)).not.toBeInTheDocument();
  });

  it('reveals the run-level toggles when opened', () => {
    renderPanel();
    fireEvent.click(screen.getByRole('button', { name: /options/i }));
    expect(screen.getByText(/recall facts from my prior chats/i)).toBeInTheDocument();
    expect(screen.getByText(/allow live web search on follow-ups/i)).toBeInTheDocument();
  });

  it('emits cross-session memory + live-search toggles', () => {
    const { props } = renderPanel();
    fireEvent.click(screen.getByRole('button', { name: /options/i }));
    fireEvent.click(screen.getByText(/recall facts from my prior chats/i).querySelector('input')!);
    expect(props.onCrossSessionMemoryChange).toHaveBeenCalledWith(true);
    fireEvent.click(screen.getByText(/allow live web search on follow-ups/i).querySelector('input')!);
    expect(props.onAllowLiveSearchChange).toHaveBeenCalledWith(true);
  });

  it('shows an active-overrides count badge', () => {
    renderPanel({ enableCrossSessionMemory: true, allowLiveSearch: true });
    // badge text "2" appears in the trigger button
    expect(screen.getByRole('button', { name: /options/i })).toHaveTextContent('2');
  });

  it('hides plan-review + verify rows when their show flags are false', () => {
    renderPanel({ showVerify: false, showPlanReview: false });
    fireEvent.click(screen.getByRole('button', { name: /options/i }));
    expect(screen.queryByText(/verify sources/i)).not.toBeInTheDocument();
    expect(screen.queryByText(/review plan before research/i)).not.toBeInTheDocument();
    // but the run-level overrides remain
    expect(screen.getByText(/recall facts from my prior chats/i)).toBeInTheDocument();
  });
});
