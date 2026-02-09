import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';

// Mock fetch globally for AIditor API calls
const mockFetch = vi.fn();
global.fetch = mockFetch;

// Mock localStorage
const localStorageMock = (() => {
  let store: Record<string, string> = {};
  return {
    getItem: vi.fn((key: string) => store[key] ?? null),
    setItem: vi.fn((key: string, value: string) => { store[key] = value; }),
    removeItem: vi.fn((key: string) => { delete store[key]; }),
    clear: vi.fn(() => { store = {}; }),
  };
})();
Object.defineProperty(window, 'localStorage', { value: localStorageMock });

// Must import AFTER mocks are set up
import { AIditor } from '../aiditor';

describe('AIditor Chat Integration', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    localStorageMock.clear();
    // Default fetch responses for models and MCP endpoints
    mockFetch.mockImplementation((url: string) => {
      if (typeof url === 'string' && url.includes('/models')) {
        return Promise.resolve({
          ok: true,
          json: () => Promise.resolve({
            models: [{ name: 'test-model', display_name: 'Test Model', status: 'READY', task: 'llm/v1/chat' }],
            default_model: 'test-model',
          }),
        });
      }
      if (typeof url === 'string' && url.includes('/mcp/endpoints')) {
        return Promise.resolve({
          ok: true,
          json: () => Promise.resolve({
            genie_spaces: [],
            vector_indexes: [],
            external_connections: [],
            knowledge_assistants: [],
          }),
        });
      }
      return Promise.resolve({ ok: false, text: () => Promise.resolve('Not found') });
    });
  });

  it('renders without crashing', async () => {
    render(<AIditor />);
    // The heading renders "AI" + "ditor" in separate spans, use subtitle instead
    await waitFor(() => {
      expect(screen.getByText('AI-Assisted Markdown Editor')).toBeInTheDocument();
    });
  });

  it('loads initialContent into the editor', async () => {
    const onContentConsumed = vi.fn();
    render(
      <AIditor
        initialContent="# Hello from Chat"
        onContentConsumed={onContentConsumed}
      />
    );

    await waitFor(() => {
      expect(onContentConsumed).toHaveBeenCalled();
    });

    // The markdown editor textarea should contain the initial content
    const textarea = screen.getByRole('textbox');
    expect(textarea).toHaveValue('# Hello from Chat');
  });

  it('does not load when initialContent is null', () => {
    const onContentConsumed = vi.fn();
    render(
      <AIditor
        initialContent={null}
        onContentConsumed={onContentConsumed}
      />
    );

    expect(onContentConsumed).not.toHaveBeenCalled();
  });

  it('shows Export to Chat button when onExportToChat is provided', async () => {
    const onExportToChat = vi.fn();
    render(<AIditor onExportToChat={onExportToChat} />);

    await waitFor(() => {
      expect(screen.getByText('Export to Chat')).toBeInTheDocument();
    });
  });

  it('does not show Export to Chat button when onExportToChat is not provided', async () => {
    render(<AIditor />);

    // Wait for component to finish loading
    await waitFor(() => {
      expect(screen.getByText('AI-Assisted Markdown Editor')).toBeInTheDocument();
    });

    expect(screen.queryByText('Export to Chat')).not.toBeInTheDocument();
  });

  it('calls onExportToChat with current markdown when button clicked', async () => {
    const onExportToChat = vi.fn();
    const user = userEvent.setup();

    render(
      <AIditor
        initialContent="# Edited Content"
        onContentConsumed={() => {}}
        onExportToChat={onExportToChat}
      />
    );

    await waitFor(() => {
      expect(screen.getByText('Export to Chat')).toBeInTheDocument();
    });

    await user.click(screen.getByText('Export to Chat'));

    expect(onExportToChat).toHaveBeenCalledWith('# Edited Content');
  });

  it('does not re-inject initialContent after consumed', async () => {
    const onContentConsumed = vi.fn();
    const { rerender } = render(
      <AIditor
        initialContent="# First load"
        onContentConsumed={onContentConsumed}
      />
    );

    await waitFor(() => {
      expect(onContentConsumed).toHaveBeenCalledTimes(1);
    });

    // Re-render with null (consumed)
    rerender(
      <AIditor
        initialContent={null}
        onContentConsumed={onContentConsumed}
      />
    );

    // Should not call again
    expect(onContentConsumed).toHaveBeenCalledTimes(1);
  });
});
