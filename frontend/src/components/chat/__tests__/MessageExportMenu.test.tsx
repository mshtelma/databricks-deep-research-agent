import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { MessageExportMenu } from '../MessageExportMenu';

// Mock the API client
vi.mock('@/api/client', () => ({
  messagesApi: {
    exportReport: vi.fn().mockResolvedValue({
      content: '# Test Report\n\nSome content',
      filename: 'report.md',
    }),
    exportProvenance: vi.fn().mockResolvedValue({
      content: '# Verification Report',
      filename: 'verification.md',
    }),
  },
}));

describe('MessageExportMenu', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('renders the trigger button', () => {
    render(
      <MessageExportMenu messageId="test-id" hasClaims={false} />
    );
    expect(screen.getByLabelText('Export options')).toBeInTheDocument();
  });

  it('opens menu on click', () => {
    render(
      <MessageExportMenu messageId="test-id" hasClaims={false} />
    );
    fireEvent.click(screen.getByLabelText('Export options'));
    expect(screen.getByText('Export Report')).toBeInTheDocument();
    expect(screen.getByText('Copy to Clipboard')).toBeInTheDocument();
  });

  it('shows verification report option when hasClaims is true', () => {
    render(
      <MessageExportMenu messageId="test-id" hasClaims={true} />
    );
    fireEvent.click(screen.getByLabelText('Export options'));
    expect(screen.getByText('Verification Report')).toBeInTheDocument();
  });

  it('hides verification report option when hasClaims is false', () => {
    render(
      <MessageExportMenu messageId="test-id" hasClaims={false} />
    );
    fireEvent.click(screen.getByLabelText('Export options'));
    expect(screen.queryByText('Verification Report')).not.toBeInTheDocument();
  });

  it('closes menu on escape key', () => {
    render(
      <MessageExportMenu messageId="test-id" hasClaims={false} />
    );
    fireEvent.click(screen.getByLabelText('Export options'));
    expect(screen.getByText('Export Report')).toBeInTheDocument();

    fireEvent.keyDown(document, { key: 'Escape' });
    expect(screen.queryByText('Export Report')).not.toBeInTheDocument();
  });

  it('calls exportReport and triggers download on Export Report click', async () => {
    const { messagesApi } = await import('@/api/client');

    render(
      <MessageExportMenu messageId="msg-123" hasClaims={false} />
    );

    fireEvent.click(screen.getByLabelText('Export options'));
    fireEvent.click(screen.getByTestId('export-report-button'));

    await waitFor(() => {
      expect(messagesApi.exportReport).toHaveBeenCalledWith('msg-123');
    });
  });

  it('calls exportReport and copies to clipboard on Copy click', async () => {
    const { messagesApi } = await import('@/api/client');
    const writeText = vi.fn().mockResolvedValue(undefined);
    Object.assign(navigator, { clipboard: { writeText } });

    render(
      <MessageExportMenu messageId="msg-123" hasClaims={false} />
    );

    fireEvent.click(screen.getByLabelText('Export options'));
    fireEvent.click(screen.getByTestId('copy-to-clipboard-button'));

    await waitFor(() => {
      expect(messagesApi.exportReport).toHaveBeenCalledWith('msg-123');
      expect(writeText).toHaveBeenCalledWith('# Test Report\n\nSome content');
    });
  });
});
