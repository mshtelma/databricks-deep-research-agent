/**
 * Tests for UndeployConfirmDialog — verifies mode-specific title + impact
 * text, double-submit guard, Escape close, 409 inline retry banner.
 */

import { fireEvent, render, screen } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'
import '@testing-library/jest-dom'

import { DeploymentApiError } from '@/api/deployments'
import type { DeploymentResponse } from '@/types/deployment'

import { UndeployConfirmDialog } from '../UndeployConfirmDialog'

function makeRow(overrides: Partial<DeploymentResponse>): DeploymentResponse {
  return {
    id: 'dep-1',
    agent_id: 'agent-1',
    revision_id: 'rev-1',
    mode: 'shell_app',
    status: 'active',
    config: {},
    endpoint_name: null,
    model_name: null,
    external_resource_ids: { app_name: 'dr-shell-foo' },
    error_message: null,
    cleanup_attempts: 0,
    cancel_requested: false,
    deployed_by: 'user-1',
    created_at: '2026-05-25T10:00:00Z',
    updated_at: '2026-05-25T10:00:00Z',
    deactivated_at: null,
    ...overrides,
  }
}

describe('UndeployConfirmDialog', () => {
  it('renders title and impact text for shell_app + active', () => {
    render(
      <UndeployConfirmDialog
        deployment={makeRow({})}
        isPending={false}
        error={null}
        onConfirm={vi.fn()}
        onCancel={vi.fn()}
      />,
    )
    expect(screen.getByText(/^Undeploy\?$/)).toBeInTheDocument()
    expect(screen.getByText(/dr-shell-foo/)).toBeInTheDocument()
    expect(screen.getByText(/agent.*preserved/i)).toBeInTheDocument()
  })

  it('renders Unregister title for in_app + active', () => {
    render(
      <UndeployConfirmDialog
        deployment={makeRow({ mode: 'in_app' })}
        isPending={false}
        error={null}
        onConfirm={vi.fn()}
        onCancel={vi.fn()}
      />,
    )
    expect(screen.getByText(/^Unregister\?$/)).toBeInTheDocument()
    expect(screen.getByText(/chat picker/i)).toBeInTheDocument()
  })

  it('renders Cancel title for deploying rows', () => {
    render(
      <UndeployConfirmDialog
        deployment={makeRow({ status: 'deploying' })}
        isPending={false}
        error={null}
        onConfirm={vi.fn()}
        onCancel={vi.fn()}
      />,
    )
    expect(screen.getByText(/^Cancel\?$/)).toBeInTheDocument()
    expect(screen.getByText(/in-flight deployment/i)).toBeInTheDocument()
  })

  it('calls onConfirm when the destructive button is clicked', () => {
    const onConfirm = vi.fn()
    render(
      <UndeployConfirmDialog
        deployment={makeRow({})}
        isPending={false}
        error={null}
        onConfirm={onConfirm}
        onCancel={vi.fn()}
      />,
    )
    fireEvent.click(screen.getByTestId('undeploy-confirm-action'))
    expect(onConfirm).toHaveBeenCalledTimes(1)
  })

  it('disables both buttons while isPending (double-submit guard)', () => {
    render(
      <UndeployConfirmDialog
        deployment={makeRow({})}
        isPending={true}
        error={null}
        onConfirm={vi.fn()}
        onCancel={vi.fn()}
      />,
    )
    expect(screen.getByTestId('undeploy-confirm-action')).toBeDisabled()
    expect(screen.getByTestId('undeploy-confirm-cancel')).toBeDisabled()
  })

  it('closes on Escape', () => {
    const onCancel = vi.fn()
    render(
      <UndeployConfirmDialog
        deployment={makeRow({})}
        isPending={false}
        error={null}
        onConfirm={vi.fn()}
        onCancel={onCancel}
      />,
    )
    fireEvent.keyDown(window, { key: 'Escape' })
    expect(onCancel).toHaveBeenCalled()
  })

  it('does not close on Escape while pending', () => {
    const onCancel = vi.fn()
    render(
      <UndeployConfirmDialog
        deployment={makeRow({})}
        isPending={true}
        error={null}
        onConfirm={vi.fn()}
        onCancel={onCancel}
      />,
    )
    fireEvent.keyDown(window, { key: 'Escape' })
    expect(onCancel).not.toHaveBeenCalled()
  })

  it('renders 409 cleanup-failed inline banner with attempts/max info', () => {
    const error = new DeploymentApiError(409, {
      detail: {
        error_kind: 'deployment_cleanup_failed',
        attempts: 2,
        max_attempts: 3,
        message: 'apps.delete returned 503',
      },
    })
    render(
      <UndeployConfirmDialog
        deployment={makeRow({})}
        isPending={false}
        error={error}
        onConfirm={vi.fn()}
        onCancel={vi.fn()}
      />,
    )
    expect(screen.getByText(/Cleanup failed/i)).toBeInTheDocument()
    expect(screen.getByText(/attempt 2 of 3/i)).toBeInTheDocument()
    expect(screen.getByText(/apps\.delete returned 503/)).toBeInTheDocument()
  })

  it('renders generic error banner for non-cleanup-failed errors', () => {
    const error = new Error('network exploded')
    render(
      <UndeployConfirmDialog
        deployment={makeRow({})}
        isPending={false}
        error={error}
        onConfirm={vi.fn()}
        onCancel={vi.fn()}
      />,
    )
    expect(screen.getByText(/network exploded/)).toBeInTheDocument()
  })
})
