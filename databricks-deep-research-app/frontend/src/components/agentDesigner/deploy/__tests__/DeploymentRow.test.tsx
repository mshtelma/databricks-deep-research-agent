/**
 * Tests for DeploymentRow — verifies the action button label resolves
 * correctly per (status, mode), respects the DEACTIVATED greyed state,
 * and propagates onActionClick.
 */

import { fireEvent, render, screen } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'
import '@testing-library/jest-dom'

import type { DeploymentResponse } from '@/types/deployment'

import { DeploymentRow } from '../DeploymentRow'

function makeRow(overrides: Partial<DeploymentResponse>): DeploymentResponse {
  return {
    id: 'dep-1',
    agent_id: 'agent-1',
    revision_id: 'rev-1',
    mode: 'shell_app',
    status: 'active',
    config: { mode: 'shell_app', app_name: 'dr-shell-foo' },
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

describe('DeploymentRow', () => {
  it('renders Undeploy button on an active shell_app row', () => {
    const onAction = vi.fn()
    render(<DeploymentRow deployment={makeRow({})} onActionClick={onAction} />)
    expect(
      screen.getByRole('button', { name: /Undeploy/i }),
    ).toBeInTheDocument()
    expect(screen.getByText('Shell App')).toBeInTheDocument()
    expect(screen.getByText('dr-shell-foo')).toBeInTheDocument()
  })

  it('renders Unregister button on an active in_app row', () => {
    render(
      <DeploymentRow
        deployment={makeRow({ mode: 'in_app', status: 'active' })}
        onActionClick={vi.fn()}
      />,
    )
    expect(
      screen.getByRole('button', { name: /Unregister/i }),
    ).toBeInTheDocument()
  })

  it('renders Cancel button on a pending row', () => {
    render(
      <DeploymentRow
        deployment={makeRow({ status: 'pending' })}
        onActionClick={vi.fn()}
      />,
    )
    expect(
      screen.getByRole('button', { name: /Cancel/i }),
    ).toBeInTheDocument()
  })

  it('renders Retry cleanup button on cleanup_failed', () => {
    render(
      <DeploymentRow
        deployment={makeRow({ status: 'cleanup_failed' })}
        onActionClick={vi.fn()}
      />,
    )
    expect(
      screen.getByRole('button', { name: /Retry cleanup/i }),
    ).toBeInTheDocument()
  })

  it('renders no action button on deactivated rows', () => {
    render(
      <DeploymentRow
        deployment={makeRow({
          status: 'deactivated',
          deactivated_at: '2026-05-25T11:00:00Z',
        })}
        onActionClick={vi.fn()}
      />,
    )
    expect(screen.queryByRole('button')).not.toBeInTheDocument()
    // Row carries the Deactivated label (status badge + timestamp line
    // both mention it — assert at least one match exists).
    expect(screen.getAllByText(/Deactivated/).length).toBeGreaterThan(0)
  })

  it('calls onActionClick with the deployment when the action button is clicked', () => {
    const onAction = vi.fn()
    const deployment = makeRow({})
    render(<DeploymentRow deployment={deployment} onActionClick={onAction} />)
    fireEvent.click(screen.getByRole('button', { name: /Undeploy/i }))
    expect(onAction).toHaveBeenCalledWith(deployment)
  })

  it('disables the action button when isPending', () => {
    render(
      <DeploymentRow
        deployment={makeRow({})}
        onActionClick={vi.fn()}
        isPending
      />,
    )
    const btn = screen.getByRole('button', { name: /Undeploy/i })
    expect(btn).toBeDisabled()
  })

  it('shows Cancelling… badge for cancel_requested + deploying', () => {
    render(
      <DeploymentRow
        deployment={makeRow({ status: 'deploying', cancel_requested: true })}
        onActionClick={vi.fn()}
      />,
    )
    expect(screen.getByText(/Cancelling…/)).toBeInTheDocument()
  })
})
