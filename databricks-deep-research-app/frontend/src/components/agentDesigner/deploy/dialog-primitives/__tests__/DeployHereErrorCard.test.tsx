/**
 * Unit tests for DeployHereErrorCard — one test per error_kind plus edge cases.
 */

import { render, screen, fireEvent } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'
import '@testing-library/jest-dom'

import { DeployHereErrorCard } from '../DeployHereErrorCard'

describe('DeployHereErrorCard', () => {
  it('returns null for empty errorKind', () => {
    const { container } = render(
      <DeployHereErrorCard errorKind="" externalResourceIds={null} />,
    )
    expect(container.firstChild).toBeNull()
  })

  it('missing_workspace_permission — lava card + Switch to Export button calls onAction', () => {
    const onAction = vi.fn()
    render(
      <DeployHereErrorCard
        errorKind="missing_workspace_permission"
        externalResourceIds={null}
        onAction={onAction}
      />,
    )
    expect(screen.getByText(/doesn't have permission/i)).toBeInTheDocument()
    fireEvent.click(screen.getByRole('button', { name: /switch to export/i }))
    expect(onAction).toHaveBeenCalledWith('switch_to_export')
  })

  it('missing_obo_token — yellow card, no action button', () => {
    render(
      <DeployHereErrorCard
        errorKind="missing_obo_token"
        externalResourceIds={null}
      />,
    )
    expect(screen.getByText(/authentication missing/i)).toBeInTheDocument()
    expect(screen.queryByRole('button')).toBeNull()
  })

  it('deploy_already_in_progress — yellow card, no action button', () => {
    render(
      <DeployHereErrorCard
        errorKind="deploy_already_in_progress"
        externalResourceIds={null}
      />,
    )
    expect(screen.getByText(/another deploy is in flight/i)).toBeInTheDocument()
    expect(screen.queryByRole('button')).toBeNull()
  })

  it('artifact_too_large — yellow card, no action button', () => {
    render(
      <DeployHereErrorCard
        errorKind="artifact_too_large"
        externalResourceIds={null}
      />,
    )
    expect(screen.getByText(/too large to deploy/i)).toBeInTheDocument()
    expect(screen.queryByRole('button')).toBeNull()
  })

  it('redeploy_requires_confirmation — blue card + Replace existing calls onAction', () => {
    const onAction = vi.fn()
    render(
      <DeployHereErrorCard
        errorKind="redeploy_requires_confirmation"
        externalResourceIds={null}
        onAction={onAction}
      />,
    )
    expect(screen.getByText(/already deployed/i)).toBeInTheDocument()
    fireEvent.click(screen.getByRole('button', { name: /replace existing/i }))
    expect(onAction).toHaveBeenCalledWith('redeploy_confirmed')
  })

  it('mode_does_not_support_inline_deploy — lava card + Switch to Export calls onAction', () => {
    const onAction = vi.fn()
    render(
      <DeployHereErrorCard
        errorKind="mode_does_not_support_inline_deploy"
        externalResourceIds={null}
        onAction={onAction}
      />,
    )
    expect(screen.getByText(/doesn't support inline deploy/i)).toBeInTheDocument()
    fireEvent.click(screen.getByRole('button', { name: /switch to export/i }))
    expect(onAction).toHaveBeenCalledWith('switch_to_export')
  })

  it('app_name_collision — shows owner + suggested name, calls onSuggestedName AND onAction', () => {
    const onAction = vi.fn()
    const onSuggestedName = vi.fn()
    render(
      <DeployHereErrorCard
        errorKind="app_name_collision"
        externalResourceIds={{
          existing_owner: 'other@user.com',
          suggested_name: 'dr-shell-x-me',
        }}
        appName="dr-shell-myapp"
        onAction={onAction}
        onSuggestedName={onSuggestedName}
      />,
    )
    expect(screen.getByText(/other@user\.com/i)).toBeInTheDocument()
    const useBtn = screen.getByRole('button', { name: /use.*dr-shell-x-me.*instead/i })
    expect(useBtn).toBeInTheDocument()
    fireEvent.click(useBtn)
    expect(onSuggestedName).toHaveBeenCalledWith('dr-shell-x-me')
    expect(onAction).toHaveBeenCalledWith('use_suggested_name')
  })

  it('app_name_collision — falls back to "another user" when existing_owner is null', () => {
    render(
      <DeployHereErrorCard
        errorKind="app_name_collision"
        externalResourceIds={{
          existing_owner: null,
          suggested_name: 'dr-shell-x-me',
        }}
        appName="dr-shell-myapp"
      />,
    )
    expect(screen.getByText(/another user/i)).toBeInTheDocument()
  })

  it('framework_tag_unreachable — shows git tag', () => {
    render(
      <DeployHereErrorCard
        errorKind="framework_tag_unreachable"
        externalResourceIds={{ git_tag: 'v9.9.9' }}
      />,
    )
    expect(screen.getByText(/v9\.9\.9/)).toBeInTheDocument()
    expect(screen.getByText(/not reachable/i)).toBeInTheDocument()
  })

  it('reachability_timeout — renders logs in CodeBlock + truncation hint + Retry button', () => {
    const onAction = vi.fn()
    render(
      <DeployHereErrorCard
        errorKind="reachability_timeout"
        externalResourceIds={{
          last_logs: 'ImportError: cannot import name foo\n',
          logs_truncated: true,
        }}
        onAction={onAction}
      />,
    )
    expect(screen.getByText(/ImportError/)).toBeInTheDocument()
    expect(screen.getByText(/logs truncated/i)).toBeInTheDocument()
    const retryBtn = screen.getByRole('button', { name: /retry deploy/i })
    expect(retryBtn).toBeInTheDocument()
    fireEvent.click(retryBtn)
    expect(onAction).toHaveBeenCalledWith('retry')
  })

  it('unknown error_kind — renders generic lava card', () => {
    render(
      <DeployHereErrorCard
        errorKind="some_future_error"
        externalResourceIds={null}
      />,
    )
    expect(screen.getByText(/some_future_error/)).toBeInTheDocument()
  })
})
