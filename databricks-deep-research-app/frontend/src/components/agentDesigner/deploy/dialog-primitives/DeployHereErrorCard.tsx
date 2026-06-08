/**
 * DeployHereErrorCard — maps every DeployHereErrorKind to its <InfoCard>.
 *
 * All 9 error kinds from Section S are handled here. The wizard stays thin by
 * delegating all error-kind logic into this single component.
 */

import * as React from 'react'

import { Button } from '@/components/ui/button'
import type { DeployHereErrorKind } from '@/types/deployment'

import { CodeBlock } from './CodeBlock'
import { InfoCard } from './InfoCard'

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export type DeployHereAction =
  | 'switch_to_export'
  | 'use_suggested_name'
  | 'redeploy_confirmed'
  | 'retry'

export interface DeployHereErrorCardProps {
  errorKind: DeployHereErrorKind | string
  externalResourceIds: Record<string, unknown> | null
  appName?: string
  /** Called when the user clicks the action button on the card. */
  onAction?: (action: DeployHereAction) => void
  /** Called with the suggested name for app_name_collision. */
  onSuggestedName?: (name: string) => void
}

// Max bytes to display from last_logs before truncating display (5 KB).
const LOGS_DISPLAY_MAX = 5 * 1024

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export function DeployHereErrorCard({
  errorKind,
  externalResourceIds,
  appName,
  onAction,
  onSuggestedName,
}: DeployHereErrorCardProps): React.ReactElement | null {
  if (!errorKind) return null

  switch (errorKind) {
    case 'missing_workspace_permission':
      return (
        <InfoCard color="lava">
          Your account doesn&apos;t have permission to deploy Databricks Apps in
          this workspace. Contact your workspace admin, or switch to{' '}
          <strong>Export for another workspace</strong>.{' '}
          {onAction && (
            <Button
              variant="outline"
              size="sm"
              onClick={() => onAction('switch_to_export')}
              type="button"
            >
              Switch to Export
            </Button>
          )}
        </InfoCard>
      )

    case 'missing_obo_token':
      return (
        <InfoCard color="yellow">
          Authentication missing — running in Databricks Apps mode but no OBO
          token was forwarded. Refresh the page.
        </InfoCard>
      )

    case 'deploy_already_in_progress':
      return (
        <InfoCard color="yellow">
          Another deploy is in flight. The status panel will update when it
          settles.
        </InfoCard>
      )

    case 'artifact_too_large':
      return (
        <InfoCard color="yellow">
          The agent bundle is too large to deploy (max 50 MB total, 500 MB per
          file). Trim the agent&apos;s tools or files and try again.
        </InfoCard>
      )

    case 'redeploy_requires_confirmation':
      return (
        <InfoCard color="blue">
          An app with this name is already deployed. Confirm replace?{' '}
          {onAction && (
            <Button
              variant="outline"
              size="sm"
              onClick={() => onAction('redeploy_confirmed')}
              type="button"
            >
              Replace existing
            </Button>
          )}
        </InfoCard>
      )

    case 'mode_does_not_support_inline_deploy':
      return (
        <InfoCard color="lava">
          This deployment mode doesn&apos;t support inline deploy. Use{' '}
          <strong>Export for another workspace</strong> to get the artifact.{' '}
          {onAction && (
            <Button
              variant="outline"
              size="sm"
              onClick={() => onAction('switch_to_export')}
              type="button"
            >
              Switch to Export
            </Button>
          )}
        </InfoCard>
      )

    case 'app_name_collision': {
      const existingOwner =
        (externalResourceIds?.existing_owner as string | null | undefined) ??
        null
      const suggestedName =
        (externalResourceIds?.suggested_name as string | undefined) ?? ''
      return (
        <InfoCard color="lava">
          An app named <code>{appName}</code> already exists in this workspace,
          owned by <strong>{existingOwner ?? 'another user'}</strong>. Try a
          different name.{' '}
          {suggestedName && (onAction ?? onSuggestedName) && (
            <Button
              variant="outline"
              size="sm"
              onClick={() => {
                if (onSuggestedName) onSuggestedName(suggestedName)
                if (onAction) onAction('use_suggested_name')
              }}
              type="button"
            >
              Use <code>{suggestedName}</code> instead
            </Button>
          )}
        </InfoCard>
      )
    }

    case 'framework_tag_unreachable': {
      const gitTag =
        (externalResourceIds?.git_tag as string | undefined) ?? 'unknown'
      return (
        <InfoCard color="lava">
          The framework Git ref <code>{gitTag}</code> is not reachable from this
          workspace. Either the ref has been deleted upstream or this workspace
          blocks <code>github.com</code>. Pick a different ref or contact your
          admin.
        </InfoCard>
      )
    }

    case 'reachability_timeout': {
      const lastLogs =
        (externalResourceIds?.last_logs as string | null | undefined) ?? null
      const logsTruncated =
        (externalResourceIds?.logs_truncated as boolean | null | undefined) ??
        false

      const displayLogs =
        lastLogs !== null && lastLogs.length > LOGS_DISPLAY_MAX
          ? lastLogs.slice(0, LOGS_DISPLAY_MAX)
          : lastLogs

      return (
        <InfoCard color="lava">
          <div style={{ fontWeight: 500, marginBottom: 4 }}>
            App didn&apos;t reach RUNNING within timeout.
          </div>
          {displayLogs !== null && (
            <CodeBlock
              code={displayLogs}
              label="last logs"
              multiline={true}
            />
          )}
          {logsTruncated && (
            <div
              style={{
                marginTop: 4,
                fontSize: 11,
                opacity: 0.75,
              }}
            >
              (logs truncated)
            </div>
          )}
          <div style={{ marginTop: 8 }}>
            Open the app in the Databricks UI for full logs.
          </div>
          {onAction && (
            <div style={{ marginTop: 8 }}>
              <Button
                variant="outline"
                size="sm"
                onClick={() => onAction('retry')}
                type="button"
              >
                Retry deploy
              </Button>
            </div>
          )}
        </InfoCard>
      )
    }

    default:
      return (
        <InfoCard color="lava">
          Deployment failed with error: <code>{errorKind}</code>. Please try
          again or contact support.
        </InfoCard>
      )
  }
}
