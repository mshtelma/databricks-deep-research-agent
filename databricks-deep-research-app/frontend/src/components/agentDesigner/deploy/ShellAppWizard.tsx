/**
 * ShellAppWizard — dual-mode Deploy as Databricks App dialog (C3).
 *
 * Mode "here"  → real auto-deploy via POST /deployments + /actions/deploy-here
 *                with full error handling (redeploy confirm, permission, etc.).
 * Mode "other" → zip-download flow: POST /deployments + GET /{id}/export-zip.
 *
 * Plan reference: we-don-t-need-legacy-composed-wren.md Section P.
 */

import * as React from 'react'
import * as RadixDialog from '@radix-ui/react-dialog'
import { Box, CheckCircle } from 'lucide-react'

import { Button } from '@/components/ui/button'
import {
  createDeployment,
  deployHereAction,
  DeploymentActionError,
  formatDefaultRevisionNotDeployableError,
} from '@/api/deployments'
import {
  useCanDeployHere,
  useDeploymentDefaults,
  useDeploymentStatusPoll,
  useRefreshCanDeployHere,
} from '@/hooks/useDeployments'
import type {
  CanDeployHereResponse,
  DeployHereErrorKind,
  DeployHereProbeStatus,
  DeploymentResponse,
} from '@/types/deployment'
import { RevisionProvenanceCard } from './RevisionProvenanceCard'
import type { RevisionProvenance } from './revisionProvenance'

import {
  DeployHereErrorCard,
  DialogShell,
  ModeTabs,
  ProgressList,
  Step,
  InfoCard,
  FileTree,
  SectionTitle,
  HostField,
  type ProgressStep,
} from './dialog-primitives'

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

type Mode = 'here' | 'other'
type Phase = 'idle' | 'running' | 'done' | 'error'

export interface ShellAppWizardProps {
  open: boolean
  onOpenChange: (open: boolean) => void
  agentId: string
  agentName: string
  revisionId: string
  revisionProvenance?: RevisionProvenance | null
  /** Fired with the created deployment row after a successful POST. */
  onDeployed: (deployment: DeploymentResponse) => void
}

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const APP_NAME_PATTERN = /^(?=.{2,30}$)dr-shell-[a-z0-9-]+$/
const APP_NAME_MAX_LENGTH = 30

function defaultAppName(agentId: string): string {
  return `dr-shell-${agentId.slice(0, 8)}`
}

function inferProbeStatus(
  data: CanDeployHereResponse | undefined,
): DeployHereProbeStatus | null {
  if (!data) return null
  if (data.probe_status) return data.probe_status
  if (
    data.reason === 'missing_workspace_permission' ||
    data.reason === 'missing_obo_token'
  ) {
    return 'denied'
  }
  if (data.can_deploy) return 'ok'
  return 'unknown'
}

function formatRefreshError(error: unknown): string {
  return error instanceof Error ? error.message : 'Permission re-check failed.'
}

function formatCreateDeploymentError(error: unknown): string {
  const blocked = formatDefaultRevisionNotDeployableError(error)
  if (blocked) return blocked
  return error instanceof Error ? error.message : 'Failed to create deployment row.'
}

const DEPLOY_STEPS: ProgressStep[] = [
  {
    id: 's1',
    label: 'Building zip artifact (agent code + chat UI)',
    detail: 'src/, app.yaml, databricks.yml, requirements.txt',
  },
  {
    id: 's2',
    label: 'Uploading source to workspace',
    detail: '/Workspace/Users/me/.bundle/{appName}/{target}',
  },
  {
    id: 's3',
    label: 'Creating Apps compute (or attaching existing)',
    detail: 'databricks apps create {appName}',
  },
  {
    id: 's4',
    label: 'Wiring resources (UC, serving endpoints, secrets)',
    detail: 'OBO scopes: apps · sql · serving-endpoints',
  },
  {
    id: 's5',
    label: 'Deploying & starting app (this takes ~2 min)',
    detail: 'databricks bundle run {appName}',
  },
  {
    id: 's6',
    label: 'Health check on / and /chat',
    detail: 'HTTP 200 · streaming OK',
  },
]

const BUNDLE_FILES = [
  { path: 'app.py', note: 'FastAPI entry point' },
  { path: 'app.yaml', note: 'Apps entry point + env vars' },
  { path: 'agent.yaml', note: 'Agent definition' },
  { path: 'databricks.yml', note: 'DAB config — defines the app resource' },
  { path: 'pyproject.toml', note: 'Pinned to framework via Git ref' },
  { path: 'entrypoint.sh', note: 'Container start script' },
  { path: 'static/index.html', note: 'Bundled chat UI' },
  { path: 'README.md', note: 'Same instructions as below' },
]

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export function ShellAppWizard({
  open,
  onOpenChange,
  agentId,
  agentName,
  revisionId,
  revisionProvenance = null,
  onDeployed,
}: ShellAppWizardProps): React.ReactElement {
  // ---- form state ----
  const [appName, setAppName] = React.useState(() => defaultAppName(agentId))
  const [gitTag, setGitTag] = React.useState('')
  const [target, setTarget] = React.useState('dev')

  // ---- mode / phase state ----
  const [mode, setMode] = React.useState<Mode>('other')
  const [phase, setPhase] = React.useState<Phase>('idle')
  const [stepIdx, setStepIdx] = React.useState(0)
  const [appUrl, setAppUrl] = React.useState<string | null>(null)

  // ---- "other" mode zip state ----
  const [created, setCreated] = React.useState<DeploymentResponse | null>(null)
  const [otherPending, setOtherPending] = React.useState(false)
  const [otherError, setOtherError] = React.useState<string | null>(null)

  // ---- host field for "other" mode ----
  const [host, setHost] = React.useState('https://acme-prod.cloud.databricks.com')

  // ---- "here" mode state ----
  const [hereDeploymentId, setHereDeploymentId] = React.useState<string | null>(null)
  const [redeployModalOpen, setRedeployModalOpen] = React.useState(false)
  const [redeployRowId, setRedeployRowId] = React.useState<string | null>(null)
  const [hereError, setHereError] = React.useState<React.ReactNode | null>(null)

  // ---- hooks ----
  const { data: defaults } = useDeploymentDefaults()
  const defaultGitTag = defaults?.frameworkGitTag ?? ''

  // ---- can-deploy-here probe ----
  const canDeployHereQuery = useCanDeployHere()
  const canDeployHereData = canDeployHereQuery.data
  const hereDisabledReason = canDeployHereData?.reason ?? null
  const probeStatus = inferProbeStatus(canDeployHereData)
  const hereDisabled =
    probeStatus === 'denied' ||
    hereDisabledReason === 'missing_workspace_permission' ||
    hereDisabledReason === 'missing_obo_token'
  const probeUnknown = probeStatus === 'unknown'
  const isProbing = canDeployHereQuery.isLoading
  const isSpFallback = canDeployHereData?.actor === 'sp_fallback'
  const refreshCanDeployHere = useRefreshCanDeployHere()

  // Diagnostic — surfaces in the browser devtools so we can verify the hook
  // fires AND see what the backend returned without needing server logs.
  // Marked with a stable prefix so it can be grepped in screenshots / video.
  React.useEffect(() => {
    console.log(
      '[CAN_DEPLOY_HERE_CLIENT] state=',
      canDeployHereQuery.status,
      'data=',
      canDeployHereData,
      'error=',
      canDeployHereQuery.error?.message,
    )
  }, [canDeployHereQuery.status, canDeployHereData, canDeployHereQuery.error])

  // Tracks whether the user has manually typed in the gitTag field since the
  // dialog last opened. Once true, neither the on-open reset nor a later
  // defaults resolution may overwrite the field — W1 regression fix.
  const userEditedGitTag = React.useRef(false)

  // Timer ref for wall-clock step advancement during 'deploying' status.
  const deployTimer = React.useRef<ReturnType<typeof setTimeout> | null>(null)

  // Status poll — active while phase === 'running' and we have a deployment id.
  const { data: pollData } = useDeploymentStatusPoll(hereDeploymentId, {
    enabled: phase === 'running',
  })

  // ---- reset on open/close ----
  React.useEffect(() => {
    if (open) {
      setAppName(defaultAppName(agentId))
      if (!userEditedGitTag.current) {
        setGitTag(defaultGitTag)
      }
      setCreated(null)
      setOtherPending(false)
      setOtherError(null)
      // When the probe is already settled, choose the mode that's actually
      // available. If the user has the right permissions ('here' is enabled),
      // default to 'here' so the one-click path is front-and-center; otherwise
      // fall back to 'other'. While the probe is loading (data still undefined)
      // we default to 'other' to avoid flicker once it resolves to 'false'.
      setMode(probeStatus !== null && !hereDisabled ? 'here' : 'other')
      setPhase('idle')
      setStepIdx(0)
      setAppUrl(null)
      setHereDeploymentId(null)
      setHereError(null)
      setRedeployModalOpen(false)
      setRedeployRowId(null)
    } else {
      userEditedGitTag.current = false
      if (deployTimer.current !== null) {
        clearTimeout(deployTimer.current)
        deployTimer.current = null
      }
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [open, agentId, defaultGitTag])

  // Clear timer on unmount.
  React.useEffect(() => {
    return () => {
      if (deployTimer.current !== null) clearTimeout(deployTimer.current)
    }
  }, [])

  // ---- status-poll → stepIdx/phase mapping ----
  React.useEffect(() => {
    if (!pollData || phase !== 'running') return
    const status = pollData.status
    if (status === 'pending') {
      setStepIdx(0)
    } else if (status === 'deploying') {
      setStepIdx((prev) => Math.max(prev, 1))
      if (deployTimer.current === null) {
        let i = 1
        const intervalMs = 20_000 / (DEPLOY_STEPS.length - 1)
        const tick = () => {
          i += 1
          if (i < DEPLOY_STEPS.length) {
            setStepIdx(i)
            deployTimer.current = setTimeout(tick, intervalMs)
          }
        }
        deployTimer.current = setTimeout(tick, intervalMs)
      }
    } else if (status === 'active') {
      if (deployTimer.current !== null) {
        clearTimeout(deployTimer.current)
        deployTimer.current = null
      }
      setStepIdx(DEPLOY_STEPS.length)
      setPhase('done')
      const extIds = (pollData as unknown as { external_resource_ids?: Record<string, string> })
        .external_resource_ids
      setAppUrl(extIds?.app_url ?? null)
    } else if (status === 'failed') {
      if (deployTimer.current !== null) {
        clearTimeout(deployTimer.current)
        deployTimer.current = null
      }
      setPhase('error')
    }
  }, [pollData, phase])

  // ---- form handlers ----
  const handleGitTagChange = React.useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      userEditedGitTag.current = true
      setGitTag(e.target.value)
    },
    [],
  )

  const appNameValid = APP_NAME_PATTERN.test(appName)
  const submitDisabled = !appNameValid || gitTag.trim() === '' || otherPending

  // ---- "other" mode: generate bundle ----
  const handleSubmit = React.useCallback(async () => {
    if (submitDisabled) return
    setOtherPending(true)
    setOtherError(null)
    try {
      const deployment = await createDeployment({
        agent_id: agentId,
        revision_id: revisionId,
        config: {
          mode: 'shell_app',
          app_name: appName,
          framework_git_tag: gitTag,
          target,
        },
      })
      setCreated(deployment)
      onDeployed(deployment)
    } catch (err) {
      setOtherError(formatCreateDeploymentError(err))
    } finally {
      setOtherPending(false)
    }
  }, [agentId, appName, gitTag, onDeployed, revisionId, submitDisabled, target])

  // ---- "other" mode: download zip ----
  const handleDownload = React.useCallback(async () => {
    if (!created) return
    const resp = await fetch(`/api/v1/deployments/${created.id}/export-zip`)
    if (!resp.ok) return
    const blob = await resp.blob()
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `${appName}.zip`
    document.body.appendChild(a)
    a.click()
    a.remove()
    URL.revokeObjectURL(url)
  }, [appName, created])

  // ---- "here" mode: real deploy flow (two explicit steps to capture pending row ID) ----
  const handleDeployHere = React.useCallback(async () => {
    setPhase('running')
    setStepIdx(0)
    setHereError(null)

    // Step 1: create the deployment row
    let pendingRow: DeploymentResponse
    try {
      pendingRow = await createDeployment(
        {
          agent_id: agentId,
          revision_id: revisionId,
          config: {
            mode: 'shell_app',
            app_name: appName,
            framework_git_tag: gitTag,
            target,
          },
        },
        {
          runAsync: false,
        },
      )
    } catch (err) {
      setPhase('error')
      setHereError(
        <InfoCard color="lava">
          {formatCreateDeploymentError(err)}
        </InfoCard>,
      )
      return
    }

    // Step 2: trigger the inline deploy action
    try {
      const dep = await deployHereAction(pendingRow.id)
      // If the action returned ACTIVE immediately, resolve right away without polling
      if (dep.status === 'active') {
        const extIds = dep.external_resource_ids as Record<string, string> | null
        setAppUrl(extIds?.app_url ?? null)
        setStepIdx(DEPLOY_STEPS.length)
        setPhase('done')
      } else if (dep.status === 'failed') {
        setPhase('error')
        setHereError(
          <InfoCard color="lava">
            {dep.error_message ?? 'Deployment failed.'}
          </InfoCard>,
        )
      } else {
        // Still in progress — hand off to status polling
        setHereDeploymentId(dep.id)
      }
      // Poll effect drives phase → done or error (for non-terminal returns)
    } catch (err) {
      if (err instanceof DeploymentActionError) {
        if (err.error_kind === 'redeploy_requires_confirmation') {
          // Store the pending row ID so confirm can re-fire
          setRedeployRowId(pendingRow.id)
        }
        handleDeployHereError(err)
      } else {
        setPhase('error')
        setHereError(
          <InfoCard color="lava">
            {err instanceof Error ? err.message : 'Deploy failed unexpectedly.'}
          </InfoCard>,
        )
      }
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [agentId, appName, gitTag, revisionId, target])

  const handleDeployHereError = React.useCallback(
    (err: DeploymentActionError) => {
      if (err.error_kind === 'redeploy_requires_confirmation') {
        setRedeployModalOpen(true)
        setPhase('idle')
        return
      }
      // All other error kinds: show error phase + route through DeployHereErrorCard
      setPhase('error')
      const externalResourceIds = err.externalResourceIds ?? null
      setHereError(
        <DeployHereErrorCard
          errorKind={err.error_kind as DeployHereErrorKind}
          externalResourceIds={externalResourceIds}
          appName={appName}
          onAction={(action) => {
            if (action === 'switch_to_export') {
              setMode('other')
              setHereError(null)
              setPhase('idle')
            } else if (action === 'redeploy_confirmed') {
              setRedeployModalOpen(true)
              setPhase('idle')
            } else if (action === 'retry') {
              setHereError(null)
              setPhase('idle')
            }
          }}
          onSuggestedName={(name) => {
            setAppName(name)
            setPhase('idle')
            setHereError(null)
          }}
        />,
      )
    },
    [appName],
  )

  // ---- confirm redeploy ----
  const handleConfirmRedeploy = React.useCallback(async () => {
    setRedeployModalOpen(false)
    if (!redeployRowId) return
    setPhase('running')
    setStepIdx(0)
    setHereError(null)
    try {
      const dep = await deployHereAction(redeployRowId, { confirmRedeploy: true })
      setHereDeploymentId(dep.id)
    } catch (err) {
      setPhase('error')
      setHereError(
        <InfoCard color="lava">
          {err instanceof Error ? err.message : 'Redeploy failed.'}
        </InfoCard>,
      )
    }
  }, [redeployRowId])

  // ---- derived subtitle ----
  const subtitle =
    mode === 'here'
      ? `Automatically deploys "${agentName}" into this workspace using your OAuth token — no zip download, no manual steps.`
      : `Generates a self-contained zip of "${agentName}" as a Databricks App with a built-in chat UI. Extract and run databricks bundle deploy to ship it.`

  // ---- footer ----
  const footer = React.useMemo(() => {
    if (mode === 'here') {
      if (phase === 'idle' || phase === 'error') {
        return (
          <>
            <Button variant="outline" onClick={() => onOpenChange(false)}>
              Cancel
            </Button>
            <Button
              data-testid="shell-app-deploy-here"
              onClick={() => { void handleDeployHere() }}
              className="ml-auto"
            >
              Deploy now
            </Button>
          </>
        )
      }
      if (phase === 'running') {
        return (
          <>
            <Button variant="outline" onClick={() => onOpenChange(false)}>
              Cancel
            </Button>
            <Button disabled className="ml-auto">
              Deploying…
            </Button>
          </>
        )
      }
      // phase === 'done'
      return (
        <>
          <Button variant="outline" onClick={() => onOpenChange(false)}>
            Done
          </Button>
          {appUrl !== null && (
            <a
              href={appUrl}
              target="_blank"
              rel="noopener noreferrer"
              className="ml-auto inline-flex items-center justify-center whitespace-nowrap rounded-md bg-primary px-4 py-2 text-sm font-medium text-primary-foreground shadow transition-colors hover:bg-primary/90 focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring"
            >
              Open app
            </a>
          )}
        </>
      )
    }

    // mode === 'other'
    if (created !== null) {
      return (
        <>
          <Button
            data-testid="shell-app-download-button"
            onClick={() => { void handleDownload() }}
            className="ml-auto"
          >
            Download zip
          </Button>
          <Button variant="outline" onClick={() => onOpenChange(false)}>
            Done
          </Button>
        </>
      )
    }
    return (
      <>
        <Button
          variant="outline"
          onClick={() => onOpenChange(false)}
          disabled={otherPending}
        >
          Cancel
        </Button>
        <Button
          data-testid="shell-app-wizard-submit"
          onClick={() => { void handleSubmit() }}
          disabled={submitDisabled}
          className="ml-auto"
        >
          {otherPending ? 'Generating…' : 'Generate zip'}
        </Button>
      </>
    )
  }, [
    mode,
    phase,
    appUrl,
    created,
    otherPending,
    submitDisabled,
    onOpenChange,
    handleDeployHere,
    handleDownload,
    handleSubmit,
  ])

  return (
    <DialogShell
      open={open}
      onOpenChange={onOpenChange}
      icon={Box}
      iconBg="var(--db-lava-100)"
      iconColor="var(--db-lava-600)"
      title="Deploy as Databricks App"
      subtitle={subtitle}
      width={720}
      footer={footer}
    >
      <RevisionProvenanceCard provenance={revisionProvenance} />

      {/* SP-fallback hint — shown above tabs when running as service principal */}
      {isSpFallback && (
        <InfoCard color="blue">
          Local-dev mode: running as the app&apos;s service principal, not your
          user account.
        </InfoCard>
      )}

      {probeUnknown && (
        <InfoCard color="yellow">
          Permission check is unavailable. You can still try deploying; the
          deploy step will report the Databricks Apps error if it fails.
        </InfoCard>
      )}

      {refreshCanDeployHere.isError && (
        <InfoCard color="lava">
          Could not re-check permissions: {formatRefreshError(refreshCanDeployHere.error)}
        </InfoCard>
      )}

      {/* Mode tabs — always visible */}
      <ModeTabs
        value={mode}
        onChange={setMode}
        disabledTabs={hereDisabled ? ['here'] : []}
        disabledTooltips={{
          here: hereDisabledReason === 'missing_workspace_permission'
            ? 'Requires CAN_MANAGE_APP in this workspace.'
            : hereDisabledReason === 'missing_obo_token'
              ? 'Authentication missing — refresh the page.'
              : isProbing
                ? 'Could not check permissions — try again later.'
                : 'In-workspace deploy isn\'t available.',
        }}
      />

      {/* Re-check permissions button */}
      <div style={{ display: 'flex', justifyContent: 'flex-end', marginTop: 4 }}>
        <Button
          variant="outline"
          size="sm"
          type="button"
          onClick={() => { void refreshCanDeployHere.mutate() }}
          disabled={refreshCanDeployHere.isPending}
          data-testid="recheck-permissions-button"
        >
          {refreshCanDeployHere.isPending ? 'Checking…' : 'Re-check permissions'}
        </Button>
      </div>

      {/* Configure form — always visible while not yet running */}
      {(mode === 'other' || phase === 'idle') && (
        <div className="mt-4 grid grid-cols-3 gap-2.5">
          <div>
            <label
              className="mb-1 block text-xs font-medium"
              style={{ color: 'var(--db-navy-800)' }}
            >
              App name
              <input
                data-testid="shell-app-name-input"
                value={appName}
                onChange={(e) => setAppName(e.target.value)}
                className="mt-1 w-full rounded-md border border-zinc-300 bg-white px-2 py-1 font-mono text-xs text-zinc-900 focus:border-transparent focus:outline-none focus:ring-2 focus:ring-navy-400"
                placeholder="dr-shell-research"
                maxLength={APP_NAME_MAX_LENGTH}
                disabled={!!created}
              />
              {!appNameValid && appName.length > 0 ? (
                <span
                  role="alert"
                  className="mt-1 block text-[11px] text-red-700"
                >
                  Must match <code>^dr-shell-[a-z0-9-]+$</code> and be 30 chars or fewer.
                </span>
              ) : null}
            </label>
          </div>

          <div>
            <label
              className="mb-1 block text-xs font-medium"
              style={{ color: 'var(--db-navy-800)' }}
            >
              Framework Git ref
              <input
                data-testid="shell-app-git-tag-input"
                value={gitTag}
                onChange={handleGitTagChange}
                className="mt-1 w-full rounded-md border border-zinc-300 bg-white px-2 py-1 font-mono text-xs text-zinc-900 focus:border-transparent focus:outline-none focus:ring-2 focus:ring-navy-400"
                placeholder={defaultGitTag || 'v0.0.0'}
                disabled={!!created}
              />
              <span className="mt-1 block text-[11px] text-zinc-500">
                Pinned in <code>pyproject.toml</code>.
              </span>
            </label>
          </div>

          <div>
            <label
              className="mb-1 block text-xs font-medium"
              style={{ color: 'var(--db-navy-800)' }}
            >
              DAB target
              <input
                data-testid="shell-app-target-input"
                value={target}
                onChange={(e) => setTarget(e.target.value)}
                className="mt-1 w-full rounded-md border border-zinc-300 bg-white px-2 py-1 font-mono text-xs text-zinc-900 focus:border-transparent focus:outline-none focus:ring-2 focus:ring-navy-400"
                disabled={!!created}
              />
              <span className="mt-1 block text-[11px] text-db-gray-text">
                Target name from databricks.yml (e.g. dev, staging, prod)
              </span>
            </label>
          </div>
        </div>
      )}

      {/* The target input must always be present in DOM for tests when mode='here' */}
      {mode === 'here' && phase !== 'idle' && (
        <input
          data-testid="shell-app-target-input"
          value={target}
          onChange={(e) => setTarget(e.target.value)}
          className="sr-only"
          aria-hidden="true"
          tabIndex={-1}
        />
      )}

      {otherError ? (
        <p
          role="alert"
          className="mt-3 rounded-md border border-red-200 bg-red-50 px-2 py-1 text-xs text-red-700"
        >
          {otherError}
        </p>
      ) : null}

      {/* ------------------------------------------------------------------ */}
      {/* HERE mode body                                                       */}
      {/* ------------------------------------------------------------------ */}
      {mode === 'here' && (
        <div className="mt-4">
          {/* Inline error card (permission errors, etc.) */}
          {hereError}

          {phase === 'idle' && !hereError && (
            <>
              <SectionTitle style={{ marginTop: 14 }}>
                What happens when you click Deploy
              </SectionTitle>
              <ProgressList steps={DEPLOY_STEPS} currentIdx={-1} />
            </>
          )}

          {(phase === 'running' || phase === 'done' || phase === 'error') && !hereError && (
            <>
              <SectionTitle style={{ marginTop: 14 }}>
                {phase === 'done'
                  ? 'Deployment complete'
                  : phase === 'error'
                    ? 'Deployment failed'
                    : 'Deploying…'}
              </SectionTitle>
              <ProgressList steps={DEPLOY_STEPS} currentIdx={stepIdx} />

              {phase === 'done' && appUrl !== null && (
                <InfoCard color="green">
                  <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
                    <CheckCircle
                      size={15}
                      color="var(--db-green-700, #00875C)"
                      strokeWidth={2.5}
                      style={{ flexShrink: 0 }}
                    />
                    <span>
                      Deployed at{' '}
                      <a
                        href={appUrl}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="underline font-mono text-[11px]"
                      >
                        {appUrl}
                      </a>
                    </span>
                  </div>
                </InfoCard>
              )}

              {phase === 'error' && pollData?.error_message && (
                <InfoCard color="lava">
                  {pollData.error_message}
                </InfoCard>
              )}
            </>
          )}
        </div>
      )}

      {/* Redeploy confirmation modal */}
      <RadixDialog.Root open={redeployModalOpen} onOpenChange={setRedeployModalOpen}>
        <RadixDialog.Portal>
          <RadixDialog.Overlay
            style={{
              position: 'fixed',
              inset: 0,
              background: 'rgba(0,0,0,0.4)',
              zIndex: 9998,
            }}
          />
          <RadixDialog.Content
            style={{
              position: 'fixed',
              top: '50%',
              left: '50%',
              transform: 'translate(-50%,-50%)',
              zIndex: 9999,
              background: '#fff',
              borderRadius: 10,
              padding: '24px 28px',
              width: 400,
              boxShadow: '0 8px 32px rgba(0,0,0,0.18)',
            }}
          >
            <RadixDialog.Title
              style={{ fontSize: 15, fontWeight: 600, marginBottom: 8 }}
            >
              Replace running app?
            </RadixDialog.Title>
            <RadixDialog.Description
              style={{ fontSize: 13, color: 'var(--db-gray-text)', marginBottom: 20 }}
            >
              This will replace the running app{' '}
              <strong className="font-mono">{appName}</strong>. Continue?
            </RadixDialog.Description>
            <div style={{ display: 'flex', gap: 8, justifyContent: 'flex-end' }}>
              <Button
                variant="outline"
                onClick={() => {
                  setRedeployModalOpen(false)
                  setPhase('idle')
                }}
              >
                Cancel
              </Button>
              <Button
                data-testid="redeploy-confirm-button"
                onClick={() => { void handleConfirmRedeploy() }}
              >
                Replace app
              </Button>
            </div>
          </RadixDialog.Content>
        </RadixDialog.Portal>
      </RadixDialog.Root>

      {/* ------------------------------------------------------------------ */}
      {/* OTHER mode body                                                      */}
      {/* ------------------------------------------------------------------ */}
      {mode === 'other' && (
        <div className="mt-4">
          <SectionTitle>Files in this bundle</SectionTitle>
          <FileTree files={BUNDLE_FILES} />

          <SectionTitle style={{ marginTop: 18 }}>
            Target workspace
          </SectionTitle>
          <HostField
            value={host}
            onChange={setHost}
            hint="The host of the workspace you're shipping to. Used in the commands below."
          />

          <SectionTitle style={{ marginTop: 18 }}>Deploy steps</SectionTitle>
          <ol style={{ listStyle: 'none', padding: 0, margin: '8px 0 0' }}>
            <Step
              n={1}
              title="Generate and download the bundle"
              body={
                created !== null
                  ? 'Bundle ready — click Download zip to save it.'
                  : 'We render the 8-file zip with your agent embedded. Click "Generate zip" in the footer to create it.'
              }
              note="The bundle pins the framework via Git ref — it is reproducible across workspaces."
            />
            {/* Download action lives below Step 1 once the bundle is created */}
            {created !== null && (
              <li
                style={{
                  padding: '6px 0 12px 34px',
                  borderBottom: '1px dashed var(--db-gray-lines)',
                }}
              >
                <Button
                  data-testid="shell-app-download-button-inline"
                  onClick={() => { void handleDownload() }}
                  size="sm"
                >
                  Download zip
                </Button>
              </li>
            )}

            <Step
              n={2}
              title="Extract and inspect"
              code={`unzip ${appName}.zip -d ${appName} && cd ${appName}`}
              codeLang="bash"
              codeLabel="extract"
            />

            <Step
              n={3}
              title="Authenticate against the target workspace"
              code={`databricks auth login --host https://your-workspace.cloud.databricks.com`}
              codeLang="bash"
              codeLabel="auth"
            />

            <Step
              n={4}
              title="Deploy the bundle"
              code={`databricks bundle deploy --target ${target}`}
              codeLang="bash"
              codeLabel="deploy"
            />

            <Step
              n={5}
              title="Start the app"
              code={`databricks bundle run ${appName} --target ${target}`}
              codeLang="bash"
              codeLabel="run"
            />
          </ol>
        </div>
      )}
    </DialogShell>
  )
}
