/**
 * InAppWizard — dual-mode dialog for Mode 1 (in-app picker) deployment.
 *
 * 'here' mode: one-click deployment into the current workspace's chat picker.
 * 'other' mode: read-only curl/CLI instructions for registering in a remote workspace.
 *
 * Submitting in 'here' mode calls POST /api/v1/deployments with mode='in_app'.
 * On success, the ProgressList animates through both steps and onDeployed fires.
 */

import * as React from 'react'
import { Bot } from 'lucide-react'

import { Button } from '@/components/ui/button'
import { formatDefaultRevisionNotDeployableError } from '@/api/deployments'
import { useCreateDeployment } from '@/hooks/useDeployments'
import type { DeploymentResponse } from '@/types/deployment'
import {
  DialogShell,
  ModeTabs,
  ProgressList,
  Step,
  InfoCard,
  HostField,
} from './dialog-primitives'
import type { ProgressStep } from './dialog-primitives'
import { RevisionProvenanceCard } from './RevisionProvenanceCard'
import type { RevisionProvenance } from './revisionProvenance'

interface InAppWizardProps {
  open: boolean
  onOpenChange: (open: boolean) => void
  agentId: string
  agentName: string
  revisionId: string
  revisionProvenance?: RevisionProvenance | null
  /** Fired after the deployment row is created (status will be ACTIVE). */
  onDeployed: (deployment: DeploymentResponse) => void
}

type Mode = 'here' | 'other'
type Phase = 'idle' | 'running' | 'done'

const PROGRESS_STEPS: ProgressStep[] = [
  {
    id: 's1',
    label: 'Save revision',
    detail: 'POST /api/v1/agents/{id}/revisions',
  },
  {
    id: 's2',
    label: 'Register in agent_deployments',
    detail: 'POST /api/v1/deployments (mode=in_app)',
  },
]

function formatDeploymentError(error: unknown): string {
  return (
    formatDefaultRevisionNotDeployableError(error) ??
    (error instanceof Error ? error.message : 'Deployment failed.')
  )
}

export function InAppWizard({
  open,
  onOpenChange,
  agentId,
  agentName,
  revisionId,
  revisionProvenance = null,
  onDeployed,
}: InAppWizardProps): React.ReactElement {
  const mutation = useCreateDeployment()

  const [mode, setMode] = React.useState<Mode>('here')
  const [phase, setPhase] = React.useState<Phase>('idle')
  const [stepIdx, setStepIdx] = React.useState<number>(-1)
  const [targetHost, setTargetHost] = React.useState<string>('')

  // Reset state when dialog opens/closes
  React.useEffect(() => {
    if (!open) {
      setMode('here')
      setPhase('idle')
      setStepIdx(-1)
      setTargetHost('')
    }
  }, [open])

  const handleSubmit = React.useCallback(async () => {
    setPhase('running')
    setStepIdx(0)

    // Animate step 0 briefly, then fire the real mutation
    const stepTimer = setTimeout(() => {
      setStepIdx(1)
    }, 600)

    try {
      const created = await mutation.mutateAsync({
        agent_id: agentId,
        revision_id: revisionId,
        config: { mode: 'in_app' },
      })
      clearTimeout(stepTimer)
      setStepIdx(PROGRESS_STEPS.length)
      setPhase('done')
      onDeployed(created)
      onOpenChange(false)
    } catch {
      clearTimeout(stepTimer)
      setPhase('idle')
      setStepIdx(-1)
      // Error surfaces via mutation.isError / mutation.error below
    }
  }, [agentId, mutation, onDeployed, onOpenChange, revisionId])

  const subtitle =
    mode === 'here'
      ? `Registers "${agentName}" as a picker entry in this workspace's chat composer. No separate compute is provisioned.`
      : `Follow these steps to register "${agentName}" in a different workspace running the deep-research app.`

  const host = targetHost || '<TARGET_HOST>'

  const footer =
    mode === 'here' ? (
      phase === 'idle' ? (
        <div style={{ display: 'flex', gap: 8, marginLeft: 'auto' }}>
          <Button variant="outline" onClick={() => onOpenChange(false)}>
            Cancel
          </Button>
          <Button
            data-testid="in-app-wizard-submit"
            onClick={() => {
              void handleSubmit()
            }}
            disabled={mutation.isPending}
          >
            Deploy now
          </Button>
        </div>
      ) : phase === 'running' ? (
        <div style={{ display: 'flex', gap: 8, marginLeft: 'auto' }}>
          <Button variant="outline" onClick={() => onOpenChange(false)}>
            Cancel
          </Button>
          <Button data-testid="in-app-wizard-submit" disabled>
            Deploying…
          </Button>
        </div>
      ) : (
        <div style={{ marginLeft: 'auto' }}>
          <Button onClick={() => onOpenChange(false)}>Done</Button>
        </div>
      )
    ) : (
      <div style={{ marginLeft: 'auto' }}>
        <Button variant="outline" onClick={() => onOpenChange(false)}>
          Close
        </Button>
      </div>
    )

  return (
    <DialogShell
      open={open}
      onOpenChange={onOpenChange}
      icon={Bot}
      iconBg="var(--db-blue-100)"
      iconColor="var(--db-blue-700)"
      title="Deploy in-app"
      subtitle={subtitle}
      width={680}
      footer={footer}
    >
      <RevisionProvenanceCard provenance={revisionProvenance} />

      <ModeTabs value={mode} onChange={setMode} />

      {mode === 'here' && (
        <div style={{ marginTop: 16 }}>
          <InfoCard color="blue">
            This registers the agent in your workspace&apos;s chat composer.
            Users with workspace access will see it in the agent picker.
          </InfoCard>

          <div style={{ marginTop: 16 }}>
            <ProgressList
              steps={PROGRESS_STEPS}
              currentIdx={stepIdx}
              error={mutation.isError}
            />
          </div>

          {phase === 'done' && (
            <InfoCard color="green">
              ✓ Available in the chat picker
            </InfoCard>
          )}

          {mutation.isError && (
            <p
              role="alert"
              style={{
                marginTop: 8,
                padding: '6px 10px',
                borderRadius: 6,
                background: 'var(--db-lava-100)',
                border: '1px solid var(--db-lava-300)',
                fontSize: 12,
                color: 'var(--db-lava-700)',
              }}
            >
              {formatDeploymentError(mutation.error)}
            </p>
          )}
        </div>
      )}

      {mode === 'other' && (
        <div style={{ marginTop: 16 }}>
          <InfoCard color="yellow">
            The target workspace must be running an instance of this deep-research app.
          </InfoCard>

          <HostField
            value={targetHost}
            onChange={setTargetHost}
            label="Target workspace host"
            hint="Where the destination Databricks app lives."
          />

          <ol style={{ listStyle: 'none', padding: 0, margin: '12px 0 0' }}>
            <Step
              n={1}
              title="Save the revision in this workspace"
              body="The target workspace will refer to this revision by ID."
            />
            <Step
              n={2}
              title="Get an auth token for the target workspace"
              code={`databricks auth login --host ${host}`}
              codeLang="bash"
              codeLabel="auth"
            />
            <Step
              n={3}
              title="POST the registration to the target's deep-research app"
              code={`curl -X POST ${host}/api/v1/deployments \\\n  -H "Authorization: Bearer $(databricks auth token --host ${host})" \\\n  -H "Content-Type: application/json" \\\n  -d '{"agent_id":"${agentId}","revision_id":"${revisionId}","config":{"mode":"in_app"}}'`}
              codeLang="bash"
              codeLabel="register"
            />
            <Step
              n={4}
              title="Verify the agent shows up"
              note="Open the target workspace's deep-research app and check the agent picker."
            />
          </ol>
        </div>
      )}
    </DialogShell>
  )
}
