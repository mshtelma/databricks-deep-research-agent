/**
 * MlflowAgentWizard — Mode 3 (Mosaic AI Agent Framework deployment).
 *
 * Dual-mode dialog:
 *  - 'here'  → auto-deploy via MLflow + Mosaic AI in the current workspace
 *  - 'other' → offline register + deploy in another workspace
 *
 * Submitting POSTs /api/v1/deployments with mode='mlflow_agent' so
 * `MlflowAgentTranslator` runs:
 *   mlflow.pyfunc.log_model → mlflow.register_model('databricks-uc') →
 *   databricks.agents.deploy(uc_name, version)
 *
 * Plan reference: we-don-t-need-legacy-composed-wren.md Section C3.
 */

import * as React from 'react'
import { Link } from 'lucide-react'

import { Button } from '@/components/ui/button'
import { formatDefaultRevisionNotDeployableError } from '@/api/deployments'
import { useCreateDeployment, useDeploymentStatusPoll } from '@/hooks/useDeployments'
import type { DeploymentResponse } from '@/types/deployment'

import {
  DialogShell,
  ModeTabs,
  ProgressList,
  Step,
  HostField,
  InfoCard,
} from './dialog-primitives'
import type { ProgressStep } from './dialog-primitives'
import { RevisionProvenanceCard } from './RevisionProvenanceCard'
import type { RevisionProvenance } from './revisionProvenance'

interface MlflowAgentWizardProps {
  open: boolean
  onOpenChange: (open: boolean) => void
  agentId: string
  agentName: string
  revisionId: string
  revisionProvenance?: RevisionProvenance | null
  onDeployed: (deployment: DeploymentResponse) => void
}

const UC_IDENT = /^[A-Za-z_][A-Za-z0-9_-]*$/
const ENDPOINT_NAME = /^dr-agent-[a-z0-9-]+$/

type Phase = 'idle' | 'running' | 'done'

function defaultModelName(agentId: string): string {
  const safe = agentId.slice(0, 8).replace(/[^A-Za-z0-9_-]/g, '_')
  return `dr_${safe}`
}

function slugFromName(name: string): string {
  return name
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-|-$/g, '')
    .slice(0, 30)
}

const SUBTITLES: Record<'here' | 'other', string> = {
  here: 'Auto-deploy via MLflow + Mosaic AI — runs in the current workspace.',
  other: 'Generate an offline package to register and deploy in another workspace.',
}

function formatDeploymentError(error: unknown): string {
  return (
    formatDefaultRevisionNotDeployableError(error) ??
    (error instanceof Error ? error.message : 'Deployment failed.')
  )
}

export function MlflowAgentWizard({
  open,
  onOpenChange,
  agentId,
  agentName,
  revisionId,
  revisionProvenance = null,
  onDeployed,
}: MlflowAgentWizardProps): React.ReactElement {
  // ── Form state ──────────────────────────────────────────────────────────
  const [ucCatalog, setUcCatalog] = React.useState('main')
  const [ucSchema, setUcSchema] = React.useState('agents')
  const [ucModelName, setUcModelName] = React.useState(() =>
    defaultModelName(agentId),
  )
  const [endpointName, setEndpointName] = React.useState('')
  const [envOverridesJson, setEnvOverridesJson] = React.useState('')

  // ── New mode / phase state ───────────────────────────────────────────────
  const [mode, setMode] = React.useState<'here' | 'other'>('here')
  const [phase, setPhase] = React.useState<Phase>('idle')
  const [stepIdx, setStepIdx] = React.useState(-1)
  const [targetHost, setTargetHost] = React.useState('')
  const [deploymentId, setDeploymentId] = React.useState<string | null>(null)
  const [endpointUrl, setEndpointUrl] = React.useState<string | null>(null)

  // Wall-clock tick timer for advancing stepIdx during 'deploying' status
  const tickTimer = React.useRef<ReturnType<typeof setTimeout> | null>(null)

  const mutation = useCreateDeployment()

  // Status poll — active while phase === 'running'
  const { data: pollData } = useDeploymentStatusPoll(deploymentId, {
    enabled: phase === 'running',
  })

  React.useEffect(() => {
    if (open) {
      setUcModelName(defaultModelName(agentId))
      setPhase('idle')
      setStepIdx(-1)
      setDeploymentId(null)
      setEndpointUrl(null)
      if (tickTimer.current !== null) {
        clearTimeout(tickTimer.current)
        tickTimer.current = null
      }
      mutation.reset()
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [open, agentId])

  React.useEffect(() => {
    return () => {
      if (tickTimer.current !== null) clearTimeout(tickTimer.current)
    }
  }, [])

  // ── Status-poll → stepIdx/phase mapping ─────────────────────────────────
  React.useEffect(() => {
    if (!pollData || phase !== 'running') return
    const status = pollData.status
    if (status === 'pending') {
      setStepIdx(0)
    } else if (status === 'deploying') {
      setStepIdx((prev) => Math.max(prev, 1))
      if (tickTimer.current === null) {
        let i = 1
        const maxSteps = 3 // advance through steps 1, 2, 3 over 20s
        const intervalMs = 20_000 / maxSteps
        const tick = () => {
          i += 1
          if (i <= maxSteps) {
            setStepIdx(i)
            tickTimer.current = setTimeout(tick, intervalMs)
          }
        }
        tickTimer.current = setTimeout(tick, intervalMs)
      }
    } else if (status === 'active') {
      if (tickTimer.current !== null) {
        clearTimeout(tickTimer.current)
        tickTimer.current = null
      }
      setStepIdx(5) // all 5 steps complete
      setPhase('done')
      const extIds = (pollData as unknown as { external_resource_ids?: Record<string, string> })
        .external_resource_ids
      setEndpointUrl(extIds?.endpoint_url ?? null)
    } else if (status === 'failed') {
      if (tickTimer.current !== null) {
        clearTimeout(tickTimer.current)
        tickTimer.current = null
      }
      setPhase('idle')
      // Error surfaces via mutation.isError
    }
  }, [pollData, phase])

  // ── Validation ───────────────────────────────────────────────────────────
  const ucCatalogValid = UC_IDENT.test(ucCatalog)
  const ucSchemaValid = UC_IDENT.test(ucSchema)
  const ucModelNameValid = UC_IDENT.test(ucModelName)
  const endpointNameValid =
    endpointName === '' || ENDPOINT_NAME.test(endpointName)

  let envOverrides: Record<string, string> | undefined
  let envOverridesError: string | null = null
  if (envOverridesJson.trim()) {
    try {
      const parsed = JSON.parse(envOverridesJson) as unknown
      if (
        typeof parsed !== 'object' ||
        parsed === null ||
        Array.isArray(parsed)
      ) {
        envOverridesError = 'env_overrides must be a JSON object'
      } else {
        const all = Object.values(parsed as Record<string, unknown>).every(
          (v) => typeof v === 'string',
        )
        if (!all) {
          envOverridesError = 'env_overrides values must all be strings'
        } else {
          envOverrides = parsed as Record<string, string>
        }
      }
    } catch (err) {
      envOverridesError = (err as Error).message
    }
  }

  const formValid =
    ucCatalogValid &&
    ucSchemaValid &&
    ucModelNameValid &&
    endpointNameValid &&
    !envOverridesError

  const submitDisabled = !formValid || mutation.isPending

  // ── Derived values ───────────────────────────────────────────────────────
  const effectiveEndpointName =
    endpointName || `dr-agent-${slugFromName(agentName) || agentId.slice(0, 12)}`

  const mockEndpointUrl = `https://${targetHost || '<workspace-host>'}/serving-endpoints/${effectiveEndpointName}/invocations`

  // ── Progress steps for 'here' mode ──────────────────────────────────────
  const progressSteps: ProgressStep[] = [
    {
      id: 'log',
      label: 'Log model via mlflow.pyfunc.log_model',
      detail: 'ResponsesAgent wrapper + workflow_definition artifact',
    },
    {
      id: 'register',
      label: 'Register in Unity Catalog',
      detail: `${ucCatalog}.${ucSchema}.${ucModelName}`,
    },
    {
      id: 'deploy',
      label: 'Deploy via databricks.agents.deploy',
      detail: `endpoint: ${effectiveEndpointName}`,
    },
    {
      id: 'scaffold',
      label: 'Wire scaffolding (review app, eval review UI)',
      detail: 'may take ~3-8 min on cold compute',
    },
    {
      id: 'health',
      label: 'Health check on /predict',
      detail: 'OpenAI-compatible · HTTP 200',
    },
  ]

  // ── Submit handler (here mode) ───────────────────────────────────────────
  const handleDeploy = React.useCallback(async () => {
    if (!formValid || mutation.isPending) return
    setPhase('running')
    setStepIdx(0)

    try {
      const deployment = await mutation.mutateAsync({
        agent_id: agentId,
        revision_id: revisionId,
        config: {
          mode: 'mlflow_agent',
          uc_catalog: ucCatalog,
          uc_schema: ucSchema,
          uc_model_name: ucModelName,
          ...(endpointName ? { endpoint_name: endpointName } : {}),
          ...(envOverrides ? { env_overrides: envOverrides } : {}),
        },
      })
      // Start polling — the poll effect will drive stepIdx and phase
      setDeploymentId(deployment.id)
      onDeployed(deployment)
    } catch {
      setPhase('idle')
      setStepIdx(-1)
      // Error surfaces via mutation.isError
    }
  }, [
    agentId,
    endpointName,
    envOverrides,
    formValid,
    mutation,
    onDeployed,
    revisionId,
    ucCatalog,
    ucModelName,
    ucSchema,
  ])

  // ── Footer ───────────────────────────────────────────────────────────────
  const footer = (
    <div
      style={{
        display: 'flex',
        alignItems: 'center',
        gap: 8,
        marginLeft: 'auto',
      }}
    >
      {mode === 'here' && phase === 'idle' && (
        <>
          <Button variant="outline" onClick={() => onOpenChange(false)}>
            Cancel
          </Button>
          <Button
            data-testid="mlflow-agent-wizard-submit"
            onClick={() => {
              void handleDeploy()
            }}
            disabled={submitDisabled}
          >
            Deploy MLflow agent
          </Button>
        </>
      )}
      {mode === 'here' && phase === 'running' && (
        <>
          <Button variant="outline" onClick={() => onOpenChange(false)}>
            Cancel
          </Button>
          <Button
            data-testid="mlflow-agent-wizard-submit"
            disabled
          >
            Deploying…
          </Button>
        </>
      )}
      {mode === 'here' && phase === 'done' && (
        <>
          <a
            href={mockEndpointUrl}
            target="_blank"
            rel="noreferrer"
            className="inline-flex items-center rounded-md border border-zinc-300 bg-white px-3 py-1.5 text-xs font-medium text-zinc-700 hover:bg-zinc-50"
          >
            View endpoint
          </a>
          <Button onClick={() => onOpenChange(false)}>Done</Button>
        </>
      )}
      {mode === 'other' && (
        <>
          <Button variant="outline" onClick={() => onOpenChange(false)}>
            Cancel
          </Button>
          <Button
            onClick={() => {
              // TODO: tarball generation is a follow-up backend turn
            }}
          >
            Generate offline package
          </Button>
        </>
      )}
    </div>
  )

  return (
    <DialogShell
      open={open}
      onOpenChange={onOpenChange}
      icon={Link}
      iconBg="var(--db-oat-medium)"
      iconColor="var(--db-navy-800)"
      title="Deploy as API Endpoint (MLflow agent)"
      subtitle={SUBTITLES[mode]}
      width={720}
      footer={footer}
    >
      <RevisionProvenanceCard provenance={revisionProvenance} />

      {/* Mode tabs */}
      <ModeTabs value={mode} onChange={setMode} />

      {/* ── Configure form (always visible above mode body) ── */}
      <fieldset
        style={{ border: 0, padding: 0, margin: '16px 0 0' }}
        disabled={phase === 'running'}
      >
        {/* 3-column UC fields */}
        <div className="grid grid-cols-3 gap-3">
          <label className="block text-xs font-medium text-zinc-700">
            UC catalog
            <input
              data-testid="mlflow-uc-catalog-input"
              value={ucCatalog}
              onChange={(e) => setUcCatalog(e.target.value)}
              className="mt-1 w-full rounded-md border border-zinc-300 px-2 py-1 font-mono text-xs"
            />
            {!ucCatalogValid ? (
              <span role="alert" className="mt-1 block text-[11px] text-red-700">
                Invalid UC identifier
              </span>
            ) : null}
          </label>
          <label className="block text-xs font-medium text-zinc-700">
            UC schema
            <input
              data-testid="mlflow-uc-schema-input"
              value={ucSchema}
              onChange={(e) => setUcSchema(e.target.value)}
              className="mt-1 w-full rounded-md border border-zinc-300 px-2 py-1 font-mono text-xs"
            />
            {!ucSchemaValid ? (
              <span role="alert" className="mt-1 block text-[11px] text-red-700">
                Invalid UC identifier
              </span>
            ) : null}
          </label>
          <label className="block text-xs font-medium text-zinc-700">
            UC model name
            <input
              data-testid="mlflow-uc-model-name-input"
              value={ucModelName}
              onChange={(e) => setUcModelName(e.target.value)}
              className="mt-1 w-full rounded-md border border-zinc-300 px-2 py-1 font-mono text-xs"
            />
            {!ucModelNameValid ? (
              <span role="alert" className="mt-1 block text-[11px] text-red-700">
                Invalid UC identifier
              </span>
            ) : null}
          </label>
        </div>

        <div style={{ marginTop: 12 }}>
          <label className="block text-xs font-medium text-zinc-700">
            Endpoint name (optional override)
            <input
              data-testid="mlflow-endpoint-name-input"
              value={endpointName}
              onChange={(e) => setEndpointName(e.target.value)}
              className="mt-1 w-full rounded-md border border-zinc-300 px-2 py-1 font-mono text-xs"
              placeholder="dr-agent-research (auto-generated when blank)"
            />
            {!endpointNameValid ? (
              <span role="alert" className="mt-1 block text-[11px] text-red-700">
                Must match <code>^dr-agent-[a-z0-9-]+$</code>
              </span>
            ) : null}
          </label>
        </div>

        <div style={{ marginTop: 12 }}>
          <label className="block text-xs font-medium text-zinc-700">
            Optional env_overrides (JSON object: string → string)
            <textarea
              data-testid="mlflow-env-overrides-input"
              value={envOverridesJson}
              onChange={(e) => setEnvOverridesJson(e.target.value)}
              rows={3}
              className="mt-1 w-full rounded-md border border-zinc-300 px-2 py-1 font-mono text-xs"
              placeholder='{"BRAVE_API_KEY_REF": "scope/key"}'
            />
            {envOverridesError ? (
              <span role="alert" className="mt-1 block text-[11px] text-red-700">
                {envOverridesError}
              </span>
            ) : null}
          </label>
        </div>
      </fieldset>

      {/* ── Mutation error ── */}
      {mutation.isError ? (
        <p
          role="alert"
          className="mt-3 rounded-md border border-red-200 bg-red-50 px-2 py-1 text-xs text-red-700"
        >
          {formatDeploymentError(mutation.error)}
        </p>
      ) : null}

      {/* ── Mode body ── */}
      <div style={{ marginTop: 20 }}>
        {mode === 'here' && (
          <>
            <InfoCard color="green">
              MLflow live deploy is the working backend path — runs as the
              app's service principal in the current workspace.
            </InfoCard>

            <div style={{ marginTop: 16 }}>
              <ProgressList
                steps={progressSteps}
                currentIdx={stepIdx}
                error={mutation.isError}
              />
            </div>

            {phase === 'done' && (
              <div
                style={{
                  marginTop: 14,
                  padding: '10px 12px',
                  borderRadius: 6,
                  background: 'var(--db-green-300)',
                  border: '1px solid var(--db-green-700)',
                  fontSize: 12,
                  color: 'var(--db-navy-800)',
                  lineHeight: 1.6,
                }}
              >
                <strong>Deployment active.</strong> Endpoint URL:{' '}
                <code className="font-mono text-xs">
                  {endpointUrl ?? mockEndpointUrl}
                </code>
              </div>
            )}
          </>
        )}

        {mode === 'other' && (
          <>
            <HostField
              value={targetHost}
              onChange={setTargetHost}
              hint="The MLflow tracking server and UC must both be in this workspace."
            />

            <ol style={{ listStyle: 'none', padding: 0, margin: '8px 0 0' }}>
              <Step
                n={1}
                title="Download the offline package"
                body="A tarball with workflow_definition.json + register.py + requirements.txt."
              />
              <Step
                n={2}
                title="Install dependencies"
                code="pip install mlflow databricks-agents databricks-deep-research"
                codeLang="bash"
                codeLabel="pip"
              />
              <Step
                n={3}
                title="Authenticate"
                code={`databricks auth login --host ${targetHost || '<TARGET_HOST>'}\nexport MLFLOW_TRACKING_URI=databricks\nexport MLFLOW_REGISTRY_URI=databricks-uc`}
                codeLang="bash"
                codeLabel="auth"
              />
              <Step
                n={4}
                title="Run the registration helper"
                code={`python register.py --uc-catalog ${ucCatalog} --uc-schema ${ucSchema} --uc-model-name ${ucModelName}${endpointName ? ' --endpoint-name ' + endpointName : ''}`}
                codeLang="bash"
                codeLabel="register"
              />
              <Step
                n={5}
                title="Verify the endpoint"
                code={`databricks serving-endpoints get ${endpointName || '<endpoint-name>'}`}
                codeLang="bash"
                codeLabel="verify"
              />
            </ol>
          </>
        )}
      </div>
    </DialogShell>
  )
}
