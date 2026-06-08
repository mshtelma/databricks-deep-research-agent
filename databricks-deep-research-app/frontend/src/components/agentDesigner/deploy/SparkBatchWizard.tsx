/**
 * SparkBatchWizard — Mode 4 (Lakeflow Declarative Pipeline + ai_query).
 *
 * Dual-mode dialog:
 *   'here'  — preview-only auto-create path (animates 5 steps, no real backend call yet)
 *   'other' — download SQL + manual deploy steps (wires to existing useCreateDeployment)
 *
 * Plan reference: agent-designer-deployment.md Section C3 / F (Lakeflow + SQL).
 *
 * OBO limitation (plan Section F.4): Lakeflow pipelines do NOT support OBO --
 * they run as the pipeline owner / RUN_AS service principal. The wizard
 * surfaces a banner so the user understands the auth model before deploy.
 */

import * as React from 'react'
import { Database } from 'lucide-react'

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
  SectionTitle,
} from './dialog-primitives'
import type { ProgressStep } from './dialog-primitives'
import { RevisionProvenanceCard } from './RevisionProvenanceCard'
import type { RevisionProvenance } from './revisionProvenance'

interface SparkBatchWizardProps {
  open: boolean
  onOpenChange: (open: boolean) => void
  agentId: string
  agentName: string
  revisionId: string
  revisionProvenance?: RevisionProvenance | null
  onDeployed: (deployment: DeploymentResponse) => void
}

const THREE_LEVEL_NAME = /^[a-zA-Z_][\w-]*\.[a-zA-Z_][\w-]*\.[a-zA-Z_][\w-]*$/

type Mode = 'here' | 'other'
type Phase = 'idle' | 'running' | 'done'

function slugify(name: string): string {
  return name
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '_')
    .replace(/^_+|_+$/g, '')
}

const PROGRESS_STEPS: ProgressStep[] = [
  {
    id: 'b1',
    label: 'Build SQL artifact',
    detail: "ai_query() + pipelines.channel='preview'",
  },
  {
    id: 'b2',
    label: 'Upload SQL to workspace',
    detail: '/Workspace/Users/me/.lakeflow/{deployment_id}/batch.sql',
  },
  {
    id: 'b3',
    label: 'Create Lakeflow pipeline',
    detail: 'databricks pipelines create-pipeline',
  },
  {
    id: 'b4',
    label: 'Start initial update',
    detail: 'databricks pipelines start-update',
  },
  {
    id: 'b5',
    label: 'Watch first batch',
    detail: 'rows processed → output_table',
  },
]

function formatDeploymentError(error: unknown): string {
  return (
    formatDefaultRevisionNotDeployableError(error) ??
    (error instanceof Error ? error.message : 'Deployment failed.')
  )
}

export function SparkBatchWizard({
  open,
  onOpenChange,
  agentId,
  agentName,
  revisionId,
  revisionProvenance = null,
  onDeployed,
}: SparkBatchWizardProps): React.ReactElement {
  const [mode, setMode] = React.useState<Mode>('other')
  const [phase, setPhase] = React.useState<Phase>('idle')
  const [stepIdx, setStepIdx] = React.useState(0)
  const [targetHost, setTargetHost] = React.useState(
    'https://acme-prod.cloud.databricks.com',
  )

  // Form state
  const [targetEndpoint, setTargetEndpoint] = React.useState(
    'databricks-claude-sonnet-4-5',
  )
  const [inputTable, setInputTable] = React.useState('main.research.queries')
  const [outputTable, setOutputTable] = React.useState('main.research.results')
  const [promptColumn, setPromptColumn] = React.useState('query')
  const [responseFormatJson, setResponseFormatJson] = React.useState('')

  // 'other' mode post-submit download state
  const [created, setCreated] = React.useState<DeploymentResponse | null>(null)

  const mutation = useCreateDeployment()
  const timerRef = React.useRef<ReturnType<typeof setTimeout> | null>(null)

  // Reset on open/close
  React.useEffect(() => {
    if (open) {
      mutation.reset()
      setPhase('idle')
      setStepIdx(0)
      setCreated(null)
    } else {
      if (timerRef.current !== null) {
        clearTimeout(timerRef.current)
      }
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [open])

  // Cleanup timer on unmount
  React.useEffect(() => {
    return () => {
      if (timerRef.current !== null) {
        clearTimeout(timerRef.current)
      }
    }
  }, [])

  const inputTableValid = THREE_LEVEL_NAME.test(inputTable)
  const outputTableValid = THREE_LEVEL_NAME.test(outputTable)

  // Optional response_format must parse as JSON object if provided.
  let responseFormat: Record<string, unknown> | null = null
  let responseFormatError: string | null = null
  if (responseFormatJson.trim()) {
    try {
      const parsed = JSON.parse(responseFormatJson) as unknown
      if (typeof parsed !== 'object' || parsed === null) {
        responseFormatError = 'responseFormat must be a JSON object'
      } else {
        responseFormat = parsed as Record<string, unknown>
      }
    } catch (err) {
      responseFormatError = (err as Error).message
    }
  }

  const formValid =
    targetEndpoint.trim() !== '' &&
    inputTableValid &&
    outputTableValid &&
    promptColumn.trim() !== '' &&
    !responseFormatError

  const submitDisabled = !formValid || mutation.isPending

  // 'here' mode: simulate 5-step animation
  const startHereDeploy = React.useCallback(() => {
    setPhase('running')
    setStepIdx(0)
    let i = 0
    const tick = () => {
      i += 1
      if (i >= PROGRESS_STEPS.length) {
        setStepIdx(PROGRESS_STEPS.length)
        setPhase('done')
        return
      }
      setStepIdx(i)
      timerRef.current = setTimeout(tick, 1000 + Math.random() * 800)
    }
    timerRef.current = setTimeout(tick, 700)
  }, [])

  // 'other' mode: POST to backend then enable download
  const handleOtherSubmit = React.useCallback(async () => {
    if (submitDisabled) return
    try {
      const deployment = await mutation.mutateAsync({
        agent_id: agentId,
        revision_id: revisionId,
        config: {
          mode: 'batch',
          target_endpoint: targetEndpoint,
          input_table: inputTable,
          output_table: outputTable,
          prompt_column: promptColumn,
          response_format: responseFormat,
        },
      })
      setCreated(deployment)
      onDeployed(deployment)
      onOpenChange(false)
    } catch {
      // Error surfaces via mutation.isError below.
    }
  }, [
    agentId,
    inputTable,
    mutation,
    onDeployed,
    onOpenChange,
    outputTable,
    promptColumn,
    responseFormat,
    revisionId,
    submitDisabled,
    targetEndpoint,
  ])

  const handleDownload = React.useCallback(async () => {
    if (!created) return
    const resp = await fetch(`/api/v1/deployments/${created.id}/export-sql`)
    if (!resp.ok) return
    const blob = await resp.blob()
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `${slugify(agentName)}_batch.sql`
    document.body.appendChild(a)
    a.click()
    a.remove()
    URL.revokeObjectURL(url)
  }, [agentName, created])

  const slugAgent = slugify(agentName)

  // Subtitle varies by mode
  const subtitle =
    mode === 'here'
      ? 'Auto-creates a Lakeflow Declarative Pipeline in this workspace that calls ai_query() against the agent for every row of your input table. (Preview — backend wiring lands in next turn.)'
      : 'Generates the batch SQL artifact and provides CLI steps to deploy the pipeline manually in any target workspace.'

  // Footer content per mode/phase
  const footer: React.ReactNode =
    mode === 'here' ? (
      phase === 'idle' ? (
        <div className="flex w-full items-center gap-2">
          <Button
            variant="outline"
            onClick={() => onOpenChange(false)}
          >
            Cancel
          </Button>
          <Button
            className="ml-auto"
            onClick={startHereDeploy}
          >
            Deploy now
          </Button>
        </div>
      ) : phase === 'running' ? (
        <div className="flex w-full items-center gap-2">
          <Button variant="outline" onClick={() => onOpenChange(false)}>
            Cancel
          </Button>
          <Button className="ml-auto" disabled>
            Deploying…
          </Button>
        </div>
      ) : (
        <div className="flex w-full items-center gap-2">
          <Button className="ml-auto" onClick={() => onOpenChange(false)}>
            Done
          </Button>
        </div>
      )
    ) : created ? (
      <div className="flex w-full items-center gap-2">
        <Button
          onClick={() => {
            void handleDownload()
          }}
        >
          Download SQL
        </Button>
        <Button variant="outline" onClick={() => onOpenChange(false)}>
          Done
        </Button>
      </div>
    ) : (
      <div className="flex w-full items-center gap-2">
        <Button variant="outline" onClick={() => onOpenChange(false)} disabled={mutation.isPending}>
          Cancel
        </Button>
        <Button
          data-testid="spark-batch-wizard-submit"
          className="ml-auto"
          onClick={() => {
            void handleOtherSubmit()
          }}
          disabled={submitDisabled}
        >
          {mutation.isPending ? 'Generating SQL…' : 'Generate batch SQL'}
        </Button>
      </div>
    )

  return (
    <DialogShell
      open={open}
      onOpenChange={onOpenChange}
      icon={Database}
      iconBg="bg-db-green-300"
      iconColor="text-db-green-700"
      title="Spark Batch (Lakeflow)"
      subtitle={subtitle}
      width={780}
      footer={footer}
    >
      <RevisionProvenanceCard provenance={revisionProvenance} />

      {/* Persistent OBO banner — visible in both modes */}
      <div
        role="alert"
        data-testid="spark-batch-obo-banner"
      >
        <InfoCard color="yellow">
          <strong>OBO not supported.</strong> Lakeflow pipelines run as the
          pipeline owner / RUN_AS service principal — not the end user. Ensure
          the owner has{' '}
          <code style={{ fontFamily: 'var(--font-mono-db)' }}>CAN_QUERY</code>{' '}
          on the target endpoint.
        </InfoCard>
      </div>

      {/* Mode tabs */}
      <div style={{ marginTop: 14 }}>
        <ModeTabs value={mode} onChange={setMode} />
      </div>

      {/* Configure form — always visible */}
      <fieldset
        style={{ marginTop: 16, border: 0, padding: 0 }}
        disabled={mutation.isPending || phase === 'running'}
      >
        <div style={{ marginBottom: 10 }}>
          <label
            style={{
              display: 'block',
              fontSize: 12,
              fontWeight: 500,
              color: 'var(--db-navy-800)',
              marginBottom: 4,
            }}
          >
            Target serving endpoint
          </label>
          <input
            data-testid="spark-batch-endpoint-input"
            value={targetEndpoint}
            onChange={(e) => setTargetEndpoint(e.target.value)}
            placeholder="databricks-claude-sonnet-4-5"
            style={{
              width: '100%',
              border: '1px solid var(--db-gray-lines)',
              background: '#fff',
              borderRadius: 6,
              padding: '7px 10px',
              fontFamily: 'var(--font-mono-db)',
              fontSize: 12,
              color: 'var(--db-navy-800)',
              outline: 'none',
              boxSizing: 'border-box',
            }}
          />
          <div
            style={{
              fontSize: 11,
              color: 'var(--db-gray-text)',
              marginTop: 3,
            }}
          >
            Any serving endpoint (a Mode-3 deployment, a Databricks bundled
            foundation endpoint, or a custom one).
          </div>
        </div>

        <div
          style={{
            display: 'grid',
            gridTemplateColumns: '1fr 1fr',
            gap: 10,
            marginBottom: 10,
          }}
        >
          <div>
            <label
              style={{
                display: 'block',
                fontSize: 12,
                fontWeight: 500,
                color: 'var(--db-navy-800)',
                marginBottom: 4,
              }}
            >
              Input table
            </label>
            <input
              data-testid="spark-batch-input-table-input"
              value={inputTable}
              onChange={(e) => setInputTable(e.target.value)}
              placeholder="catalog.schema.table"
              style={{
                width: '100%',
                border: '1px solid var(--db-gray-lines)',
                background: '#fff',
                borderRadius: 6,
                padding: '7px 10px',
                fontFamily: 'var(--font-mono-db)',
                fontSize: 12,
                color: 'var(--db-navy-800)',
                outline: 'none',
                boxSizing: 'border-box',
              }}
            />
            {!inputTableValid && inputTable.length > 0 ? (
              <span
                role="alert"
                style={{ fontSize: 11, color: 'var(--db-lava-600)', marginTop: 2, display: 'block' }}
              >
                Must be a 3-level UC name
              </span>
            ) : null}
          </div>
          <div>
            <label
              style={{
                display: 'block',
                fontSize: 12,
                fontWeight: 500,
                color: 'var(--db-navy-800)',
                marginBottom: 4,
              }}
            >
              Output table
            </label>
            <input
              data-testid="spark-batch-output-table-input"
              value={outputTable}
              onChange={(e) => setOutputTable(e.target.value)}
              placeholder="catalog.schema.table"
              style={{
                width: '100%',
                border: '1px solid var(--db-gray-lines)',
                background: '#fff',
                borderRadius: 6,
                padding: '7px 10px',
                fontFamily: 'var(--font-mono-db)',
                fontSize: 12,
                color: 'var(--db-navy-800)',
                outline: 'none',
                boxSizing: 'border-box',
              }}
            />
            {!outputTableValid && outputTable.length > 0 ? (
              <span
                role="alert"
                style={{ fontSize: 11, color: 'var(--db-lava-600)', marginTop: 2, display: 'block' }}
              >
                Must be a 3-level UC name
              </span>
            ) : null}
          </div>
        </div>

        <div style={{ marginBottom: 10 }}>
          <label
            style={{
              display: 'block',
              fontSize: 12,
              fontWeight: 500,
              color: 'var(--db-navy-800)',
              marginBottom: 4,
            }}
          >
            Prompt column
          </label>
          <input
            data-testid="spark-batch-prompt-column-input"
            value={promptColumn}
            onChange={(e) => setPromptColumn(e.target.value)}
            placeholder="query"
            style={{
              width: '100%',
              border: '1px solid var(--db-gray-lines)',
              background: '#fff',
              borderRadius: 6,
              padding: '7px 10px',
              fontFamily: 'var(--font-mono-db)',
              fontSize: 12,
              color: 'var(--db-navy-800)',
              outline: 'none',
              boxSizing: 'border-box',
            }}
          />
        </div>

        <div>
          <label
            style={{
              display: 'block',
              fontSize: 12,
              fontWeight: 500,
              color: 'var(--db-navy-800)',
              marginBottom: 4,
            }}
          >
            Optional response_format (JSON)
          </label>
          <textarea
            data-testid="spark-batch-response-format-input"
            value={responseFormatJson}
            onChange={(e) => setResponseFormatJson(e.target.value)}
            rows={3}
            placeholder='{"type": "json_schema", "schema": {"type": "object"}}'
            style={{
              width: '100%',
              border: '1px solid var(--db-gray-lines)',
              background: '#fff',
              borderRadius: 6,
              padding: '7px 10px',
              fontFamily: 'var(--font-mono-db)',
              fontSize: 12,
              color: 'var(--db-navy-800)',
              outline: 'none',
              resize: 'vertical',
              boxSizing: 'border-box',
            }}
          />
          {responseFormatError ? (
            <span
              role="alert"
              style={{ fontSize: 11, color: 'var(--db-lava-600)', marginTop: 2, display: 'block' }}
            >
              {responseFormatError}
            </span>
          ) : null}
        </div>
      </fieldset>

      {/* Mutation error */}
      {mutation.isError ? (
        <div
          role="alert"
          style={{
            marginTop: 12,
            padding: '8px 12px',
            borderRadius: 6,
            background: 'var(--db-lava-100)',
            border: '1px solid var(--db-lava-300)',
            fontSize: 12,
            color: 'var(--db-lava-700)',
          }}
        >
          {formatDeploymentError(mutation.error)}
        </div>
      ) : null}

      {/* 'here' mode body */}
      {mode === 'here' && (
        <div style={{ marginTop: 18 }}>
          <InfoCard color="yellow">
            Live Lakeflow pipeline creation is not yet wired — this preview
            animates the steps. The actual backend call lands in a follow-up
            turn.
          </InfoCard>

          {phase === 'idle' && (
            <div style={{ marginTop: 14 }}>
              <SectionTitle>What happens when you click Deploy now</SectionTitle>
              <ProgressList steps={PROGRESS_STEPS} currentIdx={-1} />
            </div>
          )}

          {phase !== 'idle' && (
            <div style={{ marginTop: 14 }}>
              <SectionTitle>
                {phase === 'done' ? 'Pipeline ready' : 'Setting up…'}
              </SectionTitle>
              <ProgressList steps={PROGRESS_STEPS} currentIdx={stepIdx} />
              {phase === 'done' && (
                <InfoCard color="green">
                  <strong>Pipeline created</strong> · mock pipeline ID:{' '}
                  <code style={{ fontFamily: 'var(--font-mono-db)' }}>
                    {slugAgent}-pipeline-preview
                  </code>
                  . The first batch ran successfully.
                </InfoCard>
              )}
            </div>
          )}
        </div>
      )}

      {/* 'other' mode body */}
      {mode === 'other' && (
        <div style={{ marginTop: 18 }}>
          <HostField
            value={targetHost}
            onChange={setTargetHost}
            hint="The Lakeflow workspace where the pipeline will live."
          />

          <ol style={{ listStyle: 'none', padding: 0, margin: '16px 0 0' }}>
            <Step
              n={1}
              title="Generate the batch SQL artifact"
              body="We render an ai_query() SQL block with your endpoint + table refs."
              note="Lakeflow pipelines require pipelines.channel = 'preview' for ai_query — already embedded."
            />
            <Step
              n={2}
              title="Authenticate against the target workspace"
              code={`databricks auth login --host ${targetHost || '<TARGET_HOST>'}`}
              codeLang="bash"
              codeLabel="auth"
            />
            <Step
              n={3}
              title="Upload the SQL to the workspace"
              code={`databricks workspace import-dir ./batch /Workspace/Users/$DATABRICKS_USER/.lakeflow/${slugAgent}/ --overwrite`}
              codeLang="bash"
              codeLabel="upload"
            />
            <Step
              n={4}
              title="Create the pipeline"
              code="databricks pipelines create --json @pipeline.json"
              codeLang="bash"
              codeLabel="create"
            />
            <Step
              n={5}
              title="Trigger the first update"
              code="databricks pipelines start-update --pipeline-id $PIPELINE_ID"
              codeLang="bash"
              codeLabel="start"
            />
          </ol>
        </div>
      )}
    </DialogShell>
  )
}
