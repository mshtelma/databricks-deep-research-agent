/**
 * DeployDropdown — Deploy menu trigger + popover.
 *
 * Visual treatment imported from the canonical design package (Anthropic
 * Design `agentic-designer/project/src/deploy-dialogs.jsx`, `DeployMenu`):
 * every entry has a tone-coloured 30x30 icon badge, a tag chip ("Lightest" /
 * "Most popular" / etc.), a chev-right affordance, and the popover has a
 * subtitle header + workspace footer. The Radix Popover infrastructure and
 * all `data-testid` selectors are preserved so the existing test surface
 * keeps passing.
 */

import * as Popover from '@radix-ui/react-popover'
import * as React from 'react'
import {
  Bot,
  Box,
  ChevronRight,
  Database,
  Info,
  Link as LinkIcon,
  Rocket,
  type LucideIcon,
} from 'lucide-react'

import { Button } from '@/components/ui/button'

import { getRevision } from '@/api/agentsV2'
import { InAppWizard } from './InAppWizard'
import { MlflowAgentWizard } from './MlflowAgentWizard'
import { ShellAppWizard } from './ShellAppWizard'
import { SparkBatchWizard } from './SparkBatchWizard'
import type { DeploymentMode, DeploymentResponse } from '@/types/deployment'
import {
  buildRevisionProvenance,
  type RevisionProvenance,
} from './revisionProvenance'

interface DeployDropdownProps {
  agentId: string
  agentName: string
  revisionId: string
  /** Disable the entire dropdown (e.g. while another mutation is pending). */
  disabled?: boolean
  /** Fired when any wizard finishes a deploy. */
  onDeployed: (deployment: DeploymentResponse) => void
  /**
   * Optional pre-deploy hook (W5): called when the user picks a mode entry
   * but before the wizard opens. The parent typically auto-saves dirty
   * canvas edits here and returns the freshly-saved revision id. Return
   * `null` to abort (e.g., save failed or user cancelled). When omitted,
   * the dropdown opens the wizard with the static ``revisionId`` prop.
   */
  onBeforeDeploy?: () => Promise<string | null>
}

interface ModeEntry {
  mode: DeploymentMode
  /** Lucide icon component for the per-mode tone badge. */
  icon: LucideIcon
  /** Tailwind classes for the 30x30 tone badge (bg + icon color). */
  badgeBg: string
  badgeIcon: string
  /** Title text shown in the entry. */
  label: string
  /** Description text shown below the title. */
  description: string
  /** Small mode-tag chip ("Lightest", "Most popular", etc.). */
  tag: string
  /** Tailwind classes for the tag chip (bg + text). */
  tagBg: string
  tagText: string
}

const ENTRIES: readonly ModeEntry[] = [
  {
    mode: 'in_app',
    icon: Bot,
    badgeBg: 'bg-db-blue-100',
    badgeIcon: 'text-db-blue-700',
    label: 'In-App (chat picker)',
    description:
      'Make this agent selectable from the Databricks workspace chat composer.',
    tag: 'Lightest',
    tagBg: 'bg-db-oat-medium',
    tagText: 'text-db-gray-text',
  },
  {
    mode: 'shell_app',
    icon: Box,
    badgeBg: 'bg-db-lava-100',
    badgeIcon: 'text-db-lava-600',
    label: 'Databricks App (chat UI)',
    description:
      'Standalone Databricks App with a built-in chat UI, served on Apps compute.',
    tag: 'Most popular',
    tagBg: 'bg-db-lava-100',
    tagText: 'text-db-lava-700',
  },
  {
    mode: 'mlflow_agent',
    icon: LinkIcon,
    badgeBg: 'bg-db-oat-medium',
    badgeIcon: 'text-db-navy-800',
    label: 'API Endpoint (MLflow)',
    description:
      'Log as MLflow ResponsesAgent, register in Unity Catalog, deploy as a Mosaic AI serving endpoint (OpenAI-compatible).',
    tag: 'For integrations',
    tagBg: 'bg-db-oat-medium',
    tagText: 'text-db-navy-800',
  },
  {
    mode: 'batch',
    icon: Database,
    badgeBg: 'bg-db-green-300',
    badgeIcon: 'text-db-green-700',
    label: 'Spark Batch (Lakeflow)',
    description:
      'Generate a Lakeflow Declarative Pipeline that runs ai_query() with this agent over each row of a Delta table.',
    tag: 'Batch jobs',
    tagBg: 'bg-db-green-300',
    tagText: 'text-db-green-700',
  },
]

export function DeployDropdown({
  agentId,
  agentName,
  revisionId,
  disabled = false,
  onDeployed,
  onBeforeDeploy,
}: DeployDropdownProps): React.ReactElement {
  const [open, setOpen] = React.useState(false)
  const [inAppOpen, setInAppOpen] = React.useState(false)
  const [shellAppOpen, setShellAppOpen] = React.useState(false)
  const [batchOpen, setBatchOpen] = React.useState(false)
  const [mlflowAgentOpen, setMlflowAgentOpen] = React.useState(false)
  const [pendingRevisionId, setPendingRevisionId] = React.useState<
    string | null
  >(null)
  const [pendingProvenance, setPendingProvenance] =
    React.useState<RevisionProvenance | null>(null)
  const [preparing, setPreparing] = React.useState(false)
  const [prepareError, setPrepareError] = React.useState<string | null>(null)

  const handleEntry = async (entry: ModeEntry) => {
    setOpen(false)
    setPrepareError(null)

    let resolvedRevisionId: string | null = revisionId
    setPreparing(true)
    try {
      if (onBeforeDeploy) {
        resolvedRevisionId = await onBeforeDeploy()
      }
      if (resolvedRevisionId === null) return

      const revision = await getRevision(agentId, resolvedRevisionId)
      const provenance = buildRevisionProvenance(agentId, revision)
      setPendingRevisionId(resolvedRevisionId)
      setPendingProvenance(provenance)
      console.debug('[DEPLOY_REVISION_PROVENANCE]', {
        agent_id: agentId,
        revision_id: resolvedRevisionId,
        workflow_name: provenance.workflowName,
        workflow_description: provenance.descriptionPreview,
        root_child_summary: provenance.rootChildSummary,
        default_scaffold: provenance.isDefaultScaffold,
      })
    } catch (error) {
      setPrepareError(
        error instanceof Error
          ? error.message
          : 'Could not load the saved revision before deploying.',
      )
      setOpen(true)
      return
    } finally {
      setPreparing(false)
    }

    if (entry.mode === 'in_app') {
      setInAppOpen(true)
    } else if (entry.mode === 'shell_app') {
      setShellAppOpen(true)
    } else if (entry.mode === 'batch') {
      setBatchOpen(true)
    } else if (entry.mode === 'mlflow_agent') {
      setMlflowAgentOpen(true)
    }
  }

  const activeRevisionId = pendingRevisionId ?? revisionId
  const activeAgentName = pendingProvenance?.workflowName || agentName

  return (
    <>
      <Popover.Root open={open} onOpenChange={setOpen}>
        <Popover.Trigger asChild>
          <Button
            data-testid="deploy-dropdown-trigger"
            disabled={disabled || preparing}
            variant="default"
          >
            <Rocket size={14} className="mr-1.5" />
            {preparing ? 'Saving…' : 'Deploy'}
          </Button>
        </Popover.Trigger>
        <Popover.Portal>
          <Popover.Content
            data-testid="deploy-dropdown-menu"
            sideOffset={8}
            align="end"
            className="z-50 w-[460px] overflow-hidden rounded-db-lg border border-db-gray-lines bg-white shadow-db-xl"
          >
            {/* Header: title + subtitle */}
            <div className="px-4 pb-1.5 pt-3">
              <div className="flex items-center gap-2">
                <LinkIcon size={14} className="text-db-lava-600" />
                <span className="font-db-sans text-[13px] font-semibold text-db-navy-800">
                  Deploy this agent
                </span>
              </div>
              <p className="mt-1 font-db-sans text-[11px] leading-[1.5] text-db-gray-text">
                Pick a deployment surface. You can run the deploy in-place or
                export a self-contained bundle to ship to another workspace.
              </p>
              {prepareError ? (
                <p
                  role="alert"
                  className="mt-2 rounded-md border border-db-lava-300 bg-db-lava-100 px-2 py-1 text-[11px] text-db-lava-700"
                >
                  {prepareError}
                </p>
              ) : null}
            </div>

            {/* Mode entries with tone badge + tag chip + chevRight */}
            <div className="p-2">
              {ENTRIES.map((entry) => (
                <button
                  key={entry.mode}
                  type="button"
                  data-testid={`deploy-dropdown-${entry.mode}`}
                  onClick={() => {
                    void handleEntry(entry)
                  }}
                  className="flex w-full items-start gap-3 rounded-db-md border-0 bg-transparent px-3 py-2.5 text-left transition-colors hover:bg-db-oat-light"
                >
                  <span
                    className={`flex h-[30px] w-[30px] shrink-0 items-center justify-center rounded-[7px] ${entry.badgeBg}`}
                  >
                    <entry.icon
                      size={15}
                      className={entry.badgeIcon}
                    />
                  </span>
                  <span className="min-w-0 flex-1">
                    <span className="flex items-center gap-1.5">
                      <span className="font-db-sans text-[13px] font-medium text-db-navy-800">
                        {entry.label}
                      </span>
                      <span
                        className={`rounded-sm px-1.5 py-px font-db-mono text-[9px] font-semibold uppercase tracking-[0.05em] ${entry.tagBg} ${entry.tagText}`}
                      >
                        {entry.tag}
                      </span>
                    </span>
                    <span className="mt-[3px] block font-db-sans text-[11px] leading-[1.5] text-db-gray-text">
                      {entry.description}
                    </span>
                  </span>
                  <ChevronRight
                    size={13}
                    className="mt-1 shrink-0 text-db-navy-300"
                  />
                </button>
              ))}
            </div>

            {/* Footer: provenance + workspace indicator */}
            <div className="flex items-center gap-1.5 border-t border-db-gray-lines bg-db-oat-light px-4 py-2 font-db-sans text-[11px] text-db-gray-text">
              <Info size={11} />
              <span>Saved revisions are deployable.</span>
            </div>
          </Popover.Content>
        </Popover.Portal>
      </Popover.Root>

      <InAppWizard
        open={inAppOpen}
        onOpenChange={setInAppOpen}
        agentId={agentId}
        agentName={activeAgentName}
        revisionId={activeRevisionId}
        revisionProvenance={pendingProvenance}
        onDeployed={onDeployed}
      />

      <ShellAppWizard
        open={shellAppOpen}
        onOpenChange={setShellAppOpen}
        agentId={agentId}
        agentName={activeAgentName}
        revisionId={activeRevisionId}
        revisionProvenance={pendingProvenance}
        onDeployed={onDeployed}
      />

      <SparkBatchWizard
        open={batchOpen}
        onOpenChange={setBatchOpen}
        agentId={agentId}
        agentName={activeAgentName}
        revisionId={activeRevisionId}
        revisionProvenance={pendingProvenance}
        onDeployed={onDeployed}
      />

      <MlflowAgentWizard
        open={mlflowAgentOpen}
        onOpenChange={setMlflowAgentOpen}
        agentId={agentId}
        agentName={activeAgentName}
        revisionId={activeRevisionId}
        revisionProvenance={pendingProvenance}
        onDeployed={onDeployed}
      />
    </>
  )
}
