import type { RevisionDetail } from '@/api/agentsV2'
import type { AST, Block } from '@/types/ast'

const DEFAULT_WORKFLOW_NAME = 'Untitled Agent'
const DEFAULT_ROOT_CHILD_IDS = ['coordinator', 'plan-and-execute', 'synthesizer']

export interface RevisionProvenance {
  agentId: string
  revisionId: string
  shortAgentId: string
  shortRevisionId: string
  workflowName: string
  descriptionPreview: string
  rootChildSummary: string[]
  isDefaultScaffold: boolean
  plannerGuidancePresent: boolean
}

function cleanText(value: unknown): string {
  return typeof value === 'string' ? value.trim() : ''
}

export function shortId(value: string): string {
  return value.slice(0, 8)
}

function blockChildren(definition: AST | Record<string, unknown>): Block[] {
  const root = definition.root
  if (!root || typeof root !== 'object' || Array.isArray(root)) return []
  const children = (root as { children?: unknown }).children
  return Array.isArray(children) ? (children as Block[]) : []
}

export function summarizeRootChildren(
  definition: AST | Record<string, unknown>,
): string[] {
  return blockChildren(definition)
    .filter((child) => child && typeof child === 'object')
    .map((child) => {
      const nodeId = cleanText(child.id) || '<unnamed>'
      const nodeType = cleanText(child.type) || '<unknown>'
      const label = cleanText(child.label)
      return `${nodeId}:${nodeType}:${label}`
    })
}

function rootChildIds(definition: AST | Record<string, unknown>): string[] {
  return blockChildren(definition).map((child) => cleanText(child.id))
}

export function hasPlannerGuidance(value: unknown): boolean {
  if (Array.isArray(value)) {
    return value.some((item) => hasPlannerGuidance(item))
  }
  if (!value || typeof value !== 'object') return false
  const record = value as Record<string, unknown>
  const plannerGuidance = cleanText(record.planner_guidance)
  const promptGuidance = cleanText(record.prompt_guidance)
  if (plannerGuidance || promptGuidance) return true
  return Object.values(record).some((item) => hasPlannerGuidance(item))
}

export function isDefaultScaffoldDefinition(
  definition: AST | Record<string, unknown>,
): boolean {
  const workflowName = cleanText(definition.name)
  const workflowDescription = cleanText(definition.description)
  const children = rootChildIds(definition)
  return (
    (workflowName === '' || workflowName === DEFAULT_WORKFLOW_NAME) &&
    workflowDescription === '' &&
    children.length === DEFAULT_ROOT_CHILD_IDS.length &&
    children.every((id, idx) => id === DEFAULT_ROOT_CHILD_IDS[idx]) &&
    !hasPlannerGuidance(definition)
  )
}

export function buildRevisionProvenance(
  agentId: string,
  revision: RevisionDetail,
): RevisionProvenance {
  const definition = revision.definition
  return {
    agentId,
    revisionId: revision.rev_id,
    shortAgentId: shortId(agentId),
    shortRevisionId: shortId(revision.rev_id),
    workflowName: cleanText(definition.name) || DEFAULT_WORKFLOW_NAME,
    descriptionPreview: cleanText(definition.description).slice(0, 140),
    rootChildSummary: summarizeRootChildren(definition),
    isDefaultScaffold: isDefaultScaffoldDefinition(definition),
    plannerGuidancePresent: hasPlannerGuidance(definition),
  }
}

export function deploymentIdentityMatches(
  actual: AST | Record<string, unknown>,
  expected: AST | Record<string, unknown>,
): boolean {
  return (
    cleanText(actual.name) === cleanText(expected.name) &&
    cleanText(actual.description) === cleanText(expected.description) &&
    summarizeRootChildren(actual).join('|') === summarizeRootChildren(expected).join('|')
  )
}
