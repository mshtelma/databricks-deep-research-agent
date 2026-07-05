import type { QueryMode } from '@/types';
import type { SourceScope } from '@/types/dataSources';
import type { QuerySubmission } from '@/types/querySubmission';
import type { ResearchDepth } from '@/components/chat/ResearchDepthSelector';

export interface RunContext {
  queryMode?: QueryMode;
  researchDepth?: ResearchDepth;
  verifySources?: boolean;
  outputType?: string;
  sourceScope?: SourceScope;
  enabledSources?: string[];
  disabledSources?: string[];
  fileIds?: string[];
  agentId?: string;
  enablePlanReview?: boolean;
  turnIntent?: 'auto' | 'chat' | 'research';
  tone?: string;
  outputLanguage?: string;
  enabledMcpServers?: string[];
  enabledSkills?: string[];
  enableCrossSessionMemory?: boolean;
  allowLiveSearch?: boolean;
}

const RUN_CONTEXT_KEYS: readonly (keyof RunContext)[] = [
  'queryMode',
  'researchDepth',
  'verifySources',
  'outputType',
  'sourceScope',
  'enabledSources',
  'disabledSources',
  'fileIds',
  'agentId',
  'enablePlanReview',
  'turnIntent',
  'tone',
  'outputLanguage',
  'enabledMcpServers',
  'enabledSkills',
  'enableCrossSessionMemory',
  'allowLiveSearch',
];

export function mergeRunContext(...contexts: Array<RunContext | undefined>): RunContext {
  const merged: RunContext = {};
  for (const context of contexts) {
    if (!context) continue;
    for (const key of RUN_CONTEXT_KEYS) {
      const value = context[key];
      if (value !== undefined) {
        (merged as Record<string, unknown>)[key] = value;
      }
    }
  }
  return merged;
}

function keepArray(value: string[] | undefined): string[] | undefined {
  return value && value.length > 0 ? value : undefined;
}

export function buildQuerySubmission({
  message,
  runContext,
  surfaceInputs,
  surfaceAction,
}: {
  message: string;
  runContext: RunContext;
  surfaceInputs?: Record<string, string | number | boolean>;
  surfaceAction?: string;
}): QuerySubmission {
  return {
    message,
    queryMode: runContext.queryMode,
    researchDepth: runContext.researchDepth,
    verifySources: runContext.verifySources,
    outputType: runContext.outputType || undefined,
    sourceScope: runContext.sourceScope,
    enabledSources: keepArray(runContext.enabledSources),
    disabledSources: keepArray(runContext.disabledSources),
    fileIds: keepArray(runContext.fileIds),
    agentId: runContext.agentId,
    enablePlanReview: runContext.enablePlanReview,
    turnIntent: runContext.turnIntent,
    tone: runContext.tone || undefined,
    outputLanguage: runContext.outputLanguage || undefined,
    enabledMcpServers: keepArray(runContext.enabledMcpServers),
    enabledSkills: keepArray(runContext.enabledSkills),
    enableCrossSessionMemory: runContext.enableCrossSessionMemory,
    allowLiveSearch: runContext.allowLiveSearch,
    surfaceInputs:
      surfaceInputs && Object.keys(surfaceInputs).length > 0 ? surfaceInputs : undefined,
    surfaceAction,
  };
}

export function runContextActiveCount(
  runContext: RunContext,
  defaults: RunContext = {},
): number {
  let count = 0;
  for (const key of RUN_CONTEXT_KEYS) {
    const current = runContext[key];
    if (current === undefined) continue;
    const baseline = defaults[key];
    if (Array.isArray(current)) {
      const base = Array.isArray(baseline) ? baseline : [];
      if (
        current.length !== base.length ||
        current.some((value, index) => value !== base[index])
      ) {
        count += 1;
      }
      continue;
    }
    if (current !== baseline) count += 1;
  }
  return count;
}
