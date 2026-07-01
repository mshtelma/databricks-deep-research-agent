import type { NodeType } from '@/types/ast';

const GENERIC_LABEL_PATTERN =
  /^(agent|block|node|step|lane|researcher|planner|coordinator|reflector|synthesizer|background|tool|sequence|parallel|loop|conditional|subworkflow|plan(?:\s+and\s+execute)?)(?:\s*(?:#?\d+|[ivx]+|[a-z]))?$/i;

const ROLE_FALLBACKS: Record<string, string> = {
  coordinator: 'Research Coordinator',
  planner: 'Research Planner',
  researcher: 'Evidence Researcher',
  reflector: 'Coverage Reflector',
  synthesizer: 'Report Synthesizer',
  background: 'Background Researcher',
};

const NODE_FALLBACKS: Record<NodeType, string> = {
  agent: 'Workflow Agent',
  tool: 'Tool Step',
  sequence: 'Workflow Sequence',
  parallel: 'Parallel Workstreams',
  loop: 'Iteration Loop',
  conditional: 'Decision Branch',
  subworkflow: 'Subworkflow',
  plan_and_execute: 'Plan and Execute',
};

function cleanLabel(value: string): string {
  return value.replace(/[_-]+/g, ' ').replace(/\s+/g, ' ').trim();
}

function compact(value: string): string {
  const cleaned = cleanLabel(value).replace(/[.,:;]+$/g, '');
  return cleaned.length <= 72 ? cleaned : `${cleaned.slice(0, 69).trim()}...`;
}

function isGenericLabel(value: string): boolean {
  return GENERIC_LABEL_PATTERN.test(cleanLabel(value));
}

function titleFromIdentifier(value: string): string {
  const words = cleanLabel(value).match(/[A-Za-z][A-Za-z0-9&/+-]*/g) ?? [];
  if (words.length === 0) return '';
  return compact(
    words
      .slice(0, 7)
      .map((word) => (/^[A-Z0-9]+$/.test(word) ? word : word[0]!.toUpperCase() + word.slice(1)))
      .join(' '),
  );
}

export function semanticNodeLabel(
  nodeType: NodeType,
  requestedLabel: string,
  config: Record<string, unknown> = {},
): string {
  const label = compact(requestedLabel);
  if (label && !isGenericLabel(label)) return label;

  const subtype = typeof config.subtype === 'string' ? config.subtype.trim() : '';
  const roleFallback = subtype ? ROLE_FALLBACKS[subtype] : undefined;
  if (roleFallback) return roleFallback;

  const ref = typeof config.ref === 'string' ? config.ref : '';
  const toolName = typeof config.tool_name === 'string' ? config.tool_name : '';
  const toolLabel = titleFromIdentifier(ref || toolName);
  if (toolLabel) return nodeType === 'tool' ? `${toolLabel} Tool` : toolLabel;

  return NODE_FALLBACKS[nodeType] ?? 'Workflow Object';
}
