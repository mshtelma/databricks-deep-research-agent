import type { ToolKindSpec } from '@/types/agentDesigner';

export type ToolFamily = 'builtin' | 'databricks' | 'python' | 'mcp_external' | 'other';

export const TOOL_FAMILY_LABELS: Record<ToolFamily, string> = {
  builtin: 'Built-in',
  databricks: 'Databricks',
  python: 'Python',
  mcp_external: 'MCP/External',
  other: 'Other',
};

export const TOOL_FAMILY_ORDER: ToolFamily[] = [
  'builtin',
  'databricks',
  'python',
  'mcp_external',
  'other',
];

const FAMILY_BY_KIND: Record<string, ToolFamily> = {
  decorated: 'python',
  python_function: 'python',
  registered: 'python',
  uc_function: 'databricks',
  vector_search: 'databricks',
  vector_index: 'databricks',
  table_search: 'databricks',
  genie: 'databricks',
  mcp: 'mcp_external',
  enterprise: 'mcp_external',
};

export function familyForToolKind(spec: ToolKindSpec): ToolFamily {
  const mapped = FAMILY_BY_KIND[spec.kind];
  if (mapped) return mapped;
  if (spec.layer === 'E') {
    return 'mcp_external';
  }
  if (spec.layer === 'D') {
    return 'other';
  }
  return 'builtin';
}
