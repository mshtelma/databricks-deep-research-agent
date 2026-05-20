import type { AST, Block, NodeType, ValidationError } from '@/types/ast';

function randomId(prefix: string): string {
  if (typeof crypto !== 'undefined' && typeof crypto.randomUUID === 'function') {
    return crypto.randomUUID();
  }
  return `${prefix}-${Math.random().toString(16).slice(2)}`;
}

export function createDraftWorkflow(name = 'Untitled Agent', description = ''): AST {
  return {
    id: randomId('workflow'),
    name,
    description,
    version: 1,
    root: {
      id: randomId('root'),
      type: 'sequence',
      label: 'Workflow',
      config: {},
      children: [],
    },
    tools: [],
    pools: [],
    sources: [],
    models: {},
    required_inputs: ['query'],
    output_keys: ['output'],
    token_budget: 0,
    timeout_seconds: 1800,
    run_as: 'caller',
  };
}

function normalizeNodeType(value: unknown): NodeType {
  return value as NodeType;
}

function normalizeBlock(raw: unknown, fallbackLabel: string): Block {
  const source = typeof raw === 'object' && raw !== null && !Array.isArray(raw)
    ? raw as Record<string, unknown>
    : {};
  const rawConfig = typeof source.config === 'object' && source.config !== null && !Array.isArray(source.config)
    ? source.config as Record<string, unknown>
    : {};
  const config: Record<string, unknown> = { ...rawConfig };

  if (typeof config.body === 'object' && config.body !== null && !Array.isArray(config.body)) {
    config.body = normalizeBlock(config.body, 'Body');
  }
  if (typeof config.evaluator === 'object' && config.evaluator !== null && !Array.isArray(config.evaluator)) {
    const evaluator = config.evaluator as Record<string, unknown>;
    const nestedConfig =
      typeof evaluator.config === 'object' && evaluator.config !== null && !Array.isArray(evaluator.config)
        ? (evaluator.config as Record<string, unknown>)
        : null;
    config.evaluator = nestedConfig && typeof nestedConfig.subtype === 'string'
      ? { ...nestedConfig }
      : { ...evaluator };
  }

  const children = Array.isArray(source.children)
    ? source.children.map((child, idx) => normalizeBlock(child, `Step ${idx + 1}`))
    : [];

  const typeValue = source.type ?? source.node_type ?? 'agent';
  return {
    id: typeof source.id === 'string' && source.id.length > 0 ? source.id : randomId('block'),
    type: normalizeNodeType(typeValue),
    label: typeof source.label === 'string' && source.label.length > 0 ? source.label : fallbackLabel,
    config,
    children,
  };
}

export function normalizeWorkflowAst(raw: unknown, fallbackName = 'Untitled Agent'): AST {
  const source = typeof raw === 'object' && raw !== null && !Array.isArray(raw)
    ? raw as Record<string, unknown>
    : {};
  const draft = createDraftWorkflow(fallbackName);
  return {
    ...source,
    id: typeof source.id === 'string' && source.id.length > 0 ? source.id : draft.id,
    name: typeof source.name === 'string' && source.name.length > 0 ? source.name : fallbackName,
    description: typeof source.description === 'string' ? source.description : draft.description,
    version: typeof source.version === 'number' ? source.version : draft.version,
    root: normalizeBlock(source.root, 'Workflow'),
    tools: Array.isArray(source.tools) ? source.tools as AST['tools'] : [],
    pools: Array.isArray(source.pools) ? source.pools as AST['pools'] : [],
    sources: Array.isArray(source.sources) ? source.sources as AST['sources'] : [],
    models: typeof source.models === 'object' && source.models !== null && !Array.isArray(source.models)
      ? source.models as Record<string, unknown>
      : {},
    required_inputs: Array.isArray(source.required_inputs) ? source.required_inputs as string[] : ['query'],
    output_keys: Array.isArray(source.output_keys) ? source.output_keys as string[] : ['output'],
    token_budget: typeof source.token_budget === 'number' ? source.token_budget : 0,
    timeout_seconds: typeof source.timeout_seconds === 'number' ? source.timeout_seconds : 1800,
    run_as: source.run_as === undefined ? 'caller' : source.run_as as AST['run_as'],
  };
}

export function normalizeValidationErrors(raw: unknown): ValidationError[] {
  if (!Array.isArray(raw)) return [];
  return raw.map((item): ValidationError => {
    if (typeof item === 'string') {
      return { message: item, path: null, line: null, kind: 'validation' };
    }
    if (typeof item === 'object' && item !== null) {
      const obj = item as Record<string, unknown>;
      return {
        message: typeof obj.message === 'string' ? obj.message : String(item),
        path: typeof obj.path === 'string' ? obj.path : null,
        line: typeof obj.line === 'number' ? obj.line : null,
        kind: obj.kind === 'syntax' || obj.kind === 'schema' || obj.kind === 'validation'
          ? obj.kind
          : 'validation',
      };
    }
    return { message: String(item), path: null, line: null, kind: 'validation' };
  });
}

export function isWorkflowEmpty(ast: AST | null): boolean {
  return !ast || (ast.root.type === 'sequence' && (ast.root.children ?? []).length === 0);
}
