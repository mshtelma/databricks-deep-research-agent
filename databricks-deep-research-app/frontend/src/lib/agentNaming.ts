import type { AST } from '@/types/ast';

const DEFAULT_AGENT_NAME = 'Untitled Agent';
const MAX_AGENT_NAME_LENGTH = 48;

const STOP_WORDS = new Set([
  'a',
  'about',
  'add',
  'agent',
  'agents',
  'an',
  'and',
  'based',
  'benchmark',
  'bootstrap',
  'build',
  'builder',
  'chunks',
  'chunk',
  'create',
  'data',
  'deep',
  'design',
  'for',
  'from',
  'help',
  'main',
  'make',
  'new',
  'of',
  'on',
  'please',
  'search',
  'source',
  'sources',
  'that',
  'the',
  'to',
  'tool',
  'tools',
  'use',
  'using',
  'vector',
  'with',
  'workflow',
]);

const TERM_REWRITES: Record<string, string> = {
  doc: 'documents',
  documetns: 'documents',
  documets: 'documents',
  docs: 'documents',
  treseaury: 'treasury',
};

function clampName(value: string): string {
  const trimmed = value.replace(/\s+/g, ' ').trim();
  if (trimmed.length <= MAX_AGENT_NAME_LENGTH) return trimmed;
  const words = trimmed.split(' ');
  const selected: string[] = [];
  for (const word of words) {
    const candidate = [...selected, word].join(' ');
    if (candidate.length > MAX_AGENT_NAME_LENGTH) break;
    selected.push(word);
  }
  return selected.length > 0 ? selected.join(' ') : trimmed.slice(0, MAX_AGENT_NAME_LENGTH).trim();
}

function titleToken(value: string): string {
  const lower = value.toLowerCase();
  if (lower === 'officeqa') return 'OfficeQA';
  if (['ai', 'api', 'etl', 'kpi', 'llm', 'qa', 'sql'].includes(lower)) {
    return lower.toUpperCase();
  }
  return lower.charAt(0).toUpperCase() + lower.slice(1);
}

function quotedName(prompt: string): string | null {
  const match = prompt.match(/["'“”]([^"'“”]{3,60})["'“”]/);
  return match?.[1] ? clampName(match[1]) : null;
}

export function deriveShortAgentNameFromPrompt(prompt: string): string {
  const quoted = quotedName(prompt);
  if (quoted) return quoted;

  const seen = new Set<string>();
  const keywords = prompt
    .toLowerCase()
    .replace(/[_/.-]+/g, ' ')
    .split(/\s+/)
    .map((token) => token.replace(/[^a-z0-9]/g, ''))
    .map((token) => TERM_REWRITES[token] ?? token)
    .filter((token) => token.length >= 3)
    .filter((token) => !STOP_WORDS.has(token))
    .filter((token) => {
      if (seen.has(token)) return false;
      seen.add(token);
      return true;
    })
    .slice(0, 3);

  if (keywords.length === 0) return 'Research Agent';

  const baseName = keywords.map(titleToken).join(' ');
  return clampName(`${baseName} Agent`);
}

export function isPromptLikeAgentName(value?: string | null): boolean {
  const name = value?.replace(/\s+/g, ' ').trim();
  if (!name || name === DEFAULT_AGENT_NAME) return true;
  if (name.length > MAX_AGENT_NAME_LENGTH) return true;
  if (name.split(/\s+/).length > 7) return true;
  return /^(please\s+)?(use|create|build|make|design|bootstrap|add)\b/i.test(name);
}

export function applyBootstrapAgentName(ast: AST, prompt: string): AST {
  if (!isPromptLikeAgentName(ast.name)) return ast;
  return {
    ...ast,
    name: deriveShortAgentNameFromPrompt(prompt),
  };
}
