/**
 * Local tool-name generation for the search-first picker.
 *
 * Names are derived from the selected target (UC FQN tail, python import attr,
 * or the kind itself), sanitized to snake_case, and deduped against the
 * workflow's declared tools — the "Local tool name" field is advanced-only, so
 * these helpers own the default every declaration gets.
 */

const NAME_UNSAFE_RE = /[^a-z0-9_]+/g;

export function sanitizeToolName(raw: string): string {
  const collapsed = raw
    .trim()
    .toLowerCase()
    .replace(NAME_UNSAFE_RE, '_')
    .replace(/_{2,}/g, '_')
    .replace(/^_+|_+$/g, '');
  const named = collapsed || 'tool';
  return /^[0-9]/.test(named) ? `fn_${named}` : named;
}

/**
 * Derive a human alias from the selected target:
 * - `main.metrics.pct_change` -> `pct_change`
 * - `my_pkg.tools:normalize_text` -> `normalize_text`
 * - kind `web_search` (no target) -> `web_search`
 */
export function suggestedToolName(kind: string, targetValue?: string): string {
  const target = targetValue?.trim();
  if (target) {
    if (target.includes(':')) {
      return sanitizeToolName(target.split(':').pop() ?? target);
    }
    if (target.includes('.')) {
      return sanitizeToolName(target.split('.').pop() ?? target);
    }
    return sanitizeToolName(target);
  }
  return sanitizeToolName(kind);
}

/** `pct_change`, then `pct_change_2`, `pct_change_3`, ... */
export function uniqueToolName(
  base: string,
  existing: ReadonlyArray<{ name: string }>,
): string {
  const taken = new Set(existing.map((tool) => tool.name));
  if (!taken.has(base)) return base;
  let suffix = 2;
  while (taken.has(`${base}_${suffix}`)) {
    suffix += 1;
  }
  return `${base}_${suffix}`;
}
