/**
 * Pure-function path utilities for the Agent Designer workflow AST.
 *
 * Path format: dot-separated string, matching the backend mutations.py semantics.
 *   'root'                          — the root Block
 *   'root.children.0'               — first child of root
 *   'root.children.0.children.1'    — second child of root's first child
 *   'root.children.0.config.body'   — plan_and_execute body (single or sequence wrapper)
 *   'root.children.0.config.body.children.0' — first child inside plan_and_execute body
 *
 * All functions are pure and do NOT mutate their inputs.
 */

import type { AST, Block, BlockPath } from '@/types/ast';

// ---------------------------------------------------------------------------
// Path segment helpers
// ---------------------------------------------------------------------------

/**
 * Split a BlockPath into its dot-separated segments.
 *
 * @example
 * pathSegments('root.children.0') // ['root', 'children', '0']
 */
export function pathSegments(path: BlockPath): string[] {
  return path.split('.');
}

/**
 * Count the depth of a block path — the number of `.children.<n>` pairs.
 * 'root' → 0, 'root.children.0' → 1, 'root.children.0.children.1' → 2.
 */
export function depth(path: BlockPath): number {
  const segs = pathSegments(path);
  let count = 0;
  for (let i = 0; i < segs.length - 1; i++) {
    const cur = segs[i] ?? '';
    const next = segs[i + 1] ?? '';
    if (cur === 'children' && /^\d+$/.test(next)) {
      count++;
    }
  }
  return count;
}

/**
 * Return the parent path by dropping the trailing `.children.<n>` suffix.
 * Returns null for 'root' (no parent).
 *
 * @example
 * parentPath('root.children.0')            // 'root'
 * parentPath('root.children.0.children.1') // 'root.children.0'
 * parentPath('root')                        // null
 */
export function parentPath(path: BlockPath): BlockPath | null {
  if (path === 'root') return null;
  const segs = pathSegments(path);
  const last = segs[segs.length - 1] ?? '';
  const penultimate = segs[segs.length - 2] ?? '';
  // Drop the trailing `.children.<n>` pair
  if (segs.length >= 2 && /^\d+$/.test(last) && penultimate === 'children') {
    return segs.slice(0, segs.length - 2).join('.');
  }
  // Path ends with a non-index segment (e.g. 'root.children.0.config.body') — drop last segment only
  return segs.slice(0, segs.length - 1).join('.');
}

/**
 * Return the numeric index from the trailing `.children.<n>` suffix, or null
 * if the path does not end in `.children.<n>`.
 *
 * @example
 * childIndex('root.children.5')   // 5
 * childIndex('root')              // null
 */
export function childIndex(path: BlockPath): number | null {
  const segs = pathSegments(path);
  const last = segs[segs.length - 1] ?? '';
  const penultimate = segs[segs.length - 2] ?? '';
  if (segs.length >= 2 && /^\d+$/.test(last) && penultimate === 'children') {
    return parseInt(last, 10);
  }
  return null;
}

/**
 * Build the path to a child at `index` under `parent`.
 *
 * @example
 * appendChildPath('root', 2) // 'root.children.2'
 */
export function appendChildPath(parentPath: BlockPath, index: number): BlockPath {
  return `${parentPath}.children.${index}`;
}

/**
 * Return true if `candidatePath` is strictly a descendant of `ancestorPath`
 * (i.e. candidate starts with ancestor + '.' and they are not the same path).
 *
 * @example
 * isDescendant('root', 'root.children.0')   // true
 * isDescendant('root', 'root')              // false
 */
export function isDescendant(ancestorPath: BlockPath, candidatePath: BlockPath): boolean {
  if (ancestorPath === candidatePath) return false;
  return candidatePath.startsWith(ancestorPath + '.');
}

// ---------------------------------------------------------------------------
// Block resolution
// ---------------------------------------------------------------------------

/**
 * Walk the AST and return the Block at `path`, or null if the path is missing.
 *
 * Path navigation rules (matching mutations.py _get_at / _split_path semantics):
 *   - 'root'           → ast.root
 *   - 'root.children.N' → ast.root.children[N]
 *   - Numeric segments index into arrays; string segments index into objects.
 *   - For plan_and_execute, the body lives in config.body. Paths like
 *     'root.children.0.config.body' resolve through the Block's config dict,
 *     and 'root.children.0.config.body.children.0' resolves further into the
 *     body's children array. This matches the backend's add_block semantics
 *     (mutations.py line 187: body_path.endswith("config.body")).
 *
 * Returns the actual reference to the block (no clone). Pure — does not mutate.
 */
export function resolveBlock(ast: AST, path: BlockPath): Block | null {
  const segs = pathSegments(path);

  // First segment must be 'root'
  if (segs[0] !== 'root') return null;

  // Navigate starting from ast (as a generic object) using all segments
  // including 'root', because ast["root"] is the root Block.
  let current: unknown = ast;

  for (const seg of segs) {
    if (current === null || current === undefined) return null;

    if (/^\d+$/.test(seg)) {
      // Numeric: index into array
      if (!Array.isArray(current)) return null;
      const idx = parseInt(seg, 10);
      if (idx >= current.length) return null;
      current = current[idx];
    } else {
      // String: index into object
      if (typeof current !== 'object' || Array.isArray(current)) return null;
      const obj = current as Record<string, unknown>;
      if (!(seg in obj)) return null;
      current = obj[seg];
    }
  }

  if (current === null || current === undefined || typeof current !== 'object' || Array.isArray(current)) {
    return null;
  }

  return current as Block;
}
