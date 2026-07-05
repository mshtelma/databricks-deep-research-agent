import type { Surface, SurfaceComponent, SurfaceSectionLayout } from '@/types/surface';
import { isPathRef } from '@/types/surface';
import { getAtPointer } from '@/lib/surfaceState';
import { INPUT_COMPONENTS, RESULT_COMPONENTS } from './surfaceComponents';

export interface DerivedSurfaceLayout {
  inputs: SurfaceSectionLayout;
  results: SurfaceSectionLayout;
  actions: 'inline' | 'host_bar';
}

const LEGACY_RUN_OPTION_POINTERS = new Set([
  '/options/research_depth',
  '/options/verify_sources',
]);

function byId(surface: Surface): Map<string, SurfaceComponent> {
  return new Map(surface.components.map((component) => [component.id, component]));
}

/** Cycle-safe: does the subtree rooted at `id` contain a component matching `pred`?
 * Shared by the input/result classifiers. (The frontend normalizer does not reject
 * cycles, so the `seen` guard is load-bearing.) */
function subtreeMatches(
  id: string,
  components: Map<string, SurfaceComponent>,
  pred: (component: SurfaceComponent) => boolean,
  seen = new Set<string>(),
): boolean {
  if (seen.has(id)) return false;
  seen.add(id);
  const component = components.get(id);
  if (!component) return false;
  if (pred(component)) return true;
  return component.children.some((childId) =>
    subtreeMatches(childId, components, pred, seen),
  );
}

function hasInputDescendant(
  id: string,
  components: Map<string, SurfaceComponent>,
): boolean {
  return subtreeMatches(id, components, (c) => INPUT_COMPONENTS.has(c.component));
}

function hasResultDescendant(
  id: string,
  components: Map<string, SurfaceComponent>,
): boolean {
  return subtreeMatches(id, components, (c) => RESULT_COMPONENTS.has(c.component));
}

export function legacyRunOptionComponentIds(surface: Surface): Set<string> {
  const components = byId(surface);
  const suppressed = new Set<string>();

  for (const component of surface.components) {
    const value = component.props['value'];
    if (isPathRef(value) && LEGACY_RUN_OPTION_POINTERS.has(value.path)) {
      suppressed.add(component.id);
    }
  }

  let changed = true;
  while (changed) {
    changed = false;
    for (const component of surface.components) {
      if (suppressed.has(component.id)) continue;
      if (component.children.length === 0) continue;
      if (component.children.every((childId) => suppressed.has(childId))) {
        suppressed.add(component.id);
        changed = true;
      }
    }
  }

  for (const id of [...suppressed]) {
    if (!components.has(id)) suppressed.delete(id);
  }
  return suppressed;
}

export function legacyRunOptionDefaults(surface: Surface): {
  researchDepth?: string;
  verifySources?: boolean;
} {
  const researchDepth = getAtPointer(surface.data_model, '/options/research_depth');
  const verifySources = getAtPointer(surface.data_model, '/options/verify_sources');
  return {
    researchDepth: typeof researchDepth === 'string' ? researchDepth : undefined,
    verifySources: typeof verifySources === 'boolean' ? verifySources : undefined,
  };
}

function sectionForRole(
  surface: Surface,
  role: 'inputs' | 'results',
): SurfaceSectionLayout | null {
  return surface.layout?.sections?.find((section) => section.role === role) ?? null;
}

/**
 * Classify each top-level (root) child into inputs vs results by CONTENT:
 * an input-bearing subtree → inputs; an otherwise result-only subtree → results;
 * a purely static subtree (headings/spacers) → inputs (shown above the form).
 * Content-based, so a form card that also holds a StatusBadge stays in inputs.
 */
function partitionRoot(surface: Surface): {
  inputChildren: string[];
  resultChildren: string[];
  rootChildren: string[];
} {
  const components = byId(surface);
  const rootChildren = components.get('root')?.children ?? ['root'];
  const inputChildren: string[] = [];
  const resultChildren: string[] = [];
  for (const childId of rootChildren) {
    if (hasInputDescendant(childId, components)) {
      inputChildren.push(childId);
    } else if (hasResultDescendant(childId, components)) {
      resultChildren.push(childId);
    } else {
      inputChildren.push(childId);
    }
  }
  return { inputChildren, resultChildren, rootChildren };
}

/** Explicit section children win when non-empty; an empty/missing list falls back
 * to the tree-derived children so a declared-but-unpopulated section still renders. */
function resolveSectionChildren(
  section: SurfaceSectionLayout,
  derived: string[],
): string[] {
  return (section.children ?? []).length > 0 ? section.children : derived;
}

/**
 * Resolve the host Inputs/Results sections for a surface.
 *
 * Contract: `layout.sections` declare section identity/title/role/order; their
 * `children` are OPTIONAL. When a section's children is empty/missing the host
 * derives them from the component tree — extending the documented
 * "missing layout → inference" behavior to missing section children.
 *
 * The "never dead-empty" `rootChildren` fallback applies ONLY to the no-layout
 * path (a layout-less surface must never render nothing). In the explicit-sections
 * path the tree partition is used as-is, so a legitimately results-only surface
 * keeps its correct empty Inputs instead of duplicating the whole tree.
 */
export function deriveSurfaceLayout(surface: Surface): DerivedSurfaceLayout {
  const { inputChildren, resultChildren, rootChildren } = partitionRoot(surface);
  const explicitInputs = sectionForRole(surface, 'inputs');
  const explicitResults = sectionForRole(surface, 'results');
  const actions = surface.layout?.actions ?? 'inline';

  if (explicitInputs || explicitResults) {
    if (
      import.meta.env.DEV &&
      ((explicitInputs && (explicitInputs.children ?? []).length === 0) ||
        (explicitResults && (explicitResults.children ?? []).length === 0))
    ) {
      console.debug(
        '[surface] explicit layout.sections had no children; deriving from component tree',
      );
    }
    return {
      inputs: explicitInputs
        ? {
            ...explicitInputs,
            children: resolveSectionChildren(explicitInputs, inputChildren),
          }
        : { id: 'inputs', title: 'Inputs', role: 'inputs', children: inputChildren },
      results: explicitResults
        ? {
            ...explicitResults,
            children: resolveSectionChildren(explicitResults, resultChildren),
          }
        : { id: 'results', title: 'Results', role: 'results', children: resultChildren },
      actions,
    };
  }

  return {
    inputs: {
      id: 'inputs',
      title: 'Inputs',
      role: 'inputs',
      children: inputChildren.length > 0 ? inputChildren : rootChildren,
      default_open: 'before_first_run',
    },
    results: {
      id: 'results',
      title: 'Results',
      role: 'results',
      children: resultChildren,
      default_open: 'after_run',
    },
    actions,
  };
}

export function surfaceInputSummary(
  surface: Surface,
  dataModel: Record<string, unknown>,
): string {
  const items: string[] = [];
  for (const component of surface.components) {
    if (!INPUT_COMPONENTS.has(component.component)) {
      continue;
    }
    const value = component.props['value'];
    if (!isPathRef(value)) continue;
    if (LEGACY_RUN_OPTION_POINTERS.has(value.path)) continue;
    const resolved = getAtPointer(dataModel, value.path);
    if (resolved === undefined || resolved === null || resolved === '') continue;
    const label =
      typeof component.props['label'] === 'string' ? component.props['label'] : component.id;
    items.push(`${label}: ${String(resolved)}`);
    if (items.length === 2) break;
  }
  return items.length > 0 ? items.join(' | ') : 'No inputs filled';
}

export function actionLabel(surface: Surface, action: string): string {
  const button = surface.components.find(
    (component) =>
      component.component === 'Button' && component.props['action'] === action,
  );
  const label = button?.props['label'];
  if (typeof label === 'string' && label.trim()) return label;
  return action
    .replace(/[_-]+/g, ' ')
    .replace(/\b\w/g, (char) => char.toUpperCase());
}
