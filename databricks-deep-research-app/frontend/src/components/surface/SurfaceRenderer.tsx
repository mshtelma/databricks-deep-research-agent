/* eslint-disable react-refresh/only-export-components */
/**
 * SurfaceRenderer — renders a declarative Surface into React elements.
 *
 * Finds the "root" component, resolves children recursively via the catalog
 * map, and guards against render cycles (skips ids already on the path).
 * Missing child ids produce an error chip rather than crashing.
 *
 * Also exports `useSurfaceDataModel` — a small hook for local data-model state.
 */

import * as React from 'react';
import type { ReactNode } from 'react';
import type { Surface, SurfaceComponent, RunReference } from '@/types/surface';
import type { CitationContext } from '@/components/common';
import { setAtPointer } from '@/lib/surfaceState'; // used by useSurfaceDataModel
import { renderComponent } from './catalog';
import type { SurfaceRenderContext } from './catalog';

// ---------------------------------------------------------------------------
// Props
// ---------------------------------------------------------------------------

export interface SurfaceRendererProps {
  surface: Surface;
  dataModel: Record<string, unknown>;
  /** Called with (pointer, value) whenever an input changes. */
  onDataModelChange: (pointer: string, value: unknown) => void;
  onAction: (action: string) => void;
  actionDisabled?: boolean;
  resolveRunReference?: (ref: RunReference | null) => ReactNode;
  /** Per-message citation data for structured-output cells with [Key] markers. */
  resolveCitations?: (messageId: string) => Map<string, CitationContext> | undefined;
  /** Re-run structured-output wires for a message + slots (failed-slot retry). */
  retryStructuring?: (messageId: string, slots: string[]) => void;
  /** Optional subset of top-level roots to render. Defaults to ["root"]. */
  rootIds?: string[];
  /** Component ids hidden by the host frame (legacy platform controls, host action bar). */
  suppressComponentIds?: ReadonlySet<string>;
}

// ---------------------------------------------------------------------------
// SurfaceRenderer
// ---------------------------------------------------------------------------

export function SurfaceRenderer({
  surface,
  dataModel,
  onDataModelChange,
  onAction,
  actionDisabled = false,
  resolveRunReference,
  resolveCitations,
  retryStructuring,
  rootIds,
  suppressComponentIds,
}: SurfaceRendererProps): React.ReactElement {
  // Build a lookup map of components by id
  const byId = React.useMemo<Map<string, SurfaceComponent>>(() => {
    const m = new Map<string, SurfaceComponent>();
    for (const comp of surface.components) {
      m.set(comp.id, comp);
    }
    return m;
  }, [surface.components]);

  const setValue = React.useCallback(
    (pointer: string, value: unknown) => {
      onDataModelChange(pointer, value);
    },
    [onDataModelChange],
  );

  // Recursive render — renderPath tracks ids already on the current branch to
  // break cycles (a guard; the validator should have caught them already).
  function renderNode(id: string, renderPath: ReadonlySet<string>): React.ReactElement {
    if (suppressComponentIds?.has(id)) {
      return <React.Fragment key={`suppressed-${id}`} />;
    }
    if (renderPath.has(id)) {
      return (
        <span
          key={`cycle-${id}`}
          className="inline-flex items-center rounded bg-db-lava-300 px-2 py-0.5 font-db-mono text-[11px] text-db-lava-800"
        >
          Cycle detected: &quot;{id}&quot;
        </span>
      );
    }

    const comp = byId.get(id);
    if (!comp) {
      return (
        <span
          key={`missing-${id}`}
          className="inline-flex items-center rounded bg-db-lava-300 px-2 py-0.5 font-db-mono text-[11px] text-db-lava-800"
        >
          Missing component: &quot;{id}&quot;
        </span>
      );
    }

    const nextPath = new Set(renderPath);
    nextPath.add(id);

    const ctx: SurfaceRenderContext = {
      dataModel,
      setValue,
      onAction,
      actionDisabled,
      renderChildren: (childIds: string[]): ReactNode =>
        childIds.map((childId) => renderNode(childId, nextPath)),
      resolveRunReference,
      resolveCitations,
      retryStructuring,
      getComponent: (componentId: string) => byId.get(componentId),
    };

    return React.cloneElement(renderComponent(comp, ctx), { key: comp.id });
  }

  const roots = rootIds && rootIds.length > 0 ? rootIds : ['root'];
  const missingRoots = roots.filter((rootId) => !byId.has(rootId));
  if (missingRoots.length > 0) {
    return (
      <span className="inline-flex items-center rounded bg-db-lava-300 px-2 py-0.5 font-db-mono text-[11px] text-db-lava-800">
        Surface error: missing root component{missingRoots.length > 1 ? 's' : ''}{' '}
        &quot;{missingRoots.join(', ')}&quot;
      </span>
    );
  }

  return <>{roots.map((rootId) => renderNode(rootId, new Set()))}</>;
}

// ---------------------------------------------------------------------------
// useSurfaceDataModel
// ---------------------------------------------------------------------------

/**
 * Small hook that manages local data model state for a Surface.
 *
 * Returns [dataModel, setValue, reset].
 * - `setValue(pointer, value)` immutably updates the data model.
 * - `reset()` restores the initial data model.
 */
export function useSurfaceDataModel(
  initial: Record<string, unknown>,
): [
  Record<string, unknown>,
  (pointer: string, value: unknown) => void,
  (next?: Record<string, unknown>) => void,
] {
  const [dataModel, setDataModel] = React.useState<Record<string, unknown>>(initial);

  const setValue = React.useCallback(
    (pointer: string, value: unknown) => {
      setDataModel((prev) => setAtPointer(prev, pointer, value));
    },
    [],
  );

  const reset = React.useCallback(
    (next?: Record<string, unknown>) => {
      setDataModel(next ?? initial);
    },
    [initial],
  );

  return [dataModel, setValue, reset];
}
