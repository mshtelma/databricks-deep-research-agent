/**
 * AgentSurfacePanel — pinned right-side panel that renders a declarative
 * agent UI (Surface) and compiles ActionBindings into QuerySubmissions.
 *
 * Keeps form state (useSurfaceDataModel) and run state (runState prop) separate:
 * runState is overlaid read-only into a derived model for rendering without
 * mutating the hook's form state.
 */

import * as React from 'react';
import { ChevronDown, X } from 'lucide-react';
import type { Surface, RunReference } from '@/types/surface';
import type { AvailableSource } from '@/types/dataSources';
import { mergeDataModel, setAtPointer } from '@/lib/surfaceState';
import {
  compileSurfaceAction,
  type CompiledSurfaceSubmission,
} from '@/lib/surfaceCompile';
import type { RunContext } from '@/lib/runContext';
import {
  deriveSurfaceLayout,
  legacyRunOptionComponentIds,
  legacyRunOptionDefaults,
  surfaceInputSummary,
  actionLabel,
} from '@/lib/surfaceLayout';
import {
  deriveEnabledMcpServerNamesForSubmit,
  deriveEnabledSourceIdsForSubmit,
  deriveSourceScopeFromComposerSources,
  isEnterpriseAvailableSource,
} from '@/components/chat/sourceRouting';
import {
  SurfaceRunControlsBar,
  type SurfaceRunControlsState,
} from '@/components/runControls/SurfaceRunControlsBar';
import { SurfaceRenderer, useSurfaceDataModel } from './SurfaceRenderer';

// ---------------------------------------------------------------------------
// Props
// ---------------------------------------------------------------------------

export interface AgentSurfacePanelProps {
  agentName: string;
  surface: Surface;
  /** Stable identity for the selected agent surface revision. */
  surfaceIdentity?: string;
  selectedAgentId?: string;
  runContextDefaults?: RunContext;
  availableSources?: AvailableSource[];
  disabledSourceIds?: string[];
  onBrowseSources?: () => void;
  isDiscoveringSources?: boolean;
  agentDefinesSources?: boolean;
  onRunAction: (compiled: CompiledSurfaceSubmission) => void;
  runDisabled: boolean;
  runState: Record<string, RunReference | null>;
  resolveRunReference?: (ref: RunReference | null) => React.ReactNode;
  /** Per-message citation data for structured-output cells with [Key] markers. */
  resolveCitations?: (
    messageId: string,
  ) => Map<string, import('@/components/common').CitationContext> | undefined;
  /** Re-run structured-output wires for a message + slots (failed-slot retry). */
  retryStructuring?: (messageId: string, slots: string[]) => void;
  onClose?: () => void;
  /**
   * When provided and non-empty, seeds the form on first mount (or when
   * surface identity changes) instead of surface.data_model.
   */
  initialDataModel?: Record<string, unknown>;
  /**
   * Called after every user edit with the new data model.
   */
  onFormStateChange?: (dataModel: Record<string, unknown>) => void;
}

// ---------------------------------------------------------------------------
// AgentSurfacePanel
// ---------------------------------------------------------------------------

export function AgentSurfacePanel({
  agentName,
  surface,
  surfaceIdentity,
  selectedAgentId,
  runContextDefaults,
  availableSources = [],
  disabledSourceIds = [],
  onBrowseSources,
  isDiscoveringSources = false,
  agentDefinesSources = false,
  onRunAction,
  runDisabled,
  runState,
  resolveRunReference,
  resolveCitations,
  retryStructuring,
  onClose,
  initialDataModel,
  onFormStateChange,
}: AgentSurfacePanelProps): React.ReactElement {
  // Seed with initialDataModel when non-empty, otherwise fall back to surface.data_model.
  // Deep-merge persisted form state onto the current surface defaults, so a saved
  // form keeps the user's entries for fields that still exist while renamed/new
  // fields get their surface defaults (mergeDataModel is pointer-deep — form values
  // nest under a namespace like /form or /inputs).
  const seedModel = React.useMemo(
    () => mergeDataModel(surface.data_model, initialDataModel),
    [surface.data_model, initialDataModel],
  );
  const [formDataModel, setValue, reset] = useSurfaceDataModel(seedModel);
  const [inlineError, setInlineError] = React.useState<string | null>(null);
  const layout = React.useMemo(() => deriveSurfaceLayout(surface), [surface]);
  const runtimeControls = React.useMemo(
    () =>
      agentDefinesSources
        ? { ...surface.runtime_controls, sources: 'hide' as const }
        : surface.runtime_controls,
    [agentDefinesSources, surface.runtime_controls],
  );
  const sourceControlsActive = runtimeControls?.sources !== 'hide';
  const suppressedComponentIds = React.useMemo(() => {
    const ids = legacyRunOptionComponentIds(surface);
    if (layout.actions === 'host_bar') {
      for (const binding of surface.bindings) {
        const button = surface.components.find(
          (component) =>
            component.component === 'Button' &&
            component.props['action'] === binding.action,
        );
        if (button) ids.add(button.id);
      }
    }
    return ids;
  }, [layout.actions, surface]);
  const legacyDefaults = React.useMemo(() => legacyRunOptionDefaults(surface), [surface]);
  const initialRunControls = React.useMemo<SurfaceRunControlsState>(
    () => ({
      researchDepth:
        (legacyDefaults.researchDepth as SurfaceRunControlsState['researchDepth'] | undefined) ??
        runContextDefaults?.researchDepth ??
        'auto',
      sources: { web: true, ent: true, mcp: true },
      verifySources:
        legacyDefaults.verifySources ?? runContextDefaults?.verifySources ?? true,
      enablePlanReview: runContextDefaults?.enablePlanReview ?? false,
      enableCrossSessionMemory: runContextDefaults?.enableCrossSessionMemory ?? false,
      allowLiveSearch: runContextDefaults?.allowLiveSearch ?? false,
      tone: runContextDefaults?.tone ?? '',
      outputLanguage: runContextDefaults?.outputLanguage ?? '',
    }),
    [
      legacyDefaults.researchDepth,
      legacyDefaults.verifySources,
      runContextDefaults?.allowLiveSearch,
      runContextDefaults?.enableCrossSessionMemory,
      runContextDefaults?.enablePlanReview,
      runContextDefaults?.researchDepth,
      runContextDefaults?.tone,
      runContextDefaults?.outputLanguage,
      runContextDefaults?.verifySources,
    ],
  );
  const [runControls, setRunControls] =
    React.useState<SurfaceRunControlsState>(initialRunControls);
  const [sections, setSections] = React.useState({
    inputs: { open: true, userSet: false },
    results: { open: false, userSet: false },
  });

  const resolvedSurfaceIdentity = React.useMemo(
    () =>
      surfaceIdentity ??
      JSON.stringify({
        version: surface.version,
        components: surface.components,
        bindings: surface.bindings,
        data_model: surface.data_model,
      }),
    [surfaceIdentity, surface],
  );

  const dirtyRef = React.useRef(false);
  const lastSurfaceIdentityRef = React.useRef(resolvedSurfaceIdentity);
  const lastSeedRef = React.useRef(seedModel);

  // Reset on real surface identity changes. When persisted/default form state
  // arrives late for the same identity, adopt it only while the user has not
  // edited the current form.
  React.useEffect(() => {
    const identityChanged =
      lastSurfaceIdentityRef.current !== resolvedSurfaceIdentity;
    const seedChanged = lastSeedRef.current !== seedModel;

    if (identityChanged) {
      reset(seedModel);
      formDataModelRef.current = seedModel;
      dirtyRef.current = false;
      setInlineError(null);
      setRunControls(initialRunControls);
      setSections({
        inputs: { open: true, userSet: false },
        results: { open: false, userSet: false },
      });
      lastSurfaceIdentityRef.current = resolvedSurfaceIdentity;
      lastSeedRef.current = seedModel;
      return;
    }

    if (seedChanged && !dirtyRef.current) {
      reset(seedModel);
      formDataModelRef.current = seedModel;
      setInlineError(null);
    }
    lastSeedRef.current = seedModel;
  }, [initialRunControls, resolvedSurfaceIdentity, reset, seedModel]);

  // Wrap setValue so onFormStateChange fires after every user edit.
  const onFormStateChangeRef = React.useRef(onFormStateChange);
  React.useEffect(() => {
    onFormStateChangeRef.current = onFormStateChange;
  }, [onFormStateChange]);

  const handleSetValue = React.useCallback(
    (pointer: string, value: unknown) => {
      dirtyRef.current = true;
      setSections((prev) => ({
        ...prev,
        inputs: { ...prev.inputs, open: true, userSet: true },
      }));
      setValue(pointer, value);
      // Notify after the state update is scheduled; read the latest model via
      // the functional updater pattern isn't available here, so we reconstruct
      // the new model using setAtPointer directly for the callback.
      if (onFormStateChangeRef.current) {
        setFormDataModelForCallback(pointer, value);
      }
    },
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [setValue],
  );

  // Keep a ref to the current formDataModel so we can compute the next value
  // for the onFormStateChange callback without an extra render cycle.
  const formDataModelRef = React.useRef(formDataModel);
  React.useEffect(() => {
    formDataModelRef.current = formDataModel;
  }, [formDataModel]);

  const setFormDataModelForCallback = React.useCallback(
    (pointer: string, value: unknown) => {
      const next = setAtPointer(formDataModelRef.current, pointer, value);
      formDataModelRef.current = next;
      onFormStateChangeRef.current?.(next);
    },
    [],
  );

  // Overlay runState into the data model for rendering (read-only, no form mutation).
  const renderedDataModel = React.useMemo<Record<string, unknown>>(() => {
    let model: Record<string, unknown> = formDataModel;
    for (const binding of surface.bindings) {
      const ref = runState[binding.action];
      if (ref !== undefined && binding.output?.target) {
        model = setAtPointer(model, binding.output.target, ref);
      }
    }
    return model;
  }, [formDataModel, runState, surface.bindings]);

  const hasAnyRun = React.useMemo(
    () => Object.values(runState).some((ref) => ref !== null && ref !== undefined),
    [runState],
  );
  React.useEffect(() => {
    if (!hasAnyRun) return;
    setSections((prev) => ({
      inputs: prev.inputs.userSet
        ? prev.inputs
        : { ...prev.inputs, open: false },
      results: prev.results.userSet ? prev.results : { ...prev.results, open: true },
    }));
  }, [hasAnyRun]);

  const hostRunContext = React.useMemo<RunContext>(() => {
    const enabledSources =
      sourceControlsActive && availableSources.length > 0
        ? deriveEnabledSourceIdsForSubmit(availableSources, runControls.sources)
        : undefined;
    const enabledMcpServers =
      sourceControlsActive && availableSources.length > 0
        ? deriveEnabledMcpServerNamesForSubmit(availableSources, runControls.sources)
        : undefined;
    return {
      ...runContextDefaults,
      queryMode: runContextDefaults?.queryMode ?? 'deep_research',
      researchDepth: runControls.researchDepth,
      verifySources: runControls.verifySources,
      sourceScope: sourceControlsActive
        ? deriveSourceScopeFromComposerSources(runControls.sources)
        : undefined,
      enabledSources,
      disabledSources:
        sourceControlsActive && disabledSourceIds.length > 0
          ? disabledSourceIds
          : undefined,
      enabledMcpServers,
      enablePlanReview: runControls.enablePlanReview,
      enableCrossSessionMemory: runControls.enableCrossSessionMemory,
      allowLiveSearch: runControls.allowLiveSearch,
      tone: runControls.tone || undefined,
      outputLanguage: runControls.outputLanguage || undefined,
      agentId: selectedAgentId ?? runContextDefaults?.agentId,
      turnIntent: 'research',
    };
  }, [
    availableSources,
    disabledSourceIds,
    runContextDefaults,
    runControls.allowLiveSearch,
    runControls.enableCrossSessionMemory,
    runControls.enablePlanReview,
    runControls.outputLanguage,
    runControls.researchDepth,
    runControls.sources,
    runControls.tone,
    runControls.verifySources,
    selectedAgentId,
    sourceControlsActive,
  ]);

  const sourceValidation = React.useMemo(() => {
    if (!sourceControlsActive) {
      return { isValid: true, message: null as string | null };
    }

    const sourceScope = deriveSourceScopeFromComposerSources(runControls.sources);
    if (sourceScope !== 'enterprise_only') {
      return { isValid: true, message: null as string | null };
    }

    const enabledMcpServerNames = deriveEnabledMcpServerNamesForSubmit(
      availableSources,
      runControls.sources,
    );
    const hasEnabledMcpServer =
      runControls.sources.mcp && enabledMcpServerNames.length > 0;
    if (runControls.sources.mcp && !hasEnabledMcpServer && !runControls.sources.ent) {
      return {
        isValid: false,
        message: 'Select an MCP server or turn on Web.',
      };
    }

    if (!hasEnabledMcpServer) {
      const enterpriseSources = availableSources.filter(isEnterpriseAvailableSource);
      const enabledEnterpriseSources = enterpriseSources.filter((source) => source.isEnabled);
      if (enterpriseSources.length === 0) {
        return {
          isValid: false,
          message: 'No enterprise data sources available. Select an MCP server or enable Web.',
        };
      }
      if (enabledEnterpriseSources.length === 0) {
        return {
          isValid: false,
          message:
            'All enterprise sources are disabled. Enable Enterprise, select an MCP server, or turn on Web.',
        };
      }
    }

    return { isValid: true, message: null as string | null };
  }, [availableSources, runControls.sources, sourceControlsActive]);

  const actionDisabled = runDisabled || !sourceValidation.isValid;

  const handleAction = React.useCallback(
    (action: string) => {
      const binding = surface.bindings.find((b) => b.action === action);
      if (!binding) return;
      if (!sourceValidation.isValid) {
        setInlineError(sourceValidation.message);
        setSections((prev) => ({
          ...prev,
          inputs: { ...prev.inputs, open: true },
        }));
        return;
      }
      // Tolerant compile: when the bound query field is empty, the query is
      // composed from the free-text inputs the user filled (see deriveEffectiveQuery).
      const compiled = compileSurfaceAction({
        surface,
        binding,
        dataModel: formDataModel,
        runContext: hostRunContext,
        selectedAgentId,
      });
      if (!compiled.query.trim()) {
        // Only when nothing runnable was provided anywhere on the form.
        setInlineError('Enter your request or fill at least one field to run.');
        setSections((prev) => ({
          ...prev,
          inputs: { ...prev.inputs, open: true },
        }));
        return;
      }
      setInlineError(null);
      setSections((prev) => ({
        inputs: prev.inputs.userSet
          ? prev.inputs
          : { ...prev.inputs, open: false },
        results: { ...prev.results, open: true },
      }));
      onRunAction(compiled);
    },
    [
      surface,
      sourceValidation.isValid,
      sourceValidation.message,
      formDataModel,
      hostRunContext,
      onRunAction,
      selectedAgentId,
    ],
  );

  const toggleSection = React.useCallback((section: 'inputs' | 'results') => {
    setSections((prev) => ({
      ...prev,
      [section]: {
        open: !prev[section].open,
        userSet: true,
      },
    }));
  }, []);

  const summary = React.useMemo(
    () => surfaceInputSummary(surface, formDataModel),
    [surface, formDataModel],
  );

  const actionBar =
    layout.actions === 'host_bar' ? (
      <div className="mt-3 flex flex-wrap items-center gap-2 border-t border-db-gray-lines pt-3">
        {surface.bindings.map((binding, index) => {
          const ref = runState[binding.action];
          const running = ref?.status === 'running';
          return (
            <button
              key={binding.action}
              type="button"
              data-testid={`surface-host-action-${binding.action}`}
              disabled={actionDisabled}
              title={sourceValidation.message ?? undefined}
              onClick={() => handleAction(binding.action)}
              className={
                index === 0
                  ? 'rounded-db-md bg-db-lava-600 px-3 py-1.5 text-[13px] font-medium text-white transition-colors hover:bg-db-lava-700 disabled:cursor-not-allowed disabled:opacity-50'
                  : 'rounded-db-md border border-db-gray-lines bg-white px-3 py-1.5 text-[13px] font-medium text-db-navy-800 transition-colors hover:bg-db-oat-light disabled:cursor-not-allowed disabled:opacity-50'
              }
            >
              {running ? 'Running...' : actionLabel(surface, binding.action)}
            </button>
          );
        })}
      </div>
    ) : null;

  const renderSection = (
    section: 'inputs' | 'results',
    title: string,
    rootIds: string[],
  ) => {
    const state = sections[section];
    return (
      <section className="rounded-db-md border border-db-gray-lines bg-white">
        <button
          type="button"
          data-testid={`surface-${section}-toggle`}
          aria-expanded={state.open}
          onClick={() => toggleSection(section)}
          className="flex w-full items-center justify-between gap-3 px-3 py-2 text-left"
        >
          <span className="min-w-0">
            <span className="block text-[13px] font-semibold text-db-navy-800">
              {title}
            </span>
            {section === 'inputs' && !state.open && (
              <span className="block truncate text-[11px] text-db-gray-text">{summary}</span>
            )}
          </span>
          <ChevronDown
            size={15}
            className={`shrink-0 text-db-gray-text transition-transform ${
              state.open ? 'rotate-180' : ''
            }`}
          />
        </button>
        {state.open && (
          <div className="border-t border-db-gray-lines p-3">
            {rootIds.length > 0 ? (
              <SurfaceRenderer
                surface={surface}
                dataModel={renderedDataModel}
                onDataModelChange={handleSetValue}
                onAction={handleAction}
                actionDisabled={actionDisabled}
                resolveRunReference={resolveRunReference}
                resolveCitations={resolveCitations}
                retryStructuring={retryStructuring}
                rootIds={rootIds}
                suppressComponentIds={suppressedComponentIds}
              />
            ) : (
              <p className="text-[12px] text-db-gray-text">
                {section === 'results'
                  ? 'Run the agent to see results here.'
                  : 'No input fields.'}
              </p>
            )}
            {section === 'inputs' && actionBar}
            {section === 'inputs' && inlineError && (
              <p className="mt-2 text-[12px] text-db-lava-700">{inlineError}</p>
            )}
          </div>
        )}
      </section>
    );
  };

  return (
    <div data-testid="agent-surface-panel" className="flex h-full flex-col">
      {/* Header */}
      <div className="flex shrink-0 items-center justify-between border-b border-db-gray-lines px-4 py-3">
        <div className="min-w-0">
          <p className="text-[10px] font-semibold uppercase tracking-wide text-db-gray-text">
            Agent UI
          </p>
          <p className="truncate text-[13px] font-semibold text-db-navy-800">{agentName}</p>
        </div>
        {onClose && (
          <button
            type="button"
            aria-label="Close Agent UI panel"
            onClick={onClose}
            className="ml-2 shrink-0 rounded p-1 text-db-gray-text hover:bg-db-gray-50 hover:text-db-navy-800"
          >
            <X size={14} />
          </button>
        )}
      </div>
      <SurfaceRunControlsBar
        value={runControls}
        onChange={setRunControls}
        discoveredSourceCount={availableSources.length}
        disabled={runDisabled}
        runtimeControls={runtimeControls}
        onBrowseSources={onBrowseSources}
        isDiscoveringSources={isDiscoveringSources}
      />
      {sourceValidation.message && (
        <div
          data-testid="surface-source-validation"
          className="border-b border-amber-200 bg-amber-50 px-4 py-1.5 text-[12px] text-amber-800"
        >
          {sourceValidation.message}
        </div>
      )}

      {/* Body */}
      <div className="flex-1 space-y-3 overflow-y-auto bg-db-gray-50 p-3">
        {renderSection('inputs', layout.inputs.title, layout.inputs.children)}
        {renderSection('results', layout.results.title, layout.results.children)}
      </div>

      {/* Footer hint */}
      {runDisabled && (
        <div className="shrink-0 border-t border-db-gray-lines px-4 py-2">
          <p className="text-[11px] text-db-gray-text">A run is already in progress</p>
        </div>
      )}
    </div>
  );
}

export default AgentSurfacePanel;
