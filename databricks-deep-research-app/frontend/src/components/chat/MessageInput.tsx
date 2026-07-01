import * as React from 'react';
import * as Popover from '@radix-ui/react-popover';
import {
  Zap,
  Microscope,
  Globe,
  Database,
  Boxes,
  ChevronDown,
  Check,
  Bot,
} from 'lucide-react';
import { Button } from '@/components/ui/button';
import { cn } from '@/lib/utils';
import { type ResearchDepth } from './ResearchDepthSelector';
import { ChatAttachmentsSelector } from './ChatAttachmentsSelector';
import { ChatOptionsPanel } from './ChatOptionsPanel';
import { SourceBrowserModal } from './SourceBrowserModal';
import { FileUploadZone } from '@/components/files/FileUploadZone';
import { UploadedFileList } from '@/components/files/UploadedFileList';
import { useQueryMode, useSourceScope } from '@/hooks';
import { useDiscoveredSources, useRefreshDiscovery } from '@/hooks/useDiscoveredSources';
import { useFileUpload } from '@/hooks/useFileUpload';
import { useAgentsV2List } from '@/hooks/useAgentsV2';
import { ComponentRegistry } from '@/core/plugins';
import type { AvailableSource } from '@/types/dataSources';
import type { CustomAgentSummary } from '@/types/customAgents';
import type { AgentV2Summary } from '@/types/agentDesigner';
import type { InputConfig } from '@/core/plugins/types';
import type { QuerySubmission } from '@/types/querySubmission';
import type { QueryMode } from '@/types';
import {
  deriveEnabledMcpServerNamesForSubmit,
  deriveEnabledSourceIdsForSubmit,
  deriveQueryModeFromComposerState,
  deriveSourceScopeFromComposerSources,
  isEnterpriseAvailableSource,
  type ComposerMode,
  type ComposerSources,
} from './sourceRouting';

interface MessageInputProps {
  onSubmit: (submission: QuerySubmission) => void;
  onStop?: () => void;
  isLoading?: boolean;
  placeholder?: string;
  disabled?: boolean;
  showModeSelector?: boolean;
  showDepthSelector?: boolean;
  /** Chat ID for scoping file uploads */
  sessionId?: string;
  /** Plugin-provided input configuration (overrides individual props) */
  inputConfig?: InputConfig;
}

const ENABLED_ENTERPRISE_SOURCES_KEY = 'deep-research-enabled-enterprise-sources';
const SELECTED_AGENT_KEY = 'deep-research-selected-agent';
// VariantA composer model (declutter redesign): a 2-mode picker (Answer/Deep)
// plus a Web/Enterprise/MCP source checkbox set. These persist independently of
// the legacy queryMode/sourceScope, which are *derived* from them at submit time
// so the backend contract is unchanged.
const COMPOSER_MODE_KEY = 'deep-research-composer-mode';
const COMPOSER_SOURCES_KEY = 'deep-research-composer-sources';

function readSelectedAgentId(): string | null {
  if (typeof window === 'undefined') return null;
  try {
    return localStorage.getItem(SELECTED_AGENT_KEY) || null;
  } catch {
    return null;
  }
}

function writeSelectedAgentId(agentId: string | null): void {
  if (typeof window === 'undefined') return;
  try {
    if (agentId) {
      localStorage.setItem(SELECTED_AGENT_KEY, agentId);
    } else {
      localStorage.removeItem(SELECTED_AGENT_KEY);
    }
  } catch {
    // Ignore localStorage errors
  }
}

function readEnabledEnterpriseSources(): Set<string> {
  if (typeof window === 'undefined') return new Set<string>();
  try {
    const raw = localStorage.getItem(ENABLED_ENTERPRISE_SOURCES_KEY);
    if (!raw) return new Set<string>();
    const parsed = JSON.parse(raw);
    if (!Array.isArray(parsed)) return new Set<string>();
    return new Set(parsed.filter((id): id is string => typeof id === 'string'));
  } catch {
    return new Set<string>();
  }
}

function writeEnabledEnterpriseSources(ids: Set<string>): void {
  if (typeof window === 'undefined') return;
  try {
    localStorage.setItem(ENABLED_ENTERPRISE_SOURCES_KEY, JSON.stringify(Array.from(ids)));
  } catch {
    // Ignore localStorage errors
  }
}

function readComposerMode(): ComposerMode | null {
  if (typeof window === 'undefined') return null;
  try {
    const raw = localStorage.getItem(COMPOSER_MODE_KEY);
    return raw === 'answer' || raw === 'deep' ? raw : null;
  } catch {
    return null;
  }
}

function writeComposerMode(mode: ComposerMode): void {
  if (typeof window === 'undefined') return;
  try {
    localStorage.setItem(COMPOSER_MODE_KEY, mode);
  } catch {
    // Ignore localStorage errors
  }
}

function readComposerSources(): ComposerSources | null {
  if (typeof window === 'undefined') return null;
  try {
    const raw = localStorage.getItem(COMPOSER_SOURCES_KEY);
    if (!raw) return null;
    const parsed = JSON.parse(raw);
    if (parsed && typeof parsed === 'object') {
      return {
        web: parsed.web !== false,
        ent: parsed.ent !== false,
        mcp: parsed.mcp !== false,
      };
    }
    return null;
  } catch {
    return null;
  }
}

function writeComposerSources(sources: ComposerSources): void {
  if (typeof window === 'undefined') return;
  try {
    localStorage.setItem(COMPOSER_SOURCES_KEY, JSON.stringify(sources));
  } catch {
    // Ignore localStorage errors
  }
}

function agentV2ToSelectorSummary(agent: AgentV2Summary): CustomAgentSummary {
  return {
    id: agent.id,
    name: agent.name,
    description: agent.description,
    avatarUrl: null,
    visibility: agent.visibility === 'workspace' ? 'workspace' : 'private',
    ownerId: agent.visibility === 'system' ? 'system' : agent.owner_id,
    inAppActive: agent.in_app_active,
    defaultVerifySources: agent.default_verify_sources,
  };
}

export function MessageInput({
  onSubmit,
  onStop,
  isLoading = false,
  placeholder = 'Ask a research question...',
  disabled = false,
  showModeSelector = true,
  showDepthSelector = true,
  sessionId,
  inputConfig,
}: MessageInputProps) {
  const [message, setMessage] = React.useState('');

  // Resolve effective configuration (inputConfig overrides props)
  const effectiveShowModeSelector = inputConfig?.showModeSelector ?? showModeSelector ?? true;
  const effectiveShowDepthSelector = inputConfig?.showDepthSelector ?? showDepthSelector ?? true;
  const effectiveShowVerifySources = inputConfig?.showVerifySources ?? true;
  const effectivePlaceholder = inputConfig?.placeholder ?? placeholder ?? 'Ask a research question...';

  // Persisted queryMode preference (localStorage + optional API sync). We no
  // longer read it as the primary control — the VariantA model below is the
  // source of truth — but we keep it in sync with the derived value so the
  // next session and any server-side preference reflect the effective mode.
  const { setMode: setStoredMode } = useQueryMode({
    initialMode: 'web_search',
    syncWithPreferences: effectiveShowModeSelector,
  });

  // VariantA composer model: primary mode (Answer/Deep) + a Web/Enterprise/MCP
  // source checkbox set. The legacy queryMode + sourceScope are DERIVED from
  // these so the submitted payload — and the rest of the app — is unchanged.
  const [composerMode, setComposerMode] = React.useState<ComposerMode>(
    () => readComposerMode() ?? 'deep',
  );
  const [composerSources, setComposerSources] = React.useState<ComposerSources>(
    () => readComposerSources() ?? { web: true, ent: true, mcp: true },
  );

  // Derive the legacy queryMode from (mode, sources):
  //   deep                    -> deep_research
  //   answer + any web/ent/mcp on -> web_search   (lightweight retrieval)
  //   answer + no sources    -> simple       (model only)
  // When the plugin hides the selector, honor its configured default instead.
  const derivedQueryMode: QueryMode = !effectiveShowModeSelector
    ? (inputConfig?.defaultQueryMode ?? 'deep_research')
    : deriveQueryModeFromComposerState(composerMode, composerSources);
  const queryMode = derivedQueryMode;

  // Derive the legacy SourceScope from source channels. MCP counts as non-web,
  // and the concrete MCP server selection is sent through enabledSources plus a
  // derived enabledMcpServers compatibility field.
  const derivedSourceScope = deriveSourceScopeFromComposerSources(composerSources);

  // Keep the persisted queryMode preference in sync with the derived value.
  React.useEffect(() => {
    if (effectiveShowModeSelector) setStoredMode(derivedQueryMode);
  }, [derivedQueryMode, effectiveShowModeSelector, setStoredMode]);

  // Persist the VariantA model so it survives reloads.
  React.useEffect(() => {
    writeComposerMode(composerMode);
  }, [composerMode]);
  React.useEffect(() => {
    writeComposerSources(composerSources);
  }, [composerSources]);

  // Show depth selector only when Deep Research mode is selected AND enabled
  const shouldShowDepthSelector = effectiveShowDepthSelector && queryMode === 'deep_research';
  // Show verify sources checkbox when web_search or deep_research AND enabled
  const shouldShowVerifyCheckbox = effectiveShowVerifySources && (queryMode === 'web_search' || queryMode === 'deep_research');
  // shouldShowSourceScope is computed later, after selectedAgent state is declared

  // Use plugin default for research depth when selector is hidden
  const [researchDepth, setResearchDepth] = React.useState<ResearchDepth>(
    inputConfig?.defaultResearchDepth ?? 'auto'
  );

  // Per-run report style. Empty string => omit from submission (server default).
  const [tone, setTone] = React.useState<string>('');
  const [outputLanguage, setOutputLanguage] = React.useState<string>('');

  // Default: use plugin config if selector hidden, else true for deep_research
  const [verifySources, setVerifySources] = React.useState<boolean>(
    !effectiveShowVerifySources
      ? (inputConfig?.defaultVerifySources ?? true)
      : false
  );

  // Deliverable (output type) selection. Sourced from the registered output
  // renderers (plugin-provided), so the dropdown lists exactly the structured
  // deliverables this deployment can produce. Internal/default renderers are
  // excluded; the selector only shows when there's an actual choice (>= 2).
  const deliverableOptions = React.useMemo(
    () =>
      ComponentRegistry.listOutputTypes()
        .filter((t) => !t.startsWith('__') && t !== 'synthesis_report')
        .map((t) => ({ value: t, label: ComponentRegistry.getRenderer(t)?.displayName ?? t })),
    []
  );
  const showDeliverableSelector = deliverableOptions.length >= 2;
  const [selectedOutputType, setSelectedOutputType] = React.useState<string | undefined>(
    () => inputConfig?.defaultOutputType
  );

  // State for source browser modal
  const [showSourceBrowser, setShowSourceBrowser] = React.useState(false);

  // File upload state
  const [showFileUpload, setShowFileUpload] = React.useState(false);
  const [showUploadZone, setShowUploadZone] = React.useState(true);
  const [hasActivatedFileTools, setHasActivatedFileTools] = React.useState(!!sessionId);
  const {
    files: sessionFiles,
    uploadFiles,
    isUploading,
    uploadProgress,
    deleteFile,
  } = useFileUpload(sessionId, {
    enabled: hasActivatedFileTools || !!sessionId,
  });
  const readyFiles = React.useMemo(
    () => sessionFiles.filter(f => f.processingStatus === 'ready'),
    [sessionFiles]
  );

  // Sync file tools activation when sessionId becomes available (e.g., draft → real chat)
  React.useEffect(() => {
    if (sessionId) {
      setHasActivatedFileTools(true);
    }
  }, [sessionId]);

  // Auto-collapse upload zone after files finish uploading
  const prevIsUploadingRef = React.useRef(isUploading);
  React.useEffect(() => {
    if (prevIsUploadingRef.current && !isUploading && sessionFiles.length > 0) {
      setShowUploadZone(false);
    }
    prevIsUploadingRef.current = isUploading;
  }, [isUploading, sessionFiles.length]);

  // Auto-close file pane when research completes
  const prevIsLoadingRef = React.useRef(isLoading);
  React.useEffect(() => {
    if (prevIsLoadingRef.current && !isLoading) {
      setShowFileUpload(false);
    }
    prevIsLoadingRef.current = isLoading;
  }, [isLoading]);

  // Agent selection state (with localStorage persistence — T013)
  const [selectedAgent, setSelectedAgent] = React.useState<CustomAgentSummary | null>(null);
  const [showAgentPicker, setShowAgentPicker] = React.useState(false);
  const agentPickerRef = React.useRef<HTMLDivElement>(null);

  // Dismiss agent picker on Escape key or click outside
  React.useEffect(() => {
    if (!showAgentPicker) return;

    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        setShowAgentPicker(false);
      }
    };

    const handleClickOutside = (e: MouseEvent) => {
      if (agentPickerRef.current && !agentPickerRef.current.contains(e.target as Node)) {
        setShowAgentPicker(false);
      }
    };

    document.addEventListener('keydown', handleKeyDown);
    document.addEventListener('mousedown', handleClickOutside);

    return () => {
      document.removeEventListener('keydown', handleKeyDown);
      document.removeEventListener('mousedown', handleClickOutside);
    };
  }, [showAgentPicker]);
  const {
    data: agentsData,
    isLoading: isLoadingAgents,
    isFetching: isFetchingAgents,
    isError: isAgentsError,
    refetch: refetchAgents,
  } = useAgentsV2List();
  const agents = React.useMemo(
    // Agent picker is available in EVERY mode so the run target is always
    // visible/controllable (selecting an agent makes it own the pipeline — see
    // the "Running:" indicator). Previously gated to deep_research, which hid it
    // in Simple/Web and left users unsure what actually executed.
    () => (agentsData?.items ?? []).map(agentV2ToSelectorSummary),
    [agentsData?.items],
  );

  // Restore selected agent from localStorage on mount / when agents load
  React.useEffect(() => {
    if (agents.length === 0) return;
    const savedId = readSelectedAgentId();
    if (savedId && !selectedAgent) {
      const found = agents.find((a) => a.id === savedId);
      if (found) setSelectedAgent(found);
    }
  }, [agents]); // eslint-disable-line react-hooks/exhaustive-deps

  // Persist agent selection to localStorage on change
  const handleAgentSelect = React.useCallback(
    (agent: CustomAgentSummary | null) => {
      setSelectedAgent(agent);
      writeSelectedAgentId(agent?.id ?? null);
    },
    []
  );

  // Show source scope selector when deep_research or web_search is selected (not for simple mode)
  // Hide when a custom agent defines its own source configuration (009-custom-agent-config T030)
  const agentDefinesSources = selectedAgent?.hasSourceConfig === true;
  const shouldShowSourceScope =
    effectiveShowModeSelector &&
    (queryMode === 'deep_research' || queryMode === 'web_search') && !agentDefinesSources;

  // Plan review toggle state
  const [enablePlanReview, setEnablePlanReview] = React.useState(false);
  // Per-turn routing for custom-agent chats (auto-detect by default). Only sent
  // when an agent is selected; the backend ignores it on first turns (no prior
  // research) and for non-agent chats.
  const [turnIntent, setTurnIntent] = React.useState<'auto' | 'chat' | 'research'>('auto');

  // Chat-attached skills for THIS query (Feature 2.2 — E1). MCP is selected in
  // the unified Sources browser as mcp_server sources.
  const [enabledSkills, setEnabledSkills] = React.useState<string[]>([]);

  // Run-level overrides surfaced in the Options panel (P2). Default false =>
  // inherit the global flag; the user opts in per query.
  const [enableCrossSessionMemory, setEnableCrossSessionMemory] = React.useState(false);
  const [allowLiveSearch, setAllowLiveSearch] = React.useState(false);

  // Fetch discovered data sources for the source toggle UI
  const {
    data: discoveryData,
    isLoading: isDiscoveryLoading,
    error: discoveryError,
    refetch: refetchDiscovery,
  } = useDiscoveredSources({
    enabled: shouldShowSourceScope || showSourceBrowser,
  });
  const refreshDiscoveryMutation = useRefreshDiscovery();

  // Extract valid source IDs from discovered sources for stale ID filtering
  const validSourceIds = React.useMemo(() => {
    if (!discoveryData?.sources) return undefined;
    return discoveryData.sources
      .filter((s) => s.status === 'ready')
      .map((s) => s.source_id);
  }, [discoveryData?.sources]);

  // Stable reference for passing to child components (avoids new [] on each render)
  const discoveredSources = React.useMemo(
    () => discoveryData?.sources ?? [],
    [discoveryData?.sources]
  );

  // Per-source disabled list + Browse-modal selection (T050-T051). The high-level
  // scope (web/enterprise/all) is now derived from the VariantA source checkboxes
  // (`derivedSourceScope`), so we only consume the per-source bits here.
  const {
    disabledSources,
    setDisabledSources,
  } = useSourceScope({ validSourceIds });

  // Convert discovered sources to AvailableSource format (used for submit
  // payload, enterprise validation, and the Browse modal selection)
  const availableSources: AvailableSource[] = React.useMemo(() => {
    if (!discoveryData?.sources) return [];
    return discoveryData.sources
      .filter((s) => s.status === 'ready')
      .map((source) => ({
        id: source.source_id,
        name: source.name,
        type: source.source_type,
        description: source.description ?? null,
        isEnabled: !disabledSources.includes(source.source_id),
      }));
  }, [discoveryData?.sources, disabledSources]);

  const textareaRef = React.useRef<HTMLTextAreaElement>(null);

  // Keep modal selection state derived from the persisted disabled source list.
  const selectedSourceIds = React.useMemo(
    () => availableSources.filter((source) => source.isEnabled).map((source) => source.id),
    [availableSources]
  );

  // Enterprise sources are opt-in: disabled by default unless explicitly enabled.
  React.useEffect(() => {
    if (!discoveryData?.sources) return;

    const readySelectableNonWebSourceIds = discoveryData.sources
      .filter(
        (source) =>
          source.status === 'ready' &&
          source.source_type !== 'web_search' &&
          source.source_type !== 'uploaded_file'
      )
      .map((source) => source.source_id);

    if (readySelectableNonWebSourceIds.length === 0) return;

    const enabledSet = readEnabledEnterpriseSources();
    const shouldDisable = readySelectableNonWebSourceIds.filter((id) => !enabledSet.has(id));

    if (shouldDisable.length === 0) return;

    // Merge with existing disabledSources (avoid duplicates)
    const nextDisabled = Array.from(new Set([...disabledSources, ...shouldDisable]));
    if (nextDisabled.length !== disabledSources.length) {
      setDisabledSources(nextDisabled);
    }
  }, [discoveryData?.sources, disabledSources, setDisabledSources]);

  // Reset verifySources when query mode changes — only when the selector is
  // visible AND no custom agent is selected. For a selected agent the seeding
  // effect below owns the default, so this never clobbers a seeded/edited value.
  React.useEffect(() => {
    if (effectiveShowVerifySources && !selectedAgent) {
      setVerifySources(false);
    }
  }, [queryMode, effectiveShowVerifySources, selectedAgent]);

  // Seed the verify toggle from the selected agent's authored default
  // (reclaim => on; classical_lite/none => off). Re-seeds only when the selected
  // agent changes (keyed on id), so a user's later toggle change overrides and
  // persists for that agent ("toggle wins"). Absent field => true (safe floor).
  React.useEffect(() => {
    if (selectedAgent) {
      setVerifySources(selectedAgent.defaultVerifySources ?? true);
    }
  }, [selectedAgent?.id]); // eslint-disable-line react-hooks/exhaustive-deps

  // T053-T054: Validation - Check if submission should be blocked due to source scope issues
  const enterpriseSources = React.useMemo(() => {
    return availableSources.filter(isEnterpriseAvailableSource);
  }, [availableSources]);

  const enabledEnterpriseSources = React.useMemo(() => {
    return enterpriseSources.filter((s) => s.isEnabled);
  }, [enterpriseSources]);

  const enabledMcpServerNames = React.useMemo(
    () => deriveEnabledMcpServerNamesForSubmit(availableSources, composerSources),
    [availableSources, composerSources],
  );

  const hasEnabledMcpServer = composerSources.mcp && enabledMcpServerNames.length > 0;

  // Determine if we should block submission due to source configuration
  const sourceValidation = React.useMemo(() => {
    if (!shouldShowSourceScope) {
      return { isValid: true, message: null };
    }

    if (derivedSourceScope === 'enterprise_only') {
      if (composerSources.mcp && !hasEnabledMcpServer && !composerSources.ent) {
        return {
          isValid: false,
          message: 'Select an MCP server or turn on Web.',
        };
      }

      if (!hasEnabledMcpServer) {
        // T053: Block when Enterprise Only but no enterprise sources discovered
        if (enterpriseSources.length === 0) {
          return {
            isValid: false,
            message: 'No enterprise data sources available. Select an MCP server or enable Web.',
          };
        }
        // T054: Block when Enterprise Only but all enterprise sources are disabled
        if (enabledEnterpriseSources.length === 0) {
          return {
            isValid: false,
            message: 'All enterprise sources are disabled. Enable Enterprise, select an MCP server, or turn on Web.',
          };
        }
      }
    }

    return { isValid: true, message: null };
  }, [
    shouldShowSourceScope,
    derivedSourceScope,
    composerSources.mcp,
    composerSources.ent,
    hasEnabledMcpServer,
    enterpriseSources.length,
    enabledEnterpriseSources.length,
  ]);

  // Check if any files are still processing (not ready yet)
  const hasProcessingFiles = React.useMemo(() => {
    return sessionFiles.some(
      f => f.processingStatus === 'pending' || f.processingStatus === 'processing'
    );
  }, [sessionFiles]);

  // Combined disabled state for submit button and keyboard submit.
  const isSubmitDisabled =
    !message.trim() || isLoading || disabled || !sourceValidation.isValid || hasProcessingFiles;

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!isSubmitDisabled) {
      // Compute enabledSources from availableSources minus disabledSources
      const enabledSourceIds = shouldShowSourceScope && availableSources.length > 0
        ? deriveEnabledSourceIdsForSubmit(availableSources, composerSources)
        : undefined;

      // Only pass disabledSources if there are any disabled
      const disabledSourceIds = shouldShowSourceScope && disabledSources.length > 0
        ? disabledSources
        : undefined;

      const submission: QuerySubmission = {
        message: message.trim(),
        queryMode,
        researchDepth,
        verifySources,
        outputType: selectedOutputType ?? inputConfig?.defaultOutputType,
        sourceScope: shouldShowSourceScope ? derivedSourceScope : undefined,
        enabledSources: enabledSourceIds,
        disabledSources: disabledSourceIds,
        fileIds: readyFiles.length > 0 ? readyFiles.map(f => f.id) : undefined,
        agentId: selectedAgent?.id ?? undefined,
        enablePlanReview: enablePlanReview || undefined,
        turnIntent: selectedAgent ? turnIntent : undefined,
        tone: tone || undefined,
        outputLanguage: outputLanguage || undefined,
        enabledSkills: enabledSkills.length > 0 ? enabledSkills : undefined,
        // Compatibility field for the backend MCP attachment path. The source
        // browser is now the source of truth via enabledSources mcp:* IDs.
        enabledMcpServers:
          enabledMcpServerNames.length > 0
            ? enabledMcpServerNames
            : undefined,
        enableCrossSessionMemory: enableCrossSessionMemory || undefined,
        allowLiveSearch: allowLiveSearch || undefined,
      };

      onSubmit(submission);
      setMessage('');
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit(e);
    }
  };

  // Auto-resize textarea
  React.useEffect(() => {
    const textarea = textareaRef.current;
    if (textarea) {
      textarea.style.height = 'auto';
      textarea.style.height = `${Math.min(textarea.scrollHeight, 200)}px`;
    }
  }, [message]);

  return (
    <form onSubmit={handleSubmit} className="db-root border-t border-db-gray-lines bg-white">
      {/* VariantA grouped toolbar — primary mode + contextual scope on the left,
          run target + advanced on the right; everything else behind the gear. */}
      <div className="flex flex-wrap items-center gap-2 px-4 pt-2.5">
        {/* Primary mode: Answer / Deep Research */}
        {effectiveShowModeSelector && (
          <ComposerSegmented
            mode={composerMode}
            onChange={setComposerMode}
            disabled={disabled || isLoading}
          />
        )}

        {/* Contextual scope: Depth (built-in Deep only) + Sources (Web/Ent/MCP) */}
        {effectiveShowModeSelector &&
          ((shouldShowDepthSelector && !selectedAgent) || !agentDefinesSources) && (
            <span className="mx-0.5 h-5 w-px shrink-0 bg-db-gray-lines" aria-hidden="true" />
          )}
        {/* Depth applies only to the built-in Deep Research pipeline. A selected
            custom agent runs its own saved workflow, so depth is moot — hide it. */}
        {shouldShowDepthSelector && !selectedAgent && (
          <DepthChip
            value={researchDepth}
            onChange={setResearchDepth}
            disabled={disabled || isLoading}
          />
        )}
        {/* Effort: a selected CUSTOM agent runs its own saved workflow, so the
            built-in Depth control is hidden — this scales that agent's researcher
            tool budgets + loop iterations for THIS turn. Auto = the agent's saved
            Research-depth default (set in the Designer). */}
        {selectedAgent && (
          <>
            <span
              className="mx-0.5 h-5 w-px shrink-0 bg-db-gray-lines"
              aria-hidden="true"
            />
            <EffortChip
              value={researchDepth}
              onChange={setResearchDepth}
              disabled={disabled || isLoading}
            />
          </>
        )}
        {/* Sources is the control that decides Answer→Simple vs Web; show it in
            both modes (unless the selected agent defines its own sources). */}
        {effectiveShowModeSelector && !agentDefinesSources && (
          <SourcesChip
            sources={composerSources}
            onChange={setComposerSources}
            onBrowse={() => setShowSourceBrowser(true)}
            browseCount={selectedSourceIds.length}
            isDiscovering={isDiscoveryLoading}
            disabled={disabled || isLoading}
          />
        )}

        <span className="flex-1" />

        {/* Deliverable selector — choose the structured output type when >1 exists */}
        {showDeliverableSelector && (
          <label className="flex select-none items-center gap-1.5 text-[12px] text-db-gray-text">
            <span>Deliverable</span>
            <select
              data-testid="deliverable-selector"
              value={selectedOutputType ?? deliverableOptions[0]?.value ?? ''}
              onChange={(e) => setSelectedOutputType(e.target.value)}
              disabled={disabled || isLoading}
              className="cursor-pointer rounded-db-md border border-db-gray-lines bg-white px-1.5 py-1 text-[12px] text-db-navy-800"
            >
              {deliverableOptions.map((o) => (
                <option key={o.value} value={o.value}>
                  {o.label}
                </option>
              ))}
            </select>
          </label>
        )}

        {/* Agent chip — shown in every mode so the run target is always
            visible and selectable. */}
        {effectiveShowModeSelector && (
          <div className="relative" ref={agentPickerRef}>
            <button
              type="button"
              data-testid="agent-selector-trigger"
              onClick={() => {
                if (!showAgentPicker) {
                  void refetchAgents();
                }
                setShowAgentPicker(!showAgentPicker);
              }}
              disabled={disabled || isLoading}
              className={cn(
                'inline-flex items-center gap-1.5 rounded-db-md border px-2.5 py-1.5 text-[12.5px] font-medium transition-colors',
                selectedAgent
                  ? 'border-db-navy-300 bg-db-oat-medium text-db-navy-800'
                  : 'border-db-gray-lines bg-white text-db-navy-800 hover:bg-db-oat-light',
                (disabled || isLoading) && 'cursor-not-allowed opacity-50',
              )}
            >
              <Bot size={14} className="text-db-gray-text" />
              <span data-testid="agent-selected-name" className="max-w-[120px] truncate">
                {selectedAgent?.name ?? 'Default Agent'}
              </span>
              {selectedAgent && (
                <>
                  <span data-testid="agent-selected-badge" className="sr-only">selected</span>
                  <button
                    type="button"
                    data-testid="agent-clear-button"
                    onClick={(e) => {
                      e.stopPropagation();
                      handleAgentSelect(null);
                    }}
                    className="ml-0.5 hover:text-db-lava-700"
                    title="Clear agent selection"
                  >
                    ×
                  </button>
                </>
              )}
            </button>
            {selectedAgent && selectedAgent.sourceScope && selectedAgent.sourceScope !== 'all' && (
              <span
                data-testid="agent-source-scope-indicator"
                className="ml-1 text-[11px] text-db-gray-text"
              >
                {selectedAgent.sourceScope}
              </span>
            )}
            {showAgentPicker && (
              <AgentPickerDropdown
                agents={agents}
                isLoading={isLoadingAgents || isFetchingAgents}
                isError={isAgentsError}
                selectedAgent={selectedAgent}
                onSelect={(agent) => {
                  handleAgentSelect(agent);
                  setShowAgentPicker(false);
                }}
              />
            )}
          </div>
        )}

        {/* Follow-up routing (custom-agent chats): converse vs re-run. */}
        {effectiveShowModeSelector && selectedAgent && (
          <label className="flex select-none items-center gap-1.5 text-[12px] text-db-gray-text">
            <span>Follow-up</span>
            <select
              data-testid="turn-intent-select"
              value={turnIntent}
              onChange={(e) => setTurnIntent(e.target.value as 'auto' | 'chat' | 'research')}
              disabled={disabled || isLoading}
              className="rounded-db-md border border-db-gray-lines bg-white px-1.5 py-1 text-[12px] text-db-navy-800"
              title="How to handle this message: auto-detect, chat about already-gathered data, or re-run the agent"
            >
              <option value="auto">Auto</option>
              <option value="chat">Chat about results</option>
              <option value="research">Re-run research</option>
            </select>
          </label>
        )}

        {/* Skills attachments for this query */}
        <ChatAttachmentsSelector
          selectedSkills={enabledSkills}
          onChange={setEnabledSkills}
          disabled={disabled || isLoading}
        />

        {/* Attach files */}
        {sessionId && (
          <button
            type="button"
            onClick={() => {
              setHasActivatedFileTools(true);
              const next = !showFileUpload;
              setShowFileUpload(next);
              if (next) setShowUploadZone(sessionFiles.length === 0);
            }}
            disabled={disabled || isLoading}
            className={cn(
              'inline-flex items-center gap-1 rounded-db-md border px-2 py-1.5 text-[12px] transition-colors',
              showFileUpload
                ? 'border-db-navy-300 bg-db-oat-medium text-db-navy-800'
                : 'border-db-gray-lines bg-white text-db-navy-800 hover:bg-db-oat-light',
              (disabled || isLoading) && 'cursor-not-allowed opacity-50',
            )}
            title="Attach files"
          >
            <PaperclipIcon className="h-3.5 w-3.5" />
            {sessionFiles.length > 0 && (
              <span className="rounded-full bg-db-navy-800 px-1.5 py-0.5 text-[10px] font-medium leading-none text-white">
                {sessionFiles.length}
              </span>
            )}
          </button>
        )}

        {/* Advanced (report style + this-run options) behind one gear */}
        {effectiveShowModeSelector && queryMode !== 'simple' && (
          <ChatOptionsPanel
            tone={tone}
            outputLanguage={outputLanguage}
            onToneChange={setTone}
            onLanguageChange={setOutputLanguage}
            showVerify={shouldShowVerifyCheckbox}
            verifySources={verifySources}
            onVerifyChange={setVerifySources}
            showPlanReview={queryMode === 'deep_research' && effectiveShowModeSelector}
            enablePlanReview={enablePlanReview}
            onPlanReviewChange={setEnablePlanReview}
            enableCrossSessionMemory={enableCrossSessionMemory}
            onCrossSessionMemoryChange={setEnableCrossSessionMemory}
            allowLiveSearch={allowLiveSearch}
            onAllowLiveSearchChange={setAllowLiveSearch}
            disabled={disabled || isLoading}
          />
        )}
      </div>

      {/* Run target hint — what actually runs (kept for clarity + tests) */}
      {effectiveShowModeSelector && (
        <div className="px-4 pt-1.5">
          <span
            data-testid="run-target-indicator"
            className="text-[11px] text-db-gray-text"
            title={
              selectedAgent
                ? 'A selected agent runs its own saved workflow; the mode/sources do not change it.'
                : 'No agent selected — the built-in pipeline for the current mode runs.'
            }
          >
            Running:{' '}
            <span className="font-medium text-db-navy-800">
              {selectedAgent
                ? selectedAgent.name
                : queryMode === 'simple'
                  ? 'Answer · model only'
                  : queryMode === 'web_search'
                    ? 'Answer · with sources'
                    : 'Deep Research'}
            </span>
          </span>
        </div>
      )}
      {/* Source validation warning (T053-T054) */}
      {!sourceValidation.isValid && sourceValidation.message && (
        <div className="px-4 py-1">
          <p className="text-xs text-amber-600 dark:text-amber-500">
            {sourceValidation.message}
          </p>
        </div>
      )}
      {/* File processing indicator */}
      {hasProcessingFiles && (
        <div className="px-4 py-1">
          <p className="text-xs text-blue-600 dark:text-blue-400 animate-pulse">
            Processing files... Please wait before sending.
          </p>
        </div>
      )}
      {/* File upload pane (collapsible) */}
      {showFileUpload && sessionId && (
        <div className="px-4 pt-2">
          {/* Header row */}
          <div className="flex items-center justify-between mb-1.5">
            <div className="flex items-center gap-2">
              {sessionFiles.length > 0 && (
                <span className="text-xs font-medium text-muted-foreground">
                  {sessionFiles.length} file{sessionFiles.length !== 1 ? 's' : ''} attached
                </span>
              )}
              {!showUploadZone && sessionFiles.length > 0 && (
                <button
                  type="button"
                  onClick={() => setShowUploadZone(true)}
                  className="text-xs text-primary hover:underline"
                >
                  Add more
                </button>
              )}
            </div>
            <button
              type="button"
              onClick={() => setShowFileUpload(false)}
              className="rounded-sm p-0.5 text-muted-foreground hover:text-foreground hover:bg-accent transition-colors"
              aria-label="Close file pane"
            >
              <XCloseIcon className="h-4 w-4" />
            </button>
          </div>
          {/* Upload zone (shown when toggled) */}
          {showUploadZone && (
            <FileUploadZone
              onFilesSelected={(files) => uploadFiles(files)}
              isUploading={isUploading}
              uploadProgress={uploadProgress}
              disabled={disabled || isLoading}
              onClose={() => setShowUploadZone(false)}
            />
          )}
          {/* Compact file list (always shown when files exist) */}
          {sessionFiles.length > 0 && (
            <UploadedFileList
              files={sessionFiles}
              onDelete={(file) => deleteFile(file.id)}
              compact={true}
              className="mt-1.5"
            />
          )}
        </div>
      )}
      <div className="flex items-end gap-2 px-4 pb-4 pt-2.5">
        <textarea
          data-testid="message-input"
          ref={textareaRef}
          value={message}
          onChange={(e) => setMessage(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder={effectivePlaceholder}
          disabled={disabled || isLoading}
          rows={1}
          aria-label="Message input"
          className={cn(
            'flex-1 resize-none rounded-db-md border border-db-gray-lines bg-white px-3 py-2 text-[14px] text-db-navy-800 shadow-db-xs transition-colors',
            'placeholder:text-db-navy-300 focus:border-db-navy-400 focus:shadow-db-focus focus:outline-none',
            'disabled:cursor-not-allowed disabled:opacity-50',
            'min-h-[40px] max-h-[200px]'
          )}
        />
        {isLoading && onStop ? (
          <Button
            data-testid="stop-button"
            type="button"
            variant="outline"
            onClick={onStop}
            className="self-end"
          >
            Stop
          </Button>
        ) : (
          <Button
            data-testid="send-button"
            type="submit"
            disabled={isSubmitDisabled}
            className="self-end bg-db-lava-600 text-white hover:bg-db-lava-700"
            title={sourceValidation.message ?? undefined}
          >
            Send
          </Button>
        )}
      </div>

      {/* Source Browser Modal */}
      <SourceBrowserModal
        isOpen={showSourceBrowser}
        onClose={() => setShowSourceBrowser(false)}
        initialSelectedIds={selectedSourceIds}
        onApply={(ids) => {
          const allIds = availableSources.map((s) => s.id);
          const nextDisabled = allIds.filter((id) => !ids.includes(id));
          setDisabledSources(nextDisabled);

          // Update persisted non-web source tracking. Databricks enterprise
          // sources and MCP servers are opt-in, while web stays controlled by
          // the high-level Web channel.
          const enabledSet = new Set<string>();
          for (const id of ids) {
            const source = discoveredSources.find((s) => s.source_id === id);
            if (source && source.source_type !== 'web_search' && source.source_type !== 'uploaded_file') {
              enabledSet.add(id);
            }
          }
          writeEnabledEnterpriseSources(enabledSet);
        }}
        sources={discoveredSources}
        isDiscoveryLoading={isDiscoveryLoading}
        discoveryError={discoveryError ?? null}
        onRefetch={() => refetchDiscovery()}
        onRefresh={() => refreshDiscoveryMutation.mutate({})}
        isRefreshing={refreshDiscoveryMutation.isPending}
      />
    </form>
  );
}

// =============================================================================
// VariantA composer primitives (Databricks-styled — declutter redesign)
// =============================================================================

/** Answer / Deep Research segmented control. */
function ComposerSegmented({
  mode,
  onChange,
  disabled,
}: {
  mode: ComposerMode;
  onChange: (m: ComposerMode) => void;
  disabled?: boolean;
}) {
  const opts: { id: ComposerMode; label: string; icon: typeof Zap }[] = [
    { id: 'answer', label: 'Answer', icon: Zap },
    { id: 'deep', label: 'Deep Research', icon: Microscope },
  ];
  return (
    <div
      role="radiogroup"
      aria-label="Research mode"
      className="inline-flex items-center gap-0.5 rounded-db-lg border border-db-gray-lines bg-db-oat-medium p-0.5"
    >
      {opts.map((o) => {
        const active = o.id === mode;
        const Ico = o.icon;
        return (
          <button
            key={o.id}
            type="button"
            role="radio"
            aria-checked={active}
            data-testid={`composer-mode-${o.id}`}
            disabled={disabled}
            onClick={() => onChange(o.id)}
            className={cn(
              'inline-flex items-center gap-1.5 rounded-db-md px-3 py-1.5 text-[12.5px] font-medium transition-colors',
              active
                ? 'bg-white text-db-navy-800 shadow-db-xs'
                : 'text-db-gray-text hover:text-db-navy-800',
              disabled && 'cursor-not-allowed opacity-50',
            )}
          >
            <Ico size={14} className={active ? 'text-db-lava-600' : 'text-db-gray-text'} />
            {o.label}
          </button>
        );
      })}
    </div>
  );
}

/** Shared chip trigger style for the Depth/Sources popover buttons. */
function composerChipClass(active: boolean): string {
  return cn(
    'inline-flex items-center gap-1.5 rounded-db-md border px-2.5 py-1.5 text-[12.5px] font-medium transition-colors',
    active
      ? 'border-db-navy-300 bg-db-oat-medium text-db-navy-800'
      : 'border-db-gray-lines bg-white text-db-navy-800 hover:bg-db-oat-light',
  );
}

const DEPTH_CHIP_OPTIONS: { value: ResearchDepth; label: string }[] = [
  { value: 'auto', label: 'Auto' },
  { value: 'light', label: 'Light' },
  { value: 'medium', label: 'Medium' },
  { value: 'extended', label: 'Extended' },
];

/** Depth chip + popover (Auto / Light / Medium / Extended). */
function DepthChip({
  value,
  onChange,
  disabled,
}: {
  value: ResearchDepth;
  onChange: (d: ResearchDepth) => void;
  disabled?: boolean;
}) {
  const [open, setOpen] = React.useState(false);
  const label = DEPTH_CHIP_OPTIONS.find((o) => o.value === value)?.label ?? 'Auto';
  return (
    <Popover.Root open={open} onOpenChange={setOpen}>
      <Popover.Trigger asChild>
        <button
          type="button"
          disabled={disabled}
          data-testid="composer-depth-chip"
          className={composerChipClass(open)}
        >
          <Zap size={14} className="text-db-gray-text" />
          <span className="text-db-gray-text">Depth</span>
          <span className="font-semibold text-db-navy-800">{label}</span>
          <ChevronDown size={13} className="text-db-navy-400" />
        </button>
      </Popover.Trigger>
      <Popover.Portal>
        <Popover.Content
          side="top"
          align="start"
          sideOffset={6}
          collisionPadding={8}
          className="z-50 w-44 rounded-db-md border border-db-gray-lines bg-white p-1 shadow-lg"
        >
          {DEPTH_CHIP_OPTIONS.map((o) => {
            const active = o.value === value;
            return (
              <button
                key={o.value}
                type="button"
                onClick={() => {
                  onChange(o.value);
                  setOpen(false);
                }}
                className={cn(
                  'flex w-full items-center gap-2 rounded-db-md px-2 py-1.5 text-left text-[13px] transition-colors',
                  active
                    ? 'bg-db-oat-light font-medium text-db-navy-800'
                    : 'text-db-navy-800 hover:bg-db-oat-light',
                )}
              >
                {active ? (
                  <Check size={13} className="text-db-lava-600" />
                ) : (
                  <span className="w-[13px]" />
                )}
                {o.label}
              </button>
            );
          })}
        </Popover.Content>
      </Popover.Portal>
    </Popover.Root>
  );
}

// Effort chip for a selected CUSTOM agent. Reuses the ResearchDepth values the
// backend already understands, relabelled to the effort vocabulary: 'medium' is
// the no-op midpoint (Standard), 'extended' is Deep, and 'auto' defers to the
// agent's saved research_effort default.
const EFFORT_CHIP_OPTIONS: { value: ResearchDepth; label: string }[] = [
  { value: 'auto', label: 'Auto' },
  { value: 'light', label: 'Light' },
  { value: 'medium', label: 'Standard' },
  { value: 'extended', label: 'Deep' },
];

/** Effort chip + popover (Auto / Light / Standard / Deep) for custom agents. */
function EffortChip({
  value,
  onChange,
  disabled,
}: {
  value: ResearchDepth;
  onChange: (d: ResearchDepth) => void;
  disabled?: boolean;
}) {
  const [open, setOpen] = React.useState(false);
  const label = EFFORT_CHIP_OPTIONS.find((o) => o.value === value)?.label ?? 'Auto';
  return (
    <Popover.Root open={open} onOpenChange={setOpen}>
      <Popover.Trigger asChild>
        <button
          type="button"
          disabled={disabled}
          data-testid="composer-effort-chip"
          className={composerChipClass(open)}
        >
          <Zap size={14} className="text-db-gray-text" />
          <span className="text-db-gray-text">Effort</span>
          <span className="font-semibold text-db-navy-800">{label}</span>
          <ChevronDown size={13} className="text-db-navy-400" />
        </button>
      </Popover.Trigger>
      <Popover.Portal>
        <Popover.Content
          side="top"
          align="start"
          sideOffset={6}
          collisionPadding={8}
          className="z-50 w-44 rounded-db-md border border-db-gray-lines bg-white p-1 shadow-lg"
        >
          {EFFORT_CHIP_OPTIONS.map((o) => {
            const active = o.value === value;
            return (
              <button
                key={o.value}
                type="button"
                onClick={() => {
                  onChange(o.value);
                  setOpen(false);
                }}
                className={cn(
                  'flex w-full items-center gap-2 rounded-db-md px-2 py-1.5 text-left text-[13px] transition-colors',
                  active
                    ? 'bg-db-oat-light font-medium text-db-navy-800'
                    : 'text-db-navy-800 hover:bg-db-oat-light',
                )}
              >
                {active ? (
                  <Check size={13} className="text-db-lava-600" />
                ) : (
                  <span className="w-[13px]" />
                )}
                {o.label}
              </button>
            );
          })}
        </Popover.Content>
      </Popover.Portal>
    </Popover.Root>
  );
}

const SOURCE_CHANNELS: { id: keyof ComposerSources; label: string; icon: typeof Globe }[] = [
  { id: 'web', label: 'Web', icon: Globe },
  { id: 'ent', label: 'Enterprise', icon: Database },
  { id: 'mcp', label: 'MCP', icon: Boxes },
];

function sourcesSummary(s: ComposerSources): string {
  const on = SOURCE_CHANNELS.filter((c) => s[c.id]);
  if (on.length === 0) return 'None';
  if (on.length === SOURCE_CHANNELS.length) return 'All';
  return on.map((c) => c.label).join(', ');
}

/** Sources chip + popover — Web / Enterprise / MCP channels + Browse. */
function SourcesChip({
  sources,
  onChange,
  onBrowse,
  browseCount,
  isDiscovering,
  disabled,
}: {
  sources: ComposerSources;
  onChange: (s: ComposerSources) => void;
  onBrowse: () => void;
  browseCount: number;
  isDiscovering?: boolean;
  disabled?: boolean;
}) {
  const [open, setOpen] = React.useState(false);
  const summary = sourcesSummary(sources);
  const toggle = (id: keyof ComposerSources) =>
    onChange({ ...sources, [id]: !sources[id] });
  return (
    <Popover.Root open={open} onOpenChange={setOpen}>
      <Popover.Trigger asChild>
        <button
          type="button"
          disabled={disabled}
          data-testid="composer-sources-chip"
          className={composerChipClass(open)}
        >
          <Database size={14} className="text-db-gray-text" />
          <span className="text-db-gray-text">Sources</span>
          <span className="font-semibold text-db-navy-800">{summary}</span>
          <ChevronDown size={13} className="text-db-navy-400" />
        </button>
      </Popover.Trigger>
      <Popover.Portal>
        <Popover.Content
          side="top"
          align="start"
          sideOffset={6}
          collisionPadding={8}
          className="z-50 w-60 rounded-db-md border border-db-gray-lines bg-white p-2 shadow-lg"
        >
          {SOURCE_CHANNELS.map((c) => {
            const on = sources[c.id];
            const Ico = c.icon;
            return (
              <button
                key={c.id}
                type="button"
                data-testid={`composer-source-${c.id}`}
                aria-pressed={on}
                onClick={() => toggle(c.id)}
                className="flex w-full items-center gap-2.5 rounded-db-md px-2 py-1.5 text-left transition-colors hover:bg-db-oat-light"
              >
                <span
                  className={cn(
                    'flex h-[18px] w-[18px] shrink-0 items-center justify-center rounded-[5px] border-[1.5px] transition-colors',
                    on ? 'border-db-lava-600 bg-db-lava-600' : 'border-db-navy-300 bg-white',
                  )}
                >
                  {on && <Check size={11} className="text-white" strokeWidth={3} />}
                </span>
                <Ico size={15} className="text-db-gray-text" />
                <span className="text-[13px] font-medium text-db-navy-800">{c.label}</span>
              </button>
            );
          })}
          {summary === 'None' && (
            <p className="px-2 pt-1.5 text-[11px] italic text-db-gray-text">
              No retrieval — a plain model answer.
            </p>
          )}
          <div className="mt-1.5 border-t border-db-gray-lines pt-1.5">
            <button
              type="button"
              onClick={() => {
                onBrowse();
                setOpen(false);
              }}
              className="flex w-full items-center justify-between rounded-db-md px-2 py-1.5 text-[12.5px] font-medium text-db-navy-800 transition-colors hover:bg-db-oat-light"
            >
              <span>Browse all sources…</span>
              <span className="font-db-mono text-[11px] text-db-gray-text">{browseCount}</span>
            </button>
            {isDiscovering && (
              <p className="px-2 pt-1 text-[11px] text-db-gray-text">Discovering sources…</p>
            )}
          </div>
        </Popover.Content>
      </Popover.Portal>
    </Popover.Root>
  );
}

// Icons

function PaperclipIcon({ className }: { className?: string }) {
  return (
    <svg
      xmlns="http://www.w3.org/2000/svg"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
      className={className}
    >
      <path d="m21.44 11.05-9.19 9.19a6 6 0 0 1-8.49-8.49l8.57-8.57A4 4 0 1 1 18 8.84l-8.59 8.57a2 2 0 0 1-2.83-2.83l8.49-8.48" />
    </svg>
  );
}

function XCloseIcon({ className }: { className?: string }) {
  return (
    <svg
      xmlns="http://www.w3.org/2000/svg"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
      className={className}
    >
      <line x1="18" x2="6" y1="6" y2="18" />
      <line x1="6" x2="18" y1="6" y2="18" />
    </svg>
  );
}

// =============================================================================
// Agent Picker Dropdown with ownership grouping (T014)
// =============================================================================

function AgentPickerDropdown({
  agents,
  isLoading,
  isError,
  selectedAgent,
  onSelect,
}: {
  agents: CustomAgentSummary[];
  isLoading: boolean;
  isError: boolean;
  selectedAgent: CustomAgentSummary | null;
  onSelect: (agent: CustomAgentSummary) => void;
}) {
  // Group agents: private ("My Agents") vs in_app-active ("Workspace").
  // TODO(Q4): remove visibility='workspace' shim from DeploymentJobRunner once
  // this filter has soaked and the backfill migration (029) has run everywhere.
  const myAgents = agents.filter((a) => a.visibility === 'private');
  const workspaceAgents = agents.filter((a) => a.inAppActive === true);

  const renderAgent = (agent: CustomAgentSummary) => (
    <button
      key={agent.id}
      type="button"
      data-testid={`agent-option-${agent.id}`}
      onClick={() => onSelect(agent)}
      className={cn(
        'w-full text-left px-3 py-2 rounded-sm text-sm transition-colors',
        'hover:bg-accent hover:text-accent-foreground',
        selectedAgent?.id === agent.id && 'bg-accent text-accent-foreground'
      )}
    >
      <div className="flex items-center gap-1.5">
        <span className="font-medium truncate">{agent.name}</span>
        {agent.hasModelOverrides && (
          <span className="shrink-0 text-[10px] px-1 rounded bg-muted text-muted-foreground" title="Has model overrides">
            M
          </span>
        )}
        {agent.hasDomainFilter && (
          <span className="shrink-0 text-[10px] px-1 rounded bg-muted text-muted-foreground" title="Has domain filter">
            D
          </span>
        )}
      </div>
      {agent.description && (
        <div className="text-xs text-muted-foreground truncate mt-0.5">
          {agent.description}
        </div>
      )}
    </button>
  );

  return (
    <div data-testid="agent-selector-dropdown" className="absolute left-0 bottom-full z-50 mb-1 w-64 max-h-64 overflow-auto rounded-md border bg-popover p-1 shadow-md">
      {myAgents.length > 0 && (
        <>
          <div className="px-3 py-1 text-xs font-semibold text-muted-foreground uppercase tracking-wider">
            My Agents
          </div>
          {myAgents.map(renderAgent)}
        </>
      )}
      {workspaceAgents.length > 0 && (
        <>
          {myAgents.length > 0 && <div className="my-1 border-t" />}
          <div className="px-3 py-1 text-xs font-semibold text-muted-foreground uppercase tracking-wider">
            Workspace
          </div>
          {workspaceAgents.map(renderAgent)}
        </>
      )}
      {isLoading && agents.length === 0 && (
        <div className="px-3 py-2 text-sm text-muted-foreground">
          Loading agents...
        </div>
      )}
      {!isLoading && isError && agents.length === 0 && (
        <div className="px-3 py-2 text-sm text-destructive">
          Failed to load agents
        </div>
      )}
      {!isLoading && !isError && myAgents.length === 0 && workspaceAgents.length === 0 && (
        <div className="px-3 py-2 text-sm text-muted-foreground">
          No agents yet —{' '}
          <a href="/agents" className="text-primary hover:underline">
            Create one
          </a>
        </div>
      )}
    </div>
  );
}
