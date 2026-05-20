import * as React from 'react';
import { Button } from '@/components/ui/button';
import { cn } from '@/lib/utils';
import { ResearchDepthSelector, type ResearchDepth } from './ResearchDepthSelector';
import { QueryModeSelector } from './QueryModeSelector';
import { SourceScopeSelector } from '@/components/research/SourceScopeSelector';
import { SourceBrowserModal } from './SourceBrowserModal';
import { FileUploadZone } from '@/components/files/FileUploadZone';
import { UploadedFileList } from '@/components/files/UploadedFileList';
import { useQueryMode, useSourceScope } from '@/hooks';
import { useDiscoveredSources, useRefreshDiscovery } from '@/hooks/useDiscoveredSources';
import { useFileUpload } from '@/hooks/useFileUpload';
import { useAgentsV2List } from '@/hooks/useAgentsV2';
import type { AvailableSource } from '@/types/dataSources';
import type { CustomAgentSummary } from '@/types/customAgents';
import type { AgentV2Summary } from '@/types/agentDesigner';
import type { InputConfig } from '@/core/plugins/types';
import type { QuerySubmission } from '@/types/querySubmission';

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

function agentV2ToSelectorSummary(agent: AgentV2Summary): CustomAgentSummary {
  return {
    id: agent.id,
    name: agent.name,
    description: agent.description,
    avatarUrl: null,
    visibility: agent.visibility === 'workspace' ? 'workspace' : 'private',
    ownerId: agent.visibility === 'system' ? 'system' : agent.owner_id,
    inAppActive: agent.in_app_active,
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

  // Use hook for persistence (localStorage + optional API sync)
  // Only sync with preferences when mode selector is visible
  const { mode: storedMode, setMode: setStoredMode } = useQueryMode({
    initialMode: 'web_search',
    syncWithPreferences: effectiveShowModeSelector, // Only sync when visible
  });

  // Effective query mode: plugin default when selector hidden, else user's choice
  const queryMode = effectiveShowModeSelector
    ? storedMode
    : (inputConfig?.defaultQueryMode ?? 'deep_research');

  // Show depth selector only when Deep Research mode is selected AND selector is enabled
  const shouldShowDepthSelector = effectiveShowDepthSelector && queryMode === 'deep_research';
  // Show verify sources checkbox when web_search or deep_research is selected AND checkbox is enabled
  const shouldShowVerifyCheckbox = effectiveShowVerifySources && (queryMode === 'web_search' || queryMode === 'deep_research');
  // shouldShowSourceScope is computed later, after selectedAgent state is declared

  // Only allow mode changes when selector is visible
  const setQueryMode = effectiveShowModeSelector ? setStoredMode : () => {};

  // Use plugin default for research depth when selector is hidden
  const [researchDepth, setResearchDepth] = React.useState<ResearchDepth>(
    inputConfig?.defaultResearchDepth ?? 'auto'
  );

  // Default: use plugin config if selector hidden, else true for deep_research
  const [verifySources, setVerifySources] = React.useState<boolean>(
    !effectiveShowVerifySources
      ? (inputConfig?.defaultVerifySources ?? true)
      : false
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
    () => (queryMode === 'deep_research' ? (agentsData?.items ?? []).map(agentV2ToSelectorSummary) : []),
    [agentsData?.items, queryMode],
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

  // Source scope and disabled sources with localStorage persistence (T050-T051)
  const {
    scope: sourceScope,
    setScope: setSourceScope,
    disabledSources,
    setDisabledSources,
    toggleSource: handleSourceToggle,
  } = useSourceScope({ validSourceIds });

  // Wrap toggle to track explicitly enabled enterprise sources
  const wrappedSourceToggle = React.useCallback(
    (sourceId: string, enabled: boolean) => {
      // Check if this is an enterprise source
      const source = discoveredSources.find((s) => s.source_id === sourceId);
      const isEnterprise =
        source && source.source_type !== 'web_search' && source.source_type !== 'uploaded_file';

      if (isEnterprise) {
        const enabledSet = readEnabledEnterpriseSources();
        if (enabled) {
          enabledSet.add(sourceId);
        } else {
          enabledSet.delete(sourceId);
        }
        writeEnabledEnterpriseSources(enabledSet);
      }

      handleSourceToggle(sourceId, enabled);
    },
    [handleSourceToggle, discoveredSources]
  );

  // Convert discovered sources to AvailableSource format for SourceScopeSelector
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

    const readyEnterpriseSourceIds = discoveryData.sources
      .filter(
        (source) =>
          source.status === 'ready' &&
          source.source_type !== 'web_search' &&
          source.source_type !== 'uploaded_file'
      )
      .map((source) => source.source_id);

    if (readyEnterpriseSourceIds.length === 0) return;

    const enabledSet = readEnabledEnterpriseSources();
    const shouldDisable = readyEnterpriseSourceIds.filter((id) => !enabledSet.has(id));

    if (shouldDisable.length === 0) return;

    // Merge with existing disabledSources (avoid duplicates)
    const nextDisabled = Array.from(new Set([...disabledSources, ...shouldDisable]));
    if (nextDisabled.length !== disabledSources.length) {
      setDisabledSources(nextDisabled);
    }
  }, [discoveryData?.sources, disabledSources, setDisabledSources]);

  // Reset verifySources when query mode changes (only when selector is visible)
  // Default to OFF - user must explicitly enable source verification
  React.useEffect(() => {
    if (effectiveShowVerifySources) {
      setVerifySources(false);
    }
  }, [queryMode, effectiveShowVerifySources]);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (message.trim() && !isLoading && !disabled) {
      // Compute enabledSources from availableSources minus disabledSources
      const enabledSourceIds = shouldShowSourceScope && availableSources.length > 0
        ? availableSources
            .filter((s) => s.isEnabled)
            .map((s) => s.id)
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
        outputType: inputConfig?.defaultOutputType,
        sourceScope: shouldShowSourceScope ? sourceScope : undefined,
        enabledSources: enabledSourceIds,
        disabledSources: disabledSourceIds,
        fileIds: readyFiles.length > 0 ? readyFiles.map(f => f.id) : undefined,
        agentId: selectedAgent?.id ?? undefined,
        enablePlanReview: enablePlanReview || undefined,
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

  // T053-T054: Validation - Check if submission should be blocked due to source scope issues
  const enterpriseSources = React.useMemo(() => {
    return availableSources.filter(
      (s) => s.type !== 'web_search' && s.type !== 'uploaded_file'
    );
  }, [availableSources]);

  const enabledEnterpriseSources = React.useMemo(() => {
    return enterpriseSources.filter((s) => s.isEnabled);
  }, [enterpriseSources]);

  // Determine if we should block submission due to source configuration
  const sourceValidation = React.useMemo(() => {
    if (!shouldShowSourceScope) {
      return { isValid: true, message: null };
    }

    if (sourceScope === 'enterprise_only') {
      // T053: Block when Enterprise Only but no enterprise sources discovered
      if (enterpriseSources.length === 0) {
        return {
          isValid: false,
          message: 'No enterprise data sources available. Select "Web Only" or "All Sources".',
        };
      }
      // T054: Block when Enterprise Only but all enterprise sources are disabled
      if (enabledEnterpriseSources.length === 0) {
        return {
          isValid: false,
          message: 'All enterprise sources are disabled. Enable at least one source or change scope.',
        };
      }
    }

    return { isValid: true, message: null };
  }, [shouldShowSourceScope, sourceScope, enterpriseSources.length, enabledEnterpriseSources.length]);

  // Check if any files are still processing (not ready yet)
  const hasProcessingFiles = React.useMemo(() => {
    return sessionFiles.some(
      f => f.processingStatus === 'pending' || f.processingStatus === 'processing'
    );
  }, [sessionFiles]);

  // Combined disabled state for submit button
  const isSubmitDisabled =
    !message.trim() || isLoading || disabled || !sourceValidation.isValid || hasProcessingFiles;

  return (
    <form onSubmit={handleSubmit} className="border-t bg-background">
      <div className="px-4 pt-2 flex flex-wrap gap-4 items-center">
        {effectiveShowModeSelector && (
          <QueryModeSelector
            value={queryMode}
            onChange={setQueryMode}
            disabled={disabled || isLoading}
          />
        )}
        {shouldShowDepthSelector && (
          <ResearchDepthSelector
            value={researchDepth}
            onChange={setResearchDepth}
            disabled={disabled || isLoading}
          />
        )}
        {shouldShowVerifyCheckbox && (
          <label className="flex items-center gap-1.5 text-xs text-muted-foreground cursor-pointer select-none">
            <input
              type="checkbox"
              checked={verifySources}
              onChange={(e) => setVerifySources(e.target.checked)}
              disabled={disabled || isLoading}
              className="h-3.5 w-3.5 rounded border-input cursor-pointer accent-primary"
            />
            <span>Verify sources</span>
          </label>
        )}
        {/* Agent selector (deep_research mode) */}
        {queryMode === 'deep_research' && effectiveShowModeSelector && (
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
                'flex items-center gap-1.5 px-2 py-1 rounded text-xs transition-colors',
                selectedAgent
                  ? 'bg-accent text-accent-foreground'
                  : 'text-muted-foreground hover:text-foreground hover:bg-accent',
                (disabled || isLoading) && 'opacity-50 cursor-not-allowed'
              )}
            >
              <AgentIcon className="h-3.5 w-3.5" />
              <span data-testid="agent-selected-name" className="max-w-[100px] truncate">
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
                    className="ml-0.5 hover:text-foreground"
                    title="Clear agent selection"
                  >
                    ×
                  </button>
                </>
              )}
            </button>
            {selectedAgent && selectedAgent.sourceScope && selectedAgent.sourceScope !== 'all' && (
              <span data-testid="agent-source-scope-indicator" className="text-xs text-muted-foreground">
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
        {/* Plan review toggle (deep_research mode) */}
        {queryMode === 'deep_research' && effectiveShowModeSelector && (
          <label className="flex items-center gap-1.5 text-xs text-muted-foreground cursor-pointer select-none">
            <input
              type="checkbox"
              checked={enablePlanReview}
              onChange={(e) => setEnablePlanReview(e.target.checked)}
              disabled={disabled || isLoading}
              className="h-3.5 w-3.5 rounded border-input cursor-pointer accent-primary"
            />
            <span>Review plan</span>
          </label>
        )}
        {shouldShowSourceScope && (
          <>
            <SourceScopeSelector
              selectedScope={sourceScope}
              onScopeChange={setSourceScope}
              availableSources={availableSources}
              onSourceToggle={wrappedSourceToggle}
              disabled={disabled || isLoading}
              compact={true}
            />
            {/* Browse button to open full source browser modal */}
            <Button
              type="button"
              variant="outline"
              size="sm"
              onClick={() => setShowSourceBrowser(true)}
              disabled={disabled || isLoading}
              className="h-7 text-xs"
            >
              Browse ({selectedSourceIds.length})
            </Button>
            {/* Discovery status feedback */}
            {isDiscoveryLoading && (
              <span className="text-xs text-muted-foreground animate-pulse">
                Discovering sources...
              </span>
            )}
            {discoveryError && !isDiscoveryLoading && (
              <div className="flex items-center gap-2">
                <span className="text-xs text-destructive">Discovery failed</span>
                <Button
                  type="button"
                  variant="ghost"
                  size="sm"
                  onClick={() => {
                    refreshDiscoveryMutation.mutate({});
                    refetchDiscovery();
                  }}
                  disabled={refreshDiscoveryMutation.isPending}
                  className="h-6 px-2 text-xs"
                >
                  {refreshDiscoveryMutation.isPending ? 'Retrying...' : 'Retry'}
                </Button>
              </div>
            )}
            {!isDiscoveryLoading && !discoveryError && availableSources.length === 0 && (
              <span className="text-xs text-muted-foreground">
                No enterprise sources discovered.
              </span>
            )}
          </>
        )}
        {/* File attach button — last position in toolbar */}
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
              'flex items-center gap-1 px-2 py-1 rounded text-xs transition-colors',
              showFileUpload
                ? 'bg-accent text-accent-foreground'
                : 'text-muted-foreground hover:text-foreground hover:bg-accent',
              (disabled || isLoading) && 'opacity-50 cursor-not-allowed'
            )}
            title="Attach files"
          >
            <PaperclipIcon className="h-3.5 w-3.5" />
            <span>Attach files</span>
            {sessionFiles.length > 0 && (
              <span className="bg-primary text-primary-foreground rounded-full px-1.5 py-0.5 text-[10px] font-medium leading-none">
                {sessionFiles.length}
              </span>
            )}
          </button>
        )}
      </div>
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
      <div className="flex gap-2 p-4 pt-2">
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
            'flex-1 resize-none rounded-md border border-input bg-transparent px-3 py-2 text-sm shadow-sm transition-colors',
            'placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring',
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
            className="self-end"
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

          // Update enabled enterprise sources tracking
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
    <div data-testid="agent-selector-dropdown" className="absolute left-0 top-full z-50 mt-1 w-64 max-h-64 overflow-auto rounded-md border bg-popover p-1 shadow-md">
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

function AgentIcon({ className }: { className?: string }) {
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
      <path d="M12 8V4H8" />
      <rect width="16" height="12" x="4" y="8" rx="2" />
      <path d="M2 14h2M20 14h2M15 13v2M9 13v2" />
    </svg>
  );
}
