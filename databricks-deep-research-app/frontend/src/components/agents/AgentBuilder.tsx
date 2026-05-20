/**
 * AgentBuilder - Multi-section form for creating/editing custom agents.
 *
 * Sections:
 * - Identity: Name, description, avatar URL
 * - Prompts: System prompt template, synthesis template, inline options
 * - Sources: Source scope, enabled/disabled sources
 * - Workflow: Default depth, workflow mode, clarification
 * - Output: Format selector, JSON schema
 * - Preset Steps: If workflow_mode is not 'planner', manage preset steps
 */

import * as React from 'react';
import { cn } from '@/lib/utils';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { ScrollArea } from '@/components/ui/scroll-area';
import { AgentPresetSteps } from './AgentPresetSteps';
import { ModelConfigSection } from './ModelConfigSection';
import { DomainFilterSection } from './DomainFilterSection';
import { TemplatePickerDropdown } from './TemplatePickerDropdown';
import { SourcesSection } from './SourcesSection';
import { useDiscoveredSources, useRefreshDiscovery } from '@/hooks/useDiscoveredSources';
import type {
  CustomAgent,
  CreateCustomAgentRequest,
  UpdateCustomAgentRequest,
  AgentVisibility,
  WorkflowMode,
  ResearchDepth,
  OutputFormat,
  PresetStep,
  AgentSourceConfig,
  AgentWorkflowConfig,
  AgentOutputConfig,
} from '@/types/customAgents';
import {
  AGENT_VISIBILITY_LABELS,
  WORKFLOW_MODE_LABELS,
  RESEARCH_DEPTH_LABELS,
  OUTPUT_FORMAT_LABELS,
  getDefaultSourceConfig,
  getDefaultWorkflowConfig,
  getDefaultOutputConfig,
} from '@/types/customAgents';

/** Minimal source shape used by SourcesSection and AgentPresetSteps. */
export interface SelectableSource {
  id: string;
  name: string;
  type: string;
  description: string | null;
}

interface AgentBuilderProps {
  /** Existing agent for editing (undefined for create mode) */
  agent?: CustomAgent;
  /** Preset steps for the agent (only relevant when editing) */
  presetSteps?: PresetStep[];
  /** Callback when form is saved */
  onSave: (data: CreateCustomAgentRequest | UpdateCustomAgentRequest, steps?: PresetStep[]) => void;
  /** Callback when form is cancelled */
  onCancel: () => void;
  /** Whether the form is in loading/saving state */
  isLoading?: boolean;
  /** Additional CSS classes */
  className?: string;
}

type Section = 'identity' | 'prompts' | 'sources' | 'models' | 'domains' | 'workflow' | 'output' | 'steps';

const SECTIONS: { key: Section; label: string; icon: React.ReactNode }[] = [
  { key: 'identity', label: 'Identity', icon: <UserIcon className="h-4 w-4" /> },
  { key: 'prompts', label: 'Prompts', icon: <FileTextIcon className="h-4 w-4" /> },
  { key: 'sources', label: 'Sources', icon: <DatabaseIcon className="h-4 w-4" /> },
  { key: 'models', label: 'Models', icon: <CpuIcon className="h-4 w-4" /> },
  { key: 'domains', label: 'Domains', icon: <GlobeIcon className="h-4 w-4" /> },
  { key: 'workflow', label: 'Workflow', icon: <GitBranchIcon className="h-4 w-4" /> },
  { key: 'output', label: 'Output', icon: <FileJsonIcon className="h-4 w-4" /> },
  { key: 'steps', label: 'Preset Steps', icon: <ListIcon className="h-4 w-4" /> },
];

export function AgentBuilder({
  agent,
  presetSteps: initialPresetSteps = [],
  onSave,
  onCancel,
  isLoading = false,
  className,
}: AgentBuilderProps) {
  const isEditMode = !!agent;

  // Form state
  const [activeSection, setActiveSection] = React.useState<Section>('identity');

  // Identity — agent fields are camelCase (matching backend serialization)
  const [name, setName] = React.useState(agent?.name || '');
  const [description, setDescription] = React.useState(agent?.description || '');
  const [avatarUrl, setAvatarUrl] = React.useState(agent?.avatarUrl || '');
  const [visibility, setVisibility] = React.useState<AgentVisibility>(
    agent?.visibility || 'private'
  );

  // Prompts
  const [systemPromptTemplateId, setSystemPromptTemplateId] = React.useState<string | null>(
    agent?.systemPromptTemplateId || null
  );
  const [synthesisTemplateId, setSynthesisTemplateId] = React.useState<string | null>(
    agent?.synthesisTemplateId || null
  );
  const [inlineSystemPrompt, setInlineSystemPrompt] = React.useState(
    agent?.inlineSystemPrompt || ''
  );
  const [inlineSynthesisPrompt, setInlineSynthesisPrompt] = React.useState(
    agent?.inlineSynthesisPrompt || ''
  );
  const [useInlineSystem, setUseInlineSystem] = React.useState(!!agent?.inlineSystemPrompt);
  const [useInlineSynthesis, setUseInlineSynthesis] = React.useState(!!agent?.inlineSynthesisPrompt);

  // Sources
  const [sourceConfig, setSourceConfig] = React.useState<AgentSourceConfig>(
    agent?.sourceConfig ?? getDefaultSourceConfig()
  );

  // Workflow
  const [workflowConfig, setWorkflowConfig] = React.useState<AgentWorkflowConfig>(
    agent?.workflowConfig ?? getDefaultWorkflowConfig()
  );

  // Output
  const [outputConfig, setOutputConfig] = React.useState<AgentOutputConfig>(
    agent?.outputConfig ?? getDefaultOutputConfig()
  );

  // Model overrides (009-custom-agent-config)
  const [modelOverrides, setModelOverrides] = React.useState<Record<string, string> | null>(
    agent?.modelOverrides ?? null
  );

  // Domain filter (009-custom-agent-config)
  const [domainFilterMode, setDomainFilterMode] = React.useState<string | null>(
    agent?.domainFilterMode ?? null
  );
  const [includeDomains, setIncludeDomains] = React.useState<string[] | null>(
    agent?.includeDomains ?? null
  );
  const [excludeDomains, setExcludeDomains] = React.useState<string[] | null>(
    agent?.excludeDomains ?? null
  );
  // Soft reputation lists — re-rank without filtering. Independent of mode.
  const [preferredDomains, setPreferredDomains] = React.useState<string[] | null>(
    (agent as unknown as { preferredDomains?: string[] | null })?.preferredDomains ?? null
  );
  const [deprecatedDomains, setDeprecatedDomains] = React.useState<string[] | null>(
    (agent as unknown as { deprecatedDomains?: string[] | null })?.deprecatedDomains ?? null
  );

  // Preset Steps
  const [presetSteps, setPresetSteps] = React.useState<PresetStep[]>(initialPresetSteps);

  // Fetch data — use auto-discovered sources (not manually-created data sources)
  const { data: discoveryData, isLoading: sourcesLoading, error: sourcesError } = useDiscoveredSources();
  const refreshDiscovery = useRefreshDiscovery();
  const availableSources: SelectableSource[] = (discoveryData?.sources ?? []).map(s => ({
    id: s.source_id,
    name: s.name,
    type: s.source_type,
    description: s.description ?? null,
  }));

  // Validation
  const isFormValid = name.trim().length > 0;

  const handleSave = () => {
    // Build flat payload matching backend Pydantic schema (snake_case accepted via populate_by_name).
    // The backend expects top-level fields, NOT nested config objects.
    const payload: Record<string, unknown> = {
      name: name.trim(),
      description: description.trim() || undefined,
      avatar_url: avatarUrl.trim() || undefined,
      visibility,
      // Prompts
      system_prompt_template_id: useInlineSystem ? undefined : systemPromptTemplateId || undefined,
      synthesis_template_id: useInlineSynthesis ? undefined : synthesisTemplateId || undefined,
      inline_system_prompt: useInlineSystem ? inlineSystemPrompt.trim() || undefined : undefined,
      inline_synthesis_prompt: useInlineSynthesis ? inlineSynthesisPrompt.trim() || undefined : undefined,
      // Sources — flat (was nested in sourceConfig)
      source_scope: sourceConfig.scope,
      enabled_sources: sourceConfig.enabledSources.length > 0 ? sourceConfig.enabledSources : undefined,
      disabled_sources: sourceConfig.disabledSources,
      // Workflow — flat (was nested in workflowConfig); use_planner DERIVED from workflowMode
      use_planner: workflowConfig.workflowMode === 'planner',
      default_depth: workflowConfig.defaultDepth,
      default_mode: workflowConfig.workflowMode,
      enable_clarification: workflowConfig.enableClarification,
      // Output — flat (was nested in outputConfig)
      output_format: outputConfig.format,
      output_schema: outputConfig.jsonSchema,
      // Model overrides (009-custom-agent-config)
      model_overrides: modelOverrides ?? undefined,
      // Domain filter (009-custom-agent-config)
      domain_filter_mode: domainFilterMode ?? undefined,
      include_domains: includeDomains ?? undefined,
      exclude_domains: excludeDomains ?? undefined,
      // Soft per-agent ranking (PR 3) — independent of filter mode.
      preferred_domains: preferredDomains ?? undefined,
      deprecated_domains: deprecatedDomains ?? undefined,
    };

    // Include preset steps for non-planner modes
    if (workflowConfig.workflowMode !== 'planner' && presetSteps.length > 0) {
      payload.preset_steps = presetSteps.map((step, index) => ({
        title: step.title,
        description: step.description || undefined,
        order: index + 1,
        is_required: step.isRequired,
        source_scope: step.sourceScope || undefined,
        source_hints: step.sourceHints?.length
          ? {
              preferred_sources: step.sourceHints.map((h) => h.sourceName),
              search_queries: step.sourceHints
                .filter((h) => h.queryHint)
                .map((h) => h.queryHint),
            }
          : undefined,
      }));
    }

    // TODO: Align frontend type definitions (CreateCustomAgentRequest, UpdateCustomAgentRequest)
    // with the backend flat schema to remove this type assertion.
    onSave(payload as CreateCustomAgentRequest | UpdateCustomAgentRequest);
  };

  const updateSourceConfig = (updates: Partial<AgentSourceConfig>) => {
    setSourceConfig((prev) => ({ ...prev, ...updates }));
  };

  const updateWorkflowConfig = (updates: Partial<AgentWorkflowConfig>) => {
    setWorkflowConfig((prev) => ({ ...prev, ...updates }));
  };

  const updateOutputConfig = (updates: Partial<AgentOutputConfig>) => {
    setOutputConfig((prev) => ({ ...prev, ...updates }));
  };

  // Show steps section only when not using planner
  const visibleSections = SECTIONS.filter(
    (s) => s.key !== 'steps' || workflowConfig.workflowMode !== 'planner'
  );

  return (
    <div className={cn('flex flex-col h-full', className)}>
      {/* Header */}
      <div className="px-4 py-3 border-b flex items-center justify-between">
        <h2 className="text-lg font-semibold">
          {isEditMode ? 'Edit Agent' : 'Create Agent'}
        </h2>
        <div className="flex items-center gap-2">
          <Button variant="outline" onClick={onCancel} disabled={isLoading}>
            Cancel
          </Button>
          <Button onClick={handleSave} disabled={!isFormValid || isLoading} loading={isLoading}>
            {isLoading ? 'Saving...' : isEditMode ? 'Save Changes' : 'Create Agent'}
          </Button>
        </div>
      </div>

      <div className="flex flex-1 overflow-hidden">
        {/* Section Navigation */}
        <div className="w-48 border-r bg-muted/30 p-2">
          <nav className="space-y-1">
            {visibleSections.map((section) => (
              <button
                key={section.key}
                type="button"
                onClick={() => setActiveSection(section.key)}
                className={cn(
                  'w-full flex items-center gap-2 px-3 py-2 rounded-md text-sm text-left',
                  'transition-colors',
                  activeSection === section.key
                    ? 'bg-primary/10 text-primary font-medium'
                    : 'text-muted-foreground hover:bg-muted hover:text-foreground'
                )}
              >
                {section.icon}
                {section.label}
              </button>
            ))}
          </nav>
        </div>

        {/* Section Content */}
        <ScrollArea className="flex-1">
          <div className="p-6 max-w-2xl">
            {/* Identity Section */}
            {activeSection === 'identity' && (
              <IdentitySection
                name={name}
                description={description}
                avatarUrl={avatarUrl}
                visibility={visibility}
                onNameChange={setName}
                onDescriptionChange={setDescription}
                onAvatarUrlChange={setAvatarUrl}
                onVisibilityChange={setVisibility}
                disabled={isLoading}
              />
            )}

            {/* Prompts Section */}
            {activeSection === 'prompts' && (
              <PromptsSection
                systemPromptTemplateId={systemPromptTemplateId}
                synthesisTemplateId={synthesisTemplateId}
                inlineSystemPrompt={inlineSystemPrompt}
                inlineSynthesisPrompt={inlineSynthesisPrompt}
                useInlineSystem={useInlineSystem}
                useInlineSynthesis={useInlineSynthesis}
                onSystemTemplateChange={setSystemPromptTemplateId}
                onSynthesisTemplateChange={setSynthesisTemplateId}
                onInlineSystemChange={setInlineSystemPrompt}
                onInlineSynthesisChange={setInlineSynthesisPrompt}
                onUseInlineSystemChange={setUseInlineSystem}
                onUseInlineSynthesisChange={setUseInlineSynthesis}
                disabled={isLoading}
              />
            )}

            {/* Sources Section */}
            {activeSection === 'sources' && (
              <SourcesSection
                config={sourceConfig}
                sources={discoveryData?.sources ?? []}
                onChange={updateSourceConfig}
                disabled={isLoading}
                isLoadingSources={sourcesLoading}
                sourcesError={sourcesError}
                onRefresh={() => refreshDiscovery.mutate(undefined)}
                isRefreshing={refreshDiscovery.isPending}
              />
            )}

            {/* Models Section (009-custom-agent-config) */}
            {activeSection === 'models' && (
              <ModelConfigSection
                modelOverrides={modelOverrides}
                onChange={setModelOverrides}
                disabled={isLoading}
              />
            )}

            {/* Domain Filter Section (009-custom-agent-config) */}
            {activeSection === 'domains' && (
              <DomainFilterSection
                domainFilterMode={domainFilterMode}
                includeDomains={includeDomains}
                excludeDomains={excludeDomains}
                preferredDomains={preferredDomains}
                deprecatedDomains={deprecatedDomains}
                onChange={(mode, include, exclude, preferred, deprecated) => {
                  setDomainFilterMode(mode);
                  setIncludeDomains(include);
                  setExcludeDomains(exclude);
                  if (preferred !== undefined) setPreferredDomains(preferred);
                  if (deprecated !== undefined) setDeprecatedDomains(deprecated);
                }}
                disabled={isLoading}
              />
            )}

            {/* Workflow Section */}
            {activeSection === 'workflow' && (
              <WorkflowSection
                config={workflowConfig}
                onChange={updateWorkflowConfig}
                disabled={isLoading}
              />
            )}

            {/* Output Section */}
            {activeSection === 'output' && (
              <OutputSection
                config={outputConfig}
                onChange={updateOutputConfig}
                disabled={isLoading}
              />
            )}

            {/* Preset Steps Section */}
            {activeSection === 'steps' && workflowConfig.workflowMode !== 'planner' && (
              <div className="space-y-4">
                <div>
                  <h3 className="text-lg font-medium">Preset Steps</h3>
                  <p className="text-sm text-muted-foreground mt-1">
                    Define the fixed research steps this agent will follow.
                  </p>
                </div>
                <AgentPresetSteps
                  steps={presetSteps}
                  onChange={setPresetSteps}
                  availableSources={availableSources}
                  readOnly={isLoading}
                />
              </div>
            )}
          </div>
        </ScrollArea>
      </div>
    </div>
  );
}

// =============================================================================
// Identity Section
// =============================================================================

interface IdentitySectionProps {
  name: string;
  description: string;
  avatarUrl: string;
  visibility: AgentVisibility;
  onNameChange: (value: string) => void;
  onDescriptionChange: (value: string) => void;
  onAvatarUrlChange: (value: string) => void;
  onVisibilityChange: (value: AgentVisibility) => void;
  disabled: boolean;
}

function IdentitySection({
  name,
  description,
  avatarUrl,
  visibility,
  onNameChange,
  onDescriptionChange,
  onAvatarUrlChange,
  onVisibilityChange,
  disabled,
}: IdentitySectionProps) {
  return (
    <div className="space-y-6">
      <div>
        <h3 className="text-lg font-medium">Identity</h3>
        <p className="text-sm text-muted-foreground mt-1">
          Basic information about your custom agent.
        </p>
      </div>

      <div className="space-y-4">
        {/* Name */}
        <div>
          <label className="text-sm font-medium mb-1.5 block">Name *</label>
          <Input
            value={name}
            onChange={(e) => onNameChange(e.target.value)}
            placeholder="My Custom Agent"
            disabled={disabled}
          />
        </div>

        {/* Description */}
        <div>
          <label className="text-sm font-medium mb-1.5 block">Description</label>
          <textarea
            value={description}
            onChange={(e) => onDescriptionChange(e.target.value)}
            placeholder="Describe what this agent specializes in..."
            rows={3}
            disabled={disabled}
            className={cn(
              'w-full resize-none rounded-md border border-input bg-transparent px-3 py-2 text-sm',
              'placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring',
              'disabled:cursor-not-allowed disabled:opacity-50'
            )}
          />
        </div>

        {/* Avatar URL */}
        <div>
          <label className="text-sm font-medium mb-1.5 block">Avatar URL</label>
          <div className="flex items-start gap-3">
            <Input
              value={avatarUrl}
              onChange={(e) => onAvatarUrlChange(e.target.value)}
              placeholder="https://example.com/avatar.png"
              disabled={disabled}
              className="flex-1"
            />
            {avatarUrl && (
              <div className="shrink-0">
                <img
                  src={avatarUrl}
                  alt="Avatar preview"
                  className="h-10 w-10 rounded-full object-cover border"
                  onError={(e) => {
                    (e.target as HTMLImageElement).style.display = 'none';
                  }}
                />
              </div>
            )}
          </div>
          <p className="text-xs text-muted-foreground mt-1">
            Optional URL for the agent's avatar image
          </p>
        </div>

        {/* Visibility */}
        <div>
          <label className="text-sm font-medium mb-1.5 block">Visibility</label>
          <div className="flex gap-2">
            {(Object.keys(AGENT_VISIBILITY_LABELS) as AgentVisibility[]).map((v) => (
              <button
                key={v}
                type="button"
                onClick={() => onVisibilityChange(v)}
                disabled={disabled}
                className={cn(
                  'px-3 py-1.5 rounded-md text-sm transition-colors',
                  'border',
                  visibility === v
                    ? 'border-primary bg-primary/10 text-primary font-medium'
                    : 'border-input text-muted-foreground hover:border-primary/50 hover:text-foreground'
                )}
              >
                {AGENT_VISIBILITY_LABELS[v]}
              </button>
            ))}
          </div>
          <p className="text-xs text-muted-foreground mt-1">
            {visibility === 'private'
              ? 'Only you can see and use this agent'
              : 'Anyone in your workspace can see and use this agent'}
          </p>
        </div>
      </div>
    </div>
  );
}

// =============================================================================
// Prompts Section
// =============================================================================

interface PromptsSectionProps {
  systemPromptTemplateId: string | null;
  synthesisTemplateId: string | null;
  inlineSystemPrompt: string;
  inlineSynthesisPrompt: string;
  useInlineSystem: boolean;
  useInlineSynthesis: boolean;
  onSystemTemplateChange: (id: string | null) => void;
  onSynthesisTemplateChange: (id: string | null) => void;
  onInlineSystemChange: (value: string) => void;
  onInlineSynthesisChange: (value: string) => void;
  onUseInlineSystemChange: (value: boolean) => void;
  onUseInlineSynthesisChange: (value: boolean) => void;
  disabled: boolean;
}

function PromptsSection({
  systemPromptTemplateId,
  synthesisTemplateId,
  inlineSystemPrompt,
  inlineSynthesisPrompt,
  useInlineSystem,
  useInlineSynthesis,
  onSystemTemplateChange,
  onSynthesisTemplateChange,
  onInlineSystemChange,
  onInlineSynthesisChange,
  onUseInlineSystemChange,
  onUseInlineSynthesisChange,
  disabled,
}: PromptsSectionProps) {
  return (
    <div className="space-y-6">
      <div>
        <h3 className="text-lg font-medium">Prompts</h3>
        <p className="text-sm text-muted-foreground mt-1">
          Configure the system and synthesis prompts for this agent.
        </p>
      </div>

      {/* System Prompt */}
      <div className="space-y-3">
        <div className="flex items-center justify-between">
          <label className="text-sm font-medium">System Prompt</label>
          <label className="flex items-center gap-2 text-sm cursor-pointer">
            <input
              type="checkbox"
              checked={useInlineSystem}
              onChange={(e) => onUseInlineSystemChange(e.target.checked)}
              disabled={disabled}
              className="rounded border-input"
            />
            Write inline
          </label>
        </div>

        {useInlineSystem ? (
          <textarea
            value={inlineSystemPrompt}
            onChange={(e) => onInlineSystemChange(e.target.value)}
            placeholder="Enter your custom system prompt..."
            rows={6}
            disabled={disabled}
            className={cn(
              'w-full resize-none rounded-md border border-input bg-transparent px-3 py-2 text-sm font-mono',
              'placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring',
              'disabled:cursor-not-allowed disabled:opacity-50'
            )}
          />
        ) : (
          <TemplatePickerDropdown
            selectedTemplateId={systemPromptTemplateId}
            onChange={onSystemTemplateChange}
            templateType="system"
            disabled={disabled}
          />
        )}
      </div>

      {/* Synthesis Prompt */}
      <div className="space-y-3">
        <div className="flex items-center justify-between">
          <label className="text-sm font-medium">Synthesis Prompt</label>
          <label className="flex items-center gap-2 text-sm cursor-pointer">
            <input
              type="checkbox"
              checked={useInlineSynthesis}
              onChange={(e) => onUseInlineSynthesisChange(e.target.checked)}
              disabled={disabled}
              className="rounded border-input"
            />
            Write inline
          </label>
        </div>

        {useInlineSynthesis ? (
          <textarea
            value={inlineSynthesisPrompt}
            onChange={(e) => onInlineSynthesisChange(e.target.value)}
            placeholder="Enter your custom synthesis prompt..."
            rows={6}
            disabled={disabled}
            className={cn(
              'w-full resize-none rounded-md border border-input bg-transparent px-3 py-2 text-sm font-mono',
              'placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring',
              'disabled:cursor-not-allowed disabled:opacity-50'
            )}
          />
        ) : (
          <TemplatePickerDropdown
            selectedTemplateId={synthesisTemplateId}
            onChange={onSynthesisTemplateChange}
            templateType="synthesis"
            disabled={disabled}
          />
        )}
      </div>
    </div>
  );
}

// =============================================================================
// Workflow Section
// =============================================================================

interface WorkflowSectionProps {
  config: AgentWorkflowConfig;
  onChange: (updates: Partial<AgentWorkflowConfig>) => void;
  disabled: boolean;
}

function WorkflowSection({ config, onChange, disabled }: WorkflowSectionProps) {
  return (
    <div className="space-y-6">
      <div>
        <h3 className="text-lg font-medium">Workflow</h3>
        <p className="text-sm text-muted-foreground mt-1">
          Configure how this agent plans and executes research.
        </p>
      </div>

      {/* Default Research Depth */}
      <div className="space-y-3">
        <label className="text-sm font-medium">Default Research Depth</label>
        <div className="flex gap-2">
          {(Object.keys(RESEARCH_DEPTH_LABELS) as ResearchDepth[]).map((depth) => (
            <button
              key={depth}
              type="button"
              onClick={() => onChange({ defaultDepth: depth })}
              disabled={disabled}
              className={cn(
                'px-3 py-1.5 rounded-md text-sm transition-colors border',
                config.defaultDepth === depth
                  ? 'border-primary bg-primary/10 text-primary font-medium'
                  : 'border-input text-muted-foreground hover:border-primary/50 hover:text-foreground'
              )}
            >
              {RESEARCH_DEPTH_LABELS[depth]}
            </button>
          ))}
        </div>
      </div>

      {/* Workflow Mode */}
      <div className="space-y-3">
        <label className="text-sm font-medium">Workflow Mode</label>
        <div className="space-y-2">
          {(Object.keys(WORKFLOW_MODE_LABELS) as WorkflowMode[]).map((mode) => (
            <label
              key={mode}
              className={cn(
                'flex items-center gap-3 p-3 rounded-lg border cursor-pointer',
                config.workflowMode === mode
                  ? 'border-primary bg-primary/10'
                  : 'border-input hover:border-primary/50',
                disabled && 'opacity-50 cursor-not-allowed'
              )}
            >
              <input
                type="radio"
                name="workflow_mode"
                checked={config.workflowMode === mode}
                onChange={() => onChange({ workflowMode: mode })}
                disabled={disabled}
                className="h-4 w-4"
              />
              <div>
                <p className="font-medium text-sm">{WORKFLOW_MODE_LABELS[mode]}</p>
                <p className="text-xs text-muted-foreground">
                  {mode === 'planner' && 'Agent creates and follows a dynamic plan'}
                  {mode === 'manual' && 'Follow preset steps defined by you'}
                  {mode === 'hybrid' && 'Combine preset steps with dynamic planning'}
                </p>
              </div>
            </label>
          ))}
        </div>
      </div>

      {/* Enable Clarification */}
      <div className="space-y-3">
        <label className="flex items-center gap-3 cursor-pointer">
          <input
            type="checkbox"
            checked={config.enableClarification}
            onChange={(e) => onChange({ enableClarification: e.target.checked })}
            disabled={disabled}
            className="rounded border-input h-4 w-4"
          />
          <div>
            <p className="font-medium text-sm">Enable Clarification Questions</p>
            <p className="text-xs text-muted-foreground">
              Allow the agent to ask clarifying questions before starting research
            </p>
          </div>
        </label>
      </div>
    </div>
  );
}

// =============================================================================
// Output Section
// =============================================================================

interface OutputSectionProps {
  config: AgentOutputConfig;
  onChange: (updates: Partial<AgentOutputConfig>) => void;
  disabled: boolean;
}

function OutputSection({ config, onChange, disabled }: OutputSectionProps) {
  return (
    <div className="space-y-6">
      <div>
        <h3 className="text-lg font-medium">Output</h3>
        <p className="text-sm text-muted-foreground mt-1">
          Configure the output format for this agent.
        </p>
      </div>

      {/* Output Format */}
      <div className="space-y-3">
        <label className="text-sm font-medium">Output Format</label>
        <div className="flex gap-2">
          {(Object.keys(OUTPUT_FORMAT_LABELS) as OutputFormat[]).map((format) => (
            <button
              key={format}
              type="button"
              onClick={() => onChange({ format })}
              disabled={disabled}
              className={cn(
                'px-4 py-2 rounded-md text-sm transition-colors border',
                config.format === format
                  ? 'border-primary bg-primary/10 text-primary font-medium'
                  : 'border-input text-muted-foreground hover:border-primary/50 hover:text-foreground'
              )}
            >
              {OUTPUT_FORMAT_LABELS[format]}
            </button>
          ))}
        </div>
        <p className="text-xs text-muted-foreground">
          {config.format === 'markdown'
            ? 'Agent will output a formatted markdown report'
            : 'Agent will output structured JSON matching your schema'}
        </p>
      </div>

      {/* JSON Schema */}
      {config.format === 'json' && (
        <div className="space-y-3">
          <label className="text-sm font-medium">JSON Schema</label>
          <textarea
            value={config.jsonSchema || ''}
            onChange={(e) => onChange({ jsonSchema: e.target.value || null })}
            placeholder='{\n  "type": "object",\n  "properties": {\n    "summary": { "type": "string" },\n    "findings": { "type": "array" }\n  }\n}'
            rows={10}
            disabled={disabled}
            className={cn(
              'w-full resize-none rounded-md border border-input bg-transparent px-3 py-2 text-sm font-mono',
              'placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring',
              'disabled:cursor-not-allowed disabled:opacity-50'
            )}
          />
          <p className="text-xs text-muted-foreground">
            Define a JSON Schema that the agent's output should conform to
          </p>
        </div>
      )}
    </div>
  );
}

// =============================================================================
// Icons
// =============================================================================

function UserIcon({ className }: { className?: string }) {
  return (
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className={className}>
      <path d="M19 21v-2a4 4 0 0 0-4-4H9a4 4 0 0 0-4 4v2" /><circle cx="12" cy="7" r="4" />
    </svg>
  );
}

function FileTextIcon({ className }: { className?: string }) {
  return (
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className={className}>
      <path d="M14.5 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V7.5L14.5 2z" /><polyline points="14 2 14 8 20 8" /><line x1="16" x2="8" y1="13" y2="13" /><line x1="16" x2="8" y1="17" y2="17" /><line x1="10" x2="8" y1="9" y2="9" />
    </svg>
  );
}

function DatabaseIcon({ className }: { className?: string }) {
  return (
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className={className}>
      <ellipse cx="12" cy="5" rx="9" ry="3" /><path d="M3 5v14a9 3 0 0 0 18 0V5" /><path d="M3 12a9 3 0 0 0 18 0" />
    </svg>
  );
}

function GitBranchIcon({ className }: { className?: string }) {
  return (
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className={className}>
      <line x1="6" x2="6" y1="3" y2="15" /><circle cx="18" cy="6" r="3" /><circle cx="6" cy="18" r="3" /><path d="M18 9a9 9 0 0 1-9 9" />
    </svg>
  );
}

function FileJsonIcon({ className }: { className?: string }) {
  return (
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className={className}>
      <path d="M14.5 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V7.5L14.5 2z" /><polyline points="14 2 14 8 20 8" /><path d="M10 12a1 1 0 0 0-1 1v1a1 1 0 0 1-1 1 1 1 0 0 1 1 1v1a1 1 0 0 0 1 1" /><path d="M14 18a1 1 0 0 0 1-1v-1a1 1 0 0 1 1-1 1 1 0 0 1-1-1v-1a1 1 0 0 0-1-1" />
    </svg>
  );
}

function ListIcon({ className }: { className?: string }) {
  return (
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className={className}>
      <line x1="8" x2="21" y1="6" y2="6" /><line x1="8" x2="21" y1="12" y2="12" /><line x1="8" x2="21" y1="18" y2="18" /><line x1="3" x2="3.01" y1="6" y2="6" /><line x1="3" x2="3.01" y1="12" y2="12" /><line x1="3" x2="3.01" y1="18" y2="18" />
    </svg>
  );
}

function CpuIcon({ className }: { className?: string }) {
  return (
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className={className}>
      <rect width="16" height="16" x="4" y="4" rx="2" /><rect width="6" height="6" x="9" y="9" rx="1" /><path d="M15 2v2M15 20v2M2 15h2M2 9h2M20 15h2M20 9h2M9 2v2M9 20v2" />
    </svg>
  );
}

function GlobeIcon({ className }: { className?: string }) {
  return (
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className={className}>
      <circle cx="12" cy="12" r="10" /><line x1="2" x2="22" y1="12" y2="12" /><path d="M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10z" />
    </svg>
  );
}

export default AgentBuilder;
