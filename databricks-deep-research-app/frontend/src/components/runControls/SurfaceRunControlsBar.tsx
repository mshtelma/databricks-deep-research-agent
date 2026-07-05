import * as React from 'react';
import * as Popover from '@radix-ui/react-popover';
import { Boxes, Check, ChevronDown, Database, Globe, Zap } from 'lucide-react';

import { ChatOptionsPanel } from '@/components/chat/ChatOptionsPanel';
import type { ComposerSources } from '@/components/chat/sourceRouting';
import type { ResearchDepth } from '@/components/chat/ResearchDepthSelector';
import { cn } from '@/lib/utils';
import type { SurfaceRuntimeControls } from '@/types/surface';

const EFFORT_OPTIONS: { value: ResearchDepth; label: string }[] = [
  { value: 'auto', label: 'Auto' },
  { value: 'light', label: 'Light' },
  { value: 'medium', label: 'Standard' },
  { value: 'extended', label: 'Deep' },
];

const SOURCE_CHANNELS: { id: keyof ComposerSources; label: string; icon: typeof Globe }[] = [
  { id: 'web', label: 'Web', icon: Globe },
  { id: 'ent', label: 'Enterprise', icon: Database },
  { id: 'mcp', label: 'MCP', icon: Boxes },
];

function chipClass(active: boolean): string {
  return cn(
    'inline-flex items-center gap-1.5 rounded-db-md border px-2.5 py-1.5 text-[12.5px] font-medium transition-colors',
    active
      ? 'border-db-navy-300 bg-db-oat-medium text-db-navy-800'
      : 'border-db-gray-lines bg-white text-db-navy-800 hover:bg-db-oat-light',
  );
}

function sourcesSummary(sources: ComposerSources): string {
  const active = SOURCE_CHANNELS.filter((channel) => sources[channel.id]);
  if (active.length === 0) return 'None';
  if (active.length === SOURCE_CHANNELS.length) return 'All';
  return active.map((channel) => channel.label).join(', ');
}

function policyFor(
  controls: SurfaceRuntimeControls | undefined,
  key: keyof SurfaceRuntimeControls,
): 'show' | 'hide' | 'locked' | 'advanced' {
  return controls?.[key] ?? 'show';
}

function isVisible(policy: 'show' | 'hide' | 'locked' | 'advanced'): boolean {
  return policy !== 'hide';
}

function isLocked(policy: 'show' | 'hide' | 'locked' | 'advanced'): boolean {
  return policy === 'locked';
}

function EffortChip({
  value,
  onChange,
  disabled,
}: {
  value: ResearchDepth;
  onChange: (value: ResearchDepth) => void;
  disabled?: boolean;
}): React.ReactElement {
  const [open, setOpen] = React.useState(false);
  const label = EFFORT_OPTIONS.find((option) => option.value === value)?.label ?? 'Auto';
  return (
    <Popover.Root open={open} onOpenChange={setOpen}>
      <Popover.Trigger asChild>
        <button
          type="button"
          disabled={disabled}
          data-testid="surface-effort-chip"
          className={chipClass(open)}
        >
          <Zap size={14} className="text-db-gray-text" />
          <span className="text-db-gray-text">Effort</span>
          <span className="font-semibold text-db-navy-800">{label}</span>
          <ChevronDown size={13} className="text-db-navy-400" />
        </button>
      </Popover.Trigger>
      <Popover.Portal>
        <Popover.Content
          side="bottom"
          align="start"
          sideOffset={6}
          collisionPadding={8}
          className="z-50 w-44 rounded-db-md border border-db-gray-lines bg-white p-1 shadow-lg"
        >
          {EFFORT_OPTIONS.map((option) => {
            const active = option.value === value;
            return (
              <button
                key={option.value}
                type="button"
                onClick={() => {
                  onChange(option.value);
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
                {option.label}
              </button>
            );
          })}
        </Popover.Content>
      </Popover.Portal>
    </Popover.Root>
  );
}

function SourcesChip({
  sources,
  onChange,
  onBrowse,
  browseCount,
  isDiscovering,
  disabled,
}: {
  sources: ComposerSources;
  onChange: (sources: ComposerSources) => void;
  onBrowse?: () => void;
  browseCount: number;
  isDiscovering?: boolean;
  disabled?: boolean;
}): React.ReactElement {
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
          data-testid="surface-sources-chip"
          className={chipClass(open)}
        >
          <Database size={14} className="text-db-gray-text" />
          <span className="text-db-gray-text">Sources</span>
          <span className="font-semibold text-db-navy-800">{summary}</span>
          <ChevronDown size={13} className="text-db-navy-400" />
        </button>
      </Popover.Trigger>
      <Popover.Portal>
        <Popover.Content
          side="bottom"
          align="start"
          sideOffset={6}
          collisionPadding={8}
          className="z-50 w-60 rounded-db-md border border-db-gray-lines bg-white p-2 shadow-lg"
        >
          {SOURCE_CHANNELS.map((channel) => {
            const active = sources[channel.id];
            const Icon = channel.icon;
            return (
              <button
                key={channel.id}
                type="button"
                data-testid={`surface-source-${channel.id}`}
                aria-pressed={active}
                onClick={() => toggle(channel.id)}
                className="flex w-full items-center gap-2.5 rounded-db-md px-2 py-1.5 text-left transition-colors hover:bg-db-oat-light"
              >
                <span
                  className={cn(
                    'flex h-[18px] w-[18px] shrink-0 items-center justify-center rounded-[5px] border-[1.5px] transition-colors',
                    active ? 'border-db-lava-600 bg-db-lava-600' : 'border-db-navy-300 bg-white',
                  )}
                >
                  {active && <Check size={11} className="text-white" strokeWidth={3} />}
                </span>
                <Icon size={15} className="text-db-gray-text" />
                <span className="text-[13px] font-medium text-db-navy-800">{channel.label}</span>
              </button>
            );
          })}
          {summary === 'None' && (
            <p className="px-2 pt-1.5 text-[11px] italic text-db-gray-text">
              No retrieval - a plain model answer.
            </p>
          )}
          <div className="mt-1.5 border-t border-db-gray-lines pt-1.5">
            <button
              type="button"
              disabled={disabled || !onBrowse}
              onClick={() => {
                onBrowse?.();
                setOpen(false);
              }}
              className="flex w-full items-center justify-between rounded-db-md px-2 py-1.5 text-[12.5px] font-medium text-db-navy-800 transition-colors hover:bg-db-oat-light disabled:cursor-not-allowed disabled:opacity-50"
            >
              <span>Browse all sources...</span>
              <span className="font-db-mono text-[11px] text-db-gray-text">
                {browseCount}
              </span>
            </button>
            {isDiscovering && (
              <p className="px-2 pt-1 text-[11px] text-db-gray-text">
                Discovering sources...
              </p>
            )}
          </div>
        </Popover.Content>
      </Popover.Portal>
    </Popover.Root>
  );
}

export interface SurfaceRunControlsState {
  researchDepth: ResearchDepth;
  sources: ComposerSources;
  verifySources: boolean;
  enablePlanReview: boolean;
  enableCrossSessionMemory: boolean;
  allowLiveSearch: boolean;
  tone: string;
  outputLanguage: string;
}

export interface SurfaceRunControlsBarProps {
  value: SurfaceRunControlsState;
  onChange: (next: SurfaceRunControlsState) => void;
  discoveredSourceCount?: number;
  isDiscoveringSources?: boolean;
  onBrowseSources?: () => void;
  runtimeControls?: SurfaceRuntimeControls;
  disabled?: boolean;
}

export function SurfaceRunControlsBar({
  value,
  onChange,
  discoveredSourceCount = 0,
  isDiscoveringSources = false,
  onBrowseSources,
  runtimeControls,
  disabled = false,
}: SurfaceRunControlsBarProps): React.ReactElement {
  const patch = React.useCallback(
    (partial: Partial<SurfaceRunControlsState>) => onChange({ ...value, ...partial }),
    [onChange, value],
  );

  const effortPolicy = policyFor(runtimeControls, 'effort');
  const sourcesPolicy = policyFor(runtimeControls, 'sources');
  const verifyPolicy = policyFor(runtimeControls, 'verify_sources');
  const planReviewPolicy = policyFor(runtimeControls, 'plan_review');
  const reportStylePolicy = policyFor(runtimeControls, 'report_style');
  const memoryPolicy = policyFor(runtimeControls, 'cross_session_memory');
  const liveSearchPolicy = policyFor(runtimeControls, 'live_search');
  const showOptions =
    isVisible(verifyPolicy) ||
    isVisible(planReviewPolicy) ||
    isVisible(reportStylePolicy) ||
    isVisible(memoryPolicy) ||
    isVisible(liveSearchPolicy);

  return (
    <div
      data-testid="surface-run-controls"
      className="flex flex-wrap items-center gap-2 border-b border-db-gray-lines bg-db-gray-50 px-4 py-2"
    >
      {isVisible(effortPolicy) && (
        <EffortChip
          value={value.researchDepth}
          onChange={(researchDepth) => patch({ researchDepth })}
          disabled={disabled || isLocked(effortPolicy)}
        />
      )}
      {isVisible(sourcesPolicy) && (
        <SourcesChip
          sources={value.sources}
          onChange={(sources) => patch({ sources })}
          onBrowse={onBrowseSources}
          browseCount={discoveredSourceCount}
          isDiscovering={isDiscoveringSources}
          disabled={disabled || isLocked(sourcesPolicy)}
        />
      )}
      <span className="flex-1" />
      {showOptions && (
        <ChatOptionsPanel
          tone={value.tone}
          outputLanguage={value.outputLanguage}
          onToneChange={(tone) => patch({ tone })}
          onLanguageChange={(outputLanguage) => patch({ outputLanguage })}
          showReportStyle={isVisible(reportStylePolicy)}
          reportStyleDisabled={isLocked(reportStylePolicy)}
          showVerify={isVisible(verifyPolicy)}
          verifyDisabled={isLocked(verifyPolicy)}
          verifySources={value.verifySources}
          onVerifyChange={(verifySources) => patch({ verifySources })}
          showPlanReview={isVisible(planReviewPolicy)}
          planReviewDisabled={isLocked(planReviewPolicy)}
          enablePlanReview={value.enablePlanReview}
          onPlanReviewChange={(enablePlanReview) => patch({ enablePlanReview })}
          showCrossSessionMemory={isVisible(memoryPolicy)}
          crossSessionMemoryDisabled={isLocked(memoryPolicy)}
          enableCrossSessionMemory={value.enableCrossSessionMemory}
          onCrossSessionMemoryChange={(enableCrossSessionMemory) =>
            patch({ enableCrossSessionMemory })
          }
          showLiveSearch={isVisible(liveSearchPolicy)}
          liveSearchDisabled={isLocked(liveSearchPolicy)}
          allowLiveSearch={value.allowLiveSearch}
          onAllowLiveSearchChange={(allowLiveSearch) => patch({ allowLiveSearch })}
          disabled={disabled}
        />
      )}
    </div>
  );
}
