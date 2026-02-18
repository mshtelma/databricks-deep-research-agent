/**
 * AgentSelector - Browse and select from available agents.
 *
 * Features:
 * - Browse available agents grouped by visibility:
 *   - System agents
 *   - Workspace agents
 *   - Your agents
 * - Show agent card with name, description, avatar
 * - Capabilities preview
 * - Select button
 */

import * as React from 'react';
import { cn } from '@/lib/utils';
import { Input } from '@/components/ui/input';
import { Button } from '@/components/ui/button';
import { ScrollArea } from '@/components/ui/scroll-area';
import { useGroupedAgents } from '@/hooks/useCustomAgents';
import type { CustomAgentSummary, AgentVisibility, AgentCapability } from '@/types/customAgents';
import { AGENT_CAPABILITY_LABELS } from '@/types/customAgents';

interface AgentSelectorProps {
  /** Callback when an agent is selected */
  onSelectAgent: (agent: CustomAgentSummary) => void;
  /** Currently selected agent ID */
  selectedAgentId?: string | null;
  /** Filter options */
  filter?: {
    visibility?: AgentVisibility;
    includeSystem?: boolean;
    search?: string;
  };
  /** Additional CSS classes */
  className?: string;
  /** Maximum height for the selector */
  maxHeight?: string;
  /** Show as compact list */
  compact?: boolean;
}

interface CategoryConfig {
  key: 'systemAgents' | 'workspaceAgents' | 'userAgents';
  label: string;
  icon: React.ReactNode;
  description: string;
}

const AGENT_CATEGORIES: CategoryConfig[] = [
  {
    key: 'systemAgents',
    label: 'System Agents',
    icon: <ShieldIcon className="h-4 w-4" />,
    description: 'Built-in agents with default configurations',
  },
  {
    key: 'workspaceAgents',
    label: 'Workspace Agents',
    icon: <UsersIcon className="h-4 w-4" />,
    description: 'Agents shared across your workspace',
  },
  {
    key: 'userAgents',
    label: 'Your Agents',
    icon: <UserIcon className="h-4 w-4" />,
    description: 'Your personal custom agents',
  },
];

export function AgentSelector({
  onSelectAgent,
  selectedAgentId,
  filter,
  className,
  maxHeight = '500px',
  compact = false,
}: AgentSelectorProps) {
  const [searchQuery, setSearchQuery] = React.useState(filter?.search || '');
  const [expandedCategories, setExpandedCategories] = React.useState<Set<string>>(
    new Set(AGENT_CATEGORIES.map((c) => c.key))
  );

  const { grouped, isLoading, error } = useGroupedAgents({
    visibility: filter?.visibility,
    include_system: filter?.includeSystem ?? true,
    search: searchQuery || undefined,
  });

  // Filter agents by search query (client-side filtering for instant feedback)
  const filterAgents = (agents: CustomAgentSummary[]) => {
    if (!searchQuery.trim()) return agents;
    const query = searchQuery.toLowerCase();
    return agents.filter(
      (a) =>
        a.name.toLowerCase().includes(query) ||
        (a.description && a.description.toLowerCase().includes(query))
    );
  };

  const toggleCategory = (key: string) => {
    setExpandedCategories((prev) => {
      const next = new Set(prev);
      if (next.has(key)) {
        next.delete(key);
      } else {
        next.add(key);
      }
      return next;
    });
  };

  const isSelected = (agentId: string) => selectedAgentId === agentId;

  if (isLoading) {
    return (
      <div className={cn('flex items-center justify-center p-8', className)}>
        <div className="text-sm text-muted-foreground">Loading agents...</div>
      </div>
    );
  }

  if (error) {
    return (
      <div className={cn('flex items-center justify-center p-8', className)}>
        <div className="text-sm text-destructive">Failed to load agents</div>
      </div>
    );
  }

  return (
    <div className={cn('flex flex-col', className)}>
      {/* Search */}
      <div className="p-3 border-b">
        <div className="relative">
          <SearchIcon className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
          <Input
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            placeholder="Search agents..."
            className="pl-9"
          />
        </div>
      </div>

      {/* Categories */}
      <ScrollArea className="flex-1" style={{ maxHeight }}>
        <div className="p-3 space-y-4">
          {AGENT_CATEGORIES.map((category) => {
            const agents = filterAgents(grouped[category.key]);
            const isExpanded = expandedCategories.has(category.key);

            // Skip empty categories
            if (agents.length === 0 && !isExpanded) return null;

            return (
              <div key={category.key} className="border rounded-lg">
                {/* Category header */}
                <button
                  type="button"
                  onClick={() => toggleCategory(category.key)}
                  className={cn(
                    'w-full flex items-center justify-between p-3 text-left',
                    'hover:bg-muted/50 transition-colors rounded-t-lg',
                    !isExpanded && 'rounded-b-lg'
                  )}
                >
                  <div className="flex items-center gap-3">
                    <span className="text-muted-foreground">{category.icon}</span>
                    <div>
                      <h4 className="font-medium text-sm">{category.label}</h4>
                      <p className="text-xs text-muted-foreground">{category.description}</p>
                    </div>
                  </div>
                  <div className="flex items-center gap-2">
                    <span className="text-xs text-muted-foreground bg-muted px-2 py-0.5 rounded-full">
                      {agents.length}
                    </span>
                    <ChevronIcon
                      className={cn(
                        'h-4 w-4 text-muted-foreground transition-transform',
                        isExpanded && 'rotate-180'
                      )}
                    />
                  </div>
                </button>

                {/* Agents list */}
                {isExpanded && agents.length > 0 && (
                  <div className="p-3 pt-0 space-y-2 border-t">
                    {agents.map((agent) =>
                      compact ? (
                        <CompactAgentItem
                          key={agent.id}
                          agent={agent}
                          isSelected={isSelected(agent.id)}
                          onClick={() => onSelectAgent(agent)}
                        />
                      ) : (
                        <AgentCard
                          key={agent.id}
                          agent={agent}
                          isSelected={isSelected(agent.id)}
                          onSelect={() => onSelectAgent(agent)}
                        />
                      )
                    )}
                  </div>
                )}

                {/* Empty state */}
                {isExpanded && agents.length === 0 && (
                  <div className="p-4 text-center text-sm text-muted-foreground border-t">
                    No {category.label.toLowerCase()} available
                  </div>
                )}
              </div>
            );
          })}
        </div>
      </ScrollArea>
    </div>
  );
}

// =============================================================================
// Agent Card
// =============================================================================

interface AgentCardProps {
  agent: CustomAgentSummary;
  isSelected: boolean;
  onSelect: () => void;
}

function AgentCard({ agent, isSelected, onSelect }: AgentCardProps) {
  return (
    <div
      className={cn(
        'p-3 border rounded-lg transition-all',
        'hover:border-primary/50 hover:bg-muted/30',
        isSelected && 'ring-2 ring-primary ring-offset-2 border-primary'
      )}
    >
      <div className="flex items-start gap-3">
        {/* Avatar */}
        <div
          className={cn(
            'h-10 w-10 rounded-full flex items-center justify-center shrink-0',
            'bg-primary/10 text-primary'
          )}
        >
          {agent.avatarUrl ? (
            <img
              src={agent.avatarUrl}
              alt={agent.name}
              className="h-10 w-10 rounded-full object-cover"
            />
          ) : (
            <BotIcon className="h-5 w-5" />
          )}
        </div>

        {/* Content */}
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2">
            <h4 className="font-medium text-sm truncate">{agent.name}</h4>
            {!agent.isActive && (
              <span className="px-1.5 py-0.5 rounded text-xs bg-gray-100 text-gray-600 dark:bg-gray-800 dark:text-gray-400">
                Inactive
              </span>
            )}
          </div>
          {agent.description && (
            <p className="text-xs text-muted-foreground mt-0.5 line-clamp-2">
              {agent.description}
            </p>
          )}

          {/* Capabilities */}
          {agent.capabilities.length > 0 && (
            <div className="flex flex-wrap gap-1 mt-2">
              {agent.capabilities.slice(0, 4).map((cap) => (
                <CapabilityBadge key={cap} capability={cap} />
              ))}
              {agent.capabilities.length > 4 && (
                <span className="text-xs text-muted-foreground">
                  +{agent.capabilities.length - 4} more
                </span>
              )}
            </div>
          )}
        </div>

        {/* Select button */}
        <Button
          variant={isSelected ? 'default' : 'outline'}
          size="sm"
          onClick={(e) => {
            e.stopPropagation();
            onSelect();
          }}
          className="shrink-0"
        >
          {isSelected ? 'Selected' : 'Select'}
        </Button>
      </div>
    </div>
  );
}

// =============================================================================
// Compact Agent Item
// =============================================================================

interface CompactAgentItemProps {
  agent: CustomAgentSummary;
  isSelected: boolean;
  onClick: () => void;
}

function CompactAgentItem({ agent, isSelected, onClick }: CompactAgentItemProps) {
  return (
    <button
      type="button"
      onClick={onClick}
      className={cn(
        'w-full flex items-center gap-3 p-2 rounded-md text-left',
        'hover:bg-muted transition-colors',
        isSelected && 'bg-primary/10 ring-1 ring-primary'
      )}
    >
      {/* Avatar */}
      <div
        className={cn(
          'h-8 w-8 rounded-full flex items-center justify-center shrink-0',
          'bg-primary/10 text-primary'
        )}
      >
        {agent.avatarUrl ? (
          <img
            src={agent.avatarUrl}
            alt={agent.name}
            className="h-8 w-8 rounded-full object-cover"
          />
        ) : (
          <BotIcon className="h-4 w-4" />
        )}
      </div>

      <div className="flex-1 min-w-0">
        <p className="text-sm font-medium truncate">{agent.name}</p>
        <p className="text-xs text-muted-foreground truncate">
          {agent.description || 'No description'}
        </p>
      </div>

      {isSelected && (
        <CheckIcon className="h-4 w-4 text-primary shrink-0" />
      )}
    </button>
  );
}

// =============================================================================
// Capability Badge
// =============================================================================

function CapabilityBadge({ capability }: { capability: AgentCapability }) {
  const colorMap: Record<AgentCapability, string> = {
    web_search: 'bg-orange-100 text-orange-800 dark:bg-orange-900 dark:text-orange-200',
    enterprise_sources: 'bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-200',
    structured_output: 'bg-purple-100 text-purple-800 dark:bg-purple-900 dark:text-purple-200',
    manual_workflow: 'bg-amber-100 text-amber-800 dark:bg-amber-900 dark:text-amber-200',
    custom_prompts: 'bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200',
  };

  return (
    <span className={cn('px-1.5 py-0.5 rounded text-xs', colorMap[capability])}>
      {AGENT_CAPABILITY_LABELS[capability]}
    </span>
  );
}

// =============================================================================
// Icons
// =============================================================================

function SearchIcon({ className }: { className?: string }) {
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
      <circle cx="11" cy="11" r="8" />
      <path d="m21 21-4.3-4.3" />
    </svg>
  );
}

function ShieldIcon({ className }: { className?: string }) {
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
      <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10" />
    </svg>
  );
}

function UsersIcon({ className }: { className?: string }) {
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
      <path d="M16 21v-2a4 4 0 0 0-4-4H6a4 4 0 0 0-4 4v2" />
      <circle cx="9" cy="7" r="4" />
      <path d="M22 21v-2a4 4 0 0 0-3-3.87" />
      <path d="M16 3.13a4 4 0 0 1 0 7.75" />
    </svg>
  );
}

function UserIcon({ className }: { className?: string }) {
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
      <path d="M19 21v-2a4 4 0 0 0-4-4H9a4 4 0 0 0-4 4v2" />
      <circle cx="12" cy="7" r="4" />
    </svg>
  );
}

function BotIcon({ className }: { className?: string }) {
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
      <path d="M2 14h2" />
      <path d="M20 14h2" />
      <path d="M15 13v2" />
      <path d="M9 13v2" />
    </svg>
  );
}

function ChevronIcon({ className }: { className?: string }) {
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
      <path d="m6 9 6 6 6-6" />
    </svg>
  );
}

function CheckIcon({ className }: { className?: string }) {
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
      <path d="M20 6 9 17l-5-5" />
    </svg>
  );
}

export default AgentSelector;
