/**
 * TemplateLibrary - Browse and select from available templates.
 *
 * Features:
 * - Filter tabs by type (All, System, Step, Synthesis, Query)
 * - Search by name/tags
 * - Grid or list of template cards
 * - Owner indicator (system/plugin/user)
 * - "Use Template" and "Edit" buttons
 * - Create new template button
 */

import * as React from 'react';
import { cn } from '@/lib/utils';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Badge } from '@/components/ui/badge';
import { Tabs, TabsList, TabsTrigger, TabsContent } from '@/components/ui/tabs';
import { Card, CardHeader, CardTitle, CardDescription, CardContent, CardFooter } from '@/components/ui/card';
import { ScrollArea } from '@/components/ui/scroll-area';
import { useGroupedTemplates } from '@/hooks/useTemplates';
import type { Template, TemplateType, TemplateOrigin } from '@/types/templates';
import {
  getTemplateTypeLabel,
  getTemplateOriginLabel,
  TEMPLATE_TYPE_COLORS,
} from '@/types/templates';

interface TemplateLibraryProps {
  /** Callback when a template is selected */
  onSelectTemplate?: (template: Template) => void;
  /** Currently selected template ID */
  selectedTemplateId?: string;
  /** Callback to create a new template */
  onCreateTemplate?: () => void;
  /** Callback to edit a template */
  onEditTemplate?: (template: Template) => void;
  /** Whether to show the create button */
  showCreateButton?: boolean;
  /** Filter to specific template types */
  allowedTypes?: TemplateType[];
  /** Additional CSS classes */
  className?: string;
  /** Maximum height for the library */
  maxHeight?: string;
}

type FilterTab = 'all' | TemplateType;

export function TemplateLibrary({
  onSelectTemplate,
  selectedTemplateId,
  onCreateTemplate,
  onEditTemplate,
  showCreateButton = true,
  allowedTypes,
  className,
  maxHeight = '600px',
}: TemplateLibraryProps) {
  const [activeTab, setActiveTab] = React.useState<FilterTab>('all');
  const [searchQuery, setSearchQuery] = React.useState('');

  const { grouped, templates, isLoading, error, userTemplates, workspaceTemplates } =
    useGroupedTemplates();

  // Get filtered templates based on tab and search
  const filteredTemplates = React.useMemo(() => {
    let filtered: Template[] = [];

    if (activeTab === 'all') {
      filtered = templates;
    } else {
      filtered = grouped[activeTab] || [];
    }

    // Apply allowed types filter
    if (allowedTypes && allowedTypes.length > 0) {
      filtered = filtered.filter((t) => allowedTypes.includes(t.type));
    }

    // Apply search filter
    if (searchQuery.trim()) {
      const query = searchQuery.toLowerCase();
      filtered = filtered.filter(
        (t) =>
          t.name.toLowerCase().includes(query) ||
          t.description?.toLowerCase().includes(query) ||
          t.tags.some((tag) => tag.toLowerCase().includes(query))
      );
    }

    return filtered;
  }, [activeTab, templates, grouped, searchQuery, allowedTypes]);

  // Determine which tabs to show
  const availableTabs = React.useMemo(() => {
    const tabs: FilterTab[] = ['all'];
    const types: TemplateType[] = ['system', 'step', 'synthesis', 'query'];

    for (const type of types) {
      if (!allowedTypes || allowedTypes.length === 0 || allowedTypes.includes(type)) {
        tabs.push(type);
      }
    }

    return tabs;
  }, [allowedTypes]);

  if (isLoading) {
    return (
      <div className={cn('flex items-center justify-center p-8', className)}>
        <div className="text-sm text-muted-foreground">Loading templates...</div>
      </div>
    );
  }

  if (error) {
    return (
      <div className={cn('flex items-center justify-center p-8', className)}>
        <div className="text-sm text-destructive">Failed to load templates</div>
      </div>
    );
  }

  return (
    <div className={cn('flex flex-col', className)}>
      {/* Header with search and create */}
      <div className="p-4 border-b space-y-4">
        <div className="flex items-center justify-between">
          <div>
            <h3 className="font-semibold">Template Library</h3>
            <p className="text-xs text-muted-foreground">
              {templates.length} templates ({userTemplates} custom, {workspaceTemplates} workspace)
            </p>
          </div>
          {showCreateButton && onCreateTemplate && (
            <Button onClick={onCreateTemplate} size="sm">
              <PlusIcon className="h-4 w-4 mr-1.5" />
              New Template
            </Button>
          )}
        </div>

        {/* Search */}
        <div className="relative">
          <SearchIcon className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
          <Input
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            placeholder="Search templates by name or tags..."
            className="pl-9"
          />
        </div>
      </div>

      {/* Type filter tabs */}
      <Tabs value={activeTab} onValueChange={(v) => setActiveTab(v as FilterTab)}>
        <div className="px-4 pt-4">
          <TabsList className="w-full justify-start">
            {availableTabs.map((tab) => (
              <TabsTrigger key={tab} value={tab} className="text-xs">
                {tab === 'all' ? 'All' : getTemplateTypeLabel(tab)}
                <span className="ml-1.5 text-muted-foreground">
                  ({tab === 'all' ? templates.length : grouped[tab]?.length ?? 0})
                </span>
              </TabsTrigger>
            ))}
          </TabsList>
        </div>

        {/* Templates grid */}
        <TabsContent value={activeTab} className="mt-0">
          <ScrollArea style={{ maxHeight }}>
            <div className="p-4">
              {filteredTemplates.length === 0 ? (
                <div className="text-center py-12 text-muted-foreground">
                  <FileTextIcon className="h-10 w-10 mx-auto mb-3 opacity-50" />
                  <p className="text-sm">No templates found</p>
                  {searchQuery && (
                    <p className="text-xs mt-1">Try adjusting your search</p>
                  )}
                </div>
              ) : (
                <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
                  {filteredTemplates.map((template) => (
                    <TemplateCard
                      key={template.id}
                      template={template}
                      isSelected={template.id === selectedTemplateId}
                      onSelect={() => onSelectTemplate?.(template)}
                      onEdit={() => onEditTemplate?.(template)}
                      canEdit={(template.origin ?? 'user') === 'user'}
                    />
                  ))}
                </div>
              )}
            </div>
          </ScrollArea>
        </TabsContent>
      </Tabs>
    </div>
  );
}

// =============================================================================
// Template Card Component
// =============================================================================

interface TemplateCardProps {
  template: Template;
  isSelected: boolean;
  onSelect: () => void;
  onEdit: () => void;
  canEdit: boolean;
}

function TemplateCard({
  template,
  isSelected,
  onSelect,
  onEdit,
  canEdit,
}: TemplateCardProps) {
  const typeColor = TEMPLATE_TYPE_COLORS[template.type] || 'gray';

  return (
    <Card
      className={cn(
        'cursor-pointer transition-all hover:shadow-md',
        isSelected && 'ring-2 ring-primary ring-offset-2'
      )}
      onClick={onSelect}
    >
      <CardHeader className="pb-2">
        <div className="flex items-start justify-between gap-2">
          <CardTitle className="text-sm font-medium line-clamp-1">
            {template.name}
          </CardTitle>
          <TypeBadge type={template.type} color={typeColor} />
        </div>
        <CardDescription className="text-xs line-clamp-2">
          {template.description || 'No description'}
        </CardDescription>
      </CardHeader>

      <CardContent className="pb-2">
        {/* Origin indicator */}
        <div className="flex items-center gap-2 mb-2">
          <OriginBadge origin={template.origin ?? 'user'} />
          {template.isDefault && (
            <Badge variant="outline" className="text-xs py-0">
              Default
            </Badge>
          )}
        </div>

        {/* Tags */}
        {template.tags.length > 0 && (
          <div className="flex flex-wrap gap-1">
            {template.tags.slice(0, 3).map((tag) => (
              <span
                key={tag}
                className="text-xs px-1.5 py-0.5 rounded bg-muted text-muted-foreground"
              >
                {tag}
              </span>
            ))}
            {template.tags.length > 3 && (
              <span className="text-xs text-muted-foreground">
                +{template.tags.length - 3}
              </span>
            )}
          </div>
        )}

        {/* Variable count */}
        <div className="text-xs text-muted-foreground mt-2">
          {(template.variables ?? []).length} variable(s)
        </div>
      </CardContent>

      <CardFooter className="pt-2 gap-2">
        <Button
          variant="default"
          size="sm"
          className="flex-1"
          onClick={(e) => {
            e.stopPropagation();
            onSelect();
          }}
        >
          Use Template
        </Button>
        {canEdit && (
          <Button
            variant="outline"
            size="sm"
            onClick={(e) => {
              e.stopPropagation();
              onEdit();
            }}
          >
            Edit
          </Button>
        )}
      </CardFooter>
    </Card>
  );
}

// =============================================================================
// Helper Components
// =============================================================================

function TypeBadge({ type, color }: { type: TemplateType; color: string }) {
  const colorClasses: Record<string, string> = {
    blue: 'bg-blue-100 text-blue-700 dark:bg-blue-900 dark:text-blue-300',
    green: 'bg-green-100 text-green-700 dark:bg-green-900 dark:text-green-300',
    purple: 'bg-purple-100 text-purple-700 dark:bg-purple-900 dark:text-purple-300',
    orange: 'bg-orange-100 text-orange-700 dark:bg-orange-900 dark:text-orange-300',
    gray: 'bg-gray-100 text-gray-700 dark:bg-gray-800 dark:text-gray-300',
  };

  return (
    <span
      className={cn(
        'text-xs px-1.5 py-0.5 rounded-full whitespace-nowrap',
        colorClasses[color] || colorClasses.gray
      )}
    >
      {getTemplateTypeLabel(type)}
    </span>
  );
}

function OriginBadge({ origin }: { origin: TemplateOrigin }) {
  const originConfig: Record<TemplateOrigin, { icon: React.ReactNode; className: string }> = {
    system: {
      icon: <LockIcon className="h-3 w-3" />,
      className: 'text-muted-foreground',
    },
    plugin: {
      icon: <PlugIcon className="h-3 w-3" />,
      className: 'text-purple-600 dark:text-purple-400',
    },
    user: {
      icon: <UserIcon className="h-3 w-3" />,
      className: 'text-blue-600 dark:text-blue-400',
    },
  };

  const config = originConfig[origin];

  return (
    <div className={cn('flex items-center gap-1 text-xs', config.className)}>
      {config.icon}
      <span>{getTemplateOriginLabel(origin)}</span>
    </div>
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

function PlusIcon({ className }: { className?: string }) {
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
      <path d="M5 12h14" />
      <path d="M12 5v14" />
    </svg>
  );
}

function FileTextIcon({ className }: { className?: string }) {
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
      <path d="M14.5 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V7.5L14.5 2z" />
      <polyline points="14 2 14 8 20 8" />
      <line x1="16" x2="8" y1="13" y2="13" />
      <line x1="16" x2="8" y1="17" y2="17" />
      <line x1="10" x2="8" y1="9" y2="9" />
    </svg>
  );
}

function LockIcon({ className }: { className?: string }) {
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
      <rect width="18" height="11" x="3" y="11" rx="2" ry="2" />
      <path d="M7 11V7a5 5 0 0 1 10 0v4" />
    </svg>
  );
}

function PlugIcon({ className }: { className?: string }) {
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
      <path d="M12 22v-5" />
      <path d="M9 8V2" />
      <path d="M15 8V2" />
      <path d="M18 8v5a4 4 0 0 1-4 4h-4a4 4 0 0 1-4-4V8Z" />
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

export default TemplateLibrary;
