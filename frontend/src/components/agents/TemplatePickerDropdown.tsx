/**
 * TemplatePickerDropdown - Select or create prompt templates inline.
 *
 * Features:
 * - Browse existing templates grouped by ownership (My / Workspace)
 * - "Create New..." inline form with name + content
 * - Auto-selects newly created template
 *
 * Part of 009-custom-agent-config (T045-T046).
 */

import * as React from 'react';
import { cn } from '@/lib/utils';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { useTemplates, useCreateTemplate } from '@/hooks/useTemplates';
import type { TemplateType } from '@/types/templates';

interface TemplatePickerDropdownProps {
  /** Currently selected template ID */
  selectedTemplateId: string | null;
  /** Callback when selection changes */
  onChange: (id: string | null) => void;
  /** Template type to filter by */
  templateType: 'system' | 'synthesis';
  /** Whether the picker is disabled */
  disabled?: boolean;
}

export function TemplatePickerDropdown({
  selectedTemplateId,
  onChange,
  templateType,
  disabled = false,
}: TemplatePickerDropdownProps) {
  const [showCreateForm, setShowCreateForm] = React.useState(false);
  const [newName, setNewName] = React.useState('');
  const [newContent, setNewContent] = React.useState('');
  const [createError, setCreateError] = React.useState<string | null>(null);

  const queryType: TemplateType = templateType === 'synthesis' ? 'synthesis' : 'system';
  const { data: templatesData, isLoading } = useTemplates({ type: queryType });
  const createMutation = useCreateTemplate();

  const templates = templatesData?.templates ?? [];
  const myTemplates = templates.filter((t) => t.visibility === 'private');
  const workspaceTemplates = templates.filter((t) => t.visibility === 'workspace');

  const handleCreate = async () => {
    if (!newName.trim() || !newContent.trim()) {
      setCreateError('Name and content are required');
      return;
    }
    setCreateError(null);

    try {
      const created = await createMutation.mutateAsync({
        name: newName.trim(),
        type: queryType,
        content: newContent.trim(),
        visibility: 'private',
      });
      onChange(created.id);
      setShowCreateForm(false);
      setNewName('');
      setNewContent('');
    } catch (err) {
      setCreateError(err instanceof Error ? err.message : 'Failed to create template');
    }
  };

  const handleCancelCreate = () => {
    setShowCreateForm(false);
    setNewName('');
    setNewContent('');
    setCreateError(null);
  };

  if (showCreateForm) {
    return (
      <div className="space-y-3 rounded-md border border-primary/30 p-3">
        <div className="flex items-center justify-between">
          <h4 className="text-sm font-medium">
            New {templateType === 'synthesis' ? 'Synthesis' : 'System'} Template
          </h4>
          <Button
            type="button"
            variant="ghost"
            size="sm"
            onClick={handleCancelCreate}
            className="h-7 text-xs"
          >
            Cancel
          </Button>
        </div>
        <Input
          value={newName}
          onChange={(e) => setNewName(e.target.value)}
          placeholder="Template name..."
          disabled={createMutation.isPending}
        />
        <textarea
          value={newContent}
          onChange={(e) => setNewContent(e.target.value)}
          placeholder={`Enter your ${templateType} prompt template...`}
          rows={4}
          disabled={createMutation.isPending}
          className={cn(
            'w-full resize-none rounded-md border border-input bg-transparent px-3 py-2 text-sm font-mono',
            'placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring',
            'disabled:cursor-not-allowed disabled:opacity-50'
          )}
        />
        {createError && (
          <p className="text-xs text-destructive">{createError}</p>
        )}
        <Button
          type="button"
          size="sm"
          onClick={handleCreate}
          disabled={createMutation.isPending || !newName.trim() || !newContent.trim()}
        >
          {createMutation.isPending ? 'Creating...' : 'Create & Select'}
        </Button>
      </div>
    );
  }

  return (
    <div className="space-y-2">
      <select
        value={selectedTemplateId || ''}
        onChange={(e) => {
          const val = e.target.value;
          if (val === '__create__') {
            setShowCreateForm(true);
          } else {
            onChange(val || null);
          }
        }}
        disabled={disabled || isLoading}
        className="w-full rounded-md border border-input bg-background px-3 py-2 text-sm"
      >
        <option value="">
          Use default {templateType === 'synthesis' ? 'synthesis' : 'system'} prompt
        </option>
        {myTemplates.length > 0 && (
          <optgroup label="My Templates">
            {myTemplates.map((t) => (
              <option key={t.id} value={t.id}>
                {t.name}
              </option>
            ))}
          </optgroup>
        )}
        {workspaceTemplates.length > 0 && (
          <optgroup label="Workspace Templates">
            {workspaceTemplates.map((t) => (
              <option key={t.id} value={t.id}>
                {t.name}
              </option>
            ))}
          </optgroup>
        )}
        <option value="__create__">+ Create New...</option>
      </select>
      {isLoading && (
        <p className="text-xs text-muted-foreground animate-pulse">Loading templates...</p>
      )}
    </div>
  );
}
