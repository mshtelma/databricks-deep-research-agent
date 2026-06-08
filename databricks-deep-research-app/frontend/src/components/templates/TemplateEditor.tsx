/**
 * TemplateEditor - Editor for prompt templates with variable highlighting.
 *
 * Features:
 * - Template content editor with syntax highlighting for {{variable}} patterns
 * - Variable metadata editor showing detected variables
 * - Preview section with sample values
 * - Variable type, required toggle, and default value configuration
 */

import * as React from 'react';
import DOMPurify from 'dompurify';
import { cn } from '@/lib/utils';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Tabs, TabsList, TabsTrigger, TabsContent } from '@/components/ui/tabs';
import { VariableInput } from './VariableInput';
import type {
  Template,
  TemplateVariable,
} from '@/types/templates';
import { extractVariables } from '@/types/templates';

const SANITIZE_CONFIG = { ALLOWED_TAGS: ['span'], ALLOWED_ATTR: ['class'] };

interface TemplateEditorProps {
  /** Template being edited (null for new template) */
  template: Partial<Template> | null;
  /** Callback when template changes */
  onChange: (updates: Partial<Template>) => void;
  /** Callback when saving */
  onSave: () => void;
  /** Whether template is being saved */
  isSaving?: boolean;
  /** Whether in edit mode (vs view mode) */
  isEditing?: boolean;
  /** Additional CSS classes */
  className?: string;
}

export function TemplateEditor({
  template,
  onChange,
  onSave,
  isSaving = false,
  isEditing = true,
  className,
}: TemplateEditorProps) {
  const [activeTab, setActiveTab] = React.useState<'editor' | 'preview'>('editor');
  const [previewValues, setPreviewValues] = React.useState<Record<string, unknown>>({});
  const textareaRef = React.useRef<HTMLTextAreaElement>(null);

  // Extract current state
  const content = template?.content ?? '';
  const variables: TemplateVariable[] = React.useMemo(
    () => template?.variables ?? [],
    [template?.variables],
  );

  // Detect variables from content for preview
  const detectedVariableNames = React.useMemo(() => extractVariables(content), [content]);

  // Merge detected variables with existing metadata for preview
  const mergedVariables = React.useMemo(() => {
    const merged: TemplateVariable[] = [];

    for (const v of variables) {
      if (detectedVariableNames.includes(v.name)) {
        merged.push(v);
      }
    }

    for (const name of detectedVariableNames) {
      if (!merged.find((v) => v.name === name)) {
        merged.push({
          name,
          type: 'string',
          required: false,
        });
      }
    }

    return merged;
  }, [variables, detectedVariableNames]);

  // Update content with highlighting
  const handleContentChange = (newContent: string) => {
    onChange({ content: newContent });
  };

  // Render preview
  const renderPreview = () => {
    let rendered = content;
    for (const [name, value] of Object.entries(previewValues)) {
      const regex = new RegExp(`\\{\\{${name}\\}\\}`, 'g');
      rendered = rendered.replace(regex, String(value ?? `{{${name}}}`));
    }
    // Highlight remaining unsubstituted variables
    rendered = rendered.replace(
      /\{\{(\w+)\}\}/g,
      '<span class="bg-yellow-200 dark:bg-yellow-800 px-1 rounded">{{$1}}</span>'
    );
    return DOMPurify.sanitize(rendered, SANITIZE_CONFIG);
  };

  // Auto-resize textarea
  React.useEffect(() => {
    const textarea = textareaRef.current;
    if (textarea) {
      textarea.style.height = 'auto';
      textarea.style.height = `${Math.max(textarea.scrollHeight, 200)}px`;
    }
  }, [content]);

  return (
    <div className={cn('space-y-4', className)}>
      {/* Template Name */}
      <div>
        <label htmlFor="template-name" className="text-sm font-medium block mb-1.5">
          Template Name
        </label>
        <Input
          id="template-name"
          value={template?.name ?? ''}
          onChange={(e) => onChange({ name: e.target.value })}
          placeholder="Enter template name..."
          disabled={!isEditing || isSaving}
        />
      </div>

      {/* Template Description */}
      <div>
        <label htmlFor="template-description" className="text-sm font-medium block mb-1.5">
          Description
        </label>
        <Input
          id="template-description"
          value={template?.description ?? ''}
          onChange={(e) => onChange({ description: e.target.value })}
          placeholder="Optional description..."
          disabled={!isEditing || isSaving}
        />
      </div>

      {/* Tabs */}
      <Tabs value={activeTab} onValueChange={(v) => setActiveTab(v as typeof activeTab)}>
        <TabsList>
          <TabsTrigger value="editor">
            <CodeIcon className="h-4 w-4 mr-1.5" />
            Content
          </TabsTrigger>
          <TabsTrigger value="preview">
            <EyeIcon className="h-4 w-4 mr-1.5" />
            Preview
          </TabsTrigger>
        </TabsList>

        {/* Editor Tab */}
        <TabsContent value="editor" className="mt-4">
          <div className="relative">
            <HighlightedTextarea
              ref={textareaRef}
              value={content}
              onChange={handleContentChange}
              disabled={!isEditing || isSaving}
              placeholder="Enter template content... Use {{variable_name}} for placeholders"
            />
            <p className="text-xs text-muted-foreground mt-2">
              Use <code className="bg-muted px-1 py-0.5 rounded">{'{{variable_name}}'}</code> syntax
              for variables
            </p>
          </div>
        </TabsContent>

        {/* Preview Tab */}
        <TabsContent value="preview" className="mt-4 space-y-4">
          {mergedVariables.length > 0 && (
            <div className="border rounded-lg p-4 space-y-3">
              <h4 className="text-sm font-medium">Sample Values</h4>
              <VariableInput
                variables={mergedVariables}
                values={previewValues}
                onChange={setPreviewValues}
                disabled={isSaving}
              />
            </div>
          )}

          <div className="border rounded-lg p-4">
            <h4 className="text-sm font-medium mb-2">Rendered Preview</h4>
            <div
              className="prose prose-sm dark:prose-invert max-w-none whitespace-pre-wrap"
              dangerouslySetInnerHTML={{ __html: renderPreview() }}
            />
          </div>
        </TabsContent>
      </Tabs>

      {/* Save Button */}
      {isEditing && (
        <div className="flex justify-end pt-4 border-t">
          <Button
            onClick={onSave}
            disabled={isSaving || !template?.name?.trim() || !content.trim()}
            loading={isSaving}
          >
            {isSaving ? 'Saving...' : 'Save Template'}
          </Button>
        </div>
      )}
    </div>
  );
}

// =============================================================================
// Sub-components
// =============================================================================

interface HighlightedTextareaProps {
  value: string;
  onChange: (value: string) => void;
  disabled?: boolean;
  placeholder?: string;
}

const HighlightedTextarea = React.forwardRef<HTMLTextAreaElement, HighlightedTextareaProps>(
  ({ value, onChange, disabled, placeholder }, ref) => {
    // Create highlighted version for display
    const highlightedContent = React.useMemo(() => {
      const raw = value.replace(
        /(\{\{\w+\}\})/g,
        '<span class="bg-blue-100 dark:bg-blue-900 text-blue-700 dark:text-blue-300 px-0.5 rounded">$1</span>'
      );
      return DOMPurify.sanitize(raw, SANITIZE_CONFIG);
    }, [value]);

    return (
      <div className="relative">
        {/* Highlighted overlay (not interactive) */}
        <div
          className={cn(
            'absolute inset-0 pointer-events-none overflow-hidden',
            'w-full rounded-md border border-transparent px-3 py-2 text-sm',
            'whitespace-pre-wrap break-words'
          )}
          aria-hidden="true"
          dangerouslySetInnerHTML={{ __html: highlightedContent || '&nbsp;' }}
        />

        {/* Actual textarea (transparent text) */}
        <textarea
          ref={ref}
          value={value}
          onChange={(e) => onChange(e.target.value)}
          disabled={disabled}
          placeholder={placeholder}
          className={cn(
            'w-full min-h-[200px] rounded-md border border-input bg-transparent px-3 py-2 text-sm',
            'placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring',
            'disabled:cursor-not-allowed disabled:opacity-50 resize-none',
            // Make text visible since overlay doesn't work well
            'text-foreground'
          )}
          style={{ caretColor: 'auto' }}
        />
      </div>
    );
  }
);

HighlightedTextarea.displayName = 'HighlightedTextarea';

// =============================================================================
// Icons
// =============================================================================

function CodeIcon({ className }: { className?: string }) {
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
      <polyline points="16 18 22 12 16 6" />
      <polyline points="8 6 2 12 8 18" />
    </svg>
  );
}

function EyeIcon({ className }: { className?: string }) {
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
      <path d="M2 12s3-7 10-7 10 7 10 7-3 7-10 7-10-7-10-7Z" />
      <circle cx="12" cy="12" r="3" />
    </svg>
  );
}

export default TemplateEditor;
