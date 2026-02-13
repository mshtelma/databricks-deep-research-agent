/**
 * TemplatesPage - Management page for prompt templates.
 *
 * Features:
 * - Browse existing templates with TemplateLibrary
 * - Create new templates with TemplateEditor
 * - Edit existing templates
 * - Delete templates
 */

import * as React from 'react'
import { useNavigate } from 'react-router-dom'
import { Button } from '@/components/ui/button'
import { TemplateLibrary } from '@/components/templates/TemplateLibrary'
import { TemplateEditor } from '@/components/templates/TemplateEditor'
import {
  useTemplates,
  useCreateTemplate,
  useUpdateTemplate,
} from '@/hooks/useTemplates'
import type { Template, CreateTemplateRequest, UpdateTemplateRequest } from '@/types/templates'

type PageMode = 'browse' | 'create' | 'edit'

export function TemplatesPage() {
  const navigate = useNavigate()
  const { data: templatesData, isLoading, error } = useTemplates()
  const createTemplate = useCreateTemplate()
  const updateTemplate = useUpdateTemplate()

  const [mode, setMode] = React.useState<PageMode>('browse')
  const [selectedTemplate, setSelectedTemplate] = React.useState<Template | null>(null)
  const [editDraft, setEditDraft] = React.useState<Partial<Template> | null>(null)

  const handleSelectTemplate = (template: Template) => {
    setSelectedTemplate(template)
    setEditDraft(template)
    setMode('edit')
  }

  const handleCreateNew = () => {
    setSelectedTemplate(null)
    setEditDraft({
      name: '',
      content: '',
      type: 'system',
      visibility: 'private',
      description: '',
    })
    setMode('create')
  }

  const handleSave = async () => {
    if (!editDraft) return

    try {
      if (mode === 'create') {
        await createTemplate.mutateAsync(editDraft as CreateTemplateRequest)
      } else if (mode === 'edit' && selectedTemplate) {
        await updateTemplate.mutateAsync({
          templateId: selectedTemplate.id,
          data: editDraft as UpdateTemplateRequest,
        })
      }
      setMode('browse')
      setSelectedTemplate(null)
      setEditDraft(null)
    } catch {
      // Error handled by mutation
    }
  }

  const handleCancel = () => {
    setMode('browse')
    setSelectedTemplate(null)
    setEditDraft(null)
  }

  if (isLoading) {
    return (
      <div className="flex-1 flex items-center justify-center">
        <div className="text-muted-foreground">Loading templates...</div>
      </div>
    )
  }

  if (error) {
    return (
      <div className="flex-1 flex items-center justify-center">
        <div className="text-destructive">Failed to load templates</div>
      </div>
    )
  }

  return (
    <div className="flex-1 overflow-auto">
      <div className="max-w-4xl mx-auto p-6 space-y-6">
        {/* Header */}
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-bold">Prompt Templates</h1>
            <p className="text-muted-foreground mt-1">
              Create and manage reusable prompt templates for research agents.
            </p>
          </div>
          <div className="flex items-center gap-2">
            {mode === 'browse' && (
              <Button onClick={handleCreateNew}>
                Create Template
              </Button>
            )}
            <Button variant="outline" onClick={() => navigate('/chat')}>
              Back to Chat
            </Button>
          </div>
        </div>

        {/* Stats */}
        {mode === 'browse' && (
          <div className="grid grid-cols-2 gap-4">
            <div className="border rounded-lg p-4">
              <p className="text-2xl font-bold">{templatesData?.templates?.length ?? 0}</p>
              <p className="text-xs text-muted-foreground">Total Templates</p>
            </div>
            <div className="border rounded-lg p-4">
              <p className="text-2xl font-bold">
                {templatesData?.templates?.filter((t: Template) => t.visibility === 'workspace').length ?? 0}
              </p>
              <p className="text-xs text-muted-foreground">Workspace Templates</p>
            </div>
          </div>
        )}

        {/* Content */}
        {mode === 'browse' && (
          <TemplateLibrary
            onSelectTemplate={handleSelectTemplate}
            selectedTemplateId={selectedTemplate?.id}
            onCreateTemplate={handleCreateNew}
            onEditTemplate={handleSelectTemplate}
            showCreateButton={false}
          />
        )}

        {(mode === 'create' || mode === 'edit') && (
          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <h2 className="text-lg font-semibold">
                {mode === 'create' ? 'Create Template' : 'Edit Template'}
              </h2>
              <Button variant="ghost" size="sm" onClick={handleCancel}>
                Cancel
              </Button>
            </div>
            <TemplateEditor
              template={editDraft}
              onChange={(updates) => setEditDraft(prev => prev ? { ...prev, ...updates } : updates)}
              onSave={handleSave}
              isSaving={createTemplate.isPending || updateTemplate.isPending}
              isEditing={true}
            />
          </div>
        )}
      </div>
    </div>
  )
}

export default TemplatesPage
