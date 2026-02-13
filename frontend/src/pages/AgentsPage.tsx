/**
 * AgentsPage - Management page for custom research agents.
 *
 * Features:
 * - Browse existing agents with AgentSelector
 * - Create new agents with AgentBuilder
 * - Edit existing agents
 * - Delete agents
 */

import * as React from 'react'
import { useNavigate } from 'react-router-dom'
import { Button } from '@/components/ui/button'
import { AgentSelector } from '@/components/agents/AgentSelector'
import { AgentBuilder } from '@/components/agents/AgentBuilder'
import {
  useCustomAgents,
  useCustomAgent,
  useCreateAgent,
  useUpdateAgent,
  useDeleteAgent,
  useAgentPresetSteps,
} from '@/hooks/useCustomAgents'
import type { CustomAgent, CustomAgentSummary, CreateCustomAgentRequest, UpdateCustomAgentRequest, PresetStep } from '@/types/customAgents'

type PageMode = 'browse' | 'create' | 'edit'

export function AgentsPage() {
  const navigate = useNavigate()
  const { data: agentsData, isLoading, error } = useCustomAgents({ include_system: true })
  const createAgent = useCreateAgent()
  const updateAgent = useUpdateAgent()
  const deleteAgent = useDeleteAgent()

  const [mode, setMode] = React.useState<PageMode>('browse')
  const [selectedAgentId, setSelectedAgentId] = React.useState<string | null>(null)
  const [deleteConfirmId, setDeleteConfirmId] = React.useState<string | null>(null)

  // Fetch full agent details when editing
  const { data: selectedAgentFull } = useCustomAgent(
    mode === 'edit' && selectedAgentId ? selectedAgentId : undefined
  )
  const { data: presetStepsData } = useAgentPresetSteps(
    mode === 'edit' && selectedAgentId ? selectedAgentId : undefined
  )

  const handleSelectAgent = (agent: CustomAgentSummary) => {
    setSelectedAgentId(agent.id)
    setMode('edit')
  }

  const handleSave = async (data: CreateCustomAgentRequest | UpdateCustomAgentRequest, _steps?: PresetStep[]) => {
    try {
      if (mode === 'create') {
        await createAgent.mutateAsync(data as CreateCustomAgentRequest)
      } else if (mode === 'edit' && selectedAgentId) {
        await updateAgent.mutateAsync({ agentId: selectedAgentId, data: data as UpdateCustomAgentRequest })
      }
      setMode('browse')
      setSelectedAgentId(null)
    } catch {
      // Error handled by mutation
    }
  }

  const handleDelete = async (agentId: string) => {
    try {
      await deleteAgent.mutateAsync(agentId)
      setDeleteConfirmId(null)
      if (selectedAgentId === agentId) {
        setMode('browse')
        setSelectedAgentId(null)
      }
    } catch {
      // Error handled by mutation
    }
  }

  const handleCancel = () => {
    setMode('browse')
    setSelectedAgentId(null)
  }

  if (isLoading) {
    return (
      <div className="flex-1 flex items-center justify-center">
        <div className="text-muted-foreground">Loading agents...</div>
      </div>
    )
  }

  if (error) {
    return (
      <div className="flex-1 flex items-center justify-center">
        <div className="text-destructive">Failed to load agents</div>
      </div>
    )
  }

  return (
    <div className="flex-1 overflow-auto">
      <div className="max-w-4xl mx-auto p-6 space-y-6">
        {/* Header */}
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-bold">Custom Agents</h1>
            <p className="text-muted-foreground mt-1">
              Create and manage custom research agents with tailored configurations.
            </p>
          </div>
          <div className="flex items-center gap-2">
            {mode === 'browse' && (
              <Button onClick={() => setMode('create')}>
                Create Agent
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
              <p className="text-2xl font-bold">{agentsData?.agents?.length ?? 0}</p>
              <p className="text-xs text-muted-foreground">Total Agents</p>
            </div>
            <div className="border rounded-lg p-4">
              <p className="text-2xl font-bold">
                {agentsData?.agents?.filter(a => a.visibility === 'workspace').length ?? 0}
              </p>
              <p className="text-xs text-muted-foreground">Workspace Agents</p>
            </div>
          </div>
        )}

        {/* Content */}
        {mode === 'browse' && (
          <div className="space-y-4">
            <AgentSelector
              onSelectAgent={handleSelectAgent}
              selectedAgentId={selectedAgentId}
            />

            {/* Delete confirmation */}
            {deleteConfirmId && (
              <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
                <div className="bg-background rounded-lg p-6 max-w-sm mx-4">
                  <p className="font-medium mb-4">Delete this agent?</p>
                  <div className="flex gap-2 justify-end">
                    <Button variant="outline" size="sm" onClick={() => setDeleteConfirmId(null)}>
                      Cancel
                    </Button>
                    <Button
                      variant="destructive"
                      size="sm"
                      onClick={() => handleDelete(deleteConfirmId)}
                      disabled={deleteAgent.isPending}
                    >
                      {deleteAgent.isPending ? 'Deleting...' : 'Delete'}
                    </Button>
                  </div>
                </div>
              </div>
            )}
          </div>
        )}

        {(mode === 'create' || mode === 'edit') && (
          <AgentBuilder
            agent={mode === 'edit' ? (selectedAgentFull as CustomAgent | undefined) : undefined}
            presetSteps={presetStepsData?.steps}
            onSave={handleSave}
            onCancel={handleCancel}
            isLoading={createAgent.isPending || updateAgent.isPending}
          />
        )}
      </div>
    </div>
  )
}

export default AgentsPage
