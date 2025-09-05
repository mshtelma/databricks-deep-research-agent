import { create } from 'zustand'
import { devtools, persist } from 'zustand/middleware'
import { AgentConfig } from '@/types/chat'

export interface SettingsState {
  agentConfig: AgentConfig
  isSettingsOpen: boolean

  // Actions
  updateAgentConfig: (config: Partial<AgentConfig>) => void
  resetToDefaults: () => void
  setSettingsOpen: (open: boolean) => void
}

const defaultAgentConfig: AgentConfig = {
  reportStyle: 'professional',
  verificationLevel: 'moderate',
  enableIterativePlanning: true,
  enableBackgroundInvestigation: true,
  autoAcceptPlan: true,
  maxPlanIterations: 3
}

export const useSettingsStore = create<SettingsState>()(
  devtools(
    persist(
      (set, get) => ({
        agentConfig: defaultAgentConfig,
        isSettingsOpen: false,

        updateAgentConfig: (config) =>
          set((state) => ({
            agentConfig: { ...state.agentConfig, ...config }
          })),

        resetToDefaults: () =>
          set({ agentConfig: defaultAgentConfig }),

        setSettingsOpen: (open) =>
          set({ isSettingsOpen: open })
      }),
      {
        name: 'agent-settings',
        // Only persist the agentConfig, not UI state
        partialize: (state) => ({ agentConfig: state.agentConfig })
      }
    ),
    { name: 'settings-store' }
  )
)