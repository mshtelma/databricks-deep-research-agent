import { create } from 'zustand'
import { devtools } from 'zustand/middleware'
import { ChatMessage, ResearchProgress, ResearchMetadata, IntermediateEvent } from '@/types/chat'

export interface ChatState {
  messages: ChatMessage[]
  isLoading: boolean
  currentStreamingId: string | null
  researchProgress: ResearchProgress
  intermediateEvents: IntermediateEvent[]
  showThoughts: boolean
  
  // Event history for current message being streamed
  currentMessageEvents: IntermediateEvent[]
  // Complete event history for all messages
  eventHistory: Map<string, IntermediateEvent[]>
  // Filter settings for event feed
  eventFilters: {
    categories: string[]
    searchQuery: string
    minPriority: number
  }

  // Actions
  addMessage: (message: Omit<ChatMessage, 'id' | 'timestamp'>) => string
  updateStreamingMessage: (id: string, content: string, metadata?: ResearchMetadata) => void
  setLoading: (loading: boolean) => void
  setResearchProgress: (progress: Partial<ResearchProgress>) => void
  addIntermediateEvent: (event: IntermediateEvent) => void
  addIntermediateEvents: (events: IntermediateEvent[]) => void
  clearIntermediateEvents: () => void
  setShowThoughts: (show: boolean) => void
  clearChat: () => void
  removeMessage: (id: string) => void
  
  // New event management actions
  finalizeMessageEvents: (messageId: string) => void
  getEventsForMessage: (messageId: string) => IntermediateEvent[]
  setEventFilters: (filters: Partial<ChatState['eventFilters']>) => void
  
  // Progress utilities
  updateProgressWithETA: (progress: Partial<ResearchProgress>) => void
}

export const useChatStore = create<ChatState>()(
  devtools((set, get) => ({
    messages: [],
    isLoading: false,
    currentStreamingId: null,
    intermediateEvents: [],
    currentMessageEvents: [],
    eventHistory: new Map(),
    eventFilters: {
      categories: [],
      searchQuery: '',
      minPriority: 0
    },
    showThoughts: false,
    researchProgress: {
      currentPhase: 'complete',
      queriesGenerated: 0,
      sourcesFound: 0,
      iterationsComplete: 0,
      progressPercentage: 0,
      elapsedTime: 0,
      currentNode: '',
      vectorResultsCount: 0,
      estimatedTimeRemaining: 0,
      currentOperation: '',
      
      // Multi-agent fields
      currentAgent: '',
      agentHandoffs: [],
      planIterations: 0,
      factualityScore: 0,
      researchQualityScore: 0,
      coverageScore: 0
    },

    addMessage: (message) => {
      const newMessage: ChatMessage = {
        ...message,
        id: crypto.randomUUID(),
        timestamp: new Date()
      }
      set(state => ({
        messages: [...state.messages, newMessage]
      }))
      return newMessage.id
    },

    updateStreamingMessage: (id, content, metadata) => {
      set(state => ({
        messages: state.messages.map(msg =>
          msg.id === id
            ? { ...msg, content, metadata, isStreaming: !metadata }
            : msg
        )
      }))
    },

    setLoading: (loading) => set({
      isLoading: loading,
      currentStreamingId: loading ? get().currentStreamingId : null
    }),

    setResearchProgress: (progress) =>
      set(state => ({
        researchProgress: { ...state.researchProgress, ...progress }
      })),

    addIntermediateEvent: (event) =>
      set(state => ({
        intermediateEvents: [...state.intermediateEvents, event],
        currentMessageEvents: [...state.currentMessageEvents, event]
      })),

    addIntermediateEvents: (events) =>
      set(state => ({
        intermediateEvents: [...state.intermediateEvents, ...events],
        currentMessageEvents: [...state.currentMessageEvents, ...events]
      })),

    clearIntermediateEvents: () =>
      set({ 
        intermediateEvents: [],
        currentMessageEvents: []
      }),

    setShowThoughts: (show) =>
      set({ showThoughts: show }),

    clearChat: () => set({
      messages: [],
      isLoading: false,
      currentStreamingId: null,
      intermediateEvents: [],
      currentMessageEvents: [],
      eventHistory: new Map(),
      researchProgress: {
        currentPhase: 'complete',
        queriesGenerated: 0,
        sourcesFound: 0,
        iterationsComplete: 0,
        progressPercentage: 0,
        elapsedTime: 0,
        currentNode: '',
        vectorResultsCount: 0,
        estimatedTimeRemaining: 0,
        currentOperation: '',
        
        // Multi-agent fields
        currentAgent: '',
        agentHandoffs: [],
        planIterations: 0,
        factualityScore: 0,
        researchQualityScore: 0,
        coverageScore: 0
      }
    }),

    removeMessage: (id) =>
      set(state => ({
        messages: state.messages.filter(msg => msg.id !== id)
      })),

    finalizeMessageEvents: (messageId) =>
      set(state => {
        const newEventHistory = new Map(state.eventHistory)
        newEventHistory.set(messageId, [...state.currentMessageEvents])
        return {
          eventHistory: newEventHistory,
          currentMessageEvents: []
        }
      }),

    getEventsForMessage: (messageId) => {
      const state = get()
      return state.eventHistory.get(messageId) || []
    },

    setEventFilters: (filters) =>
      set(state => ({
        eventFilters: { ...state.eventFilters, ...filters }
      })),

    updateProgressWithETA: (progress) => {
      set(state => {
        const currentProgress = state.researchProgress
        const newProgress = { ...currentProgress, ...progress }
        
        // Calculate ETA if we have progress percentage and elapsed time
        if (newProgress.progressPercentage && newProgress.elapsedTime && newProgress.progressPercentage > 0) {
          const remainingProgress = 100 - newProgress.progressPercentage
          const progressRate = newProgress.progressPercentage / newProgress.elapsedTime // progress per second
          newProgress.estimatedTimeRemaining = remainingProgress / progressRate
        }
        
        return {
          researchProgress: newProgress
        }
      })
    }
  }), {
    name: 'chat-store'
  })
)