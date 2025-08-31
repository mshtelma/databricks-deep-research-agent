import { create } from 'zustand'
import { devtools } from 'zustand/middleware'
import { ChatMessage, ResearchProgress, ResearchMetadata } from '@/types/chat'

export interface ChatState {
  messages: ChatMessage[]
  isLoading: boolean
  currentStreamingId: string | null
  researchProgress: ResearchProgress
  
  // Actions
  addMessage: (message: Omit<ChatMessage, 'id' | 'timestamp'>) => string
  updateStreamingMessage: (id: string, content: string, metadata?: ResearchMetadata) => void
  setLoading: (loading: boolean) => void
  setResearchProgress: (progress: Partial<ResearchProgress>) => void
  clearChat: () => void
  removeMessage: (id: string) => void
}

export const useChatStore = create<ChatState>()(
  devtools((set, get) => ({
    messages: [],
    isLoading: false,
    currentStreamingId: null,
    researchProgress: {
      currentPhase: 'complete',
      queriesGenerated: 0,
      sourcesFound: 0,
      iterationsComplete: 0
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
    
    clearChat: () => set({ 
      messages: [], 
      isLoading: false, 
      currentStreamingId: null,
      researchProgress: {
        currentPhase: 'complete',
        queriesGenerated: 0,
        sourcesFound: 0,
        iterationsComplete: 0
      }
    }),
    
    removeMessage: (id) => 
      set(state => ({
        messages: state.messages.filter(msg => msg.id !== id)
      }))
  }), {
    name: 'chat-store'
  })
)