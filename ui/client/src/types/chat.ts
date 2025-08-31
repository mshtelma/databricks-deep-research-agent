export interface ChatMessage {
  id: string
  role: 'user' | 'assistant'
  content: string
  timestamp: Date
  metadata?: ResearchMetadata
  isStreaming?: boolean
}

export interface ResearchMetadata {
  searchQueries: string[]
  sources: Array<{
    url: string
    title: string
    relevanceScore?: number
    snippet?: string
  }>
  researchIterations: number
  confidenceScore?: number
  reasoningSteps: string[]
}

export interface ResearchProgress {
  currentPhase: 'querying' | 'searching' | 'analyzing' | 'synthesizing' | 'complete'
  queriesGenerated: number
  sourcesFound: number
  iterationsComplete: number
}

export interface StreamEvent {
  type: 'stream_start' | 'content_delta' | 'research_update' | 'message_complete' | 'stream_end' | 'error'
  content?: string
  metadata?: ResearchMetadata
  error?: string
}

export interface ChatRequest {
  messages: Array<{ role: string; content: string }>
  config?: {
    maxTokens?: number
    temperature?: number
    researchMode?: 'quick' | 'standard' | 'deep'
  }
}

export interface ChatResponse {
  message: { role: string; content: string }
  metadata?: ResearchMetadata
}