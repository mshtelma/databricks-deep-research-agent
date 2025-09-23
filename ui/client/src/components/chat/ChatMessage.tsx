import { useState } from 'react'
import { format } from 'date-fns'
import { User, Bot, ChevronDown, ChevronUp, Copy, Check, BarChart } from 'lucide-react'
import { Card, CardContent } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from '@/components/ui/collapsible'
import { ChatMessage as ChatMessageType } from '@/types/chat'
import { MarkdownRenderer } from './MarkdownRenderer'
import { WorkflowVisualizer } from './WorkflowVisualizer'
import { PlanViewer } from './PlanViewer'
import { GroundingReport } from './GroundingReport'
import { StreamingProgress } from './StreamingProgress'
import { SourcesPanel } from './SourcesPanel'
import { ResearchEventsPanel } from './ResearchEventsPanel'
import { useChatStore } from '@/stores/chatStore'
import { cn } from '@/lib/utils'

interface ChatMessageProps {
  message: ChatMessageType
}

export function ChatMessage({ message }: ChatMessageProps) {
  const [multiAgentExpanded, setMultiAgentExpanded] = useState(false)
  const [copied, setCopied] = useState(false)
  const isUser = message.role === 'user'
  
  // Get intermediate events for inline progress display
  const { intermediateEvents, currentStreamingId } = useChatStore()
  const isActivelyStreaming = message.isStreaming && currentStreamingId === message.id
  
  const handleCopy = async () => {
    await navigator.clipboard.writeText(message.content)
    setCopied(true)
    setTimeout(() => setCopied(false), 2000)
  }
  
  return (
    <div 
      className={cn("flex gap-3", isUser ? "justify-end" : "justify-start")}
      data-testid="chat-message"
      data-streaming={message.isStreaming ? "true" : "false"}
      data-role={message.role}
    >
      {/* Assistant Avatar */}
      {!isUser && (
        <div className="w-8 h-8 bg-gradient-to-r from-blue-500 to-purple-600 rounded-full flex items-center justify-center flex-shrink-0">
          <Bot className="w-4 h-4 text-white" />
        </div>
      )}
      
      {/* Message Content */}
      <div className={cn("max-w-3xl flex-1", isUser ? "order-first" : "")}>
        <Card className={cn(
          "shadow-sm hover:shadow-md transition-shadow",
          isUser 
            ? "bg-blue-600 text-white border-blue-600" 
            : "bg-white dark:bg-gray-800 border-gray-200 dark:border-gray-700"
        )}>
          <CardContent className="p-4">
            {/* Message Header */}
            <div className="flex items-center justify-between mb-3">
              <div className="flex items-center gap-2">
                {isUser && <User className="w-4 h-4" />}
                <span className={cn(
                  "text-xs",
                  isUser ? "text-blue-100" : "text-gray-500 dark:text-gray-400"
                )}>
                  {format(message.timestamp, 'HH:mm')}
                </span>
                {message.isStreaming && (
                  <div className="flex items-center gap-1">
                    <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse" />
                    <span className={cn(
                      "text-xs",
                      isUser ? "text-blue-100" : "text-gray-500 dark:text-gray-400"
                    )}>
                      Thinking...
                    </span>
                  </div>
                )}
              </div>
              
              <div className="flex items-center gap-2">
                {/* Enhanced Research Metadata Badges */}
                {message.metadata && !isUser && (
                  <div className="flex items-center gap-1">
                    {message.metadata.researchIterations > 0 && (
                      <Badge variant="secondary" className="text-xs">
                        {message.metadata.researchIterations} iterations
                      </Badge>
                    )}
                    {message.metadata.sources.length > 0 && (
                      <Badge variant="secondary" className="text-xs">
                        {message.metadata.sources.length} sources
                      </Badge>
                    )}
                    {message.metadata.confidenceScore && (
                      <Badge variant="secondary" className="text-xs">
                        {Math.round(message.metadata.confidenceScore * 100)}% confidence
                      </Badge>
                    )}
                    {/* Multi-agent badges */}
                    {message.metadata.currentAgent && (
                      <Badge variant="outline" className="text-xs">
                        🤖 {message.metadata.currentAgent}
                      </Badge>
                    )}
                    {message.metadata.factualityScore && (
                      <Badge variant="secondary" className="text-xs">
                        🎯 {Math.round(message.metadata.factualityScore * 100)}% factual
                      </Badge>
                    )}
                    {message.metadata.reportStyle && (
                      <Badge variant="outline" className="text-xs capitalize">
                        📄 {message.metadata.reportStyle}
                      </Badge>
                    )}
                  </div>
                )}
                
                {/* Copy Button */}
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={handleCopy}
                  className={cn(
                    "h-6 w-6 p-0 opacity-50 hover:opacity-100 transition-opacity",
                    isUser ? "text-blue-100 hover:text-white" : ""
                  )}
                >
                  {copied ? <Check className="h-3 w-3" /> : <Copy className="h-3 w-3" />}
                </Button>
              </div>
            </div>
            
            {/* Message Content */}
            <div className="space-y-3">
              {isUser ? (
                <p className="text-white leading-relaxed">{message.content}</p>
              ) : (
                <MarkdownRenderer 
                  content={message.content} 
                  className={cn(
                    "prose-sm",
                    isUser ? "prose-invert" : "prose dark:prose-invert"
                  )}
                  isStreaming={message.isStreaming}
                />
              )}
              
              {/* Inline Streaming Progress - Show real-time agent activities */}
              {isActivelyStreaming && !isUser && (
                <StreamingProgress 
                  events={intermediateEvents}
                  isActive={isActivelyStreaming}
                  className="mt-3"
                />
              )}
              
              {/* Multi-Agent Workflow Visualization - Show during streaming if metadata available */}
              {!isUser && (message.metadata?.planDetails || message.metadata?.grounding || message.metadata?.currentAgent || !message.isStreaming) && (
                <div className="space-y-3">
                  {/* Compact Workflow Visualizer - Always show for assistant messages */}
                  <div className="flex items-center gap-3 p-3 bg-gray-50 dark:bg-gray-800 rounded-lg">
                    <span className="text-xs font-medium text-gray-700 dark:text-gray-300">
                      Multi-Agent Workflow:
                    </span>
                    <WorkflowVisualizer compact />
                    {(message.metadata?.planDetails || message.metadata?.grounding || message.metadata?.currentAgent) && (
                      <Button
                        variant="ghost"
                        size="sm"
                        onClick={() => setMultiAgentExpanded(!multiAgentExpanded)}
                        className="ml-auto h-6 px-2 text-xs"
                      >
                        <BarChart className="w-3 h-3 mr-1" />
                        Details
                        {multiAgentExpanded ? <ChevronUp className="w-3 h-3 ml-1" /> : <ChevronDown className="w-3 h-3 ml-1" />}
                      </Button>
                    )}
                    {message.isStreaming && (
                      <div className="flex items-center gap-1 ml-2">
                        <div className="w-2 h-2 bg-blue-500 rounded-full animate-pulse" />
                        <span className="text-xs text-blue-600 dark:text-blue-400">
                          Live Progress
                        </span>
                      </div>
                    )}
                  </div>

                  {/* Detailed Multi-Agent Info - Show during streaming if data available */}
                  {(multiAgentExpanded || message.isStreaming) && (message.metadata?.planDetails || message.metadata?.grounding || message.metadata?.currentAgent) && (
                    <div className="space-y-3">
                      <WorkflowVisualizer />
                      
                      {/* Research Plan - Show during streaming */}
                      {message.metadata?.planDetails && (
                        <PlanViewer planData={message.metadata.planDetails} isStreaming={message.isStreaming} />
                      )}
                      
                      {/* Grounding Report - Show during streaming */}
                      {message.metadata?.grounding && (
                        <GroundingReport groundingData={message.metadata.grounding} />
                      )}
                      
                      {/* Basic Agent Info if no detailed metadata */}
                      {!message.metadata?.planDetails && !message.metadata?.grounding && message.metadata?.currentAgent && (
                        <div className="p-3 bg-blue-50 dark:bg-blue-950 rounded-lg">
                          <div className="text-sm font-medium text-blue-700 dark:text-blue-300">
                            Current Agent: {message.metadata.currentAgent}
                          </div>
                          <div className="text-xs text-blue-600 dark:text-blue-400 mt-1">
                            {message.isStreaming ? 'Research workflow in progress...' : 'Research workflow completed successfully.'}
                          </div>
                        </div>
                      )}
                    </div>
                  )}
                </div>
              )}

              {/* Sources Panel - Show during streaming if sources available */}
              {message.metadata?.sources && message.metadata.sources.length > 0 && !isUser && (
                <SourcesPanel sources={message.metadata.sources} isStreaming={message.isStreaming} />
              )}
            </div>
          </CardContent>
        </Card>
        
        {/* Research Events Panel - Show intermediate events for completed messages */}
        {!isUser && !message.isStreaming && intermediateEvents.length > 0 && (
          <ResearchEventsPanel 
            events={intermediateEvents.filter(event => 
              // Only show events that are related to this message
              // You might want to add message ID correlation here
              true
            )}
          />
        )}
      </div>
      
      {/* User Avatar */}
      {isUser && (
        <div className="w-8 h-8 bg-gray-300 dark:bg-gray-600 rounded-full flex items-center justify-center flex-shrink-0">
          <User className="w-4 h-4 text-gray-600 dark:text-gray-300" />
        </div>
      )}
    </div>
  )
}