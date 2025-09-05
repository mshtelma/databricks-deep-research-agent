import { useState } from 'react'
import { format } from 'date-fns'
import { User, Bot, ExternalLink, ChevronDown, ChevronUp, Copy, Check, Settings, BarChart } from 'lucide-react'
import { Card, CardContent } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from '@/components/ui/collapsible'
import { ChatMessage as ChatMessageType } from '@/types/chat'
import { MarkdownRenderer } from './MarkdownRenderer'
import { WorkflowVisualizer } from './WorkflowVisualizer'
import { PlanViewer } from './PlanViewer'
import { GroundingReport } from './GroundingReport'
import { cn } from '@/lib/utils'

interface ChatMessageProps {
  message: ChatMessageType
}

export function ChatMessage({ message }: ChatMessageProps) {
  const [sourcesExpanded, setSourcesExpanded] = useState(false)
  const [multiAgentExpanded, setMultiAgentExpanded] = useState(false)
  const [planExpanded, setPlanExpanded] = useState(false)
  const [groundingExpanded, setGroundingExpanded] = useState(false)
  const [copied, setCopied] = useState(false)
  const isUser = message.role === 'user'
  
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
              
              {/* Multi-Agent Workflow Visualization */}
              {message.metadata && !isUser && !message.isStreaming && (
                <div className="space-y-3">
                  {/* Compact Workflow Visualizer */}
                  <div className="flex items-center gap-3 p-3 bg-gray-50 dark:bg-gray-800 rounded-lg">
                    <span className="text-xs font-medium text-gray-700 dark:text-gray-300">
                      Multi-Agent Workflow:
                    </span>
                    <WorkflowVisualizer compact />
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
                  </div>

                  {/* Detailed Multi-Agent Info */}
                  {multiAgentExpanded && (
                    <div className="space-y-3">
                      <WorkflowVisualizer />
                      
                      {/* Research Plan */}
                      {message.metadata.planDetails && (
                        <PlanViewer planData={message.metadata.planDetails} />
                      )}
                      
                      {/* Grounding Report */}
                      {message.metadata.grounding && (
                        <GroundingReport groundingData={message.metadata.grounding} />
                      )}
                    </div>
                  )}
                </div>
              )}

              {/* Research Sources */}
              {message.metadata?.sources && message.metadata.sources.length > 0 && !isUser && (
                <Collapsible open={sourcesExpanded} onOpenChange={setSourcesExpanded}>
                  <CollapsibleTrigger asChild>
                    <Button 
                      variant="ghost" 
                      size="sm" 
                      className="h-8 px-3 text-xs text-gray-600 dark:text-gray-400 hover:text-gray-800 dark:hover:text-gray-200"
                    >
                      <ExternalLink className="w-3 h-3 mr-2" />
                      <span>View Sources ({message.metadata.sources.length})</span>
                      {sourcesExpanded ? <ChevronUp className="w-3 h-3 ml-2" /> : <ChevronDown className="w-3 h-3 ml-2" />}
                    </Button>
                  </CollapsibleTrigger>
                  <CollapsibleContent className="mt-3">
                    <div className="space-y-2 border-t border-gray-200 dark:border-gray-600 pt-3">
                      {message.metadata.sources.map((source, index) => (
                        <div 
                          key={index} 
                          className="flex items-start gap-3 p-3 bg-gray-50 dark:bg-gray-700 rounded-lg text-sm"
                        >
                          <ExternalLink className="w-4 h-4 mt-0.5 flex-shrink-0 text-gray-400" />
                          <div className="flex-1 min-w-0">
                            <div className="font-medium text-gray-900 dark:text-gray-100 truncate">
                              {source.title}
                            </div>
                            <a 
                              href={source.url}
                              target="_blank"
                              rel="noopener noreferrer"
                              className="text-blue-600 dark:text-blue-400 hover:underline truncate block"
                            >
                              {source.url}
                            </a>
                            {source.snippet && (
                              <p className="mt-1 text-gray-600 dark:text-gray-300 text-xs line-clamp-2">
                                {source.snippet}
                              </p>
                            )}
                          </div>
                          {source.relevanceScore && (
                            <Badge variant="outline" className="text-xs flex-shrink-0">
                              {Math.round(source.relevanceScore * 100)}%
                            </Badge>
                          )}
                        </div>
                      ))}
                    </div>
                  </CollapsibleContent>
                </Collapsible>
              )}
            </div>
          </CardContent>
        </Card>
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