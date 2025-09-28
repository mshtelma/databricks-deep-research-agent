import { useState } from 'react'
import { User, Bot, Copy, Check, BarChart, ChevronDown, ChevronUp } from 'lucide-react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { ChatMessage as ChatMessageType } from '../hooks/useAgentClient'
import { cn } from '../utils/cn'
import { formatTime, getAgentIcon, formatConfidenceScore, formatSourceCount } from '../utils/formatters'
import { PlanTracker } from './PlanTracker'
import { SourcesPanel } from './SourcesPanel'
import { StreamingProgress } from './StreamingProgress'
import { ResearchProgress } from './ResearchProgress'
import { StructuredProgress } from '../types/progress'

interface ChatMessageProps {
  message: ChatMessageType
  intermediateEvents?: any[]
  isActivelyStreaming?: boolean
  researchProgress?: StructuredProgress
}

export function ChatMessage({ message, intermediateEvents = [], isActivelyStreaming = false, researchProgress }: ChatMessageProps) {
  const [detailsExpanded, setDetailsExpanded] = useState(false)
  const [copied, setCopied] = useState(false)
  const isUser = message.role === 'user'

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(message.content)
      setCopied(true)
      setTimeout(() => setCopied(false), 2000)
    } catch (err) {
      console.error('Failed to copy text:', err)
    }
  }

  return (
    <div className={cn("flex gap-3 mb-6", isUser ? "justify-end" : "justify-start")}>
      {/* Assistant Avatar */}
      {!isUser && (
        <div className="w-8 h-8 bg-gradient-to-r from-databricks-orange to-databricks-blue rounded-full flex items-center justify-center flex-shrink-0">
          <Bot className="w-4 h-4 text-white" />
        </div>
      )}

      {/* Message Content */}
      <div className={cn("max-w-3xl flex-1", isUser ? "order-first" : "")}>
        <div className={cn(
          "rounded-lg shadow-sm hover:shadow-md transition-shadow",
          isUser
            ? "bg-databricks-blue text-white ml-auto max-w-lg"
            : "bg-white border border-gray-200"
        )}>
          <div className="p-4">
            {/* Message Header */}
            <div className="flex items-center justify-between mb-3">
              <div className="flex items-center gap-2">
                {isUser && <User className="w-4 h-4" />}
                <span className={cn(
                  "text-xs",
                  isUser ? "text-blue-100" : "text-gray-500"
                )}>
                  {formatTime(message.timestamp || Date.now())}
                </span>
                {message.isStreaming && (
                  <div className="flex items-center gap-1">
                    <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse" />
                    <span className={cn(
                      "text-xs",
                      isUser ? "text-blue-100" : "text-green-600"
                    )}>
                      Researching...
                    </span>
                  </div>
                )}
              </div>

              <div className="flex items-center gap-2">
                {/* Metadata Badges */}
                {message.metadata && !isUser && (
                  <div className="flex items-center gap-1 flex-wrap">
                    {message.metadata.researchIterations && message.metadata.researchIterations > 0 && (
                      <span className="bg-purple-100 text-purple-800 text-xs px-2 py-1 rounded-full">
                        {message.metadata.researchIterations} iterations
                      </span>
                    )}
                    {message.metadata.sources && message.metadata.sources.length > 0 && (
                      <span className="bg-blue-100 text-blue-800 text-xs px-2 py-1 rounded-full">
                        {formatSourceCount(message.metadata.sources.length)}
                      </span>
                    )}
                    {message.metadata.confidenceScore && (
                      <span className="bg-green-100 text-green-800 text-xs px-2 py-1 rounded-full">
                        {formatConfidenceScore(message.metadata.confidenceScore)} confidence
                      </span>
                    )}
                    {message.metadata.currentAgent && (
                      <span className={cn(
                        "agent-badge",
                        message.metadata.currentAgent.toLowerCase()
                      )}>
                        {getAgentIcon(message.metadata.currentAgent)} {message.metadata.currentAgent}
                      </span>
                    )}
                    {message.metadata.factualityScore && (
                      <span className="bg-red-100 text-red-800 text-xs px-2 py-1 rounded-full">
                        🎯 {formatConfidenceScore(message.metadata.factualityScore)} factual
                      </span>
                    )}
                    {message.metadata.reportStyle && (
                      <span className="bg-gray-100 text-gray-800 text-xs px-2 py-1 rounded-full capitalize">
                        📄 {message.metadata.reportStyle}
                      </span>
                    )}
                  </div>
                )}

                {/* Copy Button */}
                <button
                  onClick={handleCopy}
                  className={cn(
                    "p-1 rounded hover:bg-gray-100 transition-colors",
                    isUser && "hover:bg-blue-700"
                  )}
                  title="Copy message"
                >
                  {copied ? (
                    <Check className="w-3 h-3" />
                  ) : (
                    <Copy className="w-3 h-3" />
                  )}
                </button>
              </div>
            </div>

            {/* Message Content */}
            <div className="space-y-3">
              {isUser ? (
                <p className="text-white leading-relaxed">{message.content}</p>
              ) : (
                <div>
                  <div className="prose prose-sm max-w-none prose-gray dark:prose-invert">
                    <ReactMarkdown
                      remarkPlugins={[remarkGfm]}
                      components={{
                        // Custom rendering for security and styling
                        a: ({node, ...props}) => (
                          <a {...props} target="_blank" rel="noopener noreferrer" className="text-databricks-blue hover:text-databricks-dark-blue hover:underline" />
                        ),
                        code: ({node, className, children, ...props}) => {
                          const isInline = !className?.includes('language-')
                          return isInline ? (
                            <code className="bg-gray-100 px-1 py-0.5 rounded text-sm font-mono" {...props}>
                              {children}
                            </code>
                          ) : (
                            <pre className="bg-gray-100 p-3 rounded overflow-x-auto">
                              <code className={`${className} font-mono text-sm`} {...props}>
                                {children}
                              </code>
                            </pre>
                          )
                        },
                        h1: ({node, ...props}) => <h1 className="text-xl font-bold mt-4 mb-2 text-gray-900" {...props} />,
                        h2: ({node, ...props}) => <h2 className="text-lg font-semibold mt-3 mb-2 text-gray-900" {...props} />,
                        h3: ({node, ...props}) => <h3 className="text-base font-medium mt-2 mb-1 text-gray-900" {...props} />,
                        ul: ({node, ...props}) => <ul className="list-disc list-inside my-2 space-y-1" {...props} />,
                        ol: ({node, ...props}) => <ol className="list-decimal list-inside my-2 space-y-1" {...props} />,
                        blockquote: ({node, ...props}) => (
                          <blockquote className="border-l-4 border-databricks-orange pl-4 italic my-2 text-gray-700" {...props} />
                        ),
                        table: ({node, ...props}) => (
                          <table className="border-collapse border border-gray-300 my-2 w-full" {...props} />
                        ),
                        th: ({node, ...props}) => (
                          <th className="border border-gray-300 px-3 py-1 bg-gray-100 font-semibold text-left" {...props} />
                        ),
                        td: ({node, ...props}) => (
                          <td className="border border-gray-300 px-3 py-1" {...props} />
                        ),
                      }}
                    >
                      {message.content}
                    </ReactMarkdown>
                  </div>
                  {message.isStreaming && (
                    <span className="inline-block w-2 h-4 bg-gray-500 animate-pulse ml-1" />
                  )}
                </div>
              )}

              {/* Research Progress - Show workflow progress for assistant messages */}
              {!isUser && researchProgress && (researchProgress.workflowPhases.length > 0 || researchProgress.planSteps.length > 0) && (
                <ResearchProgress
                  structuredProgress={researchProgress}
                  isStreaming={message.isStreaming}
                  className="mt-4"
                  showTimestamps={true}
                  autoScroll={isActivelyStreaming}
                  collapsible={!isActivelyStreaming}
                />
              )}

              {/* Inline Streaming Progress - Show real-time agent activities */}
              {isActivelyStreaming && !isUser && intermediateEvents.length > 0 && (
                <StreamingProgress
                  events={intermediateEvents}
                  isActive={isActivelyStreaming}
                  className="mt-3"
                />
              )}
            </div>
          </div>

          {/* Details Section */}
          {!isUser && (message.metadata?.planDetails || message.metadata?.sources || message.metadata?.currentAgent) && (
            <div className="border-t border-gray-200">
              {/* Compact Summary */}
              <div className="flex items-center gap-3 p-3 bg-gray-50">
                <span className="text-xs font-medium text-gray-700">
                  Research Details:
                </span>
                <div className="flex items-center gap-2 flex-1">
                  {message.metadata.planDetails && (
                    <span className="text-xs text-gray-600">
                      Plan ({message.metadata.planDetails.steps.length} steps)
                    </span>
                  )}
                  {message.metadata.sources && (
                    <span className="text-xs text-gray-600">
                      Sources ({message.metadata.sources.length})
                    </span>
                  )}
                  {message.isStreaming && (
                    <div className="flex items-center gap-1 ml-2">
                      <div className="w-2 h-2 bg-blue-500 rounded-full animate-pulse" />
                      <span className="text-xs text-blue-600">
                        Live Progress
                      </span>
                    </div>
                  )}
                </div>

                {(message.metadata?.planDetails || message.metadata?.sources) && (
                  <button
                    onClick={() => setDetailsExpanded(!detailsExpanded)}
                    className="flex items-center gap-1 text-xs text-databricks-blue hover:underline"
                  >
                    <BarChart className="w-3 h-3" />
                    Details
                    {detailsExpanded ? (
                      <ChevronUp className="w-3 h-3" />
                    ) : (
                      <ChevronDown className="w-3 h-3" />
                    )}
                  </button>
                )}
              </div>

              {/* Expanded Details */}
              {detailsExpanded && (
                <div className="p-3 space-y-4">
                  {/* Plan Details */}
                  {message.metadata?.planDetails && (
                    <PlanTracker planDetails={message.metadata.planDetails} />
                  )}

                  {/* Sources */}
                  {message.metadata?.sources && message.metadata.sources.length > 0 && (
                    <SourcesPanel
                      sources={message.metadata.sources}
                      isStreaming={message.isStreaming}
                    />
                  )}
                </div>
              )}
            </div>
          )}
        </div>
      </div>

      {/* User Avatar */}
      {isUser && (
        <div className="w-8 h-8 bg-gray-300 rounded-full flex items-center justify-center flex-shrink-0">
          <User className="w-4 h-4 text-gray-600" />
        </div>
      )}
    </div>
  )
}