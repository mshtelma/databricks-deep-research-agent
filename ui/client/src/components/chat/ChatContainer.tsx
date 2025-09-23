import { useEffect, useRef, useState } from 'react'
import { Trash2, PanelLeftClose, PanelLeftOpen } from 'lucide-react'
import { useChatStore } from '@/stores/chatStore'
import { ChatMessage } from './ChatMessage'
import { ChatInput } from './ChatInput'
import { AgentSettings } from './AgentSettings'
import { ResearchTracePanel } from './ResearchTracePanel'
import { FinalAnswerPanel } from './FinalAnswerPanel'
import { ScrollArea } from '@/components/ui/scroll-area'
import { Button } from '@/components/ui/button'

export function ChatContainer() {
  const { messages, clearChat, finalizeMessageEvents } = useChatStore()
  const scrollAreaRef = useRef<HTMLDivElement>(null)
  const [showTracePanel, setShowTracePanel] = useState(true)
  const [tracePanelExpanded, setTracePanelExpanded] = useState(false)
  
  // Auto-scroll to bottom when new messages arrive
  useEffect(() => {
    if (scrollAreaRef.current) {
      const scrollElement = scrollAreaRef.current.querySelector('[data-radix-scroll-area-viewport]')
      if (scrollElement) {
        scrollElement.scrollTop = scrollElement.scrollHeight
      }
    }
  }, [messages])
  
  // Finalize events when a message stops streaming
  useEffect(() => {
    messages.forEach(message => {
      if (!message.isStreaming && message.role === 'assistant') {
        // Check if events have been finalized for this message
        finalizeMessageEvents(message.id)
      }
    })
  }, [messages, finalizeMessageEvents])
  
  const handleClearChat = () => {
    if (window.confirm('Are you sure you want to clear the chat history?')) {
      clearChat()
    }
  }
  
  return (
    <div className="flex h-screen bg-gradient-to-br from-slate-50 to-blue-50 dark:from-gray-900 dark:to-gray-800">
      {/* Research Trace Panel */}
      {showTracePanel && (
        <ResearchTracePanel 
          isExpanded={tracePanelExpanded}
          onToggleExpand={() => setTracePanelExpanded(!tracePanelExpanded)}
        />
      )}
      
      {/* Main Chat Interface */}
      <div className="flex-1 flex flex-col min-w-0">
        {/* Header */}
        <div className="flex-shrink-0 border-b bg-white/80 backdrop-blur-sm dark:bg-gray-900/80 p-4 shadow-sm">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              {/* Toggle Research Trace Panel */}
              <Button
                variant="ghost"
                size="sm"
                onClick={() => setShowTracePanel(!showTracePanel)}
                className="p-1"
              >
                {showTracePanel ? <PanelLeftClose className="h-4 w-4" /> : <PanelLeftOpen className="h-4 w-4" />}
              </Button>
              
              <div className="w-10 h-10 bg-gradient-to-r from-blue-500 to-purple-600 rounded-lg flex items-center justify-center shadow-md">
                <span className="text-white font-bold text-lg">DR</span>
              </div>
              <div>
                <h1 className="text-xl font-bold bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">
                  Deep Research Agent
                </h1>
                <p className="text-sm text-gray-500 dark:text-gray-400">
                  Transparent AI Research • Multi-Agent Architecture
                </p>
              </div>
            </div>
            
            {/* Header Actions */}
            <div className="flex items-center gap-2">
              {messages.length > 0 && (
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={handleClearChat}
                  className="text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-200"
                >
                  <Trash2 className="h-4 w-4 mr-2" />
                  Clear Chat
                </Button>
              )}
              <AgentSettings />
            </div>
          </div>
        </div>
        
        {/* Chat Messages */}
        <div className="flex-1 flex flex-col min-h-0">
          <ScrollArea className="flex-1 px-4 py-6" ref={scrollAreaRef} data-testid="chat-messages">
            {messages.length === 0 ? (
              <div className="flex items-center justify-center h-full">
                <div className="text-center space-y-4 max-w-md">
                  <div className="w-16 h-16 bg-gradient-to-r from-blue-500 to-purple-600 rounded-full flex items-center justify-center mx-auto shadow-lg">
                    <span className="text-white font-bold text-2xl">DR</span>
                  </div>
                  <h2 className="text-2xl font-semibold text-gray-800 dark:text-gray-200">
                    Welcome to Transparent AI Research
                  </h2>
                  <p className="text-gray-600 dark:text-gray-400 leading-relaxed">
                    Watch your AI research assistant work in real-time. The left panel shows live research 
                    activities, reasoning, and progress as the multi-agent system conducts comprehensive research.
                  </p>
                  <div className="grid grid-cols-1 gap-2 text-left">
                    <div className="p-3 bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700">
                      <p className="text-sm text-gray-700 dark:text-gray-300">
                        🔍 <strong>Live Research Trace:</strong> Watch agents search, reason, and verify information
                      </p>
                    </div>
                    <div className="p-3 bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700">
                      <p className="text-sm text-gray-700 dark:text-gray-300">
                        🧠 <strong>Transparent Reasoning:</strong> See the agent's thought process and decision making
                      </p>
                    </div>
                    <div className="p-3 bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700">
                      <p className="text-sm text-gray-700 dark:text-gray-300">
                        📊 <strong>Multi-Agent Workflow:</strong> Coordinator → Planner → Researcher → Fact Checker → Reporter
                      </p>
                    </div>
                  </div>
                </div>
              </div>
            ) : (
              <div className="space-y-6 pb-4">
                {messages.map((message) => (
                  message.role === 'user' ? (
                    <ChatMessage key={message.id} message={message} />
                  ) : (
                    <FinalAnswerPanel key={message.id} message={message} />
                  )
                ))}
              </div>
            )}
          </ScrollArea>
          
          {/* Input Area */}
          <div className="flex-shrink-0 border-t bg-white/80 backdrop-blur-sm dark:bg-gray-900/80 p-4">
            <ChatInput />
          </div>
        </div>
      </div>
    </div>
  )
}