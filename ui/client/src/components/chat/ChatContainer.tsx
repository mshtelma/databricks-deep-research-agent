import { useEffect, useRef } from 'react'
import { Trash2, Settings } from 'lucide-react'
import { useChatStore } from '@/stores/chatStore'
import { ChatMessage } from './ChatMessage'
import { ChatInput } from './ChatInput'
import { ResearchProgress } from './ResearchProgress'
import { ScrollArea } from '@/components/ui/scroll-area'
import { Button } from '@/components/ui/button'

export function ChatContainer() {
  const { messages, clearChat } = useChatStore()
  const scrollAreaRef = useRef<HTMLDivElement>(null)
  
  // Auto-scroll to bottom when new messages arrive
  useEffect(() => {
    if (scrollAreaRef.current) {
      const scrollElement = scrollAreaRef.current.querySelector('[data-radix-scroll-area-viewport]')
      if (scrollElement) {
        scrollElement.scrollTop = scrollElement.scrollHeight
      }
    }
  }, [messages])
  
  const handleClearChat = () => {
    if (window.confirm('Are you sure you want to clear the chat history?')) {
      clearChat()
    }
  }
  
  return (
    <div className="flex h-screen flex-col bg-gradient-to-br from-slate-50 to-blue-50 dark:from-gray-900 dark:to-gray-800">
      {/* Header */}
      <div className="border-b bg-white/80 backdrop-blur-sm dark:bg-gray-900/80 p-4 shadow-sm">
        <div className="flex items-center justify-between max-w-6xl mx-auto">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 bg-gradient-to-r from-blue-500 to-purple-600 rounded-lg flex items-center justify-center shadow-md">
              <span className="text-white font-bold text-lg">DR</span>
            </div>
            <div>
              <h1 className="text-xl font-bold bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">
                Deep Research Agent
              </h1>
              <p className="text-sm text-gray-500 dark:text-gray-400">
                Powered by LangGraph • Multi-source research
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
            <Button
              variant="ghost"
              size="sm"
              className="text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-200"
            >
              <Settings className="h-4 w-4" />
            </Button>
          </div>
        </div>
      </div>
      
      {/* Research Progress */}
      <div className="flex-shrink-0">
        <div className="max-w-6xl mx-auto px-4 py-2">
          <ResearchProgress />
        </div>
      </div>
      
      {/* Chat Messages */}
      <div className="flex-1 flex flex-col max-w-6xl mx-auto w-full">
        <ScrollArea className="flex-1 px-4 py-6" ref={scrollAreaRef} data-testid="chat-messages">
          {messages.length === 0 ? (
            <div className="flex items-center justify-center h-full">
              <div className="text-center space-y-4 max-w-md">
                <div className="w-16 h-16 bg-gradient-to-r from-blue-500 to-purple-600 rounded-full flex items-center justify-center mx-auto shadow-lg">
                  <span className="text-white font-bold text-2xl">DR</span>
                </div>
                <h2 className="text-2xl font-semibold text-gray-800 dark:text-gray-200">
                  Welcome to Deep Research Agent
                </h2>
                <p className="text-gray-600 dark:text-gray-400 leading-relaxed">
                  I'm your AI research assistant powered by advanced multi-agent architecture. 
                  Ask me anything and I'll conduct comprehensive research using multiple sources 
                  to provide you with detailed, well-sourced answers.
                </p>
                <div className="grid grid-cols-1 gap-2 text-left">
                  <div className="p-3 bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700">
                    <p className="text-sm text-gray-700 dark:text-gray-300">
                      💡 <strong>Try asking:</strong> "What are the latest developments in AI agent frameworks?"
                    </p>
                  </div>
                  <div className="p-3 bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700">
                    <p className="text-sm text-gray-700 dark:text-gray-300">
                      📊 <strong>Research topics:</strong> Technology trends, market analysis, academic papers
                    </p>
                  </div>
                  <div className="p-3 bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700">
                    <p className="text-sm text-gray-700 dark:text-gray-300">
                      🔍 <strong>Deep analysis:</strong> Multi-iteration research with source verification
                    </p>
                  </div>
                </div>
              </div>
            </div>
          ) : (
            <div className="space-y-6 pb-4">
              {messages.map((message) => (
                <ChatMessage key={message.id} message={message} />
              ))}
            </div>
          )}
        </ScrollArea>
        
        {/* Input Area */}
        <div className="border-t bg-white/80 backdrop-blur-sm dark:bg-gray-900/80 p-4">
          <ChatInput />
        </div>
      </div>
    </div>
  )
}