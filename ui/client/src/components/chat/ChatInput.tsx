import React, { useState, useRef, useEffect } from 'react'
import { Send, Square } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { useChatStream } from '@/hooks/useChatStream'
import { useChatStore } from '@/stores/chatStore'
import { cn } from '@/lib/utils'

export function ChatInput() {
  const [input, setInput] = useState('')
  const textareaRef = useRef<HTMLTextAreaElement>(null)
  const { sendMessage, stopStream } = useChatStream()
  const { isLoading } = useChatStore()
  
  // Auto-resize textarea
  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto'
      textareaRef.current.style.height = `${Math.min(textareaRef.current.scrollHeight, 200)}px`
    }
  }, [input])
  
  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    
    if (!input.trim() || isLoading) return
    
    const message = input.trim()
    setInput('')
    
    await sendMessage(message)
  }
  
  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSubmit(e)
    }
  }
  
  const handleStop = () => {
    stopStream()
  }
  
  return (
    <div className="w-full max-w-4xl mx-auto">
      <form onSubmit={handleSubmit} className="relative">
        <div className="relative border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-800 shadow-sm hover:shadow-md transition-shadow">
          <textarea
            ref={textareaRef}
            data-testid="chat-input"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder={isLoading ? "Research Agent is thinking..." : "Ask me anything to research..."}
            disabled={isLoading}
            rows={1}
            className={cn(
              "w-full resize-none border-0 bg-transparent px-4 py-3 pr-12 text-sm",
              "placeholder:text-gray-500 dark:placeholder:text-gray-400",
              "focus:outline-none focus:ring-0",
              "disabled:cursor-not-allowed disabled:opacity-50",
              "max-h-[200px] overflow-y-auto"
            )}
          />
          
          {/* Submit/Stop Button */}
          <div className="absolute right-2 bottom-2">
            {isLoading ? (
              <Button
                type="button"
                size="sm"
                variant="ghost"
                onClick={handleStop}
                className="h-8 w-8 p-0 text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-200"
              >
                <Square className="h-4 w-4" />
                <span className="sr-only">Stop</span>
              </Button>
            ) : (
              <Button
                type="submit"
                size="sm"
                disabled={!input.trim()}
                className="h-8 w-8 p-0 bg-blue-600 hover:bg-blue-700 disabled:bg-gray-300 disabled:cursor-not-allowed"
              >
                <Send className="h-4 w-4" />
                <span className="sr-only">Send</span>
              </Button>
            )}
          </div>
        </div>
        
        {/* Hint Text */}
        <div className="mt-2 text-xs text-gray-500 dark:text-gray-400 text-center">
          {isLoading ? (
            "Researching your query with multiple sources..."
          ) : (
            <>Press <kbd className="px-1 py-0.5 bg-gray-100 dark:bg-gray-700 rounded text-xs">Enter</kbd> to send, <kbd className="px-1 py-0.5 bg-gray-100 dark:bg-gray-700 rounded text-xs">Shift+Enter</kbd> for new line</>
          )}
        </div>
      </form>
    </div>
  )
}