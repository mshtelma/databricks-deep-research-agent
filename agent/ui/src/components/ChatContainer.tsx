import React, { useState } from 'react'
import { useAgentClient } from '../hooks/useAgentClient'
import { ChatMessage } from './ChatMessage'
import { cn } from '../utils/cn'


const ChatContainer: React.FC = () => {
  const [inputValue, setInputValue] = useState('')
  const {
    messages,
    isStreaming,
    intermediateEvents,
    currentStreamingId,
    researchProgress,
    sendStreamingMessage,
    stopStreaming,
    clearMessages
  } = useAgentClient()

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!inputValue.trim() || isStreaming) return

    const message = inputValue.trim()
    setInputValue('')
    await sendStreamingMessage(message)
  }

  return (
    <div className="flex flex-col h-screen bg-gray-50">
      {/* Header */}
      <div className="bg-white border-b border-gray-200 px-6 py-4 flex justify-between items-center">
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 databricks-gradient rounded-lg flex items-center justify-center">
            <span className="text-white font-bold text-sm">DB</span>
          </div>
          <div>
            <h1 className="text-xl font-semibold text-gray-800">
              Deep Research Agent
            </h1>
            <p className="text-xs text-gray-500">
              Multi-agent research with live progress tracking
            </p>
          </div>
        </div>
        <div className="flex items-center gap-2">
          {isStreaming && (
            <div className="flex items-center gap-2 px-3 py-1 bg-green-50 border border-green-200 rounded-lg">
              <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse" />
              <span className="text-xs text-green-700 font-medium">Researching</span>
            </div>
          )}
          {isStreaming && (
            <button
              onClick={stopStreaming}
              className="px-3 py-1 bg-red-500 text-white text-sm rounded-lg hover:bg-red-600 transition-colors"
            >
              Stop
            </button>
          )}
          <button
            onClick={clearMessages}
            className="px-3 py-1 bg-gray-500 text-white text-sm rounded-lg hover:bg-gray-600 transition-colors"
          >
            Clear
          </button>
        </div>
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto px-6 py-4">
        {messages.length === 0 ? (
          <div className="text-center text-gray-500 mt-8 max-w-2xl mx-auto">
            <div className="w-16 h-16 databricks-gradient rounded-2xl flex items-center justify-center mx-auto mb-4">
              <span className="text-white font-bold text-2xl">🔬</span>
            </div>
            <h2 className="text-2xl font-semibold text-gray-800 mb-3">
              Welcome to Deep Research Agent
            </h2>
            <p className="text-gray-600 mb-6 leading-relaxed">
              Ask me any research question and I'll provide a comprehensive analysis
              with multi-agent coordination, live progress tracking, and detailed reports.
            </p>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-8">
              <div className="p-4 bg-white rounded-lg border border-gray-200">
                <div className="text-2xl mb-2">🎯</div>
                <h3 className="font-medium text-gray-800 mb-1">Intelligent Coordination</h3>
                <p className="text-xs text-gray-600">Smart routing with specialized agents</p>
              </div>
              <div className="p-4 bg-white rounded-lg border border-gray-200">
                <div className="text-2xl mb-2">📋</div>
                <h3 className="font-medium text-gray-800 mb-1">Dynamic Planning</h3>
                <p className="text-xs text-gray-600">Adaptive research plans with quality assessment</p>
              </div>
              <div className="p-4 bg-white rounded-lg border border-gray-200">
                <div className="text-2xl mb-2">🔎</div>
                <h3 className="font-medium text-gray-800 mb-1">Fact Verification</h3>
                <p className="text-xs text-gray-600">Multi-layer claim validation and grounding</p>
              </div>
            </div>

            <div className="text-left bg-white p-4 rounded-lg border border-gray-200">
              <p className="font-medium text-gray-800 mb-2">Example Research Questions:</p>
              <ul className="space-y-2 text-sm text-gray-600">
                <li className="flex items-start gap-2">
                  <span className="text-databricks-orange">•</span>
                  "What are the latest developments in quantum computing and their potential applications?"
                </li>
                <li className="flex items-start gap-2">
                  <span className="text-databricks-blue">•</span>
                  "Analyze the impact of generative AI on healthcare and medical research"
                </li>
                <li className="flex items-start gap-2">
                  <span className="text-databricks-orange">•</span>
                  "Compare different renewable energy technologies and their scalability"
                </li>
              </ul>
            </div>
          </div>
        ) : (
          <div className="max-w-4xl mx-auto">
            {messages.map(message => (
              <ChatMessage
                key={message.id}
                message={message}
                intermediateEvents={intermediateEvents}
                isActivelyStreaming={isStreaming && currentStreamingId === message.id}
                researchProgress={isStreaming && currentStreamingId === message.id ? researchProgress : undefined}
              />
            ))}
          </div>
        )}
      </div>

      {/* Input */}
      <div className="bg-white border-t border-gray-200 px-6 py-4">
        <div className="max-w-4xl mx-auto">
          <form onSubmit={handleSubmit} className="flex gap-3">
            <div className="flex-1 relative">
              <input
                type="text"
                value={inputValue}
                onChange={(e) => setInputValue(e.target.value)}
                placeholder="Ask a research question..."
                disabled={isStreaming}
                className={cn(
                  "w-full px-4 py-3 border border-gray-300 rounded-lg",
                  "focus:outline-none focus:ring-2 focus:ring-databricks-blue focus:border-transparent",
                  "disabled:opacity-50 disabled:cursor-not-allowed",
                  "placeholder:text-gray-400"
                )}
              />
              {isStreaming && (
                <div className="absolute right-3 top-1/2 transform -translate-y-1/2">
                  <div className="flex space-x-1">
                    <div className="w-1 h-1 bg-databricks-blue rounded-full thinking-dot" />
                    <div className="w-1 h-1 bg-databricks-blue rounded-full thinking-dot" />
                    <div className="w-1 h-1 bg-databricks-blue rounded-full thinking-dot" />
                  </div>
                </div>
              )}
            </div>
            <button
              type="submit"
              disabled={!inputValue.trim() || isStreaming}
              className={cn(
                "px-6 py-3 rounded-lg font-medium transition-all duration-200",
                "focus:outline-none focus:ring-2 focus:ring-offset-2",
                !inputValue.trim() || isStreaming
                  ? "bg-gray-300 text-gray-500 cursor-not-allowed"
                  : "databricks-gradient text-white hover:shadow-lg focus:ring-databricks-blue"
              )}
            >
              {isStreaming ? 'Researching...' : 'Send'}
            </button>
          </form>

          <div className="flex items-center justify-center mt-3 text-xs text-gray-500">
            <span>Powered by multi-agent research system with live progress tracking</span>
          </div>
        </div>
      </div>
    </div>
  )
}

export default ChatContainer