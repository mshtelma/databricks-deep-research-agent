import { useState } from 'react'
import { ExternalLink, ChevronDown, ChevronUp, Star } from 'lucide-react'
import { cn } from '../utils/cn'

interface Source {
  url: string
  title: string
  relevanceScore?: number
  snippet?: string
}

interface SourcesPanelProps {
  sources: Source[]
  isStreaming?: boolean
  className?: string
}

export function SourcesPanel({ sources, isStreaming = false, className = '' }: SourcesPanelProps) {
  const [expanded, setExpanded] = useState(false)
  const [showAll, setShowAll] = useState(false)

  if (!sources || sources.length === 0) {
    return null
  }

  const displaySources = showAll ? sources : sources.slice(0, 3)
  const hasMoreSources = sources.length > 3

  const getRelevanceColor = (score?: number) => {
    if (!score) return 'text-gray-400'
    if (score >= 0.8) return 'text-green-500'
    if (score >= 0.6) return 'text-yellow-500'
    return 'text-orange-500'
  }

  const formatRelevanceScore = (score?: number) => {
    if (!score) return 'N/A'
    return `${Math.round(score * 100)}%`
  }

  return (
    <div className={cn("border border-gray-200 rounded-lg bg-white", className)}>
      {/* Header */}
      <div
        className="flex items-center justify-between p-3 cursor-pointer hover:bg-gray-50"
        onClick={() => setExpanded(!expanded)}
      >
        <div className="flex items-center gap-2">
          <h4 className="text-sm font-semibold text-gray-800">Sources</h4>
          <span className="bg-databricks-blue text-white text-xs px-2 py-1 rounded-full">
            {sources.length}
          </span>
          {isStreaming && (
            <div className="flex items-center gap-1">
              <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse" />
              <span className="text-xs text-green-600">Live</span>
            </div>
          )}
        </div>
        <div className="flex-shrink-0">
          {expanded ? (
            <ChevronUp className="w-4 h-4 text-gray-500" />
          ) : (
            <ChevronDown className="w-4 h-4 text-gray-500" />
          )}
        </div>
      </div>

      {/* Expanded Content */}
      {expanded && (
        <div className="px-3 pb-3 space-y-3">
          {displaySources.map((source, index) => (
            <div
              key={index}
              className="border border-gray-100 rounded-lg p-3 hover:bg-gray-50 transition-colors"
            >
              <div className="flex items-start gap-3">
                <div className="flex-grow min-w-0">
                  <div className="flex items-center gap-2 mb-1">
                    <a
                      href={source.url}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="text-sm font-medium text-databricks-blue hover:underline flex items-center gap-1 truncate"
                    >
                      {source.title}
                      <ExternalLink className="w-3 h-3 flex-shrink-0" />
                    </a>
                  </div>

                  {source.snippet && (
                    <p className="text-xs text-gray-600 line-clamp-2 mb-2">
                      {source.snippet}
                    </p>
                  )}

                  <p className="text-xs text-gray-400 truncate">
                    {source.url}
                  </p>
                </div>

                {source.relevanceScore && (
                  <div className="flex-shrink-0 flex items-center gap-1">
                    <Star className={cn("w-3 h-3", getRelevanceColor(source.relevanceScore))} />
                    <span className={cn("text-xs", getRelevanceColor(source.relevanceScore))}>
                      {formatRelevanceScore(source.relevanceScore)}
                    </span>
                  </div>
                )}
              </div>
            </div>
          ))}

          {/* Show More/Less Button */}
          {hasMoreSources && (
            <button
              onClick={() => setShowAll(!showAll)}
              className="w-full text-sm text-databricks-blue hover:underline py-2"
            >
              {showAll
                ? 'Show less'
                : `Show ${sources.length - 3} more sources`
              }
            </button>
          )}
        </div>
      )}
    </div>
  )
}
