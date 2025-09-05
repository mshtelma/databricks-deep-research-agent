import React, { useState } from 'react'
import { ChevronDown, ChevronRight, Shield, AlertTriangle, CheckCircle, ExternalLink } from 'lucide-react'
import { Badge } from '@/components/ui/badge'
import { Progress } from '@/components/ui/progress'
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from '@/components/ui/collapsible'
import { GroundingMetadata } from '@/types/chat'
import { cn } from '@/lib/utils'

interface GroundingReportProps {
  groundingData: GroundingMetadata
  className?: string
}

export function GroundingReport({ groundingData, className = '' }: GroundingReportProps) {
  const [isExpanded, setIsExpanded] = useState(true)
  const [showResolved, setShowResolved] = useState(false)

  if (!groundingData) {
    return null
  }

  const factualityScore = groundingData.factualityScore || 0
  const contradictions = groundingData.contradictions || []
  const verifications = groundingData.verifications || []
  
  const unresolvedContradictions = contradictions.filter(c => !c.resolved)
  const resolvedContradictions = contradictions.filter(c => c.resolved)
  const verifiedFacts = verifications.filter(v => v.verified).length
  const totalFacts = verifications.length

  const getFactualityColor = (score: number) => {
    if (score >= 0.8) return 'text-green-600'
    if (score >= 0.6) return 'text-yellow-600'
    return 'text-red-600'
  }

  const getFactualityBgColor = (score: number) => {
    if (score >= 0.8) return 'bg-green-50 border-green-200'
    if (score >= 0.6) return 'bg-yellow-50 border-yellow-200'
    return 'bg-red-50 border-red-200'
  }

  const getSeverityColor = (severity: string) => {
    switch (severity) {
      case 'high':
        return 'text-red-600 bg-red-50 border-red-200'
      case 'medium':
        return 'text-yellow-600 bg-yellow-50 border-yellow-200'
      case 'low':
        return 'text-blue-600 bg-blue-50 border-blue-200'
      default:
        return 'text-gray-600 bg-gray-50 border-gray-200'
    }
  }

  const getVerificationIcon = (verified: boolean, confidence: number) => {
    if (verified && confidence >= 0.8) {
      return <CheckCircle className="w-4 h-4 text-green-600" />
    } else if (verified && confidence >= 0.6) {
      return <CheckCircle className="w-4 h-4 text-yellow-600" />
    } else {
      return <AlertTriangle className="w-4 h-4 text-red-600" />
    }
  }

  return (
    <div className={`bg-white dark:bg-gray-800 border rounded-lg ${className}`}>
      <Collapsible open={isExpanded} onOpenChange={setIsExpanded}>
        <CollapsibleTrigger className="w-full p-4 flex items-center justify-between hover:bg-gray-50 dark:hover:bg-gray-700 transition-colors">
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 bg-blue-100 rounded-full flex items-center justify-center">
              <Shield className="w-4 h-4 text-blue-600" />
            </div>
            <div className="text-left">
              <h3 className="text-sm font-medium">Fact Verification Report</h3>
              <p className="text-xs text-gray-500">
                {Math.round(factualityScore * 100)}% factuality score
                {contradictions.length > 0 && (
                  <span className="ml-2">• {unresolvedContradictions.length} issues</span>
                )}
              </p>
            </div>
          </div>
          <div className="flex items-center gap-2">
            <Badge 
              variant="secondary" 
              className={cn("text-xs", getFactualityColor(factualityScore))}
            >
              {groundingData.verificationLevel} verification
            </Badge>
            {isExpanded ? (
              <ChevronDown className="w-4 h-4 text-gray-400" />
            ) : (
              <ChevronRight className="w-4 h-4 text-gray-400" />
            )}
          </div>
        </CollapsibleTrigger>

        <CollapsibleContent className="px-4 pb-4">
          {/* Factuality Score */}
          <div className={cn("mb-4 p-4 border rounded-lg", getFactualityBgColor(factualityScore))}>
            <div className="flex justify-between items-center mb-2">
              <span className="text-sm font-medium">Overall Factuality</span>
              <span className={cn("text-lg font-bold", getFactualityColor(factualityScore))}>
                {Math.round(factualityScore * 100)}%
              </span>
            </div>
            <Progress value={factualityScore * 100} className="h-2" />
            <div className="mt-2 text-xs text-gray-600">
              {verifiedFacts}/{totalFacts} facts verified • {groundingData.verificationLevel} level verification
            </div>
          </div>

          {/* Contradictions Section */}
          {contradictions.length > 0 && (
            <div className="mb-4">
              <div className="flex items-center justify-between mb-3">
                <h4 className="text-sm font-medium flex items-center gap-2">
                  <AlertTriangle className="w-4 h-4" />
                  Contradictions Found
                </h4>
                {resolvedContradictions.length > 0 && (
                  <button
                    onClick={() => setShowResolved(!showResolved)}
                    className="text-xs text-blue-600 hover:text-blue-800"
                  >
                    {showResolved ? 'Hide' : 'Show'} resolved ({resolvedContradictions.length})
                  </button>
                )}
              </div>

              <div className="space-y-2">
                {unresolvedContradictions.map((contradiction) => (
                  <div
                    key={contradiction.id}
                    className={cn("p-3 border rounded-lg", getSeverityColor(contradiction.severity))}
                  >
                    <div className="flex items-start justify-between mb-2">
                      <Badge 
                        variant="outline" 
                        className={cn("text-xs capitalize", getSeverityColor(contradiction.severity))}
                      >
                        {contradiction.severity} severity
                      </Badge>
                    </div>
                    <div className="text-sm font-medium mb-1">
                      Claim: "{contradiction.claim}"
                    </div>
                    <div className="text-xs text-gray-600">
                      Evidence: {contradiction.evidence}
                    </div>
                  </div>
                ))}

                {showResolved && resolvedContradictions.map((contradiction) => (
                  <div
                    key={contradiction.id}
                    className="p-3 border rounded-lg bg-green-50 border-green-200 opacity-75"
                  >
                    <div className="flex items-start justify-between mb-2">
                      <Badge variant="outline" className="text-xs text-green-600 bg-green-50">
                        ✓ Resolved
                      </Badge>
                    </div>
                    <div className="text-sm font-medium mb-1">
                      Claim: "{contradiction.claim}"
                    </div>
                    {contradiction.resolution && (
                      <div className="text-xs text-green-700 mt-2">
                        Resolution: {contradiction.resolution}
                      </div>
                    )}
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Fact Verifications */}
          {verifications.length > 0 && (
            <div className="mb-4">
              <h4 className="text-sm font-medium mb-3 flex items-center gap-2">
                <CheckCircle className="w-4 h-4" />
                Fact Verifications
              </h4>
              <div className="space-y-2 max-h-64 overflow-y-auto">
                {verifications.map((verification) => (
                  <div
                    key={verification.id}
                    className="p-3 border rounded-lg bg-gray-50 dark:bg-gray-700"
                  >
                    <div className="flex items-start gap-3">
                      <div className="flex-shrink-0 mt-0.5">
                        {getVerificationIcon(verification.verified, verification.confidence)}
                      </div>
                      <div className="flex-grow min-w-0">
                        <div className="text-sm font-medium mb-1">
                          {verification.fact}
                        </div>
                        <div className="flex items-center gap-4 text-xs text-gray-600">
                          <span>
                            Confidence: {Math.round(verification.confidence * 100)}%
                          </span>
                          <span>
                            {verification.sources.length} source{verification.sources.length !== 1 ? 's' : ''}
                          </span>
                        </div>
                        {verification.sources.length > 0 && (
                          <div className="mt-2 text-xs">
                            {verification.sources.slice(0, 2).map((source, index) => (
                              <a
                                key={index}
                                href={source}
                                target="_blank"
                                rel="noopener noreferrer"
                                className="inline-flex items-center gap-1 text-blue-600 hover:text-blue-800 mr-3"
                              >
                                <ExternalLink className="w-3 h-3" />
                                Source {index + 1}
                              </a>
                            ))}
                            {verification.sources.length > 2 && (
                              <span className="text-gray-500">
                                +{verification.sources.length - 2} more
                              </span>
                            )}
                          </div>
                        )}
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Summary */}
          <div className="pt-3 border-t text-xs text-gray-600">
            <div className="grid grid-cols-3 gap-4 text-center">
              <div>
                <div className="font-medium text-green-600">
                  {verifiedFacts}
                </div>
                <div>Verified Facts</div>
              </div>
              <div>
                <div className="font-medium text-red-600">
                  {unresolvedContradictions.length}
                </div>
                <div>Open Issues</div>
              </div>
              <div>
                <div className="font-medium text-blue-600">
                  {Math.round(factualityScore * 100)}%
                </div>
                <div>Overall Score</div>
              </div>
            </div>
          </div>
        </CollapsibleContent>
      </Collapsible>
    </div>
  )
}