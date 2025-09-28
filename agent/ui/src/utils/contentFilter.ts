/**
 * Content filtering utilities to remove progress markers and metadata
 * from displayed content while preserving structured data for progress tracking
 */

export interface FilterResult {
  cleanContent: string
  removedMarkers: string[]
  hasProgressMarkers: boolean
}

export function filterContent(content: string): FilterResult {
  if (!content) {
    return {
      cleanContent: '',
      removedMarkers: [],
      hasProgressMarkers: false
    }
  }

  const removedMarkers: string[] = []
  let cleaned = content

  // Define all progress marker patterns
  const progressPatterns = [
    // Phase markers - e.g., [PHASE:ANALYZING], [PHASE:SEARCHING]
    {
      pattern: /\[PHASE:[^\]]+\]/g,
      description: 'phase markers'
    },
    // Meta markers - e.g., [META:node:coordinator], [META:progress:50], [META:elapsed:1.2]
    {
      pattern: /\[META:[^\]]+\]/g,
      description: 'meta markers'
    },
    // Progress text patterns with emojis
    {
      pattern: /^[🔍📋🌐🗄️📊🤔✍️⚙️🎯🔬🔎📄]\s*.*?(?:Analyzing request|Gathering background|Created research plan|Research in progress|Fact checking|Synthesizing report|routing to appropriate agents|background information).*?$/gm,
      description: 'progress text with emojis'
    },
    // Standalone progress emojis at start of line
    {
      pattern: /^[🔍📋🌐🗄️📊🤔✍️⚙️🎯🔬🔎📄]\s+/gm,
      description: 'progress emojis'
    },
    // Lines that start with markers and emojis together
    {
      pattern: /^\[(?:PHASE|META):[^\]]+\].*?[🔍📋🌐🗄️📊🤔✍️⚙️🎯🔬🔎📄].*?$/gm,
      description: 'marker lines with emojis'
    },
    // Complete lines with multiple markers followed by emoji and text (exact user format)
    {
      pattern: /^\[PHASE:[^\]]+\]\s*\[META:[^\]]+\].*?[🔍📋🌐🗄️📊🤔✍️⚙️🎯🔬🔎📄].*?$/gm,
      description: 'full progress lines'
    },
    // Any line starting with bracket markers (aggressive catch-all)
    {
      pattern: /^.*?\[(?:PHASE|META):[^\]]+\].*$/gm,
      description: 'any line with markers'
    }
  ]

  // Apply each pattern and track what was removed
  for (const { pattern } of progressPatterns) {
    const matches = cleaned.match(pattern)
    if (matches) {
      removedMarkers.push(...matches)
      cleaned = cleaned.replace(pattern, '')
    }
  }

  // Aggressive cleanup of whitespace that results from marker removal
  cleaned = cleaned
    .split('\n') // Split into lines
    .map(line => line.trim()) // Trim each line
    .filter(line => line.length > 0) // Remove empty lines
    .filter(line => !/^\s*$/.test(line)) // Remove whitespace-only lines
    .join('\n') // Rejoin
    .replace(/\n{3,}/g, '\n\n') // Multiple newlines to double newlines
    .replace(/^\s+|\s+$/g, '') // Trim whitespace from start and end
    .replace(/\s{2,}/g, ' ') // Multiple spaces to single space

  return {
    cleanContent: cleaned,
    removedMarkers,
    hasProgressMarkers: removedMarkers.length > 0
  }
}

/**
 * Extract phase information from content before filtering
 */
export function extractPhaseInfo(content: string): {
  phase?: string
  node?: string
  progress?: number
  elapsed?: number
} {
  const phaseMatch = content.match(/\[PHASE:(\w+)\]/)
  const nodeMatch = content.match(/\[META:node:(\w+)\]/)
  const progressMatch = content.match(/\[META:progress:(\d+)\]/)
  const elapsedMatch = content.match(/\[META:elapsed:([\d.]+)\]/)

  return {
    phase: phaseMatch?.[1]?.toLowerCase(),
    node: nodeMatch?.[1]?.toLowerCase(),
    progress: progressMatch ? parseInt(progressMatch[1], 10) : undefined,
    elapsed: elapsedMatch ? parseFloat(elapsedMatch[1]) : undefined
  }
}

/**
 * Check if content contains progress markers
 */
export function hasProgressMarkers(content: string): boolean {
  const patterns = [
    /\[PHASE:\w+\]/,
    /\[META:\w+:[^\]]+\]/
  ]

  return patterns.some(pattern => pattern.test(content))
}

/**
 * Remove only phase and meta markers, preserve other content
 */
export function removeMarkersOnly(content: string): string {
  return content
    .replace(/\[PHASE:[^\]]+\]/g, '')
    .replace(/\[META:[^\]]+\]/g, '')
    .replace(/\s{2,}/g, ' ')
    .trim()
}

/**
 * Test function to verify filtering works correctly
 */
export function testFiltering(): void {
  const testContent = `[PHASE:ANALYZING] [META:node:coordinator] [META:progress:16] [META:elapsed:1.2]🎯 Analyzing request and routing to appropriate agents...
[PHASE:SEARCHING] [META:node:background_investigation] [META:progress:33] [META:elapsed:2.3]🔍 Gathering background information...

This is actual content that should remain.

More content here.`

  const result = filterContent(testContent)
  console.log('🔧 Content Filter Test:')
  console.log('📥 Input:', testContent)
  console.log('📤 Output:', result.cleanContent)
  console.log('🗑️ Removed:', result.removedMarkers)
  console.log('✅ Has markers:', result.hasProgressMarkers)
}