/**
 * Test data generators and constants for E2E tests.
 */

/**
 * Test query types with expected response characteristics.
 */
export interface TestQuery {
  text: string;
  type: 'smoke' | 'research' | 'follow-up' | 'long-running';
  expectedResponseTimeMs: number;
  description: string;
}

/**
 * Smoke test queries - simple, fast responses.
 * Note: The Deep Research Agent performs multi-step research even for simple queries.
 * With real LLM calls, web searches, and citation verification, research flows
 * typically take 3-5 minutes. Use 6-minute timeout to provide buffer.
 */
export const SMOKE_QUERIES: TestQuery[] = [
  {
    text: 'Hello',
    type: 'smoke',
    expectedResponseTimeMs: 360000,  // 6 minutes - research agent with citation verification
    description: 'Basic greeting',
  },
  {
    text: 'What is 2+2?',
    type: 'smoke',
    expectedResponseTimeMs: 360000,
    description: 'Simple arithmetic',
  },
  {
    text: 'What is Python?',
    type: 'smoke',
    expectedResponseTimeMs: 360000,
    description: 'Simple factual question',
  },
];

/**
 * Research queries - trigger full research flow, expect citations.
 * These queries involve web search, crawling, synthesis, and citation verification.
 * Typical duration is 3-5 minutes. Use 6-minute timeout for reliability.
 */
export const RESEARCH_QUERIES: TestQuery[] = [
  {
    text: 'What are the latest developments in quantum computing error correction?',
    type: 'research',
    expectedResponseTimeMs: 360000,  // 6 minutes for full research with citations
    description: 'Complex research topic',
  },
  {
    text: 'Compare React and Vue.js for enterprise web applications',
    type: 'research',
    expectedResponseTimeMs: 360000,
    description: 'Comparison research',
  },
  {
    text: 'What are the current best practices for AI safety?',
    type: 'research',
    expectedResponseTimeMs: 360000,
    description: 'Current events research',
  },
];

/**
 * Follow-up queries - context-dependent questions.
 */
export const FOLLOW_UP_QUERIES: TestQuery[] = [
  {
    text: 'Can you give me an example?',
    type: 'follow-up',
    expectedResponseTimeMs: 30000,
    description: 'Example request',
  },
  {
    text: 'What about the performance implications?',
    type: 'follow-up',
    expectedResponseTimeMs: 30000,
    description: 'Performance follow-up',
  },
  {
    text: 'How does this compare to alternatives?',
    type: 'follow-up',
    expectedResponseTimeMs: 30000,
    description: 'Comparison follow-up',
  },
];

/**
 * Long-running queries - for testing stop/cancel functionality.
 * These are deliberately complex queries that take a long time.
 */
export const LONG_RUNNING_QUERIES: TestQuery[] = [
  {
    text: 'Write a comprehensive analysis of global economic trends over the past decade',
    type: 'long-running',
    expectedResponseTimeMs: 420000,  // 7 minutes for extensive research
    description: 'Very long research task',
  },
  {
    text: 'Provide a detailed comparison of all major cloud computing platforms',
    type: 'long-running',
    expectedResponseTimeMs: 420000,
    description: 'Extensive comparison',
  },
];

/**
 * Get a random query of the specified type.
 * @param type The type of query to get
 * @returns A random query of the specified type
 */
export function getRandomQuery(type: TestQuery['type']): TestQuery {
  let queries: TestQuery[];
  switch (type) {
    case 'smoke':
      queries = SMOKE_QUERIES;
      break;
    case 'research':
      queries = RESEARCH_QUERIES;
      break;
    case 'follow-up':
      queries = FOLLOW_UP_QUERIES;
      break;
    case 'long-running':
      queries = LONG_RUNNING_QUERIES;
      break;
    default:
      queries = SMOKE_QUERIES;
  }
  return queries[Math.floor(Math.random() * queries.length)];
}

/**
 * Generate a unique test identifier for tracking.
 * @returns A unique string identifier
 */
export function generateTestId(): string {
  return `test-${Date.now()}-${Math.random().toString(36).substring(7)}`;
}

/**
 * Context patterns that indicate context-aware responses.
 */
export const CONTEXT_PATTERNS = {
  machineLearning: /machine|learning|ml|algorithm|model|training/i,
  python: /python|programming|language|code/i,
  javascript: /javascript|js|typescript|node|react/i,
  general: /example|instance|case|scenario/i,
};

/**
 * Error patterns for validating error handling.
 */
export const ERROR_PATTERNS = {
  networkError: /network|connection|failed|offline/i,
  timeout: /timeout|timed out|took too long/i,
  serverError: /server|500|internal|error/i,
  rateLimit: /rate limit|too many|slow down/i,
};

// ==================== Parallel Research Test Data ====================

/**
 * Query mode types for parallel research tests.
 */
export type QueryMode = 'simple' | 'web_search' | 'deep_research';

/**
 * Parallel research query configuration.
 */
export interface ParallelResearchQuery {
  initial: {
    text: string;
    mode: QueryMode;
    keywords: string[];
  };
  followUp: {
    text: string;
    mode: QueryMode;
    expectedContext: string[];
  };
}

/**
 * Parallel research queries - distinct domains for isolation verification.
 * Each query targets a different domain to detect cross-contamination.
 */
export const PARALLEL_RESEARCH_QUERIES: ParallelResearchQuery[] = [
  {
    initial: {
      text: 'quantum computing error correction',
      mode: 'deep_research',
      keywords: ['quantum', 'qubit', 'error', 'correction'],
    },
    followUp: {
      text: 'What companies are leading in this field?',
      mode: 'simple',
      expectedContext: ['quantum', 'computing'],
    },
  },
  {
    initial: {
      text: 'React vs Vue.js for enterprise applications',
      mode: 'deep_research',
      keywords: ['react', 'vue', 'enterprise', 'frontend'],
    },
    followUp: {
      text: 'Which framework has better performance?',
      mode: 'web_search',
      expectedContext: ['react', 'vue', 'performance'],
    },
  },
  {
    initial: {
      text: 'Kubernetes security best practices',
      mode: 'deep_research',
      keywords: ['kubernetes', 'container', 'security', 'k8s'],
    },
    followUp: {
      text: 'Give me a simple checklist for securing clusters',
      mode: 'simple',
      expectedContext: ['kubernetes', 'security', 'cluster'],
    },
  },
];

/**
 * Follow-up flow types for testing different mode transitions.
 */
export const FOLLOW_UP_FLOWS = {
  researchToSimple: { initial: 'deep_research' as QueryMode, followUp: 'simple' as QueryMode },
  researchToWebSearch: { initial: 'deep_research' as QueryMode, followUp: 'web_search' as QueryMode },
  webSearchToWebSearch: { initial: 'web_search' as QueryMode, followUp: 'web_search' as QueryMode },
  researchToResearch: { initial: 'deep_research' as QueryMode, followUp: 'deep_research' as QueryMode },
};

/**
 * Simple test queries for web_search mode testing.
 */
export const WEB_SEARCH_QUERIES = [
  {
    text: 'Latest news on electric vehicles 2024',
    keywords: ['electric', 'ev', 'vehicle', 'car'],
  },
  {
    text: 'Current stock market trends',
    keywords: ['stock', 'market', 'trading', 'investment'],
  },
];

/**
 * Timeout configuration for parallel tests (in milliseconds).
 */
export const PARALLEL_TIMEOUTS = {
  deepResearch: 360000,    // 6 minutes for deep research
  webSearch: 60000,        // 1 minute for web search
  simple: 30000,           // 30 seconds for simple queries
  parallelAll: 420000,     // 7 minutes for all parallel operations
  fullScenario: 900000,    // 15 minutes for complete scenario
};
