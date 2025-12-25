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
 * Note: The Deep Research Agent performs multi-step research even for simple queries,
 * so we use a longer timeout (3 minutes) to allow the research flow to complete.
 */
export const SMOKE_QUERIES: TestQuery[] = [
  {
    text: 'Hello',
    type: 'smoke',
    expectedResponseTimeMs: 180000,  // 3 minutes - research agent needs time
    description: 'Basic greeting',
  },
  {
    text: 'What is 2+2?',
    type: 'smoke',
    expectedResponseTimeMs: 180000,
    description: 'Simple arithmetic',
  },
  {
    text: 'What is Python?',
    type: 'smoke',
    expectedResponseTimeMs: 180000,
    description: 'Simple factual question',
  },
];

/**
 * Research queries - trigger full research flow, expect citations.
 */
export const RESEARCH_QUERIES: TestQuery[] = [
  {
    text: 'What are the latest developments in quantum computing error correction?',
    type: 'research',
    expectedResponseTimeMs: 180000,
    description: 'Complex research topic',
  },
  {
    text: 'Compare React and Vue.js for enterprise web applications',
    type: 'research',
    expectedResponseTimeMs: 180000,
    description: 'Comparison research',
  },
  {
    text: 'What are the current best practices for AI safety?',
    type: 'research',
    expectedResponseTimeMs: 180000,
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
 */
export const LONG_RUNNING_QUERIES: TestQuery[] = [
  {
    text: 'Write a comprehensive analysis of global economic trends over the past decade',
    type: 'long-running',
    expectedResponseTimeMs: 180000,
    description: 'Very long research task',
  },
  {
    text: 'Provide a detailed comparison of all major cloud computing platforms',
    type: 'long-running',
    expectedResponseTimeMs: 180000,
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
