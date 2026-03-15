/**
 * Helper utilities for parallel research E2E tests.
 *
 * These helpers enable running multiple research sessions in parallel
 * using isolated browser contexts to prevent interference.
 */

import { type Browser, type BrowserContext, type Page } from '@playwright/test';
import { ChatPage } from '../pages/chat.page';
import { type QueryMode, PARALLEL_TIMEOUTS } from './test-data';

/**
 * Represents an isolated research session with its own browser context.
 */
export interface ParallelSession {
  /** Unique identifier for this session */
  id: number;
  /** Isolated browser context */
  context: BrowserContext;
  /** Page instance within the context */
  page: Page;
  /** ChatPage object for interacting with the chat UI */
  chatPage: ChatPage;
}

/**
 * Result from executing a research query in a session.
 */
export interface ResearchResult {
  /** Session identifier */
  sessionId: number;
  /** Whether the query completed successfully */
  success: boolean;
  /** The response text (if successful) */
  response: string;
  /** The chat ID from the URL */
  chatId: string | null;
  /** Error message (if failed) */
  error?: string;
  /** Time taken in milliseconds */
  durationMs: number;
}

/**
 * Result from isolation verification.
 */
export interface IsolationResult {
  /** Whether all sessions are properly isolated */
  isolated: boolean;
  /** Details about any cross-contamination found */
  contaminations: Array<{
    sessionId: number;
    foundKeywordsFromSession: number;
    keywords: string[];
  }>;
}

/**
 * Create multiple isolated browser contexts for parallel testing.
 * Each session has its own localStorage, cookies, and page instance.
 *
 * @param browser The Playwright browser instance
 * @param count Number of parallel sessions to create
 * @returns Array of ParallelSession objects
 */
export async function createParallelSessions(
  browser: Browser,
  count: number
): Promise<ParallelSession[]> {
  const sessions: ParallelSession[] = [];

  for (let i = 0; i < count; i++) {
    // Create isolated browser context with unique storage state
    const context = await browser.newContext({
      // Each context has its own storage
      storageState: undefined,
    });

    const page = await context.newPage();

    // Navigate to the app and clear localStorage to ensure clean state
    await page.goto('/');
    await page.evaluate(() => localStorage.clear());
    await page.reload();

    const chatPage = new ChatPage(page);
    await chatPage.waitForReady();

    sessions.push({
      id: i,
      context,
      page,
      chatPage,
    });
  }

  return sessions;
}

/**
 * Clean up all parallel sessions by closing their contexts.
 * Should be called in test cleanup/afterEach.
 *
 * @param sessions Array of ParallelSession objects to clean up
 */
export async function cleanupParallelSessions(sessions: ParallelSession[]): Promise<void> {
  const cleanupPromises = sessions.map(async (session) => {
    try {
      await session.context.close();
    } catch (error) {
      // Ignore errors during cleanup - context might already be closed
      console.warn(`Failed to close context for session ${session.id}:`, error);
    }
  });

  await Promise.all(cleanupPromises);
}

/**
 * Get the timeout for a specific query mode.
 *
 * @param mode The query mode
 * @returns Timeout in milliseconds
 */
export function getTimeoutForMode(mode: QueryMode): number {
  switch (mode) {
    case 'deep_research':
      return PARALLEL_TIMEOUTS.deepResearch;
    case 'web_search':
      return PARALLEL_TIMEOUTS.webSearch;
    case 'simple':
      return PARALLEL_TIMEOUTS.simple;
    default:
      return PARALLEL_TIMEOUTS.deepResearch;
  }
}

/**
 * Execute a single research query in a session.
 *
 * @param session The parallel session to use
 * @param query The query text to send
 * @param mode The query mode (simple, web_search, deep_research)
 * @param timeout Maximum wait time in milliseconds (auto-detected from mode if not provided)
 * @returns ResearchResult with success status and response
 */
export async function executeResearchInSession(
  session: ParallelSession,
  query: string,
  mode: QueryMode,
  timeout?: number
): Promise<ResearchResult> {
  const startTime = Date.now();
  const effectiveTimeout = timeout ?? getTimeoutForMode(mode);

  try {
    // Send the message with the specified mode
    await session.chatPage.sendMessageWithMode(query, mode);

    // Wait for response
    await session.chatPage.waitForAgentResponse(effectiveTimeout);

    // Get the response
    const response = await session.chatPage.getLastAgentResponse();
    const chatId = await session.chatPage.getChatIdFromUrl();

    return {
      sessionId: session.id,
      success: true,
      response,
      chatId,
      durationMs: Date.now() - startTime,
    };
  } catch (error) {
    return {
      sessionId: session.id,
      success: false,
      response: '',
      chatId: null,
      error: error instanceof Error ? error.message : String(error),
      durationMs: Date.now() - startTime,
    };
  }
}

/**
 * Execute a follow-up query in a session (assumes initial query already done).
 *
 * @param session The parallel session to use
 * @param query The follow-up query text
 * @param mode The query mode for the follow-up
 * @param expectedResponseNumber Which response number to wait for (usually 2 for first follow-up)
 * @param timeout Maximum wait time in milliseconds
 * @returns ResearchResult with success status and response
 */
export async function executeFollowUpInSession(
  session: ParallelSession,
  query: string,
  mode: QueryMode,
  expectedResponseNumber: number,
  timeout?: number
): Promise<ResearchResult> {
  const startTime = Date.now();
  const effectiveTimeout = timeout ?? getTimeoutForMode(mode);

  try {
    // Send the follow-up message
    await session.chatPage.sendMessageWithMode(query, mode);

    // Wait for the expected response number
    await session.chatPage.waitForNthAgentResponse(expectedResponseNumber, effectiveTimeout);

    // Get the latest response
    const response = await session.chatPage.getLastAgentResponse();
    const chatId = await session.chatPage.getChatIdFromUrl();

    return {
      sessionId: session.id,
      success: true,
      response,
      chatId,
      durationMs: Date.now() - startTime,
    };
  } catch (error) {
    return {
      sessionId: session.id,
      success: false,
      response: '',
      chatId: null,
      error: error instanceof Error ? error.message : String(error),
      durationMs: Date.now() - startTime,
    };
  }
}

/**
 * Query configuration for parallel execution.
 */
export interface ParallelQueryConfig {
  query: string;
  mode: QueryMode;
}

/**
 * Execute multiple research queries in parallel using Promise.all().
 *
 * @param sessions Array of parallel sessions
 * @param queries Array of query configurations (must match sessions length)
 * @param timeout Maximum wait time per query in milliseconds
 * @returns Array of ResearchResult objects
 */
export async function executeParallelResearch(
  sessions: ParallelSession[],
  queries: ParallelQueryConfig[],
  timeout: number = PARALLEL_TIMEOUTS.parallelAll
): Promise<ResearchResult[]> {
  if (sessions.length !== queries.length) {
    throw new Error(
      `Session count (${sessions.length}) must match query count (${queries.length})`
    );
  }

  // Execute all queries in parallel
  const results = await Promise.all(
    sessions.map((session, index) =>
      executeResearchInSession(session, queries[index].query, queries[index].mode, timeout)
    )
  );

  return results;
}

/**
 * Execute parallel follow-up queries (assumes initial queries already completed).
 *
 * @param sessions Array of parallel sessions
 * @param queries Array of follow-up query configurations
 * @param expectedResponseNumber Which response number to wait for
 * @param timeout Maximum wait time per query
 * @returns Array of ResearchResult objects
 */
export async function executeParallelFollowUps(
  sessions: ParallelSession[],
  queries: ParallelQueryConfig[],
  expectedResponseNumber: number = 2,
  timeout?: number
): Promise<ResearchResult[]> {
  if (sessions.length !== queries.length) {
    throw new Error(
      `Session count (${sessions.length}) must match query count (${queries.length})`
    );
  }

  const results = await Promise.all(
    sessions.map((session, index) =>
      executeFollowUpInSession(
        session,
        queries[index].query,
        queries[index].mode,
        expectedResponseNumber,
        timeout
      )
    )
  );

  return results;
}

/**
 * Verify that responses are properly isolated and no cross-contamination occurred.
 * Checks that each response contains its own domain keywords and NOT keywords
 * from other sessions' queries.
 *
 * @param results Array of research results
 * @param queryKeywords Array of keyword arrays, one per query (matching results order)
 * @returns IsolationResult with isolation status and any contaminations found
 */
export function verifyResponseIsolation(
  results: ResearchResult[],
  queryKeywords: string[][]
): IsolationResult {
  const contaminations: IsolationResult['contaminations'] = [];

  for (let i = 0; i < results.length; i++) {
    const result = results[i];
    if (!result.success) continue;

    const responseLower = result.response.toLowerCase();

    // Check for keywords from OTHER sessions that shouldn't be present
    for (let j = 0; j < queryKeywords.length; j++) {
      if (i === j) continue; // Skip own keywords

      const foreignKeywords = queryKeywords[j];
      const foundForeignKeywords = foreignKeywords.filter((keyword) =>
        responseLower.includes(keyword.toLowerCase())
      );

      // If we find keywords from another session, it might indicate contamination
      // Note: Some generic terms might overlap, so we use a threshold
      if (foundForeignKeywords.length >= 2) {
        contaminations.push({
          sessionId: i,
          foundKeywordsFromSession: j,
          keywords: foundForeignKeywords,
        });
      }
    }
  }

  return {
    isolated: contaminations.length === 0,
    contaminations,
  };
}

/**
 * Verify that a follow-up response maintains context from the initial query.
 *
 * @param response The follow-up response text
 * @param expectedContextKeywords Keywords that should appear in the response
 * @returns Object with match status and found keywords
 */
export function verifyContextMaintained(
  response: string,
  expectedContextKeywords: string[]
): { maintained: boolean; foundKeywords: string[]; missingKeywords: string[] } {
  const responseLower = response.toLowerCase();

  const foundKeywords = expectedContextKeywords.filter((keyword) =>
    responseLower.includes(keyword.toLowerCase())
  );

  const missingKeywords = expectedContextKeywords.filter(
    (keyword) => !responseLower.includes(keyword.toLowerCase())
  );

  // Context is maintained if at least one expected keyword is found
  return {
    maintained: foundKeywords.length >= 1,
    foundKeywords,
    missingKeywords,
  };
}

/**
 * Wait for all sessions to be ready for interaction.
 *
 * @param sessions Array of parallel sessions
 * @param timeout Timeout per session in milliseconds
 */
export async function waitForAllSessionsReady(
  sessions: ParallelSession[],
  timeout: number = 30000
): Promise<void> {
  await Promise.all(sessions.map((session) => session.chatPage.waitForReady(timeout)));
}

/**
 * Get message counts from all sessions.
 *
 * @param sessions Array of parallel sessions
 * @returns Array of message count objects
 */
export async function getAllMessageCounts(
  sessions: ParallelSession[]
): Promise<Array<{ sessionId: number; user: number; agent: number }>> {
  const counts = await Promise.all(
    sessions.map(async (session) => {
      const count = await session.chatPage.getMessageCount();
      return {
        sessionId: session.id,
        ...count,
      };
    })
  );

  return counts;
}
