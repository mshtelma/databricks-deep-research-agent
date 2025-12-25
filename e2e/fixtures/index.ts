/**
 * Re-exports all fixtures for convenient importing.
 *
 * Usage:
 *   import { test, expect } from '../fixtures';
 *
 * The test object includes:
 *   - chatPage: ChatPage instance for chat interactions
 *   - sidebarPage: SidebarPage instance for sidebar interactions
 *   - researchPage: ResearchPage instance for research panel interactions
 */
export { test, expect } from './chat.fixture';
export type { ChatFixtures } from './chat.fixture';
