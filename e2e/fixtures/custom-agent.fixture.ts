/**
 * Custom agent fixture — extends the chat fixture with agent-specific support.
 *
 * Fixture chain: base (Playwright) → chat.fixture → custom-agent.fixture
 *
 * Provides:
 *   - All existing fixtures: chatPage, sidebarPage, researchPage, citationsPage
 *   - customAgentPage: Page object for agent selector UI interactions
 *   - agentApi: AgentApiHelper with automatic cleanup after each test
 */

import { test as chatTest, expect } from './chat.fixture';
import { CustomAgentPage } from '../pages/custom-agent.page';
import { AgentApiHelper } from '../utils/custom-agent-api';

export interface CustomAgentFixtures {
  customAgentPage: CustomAgentPage;
  agentApi: AgentApiHelper;
}

export const test = chatTest.extend<CustomAgentFixtures>({
  customAgentPage: async ({ page, chatPage: _chatPage }, use) => {
    await use(new CustomAgentPage(page));
  },

  agentApi: async ({ page }, use) => {
    const api = new AgentApiHelper(page);
    await use(api);
    // Auto-cleanup all tracked agents after each test
    await api.cleanupAll();
  },
});

export { expect };
