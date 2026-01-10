import { test as base } from '@playwright/test';
import { ChatPage } from '../pages/chat.page';
import { SidebarPage } from '../pages/sidebar.page';
import { ResearchPage } from '../pages/research.page';
import { CitationsPage } from '../pages/citations.page';

/**
 * Chat-specific fixture that provides page objects for chat testing.
 * Extends the base test with ChatPage, SidebarPage, ResearchPage, and CitationsPage instances.
 */
export interface ChatFixtures {
  chatPage: ChatPage;
  sidebarPage: SidebarPage;
  researchPage: ResearchPage;
  citationsPage: CitationsPage;
}

export const test = base.extend<ChatFixtures>({
  chatPage: async ({ page }, use) => {
    // Clear localStorage to remove stale drafts before each test
    await page.goto('/');
    await page.evaluate(() => localStorage.clear());
    await page.reload();
    const chatPage = new ChatPage(page);
    await chatPage.waitForReady();
    await use(chatPage);
  },

  sidebarPage: async ({ page }, use) => {
    // Clear localStorage to remove stale drafts before each test
    await page.goto('/');
    await page.evaluate(() => localStorage.clear());
    await page.reload();
    const sidebarPage = new SidebarPage(page);
    await sidebarPage.newChatButton.waitFor({ state: 'visible', timeout: 30000 });
    await use(sidebarPage);
  },

  researchPage: async ({ page }, use) => {
    const researchPage = new ResearchPage(page);
    await use(researchPage);
  },

  citationsPage: async ({ page }, use) => {
    const citationsPage = new CitationsPage(page);
    await use(citationsPage);
  },
});

export { expect } from '@playwright/test';
