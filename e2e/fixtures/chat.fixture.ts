import { test as base } from '@playwright/test';
import { ChatPage } from '../pages/chat.page';
import { SidebarPage } from '../pages/sidebar.page';
import { ResearchPage } from '../pages/research.page';

/**
 * Chat-specific fixture that provides page objects for chat testing.
 * Extends the base test with ChatPage, SidebarPage, and ResearchPage instances.
 */
export interface ChatFixtures {
  chatPage: ChatPage;
  sidebarPage: SidebarPage;
  researchPage: ResearchPage;
}

export const test = base.extend<ChatFixtures>({
  chatPage: async ({ page }, use) => {
    const chatPage = new ChatPage(page);
    await chatPage.goto();
    await use(chatPage);
  },

  sidebarPage: async ({ page }, use) => {
    const sidebarPage = new SidebarPage(page);
    await use(sidebarPage);
  },

  researchPage: async ({ page }, use) => {
    const researchPage = new ResearchPage(page);
    await use(researchPage);
  },
});

export { expect } from '@playwright/test';
