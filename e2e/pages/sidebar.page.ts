import { type Locator, type Page, expect } from '@playwright/test';

/**
 * Page Object Model for the Sidebar/Chat List.
 * Encapsulates selectors and common actions for sidebar interactions.
 */
export class SidebarPage {
  readonly page: Page;
  readonly chatList: Locator;
  readonly newChatButton: Locator;

  constructor(page: Page) {
    this.page = page;
    this.chatList = page.getByTestId('chat-list');
    this.newChatButton = page.getByTestId('new-chat-button');
  }

  /**
   * Create a new chat by clicking the new chat button.
   */
  async createNewChat(): Promise<void> {
    await this.newChatButton.click();
  }

  /**
   * Get the count of chats in the sidebar.
   * @returns Number of chat items
   */
  async getChatCount(): Promise<number> {
    const chatItems = this.page.locator('[data-testid^="chat-item-"]');
    return chatItems.count();
  }

  /**
   * Select a chat by index.
   * @param index The index of the chat to select (0-based)
   */
  async selectChat(index: number): Promise<void> {
    const chatItems = this.page.locator('[data-testid^="chat-item-"]');
    await chatItems.nth(index).click();
  }

  /**
   * Select a chat by its ID.
   * @param chatId The ID of the chat to select
   */
  async selectChatById(chatId: string): Promise<void> {
    await this.page.getByTestId(`chat-item-${chatId}`).click();
  }

  /**
   * Get the titles of all chats in the sidebar.
   * @returns Array of chat titles
   */
  async getChatTitles(): Promise<string[]> {
    const chatItems = this.page.locator('[data-testid^="chat-item-"]');
    const count = await chatItems.count();
    const titles: string[] = [];
    for (let i = 0; i < count; i++) {
      const text = await chatItems.nth(i).textContent();
      titles.push(text ?? '');
    }
    return titles;
  }

  /**
   * Check if the sidebar contains a chat with the given title.
   * @param title The title to search for
   * @returns True if a chat with the title exists
   */
  async hasChatWithTitle(title: string): Promise<boolean> {
    const titles = await this.getChatTitles();
    return titles.some((t) => t.includes(title));
  }

  /**
   * Wait for the chat list to be visible.
   */
  async waitForChatList(): Promise<void> {
    await expect(this.chatList).toBeVisible();
  }
}
