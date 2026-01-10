import { type Locator, type Page, expect } from '@playwright/test';

/**
 * Page Object Model for the Sidebar/Chat List.
 * Encapsulates selectors and common actions for sidebar interactions.
 */
export class SidebarPage {
  readonly page: Page;
  readonly chatList: Locator;
  readonly newChatButton: Locator;

  // Search
  readonly searchInput: Locator;
  readonly searchClear: Locator;

  // Status filter tabs
  readonly statusFilterActive: Locator;
  readonly statusFilterArchived: Locator;
  readonly statusFilterAll: Locator;

  // Empty/loading states
  readonly chatListEmpty: Locator;
  readonly chatListLoading: Locator;

  // Context menu actions
  readonly actionRename: Locator;
  readonly actionExport: Locator;
  readonly actionArchive: Locator;
  readonly actionRestore: Locator;
  readonly actionDelete: Locator;

  // Delete confirmation dialog
  readonly deleteDialogConfirm: Locator;

  constructor(page: Page) {
    this.page = page;
    this.chatList = page.getByTestId('chat-list');
    this.newChatButton = page.getByTestId('new-chat-button');

    // Search
    this.searchInput = page.getByTestId('chat-search-input');
    this.searchClear = page.getByTestId('chat-search-clear');

    // Status filter tabs
    this.statusFilterActive = page.getByTestId('status-filter-active');
    this.statusFilterArchived = page.getByTestId('status-filter-archived');
    this.statusFilterAll = page.getByTestId('status-filter-all');

    // Empty/loading states
    this.chatListEmpty = page.getByTestId('chat-list-empty');
    this.chatListLoading = page.getByTestId('chat-list-loading');

    // Context menu actions (visible after opening menu)
    this.actionRename = page.getByTestId('chat-action-rename');
    this.actionExport = page.getByTestId('chat-action-export');
    this.actionArchive = page.getByTestId('chat-action-archive');
    this.actionRestore = page.getByTestId('chat-action-restore');
    this.actionDelete = page.getByTestId('chat-action-delete');

    // Delete confirmation dialog - find the destructive button in the dialog
    this.deleteDialogConfirm = page.locator('[role="dialog"]').getByRole('button', { name: 'Delete' });
  }

  /**
   * Create a new chat by clicking the new chat button.
   */
  async createNewChat(): Promise<void> {
    await this.newChatButton.click();
  }

  /**
   * Create a new draft chat and wait for it to appear in the sidebar.
   * Note: Draft chats don't have context menu actions (rename, delete, archive).
   * Use createPersistedChat() for tests that need menu actions.
   * @param timeout Maximum time to wait for the chat to appear
   * @returns The ID of the newly created chat, or null if not found
   */
  async createNewChatAndWait(timeout: number = 10000): Promise<string | null> {
    await this.newChatButton.click();

    // Wait for URL to change to draft pattern: /chat/{uuid}?draft=1
    await expect(this.page).toHaveURL(/\/chat\/([a-f0-9-]+)\?draft=1/, { timeout });

    // Extract chat ID from URL
    const url = this.page.url();
    const match = url.match(/\/chat\/([a-f0-9-]+)\?draft=1/);
    if (!match) {
      return null;
    }
    const chatId = match[1];

    // Wait for the chat item to appear in the sidebar
    const chatItem = this.page.getByTestId(`chat-item-${chatId}`);
    await expect(chatItem).toBeVisible({ timeout });

    return chatId;
  }

  /**
   * Create a persisted chat via API and wait for it to appear in the sidebar.
   * Unlike draft chats, persisted chats have context menu actions (rename, delete, archive).
   * @param title Optional title for the chat
   * @param timeout Maximum time to wait for the chat to appear
   * @returns The ID of the newly created chat, or null if creation failed
   */
  async createPersistedChat(title?: string, timeout: number = 10000): Promise<string | null> {
    // Create chat via API
    const response = await this.page.request.post('/api/v1/chats', {
      data: title ? { title } : {},
    });

    if (!response.ok()) {
      console.error('Failed to create chat via API:', response.status());
      return null;
    }

    const chat = await response.json();
    const chatId = chat.id;

    // Navigate to the new chat
    await this.page.goto(`/chat/${chatId}`);

    // Wait for the chat item to appear in the sidebar
    const chatItem = this.page.getByTestId(`chat-item-${chatId}`);
    await expect(chatItem).toBeVisible({ timeout });

    return chatId;
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

  // ==================== Search Methods ====================

  /**
   * Search chats by title.
   */
  async searchChats(query: string): Promise<void> {
    await this.searchInput.fill(query);
  }

  /**
   * Clear the search input.
   */
  async clearSearch(): Promise<void> {
    const isVisible = await this.searchClear.isVisible().catch(() => false);
    if (isVisible) {
      await this.searchClear.click();
    } else {
      await this.searchInput.clear();
    }
  }

  /**
   * Get current search query.
   */
  async getSearchQuery(): Promise<string> {
    return (await this.searchInput.inputValue()) ?? '';
  }

  // ==================== Filter Methods ====================

  /**
   * Filter chats by status.
   */
  async filterByStatus(status: 'active' | 'archived' | 'all'): Promise<void> {
    switch (status) {
      case 'active':
        await this.statusFilterActive.click();
        break;
      case 'archived':
        await this.statusFilterArchived.click();
        break;
      case 'all':
        await this.statusFilterAll.click();
        break;
    }
  }

  // ==================== Context Menu Methods ====================

  /**
   * Get the menu trigger button for a chat.
   */
  getMenuTrigger(chatId: string): Locator {
    return this.page.getByTestId(`chat-menu-trigger-${chatId}`);
  }

  /**
   * Open the context menu for a chat.
   */
  async openChatMenu(chatId: string): Promise<void> {
    // First hover over the chat item to reveal the menu trigger
    const chatItem = this.page.getByTestId(`chat-item-${chatId}`);
    await chatItem.hover();
    // Then click the menu trigger
    const menuTrigger = this.getMenuTrigger(chatId);
    await menuTrigger.click();
  }

  /**
   * Rename a chat.
   * Note: This triggers a browser prompt() dialog which Playwright can handle.
   */
  async renameChat(chatId: string, newTitle: string): Promise<void> {
    await this.openChatMenu(chatId);

    // Set up dialog handler before clicking rename
    this.page.once('dialog', async (dialog) => {
      await dialog.accept(newTitle);
    });

    await this.actionRename.click();
  }

  /**
   * Cancel renaming a chat (dismiss the dialog).
   */
  async cancelRename(chatId: string): Promise<void> {
    await this.openChatMenu(chatId);

    this.page.once('dialog', async (dialog) => {
      await dialog.dismiss();
    });

    await this.actionRename.click();
  }

  /**
   * Delete a chat.
   * Opens context menu, clicks delete, then confirms in the dialog.
   */
  async deleteChat(chatId: string): Promise<void> {
    await this.openChatMenu(chatId);
    await this.actionDelete.click();
    // Wait for dialog to appear and click confirm
    await this.deleteDialogConfirm.waitFor({ state: 'visible' });
    await this.deleteDialogConfirm.click();
  }

  /**
   * Archive a chat.
   */
  async archiveChat(chatId: string): Promise<void> {
    await this.openChatMenu(chatId);
    await this.actionArchive.click();
  }

  /**
   * Restore (unarchive) a chat.
   */
  async restoreChat(chatId: string): Promise<void> {
    await this.openChatMenu(chatId);
    await this.actionRestore.click();
  }

  /**
   * Export a chat.
   */
  async exportChat(chatId: string): Promise<void> {
    await this.openChatMenu(chatId);
    await this.actionExport.click();
  }

  /**
   * Close context menu by pressing Escape.
   */
  async closeMenu(): Promise<void> {
    await this.page.keyboard.press('Escape');
  }

  // ==================== State Check Methods ====================

  /**
   * Check if the chat list is empty.
   */
  async isChatListEmpty(): Promise<boolean> {
    return this.chatListEmpty.isVisible();
  }

  /**
   * Check if the chat list is loading.
   */
  async isChatListLoading(): Promise<boolean> {
    return this.chatListLoading.isVisible();
  }

  /**
   * Wait for a chat to disappear from the list.
   */
  async waitForChatToDisappear(chatId: string, timeout: number = 5000): Promise<void> {
    const chatItem = this.page.getByTestId(`chat-item-${chatId}`);
    await expect(chatItem).toBeHidden({ timeout });
  }

  /**
   * Wait for a chat to appear in the list.
   */
  async waitForChatToAppear(chatId: string, timeout: number = 5000): Promise<void> {
    const chatItem = this.page.getByTestId(`chat-item-${chatId}`);
    await expect(chatItem).toBeVisible({ timeout });
  }

  /**
   * Get the first chat item in the list.
   */
  async getFirstChatId(): Promise<string | null> {
    const chatItems = this.page.locator('[data-testid^="chat-item-"]');
    const count = await chatItems.count();
    if (count === 0) return null;

    const testId = await chatItems.first().getAttribute('data-testid');
    return testId?.replace('chat-item-', '') ?? null;
  }

  /**
   * Check if a chat is selected (has active styling).
   */
  async isChatSelected(chatId: string): Promise<boolean> {
    const chatItem = this.page.getByTestId(`chat-item-${chatId}`);
    const classes = await chatItem.getAttribute('class');
    return classes?.includes('bg-accent') ?? false;
  }
}
