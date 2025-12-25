import { test as base } from '@playwright/test';

/**
 * App-level fixture for base URL and authentication setup.
 * Extends the base Playwright test with app-specific configuration.
 */
export const test = base.extend({
  // Configure base URL from environment
  baseURL: async ({}, use) => {
    const baseURL = process.env.E2E_BASE_URL || 'http://localhost:8000';
    await use(baseURL);
  },

  // Auto-navigate to app and verify it's ready
  page: async ({ page, baseURL }, use) => {
    // Navigate to the app
    await page.goto(baseURL ?? '/');

    // Wait for the app to be ready (message input should be visible)
    await page.waitForSelector('[data-testid="message-input"]', {
      state: 'visible',
      timeout: 30000,
    });

    await use(page);
  },
});

export { expect } from '@playwright/test';
