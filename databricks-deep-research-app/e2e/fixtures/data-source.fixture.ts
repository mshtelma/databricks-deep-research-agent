/**
 * Data source fixture — extends the chat fixture with data source API support.
 *
 * Fixture chain: base (Playwright) -> chat.fixture -> data-source.fixture
 *
 * Provides:
 *   - All existing fixtures: chatPage, sidebarPage, researchPage, citationsPage
 *   - dsApi: DataSourceApiHelper with automatic cleanup after each test
 */

import { test as chatTest, expect } from './chat.fixture';
import { DataSourceApiHelper } from '../utils/data-source-api';

export interface DataSourceFixtures {
  dsApi: DataSourceApiHelper;
}

export const test = chatTest.extend<DataSourceFixtures>({
  dsApi: async ({ page }, use) => {
    const api = new DataSourceApiHelper(page);
    await use(api);
    // Auto-cleanup all tracked sources after each test
    await api.cleanupAll();
  },
});

export { expect };
