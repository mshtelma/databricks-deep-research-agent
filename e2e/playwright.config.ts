import { defineConfig, devices } from '@playwright/test';

/**
 * Playwright configuration for Deep Research Agent E2E tests.
 * See https://playwright.dev/docs/test-configuration
 */
export default defineConfig({
  testDir: './tests',
  timeout: 120000, // 2 minutes for research operations
  expect: {
    timeout: 10000, // 10s for assertions
  },
  fullyParallel: false, // Run tests sequentially to avoid race conditions with shared SSE server
  forbidOnly: !!process.env.CI,
  retries: process.env.CI ? 2 : 1, // Add 1 retry locally for flaky tests
  workers: 1, // Single worker for stability with SSE streaming
  reporter: [
    ['html', { open: 'never' }],
    ['list'],
    ['json', { outputFile: 'test-results.json' }],
  ],
  use: {
    baseURL: process.env.E2E_BASE_URL || 'http://localhost:8000',
    trace: 'on-first-retry',
    screenshot: 'only-on-failure',
    video: 'retain-on-failure',
  },
  projects: [
    {
      name: 'chromium',
      use: { ...devices['Desktop Chrome'] },
    },
    // Uncomment for cross-browser testing:
    // {
    //   name: 'firefox',
    //   use: { ...devices['Desktop Firefox'] },
    // },
    // {
    //   name: 'webkit',
    //   use: { ...devices['Desktop Safari'] },
    // },
  ],
  // WebServer: Auto-start backend with static file serving
  // Uses Lakebase from .env configuration (or local PostgreSQL if DATABASE_URL is set)
  // Uses lightweight E2E config for faster tests
  webServer: {
    command: 'cd .. && LAKEBASE_DATABASE=deep_research APP_CONFIG_PATH=config/app.e2e.yaml SERVE_STATIC=true uv run uvicorn deep_research.main:app --host 0.0.0.0 --port 8000',
    url: 'http://localhost:8000/health',
    reuseExistingServer: !process.env.CI, // Reuse locally, fresh in CI
    timeout: 120000,
    stdout: 'pipe',
    stderr: 'pipe',
  },
});
