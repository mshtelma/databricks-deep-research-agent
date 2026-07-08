import { defineConfig, devices } from '@playwright/test';

/**
 * Playwright configuration for Deep Research Agent E2E tests.
 * See https://playwright.dev/docs/test-configuration
 *
 * Two modes:
 * - Local (default): the `webServer` block boots a uvicorn process; tests run
 *   against http://localhost:8000.
 * - Deployed: set E2E_BASE_URL to the deployed app URL (e.g. a Databricks Apps
 *   URL). The local webServer is then NOT started, and E2E_BEARER_TOKEN (a
 *   workspace OAuth token) is attached as `Authorization: Bearer` on every
 *   request — the Apps proxy authenticates it and forwards it as the OBO user
 *   identity. See `make e2e-deployed`.
 */
const DEPLOYED = !!process.env.E2E_BASE_URL;
const BEARER = process.env.E2E_BEARER_TOKEN;

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
    // Deployed mode: Bearer authenticates at the Apps proxy (forwarded as OBO);
    // Origin satisfies the app's CSRF middleware (origin-allowlist) on POSTs.
    ...(DEPLOYED
      ? {
          extraHTTPHeaders: {
            ...(BEARER ? { Authorization: `Bearer ${BEARER}` } : {}),
            Origin: process.env.E2E_BASE_URL as string,
          },
        }
      : {}),
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
  // WebServer: Auto-start backend with static file serving (LOCAL mode only).
  // Skipped in deployed mode (E2E_BASE_URL set) — tests hit the deployed app.
  // Uses Lakebase from .env configuration (or local PostgreSQL if DATABASE_URL is set)
  // Uses lightweight E2E config for faster tests
  webServer: DEPLOYED
    ? undefined
    : {
        command: `cd ${__dirname}/.. && LAKEBASE_DATABASE=${process.env.LAKEBASE_DATABASE || 'deep_research_e2e'} APP_CONFIG_PATH=config/app.e2e.yaml SERVE_STATIC=true uv run uvicorn deep_research.main:app --host 0.0.0.0 --port 8000`,
        url: 'http://localhost:8000/health',
        reuseExistingServer: !process.env.CI, // Reuse locally, fresh in CI
        timeout: 120000,
        stdout: 'pipe',
        stderr: 'pipe',
      },
});
