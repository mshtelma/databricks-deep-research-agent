import { defineConfig, devices } from '@playwright/test';

const isRealAgent = process.env.TEST_AGENT_MODE === 'real';

export default defineConfig({
  testDir: './tests',
  
  // Configure for sequential execution when testing real agent
  fullyParallel: !isRealAgent,
  
  forbidOnly: !!process.env.CI,
  
  // More retries for real agent due to potential endpoint scaling issues
  retries: isRealAgent ? 3 : (process.env.CI ? 2 : 0),
  
  // Force single worker for real agent to prevent overwhelming the endpoint
  workers: isRealAgent ? 1 : (process.env.CI ? 1 : undefined),
  
  reporter: 'html',
  
  use: {
    baseURL: 'http://localhost:5173',
    trace: 'on-first-retry',
    
    // Slower navigation for real agent to allow for endpoint response times
    navigationTimeout: isRealAgent ? 60000 : 30000,
    actionTimeout: isRealAgent ? 30000 : 10000,
  },

  projects: [
    {
      name: 'chromium',
      use: { 
        ...devices['Desktop Chrome'],
        // Add delay between actions for real agent
        launchOptions: {
          slowMo: isRealAgent ? 1000 : 0, // 1 second delay between actions
        },
      },
    },
    // Optional: separate project specifically for real agent tests
    ...(isRealAgent ? [{
      name: 'real-agent',
      use: {
        ...devices['Desktop Chrome'],
        launchOptions: {
          slowMo: 2000, // Even slower for real agent tests
          devtools: true, // Enable devtools for debugging
        },
      },
    }] : []),
  ],

  webServer: {
    command: 'bash -lc "cd .. && SKIP_DATABRICKS_AUTH=1 ./watch.sh"',
    port: 5173,
    reuseExistingServer: true,
    timeout: 60000,
  },
});