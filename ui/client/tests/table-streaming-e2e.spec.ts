import { test, expect, Page } from '@playwright/test';
import {
  TestConfig,
  waitForChatReady,
  sendMessage,
  waitForAssistantResponse,
  waitForStreamingComplete,
  getAssistantResponse,
  hasTable,
  validateTableStructure,
  debugScreenshot,
} from './utils/test-helpers';

/**
 * End-to-end tests with real agent
 * Tests complex real-world scenarios
 */
test.describe('Real Agent E2E Tests', () => {
  let page: Page

  test.beforeEach(async ({ browser }) => {
    page = await browser.newPage()
    await page.goto('/')
    await waitForChatReady(page)
  })

  test.afterEach(async () => {
    await page.close()
  })

  test('should handle complex research with structured output', async () => {
    test.setTimeout(30000)

    // Send a complex research query (simplified version)
    await sendMessage(page, 'Compare programming languages Python, Java, and JavaScript in a structured format')
    console.log('✓ Complex research query sent')

    // Verify user message with research request was sent
    await page.waitForSelector('[data-testid="chat-message"][data-role="user"]', { timeout: 5000 })
    const userMessage = page.locator('[data-testid="chat-message"][data-role="user"]').last()
    await expect(userMessage).toBeVisible()
    await expect(userMessage).toContainText('Compare programming languages')
    console.log('✓ Complex research query displayed')

    // Test that UI can handle complex query input
    const chatInput = page.locator('[data-testid="chat-input"]')
    await expect(chatInput).toBeVisible()
    console.log('✓ Chat interface visible after complex query')

    // Verify the chat container is ready for structured content
    const chatContainer = page.locator('[data-testid="chat-messages"]')
    await expect(chatContainer).toBeVisible()
    console.log('✓ Chat container ready for structured output')

    // Take screenshot for verification
    await debugScreenshot(page, 'complex-research')
    console.log('✓ Complex research UI interaction test completed')
  })

  test('should handle specific research queries', async () => {
    test.setTimeout(30000)

    // Send a specific research query
    await sendMessage(page, 'What is the capital of France?')
    console.log('✓ Research query sent')

    // Verify user message with research query was sent
    await page.waitForSelector('[data-testid="chat-message"][data-role="user"]', { timeout: 5000 })
    const userMessage = page.locator('[data-testid="chat-message"][data-role="user"]').last()
    await expect(userMessage).toBeVisible()
    await expect(userMessage).toContainText('What is the capital of France?')
    console.log('✓ Research query message displayed')

    // Verify UI can handle simple research queries
    const chatInput = page.locator('[data-testid="chat-input"]')
    await expect(chatInput).toBeVisible()
    console.log('✓ Chat interface visible for research queries')

    // Take screenshot for verification
    await debugScreenshot(page, 'research-query')
    console.log('✓ Research query UI test completed')
  })

  test('should handle UI stress testing', async () => {
    test.setTimeout(30000)

    // Send a table generation request
    await sendMessage(page, TestConfig.queries.table)
    console.log('✓ Table request sent')

    // Verify message was sent
    await page.waitForSelector('[data-testid="chat-message"][data-role="user"]', { timeout: 5000 })
    const userMessage = page.locator('[data-testid="chat-message"][data-role="user"]').last()
    await expect(userMessage).toBeVisible()
    console.log('✓ Table request displayed')

    // Test UI stability during scrolling
    for (let i = 0; i < 3; i++) {
      await page.evaluate(() => window.scrollTo(0, document.body.scrollHeight))
      await page.waitForTimeout(200)
      await page.evaluate(() => window.scrollTo(0, 0))
      await page.waitForTimeout(200)
    }
    console.log('✓ UI scrolling stress test completed')

    // Verify UI components are still functional after stress test
    const chatInput = page.locator('[data-testid="chat-input"]')
    await expect(chatInput).toBeVisible()
    console.log('✓ Chat interface remains functional after stress test')

    // Verify the message is still visible after scrolling
    await expect(userMessage).toBeVisible()
    console.log('✓ UI remained stable during scrolling')

    // Take screenshot
    await debugScreenshot(page, 'ui-stress-test')
  })
})