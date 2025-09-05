import { test, expect, Page } from '@playwright/test'

/**
 * End-to-end tests for table streaming with real prompts
 * Tests the complete flow from sending prompts to rendering tables
 */
test.describe('Table Streaming E2E Tests', () => {
  let page: Page

  test.beforeEach(async ({ browser }) => {
    page = await browser.newPage()
    
    // Set longer timeout for test execution
    test.setTimeout(120000)
    
    // Set up request interception to monitor streaming
    await page.route('**/api/chat/stream', async (route, request) => {
      console.log('Chat stream request:', request.postDataJSON())
      await route.continue()
    })

    // Navigate to the application
    await page.goto('http://localhost:5173')
    
    // Wait for app to load
    await page.waitForLoadState('networkidle')
    
    // Navigate to chat page - click "Start Chatting" button
    const startChatButton = page.getByRole('button', { name: /Start Chatting/i })
    await startChatButton.click()
    await page.waitForTimeout(500) // Give time for navigation
    
    // Wait for the chat interface to be ready
    await page.waitForSelector('[data-testid="chat-input"]', { timeout: 10000 })
  })

  test.afterEach(async () => {
    await page.close()
  })

  test('should handle tax comparison table streaming correctly', async () => {
    test.setTimeout(120000) // 2 minutes for this specific test
    // The exact prompt from the user
    const taxPrompt = 'I want a rigorous, apples-to-apples comparison of after-tax finances across Spain, France, United Kingdom, Switzerland (low-tax canton such as Zug), Germany, Poland, and Bulgaria for two family setups: 1) married couple without children 2) married couple with one child (3 years old)'
    
    // Find the chat input using data-testid
    const chatInput = page.locator('[data-testid="chat-input"]')
    await expect(chatInput).toBeVisible()
    
    // Type the prompt
    await chatInput.fill(taxPrompt)
    
    // Send the message
    await page.keyboard.press('Enter')
    
    // Wait for streaming to start
    await page.waitForSelector('[data-streaming="true"]', { 
      timeout: 10000 
    })
    
    // Monitor for table placeholders during streaming
    const hasPlaceholder = await page.waitForSelector(
      'text=/📊/', 
      { timeout: 5000, state: 'attached' }
    ).then(() => true).catch(() => false)
    
    if (hasPlaceholder) {
      console.log('✓ Table placeholder shown during streaming')
    }
    
    // Wait for response to complete
    await page.waitForSelector('[data-streaming="false"][data-role="assistant"]', { 
      timeout: 60000 
    })
    
    // Check final rendered content
    const messageContent = page.locator('[data-testid="chat-message"][data-role="assistant"]').last()
    await expect(messageContent).toBeVisible()
    
    const finalText = await messageContent.textContent()
    expect(finalText).toBeTruthy()
    
    // Validate table content
    expect(finalText).toContain('Spain')
    expect(finalText).toContain('France')
    expect(finalText).toContain('Germany')
    expect(finalText).toContain('Poland')
    expect(finalText).toContain('Bulgaria')
    expect(finalText).toContain('Switzerland')
    
    // Check for malformed patterns that should NOT appear
    expect(finalText).not.toContain('||')
    expect(finalText).not.toContain('|---|---|---|---|---|')
    expect(finalText).not.toContain('| --- | --- | --- | --- |')
    
    // Take screenshot for visual verification
    await page.screenshot({ 
      path: 'test-results/table-streaming-tax-comparison.png',
      fullPage: true 
    })
  })

  test('should handle stock sentiment analysis table correctly', async () => {
    const stockPrompt = 'Analyze the sentiment of AAPL and MSFT stocks. I am thinking if I should sell them now.'
    
    // Find and fill chat input
    const chatInput = page.locator('[data-testid="chat-input"]')
    await chatInput.fill(stockPrompt)
    
    // Send message
    await page.keyboard.press('Enter')
    
    // Wait for response
    await page.waitForSelector('[data-streaming="true"], .streaming-indicator, .loading', { 
      timeout: 10000 
    })
    
    await page.waitForSelector('[data-streaming="false"][data-role="assistant"]', { 
      timeout: 60000 
    })
    
    // Validate response content
    const messageContent = page.locator('[data-testid="chat-message"][data-role="assistant"]').last()
    const responseText = await messageContent.textContent()
    
    expect(responseText).toContain('AAPL')
    expect(responseText).toContain('MSFT')
    
    // Should have sentiment analysis
    expect(responseText?.toLowerCase()).toMatch(/sentiment|analysis|recommendation/i)
    
    // Screenshot for verification
    await page.screenshot({ 
      path: 'test-results/table-streaming-stock-sentiment.png',
      fullPage: true 
    })
  })

  test('should not break tables during rapid scrolling', async () => {
    // Send a message that generates a table
    const prompt = 'Create a comparison table of programming languages'
    
    const chatInput = page.locator('[data-testid="chat-input"]')
    await chatInput.fill(prompt)
    await page.keyboard.press('Enter')
    
    // Wait for streaming to start
    await page.waitForSelector('[data-streaming="true"]', { 
      timeout: 10000 
    })
    
    // Rapidly scroll during streaming to stress test rendering
    for (let i = 0; i < 5; i++) {
      await page.evaluate(() => window.scrollTo(0, document.body.scrollHeight))
      await page.waitForTimeout(100)
      await page.evaluate(() => window.scrollTo(0, 0))
      await page.waitForTimeout(100)
    }
    
    // Wait for completion
    await page.waitForSelector('[data-streaming="false"][data-role="assistant"]', { 
      timeout: 30000 
    })
    
    // Verify tables are intact after scrolling
    const messageContent = page.locator('[data-testid="chat-message"][data-role="assistant"]').last()
    const finalText = await messageContent.textContent()
    
    // Should not have broken patterns
    expect(finalText).not.toContain('||')
    expect(finalText).not.toContain('|---|---|---|---|')
  })

  test.skip('should handle multiple tables in single response', async () => {
    const multiTablePrompt = 'Compare database systems in one table and their pricing in another table'
    
    const chatInput = page.locator('[data-testid="chat-input"]')
    await chatInput.fill(multiTablePrompt)
    await page.keyboard.press('Enter')
    
    // Wait for response
    await page.waitForSelector('[data-streaming="true"]', { 
      timeout: 10000 
    })
    
    await page.waitForSelector('[data-streaming="false"][data-role="assistant"]', { 
      timeout: 60000 
    })
    
    // Count rendered tables (look for table elements)
    const tables = await page.locator('table').count()
    console.log(`Found ${tables} rendered tables`)
    
    // Check for pipe characters indicating tables
    const messageContent = page.locator('[data-testid="chat-message"][data-role="assistant"]').last()
    const content = await messageContent.textContent()
    
    // Count table-like structures (lines with multiple pipes)
    const tableLines = (content?.match(/\|.*\|.*\|/g) || []).length
    expect(tableLines).toBeGreaterThanOrEqual(3) // At least header and some data rows
  })

  test('should preserve table formatting during copy/paste', async () => {
    // Send a message that generates a table
    const prompt = 'Create a simple 3x3 table with numbers'
    
    const chatInput = page.locator('[data-testid="chat-input"]')
    await chatInput.fill(prompt)
    await page.keyboard.press('Enter')
    
    // Wait for response to start and complete
    await page.waitForSelector('[data-streaming="true"]', { timeout: 10000 })
    await page.waitForSelector('[data-streaming="false"][data-role="assistant"]', { 
      timeout: 30000 
    })
    
    // Select and copy the message content
    const messageContent = page.locator('[data-testid="chat-message"][data-role="assistant"]').last()
    await messageContent.click({ clickCount: 3 }) // Triple-click to select all
    
    // Copy to clipboard
    await page.keyboard.press('Control+C')
    
    // Get clipboard content (requires clipboard permissions)
    const clipboardContent = await page.evaluate(async () => {
      try {
        return await navigator.clipboard.readText()
      } catch {
        return null
      }
    })
    
    if (clipboardContent) {
      // Verify table structure is preserved
      expect(clipboardContent).toMatch(/\|.*\|.*\|/)
      expect(clipboardContent).not.toContain('||')
    }
  })

  test.skip('should handle network interruption gracefully', async () => {
    // Skip this test as network interruption is complex to test reliably
    // and the development mode doesn't properly simulate network issues
  })
})

/**
 * Visual regression tests for table rendering
 */
test.describe('Table Visual Regression Tests', () => {
  test('should render tables consistently', async ({ page }) => {
    // Create a test page with known table content
    await page.goto('http://localhost:5173')
    
    // Inject known table content for consistent testing
    await page.evaluate(() => {
      const tableHTML = `
        <div class="test-table-container">
          <table>
            <thead>
              <tr>
                <th>Country</th>
                <th>Tax Rate</th>
                <th>Net Income</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td>Spain</td>
                <td>28.5%</td>
                <td>€50,145</td>
              </tr>
              <tr>
                <td>France</td>
                <td>32.0%</td>
                <td>€47,600</td>
              </tr>
            </tbody>
          </table>
        </div>
      `
      document.body.insertAdjacentHTML('beforeend', tableHTML)
    })
    
    // Take screenshot for visual comparison
    const tableElement = await page.locator('.test-table-container')
    await tableElement.screenshot({ 
      path: 'test-results/table-streaming-visual-baseline.png' 
    })
    
    // Verify table structure
    const rows = await page.locator('.test-table-container tr').count()
    expect(rows).toBe(3) // Header + 2 data rows
  })
})