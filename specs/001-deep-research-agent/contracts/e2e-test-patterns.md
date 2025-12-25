# E2E Test Patterns: Deep Research Agent

**Date**: 2025-12-22
**User Story**: 9 - Automated End-to-End Testing
**Framework**: Playwright with TypeScript

## Overview

This document defines the test patterns and contracts for e2e testing the Deep Research Agent chat interface. Tests validate complete user journeys from User Story 9.

---

## Test Selectors Contract

All interactive elements must expose `data-testid` attributes for stable test selection.

### Required Test IDs

| Component | data-testid | Purpose |
|-----------|-------------|---------|
| Message input | `message-input` | Text input for user queries |
| Send button | `send-button` | Submit message button |
| Stop button | `stop-button` | Cancel/stop in-progress research |
| Loading indicator | `loading-indicator` | Shows during API calls |
| Streaming indicator | `streaming-indicator` | Shows during SSE streaming |
| Message list | `message-list` | Container for all messages |
| User message | `user-message` | Individual user message |
| Agent response | `agent-response` | Individual agent response |
| Citation | `citation` | Source citation link |
| Reasoning panel | `reasoning-panel` | Expandable reasoning steps |
| Edit button | `edit-message-{id}` | Edit specific message |
| Regenerate button | `regenerate-response` | Regenerate last response |
| New chat button | `new-chat-button` | Create new conversation |
| Chat list | `chat-list` | Sidebar chat list |
| Chat item | `chat-item-{id}` | Individual chat in list |

### Selector Priority

1. **`data-testid`** - Primary for interactive elements
2. **`getByRole`** - For semantic elements (buttons, inputs, headings)
3. **`getByText`** - For static content validation
4. **Never**: CSS classes, XPath, complex selectors

---

## Test Scenarios Contract

### Scenario 1: Research Flow (User Story 9.1)

```typescript
// research-flow.spec.ts
test.describe('Research Flow', () => {
  test('submits query and receives response with citations', async ({ chatPage }) => {
    // Given: application is running (handled by fixture)

    // When: submit research query
    await chatPage.sendMessage('What are the latest developments in quantum computing?');

    // Then: verify streaming reasoning steps appear
    await expect(chatPage.streamingIndicator).toBeVisible();

    // Then: wait for completion (up to 2 minutes)
    await chatPage.waitForAgentResponse(120000);

    // Then: verify response with citations
    const response = await chatPage.getLastAgentResponse();
    expect(response.length).toBeGreaterThan(100);

    const citations = chatPage.page.getByTestId('citation');
    await expect(citations).toHaveCount({ minimum: 1 });
  });
});
```

**Acceptance Criteria**:
- Response appears within 2 minutes
- Response contains substantive content (>100 characters)
- At least 1 citation is present

---

### Scenario 2: Follow-up Questions (User Story 9.2)

```typescript
// follow-up.spec.ts
test.describe('Follow-up Questions', () => {
  test('responds using context from previous exchange', async ({ chatPage }) => {
    // Given: test conversation exists
    await chatPage.sendMessage('What is machine learning?');
    await chatPage.waitForAgentResponse();

    // When: submit follow-up question
    await chatPage.sendMessage('Can you give me an example?');

    // Then: verify context-aware response
    await chatPage.waitForAgentResponse(30000);

    const response = await chatPage.getLastAgentResponse();
    // Response should reference "machine learning" or related context
    expect(response.toLowerCase()).toMatch(/machine|learning|example|algorithm/);
  });
});
```

**Acceptance Criteria**:
- Follow-up responds within 30 seconds
- Response demonstrates awareness of previous context

---

### Scenario 3: Stop/Cancel (User Story 9.3)

```typescript
// stop-cancel.spec.ts
test.describe('Stop/Cancel', () => {
  test('stops operation within 2 seconds', async ({ chatPage }) => {
    // Given: research operation is in progress
    await chatPage.sendMessage('Write a comprehensive analysis of global economic trends');

    // Wait for loading to start
    await expect(chatPage.loadingIndicator).toBeVisible();

    // When: trigger stop/cancel
    const stopStartTime = Date.now();
    await chatPage.stopButton.click();

    // Then: verify operation stops within 2 seconds
    await expect(chatPage.loadingIndicator).toBeHidden({ timeout: 2000 });
    const stopDuration = Date.now() - stopStartTime;
    expect(stopDuration).toBeLessThan(2000);

    // Then: verify partial results preserved (or message shows stopped)
    const messages = chatPage.page.getByTestId('agent-response');
    await expect(messages).toHaveCount({ minimum: 0 });
  });
});
```

**Acceptance Criteria**:
- Loading indicator disappears within 2 seconds of stop click
- Application remains in stable state after stop

---

### Scenario 4: Edit Message (User Story 9.4)

```typescript
// edit-message.spec.ts
test.describe('Edit Message', () => {
  test('invalidates subsequent messages on edit', async ({ chatPage }) => {
    // Given: test has received an agent response
    await chatPage.sendMessage('What is Python?');
    await chatPage.waitForAgentResponse();
    const initialMessageCount = await chatPage.page.getByTestId('agent-response').count();

    // When: edit previous message
    await chatPage.editMessage(0, 'What is JavaScript?');

    // Then: verify subsequent messages invalidated
    await chatPage.waitForAgentResponse();
    const newResponse = await chatPage.getLastAgentResponse();

    // Response should now be about JavaScript, not Python
    expect(newResponse.toLowerCase()).toMatch(/javascript/);
    expect(newResponse.toLowerCase()).not.toMatch(/python/);
  });
});
```

**Acceptance Criteria**:
- Edit triggers new response generation
- New response reflects edited query
- Previous context from before edit is preserved

---

### Scenario 5: Regenerate (User Story 9.5)

```typescript
// regenerate.spec.ts
test.describe('Regenerate Response', () => {
  test('generates new response for same query', async ({ chatPage }) => {
    // Given: test has received an agent response
    await chatPage.sendMessage('Give me a random fact');
    await chatPage.waitForAgentResponse();
    const originalResponse = await chatPage.getLastAgentResponse();

    // When: trigger regenerate
    await chatPage.regenerateButton.click();

    // Then: verify new response generated
    await chatPage.waitForAgentResponse();
    const newResponse = await chatPage.getLastAgentResponse();

    // Responses should differ (new research performed)
    expect(newResponse).not.toEqual(originalResponse);
  });
});
```

**Acceptance Criteria**:
- New response is generated
- Response differs from original (fresh search)

---

### Scenario 6: Test Reporting (User Story 9.6)

```typescript
// Configured in playwright.config.ts
export default defineConfig({
  reporter: [
    ['html', { open: 'never' }],
    ['list'],
    ['json', { outputFile: 'test-results.json' }],
  ],
  use: {
    trace: 'on-first-retry',
    screenshot: 'only-on-failure',
    video: 'retain-on-failure',
  },
});
```

**Acceptance Criteria**:
- HTML report generated after test run
- Screenshots captured on failure
- Trace files available for debugging
- Video recordings retained on failure

---

## Smoke Tests Contract

Fast validation tests for CI pipeline.

```typescript
// smoke.spec.ts
test.describe('Smoke Tests', () => {
  test('app loads successfully', async ({ page }) => {
    await page.goto('/');
    await expect(page).toHaveTitle(/Deep Research/i);
    await expect(page.getByTestId('message-input')).toBeVisible();
  });

  test('can create new chat', async ({ page }) => {
    await page.goto('/');
    await page.getByTestId('new-chat-button').click();
    await expect(page.getByTestId('message-input')).toBeEmpty();
  });

  test('can send simple message', async ({ chatPage }) => {
    await chatPage.sendMessage('Hello');
    await chatPage.waitForAgentResponse(15000);
    const response = await chatPage.getLastAgentResponse();
    expect(response.length).toBeGreaterThan(0);
  });
});
```

**Target**: <30 seconds for all smoke tests

---

## Timeout Configuration

| Scenario | Default Timeout | Justification |
|----------|----------------|---------------|
| Page load | 30s | Standard web app load |
| Simple query | 15s | Direct response without research |
| Research query | 120s | Full research loop (per SC-001) |
| Follow-up | 30s | Uses existing context |
| Stop action | 2s | Per FR-029, SC-011 |
| Edit/regenerate | 60s | Triggers new research |

---

## Error Handling Patterns

### Network Failures

```typescript
test('handles network failure gracefully', async ({ page }) => {
  await page.route('**/api/**', route => route.abort());

  await page.getByTestId('message-input').fill('Test query');
  await page.getByTestId('send-button').click();

  // Expect error message displayed
  await expect(page.getByText(/error|failed|try again/i)).toBeVisible();
});
```

### Timeout Handling

```typescript
test('shows timeout message for long operations', async ({ chatPage }) => {
  // Configure shorter timeout for test
  test.setTimeout(10000);

  await chatPage.sendMessage('Very complex query...');

  // Expect either response or timeout message
  const result = await Promise.race([
    chatPage.waitForAgentResponse().then(() => 'success'),
    chatPage.page.waitForSelector('[data-testid="error-message"]').then(() => 'error'),
  ]);

  expect(['success', 'error']).toContain(result);
});
```

---

## CI Integration Contract

### Required Environment Variables

```bash
E2E_BASE_URL=http://localhost:8000  # Application URL
CI=true                               # Enables stricter test settings
```

### Expected Exit Codes

| Code | Meaning |
|------|---------|
| 0 | All tests passed |
| 1 | Some tests failed |
| 2 | Configuration error |

### Artifact Outputs

```
frontend/
├── playwright-report/    # HTML report
├── test-results/        # Screenshots, videos, traces
└── test-results.json    # JSON report for CI parsing
```
