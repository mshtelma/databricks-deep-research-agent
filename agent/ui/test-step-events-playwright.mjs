#!/usr/bin/env node

import { chromium } from 'playwright';

async function testStepEvents() {
  console.log('🧪 Testing step events with Playwright...\n');

  const browser = await chromium.launch({
    headless: false,
    devtools: true
  });
  const context = await browser.newContext();
  const page = await context.newPage();

  // Collect console messages
  const consoleLogs = [];
  page.on('console', msg => {
    const text = msg.text();
    consoleLogs.push(text);

    // Look for step event logs
    if (text.includes('Step event received')) {
      console.log('✅ STEP EVENT DETECTED:', text);
    }
    if (text.includes('Step status updated')) {
      console.log('✅ STATUS UPDATE:', text);
    }
    if (text.includes('Failed to update step')) {
      console.log('❌ UPDATE FAILED:', text);
    }
  });

  // Navigate to the app
  await page.goto('http://localhost:8000');
  await page.waitForTimeout(2000);

  // Enter a query
  const textarea = await page.locator('textarea[placeholder*="Ask me anything"]');
  await textarea.fill('What are the main types of machine learning?');

  // Submit the query
  await page.keyboard.press('Enter');

  console.log('📤 Query submitted, monitoring for events...\n');

  // Wait and collect events
  let planCreated = false;
  let stepEventsReceived = 0;
  const startTime = Date.now();

  while (Date.now() - startTime < 60000) { // Wait up to 60 seconds
    await page.waitForTimeout(1000);

    // Check for plan creation
    if (!planCreated) {
      const planLogs = consoleLogs.filter(log => log.includes('Plan step created'));
      if (planLogs.length > 0) {
        planCreated = true;
        console.log(`📋 Plan created with ${planLogs.length} steps`);
      }
    }

    // Check for step events
    const stepEventLogs = consoleLogs.filter(log => log.includes('Step event received'));
    if (stepEventLogs.length > stepEventsReceived) {
      stepEventsReceived = stepEventLogs.length;
      console.log(`🎯 Step events received: ${stepEventsReceived}`);
    }

    // Check Research Progress panel
    const progressItems = await page.locator('.research-progress-item').count();
    if (progressItems > 0) {
      console.log(`📊 Research Progress items visible: ${progressItems}`);

      // Check for completed items
      const completedItems = await page.locator('.research-progress-item.completed').count();
      const activeItems = await page.locator('.research-progress-item.active').count();

      if (completedItems > 0) {
        console.log(`✅ Completed steps: ${completedItems}`);
      }
      if (activeItems > 0) {
        console.log(`🔄 Active steps: ${activeItems}`);
      }
    }

    // Break if we see completion
    if (consoleLogs.some(log => log.includes('Stream completed'))) {
      console.log('✅ Stream completed');
      break;
    }
  }

  // Final analysis
  console.log('\n' + '='.repeat(60));
  console.log('📊 FINAL ANALYSIS:');
  console.log('='.repeat(60));

  console.log('\n📋 Console Log Summary:');
  const planStepLogs = consoleLogs.filter(log => log.includes('Plan step created'));
  const stepEventLogs = consoleLogs.filter(log => log.includes('Step event received'));
  const updateLogs = consoleLogs.filter(log => log.includes('Step status updated'));
  const failureLogs = consoleLogs.filter(log => log.includes('Failed to update step'));

  console.log(`  Plan steps created: ${planStepLogs.length}`);
  console.log(`  Step events received: ${stepEventLogs.length}`);
  console.log(`  Successful updates: ${updateLogs.length}`);
  console.log(`  Failed updates: ${failureLogs.length}`);

  if (stepEventLogs.length === 0) {
    console.log('\n❌ NO STEP EVENTS RECEIVED!');
    console.log('This indicates the events are not being streamed from the server.');
  } else if (failureLogs.length > 0) {
    console.log('\n⚠️ Some step updates failed. Check the logs for details.');
  } else if (updateLogs.length > 0) {
    console.log('\n✅ Step events are being received and processed!');
  }

  // Show some actual console logs
  console.log('\n📝 Sample Console Logs:');
  consoleLogs.slice(-10).forEach(log => {
    if (log.length < 200) {
      console.log(`  ${log}`);
    }
  });

  await page.waitForTimeout(5000); // Keep browser open for inspection
  await browser.close();
}

testStepEvents().catch(console.error);