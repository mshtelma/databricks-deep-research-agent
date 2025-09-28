#!/usr/bin/env node

import fetch from 'node-fetch';

const API_URL = 'http://localhost:8000/invocations';

async function testStepTracking() {
  console.log('🧪 Testing step activation/completion tracking...\n');

  try {
    // Simple request that should trigger plan creation and execution
    const requestData = {
      input: [
        {
          role: "user",
          content: "What is machine learning?"
        }
      ]
    };

    const response = await fetch(API_URL, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(requestData)
    });

    if (!response.ok) {
      const errorText = await response.text();
      console.error(`HTTP error! status: ${response.status}`);
      console.error('Error response:', errorText);
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    const reader = response.body;
    const decoder = new TextDecoder();
    let buffer = '';

    console.log('📡 Monitoring step events...\n');

    let planSteps = [];
    let stepEvents = [];
    let eventCount = 0;

    for await (const chunk of reader) {
      buffer += decoder.decode(chunk, { stream: true });
      const lines = buffer.split('\n');
      buffer = lines.pop() || '';

      for (const line of lines) {
        if (line.startsWith('data: ')) {
          const data = line.slice(6);
          if (data === '[DONE]') {
            console.log('\n✅ Stream completed\n');

            // Final analysis
            console.log('=' .repeat(60));
            console.log('📊 FINAL ANALYSIS:');
            console.log('=' .repeat(60));

            console.log('\n📋 Plan Steps Found:');
            if (planSteps.length > 0) {
              planSteps.forEach((step, idx) => {
                console.log(`  ${idx + 1}. ID: "${step.id}" - ${step.title}`);
              });
            } else {
              console.log('  ❌ No plan steps found');
            }

            console.log('\n🎯 Step Events Captured:');
            if (stepEvents.length > 0) {
              stepEvents.forEach((event, idx) => {
                console.log(`  ${idx + 1}. ${event.type}: step_id="${event.stepId}" (${event.stepName || 'no name'})`);
              });
            } else {
              console.log('  ❌ No step events captured');
            }

            // Check ID matching
            if (planSteps.length > 0 && stepEvents.length > 0) {
              console.log('\n🔍 ID Consistency Check:');
              const planStepIds = planSteps.map(s => s.id);

              stepEvents.forEach(event => {
                if (planStepIds.includes(event.stepId)) {
                  console.log(`  ✅ Event step_id "${event.stepId}" matches plan`);
                } else {
                  console.log(`  ❌ Event step_id "${event.stepId}" NOT in plan [${planStepIds.join(', ')}]`);
                }
              });

              // Check for missing events
              console.log('\n📝 Coverage Check:');
              planSteps.forEach(step => {
                const activated = stepEvents.find(e => e.type === 'step_activated' && e.stepId === step.id);
                const completed = stepEvents.find(e => e.type === 'step_completed' && e.stepId === step.id);

                if (activated && completed) {
                  console.log(`  ✅ Step "${step.id}" - both activated & completed`);
                } else if (activated) {
                  console.log(`  ⚠️ Step "${step.id}" - activated but not completed`);
                } else if (completed) {
                  console.log(`  ⚠️ Step "${step.id}" - completed but not activated`);
                } else {
                  console.log(`  ❌ Step "${step.id}" - no events`);
                }
              });
            }

            return;
          }

          try {
            const event = JSON.parse(data);
            eventCount++;

            // Look for different event structures
            if (event.type === 'intermediate_event' && event.intermediate_event) {
              const ie = event.intermediate_event;
              const eventType = ie.event_type;

              // Plan events
              if (eventType === 'plan_created' || eventType === 'plan_updated') {
                const plan = ie.data?.plan;
                if (plan?.steps) {
                  planSteps = plan.steps.map(s => ({
                    id: s.step_id,
                    title: s.title || s.description || 'Untitled'
                  }));
                  console.log(`\n📋 PLAN ${eventType === 'plan_created' ? 'CREATED' : 'UPDATED'} with ${planSteps.length} steps:`);
                  planSteps.forEach((s, i) => console.log(`   ${i+1}. "${s.id}" - ${s.title}`));
                }
              }

              // Step events
              if (eventType === 'step_activated' || eventType === 'step_completed') {
                const stepId = ie.data?.step_id;
                const stepName = ie.data?.step_name || ie.data?.title;

                if (stepId) {
                  stepEvents.push({
                    type: eventType,
                    stepId: stepId,
                    stepName: stepName
                  });
                  console.log(`\n🎯 STEP EVENT: ${eventType}`);
                  console.log(`   Step ID: "${stepId}"`);
                  console.log(`   Step Name: ${stepName || 'N/A'}`);
                  console.log(`   Data keys: ${Object.keys(ie.data || {}).join(', ')}`);
                }
              }
            }
          } catch (err) {
            // Ignore parse errors
          }
        }
      }
    }
  } catch (error) {
    console.error('❌ Error:', error.message);
    process.exit(1);
  }
}

// Run the test
console.log('Starting test in 2 seconds...\n');
setTimeout(testStepTracking, 2000);