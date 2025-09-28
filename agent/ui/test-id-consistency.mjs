#!/usr/bin/env node

import fetch from 'node-fetch';

const API_URL = 'http://localhost:8000/invocations';

async function testIDConsistency() {
  console.log('🧪 Testing ID consistency between plan and step events...\n');

  try {
    const requestData = {
      messages: [
        {
          role: "user",
          content: "What are the benefits of renewable energy?"
        }
      ],
      max_tokens: 4000,
      temperature: 0.7
    };

    const response = await fetch(API_URL, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(requestData)
    });

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    const reader = response.body;
    const decoder = new TextDecoder();
    let buffer = '';

    console.log('📡 Looking for plan and step events...\n');

    let planSteps = null;
    let stepEvents = [];

    for await (const chunk of reader) {
      buffer += decoder.decode(chunk, { stream: true });
      const lines = buffer.split('\n');
      buffer = lines.pop() || '';

      for (const line of lines) {
        if (line.startsWith('data: ')) {
          const data = line.slice(6);
          if (data === '[DONE]') {
            console.log('\n✅ Stream completed');

            // Final analysis
            console.log('\n📊 Analysis:');
            if (planSteps) {
              console.log('Plan steps found:', planSteps);
            } else {
              console.log('❌ No plan steps found');
            }

            console.log('\nStep events found:', stepEvents.length);
            stepEvents.forEach(event => {
              console.log(`  - ${event.type}: step_id="${event.stepId}"`);
            });

            // Check ID consistency
            if (planSteps && stepEvents.length > 0) {
              console.log('\n🔍 Checking ID consistency:');
              stepEvents.forEach(event => {
                const matchingStep = planSteps.find(s => s.id === event.stepId);
                if (matchingStep) {
                  console.log(`  ✅ "${event.stepId}" matches plan step`);
                } else {
                  console.log(`  ❌ "${event.stepId}" NOT found in plan steps [${planSteps.map(s => s.id).join(', ')}]`);
                }
              });
            }

            return;
          }

          try {
            const event = JSON.parse(data);

            // Check for plan events
            if (event.type === 'intermediate_event' && event.intermediate_event) {
              const eventType = event.intermediate_event.event_type;
              const eventData = event.intermediate_event.data;

              // Look for plan events
              if (eventType === 'plan_created' || eventType === 'plan_updated') {
                if (eventData?.plan?.steps) {
                  planSteps = eventData.plan.steps.map(s => ({
                    id: s.step_id,
                    title: s.title
                  }));
                  console.log(`📋 Found plan with ${planSteps.length} steps:`, planSteps.map(s => s.id));
                }
              }

              // Look for step events
              if (eventType === 'step_activated' || eventType === 'step_completed') {
                const stepId = eventData?.step_id;
                if (stepId) {
                  stepEvents.push({
                    type: eventType,
                    stepId: stepId
                  });
                  console.log(`🎯 Found ${eventType}: step_id="${stepId}"`);
                }
              }
            }
          } catch (err) {
            // Ignore parse errors for non-JSON lines
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
testIDConsistency();