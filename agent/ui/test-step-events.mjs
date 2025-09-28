#!/usr/bin/env node

import fetch from 'node-fetch';

const API_URL = 'http://localhost:8000/invocations';

async function testStepEvents() {
  console.log('🧪 Testing step activation/completion events...\n');

  try {
    // Request format that matches MLflow ResponsesAgent
    const requestData = {
      input: [
        {
          role: "user",
          content: "What is machine learning and its main types?"
        }
      ],
      params: {
        max_tokens: 4000,
        temperature: 0.7
      }
    };

    console.log('📤 Sending request:', JSON.stringify(requestData, null, 2));
    console.log('');

    const response = await fetch(API_URL, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(requestData)
    });

    if (!response.ok) {
      const errorText = await response.text();
      console.error(`❌ HTTP error! status: ${response.status}`);
      console.error('Error response:', errorText);
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    const reader = response.body;
    const decoder = new TextDecoder();
    let buffer = '';

    console.log('📡 Monitoring step events...\n');
    console.log('=' .repeat(60));

    let planSteps = [];
    let stepActivations = [];
    let stepCompletions = [];
    let eventCount = 0;

    for await (const chunk of reader) {
      buffer += decoder.decode(chunk, { stream: true });
      const lines = buffer.split('\n');
      buffer = lines.pop() || '';

      for (const line of lines) {
        if (line.startsWith('data: ')) {
          const data = line.slice(6);
          if (data === '[DONE]') {
            console.log('\n' + '=' .repeat(60));
            console.log('✅ Stream completed\n');

            // Final analysis
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

            console.log('\n🎯 Step Activations:');
            if (stepActivations.length > 0) {
              stepActivations.forEach((event, idx) => {
                console.log(`  ${idx + 1}. step_id="${event.stepId}" (${event.stepName || 'no name'})`);
              });
            } else {
              console.log('  ❌ No step activations captured');
            }

            console.log('\n✅ Step Completions:');
            if (stepCompletions.length > 0) {
              stepCompletions.forEach((event, idx) => {
                console.log(`  ${idx + 1}. step_id="${event.stepId}" (${event.stepName || 'no name'})`);
              });
            } else {
              console.log('  ❌ No step completions captured');
            }

            // Check ID matching
            if (planSteps.length > 0 && (stepActivations.length > 0 || stepCompletions.length > 0)) {
              console.log('\n🔍 ID Consistency Check:');
              const planStepIds = planSteps.map(s => s.id);

              const allStepEvents = [...stepActivations, ...stepCompletions];
              const uniqueEventStepIds = [...new Set(allStepEvents.map(e => e.stepId))];

              uniqueEventStepIds.forEach(stepId => {
                if (planStepIds.includes(stepId)) {
                  console.log(`  ✅ Event step_id "${stepId}" matches plan`);
                } else {
                  console.log(`  ❌ Event step_id "${stepId}" NOT in plan [${planStepIds.join(', ')}]`);
                }
              });

              // Check for missing events
              console.log('\n📝 Coverage Check:');
              planSteps.forEach(step => {
                const activated = stepActivations.find(e => e.stepId === step.id);
                const completed = stepCompletions.find(e => e.stepId === step.id);

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

            console.log('\n' + '=' .repeat(60));
            console.log('📈 Summary:');
            console.log(`  Total events processed: ${eventCount}`);
            console.log(`  Plan steps: ${planSteps.length}`);
            console.log(`  Step activations: ${stepActivations.length}`);
            console.log(`  Step completions: ${stepCompletions.length}`);

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
                    id: s.step_id || s.id,
                    title: s.title || s.description || 'Untitled'
                  }));
                  console.log(`\n📋 PLAN ${eventType === 'plan_created' ? 'CREATED' : 'UPDATED'} with ${planSteps.length} steps:`);
                  planSteps.forEach((s, i) => console.log(`   ${i+1}. "${s.id}" - ${s.title}`));
                }
              }

              // Step activation events
              if (eventType === 'step_activated') {
                const stepId = ie.data?.step_id || ie.data?.id;
                const stepName = ie.data?.step_name || ie.data?.title || ie.data?.description;

                if (stepId) {
                  stepActivations.push({
                    stepId: stepId,
                    stepName: stepName
                  });
                  console.log(`\n🎯 STEP ACTIVATED:`);
                  console.log(`   Step ID: "${stepId}"`);
                  console.log(`   Step Name: ${stepName || 'N/A'}`);
                  console.log(`   Data keys: ${Object.keys(ie.data || {}).join(', ')}`);
                }
              }

              // Step completion events
              if (eventType === 'step_completed') {
                const stepId = ie.data?.step_id || ie.data?.id;
                const stepName = ie.data?.step_name || ie.data?.title || ie.data?.description;
                const result = ie.data?.result;

                if (stepId) {
                  stepCompletions.push({
                    stepId: stepId,
                    stepName: stepName,
                    result: result
                  });
                  console.log(`\n✅ STEP COMPLETED:`);
                  console.log(`   Step ID: "${stepId}"`);
                  console.log(`   Step Name: ${stepName || 'N/A'}`);
                  if (result) {
                    console.log(`   Result: ${typeof result === 'string' ? result.substring(0, 100) + '...' : JSON.stringify(result).substring(0, 100) + '...'}`);
                  }
                  console.log(`   Data keys: ${Object.keys(ie.data || {}).join(', ')}`);
                }
              }

              // Debug: Show any other intermediate events
              if (!['plan_created', 'plan_updated', 'step_activated', 'step_completed'].includes(eventType)) {
                if (eventType !== 'progress' && eventType !== 'status_update') {
                  console.log(`\n🔔 Other event: ${eventType}`);
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
    console.error('\n❌ Error:', error.message);
    if (error.message.includes('ECONNREFUSED')) {
      console.error('\n💡 Make sure the agent server is running:');
      console.error('   cd agent && ./start_agent_with_ui.sh');
    }
    process.exit(1);
  }
}

// Run the test
console.log('Starting test in 2 seconds...\n');
setTimeout(testStepEvents, 2000);
