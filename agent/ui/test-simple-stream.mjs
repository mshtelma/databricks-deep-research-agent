#!/usr/bin/env node

// Simple test to check if streaming works at all
const API_URL = 'http://localhost:8000/invocations';

async function testSimpleStream() {
  console.log('🧪 Testing basic streaming...\n');

  try {
    const response = await fetch(API_URL, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        input: [
          {
            role: "user",
            content: "Hi, just say hello back"
          }
        ]
      })
    });

    if (!response.ok) {
      console.error(`❌ HTTP error! status: ${response.status}`);
      const text = await response.text();
      console.error('Response:', text);
      return;
    }

    console.log('✅ Got response, status:', response.status);
    console.log('Headers:', response.headers);

    const reader = response.body;
    const decoder = new TextDecoder();
    let buffer = '';
    let eventCount = 0;

    console.log('\n📡 Reading stream...\n');

    for await (const chunk of reader) {
      const text = decoder.decode(chunk, { stream: true });
      buffer += text;

      // Process complete lines
      const lines = buffer.split('\n');
      buffer = lines.pop() || '';

      for (const line of lines) {
        if (line.trim()) {
          console.log(`Line ${++eventCount}: ${line.substring(0, 100)}${line.length > 100 ? '...' : ''}`);

          if (line.startsWith('data: ')) {
            const data = line.slice(6);
            if (data === '[DONE]') {
              console.log('\n✅ Stream completed');
              return;
            }

            try {
              const event = JSON.parse(data);
              if (event.type === 'intermediate_event') {
                console.log('  🎯 Found intermediate event:', event.intermediate_event?.event_type);
              }
            } catch (e) {
              // Ignore parse errors
            }
          }
        }
      }

      // Stop after 100 events to avoid infinite loop
      if (eventCount > 100) {
        console.log('\n⚠️ Stopping after 100 events');
        break;
      }
    }

    console.log(`\n📊 Total events received: ${eventCount}`);

  } catch (error) {
    console.error('❌ Error:', error.message);
    console.error(error.stack);
  }
}

testSimpleStream();