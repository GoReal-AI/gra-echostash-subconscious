/**
 * Quick test — Subconscious with Google AI (Gemini Flash).
 *
 * Run: GOOGLE_AI_API_KEY=... npx tsx examples/google-test.ts
 */

import { Subconscious, MemoryKVStore, MemoryVectorStore } from '../src/index.js';
import { GoogleAdapter } from '../src/llm/google.js';
import type { Message } from '../src/types.js';

function msg(role: 'user' | 'assistant', content: string): Message {
  return {
    id: `msg-${Date.now()}-${Math.random().toString(36).slice(2, 6)}`,
    role,
    content,
    timestamp: Date.now(),
  };
}

async function main(): Promise<void> {
  const llm = new GoogleAdapter();

  console.log('Testing Google AI adapter...\n');

  // Quick sanity: can we complete and embed?
  console.log('1. Testing completion...');
  const completion = await llm.complete([
    { id: '1', role: 'user', content: 'Say "hello" and nothing else.', timestamp: Date.now() },
  ]);
  console.log(`   Response: "${completion.content}"`);
  console.log(`   Usage: ${completion.usage?.inputTokens ?? '?'} in, ${completion.usage?.outputTokens ?? '?'} out\n`);

  console.log('2. Testing embedding...');
  const embedding = await llm.embed('Hello world');
  console.log(`   Embedding dimension: ${embedding.length}\n`);

  // Now test full Subconscious flow
  console.log('3. Testing Subconscious with Google AI...\n');

  const sub = new Subconscious({
    vector: new MemoryVectorStore(),
    kv: new MemoryKVStore(),
    llm,
    tokenBudget: 4000,
    onStatus: (e) => console.log(`   [${e.phase}] ${e.message}`),
  });

  const turns = [
    'Hi! I want to build a REST API for managing books.',
    'Let\'s use PostgreSQL for the database.',
    'We decided to use JWT for authentication.',
    'What database did we decide on?',
  ];

  for (const text of turns) {
    console.log(`User: ${text}`);
    const result = await sub.prepare(msg('user', text));
    console.log(`   Classification: ${result.classification}`);
    console.log(`   Context: ${result.messages.length} msgs, ~${result.totalTokens} tokens`);
    for (const a of result.actions) {
      console.log(`   Action: [${a.type}] ${a.detail}`);
    }

    // Simulate assistant response
    await sub.ingest(msg('assistant', `(mock response to: ${text.slice(0, 30)}...)`));
    await sub.flush();
    console.log('');
  }

  // Test recall
  console.log('4. Testing explicit recall...');
  const recalled = await sub.recall('database choice');
  console.log(`   Recalled ${recalled.length} messages about "database choice":`);
  for (const m of recalled.slice(0, 3)) {
    console.log(`   [turn ${m.meta.turn}] ${m.content.slice(0, 60)}...`);
  }

  console.log('\nDone!');
}

main().catch(console.error);
