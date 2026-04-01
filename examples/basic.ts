/**
 * Basic example — The Subconscious in action.
 *
 * Shows a multi-turn conversation where the Subconscious manages context:
 * - Messages are enriched with metadata (tokens, priority, turn)
 * - Context is curated (passthrough, recall, or reshape)
 * - Status events show what's happening
 * - The developer can inspect every decision
 *
 * Run:
 *   OPENAI_API_KEY=sk-... npx tsx examples/basic.ts
 *
 * Or with Anthropic:
 *   ANTHROPIC_API_KEY=sk-ant-... OPENAI_API_KEY=sk-... npx tsx examples/basic.ts --anthropic
 */

import { Subconscious, MemoryKVStore, MemoryVectorStore } from '../src/index.js';
import { OpenAIAdapter } from '../src/llm/openai.js';
import type { LLMAdapter, Message } from '../src/types.js';

// ---------------------------------------------------------------------------
// Setup
// ---------------------------------------------------------------------------

async function createLLM(): Promise<LLMAdapter> {
  if (process.argv.includes('--anthropic')) {
    const { AnthropicAdapter } = await import('../src/llm/anthropic.js');
    const openai = new OpenAIAdapter(); // for embeddings
    return new AnthropicAdapter({
      embedder: (text) => openai.embed(text),
    });
  }
  return new OpenAIAdapter();
}

function msg(role: 'user' | 'assistant', content: string): Message {
  return {
    id: `msg-${Date.now()}-${Math.random().toString(36).slice(2, 6)}`,
    role,
    content,
    timestamp: Date.now(),
  };
}

// ---------------------------------------------------------------------------
// Simulate a main agent (just echoes back via the same LLM)
// ---------------------------------------------------------------------------

async function mainAgentRespond(llm: LLMAdapter, messages: Message[]): Promise<Message> {
  const response = await llm.complete([
    {
      id: 'sys',
      role: 'system',
      content: 'You are a helpful assistant. Keep responses concise (1-2 sentences).',
      timestamp: Date.now(),
    },
    ...messages,
  ]);

  return msg('assistant', response.content);
}

// ---------------------------------------------------------------------------
// Run the conversation
// ---------------------------------------------------------------------------

async function main(): Promise<void> {
  const llm = await createLLM();

  const sub = new Subconscious({
    vector: new MemoryVectorStore(),
    kv: new MemoryKVStore(),
    llm,
    tokenBudget: 4000,
    recentWindow: 6,
    onStatus: (event) => {
      console.log(`  [${event.phase}] ${event.message}`);
    },
  });

  console.log('=== The Subconscious — Basic Example ===\n');

  // Simulate a multi-turn conversation
  const userMessages = [
    'Hi! I want to build a REST API for a todo app.',
    'What database should I use? I need something simple.',
    'Ok let\'s go with SQLite. What about authentication?',
    'We decided to use JWT tokens for auth. Can you summarize what we\'ve planned so far?',
    // Topic shift — the Subconscious should detect this
    'Actually, let\'s switch topics. Tell me about WebSocket connections.',
    // Reference to earlier context — should trigger recall
    'Wait, going back to the API — what database did we decide on?',
  ];

  for (const userText of userMessages) {
    console.log(`\nUser: ${userText}`);

    // --- PREPARE: Subconscious curates context ---
    const prepared = await sub.prepare(msg('user', userText));

    console.log(`  Classification: ${prepared.classification}`);
    console.log(`  Context: ${prepared.messages.length} messages, ~${prepared.totalTokens} tokens`);
    if (prepared.recalled > 0) {
      console.log(`  Recalled: ${prepared.recalled} messages from history`);
    }
    for (const action of prepared.actions) {
      console.log(`  Action: [${action.type}] ${action.detail}`);
    }

    // --- MAIN AGENT: responds using curated context ---
    const response = await mainAgentRespond(llm, prepared.messages);
    console.log(`\nAssistant: ${response.content}`);

    // --- INGEST: Subconscious stores the response ---
    const ingestResult = await sub.ingest(response);
    console.log(`  Stored as: ${ingestResult.representation}`);

    // Let background tasks finish
    await sub.flush();

    // Small delay to avoid rate limits
    await new Promise((r) => setTimeout(r, 500));
  }

  // --- Show final state ---
  console.log('\n=== Final State ===');
  console.log(`Session: ${sub.sessionId}`);
  console.log(`Turns: ${sub.currentTurn}`);
  console.log(`Context size: ${sub.getContext().length} messages`);
  console.log(`Summary: ${await sub.getSummary()}`);

  // --- Demo explicit recall ---
  console.log('\n=== Explicit Recall ===');
  const recalled = await sub.recall('database choice');
  console.log(`Recalled ${recalled.length} messages about "database choice":`);
  for (const m of recalled.slice(0, 3)) {
    console.log(`  [${m.meta.source}/${m.role}, turn ${m.meta.turn}] ${m.content.slice(0, 80)}...`);
  }
}

main().catch(console.error);
