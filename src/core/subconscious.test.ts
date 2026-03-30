import { describe, it, expect } from 'vitest';
import { Subconscious } from './subconscious.js';
import { MemoryKVStore, MemoryVectorStore } from '../storage/memory.js';
import type { LLMAdapter, Message, StatusEvent } from '../types.js';

function createMockLLM(): LLMAdapter {
  return {
    async complete(messages: Message[]) {
      const sys = messages[0]?.content ?? '';

      if (sys.includes('Analyze the incoming message')) {
        return {
          content: JSON.stringify({
            action: 'passthrough',
            reasoning: 'self-contained message',
          }),
        };
      }
      if (sys.includes('Update the running conversation summary')) {
        return { content: 'User greeted the assistant.' };
      }
      if (sys.includes('Compress')) {
        return { content: 'Compressed: earlier discussion.' };
      }
      if (sys.includes('assistant gave a long response')) {
        return { content: 'Summary of the long response.' };
      }
      if (sys.includes('conversation has shifted')) {
        return {
          content: JSON.stringify({
            newTopic: 'deployment',
            preserveFromOld: ['API keys use Vault'],
            searchQueries: ['deployment'],
          }),
        };
      }
      return { content: 'mock response' };
    },
    async embed(text: string) {
      const vec = new Array(8).fill(0) as number[];
      for (let i = 0; i < text.length; i++) {
        vec[i % 8]! += text.charCodeAt(i);
      }
      const mag = Math.sqrt(vec.reduce((s, v) => s + v * v, 0));
      return mag > 0 ? vec.map((v) => v / mag) : vec;
    },
  };
}

function createRecallLLM(): LLMAdapter {
  const base = createMockLLM();
  return {
    ...base,
    async complete(messages: Message[]) {
      const sys = messages[0]?.content ?? '';
      if (sys.includes('Analyze the incoming message')) {
        return {
          content: JSON.stringify({
            action: 'recall',
            reasoning: 'references earlier context',
          }),
        };
      }
      return base.complete(messages);
    },
  };
}

function msg(role: 'user' | 'assistant', content: string): Message {
  return {
    id: `msg-${Date.now()}-${Math.random().toString(36).slice(2, 6)}`,
    role,
    content,
    timestamp: Date.now(),
  };
}

describe('Subconscious', () => {
  it('should initialize', () => {
    const sub = new Subconscious({
      vector: new MemoryVectorStore(),
      kv: new MemoryKVStore(),
      llm: createMockLLM(),
    });
    expect(sub.sessionId).toBeTruthy();
    expect(sub.currentTurn).toBe(0);
  });

  it('should enrich messages with metadata on prepare', async () => {
    const sub = new Subconscious({
      vector: new MemoryVectorStore(),
      kv: new MemoryKVStore(),
      llm: createMockLLM(),
    });

    const result = await sub.prepare(msg('user', 'Hello!'));
    const enriched = result.messages[0]!;

    expect(enriched.meta).toBeDefined();
    expect(enriched.meta.turn).toBe(0);
    expect(enriched.meta.tokens).toBeGreaterThan(0);
    expect(enriched.meta.source).toBe('user');
    expect(enriched.meta.priority).toBe('low'); // "Hello!" matches low patterns
    expect(enriched.meta.pinned).toBe(false);
    expect(enriched.meta.compressed).toBe(false);
  });

  it('should passthrough — fast path', async () => {
    const sub = new Subconscious({
      vector: new MemoryVectorStore(),
      kv: new MemoryKVStore(),
      llm: createMockLLM(),
    });

    const result = await sub.prepare(msg('user', 'Hello!'));

    expect(result.classification).toBe('passthrough');
    expect(result.messages).toHaveLength(1);
    expect(result.messages[0]!.content).toBe('Hello!'); // NEVER modified
    expect(result.totalTokens).toBeGreaterThan(0);
    expect(result.pendingBackground).toBe(true);
  });

  it('should recall and deduplicate', async () => {
    const sub = new Subconscious({
      vector: new MemoryVectorStore(),
      kv: new MemoryKVStore(),
      llm: createRecallLLM(),
    });

    await sub.prepare(msg('user', 'Store API keys in Vault'));
    await sub.flush();

    const result = await sub.prepare(msg('user', 'What about API keys?'));

    expect(result.classification).toBe('recall');
    expect(result.recalled).toBeGreaterThan(0);

    // No duplicate IDs in context
    const ids = result.messages.map((m) => m.id);
    expect(new Set(ids).size).toBe(ids.length);
  });

  it('should assign priority based on content', async () => {
    const sub = new Subconscious({
      vector: new MemoryVectorStore(),
      kv: new MemoryKVStore(),
      llm: createMockLLM(),
    });

    // Critical: contains "decided"
    const r1 = await sub.prepare(msg('user', 'We decided to use PostgreSQL'));
    expect(r1.messages.at(-1)!.meta.priority).toBe('critical');

    // High: contains code block
    const r2 = await sub.prepare(msg('user', 'Here is the fix:\n```\ncode\n```'));
    expect(r2.messages.at(-1)!.meta.priority).toBe('high');

    // Low: greeting
    const r3 = await sub.prepare(msg('user', 'thanks'));
    expect(r3.messages.at(-1)!.meta.priority).toBe('low');
  });

  it('should count tokens on every message', async () => {
    const sub = new Subconscious({
      vector: new MemoryVectorStore(),
      kv: new MemoryKVStore(),
      llm: createMockLLM(),
    });

    const result = await sub.prepare(msg('user', 'This is a test message with several words'));
    const enriched = result.messages[0]!;

    expect(enriched.meta.tokens).toBeGreaterThan(0);
    expect(result.totalTokens).toBe(enriched.meta.tokens);
  });

  it('should respect pinned messages', async () => {
    const sub = new Subconscious({
      vector: new MemoryVectorStore(),
      kv: new MemoryKVStore(),
      llm: createMockLLM(),
      tokenBudget: 50, // very small budget to force trimming
      recentWindow: 1,
    });

    // Add a pinned message
    await sub.prepare({ ...msg('user', 'IMPORTANT: always use Vault'), pinned: true });

    // Add several more messages to exceed budget
    for (let i = 0; i < 5; i++) {
      await sub.prepare(msg('user', `Filler message number ${i} with some extra words to increase token count`));
    }

    const context = sub.getContext();
    const pinned = context.find((m) => m.meta.pinned);
    expect(pinned).toBeDefined();
    expect(pinned!.content).toContain('IMPORTANT');
  });

  it('should ingest responses with metadata', async () => {
    const sub = new Subconscious({
      vector: new MemoryVectorStore(),
      kv: new MemoryKVStore(),
      llm: createMockLLM(),
    });

    await sub.prepare(msg('user', 'Tell me about deployment'));

    const result = await sub.ingest(msg('assistant', 'Use Docker on Cloud Run'));
    expect(result.representation).toBe('full');

    const context = sub.getContext();
    const response = context.find((m) => m.role === 'assistant');
    expect(response).toBeDefined();
    expect(response!.meta.source).toBe('assistant');
    expect(response!.meta.tokens).toBeGreaterThan(0);
  });

  it('should summarize long responses during ingest', async () => {
    const sub = new Subconscious({
      vector: new MemoryVectorStore(),
      kv: new MemoryKVStore(),
      llm: createMockLLM(),
    });

    await sub.prepare(msg('user', 'Explain everything'));

    const longContent = 'This is a long explanation. '.repeat(200);
    const result = await sub.ingest(msg('assistant', longContent));

    expect(result.representation).toBe('summarized');
  });

  it('should emit status events', async () => {
    const statuses: StatusEvent[] = [];

    const sub = new Subconscious({
      vector: new MemoryVectorStore(),
      kv: new MemoryKVStore(),
      llm: createMockLLM(),
      onStatus: (e) => statuses.push(e),
    });

    await sub.prepare(msg('user', 'Hello'));

    expect(statuses[0]!.phase).toBe('classifying');
    expect(statuses.some((s) => s.phase === 'ready')).toBe(true);
  });

  it('should track turns across prepare and ingest', async () => {
    const sub = new Subconscious({
      vector: new MemoryVectorStore(),
      kv: new MemoryKVStore(),
      llm: createMockLLM(),
    });

    expect(sub.currentTurn).toBe(0);

    await sub.prepare(msg('user', 'Hello'));
    expect(sub.currentTurn).toBe(1);

    await sub.ingest(msg('assistant', 'Hi!'));
    expect(sub.currentTurn).toBe(2);

    await sub.prepare(msg('user', 'How are you?'));
    expect(sub.currentTurn).toBe(3);
  });

  it('should never modify the current user message content', async () => {
    const sub = new Subconscious({
      vector: new MemoryVectorStore(),
      kv: new MemoryKVStore(),
      llm: createMockLLM(),
    });

    const original = 'This is my exact message, do not change it!';
    const result = await sub.prepare(msg('user', original));

    // The last message in context should be the user's message, unmodified
    const lastMsg = result.messages[result.messages.length - 1]!;
    expect(lastMsg.content).toBe(original);
    expect(lastMsg.role).toBe('user');
  });
});
