import { describe, it, expect } from 'vitest';
import {
  countTokens,
  totalTokens,
  assignPriority,
  computeRelevancy,
  enrich,
  enforceTokenBudget,
  deduplicate,
  orderChronologically,
  removeExpired,
  inferSource,
} from './scripted.js';
import type { EnrichedMessage, Message } from '../types.js';

function msg(role: 'user' | 'assistant', content: string, id?: string): Message {
  return { id: id ?? `msg-${Math.random()}`, role, content, timestamp: Date.now() };
}

function enriched(role: 'user' | 'assistant', content: string, turn: number, overrides?: Partial<EnrichedMessage['meta']>): EnrichedMessage {
  const e = enrich(msg(role, content), turn);
  if (overrides) Object.assign(e.meta, overrides);
  return e;
}

describe('countTokens', () => {
  it('should count tokens for normal text', () => {
    const tokens = countTokens('This is a test message');
    expect(tokens).toBeGreaterThan(0);
    expect(tokens).toBeLessThan(20); // 5 words * 1.3 ≈ 7
  });

  it('should return 0 for empty string', () => {
    expect(countTokens('')).toBe(0);
  });

  it('should count more tokens for longer text', () => {
    const short = countTokens('Hello');
    const long = countTokens('Hello world this is a longer sentence with more words');
    expect(long).toBeGreaterThan(short);
  });
});

describe('assignPriority', () => {
  it('should assign critical for decisions', () => {
    expect(assignPriority('We decided to use PostgreSQL')).toBe('critical');
    expect(assignPriority("Let's go with Redis")).toBe('critical');
    expect(assignPriority('Confirmed the deployment plan')).toBe('critical');
  });

  it('should assign high for code and action items', () => {
    expect(assignPriority('Here is the code:\n```\nconst x = 1;\n```')).toBe('high');
    expect(assignPriority('TODO: fix the bug')).toBe('high');
    expect(assignPriority('We need to update the config')).toBe('high');
  });

  it('should assign low for greetings', () => {
    expect(assignPriority('hi')).toBe('low');
    expect(assignPriority('thanks')).toBe('low');
    expect(assignPriority('ok')).toBe('low');
  });

  it('should assign normal for regular messages', () => {
    expect(assignPriority('What does this function do?')).toBe('normal');
    expect(assignPriority('Can you explain the architecture?')).toBe('normal');
  });
});

describe('inferSource', () => {
  it('should infer from role', () => {
    expect(inferSource({ id: '1', role: 'user', content: '', timestamp: 0 })).toBe('user');
    expect(inferSource({ id: '1', role: 'assistant', content: '', timestamp: 0 })).toBe('assistant');
    expect(inferSource({ id: '1', role: 'tool', content: '', timestamp: 0 })).toBe('tool');
  });

  it('should use explicit source over role', () => {
    expect(inferSource({ id: '1', role: 'system', content: '', timestamp: 0, source: 'rag' })).toBe('rag');
    expect(inferSource({ id: '1', role: 'user', content: '', timestamp: 0, source: 'skill' })).toBe('skill');
  });
});

describe('enrich', () => {
  it('should add full metadata', () => {
    const e = enrich(msg('user', 'Hello world'), 5);

    expect(e.meta.turn).toBe(5);
    expect(e.meta.tokens).toBeGreaterThan(0);
    expect(e.meta.source).toBe('user');
    expect(e.meta.priority).toBe('low'); // "Hello" matches low
    expect(e.meta.pinned).toBe(false);
    expect(e.meta.compressed).toBe(false);
    expect(e.meta.recalled).toBe(false);
    expect(e.meta.summarized).toBe(false);
    expect(e.meta.recallCount).toBe(0);
    expect(e.meta.relevancy).toBe(1.0); // starts at max
  });

  it('should preserve pinned flag from message', () => {
    const e = enrich({ ...msg('user', 'Important'), pinned: true }, 0);
    expect(e.meta.pinned).toBe(true);
  });
});

describe('computeRelevancy', () => {
  it('should return higher score for recent messages', () => {
    const recent = enriched('user', 'test', 18);
    const old = enriched('user', 'test', 0);

    const recentScore = computeRelevancy(recent, 0.5, 20);
    const oldScore = computeRelevancy(old, 0.5, 20);

    expect(recentScore).toBeGreaterThan(oldScore);
  });

  it('should return higher score for higher priority', () => {
    const critical = enriched('user', 'test', 10, { priority: 'critical' });
    const low = enriched('user', 'test', 10, { priority: 'low' });

    const criticalScore = computeRelevancy(critical, 0.5, 20);
    const lowScore = computeRelevancy(low, 0.5, 20);

    expect(criticalScore).toBeGreaterThan(lowScore);
  });

  it('should return higher score for higher semantic similarity', () => {
    const m = enriched('user', 'test', 10);

    const highSim = computeRelevancy(m, 0.9, 20);
    const lowSim = computeRelevancy(m, 0.1, 20);

    expect(highSim).toBeGreaterThan(lowSim);
  });

  it('should boost recalled messages', () => {
    const neverRecalled = enriched('user', 'test', 10, { recallCount: 0 });
    const recalled = enriched('user', 'test', 10, { recallCount: 3 });

    const neverScore = computeRelevancy(neverRecalled, 0.5, 20);
    const recalledScore = computeRelevancy(recalled, 0.5, 20);

    expect(recalledScore).toBeGreaterThan(neverScore);
  });
});

describe('enforceTokenBudget', () => {
  it('should return all messages if under budget', () => {
    const messages = [
      enriched('user', 'Hello', 0),
      enriched('assistant', 'Hi', 1),
    ];
    const { kept, dropped } = enforceTokenBudget(messages, 10000, 2);
    expect(kept).toHaveLength(2);
    expect(dropped).toHaveLength(0);
  });

  it('should drop lowest relevancy messages first', () => {
    const messages = [
      enriched('user', 'Low relevancy message with extra words to use tokens', 0, { relevancy: 0.1 }),
      enriched('user', 'High relevancy message with extra words to use tokens', 1, { relevancy: 0.9 }),
      enriched('user', 'Current message', 2, { relevancy: 1.0 }),
    ];
    // Budget too tight for all three — something must be dropped
    const allTokens = totalTokens(messages);
    const { kept, dropped } = enforceTokenBudget(messages, Math.floor(allTokens * 0.6), 1);
    // The low relevancy message should be dropped first
    expect(dropped.length).toBeGreaterThan(0);
    expect(dropped[0]!.meta.relevancy).toBe(0.1);
  });

  it('should never drop pinned messages', () => {
    const messages = [
      enriched('user', 'Pinned important', 0, { relevancy: 0.01, pinned: true }),
      enriched('user', 'Not pinned', 1, { relevancy: 0.5 }),
      enriched('user', 'Current', 2),
    ];
    const { kept } = enforceTokenBudget(messages, 15, 1);
    expect(kept.some((m) => m.meta.pinned)).toBe(true);
  });

  it('should never drop the last message (current user message)', () => {
    const messages = [
      enriched('user', 'Old message', 0, { relevancy: 0.5 }),
      enriched('user', 'Current message sacred', 1, { relevancy: 0.01 }),
    ];
    const { kept } = enforceTokenBudget(messages, 5, 0);
    expect(kept.at(-1)!.content).toBe('Current message sacred');
  });
});

describe('deduplicate', () => {
  it('should remove duplicate IDs', () => {
    const m1 = enriched('user', 'Hello', 0);
    const m2 = { ...enriched('user', 'Hello again', 1), id: m1.id };

    const result = deduplicate([m1, m2]);
    expect(result).toHaveLength(1);
  });

  it('should keep first occurrence', () => {
    const m1 = enrich({ id: 'dup', role: 'user', content: 'First', timestamp: 1 }, 0);
    const m2 = enrich({ id: 'dup', role: 'user', content: 'Second', timestamp: 2 }, 1);

    const result = deduplicate([m1, m2]);
    expect(result[0]!.content).toBe('First');
  });
});

describe('orderChronologically', () => {
  it('should sort by turn', () => {
    const m1 = enriched('user', 'Third', 2);
    const m2 = enriched('user', 'First', 0);
    const m3 = enriched('assistant', 'Second', 1);

    const ordered = orderChronologically([m1, m2, m3]);
    expect(ordered.map((m) => m.meta.turn)).toEqual([0, 1, 2]);
  });

  it('should keep system/compressed messages at the top', () => {
    const sys = enriched('user', 'Compressed', 0, { compressed: true });
    sys.role = 'system';
    sys.meta.source = 'system';
    const user = enriched('user', 'User msg', 1);

    const ordered = orderChronologically([user, sys]);
    expect(ordered[0]!.meta.compressed).toBe(true);
  });
});

describe('removeExpired', () => {
  it('should remove messages past their TTL', () => {
    const alive = enriched('user', 'Alive', 0);
    const expired = enriched('user', 'Expired', 1, { expiresAt: Date.now() - 1000 });

    const result = removeExpired([alive, expired]);
    expect(result).toHaveLength(1);
    expect(result[0]!.content).toBe('Alive');
  });

  it('should keep messages without expiresAt', () => {
    const m = enriched('user', 'No expiry', 0);
    expect(removeExpired([m])).toHaveLength(1);
  });
});

describe('totalTokens', () => {
  it('should sum tokens across messages', () => {
    const messages = [
      enriched('user', 'Hello world', 0),
      enriched('assistant', 'Hi there friend', 1),
    ];
    const total = totalTokens(messages);
    expect(total).toBe(messages[0]!.meta.tokens + messages[1]!.meta.tokens);
  });
});
