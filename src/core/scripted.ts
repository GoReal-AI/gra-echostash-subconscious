/**
 * Scripted layer — deterministic operations that run on every message.
 * Zero LLM calls. Pure math and rules.
 *
 * - Token counting
 * - Priority assignment (keyword-based)
 * - Relevancy scoring (cosine similarity + recency + priority + recall boost)
 * - Source inference
 * - Budget enforcement
 * - Deduplication
 * - Ordering
 * - Enrichment (Message → EnrichedMessage)
 * - Expiry
 */

import type {
  EnrichedMessage,
  Message,
  MessageMeta,
  MessageSource,
  Priority,
} from '../types.js';

// ---------------------------------------------------------------------------
// Token counting
// ---------------------------------------------------------------------------

/**
 * Count tokens in a string.
 *
 * Uses a fast heuristic: split on whitespace + punctuation boundaries.
 * More accurate than chars/4, cheaper than a real tokenizer.
 * Good enough for budget enforcement — not billing.
 */
export function countTokens(text: string): number {
  if (!text) return 0;
  // Split on whitespace and count. This is roughly 1.3 tokens per word
  // for English text, which aligns with most tokenizers.
  const words = text.split(/\s+/).filter(Boolean);
  return Math.ceil(words.length * 1.3);
}

/** Total tokens across all messages. */
export function totalTokens(messages: EnrichedMessage[]): number {
  return messages.reduce((sum, m) => sum + m.meta.tokens, 0);
}

// ---------------------------------------------------------------------------
// Priority assignment
// ---------------------------------------------------------------------------

const CRITICAL_PATTERNS = [
  /\bdecid(ed|e|ing)\b/i,
  /\blet'?s go with\b/i,
  /\bwe('ll| will) use\b/i,
  /\bagreed\b/i,
  /\bconfirm(ed)?\b/i,
  /\bapproved?\b/i,
  /\bmust\b/i,
  /\brequirement\b/i,
];

const HIGH_PATTERNS = [
  /\btodo\b/i,
  /\bneed to\b/i,
  /\bwill do\b/i,
  /\baction item/i,
  /\bnext step/i,
  /```/,               // code blocks
  /\bconfig\b/i,
  /\berror\b/i,
  /\bbug\b/i,
  /\bfix\b/i,
];

const LOW_PATTERNS = [
  /^(hi|hello|hey|thanks|thank you|ok|okay|sure|got it|sounds good|great|cool|nice)/i,
  /^(yes|no|yep|nope|yeah|nah)$/i,
];

/** Assign priority based on content signals. No LLM — just pattern matching. */
export function assignPriority(content: string): Priority {
  if (CRITICAL_PATTERNS.some((p) => p.test(content))) return 'critical';
  if (HIGH_PATTERNS.some((p) => p.test(content))) return 'high';
  if (LOW_PATTERNS.some((p) => p.test(content))) return 'low';
  return 'normal';
}

// ---------------------------------------------------------------------------
// Source inference
// ---------------------------------------------------------------------------

/** Infer message source from role if not explicitly set. */
export function inferSource(message: Message): MessageSource {
  if (message.source) return message.source;
  switch (message.role) {
    case 'user': return 'user';
    case 'assistant': return 'assistant';
    case 'tool': return 'tool';
    case 'system': return 'system';
    default: return 'system';
  }
}

// ---------------------------------------------------------------------------
// Relevancy scoring
// ---------------------------------------------------------------------------

/**
 * Weights for the relevancy scoring formula.
 * These can be tuned — they're a starting point.
 */
export const RELEVANCY_WEIGHTS = {
  semantic: 0.4,
  recency: 0.3,
  priority: 0.2,
  recallBoost: 0.1,
} as const;

/** Priority to numeric weight. */
function priorityWeight(priority: Priority): number {
  switch (priority) {
    case 'critical': return 1.0;
    case 'high': return 0.7;
    case 'normal': return 0.4;
    case 'low': return 0.1;
  }
}

/** Time decay — 1.0 for recent, decays toward 0. Half-life ~20 turns. */
function recencyScore(messageTurn: number, currentTurn: number): number {
  const age = currentTurn - messageTurn;
  if (age <= 0) return 1.0;
  return Math.exp(-age / 20);
}

/** Recall boost — messages that have been recalled before are more important. */
function recallBoost(recallCount: number): number {
  if (recallCount === 0) return 0;
  // Diminishing returns: 1 recall = 0.5, 2 = 0.75, 3+ = ~0.87
  return 1 - 1 / (1 + recallCount);
}

/**
 * Compute relevancy score for a message.
 *
 * @param message - The enriched message
 * @param semanticSimilarity - Cosine similarity to the current topic (0-1)
 * @param currentTurn - The current conversation turn number
 * @returns Relevancy score (0-1)
 */
export function computeRelevancy(
  message: EnrichedMessage,
  semanticSimilarity: number,
  currentTurn: number,
): number {
  const w = RELEVANCY_WEIGHTS;
  return (
    semanticSimilarity * w.semantic +
    recencyScore(message.meta.turn, currentTurn) * w.recency +
    priorityWeight(message.meta.priority) * w.priority +
    recallBoost(message.meta.recallCount) * w.recallBoost
  );
}

/**
 * Recompute relevancy for all messages in context.
 * Takes an array of semantic similarities (one per message, same order).
 */
export function recomputeAllRelevancy(
  messages: EnrichedMessage[],
  semanticSimilarities: number[],
  currentTurn: number,
): void {
  for (let i = 0; i < messages.length; i++) {
    const msg = messages[i];
    const sim = semanticSimilarities[i] ?? 0;
    if (msg) {
      msg.meta.relevancy = computeRelevancy(msg, sim, currentTurn);
    }
  }
}

// ---------------------------------------------------------------------------
// Enrichment — Message → EnrichedMessage
// ---------------------------------------------------------------------------

/**
 * Enrich a raw Message with full metadata.
 * Called once when a message first enters the Subconscious.
 */
export function enrich(message: Message, turn: number): EnrichedMessage {
  const tokens = countTokens(message.content);
  const source = inferSource(message);
  const priority = assignPriority(message.content);

  const meta: MessageMeta = {
    turn,
    tokens,
    source,
    relevancy: 1.0, // starts at max, decays over time
    priority,
    pinned: message.pinned ?? false,
    compressed: false,
    originalIds: [],
    recalled: false,
    summarized: false,
    topic: '',
    references: message.replyTo ? [message.replyTo] : [],
    recallCount: 0,
  };

  return { ...message, meta };
}

// ---------------------------------------------------------------------------
// Budget enforcement
// ---------------------------------------------------------------------------

/**
 * Enforce a token budget on the context.
 *
 * Hard rules (scripted, non-negotiable):
 * 1. Current user message is NEVER removed
 * 2. Pinned messages are NEVER removed
 * 3. Recent window (last N messages) is kept intact
 * 4. Remaining messages are sorted by relevancy, lowest dropped first
 *
 * Returns the trimmed context and the messages that were dropped.
 */
export function enforceTokenBudget(
  messages: EnrichedMessage[],
  budget: number,
  recentWindow: number,
): { kept: EnrichedMessage[]; dropped: EnrichedMessage[] } {
  const current = totalTokens(messages);
  if (current <= budget) {
    return { kept: messages, dropped: [] };
  }

  // Separate into protected and candidates
  const lastIndex = messages.length - 1;
  const recentStart = Math.max(0, messages.length - recentWindow);

  const protectedSet = new Set<number>();
  for (let i = 0; i < messages.length; i++) {
    const msg = messages[i]!;
    // Protect: current message (last), pinned, recent window
    if (i === lastIndex || msg.meta.pinned || i >= recentStart) {
      protectedSet.add(i);
    }
  }

  // Sort candidates by relevancy (lowest first — drop these first)
  const candidates: Array<{ index: number; msg: EnrichedMessage }> = [];
  for (let i = 0; i < messages.length; i++) {
    if (!protectedSet.has(i)) {
      candidates.push({ index: i, msg: messages[i]! });
    }
  }
  candidates.sort((a, b) => a.msg.meta.relevancy - b.msg.meta.relevancy);

  // Drop lowest-relevancy messages until under budget
  const dropSet = new Set<number>();
  let tokens = current;
  for (const candidate of candidates) {
    if (tokens <= budget) break;
    dropSet.add(candidate.index);
    tokens -= candidate.msg.meta.tokens;
  }

  const kept: EnrichedMessage[] = [];
  const dropped: EnrichedMessage[] = [];
  for (let i = 0; i < messages.length; i++) {
    if (dropSet.has(i)) {
      dropped.push(messages[i]!);
    } else {
      kept.push(messages[i]!);
    }
  }

  return { kept, dropped };
}

// ---------------------------------------------------------------------------
// Deduplication
// ---------------------------------------------------------------------------

/** Remove duplicate messages by ID. Keeps the first occurrence. */
export function deduplicate(messages: EnrichedMessage[]): EnrichedMessage[] {
  const seen = new Set<string>();
  return messages.filter((m) => {
    if (seen.has(m.id)) return false;
    seen.add(m.id);
    return true;
  });
}

// ---------------------------------------------------------------------------
// Ordering
// ---------------------------------------------------------------------------

/**
 * Sort messages chronologically.
 * System/briefing messages stay at the top, everything else by turn then timestamp.
 */
export function orderChronologically(messages: EnrichedMessage[]): EnrichedMessage[] {
  return [...messages].sort((a, b) => {
    // Briefings and compressed context always first
    const aIsSystem = a.meta.compressed || (a.role === 'system' && a.meta.source === 'system');
    const bIsSystem = b.meta.compressed || (b.role === 'system' && b.meta.source === 'system');
    if (aIsSystem && !bIsSystem) return -1;
    if (!aIsSystem && bIsSystem) return 1;

    // Then by turn
    if (a.meta.turn !== b.meta.turn) return a.meta.turn - b.meta.turn;

    // Then by timestamp
    return a.timestamp - b.timestamp;
  });
}

// ---------------------------------------------------------------------------
// Expiry
// ---------------------------------------------------------------------------

/** Remove expired messages. */
export function removeExpired(messages: EnrichedMessage[], now: number = Date.now()): EnrichedMessage[] {
  return messages.filter((m) => !m.meta.expiresAt || m.meta.expiresAt > now);
}
