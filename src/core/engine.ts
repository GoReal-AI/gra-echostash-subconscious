/**
 * The Subconscious Engine — intelligent layer.
 *
 * These are the operations that require an LLM call.
 * The scripted layer (scripted.ts) handles everything deterministic.
 *
 * CRITICAL PATH (blocking):
 *   classify → maybe recall or reshape → return context
 *
 * BACKGROUND (non-blocking):
 *   stash (embed+store) → summarize → compress
 */

import type {
  ClassifyResult,
  EnrichedMessage,
  KVStore,
  LLMAdapter,
  Message,
  VectorStore,
} from '../types.js';
import { countTokens, enrich } from './scripted.js';

// ---------------------------------------------------------------------------
// KV key helpers
// ---------------------------------------------------------------------------

export function sessionKey(sessionId: string, suffix: string): string {
  return `sub:${sessionId}:${suffix}`;
}

// ---------------------------------------------------------------------------
// Engine config
// ---------------------------------------------------------------------------

export interface EngineConfig {
  vector: VectorStore;
  kv: KVStore;
  llm: LLMAdapter;
  sessionId: string;
  tokenBudget: number;
  recentWindow: number;
}

// ===================================================================
// CRITICAL PATH
// ===================================================================

/**
 * Classify the incoming message.
 * The ONE blocking LLM call per turn. ~100ms with Haiku.
 */
export async function classify(
  config: EngineConfig,
  newMessage: EnrichedMessage,
  summary: string,
  contextTokens: number,
): Promise<ClassifyResult> {
  const { llm } = config;

  const response = await llm.complete([
    {
      id: 'sys-classify',
      role: 'system',
      content: `You are the Subconscious — a context management agent. Analyze the incoming message and decide what action is needed.

Current conversation summary:
${summary || '(new conversation)'}

Current context: ${contextTokens} tokens (budget: ${config.tokenBudget})

Respond with JSON: { "action": "passthrough" | "recall" | "reshape", "reasoning": "..." }

Actions:
- "passthrough": The message is self-contained. The current context has everything needed.
- "recall": The message references or needs information from earlier that is NOT in the current context. A quick vector search will find it.
- "reshape": The conversation has shifted topic significantly. The current context is stale and needs rebuilding.

Default to "passthrough". Use "reshape" only when the context would actively mislead the main agent.`,
      timestamp: Date.now(),
    },
    {
      id: 'user-classify',
      role: 'user',
      content: `[${newMessage.meta.source}/${newMessage.role}] ${newMessage.content}`,
      timestamp: Date.now(),
    },
  ]);

  try {
    // Some models (Gemini) wrap JSON in markdown code blocks
    const raw = response.content.replace(/```(?:json)?\s*/g, '').replace(/```\s*/g, '').trim();
    return JSON.parse(raw) as ClassifyResult;
  } catch {
    return { action: 'passthrough', reasoning: 'parse failed — defaulting to passthrough' };
  }
}

/**
 * Recall — vector search for relevant messages. ~50-100ms.
 */
export async function recall(
  config: EngineConfig,
  query: EnrichedMessage | Message,
  topK: number = 5,
): Promise<EnrichedMessage[]> {
  const { vector, llm } = config;

  const embedding = await llm.embed(query.content);
  const results = await vector.search(embedding, topK);

  // Mark as recalled
  return results.map((r) => {
    r.message.meta.recalled = true;
    r.message.meta.recallCount++;
    return r.message;
  });
}

/**
 * Reshape — heavy context rebuild. Multiple LLM calls. Shows status.
 */
export async function reshape(
  config: EngineConfig,
  newMessage: EnrichedMessage,
  currentContext: EnrichedMessage[],
  summary: string,
  currentTurn: number,
): Promise<{ messages: EnrichedMessage[]; briefing: EnrichedMessage }> {
  const { llm } = config;

  // Step 1: Analyze the topic shift
  const topicResponse = await llm.complete([
    {
      id: 'sys-topic',
      role: 'system',
      content: `You are the Subconscious. The conversation has shifted. Determine:
1. What the conversation is NOW about
2. Key decisions/facts from the OLD topic to preserve
3. Search queries to find relevant history for the NEW topic

Current summary: ${summary}

Respond with JSON:
{ "newTopic": "...", "preserveFromOld": ["..."], "searchQueries": ["..."] }`,
      timestamp: Date.now(),
    },
    {
      id: 'user-topic',
      role: 'user',
      content: `New message: ${newMessage.content}`,
      timestamp: Date.now(),
    },
  ]);

  let newTopic = newMessage.content.slice(0, 100);
  let preserveFromOld: string[] = [];
  let searchQueries: string[] = [];

  try {
    const parsed = JSON.parse(topicResponse.content) as {
      newTopic: string;
      preserveFromOld: string[];
      searchQueries: string[];
    };
    newTopic = parsed.newTopic;
    preserveFromOld = parsed.preserveFromOld;
    searchQueries = parsed.searchQueries;
  } catch {
    // proceed with defaults
  }

  // Step 2: Search for relevant history
  const recalledMessages: EnrichedMessage[] = [];
  for (const query of searchQueries) {
    const queryMsg: Message = {
      id: 'reshape-query',
      role: 'user',
      content: query,
      timestamp: Date.now(),
    };
    const results = await recall(config, queryMsg, 3);
    for (const msg of results) {
      if (!recalledMessages.some((m) => m.id === msg.id)) {
        recalledMessages.push(msg);
      }
    }
  }

  // Step 3: Compress old context
  const compressedOld = await compressMessages(
    config,
    currentContext,
    `Focus on preserving: ${preserveFromOld.join(', ')}`,
    currentTurn,
  );

  // Step 4: Build briefing — NEVER mention context management to user
  const briefing = enrich(
    {
      id: `briefing-${Date.now()}`,
      role: 'system',
      content: `[Context Update] The conversation has shifted to: ${newTopic}.

Previous context has been optimized. Key preserved facts:
${preserveFromOld.map((f) => `- ${f}`).join('\n')}

${recalledMessages.length > 0 ? 'Relevant history has been loaded for the new topic.' : ''}

IMPORTANT: Do NOT reference this context update, context management, or any restructuring to the user. Respond naturally as a continuous conversation. Use recall() if you need more details from earlier.`,
      timestamp: Date.now(),
    },
    currentTurn,
  );
  briefing.meta.pinned = true; // briefings survive until next reshape

  // Step 5: Keep recent + pinned from old context
  const recentMessages = currentContext
    .slice(-config.recentWindow)
    .filter((m) => !m.meta.compressed);
  const pinnedMessages = currentContext.filter(
    (m) => m.meta.pinned && !recentMessages.some((r) => r.id === m.id),
  );

  const newContext: EnrichedMessage[] = [
    briefing,
    compressedOld,
    ...pinnedMessages,
    ...recalledMessages,
    ...recentMessages,
  ];

  return { messages: newContext, briefing };
}

// ===================================================================
// BACKGROUND
// ===================================================================

/**
 * Stash — embed and store in vector DB + KV.
 */
export async function stash(
  config: EngineConfig,
  message: EnrichedMessage,
): Promise<void> {
  const { vector, kv, llm, sessionId } = config;

  const embedding = await llm.embed(message.content);

  await Promise.all([
    vector.store(message.id, embedding, message),
    kv.set(sessionKey(sessionId, `msg:${message.id}`), message),
  ]);

  const messageIds =
    (await kv.get<string[]>(sessionKey(sessionId, 'message_ids'))) ?? [];
  messageIds.push(message.id);
  await kv.set(sessionKey(sessionId, 'message_ids'), messageIds);
}

/**
 * Update running summary.
 */
export async function summarize(
  config: EngineConfig,
  currentSummary: string,
  recentMessages: EnrichedMessage[],
): Promise<string> {
  const { llm, kv, sessionId } = config;

  const messagesText = recentMessages
    .map((m) => `[${m.meta.source}/${m.role}] ${m.content}`)
    .join('\n');

  const response = await llm.complete([
    {
      id: 'sys-summarize',
      role: 'system',
      content: `You are the Subconscious. Update the running conversation summary.

Current summary:
${currentSummary || '(new conversation)'}

Rules:
- Keep it under 300 words
- Preserve the narrative flow — what was discussed, decided, pending
- Drop details that are no longer relevant
- Write in present tense`,
      timestamp: Date.now(),
    },
    {
      id: 'user-summarize',
      role: 'user',
      content: `New messages:\n${messagesText}`,
      timestamp: Date.now(),
    },
  ]);

  const newSummary = response.content;
  await kv.set(sessionKey(sessionId, 'summary'), newSummary);
  return newSummary;
}

/**
 * Compress messages into a single enriched summary message.
 */
export async function compressMessages(
  config: EngineConfig,
  messages: EnrichedMessage[],
  focus?: string,
  currentTurn: number = 0,
): Promise<EnrichedMessage> {
  const { llm } = config;

  if (messages.length === 0) {
    const empty = enrich(
      {
        id: `compressed-empty-${Date.now()}`,
        role: 'system',
        content: '[Compressed context] (empty)',
        timestamp: Date.now(),
      },
      currentTurn,
    );
    empty.meta.compressed = true;
    return empty;
  }

  const messagesText = messages
    .map((m) => `[${m.meta.source}/${m.role}, priority:${m.meta.priority}] ${m.content}`)
    .join('\n');

  const response = await llm.complete([
    {
      id: 'sys-compress',
      role: 'system',
      content: `You are the Subconscious. Compress these messages into a concise summary preserving:
- Key decisions and outcomes
- Important facts, names, values
- Action items and commitments
${focus ? `\nSpecial focus: ${focus}` : ''}

Be concise. Write as a narrative.`,
      timestamp: Date.now(),
    },
    {
      id: 'user-compress',
      role: 'user',
      content: messagesText,
      timestamp: Date.now(),
    },
  ]);

  const compressed = enrich(
    {
      id: `compressed-${messages[0]?.id ?? 'x'}-${messages[messages.length - 1]?.id ?? 'x'}`,
      role: 'system',
      content: `[Compressed context] ${response.content}`,
      timestamp: messages[0]?.timestamp ?? Date.now(),
    },
    currentTurn,
  );
  compressed.meta.compressed = true;
  compressed.meta.originalIds = messages.map((m) => m.id);

  return compressed;
}

/**
 * Decide how to represent an assistant response in context.
 */
export async function decideRepresentation(
  config: EngineConfig,
  response: EnrichedMessage,
): Promise<{ message: EnrichedMessage; summarized: boolean }> {
  // Short responses stay in full
  if (response.meta.tokens < 500) {
    return { message: response, summarized: false };
  }

  const { llm } = config;
  const result = await llm.complete([
    {
      id: 'sys-repr',
      role: 'system',
      content: `You are the Subconscious. The assistant gave a long response. Summarize it for context. Preserve:
- What was answered/decided
- Key outputs (code, names, values)
- Action items

Keep under 200 words. Full response is stored for recall.`,
      timestamp: Date.now(),
    },
    {
      id: 'user-repr',
      role: 'user',
      content: response.content,
      timestamp: Date.now(),
    },
  ]);

  const summarized: EnrichedMessage = {
    ...response,
    content: result.content,
    meta: {
      ...response.meta,
      summarized: true,
      tokens: countTokens(result.content),
    },
  };

  return { message: summarized, summarized: true };
}
