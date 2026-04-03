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
import type { PromptLoader } from './prompts.js';

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
  prompts: PromptLoader;
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
  const { llm, prompts } = config;

  const classifyPrompt = await prompts.render('classify', {
    summary: summary || '',
    contextTokens: String(contextTokens),
    tokenBudget: String(config.tokenBudget),
  });

  const response = await llm.complete([
    {
      id: 'sys-classify',
      role: 'system',
      content: classifyPrompt,
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
  const { llm, prompts } = config;

  // Step 1: Analyze the topic shift
  const topicPrompt = await prompts.render('topicAnalysis', {
    summary: summary || '',
  });

  const topicResponse = await llm.complete([
    {
      id: 'sys-topic',
      role: 'system',
      content: topicPrompt,
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

  try {
    const raw = topicResponse.content.replace(/```(?:json)?\s*/g, '').replace(/```\s*/g, '').trim();
    const parsed = JSON.parse(raw) as {
      newTopic: string;
      preserveFromOld: string[];
      searchQueries: string[];
    };
    newTopic = parsed.newTopic;
    preserveFromOld = parsed.preserveFromOld;
  } catch {
    // proceed with defaults
  }

  // Step 2: Recall based on the new message directly — no LLM-generated queries
  const recalled = await recall(config, newMessage, 3);
  // Deduplicate — don't inject what's already in recent context
  const existingIds = new Set(currentContext.map((m) => m.id));
  const uniqueRecalled = recalled.filter((m) => !existingIds.has(m.id));

  // Step 3: Compress old context — preserves key facts in minimal tokens
  const compressedOld = await compressMessages(
    config,
    currentContext,
    `Focus on preserving: ${preserveFromOld.join(', ')}`,
    currentTurn,
  );

  // Step 4: Build briefing — NEVER mention context management to user
  const briefingContent = await prompts.render('briefing', {
    newTopic,
    preservedFacts: preserveFromOld.map((f) => `- ${f}`).join('\n'),
    hasRecalled: uniqueRecalled.length > 0 ? 'true' : '',
  });

  const briefing = enrich(
    {
      id: `briefing-${Date.now()}`,
      role: 'system',
      content: briefingContent,
      timestamp: Date.now(),
    },
    currentTurn,
  );
  // Briefings are NOT pinned — relevancy decides their fate like everything else.
  // They ARE stashed in vector DB so the Sub can search them for conversation history.

  // Step 5: Keep recent from old context
  const recentMessages = currentContext
    .slice(-config.recentWindow)
    .filter((m) => !m.meta.compressed);

  const newContext: EnrichedMessage[] = [
    briefing,
    compressedOld,
    ...uniqueRecalled,
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

  const summarizePrompt = await config.prompts.render('summarize', {
    currentSummary: currentSummary || '',
  });

  const response = await llm.complete([
    {
      id: 'sys-summarize',
      role: 'system',
      content: summarizePrompt,
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

  const compressPrompt = await config.prompts.render('compress', {
    focus: focus || '',
  });

  const response = await llm.complete([
    {
      id: 'sys-compress',
      role: 'system',
      content: compressPrompt,
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
  const reprPrompt = await config.prompts.render('representation', {});

  const result = await llm.complete([
    {
      id: 'sys-repr',
      role: 'system',
      content: reprPrompt,
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
