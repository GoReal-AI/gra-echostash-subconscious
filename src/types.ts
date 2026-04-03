/**
 * Core types for the Subconscious.
 */

// ---------------------------------------------------------------------------
// Messages — what developers pass in
// ---------------------------------------------------------------------------

export type Role = 'system' | 'user' | 'assistant' | 'tool';

export type MessageSource = 'user' | 'assistant' | 'tool' | 'skill' | 'rag' | 'system';

export type Priority = 'critical' | 'high' | 'normal' | 'low';

/** What the developer passes in — minimal, no metadata required. */
export interface Message {
  id: string;
  role: Role;
  content: string;
  timestamp: number;
  /** Where this message came from. Defaults to role-based inference. */
  source?: MessageSource;
  /** Developer can pin a message so it's never compressed or removed. */
  pinned?: boolean;
  /** ID of the message this is a response/reaction to (links tool results to user messages) */
  replyTo?: string;
}

// ---------------------------------------------------------------------------
// Enriched messages — what the Subconscious produces internally
// ---------------------------------------------------------------------------

/** Full metadata tracked per message by the Subconscious. */
export interface MessageMeta {
  /** Sequential turn number (0, 1, 2...) */
  turn: number;
  /** Exact token count, computed once on arrival */
  tokens: number;
  /** Message source */
  source: MessageSource;
  /** Dynamic relevancy score (0–1), recomputed as conversation evolves */
  relevancy: number;
  /** Priority tier */
  priority: Priority;
  /** Pinned — never compress, never remove */
  pinned: boolean;
  /** Is this a compressed representation of multiple messages? */
  compressed: boolean;
  /** Original message IDs if compressed */
  originalIds: string[];
  /** Was this injected by recall? */
  recalled: boolean;
  /** Was the content summarized for context? (full version in storage) */
  summarized: boolean;
  /** Topic cluster this message belongs to */
  topic: string;
  /** Message IDs this message references */
  references: string[];
  /** How many times this message has been recalled */
  recallCount: number;
  /** TTL — if set, message expires from context after this timestamp */
  expiresAt?: number;
}

/** A message enriched by the Subconscious with full metadata. */
export interface EnrichedMessage extends Message {
  meta: MessageMeta;
}

// ---------------------------------------------------------------------------
// Storage adapters
// ---------------------------------------------------------------------------

export interface VectorSearchResult {
  id: string;
  message: EnrichedMessage;
  score: number;
}

export interface VectorStore {
  store(id: string, embedding: number[], message: EnrichedMessage): Promise<void>;
  search(embedding: number[], topK: number): Promise<VectorSearchResult[]>;
  delete(id: string): Promise<void>;
}

export interface KVStore {
  get<T = unknown>(key: string): Promise<T | null>;
  set<T = unknown>(key: string, value: T): Promise<void>;
  delete(key: string): Promise<void>;
  has(key: string): Promise<boolean>;
}

// ---------------------------------------------------------------------------
// LLM adapter
// ---------------------------------------------------------------------------

export interface LLMResponse {
  content: string;
  usage?: {
    inputTokens: number;
    outputTokens: number;
  };
}

export interface LLMAdapter {
  complete(messages: Message[]): Promise<LLMResponse>;
  embed(text: string): Promise<number[]>;
}

// ---------------------------------------------------------------------------
// Classification
// ---------------------------------------------------------------------------

export type ClassifyAction = 'passthrough' | 'recall' | 'reshape';

export interface ClassifyResult {
  action: ClassifyAction;
  reasoning: string;
}

// ---------------------------------------------------------------------------
// Status events
// ---------------------------------------------------------------------------

export interface StatusEvent {
  phase: 'classifying' | 'recalling' | 'reshaping' | 'compressing' | 'ready';
  message: string;
  progress?: number;
}

export type StatusCallback = (event: StatusEvent) => void;

// ---------------------------------------------------------------------------
// Background tasks
// ---------------------------------------------------------------------------

export type BackgroundTaskType = 'embed' | 'store' | 'summarize' | 'compress' | 'recompute-relevancy';

export interface BackgroundTask {
  type: BackgroundTaskType;
  execute: () => Promise<void>;
}

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

export interface SubconsciousConfig {
  vector: VectorStore;
  kv: KVStore;
  llm: LLMAdapter;
  sessionId?: string;
  /** Max tokens for the main agent's context window. Default: 4000 */
  tokenBudget?: number;
  /** Always keep the last N messages uncompressed. Default: 10 */
  recentWindow?: number;
  /** Status callback — shows the user what the Subconscious is doing */
  onStatus?: StatusCallback;
}

// ---------------------------------------------------------------------------
// Prepare result
// ---------------------------------------------------------------------------

export interface PrepareAction {
  type: 'passthrough' | 'recall' | 'reshape' | 'inject' | 'compress' | 'briefing' | 'budget-trim';
  detail: string;
}

export interface PrepareResult {
  /** The curated message array — ready to send to the main agent */
  messages: EnrichedMessage[];
  /** What the Subconscious did this turn (for developer inspection) */
  actions: PrepareAction[];
  /** The classification action taken */
  classification: ClassifyAction;
  /** The current running summary */
  summary: string;
  /** How many messages were recalled from storage */
  recalled: number;
  /** How many messages were compressed */
  compressed: number;
  /** Total tokens in the curated context */
  totalTokens: number;
  /** Whether background tasks are pending (call flush() to process) */
  pendingBackground: boolean;
}

// ---------------------------------------------------------------------------
// Ingest result
// ---------------------------------------------------------------------------

export interface IngestResult {
  representation: 'full' | 'summarized';
  pendingBackground: boolean;
}
