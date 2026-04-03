export { Subconscious } from './core/subconscious.js';
export { BackgroundQueue } from './core/background.js';
export { PromptLoader } from './core/prompts.js';
export type { EchostashPromptConfig } from './core/prompts.js';

// Scripted utilities — useful for developers building custom integrations
export {
  countTokens,
  totalTokens,
  assignPriority,
  computeRelevancy,
  enrich,
  enforceTokenBudget,
  deduplicate,
  orderChronologically,
  removeExpired,
  RELEVANCY_WEIGHTS,
} from './core/scripted.js';

export type {
  // Messages
  Message,
  EnrichedMessage,
  MessageMeta,
  Role,
  MessageSource,
  Priority,
  // Storage
  VectorStore,
  VectorSearchResult,
  KVStore,
  // LLM
  LLMAdapter,
  LLMResponse,
  // Config
  SubconsciousConfig,
  // Results
  PrepareResult,
  PrepareAction,
  IngestResult,
  // Classification
  ClassifyAction,
  ClassifyResult,
  // Status
  StatusEvent,
  StatusCallback,
  // Background
  BackgroundTask,
  BackgroundTaskType,
} from './types.js';

export { MemoryKVStore, MemoryVectorStore } from './storage/memory.js';
