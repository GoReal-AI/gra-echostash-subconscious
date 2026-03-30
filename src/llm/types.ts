/**
 * LLM adapter types.
 *
 * The Subconscious needs a cheap/fast LLM for its own reasoning
 * (classify, compress, summarize) and an embedding model.
 *
 * Developers provide an adapter matching the LLMAdapter interface.
 * We'll ship adapters for common providers (Anthropic, OpenAI, etc.).
 */

export type { LLMAdapter, LLMResponse } from '../types.js';
