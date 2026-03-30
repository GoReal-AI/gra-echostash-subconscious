/**
 * Framework middleware adapters.
 * Framework-specific adapters (LangChain, Vercel AI, etc.) will follow.
 */

import type { Message, PrepareResult, IngestResult } from '../types.js';

export interface SubconsciousMiddleware {
  beforeRequest(messages: Message[]): Promise<PrepareResult>;
  afterResponse(response: Message): Promise<IngestResult>;
}

export type { Message, EnrichedMessage, PrepareResult, IngestResult } from '../types.js';
