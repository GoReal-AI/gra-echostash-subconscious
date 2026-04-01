/**
 * OpenAI LLM adapter.
 *
 * Uses GPT-4o-mini for completions and text-embedding-3-small for embeddings.
 * Both capabilities in one SDK — simplest adapter to get started.
 *
 * Usage:
 *   import { OpenAIAdapter } from '@echostash/subconscious/llm/openai';
 *
 *   const llm = new OpenAIAdapter({ apiKey: process.env.OPENAI_API_KEY });
 *   // or
 *   const llm = new OpenAIAdapter({ apiKey: '...', model: 'gpt-4.1-mini', embeddingModel: 'text-embedding-3-large' });
 *
 * Requires: npm install openai
 */

import type { LLMAdapter, LLMResponse, Message } from '../types.js';

export interface OpenAIAdapterConfig {
  /** OpenAI API key. Defaults to OPENAI_API_KEY env var. */
  apiKey?: string;
  /** Model for completions. Default: 'gpt-4o-mini' */
  model?: string;
  /** Model for embeddings. Default: 'text-embedding-3-small' */
  embeddingModel?: string;
  /** Base URL override (for proxies, Azure, etc.) */
  baseURL?: string;
}

export class OpenAIAdapter implements LLMAdapter {
  private readonly config: Required<Pick<OpenAIAdapterConfig, 'model' | 'embeddingModel'>> & OpenAIAdapterConfig;
  private client: OpenAIClient | null = null;

  constructor(config: OpenAIAdapterConfig = {}) {
    this.config = {
      ...config,
      model: config.model ?? 'gpt-4o-mini',
      embeddingModel: config.embeddingModel ?? 'text-embedding-3-small',
    };
  }

  private async getClient(): Promise<OpenAIClient> {
    if (this.client) return this.client;

    // Dynamic import so openai isn't required at install time
    const { default: OpenAI } = await import('openai');
    this.client = new OpenAI({
      apiKey: this.config.apiKey,
      baseURL: this.config.baseURL,
    }) as unknown as OpenAIClient;
    return this.client;
  }

  async complete(messages: Message[]): Promise<LLMResponse> {
    const client = await this.getClient();

    const response = await client.chat.completions.create({
      model: this.config.model,
      messages: messages.map((m) => ({
        role: m.role as 'system' | 'user' | 'assistant',
        content: m.content,
      })),
      temperature: 0,
    });

    const choice = response.choices[0];

    return {
      content: choice?.message?.content ?? '',
      usage: response.usage
        ? {
            inputTokens: response.usage.prompt_tokens,
            outputTokens: response.usage.completion_tokens,
          }
        : undefined,
    };
  }

  async embed(text: string): Promise<number[]> {
    const client = await this.getClient();

    const response = await client.embeddings.create({
      model: this.config.embeddingModel,
      input: text,
    });

    return response.data[0]?.embedding ?? [];
  }
}

// Minimal type for the OpenAI client to avoid importing the full SDK at type level
interface OpenAIClient {
  chat: {
    completions: {
      create(params: Record<string, unknown>): Promise<{
        choices: Array<{ message?: { content?: string | null } }>;
        usage?: { prompt_tokens: number; completion_tokens: number };
      }>;
    };
  };
  embeddings: {
    create(params: Record<string, unknown>): Promise<{
      data: Array<{ embedding: number[] }>;
    }>;
  };
}
