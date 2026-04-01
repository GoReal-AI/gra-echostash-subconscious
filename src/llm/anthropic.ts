/**
 * Anthropic LLM adapter.
 *
 * Uses Claude Haiku for completions. Since Anthropic doesn't offer embeddings,
 * you provide your own embedding function (OpenAI, Voyage, Cohere, local, etc.).
 *
 * Usage:
 *   import { AnthropicAdapter } from '@echostash/subconscious/llm/anthropic';
 *   import { OpenAIAdapter } from '@echostash/subconscious/llm/openai';
 *
 *   // Use OpenAI for embeddings
 *   const openai = new OpenAIAdapter();
 *   const llm = new AnthropicAdapter({
 *     apiKey: process.env.ANTHROPIC_API_KEY,
 *     embedder: (text) => openai.embed(text),
 *   });
 *
 * Requires: npm install @anthropic-ai/sdk
 */

import type { LLMAdapter, LLMResponse, Message } from '../types.js';

export interface AnthropicAdapterConfig {
  /** Anthropic API key. Defaults to ANTHROPIC_API_KEY env var. */
  apiKey?: string;
  /** Model for completions. Default: 'claude-haiku-4-5-20251001' */
  model?: string;
  /** Embedding function — Anthropic doesn't have embeddings, bring your own. */
  embedder: (text: string) => Promise<number[]>;
  /** Max tokens for completion. Default: 1024 */
  maxTokens?: number;
}

export class AnthropicAdapter implements LLMAdapter {
  private readonly config: Required<Pick<AnthropicAdapterConfig, 'model' | 'maxTokens'>> & AnthropicAdapterConfig;
  private client: AnthropicClient | null = null;

  constructor(config: AnthropicAdapterConfig) {
    this.config = {
      ...config,
      model: config.model ?? 'claude-haiku-4-5-20251001',
      maxTokens: config.maxTokens ?? 1024,
    };
  }

  private async getClient(): Promise<AnthropicClient> {
    if (this.client) return this.client;

    const { default: Anthropic } = await import('@anthropic-ai/sdk');
    this.client = new Anthropic({
      apiKey: this.config.apiKey,
    }) as unknown as AnthropicClient;
    return this.client;
  }

  async complete(messages: Message[]): Promise<LLMResponse> {
    const client = await this.getClient();

    // Anthropic separates system from messages
    const systemMessages = messages.filter((m) => m.role === 'system');
    const nonSystemMessages = messages.filter((m) => m.role !== 'system');

    const systemText = systemMessages.map((m) => m.content).join('\n\n');

    const response = await client.messages.create({
      model: this.config.model,
      max_tokens: this.config.maxTokens,
      system: systemText || undefined,
      messages: nonSystemMessages.map((m) => ({
        role: m.role === 'assistant' ? 'assistant' : 'user',
        content: m.content,
      })),
    });

    const content = response.content
      .filter((block: { type: string }) => block.type === 'text')
      .map((block: { type: string; text: string }) => block.text)
      .join('');

    return {
      content,
      usage: response.usage
        ? {
            inputTokens: response.usage.input_tokens,
            outputTokens: response.usage.output_tokens,
          }
        : undefined,
    };
  }

  async embed(text: string): Promise<number[]> {
    return this.config.embedder(text);
  }
}

// Minimal type for the Anthropic client
interface AnthropicClient {
  messages: {
    create(params: Record<string, unknown>): Promise<{
      content: Array<{ type: string; text: string }>;
      usage?: { input_tokens: number; output_tokens: number };
    }>;
  };
}
