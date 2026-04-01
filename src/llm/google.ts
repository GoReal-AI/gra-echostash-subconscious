/**
 * Google AI (Gemini) LLM adapter.
 *
 * Uses Gemini Flash for completions and Google's embedding model.
 * Both capabilities in one SDK — and free tier available.
 *
 * Usage:
 *   import { GoogleAdapter } from '@echostash/subconscious/llm/google';
 *
 *   const llm = new GoogleAdapter({ apiKey: process.env.GOOGLE_AI_API_KEY });
 *   // or
 *   const llm = new GoogleAdapter({ apiKey: '...', model: 'gemini-2.5-flash' });
 *
 * Requires: npm install @google/generative-ai
 */

import type { LLMAdapter, LLMResponse, Message } from '../types.js';

export interface GoogleAdapterConfig {
  /** Google AI API key. Defaults to GOOGLE_AI_API_KEY or VERTEX_AI_API_KEY env var. */
  apiKey?: string;
  /** Model for completions. Default: 'gemini-2.5-flash' */
  model?: string;
  /** Model for embeddings. Default: 'text-embedding-004' */
  embeddingModel?: string;
}

export class GoogleAdapter implements LLMAdapter {
  private readonly config: Required<Pick<GoogleAdapterConfig, 'model' | 'embeddingModel'>> & GoogleAdapterConfig;
  private genAI: GoogleGenAI | null = null;

  constructor(config: GoogleAdapterConfig = {}) {
    this.config = {
      ...config,
      model: config.model ?? 'gemini-2.5-flash',
      embeddingModel: config.embeddingModel ?? 'gemini-embedding-001',
    };
  }

  private async getClient(): Promise<GoogleGenAI> {
    if (this.genAI) return this.genAI;

    const { GoogleGenerativeAI } = await import('@google/generative-ai');
    const apiKey = this.config.apiKey
      ?? process.env.GOOGLE_AI_API_KEY
      ?? process.env.VERTEX_AI_API_KEY
      ?? '';

    this.genAI = new GoogleGenerativeAI(apiKey) as unknown as GoogleGenAI;
    return this.genAI;
  }

  async complete(messages: Message[]): Promise<LLMResponse> {
    const client = await this.getClient();
    const model = client.getGenerativeModel({ model: this.config.model });

    // Separate system from conversation messages
    const systemMessages = messages.filter((m) => m.role === 'system');
    const nonSystem = messages.filter((m) => m.role !== 'system');

    const systemInstruction = systemMessages.map((m) => m.content).join('\n\n') || undefined;

    // Convert to Gemini format
    const contents = nonSystem.map((m) => ({
      role: m.role === 'assistant' ? 'model' : 'user',
      parts: [{ text: m.content }],
    }));

    const result = await model.generateContent({
      contents,
      systemInstruction: systemInstruction ? { role: 'user', parts: [{ text: systemInstruction }] } : undefined,
      generationConfig: { temperature: 0 },
    });

    const response = result.response;
    const text = response.text?.() ?? response.candidates?.[0]?.content?.parts?.[0]?.text ?? '';
    const usage = response.usageMetadata;

    return {
      content: text,
      usage: usage
        ? {
            inputTokens: usage.promptTokenCount ?? 0,
            outputTokens: usage.candidatesTokenCount ?? 0,
          }
        : undefined,
    };
  }

  async embed(text: string): Promise<number[]> {
    const { GoogleGenerativeAI } = await import('@google/generative-ai');
    const apiKey = this.config.apiKey
      ?? process.env.GOOGLE_AI_API_KEY
      ?? process.env.VERTEX_AI_API_KEY
      ?? '';

    const genAI = new GoogleGenerativeAI(apiKey);
    const model = genAI.getGenerativeModel({ model: this.config.embeddingModel });

    const result = await model.embedContent(text);
    return result.embedding?.values ?? [];
  }
}

// Minimal types for the generative AI client
interface GoogleGenAI {
  getGenerativeModel(params: { model: string }): GoogleModel;
}

interface GoogleModel {
  generateContent(params: {
    contents: Array<{ role: string; parts: Array<{ text: string }> }>;
    systemInstruction?: { role: string; parts: Array<{ text: string }> };
    generationConfig?: { temperature?: number };
  }): Promise<{
    response: {
      text?: () => string;
      candidates?: Array<{ content?: { parts?: Array<{ text?: string }> } }>;
      usageMetadata?: { promptTokenCount?: number; candidatesTokenCount?: number };
    };
  }>;
}
