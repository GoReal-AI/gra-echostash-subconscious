/**
 * In-memory storage adapters for development and testing.
 */

import type { EnrichedMessage, KVStore, VectorSearchResult, VectorStore } from '../types.js';

export class MemoryKVStore implements KVStore {
  private store = new Map<string, unknown>();

  async get<T = unknown>(key: string): Promise<T | null> {
    return (this.store.get(key) as T) ?? null;
  }

  async set<T = unknown>(key: string, value: T): Promise<void> {
    this.store.set(key, value);
  }

  async delete(key: string): Promise<void> {
    this.store.delete(key);
  }

  async has(key: string): Promise<boolean> {
    return this.store.has(key);
  }

  clear(): void {
    this.store.clear();
  }
}

export class MemoryVectorStore implements VectorStore {
  private entries: Array<{
    id: string;
    embedding: number[];
    message: EnrichedMessage;
  }> = [];

  async store(id: string, embedding: number[], message: EnrichedMessage): Promise<void> {
    this.entries = this.entries.filter((e) => e.id !== id);
    this.entries.push({ id, embedding, message });
  }

  async search(embedding: number[], topK: number): Promise<VectorSearchResult[]> {
    const scored = this.entries.map((entry) => ({
      id: entry.id,
      message: entry.message,
      score: cosineSimilarity(embedding, entry.embedding),
    }));

    scored.sort((a, b) => b.score - a.score);
    return scored.slice(0, topK);
  }

  async delete(id: string): Promise<void> {
    this.entries = this.entries.filter((e) => e.id !== id);
  }

  clear(): void {
    this.entries = [];
  }
}

function cosineSimilarity(a: number[], b: number[]): number {
  if (a.length !== b.length) return 0;

  let dotProduct = 0;
  let normA = 0;
  let normB = 0;

  for (let i = 0; i < a.length; i++) {
    dotProduct += (a[i] ?? 0) * (b[i] ?? 0);
    normA += (a[i] ?? 0) ** 2;
    normB += (b[i] ?? 0) ** 2;
  }

  const denominator = Math.sqrt(normA) * Math.sqrt(normB);
  if (denominator === 0) return 0;
  return dotProduct / denominator;
}
