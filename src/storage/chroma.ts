/**
 * Chroma vector store adapter.
 *
 * Uses ChromaDB for semantic search over conversation history.
 * Messages are stored with their embeddings and full metadata.
 *
 * Usage:
 *   import { ChromaVectorStore } from '@echostash/subconscious/storage/chroma';
 *
 *   const vector = new ChromaVectorStore({
 *     collectionName: 'my-agent-memory',
 *     url: 'http://localhost:8000',
 *   });
 *
 * Requires: npm install chromadb
 */

import type { EnrichedMessage, VectorSearchResult, VectorStore } from '../types.js';

export interface ChromaVectorStoreConfig {
  /** Chroma collection name. Default: 'subconscious' */
  collectionName?: string;
  /** Chroma server URL. Default: 'http://localhost:8000' */
  url?: string;
  /** Chroma auth token (if using authenticated Chroma) */
  authToken?: string;
  /** Tenant name. Default: 'default_tenant' */
  tenant?: string;
  /** Database name. Default: 'default_database' */
  database?: string;
}

export class ChromaVectorStore implements VectorStore {
  private collection: ChromaCollection | null = null;
  private readonly config: Required<Pick<ChromaVectorStoreConfig, 'collectionName'>> & ChromaVectorStoreConfig;

  constructor(config: ChromaVectorStoreConfig = {}) {
    this.config = {
      ...config,
      collectionName: config.collectionName ?? 'subconscious',
    };
  }

  private async getCollection(): Promise<ChromaCollection> {
    if (this.collection) return this.collection;

    const chromadb = await import('chromadb');
    const client = new chromadb.ChromaClient({
      path: this.config.url ?? 'http://localhost:8000',
    }) as unknown as ChromaClient;

    this.collection = await client.getOrCreateCollection({
      name: this.config.collectionName,
    }) as ChromaCollection;

    return this.collection;
  }

  async store(id: string, embedding: number[], message: EnrichedMessage): Promise<void> {
    const collection = await this.getCollection();

    await collection.upsert({
      ids: [id],
      embeddings: [embedding],
      metadatas: [{
        role: message.role,
        source: message.meta.source,
        turn: message.meta.turn,
        priority: message.meta.priority,
        timestamp: message.timestamp,
        tokens: message.meta.tokens,
      }],
      documents: [message.content],
    });
  }

  async search(embedding: number[], topK: number): Promise<VectorSearchResult[]> {
    const collection = await this.getCollection();

    const results = await collection.query({
      queryEmbeddings: [embedding],
      nResults: topK,
      include: ['embeddings', 'metadatas', 'documents', 'distances'],
    });

    if (!results.ids[0]) return [];

    return results.ids[0].map((id, i) => {
      const metadata = results.metadatas?.[0]?.[i] ?? {};
      const content = results.documents?.[0]?.[i] ?? '';
      const distance = results.distances?.[0]?.[i] ?? 1;

      // Chroma returns L2 distance; convert to similarity score (0-1)
      const score = 1 / (1 + distance);

      return {
        id,
        score,
        message: {
          id,
          role: (metadata.role as string) ?? 'user',
          content,
          timestamp: (metadata.timestamp as number) ?? 0,
          meta: {
            turn: (metadata.turn as number) ?? 0,
            tokens: (metadata.tokens as number) ?? 0,
            source: (metadata.source as string) ?? 'user',
            relevancy: score,
            priority: (metadata.priority as string) ?? 'normal',
            pinned: false,
            compressed: false,
            originalIds: [],
            recalled: true,
            summarized: false,
            topic: '',
            references: [],
            recallCount: 0,
          },
        } as EnrichedMessage,
      };
    });
  }

  async delete(id: string): Promise<void> {
    const collection = await this.getCollection();
    await collection.delete({ ids: [id] });
  }
}

// Minimal types for ChromaDB client
interface ChromaClient {
  getOrCreateCollection(params: { name: string }): Promise<ChromaCollection>;
}

interface ChromaCollection {
  upsert(params: {
    ids: string[];
    embeddings: number[][];
    metadatas?: Record<string, unknown>[];
    documents?: string[];
  }): Promise<void>;
  query(params: {
    queryEmbeddings: number[][];
    nResults: number;
    include?: string[];
  }): Promise<{
    ids: string[][];
    metadatas?: Array<Array<Record<string, unknown>>>;
    documents?: Array<Array<string>>;
    distances?: Array<Array<number>>;
  }>;
  delete(params: { ids: string[] }): Promise<void>;
}
