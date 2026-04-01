/**
 * Redis KV store adapter.
 *
 * Uses the `redis` npm package. Stores all values as JSON strings.
 *
 * Usage:
 *   import { RedisKVStore } from '@echostash/subconscious/storage/redis';
 *
 *   const kv = new RedisKVStore({ url: 'redis://localhost:6379' });
 *   await kv.connect();
 *
 * Requires: npm install redis
 */

import type { KVStore } from '../types.js';

export interface RedisKVStoreConfig {
  /** Redis connection URL. Default: 'redis://localhost:6379' */
  url?: string;
  /** Key prefix to namespace all Subconscious keys. Default: 'sub:' */
  prefix?: string;
  /** TTL in seconds for stored values. Default: none (no expiry) */
  ttl?: number;
}

export class RedisKVStore implements KVStore {
  private client: RedisClient | null = null;
  private readonly config: Required<Pick<RedisKVStoreConfig, 'prefix'>> & RedisKVStoreConfig;

  constructor(config: RedisKVStoreConfig = {}) {
    this.config = {
      ...config,
      prefix: config.prefix ?? 'sub:',
    };
  }

  async connect(): Promise<void> {
    if (this.client) return;

    const redis = await import('redis');
    this.client = redis.createClient({ url: this.config.url }) as unknown as RedisClient;
    await this.client.connect();
  }

  private async getClient(): Promise<RedisClient> {
    if (!this.client) await this.connect();
    return this.client!;
  }

  private key(k: string): string {
    return `${this.config.prefix}${k}`;
  }

  async get<T = unknown>(key: string): Promise<T | null> {
    const client = await this.getClient();
    const value = await client.get(this.key(key));
    if (value === null || value === undefined) return null;
    return JSON.parse(value) as T;
  }

  async set<T = unknown>(key: string, value: T): Promise<void> {
    const client = await this.getClient();
    const serialized = JSON.stringify(value);
    if (this.config.ttl) {
      await client.set(this.key(key), serialized, { EX: this.config.ttl });
    } else {
      await client.set(this.key(key), serialized);
    }
  }

  async delete(key: string): Promise<void> {
    const client = await this.getClient();
    await client.del(this.key(key));
  }

  async has(key: string): Promise<boolean> {
    const client = await this.getClient();
    const exists = await client.exists(this.key(key));
    return exists === 1;
  }

  async disconnect(): Promise<void> {
    if (this.client) {
      await this.client.quit();
      this.client = null;
    }
  }
}

// Minimal type for the redis client
interface RedisClient {
  connect(): Promise<void>;
  quit(): Promise<void>;
  get(key: string): Promise<string | null>;
  set(key: string, value: string, options?: { EX?: number }): Promise<unknown>;
  del(key: string): Promise<number>;
  exists(key: string): Promise<number>;
}
