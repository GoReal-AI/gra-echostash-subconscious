# @echostash/subconscious

**The Subconscious** — framework-agnostic context management middleware for AI agents.

A dedicated, lightweight agent that manages your main agent's context window and memory. It sits between the world and your agent — storing every message, recalling relevant history, compressing old context, and enriching inbound messages so the main agent always has the right context at the right time.

Your agent focuses on its task. The Subconscious keeps it sharp.

## Why

Every AI agent has the same problem: context windows are finite, conversations are not. As conversations grow, agents lose track of earlier decisions, repeat themselves, or hallucinate details. Most solutions compress or truncate — losing information in the process.

The Subconscious takes a different approach. It's a dedicated agent that manages your main agent's memory. It stores everything, recalls what's relevant, and assembles an optimized context window on every turn. Your agent never manages its own context — it just always has what it needs.

## How It Works

```
Any inbound message (user / tool / system / RAG / skill)
        │
        ▼
  ┌──────────────────────────────────────────────────┐
  │  Subconscious                                     │
  │                                                    │
  │  SCRIPTED (deterministic, every message):          │
  │  • Enrich with metadata (tokens, priority, turn)   │
  │  • Score relevancy (semantic + recency + priority)  │
  │  • Enforce token budget                            │
  │  • Deduplicate, order, expire                      │
  │                                                    │
  │  INTELLIGENT (LLM, only when needed):              │
  │  • Classify: passthrough / recall / reshape        │
  │  • Recall: vector search for relevant history      │
  │  • Reshape: rebuild context on topic shift         │
  │                                                    │
  │  BACKGROUND (non-blocking, during free time):      │
  │  • Embed and store every message                   │
  │  • Update running summary                          │
  │  • Compress old messages                           │
  └──────────┬───────────────────────────────────────┘
             │
             ▼
  ┌──────────────────┐
  │  Your Agent       │  ← sees optimized context, not raw history
  │  (any framework)  │  ← has recall() for explicit lookups
  └──────────────────┘
```

## Quick Start

```bash
npm install @echostash/subconscious
```

```typescript
import { Subconscious, MemoryKVStore, MemoryVectorStore } from '@echostash/subconscious';

const sub = new Subconscious({
  vector: new MemoryVectorStore(),  // swap for Chroma, pgvector, etc.
  kv: new MemoryKVStore(),          // swap for Redis, DynamoDB, etc.
  llm: yourLLMAdapter,              // any cheap/fast model (Haiku, GPT-4o-mini, etc.)
});

// Before your agent's LLM call — get curated context
const { messages } = await sub.prepare(incomingMessage);

// Your agent sees optimized context
const response = await yourAgent.chat(messages);

// After your agent responds — store for future recall
await sub.ingest(response);
```

## Key Concepts

### Two Layers

The Subconscious has a **scripted layer** (deterministic, zero LLM calls, runs on every message) and an **intelligent layer** (LLM-powered, runs selectively):

| Layer | What | Cost | When |
|-------|------|------|------|
| **Scripted** | Token counting, priority assignment, relevancy scoring, budget enforcement, deduplication | Free | Every message |
| **Intelligent** | Classification, recall, reshape, compression, summarization | 1-3 cheap LLM calls | When needed |

### Three Actions

On every turn, the Subconscious classifies the inbound message into one of three actions:

| Action | Latency | Frequency | What Happens |
|--------|---------|-----------|-------------|
| **Passthrough** | ~100ms | ~85% of turns | Message is self-contained. Pass it through. |
| **Recall** | ~200ms | ~10% of turns | Message references old context. Quick vector search, inject relevant messages. |
| **Reshape** | ~30-60s | ~5% of turns | Conversation shifted topic. Full context rebuild with status events. |

### Background Processing

Heavy work (embedding, storing, summarizing, compressing) runs in the background while your main agent is thinking — using time that would otherwise be wasted:

```typescript
// The main agent takes 2-10 seconds to respond.
// During that time, the Subconscious:
// - Embeds and stores the message in vector DB
// - Updates the running conversation summary
// - Compresses old messages if needed

// You can also force-process background tasks:
await sub.flush();
```

### Enriched Messages

Every message gets enriched with metadata on arrival:

```typescript
interface MessageMeta {
  turn: number;           // Sequential turn number
  tokens: number;         // Exact token count
  source: MessageSource;  // 'user' | 'assistant' | 'tool' | 'skill' | 'rag' | 'system'
  relevancy: number;      // Dynamic score (0-1), recomputed as conversation evolves
  priority: Priority;     // 'critical' | 'high' | 'normal' | 'low'
  pinned: boolean;        // Never compress or remove
  compressed: boolean;    // Is this a compressed representation?
  recalled: boolean;      // Was this injected by recall?
  summarized: boolean;    // Was this summarized for context?
  topic: string;          // Topic cluster
  recallCount: number;    // How often this was recalled
  expiresAt?: number;     // TTL — auto-removed after this timestamp
}
```

### Hard Rules

These are scripted and non-negotiable — no LLM decides them:

1. **Current user message is sacred** — never modified, never compressed, never summarized
2. **Recent window** — last N messages always in context, always uncompressed
3. **Token budget** — hard ceiling enforced by dropping lowest-relevancy messages first
4. **Pinned messages** — never removed, never compressed
5. **Deduplication** — recalled messages already in context aren't duplicated
6. **Chronological order** — maintained within context (system messages at top)

### Status Events

During heavy operations (reshape), the Subconscious communicates what it's doing:

```typescript
const sub = new Subconscious({
  // ...
  onStatus: (event) => {
    console.log(`[${event.phase}] ${event.message}`);
    // → [classifying] Analyzing message...
    // → [reshaping] Conversation shifted — reshaping context...
    // → [ready] Ready.
  },
});
```

### Explicit Recall

The main agent can ask for more context when it needs it:

```typescript
const recalled = await sub.recall('what did we decide about API keys?');
```

## Storage Adapters

The Subconscious uses pluggable storage. Bring your own:

| Interface | Purpose | Built-in | Production Options |
|-----------|---------|----------|-------------------|
| `VectorStore` | Semantic search over message history | `MemoryVectorStore` | Chroma, pgvector, Qdrant, Pinecone |
| `KVStore` | Message storage, summaries, metadata | `MemoryKVStore` | Redis, DynamoDB, Firestore |
| `LLMAdapter` | Subconscious reasoning + embeddings | — | Haiku, GPT-4o-mini, Gemini Flash |

### Implementing an Adapter

```typescript
import type { VectorStore, KVStore, LLMAdapter } from '@echostash/subconscious';

// Vector store example (Chroma)
class ChromaStore implements VectorStore {
  async store(id: string, embedding: number[], message: EnrichedMessage) { /* ... */ }
  async search(embedding: number[], topK: number) { /* ... */ }
  async delete(id: string) { /* ... */ }
}

// KV store example (Redis)
class RedisStore implements KVStore {
  async get<T>(key: string): Promise<T | null> { /* ... */ }
  async set<T>(key: string, value: T) { /* ... */ }
  async delete(key: string) { /* ... */ }
  async has(key: string): Promise<boolean> { /* ... */ }
}

// LLM adapter example (Anthropic)
class HaikuAdapter implements LLMAdapter {
  async complete(messages: Message[]) { /* ... */ }
  async embed(text: string) { /* ... */ }
}
```

## API Reference

### `Subconscious`

```typescript
const sub = new Subconscious({
  vector: VectorStore,          // required
  kv: KVStore,                  // required
  llm: LLMAdapter,             // required
  sessionId?: string,           // default: auto-generated
  tokenBudget?: number,         // default: 4000
  recentWindow?: number,        // default: 10
  onStatus?: StatusCallback,    // optional
});
```

| Method | Returns | Description |
|--------|---------|-------------|
| `prepare(message)` | `PrepareResult` | Process inbound message, return curated context |
| `ingest(response)` | `IngestResult` | Store agent's response for future recall |
| `recall(query, topK?)` | `EnrichedMessage[]` | Explicit search over conversation history |
| `flush()` | `void` | Process all pending background tasks |
| `getContext()` | `EnrichedMessage[]` | Current curated context snapshot |
| `getSummary()` | `string` | Current running conversation summary |

### Scripted Utilities

These are exported for developers building custom integrations:

```typescript
import {
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
} from '@echostash/subconscious';
```

## Development

```bash
git clone https://github.com/GoReal-AI/gra-echostash-subconscious.git
cd gra-echostash-subconscious
npm install
npm test          # run tests
npm run build     # build ESM + CJS
npm run typecheck # strict TypeScript checks
npm run dev       # watch mode
```

## Roadmap

- [x] Storage adapters: Chroma, Redis (+ in-memory for dev)
- [x] LLM adapters: Google Gemini, OpenAI, Anthropic
- [ ] Storage adapters: pgvector, Qdrant, Pinecone
- [ ] Framework middleware: LangChain, Vercel AI SDK, CrewAI
- [ ] PDK template integration for customizable Subconscious behavior
- [ ] Cross-session memory
- [ ] Conversation-level evals
- [ ] Echostash platform integration (managed templates, analytics, skills)

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

[MIT](LICENSE)

---

Built by [GoReal AI](https://github.com/GoReal-AI). The Subconscious works standalone, but it's even better with [Echostash](https://echostash.com) — the prompt management platform for AI agents.
