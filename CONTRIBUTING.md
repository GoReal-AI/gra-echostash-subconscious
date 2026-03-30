# Contributing to @echostash/subconscious

Thanks for your interest in contributing! The Subconscious is an open-source project and we welcome contributions of all kinds.

## Getting Started

```bash
git clone https://github.com/GoReal-AI/gra-echostash-subconscious.git
cd gra-echostash-subconscious
npm install
```

## Development

```bash
npm test          # run tests (vitest)
npm run test:watch # watch mode
npm run build     # build ESM + CJS (tsup)
npm run typecheck # TypeScript strict mode
npm run lint      # ESLint
```

## Project Structure

```
src/
├── types.ts                  # All type definitions
├── index.ts                  # Public API exports
├── core/
│   ├── subconscious.ts       # Main Subconscious class
│   ├── engine.ts             # Intelligent layer (LLM-powered)
│   ├── scripted.ts           # Scripted layer (deterministic)
│   ├── background.ts         # Background task queue
│   ├── subconscious.test.ts  # Integration tests
│   └── scripted.test.ts      # Unit tests for scripted layer
├── storage/
│   ├── index.ts              # Storage exports
│   └── memory.ts             # In-memory adapters (dev/test)
├── llm/
│   └── types.ts              # LLM adapter types
└── middleware/
    └── index.ts              # Framework adapter interface
```

### Architecture: Two Layers

The codebase has a clear separation:

- **Scripted layer** (`core/scripted.ts`): Deterministic operations — token counting, priority assignment, relevancy scoring, budget enforcement. Zero LLM calls. Runs on every message.
- **Intelligent layer** (`core/engine.ts`): LLM-powered operations — classification, recall, reshape, compression, summarization. Runs selectively.

When adding new functionality, ask: does this need an LLM? If no, it goes in the scripted layer.

## How to Contribute

### Storage Adapters

We need adapters for popular vector and KV stores. Each adapter implements the `VectorStore` or `KVStore` interface from `types.ts`.

Target adapters:
- Chroma (`@echostash/subconscious-chroma`)
- Redis (`@echostash/subconscious-redis`)
- pgvector (`@echostash/subconscious-pgvector`)
- Qdrant (`@echostash/subconscious-qdrant`)

### LLM Adapters

Adapters for cheap/fast models that the Subconscious uses internally:
- Anthropic Haiku
- OpenAI GPT-4o-mini
- Google Gemini Flash

### Framework Middleware

Adapters that wire the Subconscious into agent frameworks:
- LangChain / deepagents
- Vercel AI SDK
- CrewAI
- OpenAI Agents SDK

### Improving the Scripted Layer

The scripted layer is where we can improve without adding LLM costs:
- Better token counting (currently word-based heuristic)
- Smarter priority patterns
- Relevancy weight tuning
- New scoring signals

## Conventions

- **TypeScript strict mode** — no `any`, explicit return types on public APIs
- **Tests** — co-located with source files (`*.test.ts`). All new code needs tests.
- **Commits** — `type(scope): message` (e.g., `feat(storage): add Chroma adapter`)
- **Branches** — `feat/`, `fix/`, `chore/`

## Pull Requests

1. Fork the repo
2. Create a feature branch (`feat/my-feature`)
3. Write tests for your changes
4. Make sure `npm test` and `npm run typecheck` pass
5. Open a PR with a clear description

## Code of Conduct

Be kind. Be constructive. We're building something to make agents better for everyone.

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
