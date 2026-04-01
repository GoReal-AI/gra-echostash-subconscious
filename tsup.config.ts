import { defineConfig } from 'tsup';

export default defineConfig({
  entry: {
    index: 'src/index.ts',
    'llm/openai': 'src/llm/openai.ts',
    'llm/anthropic': 'src/llm/anthropic.ts',
    'storage/index': 'src/storage/index.ts',
    'storage/redis': 'src/storage/redis.ts',
    'storage/chroma': 'src/storage/chroma.ts',
    'middleware/index': 'src/middleware/index.ts',
  },
  format: ['cjs', 'esm'],
  dts: true,
  sourcemap: true,
  clean: true,
  splitting: false,
  treeshake: true,
  external: ['openai', '@anthropic-ai/sdk', 'redis', 'chromadb'],
});
