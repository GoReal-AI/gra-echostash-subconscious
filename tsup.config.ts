import { defineConfig } from 'tsup';

export default defineConfig({
  entry: {
    index: 'src/index.ts',
    'llm/openai': 'src/llm/openai.ts',
    'llm/anthropic': 'src/llm/anthropic.ts',
    'middleware/index': 'src/middleware/index.ts',
    'storage/index': 'src/storage/index.ts',
  },
  format: ['cjs', 'esm'],
  dts: true,
  sourcemap: true,
  clean: true,
  splitting: false,
  treeshake: true,
  external: ['openai', '@anthropic-ai/sdk'],
});
