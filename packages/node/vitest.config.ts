import { defineConfig } from 'vitest/config';

// Live test files take turns. They share one Mac's model, speech engine and audio,
// so in parallel they measure each other: a latency guard read 6.8s against its 4s
// limit, for no reason but the neighbours.
const live = process.env.APPLE_LLM_LIVE === '1';

export default defineConfig({
  test: { include: ['test/**/*.test.ts'], testTimeout: 300_000, fileParallelism: !live },
});
