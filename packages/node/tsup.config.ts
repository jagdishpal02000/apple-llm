import { defineConfig } from 'tsup';

export default defineConfig([
  {
    // The library, as ESM and CJS with types for each.
    entry: ['src/index.ts', 'src/ai-sdk.ts', 'src/server.ts'],
    format: ['esm', 'cjs'],
    dts: true,
    clean: true,
    // Split for CJS too, so the subpath entries share one copy of the classes:
    // without it `apple-llm/ai-sdk` would throw an AppleLLMError that fails
    // `instanceof` against the one exported from `apple-llm`.
    splitting: true,
    // `import.meta.url` is used to locate the shipped swift/helper.swift; the
    // shim makes that work in the CJS build too.
    shims: true,
    // Types only; the provider never loads it at runtime.
    external: ['@ai-sdk/provider'],
    outExtension: ({ format }) => ({ js: format === 'cjs' ? '.cjs' : '.js' }),
  },
  {
    // The CLI is an executable, not an import: ESM only.
    entry: ['src/cli.ts'],
    format: ['esm'],
    splitting: false,
    shims: true,
    external: ['@ai-sdk/provider'],
  },
]);
