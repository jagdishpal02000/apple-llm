import { defineConfig } from 'vite'
import { readFileSync } from 'node:fs'

// The page shows the published npm version, so a release updates the site.
const { version } = JSON.parse(
  readFileSync(new URL('../packages/node/package.json', import.meta.url), 'utf8'),
)

export default defineConfig({
  base: '/apple-llm/',
  plugins: [
    {
      name: 'apple-llm-version',
      transformIndexHtml: (html) => html.replaceAll('%APPLE_LLM_VERSION%', version),
    },
  ],
})
