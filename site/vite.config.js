import { defineConfig } from 'vite'
import { readFileSync } from 'node:fs'
import { fileURLToPath } from 'node:url'

// The page shows the published npm version, so a release updates the site.
const { version } = JSON.parse(
  readFileSync(new URL('../packages/node/package.json', import.meta.url), 'utf8'),
)
const agentGuide = new URL('../llms.txt', import.meta.url)

export default defineConfig({
  base: '/apple-llm/',
  plugins: [
    {
      name: 'apple-llm-agent-guide',
      configureServer(server) {
        const paths = new Set(['/llms.txt', `${server.config.base}llms.txt`])
        server.middlewares.use((req, res, next) => {
          if (!['GET', 'HEAD'].includes(req.method) || !paths.has(req.url?.split('?')[0])) return next()
          res.setHeader('Content-Type', 'text/plain; charset=utf-8')
          res.end(readFileSync(agentGuide, 'utf8'))
        })
      },
      buildStart() {
        this.addWatchFile(fileURLToPath(agentGuide))
      },
      generateBundle() {
        this.emitFile({ type: 'asset', fileName: 'llms.txt', source: readFileSync(agentGuide, 'utf8') })
      },
    },
    {
      name: 'apple-llm-version',
      transformIndexHtml: (html) => html.replaceAll('%APPLE_LLM_VERSION%', version),
    },
  ],
})
