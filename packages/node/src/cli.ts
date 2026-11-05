#!/usr/bin/env node
/**
 * apple-llm CLI — how people debug this, and how it is tested by hand.
 *
 *   apple-llm probe
 *   apple-llm setup-cloud [--web-search] [--force]
 *   apple-llm run --tier device --system "..." -      # prompt on stdin
 *   apple-llm run --tier cloud --schema s.json "Extract the fields"
 *   apple-llm chat                                     # interactive
 *   apple-llm serve                                    # OpenAI-compatible API
 */
import { fstatSync } from 'node:fs';
import { readFile } from 'node:fs/promises';
import { AppleLLM, captureScreenshot, parseImageFlag, probe } from './index.js';
import { installCloudShortcut } from './cloud.js';
import { AppleLLMError } from './errors.js';
import { runRepl } from './repl.js';
import { DEFAULT_PORT, serve } from './server.js';

interface Flags {
  tier?: string;
  fallback?: string;
  system?: string;
  schema?: string;
  temperature?: string;
  maxTokens?: string;
  webSearch?: boolean;
  force?: boolean;
  json?: boolean;
  image: string[];
  document: string[];
  tool: string[];
  session?: string;
  stream?: boolean;
  mode?: string;
  useCase?: string;
  guardrails?: string;
  seed?: string;
  greedy?: boolean;
  timeout?: string;
  port?: string;
  host?: string;
  apiKey?: string;
  cors?: string;
  positional: string[];
}

function parseArgs(argv: string[]): Flags {
  const flags: Flags = { positional: [], image: [], document: [], tool: [] };
  for (let i = 0; i < argv.length; i += 1) {
    const arg = argv[i];
    const take = (flag: string): string => {
      const v = argv[++i];
      if (v === undefined || v.startsWith('--')) {
        throw new Error(`${flag} needs a value.\n\n${USAGE}`);
      }
      return v;
    };
    switch (arg) {
      case '--tier': flags.tier = take('--tier'); break;
      case '--fallback': flags.fallback = take('--fallback'); break;
      case '--system': flags.system = take('--system'); break;
      case '--schema': flags.schema = take('--schema'); break;
      case '--temperature': flags.temperature = take('--temperature'); break;
      case '--max-tokens': flags.maxTokens = take('--max-tokens'); break;
      case '--web-search': flags.webSearch = true; break;
      case '--force': flags.force = true; break;
      case '--json': flags.json = true; break;
      case '--image': flags.image.push(take('--image')); break;
      case '--document': flags.document.push(take('--document')); break;
      case '--tool': flags.tool.push(take('--tool')); break;
      case '--session': flags.session = take('--session'); break;
      case '--stream': flags.stream = true; break;
      case '--mode': flags.mode = take('--mode'); break;
      case '--use-case': flags.useCase = take('--use-case'); break;
      case '--guardrails': flags.guardrails = take('--guardrails'); break;
      case '--seed': flags.seed = take('--seed'); break;
      case '--greedy': flags.greedy = true; break;
      case '--timeout': flags.timeout = take('--timeout'); break;
      case '--port': flags.port = take('--port'); break;
      case '--host': flags.host = take('--host'); break;
      case '--api-key': flags.apiKey = take('--api-key'); break;
      case '--cors': flags.cors = take('--cors'); break;
      default: flags.positional.push(arg);
    }
  }
  return flags;
}

async function readStdin(): Promise<string> {
  const chunks: Buffer[] = [];
  for await (const chunk of process.stdin) chunks.push(chunk as Buffer);
  return Buffer.concat(chunks).toString('utf8');
}

/**
 * Text piped in beside a prompt — `cat notes.md | apple-llm run "Summarize
 * this"` — or undefined. Only a pipe or a redirected file is read: a terminal
 * is interactive, and a socket (how Node spawns children) may never close.
 */
async function pipedInput(): Promise<string | undefined> {
  let stat;
  try {
    stat = fstatSync(0);
  } catch {
    return undefined;
  }
  if (!stat.isFIFO() && !stat.isFile()) return undefined;
  const text = await readStdin();
  return text.trim() === '' ? undefined : text;
}

/** With text piped in, the prompt says what to do with it — after it, as people read it. */
async function promptFrom(arg: string): Promise<string> {
  if (arg === '-') return readStdin();
  const piped = await pipedInput();
  return piped === undefined ? arg : `${piped.trimEnd()}\n\n${arg}`;
}

const KNOWN_TOOLS = new Set(['ocr', 'barcode', 'spotlight']);

function parseTools(values: string[], usageError: (m: string) => number): string[] | undefined {
  if (values.length === 0) return undefined;
  const out: string[] = [];
  for (const value of values) {
    for (const part of value.split(',').map((s) => s.trim()).filter(Boolean)) {
      if (!KNOWN_TOOLS.has(part)) {
        throw new Error(`--tool must be ocr, barcode, or spotlight (got "${part}").`);
      }
      if (!out.includes(part)) out.push(part);
    }
  }
  return out;
}

const USAGE = `apple-llm — Apple's on-device and Private Cloud Compute models

  apple-llm "<prompt>"                   one completion (short for run)
  apple-llm probe                        what this machine can do
  apple-llm setup-cloud [--web-search]   install the Shortcut the cloud tier needs
  apple-llm run [options] <prompt|->     one completion ("-" reads stdin)
  apple-llm chat [options]               interactive conversation, streamed
  apple-llm serve [options]              OpenAI- and Ollama-compatible API on localhost
  apple-llm count [options] <prompt|->   tokens this prompt costs, before sending
  apple-llm ask-screen [options] <question|->   screenshot, then ask (device tier)
  apple-llm history --session <id>       show a conversation's mirrored turns
  apple-llm reset [--session <id>]       drop one conversation, or all
  apple-llm rewrite|proofread|summarize|draft <text|->   Write with Siri presets

  Text piped into run goes in before the prompt:
    cat notes.md | apple-llm "Summarize this"
    git diff | apple-llm "Write a commit message for this"

run options:
  --tier device|cloud|auto   default: auto
  --fallback device          cloud tier: answer on device when Apple rate-limits the cloud
  --system <text>            system instructions
  --schema <file.json>       ask for JSON matching this JSON Schema
  --temperature <n>          default 0.4 (0 degenerates; see the README)
  --max-tokens <n>
  --image <path[::label]>    attach an image, optionally labelled (repeatable; macOS 27+)
  --document <path>          inline a text document into the prompt (repeatable)
  --tool ocr|barcode|spotlight   built-in on-device tool (repeatable)
  --session <id>             continue a named conversation across calls
  --stream                   print text as it arrives (the cloud tier sends it whole)
  --use-case contentTagging  Apple's tagging-specialised model
  --guardrails permissive    relax guardrails for rewriting tasks
  --seed <n>                 reproducible sampling (top-k, seeded)
  --greedy                   greedy decoding (deterministic, but degenerates)
  --web-search               cloud tier only
  --timeout <seconds>        give up after this long
  --json                     print the full result: text, usage, tool calls
                             (with --schema --stream: one partial object per line)

chat options: --tier, --fallback, --system, --tool, --use-case, --guardrails, --max-tokens.
  In the chat: /reset, /system <text>, /image <path>, /tokens, /help, /exit.

serve options:
  --port <n>                 default ${DEFAULT_PORT}
  --host <addr>              default 127.0.0.1 (0.0.0.0 exposes it to your network)
  --api-key <key>            require "Authorization: Bearer <key>"
  --cors <origin>            allow browser clients from this origin
  --tier device|cloud|auto   for requests not naming a model; default device
  --fallback device          answer apple-private-cloud requests on device when rate-limited

ask-screen options: --mode interactive|window|fullscreen (default interactive),
  plus --system, --session, --max-tokens.
`;

const COMMANDS = new Set([
  'probe', 'setup-cloud', 'run', 'chat', 'serve', 'count', 'ask-screen', 'history', 'reset',
  'rewrite', 'proofread', 'summarize', 'draft', 'help',
]);

async function main(): Promise<number> {
  let [command, ...rest] = process.argv.slice(2);
  // `apple-llm "What does this error mean?"` is a prompt, not a mistyped
  // command: a command never contains a space, and a typo still errors.
  if (command !== undefined && !COMMANDS.has(command) && /\s/.test(command)) {
    rest = [command, ...rest];
    command = 'run';
  }
  let flags: Flags;
  try {
    flags = parseArgs(rest);
  } catch (err) {
    process.stderr.write(`${err instanceof Error ? err.message : String(err)}\n`);
    return 2;
  }

  const usageError = (message: string): number => {
    process.stderr.write(`${message}\n\n${USAGE}`);
    return 2;
  };

  if (command === undefined || command === '--help' || command === '-h' || command === 'help') {
    process.stdout.write(USAGE);
    return 0;
  }
  if (flags.fallback !== undefined && flags.fallback !== 'device') {
    return usageError(`--fallback must be device (got "${flags.fallback}").`);
  }
  const fallback = flags.fallback as 'device' | undefined;

  if (command === 'probe') {
    const result = await probe((p) => process.stderr.write(`  ${p.status}\n`));
    if (flags.json === true) {
      process.stdout.write(`${JSON.stringify(result, null, 2)}\n`);
      return result.device.available || result.cloud.available ? 0 : 1;
    }
    const d = result.device;
    process.stdout.write('on-device\n');
    process.stdout.write(`  available   ${d.available}\n`);
    if (d.variant !== undefined) process.stdout.write(`  variant     ${d.variant}\n`);
    if (d.contextSize !== undefined) process.stdout.write(`  context     ${d.contextSize} tokens\n`);
    if (d.capabilities !== undefined) {
      const on = Object.entries(d.capabilities).filter(([, v]) => v).map(([k]) => k);
      process.stdout.write(`  supports    ${on.join(', ') || 'none reported'}\n`);
    }
    if (d.useCases !== undefined) process.stdout.write(`  use cases   ${d.useCases.join(', ')}\n`);
    if (d.reason !== undefined) process.stdout.write(`  reason      ${d.reason}\n`);
    const c = result.cloud;
    process.stdout.write('\ncloud (private cloud compute)\n');
    process.stdout.write(`  available   ${c.available}\n`);
    process.stdout.write(`  installed   ${c.installed === true}\n`);
    if (c.contextSize !== undefined) process.stdout.write(`  context     ${c.contextSize} tokens\n`);
    if (c.quota !== undefined) {
      const approaching = c.quota.approachingLimit === true ? ' (approaching limit)' : '';
      process.stdout.write(`  quota       ${c.quota.status}${approaching}\n`);
      if (c.quota.resetDate !== undefined) process.stdout.write(`  resets      ${c.quota.resetDate}\n`);
    }
    if (c.reason !== undefined) process.stdout.write(`  reason      ${c.reason}\n`);
    return d.available || c.available ? 0 : 1;
  }

  if (command === 'setup-cloud') {
    await installCloudShortcut((p) => process.stderr.write(`  ${p.status}\n`), {
      force: flags.force,
      webSearch: flags.webSearch,
    });
    return 0;
  }

  if (command === 'count') {
    const arg = flags.positional[0];
    if (arg === undefined) {
      process.stderr.write('count needs a prompt, or "-" to read stdin.\n');
      return 2;
    }
    const prompt = arg === '-' ? await readStdin() : arg;
    const llm = new AppleLLM({ tier: 'device' });
    let tools: ('ocr' | 'barcode' | 'spotlight')[] | undefined;
    try {
      tools = parseTools(flags.tool, usageError) as typeof tools;
    } catch (err) {
      return usageError(err instanceof Error ? err.message : String(err));
    }
    try {
      const { tokens, contextSize } = await llm.countTokens(prompt, {
        system: flags.system,
        images: flags.image.length > 0 ? flags.image.map(parseImageFlag) : undefined,
        tools,
      });
      const pct = contextSize > 0 ? Math.round((tokens / contextSize) * 100) : 0;
      process.stdout.write(`${tokens} tokens of ${contextSize} (${pct}%)\n`);
      return tokens > contextSize ? 1 : 0;
    } finally {
      llm.close();
    }
  }

  if (command === 'history') {
    if (flags.session === undefined) return usageError('history needs --session <id>.');
    const llm = new AppleLLM({ tier: 'device' });
    try {
      const { instructions, history } = await llm.history(flags.session);
      if (flags.json === true) {
        process.stdout.write(`${JSON.stringify({ sessionId: flags.session, instructions, history }, null, 2)}\n`);
        return 0;
      }
      for (const turn of history) {
        process.stdout.write(`${turn.role === 'user' ? 'you' : 'model'}: ${turn.content}\n`);
      }
      return 0;
    } finally {
      llm.close();
    }
  }

  if (command === 'reset') {
    const llm = new AppleLLM({ tier: 'device' });
    try {
      await llm.resetSession(flags.session);
      process.stdout.write(flags.session ? `reset ${flags.session}\n` : 'reset all sessions\n');
      return 0;
    } finally {
      llm.close();
    }
  }

  if (command === 'rewrite' || command === 'proofread' || command === 'summarize' || command === 'draft') {
    const arg = flags.positional[0];
    if (arg === undefined) {
      process.stderr.write(`${command} needs text, or "-" to read stdin.\n`);
      return 2;
    }
    const text = arg === '-' ? await readStdin() : arg;
    const llm = new AppleLLM({
      tier: 'device',
      guardrails: 'permissive',
      onProgress: (p) => process.stderr.write(`  ${p.status}\n`),
    });
    try {
      const out =
        command === 'rewrite'
          ? await llm.rewrite(text, { system: flags.system })
          : command === 'proofread'
            ? await llm.proofread(text, { system: flags.system })
            : command === 'summarize'
              ? await llm.summarize(text, { system: flags.system })
              : await llm.draft(text, { system: flags.system });
      process.stdout.write(`${out}\n`);
      return 0;
    } finally {
      llm.close();
    }
  }

  if (command === 'ask-screen') {
    const arg = flags.positional[0];
    if (arg === undefined) {
      process.stderr.write('ask-screen needs a question, or "-" to read stdin.\n');
      return 2;
    }
    if (flags.mode !== undefined && !['interactive', 'window', 'fullscreen'].includes(flags.mode)) {
      return usageError(`--mode must be interactive, window, or fullscreen (got "${flags.mode}").`);
    }
    const question = arg === '-' ? await readStdin() : arg;
    const maxTokens = flags.maxTokens === undefined ? undefined : Number(flags.maxTokens);
    if (maxTokens !== undefined && Number.isNaN(maxTokens)) {
      return usageError(`--max-tokens must be a number (got "${flags.maxTokens}").`);
    }
    process.stderr.write('  select a screen region (Esc cancels)\n');
    const shot = await captureScreenshot(
      (flags.mode ?? 'interactive') as 'interactive' | 'window' | 'fullscreen',
    );
    const llm = new AppleLLM({
      tier: 'device',
      maxTokens,
      onProgress: (p) => process.stderr.write(`  ${p.status}\n`),
    });
    try {
      const out = await llm.text(question, {
        system: flags.system,
        sessionId: flags.session,
        images: [...flags.image.map(parseImageFlag), shot],
      });
      process.stdout.write(`${out}\n`);
      return 0;
    } finally {
      const { rm } = await import('node:fs/promises');
      await rm(shot, { force: true }).catch(() => undefined);
      llm.close();
    }
  }

  if (command === 'serve') {
    const port = flags.port === undefined ? DEFAULT_PORT : Number(flags.port);
    if (!Number.isInteger(port) || port < 0 || port > 65535) return usageError(`--port must be a port number (got "${flags.port}").`);
    if (flags.tier !== undefined && !['device', 'cloud', 'auto'].includes(flags.tier)) {
      return usageError(`--tier must be device, cloud, or auto (got "${flags.tier}").`);
    }
    let running: Awaited<ReturnType<typeof serve>>;
    try {
      running = await serve({
        port,
        host: flags.host,
        apiKey: flags.apiKey,
        cors: flags.cors,
        tier: flags.tier as 'device' | 'cloud' | 'auto' | undefined,
        fallback,
        log: (line) => process.stderr.write(`  ${line}\n`),
      });
    } catch (err) {
      if ((err as NodeJS.ErrnoException).code !== 'EADDRINUSE') throw err;
      process.stderr.write(
        `Port ${port} is already in use${port === 11434 ? ' — probably by Ollama itself; quit it first' : ''}.\n` +
          'Pick another with --port <n>.\n',
      );
      return 1;
    }
    const root = running.url.replace(/\/v1$/, '');
    process.stderr.write(
      `apple-llm serving Apple's models\n` +
        `  OpenAI: ${running.url}  (base URL for OpenAI SDKs, LangChain, Open WebUI, editor plugins)\n` +
        `  Ollama: ${root}${port === 11434 ? '' : '  (--port 11434 for apps that only look on Ollama\'s port)'}\n` +
        `  models: apple-on-device${flags.tier === 'cloud' ? ' (default: apple-private-cloud)' : ''}, apple-private-cloud\n` +
        `  try:    curl ${running.url}/chat/completions -H 'content-type: application/json' ` +
        `-d '{"model":"apple-on-device","messages":[{"role":"user","content":"Hello"}]}'\n` +
        (flags.host === '0.0.0.0' && flags.apiKey === undefined
          ? '  warning: listening on every interface with no --api-key; anyone on your network can use it\n'
          : ''),
    );
    await new Promise<void>((resolve) => {
      const stop = (): void => {
        void running.close().then(resolve);
      };
      process.once('SIGINT', stop);
      process.once('SIGTERM', stop);
    });
    return 0;
  }

  if (command === 'chat') {
    if (flags.tier !== undefined && !['device', 'cloud', 'auto'].includes(flags.tier)) {
      return usageError(`--tier must be device, cloud, or auto (got "${flags.tier}").`);
    }
    let tools: ('ocr' | 'barcode' | 'spotlight')[] | undefined;
    try {
      tools = parseTools(flags.tool, usageError) as typeof tools;
    } catch (err) {
      return usageError(err instanceof Error ? err.message : String(err));
    }
    const maxTokens = flags.maxTokens === undefined ? undefined : Number(flags.maxTokens);
    const llm = new AppleLLM({
      tier: (flags.tier ?? 'auto') as 'device' | 'cloud' | 'auto',
      fallback,
      maxTokens,
      useCase: flags.useCase as 'general' | 'contentTagging' | undefined,
      guardrails: flags.guardrails as 'default' | 'permissive' | undefined,
    });
    try {
      await runRepl(llm, { system: flags.system, call: { tools } });
    } finally {
      llm.close();
    }
    return 0;
  }

  if (command === 'run') {
    const arg = flags.positional[0];
    if (arg === undefined) {
      process.stderr.write('run needs a prompt, or "-" to read stdin.\n');
      return 2;
    }
    if (flags.tier !== undefined && flags.tier !== 'device' && flags.tier !== 'cloud' && flags.tier !== 'auto') {
      return usageError(`--tier must be device, cloud, or auto (got "${flags.tier}").`);
    }
    const temperature = flags.temperature === undefined ? undefined : Number(flags.temperature);
    if (temperature !== undefined && Number.isNaN(temperature)) {
      return usageError(`--temperature must be a number (got "${flags.temperature}").`);
    }
    const maxTokens = flags.maxTokens === undefined ? undefined : Number(flags.maxTokens);
    if (maxTokens !== undefined && Number.isNaN(maxTokens)) {
      return usageError(`--max-tokens must be a number (got "${flags.maxTokens}").`);
    }
    const timeoutMs = flags.timeout === undefined ? undefined : Number(flags.timeout) * 1000;
    if (timeoutMs !== undefined && (Number.isNaN(timeoutMs) || timeoutMs <= 0)) {
      return usageError(`--timeout must be a number of seconds (got "${flags.timeout}").`);
    }
    const seed = flags.seed === undefined ? undefined : Number(flags.seed);
    if (seed !== undefined && Number.isNaN(seed)) {
      return usageError(`--seed must be a number (got "${flags.seed}").`);
    }
    if (flags.useCase !== undefined && flags.useCase !== 'general' && flags.useCase !== 'contentTagging') {
      return usageError(`--use-case must be general or contentTagging (got "${flags.useCase}").`);
    }
    if (flags.guardrails !== undefined && flags.guardrails !== 'default' && flags.guardrails !== 'permissive') {
      return usageError(`--guardrails must be default or permissive (got "${flags.guardrails}").`);
    }
    const prompt = await promptFrom(arg);
    const tier = (flags.tier ?? 'auto') as 'device' | 'cloud' | 'auto';
    let tools: ('ocr' | 'barcode' | 'spotlight')[] | undefined;
    try {
      tools = parseTools(flags.tool, usageError) as typeof tools;
    } catch (err) {
      return usageError(err instanceof Error ? err.message : String(err));
    }
    const sampling =
      flags.greedy === true
        ? ({ mode: 'greedy' } as const)
        : seed !== undefined
          ? ({ mode: 'topK', k: 50, seed } as const)
          : undefined;
    const llm = new AppleLLM({
      tier,
      fallback,
      temperature,
      maxTokens,
      useCase: flags.useCase as 'general' | 'contentTagging' | undefined,
      guardrails: flags.guardrails as 'default' | 'permissive' | undefined,
      sampling,
      timeoutMs,
      onProgress: (p) => process.stderr.write(`  ${p.status}\n`),
    });
    const images = flags.image.length > 0 ? flags.image.map(parseImageFlag) : undefined;
    const documents = flags.document.length > 0 ? flags.document : undefined;
    try {
      if (flags.schema !== undefined) {
        let schemaText: string;
        try {
          schemaText = await readFile(flags.schema, 'utf8');
        } catch (err) {
          throw new AppleLLMError(
            `Could not read schema file "${flags.schema}": ${err instanceof Error ? err.message : String(err)}`,
          );
        }
        let schema: Record<string, unknown>;
        try {
          schema = JSON.parse(schemaText) as Record<string, unknown>;
        } catch (err) {
          throw new AppleLLMError(
            `Schema file "${flags.schema}" is not valid JSON: ${err instanceof Error ? err.message : String(err)}`,
          );
        }
        const call = {
          schema,
          system: flags.system,
          webSearch: flags.webSearch,
          images,
          documents,
          sessionId: flags.session,
          tools,
        };
        if (flags.stream === true) {
          // Newline-delimited partial objects, then the final one: pipeable.
          const stream = llm.streamJson(prompt, call);
          for await (const partial of stream) process.stdout.write(`${JSON.stringify(partial)}\n`);
          const final = await stream;
          process.stdout.write(`${JSON.stringify(final)}\n`);
        } else if (flags.json === true) {
          process.stdout.write(`${JSON.stringify(await llm.generate(prompt, call), null, 2)}\n`);
        } else {
          process.stdout.write(`${JSON.stringify(await llm.json(prompt, call), null, 2)}\n`);
        }
      } else if (flags.stream === true) {
        const out = await llm.stream(prompt, {
          system: flags.system,
          images,
          documents,
          sessionId: flags.session,
          tools,
          onDelta: (delta) => process.stdout.write(delta),
        });
        process.stdout.write('\n');
        void out;
      } else {
        const call = {
          system: flags.system,
          webSearch: flags.webSearch,
          images,
          documents,
          sessionId: flags.session,
          tools,
        };
        if (flags.json === true) {
          process.stdout.write(`${JSON.stringify(await llm.generate(prompt, call), null, 2)}\n`);
        } else {
          process.stdout.write(`${await llm.text(prompt, call)}\n`);
        }
      }
    } finally {
      llm.close();
    }
    return 0;
  }

  process.stderr.write(`Unknown command "${command}".\n\n${USAGE}`);
  return 2;
}

main()
  .then((code) => {
    process.exitCode = code;
  })
  .catch((err: unknown) => {
    if (err instanceof AppleLLMError) {
      process.stderr.write(`${err.name}: ${err.message}\n`);
    } else {
      process.stderr.write(`${err instanceof Error ? err.stack ?? err.message : String(err)}\n`);
    }
    process.exitCode = 1;
  });
