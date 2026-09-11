#!/usr/bin/env node
/**
 * apple-llm CLI — how people debug this, and how it is tested by hand.
 *
 *   apple-llm probe
 *   apple-llm setup-cloud [--web-search] [--force]
 *   apple-llm run --tier device --system "..." -      # prompt on stdin
 *   apple-llm run --tier cloud --schema s.json "Extract the fields"
 */
import { readFile } from 'node:fs/promises';
import { AppleLLM, captureScreenshot, parseImageFlag, probe } from './index.js';
import { installCloudShortcut } from './cloud.js';
import { AppleLLMError } from './errors.js';

interface Flags {
  tier?: string;
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

  apple-llm probe                        what this machine can do
  apple-llm setup-cloud [--web-search]   install the Shortcut the cloud tier needs
  apple-llm run [options] <prompt|->     one completion ("-" reads stdin)
  apple-llm count [options] <prompt|->   tokens this prompt costs, before sending
  apple-llm ask-screen [options] <question|->   screenshot, then ask (device tier)
  apple-llm history --session <id>       show a conversation's mirrored turns
  apple-llm reset [--session <id>]       drop one conversation, or all
  apple-llm rewrite|proofread|summarize|draft <text|->   Write with Siri presets

run options:
  --tier device|cloud|auto   default: auto
  --system <text>            system instructions
  --schema <file.json>       ask for JSON matching this JSON Schema
  --temperature <n>          default 0.4 (0 degenerates; see the README)
  --max-tokens <n>
  --image <path[::label]>    attach an image, optionally labelled (repeatable; macOS 27+)
  --document <path>          inline a text document into the prompt (repeatable)
  --tool ocr|barcode|spotlight   built-in on-device tool (repeatable)
  --session <id>             continue a named conversation across calls
  --stream                   print partials as they arrive (device text only)
  --use-case contentTagging  Apple's tagging-specialised model
  --guardrails permissive    relax guardrails for rewriting tasks
  --seed <n>                 reproducible sampling (top-k, seeded)
  --greedy                   greedy decoding (deterministic, but degenerates)
  --web-search               cloud tier only

ask-screen options: --mode interactive|window|fullscreen (default interactive),
  plus --system, --session, --max-tokens.
`;

async function main(): Promise<number> {
  const [command, ...rest] = process.argv.slice(2);
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
    const prompt = arg === '-' ? await readStdin() : arg;
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
      temperature,
      maxTokens,
      useCase: flags.useCase as 'general' | 'contentTagging' | undefined,
      guardrails: flags.guardrails as 'default' | 'permissive' | undefined,
      sampling,
      onProgress: (p) => process.stderr.write(`  ${p.status}\n`),
    });
    const images = flags.image.length > 0 ? flags.image.map(parseImageFlag) : undefined;
    const documents = flags.document.length > 0 ? flags.document : undefined;
    try {
      if (flags.schema !== undefined) {
        if (flags.stream === true) {
          return usageError('--stream is text only; schemas need a complete response.');
        }
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
        const out = await llm.json(prompt, {
          schema,
          system: flags.system,
          webSearch: flags.webSearch,
          images,
          documents,
          sessionId: flags.session,
          tools,
        });
        process.stdout.write(`${JSON.stringify(out, null, 2)}\n`);
      } else if (flags.stream === true) {
        if (tier === 'cloud') {
          return usageError('--stream needs the on-device tier.');
        }
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
        const out = await llm.text(prompt, {
          system: flags.system,
          webSearch: flags.webSearch,
          images,
          documents,
          sessionId: flags.session,
          tools,
        });
        process.stdout.write(`${out}\n`);
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
