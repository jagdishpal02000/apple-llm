import { execFile } from 'node:child_process';
import { readFileSync, readdirSync } from 'node:fs';
import { rm } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { promisify } from 'node:util';
import { afterAll, beforeAll, describe, expect, it } from 'vitest';
import { AppleLLM, probe } from '../src/index.js';
import { DeviceClient, probeDevice } from '../src/device.js';
import { cacheDir, ensureBinary, fingerprint, helperSource } from '../src/compile.js';
import { hostTarget } from '../src/target.js';
import { toAppleSchema } from '../src/schema.js';

const execFileAsync = promisify(execFile);

/**
 * These need the hardware. They are gated on an env var *and* a runtime probe,
 * so a machine without Apple Intelligence skips cleanly rather than failing.
 *
 *   npm run test:live
 */
const LIVE = process.env.APPLE_LLM_LIVE === '1';
/** Cloud calls burn a shared quota, so they need their own opt-in. */
const LIVE_CLOUD = process.env.APPLE_LLM_LIVE_CLOUD === '1';

const here = path.dirname(fileURLToPath(import.meta.url));
const FIXTURES = path.resolve(here, '..', '..', '..', 'tests', 'fixtures', 'schema');

let deviceAvailable = false;
beforeAll(async () => {
  if (!LIVE) return;
  deviceAvailable = (await probeDevice()).available;
});

const onDevice = LIVE ? describe : describe.skip;

onDevice('live: on-device', () => {
  const client = new DeviceClient();
  const gate = (): boolean => !deviceAvailable;

  it('probes as available with a context window and a variant', async () => {
    if (gate()) return;
    const result = await probe();
    expect(result.device.available).toBe(true);
    expect(result.device.contextSize).toBeGreaterThanOrEqual(4096);
    expect(typeof result.device.variant).toBe('string');
  });

  it('returns free text', async () => {
    if (gate()) return;
    const out = await client.text('Name exactly one primary colour. Reply with the word only.', {
      system: 'You are terse.',
      maxTokens: 30,
    });
    expect(out.trim().length).toBeGreaterThan(0);
    expect(out.length).toBeLessThan(200);
  });

  it('honours the schema, not merely the request', async () => {
    if (gate()) return;
    // Constrained decoding makes this a guarantee. If it ever fails, the schema
    // transform is wrong — the model cannot disobey a schema it was given.
    const out = (await client.json('Ada Lovelace, born 1815, worked in mathematics.', {
      schema: {
        type: 'object',
        required: ['name', 'born', 'field'],
        properties: {
          name: { type: 'string' },
          born: { type: 'integer' },
          field: { enum: ['mathematics', 'physics', 'poetry'] },
        },
      },
      maxTokens: 200,
    })) as Record<string, unknown>;

    expect(typeof out.name).toBe('string');
    expect(typeof out.born).toBe('number');
    expect(Number.isInteger(out.born)).toBe(true);
    expect(['mathematics', 'physics', 'poetry']).toContain(out.field);
    expect(Object.keys(out).sort()).toEqual(['born', 'field', 'name']);
  });

  it('accepts every schema in the shared corpus', async () => {
    if (gate()) return;
    // The strongest check available: Apple's own GenerationSchema decoder is the
    // authority on the dialect, so the corpus is validated against it and not
    // just against this implementation.
    const files = readdirSync(FIXTURES).filter((f) => f.endsWith('.json')).sort();
    const rejected: string[] = [];
    for (const file of files) {
      const fx = JSON.parse(readFileSync(path.join(FIXTURES, file), 'utf8')) as {
        name: string;
        input: Record<string, unknown>;
      };
      try {
        await client.complete({
          prompt: 'Produce a minimal example.',
          schema: toAppleSchema(fx.input),
          maxTokens: 120,
        });
      } catch (err) {
        rejected.push(`${fx.name}: ${err instanceof Error ? err.message.slice(0, 120) : String(err)}`);
      }
    }
    expect(rejected).toEqual([]);
  }, 600_000);

  it('keeps the median call under 4s', async () => {
    if (gate()) return;
    // This is the regression guard for the session-reuse trap. A fresh process
    // per request measured ~17s a call against ~1.5s once resident, and both
    // failure modes produce *correct* output — only slower — so nothing else
    // would catch it. Medians, never a single run: wall-clock swings.
    const latencies: number[] = [];
    for (let i = 0; i < 5; i += 1) {
      const started = Date.now();
      await client.text(`Name one primary colour. Attempt ${i}.`, { maxTokens: 20 });
      latencies.push(Date.now() - started);
    }
    const sorted = [...latencies].sort((a, b) => a - b);
    const median = sorted[Math.floor(sorted.length / 2)];
    // eslint-disable-next-line no-console
    console.log(`  median ${median}ms of ${JSON.stringify(latencies)}`);
    expect(median).toBeLessThan(4000);
  }, 300_000);

  it('reports a schema Apple rejects as a SchemaRejectedError', async () => {
    if (gate()) return;
    await expect(
      // Raw JSON Schema, not run through toAppleSchema: no title, no x-order.
      client.complete({ prompt: 'hi', schema: { type: 'object', properties: { a: { type: 'string' } } } }),
    ).rejects.toThrow(/schema/i);
  });

  it('closes without leaving its helper running', async () => {
    if (gate()) return;
    // Other clients in this file legitimately hold a helper open, so compare
    // pids rather than asserting none is running at all.
    const pids = async (): Promise<Set<string>> => {
      const { stdout } = await execFileAsync('bash', ['-c', 'pgrep -f "fm-helper-.* --serve" || true']);
      return new Set(stdout.split('\n').map((l) => l.trim()).filter(Boolean));
    };
    const before = await pids();
    const own = new DeviceClient();
    await own.text('hi', { maxTokens: 10 });
    const during = await pids();
    const started = [...during].filter((p) => !before.has(p));
    expect(started.length).toBeGreaterThan(0);

    own.close();
    await new Promise((r) => setTimeout(r, 500));
    const after = await pids();
    expect(started.filter((p) => after.has(p))).toEqual([]);
  });
});

const onCompile = LIVE ? describe : describe.skip;

onCompile('live: compile from a cold cache', () => {
  it('rebuilds the helper when the cached binary is gone', async () => {
    if (!deviceAvailable) return;
    const source = await helperSource();
    const triple = (await hostTarget()) ?? process.arch;
    const target = path.join(cacheDir(), `fm-helper-${fingerprint(source, triple)}`);
    await rm(target, { force: true });

    const progress: string[] = [];
    const rebuilt = await ensureBinary((p) => progress.push(p.status), { force: true });
    expect(rebuilt).toBe(target);
    // The one-time compile takes seconds and must be reported, not look like a hang.
    expect(progress.join(' ')).toMatch(/building/i);
    const { stdout } = await execFileAsync(rebuilt, ['--probe'], { timeout: 30_000 });
    expect(JSON.parse(stdout.trim()).available).toBe(true);
  }, 300_000);
});

const onCloud = LIVE && LIVE_CLOUD ? describe : describe.skip;

onCloud('live: private cloud compute (consumes quota)', () => {
  it('completes one round trip', async () => {
    const result = await probe();
    if (!result.cloud.available) return;
    const llm = new AppleLLM({ tier: 'cloud' });
    try {
      const out = await llm.text('Reply with the single word: pong');
      expect(out.toLowerCase()).toContain('pong');
    } finally {
      llm.close();
    }
  }, 180_000);
});

const onExtras = LIVE ? describe : describe.skip;

/**
 * Capabilities macOS 27 added on top of what api-scribe used. Each was
 * confirmed against the framework before being wired up; see the README.
 */
onExtras('live: macOS 27 capabilities', () => {
  const client = new DeviceClient();

  it('reports what the model can actually do', async () => {
    if (!deviceAvailable) return;
    const { device } = await probe();
    // Absent on macOS 26 — the framework only reports capabilities from 27.
    if (device.capabilities === undefined) return;
    expect(device.capabilities.guidedGeneration).toBe(true);
    expect(typeof device.capabilities.vision).toBe('boolean');
    expect(typeof device.capabilities.reasoning).toBe('boolean');
    expect(device.useCases).toContain('contentTagging');
  });

  it('reads the Private Cloud Compute quota without calling it', async () => {
    if (!deviceAvailable) return;
    const { cloud } = await probe();
    if (cloud.quota === undefined) return;
    // The entitlement that blocks PCC inference does not block reading this.
    expect(['belowLimit', 'limitReached', 'unknown']).toContain(cloud.quota.status);
    expect(typeof cloud.quota.isAvailable).toBe('boolean');
  });

  it('counts tokens before sending, against the real context window', async () => {
    if (!deviceAvailable) return;
    const short = await client.countTokens('The quick brown fox jumps over the lazy dog.');
    expect(short.tokens).toBeGreaterThan(5);
    expect(short.tokens).toBeLessThan(20);
    expect(short.contextSize).toBeGreaterThanOrEqual(4096);

    // Instructions share the window, so they have to be counted too.
    const withSystem = await client.countTokens('The quick brown fox jumps over the lazy dog.', {
      system: 'You are a helpful assistant that always answers very briefly indeed.',
    });
    expect(withSystem.tokens).toBeGreaterThan(short.tokens);

    const long = await client.countTokens('word '.repeat(3000));
    expect(long.tokens).toBeGreaterThan(2000);
  }, 120_000);

  it('prewarms without error and leaves the model usable', async () => {
    if (!deviceAvailable) return;
    await client.prewarm('You are terse.');
    const out = await client.text('Name one primary colour. One word.', {
      system: 'You are terse.',
      maxTokens: 20,
    });
    expect(out.trim().length).toBeGreaterThan(0);
  }, 120_000);

  it('makes seeded sampling reproducible', async () => {
    if (!deviceAvailable) return;
    // This is the answer to the temperature-0 trap: determinism without greedy
    // decoding's degeneration. It works only because sessions are fresh per
    // request — reusing one changes the transcript and with it the output.
    const runs: string[] = [];
    for (let i = 0; i < 3; i += 1) {
      runs.push(
        await client.text('Invent a name for a coffee shop.', {
          system: 'You write one short sentence.',
          temperature: 0.9,
          maxTokens: 40,
          sampling: { mode: 'topK', k: 50, seed: 42 },
        }),
      );
    }
    expect(new Set(runs).size).toBe(1);
  }, 180_000);

  it('keeps seeded sampling reproducible across a prewarm', async () => {
    if (!deviceAvailable) return;
    // Regression: prewarming used to leave a warmed session that the next call
    // reused, so the first call after a prewarm ran against different session
    // state than the ones after it — identical seeds, different text, no error.
    const fresh = new DeviceClient();
    try {
      await fresh.prewarm();
      const runs: string[] = [];
      for (let i = 0; i < 3; i += 1) {
        runs.push(
          await fresh.text('Invent a name for a coffee shop.', {
            temperature: 0.9,
            maxTokens: 30,
            sampling: { mode: 'topK', k: 50, seed: 7 },
          }),
        );
      }
      expect(new Set(runs).size).toBe(1);
    } finally {
      fresh.close();
    }
  }, 180_000);

  it('describes an image', async () => {
    if (!deviceAvailable) return;
    const { device } = await probe();
    if (device.capabilities?.vision !== true) return;
    const image = path.resolve(here, '..', '..', '..', 'tests', 'fixtures', 'images', 'red-square-blue-circle.png');
    const out = await client.text('What shape and colours are in this image? One sentence.', {
      system: 'You describe images literally and briefly.',
      images: [image],
      maxTokens: 80,
    });
    expect(out.toLowerCase()).toMatch(/blue/);
    expect(out.toLowerCase()).toMatch(/circle|round/);
  }, 120_000);

  it('rejects a missing image rather than silently ignoring it', async () => {
    if (!deviceAvailable) return;
    await expect(
      client.text('describe', { images: ['/definitely/not/here.png'] }),
    ).rejects.toThrow(/image not found/i);
  });

  it('tags with the contentTagging use case', async () => {
    if (!deviceAvailable) return;
    const tagger = new DeviceClient({ useCase: 'contentTagging' });
    try {
      const out = (await tagger.json(
        'A recipe for sourdough bread using a rye starter and a cast iron dutch oven.',
        {
          schema: {
            type: 'object',
            required: ['tags'],
            properties: { tags: { type: 'array', maxItems: 5, items: { type: 'string' } } },
          },
          maxTokens: 80,
        },
      )) as { tags: string[] };
      expect(Array.isArray(out.tags)).toBe(true);
      expect(out.tags.length).toBeGreaterThan(0);
      expect(out.tags.join(' ').toLowerCase()).toMatch(/sourdough|bread|rye/);
    } finally {
      tagger.close();
    }
  }, 120_000);

  it('accepts permissive guardrails for rewriting', async () => {
    if (!deviceAvailable) return;
    const permissive = new DeviceClient({ guardrails: 'permissive' });
    try {
      const out = await permissive.text('Rewrite more formally: gonna grab a bite', {
        system: 'You rewrite text.',
        maxTokens: 50,
      });
      expect(out.trim().length).toBeGreaterThan(0);
    } finally {
      permissive.close();
    }
  }, 120_000);

  it('streams text with deltas that concatenate to the content', async () => {
    if (!deviceAvailable) return;
    const deltas: string[] = [];
    const out = await client.stream('Count to three, one per line.', {
      system: 'You are terse.',
      maxTokens: 40,
      onDelta: (d) => deltas.push(d),
    });
    expect(out.trim().length).toBeGreaterThan(0);
    expect(deltas.join('')).toBe(out);
  }, 120_000);

  it('holds a named conversation across turns, then resets it', async () => {
    if (!deviceAvailable) return;
    const sessionId = `live-node-${Date.now()}`;
    try {
      await client.text('My cat is called Biscuit. Reply with one word.', {
        system: 'You are terse and you remember names.',
        sessionId,
        maxTokens: 30,
      });
      const recall = await client.text('What is my cat called? One word.', {
        system: 'You are terse and you remember names.',
        sessionId,
        maxTokens: 30,
      });
      expect(recall.toLowerCase()).toContain('biscuit');

      const { history } = await client.history(sessionId);
      expect(history.length).toBeGreaterThanOrEqual(4);
      expect(history[0].role).toBe('user');
    } finally {
      await client.resetSession(sessionId);
    }
    const { history } = await client.history(sessionId);
    expect(history).toEqual([]);
  }, 180_000);

  it('accepts a labelled image attachment', async () => {
    if (!deviceAvailable) return;
    const { device } = await probe();
    if (device.capabilities?.vision !== true) return;
    const image = path.resolve(here, '..', '..', '..', 'tests', 'fixtures', 'images', 'red-square-blue-circle.png');
    const out = await client.text('What is in the image labelled chart? One sentence.', {
      system: 'You describe images literally and briefly.',
      images: [{ path: image, label: 'chart' }],
      maxTokens: 80,
    });
    expect(out.toLowerCase()).toMatch(/blue/);
  }, 120_000);

  it('uses the built-in ocr tool on an image', async () => {
    if (!deviceAvailable) return;
    const { device } = await probe();
    if (device.capabilities?.toolCalling !== true) return;
    const image = path.resolve(here, '..', '..', '..', 'tests', 'fixtures', 'images', 'red-square-blue-circle.png');
    const out = await client.text('Read any text in this image. If there is none, say "none".', {
      system: 'You read text in images.',
      images: [image],
      tools: ['ocr'],
      maxTokens: 80,
    });
    expect(out.trim().length).toBeGreaterThan(0);
  }, 120_000);

  it('rewrites through the Write-with-Siri preset', async () => {
    if (!deviceAvailable) return;
    const llm = new AppleLLM({ tier: 'device' });
    try {
      const out = await llm.rewrite('gonna grab a bite', { instruction: 'Make it formal.' });
      expect(out.trim().length).toBeGreaterThan(0);
    } finally {
      llm.close();
    }
  }, 120_000);

  afterAll(() => client.close());
});
