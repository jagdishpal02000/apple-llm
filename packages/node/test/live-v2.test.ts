/**
 * The 0.2 API against the real model. Gated like live.test.ts: an env var
 * *and* a runtime probe, so a machine without Apple Intelligence skips.
 *
 *   APPLE_LLM_LIVE=1 npx vitest run test/live-v2.test.ts
 */
import { execFile } from 'node:child_process';
import { mkdtemp, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { promisify } from 'node:util';
import { afterAll, beforeAll, describe, expect, it } from 'vitest';
import { z } from 'zod';
import { AbortError, AppleLLM, TimeoutError, probeDevice, tool, type ChatMessage } from '../src/index.js';

const execFileAsync = promisify(execFile);
const LIVE = process.env.APPLE_LLM_LIVE === '1';
const here = path.dirname(fileURLToPath(import.meta.url));

let available = false;
beforeAll(async () => {
  if (LIVE) available = (await probeDevice()).available;
});

const live = LIVE ? describe : describe.skip;

live('live: 0.2 API', () => {
  const llm = new AppleLLM({ tier: 'device', system: 'You are terse.' });
  afterAll(() => llm.close());
  const gate = (): boolean => !available;

  it('runs a JS function tool with typed, validated arguments', async () => {
    if (gate()) return;
    const cities: string[] = [];
    const result = await llm.generate('What is the weather in Paris right now?', {
      tools: [
        tool({
          name: 'getWeather',
          description: 'Get the current weather for a city.',
          parameters: z.object({ city: z.string().describe('City name') }),
          execute: async ({ city }) => {
            cities.push(city);
            return { tempC: 21, sky: 'sunny' };
          },
        }),
      ],
    });
    expect(cities.map((c) => c.toLowerCase())).toContain('paris');
    expect(result.toolCalls[0]).toMatchObject({ name: 'getWeather', output: '{"tempC":21,"sky":"sunny"}' });
    expect(result.text).toMatch(/21/);
    expect(result.message.toolCalls?.[0]?.name).toBe('getWeather');
  });

  it('drives a client-side tool loop through message lists', async () => {
    if (gate()) return;
    const getWeather = tool({
      name: 'getWeather',
      description: 'Get the current weather for a city.',
      parameters: { type: 'object', properties: { city: { type: 'string' } }, required: ['city'] },
    });
    const messages: ChatMessage[] = [{ role: 'user', content: 'What is the weather in Tokyo?' }];
    const first = await llm.generate(messages, { tools: [getWeather] });
    expect(first.finishReason).toBe('tool-calls');
    const call = first.toolCalls[0];
    expect(call.name).toBe('getWeather');

    messages.push(first.message, { role: 'tool', toolCallId: call.id, content: '{"tempC": 14, "sky": "heavy rain"}' });
    const second = await llm.generate(messages, { tools: [getWeather] });
    expect(second.finishReason).toBe('stop');
    expect(second.text.toLowerCase()).toMatch(/14|rain/);
  });

  it('streams with for await, and a break cancels the generation', async () => {
    if (gate()) return;
    const deltas: string[] = [];
    const started = Date.now();
    const stream = llm.stream('Write a long essay about the history of bridges.', { maxTokens: 1500 });
    for await (const delta of stream) {
      deltas.push(delta);
      if (deltas.length >= 2) break;
    }
    await expect(stream.result).rejects.toBeInstanceOf(AbortError);
    // The helper is free again quickly, not after 1500 tokens of essay.
    const text = await llm.text('Reply with the word OK.', { maxTokens: 10 });
    expect(text.length).toBeGreaterThan(0);
    expect(Date.now() - started).toBeLessThan(15_000);
  });

  it('aborts a running generation with an AbortSignal', async () => {
    if (gate()) return;
    const controller = new AbortController();
    const pending = llm.text('Write a long essay about the history of bridges.', {
      maxTokens: 1500,
      signal: controller.signal,
    });
    setTimeout(() => controller.abort(), 300);
    const started = Date.now();
    await expect(pending).rejects.toBeInstanceOf(AbortError);
    expect(Date.now() - started).toBeLessThan(3000);
    expect((await llm.text('Say OK.', { maxTokens: 10 })).length).toBeGreaterThan(0);
  });

  it('times out with timeoutMs as a TimeoutError', async () => {
    if (gate()) return;
    await expect(
      llm.text('Write a long essay about the history of bridges.', { maxTokens: 1500, timeoutMs: 400 }),
    ).rejects.toBeInstanceOf(TimeoutError);
  });

  it('streams partial objects, then resolves the validated object', async () => {
    if (gate()) return;
    const Person = z.object({ name: z.string(), born: z.number().int(), skills: z.array(z.string()).max(3) });
    const partials: unknown[] = [];
    const stream = llm.streamJson('Ada Lovelace, born 1815, mathematician and writer.', { schema: Person });
    for await (const partial of stream) partials.push(partial);
    const person = await stream;
    expect(partials.length).toBeGreaterThan(0);
    expect(person.born).toBe(1815);
    expect(person.name).toMatch(/Ada/);
  });

  it('types json() from a Zod schema and restores nulls and defaults', async () => {
    if (gate()) return;
    const Contact = z.object({
      name: z.string(),
      phone: z.string().nullable().describe('Phone number, or null when none is given'),
      priority: z.number().int().default(3),
    });
    const contact = await llm.json('Contact: Grace Hopper. No phone number given.', { schema: Contact });
    expect(contact.name).toMatch(/Grace/);
    expect(contact).toHaveProperty('phone');
    expect(typeof contact.priority).toBe('number');
  });

  it('accepts Zod string formats, whose patterns Apple cannot decode', async () => {
    if (gate()) return;
    // z.string().email() carries a regex `pattern`, which Apple rejects outright
    // (dialect rule 9). It is stripped for the decoder and enforced by Zod.
    const Signup = z.object({ email: z.string().email(), id: z.string().uuid().nullable() });
    const signup = await llm.json('Sign up grace@navy.mil. No id was assigned yet.', { schema: Signup });
    expect(signup.email).toBe('grace@navy.mil');
  });

  it('continues a conversation from a message list', async () => {
    if (gate()) return;
    const reply = await llm.text([
      { role: 'user', content: 'My cat is called Biscuit.' },
      { role: 'assistant', content: 'Lovely name!' },
      { role: 'user', content: 'What is my cat called?' },
    ]);
    expect(reply).toMatch(/Biscuit/);
  });

  it('reports usage and a length finish', async () => {
    if (gate()) return;
    const cut = await llm.generate('Describe the ocean in detail.', { maxTokens: 10 });
    expect(cut.finishReason).toBe('length');
    expect(cut.usage?.outputTokens).toBe(10);
    expect(cut.usage?.inputTokens).toBeGreaterThan(0);
  });

  it('counts history and schema tokens too', async () => {
    if (gate()) return;
    const bare = await llm.countTokens('What is my cat called?');
    const full = await llm.countTokens(
      [
        { role: 'user', content: 'My cat is called Biscuit and she is nine years old.' },
        { role: 'assistant', content: 'Lovely!' },
        { role: 'user', content: 'What is my cat called?' },
      ],
      { schema: z.object({ name: z.string() }) },
    );
    expect(full.tokens).toBeGreaterThan(bare.tokens);
    expect(full.contextSize).toBe(bare.contextSize);
  });

  it('summarises text longer than the context window, in parts', async () => {
    if (gate()) return;
    const section = (n: number): string =>
      `Section ${n}. The committee met to review bridge ${n}. ` +
      'Inspectors described corrosion, traffic loads, drainage and repainting schedules at length. '.repeat(40);
    const long = Array.from({ length: 24 }, (_, i) => section(i + 1)).join('\n\n');
    const { tokens, contextSize } = await llm.countTokens(long);
    expect(tokens).toBeGreaterThan(contextSize);
    const summary = await llm.summarize(long);
    expect(summary.length).toBeGreaterThan(20);
    expect(summary.length).toBeLessThan(long.length / 10);
  }, 300_000);

  it('lets a script exit without close()', async () => {
    if (gate()) return;
    // The helper is unref'd between calls. A script that forgets close() used
    // to hang forever on the idle child process.
    const dir = await mkdtemp(path.join(tmpdir(), 'apple-llm-exit-'));
    const script = path.join(dir, 'no-close.mts');
    await writeFile(
      script,
      `import { AppleLLM } from ${JSON.stringify(path.join(here, '..', 'src', 'index.ts'))};
       const llm = new AppleLLM({ tier: 'device' });
       console.log(await llm.text('Say OK.', { maxTokens: 10 }));`,
    );
    const started = Date.now();
    const { stdout } = await execFileAsync('npx', ['tsx', script], { timeout: 60_000, cwd: path.join(here, '..') });
    expect(stdout.trim().length).toBeGreaterThan(0);
    expect(Date.now() - started).toBeLessThan(45_000);
  }, 90_000);
});

live('live: the Golden Gate server model, described without calling it', () => {
  it('reports its capabilities and context from the framework', async () => {
    const { probe } = await import('../src/index.js');
    const { device, cloud } = await probe();
    if (!device.available || device.capabilities === undefined) return; // needs macOS 27
    expect(cloud.capabilities?.vision).toBe(true);
    expect(cloud.capabilities?.reasoning).toBe(true);
    expect(cloud.contextSize).toBeGreaterThanOrEqual(32_768);
    expect(cloud.quota?.status).toMatch(/belowLimit|limitReached/);
  });
});

/** These spend Private Cloud Compute quota and send prompts off the machine. */
const liveCloud = LIVE && process.env.APPLE_LLM_LIVE_CLOUD === '1' ? describe : describe.skip;

liveCloud('live: cloud tier on the Golden Gate server model', () => {
  const cloud = new AppleLLM({ tier: 'cloud' });
  let ready = false;
  beforeAll(async () => {
    try {
      await cloud.ensureReady();
      ready = true;
    } catch {
      ready = false; // shortcut not installed: `apple-llm setup-cloud`
    }
  });

  it('reads an image', async () => {
    if (!ready) return;
    const image = path.join(here, '..', '..', '..', 'tests', 'fixtures', 'images', 'red-square-blue-circle.png');
    const reply = await cloud.text('Start your reply with PINEAPPLE, then name the colour of the circle.', { images: [image] });
    expect(reply).toMatch(/PINEAPPLE/);
    expect(reply.toLowerCase()).toMatch(/blue/);
  });

  it('reasons through a puzzle the on-device model loses track of', async () => {
    if (!ready) return;
    const reply = await cloud.text(
      'Three friends, Ana, Ben and Cy, each own exactly one pet: a cat, a dog or a fish, and each lives in a red, green or blue house. ' +
        'The dog owner lives in the red house. Ana does not own the cat. Ben lives in the blue house. Cy does not live in the red house and does not own the fish. ' +
        'Who owns the fish, and what colour is their house? Reason step by step, then end with: ANSWER: <name>, <colour>',
    );
    expect(reply).toMatch(/ANSWER:\s*Ben,\s*blue/i);
  });

  it('returns typed JSON validated by Zod', async () => {
    if (!ready) return;
    const person = await cloud.json('Ada Lovelace, born 1815, mathematician. Email ada@analytical.org.', {
      schema: z.object({ name: z.string(), born: z.number().int(), email: z.string().email() }),
    });
    expect(person.born).toBe(1815);
    expect(person.email).toBe('ada@analytical.org');
  });

  it('answers about current events with web search', async () => {
    if (!ready) return;
    const reply = await cloud
      .text('Who won the 2026 FIFA World Cup final? Answer with the country only.', { webSearch: true })
      .catch((error: Error) => (/setup-cloud --web-search/.test(error.message) ? 'skipped' : Promise.reject(error)));
    if (reply === 'skipped') return;
    expect(reply).toMatch(/Spain/i);
  });
});
