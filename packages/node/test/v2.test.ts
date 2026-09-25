/**
 * Protocol v2 and the 0.2 API, without Apple hardware: a scripted helper
 * stands in for the Swift side, so every branch of the event framing, the tool
 * loop, cancellation and schema interop is exercised on any machine.
 */
import { describe, expect, it } from 'vitest';
import { z } from 'zod';
import { DeviceClient } from '../src/device.js';
import {
  AbortError,
  SchemaValidationError,
  TimeoutError,
  ToolExecutionError,
  UnsupportedError,
  ContextLengthError,
} from '../src/errors.js';
import { ReplayBook, canonicalJson, renderHistory, splitMessages } from '../src/messages.js';
import { toAppleSchema } from '../src/schema.js';
import { isStandardSchema, resolveSchema, restoreNulls, validateWith } from '../src/standard-schema.js';
import { ResultStream } from '../src/stream.js';
import { normalizeTools, tool, toolOutputText } from '../src/tools.js';
import { fakeServer, tick, type FakeChild } from './fake-helper.js';

const done = (content: string, extra: Record<string, unknown> = {}) => ({ ok: true, content, ...extra });

describe('HelperServer, protocol v2', () => {
  it('routes done:false lines to the request as events, then resolves on the final line', async () => {
    const { server } = fakeServer((_m, child) => {
      child.reply({ ok: true, delta: 'He', done: false });
      child.reply({ ok: true, partial: '{"a":1}', done: false });
      child.reply(done('Hello', { done: true }));
    });
    const events: unknown[] = [];
    const raw = await server.request('{}', { onEvent: (e) => events.push(e) });
    expect(events).toHaveLength(2);
    expect(JSON.parse(raw).content).toBe('Hello');
    server.stop();
  });

  it('treats a line without done:false as final, even mid-"stream"', async () => {
    // Version-1 streaming treated any ok line without done as a delta; v2 keys
    // on done === false exactly, so an error envelope always ends the request.
    const { server } = fakeServer((_m, child) => child.reply({ ok: false, kind: 'unsupported', error: 'no' }));
    const raw = await server.stream('{}', () => undefined);
    expect(JSON.parse(raw).ok).toBe(false);
    server.stop();
  });

  it('lets an event handler write control lines back to the helper', async () => {
    const { server, children } = fakeServer((m, child) => {
      if (m.op === 'toolResult') child.reply(done(`got ${m.output}`));
      else child.reply({ ok: true, toolCall: { id: 'c1', name: 't', arguments: {} }, done: false });
    });
    const raw = await server.request('{"op":"generate"}', {
      onEvent: (event, write) => write({ op: 'toolResult', callId: 'c1', output: 'x' }),
    });
    expect(JSON.parse(raw).content).toBe('got x');
    expect(children[0].controls('toolResult')).toEqual([{ op: 'toolResult', callId: 'c1', output: 'x' }]);
    server.stop();
  });

  it('never writes a request that was aborted while queued', async () => {
    let release: () => void = () => undefined;
    const { server, children } = fakeServer((m, child) => {
      if (m.n === 1) release = () => child.reply(done('first'));
      else child.reply(done('later'));
    });
    const first = server.request('{"n":1}');
    const controller = new AbortController();
    const second = server.request('{"n":2}', { signal: controller.signal });
    await tick();
    controller.abort();
    await expect(second).rejects.toBeInstanceOf(AbortError);
    release();
    expect(JSON.parse(await first).content).toBe('first');
    expect(children[0].requests.map((r) => r.n)).toEqual([1]);
    server.stop();
  });

  it('cancels a running request in the helper and keeps replies paired', async () => {
    // The helper finishes the cancelled request with a line of its own; that
    // line must be consumed by the aborted request, not handed to the next one.
    const { server, children } = fakeServer((m, child) => {
      if (m.op === 'cancel') {
        setTimeout(() => child.reply({ ok: false, kind: 'cancelled', error: 'cancelled' }), 20);
      } else if (m.n === 2) {
        child.reply(done('second'));
      }
    });
    const controller = new AbortController();
    const first = server.request('{"n":1,"id":"a"}', { signal: controller.signal, id: 'a' });
    const second = server.request('{"n":2}');
    await tick();
    controller.abort();
    await expect(first).rejects.toBeInstanceOf(AbortError);
    expect(JSON.parse(await second).content).toBe('second');
    expect(children[0].controls('cancel')).toEqual([{ op: 'cancel', id: 'a' }]);
    // The second request was only written after the helper answered the first.
    expect(children[0].requests.map((r) => r.n)).toEqual([1, 2]);
    server.stop();
  });

  it('maps an AbortSignal.timeout to a TimeoutError', async () => {
    const { server } = fakeServer(() => undefined, 5_000);
    await expect(server.request('{}', { signal: AbortSignal.timeout(20), id: 'x' })).rejects.toBeInstanceOf(TimeoutError);
    server.stop();
  });

  it('pauses the silence timer while the client runs a tool', async () => {
    const { server } = fakeServer((m, child) => {
      if (m.op === 'toolResult') child.reply(done('ok'));
      else child.reply({ ok: true, toolCall: { id: 'c', name: 't', arguments: {} }, done: false });
    }, 50);
    const raw = await server.request('{}', {
      onEvent: (_e, write) => {
        server.pause();
        // A tool slower than the silence timeout must not be mistaken for a wedge.
        setTimeout(() => {
          server.resume();
          write({ op: 'toolResult', callId: 'c', output: 'late' });
        }, 150);
      },
    });
    expect(JSON.parse(raw).content).toBe('ok');
    server.stop();
  });

  it('refs the helper only while a request is in flight', async () => {
    const { server, children } = fakeServer((_m, child) => child.reply(done('x')));
    await server.request('{}');
    const child = children[0];
    // Referenced for the request, released after: an idle helper never keeps
    // the process alive, so forgetting close() does not hang a script.
    expect(child.refs).toBeGreaterThan(0);
    expect(child.unrefs).toBeGreaterThan(0);
    server.stop();
  });

  it('quotes the helper’s stderr when it dies', async () => {
    const { server } = fakeServer((_m, child) => {
      child.stderr.write('Fatal error: something specific\n');
      setTimeout(() => child.crash(6), 5);
    });
    await expect(server.request('{}')).rejects.toThrow(/exited with code 6: Fatal error: something specific/);
    server.stop();
  });
});

/** A DeviceClient over a scripted helper. */
function device(onLine: (message: Record<string, unknown>, child: FakeChild) => void): {
  client: DeviceClient;
  children: FakeChild[];
} {
  const { server, children } = fakeServer(onLine);
  return { client: new DeviceClient({ server }), children };
}

const weatherTool = (execute?: (args: { city: string }) => unknown) =>
  normalizeTools([
    tool({
      name: 'getWeather',
      description: 'Weather for a city',
      parameters: z.object({ city: z.string() }),
      execute: execute as never,
    }),
  ]).functions;

describe('DeviceClient tool loop', () => {
  it('runs a function tool and feeds its result back', async () => {
    const { client, children } = device((m, child) => {
      if (m.op === 'toolResult') {
        child.reply(done(`It is ${JSON.parse(String(m.output)).tempC}C`, { finishReason: 'stop' }));
      } else {
        child.reply({ ok: true, toolCall: { id: 'c1', name: 'getWeather', arguments: { city: 'Paris' } }, done: false });
      }
    });
    const seen: unknown[] = [];
    const outcome = await client.run({
      prompt: 'Weather in Paris?',
      functions: weatherTool(async ({ city }) => {
        seen.push(city);
        return { tempC: 21 };
      }),
    });
    expect(seen).toEqual(['Paris']);
    expect(outcome.content).toBe('It is 21C');
    expect(outcome.toolCalls).toEqual([
      { id: 'c1', name: 'getWeather', arguments: { city: 'Paris' }, output: '{"tempC":21}' },
    ]);
    // Parameters go down in Apple's dialect, titled after the tool.
    const fn = (children[0].requests[0].functions as Array<{ parameters: Record<string, unknown> }>)[0];
    expect(fn.parameters.title).toBe('getWeatherArguments');
    expect(fn.parameters['x-order']).toEqual(['city']);
    client.close();
  });

  it('stops for a tool without execute, reporting the call as tool-calls', async () => {
    const { client, children } = device((m, child) => {
      if (m.op === 'toolResult') child.reply({ ok: true, content: '', finishReason: 'toolCalls' });
      else child.reply({ ok: true, toolCall: { id: 'c9', name: 'getWeather', arguments: { city: 'Oslo' } }, done: false });
    });
    const outcome = await client.run({ prompt: 'x', functions: weatherTool() });
    expect(outcome.finishReason).toBe('tool-calls');
    expect(outcome.toolCalls).toEqual([{ id: 'c9', name: 'getWeather', arguments: { city: 'Oslo' } }]);
    expect(children[0].controls('toolResult')[0]).toMatchObject({ callId: 'c9', stop: true });
    client.close();
  });

  it('replays a recorded result instead of executing the tool again', async () => {
    const { client, children } = device((m, child) => {
      if (m.op === 'toolResult') child.reply(done('replayed'));
      else child.reply({ ok: true, toolCall: { id: 'c2', name: 'getWeather', arguments: { city: 'Oslo' } }, done: false });
    });
    let executed = 0;
    await client.run(
      { prompt: 'x', functions: weatherTool(() => (executed += 1)) },
      'generate',
      { replay: new ReplayBook([{ name: 'getWeather', arguments: { city: 'Oslo' }, output: 'rain' }]) },
    );
    expect(executed).toBe(0);
    expect(children[0].controls('toolResult')[0]).toMatchObject({ output: 'rain' });
    client.close();
  });

  it('surfaces a throwing tool as ToolExecutionError with the original cause', async () => {
    const boom = new Error('database is down');
    const { client } = device((m, child) => {
      if (m.op === 'toolResult') child.reply({ ok: false, kind: 'tool', error: 'tool failed', tool: 'getWeather' });
      else child.reply({ ok: true, toolCall: { id: 'c3', name: 'getWeather', arguments: { city: 'X' } }, done: false });
    });
    const run = client.run({
      prompt: 'x',
      functions: weatherTool(() => {
        throw boom;
      }),
    });
    await expect(run).rejects.toBeInstanceOf(ToolExecutionError);
    await expect(run).rejects.toMatchObject({ toolName: 'getWeather', cause: boom });
    client.close();
  });

  it('tells the model to stop after too many tool calls', async () => {
    let n = 0;
    const { client, children } = device((m, child) => {
      if (m.op === 'toolResult') {
        if (n >= 3) child.reply(done('gave up'));
        else {
          n += 1;
          child.reply({ ok: true, toolCall: { id: `c${n}`, name: 'getWeather', arguments: { city: 'X' } }, done: false });
        }
      } else {
        n += 1;
        child.reply({ ok: true, toolCall: { id: `c${n}`, name: 'getWeather', arguments: { city: 'X' } }, done: false });
      }
    });
    const outcome = await client.run({ prompt: 'x', functions: weatherTool(() => 'sunny') }, 'generate', { maxToolCalls: 2 });
    expect(outcome.content).toBe('gave up');
    const outputs = children[0].controls('toolResult').map((c) => c.output);
    expect(outputs.slice(0, 2)).toEqual(['sunny', 'sunny']);
    expect(String(outputs[2])).toMatch(/limit reached/i);
    client.close();
  });

  it('maps usage, finish reason and trimmed turns', async () => {
    const { client } = device((_m, child) =>
      child.reply(done('cut', { finishReason: 'length', usage: { inputTokens: 10, outputTokens: 5, cachedInputTokens: 2 }, trimmedTurns: 1 })),
    );
    const outcome = await client.run({ prompt: 'x' });
    expect(outcome.finishReason).toBe('length');
    expect(outcome.usage).toEqual({ inputTokens: 10, outputTokens: 5, totalTokens: 15, cachedInputTokens: 2 });
    expect(outcome.trimmedTurns).toBe(1);
    client.close();
  });

  it('carries the token counts on a context overflow', async () => {
    const { client } = device((_m, child) =>
      child.reply({ ok: false, kind: 'context', error: 'too long', contextSize: 8192, tokenCount: 9000 }),
    );
    await expect(client.run({ prompt: 'x' })).rejects.toMatchObject({
      constructor: ContextLengthError,
      contextSize: 8192,
      tokenCount: 9000,
    });
    client.close();
  });

  it('raises unsupported as UnsupportedError', async () => {
    const { client } = device((_m, child) => child.reply({ ok: false, kind: 'unsupported', error: 'no vision' }));
    await expect(client.run({ prompt: 'x' })).rejects.toBeInstanceOf(UnsupportedError);
    client.close();
  });
});

describe('message lists', () => {
  it('splits system, history and the final prompt', () => {
    const split = splitMessages([
      { role: 'system', content: 'Be terse.' },
      { role: 'user', content: 'My cat is Biscuit.' },
      { role: 'assistant', content: 'Nice.' },
      { role: 'user', content: 'What is my cat called?' },
    ]);
    expect(split.system).toBe('Be terse.');
    expect(split.prompt).toBe('What is my cat called?');
    expect(split.history).toEqual([
      { role: 'user', content: 'My cat is Biscuit.' },
      { role: 'assistant', content: 'Nice.' },
    ]);
    expect(split.replay).toEqual([]);
  });

  it('turns a tool exchange after the last user message into replayed results', () => {
    const split = splitMessages([
      { role: 'user', content: 'Weather in Oslo?' },
      { role: 'assistant', content: null, toolCalls: [{ id: 'c1', name: 'getWeather', arguments: { city: 'Oslo' } }] },
      { role: 'tool', toolCallId: 'c1', content: 'rain' },
    ]);
    expect(split.prompt).toBe('Weather in Oslo?');
    expect(split.history).toEqual([]);
    expect(split.replay).toEqual([{ name: 'getWeather', arguments: { city: 'Oslo' }, output: 'rain' }]);
  });

  it('keeps earlier tool turns as native history, resolving tool names by call id', () => {
    const split = splitMessages([
      { role: 'user', content: 'Weather in Oslo?' },
      { role: 'assistant', toolCalls: [{ id: 'c1', name: 'getWeather', arguments: { city: 'Oslo' } }] },
      { role: 'tool', toolCallId: 'c1', content: 'rain' },
      { role: 'assistant', content: 'Rainy.' },
      { role: 'user', content: 'Umbrella?' },
    ]);
    expect(split.history[2]).toEqual({ role: 'tool', toolCallId: 'c1', name: 'getWeather', content: 'rain' });
    expect(renderHistory(split.history)).toContain('Tool getWeather returned: rain');
  });

  it('rejects a list with no user message, or one ending on the assistant', () => {
    expect(() => splitMessages([{ role: 'system', content: 'x' }])).toThrow(/user message/);
    expect(() =>
      splitMessages([
        { role: 'user', content: 'a' },
        { role: 'assistant', content: 'b' },
      ]),
    ).toThrow(/last message/);
  });

  it('replays exact matches first, then by tool name', () => {
    const book = new ReplayBook([
      { name: 't', arguments: { a: 1, b: 2 }, output: 'first' },
      { name: 't', arguments: { a: 9 }, output: 'second' },
    ]);
    expect(book.take('t', { b: 2, a: 1 })).toBe('first');
    expect(book.take('t', { a: 'different' })).toBe('second');
    expect(book.take('t', {})).toBeUndefined();
    expect(canonicalJson({ b: [1, { d: 1, c: 2 }], a: null })).toBe('{"a":null,"b":[1,{"c":2,"d":1}]}');
  });
});

describe('Standard Schema interop', () => {
  const Person = z.object({
    name: z.string(),
    email: z.string().email(),
    nickname: z.string().nullable(),
    age: z.number().int().default(30),
  });

  it('detects a Standard Schema and derives JSON Schema from it', () => {
    expect(isStandardSchema(Person)).toBe(true);
    expect(isStandardSchema({ type: 'object' })).toBe(false);
    const { json, validate } = resolveSchema(Person);
    expect(json.type).toBe('object');
    expect(typeof validate).toBe('function');
    // The input schema: a defaulted field is optional for the model.
    expect(json.required).not.toContain('age');
  });

  it('converts the Zod JSON Schema into Apple’s dialect', () => {
    const apple = toAppleSchema(resolveSchema(Person).json);
    expect(apple.$schema).toBeUndefined();
    const props = apple.properties as Record<string, Record<string, unknown>>;
    expect(props.nickname.type).toBe('string');
    expect(apple.required).not.toContain('nickname');
    expect(apple['x-order']).toEqual(['name', 'email', 'nickname', 'age']);
  });

  it('restores nulls the dialect dropped, then validates with the library', async () => {
    const resolved = resolveSchema(Person);
    const fromModel = { name: 'Ada', email: 'ada@example.com' };
    const restored = restoreNulls(fromModel, resolved.json);
    expect(restored).toEqual({ name: 'Ada', email: 'ada@example.com', nickname: null });
    // Zod applies its own default on top.
    expect(await validateWith(resolved, restored, '', 'device')).toEqual({
      name: 'Ada',
      email: 'ada@example.com',
      nickname: null,
      age: 30,
    });
  });

  it('raises a refinement the decoder cannot enforce as SchemaValidationError', async () => {
    const resolved = resolveSchema(Person);
    const bad = { name: 'Ada', email: 'not-an-email', nickname: null };
    await expect(validateWith(resolved, bad, '{"email":"not-an-email"}', 'device')).rejects.toMatchObject({
      constructor: SchemaValidationError,
      text: '{"email":"not-an-email"}',
    });
  });

  it('restores nulls through $ref, arrays and pydantic-style Optional', () => {
    const schema = {
      type: 'object',
      required: ['items'],
      properties: { items: { type: 'array', items: { $ref: '#/$defs/Item' } } },
      $defs: {
        Item: {
          type: 'object',
          required: ['id', 'parent'],
          properties: {
            id: { type: 'integer' },
            parent: { anyOf: [{ $ref: '#/$defs/Ref' }, { type: 'null' }] },
          },
        },
        Ref: { type: 'object', required: ['note'], properties: { note: { type: ['string', 'null'] } } },
      },
    };
    expect(restoreNulls({ items: [{ id: 1 }, { id: 2, parent: {} }] }, schema)).toEqual({
      items: [
        { id: 1, parent: null },
        { id: 2, parent: { note: null } },
      ],
    });
  });

  it('never overwrites a value the model produced', () => {
    const schema = { type: 'object', required: ['a'], properties: { a: { type: ['string', 'null'] } } };
    expect(restoreNulls({ a: 'kept' }, schema)).toEqual({ a: 'kept' });
  });

  it('explains how to fix a schema library that cannot describe itself', () => {
    const noJson = { '~standard': { version: 1, vendor: 'valibot', validate: () => ({ value: 1 }) } };
    expect(() => resolveSchema(noJson as never)).toThrow(/toStandardJsonSchema/);
  });
});

describe('the anyOf-with-null dialect rule', () => {
  it('unwraps a nullable object union into an optional object', () => {
    const out = toAppleSchema({
      type: 'object',
      required: ['o'],
      properties: {
        o: { anyOf: [{ type: 'object', properties: { a: { type: 'string' } }, required: ['a'] }, { type: 'null' }] },
      },
    });
    const o = (out.properties as Record<string, Record<string, unknown>>).o;
    expect(o.anyOf).toBeUndefined();
    expect(o.type).toBe('object');
    expect(out.required).toEqual([]);
  });

  it('keeps a genuine multi-branch union, minus the null branch', () => {
    const out = toAppleSchema({
      type: 'object',
      properties: { v: { anyOf: [{ type: 'string' }, { type: 'integer' }, { type: 'null' }] } },
    });
    const v = (out.properties as Record<string, Record<string, unknown>>).v;
    expect(v.anyOf).toEqual([{ type: 'string' }, { type: 'integer' }]);
  });
});

describe('function tools', () => {
  it('accepts built-ins, arrays and records', () => {
    const fromArray = normalizeTools(['ocr', tool({ name: 'a', execute: () => 1 })]);
    expect(fromArray.builtIn).toEqual(['ocr']);
    expect(fromArray.functions.map((f) => f.name)).toEqual(['a']);
    const fromRecord = normalizeTools({ lookup: { description: 'd', execute: () => 1 } });
    expect(fromRecord.functions[0].name).toBe('lookup');
    // No parameters means an empty object, which Apple still needs spelled out.
    expect(fromRecord.functions[0].json).toEqual({ type: 'object', properties: {}, required: [] });
  });

  it('rejects bad names and duplicates before any model call', () => {
    expect(() => normalizeTools([tool({ name: 'has space' })])).toThrow(/invalid/);
    expect(() => normalizeTools([tool({ name: 'a' }), tool({ name: 'a' })])).toThrow(/twice/);
    expect(() => normalizeTools(['laser' as never])).toThrow(/Unknown tool/);
  });

  it('turns tool results into text the model can read', () => {
    expect(toolOutputText('plain')).toBe('plain');
    expect(toolOutputText({ a: 1 })).toBe('{"a":1}');
    expect(toolOutputText(undefined)).toBe('');
  });
});

describe('ResultStream', () => {
  const make = (chunks: string[], fail?: Error) =>
    new ResultStream<string, string, { text: string }>(
      async (emit, signal) => {
        for (const chunk of chunks) {
          if (signal.aborted) throw new AbortError('aborted');
          emit(chunk);
          await tick(2);
        }
        if (fail !== undefined) throw fail;
        return { text: chunks.join('') };
      },
      (r) => r.text,
    );

  it('iterates deltas and awaits to the full text', async () => {
    const stream = make(['a', 'b', 'c']);
    const seen: string[] = [];
    for await (const d of stream) seen.push(d);
    expect(seen).toEqual(['a', 'b', 'c']);
    expect(await stream).toBe('abc');
    expect((await stream.result).text).toBe('abc');
  });

  it('can be awaited without iterating', async () => {
    expect(await make(['x', 'y'])).toBe('xy');
  });

  it('cancels generation when the reader breaks out', async () => {
    const stream = make(['a', 'b', 'c', 'd', 'e']);
    for await (const d of stream) {
      if (d === 'b') break;
    }
    await expect(stream.result).rejects.toBeInstanceOf(AbortError);
  });

  it('surfaces an error through the iterator and the promise', async () => {
    const boom = new Error('boom');
    const stream = make(['a'], boom);
    await expect(
      (async () => {
        for await (const _ of stream) {
          // drain
        }
      })(),
    ).rejects.toBe(boom);
    await expect(stream).rejects.toBe(boom);
  });

  it('does not raise an unhandled rejection when nobody consumes it', async () => {
    const unhandled: unknown[] = [];
    const listener = (reason: unknown): void => {
      unhandled.push(reason);
    };
    process.on('unhandledRejection', listener);
    make([], new Error('ignored'));
    await tick(30);
    process.off('unhandledRejection', listener);
    expect(unhandled).toEqual([]);
  });
});

describe('busy model manager', () => {
  it('retries a request the model manager turned away, then succeeds', async () => {
    let calls = 0;
    const { client, children } = device((_m, child) => {
      calls += 1;
      if (calls < 3) child.reply({ ok: false, kind: 'busy', error: 'ModelManagerError Code=1042' });
      else child.reply(done('finally'));
    });
    const outcome = await client.run({ prompt: 'x' });
    expect(outcome.content).toBe('finally');
    // Each attempt carries its own id, so a cancel always targets the live one.
    const ids = children[0].requests.map((r) => r.id);
    expect(new Set(ids).size).toBe(3);
    client.close();
  });

  it('gives up with ModelBusyError after the backoff runs out', async () => {
    const { client } = device((_m, child) => child.reply({ ok: false, kind: 'busy', error: '1042' }));
    await expect(client.run({ prompt: 'x' })).rejects.toMatchObject({ code: 'MODEL_BUSY' });
    client.close();
  }, 10_000);

  it('never retries once events have reached the caller', async () => {
    let calls = 0;
    const { client } = device((_m, child) => {
      calls += 1;
      child.reply({ ok: true, delta: 'partial', done: false });
      child.reply({ ok: false, kind: 'busy', error: '1042' });
    });
    await expect(client.run({ prompt: 'x' }, 'stream', { onDelta: () => undefined })).rejects.toMatchObject({ code: 'MODEL_BUSY' });
    expect(calls).toBe(1);
    client.close();
  });
});

describe('cloud tier, with Shortcuts stubbed out', () => {
  // No quota spent and nothing sent anywhere: CloudClient's two touch points
  // with Shortcuts are replaced, so only this package's own logic runs.
  const cloud = async (replies: string[]) => {
    const { vi } = await import('vitest');
    const { CloudClient } = await import('../src/cloud.js');
    const { AppleLLM } = await import('../src/client.js');
    const prompts: Array<{ system?: string; prompt: string }> = [];
    vi.spyOn(CloudClient.prototype, 'ensureReady').mockResolvedValue();
    vi.spyOn(CloudClient.prototype, 'text').mockImplementation(async (request) => {
      prompts.push({ system: request.system, prompt: request.prompt });
      return replies.shift() ?? '';
    });
    return { llm: new AppleLLM({ tier: 'cloud' }), prompts, restore: () => vi.restoreAllMocks() };
  };

  it('carries message history as a transcript preamble', async () => {
    const { llm, prompts, restore } = await cloud(['Biscuit.']);
    const reply = await llm.text([
      { role: 'user', content: 'My cat is Biscuit.' },
      { role: 'assistant', content: 'Nice.' },
      { role: 'user', content: 'What is my cat called?' },
    ]);
    expect(reply).toBe('Biscuit.');
    expect(prompts[0].prompt).toContain('User: My cat is Biscuit.');
    expect(prompts[0].prompt).toContain('New message from the user:\nWhat is my cat called?');
    restore();
  });

  it('recovers JSON from prose, validates it, and repairs a bad reply once', async () => {
    const { llm, prompts, restore } = await cloud([
      'Sure! ```json\n{"name":"Ada","email":"nope"}\n```',
      '{"name":"Ada","email":"ada@example.com"}',
    ]);
    const person = await llm.json('Ada, ada@example.com', {
      schema: z.object({ name: z.string(), email: z.string().email() }),
    });
    expect(person).toEqual({ name: 'Ada', email: 'ada@example.com' });
    expect(prompts).toHaveLength(2);
    expect(prompts[1].prompt).toMatch(/previous answer was rejected: email/i);
    expect(prompts[0].system).toContain('"email"');
    restore();
  });

  it('streams the reply as one chunk', async () => {
    const { llm, restore } = await cloud(['whole reply']);
    const chunks: string[] = [];
    for await (const chunk of llm.stream('hi')) chunks.push(chunk);
    expect(chunks).toEqual(['whole reply']);
    restore();
  });

  it('refuses what only the device can do, before any call', async () => {
    const { llm, prompts, restore } = await cloud([]);
    await expect(llm.text('hi', { tools: [tool({ name: 't', execute: () => 1 })] })).rejects.toBeInstanceOf(UnsupportedError);
    await expect(llm.text('hi', { sessionId: 's' })).rejects.toBeInstanceOf(UnsupportedError);
    expect(prompts).toHaveLength(0);
    restore();
  });

  it('sends images only when the server model is known to read them', async () => {
    const { vi } = await import('vitest');
    const { CloudClient } = await import('../src/cloud.js');
    const { llm, prompts, restore } = await cloud(['a blue circle']);
    const vision = vi.spyOn(CloudClient.prototype, 'supportsImages', 'get').mockReturnValue(false);
    await expect(llm.text('What is this?', { images: ['x.png'] })).rejects.toThrow(/Golden Gate/);
    expect(prompts).toHaveLength(0);
    vision.mockReturnValue(true);
    expect(await llm.text('What is this?', { images: ['x.png'] })).toBe('a blue circle');
    restore();
  });
});

describe('cloud tier against a fake `shortcuts` binary', () => {
  // A stand-in for /usr/bin/shortcuts on PATH: it records its arguments and
  // writes a reply, so CloudClient runs for real with no quota spent.
  const onAppleSilicon = process.platform === 'darwin' && process.arch === 'arm64';
  const withFakeShortcuts = async (run: (argsFile: string) => Promise<void>): Promise<void> => {
    const { mkdtemp, writeFile, chmod, rm } = await import('node:fs/promises');
    const { tmpdir } = await import('node:os');
    const path = await import('node:path');
    const dir = await mkdtemp(path.join(tmpdir(), 'fake-shortcuts-'));
    const argsFile = path.join(dir, 'args.txt');
    const script = path.join(dir, 'shortcuts');
    await writeFile(
      script,
      [
        '#!/bin/bash',
        'if [ "$1" = "list" ]; then echo "Apple LLM Cloud"; exit 0; fi',
        `printf '%s\n' "$@" > ${JSON.stringify(argsFile)}`,
        'out=""; prev=""; for a in "$@"; do if [ "$prev" = "-o" ]; then out="$a"; fi; prev="$a"; done',
        'echo "fake reply" > "$out"',
      ].join('\n'),
    );
    await chmod(script, 0o755);
    const saved = process.env.PATH;
    process.env.PATH = `${dir}:${saved}`;
    try {
      await run(argsFile);
    } finally {
      process.env.PATH = saved;
      await rm(dir, { recursive: true, force: true });
    }
  };

  it.runIf(onAppleSilicon)('passes the prompt file and each image as inputs', async () => {
    const { readFile } = await import('node:fs/promises');
    const { fileURLToPath } = await import('node:url');
    const { CloudClient } = await import('../src/cloud.js');
    const image = fileURLToPath(new URL('../../../tests/fixtures/images/red-square-blue-circle.png', import.meta.url));
    await withFakeShortcuts(async (argsFile) => {
      const client = new CloudClient();
      client.setQuota({ isAvailable: true, status: 'belowLimit', capabilities: { vision: true, guidedGeneration: true, reasoning: true, toolCalling: true } });
      const reply = await client.text({ prompt: 'What shapes?', images: [{ path: image, label: 'shapes' }] });
      expect(reply.trim()).toBe('fake reply');
      const args = (await readFile(argsFile, 'utf8')).trim().split('\n');
      expect(args.slice(0, 2)).toEqual(['run', 'Apple LLM Cloud']);
      const inputs = args.flatMap((a, i) => (args[i - 1] === '-i' ? [a] : []));
      expect(inputs).toHaveLength(2);
      expect(inputs[0]).toMatch(/prompt\.txt$/);
      expect(inputs[1]).toBe(image);
    });
  });

  it.runIf(onAppleSilicon)('refuses a missing image rather than dropping it', async () => {
    const { CloudClient } = await import('../src/cloud.js');
    await withFakeShortcuts(async () => {
      const client = new CloudClient();
      await expect(client.text({ prompt: 'x', images: ['/definitely/not/here.png'] })).rejects.toThrow(/Image not found/);
    });
  });
});

describe('dialect rule 9: pattern', () => {
  it('is stripped for Apple, at any depth', () => {
    const out = toAppleSchema({
      type: 'object',
      properties: {
        code: { type: 'string', pattern: '^[A-Z]{3}$' },
        list: { type: 'array', items: { type: 'string', pattern: '^x' } },
        // A *property* called pattern is data, not the keyword.
        pattern: { type: 'string' },
      },
    });
    const props = out.properties as Record<string, Record<string, unknown>>;
    expect(props.code.pattern).toBeUndefined();
    expect((props.list.items as Record<string, unknown>).pattern).toBeUndefined();
    expect(props.pattern).toEqual({ type: 'string' });
  });

  it('is still enforced on the reply for plain JSON Schema', async () => {
    const schema = {
      type: 'object',
      properties: { code: { type: 'string', pattern: '^[A-Z]{3}$' }, tags: { type: 'array', items: { type: 'string', pattern: '^#' } } },
    };
    const resolved = resolveSchema(schema);
    await expect(validateWith(resolved, { code: 'ABC', tags: ['#a'] }, '', 'device')).resolves.toBeDefined();
    await expect(validateWith(resolved, { code: 'abc', tags: ['#a', 'b'] }, '', 'device')).rejects.toMatchObject({
      constructor: SchemaValidationError,
      issues: [
        { message: 'must match the pattern ^[A-Z]{3}$', path: ['code'] },
        { message: 'must match the pattern ^#', path: ['tags', 1] },
      ],
    });
  });

  it('turns a Zod email into a schema Apple accepts', () => {
    const apple = toAppleSchema(resolveSchema(z.object({ email: z.string().email() })).json);
    expect(JSON.stringify(apple)).not.toContain('"pattern"');
  });
});
