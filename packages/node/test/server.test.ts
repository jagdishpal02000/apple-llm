/**
 * The OpenAI-compatible server: routing, auth and error envelopes offline, and
 * real completions through the official `openai` SDK when APPLE_LLM_LIVE=1.
 */
import { readFile } from 'node:fs/promises';
import OpenAI from 'openai';
import { afterAll, beforeAll, describe, expect, it } from 'vitest';
import { probeDevice } from '../src/index.js';
import { serve } from '../src/server.js';

type Running = Awaited<ReturnType<typeof serve>>;

describe('server, without the model', () => {
  let running: Running;
  beforeAll(async () => {
    running = await serve({ port: 0, apiKey: 'secret' });
  });
  afterAll(() => running.close());

  const get = (path: string, key?: string): Promise<Response> =>
    fetch(`${running.url.replace(/\/v1$/, '')}${path}`, key ? { headers: { authorization: `Bearer ${key}` } } : {});

  it('binds to localhost by default', () => {
    expect(running.url).toMatch(/^http:\/\/127\.0\.0\.1:\d+\/v1$/);
  });

  it('requires the API key when one is set', async () => {
    const res = await get('/v1/models');
    expect(res.status).toBe(401);
    expect(((await res.json()) as { error: { code: string } }).error.code).toBe('invalid_api_key');
  });

  it('lists the Apple models', async () => {
    const res = await get('/v1/models', 'secret');
    const body = (await res.json()) as { data: Array<{ id: string }> };
    expect(body.data.map((m) => m.id)).toEqual(['apple-on-device', 'apple-private-cloud']);
  });

  it('answers unknown routes and bad bodies with OpenAI-style errors', async () => {
    expect((await get('/v1/nope', 'secret')).status).toBe(404);
    const bad = await fetch(`${running.url}/chat/completions`, {
      method: 'POST',
      headers: { authorization: 'Bearer secret', 'content-type': 'application/json' },
      body: '{not json',
    });
    expect(bad.status).toBe(400);
    expect(((await bad.json()) as { error: { message: string } }).error.message).toMatch(/not valid JSON/);
  });

  it('sends no CORS headers unless asked to', async () => {
    const res = await get('/v1/models', 'secret');
    expect(res.headers.get('access-control-allow-origin')).toBeNull();
  });
});

describe('server, when Apple rate-limits the cloud', () => {
  const stubCloud = async () => {
    const { vi } = await import('vitest');
    const { CloudClient, describeRunFailure } = await import('../src/cloud.js');
    const { DeviceClient } = await import('../src/device.js');
    vi.spyOn(CloudClient.prototype, 'ensureReady').mockResolvedValue();
    vi.spyOn(CloudClient.prototype, 'text').mockRejectedValue(
      describeRunFailure({ stderr: "Error: Your request can't be completed right now." }),
    );
    vi.spyOn(DeviceClient.prototype, 'ensureReady').mockResolvedValue();
    vi.spyOn(DeviceClient.prototype, 'run').mockResolvedValue({
      content: 'from the device',
      finishReason: 'stop',
      toolCalls: [],
      trimmedTurns: 0,
    });
    return () => vi.restoreAllMocks();
  };
  const ask = (url: string): Promise<Response> =>
    fetch(`${url}/chat/completions`, {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ model: 'apple-private-cloud', messages: [{ role: 'user', content: 'hi' }] }),
    });

  it('answers 429 and tells OpenAI clients not to retry seconds later', async () => {
    const restore = await stubCloud();
    const running = await serve({ port: 0 });
    try {
      const res = await ask(running.url);
      expect(res.status).toBe(429);
      expect(res.headers.get('x-should-retry')).toBe('false');
      const body = (await res.json()) as { error: { code: string; message: string } };
      expect(body.error.code).toBe('quota_exceeded');
      expect(body.error.message).toMatch(/rate-limiting this Mac/);
    } finally {
      await running.close();
      restore();
    }
  });

  it('gives Ollama clients a real 429 even on a streaming request', async () => {
    const restore = await stubCloud();
    const running = await serve({ port: 0 });
    try {
      const res = await fetch(`${running.url.replace(/\/v1$/, '')}/api/chat`, {
        method: 'POST',
        body: JSON.stringify({ model: 'apple-private-cloud', messages: [{ role: 'user', content: 'hi' }] }),
      });
      expect(res.status).toBe(429);
      expect(((await res.json()) as { error: string }).error).toMatch(/rate-limiting this Mac/);
    } finally {
      await running.close();
      restore();
    }
  });

  it('answers on device with fallback, and says which model answered', async () => {
    const restore = await stubCloud();
    const running = await serve({ port: 0, fallback: 'device' });
    try {
      const res = await ask(running.url);
      expect(res.status).toBe(200);
      const body = (await res.json()) as { model: string; choices: Array<{ message: { content: string } }> };
      expect(body.choices[0].message.content).toBe('from the device');
      expect(body.model).toBe('apple-on-device');
    } finally {
      await running.close();
      restore();
    }
  });
});

describe('server, Ollama API', () => {
  let running: Running;
  let base: string;
  const runs: Array<Record<string, unknown>> = [];
  let replies: Array<Record<string, unknown>> = [];
  beforeAll(async () => {
    const { vi } = await import('vitest');
    const { DeviceClient } = await import('../src/device.js');
    vi.spyOn(DeviceClient.prototype, 'ensureReady').mockResolvedValue();
    vi.spyOn(DeviceClient.prototype, 'run').mockImplementation(async (request, op, hooks) => {
      runs.push({ ...request, op });
      const next = replies.shift() ?? { content: 'Hello there' };
      if (op === 'stream' && next.toolCalls === undefined) {
        for (const piece of String(next.content).split(/(?<= )/)) hooks?.onDelta?.(piece);
      }
      return { finishReason: 'stop', toolCalls: [], trimmedTurns: 0, usage: { inputTokens: 7, outputTokens: 3, totalTokens: 10 }, ...next } as never;
    });
    running = await serve({ port: 0 });
    base = running.url.replace(/\/v1$/, '');
  });
  afterAll(async () => {
    await running.close();
    const { vi } = await import('vitest');
    vi.restoreAllMocks();
  });
  const post = (path: string, body: unknown): Promise<Response> =>
    fetch(`${base}${path}`, { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify(body) });
  const lines = async (res: Response): Promise<Array<Record<string, any>>> =>
    (await res.text()).trim().split('\n').map((line) => JSON.parse(line) as Record<string, any>);

  it('answers the discovery calls Ollama apps make first', async () => {
    expect(await (await fetch(`${base}/`)).text()).toBe('Ollama is running');
    const version = (await (await fetch(`${base}/api/version`)).json()) as { version: string };
    expect(version.version).toMatch(/^\d+\.\d+\.\d+$/);
    const tags = (await (await fetch(`${base}/api/tags`)).json()) as { models: Array<{ name: string; details: unknown }> };
    expect(tags.models.map((m) => m.name)).toEqual(['apple-on-device:latest', 'apple-private-cloud:latest']);
    expect(((await (await fetch(`${base}/api/ps`)).json()) as { models: unknown[] }).models).toEqual([]);
  });

  it('streams /api/chat as NDJSON by default, ending with done and token counts', async () => {
    runs.length = 0;
    const res = await post('/api/chat', {
      model: 'apple-on-device:latest',
      messages: [
        { role: 'system', content: 'You are terse.' },
        { role: 'user', content: 'Say hello' },
      ],
    });
    expect(res.headers.get('content-type')).toBe('application/x-ndjson');
    const chunks = await lines(res);
    const last = chunks.pop()!;
    expect(chunks.map((c) => c.message.content).join('')).toBe('Hello there');
    expect(chunks.every((c) => c.done === false && c.model === 'apple-on-device:latest')).toBe(true);
    expect(last).toMatchObject({ done: true, done_reason: 'stop', prompt_eval_count: 7, eval_count: 3 });
    expect(runs[0]).toMatchObject({ system: 'You are terse.', prompt: 'Say hello', op: 'stream' });
  });

  it('answers /api/generate in one object with stream:false', async () => {
    const res = await post('/api/generate', { model: 'apple-on-device', prompt: 'Say hello', system: 'Be brief.', stream: false });
    const body = (await res.json()) as Record<string, unknown>;
    expect(body).toMatchObject({ model: 'apple-on-device', response: 'Hello there', done: true, done_reason: 'stop' });
  });

  it('treats an empty request as a model load, without calling the model', async () => {
    runs.length = 0;
    const res = await post('/api/generate', { model: 'apple-on-device' });
    expect(await res.json()).toMatchObject({ done: true, done_reason: 'load', response: '' });
    expect(runs).toHaveLength(0);
  });

  it('maps format: a schema is constrained decoding, "json" is unwrapped from prose', async () => {
    runs.length = 0;
    replies = [{ content: '{"city":"Paris"}' }];
    const schema = { type: 'object', properties: { city: { type: 'string' } }, required: ['city'] };
    const structured = await post('/api/chat', { messages: [{ role: 'user', content: 'A city' }], format: schema, stream: false });
    expect(((await structured.json()) as { message: { content: string } }).message.content).toBe('{"city":"Paris"}');
    expect(runs[0].schema).toBeTruthy();

    replies = [{ content: 'Sure! ```json\n{"city": "Lyon"}\n```' }];
    const loose = await post('/api/chat', { messages: [{ role: 'user', content: 'A city' }], format: 'json', stream: false });
    expect(((await loose.json()) as { message: { content: string } }).message.content).toBe('{"city":"Lyon"}');
  });

  it('returns tool calls in Ollama form and pairs results by tool_name', async () => {
    runs.length = 0;
    replies = [{ content: '', finishReason: 'tool-calls', toolCalls: [{ id: 'x', name: 'weather', arguments: { city: 'Paris' } }] }];
    const tools = [{ type: 'function', function: { name: 'weather', parameters: { type: 'object', properties: { city: { type: 'string' } } } } }];
    const first = await post('/api/chat', { messages: [{ role: 'user', content: 'Weather in Paris?' }], tools, stream: false });
    const message = ((await first.json()) as { message: Record<string, unknown> }).message;
    expect(message.tool_calls).toEqual([{ function: { name: 'weather', arguments: { city: 'Paris' } } }]);

    await post('/api/chat', {
      messages: [
        { role: 'user', content: 'Weather in Paris?' },
        { role: 'assistant', content: '', tool_calls: [{ function: { name: 'weather', arguments: { city: 'Paris' } } }] },
        { role: 'tool', tool_name: 'weather', content: 'Sunny, 21C' },
      ],
      tools,
      stream: false,
    });
    expect(runs[1].prompt).toBe('Weather in Paris?');
  });

  it('maps options onto sampling and the token limit', async () => {
    runs.length = 0;
    await post('/api/generate', { prompt: 'x', stream: false, options: { temperature: 0.7, num_predict: 64, seed: 42 } });
    await post('/api/generate', { prompt: 'x', stream: false, options: { num_predict: -1 } });
    expect(runs[0]).toMatchObject({ temperature: 0.7, maxTokens: 64, sampling: { mode: 'topK', k: 50, seed: 42 } });
    expect(runs[1].maxTokens).toBeUndefined();
  });

  it('answers errors in Ollama form', async () => {
    const show = await post('/api/show', { model: 'llama3' });
    expect(show.status).toBe(404);
    expect(await show.json()).toEqual({ error: "model 'llama3' not found" });
    // What the Ollama CLI actually sends: the name, with model left empty.
    const cli = await post('/api/show', { model: '', name: 'nope' });
    expect(await cli.json()).toEqual({ error: "model 'nope' not found" });
    expect((await post('/api/embed', {})).status).toBe(404);
  });
});

const LIVE = process.env.APPLE_LLM_LIVE === '1';
const live = LIVE ? describe : describe.skip;

live('live: server through the openai SDK', () => {
  let running: Running;
  let openai: OpenAI;
  let available = false;
  beforeAll(async () => {
    available = (await probeDevice()).available;
    running = await serve({ port: 0 });
    openai = new OpenAI({ baseURL: running.url, apiKey: 'unused' });
  });
  afterAll(() => running.close());
  const gate = (): boolean => !available;

  it('completes a chat', async () => {
    if (gate()) return;
    const completion = await openai.chat.completions.create({
      model: 'apple-on-device',
      messages: [
        { role: 'system', content: 'You are terse.' },
        { role: 'user', content: 'My cat is called Biscuit.' },
        { role: 'assistant', content: 'Lovely name.' },
        { role: 'user', content: 'What is my cat called?' },
      ],
    });
    expect(completion.choices[0].message.content).toMatch(/Biscuit/);
    expect(completion.choices[0].finish_reason).toBe('stop');
    expect(completion.usage?.prompt_tokens).toBeGreaterThan(0);
  });

  it('streams chunks with usage', async () => {
    if (gate()) return;
    const stream = await openai.chat.completions.create({
      model: 'gpt-4o', // unknown names fall back to the default tier
      messages: [{ role: 'user', content: 'Count from one to five in words.' }],
      stream: true,
      stream_options: { include_usage: true },
    });
    let text = '';
    let finish: string | null = null;
    let usage: unknown;
    for await (const chunk of stream) {
      text += chunk.choices[0]?.delta?.content ?? '';
      finish = chunk.choices[0]?.finish_reason ?? finish;
      usage = chunk.usage ?? usage;
    }
    expect(text.toLowerCase()).toMatch(/one/);
    expect(finish).toBe('stop');
    expect(usage).toBeDefined();
  });

  it('round-trips a tool call', async () => {
    if (gate()) return;
    const tools = [
      {
        type: 'function' as const,
        function: {
          name: 'getWeather',
          description: 'Get the current weather for a city.',
          parameters: { type: 'object', properties: { city: { type: 'string' } }, required: ['city'] },
        },
      },
    ];
    const messages: OpenAI.Chat.ChatCompletionMessageParam[] = [{ role: 'user', content: 'What is the weather in Lisbon?' }];
    const first = await openai.chat.completions.create({ model: 'apple-on-device', messages, tools });
    expect(first.choices[0].finish_reason).toBe('tool_calls');
    const call = first.choices[0].message.tool_calls![0] as OpenAI.Chat.ChatCompletionMessageFunctionToolCall;
    expect(call.function.name).toBe('getWeather');
    expect(JSON.parse(call.function.arguments).city).toMatch(/Lisbon/i);

    messages.push(first.choices[0].message, { role: 'tool', tool_call_id: call.id, content: '{"tempC": 26, "sky": "sunny"}' });
    const second = await openai.chat.completions.create({ model: 'apple-on-device', messages, tools });
    expect(second.choices[0].finish_reason).toBe('stop');
    expect(second.choices[0].message.content).toMatch(/26|sunny/i);
  });

  it('honours response_format json_schema', async () => {
    if (gate()) return;
    const completion = await openai.chat.completions.create({
      model: 'apple-on-device',
      messages: [{ role: 'user', content: 'Ada Lovelace was born in 1815.' }],
      response_format: {
        type: 'json_schema',
        json_schema: {
          name: 'person',
          schema: { type: 'object', properties: { name: { type: 'string' }, born: { type: 'integer' } }, required: ['name', 'born'] },
        },
      },
    });
    expect(JSON.parse(completion.choices[0].message.content!).born).toBe(1815);
  });

  it('accepts an image as a data URL', async () => {
    if (gate()) return;
    const png = await readFile(new URL('../../../tests/fixtures/images/red-square-blue-circle.png', import.meta.url));
    const completion = await openai.chat.completions.create({
      model: 'apple-on-device',
      messages: [
        {
          role: 'user',
          content: [
            { type: 'text', text: 'What shapes are in this image? Answer briefly.' },
            { type: 'image_url', image_url: { url: `data:image/png;base64,${png.toString('base64')}` } },
          ],
        },
      ],
    });
    expect(completion.choices[0].message.content!.toLowerCase()).toMatch(/square|circle/);
  });

  it('reports health', async () => {
    const res = await fetch(`${running.url.replace(/\/v1$/, '')}/health`);
    const body = (await res.json()) as { status: string; device: { available: boolean } };
    expect(body.device.available).toBe(available);
  });
});

live('live: server when the client hangs up', () => {
  it('stops generating and keeps serving', async () => {
    if (!(await probeDevice()).available) return;
    const running = await serve({ port: 0 });
    try {
      const controller = new AbortController();
      const res = await fetch(`${running.url}/chat/completions`, {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify({
          model: 'apple-on-device',
          stream: true,
          max_tokens: 1500,
          messages: [{ role: 'user', content: 'Write a long essay about the history of bridges.' }],
        }),
        signal: controller.signal,
      });
      const reader = res.body!.getReader();
      await reader.read();
      await reader.read();
      controller.abort();
      // The helper was told to stop, so the next request is not stuck behind
      // 1500 tokens of an essay nobody is reading.
      const started = Date.now();
      const next = await fetch(`${running.url}/chat/completions`, {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify({ model: 'apple-on-device', messages: [{ role: 'user', content: 'Say OK.' }], max_tokens: 10 }),
      });
      expect(next.status).toBe(200);
      expect(Date.now() - started).toBeLessThan(8000);
    } finally {
      await running.close();
    }
  });
});
