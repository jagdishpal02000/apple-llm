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
