/**
 * The AI SDK provider, through the real `ai` package, against the real model.
 *
 *   APPLE_LLM_LIVE=1 npx vitest run test/live-ai-sdk.test.ts
 */
import { generateObject, generateText, stepCountIs, streamText, tool } from 'ai';
import { afterAll, beforeAll, describe, expect, it } from 'vitest';
import { z } from 'zod';
import { createApple } from '../src/ai-sdk.js';
import { AppleLLM, probeDevice } from '../src/index.js';

const LIVE = process.env.APPLE_LLM_LIVE === '1';
let available = false;
beforeAll(async () => {
  if (LIVE) available = (await probeDevice()).available;
});

const live = LIVE ? describe : describe.skip;

live('live: AI SDK provider', () => {
  const client = new AppleLLM({ tier: 'device' });
  const apple = createApple({ client });
  afterAll(() => client.close());
  const gate = (): boolean => !available;

  it('generateText', async () => {
    if (gate()) return;
    const { text, usage, finishReason } = await generateText({
      model: apple(),
      system: 'You are terse.',
      prompt: 'Name one primary colour. One word.',
    });
    expect(text.trim().length).toBeGreaterThan(0);
    expect(finishReason).toBe('stop');
    expect(usage.inputTokens).toBeGreaterThan(0);
  });

  it('streamText', async () => {
    if (gate()) return;
    const result = streamText({ model: apple(), prompt: 'Count from one to five in words.' });
    const chunks: string[] = [];
    for await (const chunk of result.textStream) chunks.push(chunk);
    expect(chunks.join('')).toMatch(/one/i);
    expect((await result.usage).outputTokens).toBeGreaterThan(0);
  });

  it('runs a multi-step tool loop driven by the SDK', async () => {
    if (gate()) return;
    const cities: string[] = [];
    const { text, steps } = await generateText({
      model: apple(),
      prompt: 'What is the weather in Paris right now?',
      tools: {
        getWeather: tool({
          description: 'Get the current weather for a city.',
          inputSchema: z.object({ city: z.string() }),
          execute: async ({ city }) => {
            cities.push(city);
            return { tempC: 23, sky: 'clear' };
          },
        }),
      },
      stopWhen: stepCountIs(4),
    });
    expect(cities.length).toBe(1);
    expect(steps.length).toBeGreaterThanOrEqual(2);
    expect(text).toMatch(/23/);
  });

  it('generateObject is constrained on device', async () => {
    if (gate()) return;
    const { object } = await generateObject({
      model: apple(),
      schema: z.object({ name: z.string(), born: z.number().int() }),
      prompt: 'Ada Lovelace was born in 1815.',
    });
    expect(object.born).toBe(1815);
  });

  it('passes images through as attachments', async () => {
    if (gate()) return;
    const { readFile } = await import('node:fs/promises');
    const { fileURLToPath } = await import('node:url');
    const image = await readFile(
      fileURLToPath(new URL('../../../tests/fixtures/images/red-square-blue-circle.png', import.meta.url)),
    );
    const { text } = await generateText({
      model: apple(),
      messages: [
        {
          role: 'user',
          content: [
            { type: 'text', text: 'What shapes are in this image? Answer briefly.' },
            { type: 'image', image },
          ],
        },
      ],
    });
    expect(text.toLowerCase()).toMatch(/square|circle/);
  });

  it('cancels with an abort signal', async () => {
    if (gate()) return;
    const controller = new AbortController();
    setTimeout(() => controller.abort(), 300);
    await expect(
      generateText({
        model: apple(),
        prompt: 'Write a long essay about the history of bridges.',
        maxOutputTokens: 1500,
        abortSignal: controller.signal,
      }),
    ).rejects.toThrow();
  });
});
