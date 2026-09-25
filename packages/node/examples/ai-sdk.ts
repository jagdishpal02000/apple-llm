/**
 * The Vercel AI SDK, with Apple's on-device model.
 *
 *   npm install ai zod
 *   npx tsx examples/ai-sdk.ts
 */
import { generateText, stepCountIs, streamText, tool } from 'ai';
import { z } from 'zod';
import { apple } from 'apple-llm/ai-sdk';

const { text, steps } = await generateText({
  model: apple(), // on device; apple('cloud') for Private Cloud Compute
  prompt: 'What is the weather in Paris?',
  tools: {
    getWeather: tool({
      description: 'Current weather for a city',
      inputSchema: z.object({ city: z.string() }),
      execute: async ({ city }) => ({ city, tempC: 21, sky: 'sunny' }),
    }),
  },
  stopWhen: stepCountIs(3),
});
console.log(text, `(${steps.length} steps)`);

const result = streamText({ model: apple(), prompt: 'Write a haiku about rain.' });
for await (const chunk of result.textStream) process.stdout.write(chunk);
process.stdout.write('\n');
