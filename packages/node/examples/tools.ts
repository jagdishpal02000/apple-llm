/**
 * Tools: JavaScript functions the model calls mid-generation — and the same
 * tool driven by hand, for callers that run tools themselves.
 *
 *   npx tsx examples/tools.ts
 */
import { z } from 'zod';
import { AppleLLM, tool, type ChatMessage } from 'apple-llm';

const llm = new AppleLLM({ tier: 'device' });

async function fetchWeather(city: string): Promise<{ city: string; tempC: number; sky: string }> {
  return { city, tempC: 21, sky: 'sunny' }; // your real API call goes here
}

// With execute: the model calls it, the result goes straight back, and the
// reply already accounts for it.
const getWeather = tool({
  name: 'getWeather',
  description: 'Current weather for a city',
  parameters: z.object({ city: z.string() }),
  execute: async ({ city }) => fetchWeather(city),
});

const result = await llm.generate('Do I need an umbrella in Paris?', { tools: [getWeather] });
console.log(result.text);
console.log(result.toolCalls);

// Without execute: generation stops at the call, you run it, and continue.
const manual = tool({
  name: 'getWeather',
  description: 'Current weather for a city',
  parameters: z.object({ city: z.string() }),
});
const messages: ChatMessage[] = [{ role: 'user', content: 'What is the weather in Oslo?' }];
const first = await llm.generate(messages, { tools: [manual] });
if (first.finishReason === 'tool-calls') {
  messages.push(first.message);
  for (const call of first.toolCalls) {
    const { city } = call.arguments as { city: string };
    messages.push({ role: 'tool', toolCallId: call.id, content: JSON.stringify(await fetchWeather(city)) });
  }
  console.log((await llm.generate(messages, { tools: [manual] })).text);
}
