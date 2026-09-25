/**
 * The OpenAI-compatible server, started from code and called with the
 * official `openai` client. From a shell, `npx apple-llm serve` does the same.
 *
 *   npm install openai
 *   npx tsx examples/openai-server.ts
 */
import OpenAI from 'openai';
import { serve } from 'apple-llm/server';

const server = await serve({ port: 0 }); // any free port; 11436 by default
const openai = new OpenAI({ baseURL: server.url, apiKey: 'unused' });

const completion = await openai.chat.completions.create({
  model: 'apple-on-device',
  messages: [{ role: 'user', content: 'Say hello in French.' }],
});
console.log(completion.choices[0].message.content, completion.usage);

const stream = await openai.chat.completions.create({
  model: 'apple-on-device',
  messages: [{ role: 'user', content: 'Count to five.' }],
  stream: true,
});
for await (const chunk of stream) process.stdout.write(chunk.choices[0]?.delta?.content ?? '');
process.stdout.write('\n');

await server.close();
