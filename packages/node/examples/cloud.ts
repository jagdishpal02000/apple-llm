/**
 * The big model: Apple's server model on Private Cloud Compute — on macOS 27
 * Golden Gate, the one Siri AI is built on. Your prompt leaves the Mac, and
 * each call spends Apple's per-device quota.
 *
 *   npx apple-llm setup-cloud                # once
 *   npx apple-llm setup-cloud --web-search   # once, for webSearch
 *   npx tsx examples/cloud.ts
 */
import { z } from 'zod';
import { AppleLLM, probe } from 'apple-llm';

// What the server model can do, read from the framework without calling it.
const { cloud: info } = await probe();
console.log(info.capabilities, info.contextSize, info.quota?.status);

const cloud = new AppleLLM({ tier: 'cloud' });

const puzzle =
  'Three friends, Ana, Ben and Cy, each own one pet: a cat, a dog or a fish, and live in a red, green or blue house. ' +
  'The dog owner lives in the red house. Ana does not own the cat. Ben lives in the blue house. ' +
  'Cy does not live in the red house and does not own the fish. Who owns the fish, and what colour is their house?';
console.log(await cloud.text(puzzle));

console.log(await cloud.text('What is in this image?', { images: ['../../tests/fixtures/images/red-square-blue-circle.png'] }));

console.log(await cloud.text('Who won the most recent FIFA World Cup, and what was the score?', { webSearch: true }));

const Person = z.object({ name: z.string(), born: z.number().int() });
console.log(await cloud.json('Ada Lovelace was born in 1815.', { schema: Person }));
