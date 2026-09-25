/**
 * Structured output, typed by a Zod schema — then the same, streamed.
 *
 *   npx tsx examples/structured.ts
 */
import { z } from 'zod';
import { AppleLLM } from 'apple-llm';

const llm = new AppleLLM();

const Contact = z.object({
  name: z.string(),
  email: z.string().email(),
  phone: z.string().nullable().describe('Phone number, or null when none is given'),
  tags: z.array(z.string()).max(3),
});

// On device the shape is guaranteed by constrained decoding; Zod then checks
// what the decoder cannot (.email()) and types the result.
const contact = await llm.json('Reach Grace Hopper at grace@navy.mil. Admiral, computing pioneer.', {
  schema: Contact,
});
console.log(contact.name, contact.email, contact.phone);

// Partial objects as they fill in: render a card before the reply is done.
for await (const partial of llm.streamJson('Ada Lovelace, ada@analytical.org, mathematician.', { schema: Contact })) {
  console.log('partial:', partial);
}
