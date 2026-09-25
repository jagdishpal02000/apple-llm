/**
 * Multi-turn conversations: a message list you keep, or a named session the
 * helper keeps for you (it survives restarts).
 *
 *   npx tsx examples/chat.ts
 */
import { AppleLLM, type ChatMessage } from 'apple-llm';

const llm = new AppleLLM({ tier: 'device' });

// Stateless: you own the history. It is sent as Apple's native transcript.
const messages: ChatMessage[] = [
  { role: 'system', content: 'You are terse.' },
  { role: 'user', content: 'My cat is called Biscuit.' },
];
const reply = await llm.generate(messages);
messages.push(reply.message, { role: 'user', content: 'What is my cat called?' });
console.log(await llm.text(messages)); // "Biscuit."

// Stateful: the helper keeps the thread, and rebuilds it after a restart.
const chat = llm.conversation('example-thread', { system: 'You are terse.' });
await chat.text('My dog is called Pepper.');
console.log(await chat.text('What is my dog called?')); // "Pepper."
await chat.reset();
