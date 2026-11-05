/**
 * A voice assistant in a dozen lines, entirely on your Mac: the microphone to
 * text, text to Apple's on-device model, its reply to speech.
 *
 *   npx tsx examples/voice-assistant.ts
 *
 * Say "goodbye" to stop. Use headphones, or it will hear itself.
 */
import { AppleLLM, listen, speak, type ChatMessage } from 'apple-llm';

const llm = new AppleLLM({ tier: 'device' });
const history: ChatMessage[] = [
  { role: 'system', content: 'You are a helpful voice assistant. Answer in one or two short sentences.' },
];

await speak('Hello. What would you like to know?');
for (;;) {
  const said = await listen({ silence: 1.5, onListening: () => console.log('listening…') });
  if (said === '') continue;
  console.log(`you: ${said}`);
  if (/\bgoodbye\b/i.test(said)) break;

  history.push({ role: 'user', content: said });
  const { text, message } = await llm.generate(history);
  history.push(message);
  console.log(`mac: ${text}`);
  await speak(text);
}
await speak('Goodbye.');
llm.close();
