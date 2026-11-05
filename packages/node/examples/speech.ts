/**
 * Text to speech and speech to text, on device.
 *
 *   npx tsx examples/speech.ts
 */
import { listVoices, speak, synthesize, systemVoice, toSRT, transcribe, transcribeStream } from 'apple-llm';

// Out loud, in your System voice.
await speak('Hello from your Mac.');

// Any voice, by name. `apple-llm voices` lists them.
const english = await listVoices({ language: 'en', novelty: false });
console.log(english.map((voice) => `${voice.name} (${voice.language})`).join(', '));
console.log('System voice:', (await systemVoice()).name);
await speak('This one is a different voice, and a little faster.', { voice: english[0].name, speed: 1.2 });

// To a file, in the format its extension names, however long the text is.
const saved = await synthesize('Saved to a file.', { voice: 'Samantha', out: 'speech.m4a' });
console.log(`${saved.path}: ${saved.duration.toFixed(1)}s, ${saved.bytes} bytes`);

// Or to memory, with the time of every word.
const { audio, words } = await synthesize('Every word has a time.', { voice: 'Samantha', format: 'wav', words: true });
console.log(`${audio.length} bytes`, words?.map((word) => `${word.text}@${word.start}s`).join(' '));

// Back to text: a path or the bytes, audio or video.
const transcript = await transcribe('speech.m4a', { language: 'en-US', words: true });
console.log(transcript.text);
console.log(toSRT(transcript));

// A long recording, segment by segment.
for await (const segment of transcribeStream('speech.m4a', { language: 'en-US' })) {
  console.log(`[${segment.start}s] ${segment.text}`);
}
