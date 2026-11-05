/**
 * Speech against the real frameworks: `APPLE_LLM_LIVE=1 vitest run`.
 *
 * Silent: nothing plays out loud (read-alouds run at volume 0), and nothing here
 * opens the microphone, so it can run anywhere. Everything is on device.
 */
import { randomUUID } from 'node:crypto';
import { existsSync, statSync } from 'node:fs';
import { copyFile, mkdtemp, readdir, rm } from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import OpenAI, { toFile } from 'openai';
import { afterAll, beforeAll, describe, expect, it } from 'vitest';
import { ensureBinary } from '../src/compile.js';
import { AbortError, SetupRequiredError, UnsupportedError } from '../src/errors.js';
import { HelperServer } from '../src/protocol.js';
import { serve } from '../src/server.js';
import { AppleSpeech, toSRT, type AudioFormat } from '../src/speech.js';

const live = process.env.APPLE_LLM_LIVE === '1';
const SENTENCE = 'The quick brown fox jumps over the lazy dog.';
const words = (text: string): string[] => text.toLowerCase().match(/[a-z0-9']+/g) ?? [];

/** Word error rate, for "did it hear what was said" without demanding every word. */
function errorRate(said: string, heard: string): number {
  const a = words(said);
  const b = words(heard);
  let previous = b.map((_, i) => i + 1);
  previous.unshift(0);
  for (let i = 1; i <= a.length; i += 1) {
    const current = [i];
    for (let j = 1; j <= b.length; j += 1) {
      current[j] = Math.min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (a[i - 1] === b[j - 1] ? 0 : 1));
    }
    previous = current;
  }
  return previous[b.length] / a.length;
}

describe.skipIf(!live)('speech, live', () => {
  const speech = new AppleSpeech();
  let directory: string;
  const file = (name: string): string => path.join(directory, name);

  beforeAll(async () => {
    directory = await mkdtemp(path.join(os.tmpdir(), 'apple-llm-live-'));
  });
  afterAll(async () => {
    speech.close();
    await rm(directory, { recursive: true, force: true });
  });

  it('lists voices, with exactly one System voice', async () => {
    const voices = await speech.voices();
    expect(voices.length).toBeGreaterThan(10);
    expect(voices.filter((voice) => voice.isDefault)).toHaveLength(1);
    expect(new Set(voices.map((voice) => voice.id)).size).toBe(voices.length);
    const system = await speech.systemVoice();
    expect(system.listed).toBe(true);
    expect(voices.find((voice) => voice.isDefault)?.name).toBe(system.name);
  });

  it('writes every format, and each one reads back as the same speech', async () => {
    const formats: AudioFormat[] = ['wav', 'aiff', 'caf', 'm4a', 'aac', 'flac'];
    for (const format of formats) {
      const out = file(`round.${format}`);
      const saved = await speech.synthesize(SENTENCE, { voice: 'Samantha', out });
      expect(saved.format).toBe(format);
      expect(saved.bytes).toBe(statSync(out).size);
      const { text, duration } = await speech.transcribe(out, { language: 'en-US' });
      expect(errorRate(SENTENCE, text), `${format}: heard "${text}"`).toBeLessThan(0.25);
      expect(Math.abs(duration - saved.duration), format).toBeLessThan(0.2);
    }
  }, 60_000);

  it('gives raw PCM of exactly the length it says', async () => {
    const made = await speech.synthesize(SENTENCE, { voice: 'Samantha', format: 'pcm', sampleRate: 24000 });
    expect(made.sampleRate).toBe(24000);
    expect(made.audio.length / 2 / 24000).toBeCloseTo(made.duration, 1);
  });

  it.each([8000, 16000, 44100, 48000])('resamples to %i Hz without changing the length', async (rate) => {
    const base = await speech.synthesize(SENTENCE, { voice: 'Samantha', format: 'wav' });
    const made = await speech.synthesize(SENTENCE, { voice: 'Samantha', format: 'm4a', sampleRate: rate });
    expect(made.sampleRate).toBe(rate);
    expect(Math.abs(made.duration - base.duration)).toBeLessThan(0.1);
  });

  it('refuses formats macOS cannot encode, and leaves nothing behind', async () => {
    await expect(speech.synthesize(SENTENCE, { out: file('nope.mp3') })).rejects.toBeInstanceOf(UnsupportedError);
    expect(existsSync(file('nope.mp3'))).toBe(false);
  });

  it('makes speed mean what it says', async () => {
    const normal = await speech.synthesize(SENTENCE, { voice: 'Samantha' });
    const double = await speech.synthesize(SENTENCE, { voice: 'Samantha', speed: 2 });
    const half = await speech.synthesize(SENTENCE, { voice: 'Samantha', speed: 0.6 });
    expect(normal.duration / double.duration).toBeGreaterThan(1.7);
    expect(normal.duration / double.duration).toBeLessThan(2.4);
    expect(half.duration / normal.duration).toBeGreaterThan(1.4);
  });

  it('applies volume to rendered audio', async () => {
    const peak = (audio: Buffer): number => {
      // macOS pads the header to 4 KB, so the samples are not at the usual byte 44.
      let max = 0;
      for (let i = audio.indexOf('data') + 8; i + 1 < audio.length; i += 2) max = Math.max(max, Math.abs(audio.readInt16LE(i)));
      return max;
    };
    const loud = await speech.synthesize(SENTENCE, { voice: 'Samantha', format: 'wav' });
    const quiet = await speech.synthesize(SENTENCE, { voice: 'Samantha', format: 'wav', volume: 0.1 });
    expect(peak(quiet.audio)).toBeLessThan(peak(loud.audio) / 3);
  });

  it('refuses an unknown voice, and suggests a close one', async () => {
    const error = (await speech.synthesize(SENTENCE, { voice: 'Samanth' }).catch((e: unknown) => e)) as Error;
    expect(error).toBeInstanceOf(UnsupportedError);
    expect(error.message).toMatch(/Samantha/);
  });

  it('finds a voice by language', async () => {
    const french = await speech.synthesize('Bonjour tout le monde.', { language: 'fr' });
    expect(french.voice.language).toMatch(/^fr/);
    const named = await speech.synthesize('Guten Tag.', { voice: 'Eddy', language: 'de' });
    expect(named.voice).toMatchObject({ name: 'Eddy', language: 'de-DE' });
  });

  it('reaches voices AVSpeech does not list, through say', async () => {
    const voices = await speech.voices();
    const sayOnly = voices.find((voice) => voice.engine === 'say');
    if (sayOnly === undefined) return; // Which voices are hidden varies by macOS.
    const made = await speech.synthesize('Hello there.', { voice: sayOnly.name, format: 'm4a' });
    expect(made.voice.engine).toBe('say');
    expect(made.duration).toBeGreaterThan(0.2);
    await expect(speech.synthesize('Hello.', { voice: sayOnly.name, pitch: 1.5 })).rejects.toBeInstanceOf(UnsupportedError);
  });

  it('goes through say for voice "system", matching the System voice', async () => {
    const system = await speech.systemVoice();
    const made = await speech.synthesize('Hello there.', { voice: 'system' });
    expect(made.voice).toMatchObject({ id: 'system', name: system.name, engine: 'say' });
    expect(made.duration).toBeGreaterThan(0.2);
  });

  it('understands SSML, and rejects broken SSML', async () => {
    const plain = await speech.synthesize('Hello world.', { voice: 'Samantha' });
    const paused = await speech.synthesize('<speak>Hello <break time="1s"/> world.</speak>', { voice: 'Samantha', ssml: true });
    expect(paused.duration).toBeGreaterThan(plain.duration + 0.5);
    await expect(speech.synthesize('<speak>Hello', { voice: 'Samantha', ssml: true })).rejects.toBeInstanceOf(UnsupportedError);
  });

  /**
   * Two bugs this guards: a zero-length buffer means the end of a segment, not
   * of the text, and stopping there truncated audio to its first sentences; and
   * marker offsets restart with each segment, which made the last word of a
   * 30 minute render claim to be at 11.9 seconds.
   */
  it('renders long text whole, with word timings correct to the end', async () => {
    const paragraph = 'First a sentence to make the engine work. Then another one, a little longer, so that it has more to do. Third!';
    const text = Array.from({ length: 120 }, () => paragraph).join('\n\n');
    const out = file('long.m4a');
    const saved = await speech.synthesize(text, { voice: 'Samantha', out, words: true });
    // About 8 seconds a paragraph; well past any "few seconds" a bug would stop at.
    expect(saved.duration).toBeGreaterThan(600);
    const timings = saved.words ?? [];
    expect(timings).toHaveLength(words(text).length);
    for (let i = 1; i < timings.length; i += 1) expect(timings[i].start).toBeGreaterThanOrEqual(timings[i - 1].start);
    expect(timings[timings.length - 1].start).toBeGreaterThan(saved.duration - 5);
    expect(text.slice(timings[100].offset, timings[100].offset + timings[100].length)).toBe(timings[100].text);

    const heard = await speech.transcribe(out, { language: 'en-US' });
    expect(Math.abs(heard.duration - saved.duration)).toBeLessThan(1);
    expect(errorRate(text, heard.text)).toBeLessThan(0.25);
  }, 120_000);

  it('speaks in real time at volume 0, with word events', async () => {
    const seen: string[] = [];
    const result = await speech.speak(SENTENCE, { voice: 'Samantha', volume: 0, onWord: (word) => seen.push(word.text) });
    expect(seen.join(' ')).toBe(SENTENCE);
    expect(result.duration).toBeGreaterThan(1.5);
  }, 20_000);

  /**
   * A cancelled speak followed by another one killed the helper with SIGSEGV in
   * TextToSpeech, because a delegate had been freed under a queued callback.
   */
  it('stops on abort, and the next request is unaffected, however it was timed', async () => {
    for (const delay of [0, 50, 200, 600]) {
      const controller = new AbortController();
      setTimeout(() => controller.abort(), delay);
      await expect(speech.speak(`${SENTENCE} ${SENTENCE}`, { voice: 'Samantha', volume: 0, signal: controller.signal })).rejects.toBeInstanceOf(AbortError);
      await expect(speech.speak('Done.', { voice: 'Samantha', volume: 0 })).resolves.toMatchObject({ voice: { name: 'Samantha' } });
      const render = new AbortController();
      setTimeout(() => render.abort(), delay);
      await expect(speech.synthesize(Array(300).fill(SENTENCE).join(' '), { voice: 'Samantha', out: file('stopped.wav'), signal: render.signal })).rejects.toBeInstanceOf(AbortError);
      await expect(speech.synthesize('Done.', { voice: 'Samantha' })).resolves.toMatchObject({ format: 'wav' });
    }
    expect(existsSync(file('stopped.wav'))).toBe(false);
    expect((await readdir(directory)).filter((name) => name.startsWith('.apple-llm'))).toEqual([]);
  }, 60_000);

  /**
   * Delete a running helper's file, as a disk cleaner might, and AVSpeech cancels
   * everything at once. That used to come back as success, with nothing said.
   */
  it('says so when macOS stops speech that nobody asked it to stop', async () => {
    const doomed = file('doomed-helper');
    await copyFile(await ensureBinary(), doomed);
    const server = new HelperServer(doomed, { timeoutMs: 30_000 });
    const orphaned = new AppleSpeech({ server });
    await orphaned.voices(); // Starts it, while the file still exists.
    await rm(doomed);
    await expect(orphaned.speak(SENTENCE, { voice: 'Samantha', volume: 0 })).rejects.toThrow(/before it finished/);
    await expect(orphaned.synthesize(SENTENCE, { voice: 'Samantha', format: 'wav' })).rejects.toThrow(/before it finished/);
    orphaned.close();
  }, 30_000);

  /** What stops the wedge timer killing a 30 minute read-aloud. */
  it('sends heartbeats while it speaks', async () => {
    const server = new HelperServer(await ensureBinary(), { timeoutMs: 30_000 });
    const beats: unknown[] = [];
    const id = randomUUID();
    const text = Array(12).fill(SENTENCE).join(' ');
    const reply = await server.request(JSON.stringify({ op: 'speak', id, text, voice: 'Samantha', volume: 0 }), {
      id,
      onEvent: (event) => {
        if (event.heartbeat === true) beats.push(event);
      },
    });
    expect((JSON.parse(reply) as { ok: boolean }).ok).toBe(true);
    expect(beats.length).toBeGreaterThanOrEqual(1);
    server.stop();
  }, 60_000);

  it('transcribes bytes, a path and a file URL alike, with word timings and subtitles', async () => {
    const { audio } = await speech.synthesize(SENTENCE, { voice: 'Samantha', format: 'wav' });
    const fromBytes = await speech.transcribe(audio, { language: 'en-US', words: true });
    expect(errorRate(SENTENCE, fromBytes.text)).toBeLessThan(0.25);
    expect(fromBytes.engine).toBe('transcriber');
    const timed = fromBytes.segments.flatMap((segment) => segment.words ?? []);
    expect(timed.length).toBeGreaterThan(5);
    expect(timed[1].start).toBeGreaterThan(0);
    expect(toSRT(fromBytes)).toContain(' --> ');
  });

  it('streams the segments of a recording', async () => {
    const text = Array(4).fill('This is a whole sentence about nothing much at all.').join(' ');
    const { path: out } = await speech.synthesize(text, { voice: 'Samantha', out: file('segments.m4a') });
    const seen: string[] = [];
    const stream = speech.transcribeStream(out, { language: 'en-US' });
    for await (const segment of stream) seen.push(segment.text);
    expect(seen.length).toBeGreaterThan(1);
    expect(words(seen.join(' '))).toEqual(words((await stream.result).text));
  });

  it('gives a useful error for what it cannot read', async () => {
    await expect(speech.transcribe(file('missing.wav'))).rejects.toThrow(/not found/);
    await expect(speech.transcribe(import.meta.filename ?? '/etc/hosts')).rejects.toThrow(/could not read audio/);
    await expect(speech.transcribe(Buffer.from('this is not audio'))).rejects.toThrow(/these bytes/);
    await expect(speech.transcribe(file('missing.wav'), { language: 'xx-YY' })).rejects.toBeInstanceOf(UnsupportedError);
  });

  it('asks for a download instead of failing strangely when a language is missing', async () => {
    const { languages } = await speech.languages();
    const missing = languages.find((language) => !language.installed);
    if (missing === undefined) return;
    const { audio } = await speech.synthesize('Hello.', { voice: 'Samantha', format: 'wav' });
    const error = await speech.transcribe(audio, { language: missing.id, download: false }).catch((e: unknown) => e);
    expect(error).toBeInstanceOf(SetupRequiredError);
    expect(String((error as Error).message)).toContain(missing.id);
  });

  it('lists languages with a usable default', async () => {
    const { languages, default: chosen } = await speech.languages();
    expect(languages.length).toBeGreaterThan(20);
    expect(languages.map((language) => language.id)).toContain(chosen);
    expect(new Set(languages.map((language) => language.engine))).toEqual(new Set(['transcriber', 'dictation']));
  });
});

describe.skipIf(!live)('the audio routes, live, through the openai SDK', () => {
  let running: Awaited<ReturnType<typeof serve>>;
  let client: OpenAI;
  beforeAll(async () => {
    running = await serve({ port: 0 });
    client = new OpenAI({ baseURL: running.url, apiKey: 'unused' });
  });
  afterAll(() => running.close());

  it('round-trips speech through both endpoints', async () => {
    const spoken = await client.audio.speech.create({ model: 'tts-1', voice: 'alloy', input: SENTENCE });
    expect(spoken.headers.get('content-type')).toBe('audio/aac');
    const audio = Buffer.from(await spoken.arrayBuffer());
    const heard = await client.audio.transcriptions.create({ model: 'whisper-1', file: await toFile(audio, 'speech.aac'), language: 'en' });
    expect(errorRate(SENTENCE, heard.text)).toBeLessThan(0.25);
  }, 30_000);

  it('returns word timings the way verbose_json does', async () => {
    const spoken = await client.audio.speech.create({ model: 'tts-1', voice: 'Samantha' as never, input: SENTENCE, response_format: 'wav' });
    const verbose = await client.audio.transcriptions.create({
      model: 'whisper-1',
      file: await toFile(Buffer.from(await spoken.arrayBuffer()), 'speech.wav'),
      response_format: 'verbose_json',
      timestamp_granularities: ['word'],
    });
    expect(verbose.words?.length).toBeGreaterThan(5);
    expect(verbose.duration).toBeGreaterThan(1);
  }, 30_000);

  it('reports a bad voice as a 400, not a 500', async () => {
    await expect(client.audio.speech.create({ model: 'tts-1', voice: 'Nobody' as never, input: SENTENCE })).rejects.toMatchObject({ status: 400 });
  });
});
