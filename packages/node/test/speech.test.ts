/**
 * Speech against a scripted helper, so all of it runs on any machine. The real
 * frameworks are exercised by live-speech.test.ts.
 */
import { existsSync, readFileSync, writeFileSync } from 'node:fs';
import { mkdtemp, rm } from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import { pathToFileURL } from 'node:url';
import { afterAll, beforeAll, describe, expect, it } from 'vitest';
import { AppleSpeech, toSRT, toVTT, type Transcript } from '../src/speech.js';
import { sniffAudioExtension } from '../src/speech.js';
import {
  AbortError,
  AppleLLMError,
  PermissionDeniedError,
  SetupRequiredError,
  TimeoutError,
  UnsupportedError,
} from '../src/errors.js';
import { parseMultipart, serve } from '../src/server.js';
import { fakeServer, tick, type FakeChild } from './fake-helper.js';

// Each scripted helper registers an exit handler, as a real one does, and these
// tests make far more of them than any program would.
process.setMaxListeners(100);

type Handler = (message: Record<string, unknown>, child: FakeChild) => void;

function speechWith(handler: Handler, timeoutMs = 500): { speech: AppleSpeech; children: FakeChild[] } {
  const { server, children } = fakeServer((message, child) => {
    // Control lines get no reply of their own.
    if (message.op === 'cancel' || message.op === 'endInput') return;
    handler(message, child);
  }, timeoutMs);
  return { speech: new AppleSpeech({ server }), children };
}

const VOICE = { id: 'com.apple.voice.compact.en-US.Samantha', name: 'Samantha', language: 'en-US', engine: 'avspeech' };

const TRANSCRIPT = {
  ok: true,
  text: 'Hello there.',
  segments: [{ text: 'Hello there.', start: 0, end: 1.2, confidence: 0.9 }],
  language: 'en-US',
  engine: 'transcriber',
  duration: 1.5,
};

describe('speak', () => {
  it('sends only what was asked for and reports what spoke', async () => {
    const { speech, children } = speechWith((_message, child) => child.reply({ ok: true, spoken: true, duration: 2.5, voice: VOICE }));
    const result = await speech.speak('Hello', { voice: 'Samantha', speed: 1.5 });
    expect(result).toEqual({ duration: 2.5, voice: VOICE });
    const request = children[0].requests[0];
    expect(request).toMatchObject({ op: 'speak', text: 'Hello', voice: 'Samantha', speed: 1.5 });
    expect(request).not.toHaveProperty('pitch');
    expect(request).not.toHaveProperty('words');
    expect(typeof request.id).toBe('string');
  });

  it('asks for words only when someone is listening for them, and delivers them', async () => {
    const { speech, children } = speechWith((_message, child) => {
      child.reply({ ok: true, done: false, word: { text: 'Hello', start: 0.4, offset: 0, length: 5 } });
      child.reply({ ok: true, done: false, word: { text: 'there', start: 0.8, offset: 6, length: 5 } });
      child.reply({ ok: true, spoken: true, duration: 1, voice: VOICE });
    });
    const words: string[] = [];
    await speech.speak('Hello there', { onWord: (word) => words.push(word.text) });
    expect(words).toEqual(['Hello', 'there']);
    expect(children[0].requests[0].words).toBe(true);
  });

  it('turns an abort into a cancel line for that request, and an AbortError', async () => {
    const { speech, children } = speechWith(() => undefined);
    const controller = new AbortController();
    const speaking = speech.speak('A very long story', { signal: controller.signal });
    await tick();
    controller.abort();
    await expect(speaking).rejects.toBeInstanceOf(AbortError);
    expect(children[0].controls('cancel')[0].id).toBe(children[0].requests[0].id);
    children[0].reply({ ok: false, kind: 'cancelled', error: 'the request was cancelled' });
  });
});

/**
 * "Long audio" is really a claim about this: the wedge timer measures silence,
 * and a 30 minute read-aloud is only alive to it through heartbeats.
 */
describe('long operations', () => {
  it('survives far longer than the silence timeout while heartbeats keep coming', async () => {
    const { speech } = speechWith((_message, child) => {
      let beats = 0;
      const timer = setInterval(() => {
        beats += 1;
        if (beats < 8) {
          child.reply({ ok: true, heartbeat: true, done: false });
          return;
        }
        clearInterval(timer);
        child.reply({ ok: true, spoken: true, duration: 1800, voice: VOICE });
      }, 40);
    }, 100);
    await expect(speech.speak('thirty minutes of text')).resolves.toMatchObject({ duration: 1800 });
  });

  it('still catches a helper that goes quiet', async () => {
    const { speech } = speechWith(() => undefined, 60);
    await expect(speech.speak('Hello')).rejects.toBeInstanceOf(TimeoutError);
  });

  it('does not hand heartbeats to any callback', async () => {
    const { speech } = speechWith((_message, child) => {
      child.reply({ ok: true, heartbeat: true, done: false });
      child.reply({ ok: true, done: false, segment: { text: 'One.', start: 0, end: 1 } });
      child.reply(TRANSCRIPT);
    });
    const directory = await mkdtemp(path.join(os.tmpdir(), 'apple-llm-test-'));
    const file = path.join(directory, 'a.wav');
    writeFileSync(file, 'x');
    const segments: string[] = [];
    await speech.transcribe(file, { onSegment: (segment) => segments.push(segment.text) });
    expect(segments).toEqual(['One.']);
    await rm(directory, { recursive: true, force: true });
  });
});

describe('synthesize', () => {
  it('saves to the path given, with ~ expanded, and leaves the format to the helper', async () => {
    const { speech, children } = speechWith((message, child) =>
      child.reply({ ok: true, path: message.out, format: 'm4a', duration: 2, sampleRate: 22050, bytes: 100, voice: VOICE }),
    );
    const saved = await speech.synthesize('Hello', { out: '~/speech.m4a' });
    expect(saved.path).toBe(path.join(os.homedir(), 'speech.m4a'));
    expect(saved).toMatchObject({ format: 'm4a', duration: 2, bytes: 100 });
    expect(saved).not.toHaveProperty('audio');
    expect(children[0].requests[0]).not.toHaveProperty('format');
  });

  it('returns a Buffer without one, and cleans up its temp file', async () => {
    let written = '';
    const { speech, children } = speechWith((message, child) => {
      written = message.out as string;
      writeFileSync(written, 'RIFFaudio');
      child.reply({ ok: true, path: written, format: 'wav', duration: 2, sampleRate: 22050, bytes: 9, voice: VOICE, words: [{ text: 'Hello', start: 0, end: 1, offset: 0, length: 5 }] });
    });
    const made = await speech.synthesize('Hello', { words: true, sampleRate: 16000 });
    expect(made.audio.toString()).toBe('RIFFaudio');
    expect(made.words).toHaveLength(1);
    expect(made).not.toHaveProperty('path');
    expect(children[0].requests[0]).toMatchObject({ format: 'wav', words: true, sampleRate: 16000 });
    expect(existsSync(written)).toBe(false);
    expect(existsSync(path.dirname(written))).toBe(false);
  });

  it('cleans up when the helper fails too', async () => {
    let written = '';
    const { speech } = speechWith((message, child) => {
      written = message.out as string;
      child.reply({ ok: false, kind: 'unsupported', error: 'macOS can read MP3 but has no encoder.' });
    });
    await expect(speech.synthesize('Hello', { format: 'mp3' as never })).rejects.toBeInstanceOf(UnsupportedError);
    expect(existsSync(path.dirname(written))).toBe(false);
  });
});

describe('errors', () => {
  const failing = (reply: Record<string, unknown>): Promise<unknown> =>
    speechWith((_message, child) => child.reply({ ok: false, ...reply })).speech.speak('Hello');

  it('maps every failure kind to a typed error', async () => {
    await expect(failing({ kind: 'unsupported', error: 'Unknown voice' })).rejects.toBeInstanceOf(UnsupportedError);
    await expect(failing({ kind: 'permission', error: 'No microphone access' })).rejects.toBeInstanceOf(PermissionDeniedError);
    await expect(failing({ kind: 'cancelled', error: 'cancelled' })).rejects.toBeInstanceOf(AbortError);
    await expect(failing({ kind: 'speech', error: 'Something broke' })).rejects.toMatchObject({ message: 'Something broke' });
    const setup = (await failing({ kind: 'setup', language: 'de-DE', error: 'Not installed' }).catch((e: unknown) => e)) as SetupRequiredError;
    expect(setup).toBeInstanceOf(SetupRequiredError);
    expect(setup.step).toBe('install-speech-language');
    expect(setup.code).toBe('SETUP_REQUIRED');
  });

  it('gives permission failures their own stable code', async () => {
    const error = (await failing({ kind: 'permission', error: 'No' }).catch((e: unknown) => e)) as PermissionDeniedError;
    expect(error.code).toBe('PERMISSION_DENIED');
    expect(error).toBeInstanceOf(AppleLLMError);
  });
});

describe('transcribe', () => {
  let directory: string;
  beforeAll(async () => {
    directory = await mkdtemp(path.join(os.tmpdir(), 'apple-llm-test-'));
    writeFileSync(path.join(directory, 'a.wav'), 'x');
  });
  afterAll(() => rm(directory, { recursive: true, force: true }));

  it('passes a path through and returns the transcript', async () => {
    const { speech, children } = speechWith((_message, child) => child.reply(TRANSCRIPT));
    const file = path.join(directory, 'a.wav');
    const result = await speech.transcribe(file, { language: 'en-US', hints: ['Kubernetes'], words: true, download: false });
    expect(result.text).toBe('Hello there.');
    expect(result.segments[0].confidence).toBe(0.9);
    expect(children[0].requests[0]).toMatchObject({
      op: 'transcribe', path: file, language: 'en-US', hints: ['Kubernetes'], words: true, download: false,
    });
  });

  it('accepts a file URL and refuses a remote one', async () => {
    const { speech, children } = speechWith((_message, child) => child.reply(TRANSCRIPT));
    const file = path.join(directory, 'a.wav');
    await speech.transcribe(pathToFileURL(file));
    expect(children[0].requests[0].path).toBe(file);
    await expect(speech.transcribe(new URL('https://example.com/a.mp3'))).rejects.toBeInstanceOf(UnsupportedError);
  });

  it('writes in-memory audio to a temp file named for what it is, then removes it', async () => {
    let seen = '';
    let contents = '';
    const { speech } = speechWith((message, child) => {
      seen = message.path as string;
      contents = readFileSync(seen, 'latin1');
      child.reply(TRANSCRIPT);
    });
    const wav = Buffer.concat([Buffer.from('RIFF'), Buffer.alloc(4), Buffer.from('WAVEdata')]);
    await speech.transcribe(wav);
    expect(path.extname(seen)).toBe('.wav');
    expect(contents).toContain('WAVEdata');
    expect(existsSync(path.dirname(seen))).toBe(false);
    await speech.transcribe(new Uint8Array(wav).buffer);
    expect(path.extname(seen)).toBe('.wav');
  });

  it('falls back on the filename when the bytes do not say', async () => {
    let seen = '';
    const { speech } = speechWith((message, child) => {
      seen = message.path as string;
      child.reply(TRANSCRIPT);
    });
    await speech.transcribe(Buffer.from('mystery'), { filename: 'call.ogg' });
    expect(path.extname(seen)).toBe('.ogg');
  });

  it('does not show a caller who gave it bytes a temp path in an error', async () => {
    const { speech } = speechWith((message, child) =>
      child.reply({ ok: false, kind: 'unsupported', error: `macOS could not read audio from ${String(message.path)}.` }),
    );
    const error = (await speech.transcribe(Buffer.from('mystery')).catch((e: unknown) => e)) as Error;
    expect(error).toBeInstanceOf(UnsupportedError);
    expect(error.message).toBe('macOS could not read audio from these bytes.');
  });

  it('reports segments as they finish, with progress, and a model download along the way', async () => {
    const { speech } = speechWith((_message, child) => {
      child.reply({ ok: true, done: false, download: { language: 'hi-IN', fraction: 0.5 } });
      child.reply({ ok: true, done: false, started: { language: 'hi-IN', engine: 'transcriber', duration: 10 } });
      child.reply({ ok: true, done: false, segment: { text: 'One.', start: 0, end: 5 } });
      child.reply({ ok: true, done: false, segment: { text: 'Two.', start: 5, end: 10 } });
      child.reply(TRANSCRIPT);
    });
    const segments: string[] = [];
    const progress: Array<[string, number | undefined]> = [];
    await speech.transcribe(path.join(directory, 'a.wav'), {
      onSegment: (segment) => segments.push(segment.text),
      onProgress: (p) => progress.push([p.status, p.percent]),
    });
    expect(segments).toEqual(['One.', 'Two.']);
    expect(progress).toEqual([
      ['downloading the hi-IN speech model (once)', 50],
      ['transcribing', 50],
      ['transcribing', 100],
    ]);
  });

  it('streams: iterate the segments, await the text, read the rest off .result', async () => {
    const { speech } = speechWith((_message, child) => {
      child.reply({ ok: true, done: false, segment: { text: 'One.', start: 0, end: 1 } });
      child.reply({ ok: true, done: false, segment: { text: 'Two.', start: 1, end: 2 } });
      child.reply(TRANSCRIPT);
    });
    const stream = speech.transcribeStream(path.join(directory, 'a.wav'));
    const seen: string[] = [];
    for await (const segment of stream) seen.push(segment.text);
    expect(seen).toEqual(['One.', 'Two.']);
    expect(await stream).toBe('Hello there.');
    expect((await stream.result).language).toBe('en-US');
  });

  it('stops the work when a reader breaks out of the loop', async () => {
    const { speech, children } = speechWith((_message, child) => {
      child.reply({ ok: true, done: false, segment: { text: 'One.', start: 0, end: 1 } });
    });
    for await (const _segment of speech.transcribeStream(path.join(directory, 'a.wav'))) break;
    await tick();
    expect(children[0].controls('cancel')).toHaveLength(1);
    children[0].reply({ ok: false, kind: 'cancelled', error: 'cancelled' });
  });
});

describe('listen', () => {
  const heard = (text: string, final: boolean, start = 0) => ({
    ok: true, done: false, transcript: { text, final, start, end: start + 1 },
  });

  it('stops at a pause by default, and awaits to the text', async () => {
    const { speech, children } = speechWith((_message, child) => {
      child.reply({ ok: true, done: false, listening: true, language: 'en-US' });
      child.reply(heard('Hel', false));
      child.reply(heard('Hello there.', true));
      child.reply(TRANSCRIPT);
    });
    const opened: boolean[] = [];
    await expect(speech.listen({ onListening: () => opened.push(true) })).resolves.toBe('Hello there.');
    expect(opened).toEqual([true]);
    expect(children[0].requests[0]).toMatchObject({ op: 'listen', silence: 2 });
  });

  it('runs until stopped when continuous, and stop() keeps what it heard', async () => {
    const { speech, children } = speechWith((message, child) => {
      if (message.op === 'listen') child.reply({ ok: true, done: false, listening: true, language: 'en-US' });
    });
    const session = speech.listen({ continuous: true });
    await tick();
    expect(children[0].requests[0]).not.toHaveProperty('silence');
    children[0].reply(heard('Hello there.', true));
    const iterator = session[Symbol.asyncIterator]();
    expect((await iterator.next()).value).toMatchObject({ text: 'Hello there.', final: true });
    session.stop();
    await tick();
    expect(children[0].controls('endInput')).toHaveLength(1);
    children[0].reply(TRANSCRIPT);
    expect((await session.result).text).toBe('Hello there.');
    expect(children[0].controls('cancel')).toHaveLength(0);
  });

  it('honours a stop() that came before the microphone opened', async () => {
    const { speech, children } = speechWith(() => undefined);
    const session = speech.listen({ continuous: true });
    session.stop();
    await tick();
    expect(children[0].controls('endInput')).toHaveLength(0);
    children[0].reply({ ok: true, done: false, listening: true, language: 'en-US' });
    await tick();
    expect(children[0].controls('endInput')).toHaveLength(1);
    children[0].reply(TRANSCRIPT);
    await expect(session).resolves.toBe('Hello there.');
  });

  it('abort() discards it', async () => {
    const { speech, children } = speechWith((_message, child) => {
      child.reply({ ok: true, done: false, listening: true, language: 'en-US' });
    });
    const session = speech.listen({ continuous: true });
    await tick();
    session.abort();
    await expect(session).rejects.toBeInstanceOf(AbortError);
    expect(children[0].controls('cancel')).toHaveLength(1);
    children[0].reply({ ok: false, kind: 'cancelled', error: 'cancelled' });
  });

  it('reports a refused microphone as a permission error', async () => {
    const { speech } = speechWith((_message, child) => child.reply({ ok: false, kind: 'permission', error: 'Allow your terminal app.' }));
    await expect(speech.listen()).rejects.toBeInstanceOf(PermissionDeniedError);
  });
});

describe('voices and languages', () => {
  const voices = [
    { id: 'c', name: 'Thomas', language: 'fr-FR', quality: 'default', gender: 'male', novelty: false, personal: false, default: false, engine: 'avspeech' },
    { id: 'a', name: 'Samantha', language: 'en-US', quality: 'enhanced', gender: 'female', novelty: false, personal: false, default: true, engine: 'avspeech' },
    { id: 'b', name: 'Bells', language: 'en-US', quality: 'default', gender: 'unspecified', novelty: true, personal: false, default: false, engine: 'avspeech' },
    { id: 'd', name: 'Aman', language: 'en-IN', quality: 'default', gender: 'male', novelty: false, personal: false, default: false, engine: 'say' },
  ];
  const listing = (): AppleSpeech =>
    speechWith((_message, child) =>
      child.reply({ ok: true, voices, systemVoice: { id: 'a', name: 'Samantha', language: 'en-US', siri: false, listed: true } }),
    ).speech;

  it('sorts them, and says which is the default', async () => {
    const all = await listing().voices();
    expect(all.map((voice) => voice.name)).toEqual(['Aman', 'Bells', 'Samantha', 'Thomas']);
    expect(all.find((voice) => voice.isDefault)?.name).toBe('Samantha');
    expect(all[0]).not.toHaveProperty('default');
  });

  it('filters by language family or exact tag, and can hide novelties', async () => {
    const speech = listing();
    expect((await speech.voices({ language: 'en' })).map((voice) => voice.name)).toEqual(['Aman', 'Bells', 'Samantha']);
    expect((await speech.voices({ language: 'en_IN' })).map((voice) => voice.name)).toEqual(['Aman']);
    expect((await speech.voices({ language: 'en', novelty: false })).map((voice) => voice.name)).toEqual(['Aman', 'Samantha']);
    expect(await speech.voices({ language: 'de' })).toEqual([]);
  });

  it('reports the system voice', async () => {
    await expect(listing().systemVoice()).resolves.toMatchObject({ name: 'Samantha', siri: false });
  });

  it('lists, installs and removes recognition languages', async () => {
    const { speech, children } = speechWith((message, child) => {
      if (message.op === 'speechLanguages') child.reply({ ok: true, languages: [{ id: 'hi-IN', name: 'Hindi', installed: false, engine: 'transcriber' }], default: 'en-US' });
      if (message.op === 'installSpeechLanguage') {
        child.reply({ ok: true, done: false, download: { language: 'hi-IN', fraction: 0.25 } });
        child.reply({ ok: true, language: 'hi-IN', engine: 'transcriber' });
      }
      if (message.op === 'removeSpeechLanguage') child.reply({ ok: true, removed: true });
    });
    await expect(speech.languages()).resolves.toMatchObject({ default: 'en-US', languages: [{ id: 'hi-IN', installed: false }] });
    const progress: Array<number | undefined> = [];
    await expect(speech.installLanguage('hi', { onProgress: (p) => progress.push(p.percent) })).resolves.toBe('hi-IN');
    expect(progress).toEqual([25]);
    await expect(speech.removeLanguage('hi-IN')).resolves.toBe(true);
    expect(children[0].requests.map((r) => r.language)).toEqual([undefined, 'hi', 'hi-IN']);
  });
});

describe('subtitles', () => {
  const words = (texts: string[], step = 0.5) => texts.map((text, i) => ({ text, start: i * step, end: (i + 1) * step }));
  const transcript = (segments: Transcript['segments'], language = 'en-US'): Transcript => ({
    text: '', segments, language, engine: 'transcriber', duration: 0,
  });

  it('writes SRT and VTT with their own timestamp styles', () => {
    const t = transcript([{ text: 'Hello there.', start: 1.5, end: 3.25 }, { text: 'Bye.', start: 3661.001, end: 3662 }]);
    expect(toSRT(t)).toBe('1\n00:00:01,500 --> 00:00:03,250\nHello there.\n\n2\n01:01:01,001 --> 01:01:02,000\nBye.\n');
    expect(toVTT(t)).toBe('WEBVTT\n\n00:00:01.500 --> 00:00:03.250\nHello there.\n\n01:01:01.001 --> 01:01:02.000\nBye.\n');
  });

  it('breaks a long sentence into cues short enough to read, using word timings', () => {
    const long = Array.from({ length: 40 }, (_, i) => `word${i}`);
    const srt = toSRT(transcript([{ text: long.join(' '), start: 0, end: 20, words: words(long) }]));
    const cues = srt.trim().split('\n\n');
    expect(cues.length).toBeGreaterThan(2);
    for (const cue of cues) expect(cue.split('\n')[2].length).toBeLessThanOrEqual(84);
    expect(cues.map((cue) => cue.split('\n')[2]).join(' ')).toBe(long.join(' '));
  });

  it('breaks on time as well as length', () => {
    const slow = ['one', 'two', 'three'];
    const cues = toVTT(transcript([{ text: 'one two three', start: 0, end: 15, words: words(slow, 5) }])).trim().split('\n\n').slice(1);
    expect(cues).toHaveLength(3);
  });

  it('does not put spaces between words in languages that use none', () => {
    const t = transcript([{ text: '你好世界', start: 0, end: 1, words: words(['你好', '世界']) }], 'zh-CN');
    expect(toSRT(t)).toContain('\n你好世界\n');
  });

  it('skips empty segments and keeps a segment that has no word timings whole', () => {
    const t = transcript([{ text: '', start: 0, end: 1 }, { text: 'Whole sentence.', start: 1, end: 2 }]);
    expect(toSRT(t)).toBe('1\n00:00:01,000 --> 00:00:02,000\nWhole sentence.\n');
  });
});

describe('recognising audio by its bytes', () => {
  const bytes = (text: string, extra: number[] = []): Uint8Array => Uint8Array.from([...Buffer.from(text, 'latin1'), ...extra]);

  it.each([
    ['RIFF\0\0\0\0WAVE', '.wav'],
    ['FORM\0\0\0\0AIFF', '.aiff'],
    ['FORM\0\0\0\0AIFC', '.aiff'],
    ['caff', '.caf'],
    ['fLaC', '.flac'],
    ['\0\0\0\x20ftypM4A ', '.m4a'],
    ['ID3\x04', '.mp3'],
  ])('%j is %s', (magic, extension) => {
    expect(sniffAudioExtension(bytes(magic))).toBe(extension);
  });

  it('tells MP3 from ADTS AAC, which share a sync word', () => {
    expect(sniffAudioExtension(Uint8Array.from([0xff, 0xfb, 0x90]))).toBe('.mp3');
    expect(sniffAudioExtension(Uint8Array.from([0xff, 0xf1, 0x50]))).toBe('.aac');
  });

  it('does not guess at anything else', () => {
    expect(sniffAudioExtension(bytes('hello world!'))).toBeUndefined();
    expect(sniffAudioExtension(new Uint8Array())).toBeUndefined();
  });
});

describe('parseMultipart', () => {
  it('reads fields and binary files, keeping the bytes intact', () => {
    const binary = Buffer.from([0, 13, 10, 45, 45, 255, 1, 13, 10]);
    const body = Buffer.concat([
      Buffer.from('--abc\r\nContent-Disposition: form-data; name="model"\r\n\r\nwhisper-1\r\n'),
      Buffer.from('--abc\r\nContent-Disposition: form-data; name="file"; filename="a b.wav"\r\nContent-Type: audio/wav\r\n\r\n'),
      binary,
      Buffer.from('\r\n--abc--\r\n'),
    ]);
    const parts = parseMultipart(body, 'abc');
    expect(parts.map((part) => part.name)).toEqual(['model', 'file']);
    expect(parts[0].data.toString()).toBe('whisper-1');
    expect(parts[0].filename).toBeUndefined();
    expect(parts[1].filename).toBe('a b.wav');
    expect(parts[1].data.equals(binary)).toBe(true);
  });

  it('does not confuse filename= with name=', () => {
    const body = Buffer.from('--x\r\nContent-Disposition: form-data; filename="f.wav"; name="file"\r\n\r\nD\r\n--x--\r\n');
    expect(parseMultipart(body, 'x')[0].name).toBe('file');
  });

  it('returns nothing for a body with no parts', () => {
    expect(parseMultipart(Buffer.from('nothing here'), 'x')).toEqual([]);
    expect(parseMultipart(Buffer.from('--x--\r\n'), 'x')).toEqual([]);
  });
});

describe('the OpenAI audio routes', () => {
  type Running = Awaited<ReturnType<typeof serve>>;
  let running: Running;
  let children: FakeChild[];

  beforeAll(async () => {
    const made = speechWith((message, child) => {
      if (message.op === 'synthesize') {
        writeFileSync(message.out as string, `audio:${String(message.format)}`);
        child.reply({ ok: true, path: message.out, format: message.format, duration: 1, sampleRate: 22050, bytes: 6, voice: VOICE });
      } else if (message.op === 'voices') {
        child.reply({ ok: true, voices: [{ id: 'a', name: 'Samantha', language: 'en-US', quality: 'default', gender: 'female', novelty: false, personal: false, default: true, engine: 'avspeech' }], systemVoice: {} });
      } else if (message.op === 'transcribe') {
        child.reply({
          ok: true, text: 'Hello there.', language: 'en-US', engine: 'transcriber', duration: 1.5,
          segments: [{ text: 'Hello there.', start: 0, end: 1.2, words: [{ text: 'Hello', start: 0, end: 0.5 }, { text: 'there.', start: 0.5, end: 1.2 }] }],
        });
      }
    });
    children = made.children;
    running = await serve({ port: 0, apiKey: 'secret', speech: made.speech });
  });
  afterAll(() => running.close());

  const lastRequest = (): Record<string, unknown> => children[0].requests[children[0].requests.length - 1];
  const auth = { authorization: 'Bearer secret' };
  const speech = (body: unknown): Promise<Response> =>
    fetch(`${running.url}/audio/speech`, { method: 'POST', headers: { ...auth, 'content-type': 'application/json' }, body: JSON.stringify(body) });
  const upload = (fields: Record<string, string | string[]>): Promise<Response> => {
    const form = new FormData();
    form.append('file', new File([Buffer.from('RIFF\0\0\0\0WAVE')], 'clip.wav'));
    for (const [key, value] of Object.entries(fields)) {
      for (const item of Array.isArray(value) ? value : [value]) form.append(key, item);
    }
    return fetch(`${running.url}/audio/transcriptions`, { method: 'POST', headers: auth, body: form });
  };

  it('needs the API key like every other route', async () => {
    const res = await fetch(`${running.url}/audio/speech`, { method: 'POST', body: '{}' });
    expect(res.status).toBe(401);
  });

  it('renders speech, saying what the bytes really are', async () => {
    const res = await speech({ model: 'tts-1', input: 'Hello', voice: 'Samantha', response_format: 'flac' });
    expect(res.status).toBe(200);
    expect(res.headers.get('content-type')).toBe('audio/flac');
    expect(res.headers.get('x-apple-llm-voice')).toBe('Samantha');
    expect(await res.text()).toBe('audio:flac');
    expect(lastRequest()).toMatchObject({ op: 'synthesize', text: 'Hello', voice: 'Samantha', format: 'flac' });
  });

  it('answers MP3 and Opus with AAC, since macOS cannot encode them, and says so', async () => {
    for (const body of [{ input: 'Hi', voice: 'alloy' }, { input: 'Hi', voice: 'alloy', response_format: 'mp3' }, { input: 'Hi', voice: 'alloy', response_format: 'opus' }]) {
      const res = await speech(body);
      expect(res.headers.get('content-type')).toBe('audio/aac');
      expect(await res.text()).toBe('audio:aac');
    }
  });

  it('gives raw PCM at the 24 kHz OpenAI defines', async () => {
    await speech({ input: 'Hi', response_format: 'pcm' });
    expect(lastRequest()).toMatchObject({ format: 'pcm', sampleRate: 24000 });
  });

  it('treats OpenAI voice names as "the default", for clients that cannot change them', async () => {
    await speech({ input: 'Hi', voice: 'Alloy' });
    expect(lastRequest()).not.toHaveProperty('voice');
  });

  it('keeps speed within what OpenAI allows', async () => {
    await speech({ input: 'Hi', speed: 9 });
    expect(lastRequest().speed).toBe(4);
    await speech({ input: 'Hi', speed: 0.1 });
    expect(lastRequest().speed).toBe(0.25);
  });

  it('refuses a bad speech request with an OpenAI-style error', async () => {
    const empty = await speech({ input: '   ' });
    expect(empty.status).toBe(400);
    expect(((await empty.json()) as { error: { message: string } }).error.message).toMatch(/input/);
    expect((await speech({ input: 'Hi', response_format: 'ogg' })).status).toBe(400);
  });

  it('lists voices', async () => {
    const res = await fetch(`${running.url}/audio/voices`, { headers: auth });
    expect(((await res.json()) as { data: Array<{ name: string }> }).data[0].name).toBe('Samantha');
  });

  it('transcribes an upload, with the language and prompt carried over', async () => {
    const res = await upload({ model: 'whisper-1', language: 'en', prompt: 'Kubernetes, Postgres\nRedis' });
    expect(await res.json()).toEqual({ text: 'Hello there.' });
    expect(lastRequest()).toMatchObject({ op: 'transcribe', language: 'en', hints: ['Kubernetes', 'Postgres', 'Redis'] });
    expect(path.extname(lastRequest().path as string)).toBe('.wav');
  });

  it('answers in each response_format', async () => {
    expect(await (await upload({ response_format: 'text' })).text()).toBe('Hello there.');
    expect(await (await upload({ response_format: 'srt' })).text()).toContain('00:00:00,000 --> 00:00:01,200');
    expect(await (await upload({ response_format: 'vtt' })).text()).toMatch(/^WEBVTT/);
    expect((await upload({ response_format: 'nonsense' })).status).toBe(400);
  });

  it('gives verbose_json the shape OpenAI clients read, with words when asked', async () => {
    const plain = (await (await upload({ response_format: 'verbose_json' })).json()) as Record<string, unknown>;
    expect(plain).toMatchObject({ task: 'transcribe', language: 'en-US', duration: 1.5, text: 'Hello there.' });
    expect((plain.segments as Array<Record<string, unknown>>)[0]).toMatchObject({ id: 0, start: 0, end: 1.2, text: 'Hello there.' });
    expect(plain).not.toHaveProperty('words');
    const withWords = (await (await upload({ response_format: 'verbose_json', 'timestamp_granularities[]': ['word', 'segment'] })).json()) as { words: unknown[] };
    expect(withWords.words[0]).toEqual({ word: 'Hello', start: 0, end: 0.5 });
  });

  it('explains a request that is not an upload', async () => {
    const res = await fetch(`${running.url}/audio/transcriptions`, { method: 'POST', headers: { ...auth, 'content-type': 'application/json' }, body: '{}' });
    expect(res.status).toBe(400);
    expect(((await res.json()) as { error: { message: string } }).error.message).toMatch(/multipart/);
    const noFile = new FormData();
    noFile.append('model', 'whisper-1');
    expect((await fetch(`${running.url}/audio/transcriptions`, { method: 'POST', headers: auth, body: noFile })).status).toBe(400);
  });

  it('does not offer speech as a chat model', async () => {
    const models = (await (await fetch(`${running.url}/models`, { headers: auth })).json()) as { data: Array<{ id: string }> };
    expect(models.data.map((m) => m.id)).toEqual(['apple-on-device', 'apple-private-cloud']);
  });

  it('answers a multipart body with no parts with a 400', async () => {
    const res = await fetch(`${running.url}/audio/transcriptions`, {
      method: 'POST',
      headers: { ...auth, 'content-type': 'multipart/form-data; boundary=x' },
      body: '--x--\r\n',
    });
    expect(res.status).toBe(400);
  });
});
