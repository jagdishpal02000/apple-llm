/**
 * Text to speech and speech to text, on device.
 *
 *   await speak('Hello there');
 *   const { audio } = await synthesize('Hello there', { voice: 'Samantha' });
 *   const { text } = await transcribe('meeting.m4a');
 *   const said = await listen();
 *
 * The same self-compiled Swift helper as the language model, but none of this
 * needs the model: it works with Apple Intelligence turned off, and it costs no
 * quota. Each `AppleSpeech` runs a helper process of its own, because a helper
 * handles one request at a time and a long transcription must not hold up
 * generation.
 *
 * Length is not a limit worth thinking about. Measured on an M4 Air: 60 minutes
 * of speech rendered in 40 seconds using 62 MB, and 30 minutes transcribed in
 * 28 seconds. A read-aloud as long as you like works too, because the helper
 * sends a heartbeat every 20 seconds and the wedge timer measures silence.
 */
import { randomUUID } from 'node:crypto';
import { mkdtemp, readFile, rm, writeFile } from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { ensureBinary, type OnProgress } from './compile.js';
import {
  AbortError,
  AppleLLMError,
  ModelUnavailableError,
  PermissionDeniedError,
  SetupRequiredError,
  UnsupportedError,
} from './errors.js';
import { anySignal, HelperServer, type HelperEvent } from './protocol.js';
import { ResultStream } from './stream.js';
import { isAppleSiliconMac, macosMajor } from './target.js';

/** Ceiling on helper *silence*. Twenty-second heartbeats keep a long op well inside it. */
const SILENCE_TIMEOUT_MS = 120_000;

/** The formats macOS can write. MP3, Ogg and Opus it can only read. */
export type AudioFormat = 'wav' | 'aiff' | 'caf' | 'm4a' | 'aac' | 'flac' | 'pcm';

export type VoiceQuality = 'default' | 'enhanced' | 'premium';

/** One voice installed on this Mac. */
export interface Voice {
  /** Stable identifier, e.g. `com.apple.voice.compact.en-US.Samantha`. */
  id: string;
  /** What a person calls it, e.g. `Samantha`. Accepted anywhere a voice is. */
  name: string;
  /** BCP 47 language tag, e.g. `en-US`. */
  language: string;
  /** Better voices are free downloads in System Settings; search it for "System voice". */
  quality: VoiceQuality;
  gender: 'female' | 'male' | 'unspecified';
  /** Sound effects like Bells and Zarvox rather than voices. */
  novelty: boolean;
  personal: boolean;
  /** The voice chosen in System Settings. */
  isDefault: boolean;
  /**
   * `avspeech` has word timings, pitch, volume and SSML. `say` has speed only, and
   * can lose words while the language model is busy, so a file takes two renders.
   */
  engine: 'avspeech' | 'say';
}

/** Which voice actually spoke. */
export interface VoiceInfo {
  id: string;
  name: string;
  language: string;
  engine: 'avspeech' | 'say';
}

/** The "System voice" setting. It can be a Siri voice, which is otherwise off limits to apps. */
export interface SystemVoice {
  id: string;
  name: string;
  language: string;
  siri: boolean;
  /** False when it is not in `listVoices()`, which is how a Siri voice shows up. */
  listed: boolean;
}

export interface VoiceFilter {
  /** `en` for any English, `en-GB` for one. */
  language?: string;
  /** Include sound effects like Bells and Zarvox. Default true. */
  novelty?: boolean;
}

export interface VoiceOptions {
  /**
   * A name (`Samantha`), an id from `listVoices()`, or `system` to go through
   * macOS `say` with whatever the System voice is. Default: the System voice.
   */
  voice?: string;
  /** Pick the best voice for a language, or narrow a name that several languages share. */
  language?: string;
  /** 1 is normal. Roughly 0.56 to 4 is what macOS can do; slower values are clamped. */
  speed?: number;
  /** 0.5 to 2. Needs an `avspeech` voice. */
  pitch?: number;
  /** 0 to 1. Needs an `avspeech` voice. */
  volume?: number;
  /** The text is SSML. Needs an `avspeech` voice. */
  ssml?: boolean;
  signal?: AbortSignal;
  onProgress?: OnProgress;
}

/** A word and where it falls, in seconds and in the text. */
export interface SpokenWord {
  text: string;
  start: number;
  /** Absent from live `onWord` events, which arrive as each word starts. */
  end?: number;
  /** Position in the text, in UTF-16 code units: `text.slice(offset, offset + length)`. */
  offset: number;
  length: number;
}

export interface SpeakOptions extends VoiceOptions {
  /** Called as each word starts, for highlighting. Most `avspeech` voices send these; some send none. */
  onWord?: (word: SpokenWord) => void;
}

export interface SpeakResult {
  /** Seconds it took to say, in real time. */
  duration: number;
  voice: VoiceInfo;
}

export interface SynthesizeOptions extends VoiceOptions {
  /** Default: from the extension of `out`, otherwise `wav`. */
  format?: AudioFormat;
  /** Resample to this rate. Default: the voice's own, usually 22050. */
  sampleRate?: number;
  /** Also return every word's timing. Every `avspeech` voice has these; `say` voices do not. */
  words?: boolean;
}

interface SynthesisFields {
  format: AudioFormat;
  /** Seconds of audio. */
  duration: number;
  sampleRate: number;
  bytes: number;
  voice: VoiceInfo;
  words?: SpokenWord[];
}

export interface SynthesizedSpeech extends SynthesisFields {
  /** The encoded audio file. Raw 16-bit little-endian mono samples for `pcm`. */
  audio: Buffer;
}

export interface SavedSpeech extends SynthesisFields {
  path: string;
}

/** Audio to transcribe: a path, a `file:` URL, or the bytes of an audio file. */
export type AudioInput = string | URL | Uint8Array | ArrayBuffer;

export interface RecognitionOptions {
  /**
   * `en-US`, or just `en`. Default: this Mac's language. macOS needs the language
   * named; it does not detect one.
   */
  language?: string;
  /** Words to expect that it might otherwise miss: names, jargon, product names. */
  hints?: string[];
  /** Download the language's model if it is missing. Default true; about 25 seconds. */
  download?: boolean;
  /** Include each word's timing and confidence. */
  words?: boolean;
  signal?: AbortSignal;
  onProgress?: OnProgress;
}

export interface TranscribeOptions extends RecognitionOptions {
  /** Include other readings of each segment. Breaks the text into much shorter segments. */
  alternatives?: boolean;
  /** Called with each finished segment, for long files. */
  onSegment?: (segment: TranscriptSegment) => void;
  /** A name for in-memory audio when its bytes do not say what it is, e.g. `call.mp3`. */
  filename?: string;
}

export interface TranscriptWord {
  text: string;
  start: number;
  end: number;
  /** 0 to 1. */
  confidence?: number;
}

export interface TranscriptSegment {
  text: string;
  start: number;
  end: number;
  /** 0 to 1. */
  confidence?: number;
  words?: TranscriptWord[];
  alternatives?: string[];
}

export interface Transcript {
  text: string;
  segments: TranscriptSegment[];
  /** The language it was recognised as, e.g. `en-IN`. */
  language: string;
  /** `transcriber` is Apple's better engine; `dictation` covers the languages it lacks. */
  engine: 'transcriber' | 'dictation';
  /** Seconds of audio. */
  duration: number;
}

/** What `listen()` has heard. A `final: false` event is replaced by the next one. */
export interface ListenEvent extends TranscriptSegment {
  final: boolean;
}

export interface ListenOptions extends RecognitionOptions {
  /** Seconds of quiet, after speech, that end it. Default 2. */
  silence?: number;
  /** Keep going through silence until `stop()`. */
  continuous?: boolean;
  /** Stop after this many seconds, speech or no speech. */
  maxSeconds?: number;
  /** The microphone is open: tell the speaker to start talking. */
  onListening?: () => void;
}

export interface SpeechLanguage {
  /** BCP 47 tag, e.g. `hi-IN`. */
  id: string;
  /** Its name in this Mac's language. */
  name: string;
  installed: boolean;
  engine: 'transcriber' | 'dictation';
}

export interface SpeechLanguages {
  languages: SpeechLanguage[];
  /** What `transcribe` and `listen` use when not told. */
  default: string;
}

export interface SpeechProbe {
  available: boolean;
  reason?: string;
  voices?: number;
  systemVoice?: SystemVoice;
  languages?: number;
  installedLanguages?: string[];
  defaultLanguage?: string;
}

/** Fails before compiling anything, with a reason that is about speech. */
async function assertPlatform(): Promise<void> {
  const needs = 'Speech needs macOS 26 or later on Apple Silicon';
  if (process.platform !== 'darwin') {
    throw new ModelUnavailableError(`${needs}; this machine reports "${process.platform}".`, 'unsupportedPlatform', 'device');
  }
  if (!isAppleSiliconMac()) throw new ModelUnavailableError(`${needs}; this Mac is not Apple Silicon.`, 'deviceNotEligible', 'device');
  const major = await macosMajor();
  if (major !== null && major < 26) {
    throw new ModelUnavailableError(`${needs}; this is macOS ${major}.`, 'unsupportedOSVersion', 'device');
  }
}

function raiseFor(response: Record<string, unknown>, raw: string): never {
  const detail = typeof response.error === 'string' ? response.error : raw.slice(0, 200);
  switch (response.kind) {
    case 'cancelled':
      throw new AbortError('The request was cancelled.', 'device');
    case 'unsupported':
      throw new UnsupportedError(detail, 'device');
    case 'permission':
      throw new PermissionDeniedError(detail, 'device');
    case 'setup':
      throw new SetupRequiredError(detail, 'install-speech-language', 'device');
    default:
      throw new AppleLLMError(detail, 'device');
  }
}

function resolvePath(file: string): string {
  if (file === '~') return os.homedir();
  if (file.startsWith('~/')) return path.join(os.homedir(), file.slice(2));
  return path.resolve(file);
}

/** A file extension from magic bytes, so macOS opens in-memory audio as what it is. */
export function sniffAudioExtension(bytes: Uint8Array): string | undefined {
  const text = (start: number, length: number): string => String.fromCharCode(...bytes.subarray(start, start + length));
  if (text(0, 4) === 'RIFF' && text(8, 4) === 'WAVE') return '.wav';
  if (text(0, 4) === 'FORM' && (text(8, 4) === 'AIFF' || text(8, 4) === 'AIFC')) return '.aiff';
  if (text(0, 4) === 'caff') return '.caf';
  if (text(0, 4) === 'fLaC') return '.flac';
  if (text(4, 4) === 'ftyp') return '.m4a';
  if (text(0, 3) === 'ID3' || (bytes[0] === 0xff && ((bytes[1] ?? 0) & 0xe0) === 0xe0)) {
    // ADTS AAC and MPEG audio share a sync word; the layer bits tell them apart.
    return ((bytes[1] ?? 0) & 0x06) === 0 && text(0, 3) !== 'ID3' ? '.aac' : '.mp3';
  }
  return undefined;
}

/** Run with a path to the audio, writing in-memory audio to a private temp file for the call. */
async function withAudioFile<T>(input: AudioInput, filename: string | undefined, body: (file: string) => Promise<T>): Promise<T> {
  if (typeof input === 'string') return body(resolvePath(input));
  if (input instanceof URL) {
    if (input.protocol !== 'file:') {
      throw new UnsupportedError(`transcribe reads local audio; download ${input.href} first.`, 'device');
    }
    return body(fileURLToPath(input));
  }
  const bytes = input instanceof ArrayBuffer ? new Uint8Array(input) : input;
  const named = filename === undefined ? '' : path.extname(filename);
  const directory = await mkdtemp(path.join(os.tmpdir(), 'apple-llm-audio-'));
  try {
    const file = path.join(directory, `audio${sniffAudioExtension(bytes) ?? (named === '' ? '.audio' : named)}`);
    await writeFile(file, bytes);
    try {
      return await body(file);
    } catch (error) {
      // The caller gave us bytes. A temp path in the message means nothing to them.
      if (error instanceof Error) error.message = error.message.split(file).join('these bytes');
      throw error;
    }
  } finally {
    await rm(directory, { recursive: true, force: true }).catch(() => undefined);
  }
}

function voiceFields(options: VoiceOptions): Record<string, unknown> {
  return {
    voice: options.voice,
    language: options.language,
    speed: options.speed,
    pitch: options.pitch,
    volume: options.volume,
    ssml: options.ssml,
  };
}

function matchesLanguage(tag: string, wanted: string): boolean {
  const have = tag.toLowerCase();
  const want = wanted.replace(/_/g, '-').toLowerCase();
  return have === want || have.startsWith(`${want}-`);
}

function toTranscript(response: Record<string, unknown>): Transcript {
  return {
    text: response.text as string,
    segments: (response.segments as TranscriptSegment[] | undefined) ?? [],
    language: response.language as string,
    engine: response.engine as Transcript['engine'],
    duration: response.duration as number,
  };
}

function toSaved(response: Record<string, unknown>): SavedSpeech {
  return {
    path: response.path as string,
    format: response.format as AudioFormat,
    duration: response.duration as number,
    sampleRate: response.sampleRate as number,
    bytes: response.bytes as number,
    voice: response.voice as VoiceInfo,
    ...(Array.isArray(response.words) ? { words: response.words as SpokenWord[] } : {}),
  };
}

/** A model download is seconds of nothing otherwise. */
function reportDownload(event: HelperEvent, onProgress: OnProgress | undefined): boolean {
  const download = event.download as { language?: string; fraction?: number } | undefined;
  if (download === undefined) return false;
  onProgress?.({
    status: `downloading the ${download.language} speech model (once)`,
    percent: Math.round((download.fraction ?? 0) * 100),
  });
  return true;
}

export interface AppleSpeechOptions {
  /** Called for the one-time compile and for language downloads. */
  onProgress?: OnProgress;
  /** A ready helper, for tests. */
  server?: HelperServer;
}

/** `stop()` needs the helper, which only exists once the session has started. */
interface ListenControl {
  stopped: boolean;
  send?: () => void;
}

/**
 * One `listen()`. Await it for the text, `for await` it for live events, or read
 * `.result` for the segments and timings.
 *
 * `stop()` finishes and keeps what was heard; `abort()` discards it.
 */
export class ListenSession extends ResultStream<ListenEvent, string, Transcript> {
  private readonly control: ListenControl;

  constructor(run: (emit: (event: ListenEvent) => void, signal: AbortSignal, control: ListenControl) => Promise<Transcript>) {
    const control: ListenControl = { stopped: false };
    super((emit, signal) => run(emit, signal, control), (transcript) => transcript.text);
    this.control = control;
  }

  /** Stop listening. The session still resolves, with everything heard so far. */
  stop(): void {
    this.control.stopped = true;
    this.control.send?.();
  }
}

/**
 * The speech API with a helper process of its own.
 *
 * The module-level `speak()`, `transcribe()` and friends share one of these,
 * which is enough for most programs. Create your own when something must not
 * wait: a helper handles one request at a time, so a long transcription holds
 * up a `speak()` on the same instance.
 *
 * The helper is unref'd between calls, so a script exits on its own.
 */
export class AppleSpeech {
  private server?: HelperServer;
  private readying?: Promise<HelperServer>;

  constructor(private readonly options: AppleSpeechOptions = {}) {
    this.server = options.server;
  }

  /** Compile the helper now rather than on the first call. */
  async ensureReady(onProgress?: OnProgress): Promise<void> {
    await this.helper(onProgress);
  }

  private helper(onProgress?: OnProgress): Promise<HelperServer> {
    if (this.server !== undefined) return Promise.resolve(this.server);
    this.readying ??= (async () => {
      await assertPlatform();
      const binary = await ensureBinary(onProgress ?? this.options.onProgress);
      this.server = new HelperServer(binary, { timeoutMs: SILENCE_TIMEOUT_MS });
      return this.server;
    })().finally(() => {
      this.readying = undefined;
    });
    return this.readying;
  }

  private async call(
    envelope: Record<string, unknown>,
    hooks: { signal?: AbortSignal; onProgress?: OnProgress; onEvent?: (event: HelperEvent, server: HelperServer) => void } = {},
  ): Promise<Record<string, unknown>> {
    const server = await this.helper(hooks.onProgress);
    const id = randomUUID();
    const onProgress = hooks.onProgress ?? this.options.onProgress;
    const raw = await server.request(JSON.stringify({ ...envelope, id }), {
      id,
      signal: hooks.signal,
      onEvent: (event) => {
        if (!reportDownload(event, onProgress)) hooks.onEvent?.(event, server);
      },
    });
    let response: Record<string, unknown>;
    try {
      response = JSON.parse(raw) as Record<string, unknown>;
    } catch {
      throw new AppleLLMError(`Apple helper returned an unreadable reply: ${raw.slice(0, 200)}`, 'device');
    }
    if (response.ok !== true) raiseFor(response, raw);
    return response;
  }

  /** Every voice on this Mac, sorted by language and then name. */
  async voices(filter: VoiceFilter = {}): Promise<Voice[]> {
    const response = await this.call({ op: 'voices' });
    const raw = response.voices as Array<Omit<Voice, 'isDefault'> & { default: boolean }>;
    return raw
      .map(({ default: isDefault, ...voice }) => ({ ...voice, isDefault }))
      .filter((voice) => filter.language === undefined || matchesLanguage(voice.language, filter.language))
      .filter((voice) => filter.novelty !== false || !voice.novelty)
      .sort((a, b) => a.language.localeCompare(b.language) || a.name.localeCompare(b.name));
  }

  /** The voice chosen in System Settings, which is what speech uses unless told otherwise. */
  async systemVoice(): Promise<SystemVoice> {
    const response = await this.call({ op: 'voices' });
    return response.systemVoice as SystemVoice;
  }

  /** Say it out loud, resolving when it has finished. Abort to stop it mid-word. */
  async speak(text: string, options: SpeakOptions = {}): Promise<SpeakResult> {
    const response = await this.call(
      { op: 'speak', text, words: options.onWord !== undefined ? true : undefined, ...voiceFields(options) },
      {
        signal: options.signal,
        onProgress: options.onProgress,
        onEvent: (event) => {
          if (event.word !== undefined) options.onWord?.(event.word as SpokenWord);
        },
      },
    );
    return { duration: response.duration as number, voice: response.voice as VoiceInfo };
  }

  /**
   * Render speech to audio: a file with `out`, otherwise a Buffer.
   *
   * For long text prefer `out`: thirty minutes is 80 MB as WAV and 11 MB as M4A,
   * and a Buffer holds all of it in memory.
   */
  synthesize(text: string, options: SynthesizeOptions & { out: string }): Promise<SavedSpeech>;
  synthesize(text: string, options?: SynthesizeOptions & { out?: undefined }): Promise<SynthesizedSpeech>;
  async synthesize(text: string, options: SynthesizeOptions & { out?: string } = {}): Promise<SavedSpeech | SynthesizedSpeech> {
    const fields = {
      op: 'synthesize',
      text,
      sampleRate: options.sampleRate,
      words: options.words,
      ...voiceFields(options),
    };
    const hooks = { signal: options.signal, onProgress: options.onProgress };
    if (options.out !== undefined) {
      return toSaved(await this.call({ ...fields, out: resolvePath(options.out), format: options.format }, hooks));
    }
    const format = options.format ?? 'wav';
    const directory = await mkdtemp(path.join(os.tmpdir(), 'apple-llm-speech-'));
    try {
      const file = path.join(directory, `speech.${format}`);
      const { path: _written, ...rest } = toSaved(await this.call({ ...fields, out: file, format }, hooks));
      return { ...rest, audio: await readFile(file) };
    } finally {
      await rm(directory, { recursive: true, force: true }).catch(() => undefined);
    }
  }

  /** Transcribe a recording: audio or video, any length, nothing leaves the machine. */
  transcribe(input: AudioInput, options: TranscribeOptions = {}): Promise<Transcript> {
    return withAudioFile(input, options.filename, async (file) => {
      let duration = 0;
      const onProgress = options.onProgress ?? this.options.onProgress;
      const response = await this.call(
        {
          op: 'transcribe',
          path: file,
          language: options.language,
          words: options.words,
          alternatives: options.alternatives,
          hints: options.hints,
          download: options.download,
        },
        {
          signal: options.signal,
          onProgress: options.onProgress,
          onEvent: (event) => {
            const started = event.started as { duration?: number } | undefined;
            if (started !== undefined) duration = started.duration ?? 0;
            const segment = event.segment as TranscriptSegment | undefined;
            if (segment === undefined) return;
            options.onSegment?.(segment);
            if (duration > 0) {
              onProgress?.({ status: 'transcribing', percent: Math.min(100, Math.round((segment.end / duration) * 100)) });
            }
          },
        },
      );
      return toTranscript(response);
    });
  }

  /**
   * `for await` each finished segment of a long recording, `await` the whole
   * text, or read `.result` for the full transcript. Breaking out of the loop
   * stops the work.
   */
  transcribeStream(input: AudioInput, options: Omit<TranscribeOptions, 'onSegment'> = {}): ResultStream<TranscriptSegment, string, Transcript> {
    return new ResultStream<TranscriptSegment, string, Transcript>(
      (emit, signal) =>
        this.transcribe(input, { ...options, signal: anySignal([options.signal, signal]), onSegment: emit }),
      (transcript) => transcript.text,
    );
  }

  /**
   * Listen to the microphone. By default it returns once you stop talking:
   *
   *   const said = await listen();
   *
   * With `continuous`, it goes on until `stop()`:
   *
   *   const session = listen({ continuous: true });
   *   for await (const event of session) console.log(event.text);
   *
   * macOS asks once for microphone access, on behalf of the app that launched
   * this: Terminal, VS Code, your own.
   */
  listen(options: ListenOptions = {}): ListenSession {
    return new ListenSession(async (emit, signal, control) => {
      const response = await this.call(
        {
          op: 'listen',
          language: options.language,
          silence: options.continuous === true ? undefined : (options.silence ?? 2),
          maxSeconds: options.maxSeconds,
          words: options.words,
          hints: options.hints,
          download: options.download,
        },
        {
          signal: anySignal([options.signal, signal]),
          onProgress: options.onProgress,
          onEvent: (event, server) => {
            if (event.listening === true) {
              control.send = () => server.write({ op: 'endInput' });
              // Asked to stop before the microphone even opened.
              if (control.stopped) control.send();
              options.onListening?.();
              return;
            }
            const heard = event.transcript as ListenEvent | undefined;
            if (heard !== undefined) emit(heard);
          },
        },
      );
      return toTranscript(response);
    });
  }

  /** Languages macOS can recognise, and which are already on this Mac. */
  async languages(): Promise<SpeechLanguages> {
    const response = await this.call({ op: 'speechLanguages' });
    return { languages: response.languages as SpeechLanguage[], default: response.default as string };
  }

  /** Download a language's model now rather than on first use. Resolves with its tag. */
  async installLanguage(language: string, options: { signal?: AbortSignal; onProgress?: OnProgress } = {}): Promise<string> {
    const response = await this.call({ op: 'installSpeechLanguage', language }, options);
    return response.language as string;
  }

  /** Say this helper no longer needs a language. macOS deletes it when it wants the space. */
  async removeLanguage(language: string): Promise<boolean> {
    const response = await this.call({ op: 'removeSpeechLanguage', language });
    return response.removed === true;
  }

  /** Shut the helper process down. Safe to call more than once. */
  close(): void {
    this.server?.stop();
    this.server = undefined;
  }
}

let shared: AppleSpeech | undefined;
const defaultSpeech = (): AppleSpeech => (shared ??= new AppleSpeech());

/** See {@link AppleSpeech.speak}. */
export const speak = (text: string, options?: SpeakOptions): Promise<SpeakResult> => defaultSpeech().speak(text, options);

/** See {@link AppleSpeech.synthesize}. */
export function synthesize(text: string, options: SynthesizeOptions & { out: string }): Promise<SavedSpeech>;
export function synthesize(text: string, options?: SynthesizeOptions & { out?: undefined }): Promise<SynthesizedSpeech>;
export function synthesize(text: string, options?: SynthesizeOptions & { out?: string }): Promise<SavedSpeech | SynthesizedSpeech> {
  return defaultSpeech().synthesize(text, options as SynthesizeOptions & { out: string });
}

/** See {@link AppleSpeech.voices}. */
export const listVoices = (filter?: VoiceFilter): Promise<Voice[]> => defaultSpeech().voices(filter);

/** See {@link AppleSpeech.systemVoice}. */
export const systemVoice = (): Promise<SystemVoice> => defaultSpeech().systemVoice();

/** See {@link AppleSpeech.transcribe}. */
export const transcribe = (input: AudioInput, options?: TranscribeOptions): Promise<Transcript> =>
  defaultSpeech().transcribe(input, options);

/** See {@link AppleSpeech.transcribeStream}. */
export const transcribeStream = (
  input: AudioInput,
  options?: Omit<TranscribeOptions, 'onSegment'>,
): ResultStream<TranscriptSegment, string, Transcript> => defaultSpeech().transcribeStream(input, options);

/** See {@link AppleSpeech.listen}. */
export const listen = (options?: ListenOptions): ListenSession => defaultSpeech().listen(options);

/** See {@link AppleSpeech.languages}. */
export const listSpeechLanguages = (): Promise<SpeechLanguages> => defaultSpeech().languages();

/** See {@link AppleSpeech.installLanguage}. */
export const installSpeechLanguage = (
  language: string,
  options?: { signal?: AbortSignal; onProgress?: OnProgress },
): Promise<string> => defaultSpeech().installLanguage(language, options);

/** See {@link AppleSpeech.removeLanguage}. */
export const removeSpeechLanguage = (language: string): Promise<boolean> => defaultSpeech().removeLanguage(language);

/** What speech can do on this machine. Never throws, and works with Apple Intelligence off. */
export async function probeSpeech(onProgress?: OnProgress): Promise<SpeechProbe> {
  const speech = new AppleSpeech({ onProgress });
  try {
    const [voices, system, languages] = await Promise.all([speech.voices(), speech.systemVoice(), speech.languages()]);
    return {
      available: true,
      voices: voices.length,
      systemVoice: system,
      languages: languages.languages.length,
      installedLanguages: languages.languages.filter((l) => l.installed).map((l) => l.id),
      defaultLanguage: languages.default,
    };
  } catch (error) {
    return { available: false, reason: error instanceof Error ? error.message : String(error) };
  } finally {
    speech.close();
  }
}

/** Scripts with no spaces between words, where joining words with one would be wrong. */
const UNSPACED = /^(zh|ja|yue|th|lo|km|my)\b/i;

/** Cues short enough to read: under 84 characters and 7 seconds where word timings allow it. */
function captionCues(transcript: Transcript): Array<{ start: number; end: number; text: string }> {
  const separator = UNSPACED.test(transcript.language) ? '' : ' ';
  const cues: Array<{ start: number; end: number; text: string }> = [];
  for (const segment of transcript.segments) {
    if (segment.text === '') continue;
    const words = segment.words ?? [];
    if (words.length === 0) {
      cues.push({ start: segment.start, end: segment.end, text: segment.text });
      continue;
    }
    let group: TranscriptWord[] = [];
    const flush = (): void => {
      if (group.length === 0) return;
      cues.push({
        start: group[0].start,
        end: group[group.length - 1].end,
        text: group.map((word) => word.text).join(separator),
      });
      group = [];
    };
    for (const word of words) {
      const length = group.reduce((sum, w) => sum + w.text.length + separator.length, 0) + word.text.length;
      if (group.length > 0 && (length > 84 || word.end - group[0].start > 7)) flush();
      group.push(word);
    }
    flush();
  }
  return cues;
}

function clock(seconds: number, comma: boolean): string {
  const total = Math.max(0, Math.round(seconds * 1000));
  const pad = (n: number, width = 2): string => String(n).padStart(width, '0');
  const stamp = `${pad(Math.floor(total / 3_600_000))}:${pad(Math.floor(total / 60_000) % 60)}:${pad(Math.floor(total / 1000) % 60)}`;
  return `${stamp}${comma ? ',' : '.'}${pad(total % 1000, 3)}`;
}

/** SubRip subtitles. Ask `transcribe` for `words: true` to get cues short enough to read. */
export function toSRT(transcript: Transcript): string {
  return captionCues(transcript)
    .map((cue, index) => `${index + 1}\n${clock(cue.start, true)} --> ${clock(cue.end, true)}\n${cue.text}\n`)
    .join('\n');
}

/** WebVTT subtitles. Ask `transcribe` for `words: true` to get cues short enough to read. */
export function toVTT(transcript: Transcript): string {
  const cues = captionCues(transcript).map((cue) => `${clock(cue.start, false)} --> ${clock(cue.end, false)}\n${cue.text}\n`);
  return ['WEBVTT\n', ...cues].join('\n');
}
