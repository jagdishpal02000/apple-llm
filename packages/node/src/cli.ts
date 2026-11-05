#!/usr/bin/env node
/**
 * apple-llm CLI — how people debug this, and how it is tested by hand.
 *
 *   apple-llm probe
 *   apple-llm setup-cloud [--web-search] [--force]
 *   apple-llm run --tier device --system "..." -      # prompt on stdin
 *   apple-llm run --tier cloud --schema s.json "Extract the fields"
 *   apple-llm chat                                     # interactive
 *   apple-llm serve                                    # OpenAI-compatible API
 *   apple-llm speak "Hello there"                      # out loud
 *   apple-llm transcribe meeting.m4a --format srt
 *   apple-llm listen                                   # from the microphone
 */
import { fstatSync } from 'node:fs';
import { readFile, writeFile } from 'node:fs/promises';
import path from 'node:path';
import { AppleLLM, captureScreenshot, parseImageFlag, probe } from './index.js';
import { installCloudShortcut } from './cloud.js';
import type { OnProgress } from './compile.js';
import { AppleLLMError } from './errors.js';
import { runRepl } from './repl.js';
import { DEFAULT_PORT, serve } from './server.js';
import {
  AppleSpeech,
  probeSpeech,
  toSRT,
  toVTT,
  type AudioFormat,
  type AudioInput,
  type Transcript,
} from './speech.js';

interface Flags {
  tier?: string;
  fallback?: string;
  system?: string;
  schema?: string;
  temperature?: string;
  maxTokens?: string;
  webSearch?: boolean;
  force?: boolean;
  json?: boolean;
  image: string[];
  document: string[];
  tool: string[];
  session?: string;
  stream?: boolean;
  mode?: string;
  useCase?: string;
  guardrails?: string;
  seed?: string;
  greedy?: boolean;
  timeout?: string;
  port?: string;
  host?: string;
  apiKey?: string;
  cors?: string;
  voice?: string;
  language?: string;
  speed?: string;
  pitch?: string;
  volume?: string;
  ssml?: boolean;
  out?: string;
  format?: string;
  sampleRate?: string;
  file?: string;
  words?: boolean;
  hint: string[];
  alternatives?: boolean;
  noDownload?: boolean;
  all?: boolean;
  once?: boolean;
  silence?: string;
  maxSeconds?: string;
  positional: string[];
}

function parseArgs(argv: string[]): Flags {
  const flags: Flags = { positional: [], image: [], document: [], tool: [], hint: [] };
  for (let i = 0; i < argv.length; i += 1) {
    const arg = argv[i];
    const take = (flag: string): string => {
      const v = argv[++i];
      if (v === undefined || v.startsWith('--')) {
        throw new Error(`${flag} needs a value.\n\n${USAGE}`);
      }
      return v;
    };
    switch (arg) {
      case '--tier': flags.tier = take('--tier'); break;
      case '--fallback': flags.fallback = take('--fallback'); break;
      case '--system': flags.system = take('--system'); break;
      case '--schema': flags.schema = take('--schema'); break;
      case '--temperature': flags.temperature = take('--temperature'); break;
      case '--max-tokens': flags.maxTokens = take('--max-tokens'); break;
      case '--web-search': flags.webSearch = true; break;
      case '--force': flags.force = true; break;
      case '--json': flags.json = true; break;
      case '--image': flags.image.push(take('--image')); break;
      case '--document': flags.document.push(take('--document')); break;
      case '--tool': flags.tool.push(take('--tool')); break;
      case '--session': flags.session = take('--session'); break;
      case '--stream': flags.stream = true; break;
      case '--mode': flags.mode = take('--mode'); break;
      case '--use-case': flags.useCase = take('--use-case'); break;
      case '--guardrails': flags.guardrails = take('--guardrails'); break;
      case '--seed': flags.seed = take('--seed'); break;
      case '--greedy': flags.greedy = true; break;
      case '--timeout': flags.timeout = take('--timeout'); break;
      case '--port': flags.port = take('--port'); break;
      case '--host': flags.host = take('--host'); break;
      case '--api-key': flags.apiKey = take('--api-key'); break;
      case '--cors': flags.cors = take('--cors'); break;
      case '--voice': flags.voice = take('--voice'); break;
      case '--language': case '--lang': case '-l': flags.language = take('--language'); break;
      case '--speed': flags.speed = take('--speed'); break;
      case '--pitch': flags.pitch = take('--pitch'); break;
      case '--volume': flags.volume = take('--volume'); break;
      case '--ssml': flags.ssml = true; break;
      case '--out': case '-o': flags.out = take('--out'); break;
      case '--format': flags.format = take('--format'); break;
      case '--sample-rate': flags.sampleRate = take('--sample-rate'); break;
      case '--file': flags.file = take('--file'); break;
      case '--words': flags.words = true; break;
      case '--hint': flags.hint.push(take('--hint')); break;
      case '--alternatives': flags.alternatives = true; break;
      case '--no-download': flags.noDownload = true; break;
      case '--all': flags.all = true; break;
      case '--once': flags.once = true; break;
      case '--silence': flags.silence = take('--silence'); break;
      case '--max-seconds': flags.maxSeconds = take('--max-seconds'); break;
      default: flags.positional.push(arg);
    }
  }
  return flags;
}

async function readStdin(): Promise<string> {
  const chunks: Buffer[] = [];
  for await (const chunk of process.stdin) chunks.push(chunk as Buffer);
  return Buffer.concat(chunks).toString('utf8');
}

/**
 * Text piped in beside a prompt — `cat notes.md | apple-llm run "Summarize
 * this"` — or undefined. Only a pipe or a redirected file is read: a terminal
 * is interactive, and a socket (how Node spawns children) may never close.
 */
async function pipedInput(): Promise<string | undefined> {
  let stat;
  try {
    stat = fstatSync(0);
  } catch {
    return undefined;
  }
  if (!stat.isFIFO() && !stat.isFile()) return undefined;
  const text = await readStdin();
  return text.trim() === '' ? undefined : text;
}

/** With text piped in, the prompt says what to do with it — after it, as people read it. */
async function promptFrom(arg: string): Promise<string> {
  if (arg === '-') return readStdin();
  const piped = await pipedInput();
  return piped === undefined ? arg : `${piped.trimEnd()}\n\n${arg}`;
}

/**
 * One status line on stderr, rewritten in place on a terminal. When piped, each
 * status appears once with no percentages, so a log is not a wall of updates.
 */
function statusLine(): { onProgress: OnProgress; clear: () => void } {
  const tty = process.stderr.isTTY === true;
  let active = false;
  let last = '';
  return {
    onProgress: (progress) => {
      if (!tty) {
        if (progress.status !== last) process.stderr.write(`  ${progress.status}\n`);
        last = progress.status;
        return;
      }
      const percent = progress.percent === undefined ? '' : ` ${progress.percent}%`;
      process.stderr.write(`\r\x1b[K  ${progress.status}${percent}`);
      active = true;
    },
    clear: () => {
      if (active) process.stderr.write('\r\x1b[K');
      active = false;
    },
  };
}

/** A number from a flag, or a usage error naming the flag. */
function numberFlag(name: string, value: string | undefined): number | undefined {
  if (value === undefined) return undefined;
  const parsed = Number(value);
  if (Number.isNaN(parsed)) throw new Error(`${name} must be a number (got "${value}").`);
  return parsed;
}

/**
 * What `speak` should say. Words on the command line need no quotes, so
 * `apple-llm speak hello there` works. Bare flags left over are typos: without
 * this check, a mistyped option would be read aloud.
 */
async function textToSpeak(flags: Flags): Promise<string | undefined> {
  if (flags.file !== undefined) return readFile(flags.file, 'utf8');
  if (flags.positional.length === 1 && flags.positional[0] === '-') return readStdin();
  if (flags.positional.length > 0) return flags.positional.join(' ');
  return pipedInput();
}

function rejectUnknownOptions(flags: Flags): void {
  const unknown = flags.positional.find((arg) => arg.startsWith('--'));
  if (unknown !== undefined) throw new Error(`Unknown option ${unknown}.`);
}

const TRANSCRIPT_FORMATS = new Set(['text', 'json', 'srt', 'vtt']);

function renderTranscript(transcript: Transcript, format: string): string {
  if (format === 'json') return `${JSON.stringify(transcript, null, 2)}\n`;
  if (format === 'srt') return toSRT(transcript);
  if (format === 'vtt') return toVTT(transcript);
  return transcript.text === '' ? '' : `${transcript.text}\n`;
}

/** `notes.srt` means subtitles, `notes.json` means JSON; anything else is text. */
function transcriptFormatFor(out: string | undefined): string {
  const extension = out === undefined ? '' : path.extname(out).slice(1).toLowerCase();
  return TRANSCRIPT_FORMATS.has(extension) ? extension : 'text';
}

const SYSTEM_VOICE_HINT = 'search System Settings for "System voice"';

const KNOWN_TOOLS = new Set(['ocr', 'barcode', 'spotlight']);

function parseTools(values: string[], usageError: (m: string) => number): string[] | undefined {
  if (values.length === 0) return undefined;
  const out: string[] = [];
  for (const value of values) {
    for (const part of value.split(',').map((s) => s.trim()).filter(Boolean)) {
      if (!KNOWN_TOOLS.has(part)) {
        throw new Error(`--tool must be ocr, barcode, or spotlight (got "${part}").`);
      }
      if (!out.includes(part)) out.push(part);
    }
  }
  return out;
}

const USAGE = `apple-llm — Apple's on-device and Private Cloud Compute models

  apple-llm "<prompt>"                   one completion (short for run)
  apple-llm probe                        what this machine can do
  apple-llm setup-cloud [--web-search]   install the Shortcut the cloud tier needs
  apple-llm run [options] <prompt|->     one completion ("-" reads stdin)
  apple-llm chat [options]               interactive conversation, streamed
  apple-llm serve [options]              OpenAI- and Ollama-compatible API on localhost
  apple-llm count [options] <prompt|->   tokens this prompt costs, before sending
  apple-llm ask-screen [options] <question|->   screenshot, then ask (device tier)
  apple-llm history --session <id>       show a conversation's mirrored turns
  apple-llm reset [--session <id>]       drop one conversation, or all
  apple-llm rewrite|proofread|summarize|draft <text|->   Write with Siri presets

  Speech, on this Mac, with or without Apple Intelligence:
  apple-llm speak <text|->               say it out loud, or save it with -o file.m4a
  apple-llm voices                       the voices on this Mac
  apple-llm transcribe <file|->          audio or video to text, any length
  apple-llm listen                       the microphone to text, live
  apple-llm languages [install|remove <code>]   languages it can recognise

  Text piped into run goes in before the prompt:
    cat notes.md | apple-llm "Summarize this"
    git diff | apple-llm "Write a commit message for this"

run options:
  --tier device|cloud|auto   default: auto
  --fallback device          cloud tier: answer on device when Apple rate-limits the cloud
  --system <text>            system instructions
  --schema <file.json>       ask for JSON matching this JSON Schema
  --temperature <n>          default 0.4 (0 degenerates; see the README)
  --max-tokens <n>
  --image <path[::label]>    attach an image, optionally labelled (repeatable; macOS 27+)
  --document <path>          inline a text document into the prompt (repeatable)
  --tool ocr|barcode|spotlight   built-in on-device tool (repeatable)
  --session <id>             continue a named conversation across calls
  --stream                   print text as it arrives (the cloud tier sends it whole)
  --use-case contentTagging  Apple's tagging-specialised model
  --guardrails permissive    relax guardrails for rewriting tasks
  --seed <n>                 reproducible sampling (top-k, seeded)
  --greedy                   greedy decoding (deterministic, but degenerates)
  --web-search               cloud tier only
  --timeout <seconds>        give up after this long
  --json                     print the full result: text, usage, tool calls
                             (with --schema --stream: one partial object per line)

chat options: --tier, --fallback, --system, --tool, --use-case, --guardrails, --max-tokens.
  In the chat: /reset, /system <text>, /image <path>, /tokens, /help, /exit.

serve options:
  --port <n>                 default ${DEFAULT_PORT}
  --host <addr>              default 127.0.0.1 (0.0.0.0 exposes it to your network)
  --api-key <key>            require "Authorization: Bearer <key>"
  --cors <origin>            allow browser clients from this origin
  --tier device|cloud|auto   for requests not naming a model; default device
  --fallback device          answer apple-private-cloud requests on device when rate-limited

ask-screen options: --mode interactive|window|fullscreen (default interactive),
  plus --system, --session, --max-tokens.

speak options:
  --voice <name|id|system>   default: your System voice (apple-llm voices lists them)
  --language <code>          the best voice for a language: fr, en-GB, hi-IN
  --speed <n>                1 is normal, 2 is twice as fast (about 0.56 to 4)
  --pitch <n>                0.5 to 2
  --volume <n>               0 to 1
  --ssml                     the text is SSML
  --file <path>              read the text from a file
  -o, --out <file|->         save instead of playing: .wav .m4a .aac .flac .aiff .caf .pcm
                             ("-" writes the audio to stdout)
  --format <fmt>             when the extension does not say
  --sample-rate <hz>         resample, e.g. 16000
  --words                    with --json: every word's timing
  --json                     print details of what was made

voices options: --language <code>, --all (every language, and novelty voices), --json.

transcribe options:
  --language <code>          default: this Mac's language; it is not detected from the audio
  --format text|json|srt|vtt default: from the -o extension, otherwise text
  -o, --out <file>           write there instead of stdout
  --words                    with --format json: every word's timing and confidence
  --hint <term>              a name or term to expect (repeatable)
  --alternatives             with --format json: other readings of each segment
  --no-download              fail rather than download a missing language

listen options:
  --once                     stop when you pause (--silence sets how long; default 2s)
  --silence <seconds>        seconds of quiet that end it
  --max-seconds <n>          stop after this long, speech or no speech
  --language, --hint, --no-download   as for transcribe
  -o, --out <file>           also write the final text there
  --json                     one JSON event per line, partial results included
  Stops with Ctrl-C, and keeps what it heard.
`;

const COMMANDS = new Set([
  'probe', 'setup-cloud', 'run', 'chat', 'serve', 'count', 'ask-screen', 'history', 'reset',
  'rewrite', 'proofread', 'summarize', 'draft', 'help',
  'speak', 'voices', 'transcribe', 'listen', 'languages',
]);

async function main(): Promise<number> {
  let [command, ...rest] = process.argv.slice(2);
  // `apple-llm "What does this error mean?"` is a prompt, not a mistyped
  // command: a command never contains a space, and a typo still errors.
  if (command !== undefined && !COMMANDS.has(command) && /\s/.test(command)) {
    rest = [command, ...rest];
    command = 'run';
  }
  let flags: Flags;
  try {
    flags = parseArgs(rest);
  } catch (err) {
    process.stderr.write(`${err instanceof Error ? err.message : String(err)}\n`);
    return 2;
  }

  const usageError = (message: string): number => {
    process.stderr.write(`${message}\n\n${USAGE}`);
    return 2;
  };

  if (command === undefined || command === '--help' || command === '-h' || command === 'help') {
    process.stdout.write(USAGE);
    return 0;
  }
  if (flags.fallback !== undefined && flags.fallback !== 'device') {
    return usageError(`--fallback must be device (got "${flags.fallback}").`);
  }
  const fallback = flags.fallback as 'device' | undefined;

  if (command === 'probe') {
    const result = await probe((p) => process.stderr.write(`  ${p.status}\n`));
    const speech = await probeSpeech();
    if (flags.json === true) {
      process.stdout.write(`${JSON.stringify({ ...result, speech }, null, 2)}\n`);
      return result.device.available || result.cloud.available ? 0 : 1;
    }
    const d = result.device;
    process.stdout.write('on-device\n');
    process.stdout.write(`  available   ${d.available}\n`);
    if (d.variant !== undefined) process.stdout.write(`  variant     ${d.variant}\n`);
    if (d.contextSize !== undefined) process.stdout.write(`  context     ${d.contextSize} tokens\n`);
    if (d.capabilities !== undefined) {
      const on = Object.entries(d.capabilities).filter(([, v]) => v).map(([k]) => k);
      process.stdout.write(`  supports    ${on.join(', ') || 'none reported'}\n`);
    }
    if (d.useCases !== undefined) process.stdout.write(`  use cases   ${d.useCases.join(', ')}\n`);
    if (d.reason !== undefined) process.stdout.write(`  reason      ${d.reason}\n`);
    const c = result.cloud;
    process.stdout.write('\ncloud (private cloud compute)\n');
    process.stdout.write(`  available   ${c.available}\n`);
    process.stdout.write(`  installed   ${c.installed === true}\n`);
    if (c.contextSize !== undefined) process.stdout.write(`  context     ${c.contextSize} tokens\n`);
    if (c.quota !== undefined) {
      const approaching = c.quota.approachingLimit === true ? ' (approaching limit)' : '';
      process.stdout.write(`  quota       ${c.quota.status}${approaching}\n`);
      if (c.quota.resetDate !== undefined) process.stdout.write(`  resets      ${c.quota.resetDate}\n`);
    }
    if (c.reason !== undefined) process.stdout.write(`  reason      ${c.reason}\n`);
    process.stdout.write('\nspeech\n');
    process.stdout.write(`  available   ${speech.available}\n`);
    if (speech.voices !== undefined) process.stdout.write(`  voices      ${speech.voices}\n`);
    if (speech.systemVoice !== undefined) {
      process.stdout.write(`  speaks as   ${speech.systemVoice.name} (${speech.systemVoice.language})\n`);
    }
    if (speech.languages !== undefined) {
      process.stdout.write(`  recognises  ${speech.languages} languages, ${speech.defaultLanguage} by default\n`);
    }
    if (speech.reason !== undefined) process.stdout.write(`  reason      ${speech.reason}\n`);
    return d.available || c.available ? 0 : 1;
  }

  if (command === 'speak') {
    let text: string | undefined;
    let speed: number | undefined;
    let pitch: number | undefined;
    let volume: number | undefined;
    let sampleRate: number | undefined;
    try {
      rejectUnknownOptions(flags);
      speed = numberFlag('--speed', flags.speed);
      pitch = numberFlag('--pitch', flags.pitch);
      volume = numberFlag('--volume', flags.volume);
      sampleRate = numberFlag('--sample-rate', flags.sampleRate);
      text = await textToSpeak(flags);
    } catch (err) {
      return usageError(err instanceof Error ? err.message : String(err));
    }
    if (text === undefined) return usageError('speak needs some text, "-" to read stdin, or --file <path>.');
    const status = statusLine();
    const speech = new AppleSpeech({ onProgress: status.onProgress });
    const voiceOptions = { voice: flags.voice, language: flags.language, speed, pitch, volume, ssml: flags.ssml };
    try {
      const toStdout = flags.out === '-' || (flags.out === undefined && flags.format !== undefined);
      if (toStdout) {
        if (process.stdout.isTTY === true) return usageError('That would print audio to the terminal. Add -o <file>, or redirect it.');
        const made = await speech.synthesize(text, {
          ...voiceOptions,
          format: flags.format as AudioFormat | undefined,
          sampleRate,
        });
        status.clear();
        process.stdout.write(made.audio);
        return 0;
      }
      if (flags.out !== undefined) {
        const saved = await speech.synthesize(text, {
          ...voiceOptions,
          out: flags.out,
          format: flags.format as AudioFormat | undefined,
          sampleRate,
          words: flags.words,
        });
        status.clear();
        if (flags.json === true) {
          process.stdout.write(`${JSON.stringify(saved, null, 2)}\n`);
        } else {
          process.stderr.write(`wrote ${flags.out} (${saved.duration.toFixed(1)}s, ${saved.voice.name})\n`);
        }
        return 0;
      }
      const spoken = await speech.speak(text, voiceOptions);
      status.clear();
      if (flags.json === true) process.stdout.write(`${JSON.stringify(spoken, null, 2)}\n`);
      return 0;
    } finally {
      status.clear();
      speech.close();
    }
  }

  if (command === 'voices') {
    const speech = new AppleSpeech({ onProgress: (p) => process.stderr.write(`  ${p.status}\n`) });
    try {
      const everything = await speech.voices({ language: flags.language });
      if (flags.json === true) {
        process.stdout.write(`${JSON.stringify(everything, null, 2)}\n`);
        return 0;
      }
      const system = await speech.systemVoice();
      process.stdout.write(`Your System voice is ${system.name} (${system.language}). To change it, ${SYSTEM_VOICE_HINT}.\n\n`);
      // A hundred and eighty lines is not an answer to "what can I use?".
      const family = system.language.split('-')[0];
      const shown = flags.all === true
        ? everything
        : everything.filter((voice) => !voice.novelty && (flags.language !== undefined || voice.language.split('-')[0] === family));
      if (shown.length === 0) process.stdout.write('No voices match that.\n');
      const width = Math.max(4, ...shown.map((voice) => voice.name.length));
      for (const voice of shown) {
        const notes = [
          voice.quality === 'default' ? '' : voice.quality,
          voice.novelty ? 'novelty' : '',
          voice.engine === 'say' ? 'speed only' : '',
          voice.isDefault ? 'your System voice' : '',
        ].filter(Boolean).join(', ');
        const gender = voice.gender === 'unspecified' ? '' : voice.gender;
        process.stdout.write(`  ${voice.name.padEnd(width)}  ${voice.language.padEnd(7)}  ${gender.padEnd(6)}  ${notes}`.trimEnd() + '\n');
      }
      if (flags.all !== true) {
        const hidden = everything.length - shown.length;
        if (hidden > 0) process.stdout.write(`\n${hidden} more: --all lists every language and novelty voice, --language fr filters.\n`);
      }
      process.stdout.write(
        '\nTry one:  apple-llm speak "Hello there" --voice ' + (shown.find((voice) => !voice.isDefault)?.name ?? system.name) + '\n' +
          `Better voices are free downloads in the same place, and so are Siri voices where macOS offers them: ${SYSTEM_VOICE_HINT}.\n`,
      );
      return 0;
    } finally {
      speech.close();
    }
  }

  if (command === 'languages') {
    const speech = new AppleSpeech();
    const [action, code] = flags.positional;
    try {
      if (action === 'install') {
        if (code === undefined) return usageError('languages install needs a language, e.g. hi-IN.');
        const status = statusLine();
        const installed = await speech.installLanguage(code, { onProgress: status.onProgress });
        status.clear();
        process.stdout.write(`${installed} is ready.\n`);
        return 0;
      }
      if (action === 'remove') {
        if (code === undefined) return usageError('languages remove needs a language, e.g. hi-IN.');
        const removed = await speech.removeLanguage(code);
        process.stdout.write(removed ? `Released ${code}. macOS removes it when it wants the space.\n` : `${code} was not installed by apple-llm, so nothing changed.\n`);
        return 0;
      }
      if (action !== undefined) return usageError(`languages takes install or remove (got "${action}").`);
      const { languages, default: chosen } = await speech.languages();
      if (flags.json === true) {
        process.stdout.write(`${JSON.stringify({ languages, default: chosen }, null, 2)}\n`);
        return 0;
      }
      process.stdout.write('Languages macOS can recognise. A * is already on this Mac; the rest download on first use, in about 25 seconds.\n\n');
      const width = Math.max(...languages.map((language) => language.id.length));
      for (const language of languages) {
        const note = language.id === chosen ? '  (the default)' : '';
        process.stdout.write(`${language.installed ? '*' : ' '} ${language.id.padEnd(width)}  ${language.name}${note}\n`);
      }
      return 0;
    } finally {
      speech.close();
    }
  }

  if (command === 'transcribe') {
    const arg = flags.positional[0];
    if (arg === undefined) return usageError('transcribe needs a file, or "-" to read audio from stdin.');
    try {
      rejectUnknownOptions(flags);
    } catch (err) {
      return usageError(err instanceof Error ? err.message : String(err));
    }
    const format = flags.format ?? transcriptFormatFor(flags.out);
    if (!TRANSCRIPT_FORMATS.has(format)) return usageError(`--format must be text, json, srt or vtt (got "${format}").`);
    let input: AudioInput = arg;
    if (arg === '-') {
      const chunks: Buffer[] = [];
      for await (const chunk of process.stdin) chunks.push(chunk as Buffer);
      input = Buffer.concat(chunks);
    }
    const status = statusLine();
    const speech = new AppleSpeech({ onProgress: status.onProgress });
    try {
      const transcript = await speech.transcribe(input, {
        language: flags.language,
        // Subtitles use word timings to keep each cue short enough to read.
        words: flags.words === true || format === 'srt' || format === 'vtt',
        alternatives: flags.alternatives,
        hints: flags.hint.length > 0 ? flags.hint : undefined,
        download: flags.noDownload === true ? false : undefined,
        onProgress: status.onProgress,
      });
      status.clear();
      if (transcript.text === '') process.stderr.write('No speech found.\n');
      const rendered = renderTranscript(transcript, format);
      if (flags.out !== undefined) {
        await writeFile(flags.out, rendered);
        process.stderr.write(`wrote ${flags.out}\n`);
      } else {
        process.stdout.write(rendered);
      }
      return 0;
    } finally {
      status.clear();
      speech.close();
    }
  }

  if (command === 'listen') {
    let silence: number | undefined;
    let maxSeconds: number | undefined;
    try {
      rejectUnknownOptions(flags);
      silence = numberFlag('--silence', flags.silence);
      maxSeconds = numberFlag('--max-seconds', flags.maxSeconds);
    } catch (err) {
      return usageError(err instanceof Error ? err.message : String(err));
    }
    const once = flags.once === true || silence !== undefined;
    const status = statusLine();
    const speech = new AppleSpeech({ onProgress: status.onProgress });
    const interactive = process.stderr.isTTY === true;
    const session = speech.listen({
      language: flags.language,
      continuous: !once,
      silence,
      maxSeconds,
      words: flags.words,
      hints: flags.hint.length > 0 ? flags.hint : undefined,
      download: flags.noDownload === true ? false : undefined,
      onProgress: status.onProgress,
      onListening: () => {
        status.clear();
        process.stderr.write(once ? 'Listening. It stops when you pause.\n' : 'Listening. Press Ctrl-C to stop.\n');
      },
    });
    // The first Ctrl-C finishes and keeps the text; a second is for a real hurry.
    const onInterrupt = (): void => {
      session.stop();
      process.once('SIGINT', () => process.exit(130));
    };
    process.once('SIGINT', onInterrupt);
    try {
      for await (const event of session) {
        if (flags.json === true) {
          process.stdout.write(`${JSON.stringify(event)}\n`);
          continue;
        }
        if (interactive) {
          const room = Math.max(20, (process.stderr.columns ?? 80) - 1);
          process.stderr.write(`\r\x1b[K${event.final ? '' : `\x1b[2m${event.text.slice(-room)}\x1b[0m`}`);
        }
        if (event.final) process.stdout.write(`${event.text}\n`);
      }
      const transcript = await session.result;
      if (flags.out !== undefined) {
        await writeFile(flags.out, renderTranscript(transcript, transcriptFormatFor(flags.out)));
        process.stderr.write(`wrote ${flags.out}\n`);
      }
      return 0;
    } finally {
      process.removeListener('SIGINT', onInterrupt);
      status.clear();
      speech.close();
    }
  }

  if (command === 'setup-cloud') {
    await installCloudShortcut((p) => process.stderr.write(`  ${p.status}\n`), {
      force: flags.force,
      webSearch: flags.webSearch,
    });
    return 0;
  }

  if (command === 'count') {
    const arg = flags.positional[0];
    if (arg === undefined) {
      process.stderr.write('count needs a prompt, or "-" to read stdin.\n');
      return 2;
    }
    const prompt = arg === '-' ? await readStdin() : arg;
    const llm = new AppleLLM({ tier: 'device' });
    let tools: ('ocr' | 'barcode' | 'spotlight')[] | undefined;
    try {
      tools = parseTools(flags.tool, usageError) as typeof tools;
    } catch (err) {
      return usageError(err instanceof Error ? err.message : String(err));
    }
    try {
      const { tokens, contextSize } = await llm.countTokens(prompt, {
        system: flags.system,
        images: flags.image.length > 0 ? flags.image.map(parseImageFlag) : undefined,
        tools,
      });
      const pct = contextSize > 0 ? Math.round((tokens / contextSize) * 100) : 0;
      process.stdout.write(`${tokens} tokens of ${contextSize} (${pct}%)\n`);
      return tokens > contextSize ? 1 : 0;
    } finally {
      llm.close();
    }
  }

  if (command === 'history') {
    if (flags.session === undefined) return usageError('history needs --session <id>.');
    const llm = new AppleLLM({ tier: 'device' });
    try {
      const { instructions, history } = await llm.history(flags.session);
      if (flags.json === true) {
        process.stdout.write(`${JSON.stringify({ sessionId: flags.session, instructions, history }, null, 2)}\n`);
        return 0;
      }
      for (const turn of history) {
        process.stdout.write(`${turn.role === 'user' ? 'you' : 'model'}: ${turn.content}\n`);
      }
      return 0;
    } finally {
      llm.close();
    }
  }

  if (command === 'reset') {
    const llm = new AppleLLM({ tier: 'device' });
    try {
      await llm.resetSession(flags.session);
      process.stdout.write(flags.session ? `reset ${flags.session}\n` : 'reset all sessions\n');
      return 0;
    } finally {
      llm.close();
    }
  }

  if (command === 'rewrite' || command === 'proofread' || command === 'summarize' || command === 'draft') {
    const arg = flags.positional[0];
    if (arg === undefined) {
      process.stderr.write(`${command} needs text, or "-" to read stdin.\n`);
      return 2;
    }
    const text = arg === '-' ? await readStdin() : arg;
    const llm = new AppleLLM({
      tier: 'device',
      guardrails: 'permissive',
      onProgress: (p) => process.stderr.write(`  ${p.status}\n`),
    });
    try {
      const out =
        command === 'rewrite'
          ? await llm.rewrite(text, { system: flags.system })
          : command === 'proofread'
            ? await llm.proofread(text, { system: flags.system })
            : command === 'summarize'
              ? await llm.summarize(text, { system: flags.system })
              : await llm.draft(text, { system: flags.system });
      process.stdout.write(`${out}\n`);
      return 0;
    } finally {
      llm.close();
    }
  }

  if (command === 'ask-screen') {
    const arg = flags.positional[0];
    if (arg === undefined) {
      process.stderr.write('ask-screen needs a question, or "-" to read stdin.\n');
      return 2;
    }
    if (flags.mode !== undefined && !['interactive', 'window', 'fullscreen'].includes(flags.mode)) {
      return usageError(`--mode must be interactive, window, or fullscreen (got "${flags.mode}").`);
    }
    const question = arg === '-' ? await readStdin() : arg;
    const maxTokens = flags.maxTokens === undefined ? undefined : Number(flags.maxTokens);
    if (maxTokens !== undefined && Number.isNaN(maxTokens)) {
      return usageError(`--max-tokens must be a number (got "${flags.maxTokens}").`);
    }
    process.stderr.write('  select a screen region (Esc cancels)\n');
    const shot = await captureScreenshot(
      (flags.mode ?? 'interactive') as 'interactive' | 'window' | 'fullscreen',
    );
    const llm = new AppleLLM({
      tier: 'device',
      maxTokens,
      onProgress: (p) => process.stderr.write(`  ${p.status}\n`),
    });
    try {
      const out = await llm.text(question, {
        system: flags.system,
        sessionId: flags.session,
        images: [...flags.image.map(parseImageFlag), shot],
      });
      process.stdout.write(`${out}\n`);
      return 0;
    } finally {
      const { rm } = await import('node:fs/promises');
      await rm(shot, { force: true }).catch(() => undefined);
      llm.close();
    }
  }

  if (command === 'serve') {
    const port = flags.port === undefined ? DEFAULT_PORT : Number(flags.port);
    if (!Number.isInteger(port) || port < 0 || port > 65535) return usageError(`--port must be a port number (got "${flags.port}").`);
    if (flags.tier !== undefined && !['device', 'cloud', 'auto'].includes(flags.tier)) {
      return usageError(`--tier must be device, cloud, or auto (got "${flags.tier}").`);
    }
    let running: Awaited<ReturnType<typeof serve>>;
    try {
      running = await serve({
        port,
        host: flags.host,
        apiKey: flags.apiKey,
        cors: flags.cors,
        tier: flags.tier as 'device' | 'cloud' | 'auto' | undefined,
        fallback,
        log: (line) => process.stderr.write(`  ${line}\n`),
      });
    } catch (err) {
      if ((err as NodeJS.ErrnoException).code !== 'EADDRINUSE') throw err;
      process.stderr.write(
        `Port ${port} is already in use${port === 11434 ? ' — probably by Ollama itself; quit it first' : ''}.\n` +
          'Pick another with --port <n>.\n',
      );
      return 1;
    }
    const root = running.url.replace(/\/v1$/, '');
    process.stderr.write(
      `apple-llm serving Apple's models\n` +
        `  OpenAI: ${running.url}  (base URL for OpenAI SDKs, LangChain, Open WebUI, editor plugins)\n` +
        `  Ollama: ${root}${port === 11434 ? '' : '  (--port 11434 for apps that only look on Ollama\'s port)'}\n` +
        `  models: apple-on-device${flags.tier === 'cloud' ? ' (default: apple-private-cloud)' : ''}, apple-private-cloud\n` +
        `  audio:  ${running.url}/audio/speech, ${running.url}/audio/transcriptions\n` +
        `  try:    curl ${running.url}/chat/completions -H 'content-type: application/json' ` +
        `-d '{"model":"apple-on-device","messages":[{"role":"user","content":"Hello"}]}'\n` +
        (flags.host === '0.0.0.0' && flags.apiKey === undefined
          ? '  warning: listening on every interface with no --api-key; anyone on your network can use it\n'
          : ''),
    );
    await new Promise<void>((resolve) => {
      const stop = (): void => {
        void running.close().then(resolve);
      };
      process.once('SIGINT', stop);
      process.once('SIGTERM', stop);
    });
    return 0;
  }

  if (command === 'chat') {
    if (flags.tier !== undefined && !['device', 'cloud', 'auto'].includes(flags.tier)) {
      return usageError(`--tier must be device, cloud, or auto (got "${flags.tier}").`);
    }
    let tools: ('ocr' | 'barcode' | 'spotlight')[] | undefined;
    try {
      tools = parseTools(flags.tool, usageError) as typeof tools;
    } catch (err) {
      return usageError(err instanceof Error ? err.message : String(err));
    }
    const maxTokens = flags.maxTokens === undefined ? undefined : Number(flags.maxTokens);
    const llm = new AppleLLM({
      tier: (flags.tier ?? 'auto') as 'device' | 'cloud' | 'auto',
      fallback,
      maxTokens,
      useCase: flags.useCase as 'general' | 'contentTagging' | undefined,
      guardrails: flags.guardrails as 'default' | 'permissive' | undefined,
    });
    try {
      await runRepl(llm, { system: flags.system, call: { tools } });
    } finally {
      llm.close();
    }
    return 0;
  }

  if (command === 'run') {
    const arg = flags.positional[0];
    if (arg === undefined) {
      process.stderr.write('run needs a prompt, or "-" to read stdin.\n');
      return 2;
    }
    if (flags.tier !== undefined && flags.tier !== 'device' && flags.tier !== 'cloud' && flags.tier !== 'auto') {
      return usageError(`--tier must be device, cloud, or auto (got "${flags.tier}").`);
    }
    const temperature = flags.temperature === undefined ? undefined : Number(flags.temperature);
    if (temperature !== undefined && Number.isNaN(temperature)) {
      return usageError(`--temperature must be a number (got "${flags.temperature}").`);
    }
    const maxTokens = flags.maxTokens === undefined ? undefined : Number(flags.maxTokens);
    if (maxTokens !== undefined && Number.isNaN(maxTokens)) {
      return usageError(`--max-tokens must be a number (got "${flags.maxTokens}").`);
    }
    const timeoutMs = flags.timeout === undefined ? undefined : Number(flags.timeout) * 1000;
    if (timeoutMs !== undefined && (Number.isNaN(timeoutMs) || timeoutMs <= 0)) {
      return usageError(`--timeout must be a number of seconds (got "${flags.timeout}").`);
    }
    const seed = flags.seed === undefined ? undefined : Number(flags.seed);
    if (seed !== undefined && Number.isNaN(seed)) {
      return usageError(`--seed must be a number (got "${flags.seed}").`);
    }
    if (flags.useCase !== undefined && flags.useCase !== 'general' && flags.useCase !== 'contentTagging') {
      return usageError(`--use-case must be general or contentTagging (got "${flags.useCase}").`);
    }
    if (flags.guardrails !== undefined && flags.guardrails !== 'default' && flags.guardrails !== 'permissive') {
      return usageError(`--guardrails must be default or permissive (got "${flags.guardrails}").`);
    }
    const prompt = await promptFrom(arg);
    const tier = (flags.tier ?? 'auto') as 'device' | 'cloud' | 'auto';
    let tools: ('ocr' | 'barcode' | 'spotlight')[] | undefined;
    try {
      tools = parseTools(flags.tool, usageError) as typeof tools;
    } catch (err) {
      return usageError(err instanceof Error ? err.message : String(err));
    }
    const sampling =
      flags.greedy === true
        ? ({ mode: 'greedy' } as const)
        : seed !== undefined
          ? ({ mode: 'topK', k: 50, seed } as const)
          : undefined;
    const llm = new AppleLLM({
      tier,
      fallback,
      temperature,
      maxTokens,
      useCase: flags.useCase as 'general' | 'contentTagging' | undefined,
      guardrails: flags.guardrails as 'default' | 'permissive' | undefined,
      sampling,
      timeoutMs,
      onProgress: (p) => process.stderr.write(`  ${p.status}\n`),
    });
    const images = flags.image.length > 0 ? flags.image.map(parseImageFlag) : undefined;
    const documents = flags.document.length > 0 ? flags.document : undefined;
    try {
      if (flags.schema !== undefined) {
        let schemaText: string;
        try {
          schemaText = await readFile(flags.schema, 'utf8');
        } catch (err) {
          throw new AppleLLMError(
            `Could not read schema file "${flags.schema}": ${err instanceof Error ? err.message : String(err)}`,
          );
        }
        let schema: Record<string, unknown>;
        try {
          schema = JSON.parse(schemaText) as Record<string, unknown>;
        } catch (err) {
          throw new AppleLLMError(
            `Schema file "${flags.schema}" is not valid JSON: ${err instanceof Error ? err.message : String(err)}`,
          );
        }
        const call = {
          schema,
          system: flags.system,
          webSearch: flags.webSearch,
          images,
          documents,
          sessionId: flags.session,
          tools,
        };
        if (flags.stream === true) {
          // Newline-delimited partial objects, then the final one: pipeable.
          const stream = llm.streamJson(prompt, call);
          for await (const partial of stream) process.stdout.write(`${JSON.stringify(partial)}\n`);
          const final = await stream;
          process.stdout.write(`${JSON.stringify(final)}\n`);
        } else if (flags.json === true) {
          process.stdout.write(`${JSON.stringify(await llm.generate(prompt, call), null, 2)}\n`);
        } else {
          process.stdout.write(`${JSON.stringify(await llm.json(prompt, call), null, 2)}\n`);
        }
      } else if (flags.stream === true) {
        const out = await llm.stream(prompt, {
          system: flags.system,
          images,
          documents,
          sessionId: flags.session,
          tools,
          onDelta: (delta) => process.stdout.write(delta),
        });
        process.stdout.write('\n');
        void out;
      } else {
        const call = {
          system: flags.system,
          webSearch: flags.webSearch,
          images,
          documents,
          sessionId: flags.session,
          tools,
        };
        if (flags.json === true) {
          process.stdout.write(`${JSON.stringify(await llm.generate(prompt, call), null, 2)}\n`);
        } else {
          process.stdout.write(`${await llm.text(prompt, call)}\n`);
        }
      }
    } finally {
      llm.close();
    }
    return 0;
  }

  process.stderr.write(`Unknown command "${command}".\n\n${USAGE}`);
  return 2;
}

main()
  .then((code) => {
    process.exitCode = code;
  })
  .catch((err: unknown) => {
    if (err instanceof AppleLLMError) {
      process.stderr.write(`${err.name}: ${err.message}\n`);
    } else {
      process.stderr.write(`${err instanceof Error ? err.stack ?? err.message : String(err)}\n`);
    }
    process.exitCode = 1;
  });
