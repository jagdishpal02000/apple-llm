# apple-llm

Apple's built-in LLMs from Node — the on-device model and Private Cloud
Compute — plus the voices and speech recognition built into macOS. No API key,
no account, no model download, **zero runtime dependencies**.

[![npm](https://img.shields.io/npm/v/apple-llm)](https://www.npmjs.com/package/apple-llm)
[![npm downloads](https://img.shields.io/npm/dw/apple-llm)](https://www.npmjs.com/package/apple-llm)
[![GitHub stars](https://img.shields.io/github/stars/jagdishpal02000/apple-llm?style=social)](https://github.com/jagdishpal02000/apple-llm)

[Website](https://jagdishpal02000.github.io/apple-llm/) ·
[GitHub](https://github.com/jagdishpal02000/apple-llm) ·
[Changelog](https://github.com/jagdishpal02000/apple-llm/blob/main/packages/node/CHANGELOG.md) ·
[Issues](https://github.com/jagdishpal02000/apple-llm/issues)

```bash
npm install apple-llm
```

```ts
import { AppleLLM } from 'apple-llm';

const llm = new AppleLLM();
await llm.text('Summarize this in one line: …');
```

That is the whole setup. A small Swift helper compiles on first use (a few
seconds, once, with the Xcode Command Line Tools) and stays warm between calls.

- **Everything an LLM SDK should have** — streaming (`for await`), structured
  output typed by Zod, tool calling with your own JavaScript functions,
  multi-turn chat, cancellation, timeouts, token counting, usage.
- **Four ways in** — this library; a [server](#openai-and-ollama-compatible-server)
  that speaks both OpenAI's API and Ollama's, so apps built for either use
  Apple's model with no code; a [Vercel AI SDK provider](#vercel-ai-sdk); and
  the [shell](#cli): `git diff | npx apple-llm "Write a commit message for this"`.
- **Two tiers** — on-device (private, fast, free) and Apple's big server
  model on Private Cloud Compute — on macOS 27 Golden Gate, the one Siri AI is
  built on: reasoning, images, live web knowledge, 32k context. Free, no key,
  behind the same API.
- **[Speech](#speech)** — every voice on your Mac, and speech recognition for
  recordings of any length or live from the microphone. No Apple Intelligence
  needed: `await speak('Hello there')`, or `npx apple-llm speak "Hello there"`.

## Read this first

- **macOS 26+ on Apple Silicon, with Apple Intelligence on** for the models
  (speech does not need it). Nothing works elsewhere, but importing always succeeds
  and `probe()` says exactly why.
- **The Xcode Command Line Tools** (`xcode-select --install`), for compiling
  the helper. Full Xcode is not needed.
- **The on-device model is small** (~20B sparse, 1–4B active; 8,192-token
  context). Good at extraction, classification, tagging, rewriting, summaries
  and short answers. Bad at code and long reasoning. Use it for the jobs it
  suits.
- **The cloud tier is not local.** `tier: 'cloud'` sends your prompt to Apple's
  Private Cloud Compute. The default, `auto`, uses the device when it can.
- **macOS's voices come with macOS's terms.** Apple's licence allows them for
  personal, non-commercial use and rules out recording or publishing them in a
  commercial context. [Read this](https://github.com/jagdishpal02000/apple-llm#the-licence-on-voices)
  before building a product on text to speech.

## Quick tour

### Structured output, typed by your schema

On device, the shape is **guaranteed** by constrained decoding — the model
cannot produce anything else. Pass a Zod 4 (or ArkType, or Valibot) schema and
the result is typed and validated; plain JSON Schema works too.

```ts
import { z } from 'zod';

const Contact = z.object({
  name: z.string(),
  email: z.string().email(),
  phone: z.string().nullable(),
});

const contact = await llm.json('Reach Grace Hopper at grace@navy.mil', { schema: Contact });
//    ^? { name: string; email: string; phone: string | null }
```

Refinements the decoder cannot enforce (`.email()`, `.min()`) are checked
afterwards; a reply that fails gets one automatic retry with the problems
pointed out, then a `SchemaValidationError` carrying the issues and the raw
text.

### Streaming

```ts
for await (const delta of llm.stream('Write a haiku about bridges')) {
  process.stdout.write(delta);
}

const text = await llm.stream('…');              // or just await the whole reply
const { usage } = await llm.stream('…').result;   // or everything about it
```

Breaking out of the loop cancels generation. Structured output streams too —
partial objects as the model fills them in, for rendering a card or form while
it is still being written:

```ts
for await (const partial of llm.streamJson(prompt, { schema: Contact })) {
  render(partial); // { name: 'Grace' } → { name: 'Grace', email: '…' } → …
}
```

### Tools: your JavaScript, called by the model

```ts
import { tool } from 'apple-llm';

const getWeather = tool({
  name: 'getWeather',
  description: 'Current weather for a city',
  parameters: z.object({ city: z.string() }),
  execute: async ({ city }) => fetchWeather(city), // city: string
});

const result = await llm.generate('Do I need an umbrella in Paris?', {
  tools: [getWeather],
});
result.text;      // "No — it's 21°C and sunny in Paris."
result.toolCalls; // [{ name: 'getWeather', arguments: { city: 'Paris' }, output: '…' }]
```

`execute` runs in your process, mid-generation, with your credentials and
database. Arguments are schema-guaranteed like `json()`. A tool that throws
ends the call with a `ToolExecutionError` whose `cause` is your error. Tools
mix freely with Apple's built-ins: `tools: ['ocr', getWeather]`.

Leave out `execute` and generation stops at the call instead
(`finishReason: 'tool-calls'`), so you can run it yourself — send the result
back as a `tool` message to continue.

### Conversations

Pass a message list anywhere a prompt goes:

```ts
import type { ChatMessage } from 'apple-llm';

const messages: ChatMessage[] = [
  { role: 'system', content: 'You are terse.' },
  { role: 'user', content: 'My cat is called Biscuit.' },
];
const reply = await llm.generate(messages);
messages.push(reply.message, { role: 'user', content: 'What is my cat called?' });
await llm.text(messages); // "Biscuit."
```

History becomes Apple's native transcript, not a text preamble. When a long
conversation outgrows the context window, the oldest turns are dropped to fit
and `result.trimmedTurns` says how many (`trimHistory: false` to get a
`ContextLengthError` instead).

Or let the helper keep the thread — it survives restarts:

```ts
const chat = llm.conversation('support-ticket-42', { system: 'You are terse.' });
await chat.text('My cat is called Biscuit.');
await chat.text('What is my cat called?'); // "Biscuit."
```

### Cancellation and timeouts

```ts
await llm.text(prompt, { signal: controller.signal }); // AbortError
await llm.text(prompt, { timeoutMs: 5_000 });          // TimeoutError
new AppleLLM({ timeoutMs: 30_000 });                   // default for every call
```

Cancelling stops the model itself, not just the promise: the next call does not
wait behind a reply nobody wants.

### Images, documents, long text

```ts
await llm.text('What is in this photo?', { images: ['./photo.jpg'] });
await llm.text('Read the total.', { images: ['./receipt.png'], tools: ['ocr'] });
await llm.text('Compare these quotes.', { documents: ['./a.md', './b.md'] });
await llm.summarize(fiftyPageReport); // longer than the window? summarised in parts
```

### Knowing what a call costs

```ts
const { tokens, contextSize } = await llm.countTokens(messages, { schema, tools });
const { usage, finishReason } = await llm.generate(prompt);
// usage: { inputTokens, outputTokens, totalTokens, cachedInputTokens }
// finishReason: 'stop' | 'length' | 'tool-calls' — 'length' means maxTokens cut it off
```

### The big model: Private Cloud Compute

```ts
const cloud = new AppleLLM({ tier: 'cloud' }); // after `npx apple-llm setup-cloud`, once

await cloud.text(puzzle);                                   // reasons it through
await cloud.text('What is in this photo?', { images: ['./photo.jpg'] });
await cloud.text('Who won the last World Cup?', { webSearch: true }); // live web knowledge
await cloud.json(prompt, { schema: Contact });              // typed, validated
```

On macOS 27 Golden Gate this reaches the next-generation server model Apple
built Siri AI on. `probe()` reads what it can do straight from the framework —
reasoning, vision, tool calling and a 32,768-token window — without calling it.
Measured against the on-device model: it solves multi-step logic puzzles the
small model loses track of, finds a fact buried in 25,000 tokens, reads images,
and with `webSearch` answers questions about last month's news correctly. It
costs no API key and no money, only Apple's per-device quota
(`probe().cloud.quota`), and your prompt leaves the Mac.

**Apple rate-limits it per Mac**, separately from that quota and invisibly to
`probe()`: ten calls in about two minutes tripped the limit here, and every call
was then refused for a few minutes. A refused call is a `QuotaError` (HTTP 429 from
the server). Use the cloud for the calls that need it, not for batch jobs, and
let the device answer when it says no:

```ts
const cloud = new AppleLLM({ tier: 'cloud', fallback: 'device' });
const { text, tier } = await cloud.generate(prompt); // tier: 'device' when the cloud said no
```

## Speech

```ts
import { listen, listVoices, speak, synthesize, toSRT, transcribe, transcribeStream } from 'apple-llm';

await speak('Hello there');                                  // out loud, in your System voice
await speak('Bonjour', { language: 'fr' });                  // the best French voice on this Mac
await speak('Faster', { voice: 'Samantha', speed: 1.5, pitch: 1.1, volume: 0.8 });

const voices = await listVoices({ language: 'en', novelty: false });

const { audio, duration } = await synthesize('Hello there', { format: 'wav' });     // a Buffer
await synthesize(longArticle, { voice: 'Samantha', out: 'article.m4a' });           // or a file
const { words } = await synthesize('Every word has a time.', { words: true });

const { text, segments, language } = await transcribe('meeting.mp4');               // audio or video
const subtitles = toSRT(await transcribe(buffer, { language: 'en-US', words: true }));
for await (const segment of transcribeStream('long-interview.m4a')) console.log(segment.text);

const said = await listen();                                 // returns when you pause
const session = listen({ continuous: true });                // until session.stop()
for await (const event of session) console.log(event.text, event.final);
```

None of this needs Apple Intelligence or costs quota, and it runs on your Mac; the
only network use is downloading a recognition language the first time you use it.

- **Length is not a limit.** On an M4 Air, 30 minutes of speech renders in 18
  seconds and transcribes in 28, and a read-aloud can run as long as you like. For
  long text pass `out`: 30 minutes is 11.3 MB as M4A and 80.4 MB as WAV, and without
  `out` it is all in memory.
- **Voices.** `voice` is a name or an id; an unknown one is an error suggesting
  close matches. The default is your System voice, and better ones are free
  downloads: search System Settings for "System voice". A few voices are only
  reachable through `say` (`engine: 'say'`): those have speed, but no pitch, volume,
  SSML or word timings, and asking for those is an error. They can also lose words
  while Apple's language model is busy in another process, so a file is rendered
  twice and kept only when the two agree; speaking aloud cannot be checked that way.
  `avspeech` voices are not affected.
- **Siri voices.** Apple does not let apps choose them. Where macOS lists one under
  the System voice setting, choose it there and `speak()` with no `voice` should
  use it; this path has not been tested with a Siri voice selected.
  [More](https://github.com/jagdishpal02000/apple-llm#siri-voices).
- **Formats.** `wav`, `m4a`, `aac`, `flac`, `aiff`, `caf` and `pcm` (raw samples).
  macOS cannot encode MP3, Ogg or Opus. It reads WAV, AIFF, CAF, MP3, M4A, AAC, FLAC
  and MP4 or MOV video, but not WebM, Ogg or MKV.
- **Recognition.** Name the `language`, because it is not detected; the default is
  this Mac's. 68 languages on macOS 27; a missing one downloads on first use, in
  about 25 seconds, unless `download: false`. `words: true` adds word timings,
  `hints` names terms it might miss, and `onProgress` reports both downloads and
  progress through a long file.
- **The microphone.** macOS asks once, on behalf of whichever app launched this. It
  hears everything, including `speak()` through your speakers. `stop()` finishes and
  keeps what it heard; `abort()` discards it. A `final: false` event is a guess the
  next one replaces.
- **One at a time.** The functions share one helper, which handles one request at a
  time. Use `new AppleSpeech()` for work that must not wait; it has a helper of its
  own, and `close()` releases it.

[examples/speech.ts](https://github.com/jagdishpal02000/apple-llm/blob/main/packages/node/examples/speech.ts)
runs through all of this, and
[examples/voice-assistant.ts](https://github.com/jagdishpal02000/apple-llm/blob/main/packages/node/examples/voice-assistant.ts)
is a voice assistant in about twenty lines. The
[repository README](https://github.com/jagdishpal02000/apple-llm#speech) has the
measurements and the licence.

## OpenAI- and Ollama-compatible server

```bash
npx apple-llm serve
#   OpenAI: http://127.0.0.1:11436/v1
#   Ollama: http://127.0.0.1:11436
```

Point any OpenAI client at it — the official SDKs, LangChain, LlamaIndex,
editor plugins, Open WebUI — with any API key:

```ts
import OpenAI from 'openai';
const openai = new OpenAI({ baseURL: 'http://127.0.0.1:11436/v1', apiKey: 'unused' });
await openai.chat.completions.create({ model: 'apple-on-device', messages });
```

`/v1/chat/completions` supports streaming, tool calls, `response_format` with
a JSON Schema (guaranteed on device), images as data or http URLs, `seed` and
`top_p`. It binds to localhost, sends no CORS headers unless you pass
`--cors <origin>`, can require `--api-key`, and serves the on-device model
unless a client asks for `apple-private-cloud` by name. Or from code:
`import { serve } from 'apple-llm/server'`.

### Audio

`POST /v1/audio/speech`, `POST /v1/audio/transcriptions` and `GET /v1/audio/voices`
do text to speech and speech recognition for OpenAI clients:

```ts
const speech = await openai.audio.speech.create({ model: 'tts-1', voice: 'Samantha', input: 'Hello there' });
await openai.audio.transcriptions.create({ model: 'whisper-1', file: await toFile(buffer, 'clip.m4a') });
```

`model` is ignored. OpenAI's voice names mean your System voice, since many clients
cannot be told otherwise. Requests for `mp3` or `opus`, and requests naming no format,
get AAC with a `Content-Type` that says so, because macOS cannot encode them and
refusing would lock out every app that hardcodes MP3. Transcripts come as `json`,
`text`, `srt`, `vtt` or `verbose_json`.

### Ollama apps

Most Mac apps for local models — Open WebUI, Enchanted, Msty, Page Assist,
Continue, Zed, Obsidian plugins — speak Ollama's API. The same server answers
it, so they can use Apple's model with no code. Start it on Ollama's port (with
Ollama itself not running) and pick `apple-on-device` in the app:

```bash
npx apple-llm serve --port 11434
ollama run apple-on-device "Why is the sky blue?"   # even the Ollama CLI works against it
```

`/api/chat` and `/api/generate` stream NDJSON unless `stream: false`, as in
Ollama, and take `format` (`"json"`, or a JSON Schema — guaranteed on device),
tools, images and `options` (`temperature`, `num_predict`, `seed`, `top_p`).
`/api/tags`, `/api/show` (context length and capabilities), `/api/ps` and
`/api/version` answer the discovery calls apps make first. Embeddings are not
served: Apple's models do not produce them.

## Vercel AI SDK

```ts
import { generateText, streamText, stepCountIs, tool } from 'ai';
import { apple } from 'apple-llm/ai-sdk';

const { text } = await generateText({
  model: apple(), // on device; apple('cloud') for Private Cloud Compute
  prompt: 'What is the weather in Paris?',
  tools: { getWeather: tool({ inputSchema: z.object({ city: z.string() }), execute: … }) },
  stopWhen: stepCountIs(3),
});
```

AI SDK 6 and 7. `generateText`, `streamText`, `generateObject` / `Output.object`
(constrained on device), multi-step tools, images and abort signals. Apple-only
settings go in `providerOptions: { apple: { useCase, guardrails, builtInTools } }`.

## CLI

```bash
apple-llm "What is the capital of Australia?"   # one completion (short for run)
git diff | apple-llm "Write a commit message"   # piped text goes in before the prompt
cat error.log | apple-llm "What went wrong?"
apple-llm chat                                  # interactive, streamed; Ctrl+C stops a reply
apple-llm run "Summarize: …"                    # one completion; "-" reads stdin
apple-llm run --json "…"                        # text + usage + tool calls as JSON
apple-llm run --schema person.json "Ada, 1815"  # guaranteed JSON
apple-llm run --schema person.json --stream "…" # partial objects, one per line
apple-llm run --image photo.png "What is this?"
apple-llm run --tool ocr --image receipt.png "Read the total."
apple-llm serve [--port 11436] [--api-key k]    # OpenAI- and Ollama-compatible API
apple-llm count --system "…" -                  # tokens before sending
apple-llm probe                                 # what this machine can do
apple-llm setup-cloud                           # one-time, for the cloud tier
apple-llm speak "Hello there" --voice Samantha  # out loud; no quotes needed
apple-llm speak --file notes.txt -o notes.m4a   # or save it, any length
apple-llm voices [--language fr] [--all]        # the voices on this Mac
apple-llm transcribe meeting.mp4 -o meeting.srt # the extension picks text, json, srt or vtt
apple-llm listen [--once]                       # the microphone, live; Ctrl-C keeps what it heard
apple-llm languages [install|remove <code>]     # what it can recognise
```

`apple-llm --help` for everything, including the Write-with-Siri presets
(`rewrite`, `proofread`, `summarize`, `draft`) and `ask-screen`.

## Two traps worth knowing

**Never set `temperature: 0`.** Constrained decoding already guarantees the
schema, so greedy decoding buys nothing and reliably degenerates — it padded an
unbounded array forever, then ran away inside a single string, turning a 2s
call into 20s. The default is `0.4`. For reproducible output use a seed
instead: `sampling: { mode: 'topK', k: 50, seed: 42 }` returns byte-identical
text across runs, without the degeneration.

**Bound your arrays.** Apple honours `maxItems` (`.max(5)` in Zod) but ignores
`maxLength` on strings.

## The two tiers

|  | on-device | cloud (Private Cloud Compute) |
|---|---|---|
| Model | AFM 3 Core (~1–4B active) | Apple's next-gen server model (Siri AI's, on macOS 27) |
| Privacy | nothing leaves the Mac | **your prompt leaves the Mac** |
| Context | 8,192 tokens (4,096 on macOS 26) | 32,768 tokens |
| Reasoning | no | yes |
| Latency | ~0.3–2s warm | ~2s, ~10s at 25k tokens |
| JSON | guaranteed by constrained decoding | requested, recovered, validated |
| Streaming | yes, text and objects | one chunk (Shortcuts is not incremental) |
| Images | yes (macOS 27) | yes (macOS 27 Golden Gate) |
| Web knowledge | no | yes, with `webSearch` |
| Your tools, sessions | yes | no |
| Setup | none | `apple-llm setup-cloud`, once |
| Limits | none | a per-Mac rate limit, and a quota (`probe().cloud.quota`) |

## Errors

Every error is an `AppleLLMError` with a stable `code`, and a class to branch on:

| class | when |
|---|---|
| `ModelUnavailableError` | Apple Intelligence off, model downloading, unsupported Mac/OS (`.reason` says which) |
| `ContextLengthError` | the request does not fit (`.contextSize`, `.tokenCount`) |
| `SchemaRejectedError` | Apple's decoder refused a schema |
| `SchemaValidationError` | the reply failed your schema's validation (`.issues`, `.text`) |
| `ToolExecutionError` | your tool threw (`.toolName`, `.cause`) |
| `AbortError` / `TimeoutError` | your signal fired / the deadline passed |
| `QuotaError` | Apple's cloud rate limit or quota (`.resetDate` when Apple gives one); `fallback: 'device'` answers on device instead |
| `RefusalError` | the model's guardrails declined |
| `ModelBusyError` | other processes are using the model; already retried with backoff |
| `SetupRequiredError`, `UnsupportedError` | a setup step or speech language is missing / the tier, voice or OS cannot do that |
| `PermissionDeniedError` | macOS refused the microphone; the message names the setting |

## Resource use

One `AppleLLM` keeps one helper process warm — construct it once. The helper
does not keep your process alive, so a script exits without `close()`; call
`close()` (or use `await using llm = new AppleLLM()`) to release it early in a
long-running program. Apple serialises inference, so requests queue rather than
run in parallel — concurrency buys nothing here.

## More

The [repository README](https://github.com/jagdishpal02000/apple-llm#readme)
has the measurements behind every default, the nine rules of Apple's schema
dialect, and why the cloud tier goes through Shortcuts. If apple-llm saved you
time, a [⭐ on GitHub](https://github.com/jagdishpal02000/apple-llm) helps other
Mac developers find it.

MIT. Not affiliated with or endorsed by Apple Inc.
