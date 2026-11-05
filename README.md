# apple-llm

**The free LLM already on your Mac, from Node and Python.** Apple's two
built-in models — the on-device Foundation Models LLM behind Apple
Intelligence, and Apple's large model on Private Cloud Compute. No API key, no
account, no developer program membership, no Xcode — nothing to install beyond
the package and Apple's Command Line Tools.

**It also gives you the rest of what macOS has built in:** text to speech in every
voice on your Mac, and speech to text for recordings of any length or live from the
microphone. [Speech](#speech) needs no Apple Intelligence, and it runs on your Mac.

[![npm](https://img.shields.io/npm/v/apple-llm?label=npm)](https://www.npmjs.com/package/apple-llm)
[![npm downloads](https://img.shields.io/npm/dw/apple-llm)](https://www.npmjs.com/package/apple-llm)
[![PyPI](https://img.shields.io/pypi/v/apple-llm)](https://pypi.org/project/apple-llm/)
[![PyPI downloads](https://img.shields.io/pypi/dm/apple-llm)](https://pypi.org/project/apple-llm/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue)](LICENSE)
![macOS 26+ on Apple Silicon](https://img.shields.io/badge/macOS_26%2B-Apple_Silicon-black?logo=apple)

**[Website](https://jagdishpal02000.github.io/apple-llm/)** ·
[Node API](packages/node/README.md) ·
[Python](packages/python/README.md) ·
[AI agent guide](llms.txt) ·
[Changelog](packages/node/CHANGELOG.md)

```bash
npm install apple-llm
pip install apple-llm
```

```ts
import { AppleLLM } from 'apple-llm';

const llm = new AppleLLM({ tier: 'device' });
await llm.text('Summarize this in one line', { system: 'You are terse.' });
await llm.json('Extract the fields', { schema });   // guaranteed to match, on device
llm.close();
```

```python
from apple_llm import AppleLLM

with AppleLLM(tier="device") as llm:
    llm.text("Summarize this in one line", system="You are terse.")
    llm.json("Extract the fields", schema=Person)   # a pydantic model, validated back
```

Or skip the SDK entirely. From the shell:

```bash
npx apple-llm "What is the capital of Australia?"
git diff | npx apple-llm "Write a commit message for this"   # piped text goes before the prompt
```

Or point any app at it. The server speaks both OpenAI's API — the official
SDKs, LangChain, LlamaIndex, editor plugins — and **Ollama's**, which most Mac
apps for local models use (Open WebUI, Enchanted, Msty, Page Assist, Continue,
Zed, Obsidian plugins):

```bash
npx apple-llm serve                # OpenAI at http://127.0.0.1:11436/v1, model "apple-on-device"
npx apple-llm serve --port 11434   # Ollama's port (with Ollama itself not running)
ollama run apple-on-device "Hello" # even the Ollama CLI works against it
```

Or use your Mac's voices and ears:

```bash
npx apple-llm speak "Hello there"             # out loud, in your System voice
npx apple-llm speak --file book.txt -o book.m4a   # any length: 30 minutes takes 18 seconds
npx apple-llm transcribe meeting.mp4          # audio or video to text
npx apple-llm listen                          # the microphone, live
```

Both packages compile one small Swift helper on first use and share the compiled
binary, so installing both costs one compile, not two.

## Integrate with an AI agent

Give your coding agent this single reference:
**[apple-llm/llms.txt](https://jagdishpal02000.github.io/apple-llm/llms.txt)**
([source](llms.txt)). It covers setup, local LLM, TTS, STT, Node/Python SDKs,
HTTP APIs, examples, and troubleshooting.

> Read https://jagdishpal02000.github.io/apple-llm/llms.txt and integrate
> apple-llm into this project for local LLM, text to speech, and speech to text.
> Use the project's existing language, keep processing on device, and include
> availability checks and setup instructions.

The link is an agent-readable guide. Agents integrate through the SDK, CLI, or
OpenAI-compatible server; an MCP host can wrap those APIs in its own adapter.

## Read this before you use it

**macOS 26 or later, on Apple Silicon.** Nothing here works on Intel Macs,
Linux, or Windows. Importing the package still succeeds everywhere — `probe()`
tells you why the model is unavailable rather than failing at install time — but
there is no fallback. This is a wrapper around hardware you either have or
don't.

**The Xcode Command Line Tools.** The Swift helper is compiled on first use
with `swiftc`, which ships with them (`xcode-select --install`). Full Xcode is
not needed.

**The on-device model is small.** `AFM 3 Core Advanced` is roughly 20B sparse
with 1–4B active, with an 8192-token context on macOS 27 (4096 on macOS 26). It
is good at classification, extraction, tagging, rewriting, and short prose. It
is **bad at code generation and long reasoning**, and no amount of prompting
fixes that. Use it for the shape of work it suits.

**The cloud tier is not local.** It sends your prompt to Apple's Private Cloud
Compute — off your machine. Never describe it as local. It is free but quota'd,
it is reached through a private Shortcuts action that Apple can change or remove
in any OS release, and it has no constrained decoding, so `json()` there is a
request rather than a guarantee.

**The cloud tier is rate-limited per Mac.** Apple's Private Cloud Compute stops
answering a Mac that calls it too often: ten calls in about two minutes tripped
it here, and it then refused every call for a few minutes — while `probe()` still
reported the quota as `belowLimit`, because this is a separate, short-term
limiter that nothing can read in advance. A refused call raises `QuotaError`
(HTTP 429 from the server). Use the cloud for the calls that need it rather
than for batch jobs, and pass `fallback: 'device'` to have the on-device model
answer when the cloud says no.

**Speech uses macOS's voices, on macOS's terms.** Apple's licence allows its
system voices for your personal, non-commercial use, and rules out recording or
publishing them in a commercial context. That covers every voice, not just Siri's,
and it is the one thing to check before building on text to speech:
[the licence on voices](#the-licence-on-voices).

**What 0.3 adds.** [Text to speech and speech to text](#speech), in both packages
and on the server. Fixes for the two ways the cloud tier failed — Apple's
per-device rate limit surfaced as an opaque "Your request can't be completed
right now", and cloud `json()` answered in prose — plus `fallback: 'device'`,
an Ollama-compatible API on `apple-llm serve`, and a shell-friendly CLI
(`apple-llm "question"`, piped input). Both packages; see the
[CHANGELOG](packages/node/CHANGELOG.md).

**What 0.2 adds (npm package).** Function tools executed in your process,
Zod/Standard Schema structured output, `for await` streaming of text and of
partial objects, message-list conversations with automatic trimming,
cancellation and timeouts, token usage, an OpenAI-compatible server
(`apple-llm serve`) and a Vercel AI SDK provider — and, on macOS 27 Golden
Gate, images on the cloud tier, whose server model is the one Siri AI is built
on (see [below](#macos-27-golden-gate-reaching-siri-ais-model)). See
[packages/node/README.md](packages/node/README.md) and its
[CHANGELOG](packages/node/CHANGELOG.md). The Python package shares the new helper
and keeps working unchanged; it does not expose these features yet.

## The two tiers

|  | on-device | cloud (Private Cloud Compute) |
|---|---|---|
| Model | AFM 3 Core Advanced (~20B sparse, 1–4B active) | Apple's next-gen server model — Siri AI's, on macOS 27 Golden Gate |
| Context | 8192 tokens (4096 on macOS 26) | 32768 tokens (a needle at 25k found; 41k refused) |
| Typical latency | 0.3–2s warm | ~2s, ~11s at 25k tokens |
| JSON | **guaranteed** by constrained decoding | asked for in the prompt, recovered from prose |
| Privacy | nothing leaves the machine | **your prompt leaves the machine** |
| Setup | none | one-time `apple-llm setup-cloud` |
| Limits | none | a per-Mac rate limit and a quota ([above](#read-this-before-you-use-it)) |
| Vision | yes (macOS 27) | yes (macOS 27 Golden Gate), passed as a Shortcuts input |
| Reasoning | **no** (the model says so) | yes |
| Web search | no | optional `WFAllowWebSearch` — answers about last month's news, correctly |
| Streaming | text and partial objects | one chunk |
| Tools | your functions + `ocr`, `barcode`, `spotlight` (macOS 27) | no |
| Conversations | message lists, named `sessionId` threads | message lists, as text |

`tier: 'auto'` uses the device when it is available, else the cloud when the
shortcut is installed, else raises an error naming the setup step.

## Speech

The voices and speech recognition built into macOS, through the same helper.
Neither touches the language model, so both work with **Apple Intelligence turned
off**, and they cost no quota. Everything runs on your Mac; the only network use
is downloading a recognition language the first time you use it.

From the shell, no code needed:

```bash
apple-llm speak "Hello there"                          # out loud, in your System voice
apple-llm speak hello there --voice Samantha           # any voice; no quotes needed
apple-llm speak "Bonjour tout le monde" --language fr  # the best French voice on this Mac
apple-llm speak --file notes.txt --speed 1.5           # read a file aloud, faster
apple-llm speak "Welcome" -o welcome.m4a               # save it instead
apple-llm voices                                       # the voices you have
apple-llm transcribe meeting.mp4                       # audio or video to text
apple-llm transcribe meeting.mp4 -o meeting.srt        # subtitles; the extension picks the format
apple-llm listen                                       # the microphone, live; Ctrl-C stops and keeps it
apple-llm languages                                    # what it can recognise
```

From code:

```ts
import { listen, speak, synthesize, transcribe } from 'apple-llm';

await speak('Hello there', { voice: 'Samantha', speed: 1.2 });

const { audio } = await synthesize('Hello there', { format: 'wav' });   // a Buffer
await synthesize(longArticle, { voice: 'Samantha', out: 'article.m4a' }); // or a file

const { text, segments } = await transcribe('meeting.m4a', { language: 'en-US' });

const said = await listen();                                            // returns when you pause
```

```python
from apple_llm import listen_once, speak, synthesize, transcribe

speak("Hello there", voice="Samantha", speed=1.2)

audio = synthesize("Hello there", format="wav").audio                   # bytes
synthesize(long_article, voice="Samantha", out="article.m4a")           # or a file

result = transcribe("meeting.m4a", language="en-US")                    # .text, .segments

said = listen_once().text                                               # returns when you pause
```

### Voices

`voice` takes a name (`Samantha`) or an id from `listVoices()` / `list_voices()`.
An unknown one is an error that suggests close matches, unlike `say -v`, which
exits 0 and quietly uses the default. With no `voice` you get your **System
voice**; `language` picks the best voice for a language instead. To change the
System voice or download better ones, which are free, search System Settings for
"System voice".

The Mac this was built on has 184 voices. Two engines reach them, chosen for you:

| `engine` | Voices | Speed | Pitch, volume, SSML | Word timings |
|---|---|---|---|---|
| `avspeech` | almost all of them | yes | yes | all when rendering; live `onWord` events from most |
| `say` | the few `AVSpeechSynthesizer` does not list, like Aman and Tara | yes | no, and asking is an error | no |

`avspeech` is the sturdier of the two. `say` can lose words while Apple's language
model is busy in another process. A file is rendered twice and kept only when the
two agree, so it takes twice as long and is either right or an error, but speaking
aloud cannot be checked that way.

### Siri voices

Apple does not let apps choose Siri voices. None appears in
`AVSpeechSynthesizer`, `NSSpeechSynthesizer` or `say -v`, and asking for one by
name got the default voice instead (macOS 27.2, checked by comparing the audio).

The route that remains is the one Apple offers to people: where macOS lists a
Siri voice under the System voice setting, choose it there, and `speak()` with
no `voice` should use it, through `say`. Pass `voice: 'system'` to force that.
Some macOS versions are reported to speak these voices aloud but render silence
to a file; that raises an error here rather than handing you a silent file.
**This path has not been tested with a Siri voice selected**, so reports are
welcome. Note that the default voice may simply sound like Siri without being one.

### Long audio

There is no length limit to work around. Measured on an M4 MacBook Air, macOS 27.2:

| | Audio | Took | |
|---|---|---|---|
| Render to a file | 30 minutes | 18 seconds | 11.3 MB as M4A, 80.4 MB as WAV, 50 MB of memory |
| Render to a file | 60 minutes | 40 seconds | 22.5 MB as M4A, 62 MB of memory |
| Read aloud | 30 minutes | 30 minutes | it speaks in real time |
| Transcribe, with subtitles | 30 minutes | 28 seconds | 538 cues |

For long text pass `out`, because without it the whole file is held in memory.

### Speech to text

`transcribe` takes a path or the bytes of a file: WAV, AIFF, CAF, MP3, M4A, AAC,
FLAC, or MP4 and MOV video. WebM, Ogg and MKV need converting first, because macOS
cannot read them.

- **Name the language.** macOS does not detect it; the default is this Mac's
  language. `en` and `en-US` both work. It recognises 68 languages on macOS 27: 45
  through `SpeechTranscriber`, and 23 more, such as Russian, Polish, Dutch, Turkish
  and Arabic, through `DictationTranscriber`. The result's `engine` says which ran.
- **Missing languages download on first use**, in about 25 seconds, unless you
  pass `download: false`, which raises `SetupRequiredError` instead. Hindi comes
  back in Latin script (`namaste aaj mausam…`) on macOS 27.2.
- **Timings.** `words: true` adds each word's timing and confidence, and `toSRT()`
  / `toVTT()` use them to keep subtitle cues short enough to read.
- **Hints** name terms it might miss: `hints: ['Kubernetes', 'Postgres']`.
- **Accuracy.** A 3 to 4% word error rate here, on clean synthetic speech. Real
  recordings will do worse, so try it on yours.

### The microphone

```ts
const said = await listen({ silence: 1.5 });        // one utterance

const session = listen({ continuous: true });       // until stop()
for await (const event of session) console.log(event.text, event.final);
session.stop();                                     // keeps what it heard; abort() discards it
```

```python
with listen(continuous=True) as session:
    for event in session:
        print(event.text, event.final)              # session.stop() from another thread
```

A `final: false` event is a guess that the next one replaces. macOS asks once for
microphone access, on behalf of whichever app launched this (Terminal, VS Code,
your own app), and a refusal raises `PermissionDeniedError` naming the setting. It
hears everything, including `speak()` through your speakers, so use headphones
for a conversation. [examples/voice-assistant.ts](packages/node/examples/voice-assistant.ts)
is a voice assistant in about twenty lines.

### Audio formats

| `format` | Extension | |
|---|---|---|
| `wav` | `.wav` | the default for bytes. macOS pads the header to 4 KB, so samples are not at byte 44 |
| `m4a` | `.m4a` | small, plays everywhere |
| `aac` | `.aac` | raw AAC, as browsers and streams like it |
| `flac` | `.flac` | lossless |
| `aiff`, `caf` | `.aiff`, `.caf` | |
| `pcm` | `.pcm` | raw 16-bit little-endian mono samples: use this if you want the samples |

`sampleRate` / `sample_rate` resamples, for example to the 16 kHz many speech tools
want. **macOS cannot encode MP3, Ogg or Opus**, only read them, so asking for those
is an error that says so.

### On the OpenAI-compatible server

`apple-llm serve` (npm) answers `POST /v1/audio/speech`,
`POST /v1/audio/transcriptions` and `GET /v1/audio/voices`, so OpenAI clients can
use it unchanged:

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:11436/v1", api_key="unused")

speech = client.audio.speech.create(model="tts-1", voice="Samantha", input="Hello there")
speech.write_to_file("hello.aac")

client.audio.transcriptions.create(model="whisper-1", file=open("hello.aac", "rb")).text
```

`model` is ignored. OpenAI's voice names (`alloy`, `nova`…) mean your System voice,
because many clients cannot be told otherwise. Requests for `mp3` or `opus`, and
requests naming no format, get AAC, with a `Content-Type` that says so: refusing
would lock out every app that hardcodes MP3. `response_format` works for `wav`,
`flac`, `aac`, `pcm` (24 kHz, as OpenAI defines it), and for transcripts `json`,
`text`, `srt`, `vtt` and `verbose_json`, with `timestamp_granularities[]=word`.
Uploads can be up to 512 MB.

### The licence on voices

This is about **Apple's voices rather than this package**. It applies to all of
them, not just Siri's. Section F of the macOS licence ("Voices; Live Captions")
lets you use them "while running the Apple Software" and "to create your own
original content and projects for your personal, non-commercial use", then says:

> No other creation or use of the System Voices, Live Captions or Personal Voice
> is permitted by this License, including but not limited to the use, reproduction,
> display, performance, recording, publishing or redistribution of any of the System
> Voices, Live Captions or Personal Voice in a profit, non-profit, public sharing or
> commercial context.

So a personal project is fine, and shipping generated audio in a commercial product
is what that clause rules out. Its wording also reaches publishing a recording
publicly, even from a free or non-profit project. Live playback inside a commercial app is less clear,
and Apple has not answered [the developer forum question about
it](https://developer.apple.com/forums/thread/812285). If this matters to your
project, ask a lawyer rather than this README. That clause does not mention
`SpeechAnalyzer`. Read the current text at
[apple.com/legal/sla](https://www.apple.com/legal/sla/).

### What it cannot do

Encode MP3, Ogg or Opus. Detect a language from the audio. Read WebM, Ogg or MKV.
Use Personal Voice: macOS reports it as denied to a command-line helper, so it is
not wired up. Run two things at once on one `AppleSpeech`: a helper handles one
request at a time, so create a second instance for work that must not wait.

## What macOS 27 adds

`probe()` reports what the framework actually says, rather than what the docs
imply — on this machine the on-device model reports vision, guided generation
and tool calling, but **not** reasoning:

```
on-device
  variant     AFM 3 Core Advanced
  context     8192 tokens
  supports    guidedGeneration, toolCalling, vision
  use cases   general, contentTagging

cloud (private cloud compute)
  quota       belowLimit
```

**Count tokens before you send.** Turns a `ContextLengthError` into arithmetic:

```ts
const { tokens, contextSize } = await llm.countTokens(prompt, { system });
```
```bash
apple-llm count --system "You are terse." "The quick brown fox…"   # 64 tokens of 8192 (1%)
```

**Reproducible output, without the `temperature: 0` trap.** Greedy decoding is
deterministic *and* degenerates. Seeded top-k sampling is deterministic and does
not — three fresh runs return byte-identical text:

```ts
await llm.text(prompt, { sampling: { mode: 'topK', k: 50, seed: 42 }, temperature: 0.9 });
```
```python
llm.text(prompt, sampling=sampling_seeded(42), temperature=0.9)
```

This works *only* because the helper uses a fresh session per request. Reusing
one changes the transcript and with it the output — the same design choice that
keeps unrelated calls from sharing context.

**Vision.** Attach images by path; a missing file is an error, never a silent
drop (an answer that quietly ignored your image is worse than a failure).
Entries may carry a label so follow-up turns can refer to them:

```ts
await llm.text('What is in this image?', { images: ['./photo.png'] });
await llm.text('What is in the image labelled chart?', {
  images: [{ path: './scan.png', label: 'chart' }],
});
```
```bash
apple-llm run --image ./photo.png "What shape and colours are in this image?"
apple-llm run --image ./scan.png::chart "What is in the image labelled chart?"
```

**Streaming.** Deltas arrive as the model generates; the promise still resolves
with the full text. Text only — the helper rejects schema+stream, because a
partial JSON document is not a usable delta:

```ts
await llm.stream('Count to three.', { onDelta: (d) => process.stdout.write(d) });
```
```bash
apple-llm run --stream "Count to three."
```

**Conversations.** A `sessionId` keeps one native transcript across calls —
one thread in the Siri-app sense. History is mirrored to
`~/Library/Caches/apple-llm/sessions/` so it survives helper restarts; the
native transcript does not, so after a restart the recent turns are reprised
as prompt context (a context break, not a loss):

```ts
const chat = llm.conversation('trip-planning', { system: 'You are terse.' });
await chat.text('My cat is called Biscuit.');
await chat.text('What is my cat called?');   // Biscuit.
await chat.history();                        // mirrored turns, oldest first
await chat.reset();                          // drop this thread
```
```bash
apple-llm run --session trip "My cat is Biscuit."
apple-llm run --session trip "What is my cat called?"
apple-llm history --session trip
apple-llm reset --session trip
```

**Built-in tools.** Apple's on-device OCR, barcode reading (Vision) and the
Spotlight semantic index (local RAG) — all macOS 27+, nothing leaves the
machine. Unknown names fail fast, before spending a model call:

```ts
await llm.text('Read the total on this receipt.', {
  images: ['./receipt.png'],
  tools: ['ocr'],
});
await llm.text('What meeting notes mention the bridge repair?', {
  tools: ['spotlight'],
});
```
```bash
apple-llm run --tool ocr --image receipt.png "Read the total."
```

First tool use pays a setup cost (tens of seconds observed for OCR on first
call); later calls are normal speed.

**Documents.** Text files are inlined client-side with a delimiter, so multi-
document prompts work without teaching Swift about file types. Missing or
binary files are errors, never silent drops:

```ts
await llm.text('Compare these quotes and help me pick one.', {
  documents: ['./quote-a.md', './quote-b.md'],
});
```
```bash
apple-llm run --document quote-a.md --document quote-b.md "Compare these and help me pick one."
```

**Write with Siri.** Drafting, rewriting and feedback presets built on the
permissive guardrails, for use anywhere you type:

```ts
await llm.rewrite('gonna grab a bite', { instruction: 'Make it formal.' });
await llm.proofread('Their going to the store...');
await llm.summarize(longText);
await llm.draft('A launch announcement for a bike light.');
await llm.tone(draft, 'professional');
```
```bash
apple-llm rewrite "gonna grab a bite"
apple-llm proofread - < draft.txt
apple-llm ask-screen "Summarize what is on this schedule."
```

**A model specialised for tagging.** `useCase: 'contentTagging'` selects the
variant Apple ships for tagging and topic extraction:

```bash
apple-llm run --use-case contentTagging --schema tags.json "A recipe for sourdough…"
# {"tags": ["sourdough", "rye starter", "cast iron", "dutch oven", "recipe"]}
```

**Permissive guardrails** for rewriting and summarising, where the default
guardrails refuse to touch the text: `guardrails: 'permissive'`.

**Prewarming.** `await llm.prewarm()` loads the model assets up front. Measured
honestly: on a machine where the assets are already resident it is worth almost
nothing (0.31s against 0.36s, inside the noise). The win is a genuinely cold
system, where the first framework call here took 7.8s.

**Real cloud quota, without a cloud call.** The entitlement that blocks PCC
*inference* does not block reading `PrivateCloudComputeLanguageModel.quotaUsage`,
so the on-device helper reads it and `probe()` reports it. An exhausted quota
becomes an immediate `QuotaError` instead of a wasted Shortcuts round trip.

## API

Both bindings are shaped the same.

**Node** — TypeScript, ESM + CJS, async. The full 0.2 API (tools, Zod,
streams, message lists, the server and the AI SDK provider) is in
[packages/node/README.md](packages/node/README.md); the core is shared with Python:

```ts
import { AppleLLM, probe } from 'apple-llm';

await probe();
// { device: { available, contextSize, variant }, cloud: { available, installed } }

const llm = new AppleLLM({ tier: 'auto', onProgress: (p) => console.log(p.status) });
await llm.text('...', { system, temperature, maxTokens, images, sampling });
await llm.stream('...', { onDelta: (d) => process.stdout.write(d) });
await llm.json('...', { system, schema });
await llm.countTokens('...', { system });   // { tokens, contextSize }
await llm.history('trip-planning');         // { instructions, history }
await llm.resetSession('trip-planning');
llm.conversation('trip-planning');          // threads sharing one transcript
await llm.rewrite('...');                   // + proofread/summarize/draft/tone
await llm.askScreen('What is on this schedule?');
await llm.prewarm();
llm.close();
```

**Python** — 3.10+, sync and async, pure-Python wheel.

```python
from apple_llm import AppleLLM, AsyncAppleLLM, probe

probe()

with AppleLLM(tier="cloud", fallback="device") as llm:   # on device when the cloud says no
    llm.text("...")

with AppleLLM(tier="auto", use_case="contentTagging") as llm:
    llm.text("...", system=..., temperature=..., images=[...], sampling=...)
    llm.stream("...", on_delta=print)
    llm.json("...", schema={...})          # a plain dict
    llm.json("...", schema=MyModel)        # or a pydantic model, validated back
    llm.count_tokens("...", system=...)    # {"tokens": ..., "context_size": ...}
    llm.history("trip-planning")           # {"instructions": ..., "history": [...]}
    llm.reset_session("trip-planning")
    llm.conversation("trip-planning")      # threads sharing one transcript
    llm.rewrite("...")                     # + proofread/summarize/draft/tone
    llm.ask_screen("What is on this schedule?")
    llm.prewarm()

async with AsyncAppleLLM(tier="device") as llm:
    await llm.text("...")
```

Pydantic is an optional extra (`pip install 'apple-llm[pydantic]'`); plain dict
schemas need no extras.

**Speech**, in both, is plain functions over a shared helper, or an `AppleSpeech`
instance with a helper of its own ([details](#speech)):

```ts
import { installSpeechLanguage, listSpeechLanguages, listVoices, listen, probeSpeech, removeSpeechLanguage, speak,
  synthesize, systemVoice, toSRT, toVTT, transcribe, transcribeStream, AppleSpeech } from 'apple-llm';
```

```python
from apple_llm import (install_speech_language, list_speech_languages, list_voices, listen, listen_once, probe_speech,
    remove_speech_language, speak, synthesize, system_voice, to_srt, to_vtt, transcribe, AppleSpeech, AsyncAppleSpeech)
```

### Errors

Typed, so you branch on a class rather than matching a message:
`ModelUnavailableError` (with `reason` distinguishing
`appleIntelligenceNotEnabled` / `modelNotReady` / `deviceNotEligible`),
`SchemaRejectedError`, `ContextLengthError`, `QuotaError` (Apple's cloud rate
limit or quota, with `resetDate` where Apple provides one), `TimeoutError`,
`SetupRequiredError`, `RefusalError`, `UnsupportedError` (an unknown voice, a
language macOS cannot recognise, an MP3 output file), `PermissionDeniedError` (the
microphone).

### CLI

Both packages install the same one:

```bash
apple-llm "What is the capital of Australia?"             # short for run
git diff | apple-llm "Write a commit message for this"    # piped text goes before the prompt
apple-llm probe
apple-llm setup-cloud [--web-search]
apple-llm chat                                            # interactive (npm)
apple-llm serve [--port 11434]                            # OpenAI- and Ollama-compatible API (npm)
apple-llm run --tier cloud --fallback device "…"          # on device when the cloud says no
apple-llm run --tier device --system "You are terse." -   # prompt on stdin
apple-llm run --tier cloud --schema schema.json "Extract the fields"
apple-llm run --image photo.png "What is in this image?"
apple-llm run --image scan.png::chart "What is in the image labelled chart?"
apple-llm run --tool ocr --image receipt.png "Read the total."
apple-llm run --document a.md --document b.md "Compare these and help me pick one."
apple-llm run --session trip "My cat is Biscuit."
apple-llm run --stream --session trip "What is my cat called?"
apple-llm history --session trip
apple-llm reset --session trip
apple-llm ask-screen "What is on this schedule?"
apple-llm rewrite "gonna grab a bite"
apple-llm proofread - < draft.txt
apple-llm run --seed 42 "Reproducible output"
apple-llm count --system "You are terse." -               # tokens before sending
apple-llm speak "Hello there" --voice Samantha            # out loud; -o file.m4a saves it
apple-llm voices [--language fr] [--all]
apple-llm transcribe meeting.mp4 [-o meeting.srt]         # audio or video to text
apple-llm listen [--once]                                 # the microphone, live
apple-llm languages [install|remove <code>]
```

## What this package really is

It is fifteen-odd discovered constraints, written down as working code so nobody
has to find them again. The two that cost the most:

**Do not set `temperature: 0`.** Constrained decoding already guarantees the
schema, so greedy decoding buys nothing and reliably degenerates. At 0 the model
padded an unbounded array forever, then ran away *inside a single string* — 2.7KB
of `"tasks-tasks-tasks-…"`. A 2s call became 20s. The default here is `0.4`.
Apple honours `maxItems` but ignores `maxLength`, so bounding your strings is not
an available fix. Bound your arrays.

**One long-lived helper process, not one per call.** Spawning a helper per
request measured ~17s a call against ~1.5s once the model is resident. Both of
these traps produce *correct* output, only slower, so neither looks like a bug —
which is why the live test suite fails if the median call exceeds 4s.

Apple serialises inference regardless: four concurrent requests measured 29.19s
against 29.45s sequentially. The request queue serialises deliberately, and
concurrency above 1 buys nothing.

### What speech cost to get right

Each of these looked fine until it didn't, and most would have shipped silently:

- **A zero-length buffer does not mean the end of the speech.**
  `AVSpeechSynthesizer.write` emits them between segments of the text, so stopping
  at the first one truncated a passage to its first two sentences, which looked
  exactly like Apple limiting audio to a few seconds. Only `didFinish` means done.
- **Word offsets restart with each segment.** Taken at face value, the last word
  of a 30 minute render claimed to be at 11.9 seconds.
- **`say -v` never says no.** Any name, real or not, exits 0 and uses the default
  voice, so it took comparing the audio to learn that none of the Siri names work.
- **A missing language fails as "Audio format is not supported",** and
  `AssetInventory.status` reports `.supported` for models that are installed and
  working. So the check is `installedLocales`, before any audio is touched.
- **A new synthesizer costs 300ms before its first buffer; a reused one, 10ms.**
  But reusing one with a delegate per request crashed the helper with SIGSEGV in
  `TextToSpeech`, because a queued callback outlived its delegate. One delegate now
  lives as long as the process.
- **The only real limit on long audio was ours.** The wedge timer measures silence,
  and a read-aloud is silent on the wire, so the helper sends a heartbeat every 20
  seconds.
- **macOS can cancel speech on its own.** If the helper's file is deleted while it
  runs, as a disk cleaner might do to a cache, `AVSpeechSynthesizer` fires
  `didCancel` within 0.1s and says nothing. Treated as finished, that was `ok: true`
  with no sound. Counting words cannot catch it, because some voices send none.
- **`say` exits 0 with words missing.** With the language model busy elsewhere, a
  quarter of renders in Aman and Tara lost words, and a few were silent or 0.02s
  long. Checking for errors or silence would never have shown it; transcribing the
  files did. `AVSpeechSynthesizer` rendered 4,130 identical files under the same
  load. So `say` renders run twice and must agree.

### The `GenerationSchema` dialect

`json()` on device is a guarantee, not a request — but only because
`toAppleSchema()` rewrites your JSON Schema into the restricted dialect Apple's
decoder accepts. Nine rules, each found by a rejected schema:

1. **No union types.** `type: ["string","null"]` fails. Nullability is expressed
   by leaving the key out of `required`.
2. **Every object needs `x-order`**, naming its properties in generation order.
3. **No bare `enum`.** A node may only carry `type`, `const`, `$ref` or `anyOf`,
   so enums become `anyOf` of `const` branches.
4. **Every object and every `anyOf` needs a `title`**, and `$ref` resolves *by
   title*, not by JSON pointer — `#/$defs/param` only resolves to a schema
   titled exactly `param`.
5. **Every object must state `additionalProperties`** (macOS 27; macOS 26
   tolerated its absence).
6. **`const` is always decoded as a String.** A numeric enum cannot be expressed
   as const branches, so it keeps its type and loses the literal constraint —
   stringifying it would silently change the field's type.
7. **An object with no properties still needs `properties: {}`**, or
   `additionalProperties` is read as a value schema instead.
8. **A `title` on a `string` is rejected** as a "named string type" needing an
   enum. Titles are stripped from plain strings. This matters most for pydantic,
   which titles every property.
9. **Every `pattern` is rejected** at generation time with `UnsupportedGuide` —
   even `^[A-Z]{3}$`. Zod puts one on `.email()`, `.uuid()` and `.datetime()`, so
   this broke ordinary schemas. Patterns are stripped for the decoder and checked
   on the reply instead.

Rules 1–5 were found by decoding real schemas against macOS 26 and 27 until
they were accepted; rules 6–9 by decoding this repo's fixture corpus against
macOS 27. The live test suite still checks every
fixture against Apple's own decoder, because Apple is the authority on its
dialect, not this implementation. macOS 27.2 relaxed rules 2, 4 and 5 for
objects (an untitled object without `x-order` now decodes); they are still
applied, since earlier releases require them.

A `{type: "null"}` branch in an `anyOf` — how Zod and pydantic spell a nullable
object — follows rule 1: it becomes "not required", a lone remaining branch is
unwrapped, and the missing key is filled back in with `null` on the reply.

**Why the dialect matters more than it looks:** constrained decoding makes a
schema mistake invisible but total. Collapsing a `["number","string"]` union to
`"string"` made it physically impossible for the model to emit a status code — it
wrote `": 201"` and the literal `"default"` instead. A schema the model cannot
satisfy is not a warning; it is a wrong answer.

### Why the cloud tier goes through Shortcuts

`FoundationModels.PrivateCloudComputeLanguageModel` is public API on macOS 27 and
reports `isAvailable: true`, but every call fails with `ModelManagerError 1046`
unless the process carries `com.apple.developer.private-cloud-compute`. That
entitlement is AMFI-restricted: an ad-hoc-signed binary carrying it is SIGKILLed,
and wrapping it in a signed `.app` with a real bundle ID does not help. It needs
a paid Developer Program provisioning profile, which no installable package can
ship.

Shortcuts.app already holds the entitlement and `/usr/bin/shortcuts` is public,
so `setup-cloud` generates the shortcut and signs it **on your machine** —
`shortcuts sign -m anyone` issues a per-signature certificate chaining to Apple
Root CA - G3 with zero signing identities. Nothing is pre-signed, hosted, or
downloaded.

Two notes for anyone reading that code: the prompt key is `WFLLMPrompt`, *not*
the `WFInput` the action's own strings suggest — a wrong key imports fine, is
silently discarded, and then blocks forever on an interactive panel. And
duplicate shortcut names break `shortcuts run` outright, so setup refuses to
proceed when it finds them.

`/usr/bin/fm` on macOS 27 is a different thing and is deliberately not used: it
is gated behind a machine-wide `sudo fm license`.

### macOS 27 Golden Gate: reaching Siri AI's model

Siri AI in Golden Gate runs on what Apple calls the next generation of Apple
Foundation Models on Private Cloud Compute (press reports say it is built with
Google's Gemini; Apple's APIs do not say, and the model describes itself as
Apple's). Every door to it was tried on macOS 27.2 (26B5091g). What was found:

- **The framework describes it, and still will not run it.**
  `PrivateCloudComputeLanguageModel` reports reasoning, vision, tool calling and
  guided generation with a 32,768-token window — readable without the
  entitlement, so `probe()` now reports them — but `respond` still fails without
  `com.apple.developer.private-cloud-compute`. Apple offers it to apps through
  the App Store Small Business Program, which an installable package cannot use.
- **Shortcuts reaches it, and that is what the cloud tier uses.** Golden Gate
  replaced the Use Model action's choices: Apple's own gallery shortcuts set
  `WFLLMModel` to `"Apple Intelligence"`; the old `"Private Cloud Compute"` and
  `"On-Device"` values now fail with "An error occurred while loading the
  model". Leaving the key out — as this package always has — gives byte-identical
  output to `"Apple Intelligence"`, so existing installs already use the new
  model. Measured through it: a logic puzzle the on-device model lost track of,
  solved; a fact buried in 25,000 tokens, found; an image, described; and with
  web search, the 2026 World Cup final result, correct.
- **Images work through Shortcuts** as extra `-i` inputs beside the prompt
  file — with the existing shortcut, no reinstall. The model reads the prompt
  and sees the image in either order.
- **The Siri AI app offers nothing to automate.** `Siri AI.app`
  (`com.apple.campo`) ships an App Intents manifest with no actions.
- **Gemini itself** comes to the framework through Google's Firebase SDK, with
  your own Gemini API key — a different, paid-or-free-tier Google service, not
  Apple's model and not free through Apple.
- **Not reachable:** the Use Model action's reasoning effort (`thinkingEffort`
  exists internally, with no user-facing parameter), and constrained decoding
  on the server model (Shortcuts has no schema input), so cloud JSON stays
  "requested, validated, repaired".

## Layout

```
swift/helper.swift          the single source of truth; both packages ship a copy
scripts/embed-helper.mjs    regenerates both copies and verifies the hashes
packages/node/              npm: apple-llm (examples/: one runnable file per README section)
packages/python/            pip: apple-llm
tests/fixtures/schema/      golden corpus, read by BOTH test suites
tests/fixtures/images/      image fixture, likewise shared
```

The Swift source is copied rather than embedded as a string literal, so the two
packages cannot drift; a test in each asserts its copy hashes equal to
`swift/helper.swift`, and both derive the same cache key so they share one
compiled binary.

## Testing

```bash
# packages/node
npm test                  # everything that runs without Apple hardware
APPLE_LLM_LIVE=1 npm test # adds the on-device tests, the AI SDK provider
                          # through `ai`, and the server through `openai`
npm run check:package     # build, then publint and are-the-types-wrong

# packages/python
pytest
APPLE_LLM_LIVE=1 pytest
```

Cloud tests consume quota and need `APPLE_LLM_LIVE_CLOUD=1` as well. Everything
live is gated on an env var *and* a runtime probe, so a machine without Apple
Intelligence skips cleanly instead of failing. The live speech tests are silent:
read-alouds run at volume 0, and none opens the microphone.

## Contributing

Issues and PRs are welcome — reports from macOS versions and Macs this has not
been tested on most of all. If apple-llm saved you time, a ⭐ helps other Mac
developers find it.

## License

MIT. Not affiliated with or endorsed by Apple Inc.
