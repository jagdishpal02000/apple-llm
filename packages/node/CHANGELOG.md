# Changelog

## 0.3.0

### Fixed

- **The cloud tier "not working": `Apple cloud generation failed: Error: Your
  request can't be completed right now.`** That sentence is all `shortcuts run`
  says when Private Cloud Compute's per-device rate limit turns a request away
  (`deniedDueToUserDeviceRateLimit` underneath). It is a different limiter from
  the quota `probe()` reads, which keeps saying `belowLimit` the whole time, so
  nothing warned of it and the error read like a bug. Ten calls in about two
  minutes tripped it, and it then refused every call for a few minutes. It
  is now a `QuotaError` that says what happened and what to do; the server
  answers it with HTTP 429 and `x-should-retry: false`, so OpenAI SDKs fail fast
  instead of retrying seconds later into a lockout that lasts much longer.
- **`json()` on the cloud tier answering in prose.** With the JSON request placed
  before the question, the server model answered "Give me a city in France"
  with `Paris, population 2161000` — the right fields, not JSON — and the repair
  attempt failed the same way. The request now follows the question, and the
  repair shows the model the answer it is repairing: every call is a fresh
  session, so "your previous answer" meant nothing to it.
- The server's `model` field names the model that actually answered; with
  `--tier auto` it said `apple-on-device` even when the cloud had answered.

### Added

- **Speech: text to speech and speech to text, on device.** `speak()`,
  `synthesize()`, `listVoices()`, `transcribe()`, `transcribeStream()`, `listen()`
  and the `apple-llm speak`, `voices`, `transcribe`, `listen` and `languages`
  commands. Neither needs Apple Intelligence or costs quota. Text to speech reaches
  every voice on the Mac, out loud or to WAV, M4A, AAC, FLAC, AIFF, CAF or raw PCM,
  with speed, pitch, volume, SSML and word timings. Speech to text reads audio and
  MP4 or MOV video of any length, or the microphone live, in 68 languages, with word
  timings and SRT or VTT subtitles. On an M4 Air, 60 minutes of speech renders in 40
  seconds and 30 minutes transcribes in 28. The helper gains all this, so the first
  call after upgrading compiles it once more.
- **`POST /v1/audio/speech`, `POST /v1/audio/transcriptions` and
  `GET /v1/audio/voices` on `apple-llm serve`**, checked with the official `openai`
  SDK. macOS cannot encode MP3 or Opus, so requests for those get AAC, with a
  `Content-Type` that says so.
- **`PermissionDeniedError`**, for a refused microphone.
- **`fallback: 'device'`** (`--fallback device` on `run`, `chat` and `serve`):
  with the cloud tier, answer on device when Apple turns a request away. The
  result's `tier` says which model answered.
- **Ollama-compatible API on `apple-llm serve`**, beside the OpenAI one:
  `/api/chat` and `/api/generate` (NDJSON streaming, `format` as `"json"` or a
  JSON Schema, tools, images, `options`), `/api/tags`, `/api/show`, `/api/ps`
  and `/api/version`. Most Mac apps for local models speak Ollama's API, so
  `apple-llm serve --port 11434` makes Apple's model available in them with no
  code. Checked against the Ollama CLI itself: `ollama list`, `show` and `run`,
  with `--format json`.
- **A shell-friendly CLI.** `apple-llm "question"` is short for `apple-llm run
  "question"`, and text piped into `run` goes in before the prompt:
  `git diff | apple-llm "Write a commit message for this"`. Only a pipe or a
  redirected file is read, so a script that leaves stdin open does not hang it.
- A `ContextLengthError` on device now points at the cloud tier's 32k window,
  and `serve` explains a port that is already taken (on 11434, usually Ollama).

### Python

- **Speech**, as in Node: `speak()`, `synthesize()`, `list_voices()`,
  `transcribe()`, `listen()`, `listen_once()`, `AppleSpeech`, `AsyncAppleSpeech` and
  the same commands. Adds `UnsupportedError`, `PermissionDeniedError` and
  `CancelledError`.
- The same rate-limit and cloud-JSON fixes, `fallback="device"`, and the CLI
  shorthand and piping.
- **Fixed `apple-llm run --stream` crashing** with `TypeError: AppleLLM.stream()
  got an unexpected keyword argument 'web_search'`, on every tier.
- `stream()` and `documents` now work on the cloud tier, as in Node (the reply
  arrives as one delta). A cloud `json()` reply with no JSON in it raises
  `SchemaValidationError` after one repair — still a `ValueError`, which is what
  it raised before.

## 0.2.0

The helper speaks a new protocol, so the first call after upgrading compiles it
once more (a few seconds). Every 0.1 call still works; the changes in behaviour
below are the ones worth reading before you upgrade.

### Added

- **Function tools.** `tool({ name, description, parameters, execute })` — the
  model calls your JavaScript mid-generation, with arguments guaranteed to match
  the parameter schema. Tools without `execute` stop generation with
  `finishReason: 'tool-calls'` so you can run them yourself and continue with a
  `tool` message. Mixes with the built-in `'ocr'`, `'barcode'`, `'spotlight'`.
  `maxToolCalls` (default 8) stops a model that loops on a tool.
- **Standard Schema support.** Pass a Zod 4, ArkType or Valibot
  (`toStandardJsonSchema`) schema anywhere a JSON Schema goes. `json()` is typed
  by it and validated with it; a reply that fails validation is retried once
  with the problems pointed out, then raised as `SchemaValidationError`.
- **Message lists.** Every method takes `ChatMessage[]` as well as a string.
  History is sent as Apple's native transcript, and trimmed from the oldest end
  if it outgrows the window (`trimHistory`, `result.trimmedTurns`).
- **`generate()`** returns everything about a call: `text`, `object`,
  `finishReason` (`'stop' | 'length' | 'tool-calls'`), `usage` (input, output,
  cached tokens — macOS 27), `toolCalls` with their outputs, `durationMs`, and
  the assistant `message` to append for the next turn.
- **`for await` streaming.** `stream()` returns a stream you can iterate or
  await; breaking out cancels generation. `streamJson()` streams partial objects
  as the model fills them in.
- **Cancellation.** `signal` on every call stops the model itself, not just the
  promise. `timeoutMs` per call or per client.
- **OpenAI-compatible server**: `apple-llm serve`, or `serve()` from
  `apple-llm/server`. Chat completions with streaming, tools, `response_format`
  JSON Schema and images. Localhost-only, no CORS and on-device by default.
- **Vercel AI SDK provider**: `apple()` from `apple-llm/ai-sdk`, for AI SDK 6
  and 7 — `generateText`, `streamText`, `generateObject`, multi-step tools,
  images.
- **`apple-llm chat`**, an interactive terminal chat. `run --json` prints the
  full result; `run --schema --stream` prints partial objects, one per line;
  `run --timeout`.
- **Long text.** `summarize()` handles text longer than the context window by
  summarising it in parts.
- **Named sessions recover from overflow.** A conversation that outgrows the
  window is rebuilt from its newest turns instead of failing forever, and after
  a helper restart it resumes as a native transcript rather than a text
  preamble.
- `countTokens()` now counts message history, schema and tools too.
- `using` / `await using` dispose a client. Error classes `AbortError`,
  `ToolExecutionError` (your error is its `cause`), `SchemaValidationError`,
  `ModelBusyError` and `UnsupportedError`, and a stable `code` on every error.
- Images on `user` messages, and images as bytes, data URLs or http URLs
  through the server and the AI SDK provider.
- **The cloud tier on macOS 27 Golden Gate** reaches the next-generation server
  model Siri AI is built on — the Shortcuts route already selected it, and this
  release makes it visible and uses more of it: `probe().cloud.capabilities` and
  `contextSize` come from the framework (reasoning, vision, tool calling, 32k),
  and **the cloud tier now reads images**, passed as extra Shortcuts inputs with
  the existing shortcut — no reinstall. Images are refused, not dropped, where
  the server model's vision cannot be confirmed (macOS 26).

### Changed

- **A script no longer hangs without `close()`.** The helper is unref'd between
  calls, so Node exits when your work is done. `close()` still releases it early.
- **`json()` fills in `null` for required nullable fields** the model left out.
  Apple's dialect cannot express `null`, so such fields become optional on the
  way in; your schema said they are present, so they are put back on the way
  out. Code that checked `'key' in result` for these fields will now see them.
- **Schemas lose `pattern`** before reaching Apple's decoder, which rejects
  every regex (`UnsupportedGuide`) — this is what made `z.string().email()`
  fail. The pattern is checked on the reply instead.
- **The cloud tier streams** as a single chunk instead of throwing, accepts
  `documents` and message lists, and now refuses `sessionId` and `tools` on
  `json()` as it already did on `text()`, instead of ignoring them.
- `stream()` returns a `ResultStream` rather than a `Promise<string>`. Awaiting
  it still gives the text.
- Helper errors of kind `unsupported` are `UnsupportedError`, a subclass of
  `AppleLLMError`, so existing `instanceof AppleLLMError` checks still match.
- Concurrent first calls share one probe-and-compile instead of racing.

### Fixed

- Types resolve for CommonJS consumers (`require`) under `node16`/`nodenext`
  module resolution, and the package passes publint and are-the-types-wrong.
- A single-entry `allOf` holding an inline object (rather than a `$ref`) was
  sent to Apple as an object with no properties; it now keeps them. The same
  fix covers nullable object unions, which are new in this release.
- A `Conversation`'s default `system` is no longer wiped by a call that passes
  `system: undefined`.
- The helper's stderr is drained, so a chatty helper can no longer block on a
  full pipe, and its last lines are quoted when it exits unexpectedly.
- A request turned away because other processes are using the model
  (`ModelManagerError 1042`) is retried with backoff instead of failing.
