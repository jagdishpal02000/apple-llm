# Changelog

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
