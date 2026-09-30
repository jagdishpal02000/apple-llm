# apple-llm

Apple's built-in LLMs from Node — the on-device model and Private Cloud
Compute. No API key, no account, no model download, **zero runtime
dependencies**.

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
- **Three ways in** — this library, an [OpenAI-compatible server](#openai-compatible-server)
  for any tool that speaks OpenAI, and a [Vercel AI SDK provider](#vercel-ai-sdk).
- **Two tiers** — on-device (private, fast, free) and Apple's big server
  model on Private Cloud Compute — on macOS 27 Golden Gate, the one Siri AI is
  built on: reasoning, images, live web knowledge, 32k context. Free, no key,
  behind the same API.

## Read this first

- **macOS 26+ on Apple Silicon, with Apple Intelligence on.** Nothing works
  elsewhere, but importing always succeeds and `probe()` says exactly why.
- **The Xcode Command Line Tools** (`xcode-select --install`), for compiling
  the helper. Full Xcode is not needed.
- **The on-device model is small** (~20B sparse, 1–4B active; 8,192-token
  context). Good at extraction, classification, tagging, rewriting, summaries
  and short answers. Bad at code and long reasoning. Use it for the jobs it
  suits.
- **The cloud tier is not local.** `tier: 'cloud'` sends your prompt to Apple's
  Private Cloud Compute. The default, `auto`, uses the device when it can.

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

## OpenAI-compatible server

```bash
npx apple-llm serve
# apple-llm serving an OpenAI-compatible API at http://127.0.0.1:11436/v1
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
apple-llm chat                                  # interactive, streamed; Ctrl+C stops a reply
apple-llm run "Summarize: …"                    # one completion; "-" reads stdin
apple-llm run --json "…"                        # text + usage + tool calls as JSON
apple-llm run --schema person.json "Ada, 1815"  # guaranteed JSON
apple-llm run --schema person.json --stream "…" # partial objects, one per line
apple-llm run --image photo.png "What is this?"
apple-llm run --tool ocr --image receipt.png "Read the total."
apple-llm serve [--port 11436] [--api-key k]    # OpenAI-compatible API
apple-llm count --system "…" -                  # tokens before sending
apple-llm probe                                 # what this machine can do
apple-llm setup-cloud                           # one-time, for the cloud tier
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
| Limits | none | Apple's quota (read before calling: `probe().cloud.quota`) |

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
| `QuotaError` | rate limited (`.resetDate` when Apple gives one) |
| `RefusalError` | the model's guardrails declined |
| `ModelBusyError` | other processes are using the model; already retried with backoff |
| `SetupRequiredError`, `UnsupportedError` | a setup step is missing / the tier cannot do that |

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
