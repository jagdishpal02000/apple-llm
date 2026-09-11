# apple-llm

Apple's on-device and Private Cloud Compute LLMs, from Node. No API key, no
account, no developer program membership.

```bash
npm install apple-llm
```

```ts
import { AppleLLM, probe } from 'apple-llm';

await probe();
// { device: { available, contextSize, variant }, cloud: { available, installed } }

const llm = new AppleLLM({ tier: 'device' });          // 'device' | 'cloud' | 'auto'
await llm.text('Summarize this', { system: 'You are terse.' });
await llm.json('Extract the fields', { schema });       // guaranteed to match, on device
llm.close();
```

ESM + CJS, fully typed, **zero runtime dependencies**. A small Swift helper is
compiled on first use and cached; there is no postinstall script, so installing
on Linux or an Intel Mac always succeeds.

## Read this before you use it

- **macOS 26+ on Apple Silicon.** No fallback anywhere else. Import always
  works; `probe()` returns `available: false` with an actionable reason.
- **The on-device model is small** (~20B sparse, 1–4B active, 8192-token context
  on macOS 27). Good at classification, extraction, tagging, rewriting and short
  prose. **Bad at code generation and long reasoning.**
- **The cloud tier is not local.** It sends your prompt to Apple's Private Cloud
  Compute, off your machine. Free but quota'd, reached through a private
  Shortcuts action Apple can change in any OS release, and with no constrained
  decoding — so `json()` there is a request, not a guarantee.
- **Streaming is text-only, tools are Apple's built-ins.** `stream()` gives
  partials-as-they-arrive; `tools: ['ocr', 'barcode', 'spotlight']` enables
  on-device Vision/Spotlight tools. Generic user-supplied function calling is
  still a later version.

## Two traps worth knowing

**Never set `temperature: 0`.** Constrained decoding already guarantees the
schema, so greedy decoding buys nothing and reliably degenerates — it padded an
unbounded array forever, then ran away inside a single string (2.7KB of
`"tasks-tasks-tasks-…"`), turning a 2s call into 20s. The default is `0.4`.
Apple honours `maxItems` but ignores `maxLength`: bound your arrays.

**Keep the client alive.** One long-lived helper process holds the model
resident; spawning one per call measured ~17s against ~1.5s. Construct
`AppleLLM` once, `close()` when you are done. Both traps produce *correct*
output, only slower, so neither looks like a bug.

Apple serialises inference regardless — 4 concurrent requests measured 29.19s
against 29.45s sequentially — so the request queue serialises deliberately.

## What macOS 27 adds

```ts
const { device, cloud } = await probe();
device.capabilities;   // { vision, guidedGeneration, reasoning, toolCalling }
device.useCases;       // ['general', 'contentTagging']
cloud.quota;           // { status: 'belowLimit' | 'limitReached', approachingLimit, resetDate }
```

On this machine the on-device model reports vision, guided generation and tool
calling, but **not** reasoning. `cloud.quota` is real quota state read from
`PrivateCloudComputeLanguageModel.quotaUsage` — the entitlement that blocks PCC
*inference* does not block reading it — so an exhausted quota becomes an
immediate `QuotaError` rather than a wasted Shortcuts round trip.

**Count tokens before sending**, turning a `ContextLengthError` into arithmetic:

```ts
const { tokens, contextSize } = await llm.countTokens(prompt, { system });
```

**Reproducible output without the `temperature: 0` trap.** Greedy decoding is
deterministic *and* degenerates; seeded top-k is deterministic and does not:

```ts
await llm.text(prompt, { sampling: { mode: 'topK', k: 50, seed: 42 }, temperature: 0.9 });
```

It works only because the helper uses a fresh session per request — reusing one
changes the transcript and with it the output.

**Vision**, by path — optionally labelled for follow-up turns. A missing file
is an error, never a silent drop:

```ts
await llm.text('What is in this image?', { images: ['./photo.png'] });
await llm.text('What is in the image labelled chart?', {
  images: [{ path: './scan.png', label: 'chart' }],
});
```

**Streaming, conversations, tools, documents, and Write-with-Siri presets:**

```ts
await llm.stream('Count to three.', { onDelta: (d) => process.stdout.write(d) });

const chat = llm.conversation('trip-planning');
await chat.text('My cat is called Biscuit.');
await chat.text('What is my cat called?');  // Biscuit.
await chat.history();                       // mirrored turns, oldest first

await llm.text('Read the total.', { images: ['./receipt.png'], tools: ['ocr'] });
await llm.text('Which notes mention the bridge?', { tools: ['spotlight'] });
await llm.text('Compare these quotes.', { documents: ['./a.md', './b.md'] });

await llm.rewrite('gonna grab a bite', { instruction: 'Make it formal.' });
await llm.proofread('Their going to the store...');
await llm.summarize(longText);
await llm.askScreen('What is on this schedule?');
```

**A tagging-specialised model**, `permissive` guardrails for rewriting, and
`prewarm()` to load model assets up front (worth little once they are resident —
0.31s against 0.36s here — but real on a cold system):

```ts
const llm = new AppleLLM({ tier: 'device', useCase: 'contentTagging' });
await new AppleLLM({ guardrails: 'permissive' }).text('Rewrite more formally: …');
await llm.prewarm();
```

## Errors

```ts
import { QuotaError, ModelUnavailableError } from 'apple-llm';
```

`ModelUnavailableError` (`.reason` is `appleIntelligenceNotEnabled` /
`modelNotReady` / `deviceNotEligible` / …), `SchemaRejectedError`,
`ContextLengthError`, `QuotaError` (`.resetDate`), `TimeoutError`,
`SetupRequiredError`, `RefusalError`.

## CLI

```bash
apple-llm probe
apple-llm setup-cloud [--web-search]     # one-time, generates + signs locally
apple-llm run --tier device --system "You are terse." -
apple-llm run --tier cloud --schema schema.json "Extract the fields"
apple-llm run --image photo.png "What is in this image?"
apple-llm run --image scan.png::chart "What is in the image labelled chart?"
apple-llm run --tool ocr --image receipt.png "Read the total."
apple-llm run --document a.md --document b.md "Compare these."
apple-llm run --session trip --stream "What is my cat called?"
apple-llm history --session trip
apple-llm reset --session trip
apple-llm ask-screen "What is on this schedule?"
apple-llm rewrite "gonna grab a bite"
apple-llm run --use-case contentTagging --schema tags.json "A recipe for sourdough…"
apple-llm run --seed 42 "Reproducible output"
apple-llm count --system "You are terse." -    # tokens before sending
```

## Schemas

`json()` runs your JSON Schema through `toAppleSchema()`, which rewrites it into
the restricted dialect Apple's `GenerationSchema` decoder accepts — eight rules
covering unions, `x-order`, enums, titles, `$ref`-by-title, `additionalProperties`,
string-typed `const`, and empty objects. See the
[full notes in the repo](https://github.com/jagdish/apple-llm#the-generationschema-dialect).

The details matter more than they look: constrained decoding makes a schema
mistake invisible but total. A union collapsed to the wrong branch does not warn
— it makes the right answer unreachable.

## Credit

Extracted from [api-scribe](https://github.com/jagdish/api-scribe) (MIT), where
both routes were discovered and shipped. MIT.
