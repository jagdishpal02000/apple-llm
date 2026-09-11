# apple-llm

Apple's two built-in LLMs, from Node and Python. No API key, no account, no
developer program membership, nothing to install beyond the package.

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

Both packages compile one small Swift helper on first use and share the compiled
binary, so installing both costs one compile, not two.

## Read this before you use it

**macOS 26 or later, on Apple Silicon.** Nothing here works on Intel Macs,
Linux, or Windows. Importing the package still succeeds everywhere — `probe()`
tells you why the model is unavailable rather than failing at install time — but
there is no fallback. This is a wrapper around hardware you either have or
don't.

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

**v1 was non-streaming with no tool calling.** The model reports the
`toolCalling` capability and `session.streamResponse` exists, and both are now
wired up: `stream()` for partials-as-they-arrive, `tools: ['ocr', 'barcode',
'spotlight']` for Apple's built-in on-device tools, and `sessionId` for named
multi-turn conversations. Streaming is text-only (partial JSON is not a usable
delta); generic function calling — user-supplied tools executed client-side —
is still a later version.

## The two tiers

|  | on-device | cloud (Private Cloud Compute) |
|---|---|---|
| Model | AFM 3 Core Advanced (~20B sparse, 1–4B active) | the large "Siri" model |
| Context | 8192 tokens (4096 on macOS 26) | 32768 tokens |
| Typical latency | 0.5–2s warm | ~2s, ~11s at 14k tokens |
| JSON | **guaranteed** by constrained decoding | asked for in the prompt, recovered from prose |
| Privacy | nothing leaves the machine | **your prompt leaves the machine** |
| Setup | none | one-time `apple-llm setup-cloud` |
| Limits | none | quota'd |
| Vision | yes (macOS 27) | not exposed here |
| Reasoning | **no** (the model says so) | yes |
| Web search | no | optional `WFAllowWebSearch` |
| Streaming | yes, text-only (`stream()`) | no |
| Built-in tools | `ocr`, `barcode`, `spotlight` (macOS 27) | no |
| Conversations | named `sessionId` threads + `history` | no |

`tier: 'auto'` uses the device when it is available, else the cloud when the
shortcut is installed, else raises an error naming the setup step.

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

**Node** — TypeScript, ESM + CJS, async.

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

### Errors

Typed, so you branch on a class rather than matching a message:
`ModelUnavailableError` (with `reason` distinguishing
`appleIntelligenceNotEnabled` / `modelNotReady` / `deviceNotEligible`),
`SchemaRejectedError`, `ContextLengthError`, `QuotaError` (with `resetDate`
where Apple provides one), `TimeoutError`, `SetupRequiredError`, `RefusalError`.

### CLI

Both packages install the same one:

```bash
apple-llm probe
apple-llm setup-cloud [--web-search]
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

### The `GenerationSchema` dialect

`json()` on device is a guarantee, not a request — but only because
`toAppleSchema()` rewrites your JSON Schema into the restricted dialect Apple's
decoder accepts. Eight rules, each found by a rejected schema:

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

Rules 1–5 came from api-scribe. Rules 6–8 were found here, by decoding this
repo's fixture corpus against macOS 27 — the live test suite still checks every
fixture against Apple's own decoder, because Apple is the authority on its
dialect, not this implementation.

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

## Layout

```
swift/helper.swift          the single source of truth; both packages ship a copy
scripts/embed-helper.mjs    regenerates both copies and verifies the hashes
packages/node/              npm: apple-llm
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
APPLE_LLM_LIVE=1 npm test # adds the on-device tests

# packages/python
pytest
APPLE_LLM_LIVE=1 pytest
```

Cloud tests consume quota and need `APPLE_LLM_LIVE_CLOUD=1` as well. Everything
live is gated on an env var *and* a runtime probe, so a machine without Apple
Intelligence skips cleanly instead of failing.

## Credit

Both routes were discovered, debugged and shipped in
[api-scribe](https://github.com/jagdish/api-scribe) (MIT), which this package
extracts from. Every measurement quoted here and in the source comments came from
that work.

MIT.
