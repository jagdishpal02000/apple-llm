# apple-llm

Apple's on-device and Private Cloud Compute LLMs, from Python. No API key, no
account, no developer program membership.

```bash
pip install apple-llm
```

```python
from apple_llm import AppleLLM, probe

probe()
# {"device": {"available": ..., "contextSize": ..., "variant": ...},
#  "cloud":  {"available": ..., "installed": ...}}

with AppleLLM(tier="device") as llm:            # "device" | "cloud" | "auto"
    llm.text("Summarize this", system="You are terse.")
    llm.json("Extract the fields", schema={...})   # guaranteed to match, on device
```

Pure-Python wheel, **no runtime dependencies**, no build step. A small Swift
helper is compiled on first use and cached, so installing on Linux or an Intel
Mac always succeeds and `probe()` explains why the model is unavailable there.

## Pydantic

```python
from pydantic import BaseModel

class Person(BaseModel):
    name: str
    born: int

with AppleLLM(tier="device") as llm:
    person = llm.json("Ada Lovelace, born 1815.", schema=Person)

person.born      # 1815 — a validated Person, not a dict
```

Optional extra: `pip install 'apple-llm[pydantic]'`. Plain dict schemas need no
extras.

## Async

```python
from apple_llm import AsyncAppleLLM

async with AsyncAppleLLM(tier="device") as llm:
    await llm.text("Summarize this")
    await llm.stream("Count to three.")
```

Cleanup is registered with both `atexit` and `weakref.finalize`, so a wedged
helper never outlives the interpreter.

## Read this before you use it

- **macOS 26+ on Apple Silicon.** No fallback anywhere else.
- **The on-device model is small** (~20B sparse, 1–4B active, 8192-token context
  on macOS 27). Good at classification, extraction, tagging, rewriting and short
  prose. **Bad at code generation and long reasoning.**
- **The cloud tier is not local.** It sends your prompt to Apple's Private Cloud
  Compute, off your machine. Free but quota'd, reached through a private
  Shortcuts action Apple can change in any OS release, and with no constrained
  decoding — so `json()` there is a request, not a guarantee.
- **Streaming is text-only, tools are Apple's built-ins.** `stream()` gives
  partials-as-they-arrive; `tools=["ocr", "barcode", "spotlight"]` enables
  on-device Vision/Spotlight tools. Generic user-supplied function calling is
  still a later version.

## Two traps worth knowing

**Never set `temperature=0`.** Constrained decoding already guarantees the
schema, so greedy decoding buys nothing and reliably degenerates — it padded an
unbounded array forever, then ran away inside a single string (2.7KB of
`"tasks-tasks-tasks-…"`), turning a 2s call into 20s. The default is `0.4`.
Apple honours `maxItems` but ignores `maxLength`: bound your arrays.

**Keep the client alive.** One long-lived helper process holds the model
resident; spawning one per call measured ~17s against ~1.5s. Use the context
manager, or construct once and `close()` when done. Both traps produce *correct*
output, only slower, so neither looks like a bug.

## What macOS 27 adds

```python
result = probe()
result["device"]["capabilities"]  # {"vision", "guidedGeneration", "reasoning", "toolCalling"}
result["device"]["useCases"]      # ["general", "contentTagging"]
result["cloud"]["quota"]          # {"status": "belowLimit", "approachingLimit": False, ...}
```

On this machine the on-device model reports vision, guided generation and tool
calling, but **not** reasoning. The quota is real state read from
`PrivateCloudComputeLanguageModel.quotaUsage` — the entitlement that blocks PCC
*inference* does not block reading it — so an exhausted quota becomes an
immediate `QuotaError` rather than a wasted Shortcuts round trip.

**Count tokens before sending**, turning a `ContextLengthError` into arithmetic:

```python
llm.count_tokens(prompt, system=...)   # {"tokens": 64, "context_size": 8192}
```

**Reproducible output without the `temperature=0` trap.** Greedy decoding is
deterministic *and* degenerates; seeded top-k is deterministic and does not:

```python
from apple_llm import sampling_seeded

llm.text(prompt, sampling=sampling_seeded(42), temperature=0.9)
```

It works only because the helper uses a fresh session per request — reusing one
changes the transcript and with it the output.

**Vision**, by path — optionally labelled for follow-up turns. A missing file
is an error, never a silent drop:

```python
llm.text("What is in this image?", images=["./photo.png"])
llm.text("What is in the image labelled chart?",
         images=[{"path": "./scan.png", "label": "chart"}])
```

**Streaming, conversations, tools, documents, and Write-with-Siri presets:**

```python
llm.stream("Count to three.", on_delta=print)

chat = llm.conversation("trip-planning")
chat.text("My cat is called Biscuit.")
chat.text("What is my cat called?")  # Biscuit.
chat.history()                       # mirrored turns, oldest first

llm.text("Read the total.", images=["./receipt.png"], tools=["ocr"])
llm.text("Which notes mention the bridge?", tools=["spotlight"])
llm.text("Compare these quotes.", documents=["./a.md", "./b.md"])

llm.rewrite("gonna grab a bite", instruction="Make it formal.")
llm.proofread("Their going to the store...")
llm.summarize(long_text)
llm.ask_screen("What is on this schedule?")
```

**A tagging-specialised model**, `permissive` guardrails for rewriting, and
`prewarm()` to load model assets up front (worth little once they are resident —
0.31s against 0.36s here — but real on a cold system):

```python
with AppleLLM(tier="device", use_case="contentTagging") as llm:
    llm.json("A recipe for sourdough…", schema=tags_schema)

with AppleLLM(guardrails="permissive") as llm:
    llm.text("Rewrite more formally: gonna grab a bite")
    llm.prewarm()
```

## Errors

`ModelUnavailableError` (`.reason` is `appleIntelligenceNotEnabled` /
`modelNotReady` / `deviceNotEligible` / …), `SchemaRejectedError`,
`ContextLengthError`, `QuotaError` (`.reset_date`), `TimeoutError`,
`SetupRequiredError`, `RefusalError` — all subclasses of `AppleLLMError`.

Note `apple_llm.TimeoutError` deliberately shadows the builtin name; import it
explicitly.

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

`json()` runs your schema through `to_apple_schema()`, which rewrites it into the
restricted dialect Apple's `GenerationSchema` decoder accepts — eight rules
covering unions, `x-order`, enums, titles, `$ref`-by-title, `additionalProperties`,
string-typed `const`, and empty objects. Rule 8 exists because pydantic titles
every property and Apple rejects a titled string. See the
[full notes in the repo](https://github.com/jagdish/apple-llm#the-generationschema-dialect).

## Credit

Extracted from [api-scribe](https://github.com/jagdish/api-scribe) (MIT), where
both routes were discovered and shipped. MIT.
