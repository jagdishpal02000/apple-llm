/**
 * The public client: one API over both tiers.
 *
 * Every generation method takes either a prompt string or a message list, and
 * funnels into one private `execute`, so a feature added there reaches `text`,
 * `json`, `generate` and both streams at once.
 */
import { probeCloud, CloudClient, CLOUD_CONTEXT_TOKENS, type CloudProbe } from './cloud.js';
import type { OnProgress } from './compile.js';
import {
  DeviceClient,
  probeDevice,
  withDocuments,
  type BuiltInTool,
  type DeviceProbe,
  type FinishReason,
  type Guardrails,
  type HistoryTurn,
  type ImageAttachment,
  type SamplingMode,
  type ToolCallRecord,
  type Usage,
  type UseCase,
} from './device.js';
import {
  AppleLLMError,
  ModelUnavailableError,
  SchemaValidationError,
  UnsupportedError,
} from './errors.js';
import { parseLlmJson } from './json-recovery.js';
import { ReplayBook, renderHistory, splitMessages, type ChatMessage, type HistoryEntry } from './messages.js';
import { abortErrorFrom } from './protocol.js';
import { toAppleSchema, type JsonSchema } from './schema.js';
import {
  resolveSchema,
  restoreNulls,
  validateWith,
  type InferSchema,
  type ResolvedSchema,
  type SchemaLike,
} from './standard-schema.js';
import { ResultStream } from './stream.js';
import { normalizeTools, type ToolSet } from './tools.js';

export type Tier = 'device' | 'cloud' | 'auto';

export interface ProbeResult {
  device: DeviceProbe;
  cloud: CloudProbe;
}

/**
 * What this machine can actually do. Never throws: on Linux, an Intel Mac or
 * macOS 25 it returns `available: false` with a reason naming the fix.
 */
export async function probe(onProgress?: OnProgress): Promise<ProbeResult> {
  const [device, cloud] = await Promise.all([probeDevice(onProgress), probeCloud()]);
  // The on-device helper is the only thing that can read the PCC quota and
  // model — the entitlement that blocks PCC *inference* does not block reading
  // them — so the cloud picture is assembled from both probes.
  if (device.cloud !== undefined) {
    const { capabilities, contextSize, ...quota } = device.cloud;
    cloud.quota = quota;
    if (capabilities !== undefined) cloud.capabilities = capabilities;
    if (contextSize !== undefined && cloud.contextSize !== undefined) cloud.contextSize = contextSize;
  }
  return { device, cloud };
}

export interface AppleLLMOptions {
  tier?: Tier;
  /** Default instructions for every call; a per-call `system` replaces it. */
  system?: string;
  /** Default sampling temperature. See DEFAULT_TEMPERATURE — do not set 0. */
  temperature?: number;
  maxTokens?: number;
  /** Default per-call deadline in milliseconds. Aborts the call with a TimeoutError. */
  timeoutMs?: number;
  /** Called for the one-time compile and the shortcut install; both take seconds. */
  onProgress?: OnProgress;
  /** `contentTagging` selects Apple's tagging-specialised model. Device tier only. */
  useCase?: UseCase;
  /** `permissive` relaxes guardrails for rewriting tasks. Device tier only. */
  guardrails?: Guardrails;
  /** Default sampling mode; a seeded one makes output reproducible. */
  sampling?: SamplingMode;
}

/** A prompt, or a whole conversation in OpenAI-style messages. */
export type Input = string | ReadonlyArray<ChatMessage>;

export interface TextOptions {
  system?: string;
  temperature?: number;
  maxTokens?: number;
  /** Cloud tier only: run with "Use Broad World Knowledge". */
  webSearch?: boolean;
  /** Image attachments; entries may carry a label for follow-up turns. */
  images?: ImageAttachment[];
  /** Text files inlined into the prompt. */
  documents?: string[];
  /** Named multi-turn conversation kept by the helper (device tier). */
  sessionId?: string;
  /**
   * Tools the model may call: Apple's built-ins (`'ocr'`, `'barcode'`,
   * `'spotlight'`) and your own function tools (see `tool()`), as an array or
   * a record keyed by name. Device tier.
   */
  tools?: ToolSet;
  /** Tool calls allowed per request before the model is told to answer. Default 8. */
  maxToolCalls?: number;
  useCase?: UseCase;
  guardrails?: Guardrails;
  sampling?: SamplingMode;
  /** Cancels the call. Rejects with an AbortError (a TimeoutError for `AbortSignal.timeout`). */
  signal?: AbortSignal;
  /** Deadline for this call in milliseconds. */
  timeoutMs?: number;
  /**
   * For message-list input: drop the oldest turns when the conversation no
   * longer fits the context window, instead of failing. Default true; the
   * result's `trimmedTurns` says how many went.
   */
  trimHistory?: boolean;
}

export interface StreamOptions extends TextOptions {
  /** Called with each delta; the returned stream can be iterated as well. */
  onDelta?: (delta: string) => void;
}

export interface JsonOptions<S extends SchemaLike = SchemaLike> extends TextOptions {
  /** JSON Schema, or a Standard Schema (Zod 4, ArkType, Valibot via toStandardJsonSchema). */
  schema: S;
}

export interface GenerateResult<T = undefined> {
  /** The reply text; for a schema, the JSON as the model wrote it. */
  text: string;
  /** The parsed and validated reply; `undefined` without a schema. */
  object: T;
  finishReason: FinishReason;
  /** Token accounting — device tier on macOS 27+. */
  usage?: Usage;
  /** Every tool call made along the way, with what each returned. */
  toolCalls: ToolCallRecord[];
  /** Oldest conversation turns dropped to fit the context window. 0 almost always. */
  trimmedTurns: number;
  tier: 'device' | 'cloud';
  durationMs: number;
  /** The assistant turn, ready to append to a message list for the next call. */
  message: Extract<ChatMessage, { role: 'assistant' }>;
}

/** Recursively optional — the shape of an object that is still being generated. */
export type DeepPartial<T> = T extends ReadonlyArray<infer U>
  ? Array<DeepPartial<U>>
  : T extends object
    ? { [K in keyof T]?: DeepPartial<T[K]> }
    : T;

/** `for await` the deltas, or `await` the whole text. `.result` has usage and the rest. */
export type TextStream = ResultStream<string, string, GenerateResult<undefined>>;

/** `for await` partial objects as they fill in, or `await` the final validated object. */
export type ObjectStream<T> = ResultStream<DeepPartial<T>, T, GenerateResult<T>>;

interface ExecuteMode {
  schema?: ResolvedSchema;
  streaming?: boolean;
  onDelta?: (delta: string) => void;
  onPartial?: (json: string) => void;
  signal?: AbortSignal;
}

/** Combine abort signals; `undefined` when there are none. */
function anySignal(signals: Array<AbortSignal | undefined>): AbortSignal | undefined {
  const list = signals.filter((s): s is AbortSignal => s !== undefined);
  if (list.length <= 1) return list[0];
  const any = (AbortSignal as unknown as { any?: (s: AbortSignal[]) => AbortSignal }).any;
  if (typeof any === 'function') return any(list);
  const controller = new AbortController();
  for (const signal of list) {
    if (signal.aborted) {
      controller.abort(signal.reason);
      break;
    }
    signal.addEventListener('abort', () => controller.abort(signal.reason), { once: true });
  }
  return controller.signal;
}

/**
 * The main entry point.
 *
 *   const llm = new AppleLLM();
 *   await llm.text('Summarize this');
 *   await llm.json('Extract the fields', { schema });
 *   for await (const d of llm.stream('Tell me a story')) process.stdout.write(d);
 *
 * One instance keeps one helper process warm. It is unref'd between calls, so
 * a script exits on its own; `close()` (or `await using`) releases it early.
 */
export class AppleLLM {
  private device?: DeviceClient;
  private cloud?: CloudClient;
  /** Which tier `auto` settled on, once resolved. */
  private resolved?: 'device' | 'cloud';
  private readying?: Promise<void>;
  private readonly options: AppleLLMOptions;

  constructor(options: AppleLLMOptions = {}) {
    this.options = options;
  }

  get tier(): Tier {
    return this.options.tier ?? 'auto';
  }

  /** Human-readable name of the tier in use, for logs. */
  get label(): string {
    if (this.resolved === 'cloud') return this.cloud?.label ?? 'apple private cloud compute';
    if (this.resolved === 'device') return this.device?.label ?? 'apple on-device';
    return `apple (${this.tier})`;
  }

  /** Context window of the resolved tier, in tokens. */
  get contextSize(): number | undefined {
    if (this.resolved === 'cloud') return this.cloud?.contextSize ?? CLOUD_CONTEXT_TOKENS;
    return this.device?.contextSize;
  }

  /**
   * Resolve the tier and do any one-time setup. Called automatically, but
   * exposed so a caller can pay the compile cost up front with a progress bar.
   * Concurrent first calls share one attempt.
   */
  async ensureReady(onProgress?: OnProgress): Promise<void> {
    if (this.resolved !== undefined) return;
    this.readying ??= this.resolve(onProgress ?? this.options.onProgress).finally(() => {
      this.readying = undefined;
    });
    return this.readying;
  }

  private newDevice(): DeviceClient {
    return new DeviceClient({ useCase: this.options.useCase, guardrails: this.options.guardrails });
  }

  private async readyCloud(progress?: OnProgress): Promise<void> {
    this.cloud ??= new CloudClient();
    await this.cloud.ensureReady(progress);
    // Best effort: the device helper can read the PCC quota, which turns an
    // exhausted quota into an immediate error instead of a wasted round trip.
    this.cloud.setQuota((await probeDevice().catch(() => undefined))?.cloud);
    this.resolved = 'cloud';
  }

  private async resolve(progress?: OnProgress): Promise<void> {
    if (this.tier === 'device') {
      this.device ??= this.newDevice();
      await this.device.ensureReady(progress);
      this.resolved = 'device';
      return;
    }
    if (this.tier === 'cloud') {
      await this.readyCloud(progress);
      return;
    }
    // auto: on-device when available, else cloud when the shortcut is installed,
    // else an error naming the setup step.
    this.device ??= this.newDevice();
    try {
      await this.device.ensureReady(progress);
      this.resolved = 'device';
    } catch (deviceError) {
      try {
        await this.readyCloud(progress);
      } catch (cloudError) {
        const deviceReason = deviceError instanceof Error ? deviceError.message : String(deviceError);
        const cloudReason = cloudError instanceof Error ? cloudError.message : String(cloudError);
        throw new ModelUnavailableError(
          `No Apple model is available.\n\nOn-device: ${deviceReason}\n\nCloud: ${cloudReason}`,
          deviceError instanceof ModelUnavailableError ? deviceError.reason : 'unknown',
        );
      }
    }
  }

  /** Free text. */
  async text(input: Input, options: TextOptions = {}): Promise<string> {
    return (await this.execute(input, options, {})).text;
  }

  /**
   * Everything about one generation: text, parsed object (with a schema),
   * usage, tool calls, finish reason, and the assistant `message` to append.
   */
  async generate<S extends SchemaLike>(
    input: Input,
    options: TextOptions & { schema: S },
  ): Promise<GenerateResult<InferSchema<S>>>;
  async generate(input: Input, options?: TextOptions & { schema?: undefined }): Promise<GenerateResult<undefined>>;
  async generate(input: Input, options: TextOptions & { schema?: SchemaLike } = {}): Promise<GenerateResult<unknown>> {
    const { schema, ...rest } = options;
    return this.execute(input, rest, { schema: schema === undefined ? undefined : resolveSchema(schema) });
  }

  /**
   * Ask for JSON. On device the *shape* is guaranteed by constrained decoding;
   * a Standard Schema's own validation (refinements, transforms) then runs on
   * top, and the result is typed by it. On cloud the shape is requested in the
   * prompt and recovered from the reply. Either way, a reply that fails
   * validation is retried once with the problems pointed out, then raised as a
   * SchemaValidationError.
   */
  async json<S extends SchemaLike>(input: Input, options: JsonOptions<S>): Promise<InferSchema<S>> {
    const { schema, ...rest } = options;
    const result = await this.execute(input, rest, { schema: resolveSchema(schema) });
    return result.object as InferSchema<S>;
  }

  /**
   * Streaming text. Iterate it for deltas, or await it for the whole reply:
   *
   *   for await (const delta of llm.stream(prompt)) process.stdout.write(delta);
   *   const text = await llm.stream(prompt, { onDelta });
   *
   * On the cloud tier the reply arrives as one chunk: Shortcuts is not
   * incremental.
   */
  stream(input: Input, options: StreamOptions = {}): TextStream {
    const { onDelta, ...rest } = options;
    return new ResultStream(
      (emit, signal) =>
        this.execute(input, rest, {
          streaming: true,
          signal,
          onDelta: (delta) => {
            onDelta?.(delta);
            emit(delta);
          },
        }) as Promise<GenerateResult<undefined>>,
      (result) => result.text,
    );
  }

  /**
   * Streaming JSON: partial objects as the model fills them in — render a form
   * or a card while it is still being written — and the final object, parsed
   * and validated, as the awaited value.
   *
   *   for await (const partial of llm.streamJson(prompt, { schema })) render(partial);
   */
  streamJson<S extends SchemaLike>(input: Input, options: JsonOptions<S>): ObjectStream<InferSchema<S>> {
    const { schema, ...rest } = options;
    const resolved = resolveSchema(schema);
    return new ResultStream(
      (emit, signal) =>
        this.execute(input, rest, {
          schema: resolved,
          streaming: true,
          signal,
          onPartial: (json) => {
            try {
              emit(JSON.parse(json) as DeepPartial<InferSchema<S>>);
            } catch {
              // A snapshot that does not parse is skipped; the next one will.
            }
          },
        }) as Promise<GenerateResult<InferSchema<S>>>,
      (result) => result.object,
    );
  }

  /** The one path every generation takes. */
  private async execute(input: Input, options: TextOptions, mode: ExecuteMode): Promise<GenerateResult<unknown>> {
    const started = Date.now();
    await this.ensureReady();

    let prompt: string;
    let history: HistoryEntry[] = [];
    let replay: ReplayBook | undefined;
    let system = options.system ?? this.options.system;
    let trimHistory: boolean | undefined;
    let images = options.images;
    if (typeof input === 'string') {
      prompt = input;
    } else {
      if (options.sessionId !== undefined) {
        throw new AppleLLMError(
          'Pass either a message list or a sessionId, not both: a named session keeps its own history.',
        );
      }
      const split = splitMessages(input);
      prompt = split.prompt;
      history = split.history;
      replay = split.replay.length > 0 ? new ReplayBook(split.replay) : undefined;
      if (split.system !== undefined) system = system === undefined ? split.system : `${system}\n\n${split.system}`;
      trimHistory = options.trimHistory ?? true;
      if (split.images.length > 0) images = [...(images ?? []), ...split.images];
    }

    const tools = normalizeTools(options.tools);
    const timeoutMs = options.timeoutMs ?? this.options.timeoutMs;
    const signal = anySignal([
      options.signal,
      mode.signal,
      timeoutMs === undefined ? undefined : AbortSignal.timeout(timeoutMs),
    ]);
    if (signal?.aborted === true) throw abortErrorFrom(signal);

    const attempt = async (feedback?: string): Promise<GenerateResult<unknown>> => {
      const promptText = feedback === undefined ? prompt : `${prompt}\n\n${feedback}`;
      let text: string;
      let finishReason: FinishReason = 'stop';
      let usage: Usage | undefined;
      let toolCalls: ToolCallRecord[] = [];
      let trimmedTurns = 0;

      if (this.resolved === 'cloud') {
        this.assertCloudCompatible({ ...options, images }, tools.builtIn.length + tools.functions.length > 0);
        let cloudPrompt = await withDocuments(promptText, options.documents);
        if (history.length > 0) {
          cloudPrompt = `Conversation so far:\n${renderHistory(history)}\n\nNew message from the user:\n${cloudPrompt}`;
        }
        let cloudSystem = system;
        if (mode.schema !== undefined) {
          const instruction =
            'Reply with a single JSON value matching this JSON Schema. ' +
            'Output only the JSON, with no commentary and no code fence.\n\n' +
            JSON.stringify(mode.schema.json, null, 2);
          cloudSystem = cloudSystem ? `${cloudSystem}\n\n${instruction}` : instruction;
        }
        text = await this.cloud!.text({
          system: cloudSystem,
          prompt: cloudPrompt,
          webSearch: options.webSearch,
          signal,
          images: images !== undefined && images.length > 0 ? images : undefined,
        });
        if (mode.schema === undefined) mode.onDelta?.(text);
      } else {
        const outcome = await this.device!.run(
          {
            system,
            prompt: promptText,
            schema: mode.schema === undefined ? null : toAppleSchema(mode.schema.json),
            temperature: options.temperature ?? this.options.temperature,
            maxTokens: options.maxTokens ?? this.options.maxTokens,
            images,
            documents: options.documents,
            sessionId: options.sessionId,
            history: history.length > 0 ? history : undefined,
            trimHistory,
            tools: tools.builtIn.length > 0 ? tools.builtIn : undefined,
            functions: tools.functions.length > 0 ? tools.functions : undefined,
            useCase: options.useCase ?? this.options.useCase,
            guardrails: options.guardrails ?? this.options.guardrails,
            sampling: options.sampling ?? this.options.sampling,
          },
          mode.streaming === true ? 'stream' : 'generate',
          {
            signal,
            onDelta: mode.onDelta,
            onPartial: mode.onPartial,
            replay,
            maxToolCalls: options.maxToolCalls,
          },
        );
        text = outcome.content;
        finishReason = outcome.finishReason;
        usage = outcome.usage;
        toolCalls = outcome.toolCalls;
        trimmedTurns = outcome.trimmedTurns;
      }

      let object: unknown;
      if (mode.schema !== undefined && finishReason !== 'tool-calls') {
        object = await this.parseObject(text, mode.schema);
        if (this.resolved === 'cloud') mode.onPartial?.(JSON.stringify(object));
      }
      return {
        text,
        object,
        finishReason,
        usage,
        toolCalls,
        trimmedTurns,
        tier: this.resolved === 'cloud' ? 'cloud' : 'device',
        durationMs: Date.now() - started,
        message: {
          role: 'assistant',
          content: finishReason === 'tool-calls' ? null : text,
          ...(toolCalls.length > 0
            ? { toolCalls: toolCalls.map(({ id, name, arguments: args }) => ({ id, name, arguments: args })) }
            : {}),
        },
      };
    };

    try {
      return await attempt();
    } catch (error) {
      // One repair attempt for a reply that failed validation. Not while
      // streaming: the reader has already seen the first attempt's output.
      if (!(error instanceof SchemaValidationError) || mode.streaming === true) throw error;
      return attempt(
        `Your previous answer was rejected: ${error.message.replace(/^The model's reply did not satisfy the schema: /, '')}. ` +
          'Answer again, fixing those problems.',
      );
    }
  }

  /** Parse, restore nulls, validate. Throws SchemaValidationError for anything the schema rejects. */
  private async parseObject(text: string, schema: ResolvedSchema): Promise<unknown> {
    const tier = this.resolved === 'cloud' ? 'cloud' : 'device';
    let value: unknown;
    try {
      value = tier === 'cloud' ? parseLlmJson(text) : JSON.parse(text);
    } catch {
      throw new SchemaValidationError('The model did not reply with valid JSON.', [{ message: 'invalid JSON' }], text, tier);
    }
    return validateWith(schema, restoreNulls(value, schema.json), text, tier);
  }

  private assertCloudCompatible(options: TextOptions, hasTools: boolean): void {
    const refuse = (what: string): never => {
      throw new UnsupportedError(`${what} needs the on-device tier.`, 'cloud');
    };
    if (options.images !== undefined && options.images.length > 0 && this.cloud?.supportsImages !== true) {
      throw new UnsupportedError(
        'images on the cloud tier need a server model that reads them — macOS 27 (Golden Gate) or later, ' +
          'where the helper can confirm it. Use the on-device tier for images here.',
        'cloud',
      );
    }
    if (options.sessionId !== undefined) refuse('sessionId');
    if (hasTools) refuse('tools');
  }

  /** Conversation history for a named session (device tier only). */
  async history(sessionId: string): Promise<{ instructions: string; history: HistoryTurn[] }> {
    await this.ensureReady();
    if (this.resolved !== 'device') {
      throw new UnsupportedError('history needs the on-device tier.', 'cloud');
    }
    return this.device!.history(sessionId);
  }

  /** Drop a named session, or all sessions when omitted (device tier only). */
  async resetSession(sessionId?: string): Promise<void> {
    await this.ensureReady();
    if (this.resolved === 'device') await this.device!.resetSession(sessionId);
  }

  /**
   * A named conversation: calls sharing one native transcript, like one thread
   * in the Siri app. History persists across helper restarts, and the
   * transcript is rebuilt from it — trimmed to fit when it outgrows the window.
   */
  conversation(sessionId: string, options: { system?: string } = {}): Conversation {
    return new Conversation(this, sessionId, options);
  }

  /**
   * Write with Siri, anywhere you type: drafting, rewriting and feedback
   * built on the permissive-content-transformation guardrails. Device tier
   * only — these are transformation tasks the default guardrails refuse.
   */
  async rewrite(text: string, options: { instruction?: string } & TextOptions = {}): Promise<string> {
    const { instruction, ...rest } = options;
    return this.text(`Rewrite the following text. ${instruction ?? 'Keep the meaning, improve clarity.'}\n\n---\n${text}`, {
      ...rest,
      guardrails: rest.guardrails ?? 'permissive',
      system: rest.system ?? 'You rewrite text. Reply with only the rewritten text, no commentary.',
    });
  }

  async proofread(text: string, options: TextOptions = {}): Promise<string> {
    return this.text(`Fix spelling, grammar and punctuation in the following text. Preserve the meaning and tone.\n\n---\n${text}`, {
      ...options,
      guardrails: options.guardrails ?? 'permissive',
      system: options.system ?? 'You proofread text. Reply with only the corrected text, no commentary.',
    });
  }

  /**
   * Summarise text of any length. Text that does not fit the context window
   * is summarised in parts, then the parts are summarised together — so a
   * long report works on an 8k-token model instead of failing.
   */
  async summarize(text: string, options: { length?: string } & TextOptions = {}): Promise<string> {
    const { length, ...rest } = options;
    const once = (body: string, what: string, size: string): Promise<string> =>
      this.text(`Summarize ${what} in ${size}.\n\n---\n${body}`, {
        ...rest,
        guardrails: rest.guardrails ?? 'permissive',
        system: rest.system ?? 'You summarize text tersely.',
      });
    let current = text;
    // Each round shrinks the text several-fold; the bound is for pathological input.
    for (let round = 0; round < 4; round += 1) {
      const chunks = await this.chunksFor(current, rest.maxTokens);
      if (chunks.length <= 1) break;
      const parts: string[] = [];
      for (const chunk of chunks) {
        parts.push(await once(chunk, 'this part of a longer document', 'a few sentences, keeping names, numbers and decisions'));
      }
      current = parts.join('\n\n');
    }
    return once(current, current === text ? 'the following text' : 'these notes on a longer document', length ?? 'one short paragraph');
  }

  /** Split text into pieces that each fit one call, on paragraph boundaries where possible. */
  private async chunksFor(text: string, maxTokens?: number): Promise<string[]> {
    await this.ensureReady();
    const window = this.contextSize ?? 4096;
    const budget = Math.max(512, window - (maxTokens ?? this.options.maxTokens ?? 1024) - 256);
    let tokens: number;
    if (this.resolved === 'device') {
      try {
        tokens = (await this.device!.countTokens(text)).tokens;
      } catch {
        tokens = Math.ceil(text.length / 3);
      }
    } else {
      tokens = Math.ceil(text.length / 3);
    }
    if (tokens <= budget) return [text];
    // Measured characters-per-token for this text, with headroom.
    const maxChars = Math.floor((text.length / tokens) * budget * 0.85);
    const chunks: string[] = [];
    let current = '';
    for (const paragraph of text.split(/\n\s*\n/)) {
      const pieces = paragraph.length > maxChars ? paragraph.match(new RegExp(`[\\s\\S]{1,${maxChars}}`, 'g')) ?? [] : [paragraph];
      for (const piece of pieces) {
        if (current.length + piece.length + 2 > maxChars && current !== '') {
          chunks.push(current);
          current = '';
        }
        current = current === '' ? piece : `${current}\n\n${piece}`;
      }
    }
    if (current !== '') chunks.push(current);
    return chunks;
  }

  async draft(topic: string, options: { kind?: string } & TextOptions = {}): Promise<string> {
    const { kind, ...rest } = options;
    return this.text(`Write a ${kind ?? 'short draft'} about the following topic.\n\n---\n${topic}`, {
      ...rest,
      system: rest.system ?? 'You are a helpful writing assistant.',
    });
  }

  async tone(text: string, tone: string, options: TextOptions = {}): Promise<string> {
    return this.text(`Rewrite the following text to sound more ${tone}.\n\n---\n${text}`, {
      ...options,
      guardrails: options.guardrails ?? 'permissive',
      system: options.system ?? 'You rewrite text. Reply with only the rewritten text, no commentary.',
    });
  }

  /**
   * Ask about what's on screen: captures a screenshot (interactive selection
   * by default, like Cmd+Shift+Space Visual Intelligence) and asks the model
   * about it with vision. Device tier, macOS 27+.
   */
  async askScreen(
    question: string,
    options: { mode?: 'interactive' | 'window' | 'fullscreen' } & TextOptions = {},
  ): Promise<string> {
    await this.ensureReady();
    if (this.resolved !== 'device') {
      throw new UnsupportedError('askScreen needs the on-device tier.', 'cloud');
    }
    const { mode, ...rest } = options;
    const shot = await captureScreenshot(mode ?? 'interactive');
    try {
      return await this.text(question, { ...rest, images: [...(rest.images ?? []), shot] });
    } finally {
      const { rm } = await import('node:fs/promises');
      await rm(shot, { force: true }).catch(() => undefined);
    }
  }

  /**
   * `json()` in the `(system, user, schema)` shape many LLM clients use, for
   * dropping this in behind an existing interface.
   */
  async completeJson(system: string, user: string, schema: JsonSchema): Promise<unknown> {
    return this.json(user, { system, schema });
  }

  /**
   * How many tokens a request costs, before sending it. Device tier only.
   *
   * Turns a ContextLengthError into arithmetic: compare against `contextSize`
   * and trim, rather than finding the ceiling by hitting it. Counts
   * everything that shares the window — instructions, message history, tools,
   * schema, images and documents.
   */
  async countTokens(
    input: Input,
    options: Pick<TextOptions, 'system' | 'images' | 'tools' | 'documents' | 'signal'> & { schema?: SchemaLike } = {},
  ): Promise<{ tokens: number; contextSize: number }> {
    await this.ensureReady();
    if (this.resolved !== 'device') {
      throw new UnsupportedError('countTokens needs the on-device tier.', 'cloud');
    }
    let prompt: string;
    let history: HistoryEntry[] | undefined;
    let system = options.system ?? this.options.system;
    if (typeof input === 'string') {
      prompt = input;
    } else {
      const split = splitMessages(input);
      prompt = split.prompt;
      history = split.history;
      if (split.system !== undefined) system = system === undefined ? split.system : `${system}\n\n${split.system}`;
    }
    const tools = normalizeTools(options.tools);
    return this.device!.countTokens(prompt, {
      system,
      images: options.images,
      documents: options.documents,
      history,
      tools: tools.builtIn.length > 0 ? tools.builtIn : undefined,
      functions: tools.functions.length > 0 ? tools.functions : undefined,
      schema: options.schema === undefined ? undefined : resolveSchema(options.schema).json,
      signal: options.signal,
    });
  }

  /**
   * Load the model assets now so the first real call does not pay for it.
   * Device tier only; a no-op elsewhere. See `DeviceClient.prewarm` for what it
   * is actually worth (little, on a warm machine).
   */
  async prewarm(system?: string): Promise<void> {
    await this.ensureReady();
    if (this.resolved === 'device') await this.device!.prewarm(system ?? this.options.system);
  }

  /** Release the long-lived helper process. Safe to call more than once. */
  close(): void {
    this.device?.close();
    this.cloud?.close();
  }

  /** `using llm = new AppleLLM()` closes it at the end of the block. */
  [Symbol.dispose](): void {
    this.close();
  }

  /** `await using llm = new AppleLLM()` closes it at the end of the block. */
  async [Symbol.asyncDispose](): Promise<void> {
    this.close();
  }
}

/**
 * One thread in the Siri-app sense: every call carries the same sessionId,
 * so the helper's native transcript accumulates across turns.
 */
export class Conversation {
  constructor(
    private readonly llm: AppleLLM,
    readonly sessionId: string,
    private readonly defaults: { system?: string } = {},
  ) {}

  private with<T extends TextOptions>(options: T): T {
    return { ...options, system: options.system ?? this.defaults.system, sessionId: this.sessionId };
  }

  async text(prompt: string, options: TextOptions = {}): Promise<string> {
    return this.llm.text(prompt, this.with(options));
  }

  stream(prompt: string, options: StreamOptions = {}): TextStream {
    return this.llm.stream(prompt, this.with(options));
  }

  async generate(prompt: string, options: TextOptions = {}): Promise<GenerateResult<undefined>> {
    return this.llm.generate(prompt, this.with(options) as TextOptions & { schema?: undefined });
  }

  async json<S extends SchemaLike>(prompt: string, options: JsonOptions<S>): Promise<InferSchema<S>> {
    return this.llm.json(prompt, this.with(options));
  }

  async history(): Promise<{ instructions: string; history: HistoryTurn[] }> {
    return this.llm.history(this.sessionId);
  }

  async reset(): Promise<void> {
    await this.llm.resetSession(this.sessionId);
  }
}

/**
 * Capture a screenshot to a temp file. Interactive selection mirrors the
 * Visual Intelligence entry point (Cmd+Shift+Space): drag to select, and the
 * path comes back ready to pass as an image attachment.
 */
export async function captureScreenshot(
  mode: 'interactive' | 'window' | 'fullscreen' = 'interactive',
): Promise<string> {
  const { execFile } = await import('node:child_process');
  const { promisify } = await import('node:util');
  const { mkdtemp, stat } = await import('node:fs/promises');
  const os = await import('node:os');
  const path = await import('node:path');
  const execFileAsync = promisify(execFile);
  const dir = await mkdtemp(path.join(os.tmpdir(), 'apple-llm-screen-'));
  const out = path.join(dir, 'screen.png');
  const args = mode === 'interactive' ? ['-i', '-x', out] : mode === 'window' ? ['-w', '-x', out] : ['-x', out];
  try {
    await execFileAsync('screencapture', args, { timeout: 120_000 });
  } catch (err) {
    throw new AppleLLMError(
      `Could not capture a screenshot (screencapture failed): ${err instanceof Error ? err.message : String(err)}`,
      'device',
    );
  }
  try {
    const st = await stat(out);
    if (st.size === 0) throw new Error('empty capture');
  } catch {
    throw new AppleLLMError('Screenshot capture was cancelled or produced no image.', 'device');
  }
  return out;
}

export type { BuiltInTool };
